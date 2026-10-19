"""
Independent NEC-2 (`nec2c`) reference for benchmark B4 (centre-fed PEC dipole).

Geometry (`wakis.benchmarks.dipole.DipoleCase`): straight PEC wire along z,
total length L = 0.150 m, radius a = 0.5e-3 m, centre feed (applied-E-field
voltage source on the centre segment), free space, 51 frequencies from
0.70 to 1.20 GHz in 10 MHz steps.

`nec2c` is GPL and is invoked strictly as an external executable on a
temporary NEC-2 card deck.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from wakis.benchmarks import get_case

HERE = Path(__file__).resolve().parent
LENGTH_M = 0.150
RADIUS_M = 0.5e-3
F_START_MHZ = 700.0
F_STEP_MHZ = 10.0
N_FREQ = 51
FREQS_HZ = (F_START_MHZ + F_STEP_MHZ * np.arange(N_FREQ)) * 1e6
KERNELS = ("thin", "extended")
N_SEGS = (51, 101, 151, 201, 301)

_NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def deck(n_seg, kernel="thin"):
    """
    Build a NEC-2 card deck for the B4 centre-fed PEC dipole.

    Parameters
    ----------
    n_seg : int
        Odd number of wire segments so the feed segment is centred at z = 0.
    kernel : {'thin', 'extended'}
        NEC-2 wire kernel: standard thin-wire kernel or extended thin-wire
        kernel (`EK` card).

    Returns
    -------
    str
        NEC-2 input deck text.
    """
    n_seg = int(n_seg)
    if n_seg < 1 or n_seg % 2 == 0:
        raise ValueError(f"n_seg must be a positive odd integer, got {n_seg}")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")

    mid = (n_seg + 1) // 2
    half_len = 0.5 * LENGTH_M
    cards = [
        "CM B4 centre-fed PEC dipole: L = 0.150 m, a = 0.5e-3 m, free space",
        "CE",
        f"GW 1 {n_seg} 0.0 0.0 {-half_len:.6f} 0.0 0.0 {half_len:.6f} {RADIUS_M:.6f}",
        "GE 0",
    ]
    if kernel == "extended":
        cards.append("EK 0")
    cards.extend(
        [
            f"EX 0 1 {mid} 0 1.0 0.0",
            f"FR 0 {N_FREQ} 0 0 {F_START_MHZ:.1f} {F_STEP_MHZ:.1f}",
            "XQ 0",
            "EN",
        ]
    )
    return "\n".join(cards) + "\n"


def parse_output(out_text, n_expected=N_FREQ):
    """
    Parse complex input impedance Z(f) = R + j X [Ohm] from `nec2c` output.

    Parameters
    ----------
    out_text : str
        Contents of the `nec2c` `.out` file.
    n_expected : int or None
        Expected number of frequency points (default 51). If not None,
        raises RuntimeError when the parsed count differs.

    Returns
    -------
    z_in : complex ndarray
        Input impedance R + j X [Ohm] per frequency step.
    """
    lines = out_text.splitlines()
    impedances = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        if "ANTENNA INPUT PARAMETERS" in lines[i]:
            found = False
            for j in range(i + 1, min(i + 12, n_lines)):
                if "ANTENNA INPUT PARAMETERS" in lines[j]:
                    break
                tokens = _NUM_RE.findall(lines[j])
                # Data line has 11 numeric fields:
                # TAG, SEG, V_re, V_im, I_re, I_im, Z_re, Z_im, Y_re, Y_im, P
                if len(tokens) >= 10:
                    r_ohm = float(tokens[6])
                    x_ohm = float(tokens[7])
                    impedances.append(r_ohm + 1j * x_ohm)
                    i = j
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"Failed to parse data row after ANTENNA INPUT PARAMETERS at line {i + 1}"
                )
        i += 1

    if n_expected is not None and len(impedances) != n_expected:
        raise RuntimeError(
            f"Expected {n_expected} impedance blocks, parsed {len(impedances)}"
        )
    return np.asarray(impedances, dtype=complex)


def run(deck_text):
    """
    Execute `nec2c -i in.nec -o out.out` in a temporary directory and parse Z(f).

    Parameters
    ----------
    deck_text : str
        NEC-2 card deck string.

    Returns
    -------
    z_in : complex ndarray
        Input impedance R + j X [Ohm] at each frequency point.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_nec = tmp / "in.nec"
        out_file = tmp / "out.out"
        in_nec.write_text(deck_text)
        subprocess.run(
            ["nec2c", "-i", str(in_nec), "-o", str(out_file)],
            check=True,
            capture_output=True,
            text=True,
        )
        return parse_output(out_file.read_text())


def resonance(freqs_hz, z_in):
    """
    Find the series resonance frequency f_res (where X = Im(Z) = 0) and R(f_res)
    by linear interpolation.

    Parameters
    ----------
    freqs_hz : ndarray
        Frequency grid [Hz].
    z_in : complex ndarray
        Complex input impedance R + j X [Ohm].

    Returns
    -------
    f_res_hz : float
        Interpolated resonance frequency [Hz].
    r_res_ohm : float
        Interpolated input resistance at f_res [Ohm].
    """
    f = np.asarray(freqs_hz, dtype=float)
    z = np.asarray(z_in, dtype=complex)
    x = z.imag
    crossings = np.flatnonzero(np.sign(x[:-1]) != np.sign(x[1:]))
    if not crossings.size:
        raise RuntimeError("No X = 0 crossing found in frequency sweep")
    idx = int(crossings[0])
    t = x[idx] / (x[idx] - x[idx + 1])
    f_res_hz = float(f[idx] + t * (f[idx + 1] - f[idx]))
    r_res_ohm = float(z.real[idx] + t * (z.real[idx + 1] - z.real[idx]))
    return f_res_hz, r_res_ohm


def main():
    """Run NEC-2 B4 sweep across kernels and segment counts, save JSON, and print table."""
    case = get_case("B4")
    ref = case.reference
    f_ref_hz = float(ref["f_res"])
    r_ref_ohm = float(ref["r_res"])

    table = []
    runs = []
    for kernel in KERNELS:
        for n_seg in N_SEGS:
            seg_len_over_a = (LENGTH_M / n_seg) / RADIUS_M
            z = run(deck(n_seg, kernel))
            f_res_hz, r_res_ohm = resonance(FREQS_HZ, z)
            row = {
                "kernel": kernel,
                "n_seg": int(n_seg),
                "delta_over_a": float(seg_len_over_a),
                "f_res_hz": f_res_hz,
                "f_res_ghz": f_res_hz * 1e-9,
                "r_res_ohm": r_res_ohm,
                "df_rel_vs_hallen": (f_res_hz - f_ref_hz) / f_ref_hz,
                "dr_rel_vs_hallen": (r_res_ohm - r_ref_ohm) / r_ref_ohm,
            }
            table.append(row)
            runs.append(
                {
                    **row,
                    "z_re_ohm": z.real.tolist(),
                    "z_im_ohm": z.imag.tolist(),
                }
            )

    out_data = {
        "case": "B4",
        "description": case.description,
        "length_m": LENGTH_M,
        "radius_m": RADIUS_M,
        "freqs_hz": FREQS_HZ.tolist(),
        "freqs_ghz": (FREQS_HZ * 1e-9).tolist(),
        "hallen_reference": {
            "n_ref": int(case.n_ref),
            "gap_m": float(case.gap),
            "f_res_hz": f_ref_hz,
            "f_res_ghz": f_ref_hz * 1e-9,
            "r_res_ohm": r_ref_ohm,
            "reference_uncertainty": {
                k: float(v) for k, v in ref["reference_uncertainty"].items()
            },
        },
        "table": table,
        "runs": runs,
    }

    out_path = HERE / "nec_b4.json"
    out_path.write_text(json.dumps(out_data, indent=2) + "\n")

    print(
        f"{'kernel':<10} {'n_seg':>5} {'Delta/a':>8} "
        f"{'f_res [GHz]':>13} {'R_res [Ohm]':>13} "
        f"{'df vs Hallen':>13} {'dR vs Hallen':>13}"
    )
    print("-" * 81)
    for row in table:
        print(
            f"{row['kernel']:<10} {row['n_seg']:>5d} {row['delta_over_a']:>8.2f} "
            f"{row['f_res_ghz']:>13.6f} {row['r_res_ohm']:>13.4f} "
            f"{row['df_rel_vs_hallen']:>+12.2%} {row['dr_rel_vs_hallen']:>+12.2%}"
        )
    print("-" * 81)
    print(
        f"Hallen MoM reference (wakis.benchmarks B4, N={case.n_ref}, gap={case.gap * 1e3:.1f} mm): "
        f"f_res = {f_ref_hz * 1e-9:.6f} GHz ({f_ref_hz:.1f} Hz), "
        f"r_res = {r_ref_ohm:.4f} Ohm"
    )


if __name__ == "__main__":
    main()
