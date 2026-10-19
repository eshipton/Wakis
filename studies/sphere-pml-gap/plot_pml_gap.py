"""
Tutorial 01b, numerical set-up check: does the distance between the sphere
and the CPML change Q_sca and the far field?

Runs (all 40 cells/um around the sphere, conformal 'all', 300 fs, same
TF/SF and near-to-far boxes; made with ``pmlgap.sh`` on cem-gpu):

* ``uniform_gap1x``          -- the tutorial's uniform grid, CPML 1.05 um
  from the sphere surface;
* ``uniform_gap1x_cfs``      -- the same with a complex-frequency-shifted
  CPML (alpha_max = 1000 S/m, alpha/eps0 ~ 0.2 omega_min);
* ``graded_gap{1.05,2.1,4.2}um`` -- graded grid (uniform out to the flux
  box, cells growing by <= 1.15 to 16 cells/um in the vacuum gap) with the
  CPML 1.05, 2.1 (~1 lambda at x = 3) and 4.2 um (~2 lambda at x = 3) from
  the surface; ``graded_gap1.05um_cfs`` with the CFS CPML;
* ``uniform_gap1x_res50``    -- the tutorial grid at 50 instead of 40
  cells/um (does the remaining deviation shrink with resolution?).

Writes ``figures/pml_gap_far_field.png``, ``figures/pml_gap_summary.png``
and ``data/summary.md``.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c0

from wakis.analytic import mie_differential_cross_section

HERE = Path(__file__).resolve().parent
R, N = 1e-6, 2.0
ARMS = [
    ("uniform_gap1x", "uniform, gap 1.05 µm (tutorial)", "#D55E00", "-"),
    ("uniform_gap1x_cfs", "uniform, gap 1.05 µm, CFS CPML", "#E69F00", "--"),
    ("graded_gap1.05um", "graded, gap 1.05 µm", "#CC79A7", "-."),
    ("graded_gap1.05um_cfs", "graded, gap 1.05 µm, CFS CPML", "#999999", ":"),
    ("graded_gap2.1um", "graded, gap 2.1 µm (≈1 λ at x = 3)", "#009E73", "--"),
    ("graded_gap4.2um", "graded, gap 4.2 µm (≈2 λ at x = 3)", "#0072B2", "-"),
    ("uniform_gap1x_res50", "uniform, gap 1.05 µm, 50 cells/µm", "#000000", (0, (1, 1))),
]


def load():
    runs = {}
    for name, *_ in ARMS:
        p = HERE / "data" / f"{name}.json"
        if p.exists():
            runs[name] = json.loads(p.read_text())
    return runs


def main():
    runs = load()
    (HERE / "figures").mkdir(exist_ok=True)
    theta = np.linspace(0, np.pi, 181)
    xs = (3, 6, 9)
    # Overlay: dsigma/dOmega [dB re r^2] vs theta, E and H planes
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for col, x in enumerate(xs):
        lam = 2 * np.pi * R / x
        for row, (plane, phi) in enumerate((("E", 0.0), ("H", np.pi / 2))):
            ax = axes[row, col]
            mie = 10 * np.log10(
                mie_differential_cross_section(N, R, lam, theta, phi) / R**2
            )
            ax.plot(np.degrees(theta), mie, "k-", lw=2.2, alpha=0.35,
                    label="exact series (Mie)")  # fmt: skip
            for name, label, color, ls in ARMS:
                if name in runs:
                    d = runs[name]["far_field"][f"x{x}_{plane}"]
                    ax.plot(np.degrees(theta), d["db_sim"], ls=ls, color=color, lw=1.1,
                            label=f"Wakis (FIT), {label}: {d['rms_db']:.2f} dB RMS")  # fmt: skip
            ax.set_ylim(mie.max() - 40, mie.max() + 3)
            ax.set_title(f"x = {x}, {plane}-plane ({'xz, |S2|²' if plane == 'E' else 'yz, |S1|²'})")
            if row == 1:
                ax.set_xlabel("scattering angle θ [deg]")
            if col == 0:
                ax.set_ylabel("dσ/dΩ / r²  [dB]")
            ax.legend(fontsize=6.3, loc="lower left")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "pml_gap_far_field.png", dpi=110)

    # Summary vs gap
    rows = []
    for name, label, *_ in ARMS:
        if name not in runs:
            continue
        r = runs[name]
        ff = r["far_field"]
        rows.append((label, r["cells"], r["run_s"], r["rel_rms_error"],
                     *[ff[f"x{x}_{p}"]["rms_db"] for x in xs for p in "EH"],
                     *[ff[f"x{x}_{p}"]["rel_l2"] for x in xs for p in "EH"]))  # fmt: skip
    lines = [
        "| run | cells | run [s] | Q_sca RMS err (x = 2–10) | "
        + " | ".join(f"x={x} {p} [dB]" for x in xs for p in "EH")
        + " | "
        + " | ".join(f"x={x} {p} rel. L2" for x in xs for p in "EH")
        + " |",
        "|" + "---|" * (4 + 12),
    ]
    for r in rows:
        lines.append(
            f"| {r[0]} | {r[1] / 1e6:.1f} M | {r[2]:.0f} | {r[3]:.2%} | "
            + " | ".join(f"{v:.2f}" for v in r[4:10])
            + " | "
            + " | ".join(f"{v:.3f}" for v in r[10:16])
            + " |"
        )
    (HERE / "data" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = [r[0] for r in rows]
    yy = np.arange(len(rows))
    a1.barh(yy, [100 * r[3] for r in rows], color="#0072B2")
    a1.set_yticks(yy, labels, fontsize=8)
    a1.set_xlabel("Q_sca relative RMS error vs Mie, x = 2–10 [%]")
    w = 0.13
    for k, (x, p) in enumerate([(x, p) for x in xs for p in "EH"]):
        a2.barh(yy + (k - 2.5) * w, [r[4 + k] for r in rows], height=w,
                label=f"x = {x}, {p}-plane")  # fmt: skip
    a2.set_yticks(yy, [""] * len(rows))
    a2.set_xlabel("far-field RMS error vs Mie [dB] (40 dB floor)")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "pml_gap_summary.png", dpi=110)


if __name__ == "__main__":
    main()
