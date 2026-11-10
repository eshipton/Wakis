"""Reduced DEPEXTRAP check: m -> 0 DC depression of the sheet-lattice beam per wall stencil.

One ``run_particle_beam_drift`` point per invocation (``--grid``, ``--m-scale`` in units of
``MODULATION_INDEX``, ``--wall-stencil``), identical to the cached lattice runs of the slow
``test_087`` two-resolution test except for ``wall_stencil``. Prints one ``DCIMG {json}`` line with
``delta = (u0 - <u_z>) / u0`` and the model values ``delta_disc_charge`` / ``delta_disc_flux``.
``--collect`` combines ``DCIMG`` lines from log files: ``delta0 = (4 delta(m/2) - delta(m)) / 3``.

Run on a GPU (Batch)::

    python studies/p5b-dcdep/dc_image.py --grid fine --m-scale 0.5 --wall-stencil image
    python studies/p5b-dcdep/dc_image.py --collect run1.log run2.log ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _space_charge_waves_common as s  # noqa: E402


def run_point(grid: str, m_scale: float, wall_stencil: str) -> dict:
    nx, _ny, nz = s.COARSE_GRID if grid == "coarse" else s.FINE_GRID
    params = s.compute_analytic_parameters(qm_scale=1.0)
    m_mod = float(m_scale) * float(s.MODULATION_INDEX)
    beam_ref = s._build_reference_linearized_beam(nx, nz, m_mod=s.MODULATION_INDEX)
    t0 = time.time()
    res = s.run_particle_beam_drift(
        nx,
        nz,
        beam_ref=beam_ref,
        qm_scale=1.0,
        seed=42,
        m_mod=m_mod,
        profile="sheet_lattice",
        n_macro_per_step=16 if nx == s.COARSE_GRID[0] else 32,
        gather="galerkin",
        shape=s.PARTICLE_SHAPE,
        wall_stencil=wall_stencil,
    )
    det = s.compute_discrete_dc_depression(
        nx, params, weighting="charge", return_details=True, shape=s.PARTICLE_SHAPE
    )
    rec = {
        "grid": grid,
        "nx": nx,
        "nz": nz,
        "m_scale": float(m_scale),
        "m_mod": m_mod,
        "wall_stencil": wall_stencil,
        "u0": float(params["u0"]),
        "u_z_mean": float(res["u_z_mean"]),
        "delta": float((params["u0"] - res["u_z_mean"]) / params["u0"]),
        "gamma_diff_mean": float(res["gamma_diff_mean"]),
        "delta_disc_charge": float(det["delta_charge"]),
        "delta_disc_flux": float(det["delta_flux"]),
        "dt": float(res["dt"]),
        "wall_s": time.time() - t0,
        "use_gpu": bool(s.USE_GPU),
    }
    print("DCIMG " + json.dumps(rec), flush=True)
    return rec


def collect(paths: list[str]) -> list[dict]:
    recs = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if line.startswith("DCIMG "):
                    recs.append(json.loads(line[6:]))
    out = []
    keys = sorted({(r["grid"], r["wall_stencil"]) for r in recs})
    for grid, ws in keys:
        by_m = {r["m_scale"]: r for r in recs if r["grid"] == grid and r["wall_stencil"] == ws}
        if 1.0 not in by_m or 0.5 not in by_m:
            continue
        d_m, d_m2 = by_m[1.0]["delta"], by_m[0.5]["delta"]
        d0 = (4.0 * d_m2 - d_m) / 3.0
        rec = {
            "grid": grid,
            "wall_stencil": ws,
            "delta_m": d_m,
            "delta_m2": d_m2,
            "delta0": d0,
            "delta_disc_charge": by_m[1.0]["delta_disc_charge"],
            "delta_disc_flux": by_m[1.0]["delta_disc_flux"],
            "gap_charge": d0 - by_m[1.0]["delta_disc_charge"],
            "gap_flux": d0 - by_m[1.0]["delta_disc_flux"],
            "gamma_diff_m": by_m[1.0]["gamma_diff_mean"],
            "gamma_diff_m2": by_m[0.5]["gamma_diff_mean"],
        }
        print("DCIMG_DELTA0 " + json.dumps(rec), flush=True)
        out.append(rec)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", choices=("coarse", "fine"), default="coarse")
    ap.add_argument("--m-scale", type=float, default=1.0)
    ap.add_argument("--wall-stencil", choices=("masked", "image"), default="image")
    ap.add_argument("--collect", nargs="*", default=None)
    args = ap.parse_args(argv)
    if args.collect is not None:
        collect(args.collect)
        return 0
    run_point(args.grid, args.m_scale, args.wall_stencil)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
