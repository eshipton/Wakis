"""One point of the P5 part B m-scan diagnostic (coarse grid, deterministic sheet_lattice beam).

Question: is the particle beta_q(m) dependence an amplitude law (beta0 + c m^2 + ...) that a
Richardson extrapolation to m -> 0 can remove, or an m-independent background in I1 whose
relative bias grows as 1/m?  The m = 0 run measures that background directly; the post-processing
(analyze.py) fits each point with and without subtracting it.

Also records the measured DC depression delta_meas against the discrete prediction, to test
whether the M3 DEPCHECK miss (2.7e-5) is the second-order <rho1 v1> correlation (scales as m^2).

Usage: python studies/p5b-mscan/run_point.py --m 0.015 --qm 1.0 --n-macro 16 --out studies/p5b-mscan/out
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
import _space_charge_waves_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, required=True)
    ap.add_argument("--qm", type=float, default=1.0)
    ap.add_argument("--n-macro", type=int, default=16)
    ap.add_argument("--settle-factor", type=float, default=None,
                    help="override n_settle = factor * Lz / (u_dep dt) (default: library 1.01)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"m{a.m:.6f}_qm{a.qm:g}_nm{a.n_macro}" + (f"_sf{a.settle_factor:g}" if a.settle_factor else "")

    nx, _, nz = C.COARSE_GRID
    params = C.compute_analytic_parameters(qm_scale=a.qm)
    t0 = time.time()
    lb = C.cached_linearized_beam_drift(nx, nz, rho0_scale=1.0, m_mod=C.MODULATION_INDEX)
    if a.settle_factor:
        C.SETTLE_FACTOR = a.settle_factor
    part = C.run_particle_beam_drift(
        nx, nz, beam_ref=lb["beam"], qm_scale=a.qm, m_mod=a.m,
        profile="sheet_lattice", n_macro_per_step=a.n_macro,
    )
    t_run = time.time() - t0
    z = lb["z_coords"]
    z_min, z_max = C.fit_window(params)
    rec = dict(m=a.m, qm=a.qm, n_macro=a.n_macro, settle_factor=a.settle_factor, t_run_s=t_run,
               u0=params["u0"], u_z_mean=float(part["u_z_mean"]),
               delta_meas=float((params["u0"] - part["u_z_mean"]) / params["u0"]),
               rho_dc_mean=float(part["rho_dc_mean"]), rho0=float(params["rho0"]),
               z_min=z_min, z_max=z_max, beta_q_th=params["beta_q"])
    dep = C.compute_discrete_dc_depression(nx, params, return_details=True)
    rec["delta_disc"] = float(dep["delta_disc"])
    if a.m > 0:
        fit = C.fit_beta_q_from_envelope(z, part["i1_part"], params["beta_q"], z_min=z_min, z_max=z_max)
        rec.update(beta_q=float(fit["beta_q"]), sigma_beta_q=float(fit["sigma_beta_q"]),
                   k_c=float(fit["k_c"]), rss=float(fit["rss"]))
        tols = C.derived_tolerances(nx, nz, params)
        rec.update(bq_pred_part=float(tols["bq_pred_part"]), slack_part=float(tols["slack_part"]))
    np.savez(os.path.join(a.out, tag + ".npz"), z=z, i1_part=part["i1_part"], i2_part=part["i2_part"],
             u_z_profile=part["u_z_profile"], rho_dc_profile=part["rho_dc_profile"],
             iz_hist_dec=part["iz_hist_dec"].astype(np.float32), n_settle=part["n_settle"])
    with open(os.path.join(a.out, tag + ".json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
