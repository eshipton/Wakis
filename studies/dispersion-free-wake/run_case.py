"""
One wakefield run (Yee or TE/TM) for the dispersion-free-wake study.

Cases
-----
box      closed rectangular PEC cavity a x a x d (the whole domain), beam on
         axis through the end walls ('direct' source); analytic reference in
         `analytic_box.py`.
pipe     smooth pipe (square: PEC domain walls; round: conformal cylinder),
         CPML at both z ends with the TFSF self-field injection; exact wake 0.
pillbox  conformal pillbox cavity (radius R, gap g) between round beam
         pipes (radius rp), CPML + TFSF.

Writes <out>.json with s, WP, loss factor, dt, Nt, cell count and timings
(solver init, time loop, per-step time). Uses the Wakis wake post-processing
(WakeSolver.calc_long_WP_3d with n_transverse_cells = 0).

Usage: python run_case.py --case box --scheme tete --dz 4e-4 --sigma 1.6e-3
       --out results/box_tete_... [--gpu] [--cfln 0.99]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.constants import c as c_light

import wakis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build(args):
    dz = args.dz
    dx = args.rxy * dz
    npml = args.npml
    common = dict(verbose=0)
    if args.case == "box":
        a, d = args.a, args.length
        nx = int(round(a / dx))
        nz = int(round(d / dz))
        grid = wakis.GridFIT3D(-a / 2, a / 2, -a / 2, a / 2, 0, d, nx, nx, nz, **common)
        bcs = (["pec"] * 3, ["pec"] * 3)
        solver_kw = dict(bg=[1.0, 1.0, 0.0])
        skip = 0
    elif args.case == "pipe":
        a = args.a
        nx = int(round(a / dx))
        nz = int(round(args.length / dz)) + 2 * npml
        zmax = nz * dz
        if args.round:
            import pyvista as pv

            cyl = pv.Cylinder(center=(0, 0, zmax / 2), direction=(0, 0, 1),
                              radius=args.rp, height=2 * zmax,
                              resolution=720).triangulate()  # fmt: skip
            grid = wakis.GridFIT3D(
                -a / 2, a / 2, -a / 2, a / 2, 0, zmax, nx, nx, nz,
                stl_solids={"pipe": cyl}, stl_materials={"pipe": "vacuum"},
                conformal=True, **common,
            )  # fmt: skip
            solver_kw = dict(bg="pec", use_stl=True)
        else:
            grid = wakis.GridFIT3D(-a / 2, a / 2, -a / 2, a / 2, 0, zmax, nx, nx, nz,
                                   **common)  # fmt: skip
            solver_kw = dict(bg=[1.0, 1.0, 0.0])
        bcs = (["pec", "pec", "cpml"], ["pec", "pec", "cpml"])
        skip = npml + 2
    elif args.case == "pillbox":
        import pyvista as pv

        R, g, rp, lp = args.R, args.gap, args.rp, args.pipe_length
        a = 2 * R * 1.06
        nx = int(round(a / dx))
        nz = int(round((g + 2 * lp) / dz)) + 2 * npml
        zmax = nz * dz
        zc = zmax / 2
        pipe = pv.Cylinder(center=(0, 0, zc), direction=(0, 0, 1), radius=rp,
                           height=2 * zmax, resolution=720).triangulate()  # fmt: skip
        cav = pv.Cylinder(center=(0, 0, zc), direction=(0, 0, 1), radius=R,
                          height=g, resolution=720).triangulate()  # fmt: skip
        grid = wakis.GridFIT3D(
            -a / 2, a / 2, -a / 2, a / 2, 0, zmax, nx, nx, nz,
            stl_solids={"pipe": pipe, "cavity": cav},
            stl_materials={"pipe": "vacuum", "cavity": "vacuum"},
            conformal=True, **common,
        )  # fmt: skip
        solver_kw = dict(bg="pec", use_stl=True)
        bcs = (["pec", "pec", "cpml"], ["pec", "pec", "cpml"])
        skip = npml + 2
    else:
        raise ValueError(args.case)
    return grid, bcs, solver_kw, skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, choices=["box", "pipe", "pillbox"])
    p.add_argument("--scheme", required=True, choices=["yee", "tete"])
    p.add_argument("--dz", type=float, required=True)
    p.add_argument("--rxy", type=float, default=2.0, help="dx = dy = rxy * dz")
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--cfln", type=float, default=0.99)
    p.add_argument("--a", type=float, default=0.032)
    p.add_argument("--length", type=float, default=0.096)
    p.add_argument("--round", action="store_true")
    p.add_argument("--rp", type=float, default=0.005)
    p.add_argument("--R", type=float, default=0.016)
    p.add_argument("--gap", type=float, default=0.016)
    p.add_argument("--pipe_length", type=float, default=0.02)
    p.add_argument("--npml", type=int, default=10)
    p.add_argument("--wakelength", type=float, default=0.3)
    p.add_argument("--tfsf_truncation", type=float, default=8.0)
    p.add_argument("--tfsf_template", default="operator",
                   choices=["operator", "staircase"])  # fmt: skip
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    wakis.sources.Beam.tfsf_truncation = args.tfsf_truncation
    # Same TEM template for both schemes (identical to 'staircase' on
    # staircase grids; the discrete TEM field of the conformal operators)
    wakis.sources.Beam.tfsf_template = args.tfsf_template
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    grid, (bc_low, bc_high), solver_kw, skip = build(args)
    t_grid = time.time() - t0
    tmp = args.out + "_tmp/"
    wake = wakis.WakeSolver(
        q=1e-9, sigmaz=args.sigma, beta=1.0, xsource=0.0, ysource=0.0,
        xtest=0.0, ytest=0.0, skip_cells=skip,
        results_folder=tmp, Ez_file=tmp + "Ez.h5", save=False, verbose=0,
    )  # fmt: skip
    wake.n_transverse_cells = 0
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    solver = wakis.SolverFIT3D(
        grid, wake, bc_low=bc_low, bc_high=bc_high, n_pml=args.npml,
        cfln=args.cfln, scheme=args.scheme, use_gpu=args.gpu, verbose=1,
        **solver_kw,
    )  # fmt: skip
    t_init = time.time() - t0

    steps = {"t": []}

    def tick(solver, t):
        steps["t"].append(time.perf_counter())

    t0 = time.time()
    solver.wakesolve(
        wakelength=args.wakelength, compute_plane="longitudinal", callback=tick
    )
    t_total = time.time() - t0
    ts = np.diff(steps["t"])
    k_loss = wake.calc_loss_factor(save=False)
    # Same wake with linear time interpolation of Ez (see wake_interp.py)
    from wake_interp import wake_potential_interp

    WP_i = wake_potential_interp(tmp + "Ez.h5", np.asarray(wake.s), wake.q,
                                 wake.v, wake.ti, skip_cells=skip)  # fmt: skip
    lam = np.asarray(wake.lambdas)
    k_loss_interp = float(np.trapezoid(WP_i * lam, wake.s) / np.trapezoid(lam, wake.s))
    res = dict(
        args=vars(args),
        cells=int(grid.Nx * grid.Ny * grid.Nz),
        shape=[int(grid.Nx), int(grid.Ny), int(grid.Nz)],
        dt=float(solver.dt), magic_dt=float(np.min(grid.dz) / c_light),
        Nt=int(solver.Nt),
        t_grid=t_grid, t_init=t_init, t_wakesolve=t_total,
        t_step_median=float(np.median(ts)) if ts.size else None,
        t_loop=float(steps["t"][-1] - steps["t"][0]) if ts.size else None,
        k_loss=float(k_loss),
        k_loss_interp=k_loss_interp,
        s=np.asarray(wake.s).tolist(), WP=np.asarray(wake.WP).tolist(),
        WP_interp=WP_i.tolist(),
        tete_dt_limits={k: float(v) for k, v in
                        getattr(solver, "tete_dt_limits", {}).items()},  # fmt: skip
        conformal_unstable_faces=int(getattr(solver, "conformal_unstable_faces", 0)),
        tete_stabilization=getattr(solver, "tete_stabilization", {}),
    )
    with open(args.out + ".json", "w") as f:
        json.dump(res, f)
    for fn in os.listdir(tmp):
        os.remove(tmp + fn)
    os.rmdir(tmp)
    print(f"done {args.out}: Nt={solver.Nt} step={res['t_step_median']} "
          f"k={k_loss:.6g} V/pC")  # fmt: skip


if __name__ == "__main__":
    main()
