"""
Hybrid Cartesian / tetrahedral FIT study (`wakis.hybrid`, milestone 1).

Parts (independent runs packed onto all cores with multiprocessing):

A. TE10 reflection |R| of a vacuum tetrahedral box patch in a PEC
   rectangular guide (a = 1 m, lambda = 1.4 a) vs Cartesian cells per a,
   N in {8, 12, 16, 20}: Kuhn patch (r = 1, 2), Delaunay patch (h_t = h with
   r = 1; h_t = h/2 with r = 2), trace mutations ('nearest', 'inject') and
   the fit/absorber floor (no patch).
B. PEC cavity 1 x 0.75 x 0.5 m: relative error of the 6 lowest resonances
   vs N in {8, 12, 16, 24, 32} for the pure Cartesian grid, a Kuhn patch and
   a Delaunay patch (middle half of each side).
C. PEC sphere, ka = 2, plane wave, scattered-field formulation: relative L2
   error of the scattered edge voltages vs the Mie series (near zone inside
   the patch box, far zone in the Cartesian shell beyond it) for the
   Cartesian staircase, the hybrid (h_t = h and h_t = h/2) and tetrahedra in
   the whole physical region, vs N; the number of unknowns is recorded for
   the equal-cost comparison.
D. Time domain on the sphere geometry in a closed PEC box (no absorber):
   lambda bounds (Cartesian, tetrahedral, composite), dt ratio to the pure
   Cartesian grid, wall time per step (LU and Chebyshev mass solves; same
   machine, 1 thread) and the energy history over 20000 steps.
E. The front and back faces of the patch reflect with opposite signs, so
   |R| at one patch length oscillates with N (accidental cancellations).
   E sweeps the length L in {0.25, ..., 1.5} a and reports, per N, the
   envelope max_L |R(L)| ~ 2 |r| of a single interface.

Usage: python studies/hybrid-cart-tet/run_study.py --parts A [--quick] [--procs N]
Writes studies/hybrid-cart-tet/results_<part>.json.
"""

import argparse
import json
import os
import platform
import resource
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from wakis.hybrid import HybridFIT3D, delaunay_box_mesh  # noqa: E402
from wakis.hybrid.verification import (  # noqa: E402
    cavity_modes,
    pec_sphere_scattering,
    waveguide_reflection,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if platform.system() == "Darwin" else 1e3)


def task_A(args):
    N, kind = args
    t0 = time.time()
    kw = {
        "floor": dict(patch=None),
        "kuhn_r1": dict(patch="kuhn", r=1),
        "kuhn_r2": dict(patch="kuhn", r=2),
        "delaunay_r1": dict(patch="delaunay", r=1),
        "delaunay_r2": dict(patch="delaunay", r=2, h_tet=0.5 / N),
        "kuhn_nearest": dict(patch="kuhn", r=1, mutate="nearest"),
        "kuhn_inject": dict(patch="kuhn", r=1, mutate="inject"),
    }[kind]
    R, s = waveguide_reflection(N, return_solver=True, **kw)
    return {"N": N, "kind": kind, "R": R, "n_free": int(s.n_free),
            "n_tets": int(s.tet.n_cells) if s.tet is not None else 0,
            "time_s": time.time() - t0, "rss_mb": _rss_mb()}  # fmt: skip


def task_B(args):
    N, kind = args
    t0 = time.time()
    patch = None if kind == "cartesian" else kind
    f, fe, s = cavity_modes(N, patch=patch, n_modes=6)
    return {"N": N, "kind": kind, "f": f.tolist(), "f_exact": fe.tolist(),
            "rel_err": ((f - fe) / fe).tolist(), "n_free": int(s.n_free),
            "time_s": time.time() - t0}  # fmt: skip


def task_C(args):
    N, kind = args
    t0 = time.time()
    method, kw = {
        "staircase": ("staircase", {}),
        "hybrid": ("hybrid", {}),
        "hybrid_fine": ("hybrid", dict(h_tet=0.5 / N, r=2)),
        "tet": ("tet", {}),
    }[kind]
    out = pec_sphere_scattering(N, method=method, ka=2.0, gap=0.2, T=0.5, **kw)
    out.update({"N": N, "kind": kind, "time_s": time.time() - t0, "rss_mb": _rss_mb()})
    return out


def task_D(args):
    (N,) = args
    from scipy.constants import c as C0

    h = 1.0 / N
    a = 2.0 / (2 * np.pi)
    nb = int(np.ceil((a + max(0.12, 2.2 * h)) / h))
    nh = nb + 3
    v = h * np.arange(-nh, nh + 1)
    i0, i1 = nh - nb, nh + nb
    out = {"N": N}
    tm = delaunay_box_mesh(v[i0 : i1 + 1], v[i0 : i1 + 1], v[i0 : i1 + 1], h=h,
                           sphere=((0, 0, 0), a), hole=True)  # fmt: skip
    cart = HybridFIT3D(v, v, v, pec=lambda X, Y, Z: X**2 + Y**2 + Z**2 < a**2)
    out["dt_cart_staircase"] = cart.dt_max
    # LU of the 3D consistent-mass block fills badly; only on the small case
    variants = (("lu", 1e-16), ("chebyshev", 1e-16), ("cheb_tol1e-4", 1e-4))
    if N > 10:
        variants = variants[1:]
    for solver, tol in variants:
        s = HybridFIT3D(v, v, v, box=((i0, i1),) * 3, tet_mesh=tm,
                        mass_solver="lu" if solver == "lu" else "chebyshev", mass_tol=tol)  # fmt: skip
        lc, lt = s.block_lambda_bounds()
        out[f"{solver}"] = {
            "lambda_cart": lc, "lambda_tet": lt, "lambda": s.lambda_max,
            "dt": s.dt_max, "n_free": int(s.n_free), "n_tets": int(tm.n_cells),
            "coupled_block": int(s.msolver.S.size), "cheb_degree": int(s.msolver.degree),
        }  # fmt: skip
        f0 = C0
        s.add_current((-nh * h + 2 * h, 0.1, 0.05), 2,
                      lambda t: np.exp(-(((t - 4 / f0) / (1 / f0)) ** 2)) * np.sin(2 * np.pi * f0 * t))  # fmt: skip
        t0 = time.time()
        s.run(200)
        out[solver]["s_per_step"] = (time.time() - t0) / 200
        if solver == "chebyshev":
            s.sources = []
            E = s.run(20000, energy=True)
            out["energy"] = E[::20].tolist()
            out["energy_rel_ptp"] = float(np.ptp(E) / np.abs(E).mean())
    t0 = time.time()
    cart.add_current((-nh * h + 2 * h, 0.1, 0.05), 2, lambda t: 0.0)
    cart.run(200)
    out["cart_s_per_step"] = (time.time() - t0) / 200
    out["cart_n_free"] = int(cart.n_free)
    return out


def task_E(args):
    N, kind, L = args
    t0 = time.time()
    kw = {"kuhn": dict(patch="kuhn"), "delaunay": dict(patch="delaunay"),
          "kuhn_nearest": dict(patch="kuhn", mutate="nearest")}[kind]  # fmt: skip
    R = waveguide_reflection(N, patch_length=L, **kw)
    return {"N": N, "kind": f"{kind}_L{L}", "base": kind, "L": L, "R": R,
            "time_s": time.time() - t0}  # fmt: skip


PARTS = {
    "A": (task_A, [(N, k) for N in (8, 12, 16, 20)
                   for k in ("floor", "kuhn_r1", "kuhn_r2", "delaunay_r1", "delaunay_r2",
                             "kuhn_nearest", "kuhn_inject")]),
    "B": (task_B, [(N, k) for N in (8, 12, 16, 24, 32) for k in ("cartesian", "kuhn", "delaunay")]),
    "C": (task_C, [(N, k) for N in (8, 10, 12, 16) for k in ("staircase", "hybrid", "hybrid_fine", "tet")]
          + [(20, "staircase"), (20, "hybrid"), (24, "staircase")]),
    "D": (task_D, [(10,), (16,)]),
    "E": (task_E, [(N, k, L) for N in (8, 12, 16, 20, 24) for k in ("kuhn", "delaunay", "kuhn_nearest")
                   for L in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)]),
}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCD")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--only", default=None, help="comma-separated N values (part C)")
    args = ap.parse_args()
    for part in args.parts:
        fn, tasks = PARTS[part]
        if args.quick:
            tasks = tasks[:2]
        if args.only:
            keep = {int(x) for x in args.only.split(",")}
            tasks = [t for t in tasks if t[0] in keep]
        t0 = time.time()
        # largest tasks first so the pool stays busy
        tasks = sorted(tasks, key=lambda t: -t[0])
        with Pool(min(args.procs, len(tasks))) as pool:
            res = pool.map(fn, tasks, chunksize=1)
        meta = {"machine": platform.node(), "cpus": os.cpu_count(), "procs": args.procs,
                "wall_s": time.time() - t0}  # fmt: skip
        suffix = f"_{args.only.replace(',', '-')}" if args.only else ""
        path = os.path.join(HERE, f"results_{part}{suffix}.json")
        with open(path, "w") as f:
            json.dump({"meta": meta, "results": res}, f, indent=1)
        print(part, "done in", meta["wall_s"], "s ->", path, flush=True)
        for r in res:
            print({k: v for k, v in r.items() if k not in ("energy", "f", "f_exact")}, flush=True)


if __name__ == "__main__":
    main()
