# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Equal-accuracy and time-domain cost study of mass variants (`consistent`,
`mixed`) and interior patch smoothing (`smooth_patch`) on the PEC sphere
(`ka = 2`, wavelength `lambda = 1.0 m`), alongside the pure-Cartesian
`staircase` and all-tetrahedral `tet` baselines and `SolverFIT3D` on the same
Cartesian node lines `h = 1 / N`.

Invocation:
    python studies/hybrid-cart-tet/run_mixed_study.py --ns 10 16 > studies/hybrid-cart-tet/results_mixed_study.json
"""

import argparse
import contextlib
import json
import os
import platform
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as C0

from wakis.field import Field
from wakis.gridFIT3D import GridFIT3D
from wakis.hybrid import HybridFIT3D, delaunay_box_mesh, smooth_patch
from wakis.hybrid.verification import pec_sphere_scattering
from wakis.solverFIT3D import SolverFIT3D

WAVELENGTH_M = 1.0
KA_SPHERE = 2.0
T_PHYS_S = float(WAVELENGTH_M / C0)
LU_MAX_COUPLED_BLOCK = 60000

FAR_ZONE_NORM = (
    "relative L2 norm ||e_num - e_ref||_2 / ||e_ref||_2 of scattered edge "
    "voltages against the analytical PEC-sphere Mie series (wakis.hybrid.mie_pec) "
    "on all free Cartesian edges in the cubic shell "
    "B + 0.25*h < ||x||_inf < B + gap - 0.25*h over all 4*pi solid angles "
    "(theta in [0, pi], phi in [0, 2*pi))"
)

CART_UNKNOWNS_DEFINITION = (
    "cart_n_unknowns = cart_n_free_edges + cart_n_faces, where "
    "cart_n_free_edges = 3*(2*nh)*(2*nh-1)^2 is the number of active primary "
    "electric edges off the PEC outer walls with ieps > 0, and "
    "cart_n_faces = 3*(2*nh)^2*(2*nh+1) is the number of active primary "
    "magnetic faces with imu > 0 and grid.iA > 0 on the (2*nh)^3 Cartesian grid "
    "with node lines v = h * arange(-nh, nh + 1); cart_n_padded = 6*Nx*Ny*Nz "
    "is the padded 3D array storage length."
)

CART_CONFORMAL_REASON = (
    "SolverFIT3D with grid option conformal='pec' cannot run the frequency-domain "
    "PEC-sphere scattered-field benchmark pec_sphere_scattering: (1) SolverFIT3D is "
    "a time-domain-only solver without solve_frequency or the complex-coordinate-"
    "stretching absorber (stretch_functions), and (2) wakis.conformal."
    "apply_conformal_materials embeds homogeneous PEC boundary conditions "
    "(E_tan = 0) into shortened open edge lengths Ds and the stabilized inverse "
    "face-area matrix iDa = T^T diag(1/A_hat) T rather than exposing sphere-surface "
    "edges for inhomogeneous scattered-field Dirichlet excitation (-v_inc)."
)

VARIANT_SPECS = {
    "consistent": {"method": "hybrid", "mass": "consistent", "smooth": False},
    "mixed": {"method": "hybrid", "mass": "mixed", "smooth": False},
    "mixed_smoothed": {"method": "hybrid", "mass": "mixed", "smooth": True},
    "consistent_smoothed": {
        "method": "hybrid",
        "mass": "consistent",
        "smooth": True,
    },
    "staircase": {"method": "staircase", "mass": "consistent", "smooth": False},
    "tet": {"method": "tet", "mass": "consistent", "smooth": False},
}


def _source_pulse():
    f0 = C0 / WAVELENGTH_M
    t_d, tau = 1.5 / f0, 0.5 / f0

    def pulse(t):
        return np.exp(-(((t - t_d) / tau) ** 2)) * np.sin(2 * np.pi * f0 * t)

    return pulse


def measure_cartesian_solverfit3d(v, n_steps=200, mutate=None):
    """Benchmark `SolverFIT3D` on the same Cartesian node lines `v`."""
    if mutate not in (None, "ignore_pec_walls"):
        raise ValueError(f"Unsupported mutate={mutate!r}")
    with contextlib.redirect_stdout(sys.stderr):
        grid = GridFIT3D(
            float(v[0]),
            float(v[-1]),
            float(v[0]),
            float(v[-1]),
            float(v[0]),
            float(v[-1]),
            x=v,
            y=v,
            z=v,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            verbose=0,
        )
        # Warm-up step to complete step_0 matrix assembly before wall-clock timing
        solver.one_step()

    # SolverFIT3D._attrcleanup() (wakis/solverFIT3D.py) deletes the temporary
    # self.BC and self.iDa attributes during step_0 (the warm-up one_step()
    # call), and SolverFIT3D exposes no persistent attribute or method for the
    # PEC boundary edge mask. Derive the PEC outer-wall tangential edge mask
    # from the public grid geometry (solver.Nx, solver.Ny, solver.Nz) and use
    # grid.iA for the primary inverse face areas: with
    # GridFIT3D(boundary_at_domain_edge=True), transverse node indices 0 (low
    # wall, v[0]) and -1 (high wall, v[-1], before the isolated edge-ghost cell
    # where solver.ieps == 0) are the tangential PEC boundary planes.
    pec_wall_edges = Field(solver.Nx, solver.Ny, solver.Nz, dtype=np.int8)
    if mutate != "ignore_pec_walls":
        for d in ("x", "y", "z"):
            if d != "x":
                pec_wall_edges[0, :, :, d] = 1
                pec_wall_edges[-1, :, :, d] = 1
            if d != "y":
                pec_wall_edges[:, 0, :, d] = 1
                pec_wall_edges[:, -1, :, d] = 1
            if d != "z":
                pec_wall_edges[:, :, 0, d] = 1
                pec_wall_edges[:, :, -1, d] = 1

    not_pec_wall = pec_wall_edges.toarray().ravel() == 0
    ieps_diag = solver.ieps.toarray().ravel() > 0
    imu_diag = solver.imu.toarray().ravel() > 0
    ia_diag = grid.iA.toarray().ravel() > 0

    cart_n_free_edges = int((not_pec_wall & ieps_diag).sum())
    cart_n_faces = int((imu_diag & ia_diag).sum())
    cart_n_unknowns = int(cart_n_free_edges + cart_n_faces)
    cart_n_padded = int(6 * solver.N)

    pulse = _source_pulse()
    nh = (len(v) - 1) // 2
    ix_src, iy_src, iz_src = min(2, nh), nh, nh

    t0 = time.perf_counter()
    for step in range(n_steps):
        solver.J[ix_src, iy_src, iz_src, "z"] = pulse(step * solver.dt)
        solver.one_step()
    wall_s = time.perf_counter() - t0

    cart_s_per_step = float(wall_s / n_steps)
    cart_cost_per_unknown_step_s = float(cart_s_per_step / cart_n_unknowns)
    cart_dt = float(solver.dt)
    cart_steps = int(np.ceil(T_PHYS_S / cart_dt))
    cart_time_to_solution_s = float(cart_steps * cart_s_per_step)

    return {
        "cart_n_unknowns": cart_n_unknowns,
        "cart_n_free_edges": cart_n_free_edges,
        "cart_n_faces": cart_n_faces,
        "cart_n_padded": cart_n_padded,
        "cart_dt": cart_dt,
        "cart_s_per_step": cart_s_per_step,
        "cart_cost_per_unknown_step_s": cart_cost_per_unknown_step_s,
        "cost_per_unknown_step_s": cart_cost_per_unknown_step_s,
        "cart_steps": cart_steps,
        "cart_time_to_solution_s": cart_time_to_solution_s,
    }


def _coupled_block_size(M):
    M = sp.csr_matrix(M)
    d = M.diagonal()
    off = M - sp.diags(d)
    off.eliminate_zeros()
    return int((np.diff(off.tocsr().indptr) > 0).sum())


def _lu_skip_reason(coupled_block, lu_max_rows=LU_MAX_COUPLED_BLOCK):
    return (
        f"Skipped SuperLU mass factorization (mass_solver='lu'): coupled_block = "
        f"{int(coupled_block)} exceeds the {int(lu_max_rows)}-row guard because 3D "
        f"consistent-mass sparse LU fill-in (4.7e7 factor entries already at 26000 "
        f"rows) makes factorization memory- and time-prohibitive."
    )


def _build_td_solver(
    N,
    variant,
    v,
    nb,
    nh,
    a,
    tm_hybrid,
    tm_hybrid_smooth,
    mass_solver="chebyshev",
    dt=None,
):
    spec = VARIANT_SPECS[variant]
    method = spec["method"]
    mass = spec["mass"]
    smooth = spec["smooth"]

    if method == "hybrid":
        i0, i1 = nh - nb, nh + nb
        box = ((i0, i1),) * 3
        tm = tm_hybrid_smooth if smooth else tm_hybrid
        return HybridFIT3D(
            v,
            v,
            v,
            box=box,
            tet_mesh=tm,
            mass=mass,
            mass_solver=mass_solver,
            dt=dt,
        )
    if method == "staircase":

        def pec(X, Y, Z):
            return X**2 + Y**2 + Z**2 < a**2

        return HybridFIT3D(v, v, v, pec=pec, mass_solver=mass_solver, dt=dt)
    if method == "tet":
        h = 1.0 / N
        i0, i1 = 1, 2 * nh - 1
        box_tet = ((i0, i1),) * 3
        tm_tet = delaunay_box_mesh(
            v[i0 : i1 + 1],
            v[i0 : i1 + 1],
            v[i0 : i1 + 1],
            h=h,
            sphere=((0.0, 0.0, 0.0), a),
            hole=True,
        )
        return HybridFIT3D(
            v,
            v,
            v,
            box=box_tet,
            tet_mesh=tm_tet,
            mass="consistent",
            mass_solver=mass_solver,
            dt=dt,
        )
    raise ValueError(f"Unsupported method={method!r}")


def measure_variant(
    N,
    variant,
    v,
    nb,
    nh,
    a,
    tm_hybrid,
    tm_hybrid_smooth,
    cart_ref,
    n_steps=200,
    gap=0.2,
    T=0.5,
    mass_solver="chebyshev",
    lu_max_rows=LU_MAX_COUPLED_BLOCK,
):
    if mass_solver not in ("chebyshev", "lu"):
        raise ValueError(
            f"mass_solver must be 'chebyshev' or 'lu', got {mass_solver!r}"
        )
    spec = VARIANT_SPECS[variant]
    fd = pec_sphere_scattering(
        N,
        method=spec["method"],
        ka=KA_SPHERE,
        gap=gap,
        T=T,
        mass=spec["mass"],
        smooth=spec["smooth"],
    )

    s = _build_td_solver(
        N,
        variant,
        v,
        nb,
        nh,
        a,
        tm_hybrid,
        tm_hybrid_smooth,
        mass_solver=mass_solver,
        dt=1.0,
    )
    n_free = int(s.n_free)
    n_faces = int(s.n_faces)
    n_unknowns = int(n_free + n_faces)
    coupled_block = _coupled_block_size(s.M)

    if mass_solver == "lu" and coupled_block > lu_max_rows:
        dt = None
        dt_max = None
        cheb_degree = None
        s_per_step = None
        cost_per_unknown_step_s = None
        steps = None
        time_to_solution_s = None
        reason = _lu_skip_reason(coupled_block, lu_max_rows=lu_max_rows)
    else:
        dt_max = float(s.dt_max)
        dt = float(0.95 * dt_max)
        s.dt = dt
        ms = s.msolver
        h = 1.0 / N
        s.add_current((-nh * h + 2 * h, 0.1, 0.05), 2, _source_pulse())

        t0 = time.perf_counter()
        s.run(n_steps)
        s_per_step = float((time.perf_counter() - t0) / n_steps)

        cheb_degree = int(ms.degree)
        cost_per_unknown_step_s = float(s_per_step / n_unknowns)
        steps = int(np.ceil(T_PHYS_S / dt))
        time_to_solution_s = float(steps * s_per_step)
        reason = None

    rec = {
        "N": int(N),
        "variant": variant,
        "method": spec["method"],
        "mass": spec["mass"],
        "mass_solver": mass_solver,
        "smooth": bool(spec["smooth"]),
        "far_zone_error": float(fd["err_far"]),
        "err_far": float(fd["err_far"]),
        "err_near": float(fd["err_near"]),
        "n_far": int(fd["n_far"]),
        "n_near": int(fd["n_near"]),
        "fd_n_free": int(fd["n_free"]),
        "fd_n_tets": int(fd["n_tets"]),
        "fd_residual": float(fd["residual"]),
        "n_unknowns": n_unknowns,
        "n_free": n_free,
        "n_faces": n_faces,
        "n_tets": int(s.tet.n_cells) if s.tet is not None else 0,
        "dt": dt,
        "dt_max": dt_max,
        "cheb_degree": cheb_degree,
        "coupled_block": coupled_block,
        "s_per_step": s_per_step,
        "cost_per_unknown_step_s": cost_per_unknown_step_s,
        "cart_cost_per_unknown_step_s": float(cart_ref["cart_cost_per_unknown_step_s"]),
        "cart_s_per_step": float(cart_ref["cart_s_per_step"]),
        "cart_n_unknowns": int(cart_ref["cart_n_unknowns"]),
        "T_phys": T_PHYS_S,
        "steps": steps,
        "time_to_solution_s": time_to_solution_s,
    }
    if mass_solver == "lu":
        rec["lu_max_coupled_block"] = int(lu_max_rows)
        rec["lu_skipped_reason"] = reason
        rec["reason"] = reason
    return rec


def find_equal_accuracy_pairs(table, rel_tol=0.02):
    """Return pairs of table rows whose `far_zone_error` agree within `rel_tol`.

    Does not claim equal accuracy unless two measured rows actually fall within
    `rel_tol` relative difference.
    """
    pairs = []
    for i, row_a in enumerate(table):
        for row_b in table[i + 1 :]:
            err_a = row_a["far_zone_error"]
            err_b = row_b["far_zone_error"]
            denom = max(abs(err_a), abs(err_b), 1e-30)
            rel_diff = abs(err_a - err_b) / denom
            if rel_diff <= rel_tol:
                pairs.append(
                    {
                        "point_a": {
                            "variant": row_a["variant"],
                            "N": row_a["N"],
                            "far_zone_error": err_a,
                            "time_to_solution_s": row_a["time_to_solution_s"],
                        },
                        "point_b": {
                            "variant": row_b["variant"],
                            "N": row_b["N"],
                            "far_zone_error": err_b,
                            "time_to_solution_s": row_b["time_to_solution_s"],
                        },
                        "rel_error_diff": float(rel_diff),
                    }
                )
    return pairs


def run_study(
    ns=(10, 16),
    variants=None,
    n_steps=200,
    gap=0.2,
    T=0.5,
    quick=False,
    mass_solver="chebyshev",
    lu_max_rows=LU_MAX_COUPLED_BLOCK,
):
    if mass_solver not in ("chebyshev", "lu"):
        raise ValueError(
            f"mass_solver must be 'chebyshev' or 'lu', got {mass_solver!r}"
        )
    if quick:
        if tuple(ns) == (10, 16):
            ns = (10,)
        if n_steps == 200:
            n_steps = 5
    if variants is None:
        variants = (
            "consistent",
            "mixed",
            "mixed_smoothed",
            "staircase",
            "tet",
        )
    t_start = time.perf_counter()
    k = 2.0 * np.pi / WAVELENGTH_M
    a = KA_SPHERE / k

    results_by_n = {}
    cartesian_refs = {}
    table = []
    lu_skipped = {}

    for N in ns:
        N = int(N)
        h = 1.0 / N
        nb = int(np.ceil((a + max(0.12, 2.2 * h)) / h))
        nh = nb + 3
        v = h * np.arange(-nh, nh + 1)
        i0, i1 = nh - nb, nh + nb

        tm_hybrid = delaunay_box_mesh(
            v[i0 : i1 + 1],
            v[i0 : i1 + 1],
            v[i0 : i1 + 1],
            h=h,
            sphere=((0.0, 0.0, 0.0), a),
            hole=True,
        )
        need_smooth = any(VARIANT_SPECS[var]["smooth"] for var in variants)
        tm_hybrid_smooth = smooth_patch(tm_hybrid) if need_smooth else None

        cart_ref = measure_cartesian_solverfit3d(v, n_steps=n_steps)
        cartesian_refs[f"N={N}"] = cart_ref

        per_n = {}
        for variant in variants:
            rec = measure_variant(
                N,
                variant,
                v,
                nb,
                nh,
                a,
                tm_hybrid,
                tm_hybrid_smooth,
                cart_ref,
                n_steps=n_steps,
                gap=gap,
                T=T,
                mass_solver=mass_solver,
                lu_max_rows=lu_max_rows,
            )
            per_n[variant] = rec
            row = {
                "variant": variant,
                "N": int(N),
                "mass_solver": mass_solver,
                "far_zone_error": rec["far_zone_error"],
                "time_to_solution_s": rec["time_to_solution_s"],
                "n_unknowns": rec["n_unknowns"],
                "dt": rec["dt"],
                "dt_max": rec["dt_max"],
                "cheb_degree": rec["cheb_degree"],
                "coupled_block": rec["coupled_block"],
                "s_per_step": rec["s_per_step"],
                "cost_per_unknown_step_s": rec["cost_per_unknown_step_s"],
                "cart_cost_per_unknown_step_s": rec["cart_cost_per_unknown_step_s"],
                "steps": rec["steps"],
            }
            if mass_solver == "lu":
                row["lu_skipped_reason"] = rec["lu_skipped_reason"]
                row["reason"] = rec["reason"]
                if rec["reason"] is not None:
                    lu_skipped[f"N={N}.{variant}"] = {
                        "coupled_block": rec["coupled_block"],
                        "lu_max_coupled_block": int(lu_max_rows),
                        "reason": rec["reason"],
                    }
            table.append(row)
        results_by_n[f"N={N}"] = per_n

    out = {
        "quick": bool(quick),
        "mass_solver": mass_solver,
        "meta": {
            "quick": bool(quick),
            "mass_solver": mass_solver,
            "machine": platform.node(),
            "cpus": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "n_steps_timed": int(n_steps),
            "ka": KA_SPHERE,
            "wavelength_m": WAVELENGTH_M,
            "wall_s": float(time.perf_counter() - t_start),
        },
        "T_phys": T_PHYS_S,
        "T_phys_definition": (
            "One period T_phys = lambda / c0 = 1.0 / c0 [s] of the incident "
            "plane wave at ka = 2, wavelength lambda = 1.0 m."
        ),
        "far_zone_norm": FAR_ZONE_NORM,
        "cart_unknowns_definition": CART_UNKNOWNS_DEFINITION,
        "cart_conformal": None,
        "cart_conformal_reason": CART_CONFORMAL_REASON,
        "cartesian_solverfit3d": cartesian_refs,
        "results": results_by_n,
        "table": table,
        "equal_accuracy_pairs": find_equal_accuracy_pairs(table),
    }
    if mass_solver == "lu":
        out["lu_max_coupled_block"] = int(lu_max_rows)
        out["lu_skipped"] = lu_skipped
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="*", default=None)
    ap.add_argument(
        "--variants",
        nargs="*",
        default=[
            "consistent",
            "mixed",
            "mixed_smoothed",
            "staircase",
            "tet",
        ],
    )
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--gap", type=float, default=0.2)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument(
        "--mass-solver",
        choices=["chebyshev", "lu"],
        default="chebyshev",
        help="Solver for the coupled mass block ('chebyshev' or 'lu').",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test run with N = 10 only and n_steps = 5.",
    )
    args = ap.parse_args()
    ns = args.ns if args.ns is not None else ([10] if args.quick else [10, 16])
    n_steps = args.n_steps if args.n_steps is not None else (5 if args.quick else 200)
    out = run_study(
        ns=ns,
        variants=args.variants,
        n_steps=n_steps,
        gap=args.gap,
        T=args.T,
        quick=args.quick,
        mass_solver=args.mass_solver,
    )
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
