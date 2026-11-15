# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
3D multi-block curvilinear hexahedral (`HexMesh`) cavity convergence study.

Compares `CurvilinearFIT3D` on multi-block `HexMesh` grids (with the default
``pec_reconstruction="constrained"`` PEC corner tangency rule and sharp-octant
``D = 0`` node constraint) against Cartesian `"SolverFIT3D staircase"`
(``conformal=False``) and `"SolverFIT3D conformal"` (``conformal="pec"``)
baselines on the same bounding box at matching unknown counts:

1. ``butterfly_disk``: circular PEC cylinder of radius ``a = 1.0 m`` and axial
   length ``L = 1.2 m`` meshed with the 5-block O-grid `butterfly_disk_nodes`
   at ``n in (4, 6, 8, 12)``; fundamental ``TM010`` (`cylinder_tm010_omega`)
   and ``TE111`` (`cylinder_te111_omega`) cavity modes.
2. ``cubed_sphere``: PEC ball of radius ``a = 1.0 m`` meshed with the 7-block
   `cubed_sphere_ball_nodes` at ``n in (6, 8, 12, 16)``; lowest ``l = 1``
   magnetic multipole ``TE_l1`` (root ``x'_{1,1} = 2.743707...`` of
   ``d/dx[x j_1(x)] = 0``) and electric multipole ``TM_l1`` (root
   ``x_{1,1} = 4.493409...`` of ``j_1(x) = 0``, the second value returned by
   `sphere_lowest_omegas(a=1.0, l=1)` and the 9th cavity mode selected by
   `_select_sphere_tm_l1(omegas)` = ``omegas[8]``).

For each run and mode, the discrete eigenfrequency is selected as the positive
discrete eigenvalue (``omega > 1e3 rad/s``) nearest to the analytical angular
frequency ``omega_ana``, matching ``studies/curvilinear3d/pec_fix_measure.py``.
Runs are executed serially one at a time, and results are written incrementally
to ``studies/curvilinear3d/results/hexmesh_study.json`` and
``studies/curvilinear3d/results/hexmesh_table.md``.
"""

import argparse
import ctypes
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import c as C0

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from studies.curvilinear3d.study_cavities import (
    _analytic_edge_field_cartesian,
    _grid_edge_midpoints,
)
from wakis.curvilinear.mappings3d import (
    butterfly_disk_nodes,
    cubed_sphere_ball_nodes,
    cylinder_te111_omega,
    cylinder_tm010_omega,
    sphere_lowest_omegas,
)
from wakis.curvilinear.solver3d import CurvilinearFIT3D
from wakis.gridFIT3D import GridFIT3D
from wakis.hybrid.solver import _nested_dissection
from wakis.solverFIT3D import SolverFIT3D

ALL_GEOMETRIES = ("butterfly_disk", "cubed_sphere")
ALL_METHODS = (
    "CurvilinearFIT3D",
    "SolverFIT3D staircase",
    "SolverFIT3D conformal",
)

DEFAULT_RESOLUTIONS = {
    "butterfly_disk": (4, 6, 8, 12),
    "cubed_sphere": (6, 8, 12, 16),
}

# Pre-computed per-axis Cartesian cell counts N_cart that yield the closest
# active unknown edge count n_unknowns to CurvilinearFIT3D at each n.
CLOSEST_CARTESIAN_N = {
    ("butterfly_disk", False): {4: 7, 6: 11, 8: 14, 12: 22},
    ("butterfly_disk", "pec"): {4: 7, 6: 11, 8: 14, 12: 22},
    ("cubed_sphere", False): {6: 13, 8: 18, 12: 27, 16: 36},
    ("cubed_sphere", "pec"): {6: 14, 8: 18, 12: 28, 16: 37},
}

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_DEFAULT_JSON = _RESULTS_DIR / "hexmesh_study.json"
_DEFAULT_TABLE = _RESULTS_DIR / "hexmesh_table.md"


def _trim_heap():
    """Force garbage collection and return freed glibc heap arenas to the OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _git_commit_sha():
    """Return the current git HEAD commit SHA."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _geometry_modes(geometry, a=1.0, length=1.2):
    """Return ordered list of ``(mode_name, omega_ana_rad_s)`` for ``geometry``."""
    if geometry == "butterfly_disk":
        return [
            ("TM010", float(cylinder_tm010_omega(a=a))),
            ("TE111", float(cylinder_te111_omega(a=a, length=length))),
        ]
    if geometry == "cubed_sphere":
        w_te1, w_tm1 = sphere_lowest_omegas(a=a, l=1)
        return [
            ("TE_l1", float(w_te1)),
            ("TM_l1", float(w_tm1)),
        ]
    raise ValueError(f"Unknown geometry {geometry!r}")


def _nearest_positive_omega(omegas, omega_ana):
    """Select the positive discrete eigenfrequency (> 1e3 rad/s) nearest to
    ``omega_ana``, matching ``studies/curvilinear3d/pec_fix_measure.py``."""
    pos = np.asarray(omegas, dtype=float)
    pos = pos[pos > 1e3]
    if pos.size == 0:
        raise RuntimeError("No positive physical eigenfrequencies (> 1e3 rad/s) found.")
    idx = int(np.argmin(np.abs(pos - float(omega_ana))))
    return float(pos[idx])


def _assemble_curvilinear_eigen_operators(solver, build_diag_precond=False):
    """Assemble the reduced curl-curl + nullspace-penalty matrix ``H`` and mass
    matrix ``W_eps_u`` identically to `CurvilinearFIT3D.cavity_frequencies`."""
    C_u = solver.C_u.tocsr()
    W_eps_u = solver.W_eps_u.tocsc()
    W_nu = solver.W_nu.tocsr()

    CW = (C_u @ W_eps_u).tocsr()
    A = (CW.T @ W_nu @ CW).tocsr()
    A = (0.5 * (A + A.T)).tocsr()

    G_u = solver.block.gradient_unknown()
    alpha = float(np.max(A.diagonal()))
    GG = (G_u @ G_u.T).tocsr()
    H = (A + alpha * GG).tocsc()

    H0 = None
    if build_diag_precond:
        d_eps = sp.diags(W_eps_u.diagonal())
        d_nu = sp.diags(W_nu.diagonal())
        CW0 = (C_u @ d_eps).tocsr()
        H0 = (CW0.T @ d_nu @ CW0 + alpha * GG).tocsc()
        del CW0, d_eps, d_nu
    del GG, A, CW, G_u

    return H, W_eps_u, H0, int(solver.n_unknown_edges)


def run_curvilinear_case(geometry, n, a=1.0, length=1.2):
    """Build `HexMesh` + `CurvilinearFIT3D` and solve cavity eigenfrequencies.

    Parameters
    ----------
    geometry : {'butterfly_disk', 'cubed_sphere'}
        Multi-block hexahedral cavity geometry.
    n : int
        Per-axis block resolution parameter passed to the generator.
    a : float, optional
        Cavity radius [m] (default 1.0).
    length : float, optional
        Axial cylinder length [m] for ``butterfly_disk`` (default 1.2).

    Returns
    -------
    dict
        Run summary with keys ``n_unknowns``, ``h``, ``wall_s``, and
        ``frequencies`` mapping mode name to discrete ``omega_disc`` [rad/s].
    """
    _trim_heap()
    t0 = time.perf_counter()
    if geometry == "butterfly_disk":
        msh = butterfly_disk_nodes(int(n), a=a, length=length)
    elif geometry == "cubed_sphere":
        msh = cubed_sphere_ball_nodes(int(n), a=a)
    else:
        raise ValueError(f"Unknown geometry {geometry!r}")

    modes = _geometry_modes(geometry, a=a, length=length)
    solver = CurvilinearFIT3D(msh, _pec_reconstruction="constrained")
    n_unknowns = int(solver.n_unknown_edges)

    if n_unknowns <= 25000:
        n_modes = 6 if geometry == "butterfly_disk" else 11
        omegas = solver.cavity_frequencies(k=n_modes)
        freqs = {
            mode_name: _nearest_positive_omega(omegas, w_ana)
            for mode_name, w_ana in modes
        }
        del solver, msh
    else:
        # Memory-guarded shift-invert for n=12 (35,484 unknowns) and n=16
        # (84,560 unknowns) on cubed_sphere using ILU-preconditioned BiCGSTAB.
        H, W_eps_u, H0, n_unknowns = _assemble_curvilinear_eigen_operators(
            solver, build_diag_precond=(n_unknowns > 40000)
        )
        del solver, msh
        _trim_heap()
        freqs = {}
        if n_unknowns <= 40000:
            # Single shift-invert factorization at sigma=0 covering both l=1 modes
            ilu = spla.spilu(
                H,
                drop_tol=1e-3,
                fill_factor=10,
                permc_spec="MMD_AT_PLUS_A",
            )
            M_pre = spla.LinearOperator(H.shape, matvec=ilu.solve, dtype=np.float64)

            def _solve_op(b, H_mat=H, M_p=M_pre):
                x, _ = spla.bicgstab(H_mat, b, rtol=1e-9, atol=0.0, maxiter=200, M=M_p)
                return x

            op = spla.LinearOperator(H.shape, matvec=_solve_op, dtype=np.float64)
            rng = np.random.default_rng(0)
            v0 = rng.standard_normal(n_unknowns)
            evals = spla.eigsh(
                H,
                k=11,
                M=W_eps_u,
                sigma=0.0,
                OPinv=op,
                which="LM",
                v0=v0,
                tol=1e-6,
                return_eigenvectors=False,
            )
            del ilu, M_pre, op
            omegas = np.sqrt(np.sort(np.maximum(evals.real, 0.0)))
            for mode_name, w_ana in modes:
                freqs[mode_name] = _nearest_positive_omega(omegas, w_ana)
        else:
            # For n=16 (84,560 unknowns), precondition BiCGSTAB on H with ILU
            # of the diagonal-Hodge curl-curl + penalty matrix H0.
            ilu0 = spla.spilu(
                H0,
                drop_tol=1e-3,
                fill_factor=8,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
            del H0
            _trim_heap()
            M_pre = spla.LinearOperator(H.shape, matvec=ilu0.solve, dtype=np.float64)

            def _solve_H(b, H_mat=H, M_p=M_pre):
                x, _ = spla.bicgstab(H_mat, b, rtol=1e-10, atol=0.0, M=M_p, maxiter=400)
                return x

            op = spla.LinearOperator(H.shape, matvec=_solve_H, dtype=np.float64)
            rng = np.random.default_rng(0)
            v0 = rng.standard_normal(n_unknowns)
            evals = spla.eigsh(
                H,
                k=11,
                M=W_eps_u,
                sigma=0.0,
                which="LM",
                v0=v0,
                tol=1e-8,
                OPinv=op,
                return_eigenvectors=False,
            )
            del ilu0, M_pre, op
            _trim_heap()
            omegas = np.sqrt(np.sort(np.maximum(evals.real, 0.0)))
            for mode_name, w_ana in modes:
                freqs[mode_name] = _nearest_positive_omega(omegas, w_ana)
        del H, W_eps_u

    wall_s = float(time.perf_counter() - t0)
    _trim_heap()
    return {
        "n_unknowns": n_unknowns,
        "h": float(1.0 / int(n)),
        "wall_s": wall_s,
        "frequencies": freqs,
    }


def run_cartesian_case(geometry, n_cart, conformal=False, a=1.0, length=1.2):
    """Build `GridFIT3D` + `SolverFIT3D` and solve cavity eigenfrequencies.

    Parameters
    ----------
    geometry : {'butterfly_disk', 'cubed_sphere'}
        Benchmark cavity geometry.
    n_cart : int
        Per-axis Cartesian cell count on the bounding box.
    conformal : bool or str, optional
        ``False`` for `"SolverFIT3D staircase"` or ``"pec"`` for
        `"SolverFIT3D conformal"`.
    a : float, optional
        Cavity radius [m] (default 1.0).
    length : float, optional
        Axial cylinder length [m] for ``butterfly_disk`` (default 1.2).

    Returns
    -------
    dict
        Run summary with keys ``n_cart``, ``n_unknowns``, ``h``, ``wall_s``,
        and ``frequencies`` mapping mode name to discrete ``omega_disc`` [rad/s].
    """
    _trim_heap()
    nx = ny = nz = int(n_cart)
    t0 = time.perf_counter()
    if geometry == "butterfly_disk":
        solid = pv.Cylinder(
            center=(0.0, 0.0, 0.5 * length),
            direction=(0.0, 0.0, 1.0),
            radius=a,
            height=length,
            resolution=256,
            capping=True,
        ).triangulate()
        grid = GridFIT3D(
            -a,
            a,
            -a,
            a,
            0.0,
            length,
            nx,
            ny,
            nz,
            stl_solids={"cavity": solid},
            stl_materials={"cavity": "vacuum"},
            conformal=conformal,
            boundary_at_domain_edge=True,
            verbose=0,
        )
        cart_geo = "cylinder"
    elif geometry == "cubed_sphere":
        solid = pv.Sphere(
            radius=a,
            center=(0.0, 0.0, 0.0),
            theta_resolution=180,
            phi_resolution=180,
        ).triangulate()
        grid = GridFIT3D(
            -a,
            a,
            -a,
            a,
            -a,
            a,
            nx,
            ny,
            nz,
            stl_solids={"cavity": solid},
            stl_materials={"cavity": "vacuum"},
            conformal=conformal,
            boundary_at_domain_edge=True,
            verbose=0,
        )
        cart_geo = "sphere"
    else:
        raise ValueError(f"Unknown geometry {geometry!r}")

    solver = SolverFIT3D(
        grid,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        bg="pec",
        use_stl=True,
        verbose=0,
    )
    modes = _geometry_modes(geometry, a=a, length=length)

    C = solver.C.tocsr()
    d1 = solver.ieps.toarray() * solver.itDa.diagonal()
    d2 = solver.tDs.diagonal()
    d4 = solver.Ds.diagonal()
    imu = solver.imu.toarray()
    w_eps = d4 * d1

    W_nu = sp.diags(d2) @ solver.iDa.tocsr() @ sp.diags(imu)
    W_nu = (0.5 * (W_nu + W_nu.T)).tocsr()
    K_full = (C.T @ W_nu @ C).tocsr()
    active = np.flatnonzero((w_eps > 0.0) & (K_full.diagonal() > 0.0))
    n_unknowns = int(active.size)

    C_u = C[:, active].tocsr()
    w_eps_u = w_eps[active]
    W_eps_u = sp.diags(w_eps_u, format="csc")
    CW = (C_u @ W_eps_u).tocsr()
    A = (CW.T @ W_nu @ CW).tocsr()
    A = (0.5 * (A + A.T)).tocsr()

    G = sp.vstack([-solver.Px, -solver.Py, -solver.Pz], format="csc")
    G_act = G[active, :].tocsc()
    WC = (W_nu @ C_u).tocsr()
    WCG = (WC @ G_act).tocsc()
    col_norms = np.sqrt(np.asarray(WCG.multiply(WCG).sum(axis=0)).ravel())
    col_act_nnz = np.diff(G_act.indptr)
    w_nu_scale = float(np.max(np.abs(W_nu.data))) if W_nu.nnz > 0 else 1.0
    null_nodes = np.flatnonzero((col_act_nnz > 0) & (col_norms <= 1e-12 * w_nu_scale))
    G_u = G_act[:, null_nodes].tocsr()

    h_min = min(
        float(np.min(grid.dx)),
        float(np.min(grid.dy)),
        float(np.min(grid.dz)),
    )
    if G_u.shape[1] > 0:
        L_diag = np.asarray((G_u.multiply(G_u)).T @ (1.0 / w_eps_u)).ravel()
        omega_target_sq = (4.0 * max(nx, ny, nz) * C0 / h_min) ** 2
        D_node = sp.diags(omega_target_sq / L_diag)
        H = (A + (G_u @ D_node @ G_u.T)).tocsc()
        del D_node
    else:
        H = A.tocsc()

    X_act = None
    if n_unknowns > 40000:
        xs, ys, zs, _ = _grid_edge_midpoints(grid)
        X_act = np.column_stack([xs[active], ys[active], zs[active]])
        del xs, ys, zs

    v0_map = {}
    for mode_name, _ in modes:
        E_ana = _analytic_edge_field_cartesian(
            grid, cart_geo, mode_name, a=a, length=length
        )[active]
        d_ana = (d4[active] / w_eps_u) * E_ana
        norm_d = float(np.sqrt(max(float(d_ana @ (W_eps_u @ d_ana)), 1e-300)))
        v0_map[mode_name] = d_ana / norm_d

    del (
        solid,
        grid,
        solver,
        C,
        d1,
        d2,
        d4,
        imu,
        w_eps,
        W_nu,
        K_full,
        C_u,
        CW,
        A,
        G,
        G_act,
        WC,
        WCG,
        G_u,
    )
    _trim_heap()

    if n_unknowns > 40000:
        perm = _nested_dissection(H, X_act, leaf=16)
        del X_act
        H = H[perm][:, perm].tocsc()
        w_eps_u = w_eps_u[perm]
        W_eps_u = sp.diags(w_eps_u, format="csc")
        for mode_name in v0_map:
            v0_map[mode_name] = v0_map[mode_name][perm]
        del perm
        _trim_heap()

    freqs = {}
    sigma_factor = 0.99 if conformal else 0.94
    for mode_name, w_ana in modes:
        sigma = (sigma_factor * w_ana) ** 2
        if n_unknowns <= 40000:
            B = (H - sigma * W_eps_u).tocsc()
            lu = spla.splu(
                B,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
            op = spla.LinearOperator(B.shape, matvec=lu.solve, dtype=np.float64)
            evals, _ = spla.eigsh(
                H,
                k=6,
                M=W_eps_u,
                sigma=sigma,
                OPinv=op,
                which="LM",
                v0=v0_map[mode_name],
                tol=1e-8,
            )
            del lu, op, B
        else:
            col_idx = np.repeat(np.arange(H.shape[1]), np.diff(H.indptr))
            diag_idx = np.flatnonzero(H.indices == col_idx)
            del col_idx
            h_diag = H.data[diag_idx].copy()
            H.data[diag_idx] = h_diag - sigma * w_eps_u
            _trim_heap()
            lu = spla.splu(
                H,
                permc_spec="NATURAL",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
            H.data[diag_idx] = h_diag
            del diag_idx, h_diag
            _trim_heap()
            op = spla.LinearOperator(H.shape, matvec=lu.solve, dtype=np.float64)
            evals, _ = spla.eigsh(
                H,
                k=6,
                M=W_eps_u,
                sigma=sigma,
                OPinv=op,
                which="LM",
                v0=v0_map[mode_name],
                tol=1e-8,
            )
            del lu, op

        _trim_heap()
        omegas = np.sqrt(np.sort(np.maximum(evals.real, 0.0)))
        freqs[mode_name] = _nearest_positive_omega(omegas, w_ana)

    del H, W_eps_u, v0_map
    wall_s = float(time.perf_counter() - t0)
    _trim_heap()
    return {
        "n_cart": int(n_cart),
        "n_unknowns": n_unknowns,
        "h": float(2.0 * a / int(n_cart)),
        "wall_s": wall_s,
        "frequencies": freqs,
    }


def compute_orders_and_equal_wall(records):
    """Compute LS and two-level convergence orders and equal-wall-time comparisons.

    Parameters
    ----------
    records : list of dict
        Flat list of per-(geometry, mode, method, n) study records.

    Returns
    -------
    orders : list of dict
        Per-(geometry, mode, method) convergence order and equal-wall-time
        summary entries.
    """
    orders = []
    for geometry in ALL_GEOMETRIES:
        modes = [m for m, _ in _geometry_modes(geometry)]
        for mode_name in modes:
            curv_pts = sorted(
                [
                    r
                    for r in records
                    if r["geometry"] == geometry
                    and r["mode"] == mode_name
                    and r["method"] == "CurvilinearFIT3D"
                ],
                key=lambda r: int(r["n"]),
            )
            curv_finest_wall = float(curv_pts[-1]["wall_s"]) if curv_pts else None

            for method in ALL_METHODS:
                pts = sorted(
                    [
                        r
                        for r in records
                        if r["geometry"] == geometry
                        and r["mode"] == mode_name
                        and r["method"] == method
                    ],
                    key=lambda r: int(r["n"]),
                )
                if len(pts) < 2:
                    continue
                h_arr = np.array([float(r["h"]) for r in pts], dtype=float)
                e_arr = np.array(
                    [max(abs(float(r["rel_err"])), 1e-16) for r in pts],
                    dtype=float,
                )
                slope_ls = float(np.polyfit(np.log(h_arr), np.log(e_arr), 1)[0])
                slope_two_level = float(
                    np.log(e_arr[-2] / e_arr[-1]) / np.log(h_arr[-2] / h_arr[-1])
                )

                eq_wall_entry = "no run"
                if curv_finest_wall is not None:
                    eligible = [
                        r for r in pts if float(r["wall_s"]) <= curv_finest_wall + 1e-12
                    ]
                    if eligible:
                        best = max(eligible, key=lambda r: int(r["n"]))
                        eq_wall_entry = {
                            "n": int(best["n"]),
                            "n_cart": best.get("n_cart"),
                            "n_unknowns": int(best["n_unknowns"]),
                            "rel_err": float(best["rel_err"]),
                            "abs_rel_err": abs(float(best["rel_err"])),
                            "wall_s": float(best["wall_s"]),
                            "target_wall_s": float(curv_finest_wall),
                        }

                orders.append(
                    {
                        "geometry": geometry,
                        "mode": mode_name,
                        "method": method,
                        "resolutions": [int(r["n"]) for r in pts],
                        "n_cart_list": [r.get("n_cart") for r in pts],
                        "order_ls": slope_ls,
                        "order_two_level": slope_two_level,
                        "equal_wall_time": eq_wall_entry,
                    }
                )
    return orders


def format_markdown_table(payload):
    """Format the full study records and convergence orders as Markdown tables."""
    lines = [
        "### Per-run cavity eigenfrequency records",
        "",
        "| Geometry | Mode | Method | `n` (`N_cart`) | `n_unknowns` (`n_hex`) | `h` [m] | `omega_disc` [rad/s] | `omega_ana` [rad/s] | `rel_err` | `wall_s` [s] |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in payload["records"]:
        n_str = f"{r['n']}" if r.get("n_cart") is None else f"{r['n']} ({r['n_cart']})"
        u_str = (
            f"{r['n_unknowns']}"
            if r["method"] == "CurvilinearFIT3D"
            else f"{r['n_unknowns']} ({r['n_unknowns_hex']})"
        )
        lines.append(
            f"| `{r['geometry']}` | `{r['mode']}` | {r['method']} | {n_str} | "
            f"{u_str} | {r['h']:.6f} | {r['omega_disc']:.6e} | "
            f"{r['omega_ana']:.6e} | {r['rel_err']:+.4e} | {r['wall_s']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Convergence orders and equal-wall-time comparison",
            "",
            "| Geometry | Mode | Method | Resolutions `n` | LS order | Two-level order | Equal-wall-time (`wall_s <= curv finest`) |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
        ]
    )
    for o in payload["orders"]:
        res_str = ", ".join(str(x) for x in o["resolutions"])
        eq = o["equal_wall_time"]
        if isinstance(eq, dict):
            n_tag = (
                f"n={eq['n']}"
                if eq.get("n_cart") is None
                else f"n={eq['n']} (N={eq['n_cart']})"
            )
            eq_str = (
                f"`{n_tag}`, `rel_err={eq['rel_err']:+.4e}` "
                f"(`|rel_err|={eq['abs_rel_err']:.4e}`, `{eq['wall_s']:.2f} s` "
                f"<= `{eq['target_wall_s']:.2f} s`)"
            )
        else:
            eq_str = "no run"
        lines.append(
            f"| `{o['geometry']}` | `{o['mode']}` | {o['method']} | "
            f"`[{res_str}]` | {o['order_ls']:.2f} | "
            f"{o['order_two_level']:.2f} | {eq_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def _save_outputs(records, complete, out_json, out_table, cmdline, git_sha):
    """Persist ``hexmesh_study.json`` and ``hexmesh_table.md`` incrementally."""
    orders = compute_orders_and_equal_wall(records)
    payload = {
        "command_line": cmdline,
        "git_commit": git_sha,
        "complete": bool(complete),
        "parameters": {
            "butterfly_disk": {
                "a_m": 1.0,
                "length_m": 1.2,
                "resolutions": list(DEFAULT_RESOLUTIONS["butterfly_disk"]),
                "modes": ["TM010", "TE111"],
            },
            "cubed_sphere": {
                "a_m": 1.0,
                "resolutions": list(DEFAULT_RESOLUTIONS["cubed_sphere"]),
                "modes": ["TE_l1", "TM_l1"],
                "tm_l1_definition": (
                    "Lowest l=1 TM root j_1(x_{1,1}) = 0 (x_{1,1} = 4.493409457909064, "
                    "sphere_lowest_omegas(a=1.0, l=1)[1], corresponding to _select_sphere_tm_l1 "
                    "omegas[8]); mode selected as nearest positive discrete eigenvalue (> 1e3 rad/s)."
                ),
            },
        },
        "records": records,
        "orders": orders,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table_md = format_markdown_table(payload)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text(table_md, encoding="utf-8")
    return payload, table_md


def run_study(
    out_json=_DEFAULT_JSON,
    out_table=_DEFAULT_TABLE,
    resume=True,
    cmdline=None,
):
    """Execute the full serial hexmesh convergence study.

    Parameters
    ----------
    out_json : str or Path, optional
        Output JSON path.
    out_table : str or Path, optional
        Output Markdown table path.
    resume : bool, optional
        If ``True`` and ``out_json`` exists, reuse already completed
        ``(geometry, method, n)`` records and run only missing ones.
    cmdline : str, optional
        Command line string to store in the JSON metadata.

    Returns
    -------
    dict
        Study payload dictionary.
    """
    out_json = Path(out_json)
    out_table = Path(out_table)
    if cmdline is None:
        cmdline = " ".join(sys.argv)
    git_sha = _git_commit_sha()

    records = []
    if resume and out_json.exists():
        try:
            existing = json.loads(out_json.read_text(encoding="utf-8"))
            records = list(existing.get("records", []))
        except (OSError, ValueError):
            records = []

    done_keys = {
        (r["geometry"], r["method"], int(r["n"]))
        for r in records
        if "geometry" in r and "method" in r and "n" in r
    }
    hex_unknowns_map = {
        (r["geometry"], int(r["n"])): int(r["n_unknowns"])
        for r in records
        if r.get("method") == "CurvilinearFIT3D"
    }

    expected_runs = sum(
        len(DEFAULT_RESOLUTIONS[g]) * len(ALL_METHODS) for g in ALL_GEOMETRIES
    )
    completed_runs = len(done_keys)

    for geometry in ALL_GEOMETRIES:
        modes = dict(_geometry_modes(geometry))
        for n in DEFAULT_RESOLUTIONS[geometry]:
            # 1. CurvilinearFIT3D first so n_unknowns_hex is recorded
            key_curv = (geometry, "CurvilinearFIT3D", int(n))
            if key_curv not in done_keys:
                res_c = run_curvilinear_case(geometry, n)
                hex_unknowns_map[(geometry, int(n))] = int(res_c["n_unknowns"])
                for mode_name, w_ana in modes.items():
                    w_disc = float(res_c["frequencies"][mode_name])
                    records.append(
                        {
                            "geometry": geometry,
                            "mode": mode_name,
                            "method": "CurvilinearFIT3D",
                            "n": int(n),
                            "n_cart": None,
                            "n_unknowns": int(res_c["n_unknowns"]),
                            "n_unknowns_hex": int(res_c["n_unknowns"]),
                            "h": float(res_c["h"]),
                            "omega_disc": w_disc,
                            "omega_ana": float(w_ana),
                            "rel_err": float((w_disc - w_ana) / w_ana),
                            "wall_s": float(res_c["wall_s"]),
                        }
                    )
                done_keys.add(key_curv)
                completed_runs += 1
                _save_outputs(
                    records,
                    completed_runs == expected_runs,
                    out_json,
                    out_table,
                    cmdline,
                    git_sha,
                )

            n_hex = hex_unknowns_map[(geometry, int(n))]

            # 2. SolverFIT3D staircase and SolverFIT3D conformal
            for conformal, method_name in [
                (False, "SolverFIT3D staircase"),
                ("pec", "SolverFIT3D conformal"),
            ]:
                key_m = (geometry, method_name, int(n))
                if key_m in done_keys:
                    continue
                n_cart = CLOSEST_CARTESIAN_N[(geometry, conformal)][int(n)]
                res_m = run_cartesian_case(geometry, n_cart, conformal=conformal)
                for mode_name, w_ana in modes.items():
                    w_disc = float(res_m["frequencies"][mode_name])
                    records.append(
                        {
                            "geometry": geometry,
                            "mode": mode_name,
                            "method": method_name,
                            "n": int(n),
                            "n_cart": int(n_cart),
                            "n_unknowns": int(res_m["n_unknowns"]),
                            "n_unknowns_hex": int(n_hex),
                            "h": float(res_m["h"]),
                            "omega_disc": w_disc,
                            "omega_ana": float(w_ana),
                            "rel_err": float((w_disc - w_ana) / w_ana),
                            "wall_s": float(res_m["wall_s"]),
                        }
                    )
                done_keys.add(key_m)
                completed_runs += 1
                _save_outputs(
                    records,
                    completed_runs == expected_runs,
                    out_json,
                    out_table,
                    cmdline,
                    git_sha,
                )

    # Sort records deterministically by geometry, mode, method, n
    mode_order = {"TM010": 0, "TE111": 1, "TE_l1": 0, "TM_l1": 1}
    method_order = {m: i for i, m in enumerate(ALL_METHODS)}
    geo_order = {g: i for i, g in enumerate(ALL_GEOMETRIES)}
    records.sort(
        key=lambda r: (
            geo_order.get(r["geometry"], 99),
            mode_order.get(r["mode"], 99),
            method_order.get(r["method"], 99),
            int(r["n"]),
        )
    )
    payload, table_md = _save_outputs(
        records,
        len(done_keys) == expected_runs,
        out_json,
        out_table,
        cmdline,
        git_sha,
    )
    print(table_md)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the 3D multi-block HexMesh cavity convergence study."
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=_DEFAULT_JSON,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--out-table",
        type=Path,
        default=_DEFAULT_TABLE,
        help="Path to output Markdown table file.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run all cases from scratch even if output JSON exists.",
    )
    args = parser.parse_args(argv)
    run_study(
        out_json=args.out_json,
        out_table=args.out_table,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
