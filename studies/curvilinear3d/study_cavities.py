# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
3D curvilinear hexahedral cavity resonance convergence study driver.

Compares `CurvilinearFIT3D` on body-fitted `HexBlock3D` meshes against
Cartesian `"SolverFIT3D staircase"` (`conformal=False`) and
`"SolverFIT3D conformal"` (`conformal="pec"`) baselines on the same bounding
box across three benchmark cavity geometries:

1. ``perturbed_box``: rectangular PEC cavity ``[0, a] x [0, b] x [0, d]`` with
   sinusoidal interior node displacement of amplitude ``0.3 h`` vanishing on
   all walls; fundamental ``TE101`` mode.
2. ``cylinder``: circular PEC cylinder of radius ``a`` and length ``L`` meshed
   via the elliptic cube-to-disk map extruded in ``z``; ``TM010`` and ``TE111``
   modes.
3. ``sphere``: PEC sphere of radius ``a`` meshed via the 3D elliptic
   cube-to-ball map; lowest ``l = 1`` ``TE_l1`` (root of ``d/dx[x j_1(x)] = 0``)
   and ``TM_l1`` (root of ``j_1(x) = 0``) modes.

No plotting is performed in this module; see `figures_curvilinear3d.py`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.constants import c as C0
from scipy.sparse.linalg import eigsh
from scipy.special import j0, j1, jn_zeros, jnp_zeros, jvp, spherical_jn

from studies.curvilinear3d.perturbed_box_order import fit_order as _fit_order
from wakis.curvilinear.mappings3d import (
    box_te101_omega,
    cylinder_nodes,
    cylinder_te111_omega,
    cylinder_tm010_omega,
    min_normalised_corner_det,
    perturbed_box_nodes,
    sphere_lowest_omegas,
    sphere_lowest_roots,
    sphere_nodes,
)
from wakis.curvilinear.mesh3d import HexBlock3D
from wakis.curvilinear.solver3d import CurvilinearFIT3D
from wakis.gridFIT3D import GridFIT3D
from wakis.solverFIT3D import SolverFIT3D

ALL_GEOMETRIES = ("perturbed_box", "cylinder", "sphere")
ALL_METHODS = (
    "CurvilinearFIT3D",
    "SolverFIT3D staircase",
    "SolverFIT3D conformal",
)


def _grid_edge_midpoints(grid):
    """Return flattened Fortran-ordered edge midpoint coordinates (xs, ys, zs)
    for the three edge families of ``grid``."""
    nx, ny, nz = grid.Nx, grid.Ny, grid.Nz
    xn, yn, zn = grid.x[:-1], grid.y[:-1], grid.z[:-1]
    xm = 0.5 * (grid.x[:-1] + grid.x[1:])
    ym = 0.5 * (grid.y[:-1] + grid.y[1:])
    zm = 0.5 * (grid.z[:-1] + grid.z[1:])
    X0, Y0, Z0 = np.meshgrid(xm, yn, zn, indexing="ij")
    X1, Y1, Z1 = np.meshgrid(xn, ym, zn, indexing="ij")
    X2, Y2, Z2 = np.meshgrid(xn, yn, zm, indexing="ij")
    xs = np.concatenate([X0.ravel(order="F"), X1.ravel(order="F"), X2.ravel(order="F")])
    ys = np.concatenate([Y0.ravel(order="F"), Y1.ravel(order="F"), Y2.ravel(order="F")])
    zs = np.concatenate([Z0.ravel(order="F"), Z1.ravel(order="F"), Z2.ravel(order="F")])
    return xs, ys, zs, nx * ny * nz


def _analytic_edge_field_cartesian(
    grid, geometry, mode, a=1.0, b=0.7, d=0.8, length=1.2
):
    """Evaluate the analytical electric field component along each primal edge
    midpoint of a `GridFIT3D` for mode identification in the Cartesian
    staircase/conformal spectrum."""
    xs, ys, zs, n = _grid_edge_midpoints(grid)
    Ex = np.zeros(n, dtype=float)
    Ey = np.zeros(n, dtype=float)
    Ez = np.zeros(n, dtype=float)

    if geometry == "perturbed_box":
        Ey = np.sin(np.pi * xs[n : 2 * n] / a) * np.sin(np.pi * zs[n : 2 * n] / d)
    elif geometry == "cylinder":
        if mode == "TM010":
            x01 = float(jn_zeros(0, 1)[0])
            r = np.hypot(xs[2 * n :], ys[2 * n :])
            Ez = np.where(r <= a, j0(x01 * r / a), 0.0)
        elif mode == "TE111":
            x11p = float(jnp_zeros(1, 1)[0])
            for c, sl in enumerate((slice(0, n), slice(n, 2 * n))):
                x, y, z = xs[sl], ys[sl], zs[sl]
                r = np.hypot(x, y)
                phi = np.arctan2(y, x)
                kr = x11p * r / a
                er = (
                    np.where(r > 1e-12, j1(kr) / np.where(r > 1e-12, kr, 1.0), 0.5)
                    * np.sin(phi)
                    * np.sin(np.pi * z / length)
                )
                ephi = jvp(1, kr) * np.cos(phi) * np.sin(np.pi * z / length)
                inside = (r <= a) & (z >= 0.0) & (z <= length)
                if c == 0:
                    Ex = np.where(inside, er * np.cos(phi) - ephi * np.sin(phi), 0.0)
                else:
                    Ey = np.where(inside, er * np.sin(phi) + ephi * np.cos(phi), 0.0)
    elif geometry == "sphere":
        x1, x2 = sphere_lowest_roots(l=1)
        if mode == "TE_l1":
            k = x1 / a
            for c, sl in enumerate((slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n))):
                x, y, z = xs[sl], ys[sl], zs[sl]
                r = np.sqrt(x * x + y * y + z * z)
                kr = np.maximum(k * r, 1e-12)
                j1_val = spherical_jn(1, kr)
                dj1_val = spherical_jn(1, kr, derivative=True)
                cos_t = np.where(r > 1e-12, z / np.where(r > 1e-12, r, 1.0), 1.0)
                sin_t = np.where(
                    r > 1e-12, np.hypot(x, y) / np.where(r > 1e-12, r, 1.0), 0.0
                )
                er = 2.0 * (j1_val / kr) * cos_t
                etheta = -(j1_val / kr + dj1_val) * sin_t
                rho = np.hypot(x, y)
                cphi = np.where(rho > 1e-12, x / np.where(rho > 1e-12, rho, 1.0), 1.0)
                sphi = np.where(rho > 1e-12, y / np.where(rho > 1e-12, rho, 1.0), 0.0)
                inside = r <= a
                if c == 0:
                    Ex = np.where(inside, (er * sin_t + etheta * cos_t) * cphi, 0.0)
                elif c == 1:
                    Ey = np.where(inside, (er * sin_t + etheta * cos_t) * sphi, 0.0)
                else:
                    Ez = np.where(inside, er * cos_t - etheta * sin_t, 0.0)
        elif mode == "TM_l1":
            k = x2 / a
            for c, sl in enumerate((slice(0, n), slice(n, 2 * n))):
                x, y, z = xs[sl], ys[sl], zs[sl]
                r = np.sqrt(x * x + y * y + z * z)
                kr = np.maximum(k * r, 1e-12)
                j1_val = spherical_jn(1, kr)
                val = np.where(
                    (r <= a) & (r > 1e-12),
                    j1_val / np.where(r > 1e-12, r, 1.0),
                    0.0,
                )
                if c == 0:
                    Ex = -val * y
                else:
                    Ey = val * x
    return np.concatenate([Ex, Ey, Ez])


def baseline_cartesian_eigenfrequencies(
    geometry,
    n,
    conformal=False,
    a=1.0,
    b=0.7,
    d=0.8,
    length=1.2,
):
    """
    Compute discrete cavity eigenfrequencies and unknown edge counts on a
    Cartesian bounding-box mesh using the exact curl and material matrices
    assembled by `SolverFIT3D`.

    Operator extraction from `SolverFIT3D` and `GridFIT3D`
    ------------------------------------------------------
    `SolverFIT3D` exposes its spatial FIT operators directly as attributes
    after initialization (without needing a time-domain run):

    * ``solver.C``: topological curl matrix (``3 N x 3 N``) with entries in
      ``{-1, 0, +1}``;
    * ``solver.Px, solver.Py, solver.Pz``: node-to-edge topological partial
      difference blocks forming the full discrete gradient
      ``G = vstack([-Px, -Py, -Pz])`` with ``C @ G == 0``;
    * ``solver.Ds, solver.tDs``: diagonal primal and dual edge-length matrices
      (containing the open edge lengths ``L_open`` when ``conformal="pec"``);
    * ``solver.iDa, solver.itDa``: inverse primal and dual face-area operators
      (for ``conformal="pec"``, ``solver.iDa`` is the symmetric positive
      semi-definite Dey-Mittra stabilized face-redistribution matrix assembled
      by `wakis.conformal.apply_conformal_materials`);
    * ``solver.ieps, solver.imu``: per-edge inverse permittivity and per-face
      inverse permeability `Field` objects, zeroed on edges/faces inside PEC.

    In terms of the electric flux ``d`` on the active (non-PEC) edges where
    ``w_eps = diag(Ds) * ieps * diag(itDa) > 0`` and ``diag(C^T W_nu C) > 0``
    with ``W_nu = 1/2 (tDs iDa imu + (tDs iDa imu)^T)``, the semi-discrete
    leapfrog curl-curl eigenvalue problem is

    .. math::

        (W_{\\varepsilon, u} C_u^T W_\\nu C_u W_{\\varepsilon, u}
         + G_u D_{\\text{node}} G_u^T) d = \\omega_h^2 W_{\\varepsilon, u} d,

    where ``G_u`` contains the columns of ``G[active, :]`` in ``ker(W_nu C_u)``
    and ``D_node`` Jacobi-preconditions the node Laplacian
    ``G_u^T W_eps_u^{-1} G_u`` to shift static irrotational modes above the
    cavity spectrum.

    Parameters
    ----------
    geometry : {'perturbed_box', 'cylinder', 'sphere'}
        Benchmark cavity geometry.
    n : int or tuple of 3 ints
        Number of Cartesian cells per axis on the bounding box.
    conformal : bool or str, optional
        ``False`` for `"SolverFIT3D staircase"` or ``"pec"`` for
        `"SolverFIT3D conformal"`.
    a, b, d : float, optional
        Box side lengths ``(a, b, d)`` [m] or cylinder/sphere radius ``a`` [m].
    length : float, optional
        Cylinder axial length ``L`` [m] (default 1.2).

    Returns
    -------
    dict
        Dictionary with keys ``method``, ``n_cells``, ``n_unknowns``,
        ``frequencies`` (mapping mode name to ``omega_discrete_rad_s``), and
        ``wall_s``.
    """
    if isinstance(n, int):
        nx = ny = nz = int(n)
    else:
        nx, ny, nz = (int(x) for x in n)

    t0 = time.perf_counter()
    if geometry == "perturbed_box":
        grid = GridFIT3D(
            0.0,
            a,
            0.0,
            b,
            0.0,
            d,
            nx,
            ny,
            nz,
            conformal=conformal,
            boundary_at_domain_edge=True,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            use_stl=False,
            verbose=0,
        )
        modes = [("TE101", box_te101_omega(a=a, d=d))]
    elif geometry == "cylinder":
        cyl = pv.Cylinder(
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
            stl_solids={"cavity": cyl},
            stl_materials={"cavity": "vacuum"},
            conformal=conformal,
            boundary_at_domain_edge=True,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            bg="pec",
            use_stl=True,
            verbose=0,
        )
        modes = [
            ("TM010", cylinder_tm010_omega(a=a)),
            ("TE111", cylinder_te111_omega(a=a, length=length)),
        ]
    elif geometry == "sphere":
        sph = pv.Sphere(
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
            stl_solids={"cavity": sph},
            stl_materials={"cavity": "vacuum"},
            conformal=conformal,
            boundary_at_domain_edge=True,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            bg="pec",
            use_stl=True,
            verbose=0,
        )
        w1, w2 = sphere_lowest_omegas(a=a, l=1)
        modes = [("TE_l1", w1), ("TM_l1", w2)]
    else:
        raise ValueError(f"unknown geometry {geometry!r}")

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
    W_eps_u = sp.diags(w_eps_u, format="csr")
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

    if G_u.shape[1] > 0:
        L_diag = np.asarray((G_u.multiply(G_u)).T @ (1.0 / w_eps_u)).ravel()
        h_min = min(
            float(np.min(grid.dx)),
            float(np.min(grid.dy)),
            float(np.min(grid.dz)),
        )
        n_max = max(nx, ny, nz)
        omega_target_sq = (4.0 * n_max * C0 / h_min) ** 2
        D_node = sp.diags(omega_target_sq / L_diag)
        H = (A + (G_u @ D_node @ G_u.T)).tocsr()
    else:
        H = A

    freqs = {}
    k_eig = min(24, max(6, n_unknowns - 2))
    sigma_factor = 0.96 if conformal else 0.85
    for mode_name, w_ana in modes:
        E_ana = _analytic_edge_field_cartesian(
            grid, geometry, mode_name, a=a, b=b, d=d, length=length
        )[active]
        d_ana = (d4[active] / w_eps_u) * E_ana
        norm_d = float(np.sqrt(max(float(d_ana @ (W_eps_u @ d_ana)), 1e-300)))
        d_ana = d_ana / norm_d
        sigma = (sigma_factor * w_ana) ** 2
        evals, vecs = eigsh(
            H,
            k=k_eig,
            M=W_eps_u,
            sigma=sigma,
            which="LM",
            v0=d_ana,
        )
        overlaps = np.abs(vecs.T @ (w_eps_u * d_ana))
        best_j = int(np.argmax(overlaps))
        freqs[mode_name] = float(np.sqrt(max(evals[best_j].real, 0.0)))

    wall_s = time.perf_counter() - t0
    method_name = "SolverFIT3D conformal" if conformal else "SolverFIT3D staircase"
    return {
        "method": method_name,
        "n_cells": int(nx * ny * nz),
        "n_unknowns": n_unknowns,
        "frequencies": freqs,
        "wall_s": float(wall_s),
    }


def _select_sphere_tm_l1(omegas):
    """Return the ``j_1(k a) = 0`` (``TM_l1``) angular frequency ``omegas[8]``
    from the sorted spectrum ``omegas`` of a cube-to-ball `HexBlock3D`."""
    return float(omegas[8])


def curvilinear_eigenfrequencies(
    geometry,
    n,
    a=1.0,
    b=0.7,
    d=0.8,
    amp=0.3,
    length=1.2,
):
    """
    Compute discrete cavity eigenfrequencies, unknown edge count, minimum
    corner Jacobian, and minimum normalised corner determinant for
    `CurvilinearFIT3D` on ``geometry`` at refinement ``n``.
    """
    if isinstance(n, int):
        nx = ny = nz = int(n)
    else:
        nx, ny, nz = (int(x) for x in n)

    t0 = time.perf_counter()
    if geometry == "perturbed_box":
        X = perturbed_box_nodes(nx, ny, nz, a=a, b=b, d=d, amp=amp)
        block = HexBlock3D(X)
        solver = CurvilinearFIT3D(block)
        w = solver.cavity_frequencies(k=2)
        freqs = {"TE101": float(w[0])}
        analytic = {"TE101": box_te101_omega(a=a, d=d)}
    elif geometry == "cylinder":
        X = cylinder_nodes(nx, ny, nz, a=a, length=length)
        block = HexBlock3D(X)
        solver = CurvilinearFIT3D(block)
        w = solver.cavity_frequencies(k=4)
        freqs = {"TM010": float(w[0]), "TE111": float(w[1])}
        analytic = {
            "TM010": cylinder_tm010_omega(a=a),
            "TE111": cylinder_te111_omega(a=a, length=length),
        }
    elif geometry == "sphere":
        X = sphere_nodes(nx, ny, nz, a=a)
        block = HexBlock3D(X)
        solver = CurvilinearFIT3D(block)
        w = solver.cavity_frequencies(k=11)
        w1, w2 = sphere_lowest_omegas(a=a, l=1)
        freqs = {
            "TE_l1": float(w[0]),
            "TM_l1": _select_sphere_tm_l1(w),
        }
        analytic = {"TE_l1": w1, "TM_l1": w2}
    else:
        raise ValueError(f"unknown geometry {geometry!r}")

    wall_s = time.perf_counter() - t0
    return {
        "method": "CurvilinearFIT3D",
        "n_cells": int(block.n_cells),
        "n_unknowns": int(block.n_unknown_edges),
        "min_corner_jacobian_m3": float(np.min(block.corner_jacobians)),
        "min_corner_det": min_normalised_corner_det(block),
        "frequencies": freqs,
        "analytic": analytic,
        "wall_s": float(wall_s),
    }


def run_study(
    refinements=(6, 8, 12, 16),
    geometries=ALL_GEOMETRIES,
    methods=ALL_METHODS,
    a=1.0,
    b=0.7,
    d=0.8,
    amp=0.3,
    length=1.2,
    cmdline=None,
):
    """
    Execute the cavity convergence study across ``geometries``, ``methods``,
    and ``refinements`` and return the structured JSON-serializable dictionary.
    """
    if cmdline is None:
        cmdline = " ".join(sys.argv)
    refinements = [int(n) for n in refinements]

    raw_records = []
    for geom in geometries:
        # Cache hex and staircase unknown counts per N for equal-unknowns reporting
        hex_cache = {}
        stair_cache = {}
        for n in refinements:
            if "CurvilinearFIT3D" in methods:
                hex_cache[n] = curvilinear_eigenfrequencies(
                    geom, n, a=a, b=b, d=d, amp=amp, length=length
                )
            else:
                X = (
                    perturbed_box_nodes(n, n, n, a=a, b=b, d=d, amp=amp)
                    if geom == "perturbed_box"
                    else (
                        cylinder_nodes(n, n, n, a=a, length=length)
                        if geom == "cylinder"
                        else sphere_nodes(n, n, n, a=a)
                    )
                )
                blk = HexBlock3D(X)
                hex_cache[n] = {
                    "n_cells": int(blk.n_cells),
                    "n_unknowns": int(blk.n_unknown_edges),
                    "min_corner_jacobian_m3": float(np.min(blk.corner_jacobians)),
                    "min_corner_det": min_normalised_corner_det(blk),
                }

            if "SolverFIT3D staircase" in methods:
                stair_cache[n] = baseline_cartesian_eigenfrequencies(
                    geom, n, conformal=False, a=a, b=b, d=d, length=length
                )
            else:
                stair_cache[n] = baseline_cartesian_eigenfrequencies(
                    geom, n, conformal=False, a=a, b=b, d=d, length=length
                )

            for method in methods:
                if method == "CurvilinearFIT3D":
                    res = hex_cache[n]
                    analytic = res["analytic"]
                    min_det = res["min_corner_det"]
                    min_jac = res["min_corner_jacobian_m3"]
                elif method == "SolverFIT3D staircase":
                    res = stair_cache[n]
                    analytic = _analytic_dict(geom, a=a, d=d, length=length)
                    min_det = 1.0
                    min_jac = float(
                        ((a if geom == "perturbed_box" else 2.0 * a) / n)
                        * ((b if geom == "perturbed_box" else 2.0 * a) / n)
                        * (
                            (
                                d
                                if geom == "perturbed_box"
                                else (length if geom == "cylinder" else 2.0 * a)
                            )
                            / n
                        )
                    )
                elif method == "SolverFIT3D conformal":
                    res = baseline_cartesian_eigenfrequencies(
                        geom, n, conformal="pec", a=a, b=b, d=d, length=length
                    )
                    analytic = _analytic_dict(geom, a=a, d=d, length=length)
                    min_det = 1.0
                    min_jac = float(
                        ((a if geom == "perturbed_box" else 2.0 * a) / n)
                        * ((b if geom == "perturbed_box" else 2.0 * a) / n)
                        * (
                            (
                                d
                                if geom == "perturbed_box"
                                else (length if geom == "cylinder" else 2.0 * a)
                            )
                            / n
                        )
                    )
                else:
                    raise ValueError(f"unknown method {method!r}")

                for mode_name, w_ana in analytic.items():
                    w_disc = float(res["frequencies"][mode_name])
                    rel_err = abs(w_disc - w_ana) / w_ana
                    raw_records.append(
                        {
                            "geometry": geom,
                            "method": method,
                            "n_per_axis": int(n),
                            "n_cells": int(res["n_cells"]),
                            "n_unknowns": int(res["n_unknowns"]),
                            "n_unknowns_hex": int(hex_cache[n]["n_unknowns"]),
                            "n_unknowns_staircase": int(stair_cache[n]["n_unknowns"]),
                            "mode": mode_name,
                            "omega_analytic_rad_s": float(w_ana),
                            "omega_discrete_rad_s": w_disc,
                            "rel_error": float(rel_err),
                            "wall_s": float(res["wall_s"]),
                            "min_corner_det": float(min_det),
                            "min_corner_jacobian_m3": float(min_jac),
                            "order": None,
                            "command_line": cmdline,
                        }
                    )

    # Compute fitted order per (geometry, method, mode)
    orders = {}
    groups = {}
    for rec in raw_records:
        key = (rec["geometry"], rec["method"], rec["mode"])
        groups.setdefault(key, []).append(rec)

    for key, recs in groups.items():
        recs_sorted = sorted(recs, key=lambda r: r["n_per_axis"])
        order_val = _fit_order(
            [r["n_per_axis"] for r in recs_sorted],
            [r["rel_error"] for r in recs_sorted],
        )
        orders[f"{key[0]}|{key[1]}|{key[2]}"] = order_val
        for r in recs:
            r["order"] = order_val

    return {
        "command_line": cmdline,
        "refinements": refinements,
        "geometries": list(geometries),
        "methods": list(methods),
        "orders": orders,
        "records": raw_records,
    }


def _analytic_dict(geometry, a=1.0, d=0.8, length=1.2):
    if geometry == "perturbed_box":
        return {"TE101": box_te101_omega(a=a, d=d)}
    if geometry == "cylinder":
        return {
            "TM010": cylinder_tm010_omega(a=a),
            "TE111": cylinder_te111_omega(a=a, length=length),
        }
    if geometry == "sphere":
        w1, w2 = sphere_lowest_omegas(a=a, l=1)
        return {"TE_l1": w1, "TM_l1": w2}
    raise ValueError(f"unknown geometry {geometry!r}")


def format_summary_table(study_data):
    """Format a human-readable summary table from ``study_data``."""
    lines = [
        f"# Command: {study_data['command_line']}",
        (
            f"{'Geometry':<14} {'Method':<22} {'Mode':<8} {'N':>3} "
            f"{'n_cells':>8} {'n_unk':>8} {'omega_ana [rad/s]':>18} "
            f"{'omega_h [rad/s]':>18} {'rel_error':>11} {'min_det':>10} "
            f"{'wall_s':>8} {'order':>7}"
        ),
        "-" * 146,
    ]
    for r in study_data["records"]:
        ord_str = f"{r['order']:.3f}" if r["order"] is not None else "    n/a"
        lines.append(
            f"{r['geometry']:<14} {r['method']:<22} {r['mode']:<8} "
            f"{r['n_per_axis']:>3d} {r['n_cells']:>8d} {r['n_unknowns']:>8d} "
            f"{r['omega_analytic_rad_s']:>18.8e} {r['omega_discrete_rad_s']:>18.8e} "
            f"{r['rel_error']:>11.4e} {r['min_corner_det']:>10.4e} "
            f"{r['wall_s']:>8.3f} {ord_str:>7}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="3D curvilinear hexahedral cavity convergence study driver."
    )
    parser.add_argument(
        "--refinements",
        type=int,
        nargs="+",
        default=[6, 8, 12, 16],
        help="Cells per logical axis N (default: 6 8 12 16).",
    )
    parser.add_argument(
        "--geometries",
        type=str,
        nargs="+",
        default=list(ALL_GEOMETRIES),
        choices=list(ALL_GEOMETRIES),
        help="Geometries to run.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(ALL_METHODS),
        choices=list(ALL_METHODS),
        help="Methods to compare.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON file path (if omitted, JSON is written to stdout).",
    )
    args = parser.parse_args(argv)
    cmdline = " ".join(sys.argv if argv is None else [sys.argv[0], *argv])

    data = run_study(
        refinements=args.refinements,
        geometries=args.geometries,
        methods=args.methods,
        cmdline=cmdline,
    )
    table = format_summary_table(data)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(table)
    else:
        print(table, file=sys.stderr)
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
