# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Accuracy of 3D modal waveguide ports on general (non-extruded) tetrahedral meshes.

Measures the port reflection floor |S11| and the transmission phase error
|arg(S21 * exp(i * beta * L))| on general 3D Delaunay tetrahedral meshes
with conforming port faces (built with in-tree helpers and
``scipy.spatial.Delaunay``) and compares them at equal cells per
wavelength (cpw) against the extruded baseline (``extrude_section``) where
the discrete 3D FIT equations separate into the exact 1D ladder network
(``ExtrudedLine``).

Cases
-----
1. Straight WR-90 rectangular waveguide (a = 22.86 mm, b = 10.16 mm,
   L = 20.0 mm), fundamental TE10 mode across 8-12 GHz at 4 resolutions
   h = a / {4, 6, 8, 12}.
2. Straight 50-Ohm coaxial transmission line (r_in = 2.0 mm, r_out = 4.6 mm,
   L = 6.0 mm), fundamental TEM mode across 8-12 GHz at 4 resolutions
   h = (r_out - r_in) / {2, 3, 4, 5}.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.ticker
import numpy as np
import scipy.sparse as sp
from scipy.constants import c, epsilon_0, mu_0
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay

from wakis import figstyle
from wakis.unstructured.improve import improve_dt
from wakis.unstructured.mesh import SimplexMesh
from wakis.unstructured.meshers import _ring
from wakis.unstructured.pml3d import _snap_degenerate_duals
from wakis.unstructured.ports3d import (
    ExtrudedLine,
    ModalPort,
    PortPlane,
    attach_ports,
    coax_section,
    extrude_section,
    rectangle_section,
    s_matrix,
)
from wakis.unstructured.solver3d import UnstructuredFIT3D

ETA0 = float(np.sqrt(mu_0 / epsilon_0))
WR90_A = 22.86e-3
WR90_B = 10.16e-3
WR90_L = 20.0e-3
COAX_R_IN = 2.0e-3
COAX_R_OUT = 4.6e-3
COAX_L = 6.0e-3
FREQS_HZ = np.array([8.0e9, 10.0e9, 12.0e9])
F0_HZ = 10.0e9
LAMBDA0_M = float(c / F0_HZ)
WR90_H_DIVS = (4, 6, 8, 12)
COAX_NRS = (2, 3, 4, 5)


def add_pec_caps(mesh: SimplexMesh, length: float, tol: float = 1e-9) -> SimplexMesh:
    """Attach outward PEC cap tetrahedra to lateral boundary faces (0 < z < L).

    In ``UnstructuredFIT3D`` with ``bc='pmc'``, edges belonging to cells tagged
    as PEC (tag 0) are fixed to zero tangential electric field while contributing
    zero to the interior Voronoi dual areas and dual face lengths. Open port
    boundary faces at ``z = 0`` and ``z = L`` remain untagged and free.

    Parameters
    ----------
    mesh : SimplexMesh
        Tetrahedral mesh of the live waveguide interior (tag 1).
    length : float
        Guide length along z [m].
    tol : float, optional
        Relative tolerance for identifying the port planes ``z = 0`` and ``z = L``.

    Returns
    -------
    capped_mesh : SimplexMesh
        Mesh with lateral PEC cap tetrahedra (tag 0) appended to the live cells (tag 1).
    """
    bf = mesh.boundary_faces
    fc = mesh.nodes[mesh.faces[bf]].mean(axis=1)
    lat = (fc[:, 2] > tol * length) & (fc[:, 2] < (1.0 - tol) * length)
    lat_bf = bf[lat]
    lat_faces = mesh.faces[lat_bf]
    bcell = np.zeros(mesh.n_faces, dtype=int)
    bcell[mesh.cell_faces.ravel()] = np.repeat(np.arange(mesh.n_cells), 4)
    c_cen = mesh.nodes[mesh.cells[bcell[lat_bf]]].mean(axis=1)
    nrm = mesh.face_normal[lat_bf].copy()
    flip = np.einsum("ij,ij->i", nrm, fc[lat] - c_cen) < 0
    nrm[flip] *= -1.0
    le = np.sqrt(mesh.face_area[lat_bf])[:, None]
    apex = fc[lat] + 0.5 * le * nrm
    n0 = len(mesh.nodes)
    new_nodes = np.vstack([mesh.nodes, apex])
    cap_tets = np.column_stack([lat_faces, np.arange(n0, n0 + len(lat_bf))])
    new_tets = np.vstack([mesh.cells, cap_tets])
    tags = np.r_[
        np.ones(len(mesh.cells), dtype=int), np.zeros(len(cap_tets), dtype=int)
    ]
    return SimplexMesh(new_nodes, new_tets, tags)


def general_wr90_mesh(
    h_div: int, length: float = WR90_L, seed: int = 1
) -> tuple[SimplexMesh, np.ndarray]:
    """Build a general 3D Delaunay tetrahedral mesh of a straight WR-90 guide.

    Port faces at ``z = 0`` and ``z = length`` conform to 2D Delaunay sections
    from ``rectangle_section``, while interior nodes are staggered and perturbed
    in 3D ``(x, y, z)`` and tetrahedralized with ``scipy.spatial.Delaunay``.

    Parameters
    ----------
    h_div : int
        Subdivision count across the broad wall ``a`` (``h = a / h_div``) [-].
    length : float, optional
        Waveguide length ``L`` [m].
    seed : int, optional
        Random seed for reproducible interior node perturbation.

    Returns
    -------
    mesh : SimplexMesh
        3D tetrahedral mesh with tag 1 for the vacuum guide and tag 0 for PEC caps.
    zs : ndarray
        Nominal axial levels ``z`` [m] used for the reference ``ExtrudedLine``.
    """
    h = WR90_A / h_div
    rng = np.random.default_rng(seed)
    nodes2d, _, _ = rectangle_section(
        WR90_A, WR90_B, h, wall=None, jitter=0.15, seed=seed
    )
    nx, ny = int(np.ceil(WR90_A / h)), int(np.ceil(WR90_B / h))
    dx, dy = WR90_A / nx, WR90_B / ny
    nz = max(4, int(np.ceil(length / (h / 1.5))))
    zs = np.linspace(0.0, length, nz + 1)
    dz = zs[1] - zs[0]
    p0 = np.column_stack([nodes2d, np.zeros(len(nodes2d))])
    pL = np.column_stack([nodes2d, np.full(len(nodes2d), length)])
    on_perim = (
        (nodes2d[:, 0] == 0.0)
        | (nodes2d[:, 0] == WR90_A)
        | (nodes2d[:, 1] == 0.0)
        | (nodes2d[:, 1] == WR90_B)
    )
    perim2d = nodes2d[on_perim]
    wall_pts = []
    for k in range(1, nz):
        zk = zs[k]
        for x0, y0 in perim2d:
            corner = (x0 in (0.0, WR90_A)) and (y0 in (0.0, WR90_B))
            wx, wy = x0, y0
            wz = zk + (0.0 if k in (1, nz - 1) else 0.15 * dz * rng.uniform(-1.0, 1.0))
            if not corner:
                if x0 in (0.0, WR90_A):
                    wy = float(
                        np.clip(
                            y0
                            + ((k % 2) * 0.2 - 0.1 + 0.08 * rng.uniform(-1.0, 1.0))
                            * dy,
                            0.25 * dy,
                            WR90_B - 0.25 * dy,
                        )
                    )
                else:
                    wx = float(
                        np.clip(
                            x0
                            + ((k % 2) * 0.2 - 0.1 + 0.08 * rng.uniform(-1.0, 1.0))
                            * dx,
                            0.25 * dx,
                            WR90_A - 0.25 * dx,
                        )
                    )
            wall_pts.append([wx, wy, wz])
    wall_pts_arr = np.asarray(wall_pts, dtype=float)
    inner2d = nodes2d[~on_perim]
    in_pts = []
    for k in range(1, nz):
        zk = zs[k]
        pts_k = np.column_stack([inner2d, np.full(len(inner2d), zk)])
        pts_k[:, 0] += ((k % 2) * 0.2 - 0.1) * dx + 0.18 * dx * rng.uniform(
            -1.0, 1.0, len(inner2d)
        )
        pts_k[:, 1] += ((k % 2) * 0.2 - 0.1) * dy + 0.18 * dy * rng.uniform(
            -1.0, 1.0, len(inner2d)
        )
        if k not in (1, nz - 1):
            pts_k[:, 2] += 0.2 * dz * rng.uniform(-1.0, 1.0, len(inner2d))
        pts_k[:, 0] = np.clip(pts_k[:, 0], 0.35 * dx, WR90_A - 0.35 * dx)
        pts_k[:, 1] = np.clip(pts_k[:, 1], 0.35 * dy, WR90_B - 0.35 * dy)
        pts_k[:, 2] = np.clip(pts_k[:, 2], 0.65 * dz, length - 0.65 * dz)
        in_pts.append(pts_k)
    pts = np.vstack([p0, pL, wall_pts_arr, np.vstack(in_pts)])
    tet = Delaunay(pts).simplices
    P = pts[tet]
    vol = np.abs(np.linalg.det(P[:, 1:] - P[:, :1])) / 6.0
    tet = tet[vol > 1e-9 * h**3]
    m_live = SimplexMesh(pts, tet, np.ones(len(tet), dtype=int))
    m = add_pec_caps(m_live, length)
    m, _ = improve_dt(
        m,
        {0: "pec"},
        bc="pmc",
        n_sweeps=4,
        massless_tol=1e-4,
        project=None,
        seed=seed,
    )
    _snap_degenerate_duals(m)
    return m, zs


def extruded_wr90_mesh(
    h_div: int, length: float = WR90_L, seed: int = 1
) -> tuple[SimplexMesh, np.ndarray]:
    """Build the extruded baseline mesh of the straight WR-90 guide.

    Parameters
    ----------
    h_div : int
        Subdivision count across ``a`` (``h = a / h_div``) [-].
    length : float, optional
        Waveguide length ``L`` [m].
    seed : int, optional
        Random seed passed to ``rectangle_section``.

    Returns
    -------
    mesh : SimplexMesh
        Extruded tetrahedral mesh (3 tetrahedra per right triangular prism).
    zs : ndarray
        Axial z-levels [m].
    """
    h = WR90_A / h_div
    nodes, tris, tags = rectangle_section(WR90_A, WR90_B, h, wall=h / 2.0, seed=seed)
    nz = max(4, int(np.ceil(length / (h / 1.5))))
    zs = np.linspace(0.0, length, nz + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = extrude_section(nodes, tris, tags, zs)
    return m, zs


def general_coax_mesh(
    nr: int, length: float = COAX_L, seed: int = 42
) -> tuple[SimplexMesh, np.ndarray]:
    """Build a general 3D Delaunay tetrahedral mesh of a straight coaxial line.

    Port planes at ``z = 0`` and ``z = length`` use ``coax_section`` while the
    3D interior nodes are perturbed in ``(x, y, z)`` and tetrahedralized with
    ``scipy.spatial.Delaunay``.

    Parameters
    ----------
    nr : int
        Radial subdivision count across ``r_out - r_in`` (``h = (r_out - r_in) / nr``) [-].
    length : float, optional
        Coaxial line length ``L`` [m].
    seed : int, optional
        Random seed for interior node perturbation.

    Returns
    -------
    mesh : SimplexMesh
        3D tetrahedral mesh with tag 1 in the dielectric annulus and tag 0 on PEC caps.
    zs : ndarray
        Nominal axial levels ``z`` [m].
    """
    h = (COAX_R_OUT - COAX_R_IN) / nr
    rng = np.random.default_rng(seed)
    nodes2d, tris2d, tags2d = coax_section(COAX_R_IN, COAX_R_OUT, h, wall=None)
    ann_v = np.unique(tris2d[tags2d == 1])
    pts2d = nodes2d[ann_v]
    p0 = np.column_stack([pts2d, np.zeros(len(pts2d))])
    p1 = np.column_stack([pts2d, np.full(len(pts2d), length)])
    ring_out = _ring(COAX_R_OUT, h, preserve_area=True)
    ring_in = _ring(COAX_R_IN, h, preserve_area=True)
    r_out_eff = float(np.hypot(*ring_out[0]))
    r_in_eff = float(np.hypot(*ring_in[0]))
    r_pts2d = np.hypot(pts2d[:, 0], pts2d[:, 1])
    int2d = pts2d[(r_pts2d > r_in_eff + 1e-6) & (r_pts2d < r_out_eff - 1e-6)]
    nz = max(4, round(length / (h / 1.5)))
    zs = np.linspace(0.0, length, nz + 1)
    dz = zs[1] - zs[0]
    wall_pts, int_pts = [], []
    for k in range(1, nz):
        zk = zs[k]
        jz_out = (
            0.0
            if k in (1, nz - 1)
            else 0.08 * dz * rng.uniform(-1.0, 1.0, len(ring_out))
        )
        jz_in = (
            0.0
            if k in (1, nz - 1)
            else 0.08 * dz * rng.uniform(-1.0, 1.0, len(ring_in))
        )
        wall_pts.append(
            np.column_stack([ring_out, np.full(len(ring_out), zk) + jz_out])
        )
        wall_pts.append(np.column_stack([ring_in, np.full(len(ring_in), zk) + jz_in]))
        xy = int2d.copy()
        xy[:, 0] += 0.10 * h * rng.uniform(-1.0, 1.0, len(xy))
        xy[:, 1] += 0.10 * h * rng.uniform(-1.0, 1.0, len(xy))
        r_xy = np.hypot(xy[:, 0], xy[:, 1])
        r_clamped = np.clip(r_xy, r_in_eff + 0.62 * h, r_out_eff - 0.62 * h)
        xy *= (r_clamped / np.maximum(r_xy, 1e-12))[:, None]
        jz = (0.04 if k in (1, nz - 1) else 0.12) * dz * rng.uniform(-1.0, 1.0, len(xy))
        int_pts.append(np.column_stack([xy, np.full(len(xy), zk) + jz]))
    pts = np.vstack([p0, p1] + wall_pts + int_pts)
    cells = Delaunay(pts).simplices.copy()
    v0 = pts[cells[:, 0]]
    vol = (
        np.einsum(
            "ij,ij->i",
            np.cross(pts[cells[:, 1]] - v0, pts[cells[:, 2]] - v0),
            pts[cells[:, 3]] - v0,
        )
        / 6.0
    )
    cells = cells[np.abs(vol) > 1e-18]
    cent = pts[cells].mean(axis=1)
    r_cent = np.hypot(cent[:, 0], cent[:, 1])
    r_verts = np.hypot(pts[cells, 0], pts[cells, 1])
    inside_hole = (
        (r_cent < r_in_eff * 0.995)
        | np.all(r_verts < r_in_eff + 1e-6, axis=1)
        | np.all(r_verts > r_out_eff - 1e-6, axis=1)
    )
    cells = cells[~inside_hole]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m_live = SimplexMesh(pts, cells, np.ones(len(cells), dtype=int))
        m = add_pec_caps(m_live, length)
        _snap_degenerate_duals(m)
    return m, zs


def extruded_coax_mesh(
    nr: int, length: float = COAX_L
) -> tuple[SimplexMesh, np.ndarray]:
    """Build the extruded baseline mesh of the straight coaxial line.

    Parameters
    ----------
    nr : int
        Radial subdivision count across ``r_out - r_in`` [-].
    length : float, optional
        Coaxial line length ``L`` [m].

    Returns
    -------
    mesh : SimplexMesh
        Extruded tetrahedral mesh of the coaxial line.
    zs : ndarray
        Axial z-levels [m].
    """
    h = (COAX_R_OUT - COAX_R_IN) / nr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nodes, tris, tags = coax_section(COAX_R_IN, COAX_R_OUT, h, wall=h)
        nz = max(4, round(length / (h / 1.5)))
        zs = np.linspace(0.0, length, nz + 1)
        m = extrude_section(nodes, tris, tags, zs)
    return m, zs


def fd_two_port_voltages(ps, freqs: np.ndarray) -> np.ndarray:
    """Solve the discrete-time frequency response for both port excitations at once.

    Evaluates the exact discrete-time system of ``PortSet.frequency_response``
    at each frequency in ``freqs``, reusing one sparse LU factorization per
    frequency across the two unit-current RHS columns ``I_s = [1, 0]`` and
    ``I_s = [0, 1]``.

    Parameters
    ----------
    ps : PortSet
        Attached two-port object on ``UnstructuredFIT3D``.
    freqs : ndarray
        Frequencies [Hz].

    Returns
    -------
    V : ndarray, shape (n_freqs, 2, 2)
        Port voltage phasors ``V[f_idx, run_idx, port_idx]`` [V].
    """
    s = ps.solver
    dt = ps.dt
    C0, G = s.cap, s.cond
    nf, ns = s.free.size, s.massless.size
    Cfull = sp.hstack([s.Cf, s.Cs]).tocsr()
    K = (Cfull.T @ s.Mnu @ Cfull).tocsc()
    WgW = ps.W @ sp.diags(ps.g) @ ps.W.T
    rhs = np.zeros((nf + ns, 2), dtype=complex)
    rhs[:nf, :] = ps.W.toarray()
    out = []
    for f in np.atleast_1d(freqs):
        w = 2.0 * np.pi * f
        wt = 2.0 * np.sin(w * dt / 2.0) / dt
        ch = np.cos(w * dt / 2.0)
        Dff = sp.diags(1j * wt * C0 + ch * G) + ch * WgW
        Z = sp.bmat([[Dff, None], [None, sp.csr_matrix((ns, ns))]]) if ns else Dff
        A_mat = (Z + K / (1j * wt)).tocsc()
        U = splu(A_mat).solve(rhs)[:nf, :]
        out.append(ch * (ps.W.T @ U).T)
    return np.asarray(out)


def evaluate_guide_case(
    guide: str,
    mesh_kind: str,
    res_param: int,
    freqs: np.ndarray = FREQS_HZ,
    port_face_scale: float = 1.0,
) -> dict:
    """Run the 2-port S-parameter evaluation for one waveguide configuration.

    Parameters
    ----------
    guide : {"wr90", "coax"}
        Waveguide geometry identifier.
    mesh_kind : {"general", "extruded"}
        Mesh construction type.
    res_param : int
        Resolution parameter (``h_div`` for WR-90, ``nr`` for coax).
    freqs : ndarray, optional
        Evaluation frequencies [Hz].
    port_face_scale : float, optional
        Multiplier applied to the 2D port-face Voronoi Hodge ``plane.M``
        (default 1.0; setting ``port_face_scale != 1.0`` is used in mutation
        testing to falsify the port-face dual area weight).

    Returns
    -------
    record : dict
        Measured resolution, cell counts, |S11| floor, S21 phase error, and
        auxiliary mode error metrics.
    """
    t0 = time.time()
    if guide == "wr90":
        h = WR90_A / res_param
        length = WR90_L
        if mesh_kind == "general":
            m, zs = general_wr90_mesh(res_param, length=length)
            mats = {0: "pec"}
            massless_tol = 1e-4
        else:
            m, zs = extruded_wr90_mesh(res_param, length=length)
            mats = {0: "pec"}
            massless_tol = 1e-6
    elif guide == "coax":
        h = (COAX_R_OUT - COAX_R_IN) / res_param
        length = COAX_L
        if mesh_kind == "general":
            m, zs = general_coax_mesh(res_param, length=length)
            mats = {0: "pec"}
            massless_tol = 1e-4
        else:
            m, zs = extruded_coax_mesh(res_param, length=length)
            mats = {0: "pec", 2: "pec"}
            massless_tol = 1e-6
    else:
        raise ValueError(f"unknown guide {guide!r}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=massless_tol)
        planes = [
            PortPlane(s, (0.0, 0.0, zz), (0.0, 0.0, 1.0)) for zz in (zs[0], zs[-1])
        ]
        if guide == "wr90":
            kc0, phi0 = planes[0].te_modes(1)
            _, phi1 = planes[1].te_modes(1)
            kc_val = float(kc0[0])
            phis = [phi0[:, 0], phi1[:, 0]]
            mode_rel_err = float(abs(kc_val / (np.pi / WR90_A) - 1.0))
            beta = np.sqrt((2.0 * np.pi * freqs / c) ** 2 - (np.pi / WR90_A) ** 2)
        else:
            tm0 = planes[0].tem_mode()
            tm1 = planes[1].tem_mode()
            kc_val = 0.0
            phis = [tm0[0], tm1[0]]
            v_scale = float(tm0[1])
            z0_exact = np.log(COAX_R_OUT / COAX_R_IN) / (2.0 * np.pi)
            mode_rel_err = float(abs(v_scale**2 / z0_exact - 1.0))
            beta = 2.0 * np.pi * freqs / c

        if port_face_scale != 1.0:
            for p in planes:
                p.M = p.M * float(port_face_scale)

        line = ExtrudedLine(zs, 1.0, 1.0, kc_val, s.dt)

        def y_ref(f, line=line):
            return np.real(line.port_admittance(f) / line.cos_half(f))

        ports = [ModalPort(planes[k], phis[k], y_ref, f0=F0_HZ) for k in range(2)]
        ps = attach_ports(s, ports)
        V = fd_two_port_voltages(ps, freqs)
        Is = np.array([[1.0, 0.0], [0.0, 1.0]])[None, :, :]
        I = Is - ps.g[None, None, :] * V
        Yr = np.stack([y_ref(freqs), y_ref(freqs)], axis=1)
        S = s_matrix(V, I, Yr)

    phase_err_vec = np.abs(np.angle(S[:, 1, 0] * np.exp(1j * beta * length)))
    s11_vec = np.abs(S[:, 0, 0])
    f0_idx = int(np.argmin(np.abs(freqs - F0_HZ)))
    n_live_tets = int(np.sum(m.cell_tags == 1))
    cpw = float(LAMBDA0_M / h)

    rec = {
        "guide": guide,
        "mesh_kind": mesh_kind,
        "res_param": int(res_param),
        "h_m": float(h),
        "h_over_lambda0": float(h / LAMBDA0_M),
        "cpw": cpw,
        "length_m": float(length),
        "n_live_tets": n_live_tets,
        "n_total_tets": int(m.n_cells),
        "n_free_edges": int(s.free.size),
        "dt_s": float(s.dt),
        "s11_floor": float(s11_vec.max()),
        "s11_f0": float(s11_vec[f0_idx]),
        "s21_phase_err_rad": float(phase_err_vec.max()),
        "s21_phase_err_f0_rad": float(phase_err_vec[f0_idx]),
        "mode_rel_err": mode_rel_err,
        "wall_s": float(time.time() - t0),
    }
    del m, s, ps, ports, planes
    gc.collect()
    return rec


def fit_convergence_order(h_vals: list[float], err_vals: list[float]) -> dict:
    """Fit the log-log convergence order ``p`` in ``err ~ C * h^p``.

    Parameters
    ----------
    h_vals : list of float
        Mesh sizes ``h`` [m].
    err_vals : list of float
        Positive error quantities at each ``h``.

    Returns
    -------
    fit : dict
        Dictionary with ``order_p`` (slope vs ``log(h)``), ``intercept``,
        ``log_rms_residual`` (RMS residual of ``log(err)`` in nepers), and
        ``coarse2_order_p`` (2-point slope from the two coarsest meshes).
    """
    lh = np.log(np.asarray(h_vals, dtype=float))
    le = np.log(np.asarray(err_vals, dtype=float))
    slope, intercept = np.polyfit(lh, le, 1)
    pred = slope * lh + intercept
    rms_res = float(np.sqrt(np.mean((le - pred) ** 2)))
    coarse2_slope = float((le[1] - le[0]) / (lh[1] - lh[0]))
    delta_p = float(2.0 * np.sqrt(2.0) * rms_res / abs(lh[1] - lh[0]))
    return {
        "order_p": float(slope),
        "intercept": float(intercept),
        "log_rms_residual": rms_res,
        "coarse2_order_p": coarse2_slope,
        "delta_p_2level": delta_p,
        "theory_order_p": 2.0,
        "theory_min_order_p": float(2.0 - delta_p),
        "study_min_order_p": float(slope - delta_p),
    }


def make_study_figures(data: dict, figures_dir: str) -> list[str]:
    """Render the |S11| floor and S21 phase error convergence figures.

    Parameters
    ----------
    data : dict
        Full study JSON dictionary produced by ``run_study``.
    figures_dir : str
        Output directory for the PNG figures.

    Returns
    -------
    paths : list of str
        Written figure file paths.
    """
    os.makedirs(figures_dir, exist_ok=True)
    wg_gen = data["cases"]["wr90_general"]["records"]
    wg_ext = data["cases"]["wr90_extruded"]["records"]
    cx_gen = data["cases"]["coax_general"]["records"]
    cx_ext = data["cases"]["coax_extruded"]["records"]
    fits = data["fits"]

    paths = []
    with figstyle.apply_style():
        # Figure 1: |S11| floor vs h / lambda0 (and cpw)
        fig1, ax1 = figstyle.new_figure(7.0, 3.4, ncols=2)
        for idx, (title, gen_rows, ext_rows, fit_key) in enumerate(
            [
                (
                    "(a) WR-90 TE$_{10}$ ($L$ = 20 mm)",
                    wg_gen,
                    wg_ext,
                    "wr90_general_s11",
                ),
                ("(b) Coax TEM ($L$ = 6 mm)", cx_gen, cx_ext, "coax_general_s11"),
            ]
        ):
            a = ax1[idx]
            cpw_g = np.array([r["cpw"] for r in gen_rows])
            s11_g = np.array([r["s11_floor"] for r in gen_rows])
            cpw_e = np.array([r["cpw"] for r in ext_rows])
            s11_e = np.array([r["s11_floor"] for r in ext_rows])
            p_val = fits[fit_key]["order_p"]

            st0 = figstyle.series_style(0)
            st1 = figstyle.series_style(1)
            a.loglog(
                cpw_g,
                s11_g,
                color=st0["color"],
                ls="-",
                marker="o",
                ms=4.5,
                label=f"general tet ($p$ = {p_val:.2f})",
            )
            a.loglog(
                cpw_e,
                s11_e,
                color=st1["color"],
                ls="--",
                marker="s",
                ms=4.0,
                label="extruded baseline",
            )
            ref_y = s11_g[0] * (cpw_g / cpw_g[0]) ** (-2.0)
            a.loglog(
                cpw_g,
                ref_y,
                "k:",
                lw=1.0,
                label=r"$\mathcal{O}(\mathrm{cpw}^{-2})$ guide",
            )
            a.set(
                xlabel=r"cells per wavelength $\lambda_0 / h$ (10 GHz)",
                ylabel=r"matched-line $|S_{11}|$ floor",
                title=title,
                ylim=(1e-16, 1e-1),
            )
            figstyle.spectrum_grid(a)
            a.set_xticks([round(x, 1) for x in cpw_g])
            a.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            a.legend(fontsize=6.5, loc="center right", frameon=False)

        p1 = os.path.join(figures_dir, "general_mesh_s11_floor.png")
        figstyle.save(fig1, p1, close=True)
        paths.append(p1)

        # Figure 2: S21 phase error vs cpw with order guide lines
        fig2, ax2 = figstyle.new_figure(7.0, 3.4, ncols=2)
        for idx, (title, gen_rows, ext_rows, fit_g_key, fit_e_key) in enumerate(
            [
                (
                    "(a) WR-90 TE$_{10}$ phase error",
                    wg_gen,
                    wg_ext,
                    "wr90_general_phase",
                    "wr90_extruded_phase",
                ),
                (
                    "(b) Coax TEM phase error",
                    cx_gen,
                    cx_ext,
                    "coax_general_phase",
                    "coax_extruded_phase",
                ),
            ]
        ):
            a = ax2[idx]
            cpw_g = np.array([r["cpw"] for r in gen_rows])
            ph_g = np.array([r["s21_phase_err_rad"] for r in gen_rows])
            cpw_e = np.array([r["cpw"] for r in ext_rows])
            ph_e = np.array([r["s21_phase_err_rad"] for r in ext_rows])
            pg = fits[fit_g_key]["order_p"]
            pe = fits[fit_e_key]["order_p"]

            st0 = figstyle.series_style(0)
            st1 = figstyle.series_style(1)
            a.loglog(
                cpw_g,
                ph_g,
                color=st0["color"],
                ls="-",
                marker="o",
                ms=4.5,
                label=f"general tet ($p$ = {pg:.2f})",
            )
            a.loglog(
                cpw_e,
                ph_e,
                color=st1["color"],
                ls="--",
                marker="s",
                ms=4.0,
                label=f"extruded ($p$ = {pe:.2f})",
            )
            ref_y = ph_g[0] * 1.25 * (cpw_g / cpw_g[0]) ** (-2.0)
            a.loglog(
                cpw_g,
                ref_y,
                "k:",
                lw=1.0,
                label=r"$\mathcal{O}(\mathrm{cpw}^{-2})$ guide",
            )
            a.set(
                xlabel=r"cells per wavelength $\lambda_0 / h$ (10 GHz)",
                ylabel=r"$\max_f |\arg(S_{21} e^{i\beta L})|$ (rad)",
                title=title,
            )
            figstyle.spectrum_grid(a)
            a.set_xticks([round(x, 1) for x in cpw_g])
            a.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            a.legend(fontsize=6.5, loc="upper right", frameon=False)

        p2 = os.path.join(figures_dir, "general_mesh_s21_phase.png")
        figstyle.save(fig2, p2, close=True)
        paths.append(p2)

    return paths


def run_study() -> dict:
    """Execute the full WR-90 and coaxial general-vs-extruded accuracy study.

    Returns
    -------
    results : dict
        Complete study dictionary with records, fitted orders, residuals, and
        summary floors.
    """
    wg_gen = [evaluate_guide_case("wr90", "general", d) for d in WR90_H_DIVS]
    wg_ext = [evaluate_guide_case("wr90", "extruded", d) for d in WR90_H_DIVS]
    cx_gen = [evaluate_guide_case("coax", "general", nr) for nr in COAX_NRS]
    cx_ext = [evaluate_guide_case("coax", "extruded", nr) for nr in COAX_NRS]

    fits = {
        "wr90_general_s11": fit_convergence_order(
            [r["h_m"] for r in wg_gen], [r["s11_floor"] for r in wg_gen]
        ),
        "wr90_general_phase": fit_convergence_order(
            [r["h_m"] for r in wg_gen], [r["s21_phase_err_rad"] for r in wg_gen]
        ),
        "wr90_extruded_phase": fit_convergence_order(
            [r["h_m"] for r in wg_ext], [r["s21_phase_err_rad"] for r in wg_ext]
        ),
        "coax_general_s11": fit_convergence_order(
            [r["h_m"] for r in cx_gen], [r["s11_floor"] for r in cx_gen]
        ),
        "coax_general_phase": fit_convergence_order(
            [r["h_m"] for r in cx_gen], [r["s21_phase_err_rad"] for r in cx_gen]
        ),
        "coax_extruded_phase": fit_convergence_order(
            [r["h_m"] for r in cx_ext], [r["s21_phase_err_rad"] for r in cx_ext]
        ),
    }

    floors = {
        "wr90_general_s11_min": float(min(r["s11_floor"] for r in wg_gen)),
        "wr90_general_s11_max": float(max(r["s11_floor"] for r in wg_gen)),
        "wr90_extruded_s11_max": float(max(r["s11_floor"] for r in wg_ext)),
        "coax_general_s11_min": float(min(r["s11_floor"] for r in cx_gen)),
        "coax_general_s11_max": float(max(r["s11_floor"] for r in cx_gen)),
        "coax_extruded_s11_max": float(max(r["s11_floor"] for r in cx_ext)),
    }

    mut_wg = [
        evaluate_guide_case("wr90", "general", d, port_face_scale=0.5)
        for d in WR90_H_DIVS[:2]
    ]
    mut_fit = fit_convergence_order(
        [r["h_m"] for r in mut_wg], [r["s11_floor"] for r in mut_wg]
    )

    return {
        "parameters": {
            "f0_hz": F0_HZ,
            "lambda0_m": LAMBDA0_M,
            "freqs_hz": FREQS_HZ.tolist(),
            "wr90_a_m": WR90_A,
            "wr90_b_m": WR90_B,
            "wr90_length_m": WR90_L,
            "coax_r_in_m": COAX_R_IN,
            "coax_r_out_m": COAX_R_OUT,
            "coax_length_m": COAX_L,
            "baseline_name": "extrude_section",
        },
        "cases": {
            "wr90_general": {"records": wg_gen},
            "wr90_extruded": {"records": wg_ext},
            "coax_general": {"records": cx_gen},
            "coax_extruded": {"records": cx_ext},
        },
        "fits": fits,
        "floors": floors,
        "mutation": {
            "port_face_scale": 0.5,
            "wr90_general_records": mut_wg,
            "wr90_general_s11_fit": mut_fit,
        },
    }


def print_summary_table(data: dict) -> None:
    """Print a formatted summary table of all cases and fitted orders to stdout."""
    header = (
        f"{'Case':<16} {'Param':>5} {'h (mm)':>8} {'cpw':>7} {'Live tets':>10} "
        f"{'|S11| floor':>13} {'S21 ph err (rad)':>17} {'Mode rel err':>13}"
    )
    print(header)
    print("-" * len(header))
    for case_name, case_obj in data["cases"].items():
        for r in case_obj["records"]:
            print(
                f"{case_name:<16} {r['res_param']:>5d} {r['h_m'] * 1e3:>8.3f} "
                f"{r['cpw']:>7.2f} {r['n_live_tets']:>10d} "
                f"{r['s11_floor']:>13.4e} {r['s21_phase_err_rad']:>17.4e} "
                f"{r['mode_rel_err']:>13.4e}"
            )
    print("\nFitted log-log convergence orders (err ~ C * h^p):")
    fit_hdr = (
        f"{'Metric':<24} {'Order p':>10} {'Coarse-2 p':>12} {'Log RMS res':>13} "
        f"{'Delta p':>10} {'Theory min':>12} {'Study min':>11}"
    )
    print(fit_hdr)
    print("-" * len(fit_hdr))
    for k, v in data["fits"].items():
        print(
            f"{k:<24} {v['order_p']:>10.4f} {v['coarse2_order_p']:>12.4f} "
            f"{v['log_rms_residual']:>13.4f} {v['delta_p_2level']:>10.4f} "
            f"{v['theory_min_order_p']:>12.4f} {v['study_min_order_p']:>11.4f}"
        )
    if "mutation" in data:
        mut = data["mutation"]
        m_recs = mut["wr90_general_records"]
        m_fit = mut["wr90_general_s11_fit"]
        print(
            f"\nMutation (port_face_scale={mut['port_face_scale']}): "
            f"|S11| = [{m_recs[0]['s11_floor']:.4e}, {m_recs[1]['s11_floor']:.4e}], "
            f"coarse2_order_p = {m_fit['coarse2_order_p']:.4f}"
        )


def main() -> None:
    """CLI entry point for the general-mesh port accuracy study."""
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="out/general_mesh_accuracy.json",
        help="Output JSON path (default: out/general_mesh_accuracy.json).",
    )
    parser.add_argument(
        "--figures-dir",
        default=os.path.join(here, "figures"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the 4-resolution study sweep.",
    )
    args = parser.parse_args()

    data = run_study()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    study_out = os.path.join(here, "out", "general_mesh_accuracy.json")
    if os.path.abspath(args.out) != os.path.abspath(study_out):
        os.makedirs(os.path.dirname(study_out), exist_ok=True)
        with open(study_out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    make_study_figures(data, args.figures_dir)
    print_summary_table(data)


if __name__ == "__main__":
    main()
