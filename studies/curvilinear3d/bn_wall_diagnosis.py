"""
Brief W4-B Part 1: Diagnosis of the node-level ``|B_c . n_v| / |B_c|`` residual
on curved PEC boundaries (``0.1845750`` on ``sphere_nodes(6, 6, 6)``).

Evaluates per-class and overall boundary octant/node metrics across:
- ``sphere_nodes`` (single-block ``HexBlock3D``), ``N in {4, 6, 8, 12, 16}``
- ``cubed_sphere_ball_nodes`` (7-block ``HexMesh``), ``n in {4, 6, 8}``
- ``butterfly_disk_nodes`` (5-block ``HexMesh``), ``n in {4, 6, 8}``
- ``perturbed_box_nodes(6, 5, 4)`` (flat-wall control)

Usage::

    python studies/curvilinear3d/bn_wall_diagnosis.py [--quick] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from studies.curvilinear3d.perturbed_box_order import fit_order, local_orders
from wakis.curvilinear import CurvilinearFIT3D, HexBlock3D
from wakis.curvilinear.hexmesh import _CORNER_VTK, _LOCAL_FACE_VTK
from wakis.curvilinear.mappings3d import (
    butterfly_disk_nodes,
    cubed_sphere_ball_nodes,
    cylinder_te111_omega,
    cylinder_tm010_omega,
    perturbed_box_nodes,
    sphere_lowest_omegas,
    sphere_nodes,
)

NODE_CLASSES = (
    "smooth_wall",
    "smooth_seam",
    "sharp_smooth_seam",
    "physical_sharp_edge",
    "corner",
)

SPHERE_NS_FULL = (4, 6, 8, 12, 16)
SPHERE_NS_QUICK = (4, 6, 8, 12)
HEXMESH_NS_FULL = (4, 6, 8)
HEXMESH_NS_QUICK = (4, 6)


def _build_local_boundary_patch_grid(msh):
    """
    Reconstruct per-hex local boundary patch table of shape ``(n_cells, 3, 2)``
    with ``-1`` on interior local faces.
    """
    local_patch = np.full((msh.n_cells, 3, 2), -1, dtype=np.int64)
    local_patch[
        msh.boundary_face_hex,
        msh.boundary_face_axis,
        msh.boundary_face_side,
    ] = msh.boundary_patch
    return local_patch


def classify_boundary_nodes(msh, smooth_angle_threshold_rad=np.pi / 4.0):
    """
    Partition all boundary nodes (``boundary_node_mask == True``) of ``msh``
    into the five W4-B diagnostic node classes.

    Parameters
    ----------
    msh : HexMesh
        Assembled curvilinear hexahedral mesh.
    smooth_angle_threshold_rad : float, optional
        Angle threshold [rad] separating geometrically smooth block seams from
        physical sharp edges (default ``pi / 4``).

    Returns
    -------
    node_class : dict of int -> str
        Mapping from global node index ``v`` to class name in ``NODE_CLASSES``.
    node_max_facet_angle_deg : dict of int -> float
        Maximum pairwise angle [deg] between unit normals of boundary faces
        incident to node ``v``.
    """
    f_norm = np.linalg.norm(msh.face_areas, axis=1, keepdims=True)
    f_unit = msh.face_areas / np.maximum(f_norm, 1e-30)

    node_face_normals = {}
    node_patches = {}

    for i_bf in range(msh.n_boundary_faces):
        f = int(msh.boundary_faces[i_bf])
        h = int(msh.boundary_face_hex[i_bf])
        ax = int(msh.boundary_face_axis[i_bf])
        side = int(msh.boundary_face_side[i_bf])
        p = int(msh.boundary_patch[i_bf])
        s_out = (-1.0 if side == 0 else 1.0) * float(msh.hex_face_signs[h, ax, side])
        nf = s_out * f_unit[f]
        for c_loc in _LOCAL_FACE_VTK[ax, side]:
            v = int(msh.hexes[h, c_loc])
            node_face_normals.setdefault(v, []).append(nf)
            node_patches.setdefault(v, set()).add(p)

    node_class = {}
    node_max_facet_angle_deg = {}
    for v, n_list in node_face_normals.items():
        n_arr = np.asarray(n_list, dtype=float)
        dots = np.clip(n_arr @ n_arr.T, -1.0, 1.0)
        max_ang_rad = float(np.arccos(np.min(dots)))
        node_max_facet_angle_deg[v] = float(np.degrees(max_ang_rad))

        wc = int(msh.node_wall_count[v])
        is_sharp = bool(msh.effective_sharp_nodes[v])
        multi_patch = len(node_patches[v]) >= 2 or wc >= 2

        if wc >= 3 and is_sharp:
            cls = "corner"
        elif not is_sharp:
            cls = "smooth_seam" if multi_patch else "smooth_wall"
        else:
            if max_ang_rad < smooth_angle_threshold_rad:
                cls = "sharp_smooth_seam"
            else:
                cls = "physical_sharp_edge"
        node_class[v] = cls

    return node_class, node_max_facet_angle_deg


def exact_surface_normals(geometry_kind, coords, a=1.0, length=1.2):
    """
    Evaluate exact outward unit surface normals at boundary node coordinates.

    Parameters
    ----------
    geometry_kind : {"sphere", "cylinder", "box"}
        Surface family.
    coords : ndarray of shape (N, 3)
        Node coordinates [m].
    a : float, optional
        Sphere or cylinder radius [m] (default 1.0).
    length : float, optional
        Cylinder axial length [m] (default 1.2).

    Returns
    -------
    n_exact : ndarray of shape (N, 3)
        Exact outward unit normal vectors (dimensionless). At sharp edges or
        corners where the continuum normal is multi-valued, the normalized sum
        of the incident continuum surface normals is returned.
    """
    pts = np.asarray(coords, dtype=float)
    if geometry_kind == "sphere":
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / np.maximum(norms, 1e-30)

    if geometry_kind == "cylinder":
        xy = pts[:, :2]
        r_xy = np.linalg.norm(xy, axis=1, keepdims=True)
        n_rad = np.zeros_like(pts)
        n_rad[:, :2] = xy / np.maximum(r_xy, 1e-30)
        z = pts[:, 2]
        tol = 1e-7 * max(a, length)
        on_bot = np.abs(z) <= tol
        on_top = np.abs(z - length) <= tol
        on_lat = np.abs(r_xy[:, 0] - a) <= 1e-3 * a

        out = np.zeros_like(pts)
        for i in range(pts.shape[0]):
            vec = np.zeros(3, dtype=float)
            if on_lat[i]:
                vec += n_rad[i]
            if on_bot[i]:
                vec += np.array([0.0, 0.0, -1.0])
            if on_top[i]:
                vec += np.array([0.0, 0.0, 1.0])
            norm_v = np.linalg.norm(vec)
            out[i] = vec / norm_v if norm_v > 0.0 else n_rad[i]
        return out

    if geometry_kind == "box":
        return np.zeros_like(pts)

    raise ValueError(f"Unknown geometry_kind: {geometry_kind!r}")


def _sample_random_b(msh, seed=2026, zero_wall_b=True):
    """
    Sample the random face-flux vector ``b`` using the exact RNG draw sequence
    of ``_check_wall_octant_reconstructions`` in ``tests/test_112_curvilinear3d.py``.
    """
    rng = np.random.default_rng(seed)
    if msh._from_block_ref is not None and msh.block_face_perm is not None:
        blk = msh._from_block_ref
        _ = rng.uniform(0.5, 2.0, size=blk.n_unknown_edges) * (
            2 * rng.integers(0, 2, size=blk.n_unknown_edges) - 1
        )
        b_blk = rng.uniform(0.5, 2.0, size=blk.n_faces) * (
            2 * rng.integers(0, 2, size=blk.n_faces) - 1
        )
        if zero_wall_b:
            for ax in range(3):
                fg = blk.face_grid(ax)
                on_wall = (fg[ax] == 0) | (fg[ax] == blk.nc[ax])
                b_blk[blk.face_index(ax, *fg)[on_wall]] = 0.0
        b = np.empty(msh.n_faces, dtype=float)
        b[msh.block_face_perm] = b_blk
        return b

    _ = rng.uniform(0.5, 2.0, size=msh.n_unknown_edges) * (
        2 * rng.integers(0, 2, size=msh.n_unknown_edges) - 1
    )
    b = rng.uniform(0.5, 2.0, size=msh.n_faces) * (
        2 * rng.integers(0, 2, size=msh.n_faces) - 1
    )
    if zero_wall_b:
        b[msh.boundary_faces] = 0.0
    return b


def evaluate_wall_octant_metrics(
    sol,
    node_class,
    seed=2026,
    zero_wall_b=True,
    transpose_rc=False,
):
    """
    Evaluate octant-level face normal ratio ``m0``, node normal ratio ``m1``,
    and facet-to-node normal angle bound ``g = |n_v x n_f|`` over all boundary
    octants.

    Parameters
    ----------
    sol : CurvilinearFIT3D
        Assembled curvilinear solver.
    node_class : dict of int -> str
        Boundary node classification from :func:`classify_boundary_nodes`.
    seed : int, optional
        Random number generator seed matching ``test_pec_constraint_wall_octants``
        (default 2026).
    zero_wall_b : bool, optional
        If True (default), zero ``b`` on PEC boundary faces. Setting False is
        Mutation (a).
    transpose_rc : bool, optional
        If False (default), reconstruct ``B_c = R_c b_c``. Setting True uses
        ``R_c^T b_c`` (Mutation (b)).

    Returns
    -------
    dict
        Per-class and overall octant statistics, including locations of maxima.
    """
    msh = sol.mesh
    b = _sample_random_b(msh, seed=seed, zero_wall_b=zero_wall_b)
    local_patch = _build_local_boundary_patch_grid(msh)

    f_norm = np.linalg.norm(msh.face_areas, axis=1, keepdims=True)
    f_unit = msh.face_areas / np.maximum(f_norm, 1e-30)

    records_by_class = {cls: [] for cls in NODE_CLASSES}
    all_one_wall_records = []
    all_boundary_records = []

    for a in (0, 1):
        for b_idx in (0, 1):
            for c in (0, 1):
                c_loc = int(_CORNER_VTK[a, b_idx, c])
                R_all = sol._R_c[a, b_idx, c]
                for h in range(msh.n_cells):
                    v = int(msh.hexes[h, c_loc])
                    if v not in node_class:
                        continue
                    sides = (a, b_idx, c)
                    wall_axes = [
                        ax for ax in range(3) if local_patch[h, ax, sides[ax]] >= 0
                    ]
                    n_walls = len(wall_axes)
                    if n_walls == 0:
                        continue

                    cls = node_class[v]
                    f_trip = np.array(
                        [
                            msh.hex_faces[h, 0, a],
                            msh.hex_faces[h, 1, b_idx],
                            msh.hex_faces[h, 2, c],
                        ],
                        dtype=np.int64,
                    )
                    b_c = b[f_trip]
                    rc_mat = R_all[h].T if transpose_rc else R_all[h]
                    B_c = rc_mat @ b_c
                    B_norm = float(np.linalg.norm(B_c))
                    n_v = msh.node_normals[v]
                    coord = [float(x) for x in msh.nodes[v]]

                    m0_vals = []
                    g_vals = []
                    for ax_w in wall_axes:
                        side_w = sides[ax_w]
                        f_w = int(msh.hex_faces[h, ax_w, side_w])
                        s_out = (-1.0 if side_w == 0 else 1.0) * float(
                            msh.hex_face_signs[h, ax_w, side_w]
                        )
                        n_f = s_out * f_unit[f_w]
                        if B_norm > 1e-30:
                            m0_vals.append(float(abs(np.dot(B_c, n_f)) / B_norm))
                        else:
                            m0_vals.append(0.0)
                        g_vals.append(float(np.linalg.norm(np.cross(n_v, n_f))))

                    m0 = max(m0_vals)
                    g = max(g_vals)
                    m1 = (
                        float(abs(np.dot(B_c, n_v)) / B_norm) if B_norm > 1e-30 else 0.0
                    )
                    ratio_m1_g = float(m1 / g) if g > 1e-14 else 0.0
                    excess_m1_g = float(m1 - g)

                    rec = {
                        "h": int(h),
                        "corner": [int(a), int(b_idx), int(c)],
                        "v": v,
                        "class": cls,
                        "n_walls": int(n_walls),
                        "coords": coord,
                        "m0": m0,
                        "m1": m1,
                        "g": g,
                        "m1_over_g": ratio_m1_g,
                        "m1_minus_g": excess_m1_g,
                    }
                    records_by_class[cls].append(rec)
                    all_boundary_records.append(rec)
                    if n_walls == 1:
                        all_one_wall_records.append(rec)

    def _summarize_records(recs, nodes_in_cls):
        if not recs:
            return {
                "n_nodes": len(nodes_in_cls),
                "n_octants": 0,
                "max_m0": 0.0,
                "max_m1": 0.0,
                "max_g": 0.0,
                "max_m1_over_g": 0.0,
                "max_m1_minus_g": 0.0,
                "max_m1_location": None,
                "max_g_location": None,
            }
        idx_m0 = int(np.argmax([r["m0"] for r in recs]))
        idx_m1 = int(np.argmax([r["m1"] for r in recs]))
        idx_g = int(np.argmax([r["g"] for r in recs]))
        idx_ratio = int(np.argmax([r["m1_over_g"] for r in recs]))
        idx_exc = int(np.argmax([r["m1_minus_g"] for r in recs]))
        return {
            "n_nodes": len(nodes_in_cls),
            "n_octants": len(recs),
            "max_m0": float(recs[idx_m0]["m0"]),
            "max_m1": float(recs[idx_m1]["m1"]),
            "max_g": float(recs[idx_g]["g"]),
            "max_m1_over_g": float(recs[idx_ratio]["m1_over_g"]),
            "max_m1_minus_g": float(recs[idx_exc]["m1_minus_g"]),
            "max_m1_location": {
                "node": int(recs[idx_m1]["v"]),
                "class": recs[idx_m1]["class"],
                "coords": recs[idx_m1]["coords"],
            },
            "max_g_location": {
                "node": int(recs[idx_g]["v"]),
                "class": recs[idx_g]["class"],
                "coords": recs[idx_g]["coords"],
            },
        }

    per_class = {}
    for cls in NODE_CLASSES:
        nodes_c = [v for v, c in node_class.items() if c == cls]
        per_class[cls] = _summarize_records(records_by_class[cls], nodes_c)

    one_wall_nodes = {r["v"] for r in all_one_wall_records}
    all_b_nodes = set(node_class.keys())

    return {
        "per_class": per_class,
        "one_wall_octants": _summarize_records(all_one_wall_records, one_wall_nodes),
        "all_boundary_octants": _summarize_records(all_boundary_records, all_b_nodes),
    }


def compute_node_averaged_B(sol, b_face):
    """
    Compute corner-volume-weighted node-averaged magnetic flux density ``B_v``
    at every mesh node from face-integrated fluxes ``b_face``.

    Parameters
    ----------
    sol : CurvilinearFIT3D
        Assembled curvilinear solver.
    b_face : ndarray of shape (n_faces,)
        Face-integrated magnetic flux vector ``b``.

    Returns
    -------
    B_node : ndarray of shape (n_nodes, 3)
        Corner-volume-weighted magnetic flux density ``B_v`` [T].
    """
    msh = sol.mesh
    B_sum = np.zeros((msh.n_nodes, 3), dtype=float)
    V_sum = np.zeros(msh.n_nodes, dtype=float)
    w_hex = msh.corner_volumes

    for a in (0, 1):
        for b_idx in (0, 1):
            for c in (0, 1):
                c_nodes = msh.hexes[:, _CORNER_VTK[a, b_idx, c]]
                f0 = msh.hex_faces[:, 0, a]
                f1 = msh.hex_faces[:, 1, b_idx]
                f2 = msh.hex_faces[:, 2, c]
                b_loc = np.stack([b_face[f0], b_face[f1], b_face[f2]], axis=1)
                B_c = np.einsum("nij,nj->ni", sol._R_c[a, b_idx, c], b_loc)
                np.add.at(B_sum, c_nodes, w_hex[:, None] * B_c)
                np.add.at(V_sum, c_nodes, w_hex)

    return B_sum / np.maximum(V_sum[:, None], 1e-30)


def evaluate_eigenmode_m3(
    sol,
    node_class,
    geometry_kind,
    a=1.0,
    length=1.2,
    k_modes=12,
):
    """
    Compute ``m3 = |B_v . n| / max_v |B_v|`` for the lowest TE and TM cavity
    eigenmodes using both ``n = n_v`` (discrete area-summed node normal) and
    ``n = n_exact`` (analytic surface normal).

    To obtain a basis-invariant measure on degenerate multiplets, ``B_v`` is
    evaluated across the modes spanning the lowest TM and lowest TE multiplets
    (RMS over the multiplet normalized by its global maximum ``max_v |B_v|``).

    Parameters
    ----------
    sol : CurvilinearFIT3D
        Assembled curvilinear solver.
    node_class : dict of int -> str
        Boundary node classification.
    geometry_kind : {"sphere", "cylinder"}
        Cavity geometry kind.
    a : float, optional
        Radius [m] (default 1.0).
    length : float, optional
        Cylinder length [m] (default 1.2).
    k_modes : int, optional
        Number of positive cavity modes to compute (default 12).

    Returns
    -------
    dict
        Per-mode (``TM_lowest``, ``TE_lowest``) and per-class ``m3_nv`` and
        ``m3_exact`` maxima and locations.
    """
    msh = sol.mesh
    omegas, d_vecs = sol.cavity_frequencies(k=k_modes, return_vectors=True)
    C_u = sol.C_u
    W_eps_u = sol.W_eps_u

    if geometry_kind == "sphere":
        w_te1, w_tm1 = sphere_lowest_omegas(a=a, l=1)
        targets = {
            "TE_lowest": float(w_te1),
            "TM_lowest": float(w_tm1),
        }
    elif geometry_kind == "cylinder":
        targets = {
            "TM_lowest": float(cylinder_tm010_omega(a=a)),
            "TE_lowest": float(cylinder_te111_omega(a=a, length=length)),
        }
    else:
        raise ValueError(f"Unsupported geometry_kind for m3: {geometry_kind!r}")

    mode_groups = {}
    for label, w_ref in targets.items():
        idx0 = int(np.argmin(np.abs(omegas - w_ref)))
        grp = np.flatnonzero(
            np.abs(omegas - omegas[idx0]) <= 1e-4 * max(omegas[idx0], 1.0)
        )
        mode_groups[label] = [int(i) for i in grp]

    n_exact_all = exact_surface_normals(geometry_kind, msh.nodes, a=a, length=length)

    out = {}
    for label, indices in mode_groups.items():
        valid_indices = [i for i in indices if i < d_vecs.shape[1]]
        B_modes = []
        for idx in valid_indices:
            d_vec = d_vecs[:, idx]
            b_face = C_u @ (W_eps_u @ d_vec)
            b_face[msh.boundary_faces] = 0.0
            B_v = compute_node_averaged_B(sol, b_face)
            B_modes.append(B_v)

        B_stack = np.stack(B_modes, axis=0)
        B_mag_rms = np.sqrt(np.mean(np.sum(B_stack * B_stack, axis=-1), axis=0))
        B_global_max = float(np.max(B_mag_rms))

        dot_nv_rms = np.sqrt(
            np.mean(
                np.sum(B_stack * msh.node_normals[None, :, :], axis=-1) ** 2,
                axis=0,
            )
        )
        dot_ex_rms = np.sqrt(
            np.mean(
                np.sum(B_stack * n_exact_all[None, :, :], axis=-1) ** 2,
                axis=0,
            )
        )

        m3_nv_all = dot_nv_rms / max(B_global_max, 1e-30)
        m3_ex_all = dot_ex_rms / max(B_global_max, 1e-30)

        per_class = {}
        for cls in NODE_CLASSES:
            v_list = [v for v, c in node_class.items() if c == cls]
            if not v_list:
                per_class[cls] = {
                    "n_nodes": 0,
                    "max_m3_nv": 0.0,
                    "max_m3_exact": 0.0,
                    "max_m3_nv_location": None,
                    "max_m3_exact_location": None,
                }
                continue
            v_arr = np.asarray(v_list, dtype=np.int64)
            nv_sub = m3_nv_all[v_arr]
            ex_sub = m3_ex_all[v_arr]
            i_nv = int(np.argmax(nv_sub))
            i_ex = int(np.argmax(ex_sub))
            v_nv = int(v_arr[i_nv])
            v_ex = int(v_arr[i_ex])
            per_class[cls] = {
                "n_nodes": len(v_list),
                "max_m3_nv": float(nv_sub[i_nv]),
                "max_m3_exact": float(ex_sub[i_ex]),
                "max_m3_nv_location": {
                    "node": v_nv,
                    "class": cls,
                    "coords": [float(x) for x in msh.nodes[v_nv]],
                },
                "max_m3_exact_location": {
                    "node": v_ex,
                    "class": cls,
                    "coords": [float(x) for x in msh.nodes[v_ex]],
                },
            }

        all_b_nodes = np.asarray(sorted(node_class.keys()), dtype=np.int64)
        i_nv_all = int(np.argmax(m3_nv_all[all_b_nodes]))
        i_ex_all = int(np.argmax(m3_ex_all[all_b_nodes]))
        v_nv_all = int(all_b_nodes[i_nv_all])
        v_ex_all = int(all_b_nodes[i_ex_all])

        out[label] = {
            "omega_mean_rad_s": float(np.mean(omegas[valid_indices])),
            "per_class": per_class,
            "all_boundary_nodes": {
                "n_nodes": int(all_b_nodes.size),
                "max_m3_nv": float(m3_nv_all[v_nv_all]),
                "max_m3_exact": float(m3_ex_all[v_ex_all]),
                "max_m3_nv_location": {
                    "node": v_nv_all,
                    "class": node_class[v_nv_all],
                    "coords": [float(x) for x in msh.nodes[v_nv_all]],
                },
                "max_m3_exact_location": {
                    "node": v_ex_all,
                    "class": node_class[v_ex_all],
                    "coords": [float(x) for x in msh.nodes[v_ex_all]],
                },
            },
        }
    return out


def _attach_convergence_orders(runs):
    """
    Compute LS and local convergence orders across resolution runs for a
    cavity family.
    """
    ns = [float(r["n"]) for r in runs]
    orders = {
        "one_wall_octants": {
            "max_m1": {
                "ls_order": fit_order(
                    ns,
                    [r["octants"]["one_wall_octants"]["max_m1"] for r in runs],
                ),
                "local_orders": local_orders(
                    ns,
                    [r["octants"]["one_wall_octants"]["max_m1"] for r in runs],
                ),
            },
            "max_g": {
                "ls_order": fit_order(
                    ns,
                    [r["octants"]["one_wall_octants"]["max_g"] for r in runs],
                ),
                "local_orders": local_orders(
                    ns,
                    [r["octants"]["one_wall_octants"]["max_g"] for r in runs],
                ),
            },
        },
        "per_class": {},
    }

    for cls in NODE_CLASSES:
        m1_seq = [r["octants"]["per_class"][cls]["max_m1"] for r in runs]
        g_seq = [r["octants"]["per_class"][cls]["max_g"] for r in runs]
        cls_ord = {
            "max_m1_ls_order": fit_order(ns, m1_seq),
            "max_m1_local_orders": local_orders(ns, m1_seq),
            "max_g_ls_order": fit_order(ns, g_seq),
            "max_g_local_orders": local_orders(ns, g_seq),
        }
        if "m3" in runs[0]:
            for mode_key in ("TM_lowest", "TE_lowest"):
                nv_seq = [
                    r["m3"][mode_key]["per_class"][cls]["max_m3_nv"] for r in runs
                ]
                ex_seq = [
                    r["m3"][mode_key]["per_class"][cls]["max_m3_exact"] for r in runs
                ]
                cls_ord[f"{mode_key}_m3_nv_ls_order"] = fit_order(ns, nv_seq)
                cls_ord[f"{mode_key}_m3_nv_local_orders"] = local_orders(ns, nv_seq)
                cls_ord[f"{mode_key}_m3_exact_ls_order"] = fit_order(ns, ex_seq)
                cls_ord[f"{mode_key}_m3_exact_local_orders"] = local_orders(ns, ex_seq)
        orders["per_class"][cls] = cls_ord

    return orders


def run_diagnosis(quick=False):
    """
    Execute the full W4-B Part 1 wall ``B . n`` diagnostic study.

    Parameters
    ----------
    quick : bool, optional
        If True, run ``N in {4, 6, 8, 12}`` on ``sphere_nodes`` and the two
        coarsest resolutions ``n in {4, 6}`` on ``cubed_sphere_ball_nodes``
        and ``butterfly_disk_nodes``.

    Returns
    -------
    dict
        JSON-serialisable diagnostic results.
    """
    t0 = time.perf_counter()
    sphere_ns = SPHERE_NS_QUICK if quick else SPHERE_NS_FULL
    hex_ns = HEXMESH_NS_QUICK if quick else HEXMESH_NS_FULL

    # 1. Single-block sphere_nodes(N, N, N)
    sphere_runs = []
    for n in sphere_ns:
        blk = HexBlock3D(sphere_nodes(n, n, n, a=1.0))
        sol = CurvilinearFIT3D(blk)
        msh = sol.mesh
        node_cls, max_angles = classify_boundary_nodes(msh)
        oct_stats = evaluate_wall_octant_metrics(sol, node_cls, seed=2026)
        m3_stats = evaluate_eigenmode_m3(sol, node_cls, geometry_kind="sphere", a=1.0)
        seam_angles = [
            max_angles[v] for v, c in node_cls.items() if c == "sharp_smooth_seam"
        ]
        sphere_runs.append(
            {
                "n": int(n),
                "n_hexes": int(msh.n_cells),
                "n_wall_octants": int(sol.n_wall_octants),
                "n_sharp_octants": int(sol.n_sharp_octants),
                "max_seam_facet_angle_deg": (
                    float(max(seam_angles)) if seam_angles else 0.0
                ),
                "octants": oct_stats,
                "m3": m3_stats,
            }
        )

    # 2. Multi-block cubed_sphere_ball_nodes(n)
    cubed_runs = []
    for n in hex_ns:
        msh_in = cubed_sphere_ball_nodes(n, a=1.0)
        sol = CurvilinearFIT3D(msh_in)
        msh = sol.mesh
        node_cls, _ = classify_boundary_nodes(msh)
        oct_stats = evaluate_wall_octant_metrics(sol, node_cls, seed=2026)
        m3_stats = evaluate_eigenmode_m3(sol, node_cls, geometry_kind="sphere", a=1.0)
        cubed_runs.append(
            {
                "n": int(n),
                "n_hexes": int(msh.n_cells),
                "n_wall_octants": int(sol.n_wall_octants),
                "n_sharp_octants": int(sol.n_sharp_octants),
                "octants": oct_stats,
                "m3": m3_stats,
            }
        )

    # 3. Multi-block butterfly_disk_nodes(n)
    butterfly_runs = []
    for n in hex_ns:
        msh_in = butterfly_disk_nodes(n, a=1.0, length=1.2)
        sol = CurvilinearFIT3D(msh_in)
        msh = sol.mesh
        node_cls, _ = classify_boundary_nodes(msh)
        oct_stats = evaluate_wall_octant_metrics(sol, node_cls, seed=2026)
        m3_stats = evaluate_eigenmode_m3(
            sol, node_cls, geometry_kind="cylinder", a=1.0, length=1.2
        )
        butterfly_runs.append(
            {
                "n": int(n),
                "n_hexes": int(msh.n_cells),
                "n_wall_octants": int(sol.n_wall_octants),
                "n_sharp_octants": int(sol.n_sharp_octants),
                "octants": oct_stats,
                "m3": m3_stats,
            }
        )

    # 4. Flat-wall control: perturbed_box_nodes(6, 5, 4)
    box_blk = HexBlock3D(perturbed_box_nodes(6, 5, 4, a=1.0, b=0.7, d=0.8, amp=0.3))
    box_sol = CurvilinearFIT3D(box_blk)
    box_msh = box_sol.mesh
    box_cls, _ = classify_boundary_nodes(box_msh)
    box_oct = evaluate_wall_octant_metrics(box_sol, box_cls, seed=2026)

    # 5. Mutations at sphere_nodes(6, 6, 6)
    sol_sph6 = CurvilinearFIT3D(HexBlock3D(sphere_nodes(6, 6, 6, a=1.0)))
    cls_sph6, _ = classify_boundary_nodes(sol_sph6.mesh)
    mut_a = evaluate_wall_octant_metrics(
        sol_sph6, cls_sph6, seed=2026, zero_wall_b=False, transpose_rc=False
    )
    mut_b = evaluate_wall_octant_metrics(
        sol_sph6, cls_sph6, seed=2026, zero_wall_b=True, transpose_rc=True
    )

    max_excess_all = max(
        [
            r["octants"]["all_boundary_octants"]["max_m1_minus_g"]
            for r in sphere_runs + cubed_runs + butterfly_runs
        ]
        + [box_oct["all_boundary_octants"]["max_m1_minus_g"]]
    )
    max_m0_all = max(
        [
            r["octants"]["all_boundary_octants"]["max_m0"]
            for r in sphere_runs + cubed_runs + butterfly_runs
        ]
        + [box_oct["all_boundary_octants"]["max_m0"]]
    )

    return {
        "quick": bool(quick),
        "wall_seconds": float(time.perf_counter() - t0),
        "identity_holds_globally": bool(
            max_excess_all <= 1e-12 and max_m0_all <= 1e-12
        ),
        "max_m0_globally": float(max_m0_all),
        "max_m1_minus_g_globally": float(max_excess_all),
        "sphere_nodes": {
            "runs": sphere_runs,
            "orders": _attach_convergence_orders(sphere_runs),
        },
        "cubed_sphere_ball_nodes": {
            "runs": cubed_runs,
            "orders": _attach_convergence_orders(cubed_runs),
        },
        "butterfly_disk_nodes": {
            "runs": butterfly_runs,
            "orders": _attach_convergence_orders(butterfly_runs),
        },
        "perturbed_box_control": {
            "dims": [6, 5, 4],
            "n_hexes": int(box_msh.n_cells),
            "n_wall_octants": int(box_sol.n_wall_octants),
            "n_sharp_octants": int(box_sol.n_sharp_octants),
            "octants": box_oct,
        },
        "mutations_sphere_n6": {
            "mutation_a_unzeroed_wall_b": mut_a["one_wall_octants"],
            "mutation_b_transpose_rc": mut_b["one_wall_octants"],
        },
    }


def _print_summary(data, stream=sys.stdout):
    print(
        f"=== W4-B Part 1: Wall B.n Diagnosis (quick={data['quick']}) ===",
        file=stream,
    )
    print(
        f"global identity m1 <= g + 1e-12: {data['identity_holds_globally']} "
        f"(max_m0={data['max_m0_globally']:.6e}, "
        f"max(m1 - g)={data['max_m1_minus_g_globally']:.6e})",
        file=stream,
    )
    for fam in (
        "sphere_nodes",
        "cubed_sphere_ball_nodes",
        "butterfly_disk_nodes",
    ):
        sec = data[fam]
        ord_1w = sec["orders"]["one_wall_octants"]
        sw_ord = sec["orders"]["per_class"]["smooth_wall"]
        print(
            f"\n[{fam}] 1-wall octant LS orders: "
            f"m1={ord_1w['max_m1']['ls_order']:.4f}, "
            f"g={ord_1w['max_g']['ls_order']:.4f} | "
            f"smooth_wall m3 LS orders: "
            f"TM(nv/ex)={sw_ord['TM_lowest_m3_nv_ls_order']:.4f}/"
            f"{sw_ord['TM_lowest_m3_exact_ls_order']:.4f}, "
            f"TE(nv/ex)={sw_ord['TE_lowest_m3_nv_ls_order']:.4f}/"
            f"{sw_ord['TE_lowest_m3_exact_ls_order']:.4f}",
            file=stream,
        )
        for r in sec["runs"]:
            ow = r["octants"]["one_wall_octants"]
            tm_sw = r["m3"]["TM_lowest"]["per_class"]["smooth_wall"]
            te_sw = r["m3"]["TE_lowest"]["per_class"]["smooth_wall"]
            print(
                f"  N={r['n']:2d} (hexes={r['n_hexes']:4d}, "
                f"wall_oct={r['n_wall_octants']:4d}, "
                f"sharp_oct={r['n_sharp_octants']:3d}): "
                f"1-wall m0={ow['max_m0']:.3e} m1={ow['max_m1']:.6e} "
                f"g={ow['max_g']:.6e} m1/g={ow['max_m1_over_g']:.6f} "
                f"m1-g={ow['max_m1_minus_g']:+.3e} | "
                f"smooth_wall TM m3(nv/ex)={tm_sw['max_m3_nv']:.4e}/"
                f"{tm_sw['max_m3_exact']:.4e} "
                f"TE m3(nv/ex)={te_sw['max_m3_nv']:.4e}/"
                f"{te_sw['max_m3_exact']:.4e}",
                file=stream,
            )
    box_ow = data["perturbed_box_control"]["octants"]["one_wall_octants"]
    print(
        f"\n[perturbed_box_nodes(6,5,4)] 1-wall m0={box_ow['max_m0']:.3e} "
        f"m1={box_ow['max_m1']:.3e} g={box_ow['max_g']:.3e} "
        f"m1-g={box_ow['max_m1_minus_g']:+.3e}",
        file=stream,
    )
    mut = data["mutations_sphere_n6"]
    print(
        f"[mutations on sphere_nodes(6,6,6)] "
        f"mut_a(unzeroed b): m0={mut['mutation_a_unzeroed_wall_b']['max_m0']:.6e}, "
        f"m1={mut['mutation_a_unzeroed_wall_b']['max_m1']:.6e}, "
        f"m1-g={mut['mutation_a_unzeroed_wall_b']['max_m1_minus_g']:.6e} | "
        f"mut_b(R_c^T): m0={mut['mutation_b_transpose_rc']['max_m0']:.6e}",
        file=stream,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="W4-B Part 1: wall B.n residual diagnosis."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick resolution set.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path.",
    )
    args = parser.parse_args(argv)

    data = run_diagnosis(quick=args.quick)
    _print_summary(data, stream=sys.stdout)
    if args.out:
        payload = json.dumps(data, indent=2)
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")


if __name__ == "__main__":
    main()
