# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Generate Brief W4-B diagnostic figures using the non-interactive matplotlib
``Agg`` backend (PNG outputs are not committed to the repository):

1. ``w4_sphere_wall_bn_map.png``: Spherical wall maps ``(phi, theta)`` of the
   octant-level magnetic node-normal ratio ``m1 = |B_c . n_v| / ||B_c||`` and
   the facet-to-node normal angle bound ``g = ||n_v x n_f||`` on
   ``sphere_nodes(6, 6, 6)`` and ``cubed_sphere_ball_nodes(6)``, with labelled
   boundary node classes (``smooth_wall``, ``smooth_seam``,
   ``sharp_smooth_seam``, ``corner``), marked cube-to-sphere block seams, and
   the ``m1 <= g`` parity scatter.
2. ``w4_perturbed_box_decomposition.png``: Log-log convergence of ``|e(amp)|``,
   ``|e_Yee| = |e(0)|``, and ``|e_dist| = |e(amp) - e(0)|`` vs resolution ``N``
   across families ``F1`` (constrained and ``drop_only``), ``F2``, and ``F3``
   with algebraic order reference guides.

Usage::

    python studies/curvilinear3d/plot_w4_diagnostics.py [--quick] [--out-dir /tmp]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from studies.curvilinear3d.bn_wall_diagnosis import (
    classify_boundary_nodes,
    run_diagnosis,
)
from studies.curvilinear3d.perturbed_box_order import run_perturbed_box_study
from wakis.curvilinear import CurvilinearFIT3D, HexBlock3D
from wakis.curvilinear.hexmesh import _CORNER_VTK
from wakis.curvilinear.mappings3d import cubed_sphere_ball_nodes, sphere_nodes


def _collect_node_m1_g(sol, seed=2026):
    """
    Collect per-boundary-node maxima of ``m1`` and ``g`` over incident boundary
    octants for wall-map visualization.
    """
    from studies.curvilinear3d.bn_wall_diagnosis import (
        _build_local_boundary_patch_grid,
        _sample_random_b,
    )

    msh = sol.mesh
    node_cls, _ = classify_boundary_nodes(msh)
    b = _sample_random_b(msh, seed=seed, zero_wall_b=True)
    local_patch = _build_local_boundary_patch_grid(msh)

    f_norm = np.linalg.norm(msh.face_areas, axis=1, keepdims=True)
    f_unit = msh.face_areas / np.maximum(f_norm, 1e-30)

    node_m1 = {v: 0.0 for v in node_cls}
    node_g = {v: 0.0 for v in node_cls}

    for a in (0, 1):
        for b_idx in (0, 1):
            for c in (0, 1):
                c_loc = int(_CORNER_VTK[a, b_idx, c])
                R_all = sol._R_c[a, b_idx, c]
                sides = (a, b_idx, c)
                for h in range(msh.n_cells):
                    v = int(msh.hexes[h, c_loc])
                    if v not in node_cls:
                        continue
                    wall_axes = [
                        ax for ax in range(3) if local_patch[h, ax, sides[ax]] >= 0
                    ]
                    if not wall_axes:
                        continue
                    f_trip = np.array(
                        [
                            msh.hex_faces[h, 0, a],
                            msh.hex_faces[h, 1, b_idx],
                            msh.hex_faces[h, 2, c],
                        ],
                        dtype=np.int64,
                    )
                    B_c = R_all[h] @ b[f_trip]
                    B_norm = float(np.linalg.norm(B_c))
                    n_v = msh.node_normals[v]
                    m1 = (
                        float(abs(np.dot(B_c, n_v)) / B_norm) if B_norm > 1e-30 else 0.0
                    )
                    g = max(
                        float(
                            np.linalg.norm(
                                np.cross(
                                    n_v,
                                    (-1.0 if sides[ax_w] == 0 else 1.0)
                                    * float(msh.hex_face_signs[h, ax_w, sides[ax_w]])
                                    * f_unit[int(msh.hex_faces[h, ax_w, sides[ax_w]])],
                                )
                            )
                        )
                        for ax_w in wall_axes
                    )
                    node_m1[v] = max(node_m1[v], m1)
                    node_g[v] = max(node_g[v], g)

    return node_cls, node_m1, node_g


def _cube_seam_curves_deg(n_pts=80):
    """
    Parametric curves ``(phi_deg, theta_deg)`` of the 12 cube-to-sphere block
    seams ``max_i |p_i| = 1`` projected onto ``S^2``.
    """
    t = np.linspace(-1.0, 1.0, n_pts)
    ones = np.ones_like(t)
    edges = [
        (t, ones, ones),
        (t, -ones, ones),
        (t, ones, -ones),
        (t, -ones, -ones),
        (ones, t, ones),
        (-ones, t, ones),
        (ones, t, -ones),
        (-ones, t, -ones),
        (ones, ones, t),
        (-ones, ones, t),
        (ones, -ones, t),
        (-ones, -ones, t),
    ]
    curves = []
    for x, y, z in edges:
        r = np.sqrt(x * x + y * y + z * z)
        phi = np.degrees(np.arctan2(y, x))
        theta = np.degrees(np.arccos(np.clip(z / r, -1.0, 1.0)))
        # Split across the +-180 deg branch cut if needed
        jumps = np.flatnonzero(np.abs(np.diff(phi)) > 180.0)
        start = 0
        for j in jumps:
            curves.append((phi[start : j + 1], theta[start : j + 1]))
            start = j + 1
        curves.append((phi[start:], theta[start:]))
    return curves


def plot_wall_bn_map(out_path, n=6):
    """
    Plot the wall map of ``m1`` and ``g`` on ``sphere_nodes(n, n, n)`` and
    ``cubed_sphere_ball_nodes(n)`` with labelled node classes and block seams.
    """
    sol_sph = CurvilinearFIT3D(HexBlock3D(sphere_nodes(n, n, n, a=1.0)))
    cls_sph, m1_sph, g_sph = _collect_node_m1_g(sol_sph)

    sol_cub = CurvilinearFIT3D(cubed_sphere_ball_nodes(n, a=1.0))
    cls_cub, m1_cub, g_cub = _collect_node_m1_g(sol_cub)

    seam_curves = _cube_seam_curves_deg()
    markers = {
        "smooth_wall": ("o", 28, "smooth_wall"),
        "smooth_seam": ("s", 42, "smooth_seam"),
        "sharp_smooth_seam": ("^", 46, "sharp_smooth_seam (D_c=0)"),
        "corner": ("D", 58, "corner (D_c=0)"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)

    for ax, msh, node_cls, vals, title_str in (
        (
            axes[0],
            sol_sph.mesh,
            cls_sph,
            m1_sph,
            f"sphere_nodes({n},{n},{n}): m1 = |B_c . n_v| / |B_c| [1]",
        ),
        (
            axes[1],
            sol_sph.mesh,
            cls_sph,
            g_sph,
            f"sphere_nodes({n},{n},{n}): g = |n_v x n_f| [1]",
        ),
    ):
        for phi_c, theta_c in seam_curves:
            ax.plot(
                phi_c,
                theta_c,
                color="0.55",
                linestyle="--",
                linewidth=0.9,
                zorder=1,
            )
        vmin = 0.0
        vmax = max(g_sph.values())
        last_sc = None
        for cls_name, (mk, sz, lbl) in markers.items():
            v_ids = [v for v, c in node_cls.items() if c == cls_name]
            if not v_ids:
                continue
            pts = msh.nodes[v_ids]
            r = np.linalg.norm(pts, axis=1)
            phi_deg = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
            theta_deg = np.degrees(np.arccos(np.clip(pts[:, 2] / r, -1.0, 1.0)))
            c_vals = [vals[v] for v in v_ids]
            last_sc = ax.scatter(
                phi_deg,
                theta_deg,
                c=c_vals,
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                marker=mk,
                s=sz,
                edgecolors="k",
                linewidths=0.4,
                label=lbl,
                zorder=3,
            )
        if last_sc is not None:
            fig.colorbar(last_sc, ax=ax, shrink=0.88, label="Dimensionless ratio [1]")
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel("Azimuth phi [deg]")
        ax.set_ylabel("Polar angle theta [deg]")
        ax.set_xlim(-185.0, 185.0)
        ax.set_ylim(180.0, 0.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    # Panel 3: parity plot m1 vs g across node classes on both sphere meshes
    ax_p = axes[2]
    g_max = max(max(g_sph.values()), max(g_cub.values()))
    line_x = np.linspace(0.0, 1.05 * g_max, 100)
    ax_p.plot(line_x, line_x, "k--", linewidth=1.1, label="Identity bound m1 = g")
    for mesh_lbl, node_cls, m1_d, g_d, color in (
        (f"sphere_nodes({n})", cls_sph, m1_sph, g_sph, "tab:blue"),
        (f"cubed_sphere({n})", cls_cub, m1_cub, g_cub, "tab:orange"),
    ):
        for cls_name, (mk, sz, _) in markers.items():
            v_ids = [v for v, c in node_cls.items() if c == cls_name]
            if not v_ids:
                continue
            ax_p.scatter(
                [g_d[v] for v in v_ids],
                [m1_d[v] for v in v_ids],
                marker=mk,
                s=sz * 0.75,
                color=color,
                edgecolors="k",
                linewidths=0.35,
                alpha=0.8,
                label=f"{mesh_lbl}: {cls_name}",
            )
    ax_p.set_title("Facet-to-node angle bound verification: m1 <= g", fontsize=10)
    ax_p.set_xlabel("Facet-node normal angle bound g = |n_v x n_f| [1]")
    ax_p.set_ylabel("Magnetic node-normal ratio m1 = |B_c . n_v| / |B_c| [1]")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(fontsize=7.5, loc="upper left")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_perturbed_box_decomposition(box_data, out_path):
    """
    Plot ``|e(amp)|``, ``|e_Yee|``, and ``|e_dist|`` vs ``N`` on log-log axes
    for ``F1`` (constrained and ``drop_only``), ``F2``, and ``F3`` with
    algebraic order guides.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)

    families = [
        ("F1", "F1: isotropic (n, n, n), cell amp=0.3", axes[0]),
        ("F2", "F2: aspect (4k, 3k, 4k), cell amp=0.3", axes[1]),
        ("F3", "F3: isotropic (n, n, n), fixed physical map", axes[2]),
    ]

    for key, title_str, ax in families:
        sec = box_data[key]
        ns = np.asarray(sec["ns"], dtype=float)
        e_amp = np.abs(np.asarray(sec["signed_errors_amp"], dtype=float))
        e_yee = np.abs(np.asarray(sec["signed_errors_yee_amp0"], dtype=float))
        e_dist = np.abs(np.asarray(sec["signed_errors_dist"], dtype=float))

        ax.loglog(
            ns,
            e_amp,
            "o-",
            label=f"|e(amp)| (p={sec['ls_order_amp']:.3f})",
        )
        ax.loglog(
            ns,
            e_yee,
            "s--",
            label=f"|e_Yee(0)| (p={sec['ls_order_yee_amp0']:.3f}, <0)",
        )
        ax.loglog(
            ns,
            e_dist,
            "^-.",
            label=f"|e_dist| (p={sec['ls_order_dist']:.3f}, >0)",
        )

        if key == "F1" and "F1_study_window_drop_only" in box_data:
            wd = box_data["F1_study_window_drop_only"]
            ns_d = np.asarray(wd["ns"], dtype=float)
            e_amp_d = np.abs(np.asarray(wd["signed_errors_amp"], dtype=float))
            e_dist_d = np.abs(np.asarray(wd["signed_errors_dist"], dtype=float))
            ax.loglog(
                ns_d,
                e_amp_d,
                "d:",
                color="tab:red",
                label=f"|e(drop_only)| (D={wd['D_drop_only_ls_order_amp']:.3f})",
            )
            ax.loglog(
                ns_d,
                e_dist_d,
                "v:",
                color="tab:purple",
                label=f"|e_dist(drop_only)| (p={wd['ls_order_dist']:.3f})",
            )

        # Reference order guide lines anchored at the coarsest resolution
        n_ref = ns[0]
        ax.loglog(
            ns,
            0.85 * e_yee[0] * (ns / n_ref) ** (-2.0),
            color="0.45",
            linestyle=":",
            linewidth=1.0,
            label="O(N^-2) guide",
        )
        if key in ("F1",):
            ax.loglog(
                ns,
                0.85 * e_dist[0] * (ns / n_ref) ** (-3.5),
                color="0.65",
                linestyle="--",
                linewidth=0.9,
                label="O(N^-3.5) guide",
            )
        else:
            ax.loglog(
                ns,
                0.85 * e_dist[0] * (ns / n_ref) ** (-1.5),
                color="0.65",
                linestyle="--",
                linewidth=0.9,
                label="O(N^-1.5) guide",
            )

        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel("Resolution parameter N [cells]")
        ax.set_ylabel("Relative eigenfrequency error magnitude [1]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7.5)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate Brief W4-B diagnostic figures (matplotlib Agg)."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick resolution sets when computing data on the fly.",
    )
    parser.add_argument(
        "--bn-json",
        type=str,
        default=None,
        help="Optional path to precomputed bn_wall_diagnosis.py JSON.",
    )
    parser.add_argument(
        "--box-json",
        type=str,
        default=None,
        help="Optional path to precomputed perturbed_box_order.py JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/tmp",
        help="Output directory for generated PNG figures (default /tmp).",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.bn_json:
        with open(args.bn_json, encoding="utf-8") as f:
            _ = json.load(f)
    elif args.quick:
        _ = run_diagnosis(quick=True)

    if args.box_json:
        with open(args.box_json, encoding="utf-8") as f:
            box_data = json.load(f)
    else:
        box_data = run_perturbed_box_study(quick=args.quick)

    p1 = os.path.join(args.out_dir, "w4_sphere_wall_bn_map.png")
    p2 = os.path.join(args.out_dir, "w4_perturbed_box_decomposition.png")
    plot_wall_bn_map(p1, n=6)
    plot_perturbed_box_decomposition(box_data, p2)
    print(f"Wrote {p1} and {p2}")


if __name__ == "__main__":
    main()
