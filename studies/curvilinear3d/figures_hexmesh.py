"""Generate publication figures for the 3D curvilinear hexahedral FIT study.

Produces:
- ``studies/curvilinear3d/figures/hexmesh_convergence.png`` (Figure A):
  log-log relative eigenfrequency error ``|rel_err|`` vs mesh size ``h`` [m]
  for ``CurvilinearFIT3D``, ``SolverFIT3D conformal``, and
  ``SolverFIT3D staircase`` on the butterfly cylinder (``TM010``, ``TE111``)
  and cubed-sphere ball (``TE_l1``, ``TM_l1``), with ``O(h)`` and ``O(h^2)``
  order guide lines.
- ``studies/curvilinear3d/figures/hexmesh_meshes.png`` (Figure B):
  2D projections of the 5-block butterfly cylinder mesh and the 7-block
  cubed-sphere ball mesh, showing the hexahedral wireframe, the outer PEC
  walls, the sharp octant boundary nodes (where ``D = 0`` is enforced),
  labelled axes with physical units ``[m]``, and a ``0.5 m`` scale bar.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.collections import LineCollection

matplotlib.use("Agg")

from wakis import figstyle
from wakis.curvilinear.mappings3d import (
    butterfly_disk_nodes,
    cubed_sphere_ball_nodes,
)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

_DEFAULT_JSON = Path(__file__).resolve().parent / "results" / "hexmesh_study.json"
_DEFAULT_FIG_DIR = Path(__file__).resolve().parent / "figures"


def _draw_scale_bar(ax, x0, y0, length_m=0.5, label="0.5 m"):
    """Draw a horizontal physical scale bar of length ``length_m`` [m]."""
    cap = 0.04
    ax.plot(
        [x0, x0 + length_m],
        [y0, y0],
        color="#1a1a1a",
        lw=2.0,
        marker="none",
        solid_capstyle="butt",
        zorder=8,
    )
    ax.plot(
        [x0, x0],
        [y0 - cap, y0 + cap],
        color="#1a1a1a",
        lw=1.6,
        marker="none",
        zorder=8,
    )
    ax.plot(
        [x0 + length_m, x0 + length_m],
        [y0 - cap, y0 + cap],
        color="#1a1a1a",
        lw=1.6,
        marker="none",
        zorder=8,
    )
    ax.text(
        x0 + 0.5 * length_m,
        y0 + 0.055,
        label,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#1a1a1a",
        zorder=9,
    )


def plot_meshes(out_path=None):
    """Render Figure B: 2D projections of butterfly disk and cubed-sphere meshes.

    Parameters
    ----------
    out_path : str or Path, optional
        Output PNG path. Defaults to
        ``studies/curvilinear3d/figures/hexmesh_meshes.png``.

    Returns
    -------
    Path
        Saved figure path.
    """
    if out_path is None:
        out_path = _DEFAULT_FIG_DIR / "hexmesh_meshes.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(width_in=7.0, height_in=3.55, ncols=2, nrows=1)
        ax_cyl, ax_sph = axes

        # ------------------------------------------------------------------
        # Panel (a): Butterfly cylinder (n = 4, a = 1.0 m, L = 1.2 m), z = 0
        # ------------------------------------------------------------------
        msh_cyl = butterfly_disk_nodes(4, a=1.0, length=1.2)
        nodes_c = msh_cyl.nodes
        edges_c = msh_cyl.edge_nodes
        sharp_n_c = msh_cyl.effective_sharp_nodes

        z0_nodes = np.abs(nodes_c[:, 2]) < 1e-12
        z0_edges = z0_nodes[edges_c[:, 0]] & z0_nodes[edges_c[:, 1]]

        # Interior cross-section edges vs outer cylindrical wall edges at r = a
        r0 = np.hypot(nodes_c[edges_c[:, 0], 0], nodes_c[edges_c[:, 0], 1])
        r1 = np.hypot(nodes_c[edges_c[:, 1], 0], nodes_c[edges_c[:, 1], 1])
        wall_edges_c = z0_edges & (np.abs(r0 - 1.0) < 1e-9) & (np.abs(r1 - 1.0) < 1e-9)
        int_edges_c = z0_edges & (~wall_edges_c)

        segs_int_c = np.stack(
            [
                nodes_c[edges_c[int_edges_c, 0], :2],
                nodes_c[edges_c[int_edges_c, 1], :2],
            ],
            axis=1,
        )
        segs_wall_c = np.stack(
            [
                nodes_c[edges_c[wall_edges_c, 0], :2],
                nodes_c[edges_c[wall_edges_c, 1], :2],
            ],
            axis=1,
        )

        ax_cyl.add_collection(
            LineCollection(
                segs_int_c,
                colors="#6B7280",
                linewidths=0.85,
                alpha=0.85,
                label="Hexahedral wireframe",
                zorder=2,
            )
        )
        ax_cyl.add_collection(
            LineCollection(
                segs_wall_c,
                colors="#0072B2",
                linewidths=2.2,
                label="Outer PEC wall",
                zorder=4,
            )
        )

        # Sharp octant nodes on z = 0 rim (where r = a meets z = 0 end cap)
        sharp_z0 = z0_nodes & sharp_n_c
        ax_cyl.scatter(
            nodes_c[sharp_z0, 0],
            nodes_c[sharp_z0, 1],
            s=30,
            marker="D",
            facecolor="#D55E00",
            edgecolor="#1a1a1a",
            linewidth=0.6,
            label=f"Sharp octant nodes ({int(np.sum(sharp_n_c))} total)",
            zorder=6,
        )

        _draw_scale_bar(ax_cyl, -0.95, -1.16, length_m=0.5, label="0.5 m")
        ax_cyl.set_aspect("equal")
        ax_cyl.set_xlim(-1.18, 1.18)
        ax_cyl.set_ylim(-1.26, 1.15)
        ax_cyl.set_xlabel("x [m]")
        ax_cyl.set_ylabel("y [m]")
        ax_cyl.set_title("(a) Butterfly cylinder (n = 4)")
        figstyle.place_legend_below(ax_cyl, ncol=1, y_offset=-0.22)

        # ------------------------------------------------------------------
        # Panel (b): Cubed-sphere ball (n = 6, a = 1.0 m), z = 0 equatorial cut
        # ------------------------------------------------------------------
        msh_sph = cubed_sphere_ball_nodes(6, a=1.0)
        nodes_s = msh_sph.nodes
        edges_s = msh_sph.edge_nodes
        bnd_e_s = msh_sph.boundary_edge_mask
        sharp_n_s = msh_sph.effective_sharp_nodes

        # Equatorial plane z = 0 edges (exists for even n = 6) + full 2D wireframe projection
        z_eq = np.abs(nodes_s[:, 2]) < 1e-10
        eq_edges = z_eq[edges_s[:, 0]] & z_eq[edges_s[:, 1]]
        wall_edges_s = eq_edges & bnd_e_s
        int_edges_s = eq_edges & (~bnd_e_s)

        # Faint background projection of all boundary shell edges for 3D context
        bnd_all_segs = np.stack(
            [nodes_s[edges_s[bnd_e_s, 0], :2], nodes_s[edges_s[bnd_e_s, 1], :2]],
            axis=1,
        )
        ax_sph.add_collection(
            LineCollection(
                bnd_all_segs,
                colors="#CBD5E1",
                linewidths=0.45,
                alpha=0.55,
                zorder=1,
            )
        )

        segs_int_s = np.stack(
            [
                nodes_s[edges_s[int_edges_s, 0], :2],
                nodes_s[edges_s[int_edges_s, 1], :2],
            ],
            axis=1,
        )
        segs_wall_s = np.stack(
            [
                nodes_s[edges_s[wall_edges_s, 0], :2],
                nodes_s[edges_s[wall_edges_s, 1], :2],
            ],
            axis=1,
        )
        ax_sph.add_collection(
            LineCollection(
                segs_int_s,
                colors="#6B7280",
                linewidths=0.85,
                alpha=0.9,
                label="Equatorial hex slice (z = 0)",
                zorder=3,
            )
        )
        ax_sph.add_collection(
            LineCollection(
                segs_wall_s,
                colors="#0072B2",
                linewidths=2.2,
                label="Outer PEC wall (r = a)",
                zorder=4,
            )
        )

        # Project the 8 sharp octant cube-corner nodes (+-a/sqrt(3), +-a/sqrt(3), +-a/sqrt(3))
        sharp_xy = nodes_s[sharp_n_s, :2]
        ax_sph.scatter(
            sharp_xy[:, 0],
            sharp_xy[:, 1],
            s=42,
            marker="D",
            facecolor="#D55E00",
            edgecolor="#1a1a1a",
            linewidth=0.7,
            label=f"Sharp octant nodes ({int(np.sum(sharp_n_s))} corners, 24 oct.)",
            zorder=6,
        )

        _draw_scale_bar(ax_sph, -0.95, -1.16, length_m=0.5, label="0.5 m")
        ax_sph.set_aspect("equal")
        ax_sph.set_xlim(-1.18, 1.18)
        ax_sph.set_ylim(-1.26, 1.15)
        ax_sph.set_xlabel("x [m]")
        ax_sph.set_ylabel("y [m]")
        ax_sph.set_title("(b) Cubed-sphere ball (n = 6)")
        figstyle.place_legend_below(ax_sph, ncol=1, y_offset=-0.22)

        figstyle.save(fig, out_path, close=True, auto_legend_below=False)

    return out_path


def plot_convergence(json_path=None, out_path=None):
    """Render Figure A: log-log convergence ``|rel_err|`` vs ``h`` [m].

    Parameters
    ----------
    json_path : str or Path, optional
        Input study JSON path. Defaults to
        ``studies/curvilinear3d/results/hexmesh_study.json``.
    out_path : str or Path, optional
        Output PNG path. Defaults to
        ``studies/curvilinear3d/figures/hexmesh_convergence.png``.

    Returns
    -------
    Path
        Saved figure path.
    """
    if json_path is None:
        json_path = _DEFAULT_JSON
    if out_path is None:
        out_path = _DEFAULT_FIG_DIR / "hexmesh_convergence.png"
    json_path = Path(json_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    records = data["records"]

    method_styles = {
        "CurvilinearFIT3D": {"color": "#0072B2", "label": "CurvilinearFIT3D"},
        "SolverFIT3D conformal": {
            "color": "#009E73",
            "label": "SolverFIT3D conformal",
        },
        "SolverFIT3D staircase": {
            "color": "#D55E00",
            "label": "SolverFIT3D staircase",
        },
    }

    panels = [
        (
            "butterfly_disk",
            [("TM010", "-", "o"), ("TE111", "--", "s")],
            "(a) Butterfly cylinder (a = 1 m, L = 1.2 m)",
            "Relative error |rel_err| [-]",
        ),
        (
            "cubed_sphere",
            [("TE_l1", "-", "o"), ("TM_l1", "--", "s")],
            "(b) Cubed-sphere ball (a = 1 m)",
            "Relative error |rel_err| [-]",
        ),
    ]

    with figstyle.apply_style():
        fig, axes = figstyle.new_figure(
            width_in=7.0,
            height_in=4.4,
            ncols=2,
            nrows=1,
            gridspec_kw={"wspace": 0.34},
        )

        for ax, (geo, mode_specs, title, ylabel) in zip(axes, panels):
            h_all = []
            for method_name, m_cfg in method_styles.items():
                for mode_name, ls, marker in mode_specs:
                    pts = [
                        r
                        for r in records
                        if r["geometry"] == geo
                        and r["mode"] == mode_name
                        and r["method"] == method_name
                    ]
                    if not pts:
                        continue
                    pts.sort(key=lambda r: float(r["h"]), reverse=True)
                    h_vals = np.array([float(r["h"]) for r in pts], dtype=float)
                    err_vals = np.array(
                        [max(abs(float(r["rel_err"])), 1e-16) for r in pts],
                        dtype=float,
                    )
                    h_all.extend(h_vals.tolist())
                    short_method = (
                        "CurvFIT"
                        if method_name == "CurvilinearFIT3D"
                        else ("Conf." if "conformal" in method_name else "Stair.")
                    )
                    ax.loglog(
                        h_vals,
                        err_vals,
                        color=m_cfg["color"],
                        linestyle=ls,
                        marker=marker,
                        linewidth=1.7,
                        markersize=5.0,
                        markerfacecolor=m_cfg["color"] if ls == "-" else "white",
                        markeredgecolor=m_cfg["color"],
                        markeredgewidth=1.3,
                        label=f"{short_method} ({mode_name})",
                    )

            if h_all:
                h_min, h_max = min(h_all), max(h_all)
                h_ref = np.geomspace(h_min, h_max, 32)
                # Slope-1 and slope-2 reference guides
                c1 = 0.22 * (h_ref / h_max) ** 1
                c2 = 0.015 * (h_ref / h_max) ** 2
                ax.loglog(
                    h_ref,
                    c1,
                    color="#4B5563",
                    linestyle=":",
                    marker="none",
                    linewidth=1.3,
                    label="Slope 1: O(h)",
                )
                ax.loglog(
                    h_ref,
                    c2,
                    color="#1a1a1a",
                    linestyle="-.",
                    marker="none",
                    linewidth=1.3,
                    label="Slope 2: O(h²)",
                )

            figstyle.spectrum_grid(ax)
            ax.set_xlabel("Mesh spacing h [m]")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=9.5)
            figstyle.place_legend_below(ax, ncol=2, y_offset=-0.22)

        figstyle.save(fig, out_path, close=True, auto_legend_below=False)

    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render hexmesh study convergence and mesh figures."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=_DEFAULT_JSON,
        help="Path to hexmesh_study.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_FIG_DIR,
        help="Directory for output PNG figures.",
    )
    parser.add_argument(
        "--meshes-only",
        action="store_true",
        help="Render only hexmesh_meshes.png (does not require hexmesh_study.json).",
    )
    args = parser.parse_args(argv)

    p_mesh = plot_meshes(args.out_dir / "hexmesh_meshes.png")
    print(f"Wrote {p_mesh}")
    if not args.meshes_only and args.json.exists():
        p_conv = plot_convergence(args.json, args.out_dir / "hexmesh_convergence.png")
        print(f"Wrote {p_conv}")


if __name__ == "__main__":
    main()
