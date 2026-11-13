"""
Figures of the unstructured-adjoint shadow-focusing demo (reads results.json
and fields.npz written by shadow_focus.py; matplotlib Agg).

    python studies/unstructured-adjoint/make_figures.py

figures/00_setup_mesh.png       setup and graded triangle mesh (with zoom)
figures/01_design_evolution.png eps_r per triangle at three beta stages
figures/02_objective.png        J/J0 vs iteration, beta stages shaded
figures/03_gradient_vs_fd.png   adjoint vs central FD gradients (+ errors)
figures/04_fields.png           |Ez(f0)| bare cylinder vs binarized design
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from wakis import figstyle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
A_PEC, B_DES, R_OUT, R_PML = 0.5, 1.25, 2.5, 2.98
SRC, FOCUS = (-2.0, 0.0), (1.75, 0.0)


def markers(ax, labels=True):
    ax.plot(*SRC, marker="*", ms=11, color="#D55E00", mec="k", mew=0.6, ls="none",
            label="line source")  # fmt: skip
    ax.plot(*FOCUS, marker="o", ms=7, mfc="none", mec="#009E73", mew=2, ls="none",
            label="focus (objective)")  # fmt: skip
    ax.add_patch(Circle((0, 0), A_PEC, fc="#4d4d4d", ec="k", lw=0.8, zorder=3))
    if labels:
        ax.text(0, 0, "PEC", color="w", ha="center", va="center", zorder=4)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(os.path.join(HERE, "results.json")))
    d = np.load(os.path.join(HERE, "fields.npz"))
    nodes, cells, tags = d["nodes"], d["cells"], d["tags"]
    dcells = d["design_cells"]
    tri = Triangulation(nodes[:, 0], nodes[:, 1], cells)
    tri_d = Triangulation(nodes[:, 0], nodes[:, 1], cells[dcells])
    m = res["mesh"]
    figstyle.apply_style()

    # ---- 00 setup + mesh
    fig, (a, b) = figstyle.new_figure(7.0, 3.5, ncols=2)
    shade = np.choose(tags, [0.0, 0.55, 0.15, 0.85])
    a.tripcolor(tri, facecolors=shade, cmap="Greys", vmin=0, vmax=1.6, edgecolors="none")
    a.triplot(tri, lw=0.15, color="#6B7280")
    for r, ls in ((B_DES, "-"), (R_OUT, "--")):
        a.add_patch(Circle((0, 0), r, fc="none", ec="k", lw=0.8, ls=ls))
    markers(a)
    a.annotate("design annulus\n$1\\leq\\varepsilon_r\\leq 4$", (0.0, 0.9), (-1.2, 2.2),
               arrowprops=dict(arrowstyle="->", lw=0.8), ha="center")  # fmt: skip
    a.annotate("CFS-PML", (2.0, 2.0), (2.1, 2.75), arrowprops=dict(arrowstyle="->", lw=0.8))
    a.set(xlim=(-3.05, 3.05), ylim=(-3.05, 3.05), aspect="equal", xlabel="x / $\\lambda_0$",
          ylabel="y / $\\lambda_0$")  # fmt: skip
    a.legend(loc="lower left", fontsize=7)
    figstyle.panel_label(a, "(a)")
    b.triplot(tri, lw=0.35, color="#374151")
    b.add_patch(Circle((0, 0), A_PEC, fc="#4d4d4d", ec="k", lw=0.8, zorder=3, alpha=0.5))
    b.add_patch(Circle((0, 0), B_DES, fc="none", ec="#D55E00", lw=1.2))
    b.set(xlim=(0.3, 0.95), ylim=(-0.3, 0.35), aspect="equal", xlabel="x / $\\lambda_0$",
          ylabel="y / $\\lambda_0$",
          title=f"conforming triangles, h = $\\lambda_0$/{1 / m['h_fine']:.0f}")  # fmt: skip
    figstyle.panel_label(b, "(b)")
    fig.suptitle(f"{m['n_nodes']} nodes, {m['n_cells']} triangles "
                 f"({m['n_design_cells']} design), {res['Nt']} steps", fontsize=9)  # fmt: skip
    figstyle.save(fig, os.path.join(OUT, "00_setup_mesh.png"), close=True)

    # ---- 01 design evolution (three stages)
    betas, snaps = d["snap_betas"], d["snaps"]
    pick = [0, len(betas) // 2, len(betas) - 1]
    fig, axs = figstyle.new_figure(7.0, 2.7, ncols=3)
    for ax, i in zip(axs, pick):
        im = ax.tripcolor(tri_d, facecolors=snaps[i], cmap="viridis", vmin=1, vmax=4)
        markers(ax, labels=False)
        ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal",
               title=f"$\\beta$ = {betas[i]:g}", xlabel="x / $\\lambda_0$")  # fmt: skip
    axs[0].set_ylabel("y / $\\lambda_0$")
    cax = axs[-1].inset_axes([1.05, 0.0, 0.05, 1.0])
    fig.colorbar(im, cax=cax, label="$\\varepsilon_r$ per triangle")
    figstyle.save(fig, os.path.join(OUT, "01_design_evolution.png"), close=True,
                  auto_legend_below=False)  # fmt: skip

    # ---- 02 objective
    h = res["history"]
    J = np.array([e["J"] for e in h])
    st = np.array([e["stage"] for e in h])
    fig, ax = figstyle.new_figure(3.4, 2.8)
    it = np.arange(1, J.size + 1)
    for s_ in np.unique(st):
        sel = st == s_
        ax.axvspan(it[sel][0] - 0.5, it[sel][-1] + 0.5, color="0.92" if s_ % 2 else "white")
        ax.text(it[sel].mean(), 1.05 * J.max(), f"$\\beta$={res['betas'][s_]:g}",
                ha="center", fontsize=7)  # fmt: skip
    ax.plot(it, J, **figstyle.series_style(0), label="L-BFGS-B evaluations")
    ax.axhline(res["J_binarized"], color="#D55E00", ls="--",
               label=f"binarized: {res['J_binarized']:.0f}")  # fmt: skip
    g = res["guards"]
    ax.axhline(1 / g["shadow_depth_J0_over_Jfree"], color="#009E73", ls=":",
               label="free space (no cylinder)")  # fmt: skip
    ax.set(yscale="log", xlabel="objective evaluation",
           ylabel="$J/J_0 = |E_z(f_0)|^2 / |E_z^{bare}|^2$")  # fmt: skip
    ax.legend(fontsize=7, loc="lower right")
    figstyle.save(fig, os.path.join(OUT, "02_objective.png"), close=True,
                  auto_legend_below=False)  # fmt: skip

    # ---- 03 gradient vs FD
    gc = res["gradcheck"]
    adj, fd = np.array(gc["adjoint"]), np.array(gc["fd_h"])
    fig, (a, b) = figstyle.new_figure(7.0, 2.9, ncols=2)
    lim = 1.1 * np.abs(np.r_[adj, fd]).max()
    a.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8)
    a.plot(fd, adj, **figstyle.series_style(0, with_line=False))
    a.set(xlabel="central FD  $\\partial J/\\partial\\rho_T$", ylabel="adjoint",
          aspect="equal", xlim=(-lim, lim), ylim=(-lim, lim))  # fmt: skip
    figstyle.panel_label(a, "(a)")
    k = np.arange(adj.size)
    b.semilogy(k, gc["rel_err_fd"], **figstyle.series_style(0), label="|adj - FD(h)| / |FD|")
    b.semilogy(k, gc["trunc_estimate"], **figstyle.series_style(1),
               label="truncation est. |FD(2h) - FD(h)| / 3|FD|")  # fmt: skip
    b.semilogy(k, gc["rel_err_richardson"], **figstyle.series_style(2),
               label="|adj - Richardson| / |R|")  # fmt: skip
    b.set(xlabel="probed triangle", ylabel="relative difference", title=f"h = {gc['h']:g}")
    figstyle.place_legend_below(b, ncol=1)
    figstyle.panel_label(b, "(b)")
    figstyle.save(fig, os.path.join(OUT, "03_gradient_vs_fd.png"), close=True)

    # ---- 04 fields: intensity in dB relative to free space at the focus
    I_free = res["J0_abs"] * (1 + g["J0_rel_change_2Nt"]) / g["shadow_depth_J0_over_Jfree"]
    fig, axs = figstyle.new_figure(7.0, 3.3, ncols=2)
    titles = ("bare PEC cylinder", "optimized (binarized)")
    for ax, key, title in zip(axs, ("bare", "design"), titles):
        dB = 10 * np.log10(np.maximum(np.abs(d[key]) ** 2 / I_free, 1e-6))
        im = ax.tripcolor(tri, dB, cmap="magma", vmin=-30, vmax=15, shading="gouraud")
        markers(ax, labels=False)
        ax.add_patch(Circle((0, 0), B_DES, fc="none", ec="w", lw=0.6, ls="--"))
        ax.set(xlim=(-R_OUT, R_OUT), ylim=(-R_OUT, R_OUT), aspect="equal", title=title,
               xlabel="x / $\\lambda_0$")  # fmt: skip
    axs[0].set_ylabel("y / $\\lambda_0$")
    cax = axs[-1].inset_axes([1.05, 0.0, 0.05, 1.0])
    fig.colorbar(im, cax=cax, label="$|E_z(f_0)|^2$ / free-space value at focus [dB]")
    figstyle.save(fig, os.path.join(OUT, "04_fields.png"), close=True, auto_legend_below=False)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
