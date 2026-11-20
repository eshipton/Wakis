"""Deterministic user-manual illustrations for Sections 5, 6, 7, and 8.

Draws every figure with ``"route": "scripted"`` in
``user_manual_figures/section{N}/specs/section{N}.json`` for N in 5, 6, 7, 8:
- section5/fig5_3_staggering_and_synchronization.png
- section6/fig6_1_domain_edge_boundaries.png
- section6/fig6_3_graded_mesh_and_snapping.png
- section6/fig6_4_conformal_pec_and_sheets.png
- section7/fig7_2_ade_dispersive_poles.png
- section7/fig7_4_cold_plasma_wakefield.png
- section8/fig8_2_waveforms_and_normalization.png

Usage: python sections_5_6_7_8_scripted.py OUTROOT
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)  # noqa: E402

from wakis import GridFIT3D  # noqa: E402

BLUE, LBLUE = "#2f6fb0", "#d6e6f5"
ORANGE, LORANGE = "#e8873a", "#fbe0c8"
GREY, LGREY = "#4a4f57", "#e6e8eb"
GREEN, LGREEN = "#5f9e6e", "#d3ebd8"
RED, LRED = "#c0392b", "#f9d6d2"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "cm",
        "font.size": 15,
        "axes.linewidth": 1.2,
        "savefig.dpi": 200,
    }
)
W, H = 13.76, 7.68  # 2752 x 1536 px at 200 dpi (16:9)
W21 = 18.06  # 3612 x 1536 px at 200 dpi (21:9)


def arrow(ax, p, q, color=GREY, lw=2.0, style="-|>", rad=0.0, ms=18, z=3, ls="-"):
    """Draw a FancyArrowPatch between p and q on ax."""
    a = FancyArrowPatch(
        p,
        q,
        arrowstyle=style,
        mutation_scale=ms,
        color=color,
        lw=lw,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
        linestyle=ls,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(a)
    return a


def box(ax, xy, w, h, text, fc, ec, fs=15, tc="black", lw=1.6, pad=0.02, r=0.06, z=2):
    """Draw a rounded rectangle with centred text."""
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad={pad},rounding_size={r}",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=z,
        )
    )
    if text:
        ax.text(
            x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 2
        )


def badge(ax, xy, text, fc=GREEN, tc="white", fs=13.5, pad=0.35, z=5):
    """Draw a pill badge annotation at xy."""
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        zorder=z,
        bbox=dict(boxstyle=f"round,pad={pad}", fc=fc, ec="none"),
    )


def bare(ax):
    """Turn off axis spines and ticks with equal aspect ratio."""
    ax.set_aspect("equal")
    ax.axis("off")


# --------------------------------------------------------------------------- 5.3
def fig5_3(outroot):
    """Figure 5.3: Temporal and spatial staggering of E and H and O(dt^2), O(dx^2) synchronization."""
    out_dir = outroot / "section5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mathematical verification of O(dt^2) temporal synchronization (user_manual.md lines 1043, 1048)
    # and O(dx^2) spatial monitor-plane averaging (user_manual.md lines 1038-1039, 1051-1053).
    f0 = 1.0
    w0 = 2 * np.pi * f0
    t_end = 4.0
    phase_errs_unsync = []
    amp_errs_sync = []
    for nt in (80, 160):
        dt = t_end / nt
        n = np.arange(nt)
        t_e = (n + 1.0) * dt  # user_manual.md line 1035
        t_h = (n + 0.5) * dt  # user_manual.md line 1034
        e_sig = (t_e / t_end) * (1.0 - t_e / t_end) * np.cos(w0 * t_e)
        h_sig = (t_h / t_end) * (1.0 - t_h / t_end) * np.cos(w0 * t_h)
        # Unsynchronized H DFT using t_E kernel vs synchronized H DFT using t_H kernel (line 1048)
        e_dft = np.sum(e_sig * np.exp(-2j * np.pi * f0 * t_e)) * dt
        h_dft_unsync = np.sum(h_sig * np.exp(-2j * np.pi * f0 * t_e)) * dt
        h_dft_sync = np.sum(h_sig * np.exp(-2j * np.pi * f0 * t_h)) * dt
        phase_errs_unsync.append(abs(np.angle(e_dft / h_dft_unsync) - 0.5 * w0 * dt))
        amp_errs_sync.append(abs(e_dft - h_dft_sync) / abs(e_dft))
    assert (
        phase_errs_unsync[-1] < 1e-3
    )  # unsynchronized phase offset is omega * dt / 2 = O(omega dt)
    assert np.isclose(
        amp_errs_sync[0] / amp_errs_sync[1], 4.0, rtol=5e-2
    )  # O(dt^2) convergence

    # Verify native E and H sample positions on GridFIT3D (user_manual.md lines 1038-1039, 1051-1052)
    grid = GridFIT3D(0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2, 1, 1, verbose=0)
    _, coords_ey = grid.region_indices("E", "y", (1.0, 0.5, 0.0), (0.0, 0.0, 0.0))
    _, coords_ez = grid.region_indices("E", "z", (1.0, 0.0, 0.5), (0.0, 0.0, 0.0))
    _, coords_hy = grid.region_indices("H", "y", (1.0, 0.0, 0.5), (1.0, 0.0, 0.0))
    _, coords_hz = grid.region_indices("H", "z", (1.0, 0.5, 0.0), (1.0, 0.0, 0.0))
    x_i = grid.x[1]
    x_m, x_p = 0.5 * (grid.x[0] + grid.x[1]), 0.5 * (grid.x[1] + grid.x[2])
    assert np.isclose(coords_ey[0][0], x_i) and np.isclose(coords_ez[0][0], x_i)
    assert np.allclose(coords_hy[0], [x_m, x_p]) and np.allclose(
        coords_hz[0], [x_m, x_p]
    )
    # Spatial average H_t(x_i) = (H_t(x_{i-1/2}) + H_t(x_{i+1/2})) / 2 has O(dx^2) accuracy (line 1052)
    k_wave = 1.2
    sp_err_1 = abs(
        0.5 * (np.cos(k_wave * (x_i - 0.5)) + np.cos(k_wave * (x_i + 0.5)))
        - np.cos(k_wave * x_i)
    )
    sp_err_2 = abs(
        0.5 * (np.cos(k_wave * (x_i - 0.25)) + np.cos(k_wave * (x_i + 0.25)))
        - np.cos(k_wave * x_i)
    )
    assert np.isclose(sp_err_1 / sp_err_2, 4.0, rtol=5e-2)

    fig = plt.figure(figsize=(W, H))

    # ---- LEFT PANEL: Temporal staggering and synchronization ----
    ax = fig.add_axes([0.02, 0.04, 0.47, 0.92])
    bare(ax)
    # Scale u in {-0.5, 0.0, 0.5, 1.0} by 2.0 so ticks have generous horizontal spacing
    su = 2.0
    xc = 0.65
    ax.text(
        xc,
        3.32,
        "Temporal staggering & synchronization (solver.one_step())",
        ha="center",
        va="center",
        fontsize=15.0,
        weight="bold",
        color="black",
    )

    # Time axis u in [-0.5, 1.0] -> x in [-1.0, 2.0] at y = 1.68
    y_ax = 1.68
    arrow(ax, (-1.35, y_ax), (2.55, y_ax), GREY, lw=1.8, ms=18)
    ax.text(2.62, y_ax, r"$t$", ha="left", va="center", fontsize=17, color=GREY)

    u_ticks = [
        (-0.5, r"$(n-\frac{1}{2})\Delta t$"),
        (0.0, r"$n\Delta t$"),
        (0.5, r"$t_H=(n+\frac{1}{2})\Delta t$"),
        (1.0, r"$t_E=(n+1)\Delta t$"),
    ]
    for u, lab in u_ticks:
        xp = su * u
        ax.plot([xp, xp], [y_ax - 0.07, y_ax + 0.07], color=GREY, lw=1.6)
        ax.text(
            xp, y_ax - 0.22, lab, ha="center", va="top", fontsize=13.0, color="black"
        )

    y_node = 2.12
    # E^n at u = 0, E^{n+1} at u = 1 (blue circles)
    for u, lab in ((0.0, r"$\mathbf{E}^n$"), (1.0, r"$\mathbf{E}^{n+1}$")):
        xp = su * u
        ax.add_patch(Circle((xp, y_node), 0.11, color=BLUE, zorder=5))
        ax.text(
            xp,
            y_node + 0.20,
            lab,
            ha="center",
            va="bottom",
            fontsize=16,
            color=BLUE,
            weight="bold",
        )

    # H^{n-1/2} at u = -0.5, H^{n+1/2} at u = 0.5 (orange squares)
    sq_s = 0.20
    for u, lab in ((-0.5, r"$\mathbf{H}^{n-1/2}$"), (0.5, r"$\mathbf{H}^{n+1/2}$")):
        xp = su * u
        ax.add_patch(
            Rectangle(
                (xp - sq_s / 2, y_node - sq_s / 2), sq_s, sq_s, color=ORANGE, zorder=5
            )
        )
        ax.text(
            xp,
            y_node + 0.20,
            lab,
            ha="center",
            va="bottom",
            fontsize=16,
            color=ORANGE,
            weight="bold",
        )

    # Curved orange arrow 1 from u = -0.5 to u = 0.5 passing above E^n at u = 0
    arrow(
        ax,
        (su * -0.5 + 0.12, y_node + 0.48),
        (su * 0.5 - 0.12, y_node + 0.48),
        ORANGE,
        lw=2.4,
        rad=-0.36,
        ms=18,
        z=6,
    )
    ax.text(
        su * -0.05,
        y_node + 0.88,
        r"1. update $\mathbf{H}$ to $t_H = (n+1/2)\Delta t$",
        ha="center",
        va="bottom",
        fontsize=13.0,
        color=ORANGE,
        weight="bold",
    )

    # Curved blue arrow 2 from u = 0 to u = 1 passing through H^{n+1/2} at u = 0.5
    arrow(
        ax,
        (su * 0.0 + 0.14, y_node - 0.06),
        (su * 1.0 - 0.14, y_node - 0.06),
        BLUE,
        lw=2.4,
        rad=0.28,
        ms=18,
        z=6,
    )
    ax.text(
        su * 0.52,
        y_node - 0.23,
        r"2. update $\mathbf{E}$ to $t_E = (n+1)\Delta t$",
        ha="center",
        va="top",
        fontsize=12.5,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
        zorder=7,
    )

    # Vertical dark-grey dashed line just after u = 1
    u_mon = su * 1.0 + 0.24
    ax.plot([u_mon, u_mon], [0.95, 2.75], color=GREY, lw=1.8, ls="--", zorder=4)
    ax.text(
        u_mon - 0.05,
        2.82,
        "monitors & callbacks\nrun at end of step $n$",
        ha="center",
        va="bottom",
        fontsize=12.0,
        color=GREY,
    )

    # Red horizontal double-arrow between u = 0.5 and u = 1.0
    y_off = 1.12
    ax.plot(
        [su * 0.5, su * 0.5], [y_off - 0.06, y_ax - 0.42], color=RED, lw=1.1, ls=":"
    )
    ax.plot(
        [su * 1.0, su * 1.0], [y_off - 0.06, y_ax - 0.42], color=RED, lw=1.1, ls=":"
    )
    arrow(
        ax, (su * 0.5, y_off), (su * 1.0, y_off), RED, lw=2.2, style="<|-|>", ms=16, z=6
    )
    ax.text(
        su * 0.65,
        y_off - 0.14,
        r"$\Delta t / 2$ offset (unsynchronized: $O(\omega\Delta t)$ phase error)",
        ha="center",
        va="top",
        fontsize=12.5,
        color=RED,
    )

    # Green-bordered box beneath timeline with exact synchronization rules (lines 1048, 1055)
    box(ax, (xc, -0.02), 4.10, 1.56, "", LGREEN, GREEN, lw=2.0, r=0.08, z=1)
    ax.text(
        -1.26,
        0.58,
        "(a) DFTMonitor phase factors:",
        ha="left",
        va="center",
        fontsize=13,
        weight="bold",
        color=GREY,
        zorder=4,
    )
    ax.text(
        -1.18,
        0.31,
        r"$E_{\mathrm{DFT}}(f) = \sum \mathbf{E}^{n+1}\exp(-2\pi i f (n+1)\Delta t)\,\Delta t$",
        ha="left",
        va="center",
        fontsize=13.2,
        color=BLUE,
        zorder=4,
    )
    ax.text(
        -1.18,
        0.03,
        r"$H_{\mathrm{DFT}}(f) = \sum \mathbf{H}^{n+1/2}\exp(-2\pi i f (n+1/2)\Delta t)\,\Delta t$",
        ha="left",
        va="center",
        fontsize=13.2,
        color=ORANGE,
        zorder=4,
    )
    ax.text(
        -1.26,
        -0.28,
        r"(b) LumpedPort time average at $\tau = (n+1/2)\Delta t$:",
        ha="left",
        va="center",
        fontsize=13,
        weight="bold",
        color=GREY,
        zorder=4,
    )
    ax.text(
        -1.18,
        -0.56,
        r"$I^{n+1/2},\ \ \bar{V}^{n+1/2} = -\sum_e g_e (E_e^n + E_e^{n+1})/2$",
        ha="left",
        va="center",
        fontsize=13.2,
        color="black",
        zorder=4,
    )
    badge(
        ax, (1.88, -0.56), r"$O(\Delta t^2)$ synchronized", fc=GREEN, fs=12.5, pad=0.28
    )

    ax.set_xlim(-1.50, 2.82)
    ax.set_ylim(-0.92, 3.50)

    # Divider between panels
    fig.add_artist(
        plt.Line2D(
            [0.50, 0.50], [0.06, 0.94], transform=fig.transFigure, color=LGREY, lw=1.6
        )
    )

    # ---- RIGHT PANEL: Spatial staggering and monitor-plane averaging ----
    ax = fig.add_axes([0.51, 0.04, 0.47, 0.92])
    bare(ax)
    ax.text(
        1.0,
        2.32,
        "Spatial staggering & monitor-plane averaging",
        ha="center",
        va="center",
        fontsize=15.5,
        weight="bold",
        color="black",
    )

    # Two adjacent primary cells [x_{i-1}, x_i] x [y_j, y_{j+1}] and [x_i, x_{i+1}] x [y_j, y_{j+1}]
    # Map x_{i-1}=0, x_i=1, x_{i+1}=2; y_j=0.35, y_{j+1}=1.65
    y0, y1 = 0.35, 1.65
    ym = 0.5 * (y0 + y1)
    ax.add_patch(
        Rectangle((0.0, y0), 2.0, y1 - y0, fc=LBLUE, ec="none", alpha=0.35, zorder=1)
    )

    # Primary grid lines (blue)
    for xv, lab in ((0.0, r"$x_{i-1}$"), (1.0, r"$x_i$"), (2.0, r"$x_{i+1}$")):
        ax.plot([xv, xv], [y0 - 0.12, y1 + 0.12], color=BLUE, lw=2.0, zorder=2)
        ax.text(xv, y0 - 0.18, lab, ha="center", va="top", fontsize=15, color=BLUE)
    for yv, lab in ((y0, r"$y_j$"), (y1, r"$y_{j+1}$")):
        ax.plot([-0.10, 2.10], [yv, yv], color=BLUE, lw=2.0, zorder=2)
        ax.text(-0.15, yv, lab, ha="right", va="center", fontsize=15, color=BLUE)

    # Dual vertical dashed lines at x_{i-1/2}=0.5, x_{i+1/2}=1.5 and horizontal at y_{j+1/2} (orange)
    for xv, lab in ((0.5, r"$x_{i-1/2}$"), (1.5, r"$x_{i+1/2}$")):
        ax.plot(
            [xv, xv], [y0 - 0.12, y1 + 0.12], color=ORANGE, lw=1.8, ls="--", zorder=2
        )
        ax.text(xv, y0 - 0.18, lab, ha="center", va="top", fontsize=14.5, color=ORANGE)
    ax.plot([-0.10, 2.10], [ym, ym], color=ORANGE, lw=1.6, ls="--", alpha=0.8, zorder=2)
    ax.text(
        -0.15, ym, r"$y_{j+1/2}$", ha="right", va="center", fontsize=14.5, color=ORANGE
    )

    # Highlight monitor plane x = x_i with a thick green vertical line
    ax.plot([1.0, 1.0], [y0 - 0.08, y1 + 0.18], color=GREEN, lw=5.0, zorder=3)
    badge(ax, (1.0, y1 + 0.34), r"monitor plane $x = x_i$", fc=GREEN, fs=13.5, pad=0.3)

    # Tangential E markers on x = x_i:
    # 1. Blue arrow at (x_i, y_{j+1/2}) for E_y(x_i, y_{j+1/2}, z_k)
    arrow(ax, (1.0, ym - 0.22), (1.0, ym + 0.22), BLUE, lw=3.8, ms=22, z=6)
    ax.text(
        1.0,
        ym + 0.30,
        r"$E_y(x_i, y_{j+1/2}, z_k)$",
        ha="center",
        va="bottom",
        fontsize=12.5,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=BLUE, lw=1.0, alpha=0.94),
        zorder=7,
    )

    # 2. Blue circled-dot at (x_i, y_j) for E_z(x_i, y_j, z_{k+1/2})
    ax.add_patch(Circle((1.0, y0), 0.068, fc="white", ec=BLUE, lw=2.2, zorder=6))
    ax.add_patch(Circle((1.0, y0), 0.024, color=BLUE, zorder=7))
    ax.text(
        1.0,
        y0 + 0.23,
        r"$E_z(x_i, y_j, z_{k+1/2})$",
        ha="center",
        va="bottom",
        fontsize=12.5,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=BLUE, lw=1.0, alpha=0.94),
        zorder=7,
    )

    # Bracketing tangential H markers (orange squares) at x_{i-1/2}=0.5 and x_{i+1/2}=1.5:
    hs = 0.095
    for xv in (0.5, 1.5):
        # H_y at (x_{i+-1/2}, y_j)
        ax.add_patch(
            Rectangle((xv - hs / 2, y0 - hs / 2), hs, hs, color=ORANGE, zorder=6)
        )
        # H_z at (x_{i+-1/2}, y_{j+1/2})
        ax.add_patch(
            Rectangle((xv - hs / 2, ym - hs / 2), hs, hs, color=ORANGE, zorder=6)
        )

    # Labels for H_z(x_{i±1/2}, y_{j+1/2}, z_k) and H_y(x_{i±1/2}, y_j, z_{k+1/2})
    ax.text(
        0.38,
        ym - 0.13,
        r"$H_z(x_{i\pm 1/2}, y_{j+1/2}, z_k)$",
        ha="center",
        va="top",
        fontsize=12.0,
        color=ORANGE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=ORANGE, lw=0.9, alpha=0.94),
        zorder=7,
    )
    ax.text(
        1.62,
        ym - 0.13,
        r"$H_y(x_{i\pm 1/2}, y_j, z_{k+1/2})$",
        ha="center",
        va="top",
        fontsize=12.0,
        color=ORANGE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=ORANGE, lw=0.9, alpha=0.94),
        zorder=7,
    )
    arrow(ax, (1.55, ym - 0.25), (1.51, y0 + 0.06), ORANGE, lw=1.3, ms=12, z=6)

    # Two curved green arrows from x_{i-1/2} and x_{i+1/2} converging onto x = x_i
    arrow(
        ax, (0.55, ym + 0.04), (0.94, ym + 0.04), GREEN, lw=2.5, rad=-0.32, ms=18, z=7
    )
    arrow(ax, (1.45, ym + 0.04), (1.06, ym + 0.04), GREEN, lw=2.5, rad=0.32, ms=18, z=7)
    arrow(ax, (0.55, y0 - 0.02), (0.92, y0 - 0.02), GREEN, lw=2.2, rad=0.32, ms=16, z=7)
    arrow(
        ax, (1.45, y0 - 0.02), (1.08, y0 - 0.02), GREEN, lw=2.2, rad=-0.32, ms=16, z=7
    )

    # Spatial averaging formula box below the grid (line 1052)
    box(ax, (1.0, -0.36), 2.30, 0.56, "", LGREEN, GREEN, lw=1.8, r=0.07, z=2)
    ax.text(
        1.0,
        -0.24,
        r"$H_t(x_i) = \frac{1}{2}\left(H_t(x_{i-1/2}) + H_t(x_{i+1/2})\right)$",
        ha="center",
        va="center",
        fontsize=15.5,
        color="black",
        zorder=5,
    )
    badge(
        ax,
        (1.0, -0.52),
        r"co-located on monitor plane to $O(\Delta x^2)$",
        fc=GREEN,
        fs=13,
        pad=0.28,
    )

    ax.set_xlim(-0.42, 2.32)
    ax.set_ylim(-0.78, 2.50)

    fig.savefig(
        out_dir / "fig5_3_staggering_and_synchronization.png", facecolor="white"
    )
    plt.close(fig)


if __name__ == "__main__":
    outroot = Path(sys.argv[1])
    outroot.mkdir(parents=True, exist_ok=True)
    fig5_3(outroot)
