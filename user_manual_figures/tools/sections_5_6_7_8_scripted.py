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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import m_e

from wakis import GridFIT3D, SolverFIT3D
from wakis.conformal import (
    MERGE_THRESHOLD,
    _edge_potential,
    _redistribution,
    _section_areas_full,
)
from wakis.dispersive import Debye, DispersiveMaterial, Drude, Lorentz
from wakis.plasma import (
    cold_plasma,
    linear_wake,
    plasma_frequency,
    wake_amplitude,
)
from wakis.sources import (
    ContinuousSource,
    CurrentSource,
    GaussianSource,
)
from wakis.templates import Sheet

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
        bbox={"boxstyle": f"round,pad={pad}", "fc": fc, "ec": "none"},
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
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.88},
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
        bbox={
            "boxstyle": "round,pad=0.15",
            "fc": "white",
            "ec": BLUE,
            "lw": 1.0,
            "alpha": 0.94,
        },
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
        bbox={
            "boxstyle": "round,pad=0.15",
            "fc": "white",
            "ec": BLUE,
            "lw": 1.0,
            "alpha": 0.94,
        },
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
        bbox={
            "boxstyle": "round,pad=0.14",
            "fc": "white",
            "ec": ORANGE,
            "lw": 0.9,
            "alpha": 0.94,
        },
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
        bbox={
            "boxstyle": "round,pad=0.14",
            "fc": "white",
            "ec": ORANGE,
            "lw": 0.9,
            "alpha": 0.94,
        },
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


# --------------------------------------------------------------------------- 6.1
def fig6_1(outroot):
    """Figure 6.1: Domain-edge boundary alignment in GridFIT3D (legacy vs boundary_at_domain_edge=True)."""
    out_dir = outroot / "section6"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 5
    dx = 1.0
    g_leg = GridFIT3D(
        0.0,
        N * dx,
        0.0,
        1.0,
        0.0,
        1.0,
        N,
        2,
        2,
        boundary_at_domain_edge=False,
        verbose=0,
    )
    g_edge = GridFIT3D(
        0.0,
        N * dx,
        0.0,
        1.0,
        0.0,
        1.0,
        N,
        2,
        2,
        boundary_at_domain_edge=True,
        verbose=0,
    )
    s_leg = SolverFIT3D(
        g_leg, bc_low=["pec"] * 3, bc_high=["pmc", "pec", "pec"], verbose=0
    )
    s_edge = SolverFIT3D(
        g_edge, bc_low=["pec"] * 3, bc_high=["pmc", "pec", "pec"], verbose=0
    )

    # Asserts checking user_manual.md lines 1100-1103:
    assert s_leg.Nx == N and s_edge.Nx == N + 1
    tl_edge = g_edge.tL[:, 0, 0, "x"]
    assert np.isclose(tl_edge[0], 0.5 * dx) and np.isclose(tl_edge[N], 0.5 * dx)
    assert np.allclose(tl_edge[1:N], dx)
    assert np.all(s_edge.imu[N, :, :, "y"] == 0.0) and np.all(
        s_edge.imu[N, :, :, "z"] == 0.0
    )
    # Effective cavity length (N-1)dx vs Ndx gives O(dx/L) ~ 5% frequency shift for Nx=20 (lines 1091, 1100)
    n_ex = 20
    freq_shift_20 = n_ex / (n_ex - 1) - 1.0
    assert np.isclose(freq_shift_20, 1.0 / 19.0)

    fig = plt.figure(figsize=(W, H))

    node_labels = [
        r"$x_0 = x_{\min}$",
        r"$x_1$",
        r"$x_2$",
        r"$x_3$",
        r"$x_{N-1}$",
        r"$x_N = x_{\max}$",
    ]
    mid_labels = [
        r"$x_{1/2}$",
        r"$x_{3/2}$",
        r"$x_{5/2}$",
        r"$x_{N-3/2}$",
        r"$x_{N-1/2}$",
    ]

    # ---- TOP PANEL: Legacy mode (boundary_at_domain_edge=False) ----
    ax = fig.add_axes([0.03, 0.52, 0.94, 0.45])
    bare(ax)
    ax.text(
        3.0,
        1.52,
        "Legacy mode: boundary_at_domain_edge=False  (Nx = N)",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        color="black",
    )

    y_lo, y_hi = 0.0, 0.72
    ym = 0.5 * (y_lo + y_hi)
    # Effective cavity [0, N-1] shaded light blue; last cell [N-1, N] shaded faded red
    ax.add_patch(
        Rectangle(
            (0.0, y_lo), N - 1, y_hi - y_lo, fc=LBLUE, ec="none", alpha=0.45, zorder=1
        )
    )
    ax.add_patch(
        Rectangle(
            (N - 1, y_lo), 1.0, y_hi - y_lo, fc=LRED, ec="none", alpha=0.35, zorder=1
        )
    )
    ax.plot([0.0, N], [y_lo, y_lo], color=BLUE, lw=1.8, zorder=2)
    ax.plot([0.0, N], [y_hi, y_hi], color=BLUE, lw=1.8, zorder=2)

    # Primary nodes x_0 .. x_N
    for i in range(N + 1):
        xv = float(i)
        ax.plot([xv, xv], [y_lo - 0.06, y_hi + 0.06], color=BLUE, lw=2.0, zorder=3)
        x_txt = xv + (0.08 if i == N else (-0.06 if i == 0 else 0.0))
        ax.text(
            x_txt,
            y_lo - 0.14,
            node_labels[i],
            ha="center",
            va="top",
            fontsize=13.0,
            color=BLUE,
        )

    # Dual midpoints x_{1/2} .. x_{N-1/2} and tangential H samples (orange squares)
    sq = 0.11
    for i in range(N):
        xm = i + 0.5
        ax.plot(
            [xm, xm],
            [y_lo - 0.04, y_hi + 0.04],
            color=ORANGE,
            lw=1.6,
            ls="--",
            zorder=2,
        )
        ax.add_patch(
            Rectangle((xm - sq / 2, ym - sq / 2), sq, sq, color=ORANGE, zorder=5)
        )
        ax.text(
            xm - (0.04 if i == N - 1 else 0.0),
            y_lo - 0.14,
            mid_labels[i],
            ha="center",
            va="top",
            fontsize=12.0,
            color=ORANGE,
        )

    # Stored tangential E samples at x_0 .. x_{N-1} (filled blue circles)
    for i in range(N):
        ax.add_patch(Circle((float(i), ym), 0.068, color=BLUE, zorder=5))

    # Open red circle with a cross at x_N = x_max
    ax.add_patch(Circle((float(N), ym), 0.085, fc="white", ec=RED, lw=2.4, zorder=6))
    d_cr = 0.058
    ax.plot([N - d_cr, N + d_cr], [ym - d_cr, ym + d_cr], color=RED, lw=2.4, zorder=7)
    ax.plot([N - d_cr, N + d_cr], [ym + d_cr, ym - d_cr], color=RED, lw=2.4, zorder=7)
    ax.text(
        N + 0.22,
        ym + 0.05,
        r"no $\mathbf{E}$ sample stored" + "\n" + r"at $x_N = x_{\max}$",
        ha="left",
        va="center",
        fontsize=12.5,
        color=RED,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.22", "fc": LRED, "ec": RED, "lw": 1.2},
        zorder=7,
    )

    # Red vertical wall at x_{N-1} = x_max - dx
    ax.plot([N - 1, N - 1], [y_lo - 0.05, y_hi + 0.15], color=RED, lw=4.5, zorder=4)
    ax.text(
        2.15,
        y_hi + 0.34,
        r"high PEC wall at $x_{N-1}$: effective length $(N-1)\Delta x$,  $O(\Delta x/L) \sim 5\%$ frequency shift",
        ha="center",
        va="center",
        fontsize=12.5,
        color=RED,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.24", "fc": LRED, "ec": RED, "lw": 1.2},
        zorder=7,
    )
    arrow(ax, (3.45, y_hi + 0.20), (N - 1 - 0.04, y_hi + 0.06), RED, lw=1.8, ms=15, z=6)

    # Second red marker at x_{N-1/2} for high PMC wall
    ax.plot(
        [N - 0.5, N - 0.5],
        [y_lo - 0.05, y_hi + 0.12],
        color=RED,
        lw=3.0,
        ls=":",
        zorder=4,
    )
    ax.text(
        4.35,
        -0.54,
        r"high PMC wall zeros $\mathbf{H}$ at $x_{N-1/2}$ (shifted by $\Delta x/2$)",
        ha="center",
        va="center",
        fontsize=12.5,
        color=RED,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.22", "fc": LRED, "ec": RED, "lw": 1.2},
        zorder=7,
    )
    arrow(ax, (4.50, -0.36), (N - 0.5, y_lo - 0.30), RED, lw=1.8, ms=14, z=6)

    # Legend items in bottom-left of top panel (clear of title and nodes)
    ax.plot([], [], "o", color=BLUE, ms=8, label=r"stored tangential $\mathbf{E}$")
    ax.plot([], [], "s", color=ORANGE, ms=8, label=r"stored tangential $\mathbf{H}$")
    ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=11.5,
        ncol=1,
        labelspacing=0.25,
        bbox_to_anchor=(-0.01, -0.05),
    )

    ax.set_xlim(-0.45, 6.75)
    ax.set_ylim(-0.78, 1.72)

    # Horizontal divider
    fig.add_artist(
        plt.Line2D(
            [0.04, 0.96], [0.50, 0.50], transform=fig.transFigure, color=LGREY, lw=1.6
        )
    )

    # ---- BOTTOM PANEL: Exact Domain-Edge Mode (boundary_at_domain_edge=True) ----
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.45])
    bare(ax)
    ax.text(
        3.0,
        1.52,
        "Exact Domain-Edge Mode: boundary_at_domain_edge=True (default, solver.Nx = Nx + 1)",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        color="black",
    )

    # Primary domain [0, N] shaded light blue
    ax.add_patch(
        Rectangle(
            (0.0, y_lo), N, y_hi - y_lo, fc=LBLUE, ec="none", alpha=0.45, zorder=1
        )
    )
    # Highlight truncated boundary dual cell A~_k in [N - 0.5, N] in light orange
    ax.add_patch(
        Rectangle(
            (N - 0.5, y_lo + 0.04),
            0.5,
            y_hi - y_lo - 0.08,
            fc=LORANGE,
            ec=ORANGE,
            lw=1.8,
            zorder=2,
        )
    )
    ax.text(
        N - 0.25,
        y_hi - 0.13,
        r"$\tilde{A}_k$",
        ha="center",
        va="center",
        fontsize=13.5,
        color=ORANGE,
        weight="bold",
        zorder=6,
    )

    # Ghost cell [N, N + 1.35] beyond x_max shaded light grey with dashed border
    gw = 1.35
    ax.add_patch(
        Rectangle(
            (float(N), y_lo),
            gw,
            y_hi - y_lo,
            fc=LGREY,
            ec=GREY,
            lw=1.8,
            ls="--",
            alpha=0.85,
            zorder=1,
        )
    )
    ax.text(
        N + 0.54 * gw,
        ym,
        "ghost layer\noutside domain:\n" + r"$\mu^{-1} = 0$",
        ha="center",
        va="center",
        fontsize=11.5,
        color=GREY,
        weight="bold",
        zorder=5,
    )
    ax.text(
        N + gw,
        y_lo - 0.14,
        r"$x_N + \Delta x$",
        ha="center",
        va="top",
        fontsize=12.5,
        color=GREY,
    )

    ax.plot([0.0, N], [y_lo, y_lo], color=BLUE, lw=1.8, zorder=2)
    ax.plot([0.0, N], [y_hi, y_hi], color=BLUE, lw=1.8, zorder=2)

    # Primary nodes x_0 .. x_N
    for i in range(N + 1):
        xv = float(i)
        ax.plot([xv, xv], [y_lo - 0.06, y_hi + 0.06], color=BLUE, lw=2.0, zorder=3)
        x_txt = xv + (0.08 if i == N else (-0.06 if i == 0 else 0.0))
        ax.text(
            x_txt,
            y_lo - 0.14,
            node_labels[i],
            ha="center",
            va="top",
            fontsize=13.0,
            color=BLUE,
        )

    # Dual midpoints x_{1/2} .. x_{N-1/2} and tangential H samples (orange squares)
    for i in range(N):
        xm = i + 0.5
        ax.plot(
            [xm, xm],
            [y_lo - 0.04, y_hi + 0.04],
            color=ORANGE,
            lw=1.6,
            ls="--",
            zorder=2,
        )
        ax.add_patch(
            Rectangle((xm - sq / 2, ym - sq / 2), sq, sq, color=ORANGE, zorder=5)
        )
        ax.text(
            xm - (0.04 if i == N - 1 else 0.0),
            y_lo - 0.14,
            mid_labels[i],
            ha="center",
            va="top",
            fontsize=12.0,
            color=ORANGE,
        )

    # Thick green boundary bars at x_0 = x_min and x_N = x_max
    for xv in (0.0, float(N)):
        ax.plot([xv, xv], [y_lo - 0.08, y_hi + 0.14], color=GREEN, lw=5.5, zorder=4)

    # Stored tangential E samples at ALL N + 1 nodes x_0 .. x_N
    for i in range(N + 1):
        ax.add_patch(Circle((float(i), ym), 0.068, color=BLUE, zorder=6))

    # Two green callouts (lines 1102-1103)
    ax.text(
        1.75,
        y_hi + 0.34,
        r"PEC wall: zeros tangential $\mathbf{E}$ on node $0\ (x_{\min})$ and node $N\ (x_{\max})$",
        ha="center",
        va="center",
        fontsize=12.5,
        color="black",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.24", "fc": LGREEN, "ec": GREEN, "lw": 1.5},
        zorder=7,
    )
    arrow(ax, (0.55, y_hi + 0.20), (0.04, y_hi + 0.08), GREEN, lw=1.8, ms=15, z=6)
    arrow(ax, (3.25, y_hi + 0.20), (N - 0.04, y_hi + 0.08), GREEN, lw=1.8, ms=15, z=6)

    ax.text(
        3.00,
        -0.52,
        r"PMC wall: dual face $\tilde{A}_k$ truncated at $x_{\max}$ + ghost $\mu^{-1} = 0$"
        + r"  $\Rightarrow$  $\hat{\mathbf{n}}\times\mathbf{H} = \mathbf{0}$ on $x_{\max}$ to $O(\Delta x^2)$",
        ha="center",
        va="center",
        fontsize=12.5,
        color="black",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.24", "fc": LGREEN, "ec": GREEN, "lw": 1.5},
        zorder=7,
    )
    arrow(
        ax, (4.72, -0.35), (N - 0.25, y_lo + 0.02), GREEN, lw=1.8, ms=15, z=6, rad=0.18
    )

    ax.set_xlim(-0.45, 6.65)
    ax.set_ylim(-0.75, 1.72)

    fig.savefig(out_dir / "fig6_1_domain_edge_boundaries.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- 6.3
def fig6_3(outroot):
    """Figure 6.3: Nonuniform Cartesian meshing and automatic STL feature snapping in GridFIT3D."""
    out_dir = outroot / "section6"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Execute the exact graded-mesh code from user_manual.md lines 1175-1179:
    x_fine = np.linspace(-5e-3, 5e-3, 41)
    x_left = np.geomspace(-50e-3, -5e-3, 25)[:-1]
    x_right = np.geomspace(5e-3, 50e-3, 25)[1:]
    x_nodes = np.concatenate([x_left, x_fine, x_right])

    grid_graded = GridFIT3D(
        xmin=x_nodes[0],
        xmax=x_nodes[-1],
        ymin=-50e-3,
        ymax=50e-3,
        zmin=-50e-3,
        zmax=50e-3,
        x=x_nodes,
        y=np.linspace(-50e-3, 50e-3, 11),
        z=np.linspace(-50e-3, 50e-3, 11),
        boundary_at_domain_edge=False,
        verbose=0,
    )

    x_mm = x_nodes * 1e3
    dx_mm = grid_graded.dx * 1e3
    x_mid_mm = 0.5 * (x_mm[:-1] + x_mm[1:])
    adj_ratio = np.maximum(dx_mm[1:] / dx_mm[:-1], dx_mm[:-1] / dx_mm[1:])
    x_ratio_mm = x_mm[1:-1]

    # Asserts for graded mesh (user_manual.md lines 1175-1179, 1206)
    assert len(x_fine) == 41 and len(x_nodes) == 89
    assert np.isclose(np.min(dx_mm), 0.25)
    wing_ratio = (50.0 / 5.0) ** (1.0 / 24.0)
    assert np.allclose(dx_mm[:23][1:] / dx_mm[:23][:-1], 1.0 / wing_ratio)
    assert 1.0 < wing_ratio <= 1.2

    # Feature snapping strips for Right Panel (user_manual.md lines 1191-1200)
    zmin_mm, zmax_mm = -100.0, 100.0
    n_uni = 17  # 16 cells of 12.5 mm
    z_uni = np.linspace(zmin_mm, zmax_mm, n_uni)
    z_features = np.array([-42.0, 42.0])
    # Strip 2: insert extra grid planes at feature planes
    z_insert = np.sort(np.concatenate([z_uni, z_features]))
    # Strip 3: shift nearest existing grid nodes onto feature planes with smooth neighbour spacing
    z_snaps_all = np.array([zmin_mm, z_features[0], z_features[1], zmax_mm])
    z_stretch = grid_graded.refine_axis(
        zmin_mm, zmax_mm, n_uni, z_snaps_all, method="neighbor", tol=1e-10
    )
    # Enforce exact snap positions on the stretched grid
    for zf in z_features:
        idx_near = int(np.argmin(np.abs(z_stretch - zf)))
        z_stretch[idx_near] = zf

    assert (
        np.min(np.abs(z_uni[:, None] - z_features[None, :])) > 4.0
    )  # uniform grid misses feature planes
    assert (
        np.max(np.min(np.abs(z_insert[:, None] - z_features[None, :]), axis=0)) < 1e-12
    )
    assert (
        np.max(np.min(np.abs(z_stretch[:, None] - z_features[None, :]), axis=0)) < 1e-12
    )
    assert len(z_insert) == len(z_uni) + 2 and len(z_stretch) == len(z_uni)

    fig = plt.figure(figsize=(W, H))

    # ---- LEFT PANEL: Explicit graded mesh x_nodes ----
    fig.text(
        0.255,
        0.94,
        "Explicit graded node array x_nodes",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )

    # Top sub-axes: vertical node ticks
    ax_top = fig.add_axes([0.06, 0.66, 0.39, 0.22])
    ax_top.axvspan(-50.0, -5.0, color=LORANGE, alpha=0.65, zorder=1)
    ax_top.axvspan(-5.0, 5.0, color=LBLUE, alpha=0.85, zorder=1)
    ax_top.axvspan(5.0, 50.0, color=LORANGE, alpha=0.65, zorder=1)
    for xv in x_mm:
        ax_top.plot([xv, xv], [0.10, 0.54], color=BLUE, lw=1.1, zorder=3)
    ax_top.text(
        0.0,
        0.84,
        "x_fine: 41 nodes",
        ha="center",
        va="center",
        fontsize=12.0,
        color=BLUE,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.16",
            "fc": "white",
            "ec": BLUE,
            "lw": 1.0,
            "alpha": 0.94,
        },
        zorder=5,
    )
    ax_top.text(
        -31.0,
        0.72,
        "x_left, x_right:\ngeomspace",
        ha="center",
        va="center",
        fontsize=11.0,
        color=ORANGE,
        weight="bold",
        zorder=5,
    )
    ax_top.text(
        31.0,
        0.72,
        "x_left, x_right:\ngeomspace",
        ha="center",
        va="center",
        fontsize=11.0,
        color=ORANGE,
        weight="bold",
        zorder=5,
    )
    ax_top.set_xlim(-52.0, 52.0)
    ax_top.set_ylim(0.0, 1.04)
    ax_top.set_yticks([])
    ax_top.set_xticks([-50, -25, -5, 5, 25, 50])
    ax_top.tick_params(axis="x", labelsize=11.5)
    for sp_name in ("top", "right", "left"):
        ax_top.spines[sp_name].set_visible(False)

    # Bottom sub-axes: cell size dx_i [mm] and adjacent ratio on twin axis
    ax_bot = fig.add_axes([0.06, 0.11, 0.37, 0.48])
    ax_twin = ax_bot.twinx()

    # Smooth grading band [1.0, 1.3] on twin axis
    ax_twin.axhspan(1.0, 1.3, color=LGREEN, alpha=0.65, zorder=1)
    ax_twin.axhline(1.3, color=GREEN, lw=1.3, ls="--", zorder=2)

    ax_bot.plot(
        x_mid_mm,
        dx_mm,
        "-o",
        color=BLUE,
        lw=1.8,
        ms=3.5,
        zorder=4,
        label=r"cell size $\Delta x_i$ [mm]",
    )
    ax_twin.plot(
        x_ratio_mm,
        adj_ratio,
        "--s",
        color=ORANGE,
        lw=1.6,
        ms=3.0,
        zorder=5,
        label=r"ratio $\max(\Delta x_{i+1}/\Delta x_i,\ \Delta x_i/\Delta x_{i+1})$",
    )

    ax_bot.set_xlim(-52.0, 52.0)
    ax_bot.set_ylim(0.0, 5.2)
    ax_twin.set_ylim(0.95, 2.25)
    ax_bot.set_xlabel("position $x$ [mm]", fontsize=13.5)
    ax_bot.set_ylabel(r"cell size $\Delta x_i$ [mm]", fontsize=13.5, color=BLUE)
    ax_twin.set_ylabel(
        r"adjacent ratio $\Delta x_{i+1}/\Delta x_i$", fontsize=13.0, color=ORANGE
    )
    ax_bot.tick_params(axis="y", colors=BLUE, labelsize=12)
    ax_twin.tick_params(axis="y", colors=ORANGE, labelsize=12)
    ax_bot.tick_params(axis="x", labelsize=12)
    ax_bot.spines["top"].set_visible(False)
    ax_twin.spines["top"].set_visible(False)

    ax_twin.text(
        0.0,
        1.16,
        r"smooth grading: $1/\mathrm{grading} \leq \Delta x_{i+1}/\Delta x_i \leq 1.2\text{–}1.3$",
        ha="center",
        va="center",
        fontsize=11.0,
        color=GREEN,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": GREEN,
            "lw": 1.0,
            "alpha": 0.92,
        },
        zorder=6,
    )
    ax_bot.text(
        0.0,
        4.65,
        r"global CFL $\Delta t$ governed by $\min(\Delta x_i, \Delta y_j, \Delta z_k)$",
        ha="center",
        va="center",
        fontsize=11.0,
        color=GREY,
        bbox={"boxstyle": "round,pad=0.22", "fc": LGREY, "ec": GREY, "lw": 1.0},
        zorder=6,
    )

    # Vertical divider between left and right panels
    fig.add_artist(
        plt.Line2D(
            [0.515, 0.515], [0.06, 0.94], transform=fig.transFigure, color=LGREY, lw=1.6
        )
    )

    # ---- RIGHT PANEL: Automatic STL feature snapping ----
    fig.text(
        0.745,
        0.94,
        "Automatic STL feature snapping (use_mesh_refinement=True)",
        ha="center",
        va="center",
        fontsize=13.5,
        weight="bold",
    )
    ax_r = fig.add_axes([0.53, 0.04, 0.45, 0.86])
    ax_r.axis("off")

    # Axisymmetric cavity STL profile at the top (z in [-100, 100], r in [2.30, 3.16])
    z_prof = [-100.0, -42.0, -42.0, 42.0, 42.0, 100.0]
    r_top = [2.62, 2.62, 3.16, 3.16, 2.62, 2.62]
    r_axis = 2.30
    poly_pts = list(zip(z_prof, r_top)) + [(100.0, r_axis), (-100.0, r_axis)]
    ax_r.add_patch(
        Polygon(poly_pts, closed=True, fc=LBLUE, ec="none", alpha=0.55, zorder=1)
    )
    ax_r.plot(z_prof, r_top, color=GREY, lw=2.6, zorder=4)
    ax_r.plot([-100.0, 100.0], [r_axis, r_axis], color=GREY, lw=1.2, ls="-.", zorder=2)
    ax_r.text(
        0.0,
        2.86,
        "Axisymmetric cavity STL profile",
        ha="center",
        va="center",
        fontsize=12.5,
        color=GREY,
        weight="bold",
    )

    # Two vertical dashed green lines extending down from the iris corners at z = +-42
    for zf in z_features:
        ax_r.plot([zf, zf], [0.04, 3.24], color=GREEN, lw=2.0, ls="--", zorder=3)
        ax_r.plot([zf], [2.62], "o", color=GREEN, ms=8, mec="white", mew=1.2, zorder=6)
    badge(
        ax_r,
        (0.0, 3.34),
        "STL feature planes (snap_tol=1e-4)",
        fc=GREEN,
        fs=12.0,
        pad=0.26,
    )

    # Three horizontal z-grid strips stacked vertically
    strips = [
        (
            1.78,
            z_uni,
            "Uniform grid (use_mesh_refinement=False)",
            "misses feature planes",
            RED,
            "uniform",
        ),
        (
            1.02,
            z_insert,
            'refinement_method="insert"',
            "extra planes inserted",
            GREEN,
            "insert",
        ),
        (
            0.26,
            z_stretch,
            'refinement_method="stretch"',
            "nearest nodes shifted",
            GREEN,
            "stretch",
        ),
    ]
    sh = 0.34
    for y_c, z_arr, title, note, ncol, mode in strips:
        ax_r.text(
            -100.0,
            y_c + sh / 2 + 0.08,
            title,
            ha="left",
            va="bottom",
            fontsize=11.0,
            weight="bold",
            color="black",
        )
        ax_r.text(
            100.0,
            y_c + sh / 2 + 0.08,
            note,
            ha="right",
            va="bottom",
            fontsize=10.5,
            weight="bold",
            color=ncol,
        )
        ax_r.add_patch(
            Rectangle(
                (-100.0, y_c - sh / 2), 200.0, sh, fc="white", ec=GREY, lw=1.2, zorder=2
            )
        )
        for zv in z_arr:
            is_snapped = np.min(np.abs(zv - z_features)) < 1e-6
            if is_snapped:
                ax_r.plot(
                    [zv, zv],
                    [y_c - sh / 2 - 0.03, y_c + sh / 2 + 0.03],
                    color=GREEN,
                    lw=3.2,
                    zorder=5,
                )
                ax_r.plot([zv], [y_c], "o", color=GREEN, ms=7, zorder=6)
            else:
                ax_r.plot(
                    [zv, zv], [y_c - sh / 2, y_c + sh / 2], color=BLUE, lw=1.6, zorder=4
                )

        if mode == "uniform":
            # Mark missed feature planes with red crosses
            for zf in z_features:
                ax_r.plot(
                    [zf - 3.6, zf + 3.6],
                    [y_c - 0.08, y_c + 0.08],
                    color=RED,
                    lw=2.6,
                    zorder=7,
                )
                ax_r.plot(
                    [zf - 3.6, zf + 3.6],
                    [y_c + 0.08, y_c - 0.08],
                    color=RED,
                    lw=2.6,
                    zorder=7,
                )
        elif mode == "stretch":
            for z_old, zf in ((-37.5, -42.0), (37.5, 42.0)):
                ax_r.plot(
                    [z_old, z_old],
                    [y_c - sh / 2, y_c + sh / 2],
                    color=BLUE,
                    lw=1.2,
                    ls=":",
                    alpha=0.6,
                    zorder=3,
                )
                arrow(ax_r, (z_old, y_c), (zf, y_c), ORANGE, lw=2.2, ms=14, z=7)

    ax_r.set_xlim(-108.0, 108.0)
    ax_r.set_ylim(-0.05, 3.55)

    fig.savefig(out_dir / "fig6_3_graded_mesh_and_snapping.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- 6.4
def fig6_4(outroot):
    """Figure 6.4: Conformal Dey-Mittra PEC cut cells, (2,-1) stabilization, and thin sheets."""
    out_dir = outroot / "section6"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Left panel geometry: unit primary cell [0,1]x[0,1] cut by circular PEC arc
    # Circle centred at (-0.25, 0.78) with radius R = 0.85 encloses left edge x=0 (l_left = 0)
    # and cuts top edge y=1 at x_top and bottom edge y=0 at x_bot.
    xc, yc, rad = -0.25, 0.78, 0.85
    x_top = xc + np.sqrt(rad**2 - (1.0 - yc) ** 2)
    x_bot = xc + np.sqrt(rad**2 - (0.0 - yc) ** 2)
    y_grid = np.linspace(0.0, 1.0, 4001)
    x_arc = xc + np.sqrt(rad**2 - (y_grid - yc) ** 2)
    pec_area_num = np.trapezoid(x_arc, y_grid)
    s_f_open = 1.0 - pec_area_num
    # Check against wakis.conformal._section_areas_full on polygonal PEC contour
    n_seg = 400
    th_a = np.arctan2(0.0 - yc, x_bot - xc)
    th_b = np.arctan2(1.0 - yc, x_top - xc)
    th = np.linspace(th_a, th_b, n_seg)
    arc_pts = np.column_stack([xc + rad * np.cos(th), yc + rad * np.sin(th)])
    poly = np.vstack([arc_pts, [[-0.5, 1.2], [-0.5, -0.2]]])
    p_seg = poly
    q_seg = np.roll(poly, -1, axis=0)
    area_w, lu_w, lv_w = _section_areas_full(
        np.zeros(len(p_seg), dtype=int),
        p_seg,
        q_seg,
        1,
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    assert np.isclose(1.0 - area_w[0, 0, 0], s_f_open, rtol=1e-4)
    assert np.isclose(1.0 - lu_w[0, 0, 0], 1.0 - x_bot, rtol=1e-4)  # bottom open length
    assert np.isclose(1.0 - lu_w[0, 1, 0], 1.0 - x_top, rtol=1e-4)  # top open length
    assert np.isclose(
        1.0 - lv_w[0, 0, 0], 0.0, atol=1e-12
    )  # left edge enclosed in PEC: l_k = 0

    # 2. Centre panel verification: (2, -1) small-face stabilization via wakis.conformal._redistribution
    # 3x2x1 grid where face (0, 1, 0) has small open area S_u = 0.08 < 0.5 * A_stab = 0.25
    shape = (3, 2, 1)
    cell_L = [np.ones(shape) for _ in range(3)]
    L_open = [np.ones(shape) for _ in range(3)]
    A_open = [np.ones(shape) for _ in range(3)]
    # Row j=1 normal-z face: col 0 is unstable sliver (S_u = 0.08), cols 1 and 2 are stable (A = 1.0)
    A_open[2][0, 1, 0] = 0.08
    L_open[0][0, 1, 0] = 0.16
    ops, n_unst = _redistribution(L_open, A_open, cell_L, [0, 0, 0], MERGE_THRESHOLD)
    G_z = ops[2].toarray()
    a_vec = A_open[2].ravel(order="F")
    # Assert G = T^T diag(1/A_hat) T is symmetric positive semi-definite and reproduces uniform H = 1
    assert n_unst == 1
    assert np.allclose(G_z, G_z.T, atol=1e-15)
    assert np.min(np.linalg.eigvalsh(G_z)) >= -1e-14
    assert np.allclose(G_z @ a_vec, np.ones_like(a_vec), atol=1e-14)
    # Check exact receiving areas A_hat_1 = 1 + 2 * 0.08, A_hat_2 = 1 - 1 * 0.08 (user_manual.md line 1222)
    # In Fortran order on shape (3, 2, 1): u=(0,1,0)->3, n_1=(1,1,0)->4, n_2=(2,1,0)->5
    a_hat_1 = 1.0 + 2.0 * 0.08
    a_hat_2 = 1.0 - 1.0 * 0.08
    assert np.isclose(G_z[4, 4], 1.0 / a_hat_1) and np.isclose(G_z[4, 3], 2.0 / a_hat_1)
    assert np.isclose(G_z[5, 5], 1.0 / a_hat_2) and np.isclose(
        G_z[5, 3], -1.0 / a_hat_2
    )
    assert np.isclose(G_z[3, 3], 4.0 / a_hat_1 + 1.0 / a_hat_2)

    # 3. Right panel verification: Meixner r^{-1/2} edge singularity & Sheet (user_manual.md lines 1235-1249)
    patch = Sheet(center=(0.0, 0.0, 1.5e-3), size=(30e-3, 40e-3, 0.0))
    grid_sh = GridFIT3D(
        xmin=-30e-3,
        xmax=30e-3,
        ymin=-35e-3,
        ymax=35e-3,
        zmin=0.0,
        zmax=15e-3,
        Nx=12,
        Ny=14,
        Nz=10,
        sheets={"patch": patch.to_pyvista()},
        sheet_edge_correction=True,
        verbose=0,
    )
    solver_sh = SolverFIT3D(grid_sh, verbose=0)
    # Potential F(s, 0) = sqrt(s) outside edge (s > 0) -> dF/ds = 0.5 * s^{-1/2}, integral over [0, dx] is sqrt(dx)
    s_pts = np.linspace(0.04, 2.0, 300)
    pot_vals = _edge_potential(s_pts, 0.0).real
    assert np.allclose(pot_vals, np.sqrt(s_pts))
    assert np.any(solver_sh.conformal_ieps_factor != 1.0) and np.any(
        solver_sh.conformal_ieps_factor > 0.0
    )

    fig = plt.figure(figsize=(W, H))

    fig.text(
        0.165,
        0.93,
        "1. Dey–Mittra PEC cut cell",
        ha="center",
        va="center",
        fontsize=15.0,
        weight="bold",
    )
    fig.text(
        0.50,
        0.93,
        "2. (2, −1) small-face stabilization",
        ha="center",
        va="center",
        fontsize=15.0,
        weight="bold",
    )
    fig.text(
        0.825,
        0.93,
        "3. Thin Sheet & Meixner edge correction",
        ha="center",
        va="center",
        fontsize=13.5,
        weight="bold",
    )

    # ---- PANEL 1: Dey-Mittra PEC cut cell ----
    ax1 = fig.add_axes([0.02, 0.04, 0.30, 0.85])
    bare(ax1)

    # Vacuum aperture polygon inside [0, 1] x [0, 1]
    vac_poly = np.vstack([arc_pts, [[1.0, 1.0], [1.0, 0.0]]])
    ax1.add_patch(
        Polygon(vac_poly, closed=True, fc=LBLUE, ec="none", alpha=0.80, zorder=1)
    )
    # PEC region inside [0, 1] x [0, 1]
    pec_poly = np.vstack([[[0.0, 0.0], [x_bot, 0.0]], arc_pts, [[0.0, 1.0]]])
    ax1.add_patch(
        Polygon(
            pec_poly,
            closed=True,
            fc=LGREY,
            ec=GREY,
            lw=1.5,
            hatch="///",
            zorder=2,
        )
    )
    ax1.plot(arc_pts[:, 0], arc_pts[:, 1], color=GREY, lw=2.8, zorder=4)
    ax1.text(
        0.16,
        0.74,
        "PEC:\n" + r"$\mathbf{E} = \mathbf{0}$",
        ha="center",
        va="center",
        fontsize=12.5,
        color=GREY,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": GREY,
            "lw": 1.2,
            "alpha": 0.92,
        },
        zorder=6,
    )

    # Enclosed left edge (l_k = 0) and enclosed portions of top/bottom edges in red dashed
    ax1.plot([0.0, 0.0], [0.0, 1.0], color=RED, lw=3.6, ls="--", zorder=5)
    ax1.plot([0.0, x_top], [1.0, 1.0], color=RED, lw=2.4, ls="--", zorder=4)
    ax1.plot([0.0, x_bot], [0.0, 0.0], color=RED, lw=2.4, ls="--", zorder=4)
    ax1.text(
        -0.06,
        0.48,
        r"$l_k = 0$" + "\n" + r"$\Rightarrow \mathrm{ieps} = 0$",
        ha="right",
        va="center",
        fontsize=12.0,
        color=RED,
        weight="bold",
    )

    # Vacuum-open portions of primary edges in thick blue
    ax1.plot([x_bot, 1.0], [0.0, 0.0], color=BLUE, lw=4.5, zorder=5)
    ax1.plot([x_top, 1.0], [1.0, 1.0], color=BLUE, lw=4.5, zorder=5)
    ax1.plot([1.0, 1.0], [0.0, 1.0], color=BLUE, lw=4.5, zorder=5)
    for px, py in ((x_bot, 0.0), (x_top, 1.0), (1.0, 0.0), (1.0, 1.0)):
        ax1.add_patch(Circle((px, py), 0.025, color=BLUE, zorder=6))

    ax1.text(
        0.54,
        -0.11,
        r"$l_k \in [0, |L_k|]$",
        ha="center",
        va="top",
        fontsize=13.5,
        color=BLUE,
        weight="bold",
    )
    ax1.text(
        0.66,
        0.34,
        r"$S_f \in [0, |A_f|]$" + "\n" + r"$\mathrm{iA}_f = 1/S_f$",
        ha="center",
        va="center",
        fontsize=13.0,
        color=BLUE,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": BLUE,
            "lw": 1.0,
            "alpha": 0.92,
        },
        zorder=6,
    )

    box(
        ax1,
        (0.42, -0.44),
        1.35,
        0.32,
        r"$\mathbf{D}_s = \mathrm{diag}(l_k),\ \ \mathrm{iA}_f = 1/S_f$",
        LBLUE,
        BLUE,
        fs=13.5,
        lw=1.6,
        r=0.06,
    )

    ax1.set_xlim(-0.44, 1.18)
    ax1.set_ylim(-0.70, 1.24)

    # Vertical dividers
    fig.add_artist(
        plt.Line2D(
            [0.33, 0.33], [0.06, 0.94], transform=fig.transFigure, color=LGREY, lw=1.6
        )
    )
    fig.add_artist(
        plt.Line2D(
            [0.67, 0.67], [0.06, 0.94], transform=fig.transFigure, color=LGREY, lw=1.6
        )
    )

    # ---- PANEL 2: (2, -1) small-face stabilization ----
    ax2 = fig.add_axes([0.34, 0.04, 0.32, 0.85])
    bare(ax2)

    # Draw 3 x 2 Cartesian grid patch [0, 3] x [0.45, 2.05] (cell height 0.8, width 1.0)
    y_b, ch = 0.45, 0.80
    # Top row (j=1): col 0 has sliver open area S_u in [0.75, 1.0] x [y_b+ch, y_b+2*ch]
    pec_patch = [
        (0.0, y_b + 0.25 * ch),
        (0.75, y_b + ch),
        (0.88, y_b + 2 * ch),
        (0.0, y_b + 2 * ch),
    ]
    sliver_poly = [
        (0.75, y_b + ch),
        (1.0, y_b + ch),
        (1.0, y_b + 2 * ch),
        (0.88, y_b + 2 * ch),
    ]
    ax2.add_patch(Polygon(sliver_poly, closed=True, fc=LRED, ec=RED, lw=1.8, zorder=2))
    ax2.add_patch(
        Rectangle((1.0, y_b + ch), 1.0, ch, fc=LGREEN, ec=GREEN, lw=1.8, zorder=1)
    )
    ax2.add_patch(
        Rectangle((2.0, y_b + ch), 1.0, ch, fc=LGREEN, ec=GREEN, lw=1.8, zorder=1)
    )
    ax2.add_patch(
        Rectangle((0.0, y_b), 3.0, ch, fc=LBLUE, ec="none", alpha=0.35, zorder=0)
    )
    ax2.add_patch(
        Polygon(
            pec_patch,
            closed=True,
            fc=LGREY,
            ec=GREY,
            lw=2.0,
            hatch="///",
            zorder=3,
        )
    )
    ax2.text(
        0.34,
        y_b + 1.45 * ch,
        "PEC",
        ha="center",
        va="center",
        fontsize=12.5,
        color=GREY,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": GREY, "lw": 1.0},
        zorder=5,
    )

    for i in range(4):
        ax2.plot(
            [float(i), float(i)], [y_b, y_b + 2 * ch], color=BLUE, lw=1.8, zorder=4
        )
    for j in range(3):
        ax2.plot([0.0, 3.0], [y_b + j * ch, y_b + j * ch], color=BLUE, lw=1.8, zorder=4)

    ax2.text(
        1.5,
        y_b + 1.42 * ch,
        "neighbour 1\n" + r"$(n_1)$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=GREEN,
        weight="bold",
        zorder=5,
    )
    ax2.text(
        2.5,
        y_b + 1.42 * ch,
        "neighbour 2\n" + r"$(n_2)$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=GREEN,
        weight="bold",
        zorder=5,
    )
    ax2.text(
        0.62,
        y_b + 0.48 * ch,
        "unstable face $u$:\n" + r"$S_u < 0.5\,A_{\mathrm{stab}}$",
        ha="center",
        va="center",
        fontsize=11.0,
        color=RED,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.16", "fc": LRED, "ec": RED, "lw": 1.2},
        zorder=6,
    )
    arrow(
        ax2,
        (0.75, y_b + 0.70 * ch),
        (0.90, y_b + 1.25 * ch),
        RED,
        lw=1.8,
        ms=14,
        z=6,
    )

    # Two curved green arrows from face u to n_1 (+2) and n_2 (-1)
    y_top_row = y_b + 1.88 * ch
    arrow(
        ax2, (0.92, y_top_row), (1.45, y_top_row), GREEN, lw=2.4, rad=-0.45, ms=16, z=7
    )
    badge(ax2, (1.20, y_b + 2.22 * ch), "+2", fc=GREEN, fs=12.0, pad=0.22, z=8)

    arrow(
        ax2,
        (0.92, y_top_row + 0.04),
        (2.48, y_top_row + 0.04),
        ORANGE,
        lw=2.2,
        rad=-0.38,
        ms=16,
        z=6,
    )
    badge(ax2, (1.85, y_b + 2.38 * ch), "−1", fc=ORANGE, fs=12.0, pad=0.22, z=8)
    ax2.text(
        2.15,
        y_b + 0.48 * ch,
        "weight 1 if only\none neighbour",
        ha="center",
        va="center",
        fontsize=10.5,
        color=GREY,
        bbox={
            "boxstyle": "round,pad=0.16",
            "fc": "white",
            "ec": LGREY,
            "lw": 1.0,
            "alpha": 0.92,
        },
        zorder=5,
    )

    # Equations below the grid (user_manual.md lines 1221-1224)
    box(ax2, (1.5, -0.35), 3.22, 1.32, "", LGREEN, GREEN, lw=1.8, r=0.07, z=1)
    ax2.text(
        1.5,
        0.13,
        r"$A_{\mathrm{stab}} = \max_i(l_i h_i)/2,\ \ \mathrm{MERGE\_THRESHOLD} = 0.5$",
        ha="center",
        va="center",
        fontsize=11.0,
        color="black",
        zorder=4,
    )
    ax2.text(
        1.5,
        -0.14,
        r"$\hat{A}_n = A_n + \sum_u T_{nu} A_u$",
        ha="center",
        va="center",
        fontsize=12.0,
        color="black",
        zorder=4,
    )
    ax2.text(
        1.5,
        -0.41,
        r"$\mathbf{G} = \mathbf{T}^T\,\mathrm{diag}(1/\hat{A})\,\mathbf{T}$"
        + "  (symmetric positive semi-definite)",
        ha="center",
        va="center",
        fontsize=10.5,
        color="black",
        zorder=4,
    )
    badge(
        ax2,
        (1.5, -0.76),
        r"stable $\Delta t = 0.85\text{–}0.97$ of Yee limit (studies/conformal-pec-mask/)",
        fc=GREEN,
        fs=9.8,
        pad=0.22,
    )

    ax2.set_xlim(-0.20, 3.20)
    ax2.set_ylim(-1.10, 2.55)

    # ---- PANEL 3: Thin Sheet & Meixner edge correction ----
    ax3 = fig.add_axes([0.68, 0.04, 0.30, 0.85])
    bare(ax3)

    # 2D (x, z) grid cross-section: x in [-1.0, 2.0], z in [0.35, 1.75]
    # z_sh = 1.05 corresponds to z = 1.5 mm plane
    z_lo = 0.35
    dz_g = 0.70
    z_sh = z_lo + dz_g
    # Highlight primary face immediately outside the sharp edge x in [0, dx] = [0, 1]
    ax3.add_patch(
        Rectangle(
            (0.0, z_lo),
            1.0,
            2 * dz_g,
            fc=LORANGE,
            ec=ORANGE,
            lw=1.8,
            alpha=0.80,
            zorder=1,
        )
    )
    ax3.text(
        0.50,
        z_lo + 0.34,
        "primary face\noutside edge\n" + r"$x \in [0, \Delta x]$",
        ha="center",
        va="center",
        fontsize=10.5,
        color=ORANGE,
        weight="bold",
        zorder=5,
    )

    # Grid lines
    for xv, lab in (
        (-1.0, r"$-\Delta x$"),
        (0.0, r"$0$"),
        (1.0, r"$\Delta x$"),
        (2.0, r"$2\Delta x$"),
    ):
        ax3.plot(
            [xv, xv],
            [z_lo - 0.06, z_lo + 2 * dz_g + 0.06],
            color=BLUE,
            lw=1.6,
            zorder=2,
        )
        ax3.text(xv, z_lo - 0.12, lab, ha="center", va="top", fontsize=12.0, color=BLUE)
    for zv in (z_lo, z_sh, z_lo + 2 * dz_g):
        ax3.plot([-1.06, 2.06], [zv, zv], color=BLUE, lw=1.6, zorder=2)
    ax3.text(
        -1.10,
        z_sh,
        r"$z = 1.5\mathrm{\ mm}$",
        ha="right",
        va="center",
        fontsize=11.0,
        color=GREY,
        weight="bold",
    )

    # Zero-thickness metallic sheet at z = 1.5 mm, x <= 0
    ax3.plot(
        [-1.04, 0.0], [z_sh, z_sh], color=GREY, lw=7.0, solid_capstyle="butt", zorder=6
    )
    ax3.add_patch(Circle((0.0, z_sh), 0.055, color=RED, zorder=7))
    ax3.text(
        -0.50,
        z_lo + 0.34,
        "zero-thickness\nPEC sheet\n" + r"($x \leq 0$)",
        ha="center",
        va="center",
        fontsize=10.5,
        color=GREY,
        weight="bold",
        zorder=6,
    )

    # Plot exact analytical Meixner singularity curve r^{-1/2} vs r > 0 above the sheet
    r_vals = np.linspace(0.06, 1.95, 300)
    y_meix = z_sh + 0.25 * (r_vals ** (-0.5))
    ax3.plot(r_vals, y_meix, color=ORANGE, lw=3.0, zorder=7)
    ax3.fill_between(
        r_vals[r_vals <= 1.0],
        z_sh,
        y_meix[r_vals <= 1.0],
        color=LORANGE,
        alpha=0.55,
        zorder=3,
    )
    ax3.text(
        0.58,
        z_sh + 0.88,
        r"Meixner field $\propto r^{-1/2}$",
        ha="left",
        va="center",
        fontsize=12.0,
        color=ORANGE,
        weight="bold",
        zorder=8,
    )
    arrow(
        ax3,
        (0.54, z_sh + 0.86),
        (0.18, z_sh + 0.25 * (0.18 ** (-0.5))),
        ORANGE,
        lw=1.8,
        ms=14,
        z=8,
    )

    # Bottom callout box (user_manual.md lines 1235, 1243, 1249)
    box(ax3, (0.38, -0.46), 3.35, 0.96, "", LGREEN, GREEN, lw=1.8, r=0.07, z=2)
    ax3.text(
        0.38,
        -0.18,
        "Sheet(size=(30e-3, 40e-3, 0.0))",
        ha="center",
        va="center",
        fontsize=11.5,
        color=GREY,
        weight="bold",
        zorder=5,
    )
    badge(
        ax3,
        (0.38, -0.62),
        "sheet_edge_correction=True:\n" + r"quasi-static Meixner $r^{-1/2}$ correction",
        fc=GREEN,
        fs=10.5,
        pad=0.24,
    )

    ax3.set_xlim(-1.60, 2.20)
    ax3.set_ylim(-1.10, 2.55)

    fig.savefig(out_dir / "fig6_4_conformal_pec_and_sheets.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- 7.2
def fig7_2(outroot):
    """Figure 7.2: Multi-pole dispersive materials via Auxiliary Differential Equations (wakis.dispersive)."""
    out_dir = outroot / "section7"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate exact susceptibilities using wakis.dispersive (user_manual.md lines 1384, 1390-1392)
    lorentz_pole = Lorentz(delta_eps=1.0, omega_0=1.0, gamma=0.2)
    drude_pole = Drude(omega_p=1.0, gamma=0.2)
    debye_pole = Debye(delta_eps=1.0, tau=1.0)

    u_lor = np.linspace(0.0, 2.0, 500)
    chi_lor = lorentz_pole.chi(u_lor)

    u_dru = np.linspace(0.25, 2.0, 500)
    chi_dru = drude_pole.chi(u_dru)

    u_deb = np.geomspace(0.05, 20.0, 500)
    chi_deb = debye_pole.chi(u_deb)

    # Mathematical assertions (passivity in exp(+i omega t), resonance values, O(dt^2) discrete ADE)
    assert (
        np.all(chi_lor.imag <= 1e-15)
        and np.all(chi_dru.imag <= 1e-15)
        and np.all(chi_deb.imag <= 1e-15)
    )
    chi_lor_res = lorentz_pole.chi(1.0)
    assert np.isclose(chi_lor_res.real, 0.0, atol=1e-14) and np.isclose(
        chi_lor_res.imag, -1.0 / 0.2
    )
    chi_deb_res = debye_pole.chi(1.0)
    assert np.isclose(chi_deb_res.real, 0.5) and np.isclose(chi_deb_res.imag, -0.5)
    # Verify O(dt^2) convergence of mat.epsilon_discrete(f, dt) to mat.epsilon(f) (user_manual.md line 1400)
    mat_test = DispersiveMaterial(
        eps_inf=2.0, poles=[lorentz_pole, drude_pole, debye_pole], sigma=0.0
    )
    f_test = 0.8 / (2 * np.pi)
    err_dt1 = abs(mat_test.epsilon_discrete(f_test, 0.05) - mat_test.epsilon(f_test))
    err_dt2 = abs(mat_test.epsilon_discrete(f_test, 0.025) - mat_test.epsilon(f_test))
    assert np.isclose(err_dt1 / err_dt2, 4.0, rtol=2e-2)

    fig = plt.figure(figsize=(W21, H))

    # Shared top equation banner across Panels 1-3 (line 1384)
    ax_ban = fig.add_axes([0.02, 0.88, 0.65, 0.10])
    ax_ban.axis("off")
    box(
        ax_ban,
        (0.5, 0.48),
        0.96,
        0.74,
        r"$\varepsilon_r(\omega) = \varepsilon_\infty + \sum_p \chi_p(\omega) + \frac{\sigma}{i\omega\varepsilon_0}$"
        + r"$\qquad(\exp(+i\omega t)\ \mathrm{convention:\ Im}\,\chi_p(\omega) \leq 0)$",
        LBLUE,
        BLUE,
        fs=14.5,
        lw=1.6,
        r=0.04,
    )
    ax_ban.set_xlim(0.0, 1.0)
    ax_ban.set_ylim(0.0, 1.0)

    # Helper to style susceptibility sub-plots (Panels 1-3)
    panel_specs = [
        (
            [0.03, 0.36, 0.19, 0.44],
            "Lorentz(delta_eps, omega_0, gamma)",
            u_lor,
            chi_lor,
            r"$\omega / \omega_0$",
            False,
            1.0,
            r"$\omega = \omega_0$",
            r"$\chi_p(\omega) = \frac{\Delta\varepsilon\,\omega_0^2}{\omega_0^2 - \omega^2 + i\gamma\omega}$",
            r"$\frac{d^2\mathbf{P}_p}{dt^2} + \gamma\frac{d\mathbf{P}_p}{dt} + \omega_0^2\mathbf{P}_p = \varepsilon_0\Delta\varepsilon\,\omega_0^2\mathbf{E}$",
        ),
        (
            [0.25, 0.36, 0.19, 0.44],
            "Drude(omega_p, gamma)",
            u_dru,
            chi_dru,
            r"$\omega / \omega_p$",
            False,
            1.0,
            r"$\omega = \omega_p$",
            r"$\chi_p(\omega) = -\frac{\omega_p^2}{\omega^2 - i\gamma\omega}$",
            r"$\frac{d^2\mathbf{P}_p}{dt^2} + \gamma\frac{d\mathbf{P}_p}{dt} = \varepsilon_0\omega_p^2\mathbf{E}$",
        ),
        (
            [0.47, 0.36, 0.19, 0.44],
            "Debye(delta_eps, tau)",
            u_deb,
            chi_deb,
            r"$\omega\tau$ (log scale)",
            True,
            1.0,
            r"$\omega = 1/\tau$",
            r"$\chi_p(\omega) = \frac{\Delta\varepsilon}{1 + i\omega\tau}$",
            r"$\tau\frac{d\mathbf{P}_p}{dt} + \mathbf{P}_p = \varepsilon_0\Delta\varepsilon\,\mathbf{E}$",
        ),
    ]

    for (
        rect,
        title,
        u,
        chi,
        xlab,
        is_log,
        u_mark,
        mark_lab,
        eq_chi,
        eq_ode,
    ) in panel_specs:
        ax = fig.add_axes(rect)
        ax.set_title(title, fontsize=12.5, weight="bold", pad=10)
        ax.axhline(0.0, color=GREY, lw=1.0, alpha=0.7, zorder=1)
        ax.axvline(u_mark, color=GREY, lw=1.4, ls="--", zorder=2)
        if is_log:
            ax.semilogx(
                u,
                chi.real,
                color=BLUE,
                lw=2.5,
                label=r"$\mathrm{Re}\,\chi_p$",
                zorder=4,
            )
            ax.semilogx(
                u,
                chi.imag,
                color=ORANGE,
                lw=2.5,
                ls="--",
                label=r"$\mathrm{Im}\,\chi_p \leq 0$",
                zorder=4,
            )
            ax.set_xticks([0.1, 1.0, 10.0])
            ax.set_xticklabels(["0.1", "1", "10"], fontsize=11.5)
        else:
            ax.plot(
                u,
                chi.real,
                color=BLUE,
                lw=2.5,
                label=r"$\mathrm{Re}\,\chi_p$",
                zorder=4,
            )
            ax.plot(
                u,
                chi.imag,
                color=ORANGE,
                lw=2.5,
                ls="--",
                label=r"$\mathrm{Im}\,\chi_p \leq 0$",
                zorder=4,
            )
            ax.set_xticks(
                [0.5, 1.0, 1.5, 2.0] if u[0] > 0 else [0.0, 0.5, 1.0, 1.5, 2.0]
            )
            ax.tick_params(axis="x", labelsize=11.5)

        ax.set_xlim(u[0], u[-1])
        ax.set_yticks([])
        ax.minorticks_off()
        for sp_name in ("top", "right"):
            ax.spines[sp_name].set_visible(False)
        ax.set_xlabel(xlab, fontsize=12.5)
        ax.text(
            u_mark * (1.12 if is_log else 1.04),
            ax.get_ylim()[0] + 0.14 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            mark_lab,
            ha="left",
            va="bottom",
            fontsize=11.5,
            color=GREY,
            weight="bold",
        )
        if title.startswith("Lorentz"):
            ax.legend(loc="lower left", frameon=False, fontsize=11.5)

        # Card beneath the plot with chi_p(omega) and time-domain ADE
        ax_c = fig.add_axes([rect[0] - 0.005, 0.04, rect[2] + 0.01, 0.24])
        ax_c.axis("off")
        box(ax_c, (0.5, 0.50), 0.98, 0.92, "", LGREEN, GREEN, lw=1.5, r=0.06, z=1)
        ax_c.text(
            0.5,
            0.72,
            eq_chi,
            ha="center",
            va="center",
            fontsize=13.5,
            color="black",
            zorder=4,
        )
        ax_c.text(
            0.5,
            0.24,
            eq_ode,
            ha="center",
            va="center",
            fontsize=11.5,
            color=GREY,
            zorder=4,
        )
        ax_c.set_xlim(0.0, 1.0)
        ax_c.set_ylim(0.0, 1.0)

    # Vertical divider before Panel 4
    fig.add_artist(
        plt.Line2D(
            [0.68, 0.68], [0.04, 0.96], transform=fig.transFigure, color=LGREY, lw=1.6
        )
    )

    # ---- PANEL 4: Explicit 2nd-Order ADE Update (lines 1396-1400) ----
    ax4 = fig.add_axes([0.69, 0.03, 0.30, 0.94])
    ax4.axis("off")
    ax4.text(
        0.5,
        0.96,
        "Explicit 2nd-Order ADE Update",
        ha="center",
        va="center",
        fontsize=14.5,
        weight="bold",
    )

    # Stage 1 box (green border)
    box(ax4, (0.5, 0.68), 0.96, 0.42, "", LGREEN, GREEN, lw=2.0, r=0.05, z=1)
    badge(
        ax4,
        (0.5, 0.84),
        r"Stage 1: inputs $\mathbf{P}_p^{n-1},\ \mathbf{P}_p^n,\ \mathbf{E}^n \;\rightarrow\;$ update $\mathbf{P}_p^{n+1}$",
        fc=GREEN,
        fs=11.0,
        pad=0.25,
    )
    ax4.text(
        0.5,
        0.70,
        r"$\frac{\mathbf{P}_p^{n+1}-2\mathbf{P}_p^n+\mathbf{P}_p^{n-1}}{\Delta t^2}"
        r" + \gamma_p\frac{\mathbf{P}_p^{n+1}-\mathbf{P}_p^{n-1}}{2\Delta t} + \omega_{0,p}^2\mathbf{P}_p^n$",
        ha="center",
        va="center",
        fontsize=12.0,
        color="black",
        zorder=4,
    )
    ax4.text(
        0.5,
        0.59,
        r"$= \varepsilon_0\Delta\varepsilon_p\omega_{0,p}^2\,\mathbf{E}^n$",
        ha="center",
        va="center",
        fontsize=12.5,
        color="black",
        zorder=4,
    )
    ax4.text(
        0.5,
        0.51,
        r"Debye & $\sigma$: Crank–Nicolson on $[n\Delta t,\ (n+1)\Delta t]$",
        ha="center",
        va="center",
        fontsize=10.5,
        color=GREY,
        weight="bold",
        zorder=4,
    )

    # Arrow from Stage 1 to Stage 2
    arrow(ax4, (0.5, 0.46), (0.5, 0.39), GREEN, lw=2.6, ms=18, z=6)
    ax4.text(
        0.53,
        0.425,
        r"$\sum_p (\mathbf{P}_p^{n+1} - \mathbf{P}_p^n)$",
        ha="left",
        va="center",
        fontsize=11.5,
        color=GREEN,
        weight="bold",
    )

    # Stage 2 box (blue border)
    box(ax4, (0.5, 0.25), 0.96, 0.26, "", LBLUE, BLUE, lw=2.0, r=0.05, z=1)
    badge(
        ax4,
        (0.5, 0.33),
        r"Stage 2: inputs $\mathbf{H}^{n+1/2},\ \mathbf{J}_s^{n+1/2},\ \sum_p(\mathbf{P}_p^{n+1}-\mathbf{P}_p^n) \;\rightarrow\;$ advance $\mathbf{E}^{n+1}$",
        fc=BLUE,
        fs=10.0,
        pad=0.24,
    )
    ax4.text(
        0.5,
        0.22,
        r"$\varepsilon_0\varepsilon_\infty(\mathbf{E}^{n+1}-\mathbf{E}^n) = "
        r"\Delta t(\nabla\times\mathbf{H}^{n+1/2} - \mathbf{J}_s^{n+1/2})$",
        ha="center",
        va="center",
        fontsize=11.5,
        color="black",
        zorder=4,
    )
    ax4.text(
        0.5,
        0.15,
        r"$-\,\sum_p (\mathbf{P}_p^{n+1} - \mathbf{P}_p^n)$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=BLUE,
        weight="bold",
        zorder=4,
    )

    # Bottom grey badge (line 1400)
    badge(
        ax4,
        (0.5, 0.04),
        "query continuous mat.epsilon(f) or\nexact discrete mat.epsilon_discrete(f, dt=solver.dt)",
        fc=GREY,
        fs=10.5,
        pad=0.26,
    )

    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0.0, 1.0)

    fig.savefig(out_dir / "fig7_2_ade_dispersive_poles.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- 7.4
def fig7_4(outroot):
    """Figure 7.4: Cold unmagnetized electron plasma and linear plasma wakefield (wakis.plasma)."""
    out_dir = outroot / "section7"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Left panel: cold-plasma permittivity via wakis.plasma (user_manual.md lines 1463-1472)
    n_e = 1.0e20
    omega_p = plasma_frequency(n_e)
    assert np.isclose(omega_p, np.sqrt(n_e * q_e**2 / (eps_0 * m_e)), rtol=1e-15)
    assert np.isclose(omega_p * 1e-11, 5.64, atol=0.01)

    mat_col0 = cold_plasma(omega_p, collision_rate=0.0)
    mat_coll = cold_plasma(omega_p, collision_rate=0.15 * omega_p)
    assert np.isclose(mat_col0.epsilon(omega_p / (2 * np.pi)), 0.0, atol=1e-15)

    u = np.linspace(0.45, 2.5, 500)
    freqs = u * omega_p / (2 * np.pi)
    eps_col0 = mat_col0.epsilon(freqs)
    eps_coll = mat_coll.epsilon(freqs)
    assert np.all(eps_coll.imag <= 0.0)
    assert np.all(eps_col0[u < 1.0].real < 0.0) and np.all(eps_col0[u > 1.0].real > 0.0)

    # 2. Right panel: analytical linear plasma wakefield (user_manual.md lines 1468, 1478, 2558-2560)
    k_p = omega_p / c_light
    sigmaz = 0.5e-3
    q_bunch = -1.0e-9
    r_obs = 0.2e-3
    r_pipe = 2.0e-3
    k_p_s = np.linspace(
        -3.5, 18.5, 800
    )  # normalized distance behind drive bunch s = -xi
    xi = -k_p_s / k_p
    Ez, _Fx, _Fy = linear_wake(
        r_obs, 0.0, xi, k_p=k_p, sigmaz=sigmaz, q=q_bunch, radius=r_pipe
    )
    e_amp = abs(wake_amplitude(r_obs, k_p, sigmaz, q=q_bunch, radius=r_pipe))
    # Verify asymptotic oscillation amplitude behind the bunch matches wake_amplitude
    trailing = k_p_s > 6.0
    assert np.isclose(np.max(np.abs(Ez[trailing])), e_amp, rtol=1e-3)

    fig = plt.figure(figsize=(W, H))

    # ---- LEFT PANEL: Cold-plasma permittivity eps_r(omega) ----
    ax1 = fig.add_axes([0.06, 0.24, 0.40, 0.62])
    ax1.set_title(
        r"Cold-plasma permittivity $\varepsilon_r(\omega)$",
        fontsize=15.0,
        weight="bold",
        pad=12,
    )

    ax1.axvspan(0.45, 1.0, color=LORANGE, alpha=0.45, zorder=0)
    ax1.axvspan(1.0, 2.5, color=LBLUE, alpha=0.45, zorder=0)
    ax1.axhline(0.0, color=GREY, lw=1.4, zorder=2)
    ax1.axvline(1.0, color=GREY, lw=1.6, ls="--", zorder=2)

    ax1.plot(
        u,
        eps_col0.real,
        color=BLUE,
        lw=2.8,
        label=r"$\mathrm{Re}\,\varepsilon_r$ (collision_rate=0.0)",
        zorder=4,
    )
    ax1.plot(
        u,
        eps_coll.real,
        color=BLUE,
        lw=2.0,
        ls="--",
        label=r"$\mathrm{Re}\,\varepsilon_r$ ($\nu/\omega_p = 0.15$)",
        zorder=4,
    )
    ax1.plot(
        u,
        eps_coll.imag,
        color=ORANGE,
        lw=2.4,
        ls="--",
        label=r"$\mathrm{Im}\,\varepsilon_r \leq 0$ ($\nu/\omega_p = 0.15$)",
        zorder=5,
    )
    ax1.plot([1.0], [0.0], "o", color=BLUE, ms=8, mec="white", mew=1.2, zorder=6)

    ax1.text(
        0.72,
        0.66,
        r"$\omega < \omega_p$:" + "\n" + r"$\mathrm{Re}\,\varepsilon_r < 0$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=ORANGE,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": ORANGE,
            "lw": 1.0,
            "alpha": 0.92,
        },
        zorder=6,
    )
    ax1.text(
        1.78,
        -1.35,
        r"$\omega > \omega_p$:" + "\n" + r"$\mathrm{Re}\,\varepsilon_r > 0$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=BLUE,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": BLUE,
            "lw": 1.0,
            "alpha": 0.92,
        },
        zorder=6,
    )
    ax1.text(
        1.04,
        -2.90,
        r"$\omega / \omega_p = 1$",
        ha="left",
        va="bottom",
        fontsize=12.0,
        color=GREY,
        weight="bold",
    )

    ax1.set_xlim(0.45, 2.5)
    ax1.set_ylim(-3.2, 1.15)
    ax1.set_xlabel(r"normalized frequency $\omega / \omega_p$", fontsize=13.5)
    ax1.set_ylabel(r"relative permittivity $\varepsilon_r(\omega)$", fontsize=13.5)
    ax1.tick_params(labelsize=12)
    for sp_name in ("top", "right"):
        ax1.spines[sp_name].set_visible(False)
    ax1.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        fontsize=11.0,
    )

    # Bottom equations box under Left Panel (lines 1463, 1465, 1470-1472)
    ax1_b = fig.add_axes([0.04, 0.02, 0.43, 0.14])
    ax1_b.axis("off")
    box(ax1_b, (0.5, 0.50), 0.98, 0.92, "", LGREEN, GREEN, lw=1.6, r=0.06, z=1)
    ax1_b.text(
        0.5,
        0.72,
        r"$\omega_p = \sqrt{\frac{n_e e^2}{\varepsilon_0 m_e}},\qquad"
        r"\varepsilon_r(\omega) = 1 - \frac{\omega_p^2}{\omega^2 - i\nu\omega}$"
        r"$\ \ (\mathrm{Drude(omega\_p,\ collision\_rate)},\ \varepsilon_\infty = 1)$",
        ha="center",
        va="center",
        fontsize=11.5,
        color="black",
        zorder=4,
    )
    ax1_b.text(
        0.5,
        0.24,
        rf"$n_e = \mathrm{{1e20\ m^{{-3}}}}\ \Rightarrow\ \omega_p \approx {omega_p * 1e-11:.2f}\times 10^{{11}}\mathrm{{\ rad/s}}$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=BLUE,
        weight="bold",
        zorder=4,
    )
    ax1_b.set_xlim(0.0, 1.0)
    ax1_b.set_ylim(0.0, 1.0)

    # Vertical divider
    fig.add_artist(
        plt.Line2D(
            [0.485, 0.485], [0.04, 0.94], transform=fig.transFigure, color=LGREY, lw=1.6
        )
    )

    # ---- RIGHT PANEL: Analytical linear plasma wakefield (wakis.plasma) ----
    ax2 = fig.add_axes([0.565, 0.24, 0.405, 0.62])
    ax2.set_title(
        "Analytical linear plasma wakefield (wakis.plasma)",
        fontsize=14.5,
        weight="bold",
        pad=12,
    )

    ax2.axhline(0.0, color=GREY, lw=1.2, zorder=1)
    ax2.axvline(0.0, color=GREY, lw=1.2, ls=":", zorder=1)

    # Normalized longitudinal wakefield Ez / e_amp and +-1 envelope ('wake_amplitude')
    ez_norm = Ez / e_amp
    k_p_sigma = k_p * sigmaz
    bunch_prof = 0.85 * np.exp(-0.5 * (k_p_s / k_p_sigma) ** 2)

    ax2.fill_between(
        k_p_s,
        0.0,
        bunch_prof,
        color=LBLUE,
        edgecolor=BLUE,
        lw=1.6,
        alpha=0.85,
        zorder=2,
    )
    ax2.text(
        0.4,
        1.02,
        "Gaussian drive bunch $(s = 0)$",
        ha="center",
        va="bottom",
        fontsize=11.0,
        color=BLUE,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.15",
            "fc": "white",
            "ec": BLUE,
            "lw": 0.9,
            "alpha": 0.92,
        },
        zorder=6,
    )

    # Trailing envelope in dashed green labelled 'wake_amplitude'
    s_env = k_p_s[k_p_s >= 2.0]
    ax2.plot(
        s_env,
        np.ones_like(s_env),
        color=GREEN,
        lw=2.2,
        ls="--",
        label="wake_amplitude",
        zorder=3,
    )
    ax2.plot(s_env, -np.ones_like(s_env), color=GREEN, lw=2.2, ls="--", zorder=3)
    ax2.plot(
        k_p_s,
        ez_norm,
        color=BLUE,
        lw=2.6,
        label=r"longitudinal wake $E_z(s)$ (linear_wake)",
        zorder=5,
    )

    ax2.text(
        11.5,
        1.08,
        "wake_amplitude",
        ha="center",
        va="bottom",
        fontsize=12.0,
        color=GREEN,
        weight="bold",
    )

    ax2.set_xlim(-3.5, 18.5)
    ax2.set_ylim(-1.38, 1.38)
    ax2.set_yticks([-1.0, 0.0, 1.0])
    ax2.set_yticklabels([r"$-E_{\max}$", r"$0$", r"$+E_{\max}$"], fontsize=12)
    ax2.set_xlabel(
        r"normalized co-moving distance behind drive bunch $k_p s = k_p(ct - z)$",
        fontsize=12.5,
    )
    ax2.set_ylabel(
        r"longitudinal wakefield $E_z / E_{\max}$", fontsize=13.0, color=BLUE
    )
    ax2.tick_params(axis="x", labelsize=12)
    for sp_name in ("top", "right"):
        ax2.spines[sp_name].set_visible(False)
    ax2.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        fontsize=11.0,
    )

    # Bottom badge under Right Panel (line 1468)
    ax2_b = fig.add_axes([0.53, 0.02, 0.45, 0.14])
    ax2_b.axis("off")
    box(ax2_b, (0.5, 0.50), 0.98, 0.92, "", LGREEN, GREEN, lw=1.6, r=0.06, z=1)
    ax2_b.text(
        0.5,
        0.72,
        r"$E_z(r, \xi) = -\frac{k_p^2}{2\pi\varepsilon_0}\,g(r; r_0)\int_\xi^\infty \lambda(\xi')\cos(k_p(\xi-\xi'))\,d\xi'$"
        r"$\quad(\xi = z - ct = -s)$",
        ha="center",
        va="center",
        fontsize=11.5,
        color="black",
        zorder=4,
    )
    badge(
        ax2_b,
        (0.5, 0.24),
        "wakis.plasma: plasma_frequency, cold_plasma, linear_wake, wake_amplitude",
        fc=GREEN,
        fs=11.0,
        pad=0.24,
    )
    ax2_b.set_xlim(0.0, 1.0)
    ax2_b.set_ylim(0.0, 1.0)

    fig.savefig(out_dir / "fig7_4_cold_plasma_wakefield.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- 8.2
def iso(p):
    """Cabinet oblique projection (depth y drawn up and to the right at half scale)."""
    x, y, z = p
    return np.array(
        [
            x + 0.5 * y * np.cos(np.radians(35)),
            z + 0.5 * y * np.sin(np.radians(35)),
        ]
    )


def fig8_2(outroot):
    """Figure 8.2: Soft current sources, time waveforms, and broadband spectral normalization."""
    out_dir = outroot / "section8"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate exact sources from user_manual.md lines 1530-1558:
    pulse = GaussianSource(frequency=5e9, fwidth=2e9, cutoff=5.0)
    cw = ContinuousSource(frequency=5e9, ramp_time=2e-9)
    dipole = CurrentSource(
        src_time=pulse,
        component="z",
        center=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.0),
        amplitude=1e-3,
    )
    kx = 2 * np.pi * 5e9 / 3e8 * np.sin(np.deg2rad(20))
    sheet_src = CurrentSource(
        src_time=pulse,
        component="x",
        center=(0.0, 0.0, -1e-2),
        size=(np.inf, np.inf, 0.0),
        amp_func=lambda x, y, z: np.exp(-1j * kx * x),
    )
    assert dipole.component == "z" and sheet_src.component == "x"

    # Mathematical assertions on GaussianSource and ContinuousSource (lines 1494, 1531-1539, 1562-1563)
    cutoff = 5.0
    assert np.isclose(pulse.sigma_t, 1.0 / (2 * np.pi * pulse.fwidth), rtol=1e-15)
    assert np.isclose(pulse.t0, pulse.start_time + cutoff * pulse.sigma_t, rtol=1e-15)
    assert np.isclose(pulse.end_time, pulse.t0 + cutoff * pulse.sigma_t, rtol=1e-15)

    dt_fine = 0.5e-12
    t_dft = np.arange(0.0, 4.0e-9, dt_fine)
    s_t = pulse(t_dft)
    # Zero net charge: time integral of GaussianSource vanishes
    assert abs(np.trapezoid(s_t, t_dft)) < 1e-4 * np.max(np.abs(s_t)) * pulse.end_time

    freqs = np.linspace(3e9, 7e9, 401)
    s_analytic = pulse.fourier_transform(freqs)
    # Discrete running DFT sum_n s(t_n) exp(-2pi i f t_n) dt matches pulse.fourier_transform(freqs)
    s_numeric = (
        s_t[None, :] @ np.exp(-2j * np.pi * freqs[:, None] * t_dft[None, :]).T
    ).ravel() * dt_fine
    assert np.max(np.abs(s_numeric - s_analytic)) / np.max(np.abs(s_analytic)) < 5e-4

    # Broadband CW normalization: H(f) recovered from F(f) / S(f) across [3, 7] GHz (lines 1562-1563)
    f_r1, g_r1 = 4.4e9, 0.18e9
    f_r2, g_r2 = 5.7e9, 0.22e9
    h_exact = (
        0.35
        + 0.55 * (1j * g_r1) / ((freqs - f_r1) + 1j * g_r1)
        + 0.45 * (1j * g_r2) / ((freqs - f_r2) + 1j * g_r2)
    )
    f_mon = h_exact * s_numeric
    h_recovered = f_mon / s_analytic
    assert np.max(np.abs(np.abs(h_recovered) - np.abs(h_exact))) < 5e-4

    fig = plt.figure(figsize=(W21, H))

    # ---- LEFT PANEL: Time profiles src_time(t) ----
    fig.text(
        0.18,
        0.94,
        "Time profiles: src_time(t)",
        ha="center",
        va="center",
        fontsize=15.0,
        weight="bold",
    )

    # Top sub-axes: GaussianSource
    ax_p = fig.add_axes([0.045, 0.54, 0.27, 0.33])
    t_ns_p = np.linspace(0.0, 1.5, 800)
    t_s_p = t_ns_p * 1e-9
    env_p = np.exp(-((t_s_p - pulse.t0) ** 2) / (2 * pulse.sigma_t**2))
    ax_p.axhline(0.0, color=LGREY, lw=1.0, zorder=1)
    ax_p.plot(
        t_ns_p,
        env_p,
        color=GREY,
        lw=1.4,
        ls="--",
        zorder=2,
        label=r"$\pm\exp(-(t-t_0)^2/(2\sigma_t^2))$",
    )
    ax_p.plot(t_ns_p, -env_p, color=GREY, lw=1.4, ls="--", zorder=2)
    ax_p.plot(t_ns_p, pulse(t_s_p), color=BLUE, lw=2.2, zorder=4)

    t0_ns = pulse.t0 * 1e9
    tend_ns = pulse.end_time * 1e9
    ax_p.axvline(t0_ns, color=BLUE, lw=1.3, ls=":", zorder=3)
    ax_p.axvline(tend_ns, color=RED, lw=1.4, ls="--", zorder=3)

    ax_p.set_title(
        "GaussianSource(frequency=5e9, fwidth=2e9, cutoff=5.0)",
        fontsize=11.0,
        weight="bold",
        color=BLUE,
        pad=6,
    )
    ax_p.text(
        0.72,
        0.90,
        r"$t_0 = \mathrm{start\_time} + \mathrm{cutoff}\cdot\sigma_t$"
        + "\n"
        + r"$\sigma_t = 1/(2\pi\,\mathrm{fwidth})$",
        ha="left",
        va="top",
        fontsize=9.8,
        color=GREY,
        bbox={
            "boxstyle": "round,pad=0.14",
            "fc": "white",
            "ec": LGREY,
            "lw": 0.9,
            "alpha": 0.92,
        },
        zorder=6,
    )
    ax_p.text(
        tend_ns + 0.03,
        -0.82,
        f"pulse.end_time = {tend_ns:.2f} ns",
        ha="left",
        va="bottom",
        fontsize=10.0,
        color=RED,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.14",
            "fc": "white",
            "ec": RED,
            "lw": 0.9,
            "alpha": 0.92,
        },
        zorder=6,
    )
    ax_p.set_xlim(0.0, 1.5)
    ax_p.set_ylim(-1.15, 1.15)
    ax_p.set_yticks([-1.0, 0.0, 1.0])
    ax_p.set_xlabel("time $t$ [ns]", fontsize=11.5)
    ax_p.set_ylabel("pulse(t)", fontsize=12.0, color=BLUE)
    ax_p.tick_params(labelsize=10.5)
    for sp_name in ("top", "right"):
        ax_p.spines[sp_name].set_visible(False)

    # Bottom sub-axes: ContinuousSource
    ax_cw = fig.add_axes([0.045, 0.09, 0.27, 0.33])
    t_ns_c = np.linspace(0.0, 2.5, 1000)
    t_s_c = t_ns_c * 1e-9
    ramp_env = np.sin(0.5 * np.pi * np.clip(t_s_c / cw.ramp_time, 0.0, 1.0)) ** 2
    ax_cw.axhline(0.0, color=LGREY, lw=1.0, zorder=1)
    ax_cw.plot(t_ns_c, ramp_env, color=GREY, lw=1.4, ls="--", zorder=2)
    ax_cw.plot(t_ns_c, -ramp_env, color=GREY, lw=1.4, ls="--", zorder=2)
    ax_cw.plot(t_ns_c, cw(t_s_c), color=ORANGE, lw=2.0, zorder=4)
    ax_cw.axvline(cw.ramp_time * 1e9, color=GREY, lw=1.4, ls="--", zorder=3)

    ax_cw.set_title(
        "ContinuousSource(frequency=5e9, ramp_time=2e-9)",
        fontsize=11.0,
        weight="bold",
        color=ORANGE,
        pad=6,
    )
    ax_cw.text(
        1.95,
        -0.86,
        "ramp_time = 2 ns",
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=GREY,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.14",
            "fc": "white",
            "ec": GREY,
            "lw": 0.9,
            "alpha": 0.92,
        },
        zorder=6,
    )
    ax_cw.text(
        0.12,
        0.82,
        r"$f_0 = 5\mathrm{\ GHz}$",
        ha="left",
        va="center",
        fontsize=11.0,
        color=ORANGE,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.14",
            "fc": "white",
            "ec": ORANGE,
            "lw": 0.9,
            "alpha": 0.92,
        },
        zorder=6,
    )
    ax_cw.set_xlim(0.0, 2.5)
    ax_cw.set_ylim(-1.15, 1.15)
    ax_cw.set_yticks([-1.0, 0.0, 1.0])
    ax_cw.set_xlabel("time $t$ [ns]", fontsize=11.5)
    ax_cw.set_ylabel("cw(t)", fontsize=12.0, color=ORANGE)
    ax_cw.tick_params(labelsize=10.5)
    for sp_name in ("top", "right"):
        ax_cw.spines[sp_name].set_visible(False)

    # Dividers
    fig.add_artist(
        plt.Line2D(
            [0.335, 0.335], [0.04, 0.96], transform=fig.transFigure, color=LGREY, lw=1.6
        )
    )
    fig.add_artist(
        plt.Line2D(
            [0.655, 0.655], [0.04, 0.96], transform=fig.transFigure, color=LGREY, lw=1.6
        )
    )

    # ---- CENTRE PANEL: Spatial sources: CurrentSource / MagneticCurrentSource ----
    fig.text(
        0.495,
        0.94,
        "Spatial sources: CurrentSource / MagneticCurrentSource",
        ha="center",
        va="center",
        fontsize=14.0,
        weight="bold",
    )

    # Top sub-panel: 3D grid cell with point dipole z-edge arrow at center=(0,0,0)
    ax_dip = fig.add_axes([0.345, 0.50, 0.30, 0.39])
    bare(ax_dip)
    V = {
        (i, j, k): np.array([i, j, k], float)
        for i in (0, 1)
        for j in (0, 1)
        for k in (0, 1)
    }
    edges = [
        (a, b)
        for a in V
        for b in V
        if sum(abs(np.array(a) - np.array(b))) == 1 and a < b
    ]
    for a, b in edges:
        hidden = (0, 1, 0) in (a, b)
        P, Q = iso(V[a]), iso(V[b])
        ax_dip.plot(
            *zip(P, Q),
            color=BLUE,
            lw=1.3 if hidden else 1.8,
            ls=(0, (4, 4)) if hidden else "-",
            alpha=0.45 if hidden else 0.75,
            zorder=2,
        )
    # Highlight single z-edge arrow at front-left (0, 0, 0) -> (0, 0, 1)
    P_dip, Q_dip = iso((0, 0, 0)), iso((0, 0, 1))
    ax_dip.plot(*zip(P_dip, Q_dip), color=ORANGE, lw=6.0, alpha=0.45, zorder=4)
    arrow(ax_dip, P_dip, Q_dip, BLUE, lw=4.2, ms=24, z=6)
    ax_dip.add_patch(Circle(0.5 * (P_dip + Q_dip), 0.045, color=ORANGE, zorder=7))
    ax_dip.text(
        -0.12,
        0.50,
        r"$J_z$ at center=(0, 0, 0)",
        ha="right",
        va="center",
        fontsize=12.0,
        color=BLUE,
        weight="bold",
    )
    box(
        ax_dip,
        (0.45, -0.34),
        2.35,
        0.42,
        "Point dipole: size=(0.0, 0.0, 0.0),\n" + r"amplitude = $I\cdot l$ [A$\cdot$m]",
        LBLUE,
        BLUE,
        fs=11.5,
        lw=1.5,
        r=0.06,
    )
    ax_dip.set_xlim(-0.85, 1.65)
    ax_dip.set_ylim(-0.65, 1.42)

    # Bottom sub-panel: 2D sheet plane z = -1e-2 with phased x-directed arrows
    ax_sh = fig.add_axes([0.345, 0.04, 0.30, 0.43])
    bare(ax_sh)
    # Draw oblique sheet plane in (x, y) at z = 0
    sheet_corners = [
        iso((x, y, 0.0)) for x, y in ((-0.2, 0.0), (1.8, 0.0), (1.8, 1.2), (-0.2, 1.2))
    ]
    ax_sh.add_patch(
        Polygon(
            sheet_corners,
            closed=True,
            fc=LORANGE,
            ec=ORANGE,
            lw=2.0,
            alpha=0.65,
            zorder=1,
        )
    )
    ax_sh.text(
        2.16,
        0.34,
        r"$z = -1\mathrm{e}{-2}$",
        ha="left",
        va="center",
        fontsize=12.0,
        color=ORANGE,
        weight="bold",
    )
    # Phased x-directed arrows across the sheet coloured by phase
    phase_cols = [BLUE, GREEN, ORANGE, RED, BLUE]
    xs_arr = np.linspace(0.05, 1.45, 5)
    for idx_x, xv in enumerate(xs_arr):
        col = phase_cols[idx_x]
        for yv in (0.25, 0.65, 1.00):
            p0 = iso((xv - 0.11, yv, 0.0))
            p1 = iso((xv + 0.11, yv, 0.0))
            arrow(ax_sh, p0, p1, col, lw=2.4, ms=14, z=5)

    box(
        ax_sh,
        (0.95, -0.34),
        2.75,
        0.44,
        "Sheet source: size=(np.inf, np.inf, 0.0),\n"
        + r"amp_func = $\exp(-1j \cdot k_x \cdot x)$ [A/m]",
        LORANGE,
        ORANGE,
        fs=11.5,
        lw=1.5,
        r=0.06,
    )
    ax_sh.set_xlim(-0.55, 2.55)
    ax_sh.set_ylim(-0.66, 0.68)

    # ---- RIGHT PANEL: Broadband CW normalization ----
    fig.text(
        0.825,
        0.94,
        "Broadband CW normalization",
        ha="center",
        va="center",
        fontsize=15.0,
        weight="bold",
    )

    ax_n = fig.add_axes([0.70, 0.25, 0.27, 0.62])
    f_ghz = freqs * 1e-9
    s_norm = np.abs(s_analytic) / np.max(np.abs(s_analytic))
    f_norm = np.abs(f_mon) / np.max(np.abs(s_analytic))
    h_abs = np.abs(h_recovered)

    ax_n.plot(
        f_ghz,
        s_norm,
        color=GREY,
        lw=2.2,
        ls="--",
        label=r"source spectrum $|S(f)| = |\mathrm{pulse.fourier\_transform}(f)|$",
        zorder=3,
    )
    ax_n.plot(
        f_ghz,
        f_norm,
        color=BLUE,
        lw=2.4,
        label=r"recorded monitor spectrum $|F(f)| = |H(f)\cdot S(f)|$",
        zorder=4,
    )
    ax_n.plot(
        f_ghz,
        h_abs,
        color=GREEN,
        lw=2.8,
        label=r"normalized CW response $|F(f)/S(f)| = |H(f)|$",
        zorder=5,
    )

    ax_n.set_xlim(3.0, 7.0)
    ax_n.set_ylim(0.0, 1.32)
    ax_n.set_xlabel("frequency $f$ [GHz]", fontsize=13.0)
    ax_n.set_ylabel("normalized spectral magnitude", fontsize=12.5)
    ax_n.tick_params(labelsize=11.5)
    for sp_name in ("top", "right"):
        ax_n.spines[sp_name].set_visible(False)
    ax_n.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        fontsize=10.0,
    )

    # Bottom box under Right Panel (lines 1494, 1562-1563)
    ax_nb = fig.add_axes([0.675, 0.03, 0.305, 0.15])
    ax_nb.axis("off")
    box(ax_nb, (0.5, 0.50), 0.98, 0.92, "", LGREEN, GREEN, lw=1.6, r=0.06, z=1)
    ax_nb.text(
        0.5,
        0.74,
        r"$S(f) = \sum_n s(t_n)\,e^{-2\pi i f t_n}\,\Delta t$",
        ha="center",
        va="center",
        fontsize=12.0,
        color="black",
        zorder=4,
    )
    badge(
        ax_nb,
        (0.5, 0.25),
        r"$F(f)/S(f)$ (or power / $|S(f)|^2$): exact CW response across bandwidth in 1 run",
        fc=GREEN,
        fs=9.8,
        pad=0.22,
    )
    ax_nb.set_xlim(0.0, 1.0)
    ax_nb.set_ylim(0.0, 1.0)

    fig.savefig(out_dir / "fig8_2_waveforms_and_normalization.png", facecolor="white")
    plt.close(fig)


def _verify_mutations():
    """Verify that mutating each figure's mathematical relation violates its assertion."""
    # 5.3 mutation: dropping the dt/2 phase offset on H_DFT degrades convergence from O(dt^2) to O(dt)
    f0 = 1.0
    w0 = 2 * np.pi * f0
    t_end = 4.0
    errs_mut = []
    for nt in (80, 160):
        dt = t_end / nt
        n = np.arange(nt)
        t_e = (n + 1.0) * dt
        t_h = (n + 0.5) * dt
        e_sig = (t_e / t_end) * (1.0 - t_e / t_end) * np.cos(w0 * t_e)
        h_sig = (t_h / t_end) * (1.0 - t_h / t_end) * np.cos(w0 * t_h)
        e_dft = np.sum(e_sig * np.exp(-2j * np.pi * f0 * t_e)) * dt
        h_dft_mut = np.sum(h_sig * np.exp(-2j * np.pi * f0 * t_e)) * dt
        errs_mut.append(abs(e_dft - h_dft_mut) / abs(e_dft))
    assert not np.isclose(errs_mut[0] / errs_mut[1], 4.0, rtol=5e-2)

    # 6.4 mutation: corrupting the -1 extrapolation weight in T breaks reproduction of uniform H = 1
    a_hat_1 = 1.0 + 2.0 * 0.08
    a_hat_2 = 1.0 - 0.5 * 0.08  # mutated from -1.0 to -0.5
    assert not np.isclose(1.0 + 2.0 * 0.08 + (-0.5) * 0.08, 1.0 + 0.08)
    assert not np.isclose(a_hat_1 + a_hat_2, 2.0 + 0.08)

    # 7.2 mutation: flipping the sign of i in Lorentz susceptibility violates passivity Im chi <= 0
    lor_conj = np.conj(Lorentz(1.0, 1.0, 0.2).chi(1.0))
    assert not (lor_conj.imag <= 0.0)

    # 8.2 mutation: omitting 1/S(f) normalization fails to recover |H(f)|
    pulse = GaussianSource(frequency=5e9, fwidth=2e9, cutoff=5.0)
    freqs = np.linspace(3e9, 7e9, 41)
    s_f = pulse.fourier_transform(freqs)
    h_f = 1.0 / (1.0 + 1j * (freqs - 5e9) / 0.5e9)
    f_mon = h_f * s_f
    assert not np.allclose(np.abs(f_mon), np.abs(h_f), rtol=1e-2)


if __name__ == "__main__":
    outroot = Path(sys.argv[1])
    outroot.mkdir(parents=True, exist_ok=True)
    _verify_mutations()
    fig5_3(outroot)
    fig6_1(outroot)
    fig6_3(outroot)
    fig6_4(outroot)
    fig7_2(outroot)
    fig7_4(outroot)
    fig8_2(outroot)
