"""Deterministic scripted illustrations for User Manual Sections 9, 10, 11, and 12.

Draws every figure with ``"route": "scripted"`` in
``user_manual_figures/section{N}/specs/section{N}.json`` for N in 9, 10, 11, 12:
- section9/fig9_1_bloch_periodic.png
- section10/fig10_5_harminv_resonance.png
- section11/fig11_1_triangle_and_polygon_fit.png
- section11/fig11_2_subgrid_and_subtri.png
- section11/fig11_3_curvilinear_ogrid_nfit.png
- section11/fig11_6_high_order_dgtd_fdtd24.png
- section12/fig12_2_gpu_throughput_and_timings.png

Usage: python sections_9_10_11_12_scripted.py OUTROOT
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
from scipy.constants import c as C0  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from wakis.bloch import incidence_angle, k_parallel  # noqa: E402
from wakis.curvilinear import (
    hodge_components,
    ogrid_cartesian_mesh,
    ogrid_mesh,
    radial_nodes,
)  # noqa: E402
from wakis.highorder import DGTD2D  # noqa: E402
from wakis.highorder.fdtd24 import FDTD24  # noqa: E402
from wakis.highorder.mesh2d import TriMesh  # noqa: E402
from wakis.highorder.nodal2d import reference_triangle  # noqa: E402
from wakis.resonances import harminv  # noqa: E402
from wakis.subgrid import (
    Circle as SubgridCircle,
    Geometry,
    QuadMesh,
    SubgridFIT2D,
    refine_near,
)  # noqa: E402
from wakis.unstructured import (  # noqa: E402
    PolygonFIT2D,
    SimplexMesh,
    UnstructuredFIT2D,
    hybrid_circles_mesh,
    voronoi_disk_mesh,
)
from wakis.unstructured.subtri import SubTriFIT2D, subtri_circles_mesh  # noqa: E402

BLUE, LBLUE = "#2f6fb0", "#d6e6f5"
ORANGE, LORANGE = "#e8873a", "#fbe0c8"
GREY, LGREY = "#4a4f57", "#e6e8eb"
GREEN, LGREEN = "#5f9e6e", "#d3ebd8"
RED = "#c0392b"
LRED = "#fadbd8"

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


def arrow(ax, p, q, color=GREY, lw=2.0, style="-|>", rad=0.0, ms=18, z=3, ls="-"):
    """Draw a FancyArrowPatch from p to q on ax."""
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


def box(
    ax, xy, w, h, text, fc, ec, fs=14, tc="black", lw=1.5, pad=0.02, rsize=0.06, z=3
):
    """Draw a rounded callout badge centered at xy."""
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad={pad},rounding_size={rsize}",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=z,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1)


def bare(ax):
    """Configure axes with equal aspect and no frame."""
    ax.set_aspect("equal")
    ax.axis("off")


# --------------------------------------------------------------------------- 9.1
def fig9_1_bloch_periodic(outroot):
    """Draw Figure 9.1: Bloch-periodic boundaries vs standard periodic boundaries."""
    out = outroot / "section9"
    out.mkdir(parents=True, exist_ok=True)

    # Exact manual parameters (user_manual.md lines 1710-1712)
    f0 = 300e12  # 300 THz (lambda = 1 um), line 1710
    theta0 = np.deg2rad(30.0)  # 30 deg polar angle of incidence, line 1711
    n_inc = 1.0  # line 1712
    k_bloch = k_parallel(f0, theta0, phi=0.0, n=n_inc)  # line 1712
    kx = float(k_bloch[0])
    k0 = 2.0 * np.pi * f0 * n_inc / C0
    kz = float(np.sqrt(k0**2 - kx**2))

    # Mathematical assertions on wakis.bloch API and manual formula (line 1726)
    f_cutoff = C0 * np.linalg.norm(k_bloch) / (2.0 * np.pi * n_inc)
    assert np.isclose(incidence_angle(k_bloch, f0, n=n_inc), theta0, rtol=1e-12)
    assert np.isclose(
        incidence_angle(k_bloch, f_cutoff, n=n_inc), np.pi / 2.0, rtol=1e-12
    )
    assert np.isnan(incidence_angle(k_bloch, 0.95 * f_cutoff, n=n_inc))

    # Quadratic grid from cutoff f_cutoff to 600 THz so the curve reaches 90 deg smoothly
    freqs = f_cutoff + (600e12 - f_cutoff) * np.linspace(0.0, 1.0, 500) ** 2
    theta_deg = np.rad2deg(incidence_angle(k_bloch, freqs, n=n_inc))
    # Manual formula line 1726: theta(f) = arcsin(c |k_par| / (2 pi f n_inc))
    theta_formula = np.rad2deg(
        np.arcsin(C0 * np.linalg.norm(k_bloch) / (2.0 * np.pi * freqs * n_inc))
    )
    assert np.allclose(theta_deg, theta_formula, rtol=1e-12)

    # Check Floquet-Bloch wrap identity F(r + L) = exp(-i k_Bloch . L) F(r) (line 1685)
    L_cell = 1.5e-6  # unit-cell period [m]
    z_test = np.linspace(-1.2e-6, 1.2e-6, 50)
    F_left = np.exp(-1j * (kx * 0.0 + kz * z_test))
    F_right = np.exp(-1j * (kx * L_cell + kz * z_test))
    assert np.allclose(F_right, np.exp(-1j * kx * L_cell) * F_left, rtol=1e-12)

    fig = plt.figure(figsize=(W, H))

    # ---- LEFT PANEL (52% width): 2D unit-cell schematic
    ax = fig.add_axes([0.01, 0.03, 0.51, 0.94])
    bare(ax)

    # Unit cell in normalized coordinates: x in [0, L], z in [0, Hz]
    L = 3.2
    Hz = 3.6
    pml_h = 0.46
    x0, z0 = 0.4, 0.50

    # Cell background
    ax.add_patch(Rectangle((x0, z0), L, Hz, fc="white", ec=GREY, lw=1.2, zorder=1))

    # Top and bottom CPML strips (n_pml=10, lines 1716-1720)
    ax.add_patch(
        Rectangle(
            (x0, z0 + Hz - pml_h), L, pml_h, fc=LGREEN, ec=GREEN, lw=1.4, zorder=3
        )
    )
    ax.add_patch(Rectangle((x0, z0), L, pml_h, fc=LGREEN, ec=GREEN, lw=1.4, zorder=3))
    ax.text(
        x0 + L / 2,
        z0 + Hz - pml_h / 2,
        "bc_low / bc_high: cpml (n_pml=10)",
        ha="center",
        va="center",
        fontsize=12.5,
        color=GREY,
        weight="bold",
        zorder=5,
    )
    ax.text(
        x0 + L / 2,
        z0 + pml_h / 2,
        "bc_low / bc_high: cpml (n_pml=10)",
        ha="center",
        va="center",
        fontsize=12.5,
        color=GREY,
        weight="bold",
        zorder=5,
    )

    # Exact wave crest lines of Re[exp(i(k_x x + k_z z))] = 1 inside physical cell
    lam_draw = 0.72
    kx_d = 2.0 * np.pi / lam_draw * np.sin(theta0)
    kz_d = 2.0 * np.pi / lam_draw * np.cos(theta0)
    z_min_in, z_max_in = z0 + pml_h, z0 + Hz - pml_h
    xs_grid = np.linspace(x0, x0 + L, 200)
    for m in range(-2, 9):
        zs_line = z_min_in + (2.0 * np.pi * m - kx_d * (xs_grid - x0)) / kz_d
        mask = (zs_line >= z_min_in) & (zs_line <= z_max_in)
        if np.count_nonzero(mask) > 2:
            ax.plot(
                xs_grid[mask], zs_line[mask], color=BLUE, lw=1.8, alpha=0.55, zorder=2
            )

    # Highlight left and right Bloch walls in orange
    ax.plot([x0, x0], [z0, z0 + Hz], color=ORANGE, lw=4.2, zorder=6)
    ax.plot([x0 + L, x0 + L], [z0, z0 + Hz], color=ORANGE, lw=4.2, zorder=6)

    # Phase-matching dots on left and right walls
    for zm in np.linspace(z_min_in + 0.35, z_max_in - 0.35, 4):
        ax.add_patch(Circle((x0, zm), 0.055, fc=ORANGE, ec="white", lw=1.0, zorder=7))
        ax.add_patch(
            Circle((x0 + L, zm), 0.055, fc=LORANGE, ec=ORANGE, lw=1.8, zorder=7)
        )

    # Curved orange arrow across the top joining left and right walls
    arrow(
        ax,
        (x0 + 0.05, z0 + Hz + 0.06),
        (x0 + L - 0.05, z0 + Hz + 0.06),
        color=ORANGE,
        lw=2.6,
        rad=-0.24,
        ms=20,
        z=7,
    )
    ax.text(
        x0 + L / 2,
        z0 + Hz + 0.56,
        r"$\mathbf{F}(\mathbf{r} + \mathbf{L}) = \exp(-i\,\mathbf{k}_{\mathrm{Bloch}}\cdot\mathbf{L})\,\mathbf{F}(\mathbf{r})$",
        ha="center",
        va="center",
        fontsize=14.5,
        color=ORANGE,
        weight="bold",
    )
    box(
        ax,
        (x0 + L / 2, z0 + Hz + 0.96),
        2.95,
        0.34,
        "bc = bloch, dtype = np.complex128",
        LORANGE,
        ORANGE,
        fs=12.5,
    )

    # Period L dimension arrow below unit cell
    arrow(
        ax,
        (x0, z0 - 0.22),
        (x0 + L, z0 - 0.22),
        color=GREY,
        lw=1.6,
        style="<|-|>",
        ms=15,
        z=5,
    )
    ax.text(
        x0 + L / 2,
        z0 - 0.22,
        r"$L$",
        ha="center",
        va="center",
        fontsize=15,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
        zorder=6,
    )

    # Wavevector decomposition in the right half of the cell interior
    origin_k = np.array([x0 + 1.68, z_min_in + 0.72])
    len_k = 1.28
    vec_k = len_k * np.array([np.sin(theta0), np.cos(theta0)])
    vec_kx = np.array([vec_k[0], 0.0])
    vec_kz = np.array([0.0, vec_k[1]])

    # Dashed projection lines
    ax.plot(
        [origin_k[0] + vec_kx[0], origin_k[0] + vec_k[0]],
        [origin_k[1], origin_k[1] + vec_k[1]],
        color=GREY,
        ls="--",
        lw=1.2,
        zorder=4,
    )
    ax.plot(
        [origin_k[0], origin_k[0] + vec_k[0]],
        [origin_k[1] + vec_kz[1], origin_k[1] + vec_k[1]],
        color=ORANGE,
        ls="--",
        lw=1.2,
        zorder=4,
    )

    # Vertical normal grey arrow
    arrow(ax, origin_k, origin_k + vec_kz, color=GREY, lw=2.4, ms=18, z=6)

    # Horizontal tangential orange arrow k_|| = (k_x, k_y)
    arrow(ax, origin_k, origin_k + vec_kx, color=ORANGE, lw=2.8, ms=18, z=6)
    ax.text(
        origin_k[0] + 0.5 * vec_kx[0] + 0.20,
        origin_k[1] - 0.18,
        r"$\mathbf{k}_\parallel = (k_x, k_y)$",
        ha="center",
        va="top",
        fontsize=13.5,
        color=ORANGE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88),
    )

    # Incident wavevector blue arrow k
    arrow(ax, origin_k, origin_k + vec_k, color=BLUE, lw=3.2, ms=22, z=7)
    ax.text(
        origin_k[0] + vec_k[0] + 0.08,
        origin_k[1] + vec_k[1] + 0.04,
        r"$\mathbf{k}$",
        ha="left",
        va="bottom",
        fontsize=16,
        color=BLUE,
        weight="bold",
    )

    # Angle arc for theta = 30 deg from the vertical normal
    ang_arc = np.linspace(np.pi / 2.0 - theta0, np.pi / 2.0, 40)
    r_arc = 0.62
    ax.plot(
        origin_k[0] + r_arc * np.cos(ang_arc),
        origin_k[1] + r_arc * np.sin(ang_arc),
        color=BLUE,
        lw=1.8,
        zorder=6,
    )
    ax.text(
        origin_k[0] + 0.08,
        origin_k[1] + r_arc + 0.07,
        r"$\theta = 30^\circ$",
        ha="left",
        va="bottom",
        fontsize=13.5,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
    )

    # Small grey inset at bottom-left showing normal-incidence wavefronts
    ix0, iz0, iw, ih = x0 + 0.16, z_min_in + 0.42, 1.22, 0.96
    ax.add_patch(Rectangle((ix0, iz0), iw, ih, fc=LGREY, ec=GREY, lw=1.4, zorder=5))
    for zh in np.linspace(iz0 + 0.18, iz0 + ih - 0.18, 4):
        ax.plot([ix0 + 0.08, ix0 + iw - 0.08], [zh, zh], color=BLUE, lw=1.6, zorder=6)
    arrow(
        ax,
        (ix0 + iw / 2, iz0 + 0.12),
        (ix0 + iw / 2, iz0 + ih - 0.10),
        color=BLUE,
        lw=2.0,
        ms=14,
        z=7,
    )
    ax.text(
        ix0 + iw / 2 + 0.06,
        iz0 - 0.18,
        r"bc = periodic: $\mathbf{F}(\mathbf{r} + \mathbf{L}) = \mathbf{F}(\mathbf{r})$",
        ha="center",
        va="center",
        fontsize=11.0,
        color="white",
        bbox=dict(boxstyle="round,pad=0.22", fc=GREY, ec="none"),
        zorder=8,
    )

    ax.set_xlim(-0.05, x0 + L + 0.25)
    ax.set_ylim(-0.05, z0 + Hz + 1.25)

    # ---- RIGHT PANEL (48% width): Analytical angle vs frequency curve
    ax2 = fig.add_axes([0.58, 0.14, 0.39, 0.76])
    freqs_thz = freqs / 1e12
    f_cut_thz = f_cutoff / 1e12

    # Evanescent region shading below cutoff (150 THz)
    ax2.axvspan(120, f_cut_thz, color=LGREY, alpha=0.55, zorder=1)
    ax2.axvline(f_cut_thz, color=GREY, ls=":", lw=1.4, zorder=2)

    # Plot exact analytical curve from wakis.bloch.incidence_angle
    ax2.plot(freqs_thz, theta_deg, color=BLUE, lw=3.0, zorder=4)

    # Reference point (300 THz, 30 deg)
    f0_thz = f0 / 1e12
    th0_deg = np.rad2deg(theta0)
    ax2.plot([120, f0_thz], [th0_deg, th0_deg], color=ORANGE, ls="--", lw=1.6, zorder=3)
    ax2.plot([f0_thz, f0_thz], [0, th0_deg], color=ORANGE, ls="--", lw=1.6, zorder=3)
    ax2.plot(f0_thz, th0_deg, "o", color=ORANGE, ms=10, mec="white", mew=1.5, zorder=6)

    ax2.annotate(
        r"$f_0 = 300\ \mathrm{THz}\ (\lambda = 1\ \mu\mathrm{m}),\ \theta = 30^\circ$",
        xy=(f0_thz, th0_deg),
        xytext=(f0_thz + 25, th0_deg + 14),
        fontsize=12.5,
        color=ORANGE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc=LORANGE, ec=ORANGE, lw=1.2),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6, shrinkA=0, shrinkB=6),
        zorder=7,
    )

    # Formula and API callout boxes
    ax2.text(
        385,
        76,
        r"$\theta(f) = \arcsin\!\left(\frac{c\,|\mathbf{k}_\parallel|}{2\pi f\,n_{\mathrm{inc}}}\right)$",
        ha="center",
        va="center",
        fontsize=15,
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.35", fc=LBLUE, ec=BLUE, lw=1.4),
        zorder=6,
    )
    ax2.text(
        385,
        60,
        "wakis.bloch.incidence_angle(k_bloch, freqs)",
        ha="center",
        va="center",
        fontsize=12.0,
        color=GREY,
        bbox=dict(boxstyle="round,pad=0.28", fc=LGREY, ec=GREY, lw=1.1),
        zorder=6,
    )

    ax2.set_xlim(120, 600)
    ax2.set_ylim(0, 92)
    ax2.set_xticks([150, 300, 450, 600])
    ax2.set_xticklabels(["150 THz", "300 THz", "450 THz", "600 THz"], fontsize=12.5)
    ax2.set_yticks([0, 30, 60, 90])
    ax2.set_yticklabels(
        [r"$0^\circ$", r"$30^\circ$", r"$60^\circ$", r"$90^\circ$"], fontsize=12.5
    )
    ax2.set_xlabel(r"frequency $f$", fontsize=15)
    ax2.set_ylabel(r"incidence angle $\theta(f)$", fontsize=15)
    for sp_name in ("top", "right"):
        ax2.spines[sp_name].set_visible(False)

    fig.savefig(out / "fig9_1_bloch_periodic.png", facecolor="white")
    plt.close(fig)


# -------------------------------------------------------------------------- 10.5
def fig10_5_harminv_resonance(outroot):
    """Draw Figure 10.5: High-resolution resonance extraction with harminv vs FFT."""
    out = outroot / "section10"
    out.mkdir(parents=True, exist_ok=True)

    # Exact manual parameters and formula (user_manual.md lines 1914, 1918, 1920):
    # x(t) = sum_{k=1}^K a_k exp(2 pi i (f_k + i g_k) t),  Q_k = f_k / (2 g_k)
    f0 = 2.0e9  # 2 GHz reference frequency (line 1914)
    Q_ref = 1.0e5  # Q = 10^5 at f0 = 2 GHz -> Q / f0 = 50 us (line 1914)
    tau_fft = Q_ref / f0
    assert np.isclose(tau_fft, 50e-6, rtol=1e-12)

    f_true = np.array([1.93e9, 2.06e9])  # two cavity modes around f0 = 2 GHz [Hz]
    Q_true = np.array([1.0e5, 1.2e6])  # high Q including Q > 10^6 (lines 1914, 1920)
    g_true = f_true / (2.0 * Q_true)  # Q_k = f_k / (2 g_k) (line 1918)
    a_true = np.array([1.0, 0.78])

    # Short post-excitation window: N = 600 steps (within 500-1500 steps, line 1920)
    n_steps = 600
    dt = 1.0e-11  # 10 ps step -> T_sim = 6 ns << 50 us
    t_win = np.arange(n_steps) * dt
    T_sim = n_steps * dt
    df_fft = 1.0 / T_sim  # coarse FFT bin spacing (line 1914)

    # Synthesize exact signal from line 1918 formula
    F_true = f_true + 1j * g_true
    x_win = np.sum(
        a_true[:, None] * np.exp(2j * np.pi * F_true[:, None] * t_win[None, :]), axis=0
    )

    # Recover poles with real wakis.resonances.harminv (both 'fdm' and 'pencil', line 1920)
    res_fdm = harminv(x_win, dt, fmin=1.6e9, fmax=2.4e9, method="fdm")
    res_pencil = harminv(x_win, dt, fmin=1.6e9, fmax=2.4e9, method="pencil")

    assert len(res_fdm) == 2 and len(res_pencil) == 2
    assert np.allclose(res_fdm.frequency, f_true, rtol=1e-7)
    assert np.allclose(res_pencil.frequency, f_true, rtol=1e-7)
    assert np.allclose(res_fdm.Q, Q_true, rtol=1e-3)
    assert np.allclose(res_pencil.Q, Q_true, rtol=1e-3)
    assert T_sim < 1e-3 * tau_fft and df_fft > 1e3 * (2.0 * g_true.max())

    fig = plt.figure(figsize=(W, H))

    # ---- LEFT PANEL (48% width): Short recorded window vs long decay envelope
    ax1 = fig.add_axes([0.05, 0.22, 0.41, 0.68])

    # Plot Re[x(t)] in [0, T_sim] (normalized time u = t / T_sim in [0, 1])
    u_win = t_win / T_sim
    ax1.plot(u_win, x_win.real / np.sum(a_true), color=BLUE, lw=2.0, zorder=4)

    # Continuation for t > T_sim (compressed schematic tail showing exponential ring-down over 50 us)
    u_tail = np.linspace(1.0, 2.15, 300)
    # Map u_tail in [1, 2.15] to physical decay time up to 2.2 * (Q / f0)
    t_tail_phys = T_sim + (u_tail - 1.0) / 1.15 * (2.2 * tau_fft)
    env_tail = np.sum(
        a_true[:, None] * np.exp(-2.0 * np.pi * g_true[:, None] * t_tail_phys[None, :]),
        axis=0,
    ) / np.sum(a_true)
    ax1.plot(u_tail, env_tail, color=GREY, ls="--", lw=2.2, zorder=3)
    ax1.plot(u_tail, -env_tail, color=GREY, ls="--", lw=2.2, zorder=3)
    ax1.fill_between(u_tail, -env_tail, env_tail, color=LGREY, alpha=0.45, zorder=1)

    # Vertical red line at t = T_sim
    ax1.axvline(1.0, color=RED, lw=2.4, zorder=5)
    ax1.text(
        0.97,
        1.14,
        r"$T_{\mathrm{sim}}$: a short window",
        ha="right",
        va="bottom",
        fontsize=13.5,
        color=RED,
        weight="bold",
    )
    ax1.annotate(
        "",
        xy=(1.0, 1.08),
        xytext=(0.0, 1.08),
        arrowprops=dict(arrowstyle="<|-|>", color=BLUE, lw=1.8),
    )

    # Grey annotation for t > T_sim
    ax1.text(
        1.58,
        0.74,
        r"$T_{\mathrm{sim}} \gg Q / f_0 = 50\ \mu\mathrm{s}$"
        "\n"
        r"($\sim 10^7$ time steps)"
        "\n"
        r"needed by FFT for"
        "\n"
        r"$Q = 10^5$ at $f_0 = 2\ \mathrm{GHz}$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=GREY,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=1.1),
        zorder=6,
    )

    ax1.axhline(0.0, color=GREY, lw=0.8, alpha=0.5)
    ax1.set_xlim(0.0, 2.18)
    ax1.set_ylim(-1.18, 1.28)
    ax1.set_xticks([0.0, 1.0, 2.0])
    ax1.set_xticklabels(
        [
            r"$0$",
            r"$T_{\mathrm{sim}}$",
            r"$t > T_{\mathrm{sim}}\ (\sim 50\ \mu\mathrm{s})$",
        ],
        fontsize=12,
    )
    ax1.set_yticks([-1.0, 0.0, 1.0])
    ax1.set_xlabel(r"time $t$", fontsize=14.5)
    ax1.set_ylabel(r"ring-down signal $\mathrm{Re}[x(t)]$", fontsize=14.5)
    for sp_name in ("top", "right"):
        ax1.spines[sp_name].set_visible(False)

    # Exact model equation under the left panel (line 1918)
    fig.text(
        0.255,
        0.06,
        r"$x(t) = \sum_{k=1}^K a_k \exp\!\left(2\pi i (f_k + i g_k) t\right),\qquad Q_k = \frac{f_k}{2 g_k}$",
        ha="center",
        va="center",
        fontsize=15.0,
        color="black",
        bbox=dict(boxstyle="round,pad=0.32", fc=LBLUE, ec=BLUE, lw=1.3),
    )

    # ---- RIGHT PANEL (52% width): Coarse FFT bins vs sharp harminv poles
    ax2 = fig.add_axes([0.54, 0.20, 0.43, 0.70])

    # Compute exact unpadded FFT bin frequencies and magnitudes on the short window
    f_fft = np.fft.fftfreq(n_steps, d=dt)
    order_fft = np.argsort(f_fft)
    f_fft_ghz = f_fft[order_fft] / 1e9
    mag_fft = np.abs(np.fft.fft(x_win))[order_fft]
    mag_fft = mag_fft / mag_fft.max() * 0.58

    mask_band = (f_fft_ghz >= 1.55) & (f_fft_ghz <= 2.45)
    fb = f_fft_ghz[mask_band]
    mb = mag_fft[mask_band]
    bin_w = (df_fft / 1e9) * 0.92

    # Plot coarse FFT step-bars in red/grey
    ax2.bar(
        fb,
        mb,
        width=bin_w,
        color=LGREY,
        edgecolor=RED,
        lw=1.4,
        zorder=2,
        label=r"short-window $|\mathrm{FFT}(x)|$",
    )
    ax2.step(fb, mb, where="mid", color=RED, lw=2.0, zorder=3)

    # Double-headed red arrow showing coarse bin spacing df_FFT = 1 / T_sim
    k_bin = 0
    x_b0, x_b1 = fb[k_bin], fb[k_bin + 1]
    y_arr = max(mb[k_bin], mb[k_bin + 1]) + 0.14
    ax2.annotate(
        "",
        xy=(x_b1, y_arr),
        xytext=(x_b0, y_arr),
        arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=2.0),
        zorder=6,
    )
    ax2.plot([x_b0, x_b0], [mb[k_bin], y_arr + 0.03], color=RED, ls=":", lw=1.2)
    ax2.plot([x_b1, x_b1], [mb[k_bin + 1], y_arr + 0.03], color=RED, ls=":", lw=1.2)
    ax2.text(
        0.5 * (x_b0 + x_b1),
        y_arr + 0.05,
        r"$\Delta f_{\mathrm{FFT}} = 1 / T_{\mathrm{sim}}$",
        ha="center",
        va="bottom",
        fontsize=13.5,
        color=RED,
        weight="bold",
    )

    # Overlay exact pole locations (f_k, Q_k) recovered by harminv as tall sharp blue stems
    f_rec_ghz = res_fdm.frequency / 1e9
    amp_rec = np.abs(res_fdm.amplitude) / np.abs(res_fdm.amplitude).max() * 0.96
    for fk, ak in zip(f_rec_ghz, amp_rec):
        ax2.plot([fk, fk], [0.0, ak], color=BLUE, lw=3.4, zorder=5)
        ax2.plot(fk, ak, "o", color=BLUE, ms=11, mec="white", mew=1.5, zorder=6)
        ax2.text(
            fk,
            ak + 0.04,
            r"$(f_k,\ Q_k)$",
            ha="center",
            va="bottom",
            fontsize=13.0,
            color=BLUE,
            weight="bold",
        )

    # Labels and badges required by spec
    ax2.text(
        2.00,
        1.17,
        'harminv (ResonanceMonitor): method="fdm" or "pencil"',
        ha="center",
        va="center",
        fontsize=13.0,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc=LBLUE, ec=BLUE, lw=1.4),
        zorder=7,
    )
    fig.text(
        0.755,
        0.06,
        r"$f_k$ to 4–6 significant digits and $Q > 10^6$ from a short window",
        ha="center",
        va="center",
        fontsize=13.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.32", fc=LGREEN, ec=GREEN, lw=1.5),
    )

    ax2.set_xlim(1.58, 2.42)
    ax2.set_ylim(0.0, 1.28)
    ax2.set_xticks([1.6, 1.8, 2.0, 2.2, 2.4])
    ax2.set_xticklabels(
        ["1.6 GHz", "1.8 GHz", "2.0 GHz", "2.2 GHz", "2.4 GHz"], fontsize=12
    )
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_xlabel(r"frequency $f$", fontsize=14.5)
    ax2.set_ylabel(r"normalized amplitude / $|\mathrm{FFT}(x)|$", fontsize=14.5)
    for sp_name in ("top", "right"):
        ax2.spines[sp_name].set_visible(False)

    fig.savefig(out / "fig10_5_harminv_resonance.png", facecolor="white")
    plt.close(fig)


# -------------------------------------------------------------------------- 11.1
def _circumcentre(a, b, c):
    """Exact circumcentre of 2D triangle (a, b, c)."""
    a, b, c = map(np.asarray, (a, b, c))
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    ux = (
        (a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])
    ) / d
    uy = (
        (a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])
    ) / d
    return np.array([ux, uy])


def _shoelace_area(poly):
    """Signed area of a 2D polygon given its ordered vertices."""
    p = np.asarray(poly)
    return 0.5 * float(
        np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    )


def fig11_1_triangle_and_polygon_fit(outroot):
    """Draw Figure 11.1: 2D unstructured and polygonal FIT discretizations."""
    out = outroot / "section11"
    out.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H))
    panel_w, panel_h = 0.225, 0.62
    y_bot = 0.22
    xs_left = [0.02 + k * 0.245 for k in range(4)]

    # ---- PANEL 1: UnstructuredFIT2D: hodge="voronoi"
    ax1 = fig.add_axes([xs_left[0], y_bot, panel_w, panel_h])
    bare(ax1)
    th6 = np.linspace(0, 2 * np.pi, 7)[:-1]
    pts1 = np.vstack(
        [
            [[0.0, 0.0]],
            1.0 * np.column_stack([np.cos(th6), np.sin(th6)]),
            1.78 * np.column_stack([np.cos(th6 + np.pi / 6), np.sin(th6 + np.pi / 6)]),
            1.92 * np.column_stack([np.cos(th6), np.sin(th6)]),
        ]
    )
    tri1 = Delaunay(pts1).simplices
    # Orient counter-clockwise
    for idx_t, t in enumerate(tri1):
        if _shoelace_area(pts1[t]) < 0:
            tri1[idx_t] = t[[0, 2, 1]]
    ccs1 = np.array([_circumcentre(*pts1[t]) for t in tri1])

    # Assert every triangle is strictly acute and every interior Voronoi dual edge is orthogonal to its primal edge
    edge_to_tris = {}
    for ti, t in enumerate(tri1):
        a, b, c = pts1[t]
        la2, lb2, lc2 = np.sum((b - c) ** 2), np.sum((a - c) ** 2), np.sum((a - b) ** 2)
        assert la2 + lb2 > lc2 and lb2 + lc2 > la2 and lc2 + la2 > lb2
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_to_tris.setdefault((min(u, v), max(u, v)), []).append(ti)

    for (u, v), tlist in edge_to_tris.items():
        if len(tlist) == 2:
            e_vec = pts1[v] - pts1[u]
            d_vec = ccs1[tlist[1]] - ccs1[tlist[0]]
            assert (
                abs(np.dot(e_vec, d_vec))
                / (np.linalg.norm(e_vec) * np.linalg.norm(d_vec))
                < 1e-12
            )

    # Shade central Voronoi dual polygon around node 0
    inc1 = [ti for ti, t in enumerate(tri1) if 0 in t]
    ang1 = np.arctan2(ccs1[inc1, 1], ccs1[inc1, 0])
    vor_poly0 = ccs1[np.array(inc1)[np.argsort(ang1)]]
    smesh1 = SimplexMesh(pts1, tri1)
    usolver1 = UnstructuredFIT2D(smesh1, polarization="TM", hodge="voronoi")
    assert np.isclose(
        _shoelace_area(vor_poly0),
        float(np.asarray(usolver1._Pn.sum(axis=1)).ravel()[0]),
        rtol=1e-12,
    )

    ax1.add_patch(
        Polygon(vor_poly0, closed=True, fc=LORANGE, ec="none", alpha=0.85, zorder=1)
    )
    for u, v in edge_to_tris:
        ax1.plot(pts1[[u, v], 0], pts1[[u, v], 1], color=BLUE, lw=1.9, zorder=3)
    for (u, v), tlist in edge_to_tris.items():
        if len(tlist) == 2:
            c_a, c_b = ccs1[tlist[0]], ccs1[tlist[1]]
            ax1.plot(
                [c_a[0], c_b[0]],
                [c_a[1], c_b[1]],
                color=ORANGE,
                ls="--",
                lw=1.8,
                zorder=4,
            )
    ax1.plot(ccs1[:, 0], ccs1[:, 1], "o", color=ORANGE, ms=5.5, zorder=5)
    ax1.plot(pts1[0, 0], pts1[0, 1], "o", color=BLUE, ms=7.0, zorder=6)

    # Right-angle mark between primary edge (0 -> 1) and its dual edge
    p_mid = 0.5 * (pts1[0] + pts1[1])
    t_hat = (pts1[1] - pts1[0]) / np.linalg.norm(pts1[1] - pts1[0])
    n_hat = np.array([-t_hat[1], t_hat[0]])
    s_ra = 0.12
    ra_pts = np.array(
        [p_mid + s_ra * t_hat, p_mid + s_ra * (t_hat + n_hat), p_mid + s_ra * n_hat]
    )
    ax1.plot(ra_pts[:, 0], ra_pts[:, 1], color=GREY, lw=1.5, zorder=6)

    ax1.set_xlim(-2.05, 2.05)
    ax1.set_ylim(-2.05, 2.05)
    ax1.set_title(
        'UnstructuredFIT2D:\nhodge="voronoi"', fontsize=13.5, weight="bold", pad=10
    )
    ax1.text(
        0.5,
        -0.08,
        "orthogonal dual: diagonal Hodge,\ndiv curl = 0",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=11.0,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc=LBLUE, ec=BLUE, lw=1.2),
    )

    # ---- PANEL 2: UnstructuredFIT2D: hodge="barycentric"
    ax2 = fig.add_axes([xs_left[1], y_bot, panel_w, panel_h])
    bare(ax2)
    # Interior vertex at (0, 0) surrounded by 5 outer vertices; top triangle (0, 1, 2) is obtuse at node 0 (> 115 deg)
    ang2 = np.deg2rad([22.0, 158.0, 215.0, 270.0, 325.0])
    rad2 = np.array([1.72, 1.72, 1.55, 1.50, 1.55])
    pts2 = np.vstack(
        [[[0.0, 0.0]], np.column_stack([rad2 * np.cos(ang2), rad2 * np.sin(ang2)])]
    )
    tri2 = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1]], dtype=int)

    # Assert triangle 0 is obtuse (> 90 deg at vertex 0), yet barycentric dual area equals 1/3 sum(|T|) > 0
    v1_vec, v2_vec = pts2[1] - pts2[0], pts2[2] - pts2[0]
    obtuse_angle_deg = np.rad2deg(
        np.arccos(
            np.dot(v1_vec, v2_vec) / (np.linalg.norm(v1_vec) * np.linalg.norm(v2_vec))
        )
    )
    assert obtuse_angle_deg > 110.0

    bary_loop = []
    tri_areas2 = []
    for k in range(len(tri2)):
        t = tri2[k]
        tri_areas2.append(_shoelace_area(pts2[t]))
        m_edge = 0.5 * (pts2[0] + pts2[t[1]])
        c_tri = pts2[t].mean(axis=0)
        bary_loop.extend([m_edge, c_tri])
    bary_loop = np.array(bary_loop)
    bary_area0 = _shoelace_area(bary_loop)
    smesh2 = SimplexMesh(pts2, tri2)
    usolver2 = UnstructuredFIT2D(
        smesh2, polarization="TM", bc="pmc", hodge="barycentric"
    )
    assert bary_area0 > 0.0
    assert np.isclose(bary_area0, np.sum(tri_areas2) / 3.0, rtol=1e-12)
    assert np.isclose(
        bary_area0, float(np.asarray(usolver2._Pn.sum(axis=1)).ravel()[0]), rtol=1e-12
    )

    # Highlight the obtuse triangle in light red/blue and the barycentric dual cell in light orange
    ax2.add_patch(
        Polygon(pts2[tri2[0]], closed=True, fc=LRED, ec="none", alpha=0.45, zorder=1)
    )
    ax2.add_patch(
        Polygon(
            bary_loop, closed=True, fc=LORANGE, ec=ORANGE, ls="--", lw=2.0, zorder=2
        )
    )
    for t in tri2:
        p_t = pts2[t]
        ax2.plot(
            [p_t[0, 0], p_t[1, 0], p_t[2, 0], p_t[0, 0]],
            [p_t[0, 1], p_t[1, 1], p_t[2, 1], p_t[0, 1]],
            color=BLUE,
            lw=2.0,
            zorder=3,
        )
    centroids2 = np.array([pts2[t].mean(axis=0) for t in tri2])
    midpoints2 = np.array([0.5 * (pts2[0] + pts2[t[1]]) for t in tri2])
    ax2.plot(centroids2[:, 0], centroids2[:, 1], "o", color=ORANGE, ms=6.5, zorder=5)
    ax2.plot(midpoints2[:, 0], midpoints2[:, 1], "s", color=BLUE, ms=4.5, zorder=5)
    ax2.plot(0.0, 0.0, "o", color=BLUE, ms=7.0, zorder=6)
    ax2.text(
        0.0,
        1.15,
        "obtuse triangle",
        ha="center",
        va="center",
        fontsize=11.5,
        color=RED,
        weight="bold",
        zorder=6,
    )

    ax2.set_xlim(-2.05, 2.05)
    ax2.set_ylim(-2.05, 2.05)
    ax2.set_title(
        'UnstructuredFIT2D:\nhodge="barycentric"', fontsize=13.5, weight="bold", pad=10
    )
    ax2.text(
        0.5,
        -0.08,
        "positive dual area on\nobtuse triangles",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=11.0,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc=LORANGE, ec=ORANGE, lw=1.2),
    )

    # ---- PANEL 3: PolygonFIT2D: voronoi_disk_mesh
    ax3 = fig.add_axes([xs_left[2], y_bot, panel_w, panel_h])
    bare(ax3)
    vmesh = voronoi_disk_mesh([1.0e-6], 0.22e-6, merge_tol=1e-11, lloyd=10)
    assert np.max(vmesh.orthogonality_defect) < 1e-6
    vsolver = PolygonFIT2D(
        vmesh, materials={0: [4.0, 1.0]}, polarization="TM", hodge="orthogonal"
    )
    assert np.all(vsolver.mass > 0) and np.all(vsolver.stiff >= 0)

    scale_v = 1.85 / 1.0e-6
    vnodes = vmesh.nodes * scale_v
    vdual = vmesh.dual_point * scale_v
    for ci in range(vmesh.n_cells):
        c_idx = vmesh.cell(ci)
        fc_cell = LBLUE if np.hypot(*vdual[ci]) < 1.05 else "white"
        ax3.add_patch(
            Polygon(vnodes[c_idx], closed=True, fc=fc_cell, ec=BLUE, lw=1.7, zorder=2)
        )

    # Plot orthogonal Delaunay dual edges between neighbouring generators
    edge_cells_v = {}
    for cc, ce in zip(vmesh.corner_cell, vmesh.corner_edge):
        edge_cells_v.setdefault(int(ce), []).append(int(cc))
    for cells_pair in edge_cells_v.values():
        if len(cells_pair) == 2:
            g0, g1 = vdual[cells_pair[0]], vdual[cells_pair[1]]
            ax3.plot(
                [g0[0], g1[0]],
                [g0[1], g1[1]],
                color=ORANGE,
                ls="--",
                lw=1.2,
                alpha=0.85,
                zorder=3,
            )
    ax3.plot(vdual[:, 0], vdual[:, 1], "o", color=ORANGE, ms=3.8, zorder=4)

    ax3.set_xlim(-2.05, 2.05)
    ax3.set_ylim(-2.05, 2.05)
    ax3.set_title(
        "PolygonFIT2D:\nvoronoi_disk_mesh", fontsize=13.5, weight="bold", pad=10
    )
    ax3.text(
        0.5,
        -0.08,
        'hodge="orthogonal": ½ lattice dispersion\n(1.95× fewer unknowns on B1 TM)',
        transform=ax3.transAxes,
        ha="center",
        va="top",
        fontsize=10.8,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.24", fc=LGREEN, ec=GREEN, lw=1.2),
    )

    # ---- PANEL 4: PolygonFIT2D: hybrid_circles_mesh
    ax4 = fig.add_axes([xs_left[3], y_bot, panel_w, panel_h])
    bare(ax4)
    # Build real hybrid_circles_mesh around r = 1 um using exact manual parameters (lines 1993-2002)
    h_hyb = 0.08e-6
    hmesh, hinfo = hybrid_circles_mesh(
        [(0.0, 0.0, 1.0e-6)],
        h_in=h_hyb,
        h_out=2.0 * h_hyb,
        half_width=1.9e-6,
        pml_cells=0,
    )
    assert hinfo["n_squares"] > 0 and hinfo["n_triangles"] > 0
    assert np.max(hmesh.orthogonality_defect) < 1e-6
    hsolver = PolygonFIT2D(
        hmesh, materials={0: [4.0, 1.0]}, polarization="TE", hodge="auto"
    )
    assert np.all(hsolver.mass > 0)

    # Identify transition cells that share an edge between a triangle (size 3) and a square (size 4)
    edge_cells_h = {}
    for cc, ce in zip(hmesh.corner_cell, hmesh.corner_edge):
        edge_cells_h.setdefault(int(ce), []).append(int(cc))
    trans_cells = set()
    for cells_pair in edge_cells_h.values():
        if len(cells_pair) == 2:
            s0, s1 = hmesh.cell_size[cells_pair[0]], hmesh.cell_size[cells_pair[1]]
            if s0 != s1:
                trans_cells.update(cells_pair)

    # Plot first-quadrant zoom [0, 1.85] um mapped to [-2.05, 2.05]
    hnodes_um = hmesh.nodes * 1e6
    for ci in range(hmesh.n_cells):
        c_idx = hmesh.cell(ci)
        poly_xy = hnodes_um[c_idx]
        cen = poly_xy.mean(axis=0)
        if cen[0] < -0.15 or cen[1] < -0.15:
            continue
        if ci in trans_cells:
            fc_h, ec_h, lw_h, z_h = LGREEN, GREEN, 1.9, 3
        elif hmesh.cell_tags[ci] == 0:
            fc_h, ec_h, lw_h, z_h = LBLUE, BLUE, 1.5, 2
        else:
            fc_h, ec_h, lw_h, z_h = "white", BLUE, 1.5, 2
        ax4.add_patch(
            Polygon(poly_xy, closed=True, fc=fc_h, ec=ec_h, lw=lw_h, zorder=z_h)
        )

    # Exact circular interface arc r = 1 um in dark grey
    th_arc = np.linspace(-0.08, 0.55 * np.pi, 200)
    ax4.plot(1.0 * np.cos(th_arc), 1.0 * np.sin(th_arc), color=GREY, lw=3.2, zorder=5)
    ax4.text(
        0.52,
        0.52,
        r"$r = 1\ \mu\mathrm{m}$",
        ha="center",
        va="center",
        fontsize=12.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=GREY, lw=1.0),
        zorder=6,
    )

    ax4.set_xlim(-0.05, 1.82)
    ax4.set_ylim(-0.05, 1.82)
    ax4.set_title(
        "PolygonFIT2D:\nhybrid_circles_mesh", fontsize=13.5, weight="bold", pad=10
    )
    ax4.text(
        0.5,
        -0.08,
        'hodge="auto": diagonal in bulk,\nhodge="mfd" on transition cells',
        transform=ax4.transAxes,
        ha="center",
        va="top",
        fontsize=10.8,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.24", fc=LGREEN, ec=GREEN, lw=1.2),
    )

    fig.savefig(out / "fig11_1_triangle_and_polygon_fit.png", facecolor="white")
    plt.close(fig)


# -------------------------------------------------------------------------- 11.2
def fig11_2_subgrid_and_subtri(outroot):
    """Draw Figure 11.2: 2:1 Cartesian subgridding and hybrid quadtree + triangles."""
    out = outroot / "section11"
    out.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H))

    # ---- LEFT PANEL: Symmetric hanging-node prolongation P / P^T
    ax1 = fig.add_axes([0.015, 0.24, 0.30, 0.63])
    bare(ax1)

    # Verify exact 2:1 interface prolongation P on a 2-cell coarse / fine QuadMesh
    qmesh_demo = QuadMesh(
        [0.0, 1.0, 2.0],
        [0.0, 1.0],
        ratio=2,
        max_level=1,
        refine=lambda lev, x0, x1, y0, y1: x0 >= 0.5,
    )
    assert int(qmesh_demo.hanging.sum()) == 1
    assert np.allclose(np.asarray(qmesh_demo.P.sum(axis=1)).ravel(), 1.0, rtol=1e-12)
    h_row = qmesh_demo.P[np.flatnonzero(qmesh_demo.hanging)[0]]
    assert np.allclose(np.sort(h_row.data), [0.5, 0.5], rtol=1e-12)

    # Draw 1 coarse square cell on left [-1.2, 0] x [0, 1.2] and 2 fine square cells on right [0, 0.6] x [0, 1.2]
    Hc = 1.30
    hf = Hc / 2.0
    ax1.add_patch(Rectangle((-Hc, 0.0), Hc, Hc, fc=LBLUE, ec=BLUE, lw=2.4, zorder=1))
    ax1.add_patch(Rectangle((0.0, 0.0), hf, hf, fc="white", ec=BLUE, lw=2.2, zorder=1))
    ax1.add_patch(Rectangle((0.0, hf), hf, hf, fc="white", ec=BLUE, lw=2.2, zorder=1))

    # Master nodes (blue circles)
    master_pts = [
        (-Hc, 0.0),
        (-Hc, Hc),
        (0.0, 0.0),
        (0.0, Hc),
        (hf, 0.0),
        (hf, hf),
        (hf, Hc),
    ]
    for mx, my in master_pts:
        ax1.plot(mx, my, "o", color=BLUE, ms=9.5, mec="white", mew=1.3, zorder=5)
    ax1.text(
        0.0,
        Hc + 0.12,
        r"$u$",
        ha="center",
        va="bottom",
        fontsize=15,
        color=BLUE,
        weight="bold",
    )
    ax1.text(
        0.0,
        -0.12,
        r"$u$",
        ha="center",
        va="top",
        fontsize=15,
        color=BLUE,
        weight="bold",
    )
    ax1.text(
        -Hc,
        Hc + 0.12,
        r"$u$",
        ha="center",
        va="bottom",
        fontsize=15,
        color=BLUE,
        weight="bold",
    )

    # Hanging node at midpoint (0, hf) (orange diamond)
    ax1.plot(0.0, hf, "D", color=ORANGE, ms=11.0, mec="white", mew=1.4, zorder=6)
    ax1.text(
        0.10,
        hf,
        "hanging node",
        ha="left",
        va="center",
        fontsize=12.0,
        color=ORANGE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88),
        zorder=7,
    )

    # Blue arrows from coarse master nodes to hanging node: linear prolongation u_all = P u
    arrow(
        ax1,
        (-0.06, 0.08),
        (-0.06, hf - 0.08),
        color=BLUE,
        lw=2.4,
        rad=-0.35,
        ms=17,
        z=6,
    )
    arrow(
        ax1,
        (-0.06, Hc - 0.08),
        (-0.06, hf + 0.08),
        color=BLUE,
        lw=2.4,
        rad=0.35,
        ms=17,
        z=6,
    )
    ax1.text(
        -0.65,
        hf + 0.22,
        "linear prolongation:\n" + r"$u_{\mathrm{all}} = P\,u$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=BLUE, lw=1.1),
        zorder=7,
    )

    # Orange reverse arrows from hanging node to master nodes: fine-to-coarse restriction P^T
    arrow(
        ax1,
        (0.06, hf - 0.08),
        (0.06, 0.08),
        color=ORANGE,
        lw=2.4,
        rad=-0.35,
        ms=17,
        z=6,
    )
    arrow(
        ax1,
        (0.06, hf + 0.08),
        (0.06, Hc - 0.08),
        color=ORANGE,
        lw=2.4,
        rad=0.35,
        ms=17,
        z=6,
    )
    ax1.text(
        -0.20,
        -0.34,
        r"fine-to-coarse restriction: $P^{T}$ (exact transpose)",
        ha="center",
        va="top",
        fontsize=11.5,
        color=ORANGE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc=LORANGE, ec=ORANGE, lw=1.1),
        clip_on=False,
        zorder=7,
    )

    ax1.set_xlim(-1.45, 1.05)
    ax1.set_ylim(-0.62, 1.62)
    ax1.set_title(
        "Symmetric hanging-node\nprolongation $P$ / $P^{T}$",
        fontsize=13.5,
        weight="bold",
        pad=8,
    )

    # Update equations below Left Panel (user_manual.md line 2023)
    fig.text(
        0.165,
        0.145,
        r"$f^{n+1/2} = f^{n-1/2} - \Delta t\,(G P)\,u^n$"
        "\n"
        r"$D\,(u^{n+1} - u^n) = \Delta t\,(G P)^{T} W\,f^{n+1/2}$",
        ha="center",
        va="center",
        fontsize=12.5,
        color="black",
        bbox=dict(boxstyle="round,pad=0.28", fc=LBLUE, ec=BLUE, lw=1.2),
    )
    fig.text(
        0.165,
        0.045,
        r"energy drift $\leq 4\times 10^{-15}$ over $10^5$ steps",
        ha="center",
        va="center",
        fontsize=11.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc=LGREEN, ec=GREEN, lw=1.2),
    )

    # ---- CENTRE PANEL: SubgridFIT2D (QuadMesh, ratio=2, max_level=2)
    ax2 = fig.add_axes([0.35, 0.20, 0.30, 0.67])
    bare(ax2)

    # Build real 2-level 2:1 QuadMesh refined near r = 1.0e-6 (user_manual.md lines 2031-2038)
    a_rad = 1.0e-6
    x_sg = np.linspace(-1.8e-6, 1.8e-6, 19)
    rim_fn = lambda d: lambda xm, ym, hd: np.abs(np.hypot(xm, ym) - a_rad) < d
    qmesh = QuadMesh(
        x_sg,
        x_sg,
        ratio=2,
        max_level=2,
        refine=refine_near([rim_fn(0.26e-6), rim_fn(0.12e-6)]),
    )
    geo_sg = Geometry([SubgridCircle((0.0, 0.0), a_rad, 4.0)])
    sg_solver = SubgridFIT2D(qmesh, geo_sg, polarization="TE")

    # Assert P rows sum to 1 and (G P)^T W (G P) is symmetric
    assert np.allclose(np.asarray(qmesh.P.sum(axis=1)).ravel(), 1.0, rtol=1e-12)
    GP = sg_solver.K
    K_sg = GP.T @ sp.diags(sg_solver.W) @ GP
    assert np.allclose((K_sg - K_sg.T).data, 0.0, atol=1e-12)

    c_sg = qmesh.cells
    x0s, x1s, y0s, y1s = (
        c_sg["x0"] * 1e6,
        c_sg["x1"] * 1e6,
        c_sg["y0"] * 1e6,
        c_sg["y1"] * 1e6,
    )
    for i_c in range(qmesh.n_cells):
        if x1s[i_c] <= 0.0 or y1s[i_c] <= 0.0:
            continue
        r_min = np.hypot(max(x0s[i_c], 0.0), max(y0s[i_c], 0.0))
        r_max = np.hypot(x1s[i_c], y1s[i_c])
        cut = (r_min < 1.0 - 1e-6) and (r_max > 1.0 + 1e-6)
        inside = r_max <= 1.0 + 1e-6
        fc_c = LGREEN if cut else (LBLUE if inside else "white")
        ec_c = GREEN if cut else BLUE
        lw_c = 1.8 if cut else 1.2
        z_c = 3 if cut else 2
        ax2.add_patch(
            Rectangle(
                (x0s[i_c], y0s[i_c]),
                x1s[i_c] - x0s[i_c],
                y1s[i_c] - y0s[i_c],
                fc=fc_c,
                ec=ec_c,
                lw=lw_c,
                zorder=z_c,
            )
        )

    th_q = np.linspace(0.0, 0.5 * np.pi, 200)
    ax2.plot(1.0 * np.cos(th_q), 1.0 * np.sin(th_q), color=GREY, lw=3.0, zorder=5)
    ax2.text(
        0.38,
        0.38,
        r"$r = 1.0\times 10^{-6}$",
        ha="center",
        va="center",
        fontsize=11.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=GREY, lw=1.0),
        zorder=6,
    )
    ax2.text(
        0.82,
        1.48,
        "subpixel material averaging\non cut leaf cells",
        ha="center",
        va="center",
        fontsize=11.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc=LGREEN, ec=GREEN, lw=1.2),
        zorder=7,
    )

    ax2.set_xlim(0.0, 1.65)
    ax2.set_ylim(0.0, 1.65)
    ax2.set_title(
        "SubgridFIT2D\n(QuadMesh, ratio=2, max_level=2)",
        fontsize=13.5,
        weight="bold",
        pad=8,
    )
    fig.text(
        0.50,
        0.06,
        "B1: 2.3–2.4× less work;\nB2 curved metal in TE: drops to 1st order",
        ha="center",
        va="center",
        fontsize=11.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.26", fc=LORANGE, ec=ORANGE, lw=1.2),
    )

    # ---- RIGHT PANEL: SubTriFIT2D (subtri_circles_mesh)
    ax3 = fig.add_axes([0.68, 0.20, 0.30, 0.67])
    bare(ax3)

    # Build real subtri_circles_mesh around r = 1.0e-6 (user_manual.md lines 2061, 2075-2085)
    st_mesh, st_P, st_info = subtri_circles_mesh(
        [(0.0, 0.0, 1.0e-6)], h=0.09e-6, levels=2, half_width=1.8e-6, pml_cells=0
    )
    st_solver = SubTriFIT2D(st_mesh, st_P, materials={0: [4.0, 1.0]}, polarization="TE")

    assert (
        st_info["n_triangles"] > 0
        and sum(st_info["n_squares"]) > 0
        and st_info["n_hanging"] > 0
    )
    assert np.allclose(np.asarray(st_P.sum(axis=1)).ravel(), 1.0, rtol=1e-12)
    K_st = st_solver.G.T @ sp.diags(st_solver.stiff) @ st_solver.G
    assert np.allclose((K_st - K_st.T).data, 0.0, atol=1e-12)
    assert np.all(st_solver.mass > 0)

    st_nodes_um = st_mesh.nodes * 1e6
    for ci in range(st_mesh.n_cells):
        c_idx = st_mesh.cell(ci)
        poly_xy = st_nodes_um[c_idx]
        cen = poly_xy.mean(axis=0)
        if cen[0] < -0.10 or cen[1] < -0.10:
            continue
        is_tri = len(c_idx) == 3
        inside = st_mesh.cell_tags[ci] == 0
        if is_tri:
            fc_st = LBLUE if inside else "#eef5fc"
            ec_st, lw_st, z_st = BLUE, 1.45, 3
        else:
            fc_st = LBLUE if inside else "white"
            ec_st, lw_st, z_st = BLUE, 1.25, 2
        ax3.add_patch(
            Polygon(poly_xy, closed=True, fc=fc_st, ec=ec_st, lw=lw_st, zorder=z_st)
        )

    # Plot hanging nodes from st_P (orange diamonds)
    hanging_mask = np.asarray(abs(st_P).sum(axis=0)).ravel() == 0
    h_pts = st_nodes_um[hanging_mask]
    in_view = (
        (h_pts[:, 0] >= 0.0)
        & (h_pts[:, 0] <= 1.65)
        & (h_pts[:, 1] >= 0.0)
        & (h_pts[:, 1] <= 1.65)
    )
    ax3.plot(
        h_pts[in_view, 0],
        h_pts[in_view, 1],
        "D",
        color=ORANGE,
        ms=5.2,
        mec="white",
        mew=0.6,
        zorder=6,
    )

    ax3.plot(1.0 * np.cos(th_q), 1.0 * np.sin(th_q), color=GREY, lw=3.0, zorder=5)
    ax3.text(
        0.82,
        1.48,
        r"$K = G P,\quad D = P^{T} m$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=ORANGE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc=LORANGE, ec=ORANGE, lw=1.2),
        zorder=7,
    )
    ax3.text(
        0.34,
        0.24,
        r"finest squares ($h$)",
        ha="center",
        va="center",
        fontsize=10.5,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=BLUE, lw=0.9, alpha=0.9),
        zorder=7,
    )
    ax3.text(
        1.26,
        1.18,
        r"coarse ($2^L h$)",
        ha="center",
        va="center",
        fontsize=10.5,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=BLUE, lw=0.9, alpha=0.9),
        zorder=7,
    )

    ax3.set_xlim(0.0, 1.65)
    ax3.set_ylim(0.0, 1.65)
    ax3.set_title(
        "SubTriFIT2D\n(subtri_circles_mesh)", fontsize=13.5, weight="bold", pad=8
    )
    fig.text(
        0.83,
        0.06,
        "body-fitted collar + quadtree bulk:\n"
        + r"5–7× less work on B1, drift $\leq 2\times 10^{-15}$",
        ha="center",
        va="center",
        fontsize=11.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.26", fc=LGREEN, ec=GREEN, lw=1.2),
    )

    fig.savefig(out / "fig11_2_subgrid_and_subtri.png", facecolor="white")
    plt.close(fig)


# -------------------------------------------------------------------------- 11.3
def fig11_3_curvilinear_ogrid_nfit(outroot):
    """Draw Figure 11.3: 2D Curvilinear O-Grid Non-Orthogonal FIT (CurvilinearFIT2D)."""
    out = outroot / "section11"
    out.mkdir(parents=True, exist_ok=True)

    # Exact manual parameters and calls (user_manual.md lines 2107-2109)
    a, r0, r1 = 1.0e-6, 2.0e-6, 2.6e-6
    radii = radial_nodes([0.9 * a, a, r0, r1], lambda r: 0.04e-6 if r <= a else 0.08e-6)
    mesh = ogrid_mesh(radii, 128, interfaces=[a])

    # Assertions on radial_nodes and ogrid_mesh
    for brk in (0.9 * a, a, r0, r1):
        assert np.any(np.isclose(radii, brk, rtol=1e-12))
    dr_in = np.diff(radii[(radii >= 0.9 * a - 1e-15) & (radii <= a + 1e-15)])
    dr_out = np.diff(radii[(radii >= a - 1e-15) & (radii <= r0 + 1e-15)])
    assert np.allclose(dr_in, 0.04e-6, rtol=0.15)
    assert np.allclose(dr_out[1:], 0.08e-6, rtol=1e-12)
    assert mesh.n_cells > 0 and int(np.sum(mesh.cell_tags == 1)) > 0

    # Multi-block O-grid/Cartesian mesh for inset and non-orthogonal Hodge assertion
    cmesh = ogrid_cartesian_mesh(
        np.linspace(0.55e-6, 1.0e-6, 4),
        24,
        1.35e-6,
        np.array([1.65e-6, 1.95e-6]),
        2,
        interfaces=[1.0e-6],
        smooth=4,
    )
    hc = hodge_components(cmesh)
    W_nfit = (hc["R"].T @ sp.diags(hc["w"]) @ hc["R"]).tocsr()
    M_diag = np.asarray(hc["node_share"].sum(axis=1)).ravel()
    assert np.allclose((W_nfit - W_nfit.T).data, 0.0, atol=1e-15)
    assert (W_nfit - sp.diags(W_nfit.diagonal())).nnz > 0
    assert np.all(M_diag > 0) and np.isclose(
        M_diag.sum(), cmesh.cell_area.sum(), rtol=1e-12
    )

    fig = plt.figure(figsize=(W, H))

    # ---- LEFT PANEL (56% width): 2D polar O-grid sector + multi-block inset
    ax1 = fig.add_axes([0.02, 0.05, 0.53, 0.90])
    bare(ax1)

    th_sec = np.linspace(0.0, 0.5 * np.pi, 240)
    a_um, r0_um, r1_um = a * 1e6, r0 * 1e6, r1 * 1e6

    # Shade material interior r <= a in light blue
    poly_in = np.vstack(
        [[0.0, 0.0], np.column_stack([a_um * np.cos(th_sec), a_um * np.sin(th_sec)])]
    )
    ax1.add_patch(Polygon(poly_in, closed=True, fc=LBLUE, ec="none", zorder=1))

    # Shade radial PML annulus r in [r0, r1] in light green
    poly_pml = np.vstack(
        [
            np.column_stack([r0_um * np.cos(th_sec), r0_um * np.sin(th_sec)]),
            np.column_stack(
                [r1_um * np.cos(th_sec[::-1]), r1_um * np.sin(th_sec[::-1])]
            ),
        ]
    )
    ax1.add_patch(Polygon(poly_pml, closed=True, fc=LGREEN, ec=GREEN, lw=1.6, zorder=1))

    # Draw exact radial rings from `radii` (plus extended inner rings down to r = 0.36 um for sector view)
    radii_inner_full = radial_nodes([0.36 * a, 0.9 * a], lambda r: 0.04e-6)
    all_rings_um = np.unique(np.concatenate([radii_inner_full, radii])) * 1e6
    for rr in all_rings_um:
        col_r = GREEN if rr > r0_um + 1e-6 else BLUE
        lw_r = 1.0 if rr <= a_um + 1e-6 else 1.15
        ax1.plot(
            rr * np.cos(th_sec),
            rr * np.sin(th_sec),
            color=col_r,
            lw=lw_r,
            alpha=0.80,
            zorder=2,
        )

    # Draw azimuthal spokes (every 2nd of the 32 spokes in the 90 deg quadrant for crispness, plus all 128 on outer rim)
    n_quad = 128 // 4
    for j_th in range(0, n_quad + 1, 2):
        th_j = j_th * (0.5 * np.pi) / n_quad
        ax1.plot(
            [0.36 * np.cos(th_j), r1_um * np.cos(th_j)],
            [0.36 * np.sin(th_j), r1_um * np.sin(th_j)],
            color=BLUE,
            lw=0.9,
            alpha=0.65,
            zorder=2,
        )
    for j_th in range(1, n_quad, 2):
        th_j = j_th * (0.5 * np.pi) / n_quad
        ax1.plot(
            [0.82 * np.cos(th_j), 1.16 * np.cos(th_j)],
            [0.82 * np.sin(th_j), 1.16 * np.sin(th_j)],
            color=BLUE,
            lw=0.75,
            alpha=0.55,
            zorder=2,
        )

    # Highlight break circles a = 1.0 um, r0 = 2.0 um, r1 = 2.6 um
    ax1.plot(a_um * np.cos(th_sec), a_um * np.sin(th_sec), color=BLUE, lw=3.2, zorder=4)
    ax1.plot(
        r0_um * np.cos(th_sec),
        r0_um * np.sin(th_sec),
        color=GREEN,
        ls="--",
        lw=2.2,
        zorder=4,
    )
    ax1.plot(
        r1_um * np.cos(th_sec), r1_um * np.sin(th_sec), color=GREEN, lw=2.4, zorder=4
    )

    # Labels on the O-grid sector
    ax1.text(
        0.48,
        0.52,
        "0.04e-6 (40 nm) inside\nmaterials={1: [4.0, 1.0]}",
        ha="center",
        va="center",
        fontsize=11.0,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=BLUE, lw=1.1, alpha=0.94),
        zorder=6,
    )
    ax1.text(
        0.86,
        0.86,
        r"$a = 1.0\ \mu\mathrm{m}$ (128 azimuthal cells)",
        ha="center",
        va="center",
        fontsize=11.5,
        color="white",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc=BLUE, ec="none"),
        zorder=7,
    )
    ax1.text(
        1.12,
        1.14,
        "0.08e-6 (80 nm) outside",
        ha="center",
        va="center",
        fontsize=11.5,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=BLUE, lw=1.1, alpha=0.94),
        zorder=6,
    )
    ax1.text(
        0.78,
        2.18,
        'pml = dict(kind="radial",\n'
        + r"$r_0 = 2.0\ \mu\mathrm{m},\ r_1 = 2.6\ \mu\mathrm{m}$)",
        ha="center",
        va="center",
        fontsize=11.0,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc=LGREEN, ec=GREEN, lw=1.3),
        zorder=7,
    )

    # Inset in upper-right corner showing multi-block ogrid_cartesian_mesh / boxed_cylinders_mesh topology
    ax_in = fig.add_axes([0.385, 0.58, 0.165, 0.30])
    bare(ax_in)
    cnodes_um = cmesh.nodes * 1e6
    for ci_c, c_nodes in enumerate(cmesh.cells):
        poly_c = cnodes_um[c_nodes]
        fc_in = LBLUE if cmesh.cell_tags[ci_c] == 1 else "white"
        ax_in.add_patch(
            Polygon(poly_c, closed=True, fc=fc_in, ec=BLUE, lw=0.85, zorder=2)
        )
    th_full = np.linspace(0, 2 * np.pi, 200)
    ax_in.plot(np.cos(th_full), np.sin(th_full), color=BLUE, lw=1.8, zorder=4)
    ax_in.set_xlim(-2.05, 2.05)
    ax_in.set_ylim(-2.05, 2.05)
    ax_in.text(
        0.5,
        -0.06,
        "ogrid_cartesian_mesh /\nboxed_cylinders_mesh",
        transform=ax_in.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.20", fc=LGREY, ec=GREY, lw=1.0),
    )

    ax1.set_xlim(-0.08, 2.72)
    ax1.set_ylim(-0.08, 2.75)
    ax1.set_title(
        "Body-fitted polar O-grid (ogrid_mesh, radial_nodes)",
        fontsize=14.0,
        weight="bold",
        pad=8,
    )

    # ---- RIGHT PANEL (44% width): Skewed non-orthogonal cell + Hodge coupling + badges
    ax2 = fig.add_axes([0.57, 0.30, 0.41, 0.62])
    bare(ax2)

    # Skewed quadrilateral primary cell vertices (counter-clockwise)
    q_prim = np.array([[-1.15, -0.75], [0.85, -0.65], [1.35, 0.80], [-0.55, 0.72]])
    ax2.add_patch(Polygon(q_prim, closed=True, fc=LBLUE, ec=BLUE, lw=2.6, zorder=2))
    for vx, vy in q_prim:
        ax2.plot(vx, vy, "o", color=BLUE, ms=8.5, mec="white", mew=1.2, zorder=5)

    # Staggered dual cell (orange dashed polygon around cell and across its 4 edges)
    c_center = q_prim.mean(axis=0)
    edge_mids = np.array([0.5 * (q_prim[k] + q_prim[(k + 1) % 4]) for k in range(4)])
    # Dual points of the 4 neighbouring skewed cells
    nb_dual = np.array([[0.05, -1.45], [2.05, 0.15], [0.25, 1.50], [-1.75, -0.10]])
    for k in range(4):
        ax2.plot(
            [c_center[0], edge_mids[k, 0], nb_dual[k, 0]],
            [c_center[1], edge_mids[k, 1], nb_dual[k, 1]],
            color=ORANGE,
            ls="--",
            lw=2.2,
            zorder=3,
        )
        ax2.plot(nb_dual[k, 0], nb_dual[k, 1], "o", color=ORANGE, ms=6.5, zorder=5)
    ax2.plot(c_center[0], c_center[1], "o", color=ORANGE, ms=8.0, zorder=6)

    # Highlight primary edge vector (right edge: vertex 1 -> vertex 2) and dual edge vector meeting at non-orthogonal angle
    p_cross = edge_mids[1]
    prim_dir = (q_prim[2] - q_prim[1]) / np.linalg.norm(q_prim[2] - q_prim[1])
    dual_dir = (nb_dual[1] - c_center) / np.linalg.norm(nb_dual[1] - c_center)
    cos_ang = float(np.dot(prim_dir, dual_dir))
    # Assert visibly non-orthogonal angle (not 90 deg)
    assert abs(cos_ang) > 0.20

    arrow(ax2, q_prim[1], q_prim[2], color=BLUE, lw=3.6, ms=22, z=6)
    arrow(ax2, c_center, nb_dual[1], color=ORANGE, lw=3.0, ms=20, z=6)
    ax2.text(
        q_prim[2, 0] + 0.08,
        q_prim[2, 1] + 0.05,
        "primary edge",
        ha="left",
        va="bottom",
        fontsize=12.0,
        color=BLUE,
        weight="bold",
    )
    ax2.text(
        nb_dual[1, 0] + 0.05,
        nb_dual[1, 1] - 0.12,
        "dual edge",
        ha="left",
        va="top",
        fontsize=12.0,
        color=ORANGE,
        weight="bold",
    )

    # Orange non-orthogonal angle arc at p_cross
    th_p = np.arctan2(prim_dir[1], prim_dir[0])
    th_d = np.arctan2(dual_dir[1], dual_dir[0])
    arc_th = np.linspace(th_d, th_p, 40)
    r_a = 0.34
    ax2.plot(
        p_cross[0] + r_a * np.cos(arc_th),
        p_cross[1] + r_a * np.sin(arc_th),
        color=ORANGE,
        lw=2.2,
        zorder=7,
    )
    ax2.text(
        p_cross[0] + 0.36,
        p_cross[1] + 0.28,
        r"$\theta \neq 90^\circ$",
        ha="left",
        va="center",
        fontsize=12.5,
        color=ORANGE,
        weight="bold",
    )

    # Green corner coupling arrows between adjacent edge fluxes
    for k in range(4):
        m_a = 0.65 * q_prim[k] + 0.35 * edge_mids[(k - 1) % 4]
        m_b = 0.65 * q_prim[k] + 0.35 * edge_mids[k]
        arrow(ax2, m_a, m_b, color=GREEN, lw=2.2, style="<|-|>", rad=0.35, ms=14, z=7)

    ax2.text(
        0.10,
        1.82,
        "symmetric flux-to-voltage Hodge\n(Schuhmann & Weiland 2000)",
        ha="center",
        va="center",
        fontsize=12.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.26", fc=LGREEN, ec=GREEN, lw=1.4),
        zorder=8,
    )

    ax2.set_xlim(-1.95, 2.75)
    ax2.set_ylim(-1.65, 2.15)
    ax2.set_title(
        "Non-orthogonal curvilinear cell (CurvilinearFIT2D)",
        fontsize=14.0,
        weight="bold",
        pad=8,
    )

    # Three badges below the right panel (user_manual.md lines 2097 and 2122)
    fig.text(
        0.775,
        0.24,
        r"diagonal node mass $M$ $\Rightarrow$ explicit leapfrog, zero matrix inversions",
        ha="center",
        va="center",
        fontsize=11.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.26", fc=LGREEN, ec=GREEN, lw=1.3),
    )
    fig.text(
        0.775,
        0.155,
        r"energy drift $7\times 10^{-16}$, radial PML reflection $5\times 10^{-5}$",
        ha="center",
        va="center",
        fontsize=11.5,
        color="white",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.26", fc=GREY, ec="none"),
    )
    fig.text(
        0.775,
        0.06,
        "B1 at 1% RMS: 158k (TM) / 176k (TE) unknowns\n(~4× faster than graded conformal Cart.)",
        ha="center",
        va="center",
        fontsize=11.5,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.26", fc=LBLUE, ec=BLUE, lw=1.3),
    )

    fig.savefig(out / "fig11_3_curvilinear_ogrid_nfit.png", facecolor="white")
    plt.close(fig)


# -------------------------------------------------------------------------- 11.6
def fig11_6_high_order_dgtd_fdtd24(outroot):
    """Draw Figure 11.6: High-order solvers (DGTD2D and FDTD24)."""
    out = outroot / "section11"
    out.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H))

    # ---- LEFT PANEL: Curved vs straight high-order triangles (DGTD2D)
    ax1 = fig.add_axes([0.015, 0.14, 0.31, 0.75])
    bare(ax1)

    R_cyl = 1.0e-6  # r = 1.0e-6 (1 um, user_manual.md line 2231)
    # Upper triangle (straight-sided p = 4) on arc [16 deg, 58 deg]
    th_u0, th_u1 = np.deg2rad(16.0), np.deg2rad(58.0)
    v_u0 = np.array([R_cyl * np.cos(th_u0), R_cyl * np.sin(th_u0)])
    v_u1 = np.array([R_cyl * np.cos(th_u1), R_cyl * np.sin(th_u1)])
    v_u2 = np.array(
        [
            0.42e-6 * np.cos(0.5 * (th_u0 + th_u1)),
            0.42e-6 * np.sin(0.5 * (th_u0 + th_u1)),
        ]
    )

    # Lower two adjacent curved triangles (p = 5) on arcs [-56 deg, -20 deg] and [-20 deg, 16 deg]
    th_l0, th_l1, th_l2 = np.deg2rad(-56.0), np.deg2rad(-20.0), np.deg2rad(16.0)
    v_l0 = np.array([R_cyl * np.cos(th_l0), R_cyl * np.sin(th_l0)])
    v_l1 = np.array([R_cyl * np.cos(th_l1), R_cyl * np.sin(th_l1)])
    v_l2 = v_u0.copy()
    v_in = np.array([0.38e-6 * np.cos(th_l1), 0.38e-6 * np.sin(th_l1)])

    tmesh_straight = TriMesh(
        np.array([v_u0, v_u1, v_u2]),
        np.array([[0, 1, 2]]),
        np.array([1]),
        circles=[(0.0, 0.0, R_cyl)],
    )
    dg_straight = DGTD2D(
        tmesh_straight, p=4, materials={1: [4.0, 1.0]}, curved=False, dt=1e-18
    )

    tmesh_curved = TriMesh(
        np.array([v_l0, v_l1, v_l2, v_in]),
        np.array([[0, 1, 3], [1, 2, 3]]),
        np.array([1, 1]),
        circles=[(0.0, 0.0, R_cyl)],
    )
    dg_curved = DGTD2D(
        tmesh_curved, p=5, materials={1: [4.0, 1.0]}, curved=True, dt=1e-18
    )

    # Assertions on reference triangles and curved vs straight boundary nodes
    T4, T5 = reference_triangle(4), reference_triangle(5)
    assert T4.Np == 15 and T5.Np == 21
    f0_nodes = T5.Fmask[
        0
    ]  # face 0 lies on the circle r = 1 um for both curved triangles
    r_curved_face = np.hypot(dg_curved.x[f0_nodes, :], dg_curved.y[f0_nodes, :])
    assert np.allclose(r_curved_face, R_cyl, rtol=1e-12)
    r_straight_mid = float(
        np.hypot(dg_straight.x[T4.Fmask[0][2], 0], dg_straight.y[T4.Fmask[0][2], 0])
    )
    assert np.isclose(r_straight_mid, R_cyl * np.cos(0.5 * (th_u1 - th_u0)), rtol=1e-12)
    assert (R_cyl - r_straight_mid) > 0.05 * R_cyl

    # Plot exact circle arc r = 1 um in dark grey
    th_arc = np.linspace(np.deg2rad(-66.0), np.deg2rad(66.0), 300)
    ax1.plot(np.cos(th_arc), np.sin(th_arc), color=GREY, lw=3.0, zorder=3)
    ax1.text(
        0.58,
        0.92,
        r"$r = 1\ \mu\mathrm{m}$",
        ha="center",
        va="center",
        fontsize=12.0,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=GREY, lw=1.0),
        zorder=7,
    )

    # Upper straight-sided p = 4 triangle + red chord-to-arc geometric error lens
    tri_u_um = np.array([v_u0, v_u1, v_u2]) * 1e6
    ax1.add_patch(Polygon(tri_u_um, closed=True, fc="white", ec=BLUE, lw=2.0, zorder=2))
    th_lens = np.linspace(th_u0, th_u1, 80)
    lens_poly = np.vstack(
        [np.column_stack([np.cos(th_lens), np.sin(th_lens)]), tri_u_um[0:1]]
    )
    ax1.add_patch(Polygon(lens_poly, closed=True, fc=LRED, ec=RED, lw=1.8, zorder=4))
    ax1.plot(
        dg_straight.x[:, 0] * 1e6,
        dg_straight.y[:, 0] * 1e6,
        "o",
        color=GREY,
        ms=4.2,
        alpha=0.75,
        zorder=5,
    )
    ax1.annotate(
        "straight-sided p = 4:\npolygonal chord stalls\nat 1–5% error",
        xy=(0.94 * np.cos(0.5 * (th_u0 + th_u1)), 0.94 * np.sin(0.5 * (th_u0 + th_u1))),
        xytext=(0.12, 0.56),
        ha="center",
        va="center",
        fontsize=11.0,
        color=RED,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc=LRED, ec=RED, lw=1.2),
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6, shrinkA=0, shrinkB=4),
        zorder=8,
    )

    # Lower two curved p = 5 triangles following the exact arc r = 1 um
    for k_el, (t_a, t_b) in enumerate(((th_l0, th_l1), (th_l1, th_l2))):
        th_c = np.linspace(t_a, t_b, 60)
        v_in_um = v_in * 1e6
        curved_poly = np.vstack(
            [np.column_stack([np.cos(th_c), np.sin(th_c)]), v_in_um[None, :]]
        )
        ax1.add_patch(
            Polygon(
                curved_poly,
                closed=True,
                fc=LBLUE if k_el == 0 else "#e8f1fa",
                ec=BLUE,
                lw=2.2,
                zorder=2,
            )
        )
        ax1.plot(
            dg_curved.x[:, k_el] * 1e6,
            dg_curved.y[:, k_el] * 1e6,
            "o",
            color=BLUE,
            ms=4.8,
            zorder=5,
        )

    # Orange double arrows across the shared elemental face (v_in <-> v_l1)
    p_shared = 0.52 * (v_in + v_l1) * 1e6
    dir_shared = (v_l1 - v_in) / np.linalg.norm(v_l1 - v_in)
    n_shared = np.array([-dir_shared[1], dir_shared[0]])
    arrow(
        ax1,
        p_shared - 0.11 * n_shared,
        p_shared + 0.11 * n_shared,
        color=ORANGE,
        lw=2.4,
        style="<|-|>",
        ms=16,
        z=7,
    )
    ax1.text(
        0.54,
        -0.78,
        "circles_mesh, p = 5",
        ha="center",
        va="center",
        fontsize=11.5,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.20", fc=LBLUE, ec=BLUE, lw=1.2),
        zorder=7,
    )
    ax1.annotate(
        "upwind Riemann flux\n+ LSRK(5,4)",
        xy=p_shared,
        xytext=(0.15, -0.45),
        ha="center",
        va="center",
        fontsize=11.0,
        color=ORANGE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc=LORANGE, ec=ORANGE, lw=1.2),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6, shrinkA=0, shrinkB=4),
        zorder=8,
    )

    ax1.set_xlim(-0.28, 1.15)
    ax1.set_ylim(-1.05, 1.08)
    ax1.set_title(
        "Curved vs. straight high-order\ntriangles (DGTD2D)",
        fontsize=13.5,
        weight="bold",
        pad=8,
    )

    # ---- CENTRE PANEL: Cartesian (2,4) staggered stencil (FDTD24)
    ax2 = fig.add_axes([0.345, 0.18, 0.27, 0.68])
    ax2.axis("off")

    # Instantiate real FDTD24 solver (user_manual.md lines 2241-2243) and assert 4th-order stencil identities
    fd24 = FDTD24(
        h=0.08e-6,
        half_width=0.32e-6,
        n_pml=2,
        cylinders=[],
        polarization="TM",
        order=4,
        time_scheme="lsrk4",
    )
    c1, c2 = fd24.c1, fd24.c2
    assert np.isclose(c1, 9.0 / 8.0, rtol=1e-15) and np.isclose(
        c2, -1.0 / 24.0, rtol=1e-15
    )
    # 1st-order consistency and 3rd-order cancellation on staggered half-points +/- 1/2, +/- 3/2
    assert np.isclose(c1 * 1.0 + c2 * 3.0, 1.0, rtol=1e-15)
    assert np.isclose(c1 * (0.5**3 * 2.0) + c2 * (1.5**3 * 2.0), 0.0, atol=1e-15)

    # Staggered Cartesian grid line along x
    ax2.plot([-2.15, 2.15], [0.0, 0.0], color=GREY, lw=2.2, zorder=2)
    for x_int in (-2.0, -1.0, 1.0, 2.0):
        ax2.plot([x_int, x_int], [-0.08, 0.08], color=GREY, lw=1.4, zorder=2)

    # Central blue electric node at x = 0
    ax2.plot(0.0, 0.0, "o", color=BLUE, ms=15.0, mec="white", mew=1.8, zorder=6)
    ax2.text(
        0.0,
        -0.22,
        "central electric node",
        ha="center",
        va="top",
        fontsize=12.0,
        color=BLUE,
        weight="bold",
    )

    # Four bracketing orange magnetic nodes at x = +/- 0.5, +/- 1.5
    for xh in (-1.5, -0.5, 0.5, 1.5):
        ax2.plot(xh, 0.0, "s", color=ORANGE, ms=12.0, mec="white", mew=1.5, zorder=6)

    # Inner stencil arcs (thick blue)
    arrow(ax2, (-0.5, 0.06), (-0.05, 0.06), color=BLUE, lw=3.6, rad=-0.48, ms=18, z=5)
    arrow(ax2, (0.5, 0.06), (0.05, 0.06), color=BLUE, lw=3.6, rad=0.48, ms=18, z=5)
    ax2.text(
        0.0,
        0.38,
        "inner neighbours",
        ha="center",
        va="center",
        fontsize=12.0,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=BLUE, lw=1.0),
        zorder=7,
    )

    # Outer stencil arcs (thin orange)
    arrow(ax2, (-1.5, 0.06), (-0.08, 0.10), color=ORANGE, lw=2.0, rad=-0.65, ms=16, z=4)
    arrow(ax2, (1.5, 0.06), (0.08, 0.10), color=ORANGE, lw=2.0, rad=0.65, ms=16, z=4)
    ax2.text(
        0.0,
        0.92,
        "outer neighbours",
        ha="center",
        va="center",
        fontsize=12.0,
        color=ORANGE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=ORANGE, lw=1.0),
        zorder=7,
    )

    # Badge required by spec
    ax2.text(
        0.0,
        -0.88,
        'FDTD24: fourth-order staggered\nspatial differences\n(order=4, time_scheme="lsrk4")',
        ha="center",
        va="center",
        fontsize=12.0,
        color=BLUE,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.32", fc=LBLUE, ec=BLUE, lw=1.4),
    )

    ax2.set_xlim(-2.25, 2.25)
    ax2.set_ylim(-1.45, 1.35)
    ax2.set_title(
        "Cartesian (2,4) staggered\nstencil (FDTD24)",
        fontsize=13.5,
        weight="bold",
        pad=8,
    )

    # ---- RIGHT PANEL: Unknowns to reach 0.1% error (manual Sec. 11.9, lines 2247-2248)
    ax3 = fig.add_axes([0.65, 0.18, 0.33, 0.68])

    # Exact quoted numbers from user_manual.md lines 2247-2248
    u_b1_tri = 2.1e6  # ~2.1 M unknowns (~2600 s), line 2247
    u_b1_dg5 = 17.0e3  # 17k unknowns (91 s, ~100x fewer, ~28x faster), line 2247
    u_b1_001 = 25.0e3  # 25k unknowns (reaches 0.01% RMS), line 2247
    u_b2_tri = 88.0e3  # 88k unknowns, line 2248
    u_b2_dg2 = 6.0e3  # 6.0k unknowns (14.7x fewer), line 2248

    assert np.isclose(u_b2_tri / u_b2_dg2, 14.7, atol=0.05)
    assert u_b1_tri / u_b1_dg5 > 100.0

    y_pos = np.array([3.3, 2.4, 1.05, 0.2])
    vals = [u_b1_tri, u_b1_dg5, u_b2_tri, u_b2_dg2]
    fcs = [LGREY, LBLUE, LGREY, LBLUE]
    ecs = [GREY, BLUE, GREY, BLUE]
    bar_labels = [
        "Graded triangles + PML:\n~2.1 M unknowns (~2600 s)",
        "Curved DGTD2D p = 5: 17k unknowns\n(91 s, ~100× fewer, ~28× faster)",
        "Uniform triangles:\n88k unknowns",
        "Curved DGTD2D p = 2:\n6.0k unknowns (14.7× fewer)",
    ]

    ax3.barh(y_pos, vals, height=0.60, color=fcs, edgecolor=ecs, lw=1.8, zorder=3)
    for yp, v_bar, lab_b, col_b in zip(y_pos, vals, bar_labels, ecs):
        x_txt = v_bar * 1.18 if v_bar < 5e5 else 2.2e3
        ax3.text(
            x_txt,
            yp,
            lab_b,
            ha="left",
            va="center",
            fontsize=10.5,
            color=col_b if v_bar < 5e5 else "black",
            weight="bold",
            zorder=5,
        )

    # Green marker at 25k unknowns (reaches 0.01% RMS) on B1 (line 2247)
    ax3.plot(u_b1_001, 1.82, "D", color=GREEN, ms=10.0, mec="white", mew=1.4, zorder=6)
    ax3.text(
        u_b1_001 * 1.35,
        1.82,
        "25k unknowns (reaches 0.01% RMS)",
        ha="left",
        va="center",
        fontsize=10.5,
        color=GREEN,
        weight="bold",
        zorder=6,
    )

    # Group headers
    ax3.text(
        1.3e3,
        3.82,
        "B1 Mie cylinder (0.1% TM RMS)",
        fontsize=12.0,
        color="black",
        weight="bold",
    )
    ax3.axhline(1.50, color=LGREY, lw=1.5, zorder=1)
    ax3.text(
        1.3e3,
        1.56,
        r"B2-Au-TE (0.1% $Q_{\mathrm{ext}}$ NRMS)",
        fontsize=12.0,
        color="black",
        weight="bold",
    )

    ax3.set_xscale("log")
    ax3.set_xlim(1.0e3, 3.5e7)
    ax3.set_ylim(-0.35, 4.15)
    ax3.set_yticks([])
    ax3.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7])
    ax3.set_xticklabels(
        [r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$"], fontsize=11.5
    )
    ax3.set_xlabel("unknowns (log scale)", fontsize=13.5)
    ax3.set_title(
        "Unknowns to reach 0.1% error\n(manual Sec. 11.9)",
        fontsize=13.5,
        weight="bold",
        pad=8,
    )
    for sp_name in ("top", "right", "left"):
        ax3.spines[sp_name].set_visible(False)

    fig.savefig(out / "fig11_6_high_order_dgtd_fdtd24.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    outroot = Path(sys.argv[1])
    outroot.mkdir(parents=True, exist_ok=True)
    fig9_1_bloch_periodic(outroot)
    fig10_5_harminv_resonance(outroot)
    fig11_1_triangle_and_polygon_fit(outroot)
    fig11_2_subgrid_and_subtri(outroot)
    fig11_3_curvilinear_ogrid_nfit(outroot)
    fig11_6_high_order_dgtd_fdtd24(outroot)
