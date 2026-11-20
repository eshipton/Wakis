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

from wakis.bloch import incidence_angle, k_parallel  # noqa: E402
from wakis.resonances import harminv  # noqa: E402

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


if __name__ == "__main__":
    outroot = Path(sys.argv[1])
    outroot.mkdir(parents=True, exist_ok=True)
    fig9_1_bloch_periodic(outroot)
    fig10_5_harminv_resonance(outroot)
