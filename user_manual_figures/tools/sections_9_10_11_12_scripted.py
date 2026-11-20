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


if __name__ == "__main__":
    outroot = Path(sys.argv[1])
    outroot.mkdir(parents=True, exist_ok=True)
    fig9_1_bloch_periodic(outroot)
