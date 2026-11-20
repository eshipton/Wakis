"""Deterministic scripted illustrations for user_manual.md Sections 13, 14, 15, and 16.

Draws every figure with ``"route": "scripted"`` in
``user_manual_figures/section{N}/specs/section{N}.json`` for N in 13, 14, 15, 16:
- section13/fig13_4_subpixel_smoothed_projection.png (13.4)
- section14/fig14_2_tete_dispersion.png (14.2)
- section14/fig14_3_cold_plasma_wake.png (14.3)
- section15/fig15_2_cost_to_target.png (15.2)
- section16/fig16_2_cell_memory_breakdown.png (16.2)
- section16/fig16_3_roofline_fused_gpu.png (16.3)

Usage: python sections_13_14_15_16_scripted.py OUTROOT
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wakis.inverse import projection as ssp  # noqa: E402

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


def arrow(ax, p, q, color=GREY, lw=2.0, style="-|>", rad=0.0, ms=18, z=3, ls="-"):
    """Draw a FancyArrowPatch between 2D points p and q."""
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


def box(ax, xy, w, h, text, fc, ec, fs=15, tc="black", lw=1.6, pad=0.02, rsize=0.06, z=2):
    """Draw a rounded rectangle centred at xy with centred label text."""
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
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 2)


def bare(ax):
    """Set equal aspect and hide axes."""
    ax.set_aspect("equal")
    ax.axis("off")


def clean_axes(ax):
    """Hide top and right spines for textbook plot style."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- 13.4
def fig13_4_subpixel_smoothed_projection(out_root):
    """Figure 13.4: Subpixel-Smoothed Projection (SSP) for high-beta topology optimization.

    Source lines in user_manual.md: 2422, 2450, 2451, 2452.
    """
    out_dir = out_root / "section13"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H))

    # Divider lines between the three panels
    fig.add_artist(plt.Line2D([0.33, 0.33], [0.06, 0.94], transform=fig.transFigure, color=LGREY, lw=1.5))
    fig.add_artist(plt.Line2D([0.635, 0.635], [0.06, 0.94], transform=fig.transFigure, color=LGREY, lw=1.5))

    # ---- PANEL 1 (left, 32% width): Standard tanh_projection ----------------
    fig.text(0.165, 0.93, "Standard tanh_projection", ha="center", fontsize=17, weight="bold")

    # Formula from wakis.inverse.projection.tanh_projection (user_manual.md lines 2422, 2451)
    rho_filt = np.linspace(0.0, 1.0, 501)
    eta = 0.5  # user_manual.md line 2422
    beta_fin = 8.0  # user_manual.md line 2422
    proj_8 = ssp.tanh_projection(rho_filt, beta_fin, eta=eta)
    proj_inf = ssp.tanh_projection(rho_filt, np.inf, eta=eta)
    dproj_8 = ssp.tanh_projection_derivative(rho_filt, beta_fin, eta=eta)
    dproj_inf = ssp.tanh_projection_derivative(rho_filt, np.inf, eta=eta)

    # Mathematical assertions for Panel 1
    assert np.isclose(ssp.tanh_projection(eta, beta_fin, eta=eta), 0.5, atol=1e-15)
    assert np.all(dproj_inf[rho_filt != eta] == 0.0)
    assert np.isclose(np.trapezoid(dproj_8, rho_filt), 1.0, rtol=1e-4)

    ax1_top = fig.add_axes([0.055, 0.49, 0.25, 0.37])
    clean_axes(ax1_top)
    ax1_top.plot(rho_filt, proj_8, color=BLUE, lw=2.5, label=r"$\beta = 8.0$ ($\eta = 0.5$)")
    # Plot the two branches of the beta -> inf step limit so the jump at eta is explicit
    left_mask = rho_filt < eta
    right_mask = rho_filt > eta
    ax1_top.plot(rho_filt[left_mask], proj_inf[left_mask], color=RED, lw=2.5, label=r"$\beta \to \infty$ step limit")
    ax1_top.plot(rho_filt[right_mask], proj_inf[right_mask], color=RED, lw=2.5)
    ax1_top.plot([eta, eta], [0.0, 1.0], color=RED, lw=1.6, ls=":")
    ax1_top.axvline(eta, color=GREY, lw=1.0, ls="--", alpha=0.5)
    ax1_top.set_xlim(0.0, 1.0)
    ax1_top.set_ylim(-0.06, 1.28)
    ax1_top.set_xticks([0.0, 0.5, 1.0])
    ax1_top.set_xticklabels(["", "", ""])
    ax1_top.set_yticks([0.0, 0.5, 1.0])
    ax1_top.set_ylabel("tanh_projection(rho_filt, beta, eta)", fontsize=12)
    ax1_top.legend(loc="upper left", frameon=False, fontsize=11.5)

    ax1_bot = fig.add_axes([0.055, 0.14, 0.25, 0.27])
    clean_axes(ax1_bot)
    ax1_bot.plot(rho_filt, dproj_8, color=BLUE, lw=2.2, label=r"$\partial P / \partial \tilde{\rho}$ ($\beta = 8.0$)")
    ax1_bot.plot(rho_filt, dproj_inf, color=RED, lw=2.6, label=r"$\beta \to \infty$: 0 everywhere ($x \neq \eta$)")
    # Dirac spike indicator at rho_filt = eta
    arrow(ax1_bot, (eta, 0.0), (eta, 4.3), color=RED, lw=2.0, ms=14, ls="-")
    ax1_bot.set_xlim(0.0, 1.0)
    ax1_bot.set_ylim(-0.3, 4.8)
    ax1_bot.set_xticks([0.0, 0.5, 1.0])
    ax1_bot.set_xticklabels(["0", r"$\eta = 0.5$", "1"], fontsize=12.5)
    ax1_bot.set_yticks([0.0, 2.0, 4.0])
    ax1_bot.set_xlabel(r"filtered density $\tilde{\rho} \in [0, 1]$", fontsize=13)
    ax1_bot.set_ylabel(r"derivative $\partial P / \partial \tilde{\rho}$", fontsize=12.5)
    ax1_bot.text(
        0.50,
        2.35,
        "β → ∞: step projection has\nan exactly zero gradient",
        color=RED,
        fontsize=11.5,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=RED, lw=1.0, alpha=0.92),
        zorder=6,
    )

    # ---- PANEL 2 (centre, 32% width): Signed distance d in a cell -----------
    fig.text(0.482, 0.93, "Signed distance d in a cell", ha="center", fontsize=17, weight="bold")
    ax2 = fig.add_axes([0.34, 0.08, 0.285, 0.80])
    bare(ax2)

    # Cell [0, dx] x [0, dx] with dx = 1.0; linear filtered field rho_tilde(x, y)
    dx_cell = 1.0
    xc = np.array([0.5, 0.5])
    # Choose gradient vector g_vec = grad(rho_tilde) and point x_eta on level set rho_tilde = eta
    g_vec = np.array([0.8, 0.6])  # ||g_vec|| = 1.0
    n_hat = g_vec / np.linalg.norm(g_vec)
    t_hat = np.array([-n_hat[1], n_hat[0]])
    d_signed = 0.22  # signed distance d = (eta - rho_tilde(xc)) / ||grad rho_tilde|| (user_manual.md line 2452)
    rho_xc = eta - d_signed * np.linalg.norm(g_vec)
    x_perp = xc + ((eta - rho_xc) / np.linalg.norm(g_vec)) * n_hat

    # Assert exact signed-distance geometry in the cell
    assert np.isclose((eta - rho_xc) / np.linalg.norm(g_vec), d_signed, atol=1e-15)
    assert np.isclose(np.linalg.norm(x_perp - xc), abs(d_signed), atol=1e-15)
    assert np.isclose(rho_xc + g_vec @ (x_perp - xc), eta, atol=1e-15)

    # Clip polygon helper against half-plane {p : (p - p0) . normal <= 0}
    def clip_halfplane(poly, p0, normal):
        out_pts = []
        m = len(poly)
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            da = np.dot(a - p0, normal)
            db = np.dot(b - p0, normal)
            if da <= 0:
                out_pts.append(a)
            if (da <= 0) != (db <= 0):
                t = da / (da - db)
                out_pts.append(a + t * (b - a))
        return np.array(out_pts)

    cell_poly = np.array([[0.0, 0.0], [dx_cell, 0.0], [dx_cell, dx_cell], [0.0, dx_cell]])
    r_band = 0.30
    band_poly = clip_halfplane(cell_poly, x_perp + r_band * n_hat, n_hat)
    band_poly = clip_halfplane(band_poly, x_perp - r_band * n_hat, -n_hat)

    # Shade interface band inside the cell
    ax2.add_patch(Polygon(band_poly, closed=True, fc=LGREEN, ec="none", alpha=0.85, zorder=1))
    # Band boundary dashed lines
    for sign in (-1.0, 1.0):
        seg = clip_halfplane(cell_poly, x_perp + sign * r_band * n_hat, sign * n_hat)
        pts_on_line = [p for p in seg if abs(np.dot(p - (x_perp + sign * r_band * n_hat), n_hat)) < 1e-7]
        if len(pts_on_line) >= 2:
            ax2.plot(
                [pts_on_line[0][0], pts_on_line[1][0]],
                [pts_on_line[0][1], pts_on_line[1][1]],
                color=GREEN,
                lw=1.6,
                ls="--",
                zorder=2,
            )

    # Level set line rho_tilde = eta clipped to cell
    half_solid = clip_halfplane(cell_poly, x_perp, -n_hat)
    lvl_pts = [p for p in half_solid if abs(np.dot(p - x_perp, n_hat)) < 1e-7]
    ax2.plot([lvl_pts[0][0], lvl_pts[1][0]], [lvl_pts[0][1], lvl_pts[1][1]], color=GREY, lw=3.0, zorder=4)

    # Cell outline (blue)
    ax2.add_patch(Rectangle((0.0, 0.0), dx_cell, dx_cell, fc="none", ec=BLUE, lw=2.6, zorder=5))

    # Cell centre and normal vector d to level set
    ax2.add_patch(Circle(xc, 0.025, color=BLUE, zorder=6))
    arrow(ax2, xc, x_perp, color=ORANGE, lw=2.6, ms=18, z=6)
    # Right-angle symbol at x_perp
    ra = 0.045
    ra_pts = [x_perp - ra * n_hat, x_perp - ra * n_hat + ra * t_hat, x_perp + ra * t_hat]
    ax2.plot([p[0] for p in ra_pts], [p[1] for p in ra_pts], color=GREY, lw=1.3, zorder=5)

    # Labels inside/around the cell
    ax2.text(
        0.50,
        0.24,
        r"$d = (\eta - \tilde{\rho})\, /\, \Vert\nabla\tilde{\rho}\Vert$",
        color=ORANGE,
        fontsize=13,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=ORANGE, lw=1.1, alpha=0.95),
        zorder=7,
    )
    ax2.text(
        lvl_pts[0][0] - 0.05,
        lvl_pts[0][1] + 0.06,
        r"$\tilde{\rho} = \eta$",
        color=GREY,
        fontsize=15,
        weight="bold",
        ha="left",
        va="bottom",
        zorder=7,
    )
    ax2.text(
        0.50,
        0.86,
        "smoothed_projection\ninterface band",
        color=GREEN,
        fontsize=13,
        ha="center",
        va="center",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=GREEN, lw=1.1, alpha=0.92),
        zorder=7,
    )

    # Dimension bracket for Delta x below the cell
    by = -0.10
    ax2.annotate(
        "",
        xy=(dx_cell, by),
        xytext=(0.0, by),
        arrowprops=dict(arrowstyle="<|-|>", color=BLUE, lw=1.8, mutation_scale=15),
    )
    ax2.text(0.5 * dx_cell, by - 0.08, r"$\Delta x$", color=BLUE, fontsize=16, ha="center", va="top")
    ax2.set_xlim(-0.18, 1.18)
    ax2.set_ylim(-0.36, 1.16)

    # ---- PANEL 3 (right, 36% width): Level set moving through a cell at beta = inf
    fig.text(0.812, 0.93, "Level set moving through a cell at β = ∞", ha="center", fontsize=15.5, weight="bold")

    # Top sub-panel: step projection vs smoothed_projection cell value as level set crosses cell
    x0_cell = np.linspace(-0.85, 0.85, 501)
    R_def = ssp.RADIUS_FACTOR  # 0.55 (user_manual.md line 2452)
    rho_cell = eta + x0_cell  # unit gradient ||grad rho_tilde|| = 1 / dx, so d = -x0_cell
    step_val = ssp.tanh_projection(rho_cell, np.inf, eta=eta)
    ssp_val, _, _ = ssp.smoothed_projection_pointwise(rho_cell, np.ones_like(rho_cell), np.inf, eta=eta, radius=R_def)

    # Assert exact match between smoothed_projection_pointwise at beta=inf and fill_fraction(-x0/R)
    assert np.allclose(ssp_val, ssp.fill_fraction(-x0_cell / R_def), atol=1e-15)

    ax3_top = fig.add_axes([0.69, 0.53, 0.28, 0.34])
    clean_axes(ax3_top)
    ax3_top.axvspan(-R_def, R_def, color=LGREEN, alpha=0.55, zorder=0)
    m_neg = x0_cell < 0.0
    m_pos = x0_cell > 0.0
    ax3_top.plot(x0_cell[m_neg], step_val[m_neg], color=RED, lw=2.5, label=r"step projection ($\beta = \infty$)")
    ax3_top.plot(x0_cell[m_pos], step_val[m_pos], color=RED, lw=2.5)
    ax3_top.plot([0.0, 0.0], [0.0, 1.0], color=RED, lw=1.6, ls=":")
    ax3_top.plot(x0_cell, ssp_val, color=GREEN, lw=2.8, label="smoothed_projection")
    ax3_top.set_xlim(-0.85, 0.85)
    ax3_top.set_ylim(-0.06, 1.34)
    ax3_top.set_xticks([-0.5, 0.0, 0.5])
    ax3_top.set_xticklabels([r"$-0.5\,\Delta x$", "0", r"$+0.5\,\Delta x$"], fontsize=11.5)
    ax3_top.set_yticks([0.0, 0.5, 1.0])
    ax3_top.set_ylabel("projected cell value", fontsize=12.5)
    ax3_top.legend(loc="upper left", frameon=False, fontsize=10.8)
    ax3_top.text(
        0.82,
        0.16,
        "adjoint matches FD\nto 2×10⁻⁶–10⁻⁴",
        color="white",
        fontsize=11,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28", fc=GREEN, ec="none"),
        zorder=6,
    )

    # Bottom sub-panel: shape-derivative modulation across sub-pixel position
    # Formula from docs/physicsguide.md line 2794 & studies/ssp/ssp_study.py line 107:
    # dM/dx0 = sum_i -1/R * fill_fraction_derivative((x_i - x0) / R)
    samples = np.arange(-4, 6, dtype=float)
    x0_sub = np.linspace(0.0, 1.0, 401)

    def shape_derivative_rate(r_over_dx):
        u = (samples[None, :] - x0_sub[:, None]) / r_over_dx
        return np.sum(-ssp.fill_fraction_derivative(u) / r_over_dx, axis=1)

    rate_055 = shape_derivative_rate(0.55)
    rate_075 = shape_derivative_rate(0.75)
    mod_055 = rate_055.max() / rate_055.min()
    mod_075 = rate_075.max() / rate_075.min()

    # Assert exact modulation values quoted on user_manual.md line 2452 and studies/ssp/README.md line 44-45
    assert round(rate_055.min(), 2) == 0.10 and round(rate_055.max(), 2) == 1.70 and round(mod_055) == 17
    assert round(rate_075.min(), 2) == 0.77 and round(rate_075.max(), 2) == 1.25 and round(mod_075, 1) == 1.6
    assert np.isclose(np.trapezoid(rate_055, x0_sub), 1.0, atol=1e-12)
    assert np.isclose(np.trapezoid(rate_075, x0_sub), 1.0, atol=1e-12)

    ax3_bot = fig.add_axes([0.69, 0.13, 0.28, 0.30])
    clean_axes(ax3_bot)
    ax3_bot.axhline(1.0, color=GREY, lw=1.1, ls="--", zorder=1)
    ax3_bot.plot(x0_sub, rate_055, color=ORANGE, lw=2.5, label="R = 0.55 Δx: 17× modulation")
    ax3_bot.plot(x0_sub, rate_075, color=GREEN, lw=2.6, label="R = 0.75 Δx: 1.6× modulation")
    ax3_bot.set_xlim(0.0, 1.0)
    ax3_bot.set_ylim(0.0, 2.25)
    ax3_bot.set_xticks([0.0, 0.5, 1.0])
    ax3_bot.set_xticklabels(["0", r"$0.5\,\Delta x$", r"$\Delta x$"], fontsize=11.5)
    ax3_bot.set_yticks([0.0, 1.0, 1.7])
    ax3_bot.set_xlabel(r"sub-pixel level-set position $x_0 / \Delta x$", fontsize=12.5)
    ax3_bot.set_ylabel("shape derivative", fontsize=12.5)
    ax3_bot.legend(loc="upper center", frameon=False, fontsize=11)

    fig.savefig(out_dir / "fig13_4_subpixel_smoothed_projection.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    out_root = Path(sys.argv[1])
    out_root.mkdir(parents=True, exist_ok=True)
    fig13_4_subpixel_smoothed_projection(out_root)
