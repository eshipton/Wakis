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

from scipy.constants import c as c_light  # noqa: E402
from scipy.constants import epsilon_0 as eps_0  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402
from scipy.special import i0, i1, k0, k1  # noqa: E402

from wakis import GridFIT3D, SolverFIT3D  # noqa: E402
from wakis import estimate as est_mod  # noqa: E402
from wakis.benchmarks import cost_to_target  # noqa: E402
from wakis.inverse import projection as ssp  # noqa: E402
from wakis.plasma import linear_wake, pipe_green_function, plasma_frequency, plasma_wavelength  # noqa: E402

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


# --------------------------------------------------------------------------- 14.2
def fig14_2_tete_dispersion(out_root):
    """Figure 14.2: Eliminating numerical Cherenkov radiation with scheme='tete'.

    Source lines in user_manual.md: 2498, 2501, 2502, 2504, 2505, 2506, 2523, 2528.
    """
    out_dir = out_root / "section14"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Query the real Wakis SolverFIT3D API with the exact cell sizes of user_manual.md lines 2512-2523
    # (dx = dy = 0.10 / 30 m, dz = 0.40 / 200 m = 0.002 m; 6x6x20 grid keeps identical dx, dy, dz)
    grid = GridFIT3D(-0.01, 0.01, -0.01, 0.01, -0.02, 0.02, 6, 6, 20, verbose=0)
    solver_tete = SolverFIT3D(
        grid, scheme="tete", bc_low=["pec", "pec", "pec"], bc_high=["pec", "pec", "pec"], verbose=0
    )
    solver_yee = SolverFIT3D(
        grid, scheme="yee", cfln=1.0, bc_low=["pec", "pec", "pec"], bc_high=["pec", "pec", "pec"], verbose=0
    )
    dx, dy, dz = float(grid.dx[0]), float(grid.dy[0]), float(grid.dz[0])
    s_tete = c_light * solver_tete.dt / dz  # user_manual.md line 2523: 1.000000
    s_yee = c_light * solver_yee.dt / dz  # user_manual.md line 2498: c dt <= 1/sqrt(dx^-2+dy^-2+dz^-2) < dz

    # Assert magic-step condition and Courant inequality from user_manual.md lines 2498, 2505-2506, 2523
    dt_trans_max = 1.0 / (c_light * np.hypot(1.0 / dx, 1.0 / dy))
    assert dz <= c_light * dt_trans_max
    assert np.isclose(s_tete, 1.0, atol=1e-12), s_tete
    assert s_yee < 1.0 and np.isclose(
        c_light * solver_yee.dt, 1.0 / np.sqrt(dx**-2 + dy**-2 + dz**-2), rtol=1e-3
    )

    # Exact TE/TM vacuum dispersion relation (user_manual.md line 2504):
    # sin^2(omega dt / 2) / (c dt)^2 = sin^2(kz dz / 2) / dz^2
    #   + (sin^2(kx dx / 2)/dx^2 + sin^2(ky dy / 2)/dy^2) * (1 - (c dt / dz)^2 sin^2(kz dz / 2))
    kz_dz = np.linspace(1e-4, np.pi, 500)
    kz = kz_dz / dz

    def tete_omega(kz_arr, kx_val, ky_val, dt_val):
        cdt = c_light * dt_val
        s_z = np.sin(0.5 * kz_arr * dz) / dz
        s_perp2 = (np.sin(0.5 * kx_val * dx) / dx) ** 2 + (np.sin(0.5 * ky_val * dy) / dy) ** 2
        rhs = s_z**2 + s_perp2 * (1.0 - (cdt / dz) ** 2 * np.sin(0.5 * kz_arr * dz) ** 2)
        return (2.0 / dt_val) * np.arcsin(np.clip(cdt * np.sqrt(rhs), 0.0, 1.0))

    # (1) Axial mode (kx = ky = 0) in scheme='tete' at magic step dt = dz / c
    omega_tete_0 = tete_omega(kz, 0.0, 0.0, solver_tete.dt)
    vph_tete_0 = omega_tete_0 / (c_light * kz)

    # (2) Non-zero transverse mode (kx = kz tan(theta), ky = 0) in scheme='tete' at dt = dz / c
    # At kz -> 0, v_ph / c -> 1; at kz dz = pi, the factor (1 - (c dt / dz)^2 sin^2(kz dz / 2)) = 0
    # so omega dt = pi and v_ph / c = cos(theta) exactly.
    tan_theta = 0.42
    kx_obl = kz * tan_theta
    omega_tete_perp = tete_omega(kz, kx_obl, 0.0, solver_tete.dt)
    k_mag = np.hypot(kz, kx_obl)
    vph_tete_perp = omega_tete_perp / (c_light * k_mag)

    # (3) Standard 3D Yee axial mode (kx = ky = 0): omega dt = 2 arcsin((c dt / dz) sin(kz dz / 2)) (line 2498)
    omega_yee_0 = (2.0 / solver_yee.dt) * np.arcsin(s_yee * np.sin(0.5 * kz_dz))
    vph_yee_0 = omega_yee_0 / (c_light * kz)

    # Mathematical assertions for the three curves
    assert np.allclose(vph_tete_0, 1.0, atol=1e-14)
    assert np.all(vph_yee_0 < 1.0) and np.all(np.diff(vph_yee_0) < 0.0)
    assert np.isclose(vph_tete_perp[0], 1.0, atol=1e-6)
    assert np.isclose(vph_tete_perp[-1], 1.0 / np.hypot(1.0, tan_theta), atol=1e-12)
    assert np.all(vph_tete_perp <= 1.0 + 1e-12) and np.all(np.diff(vph_tete_perp) < 0.0)

    fig = plt.figure(figsize=(W, H))
    fig.add_artist(plt.Line2D([0.555, 0.555], [0.06, 0.94], transform=fig.transFigure, color=LGREY, lw=1.5))

    # ---- LEFT PANEL (55% width): Longitudinal phase velocity v_ph(k_z) / c ---
    fig.text(0.285, 0.93, r"Longitudinal phase velocity $v_{\mathrm{ph}}(k_z)\, /\, c$", ha="center", fontsize=17, weight="bold")
    ax1 = fig.add_axes([0.075, 0.14, 0.45, 0.72])
    clean_axes(ax1)

    # Shade Cherenkov gap between Yee curve and v = c
    ax1.fill_between(kz_dz, vph_yee_0, 1.0, color=LRED, alpha=0.65, zorder=1)
    ax1.plot(kz_dz, vph_tete_0, color=BLUE, lw=3.0, zorder=4)
    ax1.plot(kz_dz, vph_tete_perp, color=GREEN, lw=2.4, ls="--", zorder=3)
    ax1.plot(kz_dz, vph_yee_0, color=RED, lw=2.8, zorder=4)

    # Curve annotations
    ax1.text(
        0.06,
        1.012,
        "scheme='tete' at Δt = Δz/c (k_x = k_y = 0):\nv_ph(k_z) = c exact (matches bunch v = c)",
        color=BLUE,
        fontsize=12.0,
        weight="bold",
        va="bottom",
    )
    ax1.text(
        3.08,
        0.965,
        "k_x, k_y ≠ 0: (1 − (cΔt/Δz)² sin²(k_zΔz/2))",
        color=GREEN,
        fontsize=11.5,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GREEN, lw=1.0, alpha=0.92),
        zorder=5,
    )
    ax1.text(
        1.95,
        0.875,
        "v_ph(k_z) < c: numerical Cherenkov radiation\n(1.2–660 V/pC in smooth PEC pipe\nvs ≤ 3×10⁻⁸ V/pC for tete)",
        color=RED,
        fontsize=11.5,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=RED, lw=1.2, alpha=0.95),
        zorder=5,
    )
    ax1.text(
        2.15,
        0.765,
        "scheme='yee'\n(cΔt ≤ 1/√(Δx⁻² + Δy⁻² + Δz⁻²) < Δz)",
        color=RED,
        fontsize=11.5,
        weight="bold",
        ha="right",
        va="center",
    )

    ax1.set_xlim(0.0, np.pi)
    ax1.set_ylim(0.70, 1.068)
    ax1.set_xticks([0.0, np.pi])
    ax1.set_xticklabels(["0", "π"], fontsize=14)
    ax1.set_yticks([1.0])
    ax1.set_yticklabels(["1"], fontsize=14)
    ax1.set_xlabel(r"longitudinal wavenumber $k_z \Delta z$", fontsize=14)
    ax1.set_ylabel(r"normalized phase velocity $v_{\mathrm{ph}}(k_z)\, /\, c$", fontsize=14)

    # ---- RIGHT PANEL (45% width): TE/TM splitting & magic-step condition ----
    fig.text(0.775, 0.93, "TE/TM splitting & magic-step condition", ha="center", fontsize=17, weight="bold")
    ax2 = fig.add_axes([0.57, 0.05, 0.41, 0.84])
    bare(ax2)

    # Top half: horizontal time axis with integer (TM) and half-integer (TE) boxes
    y_axis = 2.05
    ax2.annotate(
        "",
        xy=(4.35, y_axis),
        xytext=(-0.25, y_axis),
        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8, mutation_scale=18),
    )
    ax2.text(4.42, y_axis, r"$t$", fontsize=15, color=GREY, va="center")

    t_nodes = [
        (0.45, "n Δt", "TM"),
        (2.05, "(n+1/2) Δt", "TE"),
        (3.65, "(n+1) Δt", "TM"),
    ]
    for tx, tlab, kind in t_nodes:
        ax2.plot([tx, tx], [y_axis - 0.08, y_axis + 0.08], color=GREY, lw=1.8)
        ax2.text(tx, y_axis - 0.24, tlab, ha="center", va="top", fontsize=13.5, color=GREY, weight="bold")
        if kind == "TM":
            box(
                ax2,
                (tx, y_axis + 0.56),
                1.36,
                0.52,
                "longitudinal-TM:\n(H_x, H_y, E_z)",
                LBLUE,
                BLUE,
                fs=12.0,
                tc=BLUE,
            )
        else:
            box(
                ax2,
                (tx, y_axis + 1.24),
                1.42,
                0.52,
                "longitudinal-TE:\n(E_x, E_y, H_z)",
                LORANGE,
                ORANGE,
                fs=12.0,
                tc=ORANGE,
            )

    # Curved leapfrog / Stormer-Verlet arrows linking TM and TE boxes
    arrow(ax2, (0.95, y_axis + 0.85), (1.42, y_axis + 1.22), color=ORANGE, lw=2.2, rad=-0.25, ms=16)
    arrow(ax2, (2.68, y_axis + 1.22), (3.15, y_axis + 0.85), color=BLUE, lw=2.2, rad=-0.25, ms=16)
    ax2.text(
        2.05,
        y_axis + 1.72,
        "Störmer–Verlet half-kick / kick / half-kick",
        ha="center",
        va="center",
        fontsize=12.5,
        color=GREY,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.24", fc=LGREY, ec=GREY, lw=1.1),
    )

    # Dispersion relation card in middle-right
    box(
        ax2,
        (2.05, 1.12),
        4.35,
        0.72,
        r"$\frac{\sin^2(\omega\Delta t/2)}{(c\Delta t)^2} = \frac{\sin^2(k_z\Delta z/2)}{\Delta z^2}"
        r" + \left(\frac{\sin^2(k_x\Delta x/2)}{\Delta x^2} + \frac{\sin^2(k_y\Delta y/2)}{\Delta y^2}\right)"
        r"\!\left(1 - \frac{(c\Delta t)^2}{\Delta z^2}\sin^2(k_z\Delta z/2)\right)$",
        "white",
        GREY,
        fs=11.0,
        tc="black",
    )

    # Bottom half: stability and magic-step box
    box(
        ax2,
        (1.02, 0.30),
        1.95,
        0.50,
        "Stability limits:\ncΔt ≤ Δz\ncΔt ≤ (Δx⁻² + Δy⁻²)⁻¹/²",
        LGREY,
        GREY,
        fs=12.0,
        tc="black",
    )
    box(
        ax2,
        (3.15, 0.30),
        2.05,
        0.50,
        f"SolverFIT3D(scheme='tete'):\nc * dt / dz = {s_tete:.6f}",
        LBLUE,
        BLUE,
        fs=12.0,
        tc=BLUE,
    )
    box(
        ax2,
        (2.05, -0.38),
        4.35,
        0.50,
        "magic timestep Δt = Δz / c when Δz ≤ (Δx⁻² + Δy⁻²)⁻¹/²",
        LGREEN,
        GREEN,
        fs=13.0,
        tc="black",
    )

    ax2.set_xlim(-0.35, 4.55)
    ax2.set_ylim(-0.78, 4.05)

    fig.savefig(out_dir / "fig14_2_tete_dispersion.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- 14.3
def fig14_3_cold_plasma_wake(out_root):
    """Figure 14.3: Linear cold-plasma wakefield in a cylindrical PEC pipe (wakis.plasma).

    Source lines in user_manual.md: 2537, 2539, 2541-2546, 2550, 2553, 2554, 2558, 2559.
    """
    out_dir = out_root / "section14"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exact benchmark parameters from user_manual.md lines 2553-2559
    n_e = 1.0e20  # m^-3 (line 2553)
    wp = float(plasma_frequency(n_e))  # 5.64e11 rad/s (line 2554)
    k_p = wp / c_light  # line 2537, 2559
    lam_p = float(plasma_wavelength(wp))  # 2*pi/k_p = 3.34 mm (line 2554)
    sigmaz = 0.5e-3  # m (line 2559)
    q = -1.0e-9  # C (line 2559)
    r_obs = 0.2e-3  # m (line 2559)
    a = 2.0e-3  # m (line 2559)

    assert np.isclose(wp, 5.64e11, rtol=2e-3)
    assert np.isclose(lam_p * 1e3, 3.34, rtol=2e-3)

    # Left panel curves: r in [0.05e-3, a] (r/a in [0.025, 1.0])
    r = np.linspace(0.05e-3, a, 400)
    r_over_a = r / a
    g_free = k0(k_p * r)
    g_pipe = pipe_green_function(r, 0.0, k_p, radius=a)
    g_formula = k0(k_p * r) - (k0(k_p * a) / i0(k_p * a)) * i0(k_p * r)  # line 2546 (m = 0 on-axis)
    minus_dg_dr_norm = k1(k_p * r) + (k0(k_p * a) / i0(k_p * a)) * i1(k_p * r)

    # Mathematical assertions for Green's function (Panel 1)
    assert np.allclose(g_pipe, g_formula, rtol=1e-12, atol=1e-14)
    assert abs(pipe_green_function(a, 0.0, k_p, radius=a)) < 1e-14  # vanishes on PEC pipe wall r = a
    h_fd = 1e-8
    dg_fd = (pipe_green_function(r + h_fd, 0.0, k_p, radius=a) - pipe_green_function(r - h_fd, 0.0, k_p, radius=a)) / (
        2.0 * h_fd
    )
    assert np.allclose(-dg_fd / k_p, minus_dg_dr_norm, rtol=1e-7)

    # Right panel curves: xi in [-6.0, 2.0] mm (user_manual.md lines 2558-2559)
    xi = np.linspace(-6.0e-3, 2.0e-3, 401)
    Ez, Fx, Fy = linear_wake(r_obs, 0.0, xi, k_p=k_p, sigmaz=sigmaz, q=q, radius=a)
    assert np.allclose(Fy, 0.0, atol=1e-12)

    # Cross-check against direct quadrature of the integrals in user_manual.md lines 2542-2543
    xi_prime = np.linspace(-6.0e-3, 6.0e-3, 2401)
    lam_qp = (q / (np.sqrt(2.0 * np.pi) * sigmaz)) * np.exp(-0.5 * (xi_prime / sigmaz) ** 2)
    g_r = float(pipe_green_function(r_obs, 0.0, k_p, radius=a))
    dg_dx = float(-k_p * (k1(k_p * r_obs) + (k0(k_p * a) / i0(k_p * a)) * i1(k_p * r_obs)))
    for idx_check in (40, 120, 200):
        xi_c = xi[idx_check]
        mask = xi_prime >= xi_c
        ez_quad = -(k_p**2 / (2.0 * np.pi * eps_0)) * g_r * np.trapezoid(
            lam_qp[mask] * np.cos(k_p * (xi_c - xi_prime[mask])), xi_prime[mask]
        )
        fx_quad = (k_p / (2.0 * np.pi * eps_0)) * dg_dx * np.trapezoid(
            lam_qp[mask] * np.sin(k_p * (xi_prime[mask] - xi_c)), xi_prime[mask]
        )
        assert np.isclose(Ez[idx_check], ez_quad, rtol=2e-4)
        assert np.isclose(Fx[idx_check], fx_quad, rtol=2e-4)

    # Assert Panofsky-Wenzel / pi/2 phase quadrature behind the drive bunch (xi <= -2 mm):
    # d E_z / d xi = -k_p (g / (dg/dx)) * k_p^2 ... -> (1 / g_r) E_z and (-k_p / dg_dx) F_x have equal amplitude
    behind = xi <= -2.5e-3
    amp_ez = np.max(np.abs(Ez[behind])) / g_r
    amp_fx = np.max(np.abs(Fx[behind])) * k_p / abs(dg_dx)
    assert np.isclose(amp_ez, amp_fx, rtol=5e-3)

    fig = plt.figure(figsize=(W, H))
    fig.add_artist(plt.Line2D([0.415, 0.415], [0.06, 0.94], transform=fig.transFigure, color=LGREY, lw=1.5))

    # ---- LEFT PANEL (42% width): Screened pipe Green's function g(r; 0) -----
    fig.text(0.21, 0.93, "Screened pipe Green's function g(r; 0)", ha="center", fontsize=16.5, weight="bold")
    ax1 = fig.add_axes([0.06, 0.14, 0.325, 0.72])
    clean_axes(ax1)

    ax1.plot(r_over_a, g_free, color=GREY, lw=2.2, ls="--", label=r"unscreened $K_0(k_p r)$", zorder=3)
    ax1.plot(
        r_over_a,
        g_pipe,
        color=GREEN,
        lw=2.8,
        label=r"$g(r; 0) = K_0(k_p r) - [K_0(k_p a)/I_0(k_p a)]\,I_0(k_p r)$",
        zorder=4,
    )
    ax1.plot(r_over_a, minus_dg_dr_norm, color=ORANGE, lw=2.4, label=r"$-\,\mathrm{d}g/\mathrm{d}r\, /\, k_p$", zorder=4)

    # Sampling radius r = 0.2 mm (r/a = 0.1)
    ax1.axvline(r_obs / a, color=BLUE, lw=1.6, ls=":")
    ax1.text(
        r_obs / a + 0.025,
        2.55,
        "r = 0.2 mm",
        color=BLUE,
        fontsize=11.5,
        weight="bold",
        ha="left",
        va="center",
    )

    # PEC pipe boundary at r / a = 1 with dark grey hatching
    ax1.axvspan(1.0, 1.07, facecolor=LGREY, edgecolor=GREY, hatch="///", lw=1.2, zorder=3)
    ax1.axvline(1.0, color=GREY, lw=2.2, zorder=4)
    ax1.plot([1.0], [0.0], "o", color=GREEN, ms=7, zorder=5)
    ax1.text(
        0.965,
        0.95,
        "PEC wall r = a\n(g = 0)",
        color=GREY,
        fontsize=11.5,
        weight="bold",
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=GREY, lw=1.0, alpha=0.92),
    )

    ax1.set_xlim(0.0, 1.07)
    ax1.set_ylim(-0.15, 3.65)
    ax1.set_xticks([0.0, 0.1, 0.5, 1.0])
    ax1.set_xticklabels(["0", "0.1", "0.5", "1.0"], fontsize=12.5)
    ax1.set_yticks([0.0, 1.0, 2.0, 3.0])
    ax1.set_xlabel(r"normalized radius $r / a$ ($a = 2.0\ \mathrm{mm}$)", fontsize=13)
    ax1.set_ylabel("Green's function & radial slope", fontsize=13)
    ax1.legend(loc="upper right", bbox_to_anchor=(0.92, 0.99), frameon=False, fontsize=10.2)

    # ---- RIGHT PANEL (58% width): Co-moving wake E_z(r, xi) and F_x(r, xi) --
    fig.text(0.71, 0.93, "Co-moving wake E_z(r, ξ) and F_x(r, ξ)", ha="center", fontsize=16.5, weight="bold")
    ax2 = fig.add_axes([0.495, 0.14, 0.47, 0.72])
    clean_axes(ax2)

    xi_mm = xi * 1e3
    ez_mv_m = Ez * 1e-6  # MV/m
    fx_mv_m = Fx * 1e-6  # MV/m

    # Shaded grey drive-bunch profile lambda(xi) centred at xi = 0
    bunch_profile = np.exp(-0.5 * (xi / sigmaz) ** 2) * (0.82 * np.max(np.abs(fx_mv_m)))
    ax2.fill_between(xi_mm, 0.0, bunch_profile, color=LGREY, edgecolor=GREY, lw=1.3, alpha=0.8, zorder=1)
    ax2.text(
        0.55,
        bunch_profile.max() + 18.0,
        "drive bunch σ_z = 0.5 mm, v = c",
        color=GREY,
        fontsize=11.2,
        weight="bold",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GREY, lw=1.0, alpha=0.94),
    )

    ax2.axhline(0.0, color=GREY, lw=1.0, alpha=0.6)
    ax2.plot(xi_mm, ez_mv_m, color=BLUE, lw=2.8, label=r"$E_z \sim \cos(k_p(\xi - \xi'))$", zorder=4)
    ax2.plot(xi_mm, fx_mv_m, color=ORANGE, lw=2.6, label=r"$F_\perp \sim \sin(k_p(\xi' - \xi))$", zorder=4)

    # Find the two positive peaks of F_x behind the bunch to mark lambda_p = 2*pi/k_p = 3.34 mm
    m_p1 = (xi_mm > -1.5) & (xi_mm < 0.2)
    idx_fx1 = int(np.flatnonzero(m_p1)[np.argmax(fx_mv_m[m_p1])])
    x_fx1 = float(xi_mm[idx_fx1])
    x_fx2 = x_fx1 - lam_p * 1e3
    y_fx_peak = float(fx_mv_m[idx_fx1])
    y_arrow = y_fx_peak + 20.0

    ax2.plot([x_fx1, x_fx1], [y_fx_peak, y_arrow + 5.0], color=GREEN, lw=1.4, ls=":")
    ax2.plot([x_fx2, x_fx2], [y_fx_peak, y_arrow + 5.0], color=GREEN, lw=1.4, ls=":")
    ax2.annotate(
        "",
        xy=(x_fx1, y_arrow),
        xytext=(x_fx2, y_arrow),
        arrowprops=dict(arrowstyle="<|-|>", color=GREEN, lw=2.2, mutation_scale=16),
    )
    ax2.text(
        0.5 * (x_fx1 + x_fx2),
        y_arrow + 6.0,
        "λ_p = 2π / k_p = 3.34 mm",
        color=GREEN,
        fontsize=12.0,
        weight="bold",
        ha="center",
        va="bottom",
    )

    # Mark pi/2 phase quadrature between E_z peak and F_x peak (quarter wavelength lambda_p / 4)
    m_ez = (xi_mm > -4.0) & (xi_mm < -2.0)
    idx_ez1 = int(np.flatnonzero(m_ez)[np.argmax(ez_mv_m[m_ez])])
    x_ez1 = float(xi_mm[idx_ez1])
    y_q = -112.0
    ax2.plot([x_ez1, x_ez1], [ez_mv_m[idx_ez1], y_q], color=GREY, lw=1.2, ls="--")
    ax2.plot([x_fx2, x_fx2], [fx_mv_m[idx_fx1], y_q], color=GREY, lw=1.2, ls="--")
    ax2.annotate(
        "",
        xy=(x_ez1, y_q),
        xytext=(x_fx2, y_q),
        arrowprops=dict(arrowstyle="<|-|>", color=GREY, lw=1.8, mutation_scale=13),
    )
    ax2.text(
        0.5 * (x_ez1 + x_fx2),
        y_q - 8.0,
        "π/2 phase offset",
        color=GREY,
        fontsize=11.5,
        weight="bold",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GREY, lw=1.0, alpha=0.94),
    )

    ax2.set_xlim(-6.0, 2.0)
    ax2.set_ylim(-155.0, 160.0)
    ax2.set_xticks([-6.0, -4.0, -2.0, 0.0, 2.0])
    ax2.set_yticks([-100, -50, 0, 50, 100, 150])
    ax2.set_xlabel(r"co-moving coordinate $\xi = z - ct\ [\mathrm{mm}]$", fontsize=13)
    ax2.set_ylabel(r"wakefield $E_z,\ F_x\ [\mathrm{MV/m}]$ at $r = 0.2\ \mathrm{mm}$", fontsize=13)
    ax2.legend(loc="lower right", frameon=False, fontsize=11.5)

    fig.savefig(out_dir / "fig14_3_cold_plasma_wake.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- 15.2
def fig15_2_cost_to_target(out_root):
    """Figure 15.2: Why wakis.benchmarks evaluates solvers by cost_to_target rather than fixed h.

    Source lines in user_manual.md: 2600, 2607, 2608.
    """
    out_dir = out_root / "section15"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H))
    fig.add_artist(plt.Line2D([0.365, 0.365], [0.06, 0.94], transform=fig.transFigure, color=LGREY, lw=1.5))

    # ---- LEFT PANEL (36% width): Why fixed h is meaningless -----------------
    fig.text(0.185, 0.93, "Why fixed h is meaningless", ha="center", fontsize=17, weight="bold")

    r_arc = 0.68
    th_arc = np.linspace(0.0, 0.5 * np.pi, 100)
    arc_xy = np.column_stack([r_arc * np.cos(th_arc), r_arc * np.sin(th_arc)])

    # Patch 1: Delaunay triangles around curved arc
    ax_p1 = fig.add_axes([0.03, 0.65, 0.14, 0.23])
    bare(ax_p1)
    rng = np.random.default_rng(7)
    grid_pts = np.array([(x, y) for x in np.linspace(0, 1, 6) for y in np.linspace(0, 1, 6)])
    arc_nodes = np.column_stack([r_arc * np.cos(np.linspace(0, np.pi / 2, 7)), r_arc * np.sin(np.linspace(0, np.pi / 2, 7))])
    keep = np.abs(np.hypot(grid_pts[:, 0], grid_pts[:, 1]) - r_arc) > 0.09
    pts_tri = np.vstack([grid_pts[keep], arc_nodes])
    tri_mesh = Delaunay(pts_tri).simplices
    for s in tri_mesh:
        pts_s = pts_tri[s]
        c_s = pts_s.mean(axis=0)
        fc_s = LGREEN if np.hypot(c_s[0], c_s[1]) < r_arc else "white"
        ax_p1.add_patch(Polygon(pts_s, closed=True, fc=fc_s, ec=BLUE, lw=1.1, zorder=2))
    ax_p1.plot(arc_xy[:, 0], arc_xy[:, 1], color=GREY, lw=2.6, zorder=4)
    ax_p1.set_xlim(-0.05, 1.05)
    ax_p1.set_ylim(-0.05, 1.05)
    fig.text(0.18, 0.765, "triangles\n(body-fitted\nDelaunay)", ha="left", va="center", fontsize=13.5, color=BLUE, weight="bold")

    # Patch 2: p = 3 curved DG elements (coarse curved triangles with (p+1)(p+2)/2 = 10 nodes per triangle)
    p_dg = 3
    n_dg_nodes = (p_dg + 1) * (p_dg + 2) // 2
    assert n_dg_nodes == 10  # exact 2D polynomial basis dimension for p = 3 (user_manual.md line 2608)

    ax_p2 = fig.add_axes([0.03, 0.39, 0.14, 0.23])
    bare(ax_p2)
    # Coarse curved triangles on [0, 1]^2 split by the arc at (r_arc, 0) and (0, r_arc)
    dg_tris = [
        (np.array([0.0, 0.0]), np.array([r_arc, 0.0]), np.array([0.0, r_arc]), True),
        (np.array([r_arc, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0]), False),
        (np.array([0.0, r_arc]), np.array([1.0, 1.0]), np.array([0.0, 1.0]), False),
        (np.array([r_arc, 0.0]), np.array([1.0, 1.0]), np.array([0.0, r_arc]), False),
    ]
    bary_p3 = [
        (i / 3.0, j / 3.0, (3 - i - j) / 3.0)
        for i in range(4)
        for j in range(4 - i)
    ]
    assert len(bary_p3) == n_dg_nodes
    # Draw shaded interior of circle
    sec_pts = np.vstack([[0.0, 0.0], arc_xy])
    ax_p2.add_patch(Polygon(sec_pts, closed=True, fc=LGREEN, ec="none", zorder=1))
    ax_p2.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, fc="none", ec=BLUE, lw=1.5, zorder=2))
    ax_p2.plot([r_arc, 1.0], [0.0, 1.0], color=BLUE, lw=1.4, zorder=2)
    ax_p2.plot([0.0, 1.0], [r_arc, 1.0], color=BLUE, lw=1.4, zorder=2)
    ax_p2.plot(arc_xy[:, 0], arc_xy[:, 1], color=GREY, lw=2.6, zorder=4)
    for v0, v1, v2, is_inner in dg_tris:
        for l0, l1, l2 in bary_p3:
            pt = l0 * v0 + l1 * v1 + l2 * v2
            # Isoparametric blend onto the circular arc for the curved edge between (r_arc,0) and (0,r_arc)
            if is_inner and (l1 + l2 > 0):
                ang = 0.5 * np.pi * l2 / (l1 + l2)
                arc_pt = np.array([r_arc * np.cos(ang), r_arc * np.sin(ang)])
                chord_pt = (l1 * v1 + l2 * v2) / (l1 + l2)
                pt = pt + 4.0 * l1 * l2 * (arc_pt - chord_pt)
            elif (not is_inner) and np.allclose(v0, [r_arc, 0.0]) and np.allclose(v2, [0.0, r_arc]) and (l0 + l2 > 0):
                ang = 0.5 * np.pi * l2 / (l0 + l2)
                arc_pt = np.array([r_arc * np.cos(ang), r_arc * np.sin(ang)])
                chord_pt = (l0 * v0 + l2 * v2) / (l0 + l2)
                pt = pt + 4.0 * l0 * l2 * (arc_pt - chord_pt)
            ax_p2.plot(pt[0], pt[1], "o", color=ORANGE, ms=3.2, zorder=5)
    ax_p2.set_xlim(-0.05, 1.05)
    ax_p2.set_ylim(-0.05, 1.05)
    fig.text(
        0.18,
        0.505,
        "p = 3 curved\nDG elements\n(high-order nodes)",
        ha="left",
        va="center",
        fontsize=13.5,
        color=ORANGE,
        weight="bold",
    )

    # Patch 3: 4-level quadtree around the curved arc
    ax_p3 = fig.add_axes([0.03, 0.13, 0.14, 0.23])
    bare(ax_p3)
    ax_p3.add_patch(Polygon(sec_pts, closed=True, fc=LGREEN, ec="none", zorder=1))

    quad_levels = set()

    def subdivide_quad(x0, y0, size, level):
        corners = np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]])
        dists = np.hypot(corners[:, 0], corners[:, 1]) - r_arc
        straddles = (dists.min() <= 0.0) and (dists.max() >= 0.0)
        if level < 3 and straddles:
            half = 0.5 * size
            for dx_q in (0.0, half):
                for dy_q in (0.0, half):
                    subdivide_quad(x0 + dx_q, y0 + dy_q, half, level + 1)
        else:
            quad_levels.add(level)
            ax_p3.add_patch(Rectangle((x0, y0), size, size, fc="none", ec=BLUE, lw=1.0, zorder=2))

    for ix in (0.0, 0.5):
        for iy in (0.0, 0.5):
            subdivide_quad(ix, iy, 0.5, 0)
    assert quad_levels == {0, 1, 2, 3}  # exact 4-level quadtree hierarchy (user_manual.md line 2608)

    ax_p3.plot(arc_xy[:, 0], arc_xy[:, 1], color=GREY, lw=2.6, zorder=4)
    ax_p3.set_xlim(-0.05, 1.05)
    ax_p3.set_ylim(-0.05, 1.05)
    fig.text(
        0.18,
        0.245,
        "4-level quadtree\n(nested Cartesian\nsubgrid)",
        ha="left",
        va="center",
        fontsize=13.5,
        color=BLUE,
        weight="bold",
    )

    # Red cross badge at bottom of Left Panel
    fig.text(
        0.185,
        0.065,
        "✗  different DOFs per h\n→ compare by cost at target error",
        ha="center",
        va="center",
        fontsize=11.5,
        color="white",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.30", fc=RED, ec="none"),
    )

    # ---- RIGHT PANEL (64% width): Log-linear interpolation in cost_to_target
    fig.text(0.685, 0.93, "Log-linear interpolation in cost_to_target", ha="center", fontsize=17, weight="bold")
    ax_r = fig.add_axes([0.445, 0.14, 0.515, 0.72])
    clean_axes(ax_r)

    # Two exact power-law convergence ladders y = C * x^{-alpha} (4 discrete markers each)
    # Blue: alpha_1 = 1 (schematic 2nd-order slope vs 2D unknowns N ~ h^-2)
    # Orange: alpha_2 = 2 (schematic high-order slope)
    x_blue = np.array([1.0, 2.2, 5.2, 10.5])
    c_blue, alpha_blue = 0.085, 1.0
    err_blue = c_blue * x_blue ** (-alpha_blue)

    x_orange = np.array([0.85, 1.45, 2.45, 4.1])
    c_orange, alpha_orange = 0.065, 2.0
    err_orange = c_orange * x_orange ** (-alpha_orange)

    # Call the real wakis.benchmarks.cost_to_target API (user_manual.md line 2608)
    records_blue = [
        {
            "metrics": {"q_sca_rms_rel": float(e)},
            "cost": {"unknowns": float(u), "steps": 100, "run_s": float(0.1 * u)},
        }
        for u, e in zip(x_blue, err_blue)
    ]
    records_orange = [
        {
            "metrics": {"q_sca_rms_rel": float(e)},
            "cost": {"unknowns": float(u), "steps": 100, "run_s": float(0.1 * u)},
        }
        for u, e in zip(x_orange, err_orange)
    ]
    target_val = 0.02
    res_blue = cost_to_target(records_blue, metric="q_sca_rms_rel", target=target_val)
    res_orange = cost_to_target(records_orange, metric="q_sca_rms_rel", target=target_val)
    x_star = res_blue["unknowns"]

    # Bracketing markers (x_k, E_k) above 0.02 and (x_{k+1}, E_{k+1}) below 0.02 on the blue ladder
    k_br = int(np.flatnonzero(err_blue > target_val)[-1])
    xk, ek = float(x_blue[k_br]), float(err_blue[k_br])
    xk1, ek1 = float(x_blue[k_br + 1]), float(err_blue[k_br + 1])
    assert ek > target_val > ek1

    # Assert log-linear interpolation identity and exact agreement with wakis.benchmarks.cost_to_target
    t_interp = (np.log10(target_val) - np.log10(ek)) / (np.log10(ek1) - np.log10(ek))
    x_star_loglin = 10.0 ** (np.log10(xk) + t_interp * (np.log10(xk1) - np.log10(xk)))
    x_star_exact = (target_val / c_blue) ** (-1.0 / alpha_blue)
    assert np.isclose(x_star, x_star_loglin, rtol=1e-14)
    assert np.isclose(x_star, x_star_exact, rtol=1e-14)
    assert np.isclose(res_blue["work"], 100.0 * x_star, rtol=1e-14)
    assert res_orange["unknowns"] < res_blue["unknowns"]

    # Three pre-registered manual threshold lines: 5% (0.05), 2% (0.02), 1% (0.01) (line 2600)
    for t_val, t_lab in ((0.05, "5% (0.05)"), (0.02, "target = 0.02 (2%)"), (0.01, "1% (0.01)")):
        lw_t = 2.0 if t_val == 0.02 else 1.4
        ax_r.axhline(t_val, color=GREEN, lw=lw_t, ls="--", zorder=2)
        ax_r.text(
            12.8,
            t_val * 1.06,
            t_lab,
            color=GREEN,
            fontsize=11.8,
            weight="bold" if t_val == 0.02 else "normal",
            ha="right",
            va="bottom",
        )

    # Plot the two convergence ladders
    ax_r.loglog(
        x_blue,
        err_blue,
        "o-",
        color=BLUE,
        lw=2.4,
        ms=8.5,
        label=r"second-order arm ($y = C\,x^{-1}$)",
        zorder=4,
    )
    ax_r.loglog(
        x_orange,
        err_orange,
        "s-",
        color=ORANGE,
        lw=2.4,
        ms=8.0,
        label=r"high-order arm ($y = C\,x^{-2}$)",
        zorder=4,
    )

    # Highlight bracketing segment (x_k, E_k) -> (x_{k+1}, E_{k+1}) and log-linear crossing
    ax_r.loglog([xk, xk1], [ek, ek1], color=GREEN, lw=4.0, zorder=5)
    ax_r.loglog([xk, xk1], [ek, ek1], "o", color=BLUE, mfc="white", mew=2.4, ms=11, zorder=6)
    ax_r.text(xk * 1.07, ek * 1.06, r"$(x_k,\ E_k)$", color=BLUE, fontsize=12.5, ha="left", va="bottom", weight="bold")
    ax_r.text(
        xk1 * 1.06,
        ek1 * 0.85,
        r"$(x_{k+1},\ E_{k+1})$",
        color=BLUE,
        fontsize=12.5,
        ha="left",
        va="top",
        weight="bold",
    )

    # Annotate coarse -> fine resolution ladder
    ax_r.annotate(
        "coarse → fine resolution ladder",
        xy=(x_blue[0] * 1.15, err_blue[0] / 1.15),
        xytext=(1.12, 0.125),
        color=GREY,
        fontsize=11.8,
        weight="bold",
        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5, mutation_scale=14),
    )

    # Intersection circle at (x_star, 0.02) and vertical drop arrow to horizontal cost axis
    ax_r.loglog([x_star], [target_val], "o", color=GREEN, mfc=LGREEN, mew=2.8, ms=14, zorder=7)
    y_bot = 0.0020
    ax_r.annotate(
        "",
        xy=(x_star, y_bot),
        xytext=(x_star, target_val * 0.88),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.6, mutation_scale=18),
        zorder=6,
    )
    ax_r.text(
        x_star,
        y_bot * 1.22,
        'cost_to_target(records, "q_sca_rms_rel", 0.02)',
        color="white",
        fontsize=11.5,
        weight="bold",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.28", fc=GREEN, ec="none"),
        zorder=8,
    )

    ax_r.set_xlim(0.72, 13.5)
    ax_r.set_ylim(y_bot, 0.16)
    ax_r.minorticks_off()
    ax_r.set_xticks([])
    ax_r.set_yticks([0.01, 0.02, 0.05])
    ax_r.set_yticklabels(["0.01", "0.02", "0.05"], fontsize=12.5)
    ax_r.set_xlabel("cost: unknowns, work = unknowns * steps, or run_s (log scale)", fontsize=13)
    ax_r.set_ylabel("relative error q_sca_rms_rel (log scale)", fontsize=13)
    ax_r.legend(loc="upper right", frameon=False, fontsize=11.8)

    fig.savefig(out_dir / "fig15_2_cost_to_target.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- 16.2
def fig16_2_cell_memory_breakdown(out_root):
    """Figure 16.2: Empirical memory footprint per grid cell in wakis/estimate.py.

    Source lines in user_manual.md: 2732-2749.
    """
    out_dir = out_root / "section16"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull coefficients directly from wakis.estimate and assert against user_manual.md lines 2732-2749
    naive_6_fields = 6 * 8  # user_manual.md line 2732: 6 x 8 = 48 B for (Ex, Ey, Ez, Hx, Hy, Hz) float64
    assert naive_6_fields == 48

    host_plain_f64_fixed, host_plain_f64 = est_mod._HOST[("cpu", False, "float64")]  # line 2738: 392 MB, 1109 B/cell
    host_plain_f32_fixed, host_plain_f32 = est_mod._HOST[("cpu", False, "float32")]  # line 2739: 393 MB, 813 B/cell
    host_cpml_f64_fixed, host_cpml_f64 = est_mod._HOST[("cpu", True, "float64")]  # line 2740: 428 MB, 953 B/cell
    host_gpu_f64_fixed, host_gpu_f64 = est_mod._HOST[("gpu", True, "float64")]  # line 2741: 587 MB, 940 B/cell
    gpu_cpml_f64 = est_mod._GPU[(True, "float64")]  # line 2741: 736 B/cell
    gpu_fused_extra = est_mod._GPU_FUSED_CPML_BYTES  # line 2741: +380 B/cell -> 1116 B/cell
    host_gpu_f32_fixed, host_gpu_f32 = est_mod._HOST[("gpu", True, "float32")]  # line 2742: 566 MB, 605 B/cell
    gpu_cpml_f32 = est_mod._GPU[(True, "float32")]  # line 2742: 457 B/cell

    conf_all_inc, conf_all_cut = est_mod._CONFORMAL_RUN["all"]  # line 2743: +191 B/cell, +4895 B/cut cell
    conf_all_fixed, conf_all_peak, _ = est_mod._CONFORMAL_SETUP["all"]  # line 2743: 420 MB, 1339 B/cell setup peak
    conf_all_gpu = est_mod._CONFORMAL_GPU["all"]  # line 2743: +142 B/cell

    conf_ten_inc, conf_ten_cut = est_mod._CONFORMAL_RUN["tensor"]  # line 2744: +311 B/cell, +3137 B/cut cell
    conf_ten_fixed, conf_ten_peak, _ = est_mod._CONFORMAL_SETUP["tensor"]  # line 2744: 424 MB, 1453 B/cell setup peak
    conf_ten_gpu = est_mod._CONFORMAL_GPU["tensor"]  # line 2744: +154 B/cell

    disp_steady = est_mod._DISPERSIVE_BYTES  # line 2745: +88 B/edge steady
    disp_peak = est_mod._DISPERSIVE_TRANSIENT  # line 2745: +160 B/edge setup peak
    tet_base = est_mod._UNSTRUCTURED_BASE  # line 2746: 396 MB fixed
    tet_per_cell = est_mod._UNSTRUCTURED_PER_TET  # line 2746: 2523 B/tet
    tet_step_ns = est_mod._UNSTRUCTURED_STEP * 1e9  # line 2746: 52 ns/tet/step

    # Assert exact numerical equality with every quoted manual value (lines 2738-2749)
    assert (host_plain_f64_fixed, host_plain_f64) == (392e6, 1109.0)
    assert (host_plain_f32_fixed, host_plain_f32) == (393e6, 813.0)
    assert (host_cpml_f64_fixed, host_cpml_f64) == (428e6, 953.0)
    assert (host_gpu_f64_fixed, host_gpu_f64, gpu_cpml_f64, gpu_fused_extra) == (587e6, 940.0, 736.0, 380.0)
    assert gpu_cpml_f64 + gpu_fused_extra == 1116.0
    assert (host_gpu_f32_fixed, host_gpu_f32, gpu_cpml_f32) == (566e6, 605.0, 457.0)
    assert (conf_all_fixed, host_plain_f64 + conf_all_inc, conf_all_peak, conf_all_gpu, conf_all_cut) == (
        420e6,
        1300.0,
        1339.0,
        142.0,
        4895.0,
    )
    assert (conf_ten_fixed, host_plain_f64 + conf_ten_inc, conf_ten_peak, conf_ten_gpu, conf_ten_cut) == (
        424e6,
        1420.0,
        1453.0,
        154.0,
        3137.0,
    )
    assert (disp_steady, disp_peak, tet_base, tet_per_cell, round(tet_step_ns)) == (88.0, 160.0, 396e6, 2523.0, 52)

    # Also verify via the public wakis.estimate.estimate API that slopes match per-cell coefficients
    e1 = est_mod.estimate(100_000, safety=1.0, backend="gpu", cpml=True, dtype="float64", fused=True)
    e2 = est_mod.estimate(200_000, safety=1.0, backend="gpu", cpml=True, dtype="float64", fused=True)
    assert np.isclose((e2.host_bytes - e1.host_bytes) / 100_000, 940.0)
    assert np.isclose((e2.gpu_bytes - e1.gpu_bytes) / 100_000, 1116.0)

    fig = plt.figure(figsize=(W, H))
    fig.add_artist(plt.Line2D([0.345, 0.345], [0.06, 0.94], transform=fig.transFigure, color=LGREY, lw=1.5))

    # ---- LEFT PANEL (34% width): 48 B naive vs ~0.6-1.4 kB actual -----------
    fig.text(0.172, 0.93, "48 B naive vs ~0.6–1.4 kB actual", ha="center", fontsize=16.5, weight="bold")
    ax1 = fig.add_axes([0.02, 0.05, 0.315, 0.84])
    bare(ax1)

    # Visual comparison of 48 B bar vs the full cell storage stack (lines 2732-2733)
    # Left column: 48 B tiny green bar; Right column: full ~0.6-1.4 kB stack
    ax1.add_patch(Rectangle((0.08, 0.65), 0.72, 0.14, fc=LGREEN, ec=GREEN, lw=2.0, zorder=3))
    ax1.text(0.44, 0.72, "48 B", ha="center", va="center", fontsize=13.5, color=GREEN, weight="bold")
    ax1.text(
        0.44,
        0.42,
        "6 × 8 = 48 B:\n(E_x, E_y, E_z,\nH_x, H_y, H_z)",
        ha="center",
        va="center",
        fontsize=11.5,
        color=GREEN,
        weight="bold",
    )

    # Full stack on right side of Panel 1 (x = 1.05 to 2.95)
    # Bottom slice: 48 B raw fields
    ax1.add_patch(Rectangle((1.02, 0.65), 1.92, 0.14, fc=LGREEN, ec=GREEN, lw=1.6, zorder=3))
    ax1.text(1.98, 0.72, "Raw (E, H) fields: 48 B", ha="center", va="center", fontsize=11.0, color=GREEN, weight="bold")

    # Middle block: Grid & material tensors (blue)
    ax1.add_patch(Rectangle((1.02, 0.82), 1.92, 0.95, fc=LBLUE, ec=BLUE, lw=1.8, zorder=3))
    ax1.text(
        1.98,
        1.295,
        "Grid & material tensors:\nL, tL, S, tS,\nieps, imu, sigma,\nE, H, J",
        ha="center",
        va="center",
        fontsize=11.8,
        color=BLUE,
        weight="bold",
    )

    # Top block: CSR curl matrices (dark grey)
    ax1.add_patch(Rectangle((1.02, 1.80), 1.92, 1.25, fc=LGREY, ec=GREY, lw=1.8, zorder=3))
    ax1.text(
        1.98,
        2.425,
        "CSR curl matrices:\nD, Dt,\ntDsiDmuiDaC,\nitDaiDepsDstC\n(indptr, indices, data)",
        ha="center",
        va="center",
        fontsize=11.8,
        color=GREY,
        weight="bold",
    )

    # Bottom grey note (lines 2734, 2738-2744, 2749)
    box(
        ax1,
        (1.50, -0.15),
        2.88,
        0.66,
        "Fixed Python/SciPy overhead: 392–587 MB\n"
        "×1.25 safety factor  |  ×2.0 for complex128 Bloch\n"
        "÷P per cell for MPI (n_ranks=P)",
        LGREY,
        GREY,
        fs=10.5,
        tc="black",
    )

    ax1.set_xlim(-0.02, 3.04)
    ax1.set_ylim(-0.58, 3.25)

    # ---- RIGHT PANEL (66% width): Empirical bytes per cell ------------------
    fig.text(0.67, 0.93, "Empirical bytes per cell (wakis/estimate.py)", ha="center", fontsize=16.5, weight="bold")
    ax2 = fig.add_axes([0.525, 0.13, 0.455, 0.69])
    clean_axes(ax2)

    labels = [
        "Plain CPU (float64, PEC)",
        "Plain CPU (float32, PEC)",
        "CPML CPU (float64)",
        "CPML GPU (float64, CuPy)",
        "CPML GPU (float32, CuPy)",
        "Conformal (all, float64)",
        "Conformal (tensor, float64)",
    ]
    y_pos = np.arange(len(labels), dtype=float)[::-1]  # top to bottom

    # Reference line at 48 B (6 fields only)
    ax2.axvline(naive_6_fields, color=GREEN, lw=1.8, ls="--", zorder=2)
    ax2.text(
        naive_6_fields + 22,
        y_pos[0] + 0.36,
        "48 B (6 fields only)",
        color=GREEN,
        fontsize=10.5,
        weight="bold",
        ha="left",
        va="bottom",
    )

    bh = 0.33  # bar height

    # (1) Plain CPU (float64, PEC): 1109 B/cell
    ax2.barh(y_pos[0], host_plain_f64, height=0.46, color=BLUE, edgecolor=BLUE, lw=1.2, label="Host RAM (steady)")
    ax2.text(host_plain_f64 + 25, y_pos[0], "1,109 B/cell", va="center", fontsize=11.0, color=BLUE, weight="bold")

    # (2) Plain CPU (float32, PEC): 813 B/cell
    ax2.barh(y_pos[1], host_plain_f32, height=0.46, color=BLUE, edgecolor=BLUE, lw=1.2)
    ax2.text(host_plain_f32 + 25, y_pos[1], "813 B/cell", va="center", fontsize=11.0, color=BLUE, weight="bold")

    # (3) CPML CPU (float64): 953 B/cell
    ax2.barh(y_pos[2], host_cpml_f64, height=0.46, color=BLUE, edgecolor=BLUE, lw=1.2)
    ax2.text(host_cpml_f64 + 25, y_pos[2], "953 B/cell", va="center", fontsize=11.0, color=BLUE, weight="bold")

    # (4) CPML GPU (float64, CuPy): Host 940 B/cell, GPU VRAM 736 B/cell + 380 B/cell fused -> 1,116 B/cell
    ax2.barh(y_pos[3] + bh / 2, host_gpu_f64, height=bh, color=BLUE, edgecolor=BLUE, lw=1.2)
    ax2.text(host_gpu_f64 + 25, y_pos[3] + bh / 2 + 0.02, "940 B/cell", va="center", fontsize=10.5, color=BLUE, weight="bold")
    ax2.barh(
        y_pos[3] - bh / 2,
        gpu_cpml_f64,
        height=bh,
        color=ORANGE,
        edgecolor=ORANGE,
        lw=1.2,
        label="GPU VRAM",
    )
    ax2.barh(
        y_pos[3] - bh / 2,
        gpu_fused_extra,
        left=gpu_cpml_f64,
        height=bh,
        facecolor=LGREEN,
        edgecolor=GREEN,
        hatch="///",
        lw=1.4,
        label="GPU fused=True (+380 B/cell)",
    )
    ax2.text(
        gpu_cpml_f64 + gpu_fused_extra + 25,
        y_pos[3] - bh / 2 - 0.02,
        "736 B/cell (+380 B/cell: fused=True → 1,116 B/cell)",
        va="center",
        fontsize=10.0,
        color=GREEN,
        weight="bold",
    )

    # (5) CPML GPU (float32, CuPy): Host 605 B/cell, GPU VRAM 457 B/cell
    ax2.barh(y_pos[4] + bh / 2, host_gpu_f32, height=bh, color=BLUE, edgecolor=BLUE, lw=1.2)
    ax2.text(host_gpu_f32 + 25, y_pos[4] + bh / 2, "605 B/cell", va="center", fontsize=10.5, color=BLUE, weight="bold")
    ax2.barh(y_pos[4] - bh / 2, gpu_cpml_f32, height=bh, color=ORANGE, edgecolor=ORANGE, lw=1.2)
    ax2.text(gpu_cpml_f32 + 25, y_pos[4] - bh / 2, "457 B/cell", va="center", fontsize=10.5, color=ORANGE, weight="bold")

    # (6) Conformal (all, float64): steady 1109 + 191 = 1300 B/cell, setup peak 1339 B/cell, GPU +142 B/cell, +4895 B/cut cell
    steady_all = host_plain_f64 + conf_all_inc
    ax2.barh(y_pos[5] + bh / 2, steady_all, height=bh, color=BLUE, edgecolor=BLUE, lw=1.2)
    ax2.barh(
        y_pos[5] + bh / 2,
        conf_all_peak - steady_all,
        left=steady_all,
        height=bh,
        facecolor="none",
        edgecolor=RED,
        lw=2.0,
        label="Conformal setup peak",
    )
    ax2.text(
        conf_all_peak + 25,
        y_pos[5] + bh / 2 + 0.02,
        "1109 + 191 = 1,300 B/cell (1,339 B/cell setup peak)",
        va="center",
        fontsize=9.8,
        color=RED,
        weight="bold",
    )
    ax2.barh(y_pos[5] - bh / 2, conf_all_gpu, height=bh, color=ORANGE, edgecolor=ORANGE, lw=1.2)
    ax2.text(
        conf_all_gpu + 25,
        y_pos[5] - bh / 2 - 0.02,
        "+142 B/cell VRAM  |  +4,895 B / cut cell",
        va="center",
        fontsize=9.8,
        color=ORANGE,
        weight="bold",
    )

    # (7) Conformal (tensor, float64): steady 1109 + 311 = 1420 B/cell, setup peak 1453 B/cell, GPU +154 B/cell, +3137 B/cut cell
    steady_ten = host_plain_f64 + conf_ten_inc
    ax2.barh(y_pos[6] + bh / 2, steady_ten, height=bh, color=BLUE, edgecolor=BLUE, lw=1.2)
    ax2.barh(
        y_pos[6] + bh / 2,
        conf_ten_peak - steady_ten,
        left=steady_ten,
        height=bh,
        facecolor="none",
        edgecolor=RED,
        lw=2.0,
    )
    ax2.text(
        conf_ten_peak + 25,
        y_pos[6] + bh / 2 + 0.02,
        "1109 + 311 = 1,420 B/cell (1,453 B/cell setup peak)",
        va="center",
        fontsize=9.8,
        color=RED,
        weight="bold",
    )
    ax2.barh(y_pos[6] - bh / 2, conf_ten_gpu, height=bh, color=ORANGE, edgecolor=ORANGE, lw=1.2)
    ax2.text(
        conf_ten_gpu + 25,
        y_pos[6] - bh / 2 - 0.02,
        "+154 B/cell VRAM  |  +3,137 B / cut cell",
        va="center",
        fontsize=9.8,
        color=ORANGE,
        weight="bold",
    )

    # Inset callout for UnstructuredFIT3D and Dispersive ADE (lines 2745-2746)
    ax2.text(
        3120,
        y_pos[1],
        "UnstructuredFIT3D:\n  2,523 B/tet, 396 MB fixed, 52 ns/tet/step\n"
        "Dispersive ADE:\n  +88 B/edge steady, +160 B/edge setup peak",
        fontsize=10.0,
        color=GREY,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28", fc=LGREY, ec=GREY, lw=1.2),
        zorder=6,
    )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=11.0, weight="bold")
    ax2.set_xlim(0, 3180)
    ax2.set_ylim(-0.65, len(labels) - 0.25)
    ax2.set_xticks([0, 500, 1000, 1500])
    ax2.set_xlabel("bytes per cell [B/cell] (before ×1.25 safety factor)", fontsize=12.5)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=2, frameon=False, fontsize=10.0)

    fig.savefig(out_dir / "fig16_2_cell_memory_breakdown.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    out_root = Path(sys.argv[1])
    out_root.mkdir(parents=True, exist_ok=True)
    fig13_4_subpixel_smoothed_projection(out_root)
    fig14_2_tete_dispersion(out_root)
    fig14_3_cold_plasma_wake(out_root)
    fig15_2_cost_to_target(out_root)
    fig16_2_cell_memory_breakdown(out_root)
