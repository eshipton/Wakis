"""Generate 480x320 schematic thumbnails for planned Wakis examples.

Each thumbnail uses a consistent light palette:
- Si core: dark slate (#334155)
- SiO2 cladding/substrate: light blue (#dbeafe)
- Metal (Au/Ag/Al/Cu): gold (#eab308)
- SiN core: teal (#0d9488)
- LiNbO3 (TFLN): violet (#7c3aed)
- Phase-change material (PCM: GST/Sb2Se3/VO2): orange (#ea580c)

Usage
-----
Run via memguard from the repository root:
    MPLBACKEND=Agg python tutorials/gallery/make_thumbnails.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
    Wedge,
)
import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent / "thumbs"

# Fixed material palette
PAL = {
    "bg": "#f8fafc",
    "sio2": "#dbeafe",
    "sio2_edge": "#93c5fd",
    "si": "#334155",
    "si_edge": "#0f172a",
    "metal": "#eab308",
    "metal_edge": "#a16207",
    "sin": "#0d9488",
    "sin_edge": "#115e59",
    "ln": "#7c3aed",
    "ln_edge": "#4c1d95",
    "pcm": "#ea580c",
    "pcm_edge": "#9a3412",
    "poly": "#db2777",
    "poly_edge": "#831843",
    "alox": "#60a5fa",
    "alox_edge": "#1d4ed8",
    "p_dop": "#ef4444",
    "n_dop": "#3b82f6",
    "light_in": "#dc2626",
    "light_out": "#2563eb",
    "text": "#0f172a",
    "muted": "#475569",
    "strip_bg": "#e2e8f0",
}


def new_canvas(caption: str, view_label: str = "top view"):
    """Create a 480x320 px figure (0..10 x 0..6.67 data coordinates)."""
    fig = plt.figure(figsize=(4.8, 3.2), dpi=100, facecolor=PAL["bg"])
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=PAL["bg"])
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 6.67)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Subtle SiO2 background slab pad inside drawing region
    pad = FancyBboxPatch(
        (0.35, 0.85),
        9.30,
        5.45,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor="#eff6ff",
        edgecolor="#cbd5e1",
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(pad)

    # Bottom caption strip
    strip = Rectangle(
        (0.0, 0.0),
        10.0,
        0.68,
        facecolor=PAL["strip_bg"],
        edgecolor="#cbd5e1",
        linewidth=0.8,
        zorder=20,
    )
    ax.add_patch(strip)
    ax.text(
        0.30,
        0.33,
        caption,
        fontsize=8.1,
        fontweight="bold",
        color=PAL["text"],
        va="center",
        ha="left",
        zorder=21,
    )
    ax.text(
        9.70,
        0.33,
        view_label,
        fontsize=7.3,
        color=PAL["muted"],
        va="center",
        ha="right",
        style="italic",
        zorder=21,
    )

    # Corner "schematic" stamp
    stamp = FancyBboxPatch(
        (8.05, 5.78),
        1.45,
        0.38,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        facecolor="#fef3c7",
        edgecolor="#d97706",
        linewidth=0.8,
        zorder=22,
    )
    ax.add_patch(stamp)
    ax.text(
        8.775,
        5.97,
        "schematic",
        fontsize=7.2,
        fontweight="bold",
        color="#92400e",
        va="center",
        ha="center",
        zorder=23,
    )
    return fig, ax


def save_thumb(fig: plt.Figure, filename: str) -> Path:
    """Save figure as an optimized PNG under 60 kB."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    # Quantize to 64 colors and optimize via Pillow to stay well under 60 kB
    with Image.open(out_path) as im:
        im_q = im.convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT)
        im_q.save(out_path, format="PNG", optimize=True)
    return out_path


def arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str = "",
    color: str = PAL["light_in"],
    lw: float = 2.0,
    label_offset: tuple[float, float] = (0.0, 0.25),
):
    """Draw an optical input/output arrow with an optional port label."""
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>,head_length=6,head_width=4",
        color=color,
        linewidth=lw,
        zorder=15,
    )
    ax.add_patch(arr)
    if label:
        xm = 0.5 * (x0 + x1) + label_offset[0]
        ym = 0.5 * (y0 + y1) + label_offset[1]
        ax.text(
            xm,
            ym,
            label,
            fontsize=7.5,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            zorder=16,
        )


def mat_badge(ax, x: float, y: float, text: str, mat: str = "si"):
    """Small material legend pill."""
    fc = PAL.get(mat, PAL["si"])
    tc = "#ffffff" if mat in ("si", "sin", "ln", "pcm", "poly", "p_dop", "n_dop") else PAL["text"]
    box = FancyBboxPatch(
        (x, y),
        len(text) * 0.14 + 0.35,
        0.34,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc,
        edgecolor="none",
        alpha=0.92,
        zorder=18,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.5 * (len(text) * 0.14 + 0.35),
        y + 0.17,
        text,
        fontsize=6.8,
        fontweight="bold",
        color=tc,
        ha="center",
        va="center",
        zorder=19,
    )


# ---------------------------------------------------------------------------
# Parametric drawing primitives by device archetype
# ---------------------------------------------------------------------------


def draw_y_junction(ax, variant: str = "standard"):
    """Draw a symmetric 1x2 Y-junction splitter with 13-segment adiabatic taper (top view)."""
    y_c = 3.55
    w = 0.34
    # Input stem + 13-segment adiabatic multimode taper + two sine S-bends as one unified polygon
    x_taper = np.linspace(2.4, 4.1, 13)
    hw_taper = np.array([0.17, 0.17, 0.21, 0.26, 0.34, 0.45, 0.50, 0.50, 0.50, 0.50, 0.47, 0.43, 0.43])
    x_sp = np.linspace(4.1, 7.2, 60)
    s = (x_sp - 4.1) / 3.1 - np.sin(2.0 * np.pi * (x_sp - 4.1) / 3.1) / (2.0 * np.pi)
    if variant == "bayes":
        s = s + 0.07 * np.sin(3.0 * np.pi * (x_sp - 4.1) / 3.1)
    y_arm_c = 0.26 + 0.94 * s

    top_x = np.concatenate([[0.8, 2.4], x_taper, x_sp, [9.0, 9.0], x_sp[::-1], [4.1], x_sp, [9.0, 9.0], x_sp[::-1], x_taper[::-1], [2.4, 0.8]])
    top_y = np.concatenate([
        [y_c + w / 2, y_c + w / 2],
        y_c + hw_taper,
        y_c + y_arm_c + w / 2,
        [y_c + 1.20 + w / 2, y_c + 1.20 - w / 2],
        (y_c + y_arm_c - w / 2)[::-1],
        [y_c],
        y_c - y_arm_c + w / 2,
        [y_c - 1.20 + w / 2, y_c - 1.20 - w / 2],
        (y_c - y_arm_c - w / 2)[::-1],
        (y_c - hw_taper)[::-1],
        [y_c - w / 2, y_c - w / 2],
    ])
    ax.add_patch(Polygon(np.column_stack([top_x, top_y]), closed=True, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.1))
    if variant == "bayes":
        for xk, hwk in zip(x_taper[::2], hw_taper[::2]):
            ax.add_patch(Circle((xk, y_c + hwk), 0.06, color=PAL["pcm"], zorder=12))
            ax.add_patch(Circle((xk, y_c - hwk), 0.06, color=PAL["pcm"], zorder=12))
    arrow(ax, 0.5, y_c, 1.6, y_c, "TE0 in", PAL["light_in"], label_offset=(0.1, 0.36))
    arrow(ax, 8.1, y_c + 1.20, 9.3, y_c + 1.20, "50%", PAL["light_out"], label_offset=(0.0, 0.34))
    arrow(ax, 8.1, y_c - 1.20, 9.3, y_c - 1.20, "50%", PAL["light_out"], label_offset=(0.0, -0.36))
    mat_badge(ax, 0.6, 5.75, "Si 220 nm (13-seg taper)", "si")
    mat_badge(ax, 3.6, 5.75, "SiO2 clad", "sio2")


def draw_ring_resonator(ax, mode: str = "bus"):
    """Draw a microring resonator with a narrow evanescent coupling gap to bus waveguide(s)."""
    mat = "ln" if mode == "thz" else "si"
    fc, ec = PAL[mat], PAL[f"{mat}_edge"]
    w_wg = 0.26
    gap = 0.032  # Visually narrow evanescent coupling gap (~12% of waveguide width)

    if mode == "bus":
        # T02: Point-coupled bus + lower half of ring truncated at top PML/absorber
        y_bus = 1.65
        r_out = 2.45
        yc = y_bus + w_wg / 2 + gap + r_out
        ax.add_patch(Rectangle((0.8, y_bus - w_wg / 2), 8.4, w_wg, facecolor=fc, edgecolor=ec, lw=1))
        w_ring = Wedge((5.0, yc), r_out, 192, 348, width=w_wg, facecolor=fc, edgecolor=ec, lw=1)
        ax.add_patch(w_ring)
        # Top absorber / PML band truncating the ring
        ax.add_patch(Rectangle((0.8, 3.60), 8.4, 0.42, facecolor="#cbd5e1", edgecolor="#64748b", lw=0.9, hatch="//", zorder=11))
        ax.text(5.0, 3.81, "PML / 60-layer adiabatic absorber (truncates ring)", fontsize=7.0, fontweight="bold", color=PAL["text"], ha="center", va="center", zorder=12)
        arrow(ax, 0.5, y_bus, 1.6, y_bus, "TE0 in", PAL["light_in"], label_offset=(0.0, -0.32))
        arrow(ax, 8.2, y_bus, 9.3, y_bus, "thru |t|²", PAL["light_out"], label_offset=(0.0, -0.32))
        arrow(ax, 6.75, 2.55, 7.25, 3.35, "drop |κ|²", PAL["light_out"], label_offset=(0.75, -0.10))
        ax.text(5.0, y_bus + 0.42, "gap = 50 nm", fontsize=7.3, fontweight="bold", color=PAL["light_in"], ha="center")
        mat_badge(ax, 0.6, 5.75, "Si core (R = 5 µm)", mat)
        return

    yc = 3.42
    r_out = 1.48 if mode != "demux" else 1.12
    y_bus = yc - r_out - gap - w_wg / 2
    ax.add_patch(Rectangle((0.8, y_bus - w_wg / 2), 8.4, w_wg, facecolor=fc, edgecolor=ec, lw=1))
    arrow(ax, 0.5, y_bus, 1.6, y_bus, "in", PAL["light_in"], label_offset=(0.0, -0.32))
    arrow(ax, 8.3, y_bus, 9.4, y_bus, "thru", PAL["light_out"], label_offset=(0.0, -0.32))

    if mode == "demux":
        y_drop = yc + r_out + gap + w_wg / 2
        ax.add_patch(Rectangle((0.8, y_drop - w_wg / 2), 8.4, w_wg, facecolor=fc, edgecolor=ec, lw=1))
        for x_r, r_i, lbl in [(3.5, 1.12, "f1"), (6.5, 1.06, "f2")]:
            w_ring = Wedge((x_r, yc), r_i, 0, 360, width=w_wg, facecolor=fc, edgecolor=ec, lw=1)
            ax.add_patch(w_ring)
            ax.text(x_r, yc, lbl, fontsize=8, fontweight="bold", ha="center", va="center", color=PAL["text"])
        arrow(ax, 1.8, y_drop, 0.7, y_drop, "drop", PAL["light_out"], label_offset=(0.0, 0.30))
    else:
        w_ring = Wedge((5.0, yc), r_out, 0, 360, width=w_wg, facecolor=fc, edgecolor=ec, lw=1)
        ax.add_patch(w_ring)
        if mode in ("add_drop", "thz"):
            y_drop = yc + r_out + gap + w_wg / 2
            ax.add_patch(Rectangle((0.8, y_drop - w_wg / 2), 8.4, w_wg, facecolor=fc, edgecolor=ec, lw=1))
            arrow(ax, 1.9, y_drop, 0.7, y_drop, "drop", PAL["light_out"], label_offset=(0.0, 0.30))
        if mode == "thermal":
            heater = Wedge(
                (5.0, yc), r_out + 0.05, 35, 145, width=0.38, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"], alpha=0.90, zorder=10
            )
            ax.add_patch(heater)
            ax.text(5.0, yc + r_out + 0.32, "TiN micro-heater (ΔT)", fontsize=7.5, fontweight="bold", color=PAL["metal_edge"], ha="center")
        ax.text(5.0, yc, f"R = {'5' if mode != 'thz' else '85'} µm\ngap = 100 nm", fontsize=7.5, fontweight="bold", ha="center", va="center", color=PAL["text"])
    mat_badge(ax, 0.6, 5.75, "Si core" if mat == "si" else "High-res Si (THz)", mat)


def draw_crossing(ax, swg: bool = False):
    """Draw a 90-degree waveguide crossing with convex cosine tapers or SWG."""
    xc, yc = 5.0, 3.45
    # Draw fill without edge seams first, then outline
    x = np.linspace(1.0, 9.0, 160)
    dx = np.abs(x - xc)
    # Cosine taper bulges out to w_m around dx=1.2 and meets central square
    w_x = np.where(dx < 2.6, 0.30 + 0.52 * np.cos(0.42 * (dx - 1.1)) ** 2, 0.30)
    pts_h = np.vstack([np.column_stack([x, yc + w_x / 2]), np.column_stack([x[::-1], (yc - w_x / 2)[::-1]])])

    y = np.linspace(1.0, 5.9, 140)
    dy = np.abs(y - yc)
    w_y = np.where(dy < 2.1, 0.30 + 0.52 * np.cos(0.52 * (dy - 0.9)) ** 2, 0.30)
    pts_v = np.vstack([np.column_stack([xc - w_y / 2, y]), np.column_stack([(xc + w_y / 2)[::-1], y[::-1]])])

    for pts in (pts_h, pts_v):
        ax.add_patch(Polygon(pts, closed=True, facecolor=PAL["si_edge"], edgecolor=PAL["si_edge"], lw=2.0, zorder=4))
        ax.add_patch(Polygon(pts, closed=True, facecolor=PAL["si"], edgecolor="none", zorder=5))

    if swg:
        for xp in np.linspace(2.2, 7.8, 15):
            if abs(xp - xc) > 0.45:
                ax.add_patch(Rectangle((xp - 0.06, yc - 0.40), 0.12, 0.80, facecolor=PAL["sio2"], edgecolor="none", zorder=8))
        mat_badge(ax, 0.6, 5.75, "SWG Si metamaterial", "si")
    else:
        # Self-imaging focal waist glow at center
        ax.add_patch(Ellipse((xc, yc), 1.4, 0.42, facecolor=PAL["light_in"], alpha=0.32, zorder=7))
        mat_badge(ax, 0.6, 5.75, "Cosine-taper Si crossing", "si")

    arrow(ax, 0.6, yc, 1.8, yc, "TE0 in", PAL["light_in"], label_offset=(0.0, 0.35))
    arrow(ax, 8.1, yc, 9.3, yc, "thru (-0.2 dB)", PAL["light_out"], label_offset=(-0.2, 0.35))
    ax.text(xc + 0.85, 5.25, "crosstalk\n< -30 dB", fontsize=7.3, color=PAL["muted"], ha="left")


def draw_bend(ax, style: str = "euler"):
    """Draw a low-loss clothoid Euler bend (compared against circular arc), CMA-ES S-bend, or 90-bend PSR."""
    if style == "sbend":
        x = np.linspace(1.0, 9.0, 100)
        s = 0.5 * (1.0 - np.cos(np.pi * (x - 1.0) / 8.0))
        yc = 2.1 + 2.6 * s + 0.18 * np.sin(2.0 * np.pi * (x - 1.0) / 8.0)
        w = 0.36
        pts = np.vstack([np.column_stack([x, yc + w / 2]), np.column_stack([x[::-1], (yc - w / 2)[::-1]])])
        ax.add_patch(Polygon(pts, closed=True, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2))
        for xk in np.linspace(2.0, 8.0, 6):
            yk = np.interp(xk, x, yc)
            ax.add_patch(Circle((xk, yk), 0.09, color=PAL["pcm"], zorder=12))
        arrow(ax, 0.6, 2.1, 1.7, 2.1, "in", PAL["light_in"], label_offset=(0.0, 0.35))
        arrow(ax, 8.2, 4.7, 9.3, 4.7, "out", PAL["light_out"], label_offset=(0.0, 0.35))
        mat_badge(ax, 0.6, 5.75, "CMA-ES spline S-bend", "si")
        return

    # True clothoid Euler bend vs constant-radius circular bend of same R_eff
    x0, y0 = 2.3, 1.55
    r_eff = 3.25
    # Parametric clothoid-like curve (p=3 superellipse / Fresnel blend that starts with zero curvature at ends and tighter R_min at 45 deg)
    u = np.linspace(0.0, 1.0, 90)
    theta = 0.5 * np.pi * (3 * u**2 - 2 * u**3)
    # Radial pull-in at 45 deg characteristic of clothoid Euler bend vs circle
    r_pull = 0.38 * np.sin(np.pi * u) ** 2
    xc_e = x0 + r_eff * np.sin(theta) - r_pull * np.sin(theta)
    yc_e = (y0 + r_eff) - r_eff * np.cos(theta) + r_pull * np.cos(theta)

    x_full = np.concatenate([[0.9, x0], xc_e, [x0 + r_eff, x0 + r_eff]])
    y_full = np.concatenate([[y0, y0], yc_e, [y0 + r_eff, 5.75]])
    dx = np.gradient(x_full)
    dy = np.gradient(y_full)
    dn = np.hypot(dx, dy)
    nx, ny = -dy / dn, dx / dn
    w = 0.38
    pts = np.vstack(
        [
            np.column_stack([x_full + nx * w / 2, y_full + ny * w / 2]),
            np.column_stack([(x_full - nx * w / 2)[::-1], (y_full - ny * w / 2)[::-1]]),
        ]
    )
    ax.add_patch(Polygon(pts, closed=True, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2, zorder=6))

    # Overlay constant-radius circular bend centerline for visual contrast
    th_c = np.linspace(0, np.pi / 2, 60)
    ax.plot(x0 + r_eff * np.sin(th_c), (y0 + r_eff) - r_eff * np.cos(th_c), "--", color=PAL["pcm"], lw=1.6, zorder=9)
    if style == "psr90":
        pts2 = np.vstack(
            [
                np.column_stack([x_full + nx * (w / 2 + 0.05), y_full + ny * (w / 2 + 0.05)]),
                np.column_stack([(x_full + nx * (w / 2 + 0.30))[::-1], (y_full + ny * (w / 2 + 0.30))[::-1]]),
            ]
        )
        ax.add_patch(Polygon(pts2, closed=True, facecolor=PAL["sin"], edgecolor=PAL["sin_edge"], lw=1.0, zorder=7))
        ax.text(6.2, 2.8, "Coupled 90° bend\nTM0 → TE1 → TE0", fontsize=7.8, fontweight="bold", color=PAL["text"])
    else:
        ax.text(6.0, 2.75, "Euler clothoid: κ(s) = s/A²\n(dashed: circular R_eff)", fontsize=7.6, fontweight="bold", color=PAL["text"])
    arrow(ax, 0.55, y0, 1.75, y0, "TE0 in", PAL["light_in"], label_offset=(0.0, 0.35))
    arrow(ax, x0 + r_eff, 5.05, x0 + r_eff, 6.05, "out", PAL["light_out"], label_offset=(0.55, 0.0))
    mat_badge(ax, 0.6, 5.75, "Si clothoid Euler bend", "si")


def draw_directional_coupler(ax, variant: str = "standard"):
    """Draw parallel coupled waveguides with a narrow evanescent coupling gap over Lc."""
    mat = "ln" if variant == "tfln" else "si"
    fc, ec = PAL[mat], PAL[f"{mat}_edge"]
    x = np.linspace(0.8, 9.2, 180)
    # Flat-top coupling section over x in [3.3, 6.7] with narrow gap
    u = np.clip((np.abs(x - 5.0) - 1.6) / 1.8, 0.0, 1.0)
    s_bend = 3 * u**2 - 2 * u**3
    w0 = 0.28
    gap_min = 0.055 if variant != "exceptional" else 0.32
    y_sep = (w0 / 2 + gap_min / 2) + 1.05 * s_bend
    w_top = np.full_like(x, w0)
    w_bot = np.full_like(x, w0)
    if variant == "broadband":
        # Asymmetric phase-control width step in center
        mask = np.exp(-((x - 5.0) / 0.9) ** 4)
        w_top = w0 + 0.09 * mask
        w_bot = w0 - 0.07 * mask
    elif variant == "tfln":
        w_top = np.linspace(0.38, 0.20, len(x))
        w_bot = np.linspace(0.20, 0.38, len(x))

    yc = 3.45
    for sign, w_arr in [(+1.0, w_top), (-1.0, w_bot)]:
        yb = yc + sign * y_sep
        pts = np.vstack(
            [
                np.column_stack([x, yb + 0.5 * w_arr]),
                np.column_stack([x[::-1], (yb - 0.5 * w_arr)[::-1]]),
            ]
        )
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, lw=1))

    if variant == "exceptional":
        for xp in np.linspace(3.3, 6.7, 13):
            ax.add_patch(Rectangle((xp - 0.06, yc - 0.11), 0.12, 0.22, facecolor=PAL["sin"], edgecolor=PAL["sin_edge"]))
        mat_badge(ax, 0.6, 5.75, "Zero-crosstalk e-skid", "si")
    else:
        mat_badge(ax, 0.6, 5.75, "TFLN (LiNbO3)" if mat == "ln" else "Si 220 nm (gap 100 nm)", mat)

    y_port = yc + (w0 / 2 + gap_min / 2) + 1.05
    arrow(ax, 0.5, y_port, 1.5, y_port, "in", PAL["light_in"], label_offset=(0.0, 0.32))
    arrow(ax, 8.3, y_port, 9.4, y_port, "thru", PAL["light_out"], label_offset=(0.0, 0.32))
    arrow(ax, 8.3, 2 * yc - y_port, 9.4, 2 * yc - y_port, "cross", PAL["light_out"], label_offset=(0.0, -0.34))
    ax.annotate("", xy=(3.3, yc + 0.48), xytext=(6.7, yc + 0.48), arrowprops=dict(arrowstyle="<->", color=PAL["muted"], lw=1.0))
    ax.text(5.0, yc + 0.72, "Coupling length Lc (narrow gap)", fontsize=7.4, fontweight="bold", color=PAL["text"], ha="center")


def draw_mmi(ax, kind: str = "1x4"):
    """Draw a multimode interference splitter/coupler with wide multimode box."""
    yc = 3.45
    box_x0, box_w = 2.6, 4.5
    box_h = 2.55 if kind in ("1x4", "hybrid90") else 1.85
    ax.add_patch(
        Rectangle((box_x0, yc - box_h / 2), box_w, box_h, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2)
    )
    # Self-imaging interference fringes inside wide MMI box
    xg = np.linspace(box_x0 + 0.12, box_x0 + box_w - 0.12, 70)
    yg = np.linspace(yc - box_h / 2 + 0.12, yc + box_h / 2 - 0.12, 45)
    XX, YY = np.meshgrid(xg, yg)
    u = (XX - box_x0) / box_w
    v = (YY - yc) / (box_h / 2)
    intensity = (
        (1 - u) * np.exp(-(v / 0.28) ** 2)
        + 4 * u * (1 - u) * (np.exp(-((v - 0.45) / 0.22) ** 2) + np.exp(-((v + 0.45) / 0.22) ** 2))
        + u * np.cos(2.0 * np.pi * v) ** 2
    )
    ax.contourf(XX, YY, intensity, levels=5, cmap="Blues", alpha=0.35, zorder=5)
    ax.text(box_x0 + box_w / 2, yc + box_h / 2 + 0.24, "Wide multimode interference (MMI) region", fontsize=7.4, fontweight="bold", color=PAL["text"], ha="center")

    if kind in ("1x4", "demux"):
        in_ys = [yc] if kind == "1x4" else [yc + 0.35]
    elif kind == "2x2":
        in_ys = [yc + 0.48, yc - 0.48]
    else:
        in_ys = [yc + 0.58, yc - 0.58]

    for yi in in_ys:
        pts_in = [[0.8, yi - 0.13], [box_x0, yi - 0.22], [box_x0, yi + 0.22], [0.8, yi + 0.13]]
        ax.add_patch(Polygon(pts_in, closed=True, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
    arrow(ax, 0.5, in_ys[0], 1.5, in_ys[0], "in", PAL["light_in"], label_offset=(0.0, 0.30))

    if kind in ("1x4", "hybrid90"):
        out_ys = [yc + 0.88, yc + 0.30, yc - 0.30, yc - 0.88]
    else:
        out_ys = [yc + 0.48, yc - 0.48]

    for idx, yo in enumerate(out_ys):
        pts_out = [[box_x0 + box_w, yo - 0.20], [8.7, yo - 0.12], [8.7, yo + 0.12], [box_x0 + box_w, yo + 0.20]]
        ax.add_patch(Polygon(pts_out, closed=True, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        lbl = f"ch{idx+1}" if kind != "demux" else ("1.55 µm" if idx == 0 else "2.0 µm")
        arrow(ax, 8.0, yo, 8.95, yo, "", PAL["light_out"])
        ax.text(9.08, yo, lbl, fontsize=7.2, fontweight="bold", color=PAL["light_out"], va="center", ha="left")

    mat_badge(ax, 0.6, 5.75, f"Si MMI ({kind})", "si")


def draw_grating_coupler(ax, variant: str = "uniform"):
    """Draw a waveguide grating coupler (side view or focusing top view)."""
    if variant == "focusing":
        x0, y0 = 1.3, 3.35
        # Continuous sector wedge connected to input strip waveguide (fits cleanly inside pad)
        ax.add_patch(Rectangle((0.6, y0 - 0.22), 1.2, 0.44, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        wedge_slab = Wedge((x0, y0), 6.5, -21, 21, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.1)
        ax.add_patch(wedge_slab)
        for idx, r_arc in enumerate(np.linspace(2.6, 6.1, 11)):
            w_arc = 0.09 + 0.08 * (idx / 10.0)
            arc = Wedge((x0, y0), r_arc, -19.5, 19.5, width=w_arc, facecolor=PAL["sio2"], edgecolor=PAL["si_edge"], lw=0.6, zorder=6)
            ax.add_patch(arc)
        spot = Ellipse((4.6, y0), 2.1, 1.55, facecolor=PAL["light_in"], edgecolor="#991b1b", lw=1.2, alpha=0.25, zorder=10)
        ax.add_patch(spot)
        ax.text(8.35, 3.35, "Apodized arcs\n+ fiber spot\n(MFD 10.4 µm)", fontsize=7.4, fontweight="bold", color="#991b1b", ha="center", va="center")
        arrow(ax, 1.9, y0, 0.65, y0, "TE0 out", PAL["light_out"], label_offset=(0.0, 0.42))
        mat_badge(ax, 0.6, 5.75, "Focusing apodized Si GC", "si")
        return

    # Cross-section view (x-z)
    ax.add_patch(Rectangle((0.6, 1.0), 8.8, 0.85, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
    ax.text(5.0, 1.42, "Si substrate", fontsize=7.2, color="#ffffff", ha="center", va="center")
    ax.add_patch(Rectangle((0.6, 1.85), 8.8, 0.90, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
    ax.text(1.6, 2.30, "SiO2 BOX (2 µm)", fontsize=7.0, color=PAL["muted"], ha="center", va="center")

    y_wg = 2.75
    h_wg = 0.44
    # Build continuous slab + teeth outline without internal horizontal seam
    ax.add_patch(Rectangle((0.6, y_wg), 8.8, h_wg * 0.48, facecolor=PAL["si"], edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((0.6, y_wg), 2.2, h_wg, facecolor=PAL["si"], edgecolor="none", zorder=5))
    n_teeth = 11
    for i in range(n_teeth):
        xt = 2.95 + i * 0.48
        w_t = 0.26 if variant == "uniform" else (0.14 + 0.22 * (i / n_teeth))
        ax.add_patch(Rectangle((xt, y_wg + h_wg * 0.45), w_t, h_wg * 0.55, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=0.8, zorder=6))
        if variant == "apodized":
            # Blazed two-step shallow + deep etch per period (T11)
            ax.add_patch(Rectangle((xt - 0.10, y_wg + h_wg * 0.45), 0.10, h_wg * 0.28, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=0.7, zorder=6))
        if variant in ("bilayer", "antenna"):
            ax.add_patch(
                Rectangle((xt + 0.10, y_wg + h_wg + 0.12), w_t, 0.26, facecolor=PAL["sin"], edgecolor=PAL["sin_edge"], zorder=7)
            )
    ax.plot([0.6, 9.4], [y_wg, y_wg], color=PAL["si_edge"], lw=1.0, zorder=7)
    ax.plot([0.6, 2.8], [y_wg + h_wg, y_wg + h_wg], color=PAL["si_edge"], lw=1.0, zorder=7)

    fiber_pts = np.array([[5.6, 5.95], [6.9, 5.75], [6.4, 4.35], [5.1, 4.55]])
    ax.add_patch(Polygon(fiber_pts, closed=True, facecolor="#cbd5e1", edgecolor="#475569", lw=1))
    ax.text(6.05, 5.2, "SMF-28", fontsize=7.2, fontweight="bold", color=PAL["text"], ha="center", rotation=-10)
    arrow(ax, 5.8, 4.7, 5.3, 3.35, "λ = 1550 nm", PAL["light_in"], label_offset=(0.85, 0.1))
    arrow(ax, 2.2, y_wg + 0.22, 0.8, y_wg + 0.22, "guided", PAL["light_out"], label_offset=(0.0, 0.38))
    mat_badge(ax, 0.6, 5.75, "Si + SiN bilayer" if variant in ("bilayer", "antenna") else "SOI grating coupler", "si")


def draw_bragg_grating(ax, pcm: bool = False):
    """Draw a sidewall-corrugated waveguide Bragg grating as a single continuous polygon."""
    yc = 3.45
    w0 = 0.44
    dw = 0.18
    # Build continuous top and bottom corrugated edges so no black vertical seams cross the waveguide
    top_pts = [[0.7, yc + w0 / 2], [2.0, yc + w0 / 2]]
    bot_pts = [[0.7, yc - w0 / 2], [2.0, yc - w0 / 2]]
    n_p = 14
    pitch = 4.2 / n_p
    for i in range(n_p):
        x0 = 2.0 + i * pitch
        x1 = x0 + 0.5 * pitch
        x2 = x0 + pitch
        top_pts.extend([[x0, yc + w0 / 2 + dw], [x1, yc + w0 / 2 + dw], [x1, yc + w0 / 2], [x2, yc + w0 / 2]])
        bot_pts.extend([[x0, yc - w0 / 2 - dw], [x1, yc - w0 / 2 - dw], [x1, yc - w0 / 2], [x2, yc - w0 / 2]])
        if pcm:
            ax.add_patch(
                Rectangle((x0 + 0.02, yc - 0.15), 0.5 * pitch - 0.04, 0.30, facecolor=PAL["pcm"], edgecolor=PAL["pcm_edge"], lw=0.7, zorder=10)
            )
    top_pts.append([9.3, yc + w0 / 2])
    bot_pts.append([9.3, yc - w0 / 2])
    poly_pts = np.vstack([top_pts, bot_pts[::-1]])
    ax.add_patch(Polygon(poly_pts, closed=True, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.1, zorder=6))

    ax.annotate("", xy=(2.0, yc + w0 / 2 + dw + 0.28), xytext=(2.0 + 2 * pitch, yc + w0 / 2 + dw + 0.28), arrowprops=dict(arrowstyle="<->", color=PAL["text"], lw=1.0))
    ax.text(2.0 + pitch, yc + w0 / 2 + dw + 0.50, "2Λ (Bragg period Λ = 318 nm)", fontsize=7.3, fontweight="bold", color=PAL["text"], ha="center")
    arrow(ax, 0.5, yc + 0.10, 1.7, yc + 0.10, "in", PAL["light_in"], label_offset=(0.0, 0.36))
    arrow(ax, 1.7, yc - 0.10, 0.5, yc - 0.10, "R(λ_B)", PAL["light_out"], label_offset=(0.0, -0.38))
    arrow(ax, 8.2, yc, 9.4, yc, "T(λ)", PAL["light_out"], label_offset=(0.0, 0.36))
    mat_badge(ax, 0.6, 5.75, "Si + Sb2Se3 PCM" if pcm else "Sidewall-corrugated Si Bragg grating", "pcm" if pcm else "si")


def draw_edge_coupler(ax, variant: str = "inverse"):
    """Draw an inverse taper, strip-to-slot converter, plasmonic slot coupler, or bilayer SiN coupler."""
    yc = 3.45
    if variant in ("strip_to_slot", "plasmonic_slot"):
        # Wang et al. symmetric double-tip strip-to-slot converter leading into a narrow 100 nm slot waveguide
        ax.add_patch(
            Polygon(
                [[0.8, yc - 0.22], [2.4, yc - 0.22], [5.8, yc], [2.4, yc + 0.22], [0.8, yc + 0.22]],
                closed=True,
                facecolor=PAL["si"],
                edgecolor=PAL["si_edge"],
                lw=1.1,
            )
        )
        fc_rail = PAL["metal"] if variant == "plasmonic_slot" else PAL["si"]
        ec_rail = PAL["metal_edge"] if variant == "plasmonic_slot" else PAL["si_edge"]
        gap_half = 0.045  # Narrow 100 nm slot between the two rails
        rail_w = 0.22
        for sign in (+1.0, -1.0):
            pts = [
                [2.4, yc + sign * 0.42],
                [5.8, yc + sign * gap_half],
                [9.0, yc + sign * gap_half],
                [9.0, yc + sign * (gap_half + rail_w)],
                [5.8, yc + sign * (gap_half + rail_w)],
                [2.4, yc + sign * 0.48],
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=fc_rail, edgecolor=ec_rail, lw=1.0))
        # High-intensity E-field inside the narrow slot
        ax.add_patch(Rectangle((5.8, yc - gap_half), 3.2, 2 * gap_half, facecolor=PAL["light_in"], alpha=0.45, zorder=9))
        ax.text(7.4, yc + 0.55, "100 nm slot (high |E|)" if variant != "plasmonic_slot" else "50 nm Au MIM plasmonic slot", fontsize=7.4, fontweight="bold", color=PAL["light_in"], ha="center")
        mat_badge(ax, 0.6, 5.75, "Si-to-Au plasmonic coupler" if variant == "plasmonic_slot" else "Strip-to-slot converter", "metal" if variant == "plasmonic_slot" else "si")
    elif variant == "bilayer_sin":
        ax.add_patch(Polygon([[1.2, yc - 0.05], [8.8, yc - 0.36], [8.8, yc + 0.36], [1.2, yc + 0.05]], closed=True, facecolor=PAL["sin"], edgecolor=PAL["sin_edge"]))
        ax.add_patch(Polygon([[3.6, yc - 0.04], [8.8, yc - 0.22], [8.8, yc + 0.22], [3.6, yc + 0.04]], closed=True, facecolor=PAL["alox"], edgecolor=PAL["alox_edge"], alpha=0.88))
        mat_badge(ax, 0.6, 5.75, "Bilayer SiN / AlOx adiabatic taper", "sin")
    else:
        w_left = 0.06 if variant == "inverse" else 0.92
        w_right = 0.42 if variant == "inverse" else 0.26
        pts = [[1.4, yc - w_left], [8.8, yc - w_right], [8.8, yc + w_right], [1.4, yc + w_left]]
        ax.add_patch(Polygon(pts, closed=True, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2))
        if variant == "inverse":
            # Lensed fiber Gaussian beam waist at chip facet
            ax.add_patch(Ellipse((1.4, yc), 0.55, 1.35, facecolor=PAL["light_in"], alpha=0.25, zorder=8))
            ax.text(2.4, yc + 0.85, "190 nm tip at facet", fontsize=7.3, color=PAL["text"])
        mat_badge(ax, 0.6, 5.75, "Inverse Si nano-taper" if variant == "inverse" else "Adiabatic mode-size converter", "si")

    arrow(ax, 0.45, yc, 1.45, yc, "in", PAL["light_in"], label_offset=(0.0, 0.38))
    arrow(ax, 8.2, yc, 9.4, yc, "out", PAL["light_out"], label_offset=(0.0, 0.38))


def draw_polarization_device(ax, variant: str = "psr"):
    """Draw polarization splitter-rotator or metamaterial PBS with visible coupling gap."""
    yc = 3.35
    mat = "ln" if variant == "tfln" else "si"
    fc, ec = PAL[mat], PAL[f"{mat}_edge"]
    # Bi-level partial-etch slab flare for TM0 -> TE1 hybrid rotation
    ax.add_patch(Polygon([[1.2, yc + 0.16], [4.4, yc + 0.42], [4.4, yc - 0.18], [1.2, yc - 0.32]], closed=True, facecolor=PAL["sio2_edge"], edgecolor=ec, lw=0.8, alpha=0.75))
    ax.add_patch(Rectangle((0.8, yc - 0.18), 8.4, 0.32, facecolor=fc, edgecolor=ec, lw=1.0, zorder=6))
    # Asymmetric counter-tapered cross branch separated by a narrow gap (y_bot = yc + 0.14 + 0.05 = yc + 0.19)
    pts_br = [
        [3.6, yc + 0.20],
        [6.6, yc + 0.20],
        [8.6, yc + 1.10],
        [9.2, yc + 1.10],
        [9.2, yc + 1.38],
        [8.5, yc + 1.38],
        [6.5, yc + 0.46],
        [3.6, yc + 0.34],
    ]
    ax.add_patch(Polygon(pts_br, closed=True, facecolor=fc, edgecolor=ec, lw=1.0, zorder=7))

    if variant in ("swg_pb", "mm_pbs", "pso_pbs"):
        for xp in np.linspace(3.7, 6.4, 11):
            ax.add_patch(Rectangle((xp - 0.06, yc - 0.52), 0.12, 1.12, facecolor=PAL["sin"], edgecolor=PAL["sin_edge"], alpha=0.85, zorder=5))
        mat_badge(ax, 0.6, 5.75, "Anisotropic SWG metamaterial PBS", "sin")
    else:
        mat_badge(ax, 0.6, 5.75, "TFLN PSR" if mat == "ln" else "Bi-level Si taper + coupler PSR", mat)

    arrow(ax, 0.5, yc - 0.02, 1.6, yc - 0.02, "TE0+TM0", PAL["light_in"], label_offset=(0.1, 0.42))
    arrow(ax, 8.2, yc - 0.02, 9.4, yc - 0.02, "TE0", PAL["light_out"], label_offset=(0.0, -0.35))
    arrow(ax, 8.2, yc + 1.24, 9.4, yc + 1.24, "TM0→TE0", PAL["light_out"], label_offset=(-0.2, 0.35))


def draw_thermo_optic_xsec(ax, variant: str = "metal_heater"):
    """Draw a 2D cross-section of a thermally tuned SOI waveguide or CPO stack."""
    # Si substrate + SiO2 BOX + cladding
    ax.add_patch(Rectangle((0.8, 1.0), 8.4, 0.85, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
    ax.text(5.0, 1.4, "Si substrate (heat sink T0 = 300 K)", fontsize=7.2, color="#ffffff", ha="center", va="center")
    ax.add_patch(Rectangle((0.8, 1.85), 8.4, 3.4, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))

    # Thermal plume contours around heater/core
    xg = np.linspace(1.2, 8.8, 60)
    yg = np.linspace(1.9, 5.1, 40)
    XX, YY = np.meshgrid(xg, yg)
    y_src = 4.4 if variant != "doped" else 2.85
    temp = np.exp(-((XX - 5.0) / 1.8) ** 2 - ((YY - y_src) / 1.1) ** 2)
    ax.contourf(XX, YY, temp, levels=5, cmap="OrRd", alpha=0.38, zorder=4)

    # Si waveguide core
    ax.add_patch(Rectangle((4.35, 2.65), 1.30, 0.50, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2, zorder=8))
    ax.text(5.0, 2.90, "Si", fontsize=8, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=9)

    if variant == "doped":
        # Side N++ doped Si heaters integrated in slab
        ax.add_patch(Rectangle((2.2, 2.65), 2.15, 0.18, facecolor=PAL["si"], edgecolor=PAL["si_edge"], zorder=7))
        ax.add_patch(Rectangle((5.65, 2.65), 2.15, 0.18, facecolor=PAL["si"], edgecolor=PAL["si_edge"], zorder=7))
        ax.add_patch(Rectangle((2.2, 2.65), 1.1, 0.50, facecolor=PAL["p_dop"], edgecolor=PAL["si_edge"], zorder=8))
        ax.add_patch(Rectangle((6.7, 2.65), 1.1, 0.50, facecolor=PAL["p_dop"], edgecolor=PAL["si_edge"], zorder=8))
        ax.text(2.75, 2.90, "N++", fontsize=7.2, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=9)
        ax.text(7.25, 2.90, "N++", fontsize=7.2, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=9)
    else:
        # Top TiN / Au metal heater strip above oxide spacer
        ax.add_patch(Rectangle((4.1, 4.25), 1.80, 0.35, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"], lw=1.2, zorder=10))
        ax.text(5.0, 4.42, "TiN Heater (P_heat)", fontsize=7.2, fontweight="bold", color=PAL["text"], ha="center", va="center", zorder=11)

    ax.text(7.6, 4.3, "Δn = (dn/dT)·ΔT(x,y)", fontsize=7.6, fontweight="bold", color=PAL["pcm_edge"], ha="center")
    mat_badge(ax, 0.6, 5.75, "Electro-thermal + optical mode", "pcm")


def draw_pin_eo_xsec(ax, variant: str = "pin"):
    """Draw a 2D cross-section of a PIN carrier modulator, photodiode, TFLN EO, or SOH slot."""
    ax.add_patch(Rectangle((0.8, 1.0), 8.4, 1.4, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
    ax.text(5.0, 1.65, "SiO2 buried oxide (BOX)", fontsize=7.2, color=PAL["muted"], ha="center", va="center")

    if variant == "tfln":
        # TFLN ridge + Au G-S-G electrodes
        ax.add_patch(Rectangle((1.2, 2.4), 7.6, 0.35, facecolor=PAL["ln"], edgecolor=PAL["ln_edge"]))
        ax.add_patch(Polygon([[4.1, 2.75], [4.3, 3.35], [5.7, 3.35], [5.9, 2.75]], closed=True, facecolor=PAL["ln"], edgecolor=PAL["ln_edge"]))
        for x_el, lbl in [(1.6, "G"), (6.5, "S (+V)")]:
            ax.add_patch(Rectangle((x_el, 2.75), 1.6, 0.65, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"], lw=1))
            ax.text(x_el + 0.8, 3.07, lbl, fontsize=8, fontweight="bold", color=PAL["text"], ha="center", va="center")
        ax.text(5.0, 4.2, "Pockels effect: Δn = -½ n³ r33 E_z", fontsize=7.8, fontweight="bold", color=PAL["ln"], ha="center")
        mat_badge(ax, 0.6, 5.75, "x-cut TFLN + Au RF electrodes", "ln")
    elif variant == "soh":
        # Silicon-organic hybrid slot waveguide with EO polymer fill
        ax.add_patch(Rectangle((2.5, 2.4), 5.0, 0.9, facecolor=PAL["poly"], edgecolor=PAL["poly_edge"], alpha=0.35))
        ax.add_patch(Rectangle((3.9, 2.4), 0.85, 0.65, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        ax.add_patch(Rectangle((5.25, 2.4), 0.85, 0.65, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        ax.add_patch(Rectangle((1.2, 2.4), 2.7, 0.18, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        ax.add_patch(Rectangle((6.1, 2.4), 2.7, 0.18, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        ax.add_patch(Rectangle((1.3, 2.58), 1.3, 0.55, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        ax.add_patch(Rectangle((7.4, 2.58), 1.3, 0.55, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        ax.text(5.0, 3.65, "EO organic cladding in 120 nm Si slot", fontsize=7.6, fontweight="bold", color=PAL["poly"], ha="center")
        mat_badge(ax, 0.6, 5.75, "SOH slot modulator", "poly")
    else:
        # Rib waveguide with P+ / intrinsic Si / N+ lateral junction + metal contacts
        ax.add_patch(Rectangle((1.2, 2.4), 2.9, 0.25, facecolor=PAL["p_dop"], edgecolor=PAL["si_edge"]))
        ax.add_patch(Rectangle((5.9, 2.4), 2.9, 0.25, facecolor=PAL["n_dop"], edgecolor=PAL["si_edge"]))
        ax.add_patch(Rectangle((4.1, 2.4), 1.8, 0.85, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2))
        ax.text(2.8, 2.52, "P+", fontsize=7.5, fontweight="bold", color="#ffffff", ha="center", va="center")
        ax.text(5.0, 2.82, "i-Si" if variant != "pd" else "Ge / i-Si", fontsize=8, fontweight="bold", color="#ffffff", ha="center", va="center")
        ax.text(7.2, 2.52, "N+", fontsize=7.5, fontweight="bold", color="#ffffff", ha="center", va="center")
        # Metal plugs
        ax.add_patch(Rectangle((1.4, 2.65), 1.1, 0.85, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        ax.add_patch(Rectangle((7.5, 2.65), 1.1, 0.85, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        ax.text(1.95, 3.07, "Anode", fontsize=7.0, fontweight="bold", ha="center", va="center")
        ax.text(8.05, 3.07, "Cathode", fontsize=7.0, fontweight="bold", ha="center", va="center")
        ax.text(5.0, 4.1, "Free-carrier plasma dispersion Δn(ΔN, ΔP)" if variant != "pd" else "Small-signal AC admittance Y(ω) = G + iωC", fontsize=7.5, color=PAL["text"], ha="center")
        mat_badge(ax, 0.6, 5.75, "Si PIN carrier junction", "si")


def draw_metasurface(ax, variant: str = "pillars"):
    """Draw periodic metasurface unit cells or pillar arrays."""
    if variant in ("vortex", "chiral", "bic", "fano", "spin"):
        # Top view of 2D unit-cell array with broken-symmetry or chiral resonators
        gx = np.linspace(1.8, 8.2, 5)
        gy = np.linspace(1.6, 4.8, 3)
        for ix, x0 in enumerate(gx):
            for iy, y0 in enumerate(gy):
                ax.add_patch(Rectangle((x0 - 0.58, y0 - 0.48), 1.16, 0.96, facecolor="#ffffff", edgecolor="#cbd5e1", lw=0.7))
                if variant == "vortex":
                    ang = (ix * 36 + iy * 45) % 180
                    ell = Ellipse((x0, y0), 0.72, 0.26, angle=ang, facecolor=PAL["si"], edgecolor=PAL["si_edge"])
                    ax.add_patch(ell)
                elif variant == "bic":
                    # Tilted ellipse pair for quasi-BIC
                    for dx, tilt in [(-0.22, 16), (0.22, -16)]:
                        ell = Ellipse((x0 + dx, y0), 0.24, 0.62, angle=tilt, facecolor=PAL["si"], edgecolor=PAL["si_edge"])
                        ax.add_patch(ell)
                elif variant == "chiral":
                    # Z-shaped PCM chiral resonator
                    pts = [[x0 - 0.3, y0 + 0.25], [x0 + 0.1, y0 + 0.25], [x0 + 0.1, y0 - 0.25], [x0 + 0.3, y0 - 0.25]]
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PAL["pcm"], lw=4.5, solid_capstyle="round")
                elif variant == "fano":
                    # Asymmetric split cube + bar
                    ax.add_patch(Rectangle((x0 - 0.34, y0 - 0.30), 0.26, 0.60, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
                    ax.add_patch(Rectangle((x0 + 0.06, y0 - 0.18), 0.26, 0.36, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
                else:
                    # DBS binary barcode pixels
                    rng = np.random.default_rng(ix * 7 + iy + 3)
                    for px in range(3):
                        for py in range(3):
                            if rng.random() > 0.4:
                                ax.add_patch(Rectangle((x0 - 0.36 + px * 0.24, y0 - 0.30 + py * 0.20), 0.22, 0.18, facecolor=PAL["si"]))
        mat_badge(ax, 0.6, 5.75, "Periodic Floquet unit cells", "pcm" if variant == "chiral" else "si")
        return

    # Side view of substrate + pillars / multi-layer stack
    ax.add_patch(Rectangle((0.8, 1.1), 8.4, 0.85, facecolor=PAL["metal" if variant in ("mim", "gradient") else "sio2"], edgecolor=PAL["si_edge"]))
    if variant == "mim":
        ax.add_patch(Rectangle((0.8, 1.95), 8.4, 0.32, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
    widths = [0.22, 0.34, 0.46, 0.58, 0.72, 0.84] if variant == "gradient" else [0.52] * 6
    fc = PAL["metal"] if variant in ("mim", "nanorod") else (PAL["sin"] if variant == "color" else PAL["si"])
    ec = PAL["metal_edge"] if variant in ("mim", "nanorod") else PAL["si_edge"]
    y_base = 2.27 if variant == "mim" else 1.95
    for i, w_p in enumerate(widths):
        xp = 1.6 + i * 1.32
        ax.add_patch(Rectangle((xp - w_p / 2, y_base), w_p, 1.15, facecolor=fc, edgecolor=ec, lw=1))
    arrow(ax, 3.2, 5.2, 3.2, 3.6, "k_inc", PAL["light_in"], label_offset=(-0.55, 0.0))
    if variant == "gradient":
        arrow(ax, 5.5, 3.5, 7.2, 5.1, "anomalous θ_r", PAL["light_out"], label_offset=(0.75, -0.1))
    else:
        arrow(ax, 6.2, 3.6, 6.2, 5.2, "R(λ), T(λ)", PAL["light_out"], label_offset=(0.75, 0.0))
    mat_badge(ax, 0.6, 5.75, "MIM plasmonic" if variant == "mim" else "Dielectric meta-atoms", "metal" if variant == "mim" else "si")


def draw_phc_lattice(ax, variant: str = "l3"):
    """Draw a 2D photonic-crystal air-hole slab with cavity, waveguide, or disordered defect."""
    if variant == "anderson":
        ax.add_patch(Rectangle((1.2, 1.05), 7.6, 4.15, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"], lw=1.2))
        rng = np.random.default_rng(42)
        for _ in range(65):
            xr = rng.uniform(1.55, 8.45)
            yr = rng.uniform(1.35, 4.90)
            ax.add_patch(Circle((xr, yr), 0.14, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=0.7, zorder=6))
        ax.add_patch(Ellipse((4.8, 3.1), 1.9, 1.2, facecolor=PAL["light_in"], alpha=0.28, zorder=7))
        ax.text(5.0, 5.42, "Localized mode in 2D disordered rod array", fontsize=7.5, fontweight="bold", color=PAL["text"], ha="center")
        mat_badge(ax, 0.6, 5.75, "Disordered Si rods (Anderson)", "si")
        return

    ax.add_patch(Rectangle((1.2, 1.05), 7.6, 4.15, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2))
    nx, ny = 11, 7
    xs = np.linspace(1.75, 8.25, nx)
    ys = np.linspace(1.45, 4.80, ny)
    for j, y0 in enumerate(ys):
        x_shift = 0.28 if (j % 2 == 1) else 0.0
        for i, x0 in enumerate(xs):
            xp = x0 + x_shift - 0.14
            if variant == "l3" and j == 3 and i in (4, 5, 6):
                # Draw subtle dashed rings showing the 3 missing holes of the L3 defect
                ax.add_patch(Circle((xp, y0), 0.18, facecolor="none", edgecolor="#93c5fd", lw=0.9, ls="--", zorder=7))
                continue
            if variant == "l3" and j == 3 and i in (3, 7):
                # Outward-shifted end holes (Δx = ±0.15a) for high-Q optimization
                dx_opt = -0.14 if i == 3 else 0.14
                ax.add_patch(Circle((xp + dx_opt, y0), 0.19, facecolor=PAL["sio2"], edgecolor=PAL["light_in"], lw=1.2, zorder=8))
                continue
            if variant == "wg" and j == 3:
                continue
            if variant == "bistable" and j == 3 and i == 5:
                ax.add_patch(Circle((xp, y0), 0.14, facecolor=PAL["pcm"], edgecolor=PAL["pcm_edge"], zorder=9))
                continue
            r_h = 0.19 if not (variant == "topo" and j >= 3) else 0.12
            fc_h = PAL["sio2"] if not (variant == "topo" and j == 3) else "#fde047"
            ax.add_patch(Circle((xp, y0), r_h, facecolor=fc_h, edgecolor=PAL["sio2_edge"], lw=0.8, zorder=6))

    if variant == "l3":
        # Label placed outside the lattice so the 3 missing holes in the center row remain completely visible
        ax.text(5.0, 5.42, "L3 defect: 3 missing holes + shifted end holes (Δx = 0.15a)", fontsize=7.4, fontweight="bold", color=PAL["light_in"], ha="center")
    elif variant == "wg":
        arrow(ax, 0.55, ys[3], 1.65, ys[3], "in", PAL["light_in"], label_offset=(0.0, 0.35))
        arrow(ax, 8.35, ys[3], 9.45, ys[3], "out", PAL["light_out"], label_offset=(0.0, 0.35))
    elif variant == "topo":
        arrow(ax, 1.5, ys[3], 8.5, ys[3], "", PAL["light_in"], lw=1.8)
        ax.text(5.0, 5.42, "Valley-Hall topological domain-wall interface", fontsize=7.4, fontweight="bold", color=PAL["text"], ha="center")
    else:
        ax.text(5.0, 5.42, "Kerr nonlinear defect cavity (bistable transmission)", fontsize=7.4, fontweight="bold", color=PAL["pcm_edge"], ha="center")
    mat_badge(ax, 0.6, 5.75, "2D PhC slab (a = 420 nm)", "si")


def draw_nanobeam(ax):
    """Draw a 1D photonic-crystal nanobeam cavity with tapered hole radii toward center."""
    yc = 3.45
    ax.add_patch(Rectangle((0.8, yc - 0.42), 8.4, 0.84, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2))
    # Holes taper down in radius and pitch toward the central defect cavity
    xs = [1.35, 2.00, 2.62, 3.20, 3.74, 4.24, 5.76, 6.26, 6.80, 7.38, 8.00, 8.65]
    for i, xp in enumerate(xs):
        dist = abs(xp - 5.0) / 3.65
        rx = 0.11 + 0.09 * dist
        ry = 0.16 + 0.11 * dist
        ax.add_patch(Ellipse((xp, yc), 2 * rx, 2 * ry, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"], lw=0.9, zorder=7))
    ax.add_patch(Ellipse((5.0, yc), 1.05, 0.55, facecolor=PAL["light_in"], alpha=0.30, zorder=8))
    ax.text(5.0, yc + 0.68, "Tapered mirror → λ/2 cavity defect", fontsize=7.4, fontweight="bold", color=PAL["text"], ha="center")
    arrow(ax, 0.45, yc, 1.3, yc, "in", PAL["light_in"], label_offset=(0.0, 0.55))
    arrow(ax, 8.7, yc, 9.5, yc, "out", PAL["light_out"], label_offset=(0.0, 0.55))
    mat_badge(ax, 0.6, 5.75, "1D Si PhC nanobeam cavity", "si")


def draw_band_diagram(ax, lattice: str = "square"):
    """Draw a photonic band structure ωa/(2πc) vs Bloch wavevector k."""
    ax.add_patch(Rectangle((1.2, 1.25), 5.6, 4.1, facecolor="#ffffff", edgecolor="#94a3b8", lw=1.0))
    k = np.linspace(0.0, 3.0, 120)
    b1 = 0.29 * np.abs(np.sin(np.pi * k / 2.0))
    b2 = 0.54 - 0.06 * np.cos(np.pi * k)
    b3 = 0.67 + 0.06 * np.cos(np.pi * k * 0.8)
    ax.fill_between(1.2 + k * (5.6 / 3.0), 1.25 + 0.32 * 5.2, 1.25 + 0.45 * 5.2, color="#fef08a", alpha=0.65, zorder=3)
    ax.text(4.0, 3.25, "Photonic Band Gap", fontsize=7.5, fontweight="bold", color="#854d0e", ha="center", va="center", zorder=6)
    for b_curve, col in [(b1, PAL["light_out"]), (b2, PAL["si"]), (b3, PAL["light_in"])]:
        ax.plot(1.2 + k * (5.6 / 3.0), 1.25 + b_curve * 5.2, color=col, lw=2.0, zorder=5)

    for xk, lbl in [(1.2, "Γ"), (3.07, "M"), (4.93, "K" if lattice == "hex" else "X"), (6.8, "Γ")]:
        ax.plot([xk, xk], [1.25, 5.35], "--", color="#cbd5e1", lw=0.8)
        ax.text(xk, 0.98, lbl, fontsize=8.0, fontweight="bold", color=PAL["text"], ha="center")

    cx, cy = 8.25, 3.4
    n_sides = 6 if lattice == "hex" else 4
    angles = np.linspace(0, 2 * np.pi, n_sides + 1)[:-1] + (np.pi / 4 if n_sides == 4 else 0)
    bz_pts = np.column_stack([cx + 0.85 * np.cos(angles), cy + 0.85 * np.sin(angles)])
    ax.add_patch(Polygon(bz_pts, closed=True, facecolor=PAL["sio2"], edgecolor=PAL["si"], lw=1.2))
    ax.text(cx, cy - 1.25, "Brillouin zone", fontsize=7.2, color=PAL["muted"], ha="center")
    mat_badge(ax, 0.6, 5.75, "Bloch eigenmodes ω(k)", "si")


def draw_lens(ax, kind: str = "luneburg"):
    """Draw a Luneburg GRIN lens, metalens, Fresnel lens, or zone plate."""
    yc = 3.50
    for y_ray in np.linspace(2.1, 4.9, 5):
        arrow(ax, 0.7, y_ray, 2.3, y_ray, "", PAL["light_in"], lw=1.4)

    if kind == "luneburg":
        r_l = 1.65
        xc = 4.5
        for frac, alpha in [(1.0, 0.25), (0.78, 0.40), (0.55, 0.60), (0.32, 0.85)]:
            ax.add_patch(Circle((xc, yc), r_l * frac, facecolor=PAL["sin"], edgecolor="none", alpha=alpha, zorder=5))
        ax.add_patch(Circle((xc, yc), r_l, facecolor="none", edgecolor=PAL["sin_edge"], lw=1.2, zorder=6))
        fx = xc + r_l
        for y_ray in np.linspace(2.1, 4.9, 5):
            ax.plot([2.85, fx], [y_ray, yc], color=PAL["light_out"], lw=1.3, zorder=8)
        ax.add_patch(Circle((fx, yc), 0.16, color=PAL["light_in"], zorder=12))
        ax.text(xc, yc - r_l - 0.35, "n(r) = √(2 - (r/R)²)", fontsize=7.6, fontweight="bold", color=PAL["text"], ha="center")
        mat_badge(ax, 0.6, 5.75, "2D GRIN Luneburg lens", "sin")
    else:
        xc = 3.2
        fx = 7.8
        ax.add_patch(Rectangle((2.5, 1.3), 0.7, 4.4, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
        for idx, yp in enumerate(np.linspace(1.5, 5.5, 13)):
            r_dist = abs(yp - yc)
            h_p = 0.42 if kind != "fresnel" else (0.15 + 0.35 * ((r_dist * 1.4) % 1.0))
            w_p = 0.10 + 0.18 * (1.0 - r_dist / 2.5)
            if kind == "zone_plate" and idx % 2 == 1:
                continue
            fc = PAL["metal"] if kind == "zone_plate" else (PAL["alox"] if kind == "visible" else PAL["si"])
            ax.add_patch(Rectangle((3.2, yp - w_p / 2), h_p, w_p, facecolor=fc, edgecolor=PAL["si_edge"], zorder=7))
        for y_ray in np.linspace(2.1, 4.9, 5):
            ax.plot([3.5, fx], [y_ray, yc], "--", color=PAL["light_out"], lw=1.2, zorder=6)
        ax.add_patch(Ellipse((fx, yc), 0.35, 0.75, facecolor=PAL["light_in"], alpha=0.45, zorder=10))
        ax.text(fx, yc - 0.7, "Focal spot", fontsize=7.5, fontweight="bold", color=PAL["light_in"], ha="center")
        mat_badge(ax, 0.6, 5.75, "TiO2 / Si metalens" if kind != "zone_plate" else "Au zone plate", "alox" if kind == "visible" else "si")


def draw_scattering(ax, kind: str = "multipole"):
    """Draw scattering, RCS, nanoantenna, or plasmonic nanoparticle cases."""
    xc, yc = 4.2, 3.45
    arrow(ax, 0.7, yc, 2.2, yc, "E_inc, k", PAL["light_in"], label_offset=(0.0, 0.38))

    if kind == "yagi":
        ax.add_patch(Rectangle((2.7, yc - 0.85), 0.24, 1.70, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        ax.add_patch(Rectangle((3.6, yc - 0.68), 0.24, 0.60, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        ax.add_patch(Rectangle((3.6, yc + 0.08), 0.24, 0.60, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        for xd in [4.5, 5.3, 6.1]:
            ax.add_patch(Rectangle((xd, yc - 0.52), 0.22, 1.04, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        theta = np.linspace(-np.pi / 2, np.pi / 2, 60)
        r_lobe = 2.2 * np.cos(theta) ** 4
        ax.plot(6.4 + r_lobe * np.cos(theta), yc + r_lobe * np.sin(theta), color=PAL["light_out"], lw=1.8)
        mat_badge(ax, 0.6, 5.75, "Au Yagi-Uda nanoantenna", "metal")
        return

    if kind == "shapes":
        ax.add_patch(Circle((3.2, yc), 0.65, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        ax.add_patch(Rectangle((4.6, yc - 0.6), 1.2, 1.2, facecolor=PAL["sin"], edgecolor=PAL["sin_edge"]))
        ax.add_patch(Polygon([[6.6, yc - 0.65], [7.9, yc - 0.65], [7.25, yc + 0.65]], closed=True, facecolor=PAL["ln"], edgecolor=PAL["ln_edge"]))
        mat_badge(ax, 0.6, 5.75, "Cylinder / Square / Triangle", "si")
        return

    fc = PAL["metal"] if kind in ("plasmonic", "photothermal", "pec") else PAL["si"]
    ec = PAL["metal_edge"] if kind in ("plasmonic", "photothermal", "pec") else PAL["si_edge"]
    ax.add_patch(Circle((xc, yc), 0.85, facecolor=fc, edgecolor=ec, lw=1.3, zorder=8))
    theta = np.linspace(0, 2 * np.pi, 100)
    r_pat = 1.45 * (1.0 + 0.55 * np.cos(theta))
    ax.plot(xc + r_pat * np.cos(theta), yc + r_pat * np.sin(theta), "--", color=PAL["light_out"], lw=1.4)
    if kind == "multipole":
        ax.text(7.3, 4.3, "ED + MD + EQ + MQ\nmultipole decomposition", fontsize=7.5, color=PAL["text"], ha="center")
    elif kind == "kerker":
        ax.text(7.4, 4.2, "Kerker condition:\nED = MD (zero back-scatter)", fontsize=7.5, color=PAL["text"], ha="center")
    elif kind == "photothermal":
        ax.add_patch(Circle((xc, yc), 1.35, facecolor=PAL["pcm"], alpha=0.28, zorder=4))
        ax.text(7.4, 4.2, "Q_abs → Joule heating\nΔT(r) around Au NP", fontsize=7.5, color=PAL["pcm_edge"], ha="center")
    elif kind == "mst":
        arrow(ax, xc + 0.9, yc, xc + 2.3, yc, "F_opt = ∮ T_ij n_j dS", PAL["pcm"], label_offset=(0.5, 0.35))
        ax.add_patch(Rectangle((xc - 1.15, yc - 1.15), 2.3, 2.3, facecolor="none", edgecolor=PAL["pcm"], lw=1.2, ls="--"))
    mat_badge(ax, 0.6, 5.75, "Au / Ag / PEC" if fc == PAL["metal"] else "High-index Si scatterer", "metal" if fc == PAL["metal"] else "si")


def draw_slit_or_grating(ax, kind: str = "single_slit"):
    """Draw single slit, double slit, diffraction grating, hexagonal or LC grating."""
    yc = 3.55
    arrow(ax, 0.6, yc, 2.2, yc, "Plane wave", PAL["light_in"], label_offset=(0.0, 0.42))
    if kind in ("single_slit", "double_slit"):
        ax.add_patch(Rectangle((2.8, 0.9), 0.28, 5.2, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        gaps = [yc] if kind == "single_slit" else [yc - 0.85, yc + 0.85]
        for yg in gaps:
            ax.add_patch(Rectangle((2.75, yg - 0.28), 0.38, 0.56, facecolor="#eff6ff", edgecolor="none", zorder=5))
            for r_w in [0.9, 1.8, 2.7, 3.6]:
                arc = Arc((3.08, yg), 2 * r_w, 2 * r_w, theta1=-55, theta2=55, color=PAL["light_out"], lw=1.2)
                ax.add_patch(arc)
        mat_badge(ax, 0.6, 5.75, "PEC screen + aperture diffraction", "metal")
    else:
        for yp in np.linspace(1.5, 5.3, 6):
            ax.add_patch(Rectangle((2.8, yp - 0.22), 0.55, 0.44, facecolor=PAL["si" if kind != "lc" else "ln"], edgecolor=PAL["si_edge"]))
        arrow(ax, 3.6, yc, 6.8, yc, "m = 0", PAL["light_out"], label_offset=(0.2, 0.28))
        arrow(ax, 3.6, yc + 0.2, 6.4, yc + 1.55, "m = +1", PAL["light_out"], label_offset=(-0.2, 0.35))
        arrow(ax, 3.6, yc - 0.2, 6.4, yc - 1.55, "m = -1", PAL["light_out"], label_offset=(-0.2, -0.38))
        mat_badge(ax, 0.6, 5.75, "Floquet diffraction orders", "ln" if kind == "lc" else "si")


def draw_dipole_or_led(ax, kind: str = "ed_pec"):
    """Draw dipole over PEC ground plane or LED / micro-LED structure."""
    if kind in ("ed_pec", "md_pec"):
        ax.add_patch(Rectangle((1.0, 1.65), 8.0, 0.24, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        ax.text(8.1, 1.40, "PEC ground (z = 0)", fontsize=7.2, color=PAL["metal_edge"], ha="center")
        xc, yc = 5.0, 2.55
        ax.add_patch(Circle((xc, yc), 0.12, color=PAL["light_in"], zorder=12))
        arrow(ax, xc - 0.55, yc, xc + 0.55, yc, "p (ED)" if kind == "ed_pec" else "m (MD)", PAL["light_in"], label_offset=(0.0, 0.34))
        theta = np.linspace(0, np.pi, 80)
        r_lobe = 2.65 * np.sin(theta) * (1.0 + 0.32 * np.cos(2 * theta))
        ax.plot(xc + r_lobe * np.cos(theta), 1.89 + r_lobe * np.sin(theta), color=PAL["light_out"], lw=1.8)
        mat_badge(ax, 0.6, 5.75, "Method of images + Near2Far", "metal")
    else:
        ax.add_patch(Rectangle((1.2, 1.1), 7.6, 0.7, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        pts_mesa = [[2.0, 1.8], [8.0, 1.8], [7.1 if kind == "uled" else 8.0, 3.6], [2.9 if kind == "uled" else 2.0, 3.6]]
        ax.add_patch(Polygon(pts_mesa, closed=True, facecolor=PAL["sin"], edgecolor=PAL["sin_edge"], lw=1.2))
        ax.add_patch(Rectangle((2.7, 2.55), 4.6, 0.22, facecolor=PAL["pcm"], edgecolor="none", zorder=8))
        ax.text(5.0, 2.66, "InGaN MQW active dipole sheet", fontsize=7.0, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=9)
        for xr in [3.8, 5.0, 6.2]:
            arrow(ax, xr, 2.8, xr + 0.25 * (xr - 5.0), 5.1, "", PAL["light_out"], lw=1.6)
        mat_badge(ax, 0.6, 5.75, "GaN / InGaN LED extraction", "sin")


def draw_hollow_core_fiber(ax):
    """Draw a 2D cross-section of an anti-resonant nodeless hollow-core fiber."""
    xc, yc = 5.0, 3.30
    r_jacket = 1.92
    ax.add_patch(Wedge((xc, yc), r_jacket, 0, 360, width=0.28, facecolor=PAL["sio2_edge"], edgecolor=PAL["si"], lw=1.2))
    r_cap = 0.48
    n_tubes = 7
    for k in range(n_tubes):
        ang = 2.0 * np.pi * k / n_tubes
        xt = xc + (r_jacket - 0.28 - r_cap) * np.cos(ang)
        yt = yc + (r_jacket - 0.28 - r_cap) * np.sin(ang)
        ax.add_patch(Wedge((xt, yt), r_cap, 0, 360, width=0.06, facecolor=PAL["sio2_edge"], edgecolor=PAL["si"], lw=1.0, zorder=8))
    for r_m, alpha in [(0.66, 0.22), (0.44, 0.40), (0.22, 0.65)]:
        ax.add_patch(Circle((xc, yc), r_m, facecolor=PAL["light_in"], alpha=alpha, zorder=6))
    ax.text(xc, yc, "HE11\nair core", fontsize=7.5, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=10)
    mat_badge(ax, 0.6, 5.75, "SiO2 capillaries (7-tube AR-HCF)", "sio2")


def draw_special_device(ax, variant: str):
    """Draw specialized physical sketches for distinct devices (T07, T22, T30, T32, T38, T39, T42, T59, T60, T62, T72, T73, T88, T96)."""
    yc = 3.35
    if variant == "neff2d":
        # T07: 3D slab cross-section mapped to 2D effective index n_eff(x, y)
        ax.add_patch(Rectangle((0.8, 1.5), 3.6, 1.0, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
        ax.add_patch(Rectangle((1.9, 2.5), 1.4, 0.65, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        ax.text(2.6, 4.1, "3D slab ε(x,y,z)\nmode profile M(z)", fontsize=7.5, fontweight="bold", ha="center", color=PAL["text"])
        arrow(ax, 4.6, 3.0, 5.6, 3.0, "∫|M(z)|²dz", PAL["light_in"], label_offset=(0.0, 0.45))
        ax.add_patch(Rectangle((5.9, 1.6), 3.2, 2.8, facecolor=PAL["sio2"], edgecolor=PAL["si_edge"]))
        ax.add_patch(Circle((7.5, 3.0), 0.95, facecolor="none", edgecolor=PAL["si"], lw=5.0))
        ax.text(7.5, 3.0, "2D ε_eff(x,y,ω)", fontsize=7.3, fontweight="bold", ha="center", va="center", color=PAL["text"])
        mat_badge(ax, 0.6, 5.75, "Variational 3D → 2D n_eff reduction", "si")
    elif variant == "demux8":
        # T22: 8-channel mode & polarization demultiplexer with cascaded asymmetric couplers onto central bus
        ax.add_patch(Rectangle((0.8, yc - 0.36), 8.4, 0.72, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        ax.text(5.0, yc, "Multimode bus (TE0–TE3, TM0–TM3)", fontsize=7.2, fontweight="bold", color="#ffffff", ha="center", va="center")
        for idx, (xp, sign, lbl) in enumerate([(2.0, 1, "TE1"), (4.2, 1, "TE2"), (6.4, 1, "TE3"), (3.1, -1, "TM1"), (5.5, -1, "TM2")]):
            y_tap = yc + sign * 0.43
            pts = [[xp, y_tap], [xp + 1.1, y_tap], [xp + 1.7, y_tap + sign * 0.95], [xp + 1.4, y_tap + sign * 0.95], [xp + 0.9, y_tap + sign * 0.18], [xp, y_tap + sign * 0.18]]
            ax.add_patch(Polygon(pts, closed=True, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
            ax.text(xp + 1.55, y_tap + sign * 1.15, lbl, fontsize=7.0, fontweight="bold", color=PAL["light_out"], ha="center")
        arrow(ax, 8.2, yc, 9.3, yc, "TE0", PAL["light_out"], label_offset=(0.0, 0.45))
        mat_badge(ax, 0.6, 5.75, "Cascaded asymmetric DCs (8-ch)", "si")
    elif variant == "co2_sensor":
        # T30: Mid-IR plasmonic slot waveguide CO2 gas sensor
        ax.add_patch(Rectangle((1.0, 1.3), 8.0, 1.1, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
        ax.add_patch(Rectangle((1.5, 2.4), 2.8, 1.2, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        ax.add_patch(Rectangle((5.7, 2.4), 2.8, 1.2, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        ax.add_patch(Rectangle((4.3, 2.4), 1.4, 1.2, facecolor=PAL["light_in"], alpha=0.35))
        for gx, gy in [(4.6, 2.8), (5.0, 3.3), (5.3, 2.7), (4.8, 3.9), (5.4, 4.1), (4.4, 4.2)]:
            ax.add_patch(Circle((gx, gy), 0.09, facecolor=PAL["pcm"], edgecolor=PAL["pcm_edge"]))
        ax.text(5.0, 4.75, "CO2 gas absorption band (λ = 4.26 µm)", fontsize=7.6, fontweight="bold", color=PAL["pcm_edge"], ha="center")
        mat_badge(ax, 0.6, 5.75, "Au plasmonic slot gas sensor", "metal")
    elif variant == "scale_inv":
        # T32: Scale-invariant effective-index waveguide
        ax.add_patch(Rectangle((1.0, 1.3), 8.0, 1.0, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
        for xc, w_c, lbl in [(3.0, 0.7, "Narrow w1"), (6.8, 1.8, "Wide w2")]:
            ax.add_patch(Rectangle((xc - w_c / 2, 2.3), w_c, 0.8, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
            ax.add_patch(Rectangle((xc - w_c / 2, 3.1), w_c, 0.35, facecolor=PAL["sin"], edgecolor=PAL["sin_edge"]))
            ax.text(xc, 4.0, f"{lbl}\nn_eff = const", fontsize=7.5, fontweight="bold", color=PAL["text"], ha="center")
        mat_badge(ax, 0.6, 5.75, "Scale-invariant multi-layer core", "si")
    elif variant == "vivaldi":
        # T38: Integrated plasmonic Vivaldi exponentially flared slot horn antenna
        ax.add_patch(Rectangle((0.8, yc - 0.18), 2.4, 0.36, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        x = np.linspace(2.6, 6.8, 80)
        y_flare = 0.06 * np.exp(0.82 * (x - 2.6))
        for sign in (+1.0, -1.0):
            pts = np.vstack([np.column_stack([x, yc + sign * y_flare]), [[6.8, yc + sign * 2.1], [2.6, yc + sign * 2.1]]])
            ax.add_patch(Polygon(pts, closed=True, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"], lw=1.1))
        for r_w in [0.9, 1.6, 2.3]:
            ax.add_patch(Arc((6.6, yc), 2 * r_w, 2 * r_w, theta1=-45, theta2=45, color=PAL["light_out"], lw=1.5))
        arrow(ax, 0.5, yc, 1.7, yc, "in", PAL["light_in"], label_offset=(0.0, 0.38))
        mat_badge(ax, 0.6, 5.75, "Au plasmonic Vivaldi slot horn", "metal")
    elif variant in ("kerr", "tpa"):
        # T39 / T42: Nonlinear χ(3) Kerr FWM sidebands or TPA/FCA in a silicon waveguide
        ax.add_patch(Rectangle((0.8, 2.0), 8.4, 0.48, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2))
        xp = np.linspace(1.2, 8.8, 200)
        wave = 0.16 * np.sin(9.0 * xp) * np.exp(-((xp - 5.0) / 2.8) ** 2)
        ax.plot(xp, 2.24 + wave, color=PAL["light_in"], lw=1.6, zorder=8)
        arrow(ax, 0.5, 2.24, 1.6, 2.24, "Pump P0", PAL["light_in"], label_offset=(0.0, 0.42))
        if variant == "kerr":
            # FWM sideband spectrum inset above waveguide
            for xf, h_pk, col, lbl in [(3.5, 1.1, PAL["light_out"], "ω_s"), (5.0, 1.9, PAL["light_in"], "ω_p"), (6.5, 1.1, PAL["ln"], "ω_i")]:
                ax.plot([xf, xf], [3.1, 3.1 + h_pk], color=col, lw=3.0)
                ax.text(xf, 3.25 + h_pk, lbl, fontsize=7.5, fontweight="bold", color=col, ha="center")
            ax.plot([2.5, 7.5], [3.1, 3.1], color=PAL["text"], lw=1.0)
            mat_badge(ax, 0.6, 5.75, "χ(3) Kerr sideband generation", "si")
        else:
            ax.text(5.0, 3.8, "2ħω > E_g → Two-Photon Absorption (TPA)\n+ Free-Carrier Absorption (FCA)", fontsize=7.6, fontweight="bold", color=PAL["pcm_edge"], ha="center")
            mat_badge(ax, 0.6, 5.75, "Nonlinear Si TPA + FCA", "pcm")
    elif variant == "bullseye":
        # T59: Circular Bragg bullseye cavity around a central quantum emitter
        xc, yc_b = 5.0, 3.35
        ax.add_patch(Circle((xc, yc_b), 2.15, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2))
        for r_g in [0.65, 1.10, 1.55, 1.95]:
            ax.add_patch(Wedge((xc, yc_b), r_g, 0, 360, width=0.14, facecolor=PAL["sio2"], edgecolor=PAL["si_edge"], lw=0.7))
        ax.add_patch(Circle((xc, yc_b), 0.12, facecolor=PAL["light_in"], zorder=12))
        ax.text(8.2, 4.6, "Concentric Bragg\ntrenches + emitter", fontsize=7.4, fontweight="bold", color=PAL["text"], ha="center")
        mat_badge(ax, 0.6, 5.75, "PSO circular bullseye cavity", "si")
    elif variant in ("dbs_switch", "ga_reflector"):
        # T60 / T62: Waveguide device with digital binary etched holes or GA reflector trenches
        ax.add_patch(Rectangle((0.8, yc - 0.25), 2.4, 0.50, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        ax.add_patch(Rectangle((3.2, yc - 1.05), 3.4, 2.10, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.2))
        rng = np.random.default_rng(19 if variant == "dbs_switch" else 7)
        for gx in np.linspace(3.55, 6.25, 7):
            for gy in np.linspace(yc - 0.78, yc + 0.78, 5):
                if rng.random() > 0.35:
                    ax.add_patch(Circle((gx, gy), 0.13, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"], lw=0.7))
        arrow(ax, 0.5, yc, 1.7, yc, "in", PAL["light_in"], label_offset=(0.0, 0.38))
        if variant == "dbs_switch":
            ax.add_patch(Rectangle((6.6, yc + 0.40), 2.4, 0.42, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
            ax.add_patch(Rectangle((6.6, yc - 0.82), 2.4, 0.42, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
            arrow(ax, 8.1, yc + 0.61, 9.3, yc + 0.61, "port 1", PAL["light_out"], label_offset=(0.0, 0.34))
            arrow(ax, 8.1, yc - 0.61, 9.3, yc - 0.61, "port 2", PAL["light_out"], label_offset=(0.0, -0.34))
            mat_badge(ax, 0.6, 5.75, "Direct-binary-search (DBS) switch", "si")
        else:
            ax.add_patch(Rectangle((6.6, yc - 0.25), 2.4, 0.50, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
            arrow(ax, 1.8, yc - 0.12, 0.6, yc - 0.12, "R(λ)", PAL["light_out"], label_offset=(0.0, -0.38))
            mat_badge(ax, 0.6, 5.75, "Genetic-algorithm Si reflector", "si")
    elif variant == "dbr":
        # T72: 1D Distributed Bragg Reflector (DBR) multilayer Fabry-Perot cavity
        x_cur = 1.5
        for i in range(5):
            ax.add_patch(Rectangle((x_cur, 1.4), 0.32, 3.6, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
            ax.add_patch(Rectangle((x_cur + 0.32, 1.4), 0.38, 3.6, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
            x_cur += 0.70
        # Central λ/2 cavity defect layer
        ax.add_patch(Rectangle((x_cur, 1.4), 0.95, 3.6, facecolor="#fef08a", edgecolor=PAL["pcm_edge"], lw=1.2))
        ax.text(x_cur + 0.475, 3.2, "λ/2\ncavity", fontsize=7.8, fontweight="bold", color=PAL["pcm_edge"], ha="center", va="center")
        x_cur += 0.95
        for i in range(4):
            ax.add_patch(Rectangle((x_cur, 1.4), 0.32, 3.6, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
            ax.add_patch(Rectangle((x_cur + 0.32, 1.4), 0.38, 3.6, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
            x_cur += 0.70
        arrow(ax, 0.5, 3.2, 1.4, 3.2, "in", PAL["light_in"], label_offset=(0.0, 0.38))
        mat_badge(ax, 0.6, 5.75, "1D DBR multilayer + λ/2 cavity", "si")
    elif variant == "cmos":
        # T73: Back-illuminated CMOS RGB image sensor pixel array with microlenses
        ax.add_patch(Rectangle((1.2, 1.1), 7.6, 1.4, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        ax.text(5.0, 1.75, "Si photodiode array + deep-trench isolation (DTI)", fontsize=7.2, color="#ffffff", ha="center", va="center")
        for idx, (xp, col, lbl) in enumerate([(2.5, "#ef4444", "R"), (5.0, "#22c55e", "G"), (7.5, "#3b82f6", "B")]):
            ax.add_patch(Rectangle((xp - 1.15, 2.5), 2.3, 0.85, facecolor=col, edgecolor=PAL["si_edge"], alpha=0.75))
            ax.text(xp, 2.92, f"{lbl} CFA", fontsize=7.8, fontweight="bold", color="#ffffff", ha="center", va="center")
            ax.add_patch(Wedge((xp, 3.35), 1.15, 0, 180, facecolor=PAL["sio2"], edgecolor=PAL["si_edge"], lw=1.0))
            arrow(ax, xp, 5.3, xp, 4.6, "", PAL["light_in"], lw=1.5)
        mat_badge(ax, 0.6, 5.75, "CMOS microlens + RGB color filters", "sio2")
    elif variant == "pcm_antenna":
        # T88: GST phase-change tunable plasmonic nanoantenna array
        ax.add_patch(Rectangle((1.0, 1.2), 8.0, 0.7, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        ax.add_patch(Rectangle((1.0, 1.9), 8.0, 0.45, facecolor=PAL["pcm"], edgecolor=PAL["pcm_edge"]))
        ax.text(5.0, 2.12, "Tunable GST phase-change spacer (amorphous ↔ crystalline)", fontsize=7.0, fontweight="bold", color="#ffffff", ha="center", va="center")
        for xp in [2.3, 4.1, 5.9, 7.7]:
            ax.add_patch(Rectangle((xp - 0.45, 2.35), 0.90, 0.45, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"], lw=1.1))
        arrow(ax, 3.2, 5.1, 3.2, 3.2, "k_inc", PAL["light_in"], label_offset=(-0.55, 0.0))
        arrow(ax, 6.2, 3.2, 6.2, 5.1, "R(λ, state)", PAL["light_out"], label_offset=(0.85, 0.0))
        mat_badge(ax, 0.6, 5.75, "Au nanoantennas + GST film", "pcm")
    elif variant == "mos2":
        # T96: Monolayer MoS2 2D exciton-polariton waveguide
        ax.add_patch(Rectangle((1.0, 1.2), 8.0, 1.5, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
        ax.text(5.0, 1.95, "SiO2 / Si substrate", fontsize=7.3, color=PAL["muted"], ha="center", va="center")
        ax.add_patch(Rectangle((1.8, 2.7), 6.4, 0.14, facecolor=PAL["pcm"], edgecolor=PAL["pcm_edge"], lw=1.2, zorder=9))
        ax.add_patch(Ellipse((5.0, 2.77), 3.6, 0.75, facecolor=PAL["light_in"], alpha=0.30, zorder=8))
        ax.text(5.0, 3.65, "Atomically thin monolayer MoS2 (t = 0.65 nm)\nexciton-polariton guided mode", fontsize=7.5, fontweight="bold", color=PAL["text"], ha="center")
        mat_badge(ax, 0.6, 5.75, "2D TMDC monolayer (MoS2)", "pcm")


def draw_invdes(ax, variant: str):
    """Draw inverse-design / adjoint topology optimization schematics (A01–A28)."""
    yc = 3.35
    if variant == "grad_check":
        # Forward + adjoint fields and dJ/dε vs finite-difference curve
        ax.add_patch(Rectangle((1.0, 1.3), 3.6, 3.6, facecolor="#ffffff", edgecolor="#94a3b8"))
        xg = np.linspace(1.1, 4.5, 40)
        yg = np.linspace(1.4, 4.8, 40)
        XX, YY = np.meshgrid(xg, yg)
        sens = np.sin(2.5 * (XX - 2.8)) * np.cos(2.5 * (YY - 3.1)) * np.exp(-((XX - 2.8) ** 2 + (YY - 3.1) ** 2))
        ax.contourf(XX, YY, sens, levels=6, cmap="coolwarm", alpha=0.8, zorder=5)
        ax.text(2.8, 5.1, "∂J/∂ε = Re(E_fwd · E_adj)", fontsize=7.3, fontweight="bold", color=PAL["text"], ha="center")
        # Plot comparing adjoint vs FD points on the right
        ax.add_patch(Rectangle((5.2, 1.3), 3.8, 3.6, facecolor="#ffffff", edgecolor="#94a3b8"))
        xp = np.linspace(5.5, 8.7, 15)
        yp = 3.1 + 1.1 * np.sin(1.8 * (xp - 5.5))
        ax.plot(xp, yp, color=PAL["light_out"], lw=1.8, label="Adjoint")
        ax.scatter(xp[::2], yp[::2], color=PAL["light_in"], s=18, zorder=9, label="FD")
        ax.text(7.1, 5.1, "Adjoint vs Finite Diff", fontsize=7.3, fontweight="bold", color=PAL["text"], ha="center")
        mat_badge(ax, 0.6, 5.75, "Exact discrete adjoint gradient", "ln")
        return

    if variant == "shape_grad":
        # Level-set / boundary normal velocity V_n along smooth interface
        theta = np.linspace(0, 2 * np.pi, 100)
        r = 1.35 + 0.22 * np.cos(3 * theta)
        xc, yc_s = 5.0, 3.30
        pts = np.column_stack([xc + r * np.cos(theta), yc_s + r * np.sin(theta)])
        ax.add_patch(Polygon(pts, closed=True, facecolor=PAL["si"], edgecolor=PAL["si_edge"], lw=1.4))
        for ang in np.linspace(0, 2 * np.pi, 10, endpoint=False):
            ra = 1.35 + 0.22 * np.cos(3 * ang)
            x0, y0 = xc + ra * np.cos(ang), yc_s + ra * np.sin(ang)
            arrow(ax, x0, y0, x0 + 0.45 * np.cos(ang), y0 + 0.45 * np.sin(ang), "", PAL["pcm"], lw=1.5)
        ax.text(5.0, yc_s, "ε(x,y)\nlevel-set φ=0", fontsize=7.8, fontweight="bold", color="#ffffff", ha="center", va="center")
        mat_badge(ax, 0.6, 5.75, "Shape / level-set boundary gradients", "pcm")
        return

    if variant == "patch_rf":
        # Microstrip patch antenna with topology-optimized slot cutout
        ax.add_patch(Rectangle((1.2, 1.2), 7.6, 4.1, facecolor=PAL["sin"], edgecolor=PAL["sin_edge"], alpha=0.35))
        ax.add_patch(Rectangle((2.8, 1.8), 4.4, 2.9, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"], lw=1.2))
        ax.add_patch(Rectangle((1.2, 3.05), 1.6, 0.40, facecolor=PAL["metal"], edgecolor=PAL["metal_edge"]))
        # Topology cutout inside patch
        ax.add_patch(Ellipse((5.0, 3.25), 1.8, 1.1, facecolor=PAL["sio2"], edgecolor=PAL["metal_edge"], lw=1.0))
        arrow(ax, 0.6, 3.25, 1.8, 3.25, "50 Ω feed", PAL["light_in"], label_offset=(0.2, 0.38))
        mat_badge(ax, 0.6, 5.75, "RF microstrip patch (Autograd)", "metal")
        return

    # Topology-optimized grayscale/binarized design region connected to waveguide ports
    ax.add_patch(Rectangle((2.8, 1.45), 4.4, 3.7, facecolor="#ffffff", edgecolor=PAL["ln"], lw=1.4, ls="--", zorder=4))
    rng = np.random.default_rng(abs(hash(variant)) % 1000)
    gx = np.linspace(3.05, 6.95, 11)
    gy = np.linspace(1.70, 4.90, 8)
    for x0 in gx:
        for y0 in gy:
            val = rng.random()
            if val > 0.38:
                col = PAL["si"] if val > 0.62 else "#64748b"
                ax.add_patch(Rectangle((x0 - 0.16, y0 - 0.17), 0.32, 0.34, facecolor=col, edgecolor="none", zorder=6))

    # Input port on left
    ax.add_patch(Rectangle((0.8, yc - 0.22), 2.0, 0.44, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
    arrow(ax, 0.5, yc, 1.6, yc, "in", PAL["light_in"], label_offset=(0.0, 0.38))

    if variant in ("wdm", "splitter", "multi_obj"):
        ax.add_patch(Rectangle((7.2, yc + 0.55), 2.0, 0.38, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        ax.add_patch(Rectangle((7.2, yc - 0.93), 2.0, 0.38, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        arrow(ax, 8.2, yc + 0.74, 9.4, yc + 0.74, "λ1" if variant == "wdm" else "out 1", PAL["light_out"], label_offset=(0.0, 0.34))
        arrow(ax, 8.2, yc - 0.74, 9.4, yc - 0.74, "λ2" if variant == "wdm" else "out 2", PAL["light_out"], label_offset=(0.0, -0.34))
    elif variant == "bend":
        ax.add_patch(Rectangle((4.78, 5.15), 0.44, 0.85, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        arrow(ax, 5.0, 5.2, 5.0, 6.1, "out", PAL["light_out"], label_offset=(0.5, 0.0))
    else:
        ax.add_patch(Rectangle((7.2, yc - 0.22), 2.0, 0.44, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
        arrow(ax, 8.2, yc, 9.4, yc, "out", PAL["light_out"], label_offset=(0.0, 0.38))

    ax.text(5.0, 1.12, f"Adjoint topology ρ(x,y) [{variant}]", fontsize=7.3, fontweight="bold", color=PAL["ln"], ha="center")
    mat_badge(ax, 0.6, 5.75, "Topology optimization ε(ρ)", "ln")


def draw_solver_feature(ax, variant: str):
    """Draw schematic diagrams for solver features and mode-solving notebooks (F01–F37)."""
    yc = 3.35
    if variant in ("mode_solver", "bent_modes", "rad_loss", "overlap", "group_delay", "low_contrast", "eme"):
        # Waveguide cross-section or EME cell with mode profile E(x,y)
        if variant == "eme":
            for i, w_s in enumerate([0.7, 1.0, 1.3, 1.6, 1.9]):
                x0 = 1.6 + i * 1.35
                ax.add_patch(Rectangle((x0, yc - w_s / 2), 1.30, w_s, facecolor=PAL["si"], edgecolor=PAL["si_edge"], alpha=0.85))
                ax.plot([x0, x0], [1.2, 5.4], "--", color=PAL["ln"], lw=1.0)
            ax.text(5.0, 5.3, "Eigenmode Expansion (EME) cell S-matrices", fontsize=7.5, fontweight="bold", color=PAL["text"], ha="center")
            mat_badge(ax, 0.6, 5.75, "Bidirectional EME solver", "ln")
            return
        ax.add_patch(Rectangle((1.2, 1.2), 7.6, 1.3, facecolor=PAL["sio2"], edgecolor=PAL["sio2_edge"]))
        ax.add_patch(Rectangle((3.4, 2.5), 3.2, 1.2, facecolor=PAL["sin" if variant == "low_contrast" else "si"], edgecolor=PAL["si_edge"], lw=1.2))
        xg = np.linspace(2.2, 7.8, 60)
        yg = np.linspace(1.6, 4.6, 40)
        XX, YY = np.meshgrid(xg, yg)
        x_peak = 5.35 if variant == "bent_modes" else 5.0
        mode_f = np.exp(-((XX - x_peak) / 1.1) ** 2 - ((YY - 3.1) / 0.65) ** 2)
        ax.contourf(XX, YY, mode_f, levels=5, cmap="YlOrRd", alpha=0.55, zorder=8)
        ax.text(5.0, 4.95, f"Eigenmode E(x,y), n_eff(ω) [{variant}]", fontsize=7.5, fontweight="bold", color=PAL["text"], ha="center")
        mat_badge(ax, 0.6, 5.75, "Waveguide mode solver", "si")
    elif variant in ("symmetry", "bc", "pml", "tfsf", "norm", "gaussian", "oblique", "custom_src"):
        # Simulation domain box with PML/symmetry/TFSF/beam annotations
        ax.add_patch(Rectangle((1.4, 1.2), 7.2, 4.2, facecolor="#ffffff", edgecolor=PAL["si_edge"], lw=1.2))
        ax.add_patch(Rectangle((1.4, 1.2), 7.2, 4.2, facecolor="none", edgecolor="#64748b", lw=6.0, alpha=0.35))
        if variant == "symmetry":
            ax.plot([5.0, 5.0], [1.0, 5.6], "--", color=PAL["light_in"], lw=1.8)
            ax.text(5.0, 5.48, "PEC / PMC mirror symmetry plane (4× speedup)", fontsize=7.4, fontweight="bold", color=PAL["light_in"], ha="center")
        elif variant == "tfsf":
            ax.add_patch(Rectangle((2.8, 1.9), 4.4, 2.8, facecolor="#eff6ff", edgecolor=PAL["light_in"], lw=1.5, ls="--"))
            ax.add_patch(Circle((5.0, 3.3), 0.65, facecolor=PAL["si"], edgecolor=PAL["si_edge"]))
            ax.text(5.0, 4.9, "Total-Field / Scattered-Field (TF/SF) box", fontsize=7.4, fontweight="bold", color=PAL["light_in"], ha="center")
        elif variant in ("gaussian", "oblique", "custom_src"):
            x = np.linspace(1.8, 8.2, 100)
            y_b = 3.3 + (0.35 * (x - 5.0) if variant == "oblique" else np.zeros_like(x))
            w_b = 0.35 * np.sqrt(1.0 + ((x - 5.0) / 1.6) ** 2)
            ax.fill_between(x, y_b - w_b, y_b + w_b, color=PAL["light_in"], alpha=0.30)
            arrow(ax, 1.9, y_b[0], 8.0, y_b[-1], "Beam / source injection", PAL["light_in"], label_offset=(0.0, 0.55))
        else:
            arrow(ax, 2.2, 3.3, 8.2, 3.3, "Incident wave → PML absorber", PAL["light_in"], label_offset=(0.0, 0.42))
            arrow(ax, 8.2, 2.6, 6.2, 2.6, "R_PML < -80 dB", PAL["light_out"], label_offset=(0.0, -0.38))
        mat_badge(ax, 0.6, 5.75, f"Solver setup: {variant}", "si")
    elif variant in ("fitting", "anisotropic", "gyrotropic", "time_mod", "beer_lambert"):
        # Material dispersion curve ε(ω), tensor ellipsoid, or Beer-Lambert exponential decay
        ax.add_patch(Rectangle((1.3, 1.25), 7.4, 4.0, facecolor="#ffffff", edgecolor="#94a3b8"))
        w = np.linspace(1.6, 8.3, 150)
        if variant == "beer_lambert":
            y_curve = 1.5 + 3.2 * np.exp(-0.55 * (w - 1.6))
            ax.plot(w, y_curve, color=PAL["light_in"], lw=2.2)
            ax.text(5.4, 4.2, "I(z) = I0 exp(-α z)\nBeer-Lambert attenuation", fontsize=7.6, fontweight="bold", color=PAL["text"], ha="center")
        else:
            eps_re = 3.2 + 1.2 * (w - 5.0) / ((w - 5.0) ** 2 + 0.45)
            eps_im = 1.4 * 0.45 / ((w - 5.0) ** 2 + 0.45)
            ax.plot(w, eps_re, color=PAL["light_out"], lw=2.0, label="Re ε(ω)")
            ax.plot(w, 1.4 + eps_im, "--", color=PAL["light_in"], lw=2.0, label="Im ε(ω)")
            ax.text(5.0, 4.85, f"Constitutive tensor / dispersion [{variant}]", fontsize=7.5, fontweight="bold", color=PAL["text"], ha="center")
        mat_badge(ax, 0.6, 5.75, f"Material model: {variant}", "pcm")
    else:
        # Resonance finder, Cavity FOM, S-matrix, PIC components, subpixel mesh, STL import, instability
        ax.add_patch(Rectangle((1.3, 1.25), 7.4, 4.0, facecolor="#ffffff", edgecolor="#94a3b8"))
        if variant in ("subpixel", "layer_ref", "stl_import"):
            for xg in np.linspace(1.3, 8.7, 12):
                ax.plot([xg, xg], [1.25, 5.25], color="#cbd5e1", lw=0.8)
            for yg in np.linspace(1.25, 5.25, 8):
                ax.plot([1.3, 8.7], [yg, yg], color="#cbd5e1", lw=0.8)
            ax.add_patch(Circle((5.0, 3.25), 1.45, facecolor=PAL["si"], edgecolor=PAL["light_in"], lw=1.6, alpha=0.80))
            ax.text(5.0, 3.25, "Conformal ε_ij\nmesh / CAD", fontsize=7.8, fontweight="bold", color="#ffffff", ha="center", va="center")
        else:
            t = np.linspace(1.6, 8.4, 180)
            env = np.exp(-0.45 * (t - 1.6)) if variant != "diverged" else np.exp(0.38 * (t - 1.6)) * 0.12
            sig = 3.25 + 1.4 * np.sin(5.5 * t) * env
            ax.plot(t, np.clip(sig, 1.35, 5.15), color=PAL["light_in" if variant == "diverged" else "light_out"], lw=1.8)
            ax.text(5.0, 4.85, f"Diagnostics / analysis [{variant}]", fontsize=7.5, fontweight="bold", color=PAL["text"], ha="center")
        mat_badge(ax, 0.6, 5.75, f"Solver analysis: {variant}", "si")


SPECS: list[tuple[str, str, str, str, str]] = [
    # (filename, caption, view_label, archetype, variant)
    ("M01_mie_scattering.png", "M01 · Mie scattering of a lossless dielectric sphere", "top view", "scatter", "multipole"),
    ("M02_plasmonic_sphere.png", "M02 · Lossy plasmonic spheres Ag/Au (10 nm–1 µm)", "top view", "scatter", "plasmonic"),
    ("M03_scatterer_shapes.png", "M03 · Shape effects of dielectric scatterers", "top view", "scatter", "shapes"),
    ("M04_radar_cross_section.png", "M04 · Radar cross section (RCS) polar bistatic pattern", "top view", "scatter", "multipole"),
    ("M05_single_slit.png", "M05 · Diffraction from a single slit", "top view", "slit", "single_slit"),
    ("M06_double_slit.png", "M06 · Young double-slit diffraction & interference", "top view", "slit", "double_slit"),
    ("M07_diffraction_grating.png", "M07 · Binary dielectric diffraction grating orders", "cross-section", "slit", "grating"),
    ("M08_hexagonal_grating.png", "M08 · Diffracted orders of a hexagonal lattice", "top view", "metasurface", "vortex"),
    ("M09_metasurface_lens.png", "M09 · Metasurface lens near-to-far-field focusing", "cross-section", "lens", "metalens"),
    ("M10_lc_polarization_grating.png", "M10 · Liquid-crystal Pancharatnam-Berry grating", "cross-section", "slit", "lc"),
    ("M11_electric_dipole_pec.png", "M11 · Electric dipole antenna with PEC ground plane", "cross-section", "dipole", "ed_pec"),
    ("M12_magnetic_dipole_pec.png", "M12 · Magnetic dipole antenna with PEC ground plane", "cross-section", "dipole", "md_pec"),
    ("M13_led_incoherent.png", "M13 · LED stochastic incoherent dipole emission", "cross-section", "dipole", "led"),
    ("M14_led_reciprocity.png", "M14 · LED emission by Lorentz reciprocity", "cross-section", "dipole", "led"),
    # T01..T45 Passive PIC
    ("T01_y_junction.png", "T01 · Waveguide Y junction (13-segment spline)", "top view", "y_junction", "standard"),
    ("T02_wg_to_ring.png", "T02 · Point-coupled waveguide-to-ring coupler", "top view", "ring", "bus"),
    ("T03_wg_crossing.png", "T03 · Waveguide crossing based on cosine tapers", "top view", "crossing", "cosine"),
    ("T04_euler_bend.png", "T04 · Low-loss clothoid Euler waveguide bend", "top view", "bend", "euler"),
    ("T05_directional_coupler.png", "T05 · Symmetric evanescent directional coupler", "top view", "dc", "standard"),
    ("T06_edge_coupler.png", "T06 · Inverse nano-taper spot-size edge coupler", "top view", "taper", "inverse"),
    ("T07_effective_index_2d.png", "T07 · 2D effective-index approximation of 3D slabs", "3D → 2D", "special", "neff2d"),
    ("T08_uniform_grating_coupler.png", "T08 · Uniform linear SOI surface grating coupler", "cross-section", "gc", "uniform"),
    ("T09_focusing_apodized_gc.png", "T09 · Focusing apodized curved grating coupler", "top view", "gc", "focusing"),
    ("T10_bilayer_sin_si_gc.png", "T10 · Unidirectional bilayer SiN/Si grating coupler", "cross-section", "gc", "bilayer"),
    ("T11_vertical_gc_bayes.png", "T11 · Bayesian-optimized vertical SOI grating coupler", "cross-section", "gc", "apodized"),
    ("T12_mmi_1x4.png", "T12 · 1x4 multimode interference (MMI) power splitter", "top view", "mmi", "1x4"),
    ("T13_mmi_2x2.png", "T13 · 2x2 paired-interference MMI 3-dB coupler", "top view", "mmi", "2x2"),
    ("T14_microring_add_drop.png", "T14 · Microring resonator add-drop wavelength filter", "top view", "ring", "add_drop"),
    ("T15_bilevel_taper_psr.png", "T15 · Bi-level taper polarization rotator-splitter", "top view", "pol", "psr"),
    ("T16_mode_size_converter.png", "T16 · Adiabatic waveguide mode-size converter", "top view", "taper", "converter"),
    ("T17_strip_to_slot.png", "T17 · Strip-to-slot optical waveguide converter", "top view", "taper", "strip_to_slot"),
    ("T18_wg_bragg_grating.png", "T18 · Sidewall-corrugated waveguide Bragg grating", "top view", "bragg", "si"),
    ("T19_pcm_bragg_filter.png", "T19 · Phase-change (Sb2Se3) tunable Bragg grating filter", "top view", "bragg", "pcm"),
    ("T20_broadband_dc.png", "T20 · Asymmetric phase-control broadband coupler", "top view", "dc", "broadband"),
    ("T21_compact_psr.png", "T21 · Compact tapered directional-coupler PSR", "top view", "pol", "psr"),
    ("T22_8ch_demux.png", "T22 · 8-channel mode & polarization demultiplexer", "top view", "special", "demux8"),
    ("T23_thz_ring_demux.png", "T23 · THz dual-ring resonator demultiplexer/filter", "top view", "ring", "demux"),
    ("T24_phc_wg_pol_filter.png", "T24 · Photonic-crystal waveguide polarization filter", "top view", "phc", "wg"),
    ("T25_90deg_optical_hybrid.png", "T25 · 4x4 MMI 90-degree coherent optical hybrid", "top view", "mmi", "hybrid90"),
    ("T26_swg_broadband_polarizer.png", "T26 · Subwavelength-grating (SWG) broadband polarizer", "top view", "pol", "swg_pb"),
    ("T27_exceptional_coupling.png", "T27 · Exceptional skin-depth coupling (zero crosstalk)", "top view", "dc", "exceptional"),
    ("T28_tfln_adiabatic_coupler.png", "T28 · Thin-film LiNbO3 (TFLN) adiabatic coupler", "top view", "dc", "tfln"),
    ("T29_90bend_psr.png", "T29 · Polarization splitter-rotator on 90-degree bends", "top view", "bend", "psr90"),
    ("T30_plasmonic_co2_sensor.png", "T30 · Mid-IR plasmonic slot waveguide CO2 sensor", "cross-section", "special", "co2_sensor"),
    ("T31_bilayer_sin_edge_coupler.png", "T31 · Visible-band bilayer SiN edge coupler", "top view", "taper", "bilayer_sin"),
    ("T32_scale_invariant_wg.png", "T32 · Scale-invariant effective-index dielectric waveguide", "cross-section", "special", "scale_inv"),
    ("T33_unidirectional_grating_antenna.png", "T33 · Unidirectional dual-layer waveguide grating antenna", "cross-section", "gc", "antenna"),
    ("T34_freeform_chip_coupler.png", "T34 · Free-form splines for chip-to-chip butt coupling", "top view", "taper", "converter"),
    ("T35_mmi_nir_mir_demux.png", "T35 · MMI-based NIR / mid-IR wavelength demultiplexer", "top view", "mmi", "demux"),
    ("T36_tapered_wg_dispersion.png", "T36 · Group-velocity dispersion (GVD) in tapered waveguide", "top view", "taper", "converter"),
    ("T37_tfln_psr.png", "T37 · Thin-film LiNbO3 (LNOI) polarization splitter-rotator", "top view", "pol", "tfln"),
    ("T38_plasmonic_vivaldi_antenna.png", "T38 · Integrated plasmonic Vivaldi flaring slot antenna", "top view", "special", "vivaldi"),
    ("T39_kerr_sidebands.png", "T39 · χ(3) Kerr four-wave mixing sideband generation", "top view", "special", "kerr"),
    ("T40_anisotropic_mm_pbs.png", "T40 · Anisotropic-metamaterial ultrashort broadband PBS", "top view", "pol", "mm_pbs"),
    ("T41_swg_wg_crossing.png", "T41 · Subwavelength-grating (SWG) waveguide crossing", "top view", "crossing", "swg"),
    ("T42_si_wg_tpa_fca.png", "T42 · Two-photon & free-carrier absorption in Si waveguide", "top view", "special", "tpa"),
    ("T43_si_to_plasmonic_coupler.png", "T43 · Bayesian-optimized Si-to-plasmonic slot coupler", "top view", "taper", "plasmonic_slot"),
    ("T44_photonic_spin_selector.png", "T44 · Direct-binary-search photonic spin selector", "top view", "metasurface", "spin"),
    ("T45_sin_to_alox_coupler.png", "T45 · Interlayer SiN-to-AlOx vertical adiabatic coupler", "top view", "taper", "bilayer_sin"),
    # T46..T57 Active & Multiphysics PIC
    ("T46_metal_heater_phase_shifter.png", "T46 · Thermally tuned SOI waveguide (TiN heater)", "cross-section", "thermo", "metal_heater"),
    ("T47_thermally_tuned_ring.png", "T47 · Thermally tuned microring resonator filter", "top view", "ring", "thermal"),
    ("T48_carrier_injection_mzm.png", "T48 · Carrier-injection Si Mach-Zehnder modulator", "cross-section", "pin", "pin"),
    ("T49_doped_si_thermo_optic.png", "T49 · Thermo-optic modulator with integrated N++ Si heater", "cross-section", "thermo", "doped"),
    ("T50_heat_dissipation_soi.png", "T50 · Steady-state heat dissipation in SOI waveguides", "cross-section", "thermo", "metal_heater"),
    ("T51_transient_thermo_optic.png", "T51 · Transient thermal response of phase shifters", "cross-section", "thermo", "metal_heater"),
    ("T52_cpo_thermal_analysis.png", "T52 · Thermal crosstalk analysis for co-packaged optics", "cross-section", "thermo", "doped"),
    ("T53_tfln_eo_modulator.png", "T53 · Thin-film LiNbO3 Pockels electro-optic modulator", "cross-section", "pin", "tfln"),
    ("T54_photothermal_si_wg.png", "T54 · Self-heating photo-thermal shift in Si waveguides", "cross-section", "thermo", "metal_heater"),
    ("T55_pin_phase_shifter.png", "T55 · Forward-bias PIN carrier + thermal phase shifter", "cross-section", "pin", "pin"),
    ("T56_pin_photodiode_ssac.png", "T56 · Small-signal AC response of vertical PIN photodiode", "cross-section", "pin", "pd"),
    ("T57_soh_slot_eo_modulator.png", "T57 · Silicon-organic hybrid (SOH) slot EO modulator", "cross-section", "pin", "soh"),
    # T58..T63 Global Optimization
    ("T58_pso_pbs.png", "T58 · Particle-swarm optimization (PSO) of a compact PBS", "top view", "pol", "pso_pbs"),
    ("T59_pso_bullseye_cavity.png", "T59 · PSO of a circular Bragg bullseye quantum cavity", "top view", "special", "bullseye"),
    ("T60_dbs_optical_switch.png", "T60 · Direct-binary-search (DBS) digital optical switch", "top view", "special", "dbs_switch"),
    ("T61_cmaes_sbend.png", "T61 · Bezier S-bend waveguide optimized with CMA-ES", "top view", "bend", "sbend"),
    ("T62_ga_si_reflector.png", "T62 · Genetic-algorithm etched Si waveguide reflector", "top view", "special", "ga_reflector"),
    ("T63_bayes_y_branch.png", "T63 · Bayesian vs adjoint optimization of a 1x2 Y branch", "top view", "y_junction", "bayes"),
    # T64..T78 Metamaterials & Periodic Structures
    ("T64_dielectric_ms_absorber.png", "T64 · Huygens all-dielectric metasurface absorber", "cross-section", "metasurface", "pillars"),
    ("T65_gradient_ms_reflector.png", "T65 · Generalized Snell's law gradient metasurface reflector", "cross-section", "metasurface", "gradient"),
    ("T66_graphene_mm_absorber.png", "T66 · Tunable graphene ribbon metamaterial THz absorber", "cross-section", "metasurface", "mim"),
    ("T67_tunable_chiral_ms_pcm.png", "T67 · Phase-change tunable chiral metasurface (CD)", "top view", "metasurface", "chiral"),
    ("T68_microwave_fss.png", "T68 · Jerusalem-cross microwave frequency-selective surface", "top view", "metasurface", "fano"),
    ("T69_high_q_si_resonator.png", "T69 · Symmetry-broken high-Q all-dielectric Si metasurface", "top view", "metasurface", "fano"),
    ("T70_ge_fano_metasurface.png", "T70 · Mid-IR Germanium Fano-resonance metasurface", "top view", "metasurface", "fano"),
    ("T71_biosensor_grating.png", "T71 · Guided-mode resonance (GMR) dielectric biosensor grating", "cross-section", "metasurface", "pillars"),
    ("T72_dbr_cavity.png", "T72 · 1D Distributed Bragg Reflector (DBR) Fabry-Perot cavity", "cross-section", "special", "dbr"),
    ("T73_cmos_rgb_sensor.png", "T73 · Back-illuminated CMOS RGB pixel with microlens", "cross-section", "special", "cmos"),
    ("T74_au_nanorod_array.png", "T74 · Plasmonic Au nanorod array lattice resonance", "cross-section", "metasurface", "nanorod"),
    ("T75_structural_color_dielectric.png", "T75 · All-dielectric Si nanopillar structural color pixels", "cross-section", "metasurface", "color"),
    ("T76_mim_resonator.png", "T76 · Metal-insulator-metal (MIM) gap-plasmon absorber", "cross-section", "metasurface", "mim"),
    ("T77_vortex_metasurface.png", "T77 · Geometric-phase optical vortex beam metasurface (OAM)", "top view", "metasurface", "vortex"),
    ("T78_quasi_bic_metasurface.png", "T78 · Tilted-ellipse quasi-BIC ultra-high-Q metasurface", "top view", "metasurface", "bic"),
    # T79..T84 Photonic Crystals
    ("T79_phc_slab_bandstructure.png", "T79 · Photonic band structure ω(k) of a 2D PhC slab", "band diagram", "band", "square"),
    ("T80_optimized_l3_cavity.png", "T80 · Quality-factor-optimized L3 photonic-crystal cavity", "top view", "phc", "l3"),
    ("T81_bistable_phc_cavity.png", "T81 · Optical Kerr bistability in a PhC defect nanocavity", "top view", "phc", "bistable"),
    ("T82_nanobeam_cavity.png", "T82 · 1D photonic-crystal tapered nanobeam cavity", "top view", "nanobeam", "standard"),
    ("T83_topological_edge_states.png", "T83 · Valley-Hall topological photonic-crystal edge state", "top view", "phc", "topo"),
    ("T84_hexagonal_lattice_bands.png", "T84 · Γ-M-K-Γ band diagram of a hexagonal PhC lattice", "band diagram", "band", "hex"),
    # T85..T88 Scattering & Far Field
    ("T85_yagi_uda_nanoantenna.png", "T85 · Directional plasmonic Yagi-Uda optical nanoantenna", "top view", "scatter", "yagi"),
    ("T86_zone_plate_projection.png", "T86 · Fresnel zone plate near-to-far-field focal projection", "cross-section", "lens", "zone_plate"),
    ("T87_multipole_expansion.png", "T87 · Exact Cartesian/spherical multipole scattering expansion", "top view", "scatter", "multipole"),
    ("T88_phase_change_antennas.png", "T88 · GST phase-change tunable plasmonic nanoantennas", "cross-section", "special", "pcm_antenna"),
    # T89..T92 Lenses
    ("T89_luneburg_lens.png", "T89 · 2D graded-index (GRIN) optical Luneburg lens", "top view", "lens", "luneburg"),
    ("T90_visible_metalens.png", "T90 · TiO2 nanofin Pancharatnam-Berry visible metalens", "cross-section", "lens", "visible"),
    ("T91_mid_ir_si_metalens.png", "T91 · Mid-IR cylindrical Si-pillar high-NA metalens", "cross-section", "lens", "metalens"),
    ("T92_spherical_fresnel_lens.png", "T92 · Concentric-grooved dielectric Fresnel zone lens", "cross-section", "lens", "fresnel"),
    # T93..T102 Nanophotonics
    ("T93_hbn_hyperbolic_polaritons.png", "T93 · Hyperbolic phonon polaritons in nanostructured hBN", "cross-section", "metasurface", "pillars"),
    ("T94_non_hermitian_metagrating.png", "T94 · Unidirectional SPP launching at an exceptional point", "cross-section", "metasurface", "gradient"),
    ("T95_anderson_localization.png", "T95 · Transverse Anderson localization in 2D disordered rods", "top view", "phc", "anderson"),
    ("T96_mos2_monolayer_wg.png", "T96 · Exciton-polariton guiding in a monolayer MoS2 waveguide", "cross-section", "special", "mos2"),
    ("T97_radiative_cooling_glass.png", "T97 · Daytime radiative-cooling photonic glass coating", "cross-section", "metasurface", "color"),
    ("T98_metal_oxide_sunscreen.png", "T98 · UV Mie attenuation of TiO2 / ZnO nanoparticles", "top view", "scatter", "multipole"),
    ("T99_disordered_plasmonic_color.png", "T99 · Disordered gap-plasmon nanoparticles for structural color", "cross-section", "metasurface", "mim"),
    ("T100_directional_si_nanodisks.png", "T100 · Kerker zero-backscattering in resonant Si nanodisks", "top view", "scatter", "kerker"),
    ("T101_plasmonic_photothermal_au.png", "T101 · Plasmonic photothermal heating of Au nanoparticles", "top view", "scatter", "photothermal"),
    ("T102_blue_micro_led.png", "T102 · GaN blue micro-LED light extraction with inclined mesa", "cross-section", "dipole", "uled"),
    # T103 Fiber
    ("T103_antiresonant_hcf.png", "T103 · Anti-resonant nodeless hollow-core fiber (2D modes)", "cross-section", "hcf", "standard"),
    # F01..F37 Solver features and mode solving
    ("F01_symmetry_planes.png", "F01 · Mirror symmetry planes (PEC / PMC)", "domain setup", "feature", "symmetry"),
    ("F02_boundary_conditions.png", "F02 · Boundary conditions (PEC/PMC/periodic/Bloch/PML)", "domain setup", "feature", "bc"),
    ("F03_pml_reflection.png", "F03 · Absorbing boundary reflection vs PML thickness", "domain setup", "feature", "pml"),
    ("F04_tfsf_source.png", "F04 · Total-Field / Scattered-Field (TF/SF) plane-wave source", "domain setup", "feature", "tfsf"),
    ("F05_source_normalization.png", "F05 · Broadband pulse source normalization", "domain setup", "feature", "norm"),
    ("F06_gaussian_beams.png", "F06 · Gaussian and advanced astigmatic beams", "beam profile", "feature", "gaussian"),
    ("F07_oblique_plane_wave.png", "F07 · Broadband plane wave at constant oblique angle", "domain setup", "feature", "oblique"),
    ("F08_custom_field_source.png", "F08 · Custom spatial field source injection", "domain setup", "feature", "custom_src"),
    ("F09_dispersive_fitting.png", "F09 · Multi-pole dispersive material fitting", "dispersion", "feature", "fitting"),
    ("F10_fully_anisotropic.png", "F10 · Fully anisotropic permittivity tensor media", "constitutive", "feature", "anisotropic"),
    ("F11_gyrotropic_media.png", "F11 · Gyrotropic (magneto-optic) media", "constitutive", "feature", "gyrotropic"),
    ("F12_time_modulated_media.png", "F12 · Time-modulated permittivity media ε(t)", "constitutive", "feature", "time_mod"),
    ("F13_resonance_finder.png", "F13 · Filter-diagonalization resonance finder (harminv)", "time trace", "feature", "harminv"),
    ("F14_cavity_fom.png", "F14 · Cavity figures of merit (Q, mode volume V, Purcell)", "diagnostics", "feature", "cavity_fom"),
    ("F15_smatrix_multiport.png", "F15 · Multiport scattering matrix S_ij extraction", "diagnostics", "feature", "smatrix"),
    ("F16_mode_solver.png", "F16 · Waveguide eigenmode solver", "cross-section", "feature", "mode_solver"),
    ("F17_modes_bent_angled.png", "F17 · Modes of bent and angled waveguides", "cross-section", "feature", "bent_modes"),
    ("F18_radiative_losses_mode.png", "F18 · Radiative bending losses from the mode solver", "cross-section", "feature", "rad_loss"),
    ("F19_mode_overlap.png", "F19 · Waveguide mode overlap integral", "cross-section", "feature", "overlap"),
    ("F20_group_delay.png", "F20 · Group index and group delay calculation", "cross-section", "feature", "group_delay"),
    ("F21_low_contrast_wg.png", "F21 · Low-index-contrast waveguide modes", "cross-section", "feature", "low_contrast"),
    ("F22_near2far_dielectric_rcs.png", "F22 · Near-to-far field: dielectric sphere RCS", "top view", "scatter", "multipole"),
    ("F23_pec_sphere_rcs.png", "F23 · Perfectly conducting (PEC) sphere RCS", "top view", "scatter", "pec"),
    ("F24_plasmonic_nanoparticle.png", "F24 · Plasmonic nanoparticle localized resonance", "top view", "scatter", "plasmonic"),
    ("F25_field_projections.png", "F25 · Near-to-far angular & planar field projections", "top view", "scatter", "multipole"),
    ("F26_grating_efficiency.png", "F26 · Grating diffraction efficiency by Floquet order", "cross-section", "slit", "grating"),
    ("F27_maxwell_stress_tensor.png", "F27 · Optical forces via Maxwell stress tensor", "top view", "scatter", "mst"),
    ("F28_beer_lambert_check.png", "F28 · Beer-Lambert exponential absorption verification", "attenuation", "feature", "beer_lambert"),
    ("F29_led_extraction_efficiency.png", "F29 · LED light-extraction efficiency (LEE)", "cross-section", "dipole", "led"),
    ("F30_phc_components.png", "F30 · Photonic-crystal waveguide & cavity components", "top view", "phc", "wg"),
    ("F31_pic_component_library.png", "F31 · Photonic integrated circuit (PIC) component library", "diagnostics", "feature", "pic_lib"),
    ("F32_mmi_meep_benchmark.png", "F32 · MMI splitter benchmark against Meep", "top view", "mmi", "2x2"),
    ("F33_diverged_simulation.png", "F33 · Diagnosing diverged FDTD/FIT instabilities", "time trace", "feature", "diverged"),
    ("F34_stl_gds_import.png", "F34 · CAD STL and GDSII layout geometry import", "mesh view", "feature", "stl_import"),
    ("F35_local_subpixel.png", "F35 · Local subpixel permittivity tensor averaging", "mesh view", "feature", "subpixel"),
    ("F36_layer_refinement.png", "F36 · Layer refinement and automatic graded meshing", "mesh view", "feature", "layer_ref"),
    ("F37_eme_solver.png", "F37 · Eigenmode expansion (EME) solver", "top view", "feature", "eme"),
    # A01..A28 Inverse Design (Autograd)
    ("A01_gradient_checking.png", "A01 · Adjoint vs finite-difference gradient verification", "sensitivity", "invdes", "grad_check"),
    ("A02_invdes_intro.png", "A02 · Introduction to adjoint topology inverse design", "top view", "invdes", "intro"),
    ("A03_multi_objective.png", "A03 · Multi-objective wavelength/mode inverse design", "top view", "invdes", "multi_obj"),
    ("A04_boundary_gradients.png", "A04 · Shape optimization with boundary gradients", "top view", "invdes", "shape_grad"),
    ("A05_invdes_grating_coupler.png", "A05 · Adjoint-optimized surface grating coupler", "cross-section", "gc", "apodized"),
    ("A06_invdes_metalens.png", "A06 · Topology-optimized flat metalens", "cross-section", "lens", "metalens"),
    ("A07_invdes_wg_bend.png", "A07 · Inverse-designed compact 90° waveguide bend", "top view", "invdes", "bend"),
    ("A08_invdes_wdm.png", "A08 · Adjoint wavelength-division multiplexer (WDM)", "top view", "invdes", "wdm"),
    ("A09_y_branch_level_set.png", "A09 · Level-set boundary optimization of a 1x2 Y branch", "top view", "y_junction", "bayes"),
    ("A10_light_extractor.png", "A10 · Inverse-designed quantum emitter light extractor", "cross-section", "dipole", "led"),
    ("A11_invdes_metasurface.png", "A11 · Topology-optimized freeform metasurface", "top view", "metasurface", "spin"),
    ("A12_invdes_antenna.png", "A12 · Inverse-designed optical dielectric nanoantenna", "top view", "invdes", "antenna"),
    ("A13_bilayer_coupler.png", "A13 · Adjoint-optimized bilayer interlayer coupler", "top view", "taper", "bilayer_sin"),
    ("A14_bandpass_filter.png", "A14 · Topology-optimized waveguide band-pass filter", "top view", "invdes", "filter"),
    ("A15_topology_bend.png", "A15 · Binarized topology-optimized waveguide bend", "top view", "invdes", "bend"),
    ("A16_apodized_coupler.png", "A16 · Gradient-optimized apodized grating coupler", "cross-section", "gc", "apodized"),
    ("A17_metalens_wg_taper.png", "A17 · Ultrashort metalens waveguide mode taper", "top view", "invdes", "taper"),
    ("A18_gap_light_extractor.png", "A18 · GaP nanopatterned quantum light extractor", "cross-section", "dipole", "uled"),
    ("A19_invdes_phc.png", "A19 · Inverse-designed photonic-crystal defect cavity", "top view", "phc", "l3"),
    ("A20_fab_aware_invdes.png", "A20 · Fabrication-aware minimum-feature inverse design", "top view", "invdes", "fab_aware"),
    ("A21_digital_splitter.png", "A21 · Pixelated digital 1x2 power splitter", "top view", "invdes", "splitter"),
    ("A22_invdes_crossing.png", "A22 · Topology-optimized compact waveguide crossing", "top view", "crossing", "cosine"),
    ("A23_diffractive_splitter.png", "A23 · Inverse-designed diffractive beam splitter", "cross-section", "slit", "grating"),
    ("A24_smatrix_gradients.png", "A24 · Multi-port S-matrix automatic differentiation", "top view", "invdes", "smatrix"),
    ("A25_fiber_lens.png", "A25 · 3D-printed / etched fiber-tip microlens", "cross-section", "lens", "fresnel"),
    ("A26_source_gradients.png", "A26 · Adjoint gradients with respect to source fields", "sensitivity", "invdes", "grad_check"),
    ("A27_gc_beam_opt.png", "A27 · Joint grating coupler and incident beam optimization", "cross-section", "gc", "bilayer"),
    ("A28_microstrip_patch_rf.png", "A28 · Inverse-designed RF microstrip patch antenna", "top view", "invdes", "patch_rf"),
]


def render_one(filename: str, caption: str, view_label: str, archetype: str, variant: str) -> Path:
    """Dispatch and save one schematic thumbnail."""
    fig, ax = new_canvas(caption, view_label)
    if archetype == "y_junction":
        draw_y_junction(ax, variant)
    elif archetype == "ring":
        draw_ring_resonator(ax, variant)
    elif archetype == "crossing":
        draw_crossing(ax, swg=(variant == "swg"))
    elif archetype == "bend":
        draw_bend(ax, variant)
    elif archetype == "dc":
        draw_directional_coupler(ax, variant)
    elif archetype == "mmi":
        draw_mmi(ax, variant)
    elif archetype == "gc":
        draw_grating_coupler(ax, variant)
    elif archetype == "bragg":
        draw_bragg_grating(ax, pcm=(variant == "pcm"))
    elif archetype == "taper":
        draw_edge_coupler(ax, variant)
    elif archetype == "pol":
        draw_polarization_device(ax, variant)
    elif archetype == "thermo":
        draw_thermo_optic_xsec(ax, variant)
    elif archetype == "pin":
        draw_pin_eo_xsec(ax, variant)
    elif archetype == "metasurface":
        draw_metasurface(ax, variant)
    elif archetype == "phc":
        draw_phc_lattice(ax, variant)
    elif archetype == "nanobeam":
        draw_nanobeam(ax)
    elif archetype == "band":
        draw_band_diagram(ax, variant)
    elif archetype == "lens":
        draw_lens(ax, variant)
    elif archetype == "scatter":
        draw_scattering(ax, variant)
    elif archetype == "slit":
        draw_slit_or_grating(ax, variant)
    elif archetype == "dipole":
        draw_dipole_or_led(ax, variant)
    elif archetype == "hcf":
        draw_hollow_core_fiber(ax)
    elif archetype == "special":
        draw_special_device(ax, variant)
    elif archetype == "invdes":
        draw_invdes(ax, variant)
    elif archetype == "feature":
        draw_solver_feature(ax, variant)
    else:
        raise ValueError(f"Unknown archetype: {archetype}")
    return save_thumb(fig, filename)


def main() -> None:
    """Generate all schematic thumbnails, remove stale files, and report size statistics."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = {spec[0] for spec in SPECS}
    for old_png in OUT_DIR.glob("*.png"):
        if old_png.name not in expected:
            old_png.unlink()
    paths = [render_one(*spec) for spec in SPECS]
    sizes_kb = [p.stat().st_size / 1024.0 for p in paths]
    total_mb = sum(sizes_kb) / 1024.0
    max_kb = max(sizes_kb)
    print(
        f"Generated {len(paths)} thumbnails in {OUT_DIR}: "
        f"max = {max_kb:.1f} kB, total = {total_mb:.2f} MB"
    )
    if max_kb >= 60.0:
        raise RuntimeError(f"Thumbnail exceeds 60 kB limit: {max_kb:.1f} kB")
    if total_mb >= 10.0:
        raise RuntimeError(f"Total thumbnail size exceeds 10 MB limit: {total_mb:.2f} MB")


if __name__ == "__main__":
    main()
