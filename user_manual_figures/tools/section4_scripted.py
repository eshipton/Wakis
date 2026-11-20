"""Deterministic section-4 illustrations whose geometry or mathematics must be exact.

Figures 4.1, 4.2, 4.4 and 4.6 of ``user_manual.md``. The generated (PaperVizAgent)
candidates for these misplaced edges, arrows or circumcentres, so they are drawn
here from the actual geometry: an isometric cube with its dual cell (4.1), the
Maxwell grid-equation diagram with an unfolded-cube proof of S C = 0 (4.2), the
exact leapfrog amplification roots and the modified energy of one mode (4.4), and
a real Whitney mass-matrix pattern plus circumcentres of acute and obtuse triangle
pairs (4.6). Style matches the generated figures of the section.

Usage: python section4_scripted.py OUTDIR
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

BLUE, LBLUE = "#2f6fb0", "#d6e6f5"
ORANGE, LORANGE = "#e8873a", "#fbe0c8"
GREY, LGREY = "#4a4f57", "#e6e8eb"
GREEN, LGREEN = "#5f9e6e", "#d3ebd8"
RED = "#c0392b"
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
    a = FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
        connectionstyle=f"arc3,rad={rad}", zorder=z, linestyle=ls, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(a)
    return a


def box(ax, xy, w, h, text, fc, ec, fs=17, tc="black"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def bare(ax):
    ax.set_aspect("equal")
    ax.axis("off")


# --------------------------------------------------------------------------- 4.1
def iso(p):
    """Cabinet oblique projection (depth y drawn up and to the right at half scale).

    Unlike an isometric view, the cube centre does not coincide with a projected vertex.
    """
    x, y, z = p
    return np.array([x + 0.5 * y * np.cos(np.radians(35)), z + 0.5 * y * np.sin(np.radians(35))])


def fig4_1(out):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.0, 0.0, 0.58, 1.0])
    bare(ax)
    V = {(i, j, k): np.array([i, j, k], float) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    edges = [(a, b) for a in V for b in V if sum(abs(np.array(a) - np.array(b))) == 1 and a < b]
    for a, b in edges:
        hidden = (0, 1, 0) in (a, b)
        P, Q = iso(V[a]), iso(V[b])
        ax.plot(*zip(P, Q), color=BLUE, lw=1.4 if hidden else 2.2, ls=(0, (4, 4)) if hidden else "-",
                alpha=0.55 if hidden else 1, zorder=2)
    # primary face A_m: x = 1
    face = [iso(p) for p in ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))]
    ax.add_patch(Polygon(face, closed=True, fc=LBLUE, ec="none", alpha=0.85, zorder=1))
    c = iso((1, 0.5, 0.5))
    ax.text(*(c + [-0.17, 0.30]), r"$A_m,\ \widehat{b}_m$", color=BLUE, fontsize=20, zorder=5)
    # dual vertices and dual edge L~_m from this cell centre through A_m to the next centre
    v0, v1 = iso((0.5, 0.5, 0.5)), iso((1.5, 0.5, 0.5))
    ax.plot(*zip(v0, iso((1, 0.5, 0.5))), color=ORANGE, lw=2.6, ls=(0, (5, 3)), zorder=3)
    ax.plot(*zip(iso((1, 0.5, 0.5)), v1), color=ORANGE, lw=2.6, zorder=6)
    arrow(ax, iso((1.3, 0.5, 0.5)), v1, ORANGE, lw=2.6, z=6)
    ax.add_patch(Circle(iso((1, 0.5, 0.5)), 0.022, color=ORANGE, zorder=7))
    ax.add_patch(Circle(v0, 0.035, color=ORANGE, zorder=7))
    ax.add_patch(Circle(v1, 0.035, fc="white", ec=ORANGE, lw=2, zorder=7))
    ax.text(*(v0 + [-0.16, 0.06]), r"$\tilde v_p$", color=ORANGE, fontsize=21)
    ax.text(*(v1 + [0.05, 0.06]), r"$\tilde L_m,\ \widehat{h}_m$", color=ORANGE, fontsize=20)
    # primary edge L_k: (0,1,0)->(1,1,0), and its dual face A~_k (plane x = 0.5)
    s = 0.32
    sq = [iso((0.5, dy, dz)) for dy, dz in ((-s, -s), (s, -s), (s, s), (-s, s))]
    ax.add_patch(Polygon(sq, closed=True, fc=LORANGE, ec=ORANGE, lw=2, alpha=0.75, zorder=4))
    P, Q = iso((0, 0, 0)), iso((1, 0, 0))
    ax.plot(*zip(P, iso((0.5, 0, 0))), color=BLUE, lw=5, zorder=3, solid_capstyle="butt")
    arrow(ax, iso((0.5, 0, 0)), Q, BLUE, lw=5, ms=26, z=6)
    ax.add_patch(Circle(iso((0.5, 0, 0)), 0.02, color=BLUE, zorder=7))
    ax.text(*(iso((0.5, -s, -s)) + [-0.62, -0.16]), r"$\tilde A_k,\ \widehat{d}_k$", color=ORANGE, fontsize=20)
    ax.text(*(Q + [-0.05, -0.20]), r"$L_k,\ \widehat{e}_k$", color=BLUE, fontsize=20)
    ax.plot([], [], color=BLUE, lw=2.4, label=r"primary grid $G$")
    ax.plot([], [], color=ORANGE, lw=2.4, label=r"dual grid $\tilde G$")
    ax.legend(loc="upper left", frameon=False, fontsize=16)
    ax.set_xlim(-0.75, 2.15)
    ax.set_ylim(-0.75, 1.55)

    # right panel: Faraday on one face
    ax = fig.add_axes([0.58, 0.04, 0.40, 0.92])
    bare(ax)
    L = 2.0
    ax.add_patch(Rectangle((0, 0), L, L, fc=LBLUE, ec="none", zorder=1))
    segs = [((0, 0), (L, 0), r"$\widehat{e}_1$", "+1", (L / 2, -0.30)),
            ((L, 0), (L, L), r"$\widehat{e}_2$", "+1", (L + 0.30, L / 2)),
            ((0, L), (L, L), r"$\widehat{e}_3$", "−1", (L / 2, L + 0.30)),
            ((0, 0), (0, L), r"$\widehat{e}_4$", "−1", (-0.30, L / 2))]
    for p, q, lab, sgn, t in segs:
        ax.plot(*zip(p, q), color=BLUE, lw=3, zorder=2)
        mid = (np.array(p) + np.array(q)) / 2
        d = (np.array(q) - np.array(p)) / L
        arrow(ax, mid - 0.15 * d, mid + 0.15 * d, BLUE, lw=3, ms=26)
        ax.text(*t, lab, ha="center", va="center", fontsize=21, color=BLUE)
        horizontal = p[1] == q[1]
        tag = np.array(t) + (np.array([0.40, 0.0]) if horizontal else np.array([0.0, -0.36]))
        ax.text(*tag, sgn, ha="center", va="center", fontsize=15, color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc=GREY, ec="none"))
    th = np.linspace(0.25 * np.pi, 1.85 * np.pi, 80)
    ax.plot(L / 2 + 0.55 * np.cos(th), L / 2 + 0.55 * np.sin(th), color=GREY, lw=1.8)
    arrow(ax, (L / 2 + 0.55 * np.cos(th[-2]), L / 2 + 0.55 * np.sin(th[-2])),
          (L / 2 + 0.55 * np.cos(th[-1]), L / 2 + 0.55 * np.sin(th[-1])), GREY, lw=1.8, ms=20)
    ax.add_patch(Circle((L / 2, L / 2), 0.09, fc="white", ec=BLUE, lw=2, zorder=4))
    ax.add_patch(Circle((L / 2, L / 2), 0.03, color=BLUE, zorder=5))
    ax.text(L / 2 + 0.14, L / 2 - 0.02, r"$\widehat{b}_m$", fontsize=21, color=BLUE, va="center")
    ax.text(L / 2, -0.95, r"$\widehat{e}_1+\widehat{e}_2-\widehat{e}_3-\widehat{e}_4=-\,\mathrm{d}\widehat{b}_m/\mathrm{d}t$",
            ha="center", fontsize=22)
    ax.text(L / 2, -1.35, "exact: no approximation", ha="center", fontsize=17, color=GREY)
    ax.set_xlim(-0.9, L + 0.9)
    ax.set_ylim(-1.6, L + 0.75)
    fig.savefig(out / "fig4_1_dual_cell_complex.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- 4.2
def fig4_2(out):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.0, 0.0, 0.62, 1.0])
    bare(ax)
    xl, xr, bw, bh = 0.0, 3.9, 2.2, 0.62
    ax.text(xl, 3.75, r"primary grid $G$", ha="center", fontsize=18, color=BLUE)
    ax.text(xr, 3.75, r"dual grid $\tilde G$", ha="center", fontsize=18, color=ORANGE)
    box(ax, (xl, 2.0), bw, bh, r"$\widehat{\mathbf{e}}$  edge voltages", LBLUE, BLUE)
    box(ax, (xl, 1.0), bw, bh, r"$\widehat{\mathbf{b}}$  face fluxes", LBLUE, BLUE)
    box(ax, (xl, 0.0), bw, bh, r"$\mathbf{0}$", LBLUE, BLUE)
    box(ax, (xr, 3.0), bw, bh, r"$\mathbf{q}$  charge", LORANGE, ORANGE)
    box(ax, (xr, 2.0), bw, bh, r"$\widehat{\mathbf{d}},\ \widehat{\mathbf{j}}$  dual face fluxes", LORANGE, ORANGE, fs=16)
    box(ax, (xr, 1.0), bw, bh, r"$\widehat{\mathbf{h}}$  dual edge voltages", LORANGE, ORANGE, fs=16)
    arrow(ax, (xl, 2.0 - bh / 2), (xl, 1.0 + bh / 2), GREY, lw=2.6, ms=24)
    arrow(ax, (xl, 1.0 - bh / 2), (xl, 0.0 + bh / 2), GREY, lw=2.6, ms=24)
    arrow(ax, (xr, 1.0 + bh / 2), (xr, 2.0 - bh / 2), GREY, lw=2.6, ms=24)
    arrow(ax, (xr, 2.0 + bh / 2), (xr, 3.0 - bh / 2), GREY, lw=2.6, ms=24)
    ax.text(xl - 0.12, 1.5, r"$\mathbf{C}$", ha="right", va="center", fontsize=22)
    ax.text(xl - 0.12, 0.5, r"$\mathbf{S}$", ha="right", va="center", fontsize=22)
    ax.text(xr + 0.12, 1.5, r"$\tilde{\mathbf{C}}=\mathbf{C}^{T}$", ha="left", va="center", fontsize=22)
    ax.text(xr + 0.12, 2.5, r"$\tilde{\mathbf{S}}$", ha="left", va="center", fontsize=22)
    for y, lab in ((2.0, r"$\mathbf{M}_\varepsilon$"), (1.0, r"$\mathbf{M}_\nu$")):
        arrow(ax, (xl + bw / 2 + 0.05, y), (xr - bw / 2 - 0.05, y), GREEN, lw=3, style="<|-|>", ms=22)
        ax.text((xl + xr) / 2, y + 0.12, lab, ha="center", va="bottom", fontsize=23, color=GREEN)
    ax.text((xl + xr) / 2, 0.15, "material matrices:\napproximation", ha="center", va="center", fontsize=15,
            color="white", bbox=dict(boxstyle="round,pad=0.4", fc=GREEN, ec="none"))
    ax.text(xl, 3.0, "vertical arrows: entries\nin {−1, 0, +1}, exact", ha="center", va="center",
            fontsize=15, color="white", bbox=dict(boxstyle="round,pad=0.4", fc=GREY, ec="none"))
    eq = (r"$\mathbf{C}\widehat{\mathbf{e}}=-\mathrm{d}\widehat{\mathbf{b}}/\mathrm{d}t$      "
          r"$\mathbf{C}^{T}\,\widehat{\mathbf{h}}=\mathrm{d}\widehat{\mathbf{d}}/\mathrm{d}t+\widehat{\mathbf{j}}$      "
          r"$\mathbf{S}\widehat{\mathbf{b}}=\mathbf{0}$      $\tilde{\mathbf{S}}\widehat{\mathbf{d}}=\mathbf{q}$")
    ax.text((xl + xr) / 2, -0.85, eq, ha="center", fontsize=19)
    ax.set_xlim(-1.4, 5.3)
    ax.set_ylim(-1.15, 4.05)

    # inset: unfolded cube, every face circulated with its outward normal
    ax = fig.add_axes([0.63, 0.12, 0.36, 0.80])
    bare(ax)
    net = [(1, 2), (0, 1), (1, 1), (2, 1), (3, 1), (1, 0)]
    for i, j in net:
        hl = (i, j) in ((1, 1), (2, 1))
        ax.add_patch(Rectangle((i, j), 1, 1, fc=LBLUE if hl else LGREY, ec=GREY, lw=1.5))
        th = np.linspace(0.3 * np.pi, 1.9 * np.pi, 50)
        r, cx = 0.2, i + 0.47
        ax.plot(cx + r * np.cos(th), j + 0.55 + r * np.sin(th), color=GREY, lw=1.4)
        arrow(ax, (cx + r * np.cos(th[-2]), j + 0.55 + r * np.sin(th[-2])),
              (cx + r * np.cos(th[-1]), j + 0.55 + r * np.sin(th[-1])), GREY, lw=1.4, ms=13)
    # shared edge x = 2 between faces (1,1) and (2,1): traversed up by the left face, down by the right
    ax.plot([2, 2], [1, 2], color="black", lw=3.5)
    arrow(ax, (1.90, 1.30), (1.90, 1.85), BLUE, lw=2.5, ms=18)
    arrow(ax, (2.10, 1.85), (2.10, 1.30), BLUE, lw=2.5, ms=18)
    ax.text(1.86, 1.13, "+1", ha="right", va="center", fontsize=15, color=BLUE)
    ax.text(2.14, 1.13, "−1", ha="left", va="center", fontsize=15, color=BLUE)
    ax.text(2.0, -0.35, "unfolded cube, outward normals:\nevery edge is used twice, once with each sign",
            ha="center", va="top", fontsize=14, color=GREY)
    ax.text(2.0, -1.25, r"$\partial\partial=0\ \Rightarrow\ \mathbf{S}\,\mathbf{C}=\mathbf{0}$", ha="center",
            fontsize=22)
    ax.text(2.0, -1.65, "magnetic flux and charge\nconserved to round-off", ha="center", va="top", fontsize=15)
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-2.4, 3.2)
    fig.savefig(out / "fig4_2_maxwell_grid_equations.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- 4.4
def roots(a):
    """Leapfrog amplification factors for one mode, a = dt^2 * lambda."""
    return np.roots([1.0, -(2.0 - a), 1.0])


def leapfrog_mode(a, n_steps):
    """One mode (M = 1, C = sqrt(lambda), dt = 1, a = dt^2 lambda): field norm |(e^n, b^{n+1/2})| and modified energy W^n.

    W^n = (e^n)^2/2 + b^{n-1/2} b^{n+1/2}/2 is conserved by every leapfrog step; it is positive (a norm)
    only for a < 4. Above the limit W stays constant but is indefinite, and the amplitude grows.
    """
    c = np.sqrt(a)
    e, b_prev = 1.0, 0.0
    amp, w = [], []
    for _ in range(n_steps):
        b_next = b_prev - c * e
        amp.append(np.hypot(e, b_next))
        w.append(0.5 * e * e + 0.5 * b_prev * b_next)
        e = e + c * b_next
        b_prev = b_next
    return np.array(amp), np.array(w)


def fig4_4(out):
    fig = plt.figure(figsize=(W * 21 / 16 * 0.85, H * 0.85))
    # left: staggered time levels
    ax = fig.add_axes([0.01, 0.10, 0.33, 0.80])
    bare(ax)
    ax.annotate("", xy=(4.6, 0), xytext=(-0.3, 0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5))
    ticks = [r"$n-1$", r"$n-\frac{1}{2}$", r"$n$", r"$n+\frac{1}{2}$", r"$n+1$"]
    for i, t in enumerate(ticks):
        ax.plot([i, i], [-0.07, 0.07], color=GREY, lw=1.5)
        ax.text(i, -0.35, t, ha="center", fontsize=17)
        if i % 2 == 0:
            ax.add_patch(Circle((i, 0.7), 0.17, color=BLUE, zorder=4))
            ax.text(i, 1.05, r"$\widehat{\mathbf{e}}$", ha="center", fontsize=19, color=BLUE)
        else:
            ax.add_patch(Rectangle((i - 0.16, 0.54), 0.32, 0.32, color=ORANGE, zorder=4))
            ax.text(i, 1.05, r"$\widehat{\mathbf{b}}$", ha="center", fontsize=19, color=ORANGE)
    ax.text(4.55, -0.35, r"$t$", fontsize=17)
    for i in range(4):
        main = i >= 2
        arrow(ax, (i + 0.15, 0.95), (i + 0.85, 0.95), BLUE if i % 2 == 0 else ORANGE,
              lw=2.4 if main else 1.4, rad=-0.6, ms=18 if main else 13)
    ax.text(2.45, 1.50, r"$-\Delta t\,\mathbf{C}$", ha="center", fontsize=18, color=BLUE)
    ax.text(3.75, 1.50, r"$\Delta t\,\mathbf{M}_\varepsilon^{-1}\mathbf{C}^{T}\mathbf{M}_\nu$", ha="center",
            fontsize=17, color=ORANGE)
    ax.text(2.0, -1.0, "leapfrog: explicit, second order", ha="center", fontsize=17)
    ax.set_xlim(-0.5, 4.8)
    ax.set_ylim(-1.3, 2.7)

    # centre: amplification factors
    ax = fig.add_axes([0.37, 0.12, 0.28, 0.80])
    ax.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=GREY, lw=1.5)
    ax.axhline(0, color=GREY, lw=1)
    ax.axvline(0, color=GREY, lw=1)
    for a in (0.6, 2.0, 3.4):
        r = roots(a)
        ax.plot(r.real, r.imag, "o", color=BLUE, ms=10, zorder=5)
    ax.plot([-1], [0], "o", color=GREY, ms=11, zorder=5)
    r = roots(4.6).real
    ax.plot(r, [0, 0], "o", color=RED, ms=10, zorder=5)
    arc = np.linspace(0.58 * np.pi, 0.90 * np.pi, 40)
    ax.plot(1.15 * np.cos(arc), 1.15 * np.sin(arc), color=BLUE, lw=1.4)
    arrow(ax, (1.15 * np.cos(arc[-2]), 1.15 * np.sin(arc[-2])), (1.15 * np.cos(arc[-1]), 1.15 * np.sin(arc[-1])),
          BLUE, lw=1.4, ms=14)
    ax.text(0.28, 1.22, r"$\Delta t^2\lambda<4$: $|\xi|=1$", color=BLUE, fontsize=16)
    ax.text(-1.08, 0.10, r"$\Delta t^2\lambda=4$", color=GREY, fontsize=15, ha="right")
    ax.text(-2.55, -0.80, r"$\Delta t^2\lambda>4$: $|\xi|>1$", color=RED, fontsize=16, ha="left")
    ax.text(1.42, -0.12, r"Re $\xi$", fontsize=15, va="top")
    ax.text(0.06, 1.55, r"Im $\xi$", fontsize=15)
    ax.set_xlim(-2.6, 1.7)
    ax.set_ylim(-1.5, 1.75)
    ax.axis("off")
    ax.text(-0.45, -1.95, r"$\Delta t<2/\sqrt{\lambda_{\max}}$", ha="center", fontsize=21)

    # right: field amplitude of one mode, stable vs unstable (W is conserved in both; positive only below CFL)
    ax = fig.add_axes([0.70, 0.18, 0.28, 0.66])
    n = 40
    amp_s, ws = leapfrog_mode(3.0, n)
    amp_u, wu = leapfrog_mode(4.05, n)
    ax.semilogy(amp_s, color=BLUE, lw=2.2, marker="o", ms=3)
    ax.semilogy(amp_u, color=RED, lw=2.6)
    ax.set_ylim(0.3, 2e5)
    ax.minorticks_off()
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_xlabel("time step $n$", fontsize=16)
    ax.set_ylabel("field norm (log scale)", fontsize=16)
    ax.text(n * 0.5, 4.0, "below CFL: bounded\n($W$ conserved and positive)", color=BLUE, fontsize=14, ha="center")
    ax.text(n * 0.03, 2.2e4, "above CFL: grows\n($W$ conserved but indefinite)", color=RED, fontsize=14)
    fig.savefig(out / "fig4_4_leapfrog_cfl.png", facecolor="white")
    plt.close(fig)
    return ws, wu, amp_u


# --------------------------------------------------------------------------- 4.6
def circumcentre(a, b, c):
    a, b, c = map(np.asarray, (a, b, c))
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
    uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
    return np.array([ux, uy])


def whitney_pattern(seed=3, n=7):
    """Edge-edge coupling of the Galerkin (Whitney) mass matrix on a jittered triangle mesh."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0, 1, n)
    pts = np.array([(x, y) for x in g for y in g]) + rng.uniform(-0.03, 0.03, (n * n, 2))
    tri = Delaunay(pts).simplices
    edges = {}
    for t in tri:
        for i, j in ((0, 1), (1, 2), (0, 2)):
            edges.setdefault(tuple(sorted((t[i], t[j]))), len(edges))
    order = np.argsort([min(e) * 1000 + max(e) for e in edges])
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    m = np.zeros((len(edges), len(edges)), bool)
    for t in tri:
        ids = [rank[edges[tuple(sorted((t[i], t[j])))]] for i, j in ((0, 1), (1, 2), (0, 2))]
        for p in ids:
            for q in ids:
                m[p, q] = True
    return m


def fig4_6(out):
    fig = plt.figure(figsize=(W, H))
    m = whitney_pattern()
    ax = fig.add_axes([0.02, 0.36, 0.13, 0.42])
    ax.spy(m, markersize=1.3, color=GREY)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Galerkin FEM $\\mathbf{M}_\\varepsilon$", fontsize=16, pad=10)
    ax.text(0.5, -0.12, "linear solve\nevery step", transform=ax.transAxes, ha="center", va="top", fontsize=16)
    ax = fig.add_axes([0.215, 0.36, 0.13, 0.42])
    ax.spy(np.eye(m.shape[0], dtype=bool), markersize=1.3, color=GREEN)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("FIT (lumped) $\\mathbf{M}_\\varepsilon$", fontsize=16, pad=10)
    ax.text(0.5, -0.12, "no solver:\nexplicit update", transform=ax.transAxes, ha="center", va="top", fontsize=16)
    fig.text(0.1825, 0.50, "mass\nlumping\n(circum-\ncentric\ndual)", ha="center", va="top", fontsize=11)
    fig.add_artist(FancyArrowPatch((0.157, 0.53), (0.208, 0.53), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=18, color=GREY, lw=1.6))
    fig.text(0.175, 0.90, "Mass matrix", ha="center", fontsize=20, weight="bold")
    fig.add_artist(plt.Line2D([0.355, 0.355], [0.05, 0.95], transform=fig.transFigure, color=LGREY, lw=1.5))
    fig.add_artist(plt.Line2D([0.665, 0.665], [0.05, 0.95], transform=fig.transFigure, color=LGREY, lw=1.5))

    # centre: well-centred vs obtuse pair sharing edge AB
    fig.text(0.51, 0.90, "Well-centred mesh?", ha="center", fontsize=20, weight="bold")
    A, B = np.array([0.0, 0.0]), np.array([2.0, 0.0])
    cases = [((1.0, 1.3), (1.0, -1.3), [0.37, 0.48, 0.25, 0.38], "dual length > 0: stable", GREEN),
             ((1.0, 0.35), (1.0, -1.4), [0.37, 0.07, 0.25, 0.38], "dual length < 0: indefinite", RED)]
    for C, D, rect, lab, col in cases:
        ax = fig.add_axes(rect)
        bare(ax)
        C, D = np.array(C), np.array(D)
        ax.add_patch(Polygon([A, B, C], closed=True, fc=LBLUE, ec=BLUE, lw=2))
        ax.add_patch(Polygon([A, B, D], closed=True, fc="white", ec=BLUE, lw=2))
        cc1, cc2 = circumcentre(A, B, C), circumcentre(A, B, D)
        hn = 0.5 * 2.0 * sum(1 / np.tan(np.arccos(np.dot(A - P, B - P) / np.linalg.norm(A - P) / np.linalg.norm(B - P)))
                             for P in (C, D))
        assert np.isclose(hn, cc1[1] - cc2[1]), (hn, cc1, cc2)  # signed dual length = h_e formula
        arrow(ax, cc1, cc2, col if col == RED else ORANGE, lw=2.6, ms=18, z=5)
        for cc in (cc1, cc2):
            ax.add_patch(Circle(cc, 0.06, color=ORANGE, zorder=6))
        ax.text(cc1[0] - 0.12, cc1[1], r"$c_1$", ha="right", va="center", fontsize=15, color=ORANGE)
        ax.text(cc2[0] - 0.12, cc2[1], r"$c_2$", ha="right", va="center", fontsize=15, color=ORANGE)
        if col == RED:
            arrow(ax, (2.25, 1.15), (2.25, 0.35), GREY, lw=1.4, ms=14, ls="--")
            ax.text(2.35, 0.75, "expected\ndirection", fontsize=11, color=GREY, va="center")
        head, tail = lab.split(": ")
        ax.text(2.15, -0.75 if col == RED else 0.0, head + ":\n" + tail, fontsize=15, color=col, ha="left",
                va="center")
        ax.set_xlim(-0.3, 3.9)
        ax.set_ylim(-1.6, 1.6)
    fig.text(0.51, 0.475, r"$c_1$: circumcentre of the shaded triangle", ha="center", fontsize=12, color=GREY)
    fig.text(0.51, 0.04, "remedy: barycentric or Whitney Hodge", ha="center", fontsize=14, color=GREY)

    # right: time vs frequency domain
    fig.text(0.83, 0.90, "Time vs frequency domain", ha="center", fontsize=20, weight="bold")
    t = np.linspace(-1, 1, 1024)
    pulse = np.exp(-(t / 0.08) ** 2) * np.cos(2 * np.pi * 9 * t)
    spec = np.abs(np.fft.rfft(pulse))[:60]
    for rect, x, y in (([0.69, 0.56, 0.11, 0.24], t, pulse), ([0.86, 0.56, 0.11, 0.24], np.arange(60), spec)):
        ax = fig.add_axes(rect)
        ax.plot(x, y, color=BLUE, lw=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.745, 0.53, "time", ha="center", fontsize=13)
    fig.text(0.915, 0.53, "frequency", ha="center", fontsize=13)
    fig.add_artist(FancyArrowPatch((0.805, 0.70), (0.85, 0.70), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=18, color=GREY, lw=1.6))
    fig.text(0.827, 0.73, "FFT", ha="center", fontsize=14)
    fig.text(0.83, 0.47, "FIT: one run, many frequencies", ha="center", fontsize=16)
    ax = fig.add_axes([0.70, 0.13, 0.27, 0.24])
    for f in (0.2, 0.4, 0.6, 0.8):
        ax.plot([f, f], [0, 0.62], color=BLUE, lw=2.2)
        ax.add_patch(Rectangle((f - 0.035, 0.68), 0.07, 0.22, fc="white", ec=GREY, lw=1.2))
        for k in range(5):
            ax.add_patch(Rectangle((f - 0.035 + k * 0.014, 0.88 - (k + 1) * 0.044), 0.014, 0.044, color=GREY))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("frequency", fontsize=13)
    fig.text(0.83, 0.04, "FEM: one solve per frequency", ha="center", fontsize=16)
    fig.savefig(out / "fig4_6_fit_vs_fem.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fig4_1(out)
    fig4_2(out)
    ws, wu, amp_u = fig4_4(out)
    fig4_6(out)
    print("4.4 check: W spread stable", float(np.ptp(ws) / ws[0]), "unstable", float(np.ptp(wu[:12]) / abs(wu[0])),
          "| unstable amplitude growth over 40 steps", float(amp_u[-1]))
    print("4.4 roots:", {a: np.round(roots(a), 4).tolist() for a in (0.6, 2.0, 3.4, 4.0, 4.6)})
