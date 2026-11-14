"""
Figures of the tetrahedral-FIT port study (docs/img/unstructured_ports3d_*.png).

Usage: python make_figures.py [results.json] [out_dir]
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from scipy.constants import c  # noqa: E402

from wakis import figstyle  # noqa: E402
from wakis.unstructured.ports3d import (  # noqa: E402
    PortPlane,
    coax_section,
    edge_path,
    extrude_section,
    rectangle_section,
)
from wakis.unstructured.solver3d import UnstructuredFIT3D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "results.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "docs", "img")
A, B = 22.86e-3, 10.16e-3
F_X = np.linspace(8e9, 12e9, 41)
F_C = np.linspace(1e9, 8e9, 36)
MM = 1e3


def _mesh_panel(ax, nodes, tris, tags, pec_tags):
    P = nodes[tris] * MM
    pec = np.isin(tags, pec_tags)
    ax.add_collection(PolyCollection(P[pec], facecolor="#b8bec8", edgecolor="#6b7280", lw=0.3))
    ax.add_collection(PolyCollection(P[~pec], facecolor="white", edgecolor="#9aa3ad", lw=0.4))
    ax.set_aspect("equal")
    ax.autoscale_view()


def _arrows(ax, plane, phi, color):
    m2 = plane.mesh2d
    E = m2.edges_to_cell_vectors(phi)
    cen = m2.nodes[m2.cells].mean(1)
    # mesh2d coordinates are in the port plane basis (u1, u2); map to x, y
    n = plane.normal
    u1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(n, u1)
    R = np.stack([u1[:2], u2[:2]], 1)  # xy = R @ (s, t)
    xy = cen @ R.T + plane.point[:2]
    Exy = E @ R.T
    ax.quiver(xy[:, 0] * MM, xy[:, 1] * MM, Exy[:, 0], Exy[:, 1], color=color,
              scale=None, width=0.004, headwidth=3.5)


def fig_setup():
    fig, ax = figstyle.new_figure(7.0, 3.2, ncols=3, gridspec_kw=dict(width_ratios=[1.5, 1.1, 1.0]))
    # (a) WR-90 section, TE10 discrete mode
    h = A / 8
    nodes, tris, tags = rectangle_section(A, B, h, wall=h / 2, seed=1)
    m = extrude_section(nodes, tris, tags, [0.0, h])
    s = UnstructuredFIT3D(m, {0: "pec"}, bc="pmc", massless_tol=1e-6)
    P = PortPlane(s, (0, 0, 0), (0, 0, 1))
    kc, phi = P.te_modes(1)
    _mesh_panel(ax[0], nodes, tris, tags, [0])
    _arrows(ax[0], P, phi[:, 0], "#c0392b")
    ax[0].set(xlabel="x (mm)", ylabel="y (mm)",
              title=f"(a) WR-90 TE$_{{10}}$, $k_{{c,h}}a/\\pi$={kc[0] * A / np.pi:.4f}")
    # (b) side view: extruded guide with ports and slab
    a = ax[1]
    zs = np.linspace(0, 40, 17)
    for zz in zs:
        a.plot([zz, zz], [0, A * MM], color="#9aa3ad", lw=0.4, ls="-", marker="none")
    a.add_patch(plt.Rectangle((15, 0), 10, A * MM, color="#9ec5e8", alpha=0.6, lw=0))
    a.text(20, A * MM / 2, "$\\varepsilon_r$ = 2.2", ha="center", va="center")
    for zz, lab in ((0, "port 1"), (40, "port 2")):
        a.plot([zz, zz], [0, A * MM], color="#c0392b", lw=2.2, ls="-", marker="none")
        a.text(zz, A * MM + 2.4, lab, ha="center", color="#c0392b")
    a.fill_between([0, 40], -1.4, 0, color="#b8bec8")
    a.fill_between([0, 40], A * MM, A * MM + 1.4, color="#b8bec8")
    a.text(20, -3.0, "L = 40 mm: prisms, 3 tets each", ha="center", va="top")
    a.set(xlim=(-4, 44), ylim=(-7, A * MM + 6), xlabel="z (mm)", ylabel="x (mm)",
          title="(b) side view, PEC walls grey")
    # (c) coax section, TEM mode and a lumped path
    nodes, tris, tags = coax_section(2e-3, 4.6e-3, 0.8e-3, wall=0.8e-3)
    m = extrude_section(nodes, tris, tags, [0.0, 1e-3])
    s = UnstructuredFIT3D(m, {0: "pec", 2: "pec"}, bc="pmc", massless_tol=1e-6)
    P = PortPlane(s, (0, 0, 0), (0, 0, 1))
    phi, _ = P.tem_mode()
    _mesh_panel(ax[2], nodes, tris, tags, [0, 2])
    _arrows(ax[2], P, phi, "#1f6fb2")
    e, _ = edge_path(s, (2e-3, 0, 0), (4.6e-3, 0, 0))
    seg = m.nodes[m.edges[e]][:, :, :2] * MM
    ax[2].add_collection(LineCollection(seg, colors="#e67e22", lw=2.6))
    ax[2].text(3.3, 0.55, "lumped\npath", color="#b9600f", ha="center", va="bottom")
    ax[2].set(xlabel="x (mm)", ylabel="y (mm)", title="(c) coax: TEM + lumped path")
    figstyle.save(fig, os.path.join(OUT, "unstructured_ports3d_setup.png"), close=True)


def _c(v):
    return np.asarray(v[0]) + 1j * np.asarray(v[1])


def fig_slab(res):
    rows = sorted([r for r in res if r["case"] == "slab"], key=lambda r: r["h_div"])
    fig, ax = figstyle.new_figure(7.0, 3.6, ncols=2)
    fg = F_X / 1e9
    r0 = rows[-1]
    ax[0].plot(fg, np.abs(_c(r0["S11c"])), "k--", lw=1.2, label="continuum |S11|")
    ax[0].plot(fg, np.abs(_c(r0["S21c"])), "k:", lw=1.2, label="continuum |S21|")
    for i, r in enumerate([rows[1], rows[-1]]):
        st = figstyle.series_style(i, with_line=False)
        ax[0].plot(fg, np.abs(_c(r["S11"])), ls="none", marker=st["marker"], color=st["color"], ms=3.5,
                   markevery=2, label=f"|S11| FIT, h = a/{r['h_div']}")
        ax[0].plot(fg, np.abs(_c(r["S21"])), ls="none", marker=st["marker"], mfc="none", color=st["color"],
                   ms=3.5, markevery=2, label=f"|S21| FIT, h = a/{r['h_div']}")
    ax[0].set(xlabel="frequency (GHz)", ylabel="|S|", title="(a) $\\varepsilon_r$ = 2.2 slab in WR-90")
    figstyle.spectrum_grid(ax[0])
    ax[0].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    for i, r in enumerate(rows):
        st = figstyle.series_style(i, with_marker=False)
        ax[1].semilogy(fg, np.abs(_c(r["S11"]) - _c(r["S11c"])), color=st["color"], ls=st["linestyle"],
                       label=f"h = a/{r['h_div']}")
    ax[1].set(xlabel="frequency (GHz)", ylabel="|S11 - S11 continuum|",
              title="(b) error vs continuum")
    figstyle.spectrum_grid(ax[1])
    ax[1].legend(fontsize=7, ncol=2)
    figstyle.save(fig, os.path.join(OUT, "unstructured_ports3d_slab.png"), close=True)


def fig_convergence(res):
    fig, ax = figstyle.new_figure(7.0, 4.2, ncols=2)
    wg = sorted([r for r in res if r["case"] == "wg"], key=lambda r: r["h"])
    sl = sorted([r for r in res if r["case"] == "slab"], key=lambda r: r["h"])
    cx = sorted([r for r in res if r["case"] == "coax"], key=lambda r: r["h"])
    lam = c / 12e9
    series = [
        ("TE$_{10}$ $|k_{c,h}/k_c - 1|$", [r["h"] for r in wg], [abs(r["kc_rel_err"]) for r in wg]),
        ("guide: max |arg S21 + $\\beta L$| (rad)", [r["h"] for r in wg],
         [np.abs(r["phase_err_cont"]).max() for r in wg]),
        ("slab: max |S11 - S11$_c$|", [r["h"] for r in sl], [r["s11_err_cont"] for r in sl]),
        ("coax: $|Z_{0,h}/Z_0 - 1|$", [r["h"] for r in cx], [abs(r["z0_rel_err"]) for r in cx]),
        ("coax load: max |Γ - Γ$_c$|", [r["h"] for r in cx], [r["g_err_cont"] for r in cx]),
    ]
    for i, (lab, h, e) in enumerate(series):
        st = figstyle.series_style(i)
        ax[0].loglog(np.array(h) / lam, e, color=st["color"], ls=st["linestyle"], marker=st["marker"],
                     ms=4, label=lab)
    hh = np.array([0.03, 0.15])
    ax[0].loglog(hh, 0.6 * (hh / 0.15) ** 2, "k-", lw=0.8)
    ax[0].text(0.07, 0.6 * (0.07 / 0.15) ** 2 * 0.5, "$h^2$", ha="center")
    ax[0].set(xlabel="h / $\\lambda_{0}$ (12 GHz)", ylabel="error vs continuum",
              title="(a) discretisation error")
    figstyle.spectrum_grid(ax[0])
    ax[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    floors = [
        ("guide |S11| floor", [r["h"] for r in wg], [r["s11_floor"] for r in wg]),
        ("guide |S21 - ladder|", [r["h"] for r in wg], [r["s21_vs_ladder"] for r in wg]),
        ("slab max |S - ladder|", [r["h"] for r in sl], [r["s_vs_ladder"] for r in sl]),
        ("coax |Γ - ladder|", [r["h"] for r in cx], [r["g_vs_ladder"] for r in cx]),
        ("slab unitarity", [r["h"] for r in sl], [r["unitarity"] for r in sl]),
    ]
    for i, (lab, h, e) in enumerate(floors):
        st = figstyle.series_style(i)
        ax[1].loglog(np.array(h) / lam, np.maximum(e, 1e-16), color=st["color"], ls=st["linestyle"],
                     marker=st["marker"], ms=4, label=lab)
    ax[1].set(xlabel="h / $\\lambda_{0}$ (12 GHz)", ylabel="deviation", ylim=(1e-15, 1e-3),
              title="(b) vs discrete ladder (exact)")
    figstyle.spectrum_grid(ax[1])
    ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    for a in ax:
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    figstyle.save(fig, os.path.join(OUT, "unstructured_ports3d_convergence.png"), close=True)


def fig_lumped(res):
    r = [x for x in res if x["case"] == "lumped"][0]
    fig, ax = figstyle.new_figure(7.0, 3.4, ncols=2)
    fg = F_C / 1e9
    td, fd = _c(r["S11_td"]), _c(r["S11_fd"])
    ax[0].plot(fg, np.abs(fd), "k-", lw=1.2, label="frequency-domain solve")
    ax[0].plot(fg, np.abs(td), ls="none", marker="o", ms=3.5, color="#d55e00", label="time domain (DTFT)")
    ax0b = ax[0].twinx()
    ax0b.semilogy(fg, np.abs(td - fd), color="#0072b2", lw=1.0, ls="-", marker="none")
    ax0b.set_ylabel("|S11 TD - FD|", color="#0072b2")
    ax[0].set(xlabel="frequency (GHz)", ylabel="|S11| (R = 50 Ω)",
              title="(a) radial lumped ports on 10 mm coax")
    ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    f = np.array(r["f_dc"])
    zr, zi = np.array(r["zin_dc"][0]), np.array(r["zin_dc"][1])
    ax[1].loglog(f, np.abs(zr - 75.0), "o-", ms=3.5, color="#009e73", label="|Re Z$_{in}$ - R$_L$|")
    ax[1].loglog(f, np.abs(zi), "s-", ms=3.5, color="#cc79a7", label="Im Z$_{in}$")
    ax[1].set(xlabel="frequency (Hz)", ylabel="Ω", title="(b) DC limit: Z$_{in}$ → R$_L$ = 75 Ω")
    figstyle.spectrum_grid(ax[1])
    ax[1].legend(fontsize=7)
    figstyle.save(fig, os.path.join(OUT, "unstructured_ports3d_lumped.png"), close=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with figstyle.apply_style():
        fig_setup()
        if os.path.exists(RES):
            res = json.load(open(RES))
            fig_slab(res)
            fig_convergence(res)
            fig_lumped(res)
