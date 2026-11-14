"""
Figures of the CAD-mesh / tetrahedral time-step study (from results/*.json).

    python plot_cadmesh.py          # writes docs/img/unstructured_cadmesh_*.png

The setup figure re-meshes the muon cavity and the layered pillbox (needs
gmsh, a few seconds).
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IMG = os.path.abspath(os.path.join(HERE, "..", "..", "docs", "img"))
COL = {"none": "#4C72B0", "gmsh": "#DD8452", "netgen": "#8C8C8C", "ours": "#55A868"}
LAB = {"none": "gmsh Delaunay (no opt.)", "gmsh": "gmsh optimizer",
       "netgen": "gmsh + Netgen optimizer", "ours": "Delaunay + improve_dt"}  # fmt: skip
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


def _slice_edges(mesh, axis=1, width=None):
    """Edges of the tetrahedra whose centroid lies within width/2 of the plane."""
    width = width or mesh.edge_length.mean()
    cen = mesh.nodes[mesh.cells].mean(1)
    sel = np.abs(cen[:, axis]) < width / 2
    e = np.unique(mesh.cell_edges[sel].ravel())
    return e, sel


def fig_setup():
    import sys

    sys.path.insert(0, HERE)
    from cadmesh_study import make_mesh

    from wakis.unstructured import UnstructuredFIT3D
    from wakis.unstructured.quality import element_stability

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1, 1.1]))
    # (a) muon cavity, cut y = 0, coloured by element time step
    m, info = make_mesh("muon", 15, "none")
    s = UnstructuredFIT3D(m, info.materials)
    st = element_stability(s)
    e, sel = _slice_edges(m)
    P = m.nodes[m.edges[e]] * 1e3
    ax[0].plot(P[:, :, 0].T, P[:, :, 2].T, color="#6B7280", lw=0.3, alpha=0.6)
    cen = m.nodes[m.cells[sel]].mean(1) * 1e3
    r = st["dt_T"][sel] / st["dt_max"]
    sc = ax[0].scatter(cen[:, 0], cen[:, 2], c=np.log10(r), s=6, cmap="viridis_r",
                       vmin=0, vmax=np.log10(30))  # fmt: skip
    w = st["worst"][0]
    cw = m.nodes[m.cells[w]].mean(0) * 1e3
    ax[0].annotate("worst element\n(dt_T = %.2f dt)" % (st["dt_T"][w] / st["dt_max"]),
                   xy=(cw[0], cw[2]), xytext=(-40, -75), fontsize=8,
                   arrowprops=dict(arrowstyle="->", lw=0.8))  # fmt: skip
    ax[0].text(-185, 55, "Be window (dropped: PEC)", fontsize=8)
    ax[0].text(-185, -65, "Be window (dropped: PEC)", fontsize=8)
    ax[0].text(120, 0, "Cu wall\n(dropped: PEC)", fontsize=8, ha="center")
    ax[0].set_aspect("equal")
    ax[0].set_ylim(-95, 80)
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("z [mm]")
    fig.colorbar(sc, ax=ax[0], shrink=0.7, label=r"$\log_{10}(\Delta t_T/\Delta t_{\max})$")
    ax[0].set_title("(a) muon-cavity STEP, vacuum cells, cpw 15:\n%d tets, cut y = 0"
                    % m.n_cells, fontsize=9, pad=12)  # fmt: skip
    # (b) layered pillbox: material tags
    m2, info2 = make_mesh("layered", 15, "none")
    e2, sel2 = _slice_edges(m2)
    P2 = m2.nodes[m2.edges[e2]] * 1e3
    ax[1].plot(P2[:, :, 0].T, P2[:, :, 2].T, color="#6B7280", lw=0.3, alpha=0.6)
    c2 = m2.nodes[m2.cells[sel2]].mean(1) * 1e3
    tg = m2.cell_tags[sel2]
    for t, col, lab in ((0, "#C44E52", r"$\varepsilon_r=4$ layer (h = %.1f mm)"
                         % (info2.sizes["diel"] * 1e3)),
                        (1, "#4C72B0", "vacuum (h = %.1f mm)" % (info2.sizes["vac"] * 1e3))):  # fmt: skip
        ax[1].scatter(c2[tg == t, 0], c2[tg == t, 2], s=6, color=col, label=lab)
    ax[1].legend(loc="upper center", fontsize=7, bbox_to_anchor=(0.5, -0.15))
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel("z [mm]")
    ax[1].set_title("(b) layered pillbox STEP, cpw 15 in the dielectric", fontsize=9)
    # (c) element dt vs radius ratio (pillbox cpw 20); elements carrying the dt-limiting mode
    b = [r for r in _load("bounds") if r["case"] == "pillbox" and r["cpw"] == 20
         and "dt_T_over_dt" in r]  # fmt: skip
    for r in b:
        x = np.array(r["radius_ratio"])
        y = np.array(r["dt_T_over_dt"])
        E = np.array(r["energy"])
        vor = r["hodge"] == "voronoi"
        ax[2].scatter(x, y, s=2, color="#4C72B0" if vor else "#DD8452", alpha=0.08,
                      marker="o" if vor else "^", rasterized=True)  # fmt: skip
        top = np.argsort(E)[::-1][: max(1, len(E) // 200)]
        ax[2].scatter(x[top], y[top], s=22 if vor else 18, marker="o" if vor else "^",
                      facecolor="#4C72B0" if vor else "#DD8452", edgecolor="k", lw=0.4,
                      label=f"{r['hodge']}: 0.5 % elements with most energy of the fastest mode")  # fmt: skip
    ax[2].axhline(1, color="k", lw=0.8)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("radius ratio 3 r_in / R")
    ax[2].set_ylabel(r"$\Delta t_T / \Delta t_{\max}$ (element bound / global)")
    ax[2].set_title("(c) pillbox cpw 20: element step vs shape\n(faint: all elements)", fontsize=9)
    ax[2].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "unstructured_cadmesh_setup.png"), dpi=170)
    plt.close(fig)


def fig_bounds():
    b = [r for r in _load("bounds") if "error" not in r]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = []
    for i, r in enumerate(b):
        labels.append(f"{r['case']}\ncpw {r['cpw']}" if r["cpw"] else r["case"].replace("_", " h="))
    uniq = list(dict.fromkeys(labels))
    for hd, mk, dx in (("voronoi", "o", -0.12), ("whitney", "^", 0.12)):
        for r, lab in zip(b, labels):
            if r["hodge"] != hd:
                continue
            x = uniq.index(lab) + dx
            lo = np.sqrt(r["lam_lower"] / r["lam_max"])
            hi = np.sqrt(r["lam_bound"] / r["lam_max"])
            ax[0].plot([x, x], [lo, hi], color="C0" if hd == "voronoi" else "C1", lw=2)
            ax[0].plot(x, hi, mk, color="C0" if hd == "voronoi" else "C1",
                       label=f"{hd}: element bound" if lab == uniq[0] else None)  # fmt: skip
            ax[0].plot(x, lo, "_", color="k", ms=10,
                       label="max K_ee/M_ee (lower)" if (lab == uniq[0] and hd == "voronoi") else None)  # fmt: skip
            if hd == "voronoi":
                ax[0].plot(x, np.sqrt(r["lam_gershgorin"] / r["lam_max"]), "x", color="C2",
                           label="Gershgorin" if lab == uniq[0] else None)  # fmt: skip
            ax[1].bar(x, r["share_worst"]["0.01"], width=0.22,
                      color="C0" if hd == "voronoi" else "C1",
                      label=f"{hd} Hodge" if lab == uniq[0] else None)  # fmt: skip
    ax[0].axhline(1, color="k", lw=0.8)
    ax[0].axhline(np.sqrt(6), color="C0", ls=":", lw=0.8)
    ax[0].text(len(uniq) - 0.5, np.sqrt(6) * 1.02, r"$\sqrt{6}$ (Voronoi guarantee)", ha="right", fontsize=7)
    ax[0].set_ylabel(r"$\sqrt{\lambda/\lambda_{\max}}$  $(=\Delta t_{\max}/\Delta t)$")
    ax[0].set_title("(a) bounds on the global time step (1 = exact)", fontsize=9)
    ax[1].set_ylabel("energy share of the worst 1 % elements")
    ax[1].set_title("(b) localization of the dt-limiting mode", fontsize=9)
    for a in ax:
        a.set_xticks(range(len(uniq)))
        a.set_xticklabels(uniq, fontsize=6.5, rotation=30)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "unstructured_cadmesh_bounds.png"), dpi=170)
    plt.close(fig)


def fig_improve():
    rs = _load("improve")
    cases = ["pillbox", "layered", "muon"]
    variants = ["none", "gmsh", "netgen", "ours"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    xt, xl = [], []
    for ci, case in enumerate(cases):
        for k, cpw in enumerate((10, 15, 20)):
            x0 = ci * 4 + k
            xt.append(x0)
            xl.append(f"{case}\n{cpw}" if k == 1 else f"{cpw}")
            for vi, v in enumerate(variants):
                r = next((r for r in rs if r["case"] == case and r["cpw"] == cpw
                          and r["variant"] == v), None)  # fmt: skip
                if r is None or "dt_max" not in r:
                    continue
                x = x0 + (vi - 1.5) * 0.2
                lab = LAB[v] if (ci == 0 and k == 0) else None
                ax[0].bar(x, r["courant_h"], width=0.2, color=COL[v], label=lab)
                base = next(q for q in rs if q["case"] == case and q["cpw"] == cpw
                            and q["variant"] == "none")  # fmt: skip
                if "dt_max" not in base:
                    continue
                if "err" in r:
                    ax[1].bar(x, np.sqrt(np.mean(np.square(r["err"]))), width=0.2,
                              color=COL[v], label=lab)  # fmt: skip
    ax[0].set_ylabel(r"$c\,\Delta t_{\max}/\bar h$")
    ax[0].set_title("(a) stable step (Voronoi Hodge, massless_tol 1e-2)", fontsize=9)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("rms relative frequency error")
    ax[1].set_title("(b) accuracy vs analytic (pillbox: 5 modes; layered: TM01)", fontsize=9)
    tm = _load("timing")
    for ci, case in enumerate(cases):
        b = next(t for t in tm if t["case"] == case and t["variant"] == "none")
        for vi, v in enumerate(("none", "ours")):
            t = next(t for t in tm if t["case"] == case and t["variant"] == v)
            ax[2].bar(ci * 4 + 1 + (vi - 0.5) * 0.4, t["t_per_ns"] / b["t_per_ns"], width=0.4,
                      color=COL[v])  # fmt: skip
            ax[2].text(ci * 4 + 1 + (vi - 0.5) * 0.4, t["t_per_ns"] / b["t_per_ns"] + 0.02,
                       "%.2f" % (t["t_per_ns"] / b["t_per_ns"]), ha="center", fontsize=7)  # fmt: skip
    ax[2].set_ylabel("wall time per simulated ns / baseline")
    ax[2].axhline(1, color="k", lw=0.8)
    ax[2].set_title("(c) serial cost of a fixed simulated time, cpw 20", fontsize=9)
    for a in ax[:2]:
        a.set_xticks(xt)
        a.set_xticklabels(xl, fontsize=7)
        a.set_xlabel("case / cells per wavelength")
    ax[2].set_xticks([ci * 4 + 1 for ci in range(3)])
    ax[2].set_xticklabels(cases)
    ax[2].set_xlabel("case (blue: Delaunay, green: + improve_dt)")
    ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "unstructured_cadmesh_improve.png"), dpi=170)
    plt.close(fig)


def _cart_modes(min_rel_amp=1e-2):
    out = []
    for fn in sorted(glob.glob(os.path.join(RES, "muon_cart_*mm.json"))):
        with open(fn) as f:
            r = json.load(f)
        a = np.array(r["amp"])
        keep = a > min_rel_amp * a.max()
        r["f_main"] = np.array(r["f_space"])[keep]
        out.append(r)
    return sorted(out, key=lambda r: -r["h_mm"])


def nearest(f, target):
    f = np.asarray(f)
    return f[np.argmin(np.abs(f - target))]


def muon_reference():
    """Cartesian TM010-like / second-mode reference: Richardson (order 2) of the two finest."""
    cart = _cart_modes()
    tet = _load("muon_tet")["eig"]
    guess = [r for r in tet if r["hodge"] == "whitney"][-1]["f"]
    ref = {}
    for j in (0, 1):
        h = np.array([r["h_mm"] for r in cart])
        f = np.array([nearest(r["f_main"], guess[j]) for r in cart])
        p = np.polyfit(h**2, f, 1)  # f = f* + C h^2
        ref[j] = dict(h=h, f=f, f_star=p[1])
    return ref


def fig_muon():
    tet = _load("muon_tet")
    ref = muon_reference()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for j, a in zip((0, 1), ax):
        fs = ref[j]["f_star"]
        a.plot(ref[j]["h"], ref[j]["f"] / fs - 1, "s-", color="k", label="Cartesian conformal-PEC FIT")
        for v, hd, col, mk in (("none", "voronoi", COL["none"], "o"),
                               ("ours", "voronoi", COL["ours"], "o"),
                               ("none", "whitney", "#8172B3", "^")):  # fmt: skip
            rr = sorted([r for r in tet["eig"] if r["variant"] == v and r["hodge"] == hd],
                        key=lambda r: r["h_mean"])  # fmt: skip
            h = np.array([r["h_mean"] for r in rr]) * 1e3
            f = np.array([nearest(r["f"], fs) for r in rr])
            a.plot(h, f / fs - 1, mk + "-", color=col,
                   label=f"tets, {hd}" + (" + improve_dt" if v == "ours" else ""))  # fmt: skip
        for r in tet["td"]:
            if r["variant"] == "ours":
                a.plot(r.get("h_mean", np.nan) * 1e3, nearest(r["f_space"], fs) / fs - 1, "*",
                       color=COL["ours"], ms=9)  # fmt: skip
        a.axhline(0, color="k", lw=0.6)
        a.set_xlabel("h [mm] (tets: mean edge; Cartesian: cell)")
        a.set_ylabel(r"$f/f^* - 1$")
        a.set_title(("(a) TM010-like" if j == 0 else "(b) second mode")
                    + r", $f^*$ = %.4f GHz (Cartesian, $h^2$ extrap.)" % (fs / 1e9), fontsize=9)  # fmt: skip
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "unstructured_cadmesh_muon.png"), dpi=170)
    plt.close(fig)


def main():
    os.makedirs(IMG, exist_ok=True)
    fig_bounds()
    fig_improve()
    fig_muon()
    fig_setup()


if __name__ == "__main__":
    main()
