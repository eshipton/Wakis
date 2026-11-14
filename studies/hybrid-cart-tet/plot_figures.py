"""
Figures of the hybrid Cartesian / tetrahedral FIT study.

Reads results_{A,B,C,D,E}*.json and builds (Agg backend):

* fig_hybrid_mesh.png       setup: xz cut through the PEC-sphere hybrid mesh
                            and the triangulated patch face over the Cartesian
                            squares (hanging edges highlighted);
* fig_hybrid_waveguide.png  TE10 |R| envelope over patch lengths vs cells per a,\n                            and |R| vs patch length;
* fig_hybrid_cavity.png     PEC-cavity resonance errors vs h;
* fig_hybrid_sphere.png     PEC-sphere scattered-field error vs N and vs
                            the number of unknowns (equal-cost view);
* fig_hybrid_field.png      Re E_x,s on the xz plane: hybrid vs Mie;
* fig_hybrid_energy.png     energy history and the lambda_max bounds.

The mesh and field figures solve one small case (N = 10, about 2 GB): run on
Batch. Usage: python studies/hybrid-cart-tet/plot_figures.py
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 150, "figure.dpi": 100})  # fmt: skip

COL = {"cartesian": "0.2", "staircase": "0.2", "kuhn": "C0", "kuhn_r1": "C0", "kuhn_r2": "C9",
       "delaunay": "C1", "delaunay_r1": "C1", "delaunay_r2": "C3", "hybrid": "C1",
       "hybrid_fine": "C3", "tet": "C2", "floor": "0.6", "kuhn_nearest": "C4",
       "kuhn_inject": "C5"}  # fmt: skip


def load(part):
    """All results of a part (several Batch jobs), one entry per (N, kind)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, f"results_{part}*.json"))):
        for r in json.load(open(p))["results"]:
            out[(r["N"], r.get("kind"))] = r
    return [out[k] for k in sorted(out, key=lambda k: (k[0], str(k[1])))]


def guide(ax, x, y0, order, label):
    x = np.asarray(x, float)
    ax.loglog(x, y0 * (x / x[0]) ** (-order), "k:", lw=1, label=label)


def fig_waveguide():
    A, E = load("A"), load("E")
    if not E:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axs[0]
    labels = {"kuhn": "Kuhn patch, $h_t = h$", "delaunay": "Delaunay patch, $h_t \\approx h$",
              "kuhn_nearest": "mutation: nearest-edge trace"}  # fmt: skip
    for base, lab in labels.items():
        Ns = sorted({r["N"] for r in E if r["base"] == base})
        env = [max(r["R"] for r in E if r["base"] == base and r["N"] == N) for N in Ns]
        ax.loglog(Ns, env, "o-" if base != "kuhn_nearest" else "s--", color=COL[base], label=lab, ms=4)
        if base == "kuhn":
            guide(ax, Ns, env[0] * 1.5, 2, "$h^2$")
    fl = sorted([r for r in A if r["kind"] == "floor"], key=lambda r: r["N"])
    if fl:
        ax.loglog([r["N"] for r in fl], [r["R"] for r in fl], "x:", color=COL["floor"],
                  label="no patch: fit/absorber floor")  # fmt: skip
    ax.set_xlabel("Cartesian cells per broad wall $a$ (h = a/N)")
    ax.set_ylabel("envelope $\\max_L |R(L)|$")
    ax.set_title("(a) TE$_{10}$ reflection of a vacuum patch, $\\lambda = 1.4a$", fontsize=10)
    Ns = sorted({r["N"] for r in E})
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.legend(fontsize=7, loc="lower left")
    ax = axs[1]
    for i, N in enumerate(Ns):
        rows = sorted([r for r in E if r["base"] == "kuhn" and r["N"] == N], key=lambda r: r["L"])
        ax.semilogy([r["L"] for r in rows], [r["R"] for r in rows], "o-", ms=3,
                    color=plt.cm.viridis(i / max(1, len(Ns) - 1)), label=f"N = {N}")  # fmt: skip
    ax.set_xlabel("patch length L along the guide [a]")
    ax.set_ylabel("$|R|$ (Kuhn patch)")
    ax.set_title("(b) front/back interference: zero near $L = \\lambda_g/2 = 0.98a$", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_hybrid_waveguide.png"))
    plt.close(fig)


def fig_cavity():
    R = load("B")
    if not R:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for kind, lab in (("cartesian", "pure Cartesian (same h)"), ("kuhn", "hybrid, Kuhn patch"),
                      ("delaunay", "hybrid, Delaunay patch")):  # fmt: skip
        rows = sorted([r for r in R if r["kind"] == kind], key=lambda r: r["N"])
        h = [1.0 / r["N"] for r in rows]
        e1 = [abs(r["rel_err"][0]) for r in rows]
        erms = [np.sqrt(np.mean(np.square(r["rel_err"]))) for r in rows]
        axs[0].loglog(h, e1, "o-", color=COL[kind], label=lab, ms=4)
        axs[1].loglog(h, erms, "o-", color=COL[kind], label=lab, ms=4)
    for ax, t in zip(axs, ("lowest mode, (m, n, p) = (1, 1, 0)", "rms over the 6 lowest modes")):
        rows = sorted([r for r in R if r["kind"] == "cartesian"], key=lambda r: r["N"])
        h = np.array([1.0 / r["N"] for r in rows])
        y0 = abs(rows[0]["rel_err"][0]) * 1.5
        ax.loglog(h, y0 * (h / h[0]) ** 2, "k:", lw=1, label="$h^2$")
        ax.set_xlabel("h [m]")
        ax.set_ylabel("relative frequency error")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("PEC cavity 1 x 0.75 x 0.5 m with a vacuum tetrahedral patch", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_hybrid_cavity.png"))
    plt.close(fig)


def fig_sphere():
    R = load("C")
    if not R:
        return
    labels = {"staircase": "Cartesian staircase", "hybrid": "hybrid, $h_t \\approx h$",
              "hybrid_fine": "hybrid, $h_t \\approx h/2$",
              "tet": "tetrahedra in the whole physical region"}  # fmt: skip
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    for kind, lab in labels.items():
        rows = sorted([r for r in R if r["kind"] == kind], key=lambda r: r["N"])
        if not rows:
            continue
        N = [r["N"] for r in rows]
        axs[0].loglog(N, [r["err_near"] for r in rows], "o-", color=COL[kind], label=lab, ms=4)
        axs[1].loglog(N, [r["err_far"] for r in rows], "o-", color=COL[kind], label=lab, ms=4)
        axs[2].loglog([r["n_free"] for r in rows], [r["err_far"] for r in rows], "o-",
                      color=COL[kind], label=lab, ms=4)  # fmt: skip
    axs[0].set_title("near zone: $a + h < |r|$, inside the patch box", fontsize=9)
    axs[1].set_title("far zone: Cartesian shell outside the box", fontsize=9)
    axs[2].set_title("far-zone error vs unknowns (equal cost)", fontsize=9)
    for ax in axs[:2]:
        ax.set_xlabel("Cartesian cells per wavelength N")
        ax.set_xticks([8, 10, 12, 16, 20])
        ax.set_xticklabels(["8", "10", "12", "16", "20"])
    axs[2].set_xlabel("number of unknowns (free edges)")
    for ax in axs:
        ax.set_ylabel("relative L2 error of $e_s$")
    axs[0].legend(fontsize=7)
    fig.suptitle("PEC sphere, ka = 2, plane wave: scattered edge voltages vs the Mie series",
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_hybrid_sphere.png"))
    plt.close(fig)


def fig_energy():
    R = load("D")
    if not R:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for r in R:
        if "energy" in r:
            E = np.asarray(r["energy"])
            n = np.arange(len(E)) * 20
            axs[0].plot(n, E / E[0] - 1, label=f"N = {r['N']} (Chebyshev mass solve)")
    axs[0].set_xlabel("time step n")
    axs[0].set_ylabel("$E^n / E^0 - 1$")
    axs[0].set_title("discrete energy, source off, PEC box", fontsize=10)
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[0].legend(fontsize=8)
    names = ["Cartesian region", "tet region", "hybrid", "staircase (same h)"]
    w = 0.38
    for i, r in enumerate(R):
        d = r["chebyshev"]
        vals = [d["lambda_cart"], d["lambda_tet"], d["lambda"],
                (2 / r["dt_cart_staircase"]) ** 2]  # fmt: skip
        axs[1].bar(np.arange(4) + (i - 0.5) * w, np.sqrt(vals) / 2, w,
                   label=f"N = {r['N']}")  # fmt: skip
    axs[1].set_xticks(range(4))
    axs[1].set_xticklabels(names, fontsize=8)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("$\\sqrt{\\lambda_{max}}/2 = 1/\\Delta t_{max}$ [1/s]")
    axs[1].set_title("stability: hybrid <= max(region bounds)", fontsize=10)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_hybrid_energy.png"))
    plt.close(fig)


def _tet_plane_cut(mesh, axis, value):
    """Polygons of the tetrahedra cut by the plane x_axis = value."""
    X = mesh.nodes
    d = X[mesh.cells][:, :, axis] - value
    cut = (d.min(1) < 0) & (d.max(1) > 0)
    polys = []
    for t in np.flatnonzero(cut):
        P, s = X[mesh.cells[t]], d[t]
        pts = []
        for i in range(4):
            for j in range(i + 1, 4):
                if s[i] * s[j] < 0:
                    u = s[i] / (s[i] - s[j])
                    pts.append(P[i] + u * (P[j] - P[i]))
        pts = np.array(pts)
        keep = [a for a in range(3) if a != axis]
        q = pts[:, keep]
        c = q.mean(0)
        q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
        polys.append(q)
    return polys


def fig_mesh_and_field():
    from matplotlib.collections import LineCollection, PolyCollection
    from scipy.constants import c as C0

    from wakis.hybrid.mie_pec import pec_sphere_field
    from wakis.hybrid.verification import pec_sphere_scattering

    N = 10
    out = pec_sphere_scattering(N, "hybrid", ka=2.0, gap=0.2, T=0.5, return_solver=True)
    s = out["solver"]
    tm = s.tet
    a, B, h = out["a"], out["B"], out["h"]
    v = s.block.nodes[0]
    Lp = B + int(np.ceil(0.2 / h)) * h
    # ---------------- mesh figure
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    ax = axs[0]
    ax.add_patch(plt.Rectangle((v[0], v[0]), v[-1] - v[0], v[-1] - v[0], color="0.9", zorder=0))
    ax.add_patch(plt.Rectangle((-Lp, -Lp), 2 * Lp, 2 * Lp, color="white", zorder=0))
    segs = []
    for x in v:
        for lo_, hi_ in ((v[0], -B), (B, v[-1])):
            segs += [[(x, lo_), (x, hi_)], [(lo_, x), (hi_, x)]]
        if abs(x) > B + 1e-9:
            segs += [[(x, -B), (x, B)], [(-B, x), (B, x)]]
    ax.add_collection(LineCollection(segs, colors="0.55", lw=0.5))
    polys = _tet_plane_cut(tm, 1, 1e-4)
    ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="C0", lw=0.35))
    ax.add_patch(plt.Circle((0, 0), a, fill=True, color="0.25", zorder=3))
    ax.add_patch(plt.Rectangle((-B, -B), 2 * B, 2 * B, fill=False, ec="C3", lw=1.6, zorder=4))
    ax.annotate("PEC sphere\n$ka = 2$", (0, 0), color="w", ha="center", va="center", zorder=5,
                fontsize=8)  # fmt: skip
    ax.text(-B, B + 0.04, "interface (box of whole Cartesian cells)", color="C3", fontsize=8,
            bbox=dict(fc="white", ec="none", pad=1), zorder=6)  # fmt: skip
    ax.text(v[0] + 0.02, v[-1] - 0.08, "complex-stretched absorber", fontsize=8)
    ax.annotate("", xy=(v[0] + 0.12, v[0] + 0.45), xytext=(v[0] + 0.12, v[0] + 0.1),
                arrowprops=dict(arrowstyle="->", color="C2", lw=1.5))  # fmt: skip
    ax.text(v[0] + 0.16, v[0] + 0.2, "$\\mathbf{k} \\parallel +z$,\n$\\mathbf{E}_i = \\hat x e^{ikz}$",
            fontsize=8, color="C2", bbox=dict(fc="0.9", ec="none", pad=1))  # fmt: skip
    ax.set_aspect("equal")
    ax.set_xlim(v[0], v[-1])
    ax.set_ylim(v[0], v[-1])
    ax.set_xlabel("x [m]  (wavelength 1 m)")
    ax.set_ylabel("z [m]")
    ax.set_title(f"(a) cut y = 0: Cartesian h = $\\lambda$/{N} + {tm.n_cells} tetrahedra", fontsize=9)
    # ---------------- face detail: x = -B
    ax = axs[1]
    X = tm.nodes
    bf = tm.boundary_faces
    F = X[tm.faces[bf]]
    on = np.all(np.abs(F[:, :, 0] + B) < 1e-9, axis=1)
    tri = [f[:, 1:] for f in F[on]]
    ax.add_collection(PolyCollection(tri, facecolors="#dbe9f6", edgecolors="C0", lw=0.9))
    vb = v[(v >= -B - 1e-9) & (v <= B + 1e-9)]
    sq = [[(x, vb[0]), (x, vb[-1])] for x in vb] + [[(vb[0], z), (vb[-1], z)] for z in vb]
    ax.add_collection(LineCollection(sq, colors="k", lw=2.0))
    ax.set_aspect("equal")
    # zoom on a corner of the face: 4 x 4 squares
    ax.set_xlim(vb[0] - 0.01, vb[4] + 0.01)
    ax.set_ylim(vb[0] - 0.01, vb[4] + 0.01)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("(b) patch face x = -B (corner, 4 x 4 squares): squares (black)\n"
                 "carry the free edges; every triangle edge (blue) hangs on\n"
                 "the Nedelec trace of its square", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_hybrid_mesh.png"))
    plt.close(fig)
    # ---------------- field figure: Re E_x,s on y = 0 (Whitney in tets, edges in Cartesian)
    k = 2 * np.pi
    g = np.linspace(-Lp + 0.5 * h, Lp - 0.5 * h, 121)
    GX, GZ = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([GX.ravel(), 0 * GX.ravel(), GZ.ravel()], 1)
    ex = np.full(len(pts), np.nan, complex)
    e_all = s.P @ s.solve_frequency(k * C0, fixed=(s.nEc + s.tet_fixed, -_inc(s, k)))
    inbox = np.all(np.abs(pts) < B - 1e-9, axis=1) & (np.linalg.norm(pts, axis=1) > a)
    if inbox.any():
        cell, W = tm.whitney_edge(pts[inbox])
        et = e_all[s.nEc :]
        ex[np.flatnonzero(inbox)] = np.einsum("nk,nk->n", W[:, :, 0], et[tm.cell_edges[cell]])
    # Cartesian: x-edges in the plane y = 0, bilinear in (x-node shift) -> nearest edge value / h
    blk = s.block
    out_box = ~np.all(np.abs(pts) <= B + 1e-9, axis=1)
    ec = e_all[: blk.edge_off[1]].reshape(blk.edge_shape[0])
    j0 = int(np.argmin(np.abs(blk.nodes[1])))
    ix = np.clip(np.searchsorted(blk.nodes[0], pts[out_box, 0]) - 1, 0, blk.nc[0] - 1)
    kz = np.clip(np.round((pts[out_box, 2] - blk.nodes[2][0]) / h).astype(int), 0, blk.nn[2] - 1)
    ex[out_box] = ec[ix, j0, kz] / h
    ref = pec_sphere_field(pts, k, a)[:, 0]
    ref[np.linalg.norm(pts, axis=1) < a] = np.nan
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    vmax = np.nanmax(np.abs(ref.real))
    for ax, data, t in ((axs[0], ex.real, f"hybrid FIT, N = {N}"), (axs[1], ref.real, "Mie series"),
                        (axs[2], np.abs(ex - ref), "|hybrid - Mie|")):  # fmt: skip
        im = ax.pcolormesh(g, g, data.reshape(GX.shape).T, cmap="RdBu_r" if "|" not in t else "magma",
                           vmin=-vmax if "|" not in t else 0, vmax=vmax if "|" not in t else 0.3 * vmax,
                           shading="nearest")  # fmt: skip
        ax.add_patch(plt.Rectangle((-B, -B), 2 * B, 2 * B, fill=False, ec="k", lw=1))
        ax.add_patch(plt.Circle((0, 0), a, fill=False, ec="k", lw=1))
        ax.set_aspect("equal")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        fig.colorbar(im, ax=ax, label="Re $E_{x,s}$ [V/m]" if "|" not in t else "[V/m]", shrink=0.85)
    fig.suptitle("PEC sphere ka = 2: scattered field on y = 0 (unit incident field)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_hybrid_field.png"))
    plt.close(fig)


def _inc(s, k):
    from wakis.hybrid.mie_pec import plane_wave_line_integrals

    tm = s.tet
    e = tm.edges[s.tet_fixed]
    return plane_wave_line_integrals(k, tm.nodes[e[:, 0]], tm.nodes[e[:, 1]])


if __name__ == "__main__":
    fig_waveguide()
    fig_cavity()
    fig_sphere()
    fig_energy()
    if os.environ.get("HYBRID_SKIP_SOLVE") != "1":
        fig_mesh_and_field()
    print("figures in", FIG)
