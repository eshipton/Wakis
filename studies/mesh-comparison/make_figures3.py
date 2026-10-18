"""Dielectric rod in a PEC cavity: grids and TM/TE convergence (see README)."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import dielectric2d as D
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import PolyCollection

import wakis

HERE = os.path.dirname(os.path.abspath(__file__))
a, b, eps = 1.0, 0.5, 6.0

# ---- grids
fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
th = np.linspace(0, 2 * np.pi, 400)


def rings(ax):
    ax.plot(np.cos(th), np.sin(th), color="tab:orange", lw=2)
    ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color="tab:red", lw=2)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# Wakis conformal: Kottke-averaged permittivity of the z-edges (Ez) on its Cartesian grid
n, aw = 32, 0.05
h = 2.2 * aw / n
cyl = lambda r: pv.Cylinder(
    center=(0, 0, 0), direction=(0, 0, 1), radius=r, height=20 * h, resolution=1440
).triangulate()
grid = wakis.GridFIT3D(
    -n * h / 2,
    n * h / 2,
    -n * h / 2,
    n * h / 2,
    -h / 2,
    h / 2,
    n,
    n,
    1,
    stl_solids={"cav": cyl(aw), "rod": cyl(aw / 2)},
    stl_materials={"cav": "vacuum", "rod": [eps, 1.0, 0.0]},
    conformal=True,
    verbose=0,
)
solver = wakis.SolverFIT3D(
    grid,
    bc_low=["pec", "pec", "periodic"],
    bc_high=["pec", "pec", "periodic"],
    bg="pec",
    use_stl=True,
    verbose=0,
)
from scipy.constants import epsilon_0

ieps_x = solver.ieps[:, :, 0, "x"]
x, y = grid.x / aw, grid.y / aw
nx, ny = len(x) - 1, len(y) - 1
er = np.where(
    ieps_x[:nx, :ny] > 0,
    1 / (epsilon_0 * np.where(ieps_x[:nx, :ny] > 0, ieps_x[:nx, :ny], 1)),
    np.nan,
)
cells = [
    [(x[i], y[j]), (x[i + 1], y[j]), (x[i + 1], y[j + 1]), (x[i], y[j + 1])]
    for i in range(nx)
    for j in range(ny)
]
pc = PolyCollection(
    cells, array=er.ravel(), cmap="viridis", edgecolors="k", linewidths=0.25
)
pc.set_clim(1, eps)
axes[0].add_collection(pc)
rings(axes[0])
fig.colorbar(
    pc, ax=axes[0], shrink=0.7, label="effective eps_r of the Ex edges (Kottke average)"
)
axes[0].set_title(
    "Wakis conformal: Cartesian grid, interface cells\nget an averaged (diagonal) permittivity",
    fontsize=10,
)

for ax, (mesh, title) in zip(
    axes[1:],
    [
        (
            D.ogrid_rings(a, b, 12),
            "Body-fitted O-grid: a ring of nodes\non the rod interface",
        ),
        (
            D.tri_rings(a, b, 0.1),
            "Delaunay triangles: nodes on the\ninterface, each triangle one material",
        ),
    ],
):
    nodes, cl, rod, _, pec = mesh
    pc = PolyCollection(
        nodes[cl],
        array=np.where(rod, eps, 1.0),
        cmap="viridis",
        edgecolors="k",
        linewidths=0.4,
    )
    pc.set_clim(1, eps)
    ax.add_collection(pc)
    rings(ax)
    ax.set_title(f"{title} ({len(cl)} cells)", fontsize=10)
fig.suptitle(
    f"Dielectric rod (eps_r = {eps:g}, radius a/2) in a PEC cavity: the three grid types",
    fontsize=12,
)
fig.savefig(os.path.join(HERE, "rod_grids.png"), dpi=150)
plt.close(fig)

# ---- convergence
w = json.load(open(os.path.join(HERE, "wakis_rod.json")))
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True, sharey=True)
for ax, pol in zip(axes, ("TM", "TE")):
    ref = D.exact_k(a, b, eps, pol)
    for kind, maker, levels, style, label in (
        ("cart", D.cartesian, (8, 16, 32, 64), "s--", "prototype Cartesian staircase"),
        ("quad", D.ogrid_rings, (6, 12, 24, 48), "o-", "body-fitted O-grid (NFIT)"),
        (
            "tri",
            D.tri_rings,
            (0.2, 0.1, 0.05, 0.025),
            "D-",
            "Delaunay triangles (Voronoi FIT)",
        ),
    ):
        pts = []
        for lv in levels:
            mesh = maker(a, b, lv)
            k = D.modes(mesh, kind, eps, pol, 2)
            ncell = (
                mesh[3].sum()
                if kind == "cart"
                else len(mesh[1]) / (2 if kind == "tri" else 1)
            )
            pts.append((ncell, abs(k[0] - ref[0]) / ref[0]))
        ax.loglog(*np.array(pts).T, style, label=label)
    for key, style, label in (
        (f"{pol}_staircase", "^:", "Wakis staircase"),
        (f"{pol}_conformal", "v-", "Wakis conformal (Kottke / cut cells)"),
    ):
        p = sorted(w[key])
        ax.loglog([e[1] for e in p], [e[2] for e in p], style, label=label)
    xx = np.array([150, 15000])
    ax.loglog(xx, 2e-1 * (xx / 150) ** -0.5, "k:", lw=0.8)
    ax.loglog(xx, 2e-3 * (xx / 150) ** -1.0, "k:", lw=0.8)
    ax.text(2000, 6e-2, "1st order", fontsize=8)
    ax.text(2000, 2e-4, "2nd order", fontsize=8)
    ax.set_title(
        f"{pol}: E {'parallel to the rod (tangential, continuous)' if pol == 'TM' else 'in-plane, crossing the interface'}",
        fontsize=10,
    )
    ax.set_xlabel("cells inside the cavity (quad equivalent)")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("lowest-mode relative frequency error")
axes[1].legend(fontsize=8, loc="lower left")
fig.suptitle(
    f"Dielectric rod (eps_r = {eps:g}) in a PEC cavity: lowest TM and TE mode",
    fontsize=12,
)
fig.savefig(os.path.join(HERE, "rod_convergence.png"), dpi=150)
plt.close(fig)
print("ok")
