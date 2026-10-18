"""Same circular cavity on four grids: Wakis staircase, Wakis conformal (cut cells),
body-fitted O-grid (NFIT) and Delaunay triangles (Voronoi-dual FIT)."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nfit2d as N
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import trifit2d as T
from matplotlib.collections import LineCollection, PolyCollection
from scipy.sparse.linalg import eigsh

import wakis
from wakis.conformal import ConformalGeometry

HERE = os.path.dirname(os.path.abspath(__file__))
a_w = 0.05  # Wakis cavity radius [m]; plotted in units of the radius


def wakis_grid(n):
    h = 2.2 * a_w / n
    cyl = pv.Cylinder(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        radius=a_w,
        height=20 * h,
        resolution=1440,
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
        stl_solids={"cav": cyl},
        stl_materials={"cav": "vacuum"},
        conformal=True,
        verbose=0,
    )
    return grid, h


def circle(ax):
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="tab:orange", lw=2)


def frame(ax, title):
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


n = 32
grid, h = wakis_grid(n)
x = grid.x / a_w
y = grid.y / a_w
staircase = wakis.SolverFIT3D(
    grid,
    bc_low=["pec", "pec", "periodic"],
    bc_high=["pec", "pec", "periodic"],
    bg="pec",
    use_stl=True,
    verbose=0,
)
ieps_z = staircase.ieps[:, :, 0, "z"]  # Ez sits on the xy nodes
geo = ConformalGeometry(grid, background_pec=True, mode="pec")
nx, ny = len(x) - 1, len(y) - 1
cell_area = np.outer(np.diff(grid.x), np.diff(grid.y))
open_frac = 1 - geo.pec_area[2][:nx, :ny, 0] / cell_area

fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)

# (a) Wakis staircase
ax = axes[0, 0]
X, Y = np.meshgrid(x[:-1], y[:-1], indexing="ij")
inside = ieps_z[: len(x) - 1, : len(y) - 1] != 0
cells_xy = [
    [(x[i], y[j]), (x[i + 1], y[j]), (x[i + 1], y[j + 1]), (x[i], y[j + 1])]
    for i in range(nx)
    for j in range(ny)
]
ax.add_collection(
    PolyCollection(cells_xy, facecolors="none", edgecolors="k", linewidths=0.3)
)
ax.plot(X[~inside], Y[~inside], ".", color="tab:red", ms=2.5, label="PEC Ez node")
ax.plot(X[inside], Y[inside], ".", color="tab:blue", ms=2.5, label="free Ez node")
circle(ax)
ax.legend(loc="lower left", fontsize=8)
frame(
    ax,
    f"Wakis FIT, Cartesian staircase\n{n}x{n} grid, h = {h / a_w:.3f} a, "
    f"{int(inside.sum())} free nodes",
)

# (b) Wakis conformal cut cells
ax = axes[0, 1]
pc = PolyCollection(
    cells_xy, array=open_frac.ravel(), cmap="viridis", edgecolors="k", linewidths=0.3
)
pc.set_clim(0, 1)
ax.add_collection(pc)
circle(ax)
fig.colorbar(pc, ax=ax, shrink=0.6, label="open (vacuum) fraction of each cell face")
cut = ((open_frac > 1e-6) & (open_frac < 1 - 1e-6)).sum()
frame(
    ax,
    f"Wakis FIT, conformal (cut cells, PBA-style)\nsame Cartesian grid, {cut} partially filled cells",
)

# (c) O-grid
ax = axes[1, 0]
nodes, cells, bnd = N.ogrid(1.0, 12)
ax.add_collection(
    PolyCollection(nodes[cells], facecolors="none", edgecolors="k", linewidths=0.4)
)
circle(ax)
ax.plot(*nodes[bnd].T, ".", color="tab:red", ms=3)
frame(
    ax,
    f"Body-fitted O-grid, nonorthogonal FIT\n{len(cells)} quadrilaterals, nodes on the circle",
)

# (d) triangles with Voronoi dual
ax = axes[1, 1]
tnodes, tri, tb = T.disk_mesh(1.0, 0.1)
ax.add_collection(
    PolyCollection(tnodes[tri], facecolors="none", edgecolors="k", linewidths=0.4)
)
# Voronoi dual edges: circumcentre to circumcentre across each interior edge
A, B, C = (tnodes[tri[:, k]] for k in range(3))
d = 2 * (
    A[:, 0] * (B[:, 1] - C[:, 1])
    + B[:, 0] * (C[:, 1] - A[:, 1])
    + C[:, 0] * (A[:, 1] - B[:, 1])
)
ux = (
    (A**2).sum(1) * (B[:, 1] - C[:, 1])
    + (B**2).sum(1) * (C[:, 1] - A[:, 1])
    + (C**2).sum(1) * (A[:, 1] - B[:, 1])
) / d
uy = (
    (A**2).sum(1) * (C[:, 0] - B[:, 0])
    + (B**2).sum(1) * (A[:, 0] - C[:, 0])
    + (C**2).sum(1) * (B[:, 0] - A[:, 0])
) / d
cc = np.stack([ux, uy], 1)
edge_tris = {}
for t_idx, t in enumerate(tri):
    for k in range(3):
        e = tuple(sorted((t[(k + 1) % 3], t[(k + 2) % 3])))
        edge_tris.setdefault(e, []).append(t_idx)
segs = [cc[ts] for ts in edge_tris.values() if len(ts) == 2]
ax.add_collection(LineCollection(segs, colors="tab:green", linewidths=0.6, alpha=0.8))
circle(ax)
ax.plot(*tnodes[tb].T, ".", color="tab:red", ms=3)
frame(
    ax,
    f"Delaunay triangles + Voronoi dual (green), triangular FIT\n{len(tri)} triangles, "
    "dual edges perpendicular to primary edges",
)
fig.suptitle(
    "Same circular PEC cavity on four FIT grids (orange: true boundary)", fontsize=13
)
fig.savefig(os.path.join(HERE, "grids_same_geometry.png"), dpi=150)
plt.close(fig)

# Convergence including triangles, and stable time steps
ref = N.exact_k(1.0)
res = {"staircase": [], "ogrid": [], "tri": []}
for nn in (8, 16, 32, 64):
    nodes, cells, bnd = N.cartesian_staircase(1.0, nn)
    k, *_ = N.modes(nodes, cells, bnd, n_modes=2)
    res["staircase"].append((np.pi * nn * nn, abs(k[0] - ref[0]) / ref[0]))
for nn in (4, 8, 16, 32, 64):
    nodes, cells, bnd = N.ogrid(1.0, nn)
    k, *_ = N.modes(nodes, cells, bnd, n_modes=2)
    res["ogrid"].append((len(cells), abs(k[0] - ref[0]) / ref[0]))
for hh in (0.2, 0.1, 0.05, 0.025, 0.0125):
    tnodes, tri, tb = T.disk_mesh(1.0, hh)
    k, *_ = T.modes(tnodes, tri, tb, n_modes=2)
    res["tri"].append(
        (len(tri) / 2, abs(k[0] - ref[0]) / ref[0])
    )  # 2 triangles ~ 1 quad
w = json.load(open(os.path.join(HERE, "wakis_circle.json")))

fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
ax.loglog(*np.array(res["staircase"]).T, "s--", label="Cartesian staircase (prototype)")
ax.loglog(
    [p[1] for p in w["staircase"]],
    [p[2] for p in w["staircase"]],
    "^:",
    label="Wakis staircase",
)
ax.loglog(*np.array(res["ogrid"]).T, "o-", label="O-grid, nonorthogonal FIT")
ax.loglog(
    *np.array(res["tri"]).T,
    "D-",
    label="Delaunay triangles, Voronoi FIT (per 2 triangles)",
)
ax.loglog(
    [p[1] for p in w["conformal"]],
    [p[2] for p in w["conformal"]],
    "v-",
    label="Wakis conformal (cut cells)",
)
ax.set_xlabel("cells inside the cavity (quadrilateral equivalent)")
ax.set_ylabel("TM01 relative frequency error")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Circular PEC cavity: convergence of the four grid types")
fig.savefig(os.path.join(HERE, "convergence_all.png"), dpi=150)
plt.close(fig)


def dt_ratio(K, dual, free, n_cells_quad):
    Di = 1 / dual[free]
    lmax = eigsh(
        sp.diags(np.sqrt(Di)) @ K[free][:, free] @ sp.diags(np.sqrt(Di)),
        k=1,
        which="LA",
    )[0][0]
    return 2 / np.sqrt(lmax) / np.sqrt(np.pi / n_cells_quad)


nodes, cells, bnd = N.ogrid(1.0, 16)
K, dual, _, _ = N.assemble(nodes, cells)
r_o = dt_ratio(K, dual, np.flatnonzero(~bnd), len(cells))
tnodes, tri, tb = T.disk_mesh(1.0, 0.05)
K, dual, _, _ = T.assemble(tnodes, tri)
r_t = dt_ratio(K, dual, np.flatnonzero(~tb), len(tri) / 2)
print(json.dumps({k: [[int(c), float(e)] for c, e in v] for k, v in res.items()}))
print(f"stable dt*c/h_avg: O-grid {r_o:.2f}, triangles {r_t:.2f} (2D Yee 0.71)")
