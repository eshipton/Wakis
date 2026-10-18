"""Figures for the nonorthogonal-FIT circular-cavity study (see README.md)."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nfit2d as N
import numpy as np
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
a = 1.0
ref = N.exact_k(a)


def draw(ax, nodes, cells, bnd, field=None, title=""):
    polys = nodes[cells]
    if field is not None:
        val = field[cells].mean(axis=1)
        pc = PolyCollection(
            polys, array=val, cmap="RdBu_r", edgecolors="k", linewidths=0.25
        )
        lim = np.abs(val).max()
        pc.set_clim(-lim, lim)
    else:
        pc = PolyCollection(polys, facecolors="none", edgecolors="k", linewidths=0.4)
    ax.add_collection(pc)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        a * np.cos(th),
        a * np.sin(th),
        color="tab:orange",
        lw=2,
        label="true PEC boundary",
    )
    ax.plot(*nodes[bnd].T, ".", color="tab:red", ms=3, label="PEC nodes (Ez = 0)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


fig, axes = plt.subplots(2, 2, figsize=(11, 11), constrained_layout=True)
for col, (name, maker, n) in enumerate(
    (
        ("Cartesian staircase (standard FIT)", N.cartesian_staircase, 8),
        ("Body-fitted O-grid (nonorthogonal FIT)", N.ogrid, 8),
    )
):
    nodes, cells, bnd = maker(a, n)
    draw(axes[0, col], nodes, cells, bnd, title=f"{name}\n{len(cells)} cells")
    k, free, vec, _ = N.modes(nodes, cells, bnd)
    ez = np.zeros(len(nodes))
    ez[free] = vec[:, 3]  # a TM21 mode
    err = abs(k[3] - ref[3]) / ref[3]
    draw(
        axes[1, col],
        nodes,
        cells,
        bnd,
        field=ez,
        title=f"TM21 mode, Ez (k a error {err:.1e})",
    )
axes[0, 0].legend(loc="lower left", fontsize=8)
fig.suptitle("Circular PEC cavity: same FIT operators, different mesh", fontsize=13)
fig.savefig(os.path.join(HERE, "meshes_and_modes.png"), dpi=150)
plt.close(fig)

# Convergence: TM01 error vs number of cells inside the cavity
fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
for name, maker, levels, style in (
    ("prototype staircase (node PEC)", N.cartesian_staircase, (8, 16, 32, 64), "s--"),
    ("prototype NFIT O-grid", N.ogrid, (4, 8, 16, 32, 64), "o-"),
):
    pts = []
    for n in levels:
        nodes, cells, bnd = maker(a, n)
        k, *_ = N.modes(nodes, cells, bnd, n_modes=2)
        cells_in = len(cells) if maker is N.ogrid else np.pi * n * n
        pts.append((cells_in, abs(k[0] - ref[0]) / ref[0]))
    ax.loglog(*np.array(pts).T, style, label=name)
w = json.load(open(os.path.join(HERE, "wakis_circle.json")))
ax.loglog(
    [p[1] for p in w["staircase"]],
    [p[2] for p in w["staircase"]],
    "^:",
    label="Wakis 3D solver, staircase",
)
ax.loglog(
    [p[1] for p in w["conformal"]],
    [p[2] for p in w["conformal"]],
    "v-",
    label="Wakis 3D solver, conformal (PBA-style)",
)
x = np.array([200, 20000])
ax.loglog(x, 3e-1 * (x / 200) ** -0.5, "k:", lw=0.8)
ax.text(1500, 1.4e-1, "1st order", fontsize=8)
ax.loglog(x, 3e-3 * (x / 200) ** -1.0, "k:", lw=0.8)
ax.text(1500, 4.5e-4, "2nd order", fontsize=8)
ax.set_xlabel("cells inside the cavity")
ax.set_ylabel("|f - f_exact| / f_exact  (TM01)")
ax.set_title("TM01 frequency error of a circular PEC cavity")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.savefig(os.path.join(HERE, "convergence.png"), dpi=150)
plt.close(fig)
print("ok")
