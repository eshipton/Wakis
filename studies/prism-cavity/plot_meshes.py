"""Mesh figure: prism cross-sections (cavity, ring) and the z layering."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "prism-ring"))


def draw(ax, pm, colors, title, um=1e6, merged_color="k"):
    m2, c2 = pm.m2, pm.c2
    P = m2.nodes[m2.cells] * um
    tag = pm.tri_tags[:, 0]
    pc = PolyCollection(P, facecolors=[colors[t] for t in tag], edgecolors="0.35", linewidths=0.25)
    ax.add_collection(pc)
    # removed (merged) edges
    rem = np.flatnonzero(c2.edge_map < 0)
    for e in rem:
        a, b = m2.nodes[m2.edges[e]] * um
        ax.plot([a[0], b[0]], [a[1], b[1]], color="w", lw=0.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")


def main():
    import prism_study as S
    import ring_study as R

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    s = S.prism_solver(8, True)
    draw(axs[0], s.mesh, {0: "#f4b183", 1: "#deebf7"},
         f"(a) cavity, prism-graded res 8: {s.mesh.c2.n_cells} cells/layer, {s.mesh.K} layers\n"
         f"(white: {s.mesh.c2.n_removed} merged near-cocircular edges)")  # fmt: skip
    axs[0].set_xlim(0, 2.05)
    axs[0].set_ylim(0, 2.05)
    r = R.prism_solver(25, sym="even")
    draw(axs[1], r.mesh, {0: "#deebf7", 1: "#f4b183"},
         f"(b) Si ring quarter, h = 25 nm in the strip: {r.mesh.c2.n_cells} cells/layer")
    axs[1].set_xlim(0.8, 1.9)
    axs[1].set_ylim(0.0, 1.1)
    z = r.mesh.z * 1e6
    ax = axs[2]
    for zz in z:
        ax.axhline(zz, color="0.3", lw=0.5)
    ax.axhspan(0, R.T / 2 * 1e6, color="#f4b183", alpha=0.6, label="Si core (half, PMC at z = 0)")
    ax.axhspan(R.Z_TOP * 1e6 - R.D_ABS * 1e6, R.Z_TOP * 1e6, color="0.85", alpha=0.6, label="z PML")
    ax.set_ylim(0, R.Z_TOP * 1e6)
    ax.set_xticks([])
    ax.set_ylabel("z [um]")
    ax.set_title(f"(c) ring z levels: {r.mesh.K} layers, dz {r.mesh.dz.min()*1e9:.0f}-{r.mesh.dz.max()*1e9:.0f} nm", fontsize=9)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "c_prism_meshes.png"), dpi=170)


if __name__ == "__main__":
    main()
