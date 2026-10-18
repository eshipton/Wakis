"""
2D TMz FIT on a Delaunay triangulation with its Voronoi dual (Weiland's
triangular/tetrahedral FIT idea): Ez on nodes, magnetic flux on triangle edges,
H on the Voronoi dual edges, which are perpendicular to the primary edges, so
the material matrix is DIAGONAL: M_e = L_dual / L_e = (cot a + cot b) / 2
(the angles opposite edge e in its two triangles). Electric dual areas: Voronoi
(mixed) areas, lumped. Eigenproblem G^T M G ez = k^2 D_A ez, as in nfit2d.py.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay


def disk_mesh(a, h):
    """Delaunay mesh of a disk: hexagonal interior lattice + boundary ring."""
    nb = int(np.ceil(2 * np.pi * a / h))
    th = 2 * np.pi * np.arange(nb) / nb
    ring = a * np.stack([np.cos(th), np.sin(th)], 1)
    xs = np.arange(-a, a + h, h)
    ys = np.arange(-a, a + h, h * np.sqrt(3) / 2)
    pts = [(x + (j % 2) * h / 2, y) for j, y in enumerate(ys) for x in xs]
    pts = np.array(pts)
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < a - 0.5 * h]
    nodes = np.vstack([ring, pts])
    tri = Delaunay(nodes).simplices
    boundary = np.zeros(len(nodes), bool)
    boundary[:nb] = True
    return nodes, tri, boundary


def assemble(nodes, tri):
    sides = np.stack(
        [tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [0, 1]]], 1
    )  # edge opposite vertex k
    key = np.sort(sides.reshape(-1, 2), axis=1)
    edges, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 3)
    P = nodes[tri]
    w = np.zeros(len(edges))
    for k in range(3):  # cotangent of the angle at vertex k, opposite edge k
        u = P[:, (k + 1) % 3] - P[:, k]
        v = P[:, (k + 2) % 3] - P[:, k]
        cot = np.einsum("ij,ij->i", u, v) / np.abs(
            u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        )
        np.add.at(w, inv[:, k], 0.5 * cot)
    n_e, n_n = len(edges), len(nodes)
    G = sp.csr_matrix(
        (
            np.r_[-np.ones(n_e), np.ones(n_e)],
            (np.r_[np.arange(n_e), np.arange(n_e)], np.r_[edges[:, 0], edges[:, 1]]),
        ),
        shape=(n_e, n_n),
    )
    # Voronoi (dual) area of each node: sum over incident edges of L_e * L_dual / 4
    L2 = np.sum((nodes[edges[:, 1]] - nodes[edges[:, 0]]) ** 2, axis=1)
    dual = np.zeros(n_n)
    np.add.at(dual, edges[:, 0], w * L2 / 4)
    np.add.at(dual, edges[:, 1], w * L2 / 4)
    K = (G.T @ sp.diags(w) @ G).tocsr()
    return K, dual, w, edges


def modes(nodes, tri, boundary, n_modes=8):
    K, dual, w, _ = assemble(nodes, tri)
    free = np.flatnonzero(~boundary)
    k2, vec = eigsh(
        K[free][:, free], k=n_modes, M=sp.diags(dual[free]), sigma=0.0, which="LM"
    )
    o = np.argsort(k2)
    return np.sqrt(k2[o]), free, vec[:, o], w
