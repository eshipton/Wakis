"""
2D TMz FIT on general quadrilateral meshes (nonorthogonal FIT), compared on a
circular PEC cavity: Cartesian staircase vs body-fitted O-grid.

Unknowns: Ez on primary nodes (z-directed primary edges), magnetic flux b on
primary 2D edges (extruded primary facets), H on the dual edges.
  Faraday:  d b/dt = -G ez               (G: node->edge incidence, topological)
  Ampere :  eps D_A d ez/dt = G^T h       (D_A: dual (barycentric) node areas)
  material: h = M_nu b                    (the only metric-dependent operator)
M_nu comes from the magnetic energy with a per-corner reconstruction of B from
the two primary edges meeting at each cell corner (exact for constant B). It is
symmetric positive definite, and diagonal (= standard FIT) on rectangles; on
nonorthogonal cells it couples the two edges at each corner, as in NFIT
(Schuhmann & Weiland). Eigenproblem: G^T M_nu G ez = k^2 D_A ez.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.special import jn_zeros


def quad_edges(cells):
    """Unique edges (i<j) and, per cell, the edge index of each side k: (p_k, p_k+1)."""
    sides = np.stack([cells, np.roll(cells, -1, axis=1)], axis=-1).reshape(-1, 2)
    key = np.sort(sides, axis=1)
    edges, inv = np.unique(key, axis=0, return_inverse=True)
    return edges, inv.reshape(-1, 4)


def assemble(nodes, cells):
    edges, cell_edges = quad_edges(cells)
    n_n, n_e = len(nodes), len(edges)
    G = sp.csr_matrix(
        (
            np.r_[-np.ones(n_e), np.ones(n_e)],
            (np.r_[np.arange(n_e), np.arange(n_e)], np.r_[edges[:, 0], edges[:, 1]]),
        ),
        shape=(n_e, n_n),
    )
    # length-weighted normal of each edge for its own orientation i->j: N = rot(-90) t
    t = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    N = np.stack([t[:, 1], -t[:, 0]], axis=1)
    P = nodes[cells]  # (nc, 4, 2), counter-clockwise
    x, y = P[..., 0], P[..., 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1))
    rows, cols, vals = [], [], []
    for k in range(4):  # corner at node k, sides k-1 and k
        e1, e2 = cell_edges[:, (k - 1) % 4], cell_edges[:, k]
        M2 = np.stack([N[e1], N[e2]], axis=1)  # (nc, 2, 2): rows N1, N2
        Minv = np.linalg.inv(M2)  # B = Minv @ [phi1, phi2]
        S = (
            np.einsum("cij,cik->cjk", Minv, Minv) * (area / 4)[:, None, None]
        )  # Minv^T Minv w
        for a, ea in enumerate((e1, e2)):
            for b, eb in enumerate((e1, e2)):
                rows.append(ea)
                cols.append(eb)
                vals.append(S[:, a, b])
    Mnu = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_e, n_e),
    )
    dual = np.zeros(n_n)
    np.add.at(dual, cells.ravel(), np.repeat(area / 4, 4))
    K = (G.T @ Mnu @ G).tocsr()
    return K, dual, Mnu, edges


def modes(nodes, cells, boundary, n_modes=8):
    K, dual, Mnu, _ = assemble(nodes, cells)
    free = np.flatnonzero(~boundary)
    Kf = K[free][:, free]
    D = sp.diags(dual[free])
    k2, vec = eigsh(Kf, k=n_modes, M=D, sigma=0.0, which="LM")
    order = np.argsort(k2)
    return np.sqrt(k2[order]), free, vec[:, order], Mnu


def cartesian_staircase(a, n):
    """Square grid on [-a, a]^2 with n cells per radius; nodes with r >= a are PEC."""
    s = np.linspace(-a, a, 2 * n + 1)
    X, Y = np.meshgrid(s, s, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], 1)
    idx = np.arange(len(nodes)).reshape(X.shape)
    cells = np.stack(
        [idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1
    ).reshape(-1, 4)
    r = np.hypot(nodes[:, 0], nodes[:, 1])
    return nodes, cells, r >= a * (1 - 1e-12)


def ogrid(a, n, m=None, s_frac=0.5):
    """Body-fitted O-grid: central square (n x n cells, half-side s_frac*a) and four
    blocks from its sides to the circle (n x m cells), merged into one quad mesh."""
    m = m or max(2, int(round(n * (1 - s_frac) / s_frac / 1.0)))
    s = s_frac * a
    blocks = []
    u = np.linspace(-1, 1, n + 1)
    U, V = np.meshgrid(u, u, indexing="ij")
    blocks.append(np.stack([s * U, s * V], -1))
    t = np.linspace(0, 1, m + 1)
    for rot in range(4):
        th = rot * np.pi / 2
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        # right side of the square (x = s) -> arc from -45 to +45 degrees
        side = np.stack([np.full_like(u, s), s * u], -1)
        phi = u * np.pi / 4
        arc = np.stack([a * np.cos(phi), a * np.sin(phi)], -1)
        T = t[:, None, None]
        pts = (1 - T) * side[None] + T * arc[None]  # (m+1, n+1, 2)
        blocks.append(pts @ R.T)
    nodes, cells = [], []
    for B in blocks:
        ni, nj = B.shape[:2]
        base = sum(len(b) for b in nodes)
        nodes.append(B.reshape(-1, 2))
        idx = base + np.arange(ni * nj).reshape(ni, nj)
        c = np.stack(
            [idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1
        ).reshape(-1, 4)
        cells.append(c)
    nodes = np.concatenate(nodes)
    cells = np.concatenate(cells)
    # merge duplicate nodes on block interfaces
    key = np.round(nodes / (a * 1e-9)).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    merged = np.zeros((len(uniq), 2))
    merged[inv.ravel()] = nodes
    cells = inv.ravel()[cells]
    # orient cells counter-clockwise
    P = merged[cells]
    sa = np.sum(
        P[..., 0] * np.roll(P[..., 1], -1, 1) - np.roll(P[..., 0], -1, 1) * P[..., 1],
        axis=1,
    )
    cells[sa < 0] = cells[sa < 0][:, ::-1]
    r = np.hypot(merged[:, 0], merged[:, 1])
    return merged, cells, r >= a * (1 - 1e-9)


def exact_k(a, n_modes=8):
    """TM_nm wavenumbers of a circular PEC cavity, with degeneracy (n>0 twice)."""
    ks = []
    for nn in range(0, 6):
        for z in jn_zeros(nn, 3):
            ks += [z / a] * (1 if nn == 0 else 2)
    return np.sort(ks)[:n_modes]


if __name__ == "__main__":
    a = 1.0
    ref = exact_k(a)
    print("exact k*a:", np.round(ref * a, 5))
    for name, maker, levels in (
        ("staircase", cartesian_staircase, (8, 16, 32, 64)),
        ("O-grid NFIT", ogrid, (4, 8, 16, 32)),
    ):
        for n in levels:
            nodes, cells, bnd = maker(a, n)
            k, *_ = modes(nodes, cells, bnd)
            h = np.sqrt(np.pi * a * a / len(cells))
            err = np.abs(k - ref) / ref
            print(
                f"{name:12s} n={n:3d} cells={len(cells):6d} h={h:.4f} max rel err={err.max():.2e} "
                f"TM01 err={err[0]:.2e}"
            )
