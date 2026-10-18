"""
Dielectric rod (radius b, eps_r) in a circular PEC cavity (radius a): TM (Ez) and
TE (Hz) eigenmodes with Cartesian staircase, body-fitted O-grid (nonorthogonal FIT)
and Delaunay/Voronoi triangles, against the exact Bessel solution.

TM: Ez on primary nodes (PEC: Ez = 0 on the wall), eps in the node (dual-area) mass,
    1/mu in the edge matrix.
TE: Hz on primary nodes (PEC wall is the natural Neumann boundary), mu in the node mass,
    1/eps in the edge matrix, assembled cell by cell (each cell has one material).
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import eigsh
from scipy.spatial import Delaunay
from scipy.special import jv, jvp, yv, yvp

# ---------------------------------------------------------------- exact modes


def _det(k, n, a, b, eps, pol):
    k1, k0 = k * np.sqrt(eps), k
    # outside: A J_n(k0 r) + B Y_n(k0 r) with Ez(a)=0 (TM) or dHz/dr(a)=0 (TE)
    if pol == "TM":
        A, B = yv(n, k0 * a), -jv(n, k0 * a)
        p_in = 1.0  # continuity of dEz/dr / mu
    else:
        A, B = yvp(n, k0 * a), -jvp(n, k0 * a)
        p_in = 1.0 / eps  # continuity of (1/eps) dHz/dr
    f_out = A * jv(n, k0 * b) + B * yv(n, k0 * b)
    df_out = k0 * (A * jvp(n, k0 * b) + B * yvp(n, k0 * b))
    f_in, df_in = jv(n, k1 * b), k1 * jvp(n, k1 * b)
    return p_in * df_in * f_out - f_in * df_out


def exact_k(a, b, eps, pol, n_modes=6, kmax=None):
    kmax = kmax or 12 / a
    roots = []
    for n in range(0, 8):
        ks = np.linspace(1e-3 / a, kmax, 6000)
        d = np.array([_det(k, n, a, b, eps, pol) for k in ks])
        for i in np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0):
            r = brentq(_det, ks[i], ks[i + 1], args=(n, a, b, eps, pol))
            if abs(_det(r, n, a, b, eps, pol)) < 1e-6 * (1 + abs(d).max()):
                roots += [r] * (1 if n == 0 else 2)
    return np.sort(roots)[:n_modes]


# ---------------------------------------------------------------- operators


def _edges(cells):
    nv = cells.shape[1]
    sides = np.stack([cells, np.roll(cells, -1, axis=1)], axis=-1).reshape(-1, 2)
    edges, inv = np.unique(np.sort(sides, axis=1), axis=0, return_inverse=True)
    return edges, inv.reshape(-1, nv)


def _incidence(edges, n_nodes):
    n_e = len(edges)
    return sp.csr_matrix(
        (
            np.r_[-np.ones(n_e), np.ones(n_e)],
            (np.r_[np.arange(n_e), np.arange(n_e)], np.r_[edges[:, 0], edges[:, 1]]),
        ),
        shape=(n_e, n_nodes),
    )


def assemble_quads(nodes, cells, cell_eps, pol):
    """Corner-reconstruction Hodge on quads; per-cell material."""
    edges, ce = _edges(cells)
    t = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    N = np.stack([t[:, 1], -t[:, 0]], 1)
    P = nodes[cells]
    area = 0.5 * np.abs(
        np.sum(
            P[..., 0] * np.roll(P[..., 1], -1, 1)
            - np.roll(P[..., 0], -1, 1) * P[..., 1],
            1,
        )
    )
    edge_coef = 1.0 if pol == "TM" else 1.0 / cell_eps  # 1/mu (TM) or 1/eps (TE)
    node_coef = cell_eps if pol == "TM" else np.ones(len(cells))  # eps (TM) or mu (TE)
    rows, cols, vals = [], [], []
    for k in range(4):
        e1, e2 = ce[:, (k - 1) % 4], ce[:, k]
        Minv = np.linalg.inv(np.stack([N[e1], N[e2]], 1))
        S = (
            np.einsum("cij,cik->cjk", Minv, Minv)
            * (area / 4 * edge_coef)[:, None, None]
        )
        for i, ei in enumerate((e1, e2)):
            for j, ej in enumerate((e1, e2)):
                rows.append(ei)
                cols.append(ej)
                vals.append(S[:, i, j])
    M = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(edges), len(edges)),
    )
    mass = np.zeros(len(nodes))
    np.add.at(mass, cells.ravel(), np.repeat(area / 4 * node_coef, 4))
    G = _incidence(edges, len(nodes))
    return (G.T @ M @ G).tocsr(), mass


def assemble_tris(nodes, tri, cell_eps, pol):
    """Voronoi-dual (cotangent) Hodge on triangles; per-triangle material."""
    sides = np.stack([tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [0, 1]]], 1)
    edges, inv = np.unique(
        np.sort(sides.reshape(-1, 2), axis=1), axis=0, return_inverse=True
    )
    inv = inv.reshape(-1, 3)
    P = nodes[tri]
    edge_coef = 1.0 if pol == "TM" else 1.0 / cell_eps
    node_coef = cell_eps if pol == "TM" else np.ones(len(tri))
    w = np.zeros(len(edges))
    mass = np.zeros(len(nodes))
    for k in range(3):
        u = P[:, (k + 1) % 3] - P[:, k]
        v = P[:, (k + 2) % 3] - P[:, k]
        cot = np.einsum("ij,ij->i", u, v) / np.abs(
            u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        )
        np.add.at(w, inv[:, k], 0.5 * cot * edge_coef)
        L2 = np.sum((u - v) ** 2, 1)  # edge opposite vertex k
        part = (
            0.5 * cot * L2 / 4 * node_coef
        )  # this triangle's share of the edge's Voronoi area
        np.add.at(mass, tri[:, (k + 1) % 3], part)
        np.add.at(mass, tri[:, (k + 2) % 3], part)
    G = _incidence(edges, len(nodes))
    return (G.T @ sp.diags(w) @ G).tocsr(), mass


def solve(K, mass, fixed, n_modes):
    free = np.flatnonzero(~fixed)
    Kf = K[free][:, free]
    shift = (
        -1e-3 * Kf.diagonal().mean() / mass[free].mean()
    )  # avoid the TE constant mode
    k2 = eigsh(
        Kf,
        k=n_modes + 1,
        M=sp.diags(mass[free]),
        sigma=shift,
        which="LM",
        return_eigenvectors=False,
    )
    k2 = np.sort(k2[k2 > 1e-8 * abs(k2).max()])
    return np.sqrt(k2[:n_modes])


# ---------------------------------------------------------------- meshes


def cartesian(a, b, n):
    s = np.linspace(-a, a, 2 * n + 1)
    X, Y = np.meshgrid(s, s, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], 1)
    idx = np.arange(len(nodes)).reshape(X.shape)
    cells = np.stack(
        [idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1
    ).reshape(-1, 4)
    centre = nodes[cells].mean(1)
    rc = np.hypot(*centre.T)
    r = np.hypot(*nodes.T)
    return (
        nodes,
        cells,
        rc < b,
        rc < a,
        r >= a * (1 - 1e-12),
    )  # rod, inside cavity, PEC node


def ogrid_rings(a, b, n):
    """O-grid with concentric circular rings from r_t to a and a ring exactly at r = b."""
    s = 0.35 * b
    r_t = 0.7 * b
    h = (
        np.pi * a / (2 * n) * 1.0
    )  # target ring spacing ~ arc spacing at the wall / 2 scale

    def rings(r0, r1):
        m = max(1, int(np.ceil((r1 - r0) / (2 * np.pi * r1 / (4 * n)))))
        return np.linspace(r0, r1, m + 1)

    radii = np.unique(np.r_[rings(r_t, b), rings(b, a)])
    u = np.linspace(-1, 1, n + 1)
    U, V = np.meshgrid(u, u, indexing="ij")
    blocks = [np.stack([s * U, s * V], -1)]
    m_tr = max(1, int(round(n * (r_t - s) / (2 * s))))
    for rot in range(4):
        th = rot * np.pi / 2
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        side = np.stack([np.full_like(u, s), s * u], -1)
        phi = u * np.pi / 4
        circ = lambda r: np.stack([r * np.cos(phi), r * np.sin(phi)], -1)
        t = np.linspace(0, 1, m_tr + 1)[:, None, None]
        trans = (1 - t) * side[None] + t * circ(r_t)[None]
        outer = np.stack([circ(r) for r in radii], 0)
        blocks.append(np.concatenate([trans[:-1], outer], 0) @ R.T)
    nodes, cells = [], []
    for B in blocks:
        ni, nj = B.shape[:2]
        base = sum(len(x) for x in nodes)
        nodes.append(B.reshape(-1, 2))
        idx = base + np.arange(ni * nj).reshape(ni, nj)
        cells.append(
            np.stack(
                [idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1
            ).reshape(-1, 4)
        )
    nodes, cells = np.concatenate(nodes), np.concatenate(cells)
    key = np.round(nodes / (a * 1e-9)).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    merged = np.zeros((len(uniq), 2))
    merged[inv.ravel()] = nodes
    cells = inv.ravel()[cells]
    P = merged[cells]
    sa = np.sum(
        P[..., 0] * np.roll(P[..., 1], -1, 1) - np.roll(P[..., 0], -1, 1) * P[..., 1], 1
    )
    cells[sa < 0] = cells[sa < 0][:, ::-1]
    rc = np.hypot(*merged[cells].mean(1).T)
    r = np.hypot(*merged.T)
    return merged, cells, rc < b, np.ones(len(cells), bool), r >= a * (1 - 1e-9)


def tri_rings(a, b, h):
    """Delaunay mesh with node rings on r = a and r = b and a hexagonal fill."""

    def ring(r):
        nb = int(np.ceil(2 * np.pi * r / h))
        th = 2 * np.pi * np.arange(nb) / nb
        return r * np.stack([np.cos(th), np.sin(th)], 1)

    xs = np.arange(-a, a + h, h)
    ys = np.arange(-a, a + h, h * np.sqrt(3) / 2)
    pts = np.array([(x + (j % 2) * h / 2, y) for j, y in enumerate(ys) for x in xs])
    r = np.hypot(*pts.T)
    keep = (r < a - 0.5 * h) & (np.abs(r - b) > 0.5 * h)
    outer = ring(a)
    nodes = np.vstack([outer, ring(b), pts[keep]])
    tri = Delaunay(nodes).simplices
    rc = np.hypot(*nodes[tri].mean(1).T)
    fixed = np.zeros(len(nodes), bool)
    fixed[: len(outer)] = True
    return nodes, tri, rc < b, np.ones(len(tri), bool), fixed


def modes(mesh, kind, eps_r, pol, n_modes):
    nodes, cells, rod, inside, pec = mesh
    cell_eps = np.where(rod, eps_r, 1.0)
    if kind == "tri":
        K, mass = assemble_tris(nodes, cells, cell_eps, pol)
    else:
        K, mass = assemble_quads(nodes, cells, cell_eps, pol)
    if pol == "TM":
        fixed = pec
    else:
        # TE: the PEC wall is natural; staircase cells outside the cavity are removed
        used = np.zeros(len(nodes), bool)
        used[cells[inside].ravel()] = True
        if kind == "cart":
            K, mass = assemble_quads(nodes, cells[inside], cell_eps[inside], pol)
        fixed = ~used
    return solve(K, mass, fixed, n_modes)
