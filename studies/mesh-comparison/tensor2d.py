"""
Option 1 prototype: full-tensor subpixel averaging (Kottke/Farjadpour style) on a
Cartesian FIT grid, through the symmetric corner Hodge of nfit2d/dielectric2d.

Dielectric rod (radius b, eps_r) in a square PEC box [-1, 1]^2 aligned with the grid,
so the only curved boundary is the dielectric interface. TE (E in-plane, crossing the
interface) and TM. Interface cells use, from their fill fraction f and the interface
normal n:
  staircase : eps of the majority material
  diagonal  : diag(T)  (what Wakis' conformal Kottke averaging does today)
  tensor    : T = <1/eps> n n^T + (I - n n^T) / <eps>   (inverse-permittivity tensor)
Reference: Delaunay/Voronoi triangles (2nd order) on two fine meshes + Richardson.
"""

import dielectric2d as D
import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay

CENTER = np.array(
    [0.031, -0.017]
)  # off-grid rod centre: interface cuts cells irregularly


def cartesian_box(n):
    s = np.linspace(-1, 1, 2 * n + 1)
    X, Y = np.meshgrid(s, s, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], 1)
    idx = np.arange(len(nodes)).reshape(X.shape)
    cells = np.stack(
        [idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1
    ).reshape(-1, 4)
    boundary = (np.abs(nodes) >= 1 - 1e-12).any(axis=1)
    return nodes, cells, boundary


def fill_and_normal(nodes, cells, b, n_sub=24):
    P = nodes[cells]
    lo, hi = P.min(1), P.max(1)
    t = (np.arange(n_sub) + 0.5) / n_sub
    sx = lo[:, None, 0] + (hi - lo)[:, None, 0] * t[None, :]
    sy = lo[:, None, 1] + (hi - lo)[:, None, 1] * t[None, :]
    r = np.hypot(sx[:, :, None] - CENTER[0], sy[:, None, :] - CENTER[1])
    f = (r < b).mean(axis=(1, 2))
    c = P.mean(1) - CENTER
    nrm = c / np.linalg.norm(c, axis=1, keepdims=True)
    return f, nrm


def assemble_tensor(nodes, cells, T, node_coef):
    """Corner Hodge with a per-cell 2x2 (inverse-material) tensor T."""
    edges, ce = D._edges(cells)
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
    rows, cols, vals = [], [], []
    for k in range(4):
        e1, e2 = ce[:, (k - 1) % 4], ce[:, k]
        Minv = np.linalg.inv(np.stack([N[e1], N[e2]], 1))
        S = np.einsum("cji,cjk,ckl->cil", Minv, T, Minv) * (area / 4)[:, None, None]
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
    G = D._incidence(edges, len(nodes))
    return (G.T @ M @ G).tocsr(), mass


def cartesian_modes(n, b, eps, pol, scheme, n_modes=4):
    nodes, cells, boundary = cartesian_box(n)
    f, nrm = fill_and_normal(nodes, cells, b)
    I = np.eye(2)[None]
    nn = np.einsum("ci,cj->cij", nrm, nrm)
    eps_avg = 1 + (eps - 1) * f  # <eps>
    inv_avg = (1 - f) + f / eps  # <1/eps>
    if scheme == "staircase":
        e_cell = np.where(f > 0.5, eps, 1.0)
        T_te, m_tm = I / e_cell[:, None, None], e_cell
    else:
        T_full = inv_avg[:, None, None] * nn + (I - nn) / eps_avg[:, None, None]
        T_te = T_full if scheme == "tensor" else T_full * I  # diagonal keeps T_xx, T_yy
        m_tm = eps_avg  # TM: E tangential -> arithmetic mean
    if pol == "TE":
        K, mass = assemble_tensor(nodes, cells, T_te, np.ones(len(cells)))
        fixed = np.zeros(len(nodes), bool)
    else:
        K, mass = assemble_tensor(nodes, cells, np.repeat(I, len(cells), 0), m_tm)
        fixed = boundary
    return D.solve(K, mass, fixed, n_modes), len(cells)


def tri_box(b, h):
    """Delaunay mesh of the square box with node rings on the rod interface."""
    m = int(round(2 / h))
    s = np.linspace(-1, 1, m + 1)
    border = np.unique(
        np.vstack(
            [
                np.stack([s, -np.ones_like(s)], 1),
                np.stack([s, np.ones_like(s)], 1),
                np.stack([-np.ones_like(s), s], 1),
                np.stack([np.ones_like(s), s], 1),
            ]
        ),
        axis=0,
    )
    nb = int(np.ceil(2 * np.pi * b / h))
    th = 2 * np.pi * np.arange(nb) / nb
    ring = CENTER + b * np.stack([np.cos(th), np.sin(th)], 1)
    xs = np.arange(-1, 1 + h, h)
    ys = np.arange(-1, 1 + h, h * np.sqrt(3) / 2)
    pts = np.array([(x + (j % 2) * h / 2, y) for j, y in enumerate(ys) for x in xs])
    r = np.hypot(*(pts - CENTER).T)
    keep = (np.abs(pts) < 1 - 0.5 * h).all(1) & (np.abs(r - b) > 0.5 * h)
    nodes = np.vstack([border, ring, pts[keep]])
    tri = Delaunay(nodes).simplices
    rod = np.hypot(*(nodes[tri].mean(1) - CENTER).T) < b
    fixed = (np.abs(nodes) >= 1 - 1e-12).any(1)
    return nodes, tri, rod, fixed


def tri_modes(h, b, eps, pol, n_modes=4):
    nodes, tri, rod, boundary = tri_box(b, h)
    cell_eps = np.where(rod, eps, 1.0)
    K, mass = D.assemble_tris(nodes, tri, cell_eps, pol)
    fixed = boundary if pol == "TM" else np.zeros(len(nodes), bool)
    return D.solve(K, mass, fixed, n_modes), len(tri)


def reference(b, eps, pol, h=(0.02, 0.01)):
    k1, _ = tri_modes(h[0], b, eps, pol)
    k2, _ = tri_modes(h[1], b, eps, pol)
    return k2 + (k2 - k1) / 3  # Richardson, 2nd order


def corner_tensors(nodes, cells, b, eps, n_sub=12):
    """Per-corner (quarter-cell) fill fraction and normal -> (nc, 4, 2, 2) tensors."""
    P = nodes[cells]
    centre = P.mean(1)
    out = np.zeros((len(cells), 4, 2, 2))
    t = (np.arange(n_sub) + 0.5) / n_sub
    I = np.eye(2)
    for k in range(4):
        lo = np.minimum(P[:, k], centre)
        hi = np.maximum(P[:, k], centre)
        sx = lo[:, None, 0] + (hi - lo)[:, None, 0] * t[None]
        sy = lo[:, None, 1] + (hi - lo)[:, None, 1] * t[None]
        r = np.hypot(sx[:, :, None] - CENTER[0], sy[:, None, :] - CENTER[1])
        f = (r < b).mean(axis=(1, 2))
        c = (lo + hi) / 2 - CENTER
        n = c / np.linalg.norm(c, axis=1, keepdims=True)
        nn = np.einsum("ci,cj->cij", n, n)
        out[:, k] = ((1 - f) + f / eps)[:, None, None] * nn + (I - nn) / (
            1 + (eps - 1) * f
        )[:, None, None]
    return out


def assemble_corner_tensor(nodes, cells, Tc):
    edges, ce = D._edges(cells)
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
    rows, cols, vals = [], [], []
    for k in range(4):
        e1, e2 = ce[:, (k - 1) % 4], ce[:, k]
        Minv = np.linalg.inv(np.stack([N[e1], N[e2]], 1))
        S = (
            np.einsum("cji,cjk,ckl->cil", Minv, Tc[:, k], Minv)
            * (area / 4)[:, None, None]
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
    np.add.at(mass, cells.ravel(), np.repeat(area / 4, 4))
    G = D._incidence(edges, len(nodes))
    return (G.T @ M @ G).tocsr(), mass


def cartesian_te_corner(n, b, eps, diagonal=False, n_modes=4):
    nodes, cells, _ = cartesian_box(n)
    Tc = corner_tensors(nodes, cells, b, eps)
    if diagonal:
        Tc = Tc * np.eye(2)
    K, mass = assemble_corner_tensor(nodes, cells, Tc)
    return D.solve(K, mass, np.zeros(len(nodes), bool), n_modes), len(cells)


def exact_fill_and_normal(nodes, cells, b, n_gauss=64):
    """Rod fill fraction per (axis-aligned) cell by 1D Gauss integration of the
    circle-rectangle overlap, only for cells cut by the interface."""
    P = nodes[cells]
    lo, hi = P.min(1), P.max(1)
    corners_r = np.hypot(*(P - CENTER).transpose(2, 0, 1))
    f = (corners_r.max(1) < b).astype(float)
    cut = np.flatnonzero((corners_r.min(1) < b) & (corners_r.max(1) > b))
    # nearest-point check for cells cut through an edge without a corner inside
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    for idx in np.array_split(cut, max(1, len(cut) // 2000)):
        x0, x1 = lo[idx, 0], hi[idx, 0]
        xs = 0.5 * (x1 - x0)[:, None] * (xg[None] + 1) + x0[:, None]
        s = np.sqrt(np.clip(b * b - (xs - CENTER[0]) ** 2, 0, None))
        y0, y1 = lo[idx, 1][:, None], hi[idx, 1][:, None]
        length = np.clip(
            np.minimum(y1, CENTER[1] + s) - np.maximum(y0, CENTER[1] - s), 0, None
        )
        f[idx] = (
            (length * wg[None]).sum(1)
            * 0.5
            * (x1 - x0)
            / ((x1 - x0) * (hi[idx, 1] - lo[idx, 1]))
        )
    c = P.mean(1) - CENTER
    return f, c / np.linalg.norm(c, axis=1, keepdims=True)
