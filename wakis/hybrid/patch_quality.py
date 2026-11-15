# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Interior node smoothing for tetrahedral patch meshes (`wakis.hybrid.patch_quality`).

`smooth_patch` relocates interior tetrahedral nodes only, keeping all nodes on
the box surface, on PEC holes/spheres, and on material-tag interfaces fixed and
leaving the tetrahedral connectivity (`cells`, `cell_tags`) unchanged. Because
smoothing touches the mesh geometry before matrix assembly and does not alter
the leapfrog update equations, every time-stepping step remains linear with an
exact transpose (`HybridFIT3D.linear_step_T`).
"""

import numpy as np
import scipy.sparse as sp

from ..unstructured.mesh import SimplexMesh
from ..unstructured.solver3d import whitney_mass_matrices
from .solver import MassSolver

__all__ = [
    "consistent_spectrum_ratio",
    "fixed_patch_nodes",
    "normalised_tet_volume",
    "smooth_patch",
    "worst_normalised_volume",
]

_LOC_E = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
_MLL = (np.ones((4, 4)) + np.eye(4)) / 20.0


def _tet_norm_volume_and_vol(P):
    """
    Normalised volume ``6 * sqrt(2) * V / L_rms^3`` and signed volume ``V``
    for tetrahedra with vertex array `P` of shape ``(..., 4, 3)``.
    """
    vol = np.linalg.det(P[..., 1:, :] - P[..., :1, :]) / 6.0
    d = P[..., _LOC_E[:, 1], :] - P[..., _LOC_E[:, 0], :]
    l_rms = np.sqrt(np.einsum("...pi,...pi->...p", d, d).mean(axis=-1))
    q = 6.0 * np.sqrt(2.0) * vol / np.maximum(l_rms**3, 1e-300)
    return q, vol


def _element_jacobi_inv_cond(P):
    """
    Inverse condition number ``lam_min(S_T) / lam_max(S_T)`` of the 6x6
    Jacobi-scaled consistent Whitney 1-form element mass matrix ``S_T`` for
    tetrahedra `P` of shape ``(..., 4, 3)``.
    """
    orig_shape = P.shape[:-2]
    Pf = P.reshape(-1, 4, 3)
    A = Pf[:, 1:, :] - Pf[:, :1, :]
    vol = np.linalg.det(A) / 6.0
    ok = vol > 0
    out = np.full(len(Pf), -np.inf)
    if not np.any(ok):
        return out.reshape(orig_shape)
    A_ok = A[ok]
    g = np.transpose(np.linalg.inv(A_ok), (0, 2, 1))
    grad = np.concatenate([-g.sum(axis=1, keepdims=True), g], axis=1)
    G = np.einsum("nik,njk->nij", grad, grad)
    M = np.empty((len(A_ok), 6, 6))
    for p, (a, b) in enumerate(_LOC_E):
        for q, (c, d) in enumerate(_LOC_E):
            M[:, p, q] = (
                _MLL[a, c] * G[:, b, d]
                - _MLL[a, d] * G[:, b, c]
                - _MLL[b, c] * G[:, a, d]
                + _MLL[b, d] * G[:, a, c]
            )
    diag = np.einsum("nii->ni", M)
    pos = np.all(diag > 0, axis=1)
    if np.any(pos):
        inv_sqrt = 1.0 / np.sqrt(diag[pos])
        S = M[pos] * inv_sqrt[:, :, None] * inv_sqrt[:, None, :]
        S = 0.5 * (S + np.transpose(S, (0, 2, 1)))
        ev = np.linalg.eigvalsh(S)
        ratio = np.where(ev[:, 0] > 0, ev[:, 0] / ev[:, -1], -np.inf)
        idx_ok = np.flatnonzero(ok)
        out[idx_ok[pos]] = ratio
    return out.reshape(orig_shape)


def normalised_tet_volume(tet_mesh, cells=None):
    """
    Normalised volume ``6 * sqrt(2) * V / L_rms^3`` [-] of each tetrahedron.

    Equals ``1`` for a regular tetrahedron, lies in ``(0, 1]`` for any
    positively oriented non-degenerate tetrahedron, and is ``<= 0`` for
    degenerate or inverted tetrahedra.

    Parameters
    ----------
    tet_mesh : SimplexMesh or ndarray of shape (n_nodes, 3)
        Tetrahedral mesh or node coordinate array [m].
    cells : ndarray of int, shape (n_cells, 4), optional
        Tetrahedron vertex indices when `tet_mesh` is a node array.

    Returns
    -------
    q : ndarray of float, shape (n_cells,)
        Normalised tetrahedron volumes [-].
    """
    if isinstance(tet_mesh, SimplexMesh):
        nodes, cells = tet_mesh.nodes, tet_mesh.cells
    else:
        nodes = np.asarray(tet_mesh, float)
        cells = np.asarray(cells, np.int64)
    q, _ = _tet_norm_volume_and_vol(nodes[cells])
    return q


def worst_normalised_volume(tet_mesh):
    """
    Minimum normalised tetrahedron volume ``min_T 6 * sqrt(2) * V_T / L_rms,T^3`` [-].

    Parameters
    ----------
    tet_mesh : SimplexMesh
        Tetrahedral mesh.

    Returns
    -------
    float
        Worst normalised tetrahedron volume [-].
    """
    return float(normalised_tet_volume(tet_mesh).min())


def fixed_patch_nodes(tet_mesh):
    """
    Boolean mask of nodes that must remain fixed during patch smoothing.

    All nodes lying on any boundary face of `tet_mesh` (which includes the
    outer box surface as well as any PEC hole surface such as a spherical
    obstacle) and all nodes on internal material-tag interfaces are marked
    True.

    Parameters
    ----------
    tet_mesh : SimplexMesh
        Tetrahedral patch mesh.

    Returns
    -------
    fixed : ndarray of bool, shape (n_nodes,)
        True on boundary and interface nodes, False on interior movable nodes.
    """
    fixed = np.zeros(tet_mesh.n_nodes, bool)
    fixed[tet_mesh.faces[tet_mesh.boundary_faces].ravel()] = True
    if np.unique(tet_mesh.cell_tags).size > 1:
        f = tet_mesh.cell_faces.ravel()
        t = np.repeat(tet_mesh.cell_tags, 4)
        lo = np.full(tet_mesh.n_faces, np.iinfo(np.int64).max)
        hi = np.full(tet_mesh.n_faces, np.iinfo(np.int64).min)
        np.minimum.at(lo, f, t)
        np.maximum.at(hi, f, t)
        fixed[tet_mesh.faces[lo != hi].ravel()] = True
    return fixed


def consistent_spectrum_ratio(tet_mesh, eps_r=1.0, live=None):
    """
    Jacobi-scaled consistent Whitney 1-form mass spectrum ratio ``lam_hi / lam_lo`` [-].

    Parameters
    ----------
    tet_mesh : SimplexMesh
        Tetrahedral mesh.
    eps_r : float or ndarray of shape (n_cells,), optional
        Relative permittivity per tetrahedron [-].
    live : ndarray of float, shape (n_cells,), optional
        1.0 on live tetrahedra, 0.0 on PEC tetrahedra.

    Returns
    -------
    float
        Condition proxy ``lam_hi / lam_lo`` of the Jacobi-scaled coupled
        consistent mass block `MassSolver(M1, method="chebyshev")`.
    """
    nc = tet_mesh.n_cells
    eps_arr = np.broadcast_to(np.asarray(eps_r, float), (nc,))
    live_arr = np.ones(nc, float) if live is None else np.asarray(live, float)
    M1, _ = whitney_mass_matrices(tet_mesh, eps_arr, np.ones(nc), live_arr)
    ms = MassSolver(M1, method="chebyshev")
    return float(ms.lam_hi / ms.lam_lo)


def smooth_patch(
    tet_mesh,
    fixed_nodes=None,
    objective="volume",
    n_sweeps=6,
    step=0.25,
    seed=0,
    mutate=None,
):
    """
    Smooth interior tetrahedral nodes without changing mesh topology.

    Each sweep visits the movable (interior) nodes in order of increasing
    worst incident-tetrahedron quality and tests damped Laplacian moves
    towards the average of the node's edge neighbours, altitudes away from the
    opposite face of its worst incident tetrahedron, and deterministic
    seeded trial points within radius ``step * h_local``. A move is accepted
    only if every incident tetrahedron keeps strictly positive volume and the
    objective on the incident tetrahedra does not decrease. Consequently, the
    global worst normalised tetrahedron volume is monotonically non-decreasing
    across sweeps.

    Smoothing modifies node coordinates only before operator assembly; the
    leapfrog update in `HybridFIT3D` remains linear with exact transpose
    `HybridFIT3D.linear_step_T`.

    Parameters
    ----------
    tet_mesh : SimplexMesh
        Input 3D tetrahedral patch mesh.
    fixed_nodes : ndarray of bool or int, optional
        Mask of shape ``(n_nodes,)`` or indices of nodes to keep fixed.
        Defaults to `fixed_patch_nodes(tet_mesh)` (all nodes on any boundary
        face, including the box surface and PEC sphere/holes, plus any
        material-tag interface nodes).
    objective : {'volume', 'spectrum'}, optional
        ``'volume'`` (default) maximises the worst normalised tetrahedron
        volume ``min_T 6 * sqrt(2) * V_T / L_rms,T^3`` on the incident
        tetrahedra of each node. ``'spectrum'`` (or ``'lam_ratio'``,
        ``'mass_cond'``) additionally requires the worst element-wise
        Jacobi-scaled Whitney 1-form mass inverse condition number on the
        incident tetrahedra not to decrease.
    n_sweeps : int, optional
        Number of smoothing sweeps (default 6).
    step : float, optional
        Damping / local trial step factor in ``(0, 1]`` (default 0.25).
    seed : int, optional
        Deterministic random seed (default 0).
    mutate : {None, 'move_boundary'}, optional
        Verification mutation: ``'move_boundary'`` allows boundary nodes on
        the box/hole surfaces to move alongside interior nodes.

    Returns
    -------
    new_mesh : SimplexMesh
        Smoothed tetrahedral mesh with identical `cells` and `cell_tags`, and
        attribute ``smooth_history`` containing the sequence of global worst
        normalised volumes ``[w_0, w_1, ..., w_{n_sweeps}]``.
    """
    if objective not in ("volume", "spectrum", "lam_ratio", "mass_cond"):
        raise ValueError(
            "objective must be 'volume' or 'spectrum' ('lam_ratio'/'mass_cond')"
        )
    if mutate not in (None, "move_boundary"):
        raise ValueError("mutate must be None or 'move_boundary'")

    X = tet_mesh.nodes.copy()
    cells = tet_mesh.cells.copy()
    nc, nn = len(cells), len(X)

    if fixed_nodes is None:
        fixed = fixed_patch_nodes(tet_mesh)
    else:
        arr = np.asarray(fixed_nodes)
        if arr.dtype == bool and arr.shape == (nn,):
            fixed = arr.copy()
        else:
            fixed = np.zeros(nn, bool)
            fixed[np.asarray(arr, int)] = True

    if mutate == "move_boundary":
        fixed[:] = False

    nt = sp.csr_matrix(
        (np.ones(4 * nc), (cells.ravel(), np.repeat(np.arange(nc), 4))),
        shape=(nn, nc),
    ).tocsr()
    edges = tet_mesh.edges
    adj = sp.csr_matrix(
        (
            np.ones(2 * len(edges)),
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(nn, nn),
    ).tocsr()

    q_all, v_all = _tet_norm_volume_and_vol(X[cells])
    use_spec = objective in ("spectrum", "lam_ratio", "mass_cond")
    s_all = _element_jacobi_inv_cond(X[cells]) if use_spec else None

    history = [float(q_all.min())]
    movable = np.flatnonzero(~fixed)
    alphas = np.array([step, 0.5 * step, 0.25 * step, 1.0])
    alt_scales = step * np.array([0.1, 0.25, 0.5, 1.0])

    for sweep in range(int(n_sweeps)):
        rng = np.random.default_rng((int(seed), sweep))
        node_worst = np.full(nn, np.inf)
        np.minimum.at(node_worst, cells.ravel(), np.repeat(q_all, 4))
        order = movable[np.argsort(node_worst[movable], kind="stable")]
        for n in order:
            T = nt.indices[nt.indptr[n] : nt.indptr[n + 1]]
            nbrs = adj.indices[adj.indptr[n] : adj.indptr[n + 1]]
            if not T.size or not nbrs.size:
                continue
            x0 = X[n]
            x_lap = X[nbrs].mean(axis=0)
            P0 = X[cells[T]]
            loc = cells[T] == n
            w_idx = int(np.argmin(q_all[T]))
            opp = P0[w_idx, ~loc[w_idx]]
            nrm = np.cross(opp[1] - opp[0], opp[2] - opp[0])
            nrm_norm = float(np.linalg.norm(nrm))
            if nrm_norm > 0:
                nrm = nrm / nrm_norm
                if np.dot(nrm, x0 - opp.mean(axis=0)) < 0:
                    nrm = -nrm
            h_loc = float(np.linalg.norm(X[nbrs] - x0, axis=1).mean())
            cand_lap = x0[None, :] + alphas[:, None] * (x_lap - x0)[None, :]
            cand_alt = x0[None, :] + (h_loc * alt_scales)[:, None] * nrm[None, :]
            dirs = rng.standard_normal((12, 3))
            dirs /= np.linalg.norm(dirs, axis=1)[:, None]
            rads = step * h_loc * (rng.random(12) ** (1.0 / 3.0))
            cand_rnd = x0[None, :] + dirs * rads[:, None]
            Y = np.vstack([x0[None, :], cand_lap, cand_alt, cand_rnd])
            m = len(Y)
            P = np.broadcast_to(P0, (m, len(T), 4, 3)).copy()
            P[:, loc] = np.repeat(Y[:, None, :], int(loc.sum()), axis=1)
            q_cand, v_cand = _tet_norm_volume_and_vol(P)
            valid = (v_cand > 0).all(axis=1)
            min_q = np.where(valid, q_cand.min(axis=1), -np.inf)
            hmean_q = np.where(
                valid,
                -np.mean(1.0 / np.maximum(q_cand, 1e-30), axis=1),
                -np.inf,
            )
            if use_spec:
                s_cand = _element_jacobi_inv_cond(P)
                min_s = np.where(valid, s_cand.min(axis=1), -np.inf)
            best = 0
            for j in range(1, m):
                if not valid[j] or min_q[j] < min_q[0]:
                    continue
                if use_spec and min_s[j] < min_s[0]:
                    continue
                if use_spec:
                    better = min_s[j] > min_s[best] * (1.0 + 1e-12) or (
                        min_s[j] >= min_s[best]
                        and min_q[j] > min_q[best] * (1.0 + 1e-12)
                    )
                else:
                    better = min_q[j] > min_q[best] * (1.0 + 1e-12) or (
                        min_q[j] >= min_q[best]
                        and hmean_q[j] > hmean_q[best] * (1.0 - 1e-12)
                    )
                if better:
                    best = j
            if best > 0:
                P_best = P0.copy()
                P_best[loc] = Y[best]
                q_new, v_new = _tet_norm_volume_and_vol(P_best)
                if np.all(v_new > 0) and q_new.min() >= q_all[T].min():
                    if use_spec:
                        s_new = _element_jacobi_inv_cond(P_best)
                        if s_new.min() < s_all[T].min():
                            continue
                        s_all[T] = s_new
                    X[n] = Y[best]
                    q_all[T] = q_new
                    v_all[T] = v_new
        history.append(float(q_all.min()))

    new_mesh = SimplexMesh(X, cells, tet_mesh.cell_tags)
    new_mesh.smooth_history = history
    return new_mesh
