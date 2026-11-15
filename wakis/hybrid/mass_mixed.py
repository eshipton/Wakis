# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Partial mass lumping (`mass="mixed"`) for the tetrahedral block of `HybridFIT3D`.

Tetrahedral edges are partitioned into a lumped set ``L`` and a consistent set
``C``. An edge belongs to ``L`` only when:

* it is not on the box surface and is not a hanging or fixed (PEC) edge;
* every tetrahedron incident to it is live and strictly well-centred with a
  relative margin (its circumcentre lies strictly inside the tetrahedron and
  all per-tetrahedron Voronoi dual face areas are strictly positive);
* no tetrahedron incident to it has a vertex or face on the box surface.

On ``L``, the consistent Whitney 1-form mass ``M1`` is replaced by the positive
diagonal Voronoi (DEC) Hodge entry ``D_V[e] = eps_0 sum_{T ni e} eps_T A_{e,T} / L_e``,
dropping the ``L-C`` and ``L`` off-diagonal couplings so that
``M_mixed = blockdiag(D_V|_L, M1|_C)`` up to edge permutation.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as EPS0

__all__ = ["mixed_partition", "mixed_tet_mass", "well_centred_tets"]


def well_centred_tets(tet_mesh, margin=1e-2):
    """
    Boolean mask of strictly well-centred tetrahedra of `tet_mesh`.

    A tetrahedron ``T`` is well-centred with relative `margin` when:

    1. its circumcentre ``c_T`` lies strictly inside ``T`` at signed distance
       ``s_{f,T} > margin * h_T`` from all four faces ``f`` (where ``h_T`` is
       the mean edge length of ``T``);
    2. all six per-tetrahedron Voronoi dual edge areas
       ``A_{e,T} = 1/2 (s_{e,f1} s_{f1,T} + s_{e,f2} s_{f2,T})`` exceed
       ``(margin * L_e)^2``.

    Parameters
    ----------
    tet_mesh : SimplexMesh
        3D tetrahedral mesh.
    margin : float, optional
        Relative interior margin (default ``1e-2``).

    Returns
    -------
    wc : ndarray of bool, shape (n_cells,)
        True for well-centred tetrahedra.
    """
    h_t = tet_mesh.edge_length[tet_mesh.cell_edges].mean(1)[:, None]
    L_e = tet_mesh.edge_length[tet_mesh.cell_edges]
    return np.all(tet_mesh.cell_dual_edge > margin * h_t, axis=1) & np.all(
        tet_mesh.cell_dual_area > (margin * L_e) ** 2, axis=1
    )


def mixed_partition(
    tet_mesh,
    eps_t=1.0,
    live=None,
    bf_box=None,
    on_box=None,
    fixed=None,
    margin=1e-2,
    mutate=None,
):
    """
    Partition the tetrahedral edges into lumped (``L``) and consistent (``C``).

    Parameters
    ----------
    tet_mesh : SimplexMesh
        Tetrahedral patch mesh.
    eps_t : float or ndarray of shape (n_cells,), optional
        Relative permittivity per tetrahedron [-].
    live : ndarray of float, shape (n_cells,), optional
        1.0 on live tetrahedra, 0.0 on PEC tetrahedra.
    bf_box : ndarray of int, optional
        Boundary face indices lying on the box surface.
    on_box : ndarray of bool, shape (n_edges,), optional
        Mask of hanging tetrahedral edges on the box surface.
    fixed : ndarray of bool, shape (n_edges,), optional
        Mask of fixed (PEC) tetrahedral edges.
    margin : float, optional
        Relative margin passed to `well_centred_tets`.
    mutate : {None, 'lump_all', 'ignore_wellcentred'}, optional
        Verification mutations: ``'lump_all'`` lumps every tetrahedral edge
        including non-well-centred and box-surface ones;
        ``'ignore_wellcentred'`` follows the standard partition path with the
        `well_centred_tets` filter omitted.

    Returns
    -------
    lump_mask : ndarray of bool, shape (n_edges,)
        Boolean mask of the lumped tetrahedral edges ``L``.
    dv : ndarray of float, shape (n_lumped,)
        Diagonal Voronoi (DEC) Hodge entries ``D_V[e]`` [F] on `lump_mask`.
    """
    n_cells, n_edges = tet_mesh.n_cells, tet_mesh.n_edges
    eps_t = np.broadcast_to(np.asarray(eps_t, float), (n_cells,))
    live = np.ones(n_cells, float) if live is None else np.asarray(live, float)
    if bf_box is None:
        X = tet_mesh.nodes
        lo, hi = X.min(0), X.max(0)
        tol = 1e-9 * np.max(hi - lo)
        bf = tet_mesh.boundary_faces
        F = X[tet_mesh.faces[bf]]
        sel = np.zeros(len(bf), bool)
        for a in range(3):
            for pl in (lo[a], hi[a]):
                sel |= np.all(np.abs(F[:, :, a] - pl) < tol, axis=1)
        bf_box = bf[sel]
    if on_box is None:
        on_box = np.zeros(n_edges, bool)
        on_box[tet_mesh.face_edges[bf_box].ravel()] = True
    if fixed is None:
        pec_t = live <= 0
        fixed = np.zeros(n_edges, bool)
        fixed[tet_mesh.cell_edges[pec_t].ravel()] = True
        other_bf = np.setdiff1d(tet_mesh.boundary_faces, bf_box)
        fixed[tet_mesh.face_edges[other_bf].ravel()] = True
        touched = np.zeros(n_edges, bool)
        touched[tet_mesh.cell_edges[~pec_t].ravel()] = True
        fixed |= ~touched
    if mutate == "lump_all":
        lump_mask = np.ones(n_edges, bool)
    elif mutate in (None, "ignore_wellcentred"):
        wc = live > 0
        if mutate is None:
            wc = well_centred_tets(tet_mesh, margin=margin) & wc
        box_nodes = np.zeros(tet_mesh.n_nodes, bool)
        box_nodes[tet_mesh.faces[bf_box].ravel()] = True
        box_faces = np.zeros(tet_mesh.n_faces, bool)
        box_faces[bf_box] = True
        tet_on_box = np.any(box_nodes[tet_mesh.cells], axis=1) | np.any(
            box_faces[tet_mesh.cell_faces], axis=1
        )
        safe_tet = wc & ~tet_on_box
        bad_edge = np.zeros(n_edges, bool)
        bad_edge[tet_mesh.cell_edges[~safe_tet].ravel()] = True
        lump_mask = ~on_box & ~fixed & ~bad_edge
    else:
        raise ValueError("mutate must be None, 'lump_all', or 'ignore_wellcentred'")
    dual_area = tet_mesh.dual_measure(eps_t * live)
    dv = EPS0 * dual_area[lump_mask] / tet_mesh.edge_length[lump_mask]
    if mutate is None and np.any(dv <= 0):
        raise RuntimeError("non-positive Voronoi dual mass on a well-centred edge")
    return lump_mask, dv


def mixed_tet_mass(M1, lump_mask, dv):
    """
    Replace the `lump_mask` block of `M1` by ``diag(dv)``, dropping ``L-C``
    and ``L`` off-diagonal couplings: ``blockdiag(D_V|_L, M1|_C)``.
    """
    ic = sp.diags((~lump_mask).astype(float))
    d = np.zeros(M1.shape[0])
    d[lump_mask] = dv
    out = (ic @ M1 @ ic + sp.diags(d)).tocsr()
    out.eliminate_zeros()
    return out
