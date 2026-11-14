# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Tetrahedral mesh quality and the element-wise stability limit of 3D FIT.

Geometric metrics (`tet_quality`): radius ratio, dihedral angles, edge
length ratio and the circumcentre position (well-centredness, which the
diagonal Voronoi Hodge cares about).

Stability (`element_stability`): the leapfrog of `UnstructuredFIT3D` is
stable for dt < 2 / sqrt(lambda_max), lambda_max the largest eigenvalue of
the pencil (K, M) with K = C^T M_nu C on the free edges and M = M_eps. Both
matrices are sums of element contributions, K = sum_T K_T and
M = sum_T M_T, with K_T positive semi-definite and M_T positive definite on
the free edges of T. Then, for every x,

    x^T K x = sum_T x_T^T K_T x_T <= sum_T lambda_T x_T^T M_T x_T
            <= (max_T lambda_T) x^T M x,

so lambda_max <= max_T lambda_T with lambda_T = lambda_max(K_T, M_T) (a 6x6
generalized eigenproblem per tetrahedron), i.e. dt_max >= min_T dt_T,
dt_T = 2 / sqrt(lambda_T). The bound is local: the element with the smallest
dt_T is the one that limits the global step, up to the tightness factor
sqrt(max_T lambda_T / lambda_max) >= 1 (measured, not assumed; see the
physics guide section "CAD meshes and the tetrahedral time step").

Splitting the matrices into element parts:

* Whitney Hodge: the consistent element mass matrices of the Whitney 1- and
  2-forms are the natural split (M_T positive definite for any
  non-degenerate tetrahedron); the bound always holds.
* Voronoi Hodge: M and M_nu are diagonal, M_e = sum_T M_{e,T},
  M_nu,f = sum_T M_nu,{f,T}, but the circumcentric partial measures are
  signed (negative when a circumcentre lies outside its tetrahedron), so
  the geometric split can fail to be positive. Any split M_e = sum_T w_{e,T}
  M_e with w >= 0, sum_T w_{e,T} = 1 is exact as well, so we use the
  geometric weights on edges whose partials are all positive and
  stiffness-proportional weights (w_{e,T} proportional to (K_T)_ee) on the
  others; faces with a negative assembled dual length (non-Delaunay) are
  dropped from K for the bound (they only lower x^T K x) and counted in
  ``n_negative_faces``: they make K indefinite, which no step size cures.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0
from scipy.sparse.linalg import LinearOperator, eigsh

__all__ = [
    "tet_quality",
    "dihedral_angles",
    "element_matrices",
    "element_stability",
    "gershgorin_bound",
    "top_mode_localization",
    "lts_partition",
    "stability_report",
]


# ---------------------------------------------------------------- geometry
def dihedral_angles(mesh):
    """
    The six interior dihedral angles of every tetrahedron [rad].

    The angle at local edge (a, b) is pi minus the angle between the outward
    normals of the two faces that contain the edge.
    """
    X = mesh.nodes[mesh.cells]  # (nc, 4, 3)
    grad = mesh.barycentric_gradients()  # grad lambda_k ~ -n_k / h_k (inward)
    n = -grad / np.linalg.norm(grad, axis=2, keepdims=True)  # outward normals
    out = np.empty((mesh.n_cells, 6))
    for p, (a, b) in enumerate(mesh._local_edges):
        i, j = [v for v in range(4) if v not in (a, b)]
        # the faces containing edge ab are those opposite i and j
        cosang = -np.einsum("ni,ni->n", n[:, i], n[:, j])
        out[:, p] = np.arccos(np.clip(cosang, -1, 1))
    del X
    return out


def tet_quality(mesh):
    """
    Per-tetrahedron shape metrics.

    Returns
    -------
    dict of ndarray (n_cells,)
        ``radius_ratio`` (3 r_in / R_circ, 1 for the regular tetrahedron,
        -> 0 for slivers, needles, caps and wedges), ``min_dihedral`` and
        ``max_dihedral`` [deg] (regular: 70.53; slivers have both small and
        near-180 angles), ``edge_ratio`` (L_max / L_min), ``circumcenter_bary``
        (smallest barycentric coordinate of the circumcentre: < 0 when it lies
        outside the tetrahedron, i.e. not well-centred), ``volume`` [m^3] and
        ``h`` (mean edge length [m]).
    """
    if mesh.dim != 3:
        raise ValueError("tet_quality needs a tetrahedral mesh")
    dih = np.degrees(dihedral_angles(mesh))
    L = mesh.edge_length[mesh.cell_edges]
    X0 = mesh.nodes[mesh.cells[:, 0]]
    A = mesh.nodes[mesh.cells[:, 1:]] - X0[:, None]
    lam = np.linalg.solve(
        np.transpose(A, (0, 2, 1)), (mesh.cell_circumcenter - X0)[..., None]
    )[..., 0]
    bary = np.concatenate([1 - lam.sum(1, keepdims=True), lam], 1)
    return {
        "radius_ratio": mesh.quality(),
        "min_dihedral": dih.min(1),
        "max_dihedral": dih.max(1),
        "edge_ratio": L.max(1) / L.min(1),
        "circumcenter_bary": bary.min(1),
        "volume": mesh.cell_volume,
        "h": L.mean(1),
    }


# ---------------------------------------------------------------- element matrices
def _local_curl(mesh):
    """Local face-edge incidence C_T (nc, 4, 6), global orientations."""
    C = mesh.C.tocsr()
    nc = mesh.n_cells
    rows = np.repeat(mesh.cell_faces, 6, axis=1).ravel()  # (nc*4*6)
    cols = np.tile(mesh.cell_edges, (1, 4)).ravel()
    vals = np.asarray(C[rows, cols]).ravel()
    return vals.reshape(nc, 4, 6)


def _whitney_local(mesh, eps_r, mu_r, live):
    """Element Whitney mass matrices M1_T (nc, 6, 6) and M2_T (nc, 4, 4)."""
    grad = mesh.barycentric_gradients()
    V = mesh.cell_volume
    Mll = (np.ones((4, 4)) + np.eye(4)) / 20
    loc_e = mesh._local_edges
    sgn_e = np.stack(
        [np.where(mesh.cells[:, a] < mesh.cells[:, b], 1.0, -1.0) for a, b in loc_e], 1
    )
    G = np.einsum("nik,njk->nij", grad, grad)  # grad_i . grad_j
    M1 = np.zeros((mesh.n_cells, 6, 6))
    for p, (a, b) in enumerate(loc_e):
        for q, (c, d) in enumerate(loc_e):
            val = (
                Mll[a, c] * G[:, b, d]
                - Mll[a, d] * G[:, b, c]
                - Mll[b, c] * G[:, a, d]
                + Mll[b, d] * G[:, a, c]
            )
            M1[:, p, q] = eps_0 * eps_r * live * V * val * sgn_e[:, p] * sgn_e[:, q]
    terms = []
    for k_loc, (i, j, k) in enumerate(mesh._local_faces):
        gi, gj, gk = grad[:, i], grad[:, j], grad[:, k]
        t = [(i, 2 * np.cross(gj, gk)), (j, 2 * np.cross(gk, gi)), (k, 2 * np.cross(gi, gj))]
        fid = mesh.cell_faces[:, k_loc]
        nrm = mesh.face_normal[fid]
        flux = sum(np.einsum("ni,ni->n", v, nrm) for _, v in t)
        s = np.sign(flux)
        terms.append([(l, v * s[:, None]) for l, v in t])
    M2 = np.zeros((mesh.n_cells, 4, 4))
    for p in range(4):
        for q in range(4):
            val = np.zeros(mesh.n_cells)
            for lp, vp in terms[p]:
                for lq, vq in terms[q]:
                    val += Mll[lp, lq] * np.einsum("ni,ni->n", vp, vq)
            M2[:, p, q] = live * V * val / (mu_0 * mu_r)
    return M1, M2


def element_matrices(solver, weights="stiffness"):
    """
    Element stiffness and mass matrices of a `UnstructuredFIT3D` on the free edges.

    Returns
    -------
    K_T : ndarray (nc, 6, 6)
        Element stiffness, sum_T K_T = K restricted to the free edges [1/H].
    M_T : ndarray (nc, 6, 6)
        Element mass (diagonal for the Voronoi Hodge), sum_T M_T = M [F].
    free_loc : ndarray of bool (nc, 6)
        Local edges that are free dofs of the solver.
    n_negative_faces : int
        Faces with a negative assembled dual length that couple free edges
        (dropped from K_T; they make K indefinite).

    Parameters
    ----------
    solver : UnstructuredFIT3D
    weights : {"stiffness", "geometric"}
        Voronoi Hodge only: split of the diagonal edge mass M_e among the
        tetrahedra around the edge, proportional to their stiffness
        (K_T)_ee (default; guarantees lambda_T <= nu_T max_e K_ee / M_e,
        nu_T <= 6) or to the positive circumcentric partial dual areas
        (stiffness weights on edges with a non-positive partial).
    """
    if weights not in ("stiffness", "geometric"):
        raise ValueError("weights must be 'stiffness' or 'geometric'")
    mesh = solver.mesh
    nc = mesh.n_cells
    live = (~solver.pec_cells).astype(float)
    isfree = np.zeros(mesh.n_edges, bool)
    isfree[solver.free] = True
    free_loc = isfree[mesh.cell_edges]
    Cl = _local_curl(mesh)
    if solver.hodge == "whitney":
        M1, M2 = _whitney_local(mesh, solver.eps_r, solver.mu_r, live)
        K = np.einsum("nkp,nkl,nlq->npq", Cl, M2, Cl)
        n_neg = 0
        M = M1
    else:
        # Face weights: partial dual lengths, negatives -> stiffness drop
        Mnu = solver.Mnu_diag
        hp = mesh.cell_dual_edge * (live / (mu_0 * solver.mu_r))[:, None]  # (nc,4)
        Af = mesh.face_area[mesh.cell_faces]
        Mf = Mnu[mesh.cell_faces]
        hpos = np.maximum(hp, 0.0)
        hsum = np.zeros(mesh.n_faces)
        np.add.at(hsum, mesh.cell_faces.ravel(), hpos.ravel())
        hs = hsum[mesh.cell_faces]
        w_f = np.where(hs > 0, hpos / np.where(hs > 0, hs, 1), 0.5)
        mf = np.maximum(Mf, 0.0) * w_f  # (nc, 4)
        del Af
        couples = np.abs(mesh.C[:, solver.free]).sum(1).A1 > 0
        n_neg = int(np.sum((Mnu < 0) & couples))
        K = np.einsum("nkp,nk,nkq->npq", Cl, mf, Cl)
        # Edge weights: geometric partials where all are positive around the edge
        P = mesh.dual_area_partial.tocsr()
        rows = mesh.cell_edges.ravel()
        cols = np.repeat(np.arange(nc), 6)
        a = np.asarray(P[rows, cols]).ravel().reshape(nc, 6)
        a = a * (solver.eps_r * live)[:, None]
        neg = np.zeros(mesh.n_edges, bool)
        np.logical_or.at(neg, rows, (a <= 0).ravel() & (live[cols] > 0))
        asum = np.zeros(mesh.n_edges)
        np.add.at(asum, rows, np.maximum(a, 0).ravel())
        kd = np.einsum("npp->np", K)
        ksum = np.zeros(mesh.n_edges)
        np.add.at(ksum, rows, kd.ravel())
        w_geo = np.maximum(a, 0) / np.where(asum > 0, asum, 1)[mesh.cell_edges]
        w_stf = kd / np.where(ksum > 0, ksum, 1)[mesh.cell_edges]
        if weights == "stiffness":
            w = w_stf
        else:
            w = np.where(neg[mesh.cell_edges], w_stf, w_geo)
        cap_full = np.zeros(mesh.n_edges)
        cap_full[solver.free] = solver.cap
        m = cap_full[mesh.cell_edges] * w
        M = np.zeros((nc, 6, 6))
        idx = np.arange(6)
        M[:, idx, idx] = m
    # Restrict to the free local edges
    fl = free_loc.astype(float)
    K = K * fl[:, :, None] * fl[:, None, :]
    M = M * fl[:, :, None] * fl[:, None, :]
    return K, M, free_loc, n_neg


def _element_lambda(K, M):
    """lambda_max of the pencils (K_T, M_T), zero rows/cols treated as absent."""
    nc = K.shape[0]
    d = np.einsum("npp->np", M)
    absent = d <= 0
    Mx = M.copy()
    i = np.arange(6)
    Mx[:, i, i] = np.where(absent, 1.0, d)
    # Edge present in K but with zero mass share: unbounded element ratio
    kd = np.einsum("npp->np", K)
    inf = np.any(absent & (kd > 0), axis=1)
    L = np.linalg.cholesky(Mx)
    Linv = np.linalg.inv(L)
    S = np.einsum("nij,njk,nlk->nil", Linv, K, Linv)
    lam = np.linalg.eigvalsh(0.5 * (S + np.transpose(S, (0, 2, 1))))[:, -1]
    lam[inf] = np.inf
    return np.maximum(lam, 0.0) if nc else lam


def gershgorin_bound(solver):
    """
    Row-sum bound max_e sum_e' |K_ee'| / sqrt(M_e M_e') >= lambda_max (Voronoi).

    For the Voronoi Hodge with massless edges this bounds the free-edge
    block K_ff, which dominates the Schur complement the solver steps.
    """
    if solver.hodge != "voronoi":
        raise ValueError("gershgorin_bound needs the diagonal (Voronoi) Hodge")
    s = 1 / np.sqrt(solver.cap)
    S = sp.diags(s) @ abs(solver.K) @ sp.diags(s)
    return float(np.asarray(S.sum(1)).max())


def element_stability(solver, eigen_tol=1e-6, weights="stiffness"):
    """
    Element-wise stability bounds of a `UnstructuredFIT3D` and their tightness.

    Parameters
    ----------
    solver : UnstructuredFIT3D
    eigen_tol : float
        Lanczos tolerance for the reference lambda_max (re-computed; the
        solver's own estimate uses 1e-4).

    Returns
    -------
    dict
        ``lam_T`` (n_cells,) element bounds [1/s^2] (0 for PEC cells),
        ``dt_T`` = 2 / sqrt(lam_T) [s] (inf where lam_T = 0), ``lam_max``
        (Lanczos) and ``dt_max`` = 2 / sqrt(lam_max), ``lam_bound`` =
        max lam_T and ``dt_bound`` = min dt_T (guaranteed stable step),
        ``tightness`` = dt_max / dt_bound (>= 1), ``worst`` (element indices
        by increasing dt_T), ``n_negative_faces``, ``lam_gershgorin``
        (Voronoi only), ``s_T`` = c_T dt_T / h_T (dimensionless element
        Courant step, c_T the local light speed, h_T the mean edge length),
        ``edge_rate`` = K_ee / M_ee per free edge and ``lam_lower`` = its
        maximum (a lower bound of lam_max: the Rayleigh quotient of a unit
        vector), so lam_lower <= lam_max <= lam_bound.
    """
    K, M, free_loc, n_neg = element_matrices(solver, weights)
    lam_T = _element_lambda(K, M)
    # Rayleigh lower bound with unit vectors: lambda_max >= max_e K_ee / M_ee
    Md = solver.cap if solver.hodge == "voronoi" else solver.M1.diagonal()
    rate = solver.K.diagonal() / Md
    lam = _lambda_max(solver, eigen_tol)
    with np.errstate(divide="ignore"):
        dt_T = np.where(lam_T > 0, 2 / np.sqrt(np.where(lam_T > 0, lam_T, 1)), np.inf)
    lam_b = float(np.max(lam_T))
    out = {
        "lam_T": lam_T,
        "dt_T": dt_T,
        "lam_max": lam,
        "dt_max": 2 / np.sqrt(lam),
        "lam_bound": lam_b,
        "dt_bound": 2 / np.sqrt(lam_b),
        "tightness": float(np.sqrt(lam_b / lam)),
        "worst": np.argsort(dt_T),
        "n_negative_faces": n_neg,
        "edge_rate": rate,
        "lam_lower": float(rate.max()),
        "K_T": K,
        "M_T": M,
    }
    h = solver.mesh.edge_length[solver.mesh.cell_edges].mean(1)
    c_T = C0 / np.sqrt(solver.eps_r * solver.mu_r)
    out["s_T"] = c_T * dt_T / h
    if solver.hodge == "voronoi":
        out["lam_gershgorin"] = gershgorin_bound(solver)
    return out


def _operator(solver):
    """Symmetric operator M^-1/2 K_eff M^-1/2 (Voronoi) or (K, M) pair (Whitney)."""
    if solver.hodge == "voronoi":
        minv = 1 / np.sqrt(solver.cap)
        Kf = solver.K
        if solver.massless.size:
            Ksf = solver._Ksf

            def mv(x):
                y = minv * x
                return minv * (Kf @ y - Ksf.T @ solver._Kss.solve(Ksf @ y))

        else:

            def mv(x):
                return minv * (Kf @ (minv * x))

        n = solver.free.size
        return LinearOperator((n, n), matvec=mv, dtype=float), minv
    return None, None


def _lambda_max(solver, tol=1e-6, return_vector=False):
    op, minv = _operator(solver)
    n = solver.free.size
    v0 = np.random.default_rng(0).standard_normal(n)
    if op is not None:
        lam, vec = eigsh(op, k=1, which="LA", tol=tol, v0=v0)
        x = minv * vec[:, 0]
    else:
        # Generalized symmetric-definite: K x = lam M1 x
        lam, vec = eigsh(
            solver.K, k=1, M=solver.M1.tocsc(), which="LA", tol=tol, v0=v0,
            Minv=LinearOperator((n, n), matvec=solver._M1.solve, dtype=float),
        )  # fmt: skip
        x = vec[:, 0]
    return (float(lam[0]), x) if return_vector else float(lam[0])


def top_mode_localization(solver, stab=None, fractions=(0.001, 0.01, 0.1)):
    """
    Where the fastest (dt-limiting) mode lives.

    Computes the top eigenvector x of (K, M) and its element stiffness
    energy E_T = x_T^T K_T x_T (normalized to sum 1). If the global dt is set
    by the worst elements, E_T concentrates on the elements of smallest dt_T.

    Returns
    -------
    dict
        ``energy`` (n_cells,), ``share_worst`` {fraction: energy share of
        that fraction of elements with the smallest dt_T},
        ``rank_of_peak`` (rank in dt_T order of the element with the largest
        E_T, 0 = worst element).
    """
    stab = stab or element_stability(solver)
    _, x = _lambda_max(solver, return_vector=True)
    xf = np.zeros(solver.mesh.n_edges)
    xf[solver.free] = x
    xl = xf[solver.mesh.cell_edges]
    E = np.einsum("np,npq,nq->n", xl, stab["K_T"], xl)
    E = E / E.sum()
    order = stab["worst"]
    nc = len(order)
    share = {f: float(E[order[: max(1, int(round(f * nc)))]].sum()) for f in fractions}
    rank = int(np.flatnonzero(order == np.argmax(E))[0])
    return {"energy": E, "share_worst": share, "rank_of_peak": rank}


def lts_partition(solver, stab, dt_coarse):
    """
    Edges that need local time stepping at the coarse step `dt_coarse`.

    Every edge of an element with dt_T < dt_coarse is fine (the element
    bound guarantees that the coarse elements alone are stable at dt_coarse).

    Returns
    -------
    fine : ndarray of bool (n_free,)
        Mask over the solver's free edges.
    fine_elements : ndarray of int
    """
    bad = np.flatnonzero(stab["dt_T"] < dt_coarse)
    e = np.unique(solver.mesh.cell_edges[bad].ravel())
    isfine = np.zeros(solver.mesh.n_edges, bool)
    isfine[e] = True
    return isfine[solver.free], bad


def stability_report(solver, stab=None):
    """Short summary dict (scalars) of `element_stability` and `tet_quality`."""
    stab = stab or element_stability(solver)
    q = tet_quality(solver.mesh)
    live = ~solver.pec_cells
    w = stab["worst"][0]
    return {
        "n_tets": int(solver.mesh.n_cells),
        "hodge": solver.hodge,
        "n_massless": int(solver.massless.size),
        "dt_max": stab["dt_max"],
        "dt_bound": stab["dt_bound"],
        "tightness": stab["tightness"],
        "n_negative_faces": stab["n_negative_faces"],
        "courant_mean_h": float(stab["dt_max"] * C0 / solver.mesh.edge_length.mean()),
        "worst_element": int(w),
        "worst_radius_ratio": float(q["radius_ratio"][w]),
        "worst_min_dihedral": float(q["min_dihedral"][w]),
        "min_radius_ratio": float(q["radius_ratio"][live].min()),
        "min_dihedral": float(q["min_dihedral"][live].min()),
        "max_dihedral": float(q["max_dihedral"][live].max()),
        "max_edge_ratio": float(q["edge_ratio"][live].max()),
        "frac_not_well_centred": float(np.mean(q["circumcenter_bary"][live] < 0)),
    }
