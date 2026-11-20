# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Convolutional PML (CFS-CPML) on the outer walls of a `SubgridFIT3D`.

Every nonzero of the face-edge incidence belongs to exactly one derivative
axis ``d`` (face normal ``a``, edge direction ``b``: ``d = 3 - a - b``), so
``K = sum_d K^(d)`` with ``K^(d) = C^(d) P_e`` (hanging edges keep the
direction of their parents). Stretching ``d/dx_d -> (1/s_d) d/dx_d``,
``s_d = 1 + sigma_d / (alpha_d + i w)`` (rates in 1/s, ``kappa = 1``), in
recursive-convolution form (Roden & Gedney 2000):

    psi_d  <- bh_d psi_d + ah_d K^(d) z                      (faces in the layer)
    b      <- b - dt (K z + sum_d psi_d)
    phi_d  <- be_d phi_d + ae_d K^(d)H W b                   (edges in the layer)
    e      <- e + dt D^-1 (K^H W b + sum_d phi_d - P^H i)

with ``bx = exp(-(sigma + alpha) dt)``, ``ax = sigma / (sigma + alpha)
(bx - 1)`` evaluated at the face centre (b side) or edge midpoint (e side)
coordinate along ``d``; ``z = R_0 e`` under LTS. Faces and edges of fine
patches inside the layer are stretched the same way (positions are
physical), so patches may reach into or through the layer under a global
step. The profile is ``sigma(u) = sigma_max u^m``, ``u`` the normalised depth,
``sigma_max = -(m + 1) ln(R0) c / (2 d)`` (the continuous normal-incidence
reflection of a PEC-backed layer of thickness ``d`` is ``R0``).

The step is linear; `SubgridCPML` provides its pieces and their exact
transposes to `SubgridLTSFIT3D`.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as C0

__all__ = [
    "SubgridCPML",
    "cpml_generator",
    "diagnose_cpml_eigenvector",
    "split_curl",
]


def split_curl(solver):
    """``[K^(0), K^(1), K^(2)]`` (faces x free edges), ``sum = K`` exactly."""
    s = solver
    Ccoo = s.C_all.tocoo()
    d_of = 3 - s.face_normal[Ccoo.row] - s.edge_dir[Ccoo.col]
    out = []
    for d in range(3):
        sel = d_of == d
        Cd = sp.csr_matrix(
            (Ccoo.data[sel], (Ccoo.row[sel], Ccoo.col[sel])), shape=s.C_all.shape
        )
        out.append((Cd[s.faces] @ s.P).tocsr())
    return out


class SubgridCPML:
    """
    CFS-CPML data of a `SubgridFIT3D`.

    Parameters
    ----------
    solver : SubgridFIT3D
    dt : float
        Step at which the layer is updated [s] (the coarse step under LTS).
    thickness : sequence of 3 (lo, hi) pairs [m]
        Layer thickness at each wall (0: no layer, the PEC/PMC wall stays).
    R0 : float
        Design reflection of the continuous layer.
    order : float
        Polynomial grading m.
    alpha_hz : float
        CFS frequency shift alpha / (2 pi) [Hz], constant in the layer.
    allow_patches : bool
        Allow faces/edges of refined patches inside the layer. With
        ``alpha_hz = 0`` such a layer grows exponentially (measured, physics
        guide "CPML": rate ~1 % of ``sigma_max``, at any CFL); a CFS shift
        larger than that rate suppressed it in the tests. Default False:
        patches must stay outside the layer.
    sigma_sampling : {"fine", "coarse_cell", "coarse_cell_dual"}, optional
        Coordinate sampling of ``sigma_d`` inside refined patches:
        - ``"fine"`` (default): evaluate ``sigma_d`` at each fine face centre
          (for ``psi_d``) and fine edge midpoint (for ``phi_d``).
        - ``"coarse_cell"`` (Candidate C1(i)): evaluate ``sigma_d`` of
          ``psi_d`` at the coarse parent cell midpoint along ``d``, keeping
          ``phi_d`` at fine edge midpoints.
        - ``"coarse_cell_dual"`` (Candidate C1(ii)): evaluate ``sigma_d`` of
          ``psi_d`` at the coarse parent cell midpoint along ``d`` and
          ``sigma_d`` of ``phi_d`` at the coarse dual cell node along ``d``.
    """

    def __init__(self, solver, dt, thickness, R0=1e-6, order=3.0, alpha_hz=0.0,
                 allow_patches=False, sigma_sampling="fine"):  # fmt: skip
        if sigma_sampling not in ("fine", "coarse_cell", "coarse_cell_dual"):
            raise ValueError(f"unknown sigma_sampling: {sigma_sampling!r}")
        s = solver
        self.dt = float(dt)
        self.R0 = float(R0)
        self.order = float(order)
        self.sigma_sampling = sigma_sampling
        nodes = s.mesh.root.nodes
        lo = np.array([v[0] for v in nodes])
        hi = np.array([v[-1] for v in nodes])
        th = np.asarray(thickness, float).reshape(3, 2)
        self.thickness = th
        alpha = 2 * np.pi * float(alpha_hz)
        self.alpha = alpha
        self.sigma_max = float(
            max(
                (-(order + 1) * np.log(R0) * C0 / (2 * t) for t in th.ravel() if t > 0),
                default=0.0,
            )
        )

        def sigma(d, u):
            out = np.zeros_like(u)
            for side in (0, 1):
                t = th[d, side]
                if t <= 0:
                    continue
                depth = (lo[d] + t - u) if side == 0 else (u - (hi[d] - t))
                x = np.clip(depth / t, 0.0, 1.0)
                x[x < 1e-9] = 0.0  # node lines on the layer's inner face: no loss
                smax = -(order + 1) * np.log(R0) * C0 / (2 * t)
                out = out + smax * x**order
            return out

        Kd = split_curl(s)
        self.Kd = Kd
        fpos = s.face_cen[s.faces]
        epos = s.edge_mid[s.free_edges]
        fblk = np.searchsorted(s.mesh.face_off, s.faces, side="right") - 1
        eblk = s.edge_block[s.free_edges]
        self.pf, self.Kf, self.bh, self.ah, self.sf = [], [], [], [], []
        self.pe, self.Ke, self.be, self.ae, self.se = [], [], [], [], []
        dt = self.dt
        for d in range(3):
            K = Kd[d]
            nodes_d = nodes[d]
            u_f = fpos[:, d].copy()
            if sigma_sampling in ("coarse_cell", "coarse_cell_dual"):
                idx_f = np.clip(
                    np.searchsorted(nodes_d, u_f, side="right") - 1,
                    0,
                    len(nodes_d) - 2,
                )
                u_f = np.where(
                    fblk > 0, 0.5 * (nodes_d[idx_f] + nodes_d[idx_f + 1]), u_f
                )
            sf = sigma(d, u_f)
            rows_nz = np.diff(K.indptr) > 0
            pf = np.flatnonzero((sf > 0) & rows_nz)
            b = np.exp(-(sf[pf] + alpha) * dt)
            self.pf.append(pf)
            self.Kf.append(K[pf].tocsr())
            self.bh.append(b)
            self.ah.append(sf[pf] / (sf[pf] + alpha) * (b - 1))
            self.sf.append(sf[pf])
            KH = K.conj().T.tocsr()
            u_e = epos[:, d].copy()
            if sigma_sampling == "coarse_cell_dual":
                mid_d = 0.5 * (nodes_d[:-1] + nodes_d[1:])
                idx_e = np.searchsorted(mid_d, u_e, side="left")
                u_e = np.where(eblk > 0, nodes_d[idx_e], u_e)
            se = sigma(d, u_e)
            cols_nz = np.diff(KH.indptr) > 0
            pe = np.flatnonzero((se > 0) & cols_nz)
            b = np.exp(-(se[pe] + alpha) * dt)
            self.pe.append(pe)
            self.Ke.append(KH[pe].tocsr())
            self.be.append(b)
            self.ae.append(se[pe] / (se[pe] + alpha) * (b - 1))
            self.se.append(se[pe])
        self.edges = np.unique(np.concatenate(self.pe))
        self.face_set = np.unique(np.concatenate(self.pf))
        in_patch = np.any(fblk[self.face_set] > 0) or np.any(eblk[self.edges] > 0)
        self.patches_in_layer = bool(in_patch)
        if in_patch and not allow_patches:
            raise ValueError(
                "a refined patch reaches into the CPML layer; keep patches outside the "
                "layer or pass allow_patches=True with a CFS shift alpha_hz > 0"
            )

    def zeros(self, dtype=float):
        return ([np.zeros(p.size, dtype) for p in self.pf],
                [np.zeros(p.size, dtype) for p in self.pe])  # fmt: skip

    @property
    def n_aux(self):
        return int(sum(p.size for p in self.pf) + sum(p.size for p in self.pe))


def cpml_generator(solver):
    """
    Assemble the sparse semi-discrete generator ``G`` (``y_dot = G y``) for
    a `SubgridLTSFIT3D` with CFS-CPML on state
    ``y = [b, e, psi_0, psi_1, psi_2, phi_0, phi_1, phi_2]``.

    The continuous-time limit as ``dt -> 0`` of the recursive-convolution step
    in `SubgridCPML` (`bh_d = 1 - (sigma_d + alpha) dt + O(dt^2)`,
    `ah_d = -sigma_d dt + O(dt^2)`) satisfies::

        b_dot     = -K z - sum_d I_{pf,d} psi_d
        e_dot     = D^-1 (K^H W b + sum_d I_{pe,d} phi_d)
        psi_d_dot = -(sigma_{f,d} + alpha) psi_d - sigma_{f,d} K_f^(d) z
        phi_d_dot = -(sigma_{e,d} + alpha) phi_d - sigma_{e,d} K_e^(d) W b

    where ``z = R_0 e`` under LTS and ``z = e`` under a global step.

    Parameters
    ----------
    solver : SubgridLTSFIT3D
        Solver instance with ``solver.cpml is not None``.

    Returns
    -------
    G : scipy.sparse.csr_matrix
        Sparse generator of shape ``(n_state, n_state)``.
    """
    cpml = solver.cpml
    if cpml is None:
        raise ValueError("solver has no SubgridCPML instance")
    nf = solver.n_faces
    ne = solver.n_free
    R0 = solver.lts.matrix() if solver.lts is not None else sp.eye(ne, format="csr")
    W = sp.diags(solver.W)
    Dinv = sp.diags(solver.Dinv)
    alpha = cpml.alpha

    n_blocks = 2 + 6
    blocks = [[None] * n_blocks for _ in range(n_blocks)]
    blocks[0][1] = (-solver.K @ R0).tocsr()
    blocks[1][0] = (Dinv @ solver.KH @ W).tocsr()

    for d in range(3):
        pf = cpml.pf[d]
        npf = pf.size
        row_psi = 2 + d
        if npf > 0:
            I_pf = sp.csr_matrix((np.ones(npf), (pf, np.arange(npf))), shape=(nf, npf))
            blocks[0][row_psi] = -I_pf
            blocks[row_psi][1] = (-sp.diags(cpml.sf[d]) @ cpml.Kf[d] @ R0).tocsr()
            blocks[row_psi][row_psi] = -sp.diags(cpml.sf[d] + alpha, format="csr")
        else:
            blocks[row_psi][row_psi] = sp.csr_matrix((0, 0))

        pe = cpml.pe[d]
        npe = pe.size
        row_phi = 5 + d
        if npe > 0:
            I_pe = sp.csr_matrix((np.ones(npe), (pe, np.arange(npe))), shape=(ne, npe))
            blocks[1][row_phi] = (Dinv @ I_pe).tocsr()
            blocks[row_phi][0] = (-sp.diags(cpml.se[d]) @ cpml.Ke[d] @ W).tocsr()
            blocks[row_phi][row_phi] = -sp.diags(cpml.se[d] + alpha, format="csr")
        else:
            blocks[row_phi][row_phi] = sp.csr_matrix((0, 0))

    return sp.bmat(blocks, format="csr")


def diagnose_cpml_eigenvector(solver, vals=None, vecs=None):
    """
    Localise the most unstable eigenvector of ``cpml_generator(solver)``
    by electromagnetic energy per block, per normalized depth quartile in the
    layer, and on interface faces/edges vs interior faces/edges.

    Parameters
    ----------
    solver : SubgridLTSFIT3D
    vals : ndarray, optional
        Precomputed eigenvalues of ``G``.
    vecs : ndarray, optional
        Precomputed right eigenvectors of ``G`` (columns).

    Returns
    -------
    info : dict
        Dictionary with keys ``lambda_max``, ``re_over_sigma_max``, ``freq_hz``,
        ``block_energy_share``, ``interface_face_share_of_b``,
        ``interface_edge_share_of_e``, ``interface_total_share``, and
        ``depth_quartile_share``.
    """
    from scipy.linalg import eig

    cpml = solver.cpml
    if vals is None or vecs is None:
        G = cpml_generator(solver)
        vals, vecs = eig(G.toarray())
    idx = int(np.argmax(vals.real))
    lam = complex(vals[idx])
    v = vecs[:, idx]
    nf = solver.n_faces
    ne = solver.n_free
    vb = v[:nf]
    ve = v[nf : nf + ne]
    eb = (np.abs(vb) ** 2) * solver.W
    ee = (np.abs(ve) ** 2) * solver.D
    eb_sum = float(eb.sum())
    ee_sum = float(ee.sum())
    etot = eb_sum + ee_sum + 1e-300

    fblk = np.searchsorted(solver.mesh.face_off, solver.faces, side="right") - 1
    eblk = solver.edge_block[solver.free_edges]
    n_blocks = len(solver.mesh.blocks)
    block_share = {
        int(b): float((eb[fblk == b].sum() + ee[eblk == b].sum()) / etot)
        for b in range(n_blocks)
    }

    # Interface free edges: coarse parent edges whose column in P has >1 nonzero
    P_csc = solver.P.tocsc()
    col_nnz = np.diff(P_csc.indptr)
    if_edge_mask = col_nnz > 1
    # Faces touching at least one interface free edge in K
    K_if = solver.K[:, if_edge_mask].tocsr()
    if_face_mask = np.diff(K_if.indptr) > 0

    if_face_share_b = float(eb[if_face_mask].sum() / (eb_sum + 1e-300))
    if_edge_share_e = float(ee[if_edge_mask].sum() / (ee_sum + 1e-300))
    if_total_share = float((eb[if_face_mask].sum() + ee[if_edge_mask].sum()) / etot)

    # Normalized depth u in [0, 1] across active layers (max over axes d)
    nodes = solver.mesh.root.nodes
    lo = np.array([v0[0] for v0 in nodes])
    hi = np.array([v0[-1] for v0 in nodes])
    th = cpml.thickness
    fpos = solver.face_cen[solver.faces]
    epos = solver.edge_mid[solver.free_edges]
    u_f = np.zeros(nf)
    u_e = np.zeros(ne)
    for d in range(3):
        for side in (0, 1):
            t = th[d, side]
            if t <= 0:
                continue
            df = (lo[d] + t - fpos[:, d]) if side == 0 else (fpos[:, d] - (hi[d] - t))
            de = (lo[d] + t - epos[:, d]) if side == 0 else (epos[:, d] - (hi[d] - t))
            u_f = np.maximum(u_f, np.clip(df / t, 0.0, 1.0))
            u_e = np.maximum(u_e, np.clip(de / t, 0.0, 1.0))

    quartiles = []
    for q in range(4):
        u0 = 0.25 * q
        u1 = 0.25 * (q + 1)
        mf = (u_f > u0) & (u_f <= u1) if q > 0 else (u_f > 0.0) & (u_f <= u1)
        me = (u_e > u0) & (u_e <= u1) if q > 0 else (u_e > 0.0) & (u_e <= u1)
        quartiles.append(float((eb[mf].sum() + ee[me].sum()) / etot))

    return {
        "lambda_max": lam,
        "re_over_sigma_max": float(lam.real / (cpml.sigma_max + 1e-300)),
        "freq_hz": float(abs(lam.imag) / (2 * np.pi)),
        "block_energy_share": block_share,
        "interface_face_share_of_b": if_face_share_b,
        "interface_edge_share_of_e": if_edge_share_e,
        "interface_total_share": if_total_share,
        "depth_quartile_share": quartiles,
    }
