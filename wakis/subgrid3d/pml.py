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

__all__ = ["SubgridCPML", "split_curl"]


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
    """

    def __init__(self, solver, dt, thickness, R0=1e-6, order=3.0, alpha_hz=0.0,
                 allow_patches=False):  # fmt: skip
        s = solver
        self.dt = float(dt)
        nodes = s.mesh.root.nodes
        lo = np.array([v[0] for v in nodes])
        hi = np.array([v[-1] for v in nodes])
        th = np.asarray(thickness, float).reshape(3, 2)
        self.thickness = th
        alpha = 2 * np.pi * float(alpha_hz)

        def sigma(d, u):
            out = np.zeros_like(u)
            for side in (0, 1):
                t = th[d, side]
                if t <= 0:
                    continue
                depth = (lo[d] + t - u) if side == 0 else (u - (hi[d] - t))
                x = np.clip(depth / t, 0.0, 1.0)
                smax = -(order + 1) * np.log(R0) * C0 / (2 * t)
                out = out + smax * x**order
            return out

        Kd = split_curl(s)
        self.Kd = Kd
        fpos = s.face_cen[s.faces]
        epos = s.edge_mid[s.free_edges]
        self.pf, self.Kf, self.bh, self.ah = [], [], [], []
        self.pe, self.Ke, self.be, self.ae = [], [], [], []
        dt = self.dt
        for d in range(3):
            K = Kd[d]
            sf = sigma(d, fpos[:, d])
            rows_nz = np.diff(K.indptr) > 0
            pf = np.flatnonzero((sf > 0) & rows_nz)
            b = np.exp(-(sf[pf] + alpha) * dt)
            self.pf.append(pf)
            self.Kf.append(K[pf].tocsr())
            self.bh.append(b)
            self.ah.append(sf[pf] / (sf[pf] + alpha) * (b - 1))
            KH = K.conj().T.tocsr()
            se = sigma(d, epos[:, d])
            cols_nz = np.diff(KH.indptr) > 0
            pe = np.flatnonzero((se > 0) & cols_nz)
            b = np.exp(-(se[pe] + alpha) * dt)
            self.pe.append(pe)
            self.Ke.append(KH[pe].tocsr())
            self.be.append(b)
            self.ae.append(se[pe] / (se[pe] + alpha) * (b - 1))
        self.edges = np.unique(np.concatenate(self.pe))
        self.face_set = np.unique(np.concatenate(self.pf))
        fblk = np.searchsorted(s.mesh.face_off, s.faces, side="right") - 1
        in_patch = np.any(fblk[self.face_set] > 0) or np.any(
            s.edge_block[s.free_edges[self.edges]] > 0
        )
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
