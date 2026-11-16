# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Dispersive (Drude/Lorentz ADE) and subpixel conformal permittivity for
`SubgridFIT3D` and `SubgridLTSFIT3D`.

Part 1 — Dispersive ADE inside patches
--------------------------------------
For each dispersive material ``k`` (a `wakis.dispersive.DispersiveMaterial`
with Lorentz and/or Drude electric poles, ``gamma >= 0``, no Debye or magnetic
poles and ``sigma == 0``) occupying a region of leaf cells:

1. The high-frequency permittivity ``eps_inf`` enters the all-edge mass
   ``m_all`` in place of ``eps_r`` on the cells of material ``k``.
2. The all-edge pole mass ``m_{p,k} = eps_0 sum_{cells of k} w`` (with the
   `CartesianBlock.cell_edges` geometric weights ``w``) is lumped onto the
   free edges with the same row-sum lumping as ``D``:

       M_k = |P_e|^T (m_{p,k} o |P_e| 1).

   When one material fills every active cell, ``m_all = eps_inf m_{p,k}``, so
   ``M_k = D / eps_inf`` identically on every free edge.
3. Each pole ``p`` carries a normalised polarisation ``p_{k,p}^n`` [V] on the
   free edges where ``M_k > 0``, updated with the second-order ADE of
   `wakis.dispersive._ade_coefficients` driven by ``e^n``:

       p_{k,p}^{n+1} = alpha p_{k,p}^n + xi p_{k,p}^{n-1} + zeta e^n,
       D (e^{n+1} - e^n) = dt (K^H W b^{n+1/2} - P_e^H i^{n+1/2})
                           - sum_k M_k sum_p (p_{k,p}^{n+1} - p_{k,p}^n).

4. For lossless poles (``gamma = 0``), eliminating ``b`` and any Drude poles
   (``omega_0 = 0``, where ``p^{n+1} - 2 p^n + p^{n-1} = dt^2 omega_p^2 e^n``)
   puts the coupled update on ``x = [e; p_Lorentz]`` in Störmer leapfrog form

       x^{n+1} - 2 x^n + x^{n-1} = -dt^2 M^-1 K x^n

   with symmetric positive definite ``M = diag(D, (omega_0^2 / s) M_k, ...)``
   and symmetric positive semi-definite ``K`` (`dispersive_pencil`). The
   quadratic energy `dispersive_energy` is conserved to round-off and positive
   definite iff ``dt < 2 / sqrt(lambda_max(M^-1 K))`` (`dispersive_dt_max`).

Part 2 — Subpixel (Kottke) permittivity averaging
-------------------------------------------------
For each ``(cell, edge)`` quarter region ``Q`` (full cell length along the
edge, half widths toward it, geometric weight ``w_Q`` proportional to its
volume ``V_Q = w_Q L_e^2``), the dual-volume averages ``<eps>_e`` and
``<1/eps>_e`` are accumulated over the active quarters of the edge's own block
along with the unit interface normal ``n_e`` (from an optional callable or the
least-squares gradient of ``eps`` over the dual volume). The diagonal Kottke
projection

    1 / eps_eff = n_a^2 <1/eps>_e + (1 - n_a^2) / <eps>_e

(``a`` the edge direction) sets ``m_e = eps_0 eps_eff sum_Q w_Q`` before the
usual row-sum lumping ``D = |P_e|^T (m_all o |P_e| 1)``.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.constants import epsilon_0 as EPS0

from ..dispersive import Debye, DispersiveMaterial, Lorentz, _ade_coefficients

__all__ = [
    "validate_dispersive_spec",
    "assemble_subgrid_media",
    "setup_dispersive_groups",
    "update_dispersive_dt",
    "step_dispersive",
    "dispersive_pencil",
    "dispersive_dt_max",
    "dispersive_eigenfrequencies",
    "dispersive_energy",
]


def _eval_mask(region, X, Y, Z):
    if region is None:
        return np.zeros(X.shape, bool)
    return np.broadcast_to(np.asarray(region(X, Y, Z), bool), X.shape)


def evaluate_eps_r(eps_r, X, Y, Z, default=1.0):
    """Evaluate `eps_r` (scalar, callable, Box, or sequence of overrides)."""
    if eps_r is None:
        return np.full(X.shape, default, float)
    if np.isscalar(eps_r):
        return np.full(X.shape, float(eps_r), float)
    if (
        hasattr(eps_r, "b")
        and hasattr(eps_r, "value")
        and not isinstance(eps_r.value, bool)
    ):
        return np.where(eps_r(X, Y, Z), float(eps_r.value), float(default))
    if isinstance(eps_r, (list, tuple)):
        out = np.full(X.shape, default, float)
        for item in eps_r:
            if np.isscalar(item):
                out = np.full(X.shape, float(item), float)
            elif (
                hasattr(item, "b")
                and hasattr(item, "value")
                and not isinstance(item.value, bool)
            ):
                out = np.where(item(X, Y, Z), float(item.value), out)
            else:
                reg, val = item
                out = np.where(_eval_mask(reg, X, Y, Z), float(val), out)
        return out
    return np.broadcast_to(np.asarray(eps_r(X, Y, Z), float), X.shape).copy()


def _extract_box_rules(eps_r, dispersive):
    """
    If `eps_r` and `dispersive` consist solely of scalars and axis-aligned
    `Box` regions, return ``(bg_eps, rules)`` where each rule is
    ``(bounds (3, 2), eps_val, disp_idx_or_None)``. Otherwise return ``None``.
    """
    bg = 1.0
    rules = []
    if eps_r is None:
        pass
    elif np.isscalar(eps_r):
        bg = float(eps_r)
    elif (
        hasattr(eps_r, "b")
        and hasattr(eps_r, "value")
        and not isinstance(eps_r.value, bool)
    ):
        rules.append((np.asarray(eps_r.b, float), float(eps_r.value), None))
    elif isinstance(eps_r, (list, tuple)):
        for item in eps_r:
            if np.isscalar(item):
                bg = float(item)
                rules.clear()
            elif (
                hasattr(item, "b")
                and hasattr(item, "value")
                and not isinstance(item.value, bool)
            ):
                rules.append((np.asarray(item.b, float), float(item.value), None))
            elif (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and hasattr(item[0], "b")
            ):
                rules.append((np.asarray(item[0].b, float), float(item[1]), None))
            else:
                return None
    else:
        return None

    for k, (reg, mat) in enumerate(dispersive):
        if hasattr(reg, "b"):
            rules.append((np.asarray(reg.b, float), float(mat.eps_inf), k))
        else:
            return None
    return bg, rules


def validate_dispersive_spec(dispersive):
    """Validate ``dispersive=[(region, DispersiveMaterial), ...]``."""
    if dispersive is None:
        return []
    out = []
    for item in dispersive:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError(
                "dispersive entries must be (region, DispersiveMaterial) pairs"
            )
        reg, mat = item
        if not isinstance(mat, DispersiveMaterial):
            raise TypeError(f"expected DispersiveMaterial, got {type(mat).__name__}")
        if mat.debye_poles or any(isinstance(p, Debye) for p in mat.poles):
            raise NotImplementedError(
                "Debye poles are not supported in wakis.subgrid3d"
            )
        if mat.has_magnetic:
            raise NotImplementedError(
                "Dispersive or non-unit magnetic permeability is not supported in wakis.subgrid3d"
            )
        if mat.sigma != 0.0:
            raise NotImplementedError(
                "Static conductivity in DispersiveMaterial is not supported in wakis.subgrid3d"
            )
        if getattr(mat, "is_gain", False) or getattr(mat, "is_nonlinear", False):
            raise NotImplementedError(
                "Gain and nonlinear media are not supported in wakis.subgrid3d"
            )
        for pole in mat.poles:
            if not isinstance(pole, Lorentz):
                raise NotImplementedError(
                    f"Unsupported pole type {type(pole).__name__} in wakis.subgrid3d"
                )
        out.append((reg, mat))
    return out


def _quarter_bounds(blk):
    """
    Quarter regions of `blk` in the exact order of `blk.cell_edges()`.

    Returns
    -------
    cell : (12 n_cells,) int
    edge : (12 n_cells,) int
    w : (12 n_cells,) float
        Geometric weight ``h_b h_c / (4 h_a)`` [m].
    vol : (12 n_cells,) float
        Quarter volume ``h_a h_b h_c / 4`` [m^3].
    q_lo, q_hi : (12 n_cells, 3) float
        Lower and upper coordinates [m] of each quarter region.
    """
    ijk = blk.cell_grid()
    cell = np.ravel_multi_index(tuple(ijk), blk.nc).ravel()
    lo_c = [blk.nodes[a][ijk[a]].ravel() for a in range(3)]
    hi_c = [blk.nodes[a][ijk[a] + 1].ravel() for a in range(3)]
    hc = [blk.h[a][ijk[a]].ravel() for a in range(3)]
    cs, es, ws, vols, qlos, qhis = [], [], [], [], [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        w = hc[b] * hc[c] / 4.0 / hc[a]
        vol = hc[a] * hc[b] * hc[c] / 4.0
        for db in (0, 1):
            for dc in (0, 1):
                q = [x.copy() for x in ijk]
                q[b] = q[b] + db
                q[c] = q[c] + dc
                e, _ = blk.edge_index(a, *q)
                qlo = [None, None, None]
                qhi = [None, None, None]
                qlo[a], qhi[a] = lo_c[a], hi_c[a]
                qlo[b] = lo_c[b] + 0.5 * db * hc[b]
                qhi[b] = lo_c[b] + 0.5 * (db + 1) * hc[b]
                qlo[c] = lo_c[c] + 0.5 * dc * hc[c]
                qhi[c] = lo_c[c] + 0.5 * (dc + 1) * hc[c]
                cs.append(cell)
                es.append(e.ravel())
                ws.append(w)
                vols.append(vol)
                qlos.append(np.stack(qlo, axis=1))
                qhis.append(np.stack(qhi, axis=1))
    return (
        np.concatenate(cs),
        np.concatenate(es),
        np.concatenate(ws),
        np.concatenate(vols),
        np.concatenate(qlos),
        np.concatenate(qhis),
    )


def _subpixel_block_edges(
    blk, act, pc, eps_r, disp_spec, n_sub=4, normal=None, mutate=None
):
    """
    Compute per-edge mass ``m_blk`` and per-dispersive-group pole mass
    ``m_p_blk`` on one block using subpixel Kottke averaging.
    """
    c, e, w, vol, qlo, qhi = _quarter_bounds(blk)
    ok = act[c] & ~pc[c]
    c_ok, e_ok, w_ok, vol_ok = c[ok], e[ok], w[ok], vol[ok]
    qlo_ok, qhi_ok = qlo[ok], qhi[ok]
    nQ = len(c_ok)
    nK = len(disp_spec)

    q_mid = 0.5 * (qlo_ok + qhi_ok)
    q_d = qhi_ok - qlo_ok

    box_info = _extract_box_rules(eps_r, disp_spec)
    if box_info is not None:
        bg, rules = box_info
        eps_q = np.full(nQ, bg, float)
        ieps_q = np.full(nQ, 1.0 / bg, float)
        # First moment integral int_Q (r - r_Q) eps(r) dV / V_Q
        mom_q = np.zeros((nQ, 3), float)
        disp_frac_q = np.zeros((nQ, nK), float)

        # Start from quarter-centre values, then refine any quarter touched by a box plane
        cut = np.zeros(nQ, bool)
        for bnds, val, d_idx in rules:
            in_mid = (
                (q_mid[:, 0] > bnds[0, 0])
                & (q_mid[:, 0] < bnds[0, 1])
                & (q_mid[:, 1] > bnds[1, 0])
                & (q_mid[:, 1] < bnds[1, 1])
                & (q_mid[:, 2] > bnds[2, 0])
                & (q_mid[:, 2] < bnds[2, 1])
            )
            eps_q = np.where(in_mid, val, eps_q)
            ieps_q = np.where(in_mid, 1.0 / val, ieps_q)
            if d_idx is not None:
                disp_frac_q[in_mid, :] = 0.0
                disp_frac_q[in_mid, d_idx] = 1.0
            else:
                disp_frac_q[in_mid, :] = 0.0
            # A quarter is cut if it overlaps the box AND is not strictly inside it
            overlaps = (
                (qhi_ok[:, 0] > bnds[0, 0] + 1e-15)
                & (qlo_ok[:, 0] < bnds[0, 1] - 1e-15)
                & (qhi_ok[:, 1] > bnds[1, 0] + 1e-15)
                & (qlo_ok[:, 1] < bnds[1, 1] - 1e-15)
                & (qhi_ok[:, 2] > bnds[2, 0] + 1e-15)
                & (qlo_ok[:, 2] < bnds[2, 1] - 1e-15)
            )
            strictly_in = (
                (qlo_ok[:, 0] >= bnds[0, 0] - 1e-15)
                & (qhi_ok[:, 0] <= bnds[0, 1] + 1e-15)
                & (qlo_ok[:, 1] >= bnds[1, 0] - 1e-15)
                & (qhi_ok[:, 1] <= bnds[1, 1] + 1e-15)
                & (qlo_ok[:, 2] >= bnds[2, 0] - 1e-15)
                & (qhi_ok[:, 2] <= bnds[2, 1] + 1e-15)
            )
            cut |= overlaps & ~strictly_in

        for qi in np.flatnonzero(cut):
            lo_i, hi_i = qlo_ok[qi], qhi_ok[qi]
            splits = []
            for a in range(3):
                pts = [lo_i[a], hi_i[a]]
                for bnds, _, _ in rules:
                    for p in bnds[a]:
                        if lo_i[a] + 1e-15 < p < hi_i[a] - 1e-15:
                            pts.append(float(p))
                splits.append(np.unique(pts))
            sx, sy, sz = splits
            dx = np.diff(sx)
            dy = np.diff(sy)
            dz = np.diff(sz)
            mx = 0.5 * (sx[:-1] + sx[1:])
            my = 0.5 * (sy[:-1] + sy[1:])
            mz = 0.5 * (sz[:-1] + sz[1:])
            MX, MY, MZ = np.meshgrid(mx, my, mz, indexing="ij")
            DV = np.meshgrid(dx, dy, dz, indexing="ij")
            sub_v = (DV[0] * DV[1] * DV[2]).ravel()
            sub_e = np.full(sub_v.shape, bg, float)
            sub_d = np.full(sub_v.shape, -1, int)
            px, py, pz = MX.ravel(), MY.ravel(), MZ.ravel()
            for bnds, val, d_idx in rules:
                ins = (
                    (px > bnds[0, 0])
                    & (px < bnds[0, 1])
                    & (py > bnds[1, 0])
                    & (py < bnds[1, 1])
                    & (pz > bnds[2, 0])
                    & (pz < bnds[2, 1])
                )
                sub_e = np.where(ins, val, sub_e)
                sub_d = np.where(ins, -1 if d_idx is None else d_idx, sub_d)
            v_tot = vol_ok[qi]
            eps_q[qi] = np.sum(sub_v * sub_e) / v_tot
            ieps_q[qi] = np.sum(sub_v / sub_e) / v_tot
            r_sub = np.stack([px, py, pz], axis=1)
            mom_q[qi] = (
                np.sum(sub_v[:, None] * (r_sub - q_mid[qi]) * sub_e[:, None], axis=0)
                / v_tot
            )
            for k in range(nK):
                disp_frac_q[qi, k] = np.sum(sub_v[sub_d == k]) / v_tot
    else:
        ns = max(1, int(n_sub))
        t = (np.arange(ns) + 0.5) / ns
        tx, ty, tz = np.meshgrid(t, t, t, indexing="ij")
        t_flat = np.stack([tx.ravel(), ty.ravel(), tz.ravel()], axis=1)  # (ns^3, 3)
        n_pts = t_flat.shape[0]
        eps_q = np.empty(nQ, float)
        ieps_q = np.empty(nQ, float)
        mom_q = np.empty((nQ, 3), float)
        disp_frac_q = np.zeros((nQ, nK), float)
        chunk_size = max(1, 20000 // max(1, n_pts))
        for s0 in range(0, nQ, chunk_size):
            s1 = min(nQ, s0 + chunk_size)
            lo_c = qlo_ok[s0:s1, None, :]
            d_c = q_d[s0:s1, None, :]
            pts = lo_c + d_c * t_flat[None, :, :]
            X, Y, Z = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]
            eq = evaluate_eps_r(eps_r, X, Y, Z, 1.0)
            df = np.zeros((s1 - s0, n_pts, nK), float)
            for k, (reg, mat) in enumerate(disp_spec):
                mk = _eval_mask(reg, X, Y, Z)
                eq = np.where(mk, float(mat.eps_inf), eq)
                df[:, :, :] = np.where(mk[:, :, None], 0.0, df)
                df[:, :, k] = np.where(mk, 1.0, df[:, :, k])
            eps_q[s0:s1] = eq.mean(axis=1)
            ieps_q[s0:s1] = (1.0 / eq).mean(axis=1)
            rc = pts - q_mid[s0:s1, None, :]
            mom_q[s0:s1] = (rc * eq[:, :, None]).mean(axis=1)
            if nK:
                disp_frac_q[s0:s1] = df.mean(axis=1)

    nE_blk = blk.n_edges
    w_sum = np.bincount(e_ok, weights=w_ok, minlength=nE_blk)
    vol_sum = np.bincount(e_ok, weights=vol_ok, minlength=nE_blk)
    q_weight = np.ones_like(vol_ok) if mutate == "count_weights" else vol_ok
    qw_sum = np.bincount(e_ok, weights=q_weight, minlength=nE_blk)

    has_q = qw_sum > 0
    eps_avg = np.ones(nE_blk, float)
    ieps_avg = np.ones(nE_blk, float)
    eps_avg[has_q] = (
        np.bincount(e_ok, weights=q_weight * eps_q, minlength=nE_blk)[has_q]
        / qw_sum[has_q]
    )
    ieps_avg[has_q] = (
        np.bincount(e_ok, weights=q_weight * ieps_q, minlength=nE_blk)[has_q]
        / qw_sum[has_q]
    )

    emid, _, edir = blk.edge_geometry()
    if normal is not None:
        nv = np.asarray(normal(emid[:, 0], emid[:, 1], emid[:, 2]), float)
        if nv.ndim == 1:
            nv = np.broadcast_to(nv, (nE_blk, 3))
        elif nv.shape[0] == 3 and nv.shape != (nE_blk, 3):
            nv = nv.T
        nrm = np.linalg.norm(nv, axis=1)
        n_hat = np.zeros((nE_blk, 3), float)
        nz = nrm > 1e-15
        n_hat[nz] = nv[nz] / nrm[nz, None]
    else:
        # Dual-volume centroid and first moment int_{V_e} (r - r_e) (eps - <eps>_e) dV
        r_bar = np.zeros((nE_blk, 3), float)
        for ax in range(3):
            r_bar[has_q, ax] = (
                np.bincount(e_ok, weights=vol_ok * q_mid[:, ax], minlength=nE_blk)[
                    has_q
                ]
                / vol_sum[has_q]
            )
        dr = q_mid - r_bar[e_ok]
        # Total dipole moment = sum_Q V_Q [ mom_q + (r_Q - r_bar_e) (eps_Q - <eps>_e) ]
        # Normalized by the variance int_{V_e} (r_ax - r_bar_e,ax)^2 dV along each axis
        g_vec = np.zeros((nE_blk, 3), float)
        for ax in range(3):
            dip_ax = np.bincount(
                e_ok,
                weights=vol_ok * (mom_q[:, ax] + dr[:, ax] * (eps_q - eps_avg[e_ok])),
                minlength=nE_blk,
            )
            var_ax = np.bincount(
                e_ok,
                weights=vol_ok * (q_d[:, ax] ** 2 / 12.0 + dr[:, ax] ** 2),
                minlength=nE_blk,
            )
            ok_v = has_q & (var_ax > 1e-30)
            g_vec[ok_v, ax] = dip_ax[ok_v] / var_ax[ok_v]
        gn = np.linalg.norm(g_vec, axis=1)
        n_hat = np.zeros((nE_blk, 3), float)
        nz = gn > 1e-14 * np.maximum(eps_avg, 1.0) / np.maximum(np.min(q_d), 1e-12)
        n_hat[nz] = g_vec[nz] / gn[nz, None]

    na = n_hat[np.arange(nE_blk), edir]
    na2 = na**2
    if mutate == "swap_projection":
        na2 = 1.0 - na2
    inv_eps_eff = na2 * ieps_avg + (1.0 - na2) / np.maximum(eps_avg, 1e-30)
    eps_eff = np.where(has_q, 1.0 / np.maximum(inv_eps_eff, 1e-30), 0.0)
    m_blk = EPS0 * eps_eff * w_sum

    m_p_blk = []
    for k in range(nK):
        m_p_blk.append(
            EPS0 * np.bincount(e_ok, weights=w_ok * disp_frac_q[:, k], minlength=nE_blk)
        )
    return m_blk, m_p_blk


def assemble_subgrid_media(
    solver, eps_r, pec_mask_per_block, dispersive=None, averaging="staircase",
    n_sub=4, normal=None, mutate=None,
):  # fmt: skip
    """
    Assemble the all-edge permittivity mass ``m_all`` and per-dispersive-group
    all-edge pole masses ``m_p_list`` across all blocks of `solver.mesh`.
    """
    if averaging not in ("staircase", "subpixel"):
        raise ValueError(
            f"averaging must be 'staircase' or 'subpixel', got {averaging!r}"
        )
    disp_spec = validate_dispersive_spec(dispersive)
    M = solver.mesh
    nE = M.n_edges
    m_all = np.zeros(nE, float)
    m_p_list = [np.zeros(nE, float) for _ in disp_spec]
    solver.eps_cells = []

    for bi, blk in enumerate(M.blocks):
        X, Y, Z = blk.cell_centres()
        eps_c = evaluate_eps_r(eps_r, X, Y, Z, 1.0).ravel()
        disp_id = np.full(eps_c.shape, -1, int)
        for k, (reg, mat) in enumerate(disp_spec):
            mk = _eval_mask(reg, X, Y, Z).ravel()
            eps_c = np.where(mk, float(mat.eps_inf), eps_c)
            disp_id = np.where(mk, k, disp_id)
        solver.eps_cells.append(eps_c)

        act = M.active[bi].ravel()
        pc = pec_mask_per_block[bi]
        e0, e1 = M.edge_off[bi], M.edge_off[bi + 1]

        if averaging == "staircase":
            c, e, w = blk.cell_edges()
            ok = act[c]
            np.add.at(m_all, e0 + e[ok], EPS0 * eps_c[c[ok]] * w[ok])
            ok_d = ok & ~pc[c]
            for k in range(len(disp_spec)):
                ok_k = ok_d & (disp_id[c] == k)
                np.add.at(m_p_list[k], e0 + e[ok_k], EPS0 * w[ok_k])
        else:
            m_blk, mp_blk = _subpixel_block_edges(
                blk,
                act,
                pc,
                eps_r,
                disp_spec,
                n_sub=n_sub,
                normal=normal,
                mutate=mutate,
            )
            m_all[e0:e1] = m_blk
            for k in range(len(disp_spec)):
                m_p_list[k][e0:e1] = mp_blk[k]

    return m_all, disp_spec, m_p_list


def setup_dispersive_groups(solver, disp_spec, m_p_list, mutate=None):
    """Lump ``m_p_list`` onto the free edges and build ``solver.dispersive_groups``."""
    absP = abs(solver.P)
    row_sum = np.asarray(absP.sum(axis=1)).ravel()
    groups = []
    for (reg, mat), m_pk in zip(disp_spec, m_p_list):
        if mutate == "lump_no_rowsum":
            M_k = np.asarray(absP.T @ m_pk).ravel()
        else:
            M_k = np.asarray(absP.T @ (m_pk * row_sum)).ravel()
        idx = np.flatnonzero(np.abs(M_k) > 0)
        M_act = M_k[idx]
        groups.append(
            {
                "material": mat,
                "region": reg,
                "m_p_all": m_pk,
                "M_full": M_k,
                "idx": idx,
                "M": M_act,
                "ratio": M_act / solver.D[idx] if idx.size else np.zeros(0, float),
                "poles": list(mat.lorentz_poles),
                "coef": [],
                "p": [],
                "p_old": [],
            }
        )
    solver.dispersive_groups = groups
    return groups


def update_dispersive_dt(solver, dt, mutate=None):
    """Compute ADE coefficients for ``dt`` on ``solver.dispersive_groups``."""
    dt = float(dt)
    mut = solver.mutate if mutate is None else mutate
    for g in solver.dispersive_groups:
        lorentz, _ = _ade_coefficients(g["material"], dt)
        if mut == "strength":
            lorentz = [(a, x, 1.01 * z) for a, x, z in lorentz]
        g["coef"] = lorentz


def step_dispersive(solver, e_n):
    """Advance the ADE polarisations from ``e_n`` and apply the feedback to ``solver.e``."""
    for g in solver.dispersive_groups:
        idx = g["idx"]
        if idx.size == 0:
            continue
        en = e_n[idx]
        dp = np.zeros(idx.size, dtype=solver.dtype)
        for k, (alpha, xi, zeta) in enumerate(g["coef"]):
            p, po = g["p"][k], g["p_old"][k]
            p_new = alpha * p + xi * po + zeta * en
            dp = dp + (p_new - p)
            g["p_old"][k], g["p"][k] = p, p_new
        solver.e[idx] = solver.e[idx] - g["ratio"] * dp


def dispersive_pencil(solver):
    """
    Assemble the sparse second-order pencil ``(K_mat, M_vec, lorentz_info)``
    of the lossless coupled system ``M x_tt + K x = 0`` on
    ``x = [e; p_Lorentz]``.

    Drude poles (``omega_0 == 0``) are eliminated and add ``omega_p^2 M_k``
    to the ``e-e`` block of ``K_mat``; each Lorentz pole (``omega_0 > 0``,
    ``s = delta_eps * omega_0^2``) adds a block of size ``len(idx)`` with
    mass ``(omega_0^2 / s) M_k`` and 2x2 stiffness block
    ``[[s M_k, -omega_0^2 M_k], [-omega_0^2 M_k, (omega_0^4 / s) M_k]]``.
    """
    n_e = solver.n_free
    S = (solver.KH @ sp.diags(solver.W) @ solver.K).tocsr()
    ee_diag = np.zeros(n_e, float)
    m_parts = [np.real(solver.D).astype(float).copy()]
    lorentz_info = []
    offset = n_e

    for gi, g in enumerate(solver.dispersive_groups):
        idx = g["idx"]
        M_k = np.real(g["M"]).astype(float)
        if idx.size == 0:
            continue
        for pi, pole in enumerate(g["poles"]):
            s = pole.strength
            w0 = pole.omega_0
            if w0 == 0.0:
                ee_diag[idx] += s * M_k
            else:
                m = idx.size
                sl = slice(offset, offset + m)
                ee_diag[idx] += s * M_k
                m_parts.append((w0**2 / s) * M_k)
                lorentz_info.append(
                    {
                        "group": gi,
                        "pole": pi,
                        "idx": idx,
                        "slice": sl,
                        "M": M_k,
                        "w0": w0,
                        "s": s,
                        "coup": -(w0**2) * M_k,
                        "pp_diag": (w0**4 / s) * M_k,
                    }
                )
                offset += m

    M_vec = np.concatenate(m_parts)
    rows, cols, vals = [], [], []
    S_coo = (S + sp.diags(ee_diag)).tocoo()
    rows.append(S_coo.row)
    cols.append(S_coo.col)
    vals.append(S_coo.data)

    for info in lorentz_info:
        idx = info["idx"]
        p_idx = np.arange(info["slice"].start, info["slice"].stop)
        rows.extend([idx, p_idx, p_idx])
        cols.extend([p_idx, idx, p_idx])
        vals.extend([info["coup"], info["coup"], info["pp_diag"]])

    K_mat = sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(offset, offset),
    )
    return K_mat, M_vec, lorentz_info


def _apply_pencil_K(solver, x, ee_diag, lorentz_info):
    """Matrix-free product ``K_mat @ x`` using ``solver.K`` and ``solver.KH``."""
    n_e = solver.n_free
    e = x[:n_e]
    out = np.empty_like(x)
    out[:n_e] = solver.KH @ (solver.W * (solver.K @ e)) + ee_diag * e
    for info in lorentz_info:
        idx = info["idx"]
        sl = info["slice"]
        p = x[sl]
        out[:n_e][idx] += info["coup"] * p
        out[sl] = info["coup"] * e[idx] + info["pp_diag"] * p
    return out


def dispersive_dt_max(solver, tol=1e-12):
    """
    Lossless stability limit ``2 / sqrt(lambda_max(M^-1 K))`` [s] by Lanczos
    on ``M^{-1/2} K M^{-1/2}``.
    """
    if not getattr(solver, "dispersive_groups", None):
        return solver.dt_max
    _, M_vec, lorentz_info = dispersive_pencil(solver)
    n_e = solver.n_free
    ee_diag = np.zeros(n_e, float)
    for g in solver.dispersive_groups:
        idx = g["idx"]
        M_k = np.real(g["M"]).astype(float)
        for pole in g["poles"]:
            ee_diag[idx] += pole.strength * M_k

    inv_sqM = 1.0 / np.sqrt(M_vec)
    n = M_vec.size
    dtype = solver.dtype

    def mv(z):
        x = inv_sqM * np.asarray(z).ravel()
        return inv_sqM * _apply_pencil_K(solver, x, ee_diag, lorentz_info)

    op = sla.LinearOperator((n, n), matvec=mv, dtype=dtype)
    v0 = np.random.default_rng(1).standard_normal(n).astype(dtype)
    lam = sla.eigsh(
        op, k=1, which="LA", tol=tol, v0=v0, return_eigenvectors=False, ncv=min(n, 40)
    )
    return float(2.0 / np.sqrt(float(np.real(lam[0]))))


def dispersive_eigenfrequencies(solver, k, f_guess):
    """
    The `k` lowest semi-discrete resonance frequencies above ``0.5 f_guess`` [Hz]
    of the coupled dispersive pencil ``K x = omega^2 M x`` (shift-invert Lanczos).
    """
    if not getattr(solver, "dispersive_groups", None):
        return solver.eigenfrequencies(k, f_guess)
    K_mat, M_vec, lorentz_info = dispersive_pencil(solver)
    sigma = (2 * np.pi * 0.5 * f_guess) ** 2
    for info in lorentz_info:
        if abs(info["w0"] ** 2 - sigma) < 1e-6 * info["w0"] ** 2:
            sigma *= 0.97
    n_e = solver.n_free
    n_tot = K_mat.shape[0]
    S = (solver.KH @ sp.diags(solver.W) @ solver.K).tocsc()
    diag_e = -sigma * np.real(solver.D).astype(float)
    for g in solver.dispersive_groups:
        idx = g["idx"]
        M_k = np.real(g["M"]).astype(float)
        for pole in g["poles"]:
            if pole.omega_0 == 0.0:
                diag_e[idx] += pole.strength * M_k
            else:
                diag_e[idx] -= sigma * (pole.strength / (pole.omega_0**2 - sigma)) * M_k
    lu = sla.splu((S + sp.diags(diag_e)).tocsc(), permc_spec="MMD_AT_PLUS_A")

    def solve_shifted(r):
        r = np.asarray(r, dtype=float).ravel()
        rhs_e = r[:n_e].copy()
        for info in lorentz_info:
            fac = info["s"] / (info["w0"] ** 2 - sigma)
            rhs_e[info["idx"]] += fac * r[info["slice"]]
        e_sol = lu.solve(rhs_e)
        out = np.empty(n_tot, dtype=float)
        out[:n_e] = e_sol
        for info in lorentz_info:
            fac = info["s"] / (info["w0"] ** 2 - sigma)
            inv_d = fac / (info["w0"] ** 2 * info["M"])
            out[info["slice"]] = fac * e_sol[info["idx"]] + inv_d * r[info["slice"]]
        return out

    op_inv = sla.LinearOperator((n_tot, n_tot), matvec=solve_shifted, dtype=float)
    kk = min(n_tot - 2, k + 8)
    v0 = np.random.default_rng(0).standard_normal(n_tot)
    lam = sla.eigsh(
        K_mat,
        k=kk,
        M=sp.diags(M_vec).tocsc(),
        sigma=sigma,
        OPinv=op_inv,
        which="LA",
        ncv=min(n_tot - 1, max(4 * kk, 40)),
        tol=1e-12,
        v0=v0,
        return_eigenvectors=False,
    )
    return (np.sort(np.sqrt(np.abs(lam))) / (2 * np.pi))[:k]


def dispersive_energy(solver, e0, e1, p0_list=None, p1_list=None):
    """
    Conserved second-order energy of the lossless coupled leapfrog:
    ``1/2 [w^H (M - dt^2/4 K) w + ybar^H K ybar]`` with
    ``w = (x^{n+1} - x^n) / dt`` and ``ybar = (x^{n+1} + x^n) / 2`` on
    ``x = [e; p_Lorentz]``.
    """
    e0 = e0.get() if hasattr(e0, "get") else np.asarray(e0)
    e1 = e1.get() if hasattr(e1, "get") else np.asarray(e1)
    if not hasattr(solver, "_disp_pencil_cache"):
        _, M_vec, lorentz_info = dispersive_pencil(solver)
        ee_diag = np.zeros(solver.n_free, float)
        for g in solver.dispersive_groups:
            idx = g["idx"]
            M_k = np.real(g["M"]).astype(float)
            for pole in g["poles"]:
                ee_diag[idx] += pole.strength * M_k
        solver._disp_pencil_cache = (M_vec, ee_diag, lorentz_info)
    M_vec, ee_diag, lorentz_info = solver._disp_pencil_cache

    x0_parts = [e0]
    x1_parts = [e1]
    for li, info in enumerate(lorentz_info):
        g = solver.dispersive_groups[info["group"]]
        pi = info["pole"]
        p0 = g["p_old"][pi] if p0_list is None else p0_list[li]
        p1 = g["p"][pi] if p1_list is None else p1_list[li]
        x0_parts.append(p0)
        x1_parts.append(p1)
    x0 = np.concatenate(x0_parts)
    x1 = np.concatenate(x1_parts)
    dt = solver.dt
    w = (x1 - x0) / dt
    yb = 0.5 * (x1 + x0)
    Kw = _apply_pencil_K(solver, w, ee_diag, lorentz_info)
    Kyb = _apply_pencil_K(solver, yb, ee_diag, lorentz_info)
    val = np.vdot(w, M_vec * w) - (dt**2 / 4.0) * np.vdot(w, Kw) + np.vdot(yb, Kyb)
    return float(0.5 * np.real(val))
