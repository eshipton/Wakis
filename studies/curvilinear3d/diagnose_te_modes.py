# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Diagnose why cylinder ``TE111`` and sphere ``TE_l1`` modes do not converge on
the single-block elliptic cube-to-disk and cube-to-ball hexahedral mappings
in `CurvilinearFIT3D`.

Overview of the four diagnostics
--------------------------------
1. **Diagnostic 1 (Mode identification by overlap)**:
   Solves the generalized curl-curl eigenvalue problem
   ``W_eps_u C_u^T W_nu C_u W_eps_u v = omega^2 W_eps_u v`` near the target
   analytic resonance ``omega_ana`` using shift-invert Lanczos (`eigsh` with
   ``sigma = omega_ana^2`` for ``k = 12`` eigenpairs, plus the full dense
   `eigh` spectrum at ``N in {6, 8}``). Constructs the discrete analytic
   electric flux vector ``d_ana,e = S_e . E_ana(x_mid,e)`` [C] on unknown
   edges from the whole dual-face area vector ``S_e`` [m^2] and selects the
   discrete mode maximising the normalised ``W_eps_u`` inner-product overlap

   .. math::

       \\mathcal{O}(v, d_{\\text{ana}})
       = \\frac{|v^T W_{\\varepsilon, u} d_{\\text{ana}}|}
               {\\sqrt{v^T W_{\\varepsilon, u} v}\\,\\sqrt{d_{\\text{ana}}^T W_{\\varepsilon, u} d_{\\text{ana}}}}.

   *Interpretation rule (Diagnostic 1)*: If the overlap-selected mode and the
   nearest-eigenvalue mode have the same discrete frequency with overlap near
   unity and every near-zero eigenvalue belongs to the interior-node gradient
   nullspace ``im(W_eps_u^{-1} G_u)``, then mode misidentification, degenerate
   pair splitting, and spurious modes are ruled out as the cause of
   non-convergence.

2. **Diagnostic 2 (Smooth polar map on a PEC quarter-annular cylinder)**:
   Separates the cube-to-disk corner mapping singularity from the underlying
   curvilinear FIT scheme by meshing a coaxial quarter-annulus
   ``r in [r_in, r_out]``, ``phi in [0, pi / 2]``, ``z in [0, L]`` with a
   smooth cylindrical polar map ``(x, y, z) = (r cos(phi), r sin(phi), z)``
   whose Jacobian determinant ``det J = r >= r_in > 0`` has no singular points.

   *Derivation of the annular TE characteristic equation*:
   In a PEC coaxial sector ``[r_in, r_out] x [0, pi / 2] x [0, L]``, a
   transverse-electric (``E_z = 0``) mode has axial magnetic field

   .. math::

       H_z(r, \\phi, z) = Z_m(k_c r) \\cos(m \\phi) \\sin(k_z z),
       \\qquad Z_m(k_c r) = A J_m(k_c r) + B Y_m(k_c r),

   with ``k_z = p pi / L`` and transverse electric components

   .. math::

       E_r \\propto \\frac{m}{k_c r} Z_m(k_c r) \\sin(m \\phi) \\sin(k_z z),
       \\qquad
       E_\\phi \\propto Z_m'(k_c r) \\cos(m \\phi) \\sin(k_z z).

   The tangential electric field ``E_z = 0`` vanishes identically on all walls;
   ``E_z, E_r, E_phi`` vanish on ``z in {0, L}`` for integer ``p >= 1``; and
   ``E_r`` vanishes on the azimuthal PEC walls ``phi in {0, pi / 2}`` when
   ``sin(m pi / 2) = 0``, so ``m = 2`` is the lowest non-axisymmetric mode.
   The remaining PEC boundary conditions ``E_phi(r_in) = E_phi(r_out) = 0``
   require ``Z_m'(k_c r_in) = Z_m'(k_c r_out) = 0``:

   .. math::

       \\begin{pmatrix}
           J_m'(k_c r_{\\text{in}}) & Y_m'(k_c r_{\\text{in}}) \\\\
           J_m'(k_c r_{\\text{out}}) & Y_m'(k_c r_{\\text{out}})
       \\end{pmatrix}
       \\begin{pmatrix} A \\\\ B \\end{pmatrix} = 0,

   which has a non-trivial solution if and only if the Bessel derivative
   cross-product vanishes:

   .. math::

       f(k_c) = J_m'(k_c r_{\\text{out}}) Y_m'(k_c r_{\\text{in}})
              - J_m'(k_c r_{\\text{in}}) Y_m'(k_c r_{\\text{out}}) = 0.

   With ``k_c`` the first positive root of ``f(k_c) = 0``, the radial profile
   ``Z_m(k_c r) = Y_m'(k_c r_in) J_m(k_c r) - J_m'(k_c r_in) Y_m(k_c r)``
   satisfies both wall conditions and gives the analytic resonant angular
   frequency ``omega_ref = c * sqrt(k_c^2 + (p pi / L)^2)`` [rad/s].

   *Interpretation rule (Diagnostic 2)*: If the convergence order on the smooth
   polar map is about 2, the curvilinear FIT scheme converges at second order on
   smoothly mapped curved PEC walls, so the cube-to-disk/ball corner cells are
   the cause; if not, the scheme itself fails on curved geometry.

3. **Diagnostic 3 (Singular corner conditioning, energy localisation, and PEC wall dual-face angle)**:
   Measures (a) the maximum condition numbers ``cond(M_c)`` and ``cond(S_c)``
   of the 3x3 primal-face and dual-face corner reconstruction matrices and their
   power-law scaling exponents with ``N``; (b) the fraction ``E_loc / E_total``
   of ``W_eps_u`` mode energy supported on edges of cells within Chebyshev
   distance 2 of the singular mapping corners; and (c) the angle
   ``asin(|S_e . n_w| / |S_e|)`` [deg] between the whole dual-face area vector
   ``S_e`` of each tangential PEC wall edge (where ``d_e = S_e . D = 0`` is
   enforced) and the wall tangent plane with unit normal ``n_w``.

   *Interpretation rule (Diagnostic 3)*: If ``cond(M_c)`` and ``cond(S_c)`` grow
   algebraically with ``N`` at the singular corners while ``E_loc / E_total``
   decreases with ``N`` and the wall angle ``asin(|S_e . n_w| / |S_e|)`` is
   large near the corners (and across the sphere wall), then non-convergence of
   modes with ``E_n != 0`` on the wall is driven by non-orthogonal boundary
   dual faces ``S_e`` coupling the tangential PEC constraint ``d_e = 0`` into
   the normal electric flux ``D_n`` together with corner metric degeneration,
   rather than by energy trapping in a corner-localised spurious mode.

4. **Diagnostic 4 (Reconciling the two perturbed-box TE101 order definitions)**:
   Evaluates Definition A (the two-level order between ``4 x 3 x 4`` and
   ``8 x 6 x 8`` cells from ``test_perturbed_box_order_small``) and Definition B
   (the four-point log-log fit over ``N in {6, 8, 12, 16}`` on isotropic
   ``N x N x N`` meshes from ``studies/curvilinear3d/study_cavities.py``) across
   both mesh families.

   *Interpretation rule (Diagnostic 4)*: If Definition A on the anisotropic
   ``(4, 3, 4) -> (8, 6, 8)`` test pair and Definition B on the isotropic
   ``(N, N, N)`` study sequence each reproduce their respective reported orders
   while both mesh families exhibit ``p ~ 1.8 - 2.0``, the discrepancy between
   the two numbers is strictly a difference of mesh family ``(4k, 3k, 4k)`` vs
   ``(N, N, N)`` and refinement range for a displacement scaled as ``0.3 h``.
"""

import argparse
import json
import sys

import numpy as np
from scipy.constants import c as C0
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.sparse.linalg import eigsh
from scipy.special import j1, jnp_zeros, jv, jvp, spherical_jn, yv, yvp

from wakis.curvilinear.mappings3d import (
    box_te101_omega,
    cylinder_nodes,
    cylinder_te111_omega,
    min_normalised_corner_det,
    perturbed_box_nodes,
    sphere_lowest_omegas,
    sphere_lowest_roots,
    sphere_nodes,
)
from wakis.curvilinear.mesh3d import HexBlock3D
from wakis.curvilinear.solver3d import CurvilinearFIT3D

INTERPRETATION_RULES = {
    "diag1": (
        "Rule: If the overlap-maximising mode and the nearest-eigenvalue mode "
        "coincide with overlap near 1 and all near-zero modes lie in the "
        "discrete gradient nullspace, mode misidentification and spurious "
        "modes are ruled out."
    ),
    "diag2": (
        "Rule: If the convergence order on the smooth polar quarter-annulus map "
        "is about 2, the curvilinear FIT scheme converges at second order on "
        "smooth curved walls and the cube-to-disk/ball corner cells are the "
        "cause; if not, the scheme itself fails on this geometry."
    ),
    "diag3": (
        "Rule: If corner reconstruction condition numbers grow as a power of N "
        "and tangential PEC boundary dual faces S_e have a large normal angle "
        "asin(|S_e . n_w| / |S_e|) while corner energy fraction E_loc / E_total "
        "shrinks with N, the error comes from boundary dual-face non-orthogonality "
        "(fixing d_e = S_e . D = 0 constrains normal D_n) and corner Jacobian "
        "degeneration rather than a corner-trapped spurious mode."
    ),
    "diag4": (
        "Rule: If the two-level (4x3x4 -> 8x6x8) definition reproduces the test "
        "order and the four-point (N = 6, 8, 12, 16 on NxNxN) definition "
        "reproduces the study order, the difference is purely the mesh family "
        "and refinement window."
    ),
}


def _fit_loglog_order(ns, errs):
    """Fit power-law convergence order ``p`` in ``err ~ C * (1 / N)^p``."""
    ns_arr = np.asarray(ns, dtype=float)
    err_arr = np.asarray(errs, dtype=float)
    valid = (ns_arr > 0.0) & (err_arr > 0.0) & np.isfinite(err_arr)
    if np.count_nonzero(valid) < 2:
        return None
    slope, _ = np.polyfit(np.log(1.0 / ns_arr[valid]), np.log(err_arr[valid]), 1)
    return float(slope)


def _eval_cylinder_te111_e(pts, a=1.0, length=1.2):
    """Evaluate the analytic ``TE111`` electric field vector ``E(x, y, z)`` [V/m]
    of a PEC circular cylinder of radius ``a`` [m] and length ``length`` [m]."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    x11p = float(jnp_zeros(1, 1)[0])
    kr = x11p * r / float(a)
    s_z = np.sin(np.pi * z / float(length))
    er = (
        np.where(r > 1e-12, j1(kr) / np.where(r > 1e-12, kr, 1.0), 0.5)
        * np.sin(phi)
        * s_z
    )
    ephi = jvp(1, kr) * np.cos(phi) * s_z
    ex = er * np.cos(phi) - ephi * np.sin(phi)
    ey = er * np.sin(phi) + ephi * np.cos(phi)
    ez = np.zeros_like(ex)
    return np.stack([ex, ey, ez], axis=-1)


def _eval_sphere_te_l1_e(pts, a=1.0):
    """Evaluate the analytic ``l = 1`` ``TE_l1`` (root of ``d/dx[x j_1(x)] = 0``)
    electric field vector ``E(x, y, z)`` [V/m] of a PEC sphere of radius ``a`` [m]."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    x1, _ = sphere_lowest_roots(l=1)
    k = x1 / float(a)
    kr = np.maximum(k * r, 1e-12)
    j1_val = spherical_jn(1, kr)
    dj1_val = spherical_jn(1, kr, derivative=True)
    cos_t = np.where(r > 1e-12, z / np.where(r > 1e-12, r, 1.0), 1.0)
    sin_t = np.where(r > 1e-12, np.hypot(x, y) / np.where(r > 1e-12, r, 1.0), 0.0)
    er = 2.0 * (j1_val / kr) * cos_t
    etheta = -(j1_val / kr + dj1_val) * sin_t
    rho = np.hypot(x, y)
    cphi = np.where(rho > 1e-12, x / np.where(rho > 1e-12, rho, 1.0), 1.0)
    sphi = np.where(rho > 1e-12, y / np.where(rho > 1e-12, rho, 1.0), 0.0)
    ex = (er * sin_t + etheta * cos_t) * cphi
    ey = (er * sin_t + etheta * cos_t) * sphi
    ez = er * cos_t - etheta * sin_t
    return np.stack([ex, ey, ez], axis=-1)


def _discrete_d_analytic(block, e_func):
    """Build ``d_ana,e = S_e . E_ana(edge_midpoint)`` restricted to unknown edges."""
    u = block.unknown_edges
    e_vecs = e_func(block.edge_midpoints[u])
    return np.einsum("ei,ei->e", block.S_e[u], e_vecs)


# ---------------------------------------------------------------- Diagnostic 1
def run_diagnostic_1(refinements, a=1.0, length=1.2):
    """Run Diagnostic 1 (mode identification by W_eps overlap) on cylinder and sphere."""
    out = {"cylinder": {}, "sphere": {}}
    cached_modes = {"cylinder": {}, "sphere": {}}

    for geom in ("cylinder", "sphere"):
        w_ana = (
            cylinder_te111_omega(a=a, length=length)
            if geom == "cylinder"
            else sphere_lowest_omegas(a=a, l=1)[0]
        )
        e_func = (
            (lambda pts: _eval_cylinder_te111_e(pts, a=a, length=length))
            if geom == "cylinder"
            else (lambda pts: _eval_sphere_te_l1_e(pts, a=a))
        )

        for n in refinements:
            n = int(n)
            X = (
                cylinder_nodes(n, n, n, a=a, length=length)
                if geom == "cylinder"
                else sphere_nodes(n, n, n, a=a)
            )
            block = HexBlock3D(X)
            solver = CurvilinearFIT3D(block)
            W_e = solver.W_eps_u
            CW = (solver.C_u @ W_e).tocsr()
            A = (CW.T @ solver.W_nu @ CW).tocsr()
            A = (0.5 * (A + A.T)).tocsr()

            d_ana = _discrete_d_analytic(block, e_func)
            d_norm = float(np.sqrt(max(float(d_ana @ (W_e @ d_ana)), 1e-300)))

            # Shift-invert k = 12 eigenpairs of K = C_u^T W_nu C_u W_eps_u near w_ana^2
            k_eig = min(12, solver.n_unknown_edges - 2)
            evals_si, evecs_si = eigsh(
                A,
                k=k_eig,
                M=W_e,
                sigma=w_ana * w_ana,
                which="LM",
                v0=d_ana,
            )
            idx_sort = np.argsort(evals_si.real)
            evals_si = evals_si.real[idx_sort]
            evecs_si = evecs_si[:, idx_sort]

            # For N in {6, 8}, also compute all eigenpairs via dense eigh
            n_int = max(0, (n - 1) ** 3)
            if n in (6, 8):
                all_evals = eigh(A.toarray(), W_e.toarray(), eigvals_only=True)
                lam_max = float(np.max(all_evals))
                n_near_zero_full = int(np.sum(all_evals < 1e-6 * lam_max))
            else:
                lam_max = float((2.0 / solver.max_stable_dt()) ** 2)
                n_near_zero_full = n_int

            n_near_zero_k12 = int(np.sum(evals_si < 1e-6 * lam_max))

            omegas_si = np.sqrt(np.maximum(evals_si, 0.0))
            v_norms = np.sqrt(
                np.maximum(np.einsum("ij,ij->j", evecs_si, W_e @ evecs_si), 1e-300)
            )
            evecs_si = evecs_si / v_norms[None, :]

            # Align degenerate multiplets (|omega_i - omega_j| <= 1e-8 * omega)
            # with d_ana so the target polarization is canonically separated
            # from its orthogonal degenerate polarization partner(s).
            used = np.zeros(k_eig, dtype=bool)
            for j in range(k_eig):
                if used[j] or omegas_si[j] <= 1e-3 * w_ana:
                    continue
                grp = np.flatnonzero(
                    np.abs(omegas_si - omegas_si[j]) <= 1e-8 * omegas_si[j]
                )
                used[grp] = True
                if grp.size > 1:
                    V_g = evecs_si[:, grp]
                    proj = V_g.T @ (W_e @ d_ana)
                    p_norm = float(np.linalg.norm(proj))
                    if p_norm > 1e-14 * d_norm:
                        u1 = proj / p_norm
                        M_rot = np.eye(grp.size)
                        M_rot[:, 0] = u1
                        Q_rot, _ = np.linalg.qr(M_rot)
                        if np.dot(Q_rot[:, 0], u1) < 0.0:
                            Q_rot[:, 0] *= -1.0
                        evecs_si[:, grp] = V_g @ Q_rot

            overlaps = np.abs(evecs_si.T @ (W_e @ d_ana)) / d_norm

            # Curl residual to flag gradient-null modes (C_u W_eps_u v == 0)
            cw_vecs = CW @ evecs_si
            curl_energies = np.einsum("ij,ij->j", cw_vecs, solver.W_nu @ cw_vecs)

            # Multiplicities within the k=12 window (relative tolerance 1e-5)
            mults = []
            for w_val in omegas_si:
                if w_val <= 1e-3 * w_ana:
                    mults.append(int(np.sum(omegas_si <= 1e-3 * w_ana)))
                else:
                    mults.append(int(np.sum(np.abs(omegas_si - w_val) <= 1e-5 * w_val)))

            sel_idx = int(np.argmax(overlaps))
            sel_omega = float(omegas_si[sel_idx])
            sel_overlap = float(overlaps[sel_idx])

            # Subspace overlap across the degenerate multiplet of sel_omega
            in_mult = np.abs(omegas_si - sel_omega) <= 1e-5 * max(sel_omega, 1.0)
            subspace_overlap = float(np.linalg.norm(overlaps[in_mult]))

            # Nearest-eigenvalue choice (breaking exact degeneracy ties by overlap)
            dist_w = np.abs(omegas_si - w_ana)
            min_dist = float(np.min(dist_w))
            near_candidates = np.flatnonzero(dist_w <= min_dist + 1e-8 * w_ana)
            near_idx = int(near_candidates[int(np.argmax(overlaps[near_candidates]))])
            near_omega = float(omegas_si[near_idx])
            near_overlap = float(overlaps[near_idx])

            eig_list = []
            for j in range(k_eig):
                is_nz = bool(evals_si[j] < 1e-6 * lam_max)
                is_grad = bool(curl_energies[j] < 1e-6 * lam_max)
                eig_list.append(
                    {
                        "index": int(j),
                        "omega": float(omegas_si[j]),
                        "overlap": float(overlaps[j]),
                        "multiplicity": int(mults[j]),
                        "is_near_zero": is_nz,
                        "is_gradient_null": is_grad,
                    }
                )

            out[geom][str(n)] = {
                "selected_index": sel_idx,
                "selected_omega": sel_omega,
                "selected_overlap": sel_overlap,
                "subspace_overlap": subspace_overlap,
                "analytic_omega": float(w_ana),
                "nearest_index": near_idx,
                "nearest_omega": near_omega,
                "nearest_overlap": near_overlap,
                "same_mode_as_nearest": bool(
                    abs(sel_omega - near_omega) <= 1e-5 * w_ana
                ),
                "n_near_zero": int(n_near_zero_full),
                "n_near_zero_in_k12": int(n_near_zero_k12),
                "n_interior_nodes": int(n_int),
                "n_spurious_non_gradient": int(max(0, n_near_zero_full - n_int)),
                "eig_list_near_target": eig_list,
            }
            cached_modes[geom][n] = (block, solver, evecs_si[:, sel_idx])

    return out, cached_modes


# ---------------------------------------------------------------- Diagnostic 2
def _annular_te_residual(kc, m=2, r_in=0.5, r_out=1.0):
    """Bessel derivative cross-product ``J'_m(kc r_out) Y'_m(kc r_in) - J'_m(kc r_in) Y'_m(kc r_out)``."""
    return float(
        jvp(m, kc * r_out) * yvp(m, kc * r_in) - jvp(m, kc * r_in) * yvp(m, kc * r_out)
    )


def _annular_te_first_root(m=2, r_in=0.5, r_out=1.0):
    """Find the lowest positive root ``kc`` [1/m] of `_annular_te_residual` via `brentq`."""
    ks = np.linspace(0.5 / r_out, 12.0 / r_out, 400)
    vals = [_annular_te_residual(k, m=m, r_in=r_in, r_out=r_out) for k in ks]
    for idx in range(len(ks) - 1):
        if vals[idx] * vals[idx + 1] <= 0.0:
            kc = float(
                brentq(
                    _annular_te_residual,
                    float(ks[idx]),
                    float(ks[idx + 1]),
                    args=(m, r_in, r_out),
                    xtol=1e-15,
                    rtol=1e-15,
                )
            )
            return kc, abs(_annular_te_residual(kc, m=m, r_in=r_in, r_out=r_out))
    raise RuntimeError("failed to bracket annular TE Bessel root")


def _annular_sector_nodes(
    nx, ny, nz, r_in=0.5, r_out=1.0, phi_max=0.5 * np.pi, length=1.2
):
    """Smooth cylindrical polar mapping for a sector ``[r_in, r_out] x [0, phi_max] x [0, length]``."""
    r = np.linspace(float(r_in), float(r_out), int(nx) + 1)
    phi = np.linspace(0.0, float(phi_max), int(ny) + 1)
    z = np.linspace(0.0, float(length), int(nz) + 1)
    R, PHI, Z = np.meshgrid(r, phi, z, indexing="ij")
    return np.stack([R * np.cos(PHI), R * np.sin(PHI), Z], axis=-1)


def run_diagnostic_2(refinements, r_in=0.5, r_out=1.0, length=1.2, m=2, p=1):
    """Run Diagnostic 2 on the smooth polar annular cylinder sector."""
    kc_root, ref_res = _annular_te_first_root(m=m, r_in=r_in, r_out=r_out)
    kz = float(p) * np.pi / float(length)
    omega_ref = float(C0 * np.hypot(kc_root, kz))

    a_coef = float(yvp(m, kc_root * r_in))
    b_coef = float(-jvp(m, kc_root * r_in))

    def eval_annular_e(pts):
        rp = np.hypot(pts[:, 0], pts[:, 1])
        phip = np.arctan2(pts[:, 1], pts[:, 0])
        zp = pts[:, 2]
        zm_val = a_coef * jv(m, kc_root * rp) + b_coef * yv(m, kc_root * rp)
        dzm_val = a_coef * jvp(m, kc_root * rp) + b_coef * yvp(m, kc_root * rp)
        er = (m / (kc_root * rp)) * zm_val * np.sin(m * phip) * np.sin(kz * zp)
        ephi = dzm_val * np.cos(m * phip) * np.sin(kz * zp)
        ex = er * np.cos(phip) - ephi * np.sin(phip)
        ey = er * np.sin(phip) + ephi * np.cos(phip)
        ez = np.zeros_like(er)
        return np.stack([ex, ey, ez], axis=-1)

    out = {}
    ns_list = []
    errs_list = []
    err_by_n = {}

    for n in refinements:
        n = int(n)
        X = _annular_sector_nodes(n, n, n, r_in=r_in, r_out=r_out, length=length)
        block = HexBlock3D(X)
        solver = CurvilinearFIT3D(block)
        min_det = min_normalised_corner_det(block)

        d_ana = _discrete_d_analytic(block, eval_annular_e)
        W_e = solver.W_eps_u
        d_norm = float(np.sqrt(max(float(d_ana @ (W_e @ d_ana)), 1e-300)))

        CW = (solver.C_u @ W_e).tocsr()
        A = (CW.T @ solver.W_nu @ CW).tocsr()
        A = (0.5 * (A + A.T)).tocsr()
        G_u = block.gradient_unknown()
        H = (
            (A + float(np.max(A.diagonal())) * (G_u @ G_u.T)).tocsr()
            if G_u.shape[1] > 0
            else A
        )

        k_eig = min(12, solver.n_unknown_edges - 2)
        evals, evecs = eigsh(
            H,
            k=k_eig,
            M=W_e,
            sigma=omega_ref * omega_ref,
            which="LM",
            v0=d_ana,
        )
        v_norms = np.sqrt(np.maximum(np.einsum("ij,ij->j", evecs, W_e @ evecs), 1e-300))
        overlaps = np.abs(evecs.T @ (W_e @ d_ana)) / (v_norms * d_norm)
        best = int(np.argmax(overlaps))
        omega_disc = float(np.sqrt(max(evals[best].real, 0.0)))
        rel_err = float(abs(omega_disc - omega_ref) / omega_ref)

        out[str(n)] = {
            "omega_disc": omega_disc,
            "omega_ref": omega_ref,
            "rel_err": rel_err,
            "min_corner_det": float(min_det),
            "selected_overlap": float(overlaps[best]),
        }
        ns_list.append(n)
        errs_list.append(rel_err)
        err_by_n[n] = rel_err

    out["order_4pt"] = _fit_loglog_order(ns_list, errs_list)
    if 8 in err_by_n and 16 in err_by_n:
        out["order_8_16"] = float(np.log2(err_by_n[8] / err_by_n[16]))
    elif len(ns_list) >= 2:
        n_a, n_b = ns_list[-2], ns_list[-1]
        out["order_8_16"] = float(
            np.log(err_by_n[n_a] / err_by_n[n_b]) / np.log(float(n_b) / float(n_a))
        )
    else:
        out["order_8_16"] = None
    out["ref_residual"] = float(ref_res)
    out["kc_root"] = float(kc_root)
    return out


# ---------------------------------------------------------------- Diagnostic 3
def _corner_condition_numbers(block):
    """Compute max 3x3 condition numbers ``cond(M_c)`` and ``cond(S_c)`` and worst
    corner locations across all cells and all 8 cell corners of ``block``."""
    ijk = block.cell_grid()
    A_f = block.A_f
    S_e = block.S_e
    X = block.X
    Nx, Ny, Nz = block.nc

    max_cond_m = -1.0
    max_cond_s = -1.0
    min_norm_det = np.inf
    worst_m = None
    worst_s = None
    worst_det = None

    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                f0 = block.face_index(0, ijk[0] + a, ijk[1], ijk[2]).ravel()
                f1 = block.face_index(1, ijk[0], ijk[1] + b, ijk[2]).ravel()
                f2 = block.face_index(2, ijk[0], ijk[1], ijk[2] + c).ravel()
                M_c = A_f[np.stack([f0, f1, f2], axis=1)]
                cond_m = np.linalg.cond(M_c)
                idx_m = int(np.argmax(cond_m))
                if float(cond_m[idx_m]) > max_cond_m:
                    max_cond_m = float(cond_m[idx_m])
                    ci, cj, ck = np.unravel_index(idx_m, block.nc)
                    worst_m = {
                        "cell": [int(ci), int(cj), int(ck)],
                        "corner": [int(a), int(b), int(c)],
                    }

                e0 = block.edge_index(0, ijk[0], ijk[1] + b, ijk[2] + c).ravel()
                e1 = block.edge_index(1, ijk[0] + a, ijk[1], ijk[2] + c).ravel()
                e2 = block.edge_index(2, ijk[0] + a, ijk[1] + b, ijk[2]).ravel()
                S_c = S_e[np.stack([e0, e1, e2], axis=1)]
                cond_s = np.linalg.cond(S_c)
                idx_s = int(np.argmax(cond_s))
                if float(cond_s[idx_s]) > max_cond_s:
                    max_cond_s = float(cond_s[idx_s])
                    ci, cj, ck = np.unravel_index(idx_s, block.nc)
                    worst_s = {
                        "cell": [int(ci), int(cj), int(ck)],
                        "corner": [int(a), int(b), int(c)],
                    }

                dxi = X[1:, b : Ny + b, c : Nz + c] - X[:-1, b : Ny + b, c : Nz + c]
                deta = X[a : Nx + a, 1:, c : Nz + c] - X[a : Nx + a, :-1, c : Nz + c]
                dzeta = X[a : Nx + a, b : Ny + b, 1:] - X[a : Nx + a, b : Ny + b, :-1]
                det = np.abs(block.corner_jacobians[:, :, :, a, b, c])
                denom = (
                    np.linalg.norm(dxi, axis=-1)
                    * np.linalg.norm(deta, axis=-1)
                    * np.linalg.norm(dzeta, axis=-1)
                )
                ndet = (det / denom).ravel()
                idx_d = int(np.argmin(ndet))
                if float(ndet[idx_d]) < min_norm_det:
                    min_norm_det = float(ndet[idx_d])
                    ci, cj, ck = np.unravel_index(idx_d, block.nc)
                    worst_det = {
                        "cell": [int(ci), int(cj), int(ck)],
                        "corner": [int(a), int(b), int(c)],
                        "min_normalised_det": min_norm_det,
                    }

    worst_corner = {
        "cell": worst_m["cell"],
        "corner": worst_m["corner"],
        "M_c": worst_m,
        "S_c": worst_s,
        "min_det": worst_det,
    }
    return max_cond_m, max_cond_s, worst_corner


def _singular_corner_energy_ratio(block, solver, v, geometry):
    """Compute ``E_loc / E_total`` for eigenvector ``v`` restricted to unknown
    edges of cells within 2 cells (Chebyshev distance) of the singular points."""
    ijk = block.cell_grid()
    Nx, Ny, Nz = block.nc
    di = np.minimum(ijk[0], Nx - 1 - ijk[0])
    dj = np.minimum(ijk[1], Ny - 1 - ijk[1])
    dk = np.minimum(ijk[2], Nz - 1 - ijk[2])

    if geometry == "cylinder":
        cell_mask = np.maximum(di, dj) < 2
    else:
        cell_mask = np.maximum(np.maximum(di, dj), dk) < 2

    ci, cj, ck = ijk[0][cell_mask], ijk[1][cell_mask], ijk[2][cell_mask]
    edge_indices = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                edge_indices.append(block.edge_index(0, ci, cj + b, ck + c).ravel())
                edge_indices.append(block.edge_index(1, ci + a, cj, ck + c).ravel())
                edge_indices.append(block.edge_index(2, ci + a, cj + b, ck).ravel())
    loc_edges = np.unique(np.concatenate(edge_indices))
    u_mask = np.isin(block.unknown_edges, loc_edges)

    W_e = solver.W_eps_u
    e_total = float(v @ (W_e @ v))
    v_loc = np.where(u_mask, v, 0.0)
    e_loc = float(v_loc @ (W_e @ v_loc))
    return float(e_loc / max(e_total, 1e-300))


def _pec_wall_dual_face_angles(block):
    """Compute ``angle = asin(|S_e . n_w| / |S_e|)`` [deg] for every tangential
    PEC boundary edge ``e`` (where ``d_e = 0``), with ``n_w`` the unit wall
    normal from the block boundary face(s) containing ``e``."""
    S_e = block.S_e
    A_f = block.A_f
    mids = block.edge_midpoints
    angles_deg = []
    worst_angle = -1.0
    worst_midpoint = None

    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        ijk = block.edge_grid(a)
        e_arr = block.edge_index(a, *ijk).ravel()
        ia, ib, ic = ijk[a].ravel(), ijk[b].ravel(), ijk[c].ravel()

        for pos in range(e_arr.size):
            e = int(e_arr[pos])
            if not block.boundary_edge_mask[e]:
                continue
            se = S_e[e]
            se_norm = float(np.linalg.norm(se))
            if se_norm <= 0.0:
                continue

            n_sum = np.zeros(3, dtype=float)
            if ib[pos] == 0 or ib[pos] == block.nc[b]:
                sgn = 1.0 if ib[pos] == block.nc[b] else -1.0
                f_vec = np.zeros(3, dtype=float)
                for c_cell in (ic[pos] - 1, ic[pos]):
                    if 0 <= c_cell < block.nc[c]:
                        q = [0, 0, 0]
                        q[a], q[b], q[c] = int(ia[pos]), int(ib[pos]), int(c_cell)
                        f_vec += sgn * A_f[block.face_index(b, *q)]
                fn_norm = float(np.linalg.norm(f_vec))
                if fn_norm > 0.0:
                    n_sum += f_vec / fn_norm

            if ic[pos] == 0 or ic[pos] == block.nc[c]:
                sgn = 1.0 if ic[pos] == block.nc[c] else -1.0
                f_vec = np.zeros(3, dtype=float)
                for b_cell in (ib[pos] - 1, ib[pos]):
                    if 0 <= b_cell < block.nc[b]:
                        q = [0, 0, 0]
                        q[a], q[b], q[c] = int(ia[pos]), int(b_cell), int(ic[pos])
                        f_vec += sgn * A_f[block.face_index(c, *q)]
                fn_norm = float(np.linalg.norm(f_vec))
                if fn_norm > 0.0:
                    n_sum += f_vec / fn_norm

            nw = n_sum / float(np.linalg.norm(n_sum))
            sin_val = float(np.clip(abs(float(np.dot(se, nw))) / se_norm, 0.0, 1.0))
            ang_deg = float(np.degrees(np.arcsin(sin_val)))
            angles_deg.append(ang_deg)
            if ang_deg > worst_angle:
                worst_angle = ang_deg
                worst_midpoint = [float(x) for x in mids[e]]

    angles_arr = np.asarray(angles_deg, dtype=float)
    return (
        float(np.max(angles_arr)),
        float(np.median(angles_arr)),
        worst_midpoint,
    )


def run_diagnostic_3(refinements, cached_modes):
    """Run Diagnostic 3 (singular corner conditioning, energy localisation, PEC wall angle)."""
    out = {"cylinder": {}, "sphere": {}, "exponents": {}}
    ns_arr = np.asarray([int(n) for n in refinements], dtype=float)

    for geom in ("cylinder", "sphere"):
        cond_m_list = []
        cond_s_list = []
        for n in refinements:
            n = int(n)
            block, solver, v_mode = cached_modes[geom][n]
            max_cm, max_cs, worst_corner = _corner_condition_numbers(block)
            e_ratio = _singular_corner_energy_ratio(block, solver, v_mode, geom)
            max_ang, med_ang, worst_mid = _pec_wall_dual_face_angles(block)

            cond_m_list.append(max_cm)
            cond_s_list.append(max_cs)
            out[geom][str(n)] = {
                "max_cond_M": float(max_cm),
                "max_cond_S": float(max_cs),
                "worst_corner": worst_corner,
                "E_loc_over_E_total": float(e_ratio),
                "max_wall_angle_deg": float(max_ang),
                "median_wall_angle_deg": float(med_ang),
                "worst_edge_midpoint": worst_mid,
            }

        exp_m = float(np.polyfit(np.log(ns_arr), np.log(cond_m_list), 1)[0])
        exp_s = float(np.polyfit(np.log(ns_arr), np.log(cond_s_list), 1)[0])
        out["exponents"][geom] = {"cond_M": exp_m, "cond_S": exp_s}
        out["exponents"][f"{geom}_cond_M"] = exp_m
        out["exponents"][f"{geom}_cond_S"] = exp_s

    return out


# ---------------------------------------------------------------- Diagnostic 4
def run_diagnostic_4(refinements, a=1.0, b=0.7, d=0.8, amp=0.3):
    """Run Diagnostic 4 reconciling the test two-level order and study four-point order."""
    omega_ana = box_te101_omega(a=a, d=d)

    # 1. Test mesh family (4k, 3k, 4k)
    test_grids = [(4, 3, 4), (8, 6, 8)]
    if max(int(n) for n in refinements) >= 12:
        test_grids.append((12, 9, 12))
    if max(int(n) for n in refinements) >= 16:
        test_grids.append((16, 12, 16))

    test_levels = {}
    for nx, ny, nz in test_grids:
        X = perturbed_box_nodes(nx, ny, nz, a=a, b=b, d=d, amp=amp)
        w_disc = float(CurvilinearFIT3D(HexBlock3D(X)).cavity_frequencies(k=2)[0])
        rel_err = float(abs(w_disc - omega_ana) / omega_ana)
        test_levels[f"{nx}x{ny}x{nz}"] = {
            "nx_ny_nz": [int(nx), int(ny), int(nz)],
            "omega_disc": w_disc,
            "omega_ana": float(omega_ana),
            "rel_err": rel_err,
        }

    # 2. Study mesh family (N, N, N)
    study_ns = sorted({4, *[int(n) for n in refinements]})
    study_levels = {}
    for n in study_ns:
        X = perturbed_box_nodes(n, n, n, a=a, b=b, d=d, amp=amp)
        w_disc = float(CurvilinearFIT3D(HexBlock3D(X)).cavity_frequencies(k=2)[0])
        rel_err = float(abs(w_disc - omega_ana) / omega_ana)
        study_levels[str(n)] = {
            "nx_ny_nz": [int(n), int(n), int(n)],
            "omega_disc": w_disc,
            "omega_ana": float(omega_ana),
            "rel_err": rel_err,
        }

    # Definition A: two-level order between (4, 3, 4) and (8, 6, 8)
    order_two_level_test = float(
        np.log2(test_levels["4x3x4"]["rel_err"] / test_levels["8x6x8"]["rel_err"])
    )
    order_4pt_test = _fit_loglog_order(
        [g[0] for g in test_grids],
        [test_levels[f"{g[0]}x{g[1]}x{g[2]}"]["rel_err"] for g in test_grids],
    )

    # Definition B: four-point log-log fit over study refinements on (N, N, N)
    ref_list = [int(n) for n in refinements]
    order_four_point_study = _fit_loglog_order(
        ref_list,
        [study_levels[str(n)]["rel_err"] for n in ref_list],
    )
    order_two_level_study_4_8 = float(
        np.log2(study_levels["4"]["rel_err"] / study_levels["8"]["rel_err"])
    )
    order_two_level_study_8_16 = (
        float(np.log2(study_levels["8"]["rel_err"] / study_levels["16"]["rel_err"]))
        if "16" in study_levels
        else None
    )

    note = (
        "In studies/curvilinear3d/study_cavities.py (curvilinear_eigenfrequencies), "
        "scalar N maps to isotropic cell counts (nx, ny, nz) = (N, N, N) via "
        "perturbed_box_nodes(N, N, N, a=1.0, b=0.7, d=0.8, amp=0.3), whereas "
        "test_perturbed_box_order_small uses anisotropic cell counts (4, 3, 4) and "
        f"(8, 6, 8). Definition A (two-level log2(err_4x3x4 / err_8x6x8) on the test "
        f"mesh) yields {order_two_level_test:.6f} (matching {order_two_level_test:.4f} "
        f"in the test log); Definition B (log-log fit over N={ref_list} on the "
        f"(N, N, N) study mesh) yields {order_four_point_study:.6f} (matching "
        f"{order_four_point_study:.4f} in study_cavities.py). On the study mesh "
        f"(N, N, N), two-level (4,4,4)->(8,8,8) gives {order_two_level_study_4_8:.6f}."
    )

    return {
        "test_levels": test_levels,
        "study_levels": study_levels,
        "order_two_level": order_two_level_test,
        "order_four_point": order_four_point_study,
        "test_family_orders": {
            "two_level_4x3x4_to_8x6x8": order_two_level_test,
            "loglog_fit": order_4pt_test,
        },
        "study_family_orders": {
            "two_level_4_to_8": order_two_level_study_4_8,
            "two_level_8_to_16": order_two_level_study_8_16,
            "loglog_fit_refinements": order_four_point_study,
        },
        "mesh_mapping_note": note,
    }


def format_stderr_table(data):
    """Format a concise human-readable diagnostic summary table for stderr."""
    lines = ["=== W3-5 Diagnostic 1: Mode identification by W_eps overlap ==="]
    lines.append(
        f"{'Geom':<10} {'N':>3} {'omega_ana':>14} {'sel_omega':>14} "
        f"{'sel_ov':>8} {'sub_ov':>8} {'near_omega':>14} {'near_ov':>8} "
        f"{'n_nz':>5} {'n_k12_nz':>8}"
    )
    for geom in ("cylinder", "sphere"):
        for n_str, rec in data["diag1"][geom].items():
            lines.append(
                f"{geom:<10} {int(n_str):>3d} {rec['analytic_omega']:>14.6e} "
                f"{rec['selected_omega']:>14.6e} {rec['selected_overlap']:>8.5f} "
                f"{rec['subspace_overlap']:>8.5f} {rec['nearest_omega']:>14.6e} "
                f"{rec['nearest_overlap']:>8.5f} {rec['n_near_zero']:>5d} "
                f"{rec['n_near_zero_in_k12']:>8d}"
            )
    lines.append(f"Interpretation: {INTERPRETATION_RULES['diag1']}")

    lines.append("\n=== W3-5 Diagnostic 2: Smooth polar quarter-annulus TE_(2,1,1) ===")
    lines.append(
        f"{'N':>3} {'omega_ref':>14} {'omega_disc':>14} {'rel_err':>12} {'min_det':>10}"
    )
    for k, rec in data["diag2"].items():
        if isinstance(rec, dict) and "omega_disc" in rec:
            lines.append(
                f"{int(k):>3d} {rec['omega_ref']:>14.6e} {rec['omega_disc']:>14.6e} "
                f"{rec['rel_err']:>12.4e} {rec['min_corner_det']:>10.6f}"
            )
    lines.append(
        f"order_4pt={data['diag2']['order_4pt']:.4f}, "
        f"order_8_16={data['diag2']['order_8_16']:.4f}, "
        f"ref_residual={data['diag2']['ref_residual']:.3e}"
    )
    lines.append(f"Interpretation: {INTERPRETATION_RULES['diag2']}")

    lines.append(
        "\n=== W3-5 Diagnostic 3: Singular corners & PEC wall dual-face angle ==="
    )
    lines.append(
        f"{'Geom':<10} {'N':>3} {'max_cond_M':>10} {'max_cond_S':>10} "
        f"{'E_loc/E_tot':>11} {'max_ang_deg':>11} {'med_ang_deg':>11} {'worst_corner_M':<24}"
    )
    for geom in ("cylinder", "sphere"):
        for n_str, rec in data["diag3"][geom].items():
            wc = f"c={rec['worst_corner']['cell']},a={rec['worst_corner']['corner']}"
            lines.append(
                f"{geom:<10} {int(n_str):>3d} {rec['max_cond_M']:>10.4f} "
                f"{rec['max_cond_S']:>10.4f} {rec['E_loc_over_E_total']:>11.5f} "
                f"{rec['max_wall_angle_deg']:>11.4f} {rec['median_wall_angle_deg']:>11.4f} "
                f"{wc:<24}"
            )
    exp = data["diag3"]["exponents"]
    lines.append(
        f"Scaling exponents: cylinder cond_M~N^{exp['cylinder']['cond_M']:.3f}, "
        f"cond_S~N^{exp['cylinder']['cond_S']:.3f}; "
        f"sphere cond_M~N^{exp['sphere']['cond_M']:.3f}, "
        f"cond_S~N^{exp['sphere']['cond_S']:.3f}"
    )
    lines.append(f"Interpretation: {INTERPRETATION_RULES['diag3']}")

    lines.append(
        "\n=== W3-5 Diagnostic 4: Perturbed box TE101 order reconciliation ==="
    )
    lines.append(f"{'Family':<8} {'Grid':<10} {'omega_disc':>14} {'rel_err':>12}")
    for label, rec in data["diag4"]["test_levels"].items():
        lines.append(
            f"{'test':<8} {label:<10} {rec['omega_disc']:>14.6e} {rec['rel_err']:>12.6e}"
        )
    for label, rec in data["diag4"]["study_levels"].items():
        g_str = f"{label}x{label}x{label}"
        lines.append(
            f"{'study':<8} {g_str:<10} {rec['omega_disc']:>14.6e} {rec['rel_err']:>12.6e}"
        )
    lines.append(
        f"Definition A (two-level 4x3x4->8x6x8): {data['diag4']['order_two_level']:.6f} | "
        f"Definition B (study 4-pt): {data['diag4']['order_four_point']:.6f}"
    )
    lines.append(f"Note: {data['diag4']['mesh_mapping_note']}")
    lines.append(f"Interpretation: {INTERPRETATION_RULES['diag4']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="W3-5 diagnosis of cylinder TE111 and sphere TE_l1 non-convergence."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test mode running N in {6, 8} only.",
    )
    parser.add_argument(
        "--refinements",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of refinements N (default: 6 8 12 16, or 6 8 with --quick).",
    )
    args = parser.parse_args(argv)

    if args.refinements is not None:
        refinements = [int(n) for n in args.refinements]
    elif args.quick:
        refinements = [6, 8]
    else:
        refinements = [6, 8, 12, 16]

    diag1, cached_modes = run_diagnostic_1(refinements)
    diag2 = run_diagnostic_2(refinements)
    diag3 = run_diagnostic_3(refinements, cached_modes)
    diag4 = run_diagnostic_4(refinements)

    result = {
        "diag1": diag1,
        "diag2": diag2,
        "diag3": diag3,
        "diag4": diag4,
        "interpretations": dict(INTERPRETATION_RULES),
    }

    print(format_stderr_table(result), file=sys.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
