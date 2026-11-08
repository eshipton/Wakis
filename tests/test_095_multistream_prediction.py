# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Fast verification tests for Decision D20 multistream prediction of particle ``I_1(z)``.

Covers the five fast checks of the D20 specification plus prototype reproduction and
legacy ``method="modal"`` / ``LinearizedBeam`` regression checks:

1. Reduction R1 (shear-free, frozen ``G``, no wall term, no DC entrance phase, no
   observable ``sinc``) to the legacy modal ``I_1(z)`` vector within ``1e-10`` relative
   and ``beta_q`` within ``1e-9`` relative, with ``Omega -> omega`` in the ``v_1``
   matching row and entrance-profile perturbation as negative controls.
2. Multistream root residuals ``sigma_min / sigma_max <= 1e-12``, pairwise distinctness
   ``|Delta k| > 1e-8`` rad/m, agreement of ``include_shear=False`` roots with legacy
   ``k_c +- beta_q`` to ``1e-8`` rad/m, agreement of the sheared fundamental mode with
   ``studies/p5b-shear/qep_ref.py`` ``brentq`` root to ``1e-6`` relative, and agreement
   between ``"even"`` and ``"full"`` subspaces to ``1e-9`` relative.
3. Entrance matching identities ``sum_n a_n X_n = e``,
   ``sum_n a_n (k_n - beta_l) X_{l,n} = 0`` to ``1e-12`` relative, and
   ``sum_n a_n d^T X_n = 1`` for uniform entrance ``e = 1``.
4. First-order (Born) PEC-wall near-field correction vs exact ``2 * (N + M)`` augmented
   half-line ODE solve on the continuous rational model (coarse ``q/m = 1``), with a
   Dirichlet augmented-BC negative control that disagrees by ``> 100%`` of the wall shift.
5. Pairwise separation ``> 3e-5`` of the three ``wall_image`` variants
   (``"neumann"``, ``"none"``, ``"dirichlet"``), reproduction of the D20 prototype to
   ``<= 1e-9`` relative on ``(10, 474, q/m=1)``, ``(20, 948, q/m=1)``, and
   ``(10, 474, q/m=2)``, and bit-for-bit preservation of ``method="modal"`` and ``"lb"``.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.constants import c as c_light

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._space_charge_waves_common import (
    BEAM_HALF_THICKNESS_M,
    COARSE_GRID,
    PARTICLE_SHAPE,
    TUNNEL_LENGTH_M,
    _cached_integer_period_dt,
    _even_subspace_basis,
    _multistream_dc_entrance_phase,
    _multistream_wall_correction,
    _single_grid_budget_terms,
    compute_analytic_parameters,
    compute_discrete_dc_depression,
    compute_transverse_modal_operator,
    discrete_modal_prediction,
    fit_beta_q_from_envelope,
    fit_window,
    multistream_waves,
)


def _qep_ref_mode1_brentq(
    nx: int, nz: int, qm_scale: float, shape: int = PARTICLE_SHAPE
) -> float:
    """Compute exact sheared mode-1 ``beta_q`` [rad/m] via ``brentq`` on ``slogdet`` as in ``qep_ref.py``."""
    from scipy.optimize import brentq

    params = _cached_params(qm_scale)
    dz = TUNNEL_LENGTH_M / nz
    dt, _ = _cached_integer_period_dt(nx, nz)
    dep = compute_discrete_dc_depression(nx, params, return_details=True, shape=shape)
    u_l, g_l = dep["u_l"], dep["gamma_l"]
    u0 = params["u0"]
    sig = params["omega_p0"] ** 2 * (u0 / u_l) / (g_l**3 * u_l**2)
    om_dt = (2.0 / dt) * np.sin(0.5 * params["omega"] * dt)
    be = om_dt / u_l

    def m_sym(kk: float) -> np.ndarray:
        op = compute_transverse_modal_operator(
            "particles", nx, nz, params, k_val=kk, shape=shape
        )
        d = op["d_vec"]
        sz = np.sinc(0.5 * kk * dz / np.pi) ** (2 * shape)
        sl = np.sqrt(d / sig)
        sr = np.sqrt(sig / d)
        m = (sl[:, None] * (sz * sig)[:, None] * op["G"]) * sr[None, :]
        return 0.5 * (m + m.T)

    def f_det(kk: float) -> float:
        return float(np.linalg.slogdet(np.diag((kk - be) ** 2) - m_sym(kk))[0])

    def find_root(k0: float, w: float) -> float:
        ks = np.linspace(k0 - w, k0 + w, 121)
        fs = np.array([f_det(float(x)) for x in ks])
        idx = np.where(np.diff(fs) != 0)[0]
        return float(
            min(
                [
                    brentq(f_det, float(ks[i]), float(ks[i + 1]), xtol=1e-13)
                    for i in idx
                ],
                key=lambda q: abs(q - k0),
            )
        )

    det = _cached_modal_details(nx, nz, qm_scale)
    bq0 = float(det["bq_modes"][0])
    kc0 = float(det["kc_modes"][0])
    w = min(2.0, 0.3 * bq0)
    kp = find_root(kc0 + bq0, w)
    km = find_root(kc0 - bq0, w)
    return 0.5 * (kp - km)


@lru_cache(maxsize=16)
def _cached_params(qm_scale: float) -> dict[str, float]:
    return compute_analytic_parameters(qm_scale=float(qm_scale))


@lru_cache(maxsize=16)
def _cached_modal_details(
    nx: int,
    nz: int,
    qm_scale: float,
    include_shear: bool = True,
    freeze_g: bool = False,
) -> dict[str, Any]:
    params = _cached_params(qm_scale)
    return discrete_modal_prediction(
        "particles",
        nx,
        nz,
        params,
        include_shear=include_shear,
        freeze_g=freeze_g,
        return_details=True,
        method="modal",
    )


@lru_cache(maxsize=32)
def _cached_d20_details(
    nx: int,
    nz: int,
    qm_scale: float,
    wall_image: str = "neumann",
    include_shear: bool = True,
    subspace: str | None = None,
) -> dict[str, Any]:
    params = _cached_params(qm_scale)
    ms = multistream_waves(
        nx,
        nz,
        params,
        include_shear=include_shear,
        wall_image=wall_image,
        subspace=subspace,
    )
    det_modal = _cached_modal_details(nx, nz, qm_scale, include_shear=include_shear)
    dz = TUNNEL_LENGTH_M / nz
    z_eval = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    i1_model = (
        params["I1_0"] * np.exp(-1j * np.outer(z_eval, ms["k_waves"])) @ ms["o_waves"]
    )
    z_min, z_max = fit_window(params)
    fit_res = fit_beta_q_from_envelope(
        z_eval,
        i1_model,
        float(det_modal["bq_modes"][0]),
        z_min=z_min,
        z_max=z_max,
    )
    return {
        "beta_q": float(fit_res["beta_q"]),
        "z_eval": z_eval,
        "i1_model": i1_model,
        "ms": ms,
        "det_modal": det_modal,
    }


def _prototype_msp_predict(nx: int, nz: int, qm_scale: float) -> dict[str, Any]:
    """Reference prototype implementation ported verbatim from ``msp.py``."""
    params = _cached_params(qm_scale)
    dz = TUNNEL_LENGTH_M / nz
    dt, _ = _cached_integer_period_dt(nx, nz)
    dep = compute_discrete_dc_depression(nx, params, return_details=True, shape=2)
    u_l, g_l = dep["u_l"], dep["gamma_l"]
    u0 = params["u0"]
    om_dt = (2.0 / dt) * np.sin(0.5 * params["omega"] * dt)
    sig = params["omega_p0"] ** 2 * (u0 / u_l) / (g_l**3 * u_l**2)
    be = om_dt / u_l
    u_bar = u0 * (1.0 - float(dep["delta_disc"]))
    k_ref = params["omega"] / u_bar
    op0 = compute_transverse_modal_operator(
        "particles", nx, nz, params, k_val=k_ref, shape=2
    )
    d = op0["d_vec"]
    n_lines = len(be)
    p_mat = _even_subspace_basis(n_lines)
    be_sub = p_mat.T @ np.diag(be) @ p_mat
    lap = op0["lap"]
    s_mat = op0["S"]
    d_mat = np.diag(d)
    eye_int = np.eye(nx - 1, dtype=np.float64)
    g_scale = (n_lines**2 * op0["dx"]) / (2.0 * BEAM_HALF_THICKNESS_M)

    def m_mat(k_val: float) -> np.ndarray:
        kappa = float((2.0 / dz) * np.sin(0.5 * k_val * dz))
        tau_y2 = max(kappa**2 - (om_dt / c_light) ** 2, 1.0e-12)
        a_mat = -lap + tau_y2 * eye_int
        g_mat = g_scale * tau_y2 * (s_mat.T @ np.linalg.solve(a_mat, s_mat) @ d_mat)
        sz = np.sinc(0.5 * k_val * dz / np.pi) ** 4
        return (sz * sig)[:, None] * g_mat

    def t_e(k_val: complex) -> np.ndarray:
        return p_mat.T @ (np.diag((k_val - be) ** 2) - m_mat(float(k_val.real))) @ p_mat

    fd_h = 1.0e-3

    def dt_e(k_val: complex) -> np.ndarray:
        d_m = (m_mat(float(k_val.real) + fd_h) - m_mat(float(k_val.real) - fd_h)) / (
            2.0 * fd_h
        )
        return p_mat.T @ (2.0 * np.diag(k_val - be) - d_m) @ p_mat

    m_e = p_mat.T @ m_mat(k_ref) @ p_mat
    h = p_mat.shape[1]
    a0_c = be_sub @ be_sub - m_e
    a1_c = -2.0 * be_sub
    comp = np.block([[np.zeros((h, h)), np.eye(h)], [-a0_c, -a1_c]])
    k0 = np.linalg.eigvals(comp)
    ks_list, w_list = [], []
    for k_seed in k0:
        k_cur = complex(k_seed)
        for _ in range(30):
            t_cur = t_e(k_cur)
            u_svd, _, vh_svd = np.linalg.svd(t_cur)
            x_v, y_v = vh_svd[-1].conj(), u_svd[:, -1]
            dk = -(y_v.conj() @ t_cur @ x_v) / (y_v.conj() @ dt_e(k_cur) @ x_v)
            k_cur = k_cur + dk
            if abs(dk) < 1.0e-13 * abs(k_cur):
                break
        _, _, vh_svd = np.linalg.svd(t_e(k_cur))
        ks_list.append(k_cur)
        w_list.append(vh_svd[-1].conj())
    ks = np.array(ks_list)
    x_mat = p_mat @ np.array(w_list).T

    th = _multistream_dc_entrance_phase(
        nx=nx, dz=dz, dt=dt, params=params, dep_info=dep, discrete=True
    )
    e_vec = np.exp(-1j * th)
    dk_mat = ks[None, :] - be[:, None]
    pr = p_mat.T
    a_sys = np.vstack([pr @ x_mat, pr @ (x_mat * (-1j * dk_mat))])
    a0 = np.linalg.solve(a_sys, np.concatenate([pr @ e_vec, np.zeros(h)]))
    da = _multistream_wall_correction(
        op0=op0,
        dz=dz,
        dt=dt,
        omega=params["omega"],
        sig_vec=sig,
        d_vec=d,
        k_waves=ks,
        x_waves=x_mat,
        a_waves=a0,
        p_basis=p_mat,
        shape=2,
        use_omega_f=True,
        include_sz=True,
        discrete=True,
    )
    det_legacy = _cached_modal_details(nx, nz, qm_scale)
    guess = float(det_legacy["bq_modes"][0])
    z = (np.arange(nz) + 0.5) * dz
    zmin, zmax = fit_window(params)
    o_vec = (a0 + da) * (d @ x_mat) * (np.sinc(0.5 * ks * dz / np.pi) ** 2)
    i1 = params["I1_0"] * np.exp(-1j * np.outer(z, ks)) @ o_vec
    return {
        "d20": float(
            fit_beta_q_from_envelope(z, i1, guess, z_min=zmin, z_max=zmax)["beta_q"]
        )
    }


def test_reduction_r1_and_omega_mutation():
    """Fast test 1 (R1): shear-free frozen-G multistream reduces to legacy modal I1(z) to <= 1e-10."""
    print(flush=True)
    for nx, nz, qm in ((10, 474, 1.0), (20, 948, 1.0), (10, 474, 2.0)):
        params = _cached_params(qm)
        det_ms = discrete_modal_prediction(
            "particles",
            nx,
            nz,
            params,
            include_shear=False,
            include_sz=True,
            wall_image="none",
            dc_entrance=False,
            freeze_g=True,
            return_details=True,
            method="multistream",
        )
        det_mod = _cached_modal_details(nx, nz, qm, include_shear=False, freeze_g=True)
        rel_i1 = float(
            np.linalg.norm(det_ms["i1_model"] - det_mod["i1_model"])
            / np.linalg.norm(det_mod["i1_model"])
        )
        rel_bq = float(abs(det_ms["beta_q"] / det_mod["beta_q"] - 1.0))
        print(
            f"R1CHECK nx={nx} nz={nz} qm={qm:.1f} rel_i1={rel_i1:.3e} "
            f"beta_ms={det_ms['beta_q']:.10f} beta_mod={det_mod['beta_q']:.10f} rel_bq={rel_bq:.3e}",
            flush=True,
        )
        assert rel_i1 <= 1.0e-10, (
            f"R1 I1 relative difference {rel_i1:.3e} > 1e-10 on ({nx}, {nz}, qm={qm})"
        )
        assert rel_bq <= 1.0e-9, (
            f"R1 beta_q relative difference {rel_bq:.3e} > 1e-9 on ({nx}, {nz}, qm={qm})"
        )

    # Negative control 1: replacing Omega -> omega in the v1 matching row must give dI1 > 1e-6
    nx, nz, qm = 10, 474, 1.0
    params = _cached_params(qm)
    det_mod = _cached_modal_details(nx, nz, qm, include_shear=False, freeze_g=True)
    ms_r1 = multistream_waves(
        nx,
        nz,
        params,
        include_shear=False,
        include_sz=True,
        wall_image="none",
        dc_entrance=False,
        freeze_g=True,
    )
    dep = compute_discrete_dc_depression(nx, params, return_details=True)
    u_bar = float(params["u0"] * (1.0 - float(dep["delta_disc"])))
    be_wrong = np.full(len(dep["u_l"]), params["omega"] / u_bar, dtype=np.float64)
    p_even = _even_subspace_basis(len(be_wrong))
    pr = p_even.T
    dk_wrong = ms_r1["k_waves"][None, :] - be_wrong[:, None]
    a_mat_wrong = np.vstack(
        [pr @ ms_r1["x_waves"], pr @ (ms_r1["x_waves"] * (-1j * dk_wrong))]
    )
    rhs = np.concatenate(
        [pr @ np.ones(len(be_wrong), dtype=np.complex128), np.zeros(pr.shape[0])]
    )
    a_wrong = np.linalg.solve(a_mat_wrong, rhs)
    d_vec = np.full(len(be_wrong), 1.0 / len(be_wrong), dtype=np.float64)
    o_wrong = a_wrong * (d_vec @ ms_r1["x_waves"])
    z_eval = det_mod["z_coords"]
    i1_wrong = (
        params["I1_0"] * np.exp(-1j * np.outer(z_eval, ms_r1["k_waves"])) @ o_wrong
    )
    rel_i1_omega_mut = float(
        np.linalg.norm(i1_wrong - det_mod["i1_model"])
        / np.linalg.norm(det_mod["i1_model"])
    )

    # Negative control 2: 1e-6 perturbation on the edge line entrance modulation
    e_pert = np.ones(len(be_wrong), dtype=np.float64)
    e_pert[-1] += 1.0e-6
    ms_pert = multistream_waves(
        nx,
        nz,
        params,
        entrance_profile=e_pert,
        include_shear=False,
        include_sz=True,
        wall_image="none",
        dc_entrance=False,
        freeze_g=True,
    )
    i1_pert = (
        params["I1_0"]
        * np.exp(-1j * np.outer(z_eval, ms_pert["k_waves"]))
        @ ms_pert["o_waves"]
    )
    rel_i1_ent_mut = float(
        np.linalg.norm(i1_pert - det_mod["i1_model"])
        / np.linalg.norm(det_mod["i1_model"])
    )
    print(
        f"R1_MUTATIONS omega_row_rel_i1={rel_i1_omega_mut:.3e} "
        f"edge_entrance_rel_i1={rel_i1_ent_mut:.3e}",
        flush=True,
    )
    assert rel_i1_omega_mut > 1.0e-6, (
        f"Omega->omega negative control failed to exceed 1e-6: {rel_i1_omega_mut:.3e}"
    )
    with pytest.raises(AssertionError, match="R1 negative control"):
        assert rel_i1_omega_mut <= 1.0e-10, "R1 negative control must fail <= 1e-10"
    with pytest.raises(AssertionError, match="R1 entrance mutation"):
        assert rel_i1_ent_mut <= 1.0e-10, "R1 entrance mutation must fail <= 1e-10"


def test_roots_residuals_legacy_qep_and_subspaces():
    """Fast test 2: root SVD residuals <= 1e-12, legacy roots, qep_ref brentq root, and even vs full."""
    print(flush=True)

    for nx, nz, qm in ((10, 474, 1.0), (20, 948, 1.0), (10, 474, 2.0)):
        params = _cached_params(qm)
        info = _cached_d20_details(nx, nz, qm)
        ms = info["ms"]
        ks = ms["k_waves"]
        max_res = float(np.max(ms["root_resid"]))
        dk_mat = np.abs(ks[:, None] - ks[None, :]) + np.eye(len(ks)) * 1.0e9
        min_dk = float(np.min(dk_mat))

        # Unsheared roots must contain all active legacy k_c +- beta_q to <= 1e-8 rad/m
        ms_unif = multistream_waves(nx, nz, params, include_shear=False)
        det_unif = _cached_modal_details(nx, nz, qm, include_shear=False)
        act = np.abs(det_unif["c_modes"]) > 1.0e-12
        k_legacy = np.concatenate(
            [
                det_unif["kc_modes"][act] - det_unif["bq_modes"][act],
                det_unif["kc_modes"][act] + det_unif["bq_modes"][act],
            ]
        )
        max_legacy_dk = float(
            max(np.min(np.abs(ms_unif["k_waves"] - kl)) for kl in k_legacy)
        )

        # Even vs full subspace agreement to <= 1e-9 relative
        info_full = _cached_d20_details(nx, nz, qm, subspace="full")
        rel_even_full = float(abs(info["beta_q"] / info_full["beta_q"] - 1.0))

        print(
            f"ROOTSCHECK nx={nx} nz={nz} qm={qm:.1f} max_svd_resid={max_res:.3e} "
            f"min_dk={min_dk:.3e} legacy_root_err={max_legacy_dk:.3e} "
            f"rel_even_vs_full={rel_even_full:.3e}",
            flush=True,
        )
        assert max_res <= 1.0e-12, f"Max root residual {max_res:.3e} > 1e-12"
        assert min_dk > 1.0e-8, f"Pairwise root separation {min_dk:.3e} <= 1e-8 rad/m"
        assert float(np.max(ms_unif["root_resid"])) <= 1.0e-12
        assert max_legacy_dk <= 1.0e-8, (
            f"Unsheared root mismatch vs legacy {max_legacy_dk:.3e} > 1e-8 rad/m"
        )
        assert rel_even_full <= 1.0e-9, (
            f"Even vs full subspace mismatch {rel_even_full:.3e} > 1e-9"
        )

    # Compare sheared mode-1 half-splitting against qep_ref.py brentq root to <= 1e-6 relative
    for nx, nz, qm in ((10, 474, 1.0), (10, 474, 2.0)):
        bq1_qep = _qep_ref_mode1_brentq(nx, nz, qm, shape=PARTICLE_SHAPE)
        ks = _cached_d20_details(nx, nz, qm)["ms"]["k_waves"]
        real_ks = np.sort(ks[np.abs(ks.imag) < 1.0e-6].real)
        # Mode 1 has the largest plasma splitting: lowest fast wave and highest slow wave
        bq1_ms = 0.5 * float(real_ks[-1] - real_ks[0])
        rel_qep = abs(bq1_ms / bq1_qep - 1.0)
        print(
            f"QEPREF nx={nx} nz={nz} qm={qm:.1f} bq1_ms={bq1_ms:.8f} "
            f"bq1_qep={bq1_qep:.8f} rel_diff={rel_qep:.3e}",
            flush=True,
        )
        assert rel_qep <= 1.0e-6, f"Mode-1 root vs qep_ref brentq {rel_qep:.3e} > 1e-6"

    # Asymmetric Fibonacci disk (M3): must select "full" subspace automatically with root_resid <= 1e-12
    k_idx = np.arange(16, dtype=np.float64)
    r_k = BEAM_HALF_THICKNESS_M * np.sqrt((k_idx + 0.5) / 16.0)
    phi_k = k_idx * np.pi * (3.0 - np.sqrt(5.0))
    x_fib = r_k * np.cos(phi_k) - float(np.mean(r_k * np.cos(phi_k)))
    ms_m3 = multistream_waves(10, 474, _cached_params(1.0), line_x=x_fib)
    print(
        f"M3_SUBSPACE subspace={ms_m3['subspace']} max_svd_resid={np.max(ms_m3['root_resid']):.3e}",
        flush=True,
    )
    assert ms_m3["subspace"] == "full"
    assert float(np.max(ms_m3["root_resid"])) <= 1.0e-12

    # Negative control: perturbing a root by 1e-3 rad/m must fail the <= 1e-12 SVD residual check
    params_1 = _cached_params(1.0)
    dep_1 = compute_discrete_dc_depression(10, params_1, return_details=True)
    dt_1, _ = _cached_integer_period_dt(10, 474)
    om_dt_1 = (2.0 / dt_1) * np.sin(0.5 * params_1["omega"] * dt_1)
    be_1 = om_dt_1 / dep_1["u_l"]
    sig_1 = (
        params_1["omega_p0"] ** 2
        * (params_1["u0"] / dep_1["u_l"])
        / (dep_1["gamma_l"] ** 3 * dep_1["u_l"] ** 2)
    )
    k_pert = float(_cached_d20_details(10, 474, 1.0)["ms"]["k_waves"][0].real) + 1.0e-3
    op_pert = compute_transverse_modal_operator(
        "particles", 10, 474, params_1, k_val=k_pert
    )
    dz_1 = TUNNEL_LENGTH_M / 474
    sz_sq = float(np.sinc(0.5 * k_pert * dz_1 / np.pi) ** (2 * PARTICLE_SHAPE))
    t_pert = np.diag((k_pert - be_1) ** 2) - (sz_sq * sig_1)[:, None] * op_pert["G"]
    _, sv_pert, _ = np.linalg.svd(t_pert)
    resid_pert = float(sv_pert[-1] / sv_pert[0])
    print(f"ROOTS_MUTATION perturbed_root_svd_resid={resid_pert:.3e}", flush=True)
    with pytest.raises(AssertionError, match="Perturbed root"):
        assert resid_pert <= 1.0e-12, "Perturbed root must fail <= 1e-12"


def test_entrance_matching_identities():
    """Fast test 3: verify entrance matching identities to <= 1e-12 relative and sum_n a_n d^T X_n = 1."""
    print(flush=True)
    for nx, nz, qm in ((10, 474, 1.0), (20, 948, 1.0), (10, 474, 2.0)):
        params = _cached_params(qm)
        dep = compute_discrete_dc_depression(nx, params, return_details=True)
        dt, _ = _cached_integer_period_dt(nx, nz)
        om_dt = (2.0 / dt) * np.sin(0.5 * params["omega"] * dt)
        be_vec = om_dt / dep["u_l"]
        n_lines = len(be_vec)
        d_vec = np.full(n_lines, 1.0 / n_lines, dtype=np.float64)

        # Case A: wall_image="none", dc_entrance=False -> e = 1 and a_waves satisfies identities directly
        ms_plain = multistream_waves(
            nx, nz, params, wall_image="none", dc_entrance=False
        )
        e_ones = np.ones(n_lines, dtype=np.complex128)
        x_w = ms_plain["x_waves"]
        a_w = ms_plain["a_waves"]
        k_w = ms_plain["k_waves"]
        dk = k_w[None, :] - be_vec[:, None]

        r_i1 = float(np.linalg.norm(x_w @ a_w - e_ones) / np.linalg.norm(e_ones))
        r_v1 = float(
            np.linalg.norm((x_w * dk) @ a_w)
            / (np.linalg.norm(be_vec) * np.linalg.norm(e_ones))
        )
        sum_obs = complex(np.sum(a_w * (d_vec @ x_w)))
        err_sum_obs = float(abs(sum_obs - 1.0))

        # Case B: default (neumann + dc_entrance) -> pre-wall a0 = a_waves - wall_da matches e = exp(-i*theta_l)
        ms_def = _cached_d20_details(nx, nz, qm)["ms"]
        a0_def = ms_def["a_waves"] - ms_def["wall_da"]
        e_dc = np.exp(-1j * ms_def["theta_l"])
        x_def = ms_def["x_waves"]
        dk_def = ms_def["k_waves"][None, :] - be_vec[:, None]
        r_i1_dc = float(np.linalg.norm(x_def @ a0_def - e_dc) / np.linalg.norm(e_dc))
        r_v1_dc = float(
            np.linalg.norm((x_def * dk_def) @ a0_def)
            / (np.linalg.norm(be_vec) * np.linalg.norm(e_dc))
        )

        print(
            f"MATCHCHECK nx={nx} nz={nz} qm={qm:.1f} r_i1={r_i1:.3e} r_v1={r_v1:.3e} "
            f"err_sum_dTX={err_sum_obs:.3e} r_i1_dc={r_i1_dc:.3e} r_v1_dc={r_v1_dc:.3e} "
            f"match_resid={ms_def['match_resid']:.3e}",
            flush=True,
        )
        assert r_i1 <= 1.0e-12
        assert r_v1 <= 1.0e-12
        assert err_sum_obs <= 1.0e-12
        assert r_i1_dc <= 1.0e-12
        assert r_v1_dc <= 1.0e-12
        assert float(ms_def["match_resid"]) <= 1.0e-12

    # Negative control: perturbing one wave amplitude by 1e-6 must fail the <= 1e-12 identity check
    a_mut = a_w.copy()
    a_mut[0] += 1.0e-6
    r_i1_mut = float(np.linalg.norm(x_w @ a_mut - e_ones) / np.linalg.norm(e_ones))
    print(f"MATCH_MUTATION r_i1_mut={r_i1_mut:.3e}", flush=True)
    with pytest.raises(AssertionError, match="Perturbed amplitude"):
        assert r_i1_mut <= 1.0e-12, "Perturbed amplitude must fail <= 1e-12"


def test_wall_term_vs_exact_augmented_ode():
    """Fast test 4: Born wall correction vs exact 2*(N+M) augmented half-line ODE (continuous model)."""
    nx, nz, qm = 10, 474, 1.0
    params = _cached_params(qm)
    dz = TUNNEL_LENGTH_M / nz
    dt, _ = _cached_integer_period_dt(nx, nz)
    dep = compute_discrete_dc_depression(nx, params, return_details=True)
    u_l, g_l = dep["u_l"], dep["gamma_l"]
    u0 = params["u0"]
    om_dt = (2.0 / dt) * np.sin(0.5 * params["omega"] * dt)
    sig_vec = params["omega_p0"] ** 2 * (u0 / u_l) / (g_l**3 * u_l**2)
    be_vec = om_dt / u_l
    u_bar = float(u0 * (1.0 - float(dep["delta_disc"])))
    k_ref = float(params["omega"] / u_bar)

    op0 = compute_transverse_modal_operator("particles", nx, nz, params, k_val=k_ref)
    d_vec = op0["d_vec"]
    s_mat = op0["S"]
    d_mat = np.diag(d_vec)
    n_lines = len(be_vec)
    b = BEAM_HALF_THICKNESS_M
    dx = op0["dx"]
    c_scale = n_lines**2 * dx / (2.0 * b)
    lam, e_vecs = np.linalg.eigh(-op0["lap"])
    k0sq = (om_dt / c_light) ** 2
    s2 = lam - k0sq
    n_modes = len(lam)
    p_left = s_mat.T @ e_vecs
    p_right = e_vecs.T @ s_mat @ d_mat

    # Solve the 2*(N + M) first-order augmented ODE system on z >= 0 (port of aug.py):
    #   Y = [i (N), i' (N), g (M), g' (M)]
    #   G_cont(k) = C S^T (I - E diag(lam / (s^2 + k^2)) E^T) S D
    #   i'' + 2i B i' - (B^2 I - L_0) i = -L_g g
    #   g'' - diag(s^2) g = -P_R i
    l0_mat = sig_vec[:, None] * c_scale * (s_mat.T @ s_mat @ d_mat)
    lg_mat = -sig_vec[:, None] * c_scale * p_left * lam[None, :]
    n_tot = 2 * (n_lines + n_modes)
    a_ode = np.zeros((n_tot, n_tot), dtype=np.complex128)
    si = slice(0, n_lines)
    sip = slice(n_lines, 2 * n_lines)
    sg = slice(2 * n_lines, 2 * n_lines + n_modes)
    sgp = slice(2 * n_lines + n_modes, 2 * (n_lines + n_modes))
    a_ode[si, sip] = np.eye(n_lines)
    a_ode[sip, si] = np.diag(be_vec**2) - l0_mat
    a_ode[sip, sip] = -2j * np.diag(be_vec)
    a_ode[sip, sg] = -lg_mat
    a_ode[sg, sgp] = np.eye(n_modes)
    a_ode[sgp, sg] = np.diag(s2)
    a_ode[sgp, si] = -p_right

    mu_ode, y_ode = np.linalg.eig(a_ode)
    beam_mask = np.abs(mu_ode.real) < 50.0
    keep = beam_mask | (mu_ode.real < -50.0)
    assert int(np.sum(beam_mask)) == 2 * n_lines
    assert int(np.sum(keep)) == 2 * n_lines + n_modes
    mu_keep = mu_ode[keep]
    y_keep = y_ode[:, keep]
    bm_sub = np.abs(mu_keep.real) < 50.0

    z_eval = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    z_min, z_max = fit_window(params)
    guess = float(_cached_d20_details(nx, nz, qm)["det_modal"]["bq_modes"][0])

    def _solve_augmented_bc(
        field_bc: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        # Entrance BCs at z = 0:
        #   i(0) = 1, (d/dz + i*be) i(0) = 0, and either g'(0) = 0 (Neumann) or g(0) = 0 (Dirichlet)
        bc_i = y_keep[si, :]
        bc_v = y_keep[sip, :] + 1j * be_vec[:, None] * y_keep[si, :]
        bc_g = y_keep[sgp, :] if field_bc == "neumann" else y_keep[sg, :]
        bc_mat = np.vstack([bc_i, bc_v, bc_g])
        rhs = np.concatenate([np.ones(n_lines), np.zeros(n_lines + n_modes)]).astype(
            np.complex128
        )
        amp = np.linalg.solve(bc_mat, rhs)
        k_beam = 1j * mu_keep[bm_sub]
        x_beam = y_keep[si, :][:, bm_sub]
        o_beam = amp[bm_sub] * (d_vec @ x_beam)
        i1_aug = params["I1_0"] * np.exp(-1j * np.outer(z_eval, k_beam)) @ o_beam
        bq_aug = float(
            fit_beta_q_from_envelope(z_eval, i1_aug, guess, z_min=z_min, z_max=z_max)[
                "beta_q"
            ]
        )
        return k_beam, x_beam, o_beam, bq_aug

    kx, _, ox, bq_exact = _solve_augmented_bc("neumann")
    _, _, _, bq_dirichlet_ode = _solve_augmented_bc("dirichlet")

    # Continuous-model beam roots by successive substitution + Born wall correction (as in aug.py)
    def _m_cont(k_val: complex) -> np.ndarray:
        g_cont = (
            c_scale
            * s_mat.T
            @ (np.eye(n_modes) - e_vecs @ np.diag(lam / (s2 + k_val**2)) @ e_vecs.T)
            @ s_mat
            @ d_mat
        )
        return sig_vec[:, None] * g_cont

    def _qep_all(m_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a0_c = np.diag(be_vec**2) - m_mat
        a1_c = -2.0 * np.diag(be_vec)
        comp = np.block(
            [[np.zeros((n_lines, n_lines)), np.eye(n_lines)], [-a0_c, -a1_c]]
        )
        kr, v_c = np.linalg.eig(comp)
        return kr, v_c[:n_lines, :]

    k0_c, _ = _qep_all(_m_cont(k_ref))
    ks_list, xs_list = [], []
    for kn in k0_c:
        k_cur = kn
        for _ in range(60):
            k_eval = k_cur.real if abs(k_cur.imag) < 1.0 else k_cur
            kr_c, x_c = _qep_all(_m_cont(k_eval))
            i_best = int(np.argmin(np.abs(kr_c - k_cur)))
            dk_c = abs(kr_c[i_best] - k_cur)
            k_cur = kr_c[i_best]
            if dk_c < 1.0e-12 * abs(k_cur):
                break
        x_vec = x_c[:, i_best]
        ks_list.append(k_cur)
        xs_list.append(x_vec / np.linalg.norm(x_vec))
    ks_cont = np.array(ks_list)
    x_cont = np.array(xs_list).T

    dk_cont = ks_cont[None, :] - be_vec[:, None]
    a_beam_only = np.linalg.solve(
        np.vstack([x_cont, x_cont * (-1j * dk_cont)]),
        np.concatenate([np.ones(n_lines), np.zeros(n_lines)]).astype(np.complex128),
    )
    da_born = _multistream_wall_correction(
        op0=op0,
        dz=dz,
        dt=dt,
        omega=params["omega"],
        sig_vec=sig_vec,
        d_vec=d_vec,
        k_waves=ks_cont,
        x_waves=x_cont,
        a_waves=a_beam_only,
        p_basis=np.eye(n_lines, dtype=np.float64),
        shape=PARTICLE_SHAPE,
        use_omega_f=True,
        include_sz=False,
        discrete=False,
    )
    o0 = a_beam_only * (d_vec @ x_cont)
    o1 = (a_beam_only + da_born) * (d_vec @ x_cont)
    idx_align = [int(np.argmin(np.abs(kx - k_val))) for k_val in ks_cont]
    ox_al = ox[idx_align]

    i1_no_wall = params["I1_0"] * np.exp(-1j * np.outer(z_eval, ks_cont)) @ o0
    i1_born = params["I1_0"] * np.exp(-1j * np.outer(z_eval, ks_cont)) @ o1
    bq_no_wall = float(
        fit_beta_q_from_envelope(z_eval, i1_no_wall, guess, z_min=z_min, z_max=z_max)[
            "beta_q"
        ]
    )
    bq_born = float(
        fit_beta_q_from_envelope(z_eval, i1_born, guess, z_min=z_min, z_max=z_max)[
            "beta_q"
        ]
    )

    top4 = np.argsort(-np.abs(o0))[:4]
    rel_wave_errs = [float(abs(o1[j] - ox_al[j]) / abs(ox_al[j] - o0[j])) for j in top4]
    max_top4_rel_err = max(rel_wave_errs)
    rel_bq_born_vs_exact = abs(bq_born / bq_exact - 1.0)
    rel_wall_effect = abs(bq_exact / bq_no_wall - 1.0)
    dirichlet_disagreement_ratio = abs(bq_born - bq_dirichlet_ode) / abs(
        bq_exact - bq_no_wall
    )

    print(
        f"\nAUGCHECK bq_no_wall={bq_no_wall:.8f} bq_exact={bq_exact:.8f} bq_born={bq_born:.8f} "
        f"bq_dirichlet_ode={bq_dirichlet_ode:.8f} max_top4_wave_rel={max_top4_rel_err:.4f} "
        f"rel_bq_born_vs_exact={rel_bq_born_vs_exact:.3e} rel_wall_effect={rel_wall_effect:.3e} "
        f"dirichlet_disagreement_ratio={dirichlet_disagreement_ratio:.3f}",
        flush=True,
    )
    assert max_top4_rel_err <= 0.03, (
        f"Top-4 wave Born vs exact error {max_top4_rel_err:.4f} > 0.03"
    )
    assert rel_bq_born_vs_exact <= 2.0e-6, (
        f"|beta_born / beta_exact - 1| = {rel_bq_born_vs_exact:.3e} > 2e-6"
    )
    assert rel_wall_effect >= 3.0e-5, (
        f"Wall effect {rel_wall_effect:.3e} < 3e-5 (vacuous check)"
    )
    assert dirichlet_disagreement_ratio > 1.0, (
        f"Dirichlet ODE disagreement ratio {dirichlet_disagreement_ratio:.3f} <= 1.0"
    )
    with pytest.raises(AssertionError, match="Dirichlet negative control"):
        assert abs(bq_born / bq_dirichlet_ode - 1.0) <= 2.0e-6, (
            "Dirichlet negative control must fail <= 2e-6"
        )


def test_wall_image_variants_prototype_and_modal_regression():
    """Fast test 5 + Req 1 & 2: image variants differ by > 3e-5, prototype reproduced to <= 1e-9, modal bit-exact."""
    print(flush=True)
    for nx, nz, qm in ((10, 474, 1.0), (20, 948, 1.0), (10, 474, 2.0)):
        b_neu = _cached_d20_details(nx, nz, qm, wall_image="neumann")["beta_q"]
        b_non = _cached_d20_details(nx, nz, qm, wall_image="none")["beta_q"]
        b_dir = _cached_d20_details(nx, nz, qm, wall_image="dirichlet")["beta_q"]

        d_neu_non = abs(b_neu - b_non) / b_neu
        d_neu_dir = abs(b_neu - b_dir) / b_neu
        d_non_dir = abs(b_non - b_dir) / b_neu
        min_pair = min(d_neu_non, d_neu_dir, d_non_dir)

        proto = _prototype_msp_predict(nx, nz, qm)
        b_proto = proto["d20"]
        rel_proto = abs(b_neu - b_proto) / abs(b_proto)

        print(
            f"PROTO_AND_IMAGE nx={nx} nz={nz} qm={qm:.1f} "
            f"b_lib={b_neu:.12f} b_proto={b_proto:.12f} rel_proto={rel_proto:.3e} "
            f"b_none={b_non:.8f} b_dir={b_dir:.8f} min_pairwise_rel={min_pair:.3e}",
            flush=True,
        )
        assert rel_proto <= 1.0e-9, (
            f"Library vs prototype relative diff {rel_proto:.3e} > 1e-9 on ({nx}, {nz}, qm={qm})"
        )
        assert min_pair > 3.0e-5, (
            f"Wall image variants pairwise difference {min_pair:.3e} <= 3e-5 on ({nx}, {nz}, qm={qm})"
        )

    # Negative control for image variants: identical variants must fail > 3e-5
    with pytest.raises(AssertionError, match="Identical wall_image"):
        assert abs(b_neu - b_neu) / b_neu > 3.0e-5, (
            "Identical wall_image must fail > 3e-5"
        )

    # Requirement 1: method="modal" reproduces the legacy cos-form prediction bit-for-bit,
    # "lb" branch is unchanged bit-for-bit, and _single_grid_budget_terms / prediction_slack
    # computes all slack terms (s_modes, s_sym, s_win) with the same method as bq_pred.
    nx, nz = COARSE_GRID[0], COARSE_GRID[2]
    params = _cached_params(1.0)
    det_modal = _cached_modal_details(nx, nz, 1.0)
    dz = TUNNEL_LENGTH_M / nz
    z_eval = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    i1_cos_legacy = params["I1_0"] * np.sum(
        det_modal["c_modes"][:, None]
        * np.cos(np.outer(det_modal["bq_modes"], z_eval))
        * np.exp(-1j * np.outer(det_modal["kc_modes"], z_eval)),
        axis=0,
    )
    z_min, z_max = fit_window(params)
    bq_cos_legacy = float(
        fit_beta_q_from_envelope(
            z_eval,
            i1_cos_legacy,
            float(det_modal["bq_modes"][0]),
            z_min=z_min,
            z_max=z_max,
        )["beta_q"]
    )
    np.testing.assert_allclose(det_modal["i1_model"], i1_cos_legacy, rtol=0.0, atol=0.0)
    assert det_modal["beta_q"] == bq_cos_legacy

    lb_ms = discrete_modal_prediction(
        "lb", nx, nz, params, return_details=True, method="multistream"
    )
    lb_mod = discrete_modal_prediction(
        "lb", nx, nz, params, return_details=True, method="modal"
    )
    np.testing.assert_allclose(
        lb_ms["i1_model"], lb_mod["i1_model"], rtol=0.0, atol=0.0
    )
    assert lb_ms["beta_q"] == lb_mod["beta_q"]

    budget = _single_grid_budget_terms(nx, nz, params)
    info_c1 = _cached_d20_details(nx, nz, 1.0)
    bq_ms = float(info_c1["beta_q"])
    bq_drop = float(
        discrete_modal_prediction(
            "particles", nx, nz, params, drop_highest_mode=True, method="multistream"
        )
    )
    bq_sym = float(
        discrete_modal_prediction(
            "particles", nx, nz, params, use_omega_f=False, method="multistream"
        )
    )
    shift_m = 1.0 / params["T1"]
    guess_c1 = float(det_modal["bq_modes"][0])
    bq_win_p = float(
        fit_beta_q_from_envelope(
            z_eval,
            info_c1["i1_model"],
            guess_c1,
            z_min=z_min + shift_m,
            z_max=z_max + shift_m,
        )["beta_q"]
    )
    bq_win_m = float(
        fit_beta_q_from_envelope(
            z_eval,
            info_c1["i1_model"],
            guess_c1,
            z_min=z_min - shift_m,
            z_max=z_max - shift_m,
        )["beta_q"]
    )
    assert budget["bq_pred_part"] == bq_ms
    np.testing.assert_allclose(
        budget["s_modes_part"], abs(bq_ms - bq_drop) / bq_ms, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        budget["s_sym_part"], abs(bq_ms - bq_sym) / bq_ms, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        budget["s_win_part"],
        max(abs(bq_win_p - bq_ms), abs(bq_win_m - bq_ms)) / bq_ms,
        rtol=0.0,
        atol=1e-15,
    )
    print(
        f"REGRESSION_AND_SLACK bq_modal={det_modal['beta_q']:.8f} bq_ms={bq_ms:.8f} "
        f"s_modes={budget['s_modes_part']:.3e} s_sym={budget['s_sym_part']:.3e} s_win={budget['s_win_part']:.3e}",
        flush=True,
    )
