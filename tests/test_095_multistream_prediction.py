# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Fast verification tests for Decision D20 multistream prediction of particle ``I_1(z)``.

Covers the D20 fast verification checks, referee-required derived bounds, monkeypatched
library-code negative controls, and exact discrete half-line wall-term verification:

1. Reduction R1 (identity check: shear-free, frozen ``G``, no wall term, no DC entrance
   phase, no observable ``sinc``) to the legacy modal ``I_1(z)`` vector within ``1e-10``
   relative and ``beta_q`` within ``1e-9`` relative, with monkeypatched ``Omega_f -> omega``
   mutations in ``_multistream_leapfrog_omega`` and ``_multistream_entrance_dk`` as
   negative controls.
2. Multistream root residuals ``sigma_min / sigma_max <= root_tol`` (where
   ``root_tol = h_sub * eps_mach * (|k_n| + max_l beta_l)^2 / sigma_max(T(k_n))``) and
   ``<= 1e-12``, pairwise distinctness ``|Delta k| > root_k_tol`` and ``> 1e-8`` rad/m,
   agreement of ``include_shear=False`` roots with legacy ``k_c +- beta_q`` to ``1e-8`` rad/m,
   agreement of the sheared fundamental mode with ``studies/p5b-shear/qep_ref.py`` ``brentq``
   root to ``1e-6`` relative, agreement between ``"even"`` and ``"full"`` subspaces to
   ``1e-9`` relative, monkeypatched ``Omega_f -> omega`` negative control, and verification
   that ``multistream_waves`` raises ``RuntimeError`` when ``tau_y2 <= 1e-12``, when a root
   residual exceeds ``root_tol``, when two roots coincide within ``root_k_tol``, or when the
   entrance matching system is singular or exceeds ``1e-12`` residual.
3. Entrance matching identities ``sum_n a_n X_n = e``,
   ``sum_n a_n (k_n - beta_l) X_{l,n} = 0`` to ``1e-12`` relative, and
   ``sum_n a_n d^T X_n = 1`` for uniform entrance ``e = 1``, with a monkeypatched
   fast/slow entrance swap in ``_multistream_entrance_dk`` as negative control.
4. First-order (Born) PEC-wall near-field correction vs exact ``2 * (N + M)`` augmented
   half-line ODE solve on the continuous rational model (coarse ``q/m = 1``) within the
   second-order Born remainder bound ``2 * eps^2`` for wave amplitudes and
   ``rel_wall_effect * 2 * eps`` for ``beta_q`` (where
   ``eps = max_{j in top4} |da_born[j] / a_0[j]|``), with a monkeypatched sign flip of
   ``_multistream_wall_correction`` as negative control.
5. Wall-image variant separation against ``prediction_slack("particles", ...)`` on the
   same grid, identity check against the reference prototype ``_prototype_msp_predict`` to
   ``<= 1e-9`` relative, bit-for-bit preservation of ``method="modal"`` and ``"lb"``,
   error-budget label verification, and monkeypatched library negative controls (wall
   sign flip, dropped observable ``sinc^p``, fast/slow entrance swap, ``Omega_f -> omega``).
6. Production discrete wall term (``include_sz=True``, ``discrete=True``, ``sinc^p``,
   ``sinh`` factors) verified against (a) the exact discrete half-line Green's function
   with image ``j -> -j - 1`` (Neumann ghost ``psi_{-1} = psi_0``) to ``<= 1e-12`` and
   (b) the exact coupled discrete half-line solve to the derived second-order Born
   remainder bounds ``2 * eps^2`` (top-4 waves) and ``rel_wall_effect * 2 * eps``
   (``beta_q``), with monkeypatched negative controls.
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

import examples._space_charge_waves_common as scw
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
    format_error_budget,
    multistream_waves,
    prediction_slack,
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


def _fibonacci_line_x(n_lines: int = 16) -> np.ndarray:
    """Return zero-mean asymmetric Fibonacci-disk transverse line positions [m] for M3."""
    k_idx = np.arange(n_lines, dtype=np.float64)
    r_k = BEAM_HALF_THICKNESS_M * np.sqrt((k_idx + 0.5) / float(n_lines))
    phi_k = k_idx * np.pi * (3.0 - np.sqrt(5.0))
    x_raw = r_k * np.cos(phi_k)
    return x_raw - float(np.mean(x_raw))


def _prototype_msp_predict(nx: int, nz: int, qm_scale: float) -> dict[str, Any]:
    """Reference prototype implementation ported verbatim from ``msp.py`` (identity check)."""
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
        tau_y2 = kappa**2 - (om_dt / c_light) ** 2
        if tau_y2 <= 1.0e-12:
            raise RuntimeError(f"tau_y2={tau_y2:.6e} <= 1e-12")
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


def test_reduction_r1_and_omega_mutation(monkeypatch: pytest.MonkeyPatch):
    """Fast test 1 (R1 identity check): shear-free frozen-G multistream reduces to legacy modal I1(z) to <= 1e-10."""
    print(flush=True)

    def _assert_r1_reduction(nx: int, nz: int, qm: float) -> None:
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
            f"R1_IDENTITY_CHECK nx={nx} nz={nz} qm={qm:.1f} rel_i1={rel_i1:.3e} "
            f"beta_ms={det_ms['beta_q']:.10f} beta_mod={det_mod['beta_q']:.10f} rel_bq={rel_bq:.3e}",
            flush=True,
        )
        assert rel_i1 <= 1.0e-10, (
            f"R1 I1 relative difference {rel_i1:.3e} > 1e-10 on ({nx}, {nz}, qm={qm})"
        )
        assert rel_bq <= 1.0e-9, (
            f"R1 beta_q relative difference {rel_bq:.3e} > 1e-9 on ({nx}, {nz}, qm={qm})"
        )

    for nx, nz, qm in ((10, 474, 1.0), (20, 948, 1.0), (10, 474, 2.0)):
        _assert_r1_reduction(nx, nz, qm)

    # Monkeypatched negative control 1a: using omega instead of Omega_f in multistream_waves
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_leapfrog_omega",
            lambda omega, dt, use_omega_f=True: float(omega),
        )
        with pytest.raises(AssertionError, match="R1 I1 relative difference"):
            _assert_r1_reduction(10, 474, 1.0)

    # Monkeypatched negative control 1b: using omega / u_bar instead of Omega_f / u_bar in the v_1 entrance row
    params_1 = _cached_params(1.0)
    dep_1 = compute_discrete_dc_depression(10, params_1, return_details=True)
    u_bar_1 = float(params_1["u0"] * (1.0 - float(dep_1["delta_disc"])))
    be_omega_1 = float(params_1["omega"] / u_bar_1)
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_entrance_dk",
            lambda k_waves, be_vec: k_waves[None, :] - be_omega_1,
        )
        with pytest.raises(AssertionError, match="R1 I1 relative difference"):
            _assert_r1_reduction(10, 474, 1.0)


def test_roots_residuals_legacy_qep_and_subspaces(monkeypatch: pytest.MonkeyPatch):
    """Fast test 2: root SVD residuals <= root_tol, legacy roots, qep_ref brentq root, subspaces, and RuntimeError guards."""
    print(flush=True)

    def _assert_roots_and_legacy(nx: int, nz: int, qm: float) -> None:
        params = _cached_params(qm)
        ms = scw.multistream_waves(nx, nz, params)
        ks = ms["k_waves"]
        max_res = float(np.max(ms["root_resid"]))
        max_tol = float(np.max(ms["root_tol"]))
        max_ratio = float(np.max(ms["root_resid"] / ms["root_tol"]))
        dk_mat = np.abs(ks[:, None] - ks[None, :])
        np.fill_diagonal(dk_mat, np.inf)
        min_dk = float(np.min(dk_mat))

        # Unsheared roots must contain all active legacy k_c +- beta_q to <= 1e-8 rad/m
        ms_unif = scw.multistream_waves(nx, nz, params, include_shear=False)
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
        info = _cached_d20_details(nx, nz, qm)
        info_full = _cached_d20_details(nx, nz, qm, subspace="full")
        rel_even_full = float(abs(info["beta_q"] / info_full["beta_q"] - 1.0))

        print(
            f"ROOTSCHECK nx={nx} nz={nz} qm={qm:.1f} max_svd_resid={max_res:.3e} "
            f"max_root_tol={max_tol:.3e} max_resid_over_tol={max_ratio:.3f} "
            f"min_dk={min_dk:.3e} root_k_tol={ms['root_k_tol']:.3e} "
            f"legacy_root_err={max_legacy_dk:.3e} rel_even_vs_full={rel_even_full:.3e}",
            flush=True,
        )
        assert bool(np.all(ms["root_resid"] <= ms["root_tol"])), (
            f"Root residual exceeds derived root_tol: max ratio={max_ratio:.3f}"
        )
        assert max_res <= 1.0e-12, f"Max root residual {max_res:.3e} > 1e-12"
        assert min_dk > ms["root_k_tol"], (
            f"Pairwise root separation {min_dk:.3e} <= root_k_tol {ms['root_k_tol']:.3e} rad/m"
        )
        assert min_dk > 1.0e-8, f"Pairwise root separation {min_dk:.3e} <= 1e-8 rad/m"
        assert bool(np.all(ms_unif["root_resid"] <= ms_unif["root_tol"]))
        assert float(np.max(ms_unif["root_resid"])) <= 1.0e-12
        assert max_legacy_dk <= 1.0e-8, (
            f"Unsheared root mismatch vs legacy {max_legacy_dk:.3e} > 1e-8 rad/m"
        )
        assert rel_even_full <= 1.0e-9, (
            f"Even vs full subspace mismatch {rel_even_full:.3e} > 1e-9"
        )

    for nx, nz, qm in ((10, 474, 1.0), (20, 948, 1.0), (10, 474, 2.0)):
        _assert_roots_and_legacy(nx, nz, qm)

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

    # Asymmetric Fibonacci disk (M3): must select "full" subspace automatically with root_resid <= root_tol
    x_fib = _fibonacci_line_x(16)
    ms_m3 = multistream_waves(10, 474, _cached_params(1.0), line_x=x_fib)
    print(
        f"M3_SUBSPACE subspace={ms_m3['subspace']} max_svd_resid={np.max(ms_m3['root_resid']):.3e} "
        f"max_root_tol={np.max(ms_m3['root_tol']):.3e}",
        flush=True,
    )
    assert ms_m3["subspace"] == "full"
    assert bool(np.all(ms_m3["root_resid"] <= ms_m3["root_tol"]))
    assert float(np.max(ms_m3["root_resid"])) <= 1.0e-12

    # Monkeypatched negative control: using omega instead of Omega_f in multistream_waves fails _assert_roots_and_legacy
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_leapfrog_omega",
            lambda omega, dt, use_omega_f=True: float(omega),
        )
        with pytest.raises(AssertionError, match="Unsheared root mismatch vs legacy"):
            _assert_roots_and_legacy(10, 474, 1.0)

    # Item 6 RuntimeError guards in multistream_waves:
    params_1 = _cached_params(1.0)
    # Guard 1: tau_y2 <= 1.0e-12 raises RuntimeError instead of clamping silently
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_leapfrog_omega",
            lambda omega, dt, use_omega_f=True: (
                float(omega) if use_omega_f else 25.0 * float(omega)
            ),
        )
        with pytest.raises(RuntimeError, match="tau_y2="):
            scw.multistream_waves(10, 474, params_1, use_omega_f=False)

    # Guard 2: root_resid > root_tol raises RuntimeError with residual and threshold numbers
    orig_eig = np.linalg.eig
    with monkeypatch.context() as mp:
        mp.setattr(
            np.linalg,
            "eig",
            lambda a: (orig_eig(a)[0] + 1.0e-3, orig_eig(a)[1]),
        )
        with pytest.raises(RuntimeError, match="has relative residual"):
            scw.multistream_waves(10, 474, params_1, freeze_g=True)

    # Guard 3: coinciding roots (|dk| <= root_k_tol when omega_p_scale=0) raises RuntimeError
    with pytest.raises(RuntimeError, match="coincide with"):
        scw.multistream_waves(10, 474, params_1, omega_p_scale=0.0)

    # Guard 4a: singular entrance matching system raises RuntimeError (no lstsq fallback)
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_entrance_dk",
            lambda k_waves, be_vec: np.zeros(
                (len(be_vec), len(k_waves)), dtype=np.complex128
            ),
        )
        with pytest.raises(RuntimeError, match="singular entrance matching system"):
            scw.multistream_waves(10, 474, params_1)

    # Guard 4b: entrance matching residual > 1e-12 raises RuntimeError
    orig_solve = np.linalg.solve
    with monkeypatch.context() as mp:

        def _perturbed_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            sol = orig_solve(a, b)
            if np.iscomplexobj(a) and a.ndim == 2 and b.ndim == 1:
                sol = sol.copy()
                sol[0] += 1.0e-6
            return sol

        mp.setattr(np.linalg, "solve", _perturbed_solve)
        with pytest.raises(RuntimeError, match="entrance matching residual"):
            scw.multistream_waves(10, 474, params_1)


def test_entrance_matching_identities(monkeypatch: pytest.MonkeyPatch):
    """Fast test 3: verify entrance matching identities to <= 1e-12 relative and monkeypatched fast/slow swap."""
    print(flush=True)

    def _assert_entrance_matching(nx: int, nz: int, qm: float) -> None:
        params = _cached_params(qm)
        dep = compute_discrete_dc_depression(nx, params, return_details=True)
        dt, _ = _cached_integer_period_dt(nx, nz)
        om_dt = (2.0 / dt) * np.sin(0.5 * params["omega"] * dt)
        be_vec = om_dt / dep["u_l"]
        n_lines = len(be_vec)
        d_vec = np.full(n_lines, 1.0 / n_lines, dtype=np.float64)

        # Case A: wall_image="none", dc_entrance=False -> e = 1 and a_waves satisfies identities directly
        ms_plain = scw.multistream_waves(
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
        ms_def = scw.multistream_waves(nx, nz, params)
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
        assert r_i1 <= 1.0e-12, f"r_i1={r_i1:.3e} > 1e-12"
        assert r_v1 <= 1.0e-12, f"r_v1={r_v1:.3e} > 1e-12"
        assert err_sum_obs <= 1.0e-12, f"err_sum_obs={err_sum_obs:.3e} > 1e-12"
        assert r_i1_dc <= 1.0e-12, f"r_i1_dc={r_i1_dc:.3e} > 1e-12"
        assert r_v1_dc <= 1.0e-12, f"r_v1_dc={r_v1_dc:.3e} > 1e-12"
        assert float(ms_def["match_resid"]) <= 1.0e-12

    for nx, nz, qm in ((10, 474, 1.0), (20, 948, 1.0), (10, 474, 2.0)):
        _assert_entrance_matching(nx, nz, qm)

    # Monkeypatched negative control: swapping fast/slow wave order in _multistream_entrance_dk fails _assert_entrance_matching
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_entrance_dk",
            lambda k_waves, be_vec: k_waves[::-1][None, :] - be_vec[:, None],
        )
        with pytest.raises(AssertionError, match="r_v1="):
            _assert_entrance_matching(10, 474, 1.0)


def test_wall_term_vs_exact_augmented_ode(monkeypatch: pytest.MonkeyPatch):
    """Fast test 4: Born wall correction vs exact 2*(N+M) augmented half-line ODE (continuous model).

    Derivation of the second-order Born remainder tolerance (Referee Item 3b):
    Let ``a_0`` be the beam-only entrance amplitude vector and ``da^(1) = K_wall a_0`` be the
    first-order Born PEC-wall amplitude shift returned by ``_multistream_wall_correction``.
    In the coupled beam-evanescent system, the exact wave amplitude vector is the Neumann series
    ``a_exact = (I - K_wall)^{-1} a_0 = a_0 + da^(1) + K_wall da^(1) + O(||K_wall||^3)``.
    With ``eps = max_{j in top4} |da^(1)_j / a_{0,j}|`` measuring the relative first-order wall
    coupling on the dominant waves, the relative error of the first-order Born shift
    ``da^(1)`` with respect to the exact shift ``a_exact - a_0`` is bounded by the second-order
    Born remainder ``2 * eps^2`` (factor 2 accounting for the fast/slow wave pair coupling and
    geometric tail ``1 / (1 - eps)``), and the induced ``beta_q`` difference between first-order
    Born and exact augmented solve is bounded by ``rel_wall_effect * 2 * eps``.
    """
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
    o0 = a_beam_only * (d_vec @ x_cont)
    idx_align = [int(np.argmin(np.abs(kx - k_val))) for k_val in ks_cont]
    ox_al = ox[idx_align]
    i1_no_wall = params["I1_0"] * np.exp(-1j * np.outer(z_eval, ks_cont)) @ o0
    bq_no_wall = float(
        fit_beta_q_from_envelope(z_eval, i1_no_wall, guess, z_min=z_min, z_max=z_max)[
            "beta_q"
        ]
    )
    rel_wall_effect = abs(bq_exact / bq_no_wall - 1.0)
    top4 = np.argsort(-np.abs(o0))[:4]

    # Reference first-order relative wall amplitude eps from unmutated wall correction:
    da_ref = scw._multistream_wall_correction(
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
    eps_born = float(max(abs(da_ref[j] / a_beam_only[j]) for j in top4))
    tol_wave_born = float(2.0 * (eps_born**2))
    tol_bq_born = float(rel_wall_effect * 2.0 * eps_born)

    def _assert_continuous_born_vs_exact() -> None:
        da_born = scw._multistream_wall_correction(
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
        o1 = (a_beam_only + da_born) * (d_vec @ x_cont)
        i1_born = params["I1_0"] * np.exp(-1j * np.outer(z_eval, ks_cont)) @ o1
        bq_born = float(
            fit_beta_q_from_envelope(z_eval, i1_born, guess, z_min=z_min, z_max=z_max)[
                "beta_q"
            ]
        )
        rel_wave_errs = [
            float(abs(o1[j] - ox_al[j]) / abs(ox_al[j] - o0[j])) for j in top4
        ]
        max_top4_rel_err = max(rel_wave_errs)
        rel_bq_born_vs_exact = abs(bq_born / bq_exact - 1.0)
        dirichlet_disagreement_ratio = abs(bq_born - bq_dirichlet_ode) / abs(
            bq_exact - bq_no_wall
        )

        print(
            f"\nAUGCHECK bq_no_wall={bq_no_wall:.8f} bq_exact={bq_exact:.8f} bq_born={bq_born:.8f} "
            f"bq_dirichlet_ode={bq_dirichlet_ode:.8f} eps={eps_born:.6f} "
            f"max_top4_wave_rel={max_top4_rel_err:.6f} tol_2eps2={tol_wave_born:.6f} "
            f"rel_bq_born_vs_exact={rel_bq_born_vs_exact:.3e} tol_bq_2eps={tol_bq_born:.3e} "
            f"rel_wall_effect={rel_wall_effect:.3e} dirichlet_disagreement_ratio={dirichlet_disagreement_ratio:.3f}",
            flush=True,
        )
        assert max_top4_rel_err <= tol_wave_born, (
            f"Top-4 wave Born vs exact error {max_top4_rel_err:.6f} > 2*eps^2={tol_wave_born:.6f}"
        )
        assert rel_bq_born_vs_exact <= tol_bq_born, (
            f"|beta_born / beta_exact - 1| = {rel_bq_born_vs_exact:.3e} > rel_wall_effect*2*eps={tol_bq_born:.3e}"
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

    _assert_continuous_born_vs_exact()

    # Monkeypatched negative control: flipping the sign of _multistream_wall_correction fails _assert_continuous_born_vs_exact
    orig_wall = scw._multistream_wall_correction
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_wall_correction",
            lambda *args, **kwargs: -orig_wall(*args, **kwargs),
        )
        with pytest.raises(AssertionError, match="Top-4 wave Born vs exact error"):
            _assert_continuous_born_vs_exact()


@pytest.mark.parametrize(
    ("case_label", "nx", "nz", "qm", "use_m3", "variant"),
    [
        ("10x474_qm1", 10, 474, 1.0, False, "none"),
        ("10x474_qm1", 10, 474, 1.0, False, "dirichlet"),
        ("10x474_qm1", 10, 474, 1.0, False, "modal"),
        ("20x948_qm1", 20, 948, 1.0, False, "none"),
        ("20x948_qm1", 20, 948, 1.0, False, "dirichlet"),
        pytest.param(
            "20x948_qm1",
            20,
            948,
            1.0,
            False,
            "modal",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Item 3a reported non-separation: on (20, 948, qm=1.0), legacy method='modal' "
                    "cancellation gives rel_diff=9.830e-06 <= slack=7.797e-05 (threshold unadjusted)."
                ),
            ),
        ),
        pytest.param(
            "10x474_qm2",
            10,
            474,
            2.0,
            False,
            "none",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Item 3a reported non-separation: on (10, 474, qm=2.0), wall_image='none' "
                    "gives rel_diff=3.165e-05 <= slack=3.690e-05 (threshold unadjusted)."
                ),
            ),
        ),
        ("10x474_qm2", 10, 474, 2.0, False, "dirichlet"),
        ("10x474_qm2", 10, 474, 2.0, False, "modal"),
        ("10x474_qm1_M3", 10, 474, 1.0, True, "none"),
        ("10x474_qm1_M3", 10, 474, 1.0, True, "dirichlet"),
        ("10x474_qm1_M3", 10, 474, 1.0, True, "modal"),
    ],
)
def test_wrong_variants_exceed_prediction_slack(
    case_label: str,
    nx: int,
    nz: int,
    qm: float,
    use_m3: bool,
    variant: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Referee Item 3a: each wrong variant ('none', 'dirichlet') and method='modal' vs prediction_slack on the same grid."""
    params = _cached_params(qm)
    line_x = _fibonacci_line_x(16) if use_m3 else None
    slack_info = prediction_slack(
        "particles", nx, nz, params, line_x=line_x, method="multistream"
    )
    b_d20 = float(slack_info["bq_pred"])
    slack = float(slack_info["slack"])

    def _assert_variant_exceeds_slack() -> None:
        if variant == "modal":
            b_var = float(
                discrete_modal_prediction(
                    "particles", nx, nz, params, line_x=line_x, method="modal"
                )
            )
        else:
            b_var = float(
                discrete_modal_prediction(
                    "particles",
                    nx,
                    nz,
                    params,
                    line_x=line_x,
                    wall_image=variant,
                    method="multistream",
                )
            )
        rel_diff = abs(b_d20 - b_var) / b_d20
        print(
            f"VARIANT_VS_SLACK case={case_label} variant={variant:9s} "
            f"b_d20={b_d20:.8f} b_var={b_var:.8f} rel_diff={rel_diff:.6e} "
            f"slack={slack:.6e} separates={rel_diff > slack}",
            flush=True,
        )
        assert rel_diff > slack, (
            f"case={case_label} variant={variant}: rel_diff={rel_diff:.6e} <= slack={slack:.6e}"
        )

    _assert_variant_exceeds_slack()

    # Monkeypatched negative control on (10, 474, qm=1.0, "none"): zeroing _multistream_wall_correction
    # makes D20 coincide with wall_image="none", failing _assert_variant_exceeds_slack.
    if case_label == "10x474_qm1" and variant == "none":
        with monkeypatch.context() as mp:
            mp.setattr(
                scw,
                "_multistream_wall_correction",
                lambda *args, **kwargs: np.zeros_like(kwargs["a_waves"]),
            )
            b_d20 = float(
                discrete_modal_prediction(
                    "particles", nx, nz, params, line_x=line_x, method="multistream"
                )
            )
            with pytest.raises(AssertionError, match="rel_diff="):
                _assert_variant_exceeds_slack()


def test_wall_image_variants_prototype_and_modal_regression(
    monkeypatch: pytest.MonkeyPatch,
):
    """Fast test 5: image variants pairwise > 3e-5, prototype identity check <= 1e-9, modal bit-exact, and library mutations."""
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
            f"PROTO_IDENTITY_AND_IMAGE nx={nx} nz={nz} qm={qm:.1f} "
            f"b_lib={b_neu:.12f} b_proto={b_proto:.12f} rel_proto_identity={rel_proto:.3e} "
            f"b_none={b_non:.8f} b_dir={b_dir:.8f} min_pairwise_rel={min_pair:.3e}",
            flush=True,
        )
        assert rel_proto <= 1.0e-9, (
            f"Library vs prototype identity relative diff {rel_proto:.3e} > 1e-9 on ({nx}, {nz}, qm={qm})"
        )
        assert min_pair > 3.0e-5, (
            f"Wall image variants pairwise difference {min_pair:.3e} <= 3e-5 on ({nx}, {nz}, qm={qm})"
        )

    # Monkeypatched library-code mutations (Referee Item 2): each must fail the D20 prediction check
    nx, nz = COARSE_GRID[0], COARSE_GRID[2]
    params = _cached_params(1.0)
    slack_1 = float(prediction_slack("particles", nx, nz, params)["slack"])
    b_ref_1 = float(_cached_d20_details(nx, nz, 1.0)["beta_q"])

    def _assert_d20_matches_reference(tol: float) -> None:
        b_cur = float(
            scw.discrete_modal_prediction(
                "particles", nx, nz, params, method="multistream"
            )
        )
        rel_err = abs(b_cur - b_ref_1) / b_ref_1
        assert rel_err <= tol, f"D20 prediction rel_err={rel_err:.6e} > tol={tol:.6e}"

    _assert_d20_matches_reference(1.0e-12)

    # Mutation A: flip sign of wall correction in _multistream_wall_correction (rel_err = 1.30e-4 > slack)
    orig_wall = scw._multistream_wall_correction
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_wall_correction",
            lambda *args, **kwargs: -orig_wall(*args, **kwargs),
        )
        with pytest.raises(AssertionError, match="D20 prediction rel_err="):
            _assert_d20_matches_reference(slack_1)

    # Mutation B: drop per-wave sinc^p in observable weights _multistream_observable_weights
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_observable_weights",
            lambda a_waves, d_vec, x_waves, k_waves, dz, shape, use_obs_sinc: (
                a_waves * (d_vec @ x_waves)
            ),
        )
        with pytest.raises(AssertionError, match="D20 prediction rel_err="):
            _assert_d20_matches_reference(1.0e-9)

    # Mutation C: swap fast/slow entrance condition in _multistream_entrance_dk (rel_err = 5.46e-5 > slack)
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_entrance_dk",
            lambda k_waves, be_vec: k_waves[::-1][None, :] - be_vec[:, None],
        )
        with pytest.raises(AssertionError, match="D20 prediction rel_err="):
            _assert_d20_matches_reference(slack_1)

    # Mutation D: use omega instead of Omega_f in _multistream_leapfrog_omega (rel_err = 3.45e-7 > 1e-9)
    with monkeypatch.context() as mp:
        mp.setattr(
            scw,
            "_multistream_leapfrog_omega",
            lambda omega, dt, use_omega_f=True: float(omega),
        )
        with pytest.raises(AssertionError, match="D20 prediction rel_err="):
            _assert_d20_matches_reference(1.0e-9)

    # Requirement 1: method="modal" reproduces the legacy cos-form prediction bit-for-bit,
    # "lb" branch is unchanged bit-for-bit, and _single_grid_budget_terms / prediction_slack
    # exposes root_resid and match_resid and formats updated D19/D20 labels.
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
    assert budget["root_resid_part"] is not None
    assert budget["root_tol_part"] is not None
    assert budget["match_resid_part"] is not None
    assert bool(np.all(budget["root_resid_part"] <= budget["root_tol_part"]))
    assert float(budget["match_resid_part"]) <= 1.0e-12

    budget_str = format_error_budget(nx, nz, budget)
    assert "bq_pred_part (D20 Particle multistream prediction)" in budget_str
    assert "amp_pred_part (H_0(wall)*cos(bq*dz/2))" in budget_str
    assert (
        "tol_amp_part (2*(v_inj/u_dep-1)*p*(be*dz)^2/24+(1-cos(bq*dz/2))+1e-12)"
        in budget_str
    )

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


def test_production_discrete_wall_term_vs_exact_half_line(
    monkeypatch: pytest.MonkeyPatch,
):
    """Fast test 6 (Referee Item 7): verify production discrete wall term against exact discrete half-line solve.

    Verifies two levels on the production discrete system (``include_sz=True``,
    ``discrete=True``, ``sinc^p`` axial shape factors, ``sinh`` discrete residue factors):

    1. **Exact discrete half-line Green's function with image ``j -> -j - 1`` (Referee V1)**:
       For each transverse mode ``m``, the discrete axial difference operator on the half-line
       ``j >= 0`` with PEC wall ``E_x(z=0) = 0 => phi_0 = 0 => psi_{-1} = psi_0`` (Neumann ghost
       reflection across ``z = 0``, image index ``j -> -j - 1``) has tridiagonal matrix
       ``L_m[0,0] = 1/dz^2 + tau_m^2``, ``L_m[n,n] = 2/dz^2 + tau_m^2`` (``n >= 1``), and
       ``L_m[n, n+-1] = -1/dz^2``. Its inverse matches the closed-form image Green's function
       ``G_m(n, j) = (lam_m * rho_m * dz^2 / (1 - rho_m^2)) * (rho_m^{|n-j|} + rho_m^{n+j+1})``
       to ``<= 100 * eps_mach``, and direct summation of the image-minus-continuation kernel
       ``rho_m^{n+j+1}`` over ``j = 0 .. J_max - 1`` reproduces the production
       ``_multistream_wall_correction(..., include_sz=True, discrete=True)`` to within the
       geometric truncation tail ``rho_0^{J_max} / (1 - rho_0) + 100 * eps_mach``.

    2. **Production first-order Born discrete wall term vs exact coupled discrete half-line solve**:
       In the exact coupled half-line ansatz ``I_1(z_n) = sum_w a_w X_w e^{-i k_w z_n} + sum_m c_m v_m rho_m^{n+1/2}``
       (including ``sinc^p`` and ``(sinh(gamma_m dz/2)/(gamma_m dz/2))^{2p}`` factors), eliminating
       the evanescent wall modes ``c_m`` yields the exact coupled linear system for ``[a_exact; c_exact]``.
       The production first-order Born approximation ``a_0 + da_prod`` agrees with ``a_exact``
       within the derived second-order Born remainder ``2 * eps^2`` for top-4 wave amplitudes
       and ``rel_wall_effect * 2 * eps`` for ``beta_q`` across ``(10, 474, qm=1)``,
       ``(20, 948, qm=1)``, and ``(10, 474, qm=2)``.
    """
    print(flush=True)
    eps_mach = float(np.finfo(np.float64).eps)

    def _check_discrete_halfline_on_grid(nx: int, nz: int, qm: float) -> None:
        params = _cached_params(qm)
        dz = TUNNEL_LENGTH_M / nz
        dt, _ = _cached_integer_period_dt(nx, nz)
        dep = compute_discrete_dc_depression(
            nx, params, return_details=True, shape=PARTICLE_SHAPE
        )
        u_l, g_l = dep["u_l"], dep["gamma_l"]
        u0 = params["u0"]
        om_dt = (2.0 / dt) * np.sin(0.5 * params["omega"] * dt)
        sig_vec = params["omega_p0"] ** 2 * (u0 / u_l) / (g_l**3 * u_l**2)
        be_vec = om_dt / u_l
        u_bar = float(u0 * (1.0 - float(dep["delta_disc"])))
        k_ref = float(params["omega"] / u_bar)
        op0 = compute_transverse_modal_operator(
            "particles", nx, nz, params, k_val=k_ref, shape=PARTICLE_SHAPE
        )
        d_vec = op0["d_vec"]
        s_mat = op0["S"]
        d_mat = np.diag(d_vec)
        n_lines = len(be_vec)
        p_basis = _even_subspace_basis(n_lines)
        h_sub = p_basis.shape[1]
        c_scale = (n_lines**2 * op0["dx"]) / (2.0 * BEAM_HALF_THICKNESS_M)

        lam, e_vecs = np.linalg.eigh(-op0["lap"])
        tau_y2 = lam - (om_dt / c_light) ** 2
        rho = np.array(
            [
                float(
                    1.0 + 0.5 * t * dz**2 - np.sqrt((1.0 + 0.5 * t * dz**2) ** 2 - 1.0)
                )
                for t in tau_y2
            ],
            dtype=np.float64,
        )
        gamma_m = -np.log(rho) / dz

        # Part 1: Tridiagonal half-line operator L_m with Neumann ghost psi_{-1} = psi_0 (image j -> -j - 1)
        if (nx, nz, qm) == (10, 474, 1.0):
            n_tridiag = 240
            n_check = 60
            idx_grid = np.arange(n_check, dtype=np.float64)
            max_green_err = 0.0
            for m_idx in range(len(lam)):
                t_m = float(tau_y2[m_idx])
                r_m = float(rho[m_idx])
                l_m = float(lam[m_idx])
                diag_main = np.full(n_tridiag, 2.0 / (dz**2) + t_m, dtype=np.float64)
                diag_main[0] = 1.0 / (dz**2) + t_m  # Neumann ghost psi_{-1} = psi_0
                diag_off = np.full(n_tridiag - 1, -1.0 / (dz**2), dtype=np.float64)
                l_half = (
                    np.diag(diag_main)
                    + np.diag(diag_off, k=1)
                    + np.diag(diag_off, k=-1)
                )
                g_num = np.linalg.inv(l_half)[:n_check, :n_check] * l_m
                n_mat = idx_grid[:, None]
                j_mat = idx_grid[None, :]
                g_img = (l_m * r_m * (dz**2) / (1.0 - r_m**2)) * (
                    (r_m ** np.abs(n_mat - j_mat)) + (r_m ** (n_mat + j_mat + 1.0))
                )
                err_m = float(np.max(np.abs(g_num - g_img)) / np.max(np.abs(g_img)))
                max_green_err = max(max_green_err, err_m)
            tol_green = 100.0 * eps_mach
            print(
                f"DISCRETE_GREEN_V1 nx={nx} nz={nz} qm={qm:.1f} "
                f"max_green_rel_err={max_green_err:.3e} tol_green={tol_green:.3e}",
                flush=True,
            )
            assert max_green_err <= tol_green, (
                f"Discrete half-line Green function error {max_green_err:.3e} > {tol_green:.3e}"
            )

        ms_none = scw.multistream_waves(nx, nz, params, wall_image="none")
        ks = ms_none["k_waves"]
        x_waves = ms_none["x_waves"]
        a0 = ms_none["a_waves"]

        # Solve tridiagonal half-line operator L_m^half on [0, n_half) with Neumann ghost psi_{-1} = psi_0
        # (image j -> -j - 1) and full-line operator L_m^full on [-n_half, n_half) to extract the
        # discrete half-line wall discrepancy kernel g_wall_row0[m, j] = G_m^half(0, j) - G_m^bulk(0, j)
        # without using the closed-form geometric series:
        n_half = 400
        g_wall_row0 = np.zeros((len(lam), n_half), dtype=np.float64)
        for m_idx in range(len(lam)):
            t_m = float(tau_y2[m_idx])
            l_m = float(lam[m_idx])
            d_main = np.full(n_half, 2.0 / (dz**2) + t_m, dtype=np.float64)
            d_main[0] = 1.0 / (dz**2) + t_m  # Neumann ghost psi_{-1} = psi_0
            d_off = np.full(n_half - 1, -1.0 / (dz**2), dtype=np.float64)
            l_half = np.diag(d_main) + np.diag(d_off, k=1) + np.diag(d_off, k=-1)
            e0 = np.zeros(n_half, dtype=np.float64)
            e0[0] = l_m
            g_half_0 = np.linalg.solve(l_half, e0)

            n_full = 2 * n_half
            d_full = np.full(n_full, 2.0 / (dz**2) + t_m, dtype=np.float64)
            d_off_f = np.full(n_full - 1, -1.0 / (dz**2), dtype=np.float64)
            l_full = np.diag(d_full) + np.diag(d_off_f, k=1) + np.diag(d_off_f, k=-1)
            e_mid = np.zeros(n_full, dtype=np.float64)
            e_mid[n_half] = l_m
            g_full_mid = np.linalg.solve(l_full, e_mid)
            g_bulk_0 = g_full_mid[n_half:]
            g_wall_row0[m_idx, :] = g_half_0 - g_bulk_0

        p_int = int(PARTICLE_SHAPE)
        sz_k = np.sinc(0.5 * ks * dz / np.pi) ** p_int
        gq_vec = (np.sinh(0.5 * gamma_m * dz) / (0.5 * gamma_m * dz)) ** p_int
        p_left = s_mat.T @ e_vecs
        p_right = e_vecs.T @ s_mat @ d_mat
        pr = p_basis.T
        px = pr @ x_waves
        v_e = np.vstack([px, px * (-1j * ks)[None, :]])
        r_mat = (
            np.linalg.solve(v_e, np.vstack([np.zeros((h_sub, h_sub)), np.eye(h_sub)]))
            @ pr
        )

        j_cells = np.arange(n_half, dtype=np.float64)
        diff_waves = np.exp(-1j * np.outer(j_cells + 0.5, ks * dz)) - np.exp(
            1j * np.outer(j_cells + 0.5, ks * dz)
        )
        k_wall_tridiag = np.zeros((len(ks), len(ks)), dtype=np.complex128)
        for m_idx in range(len(lam)):
            amp_m = ((g_wall_row0[m_idx] / np.sqrt(rho[m_idx])) @ diff_waves) * sz_k
            src_row = (p_right[m_idx] @ x_waves) * amp_m
            col_vec = (
                r_mat @ (sig_vec * c_scale * gq_vec[m_idx] * p_left[:, m_idx])
            ) / (gamma_m[m_idx] - 1j * ks)
            k_wall_tridiag += np.outer(col_vec, src_row)

        da_from_green_sum = k_wall_tridiag @ a0
        tol_green_sum = float((rho[0] ** n_half) / (1.0 - rho[0]) + 1000.0 * eps_mach)

        def _assert_green_sum_identity() -> None:
            da_prod = scw._multistream_wall_correction(
                op0=op0,
                dz=dz,
                dt=dt,
                omega=params["omega"],
                sig_vec=sig_vec,
                d_vec=d_vec,
                k_waves=ks,
                x_waves=x_waves,
                a_waves=a0,
                p_basis=p_basis,
                shape=PARTICLE_SHAPE,
                use_omega_f=True,
                include_sz=True,
                discrete=True,
            )
            rel_sum_err = float(
                np.linalg.norm(da_prod - da_from_green_sum)
                / np.linalg.norm(da_from_green_sum)
            )
            print(
                f"DISCRETE_GREEN_SUM nx={nx} nz={nz} qm={qm:.1f} "
                f"rel_sum_err={rel_sum_err:.3e} tol_green_sum={tol_green_sum:.3e}",
                flush=True,
            )
            assert rel_sum_err <= tol_green_sum, (
                f"Production wall term vs discrete Green sum {rel_sum_err:.3e} > {tol_green_sum:.3e}"
            )

        _assert_green_sum_identity()

        # Part 2: Independent coupled discrete half-line BVP for [a_exact (2*h_sub); c_exact (n_modes)]
        # using the beam companion operator A_beam (without using r_mat or 1/(qm - i*k) from
        # _multistream_wall_correction) and the tridiagonal half-line Green kernel g_wall_row0:
        sz_ref_sq = float(np.sinc(0.5 * k_ref * dz / np.pi) ** (2 * p_int))
        m_ref_sub = pr @ ((sz_ref_sq * sig_vec)[:, None] * op0["G"]) @ p_basis
        be_sub = pr @ np.diag(be_vec) @ p_basis
        a0_sub = be_sub @ be_sub - m_ref_sub
        a_beam = np.block(
            [
                [np.zeros((h_sub, h_sub)), np.eye(h_sub)],
                [a0_sub, -2j * be_sub],
            ]
        )
        n_modes = len(lam)
        w_part = np.zeros((2 * h_sub, n_modes), dtype=np.complex128)
        for m_idx in range(n_modes):
            rhs_m = np.concatenate(
                [
                    np.zeros(h_sub, dtype=np.complex128),
                    pr @ (sig_vec * c_scale * gq_vec[m_idx] * p_left[:, m_idx]),
                ]
            )
            w_part[:, m_idx] = -np.linalg.solve(
                gamma_m[m_idx] * np.eye(2 * h_sub) + a_beam, rhs_m
            )

        c_of_a = np.zeros((n_modes, len(ks)), dtype=np.complex128)
        for m_idx in range(n_modes):
            amp_m = ((g_wall_row0[m_idx] / np.sqrt(rho[m_idx])) @ diff_waves) * sz_k
            c_of_a[m_idx, :] = (p_right[m_idx] @ x_waves) * amp_m

        dk_mat = ks[None, :] - be_vec[:, None]
        w_top = w_part[:h_sub, :]
        w_bot = w_part[h_sub:, :] + 1j * (be_sub @ w_top)
        m_bvp = np.block(
            [
                [pr @ x_waves, w_top],
                [pr @ (x_waves * (-1j * dk_mat)), w_bot],
                [-c_of_a, np.eye(n_modes, dtype=np.complex128)],
            ]
        )
        e_dc = np.exp(-1j * ms_none["theta_l"])
        rhs_bvp = np.concatenate(
            [pr @ e_dc, np.zeros(h_sub + n_modes, dtype=np.complex128)]
        )
        sol_bvp = np.linalg.solve(m_bvp, rhs_bvp)
        a_exact = sol_bvp[: 2 * h_sub]

        o0 = scw._multistream_observable_weights(
            a_waves=a0,
            d_vec=d_vec,
            x_waves=x_waves,
            k_waves=ks,
            dz=dz,
            shape=PARTICLE_SHAPE,
            use_obs_sinc=True,
        )
        o_exact = scw._multistream_observable_weights(
            a_waves=a_exact,
            d_vec=d_vec,
            x_waves=x_waves,
            k_waves=ks,
            dz=dz,
            shape=PARTICLE_SHAPE,
            use_obs_sinc=True,
        )

        z_eval = (np.arange(nz, dtype=np.float64) + 0.5) * dz
        z_min, z_max = fit_window(params)
        guess = float(_cached_modal_details(nx, nz, qm)["bq_modes"][0])
        i1_0 = params["I1_0"] * np.exp(-1j * np.outer(z_eval, ks)) @ o0
        i1_exact = params["I1_0"] * np.exp(-1j * np.outer(z_eval, ks)) @ o_exact
        bq_0 = float(
            fit_beta_q_from_envelope(z_eval, i1_0, guess, z_min=z_min, z_max=z_max)[
                "beta_q"
            ]
        )
        bq_exact = float(
            fit_beta_q_from_envelope(z_eval, i1_exact, guess, z_min=z_min, z_max=z_max)[
                "beta_q"
            ]
        )
        rel_wall_effect = abs(bq_exact / bq_0 - 1.0)
        top4 = np.argsort(-np.abs(o0))[:4]

        da_ref = scw._multistream_wall_correction(
            op0=op0,
            dz=dz,
            dt=dt,
            omega=params["omega"],
            sig_vec=sig_vec,
            d_vec=d_vec,
            k_waves=ks,
            x_waves=x_waves,
            a_waves=a0,
            p_basis=p_basis,
            shape=PARTICLE_SHAPE,
            use_omega_f=True,
            include_sz=True,
            discrete=True,
        )
        eps_disc = float(max(abs(da_ref[j] / a0[j]) for j in top4))
        tol_wave_disc = float(2.0 * (eps_disc**2))
        tol_bq_disc = float(rel_wall_effect * 2.0 * eps_disc)

        def _assert_prod_born_vs_exact_half_line() -> None:
            da_prod = scw._multistream_wall_correction(
                op0=op0,
                dz=dz,
                dt=dt,
                omega=params["omega"],
                sig_vec=sig_vec,
                d_vec=d_vec,
                k_waves=ks,
                x_waves=x_waves,
                a_waves=a0,
                p_basis=p_basis,
                shape=PARTICLE_SHAPE,
                use_omega_f=True,
                include_sz=True,
                discrete=True,
            )
            o_born = scw._multistream_observable_weights(
                a_waves=a0 + da_prod,
                d_vec=d_vec,
                x_waves=x_waves,
                k_waves=ks,
                dz=dz,
                shape=PARTICLE_SHAPE,
                use_obs_sinc=True,
            )
            i1_born = params["I1_0"] * np.exp(-1j * np.outer(z_eval, ks)) @ o_born
            bq_born = float(
                fit_beta_q_from_envelope(
                    z_eval, i1_born, guess, z_min=z_min, z_max=z_max
                )["beta_q"]
            )
            max_top4_o_rel = float(
                max(abs(o_born[j] - o_exact[j]) / abs(o0[j]) for j in top4)
            )
            rel_bq_err = abs(bq_born / bq_exact - 1.0)
            print(
                f"DISCRETE_HALFLINE_SOLVE nx={nx} nz={nz} qm={qm:.1f} eps={eps_disc:.6f} "
                f"max_top4_o_rel={max_top4_o_rel:.6e} tol_2eps2={tol_wave_disc:.6e} "
                f"rel_bq_err={rel_bq_err:.3e} tol_bq_2eps={tol_bq_disc:.3e} "
                f"rel_wall_effect={rel_wall_effect:.3e}",
                flush=True,
            )
            assert max_top4_o_rel <= tol_wave_disc, (
                f"Discrete half-line top-4 wave error {max_top4_o_rel:.6e} > 2*eps^2={tol_wave_disc:.6e}"
            )
            assert rel_bq_err <= tol_bq_disc, (
                f"Discrete half-line beta_q error {rel_bq_err:.3e} > rel_wall_effect*2*eps={tol_bq_disc:.3e}"
            )

        _assert_prod_born_vs_exact_half_line()

        if (nx, nz, qm) == (10, 474, 1.0):
            # Also check on (10, 474, qm=1) that |o_born - o_exact| / |o_exact - o_0| <= 2 * eps^2
            o_ref = scw._multistream_observable_weights(
                a_waves=a0 + da_ref,
                d_vec=d_vec,
                x_waves=x_waves,
                k_waves=ks,
                dz=dz,
                shape=PARTICLE_SHAPE,
                use_obs_sinc=True,
            )
            max_top4_shift_rel = float(
                max(abs(o_ref[j] - o_exact[j]) / abs(o_exact[j] - o0[j]) for j in top4)
            )
            print(
                f"DISCRETE_HALFLINE_SHIFT_REL nx={nx} nz={nz} qm={qm:.1f} "
                f"max_top4_shift_rel={max_top4_shift_rel:.6f} tol_2eps2={tol_wave_disc:.6f}",
                flush=True,
            )
            assert max_top4_shift_rel <= tol_wave_disc

            # Monkeypatched negative control 1: flipping sign of _multistream_wall_correction
            orig_wall = scw._multistream_wall_correction
            with monkeypatch.context() as mp:
                mp.setattr(
                    scw,
                    "_multistream_wall_correction",
                    lambda *args, **kwargs: -orig_wall(*args, **kwargs),
                )
                with pytest.raises(
                    AssertionError, match="Discrete half-line top-4 wave error"
                ):
                    _assert_prod_born_vs_exact_half_line()

            # Monkeypatched negative control 2: dropping include_sz (or discrete=False) in _multistream_wall_correction
            # fails the discrete Green function sum identity check
            with monkeypatch.context() as mp:
                mp.setattr(
                    scw,
                    "_multistream_wall_correction",
                    lambda *args, **kwargs: orig_wall(
                        *args, **{**kwargs, "include_sz": False}
                    ),
                )
                with pytest.raises(
                    AssertionError, match="Production wall term vs discrete Green sum"
                ):
                    _assert_green_sum_identity()

    for nx, nz, qm in ((10, 474, 1.0), (20, 948, 1.0), (10, 474, 2.0)):
        _check_discrete_halfline_on_grid(nx, nz, qm)
