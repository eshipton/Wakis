"""Falsifier F-b1 for the M1 residual: velocity-shear term vs mode-3 leakage.

Compares the first-order discrete modal model against the exact per-line
cold-stream quadratic eigenvalue problem (QEP) across three cases
``(nx, nz, qm) in [(10, 474, 1), (20, 948, 1), (10, 474, 2)]`` using a
four-wave variable-projection fit on ``fit_window(params)``.

Wave amplitude normalisation
----------------------------
The modal prediction synthesises the fundamental axial current phasor as::

    I1(z) = I1_0 * sum_j c_j * cos(beta_q,j * z) * exp(-i * k_c,j * z)
          = sum_j (I1_0 * c_j) * 0.5 * [
                exp(-i * (k_c,j - beta_q,j) * z)
              + exp(-i * (k_c,j + beta_q,j) * z)
            ]

Writing the four-wave expansion as::

    I1(z) = sum_{n in {1-, 1+, 3-, 3+}} a_n * exp(-i * k_n * z)
          = sum_{n in {1-, 1+, 3-, 3+}} A_n * [0.5 * exp(-i * k_n * z)]

where ``a_n`` are the raw exponential coefficients in ``sum_n a_n exp(-i k_n z)``
and ``A_n = 2 * a_n`` are the cosine-envelope-normalised complex wave amplitudes,
an unperturbed mode ``j`` has ``|A_{j+}| = |A_{j-}| = I1_0 * |c_j|``. Thus::

    c3_meas = 0.5 * (|A3+| + |A3-|) / I1_0 = (|a3+| + |a3-|) / I1_0

satisfies ``c3_meas = |c_modes[mode3]| = c3_model`` (ratio ``1.0``) on a pure
two-mode synthetic ``I1(z)``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "examples"))

import _space_charge_waves_common as s

CASES: tuple[tuple[int, int, int], ...] = (
    (10, 474, 1),
    (20, 948, 1),
    (10, 474, 2),
)
SELF_CHECK_BETA1_TOL = 1.0e-6
SELF_CHECK_C3_TOL = 1.0e-4


def compute_model_and_exact_roots(
    nx: int,
    nz: int,
    qm: float,
    shape: int = s.PARTICLE_SHAPE,
) -> dict[str, Any]:
    """Compute first-order modal roots and exact per-line velocity-shear QEP roots.

    Reuses the exact formulas ``sig``, ``Om``, ``be``, ``Msym(k)``, and ``root()``
    from ``studies/p5b-shear/qep_ref.py``.

    Parameters
    ----------
    nx, nz : int
        Transverse and axial cell counts [-].
    qm : float
        Charge-to-mass ratio multiplier ``qm_scale`` [-].
    shape : int, optional
        Particle B-spline shape order ``p`` [-]. Default is ``PARTICLE_SHAPE``.

    Returns
    -------
    dict of str to Any
        Dictionary containing ``params``, ``det``, ``idx1``, ``idx3``, ``c_modes``,
        ``bqm``, ``kcm``, ``bqx``, ``kcx``, ``f0_2wave``, and ``fx_2wave``.
    """
    p = int(shape)
    P = s.compute_analytic_parameters(float(qm))
    dz = s.TUNNEL_LENGTH_M / nz
    dt, _ = s._cached_integer_period_dt(nx, nz)
    dep = s.compute_discrete_dc_depression(nx, P, return_details=True, shape=p)
    u_l = dep["u_l"]
    g_l = dep["gamma_l"]
    u0 = P["u0"]
    sig = P["omega_p0"] ** 2 * (u0 / u_l) / (g_l**3 * u_l**2)
    Om = (2.0 / dt) * np.sin(0.5 * P["omega"] * dt)
    be = Om / u_l

    def Msym(kk: float) -> np.ndarray:
        op = s.compute_transverse_modal_operator(
            "particles", nx, nz, P, k_val=kk, shape=p
        )
        d = op["d_vec"]
        sz = np.sinc(0.5 * kk * dz / np.pi) ** (2 * p)
        sl = np.sqrt(d / sig)
        sr = np.sqrt(sig / d)
        m = (sl[:, None] * (sz * sig)[:, None] * op["G"]) * sr[None, :]
        return 0.5 * (m + m.T)

    det = s.discrete_modal_prediction(
        "particles", nx, nz, P, return_details=True, shape=p
    )

    def F(kk: float) -> float:
        return float(np.linalg.slogdet(np.diag((kk - be) ** 2) - Msym(kk))[0])

    def root(k0: float, w: float) -> float:
        ks = np.linspace(k0 - w, k0 + w, 121)
        fs = np.array([F(float(x)) for x in ks])
        idx = np.where(np.diff(fs) != 0)[0]
        return float(
            min(
                [brentq(F, float(ks[i]), float(ks[i + 1]), xtol=1e-13) for i in idx],
                key=lambda q: abs(q - k0),
            )
        )

    c = det["c_modes"]
    bqm = det["bq_modes"].copy()
    kcm = det["kc_modes"].copy()
    bqx = bqm.copy()
    kcx = kcm.copy()
    for j in np.where(np.abs(c) > 1e-3)[0]:
        w = min(2.0, 0.3 * bqm[j])
        kp = root(float(kcm[j] + bqm[j]), float(w))
        km = root(float(kcm[j] - bqm[j]), float(w))
        bqx[j] = 0.5 * (kp - km)
        kcx[j] = 0.5 * (kp + km)

    top2 = np.argsort(np.abs(c))[::-1][:2]
    idx1, idx3 = int(top2[0]), int(top2[1])

    z = det["z_coords"]
    zmin, zmax = s.fit_window(P)
    i1_syn_first = P["I1_0"] * np.sum(
        c[:, None] * np.cos(np.outer(bqm, z)) * np.exp(-1j * np.outer(kcm, z)),
        axis=0,
    )
    i1_syn_exact = P["I1_0"] * np.sum(
        c[:, None] * np.cos(np.outer(bqx, z)) * np.exp(-1j * np.outer(kcx, z)),
        axis=0,
    )
    f0_2wave = float(
        s.fit_beta_q_from_envelope(
            z, i1_syn_first, float(bqm[0]), z_min=zmin, z_max=zmax
        )["beta_q"]
    )
    fx_2wave = float(
        s.fit_beta_q_from_envelope(
            z, i1_syn_exact, float(bqx[0]), z_min=zmin, z_max=zmax
        )["beta_q"]
    )

    c_2mode = np.zeros_like(c)
    c_2mode[[idx1, idx3]] = c[[idx1, idx3]]
    i1_syn_first_2mode = P["I1_0"] * np.sum(
        c_2mode[:, None] * np.cos(np.outer(bqm, z)) * np.exp(-1j * np.outer(kcm, z)),
        axis=0,
    )
    i1_syn_exact_2mode = P["I1_0"] * np.sum(
        c_2mode[:, None] * np.cos(np.outer(bqx, z)) * np.exp(-1j * np.outer(kcx, z)),
        axis=0,
    )

    return {
        "params": P,
        "det": det,
        "idx1": idx1,
        "idx3": idx3,
        "c_modes": c,
        "bqm": bqm,
        "kcm": kcm,
        "bqx": bqx,
        "kcx": kcx,
        "z_coords": z,
        "z_min": float(zmin),
        "z_max": float(zmax),
        "i1_syn_first": i1_syn_first,
        "i1_syn_exact": i1_syn_exact,
        "i1_syn_first_2mode": i1_syn_first_2mode,
        "i1_syn_exact_2mode": i1_syn_exact_2mode,
        "f0_2wave": f0_2wave,
        "fx_2wave": fx_2wave,
    }


def fit_four_wave_modes(
    z_coords: np.ndarray,
    i1_phasor: np.ndarray,
    k3_minus: float,
    k3_plus: float,
    kc1_init: float,
    bq1_init: float,
    i1_0: float,
    c3_model: float,
    z_min: float,
    z_max: float,
    tail_unit_window: np.ndarray | None = None,
    amp_norm_factor: float = 2.0,
) -> dict[str, Any]:
    """Fit the four-wave variable-projection model on ``[z_min, z_max]``.

    Models ``I1(z) = sum_{n=1}^4 a_n exp(-i k_n z) = sum_{n=1}^4 A_n [0.5 exp(-i k_n z)]``
    with ``k_{1-} = kc1 - bq1``, ``k_{1+} = kc1 + bq1`` free and ``(k_{3-}, k_{3+})``
    fixed. For given ``(kc1, bq1)``, the four complex amplitudes follow from a
    linear least-squares solve (`numpy.linalg.lstsq`), and the real/imaginary
    residual vector is minimised via `scipy.optimize.least_squares` starting at
    ``(kc1_init, bq1_init)``.

    Parameters
    ----------
    z_coords : np.ndarray
        Axial edge midpoint coordinates [m].
    i1_phasor : np.ndarray
        Complex axial current phasor ``I1(z)`` [A].
    k3_minus, k3_plus : float
        Fixed fast and slow wavenumbers [rad/m] for mode 3.
    kc1_init, bq1_init : float
        Initial guesses [rad/m] for mode-1 ``k_c`` and ``beta_q``.
    i1_0 : float
        Nominal entrance modulation current amplitude ``I1_0 = m * I0`` [A].
    c3_model : float
        Modal excitation weight ``|c_modes[mode3]|`` [-].
    z_min, z_max : float
        Fit window bounds [m].
    tail_unit_window : np.ndarray, optional
        Optional higher-mode (``j >= 4``) unit tail ``sum_{j>=4} (c_j/c_1) cos(bq_j z) exp(-i kc_j z)``
        evaluated on the fit window and locked to the mode-1 carrier amplitude [-].
    amp_norm_factor : float, optional
        Conversion factor ``A_n = amp_norm_factor * a_n`` from raw exponential
        coefficients ``a_n`` to standing-mode wave amplitudes ``A_n`` [-].
        Default is ``2.0`` (from ``cos(bq * z) = 0.5 * (exp(i*bq*z) + exp(-i*bq*z))``).

    Returns
    -------
    dict of str to Any
        Fitted ``beta1``, ``kc1``, ``k1_minus``, ``k1_plus``, ``A_waves``,
        ``c3_meas``, ``c3_ratio``, and relative residual norm ``resid_rel``.
    """
    mask = (z_coords >= z_min) & (z_coords <= z_max)
    zm = np.asarray(z_coords[mask], dtype=np.float64)
    ym = np.asarray(i1_phasor[mask], dtype=np.complex128)
    ym_norm = float(np.linalg.norm(ym))

    u3_minus = np.exp(-1j * float(k3_minus) * zm)
    u3_plus = np.exp(-1j * float(k3_plus) * zm)
    tail = (
        np.zeros_like(zm, dtype=np.complex128)
        if tail_unit_window is None
        else np.asarray(tail_unit_window, dtype=np.complex128)
    )

    def _design(kc_val: float, bq_val: float) -> np.ndarray:
        bq_pos = abs(float(bq_val))
        k1_m = float(kc_val) - bq_pos
        k1_p = float(kc_val) + bq_pos
        return np.column_stack(
            [
                np.exp(-1j * k1_m * zm) + tail,
                np.exp(-1j * k1_p * zm) + tail,
                u3_minus,
                u3_plus,
            ]
        )

    def _resid_vec(x: np.ndarray) -> np.ndarray:
        design = _design(float(x[0]), float(x[1]))
        a_coef, _, _, _ = np.linalg.lstsq(design, ym, rcond=None)
        res_c = ym - design @ a_coef
        return np.concatenate([res_c.real, res_c.imag])

    opt = least_squares(
        _resid_vec,
        x0=np.array([float(kc1_init), float(bq1_init)], dtype=np.float64),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    kc1_fit = float(opt.x[0])
    beta1_fit = abs(float(opt.x[1]))
    design_opt = _design(kc1_fit, beta1_fit)
    a_coef, _, _, _ = np.linalg.lstsq(design_opt, ym, rcond=None)
    res_c = ym - design_opt @ a_coef
    resid_rel = float(np.linalg.norm(res_c) / ym_norm)

    A_waves = float(amp_norm_factor) * a_coef
    c3_meas = float(0.5 * (abs(A_waves[2]) + abs(A_waves[3])) / i1_0)
    c3_ratio = float(c3_meas / abs(c3_model))

    return {
        "beta1": beta1_fit,
        "kc1": kc1_fit,
        "k1_minus": float(kc1_fit - beta1_fit),
        "k1_plus": float(kc1_fit + beta1_fit),
        "a_raw": a_coef,
        "A_waves": A_waves,
        "c3_meas": c3_meas,
        "c3_model": float(abs(c3_model)),
        "c3_ratio": c3_ratio,
        "resid_rel": resid_rel,
    }


def run_self_check_for_case(
    case_idx: int,
    info: dict[str, Any],
    amp_norm_factor: float = 2.0,
) -> dict[str, Any]:
    """Run Step 5 synthetic self-checks for a single ``(nx, nz, qm)`` case.

    Evaluates the identical 4-wave fitter on:
    1. Two-mode synthetic ``I1(z)`` (modes ``idx1`` and ``idx3``) and
       ``j >= 4``-tail-corrected all-mode synthetic ``I1(z)`` (both first-order
       and exact-shear), which must recover ``beta_1`` to ``rel < 1e-6`` and
       ``|c3|`` ratio to ``< 1e-4``.
    2. Unadjusted all-mode synthetic ``I1(z)`` synthesised literally as in
       ``qep_ref.py``'s ``fit()`` (including the ``j >= 4`` modes ``c[4], c[6], ...``),
       quantifying the systematic finite-window leakage of mode ``j = 4``
       (physical mode 5) into the 4-wave ``(mode 1, mode 3)`` basis.
    """
    nx, nz, qm = CASES[case_idx]
    P = info["params"]
    idx1 = info["idx1"]
    idx3 = info["idx3"]
    c = info["c_modes"]
    bqm = info["bqm"]
    kcm = info["kcm"]
    bqx = info["bqx"]
    kcx = info["kcx"]
    z = info["z_coords"]
    zmin = info["z_min"]
    zmax = info["z_max"]
    i1_0 = float(P["I1_0"])
    c3_model = float(abs(c[idx3]))

    mask = (z >= zmin) & (z <= zmax)
    zm = z[mask]
    c_hi = c.copy()
    c_hi[[idx1, idx3]] = 0.0
    tail_unit = np.sum(
        (c_hi / c[idx1])[:, None]
        * np.cos(np.outer(bqm, zm))
        * np.exp(-1j * np.outer(kcm, zm)),
        axis=0,
    )

    k3f_first = float(kcm[idx3] - bqm[idx3])
    k3s_first = float(kcm[idx3] + bqm[idx3])
    k3f_exact = float(kcx[idx3] - bqx[idx3])
    k3s_exact = float(kcx[idx3] + bqx[idx3])
    kc1_init = float(kcm[idx1])
    bq1_init = float(bqm[idx1])

    # 1. Pure 2-mode synthetic I1 (modes idx1 and idx3)
    fit_2m_first = fit_four_wave_modes(
        z,
        info["i1_syn_first_2mode"],
        k3f_first,
        k3s_first,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
        amp_norm_factor=amp_norm_factor,
    )
    fit_2m_exact = fit_four_wave_modes(
        z,
        info["i1_syn_exact_2mode"],
        k3f_exact,
        k3s_exact,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
        amp_norm_factor=amp_norm_factor,
    )

    # 2. All-mode synthetic I1 (as in reference fit()) with j >= 4 tail locked to mode 1
    fit_all_tail_first = fit_four_wave_modes(
        z,
        info["i1_syn_first"],
        k3f_first,
        k3s_first,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
        tail_unit_window=tail_unit,
        amp_norm_factor=amp_norm_factor,
    )
    fit_all_tail_exact = fit_four_wave_modes(
        z,
        info["i1_syn_exact"],
        k3f_exact,
        k3s_exact,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
        tail_unit_window=tail_unit,
        amp_norm_factor=amp_norm_factor,
    )

    # 3. Unadjusted 4-wave fit applied directly to all-mode synthetic I1 (reference fit())
    fit_all_raw_first_m3exact = fit_four_wave_modes(
        z,
        info["i1_syn_first"],
        k3f_exact,
        k3s_exact,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
        amp_norm_factor=amp_norm_factor,
    )
    fit_all_raw_first_m3first = fit_four_wave_modes(
        z,
        info["i1_syn_first"],
        k3f_first,
        k3s_first,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
        amp_norm_factor=amp_norm_factor,
    )
    fit_all_raw_exact_m3exact = fit_four_wave_modes(
        z,
        info["i1_syn_exact"],
        k3f_exact,
        k3s_exact,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
        amp_norm_factor=amp_norm_factor,
    )

    err_b1_2m_first = abs(fit_2m_first["beta1"] / float(bqm[idx1]) - 1.0)
    err_c3_2m_first = abs(fit_2m_first["c3_ratio"] - 1.0)
    err_b1_2m_exact = abs(fit_2m_exact["beta1"] / float(bqx[idx1]) - 1.0)
    err_c3_2m_exact = abs(fit_2m_exact["c3_ratio"] - 1.0)

    err_b1_tail_first = abs(fit_all_tail_first["beta1"] / float(bqm[idx1]) - 1.0)
    err_c3_tail_first = abs(fit_all_tail_first["c3_ratio"] - 1.0)
    err_b1_tail_exact = abs(fit_all_tail_exact["beta1"] / float(bqx[idx1]) - 1.0)
    err_c3_tail_exact = abs(fit_all_tail_exact["c3_ratio"] - 1.0)

    err_b1_raw_first = abs(fit_all_raw_first_m3exact["beta1"] / float(bqm[idx1]) - 1.0)
    err_c3_raw_first = abs(fit_all_raw_first_m3exact["c3_ratio"] - 1.0)
    err_b1_raw_first_m3f = abs(
        fit_all_raw_first_m3first["beta1"] / float(bqm[idx1]) - 1.0
    )
    err_c3_raw_first_m3f = abs(fit_all_raw_first_m3first["c3_ratio"] - 1.0)
    err_b1_raw_exact = abs(fit_all_raw_exact_m3exact["beta1"] / float(bqx[idx1]) - 1.0)
    err_c3_raw_exact = abs(fit_all_raw_exact_m3exact["c3_ratio"] - 1.0)

    pass_2m = bool(
        max(err_b1_2m_first, err_b1_2m_exact, err_b1_tail_first, err_b1_tail_exact)
        < SELF_CHECK_BETA1_TOL
        and max(err_c3_2m_first, err_c3_2m_exact, err_c3_tail_first, err_c3_tail_exact)
        < SELF_CHECK_C3_TOL
    )
    pass_raw_all_modes = bool(
        max(err_b1_raw_first, err_b1_raw_exact) < SELF_CHECK_BETA1_TOL
        and max(err_c3_raw_first, err_c3_raw_exact) < SELF_CHECK_C3_TOL
    )

    summary = {
        "case": case_idx,
        "nx": nx,
        "nz": nz,
        "qm": float(qm),
        "mode1_idx": idx1,
        "mode3_idx": idx3,
        "c1_model": float(c[idx1]),
        "c3_model": c3_model,
        "c_higher_sum_over_c3": float(np.sum(np.abs(c_hi)) / c3_model),
        "two_mode_and_tail_corrected": {
            "err_beta1_2m_first": float(err_b1_2m_first),
            "err_c3_2m_first": float(err_c3_2m_first),
            "err_beta1_2m_exact": float(err_b1_2m_exact),
            "err_c3_2m_exact": float(err_c3_2m_exact),
            "err_beta1_tail_first": float(err_b1_tail_first),
            "err_c3_tail_first": float(err_c3_tail_first),
            "err_beta1_tail_exact": float(err_b1_tail_exact),
            "err_c3_tail_exact": float(err_c3_tail_exact),
            "pass": pass_2m,
        },
        "raw_all_modes_ref_fit": {
            "rel_beta1_syn_first_m3exact": float(
                fit_all_raw_first_m3exact["beta1"] / float(bqm[idx1]) - 1.0
            ),
            "c3_ratio_syn_first_m3exact": float(fit_all_raw_first_m3exact["c3_ratio"]),
            "rel_beta1_syn_first_m3first": float(
                fit_all_raw_first_m3first["beta1"] / float(bqm[idx1]) - 1.0
            ),
            "c3_ratio_syn_first_m3first": float(fit_all_raw_first_m3first["c3_ratio"]),
            "rel_beta1_syn_exact_m3exact": float(
                fit_all_raw_exact_m3exact["beta1"] / float(bqx[idx1]) - 1.0
            ),
            "c3_ratio_syn_exact_m3exact": float(fit_all_raw_exact_m3exact["c3_ratio"]),
            "err_beta1_max": float(max(err_b1_raw_first, err_b1_raw_exact)),
            "err_c3_max": float(max(err_c3_raw_first, err_c3_raw_exact)),
            "err_beta1_m3first": float(err_b1_raw_first_m3f),
            "err_c3_m3first": float(err_c3_raw_first_m3f),
            "pass": pass_raw_all_modes,
        },
        "tail_unit": tail_unit,
    }
    print(
        "SELF_CHECK "
        + json.dumps({k: v for k, v in summary.items() if k != "tail_unit"}),
        flush=True,
    )
    assert pass_2m, (
        f"Self-check failed on 2-mode / tail-corrected synthetic I1 for case {case_idx} "
        f"(nx={nx}, nz={nz}, qm={qm}): "
        f"err_b1=({err_b1_2m_first:.3e}, {err_b1_2m_exact:.3e}, {err_b1_tail_first:.3e}, {err_b1_tail_exact:.3e}) vs {SELF_CHECK_BETA1_TOL:.1e}, "
        f"err_c3=({err_c3_2m_first:.3e}, {err_c3_2m_exact:.3e}, {err_c3_tail_first:.3e}, {err_c3_tail_exact:.3e}) vs {SELF_CHECK_C3_TOL:.1e}"
    )
    return summary


def run_case(case_idx: int, self_check_only: bool = False) -> dict[str, Any]:
    """Execute model roots, synthetic self-check, and (unless ``self_check_only``) particle run for ``CASES[case_idx]``."""
    nx, nz, qm = CASES[case_idx]
    info = compute_model_and_exact_roots(nx, nz, float(qm))
    idx1 = info["idx1"]
    idx3 = info["idx3"]
    c = info["c_modes"]
    bqm = info["bqm"]
    kcm = info["kcm"]
    bqx = info["bqx"]
    kcx = info["kcx"]
    P = info["params"]
    zmin = info["z_min"]
    zmax = info["z_max"]

    print(
        f"CASE {case_idx} (nx={nx}, nz={nz}, qm={qm}): "
        f"mode-1 index={idx1} (|c|={abs(c[idx1]):.6e}), "
        f"mode-3 index={idx3} (|c|={abs(c[idx3]):.6e})",
        flush=True,
    )

    sc = run_self_check_for_case(case_idx, info)
    if self_check_only:
        return sc

    # Build reference LinearizedBeam and run ParticleSolver drift exactly as in test_087 M1
    lb_res = s.cached_linearized_beam_drift(
        nx, nz, rho0_scale=1.0, m_mod=s.MODULATION_INDEX
    )
    z_coords = lb_res["z_coords"]
    part_res = s.run_particle_beam_drift(
        nx,
        nz,
        beam_ref=lb_res["beam"],
        qm_scale=float(qm),
        seed=42,
        m_mod=s.MODULATION_INDEX,
        profile="sheet_lattice",
    )
    i1_part = part_res["i1_part"]

    fit2 = s.fit_beta_q_from_envelope(
        z_coords, i1_part, P["beta_q"], z_min=zmin, z_max=zmax
    )
    beta_2wave = float(fit2["beta_q"])

    k1_fast_first = float(kcm[idx1] - bqm[idx1])
    k1_slow_first = float(kcm[idx1] + bqm[idx1])
    k3_fast_first = float(kcm[idx3] - bqm[idx3])
    k3_slow_first = float(kcm[idx3] + bqm[idx3])

    k1_fast_exact = float(kcx[idx1] - bqx[idx1])
    k1_slow_exact = float(kcx[idx1] + bqx[idx1])
    k3_fast_exact = float(kcx[idx3] - bqx[idx3])
    k3_slow_exact = float(kcx[idx3] + bqx[idx3])

    kc1_init = float(kcm[idx1])
    bq1_init = float(bqm[idx1])
    i1_0 = float(P["I1_0"])
    c3_model = float(abs(c[idx3]))

    fit4_exact_m3 = fit_four_wave_modes(
        z_coords,
        i1_part,
        k3_fast_exact,
        k3_slow_exact,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
    )
    fit4_first_m3 = fit_four_wave_modes(
        z_coords,
        i1_part,
        k3_fast_first,
        k3_slow_first,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
    )
    fit4_tail_exact_m3 = fit_four_wave_modes(
        z_coords,
        i1_part,
        k3_fast_exact,
        k3_slow_exact,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
        tail_unit_window=sc["tail_unit"],
    )
    fit4_tail_first_m3 = fit_four_wave_modes(
        z_coords,
        i1_part,
        k3_fast_first,
        k3_slow_first,
        kc1_init,
        bq1_init,
        i1_0,
        c3_model,
        zmin,
        zmax,
        tail_unit_window=sc["tail_unit"],
    )

    beta1_first = float(bqm[idx1])
    beta1_exact = float(bqx[idx1])
    beta1_fit_ex = float(fit4_exact_m3["beta1"])
    beta1_fit_fi = float(fit4_first_m3["beta1"])
    beta1_fit_tail_ex = float(fit4_tail_exact_m3["beta1"])
    beta1_fit_tail_fi = float(fit4_tail_first_m3["beta1"])

    row = {
        "case": case_idx,
        "nx": nx,
        "nz": nz,
        "qm": float(qm),
        "mode1_idx": idx1,
        "mode3_idx": idx3,
        "beta1_fit": beta1_fit_ex,
        "beta1_fit_m3_exact": beta1_fit_ex,
        "beta1_fit_m3_first": beta1_fit_fi,
        "beta_2wave": beta_2wave,
        "beta1_first": beta1_first,
        "beta1_exact": beta1_exact,
        "beta3_first": float(bqm[idx3]),
        "beta3_exact": float(bqx[idx3]),
        "beta_pred_2wave_first": info["f0_2wave"],
        "beta_pred_2wave_exact": info["fx_2wave"],
        "roots_first": {
            "k1_fast": k1_fast_first,
            "k1_slow": k1_slow_first,
            "k3_fast": k3_fast_first,
            "k3_slow": k3_slow_first,
        },
        "roots_exact": {
            "k1_fast": k1_fast_exact,
            "k1_slow": k1_slow_exact,
            "k3_fast": k3_fast_exact,
            "k3_slow": k3_slow_exact,
        },
        "rel_exact": float(beta1_fit_ex / beta1_exact - 1.0),
        "rel_first": float(beta1_fit_ex / beta1_first - 1.0),
        "rel_exact_m3_first": float(beta1_fit_fi / beta1_exact - 1.0),
        "rel_first_m3_first": float(beta1_fit_fi / beta1_first - 1.0),
        "c3_meas": float(fit4_exact_m3["c3_meas"]),
        "c3_meas_m3_first": float(fit4_first_m3["c3_meas"]),
        "c3_model": c3_model,
        "c3_ratio": float(fit4_exact_m3["c3_ratio"]),
        "c3_ratio_m3_exact": float(fit4_exact_m3["c3_ratio"]),
        "c3_ratio_m3_first": float(fit4_first_m3["c3_ratio"]),
        "resid_norm_rel": float(fit4_exact_m3["resid_rel"]),
        "resid_norm_rel_m3_exact": float(fit4_exact_m3["resid_rel"]),
        "resid_norm_rel_m3_first": float(fit4_first_m3["resid_rel"]),
        "tail_corrected": {
            "beta1_fit_m3_exact": beta1_fit_tail_ex,
            "beta1_fit_m3_first": beta1_fit_tail_fi,
            "rel_exact": float(beta1_fit_tail_ex / beta1_exact - 1.0),
            "rel_first": float(beta1_fit_tail_ex / beta1_first - 1.0),
            "rel_exact_m3_first": float(beta1_fit_tail_fi / beta1_exact - 1.0),
            "rel_first_m3_first": float(beta1_fit_tail_fi / beta1_first - 1.0),
            "c3_ratio_m3_exact": float(fit4_tail_exact_m3["c3_ratio"]),
            "c3_ratio_m3_first": float(fit4_tail_first_m3["c3_ratio"]),
            "resid_norm_rel_m3_exact": float(fit4_tail_exact_m3["resid_rel"]),
        },
    }
    print("FB1 " + json.dumps(row), flush=True)
    return row


def run_mutation_check() -> None:
    """Verify that ``run_self_check_for_case`` raises ``AssertionError`` when normalisation is mutated."""
    info = compute_model_and_exact_roots(*CASES[0])
    failed_as_expected = False
    try:
        # Mutating amp_norm_factor from 2.0 to 1.0 halves c3_meas (ratio 0.5 instead of 1.0)
        run_self_check_for_case(0, info, amp_norm_factor=1.0)
    except AssertionError:
        failed_as_expected = True
    if not failed_as_expected:
        raise RuntimeError(
            "Mutation check failed: run_self_check_for_case did not fail with amp_norm_factor=1.0"
        )
    print("MUTATION_CHECK status=PASS", flush=True)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for falsifier F-b1."""
    parser = argparse.ArgumentParser(
        description="Falsifier F-b1: velocity-shear term vs mode-3 leakage."
    )
    parser.add_argument(
        "--case",
        type=int,
        choices=range(len(CASES)),
        default=None,
        help="Case index in [(10,474,1), (20,948,1), (10,474,2)] (default: all cases).",
    )
    parser.add_argument(
        "--self-check-only",
        action="store_true",
        help="Run only the synthetic self-checks (CPU, no particle runs).",
    )
    parser.add_argument(
        "--mutate-check",
        action="store_true",
        help="Verify that the synthetic self-check fails under a normalisation mutation.",
    )
    args = parser.parse_args(argv)

    if args.mutate_check:
        run_mutation_check()
        return

    indices = [args.case] if args.case is not None else list(range(len(CASES)))
    for idx in indices:
        run_case(idx, self_check_only=args.self_check_only)


if __name__ == "__main__":
    main()
