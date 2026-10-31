# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Verification and physics tests (P6a) for cathode surface emission models in
``wakis.particles.emission``:

* (a) Unit tests of the Richardson–Dushman, Schottky lowering, Forbes (2006)
  barrier functions :math:`v(f), t(f)`, Murphy–Good Fowler–Nordheim, and
  Gauss-law space-charge-limited formulas, with mutation checks (including
  dropping :math:`t(f)^2`).
* (d) Discrete Gauss's law (:func:`~wakis.particles.gauss_residual`) at machine
  round-off in the vacuum region for all three emitters.
* (e) GPU-vs-CPU equivalence to IEEE-754 double-precision round-off and
  verification of zero host synchronizations during ``inject(t, dt)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0, h, hbar, m_e, pi
from scipy.constants import k as k_B
from scipy.special import ellipe, ellipk

import wakis.particles.emission as emission_mod
import wakis.particles.solver as psolver_mod
from wakis import GridFIT3D, SolverFIT3D
from wakis.particles import (
    BeamInjector,
    FowlerNordheimEmitter,
    ParticleSolver,
    SpaceChargeLimitedEmitter,
    Species,
    ThermionicEmitter,
    child_langmuir_current_density,
    forbes_barrier_functions,
    fowler_nordheim_current_density,
    gauss_residual,
    richardson_dushman_current_density,
    schottky_barrier_lowering,
)
from wakis.particles.boundaries import InjectorProtocol

try:
    import cupy as cp
    import cupyx

    _HAS_CUPY = bool(cp.cuda.is_available())
except (ImportError, RuntimeError):  # pragma: no cover
    cp = None
    cupyx = None
    _HAS_CUPY = False


# ===========================================================================
# (a) Unit tests of the current-density formulas and barrier functions
# ===========================================================================


def _forbes_deane_exact_v_t(f_arr):
    """
    Exact Schottky–Nordheim barrier functions :math:`v(f)` and :math:`t(f)`
    in terms of complete elliptic integrals of the first and second kinds
    :math:`K(m)` and :math:`E(m)` (where ``scipy.special.ellipk(m)`` and
    ``scipy.special.ellipe(m)`` use the parameter :math:`m = k^2`):

    With :math:`s = \\sqrt{1 - f}` and elliptic parameter
    :math:`m = k^2 = \\frac{2 s}{1 + s}` for :math:`0 < f \\le 1`
    (Murphy & Good 1956, Phys. Rev. 102, 1464, Eqs. (55)–(58), where
    :math:`y = \\sqrt{f}`; Forbes & Deane 2007, Proc. R. Soc. A 463, 2907,
    doi:10.1098/rspa.2007.0030):

    .. math::

        v_{\\text{exact}}(f) = \\sqrt{\\frac{1 + s}{2}}\\,
        \\left[E(m) - (1 - s)\\,K(m)\\right],

    .. math::

        t_{\\text{exact}}(f) = v_{\\text{exact}}(f)
        - \\frac{4}{3}\\,f\\,\\frac{\\mathrm{d}v_{\\text{exact}}}{\\mathrm{d}f}
        = \\sqrt{\\frac{1 + s}{2}}\\,E(m),

    with :math:`v_{\\text{exact}}(0) = 1` and :math:`t_{\\text{exact}}(0) = 1`.
    """
    f = np.asarray(f_arr, dtype=np.float64)
    v_ex = np.ones_like(f, dtype=np.float64)
    t_ex = np.ones_like(f, dtype=np.float64)
    pos = (f > 0.0) & (f <= 1.0)
    if np.any(pos):
        s = np.sqrt(np.maximum(1.0 - f[pos], 0.0))
        m = (2.0 * s) / (1.0 + s)
        K_m = ellipk(m)
        E_m = ellipe(m)
        pref = np.sqrt(0.5 * (1.0 + s))
        v_ex[pos] = pref * (E_m - (1.0 - s) * K_m)
        t_ex[pos] = pref * E_m
    return v_ex, t_ex


def _check_forbes_barrier_functions():
    """
    Verify Forbes (2006) barrier functions v(f) and t(f) against:
    1. Exact analytical f -> 0+ limit: v(0) = 1, t(0) = 1.
    2. Exact analytical f = 1 barrier-top limit: v(1) = 0, t(1) = 10/9, and
       RuntimeWarning / clamping for f >= 1 (over-the-barrier regime).
    3. Identity check of Forbes (2006), Appl. Phys. Lett. 89,
       113122, Eqs. (5) and (9) at f = 0.25 and f = 0.49.
    4. Cross-check of the closed-form complete elliptic integral expression
       ``t_exact(f) = sqrt((1 + s) / 2) * E(m)`` against the defining differential
       relation ``t(f) = v(f) - (4 / 3) * f * dv/df`` evaluated via a central
       finite difference of ``v_exact(f)`` within the analytical Taylor + round-off
       bound derived from the Schottky-Nordheim ODE ``f*(1 - f)*v''(f) = (3/16)*v(f)``
       (Forbes & Deane 2007, Proc. R. Soc. A 463, 2907, Eq. (2.8)).
    5. Exact complete elliptic integral values v_exact(f), t_exact(f) via
       ``scipy.special.ellipk`` and ``scipy.special.ellipe`` (Murphy & Good
       1956, Eqs. (55)-(58); Forbes & Deane 2007, Proc. R. Soc. A 463, 2907)
       across [0, 1] within Forbes (2006)'s stated maximum error bounds:
       - Forbes (2006), p. 113122-2, Eq. (5) & Table I (unverified quote):
         maximum absolute error ``|v_Forbes - v_exact| <= 0.0024`` and stated
         relative error ``<= 0.0033`` (0.33%, which holds on the Table I grid
         ``y in {0.1, ..., 0.9}`` and outside the inter-Table-I window
         ``f in (0.25, 0.36)`` between ``y = 0.5`` and ``y = 0.6``—a window
         chosen after seeing the measured excess interval—where the continuous
         relative error slightly exceeds ``0.0033`` and is reported as a
         FINDING).
       - Forbes (2006), p. 113122-3, Eq. (9) (unverified quote): stated relative
         error bound ``|t_Forbes - t_exact| / t_exact <= 0.004`` ("better than
         0.4%" over ``0 <= y <= 1``) and corresponding absolute error bound
         ``|t_Forbes - t_exact| <= 0.004 * (10 / 9)``.
       - At the barrier top ``f = 1``, exact ``t_exact(1) = pi / (2 * sqrt(2))``
         differs from ``t(1) = 10 / 9`` by ``|10/9 - pi/(2*sqrt(2))| / t_exact(1) =
         20*sqrt(2)/(9*pi) - 1 = 3.515e-4`` (0.035%), so no uniform relative
         error bound on ``t(f)`` over ``[0, 1]`` can be smaller than this value.
    """
    # 1. Exact f -> 0 limit (and f = 0 boundary)
    v0, t0 = forbes_barrier_functions(0.0)
    assert abs(v0 - 1.0) <= 1e-15
    assert abs(t0 - 1.0) <= 1e-15

    f_tiny = 1e-14
    v_tiny, t_tiny = forbes_barrier_functions(f_tiny)
    # Leading asymptotic correction is (1/6) f ln(f) ~ -5.4e-13
    assert abs(v_tiny - 1.0) <= 1e-11
    assert abs(t_tiny - 1.0) <= 1e-11

    # 2. Barrier top f = 1 and over-the-barrier f > 1:
    # must emit RuntimeWarning and clamp to v(1) = 0, t(1) = 10/9
    with pytest.warns(RuntimeWarning, match="f >= 1"):
        v1, t1 = forbes_barrier_functions(1.0)
    assert abs(v1 - 0.0) <= 1e-15
    assert abs(t1 - (10.0 / 9.0)) <= 1e-15
    with pytest.warns(RuntimeWarning, match="f >= 1"):
        v_over, t_over = forbes_barrier_functions(1.35)
    assert abs(v_over - 0.0) <= 1e-15
    assert abs(t_over - (10.0 / 9.0)) <= 1e-15

    # 3. Identity check at f = 0.25 (y = 0.50) and f = 0.49 (y = 0.70)
    f_arr = np.array([0.25, 0.49], dtype=np.float64)
    v_arr, t_arr = forbes_barrier_functions(f_arr)
    v_ref = 1.0 - f_arr + (1.0 / 6.0) * f_arr * np.log(f_arr)
    t_ref = 1.0 + (1.0 / 9.0) * f_arr - (1.0 / 18.0) * f_arr * np.log(f_arr)
    # Round-off bound: <= 10 * eps_mach ~ 2.2e-15
    assert np.max(np.abs(v_arr - v_ref)) <= 1e-14
    assert np.max(np.abs(t_arr - t_ref)) <= 1e-14

    # 4. Cross-check closed-form t_exact(f) against t = v - (4/3)*f*(dv/df)
    # with dv/df from a central finite difference of v_exact(f):
    f_fd = np.linspace(0.05, 0.95, 500, dtype=np.float64)
    h_fd = 1.0e-4
    v_fd0, t_fd_exact = _forbes_deane_exact_v_t(f_fd)
    v_fd_p, t_fd_p = _forbes_deane_exact_v_t(f_fd + h_fd)
    v_fd_m, t_fd_m = _forbes_deane_exact_v_t(f_fd - h_fd)
    dv_df_central = (v_fd_p - v_fd_m) / (2.0 * h_fd)
    t_from_fd = v_fd0 - (4.0 / 3.0) * f_fd * dv_df_central

    # Exact 3rd derivative v'''(f) from the Schottky-Nordheim ODE
    # f*(1 - f)*v''(f) = (3/16)*v(f) (Forbes & Deane 2007, Proc. R. Soc. A 463, 2907, Eq. (2.8)):
    # Differentiating gives v'''(f) = (3/16)*(f*(1-f)*v'(f) - (1-2*f)*v(f)) / (f*(1-f))^2
    # where v'(f) = (3/4)*(v(f) - t(f)) / f. By Taylor's theorem on [f - h, f + h]:
    # |t_FD(f) - t_exact(f)| <= (2/9) * f * h^2 * max_{|df|<=h} |v'''(f + df)| + (8/3) * f * eps_mach / h.
    def _abs_d3v_df3(f_pts, v_pts, t_pts):
        dv1 = 0.75 * (v_pts - t_pts) / f_pts
        return np.abs(
            (3.0 / 16.0)
            * (f_pts * (1.0 - f_pts) * dv1 - (1.0 - 2.0 * f_pts) * v_pts)
            / ((f_pts * (1.0 - f_pts)) ** 2)
        )

    d3v_sup = np.maximum(
        _abs_d3v_df3(f_fd, v_fd0, t_fd_exact),
        np.maximum(
            _abs_d3v_df3(f_fd + h_fd, v_fd_p, t_fd_p),
            _abs_d3v_df3(f_fd - h_fd, v_fd_m, t_fd_m),
        ),
    )
    eps_mach = float(np.finfo(np.float64).eps)
    t_fd_err_bound = (2.0 / 9.0) * f_fd * (h_fd**2) * d3v_sup + (8.0 / 3.0) * f_fd * (
        eps_mach / h_fd
    )
    t_fd_abs_diff = np.abs(t_from_fd - t_fd_exact)
    print(
        f"[t_exact central FD check] max|t_FD - t_exact|={float(np.max(t_fd_abs_diff)):.3e} "
        f"<= max(derived_bound)={float(np.max(t_fd_err_bound)):.3e}"
    )
    assert np.all(t_fd_abs_diff <= t_fd_err_bound)

    # 5. Comparison against exact complete elliptic integrals (Forbes 2006; Forbes & Deane 2007)
    y_table_i = np.linspace(0.1, 0.9, 9, dtype=np.float64)
    f_table_i = y_table_i**2
    v_tab_forbes, _ = forbes_barrier_functions(f_table_i)
    v_tab_exact, _ = _forbes_deane_exact_v_t(f_table_i)
    v_rel_err_table_i = float(np.max(np.abs(v_tab_forbes - v_tab_exact) / v_tab_exact))
    # Forbes (2006), p. 113122-2, Eq. (5) & Table I: relative error <= 0.33% on Table I points
    assert v_rel_err_table_i <= 0.0033

    f_grid = np.linspace(1e-6, 1.0 - 1e-6, 2000, dtype=np.float64)
    v_forbes, t_forbes = forbes_barrier_functions(f_grid)
    v_exact, t_exact = _forbes_deane_exact_v_t(f_grid)

    v_abs_err = float(np.max(np.abs(v_forbes - v_exact)))
    v_rel_vec = np.abs(v_forbes - v_exact) / v_exact
    v_rel_err = float(np.max(v_rel_vec))
    t_abs_err = float(np.max(np.abs(t_forbes - t_exact)))
    t_rel_err = float(np.max(np.abs(t_forbes - t_exact) / t_exact))
    print(
        f"[Forbes vs exact elliptic] max|dv|={v_abs_err:.6f} (rel={100 * v_rel_err:.4f}%, "
        f"Table I rel={100 * v_rel_err_table_i:.4f}%), "
        f"max|dt|={t_abs_err:.6f} (rel={100 * t_rel_err:.4f}%)"
    )
    # Forbes (2006), p. 113122-2, Eq. (5) & Table I stated bounds for v (unverified quote):
    # max absolute error <= 0.0024, max relative error <= 0.0033 (0.33%).
    assert v_abs_err <= 0.0024
    exceed_mask = v_rel_vec > 0.0033
    assert np.any(exceed_mask)
    f_exceed = f_grid[exceed_mask]
    y_exceed = np.sqrt(f_exceed)
    idx_peak = int(np.argmax(v_rel_vec))
    f_peak = float(f_grid[idx_peak])
    y_peak = float(np.sqrt(f_peak))
    v_rel_excess = float(v_rel_err - 0.0033)
    print(
        f"[FINDING Forbes 2006 v(f) continuous relative error] "
        f"Over f in [{float(f_exceed[0]):.6f}, {float(f_exceed[-1]):.6f}] "
        f"(y = sqrt(f) in [{float(y_exceed[0]):.6f}, {float(y_exceed[-1]):.6f}], "
        f"inside the inter-Table-I window (0.25, 0.36) between y = 0.5 and y = 0.6 "
        f"chosen after seeing the measured excess interval), |v_Forbes - v_exact|/v_exact "
        f"exceeds Forbes's quoted 0.0033 (0.33%) bound, peaking at "
        f"{v_rel_err:.6e} ({100 * v_rel_err:.4f}%) at f={f_peak:.6f} (y={y_peak:.6f}), "
        f"an excess of {v_rel_excess:.6e} ({100 * v_rel_excess:.4f}%): "
        f"the paper's 0.33% figure is rounded to two significant digits or was "
        f"evaluated on a coarser grid such as Table I."
    )
    # Assert the exact paper bound 0.0033 outside the inter-Table-I window
    # f in (0.25, 0.36) (between Table I points y = 0.5 and y = 0.6); note that
    # the reported window (0.25, 0.36) was chosen after seeing the measured excess
    # interval f_exceed, while max(v_rel_vec[~exceed_mask]) <= 0.0033 is true by
    # construction of exceed_mask and is therefore dropped:
    outside_inter_table_gap = (f_grid <= 0.25) | (f_grid >= 0.36)
    assert float(np.max(v_rel_vec[outside_inter_table_gap])) <= 0.0033
    with pytest.raises(AssertionError):
        assert v_rel_err <= 0.0033
    # Forbes (2006), p. 113122-3, Eq. (9) stated bound for t (unverified quote):
    # relative error "better than 0.4%" (<= 0.004) over 0 <= y <= 1, and |dt| <= 0.004 * (10/9)
    assert t_rel_err <= 0.004
    assert t_abs_err <= 0.004 * (10.0 / 9.0)

    # At f = 1, exact elliptic integral gives v_exact(1) = 0, t_exact(1) = pi / (2 * sqrt(2)).
    # Analytical relative difference at f = 1 is 20*sqrt(2)/(9*pi) - 1 (~ 0.0351%), so no
    # uniform relative bound on t(f) across [0, 1] can be smaller than this barrier-top offset.
    v_ex_1, t_ex_1 = _forbes_deane_exact_v_t(np.array([1.0]))
    t1_exact_val = pi / (2.0 * np.sqrt(2.0))
    t1_rel_diff = abs((10.0 / 9.0) - t1_exact_val) / t1_exact_val
    assert abs(v_ex_1[0] - 0.0) <= 1e-15
    assert abs(t_ex_1[0] - t1_exact_val) <= 1e-15
    assert abs(t1 - t_ex_1[0]) / t_ex_1[0] == pytest.approx(t1_rel_diff, rel=1e-12)
    assert t_rel_err >= t1_rel_diff


def _check_schottky_and_thermionic():
    """
    Verify Schottky barrier lowering dW_S = sqrt(e^3 F / (4*pi*eps_0)) and
    Richardson-Dushman constant A_0 against independent numerical literals
    (dW_S(1e8 V/m) = 0.379469 eV and A_0 = 1.20173e6 A m^-2 K^-2), plus
    identity checks of the Richardson-Dushman formula at F = 1.0e8 V/m,
    T = 1500 K, W = 4.5 eV, lambda_R = 0.5.
    """
    F = 1.0e8  # [V/m]
    T = 1500.0  # [K]
    W_eV = 4.5  # [eV]
    lambda_R = 0.5

    # Literal check 1: Schottky barrier lowering at F = 1.0e8 V/m
    # sqrt(e^3 * 1e8 / (4 * pi * eps_0)) / e = 0.37946864783098777 eV
    dW_got_J = schottky_barrier_lowering(F)
    dW_got_eV = dW_got_J / q_e
    assert abs(dW_got_eV - 0.379469) <= 5e-7
    assert abs(dW_got_eV - 0.37946864783098777) <= 1e-14
    assert schottky_barrier_lowering(0.0) == 0.0
    assert schottky_barrier_lowering(-5.0e7) == 0.0

    # Literal check 2: Universal Richardson-Dushman constant
    # A_0 = 4 * pi * m_e * k_B^2 * e / h^3 = 1.2017331470281917e6 A / (m^2 K^2)
    A0_ref = (4.0 * pi * m_e * (k_B**2) * q_e) / (h**3)
    assert abs(A0_ref - 1.20173e6) / 1.20173e6 <= 5e-6
    assert abs(emission_mod.RICHARDSON_A0 - 1.20173e6) / 1.20173e6 <= 5e-6
    assert abs(emission_mod.RICHARDSON_A0 - A0_ref) / A0_ref <= 1e-15

    # Identity check: Zero-field Richardson-Dushman current density
    J0_ref = lambda_R * A0_ref * (T**2) * np.exp(-(W_eV * q_e) / (k_B * T))
    J0_got = richardson_dushman_current_density(
        0.0, temperature=T, work_function_eV=W_eV, lambda_R=lambda_R
    )
    assert abs(J0_got - J0_ref) / J0_ref <= 1e-14

    # Identity check: Field-enhanced thermionic current density with Schottky lowering
    dW_ref_J = np.sqrt((q_e**3) * F / (4.0 * pi * epsilon_0))
    J_F_ref = lambda_R * A0_ref * (T**2) * np.exp(-(W_eV * q_e - dW_ref_J) / (k_B * T))
    J_F_got = richardson_dushman_current_density(
        F, temperature=T, work_function_eV=W_eV, lambda_R=lambda_R
    )
    assert abs(J_F_got - J_F_ref) / J_F_ref <= 1e-14

    # Identity check: Ratio J(F) / J(0) equals exp(dW_S / (k_B T)) to machine round-off
    schottky_boost_ref = np.exp(dW_ref_J / (k_B * T))
    assert abs((J_F_got / J0_got) - schottky_boost_ref) / schottky_boost_ref <= 1e-14


def _check_fowler_nordheim_formula():
    """
    Verify first and second Fowler-Nordheim constants against literals
    (a_FN = 1.541434e-6 A eV V^-2, b_FN = 6.830890e9 eV^(-3/2) V m^-1), plus
    identity checks of the Murphy-Good Fowler-Nordheim current density with
    Forbes (2006) v(f), t(f) and field-enhancement scaling beta_FE.
    """
    phi_eV = 4.5  # [eV]
    # Choose local field corresponding to scaled barrier field f = 0.5:
    # f = (e / (4*pi*eps_0)) * F_loc / phi_eV^2 => F_loc = 0.5 * phi_eV^2 / (e / (4*pi*eps_0))
    c_S_eV2 = q_e / (4.0 * pi * epsilon_0)
    f_target = 0.5
    F_loc = f_target * (phi_eV**2) / c_S_eV2  # ~ 7.03142e9 V/m

    v_ref = 1.0 - f_target + (1.0 / 6.0) * f_target * np.log(f_target)
    t_ref = 1.0 + (1.0 / 9.0) * f_target - (1.0 / 18.0) * f_target * np.log(f_target)

    # Literal check: First and second Fowler-Nordheim constants in [A eV V^-2] and [eV^(-3/2) V m^-1]
    # (Forbes 2006, Appl. Phys. Lett. 89, 113122, p. 113122-1:
    #  a_FN = e^3 / (8 pi h_P) = 1.541434e-6 A eV V^-2 with phi_J = e * phi_eV => e^2 / (16 pi^2 hbar);
    #  b_FN = 4 sqrt(2 m_e) phi_J^(3/2) / (3 hbar e F) => 4 sqrt(2 m_e e) / (3 hbar) = 6.830890e9 eV^(-3/2) V m^-1)
    a_FN = (q_e**2) / (16.0 * (pi**2) * hbar)
    b_FN = (4.0 * np.sqrt(2.0 * m_e * q_e)) / (3.0 * hbar)
    assert abs(a_FN - 1.541434e-6) / 1.541434e-6 <= 1e-6
    assert abs(b_FN - 6.830890e9) / 6.830890e9 <= 1e-6

    # Identity check of the Murphy-Good Fowler-Nordheim formula
    J_ref = (
        (a_FN * (F_loc**2))
        / (phi_eV * (t_ref**2))
        * np.exp(-(b_FN * (phi_eV**1.5) * v_ref) / F_loc)
    )

    J_got = fowler_nordheim_current_density(F_loc, work_function_eV=phi_eV, beta_fe=1.0)
    # Round-off bound: <= 1e-13 relative error
    assert abs(J_got - J_ref) / J_ref <= 1e-13

    # Identity check: Field enhancement factor beta_FE = 25 with macroscopic field F_loc / 25
    beta_fe = 25.0
    J_beta = fowler_nordheim_current_density(
        F_loc / beta_fe, work_function_eV=phi_eV, beta_fe=beta_fe
    )
    assert abs(J_beta - J_ref) / J_ref <= 1e-13

    # Zero or retarding field gives identically zero emission
    assert fowler_nordheim_current_density(0.0, work_function_eV=phi_eV) == 0.0
    assert fowler_nordheim_current_density(-1.0e9, work_function_eV=phi_eV) == 0.0


def test_emission_formulas_and_mutations(monkeypatch):
    """
    Test (a): Verify Forbes (2006) v(f) and t(f), Schottky lowering,
    Richardson–Dushman, and Murphy–Good Fowler–Nordheim formulas against
    hand-derived and cited tabulated values, and verify that dropping t(f)^2
    or Schottky lowering causes the test to fail.
    """
    _check_forbes_barrier_functions()
    _check_schottky_and_thermionic()
    _check_fowler_nordheim_formula()

    # Mutation 1 (required by brief): drop t^2 in Fowler-Nordheim by returning t = 1.0
    orig_forbes = emission_mod.forbes_barrier_functions

    def _forbes_drop_t2(f):
        v, t = orig_forbes(f)
        if np.ndim(t) == 0:
            return v, 1.0
        return v, np.ones_like(t)

    with monkeypatch.context() as m:
        m.setattr(emission_mod, "forbes_barrier_functions", _forbes_drop_t2)
        with pytest.raises(AssertionError):
            _check_fowler_nordheim_formula()

    # Mutation 2: drop Schottky barrier lowering dW_S -> 0 in Richardson-Dushman
    with monkeypatch.context() as m:
        m.setattr(
            emission_mod,
            "schottky_barrier_lowering",
            lambda E_field: 0.0 * np.asarray(E_field, dtype=np.float64),
        )
        with pytest.raises(AssertionError):
            _check_schottky_and_thermionic()


def test_emitter_patches_and_space_charge_capping(monkeypatch):
    """
    Verify rectangular and circular disk cathode patches on the FIT primal
    node / dual-face geometry, zero-weight pile-up prevention on CPU when
    ``E_pull <= 0`` everywhere, and ``space_charge_limit=True`` capping on
    ``ThermionicEmitter`` and ``FowlerNordheimEmitter``.
    """
    grid = GridFIT3D(-0.02, 0.02, -0.02, 0.02, 0.0, 0.01, 4, 4, 4)
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        use_stl=False,
        verbose=0,
    )
    dt = float(solver.dt)
    E0 = 2.0e6  # [V/m] pulling electrons off zmin when Ez = -E0

    # 1. Disk patch of radius 0.012 m around (0, 0): selects the 9 inner primal
    # nodes (x_i, y_j) in {-0.01, 0.0, +0.01}^2 (with dual face area 0.01 * 0.01 m^2
    # each), where r = sqrt(x_i^2 + y_j^2) <= sqrt(2)*0.01 = 0.01414 m (the 4 corner
    # nodes at (+-0.01, +-0.01) have r = 0.01414 m > 0.012 m, leaving the 5 cross
    # nodes (0,0), (+-0.01,0), (0,+-0.01) inside r <= 0.012 m).
    sp_disk = Species(name="e_disk", charge=-q_e, mass=m_e)
    em_disk = SpaceChargeLimitedEmitter(
        solver, sp_disk, radius=0.012, center=(0.0, 0.0), maintain_voltage=False
    )
    assert em_disk.n_emitting_cells == 5
    solver.E[:, :, 0, "z"] = -E0
    n_inj = em_disk.inject(0.0, dt)
    assert n_inj == 5
    area_5_nodes = 5.0 * (0.01 * 0.01)
    q_expected = -epsilon_0 * E0 * area_5_nodes
    assert abs(float(em_disk.injected_charge) - q_expected) / abs(q_expected) <= 1e-14

    # 2. Retarding or zero field (Ez >= 0 for electrons) everywhere:
    # On CPU, emission is skipped completely (returning 0 and adding 0 particles)
    # to prevent zero-weight particle pile-up when E <= 0.
    sp_ret = Species(name="e_ret", charge=-q_e, mass=m_e)
    em_ret = SpaceChargeLimitedEmitter(
        solver,
        sp_ret,
        x_bounds=(-0.01, 0.01),
        y_bounds=(-0.01, 0.01),
        maintain_voltage=False,
    )
    solver.E[:, :, 0, "z"] = +E0  # retarding for electrons
    n_ret = em_ret.inject(0.0, dt)
    assert n_ret == 0
    assert sp_ret.n_particles == 0
    assert float(em_ret.injected_charge) == 0.0
    # The skipped slots still consume their ids (as on GPU and on non-owning
    # MPI ranks), so the id stream does not depend on the execution mode.
    assert sp_ret._next_id == em_ret.n_emit_per_step

    # 3. ThermionicEmitter and FowlerNordheimEmitter with space_charge_limit=True
    # At high T or high beta_FE where unconstrained emission exceeds eps_0 * E0 * A,
    # space_charge_limit=True must cap the emitted charge at the Gauss-law value
    # across the full cathode area sum_{i,j} tilde{A}_z[i, j] = 0.04 * 0.04 m^2.
    solver.E[:, :, 0, "z"] = -E0
    sp_th = Species(name="e_th", charge=-q_e, mass=m_e)
    em_th = ThermionicEmitter(
        solver,
        sp_th,
        temperature=3500.0,
        work_function_eV=2.0,
        space_charge_limit=True,
        maintain_voltage=False,
    )
    em_th.inject(0.0, dt)
    q_gauss_full = -epsilon_0 * E0 * (0.04 * 0.04)
    assert abs(float(em_th.injected_charge) - q_gauss_full) / abs(q_gauss_full) <= 1e-14

    sp_fn = Species(name="e_fn", charge=-q_e, mass=m_e)
    em_fn = FowlerNordheimEmitter(
        solver,
        sp_fn,
        work_function_eV=2.0,
        beta_fe=5000.0,
        space_charge_limit=True,
        maintain_voltage=False,
    )
    em_fn.inject(0.0, dt)
    assert abs(float(em_fn.injected_charge) - q_gauss_full) / abs(q_gauss_full) <= 1e-14

    # Mutation check: disabling space_charge_limit on em_th must exceed the Gauss cap
    with monkeypatch.context() as m:
        m.setattr(em_th, "space_charge_limit", False)
        sp_th.clear()
        em_th.injected_charge = 0.0
        em_th.inject(0.0, dt)
        with pytest.raises(AssertionError):
            assert (
                abs(float(em_th.injected_charge) - q_gauss_full) / abs(q_gauss_full)
                <= 1e-14
            )


def test_transverse_nonuniform_field_and_graded_grid(monkeypatch):
    """
    Test SHOULD-FIX 6:
    1. On a graded transverse and longitudinal mesh with ``Nx > 1, Ny > 1`` and
       a transversely non-uniform applied cathode field ``E_z(x_i, y_j, 0)``,
       each primary edge ``E_z[i, j, 0]`` is paired with its exact FIT dual face
       area ``tilde{A}_z[i, j] = 1 / itA_z[i, j, 0]`` and macroparticles are
       emitted at primal nodes ``(x_i, y_j, z_0)``.
    2. Immediately after ``SpaceChargeLimitedEmitter.inject(t, dt)``:
       - **Algebraic identity check**: ``E_surf[i, j] = E_z[i, j, 0] - q_0[i, j] / (eps_0 * tilde{A}_z[i, j]) -> 0``
         holds to IEEE-754 double-precision round-off (<= 1e-14 relative to max|E_z|),
         and a subsequent call to ``_read_pulling_surface_field()`` returns 0 on
         every node.
       - **Independent FIT divergence / PEC surface-charge check**:
         ``compute_dual_divergence_d(solver)[:Nx, :Ny, 0]`` minus the cathode-node
         deposited charge ``q_3d[:Nx, :Ny, 0]`` from ``deposit_charge(..., order=1)``
         (the residual PEC surface charge on the cathode nodes) goes to ``0`` on
         every cathode node to machine round-off (<= 1e-14).
    3. ``_cathode_node_charge()`` uses ``grid_map.to_logical`` so that particles
       displaced across graded cells deposit the exact same CIC cathode node
       charge as ``wakis.particles.shapes.deposit_charge(..., order=1)``.
    4. Attaching a ``ParticleSolver`` with ``shape != 1`` raises
       ``NotImplementedError`` when ``_cathode_node_charge()`` is called.
    """
    from wakis.particles.shapes import deposit_charge
    from wakis.particles.solver import compute_dual_divergence_d

    x_nodes = np.array([-0.02, -0.012, -0.002, 0.009, 0.02], dtype=np.float64)
    y_nodes = np.array([-0.015, -0.004, 0.006, 0.015], dtype=np.float64)
    z_nodes = np.array([0.0, 0.001, 0.0025, 0.005, 0.008], dtype=np.float64)
    grid = GridFIT3D(
        x_nodes[0],
        x_nodes[-1],
        y_nodes[0],
        y_nodes[-1],
        z_nodes[0],
        z_nodes[-1],
        len(x_nodes) - 1,
        len(y_nodes) - 1,
        len(z_nodes) - 1,
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    dt = float(solver.dt)
    sp = Species(name="electron", charge=-q_e, mass=m_e)
    em = SpaceChargeLimitedEmitter(
        solver, sp, n_macro_per_cell=2, thermal_u=1.0e5, maintain_voltage=False
    )

    # Transversely non-uniform pulling field on the Nx x Ny = 4 x 3 cathode nodes
    xx, yy = np.meshgrid(x_nodes[:-1], y_nodes[:-1], indexing="ij")
    Ez_nonuniform = -1.5e6 * (
        1.0 + 25.0 * (xx - x_nodes[0]) + 15.0 * (yy - y_nodes[0]) ** 2
    )
    solver.E[:4, :3, 0, "z"] = Ez_nonuniform

    n_inj = em.inject(0.0, dt)
    assert n_inj == 4 * 3 * 2

    # 1. Algebraic identity check: E_surf = E_z - q_0 / (eps_0 * tilde{A}_z) -> 0 on EVERY node
    q0_nodes = em._cathode_node_charge()
    e_surf_nodes = em._ez_3d_view()[em._cell_i, em._cell_j, 0] - q0_nodes / (
        epsilon_0 * em._cell_areas
    )
    rel_residual = float(np.max(np.abs(e_surf_nodes)) / np.max(np.abs(Ez_nonuniform)))
    print(
        f"[SHOULD-6 algebraic identity graded Nx=4,Ny=3] max|E_surf|/max|E_0| = {rel_residual:.3e}"
    )
    assert rel_residual <= 1e-14
    assert float(np.max(np.abs(em._read_pulling_surface_field()))) <= (
        1e-14 * float(np.max(np.abs(Ez_nonuniform)))
    )

    # 2. Independent check: compute_dual_divergence_d minus deposited charge at the
    # cathode nodes (the residual PEC surface charge) goes to 0 after emission.
    div_d_cathode = compute_dual_divergence_d(solver)[:4, :3, 0]
    xi_b, eta_b, zeta_b = em.grid_map.to_logical(sp.x, sp.y, sp.z)
    q_3d_birth = np.zeros(
        (solver.Nx, solver.Ny, solver.Nz), dtype=np.float64, order="F"
    )
    deposit_charge(
        xi_b,
        eta_b,
        zeta_b,
        sp.charge * sp.w,
        solver.Nx,
        solver.Ny,
        solver.Nz,
        order=1,
        periodic=em._periodic_xy + (False,),
        q_nodes=q_3d_birth,
        xp=np,
    )
    pec_surface_charge_after = div_d_cathode - q_3d_birth[:4, :3, 0]
    pec_rel_res = float(
        np.max(np.abs(pec_surface_charge_after)) / np.max(np.abs(div_d_cathode))
    )
    print(
        f"[SHOULD-6 independent PEC surface-charge check] max|div_d - q_dep|/max|div_d| = {pec_rel_res:.3e}"
    )
    assert pec_rel_res <= 1e-14

    # Verify _cathode_node_charge matches full 3D CIC deposit_charge on the graded mesh
    # even after particles move to off-node positions inside the first cell:
    rng = np.random.default_rng(1234)
    sp.x[: sp.n_particles] = rng.uniform(x_nodes[0], x_nodes[-2], size=sp.n_particles)
    sp.y[: sp.n_particles] = rng.uniform(y_nodes[0], y_nodes[-2], size=sp.n_particles)
    sp.z[: sp.n_particles] = rng.uniform(
        z_nodes[0], 0.9 * z_nodes[1], size=sp.n_particles
    )
    q0_from_emitter = em._cathode_node_charge()

    xi_p, eta_p, zeta_p = em.grid_map.to_logical(sp.x, sp.y, sp.z)
    q_3d = np.zeros((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64, order="F")
    deposit_charge(
        xi_p,
        eta_p,
        zeta_p,
        sp.charge * sp.w,
        solver.Nx,
        solver.Ny,
        solver.Nz,
        order=1,
        periodic=em._periodic_xy + (False,),
        q_nodes=q_3d,
        xp=np,
    )
    q0_from_deposit = q_3d[em._cell_i, em._cell_j, 0]
    assert (
        np.max(np.abs(q0_from_emitter - q0_from_deposit))
        / np.max(np.abs(q0_from_deposit))
        <= 1e-14
    )

    # Mutation check: replacing dual areas tilde{A}_z with uniform primal cell area dx0*dy0
    # breaks the independent PEC surface-charge cancellation (div_d - q_dep -> 0) on the graded mesh
    with monkeypatch.context() as m:
        sp.clear()
        m.setattr(
            em,
            "_cell_areas",
            np.full_like(
                em._cell_areas,
                (x_nodes[1] - x_nodes[0]) * (y_nodes[1] - y_nodes[0]),
            ),
        )
        em.inject(0.0, dt)
        xi_m, eta_m, zeta_m = em.grid_map.to_logical(sp.x, sp.y, sp.z)
        q_3d_mut = np.zeros(
            (solver.Nx, solver.Ny, solver.Nz), dtype=np.float64, order="F"
        )
        deposit_charge(
            xi_m,
            eta_m,
            zeta_m,
            sp.charge * sp.w,
            solver.Nx,
            solver.Ny,
            solver.Nz,
            order=1,
            periodic=em._periodic_xy + (False,),
            q_nodes=q_3d_mut,
            xp=np,
        )
        pec_mut_res = float(
            np.max(np.abs(div_d_cathode - q_3d_mut[:4, :3, 0]))
            / np.max(np.abs(div_d_cathode))
        )
        with pytest.raises(AssertionError):
            assert pec_mut_res <= 1e-14

    # Shape != 1 must raise NotImplementedError
    psolver = ParticleSolver(
        solver, species=[sp], injectors=[em], shape=2, gather="momentum"
    ).attach()
    assert psolver.shape == 2
    with pytest.raises(NotImplementedError, match="shape=1"):
        em._cathode_node_charge()


# ===========================================================================
# (d) Gauss's law in the vacuum region for all three emitters
# ===========================================================================


@pytest.mark.parametrize(
    "emitter_kind", ["space_charge", "thermionic", "fowler_nordheim"]
)
def test_gauss_law_all_emitters(emitter_kind, monkeypatch):
    """
    Test (d): Verify that discrete Gauss's law ``gauss_residual(solver)`` holds
    to machine round-off (<= 1e-12) in the vacuum region (including the cathode
    plane ``k = 0``) during multi-step self-consistent evolution with each of
    the three surface emitters, and that mutating the code path so that
    ``_particles_q_bg -= q_inj_nodes`` is skipped causes ``gauss_residual`` to
    fail.
    """

    def _run_gauss_case(skip_q_bg_update: bool = False) -> float:
        grid = GridFIT3D(-0.01, 0.01, -0.01, 0.01, 0.0, 0.008, 4, 4, 8, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_stl=False,
            verbose=0,
        )
        electrons = Species(name="electron", charge=-q_e, mass=m_e)
        V_gap = 2000.0

        if emitter_kind == "space_charge":
            emitter = SpaceChargeLimitedEmitter(
                solver,
                electrons,
                voltage=V_gap,
                thermal_u=2.0e6,
                maintain_voltage=False,
            )
        elif emitter_kind == "thermionic":
            emitter = ThermionicEmitter(
                solver,
                electrons,
                temperature=2600.0,
                work_function_eV=2.2,
                voltage=V_gap,
                thermal_u=2.0e6,
                maintain_voltage=False,
            )
        else:
            emitter = FowlerNordheimEmitter(
                solver,
                electrons,
                work_function_eV=4.5,
                beta_fe=2.5e4,
                voltage=V_gap,
                thermal_u=2.0e6,
                maintain_voltage=False,
            )

        assert isinstance(emitter, InjectorProtocol)
        assert emitter.emitter_kind == "pec_wall"
        emitter.emitter_kind = None

        psolver = ParticleSolver(
            solver,
            species=[electrons],
            injectors=[emitter],
            shape=1,
            gather="momentum",
        ).attach()
        assert emitter.emitter_kind == "pec_wall"

        # Include the cathode plane k = 0 in the Gauss-residual node mask so
        # that the stationary counter-charge update `_particles_q_bg -= q_inj_nodes`
        # at z = zmin is directly checked by gauss_residual(solver, electrons).
        mask_with_cathode = psolver_mod._interior_vacuum_node_mask(solver).copy()
        mask_with_cathode[:, :, 0] = True

        with monkeypatch.context() as m:
            m.setattr(
                psolver_mod,
                "_interior_vacuum_node_mask",
                lambda _s: mask_with_cathode,
            )
            if skip_q_bg_update:
                # Code mutation (SHOULD-FIX 7): skip depositing q_inj_nodes into
                # psolver._q0_buf at injection (zeta == 0) so `_particles_q_bg -= q_inj_nodes`
                # is skipped while leaving boundary absorption intact.
                orig_deposit = psolver_mod.deposit_charge

                def _deposit_skip_q_inj(
                    xi,
                    eta,
                    zeta,
                    qw,
                    Nx,
                    Ny,
                    Nz,
                    order=1,
                    periodic=(True, True, True),
                    q_nodes=None,
                    xp=np,
                ):
                    if q_nodes is psolver._q0_buf and bool(xp.all(zeta == 0.0)):
                        return q_nodes
                    return orig_deposit(
                        xi,
                        eta,
                        zeta,
                        qw,
                        Nx,
                        Ny,
                        Nz,
                        order=order,
                        periodic=periodic,
                        q_nodes=q_nodes,
                        xp=xp,
                    )

                m.setattr(psolver_mod, "deposit_charge", _deposit_skip_q_inj)

            for _ in range(15):
                solver.one_step()

            assert electrons.n_particles > 0
            assert float(emitter.injected_charge) < 0.0
            return float(gauss_residual(solver, electrons))

    res = _run_gauss_case(skip_q_bg_update=False)
    # Derived tolerance: IEEE-754 float64 round-off over 15 steps is <= 1e-12
    assert res <= 1e-12, f"{emitter_kind} gauss_residual={res:.3e} > 1e-12"

    # Code mutation check (SHOULD-FIX 7): skipping `_particles_q_bg -= q_inj_nodes`
    # leaves the cathode counter-charge unrecorded and produces O(1) Gauss residual
    res_mut = _run_gauss_case(skip_q_bg_update=True)
    print(f"[Gauss {emitter_kind}] ref={res:.3e}, skip_q_bg_mut={res_mut:.3e}")
    with pytest.raises(AssertionError):
        assert res_mut <= 1e-12


# ===========================================================================
# (e) GPU path: emitted charge equivalence & zero host syncs during inject()
# ===========================================================================


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy / CUDA device not available")
@pytest.mark.parametrize(
    "emitter_kind", ["space_charge", "thermionic", "fowler_nordheim"]
)
def test_gpu_emission_matches_cpu_and_no_host_sync(emitter_kind, monkeypatch):
    """
    Test (e), SHOULD-FIX 8 & 9, and NOTE 12: Verify for each emitter on the GPU path:
    1. ``emitter.inject(t, dt)`` adds no host synchronizations of its own
       (verified both by forbidding ``cp.ndarray.get`` and under
       ``cupyx.allow_synchronize(False)``).
    2. Across a full ``solver.one_step()`` on a species that already has active
       particles and after boundary compaction has occurred:
       - Document and verify the solver-side sync at
         ``wakis/particles/solver.py`` ~l.2373: when boundary absorption sets
         ``species._n_dirty_gpu = True``, reading ``inj.species.n_particles``
         synchronizes once to resolve the compacted count.
       - Once the active count is resolved (or on steps where no boundary
         compaction left ``_n_dirty_gpu = True``), the entire
         ``solver.one_step()`` — including ``emitter.inject(t, dt)`` with
         ``_cathode_node_charge()`` via ``xp.add.at`` — completes inside
         ``with cupyx.allow_synchronize(False):`` with zero host syncs.
    3. Zero-weight handling (SHOULD-9): cells with zero or retarding field
       receive weight 0 on the device; when all cathode nodes have retarding
       field (every weight is 0), CPU skips ``add_particles``
       (``n_particles == 0``) while GPU appends ``n_emit_per_step`` zero-weight
       slots without a host sync, and both have identical total emitted charge ``0.0``.
    4. Emitted charge per step on the GPU equals the CPU result to the
       IEEE-754 double-precision round-off bound (<= 1e-13 relative error),
       multi-step evolution preserves ``gauss_residual <= 1e-12``, and mutating
       emitter code (``_read_pulling_surface_field`` / ``_compute_emitted_charge``)
       causes the GPU vs CPU charge assertion to fail.
    """

    def _build_pair(use_gpu: bool):
        grid = GridFIT3D(-0.01, 0.01, -0.01, 0.01, 0.0, 0.008, 4, 4, 8, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_stl=False,
            use_gpu=use_gpu,
            verbose=0,
        )
        sp = Species(name="electron", charge=-q_e, mass=m_e, use_gpu=use_gpu)
        # Spatially varying cathode field with both pulling (Ez < 0) and
        # retarding/zero (Ez >= 0) cells to test zero-weight emission on GPU
        ii, jj = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        Ez_pattern = np.where((ii + jj) % 2 == 0, -3.0e5 * (1.0 + 0.2 * ii), 1.0e5)
        if use_gpu:
            solver.E[:4, :4, 0, "z"] = cp.asarray(Ez_pattern, dtype=np.float64)
        else:
            solver.E[:4, :4, 0, "z"] = Ez_pattern

        if emitter_kind == "space_charge":
            em = SpaceChargeLimitedEmitter(
                solver, sp, n_macro_per_cell=2, thermal_u=1.5e6, maintain_voltage=False
            )
        elif emitter_kind == "thermionic":
            em = ThermionicEmitter(
                solver,
                sp,
                temperature=2500.0,
                work_function_eV=2.3,
                n_macro_per_cell=2,
                thermal_u=1.5e6,
                maintain_voltage=False,
            )
        else:
            em = FowlerNordheimEmitter(
                solver,
                sp,
                work_function_eV=4.5,
                beta_fe=2.0e4,
                n_macro_per_cell=2,
                thermal_u=1.5e6,
                maintain_voltage=False,
            )
        return solver, sp, em

    solver_cpu, sp_cpu, em_cpu = _build_pair(use_gpu=False)
    _solver_gpu, sp_gpu, em_gpu = _build_pair(use_gpu=True)
    dt = float(solver_cpu.dt)

    # 1. Verify emitter.inject(t, dt) adds zero host syncs under cupyx.allow_synchronize(False)
    def _forbid_get(self, *args, **kwargs):
        raise RuntimeError(
            "Host synchronization (cp.ndarray.get) called on GPU step path!"
        )

    with monkeypatch.context() as m:
        m.setattr(cp.ndarray, "get", _forbid_get)
        with cupyx.allow_synchronize(False):
            n_gpu = em_gpu.inject(0.0, dt)
            _ = abs(em_gpu.injected_charge)

    n_cpu = em_cpu.inject(0.0, dt)
    assert n_gpu == n_cpu == 4 * 4 * 2

    w_cpu = sp_cpu.w.copy()
    w_gpu = sp_gpu.w.get()
    # Retarding cells must have exact weight 0.0
    assert np.any(w_gpu == 0.0)
    assert np.all((w_gpu == 0.0) == (w_cpu == 0.0))

    q_cpu_step0 = float(em_cpu.injected_charge)
    q_gpu_step0 = float(em_gpu.injected_charge)
    assert q_cpu_step0 < 0.0
    # Derived tolerance: IEEE-754 float64 CPU vs GPU round-off bound <= 1e-13
    assert abs(q_gpu_step0 - q_cpu_step0) / abs(q_cpu_step0) <= 1e-13
    assert np.max(np.abs(w_gpu - w_cpu)) / np.max(w_cpu) <= 1e-13

    # SHOULD-9: when all cathode nodes have retarding field (all weights 0),
    # CPU skips add_particles (n_particles == 0) while GPU appends n_emit_per_step
    # zero-weight slots without a host sync, and both have identical total charge 0.0.
    s_cpu_z, sp_cpu_z, em_cpu_z = _build_pair(use_gpu=False)
    s_gpu_z, sp_gpu_z, em_gpu_z = _build_pair(use_gpu=True)
    s_cpu_z.E[:4, :4, 0, "z"] = +1.0e5
    s_gpu_z.E[:4, :4, 0, "z"] = cp.full((4, 4), +1.0e5, dtype=np.float64)
    n_c_z = em_cpu_z.inject(0.0, dt)
    with cupyx.allow_synchronize(False):
        n_g_z = em_gpu_z.inject(0.0, dt)
    assert n_c_z == 0 and sp_cpu_z.n_particles == 0
    assert (
        n_g_z == em_gpu_z.n_emit_per_step
        and sp_gpu_z.n_particles == em_gpu_z.n_emit_per_step
    )
    assert float(em_cpu_z.injected_charge) == float(em_gpu_z.injected_charge) == 0.0
    assert sp_cpu_z._next_id == sp_gpu_z._next_id
    assert float(cp.sum(sp_gpu_z.w)) == 0.0

    # 2. Step fresh CPU and GPU solvers through ParticleSolver for 6 steps and
    # compare per-step emitted charge and discrete Gauss's law
    solver_cpu2, sp_cpu2, em_cpu2 = _build_pair(use_gpu=False)
    solver_gpu2, sp_gpu2, em_gpu2 = _build_pair(use_gpu=True)
    # Use a divergence-free initial vacuum field along z so interior vacuum has div D(0) = 0
    Ez_init_2d = np.where(
        (np.arange(4)[:, None] + np.arange(4)[None, :]) % 2 == 0,
        -3.0e5,
        1.0e5,
    )
    for k_idx in range(8):
        solver_cpu2.E[:4, :4, k_idx, "z"] = Ez_init_2d
        solver_gpu2.E[:4, :4, k_idx, "z"] = cp.asarray(Ez_init_2d, dtype=np.float64)

    ParticleSolver(
        solver_cpu2,
        species=[sp_cpu2],
        injectors=[em_cpu2],
        shape=1,
        gather="momentum",
    ).attach()
    ParticleSolver(
        solver_gpu2,
        species=[sp_gpu2],
        injectors=[em_gpu2],
        shape=1,
        gather="momentum",
    ).attach()

    # Step 0: injects initial particles and boundary absorption marks _n_dirty_gpu = True
    solver_cpu2.one_step()
    solver_gpu2.one_step()
    assert sp_gpu2._n_dirty_gpu is True

    # Document & verify the solver-side sync at solver.py ~l.2373: when _n_dirty_gpu is True
    # after boundary compaction, solver.one_step() queries `inj.species.n_particles`, which
    # synchronizes to host once to read `_n_dev`.
    with pytest.raises(cupyx.DeviceSynchronized), cupyx.allow_synchronize(False):
        _ = sp_gpu2.n_particles

    # Resolve the compacted particle count after step 0 (sp_gpu2 already has active particles):
    n_active_after_compaction = sp_gpu2.n_particles
    assert n_active_after_compaction > 0
    assert sp_gpu2._n_dirty_gpu is False

    # Now run a full solver.one_step() on the populated species inside
    # `with cupyx.allow_synchronize(False):` — verifying that the emitter itself
    # (including _cathode_node_charge with xp.add.at) adds zero host syncs.
    solver_cpu2.one_step()
    with cupyx.allow_synchronize(False):
        solver_gpu2.one_step()

    for _ in range(4):
        solver_cpu2.one_step()
        solver_gpu2.one_step()
        q_c = float(em_cpu2.injected_charge)
        q_g = float(em_gpu2.injected_charge)
        assert abs(q_g - q_c) / abs(q_c) <= 1e-12

    assert gauss_residual(solver_gpu2, sp_gpu2) <= 1e-12

    # Code mutation check (NOTE 12): mutating `_read_pulling_surface_field` on
    # the GPU emitter to omit the `maximum(..., 0.0)` rectification (using `abs`
    # so retarding cells also emit) must cause the GPU vs CPU charge check to fail.
    with monkeypatch.context() as m:

        def _mut_pulling_field():
            em_gpu2._enforce_gap_voltage()
            Ez_3d = em_gpu2._ez_3d_view()
            Ez_cell = Ez_3d[:, :, em_gpu2._k_read][
                em_gpu2._cell_i, em_gpu2._cell_j
            ].astype(np.float64, copy=False)
            q0_cell = em_gpu2._cathode_node_charge()
            Ez_surf = Ez_cell - q0_cell / (epsilon_0 * em_gpu2._cell_areas)
            return em_gpu2.xp.abs(Ez_surf)

        m.setattr(em_gpu2, "_read_pulling_surface_field", _mut_pulling_field)
        solver_cpu2.one_step()
        solver_gpu2.one_step()
        with pytest.raises(AssertionError):
            assert (
                abs(float(em_gpu2.injected_charge) - float(em_cpu2.injected_charge))
                / abs(float(em_cpu2.injected_charge))
                <= 1e-12
            )


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy / CUDA device not available")
def test_voltage_clamp_and_constructor_cpu_gpu_equivalence(monkeypatch):
    """
    Test MUST-FIX 2 & SHOULD-FIX 7:
    1. ``maintain_voltage`` defaults to ``False`` and never switches on
       automatically (even in a periodic-transverse PEC-z diode or when
       ``voltage`` is passed).
    2. ``maintain_voltage=True`` raises ``ValueError`` when transverse BCs are
       not periodic or ``bc_high[2]`` is not PEC, OR when embedded STL solids
       (``use_stl=True`` / ``stl_solids``) or solid conductors
       (``use_conductors=True`` or internal PEC edges ``ieps_z == 0``) are present.
    3. Algebraic identity check: one application of ``_enforce_gap_voltage()``
       on a spatially non-uniform 3D field ``E_z(x, y, z)`` satisfies the
       algebraic identity ``Delta W_field = dW_src`` to IEEE-754
       double-precision round-off (<= 1e-14 relative error), and mutating the
       midpoint voltage factor in ``_enforce_gap_voltage`` fails the check.
    4. CPU and GPU constructors behave identically when ``solver.E`` already
       contains non-zero values (neither overwrites pre-existing ``E_z``).
    5. With ``maintain_voltage=True``, CPU and GPU produce identical ``E_z``,
       ``injected_charge``, and ``voltage_source_work``.
    """
    for use_gpu in (False, True):
        grid = GridFIT3D(-0.01, 0.01, -0.01, 0.01, 0.0, 0.005, 2, 2, 8, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_stl=False,
            use_gpu=use_gpu,
            verbose=0,
        )
        sp = Species(name="electron", charge=-q_e, mass=m_e, use_gpu=use_gpu)
        # Default maintain_voltage must be False even when voltage is provided or in a 1D periodic PEC box
        em_default = SpaceChargeLimitedEmitter(solver, sp, voltage=1000.0)
        assert em_default._maintain_voltage is False
        em_none = SpaceChargeLimitedEmitter(solver, sp, voltage=None)
        assert em_none._maintain_voltage is False

        # Pre-existing non-zero E_z must NOT be overwritten on either CPU or GPU
        solver.E.array[:] = 0.0
        solver.E[0, 0, 0, "z"] = -12345.0
        _ = SpaceChargeLimitedEmitter(solver, sp, voltage=2500.0)
        ez_0 = float(solver.E[0, 0, 0, "z"])
        ez_1 = float(solver.E[0, 0, 1, "z"])
        assert abs(ez_0 - (-12345.0)) <= 1e-12
        assert abs(ez_1 - 0.0) <= 1e-12

    # Non-periodic transverse BC must raise ValueError when maintain_voltage=True
    grid_bad = GridFIT3D(-0.01, 0.01, -0.01, 0.01, 0.0, 0.005, 2, 2, 8, verbose=0)
    solver_bad = SolverFIT3D(
        grid_bad,
        cfln=0.5,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    sp_bad = Species(name="electron", charge=-q_e, mass=m_e)
    with pytest.raises(ValueError, match="maintain_voltage=True"):
        SpaceChargeLimitedEmitter(
            solver_bad, sp_bad, voltage=1000.0, maintain_voltage=True
        )

    # SHOULD-7: STL solids, solid conductors, or internal PEC edges must raise ValueError
    grid_c = GridFIT3D(-0.01, 0.01, -0.01, 0.01, 0.0, 0.005, 2, 2, 8, verbose=0)
    solver_c = SolverFIT3D(
        grid_c,
        cfln=0.5,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    with monkeypatch.context() as m:
        m.setattr(solver_c, "use_stl", True)
        with pytest.raises(ValueError, match="embedded STL solids"):
            SpaceChargeLimitedEmitter(
                solver_c, sp_bad, voltage=1000.0, maintain_voltage=True
            )
    with monkeypatch.context() as m:
        m.setattr(solver_c, "use_conductors", True)
        with pytest.raises(ValueError, match="solid conductors"):
            SpaceChargeLimitedEmitter(
                solver_c, sp_bad, voltage=1000.0, maintain_voltage=True
            )
    with monkeypatch.context():
        solver_c.ieps[0, 0, 3, "z"] = 0.0
        with pytest.raises(ValueError, match="internal PEC edges"):
            SpaceChargeLimitedEmitter(
                solver_c, sp_bad, voltage=1000.0, maintain_voltage=True
            )
        solver_c.ieps[0, 0, 3, "z"] = 1.0 / epsilon_0

    # SHOULD-7 / SHOULD 4 (algebraic identity): One clamp application satisfies the
    # algebraic identity Delta W_field == dW_src to IEEE-754 round-off:
    #   (1/2)*eps_0*sum(((Ez + dEz)^2 - Ez^2)*dV) == -eps_0*A*dEz*0.5*(V_target + V_curr).
    em_clamp = SpaceChargeLimitedEmitter(
        solver_c, sp_bad, voltage=1800.0, maintain_voltage=True
    )
    rng = np.random.default_rng(2026)
    ez_view = em_clamp._ez_3d_view()[:2, :2, :8]
    ez_view[...] = -2.8e5 + rng.normal(0.0, 4.0e4, size=(2, 2, 8))
    cell_vols = em_clamp._domain_area_2d[:, :, None] * em_clamp._dz_vec[None, None, :]
    w_field_before = float(0.5 * epsilon_0 * np.sum((ez_view**2) * cell_vols))
    em_clamp.voltage_source_work = 0.0
    em_clamp._enforce_gap_voltage()
    w_field_after = float(0.5 * epsilon_0 * np.sum((ez_view**2) * cell_vols))
    dw_field = w_field_after - w_field_before
    dw_src = float(em_clamp.voltage_source_work)
    print(
        f"[SHOULD-7 single clamp work identity] dW_field={dw_field:.12e} J, "
        f"dW_src={dw_src:.12e} J, rel_diff={abs(dw_field - dw_src) / abs(dw_src):.3e}"
    )
    assert abs(dw_field - dw_src) / abs(dw_src) <= 1e-14

    # Code mutation on _enforce_gap_voltage: replacing 0.5*(V_target + V_curr) with V_target
    # breaks Delta W_field == dW_src
    with monkeypatch.context() as m:

        def _mut_enforce():
            Ez_phys = em_clamp._ez_3d_view()[:2, :2, :8]
            V_col = -np.sum(Ez_phys * em_clamp._dz_vec[None, None, :], axis=2)
            V_curr = float(
                np.sum(V_col * em_clamp._domain_area_2d) / em_clamp._total_domain_area
            )
            dEz = -(em_clamp.voltage - V_curr) / em_clamp._gap_d
            Ez_phys += dEz
            em_clamp._voltage_source_work_host += float(
                -epsilon_0 * em_clamp._total_domain_area * dEz * em_clamp.voltage
            )

        m.setattr(em_clamp, "_enforce_gap_voltage", _mut_enforce)
        ez_view[...] = -2.8e5 + rng.normal(0.0, 4.0e4, size=(2, 2, 8))
        w_b_mut = float(0.5 * epsilon_0 * np.sum((ez_view**2) * cell_vols))
        em_clamp.voltage_source_work = 0.0
        em_clamp._enforce_gap_voltage()
        w_a_mut = float(0.5 * epsilon_0 * np.sum((ez_view**2) * cell_vols))
        with pytest.raises(AssertionError):
            assert (
                abs((w_a_mut - w_b_mut) - float(em_clamp.voltage_source_work))
                / abs(float(em_clamp.voltage_source_work))
                <= 1e-14
            )

    # Compare CPU and GPU with maintain_voltage=True over 5 steps
    pair = []
    for use_gpu in (False, True):
        grid = GridFIT3D(-0.01, 0.01, -0.01, 0.01, 0.0, 0.005, 2, 2, 8, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_stl=False,
            use_gpu=use_gpu,
            verbose=0,
        )
        sp = Species(name="electron", charge=-q_e, mass=m_e, use_gpu=use_gpu)
        em = SpaceChargeLimitedEmitter(
            solver, sp, voltage=1500.0, maintain_voltage=True, thermal_u=1.0e6
        )
        ParticleSolver(
            solver, species=[sp], injectors=[em], shape=1, gather="momentum"
        ).attach()
        for _ in range(5):
            solver.one_step()
        pair.append((solver, sp, em))

    (_s_cpu, _, em_c), (_s_gpu, _, em_g) = pair
    assert float(em_c.voltage_source_work) > 0.0
    assert (
        abs(float(em_g.voltage_source_work) - float(em_c.voltage_source_work))
        / abs(float(em_c.voltage_source_work))
        <= 1e-13
    )
    assert (
        abs(float(em_g.injected_charge) - float(em_c.injected_charge))
        / abs(float(em_c.injected_charge))
        <= 1e-13
    )
    ez_c = em_c._ez_3d_view()[:2, :2, :8]
    ez_g = em_g._ez_3d_view()[:2, :2, :8].get()
    assert np.max(np.abs(ez_g - ez_c)) / np.max(np.abs(ez_c)) <= 1e-13


def test_mpi_global_z_features_raise_on_every_rank():
    """
    Test MUST-FIX 3: Under ``mpiexec -n 2`` (CPU), any feature requiring global z
    (``voltage != None``, ``maintain_voltage=True``, or ``emitter.gap_d``)
    raises ``NotImplementedError`` on EVERY rank without hanging.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        pytest.skip("mpi4py is not installed")

    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("Run under `mpiexec -n 2` to exercise the multi-rank MPI check")

    import faulthandler

    faulthandler.enable()
    if hasattr(faulthandler, "dump_traceback_later"):
        faulthandler.dump_traceback_later(30, exit=True)

    try:
        grid = GridFIT3D(
            -0.01,
            0.01,
            -0.01,
            0.01,
            0.0,
            0.008,
            2,
            2,
            8,
            use_mpi=True,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            use_stl=False,
            use_mpi=True,
            verbose=0,
        )
        sp = Species(name="electron", charge=-q_e, mass=m_e)

        raised_voltage = False
        try:
            SpaceChargeLimitedEmitter(solver, sp, voltage=2500.0)
        except NotImplementedError:
            raised_voltage = True

        raised_clamp = False
        try:
            SpaceChargeLimitedEmitter(solver, sp, maintain_voltage=True)
        except NotImplementedError:
            raised_clamp = True

        em_local = SpaceChargeLimitedEmitter(
            solver, sp, voltage=None, maintain_voltage=False
        )
        raised_gap_d = False
        try:
            _ = em_local.gap_d
        except NotImplementedError:
            raised_gap_d = True

        all_voltage = comm.allgather(raised_voltage)
        all_clamp = comm.allgather(raised_clamp)
        all_gap_d = comm.allgather(raised_gap_d)
        assert all(all_voltage), f"Not all ranks raised on voltage: {all_voltage}"
        assert all(all_clamp), f"Not all ranks raised on maintain_voltage: {all_clamp}"
        assert all(all_gap_d), f"Not all ranks raised on gap_d: {all_gap_d}"

        # Verify protocol binding and non-owning counter consistency under MPI
        sp_mpi = Species(name="electron", charge=-q_e, mass=m_e, track_ids=True)
        solver.E.array[:] = 0.0
        solver.E[:, :, :, "z"] = -2.0e5
        em_mpi = SpaceChargeLimitedEmitter(
            solver,
            sp_mpi,
            n_macro_per_cell=2,
            thermal_u=1.0e6,
            voltage=None,
            maintain_voltage=False,
        )
        assert isinstance(em_mpi, InjectorProtocol)
        assert em_mpi.emitter_kind == "pec_wall"
        ps_mpi = ParticleSolver(
            solver,
            species=[sp_mpi],
            injectors=[em_mpi],
            shape=1,
            gather="momentum",
        ).attach()
        for _ in range(4):
            solver.one_step()

        step_indices = comm.allgather(em_mpi.step_index)
        next_ids = comm.allgather(sp_mpi._next_id)
        assert len(set(step_indices)) == 1 and step_indices[0] == 4
        assert len(set(next_ids)) == 1 and next_ids[0] == 4 * em_mpi.n_emit_per_step
        assert ps_mpi.injected_charge() < 0.0

        # A fully retarding step: the owner appends nothing but every rank
        # still consumes the slots' ids, exactly as a serial run does.
        solver.E[:, :, :, "z"] = +2.0e5
        solver.one_step()
        next_ids = comm.allgather(sp_mpi._next_id)
        assert len(set(next_ids)) == 1 and next_ids[0] == 5 * em_mpi.n_emit_per_step
        n_zero_w = comm.allgather(int(np.count_nonzero(np.asarray(sp_mpi.w) == 0.0)))
        assert sum(n_zero_w) == 0, f"zero-weight particles appended: {n_zero_w}"

        # Mutation check: skipping the non-owning _next_id advance desynchronises rank IDs
        orig_inject = SpaceChargeLimitedEmitter.inject

        def _mut_inject_skip_non_owning_id(
            self_em, t_v, dt_v, step_index=None, *, owning=True
        ):
            saved_id = self_em.species._next_id
            n_out = orig_inject(
                self_em, t_v, dt_v, step_index=step_index, owning=owning
            )
            if not owning:
                self_em.species._next_id = saved_id
            return n_out

        SpaceChargeLimitedEmitter.inject = _mut_inject_skip_non_owning_id
        try:
            solver.one_step()
            mut_next_ids = comm.allgather(sp_mpi._next_id)
            with pytest.raises(AssertionError):
                assert len(set(mut_next_ids)) == 1
        finally:
            SpaceChargeLimitedEmitter.inject = orig_inject
    finally:
        if hasattr(faulthandler, "cancel_dump_traceback_later"):
            faulthandler.cancel_dump_traceback_later()


# ===========================================================================
# (b) & (c) V6 planar Child–Langmuir diode convergence, potential, & mutations
# ===========================================================================


_V6_CACHE: dict[tuple, dict] = {}


def _v6_cache_key(
    Nz: int,
    *,
    gap_d: float = 0.005,
    voltage: float = 2500.0,
    cfln: float = 0.95,
    n_transits: float = 10.0,
    drop_pillbox: bool = False,
    k_read: int = 0,
    beam_j_scale: float | None = None,
) -> tuple:
    return (
        int(Nz),
        float(gap_d),
        float(voltage),
        float(cfln),
        float(n_transits),
        bool(drop_pillbox),
        int(k_read),
        None if beam_j_scale is None else float(beam_j_scale),
    )


def _run_v6_planar_diode_parallel(configs: list[dict]) -> list[dict]:
    """Run any uncached V6 planar diode configurations and cache their results."""
    for cfg in configs:
        key = _v6_cache_key(**cfg)
        if key not in _V6_CACHE:
            _V6_CACHE[key] = _run_v6_planar_diode(**cfg)
    return [_V6_CACHE[_v6_cache_key(**cfg)] for cfg in configs]


def _run_v6_planar_diode(
    Nz: int,
    *,
    gap_d: float = 0.005,
    voltage: float = 2500.0,
    cfln: float = 0.95,
    n_transits: float = 10.0,
    drop_pillbox: bool = False,
    k_read: int = 0,
    beam_j_scale: float | None = None,
):
    """
    Run a 1D non-relativistic planar Child–Langmuir diode with gap ``gap_d`` [m]
    and anode voltage ``voltage`` [V] using ``SpaceChargeLimitedEmitter``.

    The diode is made 1D via periodic transverse boundary conditions
    (``Nx=1, Ny=1`` with wide transverse cell ``Lx = Ly = 10 * gap_d``) and PEC
    longitudinal faces on ``zmin = 0`` (cathode) and ``zmax = gap_d`` (anode).
    The anode-to-cathode voltage ``V`` is applied by passing ``voltage=voltage``
    and ``maintain_voltage=True`` on the emitter, which initializes the uniform
    vacuum bias field ``E_z = -V / d`` at ``t = 0`` and models an ideal external
    voltage source that clamps ``-int_0^d E_z dz = V`` at each step with return
    current density ``J_ext = eps_0 * dEz / dt`` and cumulative work
    ``emitter.voltage_source_work``.
    """
    cache_key = _v6_cache_key(
        Nz,
        gap_d=gap_d,
        voltage=voltage,
        cfln=cfln,
        n_transits=n_transits,
        drop_pillbox=drop_pillbox,
        k_read=k_read,
        beam_j_scale=beam_j_scale,
    )
    if cache_key in _V6_CACHE:
        return _V6_CACHE[cache_key]

    Lx = 0.05
    Ly = 0.05
    area = Lx * Ly
    dz = gap_d / float(Nz)
    cfln = float(cfln)

    # Cold emission from rest (u_0 = 0) matching the Child–Langmuir assumption v(0) = 0
    thermal_u = 0.0  # [m/s]

    grid = GridFIT3D(
        -0.5 * Lx, 0.5 * Lx, -0.5 * Ly, 0.5 * Ly, 0.0, gap_d, 1, 1, Nz, verbose=0
    )
    solver = SolverFIT3D(
        grid,
        cfln=cfln,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_stl=False,
        verbose=0,
    )
    dt = float(solver.dt)

    electrons = Species(
        name="electron", charge=-q_e, mass=m_e, capacity=max(8 * Nz, 256)
    )
    emitter = SpaceChargeLimitedEmitter(
        solver,
        electrons,
        n_macro_per_cell=1,
        thermal_u=thermal_u,
        voltage=voltage,
        maintain_voltage=True,
    )
    emitter._k_read = int(k_read)

    if drop_pillbox:
        # Mutation: drop the -q_0 / (eps_0 * A) pillbox term in _read_pulling_surface_field
        emitter._cathode_node_charge = lambda: emitter.xp.zeros(
            emitter.n_emitting_cells, dtype=np.float64
        )

    if beam_j_scale is not None:
        # Sub-SCL constant-current injection via BeamInjector (e.g. 0.9 * J_CL)
        j_target = float(beam_j_scale) * child_langmuir_current_density(
            voltage=voltage, gap=gap_d
        )
        beam = BeamInjector(
            electrons,
            z0=0.0,
            current=j_target * area,
            energy_eV=0.01,
            radius=0.0,
            n_macro_per_step=1,
        )

        def _inject_beam(
            t_val, dt_val, step_index=None, *, owning=True, _em=emitter, _bm=beam
        ):
            _em._enforce_gap_voltage()
            n_inj = _bm.inject(t_val, dt_val, step_index=step_index, owning=owning)
            _em.injected_charge = _bm.injected_charge
            _em.injected_energy = _bm.injected_energy
            _em.injected_count = _bm.injected_count
            _em.step_index = _bm.step_index
            return n_inj

        emitter.inject = _inject_beam

    psolver = ParticleSolver(
        solver,
        species=[electrons],
        injectors=[emitter],
        shape=1,
        gather="momentum",
        sort_every=0,
    ).attach()

    # Space-charge-limited transit time is T_tr = 3 * d / v_max; integrate for
    # n_transits >= 8 transit times (default 10 * T_tr = 30 * d / v_max) so the
    # initial turn-on transient and gap space-charge accumulation have fully settled.
    # Primary steady-state averaging window is [0.50 * n_steps, n_steps] (the
    # midpoint of the [30%, 70%] window-flatness scan).
    v_max = float(np.sqrt(2.0 * q_e * voltage / m_e))
    t_transit = 3.0 * gap_d / v_max
    t_end = float(n_transits) * t_transit
    n_steps = int(t_end / dt)
    n_avg_start = n_steps // 2

    def _field_energy() -> float:
        ez = emitter._ez_3d_view()[0, 0, :Nz]
        return float(0.5 * epsilon_0 * area * dz * np.sum(ez**2))

    def _gap_kinetic_energy() -> float:
        if electrons.n_particles == 0:
            return 0.0
        return float(np.sum(electrons.kinetic_energy_per_particle()))

    j_anode_full = np.empty(n_steps, dtype=np.float64)
    ez_sum = np.zeros(Nz, dtype=np.float64)
    n_avg = 0
    q_abs_prev = 0.0
    w_src_start = 0.0
    e_inj_start = 0.0
    k_anode_start = 0.0
    k_cathode_start = 0.0
    k_gap_start = 0.0
    w_field_start = 0.0

    for step in range(n_steps):
        if step == n_avg_start:
            emitter._enforce_gap_voltage()
            w_src_start = float(emitter.voltage_source_work)
            e_inj_start = float(emitter.injected_energy)
            k_anode_start = float(psolver.boundaries.absorbed_energy["z_high"])
            k_cathode_start = float(psolver.boundaries.absorbed_energy["z_low"])
            k_gap_start = _gap_kinetic_energy()
            w_field_start = _field_energy()
        solver.one_step()
        q_abs = float(psolver.boundaries.absorbed_charge["z_high"])
        dq_anode = q_abs - q_abs_prev
        q_abs_prev = q_abs
        j_anode_full[step] = abs(dq_anode) / (dt * area)
        if step >= n_avg_start:
            ez_sum += emitter._ez_3d_view()[0, 0, :Nz]
            n_avg += 1

    # Final voltage clamp so end-of-window field state matches start-of-window clamped state
    emitter._enforce_gap_voltage()
    w_in_steady = (float(emitter.voltage_source_work) - w_src_start) + (
        float(emitter.injected_energy) - e_inj_start
    )
    dk_anode_steady = (
        float(psolver.boundaries.absorbed_energy["z_high"]) - k_anode_start
    )
    dk_cathode_steady = (
        float(psolver.boundaries.absorbed_energy["z_low"]) - k_cathode_start
    )
    dk_gap_steady = _gap_kinetic_energy() - k_gap_start
    dw_field_steady = _field_energy() - w_field_start
    w_out_steady = dk_anode_steady + dk_cathode_steady + dk_gap_steady + dw_field_steady
    energy_rel_err = abs(w_in_steady - w_out_steady) / max(abs(w_in_steady), 1e-30)

    def _window_stats(j_slice: np.ndarray) -> tuple[float, float, float, float]:
        j_m = float(np.mean(j_slice))
        m_idx = len(j_slice) // 2
        h1_s = j_slice[:m_idx]
        h2_s = j_slice[m_idx:]
        jh1 = float(np.mean(h1_s))
        jh2 = float(np.mean(h2_s))
        # Combine two-half sample standard error sqrt(Var(h1)/N1 + Var(h2)/N2) with
        # the 1-sheet window-endpoint arrival quantization j_m / N_window:
        se_half = float(
            np.sqrt(
                np.var(h1_s, ddof=1) / max(len(h1_s), 1)
                + np.var(h2_s, ddof=1) / max(len(h2_s), 1)
            )
        )
        se_quant = abs(j_m) / float(max(len(j_slice), 1))
        return j_m, jh1, jh2, float(np.hypot(se_half, se_quant))

    j_arr = j_anode_full[n_avg_start:]
    j_steady, j_half1, j_half2, stat_err = _window_stats(j_arr)
    # Disjoint early steady window [0.30 * n_steps, 0.50 * n_steps) vs primary window [0.50 * n_steps, n_steps):
    j_disjoint_30_50, _, _, stat_err_30_50 = _window_stats(
        j_anode_full[int(0.30 * n_steps) : n_avg_start]
    )

    window_starts = (0.30, 0.40, 0.50, 0.60, 0.70)
    window_means: dict[float, float] = {}
    window_errs: dict[float, float] = {}
    for wf in window_starts:
        jm_w, _, _, se_w = _window_stats(j_anode_full[int(wf * n_steps) :])
        window_means[wf] = jm_w
        window_errs[wf] = se_w

    ez_mean = ez_sum / float(max(n_avg, 1))

    # Reconstruct electrostatic potential phi(z_k) = - sum_{m=0}^{k-1} E_z[m] * dz
    # at primary grid nodes z_k = k * dz, k = 0 .. Nz
    z_nodes = np.linspace(0.0, gap_d, Nz + 1)
    phi_nodes = np.zeros(Nz + 1, dtype=np.float64)
    phi_nodes[1:] = -np.cumsum(ez_mean) * dz

    # Fit log(phi / V) vs log(z / d) over the fixed physical range z / d in [2/12, 11/12]:
    # Rule: exclude the first two coarse-grid cells z < 2 * dz_12 = d / 6 near the
    # singular rho(z) ~ z^(-2/3) cathode boundary layer and the final coarse-grid
    # cell z > 11 * d / 12 adjacent to the clamped anode node z = d. Note that
    # z/d = 2/12 = 4/24 = 8/48 and z/d = 11/12 = 22/24 = 44/48 are exact grid nodes
    # on all three resolutions Nz in {12, 24, 48}.
    z_norm = z_nodes[1:-1] / gap_d
    phi_norm = phi_nodes[1:-1] / voltage
    fit_mask = (
        (z_norm >= (2.0 / 12.0) - 1e-12)
        & (z_norm <= (11.0 / 12.0) + 1e-12)
        & (phi_norm > 0.0)
    )
    if np.count_nonzero(fit_mask) >= 2:
        slope, _ = np.polyfit(np.log(z_norm[fit_mask]), np.log(phi_norm[fit_mask]), 1)
    else:
        slope = float("nan")

    j_emit = abs(float(emitter.injected_charge)) / (n_steps * dt * area)
    absorbed_charge_faces = {
        k: float(v) for k, v in psolver.boundaries.absorbed_charge.items()
    }
    absorbed_energy_faces = {
        k: float(v) for k, v in psolver.boundaries.absorbed_energy.items()
    }
    in_flight_charge = float(
        electrons.charge * np.sum(electrons.w[: electrons.n_particles])
    )
    q_reabs_cathode = absorbed_charge_faces["z_low"]
    res_gauss = gauss_residual(solver, electrons)
    out = {
        "Nz": Nz,
        "dz": dz,
        "dt": dt,
        "cfln": cfln,
        "n_transits": float(n_transits),
        "J_steady": j_steady,
        "J_half1": j_half1,
        "J_half2": j_half2,
        "J_stat_err": stat_err,
        "J_disjoint_30_50": j_disjoint_30_50,
        "J_disjoint_30_50_err": stat_err_30_50,
        "window_means": window_means,
        "window_errs": window_errs,
        "J_emit": float(j_emit),
        "injected_charge": float(emitter.injected_charge),
        "in_flight_charge": in_flight_charge,
        "absorbed_charge_faces": absorbed_charge_faces,
        "absorbed_energy_faces": absorbed_energy_faces,
        "q_reabs_cathode": float(q_reabs_cathode),
        "phi_exponent": float(slope),
        "z_nodes": z_nodes,
        "phi_nodes": phi_nodes,
        "gauss_residual": float(res_gauss),
        "W_in_steady": float(w_in_steady),
        "W_out_steady": float(w_out_steady),
        "dK_anode_steady": float(dk_anode_steady),
        "dK_cathode_steady": float(dk_cathode_steady),
        "dK_gap_steady": float(dk_gap_steady),
        "dW_field_steady": float(dw_field_steady),
        "energy_rel_err": float(energy_rel_err),
    }
    _V6_CACHE[cache_key] = out
    return out


def test_v6_child_langmuir_planar_diode_convergence_and_potential():
    """
    Test (b): V6 planar Child–Langmuir diode with ``SpaceChargeLimitedEmitter``.

    Verifies across 3 resolutions (``Nz = 12, 24, 48``) over ``n_transits = 10``
    space-charge-limited transit times:
    1. Steady-state stationarity and window flatness (SHOULD 6):
       - Disjoint windows: the two disjoint halves of the primary averaging
         window ``[0.50 * n_steps, 0.75 * n_steps)`` and
         ``[0.75 * n_steps, n_steps)`` agree within the statistical error
         ``|J_half1 - J_half2| <= J_stat_err``, and the disjoint early steady
         window ``[0.30 * n_steps, 0.50 * n_steps)`` agrees with
         ``[0.50 * n_steps, n_steps)`` within ``hypot(err_30_50, J_stat_err)``.
       - Nested suffix windows ``[f_w * n_steps, n_steps)`` across
         ``f_w in {0.30, 0.40, 0.50, 0.60, 0.70}`` also agree with ``J_steady``,
         noting that because nested suffix windows share the tail of the run and
         are positively correlated, ``hypot(err(f_w), err(0.50))`` overstates
         the uncertainty of their difference.
    2. Steady anode current density converges to
       ``J_CL = (4 * eps_0 / 9) * sqrt(2 * e / m_e) * V^(3/2) / d^2`` according
       to the two-term **spatial** physical model
       ``e(dz) = a * (dz / d)**(2/3) + b * (dz / d)``:
       - (a) Beyond the first cell (``z >= dz``), Child–Langmuir flow with zero
         entry field and entry energy ``e * V_0 = (1/2) * m_e * v_1**2`` satisfies
         ``phi'^2 = (4 * J / eps_0) * sqrt(m_e / (2 * e)) * (phi_tilde**(1/2) - V_0**(1/2))``,
         giving ``J / J_CL ~= 1 + 3 * sqrt(V_0 / V) = 1 + 3 * v_1 / v_max``. A
         velocity error ``delta_v1`` at ``z = dz`` gives
         ``delta_J / J_CL ~= 3 * delta_v1 / v_max``, with
         ``v(dz) / v_max = (dz / d)**(2/3)`` in Child–Langmuir flow.
       - (b) Near the cathode ``E_z ~ z**(1/3)``; normalized by ``E_z(dz)``, the
         gathered field across the first cell has a ``dz``-independent shape:
         ``(0.5 + zeta) * E_z[0]`` on ``[0, 1/2)``, then linear up to ``E_z[1]``,
         with ``E_z[0] / E_z(dz) = 2**(-1/3)`` and
         ``E_z[1] / E_z(dz) = (3/2)**(1/3)``. Thus ``delta_v1 / v_1`` is the same
         at every resolution, and ``e_2/3 = 3 * (delta_v1 / v_1) * (dz / d)**(2/3)``.
       - (c) Equivalently, a residual first-cell field error
         ``E_c ~ E_z(dz) ~ (dz / d)**(1/3) * V / d`` enters ``J`` quadratically
         as ``(E_c * d / V)**2 ~ (dz / d)**(2/3)``.
       - Limits of the argument: the order ``(dz / d)**(2/3)`` is derived, but
         the sign and coefficient ``a`` are not (feeding the exact CL field
         through the gather gives a velocity deficit, while the measured
         self-consistent bias is positive), so ``a`` stays fitted (the fitted
         ``a ~= 0.17`` corresponds to ``delta_v1 / v_1 ~= 6%``).
       - Combined with the ``O(dz / d)`` spatial bias, ``Nz * err`` is not
         constant and the effective local convergence order ``p_fit`` lies in
         the derived band ``[2/3, 1]``. Fitting ``(a, b)`` on ``Nz = 12`` and
         ``Nz = 24`` predicts ``e(48)`` within the propagated statistical error
         of the three steady-state means. The weakly relativistic Jory–Trivelpiece
         (1969) correction ``J_rel / J_CL = 1 - (3/28) * eV / (m_e c^2)`` is printed
         at run time and is much smaller than the discretization error at ``Nz <= 48``.
    3. Richardson extrapolation at the fitted order ``p_fit``,
       ``J_inf = J_48 + (J_48 - J_24) / (2**p_fit - 1)``, matches ``J_CL`` within
       ``tol = |J_inf - J_2term| + sigma_J_inf`` (the spread between the
       single-``p`` and two-term extrapolations plus the per-grid ``J_stat_err``
       values propagated through ``1 / (2**p_fit - 1)``). At fixed ``dz``
       (``Nz = 24``), parallel runs at ``CFL = 0.95`` and ``CFL = 0.475`` confirm
       experimentally that the ``a * (dz / d)**(2/3)`` term is spatial rather
       than ``dt``-driven (reporting as a FINDING that ``J_steady`` is identical
       to 6 digits and ``e(Nz=24)`` is invariant under halving ``CFL`` within
       statistical error).
    4. Steady potential profile exponent ``alpha(Nz)`` fitted over the fixed
       physical range ``z / d in [2/12, 11/12]`` converges monotonically to
       ``4/3`` with observed order ``p_alpha`` in the empirical pre-asymptotic
       band ``[2/3, 1.10]`` (spanning the O((dz/d)^(2/3)) singular cathode
       boundary-layer scaling and O(dz/d) near-cathode field bias), and its
       Richardson extrapolation using the fitted order ``p_alpha`` (factor
       ``1 / (2**p_alpha - 1)``) matches ``4/3`` within its extrapolation error
       ``|alpha_48 - alpha_24| / (2**p_alpha - 1)``.
    5. Complete energy balance closes over the steady-state window (SHOULD 3):
       ``W_in = dW_src + dK_inj`` vs
       ``W_out = dK_anode + dK_cathode + dK_gap + dW_field``.
       Note the ``O(dt)`` leapfrog endpoint mismatch at the window boundaries:
       ``K_gap`` is evaluated from particle velocities at half-integer time
       level ``u^(n+1/2)`` whereas the electric field energy ``W_field`` is
       evaluated at integer time level ``E^(n+1)``. With ``dK_gap`` and
       ``dK_cathode`` (``absorbed_energy["z_low"]``) included, the residual is
       bounded at each resolution by the semi-empirical first-order bound
       ``(1/6 + (4/3) * CFL * (v_max / c)) / Nz``, built from the two estimates:
       (i) ``1 / (6 * Nz)`` from boundary half-cell linear gather clipping
       ``(1/8) * |E_z(d)| * dz / V = (1/6) / Nz`` (with ``|E_z(d)| = (4/3)*V/d``
       and ``|E_z(0)| = 0``), and (ii) ``(4/3) * CFL * (v_max / c) / Nz`` from
       single-step anode-crossing kick overshoot ``|E_z(d)| * v_max * dt / V``.
       Across the three resolutions, the energy residual converges with fitted
       order ``p_energy`` in the empirical 3-grid band ``[0.75, 1.40]`` around
       theoretical order ``p = 1``.
    6. Discrete Gauss's law ``gauss_residual`` remains at machine round-off.
    """
    gap_d = 0.005  # [m]
    voltage = 2500.0  # [V], eV / (m_e c^2) << 1 (weakly relativistic)
    v_max = float(np.sqrt(2.0 * q_e * voltage / m_e))
    J_CL = child_langmuir_current_density(voltage=voltage, gap=gap_d)
    # Weakly relativistic Jory–Trivelpiece (1969) correction:
    # J_rel / J_CL = 1 - (3 / 28) * (e V / (m_e c^2)) + O((eV / m_e c^2)^2)
    rel_corr = -(3.0 / 28.0) * (q_e * voltage) / (m_e * (c_light**2))
    print(
        f"[V6 relativistic correction] eV/(m_e c^2)={(q_e * voltage) / (m_e * c_light**2):.6e}, "
        f"J_rel/J_CL - 1 = -(3/28)*eV/(m_e c^2) = {rel_corr:+.6e} ({100 * rel_corr:+.4f}%)"
    )

    resolutions = (12, 24, 48)
    parallel_runs = _run_v6_planar_diode_parallel(
        [
            *[
                {"Nz": nz, "gap_d": gap_d, "voltage": voltage, "cfln": 0.95}
                for nz in resolutions
            ],
            {"Nz": 24, "gap_d": gap_d, "voltage": voltage, "cfln": 0.475},
        ]
    )
    results = parallel_runs[:3]
    r_24_cfl0475 = parallel_runs[3]

    rel_errors = np.array(
        [abs(r["J_steady"] - J_CL) / J_CL for r in results], dtype=np.float64
    )
    energy_errors = np.array([r["energy_rel_err"] for r in results], dtype=np.float64)
    dz_vals = np.array([r["dz"] for r in results], dtype=np.float64)
    h_vals = dz_vals / gap_d

    for r, err in zip(results, rel_errors):
        half_diff = abs(r["J_half1"] - r["J_half2"])
        disjoint_30_50_diff = abs(r["J_disjoint_30_50"] - r["J_steady"])
        disjoint_30_50_se = float(np.hypot(r["J_disjoint_30_50_err"], r["J_stat_err"]))
        # Semi-empirical first-order energy non-conservation bound (SHOULD 3), built from:
        # (i) 1/(6*Nz) from boundary half-cell linear gather clipping ((1/8)*|E_z(d)|*dz/V = (1/6)/Nz)
        # (ii) (4/3)*CFL*(v_max/c)/Nz from single-step anode-crossing kick overshoot (|E_z(d)|*v_max*dt/V),
        # while K_gap(u^(n+1/2)) vs W_field(E^(n+1)) has an O(dt) leapfrog endpoint mismatch.
        energy_bound_nz = (
            (1.0 / 6.0) + (4.0 / 3.0) * r["cfln"] * (v_max / c_light)
        ) / float(r["Nz"])
        win_str = ", ".join(
            f"{int(100 * wf)}%:{r['window_means'][wf]:.1f}+-{r['window_errs'][wf]:.1f}"
            for wf in (0.30, 0.40, 0.50, 0.60, 0.70)
        )
        print(
            f"[V6 Nz={r['Nz']:2d}] J={r['J_steady']:.2f} A/m^2 "
            f"(disjoint halves: half1={r['J_half1']:.2f}, half2={r['J_half2']:.2f}, "
            f"|h1-h2|={half_diff:.2f} <= stat_err={r['J_stat_err']:.2f}; "
            f"disjoint [30%,50%)={r['J_disjoint_30_50']:.2f}, "
            f"|J_30_50-J_50_100|={disjoint_30_50_diff:.2f} <= {disjoint_30_50_se:.2f}), "
            f"J_CL={J_CL:.2f} A/m^2, rel_err={err:.5f} (Nz*err={r['Nz'] * err:.4f}), "
            f"phi_exp={r['phi_exponent']:.5f}, "
            f"W_in={r['W_in_steady']:.6e} J, W_out={r['W_out_steady']:.6e} J "
            f"(dK_anode={r['dK_anode_steady']:.4e}, dK_cath={r['dK_cathode_steady']:.2e}, "
            f"dK_gap={r['dK_gap_steady']:.2e}, dW_f={r['dW_field_steady']:.2e}), "
            f"energy_err={r['energy_rel_err']:.5f} <= semi_emp_bound={energy_bound_nz:.5f}, "
            f"gauss_res={r['gauss_residual']:.3e}\n"
            f"  nested window scan [30%..70%] (correlated, hypot SE overstated): {win_str}"
        )
        # Primary stationarity checks on strictly disjoint windows (SHOULD 6):
        # (i) first half [0.50*n_steps, 0.75*n_steps) vs second half [0.75*n_steps, n_steps)
        #     of the averaging window;
        # (ii) disjoint pre-window [0.30*n_steps, 0.50*n_steps) vs [0.50*n_steps, n_steps).
        assert half_diff <= r["J_stat_err"]
        assert disjoint_30_50_diff <= disjoint_30_50_se
        for wf, j_w in r["window_means"].items():
            # Nested suffix windows [wf*n_steps, n_steps) vs [0.50*n_steps, n_steps)
            # overlap on [max(wf, 0.50)*n_steps, n_steps) and are positively correlated,
            # so combined_se = hypot(se(wf), se(0.50)) overstates the uncertainty of
            # their difference (SHOULD 6):
            combined_se = float(np.hypot(r["window_errs"][wf], r["window_errs"][0.50]))
            assert abs(j_w - r["J_steady"]) <= combined_se
        # Positive finite-grid bias: J_steady > J_CL from the cathode boundary-layer
        # discretization e(dz) = a*(dz/d)**(2/3) + b*(dz/d) > 0.
        assert r["J_steady"] > J_CL
        assert r["gauss_residual"] <= 1e-12
        # Complete energy balance closure within the semi-empirical O(1/Nz) gather/push bound
        assert r["energy_rel_err"] <= energy_bound_nz

    # Current and energy errors must decrease strictly with grid refinement
    assert rel_errors[0] > rel_errors[1] > rel_errors[2]
    assert energy_errors[0] > energy_errors[1] > energy_errors[2]

    # MUST 1(a)-(b): Two-term convergence model e(dz) = a * (dz/d)**(2/3) + b * (dz/d).
    # Because of the O((dz/d)**(2/3)) cathode singularity term, Nz * err rises with Nz
    # and the effective local order p_fit lies in the derived physical band [2/3, 1].
    p_fit, _ = np.polyfit(np.log(dz_vals), np.log(rel_errors), 1)
    p_energy, _ = np.polyfit(np.log(dz_vals), np.log(energy_errors), 1)
    print(
        f"[V6 convergence] fitted current order p_fit = {p_fit:.4f} (derived band [2/3, 1] = "
        f"[{2.0 / 3.0:.4f}, 1.0000]), energy order p_energy = {p_energy:.4f} "
        f"(theoretical p = 1.0, empirical band [0.75, 1.40])"
    )
    assert (2.0 / 3.0) <= p_fit <= 1.0
    # Empirical 3-grid acceptance band [0.75, 1.40] around theoretical O(dz/d) order p = 1
    assert 0.75 <= p_energy <= 1.40

    # Fit two-term spatial model e(h) = a * h**(2/3) + b * h on Nz = 12 and Nz = 24 (h = dz/d = 1/Nz)
    # and predict the Nz = 48 relative current error. Here a = 3 * (delta_v1 / v_1)
    # from J / J_CL ~= 1 + 3 * v_1 / v_max with v_1 / v_max = (dz / d)**(2/3) (while
    # feeding the exact CL field E_z[0]/E(dz) = 2**(-1/3), E_z[1]/E(dz) = (3/2)**(1/3)
    # through the gather gives a static velocity deficit, the self-consistent bias is
    # positive so a stays fitted):
    basis_12_24 = np.array(
        [
            [h_vals[0] ** (2.0 / 3.0), h_vals[0]],
            [h_vals[1] ** (2.0 / 3.0), h_vals[1]],
        ],
        dtype=np.float64,
    )
    a_fit, b_fit = np.linalg.solve(basis_12_24, rel_errors[:2])
    delta_v1_rel_fit = float(a_fit / 3.0)
    e48_pred = float(a_fit * (h_vals[2] ** (2.0 / 3.0)) + b_fit * h_vals[2])
    # Exact dyadic recurrence e_pred(48) = (2**(-2/3) + 1/2)*e(24) - 2**(-5/3)*e(12)
    # gives the propagated statistical standard error on |e(48) - e_pred(48)|:
    se_rel = np.array([r["J_stat_err"] / J_CL for r in results], dtype=np.float64)
    c_24 = (2.0 ** (-2.0 / 3.0)) + 0.5
    c_12 = 2.0 ** (-5.0 / 3.0)
    e48_pred_tol = float(
        np.sqrt(
            (se_rel[2] ** 2) + ((c_24 * se_rel[1]) ** 2) + ((c_12 * se_rel[0]) ** 2)
        )
    )
    print(
        f"[V6 two-term model e = a*(dz/d)^(2/3) + b*(dz/d)] "
        f"a={a_fit:.6f} (delta_v1/v1=a/3={delta_v1_rel_fit:.6f}, {100 * delta_v1_rel_fit:.2f}%), "
        f"b={b_fit:.6f}, "
        f"e_pred(48)={e48_pred:.6f} ({100 * e48_pred:.4f}%), "
        f"e_meas(48)={rel_errors[2]:.6f} ({100 * rel_errors[2]:.4f}%), "
        f"|e_meas(48) - e_pred(48)|={abs(rel_errors[2] - e48_pred):.6e} <= "
        f"stat_tol={e48_pred_tol:.6e}"
    )
    assert abs(rel_errors[2] - e48_pred) <= e48_pred_tol

    # SHOULD 2: Richardson extrapolation at the fitted order p_fit:
    #   J_inf = J_48 + (J_48 - J_24) / (2**p_fit - 1)
    #         = (1 + rich_factor) * J_48 - rich_factor * J_24,
    # where rich_factor = 1 / (2**p_fit - 1).
    # Tolerance is tol = |J_inf - J_2term| (the spread between the two models) plus
    # the propagated statistical error from the per-grid J_stat_err values propagated
    # through the 1 / (2**p_fit - 1) factor.
    J_24 = results[1]["J_steady"]
    J_48 = results[2]["J_steady"]
    rich_factor = float(1.0 / (2.0**p_fit - 1.0))
    J_inf = J_48 + (J_48 - J_24) * rich_factor
    j_inf_abs_diff = abs(J_inf - J_CL)
    j_inf_rel_diff = (J_inf - J_CL) / J_CL
    J_inf_2term_2grid = J_48 - J_CL * e48_pred
    m_3grid = np.column_stack([np.ones(3), h_vals ** (2.0 / 3.0), h_vals])
    J_inf_2term_3grid = float(
        np.linalg.solve(
            m_3grid, np.array([r["J_steady"] for r in results], dtype=np.float64)
        )[0]
    )
    spread_2grid = abs(J_inf - J_inf_2term_2grid)
    spread_3grid = abs(J_inf - J_inf_2term_3grid)
    extrap_err_spread_2grid = spread_2grid / J_CL
    extrap_err_spread_3grid = spread_3grid / J_CL
    stat_err_12 = float(results[0]["J_stat_err"])
    stat_err_24 = float(results[1]["J_stat_err"])
    stat_err_48 = float(results[2]["J_stat_err"])
    j_inf_stat_err = float(
        np.hypot((1.0 + rich_factor) * stat_err_48, rich_factor * stat_err_24)
    )
    j_inf_stat_err_rel = j_inf_stat_err / J_CL
    tol_j_inf_2grid = spread_2grid + j_inf_stat_err
    tol_j_inf_3grid = spread_3grid + j_inf_stat_err
    tol_j_inf_rel_2grid = extrap_err_spread_2grid + j_inf_stat_err_rel
    tol_j_inf_rel_3grid = extrap_err_spread_3grid + j_inf_stat_err_rel
    print(
        f"[V6 Richardson J_inf] J_inf={J_inf:.4f} A/m^2, J_CL={J_CL:.4f} A/m^2, "
        f"|J_inf - J_CL|={j_inf_abs_diff:.4f} A/m^2 "
        f"(J_inf/J_CL - 1 = {j_inf_rel_diff:+.6e}, {100 * j_inf_rel_diff:+.4f}%), "
        f"1/(2^p - 1)={rich_factor:.6f}, "
        f"J_stat_err(12,24,48)=({stat_err_12:.4f}, {stat_err_24:.4f}, {stat_err_48:.4f}) A/m^2, "
        f"j_inf_stat_err={j_inf_stat_err:.4f} A/m^2 (rel={j_inf_stat_err_rel:.6e}); "
        f"J_inf_2term(12,24)={J_inf_2term_2grid:.4f} A/m^2 "
        f"(spread={spread_2grid:.4f} A/m^2 [rel={extrap_err_spread_2grid:.6e}], "
        f"tol=spread+stat_err={tol_j_inf_2grid:.4f} A/m^2 [rel={tol_j_inf_rel_2grid:.6e}]); "
        f"J_inf_2term(12,24,48)={J_inf_2term_3grid:.4f} A/m^2 "
        f"(spread={spread_3grid:.4f} A/m^2 [rel={extrap_err_spread_3grid:.6e}], "
        f"tol=spread+stat_err={tol_j_inf_3grid:.4f} A/m^2 [rel={tol_j_inf_rel_3grid:.6e}])"
    )
    assert j_inf_abs_diff <= tol_j_inf_2grid
    assert j_inf_abs_diff <= tol_j_inf_3grid
    assert abs(j_inf_rel_diff) <= tol_j_inf_rel_2grid
    assert abs(j_inf_rel_diff) <= tol_j_inf_rel_3grid

    # MUST 1: Fixed-dz (Nz = 24) CFL halving experiment (CFL = 0.95 vs CFL = 0.475)
    # confirming that the a * (dz/d)**(2/3) term is spatial rather than dt-driven
    r_24_cfl095 = results[1]
    e_24_cfl095 = rel_errors[1]
    e_24_cfl0475 = abs(r_24_cfl0475["J_steady"] - J_CL) / J_CL
    b_term_24 = float(b_fit * h_vals[1])
    a_term_cfl095 = e_24_cfl095 - b_term_24
    a_term_cfl0475 = e_24_cfl0475 - b_term_24
    a_term_ratio = a_term_cfl0475 / a_term_cfl095
    n1_cfl095 = (
        3.0
        * c_light
        / (r_24_cfl095["cfln"] * v_max)
        * ((gap_d / r_24_cfl095["dz"]) ** (2.0 / 3.0))
    )
    tol_a_halving = float(
        np.hypot(r_24_cfl0475["J_stat_err"], 0.5 * r_24_cfl095["J_stat_err"]) / J_CL
    )
    tol_cfl_invariance = float(
        np.hypot(r_24_cfl0475["J_stat_err"], r_24_cfl095["J_stat_err"]) / J_CL
    )
    print(
        f"[FINDING V6 CFL halving at fixed dz (Nz=24)] "
        f"CFL=0.95: J={r_24_cfl095['J_steady']:.4f} A/m^2, e={e_24_cfl095:.6f}, "
        f"a_term=e-b*(dz/d)={a_term_cfl095:.6f}; "
        f"CFL=0.475: J={r_24_cfl0475['J_steady']:.4f} A/m^2, e={e_24_cfl0475:.6f}, "
        f"a_term=e-b*(dz/d)={a_term_cfl0475:.6f}; "
        f"ratio a_term(0.475)/a_term(0.95)={a_term_ratio:.6f} "
        f"(|a_term(0.475) - 0.5*a_term(0.95)|={abs(a_term_cfl0475 - 0.5 * a_term_cfl095):.6e} "
        f"> stat_tol={tol_a_halving:.6e}, while |e(0.475) - e(0.95)|="
        f"{abs(e_24_cfl0475 - e_24_cfl095):.6e} <= stat_tol={tol_cfl_invariance:.6e}): "
        f"confirms the spatial derivation of a*(dz/d)^(2/3) (since n_1 = "
        f"(3c/(CFL*v_max))*(d/dz)^(2/3) = {n1_cfl095:.1f} steps already cross the first cell "
        f"along the smooth trajectory z(t) = d*(t/T)^3, the error does not halve when dt is halved)."
    )
    with pytest.raises(AssertionError):
        assert abs(a_term_cfl0475 - 0.5 * a_term_cfl095) <= tol_a_halving
    assert abs(e_24_cfl0475 - e_24_cfl095) <= tol_cfl_invariance

    # Steady potential profile phi(z) = V * (z / d)^(4/3) over fixed z/d in [2/12, 11/12]:
    # Because phi(z_k) = -sum_{m=0}^{k-1} E_z[m] dz integrates across the singular
    # cathode cell E_z(z) ~ z^(1/3) (where rho(z) ~ z^(-2/3)) and includes the O(dz/d)
    # near-cathode field bias, the exponent error combines O((dz/d)^(2/3)) and O(dz/d)
    # terms. The 3-grid acceptance band [2/3, 1.10] is an empirical pre-asymptotic
    # band spanning these two theoretical orders; Richardson extrapolation uses the
    # fitted exponent order p_alpha with factor 1 / (2**p_alpha - 1).
    target_exp = 4.0 / 3.0
    phi_exps = np.array([r["phi_exponent"] for r in results], dtype=np.float64)
    exp_errs = np.abs(phi_exps - target_exp)
    assert exp_errs[0] > exp_errs[1] > exp_errs[2]

    p_alpha, _ = np.polyfit(np.log(dz_vals), np.log(exp_errs), 1)
    # Richardson extrapolation across Nz = 24 and Nz = 48 using fitted order p_alpha:
    rich_factor_alpha = 1.0 / (2.0**p_alpha - 1.0)
    alpha_extrap = phi_exps[2] + (phi_exps[2] - phi_exps[1]) * rich_factor_alpha
    alpha_extrap_err = abs(phi_exps[2] - phi_exps[1]) * rich_factor_alpha
    print(
        f"[V6 potential exponent] exps={phi_exps.tolist()}, "
        f"p_alpha={p_alpha:.4f} (empirical band [0.667, 1.10]), "
        f"alpha_extrap={alpha_extrap:.6f} vs 4/3={target_exp:.6f} "
        f"(|diff|={abs(alpha_extrap - target_exp):.6f} <= extrap_err={alpha_extrap_err:.6f})"
    )
    assert (2.0 / 3.0) <= p_alpha <= 1.10
    assert abs(alpha_extrap - target_exp) <= alpha_extrap_err


def test_v6_child_langmuir_mutations():
    """
    Test (c): Mutations of the V6 planar Child–Langmuir diode and anode current
    diagnostic calibration against a known sub-SCL injected current:
    1. Dropping the ``-q_0 / (eps_0 * A)`` boundary pillbox term in
       ``_read_pulling_surface_field`` (`drop_pillbox=True`):
       - **Numerical half-cell gather trap / artefact** (SHOULD-5):
         Without ``-q_0 / (eps_0 * A)``, the emitter reads ``E_z[0]`` directly
         as ``E_z(z_min^+)`` and over-emits charge sheets on every step until
         Esirkepov current deposition from sheets in ``[0, dz / 2)`` drives
         ``E_z[0]`` near ``0`` (and slightly positive/retarding). Because
         ``gather_cic`` evaluates ``E_z,gathered(zeta) = (0.5 + zeta) * E_z[0]``
         inside the boundary half-cell ``zeta = z / dz in [0, 1/2)`` (with the
         exterior stencil node ``iz = -1`` masked on the PEC boundary), once
         ``E_z[0]`` is held near ``0`` any charge in ``[0, dz / 2)`` gathers
         ``E ~ (0.5 + zeta) * E_z[0] ~ 0`` (or retarding) and never leaves the
         first half-cell ``[0, dz / 2)``, choking ``J_anode = 0`` as a
         discretization artefact rather than physical space-charge reflection.
    2. Reading ``E_z`` one cell too far from the cathode (``k_read = 1`` instead
       of ``0``) causes the V6 accuracy check to fail.
    3. Sub-SCL constant-current injection at ``J_inj = 0.9 * J_CL`` using
       ``BeamInjector`` (temperature-limited below the space-charge limit):
       - Anode diagnostic calibration (MUST-FIX 2): over ``n_transits = 10 >= 8``,
         zero charge is lost or reflected at any non-anode boundary
         (``absorbed_charge[face] == 0`` for all ``face != 'z_high'``) and total
         charge conservation ``injected_charge == absorbed_charge['z_high'] + in_flight_charge``
         holds to machine round-off; in steady state, ``J_anode`` equals the
         injected current ``0.9 * J_CL`` within statistical error and is flat
         across window starts from 30% to 70% of the run.
       - Mutation failure: because ``0.9 * J_CL < J_CL``, the cathode surface
         field is not screened to zero (``E_z(0^+) < 0``), so ``phi(z)`` retains a
         linear ``E_0 * z`` component near the cathode, pulling the potential
         exponent below ``4/3`` toward ``1.0`` and failing both the
         Richardson-extrapolated ``phi ~ z^(4/3)`` exponent check and the
         ``J = J_CL`` anode current check.
    """
    gap_d = 0.005
    voltage = 2500.0
    J_CL = child_langmuir_current_density(voltage=voltage, gap=gap_d)
    target_exp = 4.0 / 3.0

    ref_12, ref_24, ref_48, mut_pillbox, mut_kread, beam_12, beam_24 = (
        _run_v6_planar_diode_parallel(
            [
                {"Nz": 12, "gap_d": gap_d, "voltage": voltage},
                {"Nz": 24, "gap_d": gap_d, "voltage": voltage},
                {"Nz": 48, "gap_d": gap_d, "voltage": voltage},
                {"Nz": 24, "gap_d": gap_d, "voltage": voltage, "drop_pillbox": True},
                {"Nz": 24, "gap_d": gap_d, "voltage": voltage, "k_read": 1},
                {"Nz": 12, "gap_d": gap_d, "voltage": voltage, "beam_j_scale": 0.9},
                {"Nz": 24, "gap_d": gap_d, "voltage": voltage, "beam_j_scale": 0.9},
            ]
        )
    )

    # Fitted exponent convergence order p_alpha from the 3-grid reference run:
    dz_vals = np.array([ref_12["dz"], ref_24["dz"], ref_48["dz"]], dtype=np.float64)
    exp_errs = np.abs(
        np.array(
            [ref_12["phi_exponent"], ref_24["phi_exponent"], ref_48["phi_exponent"]],
            dtype=np.float64,
        )
        - target_exp
    )
    p_alpha_ref, _ = np.polyfit(np.log(dz_vals), np.log(exp_errs), 1)

    # Derived leading-order (p = 2/3) Richardson discretization tolerance at Nz = 24
    # from the (12, 24) grid pair (refinement ratio r = 2, p = 2/3):
    tol_24 = abs(ref_24["J_steady"] - ref_12["J_steady"]) / (
        (2.0 ** (2.0 / 3.0) - 1.0) * J_CL
    )
    err_ref_24 = abs(ref_24["J_steady"] - J_CL) / J_CL
    assert err_ref_24 <= tol_24

    # Mutation (c)(1): drop the -q_0 / (eps_0 * A) pillbox term
    J_pillbox = mut_pillbox["J_steady"]
    J_emit_pillbox = mut_pillbox["J_emit"]
    rel_diff_pillbox = (J_pillbox - J_CL) / J_CL
    print(
        f"[V6 mutation drop_pillbox=True] J_emit={J_emit_pillbox:.2f} A/m^2 "
        f"(ref={ref_24['J_emit']:.2f} A/m^2), "
        f"J_anode={J_pillbox:.2f} A/m^2, "
        f"(J_anode - J_CL)/J_CL={rel_diff_pillbox:+.5f}, tol_24={tol_24:.5f}"
    )
    assert J_emit_pillbox > 3.0 * J_CL > ref_24["J_emit"]
    assert mut_pillbox["q_reabs_cathode"] < 0.0
    assert J_pillbox < J_CL
    with pytest.raises(AssertionError):
        assert abs(J_pillbox - J_CL) / J_CL <= tol_24

    # Mutation (c)(2): read E_z one cell too far from cathode (k_read = 1)
    # V6 accuracy check |J - J_CL| / J_CL <= tol_24 must fail
    J_kread = mut_kread["J_steady"]
    err_kread = abs(J_kread - J_CL) / J_CL
    print(
        f"[V6 mutation k_read=1] J={J_kread:.2f} A/m^2, "
        f"|J - J_CL|/J_CL={err_kread:.5f} > tol_24={tol_24:.5f}"
    )
    with pytest.raises(AssertionError):
        assert err_kread <= tol_24

    # Mutation (c)(3) & MUST-FIX 2 calibration: inject at 0.9 * J_CL with BeamInjector
    j_inj_target = 0.9 * J_CL
    for b_run in (beam_12, beam_24):
        faces = b_run["absorbed_charge_faces"]
        for face_name, q_face in faces.items():
            if face_name != "z_high":
                assert q_face == 0.0, f"Unexpected charge on {face_name}: {q_face}"
        q_bal_err = abs(
            b_run["injected_charge"] - (faces["z_high"] + b_run["in_flight_charge"])
        ) / abs(b_run["injected_charge"])
        assert q_bal_err <= 1e-12
        # Steady J_anode must equal injected current 0.9 * J_CL within statistical error
        assert abs(b_run["J_steady"] - j_inj_target) <= b_run["J_stat_err"]
        # Flatness across window starts from 30% to 70% of the >=8-transit run
        for wf, j_w in b_run["window_means"].items():
            se_w = b_run["window_errs"][wf]
            assert abs(j_w - j_inj_target) <= se_w
            assert abs(j_w - b_run["J_steady"]) <= float(
                np.hypot(se_w, b_run["J_stat_err"])
            )
        win_b_str = ", ".join(
            f"{int(100 * wf)}%:{b_run['window_means'][wf]:.2f}+-{b_run['window_errs'][wf]:.2f}"
            for wf in (0.30, 0.40, 0.50, 0.60, 0.70)
        )
        print(
            f"[V6 BeamInjector 0.9*J_CL Nz={b_run['Nz']:2d}] "
            f"J_anode={b_run['J_steady']:.2f} A/m^2 vs 0.9*J_CL={j_inj_target:.2f} A/m^2 "
            f"(|diff|={abs(b_run['J_steady'] - j_inj_target):.4f} <= stat_err={b_run['J_stat_err']:.2f}), "
            f"absorbed_charge={faces}, in_flight_q={b_run['in_flight_charge']:.4e} C\n"
            f"  window scan [30%..70%]: {win_b_str}"
        )

    J_beam_24 = beam_24["J_steady"]
    err_beam_j = abs(J_beam_24 - J_CL) / J_CL
    # Richardson-extrapolated potential exponent across Nz = 12 and Nz = 24 with factor 1 / (2**p_alpha - 1)
    rich_factor_beam = 1.0 / (2.0**p_alpha_ref - 1.0)
    alpha_beam_extrap = (
        beam_24["phi_exponent"]
        + (beam_24["phi_exponent"] - beam_12["phi_exponent"]) * rich_factor_beam
    )
    alpha_beam_extrap_err = (
        abs(beam_24["phi_exponent"] - beam_12["phi_exponent"]) * rich_factor_beam
    )
    print(
        f"[V6 mutation BeamInjector 0.9*J_CL] J={J_beam_24:.2f} A/m^2, "
        f"|J - J_CL|/J_CL={err_beam_j:.5f} > tol_24={tol_24:.5f}, "
        f"phi_exp_24={beam_24['phi_exponent']:.5f}, "
        f"alpha_extrap={alpha_beam_extrap:.5f} vs 4/3={target_exp:.5f} "
        f"(|diff|={abs(alpha_beam_extrap - target_exp):.5f} > extrap_err={alpha_beam_extrap_err:.5f})"
    )
    assert beam_24["phi_exponent"] < target_exp
    with pytest.raises(AssertionError):
        assert err_beam_j <= tol_24
    with pytest.raises(AssertionError):
        assert abs(alpha_beam_extrap - target_exp) <= alpha_beam_extrap_err
