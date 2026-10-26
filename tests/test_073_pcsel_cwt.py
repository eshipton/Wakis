# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Tests for 3D coupled-wave theory of PCSELs (`wakis.pcsel_cwt`)."""

import numpy as np
import pytest

import wakis.pcsel_cwt as pcsel_cwt
from wakis.pcsel_cwt import (
    PCSELLayerStack,
    _exact_slab_dispersion_residual,
    compute_cwt_band_structure,
    compute_cwt_coupling_coefficients,
    double_lattice_coupling_scan,
    fourier_coefficients_phc,
    solve_finite_cwt,
    solve_infinite_cwt,
    solve_vertical_greens_functions,
    solve_vertical_slab_mode,
)


def test_vertical_slab_mode_and_mutation(monkeypatch):
    """Verify 1D vertical TE0 slab mode normalization, exact TMM agreement, and mutation failure."""
    stack = PCSELLayerStack(a=290e-9, r_over_a=0.18)

    def _assert_slab_valid(sl):
        z = sl["z"]
        theta0 = sl["theta0"]
        norm_sq = float(np.trapezoid(theta0**2, z))
        assert np.isclose(norm_sq, 1.0, atol=1e-6)
        assert stack.n_clad < sl["n_eff"] < stack.n_active
        assert np.isclose(sl["n_eff"], sl["n_eff_exact"], rtol=5e-4)
        assert 0.15 < sl["gamma_phc"] < 0.50
        assert 0.15 < sl["gamma_act"] < 0.55
        assert sl["gamma_phc"] + sl["gamma_act"] < 1.0

    slab = solve_vertical_slab_mode(stack, nz=2001)
    _assert_slab_valid(slab)

    beta0 = slab["beta0"]
    res_exact = abs(_exact_slab_dispersion_residual(slab["n_eff_exact"], beta0, stack, stack.eps_phc_avg))
    assert res_exact < 1e-4

    # Code-path mutation check: corrupting the TMM dispersion residual sign inside wakis.pcsel_cwt
    # causes solve_vertical_slab_mode to fail the FD-vs-TMM agreement check
    orig_res = pcsel_cwt._exact_slab_dispersion_residual
    monkeypatch.setattr(
        pcsel_cwt,
        "_exact_slab_dispersion_residual",
        lambda n_eff_val, b0, st, eps_phc: orig_res(n_eff_val * 1.01, b0, st, eps_phc),
    )
    slab_mut = solve_vertical_slab_mode(stack, nz=2001)
    with pytest.raises(AssertionError):
        _assert_slab_valid(slab_mut)


def test_fourier_coefficients_bessel_vs_fft_and_mutation(monkeypatch):
    """Compare analytical Bessel Fourier coefficients against 2D FFT for single and double lattices."""
    stack_single = PCSELLayerStack(a=290e-9, r_over_a=0.18)

    # Construct high-resolution 2D unit-cell permittivity map for FFT comparison
    n_grid = 256
    coords = -0.5 + (np.arange(n_grid) + 0.5) / n_grid
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    eps_map = np.where(
        xx**2 + yy**2 <= stack_single.r_over_a**2,
        stack_single.eps_hole,
        stack_single.eps_phc_host,
    )
    xi_fft = {}
    for m in range(-2, 3):
        px = np.exp(2j * np.pi * m * coords)
        for n in range(-2, 3):
            py = np.exp(2j * np.pi * n * coords)
            xi_fft[(m, n)] = complex(np.mean(eps_map * np.outer(px, py)))

    def _assert_bessel_matches_fft(xi_dict):
        for key in [(0, 0), (1, 0), (-1, 0), (1, 1), (2, 0), (0, 2)]:
            assert np.isclose(xi_dict[key], xi_fft[key], rtol=1.5e-2, atol=5e-3)
        assert np.isclose(np.imag(xi_dict[(1, 0)]), 0.0, atol=1e-14)
        assert np.real(xi_dict[(1, 0)]) < 0.0
        assert np.isclose(xi_dict[(1, 0)], xi_dict[(-1, 0)], atol=1e-14)
        assert np.isclose(xi_dict[(1, 0)], xi_dict[(0, 1)], atol=1e-14)
        assert np.isclose(xi_dict[(1, 1)], xi_dict[(-1, 1)], atol=1e-14)

    xi_bessel = fourier_coefficients_phc(stack_single, m_max=2)
    _assert_bessel_matches_fft(xi_bessel)

    stack_double = PCSELLayerStack(
        a=290e-9,
        r_over_a=0.18,
        r2_over_a=0.14,
        dx_over_a=0.25,
        dy_over_a=0.25,
    )
    xi_double = fourier_coefficients_phc(stack_double, m_max=2, align_radiative_center=True)
    assert np.isclose(np.imag(xi_double[(1, 0)]), 0.0, atol=1e-14)
    assert np.real(xi_double[(1, 0)]) < 0.0
    assert abs(np.imag(xi_double[(2, 0)])) > 0.02
    assert np.isclose(xi_double[(-2, 0)], np.conj(xi_double[(2, 0)]), atol=1e-14)

    # Code-path mutation check: dropping the factor of 2*pi inside j1 in fourier_coefficients_phc
    orig_j1 = pcsel_cwt.j1
    monkeypatch.setattr(pcsel_cwt, "j1", lambda x: orig_j1(0.5 * x))
    xi_mut = fourier_coefficients_phc(stack_single, m_max=2)
    with pytest.raises(AssertionError):
        _assert_bessel_matches_fft(xi_mut)


def test_vertical_greens_optical_theorem_and_mutation(monkeypatch):
    """Verify the 1D vertical radiative Green's function satisfies the optical theorem."""
    stack = PCSELLayerStack(a=290e-9, r_over_a=0.18)
    slab = solve_vertical_slab_mode(stack, nz=2001)

    def _assert_optical_theorem(gr):
        lhs = gr["optical_theorem_lhs"]
        rhs = gr["optical_theorem_rhs"]
        assert lhs > 0.0
        assert np.isclose(lhs, rhs, rtol=2e-3)

    greens = solve_vertical_greens_functions(stack, slab=slab)
    _assert_optical_theorem(greens)

    # Code-path mutation check: conjugating the tridiagonal matrix in solve_banded
    # flips the Sommerfeld outgoing radiation BC (-2j -> +2j) and violates the optical theorem
    orig_solve_banded = pcsel_cwt.solve_banded
    monkeypatch.setattr(
        pcsel_cwt,
        "solve_banded",
        lambda l_and_u, ab, b: orig_solve_banded(l_and_u, np.conj(ab), b),
    )
    greens_mut = solve_vertical_greens_functions(stack, slab=slab)
    with pytest.raises(AssertionError):
        _assert_optical_theorem(greens_mut)


def test_infinite_cwt_modes_symmetries_and_mutation(monkeypatch):
    """Verify Gamma-point band-edge mode ordering, independent closed-form C_2D limit, longitudinal zeta_mn, and BICs."""
    from scipy.special import j1 as scipy_j1

    stack = PCSELLayerStack(a=290e-9, r_over_a=0.18)
    cwt = compute_cwt_coupling_coefficients(stack, include_longitudinal=False)
    cwt_long = compute_cwt_coupling_coefficients(
        stack, slab=cwt["slab"], greens=cwt["greens"], include_longitudinal=True
    )

    # Noda convention: kappa_1d = -(k0^2/(2*beta0))*Gamma_PhC*xi_{2,0} > 0 for centered air hole
    assert np.real(cwt["kappa_1d"]) > 1000.0
    assert np.imag(cwt["C_rad"][0, 1]) > 0.0

    # Independent closed-form analytical benchmark for C_2D in the thick-PhC limit (t_PhC >> a):
    # As t_PhC -> infty, Gamma_PhC -> 1, k0^2 * eps_avg -> beta0^2, and the s = m^2+n^2 = 2
    # vertical Green's function inside the PhC layer approaches I_{1,1} -> 1 / ((2 - 1) * beta0^2),
    # yielding the closed-form 2D coupling coefficient (in cm^-1):
    #   kappa_{2D, s=2}^{closed} = (pi / (2 * a * eps_avg^2)) * [xi(1) - xi(sqrt(5))]^2
    stack_thick = PCSELLayerStack(
        a=290e-9,
        r_over_a=0.18,
        n_active=float(np.sqrt(stack.eps_phc_avg)),
        n_clad=3.20,
        z_phc_min=-800e-9,
        z_phc_max=800e-9,
        z_act_min=-810e-9,
        z_act_max=-800e-9,
        z_min=-1600e-9,
        z_max=1600e-9,
    )
    f_fill = np.pi * stack_thick.r_over_a**2
    delta_eps = stack_thick.eps_hole - stack_thick.eps_phc_host

    def _xi_bessel_closed(q_norm: float) -> float:
        arg = 2.0 * np.pi * q_norm * stack_thick.r_over_a
        return float(delta_eps * 2.0 * f_fill * scipy_j1(arg) / arg)

    xi_1 = _xi_bessel_closed(1.0)
    xi_sqrt5 = _xi_bessel_closed(np.sqrt(5.0))
    kappa_2d_closed_cm = (
        (np.pi / (2.0 * stack_thick.a * stack_thick.eps_phc_avg**2)) * (xi_1 - xi_sqrt5) ** 2 * 1e-2
    )
    cwt_thick_m1 = compute_cwt_coupling_coefficients(stack_thick, m_max=1, include_longitudinal=False)
    assert np.isclose(np.real(cwt_thick_m1["kappa_2d"]), kappa_2d_closed_cm, rtol=1.5e-2)

    def _assert_infinite_cwt_valid(res_dict, cwt_dict, expected_df_ab_thz):
        f_a = res_dict["A"]["freq_thz"]
        f_b = res_dict["B"]["freq_thz"]
        f_c = res_dict["C"]["freq_thz"]
        f_d = res_dict["D"]["freq_thz"]
        assert f_a < f_b < f_c
        df_ab = f_b - f_a
        # Exact C4v eigenvalue splitting identity: Delta_f_AB = 4 * Re(kappa_2d) * f_Bragg / beta0
        beta0_cm = cwt_dict["slab"]["beta0"] * 1e-2
        f_bragg_thz = cwt_dict["slab"]["f_bragg"] * 1e-12
        exact_df_ab_thz = 4.0 * float(np.real(cwt_dict["kappa_2d"])) * f_bragg_thz / beta0_cm
        assert np.isclose(df_ab, exact_df_ab_thz, rtol=1e-12)
        assert np.isclose(df_ab, expected_df_ab_thz, rtol=1e-2)
        assert np.isclose(f_c, f_d, rtol=1e-13)
        assert abs(res_dict["A"]["alpha_v_cm"]) < 1e-10
        assert abs(res_dict["B"]["alpha_v_cm"]) < 1e-10
        assert np.isinf(res_dict["A"]["Q"])
        assert np.isinf(res_dict["B"]["Q"])
        assert res_dict["C"]["alpha_v_cm"] > 100.0
        assert 300.0 < res_dict["C"]["Q"] < 1000.0
        assert np.isclose(res_dict["C"]["Q"], res_dict["D"]["Q"], rtol=1e-12)

    res = solve_infinite_cwt(stack, cwt_data=cwt)
    _assert_infinite_cwt_valid(res, cwt, expected_df_ab_thz=0.2348)

    # Including the ad hoc local depolarisation term proportional to xi*xi / eps_avg
    # increases kappa_2d from 40.89 cm^-1 to 185.39 cm^-1 and Delta_f_AB from 0.2348 THz to 1.0648 THz
    res_long = solve_infinite_cwt(stack, cwt_data=cwt_long)
    assert np.isclose(np.real(cwt_long["kappa_2d"]), 185.388, rtol=1e-2)
    _assert_infinite_cwt_valid(res_long, cwt_long, expected_df_ab_thz=1.0648)

    # Away from Gamma along Gamma-X (in the perturbative regime k_red = 0.001, 0.002),
    # BIC symmetry is broken and radiative loss scales quadratically (~ k^2, so Q ~ 1 / k^2)
    bands_gx = compute_cwt_band_structure(stack, np.array([0.0, 0.001, 0.002]), direction="GX", cwt_data=cwt)
    assert np.sum(bands_gx["alpha_v_cm"][1, :2]) > 0.05
    ratio_alpha = np.sum(bands_gx["alpha_v_cm"][2, :2]) / np.sum(bands_gx["alpha_v_cm"][1, :2])
    assert 3.8 < ratio_alpha < 4.2

    # Code-path mutation check 1: flipping the sign of chi_rad in solve_vertical_greens_functions
    # produces negative alpha_v_cm for Modes C/D (never clipped to 0) and fails the physical check
    orig_greens = pcsel_cwt.solve_vertical_greens_functions

    def _mutated_greens_sign(st, slab=None, m_max=3):
        g = orig_greens(st, slab=slab, m_max=m_max)
        g["chi_rad"] = -g["chi_rad"]
        return g

    monkeypatch.setattr(pcsel_cwt, "solve_vertical_greens_functions", _mutated_greens_sign)
    cwt_mut_sign = compute_cwt_coupling_coefficients(stack)
    res_mut_sign = solve_infinite_cwt(stack, cwt_data=cwt_mut_sign)
    assert res_mut_sign["C"]["alpha_v_cm"] < -100.0
    with pytest.raises(AssertionError):
        _assert_infinite_cwt_valid(res_mut_sign, cwt_mut_sign, expected_df_ab_thz=0.2348)

    # Code-path mutation check 2: scaling chi_evan by 0.5 in solve_vertical_greens_functions
    # halves C_2D and fails both the closed-form thick-PhC check and the tightened Delta_f_AB tolerance
    def _mutated_greens_half_2d(st, slab=None, m_max=3):
        g = orig_greens(st, slab=slab, m_max=m_max)
        g["chi_evan"] = {k: 0.5 * v for k, v in g["chi_evan"].items()}
        return g

    monkeypatch.setattr(pcsel_cwt, "solve_vertical_greens_functions", _mutated_greens_half_2d)
    cwt_mut_2d = compute_cwt_coupling_coefficients(stack)
    res_mut_2d = solve_infinite_cwt(stack, cwt_data=cwt_mut_2d)
    with pytest.raises(AssertionError):
        _assert_infinite_cwt_valid(res_mut_2d, cwt_mut_2d, expected_df_ab_thz=0.2348)
    monkeypatch.setattr(pcsel_cwt, "solve_vertical_greens_functions", orig_greens)

    # Code-path mutation check 3: mutating `PCSELLayerStack.eps_phc_avg` (used by
    # `scale_long = pref / stack.eps_phc_avg` inside `compute_cwt_coupling_coefficients`
    # when `include_longitudinal=True`) by flipping its sign corrupts the ad hoc
    # depolarisation term `c_2d_long` and causes the Delta_f_AB check (1.0648 THz) to fail
    orig_eps_avg = stack.eps_phc_avg
    monkeypatch.setattr(PCSELLayerStack, "eps_phc_avg", property(lambda self: -orig_eps_avg))
    cwt_long_mut = compute_cwt_coupling_coefficients(
        stack, slab=cwt["slab"], greens=cwt["greens"], include_longitudinal=True
    )
    res_long_mut = solve_infinite_cwt(stack, cwt_data=cwt_long_mut)
    with pytest.raises(AssertionError):
        _assert_infinite_cwt_valid(res_long_mut, cwt_long_mut, expected_df_ab_thz=1.0648)


def test_double_lattice_hermitian_nonhermitian_cancellation_and_mutation(monkeypatch):
    """Verify double-lattice continuous root cancellation of C12/C21 and self-consistent slab update."""
    base_stack = PCSELLayerStack(a=290e-9, r_over_a=0.18)
    cwt_single = compute_cwt_coupling_coefficients(base_stack)
    c21_single = abs(cwt_single["C21_cm"])
    assert c21_single > 500.0

    d_grid = np.linspace(0.24, 0.29, 11)
    r_grid = np.linspace(0.75, 0.95, 11)

    def _assert_dl_cancellation(sc):
        assert 0.25 <= sc["opt_d_over_a"] <= 0.285
        assert 0.80 <= sc["opt_r2_over_r1"] <= 0.94
        assert sc["min_abs_C21_cm"] < 1e-6
        assert 10.0 < sc["opt_alpha_v_inf_cm"] < 25.0

    scan = double_lattice_coupling_scan(base_stack, d_grid, r_grid)
    _assert_dl_cancellation(scan)

    # Code-path mutation check: corrupting the phase of xi_{(2,0)} in fourier_coefficients_phc
    # destroys the Hermitian/non-Hermitian cancellation root in double_lattice_coupling_scan
    orig_fc = pcsel_cwt.fourier_coefficients_phc

    def _mutated_fc(st, m_max=3, eps_2d=None, align_radiative_center=True):
        xi_d = orig_fc(st, m_max=m_max, eps_2d=eps_2d, align_radiative_center=align_radiative_center)
        xi_d[(2, 0)] = abs(xi_d[(2, 0)]) + 0.0j
        xi_d[(-2, 0)] = abs(xi_d[(-2, 0)]) + 0.0j
        return xi_d

    monkeypatch.setattr(pcsel_cwt, "fourier_coefficients_phc", _mutated_fc)
    scan_mut = double_lattice_coupling_scan(base_stack, d_grid, r_grid)
    with pytest.raises(AssertionError):
        _assert_dl_cancellation(scan_mut)


def test_finite_cwt_power_balance_and_scaling_and_mutation(monkeypatch):
    """Verify 2D GLL finite-size CWT power balance identity and fair self-consistent single-lattice baseline."""
    base_stack = PCSELLayerStack(a=290e-9, r_over_a=0.18)
    cwt_single = compute_cwt_coupling_coefficients(base_stack)

    def _assert_power_balance(fin_res):
        assert fin_res["power_balance_error"] < 1e-10
        assert fin_res["lowest_alpha_th"] > 0.0

    res_100 = solve_finite_cwt(cwt_single["C_cm"], length=100e-4, n_poly=6)
    res_200 = solve_finite_cwt(cwt_single["C_cm"], length=200e-4, n_poly=6)
    _assert_power_balance(res_100)
    _assert_power_balance(res_200)

    # Larger cavity has smaller in-plane edge loss and smaller threshold gain
    assert res_200["fundamental_A"]["alpha_par"] < res_100["fundamental_A"]["alpha_par"]
    assert res_200["lowest_alpha_th"] < res_100["lowest_alpha_th"]

    # Fair self-consistent comparison at matched infinite-crystal vertical radiation loss alpha_v0:
    d_grid = np.linspace(0.25, 0.285, 9)
    r_grid = np.linspace(0.75, 0.92, 9)
    scan = double_lattice_coupling_scan(base_stack, d_grid, r_grid)
    cwt_double = scan["opt_cwt"]
    alpha_v0 = scan["opt_alpha_v_inf_cm"]

    # Asymmetric single-hole lattice with Hermitian Bragg phase in C_1D giving the exact same
    # infinite-crystal alpha_v0 from C_rad (same alpha_v definition, self-consistent Delta_alpha_th):
    cwt_single_asym = compute_cwt_coupling_coefficients(
        base_stack, slab=cwt_single["slab"], greens=cwt_single["greens"], bragg_phase_rad=0.4242156146
    )
    inf_single_asym = solve_infinite_cwt(base_stack, cwt_data=cwt_single_asym)
    assert np.isclose(inf_single_asym["A"]["alpha_v_cm"], alpha_v0, rtol=1e-4)

    # At L = 50 um (below crossover L_cross ~ 149 um), double lattice has lower relative margin than single lattice:
    fin_single_50um = solve_finite_cwt(cwt_single_asym["C_cm"], length=50e-4, n_poly=6)
    fin_double_50um = solve_finite_cwt(cwt_double["C_cm"], length=50e-4, n_poly=6)
    _assert_power_balance(fin_single_50um)
    _assert_power_balance(fin_double_50um)
    rel_single_50 = fin_single_50um["delta_alpha_th"] / fin_single_50um["lowest_alpha_th"]
    rel_double_50 = fin_double_50um["delta_alpha_th"] / fin_double_50um["lowest_alpha_th"]
    assert rel_double_50 < rel_single_50

    # At L = 1000 um (1 mm, above crossover), double lattice relative margin exceeds asymmetric single lattice by > 50x:
    fin_single_1mm = solve_finite_cwt(cwt_single_asym["C_cm"], length=1000e-4, n_poly=6)
    fin_double_1mm = solve_finite_cwt(cwt_double["C_cm"], length=1000e-4, n_poly=6)
    _assert_power_balance(fin_single_1mm)
    _assert_power_balance(fin_double_1mm)
    rel_margin_single = fin_single_1mm["delta_alpha_th"] / fin_single_1mm["lowest_alpha_th"]
    rel_margin_double = fin_double_1mm["delta_alpha_th"] / fin_double_1mm["lowest_alpha_th"]
    assert rel_margin_double > 50.0 * rel_margin_single
    assert fin_double_1mm["delta_alpha_th"] > 50.0 * fin_single_1mm["delta_alpha_th"]

    # Mutation check 1: tau_sat < 0.5 violates SBP upwind stability and raises ValueError
    with pytest.raises(ValueError, match="tau_sat"):
        solve_finite_cwt(cwt_single["C_cm"], length=100e-4, n_poly=6, tau_sat=0.1)

    # Code-path mutation check 2: perturbing GLL quadrature weights in _gll_differentiation_1d
    # breaks the discrete Summation-By-Parts identity W D + D^T W = e_N e_N^T - e_0 e_0^T
    # and causes the Poynting power balance check to fail
    orig_gll = pcsel_cwt._gll_differentiation_1d

    def _mutated_gll(n_p, length_val):
        x_v, d_v, w_v = orig_gll(n_p, length_val)
        w_mut = w_v.copy()
        w_mut[0] *= 1.5
        return x_v, d_v, w_mut

    monkeypatch.setattr(pcsel_cwt, "_gll_differentiation_1d", _mutated_gll)
    res_mut = solve_finite_cwt(cwt_single["C_cm"], length=100e-4, n_poly=6)
    with pytest.raises(AssertionError):
        _assert_power_balance(res_mut)
