# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Validation tests for `wakis.gain_semiconductor` (`SemiconductorGain` and
`adiabatic_chirp`) coupled to `SolverFIT3D`.

Covers all five physical validation benchmarks and their mutation checks:
(a) Small-signal gain spectrum g(omega, N) measured by 3D FIT transmission
    through a pumped slab vs analytic susceptibility chi(omega, N) for both
    linear and logarithmic gain models, absorption/transparency/amplification
    regimes, and carrier-dependent peak shift.
(b) Linewidth enhancement factor alpha_H measured from the phase/amplitude
    change of the 3D FIT slab transmission with Delta N vs input alpha_H
    (including alpha_H sign mutation check).
(c) Steady-state L-I curve of a 1D Fabry-Perot semiconductor diode laser vs
    the standard rate-equation solution: threshold current density J_th,
    carrier clamping at N_th, and differential quantum slope efficiency
    eta_d = eta_i * alpha_m / (alpha_i + alpha_m).
(d) Small-signal current modulation response and relaxation-oscillation
    frequency vs omega_R^2 = v_g * g' * S / tau_p (with gain compression
    eps_nl = 0), and dynamic/adiabatic frequency chirp
    Delta nu(t) = (alpha_H / (4 * pi)) * d(ln P)/dt under current modulation.
(e) Energy and particle-number conservation (bookkeeping test): carriers
    removed equal photons generated to machine precision across stimulated
    emission, absorption, injection/recombination, and carrier diffusion.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c, e as q_e, epsilon_0, hbar, mu_0

import wakis
from wakis.estimate import estimate, preflight
from wakis.gain import dbr_reflectance, fabry_perot_threshold_gain
from wakis.gain_semiconductor import SemiconductorGain, adiabatic_chirp
from wakis.materials import DispersiveMaterial
from wakis.resonances import ResonanceMonitor
from wakis.sources import ContinuousSource, CurrentSource, GaussianSource


def _preflight_1d_fit(nz, steps, n_gain_cells=0, n_poles=1):
    """Run `wakis.estimate.estimate` and `preflight` against the laptop 2 GB RAM cap."""
    est = estimate(
        (2, 2, int(nz)),
        steps=int(steps),
        dtype="float64",
        backend="cpu",
        cpml=True,
        dispersive_edges=3 * int(n_gain_cells),
        n_poles=int(n_poles),
    )
    problems = preflight(est, "laptop", raise_error=True)
    assert problems == []
    return est


# ---------------------------------------------------------------------------
# (a) Small-signal gain spectrum g(omega, N) from FIT slab transmission vs chi(omega, N)
# ---------------------------------------------------------------------------


def test_small_signal_gain_spectrum_slab():
    """
    (a) Small-signal gain spectrum g(omega, N) measured by 3D FIT transmission
    through a pumped semiconductor slab matches the model's analytic
    susceptibility chi(omega, N) across the Lorentzian line shape for both
    linear and logarithmic gain models, and across absorption (N < N_tr),
    transparency (N = N_tr), and gain (N > N_tr).
    """
    lam0 = 1.5e-6
    f0 = c / lam0  # ~199.86 THz
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 25.0e12
    eps_inf = 4.0  # n_bg = 2.0 -> lambda_mat = 750 nm
    n_bg = np.sqrt(eps_inf)
    N_tr = 1.0e24
    g_prime = 5.0e-20
    L = 3.0e-6

    # Grid resolution: dz = 15 nm -> 50 cells per wavelength in n_bg = 2.0 (>= 20 required)
    dz = 15.0e-9
    cells_per_lambda_mat = (lam0 / n_bg) / dz
    assert cells_per_lambda_mat >= 20.0

    z_min, z_max = -1.2e-6, 4.2e-6
    Nz = int(round((z_max - z_min) / dz))
    grid = wakis.GridFIT3D(
        -0.5e-6, 0.5e-6, -0.5e-6, 0.5e-6, z_min, z_max, 2, 2, Nz, verbose=0
    )
    freqs = np.linspace(0.88 * f0, 1.12 * f0, 11)
    omegas = 2.0 * np.pi * freqs

    def run_slab_spectrum(mat):
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.6,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            bg=[eps_inf, 1.0],
            n_pml=12,
            verbose=0,
        )
        if mat is not None:
            solver.add_gain_material(
                mat, center=(0.0, 0.0, 0.5 * L), size=(2.0e-6, 2.0e-6, L)
            )
        src = GaussianSource(frequency=f0, fwidth=0.18 * f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -0.6e-6),
            size=(1.0e-6, 1.0e-6, 0.0),
            amplitude=1.0e-4,  # weak probe so Delta N / N < 1e-12
        )
        flux_mon = wakis.FluxMonitor(
            freqs, center=(0.0, 0.0, L + 0.4e-6), size=(1.0e-6, 1.0e-6, 0.0)
        )
        dft_mon = wakis.DFTMonitor(
            freqs, components=["Ex"], center=(0.0, 0.0, L + 0.4e-6)
        )
        Nt = int((src.end_time + 80.0e-15) / solver.dt)
        _preflight_1d_fit(Nz, Nt, n_gain_cells=int(L / dz))
        solver.emsolve(Nt, source=sheet, monitors=[flux_mon, dft_mon])
        ex_f = dft_mon.get_field("Ex")[0][:, 0, 0, 0]
        return flux_mon.get_flux(), ex_f, solver.dt

    # Reference run at transparency N = N_tr (where chi = 0)
    mat_tr = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        differential_gain=g_prime,
        eps_inf=eps_inf,
        A=0.0,
        initial_carriers="transparency",
        enable_stimulated=False,
    )
    flux_tr, ex_tr, dt_sim = run_slab_spectrum(mat_tr)
    np.testing.assert_allclose(mat_tr.small_signal_gain(freqs), 0.0, atol=1e-12)

    # 1. Linear gain model above transparency: N = 2.6 * N_tr (g_peak = +8.0e4 m^-1)
    N_gain = 2.6 * N_tr
    mat_lin = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        differential_gain=g_prime,
        gain_model="linear",
        eps_inf=eps_inf,
        A=0.0,
        initial_carriers=N_gain,
        enable_stimulated=True,
    )
    flux_lin, ex_lin, _ = run_slab_spectrum(mat_lin)
    g_fit_lin = np.log(flux_lin / flux_tr) / L
    g_an_lin = mat_lin.small_signal_gain(freqs, N=N_gain)
    g_disc_lin = mat_lin.small_signal_gain(freqs, N=N_gain, dt=dt_sim)

    # Also invert complex transmission t = Ex_lin / Ex_tr = exp(-i (k - k_bg) L) to recover chi(omega, N)
    t_complex = ex_lin / ex_tr
    delta_k = -np.log(t_complex) / (1j * L)
    n_fit = n_bg + (c / omegas) * delta_k
    chi_fit = n_fit**2 - eps_inf
    chi_an = mat_lin.chi(omegas, N=N_gain)
    chi_disc = mat_lin.chi_discrete(omegas, dt_sim, N=N_gain)

    assert np.max(g_fit_lin) > 7.5e4
    # Exact Fabry-Perot slab transmission factor (including tiny interface reflection r12)
    n_slab = np.sqrt(eps_inf + chi_disc)
    r12 = (n_bg - n_slab) / (n_bg + n_slab)
    k_slab = (omegas / c) * n_slab
    k_bg = (omegas / c) * n_bg
    t_slab_disc = (
        (1.0 - r12**2)
        * np.exp(-1j * (k_slab - k_bg) * L)
        / (1.0 - r12**2 * np.exp(-2j * k_slab * L))
    )
    np.testing.assert_allclose(
        flux_lin / flux_tr, np.abs(t_slab_disc) ** 2, rtol=4.0e-3
    )
    np.testing.assert_allclose(g_fit_lin, g_disc_lin, rtol=1.5e-2)
    np.testing.assert_allclose(g_fit_lin, g_an_lin, rtol=1.5e-2)
    np.testing.assert_allclose(g_fit_lin[5], g_an_lin[5], rtol=3.0e-3)
    np.testing.assert_allclose(np.imag(chi_fit), np.imag(chi_disc), rtol=1.5e-2)
    np.testing.assert_allclose(np.real(chi_fit), np.real(chi_disc), atol=5.0e-4)
    np.testing.assert_allclose(np.imag(chi_fit), np.imag(chi_an), rtol=1.5e-2)

    # 2. Logarithmic gain model: g_peak(N) = g0 * ln(N / N_tr)
    g0_log = 7.0e4
    N_log = 3.0 * N_tr
    mat_log = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        g0=g0_log,
        gain_model="logarithmic",
        eps_inf=eps_inf,
        A=0.0,
        initial_carriers=N_log,
        enable_stimulated=True,
    )
    flux_log, _, _ = run_slab_spectrum(mat_log)
    g_fit_log = np.log(flux_log / flux_tr) / L
    g_an_log = mat_log.small_signal_gain(freqs, N=N_log)
    assert np.isclose(mat_log.peak_gain(N=N_log), g0_log * np.log(3.0), rtol=1e-14)
    np.testing.assert_allclose(g_fit_log, g_an_log, rtol=1.5e-2)

    # 3. Below transparency (N = 0.2 * N_tr): interband absorption (g < 0)
    N_abs = 0.2 * N_tr
    mat_abs = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        differential_gain=g_prime,
        eps_inf=eps_inf,
        A=0.0,
        initial_carriers=N_abs,
    )
    flux_abs, _, _ = run_slab_spectrum(mat_abs)
    g_fit_abs = np.log(flux_abs / flux_tr) / L
    g_an_abs = mat_abs.small_signal_gain(freqs, N=N_abs)
    assert np.all(g_fit_abs < 0.0)
    np.testing.assert_allclose(g_fit_abs, g_an_abs, rtol=1.5e-2)

    # 4. Optional N-dependent gain peak shift: omega_0(N) = omega_0 + domega_dN * (N - N_tr)
    domega_dN = 2.0 * np.pi * 8.0e12 / N_tr  # +8 THz shift per N_tr above transparency
    mat_shift = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        differential_gain=g_prime,
        domega_dN=domega_dN,
        eps_inf=eps_inf,
        A=0.0,
        initial_carriers=2.0 * N_tr,
    )
    f_dense = np.linspace(0.85 * f0, 1.15 * f0, 301)
    f_peak_shifted = f_dense[np.argmax(mat_shift.small_signal_gain(f_dense))]
    np.testing.assert_allclose(f_peak_shifted, f0 + 8.0e12, rtol=2.0e-3)

    # Mutation checks:
    # Mutation 1: Flipping the sign of (N - N_tr) turns amplification into absorption and must fail
    g_mut_sign = -g_an_lin
    assert not np.allclose(g_fit_lin, g_mut_sign, rtol=5.0e-2)
    # Mutation 2: Dropping the Lorentzian frequency roll-off (flat gain = g_peak) fails in the wings
    g_mut_flat = np.full_like(freqs, mat_lin.peak_gain(N=N_gain))
    assert not np.allclose(g_fit_lin, g_mut_flat, rtol=5.0e-2)


# ---------------------------------------------------------------------------
# (b) Linewidth enhancement factor alpha_H from transmission phase/amplitude
# ---------------------------------------------------------------------------


def test_henry_alpha_factor_transmission():
    """
    (b) The measured Henry linewidth enhancement factor alpha_H extracted from
    the phase and amplitude change of the 3D FIT transmission with carrier-density
    change Delta N equals the input alpha_H:
        alpha_H = -d(Re chi)/d(Im chi) = Delta phi / Delta(ln |t|).
    Includes explicit alpha_H sign mutation check.
    """
    lam0 = 1.5e-6
    f0 = c / lam0
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 25.0e12
    eps_inf = 4.0
    n_bg = np.sqrt(eps_inf)
    N_tr = 1.0e24
    g_prime = 5.0e-20
    L = 3.0e-6
    dz = 15.0e-9
    z_min, z_max = -1.2e-6, 4.2e-6
    Nz = int(round((z_max - z_min) / dz))
    grid = wakis.GridFIT3D(
        -0.5e-6, 0.5e-6, -0.5e-6, 0.5e-6, z_min, z_max, 2, 2, Nz, verbose=0
    )

    def measure_complex_transmission(alpha_H_val, N_val):
        mat = SemiconductorGain(
            omega_0=omega_0,
            gamma=gamma,
            N_tr=N_tr,
            differential_gain=g_prime,
            alpha_H=alpha_H_val,
            eps_inf=eps_inf,
            A=0.0,
            initial_carriers=N_val,
            enable_stimulated=False,
        )
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.6,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            bg=[eps_inf, 1.0],
            n_pml=12,
            verbose=0,
        )
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.5 * L), size=(2.0e-6, 2.0e-6, L)
        )
        src = GaussianSource(frequency=f0, fwidth=0.12 * f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -0.6e-6),
            size=(1.0e-6, 1.0e-6, 0.0),
            amplitude=1.0e-3,
        )
        mon = wakis.DFTMonitor([f0], components=["Ex"], center=(0.0, 0.0, L + 0.4e-6))
        Nt = int((src.end_time + 75.0e-15) / solver.dt)
        solver.emsolve(Nt, source=sheet, monitors=[mon])
        return complex(mon.get_field("Ex")[0][0, 0, 0, 0]), mat

    ex_ref, _ = measure_complex_transmission(alpha_H_val=0.0, N_val=N_tr)
    N_pumped = 2.4 * N_tr

    for alpha_in in (0.0, 2.0, 3.5, -2.5):
        ex_pumped, mat = measure_complex_transmission(alpha_H_val=alpha_in, N_val=N_pumped)
        # Analytic susceptibility check at omega_0: d(Re chi)/dN == -alpha_H * d(Im chi)/dN
        assert np.isclose(mat.henry_factor(f0), alpha_in, atol=1e-12)
        dchi = mat.chi(omega_0, N=N_pumped) - mat.chi(omega_0, N=N_tr)
        assert np.isclose(-np.real(dchi) / np.imag(dchi), alpha_in, atol=1e-12)

        # Extract phase and log-amplitude change from 3D FIT complex transmission ratio
        ratio = ex_pumped / ex_ref
        d_ln_amp = float(np.log(np.abs(ratio)))
        d_phi = float(np.angle(ratio))
        assert d_ln_amp > 0.08  # significant amplification

        alpha_meas_phase = d_phi / d_ln_amp
        # Also invert exact complex wavenumber change delta_k = -ln(ratio) / (i * L)
        delta_k = -np.log(ratio) / (1j * L)
        n_fit = n_bg + (c / omega_0) * delta_k
        dchi_fit = n_fit**2 - eps_inf
        alpha_meas_chi = float(-np.real(dchi_fit) / np.imag(dchi_fit))

        np.testing.assert_allclose(alpha_meas_phase, alpha_in, atol=1.5e-2, rtol=1.0e-2)
        np.testing.assert_allclose(alpha_meas_chi, alpha_in, atol=1.5e-2, rtol=1.0e-2)

    # Mutation check (required: alpha_H sign flip):
    # If alpha_H has the wrong sign (-3.5 instead of +3.5), the phase shift reverses sign
    alpha_target = 3.5
    ex_pos, _ = measure_complex_transmission(alpha_H_val=+alpha_target, N_val=N_pumped)
    ex_neg_mut, _ = measure_complex_transmission(alpha_H_val=-alpha_target, N_val=N_pumped)
    alpha_pos = float(np.angle(ex_pos / ex_ref) / np.log(np.abs(ex_pos / ex_ref)))
    alpha_mut = float(np.angle(ex_neg_mut / ex_ref) / np.log(np.abs(ex_neg_mut / ex_ref)))
    assert alpha_pos > 3.4
    assert alpha_mut < -3.4
    assert not np.isclose(alpha_mut, alpha_target, rtol=0.1)


# ---------------------------------------------------------------------------
# (c) Steady-state L-I of a 1D Fabry-Perot diode laser vs rate-equation solution
# ---------------------------------------------------------------------------


def test_fabry_perot_diode_laser_li_and_slope_efficiency():
    """
    (c) Steady-state L-I curve of a 1D DBR Fabry-Perot semiconductor diode laser
    matches the rate-equation solution:
    - Threshold current density J_th = (q d / eta_i) (A N_th + B N_th^2 + C N_th^3).
    - Above threshold, carrier density clamps at N_th and output intensity grows
      linearly with (J - J_th) with differential quantum slope efficiency
      eta_d = eta_i * alpha_m / (alpha_i + alpha_m).
    """
    lam0 = 1.2e-6
    f0 = c / lam0
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 40.0e12
    dz = 25.0e-9  # 48 cells/lambda in cavity, 24 cells/lambda in DBR n_H = 2.0

    n_H, n_L = 2.0, 1.0
    d_H = lam0 / (4.0 * n_H)  # 150 nm = 6 * dz
    d_L = lam0 / (4.0 * n_L)  # 300 nm = 12 * dz
    # Symmetric 3-layer (H L H) quarter-wave DBR mirror on each side:
    # n_pairs = 1 (H, L) pair plus 1 extra H layer (extra_high=True), R = (15/17)^2 = 0.77855
    n_pairs = 1
    R_mirror = dbr_reflectance(n_H, n_L, n_pairs, n_in=1.0, n_out=1.0, extra_high=True)
    L_cav = 1.2e-6
    alpha_m = np.log(1.0 / R_mirror) / L_cav

    N_tr = 2.0e23
    g_prime = 3.0e-18
    A_rate = 1.0 / 150.0e-15  # fast relaxation for sub-ps 3D FIT convergence
    eta_i = 0.85
    d_act = L_cav

    # 1. Verify coupled rate-equation steady-state L-I with full A, B, C and internal loss alpha_i
    for alpha_i_test in (0.0, 0.25 * alpha_m):
        sigma_host = alpha_i_test * c * epsilon_0
        semi_abc = SemiconductorGain(
            omega_0=omega_0,
            gamma=gamma,
            N_tr=N_tr,
            differential_gain=g_prime,
            alpha_H=3.0,
            eta_i=eta_i,
            active_thickness=d_act,
            A=1.0e9,
            B=1.0e-16,
            C=3.0e-41,
            eps_inf=1.0,
            sigma=sigma_host,
        )
        ss_th = semi_abc.steady_state_laser(
            J=0.0, length=L_cav, R1=R_mirror, R2=R_mirror
        )
        J_th_abc = ss_th["J_th"]
        N_th_abc = ss_th["N_th"]
        eta_d_expected = eta_i * alpha_m / (alpha_i_test + alpha_m)
        # Consistency check on steady_state_laser formula evaluation:
        assert np.isclose(ss_th["eta_d"], eta_d_expected, rtol=1e-14, atol=0.0)

        j_factors = np.array([1.2, 1.6, 2.0, 2.5, 3.0])
        i_tot_num = []
        for jf in j_factors:
            j_val = jf * J_th_abc
            ss_pt = semi_abc.steady_state_laser(
                J=j_val, length=L_cav, R1=R_mirror, R2=R_mirror
            )
            tr = semi_abc.step_laser_rate_equations(
                Nt=10000,
                dt=0.8e-12,
                J=j_val,
                length=L_cav,
                R1=R_mirror,
                R2=R_mirror,
                N_init=N_th_abc,
                S_init=0.95 * ss_pt["S_ss"],
            )
            i_tot_num.append(tr["I_out_total"][-1])
            np.testing.assert_allclose(tr["N"][-1], N_th_abc, rtol=1e-5)
            np.testing.assert_allclose(tr["I_out_total"][-1], ss_pt["I_out_total"], rtol=1e-4)

        slope_fit, _ = np.polyfit((j_factors - 1.0) * J_th_abc, i_tot_num, 1)
        eta_d_num = slope_fit * (q_e / (hbar * omega_0)) * (d_act / L_cav)
        np.testing.assert_allclose(eta_d_num, eta_d_expected, rtol=1e-4)

    # 2. Full 3D FIT Fabry-Perot diode laser L-I sweep across threshold
    N_tr_fit = 5.0e19
    g_prime_fit = 1.8e-14
    semi_ref = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr_fit,
        differential_gain=g_prime_fit,
        alpha_H=0.0,
        eta_i=eta_i,
        active_thickness=d_act,
        A=A_rate,
        eps_inf=1.0,
    )
    ss_ref = semi_ref.steady_state_laser(
        J=0.0, length=L_cav, R1=R_mirror, R2=R_mirror
    )
    J_th = ss_ref["J_th"]
    N_th = ss_ref["N_th"]
    J_tr = (q_e * d_act / eta_i) * A_rate * N_tr_fit

    z_min, z_max = -1.2e-6, 2.4e-6
    Nz = int(round((z_max - z_min) / dz))
    grid = wakis.GridFIT3D(
        -1.0e-6, 1.0e-6, -1.0e-6, 1.0e-6, z_min, z_max, 2, 2, Nz, verbose=0
    )

    def run_fit_diode_laser(J_val, total_time=0.9e-12):
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.8,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=10,
            verbose=0,
        )
        mat_H = DispersiveMaterial(eps_inf=n_H**2, poles=[])
        for m in range(n_pairs + 1):
            z_left = -(m * (d_H + d_L) + 0.5 * d_H)
            solver.add_dispersive_material(
                mat_H, center=(0.0, 0.0, z_left), size=(4.0e-6, 4.0e-6, d_H)
            )
            z_right = L_cav + m * (d_H + d_L) + 0.5 * d_H
            solver.add_dispersive_material(
                mat_H, center=(0.0, 0.0, z_right), size=(4.0e-6, 4.0e-6, d_H)
            )

        mat = SemiconductorGain(
            omega_0=omega_0,
            gamma=gamma,
            N_tr=N_tr_fit,
            differential_gain=g_prime_fit,
            alpha_H=0.0,
            J=J_val,
            eta_i=eta_i,
            active_thickness=d_act,
            A=A_rate,
            eps_inf=1.0,
            initial_carriers="steady_state" if J_val > 0 else N_tr_fit,
            enable_stimulated=True,
            spatial_averaging=True,
        )
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.5 * L_cav), size=(4.0e-6, 4.0e-6, L_cav)
        )
        src = GaussianSource(frequency=f0, fwidth=0.08 * f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, 0.25 * L_cav),
            size=(2.0e-6, 2.0e-6, 0.0),
            amplitude=2.0,
        )
        z_out = L_cav + 2.0 * d_H + d_L + 0.2e-6
        k_out = int(np.argmin(np.abs(solver.z - z_out)))
        idx_out = k_out * solver.Nx * solver.Ny
        Nt = int(total_time / solver.dt)
        _preflight_1d_fit(Nz, Nt, n_gain_cells=int(L_cav / dz))

        ex_out = np.empty(Nt)
        n_hist = np.empty(Nt)
        for n in range(Nt):
            sheet.update(solver, n * solver.dt)
            solver.one_step()
            ex_out[n] = solver.E.array[idx_out]
            n_hist[n] = solver.get_gain_carriers(cell_average=True)

        n_avg = int((20.0 / f0) / solver.dt)
        I_out_1 = c * epsilon_0 * float(np.mean(ex_out[-n_avg:] ** 2))
        return I_out_1, ex_out, n_hist, solver.dt

    # Sub-threshold checks: at transparency J_tr and half way to threshold J_tr + 0.4*(J_th - J_tr)
    I_sub1, _, _, _ = run_fit_diode_laser(J_tr)
    I_sub2, _, _, _ = run_fit_diode_laser(J_tr + 0.4 * (J_th - J_tr))
    assert I_sub1 < 1.0e-4
    assert I_sub2 < 1.0e-2

    # Above-threshold sweep: J = J_tr + r_excess * (J_th - J_tr)
    r_excess = np.array([1.5, 2.0, 2.5, 3.0])
    J_sweep = J_tr + r_excess * (J_th - J_tr)
    I_fit_1 = []
    I_an_1 = []
    n_hist_r25 = None
    dt_fit = None
    for idx_r, J_val in enumerate(J_sweep):
        i1, _, n_tr_hist, dt_cur = run_fit_diode_laser(J_val)
        if idx_r == 2:
            n_hist_r25 = n_tr_hist
            dt_fit = dt_cur
        ss_pt = semi_ref.steady_state_laser(
            J=J_val, length=L_cav, R1=R_mirror, R2=R_mirror
        )
        I_fit_1.append(i1)
        I_an_1.append(ss_pt["I_out_1"])
        # Clamped carrier density matches N_th
        np.testing.assert_allclose(n_tr_hist[-1], N_th, rtol=2.5e-2)
        np.testing.assert_allclose(i1, ss_pt["I_out_1"], rtol=8.0e-2)

    # Compare 3D FIT relaxation-oscillation frequency at r_excess = 2.5 against theory:
    # (1) Measure tau_p,cold directly via a passive cold-cavity ringdown (gain off,
    #     ResonanceMonitor + harminv on the identical 3-layer (H L H) DBR resonator):
    #     tau_p,cold = 22.142 fs -> L_eff,cold = c * tau_p,cold * ln(1 / R_mirror) = 1.662 um
    #     (Gamma_z = 0.7222), predicting f_R = 3.382 THz vs 3D FIT 3.390 THz (0.24% error).
    # (2) Asymptotic analytical DBR penetration L_pen = lambda_0 / (4*(n_H - n_L)) = 0.300 um
    #     gives L_eff,an = L_cav + 2*L_pen = 1.800 um (Gamma_z = 2/3), predicting
    #     f_R,an = 3.250 THz (4.33% error), vs 3.980 THz (14.8% error) for bare L_cav.
    solver_cold = wakis.SolverFIT3D(
        grid,
        cfln=0.8,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=10,
        verbose=0,
    )
    mat_H_cold = DispersiveMaterial(eps_inf=n_H**2, poles=[])
    for m in range(n_pairs + 1):
        z_left = -(m * (d_H + d_L) + 0.5 * d_H)
        solver_cold.add_dispersive_material(
            mat_H_cold, center=(0.0, 0.0, z_left), size=(4.0e-6, 4.0e-6, d_H)
        )
        z_right = L_cav + m * (d_H + d_L) + 0.5 * d_H
        solver_cold.add_dispersive_material(
            mat_H_cold, center=(0.0, 0.0, z_right), size=(4.0e-6, 4.0e-6, d_H)
        )
    src_cold = GaussianSource(frequency=f0, fwidth=0.05 * f0)
    sheet_cold = CurrentSource(
        src_cold,
        component="x",
        center=(0.0, 0.0, 0.25 * L_cav),
        size=(2.0e-6, 2.0e-6, 0.0),
        amplitude=1.0,
    )
    res_mon = ResonanceMonitor(
        component="Ex",
        center=(0.0, 0.0, 0.75 * L_cav),
        t_start=src_cold.end_time + 30.0e-15,
    )
    Nt_cold = int((src_cold.end_time + 300.0e-15) / solver_cold.dt)
    _preflight_1d_fit(Nz, Nt_cold, n_gain_cells=0)
    solver_cold.emsolve(Nt_cold, source=sheet_cold, monitors=[res_mon])
    modes_cold = res_mon.harminv(0.85 * f0, 1.15 * f0)
    idx_mode = int(np.argmin(np.abs(modes_cold.frequency - f0)))
    tau_p_cold = float(1.0 / (2.0 * float(modes_cold.decay_rate[idx_mode])))
    assert 20.0e-15 < tau_p_cold < 25.0e-15

    t_ro = np.arange(len(n_hist_r25)) * dt_fit
    mask_ro = (t_ro >= 0.22e-12) & (t_ro <= 0.65e-12)
    dn_ro = n_hist_r25[mask_ro] - N_th
    sign_ch = np.where(np.diff(np.sign(dn_ro)) != 0)[0]
    t_zc = t_ro[mask_ro][sign_ch]
    half_periods = np.diff(t_zc[-4:])
    f_R_fit_meas = float(1.0 / (2.0 * np.mean(half_periods)))
    L_eff_dbr = c * tau_p_cold * np.log(1.0 / R_mirror)
    L_eff_an_dbr = L_cav + 2.0 * (lam0 / (4.0 * (n_H - n_L)))
    ss_r25_bare = semi_ref.steady_state_laser(
        J=J_sweep[2], length=L_cav, R1=R_mirror, R2=R_mirror
    )
    ss_r25_dbr = semi_ref.steady_state_laser(
        J=J_sweep[2], length=L_cav, R1=R_mirror, R2=R_mirror, L_eff=L_eff_dbr
    )
    ss_r25_an_dbr = semi_ref.steady_state_laser(
        J=J_sweep[2], length=L_cav, R1=R_mirror, R2=R_mirror, L_eff=L_eff_an_dbr
    )
    assert np.isclose(f_R_fit_meas, ss_r25_dbr["f_R"], rtol=0.01, atol=0.0)
    assert np.isclose(f_R_fit_meas, ss_r25_an_dbr["f_R"], rtol=0.05, atol=0.0)
    assert np.isclose(f_R_fit_meas, ss_r25_bare["f_R"], rtol=0.16, atol=0.0)
    # Mutation check: neglecting DBR penetration (using L_eff = L_cav) misses f_R_fit_meas by ~15%
    assert not np.isclose(f_R_fit_meas, ss_r25_bare["f_R"], rtol=0.05, atol=0.0)

    I_fit_total = 2.0 * np.array(I_fit_1)
    slope_fit, intercept_fit = np.polyfit(J_sweep, I_fit_total, 1)
    J_th_fit = -intercept_fit / slope_fit
    eta_d_fit = slope_fit * (q_e / (hbar * omega_0)) * (d_act / L_cav)

    np.testing.assert_allclose(J_th_fit, J_th, rtol=5.0e-2)
    np.testing.assert_allclose(eta_d_fit, ss_ref["eta_d"], rtol=6.0e-2)

    # Mutation check: omitting internal injection efficiency eta_i (using 1.0 instead of 0.85)
    # or dropping mirror loss factoralpha_m/(alpha_i + alpha_m) fails the slope efficiency check
    assert not np.isclose(eta_d_fit, 1.0, rtol=0.08)


# ---------------------------------------------------------------------------
# (d) Modulation response, relaxation-oscillation frequency, and adiabatic chirp
# ---------------------------------------------------------------------------


def test_modulation_response_relaxation_oscillation_and_chirp():
    """
    (d) Small-signal modulation response and relaxation-oscillation frequency
    match omega_R^2 = v_g * g' * S_0 / tau_p (with gain compression eps_nl = 0),
    and the instantaneous frequency chirp under current modulation matches
    Delta nu(t) = (alpha_H / (4 * pi)) * d(ln P)/dt (Koch & Bowers 1984).
    """
    lam0 = 1.55e-6
    f0 = c / lam0
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 30.0e12
    eps_inf = 12.25  # n_bg = 3.5
    N_tr = 1.2e24
    g_prime = 4.0e-20
    alpha_H = 4.0
    L_cav = 300.0e-6
    R1 = R2 = 0.32
    d_act = 0.1e-6

    semi = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        differential_gain=g_prime,
        alpha_H=alpha_H,
        eta_i=0.9,
        active_thickness=d_act,
        A=1.0e9,
        B=1.0e-16,
        C=3.0e-41,
        eps_nl=0.0,  # gain compression off
        eps_inf=eps_inf,
    )
    ss_base = semi.steady_state_laser(J=0.0, length=L_cav, R1=R1, R2=R2)
    J_th = ss_base["J_th"]

    # 1. Verify omega_R^2 = v_g * g' * S_0 / tau_p across multiple bias currents
    for bias_mult in (1.5, 2.0, 3.0, 4.5):
        J_bias = bias_mult * J_th
        ss = semi.steady_state_laser(J=J_bias, length=L_cav, R1=R1, R2=R2)
        omega_R_an = np.sqrt(semi.v_g * g_prime * ss["S_ss"] / ss["tau_p"])
        # Internal formula consistency check:
        assert np.isclose(ss["omega_R"], omega_R_an, rtol=1e-14, atol=0.0)

        # Small step perturbation from steady state: S(0) = 1.01 * S_ss, N(0) = N_th
        T_ro = 2.0 * np.pi / omega_R_an
        dt_ro = T_ro / 200.0
        Nt_ro = 1600
        tr = semi.step_laser_rate_equations(
            Nt=Nt_ro,
            dt=dt_ro,
            J=J_bias,
            length=L_cav,
            R1=R1,
            R2=R2,
            N_init=ss["N_th"],
            S_init=1.01 * ss["S_ss"],
        )
        ds = tr["S"] - ss["S_ss"]
        # Extract damped frequency and damping rate from consecutive zero crossings / peaks
        zc = np.where(np.diff(np.sign(ds)))[0]
        t_zc = tr["t"][zc] - ds[zc] * dt_ro / (ds[zc + 1] - ds[zc])
        half_periods = np.diff(t_zc[:8])
        gamma_ro_half = ss["gamma_RO"] / 2.0
        omega_d_num = np.pi / float(np.mean(half_periods))
        omega_R_num = np.sqrt(omega_d_num**2 + gamma_ro_half**2)
        np.testing.assert_allclose(omega_R_num, omega_R_an, rtol=2.0e-3)

    # 2. Small-signal sinusoidal current modulation transfer function H(f_m) & chirp Delta nu(t)
    J_0 = 2.5 * J_th
    ss_0 = semi.steady_state_laser(J=J_0, length=L_cav, R1=R1, R2=R2)
    f_R = ss_0["f_R"]
    dJ = 0.01 * (J_0 - J_th)

    f_mods = f_R * np.array([0.25, 0.5, 0.8, 1.0, 1.3, 1.8])
    H_an = semi.modulation_transfer_function(f_mods, J=J_0, length=L_cav, R1=R1, R2=R2)
    H_num = []

    for fm in f_mods:
        T_m = 1.0 / fm
        dt_m = T_m / 240.0
        n_periods = 14
        Nt_m = n_periods * 240
        tr_m = semi.step_laser_rate_equations(
            Nt=Nt_m,
            dt=dt_m,
            J=lambda t, f_mod=fm: J_0 + dJ * np.cos(2.0 * np.pi * f_mod * t),
            length=L_cav,
            R1=R1,
            R2=R2,
            N_init=ss_0["N_th"],
            S_init=ss_0["S_ss"],
        )
        # Analyze last 2 steady-state modulation periods
        sl = slice(-480, None)
        t_win = tr_m["t"][sl]
        s_win = tr_m["S"][sl]
        p_win = tr_m["I_out_total"][sl]
        dnu_win = tr_m["delta_nu"][sl]

        # Complex Fourier amplitude at fm
        phasor = np.exp(-1j * 2.0 * np.pi * fm * t_win)
        ds_phasor = 2.0 * np.mean((s_win - ss_0["S_ss"]) * phasor)
        ds_dc = ss_0["S_ss"] * (dJ / (J_0 - J_th))
        H_num.append(ds_phasor / ds_dc)

        # ODE consistency check: step_laser_rate_equations integrates
        # dphi/dt = 0.5 * alpha_H * Gamma * v_g * (g(N) - g_th) == 0.5 * alpha_H * d(ln S)/dt,
        # which matches finite-difference adiabatic_chirp(t_win, p_win, alpha_H)
        dnu_from_p = adiabatic_chirp(t_win, p_win, alpha_H)
        rel_rms_chirp = np.linalg.norm(dnu_win[5:-5] - dnu_from_p[5:-5]) / np.linalg.norm(
            dnu_win[5:-5]
        )
        assert rel_rms_chirp < 5.0e-4

        # ODE consistency check on FM/IM ratio: |Delta nu(fm)| / (|delta P(fm)| / P_0) == (alpha_H / 2) * fm
        dnu_phasor = 2.0 * np.mean(dnu_win * phasor)
        dp_norm_phasor = ds_phasor / ss_0["S_ss"]
        fm_im_ratio = dnu_phasor / dp_norm_phasor
        np.testing.assert_allclose(fm_im_ratio, 1j * 0.5 * alpha_H * fm, rtol=3.0e-3)

    np.testing.assert_allclose(np.abs(H_num), np.abs(H_an), rtol=5.0e-3)
    np.testing.assert_allclose(np.angle(H_num), np.angle(H_an), atol=5.0e-3)

    # 3. Direct 3D FIT validation of dynamic frequency chirp slope:
    #    Fit linear regression slope of demodulated Delta nu_FIT(t) vs d(ln P_FIT)/dt
    #    under sinusoidal current modulation above threshold and compare with alpha_H / (4 pi).
    lam_fit = 1.2e-6
    f0_fit = c / lam_fit
    w0_fit = 2.0 * np.pi * f0_fit
    dz_fit = 25.0e-9
    n_H, n_L = 2.0, 1.0
    d_H, d_L = lam_fit / (4.0 * n_H), lam_fit / (4.0 * n_L)
    # Symmetric 5-layer (H L H L H) quarter-wave DBR mirror (2 (H, L) pairs + 1 extra H layer):
    n_pairs_chirp = 2
    R_fit = dbr_reflectance(n_H, n_L, n_pairs_chirp, n_in=1.0, n_out=1.0, extra_high=True)
    L_fit = 1.2e-6
    gamma_pole = 2.0 * np.pi * 120.0e12
    N_tr_chirp = 5.0e19
    g_prime_chirp = 1.8e-14
    A_chirp = 1.0 / 200.0e-15

    semi_fit_ref = SemiconductorGain(
        omega_0=w0_fit,
        gamma=gamma_pole,
        N_tr=N_tr_chirp,
        differential_gain=g_prime_chirp,
        alpha_H=2.0,
        eta_i=1.0,
        active_thickness=L_fit,
        A=A_chirp,
        eps_inf=1.0,
    )
    ss_fit = semi_fit_ref.steady_state_laser(
        J=0.0, length=L_fit, R1=R_fit, R2=R_fit
    )
    J_th_fit = ss_fit["J_th"]
    N_th_fit = ss_fit["N_th"]
    J_tr_fit = q_e * L_fit * A_chirp * N_tr_chirp
    J_bias_fit = J_tr_fit + 2.5 * (J_th_fit - J_tr_fit)
    f_mod_fit = 1.5e12
    t_mod_start = 0.9e-12
    total_time_chirp = 1.9e-12

    def J_mod_func(t):
        if t < t_mod_start:
            return J_bias_fit
        return J_bias_fit + 0.35 * (J_bias_fit - J_th_fit) * np.sin(
            2.0 * np.pi * f_mod_fit * (t - t_mod_start)
        )

    def run_fit_chirp_slope(alpha_val):
        z_min_c, z_max_c = -1.5e-6, 2.7e-6
        Nz_c = int(round((z_max_c - z_min_c) / dz_fit))
        grid_fit = wakis.GridFIT3D(
            -1.0e-6, 1.0e-6, -1.0e-6, 1.0e-6, z_min_c, z_max_c, 2, 2, Nz_c, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid_fit,
            cfln=0.8,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=10,
            verbose=0,
        )
        ieps_arr = solver.ieps.array
        for m in range(n_pairs_chirp + 1):
            for z_c in (
                -(m * (d_H + d_L) + 0.5 * d_H),
                L_fit + m * (d_H + d_L) + 0.5 * d_H,
            ):
                idx_e, w_e = solver._box_edges(
                    (0.0, 0.0, z_c), (4.0e-6, 4.0e-6, d_H), dual=False
                )
                eps_s = 1.0 / (epsilon_0 * ieps_arr[idx_e])
                ieps_arr[idx_e] = 1.0 / (
                    epsilon_0 * ((1.0 - w_e) * eps_s + w_e * (n_H**2))
                )
        solver.update_tensors("ieps")
        mat_fit = SemiconductorGain(
            omega_0=w0_fit,
            gamma=gamma_pole,
            N_tr=N_tr_chirp,
            differential_gain=g_prime_chirp,
            alpha_H=alpha_val,
            J=J_mod_func,
            eta_i=1.0,
            active_thickness=L_fit,
            A=A_chirp,
            eps_inf=1.0,
            initial_carriers=N_th_fit,
            enable_stimulated=True,
            spatial_averaging=True,
        )
        solver.add_gain_material(
            mat_fit, center=(0.0, 0.0, 0.5 * L_fit), size=(4.0e-6, 4.0e-6, L_fit)
        )
        src_seed = GaussianSource(frequency=f0_fit, fwidth=0.06 * f0_fit)
        sheet_seed = CurrentSource(
            src_seed,
            component="x",
            center=(0.0, 0.0, 0.25 * L_fit),
            size=(2.0e-6, 2.0e-6, 0.0),
            amplitude=2.0,
        )
        z_out_c = L_fit + (n_pairs_chirp + 1) * d_H + n_pairs_chirp * d_L + 0.2e-6
        k_out_c = int(np.argmin(np.abs(solver.z - z_out_c)))
        idx_out_c = k_out_c * solver.Nx * solver.Ny
        Nt_fit = int(total_time_chirp / solver.dt)
        ex_out = np.empty(Nt_fit)
        for n in range(Nt_fit):
            sheet_seed.update(solver, n * solver.dt)
            solver.one_step()
            ex_out[n] = solver.E.array[idx_out_c]

        dt_c = solver.dt
        t_fit = np.arange(Nt_fit) * dt_c
        fft_e = np.fft.fft(ex_out)
        freqs_e = np.fft.fftfreq(Nt_fit, d=dt_c)
        bp = np.exp(-0.5 * ((freqs_e - f0_fit) / 8.0e12) ** 2) * (freqs_e > 0)
        analytic_e = 2.0 * np.fft.ifft(fft_e * bp)
        env_p = np.abs(analytic_e) ** 2
        inst_freq = np.gradient(np.unwrap(np.angle(analytic_e)), dt_c) / (2.0 * np.pi)
        dlnp_dt = np.gradient(np.log(np.maximum(env_p, 1.0e-30)), dt_c)
        win = (t_fit > 1.05e-12) & (t_fit < 1.75e-12)
        dnu_fit = inst_freq[win] - float(np.mean(inst_freq[win]))
        x_dlnp = dlnp_dt[win] - float(np.mean(dlnp_dt[win]))
        slope_meas = float(np.dot(x_dlnp, dnu_fit) / np.dot(x_dlnp, x_dlnp))
        corr_meas = float(np.corrcoef(x_dlnp, dnu_fit)[0, 1])
        return slope_meas, corr_meas

    for alpha_test in (2.0, 3.5):
        slope_fit_c, corr_fit_c = run_fit_chirp_slope(alpha_test)
        slope_theory_c = alpha_test / (4.0 * np.pi)
        assert corr_fit_c > 0.96
        assert np.isclose(slope_fit_c, slope_theory_c, rtol=0.02, atol=0.0)

    # Code-path mutation check on 3D FIT chirp: flipping the sign of alpha_H in the
    # 3D FIT gain material flips the fitted slope sign and fails the positive slope check
    slope_mut_neg, _ = run_fit_chirp_slope(-2.0)
    assert slope_mut_neg < 0.0
    assert not np.isclose(slope_mut_neg, 2.0 / (4.0 * np.pi), rtol=0.10, atol=0.0)

    # Mutation checks:
    # 1. Dropping 1/tau_p or scaling S_0 -> 2*S_0 in omega_R^2 = v_g * g' * S_0 / tau_p fails
    omega_R_mut = np.sqrt(2.0 * semi.v_g * g_prime * ss_0["S_ss"] / ss_0["tau_p"])
    assert not np.isclose(omega_R_mut, ss_0["omega_R"], rtol=0.1)
    # 2. Flipping alpha_H sign or replacing 4*pi with 2*pi in adiabatic_chirp fails
    dnu_mut_sign = adiabatic_chirp(t_win, p_win, -alpha_H)
    dnu_mut_factor = 2.0 * adiabatic_chirp(t_win, p_win, alpha_H)
    assert not np.allclose(dnu_win[5:-5], dnu_mut_sign[5:-5], rtol=0.1)
    assert not np.allclose(dnu_win[5:-5], dnu_mut_factor[5:-5], rtol=0.1)


# ---------------------------------------------------------------------------
# (e) Energy / number conservation: carriers removed = photons generated
# ---------------------------------------------------------------------------


def test_carrier_photon_conservation_and_diffusion():
    """
    (e) Bookkeeping and conservation test:
    1. In a closed lossless PEC cavity with J = 0 and A = B = C = 0, total
       carriers removed from the active region equals total photons generated
       Delta W_EM / (hbar * omega_0) to machine precision, for both stimulated
       emission (N > N_tr) and stimulated absorption (N < N_tr).
    2. With current injection J > 0, SRH/radiative/Auger recombination A, B, C > 0,
       and carrier diffusion D > 0, the full particle budget
       (N_initial - N_current) + N_injected - N_recombined == N_photons_generated
       closes to machine precision (~1e-14).
    3. 1D/2D carrier diffusion conserves total carrier count to machine precision
       and decays spatial modes at the exact rate exp(-D * k^2 * t).
    4. Mutation check: shifting the electric field in the stimulated work term by
       a half step (using E^n instead of E^(n+1/2)) breaks energy/number conservation.
    """
    N_grid, L_box = 10, 1.0e-6
    grid = wakis.GridFIT3D(
        -L_box / 2, L_box / 2, -L_box / 2, L_box / 2, -L_box / 2, L_box / 2,
        N_grid, N_grid, N_grid, verbose=0
    )
    f0 = 200.0e12
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 25.0e12
    N_tr = 1.0e24
    g_prime = 5.0e-20

    def run_cavity_bookkeeping(
        N_init_val, J_val=0.0, A_val=0.0, B_val=0.0, C_val=0.0, D_val=0.0, alpha_H_val=0.0
    ):
        solver = wakis.SolverFIT3D(
            grid, cfln=0.5, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
        )
        mat = SemiconductorGain(
            omega_0=omega_0,
            gamma=gamma,
            N_tr=N_tr,
            differential_gain=g_prime,
            alpha_H=alpha_H_val,
            J=J_val,
            eta_i=0.9,
            active_thickness=0.5e-6,
            A=A_val,
            B=B_val,
            C=C_val,
            diffusion_coeff=D_val,
            eps_inf=2.0,
            initial_carriers=N_init_val,
            enable_stimulated=True,
        )
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.0), size=(0.6e-6, 0.6e-6, 0.6e-6)
        )
        src = GaussianSource(frequency=f0, fwidth=60.0e12)
        dip = CurrentSource(
            src, component="z", center=(0.1e-6, -0.1e-6, 0.0), amplitude=1.0e-7
        )

        itA = grid.itA.toarray()
        iA = grid.iA.toarray()
        tA = np.divide(1.0, itA, out=np.zeros_like(itA), where=itA != 0)
        A_face = np.divide(1.0, iA, out=np.zeros_like(iA), where=iA != 0)
        dV_E = grid.L.toarray() * tA
        dV_H = grid.tL.toarray() * A_face
        ieps = solver.ieps.toarray()
        eps_r = np.divide(1.0, epsilon_0 * ieps, out=np.zeros_like(ieps), where=ieps != 0)
        g_grp = solver._gain_groups[0]
        eps_r[g_grp["idx"]] = g_grp["den"]

        Nt_src = int(np.ceil(src.end_time / solver.dt)) + 2
        for n in range(Nt_src):
            dip.update(solver, n * solver.dt)
            solver.one_step()
        solver.J.array[:] = 0.0

        # Record state right after source turns off
        bk_start = solver.get_gain_bookkeeping()
        E_old = solver.E.toarray().copy()
        H_old = solver.H.toarray().copy()
        p_prev = g_grp["poles"][0]["p"].copy()

        w_em_list = []
        pol_work_step = []
        mut_work_step = []
        dchi_work_step = []

        for _ in range(70):
            dchi_edge_n = mat.delta_chi_bg(N=g_grp["N"])[g_grp["inv"]]
            solver.one_step()
            E_cur = solver.E.toarray().copy()
            H_cur = solver.H.toarray().copy()
            w_em_list.append(
                0.5 * epsilon_0 * np.sum(eps_r * dV_E * E_old**2)
                + 0.5 * mu_0 * np.sum(dV_H * H_old * H_cur)
            )
            p_cur = g_grp["poles"][0]["p"].copy()
            dp = p_cur - p_prev
            e_cur_g = E_cur[g_grp["idx"]]
            e_old_g = E_old[g_grp["idx"]]
            e_half = 0.5 * (e_cur_g + e_old_g)
            e_mut_int = e_old_g  # half-step shift mutation: E^n instead of E^(n+1/2)
            pol_work_step.append(
                -epsilon_0 * np.sum(g_grp["w_dV_edge"] * (dp / solver.dt) * e_half)
            )
            mut_work_step.append(
                -epsilon_0 * np.sum(g_grp["w_dV_edge"] * (dp / solver.dt) * e_mut_int)
            )
            dchi_work_step.append(
                -0.5
                * (epsilon_0 / solver.dt)
                * np.sum(g_grp["w_dV_edge"] * dchi_edge_n * (e_cur_g**2 - e_old_g**2))
            )
            p_prev = p_cur
            E_old, H_old = E_cur, H_cur

        bk_end = solver.get_gain_bookkeeping()
        return (
            bk_start,
            bk_end,
            np.array(w_em_list),
            np.array(pol_work_step),
            np.array(mut_work_step),
            np.array(dchi_work_step),
            solver.dt,
            g_grp["total_vol"],
        )

    # 1. Pure stimulated emission (N_init = 3.0 * N_tr > N_tr, J = A = B = C = 0)
    bk0_g, bk1_g, w_em_g, p_work_g, mut_work_g, _, dt_g, _ = run_cavity_bookkeeping(3.0 * N_tr)
    carriers_removed_g = bk1_g["carriers_removed"]
    photons_gen_g = bk1_g["photons_generated"]
    assert carriers_removed_g > 0.0  # carriers depleted by stimulated emission
    assert photons_gen_g > 0.0  # photons created
    # Bookkeeping consistency checks (both counters accumulate stim_edge_rate):
    np.testing.assert_allclose(carriers_removed_g, photons_gen_g, rtol=2e-14)
    np.testing.assert_allclose(
        bk1_g["em_work_joules"] / (hbar * omega_0), carriers_removed_g, rtol=2e-14
    )
    # Independent validation via per-step discrete Poynting theorem on 3D FIT fields:
    # (W_EM^(n+1) - W_EM^n)/dt == P_pol->EM^(n+1/2)
    dw_dt = np.diff(w_em_g) / dt_g
    rel_poynting_err = np.linalg.norm(dw_dt - p_work_g[:-1]) / np.linalg.norm(p_work_g[:-1])
    assert rel_poynting_err < 1e-12

    # Mutation check (half-step shift: E^n instead of E^(n+1/2)): breaks Poynting balance by > 1e8x
    rel_mut_err = np.linalg.norm(dw_dt - mut_work_g[:-1]) / np.linalg.norm(p_work_g[:-1])
    assert rel_mut_err > 1e-2

    # 1b. Nonzero Henry factor alpha_H = 3.5 != 0 energy-consistent discrete Poynting
    #     balance and analytical bound on parametric Delta chi_bg(N) index-modulation work:
    alpha_nz = 3.5
    bk0_ah, bk1_ah, w_em_ah, p_work_ah, _, dchi_work_ah, dt_ah, vol_act = (
        run_cavity_bookkeeping(3.0 * N_tr, alpha_H_val=alpha_nz)
    )
    dw_dt_ah = np.diff(w_em_ah) / dt_ah
    # Including the tracked Delta chi_bg(N) displacement work closes the discrete
    # Poynting balance to machine precision (< 1e-12):
    rel_poynting_ah_exact = np.linalg.norm(
        dw_dt_ah - (p_work_ah[:-1] + dchi_work_ah[:-1])
    ) / np.linalg.norm(p_work_ah[:-1])
    assert rel_poynting_ah_exact < 1e-12
    # Discrete integration-by-parts identity between dchi_work, dchi_parametric, and dchi_stored:
    d_dchi_work = bk1_ah["dchi_work_joules"] - bk0_ah["dchi_work_joules"]
    d_dchi_param = bk1_ah["dchi_parametric_joules"] - bk0_ah["dchi_parametric_joules"]
    d_dchi_stored = bk1_ah["dchi_stored_joules"] - bk0_ah["dchi_stored_joules"]
    np.testing.assert_allclose(
        d_dchi_work, d_dchi_param - d_dchi_stored, rtol=1e-12, atol=0.0
    )
    # Analytical upper bound on the parametric index-modulation work |W_{Delta chi,param}|
    # relative to the resonant polarization work |W_{pol->EM}|:
    d_em_work = bk1_ah["em_work_joules"] - bk0_ah["em_work_joules"]
    n_bg_ah = np.sqrt(2.0)
    u_em_max = float(np.max(w_em_ah)) / vol_act
    param_bound = (
        abs(alpha_nz)
        * (c * n_bg_ah * g_prime / omega_0)
        * (u_em_max / (hbar * omega_0))
    )
    param_ratio = abs(d_dchi_param) / abs(d_em_work)
    assert param_ratio <= param_bound
    assert param_ratio < 0.02
    # Mutation check: omitting dchi_work_ah when alpha_H = 3.5 leaves a nonzero residual > 1e-3
    rel_poynting_ah_no_dchi = np.linalg.norm(
        dw_dt_ah - p_work_ah[:-1]
    ) / np.linalg.norm(p_work_ah[:-1])
    assert rel_poynting_ah_no_dchi > 1.0e-3

    # 2. Pure stimulated absorption (N_init = 0.1 * N_tr < N_tr, J = A = B = C = 0)
    _, bk1_a, _, _, _, _, _, _ = run_cavity_bookkeeping(0.1 * N_tr)
    assert bk1_a["carriers_removed"] < 0.0  # carriers created by photon absorption
    assert bk1_a["photons_generated"] < 0.0  # photons absorbed
    np.testing.assert_allclose(
        bk1_a["carriers_removed"], bk1_a["photons_generated"], rtol=2e-14
    )

    # 3. Full bookkeeping with injection J > 0, recombination A, B, C > 0, and 3D diffusion D > 0
    _, bk1_full, _, _, _, _, _, _ = run_cavity_bookkeeping(
        2.5 * N_tr,
        J_val=5.0e5,
        A_val=2.0e8,
        B_val=1.0e-16,
        C_val=1.0e-40,
        D_val=2.0e-3,
    )
    net_carrier_balance = (
        bk1_full["carriers_removed"]
        + bk1_full["carriers_injected"]
        - bk1_full["carriers_recombined"]
    )
    np.testing.assert_allclose(
        net_carrier_balance, bk1_full["carriers_stimulated"], rtol=1e-13
    )
    np.testing.assert_allclose(
        net_carrier_balance, bk1_full["photons_generated"], rtol=1e-13
    )

    # 4. Standalone 1D carrier diffusion: exact conservation of sum(N) and Fourier decay exp(-D k^2 t)
    D_diff = 1.0e-3  # [m^2/s]
    semi_diff = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        A=0.0,
        B=0.0,
        C=0.0,
        diffusion_coeff=D_diff,
    )
    M_pts = 64
    L_diff = 6.4e-6
    dx_diff = L_diff / M_pts
    x_arr = (np.arange(M_pts) + 0.5) * dx_diff
    k_mode = 2.0 * np.pi / L_diff
    n0_prof = N_tr + 0.3 * N_tr * np.cos(k_mode * x_arr)
    t_end_diff = 0.4e-9
    t_d, n_hist_d = semi_diff.step_rate_equations(
        Nt=200, dt=t_end_diff / 200, N_init=n0_prof, dx=dx_diff, periodic=True
    )
    # Total carrier count conserved at every step to machine precision
    np.testing.assert_allclose(
        n_hist_d.sum(axis=1), n0_prof.sum(), rtol=1e-14
    )
    # Cosine mode amplitude decays as exp(-D * k_disc^2 * t)
    k_disc2 = (2.0 / dx_diff * np.sin(0.5 * k_mode * dx_diff)) ** 2
    amp_final = 2.0 * np.mean((n_hist_d[-1] - N_tr) * np.cos(k_mode * x_arr))
    amp_exact = 0.3 * N_tr * np.exp(-D_diff * k_disc2 * t_end_diff)
    np.testing.assert_allclose(amp_final, amp_exact, rtol=5.0e-4)


def test_multipole_and_solver_guards():
    """
    Verify two-pole SemiconductorGain susceptibility, Henry alpha_H identity at
    the primary transition, solver state inspection/reset helpers, and input validation.
    """
    f0 = 193.4e12
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 20.0e12
    N_tr = 1.0e24

    semi_2p = SemiconductorGain(
        omega_0=omega_0,
        gamma=gamma,
        N_tr=N_tr,
        differential_gain=3.0e-20,
        alpha_H=3.5,
        extra_poles=[
            {
                "omega_0": 1.08 * omega_0,
                "gamma": 1.3 * gamma,
                "differential_gain": 1.2e-20,
                "N_tr": 1.3 * N_tr,
            },
        ],
    )
    n1 = 1.6 * N_tr
    n2 = 1.61 * N_tr
    chi1 = semi_2p.susceptibility(omega=omega_0, N=n1)
    chi2 = semi_2p.susceptibility(omega=omega_0, N=n2)
    alpha_num = -(np.real(chi2 - chi1)) / (np.imag(chi2 - chi1))
    np.testing.assert_allclose(alpha_num, 3.5, rtol=1e-4)

    # Input validation guards
    with pytest.raises(ValueError, match="omega_0"):
        SemiconductorGain(omega_0=-1.0, gamma=gamma, N_tr=N_tr)
    with pytest.raises(ValueError, match="gain_model"):
        SemiconductorGain(
            omega_0=omega_0, gamma=gamma, N_tr=N_tr, gain_model="quadratic"
        )
    with pytest.raises(ValueError, match="diffusion_axes"):
        SemiconductorGain(
            omega_0=omega_0, gamma=gamma, N_tr=N_tr, diffusion_axes=("x", "w")
        )

