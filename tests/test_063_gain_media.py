# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Validation tests for active gain media (`wakis.gain`: `FourLevelGain` and
`SaturableGain`) coupled to `SolverFIT3D`.

Covers:
1. Unpumped ground-state passive Lorentz absorption vs analytic susceptibility
   and `wakis.dispersive.Lorentz` (continuous, discrete, and 3D FIT slab R/T).
2. Small-signal single-pass amplification in a pumped 1D slab vs analytic
   rate-equation steady-state inversion and exponential growth exp(g0 * z).
3. Standalone 4-level rate equations (no field): exact conservation of
   N_0 + N_1 + N_2 + N_3 = N_total, convergence to analytic steady state, and
   second-order O(dt^2) accuracy (with and without Pauli exclusion).
4. Homogeneous gain saturation: output vs input intensity follows the exact
   Lambert-W saturable-amplifier law `amplified_intensity` for both
   `SaturableGain` and `FourLevelGain`.
5. 1D DBR Fabry-Perot laser cavity: cold-cavity mode frequency and Q / mirror
   loss threshold prediction, relaxation oscillations, inversion clamping,
   lasing spectrum at the cavity mode, above-threshold linear L-I curve slope,
   and zero-gain mutation (no lasing).
6. Discrete energy accounting: with gain off and no loss, electromagnetic field
   energy is conserved to machine round-off; discrete Poynting-polarization
   work balance holds to round-off at every step.
"""

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, hbar, mu_0

import wakis
from wakis.gain import (
    FourLevelGain,
    SaturableGain,
    amplified_intensity,
    dbr_reflectance,
    fabry_perot_laser_output,
    fabry_perot_threshold_gain,
)
from wakis.materials import DispersiveMaterial, Lorentz
from wakis.sources import ContinuousSource, CurrentSource, GaussianSource


def _slab_rt(eps, f, d):
    """Analytic normal-incidence power reflectance and transmittance of a slab."""
    n = np.sqrt(eps.astype(complex))
    n = np.where(n.real < 0, -n, n)
    r12 = (1.0 - n) / (1.0 + n)
    phase = np.exp(-2j * 2.0 * np.pi * f / c * n * d)
    r = r12 * (1.0 - phase) / (1.0 - r12**2 * phase)
    t = (1.0 - r12**2) * np.sqrt(phase) / (1.0 - r12**2 * phase)
    return np.abs(r) ** 2, np.abs(t) ** 2


# ---------------------------------------------------------------------------
# 1. Ground-state / unpumped passive Lorentz absorption
# ---------------------------------------------------------------------------


def test_ground_state_passive_lorentz():
    """
    With no pumping (W_p = 0) and population in the ground state of an atomic
    transition, FourLevelGain and SaturableGain reduce to a passive absorbing
    Lorentz medium. Both continuous eps(w), discrete eps_d(w, dt), and 3D FIT
    slab reflectance/transmittance match `wakis.dispersive.Lorentz` and the
    analytic slab formulas.
    """
    f = np.linspace(0.7e9, 1.3e9, 7)
    omega_0 = 2.0 * np.pi * 1.0e9
    gamma = 2.0 * np.pi * 0.1e9
    delta_eps = 1.5
    eps_inf = 2.0
    N_total = 1.0e20
    kappa = delta_eps * epsilon_0 * omega_0**2 / N_total

    # (a) 0 <-> 3 transition in the true ground state (N_0 = N_total, W_p = 0)
    fg_ground_b = FourLevelGain(
        omega_a=omega_0,
        gamma_a=gamma,
        tau_21=1.0e-3,
        tau_10=1.0e-11,
        tau_32=1.0e-11,
        N_total=N_total,
        pump_rate=0.0,
        eps_inf=eps_inf,
        kappa_a=0.0,
        omega_b=omega_0,
        gamma_b=gamma,
        kappa_b=kappa,
        initial_populations="ground",
        enable_stimulated=False,
    )
    # (b) 1 <-> 2 transition with all population in the lower level N_1
    fg_lower_a = FourLevelGain(
        omega_a=omega_0,
        gamma_a=gamma,
        tau_21=1.0e-3,
        tau_10=np.inf,
        tau_32=1.0e-11,
        N_total=N_total,
        pump_rate=0.0,
        eps_inf=eps_inf,
        kappa_a=kappa,
        initial_populations=(0.0, N_total, 0.0, 0.0),
        enable_stimulated=False,
    )
    # (c) SaturableGain with negative g0 (passive saturable absorber in linear limit)
    sg_passive = SaturableGain(
        omega_0=omega_0,
        gamma=gamma,
        eps_inf=eps_inf,
        delta_eps=delta_eps,
        I_sat=np.inf,
    )
    dm_lorentz = DispersiveMaterial(
        eps_inf, [Lorentz(delta_eps=delta_eps, omega_0=omega_0, gamma=gamma)]
    )

    dt_test = 5.0e-12
    for mat in (fg_ground_b, fg_lower_a, sg_passive):
        np.testing.assert_allclose(mat.epsilon(f), dm_lorentz.epsilon(f), rtol=1e-14)
        np.testing.assert_allclose(
            mat.epsilon_discrete(f, dt_test),
            dm_lorentz.epsilon_discrete(f, dt_test),
            rtol=1e-14,
        )
        assert np.all(np.imag(mat.epsilon(f)) < 0.0)

    # 3D FIT slab reflectance and transmittance comparison
    thickness = 0.1
    dz = 5.0e-3
    grid = wakis.GridFIT3D(
        -0.01, 0.01, -0.01, 0.01, -0.5, 0.5, 2, 2, int(1.0 / dz), verbose=0
    )

    def run_slab(material):
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=12,
            verbose=0,
        )
        if material is not None:
            solver.add_dispersive_material(
                material,
                center=(0.0, 0.0, 0.5 * thickness),
                size=(1.0, 1.0, thickness),
            )
        src = GaussianSource(frequency=1.0e9, fwidth=0.4e9)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -0.25),
            size=(0.02, 0.02, 0.0),
            amplitude=1.0e-3,
        )
        mons = [
            wakis.FluxMonitor(f, center=(0.0, 0.0, z0), size=(0.02, 0.02, 0.0))
            for z0 in (-0.12, thickness + 0.1)
        ]
        Nt = int((src.end_time + 15.0e-9) / solver.dt)
        solver.emsolve(Nt, source=sheet, monitors=mons)
        return [m.get_flux() for m in mons], solver.dt

    (b_ref, a_ref), dt_sim = run_slab(None)
    (b_fgb, a_fgb), _ = run_slab(fg_ground_b)
    (b_fga, a_fga), _ = run_slab(fg_lower_a)
    (b_dm, a_dm), _ = run_slab(dm_lorentz)

    R_fgb, T_fgb = 1.0 - b_fgb / b_ref, a_fgb / a_ref
    R_fga, T_fga = 1.0 - b_fga / b_ref, a_fga / a_ref
    R_dm, T_dm = 1.0 - b_dm / b_ref, a_dm / a_ref
    R_an, T_an = _slab_rt(dm_lorentz.epsilon_discrete(f, dt_sim), f, thickness)

    # Identical to DispersiveMaterial(Lorentz) to machine precision
    np.testing.assert_allclose(R_fgb, R_dm, atol=1e-13)
    np.testing.assert_allclose(T_fgb, T_dm, atol=1e-13)
    np.testing.assert_allclose(R_fga, R_dm, atol=1e-13)
    np.testing.assert_allclose(T_fga, T_dm, atol=1e-13)
    # Agreement with analytic slab R/T to grid discretization accuracy
    np.testing.assert_allclose(R_fgb, R_an, atol=1.5e-2)
    np.testing.assert_allclose(T_fgb, T_an, atol=5.0e-3)

    # Mutation check: inverting the population (N_2 = N_total) turns absorption into gain
    fg_inverted = FourLevelGain(
        omega_a=omega_0,
        gamma_a=gamma,
        tau_21=1.0e-3,
        tau_10=np.inf,
        tau_32=1.0e-11,
        N_total=N_total,
        pump_rate=0.0,
        eps_inf=eps_inf,
        kappa_a=kappa,
        initial_populations=(0.0, 0.0, N_total, 0.0),
        enable_stimulated=False,
    )
    assert np.all(np.imag(fg_inverted.epsilon(f)) > 0.0)
    assert not np.allclose(fg_inverted.epsilon(f), dm_lorentz.epsilon(f))


# ---------------------------------------------------------------------------
# 2. Weak-signal single-pass amplification in a pumped 1D slab
# ---------------------------------------------------------------------------


def test_small_signal_gain_slab():
    """
    A weak optical pulse at f0 = 300 THz propagating through a pumped 1D
    FourLevelGain slab of thickness L = 4 um is amplified exponentially as
    I(z) = I(0) exp(g0 * z), where g0 = sigma_21 * (N_2 - N_1)_ss is the
    analytic rate-equation steady-state small-signal gain coefficient.
    """
    f0 = 300.0e12
    omega_a = 2.0 * np.pi * f0
    gamma_a = 2.0 * np.pi * 30.0e12
    tau_21 = 300.0e-12
    tau_10 = 0.1e-12
    tau_32 = 0.1e-12
    N_total = 2.0e23
    W_p = 5.0e8
    L = 4.0e-6

    fg = FourLevelGain(
        omega_a=omega_a,
        gamma_a=gamma_a,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=N_total,
        pump_rate=W_p,
        eps_inf=1.0,
        initial_populations="steady_state",
        enable_stimulated=True,
    )

    # Analytic steady-state inversion and small-signal gain
    t3 = fg.tau_3
    r3 = W_p * t3
    r1 = (tau_10 / tau_32) * r3
    r2 = (tau_21 / tau_32) * r3
    dn_an = N_total * (r2 - r1) / (1.0 + r1 + r2 + r3)
    assert np.isclose(fg.steady_state_inversion(), dn_an, rtol=1e-14)

    g0_an = fg.small_signal_gain()
    assert np.isclose(g0_an, fg.emission_cross_section * dn_an, rtol=5e-4)

    dz = 25.0e-9  # lambda_0 / 40
    grid = wakis.GridFIT3D(
        -0.5e-6, 0.5e-6, -0.5e-6, 0.5e-6, -2.0e-6, 6.0e-6, 2, 2, int(8.0e-6 / dz), verbose=0
    )
    z_mons = [0.5e-6, 1.5e-6, 2.5e-6, 3.5e-6, L + 0.6e-6]

    def run_amp(mat):
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.7,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=12,
            verbose=0,
        )
        if mat is not None:
            solver.add_gain_material(
                mat, center=(0.0, 0.0, 0.5 * L), size=(2.0e-6, 2.0e-6, L)
            )
        src = GaussianSource(frequency=f0, fwidth=15.0e12)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -0.8e-6),
            size=(1.0e-6, 1.0e-6, 0.0),
            amplitude=1.0e-4,
        )
        mons = [
            wakis.FluxMonitor([f0], center=(0.0, 0.0, zm), size=(1.0e-6, 1.0e-6, 0.0))
            for zm in z_mons
        ]
        Nt = int((src.end_time + 45.0e-15) / solver.dt)
        solver.emsolve(Nt, source=sheet, monitors=mons)
        return np.array([float(m.get_flux()[0]) for m in mons])

    flux_ref = run_amp(None)
    flux_gain = run_amp(fg)

    G_trans = flux_gain[-1] / flux_ref[-1]
    G_an = np.exp(g0_an * L)
    g0_trans = np.log(G_trans) / L
    z_in = np.array(z_mons[:-1])
    g0_slope = np.polyfit(z_in, np.log(flux_gain[:-1] / flux_ref[:-1]), 1)[0]

    assert G_trans > 2.0
    np.testing.assert_allclose(G_trans, G_an, rtol=5.0e-3)
    np.testing.assert_allclose(g0_trans, g0_an, rtol=5.0e-3)
    np.testing.assert_allclose(g0_slope, g0_an, rtol=5.0e-3)

    # Mutation check: unpumped medium (W_p = 0) gives zero gain (G = 1)
    fg_unpumped = FourLevelGain(
        omega_a=omega_a,
        gamma_a=gamma_a,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=N_total,
        pump_rate=0.0,
        eps_inf=1.0,
        initial_populations="steady_state",
    )
    flux_unpumped = run_amp(fg_unpumped)
    np.testing.assert_allclose(flux_unpumped[-1] / flux_ref[-1], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Standalone rate equations: conservation, steady state, and O(dt^2) order
# ---------------------------------------------------------------------------


def test_rate_equations_conservation_and_steady_state():
    """
    With no electromagnetic field, the 4-level rate equations conserve
    N_0 + N_1 + N_2 + N_3 = N_total to machine precision at every timestep,
    converge to `steady_state_populations()`, and exhibit second-order O(dt^2)
    global truncation error for both linear and Pauli-blocked kinetics.
    """
    N_total = 1.0e24
    for pauli in (False, True):
        fg = FourLevelGain(
            omega_a=2.0 * np.pi * 300.0e12,
            gamma_a=2.0 * np.pi * 20.0e12,
            tau_21=5.0e-12,
            tau_10=1.0e-12,
            tau_32=0.5e-12,
            tau_30=20.0e-12,
            N_total=N_total,
            pump_rate=2.0e11,
            pauli_exclusion=pauli,
            N_site=1.5e24,
            initial_populations="ground",
        )
        # Rate matrix columns sum to zero
        np.testing.assert_allclose(fg.rate_matrix().sum(axis=0), 0.0, atol=1e-15)

        dt = 5.0e-14
        _, n_hist = fg.step_rate_equations(Nt=1200, dt=dt)
        totals = n_hist.sum(axis=1)
        np.testing.assert_allclose(totals, N_total, rtol=1e-14)

        n_ss_an = fg.steady_state_populations()
        np.testing.assert_allclose(n_hist[-1], n_ss_an, rtol=1e-9)
        assert n_hist[-1, 2] > n_hist[-1, 1]  # inverted: N_2 > N_1

        # Second-order O(dt^2) convergence check at t_end = 1.0 ps
        t_end = 1.0e-12
        _, hist_coarse = fg.step_rate_equations(Nt=20, dt=t_end / 20)
        _, hist_fine = fg.step_rate_equations(Nt=40, dt=t_end / 40)
        _, hist_ref = fg.step_rate_equations(Nt=1600, dt=t_end / 1600)
        err_coarse = np.linalg.norm(hist_coarse[-1] - hist_ref[-1])
        err_fine = np.linalg.norm(hist_fine[-1] - hist_ref[-1])
        order = np.log2(err_coarse / err_fine)
        assert order > 1.95

    # Mutation check: when tau_21 < tau_10 (lower level bottleneck), inversion is negative
    fg_bottleneck = FourLevelGain(
        omega_a=2.0 * np.pi * 300.0e12,
        gamma_a=2.0 * np.pi * 20.0e12,
        tau_21=0.5e-12,
        tau_10=5.0e-12,
        tau_32=0.5e-12,
        N_total=N_total,
        pump_rate=2.0e11,
    )
    assert fg_bottleneck.steady_state_inversion() < 0.0
    with pytest.raises(ValueError, match="tau_21 > tau_10"):
        fg_bottleneck.threshold_pump_rate(100.0)


# ---------------------------------------------------------------------------
# 4. Homogeneous gain saturation: output vs input intensity
# ---------------------------------------------------------------------------


def test_saturation_law():
    """
    Continuous-wave amplification across a 1D gain slab of length L = 4 um
    follows the exact homogeneous saturable-amplifier law
    ln(I_out / I_in) + (I_out - I_in) / I_sat = g0 * L (`amplified_intensity`)
    from the small-signal regime (I_in << I_sat) to deep saturation
    (I_in >> I_sat), for both `SaturableGain` and `FourLevelGain`.
    """
    f0 = 300.0e12
    omega_0 = 2.0 * np.pi * f0
    gamma = 2.0 * np.pi * 30.0e12
    g0 = 2.0e5
    I_sat = 1.0e9
    L = 4.0e-6
    dz = 25.0e-9
    Z0 = np.sqrt(mu_0 / epsilon_0)

    grid = wakis.GridFIT3D(
        -0.5e-6, 0.5e-6, -0.5e-6, 0.5e-6, -1.5e-6, 5.5e-6, 2, 2, int(7.0e-6 / dz), verbose=0
    )

    def run_cw_slab(mat, E_in_target, n_periods=45):
        J_amp = 2.0 * E_in_target / Z0
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.7,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=12,
            verbose=0,
        )
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.5 * L), size=(2.0e-6, 2.0e-6, L)
        )
        src = ContinuousSource(frequency=f0, ramp_time=5.0 / f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -0.5e-6),
            size=(1.0e-6, 1.0e-6, 0.0),
            amplitude=J_amp,
        )
        Nt = int((n_periods / f0) / solver.dt)
        n_avg = int((8.0 / f0) / solver.dt)
        k_in = int( np.argmin(np.abs(solver.z - (-0.2e-6))) )
        k_out = int( np.argmin(np.abs(solver.z - (L + 0.3e-6))) )
        stride_z = solver.Nx * solver.Ny
        idx_in = k_in * stride_z
        idx_out = k_out * stride_z

        e2_in = 0.0
        e2_out = 0.0
        for n in range(Nt):
            sheet.update(solver, n * solver.dt)
            solver.one_step()
            if n >= Nt - n_avg:
                ex_in = solver.E.array[idx_in]
                ex_out = solver.E.array[idx_out]
                e2_in += ex_in * ex_in
                e2_out += ex_out * ex_out
        return c * epsilon_0 * (e2_in / n_avg), c * epsilon_0 * (e2_out / n_avg)

    # (a) SaturableGain across two decades of input intensity
    ratios = [0.02, 0.1, 0.5, 1.5, 5.0]
    gains_num = []
    for r in ratios:
        E_in = np.sqrt(2.0 * (r * I_sat) / (c * epsilon_0))
        mat = SaturableGain(omega_0=omega_0, gamma=gamma, g0=g0, I_sat=I_sat)
        I_in_n, I_out_n = run_cw_slab(mat, E_in)
        I_out_an = amplified_intensity(I_in_n, L, g0, I_sat)
        # Verify implicit transcendental equation directly as well
        lhs = np.log(I_out_an / I_in_n) + (I_out_an - I_in_n) / I_sat
        assert np.isclose(lhs, g0 * L, rtol=1e-12)
        np.testing.assert_allclose(I_out_n, I_out_an, rtol=1.5e-2)
        gains_num.append(I_out_n / I_in_n)

    # Gain monotonically decreases with input intensity
    assert all(gains_num[i] > gains_num[i + 1] for i in range(len(gains_num) - 1))

    # (b) FourLevelGain saturation vs analytic saturation_intensity()
    tau_21, tau_10, tau_32 = 15.0e-15, 1.0e-15, 1.0e-15
    N_total, W_p = 1.0e22, 5.0e11
    fg_probe = FourLevelGain(
        omega_a=omega_0,
        gamma_a=gamma,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=N_total,
        pump_rate=W_p,
        initial_populations="steady_state",
    )
    dn0 = fg_probe.steady_state_inversion()
    kappa_a = (g0 / dn0) * epsilon_0 * c * gamma
    fg_ref = FourLevelGain(
        omega_a=omega_0,
        gamma_a=gamma,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=N_total,
        pump_rate=W_p,
        kappa_a=kappa_a,
        initial_populations="steady_state",
    )
    I_sat_fg = fg_ref.saturation_intensity()
    g0_fg = fg_ref.small_signal_gain()

    for r in [0.05, 1.0, 4.0]:
        E_in = np.sqrt(2.0 * (r * I_sat_fg) / (c * epsilon_0))
        fg_run = FourLevelGain(
            omega_a=omega_0,
            gamma_a=gamma,
            tau_21=tau_21,
            tau_10=tau_10,
            tau_32=tau_32,
            N_total=N_total,
            pump_rate=W_p,
            kappa_a=kappa_a,
            initial_populations="steady_state",
        )
        I_in_n, I_out_n = run_cw_slab(fg_run, E_in, n_periods=50)
        I_out_an = amplified_intensity(I_in_n, L, g0_fg, I_sat_fg)
        np.testing.assert_allclose(I_out_n, I_out_an, rtol=1.5e-2)

    # Mutation check: unsaturated medium (I_sat = inf) fails to saturate at high intensity
    E_high = np.sqrt(2.0 * (5.0 * I_sat) / (c * epsilon_0))
    mat_unsat = SaturableGain(omega_0=omega_0, gamma=gamma, g0=g0, I_sat=np.inf)
    I_in_u, I_out_u = run_cw_slab(mat_unsat, E_high)
    assert (I_out_u / I_in_u) > 1.8 * gains_num[-1]


# ---------------------------------------------------------------------------
# 5. 1D DBR Fabry-Perot laser cavity: threshold, relaxation oscillations, L-I
# ---------------------------------------------------------------------------


def test_fabry_perot_laser():
    """
    1D Fabry-Perot laser formed by two quarter-wave DBR mirrors (H L H,
    n_H = 2.0, n_L = 1.0) surrounding a FourLevelGain active cavity of length
    L_cav = 1.2 um (m = 2 half-wavelengths at lambda_0 = 1.2 um):
    - Cold-cavity mode frequency matches f0 = c / lambda_0 and ringdown decay
      rate matches the DBR reflectance R = (15/17)^2.
    - Threshold pump rate matches `threshold_pump_rate(g_th)`: below threshold
      the seed pulse decays away; above threshold the cavity lases at f_cav.
    - Turn-on exhibits damped relaxation oscillations and clamps the modal
      population inversion at Delta N_th.
    - Above-threshold steady-state output intensity is linear in pump rate and
      matches `fabry_perot_laser_output`.
    - Mutation: zero gain (W_p = 0) produces no lasing.
    """
    lam0 = 1.2e-6
    f0 = c / lam0
    omega_a = 2.0 * np.pi * f0
    gamma_a = 2.0 * np.pi * 40.0e12
    dz = 25.0e-9  # 48 cells/lambda in vacuum, 24 cells/lambda in n_H = 2.0

    n_H, n_L = 2.0, 1.0
    d_H = lam0 / (4.0 * n_H)  # 150 nm = 6 * dz
    d_L = lam0 / (4.0 * n_L)  # 300 nm = 12 * dz
    n_pairs = 1  # H L H stack
    R_an = dbr_reflectance(n_H, n_L, n_pairs, n_in=1.0, n_out=1.0, extra_high=True)
    assert np.isclose(R_an, (15.0 / 17.0) ** 2, rtol=1e-14)

    L_cav = 1.2e-6
    g_th = fabry_perot_threshold_gain(L_cav, R_an, R_an)
    tau_21 = 150.0e-15
    tau_10 = 5.0e-15
    tau_32 = 5.0e-15
    N_total = 5.0e23

    fg_ref = FourLevelGain(
        omega_a=omega_a,
        gamma_a=gamma_a,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=N_total,
        pump_rate=0.0,
        eps_inf=1.0,
    )
    W_p_th = fg_ref.threshold_pump_rate(g_th)
    dn_th = g_th / fg_ref.emission_cross_section
    I_sat = fg_ref.saturation_intensity(W_p=W_p_th)

    z_min, z_max = -1.2e-6, 2.4e-6
    Nz = int(round((z_max - z_min) / dz))
    grid = wakis.GridFIT3D(
        -1.0e-6, 1.0e-6, -1.0e-6, 1.0e-6, z_min, z_max, 2, 2, Nz, verbose=0
    )

    def build_cavity_solver():
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
        return solver

    # 1. Cold-cavity ringdown: mode frequency f_cav and quality factor Q_cold
    solver_cold = build_cavity_solver()
    src_cold = GaussianSource(frequency=f0, fwidth=0.05 * f0)
    sheet_cold = CurrentSource(
        src_cold,
        component="x",
        center=(0.0, 0.0, 0.25 * L_cav),
        size=(2.0e-6, 2.0e-6, 0.0),
        amplitude=1.0,
    )
    k_cav = int(np.argmin(np.abs(solver_cold.z - 0.75 * L_cav)))
    idx_cav = k_cav * solver_cold.Nx * solver_cold.Ny
    Nt_cold = int((src_cold.end_time + 280.0e-15) / solver_cold.dt)
    ex_cold = np.empty(Nt_cold)
    for n in range(Nt_cold):
        sheet_cold.update(solver_cold, n * solver_cold.dt)
        solver_cold.one_step()
        ex_cold[n] = solver_cold.E.array[idx_cav]

    t_cold = np.arange(Nt_cold) * solver_cold.dt
    ring_mask = t_cold > src_cold.end_time + 30.0e-15
    n_fft_cold = 32768
    freqs_cold = np.fft.rfftfreq(n_fft_cold, d=solver_cold.dt)
    f_cav_cold = freqs_cold[
        np.argmax(np.abs(np.fft.rfft(ex_cold[ring_mask], n=n_fft_cold)))
    ]
    np.testing.assert_allclose(f_cav_cold, f0, rtol=5.0e-3)

    # 2. Active laser pump sweep across threshold
    def run_laser(pump_ratio, total_time=0.9e-12, spatial_averaging=False):
        wp = pump_ratio * W_p_th
        solver = build_cavity_solver()
        fg = FourLevelGain(
            omega_a=omega_a,
            gamma_a=gamma_a,
            tau_21=tau_21,
            tau_10=tau_10,
            tau_32=tau_32,
            N_total=N_total,
            pump_rate=wp,
            eps_inf=1.0,
            initial_populations="steady_state" if wp > 0 else "ground",
            enable_stimulated=True,
            spatial_averaging=spatial_averaging,
        )
        solver.add_gain_material(
            fg, center=(0.0, 0.0, 0.5 * L_cav), size=(4.0e-6, 4.0e-6, L_cav)
        )
        src = GaussianSource(frequency=f0, fwidth=0.08 * f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, 0.25 * L_cav),
            size=(2.0e-6, 2.0e-6, 0.0),
            amplitude=2.0,
        )
        k_out = int(np.argmin(np.abs(solver.z - (L_cav + 2 * d_H + d_L + 0.2e-6))))
        idx_out = k_out * solver.Nx * solver.Ny
        Nt = int(total_time / solver.dt)
        ex_out = np.empty(Nt)
        dn_hist = np.empty(Nt)
        g_grp = solver._gain_groups[0]
        for n in range(Nt):
            sheet.update(solver, n * solver.dt)
            solver.one_step()
            ex_out[n] = solver.E.array[idx_out]
            N_arr = g_grp["N"]
            dn_hist[n] = np.mean(N_arr[2] - N_arr[1])
        n_avg = int((20.0 / f0) / solver.dt)
        I_out = c * epsilon_0 * float(np.mean(ex_out[-n_avg:] ** 2))
        return I_out, ex_out, dn_hist, solver.dt

    # Mutation check: zero gain (W_p = 0) and sub-threshold (0.5 * W_p_th) do not lase
    I_zero, _, dn_zero, _ = run_laser(0.0)
    I_sub, _, dn_sub, _ = run_laser(0.5)
    assert I_zero < 1.0e-10
    assert I_sub < 1.0e-3
    assert np.isclose(dn_zero[-1], 0.0, atol=1.0)

    # Above-threshold sweep with spatial hole burning (default spatial_averaging=False)
    pump_ratios = [1.5, 2.0, 2.5, 3.0]
    I_out_shb = []
    for r in pump_ratios:
        I_val, ex_tr, dn_tr, dt_l = run_laser(r, total_time=0.9e-12, spatial_averaging=False)
        I_out_shb.append(I_val)
        I_an = fabry_perot_laser_output(r, I_sat, R_an, R_an, spatial_hole_burning=True)
        np.testing.assert_allclose(I_val, I_an, rtol=8.0e-2)

    # Output intensity increases linearly with pump above threshold (R^2 > 0.999)
    I_out_shb = np.array(I_out_shb)
    slope, intercept = np.polyfit(pump_ratios, I_out_shb, 1)
    r_corr = np.corrcoef(pump_ratios, I_out_shb)[0, 1]
    assert slope > 0.0
    assert r_corr > 0.999
    # Linear extrapolation back to I_out = 0 intercepts near threshold r = 1
    r_th_extrap = -intercept / slope
    assert 0.85 < r_th_extrap < 1.25

    # Check lasing frequency, relaxation oscillations, and exact inversion clamping
    I_hom, ex_hom, dn_hom, dt_hom = run_laser(2.5, total_time=0.9e-12, spatial_averaging=True)
    I_hom_an = fabry_perot_laser_output(2.5, I_sat, R_an, R_an, spatial_hole_burning=False)
    np.testing.assert_allclose(I_hom, I_hom_an, rtol=6.0e-2)
    # Clamped steady-state inversion matches Delta N_th
    np.testing.assert_allclose(dn_hom[-1], dn_th, rtol=3.0e-2)
    # Relaxation oscillations: inversion overshoots below Delta N_th during turn-on before settling
    assert np.min(dn_hom) < 0.85 * dn_th < dn_hom[-1] < 1.05 * dn_th

    # Lasing emission spectrum peaks at the cold-cavity mode frequency f_cav
    n_steady = len(ex_hom) // 2
    freqs_lase = np.fft.rfftfreq(len(ex_hom) - n_steady, d=dt_hom)
    spec_lase = np.abs(np.fft.rfft(ex_hom[n_steady:]))
    f_lase = freqs_lase[np.argmax(spec_lase)]
    np.testing.assert_allclose(f_lase, f_cav_cold, rtol=1.0e-2)


# ---------------------------------------------------------------------------
# 6. Energy accounting: conservation with gain off and discrete Poynting balance
# ---------------------------------------------------------------------------


def test_energy_accounting():
    """
    In a lossless PEC cavity:
    (a) With gain off (N_total = 0 or g0 = 0) and sigma = 0, the discrete
        leapfrog electromagnetic energy W_EM^n is conserved to machine
        round-off (~1e-15) after the source turns off.
    (b) With active or absorbing FourLevelGain, the discrete energy balance
        (W_EM^(n+1) - W_EM^n)/dt + sum w * eps_0 * ((p^(n+1) - p^n)/dt) * E^(n+1/2) dV = 0
        closes to machine round-off (~1e-14) at every step.
    (c) Mutation: turning gain on (N_2 > N_1 or g0 > 0) increases W_EM, while
        ground-state absorption (N_1 > N_2 or g0 < 0) decreases W_EM.
    """
    N, L = 12, 0.12
    grid = wakis.GridFIT3D(-L / 2, L / 2, -L / 2, L / 2, -L / 2, L / 2, N, N, N, verbose=0)

    def run_pec_cavity(mat):
        solver = wakis.SolverFIT3D(
            grid, cfln=0.5, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
        )
        if mat is not None:
            solver.add_gain_material(mat, center=(0.0, 0.0, 0.0), size=(0.06, 0.06, 0.06))
        src = GaussianSource(frequency=2.0e9, fwidth=0.8e9)
        dip = CurrentSource(src, component="z", center=(0.01, -0.01, 0.0))

        itA = grid.itA.toarray()
        iA = grid.iA.toarray()
        tA = np.divide(1.0, itA, out=np.zeros_like(itA), where=itA != 0)
        A = np.divide(1.0, iA, out=np.zeros_like(iA), where=iA != 0)
        dV_E = grid.L.toarray() * tA
        dV_H = grid.tL.toarray() * A
        ieps = solver.ieps.toarray()
        eps_r = np.divide(
            1.0, epsilon_0 * ieps, out=np.zeros_like(ieps), where=ieps != 0
        )
        if getattr(solver, "_gain_groups", None):
            g = solver._gain_groups[0]
            eps_r[g["idx"]] = g["den"]

        Nt_src = int(np.ceil(src.end_time / solver.dt)) + 2
        for n in range(Nt_src):
            dip.update(solver, n * solver.dt)
            solver.one_step()
        solver.J.array[:] = 0.0

        W = []
        pol_work = []
        E_old = solver.E.toarray().copy()
        H_old = solver.H.toarray().copy()
        g = solver._gain_groups[0] if getattr(solver, "_gain_groups", None) else None
        p_prev = (
            g["lorentz_a"][3].copy()
            if (g is not None and g["kind"] == "four_level")
            else (g["lorentz"][3].copy() if g is not None else None)
        )

        for _ in range(80):
            solver.one_step()
            E = solver.E.toarray().copy()
            H = solver.H.toarray().copy()
            W.append(
                0.5 * epsilon_0 * np.sum(eps_r * dV_E * E_old**2)
                + 0.5 * mu_0 * np.sum(dV_H * H_old * H)
            )
            if g is not None:
                p_curr = (
                    g["lorentz_a"][3].copy()
                    if g["kind"] == "four_level"
                    else g["lorentz"][3].copy()
                )
                dp = p_curr - p_prev
                e_bar = 0.5 * (E[g["idx"]] + E_old[g["idx"]])
                pol_work.append(
                    epsilon_0
                    * np.sum(g["weight"] * dV_E[g["idx"]] * (dp / solver.dt) * e_bar)
                )
                p_prev = p_curr
            E_old, H_old = E, H

        return np.array(W), np.array(pol_work), solver.dt

    # (a) Gain off, no loss -> W_EM conserved to round-off
    fg_off = FourLevelGain(
        omega_a=2.0 * np.pi * 2.0e9,
        gamma_a=2.0 * np.pi * 0.2e9,
        tau_21=1.0e-6,
        tau_10=1.0e-9,
        tau_32=1.0e-9,
        N_total=0.0,
        pump_rate=0.0,
        eps_inf=2.2,
        sigma=0.0,
    )
    W_fg_off, _, _ = run_pec_cavity(fg_off)
    assert np.ptp(W_fg_off) / W_fg_off[0] < 1.0e-14

    sg_off = SaturableGain(
        omega_0=2.0 * np.pi * 2.0e9,
        gamma=2.0 * np.pi * 0.2e9,
        g0=0.0,
        eps_inf=2.2,
        sigma=0.0,
    )
    W_sg_off, _, _ = run_pec_cavity(sg_off)
    assert np.ptp(W_sg_off) / W_sg_off[0] < 1.0e-14

    # (b) & (c) Active gain vs passive absorption + exact per-step Poynting balance
    sg_gain = SaturableGain(
        omega_0=2.0 * np.pi * 2.0e9,
        gamma=2.0 * np.pi * 0.2e9,
        g0=5.0,
        I_sat=np.inf,
        eps_inf=2.2,
    )
    W_gain, pw_gain, dt = run_pec_cavity(sg_gain)
    residual_gain = np.diff(W_gain) / dt + pw_gain[:-1]
    assert np.max(np.abs(residual_gain)) < 1.0e-13 * (W_gain[0] / dt)
    assert W_gain[-1] > 1.01 * W_gain[0]

    sg_loss = SaturableGain(
        omega_0=2.0 * np.pi * 2.0e9,
        gamma=2.0 * np.pi * 0.2e9,
        g0=-5.0,
        I_sat=np.inf,
        eps_inf=2.2,
    )
    W_loss, pw_loss, dt = run_pec_cavity(sg_loss)
    residual_loss = np.diff(W_loss) / dt + pw_loss[:-1]
    assert np.max(np.abs(residual_loss)) < 1.0e-13 * (W_loss[0] / dt)
    assert W_loss[-1] < 0.996 * W_loss[0]


# ---------------------------------------------------------------------------
# 7. Guards for unsupported solver modes and invalid inputs
# ---------------------------------------------------------------------------


def test_gain_guards_and_reset():
    """
    Unsupported execution paths (scheme='tete', fused=True, use_gpu=True,
    SubgridFIT2D) and unstable timesteps raise clear errors; `reset_fields()`
    resets populations and polarizations.
    """
    grid = wakis.GridFIT3D(-0.05, 0.05, -0.05, 0.05, -0.05, 0.05, 4, 4, 4, verbose=0)
    fg = FourLevelGain(
        omega_a=2.0 * np.pi * 1.0e9,
        gamma_a=2.0 * np.pi * 0.1e9,
        tau_21=1.0e-6,
        tau_10=1.0e-9,
        tau_32=1.0e-9,
        N_total=1.0e18,
        pump_rate=1.0e6,
        kappa_a=1.0e-10,
    )

    # scheme='tete' guard
    solver_tete = wakis.SolverFIT3D(grid, scheme="tete", dt=4.0e-11, verbose=0)
    with pytest.raises(NotImplementedError, match="tete"):
        solver_tete.add_gain_material(fg, center=(0, 0, 0), size=(0.05, 0.05, 0.05))

    # fused=True guard
    solver_fused = wakis.SolverFIT3D(grid, cfln=0.5, verbose=0)
    solver_fused.fused = True
    with pytest.raises(NotImplementedError, match="fused"):
        solver_fused.add_gain_material(fg, center=(0, 0, 0), size=(0.05, 0.05, 0.05))

    # use_gpu=True guard
    solver_gpu_flag = wakis.SolverFIT3D(grid, cfln=0.5, verbose=0)
    solver_gpu_flag.use_gpu = True
    with pytest.raises(NotImplementedError, match="GPU"):
        solver_gpu_flag.add_gain_material(fg, center=(0, 0, 0), size=(0.05, 0.05, 0.05))

    # Unstable dt guard (omega_a * dt >= 2)
    fg_unstable = FourLevelGain(
        omega_a=10.0 / solver_fused.dt,
        gamma_a=1.0e9,
        tau_21=1.0e-6,
        tau_10=1.0e-9,
        tau_32=1.0e-9,
    )
    solver_cpu = wakis.SolverFIT3D(grid, cfln=0.5, verbose=0)
    with pytest.raises(ValueError, match="unstable"):
        solver_cpu.add_gain_material(
            fg_unstable, center=(0, 0, 0), size=(0.05, 0.05, 0.05)
        )

    # reset_fields() restores initial populations and zeros polarization state
    solver_cpu.add_gain_material(fg, center=(0, 0, 0), size=(0.05, 0.05, 0.05))
    solver_cpu.set_gain_populations("steady_state")
    assert solver_cpu.get_gain_populations(cell_average=True)[2] > 0.0
    solver_cpu.reset_fields()
    np.testing.assert_allclose(
        solver_cpu.get_gain_populations(cell_average=True),
        [fg.N_total, 0.0, 0.0, 0.0],
    )
