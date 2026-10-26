# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Validation tests for spontaneous-emission Langevin noise and lifetime
acceleration (time scaling) in active gain media (`wakis.gain`):

1. Default deterministic behaviour (`spontaneous_emission=False`) unchanged,
   zero noise -> no lasing from rest, and reproducible seeded RNG trajectories.
2. Spontaneous emission power and decay from an inverted thin layer below
   threshold vs the fluctuation-dissipation theorem (FDT) rate
   ``f_rad,1D * V * N_2 * hbar * omega_a / tau_21``, linear scaling with ``N_2``,
   and quadratic power scaling with ``noise_amplitude``.
3. Amplified spontaneous emission (ASE) spectrum of a 1D gain slab below
   threshold vs the analytic gain-narrowed profile
   ``S(w) ~ (exp(g(w) L) - 1) g_sp(w) / g(w)``, plus mutation checks on
   ``noise_exponent`` and gain narrowing.
4. Fabry-Perot lasing from spontaneous-emission noise, light-pump (L-I) curve
   with spontaneous-emission ``beta``-factor rounding at threshold vs the
   coupled rate-equation solution, and mode competition in a multi-mode cavity
   under homogeneous broadening.
5. Lifetime acceleration (`time_scale = s` for ``s = 1, 10, 100, 1000``):
   invariance of threshold, steady-state populations, steady-state output
   intensity, and lasing spectrum peak position; scaling of turn-on delay
   ``t_th ~ 1/s`` and relaxation-oscillation frequency ``f_RO ~ sqrt(s)``; and
   adiabaticity limits ``(s / tau_sat) << kappa_cav, gamma_a``.
"""

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar

import wakis
from wakis.gain import (
    FourLevelGain,
    _rate_cn_matrices,
    ase_spectrum_1d,
    dbr_reflectance,
    fabry_perot_laser_output,
    fabry_perot_threshold_gain,
)
from wakis.resonances import ResonanceMonitor


def _make_1d_solver(z_min, z_max, dz, lx=2.0e-6, ly=2.0e-6, n_pml=10, cfln=0.8):
    """Create a 1D-periodic (x, y) SolverFIT3D with CPML along z."""
    nz = int(round((z_max - z_min) / dz))
    grid = wakis.GridFIT3D(
        -0.5 * lx,
        0.5 * lx,
        -0.5 * ly,
        0.5 * ly,
        z_min,
        z_max,
        2,
        2,
        nz,
        verbose=0,
    )
    return wakis.SolverFIT3D(
        grid,
        cfln=cfln,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        verbose=0,
    )


def _add_dbr_mirrors(solver, L_cav, d_H, d_L, n_H, n_pairs):
    """Stamp symmetric quarter-wave DBR mirrors (n_pairs (H, L) pairs + 1 extra H layer = (2*n_pairs + 1) layers) on both sides of [0, L_cav]."""
    ieps_arr = solver.ieps.array
    lx_span = 2.0 * float(solver.grid.xmax - solver.grid.xmin)
    ly_span = 2.0 * float(solver.grid.ymax - solver.grid.ymin)
    for m in range(n_pairs + 1):
        z_left = -(m * (d_H + d_L) + 0.5 * d_H)
        z_right = L_cav + m * (d_H + d_L) + 0.5 * d_H
        for z_c in (z_left, z_right):
            idx_e, w_e = solver._box_edges(
                (0.0, 0.0, z_c), (lx_span, ly_span, d_H), dual=False
            )
            eps_s = 1.0 / (eps_0 * ieps_arr[idx_e])
            eps_new = (1.0 - w_e) * eps_s + w_e * (n_H**2)
            ieps_arr[idx_e] = 1.0 / (eps_0 * eps_new)
    solver.update_tensors("ieps")


def test_default_noise_off_and_seeded_reproducibility():
    """
    Verify:
    1. Default `spontaneous_emission=False` leaves zero-field rest state silent,
       and driven deterministic pulse propagation is bit-identical to stored
       pre-F11b reference IEEE-754 floats.
    2. Seeded `spontaneous_emission=True` is bit-for-bit reproducible across runs
       and after `reset_gain_state()`.
    3. Auto-detected transverse-uniform noise is restricted to 1D-periodic plane-wave
       slabs (`Nx <= 2 and Ny <= 2`), whereas 3D periodic cells (`Nx, Ny > 2`) draw
       independent per-cell Langevin noise kicks.
    """
    lambda0 = 1.2e-6
    f0 = c_light / lambda0
    omega0 = 2.0 * np.pi * f0
    gamma_a = 1.0e14
    dz = 30.0e-9

    # 1. Default spontaneous_emission=False -> zero noise, no lasing from rest
    solver_quiet = _make_1d_solver(-0.9e-6, 0.9e-6, dz)
    mat_quiet = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=1.0e-12,
        tau_10=20.0e-15,
        tau_32=10.0e-15,
        N_total=1.0e24,
        pump_rate=1.0e11,
        initial_populations="steady_state",
    )
    assert mat_quiet.spontaneous_emission is False
    assert mat_quiet.noise_amplitude == 0.0
    assert mat_quiet.time_scale == 1.0
    solver_quiet.add_gain_material(
        mat_quiet, center=(0.0, 0.0, 0.0), size=(4.0e-6, 4.0e-6, 0.6e-6)
    )
    for _ in range(200):
        solver_quiet.one_step()
    assert np.max(np.abs(solver_quiet.E.array)) == 0.0
    assert np.max(np.abs(solver_quiet.H.array)) == 0.0

    # 1b. Bit-identity against in-process pre-F11b reference path (noise off, time_scale=1.0)
    #     Avoids platform-specific OpenBLAS vs macOS Accelerate ULP differences while
    #     verifying exact bit-identity of the default deterministic code path.
    from wakis.sources import CurrentSource, GaussianSource

    def run_deterministic_trace(
        spont=False, t_scale=1.0, noise_amp=None, use_pre_f11b_ref=False
    ):
        s_det = _make_1d_solver(-0.9e-6, 0.9e-6, dz, lx=2.0e-6, ly=2.0e-6, cfln=0.5)
        m_det = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=1.0e-12,
            tau_10=20.0e-15,
            tau_32=10.0e-15,
            N_total=1.0e23,
            pump_rate=1.0e11,
            initial_populations="steady_state",
            spontaneous_emission=spont,
            noise_amplitude=noise_amp,
            time_scale=t_scale,
            seed=42,
        )
        s_det.add_gain_material(
            m_det, center=(0.0, 0.0, 0.0), size=(4.0e-6, 4.0e-6, 0.6e-6)
        )
        if use_pre_f11b_ref:
            # Explicitly construct the pre-F11b Crank-Nicolson matrices directly from
            # the unscaled rate_matrix and strip noise_channels / RNG state:
            g0 = s_det._gain_groups[0]
            dt_raw = float(s_det.dt)
            m_raw = m_det.rate_matrix(W_p=m_det.pump_rate, apply_time_scale=False)
            eye4 = np.eye(4, dtype=float)
            lhs_inv = np.linalg.inv(eye4 - 0.5 * dt_raw * m_raw)
            g0["A_rate"] = lhs_inv @ (eye4 + 0.5 * dt_raw * m_raw)
            g0["B_rate"] = dt_raw * lhs_inv
            g0["noise_channels"] = ()
            g0["rng"] = None
        src = GaussianSource(frequency=f0, fwidth=0.1 * f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -0.45e-6),
            size=(2.0e-6, 2.0e-6, 0.0),
            amplitude=1.0,
        )
        e_tr = np.empty(10, dtype=float)
        for n in range(200):
            sheet.update(s_det, n * s_det.dt)
            s_det.one_step()
            if n % 20 == 0:
                e_tr[n // 20] = float(s_det.E[0, 0, s_det.Nz // 2, "x"])
        n_end = s_det.get_gain_populations(cell_average=True)
        return e_tr, n_end

    e_ref, n_ref = run_deterministic_trace(
        spont=False, t_scale=1.0, use_pre_f11b_ref=True
    )
    e_det, n_det = run_deterministic_trace(spont=False, t_scale=1.0)
    e_zero_amp, n_zero_amp = run_deterministic_trace(
        spont=True, t_scale=1.0, noise_amp=0.0
    )
    np.testing.assert_array_equal(e_det, e_ref)
    np.testing.assert_array_equal(n_det, n_ref)
    np.testing.assert_array_equal(e_zero_amp, e_ref)
    np.testing.assert_array_equal(n_zero_amp, n_ref)
    # Mutation check: enabling spontaneous emission or changing time_scale alters the trace
    e_mut_spont, _ = run_deterministic_trace(spont=True, t_scale=1.0)
    _, n_mut_scale = run_deterministic_trace(spont=False, t_scale=2.0)
    assert not np.array_equal(e_mut_spont, e_ref)
    assert not np.array_equal(n_mut_scale, n_ref)

    # 2. Seeded spontaneous emission -> reproducible across runs and after reset_gain_state()
    def run_seeded(seed_val):
        s = _make_1d_solver(-0.9e-6, 0.9e-6, dz)
        m = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=1.0e-12,
            tau_10=20.0e-15,
            tau_32=10.0e-15,
            N_total=1.0e24,
            pump_rate=1.0e11,
            initial_populations="steady_state",
            spontaneous_emission=True,
            seed=seed_val,
        )
        s.add_gain_material(m, center=(0.0, 0.0, 0.0), size=(4.0e-6, 4.0e-6, 0.6e-6))
        trace = np.empty(200)
        for n in range(200):
            s.one_step()
            trace[n] = float(s.E[0, 0, s.Nz // 2, "x"])
        return s, trace

    solver_a, tr_a = run_seeded(42)
    _, tr_b = run_seeded(42)
    _, tr_c = run_seeded(99)

    assert np.max(np.abs(tr_a)) > 0.0
    np.testing.assert_array_equal(tr_a, tr_b)
    assert not np.allclose(tr_a, tr_c)

    # Resetting solver_a and zeroing E/H/CPML reproduces tr_a identically
    solver_a.E.array[:] = 0.0
    solver_a.H.array[:] = 0.0
    for attr in dir(solver_a):
        if attr.startswith("psi"):
            getattr(solver_a, attr)[:] = 0.0
    solver_a.reset_gain_state()
    tr_reset = np.empty(200)
    for n in range(200):
        solver_a.one_step()
        tr_reset[n] = float(solver_a.E[0, 0, solver_a.Nz // 2, "x"])
    np.testing.assert_array_equal(tr_a, tr_reset)

    # 3. 3D periodic grid with Nx=4, Ny=4 > 2 does NOT use transverse-uniform noise
    grid_3d_per = wakis.GridFIT3D(
        -1.0e-6, 1.0e-6, -1.0e-6, 1.0e-6, -0.3e-6, 0.3e-6, 4, 4, 10, verbose=0
    )
    solver_3d_per = wakis.SolverFIT3D(
        grid_3d_per,
        cfln=0.5,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=2,
        verbose=0,
    )
    mat_3d_per = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=1.0e-12,
        tau_10=20.0e-15,
        tau_32=10.0e-15,
        N_total=1.0e24,
        pump_rate=1.0e11,
        initial_populations="steady_state",
        spontaneous_emission=True,
        seed=42,
    )
    solver_3d_per.add_gain_material(
        mat_3d_per, center=(0.0, 0.0, 0.0), size=(3.0e-6, 3.0e-6, 0.2e-6)
    )
    ch_3d = solver_3d_per._gain_groups[0]["noise_channels"][0]
    assert ch_3d["draw_inv"] is None
    solver_3d_per.one_step()
    assert float(solver_3d_per.E[0, 0, 5, "x"]) != float(solver_3d_per.E[1, 0, 5, "x"])
    # Mutation check: forcing transverse_uniform_noise=True makes transverse cells identical
    mat_3d_per.transverse_uniform_noise = True
    from wakis.gain import _build_group_noise_meta

    _build_group_noise_meta(solver_3d_per, solver_3d_per._gain_groups[0])
    solver_3d_per.E.array[:] = 0.0
    solver_3d_per.H.array[:] = 0.0
    solver_3d_per.reset_gain_state()
    solver_3d_per.one_step()
    assert float(solver_3d_per.E[0, 0, 5, "x"]) == float(solver_3d_per.E[1, 0, 5, "x"])

    # Input validation guards
    with pytest.raises(ValueError, match="noise_amplitude"):
        FourLevelGain(omega0, gamma_a, 1e-12, 1e-14, 1e-14, noise_amplitude=-1.0)
    with pytest.raises(ValueError, match="noise_exponent"):
        FourLevelGain(omega0, gamma_a, 1e-12, 1e-14, 1e-14, noise_exponent=-0.5)
    with pytest.raises(ValueError, match="time_scale"):
        FourLevelGain(omega0, gamma_a, 1e-12, 1e-14, 1e-14, time_scale=0.0)
    with pytest.raises(ValueError, match="noise_components"):
        FourLevelGain(omega0, gamma_a, 1e-12, 1e-14, 1e-14, noise_components="w")


def test_spontaneous_decay_power_and_noise_scaling():
    """
    Validation (a) & (c):
    - Multi-seed ensemble average of an optically thin inverted layer below
      threshold radiates power matching the fluctuation-dissipation theorem (FDT)
      rate ``f_rad,1D * V * N_2 * hbar * omega_a / tau_21 / (1 - (omega_a * dt / 2)^2)``
      with ``f_rad,1D = 3 * lambda_0^2 / (4 * pi * n_bg * A_perp)`` within the
      ensemble standard error ``sigma_SEM = sigma_1 / sqrt(N_seeds)``.
    - Consistency checks: with fixed seed, radiated power scales linearly with
      ``N_2`` and quadratically with ``noise_amplitude``.
    - Mutation checks: wrong radiative fraction or wrong noise exponent fails.
    """
    lambda0 = 1.2e-6
    f0 = c_light / lambda0
    omega0 = 2.0 * np.pi * f0
    gamma_a = 2.5e14
    dz = 30.0e-9
    lx = ly = 2.0e-6
    area_perp = lx * ly
    L_layer = 2.0 * dz  # 60 nm optically thin layer (lambda0 / 20)
    vol = area_perp * L_layer

    tau_21_phys = 150.0e-15
    kappa_a_phys = 6.0 * np.pi * eps_0 * c_light**3 / (omega0**2 * tau_21_phys)

    def measure_thin_layer_power(
        N2_val, noise_amp=1.0, noise_exp=0.5, n_steps=20000, seed_val=42
    ):
        solver = _make_1d_solver(-1.5e-6, 1.5e-6, dz, lx=lx, ly=ly, cfln=0.5)
        # Clamp populations in the optically thin sub-threshold regime (g0 * L << 1)
        mat = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=tau_21_phys,
            tau_10=10.0e-15,
            tau_32=5.0e-15,
            kappa_a=kappa_a_phys,
            initial_populations=(1.0e23, 0.0, N2_val, 0.0),
            enable_stimulated=False,
            spontaneous_emission=True,
            noise_amplitude=noise_amp,
            noise_exponent=noise_exp,
            seed=seed_val,
        )
        # Freeze rate matrix so N2 stays constant during the stationary power measurement
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.5 * L_layer), size=(2.0 * lx, 2.0 * ly, L_layer)
        )
        g = solver._gain_groups[0]
        g["A_rate"] = np.eye(4)
        g["B_rate"] = np.zeros((4, 4))

        k_R = int(np.argmin(np.abs(solver.z - (L_layer + 0.4e-6))))
        k_L = int(np.argmin(np.abs(solver.z - (-0.4e-6))))
        stride = solver.Nx * solver.Ny
        ex_R = np.empty(n_steps)
        ex_L = np.empty(n_steps)
        hy_R = np.empty(n_steps)
        hy_L = np.empty(n_steps)
        for n in range(n_steps):
            solver.one_step()
            ex_R[n] = solver.E.array[k_R * stride]
            ex_L[n] = solver.E.array[k_L * stride]
            hy_R[n] = 0.5 * (
                solver.H.array[solver.N + k_R * stride]
                + solver.H.array[solver.N + (k_R - 1) * stride]
            )
            hy_L[n] = 0.5 * (
                solver.H.array[solver.N + k_L * stride]
                + solver.H.array[solver.N + (k_L - 1) * stride]
            )
        # Total time-averaged discrete Poynting power exiting both faces (+z and -z)
        s_plus = float(np.mean(ex_R[2000:-1] * 0.5 * (hy_R[2000:-1] + hy_R[2001:])))
        s_minus = float(np.mean(-ex_L[2000:-1] * 0.5 * (hy_L[2000:-1] + hy_L[2001:])))
        p_meas = area_perp * (s_plus + s_minus)
        p_theory = mat.spontaneous_emission_power(
            volume=vol, N_2=N2_val, area_perp=area_perp, dt=solver.dt
        )
        return p_meas, p_theory, mat

    # Multi-seed ensemble average across 8 independent RNG seeds:
    # For bandwidth Delta f = gamma_a / (2 pi) ~ 39.8 THz and averaging window
    # T_avg = 18000 * dt ~ 0.90 ps, a single seed has statistical standard deviation
    # sigma_1 / P_th ~ 1 / sqrt(B_eff * T_avg) ~ 7.1%; averaging N_seeds = 8 reduces
    # the standard error of the mean to sigma_SEM = sigma_1 / sqrt(8) ~ 2.5%.
    seeds = (101, 102, 103, 104, 105, 106, 107, 108)
    p_ensemble = []
    for sd in seeds:
        p_sd, p_th_1, mat_1 = measure_thin_layer_power(
            1.0e14, noise_amp=1.0, seed_val=sd
        )
        p_ensemble.append(p_sd)
    p_ensemble = np.asarray(p_ensemble, dtype=float)
    p_mean = float(np.mean(p_ensemble))
    sigma_1_rel = float(np.std(p_ensemble, ddof=1) / p_th_1)
    sigma_sem_rel = float(sigma_1_rel / np.sqrt(len(seeds)))
    assert 0.03 < sigma_1_rel < 0.14
    assert 0.01 < sigma_sem_rel < 0.05

    f_rad_expected = 3.0 * lambda0**2 / (4.0 * np.pi * area_perp)
    assert np.isclose(
        mat_1.radiative_fraction_1d(area_perp), f_rad_expected, rtol=1e-12, atol=0.0
    )
    rel_err = abs(p_mean - p_th_1) / p_th_1
    # Tolerance justified from ensemble standard error: 2.5 * sigma_SEM (~6.3%)
    assert rel_err < 2.5 * sigma_sem_rel, (
        f"Expected ensemble FDT power within 2.5*sigma_SEM={2.5 * sigma_sem_rel:.4f}, "
        f"got rel_err={rel_err:.4f}"
    )

    # Consistency check: with fixed seed, linear scaling in N_2 (4x N_2 -> 4x power)
    p_4n, p_th_4n, _ = measure_thin_layer_power(4.0e14, noise_amp=1.0, n_steps=5000)
    p_1_short, _, _ = measure_thin_layer_power(1.0e14, noise_amp=1.0, n_steps=5000)
    assert np.isclose(p_4n / p_1_short, 4.0, rtol=1e-5, atol=0.0)
    assert np.isclose(p_th_4n / p_th_1, 4.0, rtol=1e-12, atol=0.0)

    # Consistency check: with fixed seed, quadratic power scaling in noise_amplitude
    p_amp2, p_th_amp2, _ = measure_thin_layer_power(1.0e14, noise_amp=2.0, n_steps=5000)
    assert np.isclose(p_amp2 / p_1_short, 4.0, rtol=1e-10, atol=0.0)
    assert np.isclose(p_th_amp2 / p_th_1, 4.0, rtol=1e-12, atol=0.0)

    # Mutation checks:
    # (1) Dropping the 1D radiative fraction f_rad,1D changes predicted power by ~11.6x
    p_mut_no_frad = mat_1.spontaneous_emission_power(
        volume=vol, N_2=1.0e14, area_perp=None
    )
    assert abs(p_mean - p_mut_no_frad) / p_th_1 > 5.0
    # (2) Mutating noise_exponent from 0.5 to 0.55 breaks the linear power ratio p(4*N2)/p(N2) = 4
    p_mut_exp, _, _ = measure_thin_layer_power(
        4.0e14, noise_amp=1.0, noise_exp=0.55, n_steps=4000
    )
    p_base_short, _, _ = measure_thin_layer_power(
        1.0e14, noise_amp=1.0, noise_exp=0.55, n_steps=4000
    )
    assert not np.isclose(p_mut_exp / p_base_short, 4.0, rtol=0.05, atol=0.0)


def test_ase_spectrum_and_noise_exponent_mutation():
    """
    Validation (b) & (d):
    - Amplified spontaneous emission (ASE) spectrum of a 1D gain slab below
      threshold matches ``S(w) ~ (exp(g(w) L) - 1) g_sp(w) / g(w)`` and exhibits
      analytic gain narrowing as ``g_0 * L`` increases.
    - Mutation check (d): wrong ``noise_exponent`` changes the ASE spectrum and
      fails the benchmark check; omitting exponential gain narrowing also fails.
    """
    lambda0 = 1.2e-6
    f0 = c_light / lambda0
    omega0 = 2.0 * np.pi * f0
    gamma_a = 3.5e14
    dz = 30.0e-9
    L_slab = 2.4e-6
    pad = 0.6e-6
    Lz = L_slab + 2.0 * pad

    def rms_bandwidth(freqs, psd, f_center=f0, half_band=80.0e12):
        mask = np.abs(freqs - f_center) <= half_band
        f_m = freqs[mask]
        w_m = psd[mask] / np.trapezoid(psd[mask], f_m)
        f_mean = np.trapezoid(f_m * w_m, f_m)
        return float(np.sqrt(np.trapezoid((f_m - f_mean) ** 2 * w_m, f_m)))

    def run_slab_ase(g0L, noise_exp=0.5, Nt=18000):
        solver = _make_1d_solver(-0.5 * Lz, 0.5 * Lz, dz)
        mat_ref = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=10.0e-12,
            tau_10=20.0e-15,
            tau_32=10.0e-15,
            N_total=1.0e25,
        )
        N2_val = (g0L / L_slab) / mat_ref.emission_cross_section
        N0_val = 1.0e25 - N2_val
        pops = (N0_val, 0.0, N2_val, 0.0)
        mat = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=1.0,
            tau_10=1.0,
            tau_32=1.0,
            kappa_a=mat_ref.kappa_a,
            initial_populations=pops,
            enable_stimulated=False,
            spontaneous_emission=True,
            noise_exponent=noise_exp,
            noise_components="xy",
            seed=100,
        )
        solver.add_gain_material(
            mat, center=(0.0, 0.0, 0.0), size=(4.0e-6, 4.0e-6, L_slab)
        )
        k_R = int(np.argmin(np.abs(solver.z - (0.5 * L_slab + 0.18e-6))))
        k_L = int(np.argmin(np.abs(solver.z - (-0.5 * L_slab - 0.18e-6))))
        idx_R_x = k_R * solver.Nx * solver.Ny
        idx_L_x = k_L * solver.Nx * solver.Ny
        idx_R_y = solver.N + idx_R_x
        idx_L_y = solver.N + idx_L_x

        traces = np.empty((4, Nt), dtype=float)
        for n in range(Nt):
            solver.one_step()
            traces[0, n] = solver.E.array[idx_R_x]
            traces[1, n] = solver.E.array[idx_L_x]
            traces[2, n] = solver.E.array[idx_R_y]
            traces[3, n] = solver.E.array[idx_L_y]

        nperseg = 2048
        step = 1024
        win = np.hanning(nperseg)
        psd = np.zeros(nperseg // 2 + 1)
        count = 0
        for ch in range(4):
            sig = traces[ch, 1000:]
            for start in range(0, len(sig) - nperseg, step):
                psd += np.abs(np.fft.rfft(sig[start : start + nperseg] * win)) ** 2
                count += 1
        psd /= count
        freqs = np.fft.rfftfreq(nperseg, d=solver.dt)
        ase_th = ase_spectrum_1d(freqs, L_slab, mat, populations=pops, dt=solver.dt)
        return freqs, psd, ase_th, mat, pops, solver.dt

    freqs_lo, psd_lo, ase_lo, mat_lo, pops_lo, dt_lo = run_slab_ase(0.15)
    freqs_hi, psd_hi, ase_hi, mat_hi, pops_hi, dt_hi = run_slab_ase(2.80)

    w_fit_lo = rms_bandwidth(freqs_lo, psd_lo)
    w_th_lo = rms_bandwidth(freqs_lo, ase_lo)
    w_fit_hi = rms_bandwidth(freqs_hi, psd_hi)
    w_th_hi = rms_bandwidth(freqs_hi, ase_hi)

    # Both low-gain and high-gain ASE spectral widths match theory within 3%
    assert np.isclose(w_fit_lo, w_th_lo, rtol=0.03, atol=0.0)
    assert np.isclose(w_fit_hi, w_th_hi, rtol=0.03, atol=0.0)
    # Significant gain narrowing (> 20% narrower at g0*L = 2.80 than at g0*L = 0.15)
    assert w_fit_hi < 0.78 * w_fit_lo
    assert mat_hi.ase_linewidth(L_slab, populations=pops_hi) < 0.65 * mat_lo.ase_linewidth(
        L_slab, populations=pops_lo
    )

    # Ratio of integrated ASE spectral power between g0*L=2.80 and g0*L=0.15 matches theory
    band = (freqs_lo >= 170.0e12) & (freqs_lo <= 330.0e12)
    ratio_fit = np.trapezoid(psd_hi[band], freqs_hi[band]) / np.trapezoid(
        psd_lo[band], freqs_lo[band]
    )
    ratio_th = np.trapezoid(ase_hi[band], freqs_hi[band]) / np.trapezoid(
        ase_lo[band], freqs_lo[band]
    )
    assert np.isclose(ratio_fit, ratio_th, rtol=0.06, atol=0.0)

    # Mutation checks (d):
    # (1) Wrong noise_exponent (p_noise = 0.55 instead of 0.50) scales the high-to-low
    #     gain ASE power ratio by (2.80 / 0.15)**(2 * (0.55 - 0.50)) = 18.67**0.10 = 1.34x
    #     (+34%), breaking agreement with the FDT (p_noise = 0.50) ASE ratio
    _, psd_hi_mut, _, _, _, _ = run_slab_ase(2.80, noise_exp=0.55, Nt=6000)
    _, psd_lo_mut, _, _, _, _ = run_slab_ase(0.15, noise_exp=0.55, Nt=6000)
    ratio_mut = np.trapezoid(psd_hi_mut[band], freqs_hi[band]) / np.trapezoid(
        psd_lo_mut[band], freqs_lo[band]
    )
    assert not np.isclose(ratio_mut, ratio_th, rtol=0.15, atol=0.0)

    # (2) Omitting the exponential gain-narrowing factor (comparing high-gain ASE against
    #     unamplified Lorentzian g_sp * L) fails the bandwidth check
    assert not np.isclose(w_fit_hi, w_th_lo, rtol=0.15, atol=0.0)


def test_fabry_perot_lasing_from_noise_beta_curve_and_mode_competition():
    """
    Verify:
    1. Fabry-Perot cavity starts lasing purely from physical FDT spontaneous
       emission noise (`noise_amplitude=1.0`, no external seed pulse). The
       spontaneous-emission coupling factor ``beta_sp`` predicted a priori from
       ``f_rad,1D``, the effective cavity length ``L_eff = L_cav + 2 * L_pen``,
       and the Lorentzian spectral overlap matches the threshold-extracted
       ``beta_sp,fit = s(1)^2``, and reproduces the full L-I curve across
       threshold both a priori (zero free parameters) and out-of-sample.
    2. In a long multi-mode Fabry-Perot cavity under homogeneous broadening,
       multiple longitudinal modes grow initially from noise, and the central
       mode wins in steady state (mode competition).
    """
    lambda0 = 1.2e-6
    f0 = c_light / lambda0
    omega0 = 2.0 * np.pi * f0
    gamma_a = 1.2e14
    dz = 25.0e-9
    lx = ly = 3.6e-6
    area_perp = lx * ly
    n_H, n_L = 2.0, 1.0
    d_H = lambda0 / (4.0 * n_H)
    d_L = lambda0 / (4.0 * n_L)
    n_pairs = 1
    L_cav = 1.2e-6

    R_dbr = dbr_reflectance(n_H, n_L, n_pairs, n_in=1.0, n_out=1.0, extra_high=True)
    g_th = fabry_perot_threshold_gain(L_cav, R_dbr, R_dbr)

    tau_21 = 150.0e-15
    tau_10 = 5.0e-15
    tau_32 = 5.0e-15
    N_total = 5.0e23
    fg_ref = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=N_total,
    )

    # A priori theoretical spontaneous-emission coupling factor beta_sp:
    # Effective cavity length including DBR phase penetration L_pen = lambda0 / (4 * (n_H - n_L)):
    L_pen = lambda0 / (4.0 * (n_H - n_L))
    L_eff = L_cav + 2.0 * L_pen
    gamma_z = L_cav / L_eff
    # Fraction of Lorentzian spontaneous emission spectrum falling within one
    # longitudinal mode spacing Delta omega_FSR = pi * c / L_eff:
    dw_fsr = np.pi * c_light / L_eff
    spectral_overlap = float((2.0 / np.pi) * np.arctan(dw_fsr / gamma_a))
    beta_theory = float(
        fg_ref.radiative_fraction_1d(area_perp) * gamma_z * spectral_overlap
    )
    assert 0.01 < beta_theory < 0.025

    def run_fp_noise_point(pump_ratio, total_time=1.25e-12, pref_scale=1.0):
        wp = fg_ref.threshold_pump_rate(pump_ratio * g_th)
        solver = _make_1d_solver(-1.2e-6, 2.4e-6, dz, lx=lx, ly=ly)
        _add_dbr_mirrors(solver, L_cav, d_H, d_L, n_H, n_pairs)
        fg = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=tau_21,
            tau_10=tau_10,
            tau_32=tau_32,
            N_total=N_total,
            pump_rate=wp,
            initial_populations="steady_state",
            spatial_averaging=True,
            spontaneous_emission=True,
            noise_amplitude=1.0,
            seed=42,
        )
        solver.add_gain_material(
            fg,
            center=(0.0, 0.0, 0.5 * L_cav),
            size=(2.0 * lx, 2.0 * ly, L_cav),
        )
        if pref_scale != 1.0:
            for ch in solver._gain_groups[0]["noise_channels"]:
                ch["pref"] = ch["pref"] * pref_scale
        z_out = L_cav + 2.0 * d_H + d_L + 0.2e-6
        k_out = int(np.argmin(np.abs(solver.z - z_out)))
        idx_out = k_out * solver.Nx * solver.Ny
        Nt = int(total_time / solver.dt)
        ex_out = np.empty(Nt)
        for n in range(Nt):
            solver.one_step()
            ex_out[n] = solver.E.array[idx_out]
        I_out = float(c_light * eps_0 * np.mean(ex_out[int(0.7 * Nt) :] ** 2))
        return I_out, fg.saturation_intensity(W_p=wp)

    ratios = np.array([0.5, 1.0, 1.5, 2.5])
    I_meas = np.empty_like(ratios)
    I_sats = np.empty_like(ratios)
    for i, r in enumerate(ratios):
        I_meas[i], I_sats[i] = run_fp_noise_point(r)

    # 1a. Compare threshold-extracted beta_fit = s(1)^2 against a priori beta_theory
    I_scale = 0.5 * np.log(1.0 / R_dbr) * I_sats
    s_meas = I_meas / I_scale
    beta_fit = float(s_meas[ratios == 1.0][0] ** 2)
    assert 0.008 < beta_fit < 0.020
    assert np.isclose(beta_fit, beta_theory, rtol=0.30, atol=0.0)

    # 1b. Claim beta_sp ONLY from the sub-threshold and threshold points (r <= 1.0),
    #     where spontaneous emission dominates. The full-range L2 across r in [0.5, 2.5]
    #     is dominated by the above-threshold stimulated slope (r >= 1.5) and is
    #     insensitive to beta_sp (~3% even for beta_sp = 0).
    I_theory_apriori = np.array(
        [
            fabry_perot_laser_output(
                r, isat, R_dbr, R_dbr, spatial_hole_burning=False, beta_sp=beta_theory
            )
            for r, isat in zip(ratios, I_sats)
        ]
    )
    I_theory_sharp = np.array(
        [
            fabry_perot_laser_output(
                r, isat, R_dbr, R_dbr, spatial_hole_burning=False, beta_sp=0.0
            )
            for r, isat in zip(ratios, I_sats)
        ]
    )
    sub_th_mask = ratios <= 1.0
    idx_r1 = int(np.where(ratios == 1.0)[0][0])
    # At threshold r = 1.0, sharp-threshold theory (beta_sp = 0) predicts I_out = 0 (100% error),
    # whereas zero-free-parameter beta_theory predicts I_out(1.0) within 15%:
    assert I_theory_sharp[idx_r1] == 0.0
    assert np.isclose(I_meas[idx_r1], I_theory_apriori[idx_r1], rtol=0.15, atol=0.0)
    l2_err_sub_th_apriori = np.linalg.norm(
        I_meas[sub_th_mask] - I_theory_apriori[sub_th_mask]
    ) / np.linalg.norm(I_theory_apriori[sub_th_mask])
    assert l2_err_sub_th_apriori < 0.16, (
        f"Expected sub-threshold/threshold (r <= 1) L2 error < 16%, got {l2_err_sub_th_apriori:.3%}"
    )

    # 1c. Above threshold (r > 1.0), the L-I curve tests the stimulated slope s(r) ~ r - 1:
    above_th_mask = ratios > 1.0
    l2_err_above_th = np.linalg.norm(
        I_meas[above_th_mask] - I_theory_apriori[above_th_mask]
    ) / np.linalg.norm(I_theory_apriori[above_th_mask])
    assert l2_err_above_th < 0.05, (
        f"Expected above-threshold stimulated slope L2 error < 5%, got {l2_err_above_th:.3%}"
    )

    # 1d. Code-path mutation check: corrupting the FDT Langevin prefactor in the solver
    #     group by 3x (9x noise power) breaks agreement with beta_theory at threshold
    I_mut_r1, isat_mut_r1 = run_fp_noise_point(1.0, pref_scale=3.0)
    beta_mut = float((I_mut_r1 / (0.5 * np.log(1.0 / R_dbr) * isat_mut_r1)) ** 2)
    assert not np.isclose(beta_mut, beta_theory, rtol=0.50, atol=0.0)

    # 2. Multi-mode cavity mode competition (L_cav = 4.8 um = 4 * lambda0)
    L_mm = 4.8e-6
    gamma_mm = 2.8e14
    tau_21_mm = 150.0e-15
    g_th_mm = fabry_perot_threshold_gain(L_mm, R_dbr, R_dbr)
    fg_mm_ref = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_mm,
        tau_21=tau_21_mm,
        tau_10=5.0e-15,
        tau_32=5.0e-15,
        N_total=5.0e23,
    )
    wp_mm = fg_mm_ref.threshold_pump_rate(2.5 * g_th_mm)
    solver_mm = _make_1d_solver(-1.2e-6, L_mm + 1.2e-6, dz)
    _add_dbr_mirrors(solver_mm, L_mm, d_H, d_L, n_H, n_pairs)
    fg_mm = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_mm,
        tau_21=tau_21_mm,
        tau_10=5.0e-15,
        tau_32=5.0e-15,
        N_total=5.0e23,
        pump_rate=wp_mm,
        initial_populations="steady_state",
        spatial_averaging=True,
        spontaneous_emission=True,
        noise_amplitude=0.05,
        seed=7,
    )
    solver_mm.add_gain_material(
        fg_mm, center=(0.0, 0.0, 0.5 * L_mm), size=(4.0e-6, 4.0e-6, L_mm)
    )
    z_out_mm = L_mm + 2.0 * d_H + d_L + 0.2e-6
    k_out_mm = int(np.argmin(np.abs(solver_mm.z - z_out_mm)))
    idx_out_mm = k_out_mm * solver_mm.Nx * solver_mm.Ny
    Nt_mm = int(2.4e-12 / solver_mm.dt)
    ex_mm = np.empty(Nt_mm)
    for n in range(Nt_mm):
        solver_mm.one_step()
        ex_mm[n] = solver_mm.E.array[idx_out_mm]

    dt_mm = solver_mm.dt
    nfft = 32768
    freqs = np.fft.rfftfreq(nfft, d=dt_mm)
    seg_early = ex_mm[int(0.05e-12 / dt_mm) : int(0.45e-12 / dt_mm)]
    seg_late = ex_mm[int(1.5e-12 / dt_mm) :]
    spec_early = np.abs(np.fft.rfft(seg_early * np.hanning(len(seg_early)), n=nfft)) ** 2
    spec_late = np.abs(np.fft.rfft(seg_late * np.hanning(len(seg_late)), n=nfft)) ** 2
    spec_early /= np.max(spec_early)
    spec_late /= np.max(spec_late)

    m_center = (freqs >= 240.0e12) & (freqs <= 260.0e12)
    m_side = ((freqs >= 212.0e12) & (freqs <= 232.0e12)) | (
        (freqs >= 268.0e12) & (freqs <= 288.0e12)
    )
    side_early = float(np.max(spec_early[m_side]))
    side_late = float(np.max(spec_late[m_side]))
    center_late = float(np.max(spec_late[m_center]))

    # Side modes are excited during build-up but suppressed by > 25 dB (< 3e-3) in steady state
    assert center_late == pytest.approx(1.0, rel=1e-6)
    assert side_early > 50.0 * side_late
    assert side_late < 1.0e-3


def test_lifetime_acceleration_invariants_and_scalings():
    """
    Requirement 2:
    Verify lifetime acceleration (`time_scale = s`):
    - Consistency checks: threshold pump rate, steady-state populations, saturation
      intensity, analytical turn-on delay ``t_th(s) = t_th(1) / s``, and analytical
      ``f_RO(s) = sqrt(s) * f_RO(1)``.
    - Numerical validation:
      1. Crank-Nicolson `step_rate_equations` turn-on delay matches exact matrix-
         exponential `turn_on_delay` with `atol=0.0` (`rtol=1e-4`).
      2. Independent cold-start 3D FIT convergence from rest + seed pulse across
         ``s = 1, 5, 20`` over ``>= 5 * tau_21`` reaches the same steady-state
         output intensity, clamped inversion, and lasing frequency.
      3. 3D FIT relaxation-oscillation quarter-period ``t_1/4(1)`` matches the
         dispersive-cavity analytical prediction (accounting for active-medium
         Lorentzian slow-light group delay), scales as ``1 / sqrt(s)`` in the
         Class-B regime ``s in {1, 10, 100}`` (``(s / tau_sat) / kappa_cav <= 0.148 < 1``),
         and exhibits the expected Class-B breakdown at ``s = 1000`` where
         ``(s / tau_sat) / kappa_cav = 1.48 > 1`` (`s > s_max_cavity = 677`).
    - Mutation checks on `_rate_cn_matrices` and solver rate matrices.
    """
    from wakis.sources import CurrentSource, GaussianSource

    lambda0 = 1.2e-6
    f0 = c_light / lambda0
    omega0 = 2.0 * np.pi * f0
    gamma_a = 1.0e14
    dz = 30.0e-9
    n_H, n_L = 2.0, 1.0
    d_H = lambda0 / (4.0 * n_H)
    d_L = lambda0 / (4.0 * n_L)
    n_pairs = 1
    L_cav = 1.2e-6

    R_dbr = dbr_reflectance(n_H, n_L, n_pairs, n_in=1.0, n_out=1.0, extra_high=True)
    g_th = fabry_perot_threshold_gain(L_cav, R_dbr, R_dbr)

    # Measure cold-cavity decay rate kappa_cav and photon lifetime tau_p_cold via
    # a cold-cavity ringdown (gain off, ResonanceMonitor / harminv) on the same
    # passive 3-layer (H L H) DBR cavity:
    from wakis.sources import CurrentSource, GaussianSource

    sol_ring = _make_1d_solver(-1.2e-6, 2.4e-6, dz)
    _add_dbr_mirrors(sol_ring, L_cav, d_H, d_L, n_H, n_pairs)
    src_ring = GaussianSource(frequency=f0, fwidth=0.05 * f0)
    sheet_ring = CurrentSource(
        src_ring,
        component="x",
        center=(0.0, 0.0, 0.25 * L_cav),
        size=(2.0e-6, 2.0e-6, 0.0),
        amplitude=1.0,
    )
    mon_ring = ResonanceMonitor(
        component="Ex",
        center=(0.0, 0.0, 0.75 * L_cav),
        t_start=src_ring.end_time + 30.0e-15,
    )
    nt_ring = int((src_ring.end_time + 300.0e-15) / sol_ring.dt)
    sol_ring.emsolve(nt_ring, source=sheet_ring, monitors=[mon_ring])
    modes_ring = mon_ring.harminv(0.85 * f0, 1.15 * f0)
    assert len(modes_ring.frequency) >= 1
    idx_cold = int(np.argmin(np.abs(modes_ring.frequency - f0)))
    kappa_cav = float(2.0 * modes_ring.decay_rate[idx_cold])
    tau_p_cold = 1.0 / kappa_cav

    tau_21 = 15.0e-12
    tau_10 = 20.0e-15
    tau_32 = 10.0e-15
    N_total = 6.0e25
    r_pump = 3.0

    ref_mat = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=N_total,
    )
    wp_th_ref = ref_mat.threshold_pump_rate(g_th)
    wp_op = ref_mat.threshold_pump_rate(r_pump * g_th)
    ref_mat.W_p = wp_op
    n_ss_ref = ref_mat.steady_state_populations()
    isat_ref = ref_mat.saturation_intensity()
    t_th_1 = ref_mat.turn_on_delay(g_th)
    f_ro_1 = ref_mat.relaxation_oscillation_frequency(r_pump, tau_p_cold, damped=False)

    # 1. Analytical scaling consistency checks + Crank-Nicolson ODE validation across s = 1, 10, 100, 1000
    scales = (1.0, 10.0, 100.0, 1000.0)
    for s in scales:
        mat_s = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=tau_21,
            tau_10=tau_10,
            tau_32=tau_32,
            N_total=N_total,
            pump_rate=wp_op,
            time_scale=s,
        )
        # Analytical consistency checks (atol=0.0):
        assert np.isclose(mat_s.threshold_pump_rate(g_th), wp_th_ref, rtol=1e-12, atol=0.0)
        assert np.isclose(mat_s.saturation_intensity(), isat_ref, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            mat_s.steady_state_populations(), n_ss_ref, rtol=1e-12, atol=0.0
        )
        assert np.isclose(mat_s.turn_on_delay(g_th) * s, t_th_1, rtol=1e-12, atol=0.0)
        f_ro_s = mat_s.relaxation_oscillation_frequency(r_pump, tau_p_cold, damped=False)
        assert np.isclose(f_ro_s / f_ro_1, np.sqrt(s), rtol=1e-12, atol=0.0)

        # Numerical validation: Crank-Nicolson step_rate_equations turn-on delay
        # matches exact matrix-exponential turn_on_delay to < 1e-4 with atol=0.0
        dt_rate = 0.2e-15
        nt_rate = int(round(2.0 * (t_th_1 / s) / dt_rate))
        t_arr, n_hist = mat_s.step_rate_equations(nt_rate, dt_rate)
        dn_arr = n_hist[:, 2] - n_hist[:, 1]
        dn_th = g_th / mat_s.emission_cross_section
        idx_cross = int(np.argmax(dn_arr >= dn_th))
        t_cross = t_arr[idx_cross - 1] + (
            (dn_th - dn_arr[idx_cross - 1])
            / (dn_arr[idx_cross] - dn_arr[idx_cross - 1])
        ) * dt_rate
        assert np.isclose(t_cross * s, t_th_1, rtol=1e-4, atol=0.0)

    # Check adiabaticity limits helper: Class-B holds for s <= 100, breaks down at s = 1000 > s_max_cavity
    lim_10 = ref_mat.time_scaling_limits(tau_p=tau_p_cold, time_scale=10.0)
    lim_100 = ref_mat.time_scaling_limits(tau_p=tau_p_cold, time_scale=100.0)
    lim_1000 = ref_mat.time_scaling_limits(tau_p=tau_p_cold, time_scale=1000.0)
    assert lim_10["sat_rate_over_kappa_cav"] < 0.05
    assert lim_100["sat_rate_over_kappa_cav"] < 0.20
    assert lim_1000["sat_rate_over_kappa_cav"] > 1.0  # exceeds Class-B cavity limit!
    assert lim_1000["sat_rate_over_kappa_cav"] == pytest.approx(
        100.0 * lim_10["sat_rate_over_kappa_cav"], rel=1e-12
    )
    assert lim_10["s_max_cavity"] < 1000.0 < lim_10["s_max_dephasing"]

    # 2. Independent cold-start 3D FIT steady-state invariance across s = 1, 5, 20:
    #    Each run starts independently from rest (E = H = 0, initial_populations="steady_state")
    #    with the same weak seed pulse and evolves for 1.4 ps > 9 * tau_21_cold (no warm-up
    #    shared across s), proving that s = 1, 5, 20 converge to the same steady state.
    tau_21_cold = 150.0e-15
    ref_cold = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=tau_21_cold,
        tau_10=5.0e-15,
        tau_32=5.0e-15,
        N_total=8.0e23,
    )
    wp_cold = ref_cold.threshold_pump_rate(2.5 * g_th)
    isat_cold = ref_cold.saturation_intensity(W_p=wp_cold)
    I_out_cold_th = fabry_perot_laser_output(
        2.5, isat_cold, R_dbr, R_dbr, spatial_hole_burning=False
    )

    def run_cold_start_fit(s_val, total_time=1.4e-12, unscaled_pump_mutation=False):
        sol = _make_1d_solver(-1.2e-6, 2.4e-6, dz)
        _add_dbr_mirrors(sol, L_cav, d_H, d_L, n_H, n_pairs)
        m_c = FourLevelGain(
            omega_a=omega0,
            gamma_a=gamma_a,
            tau_21=tau_21_cold,
            tau_10=5.0e-15,
            tau_32=5.0e-15,
            N_total=8.0e23,
            pump_rate=wp_cold,
            initial_populations="steady_state",
            spatial_averaging=True,
            time_scale=s_val,
        )
        sol.add_gain_material(
            m_c, center=(0.0, 0.0, 0.5 * L_cav), size=(4.0e-6, 4.0e-6, L_cav)
        )
        if unscaled_pump_mutation:
            # Mutate the solver's Crank-Nicolson rate matrices by omitting s on W_p
            g_c = sol._gain_groups[0]
            g_c["A_rate"], g_c["B_rate"] = _rate_cn_matrices(
                m_c, float(sol.dt), W_p=wp_cold / s_val
            )
        src_c = GaussianSource(frequency=f0, fwidth=0.08 * f0)
        sheet_c = CurrentSource(
            src_c,
            component="x",
            center=(0.0, 0.0, 0.25 * L_cav),
            size=(2.0e-6, 2.0e-6, 0.0),
            amplitude=2.0,
        )
        z_out_c = L_cav + 2.0 * d_H + d_L + 0.2e-6
        k_out_c = int(np.argmin(np.abs(sol.z - z_out_c)))
        idx_out_c = k_out_c * sol.Nx * sol.Ny
        g_c = sol._gain_groups[0]
        cw_c = g_c["cell_weight"] / g_c["eff_n_cells"]
        nt_c = int(total_time / sol.dt)
        nt_pulse = int(0.15e-12 / sol.dt)
        ex_c = np.empty(nt_c)
        dn_c = np.empty(nt_c)
        for n in range(nt_c):
            if n < nt_pulse:
                sheet_c.update(sol, n * sol.dt)
            elif n == nt_pulse:
                sol.J.array[:] = 0.0
            sol.one_step()
            ex_c[n] = sol.E.array[idx_out_c]
            dn_c[n] = float((g_c["N"][2] - g_c["N"][1]) @ cw_c)
        n_tail = int(0.4e-12 / sol.dt)
        i_ss = float(c_light * eps_0 * np.mean(ex_c[-n_tail:] ** 2))
        dn_ss = float(np.mean(dn_c[-n_tail:]))
        return i_ss, dn_ss

    cold_results = [run_cold_start_fit(sv) for sv in (1.0, 5.0, 20.0)]
    i_cold_1, dn_cold_1 = cold_results[0]
    for i_cv, dn_cv in cold_results:
        assert np.isclose(i_cv, i_cold_1, rtol=0.02, atol=0.0)
        assert np.isclose(i_cv, I_out_cold_th, rtol=0.04, atol=0.0)
        assert np.isclose(dn_cv, dn_cold_1, rtol=0.02, atol=0.0)
    # Code-path mutation check on cold-start 3D FIT: omitting s on W_p in _rate_cn_matrices
    # collapses the output intensity by > 95%
    i_cold_mut, _ = run_cold_start_fit(20.0, unscaled_pump_mutation=True)
    assert not np.isclose(i_cold_mut, i_cold_1, rtol=0.50, atol=0.0)

    # 3. 3D FIT relaxation-oscillation dynamics across s = 1, 10, 100 (Class-B) and s = 1000 (breakdown):
    solver_warm = _make_1d_solver(-1.2e-6, 2.4e-6, dz)
    _add_dbr_mirrors(solver_warm, L_cav, d_H, d_L, n_H, n_pairs)
    mat_warm = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=tau_21,
        tau_10=tau_10,
        tau_32=tau_32,
        N_total=N_total,
        pump_rate=wp_op,
        initial_populations="steady_state",
        spatial_averaging=True,
        time_scale=200.0,
    )
    solver_warm.add_gain_material(
        mat_warm, center=(0.0, 0.0, 0.5 * L_cav), size=(4.0e-6, 4.0e-6, L_cav)
    )
    src = GaussianSource(frequency=f0, fwidth=0.08 * f0)
    sheet = CurrentSource(
        src,
        component="x",
        center=(0.0, 0.0, 0.25 * L_cav),
        size=(2.0e-6, 2.0e-6, 0.0),
        amplitude=2.0,
    )
    nt_warm = int(0.8e-12 / solver_warm.dt)
    for n in range(nt_warm):
        sheet.update(solver_warm, n * solver_warm.dt)
        solver_warm.one_step()
    solver_warm.J.array[:] = 0.0
    for _ in range(int(0.2e-12 / solver_warm.dt)):
        solver_warm.one_step()

    g_warm = solver_warm._gain_groups[0]
    E_ss = solver_warm.E.array.copy()
    H_ss = solver_warm.H.array.copy()
    psi_ss = {
        a: getattr(solver_warm, a).copy()
        for a in dir(solver_warm)
        if a.startswith("psi")
    }
    pa_ss = g_warm["lorentz_a"][3].copy()
    pa_old_ss = g_warm["lorentz_a"][4].copy()
    N_ss_fit = g_warm["N"].copy()
    dn_ss_val = float(np.mean(N_ss_fit[2] - N_ss_fit[1]))
    delta_n = 0.05 * dn_ss_val
    N_pert = N_ss_fit.copy()
    N_pert[2] += delta_n
    N_pert[0] -= delta_n

    I_out_theory = fabry_perot_laser_output(
        r_pump, isat_ref, R_dbr, R_dbr, spatial_hole_burning=False
    )
    z_out = L_cav + 2.0 * d_H + d_L + 0.2e-6
    k_out = int(np.argmin(np.abs(solver_warm.z - z_out)))
    idx_out = k_out * solver_warm.Nx * solver_warm.Ny
    cw = g_warm["cell_weight"] / g_warm["eff_n_cells"]
    dt = solver_warm.dt
    n_avg = int((20.0 / f0) / dt)

    def restore_steady(n_init_arr, s_val):
        solver_warm.E.array[:] = E_ss
        solver_warm.H.array[:] = H_ss
        for a, v in psi_ss.items():
            getattr(solver_warm, a)[:] = v
        g_warm["lorentz_a"][3][:] = pa_ss
        g_warm["lorentz_a"][4][:] = pa_old_ss
        g_warm["N"][:] = n_init_arr
        mat_warm.time_scale = s_val

    I_outs = []
    dn_ss_list = []
    f_peaks = []
    t_quarter_ro = []
    sim_windows = {1.0: 1.5e-12, 10.0: 0.7e-12, 100.0: 0.40e-12, 1000.0: 0.30e-12}

    for s in scales:
        # (a) Steady-state stationarity check across s = 1, 10, 100, 1000
        restore_steady(N_ss_fit, s)
        ex_ss = np.empty(2 * n_avg)
        dn_ss_hist = np.empty(2 * n_avg)
        for n in range(2 * n_avg):
            solver_warm.one_step()
            ex_ss[n] = solver_warm.E.array[idx_out]
            dn_ss_hist[n] = float((g_warm["N"][2] - g_warm["N"][1]) @ cw)
        I_outs.append(float(c_light * eps_0 * np.mean(ex_ss[-n_avg:] ** 2)))
        dn_ss_list.append(float(np.mean(dn_ss_hist[-n_avg:])))
        spec = np.abs(np.fft.rfft(ex_ss * np.hanning(2 * n_avg), n=32768))
        freqs = np.fft.rfftfreq(32768, d=dt)
        f_peaks.append(float(freqs[np.argmax(spec)]))

        # (b) Relaxation-oscillation scaling check from small perturbation N_pert
        restore_steady(N_pert, s)
        nt = int(sim_windows[s] / dt)
        dn_trace = np.empty(nt)
        for n in range(nt):
            solver_warm.one_step()
            dn_trace[n] = float((g_warm["N"][2] - g_warm["N"][1]) @ cw) - dn_ss_val

        win = int(round(2.0 / (f0 * dt)))
        dn_smooth = np.convolve(dn_trace, np.ones(win) / win, mode="valid")
        t_smooth = np.arange(len(dn_smooth)) * dt
        zc = np.where(np.diff(np.sign(dn_smooth)) != 0)[0]
        t_quarter_ro.append(float(t_smooth[zc[0]]))

    for iv, dnv, fp in zip(I_outs, dn_ss_list, f_peaks):
        assert np.isclose(iv, I_outs[0], rtol=0.025, atol=0.0)
        assert np.isclose(iv, I_out_theory, rtol=0.04, atol=0.0)
        assert np.isclose(dnv, dn_ss_list[0], rtol=0.02, atol=0.0)
        assert np.isclose(fp, f0, rtol=0.01, atol=0.0)

    # Compare absolute s = 1 first zero-crossing time t_zc,FIT(1) with theory
    # (at dz = 25 nm in examples/014_laser_noise_and_time_scaling.json:
    #  t_zc,FIT(1) = 726.38 fs vs t_1/4,disp(1) = 730.94 fs, 0.62% difference):
    # - Bare cold-cavity rate-equation prediction (ignoring active Lorentzian group delay):
    #   t_1/4,bare(1) = 1 / (4 * f_RO,bare(1)) = 640.07 fs (~11.9% faster than 3D FIT).
    # - Dispersive-cavity rate-equation prediction including active-region Lorentzian
    #   slow-light group delay tau_p,disp = tau_p,cold * (1 + Gamma_z * kappa_cav / gamma_a)
    #   (with Gamma_z = L_cav / L_eff = 2/3): t_1/4,disp(1) = 730.94 fs (at dz = 25 nm),
    #   matching 3D FIT to 0.62% (and < 3.5% at coarse dz = 30 nm here).
    t_q_bare_1 = 0.25 / ref_mat.relaxation_oscillation_frequency(
        r_pump, tau_p_cold, time_scale=1.0, damped=True
    )
    L_pen = lambda0 / (4.0 * (n_H - n_L))
    gamma_z = L_cav / (L_cav + 2.0 * L_pen)
    tau_p_disp = tau_p_cold * (1.0 + gamma_z * kappa_cav / gamma_a)
    t_q_disp = [
        0.25
        / ref_mat.relaxation_oscillation_frequency(
            r_pump, tau_p_disp, time_scale=s_v, damped=True
        )
        for s_v in (1.0, 10.0, 100.0)
    ]
    t_q_disp_undamped = [
        0.25
        / ref_mat.relaxation_oscillation_frequency(
            r_pump, tau_p_disp, time_scale=s_v, damped=False
        )
        for s_v in (1.0, 10.0, 100.0)
    ]
    assert np.isclose(t_quarter_ro[0], t_q_disp[0], rtol=0.035, atol=0.0)
    # Neglecting active-medium slow-light dispersion (bare cold-cavity tau_p_cold)
    # underpredicts t_zc(1) by ~12-14%:
    assert not np.isclose(t_quarter_ro[0], t_q_bare_1, rtol=0.05, atol=0.0)

    # In the Class-B regime s in {1, 10, 100} (s < s_max_cavity = 677):
    # - Undamped dispersive theory predicts t_1/4,disp,0(1) / t_1/4,disp,0(100) = 10.0.
    # - Damped 2-ODE dispersive theory predicts t_1/4,disp(1) / t_1/4,disp(100) = 8.86
    #   because RO damping Gamma_RO(s) ~ s reaches 0.465 * omega_RO,0 at s = 100,
    #   lengthening the damped quarter-period (at dz = 25 nm, 3D FIT vs damped dispersive
    #   theory is -0.62% at s = 1, +5.11% at s = 10, and -11.96% at s = 100; the
    #   first-crossing phase-advance correction over-corrects, so the near-10 FIT ratio
    #   is an unexplained observation, not a confirmation of sqrt(s)).
    assert np.isclose(t_q_disp_undamped[0] / t_q_disp_undamped[2], 10.0, rtol=1e-12)
    assert np.isclose(t_q_disp[0] / t_q_disp[2], 8.863, rtol=0.02)
    assert np.isclose(t_quarter_ro[1], t_q_disp[1], rtol=0.10, atol=0.0)
    assert np.isclose(t_quarter_ro[2], t_q_disp[2], rtol=0.15, atol=0.0)
    assert np.isclose(
        t_quarter_ro[0] / t_quarter_ro[1], np.sqrt(10.0), rtol=0.10, atol=0.0
    )
    assert np.isclose(
        t_quarter_ro[0] / t_quarter_ro[2], np.sqrt(100.0), rtol=0.16, atol=0.0
    )
    # At s = 1000 > s_max_cavity = 677 (sat_rate_over_kappa_cav = 1.48 > 1),
    # Class-B adiabaticity breaks down and the step ratio t_zc(100)/t_zc(1000) = 4.60
    # departs from sqrt(10) = 3.162 by > 35%:
    assert not np.isclose(
        t_quarter_ro[2] / t_quarter_ro[3], np.sqrt(10.0), rtol=0.30, atol=0.0
    )

    # Mutation check: scaling tau_21 -> tau_21 / s without scaling W_p changes the
    # steady-state inversion below threshold and breaks lasing invariance
    mat_mut = FourLevelGain(
        omega_a=omega0,
        gamma_a=gamma_a,
        tau_21=tau_21 / 100.0,
        tau_10=tau_10 / 100.0,
        tau_32=tau_32 / 100.0,
        kappa_a=ref_mat.kappa_a,
        N_total=N_total,
        pump_rate=wp_op,  # unscaled pump_rate!
    )
    assert mat_mut.steady_state_inversion() < 0.05 * ref_mat.steady_state_inversion()
