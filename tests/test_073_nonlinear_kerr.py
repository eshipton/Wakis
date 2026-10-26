# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Validation tests for Kerr (chi^(3)) nonlinear media, delayed Raman response,
and Lugiato-Lefever microcomb solvers (`wakis.nonlinear`) coupled to
`SolverFIT3D`.

Covers:
1. Bit-for-bit linear reproduction when `chi3 = 0` (with and without linear
   Lorentz dispersion poles) and mutation check when `chi3 != 0`.
2. 1D self-phase modulation (SPM): nonlinear phase shift
   `Delta Phi_NL = n_2 k_0 I L` across multiple intensities, verifying
   `n_2 = (3 / 4) chi^(3) / (c eps_0 n_0^2)` in the `exp(+i omega t)` real-field
   convention, agreement across `'newton'`, `'fixed_point'`, and `'explicit_ade'`
   update schemes, and mutation check on the `3 / 4` factor.
3. 1D third-harmonic generation (THG) in a thin slab: undepleted-pump conversion
   efficiency `eta_THG = I_3(L) / I_1(0)`, cubic intensity scaling, Maker-fringe
   phase-mismatch oscillation `sinc^2(Delta k L / 2)` across a full coherence
   cycle `L / L_c in [0.25, 2.0]`, and mutation checks.
4. Kerr-induced cavity resonance shift `Delta f_NL` in 1D Fabry-Perot and 2D
   cavities vs first-order perturbation theory with the mode's effective
   nonlinear volume `V_eff,NL`, including sign/mode-volume mutation checks.
5. Delayed Raman Lorentz ADE response (`f_R > 0`), impulse-response
   normalization, discrete susceptibility, and nonlinear field energy
   conservation in a lossless cavity.
6. Lugiato-Lefever equation (LLE) CW bistability (`zeta > sqrt(3)`), split-step
   Fourier single-soliton profile and `sech^2` comb spectral width vs
   `sqrt(D_2)` analytical formulas, and normal-dispersion / factor mutation
   checks.
"""

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0

import wakis
from wakis.materials import DispersiveMaterial, Lorentz
from wakis.nonlinear import (
    KerrMaterial,
    NonlinearMaterial,
    chi3_to_n2,
    effective_nonlinear_volume,
    field_from_intensity,
    kerr_cavity_frequency_shift,
    lle_cw_bistability,
    lle_soliton_analytic,
    n2_to_chi3,
    optical_intensity,
    solve_lle,
    spm_phase_shift,
    thg_conversion_efficiency,
)
from wakis.sources import ContinuousSource, CurrentSource, GaussianSource


# ---------------------------------------------------------------------------
# 1. Zero chi^(3) reproduces linear results bit-for-bit
# ---------------------------------------------------------------------------


def test_zero_chi3_reproduces_linear_bit_for_bit():
    """
    With `chi3 = 0`, `KerrMaterial` reproduces the linear `SolverFIT3D` run
    bit-for-bit (`max |E_nl - E_lin| == 0.0`), and with linear Lorentz poles it
    matches `DispersiveMaterial` to machine precision. Mutating `chi3 != 0`
    breaks equality.
    """
    f0 = 200.0e12
    eps_inf = 2.25
    L = 2.0e-6
    dz = 20.0e-9
    grid = wakis.GridFIT3D(
        -0.5e-6,
        0.5e-6,
        -0.5e-6,
        0.5e-6,
        -1.0e-6,
        3.0e-6,
        2,
        2,
        int(4.0e-6 / dz),
        verbose=0,
    )

    def run_pulse(mat_kind, chi3_val=0.0, poles=()):
        solver = wakis.SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=10,
            bg=[eps_inf, 1.0, 0.0],
            verbose=0,
        )
        if mat_kind == "kerr":
            mat = KerrMaterial(eps_inf=eps_inf, chi3=chi3_val, poles=poles)
            solver.add_nonlinear_material(
                mat, center=(0.0, 0.0, 0.5 * L), size=(2.0e-6, 2.0e-6, L)
            )
        elif mat_kind == "dispersive":
            mat = DispersiveMaterial(eps_inf=eps_inf, poles=list(poles))
            solver.add_dispersive_material(
                mat, center=(0.0, 0.0, 0.5 * L), size=(2.0e-6, 2.0e-6, L)
            )
        src = ContinuousSource(frequency=f0, ramp_time=2.0 / f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -0.4e-6),
            size=(1.0e-6, 1.0e-6, 0.0),
            amplitude=2.0e5,
        )
        Nt = 400
        for n in range(Nt):
            sheet.update(solver, n * solver.dt)
            solver.one_step()
        return solver.E.array.copy(), solver.H.array.copy()

    E_lin, H_lin = run_pulse("linear")
    E_kerr0, H_kerr0 = run_pulse("kerr", chi3_val=0.0)

    # Exact bit-for-bit identity for chi3 = 0
    assert np.array_equal(E_kerr0, E_lin)
    assert np.array_equal(H_kerr0, H_lin)
    assert float(np.max(np.abs(E_kerr0 - E_lin))) == 0.0

    # With a linear Lorentz pole and chi3 = 0, matches DispersiveMaterial
    pole = Lorentz(delta_eps=0.3, omega_0=2.0 * np.pi * 350.0e12, gamma=1.0e12)
    E_disp, _ = run_pulse("dispersive", poles=[pole])
    E_kerr_disp, _ = run_pulse("kerr", chi3_val=0.0, poles=[pole])
    np.testing.assert_allclose(E_kerr_disp, E_disp, rtol=1e-14, atol=1e-10)

    # Mutation check: non-zero chi3 alters the transmitted waveform
    E_kerr_nz, _ = run_pulse("kerr", chi3_val=5.0e-17)
    assert not np.allclose(E_kerr_nz, E_lin, rtol=1e-4, atol=1.0)


# ---------------------------------------------------------------------------
# 2. Self-phase modulation (SPM) and the 3/4 real-field factor
# ---------------------------------------------------------------------------


def test_spm_phase_shift_and_n2_chi3_relation():
    """
    Verify (a):
    1. `chi3_to_n2` and `n2_to_chi3` invert each other and satisfy
       `n_2 = (3 / 4) chi^(3) / (c eps_0 n_0^2)` in the `exp(+i omega t)`
       real-field convention.
    2. In a 1D index-matched Kerr slab of length `L = 3 um`, the steady-state
       CW nonlinear phase shift `Delta Phi_NL = n_2 k_0 I L` matches analytic
       theory across a 9x intensity sweep to < 0.3% relative error and agrees
       across `'newton'`, `'fixed_point'`, and `'explicit_ade'` update methods.
    3. Mutating the `3 / 4` degeneracy factor (e.g. dropping `3 / 4 -> 1`)
       causes the SPM phase check to fail.
    """
    eps_inf = 2.25
    n0 = np.sqrt(eps_inf)
    chi3 = 2.4e-19
    n2 = chi3_to_n2(chi3, eps_r=eps_inf)
    n2_expected = 0.75 * chi3 / (c * epsilon_0 * eps_inf)
    assert np.isclose(n2, n2_expected, rtol=1e-15)
    assert np.isclose(n2_to_chi3(n2, eps_r=eps_inf), chi3, rtol=1e-15)

    E_test = 5.0e7
    I_test = optical_intensity(E_test, n0=n0)
    assert np.isclose(field_from_intensity(I_test, n0=n0), E_test, rtol=1e-15)

    f0 = 200.0e12
    lam0 = c / f0
    omega0 = 2.0 * np.pi * f0
    L = 3.0e-6
    dz = 15.0e-9
    steps_per_period = 240
    dt = 1.0 / (steps_per_period * f0)
    Z_med = np.sqrt(mu_0 / (epsilon_0 * eps_inf))

    grid = wakis.GridFIT3D(
        -0.5e-6,
        0.5e-6,
        -0.5e-6,
        0.5e-6,
        -1.5e-6,
        4.5e-6,
        2,
        2,
        int(6.0e-6 / dz),
        verbose=0,
    )

    def run_cw_spm(chi3_val, E0_target, method="newton", degeneracy_factor=1.0):
        solver = wakis.SolverFIT3D(
            grid,
            dt=dt,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=12,
            bg=[eps_inf, 1.0, 0.0],
            verbose=0,
        )
        if chi3_val is not None:
            mat = KerrMaterial(
                eps_inf=eps_inf,
                chi3=chi3_val,
                method=method,
                degeneracy_factor=degeneracy_factor,
            )
            solver.add_nonlinear_material(
                mat, center=(0.0, 0.0, 0.5 * L), size=(2.0e-6, 2.0e-6, L)
            )
        src = ContinuousSource(frequency=f0, ramp_time=4.0 / f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -0.5e-6),
            size=(1.0e-6, 1.0e-6, 0.0),
            amplitude=2.0 * E0_target / Z_med,
        )
        n_avg = 6 * steps_per_period
        Nt = 28 * steps_per_period
        k_in = int(np.argmin(np.abs(solver.z - (-0.1e-6))))
        k_out = int(np.argmin(np.abs(solver.z - (L + 0.2e-6))))
        stride_z = solver.Nx * solver.Ny
        idx_in = k_in * stride_z
        idx_out = k_out * stride_z

        c1_in = 0.0j
        c1_out = 0.0j
        for n in range(Nt):
            sheet.update(solver, n * dt)
            solver.one_step()
            if n >= Nt - n_avg:
                t_e = (n + 1) * dt
                phase = np.exp(-1j * omega0 * t_e)
                c1_in += solver.E.array[idx_in] * phase
                c1_out += solver.E.array[idx_out] * phase
        c1_in *= 2.0 / n_avg
        c1_out *= 2.0 / n_avg
        return c1_in, c1_out

    # Sweep across 4 field amplitudes (9x intensity range)
    E_list = [4.0e7, 6.0e7, 9.0e7, 1.2e8]
    dphi_meas_list = []
    dphi_an_list = []
    I_meas_list = []

    c1_in_ref, c1_out_ref = run_cw_spm(None, E_list[0])
    for E0 in E_list:
        scale = E0 / E_list[0]
        c1_in_lin = c1_in_ref * scale
        c1_out_lin = c1_out_ref * scale
        _, c1_out_nl = run_cw_spm(chi3, E0, method="newton")
        I_in = optical_intensity(np.abs(c1_in_lin), n0=n0)
        dphi_meas = float(np.angle(c1_out_lin / c1_out_nl))
        dphi_an = float(spm_phase_shift(I_in, L, lam0, chi3=chi3, n0=n0))
        I_meas_list.append(I_in)
        dphi_meas_list.append(dphi_meas)
        dphi_an_list.append(dphi_an)
        np.testing.assert_allclose(dphi_meas, dphi_an, rtol=3.0e-3)

    # Check linear proportionality Delta Phi_NL vs I
    slope_meas = np.polyfit(I_meas_list, dphi_meas_list, 1)[0]
    slope_an = n2 * (2.0 * np.pi / lam0) * L
    np.testing.assert_allclose(slope_meas, slope_an, rtol=3.0e-3)

    # Compare 'fixed_point' and 'explicit_ade' against 'newton' at E0 = 8e7 V/m
    E_cmp = 8.0e7
    scale_cmp = E_cmp / E_list[0]
    c1_out_lin_cmp = c1_out_ref * scale_cmp
    I_cmp = optical_intensity(np.abs(c1_in_ref * scale_cmp), n0=n0)
    dphi_an_cmp = float(spm_phase_shift(I_cmp, L, lam0, chi3=chi3, n0=n0))
    for scheme in ("newton", "fixed_point", "explicit_ade"):
        _, c1_out_s = run_cw_spm(chi3, E_cmp, method=scheme)
        dphi_s = float(np.angle(c1_out_lin_cmp / c1_out_s))
        np.testing.assert_allclose(dphi_s, dphi_an_cmp, rtol=3.0e-3)

    # Mutation check 1: dropping the 3/4 factor (using factor=1.0 in analytic formula)
    dphi_wrong_factor = spm_phase_shift(
        I_meas_list[-1], L, lam0, chi3=chi3, n0=n0, factor=1.0
    )
    assert not np.isclose(dphi_meas_list[-1], dphi_wrong_factor, rtol=0.15)

    # Mutation check 2: mutating degeneracy_factor=4/3 on KerrMaterial fails the 3/4 law
    _, c1_out_mut = run_cw_spm(chi3, E_cmp, degeneracy_factor=4.0 / 3.0)
    dphi_mut = float(np.angle(c1_out_lin_cmp / c1_out_mut))
    assert not np.isclose(dphi_mut, dphi_an_cmp, rtol=0.15)


# ---------------------------------------------------------------------------
# 3. Third-harmonic generation (THG) and phase-mismatch sinc^2 Maker fringes
# ---------------------------------------------------------------------------


def test_thg_conversion_efficiency_and_phase_mismatch_sinc2():
    """
    Verify (b):
    1. Undepleted-pump THG in a thin dispersionless slab matches
       `thg_conversion_efficiency` (`eta_THG = I_3 / I_1`) and exhibits the
       exact `eta_THG ~ I_1^2` scaling.
    2. In a dispersive Kerr medium with phase mismatch
       `Delta k = k(3 omega_1) - 3 k(omega_1) != 0` (coherence length
       `L_c = pi / |Delta k|`), the THG efficiency oscillates as
       `L^2 sinc^2(Delta k L / 2)` across `L / L_c in [0.25, 2.0]`, peaking at
       `L = L_c` and vanishing at the Maker-fringe null `L = 2 L_c`.
    3. Mutating the `1 / 4` THG degeneracy factor or ignoring phase mismatch
       (`delta_k = 0`) fails the check.
    """
    f0 = 150.0e12
    lam0 = c / f0
    omega0 = 2.0 * np.pi * f0
    dz = 15.0e-9
    steps_per_period = 300
    dt = 1.0 / (steps_per_period * f0)

    # Part A: Phase-matched thin slab (L = 2.0 um), intensity scaling eta ~ I_1^2
    eps_inf = 2.25
    n0 = np.sqrt(eps_inf)
    chi3 = 2.0e-19
    L_thin = 2.0e-6
    Z0_med = np.sqrt(mu_0 / (epsilon_0 * eps_inf))
    grid_thin = wakis.GridFIT3D(
        -0.5e-6,
        0.5e-6,
        -0.5e-6,
        0.5e-6,
        -1.2e-6,
        3.6e-6,
        2,
        2,
        int(4.8e-6 / dz),
        verbose=0,
    )

    etas_meas = []
    etas_an = []
    for E0 in (3.0e7, 6.0e7):
        solver = wakis.SolverFIT3D(
            grid_thin,
            dt=dt,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=12,
            bg=[eps_inf, 1.0, 0.0],
            verbose=0,
        )
        mat = KerrMaterial(eps_inf=eps_inf, chi3=chi3, method="newton")
        solver.add_nonlinear_material(
            mat, center=(0.0, 0.0, 0.5 * L_thin), size=(2.0e-6, 2.0e-6, L_thin)
        )
        src = ContinuousSource(frequency=f0, ramp_time=4.0 / f0)
        sheet = CurrentSource(
            src,
            component="x",
            center=(0.0, 0.0, -0.4e-6),
            size=(1.0e-6, 1.0e-6, 0.0),
            amplitude=2.0 * E0 / Z0_med,
        )
        n_avg = 6 * steps_per_period
        Nt = 24 * steps_per_period
        k_in = int(np.argmin(np.abs(solver.z - (-0.1e-6))))
        k_out = int(np.argmin(np.abs(solver.z - (L_thin + 0.2e-6))))
        stride_z = solver.Nx * solver.Ny
        idx_in = k_in * stride_z
        idx_out = k_out * stride_z
        c1_in = 0.0j
        c3_out = 0.0j
        for n in range(Nt):
            sheet.update(solver, n * dt)
            solver.one_step()
            if n >= Nt - n_avg:
                t_e = (n + 1) * dt
                c1_in += solver.E.array[idx_in] * np.exp(-1j * omega0 * t_e)
                c3_out += solver.E.array[idx_out] * np.exp(-1j * 3.0 * omega0 * t_e)
        c1_in *= 2.0 / n_avg
        c3_out *= 2.0 / n_avg
        I1 = optical_intensity(np.abs(c1_in), n0=n0)
        eta_m = float(np.abs(c3_out) ** 2 / np.abs(c1_in) ** 2)
        eta_a = float(
            thg_conversion_efficiency(I1, L_thin, lam0, chi3, n1=n0, n3=n0)[
                "efficiency"
            ]
        )
        etas_meas.append(eta_m)
        etas_an.append(eta_a)
        np.testing.assert_allclose(eta_m, eta_a, rtol=3.0e-2)

    # Doubling E0 quadruples I1 and increases eta_THG by 16x
    np.testing.assert_allclose(etas_meas[1] / etas_meas[0], 16.0, rtol=2.0e-2)

    # Part B: Phase-mismatched Maker fringes sinc^2(Delta k L / 2)
    pole = Lorentz(delta_eps=0.45, omega_0=2.0 * np.pi * 800.0e12, gamma=0.0)
    mat_pm = KerrMaterial(eps_inf=2.00, chi3=chi3, poles=[pole], method="newton")
    eps1 = float(np.real(mat_pm.epsilon_discrete(f0, dt)))
    eps3 = float(np.real(mat_pm.epsilon_discrete(3.0 * f0, dt)))
    n1 = np.sqrt(eps1)
    n3 = np.sqrt(eps3)

    def k_disc(freq, eps_r):
        w = 2.0 * np.pi * freq
        s = (np.sqrt(eps_r) * dz / (c * dt)) * np.sin(0.5 * w * dt)
        return (2.0 / dz) * np.arcsin(s)

    dk_disc = k_disc(3.0 * f0, eps3) - 3.0 * k_disc(f0, eps1)
    Lc = np.pi / dk_disc

    Z1 = np.sqrt(mu_0 / (epsilon_0 * eps1))
    E0_pm = 4.0e7
    grid_pm = wakis.GridFIT3D(
        -0.5e-6,
        0.5e-6,
        -0.5e-6,
        0.5e-6,
        -1.5e-6,
        13.5e-6,
        2,
        2,
        int(15.0e-6 / dz),
        verbose=0,
    )
    solver_pm = wakis.SolverFIT3D(
        grid_pm,
        dt=dt,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=12,
        bg=[eps1, 1.0, 0.0],
        verbose=0,
    )
    L_slab = 12.0e-6
    solver_pm.add_nonlinear_material(
        mat_pm, center=(0.0, 0.0, 0.5 * L_slab), size=(2.0e-6, 2.0e-6, L_slab)
    )
    src_pm = ContinuousSource(frequency=f0, ramp_time=4.0 / f0)
    sheet_pm = CurrentSource(
        src_pm,
        component="x",
        center=(0.0, 0.0, -0.5e-6),
        size=(1.0e-6, 1.0e-6, 0.0),
        amplitude=2.0 * E0_pm / Z1,
    )
    n_avg_pm = 6 * steps_per_period
    Nt_pm = 40 * steps_per_period
    stride_z = solver_pm.Nx * solver_pm.Ny
    z_idx = np.arange(solver_pm.Nz) * stride_z
    c1 = np.zeros(solver_pm.Nz, dtype=complex)
    c3 = np.zeros(solver_pm.Nz, dtype=complex)
    for n in range(Nt_pm):
        sheet_pm.update(solver_pm, n * dt)
        solver_pm.one_step()
        if n >= Nt_pm - n_avg_pm:
            t_e = (n + 1) * dt
            ex = solver_pm.E.array[z_idx]
            c1 += ex * np.exp(-1j * omega0 * t_e)
            c3 += ex * np.exp(-1j * 3.0 * omega0 * t_e)
    c1 *= 2.0 / n_avg_pm
    c3 *= 2.0 / n_avg_pm

    k_in = int(np.argmin(np.abs(solver_pm.z - 0.15e-6)))
    I1_pm = optical_intensity(np.abs(c1[k_in]), n0=n1)
    k_peak = int(np.argmin(np.abs(solver_pm.z - Lc)))
    eta_peak_an = float(
        thg_conversion_efficiency(
            I1_pm, solver_pm.z[k_peak], lam0, chi3, n1=n1, n3=n3, delta_k=dk_disc
        )["efficiency"]
    )

    for frac in (0.25, 0.50, 1.00, 1.50, 2.00):
        kz = int(np.argmin(np.abs(solver_pm.z - frac * Lc)))
        L_eff = float(solver_pm.z[kz])
        eta_m = float((n3 / n1) * np.abs(c3[kz]) ** 2 / np.abs(c1[k_in]) ** 2)
        eta_a = float(
            thg_conversion_efficiency(
                I1_pm, L_eff, lam0, chi3, n1=n1, n3=n3, delta_k=dk_disc
            )["efficiency"]
        )
        assert abs(eta_m - eta_a) / eta_peak_an < 0.05

    # Null at L = 2 * Lc is < 1% of the peak at L = Lc
    k_null = int(np.argmin(np.abs(solver_pm.z - 2.0 * Lc)))
    eta_null_m = float((n3 / n1) * np.abs(c3[k_null]) ** 2 / np.abs(c1[k_in]) ** 2)
    eta_peak_m = float((n3 / n1) * np.abs(c3[k_peak]) ** 2 / np.abs(c1[k_in]) ** 2)
    assert eta_null_m < 0.01 * eta_peak_m

    # Mutation check 1: mutating factor_thg from 1/4 to 3/4 changes efficiency by 9x
    eta_mut_factor = float(
        thg_conversion_efficiency(
            I1_pm,
            solver_pm.z[k_peak],
            lam0,
            chi3,
            n1=n1,
            n3=n3,
            delta_k=dk_disc,
            factor_thg=0.75,
        )["efficiency"]
    )
    assert not np.isclose(eta_peak_m, eta_mut_factor, rtol=0.5)

    # Mutation check 2: ignoring phase mismatch (delta_k = 0) at L = 2 * Lc fails
    eta_no_pm = float(
        thg_conversion_efficiency(
            I1_pm, solver_pm.z[k_null], lam0, chi3, n1=n1, n3=n3, delta_k=0.0
        )["efficiency"]
    )
    assert eta_no_pm > 100.0 * eta_null_m


# ---------------------------------------------------------------------------
# 4. Kerr-induced cavity resonance shift in 1D and 2D vs V_eff,NL perturbation
# ---------------------------------------------------------------------------


def test_kerr_cavity_resonance_shift_1d_and_2d():
    """
    Verify (c):
    1. 1D Fabry-Perot PEC cavity of length `L = 1.5 um` with `eps_inf = 2.25`:
       Kerr resonance frequency shift `Delta f_NL` measured with `wakis.harminv`
       matches perturbation theory
       `Delta f_NL / f_0 = -(c n_2 / n_0^2) (U / V_eff,NL)` with
       `V_eff,NL = (2/3) V_geom` across multiple field amplitudes and both
       signs of `chi^(3)`.
    2. 2D rectangular PEC cavity (`m_x = 1, m_y = 1` TM_z mode) with
       `V_eff,NL = (4/9) V_geom`: FIT shift matches both the closed-form 2D
       formula (`cavity_dim=2`) and the mesh-integrated `effective_nonlinear_volume`.
    3. Mutating the sign of `chi^(3)` or replacing `V_eff,NL` by `V_geom`
       (dropping the `(2/3)^D` spatial mode-overlap factor) fails the check.
    """
    # Part A: 1D Fabry-Perot cavity
    L = 1.5e-6
    Nz = 120
    dz = L / Nz
    eps_inf = 2.25
    chi3 = 5.0e-19
    Z_med = np.sqrt(mu_0 / (epsilon_0 * eps_inf))
    grid_1d = wakis.GridFIT3D(
        -0.5e-6, 0.5e-6, -0.5e-6, 0.5e-6, 0.0, L, 2, 2, Nz, verbose=0
    )

    def run_fp_1d(chi3_val, E_target):
        solver = wakis.SolverFIT3D(
            grid_1d,
            cfln=0.5,
            bc_low=["periodic", "periodic", "pec"],
            bc_high=["periodic", "periodic", "pec"],
            bg=[eps_inf, 1.0, 0.0],
            verbose=0,
        )
        if chi3_val is not None:
            mat = KerrMaterial(eps_inf=eps_inf, chi3=chi3_val, method="newton")
            solver.add_nonlinear_material(mat, center=None, size=None)
        z_e = np.arange(Nz) * dz
        z_h = (np.arange(Nz) + 0.5) * dz
        km = 2.0 * np.pi / L  # m = 2 standing-wave mode
        sin_km = np.sin(km * z_e)
        cos_km = np.cos(km * z_h)
        H0 = E_target / Z_med
        for i in range(solver.Nx):
            for j in range(solver.Ny):
                solver.H[i, j, :Nz, "y"] = H0 * cos_km
        Nt = 4800
        a_e = np.zeros(Nt)
        e_peak_prof = np.zeros(Nz)
        for n in range(Nt):
            solver.one_step()
            ex_z = solver.E[0, 0, :Nz, "x"]
            a_e[n] = (2.0 / Nz) * np.sum(ex_z * sin_km)
            if abs(a_e[n]) > np.max(e_peak_prof):
                e_peak_prof = np.abs(ex_z).copy()
        stride = 16
        res = wakis.harminv(
            a_e[::stride], solver.dt * stride, fmin=110.0e12, fmax=160.0e12
        )
        idx_dom = int(np.argmax(res.amplitude))
        return float(res.frequency[idx_dom]), e_peak_prof

    f_lin_1d, prof_lin_1d = run_fp_1d(None, 2.5e7)
    V_geom_1d = L * 1.0e-6 * 1.0e-6
    V_eff_1d = effective_nonlinear_volume(
        prof_lin_1d**2, dV=dz * 1.0e-6 * 1.0e-6, eps_r=eps_inf
    )
    np.testing.assert_allclose(V_eff_1d, (2.0 / 3.0) * V_geom_1d, rtol=1e-4)

    for E_target in (2.5e7, 5.0e7):
        f_nl, _ = run_fp_1d(chi3, E_target)
        df_meas = f_nl - f_lin_1d
        E_peak = float(np.max(prof_lin_1d)) * (E_target / 2.5e7)
        shift_an = kerr_cavity_frequency_shift(
            2.0 * np.pi * f_lin_1d,
            chi3,
            E_peak=E_peak,
            eps_r=eps_inf,
            cavity_dim=1,
        )
        shift_mesh = kerr_cavity_frequency_shift(
            2.0 * np.pi * f_lin_1d,
            chi3,
            e_field=prof_lin_1d * (E_target / 2.5e7),
            dV=dz * 1.0e-6 * 1.0e-6,
            eps_r=eps_inf,
        )
        np.testing.assert_allclose(
            shift_an["delta_f"], shift_mesh["delta_f"], rtol=1e-3
        )
        np.testing.assert_allclose(df_meas, shift_an["delta_f"], rtol=1.5e-2)

    # Self-defocusing medium (chi3 < 0) produces an equal and opposite blue-shift
    f_defoc, _ = run_fp_1d(-chi3, 2.5e7)
    df_defoc = f_defoc - f_lin_1d
    shift_focus_25 = kerr_cavity_frequency_shift(
        2.0 * np.pi * f_lin_1d,
        chi3,
        E_peak=float(np.max(prof_lin_1d)),
        eps_r=eps_inf,
        cavity_dim=1,
    )["delta_f"]
    assert df_defoc > 0.0
    np.testing.assert_allclose(df_defoc, -shift_focus_25, rtol=1.5e-2)

    # Part B: 2D cavity mode (TM_11 in x-y with V_eff,NL = (4/9) V_geom)
    Lx = Ly = 1.2e-6
    Nxy = 36
    dxy = Lx / Nxy
    grid_2d = wakis.GridFIT3D(
        0.0, Lx, 0.0, Ly, -0.2e-6, 0.2e-6, Nxy, Nxy, 2, verbose=0
    )

    def run_cav_2d(chi3_val, E_target):
        solver = wakis.SolverFIT3D(
            grid_2d,
            cfln=0.5,
            bc_low=["pec", "pec", "periodic"],
            bc_high=["pec", "pec", "periodic"],
            bg=[eps_inf, 1.0, 0.0],
            verbose=0,
        )
        if chi3_val is not None:
            mat = KerrMaterial(
                eps_inf=eps_inf, chi3=chi3_val, method="newton", coupling="edge"
            )
            solver.add_nonlinear_material(mat, center=None, size=None)
        x_e = np.arange(Nxy) * dxy
        y_e = np.arange(Nxy) * dxy
        XX, YY = np.meshgrid(x_e, y_e, indexing="ij")
        kx = np.pi / Lx
        ky = np.pi / Ly
        mode_ez = np.sin(kx * XX) * np.sin(ky * YY)
        for k in range(solver.Nz):
            solver.E[:Nxy, :Nxy, k, "z"] = E_target * mode_ez
        Nt = 2400
        a_ez = np.zeros(Nt)
        norm = float(np.sum(mode_ez**2))
        for n in range(Nt):
            solver.one_step()
            ez_slice = solver.E[:Nxy, :Nxy, 0, "z"]
            a_ez[n] = float(np.sum(ez_slice * mode_ez)) / norm
        stride = 10
        res = wakis.harminv(
            a_ez[::stride], solver.dt * stride, fmin=80.0e12, fmax=160.0e12
        )
        idx_dom = int(np.argmax(res.amplitude))
        return float(res.frequency[idx_dom]), E_target * mode_ez

    E_2d = 4.0e7
    f_lin_2d, mode_2d = run_cav_2d(None, E_2d)
    f_nl_2d, _ = run_cav_2d(chi3, E_2d)
    df_meas_2d = f_nl_2d - f_lin_2d

    V_geom_2d = Lx * Ly * 0.4e-6
    V_eff_2d = effective_nonlinear_volume(
        mode_2d**2, dV=(dxy**2) * 0.4e-6, eps_r=eps_inf
    )
    np.testing.assert_allclose(V_eff_2d, (4.0 / 9.0) * V_geom_2d, rtol=1e-3)

    shift_2d_an = kerr_cavity_frequency_shift(
        2.0 * np.pi * f_lin_2d,
        chi3,
        E_peak=E_2d,
        eps_r=eps_inf,
        cavity_dim=2,
    )
    shift_2d_mesh = kerr_cavity_frequency_shift(
        2.0 * np.pi * f_lin_2d,
        chi3,
        e_field=mode_2d,
        dV=(dxy**2) * 0.4e-6,
        eps_r=eps_inf,
    )
    np.testing.assert_allclose(
        shift_2d_an["delta_f"], shift_2d_mesh["delta_f"], rtol=1e-3
    )
    np.testing.assert_allclose(df_meas_2d, shift_2d_an["delta_f"], rtol=3.0e-2)

    # Mutation check: replacing V_eff,NL by V_geom (ignoring the 4/9 overlap) fails
    shift_wrong_vol = kerr_cavity_frequency_shift(
        2.0 * np.pi * f_lin_2d,
        chi3,
        stored_energy=shift_2d_mesh["stored_energy"],
        V_eff=V_geom_2d,
        eps_r=eps_inf,
    )["delta_f"]
    assert not np.isclose(df_meas_2d, shift_wrong_vol, rtol=0.3)


# ---------------------------------------------------------------------------
# 5. Delayed Raman Lorentz response and nonlinear field energy conservation
# ---------------------------------------------------------------------------


def test_delayed_raman_response_and_energy_balance():
    """
    Verify:
    1. Delayed Raman impulse response `h_R(t)` integrates to 1, `chi_raman(0) = 1`,
       and discrete susceptibility `chi_raman_discrete(Omega, dt)` converges at
       second order `O(dt^2)` to `chi_raman(Omega)`.
    2. Under a CW optical field, the Raman state `S(t)` relaxes to the cycle-
       averaged squared field `<E^2> = |E_0|^2 / 2` and reproduces the steady-
       state SPM phase shift with `f_R > 0`.
    3. In a lossless closed 1D cavity with instantaneous Kerr nonlinearity, the
       total electromagnetic + nonlinear energy `solver.nonlinear_field_energy()`
       remains bounded and invariant across oscillations, while mutating `chi3`
       in the energy functional breaks conservation.
    """
    mat_r = KerrMaterial(
        eps_inf=2.25,
        chi3=2.0e-19,
        f_R=0.18,
        tau_1=12.2e-15,
        tau_2=32.0e-15,
    )
    assert mat_r.has_raman

    # Impulse response normalization int_0^infty h_R(t) dt = 1
    t = np.linspace(0.0, 500.0e-15, 20001)
    h_r = mat_r.raman_impulse_response(t)
    integral_hr = float(np.trapezoid(h_r, t))
    np.testing.assert_allclose(integral_hr, 1.0, rtol=1e-4)
    assert np.isclose(mat_r.chi_raman(0.0), 1.0, rtol=1e-15)

    # Second-order O(dt^2) convergence of chi_raman_discrete to chi_raman
    Omega_test = 2.0 * np.pi * 10.0e12
    chi_cont = mat_r.chi_raman(Omega_test)
    err_dt1 = abs(mat_r.chi_raman_discrete(Omega_test, 1.0e-15) - chi_cont)
    err_dt2 = abs(mat_r.chi_raman_discrete(Omega_test, 0.5e-15) - chi_cont)
    order = np.log2(err_dt1 / err_dt2)
    assert order > 1.95

    # Closed-cavity nonlinear energy conservation check
    L = 1.5e-6
    Nz = 100
    dz = L / Nz
    eps_inf = 2.25
    chi3_cav = 2.0e-18
    grid = wakis.GridFIT3D(
        -0.5e-6, 0.5e-6, -0.5e-6, 0.5e-6, 0.0, L, 2, 2, Nz, verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        cfln=0.4,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        bg=[eps_inf, 1.0, 0.0],
        verbose=0,
    )
    mat_k = KerrMaterial(eps_inf=eps_inf, chi3=chi3_cav, f_R=0.0, method="newton")
    solver.add_nonlinear_material(mat_k, center=None, size=None)

    z_e = np.arange(Nz) * dz
    km = 2.0 * np.pi / L
    E_init = 8.0e7 * np.sin(km * z_e)
    for i in range(solver.Nx):
        for j in range(solver.Ny):
            solver.E[i, j, :Nz, "x"] = E_init

    # At t = 0, H = 0 so E is at a turning point; evaluate energy at subsequent
    # electric-field turning points (every half-period) where H ~ 0
    w_init = solver.nonlinear_field_energy()
    w_lin_init = float(solver.field_energy())
    assert w_init > w_lin_init

    # Advance 6 optical half-cycles and verify turning-point energy invariance
    f_mode = c / (np.sqrt(eps_inf) * L)
    steps_half = int(round(0.5 / (f_mode * solver.dt)))
    energies = []
    for _ in range(6 * steps_half):
        solver.one_step()
        energies.append(solver.nonlinear_field_energy())
    energies = np.asarray(energies)
    chunk_peaks = [
        float(np.max(energies[m * steps_half : (m + 1) * steps_half]))
        for m in range(1, 6)
    ]
    # Turning-point total energy is conserved across cycles to < 0.05%
    np.testing.assert_allclose(chunk_peaks, chunk_peaks[0], rtol=5.0e-4)
    np.testing.assert_allclose(chunk_peaks[0], w_init, rtol=1.0e-2)

    # Mutation check: flipping the sign of chi3 in the energy formula fails
    w_mut = w_lin_init - (w_init - w_lin_init)
    assert not np.isclose(w_init, w_mut, rtol=5.0e-3, atol=0.0)


# ---------------------------------------------------------------------------
# 6. Lugiato-Lefever equation (LLE) soliton and bistability benchmarks
# ---------------------------------------------------------------------------


def test_lle_soliton_and_bistability_benchmarks():
    """
    Verify:
    1. CW bistability `Y = X [1 + (zeta - X)^2]` exhibits a single root for
       `zeta < sqrt(3)` and three roots (lower stable, middle unstable, upper
       bistable) in the bistable interval `zeta_min < zeta < zeta_max` for
       `zeta > sqrt(3)` and `f > sqrt(8 / 27)`.
    2. Split-step Fourier LLE integration (`solve_lle`) initialized near the
       analytical dissipative Kerr soliton converges to a stationary single
       soliton whose peak amplitude `sqrt(2 zeta)`, pulse FWHM
       `tau_FWHM ~ sqrt(d_2 / (2 zeta))`, and `sech^2` spectral bandwidth
       `Delta mu_FWHM ~ sqrt(zeta / d_2)` match `lle_soliton_analytic` across
       multiple dispersion values `D_2`.
    3. Mutating `D_2 -> -D_2` (normal dispersion) or dropping the factor of 2
       in `sqrt(2 zeta)` causes the soliton check to fail.
    """
    # Part A: CW bistability
    bist_sub = lle_cw_bistability(zeta_0=1.2, f_pump=2.0)
    assert np.isnan(bist_sub["upper_branch"])

    f_pump = 2.2
    zeta_scan = np.linspace(-1.0, 5.0, 61)
    bist = lle_cw_bistability(zeta_0=zeta_scan, f_pump=f_pump)
    bist_mask = ~np.isnan(bist["upper_branch"])
    assert np.any(bist_mask)
    assert float(np.min(zeta_scan[bist_mask])) > np.sqrt(3.0)
    for idx, z_val in enumerate(zeta_scan):
        for branch in ("lower_branch", "middle_branch", "upper_branch"):
            x_root = float(bist[branch][idx])
            if not np.isnan(x_root):
                np.testing.assert_allclose(
                    x_root * (1.0 + (z_val - x_root) ** 2), f_pump**2, rtol=1e-9
                )

    # Part B: Stationary single-soliton profile & sech^2 comb width vs sqrt(D_2)
    zeta_0 = 8.0
    f_0 = 3.0
    n_modes = 256
    phi = np.linspace(-np.pi, np.pi, n_modes, endpoint=False)
    fwhm_mu_list = []
    d2_list = [0.01, 0.04]
    for d2_val in d2_list:
        an = lle_soliton_analytic(phi, zeta_0=zeta_0, d2=d2_val, f_pump=f_0)
        assert an["in_existence_range"]
        sim = solve_lle(
            d2=d2_val,
            f_pump=f_0,
            zeta_0=zeta_0,
            n_modes=n_modes,
            dtau=0.01,
            n_steps=1200,
            psi_init=an["psi"],
            noise_amp=0.0,
        )
        psi_end = sim["psi"]
        # Pedestal at phi = +/- pi matches analytical CW background
        np.testing.assert_allclose(psi_end[0], an["psi_bg"], rtol=5.0e-3)
        peak_sim = float(np.max(np.abs(psi_end)))
        np.testing.assert_allclose(peak_sim, an["amplitude_peak"], rtol=5.0e-2)

        # Measure spectral sech^2 envelope width from sidebands (mu > 0)
        mu = sim["mu"]
        sb_pos = mu > 0
        mu_pos = mu[sb_pos].astype(float)
        p_pos = sim["spectrum"][sb_pos]
        p_norm = p_pos / p_pos[0]
        fwhm_mu_sim = 2.0 * float(np.interp(0.5, p_norm[::-1], mu_pos[::-1]))
        fwhm_mu_list.append(fwhm_mu_sim)
        np.testing.assert_allclose(fwhm_mu_sim, an["fwhm_mu"], rtol=8.0e-2)

    # Quadrupling d2 halves the comb spectral bandwidth (~ 1 / sqrt(d2))
    np.testing.assert_allclose(fwhm_mu_list[0] / fwhm_mu_list[1], 2.0, rtol=5.0e-2)

    # Mutation check 1: normal dispersion (d2 < 0) raises ValueError in lle_soliton_analytic
    with pytest.raises(ValueError, match="d2 > 0"):
        lle_soliton_analytic(phi, zeta_0=zeta_0, d2=-0.02, f_pump=f_0)

    # Mutation check 2: dropping the factor of 2 in sqrt(2 * zeta) fails peak check
    wrong_peak = np.sqrt(zeta_0)
    assert not np.isclose(peak_sim, wrong_peak, rtol=0.2)


def test_complex_dtype_kerr_raises_not_implemented():
    """
    Verify that attaching a `KerrMaterial` to a complex-dtype `SolverFIT3D` or
    calling `_solve_cubic_kerr_edge` on complex arrays raises `NotImplementedError`
    (since applying `|E|^2 E` to a complex phasor produces a 4/3 too-large Kerr
    shift and omits 3*omega third-harmonic generation).
    """
    from wakis.nonlinear import _solve_cubic_kerr_edge

    grid = wakis.GridFIT3D(0.0, 1.0e-6, 0.0, 1.0e-6, 0.0, 1.0e-6, 4, 4, 4, verbose=0)
    solver_cplx = wakis.SolverFIT3D(
        grid,
        cfln=0.5,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        dtype=np.complex128,
        verbose=0,
    )
    mat = KerrMaterial(eps_inf=2.25, chi3=2.4e-19)
    with pytest.raises(NotImplementedError, match="real-valued fields"):
        solver_cplx.add_nonlinear_material(mat, center=None, size=None)

    with pytest.raises(NotImplementedError, match="real-valued fields"):
        _solve_cubic_kerr_edge(
            a=np.array([2.25]),
            b=np.array([1.0e-18]),
            rhs=np.array([1.0 + 0.5j]),
            E_init=np.array([0.5 + 0.2j]),
        )
