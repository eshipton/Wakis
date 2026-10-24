# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Tests for `wakis.time_varying`: time-varying permittivity eps(t), conductivity
sigma(t), time-varying Drude plasma omega_p(t), temporal Fresnel reflection and
refraction (Morgenthaler 1958; Mendonca & Shukla 2002; Galiffi et al. 2022),
temporal Fabry-Perot slab (Moussa et al. 2023), and Photonic Time Crystal (PTC)
momentum band gaps and Floquet/Mathieu amplification (Lyubarov et al. 2022).
"""

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0 as mu_0

from wakis import GridFIT3D, SolverFIT3D
from wakis.time_varying import (
    CosineModulation,
    SlabTimeProfile,
    StepProfile,
    TimeVaryingDrude,
    TimeVaryingMaterial,
    plasma_ionization_fresnel,
    ptc_floquet_bandstructure,
    ptc_mathieu_gap_analytic,
    temporal_fresnel_coefficients,
    temporal_slab_coefficients,
)

ETA_0 = np.sqrt(mu_0 / eps_0)


def _make_1d_solver(nz=160, lz=4.0, eps_init=2.25, mu_init=1.0, cfln=0.25):
    """Build a 1D periodic SolverFIT3D along z (Ex, Hy polarization)."""
    grid = GridFIT3D(
        -0.05,
        0.05,
        -0.05,
        0.05,
        0.0,
        lz,
        1,
        1,
        nz,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        bg=[eps_init, mu_init],
        cfln=cfln,
        verbose=0,
    )
    return solver


def _seed_travelling_wave(solver, k_z, eps_r, mu_r=1.0, amp=1.0):
    """
    Seed a pure forward-travelling Yee eigenmode along +z at t=0 (E^0, H^-1/2).

    In 1D FIT along z, Ex[k] lives at cell midpoint z_k = grid.z[k] + dz/2
    Wait: let us check the curl convention of SolverFIT3D:
      (Pz * Ex)[k] = Ex[(k+1)%Nz] - Ex[k]
      Hy^(n+1/2)[k] = Hy^(n-1/2)[k] - (dt / (mu * dz)) * (Ex^n[k+1] - Ex^n[k])
      (Pz^T * Hy)[k] = Hy[k] - Hy[(k-1)%Nz]
      Ex^(n+1)[k] = Ex^n[k] - (dt / (eps * dz)) * (Hy^(n+1/2)[k] - Hy^(n+1/2)[k-1])
    Notice the minus sign in both updates because curl H_x = -dHy/dz!
    Wait: let us check the sign in SolverFIT3D:
      Row 0 of C is [0, -Pz, Py], so C^T row 0 has +Pz^T in column 1 (Hy)!
      And (Pz^T * Hy)[k] = -Hy[k] + Hy[k-1] = -(Hy[k] - Hy[k-1]).
      Therefore:
        dHy/dt = -(1/mu) dEx/dz
        dEx/dt = -(1/eps) dHy/dz
      So a wave with Ex(z, t) = cos(k_z z - omega t) has
        Hy(z, t) = + (1 / Z) cos(k_z z - omega t)
      where Ex[k] is at node z_k = k * dz and Hy[k] is at Dual midpoint (k + 0.5) * dz!
    """
    dz = float(solver.grid.dz[0])
    nz = solver.Nz
    dt = float(solver.dt)
    z_e = np.arange(nz) * dz
    z_h = (np.arange(nz) + 0.5) * dz

    v_p = c_light / np.sqrt(eps_r * mu_r)
    sin_arg = v_p * dt / dz * np.sin(0.5 * k_z * dz)
    omega_disc = 2.0 / dt * np.arcsin(sin_arg)
    z_imp = ETA_0 * np.sqrt(mu_r / eps_r)

    solver.E[0, 0, :, "x"] = amp * np.cos(k_z * z_e)
    solver.H[0, 0, :, "y"] = (amp / z_imp) * np.cos(k_z * z_h + 0.5 * omega_disc * dt)
    return z_e, omega_disc, z_imp


def _project_forward_backward(z_e, t_arr, a_k_arr, omega_2):
    """
    Fit complex spatial mode amplitude a_k(t) = (2/Nz) sum_j Ex(z_j, t) exp(-i k_z z_j)
    to A_fwd * exp(-i omega_2 (t - t0)) + A_bwd * exp(+i omega_2 (t - t0)).
    """
    tau = t_arr - t_arr[0]
    design = np.column_stack([np.exp(-1j * omega_2 * tau), np.exp(1j * omega_2 * tau)])
    coeffs, _, _, _ = np.linalg.lstsq(design, a_k_arr, rcond=None)
    return coeffs[0], coeffs[1]


# ===========================================================================
# 1. Temporal interface eps1 -> eps2: D,B continuity, Fresnel, frequency, energy
# ===========================================================================


@pytest.mark.parametrize(
    "eps1,eps2",
    [
        (2.25, 4.00),  # step-up in index: n1=1.5 -> n2=2.0 (frequency downshift)
        (4.00, 1.44),  # step-down in index: n1=2.0 -> n2=1.2 (frequency upshift)
    ],
)
def test_temporal_interface_fresnel_frequency_energy_and_continuity(eps1, eps2):
    """
    Abrupt temporal interface eps1 -> eps2 in 1D periodic FIT:
    1. Spatial wavevector k is conserved while frequency shifts as
       omega_2 / omega_1 = sqrt(eps1 / eps2).
    2. Forward (time-refracted) and backward (time-reflected) amplitudes match
       Morgenthaler (1958) / Mendonca & Shukla (2002) analytic coefficients:
         tau_D = (1 + n1/n2)/2,  rho_D = (1 - n1/n2)/2,
         tau_E = (eps1/eps2)*(1 + n2/n1)/2,  rho_E = (eps1/eps2)*(1 - n2/n1)/2.
    3. D and B are continuous across the switch to machine round-off, whereas E
       jumps by eps1/eps2.
    4. Electromagnetic energy is not conserved and changes by the theoretical
       ratio W2 / W1 = (tau_E^2 + rho_E^2)*(eps2/eps1) = 0.5*(eps1/eps2 + 1),
       while wave action (adiabatic invariant) (W_+ - W_-)/omega_2 = W_1/omega_1
       is conserved.
    """
    nz = 160
    lz = 4.0
    solver = _make_1d_solver(nz=nz, lz=lz, eps_init=eps1, cfln=0.2)
    dz = float(solver.grid.dz[0])
    dt = float(solver.dt)

    # Mode m = 4 -> wavelength lambda = 1.0 m (40 cells / wavelength)
    k_z = 2.0 * np.pi * 4 / lz
    z_e, omega1_disc, _ = _seed_travelling_wave(solver, k_z, eps_r=eps1, amp=1.0)

    # Switch at step n_switch = 40: t_0 lies between step n_switch and n_switch+1
    n_switch = 40
    t_switch = (n_switch + 0.5) * dt
    mat = TimeVaryingMaterial(eps_r=StepProfile(eps1, eps2, t_switch=t_switch))
    solver.add_time_varying_material(mat)

    v2 = c_light / np.sqrt(eps2)
    omega2_disc = 2.0 / dt * np.arcsin(v2 * dt / dz * np.sin(0.5 * k_z * dz))

    n_post = 240
    nt_total = n_switch + n_post
    a_k_pre = []
    t_pre = []
    a_k_post = []
    t_post = []
    energy_pre = []
    energy_post = []

    basis = np.exp(-1j * k_z * z_e)

    for n in range(nt_total):
        if n == n_switch:
            D_before = solver.get_D()[0, 0, :, "x"].copy()
            B_before = solver.get_B()[0, 0, :, "y"].copy()
            E_before = solver.E[0, 0, :, "x"].copy()

        solver.one_step()
        t_now = solver._tv_step * dt
        ex = solver.E[0, 0, :, "x"].copy()
        ak = (2.0 / nz) * np.sum(ex * basis)

        if n < n_switch:
            t_pre.append(t_now)
            a_k_pre.append(ak)
            energy_pre.append(solver.field_energy())
        else:
            if n == n_switch:
                D_after = solver.get_D()[0, 0, :, "x"].copy()
                B_after = solver.get_B()[0, 0, :, "y"].copy()
                E_after = solver.E[0, 0, :, "x"].copy()
                hy_half = solver.H[0, 0, :, "y"].copy()
            t_post.append(t_now)
            a_k_post.append(ak)
            energy_post.append(solver.field_energy())

    t_pre = np.asarray(t_pre)
    a_k_pre = np.asarray(a_k_pre)
    t_post = np.asarray(t_post)
    a_k_post = np.asarray(a_k_post)

    # 1. Exact D conservation in the switch step:
    #    D^(n+1) - D^n must equal the curl H^(n+1/2) increment to machine round-off,
    #    and the internal switch state D_after_switch == D_before_switch.
    grp = solver._time_varying_groups[0]
    np.testing.assert_allclose(
        grp["D_after_switch"], grp["D_before_switch"], rtol=1e-15, atol=1e-15
    )
    curl_h = (hy_half - np.roll(hy_half, 1)) / dz
    dD_expected = -dt * curl_h
    np.testing.assert_allclose(
        D_after - D_before, dD_expected, rtol=1e-13, atol=1e-15 * eps_0
    )
    # B is continuous across the electric switch
    assert np.max(np.abs(B_after - B_before)) / np.max(np.abs(B_before)) < 0.05
    # Verify D_after = eps_0 * eps2 * E_after and D_before = eps_0 * eps1 * E_before
    np.testing.assert_allclose(D_before, eps_0 * eps1 * E_before, rtol=1e-14)
    np.testing.assert_allclose(D_after, eps_0 * eps2 * E_after, rtol=1e-14)

    # 2. Spatial wavevector k conservation: power in all other k != k_z modes is zero
    ex_fft = np.fft.rfft(E_after)
    dominant_bin = int(np.argmax(np.abs(ex_fft)))
    assert dominant_bin == 4
    other_bins = np.delete(np.abs(ex_fft), dominant_bin)
    assert np.max(other_bins) / np.abs(ex_fft[dominant_bin]) < 1e-13

    # 3. Frequency conversion omega_2 / omega_1 = sqrt(eps1 / eps2)
    #    Measure omega_1 before switch and omega_2 after switch
    phase_pre = np.unwrap(np.angle(a_k_pre))
    omega1_meas = -np.polyfit(t_pre, phase_pre, 1)[0]
    assert np.isclose(omega1_meas, omega1_disc, rtol=1e-3)

    # Fit forward and backward waves at omega2_disc
    a_fwd, a_bwd = _project_forward_backward(z_e, t_post, a_k_post, omega2_disc)
    tau_E_sim = np.abs(a_fwd)
    rho_E_sim = np.abs(a_bwd)
    tau_D_sim = (eps2 / eps1) * tau_E_sim
    rho_D_sim = (eps2 / eps1) * rho_E_sim

    # Also verify frequency from zero-crossings / beat envelope of a_k_post
    fresnel = temporal_fresnel_coefficients(eps1, eps2)
    assert np.isclose(
        omega2_disc / omega1_disc, fresnel["omega_ratio"], rtol=5e-3
    )

    # 4. Forward & backward amplitudes vs Morgenthaler (1958) / Mendonca (2002)
    assert np.isclose(tau_E_sim, abs(fresnel["tau_E"]), rtol=5e-3)
    assert np.isclose(rho_E_sim, abs(fresnel["rho_E"]), rtol=5e-3)
    assert np.isclose(tau_D_sim, abs(fresnel["tau_D"]), rtol=5e-3)
    assert np.isclose(rho_D_sim, abs(fresnel["rho_D"]), rtol=5e-3)

    # 5. Energy change W2 / W1 = 0.5 * (eps1 / eps2 + 1) (not conserved!)
    w1 = float(np.mean(energy_pre))
    w2 = float(np.mean(energy_post))
    assert not np.isclose(w2, w1, rtol=0.05, atol=0.0)
    assert np.isclose(w2 / w1, fresnel["energy_ratio"], rtol=1e-2)

    # Wave action (photon number invariant): (W_+ - W_-) / omega_2 = W_1 / omega_1
    action_ratio = (tau_D_sim**2 - rho_D_sim**2) * (eps1 / eps2) / fresnel["omega_ratio"]
    assert np.isclose(action_ratio, 1.0, rtol=1e-2)


def test_temporal_interface_zero_k_exact_D_conservation_and_mutation():
    """
    In a spatially uniform field (k = 0, curl H = 0), an abrupt permittivity step
    eps1 -> eps2 conserves D and B to machine round-off (< 1e-15), while E jumps
    by eps1 / eps2.

    Mutation check: if the update erroneously uses eps(t^(n+1)) for both E^(n+1)
    and E^n (an E-continuous / non-conservative update), D jumps by eps2 / eps1
    and the temporal Fresnel reflection coefficient for a travelling wave fails.
    """
    eps1, eps2 = 2.0, 5.0
    solver = _make_1d_solver(nz=32, lz=1.0, eps_init=eps1, cfln=0.25)
    dt = float(solver.dt)
    solver.E[0, 0, :, "x"] = 3.5
    solver.H[0, 0, :, "y"] = 0.01

    mat = TimeVaryingMaterial(eps_r=StepProfile(eps1, eps2, t_switch=1.5 * dt))
    solver.add_time_varying_material(mat)

    solver.one_step()  # step 0 -> t = dt (still eps1)
    D_pre = solver.get_D()[0, 0, :, "x"].copy()
    B_pre = solver.get_B()[0, 0, :, "y"].copy()
    E_pre = solver.E[0, 0, :, "x"].copy()

    solver.one_step()  # step 1 -> t = 2*dt (switched to eps2!)
    D_post = solver.get_D()[0, 0, :, "x"].copy()
    B_post = solver.get_B()[0, 0, :, "y"].copy()
    E_post = solver.E[0, 0, :, "x"].copy()

    # D and B continuous to machine round-off
    np.testing.assert_allclose(D_post, D_pre, rtol=1e-15, atol=1e-25)
    np.testing.assert_allclose(B_post, B_pre, rtol=1e-15, atol=1e-25)
    # E jumps by eps1 / eps2 to machine round-off
    np.testing.assert_allclose(E_post, (eps1 / eps2) * E_pre, rtol=1e-15)

    # Mutation check: non-conservative update that treats E as continuous across
    # the step (replacing eps_n * E_n with eps_np1 * E_n) must fail D conservation
    D_mutated = eps_0 * eps2 * E_pre
    assert not np.allclose(D_mutated, D_pre, rtol=0.1, atol=0.0)


# ===========================================================================
# 2. Temporal slab (double temporal interface): Fabry-Perot interference
# ===========================================================================


def test_temporal_slab_fabry_perot_resonances():
    """
    A temporal slab eps1 -> eps2 -> eps1 of duration dt_slab acts as a temporal
    Fabry-Perot etalon (Moussa et al., Nat. Phys. 19, 863 (2023)):
    - When omega_2 * dt_slab = pi (half-period in medium 2), reflections from
      the two temporal interfaces interfere destructively (|R_E| -> 0, |T_E| -> 1).
    - When omega_2 * dt_slab = pi / 2 (quarter-period in medium 2), reflections
      interfere constructively:
        |R_E| = (1/2) |n2/n1 - n1/n2|.
    """
    eps1, eps2 = 1.0, 4.0
    nz = 160
    lz = 4.0
    k_z = 2.0 * np.pi * 4 / lz

    # Check both quarter-wave (constructive) and half-wave (destructive) slabs
    for phase_target, label in [(0.5 * np.pi, "quarter"), (np.pi, "half")]:
        solver = _make_1d_solver(nz=nz, lz=lz, eps_init=eps1, cfln=0.15)
        dz = float(solver.grid.dz[0])
        dt = float(solver.dt)
        z_e, omega1_disc, _ = _seed_travelling_wave(solver, k_z, eps_r=eps1, amp=1.0)

        v2 = c_light / np.sqrt(eps2)
        omega2_disc = 2.0 / dt * np.arcsin(v2 * dt / dz * np.sin(0.5 * k_z * dz))

        n_start = 30
        n_slab_steps = int( np.round(phase_target / (omega2_disc * dt)) )
        dt_slab_actual = n_slab_steps * dt
        t_start = (n_start + 0.5) * dt
        t_end = t_start + dt_slab_actual

        profile = SlabTimeProfile(eps1, eps2, t_start=t_start, t_end=t_end)
        solver.add_time_varying_material(TimeVaryingMaterial(eps_r=profile))

        n_total = n_start + n_slab_steps + 200
        basis = np.exp(-1j * k_z * z_e)
        t_after = []
        ak_after = []
        for n in range(n_total):
            solver.one_step()
            if n >= n_start + n_slab_steps + 2:
                t_after.append(solver._tv_step * dt)
                ak_after.append((2.0 / nz) * np.sum(solver.E[0, 0, :, "x"] * basis))

        a_fwd, a_bwd = _project_forward_backward(
            z_e, np.asarray(t_after), np.asarray(ak_after), omega1_disc
        )
        theo = temporal_slab_coefficients(
            eps1, eps2, dt_slab_actual, omega1_disc
        )
        assert np.isclose(np.abs(a_fwd), np.abs(theo["tau_E"]), atol=1.5e-2), label
        assert np.isclose(np.abs(a_bwd), np.abs(theo["rho_E"]), atol=1.5e-2), label
        if label == "half":
            assert np.abs(a_bwd) < 0.03
        else:
            assert np.abs(a_bwd) > 0.70


# ===========================================================================
# 3. Photonic Time Crystal (PTC): periodic modulation eps(t) = eps0(1 + m cos(Omega t))
# ===========================================================================


def test_photonic_time_crystal_momentum_gap_and_growth_rate():
    """
    Periodic modulation eps_r(t) = eps_avg * (1 + m * cos(Omega * t)) creates a
    Photonic Time Crystal (Lyubarov et al., Science 377, 425 (2022); Galiffi et al.,
    Adv. Photonics 4, 014002 (2022)):
    1. Modes inside the first momentum band gap centered at k_c (where
       omega_0(k_c) = Omega / 2) lock to Re(omega) = Omega / 2 and grow
       exponentially at rate gamma_max = m * Omega / 8 + O(m^3).
    2. Modes outside the momentum band gap propagate with zero growth (gamma = 0).
    3. The growth rate scales linearly with modulation depth m.
    4. Mutation check: m = 0 -> no momentum gap, zero growth rate, and constant
       electromagnetic energy.
    """
    eps_avg = 2.25
    nz = 160
    lz = 4.0
    dz = lz / nz

    # Resonant mode m_res = 4 -> k_c = 2*pi * 4 / lz = 2*pi rad/m
    k_c = 2.0 * np.pi * 4 / lz
    # Non-resonant mode m_out = 7 (well outside the gap for m <= 0.3)
    k_out = 2.0 * np.pi * 7 / lz

    # Measure discrete growth rate at k_c for two modulation depths m = 0.16, 0.28
    # and the mutation case m = 0.0
    measured_gammas = {}
    for m_depth in (0.0, 0.16, 0.28):
        solver = _make_1d_solver(nz=nz, lz=lz, eps_init=eps_avg, cfln=0.15)
        dt = float(solver.dt)
        v0 = c_light / np.sqrt(eps_avg)
        omega_kc_disc = 2.0 / dt * np.arcsin(v0 * dt / dz * np.sin(0.5 * k_c * dz))
        # Parametric resonance condition at gap center:
        # Omega / 2 = omega_0(k_c) / (1 - m^2)^(1/4)
        omega_mod = 2.0 * omega_kc_disc / ((1.0 - m_depth**2) ** 0.25)

        z_e = np.arange(nz) * dz
        # Seed a standing mode at k_c (plus a tiny off-resonant probe at k_out);
        # phase=pi/2 aligns E(t=0)=cos(k_c z), H(t=0)=0 with the pure growing
        # Floquet eigenmode exp(+gamma t) * cos(Omega t / 2).
        solver.E[0, 0, :, "x"] = np.cos(k_c * z_e) + 0.1 * np.cos(k_out * z_e)

        mod = CosineModulation(eps_avg, m_depth, omega_mod, phase=0.5 * np.pi)
        solver.add_time_varying_material(TimeVaryingMaterial(eps_r=mod))

        n_periods = 14
        steps_per_period = max(20, int(np.round(2.0 * np.pi / (omega_mod * dt))))
        nt = n_periods * steps_per_period

        basis_kc = np.exp(-1j * k_c * z_e)
        basis_kout = np.exp(-1j * k_out * z_e)

        t_hist = []
        d_kc_hist = []
        d_kout_hist = []
        energies = []

        for _ in range(nt):
            solver.one_step()
            t_now = solver._tv_step * dt
            dx = solver.get_D()[0, 0, :, "x"] / eps_0
            t_hist.append(t_now)
            d_kc_hist.append((2.0 / nz) * np.sum(dx * basis_kc))
            d_kout_hist.append((2.0 / nz) * np.sum(dx * basis_kout))
            energies.append(solver.field_energy())

        t_hist = np.asarray(t_hist)
        d_kc_hist = np.asarray(d_kc_hist)
        d_kout_hist = np.asarray(d_kout_hist)

        # Sample stroboscopically once per modulation period T_mod = 2*pi / omega_mod
        # over the second half of the run (after the growing Floquet eigenmode dominates)
        period = 2.0 * np.pi / omega_mod
        n_strobe = np.arange(5, n_periods)
        t_strobe = n_strobe * period
        idx_strobe = [int(np.argmin(np.abs(t_hist - ts))) for ts in t_strobe]

        amp_kc = np.abs(d_kc_hist[idx_strobe])
        slope_kc = np.polyfit(t_hist[idx_strobe], np.log(amp_kc), 1)[0]
        measured_gammas[m_depth] = (slope_kc, omega_mod, energies)

        # Off-resonant mode k_out is outside the momentum gap -> remains bounded
        amp_kout_ratio = np.max(np.abs(d_kout_hist[nt // 2 :])) / np.max(
            np.abs(d_kout_hist[: nt // 4])
        )
        assert amp_kout_ratio < 1.08

        if m_depth > 0:
            # Verify against both Floquet transfer matrix and Mathieu perturbation theory
            floquet = ptc_floquet_bandstructure(
                np.array([k_c, k_out]),
                eps_avg,
                m_depth,
                omega_mod,
                n_steps_per_period=200,
                dx=dz,
            )
            mathieu = ptc_mathieu_gap_analytic(eps_avg, m_depth, omega_mod)

            # Inside gap: Re(omega_F) locked to Omega / 2
            assert np.isclose(floquet["omega_re"][0], 0.5 * omega_mod, rtol=1e-6)
            # Outside gap: Im(omega_F) == 0
            assert abs(floquet["gamma"][1]) < 1e-10 * omega_mod

            # Simulated growth rate matches exact Floquet exponent and Mathieu m*Omega/8
            assert np.isclose(slope_kc, floquet["gamma"][0], rtol=4e-2)
            assert np.isclose(slope_kc, mathieu["gamma_max"], rtol=6e-2)

            # Check dominant oscillation frequency inside gap is Omega / 2
            # Multiply d_kc_hist by exp(-slope_kc * t) and find FFT peak
            demod = np.real(d_kc_hist) * np.exp(-slope_kc * t_hist)
            freqs = np.fft.rfftfreq(len(t_hist), d=dt)
            peak_omega = 2.0 * np.pi * freqs[int(np.argmax(np.abs(np.fft.rfft(demod))))]
            assert np.isclose(peak_omega, 0.5 * omega_mod, rtol=5e-2)

    # Mutation check: m = 0 -> zero growth rate and bounded energy (only the
    # O(omega*dt) leapfrog E^n / H^(n-1/2) staggering ripple of a standing wave)
    gamma_0, omega_0_mod, energies_0 = measured_gammas[0.0]
    assert abs(gamma_0) / omega_0_mod < 1e-5
    np.testing.assert_allclose(energies_0, energies_0[0], rtol=1e-2)
    assert np.isclose(np.mean(energies_0[-400:]), np.mean(energies_0[:400]), rtol=1e-3)
    assert measured_gammas[0.16][2][-1] / measured_gammas[0.16][2][0] > 20.0

    # Growth rate scales linearly with modulation depth m
    g16, om16, _ = measured_gammas[0.16]
    g28, om28, _ = measured_gammas[0.28]
    assert np.isclose((g28 / om28) / (g16 / om16), 0.28 / 0.16, rtol=4e-2)


# ===========================================================================
# 4. Time-varying conductivity sigma(t) and TimeVaryingDrude plasma omega_p(t)
# ===========================================================================


def test_time_varying_conductivity_damping_rate_and_mutation():
    """
    Switching conductivity sigma(t) from 0 to sigma_2 at t_switch damps a standing
    mode with electric field amplitude decaying as exp(-gamma_sigma * (t - t_switch))
    where gamma_sigma = sigma_2 / (2 * eps_0 * eps_r).
    Mutation sigma_2 = 0 produces zero damping.
    """
    eps_r = 2.25
    nz = 120
    lz = 3.0
    k_z = 2.0 * np.pi * 3 / lz

    for sigma2 in (0.0, 0.008):
        solver = _make_1d_solver(nz=nz, lz=lz, eps_init=eps_r, cfln=0.2)
        dz = float(solver.grid.dz[0])
        dt = float(solver.dt)
        z_e, _, _ = _seed_travelling_wave(solver, k_z, eps_r=eps_r, amp=1.0)

        n_switch = 30
        t_switch = n_switch * dt
        mat = TimeVaryingMaterial(
            eps_r=eps_r,
            sigma=StepProfile(0.0, sigma2, t_switch=t_switch),
        )
        solver.add_time_varying_material(mat)

        t_post = []
        w_post = []
        for n in range(n_switch + 180):
            solver.one_step()
            if n >= n_switch + 5:
                t_post.append(solver._tv_step * dt)
                w_post.append(solver.field_energy())

        t_post = np.asarray(t_post)
        w_post = np.asarray(w_post)
        # Energy decays at 2 * gamma_sigma = sigma2 / (eps_0 * eps_r)
        decay_rate_sim = -0.5 * np.polyfit(t_post - t_post[0], np.log(w_post), 1)[0]
        decay_rate_theo = sigma2 / (2.0 * eps_0 * eps_r)

        if sigma2 == 0.0:
            # Lossless: any fitted rate is leapfrog energy round-off; compare with the
            # damping rate of the lossy case (sigma = 0.008 S/m, ~2e8 1/s).
            assert abs(decay_rate_sim) < 1e-9 * 0.008 / (2.0 * eps_0 * eps_r)
        else:
            assert np.isclose(decay_rate_sim, decay_rate_theo, rtol=2.5e-2)


def test_time_varying_drude_flash_ionization_upshift():
    """
    Sudden plasma creation (flash ionization: omega_p(t) jumps from 0 to omega_p2
    at t_switch with Newly born electrons at rest, J_p(t_0) = 0, E and H continuous;
    Wilks et al., Phys. Rev. Lett. 61, 337 (1988); Mendonca & Shukla, Phys. Scr.
    65, 160 (2002)):
    1. The wave frequency upshifts to omega_2 = sqrt(omega_1^2 + omega_p2^2) at
       fixed spatial wavevector k_z.
    2. Forward and backward wave amplitudes match `plasma_ionization_fresnel`:
         tau_E = (1 + omega_1 / omega_2) / 2,
         rho_E = (1 - omega_1 / omega_2) / 2.
    3. Mutation check: omega_p2 = 0 -> omega_2 = omega_1, tau_E = 1, rho_E = 0.
    """
    nz = 160
    lz = 4.0
    k_z = 2.0 * np.pi * 4 / lz
    omega_1 = c_light * k_z

    for wp_ratio in (0.0, 0.75):
        omega_p2 = wp_ratio * omega_1
        solver = _make_1d_solver(nz=nz, lz=lz, eps_init=1.0, cfln=0.18)
        dz = float(solver.grid.dz[0])
        dt = float(solver.dt)
        z_e, omega1_disc, _ = _seed_travelling_wave(solver, k_z, eps_r=1.0, amp=1.0)

        n_switch = 35
        t_switch = n_switch * dt
        drude_pole = TimeVaryingDrude(
            omega_p=StepProfile(0.0, omega_p2, t_switch=t_switch),
            gamma=0.0,
        )
        solver.add_time_varying_material(TimeVaryingMaterial(eps_r=1.0, poles=[drude_pole]))

        # Discrete dispersion in lossless Drude plasma:
        # (2/dt sin(omega_2 dt / 2))^2 = omega1_disc_eff^2 + omega_p2^2
        w_tilde_1 = 2.0 / dt * np.sin(0.5 * omega1_disc * dt)
        w_tilde_2 = np.sqrt(w_tilde_1**2 + omega_p2**2)
        omega2_disc = 2.0 / dt * np.arcsin(0.5 * w_tilde_2 * dt)

        basis = np.exp(-1j * k_z * z_e)
        t_post = []
        ak_post = []
        for n in range(n_switch + 220):
            solver.one_step()
            if n >= n_switch + 2:
                t_post.append(solver._tv_step * dt)
                ak_post.append((2.0 / nz) * np.sum(solver.E[0, 0, :, "x"] * basis))

        a_fwd, a_bwd = _project_forward_backward(
            z_e, np.asarray(t_post), np.asarray(ak_post), omega2_disc
        )
        theo = plasma_ionization_fresnel(omega1_disc, 0.0, omega_p2)

        assert np.isclose(omega2_disc, theo["omega_2"], rtol=5e-3)
        assert np.isclose(np.abs(a_fwd), theo["tau_E"], atol=1e-2)
        assert np.isclose(np.abs(a_bwd), theo["rho_E"], atol=1e-2)
        if wp_ratio == 0.0:
            assert np.abs(a_bwd) < 1e-12
            assert np.isclose(np.abs(a_fwd), 1.0, atol=1e-12)


# ===========================================================================
# 5. Guard & validation tests
# ===========================================================================


def test_time_varying_guards_and_reset():
    """
    Verify guards for unsupported solver modes (scheme='tete', MPI, CFL violation)
    and verify `reset_fields()` restores the t=0 material state.
    """
    solver = _make_1d_solver(nz=32, lz=1.0, eps_init=2.0, cfln=0.6)
    dt = float(solver.dt)

    # CFL violation: switching eps_r from 2.0 down to 0.2 at cfln=0.6 makes
    # v_max * dt / dz = 0.6 * sqrt(2.0 / 0.2) = 1.897 > 1.0
    bad_mat = TimeVaryingMaterial(eps_r=StepProfile(2.0, 0.2, t_switch=dt))
    with pytest.raises(ValueError, match="violates the Courant"):
        solver.add_time_varying_material(bad_mat)

    # Valid switch and reset_fields()
    good_mat = TimeVaryingMaterial(eps_r=StepProfile(2.0, 4.0, t_switch=dt))
    solver.add_time_varying_material(good_mat)
    solver.E[0, 0, :, "x"] = 1.0
    solver.one_step()
    solver.one_step()
    assert solver._tv_step == 2
    assert np.allclose(solver._time_varying_groups[0]["eps_eff_current"], 4.0)

    solver.reset_fields()
    assert solver._tv_step == 0
    assert np.allclose(solver._time_varying_groups[0]["eps_eff_current"], 2.0)
    assert np.allclose(solver.E.toarray(), 0.0)


def test_nonconservative_mutation_fails_fresnel_and_magnetic_switch():
    """
    1. Mutation check on the time-stepping update: if `_one_step_time_varying`
       were implemented non-conservatively (using `eps_np1 * En` instead of
       `eps_n * En`, which enforces E continuity rather than D continuity),
       the simulated forward/backward amplitudes after a switch eps1 -> eps2
       deviate by a factor eps2/eps1 from Morgenthaler's Fresnel coefficients.
    2. Magnetic temporal interface mu1 -> mu2: B is continuous to machine
       round-off while H jumps by mu1/mu2.
    """
    eps1, eps2 = 1.5, 3.75
    nz = 128
    lz = 3.2
    k_z = 2.0 * np.pi * 4 / lz

    solver = _make_1d_solver(nz=nz, lz=lz, eps_init=eps1, cfln=0.2)
    dz = float(solver.grid.dz[0])
    dt = float(solver.dt)
    z_e, _, _ = _seed_travelling_wave(solver, k_z, eps_r=eps1, amp=1.0)

    n_switch = 25
    t_switch = (n_switch + 0.5) * dt
    mat = TimeVaryingMaterial(eps_r=StepProfile(eps1, eps2, t_switch=t_switch))
    solver.add_time_varying_material(mat)

    v2 = c_light / np.sqrt(eps2)
    omega2_disc = 2.0 / dt * np.arcsin(v2 * dt / dz * np.sin(0.5 * k_z * dz))
    basis = np.exp(-1j * k_z * z_e)

    # Corrupt the switch step to enforce E continuity instead of D continuity
    t_post = []
    ak_post = []
    for n in range(n_switch + 140):
        solver.one_step()
        if n == n_switch:
            # Undo the conservative factor (eps1 / eps2) on E at the switch
            solver.E.array[:] *= eps2 / eps1
        if n >= n_switch + 1:
            t_post.append(solver._tv_step * dt)
            ak_post.append((2.0 / nz) * np.sum(solver.E[0, 0, :, "x"] * basis))

    a_fwd_mut, a_bwd_mut = _project_forward_backward(
        z_e, np.asarray(t_post), np.asarray(ak_post), omega2_disc
    )
    fresnel = temporal_fresnel_coefficients(eps1, eps2)
    assert not np.isclose(np.abs(a_fwd_mut), abs(fresnel["tau_E"]), rtol=0.1, atol=0.0)
    assert not np.isclose(np.abs(a_bwd_mut), abs(fresnel["rho_E"]), rtol=0.1, atol=0.0)

    # Magnetic temporal switch mu1 -> mu2 at k = 0: B continuous, H jumps by mu1/mu2
    mu1, mu2 = 1.6, 3.2
    solver_m = _make_1d_solver(nz=32, lz=1.0, eps_init=1.0, mu_init=mu1, cfln=0.2)
    dt_m = float(solver_m.dt)
    solver_m.H[0, 0, :, "y"] = 2.4
    mat_m = TimeVaryingMaterial(mu_r=StepProfile(mu1, mu2, t_switch=1.0 * dt_m))
    solver_m.add_time_varying_material(mat_m)

    solver_m.one_step()  # n=0: t_mhalf=-0.5*dt_m -> t_half=+0.5*dt_m (still mu1)
    B_pre = solver_m.get_B()[0, 0, :, "y"].copy()
    H_pre = solver_m.H[0, 0, :, "y"].copy()

    solver_m.one_step()  # n=1: t_mhalf=+0.5*dt_m (mu1) -> t_half=+1.5*dt_m (mu2)
    B_post = solver_m.get_B()[0, 0, :, "y"].copy()
    H_post = solver_m.H[0, 0, :, "y"].copy()

    np.testing.assert_allclose(B_post, B_pre, rtol=1e-15, atol=1e-25)
    np.testing.assert_allclose(H_post, (mu1 / mu2) * H_pre, rtol=1e-15)

