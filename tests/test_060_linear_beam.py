# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Validation, stability, energy-balance, adjoint-transpose, and mutation tests
for `wakis.beam_fluid.LinearizedBeam`.
"""

import warnings

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import m_e

import wakis
from wakis.beam_fluid import (
    LinearizedBeam,
    _beam_amplification_matrix,
    ballistic_bunching_analytic,
    beam_plasma_frequency,
    reduction_factor_cylindrical,
    reduction_factor_planar,
    reduction_factor_rectangular,
    relativistic_gamma,
    space_charge_wave_dispersion,
)
from wakis.inverse import AdjointSimulation, AdjointStepper, MaterialGrid
from wakis.plasma import cold_plasma
from wakis.sources import CurrentSource, GaussianSource


# ---------------------------------------------------------------------------
# 1. Analytic helper & reduction-factor unit tests
# ---------------------------------------------------------------------------


def test_relativistic_kinematics_and_reduction_factors():
    """
    Verify relativistic kinematics, planar/cylindrical/rectangular reduction
    factors, asymptotic limits, and the Mealy, Marosi & Capolino (2022)
    cylindrical beam benchmark (R_b=0.5 mm, R_t=2.0 mm, V0=6.7 kV, f=5 GHz).
    """
    u0, gamma0 = relativistic_gamma(voltage=6.7e3)
    u0_chk, gamma0_chk = relativistic_gamma(u0=u0)
    assert np.isclose(u0, u0_chk, rtol=1e-14)
    assert np.isclose(gamma0, gamma0_chk, rtol=1e-14)
    assert np.isclose(gamma0**2 * (1.0 - (u0 / c_light) ** 2), 1.0, atol=1e-14)

    # Asymptotic limits of planar and cylindrical reduction factors
    assert np.isclose(reduction_factor_planar(0.0, u0, 0.01, 0.02), 0.0)
    assert np.isclose(reduction_factor_cylindrical(0.0, u0, 0.001, 0.002), 0.0)
    assert reduction_factor_planar(5000.0 * u0, u0, 0.01, 0.02) > 0.998
    assert reduction_factor_cylindrical(5000.0 * u0, u0, 0.01, 0.02) > 0.998

    # Monotonicity with respect to frequency omega and filling factor b/a
    omegas = np.linspace(20.0, 400.0, 20) * u0
    r_planar = reduction_factor_planar(omegas, u0, 0.005, 0.015)
    r_cyl = reduction_factor_cylindrical(omegas, u0, 0.005, 0.015)
    assert np.all(np.diff(r_planar) > 0)
    assert np.all(np.diff(r_cyl) > 0)
    assert reduction_factor_planar(
        100.0 * u0, u0, 0.008, 0.010
    ) < reduction_factor_planar(100.0 * u0, u0, 0.008, 0.020)

    # Mealy, Marosi & Capolino (IEEE TPS 2022): R_b=0.5 mm, R_t=2.0 mm,
    # u0 = 0.16 c, f = 5 GHz -> R = 0.268
    u0_mmc = 0.16 * c_light
    omega_mmc = 2.0 * np.pi * 5.0e9
    R_exact = reduction_factor_cylindrical(omega_mmc, u0_mmc, 0.5e-3, 2.0e-3)
    R_nonrel = reduction_factor_cylindrical(
        omega_mmc, u0_mmc, 0.5e-3, 2.0e-3, relativistic=False
    )
    assert np.isclose(R_exact, 0.268, rtol=1e-2)
    assert np.isclose(R_nonrel, 0.268, rtol=1e-2)

    # Rectangular beam reduces to planar sheet beam when one transverse dimension is wide
    R_rect_wide = reduction_factor_rectangular(
        120.0 * u0, u0, 0.005, 10.0, 0.015, 10.0
    )
    R_plan_ref = reduction_factor_planar(120.0 * u0, u0, 0.005, 0.015)
    assert np.isclose(R_rect_wide, R_plan_ref, rtol=1e-3)


# ---------------------------------------------------------------------------
# 2. Space-charge wave dispersion (2D sheet beam & 3D rectangular/round beam)
# ---------------------------------------------------------------------------


def _measure_2d_sheet_dispersion(
    b_half=0.01,
    a_half=0.02,
    beta=2.0 * np.pi / 0.08,
    u0=0.35 * c_light,
    wp=2.0 * np.pi * 0.45e9,
    gamma_power=3.0,
    coupling_sign=1.0,
):
    """
    Measure fast and slow space-charge wave frequencies omega_+, omega_- and
    reduction factor R for a 2D sheet beam (Nz=1, axis='x', Ex/Ey/Hz) using
    Bloch boundary conditions along x and PEC walls at y = +-a_half.
    """
    Lx = 2.0 * np.pi / beta
    Nx, Ny, Nz = 32, 32, 1
    dx = Lx / Nx
    grid = wakis.GridFIT3D(
        0.0, Lx, -a_half, a_half, -0.5 * dx, 0.5 * dx, Nx, Ny, Nz, verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["bloch", "pec", "periodic"],
        bc_high=["bloch", "pec", "periodic"],
        k_bloch=[-beta, 0.0, 0.0],
        dtype=np.complex128,
        cfln=0.45,
        verbose=0,
    )
    _, gamma0 = relativistic_gamma(u0=u0)
    beam = LinearizedBeam(
        u0=u0,
        omega_p=wp,
        axis="x",
        half_thickness=b_half,
        gamma_power=gamma_power,
        coupling_sign=coupling_sign,
        solver=solver,
    )

    phase = np.exp(-1j * beta * beam.s_coords)[:, None]
    y_lines = grid.y[: solver.Ny][beam.keep_lines]
    prof_y = np.cos(0.5 * np.pi * y_lines / a_half)[None, :]
    beam.v1[:] = 1e4 * phase * prof_y
    solver.one_step()

    Nt = 4096
    trace = np.zeros(Nt, dtype=np.complex128)
    iy_mid = Ny // 2
    for n in range(Nt):
        solver.one_step()
        trace[n] = solver.E[Nx // 4, iy_mid, 0, "x"]

    win = np.hanning(Nt)
    n_fft = 65536
    spec = np.abs(np.fft.fft(trace * win, n=n_fft))
    freqs = np.fft.fftfreq(n_fft, d=solver.dt)
    pos = freqs > 0
    f_pos = freqs[pos]
    s_pos = spec[pos]

    f0 = beta * u0 / (2.0 * np.pi)
    band = (f_pos > 0.2 * f0) & (f_pos < 1.8 * f0)
    f_band = f_pos[band]
    s_band = s_pos[band]

    def peak_freq(mask_sub):
        idx_sub = np.where(mask_sub)[0]
        k = idx_sub[np.argmax(s_band[mask_sub])]
        if 0 < k < len(s_band) - 1:
            y0, y1, y2 = np.log(s_band[k - 1 : k + 2] + 1e-30)
            delta = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
            df = f_band[1] - f_band[0]
            return f_band[k] + delta * df
        return f_band[k]

    f_slow = peak_freq(f_band < f0)
    f_fast = peak_freq(f_band > f0)
    omega_slow = 2.0 * np.pi * f_slow
    omega_fast = 2.0 * np.pi * f_fast
    omega_q_meas = 0.5 * (omega_fast - omega_slow)
    R_meas = omega_q_meas * (gamma0**1.5) / wp
    return omega_fast, omega_slow, omega_q_meas, R_meas


@pytest.mark.parametrize("fill_ratio", [0.375, 0.5, 0.75])
def test_2d_sheet_beam_dispersion_vs_analytic(fill_ratio):
    """
    2D sheet beam in a parallel-plate guide (`SolverFIT3D` Nz=1, axis='x'):
    omega = beta u0 +- omega_q and measured reduction factor R agree with
    `space_charge_wave_dispersion` and `reduction_factor_planar` to < 1.5%.
    """
    a_half = 0.02
    b_half = fill_ratio * a_half
    beta = 2.0 * np.pi / 0.08
    u0 = 0.35 * c_light
    wp = 2.0 * np.pi * 0.45e9

    omega_fast, omega_slow, omega_q_meas, R_meas = _measure_2d_sheet_dispersion(
        b_half=b_half, a_half=a_half, beta=beta, u0=u0, wp=wp
    )
    R_th = reduction_factor_planar(beta * u0, u0, b_half, a=a_half)
    omega_fast_th, omega_slow_th = space_charge_wave_dispersion(
        beta, u0, wp, R=R_th
    )
    omega_q_th = 0.5 * (omega_fast_th - omega_slow_th)
    assert np.isclose(
        0.5 * (omega_fast + omega_slow), beta * u0, rtol=1.5e-2
    ), "Mean of fast and slow frequencies must match the beam line beta * u0"
    assert np.isclose(R_meas, R_th, rtol=1.5e-2)
    assert np.isclose(omega_q_meas, omega_q_th, rtol=1.5e-2)


def test_modulated_beam_spatial_beat_length_2d():
    """
    Measure omega_q and R from the spatial current beat length L_beat = pi / beta_q
    of a velocity-modulated sheet beam drifting along an open parallel-plate
    tunnel (`SolverFIT3D` Nz=1, axis='x'), and compare with ballistic theory.
    """
    Lx = 0.20
    a_half = 0.016
    b_half = 0.008
    Nx, Ny, Nz = 128, 16, 1
    dx = Lx / Nx
    grid = wakis.GridFIT3D(
        0.0, Lx, -a_half, a_half, -0.5 * dx, 0.5 * dx, Nx, Ny, Nz, verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "pec", "periodic"],
        bc_high=["cpml", "pec", "periodic"],
        n_pml=6,
        cfln=0.5,
        verbose=0,
    )
    u0 = 0.30 * c_light
    f_mod = 1.8e9
    omega_mod = 2.0 * np.pi * f_mod
    wp = 2.0 * np.pi * 0.55e9
    t_transit = Lx / u0

    mask = (np.abs(grid.y[: solver.Ny]) <= b_half + 1e-12)[None, :, None] * np.ones(
        (solver.Nx, solver.Ny, solver.Nz)
    )
    k_gap = 3

    beam = LinearizedBeam(
        u0=u0,
        omega_p=wp,
        axis="x",
        mask=mask,
        solver=solver,
    )
    x_gap = float(beam.s_coords[k_gap])

    def gap_field(s_coords, t):
        ramp = min(1.0, (t / (0.3 * t_transit)) ** 2)
        prof = np.exp(-0.5 * ((s_coords - x_gap) / (1.5 * dx)) ** 2)
        prof /= np.sum(prof * dx)
        return 10.0 * ramp * np.cos(omega_mod * t) * prof

    beam.gap_modulation = gap_field

    n_steps = int(3.0 * t_transit / solver.dt)
    n_period = max(int(round(1.0 / (f_mod * solver.dt))), 20)
    i1_phasor = np.zeros(beam.M, dtype=np.complex128)

    for n in range(n_steps):
        solver.one_step()
        if n >= n_steps - n_period:
            phasor = np.exp(-1j * omega_mod * (n + 0.5) * solver.dt) / n_period
            i1_phasor += beam.axial_current() * phasor

    s_drift = beam.s_coords[k_gap + 3 : beam.M - 8] - beam.s_coords[k_gap]
    j_env = np.abs(i1_phasor[k_gap + 3 : beam.M - 8])

    i_peak = int(np.argmax(j_env))
    s_peak = s_drift[i_peak]
    beta_q_meas = 0.5 * np.pi / s_peak
    omega_q_meas = beta_q_meas * u0

    R_th = reduction_factor_planar(omega_mod, u0, b_half, a=a_half)
    bunch = ballistic_bunching_analytic(
        s_drift, f_mod, u0, I0=1.0, Vm=10.0, R=R_th, omega_p=wp
    )
    assert np.isclose(omega_q_meas, bunch["omega_q"], rtol=0.03), (
        f"Measured omega_q={omega_q_meas:.4e} vs analytic {bunch['omega_q']:.4e}"
    )

    analytic_env = np.abs(bunch["Ib"])
    analytic_env /= analytic_env.max()
    meas_env = j_env / j_env.max()
    corr = np.corrcoef(meas_env, analytic_env)[0, 1]
    assert corr > 0.985, f"Envelope correlation {corr:.4f} < 0.985"


def test_3d_beam_in_rectangular_pipe_dispersion():
    """
    3D beam in a rectangular PEC pipe (`SolverFIT3D` 3D, axis='z'): measure
    fast/slow space-charge wave frequencies and reduction factor R against
    `reduction_factor_rectangular`.
    """
    a_half = 0.015
    b_half = 0.0075
    Lz = 0.08
    beta = 2.0 * np.pi / Lz
    Nx, Ny, Nz = 12, 12, 24
    grid = wakis.GridFIT3D(
        -a_half, a_half, -a_half, a_half, 0.0, Lz, Nx, Ny, Nz, verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["pec", "pec", "bloch"],
        bc_high=["pec", "pec", "bloch"],
        k_bloch=[0.0, 0.0, -beta],
        dtype=np.complex128,
        cfln=0.45,
        verbose=0,
    )
    u0 = 0.32 * c_light
    _, gamma0 = relativistic_gamma(u0=u0)
    wp = 2.0 * np.pi * 1.5e9

    beam = LinearizedBeam(
        u0=u0,
        omega_p=wp,
        axis="z",
        half_thickness=(b_half, b_half),
        solver=solver,
    )

    phase = np.exp(-1j * beta * beam.s_coords)[:, None]
    beam.v1[:] = 1e4 * phase * np.ones((beam.M, beam.N_lines))
    solver.one_step()

    Nt = 2400
    trace = np.zeros(Nt, dtype=np.complex128)
    for n in range(Nt):
        solver.one_step()
        trace[n] = solver.E[Nx // 2, Ny // 2, Nz // 4, "z"]

    win = np.hanning(Nt)
    n_fft = 32768
    spec = np.abs(np.fft.fft(trace * win, n=n_fft))
    freqs = np.fft.fftfreq(n_fft, d=solver.dt)
    pos = freqs > 0
    f_pos, s_pos = freqs[pos], spec[pos]
    f0 = beta * u0 / (2.0 * np.pi)
    band = (f_pos > 0.25 * f0) & (f_pos < 1.75 * f0)
    f_band, s_band = f_pos[band], s_pos[band]

    f_slow = f_band[f_band < f0][np.argmax(s_band[f_band < f0])]
    f_fast = f_band[f_band > f0][np.argmax(s_band[f_band > f0])]
    omega_q_meas = np.pi * (f_fast - f_slow)
    R_meas = omega_q_meas * (gamma0**1.5) / wp

    R_th = reduction_factor_rectangular(
        beta * u0, u0, b_half, b_half, ax=a_half, ay=a_half
    )
    assert np.isclose(R_meas, R_th, rtol=2.5e-2)
    assert np.isclose(0.5 * (f_fast + f_slow), f0, rtol=3.5e-2)


# ---------------------------------------------------------------------------
# 3. Exact transposed step (dot-product test ~1e-12 & FD gradient test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["3d-pec", "2d-sheet-x", "3d-bloch-complex", "3d-cpml-ramp"],
)
def test_beam_transposed_step_dot_product(case):
    """
    Dot-product identity <U x, y> = <x, U^H y> for the coupled (EM + beam)
    time step across open PEC, 2D (Nz=1), Bloch complex128, and CPML with
    smooth density ramp, including auxiliary-only support so beam state terms
    cannot hide behind field terms.
    """
    if case == "2d-sheet-x":
        grid = wakis.GridFIT3D(
            0.0, 0.12, -0.04, 0.04, -0.005, 0.005, 14, 10, 1, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "periodic"],
            bc_high=["pec", "pec", "periodic"],
            verbose=0,
        )
        beam = LinearizedBeam(
            u0=0.3 * c_light,
            rho0=-1e-3,
            axis="x",
            half_thickness=0.015,
            nu_b=2e8,
            solver=solver,
        )
    elif case == "3d-bloch-complex":
        grid = wakis.GridFIT3D(
            -0.04, 0.04, -0.04, 0.04, 0.0, 0.10, 8, 8, 12, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "bloch"],
            bc_high=["pec", "pec", "bloch"],
            k_bloch=[0.0, 0.0, 35.0],
            dtype=np.complex128,
            verbose=0,
        )
        beam = LinearizedBeam(
            u0=0.4 * c_light,
            rho0=-2e-3,
            axis="z",
            radius=0.02,
            nu_b=1e8,
            solver=solver,
        )
    elif case == "3d-cpml-ramp":
        grid = wakis.GridFIT3D(
            -0.04, 0.04, -0.04, 0.04, 0.0, 0.16, 8, 8, 18, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "cpml"],
            bc_high=["pec", "pec", "cpml"],
            n_pml=4,
            verbose=0,
        )
        beam = LinearizedBeam(
            u0=0.35 * c_light,
            rho0=-1.5e-3,
            axis="z",
            radius=0.02,
            ramp_cells=3,
            solver=solver,
        )
        # Verify that no active beam edges lie inside the CPML slabs [0..3] and [13..17]
        assert beam.k_indices[0] >= 4
        assert beam.k_indices[-1] <= 13
        assert beam.w_axial[-1] < 1.0
    else:
        grid = wakis.GridFIT3D(
            -0.04, 0.04, -0.04, 0.04, 0.0, 0.12, 8, 8, 12, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            verbose=0,
        )
        beam = LinearizedBeam(
            u0=0.35 * c_light,
            rho0=-1.5e-3,
            axis="z",
            radius=0.02,
            nu_b=1.5e8,
            solver=solver,
        )

    solver.one_step()

    stepper = AdjointStepper(solver)
    xp = stepper.xp
    rng = np.random.default_rng(42)
    keys = list(stepper.zero_state().keys())
    beam_keys = [k for k in keys if k.startswith("beam")]
    assert len(beam_keys) == 2

    is_complex = np.iscomplexobj(solver.E.array)

    def random_state(support):
        out = {}
        for k, v in stepper.zero_state().items():
            if k not in support:
                out[k] = xp.zeros_like(v)
            elif is_complex:
                vals = rng.standard_normal(v.shape) + 1j * rng.standard_normal(v.shape)
                out[k] = xp.asarray(vals, dtype=v.dtype)
            else:
                out[k] = xp.asarray(rng.standard_normal(v.shape), dtype=v.dtype)
        return out

    supports = [(keys, keys), (beam_keys, keys), (keys, beam_keys)]
    for x_sup, y_sup in supports:
        x = random_state(x_sup)
        stepper.set_state(x)
        solver.one_step()
        Ux = stepper.get_state()

        y = random_state(y_sup)
        UHy = stepper.step(y)

        lhs = stepper.inner(Ux, y)
        rhs = stepper.inner(x, UHy)
        scale = stepper.inner(
            {k: np.abs(v) for k, v in Ux.items()},
            {k: np.abs(v) for k, v in y.items()},
        ).real
        assert abs(lhs - rhs) < 1e-12 * max(scale, 1e-30), (
            f"case={case}, support=({x_sup}, {y_sup}): |lhs-rhs|/scale = "
            f"{abs(lhs - rhs) / max(scale, 1e-30):.3e}"
        )


def test_beam_adjoint_simulation_finite_difference_gradient():
    """
    Verify that `AdjointSimulation` (both `'store'` and `'checkpoint'`, plus
    automatic fallback from `'reverse'`) produces an exact gradient matching
    central finite differences for a dielectric design region coupled to a
    drifting `LinearizedBeam`.
    """
    f0 = 3.0e9
    Lx, Ly = 0.16, 0.12
    Nx, Ny, Nz = 16, 12, 1
    dx = Lx / Nx

    def make_solver():
        grid = wakis.GridFIT3D(
            -Lx / 2, Lx / 2, -Ly / 2, Ly / 2, -dx / 2, dx / 2, Nx, Ny, Nz, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["pec", "pec", "periodic"],
            bc_high=["pec", "pec", "periodic"],
            verbose=0,
        )
        mask = np.zeros((Nx, Ny, Nz), dtype=bool)
        mask[:, 4:8, :] = True
        beam = LinearizedBeam(
            u0=0.32 * c_light,
            rho0=-5e-3,
            axis="x",
            mask=mask,
            nu_b=1e8,
            solver=solver,
        )
        solver.add_beam(beam)
        return solver

    source = CurrentSource(
        GaussianSource(f0, 1.0e9), component="x", center=(-0.05, 0.0, 0.0)
    )
    monitors = {
        "probe": wakis.DFTMonitor(
            [f0], components=["Ex"], center=(0.05, 0.0, 0.0)
        )
    }
    design = MaterialGrid(
        center=(0.0, 0.025, 0.0),
        size=(0.04, 0.02, 0.0),
        shape=(3, 2, 1),
        eps_min=1.0,
        eps_max=3.5,
    )

    sim_store = AdjointSimulation(
        make_solver, [source], monitors, design, Nt=120, memory="store"
    )
    sim_chk = AdjointSimulation(
        make_solver,
        [source],
        monitors,
        design,
        Nt=120,
        memory="checkpoint",
        checkpoints=2,
        block=25,
    )
    sim_rev = AdjointSimulation(
        make_solver, [source], monitors, design, Nt=120, memory="reverse"
    )

    rho = np.array([[[0.3], [0.7]], [[0.5], [0.2]], [[0.8], [0.4]]])

    def obj_and_cot(out):
        Ex = out["probe"]["Ex"]
        scale = 1e18
        val = scale * float(np.sum(np.abs(Ex) ** 2))
        return val, {"probe": {"Ex": 2.0 * scale * Ex}}

    F_store, G = obj_and_cot(sim_store.forward(rho))
    grad_store = sim_store.vjp(G)

    F_chk, G_chk = obj_and_cot(sim_chk.forward(rho))
    grad_chk = sim_chk.vjp(G_chk)
    assert np.isclose(F_store, F_chk, rtol=1e-14)
    np.testing.assert_allclose(grad_store, grad_chk, rtol=1e-13, atol=1e-15)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim_rev.forward(rho)
        grad_rev = sim_rev.vjp(G)
    assert any("linearized electron beam" in str(w.message) for w in caught)
    np.testing.assert_allclose(grad_store, grad_rev, rtol=1e-13, atol=1e-15)

    h = 1e-4
    for idx in [(0, 0, 0), (1, 1, 0), (2, 0, 0)]:
        step = np.zeros_like(rho)
        step[idx] = h
        Fp, _ = obj_and_cot(sim_store.forward(rho + step, record=False))
        Fm, _ = obj_and_cot(sim_store.forward(rho - step, record=False))
        fd = (Fp - Fm) / (2.0 * h)
        assert np.isclose(grad_store[idx], fd, rtol=1e-5), (
            f"idx={idx}: adjoint={grad_store[idx]:.6e}, fd={fd:.6e}"
        )


# ---------------------------------------------------------------------------
# 4. Stability, Chu kinetic power theorem, and negative-energy slow wave
# ---------------------------------------------------------------------------


def test_chu_poynting_energy_conservation_and_slow_wave_sign():
    """
    1. Von Neumann eigenvalues of the Crank-Nicolson beam + FIT update satisfy
       |lambda| = 1 to < 1e-13 for nu_b = 0 and |lambda| <= 1 for nu_b > 0.
    2. In a closed smooth guide, the discrete Chu kinetic energy + staggered
       leapfrog electromagnetic energy is conserved to < 1e-12 of the
       beam-field energy exchange swing over 1,000 steps.
    3. A pure fast space-charge wave carries positive Chu kinetic power
       (P_k > 0), whereas a pure slow space-charge wave carries negative Chu
       kinetic power (P_k < 0).
    """
    # 1. Von Neumann unitarity check across Courant numbers
    dt = 1e-12
    ds = 1e-3
    u0 = 0.4 * c_light
    wp_rel = 2.0 * np.pi * 2.0e9
    q_perp = 0.25
    for ks in np.linspace(-0.9 * np.pi / ds, 0.9 * np.pi / ds, 25):
        k_tilde = np.sin(ks * ds) / ds
        q_s = 2.0 * c_light * dt * np.sin(0.5 * ks * ds) / ds
        M_lossless = _beam_amplification_matrix(
            dt, q_perp, k_tilde, u0, wp_rel, nu_b=0.0, q_s=q_s
        )
        eigs = np.linalg.eigvals(M_lossless)
        np.testing.assert_allclose(np.abs(eigs), 1.0, atol=1e-12)

        M_damped = _beam_amplification_matrix(
            dt, q_perp, k_tilde, u0, wp_rel, nu_b=5e8, q_s=q_s
        )
        eigs_d = np.linalg.eigvals(M_damped)
        assert np.all(np.abs(eigs_d) <= 1.0 + 1e-12)

    # 2. Discrete Chu-Poynting energy conservation in SolverFIT3D
    Lz = 0.12
    a_half = 0.02
    Nz = 32
    grid = wakis.GridFIT3D(
        -a_half, a_half, -a_half, a_half, 0.0, Lz, 4, 4, Nz, verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        cfln=0.45,
        verbose=0,
    )
    _, gamma0 = relativistic_gamma(u0=u0)
    wp = 2.0 * np.pi * 1.5e9
    beam = LinearizedBeam(u0=u0, omega_p=wp, axis="z", solver=solver)

    k_mode = 2.0 * np.pi / Lz
    beam.v1[:] = (
        1e5 * np.sin(k_mode * beam.s_coords)[:, None] * np.ones((beam.M, beam.N_lines))
    )

    tA_all = np.where(grid.itA.toarray() != 0, 1.0 / grid.itA.toarray(), 0.0)
    dV_vec = grid.L.toarray() * tA_all
    eps_vec = 1.0 / solver.ieps.toarray()
    mu_vec = 1.0 / solver.imu.toarray()

    solver.one_step()
    W_em_hist = []
    W_k_hist = []
    for _ in range(1000):
        E_n = solver.E.toarray().copy()
        solver.one_step()
        E_np1 = solver.E.toarray()
        H_nh = solver.H.toarray()
        W_em = 0.5 * float(
            np.sum(dV_vec * eps_vec * E_n * E_np1)
            + np.sum(dV_vec * mu_vec * H_nh**2)
        )
        W_k = beam.kinetic_energy()
        W_em_hist.append(W_em)
        W_k_hist.append(W_k)

    W_em_arr = np.array(W_em_hist)
    W_k_arr = np.array(W_k_hist)
    W_tot = W_em_arr + W_k_arr
    swing = np.ptp(W_em_arr)
    assert swing > 0.0
    rel_drift = np.ptp(W_tot) / swing
    assert rel_drift < 1e-12, f"Relative Chu-Poynting drift {rel_drift:.3e} >= 1e-12"

    # 3. Chu kinetic power sign of pure fast (+) vs pure slow (-) space-charge waves
    k_scw = 4.0 * np.pi / Lz
    omega_q_1d = wp / (gamma0**1.5)
    omega_fast = k_scw * u0 + omega_q_1d
    omega_slow = k_scw * u0 - omega_q_1d
    assert omega_slow > 0.0

    v_amp = (
        1e4 * np.cos(k_scw * beam.s_coords)[:, None] * np.ones((beam.M, beam.N_lines))
    )
    rho0_val = beam.rho0_peak

    # Fast wave: rho1 = (k_scw * rho0 / (+omega_q)) v1
    beam.v1[:] = v_amp
    beam.rho1[:] = (k_scw * rho0_val / omega_q_1d) * v_amp
    P_k_fast = np.mean(beam.kinetic_power_profile())
    assert P_k_fast > 0.0, f"Fast wave must have P_k > 0, got {P_k_fast}"

    # Slow wave: rho1 = (k_scw * rho0 / (-omega_q)) v1
    beam.v1[:] = v_amp
    beam.rho1[:] = (-k_scw * rho0_val / omega_q_1d) * v_amp
    P_k_slow = np.mean(beam.kinetic_power_profile())
    assert P_k_slow < 0.0, f"Slow wave must have P_k < 0, got {P_k_slow}"
    assert np.isclose(
        P_k_fast / abs(P_k_slow), omega_fast / omega_slow, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# 5. Mutation tests & u0 = 0 stationary cold-plasma equivalence
# ---------------------------------------------------------------------------


def test_mutations_fail_validation_and_u0_zero_matches_cold_plasma():
    """
    1. Mutation: wrong relativistic gamma power (`gamma_power=1.0` instead of
       `3.0` at beta0 = 0.6, gamma0 = 1.25) changes omega_q by gamma0 = 1.25
       and must fail the relativistic dispersion check.
    2. Mutation: flipped coupling sign (`coupling_sign=-1.0`) makes omega_q^2 < 0
       (unphysical exponential growth in a smooth pipe) and must fail the
       stability bound.
    3. Limit u0 = 0: must fail the drifting-beam Doppler split (omega_+ != beta u0 + omega_q)
       AND must match stationary `wakis.plasma.cold_plasma` to machine precision (~1e-13).
    """
    # 1. Wrong gamma_power mutation at relativistic speed u0 = 0.6 c (gamma0 = 1.25)
    u0_rel = 0.6 * c_light
    _, gamma0_rel = relativistic_gamma(u0=u0_rel)
    assert np.isclose(gamma0_rel, 1.25)

    b_half, a_half = 0.01, 0.02
    beta = 2.0 * np.pi / 0.08
    wp = 2.0 * np.pi * 1.6e9
    R_th = reduction_factor_planar(beta * u0_rel, u0_rel, b_half, a=a_half)
    omega_fast_th, omega_slow_th = space_charge_wave_dispersion(
        beta, u0_rel, wp, R=R_th
    )
    omega_q_th = 0.5 * (omega_fast_th - omega_slow_th)

    _, _, omega_q_true, _ = _measure_2d_sheet_dispersion(
        b_half=b_half,
        a_half=a_half,
        beta=beta,
        u0=u0_rel,
        wp=wp,
        gamma_power=3.0,
    )
    _, _, omega_q_mut, _ = _measure_2d_sheet_dispersion(
        b_half=b_half,
        a_half=a_half,
        beta=beta,
        u0=u0_rel,
        wp=wp,
        gamma_power=1.0,
    )
    assert np.isclose(omega_q_true, omega_q_th, rtol=2.5e-2)
    # Mutated gamma_power=1.0 overpredicts omega_q by gamma0^( (3-1)/2 ) = 1.25 (>20% error)
    assert abs(omega_q_mut - omega_q_th) / omega_q_th > 0.20

    # 2. Flipped coupling_sign=-1.0 causes exponential blow-up in a smooth guide
    grid = wakis.GridFIT3D(
        -0.01, 0.01, -0.01, 0.01, 0.0, 0.08, 4, 4, 24, verbose=0
    )
    solver_mut = wakis.SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "periodic"],
        bc_high=["periodic", "periodic", "periodic"],
        verbose=0,
    )
    beam_mut = LinearizedBeam(
        u0=0.35 * c_light,
        omega_p=wp,
        axis="z",
        coupling_sign=-1.0,
        solver=solver_mut,
    )
    beam_mut.v1[:] = (
        1e2
        * np.sin(2.0 * np.pi * beam_mut.s_coords / 0.08)[:, None]
        * np.ones((beam_mut.M, beam_mut.N_lines))
    )
    solver_mut.one_step()
    v_init = float(np.max(np.abs(beam_mut.v1)))
    for _ in range(400):
        solver_mut.one_step()
    v_final = float(np.max(np.abs(beam_mut.v1)))
    assert v_final > 50.0 * v_init, "Flipped coupling sign must trigger instability"

    # 3. Stationary limit u0 = 0 matches `wakis.plasma.cold_plasma` to machine precision
    grid_p1 = wakis.GridFIT3D(
        -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, 8, 8, 8, verbose=0
    )
    grid_p2 = wakis.GridFIT3D(
        -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, 8, 8, 8, verbose=0
    )
    solver_beam0 = wakis.SolverFIT3D(
        grid_p1, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
    )
    solver_drude = wakis.SolverFIT3D(
        grid_p2, bc_low=["pec"] * 3, bc_high=["pec"] * 3, verbose=0
    )

    f_p = 2.5e9
    wp_stat = 2.0 * np.pi * f_p
    nu_stat = 3.0e8

    for ax in ("x", "y", "z"):
        LinearizedBeam(
            u0=0.0,
            omega_p=wp_stat,
            axis=ax,
            nu_b=nu_stat,
            solver=solver_beam0,
        )

    drude_mat = cold_plasma(omega_p=wp_stat, collision_rate=nu_stat)
    n_dof = 3 * solver_drude.N
    solver_drude._add_dispersive_edges(
        drude_mat, np.arange(n_dof), np.ones(n_dof, dtype=float)
    )

    rng = np.random.default_rng(7)
    e_init = rng.standard_normal((8, 8, 8))
    for s in (solver_beam0, solver_drude):
        s.E[2:6, 2:6, 2:6, "z"] = e_init[2:6, 2:6, 2:6]

    for _ in range(150):
        solver_beam0.one_step()
        solver_drude.one_step()

    E_beam0 = solver_beam0.E.toarray()
    E_drude = solver_drude.E.toarray()
    np.testing.assert_allclose(
        E_beam0,
        E_drude,
        rtol=1e-13,
        atol=1e-13 * np.max(np.abs(E_drude)),
    )
