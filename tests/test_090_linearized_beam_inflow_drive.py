# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Tests for the public inflow drive (`inflow_rho1`, `inflow_v1`) in `LinearizedBeam`.

Verifies:
- Bitwise identity of `LinearizedBeam._pre_step` when `inflow_rho1=None` and
  `inflow_v1=None` (default).
- (a) In a drift region with negligible space charge (`rho0 -> tiny`), a
  sinusoidal entrance density perturbation `inflow_rho1(t)` advects at `u0`
  with axial phase slope matching `beta_e = omega / u0` within the scheme's
  dispersion bound `1 / cos(beta_e * dz) - 1`.
- (b) The entrance current modulation amplitude `|I1(z_0)|` equals the drive
  amplitude `|rho1_amp| * area * u0` times the predicted midpoint-sampling
  factor `1 / cos(omega * dt / 2)`, with a residual below half that correction
  (so the uncorrected ratio 1 fails).
- (c) A code-path mutation that drops `inflow_rho1` fails the amplitude and
  advection checks.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c as c_light

from wakis import GridFIT3D, SolverFIT3D
from wakis.beam_fluid import LinearizedBeam, relativistic_gamma


def _build_drift_solver_and_beam(
    inflow_rho1=None,
    inflow_v1=None,
    gap_modulation=None,
    rho0: float | None = -1.0e-18,
    current: float | None = None,
    radius: float | None = None,
    half_thickness: float | None = 0.8e-3,
    u0_sign: float = 1.0,
    ramp_cells: int = 12,
    nx: int = 10,
    ny: int = 1,
    ly: float = 1.0e-3,
    bc_y: str = "periodic",
):
    """Create a 3D drift solver and attached `LinearizedBeam`."""
    a = 2.0e-3
    lz = 0.060
    nz = 140
    freq = 6.5e9
    u0_mag, gamma0 = relativistic_gamma(voltage=10.0e3)
    u0 = float(u0_sign) * u0_mag

    dx = 2.0 * a / nx
    dy = ly / ny
    dz = lz / nz
    dt_cfl = 0.5 / (c_light * np.sqrt(1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2))
    steps_per_period = int(np.ceil(1.0 / (freq * dt_cfl)))
    dt = 1.0 / (freq * steps_per_period)

    grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, ny, nz, verbose=0)
    solver = SolverFIT3D(
        grid,
        dt=dt,
        bc_low=["pec", bc_y, "pec"],
        bc_high=["pec", bc_y, "pec"],
        use_gpu=False,
        verbose=0,
    )
    beam = LinearizedBeam(
        u0=u0,
        rho0=rho0,
        current=current,
        gamma0=gamma0,
        axis="z",
        radius=radius,
        half_thickness=half_thickness,
        ramp_cells=ramp_cells,
        gap_modulation=gap_modulation,
        inflow_rho1=inflow_rho1,
        inflow_v1=inflow_v1,
        solver=solver,
    )
    return solver, beam, freq, steps_per_period


def test_default_none_is_bitwise_unchanged():
    """With `inflow_rho1=None` and `inflow_v1=None`, `_pre_step` is bitwise unchanged."""
    omega = 2.0 * np.pi * 6.5e9

    def gap_drive(s_coords: np.ndarray, t: float) -> np.ndarray:
        return 100.0 * np.exp(-(((s_coords - 0.015) / 0.003) ** 2)) * np.sin(omega * t)

    solver_a, beam_a, _, _ = _build_drift_solver_and_beam(
        inflow_rho1=None,
        inflow_v1=None,
        gap_modulation=gap_drive,
        rho0=-1.6e-3,
    )
    solver_b, beam_b, _, _ = _build_drift_solver_and_beam(
        gap_modulation=gap_drive,
        rho0=-1.6e-3,
    )

    # Replace beam_b._pre_step with the exact pre-change reference sequence
    def _reference_pre_step(s, b=beam_b):
        xp = b.xp
        e_flat = s.E.array[b.flat_idx]
        e_host = np.asarray(e_flat.get() if hasattr(e_flat, "get") else e_flat)
        e_b = e_host.reshape(b.M, b.N_lines)
        if b.gap_modulation is not None:
            t = b._step_count * float(s.dt)
            e_mod = np.asarray(b.gap_modulation(b.s_coords, t))
            if e_mod.ndim == 1:
                e_mod = e_mod[:, None]
            e_b = e_b + e_mod
        b._step_count += 1
        v_old = b.v1
        rho_old = b.rho1
        b.v1_prev = v_old
        b.rho1_prev = rho_old
        rhs_v = b.Mm_v @ v_old + b._alpha_E * e_b
        v_new = b._lu_v.solve(rhs_v)
        v_bar = 0.5 * (v_new + v_old)
        rhs_r = b.Mm_r @ rho_old + b._beta_v * (b.Ds @ v_bar)
        rho_new = b._lu_r.solve(rhs_r)
        j_new = b._c_v * v_new + b._c_rho * rho_new
        b.v1 = v_new
        b.rho1 = rho_new
        b.J1 = j_new
        j_dev = xp.asarray(j_new.ravel(), dtype=b.dtype)
        s.J.array[b.flat_idx] += j_dev - b.J_old
        b.J_old = j_dev

    beam_b._pre_step = _reference_pre_step

    for _ in range(120):
        solver_a.one_step()
        solver_b.one_step()

    assert np.array_equal(beam_a.v1, beam_b.v1)
    assert np.array_equal(beam_a.rho1, beam_b.rho1)
    assert np.array_equal(beam_a.J1, beam_b.J1)
    assert np.array_equal(solver_a.E.array, solver_b.E.array)
    assert np.array_equal(solver_a.H.array, solver_b.H.array)


def _run_inflow_advection(
    inflow_rho1,
    inflow_v1=None,
    rho0: float | None = -1.0e-18,
    current: float | None = None,
    radius: float | None = None,
    half_thickness: float | None = 0.8e-3,
    u0_sign: float = 1.0,
    nx: int = 10,
    ny: int = 1,
    ly: float = 1.0e-3,
    bc_y: str = "periodic",
) -> dict[str, np.ndarray | float]:
    """Run the drift tunnel with `inflow_rho1` / `inflow_v1` and compute the 1-omega DFT."""
    solver, beam, freq, steps_per_period = _build_drift_solver_and_beam(
        inflow_rho1=inflow_rho1,
        inflow_v1=inflow_v1,
        rho0=rho0,
        current=current,
        radius=radius,
        half_thickness=half_thickness,
        u0_sign=u0_sign,
        nx=nx,
        ny=ny,
        ly=ly,
        bc_y=bc_y,
    )
    dt = float(solver.dt)
    omega = 2.0 * np.pi * freq
    lz = float(solver.grid.z[-1] - solver.grid.z[0])
    n_settle = int(np.ceil(1.25 * lz / (abs(beam.u0) * dt)))
    n_dft = 4 * steps_per_period

    iz_hist = np.zeros((n_dft, beam.M), dtype=np.float64)
    v1_0_hist = np.zeros(n_dft, dtype=np.float64)
    t_hist = np.zeros(n_dft, dtype=np.float64)
    k_in = 0 if beam.u0 >= 0.0 else beam.M - 1
    for step in range(n_settle + n_dft):
        solver.one_step()
        if step >= n_settle:
            iz_hist[step - n_settle] = beam.axial_current()
            v1_0_hist[step - n_settle] = float(np.mean(beam.v1[k_in, :]))
            t_hist[step - n_settle] = (step + 0.5) * dt

    iz_zm = iz_hist - np.mean(iz_hist, axis=0, keepdims=True)
    i1_phasor = (2.0 / n_dft) * np.sum(
        iz_zm * np.exp(-1j * omega * t_hist)[:, None], axis=0
    )
    v1_0_phasor = complex(
        (2.0 / n_dft)
        * np.sum((v1_0_hist - np.mean(v1_0_hist)) * np.exp(-1j * omega * t_hist))
    )
    return {
        "beam": beam,
        "z_coords": beam.s_coords.copy(),
        "i1_phasor": i1_phasor,
        "v1_0_phasor": v1_0_phasor,
        "dt": dt,
        "dz": float(beam.ds_line[0]),
        "omega": omega,
        "beta_e": omega / abs(beam.u0),
    }


def test_inflow_rho1_advection_phase_and_entrance_amplitude():
    """Verify (a) advection phase vs z against beta_e, (b) entrance amplitude, and outflow ramp."""
    freq = 6.5e9
    omega = 2.0 * np.pi * freq
    t_rise = 2.0 / freq
    rho1_amp = -2.5e-5  # [C/m^3]

    def drive_rho1(t: float) -> float:
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        return rho1_amp * ramp * np.sin(omega * t)

    out = _run_inflow_advection(inflow_rho1=drive_rho1, rho0=-1.0e-18)
    beam = out["beam"]
    z_coords = out["z_coords"]
    i1_phasor = out["i1_phasor"]
    dt = out["dt"]
    dz = out["dz"]
    beta_e = out["beta_e"]

    # Active region clear of the 12-cell outflow ramp
    n_active = beam.M - beam.n_ramp - 4
    z_act = z_coords[:n_active]
    i1_act = i1_phasor[:n_active]

    # (a) Phase vs z against beta_e = omega / u0 within scheme dispersion 1/cos(beta_e*dz) - 1
    unwrapped_phase = np.unwrap(np.angle(i1_act))
    slope, _ = np.polyfit(z_act, unwrapped_phase, 1)
    k_measured = float(-slope)
    tol_disp = float(1.0 / np.cos(beta_e * dz) - 1.0)
    rel_k_err = abs(k_measured - beta_e) / beta_e
    assert rel_k_err <= tol_disp, (
        f"Measured advection wavenumber {k_measured:.4f} rad/m differs from "
        f"beta_e={beta_e:.4f} rad/m by {rel_k_err:.4%} > {tol_disp:.4%}"
    )

    # Pointwise phase residual after removing constant entrance phase at z_act[0]
    phase_rel = -(unwrapped_phase - unwrapped_phase[0])
    phase_th = beta_e * (z_act - z_act[0])
    max_phase_err = float(np.max(np.abs(phase_rel - phase_th)))
    max_phase_bound = tol_disp * beta_e * float(z_act[-1] - z_act[0])
    assert max_phase_err <= max_phase_bound

    # (b) Entrance current amplitude matches |rho1_amp| * area * u0 within CN time-centring bound
    i1_drive = abs(rho1_amp) * beam.area * beam.u0
    amp_ratio_0 = float(np.abs(i1_phasor[0]) / i1_drive)
    # Evaluating the drive at the step midpoint t^n instead of averaging the two
    # half-step levels raises the driven amplitude by 1 / cos(omega * dt / 2).
    # Test that prediction against the alternative "no correction" (ratio 1): the
    # threshold sits halfway between the two hypotheses, so a ratio of 1 fails.
    cos_half = float(np.cos(0.5 * omega * dt))
    tol_amp = 0.5 * (1.0 / cos_half - 1.0)
    assert abs(amp_ratio_0 * cos_half - 1.0) <= tol_amp, (
        f"Entrance amplitude ratio {amp_ratio_0:.6f} outside 1/cos(omega*dt/2) +- {tol_amp:.6e}"
    )
    # the threshold discriminates: the uncorrected ratio 1 is rejected
    assert abs(cos_half - 1.0) > tol_amp

    # Verify that the axial outflow ramp w_axial tapers the current at the exit edge
    assert float(np.abs(i1_phasor[-1])) < 0.02 * i1_drive


def test_inflow_v1_advection_and_amplitude():
    """Verify optional `inflow_v1` velocity drive advects at `u0` with expected current `|rho0| * area * v1_amp`."""
    freq = 6.5e9
    omega = 2.0 * np.pi * freq
    t_rise = 2.0 / freq
    v1_amp = 1.0e4  # [m/s]
    rho0_tiny = -1.0e-12  # [C/m^3], negligible space-charge wavenumber

    def drive_v1(t: float) -> float:
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        return v1_amp * ramp * np.sin(omega * t)

    out = _run_inflow_advection(inflow_rho1=None, inflow_v1=drive_v1, rho0=rho0_tiny)
    beam = out["beam"]
    z_coords = out["z_coords"]
    i1_phasor = out["i1_phasor"]
    dt = out["dt"]
    dz = out["dz"]
    beta_e = out["beta_e"]

    n_active = beam.M - beam.n_ramp - 4
    unwrapped_phase = np.unwrap(np.angle(i1_phasor[:n_active]))
    slope, _ = np.polyfit(z_coords[:n_active], unwrapped_phase, 1)
    k_measured = float(-slope)
    tol_disp = float(1.0 / np.cos(beta_e * dz) - 1.0)
    assert abs(k_measured - beta_e) / beta_e <= tol_disp

    # Entrance velocity perturbation |v1(z_0)| matches v1_amp within CN time-centring bound
    cos_half = float(np.cos(0.5 * omega * dt))
    tol_amp = 0.5 * (1.0 / cos_half - 1.0)
    v1_ratio_0 = float(np.abs(out["v1_0_phasor"]) / v1_amp)
    assert abs(v1_ratio_0 * cos_half - 1.0) <= tol_amp

    # At the first edge k=0 (distance dz from the ghost cell k=-1), continuity bunching
    # adds u0 * rho1(z_0) = -i * (beta_e * dz) * rho0 * v1(z_0), scaling |I1(z_0)| by sqrt(1 + (beta_e*dz)**2)
    i1_drive = (
        abs(rho0_tiny) * beam.area * v1_amp * float(np.sqrt(1.0 + (beta_e * dz) ** 2))
    )
    amp_ratio_0 = float(np.abs(i1_phasor[0]) / i1_drive)
    assert abs(amp_ratio_0 - 1.0) <= 0.5 * (beta_e * dz) ** 2


def test_mutation_dropping_inflow_drive_fails(monkeypatch):
    """(c) A mutation that drops `inflow_rho1` must fail the entrance amplitude check."""
    freq = 6.5e9
    omega = 2.0 * np.pi * freq
    rho1_amp = -2.5e-5

    orig_init = LinearizedBeam.__init__

    def _mutated_init(self, *args, **kwargs):
        kwargs = dict(kwargs, inflow_rho1=None)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(LinearizedBeam, "__init__", _mutated_init)

    out = _run_inflow_advection(
        inflow_rho1=lambda t: rho1_amp * np.sin(omega * t),
        rho0=-1.0e-18,
    )
    beam = out["beam"]
    i1_phasor = out["i1_phasor"]
    dt = out["dt"]
    i1_drive = abs(rho1_amp) * beam.area * beam.u0
    cos_half = float(np.cos(0.5 * omega * dt))
    tol_amp = 0.5 * (1.0 / cos_half - 1.0)

    with pytest.raises(AssertionError, match="Entrance amplitude") as exc_drop:
        amp_ratio_0 = float(np.abs(i1_phasor[0]) / i1_drive)
        assert abs(amp_ratio_0 * cos_half - 1.0) <= tol_amp, (
            f"Entrance amplitude ratio {amp_ratio_0:.6f} outside 1/cos(omega*dt/2) +- {tol_amp:.6e}"
        )
    print(f"T90 drop mutation failed as expected: {exc_drop.value}")


def test_mutation_inflow_ghost_coefficient_and_w_perp_eff_fail(monkeypatch):
    """Scaling the library's ghost-term coefficient by 2 or dropping `_w_perp_eff` must fail the amplitude check."""
    freq = 6.5e9
    omega = 2.0 * np.pi * freq
    t_rise = 2.0 / freq
    rho1_amp = -2.5e-5

    def drive_rho1(t: float) -> float:
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        return rho1_amp * ramp * np.sin(omega * t)

    # Mutation 1: scale the library's inflow ghost-term coefficient (dt * |u0| / (2 * ds)) by 2x
    orig_initialize = LinearizedBeam._initialize

    def _double_ghost_initialize(self, solver):
        orig_initialize(self, solver)
        self._inflow_ghost_coeff = 2.0 * self._inflow_ghost_coeff

    monkeypatch.setattr(LinearizedBeam, "_initialize", _double_ghost_initialize)
    out_2x = _run_inflow_advection(inflow_rho1=drive_rho1, rho0=-1.0e-18)
    beam_2x = out_2x["beam"]
    dt = out_2x["dt"]
    i1_drive = abs(rho1_amp) * beam_2x.area * beam_2x.u0
    cos_half = float(np.cos(0.5 * omega * dt))
    tol_amp = 0.5 * (1.0 / cos_half - 1.0)

    with pytest.raises(AssertionError, match="Entrance amplitude") as exc_ghost:
        amp_ratio_0 = float(np.abs(out_2x["i1_phasor"][0]) / i1_drive)
        assert abs(amp_ratio_0 * cos_half - 1.0) <= tol_amp, (
            f"Entrance amplitude ratio {amp_ratio_0:.6f} outside 1/cos(omega*dt/2) +- {tol_amp:.6e}"
        )
    print(f"T90 ghost-coeff 2x mutation failed as expected: {exc_ghost.value}")

    # Mutation 2: drop _w_perp_eff (set to ones across all lines including boundary cells)
    def _drop_w_perp_initialize(self, solver):
        orig_initialize(self, solver)
        self._w_perp_eff = np.ones_like(self._w_perp_eff)
        self._c_rho = self.u0 * self._w_perp_eff

    monkeypatch.setattr(LinearizedBeam, "_initialize", _drop_w_perp_initialize)
    out_nw = _run_inflow_advection(inflow_rho1=drive_rho1, rho0=-1.0e-18)
    beam_nw = out_nw["beam"]
    i1_drive_nw = abs(rho1_amp) * beam_nw.area * beam_nw.u0

    with pytest.raises(AssertionError, match="Entrance amplitude") as exc_wperp:
        amp_ratio_0 = float(np.abs(out_nw["i1_phasor"][0]) / i1_drive_nw)
        assert abs(amp_ratio_0 * cos_half - 1.0) <= tol_amp, (
            f"Entrance amplitude ratio {amp_ratio_0:.6f} outside 1/cos(omega*dt/2) +- {tol_amp:.6e}"
        )
    print(f"T90 drop _w_perp_eff mutation failed as expected: {exc_wperp.value}")


def test_inflow_rho1_negative_u0_sign():
    """Verify `u0 < 0` injects at the upstream `s_max` boundary `k = M - 1` and tapers at `k = 0`."""
    freq = 6.5e9
    omega = 2.0 * np.pi * freq
    t_rise = 2.0 / freq
    rho1_amp = -2.5e-5

    def drive_rho1(t: float) -> float:
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        return rho1_amp * ramp * np.sin(omega * t)

    out = _run_inflow_advection(inflow_rho1=drive_rho1, rho0=-1.0e-18, u0_sign=-1.0)
    beam = out["beam"]
    z_coords = out["z_coords"]
    i1_phasor = out["i1_phasor"]
    dt = out["dt"]
    dz = out["dz"]
    beta_e = out["beta_e"]

    assert beam.u0 < 0.0
    # Active region clear of the downstream outflow ramp at k = 0 .. n_ramp - 1
    i_start = beam.n_ramp + 4
    z_act = z_coords[i_start:]
    i1_act = i1_phasor[i_start:]

    # For u0 < 0, waves propagate in -z so arg(I1(z)) increases with +z at slope +|beta_e|
    unwrapped_phase = np.unwrap(np.angle(i1_act))
    slope, _ = np.polyfit(z_act, unwrapped_phase, 1)
    k_measured = float(slope)
    tol_disp = float(1.0 / np.cos(beta_e * dz) - 1.0)
    assert abs(k_measured - beta_e) / beta_e <= tol_disp

    # Entrance current magnitude at k = M - 1 matches |rho1_amp| * area * |u0|
    i1_drive = abs(rho1_amp) * beam.area * abs(beam.u0)
    amp_ratio_in = float(np.abs(i1_phasor[-1]) / i1_drive)
    cos_half = float(np.cos(0.5 * omega * dt))
    tol_amp = 0.5 * (1.0 / cos_half - 1.0)
    assert abs(amp_ratio_in * cos_half - 1.0) <= tol_amp

    # Downstream exit edge k = 0 is tapered by the outflow ramp
    assert float(np.abs(i1_phasor[0])) < 0.02 * i1_drive


def test_inflow_rho1_with_current_spec():
    """Verify the `current=` branch on a 3D circular beam where `area_eff != area` (`scale != 1`)."""
    freq = 6.5e9
    omega = 2.0 * np.pi * freq
    t_rise = 2.0 / freq
    rho1_amp = -2.5e-5
    b = 0.8e-3

    def drive_rho1(t: float) -> float:
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        return rho1_amp * ramp * np.sin(omega * t)

    # 3D square tunnel [-2 mm, 2 mm]^2 (nx=8, ny=8) with circular beam radius b = 0.8 mm:
    # analytic area_eff = pi * b^2 differs from the discrete 8x8 subpixel area beam.area
    out = _run_inflow_advection(
        inflow_rho1=drive_rho1,
        rho0=None,
        current=1.0e-12,
        radius=b,
        half_thickness=None,
        nx=8,
        ny=8,
        ly=4.0e-3,
        bc_y="pec",
    )
    beam = out["beam"]
    z_coords = out["z_coords"]
    i1_phasor = out["i1_phasor"]
    dt = out["dt"]
    dz = out["dz"]
    beta_e = out["beta_e"]
    cos_half = float(np.cos(0.5 * omega * dt))
    tol_amp = 0.5 * (1.0 / cos_half - 1.0)

    # Analytic area_eff = pi * b^2 differs from discrete beam.area (scale != 1 by > 10 * tol_amp)
    expected_area_eff = float(np.pi * b**2)
    assert np.isclose(beam.area_eff, expected_area_eff, rtol=1e-12, atol=0.0)
    scale = beam.area_eff / beam.area
    assert abs(scale - 1.0) > 10.0 * tol_amp

    # _w_perp_eff cross-sectional area integral equals analytic beam.area_eff (not unscaled beam.area)
    assert np.isclose(
        float(np.sum(beam.dA_perp[0] * beam._w_perp_eff[0])),
        beam.area_eff,
        rtol=1e-12,
        atol=0.0,
    )

    n_active = beam.M - beam.n_ramp - 4
    unwrapped_phase = np.unwrap(np.angle(i1_phasor[:n_active]))
    slope, _ = np.polyfit(z_coords[:n_active], unwrapped_phase, 1)
    k_measured = float(-slope)
    tol_disp = float(1.0 / np.cos(beta_e * dz) - 1.0)
    assert abs(k_measured - beta_e) / beta_e <= tol_disp

    i1_drive = abs(rho1_amp) * beam.area_eff * beam.u0
    amp_ratio_0 = float(np.abs(i1_phasor[0]) / i1_drive)
    print(
        f"T90 current= 3D circle: area={beam.area:.6e} m^2, area_eff={beam.area_eff:.6e} m^2, "
        f"scale={scale:.6f} (scale-1={scale - 1.0:+.4%}), amp_ratio_0={amp_ratio_0:.6f} "
        f"(tol_amp={tol_amp:.6e})"
    )
    assert abs(amp_ratio_0 * cos_half - 1.0) <= tol_amp


def test_inflow_rejected_without_open_drifting_axis():
    """The inflow drive raises instead of being silently ignored when the beam
    axis is periodic or the beam does not drift (no upstream ghost edge)."""
    drive = lambda t: 0.0  # noqa: E731
    with pytest.raises(ValueError, match="drifting beam"):
        LinearizedBeam(u0=0.0, rho0=-1.0e-18, gamma0=1.0, axis="z", inflow_rho1=drive)

    grid = GridFIT3D(-1e-3, 1e-3, -1e-3, 1e-3, 0.0, 0.01, 4, 4, 20, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["pec", "pec", "periodic"],
        bc_high=["pec", "pec", "periodic"],
        use_gpu=False,
        verbose=0,
    )
    u0, gamma0 = relativistic_gamma(voltage=10.0e3)
    with pytest.raises(ValueError, match="periodic"):
        LinearizedBeam(
            u0=u0,
            rho0=-1.0e-18,
            gamma0=gamma0,
            axis="z",
            inflow_v1=drive,
            solver=solver,
        )
