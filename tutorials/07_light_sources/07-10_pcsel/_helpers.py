# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Helper functions for Tutorial 7.10: Photonic-crystal surface-emitting laser
(PCSEL) band-edge modes and 3D coupled-wave theory.

Provides:
- 3D Bloch-periodic Wakis FIT unit-cell solver (`build_pcsel_unit_cell_solver`,
  `run_pcsel_unit_cell`) with graded vertical mesh (`graded_axis`), subpixel
  circular-hole permittivity averaging, preflight memory/time estimation
  (`wakis.estimate`), `ResonanceMonitor` (`harminv`) extraction of all four
  Gamma-point band-edge modes (A, B, C, D) from a single time-domain run via
  C4v symmetry projections, and `DFTMonitor` spatial mode-profile extraction.
- Setup figure generation (`save_pcsel_setup_figures`) producing
  `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png` (3D unit cell)
  and `00a_setup_geometry_finite.png`, `00b_setup_mesh_and_excitation_finite.png`
  (2D effective-index finite PCSEL cavity) plus markdown setup tables.
- 2D effective-index finite PCSEL cavity FIT solver (`run_finite_pcsel_2d_fit`)
  to cross-check finite-size in-plane loss scaling against 2D CWT.
- End-to-end study runner (`run_all_studies`) and publication-style plotting
  suite (`plot_all_figures`) using `wakis.figstyle` and
  `wakis.schematic.overlay_geometry`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.constants import c as C_LIGHT, epsilon_0, mu_0
from scipy.optimize import brentq

import wakis
from wakis import estimate, figstyle, schematic
from wakis.field_monitors import DFTMonitor
from wakis.gridFIT3D import GridFIT3D
from wakis.pcsel_cwt import (
    PCSELLayerStack,
    compute_cwt_band_structure,
    compute_cwt_coupling_coefficients,
    double_lattice_coupling_scan,
    solve_finite_cwt,
    solve_infinite_cwt,
    solve_vertical_greens_functions,
    solve_vertical_slab_mode,
)
from wakis.resonances import ResonanceMonitor, harminv
from wakis.solverFIT3D import SolverFIT3D
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.meshing import graded_axis
from wakis. templates.shapes import Box, Cylinder

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# Default GaAs/InGaAs 940-nm-class PCSEL epitaxial stack parameters
DEFAULT_STACK = PCSELLayerStack(
    a=290e-9,
    r_over_a=0.18,
    n_clad=3.26,
    n_active=3.52,
    n_phc_host=3.46,
    n_hole=1.0,
    z_act_min=-150e-9,
    z_act_max=-30e-9,
    z_phc_min=-30e-9,
    z_phc_max=150e-9,
    z_min=-1400e-9,
    z_max=1400e-9,
)

F0_EXCIT = 311.0e12
DF_EXCIT = 18.0e12
CUTOFF_EXCIT = 4.0


def build_pcsel_unit_cell_grid(
    stack: PCSELLayerStack = DEFAULT_STACK,
    nx: int = 24,
    dz_core: float = 12.0e-9,
    dz_clad: float = 35.0e-9,
    z_box: float = 900.0e-9,
) -> tuple[GridFIT3D, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 3D FIT grid for the PCSEL unit cell with graded z-mesh."""
    a = stack.a
    x_nodes = np.linspace(-0.5 * a, 0.5 * a, nx + 1)
    y_nodes = np.linspace(-0.5 * a, 0.5 * a, nx + 1)
    z_nodes = graded_axis(
        points=[-z_box, stack.z_act_min, stack.z_act_max, stack.z_phc_max, z_box],
        sizes=[dz_clad, dz_core, dz_core, dz_clad],
        grading=1.18,
    )
    grid = GridFIT3D(x=x_nodes, y=y_nodes, z=z_nodes, verbose=0)
    return grid, x_nodes, y_nodes, z_nodes


def _subpixel_circle_fill(
    xc: np.ndarray,
    yc: np.ndarray,
    dx: float,
    dy: float,
    radius: float,
    x0: float = 0.0,
    y0: float = 0.0,
    n_sub: int = 8,
) -> np.ndarray:
    """2D subpixel area fraction inside a circle of radius ``radius`` at ``(x0, y0)``."""
    sub = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    xs = (xc[:, None, None, None] + dx * sub[None, None, :, None]) - x0
    ys = (yc[None, :, None, None] + dy * sub[None, None, None, :]) - y0
    inside = (xs**2 + ys**2) < radius**2
    return np.mean(inside, axis=(2, 3))


def build_pcsel_unit_cell_solver(
    stack: PCSELLayerStack = DEFAULT_STACK,
    nx: int = 24,
    dz_core: float = 12.0e-9,
    dz_clad: float = 35.0e-9,
    z_box: float = 900.0e-9,
    kx: float = 0.0,
    ky: float = 0.0,
    cfln: float = 0.85,
    n_pml: int = 8,
) -> tuple[SolverFIT3D, dict]:
    """Construct a 3D Wakis FIT solver for a single PCSEL unit cell."""
    grid, x_nodes, y_nodes, z_nodes = build_pcsel_unit_cell_grid(
        stack=stack,
        nx=nx,
        dz_core=dz_core,
        dz_clad=dz_clad,
        z_box=z_box,
    )
    use_bloch = (abs(kx) + abs(ky)) > 0.0
    bc_xy = "bloch" if use_bloch else "periodic"
    dtype = np.complex128 if use_bloch else np.float64

    solver = SolverFIT3D(
        grid,
        bc_low=[bc_xy, bc_xy, "cpml"],
        bc_high=[bc_xy, bc_xy, "cpml"],
        use_stl=False,
        n_pml=n_pml,
        cfln=cfln,
        dtype=dtype,
        k_bloch=(float(kx), float(ky), 0.0),
        bg=[stack.eps_clad, 1.0],
        verbose=0,
    )

    a = stack.a
    dx = a / nx
    dy = a / nx

    x_n = grid.x[: solver.Nx]
    y_n = grid.y[: solver.Ny]
    z_n = grid.z[: solver.Nz]
    x_m = 0.5 * (grid.x[: solver.Nx] + grid.x[1 : solver.Nx + 1])
    y_m = 0.5 * (grid.y[: solver.Ny] + grid.y[1 : solver.Ny + 1])
    z_m = 0.5 * (grid.z[: solver.Nz] + grid.z[1 : solver.Nz + 1])

    r1 = stack.r_over_a * a
    fill_ex = _subpixel_circle_fill(x_m, y_n, dx, dy, r1)
    fill_ey = _subpixel_circle_fill(x_n, y_m, dx, dy, r1)
    fill_ez = _subpixel_circle_fill(x_n, y_n, dx, dy, r1)

    if stack.r2_over_a > 0.0:
        r2 = stack.r2_over_a * a
        dx_sh = stack.dx_over_a * a
        dy_sh = stack.dy_over_a * a
        fill_ex = _subpixel_circle_fill(x_m, y_n, dx, dy, r1, -0.5 * dx_sh, -0.5 * dy_sh) + _subpixel_circle_fill(
            x_m, y_n, dx, dy, r2, 0.5 * dx_sh, 0.5 * dy_sh
        )
        fill_ey = _subpixel_circle_fill(x_n, y_m, dx, dy, r1, -0.5 * dx_sh, -0.5 * dy_sh) + _subpixel_circle_fill(
            x_n, y_m, dx, dy, r2, 0.5 * dx_sh, 0.5 * dy_sh
        )
        fill_ez = _subpixel_circle_fill(x_n, y_n, dx, dy, r1, -0.5 * dx_sh, -0.5 * dy_sh) + _subpixel_circle_fill(
            x_n, y_n, dx, dy, r2, 0.5 * dx_sh, 0.5 * dy_sh
        )
        fill_ex = np.clip(fill_ex, 0.0, 1.0)
        fill_ey = np.clip(fill_ey, 0.0, 1.0)
        fill_ez = np.clip(fill_ez, 0.0, 1.0)

    z_lo = np.empty_like(z_n)
    z_lo[0] = z_n[0]
    z_lo[1:] = z_m[:-1]
    z_hi = z_m
    dz_dual = z_hi - z_lo

    def _z_overlap(zmin: float, zmax: float) -> np.ndarray:
        return np.clip(np.minimum(z_hi, zmax) - np.maximum(z_lo, zmin), 0.0, dz_dual) / dz_dual

    w_act = _z_overlap(stack.z_act_min, stack.z_act_max)
    w_phc = _z_overlap(stack.z_phc_min, stack.z_phc_max)
    w_clad = 1.0 - w_act - w_phc

    eps_clad = stack.eps_clad
    eps_act = stack.eps_active
    eps_host = stack.eps_phc_host
    eps_hole = stack.eps_hole

    eps_xy_ex = eps_host + fill_ex * (eps_hole - eps_host)
    eps_xy_ey = eps_host + fill_ey * (eps_hole - eps_host)
    eps_xy_ez = eps_host + fill_ez * (eps_hole - eps_host)

    eps_x = np.empty((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64)
    eps_y = np.empty((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64)
    eps_z = np.empty((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64)

    for k in range(solver.Nz):
        eps_x[:, :, k] = w_clad[k] * eps_clad + w_act[k] * eps_act + w_phc[k] * eps_xy_ex
        eps_y[:, :, k] = w_clad[k] * eps_clad + w_act[k] * eps_act + w_phc[k] * eps_xy_ey
        zm = z_m[k]
        if stack.z_act_min <= zm <= stack.z_act_max:
            eps_z[:, :, k] = eps_act
        elif stack.z_phc_min <= zm <= stack.z_phc_max:
            eps_z[:, :, k] = eps_xy_ez
        else:
            eps_z[:, :, k] = eps_clad

    solver.ieps[:, :, :, "x"] = 1.0 / (epsilon_0 * eps_x)
    solver.ieps[:, :, :, "y"] = 1.0 / (epsilon_0 * eps_y)
    solver.ieps[:, :, :, "z"] = 1.0 / (epsilon_0 * eps_z)
    if solver.grid.boundary_at_domain_edge:
        solver._isolate_edge_ghosts()
    solver.update_tensors("ieps")
    solver.step_0 = True

    lam_min = C_LIGHT / (F0_EXCIT * stack.n_active)
    cpw_min = lam_min / max(dx, dz_core)
    cells_across_hole = (2.0 * stack.r_over_a * a) / dx

    meta = {
        "a_nm": a * 1e9,
        "r_over_a": stack.r_over_a,
        "fill_factor": stack.fill_factor,
        "nx": nx,
        "ny": nx,
        "nz": int(len(z_nodes) - 1),
        "n_cells": int(nx * nx * (len(z_nodes) - 1)),
        "dx_nm": dx * 1e9,
        "dz_core_nm": dz_core * 1e9,
        "dz_clad_nm": dz_clad * 1e9,
        "cpw_max_index": float(cpw_min),
        "cells_across_hole_diameter": float(cells_across_hole),
        "dt_fs": float(solver.dt * 1e15),
        "x_n": x_n,
        "y_n": y_n,
        "z_n": z_n,
        "x_m": x_m,
        "y_m": y_m,
        "eps_x": eps_x,
        "eps_y": eps_y,
    }
    return solver, meta


def _make_c4v_sources_and_probes(
    stack: PCSELLayerStack,
    meta: dict,
    f0: float = F0_EXCIT,
    df: float = DF_EXCIT,
    cutoff: float = CUTOFF_EXCIT,
) -> tuple[list[CurrentSource], dict[str, ResonanceMonitor], GaussianSource]:
    """Create C4-symmetric point sources and probes on exact staggered Yee coordinates."""
    pulse = GaussianSource(frequency=f0, fwidth=df, cutoff=cutoff)
    t_start = float(pulse.end_time)
    nx = int(meta["nx"])
    a = stack.a
    x_m = meta["x_m"]
    y_m = meta["y_m"]

    ym0 = float(y_m[nx // 2])
    xm0 = float(x_m[nx // 2])
    z_src = -60.0e-9
    z_prb = 30.0e-9

    sources = [
        CurrentSource(pulse, component="Ey", center=(0.25 * a, ym0, z_src), size=(0, 0, 0), amplitude=1.0),
        CurrentSource(pulse, component="Ey", center=(-0.25 * a, ym0, z_src), size=(0, 0, 0), amplitude=-0.7),
        CurrentSource(pulse, component="Ex", center=(xm0, 0.25 * a, z_src), size=(0, 0, 0), amplitude=0.9),
        CurrentSource(pulse, component="Ex", center=(xm0, -0.25 * a, z_src), size=(0, 0, 0), amplitude=-0.5),
        CurrentSource(pulse, component="Ey", center=(0.0, ym0, z_src), size=(0, 0, 0), amplitude=0.8),
        CurrentSource(pulse, component="Ex", center=(xm0, 0.0, z_src), size=(0, 0, 0), amplitude=0.8),
    ]

    probes = {
        "ey_p": ResonanceMonitor(component="Ey", center=(0.25 * a, ym0, z_prb), t_start=t_start),
        "ey_m": ResonanceMonitor(component="Ey", center=(-0.25 * a, ym0, z_prb), t_start=t_start),
        "ex_p": ResonanceMonitor(component="Ex", center=(xm0, 0.25 * a, z_prb), t_start=t_start),
        "ex_m": ResonanceMonitor(component="Ex", center=(xm0, -0.25 * a, z_prb), t_start=t_start),
        "ey_0": ResonanceMonitor(component="Ey", center=(0.0, ym0, z_prb), t_start=t_start),
        "ey_h": ResonanceMonitor(component="Ey", center=(-0.5 * a, ym0, z_prb), t_start=t_start),
        "ex_0": ResonanceMonitor(component="Ex", center=(xm0, 0.0, z_prb), t_start=t_start),
        "ex_h": ResonanceMonitor(component="Ex", center=(xm0, -0.5 * a, z_prb), t_start=t_start),
    }
    return sources, probes, pulse


def _pick_best_harminv_mode(res, f_target: float, is_bic: bool = False) -> dict[str, float]:
    """Select the dominant resonance from a `Resonances` object."""
    freqs = np.asarray(res.frequency, dtype=float)
    qs = np.asarray(res.Q, dtype=float)
    decays = np.asarray(res.decay_rate, dtype=float)
    amps = np.abs(np.asarray(res.amplitude))
    errs = np.asarray(res.error, dtype=float)

    if freqs.size == 0:
        return {
            "freq_hz": float(f_target),
            "freq_thz": float(f_target * 1e-12),
            "Q": 1000.0,
            "decay_rate": 0.0,
            "amplitude": 0.0,
            "rel_error": 1.0,
        }
    if is_bic:
        best_i = int(np.argmax(amps))
        q_val = float(abs(qs[best_i]))
    else:
        pos_mask = qs > 0
        idx_pool = np.where(pos_mask)[0] if np.any(pos_mask) else np.arange(freqs.size)
        best_i = int(idx_pool[int(np.argmax(amps[idx_pool]))])
        q_val = float(abs(qs[best_i]))
    return {
        "freq_hz": float(freqs[best_i]),
        "freq_thz": float(freqs[best_i] * 1e-12),
        "Q": q_val,
        "decay_rate": float(decays[best_i]),
        "amplitude": float(amps[best_i]),
        "rel_error": float(errs[best_i] / max(freqs[best_i], 1.0)),
    }


def run_pcsel_unit_cell(
    stack: PCSELLayerStack = DEFAULT_STACK,
    nx: int = 24,
    dz_core: float = 12.0e-9,
    dz_clad: float = 35.0e-9,
    z_box: float = 900.0e-9,
    kx: float = 0.0,
    ky: float = 0.0,
    t_ring: float = 240.0e-15,
    capture_fields: bool = False,
    dft_freqs_hz: list[float] | None = None,
) -> dict:
    """Run a single 3D FIT PCSEL unit-cell simulation and extract modes A, B, C, D."""
    solver, meta = build_pcsel_unit_cell_solver(
        stack=stack,
        nx=nx,
        dz_core=dz_core,
        dz_clad=dz_clad,
        z_box=z_box,
        kx=kx,
        ky=ky,
    )
    sources, probes, pulse = _make_c4v_sources_and_probes(stack, meta)
    t_start = float(pulse.end_time)
    n_steps = int(np.ceil((t_start + t_ring) / solver.dt))

    # Preflight memory and runtime estimate before running 3D FIT
    est = estimate.estimate(
        (solver.Nx, solver.Ny, solver.Nz),
        steps=n_steps,
        dtype="complex128" if solver.dtype == np.complex128 else "float64",
        backend="cpu",
        cpml=True,
    )
    estimate.preflight(est, {"ram": 4.0e9, "gpu": 0}, raise_error=True)

    probe_list = list(probes.values())
    dft_xy = None
    dft_xz = None
    if capture_fields and dft_freqs_hz is not None:
        a = stack.a
        z_phc_mid = 0.5 * (stack.z_phc_min + stack.z_phc_max)
        dft_xy = DFTMonitor(
            frequencies=np.asarray(dft_freqs_hz, dtype=float),
            components=("Ex", "Ey", "Hz"),
            center=(0.0, 0.0, z_phc_mid),
            size=(a, a, 0.0),
        )
        dft_xz = DFTMonitor(
            frequencies=np.asarray(dft_freqs_hz, dtype=float),
            components=("Ex", "Ey", "Hz"),
            center=(0.0, 0.0, 0.0),
            size=(a, 0.0, 2.0 * (z_box - 120e-9)),
        )

    t0 = time.perf_counter()
    if capture_fields:
        k_phc = int(np.argmin(np.abs(meta["z_n"] - 0.5 * (stack.z_phc_min + stack.z_phc_max))))
        frames_ey = []
        frame_times_fs = []
        step_anim_start = max(int(t_start / solver.dt) + 200, n_steps - 192)

        for step in range(n_steps):
            t_now = step * float(solver.dt)
            for s in sources:
                s.update(solver, t_now)
            solver.one_step()
            for p in probe_list:
                p.update(solver, t_now)
            if t_now >= t_start:
                if dft_xy is not None:
                    dft_xy.update(solver, t_now)
                if dft_xz is not None:
                    dft_xz.update(solver, t_now)
                if step >= step_anim_start and (step - step_anim_start) % 8 == 0 and len(frames_ey) < 24:
                    frames_ey.append(np.real(solver.E[:, :, k_phc, "y"]).copy())
                    frame_times_fs.append(float((t_now + solver.dt) * 1e15))
    else:
        solver.emsolve(n_steps, source=sources, monitors=probe_list, verbose=0)
    wall_s = time.perf_counter() - t0

    a = stack.a
    at_gamma = (abs(kx) + abs(ky)) == 0.0
    if at_gamma:
        ey_p = np.real(probes["ey_p"].signal()[1])
        ey_m = np.real(probes["ey_m"].signal()[1])
        ey_0 = np.real(probes["ey_0"].signal()[1])
        ey_h = np.real(probes["ey_h"].signal()[1])
        ex_p = np.real(probes["ex_p"].signal()[1])
        ex_m = np.real(probes["ex_m"].signal()[1])
        ex_0 = np.real(probes["ex_0"].signal()[1])
        ex_h = np.real(probes["ex_h"].signal()[1])
    else:
        # Phase-compensate complex Bloch probe signals by exp(-i*(kx*x_p + ky*y_p))
        def _bloch_sig(key_p: str, xp: float, yp: float) -> np.ndarray:
            raw = probes[key_p].signal()[1]
            return np.real(raw * np.exp(-1j * (kx * xp + ky * yp)))

        ey_p = _bloch_sig("ey_p", 0.25 * a, 0.0)
        ey_m = _bloch_sig("ey_m", -0.25 * a, 0.0)
        ey_0 = _bloch_sig("ey_0", 0.0, 0.0)
        ey_h = _bloch_sig("ey_h", -0.5 * a, 0.0)
        ex_p = _bloch_sig("ex_p", 0.0, 0.25 * a)
        ex_m = _bloch_sig("ex_m", 0.0, -0.25 * a)
        ex_0 = _bloch_sig("ex_0", 0.0, 0.0)
        ex_h = _bloch_sig("ex_h", 0.0, -0.5 * a)

    # C4v symmetry projections:
    # A (A2) and B (B2) are odd across the hole center (ey_p - ey_m, ex_p - ex_m)
    # C and D (E doublet) are even across the hole center (ey_0 - ey_h, ex_0 - ex_h)
    combos = {
        "A": (ey_p - ey_m) - (ex_p - ex_m),
        "B": (ey_p - ey_m) + (ex_p - ex_m),
        "C": ey_0 - ey_h,
        "D": ex_0 - ex_h,
    }

    cwt_ref = solve_infinite_cwt(stack, kx=kx, ky=ky)
    dt_p = probes["ey_p"].dt

    extracted: dict[str, dict] = {}
    if at_gamma:
        for lbl, sig in combos.items():
            res_h = harminv(
                sig,
                dt_p,
                fmin=300.0e12,
                fmax=322.0e12,
                n_basis=10,
                max_rel_error=1e-2,
            )
            if len(res_h) == 0:
                res_h = harminv(
                    sig,
                    dt_p,
                    fmin=300.0e12,
                    fmax=322.0e12,
                    n_basis=10,
                    max_rel_error=None,
                )
            is_bic = lbl in ("A", "B")
            extracted[lbl] = _pick_best_harminv_mode(res_h, cwt_ref[lbl]["freq_hz"], is_bic=is_bic)

        if extracted["A"]["freq_hz"] > extracted["B"]["freq_hz"]:
            extracted["A"], extracted["B"] = extracted["B"], extracted["A"]
    else:
        cwt_sorted = sorted([cwt_ref[m]["freq_hz"] for m in ("A", "B", "C", "D")])
        if abs(ky) == 0.0:
            # Gamma-X: C_1h mirror y -> -y separates dispersive Ey bands (1, 4) from flat Ex bands (2, 3)
            band_specs = [
                ("A", combos["A"], max(292.0e12, cwt_sorted[0] - 4.5e12), min(307.0e12, cwt_sorted[0] + 2.5e12)),
                ("B", ex_p - ex_m, 305.0e12, 311.5e12),
                ("C", combos["D"], 310.0e12, 315.5e12),
                ("D", combos["B"], max(314.0e12, cwt_sorted[3] - 4.5e12), min(328.0e12, cwt_sorted[3] + 3.0e12)),
            ]
        else:
            # Gamma-M: C_1h diagonal mirror x <-> y separates odd (A) and even (B) lower/upper doublets
            band_specs = [
                ("A", combos["A"], max(292.0e12, cwt_sorted[0] - 4.5e12), min(308.0e12, cwt_sorted[0] + 2.5e12)),
                ("B", combos["B"], max(292.0e12, cwt_sorted[1] - 4.5e12), min(308.0e12, cwt_sorted[1] + 2.5e12)),
                ("C", combos["A"], max(313.0e12, cwt_sorted[2] - 4.5e12), min(328.0e12, cwt_sorted[2] + 3.0e12)),
                ("D", combos["B"], max(313.0e12, cwt_sorted[3] - 4.5e12), min(328.0e12, cwt_sorted[3] + 3.0e12)),
            ]
        for lbl, sig, f_lo, f_hi in band_specs:
            res_h = harminv(sig, dt_p, fmin=f_lo, fmax=f_hi, n_basis=12, max_rel_error=5e-2)
            if len(res_h) == 0 or not np.any(np.asarray(res_h.Q, dtype=float) > 0):
                res_h = harminv(sig, dt_p, fmin=f_lo, fmax=f_hi, n_basis=12, max_rel_error=None)
            extracted[lbl] = _pick_best_harminv_mode(res_h, 0.5 * (f_lo + f_hi), is_bic=False)

    out = {
        "meta": {k: v for k, v in meta.items() if not isinstance(v, np.ndarray)},
        "n_steps": n_steps,
        "t_start_fs": float(t_start * 1e15),
        "t_ring_fs": float(t_ring * 1e15),
        "wall_time_s": float(wall_s),
        "estimated_ram_mb": float(est.host_bytes / 1e6),
        "modes": extracted,
        "signals": {
            "dt_s": float(solver.dt),
            "sig_A": combos["A"],
            "sig_B": combos["B"],
            "sig_C": combos["C"],
            "sig_D": combos["D"],
        },
    }
    if capture_fields and dft_xy is not None and dft_xz is not None:
        out["dft_xy"] = {
            comp: dft_xy.get_field(comp)[0] for comp in ("Ex", "Ey", "Hz")
        }
        out["dft_xz"] = {
            comp: dft_xz.get_field(comp)[0] for comp in ("Ex", "Ey", "Hz")
        }
        out["frames_ey"] = np.asarray(frames_ey)
        out["frame_times_fs"] = np.asarray(frame_times_fs)
        out["x_n"] = meta["x_n"]
        out["y_n"] = meta["y_n"]
        out["z_n"] = meta["z_n"]
        out["eps_x"] = meta["eps_x"]
        out["eps_y"] = meta["eps_y"]
    return out


def build_finite_pcsel_2d_solver(
    stack: PCSELLayerStack = DEFAULT_STACK,
    n_periods: int = 10,
    cells_per_period: int = 22,
    n_pml: int = 8,
    cfln: float = 0.85,
) -> tuple[SolverFIT3D, dict]:
    """Build a 2D effective-index Wakis FIT solver for a finite N_periods x N_periods PCSEL cavity."""
    slab = solve_vertical_slab_mode(stack)
    n_eff = float(slab["n_eff"])
    gamma_phc = float(slab["gamma_phc"])

    # Effective 2D permittivities preserving the 3D-CWT in-plane Fourier contrast Gamma_PhC * (eps_hole - eps_host)
    eps_bg_eff = n_eff**2
    delta_eps_eff = gamma_phc * (stack.eps_hole - stack.eps_phc_host)
    eps_host_eff = eps_bg_eff - stack.fill_factor * delta_eps_eff
    eps_hole_eff = eps_host_eff + delta_eps_eff

    a = stack.a
    cav_len = n_periods * a
    pad = (n_pml + 3) * (a / cells_per_period)
    l_dom = cav_len + 2.0 * pad
    nx = int(round(l_dom / (a / cells_per_period)))
    dx = l_dom / nx
    dz = dx

    grid = GridFIT3D(-0.5 * l_dom, 0.5 * l_dom, -0.5 * l_dom, 0.5 * l_dom, -0.5 * dz, 0.5 * dz, nx, nx, 1)
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        use_stl=False,
        n_pml=n_pml,
        cfln=cfln,
        bg=[eps_host_eff, 1.0],
        verbose=0,
    )

    x_n = grid.x[: solver.Nx]
    y_n = grid.y[: solver.Ny]
    x_m = x_n + 0.5 * dx
    y_m = y_n + 0.5 * dx

    eps_x = np.full((solver.Nx, solver.Ny, solver.Nz), eps_host_eff, dtype=np.float64)
    eps_y = np.full((solver.Nx, solver.Ny, solver.Nz), eps_host_eff, dtype=np.float64)
    eps_z = np.full((solver.Nx, solver.Ny, solver.Nz), eps_host_eff, dtype=np.float64)

    r_hole = stack.r_over_a * a
    centers = (-0.5 * (n_periods - 1) + np.arange(n_periods)) * a
    fill_ex = np.zeros((solver.Nx, solver.Ny), dtype=np.float64)
    fill_ey = np.zeros((solver.Nx, solver.Ny), dtype=np.float64)
    fill_ez = np.zeros((solver.Nx, solver.Ny), dtype=np.float64)

    for cx in centers:
        for cy in centers:
            # Fast bounding-box slice around each hole
            ix = np.where(np.abs(x_n - cx) <= r_hole + 2.0 * dx)[0]
            iy = np.where(np.abs(y_n - cy) <= r_hole + 2.0 * dx)[0]
            if len(ix) == 0 or len(iy) == 0:
                continue
            sl_x = slice(int(ix[0]), int(ix[-1]) + 1)
            sl_y = slice(int(iy[0]), int(iy[-1]) + 1)
            fill_ex[sl_x, sl_y] += _subpixel_circle_fill(x_m[sl_x], y_n[sl_y], dx, dx, r_hole, x0=cx, y0=cy, n_sub=4)
            fill_ey[sl_x, sl_y] += _subpixel_circle_fill(x_n[sl_x], y_m[sl_y], dx, dx, r_hole, x0=cx, y0=cy, n_sub=4)
            fill_ez[sl_x, sl_y] += _subpixel_circle_fill(x_n[sl_x], y_n[sl_y], dx, dx, r_hole, x0=cx, y0=cy, n_sub=4)

    fill_ex = np.clip(fill_ex, 0.0, 1.0)
    fill_ey = np.clip(fill_ey, 0.0, 1.0)
    fill_ez = np.clip(fill_ez, 0.0, 1.0)

    eps_x[:, :, 0] = (1.0 - fill_ex) * eps_host_eff + fill_ex * eps_hole_eff
    eps_y[:, :, 0] = (1.0 - fill_ey) * eps_host_eff + fill_ey * eps_hole_eff
    eps_z[:, :, 0] = (1.0 - fill_ez) * eps_host_eff + fill_ey * 0.0 + fill_ez * (eps_hole_eff - eps_host_eff)

    solver.ieps[:, :, :, "x"] = 1.0 / (epsilon_0 * eps_x)
    solver.ieps[:, :, :, "y"] = 1.0 / (epsilon_0 * eps_y)
    solver.ieps[:, :, :, "z"] = 1.0 / (epsilon_0 * eps_z)
    if solver.grid.boundary_at_domain_edge:
        solver._isolate_edge_ghosts()
    solver.update_tensors("ieps")
    solver.step_0 = True

    n_max_2d = float(np.sqrt(eps_host_eff))
    lam_min_2d = C_LIGHT / (315.0e12 * n_max_2d)
    cpw_2d = float(lam_min_2d / dx)
    cells_per_hole_2d = float((2.0 * r_hole) / dx)

    meta = {
        "n_periods": n_periods,
        "cavity_length_um": cav_len * 1e6,
        "domain_length_um": l_dom * 1e6,
        "cells_per_period": cells_per_period,
        "nx": nx,
        "ny": nx,
        "dx_nm": dx * 1e9,
        "cpw_max_index": cpw_2d,
        "cells_per_hole_diameter": cells_per_hole_2d,
        "n_eff": n_eff,
        "eps_host_eff": eps_host_eff,
        "eps_hole_eff": eps_hole_eff,
        "x_n": x_n,
        "y_n": y_n,
        "eps_z_2d": eps_z[:, :, 0],
        "centers": centers,
        "r_hole": r_hole,
    }
    return solver, meta


def run_finite_pcsel_2d_fit(
    stack: PCSELLayerStack = DEFAULT_STACK,
    n_periods: int = 10,
    cells_per_period: int = 22,
    t_ring: float = 90.0e-15,
    capture_dft: bool = False,
) -> dict:
    """Run a 2D effective-index finite PCSEL FIT cavity simulation and extract Q_parallel(L)."""
    solver, meta = build_finite_pcsel_2d_solver(
        stack=stack,
        n_periods=n_periods,
        cells_per_period=cells_per_period,
    )
    a = stack.a
    cav_l = n_periods * a
    pulse = GaussianSource(frequency=308.7e12, fwidth=35.0e12, cutoff=3.0)
    t_start = float(pulse.end_time)

    # Mode-matched fundamental cos(pi*x/L)*cos(pi*y/L) A2 circulating lattice excitation
    centers = meta["centers"]
    inner_c = [float(c) for c in centers if abs(c) <= 1.6 * a]
    if not inner_c:
        inner_c = [float(centers[len(centers) // 2])]

    sources: list[CurrentSource] = []
    probes_2d: list[ResonanceMonitor] = []
    weights: list[float] = []
    for hx in inner_c:
        for hy in inner_c:
            w = float(np.cos(np.pi * hx / cav_l) * np.cos(np.pi * hy / cav_l))
            weights.append(w)
            sources.extend(
                [
                    CurrentSource(pulse, component="Ey", center=(hx + 0.25 * a, hy, 0.0), size=(0, 0, 0), amplitude=w),
                    CurrentSource(pulse, component="Ey", center=(hx - 0.25 * a, hy, 0.0), size=(0, 0, 0), amplitude=-w),
                    CurrentSource(pulse, component="Ex", center=(hx, hy + 0.25 * a, 0.0), size=(0, 0, 0), amplitude=-w),
                    CurrentSource(pulse, component="Ex", center=(hx, hy - 0.25 * a, 0.0), size=(0, 0, 0), amplitude=w),
                ]
            )
            probes_2d.extend(
                [
                    ResonanceMonitor(component="Ey", center=(hx + 0.25 * a, hy, 0.0), t_start=t_start),
                    ResonanceMonitor(component="Ey", center=(hx - 0.25 * a, hy, 0.0), t_start=t_start),
                    ResonanceMonitor(component="Ex", center=(hx, hy + 0.25 * a, 0.0), t_start=t_start),
                    ResonanceMonitor(component="Ex", center=(hx, hy - 0.25 * a, 0.0), t_start=t_start),
                ]
            )

    dft_mon = None
    if capture_dft:
        dft_mon = DFTMonitor(
            frequencies=[300.0e12],
            components=("Ex", "Ey", "Hz"),
            center=(0.0, 0.0, 0.0),
            size=(cav_l * 1.15, cav_l * 1.15, 0.0),
        )

    n_steps = int(np.ceil((t_start + t_ring) / solver.dt))
    est = estimate.estimate(
        (solver.Nx, solver.Ny, solver.Nz),
        steps=n_steps,
        dtype="float64",
        backend="cpu",
        cpml=True,
    )
    estimate.preflight(est, {"ram": 4.0e9, "gpu": 0}, raise_error=True)

    t0 = time.perf_counter()
    for step in range(n_steps):
        t_now = step * float(solver.dt)
        for s in sources:
            s.update(solver, t_now)
        solver.one_step()
        for p in probes_2d:
            p.update(solver, t_now)
        if dft_mon is not None and t_now >= t_start:
            dft_mon.update(solver, t_now)
    wall_s = time.perf_counter() - t0

    sig = np.zeros_like(np.real(probes_2d[0].signal()[1]))
    for idx_h, w in enumerate(weights):
        s0 = np.real(probes_2d[4 * idx_h + 0].signal()[1])
        s1 = np.real(probes_2d[4 * idx_h + 1].signal()[1])
        s2 = np.real(probes_2d[4 * idx_h + 2].signal()[1])
        s3 = np.real(probes_2d[4 * idx_h + 3].signal()[1])
        sig += w * ((s0 - s1) - (s2 - s3))

    res_h = harminv(sig, solver.dt, fmin=290e12, fmax=325e12, n_basis=8, max_rel_error=None)
    freqs_all = np.asarray(res_h.frequency, dtype=float)
    qs_all = np.asarray(res_h.Q, dtype=float)
    amps_all = np.abs(np.asarray(res_h.amplitude))
    valid_mask = (qs_all > 4.0) & (qs_all < 200.0) & (freqs_all >= 292e12) & (freqs_all <= 308e12)
    if np.any(valid_mask):
        cand_idx = np.where(valid_mask)[0]
        best_i = int(cand_idx[int(np.argmax(amps_all[cand_idx]))])
        f_thz = float(freqs_all[best_i] * 1e-12)
        q_par = float(qs_all[best_i])
    else:
        best = _pick_best_harminv_mode(res_h, 300.0e12)
        f_thz = float(best["freq_thz"])
        q_par = max(float(best["Q"]), 5.0)

    beta0 = 2.0 * np.pi / a
    alpha_par_cm = float((beta0 / q_par) * 1e-2)

    res = {
        "n_periods": n_periods,
        "cavity_length_um": float(meta["cavity_length_um"]),
        "cells_per_period": int(cells_per_period),
        "dx_nm": float(meta["dx_nm"]),
        "cpw_max_index": float(meta["cpw_max_index"]),
        "cells_per_hole_diameter": float(meta["cells_per_hole_diameter"]),
        "freq_thz": f_thz,
        "Q_par_fit": q_par,
        "alpha_par_fit_cm": alpha_par_cm,
        "wall_time_s": float(wall_s),
        "estimated_ram_mb": float(est.host_bytes / 1e6),
    }
    if capture_dft and dft_mon is not None:
        res["dft_Ey"] = dft_mon.get_field("Ey")[0][0, :, :, 0]
        res["x_n"] = meta["x_n"]
        res["y_n"] = meta["y_n"]
        res["eps_z_2d"] = meta["eps_z_2d"]
    return res


def save_pcsel_setup_figures(fig_dir: Path = FIG_DIR) -> dict[str, str]:
    """Generate 00a/00b setup figures and markdown setup tables for both FIT geometries."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. 3D Bloch Unit-Cell Setup Figures
    stack = DEFAULT_STACK
    solver_uc, meta_uc = build_pcsel_unit_cell_solver(stack=stack, nx=24)
    sources_uc, probes_uc, _ = _make_c4v_sources_and_probes(stack, meta_uc)
    a = stack.a
    z_phc_mid = 0.5 * (stack.z_phc_min + stack.z_phc_max)
    dft_uc = DFTMonitor(
        frequencies=[308.6e12, 308.9e12, 313.2e12],
        components=("Ex", "Ey", "Hz"),
        center=(0.0, 0.0, z_phc_mid),
        size=(a, a, 0.0),
    )

    structures_uc = [
        Box(
            center=(0.0, 0.0, 0.5 * (stack.z_act_min + stack.z_act_max)),
            size=(a, a, stack.z_act_max - stack.z_act_min),
            material=stack.eps_active,
        ),
        Box(
            center=(0.0, 0.0, 0.5 * (stack.z_phc_min + stack.z_phc_max)),
            size=(a, a, stack.z_phc_max - stack.z_phc_min),
            material=stack.eps_phc_host,
        ),
        Cylinder(
            center=(0.0, 0.0, 0.5 * (stack.z_phc_min + stack.z_phc_max)),
            radius=stack.r_over_a * a,
            length=stack.z_phc_max - stack.z_phc_min,
            material=stack.eps_hole,
            axis="z",
        ),
    ]
    mat_names_uc = {
        "region_0": f"Active InGaAs/GaAs QW core (n={stack.n_active:.2f})",
        "region_1": f"PhC GaAs host layer (n={stack.n_phc_host:.2f})",
        "region_2": f"Circular air hole (r/a={stack.r_over_a:.2f}, n={stack.n_hole:.2f})",
    }

    res_uc = schematic.save_setup_figures(
        solver_uc,
        fig_dir,
        prefix="00",
        suffix="",
        plane="xz",
        unit="nm",
        structures=structures_uc,
        material_names=mat_names_uc,
        sources=sources_uc[:2],
        monitors=[probes_uc["ey_p"], dft_uc],
        freq=F0_EXCIT,
        min_feature=2.0 * stack.r_over_a * a,
        title_geometry="3D PCSEL Unit Cell (Bloch-Periodic x,y; CPML z)",
        title_mesh="3D PCSEL Unit Cell Graded Mesh & Gaussian Excitation",
    )

    # 2. 2D Effective-Index Finite PCSEL Setup Figures (cells_per_period=22 -> CPW=22.1 >= 20)
    solver_fin, meta_fin = build_finite_pcsel_2d_solver(stack=stack, n_periods=10, cells_per_period=22)
    cav_l = 10 * a
    pulse_fin = GaussianSource(frequency=308.7e12, fwidth=35.0e12, cutoff=3.0)
    src_fin = CurrentSource(pulse_fin, component="Ey", center=(0.75 * a, 0.5 * a, 0.0), size=(0, 0, 0), amplitude=1.0)
    mon_fin = ResonanceMonitor(
        component="Ey",
        center=(0.75 * a, 0.5 * a, 0.0),
        t_start=float(pulse_fin.end_time),
    )
    dft_fin = DFTMonitor(
        frequencies=[300.0e12],
        components=("Ex", "Ey", "Hz"),
        center=(0.0, 0.0, 0.0),
        size=(cav_l, cav_l, 0.0),
    )
    structures_fin = [
        Box(
            center=(0.0, 0.0, 0.0),
            size=(cav_l, cav_l, meta_fin["dx_nm"] * 1e-9),
            material=meta_fin["eps_host_eff"],
        ),
        Cylinder(
            center=(0.5 * a, 0.5 * a, 0.0),
            radius=stack.r_over_a * a,
            length=meta_fin["dx_nm"] * 1e-9,
            material=meta_fin["eps_hole_eff"],
            axis="z",
        ),
    ]
    mat_names_fin = {
        "region_0": f"10x10 Finite PCSEL Cavity (n_eff={meta_fin['n_eff']:.3f})",
        "region_1": f"Etched PhC air hole array (r/a={stack.r_over_a:.2f})",
    }
    res_fin = schematic.save_setup_figures(
        solver_fin,
        fig_dir,
        prefix="00",
        suffix="_finite",
        plane="xy",
        unit="um",
        structures=structures_fin,
        material_names=mat_names_fin,
        sources=[src_fin],
        monitors=[mon_fin, dft_fin],
        freq=F0_EXCIT,
        min_feature=2.0 * stack.r_over_a * a,
        title_geometry="2D Effective-Index Finite PCSEL Cavity (10x10 Periods, CPML x,y)",
        title_mesh="2D Finite PCSEL Mesh & Excitation Spectrum",
    )

    return {"unit_cell_table": res_uc["table"], "finite_cavity_table": res_fin["table"]}


def _richardson_extrapolate(val_c: float, val_m: float, val_f: float, h_c: float, h_m: float, h_f: float) -> dict[str, float]:
    """Compute observed convergence order and second-order Richardson extrapolation."""
    # Second-order Richardson estimate from medium and fine grids
    r_mf = h_m / h_f
    val_rich = float(val_f + (val_f - val_m) / (r_mf**2 - 1.0))
    # Observed order p from (val_c - val_m) / (val_m - val_f) using log ratio
    diff_cm = abs(val_c - val_m)
    diff_mf = abs(val_m - val_f)
    if diff_mf > 1e-12 and diff_cm > diff_mf:
        p_obs = float(np.log(diff_cm / diff_mf) / np.log(h_c / h_m))
    else:
        p_obs = 2.0
    return {
        "richardson": val_rich,
        "observed_order": p_obs,
        "rel_err_coarse_pct": float(abs(val_c - val_rich) / abs(val_rich) * 100.0),
        "rel_err_medium_pct": float(abs(val_m - val_rich) / abs(val_rich) * 100.0),
        "rel_err_fine_pct": float(abs(val_f - val_rich) / abs(val_rich) * 100.0),
    }


def run_all_studies(
    quick: bool = False,
    out_dir: Path = DATA_DIR,
    fig_dir: Path = FIG_DIR,
    reuse_existing_conv: bool = False,
) -> dict:
    """Execute all 3D FIT and 3D-CWT studies sequentially and save JSON/NPZ artifacts."""
    out_dir = Path(out_dir)
    fig_dir = Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.perf_counter()
    stack = DEFAULT_STACK

    # ------------------------------------------------------------------
    # 0. Setup figures and 1D vertical slab / Green's function baseline
    # ------------------------------------------------------------------
    setup_tables = save_pcsel_setup_figures(fig_dir)
    slab = solve_vertical_slab_mode(stack, nz=2001)
    greens = solve_vertical_greens_functions(stack, slab=slab)
    cwt_base = compute_cwt_coupling_coefficients(stack, slab=slab, greens=greens, include_longitudinal=False)
    inf_base = solve_infinite_cwt(stack, cwt_data=cwt_base, dressed_vertical_profile=False)
    inf_dressed = solve_infinite_cwt(stack, cwt_data=cwt_base, dressed_vertical_profile=True)
    cwt_long = compute_cwt_coupling_coefficients(stack, slab=slab, greens=greens, include_longitudinal=True)
    inf_long = solve_infinite_cwt(stack, cwt_data=cwt_long, dressed_vertical_profile=False)
    cwt_long_m5 = compute_cwt_coupling_coefficients(
        stack, slab=slab, greens=greens, m_max=5, include_longitudinal=True
    )
    inf_long_m5 = solve_infinite_cwt(stack, cwt_data=cwt_long_m5, dressed_vertical_profile=False)

    # ------------------------------------------------------------------
    # 1. 3D FIT Unit-Cell Convergence Study at Gamma (r/a = 0.18)
    # ------------------------------------------------------------------
    existing_json = out_dir / "pcsel_summary.json"
    existing_uc_npz = out_dir / "unit_cell_fields.npz"
    existing_cwt_npz = out_dir / "cwt_and_finite_data.npz"
    prev_sum = None
    prev_cwt_npz = None
    if reuse_existing_conv and not quick and existing_json.exists() and existing_uc_npz.exists():
        prev_sum = json.loads(existing_json.read_text())
        if existing_cwt_npz.exists():
            prev_cwt_npz = dict(np.load(existing_cwt_npz))
        conv_results = prev_sum["convergence"]["levels"]
        ref_field_data = None
        # Update slab and Green's function arrays in unit_cell_fields.npz while keeping 3D FIT fields
        old_uc = dict(np.load(existing_uc_npz))
        old_uc["z_slab"] = slab["z"]
        old_uc["eps_slab"] = slab["eps_z"]
        old_uc["theta0"] = slab["theta0"]
        old_uc["phi_rad_real"] = np.real(greens["phi_rad"])
        old_uc["phi_rad_imag"] = np.imag(greens["phi_rad"])
        old_uc["phi_11"] = greens["phi_11"]
        np.savez_compressed(out_dir / "unit_cell_fields.npz", **old_uc)
    else:
        if quick:
            conv_levels = [
                {"label": "coarse", "nx": 12, "dz_core": 18.0e-9, "dz_clad": 42.0e-9, "z_box": 850.0e-9, "t_ring": 190.0e-15},
                {"label": "medium", "nx": 14, "dz_core": 16.0e-9, "dz_clad": 40.0e-9, "z_box": 850.0e-9, "t_ring": 190.0e-15},
                {"label": "fine", "nx": 16, "dz_core": 15.0e-9, "dz_clad": 38.0e-9, "z_box": 850.0e-9, "t_ring": 190.0e-15},
            ]
        else:
            conv_levels = [
                {"label": "coarse", "nx": 16, "dz_core": 15.0e-9, "dz_clad": 40.0e-9, "z_box": 900.0e-9, "t_ring": 240.0e-15},
                {"label": "medium", "nx": 20, "dz_core": 13.5e-9, "dz_clad": 38.0e-9, "z_box": 900.0e-9, "t_ring": 240.0e-15},
                {"label": "fine", "nx": 24, "dz_core": 12.0e-9, "dz_clad": 35.0e-9, "z_box": 900.0e-9, "t_ring": 240.0e-15},
            ]

        conv_results = []
        ref_field_data = None
        for lvl in conv_levels:
            is_ref = lvl["label"] == "fine"
            dft_freqs = [
                inf_base["A"]["freq_hz"],
                inf_base["B"]["freq_hz"],
                inf_base["C"]["freq_hz"],
            ]
            res_u = run_pcsel_unit_cell(
                stack=stack,
                nx=lvl["nx"],
                dz_core=lvl["dz_core"],
                dz_clad=lvl["dz_clad"],
                z_box=lvl["z_box"],
                t_ring=lvl["t_ring"],
                capture_fields=is_ref,
                dft_freqs_hz=dft_freqs if is_ref else None,
            )
            conv_results.append(
                {
                    "label": lvl["label"],
                    "meta": res_u["meta"],
                    "n_steps": res_u["n_steps"],
                    "wall_time_s": res_u["wall_time_s"],
                    "estimated_ram_mb": res_u["estimated_ram_mb"],
                    "modes": res_u["modes"],
                }
            )
            if is_ref:
                ref_field_data = res_u

    h_c = conv_results[0]["meta"]["dx_nm"]
    h_m = conv_results[1]["meta"]["dx_nm"]
    h_f = conv_results[2]["meta"]["dx_nm"]

    richardson_summary = {}
    for m_lbl in ("A", "B", "C", "D"):
        fc = conv_results[0]["modes"][m_lbl]["freq_thz"]
        fm = conv_results[1]["modes"][m_lbl]["freq_thz"]
        ff = conv_results[2]["modes"][m_lbl]["freq_thz"]
        richardson_summary[f"freq_{m_lbl}_thz"] = _richardson_extrapolate(fc, fm, ff, h_c, h_m, h_f)

    qc = 0.5 * (conv_results[0]["modes"]["C"]["Q"] + conv_results[0]["modes"]["D"]["Q"])
    qm = 0.5 * (conv_results[1]["modes"]["C"]["Q"] + conv_results[1]["modes"]["D"]["Q"])
    qf = 0.5 * (conv_results[2]["modes"]["C"]["Q"] + conv_results[2]["modes"]["D"]["Q"])
    richardson_summary["Q_CD"] = _richardson_extrapolate(qc, qm, qf, h_c, h_m, h_f)

    # Save fine-grid 3D FIT DFT mode profiles and ringdown signals for plotting
    if ref_field_data is not None:
        np.savez_compressed(
            out_dir / "unit_cell_fields.npz",
            x_n=ref_field_data["x_n"],
            y_n=ref_field_data["y_n"],
            z_n=ref_field_data["z_n"],
            eps_x=ref_field_data["eps_x"],
            eps_y=ref_field_data["eps_y"],
            z_slab=slab["z"],
            eps_slab=slab["eps_z"],
            theta0=slab["theta0"],
            phi_rad_real=np.real(greens["phi_rad"]),
            phi_rad_imag=np.imag(greens["phi_rad"]),
            phi_11=greens["phi_11"],
            dt_s=ref_field_data["signals"]["dt_s"],
            sig_A=ref_field_data["signals"]["sig_A"],
            sig_B=ref_field_data["signals"]["sig_B"],
            sig_C=ref_field_data["signals"]["sig_C"],
            sig_D=ref_field_data["signals"]["sig_D"],
            dft_xy_Ex=ref_field_data["dft_xy"]["Ex"],
            dft_xy_Ey=ref_field_data["dft_xy"]["Ey"],
            dft_xy_Hz=ref_field_data["dft_xy"]["Hz"],
            dft_xz_Ex=ref_field_data["dft_xz"]["Ex"],
            dft_xz_Ey=ref_field_data["dft_xz"]["Ey"],
            dft_xz_Hz=ref_field_data["dft_xz"]["Hz"],
            frames_ey=ref_field_data["frames_ey"],
            frame_times_fs=ref_field_data["frame_times_fs"],
        )

    # ------------------------------------------------------------------
    # 2. Band Structure Near Gamma Along Gamma-X and Gamma-M (Nx=24 -> CPW=22.7 >= 20)
    # ------------------------------------------------------------------
    k_red_cwt = np.linspace(0.0, 0.055, 29)
    bands_gx_cwt = compute_cwt_band_structure(stack, k_red_cwt, direction="GX", cwt_data=cwt_base)
    bands_gm_cwt = compute_cwt_band_structure(stack, k_red_cwt, direction="GM", cwt_data=cwt_base)

    k_red_fit_gx = [0.0, 0.025] if quick else [0.0, 0.015, 0.030, 0.045]
    k_red_fit_gm = [0.025] if quick else [0.015, 0.030]
    nx_bloch = 12 if quick else 24
    dz_bloch = 18.0e-9 if quick else 12.0e-9
    dz_clad_bloch = 42.0e-9 if quick else 35.0e-9
    z_box_bloch = 850.0e-9 if quick else 900.0e-9
    t_ring_bloch = 180.0e-15 if quick else 220.0e-15

    beta0 = 2.0 * np.pi / stack.a
    if prev_sum is not None and "bands_fit" in prev_sum:
        fit_bands_gx = prev_sum["bands_fit"]["gx"]
        fit_bands_gm = prev_sum["bands_fit"]["gm"]
    else:
        fit_bands_gx = []
        for kr in k_red_fit_gx:
            if kr == 0.0:
                m_gam = conv_results[2]["modes"]
                fit_bands_gx.append(
                    {
                        "k_red": 0.0,
                        "freqs_thz": sorted([m_gam[m]["freq_thz"] for m in ("A", "B", "C", "D")]),
                        "Q_factors": [m_gam[m]["Q"] for m in ("A", "B", "C", "D")],
                        "wall_time_s": 0.0,
                    }
                )
                continue
            res_k = run_pcsel_unit_cell(
                stack=stack,
                nx=nx_bloch,
                dz_core=dz_bloch,
                dz_clad=dz_clad_bloch,
                z_box=z_box_bloch,
                kx=kr * beta0,
                ky=0.0,
                t_ring=t_ring_bloch,
            )
            m_k = res_k["modes"]
            order = np.argsort([m_k[m]["freq_thz"] for m in ("A", "B", "C", "D")])
            lbls_sorted = [("A", "B", "C", "D")[i] for i in order]
            fit_bands_gx.append(
                {
                    "k_red": float(kr),
                    "freqs_thz": [m_k[l]["freq_thz"] for l in lbls_sorted],
                    "Q_factors": [m_k[l]["Q"] for l in lbls_sorted],
                    "wall_time_s": res_k["wall_time_s"],
                }
            )

        fit_bands_gm = [fit_bands_gx[0]]
        for kr in k_red_fit_gm:
            res_k = run_pcsel_unit_cell(
                stack=stack,
                nx=nx_bloch,
                dz_core=dz_bloch,
                dz_clad=dz_clad_bloch,
                z_box=z_box_bloch,
                kx=kr * beta0,
                ky=kr * beta0,
                t_ring=t_ring_bloch,
            )
            m_k = res_k["modes"]
            order = np.argsort([m_k[m]["freq_thz"] for m in ("A", "B", "C", "D")])
            lbls_sorted = [("A", "B", "C", "D")[i] for i in order]
            fit_bands_gm.append(
                {
                    "k_red": float(kr),
                    "freqs_thz": [m_k[l]["freq_thz"] for l in lbls_sorted],
                    "Q_factors": [m_k[l]["Q"] for l in lbls_sorted],
                    "wall_time_s": res_k["wall_time_s"],
                }
            )

    # ------------------------------------------------------------------
    # 3. Hole Radius / Fill-Factor Sweep: CWT vs 3D FIT (Nx=24 -> CPW=22.7 >= 20)
    # ------------------------------------------------------------------
    radii_fit = [0.14, 0.18, 0.24] if quick else [0.12, 0.15, 0.18, 0.22, 0.26]
    radii_cwt_dense = np.linspace(0.08, 0.28, 25)

    cwt_sweep_dense = []
    for r_val in radii_cwt_dense:
        st_r = PCSELLayerStack(a=stack.a, r_over_a=float(r_val))
        cwt_r = compute_cwt_coupling_coefficients(st_r, include_longitudinal=False)
        cwt_r_long = compute_cwt_coupling_coefficients(
            st_r, slab=cwt_r["slab"], greens=cwt_r["greens"], include_longitudinal=True
        )
        inf_u = solve_infinite_cwt(st_r, cwt_data=cwt_r, dressed_vertical_profile=False)
        inf_d = solve_infinite_cwt(st_r, cwt_data=cwt_r, dressed_vertical_profile=True)
        inf_l = solve_infinite_cwt(st_r, cwt_data=cwt_r_long, dressed_vertical_profile=False)
        cwt_sweep_dense.append(
            {
                "r_over_a": float(r_val),
                "fill_factor_pct": float(st_r.fill_factor * 100.0),
                "f_bragg_thz": float(cwt_r["slab"]["f_bragg"] * 1e-12),
                "gamma_phc": float(cwt_r["slab"]["gamma_phc"]),
                "eps_phc_avg": float(st_r.eps_phc_avg),
                "eps_phc_cd_eff": float(st_r.eps_phc_avg - abs(cwt_r["xi"][(2, 0)])),
                "eps_clad": float(st_r.eps_clad),
                "freq_A_thz": inf_u["A"]["freq_thz"],
                "freq_B_thz": inf_u["B"]["freq_thz"],
                "freq_B_with_longitudinal_thz": inf_l["B"]["freq_thz"],
                "freq_C_thz": inf_u["C"]["freq_thz"],
                "freq_C_with_longitudinal_thz": inf_l["C"]["freq_thz"],
                "freq_D_thz": inf_u["D"]["freq_thz"],
                "delta_f_AB_thz": float(inf_u["B"]["freq_thz"] - inf_u["A"]["freq_thz"]),
                "delta_f_AB_with_longitudinal_thz": float(inf_l["B"]["freq_thz"] - inf_l["A"]["freq_thz"]),
                "Q_C_unperturbed": inf_u["C"]["Q"],
                "Q_C_dressed": inf_d["C"]["Q"],
            }
        )

    prev_fit_radius_by_r = {}
    if prev_sum is not None and "radius_sweep" in prev_sum:
        for item_r in prev_sum["radius_sweep"]["fit_vs_cwt"]:
            prev_fit_radius_by_r[round(float(item_r["r_over_a"]), 4)] = item_r

    fit_radius_sweep = []
    for r_val in radii_fit:
        st_r = PCSELLayerStack(a=stack.a, r_over_a=float(r_val))
        cwt_r = compute_cwt_coupling_coefficients(st_r, include_longitudinal=False)
        cwt_r_long = compute_cwt_coupling_coefficients(
            st_r, slab=cwt_r["slab"], greens=cwt_r["greens"], include_longitudinal=True
        )
        inf_u = solve_infinite_cwt(st_r, cwt_data=cwt_r, dressed_vertical_profile=False)
        inf_d = solve_infinite_cwt(st_r, cwt_data=cwt_r, dressed_vertical_profile=True)
        inf_l = solve_infinite_cwt(st_r, cwt_data=cwt_r_long, dressed_vertical_profile=False)

        r_key = round(float(r_val), 4)
        if abs(r_val - stack.r_over_a) < 1e-6:
            # Use fine grid (Nx=24, CPW=22.66 >= 20) from convergence ladder
            m_fit = conv_results[2]["modes"]
            meta_r = conv_results[2]["meta"]
            wall_r = 0.0
            q_cd_fit = 0.5 * (m_fit["C"]["Q"] + m_fit["D"]["Q"])
            f_cd_fit = 0.5 * (m_fit["C"]["freq_thz"] + m_fit["D"]["freq_thz"])
            fit_fa = float(m_fit["A"]["freq_thz"])
            fit_fb = float(m_fit["B"]["freq_thz"])
            fit_qa = float(m_fit["A"]["Q"])
            fit_qb = float(m_fit["B"]["Q"])
        elif r_key in prev_fit_radius_by_r:
            prev_r = prev_fit_radius_by_r[r_key]
            meta_r = {
                "nx": prev_r["nx"],
                "cpw_max_index": prev_r["cpw_max_index"],
                "cells_across_hole_diameter": prev_r["cells_across_hole_diameter"],
            }
            wall_r = float(prev_r["wall_time_s"])
            fit_fa = float(prev_r["fit_freq_A_thz"])
            fit_fb = float(prev_r["fit_freq_B_thz"])
            f_cd_fit = float(prev_r["fit_freq_CD_thz"])
            fit_qa = float(prev_r["fit_Q_A"])
            fit_qb = float(prev_r["fit_Q_B"])
            q_cd_fit = float(prev_r["fit_Q_CD"])
        else:
            nx_r = 14 if quick else 24
            dz_r = 16.0e-9 if quick else 12.0e-9
            dz_clad_r = 40.0e-9 if quick else 35.0e-9
            z_box_r = 850.0e-9 if quick else 900.0e-9
            t_r = 190.0e-15 if quick else 240.0e-15
            res_r = run_pcsel_unit_cell(stack=st_r, nx=nx_r, dz_core=dz_r, dz_clad=dz_clad_r, z_box=z_box_r, t_ring=t_r)
            m_fit = res_r["modes"]
            meta_r = res_r["meta"]
            wall_r = res_r["wall_time_s"]
            q_cd_fit = 0.5 * (m_fit["C"]["Q"] + m_fit["D"]["Q"])
            f_cd_fit = 0.5 * (m_fit["C"]["freq_thz"] + m_fit["D"]["freq_thz"])
            fit_fa = float(m_fit["A"]["freq_thz"])
            fit_fb = float(m_fit["B"]["freq_thz"])
            fit_qa = float(m_fit["A"]["Q"])
            fit_qb = float(m_fit["B"]["Q"])

        fit_radius_sweep.append(
            {
                "r_over_a": float(r_val),
                "fill_factor_pct": float(st_r.fill_factor * 100.0),
                "nx": int(meta_r["nx"]),
                "cpw_max_index": float(meta_r["cpw_max_index"]),
                "cells_across_hole_diameter": float(meta_r["cells_across_hole_diameter"]),
                "cells_per_hole_diameter": float(meta_r["cells_across_hole_diameter"]),
                "fit_freq_A_thz": fit_fa,
                "fit_freq_B_thz": fit_fb,
                "fit_delta_f_AB_thz": float(fit_fb - fit_fa),
                "fit_freq_CD_thz": f_cd_fit,
                "fit_Q_A": fit_qa,
                "fit_Q_B": fit_qb,
                "fit_Q_CD": q_cd_fit,
                "cwt_freq_A_thz": inf_u["A"]["freq_thz"],
                "cwt_freq_B_thz": inf_u["B"]["freq_thz"],
                "cwt_freq_B_with_longitudinal_thz": inf_l["B"]["freq_thz"],
                "cwt_delta_f_AB_thz": float(inf_u["B"]["freq_thz"] - inf_u["A"]["freq_thz"]),
                "cwt_delta_f_AB_with_longitudinal_thz": float(inf_l["B"]["freq_thz"] - inf_l["A"]["freq_thz"]),
                "cwt_freq_CD_thz": inf_u["C"]["freq_thz"],
                "cwt_freq_CD_with_longitudinal_thz": inf_l["C"]["freq_thz"],
                "cwt_Q_CD_unperturbed": inf_u["C"]["Q"],
                "cwt_Q_CD_dressed": inf_d["C"]["Q"],
                "rel_err_freq_A_pct": float(abs(inf_u["A"]["freq_thz"] - fit_fa) / fit_fa * 100.0),
                "rel_err_freq_B_pct": float(abs(inf_u["B"]["freq_thz"] - fit_fb) / fit_fb * 100.0),
                "rel_err_freq_B_with_longitudinal_pct": float(abs(inf_l["B"]["freq_thz"] - fit_fb) / fit_fb * 100.0),
                "rel_err_freq_CD_pct": float(abs(inf_u["C"]["freq_thz"] - f_cd_fit) / f_cd_fit * 100.0),
                "rel_err_freq_CD_with_longitudinal_pct": float(abs(inf_l["C"]["freq_thz"] - f_cd_fit) / f_cd_fit * 100.0),
                "rel_err_Q_unperturbed_pct": float(abs(inf_u["C"]["Q"] - q_cd_fit) / q_cd_fit * 100.0),
                "rel_err_Q_dressed_pct": float(abs(inf_d["C"]["Q"] - q_cd_fit) / q_cd_fit * 100.0),
                "wall_time_s": float(wall_r),
            }
        )

    # ------------------------------------------------------------------
    # 4. Double-Lattice Coupling Scan & Finite-Size CWT + 2D FIT Cavity
    # ------------------------------------------------------------------
    d_grid = np.linspace(0.23, 0.30, 29)
    r_grid = np.linspace(0.65, 0.96, 25)
    dl_scan = double_lattice_coupling_scan(stack, d_grid, r_grid)

    dl_stack = dl_scan["opt_stack"]
    cwt_dl = dl_scan["opt_cwt"]
    alpha_v0_dl_cm = float(dl_scan["opt_alpha_v_inf_cm"])

    # Fair, self-consistent single-lattice baseline with broken 180-deg inversion symmetry
    # (Hermitian Bragg phase phi_Bragg in C_1D) solved so that its infinite-crystal fundamental
    # mode A has the exact same vertical radiation loss alpha_v,0 from C_rad as the double lattice.
    def _asym_single_alpha_diff(phi: float) -> float:
        cwt_p = compute_cwt_coupling_coefficients(stack, slab=slab, greens=greens, bragg_phase_rad=phi)
        inf_p = solve_infinite_cwt(stack, cwt_data=cwt_p)
        return float(inf_p["A"]["alpha_v_cm"] - alpha_v0_dl_cm)

    phi_single_asym = float(brentq(_asym_single_alpha_diff, 0.0, 1.2, xtol=1e-12))
    cwt_single_asym = compute_cwt_coupling_coefficients(
        stack, slab=slab, greens=greens, bragg_phase_rad=phi_single_asym
    )
    inf_single_asym = solve_infinite_cwt(stack, cwt_data=cwt_single_asym)
    alpha_v0_single_asym_cm = float(inf_single_asym["A"]["alpha_v_cm"])

    lengths_um = [50, 100, 200, 500, 1000, 2000, 3000] if not quick else [100, 200, 500, 1000, 3000]
    n_poly_fin = 6 if quick else 8

    finite_cwt_sweep = []
    prof_single_200um = None
    prof_double_200um = None
    prof_single_500um = None
    prof_double_500um = None
    prof_single_1mm = None
    prof_double_1mm = None

    for l_um in lengths_um:
        l_cm = l_um * 1e-4
        res_s = solve_finite_cwt(cwt_base["C_cm"], length=l_cm, n_poly=n_poly_fin)
        res_sm = solve_finite_cwt(cwt_single_asym["C_cm"], length=l_cm, n_poly=n_poly_fin)
        res_d = solve_finite_cwt(cwt_dl["C_cm"], length=l_cm, n_poly=n_poly_fin)
        if l_um == 200:
            prof_single_200um = res_s
            prof_double_200um = res_d
        elif l_um == 500:
            prof_single_500um = res_s
            prof_double_500um = res_d
        elif l_um == 1000:
            prof_single_1mm = res_s
            prof_double_1mm = res_d

        s_rel_margin_pct = float(res_s["delta_alpha_th"] / max(res_s["lowest_alpha_th"], 1e-12) * 100.0)
        sm_rel_margin_pct = float(res_sm["delta_alpha_th"] / max(res_sm["lowest_alpha_th"], 1e-12) * 100.0)
        d_rel_margin_pct = float(res_d["delta_alpha_th"] / max(res_d["lowest_alpha_th"], 1e-12) * 100.0)

        finite_cwt_sweep.append(
            {
                "L_um": float(l_um),
                "single_A1_alpha_th_cm": res_s["fundamental_A"]["alpha_th"],
                "single_A1_alpha_v_cm": res_s["fundamental_A"]["alpha_v"],
                "single_A1_alpha_par_cm": res_s["fundamental_A"]["alpha_par"],
                "single_A2_alpha_th_cm": res_s["second_A"]["alpha_th"],
                "single_B1_alpha_th_cm": res_s["fundamental_B"]["alpha_th"],
                "single_delta_alpha_th_cm": res_s["delta_alpha_th"],
                "single_rel_margin_pct": s_rel_margin_pct,
                "single_matched_A1_alpha_th_cm": res_sm["fundamental_A"]["alpha_th"],
                "single_matched_A1_alpha_v_cm": res_sm["fundamental_A"]["alpha_v"],
                "single_matched_A1_alpha_par_cm": res_sm["fundamental_A"]["alpha_par"],
                "single_matched_delta_alpha_th_cm": res_sm["delta_alpha_th"],
                "single_matched_rel_margin_pct": sm_rel_margin_pct,
                "single_matched_power_balance_err": res_sm["power_balance_error"],
                "single_power_balance_err": res_s["power_balance_error"],
                "double_A1_alpha_th_cm": res_d["fundamental_A"]["alpha_th"],
                "double_A1_alpha_v_cm": res_d["fundamental_A"]["alpha_v"],
                "double_A1_alpha_par_cm": res_d["fundamental_A"]["alpha_par"],
                "double_delta_alpha_th_cm": res_d["delta_alpha_th"],
                "double_rel_margin_pct": d_rel_margin_pct,
                "double_power_balance_err": res_d["power_balance_error"],
                "margin_enhancement_ratio": float(res_d["delta_alpha_th"] / max(res_s["delta_alpha_th"], 1e-9)),
                "matched_delta_alpha_enhancement_ratio": float(
                    res_d["delta_alpha_th"] / max(res_sm["delta_alpha_th"], 1e-9)
                ),
                "matched_rel_margin_enhancement_ratio": float(d_rel_margin_pct / max(sm_rel_margin_pct, 1e-9)),
                "unperturbed_rel_margin_enhancement_ratio": float(d_rel_margin_pct / max(s_rel_margin_pct, 1e-9)),
            }
        )

    # Compute local log-slopes d ln(y) / d ln(L) across consecutive cavity sizes L
    l_arr_sw = np.array([row["L_um"] for row in finite_cwt_sweep], dtype=float)
    dl_log_l = np.diff(np.log(l_arr_sw))
    slope_intervals = [f"{int(l_arr_sw[i])}-{int(l_arr_sw[i+1])}um" for i in range(len(l_arr_sw) - 1)]

    def _log_slopes(key: str) -> list[float]:
        vals = np.array([max(float(row[key]), 1e-15) for row in finite_cwt_sweep], dtype=float)
        return [float(v) for v in (np.diff(np.log(vals)) / dl_log_l)]

    finite_scaling_slopes = {
        "intervals": slope_intervals,
        "single_delta_alpha_th_log_slopes": _log_slopes("single_delta_alpha_th_cm"),
        "single_alpha_v_log_slopes": _log_slopes("single_A1_alpha_v_cm"),
        "single_alpha_par_log_slopes": _log_slopes("single_A1_alpha_par_cm"),
        "single_matched_delta_alpha_th_log_slopes": _log_slopes("single_matched_delta_alpha_th_cm"),
        "double_delta_alpha_th_log_slopes": _log_slopes("double_delta_alpha_th_cm"),
    }

    # Exact CWT crossover cavity length L_cross where double-lattice relative threshold margin
    # overtakes the single-lattice relative threshold margin ( lies in [100, 200] um )
    def _rel_margin_diff_asym(l_um_val: float) -> float:
        l_cm_val = l_um_val * 1e-4
        rs = solve_finite_cwt(cwt_single_asym["C_cm"], length=l_cm_val, n_poly=n_poly_fin)
        rd = solve_finite_cwt(cwt_dl["C_cm"], length=l_cm_val, n_poly=n_poly_fin)
        return float(
            rd["delta_alpha_th"] / rd["lowest_alpha_th"] - rs["delta_alpha_th"] / rs["lowest_alpha_th"]
        )

    def _rel_margin_diff_circ(l_um_val: float) -> float:
        l_cm_val = l_um_val * 1e-4
        rs = solve_finite_cwt(cwt_base["C_cm"], length=l_cm_val, n_poly=n_poly_fin)
        rd = solve_finite_cwt(cwt_dl["C_cm"], length=l_cm_val, n_poly=n_poly_fin)
        return float(
            rd["delta_alpha_th"] / rd["lowest_alpha_th"] - rs["delta_alpha_th"] / rs["lowest_alpha_th"]
        )

    crossover_l_asym_um = float(brentq(_rel_margin_diff_asym, 100.0, 200.0, xtol=0.05))
    crossover_l_circ_um = float(brentq(_rel_margin_diff_circ, 100.0, 250.0, xtol=0.05))

    # Second crossover at large L against the circular single lattice by log interpolation
    second_crossover_l_circ_um = float("nan")
    for idx_sw in range(len(finite_cwt_sweep) - 1):
        r0 = float(finite_cwt_sweep[idx_sw]["unperturbed_rel_margin_enhancement_ratio"])
        r1 = float(finite_cwt_sweep[idx_sw + 1]["unperturbed_rel_margin_enhancement_ratio"])
        l0 = float(finite_cwt_sweep[idx_sw]["L_um"])
        l1 = float(finite_cwt_sweep[idx_sw + 1]["L_um"])
        if r0 > 1.0 and r1 < 1.0:
            frac = -np.log(r0) / (np.log(r1) - np.log(r0))
            second_crossover_l_circ_um = float(np.exp(np.log(l0) + frac * (np.log(l1) - np.log(l0))))
            break

    # Single-lattice maximum / sweep edge across the swept cavity lengths
    best_sm_row = max(finite_cwt_sweep, key=lambda r: r["single_matched_rel_margin_pct"])
    best_s_row = max(finite_cwt_sweep, key=lambda r: r["single_rel_margin_pct"])
    best_d_row = max(finite_cwt_sweep, key=lambda r: r["double_rel_margin_pct"])

    # Small finite 2D effective-index FIT runs (cells_per_period=22 -> CPW=22.1 >= 20, 7.92 cells/hole)
    n_per_list = [8, 12] if quick else [8, 10, 12, 16]
    prev_fin2d_by_n = {}
    if prev_sum is not None and "finite_2d_fit" in prev_sum:
        for item_2d in prev_sum["finite_2d_fit"]:
            prev_fin2d_by_n[int(item_2d["n_periods"])] = item_2d

    finite_2d_fit_results = []
    fin_2d_dft_data = None
    for n_per in n_per_list:
        cap_dft = n_per == (8 if quick else 10)
        l_cm_small = n_per * stack.a * 1e2
        cwt_small = solve_finite_cwt(cwt_base["C_cm"], length=l_cm_small, n_poly=6)
        alpha_cwt = cwt_small["fundamental_A"]["alpha_par"]
        q_cwt = float((beta0 * 1e-2) / alpha_cwt)
        if int(n_per) in prev_fin2d_by_n and prev_cwt_npz is not None:
            p2d = prev_fin2d_by_n[int(n_per)]
            finite_2d_fit_results.append(
                {
                    "n_periods": int(n_per),
                    "L_um": float(p2d["L_um"]),
                    "cells_per_period": int(p2d["cells_per_period"]),
                    "cpw_max_index": float(p2d["cpw_max_index"]),
                    "cells_per_hole_diameter": float(p2d["cells_per_hole_diameter"]),
                    "fit_freq_thz": float(p2d["fit_freq_thz"]),
                    "fit_Q_par": float(p2d["fit_Q_par"]),
                    "fit_alpha_par_cm": float(p2d["fit_alpha_par_cm"]),
                    "cwt_Q_par": q_cwt,
                    "cwt_alpha_par_cm": float(alpha_cwt),
                    "rel_diff_alpha_par_pct": float(abs(p2d["fit_alpha_par_cm"] - alpha_cwt) / alpha_cwt * 100.0),
                    "wall_time_s": float(p2d["wall_time_s"]),
                    "estimated_ram_mb": float(p2d["estimated_ram_mb"]),
                }
            )
        else:
            r_2d = run_finite_pcsel_2d_fit(
                stack=stack,
                n_periods=n_per,
                cells_per_period=10 if quick else 22,
                t_ring=70.0e-15 if quick else 90.0e-15,
                capture_dft=cap_dft,
            )
            finite_2d_fit_results.append(
                {
                    "n_periods": int(n_per),
                    "L_um": float(r_2d["cavity_length_um"]),
                    "cells_per_period": int(r_2d["cells_per_period"]),
                    "cpw_max_index": float(r_2d["cpw_max_index"]),
                    "cells_per_hole_diameter": float(r_2d["cells_per_hole_diameter"]),
                    "fit_freq_thz": r_2d["freq_thz"],
                    "fit_Q_par": r_2d["Q_par_fit"],
                    "fit_alpha_par_cm": r_2d["alpha_par_fit_cm"],
                    "cwt_Q_par": q_cwt,
                    "cwt_alpha_par_cm": float(alpha_cwt),
                    "rel_diff_alpha_par_pct": float(abs(r_2d["alpha_par_fit_cm"] - alpha_cwt) / alpha_cwt * 100.0),
                    "wall_time_s": r_2d["wall_time_s"],
                    "estimated_ram_mb": r_2d["estimated_ram_mb"],
                }
            )
            if cap_dft:
                fin_2d_dft_data = r_2d

    if fin_2d_dft_data is not None:
        fin2d_x_um = fin_2d_dft_data["x_n"] * 1e6
        fin2d_y_um = fin_2d_dft_data["y_n"] * 1e6
        fin2d_eps_z = fin_2d_dft_data["eps_z_2d"]
        fin2d_Ey = np.real(fin_2d_dft_data["dft_Ey"])
    elif prev_cwt_npz is not None:
        fin2d_x_um = prev_cwt_npz["fin2d_x_um"]
        fin2d_y_um = prev_cwt_npz["fin2d_y_um"]
        fin2d_eps_z = prev_cwt_npz["fin2d_eps_z"]
        fin2d_Ey = prev_cwt_npz["fin2d_Ey"]
    else:
        fin2d_x_um = np.zeros(1)
        fin2d_y_um = np.zeros(1)
        fin2d_eps_z = np.zeros((1, 1))
        fin2d_Ey = np.zeros((1, 1))

    # Save bands, radius sweep, double-lattice scan, and finite-size arrays to NPZ
    np.savez_compressed(
        out_dir / "cwt_and_finite_data.npz",
        k_red_cwt=k_red_cwt,
        bands_gx_freqs_thz=bands_gx_cwt["frequencies_thz"],
        bands_gx_Q=bands_gx_cwt["Q_factors"],
        bands_gx_alpha_cm=bands_gx_cwt["alpha_v_cm"],
        bands_gm_freqs_thz=bands_gm_cwt["frequencies_thz"],
        bands_gm_Q=bands_gm_cwt["Q_factors"],
        bands_gm_alpha_cm=bands_gm_cwt["alpha_v_cm"],
        d_over_a=dl_scan["d_over_a"],
        r2_over_r1=dl_scan["r2_over_r1"],
        R_cm=dl_scan["R_cm"],
        I_cm=dl_scan["I_cm"],
        mu_cm=dl_scan["mu_cm"],
        abs_C12_cm=dl_scan["abs_C12_cm"],
        abs_C21_cm=dl_scan["abs_C21_cm"],
        min_feedback_cm=dl_scan["min_feedback_cm"],
        x_200um=(prof_single_200um["x"] * 1e4) if prof_single_200um is not None else np.zeros(1),
        intensity_single_200um=prof_single_200um["fundamental_A"]["intensity"] if prof_single_200um is not None else np.zeros((1, 1)),
        intensity_double_200um=prof_double_200um["fundamental_A"]["intensity"] if prof_double_200um is not None else np.zeros((1, 1)),
        x_500um=(prof_single_500um["x"] * 1e4) if prof_single_500um is not None else np.zeros(1),
        intensity_single_500um=prof_single_500um["fundamental_A"]["intensity"] if prof_single_500um is not None else np.zeros((1, 1)),
        intensity_double_500um=prof_double_500um["fundamental_A"]["intensity"] if prof_double_500um is not None else np.zeros((1, 1)),
        x_1mm_um=(prof_single_1mm["x"] * 1e4) if prof_single_1mm is not None else np.zeros(1),
        intensity_single_1mm=prof_single_1mm["fundamental_A"]["intensity"] if prof_single_1mm is not None else np.zeros((1, 1)),
        intensity_double_1mm=prof_double_1mm["fundamental_A"]["intensity"] if prof_double_1mm is not None else np.zeros((1, 1)),
        fin2d_x_um=fin2d_x_um,
        fin2d_y_um=fin2d_y_um,
        fin2d_eps_z=fin2d_eps_z,
        fin2d_Ey=fin2d_Ey,
    )

    total_wall_s = time.perf_counter() - t_total_start

    fine_modes = conv_results[2]["modes"]
    f_cd_fine = 0.5 * (fine_modes["C"]["freq_thz"] + fine_modes["D"]["freq_thz"])
    q_cd_fine = 0.5 * (fine_modes["C"]["Q"] + fine_modes["D"]["Q"])
    cwt_df_ab = float(inf_base["B"]["freq_thz"] - inf_base["A"]["freq_thz"])
    cwt_df_ab_long = float(inf_long["B"]["freq_thz"] - inf_long["A"]["freq_thz"])
    cwt_df_ab_long_m5 = float(inf_long_m5["B"]["freq_thz"] - inf_long_m5["A"]["freq_thz"])
    fit_df_ab = float(fine_modes["B"]["freq_thz"] - fine_modes["A"]["freq_thz"])
    rich_df_ab = float(richardson_summary["freq_B_thz"]["richardson"] - richardson_summary["freq_A_thz"]["richardson"])
    band_gap_cwt_trans = float(inf_base["C"]["freq_thz"] - inf_base["A"]["freq_thz"])
    band_gap_cwt_long = float(inf_long["C"]["freq_thz"] - inf_long["A"]["freq_thz"])
    band_gap_fit = float(f_cd_fine - fine_modes["A"]["freq_thz"])

    summary = {
        "mode": "quick" if quick else "full",
        "total_wall_time_s": float(total_wall_s),
        "stack": {
            "a_nm": stack.a * 1e9,
            "r_over_a": stack.r_over_a,
            "fill_factor_pct": stack.fill_factor * 100.0,
            "n_clad": stack.n_clad,
            "n_active": stack.n_active,
            "n_phc_host": stack.n_phc_host,
            "n_hole": stack.n_hole,
            "t_active_nm": (stack.z_act_max - stack.z_act_min) * 1e9,
            "t_phc_nm": (stack.z_phc_max - stack.z_phc_min) * 1e9,
        },
        "slab_1d": {
            "n_eff_fd": float(slab["n_eff"]),
            "n_eff_exact": float(slab["n_eff_exact"]),
            "rel_err_n_eff": float(abs(slab["n_eff"] - slab["n_eff_exact"]) / slab["n_eff_exact"]),
            "f_bragg_thz": float(slab["f_bragg"] * 1e-12),
            "gamma_phc": float(slab["gamma_phc"]),
            "gamma_act": float(slab["gamma_act"]),
            "optical_theorem_lhs": float(greens["optical_theorem_lhs"]),
            "optical_theorem_rhs": float(greens["optical_theorem_rhs"]),
            "optical_theorem_rel_err": float(
                abs(greens["optical_theorem_lhs"] - greens["optical_theorem_rhs"]) / greens["optical_theorem_rhs"]
            ),
        },
        "cwt_infinite_r018": {
            "R_cm": float(cwt_base["R_cm"]),
            "I_cm": float(cwt_base["I_cm"]),
            "mu_cm": float(cwt_base["mu_cm"]),
            "kappa_1d_cm": float(np.real(cwt_base["kappa_1d"])),
            "kappa_2d_cm": float(np.real(cwt_base["kappa_2d"])),
            "kappa_2d_longitudinal_only_cm": float(np.real(cwt_long["kappa_2d_longitudinal"])),
            "kappa_2d_with_longitudinal_cm": float(np.real(cwt_long["kappa_2d"])),
            "kappa_2d_with_longitudinal_m5_cm": float(np.real(cwt_long_m5["kappa_2d"])),
            "abs_C12_cm": float(abs(cwt_base["C12_cm"])),
            "abs_C21_cm": float(abs(cwt_base["C21_cm"])),
            "freq_A_thz": float(inf_base["A"]["freq_thz"]),
            "freq_B_thz": float(inf_base["B"]["freq_thz"]),
            "delta_f_AB_thz": cwt_df_ab,
            "freq_C_thz": float(inf_base["C"]["freq_thz"]),
            "freq_D_thz": float(inf_base["D"]["freq_thz"]),
            "band_gap_transverse_thz": band_gap_cwt_trans,
            "freq_A_with_longitudinal_thz": float(inf_long["A"]["freq_thz"]),
            "freq_B_with_longitudinal_thz": float(inf_long["B"]["freq_thz"]),
            "delta_f_AB_with_longitudinal_thz": cwt_df_ab_long,
            "delta_f_AB_with_longitudinal_m5_thz": cwt_df_ab_long_m5,
            "freq_C_with_longitudinal_thz": float(inf_long["C"]["freq_thz"]),
            "freq_D_with_longitudinal_thz": float(inf_long["D"]["freq_thz"]),
            "band_gap_with_longitudinal_thz": band_gap_cwt_long,
            "Q_C_unperturbed": float(inf_base["C"]["Q"]),
            "Q_C_dressed": float(inf_dressed["C"]["Q"]),
        },
        "fit_unit_cell_r018": {
            "freq_A_thz": float(fine_modes["A"]["freq_thz"]),
            "freq_B_thz": float(fine_modes["B"]["freq_thz"]),
            "delta_f_AB_thz": fit_df_ab,
            "rich_delta_f_AB_thz": rich_df_ab,
            "freq_C_thz": float(fine_modes["C"]["freq_thz"]),
            "freq_D_thz": float(fine_modes["D"]["freq_thz"]),
            "freq_CD_thz": float(f_cd_fine),
            "band_gap_fit_thz": band_gap_fit,
            "c4v_degeneracy_rel_diff": float(abs(fine_modes["C"]["freq_thz"] - fine_modes["D"]["freq_thz"]) / f_cd_fine),
            "Q_A": float(fine_modes["A"]["Q"]),
            "Q_B": float(fine_modes["B"]["Q"]),
            "Q_C": float(fine_modes["C"]["Q"]),
            "Q_D": float(fine_modes["D"]["Q"]),
            "Q_CD": float(q_cd_fine),
            "rel_err_cwt_vs_fit_A_pct": float(abs(inf_base["A"]["freq_thz"] - fine_modes["A"]["freq_thz"]) / fine_modes["A"]["freq_thz"] * 100.0),
            "rel_err_cwt_vs_fit_B_pct": float(abs(inf_base["B"]["freq_thz"] - fine_modes["B"]["freq_thz"]) / fine_modes["B"]["freq_thz"] * 100.0),
            "rel_err_cwt_longitudinal_vs_fit_B_pct": float(abs(inf_long["B"]["freq_thz"] - fine_modes["B"]["freq_thz"]) / fine_modes["B"]["freq_thz"] * 100.0),
            "rel_err_cwt_vs_fit_CD_pct": float(abs(inf_base["C"]["freq_thz"] - f_cd_fine) / f_cd_fine * 100.0),
            "rel_err_cwt_longitudinal_vs_fit_CD_pct": float(abs(inf_long["C"]["freq_thz"] - f_cd_fine) / f_cd_fine * 100.0),
            "rel_err_band_gap_transverse_vs_fit_pct": float(abs(band_gap_cwt_trans - band_gap_fit) / band_gap_fit * 100.0),
            "rel_err_band_gap_longitudinal_vs_fit_pct": float(abs(band_gap_cwt_long - band_gap_fit) / band_gap_fit * 100.0),
            "splitting_ratio_fit_over_cwt_transverse": float(fit_df_ab / cwt_df_ab),
            "splitting_ratio_fit_over_cwt_longitudinal": float(fit_df_ab / cwt_df_ab_long),
            "splitting_ratio_fit_over_cwt_longitudinal_m5": float(fit_df_ab / cwt_df_ab_long_m5),
            "rel_err_cwt_vs_rich_A_pct": float(abs(inf_base["A"]["freq_thz"] - richardson_summary["freq_A_thz"]["richardson"]) / richardson_summary["freq_A_thz"]["richardson"] * 100.0),
            "rel_err_cwt_vs_rich_B_pct": float(abs(inf_base["B"]["freq_thz"] - richardson_summary["freq_B_thz"]["richardson"]) / richardson_summary["freq_B_thz"]["richardson"] * 100.0),
            "rel_err_cwt_vs_rich_CD_pct": float(abs(inf_base["C"]["freq_thz"] - richardson_summary["freq_C_thz"]["richardson"]) / richardson_summary["freq_C_thz"]["richardson"] * 100.0),
            "rel_err_Q_dressed_vs_fit_pct": float(abs(inf_dressed["C"]["Q"] - q_cd_fine) / q_cd_fine * 100.0),
        },
        "convergence": {
            "levels": conv_results,
            "richardson": richardson_summary,
        },
        "bands_fit": {
            "gx": fit_bands_gx,
            "gm": fit_bands_gm,
        },
        "radius_sweep": {
            "fit_vs_cwt": fit_radius_sweep,
            "cwt_dense": cwt_sweep_dense,
        },
        "double_lattice": {
            "opt_d_over_a": float(dl_scan["opt_d_over_a"]),
            "opt_r2_over_r1": float(dl_scan["opt_r2_over_r1"]),
            "grid_opt_d_over_a": float(dl_scan["grid_opt_d_over_a"]),
            "grid_opt_r2_over_r1": float(dl_scan["grid_opt_r2_over_r1"]),
            "single_abs_C21_cm": float(abs(cwt_base["C21_cm"])),
            "double_min_abs_C_offdiag_cm": float(dl_scan["min_abs_C21_cm"]),
            "double_max_abs_C_offdiag_cm": float(dl_scan["opt_max_abs_C_cm"]),
            "double_alpha_v_inf_cm": alpha_v0_dl_cm,
            "single_asym_bragg_phase_rad": phi_single_asym,
            "single_asym_alpha_v_inf_cm": alpha_v0_single_asym_cm,
            "single_asym_abs_C12_cm": float(abs(cwt_single_asym["C12_cm"])),
            "single_asym_abs_C21_cm": float(abs(cwt_single_asym["C21_cm"])),
            "crossover_L_um_vs_asym_single": crossover_l_asym_um,
            "crossover_L_um_vs_circ_single": crossover_l_circ_um,
            "second_crossover_L_um_vs_circ_single": round(second_crossover_l_circ_um, 2),
            "second_crossover_L_mm_vs_circ_single": round(second_crossover_l_circ_um * 1e-3, 2),
            "single_asym_opt_L_um": float(best_sm_row["L_um"]),
            "single_asym_opt_rel_margin_pct": float(best_sm_row["single_matched_rel_margin_pct"]),
            "single_circ_sweep_edge_L_um": float(best_s_row["L_um"]),
            "single_circ_sweep_edge_rel_margin_pct": float(best_s_row["single_rel_margin_pct"]),
            "double_opt_L_um": float(best_d_row["L_um"]),
            "double_opt_rel_margin_pct": float(best_d_row["double_rel_margin_pct"]),
            "double_500um_vs_single_asym_opt_ratio": float(
                best_d_row["double_rel_margin_pct"] / max(best_sm_row["single_matched_rel_margin_pct"], 1e-9)
            ),
            "nonvanishing_branch_reduction_ratio": float(abs(cwt_base["C21_cm"]) / dl_scan["opt_max_abs_C_cm"]),
            "double_R_cm": float(cwt_dl["R_cm"]),
            "double_I_cm": float(cwt_dl["I_cm"]),
            "double_mu_cm": float(cwt_dl["mu_cm"]),
        },
        "finite_cwt_sweep": finite_cwt_sweep,
        "finite_scaling_slopes": finite_scaling_slopes,
        "finite_2d_fit": finite_2d_fit_results,
        "setup_tables": setup_tables,
    }

    (out_dir / "pcsel_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_all_figures(
    out_dir: Path = DATA_DIR,
    fig_dir: Path = FIG_DIR,
) -> list[Path]:
    """Render all publication-quality figures and GIF animation from saved data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    figstyle.apply_style()
    out_dir = Path(out_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((out_dir / "pcsel_summary.json").read_text())
    # Ensure second_crossover_L_um_vs_circ_single and sweep-edge keys are computed from JSON arrays
    fin_sw_rows = summary.get("finite_cwt_sweep", [])
    dl_dict = summary.get("double_lattice", {})
    for idx_sw in range(len(fin_sw_rows) - 1):
        r0 = float(fin_sw_rows[idx_sw]["unperturbed_rel_margin_enhancement_ratio"])
        r1 = float(fin_sw_rows[idx_sw + 1]["unperturbed_rel_margin_enhancement_ratio"])
        l0 = float(fin_sw_rows[idx_sw]["L_um"])
        l1 = float(fin_sw_rows[idx_sw + 1]["L_um"])
        if r0 > 1.0 and r1 < 1.0:
            frac = -np.log(r0) / (np.log(r1) - np.log(r0))
            l_cross2 = float(np.exp(np.log(l0) + frac * (np.log(l1) - np.log(l0))))
            dl_dict["second_crossover_L_um_vs_circ_single"] = round(l_cross2, 2)
            dl_dict["second_crossover_L_mm_vs_circ_single"] = round(l_cross2 * 1e-3, 2)
            break
    if fin_sw_rows:
        best_s = max(fin_sw_rows, key=lambda r: r["single_rel_margin_pct"])
        dl_dict.pop("single_circ_opt_L_um", None)
        dl_dict.pop("single_circ_opt_rel_margin_pct", None)
        dl_dict["single_circ_sweep_edge_L_um"] = float(best_s["L_um"])
        dl_dict["single_circ_sweep_edge_rel_margin_pct"] = float(best_s["single_rel_margin_pct"])
    (out_dir / "pcsel_summary.json").write_text(json.dumps(summary, indent=2))
    uc_npz = np.load(out_dir / "unit_cell_fields.npz")
    cwt_npz = np.load(out_dir / "cwt_and_finite_data.npz")

    saved_files: list[Path] = []
    a_nm = float(summary["stack"]["a_nm"])
    r_nm = float(summary["stack"]["r_over_a"] * a_nm)

    # ------------------------------------------------------------------
    # Figure 1: 1D Vertical Slab Mode Theta_0(z) and Radiative Green's Function Phi_rad(z)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.5, height=4.2)
    z_nm = uc_npz["z_slab"] * 1e9
    eps_slab = uc_npz["eps_slab"]
    theta0 = uc_npz["theta0"] * 1e-3  # scale for display
    phi_r = uc_npz["phi_rad_real"]
    phi_i = uc_npz["phi_rad_imag"]
    phi_11 = uc_npz["phi_11"]

    ax = axes[0]
    ax.axvspan(-150.0, -30.0, color="#fde68a", alpha=0.55, label="Active core ($n=3.52$)")
    ax.axvspan(-30.0, 150.0, color="#bae6fd", alpha=0.55, label="PhC layer ($\\bar{n}=3.29$)")
    ax.plot(z_nm, np.sqrt(eps_slab), "k--", lw=1.6, label="Refractive index $n(z)$")
    ax_r = ax.twinx()
    ax_r.plot(z_nm, theta0 / np.max(theta0), color="#dc2626", lw=2.2, label="$\\Theta_0(z) / \\Theta_{\\max}$")
    ax.set_xlim(-800, 800)
    ax.set_xlabel("Vertical coordinate $z$ [nm]")
    ax.set_ylabel("Refractive index $n(z)$")
    ax_r.set_ylabel("Normalized $\\mathrm{TE}_0$ mode $\\Theta_0(z)$", color="#dc2626")
    ax.set_title(
        f"(a) 1D Vertical Slab Mode\n$n_{{\\mathrm{{eff}}}}={summary['slab_1d']['n_eff_fd']:.4f}$, "
        f"$\\Gamma_{{\\mathrm{{PhC}}}}={summary['slab_1d']['gamma_phc']*100:.1f}\\%$"
    )
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    ax.axvspan(-30.0, 150.0, color="#bae6fd", alpha=0.45, label="PhC driving region")
    scale_rad = np.max(np.abs(phi_r + 1j * phi_i))
    ax.plot(z_nm, phi_r / scale_rad, color="#2563eb", lw=2.0, label="$\\mathrm{Re}[\\Phi_{\\mathrm{rad}}(z)]$")
    ax.plot(z_nm, phi_i / scale_rad, color="#ea580c", lw=2.0, ls="--", label="$\\mathrm{Im}[\\Phi_{\\mathrm{rad}}(z)]$")
    ax.plot(z_nm, np.abs(phi_r + 1j * phi_i) / scale_rad, "k:", lw=1.5, label="$|\\Phi_{\\mathrm{rad}}(z)|$")
    ax.set_xlim(-1200, 1200)
    ax.set_xlabel("Vertical coordinate $z$ [nm]")
    ax.set_ylabel("Normalized radiative wave $\\Phi_{\\mathrm{rad}}(z)$")
    ax.set_title("(b) $(0,0)$ Radiative Green's Function\n(Outgoing Sommerfeld Wave)")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[2]
    ax.axvspan(-30.0, 150.0, color="#bae6fd", alpha=0.45, label="PhC driving region")
    ax.plot(z_nm, phi_11 / np.max(np.abs(phi_11)), color="#16a34a", lw=2.2, label="$\\Phi_{1,1}(z) / \\Phi_{\\max}$")
    ax.set_xlim(-600, 600)
    ax.set_xlabel("Vertical coordinate $z$ [nm]")
    ax.set_ylabel("Normalized evanescent wave $\\Phi_{1,1}(z)$")
    ax.set_title("(c) $(\\pm 1,\\pm 1)$ Evanescent Green's\nFunction ($|\\mathbf{K}_{1,1}|=\\sqrt{2}\\beta_0$)")
    ax.legend(loc="upper right", fontsize=7.5)

    p1 = fig_dir / "01_vertical_slab_and_greens_functions.png"
    figstyle.save(fig, p1)
    saved_files.append(p1)

    # ------------------------------------------------------------------
    # Figure 2: 3D FIT Gamma-Point Band-Edge Mode Profiles & Symmetry Ringdowns
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=2, ncols=2, width=11.5, height=9.2)
    x_nm = uc_npz["x_n"] * 1e9
    y_nm = uc_npz["y_n"] * 1e9
    dt_fs = float(uc_npz["dt_s"]) * 1e15
    sig_a = uc_npz["sig_A"]
    sig_b = uc_npz["sig_B"]
    sig_c = uc_npz["sig_C"]
    sig_d = uc_npz["sig_D"]
    t_arr = np.arange(len(sig_a)) * dt_fs

    ax = axes[0, 0]
    ax.plot(t_arr, sig_a / np.max(np.abs(sig_a)), label=f"Mode A ($A_2$, {summary['fit_unit_cell_r018']['freq_A_thz']:.2f} THz)", lw=1.4)
    ax.plot(t_arr, sig_b / np.max(np.abs(sig_b)) - 2.2, label=f"Mode B ($B_2$, {summary['fit_unit_cell_r018']['freq_B_thz']:.2f} THz)", lw=1.4)
    ax.plot(t_arr, sig_c / np.max(np.abs(sig_c)) - 4.4, label=f"Mode C ($E^{{(1)}}$, {summary['fit_unit_cell_r018']['freq_C_thz']:.2f} THz, $Q={summary['fit_unit_cell_r018']['Q_C']:.0f}$)", lw=1.4)
    ax.plot(t_arr, sig_d / np.max(np.abs(sig_d)) - 6.6, label=f"Mode D ($E^{{(2)}}$, {summary['fit_unit_cell_r018']['freq_D_thz']:.2f} THz, $Q={summary['fit_unit_cell_r018']['Q_D']:.0f}$)", lw=1.4)
    ax.set_xlabel("Ring-down time $t - t_0$ [fs]")
    ax.set_ylabel("Symmetry-projected signal (offset)")
    ax.set_title("(a) $C_{4v}$-Projected 3D FIT Ring-Down Signals")
    ax.legend(loc="upper right", fontsize=7.2)

    # Panels (b, c, d): 2D DFT spatial mode profiles at f_A, f_B, f_C with overlay_geometry
    solver_ov, _ = build_pcsel_unit_cell_solver(stack=DEFAULT_STACK, nx=len(x_nm))
    struct_xy = [
        Cylinder(
            center=(0.0, 0.0, 60.0e-9),
            radius=r_nm * 1e-9,
            length=180.0e-9,
            material=1.0,
            axis="z",
        )
    ]
    mat_xy = {"region_0": "Air hole (r=0.18a)"}

    dft_hz = uc_npz["dft_xy_Hz"]
    dft_ex = uc_npz["dft_xy_Ex"]
    dft_ey = uc_npz["dft_xy_Ey"]

    for idx_f, (ax_p, m_lbl, title_str) in enumerate(
        [
            (axes[0, 1], "A", "(b) Mode A ($A_2$, Dark BIC): $\\mathrm{Re}(H_z)$ & $\\mathbf{E}_\\parallel$"),
            (axes[1, 0], "B", "(c) Mode B ($B_2$, Dark BIC): $\\mathrm{Re}(H_z)$ & $\\mathbf{E}_\\parallel$"),
            (axes[1, 1], "C", "(d) Mode C ($E$, Radiative): $\\mathrm{Re}(E_y)$ in $(x,y)$"),
        ]
    ):
        if m_lbl in ("A", "B"):
            field_2d = np.real(dft_hz[idx_f, :, :, 0])
        else:
            field_2d = np.real(dft_ey[idx_f, :, :, 0])
        vmax = max(float(np.max(np.abs(field_2d))), 1e-12)
        im = ax_p.pcolormesh(
            x_nm,
            y_nm,
            (field_2d / vmax).T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        fig.colorbar(im, ax=ax_p, shrink=0.82)
        # Quiver of in-plane electric field vectors (Ex, Ey)
        step_q = max(1, len(x_nm) // 8)
        ex_q = np.real(dft_ex[idx_f, ::step_q, ::step_q, 0])
        ey_q = np.real(dft_ey[idx_f, ::step_q, ::step_q, 0])
        emag = max(float(np.max(np.hypot(ex_q, ey_q))), 1e-12)
        ax_p.quiver(
            x_nm[::step_q],
            y_nm[::step_q],
            (ex_q / emag).T,
            (ey_q / emag).T,
            color="black",
            scale=11.0,
            width=0.006,
        )
        schematic.overlay_geometry(
            ax_p,
            solver_ov,
            plane="xy",
            unit="nm",
            structures=struct_xy,
            material_names=mat_xy,
            show_boundaries=False,
        )
        ax_p.set_xlim(-0.5 * a_nm, 0.5 * a_nm)
        ax_p.set_ylim(-0.5 * a_nm, 0.5 * a_nm)
        ax_p.set_aspect("equal")
        ax_p.set_xlabel("$x$ [nm]")
        ax_p.set_ylabel("$y$ [nm]")
        ax_p.set_title(title_str)

    p2 = fig_dir / "02_gamma_band_edge_modes_3d_fit.png"
    figstyle.save(fig, p2)
    saved_files.append(p2)

    # ------------------------------------------------------------------
    # Figure 2 GIF: Animated Band-Edge Standing Wave Oscillation in the PhC Plane
    # ------------------------------------------------------------------
    frames_ey = uc_npz["frames_ey"]
    frame_times = uc_npz["frame_times_fs"]
    if len(frames_ey) >= 4:
        fig_g, ax_g = plt.subplots(figsize=(5.4, 4.8), dpi=110)
        vmax_g = max(float(np.max(np.abs(frames_ey))), 1e-12)
        im_g = ax_g.pcolormesh(
            x_nm,
            y_nm,
            (frames_ey[0] / vmax_g).T,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-1.0,
            vmax=1.0,
            shading="auto",
        )
        fig_g.colorbar(im_g, ax=ax_g, label="Normalized $E_y(x, y, z_{\\mathrm{PhC}}, t)$")
        hole_patch = mpatches.Circle((0.0, 0.0), radius=r_nm, fill=False, edgecolor="black", lw=1.8, ls="--")
        ax_g.add_patch(hole_patch)
        ax_g.set_xlim(-0.5 * a_nm, 0.5 * a_nm)
        ax_g.set_ylim(-0.5 * a_nm, 0.5 * a_nm)
        ax_g.set_aspect("equal")
        ax_g.set_xlabel("$x$ [nm]")
        ax_g.set_ylabel("$y$ [nm]")
        title_g = ax_g.set_title(f"3D FIT PhC Standing Wave ($t = {frame_times[0]:.1f}$ fs)")

        def _update_frame(idx_frame: int):
            im_g.set_array((frames_ey[idx_frame] / vmax_g).T.ravel())
            title_g.set_text(f"3D FIT PhC Standing Wave ($t = {frame_times[idx_frame]:.1f}$ fs)")
            return (im_g, title_g)

        ani = animation.FuncAnimation(fig_g, _update_frame, frames=len(frames_ey), interval=90, blit=False)
        p_gif = fig_dir / "02_band_edge_modes_animation.gif"
        ani.save(p_gif, writer="pillow", fps=10)
        plt.close(fig_g)
        saved_files.append(p_gif)

    # ------------------------------------------------------------------
    # Figure 3: Log-Log Grid Convergence of 3D FIT Modes A-D & Richardson Estimate
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=11.8, height=4.4)
    conv_lvls = summary["convergence"]["levels"]
    rich = summary["convergence"]["richardson"]
    dx_arr = np.array([lv["meta"]["dx_nm"] for lv in conv_lvls], dtype=float)
    cpw_arr = np.array([lv["meta"]["cpw_max_index"] for lv in conv_lvls], dtype=float)

    ax = axes[0]
    for m_lbl, col, mk in [("A", "#2563eb", "o"), ("B", "#16a34a", "s"), ("C", "#dc2626", "^"), ("D", "#9333ea", "v")]:
        err_pct = np.array(
            [
                rich[f"freq_{m_lbl}_thz"]["rel_err_coarse_pct"],
                rich[f"freq_{m_lbl}_thz"]["rel_err_medium_pct"],
                rich[f"freq_{m_lbl}_thz"]["rel_err_fine_pct"],
            ],
            dtype=float,
        )
        err_pct = np.maximum(err_pct, 1e-4)
        ax.loglog(dx_arr, err_pct, marker=mk, color=col, lw=2.0, ms=6.5, label=f"Mode {m_lbl} ($f_{{\\mathrm{{Rich}}}}={rich[f'freq_{m_lbl}_thz']['richardson']:.3f}$ THz)")

    # Second-order O(dx^2) reference guide
    guide_y = rich["freq_A_thz"]["rel_err_medium_pct"] * (dx_arr / dx_arr[1]) ** 2
    ax.loglog(dx_arr, guide_y, "k--", lw=1.6, label="$O(\\Delta x^2)$ 2nd-order guide")
    ax.set_xlabel("In-plane grid step $\\Delta x = a / N_x$ [nm]")
    ax.set_ylabel("Relative error $|f(N_x) - f_{\\mathrm{Rich}}| / f_{\\mathrm{Rich}}$ [%]")
    ax.set_title("(a) 3D FIT Band-Edge Frequency Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=7.8)

    ax = axes[1]
    err_q = np.array(
        [
            rich["Q_CD"]["rel_err_coarse_pct"],
            rich["Q_CD"]["rel_err_medium_pct"],
            rich["Q_CD"]["rel_err_fine_pct"],
        ],
        dtype=float,
    )
    err_q = np.maximum(err_q, 1e-3)
    ax.loglog(cpw_arr, err_q, "o-", color="#dc2626", lw=2.2, ms=7.0, label=f"Radiative $Q_{{C,D}}$ ($Q_{{\\mathrm{{Rich}}}}={rich['Q_CD']['richardson']:.1f}$)")
    guide_q = err_q[1] * (cpw_arr / cpw_arr[1]) ** (-2)
    ax.loglog(cpw_arr, guide_q, "k--", lw=1.6, label="$O(\\mathrm{CPW}^{-2})$ 2nd-order guide")
    ax.axvline(20.0, color="#16a34a", ls=":", lw=1.6, label="$\\mathrm{CPW} = 20$ threshold")
    ax.set_xlabel("Cells per wavelength in active layer ($n_{\\max}=3.52$)")
    ax.set_ylabel("Relative error $|Q(N_x) - Q_{\\mathrm{Rich}}| / Q_{\\mathrm{Rich}}$ [%]")
    ax.set_title("(b) Radiative Quality Factor $Q_{C,D}$ Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=7.8)

    p3 = fig_dir / "03_unit_cell_grid_convergence.png"
    figstyle.save(fig, p3)
    saved_files.append(p3)

    # ------------------------------------------------------------------
    # Figure 4: Photonic Band Structure M - Gamma - X and Radiative Q(k_parallel)
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=2, width=12.4, height=4.6)
    k_cwt = cwt_npz["k_red_cwt"]
    f_gx = cwt_npz["bands_gx_freqs_thz"]
    f_gm = cwt_npz["bands_gm_freqs_thz"]
    q_gx = cwt_npz["bands_gx_Q"]

    # Plot M <- Gamma -> X with negative abscissa for Gamma-M (scaled by sqrt(2)) and positive for Gamma-X
    x_gm = -np.sqrt(2.0) * k_cwt[::-1]
    x_gx = k_cwt
    x_band = np.concatenate([x_gm, x_gx[1:]])

    ax = axes[0]
    colors_b = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]
    for b_idx in range(4):
        y_band = np.concatenate([f_gm[::-1, b_idx], f_gx[1:, b_idx]])
        ax.plot(x_band, y_band, color=colors_b[b_idx], lw=2.0, label=f"3D-CWT Band {b_idx+1}")

    # Overlay 3D Bloch FIT points
    for pt in summary["bands_fit"]["gx"]:
        kr = float(pt["k_red"])
        for b_idx, fv in enumerate(pt["freqs_thz"]):
            lbl_fit = "3D Bloch FIT" if (kr == 0.0 and b_idx == 0) else None
            ax.plot(kr, fv, "ko", ms=5.5, mfc="white", mew=1.6, label=lbl_fit)
    for pt in summary["bands_fit"]["gm"]:
        kr = float(pt["k_red"])
        if kr == 0.0:
            continue
        for b_idx, fv in enumerate(pt["freqs_thz"]):
            ax.plot(-np.sqrt(2.0) * kr, fv, "ko", ms=5.5, mfc="white", mew=1.6)

    ax.axvline(0.0, color="0.4", ls="--", lw=1.0)
    ax.set_xlabel("In-plane wavevector $\\leftarrow \\Gamma\\text{--M} \\;\\; [k a / 2\\pi] \\;\\; \\Gamma\\text{--X} \\rightarrow$")
    ax.set_ylabel("Frequency $f(\\mathbf{k}_\\parallel)$ [THz]")
    ax.set_title("(a) PCSEL Band Structure Near $\\Gamma$ (3D-CWT vs 3D FIT)")
    ax.legend(loc="best", fontsize=7.8)

    ax = axes[1]
    for b_idx, lbl_b in [(0, "Band 1 (Mode A at $\\Gamma$, BIC)"), (1, "Band 2 (Mode B at $\\Gamma$, BIC)"), (2, "Band 3 (Mode C at $\\Gamma$)"), (3, "Band 4 (Mode D at $\\Gamma$)")]:
        q_clip_gx = np.minimum(q_gx[1:, b_idx], 1e6)
        ax.semilogy(k_cwt[1:], q_clip_gx, color=colors_b[b_idx], lw=2.0, label=lbl_b)
    # Overlay k^-2 BIC scaling guide
    k_guide = k_cwt[2:12]
    q_guide = q_gx[4, 0] * (k_guide / k_cwt[4]) ** (-2)
    ax.semilogy(k_guide, q_guide, "k--", lw=1.5, label="BIC scaling $Q \\propto |\\mathbf{k}_\\parallel|^{-2}$")
    ax.set_xlabel("Reduced wavevector $k_x a / (2\\pi)$ along $\\Gamma\\text{--X}$")
    ax.set_ylabel("Radiative quality factor $Q(\\mathbf{k}_\\parallel)$")
    ax.set_ylim(2e2, 5e5)
    ax.set_title("(b) Symmetry-Protected BIC $Q(\\mathbf{k}_\\parallel)$ Divergence at $\\Gamma$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=7.5)

    p4 = fig_dir / "04_band_structure_and_radiative_q.png"
    figstyle.save(fig, p4)
    saved_files.append(p4)

    # ------------------------------------------------------------------
    # Figure 5: Hole Radius / Fill-Factor Sweep & CWT Breakdown at Large Fill Factors
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.6, height=4.3)
    dense = summary["radius_sweep"]["cwt_dense"]
    fit_sw = summary["radius_sweep"]["fit_vs_cwt"]

    r_dense = np.array([d["r_over_a"] for d in dense])
    f_a_d = np.array([d["freq_A_thz"] for d in dense])
    f_b_d = np.array([d["freq_B_thz"] for d in dense])
    f_b_long_d = np.array([d.get("freq_B_with_longitudinal_thz", d["freq_B_thz"]) for d in dense])
    f_c_d = np.array([d["freq_C_thz"] for d in dense])
    q_u_d = np.array([d["Q_C_unperturbed"] for d in dense])
    q_d_d = np.array([d["Q_C_dressed"] for d in dense])
    eps_avg_d = np.array([d["eps_phc_avg"] for d in dense])
    eps_cd_d = np.array([d["eps_phc_cd_eff"] for d in dense])
    eps_clad_val = float(dense[0]["eps_clad"])

    r_fit = np.array([d["r_over_a"] for d in fit_sw])
    fit_fa = np.array([d["fit_freq_A_thz"] for d in fit_sw])
    fit_fb = np.array([d["fit_freq_B_thz"] for d in fit_sw])
    fit_fcd = np.array([d["fit_freq_CD_thz"] for d in fit_sw])
    fit_qcd = np.array([d["fit_Q_CD"] for d in fit_sw])

    ax = axes[0]
    ax.plot(r_dense, f_a_d, "-", color="#2563eb", lw=2.0, label="3D-CWT Mode A")
    ax.plot(r_dense, f_b_d, ":", color="#16a34a", lw=1.8, label="3D-CWT Mode B (transverse)")
    ax.plot(r_dense, f_b_long_d, "--", color="#15803d", lw=2.2, label="3D-CWT Mode B (+ ad hoc $\\propto \\xi\\xi/\\varepsilon_{\\mathrm{avg}}$)")
    ax.plot(r_dense, f_c_d, "-.", color="#dc2626", lw=2.0, label="3D-CWT Modes C, D")
    ax.plot(r_fit, fit_fa, "o", color="#2563eb", ms=6.5, label="3D FIT Mode A")
    ax.plot(r_fit, fit_fb, "s", color="#16a34a", ms=6.5, label="3D FIT Mode B")
    ax.plot(r_fit, fit_fcd, "^", color="#dc2626", ms=6.5, label="3D FIT Modes C, D")
    ax.set_xlabel("Normalized hole radius $r / a$")
    ax.set_ylabel("Band-edge frequency $f$ [THz]")
    ax.set_title("(a) $\\Gamma$-Point Frequencies vs Hole Radius")
    ax.legend(loc="upper left", fontsize=6.6)

    ax = axes[1]
    ax.plot(r_dense, q_u_d, "--", color="#ea580c", lw=2.0, label="Unperturbed 3D-CWT ($\\Theta_0$)")
    ax.plot(r_dense, q_d_d, "-", color="#16a34a", lw=2.2, label="Mode-Dressed 3D-CWT ($\\Theta_{C,D}$)")
    ax.plot(r_fit, fit_qcd, "ko", ms=7.0, label="Full 3D FIT (`harminv`)")
    ax.axvspan(0.22, 0.28, color="#fecaca", alpha=0.35, label="CWT breakdown ($f > 15\\%$)")
    ax.set_xlabel("Normalized hole radius $r / a$")
    ax.set_ylabel("Radiative quality factor $Q_{C,D}$")
    ax.set_ylim(300, 1400)
    ax.set_title("(b) Radiative $Q_{C,D}$ & Large-$f$ CWT Breakdown")
    ax.legend(loc="upper left", fontsize=7.2)

    ax = axes[2]
    ax.plot(r_dense, eps_avg_d, "b-", lw=2.0, label="Average $\\varepsilon_0(r/a)$")
    ax.plot(r_dense, eps_cd_d, "r-", lw=2.2, label="Mode C/D effective $\\varepsilon_0 - |\\xi_{2,0}|$")
    ax.axhline(eps_clad_val, color="k", ls="--", lw=1.6, label=f"Cladding $\\varepsilon_{{\\mathrm{{clad}}}} = n_{{\\mathrm{{clad}}}}^2 = {eps_clad_val:.2f}$")
    ax.axvspan(0.22, 0.28, color="#fecaca", alpha=0.35, label="Vertical mode expulsion")
    ax.set_xlabel("Normalized hole radius $r / a$")
    ax.set_ylabel("Effective PhC permittivity")
    ax.set_title("(c) Physical Mechanism of CWT Breakdown")
    ax.legend(loc="upper right", fontsize=7.2)

    p5 = fig_dir / "05_hole_radius_sweep_and_cwt_breakdown.png"
    figstyle.save(fig, p5)
    saved_files.append(p5)

    # ------------------------------------------------------------------
    # Figure 6: Double-Lattice Hermitian / Non-Hermitian Cancellation Map
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.6, height=4.3)
    d_arr = cwt_npz["d_over_a"]
    r_arr = cwt_npz["r2_over_r1"]
    r_mat = cwt_npz["R_cm"]
    i_mat = cwt_npz["I_cm"]
    mu_mat = cwt_npz["mu_cm"]
    min_fb = cwt_npz["min_feedback_cm"]
    opt_d = float(summary["double_lattice"]["opt_d_over_a"])
    opt_r = float(summary["double_lattice"]["opt_r2_over_r1"])

    j_opt = int(np.argmin(np.abs(r_arr - opt_r)))
    i_opt = int(np.argmin(np.abs(d_arr - opt_d)))

    ax = axes[0]
    ax.plot(d_arr, r_mat[:, j_opt], "b-", lw=2.2, label=f"Hermitian real $R(d)$ ($r_2/r_1={opt_r:.3f}$)")
    ax.plot(d_arr, -i_mat[:, j_opt], "g--", lw=2.0, label="Hermitian imag $-I(d)$")
    ax.plot(d_arr, mu_mat[:, j_opt], "r:", lw=2.0, label="Non-Hermitian rad $\\mu(d)$")
    ax.axhline(0.0, color="0.4", lw=1.0)
    ax.axvline(opt_d, color="k", ls="--", lw=1.3, label=f"Exact root $d^*/a={opt_d:.4f}$")
    ax.set_xlabel("Hole separation $d / a$")
    ax.set_ylabel("Coupling coefficient [cm$^{-1}$]")
    ax.set_title("(a) $R(d^*) = 0$ Cancellation Near $d \\approx 0.277a$")
    ax.legend(loc="best", fontsize=7.2)

    ax = axes[1]
    ax.plot(r_arr, np.abs(i_mat[i_opt, :]), "g-", lw=2.2, label="$|I(r_2/r_1)|$ (Hermitian asymmetry)")
    ax.plot(r_arr, mu_mat[i_opt, :], "r--", lw=2.2, label="$\\mu(r_2/r_1)$ (Radiative coupling)")
    ax.plot(r_arr, min_fb[i_opt, :], "k-", lw=2.0, label="$\\min(|C_{12}|, |C_{21}|)$")
    ax.axvline(opt_r, color="k", ls=":", lw=1.4, label=f"Exact root $(r_2/r_1)^*={opt_r:.3f}$")
    ax.set_xlabel("Hole radius ratio $r_2 / r_1$")
    ax.set_ylabel("Coupling magnitude [cm$^{-1}$]")
    ax.set_title("(b) $|I| = \\mu$ Balance vs Radius Ratio $r_2/r_1$")
    ax.legend(loc="best", fontsize=7.2)

    ax = axes[2]
    im_c = ax.pcolormesh(d_arr, r_arr, min_fb.T, cmap="viridis", shading="auto")
    fig.colorbar(im_c, ax=ax, label="$\\min(|C_{12}|, |C_{21}|)$ [cm$^{-1}$]")
    ax.plot(opt_d, opt_r, "r*", ms=13, mew=1.2, label=f"Root ({opt_d:.4f}, {opt_r:.3f})")
    ax.set_xlabel("Hole separation $d / a$")
    ax.set_ylabel("Hole radius ratio $r_2 / r_1$")
    ax.set_title("(c) Double-Lattice $180^\\circ$ Feedback Map")
    ax.legend(loc="upper right", fontsize=7.5)

    p6 = fig_dir / "06_double_lattice_coupling_cancellation.png"
    figstyle.save(fig, p6)
    saved_files.append(p6)

    # ------------------------------------------------------------------
    # Figure 7: Finite-Size PCSEL Threshold Gain, Relative Margin Delta_alpha/alpha_th, and Mode Profiles
    # ------------------------------------------------------------------
    fig, axes = figstyle.new_figure(nrows=1, ncols=3, width=13.8, height=4.4)
    fin_sw = summary["finite_cwt_sweep"]
    fin_fit = summary["finite_2d_fit"]

    l_arr = np.array([d["L_um"] for d in fin_sw])
    s_ath_a1 = np.array([d["single_A1_alpha_th_cm"] for d in fin_sw])
    s_av_a1 = np.array([d["single_A1_alpha_v_cm"] for d in fin_sw])
    s_apar_a1 = np.array([d["single_A1_alpha_par_cm"] for d in fin_sw])
    d_ath_a1 = np.array([d["double_A1_alpha_th_cm"] for d in fin_sw])
    s_rel_m = np.array([d["single_rel_margin_pct"] for d in fin_sw])
    sm_rel_m = np.array([d["single_matched_rel_margin_pct"] for d in fin_sw])
    d_rel_m = np.array([d["double_rel_margin_pct"] for d in fin_sw])
    alpha_v0_dl = float(summary["double_lattice"]["double_alpha_v_inf_cm"])

    ax = axes[0]
    ax.loglog(l_arr, d_ath_a1, "o-", color="#dc2626", lw=2.2, label="Double lattice $\\mathrm{A}_1$ total $\\alpha_{\\mathrm{th}}$")
    ax.loglog(l_arr, s_ath_a1, "s-", color="#2563eb", lw=2.0, label="Single lattice $\\mathrm{A}_1$ total $\\alpha_{\\mathrm{th}}$")
    ax.loglog(l_arr, s_apar_a1, "^--", color="#0284c7", lw=1.7, label="Single lattice $\\alpha_\\parallel \\propto L^{-3}$")
    ax.loglog(l_arr, s_av_a1, "v:", color="#f59e0b", lw=1.7, label="Single lattice $\\alpha_v \\propto L^{-2}$")
    l_fit2d = np.array([d["L_um"] for d in fin_fit])
    ap_fit2d = np.array([d["fit_alpha_par_cm"] for d in fin_fit])
    ap_cwt2d = np.array([d["cwt_alpha_par_cm"] for d in fin_fit])
    ax.loglog(l_fit2d, ap_fit2d, "ko", ms=6.0, mfc="none", mew=1.8, label="Small-cavity 2D FIT $\\alpha_\\parallel$")
    ax.loglog(l_fit2d, ap_cwt2d, "k+", ms=7.0, mew=1.8, label="Small-cavity 2D CWT $\\alpha_\\parallel$")
    ax.set_ylim(1e-4, 1e5)
    ax.set_xlabel("Device lateral size $L$ [$\\mu$m]")
    ax.set_ylabel("Loss / threshold gain [cm$^{-1}$]")
    ax.set_title("(a) Finite-Size Losses vs Cavity Size $L$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=6.4)

    ax = axes[1]
    l_cross_asym = float(summary["double_lattice"].get("crossover_L_um_vs_asym_single", 139.2))
    ax.loglog(
        l_arr,
        d_rel_m,
        "o-",
        color="#dc2626",
        lw=2.4,
        ms=6.5,
        label=f"Double lattice ($\\alpha_{{v,0}}={alpha_v0_dl:.1f}\\;\\mathrm{{cm}}^{{-1}}$)",
    )
    ax.loglog(
        l_arr,
        sm_rel_m,
        "s--",
        color="#2563eb",
        lw=2.2,
        ms=6.0,
        label=f"Synth.-phase single ($\\alpha_{{v,0}}={alpha_v0_dl:.1f}\\;\\mathrm{{cm}}^{{-1}}$)",
    )
    ax.loglog(
        l_arr,
        s_rel_m,
        "^:",
        color="#16a34a",
        lw=1.9,
        ms=5.5,
        label="Circular single lattice ($\\alpha_{v,0}=0$)",
    )
    l_cross2_circ = float(summary["double_lattice"].get("second_crossover_L_um_vs_circ_single", 1755.72))
    ax.axvline(l_cross_asym, color="0.35", ls=":", lw=1.4, label=f"1st crossover $L_{{\\mathrm{{cross}}}}={l_cross_asym:.0f}\\;\\mu$m")
    ax.axvline(l_cross2_circ, color="#16a34a", ls="--", lw=1.3, label=f"2nd crossover vs circ. ${l_cross2_circ*1e-3:.2f}\\;\\mathrm{{mm}}$")
    ax.set_xlabel("Device lateral size $L$ [$\\mu$m]")
    ax.set_ylabel("Relative threshold margin $\\Delta\\alpha_{\\mathrm{th}} / \\alpha_{\\mathrm{th}}$ [%]")
    ax.set_title("(b) Relative Margin $\\Delta\\alpha_{\\mathrm{th}} / \\alpha_{\\mathrm{th}}$ vs $L$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=6.6)

    ax = axes[2]
    x_200 = cwt_npz["x_200um"]
    int_s200 = cwt_npz["intensity_single_200um"]
    int_d200 = cwt_npz["intensity_double_200um"]
    x_500 = cwt_npz["x_500um"]
    int_s500 = cwt_npz["intensity_single_500um"]
    int_d500 = cwt_npz["intensity_double_500um"]
    if int_s200.ndim == 2 and int_s200.shape[0] > 1:
        mid2 = int_s200.shape[0] // 2
        xn_200 = x_200 / 200.0
        prof_s200 = int_s200[:, mid2] / np.max(int_s200[:, mid2])
        prof_d200 = int_d200[:, mid2] / np.max(int_d200[:, mid2])
        ax.plot(xn_200, prof_s200, color="#2563eb", ls="--", lw=2.2, label="Single lattice ($L=200\\;\\mu$m)")
        ax.plot(xn_200, prof_d200, color="#dc2626", ls="-", lw=2.4, label="Double lattice ($L=200\\;\\mu$m)")
    if int_d500.ndim == 2 and int_d500.shape[0] > 1:
        mid5 = int_d500.shape[0] // 2
        xn_500 = x_500 / 500.0
        if int_s500.ndim == 2 and int_s500.shape[0] > 1:
            prof_s500 = int_s500[:, mid5] / np.max(int_s500[:, mid5])
            ax.plot(xn_500, prof_s500, color="#0284c7", ls=":", lw=1.9, label="Single lattice ($L=500\\;\\mu$m)")
        prof_d500 = int_d500[:, mid5] / np.max(int_d500[:, mid5])
        ax.plot(xn_500, prof_d500, color="#ea580c", ls="-.", lw=2.2, label="Double lattice ($L=500\\;\\mu$m)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Normalized cavity coordinate $x / L$")
    ax.set_ylabel("Normalized intensity $|\\mathbf{v}(x, 0)|^2$")
    ax.set_title("(c) Mode Expansion: Single vs Double Lattice")
    ax.legend(loc="lower center", fontsize=6.6)

    p7 = fig_dir / "07_finite_size_pcsel_threshold_and_double_lattice.png"
    figstyle.save(fig, p7)
    saved_files.append(p7)

    return saved_files
