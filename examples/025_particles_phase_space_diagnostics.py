# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 025 — Ballistic Bunching and GPU-Resident Particle Phase-Space Diagnostics
==================================================================================

Demonstrates the P8 GPU-resident, MPI-aware particle diagnostics in
:mod:`wakis.particles` (:class:`PhaseSpaceMonitor`, :class:`EnergySpectrumMonitor`,
:class:`PlaneCurrentProbe`, :class:`EmittanceProbe`, and :class:`TrackRecorder`):

1. **Ballistic velocity bunching of a modulated relativistic electron beam**:
   - An electron beam is initialized with drift proper velocity ``u_0`` and a
     sinusoidal longitudinal velocity modulation
     ``u_z(z_0) = u_0 * (1 + alpha_u * sin(k_m * (z_0 - z_head)))``.
   - As the beam drifts downstream, faster tail electrons overtake slower head
     electrons inside each modulation wavelength ``lambda_m``, shearing the
     ``(z, u_z)`` phase space into microbunches near the analytical ballistic
     bunching distance ``L_b = gamma_0^2 / (k_m * alpha_u)``.
2. **In-situ GPU/CPU diagnostics without per-step host synchronization**:
   - :class:`PhaseSpaceMonitor` records a ring buffer of ``(z, u_z)`` snapshot
     frames capturing the progressive phase-space shearing and microbunching.
   - Two :class:`PlaneCurrentProbe` monitors at upstream plane ``z_1`` and
     downstream plane ``z_2`` measure the current compression ``I_2(t)`` relative
     to ``I_1(t)`` and verify exact total charge conservation ``Q_1 == Q_2``.
   - :class:`EnergySpectrumMonitor` resolves the double-horned kinetic-energy
     distribution characteristic of sinusoidal velocity modulation.
   - :class:`EmittanceProbe` and :class:`TrackRecorder` monitor the transverse
     RMS normalized emittance and individual marker particle trajectories.

Usage::

    python examples/025_particles_phase_space_diagnostics.py --smoke
    python examples/025_particles_phase_space_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import m_e

from wakis import GridFIT3D, SolverFIT3D, figstyle
from wakis.particles import (
    Collector,
    EmittanceProbe,
    EnergySpectrumMonitor,
    ParticleSolver,
    PhaseSpaceMonitor,
    PlaneCurrentProbe,
    Species,
    TrackRecorder,
)

try:
    import cupy as cp

    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001 - import or CUDA runtime/driver errors both mean no GPU
    cp = None
    HAS_GPU = False

if HAS_GPU:
    from wakis.gpu_fused import FusedStep

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "docs" / "img"
JSON_PATH = Path(__file__).resolve().with_suffix(".json")


def run_ballistic_bunching(smoke: bool = False, use_gpu: bool | None = None) -> dict:
    """Run the velocity-modulated drifting electron beam simulation and collect diagnostics."""
    if use_gpu is None:
        use_gpu = HAS_GPU

    # Domain: Lx = Ly = 0.08 m (half-width 40 mm = 5 * sigma_r), Lz = 0.16 m
    # Grid: 40 x 40 x Nz (dx = dy = 2.0 mm, 4 cells per sigma_r = 8 mm).
    Nx = 40
    Ny = 40
    Nz = 32 if smoke else 64
    n_macro = 4000 if smoke else 24000

    grid = GridFIT3D(
        xmin=-0.04,
        xmax=0.04,
        ymin=-0.04,
        ymax=0.04,
        zmin=0.0,
        zmax=0.16,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid,
        cfln=0.45,
        bc_low=["periodic", "periodic", "pec"],
        bc_high=["periodic", "periodic", "pec"],
        use_gpu=use_gpu,
        verbose=0,
    )
    if use_gpu:
        solver._fused_step = FusedStep(solver)
        solver.one_step = solver._fused_step

    # Beam physical parameters
    beta0 = 0.75
    gamma0 = float(1.0 / np.sqrt(1.0 - beta0 * beta0))
    u0 = gamma0 * beta0 * c_light
    alpha_u = 0.14  # 14% proper-velocity modulation amplitude
    lambda_m = 0.020  # 20 mm modulation wavelength (2 periods across 40 mm pulse)
    k_m = 2.0 * np.pi / lambda_m

    # Initial beam pulse occupies z in [0.010, 0.050] m
    rng = np.random.default_rng(2026025)
    z_init = np.linspace(0.010, 0.050, n_macro)
    sigma_r = 0.008  # 8 mm transverse RMS size
    sigma_xp = 1.5e-3
    rho_xxp = 0.60
    cov_t = np.array(
        [
            [sigma_r**2, rho_xxp * sigma_r * sigma_xp],
            [rho_xxp * sigma_r * sigma_xp, sigma_xp**2],
        ]
    )
    xxp = rng.multivariate_normal([0.0, 0.0], cov_t, size=n_macro)
    yyp = rng.multivariate_normal([0.0, 0.0], cov_t, size=n_macro)
    x_init = xxp[:, 0]
    y_init = yyp[:, 0]

    # Sinusoidal velocity modulation: positive slope d(u_z)/dz < 0 across each
    # period center so faster tail electrons overtake slower head electrons.
    uz_init = u0 * (1.0 - alpha_u * np.sin(k_m * (z_init - 0.010)))
    ux_init = xxp[:, 1] * uz_init
    uy_init = yyp[:, 1] * uz_init

    # Low total bunch charge (1 pC) in the ballistic tenuous-beam regime
    q_bunch_C = 1.0e-12
    w_init = np.full(n_macro, (q_bunch_C / e_charge) / n_macro, dtype=np.float64)

    sp = Species(
        name="electrons",
        charge=-e_charge,
        mass=m_e,
        use_gpu=use_gpu,
        track_ids=True,
    )
    sp.add_particles(
        x=x_init,
        y=y_init,
        z=z_init,
        ux=ux_init,
        uy=uy_init,
        uz=uz_init,
        w=w_init,
    )

    col = Collector(name="downstream_dump", plane=("z", 0.152, 1))
    ps = ParticleSolver(
        solver,
        species=[sp],
        collectors=[col],
        pusher="boris",
        shape=2,
    ).attach()

    # Choose n_steps so the slowest tail electron at z = 0.010 m transits well past
    # z_probe_2 = 0.080 m before reach of the collector at z = 0.152 m.
    dt = float(solver.dt)
    n_steps = int(np.ceil(0.092 / (beta0 * c_light * dt)))
    every_frame = max(n_steps // 12, 1)

    # Kinetic energy range in eV around the modulated beam energy
    ek0_eV = (m_e * c_light**2 / e_charge) * (gamma0 - 1.0)
    e_range_eV = (0.55 * ek0_eV, 1.55 * ek0_eV)

    mon_ps = PhaseSpaceMonitor(
        axes=("z", "uz"),
        bins=(64, 48),
        ranges=((0.005, 0.145), (0.80 * u0, 1.20 * u0)),
        weighted=True,
        every=every_frame,
        species="electrons",
        ring_size=16,
    )
    mon_es = EnergySpectrumMonitor(
        bins=48,
        range_eV=e_range_eV,
        weighted=True,
        every=1,
        species="electrons",
        ring_size=8,
    )
    z_probe_1 = 0.055
    z_probe_2 = 0.080
    probe_1 = PlaneCurrentProbe(z_plane=z_probe_1, every=1, species="electrons")
    probe_2 = PlaneCurrentProbe(z_plane=z_probe_2, every=1, species="electrons")
    mon_em = EmittanceProbe(z_window=(0.010, 0.150), every=1, species="electrons")
    sample_ids = np.linspace(0, n_macro - 1, 6, dtype=np.int64)
    mon_tr = TrackRecorder(
        ids=sample_ids, every=1, species="electrons", ring_size=n_steps + 4
    )

    for diag in (mon_ps, mon_es, probe_1, probe_2, mon_em, mon_tr):
        ps.add_diagnostic(diag)

    t0_wall = time.perf_counter()
    for _ in range(n_steps):
        solver.one_step()
    wall_s = time.perf_counter() - t0_wall

    res_ps = mon_ps.result()
    res_es = mon_es.result()
    res_p1 = probe_1.result()
    res_p2 = probe_2.result()
    res_em = mon_em.result()
    res_tr = mon_tr.result()

    # Separate check (1): NumPy emittance of the INITIAL loaded arrays against
    # analytic sigma_r * sigma_xp * sqrt(1 - rho^2), within 3.5 * eps / sqrt(N_eff).
    xp_init = ux_init / uz_init
    w_init_sum = float(np.sum(w_init))
    n_eff_init = float((w_init_sum * w_init_sum) / np.sum(w_init * w_init))
    x_init_mean = float(np.sum(w_init * x_init) / w_init_sum)
    xp_init_mean = float(np.sum(w_init * xp_init) / w_init_sum)
    s_xx_init = float(np.sum(w_init * (x_init - x_init_mean) ** 2) / w_init_sum)
    s_xpxp_init = float(np.sum(w_init * (xp_init - xp_init_mean) ** 2) / w_init_sum)
    s_xxp_init = float(
        np.sum(w_init * (x_init - x_init_mean) * (xp_init - xp_init_mean)) / w_init_sum
    )
    eps_x_init_numpy = float(
        np.sqrt(max(s_xx_init * s_xpxp_init - s_xxp_init * s_xxp_init, 0.0))
    )
    eps_x_analytic = float(sigma_r * sigma_xp * np.sqrt(1.0 - rho_xxp * rho_xxp))
    eps_x_init_stat_err = float(eps_x_analytic / np.sqrt(n_eff_init))
    eps_x_init_stat_tol = float(3.5 * eps_x_init_stat_err)
    eps_x_init_abs_err = float(abs(eps_x_init_numpy - eps_x_analytic))
    assert eps_x_init_abs_err <= eps_x_init_stat_tol

    # Check (2): Monitor vs NumPy on the same (final) snapshot is an identity
    # check; its tolerance must therefore be round-off: 8 * gamma_N * kappa * eps,
    # as in test (c) of test_088_particles_diagnostics.py.
    n_act = sp.n_particles
    x_np = np.asarray(sp.x.get() if use_gpu else sp.x[:n_act], dtype=np.float64)
    z_np = np.asarray(sp.z.get() if use_gpu else sp.z[:n_act], dtype=np.float64)
    ux_np = np.asarray(sp.ux.get() if use_gpu else sp.ux[:n_act], dtype=np.float64)
    uy_np = np.asarray(sp.uy.get() if use_gpu else sp.uy[:n_act], dtype=np.float64)
    uz_np = np.asarray(sp.uz.get() if use_gpu else sp.uz[:n_act], dtype=np.float64)
    w_np = np.asarray(sp.w.get() if use_gpu else sp.w[:n_act], dtype=np.float64)

    z_lo, z_hi = mon_em.z_window
    win_mask = (z_np >= z_lo) & (z_np <= z_hi) & (uz_np != 0.0)
    n_win = int(np.sum(win_mask))
    x_w = x_np[win_mask]
    xp_w = ux_np[win_mask] / uz_np[win_mask]
    u_mag_w = np.sqrt(
        ux_np[win_mask] ** 2 + uy_np[win_mask] ** 2 + uz_np[win_mask] ** 2
    )
    w_w = w_np[win_mask]
    w_sum = float(np.sum(w_w))
    n_eff = float((w_sum * w_sum) / np.sum(w_w * w_w))
    x_mean = float(np.sum(w_w * x_w) / w_sum)
    xp_mean = float(np.sum(w_w * xp_w) / w_sum)
    s_xx = float(np.sum(w_w * (x_w - x_mean) ** 2) / w_sum)
    s_xpxp = float(np.sum(w_w * (xp_w - xp_mean) ** 2) / w_sum)
    s_xxp = float(np.sum(w_w * (x_w - x_mean) * (xp_w - xp_mean)) / w_sum)
    eps_x_numpy = float(np.sqrt(max(s_xx * s_xpxp - s_xxp * s_xxp, 0.0)))
    bg_numpy = float(np.sum(w_w * (u_mag_w / c_light)) / w_sum)
    norm_eps_x_numpy = float(bg_numpy * eps_x_numpy)
    eps_x_stat_err = float(eps_x_numpy / np.sqrt(n_eff))

    ux_w = ux_np[win_mask]
    ux_mean = float(np.sum(w_w * ux_w) / w_sum)
    s_uxux = float(np.sum(w_w * (ux_w - ux_mean) ** 2) / w_sum)
    s_xux = float(np.sum(w_w * (x_w - x_mean) * (ux_w - ux_mean)) / w_sum)
    eps_n_u_x_numpy = float(np.sqrt(max(s_xx * s_uxux - s_xux * s_xux, 0.0)) / c_light)

    u_round = 0.5 * float(np.finfo(np.float64).eps)
    nu_win = max(n_win, 1) * u_round
    gamma_n = float(nu_win / max(1.0 - nu_win, 0.5))
    kappa = float(
        np.sqrt(
            1.0
            + (x_mean * x_mean) / max(s_xx, 1.0e-300)
            + (xp_mean * xp_mean) / max(s_xpxp, 1.0e-300)
        )
    )
    eps_x_roundoff_tol = float(8.0 * gamma_n * kappa * eps_x_numpy)
    norm_eps_x_roundoff_tol = float(8.0 * gamma_n * kappa * norm_eps_x_numpy)
    eps_n_u_x_roundoff_tol = float(8.0 * gamma_n * kappa * eps_n_u_x_numpy)

    eps_x_mon = float(res_em["emittance_x"][-1])
    norm_eps_x_mon = float(res_em["normalized_emittance_x"][-1])
    eps_n_u_x_mon = float(res_em["momentum_emittance_x"][-1])
    mon_vs_np_eps_x_diff = float(abs(eps_x_mon - eps_x_numpy))
    mon_vs_np_norm_eps_x_diff = float(abs(norm_eps_x_mon - norm_eps_x_numpy))
    mon_vs_np_eps_n_u_x_diff = float(abs(eps_n_u_x_mon - eps_n_u_x_numpy))
    assert mon_vs_np_eps_x_diff <= eps_x_roundoff_tol
    assert mon_vs_np_norm_eps_x_diff <= norm_eps_x_roundoff_tol
    assert mon_vs_np_eps_n_u_x_diff <= eps_n_u_x_roundoff_tol

    L_bunch_analytic = float((gamma0**2) / (k_m * alpha_u))

    return {
        "config": {
            "smoke": bool(smoke),
            "use_gpu": bool(use_gpu),
            "Nx": Nx,
            "Ny": Ny,
            "Nz": Nz,
            "n_macro": n_macro,
            "n_steps": n_steps,
            "dt_s": float(solver.dt),
            "beta0": beta0,
            "gamma0": gamma0,
            "u0_m_s": float(u0),
            "ek0_eV": float(ek0_eV),
            "alpha_u": alpha_u,
            "lambda_m_m": lambda_m,
            "sigma_r_m": float(sigma_r),
            "sigma_xp_rad": float(sigma_xp),
            "rho_xxp": float(rho_xxp),
            "L_bunch_analytic_m": L_bunch_analytic,
            "n_eff_init": n_eff_init,
            "eps_x_analytic_m_rad": eps_x_analytic,
            "eps_x_init_numpy_m_rad": eps_x_init_numpy,
            "eps_x_init_abs_err_m_rad": eps_x_init_abs_err,
            "eps_x_init_stat_err_m_rad": eps_x_init_stat_err,
            "eps_x_init_stat_tol_m_rad": eps_x_init_stat_tol,
            "n_eff": n_eff,
            "gamma_n": gamma_n,
            "kappa": kappa,
            "eps_x_numpy_m_rad": eps_x_numpy,
            "norm_eps_x_numpy_m_rad": norm_eps_x_numpy,
            "eps_n_u_x_numpy_m_rad": eps_n_u_x_numpy,
            "eps_x_roundoff_tol_m_rad": eps_x_roundoff_tol,
            "norm_eps_x_roundoff_tol_m_rad": norm_eps_x_roundoff_tol,
            "eps_n_u_x_roundoff_tol_m_rad": eps_n_u_x_roundoff_tol,
            "mon_vs_np_eps_x_diff_m_rad": mon_vs_np_eps_x_diff,
            "mon_vs_np_norm_eps_x_diff_m_rad": mon_vs_np_norm_eps_x_diff,
            "mon_vs_np_eps_n_u_x_diff_m_rad": mon_vs_np_eps_n_u_x_diff,
            "eps_x_stat_err_m_rad": eps_x_stat_err,
            "z_probe_1_m": z_probe_1,
            "z_probe_2_m": z_probe_2,
            "wall_time_s": float(wall_s),
        },
        "ps": res_ps,
        "es": res_es,
        "p1": res_p1,
        "p2": res_p2,
        "em": res_em,
        "tr": res_tr,
    }


def run_gpu_benchmark(
    counts: tuple[int, ...] = (1_000_000, 10_000_000),
    n_warmup: int = 3,
    n_repeats: int = 20,
) -> dict:
    """Measure stream time per step including host launch latency with and without all 5 monitors."""
    if not HAS_GPU:
        raise RuntimeError("--benchmark requires a CUDA GPU and CuPy.")
    if int(n_repeats) < 20:
        raise ValueError(f"n_repeats must be >= 20, got {n_repeats}.")

    dev_id = int(cp.cuda.Device().id)
    dev_props = cp.cuda.runtime.getDeviceProperties(dev_id)
    raw_name = dev_props.get("name", b"unknown")
    gpu_name = (
        raw_name.decode("utf-8", errors="replace")
        if isinstance(raw_name, bytes)
        else str(raw_name)
    )
    cupy_version = str(cp.__version__)
    cuda_runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
    timing_label = "stream time per step including host launch latency"
    grid_shape = (16, 16, 32)

    results: dict[str, dict] = {}
    for n_p in counts:
        n_p = int(n_p)
        grid = GridFIT3D(
            xmin=-0.04,
            xmax=0.04,
            ymin=-0.04,
            ymax=0.04,
            zmin=0.0,
            zmax=0.16,
            Nx=grid_shape[0],
            Ny=grid_shape[1],
            Nz=grid_shape[2],
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            cfln=0.45,
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            use_gpu=True,
            verbose=0,
        )
        solver._fused_step = FusedStep(solver)
        solver.one_step = solver._fused_step

        rng = np.random.default_rng(2026025 + n_p)
        sp = Species(
            name="electrons",
            charge=-e_charge,
            mass=m_e,
            use_gpu=True,
            track_ids=True,
        )
        u0 = 0.8 * c_light
        sp.add_particles(
            x=rng.normal(0.0, 0.006, size=n_p),
            y=rng.normal(0.0, 0.006, size=n_p),
            z=rng.uniform(0.01, 0.15, size=n_p),
            ux=rng.normal(0.0, 1.0e-3 * u0, size=n_p),
            uy=rng.normal(0.0, 1.0e-3 * u0, size=n_p),
            uz=rng.normal(u0, 0.02 * u0, size=n_p),
            w=np.ones(n_p, dtype=np.float64),
        )
        ps = ParticleSolver(solver, species=[sp], pusher="boris", shape=1).attach()
        sort_every = int(ps.sort_every)

        for _ in range(n_warmup):
            solver.one_step()
        cp.cuda.runtime.deviceSynchronize()

        without_ms: list[float] = []
        for _ in range(n_repeats):
            ev0 = cp.cuda.Event()
            ev1 = cp.cuda.Event()
            ev0.record()
            solver.one_step()
            ev1.record()
            ev1.synchronize()
            without_ms.append(float(cp.cuda.get_elapsed_time(ev0, ev1)))

        diags = (
            PhaseSpaceMonitor(
                axes=("z", "uz"),
                bins=(64, 48),
                ranges=((0.0, 0.16), (0.6 * u0, 1.4 * u0)),
                every=1,
                ring_size=8,
            ),
            EnergySpectrumMonitor(
                bins=48,
                range_eV=(1.0e4, 5.0e5),
                every=1,
                ring_size=8,
            ),
            PlaneCurrentProbe(z_plane=0.08, every=1),
            EmittanceProbe(z_window=(0.01, 0.15), every=1),
            TrackRecorder(
                ids=np.linspace(0, n_p - 1, 6, dtype=np.int64),
                every=1,
                species="electrons",
                ring_size=32,
            ),
        )
        for d in diags:
            ps.add_diagnostic(d)

        for _ in range(n_warmup):
            solver.one_step()
        cp.cuda.runtime.deviceSynchronize()

        with_ms: list[float] = []
        for _ in range(n_repeats):
            ev0 = cp.cuda.Event()
            ev1 = cp.cuda.Event()
            ev0.record()
            solver.one_step()
            ev1.record()
            ev1.synchronize()
            with_ms.append(float(cp.cuda.get_elapsed_time(ev0, ev1)))

        med_no = float(np.median(without_ms))
        min_no = float(np.min(without_ms))
        max_no = float(np.max(without_ms))
        med_with = float(np.median(with_ms))
        min_with = float(np.min(with_ms))
        max_with = float(np.max(with_ms))
        results[f"N_{n_p}"] = {
            "timing_label": timing_label,
            "gpu_name": gpu_name,
            "cupy_version": cupy_version,
            "cuda_runtime_version": cuda_runtime_version,
            "grid_shape": list(grid_shape),
            "sort_every": sort_every,
            "N": n_p,
            "n_particles": n_p,
            "n_warmup": int(n_warmup),
            "warmup_steps": int(n_warmup),
            "n_repeats": int(n_repeats),
            "repeats": int(n_repeats),
            "without_monitors_ms": without_ms,
            "with_all_monitors_ms": with_ms,
            "without_monitors_median_ms": med_no,
            "without_monitors_min_ms": min_no,
            "without_monitors_max_ms": max_no,
            "without_monitors_min_max_ms": [min_no, max_no],
            "with_all_monitors_median_ms": med_with,
            "with_all_monitors_min_ms": min_with,
            "with_all_monitors_max_ms": max_with,
            "with_all_monitors_min_max_ms": [min_with, max_with],
            "monitors_overhead_median_ms": float(med_with - med_no),
        }

        del diags, ps, sp, solver, grid
        cp.get_default_memory_pool().free_all_blocks()

    return results


def _assert_no_text_overlap(fig: plt.Figure) -> None:
    """Assert that all title, legend, and label window extents are pairwise disjoint."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    macro_items: list[tuple[str, Any]] = []
    fine_items: list[tuple[str, Any]] = []

    st = getattr(fig, "_suptitle", None)
    if st is not None and st.get_visible() and st.get_text().strip():
        bbox = st.get_window_extent(renderer)
        macro_items.append(("suptitle", bbox))
        fine_items.append(("suptitle", bbox))

    for idx, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        for t_name, t_art in (
            (f"ax[{idx}].title", ax.title),
            (f"ax[{idx}].left_title", ax._left_title),
            (f"ax[{idx}].right_title", ax._right_title),
            (f"ax[{idx}].xlabel", ax.xaxis.label),
            (f"ax[{idx}].ylabel", ax.yaxis.label),
        ):
            if t_art is not None and t_art.get_visible() and t_art.get_text().strip():
                bbox = t_art.get_window_extent(renderer)
                macro_items.append((t_name, bbox))
                fine_items.append((t_name, bbox))

        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            macro_items.append((f"ax[{idx}].legend", leg.get_window_extent(renderer)))
            leg_title = leg.get_title()
            if (
                leg_title is not None
                and leg_title.get_visible()
                and leg_title.get_text().strip()
            ):
                fine_items.append(
                    (f"ax[{idx}].legend.title", leg_title.get_window_extent(renderer))
                )
            handles = getattr(leg, "legend_handles", getattr(leg, "legendHandles", []))
            for h_idx, h in enumerate(handles):
                if h is not None and h.get_visible():
                    fine_items.append(
                        (
                            f"ax[{idx}].legend.handle[{h_idx}]",
                            h.get_window_extent(renderer),
                        )
                    )
            for l_idx, lt in enumerate(leg.get_texts()):
                if lt is not None and lt.get_visible() and lt.get_text().strip():
                    fine_items.append(
                        (
                            f"ax[{idx}].legend.text[{l_idx}]",
                            lt.get_window_extent(renderer),
                        )
                    )

    for l_idx, leg in enumerate(fig.legends):
        if leg is not None and leg.get_visible():
            macro_items.append(
                (f"fig.legend[{l_idx}]", leg.get_window_extent(renderer))
            )
            for t_idx, lt in enumerate(leg.get_texts()):
                if lt is not None and lt.get_visible() and lt.get_text().strip():
                    fine_items.append(
                        (
                            f"fig.legend[{l_idx}].text[{t_idx}]",
                            lt.get_window_extent(renderer),
                        )
                    )

    for items in (macro_items, fine_items):
        for i in range(len(items)):
            name_i, bbox_i = items[i]
            for j in range(i + 1, len(items)):
                name_j, bbox_j = items[j]
                if bbox_i.overlaps(bbox_j):
                    raise AssertionError(
                        f"Overlapping figure artists: {name_i} {bbox_i.bounds} "
                        f"intersects {name_j} {bbox_j.bounds}"
                    )


def make_figure_1(data: dict, out_path: Path) -> None:
    """Plot 3 longitudinal phase-space (z, u_z/c) frames showing ballistic bunching."""
    ps = data["ps"]
    frames = ps["frames"]
    times_ps = ps["times"] * 1.0e12  # [ps]
    z_edges_mm = ps["x_edges"] * 1.0e3
    uz_edges_c = ps["y_edges"] / c_light

    n_f = frames.shape[0]
    indices = [0, n_f // 2, n_f - 1]
    titles = [
        f"(a) $t = {times_ps[indices[0]]:.0f}$ ps",
        f"(b) $t = {times_ps[indices[1]]:.0f}$ ps",
        f"(c) $t = {times_ps[indices[2]]:.0f}$ ps",
    ]

    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.3,
        nrows=1,
        ncols=3,
        sharey=True,
    )
    fig._wakis_custom_layout = True
    fig.subplots_adjust(left=0.09, right=0.87, bottom=0.18, top=0.80, wspace=0.18)
    fig.suptitle(
        "Ballistic velocity bunching in longitudinal phase space",
        y=0.96,
    )

    sel_frames = frames[indices]
    pos_vals = sel_frames[sel_frames > 0.0]
    vmax = float(np.percentile(pos_vals, 95.0)) if pos_vals.size > 0 else 1.0
    if vmax <= 0.0:
        vmax = 1.0

    uz_occ = np.flatnonzero(np.sum(sel_frames, axis=(0, 1)) > 0.0)
    if uz_occ.size > 0:
        uz_lo = float(uz_edges_c[int(uz_occ[0])])
        uz_hi = float(uz_edges_c[int(uz_occ[-1]) + 1])
        uz_pad = max(0.10 * (uz_hi - uz_lo), 0.025)
        ylim_c = (
            max(float(uz_edges_c[0]), uz_lo - uz_pad),
            min(float(uz_edges_c[-1]), uz_hi + uz_pad),
        )
    else:
        ylim_c = (float(uz_edges_c[0]), float(uz_edges_c[-1]))

    for ax, f_idx, title in zip(axes, indices, titles):
        frame = frames[f_idx]
        im = ax.pcolormesh(
            z_edges_mm,
            uz_edges_c,
            np.clip(frame.T / vmax, 0.0, 1.0),
            cmap="inferno",
            shading="auto",
            vmin=0.0,
            vmax=1.0,
        )
        z_occ = np.flatnonzero(np.sum(frame, axis=1) > 0.0)
        if z_occ.size > 0:
            z_lo = float(z_edges_mm[int(z_occ[0])])
            z_hi = float(z_edges_mm[int(z_occ[-1]) + 1])
            z_pad = max(0.14 * (z_hi - z_lo), 5.0)
            ax.set_xlim(
                max(float(z_edges_mm[0]), z_lo - z_pad),
                min(float(z_edges_mm[-1]), z_hi + z_pad),
            )
        ax.set_ylim(*ylim_c)
        ax.set_xlabel(r"Position $z$ [mm]")
        ax.set_title(title)
        figstyle.spectrum_grid(ax)

    axes[0].set_ylabel(r"Normalized proper velocity $u_z / c$")
    cax = fig.add_axes([0.89, 0.18, 0.02, 0.62])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Normalized phase-space density")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figstyle.save(fig, out_path, auto_legend_below=False)
    fig_warnings = [
        w
        for w in caught
        if "figstyle" in str(w.filename) or issubclass(w.category, UserWarning)
    ]
    assert not fig_warnings, f"Unexpected figure warnings: {fig_warnings}"
    _assert_no_text_overlap(fig)
    plt.close(fig)


def make_figure_2(data: dict, out_path: Path) -> None:
    """Plot (a) I(t) at two PlaneCurrentProbes and (b) EnergySpectrumMonitor histogram."""
    p1 = data["p1"]
    p2 = data["p2"]
    es = data["es"]
    cfg = data["config"]

    fig, axes = figstyle.new_figure(
        width_in=figstyle.FULL_WIDTH_IN,
        height_in=3.4,
        nrows=1,
        ncols=2,
    )
    fig._wakis_custom_layout = True
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.88, wspace=0.34)
    ax_cur, ax_eng = axes[0], axes[1]

    t_ps = p1["t"] * 1.0e12
    i1_mA = np.abs(p1["I"]) * 1.0e3
    i2_mA = np.abs(p2["I"]) * 1.0e3

    st1 = figstyle.series_style(0, with_marker=False, lw=1.9)
    st2 = figstyle.series_style(1, with_marker=False, lw=1.9)
    ax_cur.plot(
        t_ps,
        i1_mA,
        label=rf"$z_1 = {cfg['z_probe_1_m'] * 1e3:.0f}$ mm ($Q_1 = {abs(p1['Q_total']) * 1e12:.2f}$ pC)",
        **st1,
    )
    ax_cur.plot(
        t_ps,
        i2_mA,
        label=rf"$z_2 = {cfg['z_probe_2_m'] * 1e3:.0f}$ mm ($Q_2 = {abs(p2['Q_total']) * 1e12:.2f}$ pC)",
        **st2,
    )
    ax_cur.set_xlabel(r"Time $t$ [ps]")
    ax_cur.set_ylabel(r"Beam current magnitude $|I(t)|$ [mA]")
    ax_cur.set_title("(a) Probe currents")
    figstyle.spectrum_grid(ax_cur)
    figstyle.place_legend_below(
        ax_cur, ncol=1, y_offset=-0.26, handlelength=2.2, handletextpad=0.6
    )

    ek_keV = es["centers_eV"] * 1.0e-3
    hist_norm = es["hist"] / max(float(np.max(es["hist"])), 1.0)
    st_e = figstyle.series_style(2, with_marker=True, markevery=3, ms=4.0, lw=1.8)
    ax_eng.plot(
        ek_keV,
        hist_norm,
        label=r"Spectrum $E_k^{n+1/2}$",
        **st_e,
    )
    ax_eng.axvline(
        cfg["ek0_eV"] * 1.0e-3,
        color=figstyle.INK,
        linestyle="--",
        marker="none",
        linewidth=1.2,
        label=rf"Drift energy $E_{{k,0}} = {cfg['ek0_eV'] * 1e-3:.1f}$ keV",
    )
    ax_eng.set_xlabel(r"Kinetic energy $E_k$ [keV]")
    ax_eng.set_ylabel("Normalized spectral weight")
    ax_eng.set_title("(b) Energy spectrum")
    figstyle.spectrum_grid(ax_eng)
    figstyle.place_legend_below(
        ax_eng, ncol=1, y_offset=-0.26, handlelength=2.2, handletextpad=0.6
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figstyle.save(fig, out_path, auto_legend_below=False)
    fig_warnings = [
        w
        for w in caught
        if "figstyle" in str(w.filename) or issubclass(w.category, UserWarning)
    ]
    assert not fig_warnings, f"Unexpected figure warnings: {fig_warnings}"
    _assert_no_text_overlap(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Example 025: Ballistic bunching and particle phase-space diagnostics."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast smoke test on a reduced grid and particle count.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure rendering.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure GPU stream time per step including host launch latency with and without all 5 monitors (20 repeats).",
    )
    parser.add_argument(
        "--benchmark-counts",
        type=int,
        nargs="+",
        default=None,
        help="Particle counts for --benchmark (default: 1000000 10000000).",
    )
    args = parser.parse_args()

    out = run_ballistic_bunching(smoke=args.smoke)
    cfg = out["config"]
    p1 = out["p1"]
    p2 = out["p2"]
    em = out["em"]
    es = out["es"]
    ps = out["ps"]

    fig1_path = IMG_DIR / "particles_phase_space_bunching.png"
    fig2_path = IMG_DIR / "particles_current_probes_and_spectrum.png"
    if not args.no_plot:
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        make_figure_1(out, fig1_path)
        make_figure_2(out, fig2_path)

    q1_C = float(p1["Q_total"])
    q2_C = float(p2["Q_total"])
    rel_q_diff = abs(q1_C - q2_C) / max(abs(q1_C), 1.0e-30)

    payload = {
        "config": cfg,
        "diagnostics_summary": {
            "phase_space_frames_recorded": int(ps["frames"].shape[0]),
            "phase_space_overflow_count": int(ps["overflow_count"]),
            "energy_spectrum_overflow_count": int(es["overflow_count"]),
            "probe_1_integrated_charge_C": q1_C,
            "probe_2_integrated_charge_C": q2_C,
            "probe_relative_charge_difference": float(rel_q_diff),
            "probe_1_peak_current_A": float(np.max(np.abs(p1["I"]))),
            "probe_2_peak_current_A": float(np.max(np.abs(p2["I"]))),
            "initial_numpy_emittance_x_m_rad": cfg["eps_x_init_numpy_m_rad"],
            "analytic_emittance_x_m_rad": cfg["eps_x_analytic_m_rad"],
            "initial_emittance_x_abs_err_m_rad": cfg["eps_x_init_abs_err_m_rad"],
            "initial_emittance_x_stat_err_m_rad": cfg["eps_x_init_stat_err_m_rad"],
            "initial_emittance_x_stat_tol_m_rad": cfg["eps_x_init_stat_tol_m_rad"],
            "final_emittance_x_m_rad": float(em["emittance_x"][-1]),
            "final_normalized_emittance_x_m_rad": float(
                em["normalized_emittance_x"][-1]
            ),
            "final_momentum_emittance_x_m_rad": float(em["momentum_emittance_x"][-1]),
            "numpy_emittance_x_m_rad": cfg["eps_x_numpy_m_rad"],
            "numpy_normalized_emittance_x_m_rad": cfg["norm_eps_x_numpy_m_rad"],
            "numpy_momentum_emittance_x_m_rad": cfg["eps_n_u_x_numpy_m_rad"],
            "monitor_vs_numpy_emittance_x_diff_m_rad": cfg[
                "mon_vs_np_eps_x_diff_m_rad"
            ],
            "monitor_vs_numpy_normalized_emittance_x_diff_m_rad": cfg[
                "mon_vs_np_norm_eps_x_diff_m_rad"
            ],
            "monitor_vs_numpy_momentum_emittance_x_diff_m_rad": cfg[
                "mon_vs_np_eps_n_u_x_diff_m_rad"
            ],
            "emittance_x_roundoff_tol_m_rad": cfg["eps_x_roundoff_tol_m_rad"],
            "normalized_emittance_x_roundoff_tol_m_rad": cfg[
                "norm_eps_x_roundoff_tol_m_rad"
            ],
            "momentum_emittance_x_roundoff_tol_m_rad": cfg[
                "eps_n_u_x_roundoff_tol_m_rad"
            ],
            "emittance_x_stat_err_m_rad": cfg["eps_x_stat_err_m_rad"],
            "n_eff": cfg["n_eff"],
        },
        "figures": [
            str(fig1_path.relative_to(REPO_ROOT)),
            str(fig2_path.relative_to(REPO_ROOT)),
        ],
    }
    if args.benchmark:
        bench_counts = (
            tuple(int(n) for n in args.benchmark_counts)
            if args.benchmark_counts is not None
            else (1_000_000, 10_000_000)
        )
        payload["benchmark"] = run_gpu_benchmark(counts=bench_counts)

    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[Example 025] Q1={q1_C:.6e} C, Q2={q2_C:.6e} C, "
        f"|Q1-Q2|/|Q1|={rel_q_diff:.3e}, "
        f"I1_peak={payload['diagnostics_summary']['probe_1_peak_current_A'] * 1e3:.3f} mA, "
        f"I2_peak={payload['diagnostics_summary']['probe_2_peak_current_A'] * 1e3:.3f} mA, "
        f"eps_x_init={cfg['eps_x_init_numpy_m_rad']:.6e} m*rad "
        f"(analytic={cfg['eps_x_analytic_m_rad']:.6e} m*rad, "
        f"3.5*SE={cfg['eps_x_init_stat_tol_m_rad']:.3e} m*rad), "
        f"eps_x_final={em['emittance_x'][-1]:.6e} m*rad "
        f"(numpy={cfg['eps_x_numpy_m_rad']:.6e} m*rad, "
        f"|diff|={cfg['mon_vs_np_eps_x_diff_m_rad']:.3e} <= "
        f"roundoff_tol={cfg['eps_x_roundoff_tol_m_rad']:.3e} m*rad)"
    )
    if args.benchmark:
        for k, b in payload["benchmark"].items():
            print(
                f"[Example 025 Benchmark {k}] ({b['timing_label']}, "
                f"GPU={b['gpu_name']}, grid={tuple(b['grid_shape'])}, "
                f"sort_every={b['sort_every']}, warmup={b['n_warmup']}, "
                f"repeats={b['n_repeats']}) "
                f"without_monitors: median={b['without_monitors_median_ms']:.3f} ms "
                f"(min-max={b['without_monitors_min_ms']:.3f}-{b['without_monitors_max_ms']:.3f} ms), "
                f"with_all_monitors: median={b['with_all_monitors_median_ms']:.3f} ms "
                f"(min-max={b['with_all_monitors_min_ms']:.3f}-{b['with_all_monitors_max_ms']:.3f} ms), "
                f"overhead_median={b['monitors_overhead_median_ms']:.3f} ms"
            )


if __name__ == "__main__":
    main()
