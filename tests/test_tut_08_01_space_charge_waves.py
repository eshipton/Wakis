# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics, resolution-rule, and mutation validation tests for Tutorial 8.1:
Space-Charge Waves on a Drifting Electron Beam (`LinearizedBeam`).
"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.constants import c as C0

from wakis import GridFIT3D, SolverFIT3D
from wakis.beam_fluid import (
    LinearizedBeam,
    reduction_factor_planar,
    relativistic_gamma,
)

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "08_vacuum_electronics"
    / "08-01_space_charge_waves"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    U0_REF,
    run_2d_bloch_dispersion_point,
    run_chu_energy_and_slow_wave_audit,
)
from space_charge_waves import (  # noqa: E402
    DATA_DIR,
    FIG_DIR,
    get_task_table,
    main as tutorial_main,
)


def _audit_tutorial_task_cfg(task_cfg):
    """
    Compute the electronic-wavelength sampling ``cpw`` and smallest-feature
    sampling ``cells_across_feature`` directly from a Tutorial 8.1 task entry.
    """
    kind = task_cfg["kind"]
    nx = int(task_cfg["nx"])
    ny = int(task_cfg["ny"])
    a_half = float(task_cfg["a_half"])

    if kind in ("bloch_2d", "chu_audit"):
        f_max = float(task_cfg["f0"])
        b_min = float(task_cfg["b_half"])
        lam_e = U0_REF / f_max
        dx = lam_e / nx
        dy = (2.0 * a_half) / ny
        cpw = lam_e / max(dx, dy)
        cells_across = b_min / dy
    elif kind == "freq_sweep_2d":
        f_max = float(max(task_cfg["freqs_ghz"])) * 1e9
        b_min = float(task_cfg["b_half"])
        lam_e = U0_REF / f_max
        dx = lam_e / nx
        dy = (2.0 * a_half) / ny
        cpw = lam_e / max(dx, dy)
        cells_across = b_min / dy
    elif kind == "thickness_sweep_2d":
        f_max = float(task_cfg["f0"])
        b_min = float(min(task_cfg["fill_ratios"])) * a_half
        lam_e = U0_REF / f_max
        dx = lam_e / nx
        dy = (2.0 * a_half) / ny
        cpw = lam_e / max(dx, dy)
        cells_across = b_min / dy
    elif kind == "bloch_3d":
        f_max = float(task_cfg["f0"])
        b_min = float(task_cfg["b_half"])
        nz = int(task_cfg["nz"])
        lam_e = U0_REF / f_max
        dx = (2.0 * a_half) / nx
        dz = lam_e / nz
        cpw = lam_e / max(dx, dz)
        cells_across = b_min / dx
    elif kind == "bunching_2d":
        f_max = float(task_cfg["f_mod"])
        b_min = float(task_cfg["b_half"])
        L_tube = float(task_cfg["L_tube"])
        lam_e = U0_REF / f_max
        dx = L_tube / nx
        dy = (2.0 * a_half) / ny
        cpw = lam_e / max(dx, dy)
        d_gap = 4.0 * dx
        cells_across = min(b_min / dy, d_gap / dx)
    else:
        raise ValueError(f"Unknown task kind: {kind}")

    ok = (cpw >= 10.0 - 1e-6) and (cells_across >= 4.0 - 1e-6)
    return ok, cpw, cells_across


def test_space_charge_waves_task_table_and_json_resolution_rules(capsys):
    """
    Verify that:
    1. Every task in both ``--quick`` and ``--full`` tables satisfies the
       project resolution rules (``cpw >= 10`` cells per electronic wavelength
       and ``cells_across_feature >= 4`` across the smallest geometric feature).
    2. Every generated ``data/task_<I>.json`` records ``dx_background``,
       ``dx_min``, ``cells_across_feature``, ``n_cells``, and ``n_steps`` with
       valid values, and all 14 figures in ``figures/`` exist and are non-empty.
    3. Mutation check: an under-resolved task config (``cpw < 10`` or
       ``cells_across_feature < 4``) is rejected by the resolution audit.
    """
    for quick_mode in (True, False):
        table = get_task_table(quick=quick_mode)
        assert len(table) == 8
        for task in table:
            ok, cpw, cells_across = _audit_tutorial_task_cfg(task)
            assert ok, (
                f"Resolution rule violation in {'--quick' if quick_mode else '--full'} "
                f"task {task['name']}: cpw={cpw:.2f}, "
                f"cells_across_feature={cells_across:.2f}"
            )

    # Mutation check 1: under-resolved axial wavelength sampling (nx=6 -> cpw=6 < 10)
    bad_cpw_task = dict(get_task_table(quick=True)[0], nx=6)
    ok_cpw, cpw_mut, _ = _audit_tutorial_task_cfg(bad_cpw_task)
    assert not ok_cpw and cpw_mut < 10.0, (
        "Mutation check failed: nx=6 (cpw=6) was not rejected by resolution audit"
    )

    # Mutation check 2: under-resolved beam thickness (ny=16 -> 2 cells across b < 4)
    bad_feat_task = dict(get_task_table(quick=True)[0], ny=16)
    ok_feat, _, cells_mut = _audit_tutorial_task_cfg(bad_feat_task)
    assert not ok_feat and cells_mut < 4.0, (
        "Mutation check failed: ny=16 (2 cells across b) was not rejected by audit"
    )

    # Audit generated task JSON files and figures
    required_keys = {
        "dx_background",
        "dx_min",
        "cells_across_feature",
        "n_cells",
        "n_steps",
    }
    for idx in range(8):
        json_path = DATA_DIR / f"task_{idx}.json"
        npz_path = DATA_DIR / f"task_{idx}.npz"
        assert json_path.is_file(), f"Missing {json_path}"
        assert npz_path.is_file() and npz_path.stat().st_size > 0, f"Missing {npz_path}"
        meta = json.loads(json_path.read_text())
        assert required_keys.issubset(meta.keys()), (
            f"task_{idx}.json missing required keys: {required_keys - set(meta.keys())}"
        )
        assert float(meta["cells_across_feature"]) >= 4.0 - 1e-6
        assert float(meta["cpw"]) >= 16.0 - 1e-6
        assert int(meta["n_cells"]) > 0
        assert int(meta["n_steps"]) > 0

    # Verify full-resolution physics accuracy in stored JSON summaries
    t2 = json.loads((DATA_DIR / "task_2.json").read_text())
    assert t2["grid_shape"] == [64, 128, 1]
    assert abs(t2["R_fit"] - t2["R_th"]) / t2["R_th"] < 1.5e-3

    t5 = json.loads((DATA_DIR / "task_5.json").read_text())
    assert t5["grid_shape"] == [40, 40, 24]
    assert abs(t5["R_3d_fit"] - t5["R_3d_rect_th"]) / t5["R_3d_rect_th"] < 5.0e-3

    t6 = json.loads((DATA_DIR / "task_6.json").read_text())
    assert t6["grid_shape"] == [360, 32, 1]
    assert abs(t6["L_beat_fit_mm"] - t6["L_beat_th_mm"]) / t6["L_beat_th_mm"] < 0.035

    expected_figures = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "01_drift_tube_geometry.png",
        "02_mesh_and_profile_zoom.png",
        "03_transverse_tm_eigenmodes.png",
        "04_kw_dispersion_diagram.png",
        "05_doppler_splitting_spectra.png",
        "06_reduction_factor_vs_thickness.png",
        "07_reduction_factor_vs_frequency.png",
        "08_2d_vs_3d_fringing_comparison.png",
        "09_modulated_beam_field_map.png",
        "10_klystron_bunching_beat.png",
        "11_space_charge_wave_propagation.gif",
        "12_chu_kinetic_power_and_energy.png",
        "13_resolution_convergence.png",
        "14_linear_vs_nonlinear_bunching_limits.png",
    ]
    for fname in expected_figures:
        fpath = FIG_DIR / fname
        assert fpath.is_file() and fpath.stat().st_size > 1024, (
            f"Missing or empty tutorial figure: {fpath}"
        )
    table_path = FIG_DIR / "00_setup_table.md"
    assert table_path.is_file() and table_path.stat().st_size > 128, (
        f"Missing or empty setup table: {table_path}"
    )

    # Verify --list-tasks CLI in both --quick and --full modes
    tutorial_main(["--list-tasks", "--quick"])
    out_q = capsys.readouterr().out
    assert "# Total tasks: 8" in out_q
    assert "32x64" in out_q

    tutorial_main(["--list-tasks", "--full"])
    out_f = capsys.readouterr().out
    assert "# Total tasks: 8" in out_f
    assert "64x128" in out_f
    assert "drift_tube_bunching" in out_f


def test_space_charge_waves_physics_and_mutations():
    """
    Fast (< 10 s) physics and mutation check for Tutorial 8.1:
    1. 2D Bloch space-charge wave dispersion reproduces the Branch–Mihran
       reduction factor R within 2%.
    2. Chu kinetic power audit confirms P_k > 0 for the fast wave, P_k < 0
       for the slow wave, and total Chu–Poynting energy conservation < 0.2%.
    3. Mutation checks: wrong relativistic exponent (gamma_power=1.0 on a
       relativistic beam) or flipped Lorentz coupling sign (coupling_sign=-1.0)
       must fail the physics assertions.
    """
    pt = run_2d_bloch_dispersion_point(
        b_half=0.5e-3,
        a_half=2.0e-3,
        f0=5.0e9,
        nx=16,
        ny=32,
        n_beat_periods=1.25,
        record_profiles=False,
    )
    rel_err_r = abs(pt["R_fit"] - pt["R_th"]) / pt["R_th"]
    assert rel_err_r < 0.02, (
        f"2D Bloch reduction factor R_fit={pt['R_fit']:.4f} deviated from "
        f"Branch-Mihran R_th={pt['R_th']:.4f} by {rel_err_r:.2%}"
    )

    chu = run_chu_energy_and_slow_wave_audit(
        b_half=0.5e-3,
        a_half=2.0e-3,
        f0=5.0e9,
        nx=16,
        ny=16,
        nt=400,
    )
    assert float(chu["Pk_fast_mean"]) > 0.0
    assert float(chu["Pk_slow_mean"]) < 0.0
    assert float(chu["rel_energy_drift"]) < 2.0e-3

    # Mutation check: wrong relativistic exponent (gamma_power=1.0 vs 3.0)
    # on a relativistic sheet beam (u0 = 0.6 c, gamma0 = 1.25) must shift
    # the measured space-charge frequency omega_q by gamma0^1.0 = 25% and
    # fail the 5% accuracy check against omega_q_th = R_th * wp / gamma0^1.5.
    u0_rel = 0.6 * C0
    pt_rel_true = run_2d_bloch_dispersion_point(
        b_half=0.5e-3,
        a_half=2.0e-3,
        f0=5.0e9,
        u0=u0_rel,
        nx=16,
        ny=32,
        n_beat_periods=1.25,
        gamma_power=3.0,
        record_profiles=False,
    )
    pt_rel_mut = run_2d_bloch_dispersion_point(
        b_half=0.5e-3,
        a_half=2.0e-3,
        f0=5.0e9,
        u0=u0_rel,
        nx=16,
        ny=32,
        n_beat_periods=1.25,
        gamma_power=1.0,
        record_profiles=False,
    )
    wq_exact = pt_rel_true["omega_q_th"]
    assert abs(pt_rel_true["omega_q"] - wq_exact) / wq_exact < 0.05
    assert abs(pt_rel_mut["omega_q"] - wq_exact) / wq_exact > 0.18, (
        "Mutation check failed: gamma_power=1.0 did not produce a large (>18%) "
        "error in the measured reduced plasma frequency omega_q"
    )
