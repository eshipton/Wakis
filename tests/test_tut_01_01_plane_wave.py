# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Analytical, Wakis 1D/2D/3D FIT solver, and mutation tests for Tutorial 1.1:
`tutorials/01_fields_on_a_grid/01-01_plane_wave/`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as c0

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "01_fields_on_a_grid"
    / "01-01_plane_wave"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

hlp = _load_module("_tut_01_01_helpers", TUTORIAL_DIR / "_helpers.py")
pw = _load_module("_tut_01_01_plane_wave", TUTORIAL_DIR / "plane_wave.py")


def _assert_task_resolution_rules(meta: dict, full_mode: bool = False) -> None:
    """Enforce project grid resolution rules on a task metadata dictionary."""
    min_cpw = 10
    assert int(meta["cpw"]) >= min_cpw, (
        f"Rule 1 violated: cpw={meta['cpw']} < {min_cpw}"
    )
    assert float(meta["cells_across_feature"]) >= 4.0 - 1e-6, (
        f"Rule 2 violated: cells_across_feature={meta['cells_across_feature']} < 4.0"
    )
    assert float(meta["dx_min_nm"]) <= float(meta["min_feature_nm"]) / 4.0 + 1e-6, (
        f"Rule 2 violated: dx_min_nm={meta['dx_min_nm']} > min_feature_nm/4"
    )
    if meta.get("cells_per_skin_depth") is not None:
        assert float(meta["cells_per_skin_depth"]) >= 4.0 - 1e-6, (
            f"Rule 3 violated: cells_per_skin_depth={meta['cells_per_skin_depth']} < 4.0"
        )
    if full_mode and meta["name"] == "soft_source_and_tfsf":
        assert int(meta["cpw"]) >= 20, "Rule 1 (--full): production source task must have >= 20 cpw"


def test_analytic_dispersion_and_courant_identities():
    """Verify 1D magic step, 2D angular anisotropy, O(cpw^-2) scaling, and Courant roots."""
    # 1. At the 1D magic time step dt = dz / c0, numerical dispersion vanishes identically
    dz = hlp.LAM0 / 12.0
    dt_magic = dz / c0
    freqs = np.linspace(0.6 * hlp.F0, 1.4 * hlp.F0, 17)
    k_magic, vp_magic = hlp.discrete_dispersion_1d(freqs, dz, dt_magic)
    np.testing.assert_allclose(vp_magic, 1.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(k_magic, 2.0 * np.pi * freqs / c0, rtol=1e-14)

    # 2. In 2D at S_3D = 0.5, phase velocity is slower along axes (0, 90 deg) than along 45 deg diagonal
    dx = hlp.LAM0 / 20.0
    dt_3d = 0.5 / (c0 * np.sqrt(3.0 / dx**2))
    _, vp_ang = hlp.discrete_dispersion_2d_angle(
        hlp.F0, dx, dt_3d, np.deg2rad([0.0, 45.0, 90.0])
    )
    assert vp_ang[0] == pytest.approx(vp_ang[2], rel=1e-14)
    assert vp_ang[0] < vp_ang[1] < 1.0
    assert abs(1.0 - vp_ang[0]) > 1.7 * abs(1.0 - vp_ang[1])

    # 3. Theoretical O(cpw^-2) convergence slope across cpw = [10, 20, 40, 80]
    cpws = np.array([10.0, 20.0, 40.0, 80.0])
    errs_0 = []
    for c_val in cpws:
        h = hlp.LAM0 / c_val
        dt_h = 0.5 / (c0 * np.sqrt(3.0 / h**2))
        _, vp0 = hlp.discrete_dispersion_2d_angle(hlp.F0, h, dt_h, 0.0)
        errs_0.append(abs(1.0 - float(vp0[0])))
    slope = float(np.polyfit(np.log(cpws), np.log(errs_0), 1)[0])
    assert slope == pytest.approx(-2.0, abs=0.03)

    # 4. Courant amplification factor: |lambda| == 1 for S <= 1, > 1 for S > 1
    s_sub = np.array([0.2, 0.5, 0.99, 1.0])
    np.testing.assert_allclose(hlp.courant_amplification_factor(s_sub), 1.0, atol=1e-15)
    s_sup = 1.02
    g_expected = (s_sup + np.sqrt(s_sup**2 - 1.0)) ** 2
    assert float(hlp.courant_amplification_factor(s_sup)[0]) == pytest.approx(
        g_expected, rel=1e-14
    )


def test_soft_source_normalization_and_tfsf_leakage():
    """Verify Wakis 1D FIT soft source normalization, zero-DC pulse, and TFSFSource cancellation."""
    res = hlp.run_soft_source_and_tfsf_task(
        {"name": "soft_source_and_tfsf", "cpw": 20}
    )
    # Exact discrete vector identity S @ C == 0
    assert float(res["c_duality_diff"]) == 0.0
    # Zero-DC integral of derivative Gaussian pulse
    assert float(res["dc_integral_rel"]) < 1e-4
    # Preflight check on 2 GB laptop profile passes
    assert int(res["preflight_ok"]) == 1
    # Source-normalized transfer impedance |Ex(z, f) / S(f)| matches discrete sheet formula to < 0.3%
    assert float(res["norm_rel_err"]) < 3.0e-3
    # TFSFSource exterior scattered-field leakage is at machine precision (< 1e-14)
    assert float(res["tfsf_leakage"]) < 1.0e-14


def test_courant_stability_and_supercritical_blowup():
    """Verify bounded fields for S <= 1 and exact exponential blow-up for S > 1."""
    res = hlp.run_courant_stability_task({"name": "courant_stability", "cpw": 20})
    s_list = res["s_list"]
    g_sim = res["g_sim"]
    g_th = res["g_th"]

    sub_mask = s_list <= 1.0
    sup_mask = s_list > 1.0
    # Subcritical runs remain bounded
    np.testing.assert_allclose(g_sim[sub_mask], 1.0, atol=0.03)
    # Supercritical runs blow up with growth factor matching (S + sqrt(S^2 - 1))^2 within 1%
    np.testing.assert_allclose(g_sim[sup_mask], g_th[sup_mask], rtol=0.01)
    # Unstable spatial profile exhibits period-2 Nyquist checkerboard alternation (-1)^k
    prof = res["unstable_profile_z"]
    alternating_signs = prof[1:-1] * prof[2:]
    assert float(np.mean(alternating_signs < 0)) > 0.85


def test_committed_full_study_data_and_figures():
    """Verify committed data/task_0..7.{json,npz} and figures/01..06 satisfy all accuracy & grid rules."""
    full_tasks = pw.get_task_table(quick=False)
    assert len(full_tasks) == 8

    for t_cfg in full_tasks:
        idx = int(t_cfg["index"])
        json_p = TUTORIAL_DIR / "data" / f"task_{idx}.json"
        npz_p = TUTORIAL_DIR / "data" / f"task_{idx}.npz"
        assert json_p.is_file(), f"Missing {json_p}"
        assert npz_p.is_file(), f"Missing {npz_p}"
        meta = json.loads(json_p.read_text())
        assert meta["task_index"] == idx
        assert meta["name"] == t_cfg["name"]
        _assert_task_resolution_rules(meta, full_mode=True)

    # Check dispersion tasks (tasks 2..7: cpw = 10, 14, 20, 28, 40, 60)
    cpws = []
    err_0 = []
    err_45 = []
    for idx in range(2, 8):
        d = np.load(TUTORIAL_DIR / "data" / f"task_{idx}.npz")
        cpws.append(float(d["cpw"]))
        assert float(d["max_err_1d_vs_th"]) < 2.0e-4
        assert float(d["max_err_2d_vs_th"]) < (6.5e-4 if int(d["cpw"]) == 10 else 2.0e-4)
        # Bloch fixed-k eigenmode matches fixed-omega 2D discrete dispersion up to O(cpw^-4)
        np.testing.assert_allclose(
            d["vp_bloch_2d"],
            d["vp_th_2d"],
            rtol=5.5e-4 if int(d["cpw"]) == 10 else 1.5e-4,
        )
        err_0.append(abs(1.0 - float(d["vp_0deg_sim"])))
        err_45.append(abs(1.0 - float(d["vp_45deg_sim"])))

    slope_0 = float(np.polyfit(np.log(cpws), np.log(err_0), 1)[0])
    slope_45 = float(np.polyfit(np.log(cpws), np.log(err_45), 1)[0])
    assert slope_0 == pytest.approx(-2.0, abs=0.06)
    assert slope_45 == pytest.approx(-2.0, abs=0.08)

    # At 20 cpw (task_4), axial phase-velocity error is ~0.38% (< 0.42%) and diagonal < 0.20%,
    # while the under-resolved 6 cpw warning point on task_7 has > 3.0% error
    t_20 = np.load(TUTORIAL_DIR / "data" / "task_4.npz")
    assert abs(1.0 - float(t_20["vp_0deg_sim"])) < 0.0042
    assert abs(1.0 - float(t_20["vp_45deg_sim"])) < 0.0020
    t_ref = np.load(TUTORIAL_DIR / "data" / "task_7.npz")
    assert abs(1.0 - float(t_ref["demo_vp_0deg"])) > 0.030

    expected_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "01_fit_grid_and_source_pulse.png",
        "02_soft_source_and_normalization.png",
        "03_courant_stability_limit.png",
        "04_numerical_dispersion_1d_and_2d.png",
        "05_resolution_convergence_20cpw_rule.png",
        "06_plane_wave_propagation.gif",
    ]
    for fname in expected_figs:
        fpath = TUTORIAL_DIR / "figures" / fname
        assert fpath.is_file() and fpath.stat().st_size > 1000, f"Missing or empty figure {fpath}"


def test_mutation_checks_provably_fail():
    """Mutation checks: omitting S(f) normalization, continuum k0 at coarse cpw, S > 1, or < 10 cpw MUST fail."""
    t0 = np.load(TUTORIAL_DIR / "data" / "task_0.npz")
    # Mutation 1: Using raw unnormalized DFT amplitude instead of dividing by S(f)
    # leaves the Gaussian spectral envelope, failing the flat impedance check
    raw_rel_variation = float(
        (np.max(t0["amp_raw"]) - np.min(t0["amp_raw"])) / np.mean(t0["amp_raw"])
    )
    norm_rel_variation = float(
        (np.max(t0["amp_sim"]) - np.min(t0["amp_sim"])) / np.mean(t0["amp_sim"])
    )
    assert norm_rel_variation < 0.02
    with pytest.raises(AssertionError):
        assert raw_rel_variation < 0.02

    # Mutation 2: Comparing measured 10 cpw phase velocity against continuum v_p/c_0 = 1.0
    # instead of the discrete dispersion relation MUST fail the 2e-4 accuracy tolerance
    t_10 = np.load(TUTORIAL_DIR / "data" / "task_2.npz")
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(t_10["vp_sim_1d"], 1.0, atol=2.0e-4)

    # Mutation 3: Supercritical Courant number S = 1.05 MUST fail the stability bound |lambda| <= 1.03
    t_cfl = np.load(TUTORIAL_DIR / "data" / "task_1.npz")
    idx_105 = int(np.where(np.isclose(t_cfl["s_list"], 1.05))[0][0])
    with pytest.raises(AssertionError):
        assert float(t_cfl["g_sim"][idx_105]) <= 1.03

    # Mutation 4: Task metadata violating Rule 1 (cpw = 6 < 10) or Rule 2 (cells_across_feature = 2 < 4) MUST fail
    bad_meta_cpw = {
        "name": "bad_cpw",
        "cpw": 6,
        "cells_across_feature": 6.0,
        "dx_min_nm": 166.7,
        "min_feature_nm": 1000.0,
    }
    with pytest.raises(AssertionError, match="Rule 1"):
        _assert_task_resolution_rules(bad_meta_cpw)

    bad_meta_feat = {
        "name": "bad_feat",
        "cpw": 12,
        "cells_across_feature": 2.5,
        "dx_min_nm": 80.0,
        "min_feature_nm": 200.0,
    }
    with pytest.raises(AssertionError, match="Rule 2"):
        _assert_task_resolution_rules(bad_meta_feat)
