# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Analytical Fresnel/Airy transfer-matrix, Wakis two-run FluxMonitor normalization,
CPML reflection, Kottke subpixel convergence, and mutation tests for Tutorial 1.2:
`tutorials/01_fields_on_a_grid/01-02_slab_rt/`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wakis.templates.formulas import slab_reflection_transmission

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "01_fields_on_a_grid"
    / "01-02_slab_rt"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

hlp = _load_module("_tut_01_02_helpers", TUTORIAL_DIR / "_helpers.py")
sr = _load_module("_tut_01_02_slab_rt", TUTORIAL_DIR / "slab_rt.py")


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
    if full_mode and meta["name"] == "fabry_perot_and_oblique":
        assert int(meta["cpw"]) >= 20, (
            "Rule 1 (--full): production Fabry-Perot task must have >= 20 cpw in slab"
        )


def test_analytic_fresnel_brewster_and_transfer_matrix():
    """Verify Fresnel single-interface formulas, Brewster's angle, and Airy slab identities."""
    # 1. Normal incidence at n1=1 -> n2=2: r = -1/3, R = 1/9, T = 8/9, R + T = 1
    r0_s, t0_s, R0_s, T0_s = hlp.fresnel_single_interface(1.0, hlp.N_SLAB, 0.0, "s")
    r0_p, t0_p, R0_p, T0_p = hlp.fresnel_single_interface(1.0, hlp.N_SLAB, 0.0, "p")
    assert float(np.real(r0_s[0])) == pytest.approx(-1.0 / 3.0, rel=1e-14)
    assert float(R0_s[0]) == pytest.approx(1.0 / 9.0, rel=1e-14)
    assert float(T0_s[0]) == pytest.approx(8.0 / 9.0, rel=1e-14)
    assert float(R0_p[0]) == pytest.approx(1.0 / 9.0, rel=1e-14)
    assert float(R0_s[0] + T0_s[0]) == pytest.approx(1.0, rel=1e-14)

    # 2. Brewster's angle theta_B = arctan(n2/n1): p/TM reflectance vanishes identically
    theta_b = np.arctan(hlp.N_SLAB / 1.0)
    _, _, Rb_p, Tb_p = hlp.fresnel_single_interface(1.0, hlp.N_SLAB, theta_b, "p")
    assert float(Rb_p[0]) == pytest.approx(0.0, abs=1e-15)
    assert float(Tb_p[0]) == pytest.approx(1.0, rel=1e-14)

    # 3. Oblique transfer-matrix matches Wakis slab_reflection_transmission at theta=0
    freqs = np.linspace(0.65 * hlp.F0, 1.35 * hlp.F0, 41)
    d_slab = 1.25 * (hlp.LAM0 / hlp.N_SLAB)
    R_wk, T_wk = slab_reflection_transmission(hlp.EPS_SLAB, d_slab, freqs)
    ob_s = hlp.slab_rt_oblique(hlp.EPS_SLAB, d_slab, freqs, 0.0, "s")
    ob_p = hlp.slab_rt_oblique(hlp.EPS_SLAB, d_slab, freqs, 0.0, "p")
    np.testing.assert_allclose(ob_s["R"], R_wk, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(ob_s["T"], T_wk, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(ob_p["R"], R_wk, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(ob_s["R"] + ob_s["T"], 1.0, atol=1e-14)

    # 4. Quarter-wave maximum R_max = ((n^2 - 1)/(n^2 + 1))^2 = 0.36 and half-wave R_min = 0
    R_qw, T_qw = slab_reflection_transmission(
        hlp.EPS_SLAB, 0.25 * (hlp.LAM0 / hlp.N_SLAB), np.array([hlp.F0])
    )
    R_hw, T_hw = slab_reflection_transmission(
        hlp.EPS_SLAB, 0.50 * (hlp.LAM0 / hlp.N_SLAB), np.array([hlp.F0])
    )
    assert float(R_qw[0]) == pytest.approx(0.36, rel=1e-14)
    assert float(T_qw[0]) == pytest.approx(0.64, rel=1e-14)
    assert float(R_hw[0]) == pytest.approx(0.0, abs=1e-15)
    assert float(T_hw[0]) == pytest.approx(1.0, rel=1e-14)

    # 5. At Brewster's angle, a lossless dielectric slab of any thickness has R_p = 0, T_p = 1
    for mult in (0.19, 0.375, 1.13):
        ob_b = hlp.slab_rt_oblique(
            hlp.EPS_SLAB, mult * (hlp.LAM0 / hlp.N_SLAB), freqs, theta_b, "p"
        )
        np.testing.assert_allclose(ob_b["R"], 0.0, atol=1e-14)
        np.testing.assert_allclose(ob_b["T"], 1.0, atol=1e-14)


def test_wakis_offgrid_subpixel_vs_staircase_live():
    """Run a fast live Wakis off-grid slab pair at 14 cpw and verify Kottke beats staircase."""
    res_k = hlp._eval_offgrid_slab(14, "kottke")
    res_s = hlp._eval_offgrid_slab(14, "staircase")
    # Kottke subpixel smoothing reduces broadband RMS error by > 4x already at 14 cpw
    assert res_k["rms_err"] < 0.025
    assert res_s["rms_err"] > 0.080
    assert res_k["rms_err"] < 0.25 * res_s["rms_err"]
    # Discrete Poynting flux conservation R(f) + T(f) = 1 holds to < 1e-4
    assert res_k["energy_err"] < 1.0e-4
    assert res_s["energy_err"] < 1.0e-4


def test_committed_full_study_data_and_figures():
    """Verify committed data/task_0..13.{json,npz} and figures/01..06 satisfy all accuracy & grid rules."""
    full_tasks = sr.get_task_table(quick=False)
    assert len(full_tasks) == 14

    for t_cfg in full_tasks:
        idx = int(t_cfg["index"])
        json_p = TUTORIAL_DIR / "data" / f"task_{idx}.json"
        npz_p = TUTORIAL_DIR / "data" / f"task_{idx}.npz"
        assert json_p.is_file(), f"Missing {json_p}"
        assert npz_p.is_file(), f"Missing {npz_p}"
        meta = json.loads(json_p.read_text())
        assert meta["index"] == idx
        assert meta["name"] == t_cfg["name"]
        _assert_task_resolution_rules(meta, full_mode=True)

    # 1. Task 0: Fabry-Perot fringes + oblique Bloch-periodic sweep + Brewster's angle
    t0 = np.load(TUTORIAL_DIR / "data" / "task_0.npz")
    assert float(t0["fp_rms_err"]) < 7.5e-3
    assert float(t0["fp_energy_err"]) < 1.0e-5
    np.testing.assert_allclose(t0["R_flux"], t0["R_dft"], atol=2.0e-3)
    np.testing.assert_allclose(t0["T_flux"], t0["T_dft"], atol=2.0e-3)
    # Brewster's angle (theta_B = 63.435 deg): p/TM reflectance < 2e-3 while s/TE reflectance ~ 0.722
    assert float(t0["brewster_Rp_sim"]) < 2.0e-3
    assert float(t0["brewster_Tp_sim"]) == pytest.approx(1.0, abs=2.0e-3)
    assert float(t0["brewster_Rs_sim"]) == pytest.approx(
        float(t0["brewster_Rs_ex"]), abs=5.0e-3
    )
    assert float(t0["oblique_max_energy_err"]) < 8.0e-3

    # 2. Task 1: CPML spurious reflection vs layer count & grading order vs PEC / Mur ABC
    t1 = np.load(TUTORIAL_DIR / "data" / "task_1.npz")
    f_mid = len(t1["freqs"]) // 2
    assert float(t1["r_db_pec"][f_mid]) == pytest.approx(0.0, abs=0.05)
    assert -55.0 < float(t1["r_db_abc"][f_mid]) < -40.0
    r_npml = t1["r_db_vs_npml_f0"]
    # Reflection drops monotonically with N_pml = [4, 6, 8, 10, 12, 16]
    assert np.all(np.diff(r_npml) < -5.0)
    # For N_pml >= 8 (index >= 2), spurious reflection is below -60 dB; at N_pml = 16, below -80 dB
    assert float(r_npml[2]) < -60.0
    assert float(r_npml[3]) < -68.0
    assert float(r_npml[-1]) < -84.0

    # 3. Tasks 2..13: Kottke O(dz^2) vs Staircase O(dz^1) off-grid convergence
    cpw_k, rms_k = [], []
    cpw_s, rms_s = [], []
    for idx in range(2, 14):
        d = np.load(TUTORIAL_DIR / "data" / f"task_{idx}.npz")
        assert float(d["energy_err"]) < 1.0e-4
        if str(d["averaging"]) == "kottke":
            cpw_k.append(float(d["cpw"]))
            rms_k.append(float(d["rms_err"]))
        else:
            cpw_s.append(float(d["cpw"]))
            rms_s.append(float(d["rms_err"]))

    slope_k = float(np.polyfit(np.log(cpw_k), np.log(rms_k), 1)[0])
    slope_s = float(np.polyfit(np.log(cpw_s), np.log(rms_s), 1)[0])
    # Kottke achieves second-order convergence (slope <= -1.9), beating staircase (slope ~ -1.58)
    assert slope_k < -1.90
    assert -1.70 < slope_s < -0.80
    assert slope_k < slope_s - 0.65
    # At 20 cpw (index 2) and 60 cpw (index 5), Kottke beats staircase by > 4x and > 9x
    assert rms_k[2] < 0.007
    assert rms_k[2] < rms_s[2] / 4.0
    assert rms_k[4] < rms_s[4] / 12.0
    assert rms_k[-1] < rms_s[-1] / 9.0

    expected_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00c_setup_oblique_and_bc_runs.png",
        "01_two_run_normalization_and_setup.png",
        "02_fabry_perot_and_oblique_fresnel.png",
        "03_cpml_reflection_benchmark.png",
        "04_subpixel_permittivity_and_spectra.png",
        "05_subpixel_convergence_study.png",
        "06_slab_fabry_perot_etalon.gif",
    ]
    for fname in expected_figs:
        fpath = TUTORIAL_DIR / "figures" / fname
        assert fpath.is_file() and fpath.stat().st_size > 1000, f"Missing or empty figure {fpath}"


def test_mutation_checks_provably_fail():
    """Mutation checks: forgetting reference subtraction, PEC boundary, staircase vs O(dz^2), or <10 cpw MUST fail."""
    t0 = np.load(TUTORIAL_DIR / "data" / "task_0.npz")
    # Mutation 1: Using raw net flux ratio Phi_r / Phi_{r,ref} as R instead of 1 - Phi_r / Phi_{r,ref}
    # violates both R_exact and energy conservation (since Phi_r = Phi_inc - Phi_refl)
    r_wrong = 1.0 - t0["R_flux"]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(r_wrong, t0["R_exact"], atol=0.05)

    # Mutation 2: Swapping s/TE and p/TM reflectance at Brewster's angle MUST fail the R_p < 2e-3 check
    with pytest.raises(AssertionError):
        assert float(t0["brewster_Rs_sim"]) < 2.0e-3

    # Mutation 3: Replacing CPML with a PEC wall MUST fail the < -60 dB boundary reflection requirement
    t1 = np.load(TUTORIAL_DIR / "data" / "task_1.npz")
    f_mid = len(t1["freqs"]) // 2
    with pytest.raises(AssertionError):
        assert float(t1["r_db_pec"][f_mid]) < -60.0

    # Mutation 4: Staircase averaging at 40 cpw (task_11) MUST fail the Kottke second-order accuracy bound (< 1.5e-3)
    t_stair_40 = np.load(TUTORIAL_DIR / "data" / "task_11.npz")
    with pytest.raises(AssertionError):
        assert float(t_stair_40["rms_err"]) < 1.5e-3

    # Mutation 5: Grid resolution rules MUST reject cpw < 10 or cells_across_feature < 4
    with pytest.raises(AssertionError, match="Rule 1"):
        _assert_task_resolution_rules(
            {
                "name": "bad_cpw",
                "cpw": 6,
                "cells_across_feature": 5.0,
                "dx_min_nm": 50.0,
                "min_feature_nm": 250.0,
            }
        )
    with pytest.raises(AssertionError, match="Rule 2"):
        _assert_task_resolution_rules(
            {
                "name": "bad_feature",
                "cpw": 14,
                "cells_across_feature": 3.0,
                "dx_min_nm": 50.0,
                "min_feature_nm": 150.0,
            }
        )
