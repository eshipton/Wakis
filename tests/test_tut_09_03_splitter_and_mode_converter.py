"""Tests for Tutorial 9.3: Multi-objective power splitter and mode converter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = REPO_ROOT / "tutorials" / "09_inverse_design" / "09-03_splitter_and_mode_converter"
SCRIPT_PATH = TUT_DIR / "splitter_and_mode_converter.py"
DATA_DIR = TUT_DIR / "data"
FIG_DIR = TUT_DIR / "figures"


@pytest.fixture(scope="module")
def tut_module():
    """Load Tutorial 9.3 module dynamically."""
    spec = importlib.util.spec_from_file_location("splitter_and_mode_converter", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_resolution_rules(table: list[dict]) -> bool:
    """Return True iff every task entry satisfies >=10 cells/wavelength and >=4 cells/feature."""
    return all(float(t["cpw"]) >= 10.0 and float(t["cells_across_feature"]) >= 4.0 for t in table)


def test_task_table_resolution_rules_and_mutation(tut_module):
    """Verify resolution rules on quick and full task tables plus a mutation check that fails."""
    for quick_flag in (True, False):
        table = tut_module.get_task_table(quick=quick_flag)
        assert len(table) == 5
        assert _check_resolution_rules(table), f"Resolution rule failed for quick={quick_flag}"

    # Mutation check: corrupting cpw or cells_across_feature must be caught by the validator
    mutated_cpw = [dict(t) for t in tut_module.get_task_table(quick=True)]
    mutated_cpw[1]["cpw"] = 7.5
    assert not _check_resolution_rules(mutated_cpw)

    mutated_feat = [dict(t) for t in tut_module.get_task_table(quick=False)]
    mutated_feat[0]["cells_across_feature"] = 2.0
    assert not _check_resolution_rules(mutated_feat)


def test_orthotope_eval_metric_exact_parity_and_mutation(tut_module):
    """Verify exact target-orthotope eval_metric against stored summary.json and mutation failure."""
    sp_summary = json.loads((DATA_DIR / "round2_splitter" / "summary.json").read_text())
    mc_summary = json.loads((DATA_DIR / "round2_mode_converter" / "summary.json").read_text())

    # 1. Power splitter exact parity across all three stored evaluators
    sp_off, sp_margins = tut_module.compute_splitter_eval_metric(np.asarray(sp_summary["official_S2"]))
    sp_est, _ = tut_module.compute_splitter_eval_metric(np.asarray(sp_summary["wakis_r1_ceviche_estimator_S2"]))
    sp_mod, _ = tut_module.compute_splitter_eval_metric(np.asarray(sp_summary["wakis_r1_modal_S2"]))

    assert abs(sp_off - float(sp_summary["official_eval_metric"])) < 1e-12
    assert abs(sp_est - float(sp_summary["wakis_r1_ceviche_estimator_eval_metric"])) < 1e-12
    assert abs(sp_mod - float(sp_summary["wakis_r1_modal_eval_metric"])) < 1e-12
    assert sp_off > 0.068
    assert sp_margins.shape == (6, 3)

    # 2. Mode converter exact parity across all three stored evaluators
    mc_off, mc_margins = tut_module.compute_modeconv_eval_metric(np.asarray(mc_summary["official_S2"]))
    mc_est, _ = tut_module.compute_modeconv_eval_metric(np.asarray(mc_summary["wakis_r1_ceviche_estimator_S2"]))
    mc_mod, _ = tut_module.compute_modeconv_eval_metric(np.asarray(mc_summary["wakis_r1_modal_S2"]))

    assert abs(mc_off - float(mc_summary["official_eval_metric"])) < 1e-12
    assert abs(mc_est - float(mc_summary["wakis_r1_ceviche_estimator_eval_metric"])) < 1e-12
    assert abs(mc_mod - float(mc_summary["wakis_r1_modal_eval_metric"])) < 1e-12
    assert 0.31 < mc_off < 0.32
    assert mc_margins.shape == (6, 2)

    # 3. Mutation check: degrading reflection on a single wavelength above -20 dB (0.01) or
    # dropping transmission below the target window must make eval_metric strictly negative
    sp_mut_refl = np.asarray(sp_summary["official_S2"], dtype=np.float64).copy()
    sp_mut_refl[2, 0, 0] = 0.015  # violates <= 0.01 (-20 dB) reflection bound
    sp_mut_eval, _ = tut_module.compute_splitter_eval_metric(sp_mut_refl)
    assert sp_mut_eval < 0.0
    assert np.isclose(sp_mut_eval, (0.01 - 0.015) / 0.01, atol=1e-6)

    mc_mut_trans = np.asarray(mc_summary["official_S2"], dtype=np.float64).copy()
    mc_mut_trans[0, 0, 1] = 0.85  # violates >= 0.89125 (-0.5 dB) conversion bound
    mc_mut_eval, _ = tut_module.compute_modeconv_eval_metric(mc_mut_trans)
    assert mc_mut_eval < 0.0


def test_length_scale_measurement_and_mutation(tut_module):
    """Verify morphological length-scale check reproduces 8/8 px and detects injected 2-px defects."""
    sp_design = np.load(DATA_DIR / "round2_splitter" / "design.npy")
    mc_design = np.load(DATA_DIR / "round2_mode_converter" / "design.npy")

    sp_ls = tut_module.measure_length_scales_px(sp_design)
    mc_ls = tut_module.measure_length_scales_px(mc_design)
    assert sp_ls == (8, 8)
    assert mc_ls == (8, 8)

    # Mutation check 1: inject a 3-px wide solid ridge inside a void region of the splitter
    sp_mut_solid = sp_design.copy()
    sp_mut_solid[10:30, 10:30] = 0.0
    sp_mut_solid[16:19, 12:26] = 1.0
    w_mut, _ = tut_module.measure_length_scales_px(sp_mut_solid)
    assert w_mut < 8, f"Expected solid width violation (< 8 px), got {w_mut}"

    # Mutation check 2: inject a 3-px wide void slot inside a solid region of the mode converter
    mc_mut_void = mc_design.copy()
    mc_mut_void[70:90, 40:60] = 1.0
    mc_mut_void[78:81, 44:56] = 0.0
    _, s_mut = tut_module.measure_length_scales_px(mc_mut_void)
    assert s_mut < 8, f"Expected void spacing violation (< 8 px), got {s_mut}"


def test_wakis_resimulation_parity_and_gap_analysis(tut_module):
    """Verify Wakis full-resolution re-simulation parity and mode converter leaderboard gap analysis."""
    if not (DATA_DIR / "wakis_resim_comparison.json").exists():
        tut_module.run_task_0_metrics_and_length_scale()
        tut_module.run_task_1_wakis_resimulation(mode="quick")
        tut_module.run_task_2_gap_and_candidate_analysis()

    resim_cmp = json.loads((DATA_DIR / "wakis_resim_comparison.json").read_text())
    assert resim_cmp["splitter_max_abs_diff_vs_wakis_r1_modal_S2"] < 1e-4
    assert resim_cmp["splitter_max_abs_diff_vs_dense_S2"] < 1e-3
    assert resim_cmp["modeconv_max_abs_diff_vs_wakis_r1_modal_S2"] < 1e-4
    assert resim_cmp["modeconv_max_abs_diff_vs_dense_S2"] < 1e-3

    gap = json.loads((DATA_DIR / "gap_and_candidate_analysis.json").read_text())
    assert gap["worst_wavelength_nm"] == 1265
    assert gap["worst_channel"] == "transmission_TE1"
    assert 0.31 < gap["worst_margin"] < 0.32
    # Stage-1 surrogate peak before morphological repair is much higher than Stage-2 entry
    assert gap["stage1_peak_surrogate_eval"] > gap["stage2_entry_surrogate_eval"] + 0.15

    # Verify across all 6 wavelengths that back-reflection is < 0.2% while radiation/TE0 leakage is > 4%
    for row in gap["mode_converter_wavelength_breakdown"]:
        assert row["reflection_S11_sq"] < 0.002
        assert row["radiation_or_TE0_leakage"] > 0.04
        assert row["limiting_channel"] == "transmission_TE1"


def test_pedagogical_toy_optimization_and_figures(tut_module):
    """Verify pedagogical toy adjoint optimization improves objective and all 8 figures exist."""
    if not (DATA_DIR / "toy_optimization_summary.json").exists():
        tut_module.run_task_3_pedagogical_toy_optimization(mode="quick")
    toy = json.loads((DATA_DIR / "toy_optimization_summary.json").read_text())
    assert toy["label"] == "PEDAGOGICAL_TOY_DEMO_NOT_INVRS_GYM"
    assert toy["objective_ratio_end_over_start"] > 1.05
    assert all(g > 0.0 for g in toy["grad_norm_history"])

    expected_figs = [
        "01_final_designs_and_length_scale.png",
        "02_optimization_traces_and_beta_schedule.png",
        "03_splitter_spectra_comparison.png",
        "04_mode_converter_spectra_and_gap_analysis.png",
        "05_splitter_fields_wavelengths.png",
        "06_mode_converter_fields_wavelengths.png",
        "07_leaderboard_and_round_comparison.png",
        "08_design_evolution.gif",
    ]
    if not all((FIG_DIR / f).exists() for f in expected_figs):
        tut_module.plot_all()
    for fname in expected_figs:
        fpath = FIG_DIR / fname
        assert fpath.exists(), f"Missing figure {fname}"
        size_bytes = fpath.stat().st_size
        assert size_bytes > 1024, f"Figure {fname} is unexpectedly small ({size_bytes} B)"
        if fname.endswith(".gif"):
            assert size_bytes < 5 * 1024 * 1024, f"GIF {fname} exceeds 5 MB limit ({size_bytes} B)"


def test_optional_invrs_gym_rescore():
    """Optional test that runs only if invrs_gym is installed in the environment."""
    invrs_gym = pytest.importorskip("invrs_gym")
    assert invrs_gym is not None
