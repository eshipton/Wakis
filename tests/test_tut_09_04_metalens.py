"""Tests for Tutorial 9.4: Visible-band RGB achromatic metalens."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = REPO_ROOT / "tutorials" / "09_inverse_design" / "09-04_metalens"
SCRIPT_PATH = TUT_DIR / "metalens.py"
DATA_DIR = TUT_DIR / "data"
FIG_DIR = TUT_DIR / "figures"


@pytest.fixture(scope="module")
def tut_module():
    """Load Tutorial 9.4 module dynamically."""
    spec = importlib.util.spec_from_file_location("metalens", SCRIPT_PATH)
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

    # Mutation check: under-resolved cpw or feature cells must fail validation
    mut_cpw = [dict(t) for t in tut_module.get_task_table(quick=True)]
    mut_cpw[0]["cpw"] = 6.0
    assert not _check_resolution_rules(mut_cpw)

    mut_feat = [dict(t) for t in tut_module.get_task_table(quick=False)]
    mut_feat[1]["cells_across_feature"] = 2.0
    assert not _check_resolution_rules(mut_feat)


def test_metalens_eval_metric_and_mutation(tut_module):
    """Verify focal intensity enhancement metric against stored summary.json and mutation checks."""
    r2_summary = json.loads((DATA_DIR / "round2_metalens" / "summary.json").read_text())
    r1_s0 = json.loads(
        (DATA_DIR / "round1" / "20261018-0430-invrs-metalens__shard_0__metalens_seed0__summary.json").read_text()
    )

    off_eval, off_eta, off_softmin = tut_module.compute_metalens_eval_metric(
        np.asarray(r2_summary["official_enhancement_ex"])
    )
    wak_eval, wak_eta, _ = tut_module.compute_metalens_eval_metric(
        np.asarray(r2_summary["wakis_enhancement_ex"])
    )

    assert abs(off_eval - float(r2_summary["official_eval_metric"])) < 1e-12
    assert abs(wak_eval - float(r2_summary["wakis_eval_metric"])) < 1e-12
    assert np.isclose(off_eval, 8.865178, atol=1e-4)
    assert np.isclose(wak_eval, 8.432886, atol=1e-4)
    assert off_eta.shape == (3,)
    assert wak_eta.shape == (3,)
    # Softmin loss lies between -mean(eta) and -min(eta)
    assert -float(np.mean(off_eta)) <= off_softmin <= -off_eval

    # Round-1 unconstrained best is ~14.07 at 2/1 px
    r1_eval, _, _ = tut_module.compute_metalens_eval_metric(np.asarray(r1_s0["official_enhancement_ex"]))
    assert np.isclose(r1_eval, 14.067628, atol=1e-4)
    assert r1_s0["length_scale_solid_void_px"] == [2, 1]

    # Mutation check 1: spoiling a single RGB channel (e.g., green 550 nm dropping to 1.2x) must pull eval_metric down to 1.2
    eta_mut = off_eta.copy()
    eta_mut[1] = 1.2
    eval_mut, _, _ = tut_module.compute_metalens_eval_metric(eta_mut)
    assert np.isclose(eval_mut, 1.2, atol=1e-12)

    # Mutation check 2: non-positive reference intensity must raise ValueError
    with pytest.raises(ValueError, match="positive"):
        tut_module.compute_metalens_eval_metric(np.array([10.0, 9.0, 9.5]), np.array([1.0, 0.0, 1.0]))


def test_length_scale_and_diffraction_limit_with_mutation(tut_module):
    """Verify 5/5 px (100 nm) length-scale check and diffraction-limited sinc^2 profile + mutations."""
    design = np.load(DATA_DIR / "round2_metalens" / "design.npy")
    ls = tut_module.measure_length_scales_px(design, max_px=10, periodic=(True, False))
    assert ls == (5, 5)

    # Mutation check on length scale: injecting a 3-px pillar or 3-px slot inside the lens must drop LS < 5
    des_mut_solid = design.copy()
    des_mut_solid[200:220, 10:30] = 0.0
    des_mut_solid[208:211, 14:26] = 1.0
    w_mut, _ = tut_module.measure_length_scales_px(des_mut_solid, max_px=10, periodic=(True, False))
    assert w_mut < 5

    des_mut_void = design.copy()
    des_mut_void[380:400, 10:30] = 1.0
    des_mut_void[388:391, 14:26] = 0.0
    _, s_mut = tut_module.measure_length_scales_px(des_mut_void, max_px=10, periodic=(True, False))
    assert s_mut < 5

    # Verify diffraction-limited sinc^2 profile and monotonic FWHM scaling with wavelength
    x_um = np.linspace(-2.0, 2.0, 801)
    _, fwhm_450 = tut_module.diffraction_limited_sinc2_profile(x_um, 0.45, na=tut_module.LENS_NA)
    prof_550, fwhm_550 = tut_module.diffraction_limited_sinc2_profile(x_um, 0.55, na=tut_module.LENS_NA)
    _, fwhm_650 = tut_module.diffraction_limited_sinc2_profile(x_um, 0.65, na=tut_module.LENS_NA)

    assert np.isclose(tut_module.LENS_NA, 0.901523, atol=1e-4)
    assert np.isclose(prof_550[400], 1.0, atol=1e-12)
    assert 0.20 < fwhm_450 < fwhm_550 < fwhm_650 < 0.35

    # Mutation check on diffraction limit: invalid NA or wavelength must raise ValueError
    with pytest.raises(ValueError, match="NA"):
        tut_module.diffraction_limited_sinc2_profile(x_um, 0.55, na=1.2)
    with pytest.raises(ValueError, match="Wavelength"):
        tut_module.diffraction_limited_sinc2_profile(x_um, -0.45, na=0.9)


def test_wakis_resimulation_parity_and_validation_summary(tut_module):
    """Verify Wakis full-resolution re-simulation parity (1.71e-7) and Rasmus123nm validation."""
    if not (DATA_DIR / "wakis_resim_comparison.json").exists():
        tut_module.run_task_0_fom_validation_and_length_scale()
        tut_module.run_task_1_wakis_resimulation(mode="quick")
        tut_module.run_task_2_focal_spot_and_diffraction_limit()

    resim_cmp = json.loads((DATA_DIR / "wakis_resim_comparison.json").read_text())
    assert resim_cmp["max_abs_diff_vs_stored_wakis"] < 1e-5
    assert resim_cmp["max_rel_diff_vs_official_fmmax"] < 0.06
    assert np.isclose(resim_cmp["wakis_eval_metric_resim"], resim_cmp["wakis_eval_metric_stored"], atol=1e-5)

    val_summary = json.loads((DATA_DIR / "fom_validation_and_length_scale_summary.json").read_text())
    assert val_summary["leaderboard_status"] == "not_included_in_bundle"
    assert val_summary["rasmus123nm_validation"]["min_eval_relative_diff"] < 0.005


def test_pedagogical_toy_optimization_and_figures(tut_module):
    """Verify pedagogical toy metalens optimization improves focal intensity and all 8 figures exist."""
    if not (DATA_DIR / "toy_optimization_summary.json").exists():
        tut_module.run_task_3_pedagogical_toy_optimization(mode="quick")
    toy = json.loads((DATA_DIR / "toy_optimization_summary.json").read_text())
    assert toy["label"] == "PEDAGOGICAL_TOY_DEMO_NOT_INVRS_GYM"
    assert toy["objective_ratio_end_over_start"] > 1.05
    assert all(g > 0.0 for g in toy["grad_norm_history"])

    expected_figs = [
        "01_metalens_profile_and_length_scale.png",
        "02_optimization_trace_and_beta_schedule.png",
        "03_intensity_enhancement_maps_rgb.png",
        "04_wavefield_real_part_rgb.png",
        "05_focal_line_cuts_vs_diffraction_limit.png",
        "06_wakis_vs_fmmax_validation_and_resim.png",
        "07_round1_vs_round2_length_scale_tradeoff.png",
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
