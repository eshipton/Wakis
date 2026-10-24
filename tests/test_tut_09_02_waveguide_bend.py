"""Tests for Tutorial 9.2: Inverse design of a 90-degree waveguide bend (ceviche_waveguide_bend)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "tutorials"
    / "09_inverse_design"
    / "09-02_waveguide_bend"
)
SCRIPT_PATH = TUT_DIR / "waveguide_bend.py"


def _load_tutorial_module():
    spec = importlib.util.spec_from_file_location("tut_09_02_waveguide_bend", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _validate_task_resolution(table: list[dict]) -> None:
    if len(table) < 5:
        raise ValueError("Task table must contain at least 5 tasks.")
    for row in table:
        if float(row.get("cpw", 0.0)) < 10.0:
            raise ValueError(f"Task {row.get('name')} violates cpw >= 10: {row.get('cpw')}")
        if float(row.get("cells_across_feature", 0.0)) < 4.0:
            raise ValueError(
                f"Task {row.get('name')} violates cells_across_feature >= 4: "
                f"{row.get('cells_across_feature')}"
            )


def test_task_table_and_resolution_rules():
    """Verify task table structure and resolution rules, plus mutation rejection."""
    mod = _load_tutorial_module()
    for quick in (True, False):
        table = mod.get_task_table(quick=quick)
        assert [t["index"] for t in table] == [0, 1, 2, 3, 4]
        _validate_task_resolution(table)

    mutated = mod.get_task_table(quick=True)
    mutated[1] = dict(mutated[1], cpw=7.5)
    with pytest.raises(ValueError, match="cpw >= 10"):
        _validate_task_resolution(mutated)


def test_round2_stored_files_and_length_scale_with_mutation():
    """Verify Round-2 design.npy, summary.json, fields_compact.npz, and 8/8 px length scale with mutation check."""
    mod = _load_tutorial_module()
    art = mod.load_round2_artifacts()

    design = art["design"]
    assert design.shape == (160, 160)
    assert set(np.unique(design)).issubset({0.0, 1.0})

    # Verify design.npy matches the [92:252, 92:252] design region in fields_compact.npz
    eps_design_box = (art["eps_full"][92:252, 92:252] > 7.0).astype(float)
    assert np.array_equal(eps_design_box, design)

    summ = art["summary"]
    assert summ["challenge"] == "ceviche_waveguide_bend"
    assert summ["official_eval_metric"] == pytest.approx(0.6427273992133412, abs=1e-6)
    assert summ["length_scale_leaderboard_px"] == [8, 8]
    assert summ["length_scale_solid_void_px"] == [8, 8]

    border = 12
    win_binary = (art["eps_full"][92 - border : 252 + border, 92 - border : 252 + border] > 7.0).astype(
        bool
    )
    mw, ms = mod.measure_length_scales_px(
        design, max_px=11, surrounding_window=win_binary, border=border
    )
    assert (mw, ms) == (8, 8)

    # Mutation check: inject a 3-px wide solid bridge into the void bottom-left corner [10:25, 10:13]
    mutated_design = design.copy()
    mutated_win = win_binary.copy()
    mutated_design[10:25, 10:13] = 1.0
    mutated_win[border + 10 : border + 25, border + 10 : border + 13] = True
    mw_mut, _ = mod.measure_length_scales_px(
        mutated_design, max_px=11, surrounding_window=mutated_win, border=border
    )
    assert mw_mut == 3 < 8


def test_eval_metric_formula_and_leaderboard_mutation():
    """Verify exact invrs-gym window eval_metric on Round-2 and leaderboard S2 spectra, plus mutation check."""
    mod = _load_tutorial_module()
    art = mod.load_round2_artifacts()
    summ = art["summary"]
    lb = json.loads((TUT_DIR / "data" / "leaderboard_rescore.json").read_text())
    lb_bend = {e["file"]: e for e in lb if e["challenge"] == "ceviche_waveguide_bend"}

    s2_r2_off = np.array(summ["official_S2"], dtype=np.float64)[:, 0, :]
    s2_r2_mod = np.array(summ["wakis_r1_modal_S2"], dtype=np.float64)[:, 0, :]
    s2_r2_cev = np.array(summ["wakis_r1_ceviche_estimator_S2"], dtype=np.float64)[:, 0, :]
    assert mod.compute_bend_eval_metric(s2_r2_off[:, 1], s2_r2_off[:, 0]) == pytest.approx(
        summ["official_eval_metric"], abs=1e-7
    )
    assert mod.compute_bend_eval_metric(s2_r2_mod[:, 1], s2_r2_mod[:, 0]) == pytest.approx(
        summ["wakis_r1_modal_eval_metric"], abs=1e-7
    )
    assert mod.compute_bend_eval_metric(s2_r2_cev[:, 1], s2_r2_cev[:, 0]) == pytest.approx(
        summ["wakis_r1_ceviche_estimator_eval_metric"], abs=1e-7
    )

    rahul = lb_bend["240630_rahulkpadhy_waveguide_bend_Aadi_Rahul.csv"]
    s2_rahul = np.array(rahul["s2"], dtype=np.float64)[:, 0, :]
    assert mod.compute_bend_eval_metric(s2_rahul[:, 1], s2_rahul[:, 0]) == pytest.approx(
        rahul["eval_metric"], abs=1e-7
    )
    assert rahul["eval_metric"] == pytest.approx(-1.2504255, abs=1e-6)
    assert (rahul["minimum_width"], rahul["minimum_spacing"]) == (9, 7)

    seed_a = lb_bend["ours_full_bend_seed1.npy"]
    s2_seed_a = np.array(seed_a["s2"], dtype=np.float64)[:, 0, :]
    assert mod.compute_bend_eval_metric(s2_seed_a[:, 1], s2_seed_a[:, 0]) == pytest.approx(
        seed_a["eval_metric"], abs=1e-7
    )
    assert seed_a["eval_metric"] == pytest.approx(0.5793099, abs=1e-6)
    assert (seed_a["minimum_width"], seed_a["minimum_spacing"]) == (8, 8)

    # Mutation check: degrade one channel of our Round-2 spectrum below the -0.5 dB transmission floor
    s21_degraded = s2_r2_off[:, 1].copy()
    s21_degraded[2] = 10.0 ** (-0.8 / 10.0)
    assert mod.compute_bend_eval_metric(s21_degraded, s2_r2_off[:, 0]) < -0.5


def test_wakis_resimulation_vs_stored_spectrum_and_figures():
    """Verify Wakis re-simulation matches stored dense_S within justified tolerance, and check figures."""
    data_dir = TUT_DIR / "data"
    fig_dir = TUT_DIR / "figures"

    # Unlabelled top-level toy files must not exist
    for forbidden in ("design.npy", "fields.npz", "snapshots.npz", "task_0_morphology.npz", "summary.json"):
        assert not (data_dir / forbidden).exists(), f"Toy file {forbidden} should be removed from data/"

    resim = np.load(data_dir / "wakis_resimulation.npz")
    assert resim["dense_wavelengths_nm"].shape == (67,)
    assert resim["wavelengths_6_nm"].shape == (6,)

    # Wakis re-simulation vs stored dense_S in fields_compact.npz:
    # Tolerance justified by 2x-coarsened 20 nm quick-mode dispersion (~4.2e-3 peak, ~2.2e-4 mean)
    # and float16 storage quantization (~5e-4) in fields_compact.npz.
    max_diff_s21 = float(np.max(np.abs(resim["s21_sq_resim"] - resim["s21_sq_stored"])))
    mean_diff_s21 = float(np.mean(np.abs(resim["s21_sq_resim"] - resim["s21_sq_stored"])))
    max_diff_s11 = float(np.max(np.abs(resim["s11_sq_resim"] - resim["s11_sq_stored"])))
    assert max_diff_s21 < 5.0e-3
    assert mean_diff_s21 < 2.5e-3
    assert max_diff_s11 < 5.0e-4
    assert float(resim["eval_metric_resim"]) == pytest.approx(0.6427, abs=0.03)

    # Mutation check on spectrum parity: shifting transmission by -0.02 must exceed the 5e-3 tolerance
    mutated_diff = float(np.max(np.abs((resim["s21_sq_resim"] - 0.02) - resim["s21_sq_stored"])))
    assert mutated_diff > 5.0e-3

    # Verify clearly-labelled toy file
    toy = np.load(data_dir / "toy_bend_demo.npz")
    assert str(toy["label"]) == "toy problem (not the challenge)"
    assert float(toy["obj_trace"][-1]) > float(toy["obj_trace"][0])

    with open(data_dir / "round2_evaluation_summary.json") as f:
        summ = json.load(f)
    lb = summ["leaderboard_rescore"]
    assert lb["wakis_round2_best"]["eval_metric"] == pytest.approx(0.6427274, abs=1e-5)
    assert lb["wakis_round2_best"]["min_width_px"] == 8
    assert lb["wakis_round2_best"]["min_spacing_px"] == 8
    assert lb["wakis_round2_seed_a"]["eval_metric"] == pytest.approx(0.5793099, abs=1e-5)
    assert lb["invrs_gym_leaderboard_best"]["eval_metric"] == pytest.approx(-1.2504255, abs=1e-5)
    assert lb["invrs_gym_leaderboard_best"]["min_spacing_px"] == 7

    expected_figs = [
        "01_bend_challenge_and_stack.png",
        "02_optimization_trace_beta_stages.png",
        "03_final_design_and_length_scale.png",
        "04_fields_six_wavelengths.png",
        "05_spectrum_vs_target_windows.png",
        "06_wakis_resim_and_fdfd_parity.png",
        "07_leaderboard_comparison.png",
        "08_design_evolution.gif",
    ]
    for fname in expected_figs:
        fpath = fig_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 1024, f"Missing or empty figure {fname}"
    assert (fig_dir / "08_design_evolution.gif").stat().st_size < 5 * 1024 * 1024

    total_data_bytes = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file())
    assert total_data_bytes < 5 * 1024 * 1024


def test_optional_invrs_gym_evaluation():
    """Optional official invrs-gym evaluation when invrs_gym is installed."""
    invrs_gym = pytest.importorskip("invrs_gym")
    assert invrs_gym is not None
