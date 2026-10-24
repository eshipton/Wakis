"""Tests for Tutorial 9.1: Adjoint gradients for time-domain Maxwell solvers."""

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
    / "09-01_adjoint_gradients"
)
SCRIPT_PATH = TUT_DIR / "adjoint_gradients.py"


def _load_tutorial_module():
    spec = importlib.util.spec_from_file_location("tut_09_01_adjoint_gradients", SCRIPT_PATH)
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

    # Mutation check: under-resolved task must be rejected
    mutated_cpw = mod.get_task_table(quick=True)
    mutated_cpw[1] = dict(mutated_cpw[1], cpw=8.5)
    with pytest.raises(ValueError, match="cpw >= 10"):
        _validate_task_resolution(mutated_cpw)

    mutated_feat = mod.get_task_table(quick=True)
    mutated_feat[2] = dict(mutated_feat[2], cells_across_feature=2.5)
    with pytest.raises(ValueError, match="cells_across_feature >= 4"):
        _validate_task_resolution(mutated_feat)


def test_filter_and_projection_chain_rule_and_mutation():
    """Verify conic filter VJP dot-product identity, chain-rule FD match, and mutation failure."""
    mod = _load_tutorial_module()
    rng = np.random.default_rng(123)
    u = rng.uniform(0.1, 0.9, (12, 10, 1))
    v = rng.standard_normal((12, 10, 1))

    u_tilde, k, den = mod.conic_filter_np(u, radius_cells=2.3)
    v_bar = mod.conic_filter_vjp_np(v, k, den)
    lhs = float(np.sum(u_tilde * v))
    rhs = float(np.sum(u * v_bar))
    assert abs(lhs - rhs) / abs(lhs) < 1e-14

    # Chain rule of filter + tanh_projection against central finite differences
    weights = rng.standard_normal(u.shape)

    def scalar_loss(rho_in: np.ndarray) -> float:
        rt, _, _ = mod.conic_filter_np(rho_in, radius_cells=2.3)
        rb, _ = mod.tanh_projection_np(rt, beta=6.0, eta=0.5)
        return float(np.sum(weights * (rb**2)))

    rt0, k0, den0 = mod.conic_filter_np(u, radius_cells=2.3)
    rb0, dbar0 = mod.tanh_projection_np(rt0, beta=6.0, eta=0.5)
    g_bar = 2.0 * weights * rb0
    g_raw = mod.conic_filter_vjp_np(g_bar * dbar0, k0, den0)

    d = rng.standard_normal(u.shape)
    d /= float(np.linalg.norm(d))
    h = 1e-6
    fd_dir = (scalar_loss(u + h * d) - scalar_loss(u - h * d)) / (2.0 * h)
    adj_dir = float(np.sum(g_raw * d))
    assert abs(fd_dir - adj_dir) / abs(fd_dir) < 1e-8

    # Mutation check: omitting the projection derivative dbar0 breaks gradient agreement
    g_raw_mutated = mod.conic_filter_vjp_np(g_bar, k0, den0)
    adj_dir_mut = float(np.sum(g_raw_mutated * d))
    assert abs(fd_dir - adj_dir_mut) / abs(fd_dir) > 0.1


def test_saved_data_and_figures():
    """Verify saved outputs from all 5 tasks and all 10 generated figures."""
    data_dir = TUT_DIR / "data"
    fig_dir = TUT_DIR / "figures"

    # Task 0: dot-product test & mutations
    with open(data_dir / "task_0_dot_product.json") as f:
        d0 = json.load(f)
    assert len(d0["cases"]) == 6
    for c in d0["cases"]:
        assert c["rel_err"] < 1e-13, f"Dot-product failed for {c['case']}: {c['rel_err']}"
    assert d0["mutation_transpose_errs"][0] < 1e-13
    assert d0["mutation_transpose_errs"][2] > 1e-9  # delta = 1e-4 mutation fails (> 1e-12 threshold)
    assert d0["mutation_transpose_errs"][3] > 1e-7  # delta = 1e-2 mutation fails
    assert d0["reverse_clean_rel_err"] < 1e-12
    assert d0["reverse_corrupt_rel_err"] > 1e-5  # corrupted CPML H_layer fails

    # Task 1: finite-difference V-curve and node-by-node parity
    d1 = np.load(data_dir / "task_1_fd_vcurve.npz")
    assert float(np.min(d1["rel_err_f64"])) < 1e-8
    # Truncation branch (largest h) and round-off branch (smallest h) both larger than minimum
    assert float(d1["rel_err_f64"][0]) > 100.0 * float(np.min(d1["rel_err_f64"]))
    assert float(d1["rel_err_f64"][-1]) > 1000.0 * float(np.min(d1["rel_err_f64"]))
    assert float(d1["iso_node_rel_err"]) < 1e-5
    assert float(d1["tensor_node_rel_err"]) < 1e-5

    # Task 2: sensitivity map & projection family
    d2 = np.load(data_dir / "task_2_sensitivity.npz")
    assert d2["g_bar"].shape == d2["g_raw"].shape
    assert np.all(np.diff(d2["binarization_vs_beta"]) >= -1e-12)
    assert float(d2["binarization_vs_beta"][-1]) > float(d2["binarization_vs_beta"][0]) + 0.3

    # Task 3: memory ladder benchmark
    with open(data_dir / "task_3_memory_ladder.json") as f:
        d3 = json.load(f)
    by_label = {r["label"]: r for r in d3["live_results"]}
    assert by_label["store_f64"]["bitwise"] is True
    assert by_label["store_f32"]["peak_bytes"] == by_label["store_f64"]["peak_bytes"] // 2
    assert 0.0 < by_label["store_f32"]["grad_rel_err"] < 1e-5
    for ck in ("ckpt_s2", "ckpt_s4", "ckpt_s8", "ckpt_s6_b1"):
        assert by_label[ck]["bitwise"] is True
        assert by_label[ck]["grad_rel_err"] == 0.0
    assert by_label["reverse"]["recompute_steps"] == 0
    assert by_label["reverse"]["grad_rel_err"] < 1e-12
    assert by_label["nyquist_d16"]["grad_rel_err"] > by_label["nyquist_d4"]["grad_rel_err"] > 0.0

    for s_key, dp_entry in d3["revolve_dp_table"].items():
        for rev_c, hal_c in zip(dp_entry["revolve_extra"], dp_entry["halving_extra"]):
            assert rev_c <= hal_c, f"revolve_cost exceeded halving for s={s_key}"

    # Task 4: 2D waveguide bend topology optimization
    d4 = np.load(data_dir / "task_4_optimization.npz")
    objs = d4["objectives"]
    assert float(objs[-1]) > 3.0 * float(objs[0])
    assert float(d4["binarization"][-1]) > float(d4["binarization"][0])

    # Figures existence and size budget
    expected_figs = [
        "01_adjoint_pipeline_schematic.png",
        "02_dot_product_and_mutations.png",
        "03_fd_step_sweep_vcurve.png",
        "04_density_filter_projection.png",
        "05_sensitivity_gradient_map.png",
        "06_memory_ladder_measured.png",
        "07_memory_scaling_and_roundoff.png",
        "08_toy_optimization_convergence.png",
        "09_initial_vs_optimized_fields.png",
        "10_design_evolution.gif",
    ]
    for fname in expected_figs:
        fpath = fig_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 1024, f"Missing or empty figure {fname}"

    total_data_bytes = sum(p.stat().st_size for p in data_dir.glob("*") if p.is_file())
    assert total_data_bytes < 5 * 1024 * 1024, f"data/ exceeds 5 MB: {total_data_bytes}"
    gif_bytes = (fig_dir / "10_design_evolution.gif").stat().st_size
    assert gif_bytes < 5 * 1024 * 1024, f"GIF exceeds 5 MB: {gif_bytes}"


def test_live_gradient_ascent_vs_wrong_sign_mutation():
    """Live physics & mutation check: +grad increases objective, -grad decreases it."""
    mod = _load_tutorial_module()
    sim, _ = mod.make_bend_problem(n_grid=26, n_design=6, Nt=75, n_pml=6, cpl=28)
    rho0 = np.full(sim.design.shape, 0.5, dtype=np.float64)

    out0 = sim.forward(rho0)
    ez0 = out0["probe"]["Ez"]
    f0 = float(np.sum(np.abs(ez0) ** 2))
    g = sim.vjp({"probe": {"Ez": 2.0 * ez0}})
    g_unit = g / (float(np.max(np.abs(g))) + 1e-30)

    step = 0.08
    out_plus = sim.forward(np.clip(rho0 + step * g_unit, 0.0, 1.0))
    f_plus = float(np.sum(np.abs(out_plus["probe"]["Ez"]) ** 2))

    # Mutated sign (-grad) must decrease the objective
    out_minus = sim.forward(np.clip(rho0 - step * g_unit, 0.0, 1.0))
    f_minus = float(np.sum(np.abs(out_minus["probe"]["Ez"]) ** 2))

    assert f_plus > f0 > f_minus
