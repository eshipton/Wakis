# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Tests for the `SubTriArm` benchmark arm and `results.fetch` work reporting."""

import time

import numpy as np

from wakis.benchmarks import get_case, run
from wakis.benchmarks.arms.subtri2d import SubTriArm
from wakis.benchmarks.results import fetch, publish


def test_b3_g5_gap_patch_and_mutation():
    """B3-g5 coarsest rung: patch and RT gap fields agree at centre; sign flip fails."""
    case = get_case("B3-g5")
    case.t_run /= 8.0
    arm = SubTriArm()
    h = arm.resolutions(case)[0]

    res = arm.run(case, h)
    e_patch = res["e_gap"]
    e_rt = res["e_gap_rt"]
    e0 = res["e0_gap"]

    assert e_patch.shape == (len(case.freqs), 9, 2)
    assert e_rt.shape == (len(case.freqs), 9, 2)
    assert e0.shape == (len(case.freqs),)
    assert np.all(np.isfinite(e_patch))
    assert np.all(np.isfinite(e_rt))
    assert res["cost"]["gap_metric"] == "patch"

    # LTS auto on B3-g5 at h = 4 nm picks p > 1 and records unknown-weighted work
    cost = res["cost"]
    assert cost["lts"] is not None and cost["lts"]["p"] > 1
    assert cost["work"] > cost["unknowns"] * cost["steps"]

    # Centre point (index 0) at peak enhancement frequency
    enh_centre = np.sum(np.abs(e_patch[:, 0]) ** 2, axis=1) / np.abs(e0) ** 2
    idx = int(np.argmax(enh_centre))
    rel_diff = float(
        np.linalg.norm(e_patch[idx, 0] - e_rt[idx, 0]) / np.linalg.norm(e_rt[idx, 0])
    )
    assert rel_diff < 0.15

    # Mutation check 1: flipping total patch sign breaks agreement with RT
    rel_diff_total_flip = float(
        np.linalg.norm(-e_patch[idx, 0] - e_rt[idx, 0]) / np.linalg.norm(e_rt[idx, 0])
    )
    assert rel_diff_total_flip > 0.15

    # Mutation check 2: flipping _patch_matrix flux-normal sign (scattered field sign)
    pvec = np.array([-np.sin(case.direction), np.cos(case.direction)])
    e_inc_centre = e0[idx] * pvec
    e_scat_patch = e_patch[idx, 0] - e_inc_centre
    e_patch_mut = -e_scat_patch + e_inc_centre
    rel_diff_scat_flip = float(
        np.linalg.norm(e_patch_mut - e_rt[idx, 0]) / np.linalg.norm(e_rt[idx, 0])
    )
    assert rel_diff_scat_flip > 0.15


def test_b2_au_te_coarsest_and_results_fetch(tmp_path):
    """B2-Au-TE full coarsest rung finishes < 60 s with valid keys and LTS search."""
    case = get_case("B2-Au-TE")
    arm = SubTriArm()
    ladders = arm.resolutions(case)
    h = ladders[0]

    t0 = time.time()
    rec = run(case, arm, h, catch=False)
    elapsed = time.time() - t0
    assert elapsed < 60.0

    for key in ("case", "arm", "resolution", "metrics", "cost", "wall_s"):
        assert key in rec

    cost = rec["cost"]
    expected_cost_keys = (
        "unknowns",
        "nodes",
        "cells",
        "steps",
        "dt",
        "run_s",
        "setup_s",
        "h",
        "h_max",
        "levels",
        "n_squares_per_level",
        "n_triangles",
        "n_hanging",
        "cfl",
        "lts",
        "lts_search",
        "work",
        "gap_metric",
    )
    for key in expected_cost_keys:
        assert key in cost, f"missing cost key {key}"

    metrics = rec["metrics"]
    assert np.isfinite(metrics["q_ext_nrms"]) and metrics["q_ext_nrms"] < 0.05
    assert np.isfinite(metrics["q_sca_nrms"])
    assert np.isfinite(metrics["q_abs_nrms"])

    # Either LTS chose p > 1 or every candidate p > 1 has a documented skip/rate entry
    if cost["lts"] is not None:
        assert cost["lts"]["p"] > 1
    else:
        for p in arm.lts_candidates:
            entry = cost["lts_search"][str(p)]
            assert (
                "skipped" in entry
                or entry["work_rate"] >= cost["lts_search"]["1"]["work_rate"]
            )

    # At the second rung (h = 20/7 nm), dt_base is small enough that LTS chooses p > 1
    _, solver_r2, _ = arm.build(case, ladders[1])
    assert solver_r2.lts is not None and solver_r2.lts.p > 1

    # Verify results.fetch uses cost['work'] when present, and falls back otherwise
    rec_lts = dict(rec)
    rec_lts["cost"] = dict(cost, work=cost["unknowns"] * cost["steps"] + 12345)
    publish(rec_lts, root=str(tmp_path), tag="lts_test")
    df = fetch("B2-Au-TE", root=str(tmp_path))
    assert len(df) == 1
    assert int(df.loc[0, "work"]) == cost["unknowns"] * cost["steps"] + 12345

    # Mutation check on fetch: without cost['work'], work falls back to unknowns * steps
    rec_no_work = dict(rec)
    cost_no_work = dict(cost)
    cost_no_work.pop("work")
    rec_no_work["cost"] = cost_no_work
    publish(rec_no_work, root=str(tmp_path / "fallback"), tag="fallback_test")
    df_fb = fetch("B2-Au-TE", root=str(tmp_path / "fallback"))
    assert int(df_fb.loc[0, "work"]) == cost["unknowns"] * cost["steps"]
    assert int(df_fb.loc[0, "work"]) != int(df.loc[0, "work"])


def test_subtri_compare_and_plots(tmp_path):
    """
    Verify `studies/subtri-hybrid/compare.py` and `plot_subtri.py` offline
    pipeline, including deduplication by latest `published` (with mutation
    check), `work` fallback, target interpolation, and figure generation.
    """
    import importlib.util
    import json
    from pathlib import Path

    import pandas as pd
    import pytest

    study_dir = Path(__file__).resolve().parents[1] / "studies" / "subtri-hybrid"
    spec_c = importlib.util.spec_from_file_location("compare", study_dir / "compare.py")
    compare = importlib.util.module_from_spec(spec_c)
    spec_c.loader.exec_module(compare)

    spec_p = importlib.util.spec_from_file_location(
        "plot_subtri", study_dir / "plot_subtri.py"
    )
    plot_subtri = importlib.util.module_from_spec(spec_p)
    spec_p.loader.exec_module(plot_subtri)

    data_dir = tmp_path / "data"
    (data_dir / "b2").mkdir(parents=True)

    # Older record at h = 4 nm has worse work (9e7) and error (0.015);
    # newer record at h = 4 nm has better work (2e7) and error (0.004 < 1% target).
    rec_old = {
        "case": "B2-Au-TE",
        "arm": "subtri-hybrid",
        "resolution": 4e-9,
        "published": "2026-10-18T10:00:00Z",
        "metrics": {"q_ext_nrms": 0.015, "q_ext_lam_peak": 0.004},
        "cost": {
            "unknowns": 5000,
            "steps": 18000,
            "work": 90_000_000,
            "run_s": 4.0,
            "lts": {"p": 2, "fine_fraction": 0.25},
        },
    }
    rec_new = {
        "case": "B2-Au-TE",
        "arm": "subtri-hybrid",
        "resolution": 4e-9,
        "published": "2026-10-19T10:00:00Z",
        "metrics": {"q_ext_nrms": 0.004, "q_ext_lam_peak": 0.001},
        "cost": {
            "unknowns": 5000,
            "steps": 4500,
            "work": 20_000_000,
            "run_s": 1.2,
            "lts": {"p": 4, "fine_fraction": 0.15},
        },
    }
    # Record without cost['work'] to verify fallback to unknowns * steps
    rec_nolts = {
        "case": "B2-Au-TE",
        "arm": "subtri-hybrid-nolts",
        "resolution": 4e-9,
        "published": "2026-10-19T10:00:00Z",
        "metrics": {"q_ext_nrms": 0.004, "q_ext_lam_peak": 0.001},
        "cost": {"unknowns": 5000, "steps": 10000, "run_s": 2.5},
    }

    (data_dir / "b2" / "old.json").write_text(json.dumps(rec_old))
    (data_dir / "b2" / "new.json").write_text(json.dumps(rec_new))
    (data_dir / "b2" / "nolts.json").write_text(json.dumps(rec_nolts))

    # Cached baseline for B2-Au-TE
    base_df = pd.DataFrame(
        [
            {
                "case": "B2-Au-TE",
                "arm": "hybrid-quadtri",
                "resolution": 4e-9,
                "published": "2026-10-15T00:00:00Z",
                "c.unknowns": 6000,
                "c.steps": 8000,
                "c.work": 48_000_000,
                "c.run_s": 3.0,
                "work": 48_000_000,
                "m.q_ext_nrms": 0.005,
                "m.q_ext_lam_peak": 0.0015,
            }
        ]
    )
    base_df.to_csv(data_dir / "baselines_B2-Au-TE.csv", index=False)

    out = compare.compare_all(data_dir=data_dir, offline=True)
    b2_targets = {
        (t["metric"], t["target"]): t for t in out["cases"]["B2-Au-TE"]["targets"]
    }
    t_1pct = b2_targets[("q_ext_nrms", 0.01)]
    # Latest published record is already <= 1% at coarsest rung -> negative upper bound -2e7
    assert t_1pct["arms"]["subtri-hybrid"]["work"] == pytest.approx(-20_000_000.0)
    # Fallback work for nolts = 5000 * 10000 = 5e7 -> upper bound -5e7
    assert t_1pct["arms"]["subtri-hybrid-nolts"]["work"] == pytest.approx(-50_000_000.0)
    assert t_1pct["best_baseline"]["arm"] == "hybrid-quadtri"
    assert t_1pct["ours_over_best_work"] == pytest.approx(20_000_000.0 / 48_000_000.0)

    md_text = (data_dir / "compare.md").read_text()
    assert "≤5.0k / ≤2.0e7 / ≤1.2 s" in md_text

    # Mutation check: if rec_old has a newer `published` timestamp than rec_new,
    # deduplication selects rec_old (err=0.015 > 0.01), so 1% target is NOT reached (None).
    rec_old_mut = dict(rec_old, published="2026-10-20T10:00:00Z")
    (data_dir / "b2" / "old.json").write_text(json.dumps(rec_old_mut))
    out_mut = compare.compare_all(data_dir=data_dir, offline=True)
    t_1pct_mut = {
        (t["metric"], t["target"]): t for t in out_mut["cases"]["B2-Au-TE"]["targets"]
    }[("q_ext_nrms", 0.01)]
    assert t_1pct_mut["arms"]["subtri-hybrid"]["work"] is None
    assert t_1pct_mut["ours_over_best_work"] != pytest.approx(
        t_1pct["ours_over_best_work"]
    )

    # Restore rec_old and verify plotting functions run headlessly and write PNGs
    (data_dir / "b2" / "old.json").write_text(json.dumps(rec_old))
    fig_dir = tmp_path / "figures"
    p_err = plot_subtri.plot_error_vs_work(
        data_dir=data_dir, fig_dir=fig_dir, offline=True
    )
    p_lts = plot_subtri.plot_lts_gain(data_dir=data_dir, fig_dir=fig_dir)
    assert p_err.exists() and p_err.stat().st_size > 1000
    assert p_lts.exists() and p_lts.stat().st_size > 1000
