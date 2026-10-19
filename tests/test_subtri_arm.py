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
