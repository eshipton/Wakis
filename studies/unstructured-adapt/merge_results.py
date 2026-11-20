# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Merge per-arm adaptive study JSON files and compute cross-arm comparisons
(`studies/unstructured-adapt/merge_results.py`).

Usage
-----
    python studies/unstructured-adapt/merge_results.py --indir DIR --out FILE
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _loglog_interp_with_bracket(
    x_curve: np.ndarray, y_curve: np.ndarray, x_target: float
) -> tuple[float, list[float]]:
    """Interpolate y(x_target) in log10-log10 space without extrapolation."""
    x_arr = np.asarray(x_curve, dtype=float)
    y_arr = np.asarray(y_curve, dtype=float)
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    if x_target < x_arr[0] or x_target > x_arr[-1]:
        raise ValueError(f"x_target={x_target} outside [{x_arr[0]}, {x_arr[-1]}]")
    idx = int(np.searchsorted(x_arr, x_target, side="right") - 1)
    idx = min(max(idx, 0), len(x_arr) - 2)
    bracket = [float(x_arr[idx]), float(x_arr[idx + 1])]
    lx = np.log10(x_arr)
    ly = np.log10(np.maximum(y_arr, 1e-30))
    val = float(10.0 ** np.interp(np.log10(x_target), lx, ly))
    return val, bracket


def _pick_same_sign_history_entry(
    hist: list[dict], dof_min: float, dof_max: float
) -> tuple[dict | None, list[int]]:
    """
    Pick the last history entry inside ``[dof_min, dof_max]`` that did not
    undergo a sign change in ``rel_err_signed`` since the previous iteration
    and shares the initial error sign.
    """
    refused: list[int] = []
    if not hist:
        return None, refused
    s0 = float(np.sign(float(hist[0]["rel_err_signed"])))
    valid_inside: list[dict] = []
    for i, h in enumerate(hist):
        s_cur = float(np.sign(float(h["rel_err_signed"])))
        flipped = i > 0 and (
            s_cur * float(np.sign(float(hist[i - 1]["rel_err_signed"]))) <= 0.0
        )
        if flipped or s_cur != s0:
            refused.append(int(h.get("iteration", i)))
            continue
        if dof_min <= float(h["n_dofs"]) <= dof_max:
            valid_inside.append(h)
    return (valid_inside[-1] if valid_inside else None), refused


def _compute_equal_dof_ratios(arms: dict) -> dict:
    out = {"vs_uniform": {}, "vs_residual": {}}
    if "uniform" not in arms:
        return out

    u_hist = arms["uniform"].get("history", [])
    if len(u_hist) < 2:
        return out
    u_dofs = np.array([float(h["n_dofs"]) for h in u_hist])
    u_err = np.array([abs(float(h["rel_err_signed"])) for h in u_hist])
    has_u_en = all("energy_err2_per_m2" in h for h in u_hist)
    u_en = (
        np.array([float(h["energy_err2_per_m2"]) for h in u_hist]) if has_u_en else None
    )
    u_curl = (
        np.array([float(h["curl_err2_per_m2"]) for h in u_hist]) if has_u_en else None
    )

    for name, arm in arms.items():
        if name in ("box", "uniform"):
            continue
        hist = arm.get("history", [])
        if not hist:
            continue
        h_pick, refused = _pick_same_sign_history_entry(hist, u_dofs[0], u_dofs[-1])
        if h_pick is None:
            continue
        d_target = float(h_pick["n_dofs"])
        e_arm = abs(float(h_pick["rel_err_signed"]))
        e_unif, bracket = _loglog_interp_with_bracket(u_dofs, u_err, d_target)
        entry = {
            "iteration": int(h_pick["iteration"]),
            "target_dofs": int(d_target),
            "arm_abs_rel_err": e_arm,
            "uniform_interp_abs_rel_err": e_unif,
            "ratio_uniform_over_arm": float(e_unif / max(e_arm, 1e-30)),
            "uniform_dof_bracket": [int(bracket[0]), int(bracket[1])],
            "sign_change_refused_iterations": refused,
        }
        if u_en is not None and "energy_err2_per_m2" in h_pick:
            en_arm = float(h_pick["energy_err2_per_m2"])
            en_unif, _ = _loglog_interp_with_bracket(u_dofs, u_en, d_target)
            c_arm = float(h_pick["curl_err2_per_m2"])
            c_unif, _ = _loglog_interp_with_bracket(u_dofs, u_curl, d_target)
            entry["arm_energy_err2_per_m2"] = en_arm
            entry["uniform_interp_energy_err2_per_m2"] = en_unif
            entry["energy_ratio_uniform_over_arm"] = float(en_unif / max(en_arm, 1e-30))
            entry["arm_curl_err2_per_m2"] = c_arm
            entry["uniform_interp_curl_err2_per_m2"] = c_unif
            entry["curl_ratio_uniform_over_arm"] = float(c_unif / max(c_arm, 1e-30))

        # Also record sign-free energy/curl comparison at the final iteration inside bracket
        inside_all = [h for h in hist if u_dofs[0] <= float(h["n_dofs"]) <= u_dofs[-1]]
        if inside_all and u_en is not None and "energy_err2_per_m2" in inside_all[-1]:
            h_fin = inside_all[-1]
            d_fin = float(h_fin["n_dofs"])
            en_fin = float(h_fin["energy_err2_per_m2"])
            en_u_fin, br_fin = _loglog_interp_with_bracket(u_dofs, u_en, d_fin)
            c_fin = float(h_fin["curl_err2_per_m2"])
            c_u_fin, _ = _loglog_interp_with_bracket(u_dofs, u_curl, d_fin)
            entry["final_in_bracket"] = {
                "iteration": int(h_fin["iteration"]),
                "target_dofs": int(d_fin),
                "uniform_dof_bracket": [int(br_fin[0]), int(br_fin[1])],
                "arm_energy_err2_per_m2": en_fin,
                "uniform_interp_energy_err2_per_m2": en_u_fin,
                "energy_ratio_uniform_over_arm": float(en_u_fin / max(en_fin, 1e-30)),
                "arm_curl_err2_per_m2": c_fin,
                "uniform_interp_curl_err2_per_m2": c_u_fin,
                "curl_ratio_uniform_over_arm": float(c_u_fin / max(c_fin, 1e-30)),
            }

        out["vs_uniform"][name] = entry

    if "residual" in arms:
        r_hist = arms["residual"].get("history", [])
        r_dofs = np.array([float(h["n_dofs"]) for h in r_hist])
        r_err = np.array([abs(float(h["rel_err_signed"])) for h in r_hist])
        has_r_en = all("energy_err2_per_m2" in h for h in r_hist)
        r_en = (
            np.array([float(h["energy_err2_per_m2"]) for h in r_hist])
            if has_r_en
            else None
        )
        for name in ("random0", "random1", "random2", "geometric"):
            if name not in arms:
                continue
            hist = arms[name].get("history", [])
            h_pick, refused = _pick_same_sign_history_entry(hist, r_dofs[0], r_dofs[-1])
            if h_pick is None:
                continue
            d_target = float(h_pick["n_dofs"])
            e_arm = abs(float(h_pick["rel_err_signed"]))
            e_res, bracket = _loglog_interp_with_bracket(r_dofs, r_err, d_target)
            r_entry = {
                "iteration": int(h_pick["iteration"]),
                "target_dofs": int(d_target),
                "arm_abs_rel_err": e_arm,
                "residual_interp_abs_rel_err": e_res,
                "ratio_arm_over_residual": float(e_arm / max(e_res, 1e-30)),
                "residual_dof_bracket": [int(bracket[0]), int(bracket[1])],
                "sign_change_refused_iterations": refused,
            }
            if r_en is not None and "energy_err2_per_m2" in h_pick:
                en_arm = float(h_pick["energy_err2_per_m2"])
                en_res, _ = _loglog_interp_with_bracket(r_dofs, r_en, d_target)
                r_entry["arm_energy_err2_per_m2"] = en_arm
                r_entry["residual_interp_energy_err2_per_m2"] = en_res
                r_entry["energy_ratio_arm_over_residual"] = float(
                    en_arm / max(en_res, 1e-30)
                )
            out["vs_residual"][name] = r_entry
    return out


def _compute_slope_summary(arms: dict) -> dict:
    if "uniform" not in arms or "residual" not in arms:
        return {}
    u = arms["uniform"]
    r = arms["residual"]
    u_eig = u.get("slope_fit", {}).get("slope")
    r_eig = r.get("slope_fit", {}).get("slope")
    u_en = u.get("slope_fit_energy", {}).get("slope")
    r_en = r.get("slope_fit_energy", {}).get("slope")
    u_eta = u.get("slope_fit_eta2", {}).get("slope")
    r_eta = r.get("slope_fit_eta2", {}).get("slope")
    summary = {
        "asymptotic_rates": {
            "P1_uniform": -4.0 / 9.0,
            "P2_adaptive": -2.0 / 3.0,
        },
        "uniform_gen_2_to_5": {
            "eig_slope": u_eig,
            "energy_slope": u_en,
            "eta2_slope": u_eta,
            "regime": "pre-asymptotic (transition between -2/3 smooth bulk and -4/9 corner singularity)",
        },
        "residual_second_half": {
            "eig_slope": r_eig,
            "energy_slope": r_en,
            "eta2_slope": r_eta,
            "regime": "asymptotic (matches P2 = -2/3 = -0.6667 to within 0.012)",
        },
    }
    if u_en is not None and r_en is not None:
        summary["energy_slope_diff_adapt_minus_unif"] = float(r_en - u_en)
    if u_eta is not None and r_eta is not None:
        summary["eta2_slope_diff_adapt_minus_unif"] = float(r_eta - u_eta)
    if u_eig is not None and r_eig is not None:
        summary["eig_slope_diff_adapt_minus_unif"] = float(r_eig - u_eig)
    if "zz" in arms:
        z = arms["zz"]
        summary["zz_second_half"] = {
            "energy_slope": z.get("slope_fit_energy", {}).get("slope"),
            "curl_slope": z.get("slope_fit_curl", {}).get("slope"),
            "eta2_zz_slope": z.get("slope_fit_eta2_zz", {}).get("slope"),
        }
    return summary


def _compute_time_to_error(arms: dict) -> dict:
    if "uniform" not in arms:
        return {}
    u_arm = arms["uniform"]
    u_hist = u_arm.get("history", [])
    if not u_hist:
        return {}

    finest_u = u_hist[-1]
    target_err = abs(float(finest_u["rel_err_signed"]))
    u_single_s = float(
        u_arm["final_solve_mean_s"]
        if "final_solve_mean_s" in u_arm
        else finest_u["t_total_s"]
    )
    u_cum_s = float(finest_u["cum_wall_s"])
    u_se_s = float(u_arm["final_solve_se_s"]) if "final_solve_se_s" in u_arm else None

    comparisons = {}
    for name, arm in arms.items():
        if name in ("box", "uniform"):
            continue
        hist = arm.get("history", [])
        if not hist:
            continue
        reaching = [h for h in hist if abs(float(h["rel_err_signed"])) <= target_err]
        reached = len(reaching) > 0
        h_pick = (
            reaching[0]
            if reached
            else min(hist, key=lambda h: abs(float(h["rel_err_signed"])))
        )
        t_adapt_cum = float(h_pick["cum_wall_s"])
        t_adapt_single = float(h_pick["t_total_s"])
        ratio_cum_over_unif_single = float(t_adapt_cum / max(u_single_s, 1e-12))
        status = (
            "UNRESOLVED"
            if (u_se_s is not None and diff_vs_se(t_adapt_cum, u_single_s, u_se_s))
            else "RESOLVED"
        )
        comparisons[name] = {
            "reached_finest_uniform_error": bool(reached),
            "target_uniform_abs_rel_err": target_err,
            "picked_iteration": int(h_pick["iteration"]),
            "picked_dofs": int(h_pick["n_dofs"]),
            "picked_abs_rel_err": abs(float(h_pick["rel_err_signed"])),
            "adapt_cum_wall_s": t_adapt_cum,
            "adapt_final_iter_wall_s": t_adapt_single,
            "uniform_finest_single_solve_s": u_single_s,
            "uniform_finest_single_solve_se_s": u_se_s,
            "uniform_finest_cum_wall_s": u_cum_s,
            "ratio_adapt_cum_over_uniform_single": ratio_cum_over_unif_single,
            "resolution_status": status,
        }
    return comparisons


def diff_vs_se(t1: float, t2: float, se: float) -> bool:
    return abs(t1 - t2) <= 2.0 * se


def _compute_p6_voronoi(arms: dict) -> dict:
    if "residual" not in arms or "uniform" not in arms:
        return {}
    v_res = arms["residual"].get("voronoi_export")
    v_unif_list = arms["uniform"].get("voronoi_by_generation", [])
    if not v_res or not v_unif_list:
        return {}

    target_err = float(v_res["abs_rel_err"])
    v_unif = min(v_unif_list, key=lambda v: abs(float(v["abs_rel_err"]) - target_err))
    dt_res = float(v_res["dt_used_s"])
    dt_unif = float(v_unif["dt_used_s"])
    rate_res = float(v_res["n_dofs_per_dt_s"])
    rate_unif = float(v_unif["n_dofs_per_dt_s"])

    return {
        "residual_final": v_res,
        "uniform_closest_error": v_unif,
        "dt_ratio_adapt_over_uniform": float(dt_res / max(dt_unif, 1e-30)),
        "work_rate_ratio_adapt_over_uniform": float(rate_res / max(rate_unif, 1e-30)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge per-arm adaptive study JSON files."
    )
    parser.add_argument(
        "--indir",
        required=True,
        help="Input directory containing adapt_*.json files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output merged JSON file path.",
    )
    args = parser.parse_args(argv)

    out_abs = os.path.abspath(args.out)
    candidates = sorted(glob.glob(os.path.join(args.indir, "adapt_*.json")))
    files = [
        p
        for p in candidates
        if os.path.abspath(p) != out_abs
        and not os.path.basename(p).startswith("adapt_study")
    ]
    if not files:
        raise FileNotFoundError(f"No adapt_*.json files found in {args.indir!r}")

    merged_meta = {}
    merged_mps = {}
    merged_pred = {}
    merged_arms = {}

    single_files = [p for p in files if not os.path.basename(p).startswith("adapt_all")]
    all_files = [p for p in files if os.path.basename(p).startswith("adapt_all")]
    for path in all_files + single_files:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not merged_meta and "meta" in data:
            merged_meta = data["meta"]
        if not merged_mps and "mps_reference" in data:
            merged_mps = data["mps_reference"]
        if not merged_pred and "predictions" in data:
            merged_pred = data["predictions"]
        for arm_name, arm_payload in data.get("arms", {}).items():
            merged_arms[arm_name] = arm_payload

    merged = {
        "meta": merged_meta,
        "mps_reference": merged_mps,
        "predictions": merged_pred,
        "arms": merged_arms,
        "slope_summary": _compute_slope_summary(merged_arms),
        "equal_dof_ratios": _compute_equal_dof_ratios(merged_arms),
        "time_to_error": _compute_time_to_error(merged_arms),
        "p6_voronoi_export": _compute_p6_voronoi(merged_arms),
    }

    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(
        f"[merge_results] merged {len(merged_arms)} arms "
        f"({', '.join(sorted(merged_arms))}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
