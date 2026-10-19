# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Equal-work comparison between the `subtri-hybrid` arms and baseline arms from
the shared benchmark store (`wakis.benchmarks.results.fetch`).

Usage
-----
    python studies/subtri-hybrid/compare.py [--offline] [--data-dir DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wakis.benchmarks import metrics as M  # noqa: E402
from wakis.benchmarks import results  # noqa: E402

os.environ.setdefault("CLOUDSDK_STORAGE_PROCESS_COUNT", "1")
os.environ.setdefault("CLOUDSDK_STORAGE_THREAD_COUNT", "4")

DATA = HERE / "data"

B1_OURS = ("subtri-hybrid-r2", "subtri-hybrid-r4")
B1_BASE = (
    "tri-graded-r2",
    "tri-graded-r4",
    "cart-conformal-all-graded-r2",
    "cart-conformal-all-graded-r4",
)

B2_OURS = ("subtri-hybrid", "subtri-hybrid-nolts")
B2_BASE = (
    "hybrid-quadtri",
    "tri-graded",
    "tri-graded-pml",
    "tri-graded-pml-lts2",
    "cart-conformal-all-graded",
    "nfit-ogrid",
    "dgtd-p2",
    "dgtd-p3",
    "subgrid-2to1",
    "voronoi-poly",
)

B3_OURS = ("subtri-hybrid", "subtri-hybrid-nolts")
B3_BASE = (
    "tri-graded-pml2-patch",
    "hybrid-quadtri",
    "tri-graded",
    "cart-conformal-all-graded",
    "nfit-boxed",
    "dgtd-p3",
    "subgrid-2to1-gap4-lts8",
)

CASE_SPECS = {
    "B1-TE": {
        "metrics": [("q_sca_rms_rel", (0.02, 0.01))],
        "ours": B1_OURS,
        "baselines": B1_BASE,
    },
    "B1-TM": {
        "metrics": [("q_sca_rms_rel", (0.02, 0.01))],
        "ours": B1_OURS,
        "baselines": B1_BASE,
    },
    "B2-Au-TE": {
        "metrics": [
            ("q_ext_nrms", (0.02, 0.01, 0.005, 0.001)),
            ("q_ext_lam_peak", (0.005, 0.002)),
        ],
        "ours": B2_OURS,
        "baselines": B2_BASE,
    },
    "B2-Au-TM": {
        "metrics": [
            ("q_ext_nrms", (0.02, 0.01, 0.005, 0.001)),
            ("q_ext_lam_peak", (0.005, 0.002)),
        ],
        "ours": B2_OURS,
        "baselines": B2_BASE,
    },
    "B2-Ag-TE": {
        "metrics": [
            ("q_ext_nrms", (0.02, 0.01, 0.005, 0.001)),
            ("q_ext_lam_peak", (0.005, 0.002)),
        ],
        "ours": B2_OURS,
        "baselines": B2_BASE,
    },
    "B2-Ag-TM": {
        "metrics": [
            ("q_ext_nrms", (0.02, 0.01, 0.005, 0.001)),
            ("q_ext_lam_peak", (0.005, 0.002)),
        ],
        "ours": B2_OURS,
        "baselines": B2_BASE,
    },
    "B3-g10": {
        "metrics": [
            ("enh_peak", (0.05, 0.02, 0.01)),
            ("q_ext_nrms", (0.02, 0.01)),
        ],
        "ours": B3_OURS,
        "baselines": B3_BASE,
    },
    "B3-g5": {
        "metrics": [
            ("enh_peak", (0.05, 0.02, 0.01)),
            ("q_ext_nrms", (0.02, 0.01)),
        ],
        "ours": B3_OURS,
        "baselines": B3_BASE,
    },
    "B3-g2": {
        "metrics": [
            ("enh_peak", (0.05, 0.02, 0.01)),
            ("q_ext_nrms", (0.02, 0.01)),
        ],
        "ours": B3_OURS,
        "baselines": B3_BASE,
    },
}


def record_to_row(r):
    """
    Convert a raw benchmark record dict to a flat row matching `results.fetch`.

    Parameters
    ----------
    r : dict
        Record produced by `wakis.benchmarks.run`.

    Returns
    -------
    dict
        Flattened row with ``work`` = ``cost['work']`` if present else
        ``unknowns * steps``.
    """
    row = {
        k: r.get(k)
        for k in (
            "case",
            "arm",
            "resolution",
            "commit",
            "cpu",
            "machine",
            "published",
            "error",
        )
    }
    for k, v in r.get("metrics", {}).items():
        if not isinstance(v, (dict, list)):
            row[f"m.{k}"] = v
    cost = r.get("cost", {})
    for k, v in cost.items():
        if not isinstance(v, (dict, list)):
            row[f"c.{k}"] = v
    if "c.unknowns" in row and "c.steps" in row:
        row["work"] = row["c.unknowns"] * row["c.steps"]
    if cost.get("work") is not None:
        row["work"] = cost["work"]
    return row


def load_local_records(data_dir=DATA):
    """
    Load local JSON benchmark records from ``data_dir``.

    Parameters
    ----------
    data_dir : Path or str
        Directory containing local JSON files (e.g. ``data/<phase>/*.json``).

    Returns
    -------
    list of dict
        Raw benchmark records.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    recs = []
    for p in sorted(data_dir.rglob("*.json")):
        if p.name == "compare.json":
            continue
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        if (
            isinstance(obj, dict)
            and "case" in obj
            and "arm" in obj
            and "resolution" in obj
        ):
            recs.append(obj)
    return recs


def load_local_df(data_dir=DATA):
    """Return local benchmark records as a flattened DataFrame."""
    rows = [record_to_row(r) for r in load_local_records(data_dir)]
    return pd.DataFrame(rows)


def load_baselines_df(case, data_dir=DATA, offline=False):
    """
    Fetch or load cached baseline records for ``case``.

    Parameters
    ----------
    case : str
        Benchmark case identifier.
    data_dir : Path or str
        Directory where ``baselines_<case>.csv`` is stored.
    offline : bool
        If True, read from cached CSV without network access.

    Returns
    -------
    pandas.DataFrame
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / f"baselines_{case}.csv"
    if not offline:
        df = results.fetch(case)
        df.to_csv(cache_path, index=False)
        return df
    if cache_path.exists():
        return pd.read_csv(cache_path)
    fallback = DATA / f"baselines_{case}.csv"
    if fallback.exists():
        return pd.read_csv(fallback)
    return pd.DataFrame()


def dedup_latest(df):
    """
    Deduplicate records per ``(arm, resolution)`` keeping the latest ``published``.

    Parameters
    ----------
    df : pandas.DataFrame
        Flattened benchmark records for a case.

    Returns
    -------
    pandas.DataFrame
        Filtered (error-free) and deduplicated DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "error" in d.columns:
        d = d[d["error"].isna() | (d["error"].astype(str).str.strip() == "")]
    if d.empty:
        return d
    # diverged runs (e.g. unstable fine DGTD rungs with errors 1e3..1e33) are
    # not converged results: drop rows whose relative errors exceed 1
    mcols = [c for c in ("m.q_ext_nrms", "m.enh_peak", "m.q_sca_rms_rel") if c in d.columns]
    if mcols:
        bad = (d[mcols].apply(pd.to_numeric, errors="coerce") > 1.0).any(axis=1)
        d = d[~bad]
    if "work" not in d.columns and "c.unknowns" in d.columns and "c.steps" in d.columns:
        d["work"] = d["c.unknowns"] * d["c.steps"]
    if "c.work" in d.columns:
        has_cw = d["c.work"].notna()
        d.loc[has_cw, "work"] = d.loc[has_cw, "c.work"]
    d["_res_key"] = np.round(d["resolution"].astype(float) * 1e9, 6)
    if "published" in d.columns:
        d["_pub_key"] = d["published"].fillna("").astype(str)
    else:
        d["_pub_key"] = ""
    d = d.sort_values("_pub_key", kind="mergesort")
    d = d.drop_duplicates(subset=["arm", "_res_key"], keep="last")
    return d.drop(columns=["_res_key", "_pub_key"])


def load_case_df(case, data_dir=DATA, offline=False, local_df=None):
    """Combine deduplicated local records and baseline records for ``case``."""
    spec = CASE_SPECS[case]
    if local_df is None:
        local_df = load_local_df(data_dir)
    ours_df = pd.DataFrame()
    if not local_df.empty and "case" in local_df.columns:
        ours_df = local_df[local_df["case"] == case]
    base_df = load_baselines_df(case, data_dir=data_dir, offline=offline)
    if not base_df.empty and "arm" in base_df.columns and not ours_df.empty:
        local_arms = set(ours_df["arm"].unique())
        base_df = base_df[~base_df["arm"].isin(local_arms)]
    parts = [p for p in (ours_df, base_df) if not p.empty]
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined = dedup_latest(combined)
    allowed = set(spec["ours"]) | set(spec["baselines"])
    return combined[combined["arm"].isin(allowed)].copy()


def interp_arm_target(df_arm, metric, target):
    """
    Interpolate ``unknowns``, ``work``, ``run_s`` to ``target`` for one arm.

    Parameters
    ----------
    df_arm : pandas.DataFrame
        Deduplicated records of a single arm and case.
    metric : str
        Metric name (without ``m.`` prefix).
    target : float
        Target relative error.

    Returns
    -------
    dict
        ``{"unknowns": float, "work": float, "run_s": float}`` where negative
        values denote upper bounds (``<=``) and ``np.nan`` denotes not reached.
    """
    mcol = f"m.{metric}"
    if df_arm is None or df_arm.empty or mcol not in df_arm.columns:
        return {"unknowns": np.nan, "work": np.nan, "run_s": np.nan}
    err_raw = pd.to_numeric(df_arm[mcol], errors="coerce").to_numpy(float)
    err = np.abs(err_raw)
    out = {}
    for key, col in (
        ("unknowns", "c.unknowns"),
        ("work", "work"),
        ("run_s", "c.run_s"),
    ):
        if col not in df_arm.columns:
            out[key] = np.nan
            continue
        cost = pd.to_numeric(df_arm[col], errors="coerce").to_numpy(float)
        mask = np.isfinite(cost) & np.isfinite(err) & (cost > 0)
        if not np.any(mask):
            out[key] = np.nan
            continue
        out[key] = float(M.interp_to_target(cost[mask], err[mask], target))
    return out


def fmt_unknowns(u):
    """Format unknowns like ``5.6k`` with ``≤`` prefix for upper bounds."""
    if not np.isfinite(u):
        return "—"
    prefix = "≤" if u < 0 else ""
    v = abs(float(u))
    if v >= 1e6:
        return f"{prefix}{v / 1e6:.2f}M"
    if v >= 1e5:
        return f"{prefix}{v / 1e3:.0f}k"
    if v >= 1e3:
        return f"{prefix}{v / 1e3:.1f}k"
    return f"{prefix}{v:.0f}"


def fmt_work(w):
    """Format work like ``4.8e7`` with ``≤`` prefix for upper bounds."""
    if not np.isfinite(w):
        return "—"
    prefix = "≤" if w < 0 else ""
    s = (
        f"{abs(float(w)):.1e}".replace("e+0", "e")
        .replace("e+", "e")
        .replace("e-0", "e-")
    )
    return f"{prefix}{s}"


def fmt_time(t):
    """Format run time [s] like ``12 s`` with ``≤`` prefix for upper bounds."""
    if not np.isfinite(t):
        return "—"
    prefix = "≤" if t < 0 else ""
    v = abs(float(t))
    if v >= 10.0:
        return f"{prefix}{v:.0f} s"
    return f"{prefix}{v:.1f} s"


def fmt_cell(res):
    """Format a table cell as ``unknowns / work / time`` or ``—``."""
    u, w, s = res["unknowns"], res["work"], res["run_s"]
    if not np.isfinite(w):
        return "—"
    return f"{fmt_unknowns(u)} / {fmt_work(w)} / {fmt_time(s)}"


def _clean_json_val(v):
    return None if (v is None or not np.isfinite(v)) else float(v)


def compare_all(data_dir=DATA, offline=False):
    """
    Run equal-work comparison across all cases, write `compare.md` and
    `compare.json`, and print a one-line summary per case.

    Parameters
    ----------
    data_dir : Path or str
        Data directory for reading local JSONs and writing outputs.
    offline : bool
        If True, use cached baseline CSVs instead of fetching from GCS.

    Returns
    -------
    dict
        Structured comparison dictionary (also saved to ``compare.json``).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    local_df = load_local_df(data_dir)

    md_lines = [
        "# Equal-Work Benchmark Comparison (`subtri-hybrid`)",
        "",
        "Cells report `unknowns / work / time` interpolated log-log to each target "
        "(`≤` = upper bound at coarsest rung, `—` = target not reached).",
        "",
    ]
    json_out = {"cases": {}}

    for case, spec in CASE_SPECS.items():
        df = load_case_df(case, data_dir=data_dir, offline=offline, local_df=local_df)
        present_arms = (
            set(df["arm"].unique()) if not df.empty and "arm" in df.columns else set()
        )
        our_arms = list(spec["ours"])
        base_arms = [a for a in spec["baselines"] if a in present_arms]
        if not base_arms:
            base_arms = list(spec["baselines"])
        all_arms = our_arms + base_arms

        md_lines.append(f"## {case}")
        md_lines.append("")
        header = (
            "| target | "
            + " | ".join(all_arms)
            + " | best baseline (work) | ours / best (work) | best baseline (time): ours / best |"
        )
        sep = "|---|" + "---|" * len(all_arms) + "---|---|---|"
        md_lines.append(header)
        md_lines.append(sep)

        case_json = {"ours": our_arms, "baselines": base_arms, "targets": []}
        summary_parts = []

        for metric, targets in spec["metrics"]:
            for target in targets:
                t_label = f"{metric} ≤ {target * 100:g} %"
                arm_res = {}
                for arm in all_arms:
                    df_arm = (
                        df[df["arm"] == arm]
                        if not df.empty and "arm" in df.columns
                        else pd.DataFrame()
                    )
                    arm_res[arm] = interp_arm_target(df_arm, metric, target)

                # Best baseline by work
                best_base_arm = None
                best_base_work = np.nan
                for arm in base_arms:
                    w = arm_res[arm]["work"]
                    if np.isfinite(w) and (
                        not np.isfinite(best_base_work) or abs(w) < abs(best_base_work)
                    ):
                        best_base_arm = arm
                        best_base_work = w

                # Best of our arms by work
                best_our_arm = None
                best_our_work = np.nan
                for arm in our_arms:
                    w = arm_res[arm]["work"]
                    if np.isfinite(w) and (
                        not np.isfinite(best_our_work) or abs(w) < abs(best_our_work)
                    ):
                        best_our_arm = arm
                        best_our_work = w

                # best baseline / ours by wall time (equal-cost check independent
                # of the per-unknown cost of each method, e.g. DG blocks)
                def _best(arms_, key):
                    b, bv = None, np.nan
                    for a_ in arms_:
                        v_ = arm_res[a_].get(key, np.nan)
                        if np.isfinite(v_) and (not np.isfinite(bv) or abs(v_) < abs(bv)):
                            b, bv = a_, v_
                    return b, bv

                bt_arm, bt = _best(base_arms, "run_s")
                ot_arm, ot = _best(our_arms, "run_s")
                t_ratio = (abs(ot) / abs(bt)) if (np.isfinite(ot) and np.isfinite(bt)) else np.nan
                t_str = (f"{bt_arm} ({abs(bt):.0f} s): {t_ratio:.2f}"
                         if np.isfinite(t_ratio) else "—")

                if best_base_arm is not None and np.isfinite(best_base_work):
                    best_base_str = f"{best_base_arm} ({fmt_work(best_base_work)})"
                else:
                    best_base_str = "—"

                if np.isfinite(best_our_work) and np.isfinite(best_base_work):
                    ratio = abs(float(best_our_work)) / abs(float(best_base_work))
                    ratio_str = f"{ratio:.2f}" if ratio >= 0.01 else f"{ratio:.2g}"
                else:
                    ratio = np.nan
                    ratio_str = "—"

                cells = [fmt_cell(arm_res[a]) for a in all_arms]
                md_lines.append(
                    f"| {t_label} | "
                    + " | ".join(cells)
                    + f" | {best_base_str} | {ratio_str} | {t_str} |"
                )

                summary_parts.append(
                    f"{metric} {target * 100:g}%: "
                    f"{best_base_arm or '—'} (ours/best={ratio_str}; time {t_str})"
                )

                case_json["targets"].append(
                    {
                        "metric": metric,
                        "target": float(target),
                        "arms": {
                            a: {k: _clean_json_val(v) for k, v in arm_res[a].items()}
                            for a in all_arms
                        },
                        "best_baseline": {
                            "arm": best_base_arm,
                            "work": _clean_json_val(best_base_work),
                        },
                        "best_ours": {
                            "arm": best_our_arm,
                            "work": _clean_json_val(best_our_work),
                        },
                        "ours_over_best_work": _clean_json_val(ratio),
                    }
                )

        md_lines.append("")
        json_out["cases"][case] = case_json
        print(f"{case}: " + " | ".join(summary_parts))

    (data_dir / "compare.md").write_text("\n".join(md_lines) + "\n")
    (data_dir / "compare.json").write_text(json.dumps(json_out, indent=2) + "\n")
    return json_out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Use cached data/baselines_<case>.csv without network access.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="Directory containing local JSON records and output files.",
    )
    args = ap.parse_args(argv)
    compare_all(data_dir=args.data_dir, offline=args.offline)


if __name__ == "__main__":
    main()
