"""
Verify and emit numbers for the 3D tetrahedral LTS physics guide subsection.

Modes:
- `--emit-markdown`: format result tables from `results/*.json` (or `*_quick.json` with `--quick`).
- `--check`: verify that every numeric literal in the study results subsection of
  `docs/physicsguide_sections/unstructured_lts3d.md` comes from the emitted JSON tables.
  Exits 0 with a notice if no full study results exist yet and the placeholder comment is intact.
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GUIDE = os.path.join(ROOT, "docs", "physicsguide_sections", "unstructured_lts3d.md")
PLACEHOLDER = "<!-- RESULTS: filled from studies/unstructured-lts3d/results/*.json after the Batch study -->"


def _has_full_results():
    return all(
        os.path.exists(os.path.join(RES, f"{name}.json"))
        for name in ("bounds", "accuracy", "timing", "stability")
    )


def emit_markdown(quick=False):
    suffix = "_quick" if quick else ""
    acc_path = os.path.join(RES, f"accuracy{suffix}.json")
    tim_path = os.path.join(RES, f"timing{suffix}.json")
    stab_path = os.path.join(RES, f"stability{suffix}.json")
    if not (
        os.path.exists(acc_path)
        and os.path.exists(tim_path)
        and os.path.exists(stab_path)
    ):
        return "No study JSON files found."

    with open(acc_path, encoding="utf-8") as fh:
        acc = json.load(fh)
    with open(tim_path, encoding="utf-8") as fh:
        tim = json.load(fh)
    with open(stab_path, encoding="utf-8") as fh:
        stab = json.load(fh)

    lines = [
        "### Accuracy and Cost Summary",
        "",
        "| Case | Size | Seed | Arm | n_tets | n_free | dt_lts / (0.95 dt_global) | err_semi | err_td | work_per_ns | wall_per_ns [s/ns] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in acc.get("records", []):
        lines.append(
            f"| {r['case']} | {r['size']} | {r['seed']} | {r['arm']} | "
            f"{r['n_tets']} | {r['n_free']} | {r['ratio']:.3f} | "
            f"{r['err_semi']:.3e} | {r['err_td']:.3e} | {r['work_per_ns']:.3e} | "
            f"{r['wall_per_ns']:.4f} |"
        )

    lines += _setup_inclusive_lines(acc)
    lines += [
        "",
        "### Interleaved Timing Summary",
        "",
        "| Case | Size | Arm | Median wall [s/ns] | Min wall [s/ns] | Max wall [s/ns] | Speedup vs G |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in tim.get("timing", []):
        lines.append(
            f"| {r['case']} | {r['size']} | {r['arm']} | "
            f"{r['wall_per_ns_median']:.4f} | {r['wall_per_ns_min']:.4f} | "
            f"{r['wall_per_ns_max']:.4f} | {r['speedup_vs_G']:.3f} |"
        )

    lines += [
        "",
        (
            f"Stability ratio: predicted={stab['pred_global_limit_ratio']:.5f}, "
            f"measured={stab['meas_dt_lts_over_dt_global']:.5f}"
        ),
    ]
    return "\n".join(lines)


def _setup_inclusive_lines(acc):
    """Equal-error stepping gains and setup-inclusive total wall gains per (case, arm).

    The total wall time of a run of the study's own simulated length
    ``T = n_steps * dt_lts`` is ``setup_total_s + wall_per_ns * T``; the baseline is
    the global-leapfrog arm of the same Hodge (``G`` for Voronoi, ``GW`` for Whitney)
    on the same (case, size, seed), run for the same ``T``. Values are medians over
    sizes and seeds, with [min, max].
    """
    import numpy as np

    recs = acc.get("records", [])
    base = {}
    for r in recs:
        if r["arm"] in ("G", "GW"):
            base[(r["case"], r["size"], r["seed"], r["arm"])] = r
    lines = [
        "",
        "### Equal-error stepping gains (setup excluded)",
        "",
        "| Case | Comparison | Status | Median gain | Range |",
        "|---|---|---|---|---|",
    ]
    for case, comps in sorted(acc.get("comparisons", {}).items()):
        for name, c in comps.items():
            if c["median_gain"] is None:
                lines.append(f"| {case} | {name} | {c['status']} | - | - |")
            else:
                lines.append(
                    f"| {case} | {name} | {c['status']} | {c['median_gain']:.2f} | "
                    f"[{c['range'][0]:.2f}, {c['range'][1]:.2f}] |"
                )
    lines += [
        "",
        "### Setup-inclusive cost at the study run length",
        "",
        "| Case | Arm | Baseline | Runs | Median T [ns] | Median setup [s] | Max setup [s] | Median certificate [s] | Setup share of total | Total-wall gain (median) | Total-wall gain [min, max] | Median break-even [ns] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    arms = sorted({r["arm"] for r in recs if r["arm"].startswith("L")})
    for case in sorted({r["case"] for r in recs}):
        for arm in arms:
            b_arm = "GW" if arm.startswith("LW") else "G"
            rows = []
            for r in recs:
                if r["case"] != case or r["arm"] != arm:
                    continue
                g = base.get((r["case"], r["size"], r["seed"], b_arm))
                if g is None:
                    continue
                T = r["n_steps"] * r["dt_lts"] * 1e9
                tot = r["setup_total_s"] + r["wall_per_ns"] * T
                tot_g = g["setup_total_s"] + g["wall_per_ns"] * T
                saving = g["wall_per_ns"] - r["wall_per_ns"]
                be = (r["setup_total_s"] - g["setup_total_s"]) / saving if saving > 0 else np.inf
                rows.append(
                    (T, r["setup_total_s"], r["setup_time_s"].get("certificate", 0.0),
                     r["setup_total_s"] / tot, tot_g / tot, be)
                )
            if not rows:
                continue
            a = np.asarray(rows, dtype=float)
            be_med = np.median(a[:, 5])
            be_txt = "never" if not np.isfinite(be_med) else f"{be_med:.0f}"
            lines.append(
                f"| {case} | {arm} | {b_arm} | {len(rows)} | {np.median(a[:, 0]):.1f} | "
                f"{np.median(a[:, 1]):.2f} | {a[:, 1].max():.1f} | {np.median(a[:, 2]):.2f} | "
                f"{np.median(a[:, 3]):.2f} | {np.median(a[:, 4]):.3f} | "
                f"[{a[:, 4].min():.3f}, {a[:, 4].max():.3f}] | {be_txt} |"
            )
    return lines


def _extract_study_subsection(text):
    """Extract the `## 4. Study:` subsection up to `## 5. Limits`."""
    m = re.search(r"##\s+4\.\s+Study.*?(?=##\s+5\.\s+Limits|\Z)", text, flags=re.DOTALL)
    return m.group(0) if m else ""


def check_guide():
    if not os.path.exists(GUIDE):
        print(f"ERROR: guide file missing at {GUIDE}", file=sys.stderr)
        return 1
    with open(GUIDE, encoding="utf-8") as fh:
        guide_text = fh.read()

    sub = _extract_study_subsection(guide_text)
    if not sub:
        print(
            "ERROR: Study subsection `## 4. Study` not found in guide.", file=sys.stderr
        )
        return 1

    if not _has_full_results():
        if PLACEHOLDER not in sub:
            print(
                "ERROR: Full study results do not exist yet, and placeholder comment is missing.",
                file=sys.stderr,
            )
            return 1
        print(
            "NOTICE: No full study JSON files in studies/unstructured-lts3d/results/ yet; "
            "study results placeholder verified."
        )
        return 0

    tables_md = emit_markdown(quick=False)
    allowed_tokens = set(re.findall(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", tables_md))
    # Ignore structural tokens in the method description (e.g., C1..C4, 1..4)
    allowed_tokens.update(
        {
            "0",
            "1",
            "2",
            "3",
            "4",
            "10",
            "15",
            "20",
            "25",
            "40",
            "95",
            "0.95",
            "1.25",
            "1.5",
            "2.0",
            "3.0",
        }
    )
    after_placeholder = sub.split(PLACEHOLDER, 1)[-1] if PLACEHOLDER in sub else sub
    found = re.findall(r"\b\d+\.\d+(?:e[+-]?\d+)?\b", after_placeholder)
    missing = [tok for tok in found if tok not in allowed_tokens]
    if missing:
        print(
            f"ERROR: Unverified numbers in study subsection: {missing}", file=sys.stderr
        )
        return 1
    print("verify_numbers: all study numbers verified against JSON tables.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-markdown", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.emit_markdown:
        print(emit_markdown(quick=args.quick))
    if args.check:
        raise SystemExit(check_guide())


if __name__ == "__main__":
    main()
