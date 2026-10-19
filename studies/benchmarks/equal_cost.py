"""Equal-cost tables from the shared benchmark store (`wakis.benchmarks.results.fetch`).

For every case and arm: unknowns, work (unknowns x steps) and run time at
which the metric first reaches each target (log-log interpolation along the
arm's ladder ordered by unknowns; '<' = already below at the coarsest run),
plus ratios vs the best Cartesian arm.

    python equal_cost.py > ../../studies/benchmarks/data/equal_cost.md
"""

import numpy as np

from wakis.benchmarks import metrics as M
from wakis.benchmarks import results

TARGETS = {
    "B2-Au-TE": ("q_ext_nrms", (1e-2, 1e-3)),
    "B2-Au-TM": ("q_ext_nrms", (1e-2, 1e-3)),
    "B2-Ag-TE": ("q_ext_nrms", (1e-2, 1e-3)),
    "B2-Ag-TM": ("q_ext_nrms", (1e-2, 1e-3)),
    "B3-g10": ("enh_peak", (0.05,)),
    "B3-g5": ("enh_peak", (0.05,)),
    "B3-g2": ("enh_peak", (0.05,)),
}


def cost_at(df, metric, target, col):
    d = df.dropna(subset=[f"m.{metric}"]).sort_values("c.unknowns")
    if d.empty:
        return np.nan, ""
    err = np.abs(d[f"m.{metric}"].to_numpy())
    cost = d[col].to_numpy(float)
    v = M.interp_to_target(cost, err, target)
    flag = "<" if np.isfinite(v) and err[0] <= target else ""
    return v, flag


def main():
    for case, (metric, targets) in TARGETS.items():
        df = results.fetch(case)
        if df.empty:
            print(f"## {case}: no records\n")
            continue
        if "error" in df:
            df = df[df["error"].isna()]
        for t in targets:
            print(f"## {case}: {metric} <= {t:g}\n")
            print(
                "| arm | runs | best error | unknowns | work | run_s (single run, noisy ±30 %) |"
            )
            print("|---|---|---|---|---|---|")
            rows = {}
            for arm, g in sorted(df.groupby("arm")):
                if f"m.{metric}" not in g:
                    continue
                u, fu = cost_at(g, metric, t, "c.unknowns")
                w, fw = cost_at(g, metric, t, "work")
                s, fs = cost_at(g, metric, t, "c.run_s")
                best = np.nanmin(np.abs(g[f"m.{metric}"].to_numpy(float)))
                rows[arm] = (u, w, s)
                print(
                    f"| {arm} | {len(g)} | {best:.3g} | {fu}{u:,.0f} | {fw}{w:.3g} | {fs}{s:,.0f} |"
                )
            cart = {
                a: v
                for a, v in rows.items()
                if a.startswith("cart") and np.isfinite(v[1])
            }
            if cart:
                best_c = min(cart, key=lambda a: cart[a][1])
                print(f"\nRatios vs the cheapest Cartesian arm by work ({best_c}): "
                      "unknowns / work / time")  # fmt: skip
                for arm, (u, w, s) in rows.items():
                    if arm.startswith("tri") and np.isfinite(w):
                        cu, cw, cs = cart[best_c]
                        print(f"- {arm}: {cu / u:.2f}x / {cw / w:.2f}x / {cs / s:.2f}x")
            print()


if __name__ == "__main__":
    main()
