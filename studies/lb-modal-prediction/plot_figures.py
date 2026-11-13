"""D21 figures: TD settle scan vs the first-pass and legacy predictions, and the legacy error decomposition.

Usage: python studies/lb-modal-prediction/plot_figures.py [OUT_DIR] [IMG_DIR]
Reads ``OUT_DIR/scan_<grid>.json`` (``td_settle_scan.py``) and ``OUT_DIR/d21_numbers_<grid>.json``
(``d21_numbers.py``) and writes ``IMG_DIR/lb_d21_settle_scan.png`` and
``IMG_DIR/lb_d21_error_decomposition.png`` (defaults: ``studies/lb-modal-prediction/out``, ``docs/img``).
Plotting only: every number comes from those JSON files.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GRIDS = (("coarse", "10x474"), ("fine", "20x948"))
N_DFT_PERIODS = 6  # td_settle_scan.py / run_linearized_beam_drift DFT window length

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.9,
        "savefig.dpi": 200,
    }
)
C_D21 = "#0b3c8c"  # dark blue
C_LEG = "#c0392b"  # dark red
C_BAND = "#9ab8e6"


def load(out_dir: str) -> dict:
    data = {}
    for name, tag in GRIDS:
        with open(os.path.join(out_dir, f"scan_{tag}.json")) as fh:
            scan = json.load(fh)
        with open(os.path.join(out_dir, f"d21_numbers_{tag}.json")) as fh:
            num = json.load(fh)
        assert (scan["nx"], scan["nz"]) == (num["nx"], num["nz"])
        data[name] = (scan, num)
    return data


def _signed_log(ax, x, r, color, marker, label):
    r = np.asarray(r)
    pos = r > 0
    ax.plot(x, np.abs(r), "-", color=color, lw=1.1, alpha=0.6, zorder=2)
    ax.plot(x[pos], r[pos], marker, color=color, ms=6, ls="none", zorder=3, label=label + " (> 0)")
    ax.plot(
        x[~pos], -r[~pos], marker, mfc="white", mec=color, mew=1.4, ms=6, ls="none",
        zorder=3, label=label + " (< 0)",
    )


def fig_settle(data: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True)
    for ax, (name, tag) in zip(axes, GRIDS):
        scan, num = data[name]
        d21, dec = num["d21"], num["decomposition"]
        dt, n_spp, t_reach0 = scan["dt"], scan["n_spp"], d21["t_reach0"]
        starts = np.array([w["start"] for w in scan["windows"]])
        x = (starts + N_DFT_PERIODS * n_spp) * dt / t_reach0
        bq = np.array([w["bq"] for w in scan["windows"]])
        r_d21 = bq / d21["bq_pred"] - 1.0
        r_leg = bq / dec["bq_legacy"] - 1.0
        x_prod = num["t_end_over_t_reach0"]
        ax.axvspan(1.0, x.max() * 1.05, color="0.9", zorder=0)
        ax.text(
            0.99 * x.max(), 2.2e-3, "reflection has reached\nthe entrance\n(first pass invalid)",
            fontsize=8, va="top", ha="right", color="0.25",
        )
        ax.fill_between(
            [x.min() * 0.9, 1.0], 1e-8, d21["slack"], color=C_BAND, alpha=0.55, lw=0,
            zorder=1, label="D21 derived slack (production window)",
        )
        ax.axvline(1.0, color="0.3", lw=0.9, ls="--", zorder=1)
        _signed_log(ax, x, r_leg, C_LEG, "s", "legacy cos form")
        _signed_log(ax, x, r_d21, C_D21, "o", "D21 first pass")
        ax.plot(
            [x_prod], [abs(num["resid_d21"])], "*", ms=15, mfc="gold", mec="k", mew=0.9,
            zorder=4, label="production run",
        )
        ax.plot(
            [x_prod], [abs(num["resid_legacy"])], "*", ms=15, mfc="gold", mec="k", mew=0.9,
            zorder=4,
        )
        ax.annotate(
            f"production\n$t_{{end}}/t_{{reach,0}} = {x_prod:.2f}$",
            xy=(x_prod, abs(num["resid_d21"])), xytext=(x_prod + 0.08, 1.2e-7),
            fontsize=8, arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8),
            va="center",
        )
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 3e-3)
        ax.set_xlim(x.min() * 0.92, x.max() * 1.04)
        ax.set_xlabel(r"DFT window end / reflection arrival, $t_{\mathrm{end}}/t_{\mathrm{reach},0}$")
        ax.set_title(f"{name} grid ({scan['nx']} × {scan['nz']})")
        ax.grid(True, which="major", color="0.85", lw=0.6)
    axes[0].set_ylabel(r"$|\beta_{q,\mathrm{LB}}/\beta_{q,\mathrm{pred}} - 1|$")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    # the slack differs per grid: label it inside each panel
    for ax, (name, _) in zip(axes, GRIDS):
        d21 = data[name][1]["d21"]
        ax.text(
            ax.get_xlim()[0] + 0.015, 1.6e-8,
            f"D21 slack\n{d21['slack']:.1e}", fontsize=8, color=C_D21, va="bottom",
        )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_decomposition(data: dict, path: str) -> None:
    labels = [
        "dispersion\nsymbol ($1/c_h$)",
        "entrance\namplitudes",
        "PEC-wall\nnear field",
        "sum\n(D21 − legacy)",
        "measured\nlegacy residual",
        "measured\nD21 residual",
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    width = 0.38
    xs = np.arange(len(labels))
    for i, (name, tag) in enumerate(GRIDS):
        scan, num = data[name]
        dec = num["decomposition"]
        vals = np.array(
            [
                dec["d_symbol"],
                dec["d_amplitude"],
                dec["d_wall"],
                dec["d_total"],
                num["resid_legacy"],
                num["resid_d21"],
            ]
        )
        off = (i - 0.5) * width
        bars = ax.bar(
            xs + off, vals * 1e4, width, color=["#4a6fa5", "#a8c3e6"][i],
            edgecolor="k", lw=0.7, hatch=["", "//"][i],
            label=f"{name} grid ({scan['nx']} × {scan['nz']})",
        )
        for b, v in zip(bars, vals):
            y = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2, y + (0.08 if y >= 0 else -0.08),
                f"{v:+.2e}", ha="center", va="bottom" if y >= 0 else "top",
                fontsize=7.2, rotation=90,
            )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(2.5, color="0.5", lw=0.8, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"relative $\beta_q$ difference  [$10^{-4}$]")
    ax.set_ylim(-3.6, 5.2)
    ax.grid(True, axis="y", color="0.85", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(
        "Legacy cos-form LinearizedBeam prediction: three cancelling errors removed by D21",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "studies/lb-modal-prediction/out"
    img_dir = sys.argv[2] if len(sys.argv) > 2 else "docs/img"
    os.makedirs(img_dir, exist_ok=True)
    data = load(out_dir)
    fig_settle(data, os.path.join(img_dir, "lb_d21_settle_scan.png"))
    fig_decomposition(data, os.path.join(img_dir, "lb_d21_error_decomposition.png"))
    for name, _ in GRIDS:
        scan, num = data[name]
        print(
            name,
            json.dumps(
                {
                    "t_end/t_reach0": num["t_end_over_t_reach0"],
                    "bq_pred": num["d21"]["bq_pred"],
                    "slack": num["d21"]["slack"],
                    "bq_legacy": num["decomposition"]["bq_legacy"],
                    "bq_lb_production": num["bq_lb_production"],
                    "bq_scan_mult1": scan["windows"][0]["bq"],
                    "resid_d21": num["resid_d21"],
                    "resid_legacy": num["resid_legacy"],
                    **{k: num["decomposition"][k] for k in ("d_symbol", "d_amplitude", "d_wall", "d_total")},
                }
            ),
        )
