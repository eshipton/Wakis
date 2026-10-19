"""Figures for the B4 thin-wire dipole prism FIT study (reads data/B4_*.json and shared store)."""

import glob
import json
import os

os.environ.setdefault("CLOUDSDK_STORAGE_PROCESS_COUNT", "1")
os.environ.setdefault("CLOUDSDK_STORAGE_THREAD_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
A_WIRE = 0.5e-3


def load_local(arm="prism-graded"):
    recs = []
    for fn in sorted(glob.glob(os.path.join(DATA, f"B4_{arm}_*.json"))):
        with open(fn) as fh:
            recs.append(json.load(fh))
    recs.sort(key=lambda r: -r["resolution"])
    return recs


def _q_label(ht):
    q = A_WIRE / ht
    if np.isclose(q, round(q)):
        return f"a/{int(round(q))}"
    return f"a/{q:g}"


def plot_zin(prism_recs):
    from wakis.benchmarks import get_case

    case = get_case("B4")
    ref = case.reference
    f_ghz = case.freqs / 1e9
    z_ref = ref["z_in"]

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    axs[0].plot(
        f_ghz,
        z_ref.real,
        "k-",
        lw=2.0,
        label="Hallen MoM (thin wire, finite gap)",
    )
    axs[1].plot(
        f_ghz,
        z_ref.imag,
        "k-",
        lw=2.0,
        label="Hallen MoM (thin wire, finite gap)",
    )
    axs[1].axhline(0.0, color="0.5", ls=":", lw=0.9)

    markers = ["o", "^", "D", "s", "v"]
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for idx, r in enumerate(prism_recs):
        sp = r["metrics"]["spectra"]
        z_re = np.asarray(sp["z_re"])
        z_im = np.asarray(sp["z_im"])
        mk = markers[idx % len(markers)]
        col = colors[idx % len(colors)]
        lab = f"Wakis FIT (prisms, graded) h_t = {_q_label(r['resolution'])}"
        axs[0].plot(
            f_ghz[::2],
            z_re[::2],
            marker=mk,
            ls="none",
            color=col,
            mfc="none",
            ms=5,
            label=lab,
        )
        axs[1].plot(
            f_ghz[::2],
            z_im[::2],
            marker=mk,
            ls="none",
            color=col,
            mfc="none",
            ms=5,
            label=lab,
        )

    axs[0].set_xlabel("frequency [GHz]")
    axs[0].set_ylabel(r"$R_{\mathrm{in}}$ [$\Omega$]")
    axs[0].set_xlim(0.7, 1.2)
    axs[0].set_title(r"Input resistance $R_{\mathrm{in}}(f)$")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=7.5)

    axs[1].set_xlabel("frequency [GHz]")
    axs[1].set_ylabel(r"$X_{\mathrm{in}}$ [$\Omega$]")
    axs[1].set_xlim(0.7, 1.2)
    axs[1].set_title(
        f"Input reactance $X_{{\\mathrm{{in}}}}(f)$ (ref $f_{{\\mathrm{{res}}}}$ = {ref['f_res'] * 1e-6:.1f} MHz, $R_{{\\mathrm{{res}}}}$ = {ref['r_res']:.2f} $\\Omega$)"
    )
    axs[1].grid(True, alpha=0.3)

    fig.suptitle(
        "B4 centre-fed PEC dipole (L = 150 mm, a = 0.5 mm, g = 1 mm): input impedance vs frequency",
        fontsize=10.5,
    )
    fig.savefig(os.path.join(FIG, "a_b4_zin.png"), dpi=150)
    plt.close(fig)


def plot_convergence(prism_recs):
    from wakis.benchmarks.results import fetch

    df = fetch("B4")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    metrics = [
        ("f_res", r"$|f_{\mathrm{res}}\ \mathrm{error}|$ [%]"),
        ("r_res", r"$|R_{\mathrm{res}}\ \mathrm{error}|$ [%]"),
        ("z_nrms", r"$Z_{\mathrm{in}}$ NRMS error [%]"),
    ]

    p_recs = sorted(prism_recs, key=lambda r: r["cost"]["unknowns"])
    p_unk = np.array([r["cost"]["unknowns"] for r in p_recs], dtype=float)

    store_arms = [
        (
            "cart-staircase-edges-graded",
            "C3",
            "^",
            "-",
            "Wakis FIT (cart-staircase-edges-graded)",
        ),
        (
            "cart-conformal-pec-graded",
            "C2",
            "D",
            "--",
            "Wakis FIT (cart-conformal-pec-graded)",
        ),
    ]

    for ax, (mkey, ylabel) in zip(axs, metrics):
        p_val = 100.0 * np.array([r["metrics"][mkey] for r in p_recs], dtype=float)
        ax.loglog(
            p_unk,
            p_val,
            "o-",
            color="C0",
            label="Wakis FIT (prisms, graded)",
        )
        for idx, r in enumerate(p_recs):
            q_str = _q_label(r["resolution"])
            if mkey == "f_res" and idx == 0:
                offset = (-18, -8)
            elif mkey == "r_res" and q_str == "a/3":
                offset = (-20, 5)
            else:
                offset = (4, 4)
            ax.annotate(
                q_str,
                (r["cost"]["unknowns"], 100.0 * r["metrics"][mkey]),
                textcoords="offset points",
                xytext=offset,
                fontsize=7,
                color="C0",
            )

        for arm_name, col, mk, ls, lab in store_arms:
            sub = df[df["arm"] == arm_name].sort_values("c.unknowns")
            if sub.empty:
                continue
            x = sub["c.unknowns"].to_numpy(dtype=float)
            y = 100.0 * sub[f"m.{mkey}"].to_numpy(dtype=float)
            ax.loglog(
                x,
                y,
                ls=ls,
                marker=mk,
                color=col,
                mfc="none",
                label=lab,
            )

        ax.set_xlabel("field unknowns (prisms: E+B dofs; Cartesian: 3 E/cell)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    axs[0].legend(fontsize=7.5)
    fig.suptitle(
        "B4 dipole error vs field unknowns (note: Cartesian counts 3 E unknowns/cell; prisms count all E+B DOFs, ~2x conservative)",
        fontsize=10,
    )
    fig.savefig(os.path.join(FIG, "b_b4_convergence.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    prism_recs = load_local("prism-graded")
    plot_zin(prism_recs)
    plot_convergence(prism_recs)


if __name__ == "__main__":
    main()
