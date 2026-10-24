# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Example 008: Optical and RF Dispersive Materials Library
========================================================

Demonstrates the optical and RF dispersive materials library in
``wakis.material_library`` / ``wakis.dispersive.dispersive_lib`` and plots
the complex refractive index ``n - i k`` of representative metals,
dielectrics, semiconductors, and uniaxial crystals against their original
source formulas and CC0 ``refractiveindex.info`` tabulated data.

Generates two figures under ``docs/img/``:
- ``docs/img/008_material_library_metals.png``
- ``docs/img/008_material_library_dielectrics.png``
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.constants import c as c_light

from wakis.dispersive import dispersive_lib
from wakis.material_library import info, list_materials, uniaxial_crystal

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "tests" / "data" / "refractiveindex_info"
IMG_DIR = ROOT / "docs" / "img"


def nk_from_eps(eps):
    """Return optical constants (n, k >= 0) from Wakis exp(+iwt) eps."""
    nk = np.sqrt(np.conj(eps))
    return nk.real, nk.imag


def load_tabulated(filename, lmin_um, lmax_um):
    """Load (lam_um, n, k) from a refractiveindex.info YAML file."""
    yml = yaml.safe_load((DATA_DIR / filename).read_text())
    if len(yml["DATA"]) == 2 and "tabulated n" in yml["DATA"][0]["type"]:
        arr_n = np.array(
            [
                [float(x) for x in r.split()]
                for r in yml["DATA"][0]["data"].strip().splitlines()
            ]
        )
        arr_k = np.array(
            [
                [float(x) for x in r.split()]
                for r in yml["DATA"][1]["data"].strip().splitlines()
            ]
        )
        mn = (arr_n[:, 0] >= lmin_um - 1e-9) & (arr_n[:, 0] <= lmax_um + 1e-9)
        mk = (arr_k[:, 0] >= lmin_um - 1e-9) & (arr_k[:, 0] <= lmax_um + 1e-9)
        return arr_n[mn, 0], arr_n[mn, 1], arr_k[mk, 1]
    arr = np.array(
        [
            [float(x) for x in r.split()]
            for r in yml["DATA"][0]["data"].strip().splitlines()
        ]
    )
    m = (arr[:, 0] >= lmin_um - 1e-9) & (arr[:, 0] <= lmax_um + 1e-9)
    lam = arr[m, 0]
    n = arr[m, 1]
    k = arr[m, 2] if arr.shape[1] > 2 else np.zeros_like(n)
    return lam, n, k


def eval_formula_n(filename, lam_um):
    """Evaluate refractiveindex.info Sellmeier formula 1 or 2 at `lam_um`."""
    yml = yaml.safe_load((DATA_DIR / filename).read_text())
    entry = yml["DATA"][0]
    ftype = entry["type"].strip()
    c = [float(x) for x in entry["coefficients"].split()]
    lam2 = np.asarray(lam_um, dtype=float) ** 2
    n2 = 1.0 + c[0]
    if ftype == "formula 1":
        for b, l_k in zip(c[1::2], c[2::2]):
            n2 = n2 + b * lam2 / (lam2 - l_k**2)
    elif ftype == "formula 2":
        for b, l2_k in zip(c[1::2], c[2::2]):
            n2 = n2 + b * lam2 / (lam2 - l2_k)
    return np.sqrt(n2)


def plot_metals(out_path):
    """Plot n, k of Rakic (1998) and Johnson & Christy (1972) metals vs data."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150)

    # Top row: Rakic et al. (1998) Lorentz-Drude models (0.2 - 5.0 um)
    ax_n, ax_k = axes[0, 0], axes[0, 1]
    rakic_items = [
        ("gold", "Au_Rakic-LD.yml", "Au (Rakić 1998)", "tab:orange"),
        ("silver", "Ag_Rakic-LD.yml", "Ag (Rakić 1998)", "tab:blue"),
        ("copper", "Cu_Rakic-LD.yml", "Cu (Rakić 1998)", "tab:red"),
        ("aluminium", "Al_Rakic-LD.yml", "Al (Rakić 1998)", "tab:green"),
        ("titanium", "Ti_Rakic-LD.yml", "Ti (Rakić 1998)", "tab:purple"),
    ]
    lam_grid = np.geomspace(0.2, 5.0, 300)
    f_grid = c_light / (lam_grid * 1e-6)
    for key, yml_file, label, color in rakic_items:
        mat = dispersive_lib[key]
        n_m, k_m = nk_from_eps(mat.epsilon(f_grid))
        ax_n.plot(lam_grid, n_m, color=color, lw=1.8, label=label)
        ax_k.plot(lam_grid, k_m, color=color, lw=1.8, label=label)
        lam_d, n_d, k_d = load_tabulated(yml_file, 0.2, 5.0)
        step = max(1, len(lam_d) // 25)
        ax_n.plot(lam_d[::step], n_d[::step], "o", color=color, ms=3.5, alpha=0.75)
        ax_k.plot(lam_d[::step], k_d[::step], "o", color=color, ms=3.5, alpha=0.75)

    ax_n.set_xscale("log")
    ax_k.set_xscale("log")
    ax_n.set_xlabel("Wavelength [µm]")
    ax_k.set_xlabel("Wavelength [µm]")
    ax_n.set_ylabel("Refractive index $n$")
    ax_k.set_ylabel("Extinction coefficient $k$")
    ax_n.set_title("Rakić et al. (1998) Lorentz–Drude: $n(\\lambda)$")
    ax_k.set_title("Rakić et al. (1998) Lorentz–Drude: $k(\\lambda)$")
    ax_n.grid(True, which="both", ls=":", alpha=0.5)
    ax_k.grid(True, which="both", ls=":", alpha=0.5)
    ax_n.legend(fontsize=8.5)
    ax_k.legend(fontsize=8.5)

    # Bottom row: Johnson & Christy (1972) fitted models (0.4 - 0.8 um and wide)
    ax_jn, ax_jk = axes[1, 0], axes[1, 1]
    jc_items = [
        ("gold jc", "gold jc wide", "Au_Johnson.yml", "Au (J&C 1972)", "tab:orange"),
        ("silver jc", "silver jc wide", "Ag_Johnson.yml", "Ag (J&C 1972)", "tab:blue"),
        ("copper jc", "copper jc wide", "Cu_Johnson.yml", "Cu (J&C 1972)", "tab:red"),
    ]
    lam_vis = np.linspace(0.4, 0.8, 200)
    lam_wide = np.linspace(0.3, 1.94, 300)
    for key_vis, key_wide, yml_file, label, color in jc_items:
        n_v, k_v = nk_from_eps(dispersive_lib[key_vis].epsilon(c_light / (lam_vis * 1e-6)))
        n_w, k_w = nk_from_eps(
            dispersive_lib[key_wide].epsilon(c_light / (lam_wide * 1e-6))
        )
        ax_jn.plot(lam_vis, n_v, "-", color=color, lw=2.0, label=f"{label} fit (0.4–0.8 µm)")
        ax_jk.plot(lam_vis, k_v, "-", color=color, lw=2.0, label=f"{label} fit (0.4–0.8 µm)")
        ax_jn.plot(lam_wide, n_w, "--", color=color, lw=1.3, alpha=0.8, label=f"{label} wide")
        ax_jk.plot(lam_wide, k_w, "--", color=color, lw=1.3, alpha=0.8, label=f"{label} wide")
        lam_d, n_d, k_d = load_tabulated(yml_file, 0.3, 1.94)
        ax_jn.plot(lam_d, n_d, "s", color=color, ms=3.5, alpha=0.75)
        ax_jk.plot(lam_d, k_d, "s", color=color, ms=3.5, alpha=0.75)

    ax_jn.set_xlabel("Wavelength [µm]")
    ax_jk.set_xlabel("Wavelength [µm]")
    ax_jn.set_ylabel("Refractive index $n$")
    ax_jk.set_ylabel("Extinction coefficient $k$")
    ax_jn.set_title("Johnson & Christy (1972) Fits vs Tabulated Data: $n(\\lambda)$")
    ax_jk.set_title("Johnson & Christy (1972) Fits vs Tabulated Data: $k(\\lambda)$")
    ax_jn.grid(True, ls=":", alpha=0.5)
    ax_jk.grid(True, ls=":", alpha=0.5)
    ax_jn.legend(fontsize=7.5, ncol=2)
    ax_jk.legend(fontsize=7.5, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_dielectrics(out_path):
    """Plot n, k of dielectrics, semiconductors, and uniaxial crystals vs sources."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Left panel: Isotropic dielectrics, polymers, and uniaxial crystals (Sellmeier)
    ax_d = axes[0]
    sellmeier_items = [
        ("fused silica", "SiO2_Malitson.yml", (0.25, 3.5), "Fused silica (Malitson 1965)", "tab:blue", "-"),
        ("bk7", "BK7_SCHOTT.yml", (0.30, 2.5), "N-BK7 (SCHOTT)", "tab:cyan", "-"),
        ("si3n4", "Si3N4_Luke.yml", (0.31, 3.5), "$\\mathrm{Si_3N_4}$ (Luke 2015)", "tab:green", "-"),
        ("sapphire o", "Al2O3_Malitson-o.yml", (0.25, 3.5), "Sapphire $n_o$ (Malitson 1972)", "tab:purple", "-"),
        ("sapphire e", "Al2O3_Malitson-e.yml", (0.25, 3.5), "Sapphire $n_e$ (Malitson 1972)", "tab:purple", "--"),
        ("linbo3 o", "LiNbO3_Zelmon-o.yml", (0.40, 3.5), "$\\mathrm{LiNbO_3}$ $n_o$ (Zelmon 1997)", "tab:red", "-"),
        ("linbo3 e", "LiNbO3_Zelmon-e.yml", (0.40, 3.5), "$\\mathrm{LiNbO_3}$ $n_e$ (Zelmon 1997)", "tab:red", "--"),
    ]
    for key, yml_file, (lmin, lmax), label, color, ls in sellmeier_items:
        lam = np.linspace(lmin, lmax, 200)
        n_m, _ = nk_from_eps(dispersive_lib[key].epsilon(c_light / (lam * 1e-6)))
        ax_d.plot(lam, n_m, ls=ls, color=color, lw=1.8, label=label)
        lam_pts = np.linspace(lmin, lmax, 12)
        n_pts = eval_formula_n(yml_file, lam_pts)
        ax_d.plot(lam_pts, n_pts, "o", color=color, ms=3.2, alpha=0.75)

    ax_d.set_xlabel("Wavelength [µm]")
    ax_d.set_ylabel("Refractive index $n$")
    ax_d.set_title("Dielectrics & Uniaxial Crystals (Lines: Wakis, Dots: Source)")
    ax_d.grid(True, ls=":", alpha=0.5)
    ax_d.legend(fontsize=7.5, loc="upper right")

    # Right panel: Semiconductors and fitted ALD TiO2 (n and k)
    ax_s = axes[1]
    ax_sk = ax_s.twinx()

    # c-Si (0.4 - 1.0 um, Green & Keevers 1995)
    lam_csi = np.linspace(0.4, 1.0, 200)
    n_csi, k_csi = nk_from_eps(dispersive_lib["c-si"].epsilon(c_light / (lam_csi * 1e-6)))
    lam_d, n_d, k_d = load_tabulated("Si_Green-1995.yml", 0.4, 1.0)
    ax_s.plot(lam_csi, n_csi, "-", color="tab:orange", lw=2.0, label="c-Si $n$ (fit to Green 1995)")
    ax_s.plot(lam_d[::5], n_d[::5], "o", color="tab:orange", ms=3.5)
    ax_sk.plot(lam_csi, k_csi, "--", color="tab:orange", lw=1.5, label="c-Si $k$ (fit to Green 1995)")
    ax_sk.plot(lam_d[::5], k_d[::5], "s", color="tab:orange", ms=3.5, alpha=0.7)

    # TiO2 ALD (0.4 - 1.69 um, Sarkar 2019)
    lam_tio2 = np.linspace(0.4, 1.69, 200)
    n_tio2, _ = nk_from_eps(dispersive_lib["tio2 ald"].epsilon(c_light / (lam_tio2 * 1e-6)))
    lam_td, n_td, _ = load_tabulated("TiO2_Sarkar.yml", 0.4, 1.69)
    ax_s.plot(lam_tio2, n_tio2, "-", color="tab:blue", lw=2.0, label="$\\mathrm{TiO_2}$ ALD $n$ (fit to Sarkar 2019)")
    ax_s.plot(lam_td[::70], n_td[::70], "o", color="tab:blue", ms=3.5)

    # GaP (0.54 - 2.0 um, Bond 1965)
    lam_gap = np.linspace(0.54, 2.0, 200)
    n_gap, _ = nk_from_eps(dispersive_lib["gap"].epsilon(c_light / (lam_gap * 1e-6)))
    lam_gd, n_gd, _ = load_tabulated("GaP_Bond.yml", 0.54, 2.0)
    ax_s.plot(lam_gap, n_gap, "-", color="tab:green", lw=2.0, label="GaP $n$ (fit to Bond 1965)")
    ax_s.plot(lam_gd, n_gd, "o", color="tab:green", ms=3.5)

    # GaAs (0.97 - 2.0 um, Skauli 2003)
    lam_gaas = np.linspace(0.97, 2.0, 150)
    n_gaas, _ = nk_from_eps(dispersive_lib["gaas"].epsilon(c_light / (lam_gaas * 1e-6)))
    lam_gad = np.linspace(0.97, 2.0, 10)
    ax_s.plot(lam_gaas, n_gaas, "-", color="tab:brown", lw=2.0, label="GaAs $n$ (Skauli 2003)")
    ax_s.plot(lam_gad, eval_formula_n("GaAs_Skauli.yml", lam_gad), "o", color="tab:brown", ms=3.5)

    ax_s.set_xlabel("Wavelength [µm]")
    ax_s.set_ylabel("Refractive index $n$")
    ax_sk.set_ylabel("Extinction coefficient $k$ (dashed)")
    ax_s.set_title("Semiconductors & Fitted Dielectrics vs Source Data")
    ax_s.grid(True, ls=":", alpha=0.5)
    lines1, labels1 = ax_s.get_legend_handles_labels()
    lines2, labels2 = ax_sk.get_legend_handles_labels()
    ax_s.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    metals_fig = IMG_DIR / "008_material_library_metals.png"
    dielectrics_fig = IMG_DIR / "008_material_library_dielectrics.png"

    plot_metals(metals_fig)
    plot_dielectrics(dielectrics_fig)

    print(f"Registered canonical materials: {len(list_materials())}")
    print(f"Registered names including aliases: {len(list_materials(include_aliases=True))}")
    ox, oy, oz = uniaxial_crystal("sapphire", axis="z")
    print(f"Sapphire (c-axis || z): ({ox.name}, {oy.name}, {oz.name})")
    print(f"Saved figures:\n  {metals_fig}\n  {dielectrics_fig}")
