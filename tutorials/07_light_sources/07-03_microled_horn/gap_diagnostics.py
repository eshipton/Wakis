# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Reproducible gap diagnostics CLI for Tutorial 7.3: MicroLED with a
gradient-index micro-horn collimator (Luce, Alaee & Abass, Sci. Rep. 16, 7391, 2026).

This module reuses the geometry, graded mesh, dispersive Kottke subpixel smoothing,
and 2D FIT solver functions from `_helpers.py` and `microled_horn.py` (via import,
without copying solver or geometry code) to run controlled single-factor diagnostic
experiments investigating the discrepancies between our 2D FIT baseline and the
published 2D FDTD benchmarks of Luce et al. (2026).

Johnson & Christy (1972) Silver (Ag) 1-Pole Drude Fit Specification
-------------------------------------------------------------------
- **Source of optical constants (n, k)**:
  P. B. Johnson and R. W. Christy, "Optical Constants of the Noble Metals,"
  *Phys. Rev. B* **6**(12), 4370-4379 (1972), Table I (Ag, p. 4374) at photon
  energy ``E = 2.76 eV`` (vacuum wavelength ``lambda = 449.3 nm``, the tabulated
  point nearest the tutorial center wavelength ``lambda_0 = 450.0 nm``):
  ``n_JC = 0.04``, ``k_JC = 2.657`` (tabulated uncertainty ``Delta n = +-0.02``,
  ``Delta k = +-0.03``).
- **Fit point and complex relative permittivity (physics sign convention exp(+i w t))**:
  Calibrated at ``lambda_fit = 450.0 nm`` (``f_fit = c / lambda_fit = 6.66205e14 Hz``,
  ``omega_fit = 2 pi f_fit = 4.18589e15 rad/s``):
  ``eps_JC(450 nm) = (n_JC - i k_JC)^2 = (0.04 - 2.657i)^2 = -7.058049 - 0.212560i``
  (yielding normal-incidence GaN/Ag reflectance ``R_GaN/Ag = 97.04%`` at ``n_GaN = 2.45``,
  electric-field skin depth ``delta_E = 26.95 nm``, intensity skin depth
  ``delta_I = 13.48 nm``; vs. Rakic 1998 Ag at ``450 nm``: ``eps_Rakic = -5.4021 - 0.6341i``,
  ``n = 0.136 - 2.328i``, ``R_GaN/Ag = 88.24%``, ``delta_E = 30.76 nm``, ``delta_I = 15.38 nm``).
- **Calibrated 1-pole Drude parameters**:
  ``eps(omega) = eps_inf - omega_p^2 / (omega (omega - i gamma))`` with:
  - ``eps_inf = 4.0`` (dimensionless high-frequency core background permittivity)
  - ``omega_p = 1.392219e16 rad/s`` (plasma angular frequency, ``hbar omega_p = 9.1637 eV``)
  - ``gamma = 8.046205e13 rad/s`` (damping rate, ``hbar gamma = 0.05296 eV``)

Command-Line Interface
----------------------
Run a single diagnostic configuration and write one JSON summary to ``--out``::

    python tutorials/07_light_sources/07-03_microled_horn/gap_diagnostics.py \\
        --variant {baseline,jc_ag,pec_horn,ap_mon,no_sio2_pass} \\
        --config {bare,air,sio2,dgrin} \\
        --ndip M \\
        --mesh {quick,full} \\
        --out data/diag
"""

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light

TUTORIAL_DIR = Path(__file__).resolve().parent
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (
    H_HORN_DEFAULT,
    N_GAN,
    N_SIO2,
    PAPER_2D_REFS,
    W_GAN,
    W_PIXEL,
    WL_0,
    Y_QW_DEFAULT,
    ag_skin_depth,
    evaluate_horn_configuration,
    make_ag_drude,
)
from wakis.dispersive import DispersiveMaterial, Drude

DEFAULT_DIAG_DIR = TUTORIAL_DIR / "data" / "diag"
DEFAULT_FIG_PATH = TUTORIAL_DIR / "figures" / "15_gap_diagnostics.png"

VARIANTS = ("baseline", "jc_ag", "pec_horn", "ap_mon", "no_sio2_pass")
CONFIGS = ("bare", "air", "sio2", "dgrin")
MESH_MODES = ("quick", "full")

# Johnson & Christy (1972), Phys. Rev. B 6, 4370, Table I (Ag at E = 2.76 eV, lambda = 449.3 nm)
JC_AG_N_450NM = 0.04
JC_AG_K_450NM = 2.657
JC_AG_EPS_INF = 4.0
JC_AG_WL_FIT = WL_0

CONFIG_GEOMETRY = {
    "bare": {"h_horn": 0.0, "theta_deg": 90.0, "x_half_min": 1.80e-6},
    "air": {"h_horn": H_HORN_DEFAULT, "theta_deg": 72.0, "x_half_min": 2.10e-6},
    "sio2": {"h_horn": H_HORN_DEFAULT, "theta_deg": 64.0, "x_half_min": 2.40e-6},
    "dgrin": {"h_horn": H_HORN_DEFAULT, "theta_deg": 60.0, "x_half_min": 2.55e-6},
}

MESH_SPEC = {
    "quick": {"dx": 15.0e-9, "dx_skin": 3.75e-9},
    "full": {"dx": 7.5e-9, "dx_skin": 3.00e-9},
}

# Active QW dipole ensemble half-span [m]: [-240, +240] nm across the 600 nm GaN core
QW_DIPOLE_SPAN_HALF = 0.24e-6


def make_ag_drude_jc(
    wl_fit=JC_AG_WL_FIT,
    eps_inf=JC_AG_EPS_INF,
    n_jc=JC_AG_N_450NM,
    k_jc=JC_AG_K_450NM,
):
    """
    Construct a 1-pole Drude dispersive silver model calibrated to Johnson & Christy (1972) Ag.

    Source and fit point:
    - P. B. Johnson and R. W. Christy, *Phys. Rev. B* **6**(12), 4370-4379 (1972),
      Table I (Ag, p. 4374) at ``E = 2.76 eV`` (``lambda = 449.3 nm``):
      ``n = 0.04``, ``k = 2.657``.
    - Fit point: ``wl_fit = 450.0 nm`` (``f_fit = 6.66205e14 Hz``,
      ``omega_fit = 4.18589e15 rad/s``), where
      ``eps_JC = (n - i k)^2 = -7.058049 - 0.212560i``.
    - With ``eps_inf = 4.0``, exact inversion of
      ``eps(omega) = eps_inf - omega_p^2 / (omega (omega - i gamma))`` gives
      ``omega_p = 1.392219e16 rad/s`` and ``gamma = 8.046205e13 rad/s``.

    Parameters
    ----------
    wl_fit : float, optional
        Calibration vacuum wavelength [m] (default: ``450e-9``).
    eps_inf : float, optional
        High-frequency background relative permittivity (default: ``4.0``).
    n_jc : float, optional
        Real part of the silver refractive index from Johnson & Christy (1972)
        (default: ``0.04``).
    k_jc : float, optional
        Extinction coefficient of silver from Johnson & Christy (1972)
        (default: ``2.657``).

    Returns
    -------
    DispersiveMaterial
        Calibrated 1-pole Drude material ``Ag_JC1972_Drude_450nm``.
    """
    f_fit = c_light / float(wl_fit)
    w_fit = 2.0 * np.pi * f_fit
    eps_jc = (float(n_jc) - 1j * float(k_jc)) ** 2
    denom = float(eps_inf) - eps_jc
    inv_d = 1.0 / denom
    gamma = -w_fit * inv_d.imag / inv_d.real
    omega_p = np.sqrt(w_fit**2 / inv_d.real)
    return DispersiveMaterial(
        eps_inf=float(eps_inf),
        poles=[Drude(omega_p=omega_p, gamma=gamma)],
        name="Ag_JC1972_Drude_450nm",
    )


def uniform_qw_dipoles(ndip, x_span_half=QW_DIPOLE_SPAN_HALF):
    """
    Generate ``M = ndip`` uniformly spaced dipole x-positions across the active QW width.

    For ``M = 1``, a single central dipole is placed at ``x_d = 0``. For ``M > 1``,
    ``M`` dipoles are uniformly spaced across ``[-x_span_half, +x_span_half]``
    (``[-240, +240] nm`` across the ``w_GaN = 600 nm`` active region). Because the
    device and mesh are symmetric under ``x <-> -x`` and `evaluate_horn_configuration`
    symmetrizes off-axis far-field intensities ``0.5 * (U(theta) + U(-theta))``,
    symmetric pairs ``(-x_d, +x_d)`` fold onto non-negative coordinates ``x_d >= 0``
    with multiplicity weight ``2.0`` (and weight ``1.0`` at ``x_d = 0``).

    Parameters
    ----------
    ndip : int
        Total number of dipoles ``M >= 1`` across the full active QW width.
    x_span_half : float, optional
        Half-span [m] of the uniform QW dipole array (default: ``0.24e-6``).

    Returns
    -------
    x_full : ndarray of float
        All ``M`` dipole x-positions [m] across ``[-x_span_half, +x_span_half]``.
    folded_xs : tuple of float
        Non-negative dipole x-positions [m] passed to `evaluate_horn_configuration`.
    folded_weights : tuple of float
        Multiplicity weights corresponding to ``folded_xs``.
    """
    m = int(ndip)
    if m < 1:
        raise ValueError(f"ndip M must be >= 1, got {m}.")
    if m == 1:
        x_full = np.array([0.0], dtype=float)
        return x_full, (0.0,), (1.0,)

    x_full = np.linspace(-float(x_span_half), float(x_span_half), m)
    if m % 2 == 1:
        half_n = (m + 1) // 2
        folded_xs = tuple(float(np.round(x, 15)) for x in np.linspace(0.0, float(x_span_half), half_n))
        folded_weights = (1.0,) + (2.0,) * (half_n - 1)
    else:
        pos_xs = x_full[m // 2 :]
        folded_xs = tuple(float(np.round(x, 15)) for x in pos_xs)
        folded_weights = (2.0,) * len(folded_xs)
    return x_full, folded_xs, folded_weights


def build_variant_kwargs(variant, config):
    """
    Return the keyword overrides for `evaluate_horn_configuration` for a diagnostic variant.

    Parameters
    ----------
    variant : {'baseline', 'jc_ag', 'pec_horn', 'ap_mon', 'no_sio2_pass'}
        Diagnostic variant name.
    config : {'bare', 'air', 'sio2', 'dgrin'}
        Micro-horn configuration name.

    Returns
    -------
    dict
        Keyword overrides for `evaluate_horn_configuration` and metadata fields.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    if config not in CONFIGS:
        raise ValueError(f"Unknown config {config!r}; expected one of {CONFIGS}.")

    geom = CONFIG_GEOMETRY[config]
    h_horn = geom["h_horn"]
    theta_deg = geom["theta_deg"]
    is_bare = config == "bare" or h_horn <= 0.0
    w_top = (
        W_PIXEL
        if is_bare
        else W_PIXEL + 2.0 * float(h_horn) / np.tan(np.deg2rad(float(theta_deg)))
    )

    kwargs = {
        "ag_material": None,
        "horn_ag_material": None,
        "passivation_material": None,
        "y_mon_offset": 0.16e-6,
        "w_mon_override": None,
    }
    if variant == "jc_ag":
        kwargs["ag_material"] = make_ag_drude_jc()
    elif variant == "pec_horn":
        kwargs["horn_ag_material"] = "pec"
    elif variant == "ap_mon":
        kwargs["y_mon_offset"] = 0.03e-6
        kwargs["w_mon_override"] = float(w_top)
    elif variant == "no_sio2_pass":
        kwargs["passivation_material"] = float(N_GAN**2)
    return kwargs


def diag_json_filename(variant, config, ndip, mesh):
    """Return the canonical JSON filename ``{variant}_{config}_M{M}_{mesh}.json``."""
    return f"{variant}_{config}_M{int(ndip)}_{mesh}.json"


def run_diagnostic(variant, config, ndip, mesh="quick", out_dir=DEFAULT_DIAG_DIR):
    """
    Execute one gap-diagnostic simulation and write its JSON result to ``out_dir``.

    Parameters
    ----------
    variant : {'baseline', 'jc_ag', 'pec_horn', 'ap_mon', 'no_sio2_pass'}
        Model variant to evaluate.
    config : {'bare', 'air', 'sio2', 'dgrin'}
        Device configuration.
    ndip : int
        Total number of uniformly spaced dipoles ``M`` across the QW width.
    mesh : {'quick', 'full'}, optional
        Mesh resolution preset (default: ``'quick'``).
    out_dir : Path or str, optional
        Output directory for the JSON file (default: ``data/diag``).

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    if mesh not in MESH_MODES:
        raise ValueError(f"Unknown mesh mode {mesh!r}; expected one of {MESH_MODES}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = CONFIG_GEOMETRY[config]
    m_spec = MESH_SPEC[mesh]
    x_full, folded_xs, folded_ws = uniform_qw_dipoles(ndip)
    v_kwargs = build_variant_kwargs(variant, config)

    t0 = time.perf_counter()
    res = evaluate_horn_configuration(
        config=config,
        h_horn=geom["h_horn"],
        theta_deg=geom["theta_deg"],
        y_qw=Y_QW_DEFAULT,
        dx=m_spec["dx"],
        dx_skin=m_spec["dx_skin"],
        x_half_min=geom["x_half_min"],
        dipole_xs=folded_xs,
        dipole_weights=folded_ws,
        collect_fields=False,
        record_movie=False,
        **v_kwargs,
    )
    wall_s = float(time.perf_counter() - t0)

    ag_mat_used = v_kwargs["ag_material"] if v_kwargs["ag_material"] is not None else make_ag_drude()
    f0 = c_light / WL_0
    eps_ag_450 = complex(ag_mat_used.epsilon(f0))
    delta_e_ag, delta_i_ag = ag_skin_depth(WL_0, ag_mat_used)

    paper_15 = float(PAPER_2D_REFS[config]["lee15_pct"])
    paper_90 = float(PAPER_2D_REFS[config]["lee90_pct"])
    lee15_eq = float(res["lee15_eq_pct"])
    lee90_eq = float(res["lee90_eq_pct"])
    lee15_x = float(res["lee15_pol_x"])
    lee15_y = float(res["lee15_pol_y"])
    lee15_z = float(res["lee15_pol_z"])
    lee90_x = float(res["lee90_pol_x"])
    lee90_y = float(res["lee90_pol_y"])
    lee90_z = float(res["lee90_pol_z"])

    record = {
        "variant": str(variant),
        "config": str(config),
        "mesh": str(mesh),
        "M": int(ndip),
        "ndip": int(ndip),
        "dipole_x_m": [round(float(x), 10) for x in x_full],
        "dipole_x_nm": [round(float(x * 1e9), 3) for x in x_full],
        "folded_dipole_xs_m": [round(float(x), 10) for x in folded_xs],
        "folded_dipole_weights": [float(w) for w in folded_ws],
        "lee15_eq_pct": round(lee15_eq, 4),
        "lee90_eq_pct": round(lee90_eq, 4),
        "lee15_pct": round(float(res["lee15_pct"]), 4),
        "lee90_pct": round(float(res["lee90_pct"]), 4),
        "lee15_wrk_pct": round(float(res["lee15_wrk_pct"]), 4),
        "lee90_wrk_pct": round(float(res["lee90_wrk_pct"]), 4),
        "lee15_pol_x": round(lee15_x, 4),
        "lee15_pol_y": round(lee15_y, 4),
        "lee15_pol_z": round(lee15_z, 4),
        "lee90_pol_x": round(lee90_x, 4),
        "lee90_pol_y": round(lee90_y, 4),
        "lee90_pol_z": round(lee90_z, 4),
        "per_pol": {
            "lee15_pct": {
                "x": round(lee15_x, 4),
                "y": round(lee15_y, 4),
                "z": round(lee15_z, 4),
            },
            "lee90_pct": {
                "x": round(lee90_x, 4),
                "y": round(lee90_y, 4),
                "z": round(lee90_z, 4),
            },
        },
        "lee15_xz_tangential_pct": round(0.5 * (lee15_x + lee15_z), 4),
        "lee90_xz_tangential_pct": round(0.5 * (lee90_x + lee90_z), 4),
        "collimation_ratio_eq_pct": round(100.0 * lee15_eq / max(lee90_eq, 1e-12), 4),
        "collimation_ratio_pct": round(float(res["collimation_ratio_pct"]), 4),
        "purcell_factor": round(float(res["purcell_factor"]), 4),
        "fwhm_deg": round(float(res["fwhm_deg"]), 4),
        "directivity_2d": round(float(res["directivity_2d"]), 4),
        "paper_2d_lee15_pct": paper_15,
        "paper_2d_lee90_pct": paper_90,
        "gap_lee15_eq_pp": round(lee15_eq - paper_15, 4),
        "gap_lee90_eq_pp": round(lee90_eq - paper_90, 4),
        "ag_eps_450nm_real": round(float(eps_ag_450.real), 4),
        "ag_eps_450nm_imag": round(float(eps_ag_450.imag), 4),
        "ag_skin_depth_E_nm": round(float(delta_e_ag * 1e9), 4),
        "ag_skin_depth_I_nm": round(float(delta_i_ag * 1e9), 4),
        "dx_background_nm": round(float(res["dx_background_nm"]), 4),
        "dx_min_nm": round(float(res["dx_min_nm"]), 4),
        "cpw_gan": round(float(res["cpw_gan"]), 4),
        "cpw_gan_min_wl": round(float(res["cpw_gan_min_wl"]), 4),
        "min_feature_nm": round(float(res["min_feature_nm"]), 4),
        "cells_across_feature": round(float(res["cells_across_feature"]), 4),
        "cells_per_skin_depth_I": round(float(res["cells_per_skin_depth_I"]), 4),
        "cells_per_skin_depth_E": round(float(res["cells_per_skin_depth_E"]), 4),
        "n_cells": int(res["n_cells"]),
        "n_steps": int(res["n_steps"]),
        "conv_ndip_full": [int(v) for v in res["conv_ndip_full"]],
        "conv_lee15_eq": [round(float(v), 4) for v in res["conv_lee15_eq"]],
        "conv_lee90_eq": [round(float(v), 4) for v in res["conv_lee90_eq"]],
        "conv_lee15": [round(float(v), 4) for v in res["conv_lee15"]],
        "conv_lee90": [round(float(v), 4) for v in res["conv_lee90"]],
        "wall_time_s": round(wall_s, 3),
        "elapsed_s": round(float(res["elapsed_s"]), 3),
    }

    out_path = out_dir / diag_json_filename(variant, config, ndip, mesh)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"[diag] {out_path.name}: LEE15_eq={record['lee15_eq_pct']:.2f}%, "
        f"LEE90_eq={record['lee90_eq_pct']:.2f}%, "
        f"pol15=({lee15_x:.2f}, {lee15_y:.2f}, {lee15_z:.2f})%, "
        f"M={int(ndip)}, mesh={mesh}, wall={wall_s:.1f}s",
        flush=True,
    )
    return out_path


def plot_gap_diagnostics(diag_dir=DEFAULT_DIAG_DIR, fig_path=DEFAULT_FIG_PATH):
    """
    Generate ``figures/15_gap_diagnostics.png`` from the committed JSON files in ``data/diag/``.

    Panel (a): ``LEE15_eq`` vs dipole count ``M in {1, 5, 9, 17}`` per configuration at
    ``--quick`` (with ``+-0.5 pp`` convergence bands around ``M = 17`` and ``--full`` markers
    at ``M = 9`` and ``M = 17``).
    Panel (b): Converged ``--full`` ``baseline`` (Rakic 1998 Ag) vs ``jc_ag`` (Johnson & Christy
    1972 Ag) vs Luce et al. (2026) published 2D FDTD ``LEE15`` benchmark bars (``M = 17`` for
    Air, SiO2, dGRIN; ``M = 9`` for Bare).
    """
    from wakis import figstyle

    figstyle.apply_style()
    diag_dir = Path(diag_dir)
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "bare": figstyle.PALETTE[7],
        "air": figstyle.PALETTE[0],
        "sio2": figstyle.PALETTE[1],
        "dgrin": figstyle.PALETTE[2],
    }
    labels = {
        "bare": r"Bare ($h=0$)",
        "air": r"Air $\mu$Horn ($72^\circ$)",
        "sio2": r"SiO$_2$ $\mu$Horn ($64^\circ$)",
        "dgrin": r"6-layer dGRIN ($60^\circ$)",
    }
    markers = {"bare": "s", "air": "^", "sio2": "D", "dgrin": "o"}
    m_vals = [1, 5, 9, 17]

    fig, (ax0, ax1) = figstyle.new_figure(nrows=1, ncols=2, width=12.8, height=4.4)

    # Panel (a): LEE15_eq vs M per config (quick mesh + full markers)
    full_star_labeled = False
    for cfg in CONFIGS:
        xs_m, ys_15 = [], []
        for m in m_vals:
            p = diag_dir / diag_json_filename("baseline", cfg, m, "quick")
            if p.exists():
                d = json.loads(p.read_text())
                xs_m.append(int(d["M"]))
                ys_15.append(float(d["lee15_eq_pct"]))
        if xs_m:
            ax0.plot(
                xs_m,
                ys_15,
                marker=markers[cfg],
                color=colors[cfg],
                lw=2.0,
                ms=6.0,
                label=f"{labels[cfg]} (--quick)",
            )
            y_ref = ys_15[-1]
            ax0.axhspan(
                y_ref - 0.5,
                y_ref + 0.5,
                color=colors[cfg],
                alpha=0.10,
            )
            ax0.axhline(y_ref, color=colors[cfg], ls=":", lw=0.9, alpha=0.7)

        # Overlay --full baseline markers at M=9 and M=17 where available
        for m_full in (9, 17):
            pf = diag_dir / diag_json_filename("baseline", cfg, m_full, "full")
            if pf.exists():
                df = json.loads(pf.read_text())
                ax0.scatter(
                    [int(df["M"])],
                    [float(df["lee15_eq_pct"])],
                    marker="*",
                    s=135,
                    color=colors[cfg],
                    edgecolor="black",
                    zorder=7,
                    label=r"--full Rakić Ag ($M=9, 17$)"
                    if not full_star_labeled
                    else None,
                )
                full_star_labeled = True

    ax0.axvline(
        9,
        color="#059669",
        ls="-.",
        lw=1.3,
        alpha=0.8,
        label=r"$M = 9$ ($|M_9 - M_{17}| < 0.5$ pp for Bare/Air/SiO$_2$)",
    )
    ax0.set_xticks(m_vals)
    ax0.set(
        xlabel=r"Number of uniformly spaced QW dipoles $M$ across $[-240, +240]$ nm",
        ylabel=r"Equal-pol forward extraction $\mathrm{LEE}_{15,\mathrm{eq}}$ [%]",
        ylim=(5.5, 43.0),
    )
    ax0.set_title(
        r"(a) Dipole-ensemble convergence $\mathrm{LEE}_{15,\mathrm{eq}}(M)$ (bands: $\pm 0.5$ pp around $M=17$)",
        fontsize=9.5,
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=7.1, framealpha=0.92)

    # Panel (b): baseline vs jc_ag vs paper bars at converged M (prefer M=17, fallback M=9)
    x_idx = np.arange(len(CONFIGS))
    base_15, jc_15, pub_15, m_used_per_cfg = [], [], [], []
    mesh_used_label = "--full"

    for cfg in CONFIGS:
        pub_15.append(float(PAPER_2D_REFS[cfg]["lee15_pct"]))
        b_val, j_val, m_picked = np.nan, np.nan, 9
        for mesh_try in ("full", "quick"):
            for m_try in (17, 9, 11, 5):
                pb = diag_dir / diag_json_filename("baseline", cfg, m_try, mesh_try)
                pj = diag_dir / diag_json_filename("jc_ag", cfg, m_try, mesh_try)
                if pb.exists() and pj.exists():
                    db = json.loads(pb.read_text())
                    dj = json.loads(pj.read_text())
                    b_val = float(db["lee15_eq_pct"])
                    j_val = float(dj["lee15_eq_pct"])
                    mesh_used_label = f"--{mesh_try}"
                    m_picked = int(m_try)
                    break
            if np.isfinite(b_val):
                break
        base_15.append(b_val)
        jc_15.append(j_val)
        m_used_per_cfg.append(m_picked)

    w_b = 0.26
    bars_b = ax1.bar(
        x_idx - w_b,
        base_15,
        width=w_b,
        color="#0284c7",
        label=f"Rakić (1998) Ag ({mesh_used_label}, converged $M=17$; Bare $M=9$)",
    )
    bars_j = ax1.bar(
        x_idx,
        jc_15,
        width=w_b,
        color="#059669",
        label=f"J&C (1972) Ag ({mesh_used_label}, converged $M=17$; Bare $M=9$)",
    )
    bars_p = ax1.bar(
        x_idx + w_b,
        pub_15,
        width=w_b,
        color="#dc2626",
        alpha=0.88,
        label="Luce et al. (2026) 2D FDTD benchmark",
    )

    for bars, fmt in ((bars_b, "{:.2f}%"), (bars_j, "{:.2f}%"), (bars_p, "{:.1f}%")):
        for b in bars:
            h_val = b.get_height()
            if np.isfinite(h_val):
                ax1.annotate(
                    fmt.format(h_val),
                    (b.get_x() + 0.5 * b.get_width(), h_val),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    fontsize=6.6,
                )

    xtick_names = [
        f"Bare ($h=0$)\n($M={m_used_per_cfg[0]}$)",
        f"Air ($72^\\circ$)\n($M={m_used_per_cfg[1]}$)",
        f"SiO$_2$ ($64^\\circ$)\n($M={m_used_per_cfg[2]}$)",
        f"6-layer dGRIN ($60^\\circ$)\n($M={m_used_per_cfg[3]}$)",
    ]
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(xtick_names)
    ax1.set(
        ylabel=r"Forward cone extraction $\mathrm{LEE}_{15,\mathrm{eq}}$ [%]",
        ylim=(
            0,
            max(max(jc_15), max(pub_15)) * 1.18
            if np.all(np.isfinite(jc_15))
            else 52,
        ),
    )
    ax1.set_title(
        f"(b) Silver optical constants ({mesh_used_label}, converged $M$): Rakić vs J&C (1972) vs Paper",
        fontsize=9.5,
    )
    ax1.legend(loc="upper left", fontsize=7.3)

    figstyle.finish(fig)
    figstyle.save(fig, fig_path, close=True)
    print(f"[diag] Rendered {fig_path}", flush=True)
    return fig_path


def print_gap_summary(diag_dir=DEFAULT_DIAG_DIR):
    """Print all committed `data/diag/*.json` numbers used in Section 6 and Section 8.2."""
    diag_dir = Path(diag_dir)
    print("=== 1. QUICK MESH M-CONVERGENCE (baseline Rakic Ag) ===")
    for cfg in CONFIGS:
        parts = []
        for m in (1, 5, 9, 17):
            p = diag_dir / diag_json_filename("baseline", cfg, m, "quick")
            if p.exists():
                d = json.loads(p.read_text())
                parts.append(
                    f"M={m:2d}: LEE15_eq={d['lee15_eq_pct']:5.2f}% (LEE90_eq={d['lee90_eq_pct']:5.2f}%)"
                )
        print(f"  {cfg:5s}: " + " | ".join(parts))

    print("\n=== 2. QUICK MESH J&C Ag (M=5, M=9) ===")
    for cfg in CONFIGS:
        for m in (5, 9):
            p = diag_dir / diag_json_filename("jc_ag", cfg, m, "quick")
            if p.exists():
                d = json.loads(p.read_text())
                print(
                    f"  {cfg:5s} M={m:2d}: LEE15_eq={d['lee15_eq_pct']:5.2f}% (gap={d['gap_lee15_eq_pp']:+5.2f} pp) | "
                    f"LEE90_eq={d['lee90_eq_pct']:5.2f}% (gap={d['gap_lee90_eq_pp']:+5.2f} pp)"
                )

    print("\n=== 3. FULL MESH M=9 AND M=17 (baseline Rakic Ag & jc_ag J&C Ag) ===")
    for variant in ("baseline", "jc_ag"):
        print(f"Variant: {variant}")
        for cfg in CONFIGS:
            for m in (9, 17):
                p = diag_dir / diag_json_filename(variant, cfg, m, "full")
                if p.exists():
                    d = json.loads(p.read_text())
                    print(
                        f"  {cfg:5s} M={m:2d}: LEE15_eq={d['lee15_eq_pct']:6.2f}% (gap={d['gap_lee15_eq_pp']:+6.2f} pp) | "
                        f"LEE90_eq={d['lee90_eq_pct']:6.2f}% (gap={d['gap_lee90_eq_pp']:+6.2f} pp) | "
                        f"eta15_eq={d['collimation_ratio_eq_pct']:5.2f}% | "
                        f"pol15=({d['lee15_pol_x']:5.2f},{d['lee15_pol_y']:5.2f},{d['lee15_pol_z']:5.2f}) | "
                        f"xz15={d['lee15_xz_tangential_pct']:5.2f}% | "
                        f"hom15={d['lee15_pct']:5.2f}% | wrk15={d['lee15_wrk_pct']:5.2f}% | "
                        f"Fp={d['purcell_factor']:5.3f} | FWHM={d['fwhm_deg']:5.1f} | D2D={d['directivity_2d']:4.2f}"
                    )


def _resolve_out_dir(out_arg):
    """Resolve ``--out data/diag`` to the tutorial directory when run from repo root."""
    p = Path(out_arg)
    if p == Path("data/diag") and not Path("data").exists() and (TUTORIAL_DIR / "data").exists():
        return TUTORIAL_DIR / "data" / "diag"
    return p


def parse_args(argv=None):
    """Parse CLI arguments for `gap_diagnostics.py`."""
    ap = argparse.ArgumentParser(
        description="Reproducible Tutorial 7.3 gap diagnostics CLI (variant, config, dipole count M, mesh)."
    )
    ap.add_argument(
        "--variant",
        choices=VARIANTS,
        default="baseline",
        help="Model variant to run (default: baseline).",
    )
    ap.add_argument(
        "--config",
        choices=CONFIGS,
        default="dgrin",
        help="Micro-horn configuration to evaluate (default: dgrin).",
    )
    ap.add_argument(
        "--ndip",
        type=int,
        default=9,
        help="Total number of uniformly spaced dipoles M across the QW width (default: 9).",
    )
    ap.add_argument(
        "--mesh",
        choices=MESH_MODES,
        default="quick",
        help="Mesh resolution preset: quick (dx=15nm, dx_skin=3.75nm) or full (dx=7.5nm, dx_skin=3.0nm).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/diag"),
        help="Output directory for diagnostic JSON files (default: data/diag).",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Render figures/15_gap_diagnostics.png from existing JSON files in --out.",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Print formatted summary of all committed diagnostic JSON files in --out.",
    )
    return ap.parse_args(argv)


def main(argv=None):
    """CLI entry point for `gap_diagnostics.py`."""
    args = parse_args(argv)
    out_dir = _resolve_out_dir(args.out)
    if args.summary:
        print_gap_summary(diag_dir=out_dir)
        return
    if args.plot:
        plot_gap_diagnostics(diag_dir=out_dir, fig_path=DEFAULT_FIG_PATH)
        return
    run_diagnostic(
        variant=args.variant,
        config=args.config,
        ndip=args.ndip,
        mesh=args.mesh,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()

