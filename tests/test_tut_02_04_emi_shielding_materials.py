# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Physics, datasheet pole-fitting, transfer-matrix, Schelkunoff decomposition,
Rozanov bound, resolution-rule, and mutation tests for Tutorial 2.4:
`tutorials/02_materials/02-04_emi_shielding_materials/`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wakis.dispersive import DispersiveMaterial, dispersive_lib
from wakis.material_library import (
    ECCOSORB_RF_LB_SOURCE_DATA,
    info as material_info,
    list_materials,
)

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "02_materials"
    / "02-04_emi_shielding_materials"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

hlp = _load_module("_tut_02_04_helpers", TUTORIAL_DIR / "_helpers.py")
tut = _load_module("_tut_02_04_emi_shielding", TUTORIAL_DIR / "emi_shielding_materials.py")


def _assert_emi_resolution_rules(
    cpw: float,
    cells_across: float,
    cells_per_skin: float = np.inf,
    min_cpw: float = 20.0,
) -> None:
    """Enforce project grid resolution rules inside an EMI slab."""
    assert float(cpw) >= float(min_cpw) - 1e-6, (
        f"Rule 1 violated: cpw={cpw:.2f} < {min_cpw}"
    )
    assert float(cells_across) >= 4.0 - 1e-6, (
        f"Rule 2 violated: cells_across={cells_across:.2f} < 4.0"
    )
    if np.isfinite(cells_per_skin):
        assert float(cells_per_skin) >= 4.0 - 1e-6, (
            f"Skin-depth rule violated: cells_per_skin_depth={cells_per_skin:.2f} < 4.0"
        )


def test_datasheet_csvs_and_emi_material_library_models():
    """Verify digitized Laird RF-LB CSVs, passivity, stability, and fit accuracy of all 6 EMI materials."""
    # 1. Digitized CSV files and required header comments
    perm_csv = TUTORIAL_DIR / "data" / "eccosorb_rf_lb_permeability.csv"
    att_csv = TUTORIAL_DIR / "data" / "eccosorb_rf_lb_attenuation.csv"
    assert perm_csv.is_file(), f"Missing {perm_csv}"
    assert att_csv.is_file(), f"Missing {att_csv}"

    perm_text = perm_csv.read_text(encoding="utf-8")
    att_text = att_csv.read_text(encoding="utf-8")
    for txt in (perm_text, att_text):
        assert "read from the datasheet plot, approximate" in txt
        assert "EM-ABS-DS-10132022" in txt

    ds = hlp.load_eccosorb_rf_lb_csv(TUTORIAL_DIR / "data")
    np.testing.assert_allclose(ds["f_mu_hz"], ECCOSORB_RF_LB_SOURCE_DATA["f_mu_hz"])
    np.testing.assert_allclose(ds["mu_real"], ECCOSORB_RF_LB_SOURCE_DATA["mu_real"])
    np.testing.assert_allclose(ds["mu_imag"], ECCOSORB_RF_LB_SOURCE_DATA["mu_imag"])
    np.testing.assert_allclose(ds["f_att_hz"], ECCOSORB_RF_LB_SOURCE_DATA["f_att_hz"])
    np.testing.assert_allclose(ds["atten_db_cm"], ECCOSORB_RF_LB_SOURCE_DATA["atten_db_cm"])

    # 2. Check all 6 EMI material classes in wakis.material_library
    emi_list = list_materials(category="emi")
    expected_keys = [c["key"] for c in hlp.EMI_CLASSES]
    for key in expected_keys:
        assert key in emi_list, f"Missing {key!r} in list_materials(category='emi')"
        meta = material_info(key)
        assert meta["category"] == "emi"
        assert "citation" in meta and "title" in meta["citation"]

    # 3. Passivity (Im eps <= 0, Im mu <= 0) and discrete ADE stability across all 6 classes
    f_check = np.geomspace(1.0e6, 20.0e9, 200)
    for key in expected_keys:
        mat = dispersive_lib[key]
        eps_v = mat.epsilon(f_check)
        mu_v = mat.mu(f_check)
        assert np.all(np.imag(eps_v) <= 1e-12), f"Non-passive epsilon in {key}"
        assert np.all(np.imag(mu_v) <= 1e-12), f"Non-passive mu in {key}"
        assert mat.is_stable(dt=0.25e-12, courant=0.8), f"Unstable ADE in {key}"

    # 4. Fitted mu(f) of Eccosorb RF-LB reproduces digitized datasheet points within digitization uncertainty
    rf_lb = dispersive_lib["eccosorb rf-lb"]
    mu_fit = rf_lb.mu(ds["f_mu_hz"])
    mu_target = ds["mu_real"] - 1j * ds["mu_imag"]
    rel_rms = float(np.sqrt(np.mean(np.abs(mu_fit - mu_target) ** 2 / np.abs(mu_target) ** 2)))
    max_d_real = float(np.max(np.abs(np.real(mu_fit) - ds["mu_real"])))
    max_d_imag = float(np.max(np.abs(-np.imag(mu_fit) - ds["mu_imag"])))
    assert rel_rms < 0.01, f"RF-LB mu fit relative RMS {rel_rms:.4f} exceeds 1%"
    assert max_d_real < 1.0, f"RF-LB max |Delta mu'| = {max_d_real:.3f} exceeds digitization tolerance"
    assert max_d_imag < 1.0, f"RF-LB max |Delta mu''| = {max_d_imag:.3f} exceeds digitization tolerance"

    # 5. Inferred eps' over the 0.1 - 1.0 GHz overlap band
    mask_ov = ds["f_att_hz"] <= 1.001e9
    inf0 = hlp.infer_eps_real_from_attenuation(
        ds["f_att_hz"][mask_ov], rf_lb.mu(ds["f_att_hz"][mask_ov]), ds["atten_db_cm"][mask_ov], tan_delta_e=0.0
    )
    assert 78.0 < inf0["eps_real"] < 86.0
    assert inf0["rms_overlap_db_cm"] < 3.0


def test_committed_tasks_figures_and_fit_vs_transfer_matrix():
    """Verify data/task_0..3.{json,npz}, setup figures 00a..00d, and FIT vs transfer-matrix accuracy."""
    for idx in range(4):
        jp = TUTORIAL_DIR / "data" / f"task_{idx}.json"
        npzp = TUTORIAL_DIR / "data" / f"task_{idx}.npz"
        assert jp.is_file(), f"Missing {jp}"
        assert npzp.is_file() and npzp.stat().st_size > 0, f"Missing {npzp}"

    # 1. Verify setup figures (00a..00d) and physics figures (01..08) exist and are within size bounds
    expected_figs = [
        "00a_free_space_slab_setup.png",
        "00b_free_space_slab_mesh_pulse.png",
        "00c_metal_backed_absorber_setup.png",
        "00d_metal_backed_absorber_mesh_pulse.png",
        "01_eccosorb_rf_lb_fit_and_attenuation.png",
        "02_six_material_classes_dispersion.png",
        "03_slab_rt_and_absorption_spectra.png",
        "04_schelkunoff_se_decomposition.png",
        "05_metal_backed_absorber_and_rozanov.png",
        "06_resolution_convergence_study.png",
        "07_pulse_slab_interaction_snapshots.png",
        "08_emi_slab_wave_propagation.gif",
    ]
    for fname in expected_figs:
        fp = TUTORIAL_DIR / "figures" / fname
        assert fp.is_file() and fp.stat().st_size > 1024, f"Missing or empty figure {fp}"
    gif_path = TUTORIAL_DIR / "figures" / "08_emi_slab_wave_propagation.gif"
    assert gif_path.stat().st_size < 5 * 1024 * 1024, "GIF exceeds 5 MB limit"

    # 2. Task 0: RF-LB in-band fit (1 Debye + 1 Lorentz < 0.5% vs 2 Debye ~ 4.5%) and 1-6 GHz extrapolation
    t0 = np.load(TUTORIAL_DIR / "data" / "task_0.npz")
    assert float(t0["rel_rms_1d1l"]) < 0.005
    assert float(t0["rel_rms_1d1l"]) < 0.10 * float(t0["rel_rms_2deb"])
    assert float(t0["rms_att_extrap_2deb"]) < 4.0
    assert abs(float(t0["att_wide_1d1l"][-1]) - float(t0["atten_ds_db_cm"][-1])) < 3.5

    # 3. Task 1: All 6 EMI classes x 3 thicknesses FIT vs transfer-matrix within resolution tolerance
    t1 = np.load(TUTORIAL_DIR / "data" / "task_1.npz")
    rms_R = t1["rms_R_table"]
    rms_T = t1["rms_T_table"]
    rms_A = t1["rms_A_table"]
    assert rms_R.shape == (6, 3)
    assert np.all(rms_R < 0.004), f"Max RMS(|R|^2) = {np.max(rms_R):.4f} exceeds tolerance"
    assert np.all(rms_T < 0.004), f"Max RMS(|T|^2) = {np.max(rms_T):.4f} exceeds tolerance"
    assert np.all(rms_A < 0.004), f"Max RMS(A) = {np.max(rms_A):.4f} exceeds tolerance"

    # Check grid resolution rules across all 18 slab runs in Task 1
    for ic, cinfo in enumerate(hlp.EMI_CLASSES):
        for it in range(3):
            ca = 4.0 if cinfo["use_thin_sheet"] else float(t1["cells_across_table"][ic, it])
            _assert_emi_resolution_rules(
                float(t1["cpw_table"][ic, it]),
                ca,
                float(t1["cells_skin_table"][ic, it]),
                min_cpw=20.0,
            )

    # Check Schelkunoff algebraic identity SE_ex == SE_R + SE_A + SE_M and physical class contrast
    se_sum = t1["SE_R_table"] + t1["SE_A_table"] + t1["SE_M_table"]
    np.testing.assert_allclose(t1["SE_ex_table"][:5], se_sum[:5], atol=1e-12)
    np.testing.assert_allclose(t1["SE_ex_table"][5], se_sum[5], atol=2e-4)

    # At d_index = 1 (2 mm for RF-LB and TiO2, 1 mm for CNT/Ni-C, 50 nm for Cu), near 2 GHz:
    idx_2g = int(np.argmin(np.abs(t1["freqs"] - 2.0e9)))
    # Magnetic absorber (RF-LB, class 0): absorption-dominated (SE_A > SE_R, A > 0.45)
    assert float(t1["SE_A_table"][0, 1, idx_2g]) > float(t1["SE_R_table"][0, 1, idx_2g])
    assert float(t1["A_fit_table"][0, 1, idx_2g]) > 0.45
    # High-eps ceramic (TiO2, class 2): reactive reflection only (A < 0.015 across whole band)
    assert float(np.max(t1["A_fit_table"][2, 1])) < 0.015
    assert float(np.max(t1["R_fit_table"][2, 1])) > 0.85
    # Conductors (CNT class 3, Ni-graphite class 4, Cu foil class 5): strong reflection loss SE_R > 9 dB
    assert float(t1["SE_R_table"][3, 1, idx_2g]) > 9.0
    assert float(t1["SE_R_table"][4, 1, idx_2g]) > 17.0
    assert float(t1["SE_R_table"][5, 1, idx_2g]) > 60.0

    # 4. Task 2: Metal-backed absorber RL(f, d) FIT vs exact and Rozanov causality bound
    t2 = np.load(TUTORIAL_DIR / "data" / "task_2.npz")
    assert float(t2["max_rms_RL_db"]) < 0.10
    # Rozanov bound: both magnetic absorbers satisfy integral <= 2*pi^2*mu_s*d and beat non-magnetic 2*pi^2*d
    assert np.all(t2["roz_int_rf_mm"] < t2["roz_bound_rf_mm"])
    assert np.all(t2["roz_int_sd_mm"] < t2["roz_bound_sd_mm"])
    assert np.all(t2["roz_int_rf_mm"] > t2["roz_bound_nonmag_mm"])
    assert np.all(t2["roz_int_sd_mm"] > t2["roz_bound_nonmag_mm"])

    # 5. Task 3: Resolution convergence and locally refined z-mesh vs uniform baseline
    t3 = np.load(TUTORIAL_DIR / "data" / "task_3.npz")
    for ic in range(4):
        # Error at 40 cpw is smaller than at 10 cpw by > 4x
        assert float(t3["rms_R_conv"][ic, -1]) < 0.25 * float(t3["rms_R_conv"][ic, 0])
    # Locally refined z-mesh uses < 35% of the cells of the uniform z-mesh at the same slab resolution
    assert np.all(t3["nz_graded_conv"][0, : len(t3["nz_uniform"])] < 0.35 * t3["nz_uniform"])


def test_emi_mutations_provably_fail():
    """Mutation checks required by specification:
    1. mu'' -> 0 makes the Eccosorb RF-LB absorption vanish (provably fails A > 0.40 and SE_A > SE_R).
    2. sigma -> 0 removes the CNT rubber shielding (provably fails SE > 12 dB).
    3. Active sign flip Im(mu) > 0 produces unphysical gain R + T > 1 (A < 0).
    4. Under-resolved cpw < 20 or cells_per_skin_depth < 4 provably fails _assert_emi_resolution_rules.
    """
    freqs = np.linspace(0.30e9, 2.50e9, 29)
    d_rf = 2.0e-3
    rf_lb = dispersive_lib["eccosorb rf-lb"]

    # Baseline RF-LB has strong magnetic absorption (mean A > 0.45, SE_A > 10 dB at 2 GHz)
    tm_rf = hlp.exact_slab_rt(freqs, rf_lb.epsilon(freqs), rf_lb.mu(freqs), d_rf)
    sch_rf = hlp.schelkunoff_decomposition(freqs, rf_lb.epsilon(freqs), rf_lb.mu(freqs), d_rf)
    assert float(np.mean(tm_rf["A"])) > 0.45
    assert float(sch_rf["SE_A_db"][-1]) > 10.0

    # Mutation 1: Setting mu'' -> 0 (lossless positive real permeability |Re(mu)|) makes RF-LB absorption vanish identically
    mu_lossless = np.maximum(np.real(rf_lb.mu(freqs)), 1.0) + 0.0j
    eps_lossless = np.real(rf_lb.epsilon(freqs)) + 0.0j
    tm_rf_mut = hlp.exact_slab_rt(freqs, eps_lossless, mu_lossless, d_rf)
    sch_rf_mut = hlp.schelkunoff_decomposition(freqs, eps_lossless, mu_lossless, d_rf)
    np.testing.assert_allclose(tm_rf_mut["A"], 0.0, atol=1e-14)
    np.testing.assert_allclose(sch_rf_mut["SE_A_db"], 0.0, atol=1e-13)
    with pytest.raises(AssertionError):
        assert float(np.mean(tm_rf_mut["A"])) > 0.45

    # Mutation 2: Setting sigma -> 0 on CNT silicone removes conductive shielding and absorption
    cnt = dispersive_lib["cnt silicone"]
    cnt_mut_no_sigma = DispersiveMaterial(eps_inf=cnt.eps_inf, sigma=0.0)
    sch_cnt = hlp.schelkunoff_decomposition(freqs, cnt.epsilon(freqs), cnt.mu(freqs), 1.0e-3)
    sch_cnt_mut = hlp.schelkunoff_decomposition(
        freqs, cnt_mut_no_sigma.epsilon(freqs), cnt_mut_no_sigma.mu(freqs), 1.0e-3
    )
    assert float(np.mean(sch_cnt["SE_total_db"])) > 14.0
    assert float(np.max(sch_cnt_mut["SE_total_db"])) < 2.0
    np.testing.assert_allclose(sch_cnt_mut["SE_A_db"], 0.0, atol=1e-13)
    with pytest.raises(AssertionError):
        assert float(np.mean(sch_cnt_mut["SE_total_db"])) > 14.0

    # Mutation 3: Wrong sign convention mu' + i*mu'' (active medium) violates energy conservation (A < 0)
    mu_active = np.conj(rf_lb.mu(freqs))
    n_act = np.sqrt(rf_lb.epsilon(freqs) * mu_active)
    z_act = np.sqrt(mu_active / rf_lb.epsilon(freqs))
    r12 = (z_act - 1.0) / (z_act + 1.0)
    P_gain = np.exp(-1j * (2.0 * np.pi * freqs / 299792458.0) * n_act * d_rf)
    t_gain = (1.0 - r12**2) * P_gain / (1.0 - (r12**2) * (P_gain**2))
    r_gain = r12 * (1.0 - P_gain**2) / (1.0 - (r12**2) * (P_gain**2))
    a_gain = 1.0 - np.abs(r_gain) ** 2 - np.abs(t_gain) ** 2
    assert np.all(a_gain < -0.10)
    with pytest.raises(AssertionError):
        assert np.all(a_gain >= 0.0)

    # Mutation 4: Grid resolution rule violations MUST fail
    with pytest.raises(AssertionError, match="Rule 1"):
        _assert_emi_resolution_rules(cpw=12.0, cells_across=8.0, cells_per_skin=5.0, min_cpw=20.0)
    with pytest.raises(AssertionError, match="Rule 2"):
        _assert_emi_resolution_rules(cpw=24.0, cells_across=2.5, cells_per_skin=5.0, min_cpw=20.0)
    with pytest.raises(AssertionError, match="Skin-depth rule"):
        _assert_emi_resolution_rules(cpw=24.0, cells_across=8.0, cells_per_skin=2.0, min_cpw=20.0)
