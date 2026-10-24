# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics and numerical validation tests for Tutorial 6.2:
Slow-Wave Structure Cold Test (2D & 3D Corrugated / Glide-Symmetric SWS).
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.constants import c as c_light

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "06_periodic_structures"
    / "06-02_sws_cold_test"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    GratingSWSSpec,
    MarosiPillarSWSSpec,
    build_2d_grating_grid,
    build_3d_grating_grid,
    build_3d_marosi_pillar_grid,
    compute_mcvey_analytic_pierce_impedance,
    compute_mcvey_harmonic_convergence,
    compute_mcvey_power_and_energy_check,
    extract_bloch_mode_dft,
    mcvey_dispersion_determinant,
    project_floquet_space_harmonics,
    run_bloch_harminv_point,
    solve_mcvey_band_frequencies,
)
from sws_cold_test import (  # noqa: E402
    DATA_DIR,
    compute_mcvey_slot_mode_extrapolation,
    get_task_table,
)


def test_mcvey_field_matching_dispersion_and_harmonic_convergence():
    """
    Verify McVey (1994) mode-matching dispersion on the 2D double-corrugated
    parallel-plate grating:
    1. Space-harmonic truncation error decreases monotonically from N_harm = 1
       to N_harm = 20 (< 5e-4 relative error vs N_harm = 35).
    2. Mirror symmetry opens a > 50 GHz first-order Bragg stopband at beta_0 d = pi
       with vanishing group velocity, whereas glide symmetry closes the stopband
       identically (Delta f_gap = 0) with finite group velocity v_g / c > 0.35.
    3. 2D-to-3D rectangular waveguide transverse cutoff mapping obeys
       f_3D = sqrt(f_2D^2 + (c / 2a)^2).
    4. First-principles Poynting power P_direct = P_tunnel + P_slot agrees with
       energy-transport power P_vgW = v_g * W / d to < 1% in both 2D and 3D,
       whereas setting P_slot = 0 (include_slot_power=False) fails by > 5%.
    """
    spec = GratingSWSSpec()
    # Verify GratingSWSSpec matches Karetnikova et al. (2018) Table I
    assert spec.period_m == pytest.approx(0.50e-3)
    assert spec.vane_length_m == pytest.approx(0.10e-3)
    assert spec.slot_width_m == pytest.approx(0.40e-3)
    assert spec.vane_height_m == pytest.approx(0.30e-3)
    assert spec.tunnel_half_height_m == pytest.approx(0.10e-3)
    assert 2.0 * spec.tunnel_half_height_m == pytest.approx(0.20e-3)
    assert spec.waveguide_width_m == pytest.approx(0.85e-3)
    assert spec.beam_half_thickness_m == pytest.approx(0.075e-3)
    assert 2.0 * spec.beam_half_thickness_m == pytest.approx(0.15e-3)
    assert spec.beam_width_m == pytest.approx(0.75e-3)
    assert spec.beam_voltage_v == pytest.approx(20.0e3)
    assert spec.total_height_m == pytest.approx(0.80e-3)
    assert spec.min_feature_m == pytest.approx(0.10e-3)

    d = spec.period_m
    beta_half_pi = 0.5 * np.pi / d
    beta_pi = np.pi / d

    # 1. Monotonic space-harmonic convergence at beta_0 d = 0.5 pi
    conv = compute_mcvey_harmonic_convergence(
        beta_half_pi,
        spec=spec,
        symmetry="glide",
        n_harm_values=(1, 2, 5, 12, 20, 35),
        m_slot_values=(1, 3),
    )
    rel_errs = conv["rel_err_vs_nharm"][-1]
    assert rel_errs[0] > 3e-3, (
        f"Expected N_harm=1 relative truncation error > 0.3%, got {rel_errs[0]:.4e}"
    )
    assert rel_errs[1] < rel_errs[0]
    assert rel_errs[2] < rel_errs[1]
    assert rel_errs[3] < rel_errs[2]
    assert rel_errs[4] < 5e-4, (
        f"Expected N_harm=20 relative error < 5e-4, got {rel_errs[4]:.4e}"
    )

    # 2. Mirror stopband vs glide stopband closure at beta_0 d = pi
    roots_mir_pi_hz = solve_mcvey_band_frequencies(
        beta_pi,
        spec=spec,
        symmetry="mirror",
        n_harm=15,
        m_slot=3,
        f_min_ghz=10.0,
        f_max_ghz=245.0,
        n_scan=260,
    )
    assert len(roots_mir_pi_hz) >= 2
    f1_mir_ghz = roots_mir_pi_hz[0] * 1e-9
    f2_mir_ghz = roots_mir_pi_hz[1] * 1e-9
    gap_mirror_ghz = f2_mir_ghz - f1_mir_ghz
    assert gap_mirror_ghz > 50.0, (
        f"Expected mirror Bragg stopband > 50 GHz at beta_0 d = pi, got {gap_mirror_ghz:.2f} GHz"
    )

    # Group velocity of mirror mode vanishes at the Bragg boundary beta_0 d = pi
    f_mir_near_hz = solve_mcvey_band_frequencies(
        0.98 * np.pi / d,
        spec=spec,
        symmetry="mirror_odd",
        n_harm=15,
        m_slot=3,
        f_min_ghz=10.0,
        f_max_ghz=200.0,
        n_scan=180,
    )[0]
    vg_mir_edge_over_c = (
        2.0
        * np.pi
        * abs(roots_mir_pi_hz[0] - f_mir_near_hz)
        / (0.02 * np.pi / d)
        / c_light
    )
    assert vg_mir_edge_over_c < 0.05

    # Glide symmetry closes the stopband at beta_0 d = pi with v_g / c > 0.35
    roots_glide_near_hz = solve_mcvey_band_frequencies(
        0.96 * np.pi / d,
        spec=spec,
        symmetry="glide_b",
        n_harm=15,
        m_slot=3,
        f_min_ghz=145.0,
        f_max_ghz=235.0,
        n_scan=160,
    )
    roots_glide_pi_hz = solve_mcvey_band_frequencies(
        beta_pi,
        spec=spec,
        symmetry="glide",
        n_harm=15,
        m_slot=3,
        f_min_ghz=145.0,
        f_max_ghz=235.0,
        n_scan=160,
    )
    assert len(roots_glide_pi_hz) >= 1
    f_glide_pi_ghz = roots_glide_pi_hz[0] * 1e-9
    assert f1_mir_ghz + 15.0 < f_glide_pi_ghz < f2_mir_ghz - 15.0, (
        f"Glide crossing frequency {f_glide_pi_ghz:.2f} GHz must lie strictly inside "
        f"the mirror stopband [{f1_mir_ghz:.2f}, {f2_mir_ghz:.2f}] GHz"
    )
    vg_glide_over_c = (
        2.0
        * np.pi
        * abs(roots_glide_pi_hz[0] - roots_glide_near_hz[0])
        / (0.04 * np.pi / d)
        / c_light
    )
    assert vg_glide_over_c > 0.25, (
        f"Expected finite glide group velocity v_g/c > 0.25 at beta_0 d = pi, got {vg_glide_over_c:.3f}"
    )

    # 3. 2D-to-3D rectangular waveguide transverse cutoff mapping and analytic K_n
    fc_te10_hz = 0.5 * c_light / spec.waveguide_width_m
    f_2d_half_pi_hz = conv["f_ref_fundamental_ghz"] * 1e9
    f_3d_half_pi_hz = solve_mcvey_band_frequencies(
        beta_half_pi,
        spec=spec,
        symmetry="glide_b",
        n_harm=15,
        m_slot=3,
        f_cut_hz=fc_te10_hz,
        f_min_ghz=180.0,
        f_max_ghz=260.0,
        n_scan=180,
    )[0]
    assert f_3d_half_pi_hz == pytest.approx(
        float(np.hypot(f_2d_half_pi_hz, fc_te10_hz)), rel=1e-4
    )

    # First-principles 2D power & energy check at beta_0 d = 0.5 pi (N_harm=15, m_slot=5)
    chk_2d = compute_mcvey_power_and_energy_check(
        beta_half_pi,
        spec=spec,
        symmetry="glide_b",
        n_harm=15,
        m_slot=5,
        f_cut_hz=0.0,
        delta_beta0_d_over_pi=1.0e-3,
        include_slot_power=True,
    )
    assert chk_2d["p_rel_diff"] < 0.01, (
        f"Expected 2D P_direct and P_vgW to agree to < 1%, got {chk_2d['p_rel_diff']*100:.3f}%"
    )
    assert chk_2d["p_slot_w"] > 0.35 * chk_2d["p_direct_w"]

    # First-principles 3D power & energy check at Task 9 point beta_0 d = 0.575 pi
    chk_3d = compute_mcvey_power_and_energy_check(
        0.575 * np.pi / d,
        spec=spec,
        symmetry="glide_b",
        n_harm=15,
        m_slot=5,
        f_cut_hz=fc_te10_hz,
        delta_beta0_d_over_pi=1.0e-3,
        include_slot_power=True,
    )
    assert chk_3d["p_rel_diff"] < 0.01, (
        f"Expected 3D P_direct and P_vgW to agree to < 1%, got {chk_3d['p_rel_diff']*100:.3f}%"
    )

    idx_0 = int(np.where(chk_2d["n_indices"] == 0)[0][0])
    idx_m1 = int(np.where(chk_2d["n_indices"] == -1)[0][0])
    # In glide_b (lower glide branch), n=0 has sinh(tau_0 y) -> K_0(y=0) == 0,
    # while n=-1 has cosh(tau_{-1} y) -> first-principles K_{-1}(y=0) = 1.8338 Ohm
    # at (N_harm=15, M_slot=5), 1.8274 Ohm at (40, 6), and 1.7896 Ohm at (40, 10)
    # (2.4% variation across M_slot in [5, 10])
    assert chk_2d["k_n_axis_ohm"][idx_0] == pytest.approx(0.0, abs=1e-12)
    assert chk_2d["k_n_axis_ohm"][idx_m1] == pytest.approx(1.8338, rel=0.025)
    assert chk_2d["k_n_quarter_ohm"][idx_m1] > chk_2d["k_n_axis_ohm"][idx_m1]
    assert chk_2d["k_n_beam_avg_ohm"][idx_m1] > chk_2d["k_n_axis_ohm"][idx_m1]
    assert chk_2d["k_n_quarter_ohm"][idx_0] > 0.05


def test_task_table_resolution_and_mesh_metrics():
    """
    Verify that the 12-task study table enforces >= 4 cells across the smallest
    geometric feature (0.10 mm) and >= 10 cells per minimum wavelength on every
    non-demo task, while the two under-resolved demo tasks have < 4 cells across
    the feature. Also validate any generated task JSON files on disk.
    """
    tasks_q = get_task_table(quick=True)
    tasks_f = get_task_table(quick=False)
    assert len(tasks_q) == 12
    assert len(tasks_f) == 12

    spec_2d = GratingSWSSpec()
    spec_marosi = MarosiPillarSWSSpec()
    wavelength_min_2d_um = (c_light / 235.0e9) * 1e6
    wavelength_min_marosi_um = (c_light / 80.0e9) * 1e6

    for mode_name, table in (("quick", tasks_q), ("full", tasks_f)):
        demo_names = [t["name"] for t in table if t["name"].startswith("under_resolved")]
        assert demo_names == ["under_resolved_staircase", "under_resolved_conformal"]

        for t in table:
            is_demo = t["name"].startswith("under_resolved")
            conformal = t["conformal"]
            kind = t["kind"]
            if kind.startswith("2d"):
                _, m = build_2d_grating_grid(
                    spec=spec_2d,
                    n_feat=int(t["n_feat"]),
                    glide_symmetric=True,
                    conformal=conformal,
                )
                cpw = wavelength_min_2d_um / m["dx_background"]
            elif kind == "3d_grating":
                _, m = build_3d_grating_grid(
                    spec=spec_2d,
                    n_feat=int(t["n_feat"]),
                    nx=int(t["nx"]),
                    glide_symmetric=True,
                    conformal=conformal,
                )
                cpw = wavelength_min_2d_um / m["dx_background"]
            else:
                _, m = build_3d_marosi_pillar_grid(
                    spec=spec_marosi,
                    nx=int(t["nx"]),
                    ny=int(t["ny"]),
                    nz=int(t["nz"]),
                    glide_symmetric=bool(t["glide"]),
                    conformal=conformal,
                )
                cpw = wavelength_min_marosi_um / m["dx_background"]

            assert cpw >= 10.0, (
                f"[{mode_name}] Task {t['name']} has CPW={cpw:.2f} < 10"
            )
            if is_demo:
                assert m["cells_across_feature"] == pytest.approx(2.0, abs=0.05)
                assert m["cells_across_feature"] < 4.0
            else:
                assert m["cells_across_feature"] >= 4.0 - 1e-6, (
                    f"[{mode_name}] Production task {t['name']} has "
                    f"cells_across_feature={m['cells_across_feature']:.2f} < 4"
                )

    # Verify generated task JSON files on disk (full-resolution study)
    required_keys = {
        "dx_background",
        "dx_min",
        "cells_across_feature",
        "n_cells",
        "n_steps",
    }
    json_files = sorted(DATA_DIR.glob("task_*.json"))
    assert len(json_files) == 12, f"Expected 12 task JSON files, found {len(json_files)}"
    meta_by_name = {}
    for jf in json_files:
        meta = json.loads(jf.read_text())
        assert required_keys.issubset(meta.keys()), (
            f"{jf.name} missing required keys: {required_keys - set(meta.keys())}"
        )
        assert meta["dx_background"] > 0.0
        assert meta["dx_min"] > 0.0
        assert meta["n_cells"] > 0
        assert meta["n_steps"] > 0
        name = meta["name"]
        meta_by_name[name] = meta
        if name.startswith("under_resolved"):
            assert meta["cells_across_feature"] == pytest.approx(2.0, abs=0.05)
        else:
            assert meta["cells_across_feature"] >= 4.0 - 1e-6

    # Full-study 3D tasks use 8 cells across the smallest feature
    for name_3d in ("3d_grating_karetnikova", "3d_marosi_glide", "3d_marosi_mirror"):
        assert meta_by_name[name_3d]["cells_across_feature"] == pytest.approx(8.0, abs=0.05)
    assert meta_by_name["3d_grating_karetnikova"]["n_cells"] == 87040

    # Both quick and full 2D sweeps must include beta_0 d = 0.5 pi directly
    for mode_name, table in (("quick", tasks_q), ("full", tasks_f)):
        for t in table:
            if str(t["kind"]).startswith("2d"):
                assert any(abs(float(b) - 0.5) < 1e-12 for b in t["bn_list"]), (
                    f"[{mode_name}] Task {t['name']} does not sample beta_0 d = 0.5 pi"
                )

    # Monotonic 2D conformal frequency convergence across L1 -> L2 -> L3 -> Reference
    f_conf_ladder = [
        meta_by_name[lv]["f1_mid_ghz"]
        for lv in ("2d_conformal_L1", "2d_conformal_L2", "2d_conformal_L3", "2d_conformal_reference")
    ]
    assert f_conf_ladder[0] < f_conf_ladder[1] < f_conf_ladder[2] < f_conf_ladder[3]
    assert abs(f_conf_ladder[3] - f_conf_ladder[2]) / f_conf_ladder[3] < 2e-3

    # Monotonic 2D conformal and staircase Pierce impedance K_{-1}(0) convergence
    k_conf_ladder = [
        meta_by_name[lv]["km1_mid_ohm"]
        for lv in ("2d_conformal_L1", "2d_conformal_L2", "2d_conformal_L3", "2d_conformal_reference")
    ]
    assert k_conf_ladder[0] > k_conf_ladder[1] > k_conf_ladder[2] > k_conf_ladder[3]
    assert abs(k_conf_ladder[3] - k_conf_ladder[2]) / k_conf_ladder[3] < 5e-3
    # Slot-mode-extrapolated McVey limit at N_harm=60 is K_{-1,inf}(0) = 1.7637 Ohm;
    # FIT Reference (1.7749 Ohm) is +0.63% above and FIT Richardson (1.7681 Ohm) is +0.25% above.
    k_fit_rich = k_conf_ladder[3] + (k_conf_ladder[3] - k_conf_ladder[2]) / (
        (16.0 / 12.0) ** 2 - 1.0
    )
    assert k_conf_ladder[3] == pytest.approx(1.76374, rel=0.007)
    assert k_fit_rich == pytest.approx(1.76374, rel=0.005)

    k_stair_ladder = [
        meta_by_name[lv]["km1_mid_ohm"]
        for lv in ("2d_staircase_L1", "2d_staircase_L2", "2d_staircase_L3")
    ]
    assert k_stair_ladder[0] > k_stair_ladder[1] > k_stair_ladder[2]

    # 3D Karetnikova grating (Task 9) with Table I 2a = 200 um & H_b = 150 um
    with np.load(DATA_DIR / "task_09_3d_grating_karetnikova.npz") as d9:
        assert d9["f3d_glide_low_ghz"][0] == pytest.approx(178.8540, rel=1e-3)
        assert d9["f3d_glide_low_ghz"][-1] == pytest.approx(241.5993, rel=1e-3)
        assert d9["f3d_glide_high_ghz"][0] == pytest.approx(260.4307, rel=1e-3)
        assert np.max(np.abs(d9["f3d_glide_low_ghz"] - d9["mcvey_3d_g_low_ghz"]) / d9["mcvey_3d_g_low_ghz"]) < 1.5e-3
        assert np.all(d9["mcvey_3d_km2_beam_ohm"] < 0.1 * d9["mcvey_3d_kp1_beam_ohm"][2])

    # 3D Marosi staggered-pillar SWS (Task 10 glide vs Task 11 mirror control)
    # Merged from 12 single-point Batch shards (--bn-index / --merge-bn); the shard folders are not committed.
    for name_key in ("3d_marosi_glide", "3d_marosi_mirror"):
        assert int(meta_by_name[name_key]["n_bn_shards"]) == 12
        assert meta_by_name[name_key]["n_steps"] > 12 * 40000

    assert meta_by_name["3d_marosi_glide"]["f_degen_ghz"] == pytest.approx(
        60.8566, rel=1e-5
    )
    with (
        np.load(DATA_DIR / "task_10_3d_marosi_glide.npz") as d10,
        np.load(DATA_DIR / "task_11_3d_marosi_mirror.npz") as d11,
    ):
        bn_m = d10["bn"]
        beta0_m = bn_m * np.pi / spec_marosi.period_m
        f_lo_g, f_up_g = d10["f_fwd_ghz"], d10["f_bwd_ghz"]
        f_lo_m, f_up_m = d11["f_fwd_ghz"], d11["f_bwd_ghz"]

        # 1. Glide 3 pi degeneracy vs mirror 3 pi Bragg stopband (> 15 GHz)
        assert f_lo_g[-1] == pytest.approx(60.8566, rel=1e-5)
        assert f_lo_g[-1] == f_up_g[-1]
        assert (f_up_m[-1] - f_lo_m[-1]) > 15.0

        # 2. Independent v_E / v_g consistency within [0.96, 1.01] for interior indices 1..10, all four branches
        for f_arr, ve_arr in (
            (f_lo_g, d10["ve_fwd_over_c"]),
            (f_up_g, d10["ve_bwd_over_c"]),
            (f_lo_m, d11["ve_fwd_over_c"]),
            (f_up_m, d11["ve_bwd_over_c"]),
        ):
            vg_arr = np.gradient(2.0 * np.pi * f_arr * 1e9, beta0_m) / c_light
            ratios = ve_arr[1:11] / vg_arr[1:11]
            assert np.all((ratios >= 0.96) & (ratios <= 1.01))

        # 3. Continuous forward-wave on-axis Z_P interpolated at 58 GHz within 3% of 3.8 Ohm
        kn_n = d10["kn_n_indices"]
        idx_p1 = int(np.where(kn_n == 1)[0][0])
        idx_m2 = int(np.where(kn_n == -2)[0][0])
        zp_fwd_lo_ax = d10["kn_lower_axis_ohm"][:11, idx_p1]
        zp_fwd_up_ax = d10["kn_upper_axis_ohm"][:11, idx_m2]
        f_fwd_cont = np.concatenate([f_lo_g[:11], f_up_g[:11][::-1]])
        zp_fwd_cont_ax = np.concatenate([zp_fwd_lo_ax, zp_fwd_up_ax[::-1]])
        zp_58_ohm = float(np.interp(58.0, f_fwd_cont, zp_fwd_cont_ax))
        assert zp_58_ohm == pytest.approx(3.8, rel=0.03)

        # 4. Glide backward-wave on-axis Z_P < 1e-3 Ohm at all finite points, AND mirror control max > 0.5 Ohm
        zp_bwd_g_up = d10["kn_upper_axis_ohm"][:11, idx_p1]
        zp_bwd_g_lo = d10["kn_lower_axis_ohm"][:11, idx_m2]
        assert np.all(np.isfinite(zp_bwd_g_up)) and np.all(zp_bwd_g_up < 1e-3)
        assert np.all(np.isfinite(zp_bwd_g_lo)) and np.all(zp_bwd_g_lo < 1e-3)

        zp_bwd_m_up = d11["kn_upper_axis_ohm"][:11, idx_p1]
        zp_bwd_m_lo = d11["kn_lower_axis_ohm"][:11, idx_m2]
        assert float(np.max(zp_bwd_m_up)) > 0.5
        assert float(np.max(zp_bwd_m_lo)) > 0.5

        # Mutation checks: mixed-mode branch jump breaks v_E/v_g in [0.96, 1.01], and
        # mislabelling upper n=-2 as backward fails the < 1e-3 Ohm suppression check
        f_mut_mixed = f_lo_g.copy()
        f_mut_mixed[5] = f_up_g[5]
        vg_mut = np.gradient(2.0 * np.pi * f_mut_mixed * 1e9, beta0_m) / c_light
        ratios_mut = d10["ve_fwd_over_c"][1:11] / vg_mut[1:11]
        assert not np.all((ratios_mut >= 0.96) & (ratios_mut <= 1.01))
        assert not np.all(zp_fwd_up_ax < 1e-3)


def test_wakis_2d_bloch_dispersion_and_dft_mode_physics():
    """
    Validate Wakis 2D conformal FIT Bloch eigenmode extraction and Stage-2
    ring-down DFT field integration at L1 (n_feat=4, dy=25 um, 640 cells):
    1. Harminv fundamental eigenfrequency agrees with McVey analytic theory
       within 1.5% at beta_0 d = 0.5 pi.
    2. Mirror symmetry opens a > 50 GHz stopband at beta_0 d = pi, whereas
       glide symmetry closes the stopband and places an eigenmode inside the
       mirror gap.
    3. DFT energy velocity v_E = P_z d / U_cell is positive and agrees with
       analytic v_g within 7%, and the n = -1 space harmonic has beta_{-1} < 0
       opposite to v_E > 0 (backward wave) with non-zero on-axis K_{-1}.
    """
    spec = GratingSWSSpec()
    d = spec.period_m
    g = spec.tunnel_half_height_m
    beta_half_pi = 0.5 * np.pi / d
    beta_pi = np.pi / d

    f_mcvey_half_pi_hz = solve_mcvey_band_frequencies(
        beta_half_pi,
        spec=spec,
        symmetry="glide_b",
        n_harm=15,
        m_slot=3,
        f_min_ghz=10.0,
        f_max_ghz=220.0,
        n_scan=180,
    )[0]

    grid_glide, _ = build_2d_grating_grid(
        spec=spec, n_feat=4, glide_symmetric=True, conformal="pec"
    )
    grid_mirror, _ = build_2d_grating_grid(
        spec=spec, n_feat=4, glide_symmetric=False, conformal="pec"
    )

    probes = [
        ("Ez", (0.0, 0.0, 0.37 * d)),
        ("Ez", (0.0, 0.55 * g, 0.61 * d)),
        ("Hx", (0.0, 0.45 * g, 0.29 * d)),
    ]
    sources = [
        ("Ez", (0.0, 0.40 * g, 0.19 * d), 1.0),
        ("Ez", (0.0, -0.20 * g, 0.67 * d), 0.7),
    ]

    # 1. Stage 1 Harminv at beta_0 d = 0.5 pi
    res_half_pi = run_bloch_harminv_point(
        grid=grid_glide,
        beta0_rad_m=beta_half_pi,
        f_center_hz=130.0e9,
        f_width_hz=100.0e9,
        f_min_hz=20.0e9,
        f_max_hz=235.0e9,
        probe_Specs=probes,
        source_specs=sources,
        n_steps=4000,
    )
    f_fit_half_pi_hz = float(res_half_pi["frequencies_hz"][0])
    rel_err = abs(f_fit_half_pi_hz - f_mcvey_half_pi_hz) / f_mcvey_half_pi_hz
    assert rel_err < 0.015, (
        f"Expected Wakis L1 frequency within 1.5% of McVey ({f_mcvey_half_pi_hz * 1e-9:.2f} GHz), "
        f"got {f_fit_half_pi_hz * 1e-9:.2f} GHz ({rel_err * 100:.2f}%)"
    )

    # 2. Mirror stopband vs glide stopband closure at beta_0 d = pi
    res_mir_pi = run_bloch_harminv_point(
        grid=grid_mirror,
        beta0_rad_m=beta_pi,
        f_center_hz=165.0e9,
        f_width_hz=110.0e9,
        f_min_hz=80.0e9,
        f_max_hz=255.0e9,
        probe_Specs=probes,
        source_specs=sources,
        n_steps=4000,
    )
    freqs_mir_ghz = res_mir_pi["frequencies_hz"] * 1e-9
    f1_mir = float(freqs_mir_ghz[0])
    higher_mir = freqs_mir_ghz[freqs_mir_ghz > 180.0]
    assert len(higher_mir) >= 1, f"Expected upper Bragg branch > 180 GHz, got {freqs_mir_ghz}"
    f2_mir = float(higher_mir[0])
    assert (f2_mir - f1_mir) > 50.0

    res_glide_pi = run_bloch_harminv_point(
        grid=grid_glide,
        beta0_rad_m=beta_pi,
        f_center_hz=175.0e9,
        f_width_hz=90.0e9,
        f_min_hz=140.0e9,
        f_max_hz=235.0e9,
        probe_Specs=probes,
        source_specs=sources,
        n_steps=4000,
    )
    f_glide_pi_ghz = float(res_glide_pi["frequencies_hz"][0] * 1e-9)
    assert f1_mir + 15.0 < f_glide_pi_ghz < f2_mir - 15.0

    # 3. Stage 2 ring-down DFT mode extraction at beta_0 d = 0.5 pi
    dft = extract_bloch_mode_dft(
        grid=grid_glide,
        beta0_rad_m=beta_half_pi,
        mode_freq_hz=f_fit_half_pi_hz,
        source_specs=[("Ez", (0.0, 0.55 * g, 0.21 * d), 1.0)],
        n_harm=4,
        n_ring_steps=2000,
        beam_y_half_m=spec.beam_half_thickness_m,
    )
    assert 0.52 <= dft["v_energy_over_c"] <= 0.62, (
        f"Expected v_E/c near analytic 0.586, got {dft['v_energy_over_c']:.4f}"
    )
    idx_0 = int(np.where(dft["n_indices"] == 0)[0][0])
    idx_m1 = int(np.where(dft["n_indices"] == -1)[0][0])
    # Odd-Hy / sinh-Ez fundamental harmonic n=0: K_0 vanishes on-axis (< 1e-3 Ohm
    # without Nz-1 cross-harmonic leakage) and is finite over the beam cross-section
    assert dft["k_n_axis_ohm"][idx_0] < 1e-3
    assert dft["k_n_beam_avg_ohm"][idx_0] > 0.15
    assert dft["k_n_beam_avg_ohm"][idx_0] > dft["k_n_axis_ohm"][idx_0]
    # Backward n = -1 space harmonic: beta_{-1} < 0 while P_z > 0, and non-zero on-axis K_{-1}
    # First-principles McVey K_{-1}(0) is 1.8274 Ohm at (40, 6) and 1.8338 Ohm at (15, 5)
    assert dft["beta_n"][idx_m1] < 0.0
    assert dft["pz_mean_w"] > 0.0
    assert dft["k_n_axis_ohm"][idx_m1] == pytest.approx(1.8274, rel=0.05)


def test_mutation_glide_phase_and_bloch_sign_reversal():
    """
    Mutation checks proving the tests catch broken physics:
    1. Mutating the glide space-harmonic phase parity (replacing 'glide' with
       'mirror') breaks glide stopband closure at beta_0 d = pi, shifting the
       determinant root by > 20 GHz onto the mirror band edge.
    2. Mutating include_slot_power=False (setting P_slot = 0) breaks the
       P_direct vs P_vgW = v_g * W / d consistency check by > 5% (~39.5% in 2D
       and ~40.6% in 3D).
    3. Mutating the Bloch phase sign beta_0 = +0.5 pi/d -> -0.5 pi/d in the
       Wakis DFT solver preserves the time-reversal-symmetric eigenfrequency
       but strictly reverses the sign of the electromagnetic power flow P_z
       and energy velocity v_E / c < -0.45.
    """
    spec = GratingSWSSpec()
    d = spec.period_m
    g = spec.tunnel_half_height_m
    beta_pi = np.pi / d
    beta_half_pi = 0.5 * np.pi / d

    # Mutation 1: Glide phase parity vs mirror at beta_0 d = pi
    f_glide_true_hz = solve_mcvey_band_frequencies(
        beta_pi,
        spec=spec,
        symmetry="glide_b",
        n_harm=15,
        m_slot=3,
        f_min_ghz=145.0,
        f_max_ghz=235.0,
        n_scan=160,
    )[0]
    f_mutated_no_glide_hz = solve_mcvey_band_frequencies(
        beta_pi,
        spec=spec,
        symmetry="mirror_odd",
        n_harm=15,
        m_slot=3,
        f_min_ghz=80.0,
        f_max_ghz=235.0,
        n_scan=160,
    )[0]
    assert abs(f_glide_true_hz - f_mutated_no_glide_hz) * 1e-9 > 20.0, (
        "Removing the glide phase alternation must break stopband closure by > 20 GHz"
    )
    # At the true glide crossing frequency, det(glide_b) == 0 whereas det(mirror_odd) != 0
    det_glide = abs(
        mcvey_dispersion_determinant(
            f_glide_true_hz,
            beta_pi,
            spec=spec,
            symmetry="glide_b",
            n_harm=15,
            m_slot=3,
        )
    )
    det_mirror = abs(
        mcvey_dispersion_determinant(
            f_glide_true_hz,
            beta_pi,
            spec=spec,
            symmetry="mirror_odd",
            n_harm=15,
            m_slot=3,
        )
    )
    assert det_glide < 1e-7
    assert det_mirror > 0.05

    # Mutation 1b: Halving tunnel_half_height_m to 0.05e-3 (2a = 100 um)
    # shifts f_1 by > 6 GHz and raises first-principles K_{-1}(0) from 1.83 Ohm to > 4.5 Ohm
    spec_old_gap = GratingSWSSpec(
        tunnel_half_height_m=0.05e-3, beam_half_thickness_m=0.025e-3
    )
    f_2d_true_hz = solve_mcvey_band_frequencies(
        beta_half_pi, spec=spec, symmetry="glide_b", n_harm=15, m_slot=5
    )[0]
    f_2d_old_hz = solve_mcvey_band_frequencies(
        beta_half_pi, spec=spec_old_gap, symmetry="glide_b", n_harm=15, m_slot=5
    )[0]
    mcvey_true = compute_mcvey_analytic_pierce_impedance(
        f_2d_true_hz, beta_half_pi, spec=spec, symmetry="glide_b", n_harm=15, m_slot=5
    )
    mcvey_old_gap = compute_mcvey_analytic_pierce_impedance(
        f_2d_old_hz, beta_half_pi, spec=spec_old_gap, symmetry="glide_b", n_harm=15, m_slot=5
    )
    idx_m1 = int(np.where(mcvey_true["n_indices"] == -1)[0][0])
    assert (f_2d_true_hz - f_2d_old_hz) * 1e-9 > 6.0
    assert mcvey_true["k_n_axis_ohm"][idx_m1] < 2.0
    assert mcvey_old_gap["k_n_axis_ohm"][idx_m1] > 4.5

    # Mutation 1c: Setting P_slot = 0 (include_slot_power=False) breaks P_direct vs P_vgW
    # consistency (> 5% error) at both the 2D operating point and the 3D Task 9 point
    chk_2d_no_slot = compute_mcvey_power_and_energy_check(
        beta_half_pi,
        spec=spec,
        symmetry="glide_b",
        n_harm=15,
        m_slot=5,
        f_cut_hz=0.0,
        delta_beta0_d_over_pi=1.0e-3,
        include_slot_power=False,
        freq_hz=f_2d_true_hz,
    )
    assert chk_2d_no_slot["p_slot_w"] == 0.0
    assert chk_2d_no_slot["p_rel_diff"] > 0.05, (
        f"Setting P_slot=0 in 2D must fail P_direct vs P_vgW by > 5%, got {chk_2d_no_slot['p_rel_diff']*100:.2f}%"
    )
    assert chk_2d_no_slot["k_n_axis_ohm"][idx_m1] > 2.9

    fc_te10_hz = 0.5 * c_light / spec.waveguide_width_m
    chk_3d_no_slot = compute_mcvey_power_and_energy_check(
        0.575 * np.pi / d,
        spec=spec,
        symmetry="glide_b",
        n_harm=15,
        m_slot=5,
        f_cut_hz=fc_te10_hz,
        delta_beta0_d_over_pi=1.0e-3,
        include_slot_power=False,
    )
    assert chk_3d_no_slot["p_slot_w"] == 0.0
    assert chk_3d_no_slot["p_rel_diff"] > 0.05, (
        f"Setting P_slot=0 in 3D must fail P_direct vs P_vgW by > 5%, got {chk_3d_no_slot['p_rel_diff']*100:.2f}%"
    )

    # Mutation 2: Reversing Bloch phase sign (+0.5 pi/d -> -0.5 pi/d) reverses P_z and v_E
    grid_glide, _ = build_2d_grating_grid(
        spec=spec, n_feat=4, glide_symmetric=True, conformal="pec"
    )
    probes = [("Ez", (0.0, 0.55 * g, 0.61 * d))]
    sources = [("Ez", (0.0, 0.55 * g, 0.21 * d), 1.0)]

    res_pos = run_bloch_harminv_point(
        grid=grid_glide,
        beta0_rad_m=+beta_half_pi,
        f_center_hz=130.0e9,
        f_width_hz=90.0e9,
        f_min_hz=50.0e9,
        f_max_hz=190.0e9,
        probe_Specs=probes,
        source_specs=sources,
        n_steps=3500,
    )
    res_neg = run_bloch_harminv_point(
        grid=grid_glide,
        beta0_rad_m=-beta_half_pi,
        f_center_hz=130.0e9,
        f_width_hz=90.0e9,
        f_min_hz=50.0e9,
        f_max_hz=190.0e9,
        probe_Specs=probes,
        source_specs=sources,
        n_steps=3500,
    )
    f_pos_hz = float(res_pos["frequencies_hz"][0])
    f_neg_hz = float(res_neg["frequencies_hz"][0])
    assert f_neg_hz * 1e-9 == pytest.approx(f_pos_hz * 1e-9, abs=0.05)

    dft_pos = extract_bloch_mode_dft(
        grid=grid_glide,
        beta0_rad_m=+beta_half_pi,
        mode_freq_hz=f_pos_hz,
        source_specs=sources,
        n_harm=2,
        n_ring_steps=1600,
    )
    dft_neg = extract_bloch_mode_dft(
        grid=grid_glide,
        beta0_rad_m=-beta_half_pi,
        mode_freq_hz=f_neg_hz,
        source_specs=sources,
        n_harm=2,
        n_ring_steps=1600,
    )
    assert dft_pos["pz_mean_w"] > 0.0
    assert dft_neg["pz_mean_w"] < 0.0
    assert dft_neg["v_energy_over_c"] == pytest.approx(
        -dft_pos["v_energy_over_c"], rel=0.02
    )


def test_floquet_harmonic_projection_full_period_vs_off_by_one():
    """
    Verify that `project_floquet_space_harmonics` sums over all N_z axial cells
    of one Bloch period (after `SolverFIT3D` trims the periodic z ghost cell so
    `int(grid.Nz) == N_z` and `grid.edge_ghosts[2] == 0`), recovering exact
    synthetic Floquet space-harmonic amplitudes to < 1e-12, whereas off-by-one
    `nz = int(grid.Nz) - 1` indexing fails by > 1e-2.
    """
    spec = GratingSWSSpec()
    d = spec.period_m
    beta0 = 0.5 * np.pi / d
    grid, _ = build_2d_grating_grid(
        spec=spec, n_feat=4, glide_symmetric=True, conformal="pec"
    )
    # Trim periodic/Bloch ghost cells exactly as SolverFIT3D does on initialization
    grid.remove_edge_ghost(0)
    grid.remove_edge_ghost(2)
    assert int(grid.edge_ghosts[2]) == 0
    nz_period = int(grid.Nz)
    assert nz_period == 20

    n_x_ez = 1
    n_y_ez = int(grid.Ny)
    y_nodes = np.linspace(grid.ymin, grid.ymax, n_y_ez)
    z_true = 0.5 * (np.asarray(grid.z[:-1]) + np.asarray(grid.z[1:]))
    assert z_true.shape == (nz_period,)

    # Exact synthetic Floquet superposition on the staggered E_z cell centers
    amp_m1 = (2.75 - 0.90j) * np.cosh(3200.0 * y_nodes)
    amp_0 = (1.15 + 0.45j) * np.sinh(1800.0 * y_nodes)
    amp_p1 = (-0.65 + 0.30j) * np.cosh(4100.0 * y_nodes)
    beta_m1 = beta0 - 2.0 * np.pi / d
    beta_p1 = beta0 + 2.0 * np.pi / d

    ez_synth = (
        amp_m1[None, :, None] * np.exp(-1j * beta_m1 * z_true[None, None, :])
        + amp_0[None, :, None] * np.exp(-1j * beta0 * z_true[None, None, :])
        + amp_p1[None, :, None] * np.exp(-1j * beta_p1 * z_true[None, None, :])
    )

    proj = project_floquet_space_harmonics(
        ez_synth, grid=grid, beta0_rad_m=beta0, n_harm=4, truncate_off_by_one=False
    )
    assert proj["z_centers_m"].shape == (nz_period,)
    n_indices = proj["n_indices"]
    ez_harm = proj["ez_harm"]

    idx_m1 = int(np.where(n_indices == -1)[0][0])
    idx_0 = int(np.where(n_indices == 0)[0][0])
    idx_p1 = int(np.where(n_indices == 1)[0][0])
    idx_m2 = int(np.where(n_indices == -2)[0][0])
    idx_p2 = int(np.where(n_indices == 2)[0][0])

    # 1. Full-period projection recovers every harmonic amplitude to < 1e-12
    assert np.max(np.abs(ez_harm[idx_m1, 0, :] - amp_m1)) < 1e-12
    assert np.max(np.abs(ez_harm[idx_0, 0, :] - amp_0)) < 1e-12
    assert np.max(np.abs(ez_harm[idx_p1, 0, :] - amp_p1)) < 1e-12
    assert np.max(np.abs(ez_harm[idx_m2, 0, :])) < 1e-12
    assert np.max(np.abs(ez_harm[idx_p2, 0, :])) < 1e-12

    # 2. Mutation check: off-by-one indexing (nz = int(grid.Nz) - 1) fails < 1e-12 by > 1e-2
    proj_old = project_floquet_space_harmonics(
        ez_synth, grid=grid, beta0_rad_m=beta0, n_harm=4, truncate_off_by_one=True
    )
    assert proj_old["z_centers_m"].shape == (nz_period - 1,)
    err_old_m1 = float(np.max(np.abs(proj_old["ez_harm"][idx_m1, 0, :] - amp_m1)))
    err_old_0 = float(np.max(np.abs(proj_old["ez_harm"][idx_0, 0, :] - amp_0)))
    assert err_old_m1 > 1e-2
    assert err_old_0 > 1e-2

    # Even if true z_centers[:Nz-1] were used over Nz-1 cells, orthogonality fails by > 1e-2
    nz_m1 = nz_period - 1
    phase_trunc = (
        np.exp(1j * beta_m1 * z_true[:nz_m1]) / float(nz_m1)
    )
    ez_m1_trunc = ez_synth[0, :, :nz_m1] @ phase_trunc
    assert float(np.max(np.abs(ez_m1_trunc - amp_m1))) > 1e-2


def test_mcvey_slot_mode_extrapolation_vs_fit_richardson():
    """
    Verify that the slot-mode-extrapolated McVey limit (y = y_inf + C * M_slot^-p
    at N_harm = 60, beta_0 d = 0.5 pi, 2D glide) agrees with the Wakis FIT
    Richardson value from committed task_05..08 data to within 0.5% for K_{-1}(0)
    (+0.25%) and 0.1% for f_1 (-0.05%), whereas the un-extrapolated M_slot = 6
    McVey value is NOT converged (|err| > 2% on K_{-1}(0)).
    """
    # 1. Load committed 2D conformal FIT ladder (Tasks 5..8) and compute p=2 Richardson limit
    task_files = [
        DATA_DIR / "task_05_2d_conformal_L1.json",
        DATA_DIR / "task_06_2d_conformal_L2.json",
        DATA_DIR / "task_07_2d_conformal_L3.json",
        DATA_DIR / "task_08_2d_conformal_reference.json",
    ]
    metas = [json.loads(p.read_text()) for p in task_files]
    n_feats = [float(m["cells_across_feature"]) for m in metas]
    assert n_feats == pytest.approx([4.0, 8.0, 12.0, 16.0], rel=1e-12)

    k_l3 = float(metas[2]["km1_mid_ohm"])
    k_ref = float(metas[3]["km1_mid_ohm"])
    f_l3 = float(metas[2]["f1_mid_ghz"])
    f_ref = float(metas[3]["f1_mid_ghz"])

    r_grid = n_feats[3] / n_feats[2]  # 16 / 12
    k_fit_rich = k_ref + (k_ref - k_l3) / (r_grid**2 - 1.0)
    f_fit_rich = f_ref + (f_ref - f_l3) / (r_grid**2 - 1.0)
    assert k_fit_rich == pytest.approx(1.76813, rel=1e-4)
    assert f_fit_rich == pytest.approx(98.0670, rel=1e-4)

    # 2. Extrapolate McVey over M_slot = 10..24 at N_harm = 60 (including M_slot = 6 for truncation check)
    ext = compute_mcvey_slot_mode_extrapolation(
        n_harm=60,
        m_slot_values=(6, 10, 12, 16, 24),
        m_fit_min=10,
        print_table=False,
    )
    m_vals = ext["m_slot_values"]
    f_vals = ext["f1_ghz"]
    k_vals = ext["km1_axis_ohm"]

    # Monotonic convergence in M_slot across 6..24
    assert np.all(np.diff(f_vals) > 0.0)
    assert np.all(np.diff(k_vals) < 0.0)

    # Check finite-M_slot table values and fitted exponents
    idx_m6 = int(np.where(m_vals == 6)[0][0])
    idx_m10 = int(np.where(m_vals == 10)[0][0])
    idx_m24 = int(np.where(m_vals == 24)[0][0])
    assert f_vals[idx_m10] == pytest.approx(97.9146, rel=1e-4)
    assert k_vals[idx_m10] == pytest.approx(1.78969, rel=1e-4)
    assert f_vals[idx_m24] == pytest.approx(98.0613, rel=1e-4)
    assert k_vals[idx_m24] == pytest.approx(1.76985, rel=1e-4)
    assert ext["km1_inf_ohm"] == pytest.approx(1.7637, rel=2e-4)
    assert ext["f1_inf_ghz"] == pytest.approx(98.120, rel=2e-4)
    assert ext["km1_p"] == pytest.approx(1.66, abs=0.05)
    assert ext["f1_p"] == pytest.approx(1.43, abs=0.05)

    # 3. Extrapolated McVey K_{-1,inf} is within 0.5% of FIT Richardson (+0.25%),
    #    and f_{1,inf} is within 0.1% of FIT Richardson (-0.05%)
    rel_err_k_inf = abs(k_fit_rich - ext["km1_inf_ohm"]) / ext["km1_inf_ohm"]
    rel_err_f_inf = abs(f_fit_rich - ext["f1_inf_ghz"]) / ext["f1_inf_ghz"]
    assert rel_err_k_inf < 0.005, (
        f"Expected extrapolated McVey K_-1,inf within 0.5% of FIT Richardson, got {rel_err_k_inf*100:.3f}%"
    )
    assert rel_err_f_inf < 0.001, (
        f"Expected extrapolated McVey f1,inf within 0.1% of FIT Richardson, got {rel_err_f_inf*100:.3f}%"
    )

    # 4. Un-extrapolated M_slot = 6 value (1.8274 Ohm) is NOT within 0.5%: |err| > 2%
    k_m6 = float(k_vals[idx_m6])
    rel_err_k_m6 = abs(k_m6 - k_fit_rich) / k_fit_rich
    assert rel_err_k_m6 > 0.02, (
        f"Expected M_slot=6 McVey K_-1(0) ({k_m6:.4f} Ohm) to differ from FIT Richardson "
        f"({k_fit_rich:.4f} Ohm) by > 2%, got {rel_err_k_m6*100:.3f}%"
    )

    # 5. Mutation check: flipping the sign of the power-law tail C * M^-p in the
    #    extrapolator (adding C * M^-p instead of subtracting it from y(M)) or
    #    using the unextrapolated M_slot=10 value fails the < 0.5% agreement threshold.
    k_mut_wrong_tail_sign = float(
        k_vals[idx_m10] + ext["km1_c"] * (10.0 ** (-ext["km1_p"]))
    )
    assert abs(k_mut_wrong_tail_sign - k_fit_rich) / k_fit_rich > 0.02
    assert abs(float(k_vals[idx_m10]) - k_fit_rich) / k_fit_rich > 0.01


