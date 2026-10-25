"""Tests for Tutorial 04-03: 77 GHz Series-Fed Patch Array (Yang et al., Sensors 2020)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "04_radiation_and_antennas"
    / "04-05_radar_patch_array_77g"
)
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _helpers as h  # noqa: E402


def test_yang2020_geometry_and_cavity_model():
    """Verify verbatim Yang et al. (2020) dimensions, source locations, cavity model, and loss budget."""
    g = h.YANG2020_GEOMETRY_UM
    assert g["f0_ghz"] == 77.0
    assert g["eps_r"] == 2.20
    assert g["tan_delta"] == 0.0009
    assert g["h_sub_um"] == 254.0
    assert g["L1_patch_um"] == 1260.0
    assert g["W1_patch_um"] == 1540.0
    assert g["L2_line_um"] == 1260.0
    assert g["W2_line_um"] == 320.0
    assert g["L3_qwt_um"] == 718.0
    assert g["W3_qwt_um"] == 429.0
    assert g["L_term_um"] == 0.0
    assert g["termination"] == "open_patch_edge"
    assert g["unit_cell_pitch_um"] == g["L1_patch_um"] + g["L2_line_um"] == 2520.0
    assert g["n_elements"] == 10
    assert g["uniform_code"] == "1111111111"
    assert g["tapered_ga_code"] == "1011111111"
    assert g["tapered_widths_um"] == [1540.0, 320.0] + [1540.0] * 8
    assert g["pure_binary_widths_um"] == [1540.0, 320.0] + [1540.0] * 8

    # Verbatim benchmark numbers from Yang et al., Sensors 20(11), 3066 (2020), with no invented ± uncertainties
    bm = h.YANG2020_BENCHMARKS
    assert "reading_uncertainty_db" not in bm
    assert bm["s11_in_source"] is None
    assert bm["design_goals"]["first_null_beamwidth_deg"] == 8.0
    assert bm["design_goals"]["sll_max_db"] == -14.0
    u_bm = bm["uniform_1111111111"]
    assert u_bm["sll_e_sim_db"] == -12.5
    assert u_bm["sll_e_meas_db"] == -10.0
    assert u_bm["hpbw_e_half_sim_deg"] == 3.7
    assert u_bm["hpbw_e_half_meas_deg"] == 3.7
    assert u_bm["gain_sim_dbi"] == 17.0
    assert u_bm["gain_meas_dbi"] == 15.9
    t_bm = bm["tapered_1011111111"]
    assert t_bm["sll_e_sim_db"] == -16.5
    assert t_bm["sll_e_meas_db"] == -14.8
    assert t_bm["hpbw_e_half_sim_deg"] == 3.7
    assert t_bm["hpbw_e_half_meas_deg"] == 3.8
    assert t_bm["gain_sim_dbi"] == 17.0
    assert t_bm["gain_meas_dbi"] == 15.6

    locs = h.YANG2020_SOURCE_LOCATIONS
    for key in (
        "Operating frequency (77 GHz)",
        "Substrate (Rogers 5880, eps_r = 2.2, h = 254 um)",
        "Patch & feeder dimensions (L1=1260 um, W1=1540 um, L2=1260 um, W2=320 um)",
        "GA chromosome '1011111111' meaning (binary on/off of uniform patches)",
        "Optimized '1011111111' Gain (17 dBi sim, 15.6 dBi meas)",
        "Optimized '1011111111' E-plane SLL (-16.5 dB sim, -14.8 dB meas)",
        "Traditional '1111111111' E-plane SLL (-12.5 dB sim, -10 dB meas)",
        "Return loss |S11| / impedance bandwidth",
    ):
        assert key in locs and len(locs[key]) > 10

    # Both 'tapered' and 'binary_ga' build the paper's binary '1011111111' with 9 uniform W1=1540 um patches
    for topo in ("tapered", "binary_ga"):
        _, geom_bin = h.build_array_structures(topo)
        assert geom_bin["active_mask"] == [True, False, True, True, True, True, True, True, True, True]
        active_widths_um = [
            round(w * 1e6, 1)
            for w, act in zip(geom_bin["patch_widths_y"], geom_bin["active_mask"])
            if act
        ]
        assert active_widths_um == [1540.0] * 9
        assert round(geom_bin["L3"] * 1e6, 1) == 718.0
        assert round(geom_bin["W3"] * 1e6, 1) == 429.0
        assert round(geom_bin["L_term"] * 1e6, 1) == 0.0

    # Separate width-taper variation ('width_taper_variation') has non-uniform widths on the 9 active patches
    _, geom_wt = h.build_array_structures("width_taper_variation")
    assert geom_wt["active_mask"] == [True, False, True, True, True, True, True, True, True, True]
    wt_active_um = [
        round(w * 1e6, 1)
        for w, act in zip(geom_wt["patch_widths_y"], geom_wt["active_mask"])
        if act
    ]
    assert len(set(wt_active_um)) > 3

    eps_p, z0_p = h.microstrip_eps_eff_and_z0(1540e-6, 254e-6, 2.20)
    eps_l, z0_l = h.microstrip_eps_eff_and_z0(320e-6, 254e-6, 2.20)
    assert 1.0 < eps_l < eps_p < 2.20
    assert z0_l > z0_p > 15.0

    cav = h.patch_cavity_resonance_hz()
    assert 100.0 < cav["delta_l_um"] < 160.0
    assert 65.0 < cav["f_res_ghz"] < 76.0
    # Increasing patch length L1 must lower the TM100 cavity resonance
    assert (
        h.patch_cavity_resonance_hz(length_m=1400e-6)["f_res_ghz"]
        < cav["f_res_ghz"]
    )

    loss = h.copper_and_dielectric_loss_77ghz()
    assert 0.20 < loss["skin_depth_um"] < 0.28
    assert loss["roughness_factor"] > 1.2
    assert loss["conductor_loss_db"] > loss["dielectric_loss_db"] > 0.0
    assert 0.6 < loss["total_loss_db"] < 1.6
    assert 65.0 < loss["estimated_efficiency_pct"] < 90.0


def test_array_factor_sidelobe_suppression():
    """Verify analytical array-factor E-plane pattern and sidelobe suppression."""
    ang = np.linspace(-90.0, 90.0, 721)

    # Ideal uniform 10-element array ('1111111111') has first sidelobe near -13.26 dB
    pat_uni = h.series_array_factor_e_plane(ang, topology="uniform")
    sll_uni = h.compute_sll_db(ang, pat_uni)
    assert -14.0 <= sll_uni <= -12.8

    # Paper's ideal equal-amplitude binary '1011111111' array factor x element factor has SLL = -15.51 dB
    pat_bin = h.series_array_factor_e_plane(ang, topology="binary")
    pat_tap = h.series_array_factor_e_plane(ang, topology="tapered")
    assert np.allclose(pat_bin, pat_tap)
    sll_bin = h.compute_sll_db(ang, pat_bin)
    assert -16.0 <= sll_bin <= -15.0
    assert sll_bin < sll_uni - 2.0

    # Separate width-taper variation ('width_taper_variation') suppresses SLL even further
    pat_wt = h.series_array_factor_e_plane(ang, topology="width_taper_variation")
    sll_wt = h.compute_sll_db(ang, pat_wt)
    assert sll_wt < sll_bin - 1.8


def test_preflight_and_project_grid_rules():
    """Verify wakis.estimate preflight and project grid rules on the fine and GPU meshes."""
    report = h.preflight_all_levels()
    assert set(report.keys()) == {
        "coarse",
        "medium",
        "fine",
        "gpu_superfine_32um",
    }
    for lvl, info in report.items():
        assert info["n_cells"] > 100_000
        assert info["memory_mb"] > 500.0
        assert info["cpu_wall_est_s"] > info["gpu_wall_est_s"] > 0.0

    # Fine CPU mesh satisfies BOTH >= 20 cpw in Rogers 5880 AND >= 4 cells across h and W2
    fine = report["fine"]
    assert fine["rule_20cpw_met"] is True
    assert fine["min_cpw_sub"] >= 20.0
    assert fine["cells_across_h_254um"] >= 4
    assert fine["cells_across_W2_320um"] >= 4
    assert fine["rule_4cells_feature_met"] is True

    # GPU superfine mesh has >= 40 cpw and >= 8 cells across h and W2
    gpu_lvl = report["gpu_superfine_32um"]
    assert gpu_lvl["min_cpw_sub"] >= 40.0
    assert gpu_lvl["cells_across_h_254um"] >= 8
    assert gpu_lvl["cells_across_W2_320um"] >= 8


def test_committed_data_and_figures():
    """Verify committed simulation outputs, resolution study, and SLL gap analysis against Yang et al. (2020)."""
    data_dir = TUT_DIR / "data"
    fig_dir = TUT_DIR / "figures"

    summary_path = data_dir / "summary.json"
    assert summary_path.exists(), "Missing data/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "reading_uncertainty_db" not in summary.get("benchmarks", {})
    assert summary["benchmarks"]["s11_in_source"] is None
    runs = summary["runs"]

    for tname in (*h.CPU_TASKS, "gpu_superfine_32um"):
        assert tname in runs, f"Missing run {tname} in summary.json"
        npz_path = data_dir / f"{tname}.npz"
        assert npz_path.exists(), f"Missing {npz_path.name}"
        assert npz_path.stat().st_size < 2 * 1024 * 1024

    uni = runs["uniform_coarse"]
    tap_c = runs["tapered_coarse"]
    tap_m = runs["res_medium"]
    tap_f = runs["res_fine"]
    tap_sf = runs["gpu_superfine_32um"]

    # Uniform '1111111111' SLL near Yang et al. measured -10.0 dB (Fig. 9a, p. 9; -9.32 dB at 77 GHz, -9.53 dB at 76 GHz)
    assert -10.5 <= uni["sll_e_db"] <= -9.0

    # Paper's binary '1011111111' SLL improves monotonically across all 4 mesh levels:
    # -9.98 dB (coarse) -> -10.27 dB (medium) -> -10.47 dB (fine) -> -10.66 dB (gpu_superfine_32um)
    slls = [r["sll_e_db"] for r in (tap_c, tap_m, tap_f, tap_sf)]
    assert slls[0] > slls[1] > slls[2] > slls[3]
    for r in (tap_c, tap_m, tap_f, tap_sf):
        assert -11.5 <= r["sll_e_db"] <= -9.8
        assert r["sll_e_db"] <= uni["sll_e_db"] - 0.6

    # Resonant frequency f_res converges on conformal sheets to 74.15-74.60 GHz across medium/fine/superfine
    for r in (tap_m, tap_f, tap_sf):
        assert 74.0 <= r["f_res_ghz"] <= 74.8
        assert r["s11_min_db"] <= -20.0

    # Peak broadside gain near Yang et al. measured 15.6 dBi (within 0.95 dB on fine/superfine)
    assert 14.5 <= tap_f["g_max_lossy_dbi"] <= 16.0
    assert 14.5 <= tap_sf["g_max_lossy_dbi"] <= 16.0
    assert 15.5 <= tap_f["d_max_pec_dbi"] <= 17.1
    assert 15.5 <= tap_sf["d_max_pec_dbi"] <= 17.1

    # Narrow E-plane beamwidth (half-HPBW ±3.84° to ±3.94°, matching Yang et al. measured ±3.8° within 0.14°)
    # and broadside beam direction (0.0° on fine and superfine)
    assert 7.5 <= tap_f["hpbw_e_full_deg"] <= 8.1
    assert 7.5 <= tap_sf["hpbw_e_full_deg"] <= 8.1
    assert abs(tap_f["theta_peak_deg"]) <= 0.5
    assert abs(tap_sf["theta_peak_deg"]) <= 0.5
    assert tap_f["hpbw_h_full_deg"] > 75.0

    # Verify SLL gap study (extracted TM100 excitations + one-at-a-time factor tests + separate width-taper variation)
    gap_path = data_dir / "sll_gap_study.json"
    assert gap_path.exists(), "Missing data/sll_gap_study.json"
    gap = json.loads(gap_path.read_text(encoding="utf-8"))
    exc_slot = gap["extracted_tm100_patch_excitations_77ghz"]["tm100_ez_slot"]
    exc_curr = gap["extracted_tm100_patch_excitations_77ghz"]["tm100_jx_current"]
    # Patch #0 has normalized amplitude 1.0, omitted Patch #1 is 0.0, downstream Patches #2-#9 are ~0.41-0.51
    assert exc_slot["amplitudes_norm"][0] == 1.0
    assert exc_slot["amplitudes_norm"][1] == 0.0
    assert all(0.40 <= a <= 0.52 for a in exc_slot["amplitudes_norm"][2:])
    # Extracted AFxEF SLL brackets the full-wave 3D FIT SLL (-9.98 to -10.66 dB)
    assert exc_curr["sll_af_times_ef_db"] <= tap_c["sll_e_db"] <= exc_slot["sll_af_times_ef_db"]
    # Dielectric loss in simulation and analytical Cu+dielectric loss slightly degrade SLL rather than improving it
    loss_sim_sll = gap["factor_b_loss_in_simulation_and_analytical"]["lossy_rogers5880_sigma_0p00846_S_per_m"]["sll_77"]
    assert loss_sim_sll > tap_c["sll_e_db"]
    assert exc_slot["sll_af_times_ef_with_analytic_loss_db"] > exc_slot["sll_af_times_ef_db"]
    # Open-circuit termination at 10th patch outperforms 800 um stub and matched 71.8 Ohm load
    fc = gap["factor_c_feed_transition_and_termination"]
    assert tap_c["sll_e_db"] < fc["termination_open_stub_800um"]["at_77ghz"]["sll_e_db"]
    assert tap_c["sll_e_db"] < fc["termination_matched_load_71p8_ohm"]["at_77ghz"]["sll_e_db"]
    # Separate design variation (raised-cosine width taper on '1011111111') achieves SLL < -15.0 dB
    wt_var = gap["design_variation_width_taper_coarse"]
    assert wt_var["sll_e_db"] < -15.0

    # Verify uniform squint study & pure binary '1011111111' study
    sq_path = data_dir / "squint_and_binary_study.json"
    assert sq_path.exists(), "Missing data/squint_and_binary_study.json"
    sq = json.loads(sq_path.read_text(encoding="utf-8"))
    u_sq = sq["uniform_squint_study"]
    b_ga = sq["pure_binary_1011111111"]
    assert 75.5 <= u_sq["f_broadside_zero_squint_ghz"] <= 76.5
    scan_by_f = {round(pt["f_ghz"], 1): pt for pt in u_sq["scan"]}
    assert abs(scan_by_f[76.0]["theta_peak_deg"]) <= 0.5
    assert 1.0 <= scan_by_f[77.0]["theta_peak_deg"] <= 2.5
    assert 3.0 <= scan_by_f[78.0]["theta_peak_deg"] <= 4.5
    assert b_ga["sll_e_db"] < uni["sll_e_db"] - 0.5
    assert wt_var["sll_e_db"] < b_ga["sll_e_db"] - 5.0

    expected_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "01_geometry_3d_views.png",
        "02_mesh_and_cpw_views.png",
        "03_near_fields_and_currents.png",
        "04_radiated_field_evolution.gif",
        "05_far_field_3d_and_cuts.png",
        "06_combined_3d_overlays.png",
        "07_s11_impedance_smith.png",
        "08_resolution_and_loss_study.png",
    ]
    for fname in expected_figs:
        fpath = fig_dir / fname
        assert fpath.exists(), f"Missing figure {fname}"
        assert fpath.stat().st_size > 10_000
    assert (fig_dir / "00_setup_table.md").exists(), "Missing 00_setup_table.md"
    assert (fig_dir / "00_setup_table.md").stat().st_size > 200
    assert (fig_dir / "04_radiated_field_evolution.gif").stat().st_size < 5 * 1024 * 1024


def test_live_3d_fit_array_and_mutations():
    """Run a fast live 3D FIT simulation and verify that physical mutations fail."""
    res = h.run_array_simulation(
        topology="tapered",
        level="coarse",
        record_fields=False,
        record_animation=False,
        compute_3d_pattern=False,
        sim_time_ps=290.0,
        energy_decay=2e-4,
    )
    assert res["d_max_pec_dbi"] > 15.1
    assert res["sll_e_db"] < -9.7
    assert abs(res["theta_peak_deg"]) <= 1.0
    assert res["hpbw_h_full_deg"] > 8.0 * res["hpbw_e_full_deg"]
    assert np.all(res["s11_db"] <= 0.05)

    # Mutation check 1: introducing a pi inter-element phase error destroys the broadside peak
    ang = np.linspace(-90.0, 90.0, 361)
    pat_in_phase = h.series_array_factor_e_plane(ang, topology="tapered", progressive_phase_rad=0.0)
    pat_out_of_phase = h.series_array_factor_e_plane(ang, topology="tapered", progressive_phase_rad=np.pi)
    i_broadside = int(np.argmin(np.abs(ang)))
    with pytest.raises(AssertionError):
        assert np.isclose(pat_out_of_phase[i_broadside], pat_in_phase[i_broadside], atol=3.0)

    # Mutation check 2: dropping the Hammerstad-Jensen fringing extension delta_L shifts f_res by > 10 GHz
    cav = h.patch_cavity_resonance_hz(length_m=1260e-6, width_m=1540e-6, height_m=254e-6, eps_r=2.20)
    f_no_fringing_ghz = (299792458.0 / (2.0 * np.sqrt(cav["eps_eff_patch"]) * 1260e-6)) / 1e9
    with pytest.raises(AssertionError):
        assert np.isclose(f_no_fringing_ghz, cav["f_res_ghz"], atol=5.0)

    # Mutation check 3: invalid inputs raise ValueError
    with pytest.raises(ValueError):
        h.microstrip_eps_eff_and_z0(-100e-6)
    with pytest.raises(ValueError):
        h.series_array_factor_e_plane(ang, weights=np.zeros(10))
