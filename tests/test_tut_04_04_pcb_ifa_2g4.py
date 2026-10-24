"""Tests for Tutorial 04-02: 2.4 GHz Printed Inverted-F PCB Antenna (TI SWRU120D)."""

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
    / "04-04_pcb_ifa_2g4"
)
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _helpers as h  # noqa: E402


def test_swru120_table_1_1_and_source_table_verbatim():
    """Verify exact TI SWRU120D Table 1-1 & Table 3-1 values and absence of invented uncertainties."""
    d = h.SWRU120_TABLE_1_1_MM
    horiz_sum = d["W1"] + d["L3"] + d["L4"] + d["L5"] + d["L2"]
    vert_sum = d["H5"] + d["H7"] + d["H4"] + d["H3"] + d["H2"]
    assert np.isclose(horiz_sum, d["L1"], atol=0.02)
    assert np.isclose(vert_sum, d["H1"], atol=1e-6)
    assert np.isclose(d["H6"], d["W1"], atol=1e-6)
    assert d["h_sub"] == 1.00

    b = h.SWRU120_BENCHMARKS
    assert b["footprint_mm"] == [25.7, 7.5]
    assert b["reflection_spec_db"] == -15.0
    assert b["bw_10db_min_hz"] == 300.0e6
    assert b["gain_xy_max_dbi"] == 1.1
    assert b["gain_xz_max_dbi"] == 3.3
    assert b["gain_yz_max_dbi"] == 1.6
    assert b["ground_size_mm_in_source"] is None
    assert b["efficiency_in_source"] is None
    assert all("uncertainty" not in k for k in b)
    assert len(h.SWRU120_SOURCE_LOCATIONS) >= 10


def test_dispersive_fr4_and_analytical_models():
    """Check dispersive FR-4 permittivity, quarter-wave resonance, and Chu-Wheeler bound."""
    freqs = np.array([2.0e9, 2.44e9, 2.8e9])
    eps_re, eps_im, tand = h.fr4_complex_permittivity(freqs)
    assert np.all((eps_re > 4.0) & (eps_re < 4.7))
    assert np.all(eps_im > 0.0)
    assert np.all((tand > 0.01) & (tand < 0.03))

    f0 = h.quarter_wave_resonance_hz(arm_length_m=21.0e-3, width_m=1.21e-3, height_m=1.0e-3)
    assert 2.15e9 < f0 < 2.65e9
    # Longer arm must lower the resonant frequency
    assert h.quarter_wave_resonance_hz(24.0e-3) < h.quarter_wave_resonance_hz(19.0e-3)

    # Tapping further from the shorting post steps up the input resistance
    r_near = h.ifa_input_resistance_tap(10.0, tap_distance_m=2.5e-3, total_arm_m=22.0e-3)
    r_nom = h.ifa_input_resistance_tap(10.0, tap_distance_m=4.8e-3, total_arm_m=22.0e-3)
    assert r_nom > r_near > 0.0

    # Chu-Wheeler Q decreases monotonically with electrical size ka
    q_small = h.chu_wheeler_q_bound(0.5, efficiency=0.9)
    q_large = h.chu_wheeler_q_bound(1.2, efficiency=0.9)
    assert q_small > q_large > 0.5
    fbw_small = h.chu_max_fractional_bandwidth(0.5, vswr_max=2.0, efficiency=0.9)
    fbw_large = h.chu_max_fractional_bandwidth(1.2, vswr_max=2.0, efficiency=0.9)
    assert 0.0 < fbw_small < fbw_large < 1.0


def test_preflight_and_project_grid_rules():
    """Verify wakis.estimate preflight and project grid rules (20 cpw and >=4 cells/feature)."""
    report = h.preflight_all_levels()
    for name, info in report.items():
        assert info["rule_20cpw_met"] is True, f"{name} violated 20 cpw in FR-4"
        assert info["min_cpw_fr4"] >= 20.0
        assert info["memory_mb"] > 400.0
        assert info["cpu_wall_est_s"] > info["gpu_wall_est_s"] > 0.0

    # Fine CPU mesh resolves the 1.00 mm substrate and 1.00-1.21 mm main traces with >= 4 cells
    fine = report["fine_0p25mm"]
    assert fine["cells_across_h_sub"] >= 4.0
    assert fine["cells_across_W1_1p21mm"] >= 4.0
    assert fine["cells_across_L5_1p00mm"] >= 4.0

    # Subfeature 0.18 mm and 0.1125 mm levels resolve W2=0.46 mm, H5=0.66 mm, and h=1.00 mm with >= 4 cells
    for lvl_name in ("subfeature_0p18mm", "gpu_subfeature_0p11mm"):
        lvl = report[lvl_name]
        assert lvl["cells_across_h_sub"] >= 4.0
        assert lvl["cells_across_W2_0p46mm"] >= 4
        assert lvl["cells_across_H5_0p66mm"] >= 4


def test_committed_data_resolution_study_and_figures():
    """Verify committed 5-level resolution data, continuum extrapolation, and figures vs SWRU120D."""
    data_dir = TUT_DIR / "data"
    fig_dir = TUT_DIR / "figures"

    for tname in h.ALL_TASKS:
        assert (data_dir / f"task_{tname}.json").exists(), f"Missing JSON for {tname}"
        assert (data_dir / f"task_{tname}.npz").exists(), f"Missing NPZ for {tname}"
        assert (data_dir / f"task_{tname}.npz").stat().st_size < 2 * 1024 * 1024

    base = json.loads((data_dir / "task_res_0p50mm_baseline.json").read_text())
    med = json.loads((data_dir / "task_res_0p35mm.json").read_text())
    fine = json.loads((data_dir / "task_res_0p25mm.json").read_text())
    sf18 = json.loads((data_dir / "task_res_0p18mm.json").read_text())
    sf11 = json.loads((data_dir / "task_gpu_subfeature_0p11mm.json").read_text())

    # Monotonic grid convergence from 0.50 mm to 0.1125 mm and Richardson extrapolation
    deltas = np.array([0.50, 0.35, 0.25, 0.18, 0.1125])
    f_arr = np.array([base["f_res_ghz"], med["f_res_ghz"], fine["f_res_ghz"], sf18["f_res_ghz"], sf11["f_res_ghz"]])
    s11_arr = np.array([base["s11_min_db"], med["s11_min_db"], fine["s11_min_db"], sf18["s11_min_db"], sf11["s11_min_db"]])
    bw_arr = np.array([base["bw_10db_mhz"], med["bw_10db_mhz"], fine["bw_10db_mhz"], sf18["bw_10db_mhz"], sf11["bw_10db_mhz"]])

    assert np.all(np.diff(f_arr) <= 1e-6)
    f_inf, f_slope, _ = h.richardson_extrapolate_ifa(deltas, f_arr)
    s11_inf, _, _ = h.richardson_extrapolate_ifa(deltas, s11_arr)
    bw_inf, _, _ = h.richardson_extrapolate_ifa(deltas, bw_arr)

    # Tolerance of 55 MHz justified by the 49 MHz grid shift (0.50 -> 0 mm) and 37.5 MHz FR-4 eps_r shift (4.28 -> 4.00)
    assert abs(sf11["f_res_ghz"] - h.SWRU120_BENCHMARKS["f_res_approx_hz"] / 1e9) <= 0.055
    assert abs(f_inf - h.SWRU120_BENCHMARKS["f_res_approx_hz"] / 1e9) <= 0.055
    # SWRU120D Table 3-1 requires reflection < -15 dB
    assert sf11["s11_min_db"] < h.SWRU120_BENCHMARKS["reflection_spec_db"]
    assert s11_inf < h.SWRU120_BENCHMARKS["reflection_spec_db"]
    assert 260.0 <= sf11["bw_10db_mhz"] <= 340.0
    assert 260.0 <= bw_inf <= 340.0
    assert np.isclose(sf11["p_flux_over_p_rad"], 1.0, atol=0.02)

    # Principal-plane peak gains vs SWRU120D Table 3-1 (XZ=3.3, YZ=1.6, XY=1.1 dBi; XZ > XY and XZ > YZ)
    assert sf11["g_xz_peak_dbi"] > sf11["g_xy_peak_dbi"] > 1.0
    assert sf11["g_xz_peak_dbi"] > sf11["g_yz_peak_dbi"] > 1.0
    assert abs(sf11["g_xz_peak_dbi"] - h.SWRU120_BENCHMARKS["gain_xz_max_dbi"]) <= 1.2
    assert abs(sf11["g_yz_peak_dbi"] - h.SWRU120_BENCHMARKS["gain_yz_max_dbi"]) <= 0.6
    assert abs(sf11["g_xy_peak_dbi"] - h.SWRU120_BENCHMARKS["gain_xy_max_dbi"]) <= 1.0

    # Controlled one-at-a-time hypothesis study verification
    hyp = {row["label"]: row for row in json.loads((data_dir / "task_hypothesis_study.json").read_text())}
    assert hyp["er_4p00_low_fr4"]["f_res_ghz"] > hyp["baseline_0p35mm_er4p28_gnd30x32_W2_0p46_gap0p66"]["f_res_ghz"] > hyp["er_4p50_high_fr4"]["f_res_ghz"]
    assert hyp["er_4p00_low_fr4"]["bw_10db_mhz"] >= 300.0
    assert hyp["er_4p00_low_fr4"]["s11_min_db"] < -15.0
    assert hyp["ground_37x45mm"]["rin_ohm"] > hyp["baseline_0p35mm_er4p28_gnd30x32_W2_0p46_gap0p66"]["rin_ohm"]

    expected_figs = [
        "01_geometry_3d_views.png",
        "02_mesh_and_cpw_views.png",
        "03_near_fields_and_currents.png",
        "04_radiated_field_evolution.gif",
        "05_far_field_3d_and_cuts.png",
        "06_combined_3d_overlays.png",
        "07_s11_impedance_smith.png",
        "08_resolution_and_param_study.png",
    ]
    for fname in expected_figs:
        fpath = fig_dir / fname
        assert fpath.exists(), f"Missing figure {fname}"
        assert fpath.stat().st_size > 10_000
    assert (fig_dir / "04_radiated_field_evolution.gif").stat().st_size < 5 * 1024 * 1024


def test_live_3d_fit_ifa_simulation_and_mutation():
    """Run a fast 3D FIT simulation of the PCB IFA and verify that physical mutations fail."""
    res_nom = h.run_ifa_simulation(
        trace_cell_mm=0.65,
        sub_cells_z=2,
        sim_time_ns=4.2,
        energy_decay=2e-4,
    )
    assert 2.28e9 <= res_nom["f_res_hz"] <= 2.58e9
    assert res_nom["s11_min_db"] < -10.0
    assert np.all(res_nom["s11_db"] <= 0.05)  # Passivity
    assert 0.5 < res_nom["eta_rad"][1] <= 1.0

    # Mutation check 1: halving the radiating arm length L2 from 16.40 mm to 7.50 mm
    # must detune the quarter-wave resonance out of the 2.4 GHz band
    res_short = h.run_ifa_simulation(
        trace_cell_mm=0.65,
        sub_cells_z=2,
        arm_length_l2_mm=7.50,
        sim_time_ns=4.2,
        energy_decay=2e-4,
    )
    with pytest.raises(AssertionError):
        assert 2.35e9 <= res_short["f_res_hz"] <= 2.52e9

    # Mutation check 2: corrupting the Chu-Wheeler Q formula by dropping 1/(ka)^3
    ka_test = 0.5
    q_true = h.chu_wheeler_q_bound(ka_test, efficiency=1.0)
    q_mutated = 1.0 / ka_test  # Missing 1/(ka)^3 reactive energy term
    with pytest.raises(AssertionError):
        assert np.isclose(q_mutated, q_true, rtol=0.1)

    # Mutation check 3: invalid non-positive electrical size must raise ValueError
    with pytest.raises(ValueError):
        h.chu_wheeler_q_bound(-0.3)

    # Mutation check 4: Richardson extrapolation on synthetic q(d) = 2.40 + 0.4 * d^1.5
    d_syn = np.array([0.50, 0.35, 0.25, 0.18, 0.1125])
    q_syn = 2.40 + 0.40 * (d_syn**1.5)
    q_inf_est, slope_est, _ = h.richardson_extrapolate_ifa(d_syn, q_syn, order=1.5)
    assert np.isclose(q_inf_est, 2.40, atol=1e-6)
    assert np.isclose(slope_est, 0.40, atol=1e-6)
    with pytest.raises(AssertionError):
        # Taking the unextrapolated coarse value instead of q_inf must fail
        assert np.isclose(q_syn[0], 2.40, atol=0.02)
    with pytest.raises(ValueError):
        h.richardson_extrapolate_ifa([0.5], [2.4])
