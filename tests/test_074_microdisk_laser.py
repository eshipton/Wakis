"""End-to-end and analytical validation tests for Tutorial 7.8 (Microdisk WGM Laser).

Covers:
1. Exact 2D Bessel/Hankel complex WGM roots (TM and TE, azimuthal orders m=4..9,
   radial orders q=1,2), exponential radiative Q_rad(m) scaling, and characteristic
   equation sign/polarization mutation checks.
2. Quasi-normal mode (QNM) confinement factor Gamma_{m,q}^{QNM} and radially truncated
   energy-integral confinement Gamma_{m,q}^{trunc}(1.45 R) (disk, ring, core) and
   the derived linear gain threshold formula g_th = n_disk * omega / (c * Gamma * Q_cold)
   compared against both FIT Q_cold and exact Bessel Q_rad, with an unconfined (Gamma=1)
   mutation check.
3. FourLevelGain steady-state L-pump curve with 2D standing-wave spatial hole burning
   and documented population time scaling (s = 100 <= 100), plus self-terminating
   bottleneck (tau_10 >= tau_21) and sub-threshold pump mutation checks.
4. Live 2D FIT cold-cavity WGM extraction and conformal subpixel vs staircase accuracy,
   with a disk-index mutation check.
5. Live 2D FIT active FourLevelGain lasing from spontaneous-emission noise
   (spontaneous_emission=True, time_scale=100.0) run to clamping vs below-threshold,
   core-pump, and oscillator coupling kappa_a mutations.
6. Quantitative Summary table in microdisk_laser.py / microdisk_laser.ipynb verified
   against data/task_0{1,2,3}_*.json and absence of the stale F11b placeholder hook.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

TUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "07_light_sources"
    / "07-08_microdisk_laser"
)
_SPEC = importlib.util.spec_from_file_location("_helpers_074_microdisk", TUT_DIR / "_helpers.py")
assert _SPEC is not None and _SPEC.loader is not None
h = importlib.util.module_from_spec(_SPEC)
sys.modules["_helpers_074_microdisk"] = h
_SPEC.loader.exec_module(h)


def _verify_summary_table_against_json(
    py_text: str,
    cold_data: dict,
    th_data: dict,
    las_data: dict,
) -> None:
    """Assert that every numerical entry in the Quantitative Summary table matches data/*.json."""
    tm_modes = {int(r["m"]): r for r in cold_data["tm_q1_modes"]}
    tm_q2_modes = {int(r["m"]): r for r in cold_data["tm_q2_modes"]}
    te_modes = {int(r["m"]): r for r in cold_data["te_q1_modes"]}
    th_modes = {int(r["m"]): r for r in th_data["modes"]}

    expected_tokens = [
        # Cold-cavity TM_(4,1), TM_(5,1), TM_(6,1), TM_(8,1), TM_(9,1), TM_(6,2)
        f"{tm_modes[4]['f_ana_thz']:.3f}",
        f"{tm_modes[4]['f_sub_thz']:.3f}",
        f"{tm_modes[4]['q_ana']:.2f}",
        f"{tm_modes[4]['q_sub']:.2f}",
        f"{tm_modes[5]['f_ana_thz']:.3f}",
        f"{tm_modes[5]['f_sub_thz']:.3f}",
        f"{tm_modes[6]['f_ana_thz']:.3f}",
        f"{tm_modes[6]['f_sub_thz']:.3f}",
        f"{cold_data['f6_rich_thz']:.3f}",
        f"{cold_data['f6_rich_rel_err'] * 100:+.4f}%",
        f"{tm_modes[6]['q_ana']:.2f}",
        f"{tm_modes[6]['q_sub']:.2f}",
        f"{tm_modes[6]['q_stair']:.2f}",
        f"{tm_modes[8]['q_ana']:.2f}",
        f"{tm_modes[8]['q_sub']:.2f}",
        f"{cold_data['q8_rich_rel_err'] * 100:+.4f}%",
        f"{cold_data['q8_rich_uncert_rel'] * 100:.2f}%",
        f"{tm_modes[9]['f_ana_thz']:.3f}",
        f"{tm_modes[9]['f_sub_thz']:.3f}",
        f"{tm_modes[9]['q_ana']:.2f}",
        f"{tm_modes[9]['q_sub']:.2f}",
        f"{tm_q2_modes[6]['f_ana_thz']:.3f}",
        f"{tm_q2_modes[6]['f_sub_thz']:.3f}",
        f"{tm_q2_modes[6]['q_ana']:.2f}",
        f"{tm_q2_modes[6]['q_sub']:.2f}",
        f"{tm_q2_modes[6]['f_sub_rel_err'] * 100:+.3f}%",
        f"{tm_q2_modes[6]['q_sub_rel_err'] * 100:+.2f}%",
        f"{te_modes[6]['q_ana']:.2f}",
        f"{te_modes[6]['q_sub']:.2f}",
        f"{te_modes[8]['q_ana']:.2f}",
        f"{te_modes[8]['q_sub']:.2f}",
        # Part 2 threshold gains m=4..8 (both vs Bessel and vs FIT Q_cold)
        f"{th_modes[6]['confinement_disk']:.4f}",
        f"{th_modes[6]['confinement_disk_trunc']:.4f}",
        *[f"{th_modes[m]['g_th_bessel_cm1']:.1f}" for m in (4, 5, 6, 7, 8)],
        *[f"{th_modes[m]['g_th_pred_cm1']:.1f}" for m in (4, 5, 6, 7, 8)],
        *[f"{th_modes[m]['g_th_fit_cm1']:.1f}" for m in (4, 5, 6, 7, 8)],
        *[f"{th_modes[m]['rel_err_vs_bessel'] * 100:+.2f}%" for m in (4, 5, 6, 7, 8)],
        *[f"{th_modes[m]['rel_err_vs_pred'] * 100:+.2f}%" for m in (4, 5, 6, 7, 8)],
        # Part 3 active lasing from noise, clamping, L-pump vs theory & mutations
        f"{las_data['dx_nm']:.0f}",
        f"{las_data['cpw_m7']:.2f}",
        f"{las_data['above_shb_post_th_tau21_eff']:.2f}",
        f"{las_data['f_lasing_sim_thz']:.3f}",
        f"{las_data['f_lasing_rel_err'] * 100:+.3f}%",
        f"{las_data['above_th_late_clamping_ratio']:.3f}",
        f"{las_data['smsr_tm6_over_tm7_db']:.2f}",
        f"{las_data['li_dn_final_over_th'][-1]:.3f}",
        f"{las_data['li_dn_shb_modal_final_over_th'][-1]:.3f}",
        *[f"{err * 100:+.2f}%" for err in las_data["li_hom_vs_theory_rel_err_above"]],
        *[f"{err * 100:+.2f}%" for err in las_data["li_shb_vs_theory_rel_err_above"]],
        f"{las_data['shb_reduction_pct_r240']:.2f}%",
        f"{las_data['shb_theory_reduction_pct_r240']:.2f}%",
        f"{las_data['shb_sw_theory_reduction_pct_r240']:.2f}%",
        f"{las_data['profile_selection']['ring_area_saving_pct']:.2f}%",
        f"{las_data['profile_selection']['ring_confinement_retention_pct']:.2f}%",
        f"{las_data['profile_selection']['ring_to_core_intensity_ratio']:.2f}",
        f"{las_data['above_over_below_intensity_ratio']:.2f}",
    ]

    match = re.search(
        r"## 4\. Quantitative Summary Table(.*?)## 5\. Limits & Failure Modes",
        py_text,
        flags=re.DOTALL,
    )
    assert match is not None, "Could not locate Section 4 Quantitative Summary Table"
    table_block = match.group(1).replace(r"\%", "%")
    for token in expected_tokens:
        assert token in table_block, (
            f"Expected JSON-derived value '{token}' missing from Quantitative Summary Table"
        )


def test_bessel_hankel_wgm_roots_and_exponential_q():
    """Verify exact 2D Bessel/Hankel complex WGM roots, exponential Q(m), and residual mutation."""
    q_tm = []
    q_te = []
    for m_val in (4, 5, 6, 7, 8, 9):
        rtm = h.solve_wgm_complex_root(m_val, 1, polarization="TM")
        rte = h.solve_wgm_complex_root(m_val, 1, polarization="TE")
        assert rtm["residual_abs"] < 1e-9
        assert rte["residual_abs"] < 1e-9
        assert rtm["x_imag"] > 0.0
        assert rtm["q_rad"] > rte["q_rad"], (
            f"Expected TM Q_rad > TE Q_rad at m={m_val} due to weaker TE boundary impedance factor"
        )
        q_tm.append(rtm["q_rad"])
        q_te.append(rte["q_rad"])

    # Radiative Q must increase strictly and exponentially with m
    assert np.all(np.diff(q_tm) > 0.0)
    assert np.all(np.diff(q_te) > 0.0)
    assert q_tm[-1] / q_tm[0] > 20.0

    # Fundamental radial order q=1 has higher Q than second radial order q=2
    r61 = h.solve_wgm_complex_root(6, 1, polarization="TM")
    r62 = h.solve_wgm_complex_root(6, 2, polarization="TM")
    assert r61["q_rad"] > 1.2 * r62["q_rad"]
    assert r62["frequency_thz"] > r61["frequency_thz"]

    # Mutation check: evaluate TE residual at the TM root (wrong boundary factor 1/n^2 vs 1)
    z_tm61 = complex(r61["x_real"], -r61["x_imag"])
    res_exact = abs(h.wgm_characteristic_residual(z_tm61, 6, 2.0, 1.0, "TM"))
    res_mutated = abs(h.wgm_characteristic_residual(z_tm61, 6, 2.0, 1.0, "TE"))
    assert res_exact < 1e-9
    with pytest.raises(AssertionError):
        assert res_mutated < 1e-9, "Mutated polarization boundary condition must fail root check"


def test_wgm_confinement_and_threshold_gain_formula():
    """Verify QNM and truncated confinement factors and g_th = n*omega/(c*Gamma*Q)."""
    r61 = h.solve_wgm_complex_root(6, 1, polarization="TM")
    r62 = h.solve_wgm_complex_root(6, 2, polarization="TM")

    # Partition of unity inside the disk: ring + core == disk (within radial quadrature spacing)
    assert np.isclose(
        r61["confinement_ring"] + r61["confinement_core"],
        r61["confinement_disk"],
        rtol=2e-3,
    )
    assert 0.85 < r61["confinement_disk_trunc"] < r61["confinement_disk"] < 0.92
    # Fundamental radial mode is >99% confined to outer ring r in [0.55 R, R], whereas q=2 has >15x more core energy
    assert r61["confinement_ring"] / r61["confinement_disk"] > 0.985
    assert r61["confinement_core"] / r61["confinement_disk"] < 0.015
    assert r62["confinement_core"] / r62["confinement_disk"] > 0.12
    assert r62["confinement_core"] > 15.0 * r61["confinement_core"]

    # Verify threshold formula against Task 2 FIT results across m = 4..8
    th_data = json.loads((h.DATA_DIR / "task_02_threshold.json").read_text(encoding="utf-8"))
    for row in th_data["modes"]:
        assert abs(row["rel_err_vs_pred"]) < 0.01, (
            f"Confined threshold formula must match 2D FIT within 1% at m={row['m']}"
        )
        assert abs(row["rel_err_vs_bessel"]) < 0.035, (
            f"Confined threshold formula must match exact Bessel g_th within 3.5% at m={row['m']}"
        )
        # Mutation check: dropping Gamma_{m,1} (setting Gamma=1) underpredicts g_th by >8%
        with pytest.raises(AssertionError):
            assert abs(row["rel_err_unconfined_vs_fit"]) < 0.01


def test_four_level_li_curve_shb_and_lifetime_scaling():
    """Verify 4-level L-pump curve, 2D spatial hole burning reduction, and s=100 time scaling."""
    r61 = h.solve_wgm_complex_root(6, 1, polarization="TM")
    omega_a = 2.0 * np.pi * r61["frequency_hz"]
    pump_ratios = np.array([0.0, 0.5, 0.95, 1.0, 1.3, 1.8, 2.4])
    li = h.wgm_four_level_li_curve(
        pump_ratios=pump_ratios,
        g_th_1_m=r61["g_th_ring_1_m"],
        omega_a=omega_a,
    )

    # Zero below/at threshold, strictly increasing above threshold;
    # ring-averaged SHB intensity s_rad_ring < s_hom due to 2m azimuthal spatial holes
    assert np.allclose(li["s_hom"][pump_ratios <= 1.0], 0.0, atol=1e-12)
    assert np.allclose(li["s_rad_ring"][pump_ratios <= 1.0], 0.0, atol=1e-12)
    assert np.all(np.diff(li["s_hom"][pump_ratios >= 1.0]) > 0.0)
    assert np.all(np.diff(li["s_rad_ring"][pump_ratios >= 1.0]) > 0.0)
    assert np.all(li["s_rad_ring"][pump_ratios > 1.0] < li["s_hom"][pump_ratios > 1.0])

    # Time scaling laws for s = 100x <= 100 and Class-B -> Class-A RO crossover
    sc = h.lifetime_scaling_laws(scale_factor=100.0)
    assert sc["scale_factor"] <= 100.0
    assert np.isclose(sc["sigma_ratio_fixed_kappa"], 1.0, rtol=1e-12)
    assert np.isclose(sc["dN_th_ratio_fixed_kappa"], 1.0, rtol=1e-12)
    assert np.isclose(sc["W_p_th_ratio_fixed_kappa"], 100.0, rtol=1e-12)
    assert np.isclose(sc["I_sat_ratio_fixed_kappa"], 100.0, rtol=1e-12)
    assert np.isclose(sc["f_ro_ratio"], np.sqrt(100.0), rtol=1e-12)
    assert sc["underdamped_phys"] is True
    assert sc["overdamped_bare_scaled"] is True
    assert sc["gamma_ro_bare_scaled_1_s"] > sc["omega_ro_bare_scaled_1_s"]
    ts_lim = sc["time_scaling_limits"]
    assert ts_lim["sat_rate_over_kappa_cav"] < 1.0
    assert ts_lim["rate_21_over_gamma_a"] < 1.0
    assert ts_lim["max_rate_over_gamma_a"] > 1.0
    assert ts_lim["max_rate_dt"] < 0.1

    # Mutation 1: Self-terminating bottleneck (tau_10 = 2 * tau_21 >= tau_21) cannot invert
    with pytest.raises(ValueError, match="Population inversion requires tau_21 > tau_10"):
        h.wgm_four_level_li_curve(
            pump_ratios=np.array([2.0]),
            g_th_1_m=r61["g_th_ring_1_m"],
            omega_a=omega_a,
            tau_21=150e-15,
            tau_10=300e-15,
        )

    # Mutation 2: Pumping below threshold (r_p = 2.4 / s < 1) produces zero steady-state lasing
    li_sub_pump = h.wgm_four_level_li_curve(
        pump_ratios=np.array([2.4 / sc["W_p_th_ratio_fixed_kappa"]]),
        g_th_1_m=r61["g_th_ring_1_m"],
        omega_a=omega_a,
    )
    with pytest.raises(AssertionError):
        assert li_sub_pump["s_rad_ring"][0] > 0.0


def test_fit_cold_cavity_and_subpixel_vs_staircase_quick():
    """Run live 2D FIT cold-cavity WGM extraction (quick grid) and subpixel vs staircase check."""
    res_sub = h.run_cold_cavity_harminv(
        dx_m=0.040e-6,
        polarization="TM",
        subpixel=True,
        m_orders=(6,),
        q_orders=(1,),
        t_after_s=180.0e-15,
        record_dft_m6=False,
    )
    res_stair = h.run_cold_cavity_harminv(
        dx_m=0.040e-6,
        polarization="TM",
        subpixel=False,
        m_orders=(6,),
        q_orders=(1,),
        t_after_s=180.0e-15,
        record_dft_m6=False,
    )
    m6_sub = res_sub["matched"][(6, 1)]
    m6_stair = res_stair["matched"][(6, 1)]

    assert abs(m6_sub["f_rel_err"]) < 0.006
    assert abs(m6_sub["f_rel_err"]) < abs(m6_stair["f_rel_err"]), (
        "Subpixel permittivity smoothing must beat staircase frequency accuracy"
    )

    # Mutation check: solving with a corrupted refractive index n_disk = 1.65 must fail frequency match
    r61_mut = h.solve_wgm_complex_root(6, 1, n_disk=1.65, polarization="TM")
    with pytest.raises(AssertionError):
        err_mut = abs(m6_sub["f_sim_thz"] - r61_mut["frequency_thz"]) / r61_mut["frequency_thz"]
        assert err_mut < 0.01


def test_fit_active_lasing_and_mutations_quick():
    """Run live 2D FIT FourLevelGain lasing from noise vs below-threshold, core-pump, and kappa_a mutations."""
    r61_ana = h.solve_wgm_complex_root(6, 1, polarization="TM")

    run_above = h.run_single_four_level_microdisk(
        pump_ratio=2.40,
        pump_profile="ring",
        kappa_scale=1.0,
        with_bus=False,
        spatial_averaging=True,
        spontaneous_emission=True,
        time_scale=100.0,
        dx_m=0.040e-6,
        n_steps=8000,
    )
    run_below = h.run_single_four_level_microdisk(
        pump_ratio=0.50,
        pump_profile="ring",
        kappa_scale=1.0,
        with_bus=False,
        spatial_averaging=True,
        spontaneous_emission=True,
        time_scale=100.0,
        dx_m=0.040e-6,
        n_steps=8000,
    )
    run_core = h.run_single_four_level_microdisk(
        pump_ratio=2.05,
        pump_profile="core",
        kappa_scale=1.0,
        with_bus=False,
        spatial_averaging=True,
        spontaneous_emission=True,
        time_scale=100.0,
        dx_m=0.040e-6,
        n_steps=8000,
    )
    run_kappa_low = h.run_single_four_level_microdisk(
        pump_ratio=1.45,
        pump_profile="ring",
        kappa_scale=0.5,
        with_bus=False,
        spatial_averaging=True,
        spontaneous_emission=True,
        time_scale=100.0,
        dx_m=0.040e-6,
        n_steps=8000,
    )

    assert run_above["spontaneous_emission"] is True
    assert run_above["time_scale"] <= 100.0
    assert run_above["buildup_gain_ratio"] > 4.0, (
        "Above-threshold ring pump must amplify spontaneous-emission noise to clamping"
    )
    assert 0.85 <= run_above["dN_final_over_th"] <= 1.18, (
        "Above-threshold population inversion must clamp near dN_th"
    )
    assert abs(run_above["f_lasing_thz"] - r61_ana["frequency_thz"]) / r61_ana["frequency_thz"] < 0.01
    assert run_above["I_cav_late_w_m2"] > 12.0 * run_below["I_cav_late_w_m2"]

    # Mutation 1: Pump below threshold -> inversion stays below threshold and intensity is suppressed >12x
    assert run_below["dN_final_over_th"] < 0.60
    with pytest.raises(AssertionError):
        assert run_below["I_cav_late_w_m2"] > 0.2 * run_above["I_cav_late_w_m2"]

    # Mutation 2: Core pump (r < 0.55 R) has <1% overlap with TM_(6,1) -> fails to lase into WGM
    assert run_above["I_cav_late_w_m2"] > 25.0 * run_core["I_cav_late_w_m2"]
    with pytest.raises(AssertionError):
        assert run_core["I_cav_late_w_m2"] > 0.2 * run_above["I_cav_late_w_m2"]

    # Mutation 3: Halving oscillator coupling (0.5 * kappa_a) doubles dN_th (r_eff = 0.725 < 1) -> unclamped
    assert run_kappa_low["dN_final_over_th"] > 1.18
    with pytest.raises(AssertionError):
        assert run_kappa_low["dN_final_over_th"] <= 1.12


def test_committed_json_data_and_quantitative_summary_table():
    """Verify Quantitative Summary table matches data/*.json and F11b noise startup is active."""
    cold_data = json.loads((h.DATA_DIR / "task_01_cold_cavity.json").read_text(encoding="utf-8"))
    th_data = json.loads((h.DATA_DIR / "task_02_threshold.json").read_text(encoding="utf-8"))
    las_data = json.loads((h.DATA_DIR / "task_03_lasing.json").read_text(encoding="utf-8"))

    py_path = TUT_DIR / "microdisk_laser.py"
    ipynb_path = TUT_DIR / "microdisk_laser.ipynb"
    helpers_path = TUT_DIR / "_helpers.py"

    py_text = py_path.read_text(encoding="utf-8")
    helpers_text = helpers_path.read_text(encoding="utf-8")

    # Stale F11b placeholder hook must be replaced by real spontaneous_emission=True lasing from noise
    stale_hook = "lasing from noise — to be added with F11b"
    assert stale_hook not in helpers_text
    assert stale_hook not in py_text
    assert "spontaneous_emission=True" in helpers_text
    assert "spontaneous_emission=True" in py_text
    assert las_data["spontaneous_emission"] is True
    assert 1.0 <= las_data["time_scale"] <= 100.0
    assert las_data["cpw_m6"] >= 20.0
    assert las_data["cpw_m7"] >= 20.0
    assert las_data["above_shb_post_th_tau21_eff"] >= 10.0
    assert las_data["above_hom_post_th_tau21_eff"] >= 10.0
    assert 0.85 <= las_data["li_dn_final_over_th"][-1] <= 1.12
    assert 0.85 <= las_data["li_dn_shb_modal_final_over_th"][-1] <= 1.12
    assert "347\\;\\mathrm{fs}" not in py_text
    assert ipynb_path.exists(), "microdisk_laser.ipynb must be built and present"

    # Quantitative Summary table must match all JSON numbers
    _verify_summary_table_against_json(py_text, cold_data, th_data, las_data)

    # Mutation check: perturbing a JSON result must cause the table consistency check to fail
    th_mutated = json.loads(json.dumps(th_data))
    th_mutated["modes"][2]["g_th_fit_cm1"] *= 1.05
    with pytest.raises(AssertionError, match="missing from Quantitative Summary Table"):
        _verify_summary_table_against_json(py_text, cold_data, th_mutated, las_data)
