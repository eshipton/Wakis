"""Tests for Tutorial 8.3 extension: 'A Maxwellian Perspective on Particle Acceleration' (job gem-z83basics).

Covers:
1. Part 0.1: Static-gap 2D (r, z) Gauss-Legendre quadrature U_int = -e E0 d (< 1e-6 rel error),
   closed-loop electrostatic zero gain, and image-charge sign/removal mutations.
2. Part 0.2: 3D Wakis FIT pillbox cavity with beam pipes — linear superposition U_int = U_iii - U_i - U_ii,
   after-exit identity -(U_int + Phi_int) = q V_d T F_b cos(phi), E0 linearity, WakeSolver k_|| agreement,
   Eq. (9) O(1) prefactor, and transit-time / phase-shift / flux-omission mutations.
3. Part A additions: Relativistic plane-wave cycle-averaged drift beta_|| (Eq. 15), effective mass
   m_* = m_e sqrt(1 + eta^2) (Eq. 17), exact vs Eq. (16) transverse amplitude r_perp, and weak-field
   ponderomotive potential m_e c^2 eta^2 / 2 (Eq. 19) for gamma_0 in {1, 5} and a_0 in {0.3, 1, 2.5, 5},
   plus Doppler-factor and polarization-convention mutations.
4. Part D additions: Formation length L_f (Eq. 36), snowplow reflection threshold (Eq. 45),
   multiplicative gain Delta U = C gamma_0 eta^2 m_e c^2 (Eqs. 47-49) across gamma_0 in {1..20},
   reconciliation with gamma_0 = 1, Barton & Alexander (1989) 5th-order paraxial check (kw0 >= 10 vs kw0 = 4),
   laser energetics ladder (Eq. 50), bunch energy spread preservation and transverse kick (Sec. 5.2.5),
   plus no-diffraction and sub-threshold mutations.
5. Quantitative Summary table vs JSON verification (and table-tampering mutation check).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import e as q_e

ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = ROOT / "tutorials" / "08_vacuum_electronics" / "08-03_vacuum_laser_acceleration"
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _maxwell_basics as mb  # noqa: E402


def test_part0_static_gap_interference_quadrature_and_mutations():
    """Verify U_int = -e E0 d to < 1e-6 by 2D (r,z) quadrature and check image-charge mutations fail."""
    e0_v_m = 2.0e6
    d_values_m = [0.5e-3, 1.0e-3, 2.0e-3, 5.0e-3, 10.0e-3]
    for d_m in d_values_m:
        res = mb.static_gap_interference_quadrature(d_m, e0_v_m=e0_v_m, n_quad=64, image_sign=-1.0)
        assert res["rel_error"] < 1e-6
        assert res["u_int_quad_ev"] == pytest.approx(-e0_v_m * d_m, rel=1e-6)

    # Closed-loop electrostatic integral vanishes identically (battery remark, Paper Sec. 2)
    sg_json = mb.DATA_DIR / "task_part0_static_gap.json"
    sg_npz = mb.DATA_DIR / "task_part0_static_gap.npz"
    saved_sg_json = sg_json.read_bytes() if sg_json.exists() else None
    saved_sg_npz = sg_npz.read_bytes() if sg_npz.exists() else None
    try:
        res_01 = mb.run_task_part0_static_gap(quick=True)
    finally:
        if saved_sg_json is not None:
            sg_json.write_bytes(saved_sg_json)
        if saved_sg_npz is not None:
            sg_npz.write_bytes(saved_sg_npz)
    assert res_01["max_rel_error"] < 1e-6
    assert abs(res_01["closed_loop_electrostatic_gain_ev"]) == 0.0

    # Mutation 1: Remove image charge (image_sign = 0.0) -> z > d flux no longer cancels -> fails Eq. (1)
    mut_no_img = mb.static_gap_interference_quadrature(
        2.0e-3, e0_v_m=e0_v_m, n_quad=64, image_sign=0.0
    )
    assert mut_no_img["rel_error"] > 0.45

    # Mutation 2: Wrong-sign image charge (image_sign = +1.0) -> z < d slab cancels instead -> fails Eq. (1)
    mut_pos_img = mb.static_gap_interference_quadrature(
        2.0e-3, e0_v_m=e0_v_m, n_quad=64, image_sign=+1.0
    )
    assert mut_pos_img["rel_error"] > 0.90


def test_part0_cavity_3d_fit_interference_wakesolver_and_mutations():
    """Verify 3D FIT pillbox cavity U_int = -q V_d T F_b cos(phi), E0 linearity, WakeSolver k_||, and mutations."""
    res_02 = json.loads((mb.DATA_DIR / "task_part0_cavity.json").read_text(encoding="utf-8"))

    # Quality rules: >= 20 cells/wavelength and >= 4 cells across smallest feature (aperture radius & gap)
    assert res_02["cells_per_wavelength"] >= 20.0
    assert res_02["cells_across_aperture_radius"] >= 4.0
    assert res_02["cells_across_gap"] >= 4.0

    # Discrete leapfrog energy conservation -(U_int + Phi_int) == int J_b . E_d dV dt
    assert res_02["energy_balance_rel_error"] < 1e-12
    # After bunch exit: -U_int matches q V_d T F_b cos(phi) and linear E0 scaling
    assert res_02["phi_scan_max_rel_error"] < 2e-3
    assert res_02["e0_scan_max_rel_error"] < 2e-3
    assert res_02["single_electron_gain_ev"] == pytest.approx(
        res_02["single_electron_gain_th_ev"], rel=1e-3
    )

    # Independent physical validation: wake mode projection onto TM010 agrees with analytical k_loss,010 and Eq. (9) O(1) factor
    assert res_02["k_loss_tm010_rel_error"] < 1e-2
    # Internal discrete energy-balance consistency check: (U_ii + Phi_ii)/q^2 == -W_joule,ii/q^2
    assert res_02["k_loss_total_fit_v_pc"] == pytest.approx(
        res_02["k_loss_wakesolver_v_pc"], rel=1e-5
    )
    assert 1.5 <= res_02["o1_prefactor_eq9"] <= 3.0

    # Mutation 1: Drop the transit-time factor T (set T = 1.0 instead of T = 0.8438) -> prediction fails by > 15%
    t_ttf = float(res_02["transit_time_factor_t"])
    mut_no_ttf_gain_uj = float(res_02["minus_u_int_on_crest_th_uj"]) / t_ttf
    rel_err_no_ttf = abs(mut_no_ttf_gain_uj - float(res_02["minus_u_int_on_crest_uj"])) / float(
        res_02["minus_u_int_on_crest_uj"]
    )
    assert rel_err_no_ttf > 0.15

    # Mutation 2: Shift drive phase by pi/2 (quadrature zero-crossing) -> net gain collapses to ~0
    d_npz = np.load(mb.DATA_DIR / "task_part0_cavity.npz")
    phi_grid = d_npz["phi_scan_rad"]
    minus_u_phi = d_npz["minus_u_int_phi_fit_j"] * 1e6
    idx_quad = int(np.argmin(np.abs(phi_grid - 0.5 * np.pi)))
    assert abs(float(minus_u_phi[idx_quad])) < 0.02 * float(res_02["minus_u_int_on_crest_uj"])


def test_part_a_plane_wave_invariants_and_mutations():
    """Verify plane-wave beta_|| (Eq. 15), m_* (Eq. 17), r_perp (Eq. 16), and U_pond (Eq. 19) for gamma_0 in {1,5}."""
    res_a = json.loads((mb.DATA_DIR / "task_part0_a_plane_wave.json").read_text(encoding="utf-8"))
    assert res_a["max_beta_parallel_rel_err"] < 1e-3
    assert res_a["max_m_star_rel_err"] < 1e-2
    assert res_a["max_r_perp_rel_err"] < 2e-3
    # Weak-field ponderomotive potential m_e c^2 eta^2 / 2 matches exact (sqrt(1+eta^2)-1) m_e c^2 at a_0 = 0.3
    assert res_a["weak_field_a0_0p3_rel_diff"] < 0.015

    # Direct Vay/Boris simulation check on (a_0 = 2.5, gamma_0 = 1 and 5)
    for g0 in [1.0, 5.0]:
        th = mb.plane_wave_maxwell_invariants_theory(2.5, gamma0=g0)
        sim = mb.simulate_plane_wave_invariants_vay_boris(2.5, gamma0=g0)
        assert sim["beta_parallel_vay"] == pytest.approx(th["beta_parallel"], rel=5e-4)
        assert sim["m_star_over_me_vay"] == pytest.approx(th["m_star_over_me"], rel=5e-3)
        assert sim["r_perp_sim_over_lambda"] == pytest.approx(
            th["r_perp_exact_over_lambda"], rel=1e-3
        )

    # Check both at-rest (gamma_0 = 1: beta_|| = a_0^2 / (4 + a_0^2)) and copropagating (gamma_0 = 5)
    for a0 in [0.3, 1.0, 2.5, 5.0]:
        th1 = mb.plane_wave_maxwell_invariants_theory(a0, gamma0=1.0)
        assert th1["beta_parallel"] == pytest.approx((a0**2) / (4.0 + a0**2), rel=1e-14)
        assert th1["m_star_over_me"] == pytest.approx(np.sqrt(1.0 + 0.5 * a0**2), rel=1e-14)
        assert th1["r_perp_exact_to_eq16_ratio"] == pytest.approx(
            np.sqrt(2.0) / (2.0 * np.pi), rel=1e-14
        )

    # Mutation 1: Drop the Doppler factor (1 + beta_0) at gamma_0 = 5 -> fails simulated Vay drift & r_perp
    th5_exact = mb.plane_wave_maxwell_invariants_theory(
        2.5, gamma0=5.0, drop_doppler_factor=False
    )
    th5_mut = mb.plane_wave_maxwell_invariants_theory(2.5, gamma0=5.0, drop_doppler_factor=True)
    assert (
        abs(th5_mut["r_perp_exact_over_lambda"] - th5_exact["r_perp_exact_over_lambda"])
        / th5_exact["r_perp_exact_over_lambda"]
        > 0.45
    )
    assert abs((1.0 - th5_mut["beta_parallel"]) - (1.0 - th5_exact["beta_parallel"])) / (
        1.0 - th5_exact["beta_parallel"]
    ) > 0.55

    # Mutation 2: Confuse linear-polarization peak a_0 with RMS eta (polarization='circular' for linear wave)
    th1_wrong_pol = mb.plane_wave_maxwell_invariants_theory(
        2.5, gamma0=1.0, polarization="circular"
    )
    th1_right_pol = mb.plane_wave_maxwell_invariants_theory(
        2.5, gamma0=1.0, polarization="linear"
    )
    assert (
        abs(th1_wrong_pol["beta_parallel"] - th1_right_pol["beta_parallel"])
        / th1_right_pol["beta_parallel"]
        > 0.20
    )


def test_part_d_formation_length_snowplow_energetics_and_mutations():
    """Verify formation length L_f, snowplow scaling (gamma_0 in 1..20), energetics ladder, and mutations."""
    res_d = json.loads((mb.DATA_DIR / "task_part_d_snowplow.json").read_text(encoding="utf-8"))

    # 1. Multiplicative gain gamma_f / gamma_0 = 1 + C eta^2 across gamma_0 in {2, 5, 10, 20} at eta = 3
    rows = res_d["gamma0_rows"]
    c_pond_rel = [r["c_prefactor_pond"] for r in rows if r["gamma0"] >= 2.0]
    c_lor_rel = [r["c_prefactor_lorentz"] for r in rows if r["gamma0"] >= 2.0]
    assert np.all(np.array(c_pond_rel) > 1.45) and np.all(np.array(c_pond_rel) < 1.75)
    assert np.all(np.array(c_lor_rel) > 1.50) and np.all(np.array(c_lor_rel) < 1.75)
    # Ponderomotive and 3D Lorentz trackers agree within 4% across all relativistic gamma_0 >= 5
    for r in rows:
        if r["gamma0"] >= 5.0:
            assert r["gamma_f_over_gamma0_pond"] == pytest.approx(
                r["gamma_f_over_gamma0_lorentz"], rel=0.04
            )

    # 2. Gradient scales as 1 / gamma_0 (Eq. 49): grad(gamma_0=10) / grad(gamma_0=20) == 2.0
    grad_10 = float(rows[3]["gradient_sim_gev_m"])
    grad_20 = float(rows[4]["gradient_sim_gev_m"])
    assert (grad_10 / grad_20) == pytest.approx(2.0, rel=0.02)

    # 3. Reconciliation at gamma_0 = 1 with Part D's Z_{R,opt} = a_0^2 c tau / 2, gamma_opt = 1 + a_0^2 / 2
    assert res_d["reconcile_g1_a0_7p4_gamma_sim"] == pytest.approx(
        res_d["reconcile_g1_a0_7p4_gamma_opt_formula"], rel=0.04
    )

    # 4. Paraxial Barton & Alexander (1989) check: <= 2.5% for k_0 w_0 >= 10 vs > 12% breakdown at k_0 w_0 = 4
    assert res_d["paraxial_ba5_rel_diff_at_kw10"] < 0.025
    assert res_d["paraxial_ba5_rel_diff_at_kw4"] > 0.12

    # 5. Laser energetics ladder U_min proportional to gamma_0^2 eta^4 (Eq. 50)
    ladder = res_d["energetics_ladder_eta3"]
    assert float(ladder[1]["u_min_eq50_mj"] / ladder[0]["u_min_eq50_mj"]) == pytest.approx(
        100.0, rel=1e-12
    )
    assert float(ladder[2]["u_min_eq50_mj"] / ladder[1]["u_min_eq50_mj"]) == pytest.approx(
        100.0, rel=1e-12
    )

    # 6. Bunch relative energy spread preservation and transverse kick p_perp ~ eta m_e c (Sec. 5.2.5)
    assert res_d["bunch_rel_spread_ratio"] == pytest.approx(1.0, rel=0.03)
    assert 0.30 <= res_d["p_perp_over_eta_mc"] <= 0.70

    # Mutation 1: Disable diffraction (z_0 -> inf, plane-wave limit k_z0 = 1e12) -> net snowplow gain vanishes!
    gf_matched = mb.solve_snowplow_ponderomotive_1d(10.0, 3.0, k_z0=4.5 * 100.0 * 9.0)
    gf_no_diff = mb.solve_snowplow_ponderomotive_1d(10.0, 3.0, k_z0=1.0e12)
    assert (gf_matched / 10.0) > 14.0
    assert abs(gf_no_diff - 10.0) < 1e-6

    # Mutation 2: Sub-threshold injection (eta^2 << eta_th^2, i.e. k_0 z_0 = 200 * gamma_0^2 * eta^2)
    gf_subthresh = mb.solve_snowplow_ponderomotive_1d(10.0, 3.0, k_z0=200.0 * 100.0 * 9.0)
    assert (gf_subthresh / 10.0) < 1.15


def _validate_summary_table_against_json(md_text: str, data_dir: Path) -> None:
    """Validate that numbers in the Quantitative Summary markdown table match data/*.json."""
    d01 = json.loads((data_dir / "task_part0_static_gap.json").read_text(encoding="utf-8"))
    d02 = json.loads((data_dir / "task_part0_cavity.json").read_text(encoding="utf-8"))
    d0a = json.loads((data_dir / "task_part0_a_plane_wave.json").read_text(encoding="utf-8"))
    da = json.loads((data_dir / "task_part_a_plane_wave.json").read_text(encoding="utf-8"))
    dbc = json.loads((data_dir / "task_part_b_c_fit_screen.json").read_text(encoding="utf-8"))
    dd = json.loads((data_dir / "task_part_d_ponderomotive.json").read_text(encoding="utf-8"))
    dsp = json.loads((data_dir / "task_part_d_snowplow.json").read_text(encoding="utf-8"))
    de = json.loads((data_dir / "task_part_e_wiggler_fel.json").read_text(encoding="utf-8"))

    row_a25_g5 = next(r for r in d0a["rows"] if r["a0"] == 2.5 and r["gamma0"] == 5.0)

    m = re.search(
        r"## Quantitative Summary vs Literature\s*\n(.*?)\n# ---",
        md_text,
         flags=re.DOTALL,
    )
    assert m is not None, "Could not locate Quantitative Summary vs Literature section."
    table_block = m.group(1)

    expected_tokens = [
        f"{abs(d01['u_int_d2mm_ev']):.1f}",
        f"{d02['minus_u_int_on_crest_uj']:.4f}",
        f"{d02['e_rad_tm010_kv_m']:.2f}",
        f"{d02['k_loss_tm010_proj_v_pc']:.4f}",
        f"{da['gamma_peak_theory']:.4f}",
        f"{row_a25_g5['beta_parallel_vay']:.6f}",
        f"{dbc['dW_screen_waist_theory_norm']:.4f}",
        f"{dd['gamma_at_z_6zr']:.2f}",
        f"{dd['bunch_rms_duration_as']:.1f}",
        f"{dsp['c_prefactor_asymptotic_pond']:.3f}",
        f"{dsp['c_prefactor_asymptotic_lorentz']:.3f}",
        f"{dsp['bunch_rel_spread_ratio']:.4f}",
        f"{de['e_req_mev_laser_0p4um']:.1f}",
        f"{de['e_req_mev_laser_1p0um']:.1f}",
        f"{de['m_a_periods_zolotorev_2002']:.0f}",
    ]
    for tok in expected_tokens:
        if tok not in table_block:
            raise AssertionError(
                f"Expected JSON-derived value {tok!r} not found in Quantitative Summary table."
            )


def test_quantitative_summary_table_matches_json_and_setup_figures():
    """Verify the Quantitative Summary table matches data/*.json and fails under a table-tampering mutation."""
    nb_py = (TUT_DIR / "vacuum_laser_acceleration.py").read_text(encoding="utf-8")
    _validate_summary_table_against_json(nb_py, mb.DATA_DIR)

    # Mutation check: tampering with a numerical result in the Quantitative Summary table must fail validation
    mutated_nb = nb_py.replace("2.4133", "9.9999")
    with pytest.raises(AssertionError, match="not found in Quantitative Summary table"):
        _validate_summary_table_against_json(mutated_nb, mb.DATA_DIR)

    # Verify all new setup figures (00a, 00b) and result figures (09..13) exist and are non-empty
    for fig_name in [
        "00a_part0_cavity_setup_geometry.png",
        "00b_part0_cavity_setup_mesh_and_excitation.png",
        "09_part0_static_gap_interference.png",
        "10_part0_cavity_interference.png",
        "11_part_a_maxwell_invariants.png",
        "12_part_d_snowplow_scaling_and_threshold.png",
        "13_part_d_energetics_and_bunch_stability.png",
    ]:
        p = mb.FIG_DIR / fig_name
        assert p.exists() and p.stat().st_size > 15_000
