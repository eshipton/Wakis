# Copyright 2026 Wakis contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Tutorial 8.7: Metamaterial SWS & Reversed Cherenkov Radiation.

Verifies:
1. CeSRR geometric partition identities and mesh resolution requirements
   (>= 4 cells across 1.0 mm slot and 1.2 mm plate, >= 20 PPW), with an
   under-resolved mesh mutation check (`n_slot=2` fails `_check_mesh_rules`).
2. 3D FIT termination reflection measurement (`termination='matched_taper'` vs
   `termination='smooth_air'`), passivity (`|Gamma(f)| <= 1`), and explicit
   `model` labeling of Duan et al. (2017, SI Note 1) analytic parameters.
3. 3D FIT unit-cell Bloch cold test at `n_slot=4` (>= 4 cells across slot/plate):
   backward dispersion (v_g < 0, v_p > 0), Poynting energy velocity
   v_E = P_z p / U_cell < 0 matching sgn(v_g), candidate Pierce impedance
   normalisations, and plain-plate slot-removal mutation (`plain_plate=True`).
4. 3D FIT reversed Cherenkov radiation (`wakis.sources.Beam`) at `n_slot=4`:
   upstream vs downstream spectral asymmetry, radiative wake exit, discrete
   leapfrog Poynting + absorber energy bookkeeping, and single-parameter
   code-path mutations (`termination='smooth_air'` and `plain_plate=True`).
5. Johnson (1955) BWO start-oscillation condition (I_st ~ N^-3), `LinearizedBeam`
   coupling, and exact agreement between the `metamaterial_sws.py` Quantitative
   Summary markdown table and `data/*.json`.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

TUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "08_vacuum_electronics"
    / "08-07_metamaterial_sws_reversed_cherenkov"
)
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _helpers as h  # noqa: E402


def _check_poynting_vs_vg(pz_w: float, ve_over_c: float, vg_over_c: float) -> bool:
    """Return True iff P_z < 0, v_E < 0, v_g < 0, and v_E agrees with v_g."""
    if not (pz_w < 0.0 and ve_over_c < 0.0 and vg_over_c < 0.0):
        return False
    rel_diff = abs(ve_over_c - vg_over_c) / max(abs(vg_over_c), 1e-12)
    return bool(rel_diff < 0.25)


def _check_energy_bookkeeping(
    w_beam_loss: float,
    u_em: float,
    w_rad_up: float,
    w_rad_dn: float,
    tol: float = 0.01,
) -> bool:
    """Return True iff discrete FIT Poynting energy balance closes within tol."""
    total_accounted = u_em + w_rad_up + w_rad_dn
    rel_err = abs(w_beam_loss - total_accounted) / max(abs(w_beam_loss), 1e-30)
    return bool(w_beam_loss > 0.0 and u_em > 0.0 and rel_err < tol)


def _check_velocity_tuning(betas: np.ndarray, f_syncs_ghz: np.ndarray) -> bool:
    """Return True iff synchronous frequency increases with beam velocity (df_sync/dbeta > 0)."""
    db = np.diff(np.asarray(betas, dtype=float))
    df = np.diff(np.asarray(f_syncs_ghz, dtype=float))
    return bool(len(db) >= 1 and np.all(db > 0.0) and np.all(df > 0.0))


def _validate_summary_table_against_json(
    script_text: str,
    json_store: dict[str, dict],
    rel_tol: float = 0.01,
) -> bool:
    """Verify formatting of every `file.json:key` row in a Quantitative Summary table generated from the same JSON."""
    pattern = re.compile(
        r"\|\s*[^|]+\|\s*`(task_[a-z0-9_]+\.json):([a-z0-9_]+)`\s*\|\s*\$([-+]?[0-9]+\.[0-9]+)"
    )
    matches = pattern.findall(script_text)
    if len(matches) < 10:
        return False
    for fname, key, val_str in matches:
        if fname not in json_store or key not in json_store[fname]:
            return False
        table_val = float(val_str)
        json_val = float(json_store[fname][key])
        if key.endswith("_rel_err"):
            json_val *= 100.0
        elif key.endswith("_j"):
            json_val *= 1e6
        if abs(table_val - json_val) > max(rel_tol * abs(json_val), 1e-4):
            return False
    return True


def _check_mesh_rules(meta: dict) -> bool:
    """Return True iff the mesh satisfies >= 4 cells/slot, >= 4 cells/plate, >= 20 PPW."""
    return bool(
        float(meta["cells_across_slot"]) >= 4.0
        and float(meta["cells_across_plate"]) >= 4.0
        and float(meta["ppw_min_at_3g5"]) >= 20.0
    )


def test_cesrr_geometry_and_mesh_identities():
    """Verify CeSRR coordinate partitions and unit-cell/finite mesh resolution rules."""
    spec = h.CeSRRSWSSpec()
    ident = h.verify_cesrr_coordinate_identities(spec)
    assert ident["x_partition_exact"]
    assert ident["z_partition_exact"]
    assert np.isclose(ident["slot_perimeter_mm"], 90.0)
    assert np.isclose(ident["beam_y_min_mm"], 1.1)
    assert np.isclose(ident["beam_y_max_mm"], 3.1)

    # Check BOTH unit-cell mesh AND finite RCR/termination mesh at n_slot=4, 6, 8
    for ns in (4, 6, 8):
        _, _, _, meta_uc = h.build_cesrr_unit_cell_nodes(spec, n_slot=ns, graded=True)
        assert _check_mesh_rules(meta_uc)

    # Check partition-preserving, slot-snapped geometry perturbations (d +- 10%, Referee Item 9):
    for d_scale in (0.9, 1.1):
        spec_p = h.make_slot_perturbed_spec(spec, d_scale)
        ident_p = h.verify_cesrr_coordinate_identities(spec_p)
        assert ident_p["x_partition_exact"]
        assert ident_p["z_partition_exact"]
        _, _, _, meta_p = h.build_cesrr_unit_cell_nodes(spec_p, n_slot=4, graded=True)
        assert _check_mesh_rules(meta_p)
        assert np.isclose(meta_p["cells_across_slot"], 4.0)

    _, meta_fin = h.build_finite_cesrr_grid(
        spec, n_cells=5, n_slot=4, n_buffer_cells=2
    )
    assert _check_mesh_rules(meta_fin)

    # Mutation check on mesh builder: under-resolving with n_slot=2 (2.67 cells/slot,
    # 2.0 cells/plate) violates the >= 4 cells/feature rule and must fail _check_mesh_rules
    _, _, _, meta_coarse = h.build_cesrr_unit_cell_nodes(spec, n_slot=2, graded=True)
    assert not _check_mesh_rules(meta_coarse)

    freqs = np.linspace(2.86e9, 3.30e9, 25)
    tm_info = h.tm11_cutoff_and_drude_mu_eff(freqs, a_m=spec.waveguide_a_m)
    assert np.isclose(tm_info["fc_tm11_hz"] * 1e-9, 14.6203, rtol=1e-3)


def test_termination_reflection_and_passivity(monkeypatch):
    """Verify 3D FIT two-probe Bloch reflection, partial-reflector band-median consistency check + phase scan, and standing-wave scatter."""
    spec = h.CeSRRSWSSpec()
    sp_npz = np.load(h.DATA_DIR / "task_sparam_retrieval.npz")
    sp_json = json.loads((h.DATA_DIR / "task_sparam_retrieval.json").read_text(encoding="utf-8"))
    assert _check_mesh_rules(sp_json["mesh_meta"])

    # (a) Verify that for termination="smooth_air" with n_cells=3, n_buffer_cells=2
    # (7 total cells ic = 0..6), both probe cells c_probes = [3, 4] contain CeSRR
    # PEC boxes while the 2 downstream buffer cells ic = 5, 6 are empty air:
    boxes_air = h.populate_finite_cesrr_boxes(
        spec=spec, n_cells=3, n_buffer_cells=2, termination="smooth_air"
    )
    p = spec.period_p_m
    box_z_centers = np.array([float(b.center[2]) for b in boxes_air], dtype=float)
    for ic_populated in (0, 1, 2, 3, 4):
        z_c = (ic_populated - 3.0) * p
        assert np.any(np.abs(box_z_centers - z_c) < 0.45 * p), (
            f"Cell ic={ic_populated} must contain CeSRR boxes"
        )
    for ic_empty in (5, 6):
        z_c = (ic_empty - 3.0) * p
        assert not np.any(np.abs(box_z_centers - z_c) < 0.45 * p), (
            f"Downstream air buffer cell ic={ic_empty} must be empty"
        )

    # (b) & (c) Passivity, regression bound (< -12 dB median across 2.96-3.22 GHz),
    # partial-reflector band-median consistency check against cross-sectional energy flux
    # |Gamma_flux| = sqrt(max(0, 1 - P_net/P_inc)) with per-frequency disagreement metrics
    # and phase-sensitivity scan, plus standing-wave sanity check (both band-median and
    # per-frequency min/max/fraction-above-1.04 scatter):
    gamma_taper_mag = np.asarray(sp_npz["gamma_taper_mag"], dtype=float)
    gamma_partial_mag = np.asarray(sp_npz["gamma_partial_mag"], dtype=float)
    gamma_partial_flux_mag = np.asarray(sp_npz["gamma_partial_flux_mag"], dtype=float)
    gamma_air_mag = np.asarray(sp_npz["gamma_air_mag"], dtype=float)
    p_net_taper_w = np.asarray(sp_npz["p_net_taper_w"], dtype=float)
    p_net_partial_w = np.asarray(sp_npz["p_net_partial_w"], dtype=float)

    assert np.all(gamma_taper_mag <= 1.0 + 1e-2)
    assert bool(sp_json["passivity_gamma_taper_le_one"])
    assert "closed_end_gamma_unity_within_tol" not in sp_json
    assert "partial_reflector_validation_within_tol" not in sp_json

    tol = float(sp_json["partial_reflector_band_median_tol"])
    assert bool(sp_json["partial_reflector_band_median_consistency_within_tol"])
    assert float(sp_json["partial_reflector_band_median_abs_diff"]) <= tol
    assert h.check_two_probe_band_median_consistency(
        gamma_partial_mag, gamma_partial_flux_mag, tol=tol
    )

    # Verify per-frequency disagreement and P_inc bias metrics match recomputed NPZ arrays (Referee Item 1b):
    per_freq_diff = np.abs(gamma_partial_mag - gamma_partial_flux_mag)
    assert np.isclose(
        float(sp_json["partial_reflector_per_freq_max_abs_diff"]),
        float(np.max(per_freq_diff)),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(sp_json["partial_reflector_per_freq_rms_diff"]),
        float(np.sqrt(np.mean((gamma_partial_mag - gamma_partial_flux_mag) ** 2))),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(sp_json["partial_reflector_per_freq_corr"]),
        float(np.corrcoef(gamma_partial_mag, gamma_partial_flux_mag)[0, 1]),
        rtol=0.0,
        atol=1e-12,
    )
    clipped_mask = (p_net_partial_w >= p_net_taper_w) | (gamma_partial_flux_mag <= 1e-12)
    assert int(sp_json["partial_reflector_flux_clipped_count"]) == int(np.sum(clipped_mask))
    assert np.isclose(
        float(sp_json["matched_taper_peak_gamma_mag"]),
        float(np.max(gamma_taper_mag)),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(sp_json["matched_taper_peak_gamma_sq_bias"]),
        float(np.max(gamma_taper_mag) ** 2),
        rtol=0.0,
        atol=1e-12,
    )

    # Verify phase-sensitivity scan: shipped phi passes, and smallest failing |dphi| exists and matches JSON (Referee Item 1c):
    scan_npz = h.scan_partial_reflector_phase_sensitivity(
        sp_npz["b_probes_partial"],
        sp_npz["phi_fit_rad"],
        sp_npz["cell_offsets"],
        gamma_partial_flux_mag,
        tol=tol,
    )
    assert scan_npz["dphi_pass_min_rad"] <= 0.0 <= scan_npz["dphi_pass_max_rad"]
    assert (
        np.isfinite(float(scan_npz["dphi_min_failing_abs_rad"]))
        and float(scan_npz["dphi_min_failing_abs_rad"]) > 0.0
    )
    assert np.isclose(
        float(scan_npz["dphi_min_failing_abs_rad"]),
        float(sp_json["partial_reflector_dphi_min_failing_abs_rad"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(scan_npz["dphi_pass_min_rad"]),
        float(sp_json["partial_reflector_dphi_pass_min_rad"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(scan_npz["dphi_pass_max_rad"]),
        float(sp_json["partial_reflector_dphi_pass_max_rad"]),
        rtol=0.0,
        atol=1e-12,
    )
    dec_fail_npz = h.decompose_two_mode_bloch_reflection(
        sp_npz["b_probes_partial"],
        sp_npz["phi_fit_rad"],
        sp_npz["cell_offsets"],
        phi_offset_mutation_rad=float(sp_json["partial_reflector_dphi_first_failing_signed_rad"]),
    )
    assert not h.check_two_probe_band_median_consistency(
        dec_fail_npz["gamma_mag"], gamma_partial_flux_mag, tol=tol
    )

    # Verify standing-wave sanity check: stored min/max/max_abs_err/frac_above_1p04 match recomputed arrays,
    # band-median check passes, and per-frequency tolerance check does not pass (Referee Item 2):
    sw_tol = float(sp_json["standing_wave_band_median_tol"])
    assert bool(sp_json["standing_wave_sanity_check_gamma_unity_within_tol"])
    assert np.isclose(
        float(sp_json["standing_wave_gamma_air_min"]),
        float(np.min(gamma_air_mag)),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(sp_json["standing_wave_gamma_air_max"]),
        float(np.max(gamma_air_mag)),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(sp_json["standing_wave_gamma_air_median"]),
        float(np.median(gamma_air_mag)),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(sp_json["standing_wave_gamma_air_max_abs_err"]),
        float(np.max(np.abs(gamma_air_mag - 1.0))),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(sp_json["standing_wave_gamma_air_frac_above_1p04"]),
        float(np.mean(gamma_air_mag > (1.0 + sw_tol))),
        rtol=0.0,
        atol=1e-12,
    )
    assert abs(float(np.median(gamma_air_mag)) - 1.0) <= sw_tol
    assert not bool(np.all(np.abs(gamma_air_mag - 1.0) <= sw_tol))
    assert float(sp_json["termination_matched_taper_refl_median_db"]) < -12.0
    assert float(sp_json["termination_matched_taper_refl_peak_db"]) < 0.0
    assert abs(float(sp_json["termination_smooth_air_refl_median_db"])) < 0.35

    # Verify that no synthetic retrieval keys exist:
    for forbidden_substr in ("retrieved", "deembed", "s11_", "s21_"):
        assert not any(forbidden_substr in k for k in sp_json)
        assert not any(forbidden_substr in k for k in sp_npz.files)

    # Basis-swap test on the 2x2 two-probe decomposition (`swap_basis_mutation=True`):
    # swapping forward (+i*m*phi) and backward (-i*m*phi) basis columns inverts Gamma -> 1/Gamma
    # so a low-reflection wave (|Gamma| < 1) becomes non-passive (|Gamma_swapped| > 1):
    phi_test = np.linspace(1.1, 2.5, 9)
    offsets_test = np.array([-1.0, 0.0])
    gamma_true = 0.15 * np.exp(0.4j)
    b_synth = (
        np.exp(+1j * offsets_test[:, None] * phi_test[None, :])
        + gamma_true * np.exp(-1j * offsets_test[:, None] * phi_test[None, :])
    )
    dec_nom = h.decompose_two_mode_bloch_reflection(b_synth, phi_test, offsets_test)
    dec_swap = h.decompose_two_mode_bloch_reflection(
        b_synth, phi_test, offsets_test, swap_basis_mutation=True
    )
    assert np.allclose(dec_nom["gamma_complex"], gamma_true, rtol=1e-10)
    assert np.allclose(dec_swap["gamma_complex"], 1.0 / gamma_true, rtol=1e-10)
    assert not np.all(dec_swap["gamma_mag"] <= 1.0 + 1e-2)

    # Live 3D FIT check of the two-probe Bloch reflection estimator:
    # 1. Matched taper (< -12 dB regression bound) and reference incident flux P_inc(f);
    # 2. Partial reflector (`termination="partial_taper"`) band-median consistency check
    #    against cross-sectional energy flux |Gamma_flux| = sqrt(max(0, 1 - P_net/P_inc)),
    #    plus phase-sensitivity scan verifying the smallest failing |dphi| exists and fails;
    # 3. Standing-wave sanity check (`termination="smooth_air"`, band-median |Gamma_air| = 1
    #    within sw_tol):
    res_taper = h.estimate_termination_reflection(
        termination="matched_taper", spec=spec, quick=True
    )
    res_partial = h.estimate_termination_reflection(
        termination="partial_taper",
        spec=spec,
        quick=True,
        p_inc_ref_w=res_taper["p_net_w"],
    )
    res_air = h.estimate_termination_reflection(
        termination="smooth_air",
        spec=spec,
        quick=True,
        p_inc_ref_w=res_taper["p_net_w"],
    )
    assert float(res_taper["refl_median_db"]) < -12.0
    assert h.check_two_probe_band_median_consistency(
        res_partial["gamma_mag"], res_partial["gamma_flux_mag"], tol=tol
    )
    scan_live = h.scan_partial_reflector_phase_sensitivity(
        res_partial["b_probes"],
        res_partial["phi_fit_rad"],
        res_partial["cell_offsets"],
        res_partial["gamma_flux_mag"],
        tol=tol,
    )
    assert np.isfinite(float(scan_live["dphi_min_failing_abs_rad"]))
    dec_partial_phi_mut = h.decompose_two_mode_bloch_reflection(
        res_partial["b_probes"],
        res_partial["phi_fit_rad"],
        res_partial["cell_offsets"],
        phi_offset_mutation_rad=float(scan_live["dphi_first_failing_signed_rad"]),
    )
    assert not h.check_two_probe_band_median_consistency(
        dec_partial_phi_mut["gamma_mag"], res_partial["gamma_flux_mag"], tol=tol
    )
    assert abs(float(np.median(res_air["gamma_mag"])) - 1.0) <= sw_tol
    assert abs(float(res_air["refl_median_db"])) < 0.35

    # Code-path mutation check (monkeypatch on h.build_matched_adiabatic_taper):
    # disabling the absorber conductivity (gamma_max_s=0) turns the matched_taper path
    # into a lossless reflector and MUST fail the low-reflection (< -12 dB regression bound) check:
    orig_taper = h.build_matched_adiabatic_taper
    monkeypatch.setattr(
        h,
        "build_matched_adiabatic_taper",
        lambda grid, dt, z_active_min, z_active_max, gamma_max_s=1.2e10, return_rates=False: orig_taper(
            grid, dt, z_active_min, z_active_max, gamma_max_s=0.0, return_rates=return_rates
        ),
    )
    res_mut = h.estimate_termination_reflection(
        termination="matched_taper", spec=spec, quick=True, t_post_s=5.0e-9
    )
    assert not (float(res_mut["refl_median_db"]) < -12.0)


def test_unit_cell_cold_test_dispersion_poynting_and_plain_plate():
    """Verify 3D FIT Bloch backward dispersion, v_E = v_g < 0, K_c candidates, and convergence."""
    spec = h.CeSRRSWSSpec()
    cold_json = json.loads((h.DATA_DIR / "task_cold_test.json").read_text(encoding="utf-8"))
    assert _check_mesh_rules(cold_json["mesh_meta"])

    # Verify exact staggered FIT v_E vs v_g agreement (< 3% across all passband points)
    assert max(cold_json["dft_ve_vg_rel_diff"]) < 0.03
    assert np.allclose(cold_json["dft_ue_over_uh"], 1.0, rtol=1e-3)

    # Verify 3-level mesh refinement n_slot = [4, 6, 8] and monotonic convergence (Referee Item 8)
    assert list(cold_json["conv_n_slot"]) == [4, 6, 8]
    d_min = np.asarray(cold_json["conv_d_min_mm"], dtype=float)
    f1_conv = np.asarray(cold_json["conv_f1_ghz"], dtype=float)
    kc_conv = np.asarray(cold_json["conv_kc_ohm"], dtype=float)
    assert np.all(np.diff(d_min) < 0.0)
    assert np.all(np.diff(f1_conv) > 0.0)
    assert np.all(np.diff(kc_conv) > 0.0)
    assert float(cold_json["conv_kc_observed_order"]) > 0.5

    # Verify candidate Pierce impedance physical properties and slow SCW dispersion relation:
    assert not any("total_rms" in k for k in cold_json)
    assert (
        0.0
        < float(cold_json["sync_99k7_kc_beam_avg_ohm"])
        < float(cold_json["sync_99k7_kc_center_ohm"])
        < float(cold_json["sync_99k7_kc_surface_ohm"])
    )
    assert np.isclose(
        float(cold_json["sync_99k7_kc_beam_avg_from_ucell_vg_ohm"]),
        float(cold_json["sync_99k7_kc_beam_avg_ohm"]),
        rtol=5e-2,
    )
    assert (
        float(cold_json["sync_99k7_kc_beam_avg_harm_minus1_ohm"])
        < 0.5 * float(cold_json["sync_99k7_kc_beam_avg_ohm"])
    )
    assert (
        float(cold_json["sync_99k7_kc_beam_avg_harm_plus1_ohm"])
        < 0.5 * float(cold_json["sync_99k7_kc_beam_avg_ohm"])
    )
    # Slow space-charge wave omega = beta_0 v_0 - omega_q intersects the backward branch
    # at a larger Bloch phase and lower frequency than the cold beam line:
    f_scw_ghz = float(cold_json["sync_99k7_scw_f_ghz"])
    phi_scw_deg = float(cold_json["sync_99k7_scw_phi_deg"])
    fq_scw_ghz = float(cold_json["sync_99k7_scw_fq_ghz"])
    beta_99k7 = float(cold_json["beta_sweep_sync"][1]["beta"])
    phi_dft_sync_deg = float(cold_json["sync_99k7_dft_phi_deg"])
    assert f_scw_ghz < float(cold_json["sync_99k7_f_ghz"])
    assert phi_scw_deg > float(cold_json["sync_99k7_phi_deg"])
    f_scw_kinematic_ghz = (
        (np.deg2rad(phi_scw_deg) / (2.0 * np.pi)) * (beta_99k7 * h.C0 / spec.period_p_m) * 1e-9
        - fq_scw_ghz
    )
    assert np.isclose(f_scw_ghz, f_scw_kinematic_ghz, rtol=1e-3)

    # Live 3D FIT unit-cell verification at n_slot=4 (>= 4 cells across slot and plate)
    p60 = h.run_unit_cell_bloch_point(60.0, spec=spec, n_slot=4, t_ring_ns=1.2)
    p_sync = h.run_unit_cell_bloch_point(
        phi_dft_sync_deg,
        spec=spec,
        n_slot=4,
        t_ring_ns=1.2,
        extract_dft=True,
    )
    assert _check_mesh_rules(p_sync["mesh_meta"])
    assert p_sync["f1_lh_hz"] < p60["f1_lh_hz"]
    assert p_sync["f2_rh_hz"] > p60["f2_rh_hz"]

    vg_est = (
        2.0
        * np.pi
        * (p_sync["f1_lh_hz"] - p60["f1_lh_hz"])
        / (np.deg2rad(phi_dft_sync_deg - 60.0) / spec.period_p_m)
    ) / h.C0
    assert _check_poynting_vs_vg(p_sync["pz_mean_w"], p_sync["v_energy_over_c"], vg_est)

    # Code-path mutation check: removing CeSRR slots (`plain_plate=True`) in
    # run_unit_cell_bloch_point destroys the left-handed mode and must fail
    p_plain = h.run_unit_cell_bloch_point(
        phi_dft_sync_deg,
        spec=spec,
        n_slot=4,
        plain_plate=True,
        t_ring_ns=1.2,
    )
    assert len(p_plain["lh_candidates"]) == 0 and not np.isfinite(p_plain["f1_lh_hz"])


def test_reversed_cherenkov_beam_asymmetry_tuning_and_energy_bookkeeping(monkeypatch):
    """Verify 3D FIT reversed Cherenkov radiation, radiative exit, and energy balance at n_slot=4."""
    spec = h.CeSRRSWSSpec()
    rcr_json = json.loads((h.DATA_DIR / "task_reversed_cherenkov.json").read_text(encoding="utf-8"))
    cold_npz = np.load(h.DATA_DIR / "task_cold_test.npz")
    assert _check_mesh_rules(rcr_json["mesh_meta"])
    assert float(rcr_json["fraction_wake_radiated"]) > 0.75
    assert 0.0 < float(rcr_json["sheet_vs_line_wake_energy_ratio"]) < 1.0
    assert float(rcr_json["energy_absorber_removed_wake_j"]) > 0.0
    # Verify flux-plane position sensitivity across >= 3 plane offsets (Referee Item 6):
    plane_offsets = list(rcr_json["asymmetry_plane_offsets_cells"])
    plane_vals_db = [float(x) for x in rcr_json["asymmetry_plane_values_db"]]
    plane_spread_db = float(rcr_json["asymmetry_plane_spread_db"])
    assert len(plane_offsets) >= 3
    assert len(plane_vals_db) == len(plane_offsets)
    assert plane_spread_db > 0.0
    assert np.isclose(plane_spread_db, max(plane_vals_db) - min(plane_vals_db), rtol=0.0, atol=1e-12)
    assert np.isclose(float(rcr_json["asymmetry_99k7_db"]), plane_vals_db[-1], rtol=0.0, atol=1e-12)
    assert not np.isclose(
        float(rcr_json["energy_absorber_removed_wake_j"]),
        float(rcr_json["energy_rad_wake_net_j"]),
    )
    assert _check_energy_bookkeeping(
        float(rcr_json["energy_stored_at_clear_j"]),
        float(rcr_json["energy_stored_em_j"]),
        float(rcr_json["energy_absorber_removed_wake_j"]),
        float(rcr_json["energy_rad_wake_net_j"]),
        tol=0.01,
    )

    beta_99k7 = float(rcr_json["beta_99k7"])
    rcr_main = h.run_single_rcr_beam_case(
        beta=beta_99k7,
        spec=spec,
        n_cells=3,
        n_slot=4,
        n_buffer_cells=2,
        plain_plate=False,
        termination="matched_taper",
        sheet_beam=True,
        t_wake_post_ns=3.8,
        record_frames=0,
        check_energy=True,
    )
    assert _check_mesh_rules(rcr_main["mesh_meta"])
    assert 2.80 <= rcr_main["f_peak_ghz"] <= 3.25
    assert rcr_main["asymmetry_db"] > 10.0
    assert rcr_main["w_wake_up_j"] > 10.0 * abs(rcr_main["w_wake_dn_j"])
    assert rcr_main["fraction_wake_radiated"] > 0.70

    # Discrete FIT Poynting energy bookkeeping checks:
    # 1. Transit: W_beam_loss = U_clear + W_rad,transit (within 1% round-off/time-centring error)
    assert _check_energy_bookkeeping(
        rcr_main["energy_beam_loss_j"],
        rcr_main["energy_stored_at_clear_j"],
        rcr_main["energy_rad_transit_tot_j"],
        0.0,
        tol=0.01,
    )
    # 2. Post-clear wake budget: U_clear = U_final + W_absorbed + W_port (within 1% error),
    #    where W_absorbed is the volumetric integral sum_t sum_taper (sigma|E|^2 + sigma_m|H|^2) dt dV
    assert rcr_main["energy_absorber_removed_wake_j"] > 0.0
    assert not np.isclose(
        rcr_main["energy_absorber_removed_wake_j"],
        rcr_main["energy_rad_wake_net_j"],
    )
    assert _check_energy_bookkeeping(
        rcr_main["energy_stored_at_clear_j"],
        rcr_main["energy_stored_em_j"],
        rcr_main["energy_absorber_removed_wake_j"],
        rcr_main["energy_rad_wake_net_j"],
        tol=0.01,
    )

    # Code-path mutation check 1 (single-parameter change: `termination="smooth_air"`):
    # replacing the matched adiabatic taper with a smooth below-cutoff air waveguide
    # traps the wake inside the cavity and MUST fail `fraction_wake_radiated > 0.70`
    rcr_closed = h.run_single_rcr_beam_case(
        beta=beta_99k7,
        spec=spec,
        n_cells=3,
        n_slot=4,
        n_buffer_cells=2,
        plain_plate=False,
        termination="smooth_air",
        sheet_beam=True,
        t_wake_post_ns=3.8,
        record_frames=0,
        check_energy=True,
    )
    assert not (rcr_closed["fraction_wake_radiated"] > 0.70)

    # Code-path mutation check 2 (single-parameter change: `plain_plate=True`):
    # removing the CeSRR slots suppresses the upstream RCR wake energy by > 40 dB
    rcr_plain = h.run_single_rcr_beam_case(
        beta=beta_99k7,
        spec=spec,
        n_cells=3,
        n_slot=4,
        n_buffer_cells=2,
        plain_plate=True,
        termination="matched_taper",
        sheet_beam=True,
        t_wake_post_ns=3.8,
        record_frames=0,
        check_energy=True,
    )
    assert rcr_plain["w_wake_up_j"] < 1e-4 * rcr_main["w_wake_up_j"]

    # Velocity-tuning discrimination check (df_sync / dbeta > 0 along backward branch)
    # and code-path mutation on the beam velocity inside h.find_beam_line_intersection
    # (`invert_velocity_mutation=True` mutates beta -> 1.15 - beta inside the solver
    # function and must fail _check_velocity_tuning):
    betas = np.asarray(rcr_json["beta_sweep"], dtype=float)
    f_peaks = np.asarray(rcr_json["beta_sweep_f_rcr_ghz"], dtype=float)
    assert _check_velocity_tuning(betas, f_peaks)
    f_sync_unmut = np.array(
        [
            h.find_beam_line_intersection(
                cold_npz["phi_deg"], cold_npz["f1_lh_hz"], float(b), spec.period_p_m
            )["f_sync_ghz"]
            for b in betas
        ]
    )
    assert _check_velocity_tuning(betas, f_sync_unmut)
    f_sync_mut = np.array(
        [
            h.find_beam_line_intersection(
                cold_npz["phi_deg"],
                cold_npz["f1_lh_hz"],
                float(b),
                spec.period_p_m,
                invert_velocity_mutation=True,
            )["f_sync_ghz"]
            for b in betas
        ]
    )
    assert not _check_velocity_tuning(betas, f_sync_mut)


def test_bwo_start_current_and_quantitative_summary_table():
    """Verify Johnson BWO start condition, space-charge mutation, and formatting of the Summary Table generated from JSON.

    Note: `_validate_summary_table_against_json` only checks formatting and numerical
    rounding of the markdown table produced from the same `data/*.json` files.
    """
    spec = h.CeSRRSWSSpec()
    cold_json = json.loads((h.DATA_DIR / "task_cold_test.json").read_text(encoding="utf-8"))
    bwo_json = json.loads((h.DATA_DIR / "task_bwo_start_current.json").read_text(encoding="utf-8"))
    phi_dft_sync_deg = float(cold_json["sync_99k7_dft_phi_deg"])
    freq_sync_hz = float(bwo_json["f_sync_ghz"]) * 1e9
    kc_sync_ohm = float(bwo_json["kc_beam_avg_sync_ohm"])
    n_cells_paper = int(bwo_json["n_cells_paper"])
    beta0_sync = np.deg2rad(phi_dft_sync_deg) / spec.period_p_m
    bwo_10 = h.johnson_bwo_start_current(
        freq_hz=freq_sync_hz, beta0_rad_m=beta0_sync, kc_ohm=kc_sync_ohm, n_cells=10
    )
    bwo_20 = h.johnson_bwo_start_current(
        freq_hz=freq_sync_hz,
        beta0_rad_m=beta0_sync,
        kc_ohm=kc_sync_ohm,
        n_cells=n_cells_paper,
    )
    assert np.isclose(float(bwo_20["i_start_a"]), float(bwo_json["i_start_beam_avg_a"]), rtol=1e-6)
    assert float(bwo_20["i_start_a"]) < float(bwo_10["i_start_a"])
    assert float(bwo_20["i_start_a"]) < spec.beam_current_a
    assert 0.05 < float(bwo_20["qc_at_ist"]) < 1.0
    assert float(bwo_20["fp_at_ist_hz"]) * 1e-9 < 1.0

    # Code-path mutation check: dropping the relativistic mass factor inside
    # johnson_bwo_start_current (`drop_relativistic_mutation=True`) must alter I_st
    bwo_mut = h.johnson_bwo_start_current(
        freq_hz=freq_sync_hz,
        beta0_rad_m=beta0_sync,
        kc_ohm=kc_sync_ohm,
        n_cells=n_cells_paper,
        drop_relativistic_mutation=True,
    )
    assert not np.isclose(float(bwo_mut["i_start_a"]), float(bwo_20["i_start_a"]), rtol=1e-2)

    assert _check_mesh_rules(bwo_json["mesh_meta"])
    assert 0.0 < float(bwo_json["linearized_beam_max_j1_abs"]) < 1.0e5

    # Verify GIF existence and non-empty size (Referee Item 7)
    gif_path = h.FIG_DIR / "reversed_cherenkov_wavefronts.gif"
    assert gif_path.exists() and gif_path.stat().st_size > 10_000

    script_path = TUT_DIR / "metamaterial_sws.py"
    script_text = script_path.read_text(encoding="utf-8")
    assert "eshipton/Wakis" in script_text and "ImanolJG/wakis" not in script_text
    assert "build_quantitative_summary_markdown" in script_text

    json_files = [
        "task_cold_test.json",
        "task_sparam_retrieval.json",
        "task_reversed_cherenkov.json",
        "task_bwo_start_current.json",
    ]
    for jf in json_files:
        assert (h.DATA_DIR / jf).exists(), f"Missing committed data file {jf}"

    json_store = {
        jf: json.loads((h.DATA_DIR / jf).read_text(encoding="utf-8")) for jf in json_files
    }
    table_md = h.build_quantitative_summary_markdown(h.DATA_DIR)
    assert _validate_summary_table_against_json(table_md, json_store, rel_tol=0.005)

    # Mutation check: perturbing a JSON number by 20% must fail the table validator
    mutated_store = {k: dict(v) for k, v in json_store.items()}
    mutated_store["task_cold_test.json"]["sync_99k7_f_ghz"] *= 1.20
    assert not _validate_summary_table_against_json(table_md, mutated_store, rel_tol=0.005)


def test_json_key_sets_and_cross_quantity_identities():
    """Verify (a) JSON key sets equal required-argument summary builder key sets and (b) cross-quantity identities."""
    spec = h.CeSRRSWSSpec()
    builders = {
        "task_cold_test.json": h.build_cold_test_summary,
        "task_sparam_retrieval.json": h.build_sparam_retrieval_summary,
        "task_reversed_cherenkov.json": h.build_reversed_cherenkov_summary,
        "task_bwo_start_current.json": h.build_bwo_start_current_summary,
    }
    store: dict[str, dict] = {}
    for fname, builder_fn in builders.items():
        data = json.loads((h.DATA_DIR / fname).read_text(encoding="utf-8"))
        store[fname] = data
        sig = inspect.signature(builder_fn)
        # Verify that no parameter in the summary builder has a default value (Referee Item 8):
        for p_name, param in sig.parameters.items():
            assert param.default is inspect.Parameter.empty, (
                f"Builder {builder_fn.__name__} parameter {p_name} must not have a default"
            )
        expected_keys = set(sig.parameters.keys())
        actual_keys = set(data.keys())
        assert actual_keys == expected_keys, (
            f"Key mismatch in {fname}: extra={sorted(actual_keys - expected_keys)}, "
            f"missing={sorted(expected_keys - actual_keys)}"
        )
        built = builder_fn(**data)
        assert set(built.keys()) == actual_keys
        # Omitting a required key must raise TypeError:
        first_key = next(iter(data.keys()))
        incomplete = {k: v for k, v in data.items() if k != first_key}
        with pytest.raises(TypeError):
            builder_fn(**incomplete)

    cold = store["task_cold_test.json"]
    rcr = store["task_reversed_cherenkov.json"]
    bwo = store["task_bwo_start_current.json"]

    # Cross-quantity identities across and within JSON files:
    assert np.isclose(
        float(rcr["f_cold_sync_99k7_ghz"]),
        float(rcr["beta_sweep_f_cold_ghz"][1]),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(rcr["f_cold_sync_99k7_ghz"]),
        float(cold["sync_99k7_f_ghz"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(cold["sync_99k7_f_ghz"]),
        float(cold["beta_sweep_sync"][1]["f_sync_ghz"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(cold["sync_99k7_phi_deg"]),
        float(cold["beta_sweep_sync"][1]["phi_sync_deg"]),
        rtol=0.0,
        atol=1e-12,
    )
    # Exact identity for the Stage-2 DFT sample at phi = 95.7 deg:
    assert np.isclose(
        float(cold["sync_99k7_dft_f_ghz"]),
        float(cold["conv_f1_ghz"][0]),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(bwo["f_sync_ghz"]),
        float(cold["sync_99k7_dft_f_ghz"]),
        rtol=0.0,
        atol=1e-12,
    )
    # Offset between computed intersection (phi ≈ 95.51 deg) and nearest DFT sample (phi = 95.7 deg)
    # bounded by the DFT phase spacing and group velocity |v_g| (Referee Item 3):
    assert np.isclose(
        float(cold["sync_99k7_dft_phi_offset_deg"]),
        float(cold["sync_99k7_phi_deg"]) - float(cold["sync_99k7_dft_phi_deg"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(cold["sync_99k7_dft_f_offset_ghz"]),
        float(cold["sync_99k7_f_ghz"]) - float(cold["sync_99k7_dft_f_ghz"]),
        rtol=0.0,
        atol=1e-12,
    )
    dft_phi_arr = np.asarray(cold["dft_phi_deg"], dtype=float)
    max_dft_half_step_deg = 0.5 * float(np.max(np.diff(dft_phi_arr)))
    assert abs(float(cold["sync_99k7_dft_phi_offset_deg"])) <= max_dft_half_step_deg
    df_dphi_ghz_per_deg = (
        (abs(float(cold["sync_99k7_vg_over_c"])) * h.C0 / spec.period_p_m)
        * (np.deg2rad(1.0) / (2.0 * np.pi))
        * 1e-9
    )
    tol_dft_f_ghz = df_dphi_ghz_per_deg * max_dft_half_step_deg
    assert np.isclose(
        float(cold["sync_99k7_f_ghz"]),
        float(cold["conv_f1_ghz"][0]),
        rtol=0.0,
        atol=tol_dft_f_ghz,
    )
    assert np.isclose(
        float(bwo["f_sync_ghz"]),
        float(cold["sync_99k7_f_ghz"]),
        rtol=0.0,
        atol=tol_dft_f_ghz,
    )
    assert np.isclose(
        float(rcr["f_peak_vs_cold_rel_err"]),
        float(rcr["beta_sweep_rel_err"][1]),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        float(bwo["kc_beam_avg_sync_ohm"]),
        float(cold["sync_99k7_kc_beam_avg_ohm"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert float(cold["wall_time_s"]) >= float(
        cold["conv_wall_s"][1] + cold["conv_wall_s"][2]
    )
    # Both w_wake_up_j (at z = z_k1, outward normal -z) and w_wake_dn_j (at z = z_k2,
    # outward normal +z) are outward Poynting energy integrals leaving the control
    # volume [z_k1, z_k2], so by Gauss's theorem the net outward radiated wake energy
    # is their sum (Referee Item 9):
    assert np.isclose(
        float(rcr["energy_rad_wake_net_j"]),
        float(rcr["w_wake_up_j"]) + float(rcr["w_wake_dn_j"]),
        rtol=1e-6,
    )



_JSON_FILES = {
    "cold": "task_cold_test.json",
    "sparam": "task_sparam_retrieval.json",
    "sp": "task_sparam_retrieval.json",
    "rcr": "task_reversed_cherenkov.json",
    "bwo": "task_bwo_start_current.json",
}


def _missing_json_keys(src: str, variables: dict[str, str]) -> list[str]:
    """Return ``var['key']`` reads in ``src`` whose key is absent from the JSON bound to ``var``.

    ``variables`` maps a variable name to a key of ``_JSON_FILES``, or to ``"<file key>.mesh_meta"``
    for the ``mesh_meta`` sub-dictionary.
    """
    cache: dict[str, dict] = {}
    missing = []
    for var, target in variables.items():
        file_key, _, sub = target.partition(".")
        if file_key not in cache:
            cache[file_key] = json.loads((TUT_DIR / "data" / _JSON_FILES[file_key]).read_text(encoding="utf-8"))
        d = cache[file_key][sub] if sub else cache[file_key]
        for key in re.findall(rf"(?<![\w.]){var}\[\s*['\"]([A-Za-z_]\w*)['\"]\s*\]", src):
            if key not in d:
                missing.append(f"{var}[{key!r}]")
    return sorted(set(missing))


def test_every_json_key_read_by_notebook_and_summary_exists():
    """Every key the notebook prose, the summary table and the figures read exists in data/*.json.

    Round 7 shipped five metadata keys under names the code never wrote; the notebook build crashed only on the
    cloud. This static check fails first (mutation below proves it can).
    """
    notebook = (TUT_DIR / "metamaterial_sws.py").read_text(encoding="utf-8")
    nb_vars = {"cold": "cold", "sparam": "sparam", "rcr": "rcr", "bwo": "bwo", "sp_meta": "sparam.mesh_meta"}
    summary = inspect.getsource(h.build_quantitative_summary_markdown)
    sm_vars = {"cold": "cold", "sp": "sp", "rcr": "rcr", "bwo": "bwo"}
    figures = inspect.getsource(h.render_all_figures)
    fig_vars = {"cold_json": "cold", "sp_json": "sparam", "rcr_json": "rcr", "bwo_json": "bwo"}

    assert _missing_json_keys(notebook, nb_vars) == []
    assert _missing_json_keys(summary, sm_vars) == []
    assert _missing_json_keys(figures, fig_vars) == []

    # Mutation: a key name the code never writes must be reported.
    mutated = notebook + "\nx = sp_meta['gamma_max_partial_taper_s_inv'] + sparam['no_such_key']\n"
    assert _missing_json_keys(mutated, nb_vars) == [
        "sp_meta['gamma_max_partial_taper_s_inv']",
        "sparam['no_such_key']",
    ]
