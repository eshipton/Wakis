"""Tests for Tutorial 7.7: DFB laser with a quarter-wave phase shift."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = ROOT / "tutorials" / "07_light_sources" / "07-07_dfb_laser"
_SPEC = importlib.util.spec_from_file_location("_helpers_073_dfb", TUT_DIR / "_helpers.py")
assert _SPEC is not None and _SPEC.loader is not None
hlp = importlib.util.module_from_spec(_SPEC)
sys.modules["_helpers_073_dfb"] = hlp
_SPEC.loader.exec_module(hlp)


def _assert_bragg_defect_mode(alpha_L: float, delta_L: float, alpha_cwt: float) -> None:
    """Assert that a fundamental quarter-wave-shifted mode is at the Bragg center with expected threshold."""
    assert abs(delta_L) < 0.12, f"Mode detuning |delta_L| = {abs(delta_L):.4f} rad exceeds 0.12 rad"
    rel_err = abs(alpha_L - alpha_cwt) / alpha_cwt
    assert rel_err < 0.03, f"Threshold gain error {rel_err:.4f} exceeds 3%"


def test_cwt_eigenvalues_energy_balance_and_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Kogelnik-Shank (1972) & Haus-Shank (1976) CWT eigenvalues, energy balance, and mutations."""
    # 1. Transfer matrix is unimodular (det M == 1) for any segment parameters
    m_seg = hlp.cwt_transfer_matrix(0.5, kappa=2.0, alpha=0.7, delta=-1.3, phase=0.4)
    assert np.isclose(np.linalg.det(m_seg), 1.0 + 0.0j, atol=1e-12)

    # 2. Uniform DFB at kL = 2.0: stopband-edge doublet outside |delta L| <= kL = 2.0
    u2 = hlp.solve_cwt_uniform(2.0, m_order=1)
    assert u2["alpha_th_L"] == pytest.approx(0.984669, rel=1e-5)
    assert u2["delta_th_L"] == pytest.approx(3.379719, rel=1e-5)
    assert u2["delta_th_L"] > 2.0
    # Both +delta_1 and -delta_1 satisfy M_total[1, 1] == 0 (exact two-mode degeneracy)
    res_pos = hlp.cwt_characteristic_residual(u2["alpha_th_L"], +u2["delta_th_L"], 2.0, phase_shift=0.0)
    res_neg = hlp.cwt_characteristic_residual(u2["alpha_th_L"], -u2["delta_th_L"], 2.0, phase_shift=0.0)
    assert abs(res_pos) < 1e-11
    assert abs(res_neg) < 1e-11

    # 2b. Gain-dispersive uniform DFB at kL = 2.0, delta_gamma * L = 2.829391 rad (12 THz FWHM):
    #     Kramers-Kronig reactive dispersion pulls the stopband edges inward from 3.3797 to 2.5072 rad
    d_gamma_L = 2.0 * np.pi * 12.0e12 * 3.0 * (30 * hlp.DEFAULT_LAMBDA) / (2.0 * hlp.c0)
    u2_disp = hlp.solve_cwt_uniform_dispersive(2.0, d_gamma_L=d_gamma_L, m_order=1)
    assert u2_disp["delta_th_disp_L"] == pytest.approx(2.507183, rel=1e-5)
    assert u2_disp["alpha0_th_disp_L"] == pytest.approx(1.757839, rel=1e-5)
    assert u2_disp["r_p_th_disp"] == pytest.approx(1.785209, rel=1e-5)
    assert u2_disp["m22_residual"] < 1e-12

    # 3. Quarter-wave-shifted (Phi = pi/2) DFB at kL = 2.0: single defect mode at delta_0 L = 0
    q2 = hlp.solve_cwt_qws(2.0, m_order=0)
    q2_side = hlp.solve_cwt_qws(2.0, m_order=1)
    assert q2["alpha_th_L"] == pytest.approx(0.697135, rel=1e-5)
    assert q2["delta_th_L"] == 0.0
    assert q2["alpha_th_L"] < u2["alpha_th_L"]
    assert q2_side["alpha_th_L"] == pytest.approx(1.429769, rel=1e-5)
    assert q2_side["delta_th_L"] == pytest.approx(5.157664, rel=1e-5)
    assert abs(hlp.cwt_characteristic_residual(q2["alpha_th_L"], 0.0, 2.0, phase_shift=0.5 * np.pi)) < 1e-11
    assert (
        abs(
            hlp.cwt_characteristic_residual(
                q2_side["alpha_th_L"], q2_side["delta_th_L"], 2.0, phase_shift=0.5 * np.pi
            )
        )
        < 1e-11
    )

    # 4. High-kL asymptotics: alpha_0 L ~ 2 kL exp(-kL) (QWS) and alpha_1,uni L ~ (pi / kL)^2 (uniform)
    q6 = hlp.solve_cwt_qws(6.0, m_order=0)
    u12 = hlp.solve_cwt_uniform(12.0, m_order=1)
    assert q6["alpha_th_L"] == pytest.approx(q6["alpha_asymp_L"], rel=0.02)
    assert u12["alpha_th_L"] == pytest.approx(u12["alpha_asymp_L"], rel=0.15)

    # 5. Poynting energy-balance identity: alpha_th * L * int_{-1/2}^{+1/2} (|R|^2 + |S|^2) du == 1
    u_grid = np.linspace(-0.5, 0.5, 1001)
    prof_qws = hlp.cwt_mode_profile(u_grid, kappa_L=2.0, phase_shift=0.5 * np.pi)
    prof_uni = hlp.cwt_mode_profile(u_grid, kappa_L=2.0, phase_shift=0.0)
    assert prof_qws["energy_balance_check"] == pytest.approx(1.0, abs=1e-5)
    assert prof_uni["energy_balance_check"] == pytest.approx(1.0, abs=1e-4)
    assert prof_qws["peak_to_edge"] == pytest.approx(2.8689, rel=1e-3)

    # Mutation checks:
    # (a) Evaluating the QWS mode (alpha_0 L, delta_0 L = 0) in a uniform grating (Phi = 0)
    #     violates the characteristic equation M_total[1, 1] = 0:
    res_mut_phi = hlp.cwt_characteristic_residual(q2["alpha_th_L"], 0.0, 2.0, phase_shift=0.0)
    assert not (abs(res_mut_phi) < 1e-3)
    # (b) Shifting Phi from pi/2 to pi/4 breaks the Bragg condition delta_th L = 0:
    q_mut_45 = hlp.solve_cwt_phase_shifted(2.0, 0.25 * np.pi)
    with pytest.raises(AssertionError):
        _assert_bragg_defect_mode(q_mut_45["alpha_th_L"], q_mut_45["delta_th_L"], q2["alpha_th_L"])
    # (c) Code-path mutation in cwt_mode_profile: dropping the diagonal gain term alpha in
    #     cwt_transfer_matrix breaks the Poynting energy-balance identity for the QWS mode:
    orig_tm = hlp.cwt_transfer_matrix
    with monkeypatch.context() as m:
        m.setattr(
            hlp,
            "cwt_transfer_matrix",
            lambda length, kappa, alpha, delta, phase=0.0: orig_tm(
                length, kappa, 0.0, delta, phase=phase
            ),
        )
        prof_mut = hlp.cwt_mode_profile(
            u_grid,
            kappa_L=2.0,
            alpha_L=q2["alpha_th_L"],
            delta_L=q2["delta_th_L"],
            phase_shift=0.5 * np.pi,
        )
    with pytest.raises(AssertionError):
        assert prof_mut["energy_balance_check"] == pytest.approx(1.0, abs=1e-3)


def test_fit_linear_gain_threshold_1d_and_2d_with_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 1D and 2D slab-waveguide linear-gain harminv threshold extraction and mutations."""
    res = hlp.run_task_00_cwt_and_linear_gain(quick=True)

    # 1. 1D QWS and Uniform threshold gain & detuning at kL = 2.0 match CWT within 1%
    _assert_bragg_defect_mode(
        res["fit_qws_kL2_alpha_L"], res["fit_qws_kL2_delta_L"], res["cwt_qws_kL2_alpha_L"]
    )
    assert res["rel_err_qws_kL2_alpha_pct"] < 1.0
    assert res["rel_err_uni_kL2_alpha_pct"] < 1.0
    assert res["rel_err_uni_kL2_delta_pct"] < 1.0

    # 2. Linear-gain decay-rate zero crossing matches CWT threshold within 1.5%
    zc_err = abs(res["fit_qws_zc_alpha_L"] - res["cwt_qws_kL2_alpha_L"]) / res["cwt_qws_kL2_alpha_L"]
    assert zc_err < 0.015

    # 3. Full kappa_L sweep [0.5 .. 3.0] agrees with CWT
    assert res["max_rel_err_qws_sweep_pct"] < 1.5
    assert res["max_rel_err_uni_sweep_pct"] < 1.0

    # 4. 2D TE0 slab-waveguide check at kL = 2.0
    assert 0.80 < res["slab_2d_Gamma_core"] < 0.84
    assert res["slab_2d_rel_err_alpha_pct"] < 4.5
    assert abs(res["slab_2d_delta_th_L"]) < 0.12

    # Code-path mutation checks:
    # (a) Monkeypatching _grating_modulation inside build_1d_dfb_solver to drop the lambda/4
    #     phase shift (phase_shift -> 0.0) while calling run_1d_harminv_resonance with unchanged
    #     inputs (kappa_L=2.0, phase_shift=pi/2) destroys the Bragg defect mode:
    orig_mod = hlp._grating_modulation
    with monkeypatch.context() as m:
        m.setattr(
            hlp,
            "_grating_modulation",
            lambda z_coords, Lambda, phase_shift, kappa_sign=1.0: orig_mod(
                z_coords, Lambda, 0.0, kappa_sign=kappa_sign
            ),
        )
        r_mut_1d = hlp.run_1d_harminv_resonance(
            kappa_L=2.0,
            phase_shift=0.5 * np.pi,
            n_periods=24,
            cells_per_period=16,
            n_cycles=110.0,
        )
    with pytest.raises(AssertionError):
        _assert_bragg_defect_mode(
            r_mut_1d["alpha_L"], r_mut_1d["delta_L"], res["cwt_qws_kL2_alpha_L"]
        )

    # (b) Monkeypatching _slab_core_index_modulation inside build_2d_slab_dfb_solver to omit the
    #     1/Gamma_core modal confinement factor while calling build_2d_slab_dfb_solver(kappa_L=2.0)
    #     with unchanged inputs weakens the 2D waveguide grating coupling and fails the < 4.5% check:
    with monkeypatch.context() as m:
        m.setattr(
            hlp,
            "_slab_core_index_modulation",
            lambda kappa, lam0, n_eff, n_core, Gamma_core: float(kappa)
            * (float(lam0) / np.pi)
            * (float(n_eff) / float(n_core)),
        )
        slv_mut, m_mut = hlp.build_2d_slab_dfb_solver(
            kappa_L=2.0,
            n_periods=24,
            gain_g0_core=0.0,
        )
    n_steps_mut = int(115.0 * m_mut["T0"] / m_mut["dt"])
    t0_m = 15.0 * m_mut["T0"]
    tau_m = 4.5 * m_mut["T0"]
    trace_mut = np.zeros(n_steps_mut, dtype=float)
    for n in range(n_steps_mut):
        t = n * m_mut["dt"]
        if t < 3.5 * t0_m:
            env = np.exp(-0.5 * ((t - t0_m) / tau_m) ** 2) * np.sin(
                2.0 * np.pi * m_mut["f_B"] * (t - t0_m)
            )
            slv_mut.E[:, :, m_mut["iz_center"], "y"] += (env * m_mut["mode_x"])[:, None]
        slv_mut.one_step()
        trace_mut[n] = float(np.dot(slv_mut.E[:, 0, m_mut["iz_third"], "y"], m_mut["mode_x"]))
    i_s = int(4.0 * t0_m / m_mut["dt"])
    h_mut = hlp.harminv(
        trace_mut[i_s:],
        m_mut["dt"],
        fmin=0.96 * m_mut["f_B"],
        fmax=1.04 * m_mut["f_B"],
        max_rel_error=2e-2,
        min_rel_amplitude=1e-3,
    )
    idx_m = int(np.argmin(np.abs(h_mut.frequency - m_mut["f_B"])))
    alpha_L_mut_2d = float(h_mut.decay_rate[idx_m] * m_mut["n_g"] / hlp.c0 * m_mut["L"])
    rel_err_mut_2d_pct = (
        abs(alpha_L_mut_2d - res["cwt_qws_kL2_alpha_L"]) / res["cwt_qws_kL2_alpha_L"] * 100.0
    )
    with pytest.raises(AssertionError):
        assert rel_err_mut_2d_pct < 4.5


def test_grid_convergence_and_richardson_with_mutation() -> None:
    """Verify monotonic grid convergence in cells per period N_Lambda and Richardson order."""
    conv_quick = hlp.run_task_01_convergence(quick=True)
    cwt_a = conv_quick["cwt_qws_alpha_L"]
    errs_qws = [abs(a - cwt_a) for a in conv_quick["qws_alpha_L"]]
    # Error must decrease monotonically with increasing N_Lambda = (12, 16, 24)
    assert errs_qws[0] > errs_qws[1] > errs_qws[2]
    assert conv_quick["finest_qws_alpha_rel_err_pct"] < 0.6

    # Also verify full 5-grid convergence artifact in data/convergence.json
    conv_full = json.loads((hlp.DATA_DIR / "convergence.json").read_text(encoding="utf-8"))
    assert 1.75 <= conv_full["p_obs_qws_alpha"] <= 2.35
    assert 1.75 <= conv_full["p_obs_uni_alpha"] <= 2.35
    assert conv_full["finest_qws_alpha_rel_err_pct"] < 0.05

    # Code-path mutation check: running the finest resolution level (N_Lambda = 24) with a perturbed
    # grating phase shift (Phi = pi/4 instead of pi/2) corrupts the finest-grid eigenvalue and breaks
    # both monotonic convergence toward CWT and the Richardson convergence slope:
    rq_mut_fine = hlp.run_1d_harminv_resonance(
        kappa_L=2.0,
        phase_shift=0.25 * np.pi,
        n_periods=30,
        cells_per_period=24,
        pad_cells=24,
        cfln_med=0.50,
        n_cycles=120.0,
    )
    qws_a_mut = np.array(
        [conv_quick["qws_alpha_L"][0], conv_quick["qws_alpha_L"][1], rq_mut_fine["alpha_L"]],
        dtype=float,
    )
    errs_mut = np.abs(qws_a_mut - cwt_a)
    with pytest.raises(AssertionError):
        assert errs_mut[0] > errs_mut[1] > errs_mut[2]


def test_fit_saturable_gain_lasing_shb_and_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify above-threshold saturable-gain lasing, SHB, L-I slope, and all 3 physical mutations."""
    las = json.loads((hlp.DATA_DIR / "lasing_dynamics.json").read_text(encoding="utf-8"))
    mut = hlp.run_task_03_mutations(quick=True)

    # 1. Single-mode QWS lasing at delta_0 L ~ 0 vs two-mode uniform doublet at +/-|delta_1 L|
    assert abs(las["qws_peak_delta_L"]) < 0.05
    assert las["qws_smsr_db"] > 30.0
    assert las["uni_peak_neg_delta_L"] < -2.8
    assert las["uni_peak_pos_delta_L"] > +2.8
    assert las["uni_secondary_mode_ratio"] > 0.30

    # 2. Central defect localization & spatial hole burning compression
    assert las["qws_peak_to_edge_fit"] == pytest.approx(las["qws_peak_to_edge_cwt"], rel=0.06)
    assert las["qws_central_gain_compression"] < 0.65

    # 3. L-I curve zero below threshold (r <= 1) and linear slope above threshold
    pump_pts = np.asarray(las["pump_points"], dtype=float)
    i_out_fit = np.asarray(las["I_out_over_Isat_fit"], dtype=float)
    assert np.all(i_out_fit[pump_pts <= 0.85] < 1e-4)
    assert np.all(np.diff(i_out_fit[pump_pts >= 1.0]) > 0.0)
    assert las["slope_rel_diff_pct"] < 10.0

    # 4. Mutation 1: Phase shift Phi != pi/2 (45 deg vs 135 deg) splits the mode by ~3.77 rad
    assert mut["phi_45_splitting_fit_delta_L"] == pytest.approx(
        mut["phi_45_splitting_cwt_delta_L"], rel=0.02
    )
    assert not (abs(mut["fit_phi_delta_L"][1]) < 0.12)

    # 5. Check 2: Flipping kappa -> -kappa preserves the m = 0 eigenvalue (when excited asymmetrically)
    #    but converts z = 0 from an E_x antinode to an E_x node, so a source at z = 0 excites m = +-1:
    assert mut["kappa_pos_center_field_norm"] > 0.95
    assert mut["kappa_neg_center_field_norm"] < 0.10
    assert mut["kappa_neg_asym_alpha_L"] == pytest.approx(mut["kappa_pos_alpha_L"], rel=0.01)
    assert mut["kappa_neg_center_excited_alpha_L"] == pytest.approx(
        mut["cwt_side_m1_alpha_L"], rel=0.03
    )
    assert mut["kappa_neg_center_excited_delta_L"] == pytest.approx(
        mut["cwt_side_m1_delta_L"], rel=0.03
    )
    with pytest.raises(AssertionError):
        _assert_bragg_defect_mode(
            mut["kappa_neg_center_excited_alpha_L"],
            mut["kappa_neg_center_excited_delta_L"],
            mut["kappa_pos_alpha_L"],
        )

    # 6. Check 3: Gain off (g_0 = 0) produces passive decay and zero steady-state lasing output
    assert mut["gain_off_I_out_over_Isat"] < 1e-12
    assert not (mut["gain_off_I_out_over_Isat"] > 0.05)

    # 7. Part D: Self-started lasing from spontaneous-emission noise (FourLevelGain, s <= 100)
    noise = json.loads((hlp.DATA_DIR / "lasing_from_noise.json").read_text(encoding="utf-8"))
    assert noise["time_scale_main"] <= 100.0
    assert all(s <= 100.0 for s in noise["time_scale_list"])
    assert noise["qws_savg_dn_clamped_over_th"] == pytest.approx(1.0, rel=0.05)
    assert noise["qws_savg_vs_theory_rel_err_pct"] < 8.0
    assert noise["qws_shb_vs_rigrod_rel_err_pct"] < 5.0
    assert abs(noise["qws_peak_delta_L"]) < 0.05
    assert noise["qws_smsr_db"] > 20.0
    assert noise["uni_disp_delta_rel_err_pct"] < 2.0
    assert noise["buildup_gain_ratio"] > 10.0
    assert 0.90 <= noise["late_clamping_ratio"] <= 1.10
    assert noise["time_scale_dn_spread_pct"] < 6.0
    ts_lim = noise["time_scaling_limits"]
    assert ts_lim["sat_rate_over_kappa_cav"] < 1.0
    assert ts_lim["rate_21_over_gamma_a"] < 0.10
    assert ts_lim["max_rate_over_gamma_a"] > 100.0

    # Live execution check on Part D (quick=True) + code-path mutation on gain-dispersive CWT
    live_noise = hlp.run_task_04_lasing_from_noise(quick=True)
    assert live_noise["qws_savg_dn_clamped_over_th"] == pytest.approx(1.0, rel=0.06)
    assert live_noise["qws_shb_dn_mean_over_th"] > live_noise["qws_savg_dn_clamped_over_th"] + 0.15
    assert live_noise["qws_savg_vs_theory_rel_err_pct"] < 10.0
    assert live_noise["uni_disp_delta_rel_err_pct"] < 3.0

    # Code-path mutation: omitting Kramers-Kronig mode pulling in solve_cwt_uniform_dispersive
    # (passing d_gamma_L -> 1e6 in the dispersionless limit) causes the uniform DFB doublet
    # splitting check against the 12 THz FourLevelGain simulation to fail by ~26%:
    orig_disp = hlp.solve_cwt_uniform_dispersive
    with monkeypatch.context() as m:
        m.setattr(
            hlp,
            "solve_cwt_uniform_dispersive",
            lambda kappa_L, d_gamma_L, m_order=1: orig_disp(
                kappa_L, 1.0e6, m_order=m_order
            ),
        )
        disp_mut = hlp.solve_cwt_uniform_dispersive(2.0, noise["d_gamma_L"], m_order=1)
        err_mut_pct = (
            abs(live_noise["uni_half_splitting_delta_L"] - disp_mut["delta_th_disp_L"])
            / disp_mut["delta_th_disp_L"]
            * 100.0
        )
    with pytest.raises(AssertionError):
        assert err_mut_pct < 3.0


def test_quantitative_summary_table_and_figures_match_json(tmp_path: Path) -> None:
    """Verify that the Quantitative Summary table in dfb_laser.py matches data/*.json and figures exist."""
    expected_table = hlp.format_summary_table()
    py_text = (TUT_DIR / "dfb_laser.py").read_text(encoding="utf-8")
    nb_obj = json.loads((TUT_DIR / "dfb_laser.ipynb").read_text(encoding="utf-8"))
    nb_md = "\n".join("".join(c.get("source", [])) for c in nb_obj.get("cells", []))

    # Every line of the generated Markdown table must appear in dfb_laser.py and dfb_laser.ipynb
    for line in expected_table.splitlines():
        assert f"# {line}" in py_text, f"Missing or mismatched table line in dfb_laser.py: {line}"
        assert line in nb_md, f"Missing or mismatched table line in dfb_laser.ipynb: {line}"

    # Verify the stale F11b hook is gone and replaced by Part D lasing from noise
    stale_hook = "to be added with F11b"
    assert stale_hook not in py_text
    assert stale_hook not in nb_md
    assert "FourLevelGain" in py_text
    assert "spontaneous_emission=True" in py_text

    # Code-path mutation check: corrupting a JSON value on disk and calling format_summary_table(tmp_path)
    # must cause the table synchronization check against dfb_laser.py to fail:
    for jfile in hlp.DATA_DIR.glob("*.json"):
        (tmp_path / jfile.name).write_text(jfile.read_text(encoding="utf-8"), encoding="utf-8")
    cwt_corrupt = json.loads((tmp_path / "cwt_and_linear_gain.json").read_text(encoding="utf-8"))
    cwt_corrupt["cwt_qws_kL2_alpha_L"] = 9.9999
    (tmp_path / "cwt_and_linear_gain.json").write_text(
        json.dumps(cwt_corrupt), encoding="utf-8"
    )
    mutated_table = hlp.format_summary_table(data_dir=tmp_path)
    with pytest.raises(AssertionError):
        for line in mutated_table.splitlines():
            assert f"# {line}" in py_text

    # Verify all setup figures, result figures, and buildup GIF exist and are non-empty
    required_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00_setup_table.md",
        "00a_setup_geometry_2d.png",
        "00b_setup_mesh_and_excitation_2d.png",
        "00_setup_table_2d.md",
        "01_cwt_and_linear_gain.png",
        "02_2d_slab_mode.png",
        "03_convergence.png",
        "04_lasing_spectra.png",
        "05_shb_profile_and_li_curve.png",
        "06_mutations.png",
        "07_dfb_lasing_buildup.gif",
        "08_lasing_from_noise.png",
    ]
    for fname in required_figs:
        fpath = hlp.FIG_DIR / fname
        assert fpath.exists() and fpath.stat().st_size > 0, f"Missing or empty figure: {fpath}"
