"""Falsifiable physics, code-path mutation, and quantitative-summary tests for Tutorial 8.6 (TWT small-signal gain)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = REPO_ROOT / "tutorials/08_vacuum_electronics/08-06_twt_small_signal_gain"
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as h
from wakis.beam_fluid import LinearizedBeam


def test_pierce_dispersion_launching_loss_and_ode_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Pierce 3-wave/4-wave dispersion, A = -9.54 dB, B = 47.26 dB, cubic symmetry, and ODE equivalence."""
    res_3w = h.solve_pierce_dispersion(b=0.0, QC=0.0, d=0.0, C=0.01, include_backward=False)
    res_4w = h.solve_pierce_dispersion(b=0.0, QC=0.0, d=0.0, C=0.01, include_backward=True)

    # At b = QC = d = 0, growing root has Re(delta_1) = sqrt(3)/2 and Im(delta_1) = -1/2
    assert np.isclose(res_3w["x1"], 0.5 * np.sqrt(3.0), rtol=1e-12)
    assert np.isclose(res_3w["y1"], -0.5, rtol=1e-12)
    assert np.isclose(res_4w["x1"], 0.5 * np.sqrt(3.0), rtol=2e-2)

    # Verify algebraic conjugation symmetry of the 1D Pierce cubic under coupling_sign = -1:
    # Flipping the sign of C^3 conjugates the roots and mirrors b -> -b in the cubic:
    # at b = 0, x1 is unchanged while y1 -> -y1; at b != 0, x1(+b, +1) == x1(-b, -1).
    r_b0_pos = h.solve_pierce_dispersion(b=0.0, QC=0.182, d=0.0, coupling_sign=+1.0)
    r_b0_neg = h.solve_pierce_dispersion(b=0.0, QC=0.182, d=0.0, coupling_sign=-1.0)
    assert np.isclose(r_b0_pos["x1"], r_b0_neg["x1"], rtol=1e-12)
    assert np.isclose(r_b0_pos["y1"], -r_b0_neg["y1"], rtol=1e-12)

    r_pb_pos = h.solve_pierce_dispersion(b=+1.0, QC=0.182, d=0.0, coupling_sign=+1.0)
    r_nb_pos = h.solve_pierce_dispersion(b=-1.0, QC=0.182, d=0.0, coupling_sign=+1.0)
    r_pb_neg = h.solve_pierce_dispersion(b=+1.0, QC=0.182, d=0.0, coupling_sign=-1.0)
    r_nb_neg = h.solve_pierce_dispersion(b=-1.0, QC=0.182, d=0.0, coupling_sign=-1.0)
    assert np.isclose(r_pb_pos["x1"], r_nb_neg["x1"], rtol=1e-12)
    assert np.isclose(r_nb_pos["x1"], r_pb_neg["x1"], rtol=1e-12)
    assert r_pb_pos["x1"] > r_nb_pos["x1"]
    assert r_nb_neg["x1"] > r_pb_neg["x1"]

    # Verify launching loss A = -20*log10(3) and slope B = 20*pi*sqrt(3)*log10(e), plus BVP == ODE
    cn_grid = np.linspace(0.0, 1.1, 111)
    beta_e_test = 1000.0
    c_test = 0.02
    l_max = 1.1 * 2.0 * np.pi / (beta_e_test * c_test)

    def _verify_bvp_and_ode_consistency() -> None:
        bvp0 = h.solve_pierce_three_wave_bvp(cn_grid, b=0.0, QC=0.0, d=0.0)
        ode0 = h.solve_coupled_mode_ode(
            L_m=l_max, beta_e=beta_e_test, C=c_test, b=0.0, QC=0.0, d=0.0, n_pts=111
        )
        a_exact = -20.0 * np.log10(3.0)
        b_exact = 40.0 * np.pi * (0.5 * np.sqrt(3.0)) * np.log10(np.e)
        assert np.isclose(bvp0["A_launch_db"], a_exact, atol=1e-12)
        assert np.isclose(bvp0["B_slope_db"], b_exact, atol=1e-12)
        assert np.max(np.abs(bvp0["gain_db"] - ode0["gain_db_vs_z"])) < 1e-7

        bvp_gen = h.solve_pierce_three_wave_bvp(cn_grid, b=0.75, QC=0.22, d=0.06)
        ode_gen = h.solve_coupled_mode_ode(
            L_m=l_max, beta_e=beta_e_test, C=c_test, b=0.75, QC=0.22, d=0.06, n_pts=111
        )
        assert np.max(np.abs(bvp_gen["gain_db"] - ode_gen["gain_db_vs_z"])) < 1e-6

    _verify_bvp_and_ode_consistency()

    # Code-path mutation: mutating solve_pierce_dispersion to flip coupling_sign=-1.0
    # breaks the BVP-vs-ODE agreement at b = 0.75, QC = 0.22, d = 0.06.
    orig_disp = h.solve_pierce_dispersion
    with monkeypatch.context() as mp:
        mp.setattr(
            h,
            "solve_pierce_dispersion",
            lambda b=0.0, QC=0.0, d=0.0, C=0.01, include_backward=False, coupling_sign=1.0: orig_disp(
                b=b, QC=QC, d=d, C=C, include_backward=include_backward, coupling_sign=-1.0
            ),
        )
        with pytest.raises(AssertionError):
            _verify_bvp_and_ode_consistency()


def test_2d_sheet_to_3d_grating_normalization_and_two_plate_eq4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify exact 2D-to-3D LSE_10 sheet-beam normalization and Karetnikova two-plate Eq. (4) D = 0.6355."""
    spec = h.KaretnikovaTWTSpec()

    def _verify_mapping_and_reduction() -> None:
        m2d = h.map_2d_sheet_to_3d_grating(spec)
        assert np.isclose(m2d["fx_lateral"], 0.565159183, rtol=1e-6)
        assert np.isclose(m2d["w_eff_m"], 0.75200052e-3, rtol=1e-6)
        assert np.isclose(m2d["eta_x_current_factor"], 0.99733973, rtol=1e-6)

        k_2d_test = 1.35e-3  # [Ohm * m]
        k_3d_test = k_2d_test / m2d["w_eff_m"]
        c3_2d = m2d["j_sheet_a_per_m"] * k_2d_test / (4.0 * spec.voltage_v)
        c3_3d = spec.current_a * k_3d_test / (4.0 * spec.voltage_v)
        assert np.isclose(c3_2d, c3_3d, rtol=1e-14)

        # Verify Karetnikova (2018) Eq. (4) two-plate sheet depression factor D = 0.635475 vs Wakis R_planar = 0.652264
        u0, gamma0 = h.relativistic_gamma(voltage=spec.voltage_v)
        beta_e_186 = 2.0 * np.pi * 186.0e9 / u0
        tau_186 = beta_e_186 / gamma0
        assert np.isclose(tau_186 * spec.beam_half_thickness_m, 1.0349, atol=2e-4)
        assert np.isclose(tau_186 * spec.tunnel_half_height_m, 1.3799, atol=2e-4)

        pp_planar = h.pierce_parameters(
            186.0e9, beta_e_186, 1.0, voltage_v=spec.voltage_v, sc_model="wakis_planar"
        )
        pp_kar = h.pierce_parameters(
            186.0e9, beta_e_186, 1.0, voltage_v=spec.voltage_v, sc_model="karetnikova_eq4"
        )
        r_planar = float(pp_planar["R_sc"])
        d_kar = float(pp_kar["R_sc"])
        assert np.isclose(r_planar, 0.652264, atol=5e-4)
        assert np.isclose(d_kar, 0.635475, atol=5e-4)

    _verify_mapping_and_reduction()

    # Code-path mutation: replacing karetnikova_depression_factor_sq with the unbounded free-space
    # sheet formula (omitting the two conducting plates at y = +-a) shifts D from 0.6355 to 0.7676
    # and must fail _verify_mapping_and_reduction().
    with monkeypatch.context() as mp:
        mp.setattr(
            h,
            "karetnikova_depression_factor_sq",
            lambda beta_e, beam_thickness_m, **kwargs: (
                1.0
                - (1.0 - np.exp(-np.asarray(beta_e) * float(beam_thickness_m)))
                / (np.asarray(beta_e) * float(beam_thickness_m))
            ),
        )
        with pytest.raises(AssertionError):
            _verify_mapping_and_reduction()


def test_smooth_tunnel_scw_time_centring_and_solver_mutation() -> None:
    """Verify smooth-tunnel LinearizedBeam SCW time-centring and live tunnel-wall mutation."""
    t1 = h.run_scw_time_centering_check(force=False)
    scw_res = t1["scw_check"]
    assert np.isclose(scw_res["gamma_h_186ghz"], 1.0349, atol=2e-4)
    assert np.isclose(scw_res["gamma_a_186ghz"], 1.3799, atol=2e-4)
    for lvl in scw_res["levels"]:
        # Error bound: 2nd-order leapfrog dispersion O((omega_q dt)^2) + 1D FFT bin resolution < 1.5%
        assert lvl["rel_err_wq"] < 0.015
        assert 0.99 < lvl["amplitude_conservation_ratio"] < 1.01

    spec = h.KaretnikovaTWTSpec()
    u0, gamma0 = h.relativistic_gamma(voltage=spec.voltage_v)
    f_ref = 186.0e9
    omega_ref = 2.0 * np.pi * f_ref
    beta_e_ref = omega_ref / u0
    lam_e = 2.0 * np.pi / beta_e_ref
    r_planar_th = float(
        h.reduction_factor_planar(
            omega_ref,
            u0,
            spec.beam_half_thickness_m,
            a=spec.tunnel_half_height_m,
            relativistic=True,
        )
    )
    wq_th = r_planar_th * 2.60e11
    tau_th = beta_e_ref / gamma0
    t_trans_th = tau_th * float(np.sqrt(max(1.0 / (r_planar_th**2) - 1.0, 0.0)))

    def _measure_scw_wq(tunnel_half_h: float) -> float:
        nz_lam = 20
        ny_tun = 16
        dx = 50.0e-6
        grid_scw = h.GridFIT3D(
            0.0,
            dx,
            -tunnel_half_h,
            +tunnel_half_h,
            0.0,
            lam_e,
            1,
            ny_tun,
            nz_lam,
            verbose=0,
        )
        solver_scw = h.SolverFIT3D(
            grid_scw,
            bc_low=["periodic", "pec", "bloch"],
            bc_high=["periodic", "pec", "bloch"],
            k_bloch=(0.0, 0.0, -float(beta_e_ref)),
            dtype=np.complex128,
            cfln=0.75,
            verbose=0,
        )
        dz = lam_e / float(nz_lam)
        sinc_dz = float(np.sin(beta_e_ref * dz) / (beta_e_ref * dz))
        u0_disc = u0 / sinc_dz
        beam_scw = LinearizedBeam(
            u0=u0_disc,
            omega_p=2.60e11 * (gamma0**1.5),
            gamma0=gamma0,
            axis="z",
            half_thickness=spec.beam_half_thickness_m,
            solver=solver_scw,
        )
        dt = float(solver_scw.dt)
        t_beat = 2.0 * np.pi / wq_th
        n_steps = max(int(np.ceil(1.8 * t_beat / dt)), 2000)
        z_c = beam_scw.s_coords
        y_nodes = grid_scw.y[: solver_scw.Ny]
        es_prof = np.where(
            np.abs(y_nodes) <= spec.beam_half_thickness_m + 1e-12,
            np.cos(t_trans_th * y_nodes),
            0.0,
        )
        beam_scw.v1[:, :] = (
            1.0e4 * np.exp(-1j * beta_e_ref * z_c)[:, None] * es_prof[beam_scw.keep_lines][None, :]
        )
        i1_proj = np.zeros(n_steps, dtype=np.complex128)
        phase_vec = np.exp(+1j * beta_e_ref * z_c) / float(nz_lam)
        for step in range(n_steps):
            solver_scw.one_step()
            i1_proj[step] = np.sum(beam_scw.axial_current() * phase_vec)
        t_arr = (np.arange(n_steps, dtype=float) + 0.5) * dt
        demod_i1 = i1_proj * np.exp(-1j * (beta_e_ref * u0) * t_arr)
        n_pad_fft = 131072
        win = np.hanning(n_steps)
        spec_fft = np.abs(np.fft.fft(demod_i1 * win, n=n_pad_fft))
        freqs_fft = np.fft.fftfreq(n_pad_fft, d=dt) * 2.0 * np.pi
        pos_mask = (freqs_fft > 0.3 * wq_th) & (freqs_fft < 1.8 * wq_th)
        idx_pk = int(np.argmax(spec_fft[pos_mask]))
        freqs_pos = freqs_fft[pos_mask]
        spec_pos = spec_fft[pos_mask]
        y_m, y_0, y_p = np.log(spec_pos[idx_pk - 1 : idx_pk + 2] + 1e-30)
        delta_bin = 0.5 * (y_m - y_p) / max(abs(y_m - 2.0 * y_0 + y_p), 1e-30)
        return float(freqs_pos[idx_pk] - delta_bin * (freqs_pos[1] - freqs_pos[0]))

    wq_valid = _measure_scw_wq(spec.tunnel_half_height_m)
    assert abs(wq_valid - wq_th) / wq_th < 0.015

    wq_mutated = _measure_scw_wq(3.0 * spec.tunnel_half_height_m)
    with pytest.raises(AssertionError):
        assert abs(wq_mutated - wq_th) / wq_th < 0.015


def test_twt_falsifiable_physics_and_live_solver_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Task 4 validation dataset, disclosed failures, and real LinearizedBeam code-path mutations."""
    t2 = json.loads(
        (TUTORIAL_DIR / "data/task_02_2d_grating_hot_convergence.json").read_text(encoding="utf-8")
    )
    t4 = json.loads(
        (TUTORIAL_DIR / "data/task_04_validation_and_failure_tests.json").read_text(encoding="utf-8")
    )

    # Grid rule verification across all 4 levels (coarse, medium, fine, ultrafine):
    # >= 20 cells per guided wavelength lambda_g = 436.07 um and >= 4 cells across 25 um beam-wall gap.
    lvls = t2["levels"]
    assert len(lvls) == 4
    for lv in lvls:
        assert lv["mesh"]["cells_per_lambda_g_186ghz"] >= 20.0
        assert lv["mesh"]["cells_across_beam_wall_gap"] >= 4
    # Monotone decrease of same-grid relative error across coarse -> medium -> fine -> ultrafine
    errs_grid = [lv["rel_err_vs_pierce_grid"] for lv in lvls]
    assert np.all(np.diff(errs_grid) < 0.0)
    assert errs_grid[-1] < 0.02

    # Pre-asymptotic convergence reporting: pairwise orders 2.53, 1.19, 4.15 oscillate,
    # so in_asymptotic_range is False and no Richardson value is claimed as converged.
    assert t2["pairwise_orders"] == [2.53, 1.19, 4.15]
    assert t2["in_asymptotic_range"] is False
    assert t2["observed_order_p"] == pytest.approx(2.223, abs=1e-3)  # cached regression check

    # 1. Coupling-sign flip (coupling_sign = -1):
    # Report and test BOTH pairs:
    # - Peak pair (+b,+1) = 202.4176 vs (-b,-1) = 215.0878 Np/m (6.26% difference; peak moves to -b).
    # - Off-peak pair (-b,+1) = 29.3319 vs (+b,-1) = 178.4078 Np/m (508.2% difference; mirror symmetry FAILS).
    csm = t4["coupling_sign_mutation"]
    assert np.isclose(csm["cubic_x1_b0_pos_sign"], csm["cubic_x1_b0_neg_sign"], rtol=1e-12)
    assert np.isclose(csm["cubic_x1_pos_b_pos_sign"], csm["cubic_x1_neg_b_neg_sign"], rtol=1e-12)
    assert np.isclose(csm["cubic_x1_neg_b_pos_sign"], csm["cubic_x1_pos_b_neg_sign"], rtol=1e-12)
    assert csm["alpha_pos_b_pos_sign_np_m"] > csm["alpha_neg_b_pos_sign_np_m"]
    assert csm["alpha_neg_b_neg_sign_np_m"] > csm["alpha_pos_b_neg_sign_np_m"]
    assert csm["mirror_peak_rel_diff"] == pytest.approx(0.06259, abs=1e-4)
    assert csm["mirror_offpeak_rel_diff"] == pytest.approx(5.08238, abs=1e-3)
    assert csm["mirror_symmetry_holds_for_both_pairs"] is False

    # 2. Current scaling at b = 0:
    # Disclose and verify both:
    # (a) Raw 22-period FIT window slope (18.75, 64.73, 145.92, 236.49 Np/m) does NOT follow I0^(1/3)
    #     (log-log slope = 1.4575 vs 1/3).
    # (b) BVP-de-embedded alpha_FIT^(QC=0) (159.40, 197.06, 241.55, 292.71 Np/m, inverted at period 19
    #     with Pierce's own b=0 and QC) has log-log slope 0.35166 (5.50% from 1/3; regression bound < 0.07
    #     consistent with the 6.93% coarse-grid baseline discretization error), but sits 28.6%-33.3%
    #     (~30%) above Pierce (123.97, 149.40, 181.25, 220.48 Np/m) at every current.
    crs = t4["current_cube_root_scaling"]
    assert crs["raw_window_loglog_slope"] == pytest.approx(1.4575, abs=5e-4)
    assert crs["loglog_slope_fit"] == pytest.approx(0.35166, abs=5e-4)
    assert crs["slope_rel_err"] == pytest.approx(0.05499, abs=5e-4)  # regression check on cached coarse run
    over_pct = np.asarray(crs["deemb_vs_pierce_overprediction_pct"], dtype=float)
    assert np.all((over_pct > 28.0) & (over_pct < 34.0))

    # 3. Velocity detuning b sweep and space-charge QC sweep:
    # Relative difference vs single-wave Pierce is 6.93% at b = 0; 20.48% at b = -0.5 (not agreement,
    # cached regression value), while the 1D total-field BVP / crestatron explanation is WITHDRAWN
    # because at b = -1.0, -0.5, 0.0 it gives -142.01, -266.46, +15.74 Np/m vs FIT +29.33, +88.29, +145.92 Np/m.
    vb = t4["velocity_detuning_b"]
    assert vb["rel_err_at_b0"] == pytest.approx(0.06931, abs=5e-4)
    assert vb["rel_err_at_bm0p5"] == pytest.approx(0.20484, abs=5e-4)  # cached regression value (not agreement)
    assert vb["rel_err_at_bm1p0"] == pytest.approx(0.54958, abs=5e-4)
    assert vb["total_field_bvp_explanation_withdrawn"] is True
    alpha_b_fit = np.asarray(vb["alpha_fit_np_m"], dtype=float)
    alpha_b_win = np.asarray(vb["alpha_bvp_total_field_win_np_m"], dtype=float)
    assert len(alpha_b_fit) == 6 and len(alpha_b_win) == 6
    # Verify the quoted disagreement at b <= 0 that required withdrawing the 1D total-field BVP model:
    assert np.all(alpha_b_fit[:3] > 25.0)
    assert alpha_b_win[0] < -140.0 and alpha_b_win[1] < -260.0 and alpha_b_win[2] < 20.0

    qc_sw = t4["space_charge_qc_reduction"]
    alpha_qc_fit = np.asarray(qc_sw["alpha_fit_np_m"], dtype=float)
    alpha_qc_pierce = np.asarray(qc_sw["alpha_pierce_np_m"], dtype=float)
    assert np.all(np.diff(alpha_qc_fit) < 0.0)
    # Tolerance labels / derivation:
    # - At QC = 0.009 (index 0), the coarse-grid finite-window (m in [8, 19]) deviation from
    #   single-wave Pierce is 0.38% (178.39 vs 179.07 Np/m); rtol = 0.05 (5% regression bound).
    # - At QC = 0.1809 (index 1, nominal b = 0 operating point), the coarse-grid discretization
    #   error vs same-grid Pierce is lvls[0]["rel_err_vs_pierce_grid"] = 6.931% (145.92 vs
    #   136.46 Np/m); rtol = 0.08 (derived as ~1.15 * 0.06931) bounds the known coarse-grid error.
    coarse_rel_err = float(lvls[0]["rel_err_vs_pierce_grid"])
    assert np.isclose(alpha_qc_fit[0], alpha_qc_pierce[0], rtol=0.05)
    assert np.isclose(alpha_qc_fit[1], alpha_qc_pierce[1], rtol=max(0.08, 1.15 * coarse_rel_err))

    # 4. Live 2D FIT solver runs and real code-path mutations on a 10-period coarse grating:
    spec = h.KaretnikovaTWTSpec()
    grid, meta = h.build_2d_grating_hot_grid(
        spec=spec, n_periods=10, n_in_smooth=2, n_out_smooth=2, mesh_level="coarse"
    )
    f0 = 186.0e9
    cold_on_grid = lvls[0]["cold_circuit_on_grid"]
    beta_c = float(cold_on_grid["beta_c_fit_rad_m"])
    c_grid = float(cold_on_grid["C_grid"]) * (0.45 / 0.25) ** (1.0 / 3.0)
    dz = spec.period_m / float(meta["nz_per_period"])
    sinc_dz = float(np.sin(beta_c * dz) / (beta_c * dz))
    u0_sync = (2.0 * np.pi * f0 / beta_c) / sinc_dz
    u0_pos_b = u0_sync * (1.0 + c_grid * 1.0)
    u0_neg_b = u0_sync * (1.0 - c_grid * 1.0)

    cold = h.run_2d_grating_hot_case(
        grid, spec, n_periods=10, freq_hz=f0, current_a=0.0, u0_override=u0_sync, t_total_s=115.0e-12
    )
    assert float(np.max(np.abs(cold["i1_period_rms"]))) == 0.0

    z_p = cold["z_period_m"]

    def _fit_hot_slope(res_h: dict) -> float:
        rat = res_h["ez_period_rms"] / np.maximum(cold["ez_period_rms"], 1e-30)
        return float(np.polyfit(z_p[4:], np.log(np.maximum(rat[4:], 1e-30)), 1)[0])

    def _run_and_assert_forward_amplification_at_pos_b() -> tuple[float, float]:
        hot_pb = h.run_2d_grating_hot_case(
            grid,
            spec,
            n_periods=10,
            freq_hz=f0,
            current_a=0.45,
            u0_override=u0_pos_b,
            coupling_sign=+1.0,
            t_total_s=115.0e-12,
        )
        hot_nb = h.run_2d_grating_hot_case(
            grid,
            spec,
            n_periods=10,
            freq_hz=f0,
            current_a=0.45,
            u0_override=u0_neg_b,
            coupling_sign=+1.0,
            t_total_s=115.0e-12,
        )
        sl_pb = _fit_hot_slope(hot_pb)
        sl_nb = _fit_hot_slope(hot_nb)
        assert sl_pb > 50.0
        assert sl_nb < 0.0
        assert sl_pb > sl_nb
        return sl_pb, sl_nb

    slope_pos_b_pos_sign, slope_neg_b_pos_sign = _run_and_assert_forward_amplification_at_pos_b()

    # Verify that when coupling_sign = -1.0 is passed explicitly on the 10-period grid,
    # the peak moves from +b to -b. On the 22-period grid (0.25 A, m in [8, 19]), mirror
    # symmetry fails for the off-peak pair (29.33 vs 178.41 Np/m). This asymmetry is not
    # predicted by the 3-wave Pierce BVP, which is exactly mirror-symmetric at all z; it is
    # unexplained. (The live 10-period test at 0.45 A, m in [5, 8], is a different current and window.)
    hot_pos_b_neg_sign = h.run_2d_grating_hot_case(
        grid,
        spec,
        n_periods=10,
        freq_hz=f0,
        current_a=0.45,
        u0_override=u0_pos_b,
        coupling_sign=-1.0,
        t_total_s=115.0e-12,
    )
    hot_neg_b_neg_sign = h.run_2d_grating_hot_case(
        grid,
        spec,
        n_periods=10,
        freq_hz=f0,
        current_a=0.45,
        u0_override=u0_neg_b,
        coupling_sign=-1.0,
        t_total_s=115.0e-12,
    )
    slope_pos_b_neg_sign = _fit_hot_slope(hot_pos_b_neg_sign)
    slope_neg_b_neg_sign = _fit_hot_slope(hot_neg_b_neg_sign)
    assert slope_neg_b_neg_sign > 50.0
    assert slope_pos_b_neg_sign < 0.0
    assert np.isclose(
        slope_pos_b_neg_sign,
        csm["short_10period_vs_22period_reconciliation"]["slope_10period_run_0p45A_periods_5_to_8_np_m"],
        atol=0.05,
    )
    assert np.isclose(slope_pos_b_pos_sign, slope_neg_b_neg_sign, rtol=0.10)

    # Real code-path mutation A: monkeypatch LinearizedBeam._pre_step so the beam-field Lorentz
    # acceleration coefficient self._alpha_E has its sign flipped inside the solver time-step
    # even when coupling_sign=+1.0 is requested. Calling _run_and_assert_forward_amplification_at_pos_b()
    # MUST fail.
    orig_pre_step = LinearizedBeam._pre_step

    def _mutated_pre_step_flipped_lorentz_sign(self: LinearizedBeam, solver: object) -> None:
        orig_alpha_e = self._alpha_E
        self._alpha_E = -orig_alpha_e
        try:
            orig_pre_step(self, solver)
        finally:
            self._alpha_E = orig_alpha_e

    with monkeypatch.context() as mp:
        mp.setattr(LinearizedBeam, "_pre_step", _mutated_pre_step_flipped_lorentz_sign)
        with pytest.raises(AssertionError):
            _run_and_assert_forward_amplification_at_pos_b()

    # Real code-path mutation B: monkeypatch LinearizedBeam._pre_step so it does not deposit J_1
    # into solver.J during time stepping. Calling _run_and_assert_forward_amplification_at_pos_b() MUST fail.
    with monkeypatch.context() as mp:
        mp.setattr(LinearizedBeam, "_pre_step", lambda self, solver: None)
        with pytest.raises(AssertionError):
            _run_and_assert_forward_amplification_at_pos_b()


def test_bwo_johnson_1955_start_oscillation_and_pierce_l_minus_3_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Johnson (1955) BWO start-oscillation (CN)_st = 0.314, b_st = 1.522, and exact Pierce L^-3 identity."""
    bwo_st = h.solve_bwo_start_oscillation(QC=0.0, d=0.0)
    assert bwo_st["residual_abs"] < 1e-10
    assert np.isclose(bwo_st["CN_start"], 0.314055, rtol=1e-4)
    assert np.isclose(bwo_st["b_start"], 1.521839, rtol=1e-4)

    # Exact Pierce algebraic identity check: at QC = d = 0, T_BWO depends on (I0, L) only through
    # CN = C * N = (I0 * K / (4 * V0_eff))^(1/3) * (L / lambda_e) = (CN)_st, so root-solving
    # T_BWO(I0, b; L) = 0 across L identically satisfies I_st(L) proportional to L^-3.
    lengths_m = np.array([12.5e-3, 17.5e-3, 25.0e-3, 35.0e-3, 50.0e-3], dtype=float)
    bwo_len = h.solve_bwo_start_current_vs_length(
        lengths_m=lengths_m,
        freq_hz=97.308e9,
        beta_m1=1.5 * np.pi / 0.5e-3,
        k_m1_ohm=2.08,
        voltage_v=12401.0,
    )
    assert np.max(bwo_len["max_residual_abs"]) < 1e-9
    assert np.isclose(bwo_len["loglog_slope"], -3.0, rtol=1e-5)

    # Code-path mutation: flipping backward_sign=-1.0 in solve_bwo_start_oscillation (replacing
    # the BWO backward-wave dispersion sign with the forward TWT sign) destroys the collector-end
    # zero-voltage root at (CN_st, b_st) = (0.314055, 1.521839).
    mut_st = h.solve_bwo_start_oscillation(QC=0.0, d=0.0, backward_sign=-1.0)
    with pytest.raises(AssertionError):
        assert mut_st["residual_abs"] < 1e-6 and np.isclose(
            mut_st["CN_start"], 0.314055, rtol=1e-2
        )


def test_quantitative_summary_table_cli_and_json_consistency() -> None:
    """Verify Marosi dimensions, CLI batch options, and that twt_small_signal_gain.py matches data/summary.json."""
    summary = json.loads((TUTORIAL_DIR / "data/summary.json").read_text(encoding="utf-8"))
    py_text = (TUTORIAL_DIR / "twt_small_signal_gain.py").read_text(encoding="utf-8")

    # Verify that fabricated fig3_plot_readings are completely absent from JSON and notebook text
    kb = summary["task_03_karetnikova_gain_vs_freq"]["karetnikova_2018_benchmarks"]
    assert "fig3_plot_readings" not in kb
    assert "fig3_plot_readings" not in py_text
    assert kb["overprediction_range_db"] == [3.11, 4.53]
    assert kb["hypothetical_midplane_shift_range_db"] == [3.45, 3.6]
    plc = kb["paper_line_citations"]
    assert plc["fig3_25mm_20kv_5_to_13db_60ghz_band"] == "l.205-206"
    assert plc["fig3_25mm_21kv_19db"] == "l.209"
    assert plc["text_40mm_20kv_10_to_20db"] == "l.289-290"
    assert plc["fig5a_25mm_20kv_10db"] == "l.328"
    assert plc["fig6_190ghz_40mm_20kv_75w_at_1p01w"] == "l.278/291"
    v21 = summary["task_03_karetnikova_gain_vs_freq"]["fig3_voltage_sweep"]["v_21kv"]
    assert v21["is_lower_sweep_boundary"] is True
    assert v21["cold_table_min_freq_ghz"] == 178.99
    assert v21["gain_25mm_at_cold_table_min_db"] == 23.74

    # Check 3D Marosi dimensions (Table I l.72-74: a = 1.40 mm; beam dimensions l.103: 0.55 mm x 0.10 mm = 100 um)
    # and cell counts across 100 um beam thickness (2.0 on coarse_3d, 2.5 on medium_3d, 8.0 on batch)
    m3d = summary["task_05_3d_marosi_and_bwo"]["marosi_3d_vband"]
    assert m3d["paper_ref_file"] == ".ref/sws/Marosi_2025_glide_sws.txt"
    assert m3d["paper_table1_lines"] == "l.72-74"
    assert m3d["paper_beam_dimensions_line"] == "l.103"
    assert m3d["paper_table1_line"] == 103
    assert m3d["waveguide_width_a_mm"] == 1.4
    assert m3d["waveguide_height_b_mm"] == 0.8
    assert m3d["period_d_mm"] == 1.05
    assert m3d["beam_width_mm"] == 0.55
    assert m3d["beam_thickness_mm"] == 0.1
    assert m3d["beam_thickness_um"] == 100.0
    assert m3d["paper_operating_voltage_kv"] == 5.2
    assert m3d["paper_operating_current_ma"] == 11.0
    assert m3d["hot_test_3d_converged"] is False
    assert m3d["hot_test_3d_levels"][0]["cells_across_100um_beam_thickness"] == 2.0
    assert m3d["hot_test_3d_levels"][1]["cells_across_100um_beam_thickness"] == 2.5
    assert m3d["full_60period_batch_estimate"]["cells_across_100um_beam_thickness"] == 8.0
    assert (
        m3d["full_60period_batch_estimate"]["note"]
        == "a preflight estimate only; the full 60-period run is not implemented"
    )

    # Verify the quoted preflight CLI options (--run-3d-marosi-full60 --nx 56 --ny 64 --nz-per 56 --n-periods 60)
    # parse cleanly and execute in preflight mode without error:
    parsed = h.parse_cli_args(
        ["--run-3d-marosi-full60", "--nx", "56", "--ny", "64", "--nz-per", "56", "--n-periods", "60", "--preflight-only"]
    )
    assert parsed.run_3d_marosi_full60 is True
    assert (parsed.nx, parsed.ny, parsed.nz_per, parsed.n_periods) == (56, 64, 56, 60)
    pf_info = h.run_3d_marosi_full_batch(
        nx=parsed.nx, ny=parsed.ny, nz_per=parsed.nz_per, n_periods=parsed.n_periods, preflight_only=True
    )
    assert pf_info["grid_shape"] == [56, 64, 3472]
    assert pf_info["n_cells"] == 12443648
    assert pf_info["cells_across_100um_beam_thickness"] == 8.0
    assert pf_info["status"] == "preflight_estimate_only"

    # Execute the notebook's Quantitative Summary table formatting and verify all numbers match JSON
    t1 = summary["task_01_pierce_and_scw"]
    t2 = summary["task_02_2d_convergence"]
    assert t2["alpha_fit_increments_np_m"] == [4.88, 1.8, 2.36]
    t3 = summary["task_03_karetnikova_gain_vs_freq"]
    t4 = summary["task_04_validation_suite"]
    t5 = summary["task_05_3d_marosi_and_bwo"]

    def _verify_notebook_uses_json_keys(text: str) -> None:
        required_refs = [
            "t1['A_launching_loss_db']",
            "t1['B_pierce_slope_db']",
            "scw['R_planar_wakis']",
            "scw['D_karetnikova_eq4']",
            "scw['gamma_h_186ghz']",
            "scw['gamma_a_186ghz']",
            't2["pairwise_orders"]',
            "t2['observed_order_p']",
            't2["alpha_fit_increments_np_m"]',
            "kb['overprediction_range_db']",
            "kb['hypothetical_midplane_shift_range_db']",
            "csm['mirror_peak_rel_diff']",
            "csm['mirror_offpeak_rel_diff']",
            "csm['alpha_pierce_neg_b_pos_sign_np_m']",
            "crs['raw_window_loglog_slope']",
            "crs['loglog_slope_fit']",
            "crs['mean_deemb_vs_pierce_overprediction_pct']",
            "m3d['waveguide_width_a_mm']",
            "m3d['beam_thickness_um']",
            "m3d['paper_ref_file']",
            "m3d['paper_table1_lines']",
            "m3d['paper_beam_dimensions_line']",
            "fb3d['note']",
            "bwo['CN_start_qc0']",
            "bwo['b_start_qc0']",
        ]
        for ref in required_refs:
            assert ref in text, f"Missing dynamic JSON reference in notebook: {ref}"
        # Stale numbers / forbidden strings must not appear anywhere in the notebook script
        stale_strings = [
            "1.0155",
            "1.3540",
            "7.6%",
            "19.6%",
            "180.5 -> 147.2",
            "17.51 vs 14.00",
            "22.12 vs 19.22",
            "3.5–4.1 dB",
            "3.5–4.5 dB",
            "0.70 mm",
            "0.14 mm",
            "140 \\mu\\mathrm{m}",
            "a = 2.50",
            " P" + "IC",
            "(P" + "IC)",
            "/root/computational-electromagnetics/wt-squash",
            "2025_arXiv2505.22927",
        ]
        for stale in stale_strings:
            assert stale not in text, f"Stale number/string still present in notebook: {stale}"

    _verify_notebook_uses_json_keys(py_text)

    corrupted_text = py_text + "\n# stale 1.0155"
    with pytest.raises(AssertionError):
        _verify_notebook_uses_json_keys(corrupted_text)
