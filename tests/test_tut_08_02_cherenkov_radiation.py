"""Tests for Tutorial 8.2: Cherenkov radiation from a moving charge."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import mu_0

ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = ROOT / "tutorials" / "08_vacuum_electronics" / "08-02_cherenkov_radiation"
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _helpers as h  # noqa: E402


def test_cherenkov_angle_and_frank_tamm_theory_with_mutations():
    """Verify analytical cone angle, Frank-Tamm spectrum, bunch form factor, and n=1 mutation."""
    cone_sup = h.cherenkov_cone_angle(beta=0.90, n_refr=1.5)
    assert bool(cone_sup["above_threshold"]) is True
    assert cone_sup["cos_theta_c"] == pytest.approx(1.0 / 1.35, rel=1e-12)
    assert cone_sup["theta_c_deg"] + cone_sup["shock_half_angle_deg"] == pytest.approx(
        90.0, abs=1e-12
    )

    cone_sub = h.cherenkov_cone_angle(beta=0.60, n_refr=1.5)
    assert bool(cone_sub["above_threshold"]) is False
    assert np.isnan(cone_sub["theta_c_deg"])

    # Frank-Tamm spectrum and Gaussian bunch form factor F(omega) = exp(-(omega*sigma_z/v)^2)
    freqs = np.array([10.0e9, 25.0e9, 50.0e9])
    omega = 2.0 * np.pi * freqs
    q_c = 1.0e-9
    sigmaz_m = 1.5e-3
    beta = 0.90
    n_refr = 1.5
    ft = h.frank_tamm_spectrum(
        freqs_hz=freqs,
        beta=beta,
        n_complex=n_refr,
        q_c=q_c,
        sigmaz_m=sigmaz_m,
    )
    expected_point = (q_c**2 * mu_0 * omega / (4.0 * np.pi)) * (
        1.0 - 1.0 / (beta * n_refr) ** 2
    )
    expected_ff = np.exp(-((omega * sigmaz_m / (beta * c_light)) ** 2))
    np.testing.assert_allclose(ft["d2W_domega_dz_point"], expected_point, rtol=1e-12)
    np.testing.assert_allclose(ft["form_factor"], expected_ff, rtol=1e-12)
    np.testing.assert_allclose(
        ft["d2W_domega_dz_bunch"], expected_point * expected_ff, rtol=1e-12
    )

    # Mutation check: setting n = 1.0 at beta = 0.90 puts the charge below threshold
    ft_n1 = h.frank_tamm_spectrum(
        freqs_hz=freqs,
        beta=beta,
        n_complex=1.0,
        q_c=q_c,
        sigmaz_m=sigmaz_m,
    )
    assert not np.any(ft_n1["above_threshold"])
    np.testing.assert_allclose(ft_n1["d2W_domega_dz_bunch"], 0.0, atol=1e-30)


def _assert_coulomb_probe_matches_theory(
    zeta_m: np.ndarray,
    ez_sim: np.ndarray,
    ez_th: np.ndarray,
) -> None:
    """Assert sign, extremum locations, normalized shape correlation, L2 error, and peak amplitude."""
    win = np.abs(zeta_m) <= 10.0e-3
    z_w = zeta_m[win]
    sim_w = ez_sim[win]
    th_w = ez_th[win]

    # Ahead of a q > 0 bunch (zeta > 0), Ez points in +z (> 0); behind (zeta < 0), Ez points in -z (< 0)
    ahead = (z_w >= 0.5e-3) & (z_w <= 6.0e-3)
    behind = (z_w >= -6.0e-3) & (z_w <= -0.5e-3)
    assert float(np.mean(sim_w[ahead])) > 0.0
    assert float(np.mean(th_w[ahead])) > 0.0
    assert float(np.mean(sim_w[behind])) < 0.0
    assert float(np.mean(th_w[behind])) < 0.0

    # Extremum positions: negative minimum behind bunch (zeta < 0), positive maximum ahead (zeta > 0)
    zeta_sim_min = float(z_w[int(np.argmin(sim_w))])
    zeta_sim_max = float(z_w[int(np.argmax(sim_w))])
    zeta_th_min = float(z_w[int(np.argmin(th_w))])
    zeta_th_max = float(z_w[int(np.argmax(th_w))])
    assert -3.2e-3 <= zeta_sim_min <= -1.5e-3
    assert -3.2e-3 <= zeta_th_min <= -1.5e-3
    assert 1.5e-3 <= zeta_sim_max <= 3.2e-3
    assert 1.5e-3 <= zeta_th_max <= 3.2e-3

    # Normalized shape correlation and unit-normalized L2 difference
    sim_unit = sim_w / np.linalg.norm(sim_w)
    th_unit = th_w / np.linalg.norm(th_w)
    corr = float(np.dot(sim_unit, th_unit))
    shape_l2_err = float(np.linalg.norm(sim_unit - th_unit))
    rel_l2_err = float(np.linalg.norm(sim_w - th_w) / np.linalg.norm(th_w))
    assert corr > 0.985
    assert shape_l2_err < 0.15
    assert rel_l2_err < 0.15

    # Positive and negative lobe amplitudes within 5% tolerance
    assert float(np.max(sim_w)) == pytest.approx(float(np.max(th_w)), rel=0.05)
    assert float(np.min(sim_w)) == pytest.approx(float(np.min(th_w)), rel=0.05)


def test_dielectric_cone_angle_threshold_and_n1_mutation():
    """Verify 3D cone angle within discretization error, sub-threshold Ez(zeta) probe, and mutations."""
    # 1. Above threshold: beta = 0.90, n = 1.5 (beta * n = 1.35 > 1)
    sup = h.run_single_dielectric_case(
        beta=0.90,
        n_refr=1.5,
        dx_m=0.55e-3,
        dz_m=0.55e-3,
        record_spectrum=True,
        n_frames=0,
    )
    cos_expected = 1.0 / (0.90 * 1.5)
    assert sup["cone"]["cone_detected"] is True
    assert sup["cone"]["back_to_fwd_ratio"] > 25.0
    assert sup["cone"]["cos_theta_c"] == pytest.approx(cos_expected, rel=0.03)
    assert sup["dW_dz_sim_J_m"] == pytest.approx(sup["dW_dz_theory_J_m"], rel=0.06)

    # 2. Below threshold: beta = 0.60, n = 1.5 (beta * n = 0.90 < 1)
    sub = h.run_single_dielectric_case(
        beta=0.60,
        n_refr=1.5,
        dx_m=0.55e-3,
        dz_m=0.55e-3,
        record_spectrum=True,
        n_frames=0,
    )
    assert sub["cone"]["cone_detected"] is False
    assert sub["cone"]["back_to_fwd_ratio"] < 2.0
    # Stated floor: sub-threshold radiated energy per unit length < 2.5e-6 J/m (< 3e-3 of super-threshold)
    assert sub["dW_dz_sim_J_m"] < 2.5e-6
    assert sub["dW_dz_sim_J_m"] < 3.0e-3 * sup["dW_dz_sim_J_m"]

    # Off-axis sub-threshold Ez(zeta) probe vs analytical contracted Coulomb theory at r_probe ~ 3.5 mm
    r_probe_target = 3.5e-3
    ix_probe = int(np.argmin(np.abs(sub["x_m"] - r_probe_target)))
    r_probe_actual = float(abs(sub["x_m"][ix_probe]))
    zeta_sub = sub["z_m"] - sub["z_bunch_snap_m"]
    ez_sim_probe = sub["Ez_xz"][ix_probe, :]
    ez_th_probe = h.subthreshold_coulomb_field(
        r_m=r_probe_actual,
        zeta_m=zeta_sub,
        beta=0.60,
        n_refr=1.5,
        q_c=sub["q_c"],
        sigmaz_m=sub["sigmaz_m"],
    )["Ez"]
    _assert_coulomb_probe_matches_theory(zeta_sub, ez_sim_probe, ez_th_probe)

    # Mutation check: flipping either the simulated or analytic Ez(zeta) sign must fail the check
    with pytest.raises(AssertionError):
        _assert_coulomb_probe_matches_theory(zeta_sub, -ez_sim_probe, ez_th_probe)
    with pytest.raises(AssertionError):
        _assert_coulomb_probe_matches_theory(zeta_sub, ez_sim_probe, -ez_th_probe)

    # 3. Mutation check: setting n = 1.0 at the same beta = 0.90 must remove the Cherenkov cone
    mut_n1 = h.run_single_dielectric_case(
        beta=0.90,
        n_refr=1.0,
        dx_m=0.55e-3,
        dz_m=0.55e-3,
        record_spectrum=False,
        n_frames=0,
    )
    assert mut_n1["cone"]["cone_detected"] is False
    assert mut_n1["cone"]["back_to_fwd_ratio"] < 2.0
    assert np.isnan(mut_n1["cone"]["theta_c_deg"])


def test_vacuum_yee_vs_tete_numerical_cherenkov():
    """Verify TETE suppresses spurious numerical Cherenkov wake energy in vacuum (beta=0.999) relative to Yee."""
    r_yee = h.run_single_vacuum_scheme(
        scheme="yee",
        beta=0.999,
        dz_m=0.80e-3,
        sigmaz_m=1.8e-3,
        n_frames=0,
    )
    r_tete = h.run_single_vacuum_scheme(
        scheme="tete",
        beta=0.999,
        dz_m=0.80e-3,
        sigmaz_m=1.8e-3,
        n_frames=0,
    )
    assert r_yee["W_wake_J"] > 1.0e-11
    assert r_tete["W_wake_J"] < 1.0e-14
    assert r_tete["W_wake_J"] < 1.0e-3 * r_yee["W_wake_J"]
    assert r_tete["peak_Ez_wake_V_m"] < 1.0e-2 * r_yee["peak_Ez_wake_V_m"]


def test_precomputed_tutorial_data_and_figures():
    """Verify precomputed tutorial datasets, convergence trends, and rendered PNG/GIF outputs."""
    for name in ("dielectric", "dispersive", "numerical_cherenkov", "convergence"):
        assert (TUT_DIR / "data" / f"task_{name}.npz").exists()
        assert (TUT_DIR / "data" / f"task_{name}.json").exists()

    disp_summary = json.loads((TUT_DIR / "data" / "task_dispersive.json").read_text())
    assert disp_summary["stop_to_pass_ratio"] < 0.01

    conv_summary = json.loads((TUT_DIR / "data" / "task_convergence.json").read_text())
    cos_errs = conv_summary["cos_rel_err"]
    spec_errs = conv_summary["spec_l2_rel_err"]
    assert conv_summary["analysed_band_ghz"] == [12.0, 30.0]
    assert conv_summary["ppw_30ghz"][-1] >= 20.0
    assert cos_errs[-1] < cos_errs[0]
    assert cos_errs[-1] < 0.015
    assert spec_errs[-1] < 0.05

    diel_npz = np.load(TUT_DIR / "data" / "task_dielectric.npz")
    _assert_coulomb_probe_matches_theory(
        diel_npz["zeta_sub_m"],
        diel_npz["Ez_sub_probe_sim"],
        diel_npz["Ez_sub_probe_theory"],
    )

    expected_pngs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "01_geometry_and_mesh.png",
        "02_field_maps_cone_vs_coulomb.png",
        "03_cone_angle_vs_beta.png",
        "04_frank_tamm_spectrum.png",
        "05_dispersive_cherenkov_band.png",
        "06_numerical_cherenkov_yee_vs_tete.png",
        "07_resolution_convergence.png",
    ]
    for fname in expected_pngs:
        fpath = TUT_DIR / "figures" / fname
        assert fpath.exists(), f"Missing figure {fname}"
        assert fpath.stat().st_size > 15_000

    table_md = TUT_DIR / "figures" / "00_setup_table.md"
    assert table_md.exists(), "Missing setup table 00_setup_table.md"
    assert table_md.stat().st_size > 200

    expected_gifs = [
        "cherenkov_cone_beta09.gif",
        "cherenkov_sub_vs_super_threshold.gif",
        "numerical_cherenkov_yee_vs_tete.gif",
    ]
    for gname in expected_gifs:
        gpath = TUT_DIR / "figures" / gname
        assert gpath.exists(), f"Missing animation {gname}"
        assert 10_000 < gpath.stat().st_size < 5 * 1024 * 1024


def test_cli_list_tasks_and_plot(tmp_path):
    """Verify CLI --list-tasks and --plot execution."""
    res_list = subprocess.run(
        [sys.executable, str(TUT_DIR / "cherenkov_radiation.py"), "--list-tasks"],
        capture_output=True,
        text=True,
        check=True,
    )
    tasks = res_list.stdout.strip().splitlines()
    assert tasks == ["dielectric", "dispersive", "numerical_cherenkov", "convergence"]

    res_plot = subprocess.run(
        [
            sys.executable,
            str(TUT_DIR / "cherenkov_radiation.py"),
            "--plot",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[figure]" in res_plot.stdout
    assert (tmp_path / "00a_setup_geometry.png").exists()
    assert (tmp_path / "00b_setup_mesh_and_excitation.png").exists()
    assert (tmp_path / "04_frank_tamm_spectrum.png").exists()
    assert (tmp_path / "cherenkov_cone_beta09.gif").exists()
