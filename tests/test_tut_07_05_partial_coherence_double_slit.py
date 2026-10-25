# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Fast verification and mutation tests for Tutorial 7.5:
Young's double slit with partially coherent light (2D Wakis FIT).
"""

import os
import sys

import numpy as np
import pytest
from scipy.constants import c as c_light

from wakis.field_monitors import DFTMonitor
from wakis.sources import GaussianSource

TUTORIAL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "tutorials",
        "07_light_sources",
        "07-05_partial_coherence_double_slit",
    )
)
if TUTORIAL_DIR not in sys.path:
    sys.path.insert(0, TUTORIAL_DIR)

import _helpers as h  # noqa: E402


def test_van_cittert_zernike_analytic_and_2d_hankel():
    """Verify paraxial sinc and exact 2D Hankel van Cittert-Zernike coherence."""
    wavelength = 0.5e-6
    d = 2.0e-6
    L1 = 8.0e-6
    D0 = wavelength * L1 / d  # 2.0 um first coherence zero

    assert np.isclose(h.van_cittert_zernike_sinc(0.0, d, wavelength, L1), 1.0)
    assert np.isclose(
        h.van_cittert_zernike_sinc(0.5 * D0, d, wavelength, L1),
        2.0 / np.pi,
        atol=1e-12,
    )
    assert np.isclose(
        h.van_cittert_zernike_sinc(D0, d, wavelength, L1), 0.0, atol=1e-12
    )

    # Secondary lobe has negative degree of coherence (fringe phase reversal)
    D_sec = 1.4303 * D0
    mu_sec_sinc = h.van_cittert_zernike_sinc(D_sec, d, wavelength, L1)
    assert np.isclose(mu_sec_sinc, -0.2172336, atol=1e-5)

    # Compare 2D Hankel integral against paraxial sinc across [0, 1.5 * D0]
    D_grid = np.array([0.0, 0.4 * D0, 0.8 * D0, D0, D_sec])
    mu_sinc = h.van_cittert_zernike_sinc(D_grid, d, wavelength, L1)
    mu_hankel = h.van_cittert_zernike_2d_hankel(D_grid, d, wavelength, L1, n_quad=201)
    assert np.max(np.abs(mu_hankel - mu_sinc)) < 0.025
    assert mu_hankel[-1] < -0.18  # Confirms negative secondary lobe in 2D Green's function


def test_gsm_mercer_coherent_mode_decomposition_and_mehler_identity():
    """Verify Hermite-Gaussian L2 orthonormality and Mehler's GSM identity."""
    w_m = 1.4e-6
    x = np.linspace(-8.0 * w_m, 8.0 * w_m, 1601)
    dx = x[1] - x[0]

    modes = np.array([h.gsm_hermite_mode(m, x, w_m=w_m) for m in range(6)])
    gram = (modes @ modes.T) * dx
    assert np.allclose(gram, np.eye(6), atol=1e-6)

    # Reconstruct cross-spectral density W(x1, x2) via Starikov & Wolf Mercer sum
    sigma_s = 1.8e-6
    sigma_mu = 1.5e-6
    params = h.gsm_mercer_params(sigma_s, sigma_mu, n_modes=18, i0=1.0)
    x_sub = np.linspace(-3.0e-6, 3.0e-6, 61)
    phi_sub = np.array(
        [h.gsm_hermite_mode(m, x_sub, w_m=params["w_m"]) for m in range(18)]
    )
    w_mercer = (params["lambdas"][:, None, None] * phi_sub[:, :, None] * phi_sub[:, None, :]).sum(
        axis=0
    )

    X1, X2 = np.meshgrid(x_sub, x_sub, indexing="ij")
    w_exact = np.exp(-(X1**2 + X2**2) / (4.0 * sigma_s**2)) * np.exp(
        -((X1 - X2) ** 2) / (2.0 * sigma_mu**2)
    )
    rel_err = np.linalg.norm(w_mercer - w_exact) / np.linalg.norm(w_exact)
    assert rel_err < 1e-4

    # Verify modal slit coherence matches exp(-d^2 / (2 * sigma_mu^2))
    d = 2.0e-6
    for ratio in [0.5, 1.0, 2.0, np.inf]:
        smu = ratio * d if np.isfinite(ratio) else np.inf
        v_ana = h.gsm_visibility_analytic(d, smu)
        v_mod = h.gsm_modal_slit_coherence(smu, d=d, w_m=w_m, n_modes=12)
        assert abs(v_mod - v_ana) < 2e-3


def test_extract_fringe_visibility_and_phase_reversal():
    """Verify fringe visibility extraction across positive, zero, and negative mu_12."""
    x_det = np.linspace(-4.5e-6, 4.5e-6, 181)
    for mu_target in [1.0, 0.65, 0.25, 0.0, -0.217]:
        intensity, _ = h.double_slit_2d_hankel_intensity(x_det, mu_target)
        vis, mu_fit = h.extract_fringe_visibility(x_det, intensity)
        assert abs(vis - abs(mu_target)) < 0.015
        assert abs(mu_fit - mu_target) < 0.015


def test_wakis_2d_double_slit_spatial_and_temporal_coherence():
    """
    Run a 2D Wakis FIT double-slit simulation with symmetric (m=0) and
    antisymmetric (m=1) Hermite-Gaussian modes and verify both spatial
    incoherent-mode superposition and broadband Wiener-Khinchin temporal coherence.
    """
    wavelength = 0.5e-6
    f0 = c_light / wavelength
    d = 2.0e-6
    a = 0.4e-6
    L2 = 6.0e-6
    w_m = 1.4e-6
    cpw = 10

    freqs = np.linspace(0.75 * f0, 1.25 * f0, 41)
    ic = int(np.argmin(np.abs(freqs - f0)))
    pulse = GaussianSource(f0, 0.22 * f0)

    dft_det_modes = []
    slit1_modes = []
    slit2_modes = []
    x_det = None

    for m in (0, 1):
        solver, grid, _ = h.build_double_slit_solver(
            x_bounds=(-4.5e-6, 4.5e-6),
            y_bounds=(-1.5e-6, 6.4e-6),
            wavelength=wavelength,
            cpw=cpw,
            n_pml=8,
            slit_sep=d,
            slit_width=a,
            screen_thickness=0.2e-6,
            screen_material="pec",
        )
        n_pml = 8
        ix_src = np.arange(n_pml, grid.Nx - n_pml)
        x_src = grid.x[ix_src]
        iy_src = int(np.argmin(np.abs(grid.y - (-1.0e-6))))
        iy_slit = int(np.argmin(np.abs(grid.y - 0.0)))
        iy_det = int(np.argmin(np.abs(grid.y - L2)))
        ix_s1 = int(np.argmin(np.abs(grid.x - (-0.5 * d))))
        ix_s2 = int(np.argmin(np.abs(grid.x - (+0.5 * d))))

        phi_m = h.gsm_hermite_mode(m, x_src, w_m=w_m)
        src = h.CustomLineSource(ix_src, iy_src, phi_m, pulse)

        mon_det = DFTMonitor(
            freqs,
            components=("Ez",),
            center=(0.0, L2, 0.0),
            size=(9.0e-6, 0.0, 0.0),
        )
        mon_slit = DFTMonitor(
            freqs,
            components=("Ez",),
            center=(0.0, 0.0, 0.0),
            size=(9.0e-6, 0.0, 0.0),
        )
        t_end = pulse.t0 + 4.5 / (0.22 * f0) + (8.5e-6 / c_light) * 1.4
        n_steps = int(np.ceil(t_end / solver.dt))
        solver.emsolve(n_steps, source=src, monitors=[mon_det, mon_slit])

        ez_det, coords_det = mon_det.get_field("Ez")
        ez_slit, coords_slit = mon_slit.get_field("Ez")
        dft_det_modes.append(ez_det[:, :, 0, 0])
        x_slit = coords_slit[0]
        ix_s1 = int(np.argmin(np.abs(x_slit - (-0.5 * d))))
        ix_s2 = int(np.argmin(np.abs(x_slit - (+0.5 * d))))
        slit1_modes.append(ez_slit[:, ix_s1, 0, 0])
        slit2_modes.append(ez_slit[:, ix_s2, 0, 0])
        if x_det is None:
            x_det = coords_det[0]

    # Normalize modes so each delivers unit amplitude at the slit centers across frequency
    scale_0 = 1.0 / np.abs(slit1_modes[0])[:, None]
    scale_1 = 1.0 / np.abs(slit1_modes[1])[:, None]
    ez0_f0 = (dft_det_modes[0] * scale_0)[ic]
    ez1_f0 = (dft_det_modes[1] * scale_1)[ic]
    fields_norm = np.stack([ez0_f0, ez1_f0], axis=0)

    # 1. Spatial coherence test: synthesizing target degree of coherence mu_12 in {1.0, 0.5, 0.0}
    for mu_target in (1.0, 0.5, 0.0):
        w0 = 0.5 * (1.0 + mu_target)
        w1 = 0.5 * (1.0 - mu_target)
        i_screen = h.incoherent_power_sum(fields_norm, weights=[w0, w1])
        vis_sim, mu_sim = h.extract_fringe_visibility(
            x_det, i_screen, d=d, L2=L2, wavelength=wavelength
        )
        assert abs(vis_sim - mu_target) < 0.025
        assert abs(mu_sim - mu_target) < 0.025

    # 2. Temporal coherence test: polychromatic screen profile vs Wiener-Khinchin envelope
    rel_bw = 0.08
    i_poly, i_inc, contrast = h.polychromatic_screen_intensity(
        freqs,
        dft_det_modes[0] * scale_0,
        dft_det_modes[1] * scale_1,
        pulse,
        f0,
        rel_bw_sigma=rel_bw,
        x_det=x_det,
    )
    delta_L = np.hypot(x_det + 0.5 * d, L2) - np.hypot(x_det - 0.5 * d, L2)
    gamma_wk, _ = h.temporal_coherence_envelope(
        delta_L, wavelength_0=wavelength, rel_bw_sigma=rel_bw
    )
    # Match effective geometric path-difference scale including finite-thickness PEC slit guide phase
    vis0, _ = h.extract_fringe_visibility(
        x_det, (dft_det_modes[0] * scale_0)[ic].real ** 2 + (dft_det_modes[0] * scale_0)[ic].imag ** 2,
        d=d, L2=L2, wavelength=wavelength,
    )
    k0 = 2.0 * np.pi / wavelength
    contrast_wk = vis0 * gamma_wk * np.cos(k0 * delta_L)
    mask = np.abs(x_det) <= 3.0e-6
    # Compare envelope decay at the m=0 peak (delta_L=0) and m=2 peak (|delta_L|=2*lambda_0)
    assert contrast[int(np.argmin(np.abs(x_det - 0.0)))] > 0.94
    gamma_m2_ana, _ = h.temporal_coherence_envelope(
        2.0 * wavelength, wavelength_0=wavelength, rel_bw_sigma=rel_bw
    )
    peak_m2_sim = float(np.max(contrast[np.abs(x_det) > 2.8e-6]))
    assert abs(peak_m2_sim - gamma_m2_ana) < 0.025
    rmse = np.sqrt(np.mean((contrast[mask] - contrast_wk[mask]) ** 2))
    assert rmse < 0.09


def test_mutation_coherent_field_sum_and_corrupted_weights_fail():
    """
    Mutation checks verifying that tests detect and fail on:
    (1) replacing incoherent power summation with coherent complex field addition
        across an extended van Cittert-Zernike strip source at D = D_0, and
    (2) dropping the (-1)^m parity sign in the Mercer slit coherence expansion.
    """
    x_det = np.linspace(-4.5e-6, 4.5e-6, 181)
    d = 2.0e-6
    L1 = 8.0e-6
    L2 = 6.0e-6
    wavelength = 0.5e-6
    k0 = 2.0 * np.pi / wavelength
    D0 = wavelength * L1 / d  # First van Cittert-Zernike coherence zero (2.0 um)

    # 21 point emitters uniformly spaced across D0 at y = -L1
    n_src = 21
    xs = ((np.arange(n_src) + 0.5) / n_src - 0.5) * D0
    r1_src = np.hypot(xs + 0.5 * d, L1)
    r2_src = np.hypot(xs - 0.5 * d, L1)
    a1 = np.exp(-1j * k0 * r1_src) / np.sqrt(r1_src)
    a2 = np.exp(-1j * k0 * r2_src) / np.sqrt(r2_src)

    r1_det = np.hypot(x_det + 0.5 * d, L2)
    r2_det = np.hypot(x_det - 0.5 * d, L2)
    g1 = np.exp(-1j * k0 * r1_det) / np.sqrt(r1_det)
    g2 = np.exp(-1j * k0 * r2_det) / np.sqrt(r2_det)

    # Field on detector screen for each independent emitter realization
    dipole_fields = a1[:, None] * g1[None, :] + a2[:, None] * g2[None, :]

    # True incoherent power sum at D = D0 washes out fringes (V ~ 0)
    i_incoherent = h.incoherent_power_sum(dipole_fields)
    vis_inc, _ = h.extract_fringe_visibility(
        x_det, i_incoherent, d=d, L2=L2, wavelength=wavelength
    )
    assert vis_inc < 0.02

    # MUTATION 1: Replacing incoherent_power_sum with coherent_field_sum_intensity
    # Coherently summing all emitters in one simulation produces V = 1.0 instead of 0.0!
    i_mutated_coherent = h.coherent_field_sum_intensity(dipole_fields)
    vis_mut, _ = h.extract_fringe_visibility(
        x_det, i_mutated_coherent, d=d, L2=L2, wavelength=wavelength
    )
    assert vis_mut > 0.95
    with pytest.raises(AssertionError):
        assert vis_mut < 0.05

    # MUTATION 2: Corrupting the (-1)^m parity in Mercer slit coherence
    sigma_mu = 1.0e-6  # sigma_mu / d = 0.5 -> true V = exp(-2) = 0.1353
    v_true = h.gsm_visibility_analytic(d, sigma_mu)
    info = h.gsm_parameters_from_modal_scale(sigma_mu, w_m=1.4e-6, n_modes=10)
    phi_vals = np.array([h.gsm_hermite_mode(m, 0.5 * d, w_m=1.4e-6) for m in range(10)])
    mu_mutated = float(
        np.sum(info["weights"] * (phi_vals**2)) / np.sum(info["weights"] * (phi_vals**2))
    )
    with pytest.raises(AssertionError):
        assert abs(mu_mutated - v_true) < 0.05


def test_silver_drude_material_and_solver_builder():
    """Verify Drude silver permittivity at 500 nm and dispersive 2D solver stability."""
    wavelength = 0.5e-6
    f0 = c_light / wavelength
    mat, skin_depth = h.silver_drude_material(wavelength)
    eps_drude = complex(mat.epsilon(f0))
    eps_rakic = complex(h.dispersive_lib["silver"].epsilon(f0))

    assert np.isclose(eps_drude, eps_rakic, rtol=1e-6)
    assert 25.0e-9 < skin_depth < 32.0e-9

    solver, grid, boxes = h.build_double_slit_solver(
        x_bounds=(-2.5e-6, 2.5e-6),
        y_bounds=(-1.0e-6, 1.5e-6),
        wavelength=wavelength,
        cpw=10,
        screen_material="silver",
    )
    assert len(boxes) == 3
    res_ag = h.measure_mesh_resolution(grid, wavelength=wavelength, screen_material="silver")
    assert res_ag["cells_per_skin_depth"] >= 8.0
    assert res_ag["cells_per_skin_depth_power"] >= 4.0
    pulse = GaussianSource(f0, 0.2 * f0)
    src = h.CustomLineSource(grid.Nx // 2, grid.Ny // 3, 1.0, pulse)
    solver.emsolve(40, source=src)
    assert np.all(np.isfinite(solver.E[:, :, 0, "z"]))


def _assert_mesh_audit_satisfies_rules(audit, require_20cpw=False):
    """Assert that a single mesh audit dictionary satisfies Rules 1, 2, and 3."""
    min_cpw = 20.0 if require_20cpw else 10.0
    assert audit["cpw_shortest_wavelength"] >= min_cpw - 1e-3, (
        f"Rule 1 violated: cpw_shortest_wavelength={audit['cpw_shortest_wavelength']} < {min_cpw}"
    )
    assert audit["cells_across_feature"] >= 4, (
        f"Rule 2 violated: cells_across_feature={audit['cells_across_feature']} < 4"
    )
    assert audit["cells_across_slit"] >= 4, (
        f"Rule 2 violated: cells_across_slit={audit['cells_across_slit']} < 4"
    )
    if "cells_per_skin_depth" in audit:
        assert audit["cells_per_skin_depth"] >= 8.0 - 5e-2, (
            f"Rule 3 violated: cells_per_skin_depth (delta_E)={audit['cells_per_skin_depth']} < 8.0"
        )
        assert audit["cells_per_skin_depth_power"] >= 4.0 - 3e-2, (
            f"Rule 3 violated: cells_per_skin_depth_power (delta_I)={audit['cells_per_skin_depth_power']} < 4.0"
        )


def test_task_tables_satisfy_mesh_resolution_rules_and_mutation():
    """
    Build the mesh (without running the solver) for every non-demo task in both
    --quick and --full task tables and verify Rules 1-3, plus mutation checks.
    """
    import partial_coherence_double_slit as tut

    for quick in (True, False):
        tasks = tut.get_task_table(quick=quick)
        demo_tasks = [t for t in tasks if t.get("demo", False)]
        assert len(demo_tasks) == 1
        assert demo_tasks[0]["name"] == "silver_skin_depth_demo"

        for task in tasks:
            audits = tut.build_task_meshes(task)
            assert len(audits) >= 1
            if task.get("demo", False):
                # The demo task starts with under-resolved uniform grids (0.58 and 1.15 cpsd_E)
                with pytest.raises(AssertionError, match="Rule 3 violated"):
                    _assert_mesh_audit_satisfies_rules(audits[0])
                continue
            require_20 = (not quick) and (task["kind"] != "conv")
            for audit in audits:
                _assert_mesh_audit_satisfies_rules(audit, require_20cpw=require_20)

    # MUTATION 1: A coarse cpw=8 mesh violates Rule 1 (< 10 cpw)
    _, bad_grid_cpw8, _ = h.build_double_slit_solver(
        x_bounds=(-5.0e-6, 5.0e-6),
        y_bounds=(-1.6e-6, 6.4e-6),
        wavelength=0.5e-6,
        cpw=8,
        screen_material="pec",
        cells_per_skin_depth="uniform",
        grid_only=True,
    )
    bad_audit_cpw8 = h.measure_mesh_resolution(bad_grid_cpw8, wavelength=0.5e-6, screen_material="pec")
    bad_audit_cpw8["cpw_shortest_wavelength"] = round(500.0 / bad_audit_cpw8["dx_background_nm"], 2)
    with pytest.raises(AssertionError, match="Rule 1 violated"):
        _assert_mesh_audit_satisfies_rules(bad_audit_cpw8)

    # MUTATION 2: An unrefined uniform cpw=10 mesh with a 100 nm screen (2 cells across feature) violates Rule 2
    _, bad_grid_feat, _ = h.build_double_slit_solver(
        x_bounds=(-5.0e-6, 5.0e-6),
        y_bounds=(-1.6e-6, 6.4e-6),
        wavelength=0.5e-6,
        cpw=10,
        screen_thickness=0.10e-6,
        screen_material="pec",
        cells_per_skin_depth="uniform",
        grid_only=True,
    )
    bad_audit_feat = h.measure_mesh_resolution(
        bad_grid_feat, wavelength=0.5e-6, screen_thickness=0.10e-6, screen_material="pec"
    )
    bad_audit_feat["cpw_shortest_wavelength"] = round(500.0 / bad_audit_feat["dx_background_nm"], 2)
    with pytest.raises(AssertionError, match="Rule 2 violated"):
        _assert_mesh_audit_satisfies_rules(bad_audit_feat)

    # MUTATION 3: An unrefined silver mesh at cpw=10 (0.58 cells per delta_E) violates Rule 3
    _, bad_grid_ag, _ = h.build_double_slit_solver(
        x_bounds=(-4.5e-6, 4.5e-6),
        y_bounds=(-1.5e-6, 6.3e-6),
        wavelength=0.5e-6,
        cpw=10,
        screen_material="silver",
        cells_per_skin_depth="uniform",
        grid_only=True,
    )
    bad_audit_ag = h.measure_mesh_resolution(bad_grid_ag, wavelength=0.5e-6, screen_material="silver")
    bad_audit_ag["cpw_shortest_wavelength"] = round(500.0 / bad_audit_ag["dx_background_nm"], 2)
    with pytest.raises(AssertionError, match="Rule 3 violated"):
        _assert_mesh_audit_satisfies_rules(bad_audit_ag)


def test_full_study_summary_metrics_and_convergence_audit():
    """
    Verify that the full-resolution data/summary_metrics.json satisfies all
    analytical coherence benchmarks, skin-depth and spatial convergence criteria,
    and mesh resolution rules (with mutation check).
    """
    import json

    summary_path = os.path.join(TUTORIAL_DIR, "data", "summary_metrics.json")
    assert os.path.isfile(summary_path)
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)

    # 1. Method A (VCZ): agreement with 2D Hankel and phase reversal in secondary lobe
    vcz = summary["vcz"]
    mu_sim = np.asarray(vcz["mu_sim"], dtype=float)
    mu_hankel = np.asarray(vcz["mu_hankel"], dtype=float)
    vis_sim = np.asarray(vcz["vis_sim"], dtype=float)
    u_sweep = np.asarray(vcz["u_sweep"], dtype=float)
    assert np.max(np.abs(mu_sim - mu_hankel)) < 0.02
    idx_null = int(np.argmin(np.abs(u_sweep - 1.0)))
    assert vis_sim[idx_null] < 0.03
    idx_sec = int(np.argmin(np.abs(u_sweep - 1.4)))
    assert mu_sim[idx_sec] < -0.19

    # 2. Method B (GSM): agreement with analytical exp(-d^2 / (2*sigma_mu^2))
    gsm = summary["gsm"]
    vis_gsm_sim = np.asarray(gsm["vis_gsm_sim"], dtype=float)
    vis_gsm_ana = np.asarray(gsm["vis_gsm_ana"], dtype=float)
    assert np.max(np.abs(vis_gsm_sim - vis_gsm_ana)) < 0.012

    # 3. Silver vs PEC and skin-depth convergence study
    ag = summary["silver_vs_pec"]
    ag_study = summary["silver_skin_depth_study"]
    assert ag["cells_per_skin_depth"] >= 10.0
    assert ag["cells_per_skin_depth_power"] >= 5.0
    assert 1.04 < ag["peak_ratio_silver_to_pec"] < 1.07
    assert ag_study["last_step_rel_change_pct"] < 0.1  # 0.0461% << 1% threshold
    r_ladder = np.asarray(ag_study["ratio_ag_to_pec"], dtype=float)
    # Ratio decreases monotonically as skin-depth collar is refined
    assert np.all(np.diff(r_ladder) < 0.0)

    # 4. Spatial mesh convergence (cpw = [10, 14, 20] vs ref = 40)
    conv = summary["convergence"]
    assert conv["cpw_ref"] == 40
    l2_pc = np.asarray(conv["l2_err_pc"], dtype=float)
    assert np.all(np.diff(l2_pc) < 0.0)
    assert l2_pc[-1] < 0.015  # 1.42% at cpw=20
    assert conv["vis_err_pc"][-1] < 0.001

    # 5. Task resolution rules on saved NPZ metadata
    for audit in summary["task_resolutions"]:
        if audit["demo"]:
            with pytest.raises(AssertionError, match="Rule 3 violated"):
                _assert_mesh_audit_satisfies_rules(audit)
        else:
            req_20 = audit["name"] != "convergence_coarse"
            _assert_mesh_audit_satisfies_rules(audit, require_20cpw=req_20)


def test_setup_schematic_and_expected_figures_exist():
    """Verify that 00a/00b setup figures, 00_setup_table.md, and all expected tutorial figures exist and are non-empty."""
    fig_dir = os.path.join(TUTORIAL_DIR, "figures")
    expected = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00a_setup_geometry_gsm_silver.png",
        "00b_setup_mesh_and_excitation_gsm_silver.png",
        "01_double_slit_schematic.png",
        "02_grid_and_slit_zoom.png",
        "03_gsm_mercer_eigenmodes.png",
        "04_coherent_vs_incoherent_2d_intensity.png",
        "04b_coherent_vs_incoherent_error_2d.png",
        "05_field_snapshots_ez.png",
        "06_poynting_and_slit_profile.png",
        "07_double_slit_propagation.gif",
        "08_vcz_fringe_profiles_vs_D.png",
        "09_vcz_visibility_vs_source_width.png",
        "10_gsm_fringe_profiles_and_visibility.png",
        "11_temporal_coherence_broadband.png",
        "12_ensemble_and_mode_convergence.png",
        "13_pec_vs_silver_screen.png",
        "14_mesh_resolution_convergence.png",
        "15_silver_skin_depth_resolution.png",
    ]
    for fname in expected:
        fpath = os.path.join(fig_dir, fname)
        assert os.path.isfile(fpath), f"Missing figure: {fname}"
        assert os.path.getsize(fpath) > 1000, f"Empty/corrupt figure: {fname}"
    for tname in ("00_setup_table.md", "00_setup_table_gsm_silver.md"):
        tpath = os.path.join(fig_dir, tname)
        assert os.path.isfile(tpath), f"Missing setup table: {tname}"
        assert os.path.getsize(tpath) > 100, f"Empty/corrupt setup table: {tname}"
    gif_path = os.path.join(fig_dir, "07_double_slit_propagation.gif")
    assert os.path.getsize(gif_path) < 5 * 1024 * 1024, "GIF exceeds 5 MB limit"



