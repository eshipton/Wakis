"""Fast analytical, numerical, and mutation tests for Tutorial 7.2 SLED tilted facet."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "07_light_sources"
    / "07-02_sled_tilted_facet"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as h  # noqa: E402
import sled_tilted_facet as tut  # noqa: E402


def _assert_marcuse_and_ar_physics(
    r00_func,
    r01_func,
    t_ar_val: float,
    eta_te1_val: float,
) -> None:
    """Validation oracle that raises AssertionError if any physics invariant is violated."""
    neff = 3.3276
    w0 = 1.293e-6
    wl = 1.49e-6
    r_f0 = ((neff - 1.0) / (neff + 1.0)) ** 2

    # 1. Normal incidence matches Fresnel reflectance and zero odd-mode conversion
    r00_0 = float(r00_func(0.0, neff=neff, w0=w0, wavelength=wl))
    r01_0 = float(r01_func(0.0, neff=neff, w0=w0, wavelength=wl))
    assert np.isclose(r00_0, r_f0, rtol=1e-6), f"Expected R_f={r_f0}, got {r00_0}"
    assert abs(r01_0) < 1e-15, f"Expected R_01(0)=0, got {r01_0}"

    # 2. At u = 1, R_00(theta_1) == R_01(theta_1) == R_f(theta_1) / e
    theta_u1 = np.rad2deg(np.arctan(wl / (2.0 * np.pi * neff * w0)))
    r_f_u1 = float(h.transfer_matrix_ar_reflectance(theta_u1, n_inc=neff, wavelength=wl))
    r00_u1 = float(r00_func(theta_u1, neff=neff, w0=w0, wavelength=wl))
    r01_u1 = float(r01_func(theta_u1, neff=neff, w0=w0, wavelength=wl))
    assert np.isclose(r00_u1, r_f_u1 / np.e, rtol=1e-6)
    assert np.isclose(r01_u1, r_f_u1 / np.e, rtol=1e-6)

    # 3. At 7 deg, R_00 drops by ~2 orders of magnitude into the [1e-3, 4e-3] range
    r00_7 = float(r00_func(7.0, neff=neff, w0=w0, wavelength=wl))
    assert 1.0e-3 < r00_7 < 4.0e-3, f"R_00(7 deg) out of physical range: {r00_7}"

    # 4. Quarter-wave AR coating suppresses normal reflectance by >50x
    r_ar_0 = float(
        h.transfer_matrix_ar_reflectance(
            0.0, n_inc=neff, wavelength=wl, n_ar=h.N_AR_COAT, t_ar=t_ar_val
        )
    )
    assert r_ar_0 < 0.02 * r_f0, f"AR coating failed to suppress reflection: {r_ar_0}"

    # 5. Incoherent dipole ensemble excites odd TE1 mode with substantial power (>25%)
    assert 0.25 < eta_te1_val < 0.65, f"Unphysical TE1 modal fraction: {eta_te1_val}"


def test_marcuse_gaussian_r00_and_r01_identities():
    """Verify Marcuse (1989) Gaussian-mode R_00 and R_01 analytical identities."""
    slab_sm = h.solve_1d_slab_modes(width=h.WIDTH_SM)
    neff = slab_sm["neff_0"]
    w0 = slab_sm["w0_overlap"]

    _assert_marcuse_and_ar_physics(
        r00_func=h.marcuse_gaussian_r00,
        r01_func=h.marcuse_gaussian_r01,
        t_ar_val=h.T_AR_COAT,
        eta_te1_val=0.43,
    )

    # Verify R_01 / R_00 == u^2 across non-zero tilt angles
    thetas = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    u = 2.0 * np.pi * neff * w0 * np.tan(np.deg2rad(thetas)) / h.WAVELENGTH_0
    ratio = h.marcuse_gaussian_r01(thetas, neff=neff, w0=w0) / h.marcuse_gaussian_r00(
        thetas, neff=neff, w0=w0
    )
    np.testing.assert_allclose(ratio, u**2, rtol=1e-12)


def test_slab_dispersion_and_marcuse_interference_null():
    """Verify 1D step-index slab cutoffs, mode orthogonality, and 10 deg null."""
    slab_sm = h.solve_1d_slab_modes(width=h.WIDTH_SM)
    assert slab_sm["v_half"] < 0.5 * np.pi
    assert slab_sm["psi_1"] is None
    assert h.N_CLAD < slab_sm["neff_0"] < h.N_CORE
    assert slab_sm["max_gauss_overlap"] > 0.99

    # Marcuse (1989) step-index slab exhibits a sidewall interference dip around 10 deg
    r_slab = h.slab_overlap_reflectivity(np.array([0.0, 7.0, 10.0, 12.0]), slab_sm)
    assert np.isclose(r_slab[0], ((slab_sm["neff_0"] - 1.0) / (slab_sm["neff_0"] + 1.0)) ** 2)
    assert 1.5e-3 < r_slab[1] < 3.5e-3
    # Non-monotonic sidelobe: R_slab(10 deg) < R_slab(12 deg)
    assert r_slab[2] < 1.0e-5
    assert r_slab[2] < 0.25 * r_slab[3]

    # Two-mode waveguide (w2 = 2.80 um) supports orthogonal TE0 and TE1 modes
    slab_mm = h.solve_1d_slab_modes(width=h.WIDTH_MM)
    assert 0.5 * np.pi < slab_mm["v_half"] < np.pi
    assert slab_mm["psi_1"] is not None
    ortho = float(np.trapezoid(slab_mm["psi_0"] * slab_mm["psi_1"], slab_mm["y"]))
    assert abs(ortho) < 1e-12
    assert h.slab_overlap_reflectivity(0.0, slab_mm, mode_out=1) < 1e-24
    # At 7 deg in the bimodal guide, odd-mode conversion R_01 exceeds R_00
    r00_7_mm = float(h.slab_overlap_reflectivity(7.0, slab_mm, mode_out=0))
    r01_7_mm = float(h.slab_overlap_reflectivity(7.0, slab_mm, mode_out=1))
    assert r01_7_mm > 5.0 * r00_7_mm


def test_ar_coating_snell_and_hakki_paoli():
    """Verify ideal quarter-wave AR null, Snell refraction, and Hakki-Paoli ripple."""
    neff = 3.3276
    n_ideal = np.sqrt(neff)
    t_ideal = h.WAVELENGTH_0 / (4.0 * n_ideal)
    r_ideal = float(
        h.transfer_matrix_ar_reflectance(
            0.0, n_inc=neff, wavelength=h.WAVELENGTH_0, n_ar=n_ideal, t_ar=t_ideal
        )
    )
    assert r_ideal < 1e-28

    # Snell deflection for Zah et al. (1987): 7 deg tilt -> 24.0 deg in air
    th_out = float(h.snell_farfield_angle_deg(7.0, neff=h.N_CORE))
    assert np.isclose(th_out, 24.0, atol=0.1)

    # Hakki-Paoli ripple identities
    assert float(h.hakki_paoli_ripple_db(0.0)) == 0.0
    assert np.isclose(float(h.hakki_paoli_ripple_db(0.5)), 20.0 * np.log10(3.0), rtol=1e-12)

    wls = np.linspace(1.480e-6, 1.484e-6, 1001)
    _, _, rip_bare = h.sled_ase_spectrum(wls, g_peak_linear=120.0, r1=3e-3, r2=3e-3)
    _, _, rip_ar = h.sled_ase_spectrum(wls, g_peak_linear=120.0, r1=3e-3, r2=3e-5)
    assert rip_bare > 5.0
    assert rip_ar < 1.0


def _assert_mesh_resolution_rules(
    cpw: int,
    metrics: dict,
    lambda_min: float = 1.43e-6,
    n_max: float = h.N_CORE,
) -> None:
    """Assert that mesh metrics satisfy Rules 1 and 2 for Tutorial 7.2."""
    dx_rule1_max_nm = (lambda_min / (10.0 * n_max)) * 1e9
    assert int(cpw) >= 10, f"Rule 1 violated: cpw={cpw} < 10"
    assert (
        metrics["dx_background_nm"] <= dx_rule1_max_nm
    ), f"Rule 1 violated: dx_background_nm={metrics['dx_background_nm']:.3f} > {dx_rule1_max_nm:.3f}"
    assert (
        metrics["dx_min_nm"] <= metrics["dx_background_nm"] + 1e-9
    ), f"Expected dx_min_nm <= dx_background_nm, got {metrics}"
    assert (
        metrics["cells_across_feature"] >= 4
    ), f"Rule 2 violated: cells_across_feature={metrics['cells_across_feature']} < 4"
    assert (
        metrics["dx_min_nm"] <= 0.25 * metrics["min_feature_nm"] + 1e-6
    ), f"Rule 2 violated: dx_min_nm={metrics['dx_min_nm']:.3f} > min_feature_nm/4"
    if "x_node_err_nm" in metrics:
        assert (
            metrics["x_node_err_nm"] < 1e-3
        ), f"AR coating interfaces not snapped to mesh nodes: err={metrics['x_node_err_nm']} nm"
    assert int(metrics["n_cells"]) > 0
    assert int(metrics["n_steps"]) > 0


def test_quick_and_full_task_tables_satisfy_mesh_rules():
    """Verify every non-demo task in --quick and --full satisfies Rules 1-2 (mesh only)."""
    for quick_mode in (True, False):
        tasks = tut.get_task_table(quick=quick_mode)
        assert len(tasks) == 11
        for task in tasks:
            if task.get("demo_under_resolved", False):
                continue
            cpw = int(task["cpw"])
            kind = task["kind"]
            width = h.WIDTH_MM if kind == "bimodal_sweep" else h.WIDTH_SM
            include_facet = kind != "pml_incoherent"
            ar_coated = kind == "ar_sweep"
            averaging = str(task.get("averaging", "kottke"))

            sim, geom_info = h.create_tilted_facet_simulation(
                theta_deg=float(task.get("theta_deg", 7.0 if include_facet else 0.0)),
                width=width,
                cpw=cpw,
                ar_coated=ar_coated,
                averaging=averaging,
                include_facet=include_facet,
            )
            # Ensure we only build 1D axes and never allocate the 3D solver
            assert sim.grid is None
            metrics = h.inspect_simulation_mesh(sim, min_feature=geom_info["t_ar_eff"])
            assert sim.grid is None
            _assert_mesh_resolution_rules(cpw=cpw, metrics=metrics)

    # Also verify saved full-study JSON and NPZ files in data/
    import json

    data_dir = TUTORIAL_DIR / "data"
    for task in tut.get_task_table(quick=False):
        json_path = data_dir / f"task_{task['index']:02d}.json"
        assert json_path.exists(), f"Missing {json_path.name}"
        saved = json.loads(json_path.read_text())
        assert int(saved["task"]["cpw"]) == int(task["cpw"])
        for req_key in (
            "dx_background_nm",
            "dx_min_nm",
            "min_feature_nm",
            "cells_across_feature",
            "n_cells",
            "n_steps",
        ):
            assert req_key in saved, f"Missing {req_key} in {json_path.name}"
        _assert_mesh_resolution_rules(cpw=int(saved["task"]["cpw"]), metrics=saved)

    # Verify auto-detection and full-study convergence & physics invariants from data/
    loaded = tut.load_all_results(data_dir)
    assert loaded["tasks"][0]["cpw"] == 20
    assert loaded["tasks"][-1]["cpw"] == 40
    slab_sm = h.solve_1d_slab_modes(width=h.WIDTH_SM)
    r7_slab = float(h.slab_overlap_reflectivity(7.0, slab_sm, mode_out=0))
    r7_fit = float(loaded["json"]["bare_tilt_sweep"]["r00_7deg"])
    assert abs(r7_fit - r7_slab) / r7_slab < 0.03
    assert float(loaded["json"]["bare_tilt_sweep"]["r00_10deg"]) < 1e-7
    r7_ar = float(loaded["json"]["ar_tilt_sweep"]["r00_ar_7deg"])
    assert r7_fit / r7_ar > 75.0
    r7_k28 = float(loaded["json"]["kottke_cpw28"]["r00_7deg"])
    r7_k40 = float(loaded["json"]["kottke_reference"]["r00_7deg"])
    assert abs(r7_k40 - r7_k28) / r7_k40 < 0.01
    neff_k40 = float(loaded["json"]["kottke_reference"]["neff_0"])
    assert abs(neff_k40 - slab_sm["neff_0"]) / slab_sm["neff_0"] < 0.001


def test_wakis_2d_fit_tilted_facet_suppression():
    """Run a fast 2D Wakis FIT check comparing theta_t = 0 deg and 7 deg at cpw = 10."""
    res_0 = h.run_tilted_facet_fit(theta_deg=0.0, width=h.WIDTH_SM, cpw=10, n_pml=8)
    res_7 = h.run_tilted_facet_fit(theta_deg=7.0, width=h.WIDTH_SM, cpw=10, n_pml=8)

    _assert_mesh_resolution_rules(cpw=10, metrics=res_0)
    _assert_mesh_resolution_rules(cpw=10, metrics=res_7)

    # Normal facet reflects ~28-30% of guided mode power
    assert 0.25 < res_0["r00"] < 0.34
    # 7 deg tilted facet suppresses guided reflection by ~80-100x into [1e-3, 5e-3]
    assert 1.0e-3 < res_7["r00"] < 5.0e-3
    assert res_7["r00"] < 0.02 * res_0["r00"]


def test_mutation_checks_detect_broken_physics():
    """Mutation check: verify our assertions fail when physical or mesh rules are broken."""
    # Mutation 1: Missing factor of 2 in Marcuse exponent (factor-of-4 error in u^2)
    def broken_r00_missing_two(theta_deg, neff, w0, wavelength=h.WAVELENGTH_0):
        u_wrong = np.pi * neff * w0 * np.tan(np.deg2rad(theta_deg)) / wavelength
        r_f = h.transfer_matrix_ar_reflectance(
            theta_deg, n_inc=neff, wavelength=wavelength
        )
        return r_f * np.exp(-(u_wrong**2))

    with pytest.raises(AssertionError):
        _assert_marcuse_and_ar_physics(
            r00_func=broken_r00_missing_two,
            r01_func=h.marcuse_gaussian_r01,
            t_ar_val=h.T_AR_COAT,
            eta_te1_val=0.43,
        )

    # Mutation 2: Half-wave thickness lambda / (2 * n_ar) instead of quarter-wave
    with pytest.raises(AssertionError):
        _assert_marcuse_and_ar_physics(
            r00_func=h.marcuse_gaussian_r00,
            r01_func=h.marcuse_gaussian_r01,
            t_ar_val=2.0 * h.T_AR_COAT,
            eta_te1_val=0.43,
        )

    # Mutation 3: Coherent dipole summation (which cancels odd TE1 mode to ~0)
    with pytest.raises(AssertionError):
        _assert_marcuse_and_ar_physics(
            r00_func=h.marcuse_gaussian_r00,
            r01_func=h.marcuse_gaussian_r01,
            t_ar_val=h.T_AR_COAT,
            eta_te1_val=9.1e-5,
        )

    # Mutation 4: Under-resolved background grid (dx_background_nm > 42.85 nm or cpw < 10)
    sim_valid, geom_valid = h.create_tilted_facet_simulation(cpw=14)
    valid_metrics = h.inspect_simulation_mesh(
        sim_valid, min_feature=geom_valid["t_ar_eff"]
    )
    with pytest.raises(AssertionError):
        _assert_mesh_resolution_rules(cpw=8, metrics=valid_metrics)
    with pytest.raises(AssertionError):
        _assert_mesh_resolution_rules(
            cpw=10,
            metrics={**valid_metrics, "dx_background_nm": 45.0},
        )

    # Mutation 5: Fewer than 4 cells across the quarter-wave AR coating (Rule 2 violation)
    with pytest.raises(AssertionError):
        _assert_mesh_resolution_rules(
            cpw=14,
            metrics={**valid_metrics, "cells_across_feature": 3},
        )
    with pytest.raises(AssertionError):
        _assert_mesh_resolution_rules(
            cpw=14,
            metrics={**valid_metrics, "x_node_err_nm": 5.0},
        )


def test_setup_schematic_and_figures_exist():
    """Verify 00_setup_schematic.png and all Tutorial 7.2 figures exist and are non-empty."""
    fig_dir = TUTORIAL_DIR / "figures"
    expected_figs = [
        "00_setup_schematic.png",
        "sled_schematic_and_mesh.png",
        "eigenmode_profiles.png",
        "steady_state_fields_0_vs_7deg.png",
        "steady_state_intensity_0_vs_7deg.png",
        "pulse_propagation_7deg.gif",
        "pulse_snapshots_7deg.png",
        "reflectivity_vs_tilt_marcuse.png",
        "broadband_ar_spectrum.png",
        "bimodal_mode_conversion.png",
        "farfield_radiation_zah1987.png",
        "sled_ase_spectral_ripple.png",
        "pml_reflectivity_floor.png",
        "incoherent_dipole_maps.png",
        "incoherent_dipole_convergence.png",
        "grid_convergence_kottke_vs_staircase.png",
    ]
    for fname in expected_figs:
        fpath = fig_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 1000, f"Missing figure {fpath}"
    obj_sled = h.build_sled_schematic_objects()
    obj_dip = h.build_incoherent_dipole_schematic_objects()
    assert len(obj_sled["structures"]) == 4
    assert len(obj_dip["sources"]) == 9

