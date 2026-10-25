"""
Fast verification and mutation tests (< 60 s) for Tutorial 7.4:
RGB microLED array — incoherent averaging, collimation, and pixel crosstalk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as C0

import wakis
from wakis.sources import CurrentSource, GaussianSource

REPO_ROOT = Path(__file__).resolve().parent.parent
TUTORIAL_DIR = REPO_ROOT / "tutorials" / "07_light_sources" / "07-04_rgb_microled_array"
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as h  # noqa: E402


def test_table1_materials_and_skin_depth():
    """Verify Table 1 parameters from Chen et al. (2025) and Silicon skin depths."""
    assert h.CHANNELS == ("B", "G", "R")
    # Unit cell pitch: 3 * (1.05 um mesa + 0.45 um polyimide trench) = 4.50 um
    pitch_calc = 3.0 * (
        h.PIXEL_GEOMETRY["channel_width_m"] + h.PIXEL_GEOMETRY["trench_width_m"]
    )
    assert np.isclose(pitch_calc, h.PIXEL_GEOMETRY["unit_pitch_m"], rtol=0, atol=1e-12)

    # Exact Table 1 checks
    assert np.isclose(h.RGB_MATERIALS["B"]["wavelength_m"], 457e-9)
    assert np.isclose(h.RGB_MATERIALS["G"]["wavelength_m"], 527e-9)
    assert np.isclose(h.RGB_MATERIALS["R"]["wavelength_m"], 611e-9)
    assert np.isclose(h.RGB_MATERIALS["B"]["n_MQW"], 2.51)
    assert np.isclose(h.RGB_MATERIALS["G"]["n_GaN"], 2.38)
    assert np.isclose(h.RGB_MATERIALS["R"]["n_Si_real"], 3.906)
    assert np.isclose(h.RGB_MATERIALS["R"]["n_Si_imag"], 0.022)

    # Silicon field skin depth at Blue (457 nm, k = 0.13): lambda / (2 * pi * k)
    dx_quick = h.grid_spacing_from_cpw(10.0)
    skin_b = h.silicon_skin_depth(457e-9, 0.13, dx_m=dx_quick)
    expected_delta_e = 457e-9 / (2.0 * np.pi * 0.13)
    assert np.isclose(skin_b["delta_field_m"], expected_delta_e, rtol=1e-12)
    assert np.isclose(skin_b["delta_power_m"], 0.5 * expected_delta_e, rtol=1e-12)
    assert skin_b["cells_per_field_skin_depth"] > 25.0


def test_lambertian_and_etendue_analytical_identities():
    """Check 2D vs 3D Lambertian cone integrals, escape cone, and 2D étendue bounds."""
    theta_deg = np.linspace(-90.0, 90.0, 1801)
    dtheta = 180.0 / len(theta_deg)
    p_lamb = h.lambertian_2d_density(theta_deg)

    # Equation (5) normalization: sum(P_r(theta) * dtheta) == 1
    assert np.isclose(np.sum(p_lamb) * dtheta, 1.0, rtol=1e-12)

    # 2D Lambertian cone fraction within +/- 20 deg is sin(20 deg) = 0.34202014...
    eta_20_num = float(np.sum(p_lamb[np.abs(theta_deg) <= 20.0]) * dtheta)
    eta_20_exact = h.lambertian_2d_cone_fraction(20.0)
    assert np.isclose(eta_20_exact, np.sin(np.deg2rad(20.0)), rtol=1e-12)
    assert np.isclose(eta_20_num, eta_20_exact, atol=2e-3)

    # 3D Lambertian cone fraction within 20 deg is sin^2(20 deg) = 0.11697778...
    eta_3d = h.lambertian_3d_cone_fraction(20.0)
    assert np.isclose(eta_3d, eta_20_exact**2, rtol=1e-12)
    assert eta_20_exact > 2.5 * eta_3d

    # Thermodynamic 2D étendue upper bound: single subpixel (1.50 um) vs full pixel (4.50 um)
    bound_sub = h.etendue_collimation_bound_2d(1.50e-6, 1.05e-6, 20.0)
    bound_unit = h.etendue_collimation_bound_2d(4.50e-6, 1.05e-6, 20.0)
    assert np.isclose(bound_sub, (1.50 / 1.05) * np.sin(np.deg2rad(20.0)), rtol=1e-12)
    assert bound_sub < 0.50  # Cannot reach Chen et al.'s ~60% with 1-subpixel aperture
    assert np.isclose(bound_unit, 1.0, rtol=1e-12)

    # 2D vs 3D planar escape cone in GaN (n = 2.41)
    esc = h.escape_cone_2d_analytical(2.41, 1.0, 4.215, 0.048)
    assert np.isclose(esc["theta_c_deg"], np.rad2deg(np.arcsin(1.0 / 2.41)), rtol=1e-12)
    assert esc["single_cone_2d"] > 2.5 * esc["single_cone_3d"]
    assert 0.15 < esc["lee_2d_planar_est"] < 0.35


def test_reset_solver_state_cpml_reproducibility_and_mutation():
    """Verify reset_solver_state zeroes CPML memory and catches un-reset CPML mutation."""
    dz = 1.0e-6
    grid = wakis.GridFIT3D(-0.8e-6, 0.8e-6, -0.8e-6, 0.8e-6, -dz / 2, dz / 2, 32, 32, 1, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=6,
        cfln=0.85,
        verbose=0,
    )
    solver.ieps[:, :, :, "z"] = 0.0
    f0 = C0 / 527e-9
    src_time = GaussianSource(frequency=f0, fwidth=0.35 * f0, cutoff=3.5)

    m1 = wakis.FluxMonitor([f0], center=(0.0, 0.25e-6, 0.0), size=(0.8e-6, 0.0, np.inf))
    solver.emsolve(
        140,
        source=CurrentSource(src_time, component="x", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, dz)),
        monitors=[m1],
    )
    flux_ref = float(m1.get_flux()[0])
    assert flux_ref > 0.0

    # Mutation check: calling solver.reset_fields() alone leaves CPML psi arrays dirty
    solver.reset_fields()
    cpml_residual = sum(
        float(np.max(np.abs(getattr(solver, name))))
        for name in dir(solver)
        if name.startswith(("psiHa_", "psiHb_", "psiEa_", "psiEb_"))
    )
    assert cpml_residual > 0.0, "Mutation check failed: CPML psi arrays should be non-zero after reset_fields()"

    m_mutated = wakis.FluxMonitor([f0], center=(0.0, 0.25e-6, 0.0), size=(0.8e-6, 0.0, np.inf))
    solver.emsolve(
        140,
        source=CurrentSource(src_time, component="x", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, dz)),
        monitors=[m_mutated],
    )
    flux_mutated = float(m_mutated.get_flux()[0])
    assert abs(flux_mutated - flux_ref) > 0.0

    # Full reset via reset_solver_state(solver) must reproduce flux_ref to 0.0 error
    h.reset_solver_state(solver)
    cpml_after_full_reset = sum(
        float(np.max(np.abs(getattr(solver, name))))
        for name in dir(solver)
        if name.startswith(("psiHa_", "psiHb_", "psiEa_", "psiEb_"))
    )
    assert cpml_after_full_reset == 0.0

    m_clean = wakis.FluxMonitor([f0], center=(0.0, 0.25e-6, 0.0), size=(0.8e-6, 0.0, np.inf))
    solver.emsolve(
        140,
        source=CurrentSource(src_time, component="x", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, dz)),
        monitors=[m_clean],
    )
    flux_clean = float(m_clean.get_flux()[0])
    assert flux_clean == pytest.approx(flux_ref, rel=1e-14, abs=0.0)


def test_2d_near2far_and_flux_box_power_conservation():
    """Verify 2D Near2Far cylindrical projection conserves 2D closed-box Poynting flux."""
    dz = 1.0e-6
    grid = wakis.GridFIT3D(-1.2e-6, 1.2e-6, -1.2e-6, 1.2e-6, -dz / 2, dz / 2, 60, 60, 1, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=8,
        cfln=0.85,
        verbose=0,
    )
    solver.ieps[:, :, :, "z"] = 0.0
    f0 = C0 / 550e-9
    k0 = 2.0 * np.pi * f0 / C0
    src_time = GaussianSource(frequency=f0, fwidth=0.35 * f0, cutoff=4.0)
    nt = int((src_time.end_time + 10e-15) / solver.dt)

    n2f = wakis.Near2FarMonitor([f0], center=(0.0, 0.0, 0.0), size=(1.0e-6, 1.0e-6, np.inf))
    n2f.faces = [face for face in n2f.faces if face[0] in (0, 1)]
    fbox = h.make_flux_box_2d([f0], center=(0.0, 0.0, 0.0), width_x=1.0e-6, height_y=1.0e-6)

    solver.emsolve(
        nt,
        source=CurrentSource(src_time, component="x", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, dz)),
        monitors=[n2f, fbox],
    )

    flux_near = float(fbox.get_flux()[0])
    phi = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    theta_3d = np.full_like(phi, 0.5 * np.pi)
    u_3d = n2f.radiation_intensity(theta_3d, phi)[0]
    u_2d = u_3d * (2.0 * np.pi / (k0 * dz))
    power_far_2d = float(np.sum(u_2d) * (2.0 * np.pi / len(phi)))

    assert np.isclose(power_far_2d, flux_near, rtol=0.03)


def test_method_a_vs_method_b_statistics_and_mutation():
    """Verify Method B RSE scaling sqrt((1-1/N)/K) and fail under coherent-phase mutation."""
    theta_deg = np.linspace(-90.0, 90.0, 181)
    theta_rad = np.deg2rad(theta_deg)
    n_dip = 20
    xs = (np.arange(n_dip) + 0.5) / n_dip * 1.05e-6 - 0.525e-6
    k0 = 2.0 * np.pi / 527e-9

    # Synthetic 20-dipole complex far-field array with realistic phase gradients
    envelope = np.sqrt(np.clip(np.cos(theta_rad), 0.0, None))
    e_ff = np.array(
        [envelope * np.exp(1j * k0 * x_m * np.sin(theta_rad)) for x_m in xs],
        dtype=np.complex128,
    )
    u_inc = np.sum(np.abs(e_ff) ** 2, axis=0)

    stats = h.evaluate_method_b_statistics(
        e_ff, u_inc, theta_deg, k_values=(1, 4, 16, 64), n_trials=500, seed=123
    )
    # Empirical RSE must follow exact analytical sqrt((1 - 1/N)/K) within 7%
    assert np.allclose(stats["rse_empirical"], stats["rse_theory"], rtol=0.07)
    # Proper ensemble ratio bias at K=64 is near zero (< 0.4 percentage points)
    assert abs(stats["eta_bias_vs_k_pct"][-1]) < 0.40

    # Mutation check: replacing incoherent power sum with coherent field sum |sum E_m|^2
    # creates massive diffraction-grating narrowing in eta_20
    eta_inc = h.summarize_angular_distribution(theta_deg, u_inc)["eta_20_pct"]
    u_mutated_coherent = np.abs(np.sum(e_ff, axis=0)) ** 2
    eta_mutated = h.summarize_angular_distribution(theta_deg, u_mutated_coherent)["eta_20_pct"]
    assert abs(eta_mutated - eta_inc) > 25.0, "Mutation check failed: coherent sum must differ strongly from incoherent sum"


def test_setup_schematic_exists():
    """Verify the setup figure pairs and setup tables exist and are non-empty."""
    fig_dir = TUTORIAL_DIR / "figures"
    for fname in (
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00a_setup_geometry_dual_routing.png",
        "00b_setup_mesh_and_excitation_dual_routing.png",
    ):
        setup_fig = fig_dir / fname
        assert setup_fig.is_file() and setup_fig.stat().st_size > 10_000, f"Missing {fname}"
    for tname in ("00_setup_table.md", "00_setup_table_dual_routing.md"):
        setup_tbl = fig_dir / tname
        assert setup_tbl.is_file() and setup_tbl.stat().st_size > 500, f"Missing {tname}"


def test_saved_study_outputs_and_crosstalk_matrices():
    """Verify saved --full study archives, figures, and physical matrix constraints."""
    data_dir = TUTORIAL_DIR / "data"
    fig_dir = TUTORIAL_DIR / "figures"

    expected_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00_setup_table.md",
        "00a_setup_geometry_dual_routing.png",
        "00b_setup_mesh_and_excitation_dual_routing.png",
        "00_setup_table_dual_routing.md",
        "fig01_geometry_permittivity.png",
        "fig02_material_dispersion_skin_depth.png",
        "fig03_incoherent_rgb_intensity_maps.png",
        "fig04_coherent_vs_incoherent_comparison.png",
        "fig05_angular_power_density_rgb.png",
        "fig06_polar_radiation_patterns.png",
        "fig07_collimation_efficiency_comparison.png",
        "fig08_dipole_count_convergence.png",
        "fig09_method_a_vs_method_b_profiles.png",
        "fig10_method_b_bias_variance_cost.png",
        "fig11_forward_crosstalk_matrices.png",
        "fig12_dual_task_color_routing.png",
        "fig13_resolution_convergence.png",
        "fig14_time_domain_snapshots.png",
        "fig15_rgb_emission_movie.gif",
    ]
    for fname in expected_figs:
        fpath = fig_dir / fname
        assert fpath.is_file() and fpath.stat().st_size > 1000, f"Missing or empty figure: {fname}"

    required_res_keys = (
        "dx_background_nm",
        "dx_min_nm",
        "min_feature_nm",
        "cells_across_feature",
        "skin_depth_nm",
        "cells_per_skin_depth",
        "n_cells",
        "n_steps",
    )

    for ch, idx in [("B", 0), ("G", 1), ("R", 2)]:
        d = np.load(data_dir / f"task_{idx:02d}_method_a_{ch}.npz")
        for k in required_res_keys:
            assert k in d, f"Missing resolution key {k} in task_{idx:02d}_method_a_{ch}.npz"
        h.validate_mesh_resolution({k: d[k] for k in d.files}, min_cpw=20.0)
        assert float(d["cpw"]) == pytest.approx(20.0)
        assert int(d["cells_across_feature"]) == 41
        assert int(d["cells_across_trench"]) == 50
        assert float(d["cells_per_skin_depth"]) >= 50.0
        eta_inc = float(d["eta_20_inc_pct"])
        eta_pub = h.PAPER_TARGETS[ch]["eta_20_uncol_pct"]
        assert abs(eta_inc - eta_pub) < 5.0, f"{ch} eta_20 ({eta_inc:.2f}%) too far from paper ({eta_pub:.2f}%)"
        assert 20.0 < float(d["lee_tot_pct"]) < 40.0

        # Crosstalk rows must be non-negative, sum to 1, and be diagonally dominant
        win_row = d["win_crosstalk_row"]
        mqw_row = d["mqw_crosstalk_row"]
        assert np.isclose(np.sum(win_row), 1.0, atol=1e-6)
        assert np.isclose(np.sum(mqw_row), 1.0, atol=1e-6)
        assert win_row[idx] > 0.85
        assert mqw_row[idx] > 0.90

    db = np.load(data_dir / "task_03_method_b_G.npz")
    for k in required_res_keys:
        assert k in db, f"Missing resolution key {k} in task_03_method_b_G.npz"
    h.validate_mesh_resolution({k: db[k] for k in db.files}, min_cpw=20.0)
    assert int(db["n_realizations"]) == 16

    dd = np.load(data_dir / "task_04_dual_routing.npz")
    for k in required_res_keys:
        assert k in dd, f"Missing resolution key {k} in task_04_dual_routing.npz"
    h.validate_mesh_resolution({k: dd[k] for k in dd.files}, min_cpw=20.0)
    c_dual = dd["c_norm_mean"]
    assert np.allclose(np.sum(c_dual, axis=0), 1.0, atol=1e-6)
    assert np.all((np.diag(c_dual) > 0.30) & (np.diag(c_dual) < 0.37))

    # Check convergence ladder monotony/convergence at accepted cpw=20 vs cpw=28
    d10 = np.load(data_dir / "task_05_conv_cpw10.npz")
    d20 = np.load(data_dir / "task_08_conv_cpw20.npz")
    d28 = np.load(data_dir / "task_09_conv_cpw28.npz")
    for k in required_res_keys:
        assert k in d10 and k in d20 and k in d28
    rel_eta_10 = abs(float(d10["eta_20_pct"]) - float(d28["eta_20_pct"])) / float(d28["eta_20_pct"])
    rel_eta = abs(float(d20["eta_20_pct"]) - float(d28["eta_20_pct"])) / float(d28["eta_20_pct"])
    rel_lee = abs(float(d20["lee_tot_pct"]) - float(d28["lee_tot_pct"])) / float(d28["lee_tot_pct"])
    assert rel_eta_10 > 0.10  # Quick 10 cpw single-dipole eta_20 is not converged (15.2% vs 28 cpw)
    assert rel_eta < 0.01  # Accepted full 20 cpw is converged (< 1%: 0.99%)
    assert rel_lee < 0.01  # Accepted full 20 cpw LEE_tot is converged (< 1%: 0.55%)


def test_resolution_rules_all_tasks_and_mutation():
    """Verify every task in --quick and --full satisfies Rules 1-3 (mesh only) and fail on mutation."""
    import rgb_microled_array as script

    for full_mode in (False, True):
        cpw_main = 20.0 if full_mode else 10.0
        for spec in script.STUDY_TASKS:
            kind = spec["kind"]
            if kind == "convergence":
                cpw = float(spec["cpw"])
                _, meta = h.build_rgb_array_mesh(
                    cpw=cpw,
                    domain_x=(-1.55e-6, 1.55e-6),
                    domain_y=(-1.38e-6, 0.58e-6),
                    channel="G",
                )
            elif kind == "dual_routing":
                # Check worst-case shortest wavelength / skin depth channel (Blue, 457 nm)
                _, meta = h.build_rgb_array_mesh(cpw=cpw_main, channel="B")
            else:
                _, meta = h.build_rgb_array_mesh(cpw=cpw_main, channel=spec["channel"])

            # Must pass strict resolution validator (Rules 1, 2, 3)
            h.validate_mesh_resolution(meta, min_cpw=10.0)
            assert meta["cpw"] >= 10.0
            if full_mode and kind != "convergence":
                assert meta["cpw"] >= 20.0
            assert meta["cells_across_feature"] >= 4
            assert meta["cells_across_trench"] >= 4
            assert meta["cells_per_skin_depth"] >= 4.0
            assert meta["dx_min_nm"] < meta["dx_background_nm"]

    # Mutation check 1: under-resolved cpw < 10 must fail validation
    _, meta_under = h.build_rgb_array_mesh(cpw=8.0, channel="B")
    with pytest.raises(AssertionError, match="violates minimum"):
        h.validate_mesh_resolution(meta_under, min_cpw=10.0)

    # Mutation check 2: fewer than 4 cells across minimum feature must fail validation
    meta_bad_feat = dict(meta)
    meta_bad_feat["cells_across_feature"] = 3
    with pytest.raises(AssertionError, match="cells_across_feature"):
        h.validate_mesh_resolution(meta_bad_feat, min_cpw=10.0)

    # Mutation check 3: fewer than 4 cells per skin depth must fail validation
    meta_bad_skin = dict(meta)
    meta_bad_skin["cells_per_skin_depth"] = 2.5
    with pytest.raises(AssertionError, match="cells_per_skin_depth"):
        h.validate_mesh_resolution(meta_bad_skin, min_cpw=10.0)

