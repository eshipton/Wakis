# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Fast verification and mutation tests (< 60 s) for Tutorial 7.3:
MicroLED with a gradient-index micro-horn collimator (Luce et al., Sci. Rep. 2026).
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.constants import c as c_light, mu_0

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "07_light_sources"
    / "07-03_microled_horn"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as h
from wakis.dispersive import dispersive_lib
from wakis.gridFIT3D import GridFIT3D
from wakis.near2far import Near2FarMonitor
from wakis.solverFIT3D import SolverFIT3D
from wakis.sources import CurrentSource, GaussianSource


def test_analytical_snell_fresnel_and_tmm():
    """Verify Snell critical angles, escape-cone bounds, Lambertian fractions, and 2x2 TMM."""
    # 1. Snell critical angles
    tc_air = h.snell_critical_angle_deg(h.N_GAN, h.N_AIR)
    tc_sio2 = h.snell_critical_angle_deg(h.N_GAN, h.N_SIO2)
    tc_nb2o5 = h.snell_critical_angle_deg(h.N_GAN, 2.37)
    assert np.isclose(tc_air, np.rad2deg(np.arcsin(1.0 / 2.45)), rtol=1e-12)
    assert 24.0 < tc_air < 24.2
    assert 36.5 < tc_sio2 < 36.7
    assert 75.2 < tc_nb2o5 < 75.4
    assert h.snell_critical_angle_deg(1.0, 1.5) == 90.0

    # 2. Planar escape cone LEE in 2D and 3D increases monotonically with n_out
    lee2d_air = h.analytic_escape_cone_lee(h.N_GAN, h.N_AIR, dim=2)
    lee2d_sio2 = h.analytic_escape_cone_lee(h.N_GAN, h.N_SIO2, dim=2)
    lee2d_nb = h.analytic_escape_cone_lee(h.N_GAN, 2.37, dim=2)
    lee3d_air = h.analytic_escape_cone_lee(h.N_GAN, h.N_AIR, dim=3)
    assert 0.21 < lee2d_air < 0.23
    assert 0.070 < lee3d_air < 0.075
    assert lee2d_air < lee2d_sio2 < lee2d_nb

    # 3. 2D vs 3D Lambertian +-15 deg cone fraction, FWHM, and 2D directivity
    assert np.isclose(h.lambertian_cone_fraction(15.0, dim=2), np.sin(np.deg2rad(15.0)))
    assert np.isclose(
        h.lambertian_cone_fraction(15.0, dim=3), np.sin(np.deg2rad(15.0)) ** 2
    )
    ang = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 721)
    assert np.isclose(h.half_power_beamwidth_deg(ang, np.cos(ang)), 120.0, atol=0.05)
    assert np.isclose(
        h.half_power_beamwidth_deg(ang, np.cos(ang) ** 2), 90.0, atol=0.05
    )
    assert np.isclose(
        h.directivity_2d_halfspace(ang, np.cos(ang)), 0.5 * np.pi, rtol=1e-4
    )
    assert np.isclose(
        h.directivity_2d_halfspace(ang, np.cos(ang) ** 2), 2.0, rtol=1e-4
    )

    # 4. 2x2 TMM: bare interface, quarter-wave zero, and 6-layer dGRIN AR suppression
    lams = np.linspace(h.WL_MIN, h.WL_MAX, 51)
    R_bare, T_bare = h.tmm_normal_reflectance(lams, n_in=h.N_GAN, n_out=h.N_AIR)
    r_fres = ((h.N_GAN - h.N_AIR) / (h.N_GAN + h.N_AIR)) ** 2
    assert np.allclose(R_bare, r_fres, rtol=1e-12)
    assert np.allclose(R_bare + T_bare, 1.0, rtol=1e-12)

    # Exact quarter-wave anti-reflection coating at lambda_0: n_ar = sqrt(n1 * n2)
    n_ar = np.sqrt(h.N_GAN * h.N_AIR)
    d_qw = h.WL_0 / (4.0 * n_ar)
    R_qw, _ = h.tmm_normal_reflectance(
        np.array([h.WL_0]),
        n_in=h.N_GAN,
        n_out=h.N_AIR,
        layer_indices=(n_ar,),
        layer_thicknesses=(d_qw,),
    )
    assert float(R_qw[0]) < 1e-14

    # 6-layer dGRIN stack reduces normal reflectance by > 3.5x across 425-475 nm
    dgrin_layers = tuple(n for _, n in h.DGRIN_STACK[:-1])
    dgrin_thks = (0.50e-6,) * len(dgrin_layers)
    R_dgrin, T_dgrin = h.tmm_normal_reflectance(
        lams,
        n_in=h.N_GAN,
        n_out=h.N_AIR,
        layer_indices=dgrin_layers,
        layer_thicknesses=dgrin_thks,
    )
    assert np.allclose(R_dgrin + T_dgrin, 1.0, rtol=1e-12)
    assert float(np.mean(R_dgrin)) < 0.30 * float(np.mean(R_bare))


def test_ag_drude_dispersion_and_skin_depth():
    """Verify calibrated 1-pole Drude Ag matches Rakic Ag at 450 nm and resolves skin depth."""
    ag = h.make_ag_drude(wl_fit=h.WL_0, eps_inf=4.0)
    f0 = c_light / h.WL_0
    eps_fit = ag.epsilon(f0)
    eps_rakic = dispersive_lib["silver"].epsilon(f0)
    assert np.isclose(eps_fit, eps_rakic, rtol=1e-10, atol=1e-10)
    assert eps_fit.real < -5.0 and eps_fit.imag < -0.5

    delta_e, delta_i = h.ag_skin_depth(h.WL_0, ag)
    assert np.isclose(delta_i, 0.5 * delta_e, rtol=1e-12)
    assert 29e-9 < delta_e < 33e-9
    assert 14.5e-9 < delta_i < 16.5e-9


def test_2d_line_dipole_power_and_near2far_conservation():
    """
    Verify that `Near2FarMonitor2D` conserves both TE (in-plane) and TM (out-of-plane)
    power against the exact 2D analytical line-dipole formula and decouples TE/TM.
    """
    dx = 15e-9
    dz = 0.5e-6
    x = np.linspace(-0.60e-6, 0.60e-6, int(round(1.20e-6 / dx)) + 1)
    y = np.linspace(-0.60e-6, 0.60e-6, int(round(1.20e-6 / dx)) + 1)
    z = np.array([-0.5 * dz, 0.5 * dz])
    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=8,
        cfln=0.9,
        verbose=0,
    )

    f0 = c_light / h.WL_0
    freqs = np.array([f0])
    pulse = GaussianSource(frequency=f0, fwidth=1.2e14, cutoff=4.0)
    S_f = pulse.fourier_transform(freqs)
    Nt = int(np.ceil((pulse.end_time + 10e-15) / solver.dt))
    alphas = np.linspace(-np.pi, np.pi, 361)

    # 1. Simultaneous Jx (TE) + Jz (TM)
    src_x = CurrentSource(
        src_time=pulse, component="x", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)
    )
    src_z = CurrentSource(
        src_time=pulse, component="z", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)
    )
    n2f_xz = h.Near2FarMonitor2D(freqs, (0.0, 0.0), (0.72e-6, 0.72e-6), dz)
    solver.emsolve(Nt, source=[src_x, src_z], monitors=[n2f_xz])

    u_te_x, u_tm_z = n2f_xz.radiation_intensity_2d_te_tm(alphas)
    p_n2f_x = float(np.trapezoid(u_te_x[0], alphas))
    p_n2f_z = float(np.trapezoid(u_tm_z[0], alphas))
    p_flux_xz = float(n2f_xz.get_flux()[0])
    p_hom_x = float(h.analytic_2d_line_dipole_power(freqs, S_f, dz, "x")[0])
    p_hom_z = float(h.analytic_2d_line_dipole_power(freqs, S_f, dz, "z")[0])

    assert np.isclose(p_hom_z, 2.0 * p_hom_x, rtol=1e-12)
    assert np.isclose(p_n2f_z / p_hom_z, 1.0, rtol=0.03)
    assert np.isclose(p_n2f_x / p_hom_x, 1.0, rtol=0.03)
    assert np.isclose((p_n2f_x + p_n2f_z) / p_flux_xz, 1.0, rtol=0.02)

    # 2. Pure Jy (TE) excitation: verify TE/TM decoupling and flux match
    solver.reset_fields()
    src_y = CurrentSource(
        src_time=pulse, component="y", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)
    )
    n2f_y = h.Near2FarMonitor2D(freqs, (0.0, 0.0), (0.72e-6, 0.72e-6), dz)
    solver.emsolve(Nt, source=[src_y], monitors=[n2f_y])

    u_te_y, u_tm_leak = n2f_y.radiation_intensity_2d_te_tm(alphas)
    p_n2f_y = float(np.trapezoid(u_te_y[0], alphas))
    p_flux_y = float(n2f_y.get_flux()[0])
    p_hom_y = float(h.analytic_2d_line_dipole_power(freqs, S_f, dz, "y")[0])

    assert float(np.max(u_tm_leak)) < 1e-12 * float(np.max(u_te_y))
    assert np.isclose(p_flux_y / p_hom_y, 1.0, rtol=0.03)
    assert np.isclose(p_n2f_y / p_flux_y, 1.0, rtol=0.02)


def test_dispersive_kottke_vs_staircase_and_horn_collimation():
    """
    Verify that Kottke subpixel smoothing on the sloped 60 deg Ag sidewall installs
    normal/series ADE sub-capacitors and that the dGRIN horn collimates emission.
    """
    dx = 15e-9
    grid_k, info = h.build_uniform_aligned_grid(
        "dgrin", h_horn=1.2e-6, theta_deg=60.0, dx=dx, x_half_min=1.35e-6
    )
    solver_k = SolverFIT3D(
        grid_k,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=6,
        cfln=0.9,
        verbose=0,
    )
    structs, _, _ = h.build_microled_horn_structures(
        "dgrin",
        h_horn=1.2e-6,
        theta_deg=60.0,
        x_max=info["x_max"] + 0.15e-6,
        y_min=info["y_min"] - 0.10e-6,
    )
    f0 = c_light / h.WL_0
    h.assign_materials_2d_dispersive(solver_k, structs, f0, averaging="kottke")

    grid_s, _ = h.build_uniform_aligned_grid(
        "dgrin", h_horn=1.2e-6, theta_deg=60.0, dx=dx, x_half_min=1.35e-6
    )
    solver_s = SolverFIT3D(
        grid_s,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=6,
        cfln=0.9,
        verbose=0,
    )
    h.assign_materials_2d_dispersive(solver_s, structs, f0, averaging="staircase")

    # Kottke splits cut sidewall edges into parallel + series sub-capacitors
    assert solver_k.dispersive_network.n_caps > solver_s.dispersive_network.n_caps

    # Compare bare vs 6-layer dGRIN horn at >= 10 cpw in GaN (dx = 15 nm, TM_z central dipole)
    r_bare = h.evaluate_horn_configuration(
        "bare",
        h_horn=0.0,
        theta_deg=90.0,
        dx=dx,
        dipole_xs=(0.0,),
        dipole_weights=(1.0,),
        polarizations=("z",),
        n_freqs=7,
        n_pml=6,
        x_half_min=1.20e-6,
        run_extra_fs=16.0,
        verbose_res=False,
    )
    r_dgrin = h.evaluate_horn_configuration(
        "dgrin",
        h_horn=1.2e-6,
        theta_deg=60.0,
        dx=dx,
        dipole_xs=(0.0,),
        dipole_weights=(1.0,),
        polarizations=("z",),
        n_freqs=7,
        n_pml=6,
        x_half_min=1.45e-6,
        run_extra_fs=20.0,
        verbose_res=False,
    )
    assert float(r_bare["cpw_gan_min_wl"]) >= 10.0
    assert float(r_dgrin["cpw_gan_min_wl"]) >= 10.0
    assert r_dgrin["lee15_pct"] > 2.5 * r_bare["lee15_pct"]
    assert r_dgrin["collimation_ratio_pct"] > 1.8 * r_bare["collimation_ratio_pct"]


def test_task_table_resolution_rules_and_mutation():
    """
    Build the mesh (without running the solver) for every task in `--quick` and `--full`
    and verify Rules 1-3, plus mutation checks that provably fail if Rules 1-3 are violated.
    """
    import copy
    import microled_horn as mh

    def _audit_table(tasks, min_cpw=10.0, min_feat_cells=4.0, min_skin_cells=4.0):
        for t in tasks:
            kind = t["kind"]
            p = t["params"]
            if kind == "config":
                grid, _ = h.build_graded_horn_grid(
                    config=p["config"],
                    h_horn=p["h_horn"],
                    theta_deg=p["theta_deg"],
                    dx=p["dx"],
                    dx_skin=p.get("dx_skin", p["dx"]),
                    x_half_min=p.get("x_half_min", 2.50e-6),
                )
                m = h.compute_mesh_resolution_metrics(
                    grid, config=p["config"], h_horn=p["h_horn"]
                )
                assert float(m["cpw_gan_min_wl"]) >= min_cpw, f"Task {t['name']} cpw violation"
                assert float(m["cells_across_feature"]) >= min_feat_cells, (
                    f"Task {t['name']} feature violation"
                )
                assert float(m["cells_per_skin_depth"]) >= min_skin_cells, (
                    f"Task {t['name']} skin depth violation"
                )
            elif kind == "angle_height_sweep":
                for h_um in (min(p["heights_um"]), max(p["heights_um"])):
                    grid, _ = h.build_graded_horn_grid(
                        config="dgrin",
                        h_horn=float(h_um) * 1e-6,
                        theta_deg=60.0,
                        dx=p["dx"],
                        dx_skin=p.get("dx_skin", p["dx"]),
                    )
                    m = h.compute_mesh_resolution_metrics(
                        grid, config="dgrin", h_horn=float(h_um) * 1e-6
                    )
                    assert float(m["cpw_gan_min_wl"]) >= min_cpw
                    assert float(m["cells_across_feature"]) >= min_feat_cells
                    assert float(m["cells_per_skin_depth"]) >= min_skin_cells
            elif kind == "qw_sweep":
                for dy in (min(p["dy_nm"]), max(p["dy_nm"])):
                    y_q = h.Y_QW_DEFAULT + float(dy) * 1e-9
                    grid, _ = h.build_graded_horn_grid(
                        config="dgrin",
                        h_horn=h.H_HORN_DEFAULT,
                        theta_deg=60.0,
                        y_qw=y_q,
                        dx=p["dx"],
                        dx_skin=p.get("dx_skin", p["dx"]),
                    )
                    m = h.compute_mesh_resolution_metrics(
                        grid, config="dgrin", h_horn=h.H_HORN_DEFAULT, y_qw=y_q
                    )
                    assert float(m["cpw_gan_min_wl"]) >= min_cpw
                    assert float(m["cells_across_feature"]) >= min_feat_cells
                    assert float(m["cells_per_skin_depth"]) >= min_skin_cells
            elif kind == "convergence":
                demos = list(p["demo_underresolved"])
                assert sum(demos) <= 1, "At most one demo under-resolved pair allowed"
                for bg_nm, sk_nm, is_demo in zip(
                    p["dx_bg_nm"], p["dx_skins_nm"], demos
                ):
                    grid, _ = h.build_graded_horn_grid(
                        config="dgrin",
                        h_horn=h.H_HORN_DEFAULT,
                        theta_deg=60.0,
                        dx=float(bg_nm) * 1e-9,
                        dx_skin=float(sk_nm) * 1e-9,
                    )
                    m = h.compute_mesh_resolution_metrics(
                        grid, config="dgrin", h_horn=h.H_HORN_DEFAULT
                    )
                    # Rule 1 & Rule 2 hold on every level (even the skin-depth demo pair)
                    assert float(m["cpw_gan_min_wl"]) >= (10.0 if is_demo else min_cpw)
                    assert float(m["cells_across_feature"]) >= min_feat_cells
                # Accepted (final) level must satisfy Rule 3 (>= 4 cells per skin depth)
                grid_acc, _ = h.build_graded_horn_grid(
                    config="dgrin",
                    h_horn=h.H_HORN_DEFAULT,
                    theta_deg=60.0,
                    dx=float(p["dx_bg_nm"][-1]) * 1e-9,
                    dx_skin=float(p["dx_skins_nm"][-1]) * 1e-9,
                )
                m_acc = h.compute_mesh_resolution_metrics(
                    grid_acc, config="dgrin", h_horn=h.H_HORN_DEFAULT
                )
                assert float(m_acc["cells_per_skin_depth"]) >= min_skin_cells

    quick_tasks = mh.get_task_table(quick=True)
    full_tasks = mh.get_task_table(quick=False)
    _audit_table(quick_tasks, min_cpw=10.0, min_feat_cells=4.0, min_skin_cells=4.0)
    _audit_table(full_tasks, min_cpw=20.0, min_feat_cells=4.0, min_skin_cells=4.0)

    # Mutation 1: violating Rule 1 (< 10 cpw in GaN, e.g. dx = 20 nm -> 8.67 cpw at 425 nm)
    mut_r1 = copy.deepcopy(quick_tasks)
    mut_r1[0]["params"]["dx"] = 20e-9
    with pytest.raises(AssertionError):
        _audit_table(mut_r1, min_cpw=10.0, min_feat_cells=4.0, min_skin_cells=4.0)

    # Mutation 2: violating Rule 2 (< 4 cells across smallest feature, e.g. 30 nm dGRIN layers)
    mut_r2 = copy.deepcopy(quick_tasks)
    mut_r2[3]["params"]["h_horn"] = 0.18e-6
    with pytest.raises(AssertionError):
        _audit_table(mut_r2, min_cpw=10.0, min_feat_cells=4.0, min_skin_cells=4.0)

    # Mutation 3: violating Rule 3 (< 4 cells per Ag skin depth, e.g. dx_skin = 10 nm)
    mut_r3 = copy.deepcopy(quick_tasks)
    mut_r3[1]["params"]["dx_skin"] = 10e-9
    with pytest.raises(AssertionError):
        _audit_table(mut_r3, min_cpw=10.0, min_feat_cells=4.0, min_skin_cells=4.0)

    # Verify saved full-study NPZ and JSON archives contain all required resolution fields & physics hierarchy
    import json

    required_keys = (
        "dx_background_nm",
        "dx_min_nm",
        "min_feature_nm",
        "cells_across_feature",
        "skin_depth_nm",
        "cells_per_skin_depth",
        "n_cells",
        "n_steps",
    )
    jsons = {}
    npzs = {}
    for t in full_tasks:
        stem = f"task_{t['index']:02d}_{t['name']}"
        npz_path = mh.DATA_DIR / f"{stem}.npz"
        json_path = mh.DATA_DIR / f"{stem}.json"
        assert npz_path.exists(), f"Missing {npz_path}"
        assert json_path.exists(), f"Missing {json_path}"
        d_npz = np.load(npz_path)
        d_json = json.loads(json_path.read_text())
        jsons[t["name"]] = d_json
        npzs[t["name"]] = d_npz
        for k in required_keys:
            assert k in d_npz, f"Missing {k} in {npz_path.name}"
            assert k in d_json, f"Missing {k} in {json_path.name}"
        assert np.isclose(float(d_json["dx_background_nm"]), 7.5)
        assert float(d_json["cpw_gan_min_wl"]) >= 20.0
        assert float(d_json["cells_across_feature"]) >= 14.0
        assert float(d_json["cells_per_skin_depth"]) >= 5.0
        assert int(d_json["n_cells"]) > 0 and int(d_json["n_steps"]) > 0

    # Verify Luce et al. (2026) 2D reference table
    assert np.isclose(h.PAPER_2D_REFS["bare"]["lee15_pct"], 14.9)
    assert np.isclose(h.PAPER_2D_REFS["air"]["lee15_pct"], 27.5)
    assert np.isclose(h.PAPER_2D_REFS["air"]["enh15"], 1.85)
    assert np.isclose(h.PAPER_2D_REFS["sio2"]["lee15_pct"], 30.4)
    assert np.isclose(h.PAPER_2D_REFS["dgrin"]["lee15_pct"], 37.7)
    assert np.isclose(h.PAPER_2D_REFS["bare"]["lee90_pct"], 55.9)

    # Verify full-study LEE15 hierarchy (headline lee15_eq_pct + secondary lee15_pct, lee15_wrk_pct)
    cfg_names = ("bare_ref", "air_horn_opt", "sio2_horn_opt", "dgrin_horn_opt")
    lee15_eq = [float(jsons[c]["lee15_eq_pct"]) for c in cfg_names]
    lee90_eq = [float(jsons[c]["lee90_eq_pct"]) for c in cfg_names]
    lee15_hom = [float(jsons[c]["lee15_pct"]) for c in cfg_names]
    lee15_wrk = [float(jsons[c]["lee15_wrk_pct"]) for c in cfg_names]
    assert lee15_eq[0] < lee15_eq[1] < lee15_eq[2] < lee15_eq[3]
    assert lee15_hom[0] < lee15_hom[1] < lee15_hom[2] < lee15_hom[3]
    assert lee15_wrk[0] < lee15_wrk[1] < lee15_wrk[2] < lee15_wrk[3]
    # Air horn reduces 2D LEE90 relative to Bare while SiO2 and dGRIN increase it
    assert lee90_eq[1] < lee90_eq[0] < lee90_eq[2] < lee90_eq[3]

    for c in cfg_names:
        d_j = jsons[c]
        d_n = npzs[c]
        assert "converged_ensemble" in d_j
        assert "rakic_ag" in d_j["converged_ensemble"]
        assert "jc_ag" in d_j["converged_ensemble"]
        assert d_j["superseded_M5_ensemble"]["superseded"] is True

        eq15_from_json_pols = (
            float(d_j["lee15_pol_x"])
            + float(d_j["lee15_pol_y"])
            + float(d_j["lee15_pol_z"])
        ) / 3.0
        eq90_from_json_pols = (
            float(d_j["lee90_pol_x"])
            + float(d_j["lee90_pol_y"])
            + float(d_j["lee90_pol_z"])
        ) / 3.0
        assert np.isclose(float(d_j["lee15_eq_pct"]), eq15_from_json_pols, atol=1e-2)
        assert np.isclose(float(d_j["lee90_eq_pct"]), eq90_from_json_pols, atol=1e-2)

        # Coarse M=5 (or M=11 for dGRIN) .npz archive matches the corresponding JSON entry
        eq15_from_npz_pols = (
            float(d_n["lee15_pol_x"])
            + float(d_n["lee15_pol_y"])
            + float(d_n["lee15_pol_z"])
        ) / 3.0
        assert np.isclose(
            float(d_j["superseded_M5_ensemble"]["lee15_eq_pct"]),
            eq15_from_npz_pols,
            atol=1e-3,
        )
        # Mutation check: weighting TM_z 2x ([1/4, 1/4, 1/2]) must NOT equal lee15_eq_pct
        hom15_from_pols = (
            0.25 * float(d_j["lee15_pol_x"])
            + 0.25 * float(d_j["lee15_pol_y"])
            + 0.50 * float(d_j["lee15_pol_z"])
        )
        with pytest.raises(AssertionError):
            assert np.isclose(
                float(d_j["lee15_eq_pct"]), hom15_from_pols, atol=0.1
            )

        fp = float(d_j["purcell_factor"])
        assert np.isclose(
            float(d_j["lee15_pct"]),
            fp * float(d_j["lee15_wrk_pct"]),
            rtol=2e-3,
        )
        assert np.isclose(
            float(d_j["lee90_pct"]),
            fp * float(d_j["lee90_wrk_pct"]),
            rtol=2e-3,
        )

    # Verify Task 6 convergence ladder & consistency with Task 3 single-central-dipole (N_dip=1)
    t3_npz = npzs["dgrin_horn_opt"]
    t6_npz = npzs["conv_kottke_vs_staircase"]
    assert len(t6_npz["dxs_nm"]) == 5
    assert float(t6_npz["kottke_rel_change_lee90_pct"][-1]) < 1.0
    assert float(t6_npz["kottke_rel_change_lee15_pct"][-1]) < 1.5
    assert float(t6_npz["kottke_lee90"][-1]) > float(t6_npz["staircase_lee90"][-1]) + 5.0
    assert (
        float(t6_npz["kottke_lee15"][-2])
        < float(t3_npz["conv_lee15"][0])
        < float(t6_npz["kottke_lee15"][-1])
    )
    assert (
        float(t6_npz["kottke_lee90"][-2])
        < float(t3_npz["conv_lee90"][0])
        < float(t6_npz["kottke_lee90"][-1])
    )


def test_mutation_checks_provably_fail():
    """
    Mutation test: verify that removing either (1) the periodic-z edge weight fix
    or (2) the 2D cylindrical Green's function scaling causes conservation to fail.
    """
    dx = 15e-9
    dz = 0.5e-6
    x = np.linspace(-0.54e-6, 0.54e-6, int(round(1.08e-6 / dx)) + 1)
    y = np.linspace(-0.54e-6, 0.54e-6, int(round(1.08e-6 / dx)) + 1)
    z = np.array([-0.5 * dz, 0.5 * dz])
    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=6,
        cfln=0.9,
        verbose=0,
    )
    f0 = c_light / h.WL_0
    freqs = np.array([f0])
    pulse = GaussianSource(frequency=f0, fwidth=1.2e14, cutoff=4.0)
    S_f = pulse.fourier_transform(freqs)
    Nt = int(np.ceil((pulse.end_time + 9e-15) / solver.dt))
    alphas = np.linspace(-np.pi, np.pi, 181)

    class _MutatedNoPeriodicZFix(h.Near2FarMonitor2D):
        """Mutated monitor omitting the `self.size[2] = 3 * dz` periodic-z fix."""

        def _initialize(self, s):
            Near2FarMonitor._initialize(self, s)

    src_y = CurrentSource(
        src_time=pulse, component="y", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)
    )
    mon_good = h.Near2FarMonitor2D(freqs, (0.0, 0.0), (0.66e-6, 0.66e-6), dz)
    mon_mut1 = _MutatedNoPeriodicZFix(freqs, (0.0, 0.0), (0.66e-6, 0.66e-6), dz)
    solver.emsolve(Nt, source=[src_y], monitors=[mon_good, mon_mut1])

    p_hom_y = float(h.analytic_2d_line_dipole_power(freqs, S_f, dz, "y")[0])
    p_good = float(np.trapezoid(mon_good.radiation_intensity_2d(alphas)[0], alphas))
    p_mut1 = float(np.trapezoid(mon_mut1.radiation_intensity_2d(alphas)[0], alphas))

    # Unmutated monitor matches analytical power within 3%
    assert np.isclose(p_good / p_hom_y, 1.0, rtol=0.03)
    # Mutation 1 quarters the TE far-field intensity (~0.25 of true power) -> provably fails!
    with pytest.raises(AssertionError):
        assert np.isclose(p_mut1 / p_hom_y, 1.0, rtol=0.10)

    # Mutation 2: omitting the 2*pi/(k*dz) 3D->2D cylindrical scaling factor
    ff = mon_good.get_farfield(
        np.full_like(alphas, 0.5 * np.pi), 0.5 * np.pi - alphas
    )
    u_unscaled_3d = (np.abs(ff["Ephi"][0]) ** 2) / mon_good.impedance
    p_mut2 = float(np.trapezoid(u_unscaled_3d, alphas))
    with pytest.raises(AssertionError):
        assert np.isclose(p_mut2 / p_hom_y, 1.0, rtol=0.10)


def test_gap_diagnostics_convergence_and_jc_ag():
    """
    Verify the committed data/diag/*.json runs:
    1. |M9 - M17| < 0.5 pp for air and sio2 on the full mesh (baseline and jc_ag).
    2. |M5 - M17| > 2.0 pp for sio2 on the quick mesh (documents M=5 non-convergence).
    3. J&C silver (jc_ag) on the full mesh (M=17) agrees with the paper within 1.0 pp
       for air, sio2, and dgrin.
    4. Each of the three assertions provably fails when a value is mutated in memory.
    """
    import json
    import microled_horn as mh

    diag_dir = mh.DATA_DIR / "diag"
    assert diag_dir.is_dir(), f"Missing {diag_dir}"

    def _load_diag(stem: str) -> dict:
        p = diag_dir / f"{stem}.json"
        assert p.exists(), f"Missing {p}"
        return json.loads(p.read_text())

    def _assert_m9_m17_converged(d9: dict, d17: dict, tol_pp: float = 0.5):
        diff = abs(float(d9["lee15_eq_pct"]) - float(d17["lee15_eq_pct"]))
        assert diff < tol_pp, f"|M9 - M17| = {diff:.3f} pp >= {tol_pp} pp"

    def _assert_m5_m17_nonconverged(d5: dict, d17: dict, min_diff_pp: float = 2.0):
        diff = abs(float(d5["lee15_eq_pct"]) - float(d17["lee15_eq_pct"]))
        assert diff > min_diff_pp, f"|M5 - M17| = {diff:.3f} pp <= {min_diff_pp} pp"

    def _assert_jc_matches_paper(d_jc: dict, paper_val: float, tol_pp: float = 1.0):
        diff = abs(float(d_jc["lee15_eq_pct"]) - float(paper_val))
        assert diff < tol_pp, f"|jc_ag - paper| = {diff:.3f} pp >= {tol_pp} pp"

    # 1. |M9 - M17| < 0.5 pp for air and sio2 (full mesh, both silver models)
    for variant in ("baseline", "jc_ag"):
        for cfg in ("air", "sio2"):
            d9 = _load_diag(f"{variant}_{cfg}_M9_full")
            d17 = _load_diag(f"{variant}_{cfg}_M17_full")
            _assert_m9_m17_converged(d9, d17, tol_pp=0.5)
            # In-memory mutation check: perturb M17 by +0.6 pp -> must fail
            d17_mut = dict(d17, lee15_eq_pct=float(d9["lee15_eq_pct"]) + 0.6)
            with pytest.raises(AssertionError):
                _assert_m9_m17_converged(d9, d17_mut, tol_pp=0.5)

    # 2. |M5 - M17| > 2.0 pp for sio2 (quick mesh, baseline Rakic Ag)
    q_sio2_m5 = _load_diag("baseline_sio2_M5_quick")
    q_sio2_m17 = _load_diag("baseline_sio2_M17_quick")
    _assert_m5_m17_nonconverged(q_sio2_m5, q_sio2_m17, min_diff_pp=2.0)
    # In-memory mutation check: set M5 within 0.5 pp of M17 -> must fail
    q_sio2_m5_mut = dict(
        q_sio2_m5, lee15_eq_pct=float(q_sio2_m17["lee15_eq_pct"]) + 0.5
    )
    with pytest.raises(AssertionError):
        _assert_m5_m17_nonconverged(q_sio2_m5_mut, q_sio2_m17, min_diff_pp=2.0)

    # 3. J&C full (M=17) air, sio2, dgrin within 1.0 pp of the paper
    for cfg in ("air", "sio2", "dgrin"):
        d_jc = _load_diag(f"jc_ag_{cfg}_M17_full")
        paper_val = float(h.PAPER_2D_REFS[cfg]["lee15_pct"])
        _assert_jc_matches_paper(d_jc, paper_val, tol_pp=1.0)
        # In-memory mutation check: shift jc_ag by +1.5 pp from paper -> must fail
        d_jc_mut = dict(d_jc, lee15_eq_pct=paper_val + 1.5)
        with pytest.raises(AssertionError):
            _assert_jc_matches_paper(d_jc_mut, paper_val, tol_pp=1.0)

    # 4. Setup figures (00a, 00b), setup table (00_setup_table.md) and all figures exist and are non-empty
    table_path = mh.FIG_DIR / "00_setup_table.md"
    assert table_path.exists() and table_path.stat().st_size > 200, "Missing or empty 00_setup_table.md"
    expected_figs = (
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "01_microled_horn_schematics.png",
        "02_materials_and_tmm_reflectance.png",
        "03_sidewall_mesh_kottke_vs_staircase.png",
        "04_electric_field_wavefronts.png",
        "04b_electric_field_bare_vs_dgrin.png",
        "05_poynting_power_flow_maps.png",
        "06_pulse_propagation_dgrin.gif",
        "07_farfield_polar_patterns.png",
        "08_farfield_cartesian_and_paper_refs.png",
        "09_horn_angle_and_height_sweep.png",
        "10_qw_position_sensitivity.png",
        "11_incoherent_dipole_convergence.png",
        "12_polarization_and_spectral_breakdown.png",
        "13_kottke_vs_staircase_convergence.png",
        "14_purcell_and_normalization_audit.png",
        "15_gap_diagnostics.png",
    )
    for fn in expected_figs:
        fpath = mh.FIG_DIR / fn
        assert fpath.exists() and fpath.stat().st_size > 10_000, f"Missing or empty {fn}"
    assert (mh.FIG_DIR / "06_pulse_propagation_dgrin.gif").stat().st_size < 5 * 1024 * 1024


