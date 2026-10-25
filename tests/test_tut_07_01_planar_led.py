# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Fast validation tests (< 60 s) for Tutorial 7.1: Planar AlGaN UV LED."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "07_light_sources"
    / "07-01_planar_led"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


helpers = _load_module("_emit01_helpers", TUTORIAL_DIR / "_helpers.py")
planar_led = _load_module("planar_led", TUTORIAL_DIR / "planar_led.py")


def _validate_2d_slab_escape(eta_sim: float, eta_expected: float, tol: float = 0.04):
    """Validation assertion helper that raises AssertionError on mismatch."""
    if not np.isfinite(eta_sim) or abs(eta_sim - eta_expected) > tol:
        raise AssertionError(
            f"Simulated 2D escape fraction {eta_sim:.4f} deviates from "
            f"expected {eta_expected:.4f} by more than tol={tol:.4f}"
        )


def _validate_te_over_tm_ratio(eta_te: float, eta_tm: float, min_ratio: float = 8.0):
    """Validation assertion helper requiring TE extraction >> TM extraction."""
    ratio = eta_te / max(eta_tm, 1e-12)
    if ratio < min_ratio:
        raise AssertionError(
            f"TE/TM extraction ratio {ratio:.2f} is below minimum {min_ratio:.2f}"
        )


def _validate_incoherent_power_profile(
    profile_combined: np.ndarray,
    profile_expected: np.ndarray,
    tol: float = 1e-10,
):
    """Validation assertion helper requiring multi-dipole mean intensity conservation."""
    max_diff = float(np.max(np.abs(profile_combined - profile_expected)))
    if max_diff > tol:
        raise AssertionError(
            f"Combined multi-dipole intensity profile deviates from incoherent mean "
            f"by {max_diff:.6e} > tol={tol:.6e}"
        )


def _validate_task_mesh_resolution(meta: dict):
    """Validate that a non-demo task mesh satisfies Resolution Rules 1, 2, and 3."""
    if meta.get("is_demo", False):
        return
    label = meta.get("label", "unknown")
    cpw = float(meta["cpw"])
    n_max = float(meta.get("n_max", 2.90))
    dx_bg = float(meta["dx_background_nm"])
    dx_min = float(meta["dx_min_nm"])
    min_feat = float(meta["min_feature_nm"])
    cells_feat = int(meta["cells_across_feature"])
    skin_nm = float(meta["skin_depth_nm"])
    cpsd = float(meta["cells_per_skin_depth"])

    max_allowed_dx_bg = planar_led.WAVELENGTH_NM / (10.0 * n_max) + 0.05
    if cpw < 10.0 or dx_bg > max_allowed_dx_bg:
        raise AssertionError(
            f"Rule 1 violated in task {label!r}: cpw={cpw:.2f} (< 10) or "
            f"dx_background_nm={dx_bg:.3f} > {max_allowed_dx_bg:.3f}"
        )
    if cells_feat < 4 or dx_min > min_feat / 4.0 + 0.05:
        raise AssertionError(
            f"Rule 2 violated in task {label!r}: cells_across_feature={cells_feat} (< 4) "
            f"or dx_min_nm={dx_min:.3f} > min_feature_nm/4={min_feat / 4.0:.3f}"
        )
    if np.isfinite(skin_nm) and cpsd < 4.0 - 1e-6:
        raise AssertionError(
            f"Rule 3 violated in task {label!r}: cells_per_skin_depth={cpsd:.2f} (< 4.0)"
        )
    if "interface_z_errors_m" in meta and float(np.max(meta["interface_z_errors_m"])) > 1e-12:
        raise AssertionError(
            f"Interface node misalignment in task {label!r}: "
            f"max error = {float(np.max(meta['interface_z_errors_m'])):.3e} m"
        )


def test_task_tables_satisfy_resolution_rules_1_to_3():
    """Build meshes only (no solver) and verify Rules 1-3 for --quick and --full tables."""
    for mode in ("quick", "full"):
        tasks = planar_led._task_table(mode)
        demo_tasks = [t for t in tasks if t.get("is_demo", False)]
        non_demo_tasks = [t for t in tasks if not t.get("is_demo", False)]

        assert len(demo_tasks) <= 1, f"At most 1 demo task allowed in {mode} mode"
        assert len(non_demo_tasks) >= 5
        for dt in demo_tasks:
            assert "demo" in dt["label"].lower()

        for t in non_demo_tasks:
            meta = planar_led.build_task_mesh_info(t)
            _validate_task_mesh_resolution(meta)
            if mode == "full" and t["kind"] != "convergence_pair":
                assert float(meta["cpw"]) >= 20.0


def test_analytic_escape_cone_2d_vs_3d():
    """Verify 2D (theta_c/pi) vs 3D ((1-cos theta_c)/2) escape cones and Fresnel ordering."""
    # Limit n1 -> n2: half of all power escapes upward (0.5) in both 2D and 3D
    assert np.isclose(helpers.escape_cone_2d_nofresnel(1.0, 1.0), 0.5, atol=1e-12)
    assert np.isclose(helpers.escape_cone_3d_nofresnel(1.0, 1.0), 0.5, atol=1e-12)

    for n in [1.5, 2.0, 2.6]:
        theta_c = np.arcsin(1.0 / n)
        eta_2d = float(helpers.escape_cone_2d_nofresnel(n, 1.0))
        eta_3d = float(helpers.escape_cone_3d_nofresnel(n, 1.0))
        eta_3d_approx = 1.0 / (4.0 * n**2)

        assert np.isclose(eta_2d, theta_c / np.pi, rtol=1e-12)
        assert np.isclose(eta_3d, 0.5 * (1.0 - np.cos(theta_c)), rtol=1e-12)
        assert eta_2d > eta_3d > eta_3d_approx > 0.0
        # 1/(4 n^2) is within 8% of the exact 3D cone for high index (n >= 2)
        if n >= 2.0:
            assert abs(eta_3d - eta_3d_approx) / eta_3d < 0.08

    # At n = 2.60 (AlGaN), verify Fresnel-weighted 2D polarization hierarchy:
    # J_x (in-plane TE, cos^2 theta) > J_y (isotropic TE) > J_z (vertical TM, sin^2 theta)
    eta_jx = helpers.escape_cone_2d_fresnel(2.60, 1.0, dipole="x")
    eta_jy = helpers.escape_cone_2d_fresnel(2.60, 1.0, dipole="y")
    eta_jz = helpers.escape_cone_2d_fresnel(2.60, 1.0, dipole="z")
    eta_no_fresnel = float(helpers.escape_cone_2d_nofresnel(2.60, 1.0))

    assert eta_jx > eta_no_fresnel > eta_jy > eta_jz > 0.0
    assert eta_jx / eta_jz > 15.0

    # Verify analytic Beer-Lambert + Fresnel stratified model for Ryu (2014)
    bl = helpers.ryu_beer_lambert_transfer(np.array([0.0, 50.0, 100.0, 200.0]))
    assert np.all(np.diff(bl["lee_2d_jx"]) < 0.0)
    assert np.all(np.diff(bl["lee_3d_te"]) < 0.0)
    assert np.all(bl["lee_2d_jx"] > 8.0 * bl["lee_2d_jz"])


def test_wakis_2d_semi_infinite_slab_vs_analytic():
    """Validate 2D Wakis FIT semi-infinite slab escape fraction against 2D Fresnel theory."""
    n_slab = 2.0
    common_kw = dict(
        n_slab=n_slab,
        cpw=10.0,
        lx=2.2e-6,
    )
    res_y = helpers.run_semi_infinite_slab_2d(dipole_pol="y", **common_kw)
    res_x = helpers.run_semi_infinite_slab_2d(dipole_pol="x", **common_kw)
    res_z = helpers.run_semi_infinite_slab_2d(dipole_pol="z", **common_kw)

    assert res_x["cells_across_feature"] >= 4
    assert res_x["dx_background_nm"] <= (280.0 / (10.0 * n_slab)) + 0.05

    # Closed-box Poynting flux in lossless dielectric must match injected power within 3%
    assert abs(res_y["box_over_inj"] - 1.0) < 0.03
    assert abs(res_x["box_over_inj"] - 1.0) < 0.03

    # Match 2D Fresnel escape-cone formulas
    _validate_2d_slab_escape(res_y["lee_top"], res_y["lee_analytic"], tol=0.03)
    _validate_2d_slab_escape(res_x["lee_top"], res_x["lee_analytic"], tol=0.06)
    _validate_2d_slab_escape(res_z["lee_top"], res_z["lee_analytic"], tol=0.02)

    # Polarization hierarchy: in-plane J_x > out-of-plane J_y >> vertical J_z
    assert res_x["lee_top"] > res_y["lee_top"] > 4.0 * res_z["lee_top"]
    _validate_te_over_tm_ratio(res_x["lee_top"], res_z["lee_top"], min_ratio=8.0)


def test_wakis_ryu_stack_te_vs_tm_and_pgan_absorption():
    """Verify TE >> TM extraction and p-GaN absorption in Ryu (2014) planar LED stack."""
    lay0, z_mqw0 = helpers.make_ryu_planar_layers(t_pgan_nm=0.0)
    lay100, z_mqw100 = helpers.make_ryu_planar_layers(t_pgan_nm=100.0)

    common_kw = dict(
        cpw=10.0,
        dz_fine_nm=6.25,
        min_cells_per_layer=4,
        Lx=2.0e-6,
        extra_time_fs=12.0,
        record_n2f=False,
    )
    res_te_0 = helpers.run_2d_dipole_simulation(
        lay0, z_mqw0, dipole_pol="x", **common_kw
    )
    res_tm_0 = helpers.run_2d_dipole_simulation(
        lay0, z_mqw0, dipole_pol="z", **common_kw
    )
    res_te_100 = helpers.run_2d_dipole_simulation(
        lay100, z_mqw100, dipole_pol="x", **common_kw
    )

    assert res_te_100["cells_across_feature"] >= 4
    assert res_te_100["cells_per_skin_depth"] >= 4.0

    # 1. TE extraction strongly dominates TM extraction
    _validate_te_over_tm_ratio(
        res_te_0["lee_top"], res_tm_0["lee_top"], min_ratio=8.0
    )

    # 2. Adding 100 nm of absorbing p-GaN (k = 0.379) strongly attenuates top extraction
    assert res_te_0["lee_top"] > 3.5 * res_te_100["lee_top"]
    assert res_te_100["frac_abs"] > res_te_0["frac_abs"] + 0.15

    # 3. 4-channel power budget sums to 100%
    for res in (res_te_0, res_tm_0, res_te_100):
        total = (
            res["lee_top"]
            + res["frac_bot"]
            + res["frac_sides"]
            + res["frac_abs"]
        )
        assert np.isclose(total, 1.0, atol=1e-9)
        assert 0.0 <= res["frac_abs"] < 1.0


def test_incoherent_vs_coherent_averaging():
    """Verify incoherent power averaging vs unphysical coherent field interference."""
    x = np.linspace(-1.0e-6, 1.0e-6, 80)
    z = np.linspace(-0.5e-6, 0.5e-6, 40)
    X, Z = np.meshgrid(x, z, indexing="ij")
    theta_deg = np.linspace(-85.0, 85.0, 171)
    theta_rad = np.radians(theta_deg)

    # Synthetic dipole runs with counter-propagating lateral phases exp(+/- i k_x X)
    ex1 = np.exp(1j * 8.0e6 * X)
    ex2 = np.exp(-1j * 8.0e6 * X)
    zeros = np.zeros_like(X, dtype=np.complex128)
    d_eta = np.maximum(0.0, np.cos(theta_rad))

    run1 = {
        "lee_top": 0.09,
        "P_inj": 1.0,
        "dft_2d": {"Ex": ex1, "Ey": zeros, "Ez": zeros, "I": np.abs(ex1) ** 2},
        "n2f": {"d_eta_dtheta": d_eta},
    }
    run2 = {
        "lee_top": 0.09,
        "P_inj": 1.0,
        "dft_2d": {"Ex": ex2, "Ey": zeros, "Ez": zeros, "I": np.abs(ex2) ** 2},
        "n2f": {"d_eta_dtheta": d_eta},
    }

    comb = helpers.incoherent_vs_coherent_combine([run1, run2])

    # Incoherent intensity is identically 1.0 everywhere; coherent sum has standing-wave nodes
    _validate_incoherent_power_profile(comb["I_incoh"], np.ones_like(comb["I_incoh"]))
    assert float(np.min(comb["I_coh"])) < 0.05
    assert np.allclose(comb["cum_lee_incoh"], 0.09, atol=1e-12)


def test_mutation_checks_can_fail():
    """Mutation check: ensure validation helpers fail when physics/formulas are corrupted."""
    # Mutation 1: Using 3D escape-cone formula 1/(4 n^2) to validate a 2D slab simulation
    n_slab = 2.0
    eta_2d_true = helpers.escape_cone_2d_fresnel(n_slab, 1.0, dipole="y")
    eta_3d_mutated = 1.0 / (4.0 * n_slab**2)
    with pytest.raises(AssertionError, match="deviates from expected"):
        _validate_2d_slab_escape(eta_2d_true, eta_3d_mutated, tol=0.03)

    # Mutation 2: Swapping TE and TM extraction efficiencies
    eta_te = helpers.escape_cone_2d_fresnel(2.60, 1.0, dipole="x")
    eta_tm = helpers.escape_cone_2d_fresnel(2.60, 1.0, dipole="z")
    with pytest.raises(AssertionError, match="TE/TM extraction ratio"):
        _validate_te_over_tm_ratio(eta_tm, eta_te, min_ratio=8.0)

    # Mutation 3: Substituting coherent sum for incoherent power average
    X = np.linspace(-1.0, 1.0, 50)[:, None]
    ex1 = np.exp(1j * 5.0 * X)
    ex2 = np.exp(-1j * 5.0 * X)
    zeros = np.zeros_like(ex1)
    run1 = {
        "lee_top": 0.1,
        "P_inj": 1.0,
        "dft_2d": {"Ex": ex1, "Ey": zeros, "Ez": zeros, "I": np.abs(ex1) ** 2},
        "n2f": {"d_eta_dtheta": np.ones(10)},
    }
    run2 = {
        "lee_top": 0.1,
        "P_inj": 1.0,
        "dft_2d": {"Ex": ex2, "Ey": zeros, "Ez": zeros, "I": np.abs(ex2) ** 2},
        "n2f": {"d_eta_dtheta": np.ones(10)},
    }
    comb = helpers.incoherent_vs_coherent_combine([run1, run2])
    with pytest.raises(AssertionError, match="deviates from incoherent mean"):
        _validate_incoherent_power_profile(comb["I_coh"], comb["I_incoh"], tol=1e-4)

    # Mutation 4: Resolution validator catches violations of Rules 1, 2, and 3
    valid_meta = planar_led.build_task_mesh_info(planar_led._task_table("quick")[1])
    _validate_task_mesh_resolution(valid_meta)
    with pytest.raises(AssertionError, match="Rule 1 violated"):
        _validate_task_mesh_resolution({**valid_meta, "cpw": 8.0})
    with pytest.raises(AssertionError, match="Rule 2 violated"):
        _validate_task_mesh_resolution({**valid_meta, "cells_across_feature": 2})
    with pytest.raises(AssertionError, match="Rule 3 violated"):
        _validate_task_mesh_resolution({**valid_meta, "cells_per_skin_depth": 2.35})


def test_planar_led_cli_and_cached_data():
    """Verify planar_led.py task tables, cached --full data files, and generated figures."""
    tasks_full = planar_led._task_table("full")
    tasks_quick = planar_led._task_table("quick")
    assert len(tasks_full) == 7
    assert len(tasks_quick) == 7

    required_meta_keys = (
        "dx_background_nm",
        "dx_min_nm",
        "min_feature_nm",
        "cells_across_feature",
        "skin_depth_nm",
        "cells_per_skin_depth",
        "n_cells",
        "n_steps",
    )
    data_dir = TUTORIAL_DIR / "data"
    fig_dir = TUTORIAL_DIR / "figures"
    for t in tasks_full:
        npz_path = data_dir / t["file"]
        assert npz_path.exists(), f"Missing cached full data file: {npz_path}"
        d = np.load(npz_path)
        for k in required_meta_keys:
            assert k in d, f"Missing metadata key {k!r} in {npz_path.name}"
        if not t.get("is_demo", False):
            n_max = 2.60 if t["kind"] == "semi_infinite" else planar_led.N_MAX_REF
            _validate_task_mesh_resolution(
                {
                    "label": t["label"],
                    "is_demo": False,
                    "cpw": float(d["cpw"]),
                    "n_max": n_max,
                    "dx_background_nm": float(d["dx_background_nm"]),
                    "dx_min_nm": float(d["dx_min_nm"]),
                    "min_feature_nm": float(d["min_feature_nm"]),
                    "cells_across_feature": int(d["cells_across_feature"]),
                    "skin_depth_nm": float(d["skin_depth_nm"]),
                    "cells_per_skin_depth": float(d["cells_per_skin_depth"]),
                }
            )

    # Check key physical invariants in cached task_01_ryu_stack_sweep.npz (full cpw=20)
    d1 = np.load(data_dir / "task_01_ryu_stack_sweep.npz")
    assert float(d1["cpw"]) == 20.0
    eta_te = d1["lee_te_2d"]
    eta_tm = d1["lee_tm_2d"]
    eta_te_3d = d1["lee_te_3d"]
    eta_tm_3d = d1["lee_tm_3d"]
    assert np.all(eta_te > 10.0 * eta_tm)
    assert np.all(eta_te_3d > 8.0 * eta_tm_3d)
    # Monotonic decrease of TE and TM LEE with p-GaN thickness
    assert np.all(np.diff(eta_te) < 0.0)
    assert np.all(np.diff(eta_tm) < 0.0)
    assert np.all(np.diff(eta_te_3d) < 0.0)
    assert np.all(np.diff(eta_tm_3d) < 0.0)

    # Check 3D-weighted TE matches Ryu (2014) Fig. 3 within 0.10 percentage points for t_pGaN >= 50 nm
    ref_te_pct = helpers.RYU_2014_REFERENCE["fig3_lee_te_3d_pct"]
    assert np.all(np.abs(eta_te_3d[2:] * 100.0 - ref_te_pct[2:]) < 0.10)

    # Check Rule 3 absorber resolution convergence (< 1% relative between accepted cpw=20 and ref cpw=28)
    d_acc_full = np.load(data_dir / "task_05_conv_cpw20.npz")
    d_ref_full = np.load(data_dir / "task_06_ref_cpw28.npz")
    for k_acc, k_ref in [
        ("kottke_te_t25", "ref_te_t25"),
        ("kottke_te", "ref_te"),
        ("kottke_te_t100", "ref_te_t100"),
        ("kottke_tm", "ref_tm"),
    ]:
        rel_diff = abs(float(d_acc_full[k_acc]) - float(d_ref_full[k_ref])) / float(
            d_ref_full[k_ref]
        )
        assert rel_diff < 0.01, f"Full relative change {rel_diff:.4f} >= 1% for {k_acc}"

    # Check all figures and setup table exist and are non-empty
    expected_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00_setup_table.md",
        "01_led_stack_geometry.png",
        "02_mesh_kottke_vs_staircase.png",
        "03_analytic_escape_cone_2d_vs_3d.png",
        "04_semi_infinite_validation.png",
        "05_field_intensity_te_vs_tm.png",
        "05b_field_intensity_pgan100.png",
        "06_field_wavefronts_te_vs_tm.png",
        "07_dipole_emission_movie.gif",
        "08_lee_vs_pgan_ryu_fig3.png",
        "09_power_budget_4channel.png",
        "10_lee_vs_algan_index_ryu_fig7.png",
        "11_farfield_n2f_vs_lambertian.png",
        "12_incoherent_vs_coherent_convergence.png",
        "13_resolution_convergence_kottke_vs_staircase.png",
        "14_time_trace_and_energy_decay.png",
    ]
    for fname in expected_figs:
        fpath = fig_dir / fname
        min_bytes = 200 if fname.endswith(".md") else 1000
        assert fpath.exists() and fpath.stat().st_size > min_bytes, f"Missing figure {fpath}"


def test_setup_schematic_and_helpers():
    """Verify setup schematic figures and table exist and schematic helper objects are well-formed."""
    for fname, min_sz in (
        ("00a_setup_geometry.png", 10000),
        ("00b_setup_mesh_and_excitation.png", 10000),
        ("00_setup_table.md", 200),
    ):
        fig_path = TUTORIAL_DIR / "figures" / fname
        assert fig_path.exists() and fig_path.stat().st_size > min_sz
    obj_semi = helpers.build_semi_infinite_schematic_objects()
    obj_ryu = helpers.build_ryu_schematic_objects()
    assert len(obj_semi["structures"]) >= 2
    assert len(obj_ryu["structures"]) >= 6
    assert len(obj_ryu["monitors"]) >= 4

