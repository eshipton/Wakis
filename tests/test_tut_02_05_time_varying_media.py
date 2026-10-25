"""Tests for Tutorial 2.5: Time interfaces and photonic time crystals."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from wakis.time_varying import temporal_fresnel_coefficients

REPO_ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = REPO_ROOT / "tutorials" / "02_materials" / "02-05_time_varying_media"
TUT_SCRIPT = TUT_DIR / "time_varying_media.py"


def _load_tutorial_module():
    spec = importlib.util.spec_from_file_location("tut_02_05_time_varying_media", TUT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tutorial_resolution_audit_and_mutations():
    """Verify tutorial grid rules (>= 20 cells/lambda, >= 4 cells/feature) and mutation failure."""
    mod = _load_tutorial_module()
    for quick in (True, False):
        table = mod.get_task_table(quick=quick)
        assert len(table) == 5
        for row in table:
            audit = mod.audit_task_resolution(row)
            assert audit["cpw"] >= 20.0
            assert audit["cells_across_feature"] >= 4.0

    # Mutation check 1: violating cells per wavelength (< 20) must raise ValueError
    bad_cpw = dict(mod.get_task_table(quick=True)[0])
    bad_cpw["nz_pkt"] = 120  # drops cpw below 20
    with pytest.raises(ValueError, match="cpw >= 20"):
        mod.audit_task_resolution(bad_cpw)

    # Mutation check 2: violating cells across feature (< 4) must raise ValueError
    bad_feat = dict(mod.get_task_table(quick=True)[0])
    bad_feat["sigma_z_pkt"] = 0.01  # drops cells_across_feature below 4
    with pytest.raises(ValueError, match="cells_across_feature >= 4"):
        mod.audit_task_resolution(bad_feat)


def test_tutorial_artifacts_and_quantitative_benchmarks():
    """Verify generated data files, figures, GIF size (< 5 MB), and paper benchmark accuracy."""
    data_dir = TUT_DIR / "data"
    fig_dir = TUT_DIR / "figures"

    assert (TUT_DIR / "time_varying_media.ipynb").is_file()
    assert (data_dir / "preflight.json").is_file()
    for idx in range(5):
        assert (data_dir / f"task_{idx}.json").is_file()
        assert (data_dir / f"task_{idx}.npz").is_file()

    expected_pngs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "01_time_varying_setup_schematic.png",
        "02_time_refraction_reflection_snapshots.png",
        "03_temporal_fresnel_and_continuity.png",
        "04_frequency_conversion_and_drude_plasma.png",
        "05_ptc_momentum_band_structure.png",
        "06_ptc_growth_rate_vs_modulation.png",
        "07_moussa_switched_tl_and_temporal_slab.png",
        "08_resolution_convergence_study.png",
    ]
    for fname in expected_pngs:
        fpath = fig_dir / fname
        assert fpath.is_file(), f"Missing figure {fname}"
        assert fpath.stat().st_size > 20_000, f"Figure {fname} appears empty or too small"

    table_md = fig_dir / "00_setup_table.md"
    assert table_md.is_file() and table_md.stat().st_size > 128, "Missing 00_setup_table.md"

    gif_path = fig_dir / "02_time_refraction_reflection.gif"
    assert gif_path.is_file(), "Missing animated GIF 02_time_refraction_reflection.gif"
    assert 20_000 < gif_path.stat().st_size < 5 * 1024 * 1024

    # Task 0: Morgenthaler (1958) temporal Fresnel coefficients, energy & action, D continuity
    t0 = json.loads((data_dir / "task_0.json").read_text())
    assert t0["cpw"] >= 20.0
    assert t0["rms_tau_E_error"] < 1.0e-3
    assert t0["rms_rho_E_error"] < 1.0e-3
    assert t0["rms_energy_ratio_error"] < 5.0e-3
    assert t0["max_action_error"] < 1.0e-3
    assert t0["d_jump_residual"] < 1.0e-13
    assert t0["d_switch_rel_err_conservative"] < 1.0e-13
    assert t0["d_switch_rel_err_mutated"] > 0.5

    # Task 1: Frequency conversion & rapidly ionized Drude plasma (Mendonca & Shukla 2002; Wilks 1988)
    t1 = json.loads((data_dir / "task_1.json").read_text())
    assert abs(t1["redshift_omega_ratio_meas"] - t1["redshift_omega_ratio_theo"]) < 0.025
    assert abs(t1["blueshift_omega_ratio_meas"] - t1["blueshift_omega_ratio_theo"]) < 0.025
    assert t1["rms_drude_tau_E_error"] < 1.0e-3
    assert t1["rms_drude_rho_E_error"] < 1.0e-3
    assert t1["sigma_decay_rel_error"] < 0.04

    # Task 2: Photonic time crystal momentum gap (Lyubarov et al. 2022; Galiffi et al. 2022)
    t2 = json.loads((data_dir / "task_2.json").read_text())
    assert abs(t2["gamma_max_sim_over_omega"] - t2["gamma_max_floquet_over_omega"]) < 2.0e-3
    assert abs(t2["gamma_max_sim_over_omega"] - t2["gamma_max_mathieu_over_omega"]) < 2.0e-3
    assert t2["rms_gamma_over_omega_error"] < 5.0e-3

    # Task 3: Growth scaling vs m, m=0 mutation, and Moussa et al. (2023) temporal slab
    t3 = json.loads((data_dir / "task_3.json").read_text())
    assert abs(t3["gamma_m0_mutation"]) < 1.0e-12
    assert t3["rms_gamma_vs_floquet"] < 1.0e-3
    assert t3["rms_slab_R_error"] < 5.0e-3
    assert t3["rms_slab_T_error"] < 5.0e-3

    # Task 4: Resolution convergence order O(dz^2 + dt^2)
    t4 = json.loads((data_dir / "task_4.json").read_text())
    assert 1.85 <= t4["empirical_convergence_order"] <= 2.15
    assert t4["err_omega_finest"] < t4["err_omega_coarsest"] / 10.0


def test_tutorial_conservative_vs_mutated_execution():
    """Directly execute the tutorial's single-mode solver in conservative vs mutated mode."""
    mod = _load_tutorial_module()
    theo = temporal_fresnel_coefficients(eps1=4.0, eps2=1.6)
    res_cons = mod.simulate_temporal_interface_mode(
        eps1=4.0,
        eps2=1.6,
        nz=128,
        mode_m=4,
        n_switch=25,
        n_post=120,
        mutate_nonconservative=False,
    )
    res_mut = mod.simulate_temporal_interface_mode(
        eps1=4.0,
        eps2=1.6,
        nz=128,
        mode_m=4,
        n_switch=25,
        n_post=120,
        mutate_nonconservative=True,
    )

    assert res_cons["d_switch_rel_err"] < 1.0e-14
    assert res_cons["d_jump_residual"] < 1.0e-14
    assert abs(res_cons["tau_E_sim"] - theo["tau_E"]) < 2.0e-3
    assert abs(res_cons["rho_E_sim"] - abs(theo["rho_E"])) < 2.0e-3

    # Non-conservative mutation violates D continuity and produces wrong Fresnel amplitudes
    assert res_mut["d_switch_rel_err"] > 0.5
    assert abs(res_mut["tau_E_sim"] - theo["tau_E"]) > 0.2
