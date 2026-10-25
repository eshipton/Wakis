# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics and numerical validation tests for Tutorial 5.2:
Hybrid Euler 90-degree SOI waveguide bend (`tutorials/05_guided_waves/05-02_euler_bend/euler_bend.py`).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = REPO_ROOT / "tutorials" / "05_guided_waves" / "05-02_euler_bend"
SCRIPT_PATH = TUTORIAL_DIR / "euler_bend.py"


def _load_tutorial_module():
    spec = importlib.util.spec_from_file_location("euler_bend_tutorial", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hybrid_euler_geometry_and_curvature_linearity():
    """
    Verify that `solve_hybrid_euler_params` and `bend_centerline_and_curvature`
    satisfy the exact C^2 clothoid-circle matching conditions at R_eff = 4.0 um
    (A = 2.4 um) and that the clothoid curvature is linear in arc length (R^2 > 0.9999).
    """
    mod = _load_tutorial_module()
    info = mod.solve_hybrid_euler_params(r_eff=4.0e-6, a_ratio=0.60)

    assert info["A"] == pytest.approx(2.4e-6, rel=1e-12)
    assert info["r_min"] == pytest.approx(info["A"] ** 2 / info["L_max"], rel=1e-12)
    assert info["a"] + info["b"] == pytest.approx(4.0e-6, rel=1e-9)
    assert 2.9e-6 < info["r_min"] < 3.1e-6
    assert 0.35 < info["p"] < 0.45

    s_h, cl_h, theta_h, kap_h, _ = mod.bend_centerline_and_curvature(
        r_eff=4.0e-6, bend_type="euler", a_ratio=0.60, n=401
    )
    # Centerline starts at (0, 0) with theta = 0 and ends at (R_eff, R_eff) with theta = pi/2
    assert np.allclose(cl_h[0], [0.0, 0.0], atol=1e-12)
    assert np.allclose(cl_h[-1], [4.0e-6, 4.0e-6], rtol=1e-4)
    assert float(theta_h[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(theta_h[-1]) == pytest.approx(0.5 * np.pi, rel=1e-6)

    # First clothoid segment: curvature kappa(s) = s / A^2 is linear in arc length
    m_clothoid = s_h <= info["L_max"]
    s_seg = s_h[m_clothoid]
    k_seg = kap_h[m_clothoid]
    slope, intercept = np.polyfit(s_seg, k_seg, 1)
    assert slope == pytest.approx(1.0 / (info["A"] ** 2), rel=1e-3)
    assert abs(intercept) < 1e-3 * float(np.max(k_seg))
    k_fit = slope * s_seg + intercept
    r_sq = 1.0 - float(np.sum((k_seg - k_fit) ** 2) / np.sum((k_seg - np.mean(k_seg)) ** 2))
    assert r_sq > 0.9999

    # Circular bend has constant curvature 1 / R_eff
    _, cl_c, _, kap_c, _ = mod.bend_centerline_and_curvature(
        r_eff=4.0e-6, bend_type="circular", n=201
    )
    assert np.allclose(cl_c[-1], [4.0e-6, 4.0e-6], rtol=1e-6)
    assert np.allclose(kap_c, 1.0 / 4.0e-6, rtol=1e-12)


def test_task_tables_resolution_floor_and_feature_cells():
    """
    Verify that every task in both the ``--quick`` (14 tasks) and ``--full``
    (16 tasks) tables satisfies:
    1. ``cpw >= 10`` (cells per wavelength in the EIM slab core).
    2. ``n_feature >= 4`` and actual measured ``cells_across_feature >= 4`` on
       the constructed ``PICSimulation`` mesh (without running the solver).
    3. ``kottke_nonuniform`` (`NonUniformEulerBendPIC`) reduces total cells
       relative to the uniform baseline (`kottke_L3` / `kottke_L2`) while
       retaining >= 10 cells across the 400 nm waveguide core, non-square
       straight leads, and graded coarse cladding cells.
    4. Mutation check: when ``n_feature`` requires finer spacing than the
       background grid (e.g. ``cpw=10, n_feature=12``), ``build_mesh_overrides``
       activates a local refinement box and increases ``cells_across_feature``
       to ``>= 12``.
    """
    mod = _load_tutorial_module()
    quick_tasks = mod.get_task_table(quick=True)
    full_tasks = mod.get_task_table(quick=False)
    assert len(quick_tasks) == 14
    assert len(full_tasks) == 16

    for mode_name, table in (("quick", quick_tasks), ("full", full_tasks)):
        res_by_name = {}
        for t in table:
            assert int(t["cpw"]) >= 10, (
                f"{mode_name} task {t['name']} has cpw={t['cpw']} < 10"
            )
            assert int(t["n_feature"]) >= 4, (
                f"{mode_name} task {t['name']} has n_feature={t['n_feature']} < 4"
            )
            pic, _, _, _, _ = mod.build_task_simulation(t, mesh_only=True)
            r_eff = float(t["r_eff"])
            min_feat = float(t.get("min_feature", mod.MIN_FEATURE))
            res = mod.measure_mesh_resolution(
                pic, r_eff=r_eff, min_feature=min_feat
            )
            res_by_name[t["name"]] = res
            assert res["cells_across_feature"] >= 4.0, (
                f"{mode_name} task {t['name']} has only "
                f"{res['cells_across_feature']:.2f} cells across feature (< 4)"
            )
            assert res["cells_across_feature"] >= float(t["n_feature"]) - 1e-6, (
                f"{mode_name} task {t['name']} measured "
                f"{res['cells_across_feature']:.2f} cells < target N={t['n_feature']}"
            )
            assert res["dx_min_nm"] <= res["dx_background_nm"] + 1e-6
            assert res["n_cells"] > 0

        base_name = "kottke_L2" if mode_name == "quick" else "kottke_L3"
        res_nu = res_by_name["kottke_nonuniform"]
        res_base = res_by_name[base_name]
        assert res_nu["n_cells"] < res_base["n_cells"] * 0.85, (
            f"Expected kottke_nonuniform ({res_nu['n_cells']} cells) < 0.85 * "
            f"{base_name} ({res_base['n_cells']} cells)"
        )
        assert res_nu["du_clad_max_nm"] > res_nu["dx_min_nm"] * 2.0
        assert res_nu["aspect_ratio_lead"] > 1.05

    # Mutation check on NonUniformEulerBendPIC: tightening du_clad from 68 nm to 26.64 nm increases cell count
    t_nu_fine = dict(full_tasks[-1])
    structs, ports, domain, _ = mod.build_bend_waveguide(r_eff=t_nu_fine["r_eff"], bend_type="euler")
    pic_nu_default = mod.NonUniformEulerBendPIC(
        mod.WL_MIN, mod.WL_MAX, structures=structs, ports=ports, n_core=mod.N_SI, n_clad=mod.N_SIO2,
        two_d=True, averaging="kottke", cells_per_wavelength=20, domain=domain, n_pml=10, mesh_only=True
    )
    pic_nu_dense = mod.NonUniformEulerBendPIC(
        mod.WL_MIN, mod.WL_MAX, structures=structs, ports=ports, n_core=mod.N_SI, n_clad=mod.N_SIO2,
        two_d=True, averaging="kottke", cells_per_wavelength=20, domain=domain, n_pml=10,
        du_clad=26.64e-9, du_lead=26.64e-9, mesh_only=True
    )
    assert pic_nu_default.grid.Nx * pic_nu_default.grid.Ny < pic_nu_dense.grid.Nx * pic_nu_dense.grid.Ny * 0.85

    # Mutation check on mesh_overrides: requesting N=12 at cpw=10 forces refinement
    t_override = dict(quick_tasks[0])
    t_override["cpw"] = 10
    t_override["n_feature"] = 12
    overrides = mod.build_mesh_overrides(
        r_eff=t_override["r_eff"],
        width=mod.MIN_FEATURE,
        cpw=10,
        n_feature=12,
    )
    assert len(overrides) == 1, "Expected mesh_overrides to activate when w/N < dx_bg"
    pic_ov, _, _, _, _ = mod.build_task_simulation(t_override, mesh_only=True)
    r_eff = float(t_override["r_eff"])
    res_ov = mod.measure_mesh_resolution(
        pic_ov, r_eff=r_eff, min_feature=mod.MIN_FEATURE
    )
    assert res_ov["cells_across_feature"] >= 12.0 - 1e-6
    assert res_ov["dx_min_nm"] < res_ov["dx_background_nm"]


def test_euler_vs_circular_physics_and_mutation():
    """
    Fast 2D EIM FIT verification (< 45 s) at L1 resolution (cpw = 10, N = 4):
    1. Euler bend loss <= circular bend loss at equal footprint at the smallest
       swept radius (R_eff = 2.5 um) and at R_eff = 4.0 um.
    2. Bending loss decreases with effective radius (R_eff = 4.0 um < R_eff = 2.5 um).
    3. Kottke subpixel smoothing yields lower insertion loss and > 50x lower
       spurious reflection than staircasing on the curved bend.
    4. Mutation check: tightening the circular bend radius to a strongly
       radiating regime (R_eff = 1.4 um) increases insertion loss by > 0.12 dB
       compared to R_eff = 2.5 um.
    """
    from wakis.templates.photonics import PICSimulation

    mod = _load_tutorial_module()

    def _run_bend(r_eff, bend_type="euler", averaging="kottke", cpw=10):
        structs, ports, domain, _ = mod.build_bend_waveguide(
            r_eff=r_eff,
            bend_type=bend_type,
            n_bend=101,
        )
        overrides = mod.build_mesh_overrides(
            r_eff=r_eff,
            width=mod.MIN_FEATURE,
            cpw=cpw,
            n_feature=4,
        )
        pic = PICSimulation(
            wavelength_min=mod.WL_MIN,
            wavelength_max=mod.WL_MAX,
            n_freqs=15,
            structures=structs,
            ports=ports,
            n_core=mod.N_SI,
            n_clad=mod.N_SIO2,
            wavelength_ref=mod.WL_REF,
            two_d=True,
            averaging=averaging,
            cells_per_wavelength=cpw,
            domain=domain,
            n_pml=8,
            mesh_overrides=overrides,
        )
        s = pic.run_s_parameters(
            drive=["in"], energy_decay=5e-4, collect_fields=False
        )
        mid = slice(3, -3)
        t_vals = np.abs(s[("out", "in")])[mid] ** 2
        r_vals = np.abs(s[("in", "in")])[mid] ** 2
        # Passivity check (accounting for fixed-f0 mode projection dispersion)
        assert float(np.mean(t_vals + r_vals)) <= 1.001
        assert float(np.max(t_vals + r_vals)) <= 1.005
        loss_db = -10.0 * np.log10(float(np.mean(t_vals)))
        r_mean = float(np.mean(r_vals))
        return loss_db, r_mean

    loss_circ_2p5, _ = _run_bend(2.5e-6, bend_type="circular", averaging="kottke")
    loss_euler_2p5, _ = _run_bend(2.5e-6, bend_type="euler", averaging="kottke")
    loss_circ_4p0, _ = _run_bend(4.0e-6, bend_type="circular", averaging="kottke")
    loss_euler_4p0, r_kottke_4p0 = _run_bend(
        4.0e-6, bend_type="euler", averaging="kottke"
    )
    loss_stair_4p0, r_stair_4p0 = _run_bend(
        4.0e-6, bend_type="euler", averaging="staircase"
    )

    # 1. Euler bend loss <= circular bend loss at smallest radius (2.5 um) and at 4.0 um
    assert loss_euler_2p5 < loss_circ_2p5, (
        f"Expected Euler loss ({loss_euler_2p5:.4f} dB) < circular loss "
        f"({loss_circ_2p5:.4f} dB) at R_eff = 2.5 um"
    )
    assert loss_euler_4p0 < loss_circ_4p0, (
        f"Expected Euler loss ({loss_euler_4p0:.4f} dB) < circular loss "
        f"({loss_circ_4p0:.4f} dB) at R_eff = 4.0 um"
    )

    # 2. Loss decreases as effective radius increases from 2.5 um to 4.0 um
    assert loss_circ_4p0 < loss_circ_2p5, (
        f"Expected circular bend loss to decrease with radius: "
        f"loss(4.0 um)={loss_circ_4p0:.4f} dB >= loss(2.5 um)={loss_circ_2p5:.4f} dB"
    )
    assert loss_euler_4p0 < loss_euler_2p5, (
        f"Expected Euler bend loss to decrease with radius: "
        f"loss(4.0 um)={loss_euler_4p0:.4f} dB >= loss(2.5 um)={loss_euler_2p5:.4f} dB"
    )

    # 3. Kottke subpixel smoothing outperforms staircasing on the curved bend
    assert loss_euler_4p0 < loss_stair_4p0, (
        f"Expected Kottke loss ({loss_euler_4p0:.4f} dB) < staircase loss ({loss_stair_4p0:.4f} dB)"
    )
    assert r_kottke_4p0 * 50.0 < r_stair_4p0, (
        f"Expected Kottke reflection ({r_kottke_4p0:.3e}) * 50 < staircase reflection ({r_stair_4p0:.3e})"
    )

    # 4. Mutation check: shrinking circular bend radius to R_eff = 1.4 um strongly increases radiation loss
    loss_circ_mut_1p4, _ = _run_bend(1.4e-6, bend_type="circular", averaging="kottke")
    assert loss_circ_mut_1p4 > loss_circ_2p5 + 0.12, (
        f"Mutation check failed: R_eff=1.4 um circular loss ({loss_circ_mut_1p4:.4f} dB) "
        f"did not exceed R_eff=2.5 um loss ({loss_circ_2p5:.4f} dB) by > 0.12 dB"
    )


def test_saved_quick_study_data_and_figures():
    """
    Verify the saved study dataset (``--full`` ``data/task_0..14.{npz,json}`` or
    ``--quick`` ``data/task_0..12.{npz,json}``) and rendered figure suite
    (``figures/*``) satisfy all resolution-ladder, monotonicity, Euler <= circular,
    and convergence invariants.
    """
    import json

    mod = _load_tutorial_module()
    data_dir = TUTORIAL_DIR / "data"
    fig_dir = TUTORIAL_DIR / "figures"
    t4_path = data_dir / "task_4.npz"
    assert t4_path.exists(), f"Missing {t4_path}"
    is_quick_data = (
        str(np.load(t4_path, allow_pickle=False)["name"]) == "kottke_reference"
    )
    expected_tasks = mod.get_task_table(quick=is_quick_data)
    assert len(expected_tasks) == (14 if is_quick_data else 16)

    required_res_keys = (
        "min_feature_nm",
        "dx_background_nm",
        "dx_min_nm",
        "du_lead_nm",
        "du_clad_max_nm",
        "aspect_ratio_lead",
        "cells_across_feature",
        "n_cells",
        "n_steps",
    )

    tasks_data = {}
    for t in expected_tasks:
        idx = t["index"]
        npz_path = data_dir / f"task_{idx}.npz"
        json_path = data_dir / f"task_{idx}.json"
        assert npz_path.exists(), f"Missing {npz_path}"
        assert json_path.exists(), f"Missing {json_path}"

        d = dict(np.load(npz_path, allow_pickle=False))
        j = json.loads(json_path.read_text())
        assert str(d["name"]) == t["name"]
        assert j["name"] == t["name"]
        assert int(d["cpw"]) >= 10
        assert int(j["cpw"]) >= 10

        for k in required_res_keys:
            assert k in d, f"Missing key {k!r} in {npz_path.name}"
            assert k in j, f"Missing key {k!r} in {json_path.name}"

        assert float(d["cells_across_feature"]) >= 4.0
        assert float(j["cells_across_feature"]) >= 4.0
        assert int(d["n_cells"]) > 0
        assert int(d["n_steps"]) > 0
        tasks_data[t["name"]] = d

    def _loss_db(d):
        w = d["wavelengths"] * 1e9
        m = (w >= 1515.0) & (w <= 1585.0)
        t_mean = float(np.mean(np.abs(d["S21"][m]) ** 2))
        return -10.0 * np.log10(max(t_mean, 1e-12))

    # Check radius sweep monotonicity and Euler <= circular at all 4 radii
    radii = (2.5, 3.0, 3.5, 4.0)
    loss_c_list, loss_e_list = [], []
    for r in radii:
        tag = f"{r:.1f}".replace(".", "p")
        lc = _loss_db(tasks_data[f"sweep_circular_R{tag}"])
        le = _loss_db(tasks_data[f"sweep_euler_R{tag}"])
        assert le <= lc, f"At R_eff={r} um, Euler loss {le:.4f} dB > circular loss {lc:.4f} dB"
        loss_c_list.append(lc)
        loss_e_list.append(le)

    assert all(
        loss_c_list[i] > loss_c_list[i + 1] for i in range(len(radii) - 1)
    ), f"Circular bend loss not monotonically decreasing with radius: {loss_c_list}"
    assert all(
        loss_e_list[i] > loss_e_list[i + 1] for i in range(len(radii) - 1)
    ), f"Euler bend loss not monotonically decreasing with radius: {loss_e_list}"

    # Check Kottke vs staircase convergence across all ladder levels against kottke_reference
    t_ref = 10.0 ** (-0.1 * _loss_db(tasks_data["kottke_reference"]))
    conv_levels = ("L1", "L2") if is_quick_data else ("L1", "L2", "L3")
    for lvl in conv_levels:
        t_k = 10.0 ** (-0.1 * _loss_db(tasks_data[f"kottke_{lvl}"]))
        t_s = 10.0 ** (-0.1 * _loss_db(tasks_data[f"staircase_{lvl}"]))
        assert abs(t_k - t_ref) < abs(t_s - t_ref)
        rk = float(np.mean(np.abs(tasks_data[f"kottke_{lvl}"]["S11"][5:-5]) ** 2))
        rs = float(np.mean(np.abs(tasks_data[f"staircase_{lvl}"]["S11"][5:-5]) ** 2))
        assert rk < rs
        if lvl in ("L1", "L2"):
            assert rk * 50.0 < rs

    if not is_quick_data:
        t_k_l2 = 10.0 ** (-0.1 * _loss_db(tasks_data["kottke_L2"]))
        t_k_l3 = 10.0 ** (-0.1 * _loss_db(tasks_data["kottke_L3"]))
        t_k_nu = 10.0 ** (-0.1 * _loss_db(tasks_data["kottke_nonuniform"]))
        assert abs(t_k_l3 - t_k_l2) / t_k_l3 < 1e-3
        assert abs(t_k_l3 - t_ref) / t_ref < 2e-3
        assert abs(t_k_nu - t_ref) / t_ref < 2e-3
        assert int(tasks_data["kottke_nonuniform"]["n_cells"]) < int(
            tasks_data["kottke_L3"]["n_cells"]
        )

    # Check all 18 figures (including 00_setup_schematic.png and 04b_mesh_upgrade_comparison.png) exist and are non-empty
    figs = list(fig_dir.glob("*.png")) + list(fig_dir.glob("*.gif"))
    assert len(figs) >= 18, f"Expected >= 18 figures in {fig_dir}, found {len(figs)}"
    assert (fig_dir / "00_setup_schematic.png").stat().st_size > 10_000
    assert (fig_dir / "04b_mesh_upgrade_comparison.png").stat().st_size > 10_000
    assert (fig_dir / "10_pulse_propagation.gif").stat().st_size > 1024
    assert (fig_dir / "10_pulse_propagation.gif").stat().st_size < 5 * 1024 * 1024

