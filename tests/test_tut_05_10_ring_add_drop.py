# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics, convergence, resolution-ladder, and mutation tests for Tutorial 5.10:
Microring resonator add-drop filter (`tutorials/05_guided_waves/05-10_ring_add_drop/`).
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TUTORIAL_DIR = REPO_ROOT / "tutorials" / "05_guided_waves" / "05-10_ring_add_drop"
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    FastModeMonitor,
    FastRingPolySlab,
    attach_fast_2d_te_step,
    cmt_add_drop_spectra,
    find_ring_resonances,
)
from ring_add_drop import (  # noqa: E402
    DATA_DIR,
    FIG_DIR,
    R_RING,
    WL_REF,
    build_ring_add_drop,
    compute_resonance_shift_metrics,
    compute_waveguide_dispersion,
    load_study_results,
    study_tasks,
)

from wakis.mode_solver import ModeMonitor, ModeSource  # noqa: E402
from wakis.templates.photonics import PolySlab, ring  # noqa: E402


def test_resolution_ladder_and_cells_across_feature_with_mutation():
    """
    Verify that every non-demo task in both `--quick` and `--full` tables obeys:
      1. `cpw >= 10` cells per wavelength in the slab effective index, and
      2. `cells_across_feature >= 4` measured directly from the built mesh (`sim.grid.y`).
    Also includes mutation checks on `coupler_gap_mesh_overrides`,
    `measure_cells_across_gap`, and `FastRingPolySlab`.
    """
    for quick_mode, expected_len in ((True, 9), (False, 11)):
        tasks = study_tasks(quick=quick_mode)
        assert len(tasks) == expected_len
        demo_keys = []
        for t in tasks:
            assert t["cpw"] >= 10, f"Task {t['key']} has cpw={t['cpw']} < 10"
            sim = build_ring_add_drop(
                cells_per_wavelength=t["cpw"],
                n_cells_across_gap=t["n_cells_across_gap"],
                subpixel_smoothing=t["subpixel"],
                gap_m=t["gap_nm"] * 1e-9,
                build_solver=False,
            )
            assert sim.grid.Nz == 1
            assert sim.n_cells == int(sim.grid.Nx) * int(sim.grid.Ny)
            assert np.isclose(sim.min_feature_nm, t["gap_nm"], rtol=1e-6)

            if t["is_demo"]:
                demo_keys.append(t["key"])
                assert t["n_cells_across_gap"] is None
                assert 1.0 < sim.cells_across_feature < 2.5
            else:
                assert t["n_cells_across_gap"] is not None and t["n_cells_across_gap"] >= 4
                assert sim.cells_across_feature >= 4.0 - 1e-6, (
                    f"Non-demo task {t['key']} (quick={quick_mode}) has only "
                    f"{sim.cells_across_feature:.2f} cells across the {t['gap_nm']:.0f} nm gap"
                )
                assert np.isclose(
                    sim.cells_across_feature,
                    float(t["n_cells_across_gap"]),
                    rtol=1e-2,
                )
                assert sim.dx_min_nm <= (t["gap_nm"] / t["n_cells_across_gap"]) * 1.01

        assert demo_keys == ["underresolved_kottke", "underresolved_staircase"]

    # Mutation 1: Doubling N from 4 to 8 at fixed cpw=10 halves dx_min_nm and doubles cells_across_feature
    sim_n4 = build_ring_add_drop(cells_per_wavelength=10, n_cells_across_gap=4, build_solver=False)
    sim_n8 = build_ring_add_drop(cells_per_wavelength=10, n_cells_across_gap=8, build_solver=False)
    assert np.isclose(sim_n8.cells_across_feature / sim_n4.cells_across_feature, 2.0, rtol=1e-3)
    assert np.isclose(sim_n4.dx_min_nm / sim_n8.dx_min_nm, 2.0, rtol=1e-3)

    # Mutation 2: Sweeping gap_m (60, 100, 150 nm) at fixed N=6 scales dx_min_nm proportionally
    sim_g60 = build_ring_add_drop(
        cells_per_wavelength=14, n_cells_across_gap=6, gap_m=60e-9, build_solver=False
    )
    sim_g150 = build_ring_add_drop(
        cells_per_wavelength=14, n_cells_across_gap=6, gap_m=150e-9, build_solver=False
    )
    assert np.isclose(sim_g60.cells_across_feature, 6.0, rtol=1e-3)
    assert np.isclose(sim_g150.cells_across_feature, 6.0, rtol=1e-3)
    assert np.isclose(sim_g150.dx_min_nm / sim_g60.dx_min_nm, 150.0 / 60.0, rtol=1e-3)


def test_ring_geometry_and_eim_dispersion():
    """
    Verify the 4-port microring geometry, vertical SOI EIM slab index,
    channel mode dispersion, theoretical free spectral range, and exactness
    of `FastRingPolySlab.fill()` vs `PolySlab.fill()`.
    """
    disp = compute_waveguide_dispersion(n_pts=21)
    assert np.isclose(disp["neff_slab_ref"], 2.8478, atol=2e-3)

    neff_1550 = float(np.interp(WL_REF, disp["wl_m"], disp["neff_eim"]))
    ng_1550 = float(np.interp(WL_REF, disp["wl_m"], disp["ng_eim"]))
    fsr_1550 = float(np.interp(WL_REF, disp["wl_m"], disp["fsr_eim_nm"]))
    fsr_3d_1550 = float(np.interp(WL_REF, disp["wl_m"], disp["fsr_3d_eq_nm"]))

    assert 2.38 < neff_1550 < 2.43
    assert 3.20 < ng_1550 < 3.40
    assert np.isclose(fsr_1550, (WL_REF**2 / (ng_1550 * 2.0 * np.pi * R_RING)) * 1e9, rtol=1e-3)
    assert 22.5 < fsr_1550 < 24.0
    assert 17.5 < fsr_3d_1550 < 19.8
    assert fsr_3d_1550 < fsr_1550

    sim = build_ring_add_drop(
        cells_per_wavelength=10,
        n_cells_across_gap=4,
        subpixel_smoothing=True,
        build_solver=False,
    )
    assert len(sim.structures_3d) == 3
    assert len(sim.structures_3d[2].holes) == 1
    assert set(sim.port_map.keys()) == {"in", "through", "drop", "add"}
    assert sim.grid.Nz == 1
    assert sim.cells_across_feature >= 4.0

    # Verify FastRingPolySlab.fill() is bit-for-bit identical to PolySlab.fill()
    outer, holes = ring(center=(0.0, 0.0), radius=2.0e-6, width=0.43e-6, n=128)
    slab_std = PolySlab(outer, -0.11e-6, 0.11e-6, material=8.1, holes=holes)
    slab_fast = FastRingPolySlab(
        outer,
        -0.11e-6,
        0.11e-6,
        material=8.1,
        holes=holes,
        center=(0.0, 0.0),
        r_in=2.0e-6 - 0.215e-6,
        r_out=2.0e-6 + 0.215e-6,
    )
    gx = np.linspace(-2.4e-6, 2.4e-6, 41)
    X, Y, Z = np.meshgrid(gx, gx, np.array([0.0]), indexing="ij")
    h_val = float(gx[1] - gx[0])
    np.testing.assert_array_equal(
        slab_fast.fill(X, Y, Z, h_val, h_val, 0.22e-6),
        slab_std.fill(X, Y, Z, h_val, h_val, 0.22e-6),
    )


def test_fast_mode_monitor_and_fast_2d_te_step_exactness():
    """
    Verify that `FastModeMonitor` matches `wakis.mode_solver.ModeMonitor` to
    machine precision (< 1e-11) and `attach_fast_2d_te_step` is bit-for-bit
    identical to `SolverFIT3D._one_step_cpml` on a resolved (cpw=10, N=4) mesh.
    """
    sim_ref = build_ring_add_drop(
        cells_per_wavelength=10,
        n_cells_across_gap=4,
        radius_m=2.0e-6,
        subpixel_smoothing=True,
        n_freqs=31,
    )
    sim_fast = build_ring_add_drop(
        cells_per_wavelength=10,
        n_cells_across_gap=4,
        radius_m=2.0e-6,
        subpixel_smoothing=True,
        n_freqs=31,
    )

    sim_ref.solver.one_step = sim_ref.solver._one_step_cpml
    attach_fast_2d_te_step(sim_fast.solver)

    in_port = sim_ref.port_map["in"]
    in_c, in_s, in_inward = sim_ref._port_plane_args(in_port)
    k_mon = int(np.argmin(np.abs(sim_ref.grid.x[:-1] - in_c[0])))
    k_src = int(np.clip(k_mon - 2 * in_inward, 2, len(sim_ref.grid.x) - 3))
    src_c = in_c.copy()
    src_c[0] = sim_ref.grid.x[k_src]

    src_ref = ModeSource(
        sim_ref.sim.pulse,
        center=src_c,
        size=in_s,
        direction="+",
        mode_index=0,
        mode_frequency=sim_ref.f_center,
    )
    src_fast = ModeSource(
        sim_fast.sim.pulse,
        center=src_c,
        size=in_s,
        direction="+",
        mode_index=0,
        mode_frequency=sim_fast.f_center,
    )

    tp_c, tp_s, _ = sim_ref._port_plane_args(sim_ref.port_map["through"])
    mon_std = ModeMonitor(
        sim_ref.frequencies,
        center=tp_c,
        size=tp_s,
        n_modes=1,
        mode_frequency=sim_ref.f_center,
    )
    mon_fast = FastModeMonitor(
        sim_fast.frequencies,
        center=tp_c,
        size=tp_s,
        n_modes=1,
        mode_frequency=sim_fast.f_center,
    )

    sim_ref.solver.emsolve(40, source=[src_ref], monitors=[mon_std])
    sim_fast.solver.emsolve(40, source=[src_fast], monitors=[mon_fast])

    np.testing.assert_allclose(sim_fast.solver.E.array, sim_ref.solver.E.array, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(sim_fast.solver.H.array, sim_ref.solver.H.array, rtol=1e-11, atol=1e-11)
    assert np.isclose(sim_ref.solver.field_energy(), sim_fast.solver.field_energy(), rtol=1e-11)

    ap_std, am_std = mon_std.get_amplitudes(0)
    ap_fast, am_fast = mon_fast.get_amplitudes(0)
    np.testing.assert_allclose(ap_fast, ap_std, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(am_fast, am_std, rtol=1e-10, atol=1e-12)


def test_yariv_cmt_and_lorentzian_peak_finder_mutation():
    """
    Verify Yariv (2000) analytical transfer-matrix spectra and Lorentzian
    peak extraction, including analytical mutation checks on radius R and
    power coupling kappa^2.
    """
    wl_m = np.linspace(1.50e-6, 1.60e-6, 1001)
    neff_0 = 2.403
    t_thru_5, t_drop_5 = cmt_add_drop_spectra(wl_m, neff_0, radius_m=5.0e-6, kappa2=0.06, a_rt=0.996)
    res_5 = find_ring_resonances(wl_m, np.sqrt(t_thru_5), np.sqrt(t_drop_5))

    assert len(res_5["resonances"]) == 3
    expected_fsr_5 = (1.55e-6**2 / (neff_0 * 2.0 * np.pi * 5.0e-6)) * 1e9
    assert np.isclose(res_5["mean_fsr_nm"], expected_fsr_5, rtol=0.03)

    # Mutation 1: Increasing ring radius R from 5.0 um to 7.5 um reduces FSR by 5.0 / 7.5 = 2/3
    t_thru_75, t_drop_75 = cmt_add_drop_spectra(wl_m, neff_0, radius_m=7.5e-6, kappa2=0.06, a_rt=0.996)
    res_75 = find_ring_resonances(wl_m, np.sqrt(t_thru_75), np.sqrt(t_drop_75))
    assert len(res_75["resonances"]) > len(res_5["resonances"])
    assert np.isclose(res_75["mean_fsr_nm"] / res_5["mean_fsr_nm"], 5.0 / 7.5, rtol=0.04)

    # Mutation 2: Doubling power coupling kappa^2 from 0.05 to 0.10 broadens FWHM and halves Q_L
    _, t_drop_weak = cmt_add_drop_spectra(wl_m, neff_0, radius_m=5.0e-6, kappa2=0.05, a_rt=1.0)
    _, t_drop_strong = cmt_add_drop_spectra(wl_m, neff_0, radius_m=5.0e-6, kappa2=0.10, a_rt=1.0)
    res_weak = find_ring_resonances(wl_m, np.sqrt(1.0 - t_drop_weak), np.sqrt(t_drop_weak))
    res_strong = find_ring_resonances(wl_m, np.sqrt(1.0 - t_drop_strong), np.sqrt(t_drop_strong))
    assert res_strong["mean_Q_loaded"] < 0.55 * res_weak["mean_Q_loaded"]


def test_ring_add_drop_resonances_fsr_and_passivity():
    """
    Verify that the 2D Wakis FIT microring add-drop filter (`kottke_L3` in
    `--full` mode, or `kottke_L2` in `--quick` mode) records all required
    resolution metadata (`min_feature_nm`, `dx_background_nm`, `dx_min_nm`,
    `cells_across_feature`, `n_cells`, `n_steps`) and produces 4 sharp
    Lorentzian resonances across 1500-1600 nm with FSR matching 2D EIM
    group-index theory, high drop transmission, deep through extinction,
    bounded 4-port power balance, and strong on-resonance cavity field
    enhancement.
    """
    results = load_study_results(DATA_DIR)
    fine_key = "kottke_L3" if "kottke_L3" in results else "kottke_L2"
    assert fine_key in results, f"Expected precomputed {fine_key} data in tutorials/05_guided_waves/05-10_ring_add_drop/data/"
    d = results[fine_key]

    req_fields = (
        "min_feature_nm",
        "dx_background_nm",
        "dx_min_nm",
        "cells_across_feature",
        "n_cells",
        "n_steps",
    )
    for k_task, entry in results.items():
        for rf in req_fields:
            assert rf in entry, f"Missing {rf} in NPZ for task {k_task}"
    for jpath in sorted(DATA_DIR.glob("task_*.json")):
        jmeta = json.loads(jpath.read_text())
        for rf in req_fields:
            assert rf in jmeta, f"Missing {rf} in {jpath.name}"
        if not jmeta.get("is_demo", False):
            assert float(jmeta["cpw"]) >= 10.0, f"Non-demo task {jpath.name} has cpw < 10"
            assert float(jmeta["cells_across_feature"]) >= 4.0 - 1e-3, (
                f"Non-demo task {jpath.name} has cells_across_feature < 4"
            )

    if fine_key == "kottke_L3":
        assert float(d["cells_across_feature"]) >= 8.0 - 1e-3
        assert int(d["cpw"]) == 20
    else:
        assert float(d["cells_across_feature"]) >= 6.0 - 1e-3
        assert int(d["cpw"]) == 14

    res = find_ring_resonances(
        d["wavelengths"],
        d["S_through_in"],
        d["S_drop_in"],
        s_in=d["S_in_in"],
        s_add=d["S_add_in"],
    )
    assert len(res["resonances"]) == 4

    # FSR matches theoretical lambda^2 / (n_g * L) = 23.17 nm within 3%
    assert np.isclose(res["mean_fsr_nm"], 23.17, rtol=0.03)

    # Every resonance has strong drop routing (> 85%) and deep through extinction (> 18 dB)
    for r in res["resonances"]:
        assert r["t_drop_peak"] > 0.85, f"Low drop peak {r['t_drop_peak']:.3f} at {r['wl0_nm']:.2f} nm"
        assert r["extinction_db"] > 18.0, f"Low extinction {r['extinction_db']:.1f} dB at {r['wl0_nm']:.2f} nm"
        assert 450.0 < r["Q_loaded"] < 6000.0

    # Off-resonance (1550 nm): high through transmission (> 90%) and low drop (< 5%)
    t_thru_1550 = float(np.interp(1550.0, res["wl_nm"], res["t_through"]))
    t_drop_1550 = float(np.interp(1550.0, res["wl_nm"], res["t_drop"]))
    assert t_thru_1550 > 0.90
    assert t_drop_1550 < 0.05

    # 4-port power balance across the interior C-band [1505, 1595] nm
    interior = (res["wl_nm"] >= 1505.0) & (res["wl_nm"] <= 1595.0)
    total_power = (
        res["t_through"][interior]
        + res["t_drop"][interior]
        + res["t_in"][interior]
        + res["t_add"][interior]
    )
    assert 0.88 <= float(np.min(total_power))
    assert float(np.max(total_power)) <= 1.15

    # On-resonance cavity intensity enhancement vs off-resonance (global and inside the ring)
    e2_on_max = float(np.max(d["dft_on_E2"]))
    e2_off_max = float(np.max(d["dft_off_E2"]))
    assert e2_on_max > 2.8 * e2_off_max
    ring_band = np.abs(d["dft_y"]) <= 5.15e-6
    e2_on_ring = float(np.max(d["dft_on_E2"][:, ring_band]))
    e2_off_ring = float(np.max(d["dft_off_E2"][:, ring_band]))
    assert e2_on_ring > 19.0 * e2_off_ring


def test_kottke_vs_staircase_and_grid_convergence():
    """
    Verify that across the resolved ladder (`L1=(10,4) -> L2=(14,6) -> L3=(20,8)`),
    Kottke subpixel smoothing converges monotonically toward `kottke_reference`
    (`R=(40,16)` in `--full`, or `L3=(20,8)` in `--quick`) in both resonance
    wavelength shift [pm] and loaded quality factor Q_L, outperforms staircase
    at every resolution level, and that the under-resolved demonstration pair
    (`1.90` cells across the gap) exhibits severe staircase failure.
    """
    results = load_study_results(DATA_DIR)
    ref_res = find_ring_resonances(
        results["kottke_reference"]["wavelengths"],
        results["kottke_reference"]["S_through_in"],
        results["kottke_reference"]["S_drop_in"],
    )

    k_l1 = find_ring_resonances(
        results["kottke_L1"]["wavelengths"],
        results["kottke_L1"]["S_through_in"],
        results["kottke_L1"]["S_drop_in"],
    )
    k_l2 = find_ring_resonances(
        results["kottke_L2"]["wavelengths"],
        results["kottke_L2"]["S_through_in"],
        results["kottke_L2"]["S_drop_in"],
    )
    s_l1 = find_ring_resonances(
        results["staircase_L1"]["wavelengths"],
        results["staircase_L1"]["S_through_in"],
        results["staircase_L1"]["S_drop_in"],
    )
    s_l2 = find_ring_resonances(
        results["staircase_L2"]["wavelengths"],
        results["staircase_L2"]["S_through_in"],
        results["staircase_L2"]["S_drop_in"],
    )

    m_k_l1 = compute_resonance_shift_metrics(k_l1, ref_res)
    m_k_l2 = compute_resonance_shift_metrics(k_l2, ref_res)
    m_s_l1 = compute_resonance_shift_metrics(s_l1, ref_res)
    m_s_l2 = compute_resonance_shift_metrics(s_l2, ref_res)

    # Monotonic convergence of Kottke resonance shift [pm], FSR error [pm], and Q_L error [%] from L1 -> L2
    assert len(k_l1["resonances"]) == 4 and len(k_l2["resonances"]) == 4
    assert m_k_l1["mean_abs_shift_pm"] > 1.8 * m_k_l2["mean_abs_shift_pm"]
    assert m_k_l1["fsr_err_pm"] > m_k_l2["fsr_err_pm"]
    assert m_k_l1["q_rel_err_pct"] > m_k_l2["q_rel_err_pct"]
    assert m_k_l2["q_rel_err_pct"] < 10.0

    # At both L1 and L2, Kottke has smaller resonance shift and smaller Q_L error than staircase
    assert m_k_l1["mean_abs_shift_pm"] < m_s_l1["mean_abs_shift_pm"]
    assert m_k_l2["mean_abs_shift_pm"] < m_s_l2["mean_abs_shift_pm"]
    assert m_k_l1["q_rel_err_pct"] < m_s_l1["q_rel_err_pct"]
    assert m_k_l2["q_rel_err_pct"] < m_s_l2["q_rel_err_pct"]

    if "kottke_L3" in results and "staircase_L3" in results:
        assert int(results["kottke_reference"]["cpw"]) == 40
        assert float(results["kottke_reference"]["cells_across_feature"]) >= 16.0 - 1e-3
        k_l3 = find_ring_resonances(
            results["kottke_L3"]["wavelengths"],
            results["kottke_L3"]["S_through_in"],
            results["kottke_L3"]["S_drop_in"],
        )
        s_l3 = find_ring_resonances(
            results["staircase_L3"]["wavelengths"],
            results["staircase_L3"]["S_through_in"],
            results["staircase_L3"]["S_drop_in"],
        )
        m_k_l3 = compute_resonance_shift_metrics(k_l3, ref_res)
        m_s_l3 = compute_resonance_shift_metrics(s_l3, ref_res)

        assert len(k_l3["resonances"]) == 4 and len(s_l3["resonances"]) == 4
        assert m_k_l2["mean_abs_shift_pm"] > 1.8 * m_k_l3["mean_abs_shift_pm"]
        assert m_k_l2["fsr_err_pm"] > m_k_l3["fsr_err_pm"]
        assert m_k_l2["q_rel_err_pct"] > m_k_l3["q_rel_err_pct"]
        assert m_k_l3["q_rel_err_pct"] < 1.0
        assert m_k_l3["mean_abs_shift_pm"] < m_s_l3["mean_abs_shift_pm"]
        assert m_k_l3["q_rel_err_pct"] < m_s_l3["q_rel_err_pct"]

    # Under-resolved demonstration pair (10 cpw, no mesh overrides -> 1.90 cells across 100 nm gap)
    u_s = find_ring_resonances(
        results["underresolved_staircase"]["wavelengths"],
        results["underresolved_staircase"]["S_through_in"],
        results["underresolved_staircase"]["S_drop_in"],
    )
    u_k = find_ring_resonances(
        results["underresolved_kottke"]["wavelengths"],
        results["underresolved_kottke"]["S_through_in"],
        results["underresolved_kottke"]["S_drop_in"],
    )
    assert len(u_s["resonances"]) == 3
    assert len(u_k["resonances"]) == 4
    m_u_s = compute_resonance_shift_metrics(u_s, ref_res)
    assert m_u_s["mean_abs_shift_pm"] > m_k_l1["mean_abs_shift_pm"]


def test_coupling_gap_mutation_and_figures_exist():
    """
    Physical mutation check: increasing the bus-ring edge-to-edge coupling gap
    from g = 60 nm -> 100 nm -> 150 nm (each resolved with N = 8 cells across
    its own gap at L3 in `--full` mode, or N = 6 at L2 in `--quick` mode)
    monotonically reduces evanescent power coupling kappa^2, narrows the
    resonance FWHM, and increases the loaded quality factor Q_L.
    Also verifies all 14 figures exist and are non-empty.
    """
    results = load_study_results(DATA_DIR)
    mid_key = "kottke_L3" if "kottke_L3" in results else "kottke_L2"
    min_cells = 8.0 if "kottke_L3" in results else 6.0
    qs, fwhms, kappas = [], [], []
    for k_name in ("kottke_gap_60nm", mid_key, "kottke_gap_150nm"):
        d = results[k_name]
        assert float(d["cells_across_feature"]) >= min_cells - 1e-3
        rk = find_ring_resonances(d["wavelengths"], d["S_through_in"], d["S_drop_in"])
        r_mid = min(rk["resonances"], key=lambda r: abs(r["wl0_nm"] - 1560.0))
        qs.append(r_mid["Q_loaded"])
        fwhms.append(r_mid["fwhm_nm"])
        kappas.append(r_mid["kappa2_cmt"])

    # Wider gap -> weaker coupling -> narrower linewidth -> higher loaded Q
    assert kappas[0] > kappas[1] > kappas[2]
    assert fwhms[0] > fwhms[1] > fwhms[2]
    assert qs[0] < qs[1] < qs[2]

    expected_figs = [
        "01_layout.png",
        "02_eim_stack.png",
        "03_coupler_geometry_zoom.png",
        "04_mesh_permittivity_zoom.png",
        "05_port_modes.png",
        "06_waveguide_dispersion_ng.png",
        "07_fields_on_vs_off_resonance.png",
        "08_field_movie.gif",
        "09_sparameters_broadband.png",
        "10_resonance_zoom_cmt_fit.png",
        "11_fsr_and_q_analysis.png",
        "12_coupling_gap_sweep.png",
        "13_kottke_convergence.png",
        "14_time_traces_ringdown.png",
    ]
    for fname in expected_figs:
        fpath = FIG_DIR / fname
        assert fpath.exists() and fpath.stat().st_size > 1000, f"Missing or empty figure: {fname}"
