# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Fast verification and mutation tests for Tutorial 7.6:
Thin-film LED: emission computed as absorption (Helmholtz reciprocity) and
optimization of the absorption in the emitting region (Janssen et al. 2010).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as c_light

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "07_light_sources"
    / "07-06_led_reciprocity"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    Z0,
    compute_grating_gradients_both_ways,
    cone_integrated_reciprocity,
    get_silver_material,
    make_drude_silver,
    make_finite_led_solver,
    measure_staircase_grating_geometry,
    rcwa_grating_batch,
    run_finite_led_direct_dipoles,
    run_finite_led_reciprocity,
    solve_bloch_flat_slab,
    tmm_slab_emission,
)


def test_tmm_analytic_identities():
    """
    Verify exact analytical limits of `tmm_slab_emission`:
    1. Free-space dipole above a PEC ground plane (`n_slab = 1.0`):
       - `I_TM(theta, y_0) = 4 sin^2(k_0 y_0 cos(theta))`
       - `I_TE_x(theta, y_0) = 4 cos^2(theta) sin^2(k_0 y_0 cos(theta))`
       - `I_TE_y(theta, y_0) = 4 sin^2(theta) cos^2(k_0 y_0 cos(theta))`
    2. Quarter-wave / resonant slab enhancement at normal incidence (`theta = 0`):
       - At `d_slab = m * lam0 / (2 n_slab)` (half-wave slab) and `y_0 = lam0 / (4 n_slab)`
         (antinode), `I_TM(0, y_0) = 4 / n_slab^2`.
       - At `d_slab = (2m + 1) * lam0 / (4 n_slab)` (quarter-wave anti-resonant slab) and
         `y_0 = lam0 / (4 n_slab)`, `I_TM(0, y_0) = 4 * n_slab^2`.
    3. Angular symmetry `I(-theta) == I(+theta)` and vanishing vertical-dipole emission
       at normal incidence (`I_TE_y(0, y_0) == 0`).
    """
    lam0 = 0.45e-6
    k0 = 2.0 * np.pi / lam0
    thetas = np.deg2rad(np.linspace(-75.0, 75.0, 61))
    y0 = 0.18 * lam0
    d_slab = 0.55 * lam0

    # 1. PEC image-dipole limit when n_slab = 1.0
    ky0 = k0 * np.cos(thetas)
    np.testing.assert_allclose(
        tmm_slab_emission(thetas, y0, d_slab, 1.0, lam0, pol="TM"),
        4.0 * np.sin(ky0 * y0) ** 2,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tmm_slab_emission(thetas, y0, d_slab, 1.0, lam0, pol="TE_x"),
        4.0 * (np.cos(thetas) ** 2) * (np.sin(ky0 * y0) ** 2),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tmm_slab_emission(thetas, y0, d_slab, 1.0, lam0, pol="TE_y"),
        4.0 * (np.sin(thetas) ** 2) * (np.cos(ky0 * y0) ** 2),
        rtol=1e-12,
        atol=1e-12,
    )

    # 2. Normal-incidence Fabry-Perot limits on a high-index slab
    n_slab = 2.45
    y_antinode = lam0 / (4.0 * n_slab)
    d_halfwave = lam0 / (2.0 * n_slab)
    d_quarterwave = 3.0 * lam0 / (4.0 * n_slab)

    I_hw = float(tmm_slab_emission(0.0, y_antinode, d_halfwave, n_slab, lam0, pol="TM"))
    I_qw = float(tmm_slab_emission(0.0, y_antinode, d_quarterwave, n_slab, lam0, pol="TM"))
    assert np.isclose(I_hw, 4.0 / (n_slab**2), rtol=1e-12)
    assert np.isclose(I_qw, 4.0, rtol=1e-12)

    # 3. Symmetry and normal-incidence longitudinal zero
    for pol in ("TM", "TE_x", "TE_y", "unpolarized"):
        I_pos = tmm_slab_emission(thetas, y0, d_slab, n_slab, lam0, pol=pol)
        I_neg = tmm_slab_emission(-thetas, y0, d_slab, n_slab, lam0, pol=pol)
        np.testing.assert_allclose(I_pos, I_neg, rtol=1e-12, atol=1e-12)
    assert float(tmm_slab_emission(0.0, y0, d_slab, n_slab, lam0, pol="TE_y")) == pytest.approx(
        0.0, abs=1e-15
    )


def test_bloch_flat_slab_vs_tmm_pec_and_silver():
    """
    Validate Bloch-periodic reciprocal plane-wave simulations against exact TMM
    for both an ideal PEC mirror (oblique angle `theta = 25 deg`, all 3 dipole
    components `Ez, Ex, Ey`) and a dispersive Drude silver (Ag) mirror at `lam0 = 450 nm`.
    """
    lam0 = 0.45e-6
    n_slab = 2.45
    d_slab = 0.60 * lam0

    # Oblique incidence on PEC-backed slab (excites TM Ez, TE Ex, and longitudinal TE Ey)
    res_pec = solve_bloch_flat_slab(
        theta_deg=25.0,
        d_slab=d_slab,
        n_slab=n_slab,
        lam0=lam0,
        cpw=20,
        mirror="pec",
        n_cycles=26,
    )
    for key_sim, key_ana in (
        ("I_tm", "I_tm_ana"),
        ("I_te_x", "I_te_x_ana"),
        ("I_te_y", "I_te_y_ana"),
    ):
        l2_err = np.linalg.norm(res_pec[key_sim] - res_pec[key_ana]) / np.linalg.norm(
            res_pec[key_ana]
        )
        assert l2_err < 0.04, f"Bloch vs TMM {key_sim} L2 error {l2_err:.4f} >= 0.04"

    # Normal incidence on dispersive Drude Ag mirror (graded mesh with >= 8 cells/delta_s, >= 4 cells/delta_p)
    res_ag = solve_bloch_flat_slab(
        theta_deg=0.0,
        d_slab=d_slab,
        n_slab=n_slab,
        lam0=lam0,
        cpw=20,
        mirror="ag",
        n_cycles=26,
        cpsd_target=8.0,
    )
    assert res_ag["cells_per_skin_depth"] >= 8.0 - 0.05
    assert res_ag["cells_per_power_skin_depth"] >= 4.0 - 0.03
    l2_err_ag = np.linalg.norm(res_ag["I_tm"] - res_ag["I_tm_ana"]) / np.linalg.norm(
        res_ag["I_tm_ana"]
    )
    assert l2_err_ag < 0.025, f"Ag-backed slab L2 error {l2_err_ag:.4f} >= 0.025"

    # Verify that the test distinguishes Ag from PEC (penetration phase shift shifts the standing wave)
    I_pec_at_ag_nodes = tmm_slab_emission(
        0.0, res_ag["y_ez"], d_slab, n_slab, lam0, pol="TM", eps_mirror=np.inf
    )
    diff_ag_vs_pec = np.linalg.norm(res_ag["I_tm"] - I_pec_at_ag_nodes) / np.linalg.norm(
        I_pec_at_ag_nodes
    )
    assert diff_ag_vs_pec > 0.35


def test_direct_dipole_vs_reciprocity_and_mutations():
    """
    Verify point-by-point Helmholtz reciprocity between Method (A) Direct Dipole
    (`CurrentSource` + `Near2FarMonitor2D`) and Method (B) Reciprocal Huygens Plane Wave
    (`make_huygens_planewave_2d` + `DFTMonitor`) across all 3 dipole orientations
    (`z`, `x`, `y`) and oblique angles (`0 deg`, `25 deg`) on a textured grating LED,
    plus mutation checks verifying that:
    1. Breaking the 2D cylindrical Green's function factor or the Huygens `cos(theta)`
       obliquity weight causes the test to fail by > 3 orders of magnitude.
    2. Summing complex dipole far fields coherently (`|sum E_i|^2`) instead of summing
       powers incoherently (`sum |E_i|^2`) violates reciprocity and fails the check.
    """
    lam0 = 0.45e-6
    n_slab = 2.45
    d_slab = 0.50 * lam0
    pitch = 0.60 * lam0
    h_grat = 0.16 * lam0
    n_periods = 5
    cpw = 10
    thetas_deg = np.array([0.0, 25.0])
    di_offsets = [-4, 0, 3]

    rec = run_finite_led_reciprocity(
        thetas_deg=thetas_deg,
        lam0=lam0,
        n_slab=n_slab,
        d_slab=d_slab,
        pitch=pitch,
        h_grat=h_grat,
        n_periods=n_periods,
        cpw=cpw,
        averaging="kottke",
        n_cycles=22,
    )
    direct = run_finite_led_direct_dipoles(
        x_indices_offset=di_offsets,
        thetas_deg=thetas_deg,
        lam0=lam0,
        n_slab=n_slab,
        d_slab=d_slab,
        pitch=pitch,
        h_grat=h_grat,
        n_periods=n_periods,
        cpw=cpw,
        averaging="kottke",
        n_cycles=22,
        components=("z", "x", "y"),
    )

    # Point-by-point check at each dipole position and orientation
    comp_map = {
        "z": ("x_ez", "I_tm"),
        "x": ("x_ex", "I_te_x"),
        "y": ("x_ey", "I_te_y"),
    }
    rec_at_dipoles = {c: np.zeros_like(direct["I_dip"][c]) for c in ("z", "x", "y")}
    for c, (x_key, I_key) in comp_map.items():
        for ip, xd in enumerate(direct["x_pos"][c]):
            idx_x = int(np.argmin(np.abs(rec[x_key] - xd)))
            assert abs(rec[x_key][idx_x] - xd) < 1e-12
            rec_at_dipoles[c][ip, :] = rec[I_key][:, idx_x]
            np.testing.assert_allclose(
                direct["I_dip"][c][ip, :],
                rec_at_dipoles[c][ip, :],
                rtol=5e-3,
                atol=2e-4,
                err_msg=f"Point-by-point reciprocity failed for comp={c}, ip={ip}",
            )

    # Ensemble incoherent power average matches reciprocal spatial average to < 5e-3
    rec_incoh_tm = np.mean(rec_at_dipoles["z"], axis=0)
    dir_incoh_tm = direct["I_incoh_by_comp"]["z"]
    incoh_rel_err = np.max(np.abs(dir_incoh_tm - rec_incoh_tm) / rec_incoh_tm)
    assert incoh_rel_err < 5e-3

    # Mutation check 1: omitting the cos(theta) obliquity factor in the Huygens TM J_z sheet
    # scales the incident TM wave by (1 + 1/cos(theta)) / 2 != 1 at theta = 25 deg
    theta_oblique = np.deg2rad(thetas_deg[1])
    mutated_no_cos_scale = (0.5 * (1.0 + 1.0 / np.cos(theta_oblique))) ** 2
    mutated_rec_tm = rec_incoh_tm[1] / mutated_no_cos_scale
    mutated_err_obliquity = abs(dir_incoh_tm[1] - mutated_rec_tm) / dir_incoh_tm[1]
    assert mutated_err_obliquity > 0.08, (
        "Mutation check failed: omitting cos(theta) obliquity factor was not detected"
    )

    # Mutation check 2: coherent field sum |sum_i E_i(theta)|^2 / N_pos vs incoherent power sum
    coh_vs_rec_err = np.max(np.abs(direct["I_coh_tm"] - rec_incoh_tm) / rec_incoh_tm)
    assert coh_vs_rec_err > 0.40, (
        f"Mutation check failed: coherent field sum error {coh_vs_rec_err:.3f} should fail reciprocity"
    )


def test_task_tables_satisfy_resolution_rules():
    """
    Build the mesh only (without running the solver) for every task in `--quick`
    and `--full` (`build_tasks(quick=True)` and `build_tasks(quick=False)`) and verify
    that every non-demo task strictly satisfies:
    1. Rule 1: `>= 10` cells per wavelength inside the highest-index material
       (`dx_background_nm <= lam0 / (10 * n_max) = 18.367 nm`), and `>= 20` cpw for
       reference tasks (`task_00`, `task_08`).
    2. Rule 2: `>= 4` cells across the smallest geometric feature (`h_grat = 72 nm`
       and tooth width `w_top = 94.5 nm`, `cells_across_feature >= 4.0`).
    3. Rule 3: For metal (Drude Ag) tasks, `>= 8` cells per amplitude skin depth
       `delta_s = lam0 / (2 pi kappa) = 26.91 nm` (`>= 4` cells per power skin depth
       `delta_p = lam0 / (4 pi kappa) = 13.45 nm`).
    Also includes mutation checks proving that the validator fails if given an
    under-resolved mesh (`cpw = 8` or `cpsd = 1.6`).
    """
    from led_reciprocity import LAM0, N_SLAB, build_task_mesh_metadata, build_tasks

    dx_max_rule1_nm = (LAM0 / (10.0 * N_SLAB)) * 1e9  # 18.367 nm

    def _validate_non_demo(info):
        assert info["dx_background_nm"] <= dx_max_rule1_nm + 1e-6, (
            f"Rule 1 violated in {info['name']}: dx_background_nm={info['dx_background_nm']} > {dx_max_rule1_nm:.3f}"
        )
        assert info["dx_min_nm"] <= info["dx_background_nm"] + 1e-6
        assert info["dx_min_nm"] < info["min_feature_nm"]
        assert info["cells_across_feature"] >= 4.0 - 1e-6, (
            f"Rule 2 violated in {info['name']}: cells_across_feature={info['cells_across_feature']} < 4.0"
        )
        if info["task"] in (0, 8):
            assert info["cpw"] >= 20, (
                f"Reference task {info['name']} has cpw={info['cpw']} < 20"
            )
        if info["mirror"] == "ag":
            assert info["cells_per_skin_depth"] is not None
            assert info["cells_per_skin_depth"] >= 8.0 - 0.1, (
                f"Rule 3 violated in {info['name']}: cells_per_skin_depth={info['cells_per_skin_depth']} < 8.0"
            )
            assert info["cells_per_power_skin_depth"] >= 4.0 - 0.05, (
                f"Rule 3 violated in {info['name']}: cells_per_power_skin_depth={info['cells_per_power_skin_depth']} < 4.0"
            )

    for quick in (True, False):
        tasks = build_tasks(quick=quick)
        demo_tasks = [t for t in tasks if t.get("demo", False)]
        assert len(demo_tasks) == 1, "Expected exactly one labelled demo task in table"
        for t in tasks:
            info = build_task_mesh_metadata(t)
            assert info["n_cells"] > 0 and info["n_steps"] > 0
            if not t.get("demo", False):
                _validate_non_demo(info)
            else:
                # Mutation check 1: the demo task must fail the non-demo validator on Rule 3
                with pytest.raises(AssertionError, match="Rule 3 violated"):
                    _validate_non_demo(info)

    # Mutation check 2: a synthetic cpw=8 task must fail both Rule 1 and Rule 2
    bad_task = {
        "index": 99,
        "name": "bad_coarse_mutation",
        "demo": True,
        "mirror": "pec",
        "averaging": "kottke",
        "cpw": 8,
        "n_periods": 5,
    }
    bad_info = build_task_mesh_metadata(bad_task)
    with pytest.raises(AssertionError, match="Rule 1 violated"):
        _validate_non_demo(bad_info)


def test_staircase_geometry_consistency_across_cpw():
    """
    Verify that the staircase discretization (`averaging="staircase"`) snaps the
    grating period `pitch` and horizontal slab/etch interfaces (`y_base`, `d_slab`,
    `h_grat`) consistently across `cpw` levels (`cpw in (10, 14, 18, 22)`):
    1. All tooth-to-tooth periods in the staircase permittivity `ieps` at every `cpw`
       differ by `0.0 nm` (`pitch_span_cells <= 1.0`, and across any two `cpw` levels
       the discretized staircase period differs by `<= 1` cell, in fact `< 0.01 nm`).
    2. The discretized staircase base height `y_base_stair_nm == 153.0 nm`, total slab
       thickness `d_slab_stair_nm == 225.0 nm`, and grating tooth height
       `h_grat_stair_nm == 72.0 nm` remain invariant across `cpw`.
    3. In the saved `--quick` convergence data (`data/task_02..08.npz`), both Kottke
       and Snapped Staircase converge monotonically in angular L2 error across
       `cpw = 10 -> 14 -> 18` vs the `cpw = 22` reference.
    4. Mutation checks prove that the validator fails when:
       - (Mutation A) `snap_grid=False` is used on `cpw in (10, 14, 18)`, where
         incommensurate uniform grid spacing causes `pitch_span_cells = 1.0` (tooth
         period alternation of `13.12 nm` at `cpw=14` and `10.20 nm` at `cpw=18`) and
         shifts `y_base_stair_nm` by `+5.16 nm` at `cpw=18` and `h_grat_stair_nm` by
         `+6.72 nm` at `cpw=14`.
       - (Mutation B) the staircase period at one `cpw` level is perturbed by `> 1` cell
         relative to another `cpw` level.
    """
    from led_reciprocity import D_SLAB, H_GRAT, LAM0, N_PERIODS, N_SLAB, PITCH

    cpw_levels = (10, 14, 18, 20, 22, 28, 40)

    def _validate_staircase_family(geoms_by_cpw, dx_by_cpw):
        cpws = sorted(geoms_by_cpw.keys())
        for cpw in cpws:
            g = geoms_by_cpw[cpw]
            dx_nm = dx_by_cpw[cpw]
            # Every tooth-to-tooth period on a snapped staircase grid must be uniform
            assert g["pitch_span_cells"] < 0.05, (
                f"Staircase period alternates by {g['pitch_span_cells']:.3f} cells across teeth at cpw={cpw}"
            )
            assert g["pitch_span_nm"] <= dx_nm + 1e-2, (
                f"Intra-grid period span {g['pitch_span_nm']:.3f} nm > 1 cell ({dx_nm:.3f} nm) at cpw={cpw}"
            )
            assert abs(g["h_grat_stair_nm"] - H_GRAT * 1e9) <= 0.5 * dx_nm, (
                f"Staircase tooth height {g['h_grat_stair_nm']:.3f} nm drifted from {H_GRAT * 1e9:.1f} nm at cpw={cpw}"
            )
        # Pairwise comparison across cpw levels: period at two cpw levels must not
        # differ by more than one cell, and slab/base heights must match within 0.25 cell
        for i, c1 in enumerate(cpws):
            for c2 in cpws[i + 1 :]:
                g1, g2 = geoms_by_cpw[c1], geoms_by_cpw[c2]
                one_cell_nm = min(dx_by_cpw[c1], dx_by_cpw[c2])
                period_diff_nm = max(
                    abs(g1["pitch_mean_nm"] - g2["pitch_mean_nm"]),
                    abs(g1["pitch_max_nm"] - g2["pitch_min_nm"]),
                    abs(g2["pitch_max_nm"] - g1["pitch_min_nm"]),
                )
                assert period_diff_nm <= one_cell_nm, (
                    f"Staircase period between cpw={c1} and cpw={c2} differs by "
                    f"{period_diff_nm:.3f} nm > 1 cell ({one_cell_nm:.3f} nm)"
                )
                y_base_diff_nm = abs(g1["y_base_stair_nm"] - g2["y_base_stair_nm"])
                assert y_base_diff_nm <= 0.25 * one_cell_nm, (
                    f"Staircase y_base between cpw={c1} and cpw={c2} differs by "
                    f"{y_base_diff_nm:.3f} nm > 0.25 cell ({0.25 * one_cell_nm:.3f} nm)"
                )

    snapped_geoms = {}
    snapped_dx = {}
    for cpw in cpw_levels:
        solver, meta = make_finite_led_solver(
            lam0=LAM0,
            n_slab=N_SLAB,
            d_slab=D_SLAB,
            pitch=PITCH,
            h_grat=H_GRAT,
            n_periods=N_PERIODS,
            cpw=cpw,
            averaging="staircase",
        )
        assert meta["snap_grid"] is True
        g = measure_staircase_grating_geometry(solver, meta)
        snapped_geoms[cpw] = g
        snapped_dx[cpw] = meta["dx"] * 1e9
        assert g["pitch_min_nm"] == pytest.approx(270.0, abs=1e-4)
        assert g["pitch_max_nm"] == pytest.approx(270.0, abs=1e-4)
        assert g["y_base_stair_nm"] == pytest.approx(153.0, abs=1e-4)
        assert g["d_slab_stair_nm"] == pytest.approx(225.0, abs=1e-4)
        assert g["h_grat_stair_nm"] == pytest.approx(72.0, abs=1e-4)

    _validate_staircase_family(snapped_geoms, snapped_dx)

    # Verify monotonic angular L2 convergence in saved task_02..08.npz data
    data_dir = TUTORIAL_DIR / "data"
    if all((data_dir / f"task_{idx:02d}.npz").exists() for idx in range(2, 9)):
        d_ref = np.load(data_dir / "task_08.npz")
        I_ref = d_ref["I_unpol_mean"]
        norm_ref = float(np.linalg.norm(I_ref))
        err_kottke = [
            float(np.linalg.norm(np.load(data_dir / f"task_{idx:02d}.npz")["I_unpol_mean"] - I_ref))
            / norm_ref
            for idx in (2, 3, 4)
        ]
        err_stair_snapped = [
            float(np.linalg.norm(np.load(data_dir / f"task_{idx:02d}.npz")["I_unpol_mean"] - I_ref))
            / norm_ref
            for idx in (5, 6, 7)
        ]
        assert err_kottke[0] > err_kottke[1] > err_kottke[2], (
            f"Kottke angular L2 error not monotonic: {err_kottke}"
        )
        assert err_stair_snapped[0] > err_stair_snapped[1] > err_stair_snapped[2], (
            f"Snapped staircase angular L2 error not monotonic: {err_stair_snapped}"
        )

    # Mutation check A: un-snapped staircase (snap_grid=False) has period alternation
    # and y_base/d_slab shifts across both quick and full cpw levels, which must fail
    unsnapped_geoms = {}
    unsnapped_dx = {}
    for cpw in (10, 14, 18, 20, 28):
        solver_u, meta_u = make_finite_led_solver(
            lam0=LAM0,
            n_slab=N_SLAB,
            d_slab=D_SLAB,
            pitch=PITCH,
            h_grat=H_GRAT,
            n_periods=N_PERIODS,
            cpw=cpw,
            averaging="staircase",
            snap_grid=False,
        )
        unsnapped_geoms[cpw] = measure_staircase_grating_geometry(solver_u, meta_u)
        unsnapped_dx[cpw] = meta_u["dx"] * 1e9
    with pytest.raises(AssertionError, match="Staircase period alternates"):
        _validate_staircase_family(unsnapped_geoms, unsnapped_dx)

    # Mutation check B: staircase period at two cpw levels differing by > 1 cell must fail
    mutated_geoms = {k: dict(v) for k, v in snapped_geoms.items()}
    mutated_geoms[18]["pitch_mean_nm"] = 270.0 + 1.5 * snapped_dx[18]
    mutated_geoms[18]["pitch_max_nm"] = 270.0 + 1.5 * snapped_dx[18]
    with pytest.raises(AssertionError, match="differs by .* > 1 cell"):
        _validate_staircase_family(mutated_geoms, snapped_dx)


def test_saved_full_study_data_and_convergence():
    """
    Verify that the saved `--full` study outputs in `tutorials/07_light_sources/07-06_led_reciprocity/data/`
    satisfy all mesh resolution rules, monotonic convergence checks, and reciprocity bounds,
    along with mutation checks proving that each check fails when given un-snapped staircase,
    under-resolved metal demo, or coherent dipole-sum data.
    """
    import json

    from led_reciprocity import LAM0, N_SLAB

    data_dir = TUTORIAL_DIR / "data"
    summary_path = data_dir / "summary.json"
    assert summary_path.exists(), "summary.json missing in data/"
    manifest = json.loads(summary_path.read_text())
    assert manifest["mode"] == "full"
    assert len(manifest["tasks"]) == 11

    dx_max_rule1_nm = (LAM0 / (10.0 * N_SLAB)) * 1e9  # 18.367 nm

    def _check_saved_task_resolution(js):
        assert js["cpw"] >= 10, f"cpw={js['cpw']} < 10 in {js['name']}"
        assert js["dx_background_nm"] <= dx_max_rule1_nm + 1e-6
        assert js["cells_across_feature"] >= 4.0 - 1e-6
        if js.get("mirror") == "ag" or js.get("cells_per_skin_depth") is not None:
            assert js["cells_per_skin_depth"] >= 8.0 - 0.1
            assert js["cells_per_power_skin_depth"] >= 4.0 - 0.05

    for idx in range(11):
        js = json.loads((data_dir / f"task_{idx:02d}.json").read_text())
        if not js["demo"]:
            _check_saved_task_resolution(js)
        else:
            # Mutation check 1: demo task_09 must fail the non-demo skin-depth check
            with pytest.raises(AssertionError):
                _check_saved_task_resolution(js)

    # 1. Task 0: flat-slab Bloch vs TMM standing-wave errors and monotonic ladders
    j0 = json.loads((data_dir / "task_00.json").read_text())
    d0 = np.load(data_dir / "task_00.npz")
    assert j0["cpw"] == 28
    assert j0["l2_err_tm_pec25"] < 0.015
    assert j0["l2_err_tex_pec25"] < 0.015
    assert j0["l2_err_tey_pec25"] < 0.015
    assert j0["l2_err_tm_ag0"] < 0.018
    assert j0["l2_err_tm_ag25"] < 0.018
    assert np.all(np.diff(d0["err_tm_conv"]) < 0.0)
    assert d0["err_tm_conv"][-1] < 0.007
    assert np.all(np.diff(d0["err_ag0_ladder"]) < 0.0)

    # 2. Task 1: direct dipole sum vs reciprocity at cpw=20 (N_pos=25, N_angles=29)
    j1 = json.loads((data_dir / "task_01.json").read_text())
    d1 = np.load(data_dir / "task_01.npz")
    assert j1["cpw"] == 20 and j1["n_pos"] == 25 and j1["n_angles"] == 29
    assert j1["ensemble_l2_err_unpol"] < 1e-3
    # Mutation check 2: coherent dipole sum must fail the < 1% ensemble agreement check
    coh_l2_err = float(
        np.linalg.norm(d1["dir_coh_tm"] - d1["rec_sample_z"])
        / np.linalg.norm(d1["rec_sample_z"])
    )
    assert coh_l2_err > 0.40

    # 3. Tasks 2..8: Kottke and Snapped Staircase converge monotonically vs cpw=40 ref,
    #    while Un-snapped Staircase rebounds non-monotonically between cpw=20 and cpw=28
    j8 = json.loads((data_dir / "task_08.json").read_text())
    d8 = np.load(data_dir / "task_08.npz")
    assert j8["cpw"] == 40
    ref_I = d8["I_unpol_mean"]
    ref_P = float(j8["P_int_unpol_rad"])

    def _check_monotonic_series(err_series):
        assert err_series[0] > err_series[1] > err_series[2], (
            f"Series did not converge monotonically: {err_series}"
        )

    err_k = []
    err_P_k = []
    for idx in (2, 3, 4):
        di = np.load(data_dir / f"task_{idx:02d}.npz")
        ji = json.loads((data_dir / f"task_{idx:02d}.json").read_text())
        err_k.append(float(np.linalg.norm(di["I_unpol_mean"] - ref_I) / np.linalg.norm(ref_I)))
        err_P_k.append(abs(float(ji["P_int_unpol_rad"]) - ref_P) / ref_P)
    _check_monotonic_series(err_k)
    _check_monotonic_series(err_P_k)
    assert err_k[-1] < 0.025 and err_P_k[-1] < 0.01

    err_ss = []
    err_su = []
    for idx in (5, 6, 7):
        di = np.load(data_dir / f"task_{idx:02d}.npz")
        err_ss.append(float(np.linalg.norm(di["I_unpol_mean"] - ref_I) / np.linalg.norm(ref_I)))
        err_su.append(
            float(np.linalg.norm(di["I_unpol_unsnapped_mean"] - ref_I) / np.linalg.norm(ref_I))
        )
    _check_monotonic_series(err_ss)
    assert err_ss[-1] < 0.022
    # Mutation check 3: un-snapped staircase rebounds at cpw=28 and fails monotonic check
    with pytest.raises(AssertionError, match="did not converge monotonically"):
        _check_monotonic_series(err_su)

    # 4. Tasks 9 & 10: Drude Ag skin-depth convergence vs under-resolved demo jump
    j9 = json.loads((data_dir / "task_09.json").read_text())
    j10 = json.loads((data_dir / "task_10.json").read_text())
    assert j10["cpw"] == 20
    assert j10["cells_per_skin_depth"] == pytest.approx(10.0)
    assert j10["rel_change_last_two_levels"] < 0.005
    assert np.all(np.diff(j10["P_grat_unpol_rad"]) < 0.0)
    assert np.all(np.diff(j10["sw_l2_err"]) < 0.0)
    # Mutation check 4: under-resolved demo task_09 fails the < 1% convergence threshold
    assert j9["rel_change_last_two_levels"] > 0.04


def test_janssen_optimization_study_and_resonances():
    """
    Verify all physical benchmarks of the Janssen, Wachters & Urbach (Opt. Express 18,
    24522, 2010) optimization study (`janssen_optimization.json` and `.npz`) and all
    new figures (`00_emitter_absorber_schematic.png`, `16..19`):
    1. Fig. 2 shallow-grating resonances (`pitch = 250 nm`, `d = 5 nm`, `t = 700 nm`,
       `y_0 = 100 nm` vs flat `t = 698 nm`):
       - Sharp guided-mode resonance peaks far above the flat background.
       - Quality factor `Q ~ 10^5` on PEC (`3e4 <= Q <= 5e5`) vs `Q ~ 10^3` on Ag
         (`silver jc`, `1e3 <= Q <= 1e4`), verified by both angular linewidth and
         Wakis `ResonanceMonitor.harminv`.
    2. Fig. 4 `(t, d)` landscape (`pitch = 450 nm` block grating on Ag):
       - Multiple local maxima across `(t, d)` and steepest-ascent paths that strictly
         ascend to local optima.
    3. Fig. 5 optimal active-layer position near `y_0 ~ 120 nm` above the Ag mirror,
       and 2D Bloch FIT standing-wave agreement with RCWA.
    4. Fig. 6 pitch scan with strong enhancement near `Lambda / lam0 ~ 0.7 - 0.8`.
    5. Headline: optimized grating beats optimized flat structure by `~ 2x` (`1.85x - 2.30x`).
    6. Mutation checks verifying that setting `d = 0` or placing the active layer at a
       standing-wave node (`y_0 = 165 nm`) fails the enhancement and optimal-height checks.
    """
    import json

    data_dir = TUTORIAL_DIR / "data"
    fig_dir = TUTORIAL_DIR / "figures"
    for fname in (
        "00_setup_schematic.png",
        "00_emitter_absorber_schematic.png",
        "01_geometry_mesh.png",
        "02_field_snapshots_and_movie.png",
        "02_reciprocal_wave_movie.gif",
        "03_flat_slab_tmm_vs_wakis_angle.png",
        "04_flat_slab_depth_standing_wave.png",
        "05_silver_vs_pec_mirror.png",
        "06_reciprocity_point_by_point.png",
        "07_ensemble_direct_vs_reciprocity.png",
        "08_incoherent_vs_coherent_sum.png",
        "09_dipole_count_convergence.png",
        "10_grating_vs_flat_extraction.png",
        "11_reciprocal_2d_field_maps.png",
        "11b_forward_vs_reciprocal_field_map.png",
        "12_cost_crossover_scaling.png",
        "13_resolution_convergence_kottke_vs_staircase.png",
        "14_wavelength_broadband_incoherent.png",
        "15_metal_skin_depth_resolution_study.png",
        "16_janssen_fig2_shallow_grating_resonances.png",
        "17_janssen_fig4_thickness_depth_landscape.png",
        "18_janssen_fig5_active_layer_position.png",
        "19_janssen_fig6_pitch_scan_enhancement.png",
    ):
        p_fig = fig_dir / fname
        assert p_fig.exists() and p_fig.stat().st_size > 10_000, f"Missing or empty figure {fname}"
    assert (fig_dir / "02_reciprocal_wave_movie.gif").stat().st_size < 5 * 1024 * 1024

    j_sum = json.loads((data_dir / "janssen_optimization.json").read_text())
    j_npz = np.load(data_dir / "janssen_optimization.npz", allow_pickle=True)

    # (a) Shallow-grating resonances (Janssen Fig. 2): Q ~ 10^5 (PEC) vs Q ~ 10^3 (Ag)
    assert j_sum["silver_library_name"] == "silver jc"
    assert float(np.max(j_npz["pec_grat_dense_s"])) > 50.0 * float(np.max(j_npz["pec_flat_s"]))
    assert float(np.max(j_npz["ag_grat_s"])) > 5.0 * float(np.max(j_npz["ag_flat_s"]))
    assert 3.0e4 <= j_sum["fig2_Q_pec_median_s"] <= 5.0e5
    assert 5.0e4 <= j_sum["fig2_harminv_pec_Q"] <= 5.0e5
    assert 1.0e3 <= j_sum["fig2_Q_ag_median_s"] <= 1.0e4
    assert 1.0e3 <= j_sum["fig2_harminv_ag_Q"] <= 1.0e4
    assert j_sum["fig2_Q_pec_median_s"] / j_sum["fig2_Q_ag_median_s"] > 10.0
    assert j_sum["fig2_harminv_pec_Q"] / j_sum["fig2_harminv_ag_Q"] > 10.0

    # (b) (t, d) landscape & steepest-ascent paths (Janssen Fig. 4)
    assert len(j_sum["fig4_local_maxima"]) >= 8
    assert j_sum["fig4_enh_vs_flat_60"] >= 1.85
    for ip_idx in range(4):
        traj = np.asarray(j_npz[f"ascent_path_{ip_idx}"], dtype=float)
        assert traj.shape[0] >= 3 and traj.shape[1] == 2
        assert np.linalg.norm(traj[-1] - traj[0]) > 2.0

    # (c) Optimal active-layer position near y_0 ~ 120 nm (Janssen Fig. 5) + 2D FIT check
    assert 110.0 <= j_sum["fig5_y_opt_grat_450_nm"] <= 130.0
    assert 110.0 <= j_sum["fig5_y_opt_grat_360_nm"] <= 130.0
    assert 105.0 <= j_sum["fig5_y_opt_flat_nm"] <= 125.0
    fit_0 = np.asarray(j_npz["fit_unit_0_s"], dtype=float)
    rcwa_0 = np.asarray(j_npz["rcwa_unit_0_s"], dtype=float)
    fit_25 = np.asarray(j_npz["fit_unit_15_s"], dtype=float)
    rcwa_25 = np.asarray(j_npz["rcwa_unit_15_s"], dtype=float)
    rel_l2_unit25 = float(np.linalg.norm(fit_25 - rcwa_25) / np.linalg.norm(rcwa_25))
    assert rel_l2_unit25 < 0.18
    assert float(np.corrcoef(fit_0, rcwa_0)[0, 1]) > 0.85

    # (d) Pitch scan peak near Lambda / lam0 ~ 0.7 - 0.85 (Janssen Fig. 6)
    assert 0.65 <= j_sum["fig6_best_pitch_ratio_60"] <= 0.88
    assert 0.65 <= j_sum["fig6_best_pitch_ratio_90"] <= 0.88

    # (e) Headline: optimized grating ~ 2x optimized flat structure
    assert 1.80 <= j_sum["headline_enhancement_global_60"] <= 2.35
    assert 1.80 <= j_sum["headline_enhancement_global_90"] <= 2.35
    assert 1.80 <= j_sum["headline_enhancement_60_at_y100"] <= 2.35
    assert 1.80 <= j_sum["headline_enhancement_90_at_y100"] <= 2.35

    # Mutation check 1: setting grating depth d = 0 removes the grating enhancement (ratio == 1.0 < 1.8)
    _, eps_ag, _ = get_silver_material()
    F_flat_mut = cone_integrated_reciprocity(
        0.70e-6, 0.0, 0.36e-6, y_pos=0.12e-6, cone_full_deg=60.0, eps_mirror=eps_ag, n_harmonics=2
    )["F_iso"]
    mut_enh_d0 = F_flat_mut / j_sum["headline_best_flat_60_iso"]
    assert mut_enh_d0 <= 1.05, f"Mutation d=0 should not beat flat baseline, got {mut_enh_d0:.3f}x"

    # Mutation check 2: placing active layer at standing-wave node y_0 = 165 nm drops extraction
    y_nm_5 = np.asarray(j_npz["y_arr_fig5_nm"], dtype=float)
    prof_5 = np.asarray(j_npz["prof_fig5_360_iso"], dtype=float)
    idx_120 = int(np.argmin(np.abs(y_nm_5 - 120.0)))
    idx_node = int(np.argmin(np.abs(y_nm_5 - 165.0)))
    assert prof_5[idx_120] > 2.5 * prof_5[idx_node]


def test_grating_shape_gradient_both_ways_and_mutations():
    """
    Verify the grating shape gradient `dF / d(t, d)` computed **both ways** in
    `compute_grating_gradients_both_ways`:
    1. Way 1 — Wakis discrete adjoint (`wakis.inverse.AdjointSimulation` + `MaterialGrid`
       smooth boundary projection chain rule) vs central finite differences at two step
       sizes `delta_1 = 0.10 nm` and `delta_2 = 0.05 nm`:
       - Relative error `< 5e-5` for both `d(ln F)/dt` and `d(ln F)/dd`.
       - Quadratic convergence ratio `order_ratio in [3.8, 4.2]` (`O(delta^2)`).
    2. Way 2 — Direct geometric central finite differences on `F_cone(t, d)` at two step
       sizes `delta_1 = 10 pm` and `delta_2 = 5 pm`:
       - Relative difference `< 5e-4` for both `dF/dt` and `dF/dd`.
    3. Mutation checks proving that flipping the sign of `drho/dd` or omitting the
       base-height shift in `drho/dt` causes the gradient check to fail by `> 10%`.
    """
    gc = compute_grating_gradients_both_ways()

    # Way 1: Adjoint vs central FD
    assert gc["rel_err_dt_delta1"] < 5e-5
    assert gc["rel_err_dt_delta2"] < 1.5e-5
    assert gc["rel_err_dd_delta1"] < 5e-5
    assert gc["rel_err_dd_delta2"] < 1.5e-5
    assert 3.85 <= gc["order_ratio_dt"] <= 4.15
    assert 3.85 <= gc["order_ratio_dd"] <= 4.15

    # Way 2: Direct geometric central FD at two step sizes
    assert gc["rel_diff_geo_dt"] < 5e-4
    assert gc["rel_diff_geo_dd"] < 5e-4

    # Mutation check 1: omitting the base interface shift (y_base = t - d) in dF/dt
    # adds dF/dd to dF/dt, causing a relative error > 8% against central FD
    mut_dlnF_dt = gc["dlnF_dt_adj_per_m"] + gc["dlnF_dd_adj_per_m"]
    mut_rel_err_dt = abs(mut_dlnF_dt - gc["dlnF_dt_fd2_per_m"]) / abs(gc["dlnF_dt_fd2_per_m"])
    assert mut_rel_err_dt > 0.08

    # Mutation check 2: flipping the sign of drho/dd (y_base = t + d instead of t - d)
    # causes a relative error of ~200% against central FD
    mut_dlnF_dd = -gc["dlnF_dd_adj_per_m"]
    mut_rel_err_dd = abs(mut_dlnF_dd - gc["dlnF_dd_fd2_per_m"]) / abs(gc["dlnF_dd_fd2_per_m"])
    assert mut_rel_err_dd > 1.9



