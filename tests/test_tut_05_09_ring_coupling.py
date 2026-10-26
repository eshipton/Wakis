# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics and mutation validation tests for Tutorial 5.9:
Silicon Waveguide-to-Ring Directional Point Coupler (2D EIM).
"""

from pathlib import Path
import sys

import numpy as np
import pytest

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "05_guided_waves"
    / "05-09_ring_coupling"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    bent_waveguide_radial_mode,
    build_ring_coupler_graded_axes,
    build_ring_coupler_hybrid_unstructured_mesh,
    cmt_coupling_vs_gap,
    fit_exponential_decay,
    ring_resonator_metrics,
    sample_angled_drop_slice,
)
from ring_coupling import (  # noqa: E402
    GAP_NOMINAL,
    H_SI,
    N_SI,
    N_SIO2,
    R_RING,
    W_WG,
    WL_MAX,
    WL_MIN,
    WL_REF,
    build_ring_coupler,
    build_task_mesh_axes,
    coupling_mesh_overrides,
    get_task_table,
    measure_mesh_resolution,
)
from wakis.templates.photonics import (  # noqa: E402
    PhotonicCircuit,
    PolySlab,
    Port,
    n_to_eps,
    straight,
)


def _build_compact_coupler(gap=GAP_NOMINAL, radius=1.5e-6, n_arc=90):
    """
    Build a compact 2D EIM waveguide-to-ring point coupler (`radius = 1.5 um`)
    for fast (< 60 s) physics and mutation tests at full `L1 = (10 cpw, 4 cells/gap)`
    resolution.
    """
    gap = float(gap)
    radius = float(radius)
    y0 = W_WG + gap + radius
    dom_ymax = y0 + 0.35e-6
    y_top = y0 + 1.00e-6
    x_port = radius + 0.25e-6
    x_dom = radius + 0.50e-6

    bus_verts = straight((-x_dom - 1.0e-6, 0.0), (x_dom + 1.0e-6, 0.0), width=W_WG)
    bus_slab = PolySlab(
        bus_verts,
        z_min=-0.5 * H_SI,
        z_max=0.5 * H_SI,
        material=n_to_eps(N_SI),
    )

    theta = np.linspace(-np.pi, 0.0, int(n_arc))
    r_in = radius - 0.5 * W_WG
    r_out = radius + 0.5 * W_WG
    c, s = np.cos(theta), np.sin(theta)
    outer_arc = np.column_stack([r_out * c, y0 + r_out * s])
    inner_arc = np.column_stack([r_in * c[::-1], y0 + r_in * s[::-1]])
    ring_verts = np.vstack(
        [
            [[-r_out, y_top]],
            outer_arc,
            [[r_out, y_top], [r_in, y_top]],
            inner_arc,
            [[-r_in, y_top]],
        ]
    )
    ring_slab = PolySlab(
        ring_verts,
        z_min=-0.5 * H_SI,
        z_max=0.5 * H_SI,
        material=n_to_eps(N_SI),
    )

    y_port = y0 + 0.15e-6
    ports = [
        Port("in", center=(-x_port, 0.0, 0.0), size=(0.0, 1.3e-6, 0.8e-6), direction=+1),
        Port(
            "through",
            center=(+x_port, 0.0, 0.0),
            size=(0.0, 1.3e-6, 0.8e-6),
            direction=+1,
        ),
        Port(
            "drop",
            center=(+radius, y_port, 0.0),
            size=(1.3e-6, 0.0, 0.8e-6),
            direction=+1,
        ),
        Port(
            "add",
            center=(-radius, y_port, 0.0),
            size=(1.3e-6, 0.0, 0.8e-6),
            direction=+1,
        ),
    ]
    domain = (-x_dom, x_dom, -0.55e-6, dom_ymax)
    x_half = min(
        float(np.sqrt(max(4.0 * radius * gap - 4.0 * gap**2, (0.35e-6) ** 2))),
        0.55e-6,
    )
    max_cell = gap / 4.0
    overrides = [
        (
            (-x_half, 0.5 * W_WG - 0.12e-6, -np.inf),
            (x_half, 0.5 * W_WG + gap + 0.12e-6, np.inf),
            (max_cell, max_cell, 1.0),
        )
    ]
    return [bus_slab, ring_slab], ports, domain, overrides


def test_ring_coupling_task_table_resolution_rules():
    """
    Verify that every non-demo task in both the `--quick` and `--full` tables
    satisfies:
    1. `cpw >= 10` cells per wavelength in the slab effective index,
    2. `cells_across_feature >= 4` measured directly from the generated mesh
       across the task's smallest geometric feature (the coupling gap),
    3. `Nz == 1` on the periodic 2D `z` axis even when `mesh_overrides` are active.

    Includes a mutation check verifying that the resolution audit catches tasks
    with `cpw < 10` or `cells_across_feature < 4`.
    """

    def _audit_task(task_cfg):
        x, y, z = build_task_mesh_axes(task_cfg)
        res = measure_mesh_resolution(x, y, gap=float(task_cfg["gap_nm"]) * 1e-9)
        nz = len(z) - 1
        return (
            int(task_cfg["cpw"]) >= 10
            and res["cells_across_feature"] >= 4.0 - 1e-6
            and nz == 1,
            res,
            nz,
        )

    for quick_mode in (True, False):
        table = get_task_table(quick=quick_mode)
        demo_names = [t["name"] for t in table if t.get("is_demo", False)]
        assert demo_names == ["underresolved_kottke", "underresolved_staircase"]

        for task in table:
            ok, res, nz = _audit_task(task)
            assert nz == 1, f"Task {task['name']} subdivided 2D z-axis into Nz={nz}"
            if task.get("is_demo", False):
                assert int(task["cpw"]) == 10
                assert res["cells_across_feature"] < 2.0, (
                    f"Demo task {task['name']} should be under-resolved (<2 cells), "
                    f"got {res['cells_across_feature']:.2f}"
                )
            else:
                assert ok, (
                    f"Resolution rule violation in {'--quick' if quick_mode else '--full'} "
                    f"task {task['name']}: cpw={task['cpw']}, "
                    f"cells_across_feature={res['cells_across_feature']:.2f} "
                    f"(target N={task['cells_per_feature']})"
                )
                assert res["cells_across_feature"] >= float(
                    task["cells_per_feature"]
                ) * (1.0 - 0.02)

    # Mutation check 1: cpw < 10 must fail the audit
    bad_cpw_task = dict(get_task_table(quick=True)[0], cpw=7)
    ok_cpw, _, _ = _audit_task(bad_cpw_task)
    assert not ok_cpw, "Mutation check failed: cpw=7 was not rejected by resolution audit"

    # Mutation check 2: cells_per_feature=2 (or missing override) must fail the audit
    bad_feat_task = dict(get_task_table(quick=True)[0], cells_per_feature=2)
    ok_feat, res_feat, _ = _audit_task(bad_feat_task)
    assert not ok_feat and res_feat["cells_across_feature"] < 4.0, (
        "Mutation check failed: cells_per_feature=2 was not rejected by resolution audit"
    )
    assert coupling_mesh_overrides(gap=50e-9, cells_per_feature=0, cpw=10) == []


def test_ring_coupler_geometry_and_cmt_analytics():
    """
    Verify geometry fill fractions around the 50 nm coupling slot, agreement
    between numerical mode-overlap CMT and closed-form CMT (< 1e-4), exponential
    decay rate vs analytical evanescent constant, and conformal bent mode shift.
    """
    structures, ports, domain = build_ring_coupler(gap=GAP_NOMINAL, extrude_into_pml=True)
    assert len(structures) == 2
    assert [p.name for p in ports] == ["in", "through", "drop", "add"]
    assert domain[0] < -5.0e-6 and domain[1] > 5.0e-6

    bus_slab, ring_slab = structures
    # Check fill = 1 inside bus core (0, 0), fill = 0 in 50 nm slot (0, 0.275 um),
    # and fill = 1 inside bottom of ring core (0, 0.55 um)
    h_small = np.array([10e-9])
    z0 = np.array([0.0])
    assert float(
        bus_slab.fill(np.array([0.0]), np.array([0.0]), z0, h_small, h_small, h_small)[0]
    ) == pytest.approx(1.0)
    assert float(
        bus_slab.fill(
            np.array([0.0]), np.array([0.275e-6]), z0, h_small, h_small, h_small
        )[0]
    ) == pytest.approx(0.0)
    assert float(
        ring_slab.fill(
            np.array([0.0]), np.array([0.275e-6]), z0, h_small, h_small, h_small
        )[0]
    ) == pytest.approx(0.0)
    assert float(
        ring_slab.fill(
            np.array([0.0]), np.array([0.55e-6]), z0, h_small, h_small, h_small
        )[0]
    ) == pytest.approx(1.0)

    # Coupled-mode theory consistency checks across 50..250 nm
    gaps = np.array([50.0, 100.0, 150.0, 200.0, 250.0]) * 1e-9
    cmt = cmt_coupling_vs_gap(
        gaps, radius=R_RING, width=W_WG, n_core=N_SI, n_clad=N_SIO2, thickness=H_SI
    )
    # 1. Numerical isolated-mode overlap matches closed-form analytic mu_0
    assert cmt["mu_0_overlap"] == pytest.approx(cmt["mu_0_analytic"], rel=1e-4)

    # 2. Unitary power conservation in CMT and supermode models
    np.testing.assert_allclose(cmt["kappa2_cmt"] + cmt["t2_cmt"], 1.0, atol=1e-14)
    np.testing.assert_allclose(
        cmt["kappa2_supermode"] + cmt["t2_supermode"], 1.0, atol=1e-14
    )

    # 3. Monotonic exponential decay with gap; fitted decay constant matches gamma_evan
    assert np.all(np.diff(cmt["kappa2_cmt"]) < 0.0)
    assert np.all(np.diff(cmt["kappa2_supermode"]) < 0.0)
    fit_cmt = fit_exponential_decay(gaps, cmt["theta_cmt"])
    assert fit_cmt["r_squared"] > 0.9999
    assert fit_cmt["decay_rate_m"] == pytest.approx(cmt["gamma_evan"], rel=1e-3)

    # 5-layer supermode slot enhancement increases coupling at g = 50 nm vs weak CMT,
    # while converging to isolated-mode CMT at g = 250 nm (within 3%)
    assert cmt["kappa2_supermode"][0] > cmt["kappa2_cmt"][0] * 1.15
    assert cmt["kappa2_supermode"][-1] == pytest.approx(cmt["kappa2_cmt"][-1], rel=0.03)

    # 4. Conformally mapped bent-waveguide radial mode shifts outward (u > 0)
    bm = bent_waveguide_radial_mode(
        cmt["neff_slab"], N_SIO2, width=W_WG, radius=R_RING, wavelength=WL_REF
    )
    assert bm["centroid_shift_nm"] > 40.0
    assert bm["neff_bent"] > bm["neff_straight"]


def test_ring_coupler_passivity_reciprocity_and_gap_mutation():
    """
    Fast 2D EIM FIT verification at `L1 = (10 cpw, 4 cells/gap)` on a compact
    coupler domain (`r = 1.5 um`) of:
    1. Mesh resolution rules (`cpw == 10`, `cells_across_feature >= 4`),
    2. Passivity (`|S11|^2 + |S21|^2 + |S31|^2 + |S41|^2 <= 1.008`),
    3. Directional point-coupling range and forward directivity (`|S31|^2 / |S41|^2 > 25`),
    4. Lorentz reciprocity (`|S21| == |S12|`) and mirror symmetry (`|S31| == |S42|`),
    5. Wavelength dispersion (`|kappa(long wl)|^2 > |kappa(short wl)|^2`),
    6. Mutation check: widening the coupling gap from 50 nm to 180 nm reduces
       drop power coupling `|kappa|^2` by > 4x and increases through power `|t|^2`.
    """
    r_test = 1.5e-6
    wl_lo, wl_hi = 1480e-9, 1620e-9
    structs_50, ports_50, dom_50, ov_50 = _build_compact_coupler(
        gap=50e-9, radius=r_test
    )
    circuit_50 = PhotonicCircuit(
        wavelength_min=wl_lo,
        wavelength_max=wl_hi,
        n_freqs=15,
        structures=structs_50,
        ports=ports_50,
        n_core=N_SI,
        n_clad=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=10,
        domain=dom_50,
        n_pml=6,
        mesh_overrides=ov_50,
    )
    res_50 = measure_mesh_resolution(circuit_50.grid.x, circuit_50.grid.y, gap=50e-9)
    assert res_50["cells_across_feature"] >= 4.0 - 1e-6

    s_50 = circuit_50.run_s_parameters(
        drive=["in", "through"], energy_decay=1e-2, collect_fields=True
    )

    wl = circuit_50.wavelengths
    order = np.argsort(wl)
    wl_s = wl[order]
    mid = slice(2, -2)

    s11 = s_50[("in", "in")][order][mid]
    s21 = s_50[("through", "in")][order][mid]
    s31 = s_50[("drop", "in")][order][mid]
    s41 = s_50[("add", "in")][order][mid]
    s12 = s_50[("in", "through")][order][mid]
    s42 = s_50[("add", "through")][order][mid]

    t2_50 = np.abs(s21) ** 2
    k2_50 = np.abs(s31) ** 2
    r2_50 = np.abs(s11) ** 2
    a2_50 = np.abs(s41) ** 2

    # 1. Passivity across interior band
    total_power = t2_50 + k2_50 + r2_50 + a2_50
    assert float(np.max(total_power)) <= 1.008, (
        f"Passivity violation: max total guided power = {np.max(total_power):.5f}"
    )
    assert float(np.min(total_power)) > 0.85

    # 2. Point-coupling magnitude & high forward directivity
    i_1550 = int(np.argmin(np.abs(wl_s[mid] - WL_REF)))
    assert 0.06 < float(k2_50[i_1550]) < 0.30
    assert 0.65 < float(t2_50[i_1550]) < 0.94
    assert float(np.max(r2_50)) < 5e-3
    assert float(np.max(a2_50)) < 5e-3
    assert float(k2_50[i_1550] / max(float(a2_50[i_1550]), 1e-12)) > 25.0

    # 3. Lorentz reciprocity |S21 - S12| and left-right mirror symmetry |S31 - S42|
    rec_err = float(np.max(np.abs(np.abs(s21) - np.abs(s12))))
    sym_err = float(np.max(np.abs(np.abs(s31) - np.abs(s42))))
    assert rec_err < 1e-4, f"Bus reciprocity error ||S21| - |S12|| = {rec_err:.3e}"
    assert sym_err < 1e-4, f"Mirror symmetry error ||S31| - |S42|| = {sym_err:.3e}"

    # 4. Longer wavelength has larger evanescent tail -> stronger coupling |kappa|^2
    assert float(k2_50[-1]) > float(k2_50[0])

    # Check angled 45-deg DFT slice and ring resonator synthesis helper on real FIT data
    fmap = circuit_50.dft_fields[WL_REF]
    sl = sample_angled_drop_slice(
        fmap["x"],
        fmap["y"],
        fmap["Ex"],
        fmap["Ey"],
        ring_center_xy=(0.0, W_WG + 50e-9 + r_test),
        radius=r_test,
        u_span=1.3e-6,
    )
    assert float(np.max(np.abs(sl["E_radial"]))) > 0.0
    rm = ring_resonator_metrics(wl_s[mid], s21, s31, radius=R_RING)
    assert 12.0 < rm["fsr_nm"] < 25.0
    assert rm["finesse"] > 5.0

    # 5. Mutation check: widen gap from 50 nm to 180 nm -> |kappa|^2 drops > 4x
    structs_180, ports_180, dom_180, ov_180 = _build_compact_coupler(
        gap=180e-9, radius=r_test
    )
    circuit_180 = PhotonicCircuit(
        wavelength_min=wl_lo,
        wavelength_max=wl_hi,
        n_freqs=15,
        structures=structs_180,
        ports=ports_180,
        n_core=N_SI,
        n_clad=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=10,
        domain=dom_180,
        n_pml=6,
        mesh_overrides=ov_180,
    )
    res_180 = measure_mesh_resolution(circuit_180.grid.x, circuit_180.grid.y, gap=180e-9)
    assert res_180["cells_across_feature"] >= 4.0 - 1e-6

    s_180 = circuit_180.run_s_parameters(
        drive=["in"], energy_decay=1e-2, collect_fields=False
    )
    k2_180 = float(np.abs(s_180[("drop", "in")][order][mid][i_1550]) ** 2)
    t2_180 = float(np.abs(s_180[("through", "in")][order][mid][i_1550]) ** 2)

    assert k2_180 < float(k2_50[i_1550]) / 4.0, (
        f"Mutation check failed: |kappa(180nm)|^2 = {k2_180:.4f} not < |kappa(50nm)|^2 / 4 = {k2_50[i_1550]/4:.4f}"
    )
    assert t2_180 > float(t2_50[i_1550]) + 0.06


def test_full_study_data_convergence_and_gap_decay_audit():
    """
    Audit the persisted ``--full`` study dataset in ``tutorials/05_guided_waves/05-09_ring_coupling/data/``:
    1. Every non-demo task in ``task_0..task_12`` satisfies ``cpw >= 10``,
       ``cells_across_feature >= 4.0``, and ``Nz == 1``.
    2. Kottke converges monotonically across ``L1 -> L2 -> L3`` with last-step
       relative change ``< 1e-3`` and relative error ``< 1%`` vs ``kottke_reference``
       (``R = (40, 16)``), outperforming ``staircase_L3`` in both ``|t|^2`` and
       ``|kappa|^2`` accuracy and backward ring isolation ``|S41|^2``.
    3. Passivity, Lorentz reciprocity, and CPML extrusion suppression (> 500x)
       hold on the primary ``kottke_L3`` dataset.
    4. Gap sweep ``|kappa(g)|^2`` (50..250 nm) decays exponentially with
       ``R^2 > 0.998`` and matches the analytic 5-layer supermode decay rate
       within 2.5%, with a mutation check on gap ordering.
    """
    import json

    data_dir = TUTORIAL_DIR / "data"
    full_table = get_task_table(quick=False)
    assert len(full_table) == 14

    tasks_npz = {}
    for cfg in full_table:
        idx = cfg["index"]
        j_path = data_dir / f"task_{idx}.json"
        npz_path = data_dir / f"task_{idx}.npz"
        assert j_path.is_file() and npz_path.is_file(), f"Missing full data for task_{idx}"
        meta = json.loads(j_path.read_text())
        d = dict(np.load(npz_path, allow_pickle=False))
        assert meta["name"] == cfg["name"] == str(d["name"])
        assert meta["grid_shape"][2] == 1
        if cfg["is_demo"]:
            assert float(meta["cells_across_feature"]) < 2.0
        else:
            assert int(meta["cpw"]) >= 10
            assert float(meta["cells_across_feature"]) >= 4.0 - 1e-6
            assert float(d["cells_across_feature"]) >= 4.0 - 1e-6
        tasks_npz[cfg["name"]] = d

    def _t2_k2_a2_1550(d):
        i0 = int(np.argmin(np.abs(d["wavelengths"] - WL_REF)))
        return (
            float(np.abs(d["S21"][i0]) ** 2),
            float(np.abs(d["S31"][i0]) ** 2),
            float(np.abs(d["S41"][i0]) ** 2),
        )

    t_ref, k_ref, a_ref = _t2_k2_a2_1550(tasks_npz["kottke_reference"])
    t_l2, k_l2, _ = _t2_k2_a2_1550(tasks_npz["kottke_L2"])
    t_l3, k_l3, a_l3 = _t2_k2_a2_1550(tasks_npz["kottke_L3"])
    ts_l3, ks_l3, as_l3 = _t2_k2_a2_1550(tasks_npz["staircase_L3"])

    # Convergence of Kottke L2 -> L3 and L3 vs R=(40, 16)
    assert abs(t_l3 - t_l2) / t_l3 < 1e-4
    assert abs(k_l3 - k_l2) / k_l3 < 1e-4
    err_t_k3 = abs(t_l3 - t_ref) / t_ref
    err_k_k3 = abs(k_l3 - k_ref) / k_ref
    err_t_s3 = abs(ts_l3 - t_ref) / t_ref
    err_k_s3 = abs(ks_l3 - k_ref) / k_ref
    assert err_t_k3 < 3e-3 and err_k_k3 < 1e-2
    assert err_t_k3 < err_t_s3 / 3.0
    assert err_k_k3 < err_k_s3 / 3.0
    assert a_l3 < as_l3 / 3.0 and a_ref < a_l3
    t_nu, k_nu, a_nu = _t2_k2_a2_1550(tasks_npz["kottke_nonuniform"])
    err_t_nu = abs(t_nu - t_ref) / t_ref
    err_k_nu = abs(k_nu - k_ref) / k_ref
    assert int(tasks_npz["kottke_nonuniform"]["n_cells"]) < 0.65 * int(tasks_npz["kottke_L3"]["n_cells"])
    assert float(tasks_npz["kottke_nonuniform"]["cells_across_feature"]) == pytest.approx(8.0, rel=1e-4)
    assert err_t_nu < 3e-3 and err_k_nu < 1.2e-2
    assert err_t_nu < err_t_s3 and err_k_nu < err_k_s3
    assert a_nu < 2.0e-5

    # Passivity, reciprocity, and CPML extrusion on primary kottke_L3
    p = tasks_npz["kottke_L3"]
    wl_nm = p["wavelengths"] * 1e9
    order = np.argsort(wl_nm)
    mask = (wl_nm[order] >= 1505.0) & (wl_nm[order] <= 1595.0)
    s11 = p["S11"][order][mask]
    s21 = p["S21"][order][mask]
    s31 = p["S31"][order][mask]
    s41 = p["S41"][order][mask]
    s12 = p["S12"][order][mask]
    s42 = p["S42"][order][mask]
    noext_s41 = p["noext_S41"][order][mask]

    p_sum = np.abs(s11) ** 2 + np.abs(s21) ** 2 + np.abs(s31) ** 2 + np.abs(s41) ** 2
    assert float(np.max(p_sum)) <= 1.002
    assert float(np.min(p_sum)) > 0.990
    assert float(np.max(np.abs(np.abs(s21) - np.abs(s12)))) < 1e-5
    assert float(np.max(np.abs(np.abs(s31) - np.abs(s42)))) < 1e-5
    assert float(np.max(np.abs(s41) ** 2)) < float(np.min(np.abs(noext_s41) ** 2)) / 500.0

    # Gap sweep decay rate vs analytic 5-layer supermode prediction
    gaps_m = np.array([50.0, 100.0, 150.0, 200.0, 250.0]) * 1e-9
    k2_gaps = np.array(
        [
            k_l3,
            _t2_k2_a2_1550(tasks_npz["gap_100nm"])[1],
            _t2_k2_a2_1550(tasks_npz["gap_150nm"])[1],
            _t2_k2_a2_1550(tasks_npz["gap_200nm"])[1],
            _t2_k2_a2_1550(tasks_npz["gap_250nm"])[1],
        ]
    )
    assert np.all(np.diff(k2_gaps) < 0.0)
    cmt_g = cmt_coupling_vs_gap(gaps_m, n_core_2d=float(p["n_eff_slab"]))
    fit_theta = fit_exponential_decay(gaps_m, np.arcsin(np.sqrt(k2_gaps)))
    sm_theta = fit_exponential_decay(gaps_m, cmt_g["theta_supermode"])
    fit_k2 = fit_exponential_decay(gaps_m, k2_gaps)
    sm_k2 = fit_exponential_decay(gaps_m, cmt_g["kappa2_supermode"])

    assert fit_theta["r_squared"] > 0.998
    assert fit_k2["r_squared"] > 0.999
    assert fit_theta["decay_rate_um"] == pytest.approx(sm_theta["decay_rate_um"], rel=0.025)
    assert fit_k2["decay_rate_um"] == pytest.approx(sm_k2["decay_rate_um"], rel=0.025)

    # Mutation check: scrambling two gap points must break the exponential fit
    scrambled_k2 = k2_gaps.copy()
    scrambled_k2[1], scrambled_k2[3] = scrambled_k2[3], scrambled_k2[1]
    bad_fit = fit_exponential_decay(gaps_m, scrambled_k2)
    assert bad_fit["r_squared"] < 0.70



def test_setup_figures_mesh_upgrade_and_mutation():
    """
    Verify:
    1. Setup figures (`00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`,
       `00_setup_table.md`), mesh comparison figure (`04b_mesh_upgrade_comparison.png`),
       and pulse animation GIF (`09_pulse_propagation.gif` < 5 MB) exist in `figures/`.
    2. `build_ring_coupler_graded_axes` produces an exact left-right mirror-symmetric
       x-axis (`x + x[::-1] == 0`), exact `8.0` cells across the `50 nm` coupling slot,
       and `< 65%` of `kottke_L3`'s cell count, with a mutation check when
       `cells_per_feature` is changed from `8` to `4`.
    3. `build_ring_coupler_hybrid_unstructured_mesh` generates a body-fitted
       square+triangle `UnstructuredMesh2D` around the circular ring rim.
    """
    fig_dir = TUTORIAL_DIR / "figures"
    for fname in (
        "00_setup_table.md",
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "04b_mesh_upgrade_comparison.png",
        "09_pulse_propagation.gif",
    ):
        fpath = fig_dir / fname
        assert fpath.is_file() and fpath.stat().st_size > 1000, f"Missing or empty figure: {fname}"
    gif_size_mb = (fig_dir / "09_pulse_propagation.gif").stat().st_size / (1024.0 * 1024.0)
    assert gif_size_mb < 5.0, f"GIF exceeds 5 MB limit: {gif_size_mb:.2f} MB"

    _, _, domain = build_ring_coupler(gap=GAP_NOMINAL, extrude_into_pml=True)
    x_nu, y_nu, z_nu, _ = build_ring_coupler_graded_axes(
        domain=domain, gap=GAP_NOMINAL, cpw=20, cells_per_feature=8
    )
    assert len(z_nu) - 1 == 1
    assert np.allclose(x_nu + x_nu[::-1], 0.0, atol=1e-15), "x_axis must be exact mirror-symmetric"
    res_nu = measure_mesh_resolution(x_nu, y_nu, gap=GAP_NOMINAL)
    assert res_nu["cells_across_feature"] == pytest.approx(8.0, rel=1e-6)
    assert res_nu["dx_min_nm"] == pytest.approx(6.25, rel=1e-6)
    assert res_nu["n_cells"] < 170000

    # Mutation check: halving `cells_per_feature` from 8 to 4 halves cells across slot
    # and doubles dx_min_nm from 6.25 nm to 12.50 nm
    x_mut, y_mut, _, _ = build_ring_coupler_graded_axes(
        domain=domain, gap=GAP_NOMINAL, cpw=20, cells_per_feature=4
    )
    res_mut = measure_mesh_resolution(x_mut, y_mut, gap=GAP_NOMINAL)
    assert res_mut["cells_across_feature"] == pytest.approx(4.0, rel=1e-6)
    assert res_mut["dx_min_nm"] == pytest.approx(12.50, rel=1e-6)
    assert res_mut["n_cells"] < res_nu["n_cells"]

    # Hybrid square+triangle unstructured mesh check on curved ring rim
    mesh_hyb, _, info_hyb = build_ring_coupler_hybrid_unstructured_mesh(
        gap=GAP_NOMINAL, radius=R_RING, width=W_WG, h=0.16e-6, levels=2
    )
    assert info_hyb["n_quads"] > 500
    assert info_hyb["n_tris"] > 500
    assert info_hyb["n_cells"] == info_hyb["n_quads"] + info_hyb["n_tris"]
    assert mesh_hyb.n_cells == info_hyb["n_cells"]
