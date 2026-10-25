# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics and geometry validation tests for Tutorial 5.4:
Silicon Strip-to-Slot Waveguide Mode Converters (2D EIM).
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.constants import c as c_light

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "05_guided_waves"
    / "05-04_strip_to_slot"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    NonUniformStripToSlotPIC,
    asymmetric_taper,
    build_mmi_graded_axes,
    slot_straight,
    slot_waveguide_mode,
    slot_waveguide_neff,
)
from strip_to_slot import (  # noqa: E402
    EPS_SI,
    GAP_SLOT,
    H_SI,
    L_MMI,
    N_SI,
    N_SIO2,
    W_MMI,
    W_RAIL,
    W_SLOT,
    W_STRIP,
    WL_MAX,
    WL_MIN,
    WL_REF,
    build_asymmetric_taper_converter,
    build_mesh_overrides,
    build_mmi_converter,
    build_symmetric_taper_converter,
    build_task_simulation,
    get_task_table,
    measure_task_resolution,
)
from wakis.templates.photonics import (  # noqa: E402
    PICSimulation,
    PolySlab,
    Port,
    slab_neff,
)


def test_asymmetric_taper_and_converter_geometries():
    """
    Verify that `asymmetric_taper` deduplicates sharp-tip vertices, raises on
    invalid widths, and that all three converter builders produce valid CCW
    `PolySlab` geometries with expected vertex counts and symmetry.
    """
    # Right-end sharp tip collapses from 4 vertices to a 3-vertex triangle
    v_right_tip = asymmetric_taper(0.0, 8.0e-6, -0.20e-6, 0.20e-6, 0.0, 0.0)
    assert v_right_tip.shape == (3, 2)

    # Left-end sharp tip collapses to a 3-vertex triangle
    v_left_tip = asymmetric_taper(0.0, 8.0e-6, 0.26e-6, 0.26e-6, 0.05e-6, 0.31e-6)
    assert v_left_tip.shape == (3, 2)

    # Degenerate zero-width at both ends or inverted y bounds must fail
    with pytest.raises(ValueError):
        asymmetric_taper(0.0, 5.0e-6, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        asymmetric_taper(0.0, 5.0e-6, 0.3e-6, -0.1e-6, -0.1e-6, 0.3e-6)

    # Straight slot helper returns two 4-vertex rail polygons separated by GAP_SLOT
    v_bot, v_top = slot_straight(0.0, 4.0e-6, W_SLOT, GAP_SLOT)
    assert v_bot.shape == (4, 2)
    assert v_top.shape == (4, 2)
    assert float(v_top[:, 1].min() - v_bot[:, 1].max()) == pytest.approx(GAP_SLOT)

    # Design 1 (MMI): single 16-vertex mirror-symmetric PolySlab
    s_mmi, p_mmi, _ = build_mmi_converter()
    assert len(s_mmi) == 1
    assert len(p_mmi) == 2
    assert s_mmi[0].vertices.shape == (16, 2)
    y_verts = s_mmi[0].vertices[:, 1]
    assert np.allclose(np.sort(y_verts), np.sort(-y_verts), atol=1e-15)

    # Design 2 (Symmetric Taper): 3 PolySlabs with mirror symmetry across y = 0
    s_sym, _, _ = build_symmetric_taper_converter()
    assert len(s_sym) == 3
    assert s_sym[0].vertices.shape == (5, 2)
    assert np.allclose(
        np.sort(s_sym[1].vertices[:, 1]),
        np.sort(-s_sym[2].vertices[:, 1]),
        atol=1e-15,
    )

    # Design 3 (Asymmetric Taper): 2 PolySlabs (upper continuous + lower tip-tapered)
    s_asym, _, _ = build_asymmetric_taper_converter()
    assert len(s_asym) == 2


def test_task_tables_resolution_and_cells_across_feature():
    """
    Verify that every non-demo task in both the `--quick` and `--full` task
    tables satisfies `cpw >= 10` and `cells_across_feature >= 4` measured
    directly from the built `PICSimulation` mesh (without running the solver),
    that the explicit under-resolved demo pair is present, and that mutating
    `n_feature` below 4 is detected.
    """
    for quick in (True, False):
        tasks = get_task_table(quick=quick)
        demo_tasks = [t for t in tasks if t.get("is_demo", False)]
        non_demo_tasks = [t for t in tasks if not t.get("is_demo", False)]

        assert [t["name"] for t in demo_tasks] == [
            "underresolved_kottke",
            "underresolved_staircase",
        ]
        assert len(non_demo_tasks) >= 11

        for t in demo_tasks:
            assert t["cpw"] >= 10
            assert t["n_feature"] == 0
            pic_demo = build_task_simulation(t, use_gpu=False)
            res_demo = measure_task_resolution(pic_demo, t)
            assert 1.0 <= res_demo["cells_across_feature"] < 2.5

        for t in non_demo_tasks:
            assert t["cpw"] >= 10, (
                f"Task {t['name']} (quick={quick}) has cpw={t['cpw']} < 10"
            )
            assert t["n_feature"] >= 4, (
                f"Task {t['name']} (quick={quick}) has n_feature={t['n_feature']} < 4"
            )
            pic = build_task_simulation(t, use_gpu=False)
            res = measure_task_resolution(pic, t)
            assert res["cells_across_feature"] >= 4.0 - 1e-6, (
                f"Task {t['name']} (quick={quick}) has only "
                f"{res['cells_across_feature']:.2f} cells across feature (< 4)"
            )
            assert res["slot_cells"] >= 4.0 - 1e-6
            assert res["dx_min_nm"] <= res["min_feature_nm"] / 4.0 + 1e-6

    # Swept gap parameter check: when gap_slot_m is varied, override scales with it
    swept_cfg = dict(get_task_table(quick=True)[2])
    swept_cfg["gap_slot_m"] = 60e-9
    swept_cfg["n_feature"] = 6
    pic_sw = build_task_simulation(swept_cfg, use_gpu=False)
    res_sw = measure_task_resolution(pic_sw, swept_cfg)
    assert res_sw["min_feature_nm"] == pytest.approx(60.0)
    assert res_sw["cells_across_feature"] == pytest.approx(6.0, rel=1e-2)
    assert res_sw["dx_min_nm"] == pytest.approx(10.0, rel=1e-2)

    # Mutation check: reducing n_feature from 4 -> 2 must drop cells_across_feature < 4
    mut_cfg = dict(get_task_table(quick=True)[2])
    mut_cfg["n_feature"] = 2
    pic_mut = build_task_simulation(mut_cfg, use_gpu=False)
    res_mut = measure_task_resolution(pic_mut, mut_cfg)
    assert res_mut["cells_across_feature"] < 4.0


def test_slot_mode_field_enhancement_and_discontinuity():
    """
    Verify the slot waveguide normal electric-field discontinuity ratio
    `(n_high / n_low)^2` at the slot walls `y = +-g/2` in both the 5-layer
    analytical model and Wakis 2D EIM `solve_port_modes` at L3=(20, 8), plus a
    refractive-index mutation check.
    """
    neff_slab = slab_neff(N_SI, N_SIO2, H_SI, WL_REF, polarization="TE")
    expected_ratio = (neff_slab / N_SIO2) ** 2  # ~4.1417 in 2D EIM

    ana = slot_waveguide_mode(
        n_high=neff_slab,
        n_low=N_SIO2,
        slot_width=GAP_SLOT,
        rail_width=W_RAIL,
        wavelength=WL_REF,
        y_span=1.6e-6,
        n_pts=1601,
    )
    assert ana["discontinuity_ratio"] == pytest.approx(expected_ratio, rel=1e-12)
    assert 2.10 < ana["neff"] < 2.22
    assert ana["slot_confinement"] > 0.50

    # Check analytical jump right across y = +g/2 = +50 nm (dy_a = 1.0 nm)
    y_a = ana["y"]
    ey_a = ana["Ey"]
    dy_a = ana["Dy_norm"]
    ey_inside = float(np.interp(0.5 * GAP_SLOT - 1.5e-9, y_a, ey_a))
    ey_outside = float(np.interp(0.5 * GAP_SLOT + 1.5e-9, y_a, ey_a))
    assert (ey_inside / ey_outside) == pytest.approx(expected_ratio, rel=2e-2)
    # Normal displacement Dy is continuous across y = +g/2
    dy_in = float(np.interp(0.5 * GAP_SLOT - 1.5e-9, y_a, dy_a))
    dy_out = float(np.interp(0.5 * GAP_SLOT + 1.5e-9, y_a, dy_a))
    assert dy_in == pytest.approx(dy_out, rel=2e-2)

    # Compare against Wakis 2D EIM eigenmode on the output slot port at L3=(20, 8)
    structures, ports, domain = build_mmi_converter()
    overrides = build_mesh_overrides("mmi", n_feature=8)
    pic = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=20,
        domain=domain,
        n_pml=8,
        mesh_overrides=overrides,
    )
    m_slot = pic.solve_port_modes("out", frequencies=c_light / WL_REF, n_modes=1)[0]
    neff_num = float(np.real(m_slot.neff))
    assert neff_num == pytest.approx(ana["neff"], abs=0.025)

    y_num = m_slot.coords["Eb"][0]
    ey_num = np.abs(m_slot.e["b"][:, 0])
    ey_slot_center = float(np.interp(0.0, y_num, ey_num))
    ey_rail_center = float(np.interp(0.18e-6, y_num, ey_num))
    enhancement_num = ey_slot_center / ey_rail_center
    enhancement_ana = float(
        np.interp(0.0, y_a, ey_a) / np.interp(0.18e-6, y_a, ey_a)
    )
    assert enhancement_num > 2.4
    assert enhancement_num == pytest.approx(enhancement_ana, rel=0.12)

    # Check jump across y = +g/2 between slot interior and rail interior
    dy_slot = GAP_SLOT / 8.0
    ey_pre_wall = float(np.interp(0.5 * GAP_SLOT - 1.2 * dy_slot, y_num, ey_num))
    ey_post_wall = float(np.interp(0.5 * GAP_SLOT + 1.8 * dy_slot, y_num, ey_num))
    wall_jump_num = ey_pre_wall / ey_post_wall
    assert wall_jump_num == pytest.approx(expected_ratio, rel=0.25)

    # Mutation check: lowering core index n_Si from 3.47 -> 2.20 must reduce
    # both slot effective index and slot-to-rail field enhancement ratio
    neff_slab_mut = slab_neff(2.20, N_SIO2, H_SI, WL_REF, polarization="TE")
    ana_mut = slot_waveguide_mode(
        n_high=neff_slab_mut,
        n_low=N_SIO2,
        slot_width=GAP_SLOT,
        rail_width=W_RAIL,
        wavelength=WL_REF,
    )
    assert ana_mut["neff"] < ana["neff"] - 0.30
    assert ana_mut["discontinuity_ratio"] < 0.5 * ana["discontinuity_ratio"]


def test_mmi_conversion_efficiency_vs_length_and_mutation():
    """
    Verify at resolved level `L1 = (cpw=10, N=4)` that 2D EIM MMI strip-to-slot
    conversion efficiency increases monotonically with converter length
    `L in {1.6, 2.4, 6.0} um`, Kottke outperforms staircasing at `L1 = (10, 4)`,
    and mutating `W_mmi` to a non-imaging width degrades transmission.
    """

    def _run_mmi(length_um, width_mmi=W_MMI, avg="kottke", cpw=10, n_feature=4):
        L_c = float(length_um) * 1e-6
        hs = 0.5 * W_STRIP
        hm = 0.5 * float(width_mmi)
        hsl = 0.5 * W_SLOT
        hg = 0.5 * GAP_SLOT
        x_in = -2.20e-6
        x_out = L_c + 2.20e-6
        verts = np.array(
            [
                [x_in, -hs],
                [0.0, -hs],
                [0.0, -hm],
                [L_MMI, -hm],
                [L_c, -hsl],
                [x_out, -hsl],
                [x_out, -hg],
                [L_MMI, -hg],
                [L_MMI, +hg],
                [x_out, +hg],
                [x_out, +hsl],
                [L_c, +hsl],
                [L_MMI, +hm],
                [0.0, +hm],
                [0.0, +hs],
                [x_in, +hs],
            ],
            dtype=float,
        )
        slab = PolySlab(verts, z_min=-0.5 * H_SI, z_max=0.5 * H_SI, material=EPS_SI)
        ports = [
            Port(
                "in",
                center=(-1.15e-6, 0.0, 0.0),
                size=(0.0, 1.75e-6, 1.50e-6),
                direction=+1,
            ),
            Port(
                "out",
                center=(L_c + 1.15e-6, 0.0, 0.0),
                size=(0.0, 1.75e-6, 1.50e-6),
                direction=+1,
            ),
        ]
        dom = (-1.95e-6, L_c + 1.95e-6, -1.42e-6, +1.42e-6)
        overrides = build_mesh_overrides("mmi", length_conv=L_c, n_feature=n_feature)
        pic = PICSimulation(
            WL_MIN,
            WL_MAX,
            n_freqs=15,
            structures=[slab],
            ports=ports,
            n_core=N_SI,
            n_clad=N_SIO2,
            wavelength_ref=WL_REF,
            two_d=True,
            averaging=avg,
            cells_per_wavelength=cpw,
            domain=dom,
            n_pml=7,
            mesh_overrides=overrides,
        )
        s = pic.run_s_parameters(
            drive=["in"], energy_decay=2e-3, collect_fields=False
        )
        i0 = int(np.argmin(np.abs(pic.wavelengths - WL_REF)))
        t0 = float(np.abs(s[("out", "in")][i0]) ** 2)
        r0 = float(np.abs(s[("in", "in")][i0]) ** 2)
        return t0, r0

    t_16, _ = _run_mmi(1.6, cpw=10, n_feature=4)
    t_24, _ = _run_mmi(2.4, cpw=10, n_feature=4)
    t_60, r_60 = _run_mmi(6.0, cpw=10, n_feature=4)

    # Efficiency increases monotonically with length from short non-adiabatic taper to 6 um
    assert t_16 < t_24 < t_60, (
        f"Expected T(1.6) < T(2.4) < T(6.0), got {t_16:.4f}, {t_24:.4f}, {t_60:.4f}"
    )
    assert t_60 > 0.94, f"Expected T(6.0 um) > 0.94, got {t_60:.4f}"
    assert r_60 < 0.02, f"Expected R(6.0 um) < 0.02, got {r_60:.4f}"
    assert t_60 + r_60 <= 1.01, f"Passivity violation T + R = {t_60 + r_60:.4f}"

    # Kottke vs staircase check at resolved L1 = (cpw=10, N=4)
    t_s_l1, _ = _run_mmi(6.0, avg="staircase", cpw=10, n_feature=4)
    assert t_60 - t_s_l1 > 0.015, (
        f"Expected Kottke to beat staircase at L1=(10,4) by >0.015, got {t_60:.4f} vs {t_s_l1:.4f}"
    )

    # Geometry mutation check: detuning MMI width W_mmi from 1.24 um to 2.40 um
    # breaks the TE0-TE2 two-fold self-imaging at x = L_mmi = 1.38 um and drops T
    t_mut, _ = _run_mmi(6.0, width_mmi=2.40e-6, avg="kottke", cpw=10, n_feature=4)
    assert t_60 - t_mut > 0.15, (
        f"Mutation check failed: detuned W_mmi=2.40 um gave T={t_mut:.4f} vs nominal {t_60:.4f}"
    )


def test_full_study_data_convergence_and_topologies():
    """
    Audit the committed `--full` dataset (`tutorials/05_guided_waves/05-04_strip_to_slot/data/`):
    1. All 15 tasks (`task_0`..`task_14`) exist and every non-demo task JSON
       satisfies `cpw >= 10` and `cells_across_feature >= 4`.
    2. Kottke error `|T(1550 nm) - T_ref|` and broadband RMS error against the
       `R = (40, 16)` reference decrease monotonically across `L1 -> L2 -> L3`,
       with relative change `|T_R - T_L3| / T_R < 5e-4`, and Kottke beats
       staircase by `> 6x` at every resolved level.
    3. Converter efficiencies match published benchmarks (`Design 1` in
       `[0.965, 0.976]`, `Designs 2 & 3 > 0.995`), 2D flux box is conserved,
       and the 5-point MMI length sweep is strictly monotonic (with a mutation
       check on length ordering and averaging swap).
    """
    data_dir = TUTORIAL_DIR / "data"
    expected_tasks = get_task_table(quick=False)
    assert len(expected_tasks) == 16

    metas = {}
    npz_by_name = {}
    for t in expected_tasks:
        idx = t["index"]
        jf = data_dir / f"task_{idx}.json"
        nf = data_dir / f"task_{idx}.npz"
        assert jf.is_file(), f"Missing {jf}"
        assert nf.is_file(), f"Missing {nf}"
        meta = json.loads(jf.read_text())
        d = dict(np.load(nf, allow_pickle=False))
        assert meta["name"] == t["name"]
        assert str(d["name"]) == t["name"]
        metas[t["name"]] = meta
        npz_by_name[t["name"]] = d

        if meta["is_demo"]:
            assert meta["cpw"] >= 10
            assert 1.0 <= meta["cells_across_feature"] < 2.5
        else:
            assert meta["cpw"] >= 10, f"{meta['name']} cpw={meta['cpw']} < 10"
            assert meta["cells_across_feature"] >= 4.0, (
                f"{meta['name']} cells_across_feature={meta['cells_across_feature']} < 4"
            )
            assert meta["slot_cells"] >= 4.0

    def _t_r_1550(d):
        w = d["wavelengths"]
        i0 = int(np.argmin(np.abs(w - WL_REF)))
        return float(np.abs(d["S21"][i0]) ** 2), float(np.abs(d["S11"][i0]) ** 2)

    def _t_band(d):
        w = d["wavelengths"] * 1e9
        order = np.argsort(w)
        m = (w[order] >= 1505.0) & (w[order] <= 1595.0)
        return np.abs(d["S21"][order][m]) ** 2

    t_ref, r_ref = _t_r_1550(npz_by_name["kottke_reference"])
    t_ref_band = _t_band(npz_by_name["kottke_reference"])
    assert metas["kottke_reference"]["cpw"] == 40
    assert metas["kottke_reference"]["n_feature"] == 16
    assert 0.965 <= t_ref <= 0.976
    assert r_ref < 1e-3

    err_k = []
    err_s = []
    rms_k = []
    rms_s = []
    for lvl in ("L1", "L2", "L3"):
        tk, _ = _t_r_1550(npz_by_name[f"kottke_{lvl}"])
        ts, _ = _t_r_1550(npz_by_name[f"staircase_{lvl}"])
        err_k.append(abs(tk - t_ref))
        err_s.append(abs(ts - t_ref))
        rms_k.append(
            float(np.sqrt(np.mean((_t_band(npz_by_name[f"kottke_{lvl}"]) - t_ref_band) ** 2)))
        )
        rms_s.append(
            float(
                np.sqrt(np.mean((_t_band(npz_by_name[f"staircase_{lvl}"]) - t_ref_band) ** 2))
            )
        )

    # Monotonic Kottke convergence across L1 -> L2 -> L3
    assert err_k[0] > err_k[1] > err_k[2], f"Non-monotonic Kottke error: {err_k}"
    assert rms_k[0] > rms_k[1] > rms_k[2], f"Non-monotonic Kottke band RMS: {rms_k}"
    # Relative change between L3=(20,8) and R=(40,16) is < 0.05%
    assert err_k[2] / t_ref < 5e-4
    # Kottke beats staircase by > 6x at every resolved level
    for lvl_idx in range(3):
        assert err_s[lvl_idx] / err_k[lvl_idx] > 6.0
        assert rms_s[lvl_idx] / rms_k[lvl_idx] > 6.0

    # Topology benchmarks and 2D Poynting flux conservation at L3=(20, 8)
    t_mmi, r_mmi = _t_r_1550(npz_by_name["kottke_L3"])
    t_sym, r_sym = _t_r_1550(npz_by_name["design2_sym_taper"])
    t_asym, r_asym = _t_r_1550(npz_by_name["design3_asym_taper"])
    assert 0.965 <= t_mmi <= 0.976 and r_mmi < 1e-3
    assert t_sym > 0.995 and r_sym < 1e-5
    assert t_asym > 0.995 and r_asym < 1e-5
    for key in ("kottke_L3", "design2_sym_taper", "design3_asym_taper"):
        f_box = npz_by_name[key]["flux_total_box"]
        assert float(np.mean(f_box)) == pytest.approx(1.0, abs=1e-2)

    # 5-point MMI length sweep monotonicity at L3=(20, 8)
    sweep_keys = [
        "length_L1.6um",
        "length_L2.2um",
        "length_L3.0um",
        "length_L4.5um",
        "kottke_L3",
    ]
    t_sweep = [_t_r_1550(npz_by_name[k])[0] for k in sweep_keys]
    assert all(t_sweep[i] < t_sweep[i + 1] for i in range(len(t_sweep) - 1)), (
        f"Expected strictly increasing T(L), got {t_sweep}"
    )

    # Mutation check: swapping Kottke and staircase errors or reversing two sweep points must fail
    assert not (err_s[0] > err_s[1] > err_s[2])
    t_sweep_mut = list(t_sweep)
    t_sweep_mut[1], t_sweep_mut[2] = t_sweep_mut[2], t_sweep_mut[1]
    assert not all(
        t_sweep_mut[i] < t_sweep_mut[i + 1] for i in range(len(t_sweep_mut) - 1)
    )

    # Graded non-uniform Cartesian mesh upgrade (kottke_nonuniform, task_15)
    t_nu, r_nu = _t_r_1550(npz_by_name["kottke_nonuniform"])
    m_u = metas["kottke_L3"]
    m_nu = metas["kottke_nonuniform"]
    assert m_nu["n_cells"] < 0.65 * m_u["n_cells"]
    assert m_nu["elapsed_s"] < 0.55 * m_u["elapsed_s"]
    assert abs(t_nu - t_ref) < 1.2e-3
    assert r_nu < 1e-3


def test_setup_schematic_and_nonuniform_mesh_upgrade():
    """
    Verify that the setup schematic figures (``00a_setup_geometry.png``,
    ``00b_setup_mesh_and_excitation.png``, ``00_setup_table.md``, and
    ``00b_taper_converters_schematic.png``), the mesh-upgrade comparison figure
    (``04b_mesh_upgrade_comparison.png``), all split 3-panel field figures, and
    the ``< 5 MB`` GIF exist, and verify ``build_mmi_graded_axes`` along with a
    cladding-grading mutation check.
    """
    fig_dir = TUTORIAL_DIR / "figures"
    required_figs = (
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00b_taper_converters_schematic.png",
        "01_converter_topologies_layout.png",
        "02_eim_vertical_stack.png",
        "03_slot_mode_physics_and_discontinuity.png",
        "04_permittivity_mesh_zoom.png",
        "04b_mesh_upgrade_comparison.png",
        "05_port_mode_profiles.png",
        "06_port_neff_dispersion.png",
        "07_mmi_field_intensity_1550.png",
        "08_mmi_field_real_ey_1550.png",
        "10_all_designs_field_comparison_1550.png",
        "10b_all_designs_real_ey_1550.png",
        "11_sparameter_spectra_linear_and_db.png",
        "12_energy_budget_and_passivity.png",
        "13_conversion_efficiency_vs_length.png",
        "14_kottke_vs_staircase_convergence.png",
        "15_resolution_spectra_comparison.png",
        "16_port_time_trace.png",
    )
    for fname in required_figs:
        fpath = fig_dir / fname
        assert fpath.is_file() and fpath.stat().st_size > 8_000, f"Missing/empty {fpath}"

    table_path = fig_dir / "00_setup_table.md"
    assert table_path.is_file() and table_path.stat().st_size > 200, (
        f"Missing/empty {table_path}"
    )

    gif_path = fig_dir / "09_pulse_propagation.gif"
    assert gif_path.is_file() and 0 < gif_path.stat().st_size < 5 * 1024 * 1024

    # Verify graded non-uniform axes properties and mutation check
    xg, yg, _ = build_mmi_graded_axes(
        domain=(-2.60e-6, 8.60e-6, -1.48e-6, +1.48e-6),
        length_conv=6.00e-6,
        length_mmi=L_MMI,
        width_mmi=W_MMI,
        width_slot=W_SLOT,
        gap_slot=GAP_SLOT,
        n_feature=8,
        cpw=20,
    )
    assert np.allclose(yg, -yg[::-1], atol=1e-15)
    dy = np.diff(yg)
    dx = np.diff(xg)
    in_slot = (yg[1:] > -0.5 * GAP_SLOT + 1e-12) & (yg[:-1] < 0.5 * GAP_SLOT - 1e-12)
    assert int(np.sum(in_slot)) == 8
    assert np.allclose(dy[in_slot], GAP_SLOT / 8.0, rtol=1e-10)
    # Non-square axial cells in taper section are > 2x the transverse slot cell size
    assert float(np.max(dx)) > 2.4 * float(np.min(dy))

    # Mutation check: disabling transverse grading (forcing uniform 12.5 nm everywhere in y)
    # must increase Ny by > 1.8x relative to build_mmi_graded_axes
    ny_graded = len(yg) - 1
    ny_ungraded = int(np.ceil((yg[-1] - yg[0]) / (GAP_SLOT / 8.0)))
    assert ny_ungraded > 1.8 * ny_graded


