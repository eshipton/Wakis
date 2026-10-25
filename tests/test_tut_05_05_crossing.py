# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics and geometry validation tests for Tutorial 5.5:
Waveguide crossing based on cosine tapers (2D EIM, Chandran et al. 2020).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wakis.templates.photonics import PICSimulation, PolySlab, Port, n_to_eps

TUTORIAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tutorials"
    / "05_guided_waves"
    / "05-05_crossing"
    / "crossing.py"
)


def _load_crossing_module():
    spec = importlib.util.spec_from_file_location("pic_03_crossing", TUTORIAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cosine_taper_geometry_and_d4_symmetry():
    """
    Verify closed-form vs fsolve cosine-taper parameters (a, b), polygon area
    ordering (cosine > linear > plain), exact D4 (90-deg rotational and mirror)
    invariance of PolySlab.fill, and the analytical TE0-TE2 phase integral.
    """
    mod = _load_crossing_module()
    a, b = mod.solve_cosine_taper_params()
    assert a * 1e-6 == pytest.approx(0.392996, rel=1e-5)
    assert b == pytest.approx(0.963732, rel=1e-5)
    assert -b / (a * 1e-6) == pytest.approx(-2.45227, rel=1e-4)

    # Endpoint and peak half-width checks on w_m * cos(a * x + b)
    x_in = -(0.5 * mod.W_OUT + mod.L_TAPER)
    x_out = -0.5 * mod.W_OUT
    x_peak = -b / a
    assert mod.W_M * np.cos(a * x_in + b) == pytest.approx(0.5 * mod.W_IN, rel=1e-9)
    assert mod.W_M * np.cos(a * x_out + b) == pytest.approx(0.5 * mod.W_OUT, rel=1e-9)
    assert mod.W_M * np.cos(a * x_peak + b) == pytest.approx(mod.W_M, rel=1e-12)

    # Polygon area ordering: convex cosine > monotonic linear > untapered plain
    def _poly_area(v):
        return 0.5 * float(
            np.abs(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]))
        )

    s_cos, _ = mod.build_crossing("cosine", n_taper=80)
    s_lin, _ = mod.build_crossing("linear", n_taper=80)
    s_pln, _ = mod.build_crossing("plain")
    area_cos = _poly_area(s_cos[0].vertices)
    area_lin = _poly_area(s_lin[0].vertices)
    area_pln = _poly_area(s_pln[0].vertices)
    assert area_cos > area_lin > area_pln

    # Exact D4 symmetry of PolySlab.fill on a symmetric 2D grid
    pts = np.linspace(-5.5e-6, 5.5e-6, 24)
    X, Y, Z = np.meshgrid(pts, pts, [0.0], indexing="ij")
    h = np.full_like(X, pts[1] - pts[0])
    hz = np.full_like(Z, 0.1e-6)
    fill_xy = s_cos[0].fill(X, Y, Z, h, h, hz, n_sub=6)[:, :, 0]
    assert np.allclose(fill_xy, fill_xy[::-1, :], atol=1e-12)
    assert np.allclose(fill_xy, fill_xy[:, ::-1], atol=1e-12)
    assert np.allclose(fill_xy, fill_xy.T, atol=1e-12)

    # Analytical 1D lateral TE0-TE2 phase integral across the 5.30 um cosine taper
    x_tap = np.linspace(x_in, x_out, 150)
    w_tap = 2.0 * mod.W_M * np.cos(a * x_tap + b)
    n0 = np.array([mod.lateral_slab_mode_neff(w, m=0) for w in w_tap])
    n2 = np.array([mod.lateral_slab_mode_neff(w, m=2) for w in w_tap])
    valid = np.isfinite(n2)
    k0 = 2.0 * np.pi / mod.WL_REF
    dphi_pi = float(np.trapezoid(k0 * (n0[valid] - n2[valid]), x_tap[valid]) / np.pi)
    assert 2.5 < dphi_pi < 4.2, f"Unexpected TE0-TE2 phase integral {dphi_pi:.2f} pi"


def test_task_table_resolution_and_mesh_overrides():
    """
    Verify that every task in both the --quick and --full task tables obeys:
    1. cells_per_wavelength (cpw) >= 10 in the highest-index slab material,
    2. cells_across_feature >= 4 across the smallest geometric feature
       (w_in = 0.35 um), measured directly on the constructed tensor-product
       mesh without running the time-domain solver.
    Also verify that crossing_mesh_overrides refines the access-waveguide
    region whenever max_cell = min_feature / N < dx_background, and that the
    mesh audit detects an under-resolved configuration (mutation check).
    """
    mod = _load_crossing_module()

    quick_tasks = mod.get_task_table(quick=True)
    full_tasks = mod.get_task_table(quick=False)
    assert len(quick_tasks) == 9
    assert len(full_tasks) == 11

    for mode_name, table in (("quick", quick_tasks), ("full", full_tasks)):
        for task in table:
            cpw = int(task["cpw"])
            n_feat_target = int(task["n_feature_min"])
            assert cpw >= 10, (
                f"Task {task['name']} in --{mode_name} has cpw={cpw} < 10"
            )
            assert n_feat_target >= 4, (
                f"Task {task['name']} in --{mode_name} has n_feature_min={n_feat_target} < 4"
            )
            metrics = mod.inspect_task_mesh(task)
            assert metrics["min_feature_nm"] == pytest.approx(350.0, rel=1e-6)
            assert metrics["cells_across_feature"] >= 4.0, (
                f"Task {task['name']} in --{mode_name} resolves w_in with only "
                f"{metrics['cells_across_feature']:.2f} cells (< 4)"
            )
            assert metrics["cells_across_feature"] >= float(n_feat_target) - 1e-6
            assert metrics["dx_min_nm"] <= metrics["dx_background_nm"] + 1e-6
            assert metrics["n_cells"] > 0
            assert metrics["n_steps_max"] > 0

    # Verify mesh override activation when N requires a cell smaller than dx_bg
    overrides_12 = mod.crossing_mesh_overrides(cpw=10, n_feature_min=12)
    assert len(overrides_12) >= 1
    refined_task = dict(quick_tasks[0])
    refined_task["n_feature_min"] = 12
    refined_task["mesh_overrides"] = overrides_12
    m_ref = mod.inspect_task_mesh(refined_task)
    assert m_ref["dx_min_nm"] < m_ref["dx_background_nm"]
    assert m_ref["cells_across_feature"] >= 12.0 - 1e-6

    # Mutation check: a coarse grid on a narrow 100 nm feature yields < 4 cells
    under_resolved = mod.inspect_task_mesh(
        {
            "cpw": 10,
            "n_feature_min": 4,
            "n_pml": 6,
            "domain": mod.DOMAIN_2D,
            "mesh_overrides": None,
        },
        min_feature=0.10e-6,
    )
    assert under_resolved["cells_across_feature"] < 4.0


def test_crossing_tapered_vs_plain_and_mutation():
    """
    Verify in 2D EIM FIT at cells_per_wavelength = 10 (L1 resolution, ~8.15
    cells across w_in = 0.35 um) on a compact-y domain containing the full
    l_t = 5.30 um horizontal cosine taper and intersection that:
    1. The cosine-tapered crossing achieves > 92% through transmission and
       < -30 dB cross-port crosstalk while respecting passivity.
    2. It outperforms the plain untapered crossing by > 14 percentage points in
       through power and > 15 dB in crosstalk suppression, focusing the beam
       waist away from the intersection corners.
    3. Mutation check: collapsing the cosine bulge from w_m = 0.75 um to
       w_m = 0.56 um degrades through transmission and increases radiative loss.
    """
    mod = _load_crossing_module()
    lx_port, lx_dom = 5.95e-6, 6.30e-6
    ly_port, ly_dom = 0.90e-6, 1.30e-6
    ports = [
        Port(
            "west",
            center=(-lx_port, 0.0, 0.0),
            size=(0.0, 1.5e-6, mod.PORT_SPAN_Z),
            direction=+1,
        ),
        Port(
            "east",
            center=(+lx_port, 0.0, 0.0),
            size=(0.0, 1.5e-6, mod.PORT_SPAN_Z),
            direction=-1,
        ),
        Port(
            "north",
            center=(0.0, +ly_port, 0.0),
            size=(1.6e-6, 0.0, mod.PORT_SPAN_Z),
            direction=-1,
        ),
        Port(
            "south",
            center=(0.0, -ly_port, 0.0),
            size=(1.6e-6, 0.0, mod.PORT_SPAN_Z),
            direction=+1,
        ),
    ]

    def _run_geom(kind="cosine", w_m=mod.W_M, collect_fields=False):
        structures, _ = mod.build_crossing(kind=kind, n_taper=26, w_m=w_m)
        pic = PICSimulation(
            wavelength_min=1.18e-6,
            wavelength_max=1.44e-6,
            n_freqs=5,
            structures=structures,
            ports=ports,
            n_core=mod.N_SI,
            n_clad=mod.N_SIO2,
            wavelength_ref=mod.WL_REF,
            two_d=True,
            averaging="kottke",
            cells_per_wavelength=10,
            domain=(-lx_dom, lx_dom, -ly_dom, ly_dom),
            n_pml=5,
            field_wavelengths=(mod.WL_REF,),
            record_movie=False,
        )
        s_mat = pic.run_s_parameters(
            drive=["west"], energy_decay=3e-2, collect_fields=collect_fields
        )
        i0 = int(np.argmin(np.abs(pic.wavelengths - mod.WL_REF)))
        t_thru = np.abs(s_mat[("east", "west")]) ** 2
        t_north = np.abs(s_mat[("north", "west")]) ** 2
        t_south = np.abs(s_mat[("south", "west")]) ** 2
        r_west = np.abs(s_mat[("west", "west")]) ** 2
        return pic, i0, t_thru, t_north, t_south, r_west

    pic_cos, i0, t_cos, xn_cos, xs_cos, r_cos = _run_geom(
        "cosine", w_m=mod.W_M, collect_fields=True
    )
    _, _, t_pln, xn_pln, xs_pln, r_pln = _run_geom("plain", collect_fields=False)
    _, _, t_mut, xn_mut, xs_mut, r_mut = _run_geom(
        "cosine", w_m=0.56e-6, collect_fields=False
    )

    mid = slice(1, -1)
    # Passivity check on all three runs
    for t_arr, xn_arr, xs_arr, r_arr in (
        (t_cos, xn_cos, xs_cos, r_cos),
        (t_pln, xn_pln, xs_pln, r_pln),
        (t_mut, xn_mut, xs_mut, r_mut),
    ):
        tot = t_arr[mid] + xn_arr[mid] + xs_arr[mid] + r_arr[mid]
        assert float(np.max(tot)) <= 1.005

    # Cosine crossing transmission and crosstalk at 1310 nm
    assert float(t_cos[i0]) > 0.92, f"Low cosine crossing T(1310) = {t_cos[i0]:.4f}"
    assert float(xn_cos[i0]) < 1e-3, f"High cosine crossing XT(1310) = {xn_cos[i0]:.3e}"
    assert float(np.max(np.abs(xn_cos[mid] - xs_cos[mid]))) < 1e-5

    # Cosine vs plain untapered crossing
    assert float(t_cos[i0] - t_pln[i0]) > 0.14, (
        f"Expected cosine T ({t_cos[i0]:.3f}) >> plain T ({t_pln[i0]:.3f})"
    )
    assert float(xn_cos[i0]) < float(xn_pln[i0]) / 30.0, (
        f"Expected >15 dB crosstalk suppression: cos={xn_cos[i0]:.3e} vs plain={xn_pln[i0]:.3e}"
    )

    # Multimode self-imaging waist focusing check at x = 0
    fmap = pic_cos.dft_fields[mod.WL_REF]
    ix0 = int(np.argmin(np.abs(fmap["x"])))
    iy0 = int(np.argmin(np.abs(fmap["y"])))
    iy_edge = int(np.argmin(np.abs(fmap["y"] - 0.5 * mod.W_OUT)))
    e2_center = float(fmap["E2"][ix0, iy0])
    e2_edge = float(fmap["E2"][ix0, iy_edge])
    assert e2_center > 10.0 * e2_edge, (
        f"Expected strong waist confinement at x=0: center={e2_center:.3e}, edge={e2_edge:.3e}"
    )

    # Mutation check: collapsing w_m from 0.75 um to 0.56 um degrades focusing,
    # reducing through transmission and increasing unguided radiative loss
    assert float(t_mut[i0]) < float(t_cos[i0]) - 0.03, (
        f"Mutated w_m=0.56 um transmission ({t_mut[i0]:.4f}) did not drop vs optimal ({t_cos[i0]:.4f})"
    )
    rad_cos = 1.0 - float(t_cos[i0] + xn_cos[i0] + xs_cos[i0] + r_cos[i0])
    rad_mut = 1.0 - float(t_mut[i0] + xn_mut[i0] + xs_mut[i0] + r_mut[i0])
    assert rad_mut > rad_cos + 0.03, (
        f"Mutated w_m=0.56 um radiative loss ({rad_mut:.4f}) did not increase vs optimal ({rad_cos:.4f})"
    )


def test_crossing_4fold_symmetry():
    """
    Drive all 4 ports ('west', 'east', 'north', 'south') of a D4-symmetric
    crossing at cells_per_wavelength = 10 and verify that all 4 through
    transmissions, 4 reflections, and 8 cross-port couplings agree within
    numerical noise, and S_ij == S_ji.
    """
    mod = _load_crossing_module()
    l_t_short = 0.35e-6
    L_p = 1.15e-6
    L_dom = 1.45e-6
    structures, _ = mod.build_crossing("cosine", n_taper=18, w_m=0.65e-6, l_t=l_t_short)
    ports = [
        Port("west", center=(-L_p, 0.0, 0.0), size=(0.0, 1.3e-6, 0.644e-6), direction=+1),
        Port("east", center=(+L_p, 0.0, 0.0), size=(0.0, 1.3e-6, 0.644e-6), direction=-1),
        Port("north", center=(0.0, +L_p, 0.0), size=(1.3e-6, 0.0, 0.644e-6), direction=-1),
        Port("south", center=(0.0, -L_p, 0.0), size=(1.3e-6, 0.0, 0.644e-6), direction=+1),
    ]
    pic = PICSimulation(
        wavelength_min=1.18e-6,
        wavelength_max=1.44e-6,
        n_freqs=7,
        structures=structures,
        ports=ports,
        n_core=mod.N_SI,
        n_clad=mod.N_SIO2,
        wavelength_ref=mod.WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=10,
        domain=(-L_dom, L_dom, -L_dom, L_dom),
        n_pml=4,
    )
    s_mat = pic.run_s_parameters(
        drive=["west", "east", "north", "south"],
        energy_decay=5e-2,
        collect_fields=False,
    )
    mid = slice(2, -2)

    thru_list = [
        np.abs(s_mat[("east", "west")][mid]) ** 2,
        np.abs(s_mat[("west", "east")][mid]) ** 2,
        np.abs(s_mat[("south", "north")][mid]) ** 2,
        np.abs(s_mat[("north", "south")][mid]) ** 2,
    ]
    for t_arr in thru_list[1:]:
        assert np.allclose(thru_list[0], t_arr, atol=5e-4)

    refl_list = [
        np.abs(s_mat[(p, p)][mid]) ** 2 for p in ("west", "east", "north", "south")
    ]
    for r_arr in refl_list[1:]:
        assert np.allclose(refl_list[0], r_arr, atol=5e-4)

    cross_pairs = [
        ("north", "west"),
        ("south", "west"),
        ("north", "east"),
        ("south", "east"),
        ("west", "north"),
        ("east", "north"),
        ("west", "south"),
        ("east", "south"),
    ]
    x0 = np.abs(s_mat[cross_pairs[0]][mid]) ** 2
    for pair in cross_pairs[1:]:
        assert np.allclose(x0, np.abs(s_mat[pair][mid]) ** 2, atol=5e-4)

    for p_i in ("west", "east", "north", "south"):
        for p_j in ("west", "east", "north", "south"):
            assert np.allclose(
                s_mat[(p_i, p_j)][mid], s_mat[(p_j, p_i)][mid], atol=1e-3
            )


def test_full_study_data_and_zoomed_solver():
    """
    Verify from tutorials/05_guided_waves/05-05_crossing/data/task_*.{json,npz} that:
    1. All 10 full-study tasks satisfy cpw >= 10 and cells_across_feature >= 4.0.
    2. Kottke band RMS error against the 40 cpw reference decreases monotonically
       from L1 -> L2 -> L3, is lower than staircase at every level, and is
       converged at L3 (|T_L3 - T_L2| / T_L3 < 0.2% and |T_R - T_L3| / T_R < 0.2%).
    3. At 1310 nm, cosine taper outperforms linear taper and plain untapered
       crossing in insertion loss and crosstalk, with 2D flux box conserving energy.
    4. build_zoomed_2d_solver reproduces the exact full-grid resolution metrics
       across w_in and distinguishes Kottke subpixel smoothing from staircasing
       (mutation check).
    """
    import json

    mod = _load_crossing_module()
    data_dir = mod.DATA_DIR
    json_files = sorted(data_dir.glob("task_*.json"))
    assert len(json_files) == 11

    tasks = {}
    for jf in json_files:
        meta = json.loads(jf.read_text())
        idx = meta["task_index"]
        assert meta["cpw"] >= 10
        assert meta["n_feature_min"] >= 4
        assert meta["min_feature_nm"] == pytest.approx(350.0, rel=1e-6)
        assert meta["cells_across_feature"] >= 4.0
        npz = dict(np.load(data_dir / f"task_{idx}.npz", allow_pickle=False))
        tasks[meta["name"]] = (meta, npz)

    def _band_metrics(name):
        meta, d = tasks[name]
        wl = d["wavelengths"] * 1e9
        o = np.argsort(wl)
        wl_s = wl[o]
        i0 = int(np.argmin(np.abs(wl_s - 1310.0)))
        mb = (wl_s >= 1270.0) & (wl_s <= 1350.0)
        t = np.abs(d["S21"][o]) ** 2
        x3 = np.abs(d["S31"][o]) ** 2
        r = np.abs(d["S11"][o]) ** 2
        return {
            "meta": meta,
            "d": d,
            "o": o,
            "mb": mb,
            "t_band": t[mb],
            "t0": float(t[i0]),
            "il0_db": float(-10.0 * np.log10(t[i0])),
            "xt0_db": float(10.0 * np.log10(x3[i0])),
            "r_mean": float(np.mean(r[mb])),
        }

    m = {k: _band_metrics(k) for k in tasks}
    t_ref = m["kottke_reference"]["t_band"]

    err_k = [
        float(np.sqrt(np.mean((m[f"kottke_{lvl}"]["t_band"] - t_ref) ** 2)))
        for lvl in ("L1", "L2", "L3")
    ]
    err_s = [
        float(np.sqrt(np.mean((m[f"staircase_{lvl}"]["t_band"] - t_ref) ** 2)))
        for lvl in ("L1", "L2", "L3")
    ]
    assert err_k[0] > err_k[1] > err_k[2]
    for ek, es, lvl in zip(err_k, err_s, ("L1", "L2", "L3")):
        assert ek < es
        assert m[f"kottke_{lvl}"]["r_mean"] < m[f"staircase_{lvl}"]["r_mean"]

    # Convergence at accepted L3 level (< 0.2% relative change from L2 and to R)
    rel_l2_l3 = abs(m["kottke_L3"]["t0"] - m["kottke_L2"]["t0"]) / m["kottke_L3"]["t0"]
    rel_l3_r = (
        abs(m["kottke_reference"]["t0"] - m["kottke_L3"]["t0"])
        / m["kottke_reference"]["t0"]
    )
    assert rel_l2_l3 < 2e-3
    assert rel_l3_r < 2e-3

    # Insertion loss, crosstalk, and taper ordering at 1310 nm (L3)
    assert m["kottke_L3"]["il0_db"] < 0.12
    assert m["kottke_reference"]["il0_db"] < 0.12
    assert m["kottke_L3"]["xt0_db"] < -40.0
    assert (
        m["kottke_L3"]["il0_db"]
        < m["linear_taper_crossing"]["il0_db"]
        < m["plain_crossing"]["il0_db"]
    )
    assert (
        m["kottke_L3"]["xt0_db"]
        < m["linear_taper_crossing"]["xt0_db"]
        < m["plain_crossing"]["xt0_db"]
    )

    # 2D Poynting flux box energy conservation on primary L3 task
    d_l3 = m["kottke_L3"]["d"]
    o_l3 = m["kottke_L3"]["o"]
    wl_l3 = d_l3["wavelengths"][o_l3] * 1e9
    mi = (wl_l3 >= 1265.0) & (wl_l3 <= 1355.0)
    assert float(np.mean(d_l3["flux_total_box"][o_l3][mi])) == pytest.approx(
        1.0, abs=1e-3
    )

    # Verify build_zoomed_2d_solver matches full-grid resolution at R (40 cpw)
    # and differs between kottke and staircase (mutation check)
    structures, _ = mod.build_crossing("cosine")
    sol_k = mod.build_zoomed_2d_solver(
        structures, cpw=40, n_feature_min=16, averaging="kottke"
    )
    sol_s = mod.build_zoomed_2d_solver(
        structures, cpw=40, n_feature_min=16, averaging="staircase"
    )
    res_z = mod.measure_grid_resolution(
        sol_k.grid.x, sol_k.grid.y, min_feature=mod.MIN_FEATURE, has_ghost=True
    )
    assert res_z["cells_across_feature"] == pytest.approx(
        tasks["kottke_reference"][0]["cells_across_feature"], abs=1e-2
    )
    ieps_k = sol_k.ieps[:, :, sol_k.Nz // 2, "y"]
    ieps_s = sol_s.ieps[:, :, sol_s.Nz // 2, "y"]
    assert not np.allclose(ieps_k, ieps_s)

    # Verify graded non-uniform mesh (kottke_nonuniform) vs kottke_L3
    assert "kottke_nonuniform" in m
    assert m["kottke_nonuniform"]["meta"]["n_cells"] < 0.50 * m["kottke_L3"]["meta"]["n_cells"]
    assert abs(m["kottke_nonuniform"]["t0"] - m["kottke_L3"]["t0"]) < 3e-3
    assert m["kottke_nonuniform"]["il0_db"] < 0.12
    assert m["kottke_nonuniform"]["xt0_db"] < -40.0


def test_setup_schematic_and_graded_mesh_mutation():
    """
    Verify that the setup schematics and mesh comparison figures exist and are
    non-empty, the GIF is < 5 MB, and ``build_crossing_graded_axes`` constructs
    an exact D4-symmetric non-uniform grid with non-square cells along the arms
    and fails its feature-resolution check when mutated to a coarse core spacing.
    """
    mod = _load_crossing_module()
    fig_dir = mod.FIG_DIR
    for fname in (
        "00_setup_schematic.png",
        "00b_crossing_geometries_schematic.png",
        "04_permittivity_mesh_zoom.png",
        "04b_mesh_upgrade_comparison.png",
        "04c_taper_subpixel_smoothing.png",
        "07_field_intensity_1310.png",
        "08_field_real_ey_1310.png",
        "09_pulse_propagation.gif",
        "13_multimode_focusing_comparison.png",
        "15_kottke_vs_staircase_convergence.png",
    ):
        p = fig_dir / fname
        assert p.exists() and p.stat().st_size > 1000, f"Missing figure {p}"
    assert (fig_dir / "09_pulse_propagation.gif").stat().st_size < 5 * 1024 * 1024

    xg, yg, zg = mod.build_crossing_graded_axes(cpw=20.0, n_feature_min=8)
    assert np.allclose(xg, yg, atol=1e-15)
    assert np.allclose(xg, -xg[::-1], atol=1e-15)
    du = np.diff(xg)
    # Non-square cells along the horizontal arm: dx in taper > dy in core
    assert float(np.max(du)) > 1.5 * float(np.min(du))
    res_good = mod.measure_grid_resolution(
        xg, yg, zg, min_feature=mod.MIN_FEATURE, has_ghost=False
    )
    assert res_good["cells_across_feature"] >= 16.0

    # Mutation check: coarsening du_fine to 120 nm drops cells across w_in below 4
    xg_mut, yg_mut, zg_mut = mod.build_crossing_graded_axes(
        cpw=20.0, n_feature_min=8, du_fine=120e-9
    )
    res_mut = mod.measure_grid_resolution(
        xg_mut, yg_mut, zg_mut, min_feature=mod.MIN_FEATURE, has_ghost=False
    )
    assert res_mut["cells_across_feature"] < 4.0



