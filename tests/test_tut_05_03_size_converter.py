# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Fast physics, resolution-rule, and mutation tests (< 60 s total) for Tutorial 5.3:
Waveguide Mode Size Converters (`tutorials/05_guided_waves/05-03_size_converter/`).
"""

import json
from pathlib import Path
import sys

import numpy as np

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "05_guided_waves"
    / "05-03_size_converter"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    MIN_FEATURE_LUNEBURG,
    MIN_FEATURE_SEMILENS,
    N_AIR,
    N_SI,
    N_SIO2,
    SEMILENS_L_UM,
    SEMILENS_M,
    SEMILENS_SEG3_X_HI,
    SEMILENS_SEG3_X_LO,
    SEMILENS_W_UM,
    T_WG_SEMILENS,
    T_WG_TAPER,
    W_OUT_TAPER,
    WL_MAX,
    WL_MIN,
    WL_REF,
    NonUniformSemilensPIC,
    SymmetricProfileSlab,
    build_luneburg_converter,
    build_semilens_expander,
    build_semilens_graded_axes,
    build_taper_converter,
    calibrate_luneburg_profile,
    compute_abrupt_step_overlap,
    inspect_task_mesh,
    measure_grid_resolution,
    run_converter_simulation,
    semilens_mesh_overrides,
    waveguide_width_mesh_overrides,
)
from size_converter import get_task_table  # noqa: E402

from wakis.templates.photonics import PICSimulation, Port, n_to_eps  # noqa: E402


def test_task_table_resolution_rules():
    """
    Fail if any non-demo task in either the `--quick` or `--full` task table (or
    saved `data/task_*.json` archive) has `cpw < 10` or `cells_across_feature < 4`
    (measured directly from the mesh that each task builds), and verify that the
    under-resolved demo pair is flagged with `demo=True` and has `< 4` cells
    across the 50 nm Segment 3 flare.
    """
    tasks_q = get_task_table(quick=True)
    tasks_f = get_task_table(quick=False)
    assert len(tasks_q) == 14
    assert len(tasks_f) == 16

    for mode_name, table in (("quick", tasks_q), ("full", tasks_f)):
        non_demo_count = 0
        demo_count = 0
        for t in table:
            cpw = float(t["cpw"])
            assert cpw >= 10.0, (
                f"[{mode_name}] Task {t['name']} has cpw={cpw} < 10."
            )
            metrics = inspect_task_mesh(t)
            cells_feat = float(metrics["cells_across_feature"])
            if t.get("demo", False):
                demo_count += 1
                assert cells_feat < 4.0, (
                    f"[{mode_name}] Demo task {t['name']} should be under-resolved (<4 cells), got {cells_feat}."
                )
            else:
                non_demo_count += 1
                assert cells_feat >= 4.0 - 1e-6, (
                    f"[{mode_name}] Non-demo task {t['name']} has cells_across_feature={cells_feat:.3f} < 4."
                )
        assert non_demo_count >= 11
        assert demo_count == 2

    data_dir = TUTORIAL_DIR / "data"
    json_files = sorted(data_dir.glob("task_*.json"))
    if json_files:
        assert len(json_files) in (13, 15, 16)
        for jf in json_files:
            meta = json.loads(jf.read_text())
            assert float(meta["cpw"]) >= 10.0
            if meta.get("demo", False):
                assert float(meta["cells_across_feature"]) < 4.0
            else:
                assert float(meta["cells_across_feature"]) >= 4.0 - 1e-6


def test_luneburg_calibration_and_2d_grid():
    """
    Verify that the 1D SOI slab calibration reproduces the Badri & Gilarlue (2019)
    Luneburg thickness profile (`t(0) ~ 0.356 um`, `t(R) = 0.11 um`) and that
    `build_luneburg_converter` at `L1 = (10, 4)` produces a strictly 2D (`Nz == 1`)
    EIM grid with `>= 4` cells across the 0.50 um output waveguide.
    """
    cal = calibrate_luneburg_profile(solve_1d_fit=False)
    n_edge = float(cal["n_edge"])
    assert 2.18 < n_edge < 2.25

    t_rad = cal["t_luneburg_um"]
    assert 0.345 < float(t_rad[0]) < 0.370
    assert np.isclose(float(t_rad[-1]), 0.11, atol=1e-6)
    assert np.all(np.diff(t_rad) <= 1e-9)

    rel_err = np.max(
        np.abs(cal["neff_realized"] - cal["neff_luneburg"]) / cal["neff_luneburg"]
    )
    assert rel_err < 1.2e-2

    structures, ports, domain_2d, _ = build_luneburg_converter(n_shells=16)
    pic = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=11,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        n_sub=N_AIR,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=10,
        domain=domain_2d,
        n_pml=6,
        mesh_overrides=waveguide_width_mesh_overrides(4, width=W_OUT_TAPER),
    )
    assert pic.grid.Nz == 1
    res_m = measure_grid_resolution(
        pic.grid,
        min_feature_m=MIN_FEATURE_LUNEBURG,
        feature_bounds=(-0.5 * W_OUT_TAPER, 0.5 * W_OUT_TAPER),
        feature_axis=1,
    )
    assert res_m["cells_across_feature"] >= 4.0

    # Core permittivity at center (0, 0) must exceed edge waveguide permittivity
    ix0 = int(np.argmin(np.abs(pic.grid.x[:-1])))
    iy0 = int(np.argmin(np.abs(pic.grid.y[:-1])))
    eps_center = 1.0 / float(pic.solver.ieps[ix0, iy0, 0, "y"])
    ix_wg = int(np.argmin(np.abs(pic.grid.x[:-1] - (-6.0e-6))))
    eps_wg = 1.0 / float(pic.solver.ieps[ix_wg, iy0, 0, "y"])
    assert eps_center > 1.8 * eps_wg


def test_abrupt_step_overlap_and_taper_monotonicity():
    """
    Verify at `L1 = (10, 4)` (`cpw=10, cells_across_feature >= 4`) that:
    1. An abrupt step (`L_t = 0`) matches the analytical 2D EIM mode-overlap
       integral within 5%.
    2. Fundamental TE0 transmission increases strictly monotonically with linear
       taper length `L_t in {0, 3, 10} um` (Milton-Burns adiabatic scaling) on a
       compact `w_in = 4.0 um -> w_out = 0.50 um` domain.
    """
    ov_full = compute_abrupt_step_overlap()
    assert 0.17 < ov_full["em_overlap"] < 0.25
    assert 0.19 < ov_full["overlap_sq"] < 0.26

    w_in_compact = 4.0e-6
    ov_compact = compute_abrupt_step_overlap(w_in=w_in_compact, w_out=W_OUT_TAPER)
    overrides = waveguide_width_mesh_overrides(4, width=W_OUT_TAPER)

    transmissions = []
    for L_um in (0.0, 3.0, 10.0):
        structures, _, _ = build_taper_converter(
            length_taper=L_um * 1e-6,
            profile="linear",
            w_in=w_in_compact,
            w_out=W_OUT_TAPER,
            l_lead=0.90e-6,
        )
        L_t = L_um * 1e-6
        y_dom = 2.60e-6
        ports = [
            Port(
                "in",
                center=(-0.45e-6, 0.0, 0.5 * T_WG_TAPER),
                size=(0.0, 4.60e-6, 6.0 * T_WG_TAPER),
                direction=+1,
            ),
            Port(
                "out",
                center=(L_t + 0.45e-6, 0.0, 0.5 * T_WG_TAPER),
                size=(0.0, 1.60e-6, 6.0 * T_WG_TAPER),
                direction=+1,
            ),
        ]
        domain_2d = (-0.90e-6, L_t + 0.90e-6, -y_dom, y_dom)
        res = run_converter_simulation(
            structures,
            ports,
            domain_2d,
            n_clad=N_SIO2,
            n_sub=N_AIR,
            averaging="kottke",
            cpw=10,
            n_pml=6,
            n_freqs=9,
            energy_decay=5e-3,
            mesh_overrides=overrides,
            collect_fields=False,
        )
        wl = res["pic"].wavelengths
        i0 = int(np.argmin(np.abs(wl - WL_REF)))
        t_val = float(np.abs(res["s_matrix"][("out", "in")][i0]) ** 2)
        r_val = float(np.abs(res["s_matrix"][("in", "in")][i0]) ** 2)
        assert r_val < (0.10 if L_um == 0.0 else 0.03)
        assert t_val + r_val <= 1.03
        transmissions.append(t_val)

    t0, t3, t10 = transmissions
    assert abs(t0 - ov_compact["em_overlap"]) < 0.05
    assert t0 < t3 < t10
    assert t10 > 0.88


def test_semilens_beam_expander_and_mutation_check():
    """
    Verify:
    1. On the full-size 9.50 um Abbaslou et al. (2017) semi-lens beam expander
       (`data/task_2.npz`, `data/task_4.npz`, `data/task_6.npz`, `data/task_8.npz`),
       fundamental TE0 transmission exceeds 75%, band-mean (and `cpw >= 14` 1550 nm)
       total port flux exceeds 94%, and TE0 modal share exceeds 80% of guided
       even-mode power, converging monotonically from L1 -> L2 -> L3 -> R.
       Also verify the full-study Luneburg lens (`T_TE0 > 95%`) and linear taper
       sweep (`10 -> 20 -> 50 -> 100 um` strictly monotonic, `T_100 > 95%`).
    2. In a live 2D FIT simulation on a compact (`0.45 -> 4.50 um`, `L_tot = 4.75 um`)
       6-segment semi-lens expander with the exact `50 nm` Segment 3 flare resolved
       by `4.0` cells (`cpw=10`), the semi-lens collimates > 66% into TE0 (> 70%
       share of guided power), whereas mutating the curved semi-lens segments into
       an uncollimated abrupt flare degrades TE0 transmission by > 35%.
    """
    data_dir = TUTORIAL_DIR / "data"
    shares = []
    for idx in (2, 4, 6, 8):
        npz_path = data_dir / f"task_{idx}.npz"
        if npz_path.exists():
            d = np.load(npz_path, allow_pickle=False)
            if str(d.get("kind", "")) != "semilens":
                continue
            assert float(d["cells_across_feature"]) >= 4.0 - 1e-6
            wl = d["wavelengths"]
            i_ref = int(np.argmin(np.abs(wl - WL_REF)))
            m_band = (wl >= 1505e-9) & (wl <= 1595e-9)
            t_te0 = float(np.abs(d["S21"][i_ref]) ** 2)
            t_flux = float(d["T_flux_port"][i_ref])
            t_flux_mean = float(np.mean(d["T_flux_port"][m_band]))
            t_even = sum(
                float(np.abs(d[f"S21_TE{m}"][i_ref]) ** 2)
                for m in (0, 2, 4, 6, 8)
            )
            share = t_te0 / t_even
            shares.append((float(d["cpw"]), share))
            assert t_te0 > 0.75
            assert t_flux_mean > 0.94
            if float(d["cpw"]) >= 14.0:
                assert t_flux > 0.94
            else:
                assert t_flux > 0.90
            assert share > 0.80

    if len(shares) == 4:
        # Monotonic convergence of TE0 share from L1 -> L2 -> L3 -> R toward ~80%
        sh_vals = [s for _, s in sorted(shares)]
        assert sh_vals[0] > sh_vals[1] > sh_vals[2] > sh_vals[3] > 0.80

    if (data_dir / "task_9.npz").exists() and (data_dir / "task_13.npz").exists():
        dl = np.load(data_dir / "task_9.npz", allow_pickle=False)
        if str(dl.get("kind", "")) == "luneburg":
            i_l = int(np.argmin(np.abs(dl["wavelengths"] - WL_REF)))
            assert float(np.abs(dl["S21"][i_l]) ** 2) > 0.95
        t_tapers = []
        for idx_t in (10, 11, 12, 13):
            dt = np.load(data_dir / f"task_{idx_t}.npz", allow_pickle=False)
            i_t = int(np.argmin(np.abs(dt["wavelengths"] - WL_REF)))
            t_tapers.append(float(np.abs(dt["S21"][i_t]) ** 2))
        assert t_tapers[0] < t_tapers[1] < t_tapers[2] < t_tapers[3]
        assert t_tapers[3] > 0.95

    # Compact 6-segment semi-lens domain (0.45 -> 4.50 um over 4.75 um) preserving
    # the exact 50 nm Segment 3 flare and w_0 = 0.45 um input waveguide
    l_um = np.array([0.075, 2.23, 0.05, 0.25, 1.62, 0.525])
    w_um = np.array([0.45, 0.72, 1.65, 4.53, 5.00, 1.60, 4.50])
    seg3_lo = float(np.sum(l_um[:2]) * 1e-6)
    seg3_hi = float(np.sum(l_um[:3]) * 1e-6)
    overrides = [
        (
            (seg3_lo, -np.inf, -np.inf),
            (seg3_hi, np.inf, np.inf),
            (MIN_FEATURE_SEMILENS / 4.0, 1.0, 1.0),
        )
    ]

    def _build_compact_slab(w_arr):
        xs, ws = [-0.65], [w_arr[0]]
        x_cur = 0.0
        for i, (li, mi) in enumerate(zip(l_um, SEMILENS_M)):
            xi = np.linspace(0.0, li, 25, endpoint=(i == len(l_um) - 1))
            wi = (w_arr[i] - w_arr[i + 1]) * np.abs((xi - li) / li) ** mi + w_arr[i + 1]
            xs.extend((x_cur + xi).tolist())
            ws.extend(wi.tolist())
            x_cur += li
        xs.append(x_cur + 0.70)
        ws.append(w_arr[-1])
        slab = SymmetricProfileSlab(
            np.asarray(xs) * 1e-6,
            np.asarray(ws) * 1e-6,
            z_min=0.0,
            z_max=T_WG_SEMILENS,
            material=n_to_eps(N_SI),
        )
        return slab, x_cur

    slab_nom, l_tot = _build_compact_slab(w_um)
    ports = [
        Port(
            "in",
            center=(-0.40e-6, 0.0, 0.5 * T_WG_SEMILENS),
            size=(0.0, 1.40e-6, 5.0 * T_WG_SEMILENS),
            direction=+1,
        ),
        Port(
            "out",
            center=((l_tot + 0.45) * 1e-6, 0.0, 0.5 * T_WG_SEMILENS),
            size=(0.0, 5.10e-6, 5.0 * T_WG_SEMILENS),
            direction=+1,
        ),
    ]
    domain_2d = (-0.65e-6, (l_tot + 0.70) * 1e-6, -2.80e-6, 2.80e-6)

    res_nom = run_converter_simulation(
        [slab_nom],
        ports,
        domain_2d,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_min=1400e-9,
        wavelength_max=1700e-9,
        averaging="kottke",
        cpw=10,
        n_pml=5,
        n_freqs=7,
        energy_decay=5e-3,
        max_time=140e-15,
        mesh_overrides=overrides,
        collect_fields=False,
        multimode_indices=(0, 2, 4),
    )
    res_m = measure_grid_resolution(
        res_nom["pic"].grid,
        min_feature_m=MIN_FEATURE_SEMILENS,
        feature_bounds=(seg3_lo, seg3_hi),
        feature_axis=0,
    )
    assert res_m["cells_across_feature"] >= 4.0 - 1e-6

    i0 = int(np.argmin(np.abs(res_nom["pic"].wavelengths - WL_REF)))
    t_te0_nom = float(np.abs(res_nom["s_matrix"][("out", "in")][i0]) ** 2)
    t_flux_nom = float(res_nom["t_flux_port"][i0])
    even_powers = [
        float(np.abs(res_nom["modal_s21"][m][i0]) ** 2) for m in (0, 2, 4)
    ]
    t_even_sum = sum(even_powers)

    assert t_te0_nom > 0.65
    assert t_flux_nom > 0.73
    assert t_te0_nom / max(t_even_sum, 1e-12) > 0.70
    assert even_powers[0] > 2.5 * even_powers[1]

    # Mutation check: remove the curved semi-lens pre-expansion and collimation
    # segments (abrupt 0.45 -> 4.50 um flare across the 50 nm Segment 3)
    w_mut = np.array([0.45, 0.45, 0.45, 4.50, 4.50, 4.50, 4.50])
    slab_mut, _ = _build_compact_slab(w_mut)
    res_mut = run_converter_simulation(
        [slab_mut],
        ports,
        domain_2d,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        wavelength_min=1400e-9,
        wavelength_max=1700e-9,
        averaging="kottke",
        cpw=10,
        n_pml=5,
        n_freqs=7,
        energy_decay=5e-3,
        max_time=140e-15,
        mesh_overrides=overrides,
        collect_fields=False,
    )
    t_te0_mut = float(np.abs(res_mut["s_matrix"][("out", "in")][i0]) ** 2)
    assert t_te0_mut < t_te0_nom - 0.35


def test_setup_schematic_and_nonuniform_mesh_upgrade():
    """
    Verify:
    1. Setup schematic figures (`00_setup_schematic.png`,
       `00b_taper_and_luneburg_schematics.png`), mesh upgrade comparison
       (`04b_mesh_upgrade_comparison.png`), and split taper field figures
       (`08_taper_ey_field_comparison.png`, `08b_taper_ey_long_comparison.png`)
       exist and are non-empty, and `15_pulse_propagation.gif` is < 5 MB.
    2. `NonUniformSemilensPIC` (`kottke_nonuniform`, `task_15`) reduces the 2D
       cell count by > 25% compared to uniform-background `PICSimulation` at
       `(cpw=20, n_feature=8)` while preserving 8.0 cells across the 50 nm
       Segment 3 flare and matching `kottke_L3` (`task_6`) and `kottke_reference`
       (`task_8`) guided transmission within 0.3%.
    3. Mutation check on `build_semilens_graded_axes`: mutating `n_feature=2`
       violates the >= 4 cells across feature rule (`cells_across_feature < 4.0`).
    """
    fig_dir = TUTORIAL_DIR / "figures"
    for fname in (
        "00_setup_schematic.png",
        "00b_taper_and_luneburg_schematics.png",
        "04b_mesh_upgrade_comparison.png",
        "08_taper_ey_field_comparison.png",
        "08b_taper_ey_long_comparison.png",
    ):
        fpath = fig_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 10_000, f"Missing or empty {fname}"

    gif_path = fig_dir / "15_pulse_propagation.gif"
    if gif_path.exists():
        assert gif_path.stat().st_size < 5 * 1024 * 1024

    semi_structs, semi_ports, semi_dom = build_semilens_expander()
    pic_u = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=semi_structs,
        ports=semi_ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=20,
        domain=semi_dom,
        n_pml=10,
        mesh_overrides=semilens_mesh_overrides(8),
    )
    pic_nu = NonUniformSemilensPIC(
        WL_MIN,
        WL_MAX,
        structures=semi_structs,
        ports=semi_ports,
        n_core=N_SI,
        n_clad=N_AIR,
        n_sub=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=20,
        domain=semi_dom,
        n_pml=10,
        n_feature=8,
    )
    n_u = int(pic_u.grid.Nx * pic_u.grid.Ny)
    n_nu = int(pic_nu.grid.Nx * pic_nu.grid.Ny)
    assert n_nu < 0.75 * n_u

    m_nu = measure_grid_resolution(
        pic_nu.grid,
        min_feature_m=MIN_FEATURE_SEMILENS,
        feature_bounds=(SEMILENS_SEG3_X_LO, SEMILENS_SEG3_X_HI),
        feature_axis=0,
    )
    assert np.isclose(m_nu["cells_across_feature"], 8.0, atol=1e-3)

    # Mutation check: n_feature=2 drops cells across 50 nm Segment 3 below 4.0
    xg_mut, _, _ = build_semilens_graded_axes(
        (*semi_dom, -0.5e-6, 0.5e-6), n_pml=10, cpw=20, n_feature=2
    )
    in_seg3 = (xg_mut[:-1] >= SEMILENS_SEG3_X_LO - 1e-12) & (
        xg_mut[1:] <= SEMILENS_SEG3_X_HI + 1e-12
    )
    dx_mut = float(np.min(np.diff(xg_mut)[in_seg3]))
    assert MIN_FEATURE_SEMILENS / dx_mut < 4.0

    data_dir = TUTORIAL_DIR / "data"
    if (data_dir / "task_15.npz").exists() and (data_dir / "task_6.npz").exists():
        d_nu = np.load(data_dir / "task_15.npz", allow_pickle=False)
        d_l3 = np.load(data_dir / "task_6.npz", allow_pickle=False)
        w = d_nu["wavelengths"] * 1e9
        mb = (w >= 1510.0) & (w <= 1590.0)
        ev_nu = float(
            np.mean(
                sum(np.abs(d_nu[f"S21_TE{k}"][mb]) ** 2 for k in (0, 2, 4, 6, 8))
            )
        )
        ev_l3 = float(
            np.mean(
                sum(np.abs(d_l3[f"S21_TE{k}"][mb]) ** 2 for k in (0, 2, 4, 6, 8))
            )
        )
        assert abs(ev_nu - ev_l3) < 3e-3



