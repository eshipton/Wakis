# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics and mutation tests for Tutorial 5.7: 2x2 MMI Power Splitter (2D EIM).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wakis.templates.photonics import PICSimulation

TUTORIAL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tutorials"
    / "05_guided_waves"
    / "05-07_mmi_2x2"
    / "mmi_2x2.py"
)


def _load_tutorial_module():
    spec = importlib.util.spec_from_file_location("mmi_2x2_tutorial", TUTORIAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mmi_2x2_geometry_and_analytic_beat_length():
    """
    Verify that `build_mmi_2x2` constructs a symmetric CCW PolySlab with 4 ports
    and that the analytical 2D EIM lateral mode solver finds 4 guided modes with
    expected beat length L_pi and 3*L_pi/2 general-interference length.
    """
    mod = _load_tutorial_module()
    structures, ports, domain = mod.build_mmi_2x2(l_mmi=mod.L_MMI)
    assert len(structures) == 1
    assert len(ports) == 4
    assert [p.name for p in ports] == ["in1", "in2", "out1", "out2"]
    assert domain == pytest.approx((-6.5e-6, 6.5e-6, -2.5e-6, 2.5e-6))

    slab = structures[0]
    lo, hi = slab.bounds
    assert lo[0] == pytest.approx(-hi[0], abs=1e-12)
    assert lo[1] == pytest.approx(-hi[1], abs=1e-12)

    # Interior of MMI box (0, 0) and all 4 access waveguides at (+-6.0, +-1.3) um have fill = 1
    pts_x = np.array([0.0, -6.0e-6, -6.0e-6, 6.0e-6, 6.0e-6])
    pts_y = np.array([0.0, +mod.Y_PORT, -mod.Y_PORT, +mod.Y_PORT, -mod.Y_PORT])
    pts_z = np.zeros_like(pts_x)
    h = np.full_like(pts_x, 0.05e-6)
    fills_core = slab.fill(pts_x, pts_y, pts_z, h, h, h, n_sub=6)
    assert np.allclose(fills_core, 1.0, atol=1e-12)

    # Oxide throat gap at (x = +-1.60 um, y = 0) between the S-bends has fill = 0
    xg = np.array([-1.60e-6, +1.60e-6])
    yg = np.zeros(2)
    zg = np.zeros(2)
    hg = np.full(2, 0.05e-6)
    fills_gap = slab.fill(xg, yg, zg, hg, hg, hg, n_sub=6)
    assert np.allclose(fills_gap, 0.0, atol=1e-12)

    # Analytical Soldano & Pennings (1995) lateral eigenmode properties
    ana = mod.analytic_mmi_model(wavelength=mod.WL_REF)
    assert len(ana["neffs"]) == 4
    assert np.all(np.diff(ana["neffs"]) < 0.0)
    assert ana["neffs"][-1] > mod.N_SIO2
    assert ana["w_eff"] > mod.W_MMI
    assert 2.40e-6 < ana["l_pi_exact"] < 2.70e-6
    assert 3.60e-6 < 1.5 * ana["l_pi_exact"] < 4.05e-6

    # Off-axis port launch excites >90% guided MMI power, dominated by m=0 and m=1
    modal_power = np.abs(ana["c_bar"]) ** 2
    assert float(np.sum(modal_power)) > 0.90
    assert float(modal_power[0] + modal_power[1]) > 0.70


def test_mmi_2x2_resolution_ladder_and_mesh_overrides():
    """
    Verify that every task in both the ``--quick`` and ``--full`` study tables
    has ``cpw >= 10`` and ``cells_across_feature >= 4`` measured from the
    actual mesh, and include a mutation check verifying that omitting the
    throat-gap mesh override at 10 cpw drops ``cells_across_feature`` below 4.
    """
    mod = _load_tutorial_module()
    for quick in (True, False):
        tasks = mod.get_task_table(quick=quick)
        assert len(tasks) == (10 if quick else 12)
        for task in tasks:
            cpw = float(task["cpw"])
            n_target = int(task["n_cells_feature"])
            assert cpw >= 10.0, (
                f"Task {task['name']} (quick={quick}) has cpw={cpw} < 10"
            )
            assert n_target >= 4, (
                f"Task {task['name']} (quick={quick}) has N={n_target} < 4"
            )
            grid = mod.build_task_grid_only(task, mod.build_mmi_2x2)
            metrics = mod.measure_grid_resolution(
                grid,
                n_pml=task["n_pml"],
                l_mmi=task["l_mmi"],
                min_feature=task["min_feature"],
            )
            caf = float(metrics["cells_across_feature"])
            assert caf >= 4.0 - 1e-6, (
                f"Task {task['name']} (quick={quick}) has cells_across_feature={caf:.3f} < 4"
            )
            assert caf == pytest.approx(float(n_target), rel=1e-2)
            assert metrics["dx_min_nm"] <= metrics["dx_background_nm"] + 1e-6

    # Mutation check: removing mesh_overrides at 10 cpw leaves < 4 cells across the 200 nm gap
    task_l1 = dict(mod.get_task_table(quick=True)[0])
    task_l1_unrefined = {**task_l1, "mesh_overrides": []}
    grid_unrefined = mod.build_task_grid_only(task_l1_unrefined, mod.build_mmi_2x2)
    metrics_unrefined = mod.measure_grid_resolution(
        grid_unrefined,
        n_pml=task_l1_unrefined["n_pml"],
        l_mmi=task_l1_unrefined["l_mmi"],
        min_feature=task_l1_unrefined["min_feature"],
    )
    assert metrics_unrefined["cells_across_feature"] < 4.0, (
        "Expected unrefined 10 cpw mesh without overrides to have < 4 cells across 200 nm gap, "
        f"got {metrics_unrefined['cells_across_feature']:.3f}"
    )


def test_mmi_2x2_passivity_quadrature_and_length_mutation():
    """
    Verify passivity (|S31|^2 + |S41|^2 <= 1), quadrature phase (~ -90 deg),
    wavelength dispersion at the nominal 3.0 um length, and a length-mutation
    check between the cross regime (L_mmi = 2.40 um) and the 2D EIM 3-dB
    general-interference regime (L_mmi = 3.25 um) at L1 = (10 cpw, N = 4).
    """
    mod = _load_tutorial_module()

    def _run_mmi_length(l_mmi):
        structures, ports, _ = mod.build_mmi_2x2(l_mmi=l_mmi, n_sbend=48)
        x_p = 0.5 * float(l_mmi) + mod.L_SBEND + 0.25e-6
        port_size = (0.0, 1.60e-6, 1.32e-6)
        compact_ports = [
            mod.Port("in1", center=(-x_p, +mod.Y_PORT, 0.0), size=port_size, direction=+1),
            mod.Port("in2", center=(-x_p, -mod.Y_PORT, 0.0), size=port_size, direction=+1),
            mod.Port("out1", center=(+x_p, +mod.Y_PORT, 0.0), size=port_size, direction=+1),
            mod.Port("out2", center=(+x_p, -mod.Y_PORT, 0.0), size=port_size, direction=+1),
        ]
        compact_domain = (-x_p - 0.30e-6, +x_p + 0.30e-6, -1.95e-6, 1.95e-6)
        overrides = mod.build_gap_mesh_overrides(
            l_mmi=l_mmi, n_cells_feature=4, min_feature=mod.GAP_MMI
        )
        pic = PICSimulation(
            wavelength_min=mod.WL_MIN,
            wavelength_max=mod.WL_MAX,
            n_freqs=15,
            structures=structures,
            ports=compact_ports,
            n_core=mod.N_SI,
            n_clad=mod.N_SIO2,
            wavelength_ref=mod.WL_REF,
            two_d=True,
            averaging="kottke",
            cells_per_wavelength=10,
            domain=compact_domain,
            n_pml=6,
            mesh_overrides=overrides,
        )
        res = mod.measure_grid_resolution(
            pic.grid, n_pml=6, l_mmi=l_mmi, min_feature=mod.GAP_MMI
        )
        assert res["cells_across_feature"] >= 4.0 - 1e-6
        s_mat = pic.run_s_parameters(
            drive=["in1"], energy_decay=1e-2, collect_fields=False
        )
        order = np.argsort(pic.wavelengths)
        wl = pic.wavelengths[order]
        s11 = s_mat[("in1", "in1")][order]
        s21 = s_mat[("in2", "in1")][order]
        s31 = s_mat[("out1", "in1")][order]
        s41 = s_mat[("out2", "in1")][order]
        return wl, s11, s21, s31, s41

    # 1. Nominal L_mmi = 3.00 um design length check
    wl, s11_nom, s21_nom, s31_nom, s41_nom = _run_mmi_length(mod.L_MMI)
    mid = slice(2, -2)
    t31_nom = np.abs(s31_nom[mid]) ** 2
    t41_nom = np.abs(s41_nom[mid]) ** 2
    r_nom = np.abs(s11_nom[mid]) ** 2 + np.abs(s21_nom[mid]) ** 2
    t_tot_nom = t31_nom + t41_nom

    # Passivity: |S31|^2 + |S41|^2 <= 1 and total 4-port guided power <= 1
    assert float(np.max(t_tot_nom)) <= 1.0
    assert float(np.max(t_tot_nom + r_nom)) <= 1.0 + 2e-3
    assert float(np.mean(t_tot_nom)) > 0.72
    assert float(np.mean(r_nom)) < 0.03

    # Quadrature phase difference near -90 deg at the design length
    i0 = int(np.argmin(np.abs(wl - mod.WL_REF)))
    dphi_nom_deg = float(np.rad2deg(np.angle(s31_nom[i0] * np.conj(s41_nom[i0]))))
    assert abs(dphi_nom_deg - (-90.0)) < 16.0, (
        f"Unexpected output phase difference {dphi_nom_deg:.2f} deg at L=3.0 um"
    )

    # Wavelength dispersion: as wavelength increases, L_pi shrinks so bar fraction increases
    bar_frac_nom = t31_nom / t_tot_nom
    assert float(bar_frac_nom[-1]) > float(bar_frac_nom[0]) + 0.05

    # 2. Mutation check: perturb L_mmi to cross-dominant (2.40 um) vs 2D EIM 3-dB length (3.25 um)
    _, _, _, s31_cross, s41_cross = _run_mmi_length(2.40e-6)
    _, _, _, s31_3db, s41_3db = _run_mmi_length(3.25e-6)

    t31_c0 = float(np.abs(s31_cross[i0]) ** 2)
    t41_c0 = float(np.abs(s41_cross[i0]) ** 2)
    bar_frac_cross = t31_c0 / (t31_c0 + t41_c0)

    t31_b0 = float(np.abs(s31_3db[i0]) ** 2)
    t41_b0 = float(np.abs(s41_3db[i0]) ** 2)
    bar_frac_3db = t31_b0 / (t31_b0 + t41_b0)
    dphi_3db_deg = float(np.rad2deg(np.angle(s31_3db[i0] * np.conj(s41_3db[i0]))))

    # At L = 2.40 um (near L_eff = L_pi), output is strongly cross-dominated
    assert bar_frac_cross < 0.20, (
        f"Expected cross-dominated split at L=2.40 um, got bar_frac={bar_frac_cross:.3f}"
    )
    # Mutating L to 3.25 um (near L_eff = 3*L_pi/2) restores 50/50 split and -90 deg quadrature
    assert bar_frac_3db > bar_frac_cross + 0.25
    assert abs(bar_frac_3db - 0.50) < 0.10, (
        f"Expected ~50/50 split at L=3.25 um, got bar_frac={bar_frac_3db:.3f}"
    )
    assert abs(dphi_3db_deg - (-90.0)) < 8.0, (
        f"Expected ~-90 deg quadrature at L=3.25 um, got {dphi_3db_deg:.2f} deg"
    )


def test_mmi_2x2_full_study_data_and_convergence():
    """
    Verify that the full-resolution study dataset in ``tutorials/05_guided_waves/05-07_mmi_2x2/data/``
    contains all 11 ``--full`` tasks meeting ``cpw >= 10`` and
    ``cells_across_feature >= 4``, that Kottke converges monotonically across
    L1 -> L2 -> L3 against the R=(40, 16) reference, and that the full-resolution
    splitting ratio, quadrature phase, and excess loss agree with analytical
    self-imaging theory.
    """
    import json

    mod = _load_tutorial_module()
    data_dir = mod.DATA_DIR
    expected_tasks = mod.get_task_table(quick=False)
    assert len(expected_tasks) == 12

    tasks_npz = {}
    for exp in expected_tasks:
        idx = exp["index"]
        json_path = data_dir / f"task_{idx}.json"
        npz_path = data_dir / f"task_{idx}.npz"
        assert json_path.is_file(), f"Missing {json_path}"
        assert npz_path.is_file(), f"Missing {npz_path}"

        meta = json.loads(json_path.read_text())
        assert meta["name"] == exp["name"]
        assert float(meta["cpw"]) == pytest.approx(float(exp["cpw"]))
        assert float(meta["cpw"]) >= 10.0
        assert float(meta["cells_across_feature"]) >= 4.0
        assert float(meta["cells_across_feature"]) == pytest.approx(
            float(exp["n_cells_feature"]), rel=1e-2
        )

        d = dict(np.load(npz_path, allow_pickle=False))
        assert str(d["name"]) == exp["name"]
        assert float(d["cpw"]) >= 10.0
        assert float(d["cells_across_feature"]) >= 4.0
        tasks_npz[exp["name"]] = d

    # Primary L3 run has DFT fields, movie frames, port dispersion, and 2D flux box closure
    prim = tasks_npz["kottke_L3"]
    assert "dft_E2" in prim and "dft_Ey" in prim and "flux_total_box" in prim
    w_all = prim["wavelengths"] * 1e9
    ord_idx = np.argsort(w_all)
    w_s = w_all[ord_idx]
    m_band = (w_s >= 1505.0) & (w_s <= 1595.0)
    flux_box_mean = float(np.mean(prim["flux_total_box"][ord_idx][m_band]))
    assert flux_box_mean == pytest.approx(1.0, abs=5e-3)

    # Convergence check across L1, L2, L3 against R = (40, 16) reference
    m_conv = (w_s >= 1510.0) & (w_s <= 1590.0)

    def _band_metrics(d):
        s31 = d["S31"][ord_idx][m_conv]
        s41 = d["S41"][ord_idx][m_conv]
        t31 = np.abs(s31) ** 2
        t41 = np.abs(s41) ** 2
        return t31, t41, t31 + t41

    t31_ref, t41_ref, ttot_ref = _band_metrics(tasks_npz["kottke_reference"])
    err_ttot_k, err_pair_k = [], []
    err_ttot_s, err_pair_s = [], []
    for lvl in ("L1", "L2", "L3"):
        t31_k, t41_k, ttot_k = _band_metrics(tasks_npz[f"kottke_{lvl}"])
        t31_s, t41_s, ttot_s = _band_metrics(tasks_npz[f"staircase_{lvl}"])
        err_ttot_k.append(float(np.sqrt(np.mean((ttot_k - ttot_ref) ** 2))))
        err_ttot_s.append(float(np.sqrt(np.mean((ttot_s - ttot_ref) ** 2))))
        err_pair_k.append(
            float(
                np.sqrt(0.5 * np.mean((t31_k - t31_ref) ** 2 + (t41_k - t41_ref) ** 2))
            )
        )
        err_pair_s.append(
            float(
                np.sqrt(0.5 * np.mean((t31_s - t31_ref) ** 2 + (t41_s - t41_ref) ** 2))
            )
        )

    # Monotonic Kottke convergence and < 1% relative change between L2 and L3 (and L3 vs R)
    assert err_ttot_k[2] < err_ttot_k[1] < err_ttot_k[0]
    assert err_pair_k[2] < err_pair_k[1] < err_pair_k[0]
    for k_err, s_err in zip(err_pair_k, err_pair_s):
        assert k_err < s_err
    assert err_ttot_k[2] < 0.5 * err_ttot_s[2]

    _, _, ttot_l2 = _band_metrics(tasks_npz["kottke_L2"])
    _, _, ttot_l3 = _band_metrics(tasks_npz["kottke_L3"])
    rel_change_l2_l3 = float(
        np.sqrt(np.mean((ttot_l3 - ttot_l2) ** 2)) / np.mean(ttot_l3)
    )
    rel_change_l3_ref = float(err_ttot_k[2] / np.mean(ttot_ref))
    assert rel_change_l2_l3 < 0.01
    assert rel_change_l3_ref < 0.01

    # Full-resolution length sweep vs Soldano & Pennings self-imaging theory at 1550 nm
    ana = mod.analytic_mmi_model(wavelength=mod.WL_REF)
    l_pi_um = ana["l_pi_exact"] * 1e6
    l_eff_offset_um = 0.56
    i1550 = int(np.argmin(np.abs(prim["wavelengths"] - mod.WL_REF)))

    bar_fracs = []
    for l_um in mod.L_MMI_SWEEP_UM:
        d_sw = (
            prim
            if np.isclose(l_um, mod.L_MMI * 1e6)
            else tasks_npz[f"length_{l_um:.2f}um".replace(".", "p")]
        )
        s31_0 = complex(d_sw["S31"][i1550])
        s41_0 = complex(d_sw["S41"][i1550])
        t31_0 = float(np.abs(s31_0) ** 2)
        t41_0 = float(np.abs(s41_0) ** 2)
        ttot_0 = t31_0 + t41_0
        assert 0.74 < ttot_0 <= 1.0
        bar_fracs.append(t31_0 / ttot_0)

    # Bar split fraction increases monotonically across L_mmi in [2.50, 3.80] um
    assert np.all(np.diff(bar_fracs) > 0.08)

    # At L_mmi = 3.25 um (2D EIM 3-dB length), splitting is 50/50 within 1%, EL < 0.7 dB, quadrature ~ -90 deg
    d_3db = tasks_npz["length_3p25um"]
    s31_3db = complex(d_3db["S31"][i1550])
    s41_3db = complex(d_3db["S41"][i1550])
    t31_3db = float(np.abs(s31_3db) ** 2)
    t41_3db = float(np.abs(s41_3db) ** 2)
    ttot_3db = t31_3db + t41_3db
    el_db_3db = float(-10.0 * np.log10(ttot_3db))
    imb_db_3db = float(10.0 * np.log10(t31_3db / t41_3db))
    dphi_3db = float(np.rad2deg(np.angle(s31_3db * np.conj(s41_3db))))
    r_bar_ana_3db = float(
        np.cos(0.5 * np.pi * (3.25 + l_eff_offset_um) / l_pi_um) ** 2
    )
    assert abs(t31_3db / ttot_3db - 0.50) < 0.01
    assert abs(t31_3db / ttot_3db - r_bar_ana_3db) < 0.01
    assert abs(imb_db_3db) < 0.15
    assert el_db_3db < 0.70
    assert abs(dphi_3db - (-90.0)) < 7.0

    # Mutation check: swapping bar and cross ports breaks the self-imaging slope
    assert not np.all(np.diff([1.0 - b for b in bar_fracs]) > 0.0)


def test_mmi_2x2_setup_schematic_and_nonuniform_mesh_upgrade():
    """
    Verify that the setup schematic figures (`00a_setup_geometry.png`,
    `00b_setup_mesh_and_excitation.png`, `00_setup_table.md`), the mesh-upgrade
    comparison (`04b_mesh_upgrade_comparison.png`), and the `< 5 MB` GIF exist,
    that `NonUniformMMI2x2PIC` (`kottke_nonuniform`, `task_11`) achieves lower
    cell count, lower wall time, zero MMI sidewall rounding error, and lower
    broadband RMS error against `kottke_reference` than `kottke_L3`, and include
    a mutation check on `build_mmi_2x2_graded_axes`.
    """
    import json

    mod = _load_tutorial_module()
    fig_dir = mod.FIG_DIR
    data_dir = mod.DATA_DIR

    for fname in (
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00_setup_table.md",
        "04b_mesh_upgrade_comparison.png",
        "07_field_intensity_1550.png",
        "08_field_real_ey_1550.png",
        "09_field_comparison_1550_vs_1580.png",
        "10_pulse_propagation.gif",
    ):
        fpath = fig_dir / fname
        assert fpath.is_file(), f"Missing required figure {fpath}"
        assert fpath.stat().st_size > 500, f"Empty figure {fpath}"

    gif_size_mb = (fig_dir / "10_pulse_propagation.gif").stat().st_size / (1024 * 1024)
    assert gif_size_mb < 5.0, f"GIF size {gif_size_mb:.2f} MB exceeds 5 MB limit"

    # Verify NonUniformMMI2x2PIC grid properties vs baseline kottke_L3
    structures, ports, domain = mod.build_mmi_2x2(l_mmi=mod.L_MMI)
    pic_nu = mod.NonUniformMMI2x2PIC(
        mod.WL_MIN,
        mod.WL_MAX,
        structures=structures,
        ports=ports,
        n_core=mod.N_SI,
        n_clad=mod.N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=20,
        domain=domain,
        n_pml=10,
        l_mmi=mod.L_MMI,
        n_cells_feature=8,
        min_feature=mod.GAP_MMI,
    )
    stats_nu = mod.measure_grid_resolution(
        pic_nu.grid, n_pml=10, l_mmi=mod.L_MMI, min_feature=mod.GAP_MMI, w_mmi=mod.W_MMI
    )
    assert (pic_nu.grid.Nx, pic_nu.grid.Ny, pic_nu.grid.Nz) == (425, 171, 1)
    assert stats_nu["n_cells"] == 72675
    assert stats_nu["cells_across_feature"] == pytest.approx(8.0, abs=1e-3)
    assert stats_nu["sidewall_err_nm"] == pytest.approx(0.0, abs=1e-3)
    assert stats_nu["dx_min_nm"] == pytest.approx(25.0, abs=1e-2)
    assert stats_nu["dx_max_nm"] > 35.0
    assert stats_nu["dy_min_nm"] == pytest.approx(25.0, abs=1e-2)
    assert stats_nu["dy_max_nm"] > 60.0

    # Compare task_11 (kottke_nonuniform) vs task_4 (kottke_L3) and task_6 (kottke_reference)
    meta_l3 = json.loads((data_dir / "task_4.json").read_text())
    meta_nu = json.loads((data_dir / "task_11.json").read_text())
    assert meta_nu["n_cells"] < 0.70 * meta_l3["n_cells"]
    assert meta_nu["elapsed_s"] < 0.70 * meta_l3["elapsed_s"]

    d_l3 = np.load(data_dir / "task_4.npz")
    d_ref = np.load(data_dir / "task_6.npz")
    d_nu = np.load(data_dir / "task_11.npz")
    wl = d_ref["wavelengths"] * 1e9
    ord_i = np.argsort(wl)
    w_s = wl[ord_i]
    m_conv = (w_s >= 1510.0) & (w_s <= 1590.0)

    def _pair_and_tot(d):
        t31 = np.abs(d["S31"][ord_i][m_conv]) ** 2
        t41 = np.abs(d["S41"][ord_i][m_conv]) ** 2
        return t31, t41, t31 + t41

    t31_r, t41_r, ttot_r = _pair_and_tot(d_ref)
    t31_3, t41_3, ttot_3 = _pair_and_tot(d_l3)
    t31_n, t41_n, ttot_n = _pair_and_tot(d_nu)

    rms_tot_l3 = float(np.sqrt(np.mean((ttot_3 - ttot_r) ** 2)))
    rms_tot_nu = float(np.sqrt(np.mean((ttot_n - ttot_r) ** 2)))
    rms_pair_l3 = float(np.sqrt(0.5 * np.mean((t31_3 - t31_r) ** 2 + (t41_3 - t41_r) ** 2)))
    rms_pair_nu = float(np.sqrt(0.5 * np.mean((t31_n - t31_r) ** 2 + (t41_n - t41_r) ** 2)))

    assert rms_tot_nu < rms_tot_l3
    assert rms_pair_nu < rms_pair_l3

    # Mutation check: setting n_cells_feature=2 in build_mmi_2x2_graded_axes drops cells_across_feature < 4.0
    axes_mut = mod.build_mmi_2x2_graded_axes(
        domain,
        l_mmi=mod.L_MMI,
        w_mmi=mod.W_MMI,
        w_wg=mod.W_WG,
        min_feature=mod.GAP_MMI,
        y_port=mod.Y_PORT,
        n_pml=10,
        cpw=20,
        n_cells_feature=2,
    )
    from wakis.gridFIT3D import GridFIT3D

    grid_mut = GridFIT3D(x=axes_mut[0], y=axes_mut[1], z=axes_mut[2], verbose=0)
    grid_mut.remove_edge_ghost(2)
    stats_mut = mod.measure_grid_resolution(
        grid_mut, n_pml=10, l_mmi=mod.L_MMI, min_feature=mod.GAP_MMI
    )
    assert stats_mut["cells_across_feature"] < 4.0, (
        "Mutation check failed: expected n_cells_feature=2 to have < 4 cells across gap, "
        f"got {stats_mut['cells_across_feature']}"
    )


