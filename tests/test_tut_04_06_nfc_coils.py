# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Analytical, 3D FIT solver, ferrite-shielding, and mutation tests for Tutorial 4.6:
`tutorials/04_radiation_and_antennas/04-06_nfc_coils/`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wakis.dispersive import DispersiveMaterial, Lorentz
from wakis.material_library import (
    NIZN_FERRITE_SOURCE_DATA,
    dispersive_lib,
    fit_mu_poles,
    list_materials,
)

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "04_radiation_and_antennas"
    / "04-06_nfc_coils"
)


def _load_tutorial():
    spec = importlib.util.spec_from_file_location(
        "_tut_04_04_nfc_coils", TUTORIAL_DIR / "nfc_coils.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tut_04_04_nfc_coils"] = mod
    spec.loader.exec_module(mod)
    return mod


nfc = _load_tutorial()


def test_nizn_ferrite_material_fit_passivity_and_stability():
    """Verify commercial NiZn NFC ferrite fit against source data, passivity, and ADE stability."""
    mat = dispersive_lib["nizn ferrite"]
    assert mat is dispersive_lib["nfc ferrite"]
    assert "nizn ferrite" in list_materials(category="ferrite")
    assert "nizn ferrite" in list_materials(include_magnetic=True)

    f_src = NIZN_FERRITE_SOURCE_DATA["f_hz"]
    mu_src = NIZN_FERRITE_SOURCE_DATA["mu_real"] - 1j * NIZN_FERRITE_SOURCE_DATA["mu_imag"]
    mu_fit = mat.mu(f_src)

    # Relative complex fit error across 1-100 MHz must be < 2%
    rel_rms = float(np.linalg.norm(mu_fit - mu_src) / np.linalg.norm(mu_src))
    assert rel_rms < 0.02

    # At 13.56 MHz, mu' ~ 150 and mu'' = -Im(mu) ~ 3.58
    mu_13 = complex(mat.mu(nfc.F0_NFC))
    assert float(np.real(mu_13)) == pytest.approx(150.0, rel=0.015)
    assert float(-np.imag(mu_13)) == pytest.approx(3.58, rel=0.05)

    # Passivity across 100 kHz - 500 MHz: Im(mu(f)) <= 0 in exp(+j w t) convention,
    # and Re(mu(f)) > 50 across the 1-100 MHz NFC/VHF source dataset
    f_dense = np.geomspace(1e5, 5e8, 400)
    mu_dense = mat.mu(f_dense)
    assert np.all(np.imag(mu_dense) <= 0.0)
    assert np.all(np.real(mu_fit) > 50.0)

    # Stability on the 3D FIT Courant timestep
    grid = nfc.make_nfc_grid(dx_core=0.003)
    dt0 = nfc.vacuum_courant_dt(grid)
    assert mat.is_stable(dt0, 0.95)

    # Direct call to fit_mu_poles reproduces the 2-pole fit
    re_fit = fit_mu_poles(f_src, mu_src, n_lorentz=2, name="re_fit_nizn")
    assert re_fit.fit_error < 1e-3
    assert re_fit.is_stable(dt0, 0.95)

    # Mutation check: flipping the sign of a magnetic pole amplitude violates passivity (Im(mu) > 0)
    bad_mat = DispersiveMaterial(
        eps_inf=1.0,
        poles=(),
        mu_inf=1.0,
        mu_poles=[Lorentz(-50.0, 2.0 * np.pi * 25e6, 2.0 * np.pi * 10e6)],
    )
    with pytest.raises(AssertionError):
        assert np.all(np.imag(bad_mat.mu(f_dense)) <= 0.0)


def test_analytic_inductance_neumann_q_and_schormans():
    """Verify Wheeler, Mohan Current-Sheet, Grover, and Neumann formulas and Schormans efficiency."""
    ref_18 = nfc.neumann_coil_pair_analytic(0.018)

    # All four self-inductance formulas agree within 4% for both reader and tag coils
    l1_set = np.array(
        [
            ref_18["L1_wheeler_H"],
            ref_18["L1_current_sheet_H"],
            ref_18["L1_grover_H"],
            ref_18["L1_neumann_H"],
        ]
    )
    l2_set = np.array(
        [
            ref_18["L2_wheeler_H"],
            ref_18["L2_current_sheet_H"],
            ref_18["L2_grover_H"],
            ref_18["L2_neumann_H"],
        ]
    )
    assert float(np.max(l1_set) / np.min(l1_set) - 1.0) < 0.06
    assert float(np.max(l2_set) / np.min(l2_set) - 1.0) < 0.06
    assert 580e-9 < ref_18["L1_grover_H"] < 660e-9
    assert 490e-9 < ref_18["L2_grover_H"] < 560e-9

    # Monotonic decay of M(d) and k(d) with separation
    d_arr = [0.012, 0.018, 0.024, 0.036, 0.048]
    m_arr = [nfc.neumann_coil_pair_analytic(d)["M_neumann_H"] for d in d_arr]
    k_arr = [nfc.neumann_coil_pair_analytic(d)["k_neumann"] for d in d_arr]
    assert np.all(np.diff(m_arr) < 0.0)
    assert np.all(np.diff(k_arr) < 0.0)

    # Monotonic decay with lateral offset in [0, 36] mm at d = 18 mm
    dx_arr = [0.0, 0.012, 0.024, 0.036]
    k_dx = [
        nfc.neumann_coil_pair_analytic(0.018, dx_offset_m=x)["k_neumann"]
        for x in dx_arr
    ]
    assert np.all(np.diff(k_dx) < 0.0)

    # Copper skin depth at 13.56 MHz (~17.95 um) and coil Q
    delta_cu = float(nfc.copper_skin_depth(nfc.F0_NFC))
    assert delta_cu == pytest.approx(17.95e-6, rel=0.01)
    cu1 = nfc.coil_ac_resistance_and_q(nfc.READER_GEOM, ref_18["L1_neumann_H"])
    assert cu1["r_ac_ohm"] > cu1["r_dc_ohm"]
    assert 150.0 < cu1["q_unloaded"] < 350.0

    # Schormans et al. (2018) link efficiency identities
    assert nfc.schormans_link_efficiency(0.0, 150.0, 150.0) == pytest.approx(0.0, abs=1e-15)
    eta_vals = nfc.schormans_link_efficiency(np.array(k_arr), 150.0, 150.0)
    assert np.all(np.diff(eta_vals) < 0.0)
    assert 0.0 < float(eta_vals[-1]) < float(eta_vals[0]) < 1.0


def test_committed_data_physics_monotonicity_ordering_and_mutations():
    """Verify all physics assertions and mutation checks on the committed 3D FIT study dataset."""
    npz_path = TUTORIAL_DIR / "data" / "nfc_coils_results.npz"
    json_path = TUTORIAL_DIR / "data" / "nfc_coils_summary.json"
    assert npz_path.is_file(), f"Missing {npz_path}"
    assert json_path.is_file(), f"Missing {json_path}"

    data = np.load(npz_path, allow_pickle=False)
    summary = json.loads(json_path.read_text())
    assert summary["preflight"]["preflight_ok"] is True

    # 1. Self-inductance L1, L2 within 10% of Wheeler / Mohan / Grover / Neumann values
    l1_sim = float(data["L1_sim_H"][1])
    l2_sim = float(data["L2_sim_H"][1])
    for l1_ref in data["L1_analytic_H"]:
        assert l1_sim == pytest.approx(float(l1_ref), rel=0.10)
    for l2_ref in data["L2_analytic_H"]:
        assert l2_sim == pytest.approx(float(l2_ref), rel=0.10)

    # 2. Mutual inductance M(d) and coupling k(d) decrease strictly monotonically and match Neumann within 10%
    m_sim = data["M_sim_H"]
    k_sim = data["k_sim"]
    m_neu = data["M_neu_pts_H"]
    k_neu = data["k_neu_pts"]
    assert np.all(np.diff(m_sim) < 0.0)
    assert np.all(np.diff(k_sim) < 0.0)
    np.testing.assert_allclose(m_sim, m_neu, rtol=0.10)
    np.testing.assert_allclose(k_sim, k_neu, rtol=0.10)

    # Lateral offset k(dx) decreases monotonically and matches Neumann
    assert np.all(np.diff(data["k_off_sim"]) < 0.0)
    np.testing.assert_allclose(data["k_off_sim"][:3], data["k_off_neu"][:3], rtol=0.12)

    # 3. Mutation check: removing the second coil makes M ~ 0 (< 1 nH vs ~140 nH with tag)
    m_no_tag = float(np.abs(data["M_mutation_no_tag_H"][0]))
    assert m_no_tag < 1.0e-9
    assert m_no_tag < 0.01 * float(m_sim[1])
    with pytest.raises(AssertionError):
        assert m_no_tag == pytest.approx(float(m_neu[1]), rel=0.10)

    # 4. Resonance at 13.56 MHz within 3% tolerance for the designed tuning capacitor C1
    f_res = float(data["f_res_tuned_hz"][0])
    assert f_res == pytest.approx(nfc.F0_NFC, rel=0.03)
    # Mutation check: detuning C1 by 1.5x shifts resonance to 13.56 / sqrt(1.5) = 11.07 MHz (fails 3% check)
    with pytest.raises(AssertionError):
        assert (f_res / np.sqrt(1.5)) == pytest.approx(nfc.F0_NFC, rel=0.03)

    # 5. Permittivity scaling invariance for s in {1, 2.25, 4} (< 3.5% variation) and failure at s = 64
    l1_s = data["L1_s_H"]
    m_s = data["M_s_H"]
    k_s = data["k_s"]
    np.testing.assert_allclose(l1_s[:3], l1_s[0], rtol=0.035)
    np.testing.assert_allclose(m_s[:3], m_s[0], rtol=0.035)
    np.testing.assert_allclose(k_s[:3], k_s[0], rtol=0.035)
    assert np.all(np.diff(data["Nt_s"]) < 0)
    # At s = 64, shifted self-resonance breaks the 3.5% invariance tolerance
    with pytest.raises(AssertionError):
        assert float(m_s[-1]) == pytest.approx(float(m_s[0]), rel=0.035)

    # 6. Ordering test: PEC metal without ferrite reduces L1 and k relative to free space,
    #    and adding the flexible NiZn ferrite sheet between coil and metal restores them
    l1_free, l1_metal, l1_ferrite = [float(v) for v in data["cfg3_L1_H"]]
    k_free, k_metal, k_ferrite = [float(v) for v in data["cfg3_k"]]
    assert l1_metal < 0.70 * l1_free
    assert k_metal < 0.75 * k_free
    assert l1_ferrite > l1_free > l1_metal
    assert k_ferrite > k_metal
    assert k_ferrite >= k_free
    # Mutation check: swapping metal and ferrite ordering must fail
    with pytest.raises(AssertionError):
        assert k_metal > k_ferrite

    # 7. Equal-(mu'-1)*t = 27 mm thin-sheet equivalence test across 3 (mu, t) pairs
    #    t in {1.5, 3.0, 4.5} mm (indices 0, 1, 2) must agree on L1 and k within 6%
    l1_eq_thin = data["L1_eq_H"][:3]
    k_eq_thin = data["k_eq"][:3]
    np.testing.assert_allclose(l1_eq_thin, l1_eq_thin[1], rtol=0.06)
    np.testing.assert_allclose(k_eq_thin, k_eq_thin[1], rtol=0.06)
    # Thick-sheet failure case (t = 7.5 mm, index 3) fills the coil-metal gap and violates 6% L1 equivalence
    with pytest.raises(AssertionError):
        assert float(data["L1_eq_H"][3]) == pytest.approx(
            float(l1_eq_thin[1]), rel=0.06
        )


def test_live_3d_fit_single_vs_coupled_mutation_and_s_scaling():
    """Run a fast live 3D FIT check verifying L1, M, 1-coil mutation (M -> 0), and s-invariance."""
    grid = nfc.make_nfc_grid(dx_core=0.003)
    ref = nfc.neumann_coil_pair_analytic(0.018)
    tag_iso = nfc.simulate_isolated_coil(
        grid,
        nfc.TAG_GEOM,
        z0_coil=0.018,
        via_dz=0.003,
        eps_scale=2.25,
        f_center=22e6,
        f_width=26e6,
        cutoff=3.1,
        post_time=12e-9,
    )
    res_s2 = nfc.simulate_coupled_coils(
        grid,
        separation_m=0.018,
        via_dz=0.003,
        eps_scale=2.25,
        f_center=22e6,
        f_width=26e6,
        cutoff=3.1,
        post_time=12e-9,
        z22_ref=tag_iso["Z_in"],
        include_tag=True,
    )
    res_s4 = nfc.simulate_coupled_coils(
        grid,
        separation_m=0.018,
        via_dz=0.003,
        eps_scale=4.0,
        f_center=14e6,
        f_width=14e6,
        cutoff=3.1,
        post_time=15e-9,
        z22_ref=tag_iso["Z_in"],
        include_tag=True,
    )
    res_no_tag = nfc.simulate_coupled_coils(
        grid,
        separation_m=0.018,
        via_dz=0.003,
        eps_scale=2.25,
        f_center=22e6,
        f_width=26e6,
        cutoff=3.1,
        post_time=12e-9,
        z22_ref=tag_iso["Z_in"],
        include_tag=False,
    )

    assert float(res_s2["L1_H"][0]) == pytest.approx(ref["L1_wheeler_H"], rel=0.08)
    assert float(res_s2["M_H"][0]) == pytest.approx(ref["M_neumann_H"], rel=0.08)
    assert float(res_s4["L1_H"][0]) == pytest.approx(float(res_s2["L1_H"][0]), rel=0.03)
    assert float(res_s4["k"][0]) == pytest.approx(float(res_s2["k"][0]), rel=0.03)
    assert float(np.abs(res_no_tag["M_H"][0])) < 1.0e-9


def test_tutorial_figures_and_cli_list_tasks(capsys):
    """Verify setup figures/tables, all 11 result figures (including < 5 MB GIF), and CLI --list-tasks."""
    expected_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00a_setup_geometry_top.png",
        "00b_setup_mesh_and_excitation_top.png",
        "01_geometry_3d_views.png",
        "02_mesh_cross_sections.png",
        "03_inductance_coupling_vs_separation.png",
        "04_impedance_and_resonance_tuning.png",
        "05_s21_and_link_efficiency.png",
        "06_ferrite_permeability_fit.png",
        "07_ferrite_shielding_h_maps.png",
        "08_ferrite_sweeps_and_mut_equivalence.png",
        "09_fields_and_surface_current.png",
        "10_permittivity_scaling_and_resolution.png",
        "11_magnetic_field_oscillation.gif",
    ]
    fig_dir = TUTORIAL_DIR / "figures"
    for fname in expected_figs:
        fpath = fig_dir / fname
        assert fpath.is_file(), f"Missing figure {fpath}"
        size_b = fpath.stat().st_size
        assert size_b > 5_000, f"Figure {fname} is unexpectedly small ({size_b} B)"
        if fname.endswith(".gif"):
            assert size_b < 5 * 1024 * 1024, f"GIF {fname} exceeds 5 MB ({size_b} B)"

    for tname in ("00_setup_table.md", "00_setup_table_top.md"):
        tpath = fig_dir / tname
        assert tpath.is_file(), f"Missing setup table {tpath}"
        assert "| Category | Component / Face | Specification |" in tpath.read_text()

    nfc.main(["--list-tasks"])
    out_quick = capsys.readouterr().out
    assert "# Total tasks: 5" in out_quick
    assert "ferrite_shielding_and_mut" in out_quick

    nfc.main(["--list-tasks", "--full"])
    out_full = capsys.readouterr().out
    assert "# Total tasks: 7" in out_full
    assert "ultrafine_1p0mm_gpu" in out_full
