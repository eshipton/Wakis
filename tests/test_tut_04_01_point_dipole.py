# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""End-to-end physics, figure, and mutation tests for Tutorial 4.1 (Point Dipole in 3D Free Space)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TUT_DIR = REPO_ROOT / "tutorials" / "04_radiation_and_antennas" / "04-01_point_dipole"
DATA_DIR = TUT_DIR / "data"
FIG_DIR = TUT_DIR / "figures"


def _load_helpers():
    spec = importlib.util.spec_from_file_location("tut_04_01_helpers", TUT_DIR / "_helpers.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tut_04_01_helpers"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_setup_schematic_and_figures_exist():
    """Verify that the setup figures (00a/00b + 00_setup_table.md) and all diagnostic figures exist."""
    expected_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "01_radial_field_zones_kr1.png",
        "02_near_and_far_field_maps.png",
        "03_far_field_pattern_and_power.png",
        "04_radiation_resistance_point_limit.png",
        "05_resolution_and_cpml_study.png",
        "06_dipole_radiation.gif",
    ]
    for fname in expected_figs:
        fpath = FIG_DIR / fname
        assert fpath.exists(), f"Missing Tutorial 4.1 figure: {fpath}"
        size_bytes = fpath.stat().st_size
        assert size_bytes > 5_000, f"Figure {fname} looks suspiciously small ({size_bytes} B)"
        if fname.endswith(".gif"):
            assert size_bytes < 5 * 1024 * 1024, f"GIF {fname} exceeds 5 MB limit ({size_bytes} B)"

    table_md = FIG_DIR / "00_setup_table.md"
    assert table_md.exists() and table_md.stat().st_size > 200, f"Missing setup table: {table_md}"

    from PIL import Image

    with Image.open(FIG_DIR / "06_dipole_radiation.gif") as im:
        assert im.n_frames <= 60, f"Expected <= 60 GIF frames, got {im.n_frames}"
        assert im.size[0] <= 900, f"Expected GIF width <= 900 px, got {im.size[0]}"


def test_radial_zones_kr1_and_power_balance():
    """Validate 3D FIT radial zones (1/r^3, 1/r^2, 1/r), sin^2(theta) pattern, and 3-way power conservation."""
    d0 = np.load(DATA_DIR / "task_radial_zones_and_near2far.npz")

    # 1. Three-way radiated power conservation at f0 = 1.0 GHz within 5%
    phi_ana = float(d0["phi_ana_f0"])
    phi_box = float(d0["phi_box_f0"])
    phi_n2f = float(d0["phi_n2f_f0"])
    w_src = float(d0["w_src_f0"])
    assert np.isclose(phi_box, phi_ana, rtol=0.045)
    assert np.isclose(phi_n2f, phi_ana, rtol=0.05)
    assert np.isclose(w_src, phi_ana, rtol=0.03)

    # 2. Equatorial (theta = 90 deg) and axial (theta = 0 deg) radial profiles match Hertzian formulas
    assert float(d0["median_rel_err_eq"]) < 0.03
    assert float(d0["median_rel_err_ax"]) < 0.03

    # Verify that the three asymptotes (static 1/r^3, induction 1/r^2, radiation 1/r) cross at kr = 1
    kr_eq = np.asarray(d0["kr_eq"], dtype=float)
    assert kr_eq[0] < 0.45 and kr_eq[-1] > 3.0
    e_stat = np.asarray(d0["etheta_static"], dtype=float)
    e_ind = np.asarray(d0["etheta_induction"], dtype=float)
    e_rad = np.asarray(d0["etheta_radiation"], dtype=float)
    assert np.allclose(e_stat * (kr_eq**3), e_rad * kr_eq, rtol=1e-12)
    assert np.allclose(e_ind * (kr_eq**2), e_rad * kr_eq, rtol=1e-12)

    # Phase angle arg(-E_theta) - arg(H_phi) transitions from negative reactive (< -55 deg) in kr < 0.5
    # toward 0 deg in kr > 2.5
    ph_fit = np.asarray(d0["phase_diff_fit_deg"], dtype=float)
    assert float(ph_fit[0]) < -55.0
    assert abs(float(ph_fit[-1])) < 20.0

    # 3. Far-field pattern matches D(theta) = 1.5 * sin^2(theta) (D0 = 1.500 = 1.761 dBi, HPBW = 90 deg)
    assert np.isclose(float(d0["d0_peak"]), 1.50, rtol=0.025)
    assert np.isclose(float(d0["hpbw_deg"]), 90.0, atol=2.5)
    assert float(d0["pattern_l2_err"]) < 0.03


def test_finite_length_breakdown_and_resolution_cpml():
    """Verify R_rad = 80*pi^2*(l/lam0)^2 for small l/lam0, departure for l/lam0 -> 0.5, and convergence."""
    d1 = np.load(DATA_DIR / "task_dipole_length_and_cell_size.npz")
    d2 = np.load(DATA_DIR / "task_cpml_distance_study.npz")
    d3 = np.load(DATA_DIR / "task_resolution_and_mesh_study.npz")

    l_over_lam = np.asarray(d1["l_over_lam"], dtype=float)
    r_fit = np.asarray(d1["r_rad_fit_ohm"], dtype=float)
    r_hertz = np.asarray(d1["r_rad_hertz_ohm"], dtype=float)
    r_unif = np.asarray(d1["r_rad_unif_ohm"], dtype=float)
    d0_fit = np.asarray(d1["d0_fit"], dtype=float)

    # At small electrical length (l/lam0 <= 0.08), FIT matches both R_Hertz and R_unif within 5%
    assert np.allclose(r_fit[:2], r_hertz[:2], rtol=0.05)
    # Across the entire sweep up to l/lam0 ~ 0.5, FIT tracks the exact uniform-filament integral
    assert np.allclose(r_fit, r_unif, rtol=0.06)
    # At l/lam0 ~ 0.5, finite-length phase cancellation reduces R_rad well below 80*pi^2*(l/lam0)^2
    # and increases directivity D0 above 1.60
    assert r_fit[-1] / r_hertz[-1] < 0.90
    assert d0_fit[-1] > 1.60

    # Resolution convergence: finest uniform grid and graded mesh have significantly lower error than coarsest grid
    near_err_u = np.asarray(d3["near_err_unif"], dtype=float)
    assert near_err_u[-1] < 0.5 * near_err_u[0]
    assert float(d3["near_err_grad"]) < near_err_u[-1]

    # CPML clearance: placing CPML at d/lam0 >= 0.30 gives much smaller FluxMonitor error than d/lam0 = 0.08
    err_box = np.asarray(d2["rel_err_box"], dtype=float)
    assert err_box[-1] < 0.03
    assert err_box[0] > 2.0 * err_box[-1]


def test_analytical_helpers_and_mutation_check():
    """Verify analytical Hertzian helpers and ensure mutated formulas fail validation."""
    mod = _load_helpers()

    # 1. Exact analytical checks at kr = 1 (r = lam0 / 2pi), theta = pi/2
    res = mod.hertzian_fields_spherical(
        r_m=np.array([mod.R_KR1_M]),
        theta_rad=np.array([0.5 * np.pi]),
        freq_hz=mod.F0_HZ,
        il_phasor=1.0,
    )
    assert np.isclose(res["kr"][0], 1.0, rtol=1e-12)
    # At kr = 1, static, induction, and radiation term magnitudes are identical:
    assert np.isclose(res["Etheta_static"][0], res["Etheta_induction"][0], rtol=1e-12)
    assert np.isclose(res["Etheta_induction"][0], res["Etheta_radiation"][0], rtol=1e-12)
    # And |E_theta(kr=1)| = |j + 1 - j| * E_rad = 1.0 * E_rad!
    assert np.isclose(np.abs(res["Etheta"][0]), res["Etheta_radiation"][0], rtol=1e-12)

    # 2. Radiation resistance at l/lam0 = 0.01 vs 80*pi^2*(l/lam0)^2
    r_hertz = float(mod.hertzian_radiation_resistance(0.01))
    r_unif = float(mod.uniform_filament_radiation_metrics(0.01)["r_rad_ohm"][0])
    assert np.isclose(r_hertz, (2.0 * np.pi / 3.0) * mod.ETA0 * 1e-4, rtol=1e-12)
    assert np.isclose(r_unif, r_hertz, rtol=1e-4)

    # 3. Mutation check: dropping the -j/(kr)^3 static term or corrupting the power prefactor MUST fail
    d0 = np.load(DATA_DIR / "task_radial_zones_and_near2far.npz")
    kr_eq = np.asarray(d0["kr_eq"], dtype=float)
    eth_fit = np.asarray(d0["etheta_eq_fit_abs"], dtype=float)
    # Mutated model missing the 1/(kr)^3 static term:
    mutated_etheta = np.asarray(d0["etheta_radiation"], dtype=float) * np.sqrt(1.0 + 1.0 / (kr_eq**2))
    mutated_rel_err = np.median(np.abs(eth_fit[:6] - mutated_etheta[:6]) / eth_fit[:6])
    assert mutated_rel_err > 0.25, "Mutation check failed: dropping 1/r^3 near-field term did not trigger error"
