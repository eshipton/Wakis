# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Regression and physics verification tests for Tutorial 4.8 (Magnetic antennas on and near a ground plane).

Covers:
0. Magnetic Lorentz and Debye ADE dispersion (`wakis.dispersive`), Heaviside magnetic
   conductivity and hesitivity invariance `h_m`, exact loss identity
   `f(mu) = mu''/|mu_r - 1|^2 = omega*mu0/h_m`, 1D FIT slab R/T/A verification,
   3D FIT rod equal-`h_m` efficiency invariance, and time-centered leapfrog energy
   conservation `W_src = P_rad + P_loss` with failing mutation checks.
1. Four-orientation PEC image theory (HMD, VMD, VED, HED) in 3D FIT vs exact
   spherical-Bessel half-space radiation resistance ratios.
2. Free-space to half-space volume convention (`Vol = 2*V_phys` on PEC vs `V_phys`
   in free space) and height-lift transition `Vol_eff(h) = F_HMD(k0*h)*V_phys`.
3. Equal-`h_m` Lorentz (4 poles) and Debye (3 poles) efficiency collapse vs
   legacy Rumsey-Weeks breakdown.
4. Horizontal permeable rod on PEC vs vertical dielectric monopole on PEC and
   horizontal dielectric rod on PMC duality, conformal permeable bar vs metal
   strip `> 10^2x` radiation resistance and EBWP, 3-resolution FIT convergence,
   and exact agreement between `data/summary.json` and the Quantitative Summary table.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import mu_0 as MU0

from wakis.dispersive import Debye, DispersiveMaterial, Lorentz

TUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "04_radiation_and_antennas"
    / "04-08_magnetic_antenna_ground"
)
_SPEC = importlib.util.spec_from_file_location("tut_04_08_helpers", TUT_DIR / "_helpers.py")
assert _SPEC is not None and _SPEC.loader is not None
h = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(h)


def _check_table_matches_summary(py_text: str, summary: dict) -> bool:
    """Return True if all formatted numbers from `summary` appear in the Quantitative Summary table."""
    sec = py_text.split("## 4. Quantitative Summary", maxsplit=1)[1].split(
        "## 5. Limits & Failure Modes", maxsplit=1
    )[0]
    expected_tokens = [
        f"{summary['c_yousefi_exact_ohm']:.2f}",
        f"{summary['c_yousefi_120pi_ohm']:.2f}",
        f"{summary['czn_hm_ohm_m']:.1f}",
        f"{summary['slab_lor_max_err']:.5f}",
        f"{summary['slab_deb_max_err']:.5f}",
        f"{summary['rod_energy_balance_rel_err']*100:.5f}\\%",
        f"{summary['rod_mutated_balance_rel_err']*100:.2f}\\%",
        f"{summary['equal_hm_pole_a_eff_db']:.2f}",
        f"{summary['equal_hm_pole_b_eff_db']:.2f}",
        f"{summary['equal_hm_diff_db']:.2f}",
        f"{summary['mutated_hm_loss_ratio_factor']:.4f}",
        f"{summary['hmd_low_h_fit']:.4f}",
        f"{summary['hmd_low_h_ana']:.4f}",
        f"{summary['hmd_low_h_rel_err_pct']:.2f}\\%",
        f"{summary['ved_low_h_fit']:.4f}",
        f"{summary['ved_low_h_ana']:.4f}",
        f"{summary['vmd_low_h_fit']:.4f}",
        f"{summary['vmd_low_h_ana']:.4f}",
        f"{summary['hed_low_h_fit']:.4f}",
        f"{summary['hed_low_h_ana']:.4f}",
        f"{summary['vol_half_pec_eff']*100:.2f}\\%",
        f"{summary['vol_full_free_eff']*100:.2f}\\%",
        f"{summary['vol_ana_full_eff']*100:.2f}\\%",
        f"{summary['vol_phys_free_eff']*100:.2f}\\%",
        f"{summary['vol_ana_phys_eff']*100:.2f}\\%",
        f"{summary['ratio_prad_full_to_half']:.4f}",
        f"{summary['ratio_ploss_full_to_half']:.4f}",
        f"{summary['lor_family_spread_db']:.3f}",
        f"{summary['deb_family_spread_db']:.3f}",
        f"{summary['lor_eta_fill_f0']:.4f}",
        f"{summary['lor_eff_pol_f0_db']:.2f}",
        f"{summary['lor_eff_tot_f0_db']:.2f}",
        f"{summary['lor_eff_ana_f0_db']:.2f}",
        f"{summary['rod_to_loop_prad_ratio_f0']:.1f}",
        f"{summary['rod_to_loop_moment_ratio_f0']:.2f}",
        f"{summary['rod_only_to_loop_moment_ratio_f0']:.2f}",
        f"{summary['loop_in_phase_prad_boost_db']:.2f}",
        f"{summary['loop_in_phase_eff_boost_db']:.2f}",
        f"{summary['lor_eff_tot_minus_pol_f0_db']:.2f}",
        f"{summary['lor_max_excess_over_bound_db']:.2f}",
        f"{summary['deb_max_excess_over_bound_db']:.2f}",
        f"{summary['dual_300_mag_db']:.2f}",
        f"{summary['dual_300_pmc_db']:.2f}",
        f"{summary['dual_300_diel_db']:.2f}",
        f"{summary['dual_80_mag_db']:.2f}",
        f"{summary['dual_80_pmc_db']:.2f}",
        f"{summary['dual_80_diel_db']:.2f}",
        f"{summary['dual_20_mag_db']:.2f}",
        f"{summary['dual_20_pmc_db']:.2f}",
        f"{summary['dual_20_diel_db']:.2f}",
        f"{summary['dual_pmc_max_diff_f0_db']:.3f}",
        f"{summary['dual_pmc_coarse_max_diff_f0_db']:.3f}",
        f"{summary['fit_mbar_to_strip_ratio']:.1f}",
        f"{summary['ebwp_min_advantage_db']:.2f}",
        f"{summary['ebwp_max_advantage_db']:.2f}",
        f"{summary['inf_gnd_r_ratio']:.3f}",
        f"{summary['inf_gnd_zenith_dir']:.3f}",
        f"{summary['finite_gnd_r_ratio_030']:.3f}",
        f"{summary['finite_gnd_r_ratio_110']:.3f}",
        f"{summary['finite_gnd_zenith_dir_030']:.3f}",
        f"{summary['finite_gnd_zenith_dir_110']:.3f}",
        f"{summary['conv_eff_coarse']:.5f}",
        f"{summary['conv_eff_medium']:.5f}",
        f"{summary['conv_eff_fine']:.5f}",
        f"{summary['conv_p_obs']:.2f}",
        f"{summary['conv_eff_rich']:.5f}",
        f"{summary['conv_coarse_rel_err_pct']:.2f}\\%",
        f"{summary['conv_medium_rel_err_pct']:.2f}\\%",
        f"{summary['conv_fine_rel_err_pct']:.2f}\\%",
    ]
    return all(tok in sec for tok in expected_tokens)


def test_hesitivity_and_loss_function_identities_with_mutations(monkeypatch):
    """Verify 6*pi*eta0 constant, Lorentz/Debye hesitivity h_m, f(mu) = omega*mu0/h_m, and code-path mutations."""
    # Exact wave-impedance constant 6*pi*eta0 = 7101.20 Ohm vs 120*pi form 720*pi^2 = 7106.12 Ohm
    assert np.isclose(h.C_YOUSEFI_EXACT, 6.0 * np.pi * h.ETA0, rtol=1e-15)
    assert np.isclose(h.C_YOUSEFI_120PI, 720.0 * (np.pi**2), rtol=1e-15)

    hm_czn = h.hesitivity_lorentz(chi_dc=39.0, f0_hz=1.5e9, alpha=4.0 / 3.0)
    assert np.isclose(hm_czn, 3.46423114e5, rtol=1e-6)

    hm_deb = h.hesitivity_debye(chi_dc=75.0, f0_hz=1.6e9)
    assert np.isclose(hm_deb, 9.47482050e5, rtol=1e-6)

    freqs = np.linspace(0.05e9, 3.0e9, 80)
    omegas = 2.0 * np.pi * freqs
    w0 = 2.0 * np.pi * 1.5e9
    mat_lor = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[Lorentz(delta_eps=39.0, omega_0=w0, gamma=(4.0 / 3.0) * w0)],
    )
    mat_deb = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[Debye(delta_eps=75.0, tau=1.0 / (2.0 * np.pi * 1.6e9))],
    )

    mu_lor = mat_lor.mu(freqs)
    mu_deb = mat_deb.mu(freqs)
    f_lor = h.magnetic_loss_function(mu_lor)
    f_deb = h.magnetic_loss_function(mu_deb)
    f_lor_exact = omegas * MU0 / hm_czn
    f_deb_exact = omegas * MU0 / hm_deb

    assert np.max(np.abs(f_lor - f_lor_exact) / f_lor_exact) < 1e-14
    assert np.max(np.abs(f_deb - f_deb_exact) / f_deb_exact) < 1e-14

    # Verify closed_form_md_efficiency against direct first-principles P_rad / (P_rad + P_loss)
    vol = 5.0e-5
    k0 = omegas / h.C0
    i_m_l = omegas * MU0 * np.abs(mu_lor - 1.0) * vol
    p_rad_exact = (k0**2) * (i_m_l**2) / (12.0 * np.pi * h.ETA0)
    p_loss_exact = 0.5 * omegas * MU0 * (-np.imag(mu_lor)) * vol
    e_direct = p_rad_exact / (p_rad_exact + p_loss_exact)
    e_closed = h.closed_form_md_efficiency(freqs, hm_czn, vol, use_120pi=False)
    assert np.allclose(e_closed, e_direct, rtol=1e-14)

    # Code-path mutation: mutating C_YOUSEFI_EXACT in _helpers to (ETA0^2)/20 (7096.29 Ohm) must fail exact match
    monkeypatch.setattr(h, "C_YOUSEFI_EXACT", float((h.ETA0**2) / 20.0))
    e_mutated = h.closed_form_md_efficiency(freqs, hm_czn, vol, use_120pi=False)
    assert not np.allclose(e_mutated, e_direct, rtol=1e-4)
    monkeypatch.undo()

    # Peak conductivity of Lorentz occurs at f = f0 and equals hm_czn
    sigma_m_f0 = float(w0 * MU0 * (-np.imag(mat_lor.mu(1.5e9))))
    assert np.isclose(sigma_m_f0, hm_czn, rtol=1e-12)


def test_schematic_magnetic_dispersive_material_and_setup_table():
    """Verify wakis.schematic._describe_material and 00_setup_table.md report mu_r = 40 and converged 4 mm grid."""
    from wakis import schematic

    w0 = 2.0 * np.pi * 1.5e9
    mat_lor = DispersiveMaterial(
        mu_inf=1.0,
        mu_poles=[Lorentz(delta_eps=39.0, omega_0=w0, gamma=(4.0 / 3.0) * w0)],
        name="CZN Lorentz laminate",
    )
    desc = schematic._describe_material(mat_lor, "Permeable rod")
    assert np.isclose(desc["mu_r"], 40.0, rtol=1e-12)
    assert "μ_s = 40.00" in desc["label"]

    table_md = (TUT_DIR / "figures" / "00_setup_table.md").read_text(encoding="utf-8")
    assert "μ_r = 40.00" in table_md
    assert "n_max = 6.32" in table_md
    assert "Δ_min = 4\u2009mm" in table_md
    assert "42.3 cells/λ" in table_md
    assert "6.0 cells across feature (24\u2009mm)" in table_md


def test_magnetic_ade_1d_slab_and_3d_rod_invariance_quick():
    """Run Part 0 in quick mode to verify 1D magnetic ADE slab, 3D energy balance, and equal-hm invariance."""
    res = h.run_task_00_ade_and_hesitivity(quick=True)

    # 1. 1D magnetic ADE slab reflection/transmission & volumetric absorption
    assert float(res["slab_lor_max_err"]) < 0.015
    assert float(res["slab_deb_max_err"]) < 0.015
    assert np.max(np.abs(res["lor_a_ade"] - res["lor_a_ana"])) < 0.020
    assert np.max(np.abs(res["deb_a_ade"] - res["deb_a_ana"])) < 0.020

    # 2. Exact discrete leapfrog power balance W_src = P_rad + P_loss vs uncentered mutation
    bal_err = float(res["balance_rel_err_a"][1])
    mut_bal_err = float(res["mutated_balance_rel_err_a"][1])
    assert bal_err < 3e-3
    # Mutation check: shifting source/ADE time centering by dt/2 must fail energy balance (> 5%)
    assert mut_bal_err > 0.05

    # 3. Equal-hm invariance (Pole A vs Pole B) vs mutated damping (Pole C, alpha -> 2*alpha)
    diff_ab_db = float(res["equal_hm_diff_db"][1])
    diff_ac_db = float(np.abs(res["eff_pole_a_db"][1] - res["eff_pole_c_mut_db"][1]))
    loss_ratio_factor = float(res["mutated_hm_loss_ratio_factor"][1])
    assert diff_ab_db < 0.25
    # Mutation check: halving h_m must change efficiency by > 2 dB and double (1/e - 1)
    assert diff_ac_db > 2.0
    assert np.isclose(loss_ratio_factor, 2.0, rtol=0.02)


def test_pec_image_theory_four_orientations_and_mutations(monkeypatch):
    """Verify 4-orientation PEC image theory closed forms, sum rules, 3D FIT data, and code-path mutations."""
    h_zero = np.array([0.0])
    assert np.isclose(h.pec_image_resistance_ratio(h_zero, "HMD")[0], 2.0, atol=1e-12)
    assert np.isclose(h.pec_image_resistance_ratio(h_zero, "VED")[0], 2.0, atol=1e-12)
    assert np.isclose(h.pec_image_resistance_ratio(h_zero, "VMD")[0], 0.0, atol=1e-12)
    assert np.isclose(h.pec_image_resistance_ratio(h_zero, "HED")[0], 0.0, atol=1e-12)

    # Exact complementary sum rules: R_HMD(h) + R_HED(h) == 2 and R_VED(h) + R_VMD(h) == 2
    h_grid = np.linspace(0.0, 0.8, 50)
    assert np.allclose(
        h.pec_image_resistance_ratio(h_grid, "HMD") + h.pec_image_resistance_ratio(h_grid, "HED"),
        2.0,
        atol=1e-12,
    )
    assert np.allclose(
        h.pec_image_resistance_ratio(h_grid, "VED") + h.pec_image_resistance_ratio(h_grid, "VMD"),
        2.0,
        atol=1e-12,
    )

    d1 = np.load(TUT_DIR / "data" / "part1_image_theory.npz")
    assert np.max(np.abs(d1["r_hmd_fit"] - h.pec_image_resistance_ratio(d1["h_mid_lam"], "HMD"))) < 0.04
    assert np.max(np.abs(d1["r_ved_fit"] - h.pec_image_resistance_ratio(d1["h_mid_lam"], "VED"))) < 0.04
    assert np.max(np.abs(d1["r_vmd_fit"] - h.pec_image_resistance_ratio(d1["h_node_lam"], "VMD"))) < 0.04
    assert np.max(np.abs(d1["r_hed_fit"] - h.pec_image_resistance_ratio(d1["h_node_lam"], "HED"))) < 0.04

    # Code-path mutation 1: flipping the HMD image sign in _helpers._PEC_IMAGE_SIGN must fail the FIT match
    monkeypatch.setitem(h._PEC_IMAGE_SIGN, "HMD", -1.0)
    assert not (np.max(np.abs(d1["r_hmd_fit"] - h.pec_image_resistance_ratio(d1["h_mid_lam"], "HMD"))) < 0.04)
    monkeypatch.undo()

    # Live 3D FIT HMD test with >= 3*dx clearance between Huygens box (half-span 4*dx, z_max 8*dx)
    # and inner CPML boundary (7*dx in x/y, 11*dx in z on a 22-cell grid with n_pml=4)
    from wakis import GridFIT3D, SolverFIT3D
    from wakis.near2far import Near2FarMonitor
    from wakis.sources import GaussianSource, MagneticCurrentSource

    f0 = 1.0e9
    lam0 = h.C0 / f0
    dx = 0.015
    x = np.arange(-11, 12, dtype=float) * dx
    z = np.arange(0, 16, dtype=float) * dx
    grid = GridFIT3D(x[0], x[-1], x[0], x[-1], z[0], z[-1], x=x, y=x, z=z, verbose=0)
    pulse = GaussianSource(frequency=f0, fwidth=0.45 * f0)
    sf = complex(pulse.fourier_transform(f0))
    p_free_mag = float(((2.0 * np.pi / lam0) ** 2) / (6.0 * np.pi * h.ETA0) * (abs(sf) ** 2))
    n_steps = int(np.ceil((pulse.end_time + 1.8 / f0) / (0.5 * dx / (np.sqrt(3.0) * h.C0))))

    def _run_live_hmd(bc_z_low: str, flip_top_normal: bool = False) -> float:
        solver = SolverFIT3D(grid, bc_low=["cpml", "cpml", bc_z_low], bc_high=["cpml"] * 3, n_pml=4, verbose=0)
        src = MagneticCurrentSource(pulse, component="x", center=(0.0, 0.0, 1.5 * dx), amplitude=1.0)
        n2f = Near2FarMonitor(
            center=(0.0, 0.0, 4.0 * dx), size=(8.0 * dx, 8.0 * dx, 8.0 * dx), frequencies=[f0], ground_plane="-z"
        )
        if flip_top_normal:
            # Code-path mutation 2: flip the outward surface normal sign on the +z Huygens face
            n2f.faces = [(ax, pos, -sgn if (ax == 2 and sgn > 0) else sgn) for (ax, pos, sgn) in n2f.faces]
        for step in range(n_steps):
            t = step * solver.dt
            src.update(solver, t)
            solver.one_step()
            n2f.update(solver, t)
        return float(n2f.get_flux()[0]) / p_free_mag

    r_hmd_pec_live = _run_live_hmd("pec", flip_top_normal=False)
    r_hmd_pmc_live = _run_live_hmd("pmc", flip_top_normal=False)
    r_hmd_mutated_normal = _run_live_hmd("pec", flip_top_normal=True)
    assert r_hmd_pec_live > 1.70
    assert r_hmd_pmc_live < 0.25
    # Mutating the Near2FarMonitor +z face outward normal sign must fail the constructive doubling check
    assert not (r_hmd_mutated_normal > 1.70)


def test_volume_convention_and_equal_hm_collapse(monkeypatch):
    """Verify half-space vs free-space volume convention, Cauchy-Schwarz consistency, passband bound, and equal-hm collapse."""
    d2 = np.load(TUT_DIR / "data" / "part2_volume_convention.npz")
    d3 = np.load(TUT_DIR / "data" / "part3_equal_hm.npz")
    d5 = np.load(TUT_DIR / "data" / "part5_ebwp_and_convergence.npz")
    idx_mid = len(d2["freqs_ghz"]) // 2

    # Verify Part 2, Part 3, and Part 6 use the exact same physical rod geometry (72 mm x 24 mm x 24 mm)
    assert np.isclose(float(d2["vol_phys_m3"]), 0.072 * 0.024 * 0.024, rtol=1e-12)
    assert np.isclose(float(d2["vol_full_m3"]), 2.0 * float(d2["vol_phys_m3"]), rtol=1e-12)

    # Half-rod on PEC equals full mirrored rod in free space within 1.5% (measured diff 0.38%)
    eff_half_pec = float(d2["eff_half_pec"][idx_mid])
    eff_full_free = float(d2["eff_full_free"][idx_mid])
    eff_ana_full = float(d2["eff_ana_full"][idx_mid])
    eff_phys_free = float(d2["eff_phys_free"][idx_mid])
    eff_ana_phys = float(d2["eff_ana_phys"][idx_mid])

    assert np.isclose(eff_half_pec, eff_full_free, rtol=0.015)
    assert np.isclose(float(d2["ratio_prad_full_to_half"][idx_mid]), 2.0, rtol=0.015)
    assert np.isclose(float(d2["ratio_ploss_full_to_half"][idx_mid]), 2.0, rtol=0.01)

    # Discretization error on the Medium (4 mm, 6 cells) headline grid is 3.65% vs Richardson:
    assert np.isclose(eff_half_pec, float(d5["conv_eff"][1]), rtol=1e-12)
    assert np.isclose(eff_half_pec, float(d5["conv_eff_rich"]), rtol=0.04)

    # Reconcile FIT efficiency with the uniform-field upper bound via Cauchy-Schwarz eta_fill = 0.8129:
    # 1. Both half-rod on PEC and physical rod in free space lie strictly below their uniform-field bounds
    assert eff_half_pec < eff_ana_full
    assert eff_phys_free < eff_ana_phys
    assert np.isclose(eff_half_pec, eff_ana_full, rtol=0.11)
    # 2. Algebraic identity / consistency check: `lor_eff_pol_db` equals `closed_form_md_efficiency(Vol * eta_fill)`
    #    algebraically (up to discrete w_tilde, mu_d vs continuous omega, mu, within 0.05 dB).
    eta_fill_f0 = float(d3["lor_eta_fill"][2, idx_mid])
    e_cs_db = float(10.0 * np.log10(h.closed_form_md_efficiency(0.28e9, h.CZN_HM_OHM_M, float(d2["vol_full_m3"]) * eta_fill_f0)))
    e_pol_dft_db = float(d3["lor_eff_pol_db"][2, idx_mid])
    assert np.isclose(e_pol_dft_db, e_cs_db, atol=0.05)

    # Mutation check: using V_phys instead of 2*V_phys for the PEC half-rod must fail (> 35% error)
    wrong_vol_rel_err = abs(eff_half_pec - eff_ana_phys) / eff_half_pec
    assert wrong_vol_rel_err > 0.35

    # Equal-hm Lorentz (4 poles) and Debye (3 poles) collapse
    assert float(d3["lor_spread_db"]) < 0.20
    assert float(d3["deb_spread_db"]) < 0.20

    # Upper-bound checks:
    # 1. Algebraic identity / consistency check: DFT core-polarization efficiency (`lor_eff_pol_db`,
    #    `deb_eff_pol_db`) equals `closed_form_md_efficiency(Vol * eta_fill)` up to discrete w_tilde/mu_d,
    #    so `e_pol < bound` follows algebraically from Cauchy-Schwarz `eta_fill < 1`.
    # 2. Real physical check: across the main passband (200..320 MHz, indices 1:5 including f0 = 280 MHz),
    #    the flux-based total loop+core 3D FIT efficiency (`lor_eff_db`, `deb_eff_db`) lies strictly below
    #    the uniform-field bound (whereas at 160, 360, and 400 MHz it exceeds `e_MD` by up to +0.25/+0.28 dB).
    assert np.all((d3["lor_eta_fill"] > 0.70) & (d3["lor_eta_fill"] <= 1.0))
    assert np.all((d3["deb_eta_fill"] > 0.70) & (d3["deb_eta_fill"] <= 1.0))
    assert np.all(d3["lor_eff_pol_db"] < d3["lor_eff_db"])
    assert np.all(d3["lor_eff_pol_db"] < d3["lor_ana_db"][None, :])
    assert np.all(d3["lor_eff_db"][:, 1:5] < d3["lor_ana_db"][None, 1:5])
    assert np.all(d3["deb_eff_pol_db"] < d3["deb_eff_db"])
    assert np.all(d3["deb_eff_pol_db"] < d3["deb_ana_db"][None, :])
    assert np.all(d3["deb_eff_db"][:, 1:5] < d3["deb_ana_db"][None, 1:5])

    # Direct 3D FIT check of the air-core vacuum loop (`material=None`): the rod+loop run radiates
    # 59.6x more power than the bare loop at f0 (rod-only moment ratio sqrt(59.6) - 1 = 6.72).
    # Because the loop's contribution is not a coherent in-phase addition to a uniform dipole moment,
    # the measured difference Delta = e_tot - e_pol_dft (+0.31 dB) is strictly positive (adding the
    # lossless loop increases radiated power, 0 < Delta) and strictly below the coherent in-phase
    # estimate 20*log10(1 + 1/sqrt(59.6))*(1 - e) = 0.94 dB (0 < Delta < 0.94 dB).
    rod_to_loop_ratio = float(d3["rod_to_loop_prad_ratio"][idx_mid])
    assert rod_to_loop_ratio > 40.0
    in_phase_eff_boost_db = float(
        20.0 * np.log10(1.0 + 1.0 / np.sqrt(rod_to_loop_ratio)) * (1.0 - eff_half_pec)
    )
    assert np.isclose(in_phase_eff_boost_db, 0.94, atol=0.01)
    delta_loop_db = float(d3["lor_eff_db"][2, idx_mid]) - e_pol_dft_db
    assert 0.0 < delta_loop_db < in_phase_eff_boost_db

    # Consistency check on the analytical bound code path: perturbing C_YOUSEFI_EXACT by +35% in
    # closed_form_md_efficiency moves the bound below `lor_eff_pol_db` and must fail the inequality.
    monkeypatch.setattr(h, "C_YOUSEFI_EXACT", h.C_YOUSEFI_EXACT * 1.35)
    mutated_ana_db = 10.0 * np.log10(
        h.closed_form_md_efficiency(d3["freqs_ghz"] * 1.0e9, h.CZN_HM_OHM_M, float(d2["vol_full_m3"]))
    )
    assert not np.all(d3["lor_eff_pol_db"] < mutated_ana_db[None, :])
    monkeypatch.undo()


def test_duality_ebwp_convergence_and_quantitative_summary_table():
    """Verify continuous PEC<->PMC duality and grading refinement, finite ground diffraction, convergence, and table."""
    d4 = np.load(TUT_DIR / "data" / "part4_dual_pair.npz")
    # Continuous Maxwell boundary-value dual (horiz mag on PEC vs horiz diel on dual-shifted PMC) agrees to < 0.06 dB
    # on the refined outer grading (g = 1.18, actual max 0.038 dB), and refining outer grading from g = 1.35 to g = 1.18
    # reduces the primal/dual grading error by > 3x for every constitutive pair
    assert np.max(d4["dual_pmc_diff_f0_db"]) < 0.06
    assert np.all(d4["dual_pmc_diff_f0_db"] < 0.35 * d4["dual_pmc_coarse_diff_f0_db"])
    assert np.all(d4["dual_pmc_diff_f0_db"] < d4["dual_pec_diff_f0_db"])
    # Sebastian orientational PEC pair (horiz mag on PEC vs vert diel monopole on PEC) tracks within 1.25 dB
    assert np.max(d4["dual_pec_diff_f0_db"]) < 1.25

    d5 = np.load(TUT_DIR / "data" / "part5_ebwp_and_convergence.npz")
    assert float(d5["fit_mbar_to_strip_ratio"]) > 100.0
    assert float(np.min(d5["ebwp_advantage_db"])) > 15.0

    # Finite PEC ground plane at h = 0.025*lam0: integrated R/R_free at 1.10*lam0 converges to infinite-PEC (1.980)
    # within 1.5%, while the two symmetric edge waves add at broadside and raise D(0) from 3.20 to 4.74 over 0.3-1.1*lam0
    assert np.isclose(float(d5["h_gnd_lam"]), 0.025, rtol=0.02)
    assert np.isclose(float(d5["r_ratio_finite_gnd"][-1]), float(d5["inf_gnd_r_ratio"]), rtol=0.015)
    assert float(d5["zenith_dir_finite_gnd"][-1]) > float(d5["inf_gnd_zenith_dir"]) * 1.45

    # Strictly monotone 3-level convergence on the Part 2/3 rod and grid resolution criteria (>= 4 cells across)
    conv_eff = np.asarray(d5["conv_eff"], dtype=float)
    assert conv_eff[0] < conv_eff[1] < conv_eff[2]
    assert float(d5["conv_rel_err"][2]) < float(d5["conv_rel_err"][1]) < float(d5["conv_rel_err"][0])
    assert float(d5["conv_rel_err"][1]) < 0.045
    assert float(d5["conv_rel_err"][2]) < 0.030
    assert 1.55 <= float(d5["conv_p_obs"]) <= 2.35
    assert int(d5["n_thick_conv"][0]) >= 4
    assert float(d5["conv_cpw_mat"][0]) >= 20.0
    assert float(d5["conv_cells_per_skin"][0]) >= 4.0

    summary = json.loads((TUT_DIR / "data" / "summary.json").read_text(encoding="utf-8"))
    py_text = (TUT_DIR / "magnetic_antenna_ground.py").read_text(encoding="utf-8")
    assert _check_table_matches_summary(py_text, summary)

    # Mutation check: perturbing a summary value must cause the Quantitative Summary table check to fail
    mutated_summary = dict(summary)
    mutated_summary["hmd_low_h_fit"] = summary["hmd_low_h_fit"] + 0.1234
    assert not _check_table_matches_summary(py_text, mutated_summary)
