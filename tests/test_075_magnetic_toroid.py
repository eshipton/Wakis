"""Tests for Tutorial 4.7 · The magnetic toroid: a closed magnetization loop radiates like an axial electric dipole."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TUT_DIR = ROOT / "tutorials" / "04_radiation_and_antennas" / "04-07_magnetic_toroid"
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _helpers as h  # noqa: E402


def _extract_markdown_summary_table_from_py(py_path: Path) -> str:
    """Extract the Quantitative Summary markdown table block from `magnetic_toroid.py`."""
    lines = py_path.read_text(encoding="utf-8").splitlines()
    in_section = False
    table_lines: list[str] = []
    for line in lines:
        if "## Quantitative Summary Table" in line:
            in_section = True
            continue
        if in_section:
            if line.startswith("# %%"):
                break
            stripped = line.lstrip("#").strip()
            if stripped.startswith("|"):
                table_lines.append(stripped)
    return "\n".join(table_lines)


def _extract_markdown_summary_table_from_ipynb(ipynb_path: Path) -> str:
    """Extract the Quantitative Summary markdown table block from `magnetic_toroid.ipynb`."""
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if "## Quantitative Summary Table" in src:
            table_lines = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("|")]
            return "\n".join(table_lines)
    return ""


def test_baseline_dipoles_and_vsh_orthogonality_with_mutation(monkeypatch):
    """Verify VSH TM/TE decomposition, exp(+i*omega*t) a_10 phase vs T_z, (ka)^4 loop scaling, and code-path mutations."""
    theta_1d, phi_1d, w_t, w_p, tt_flat, pp_flat = h.make_quadrature_sphere(24, 48)
    r_ff = 4.0
    k0 = h.K0_RAD_M

    # 1. Analytic toroidal dipole in Wakis's exp(+i*omega*t) convention: verify VSH a_10 magnitude and phase (-pi/2 vs T_z)
    tz_ref = (1.25e-6) * np.exp(0.35j)
    th_tor = h.toroidal_dipole_radiation_theory(
        h.F0_HZ, tz_ref, r_major_m=h.R_MAJOR_M, theta_rad=tt_flat, r_ff_m=r_ff
    )
    vsh_tm = h.spherical_multipole_from_farfield(
        theta_1d, phi_1d, w_t, w_p, th_tor["Etheta"], th_tor["Ephi"], l_max=5, r_ff=r_ff, k0=k0
    )
    assert vsh_tm["p_a10_frac"] == pytest.approx(1.0, abs=1e-12)
    assert vsh_tm["b1_over_a1"] < 1e-14
    assert vsh_tm["a10"] == pytest.approx(th_tor["a10_j1"], rel=1e-6)
    assert np.angle(vsh_tm["a10"] / tz_ref) == pytest.approx(-0.5 * np.pi, abs=1e-6)

    # Discrimination check & downstream VSH projector mutation: flipping the sign of sph_harm_y inside
    # spherical_multipole_from_farfield (equivalent to swapping exp(+i*omega*t) vs exp(-i*omega*t) polar-vector sign)
    # shifts arg(a10 / T_z) from -pi/2 to +pi/2 and MUST fail the original phase assertion.
    orig_sph_harm_y = h.sph_harm_y
    with monkeypatch.context() as mp:
        mp.setattr(h, "sph_harm_y", lambda l_deg, m_ord, th_in, ph_in: -orig_sph_harm_y(l_deg, m_ord, th_in, ph_in))
        vsh_wrong_conv = h.spherical_multipole_from_farfield(
            theta_1d, phi_1d, w_t, w_p, th_tor["Etheta"], th_tor["Ephi"], l_max=5, r_ff=r_ff, k0=k0
        )
        assert not (abs(np.angle(vsh_wrong_conv["a10"] / tz_ref) - (-0.5 * np.pi)) < 1e-6)

    # 2. Analytic magnetic dipole (TE_10: E_phi ~ sin(theta), E_theta = 0)
    et_te = np.zeros_like(th_tor["Etheta"])
    ep_te = th_tor["Etheta"].copy()
    vsh_te = h.spherical_multipole_from_farfield(
        theta_1d, phi_1d, w_t, w_p, et_te, ep_te, l_max=5, r_ff=r_ff, k0=k0
    )
    assert vsh_te["p_b_l"][0] / vsh_te["p_vsh_tot"] == pytest.approx(1.0, abs=1e-12)
    assert vsh_te["b1_over_a1"] > 1e14

    # Discrimination check: feeding a TE magnetic-dipole far field into the unpatched VSH projector
    # produces b1/a1 > 1e14 and MUST fail the TM closed-toroid assertion b1/a1 < 1e-4.
    assert not (vsh_te["b1_over_a1"] < 1e-4)

    # 3. Verify 3D FIT Task 1 results from data/task_01_baseline_dipoles.json
    d1 = json.loads((h.DATA_DIR / "task_01_baseline_dipoles.json").read_text())
    assert d1["electric_dipole"]["p_a10_frac"] > 0.9999
    assert d1["electric_dipole"]["b1_over_a1"] < 1e-5
    assert d1["magnetic_loop_air"]["p_b1_frac"] > 0.9998
    assert d1["magnetic_loop_air"]["a1_over_b1"] < 1e-5

    # (ka)^4 scaling of small loop radiation resistance via small_loop_radiation_resistance
    ka4 = d1["ka4_scaling"]
    r10 = h.small_loop_radiation_resistance(h.F0_HZ, 0.010)
    r15 = h.small_loop_radiation_resistance(h.F0_HZ, 0.015)
    assert (r15 / r10) == pytest.approx(5.0625, rel=1e-12)
    # Discretization-justified tolerance: sub-cell area weights give 1.84% err vs (1.5)^4 (1.23% vs J1^2).
    assert ka4["ratio_rel_err"] < 0.025
    assert ka4["ratio_rel_err_vs_exact_j1"] < 0.020

    # Discrimination check & upstream theory mutation: mutating small_loop_radiation_resistance_theory to scale as
    # (ka)^2 instead of (ka)^4 and calling the unpatched downstream small_loop_radiation_resistance MUST fail.
    with monkeypatch.context() as mp:
        mp.setattr(
            h,
            "small_loop_radiation_resistance_theory",
            lambda ka, mu_eff=1.0, n_turns=1: {
                "ka": np.asarray(ka, dtype=float),
                "r_rad_exact_ohm": (np.pi / 6.0) * h.ETA0 * float(n_turns**2) * (np.asarray(mu_eff, dtype=float) ** 2) * (np.asarray(ka, dtype=float) ** 2),
                "r_rad_20pi2_ohm": 20.0 * (np.pi**2) * float(n_turns**2) * (np.asarray(mu_eff, dtype=float) ** 2) * (np.asarray(ka, dtype=float) ** 2),
            },
        )
        r10_mut = h.small_loop_radiation_resistance(h.F0_HZ, 0.010)
        r15_mut = h.small_loop_radiation_resistance(h.F0_HZ, 0.015)
        assert not (abs(ka4["ratio_15_to_10_sim"] - (r15_mut / r10_mut)) / (r15_mut / r10_mut) < 0.025)

    # Permeable core boost and single-port PEC wire loop Re Z_in = 2*P_rad/|I_port|^2 decomposition
    assert d1["magnetic_loop_core"]["power_gain_over_air"] > 4.0
    lw = d1["lumped_wire_loop"]
    assert lw["a1_over_b1_feed_asymmetry"] > 0.30
    assert lw["z_in_real_ohm"] == pytest.approx(lw["r_rad_from_prad_ohm"], rel=0.03)
    r_circ_code = h.small_loop_radiation_resistance(h.F0_HZ, lw["a_mm"] * 1e-3)
    assert lw["r_rad_square_small_loop_ohm"] == pytest.approx(r_circ_code * ((4.0 / np.pi) ** 2), rel=1e-12)


def test_closed_toroid_dubovik_tugushev_and_radiation_with_mutations(monkeypatch):
    """Verify closed-loop m=0, Dubovik-Tugushev J_b vs M identity, p_eff radiation, octupole amplitude & slope, and mutations."""
    d2 = json.loads((h.DATA_DIR / "task_02_closed_toroid.json").read_text())
    mom = d2["moments"]
    vsh = d2["vsh_toroid_8feed"]
    peq = d2["power_equivalence"]
    f2 = d2["frequency_scaling"]
    r2 = d2["octupole_scaling"]

    # 1. Closed azimuthal magnetization loop cancels net magnetic dipole m = int M dV
    assert mom["m_cancel_ratio"] < 1e-5

    # 2. Dubovik-Tugushev integral of J_b = curl(M) equals 1/2 int (r x M) dV on the 3D FIT grid
    assert mom["dt_vs_mag_rel_err"] < 1e-12
    assert mom["dt_mutated_rel_err"] == pytest.approx(0.40, rel=1e-5)
    assert not (mom["dt_mutated_rel_err"] < 1e-12)

    # 3. Spherical multipole spectrum & exp(+i*omega*t) phase of a_10 relative to T_z (-pi/2 rad)
    assert vsh["p_a10_frac"] > 0.99999
    assert vsh["b1_over_a1"] < 1e-4
    assert vsh["even_l_power_frac"] < 1e-8
    assert vsh["pattern_sin_theta_l2_err"] < 0.005
    assert vsh["cross_pol_ratio"] < 1e-4
    assert vsh["a10_over_tz_phase_rad"] == pytest.approx(-0.5 * np.pi, abs=0.03)

    # 4. Zero-fit-parameter radiated power agreement between Near2FarMonitor, FluxMonitor, and p_eff = -i*k0*T_tot/c0
    assert peq["rel_err_n2f_vs_j1"] < 0.01
    assert peq["rel_err_flux_vs_j1"] < 0.01
    # Discrimination check (free-current T_free vs total permeable-core T_tot): passing only the free-current
    # moment T_free (omitting the permeable core magnetization M_phi) into toroidal_dipole_radiation_theory
    # underpredicts P_rad by >99% and MUST fail the <1% self-consistency assertion.
    tz_free_only = complex(mom["tz_free_dt_am3"]["real"], mom["tz_free_dt_am3"]["imag"])
    p_rad_free_only = float(
        h.toroidal_dipole_radiation_theory(h.F0_HZ, tz_free_only, r_major_m=h.R_MAJOR_M)["p_rad_j1_w"]
    )
    assert not (abs(peq["p_rad_n2f_w"] - p_rad_free_only) / p_rad_free_only < 0.01)

    # 5. Broadband self-consistency (0.55-0.95 GHz), fixed-T omega^6 scaling, and octupole amplitude & slope
    for ratio_band in f2["p_rad_over_theory_j1_band"]:
        assert ratio_band == pytest.approx(1.0, rel=0.005)
    assert abs(f2["slope_omega_fixed_t_point_corrected"] - 6.0) < 0.05
    assert f2["slope_omega_fixed_i"] - f2["slope_omega_fixed_t_point_corrected"] > 0.35
    assert abs(r2["fitted_slope_vs_kr"] - 2.0) < 0.25

    dx_core_m = 0.0025
    orig_sph_jn = h.spherical_jn
    for kr_val, sim_val, th_val in zip(r2["k0_r"], r2["a30_over_a10_sim"], r2["a30_over_a10_theory"]):
        r_maj_m = float(kr_val / h.K0_RAD_M)
        th_out = h.toroidal_dipole_radiation_theory(
            h.F0_HZ, 1.0, r_major_m=r_maj_m, r_minor_m=h.R_MINOR_M
        )
        # Verify the analytical thin-ring prefactor is sqrt(7/8)*j_3/j_1 ≈ (sqrt(14)/140)*(kR)^2 = 0.026726*(kR)^2
        th_check = float(th_out["a30_over_a10_ring"])
        assert th_val == pytest.approx(th_check, rel=1e-12)
        assert th_val == pytest.approx((np.sqrt(14.0) / 140.0) * (kr_val**2), rel=0.02)
        # Derived finite minor cross-section correction: averaging (rho^2 - 4*z^2)/R^2 over (rho - R)^2 + z^2 <= r^2
        # yields 1 - 0.75*(r/R)^2, which matches the simulated |a_30|/|a_10| within the 3.5% regression bound:
        th_fin_val = float(th_out["a30_over_a10_finite_r"])
        assert th_fin_val == pytest.approx(th_val * (1.0 - 0.75 * ((h.R_MINOR_M / r_maj_m) ** 2)), rel=1e-12)
        tol_a30_fin_reg = 0.035  # Regression bound against derived finite-r/R formula (actual errors: 0.3%-2.5%)
        assert sim_val == pytest.approx(th_fin_val, rel=tol_a30_fin_reg)
        # Regression bound against uncorrected thin-ring formula:
        tol_a30_thin_reg = float((h.R_MINOR_M / r_maj_m) ** 2 + (dx_core_m / r_maj_m) ** 2)
        assert sim_val == pytest.approx(th_val, rel=tol_a30_thin_reg)
        # Discrimination check & upstream Bessel mutation: mutating spherical_jn for n=3 by (sqrt(6)/8)/sqrt(7/8)
        # (3.055x smaller) and calling the unpatched toroidal_dipole_radiation_theory MUST fail the regression bound.
        with monkeypatch.context() as mp:
            mp.setattr(
                h,
                "spherical_jn",
                lambda n_ord, z_arg: (
                    orig_sph_jn(n_ord, z_arg) * ((np.sqrt(6.0) / 8.0) / np.sqrt(7.0 / 8.0))
                    if int(n_ord) == 3
                    else orig_sph_jn(n_ord, z_arg)
                ),
            )
            wrong_th_val = float(
                h.toroidal_dipole_radiation_theory(
                    h.F0_HZ, 1.0, r_major_m=r_maj_m, r_minor_m=h.R_MINOR_M
                )["a30_over_a10_finite_r"]
            )
            assert not (abs(sim_val - wrong_th_val) / wrong_th_val < tol_a30_fin_reg)


def test_live_quick_fit_toroid_asymmetric_grid_and_mutations(monkeypatch):
    """Run live 3D FIT checks: FIT charge conservation of wire currents, 3D helical/Marinov winding, asymmetric grid, and broken feed."""
    # 1. Verify non-tautological 3D helical winding vs planar z=0 Marinov compensating loop on the FIT grid
    dx_core_m = 0.0025
    assert (2.0 * h.R_MINOR_M) / dx_core_m >= 4.0
    grid = h.build_toroid_grid(dx_core_m=dx_core_m, graded=True)
    solver = h.SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
    core_mat = h.make_core_material(mu_dc=h.MU_DC_DEFAULT)
    pulse = h.GaussianSource(frequency=h.F0_HZ, fwidth=0.22e9)

    _, info_closed = h.setup_toroid_core_and_source(
        solver, core_mat, pulse, n_windings=8, helical_parasitic_turn=False, marinov_compensating_loop=False
    )
    _, info_hel = h.setup_toroid_core_and_source(
        solver, core_mat, pulse, n_windings=8, helical_parasitic_turn=True, marinov_compensating_loop=False
    )
    _, info_mar = h.setup_toroid_core_and_source(
        solver, core_mat, pulse, n_windings=8, helical_parasitic_turn=True, marinov_compensating_loop=True
    )
    _, info_naive_r = h.setup_toroid_core_and_source(
        solver,
        core_mat,
        pulse,
        n_windings=8,
        helical_parasitic_turn=True,
        marinov_compensating_loop=True,
        marinov_loop_radius_m=h.R_MAJOR_M,
    )
    # Discrete FIT charge-conservation check (||div_FIT J||_2 / ||J||_2 < 1e-12) for the 3D helix,
    # the planar Marinov compensation loop alone, and their combined current:
    div_hel = h.compute_fit_divergence_of_j(solver, info_hel["j_flat"])
    div_loop = h.compute_fit_divergence_of_j(solver, info_mar["j_flat"] - info_hel["j_flat"])
    div_mar = h.compute_fit_divergence_of_j(solver, info_mar["j_flat"])
    assert div_hel["rel_div_norm"] < 1e-12
    assert div_loop["rel_div_norm"] < 1e-12
    assert div_mar["rel_div_norm"] < 1e-12
    # Code-path mutation on wire deposition: if _split_polyline_at_grid_planes drops the closing segment,
    # deposit_closed_wire_on_edges leaves open endpoints and ||div_FIT J||_2 / ||J||_2 < 1e-12 MUST fail.
    orig_split = h._split_polyline_at_grid_planes
    with monkeypatch.context() as mp:
        mp.setattr(
            h,
            "_split_polyline_at_grid_planes",
            lambda *a, **kw: tuple(arr[:-5] for arr in orig_split(*a, **kw)),
        )
        _, info_open = h.setup_toroid_core_and_source(
            solver, core_mat, pulse, n_windings=8, helical_parasitic_turn=True, marinov_compensating_loop=False
        )
        div_open = h.compute_fit_divergence_of_j(solver, info_open["j_flat"])
        assert not (div_open["rel_div_norm"] < 1e-12)

    # Verify wire-on-edges helical winding + Marinov loop (same current -I_turn, no rescaling) has minimum |m_z| at R_1
    m_hel = h.compute_free_current_magnetic_dipole(grid, info_hel["j_flat"])
    m_mar = h.compute_free_current_magnetic_dipole(grid, info_mar["j_flat"])
    m_naive = h.compute_free_current_magnetic_dipole(grid, info_naive_r["j_flat"])
    i_turn = (2.0 * np.pi * h.R_MAJOR_M * 1.0) / 8.0
    r1_exact = h.R_MAJOR_M * np.sqrt(1.0 + 0.5 * ((h.R_MINOR_M / h.R_MAJOR_M) ** 2))
    # Polyline chord-discretisation error for 90 segments per poloidal turn (linspace(0, 1, 721) with N=8) is O((2*pi/90)^2)
    chord_tol = (2.0 * np.pi / 90.0) ** 2
    assert m_hel[2] == pytest.approx(np.pi * (r1_exact**2) * i_turn, rel=chord_tol)
    assert abs(m_mar[2]) < chord_tol * abs(m_hel[2])
    assert abs(m_mar[2]) < 0.01 * abs(m_naive[2])
    assert not np.allclose(info_mar["j_flat"], info_closed["j_flat"])

    # 2. Live 3D FIT run on a deliberately asymmetric grid with an off-center torus at dx_core = 2.5 mm (>= 4 cells across 2r)
    # so m=0 and b1/a1 << 1 are tested without relying on mirror-symmetric grid cancellation!
    res_asym = h.run_single_fit_case(
        geometry="toroid",
        mu_dc=h.MU_DC_DEFAULT,
        r_major_m=h.R_MAJOR_M,
        r_minor_m=h.R_MINOR_M,
        n_windings=8,
        dx_core_m=dx_core_m,
        n_cycles=3.0,
        center_offset_m=(0.00035, -0.00025, 0.00020),
        grid_asymmetry_m=(0.00060, -0.00045, 0.00030),
    )
    # Heuristic binary-staircase asymmetry bound for the shifted off-center binary torus mask
    # (shifting (rho - R)^2 + z^2 <= r^2 by (0.35, -0.25, 0.20) mm at dx_core = 2.5 mm flips 33 of 472 core cells
    # between opposite sides of the ring, giving m_cancel_ratio ≈ 6.2e-3 < 0.01 and b1/a1 ≈ 0.079 < 0.0884;
    # by contrast, the centered closed torus in Task 2/Task 4 satisfies b1/a1 < 1e-4):
    staircase_asym_tol = float((dx_core_m / (2.0 * h.R_MINOR_M)) / np.sqrt(8.0))
    assert res_asym["moments"]["m_cancel_ratio"] < 0.01
    assert res_asym["moments"]["dt_vs_mag_rel_err"] < 1e-12
    assert res_asym["vsh"]["b1_over_a1"] < staircase_asym_tol
    assert np.angle(res_asym["vsh"]["a10"] / res_asym["moments"]["tz_tot_am3"]) == pytest.approx(-0.5 * np.pi, abs=0.08)

    # 3. Live mutation: unbalance the torus by turning off one of the 8 feeds at dx_core = 2.5 mm (>= 4 cells across 2r)
    res_broken = h.run_single_fit_case(
        geometry="toroid",
        mu_dc=h.MU_DC_DEFAULT,
        r_major_m=h.R_MAJOR_M,
        r_minor_m=h.R_MINOR_M,
        n_windings=8,
        active_windings=[0, 1, 2, 3, 4, 5, 6],
        dx_core_m=dx_core_m,
        n_cycles=3.0,
    )
    assert res_broken["vsh"]["b1_over_a1"] > 5.0 * staircase_asym_tol
    assert not (res_broken["vsh"]["b1_over_a1"] < staircase_asym_tol)


def test_c4_four_rods_and_break_the_loop_cases(monkeypatch):
    """Verify C4v 4-rod square m=0 & m=+/-4 electric/magnetic residuals, all 5 broken-loop modes, and Marinov's loop."""
    d3 = json.loads((h.DATA_DIR / "task_03_square_four_rods.json").read_text())
    recs = d3["records"]
    assert len(recs) == 3
    assert recs[0]["m_cancel_ratio"] < 1e-5
    for r in recs:
        assert r["m_cancel_ratio"] < 1e-5
        assert r["b1_over_a1"] < 1e-4
        assert r["p_a10_frac"] > 0.9999
        assert r["c4_a_over_a10"] >= 0.0
        assert r["c4_b_over_a10"] > 0.0
    # C4v azimuthal harmonic |m=+/-4|/|a_10| must grow monotonically with electrical size k_0 * R_sq
    c4_vals = [r["c4_m4_over_a10"] for r in recs]
    assert c4_vals[0] < c4_vals[1] < c4_vals[2]

    # Task 4 broken-loop cases: every broken case must fail the closed-loop |b_1|/|a_1| < 1e-3 criterion
    d4 = json.loads((h.DATA_DIR / "task_04_break_the_loop.json").read_text())
    cases = d4["cases"]
    assert cases["closed_8_feeds"]["b1_over_a1"] < 1e-4
    assert cases["four_rods_closed"]["b1_over_a1"] < 1e-4

    for broken_key, min_ratio in [
        ("core_air_gap_60deg", 1.5),
        ("seven_of_eight_feeds", 0.35),
        ("mode1_antiphase", 1e4),
        ("three_of_four_rods", 1.5),
        ("helical_uncompensated", 0.35),
    ]:
        b_over_a = cases[broken_key]["b1_over_a1"]
        assert b_over_a > min_ratio, f"{broken_key} expected |b1|/|a1| > {min_ratio}, got {b_over_a}"
        assert not (b_over_a < 1e-3)

    # Mode 1 antiphase cancels a_1 by antisymmetry: verify |a_1|_Mode1 / |a_1|_closed < 1e-4
    assert cases["mode1_antiphase"]["a1_over_a1_closed"] < 1e-4

    # Marinov's reverse equatorial compensating loop at z = 0 of radius R_1 = R*sqrt(1 + 0.5*(r/R)^2)
    # driven by the exact same current -I_turn (zero rescaling) passes through a sharp minimum at R_1
    msw = d4["marinov_r_loop_sweep"]
    assert cases["helical_uncompensated"]["b1_over_a1"] == pytest.approx(
        msw["b1_over_a1_theory_uncompensated"], rel=0.03
    )
    b1_sw = msw["b1_over_a1"]
    idx_min = int(np.argmin(b1_sw))
    assert msw["r_loop_mm"][idx_min] == pytest.approx(msw["r1_marinov_mm"], rel=1e-9)
    assert b1_sw[0] > b1_sw[1] > b1_sw[2] > b1_sw[idx_min]
    assert b1_sw[idx_min] < b1_sw[4] < b1_sw[5]
    assert cases["helical_marinov_compensated"]["b1_over_a1"] < 1.5e-3
    assert cases["helical_marinov_compensated"]["b1_over_a1"] > 1e-4
    assert msw["suppression_factor_at_r1"] > 350.0
    assert msw["improvement_over_naive_r_major"] > 18.0

    # Code-path mutation: monkeypatching deposit_closed_wire_on_edges so the planar Marinov loop uses +I_turn
    # instead of -I_turn doubles the axial magnetic dipole m_z instead of cancelling it, and MUST fail.
    grid = h.build_toroid_grid(dx_core_m=0.0025, graded=True)
    solver = h.SolverFIT3D(grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0)
    core_mat = h.make_core_material(mu_dc=h.MU_DC_DEFAULT)
    pulse = h.GaussianSource(frequency=h.F0_HZ, fwidth=0.22e9)
    orig_deposit = h.deposit_closed_wire_on_edges
    with monkeypatch.context() as mp:
        mp.setattr(
            h,
            "deposit_closed_wire_on_edges",
            lambda g, xp, yp, zp, current_amp: orig_deposit(g, xp, yp, zp, abs(float(current_amp))),
        )
        _, info_mar_mut = h.setup_toroid_core_and_source(
            solver, core_mat, pulse, n_windings=8, helical_parasitic_turn=True, marinov_compensating_loop=True
        )
        m_mar_mut = h.compute_free_current_magnetic_dipole(grid, info_mar_mut["j_flat"])
        assert not (abs(m_mar_mut[2]) < 1e-6)


def test_equivalence_limits_efficiency_q_and_energy_bookkeeping_with_mutation(monkeypatch):
    """Verify ring factor j_1 null (4.4934), near-field vs dipole, core TE01 mode, Q_rad with pi & F_1^2, and energy balance."""
    d5 = json.loads((h.DATA_DIR / "task_05_equivalence_limits.json").read_text())
    # Ring factor 3*j_1(x)/x vanishes at x = 4.49340946 (first zero of spherical j_1), NOT at jn_zeros(1, 1)[0] (zero of J_1)
    el_th = d5["electrical_size_theory"]
    assert el_th["kr_first_bessel_zero"] == pytest.approx(4.49340946, rel=1e-6)
    j1_cyl_zero = float(h.jn_zeros(1, 1)[0])
    assert el_th["kr_equatorial_j1_zero"] == pytest.approx(j1_cyl_zero, rel=1e-7)
    r_null = el_th["kr_first_bessel_zero"] / h.K0_RAD_M
    assert abs(float(h.toroidal_dipole_radiation_theory(h.F0_HZ, 1.0, r_major_m=r_null)["form_factor_j1"])) < 1e-8
    r_j1_cyl = j1_cyl_zero / h.K0_RAD_M
    assert float(h.toroidal_dipole_radiation_theory(h.F0_HZ, 1.0, r_major_m=r_j1_cyl)["form_factor_j1"]) > 0.12

    nf = d5["nearfield_vs_dipole"]
    assert nf["near_zone_20_to_30mm_e_rel_diff"] > 0.30
    assert nf["outer_zone_45_to_55mm_e_rel_diff"] < 0.15
    assert nf["e_metric_reduction_factor"] > 3.0
    assert nf["near_zone_20_to_30mm_hy_rel_diff"] > 0.25
    assert nf["outer_zone_45_to_55mm_hy_rel_diff"] < 0.05
    assert nf["hy_metric_reduction_factor"] > 7.0
    assert nf["hankel_octupole_decay_25_to_50mm"] > 4.0

    # Resolved (>= 20 cells/lambda_core) internal TE01-like dielectric-magnetic core mode sweep:
    # below cutoff (eps_core = 10, 2r/lambda_core = 0.356 < 0.5) |T_z| and P_rad are enhanced in agreement
    # with the hybrid (p_eff + p_pol) dipole model; above cutoff (eps_core = 22, 2r/lambda_core = 0.529 > 0.5)
    # the internal TE01 standing wave reverses sign across the minor cross-section, rolling over |T_z| and
    # collapsing P_rad by >85% relative to both eps_core = 1 and the point-dipole prediction.
    te01 = d5["core_te01_mode_sweep"]
    for rec_te in te01:
        assert rec_te["cells_per_lambda_core"] >= 20.0
    assert te01[0]["two_r_over_lambda_core"] < 0.20
    # Regression bounds on the committed TE01 study (not derived tolerances): they catch changes to the
    # resolved-TE01 run, not physics errors of a stated size.
    assert 0.30 < te01[1]["two_r_over_lambda_core"] < 0.50
    assert te01[2]["two_r_over_lambda_core"] > 0.50
    assert te01[1]["ratio_to_eps1"] > 1.20
    assert te01[1]["ratio_to_quasistatic_formula"] > 1.25
    assert te01[1]["p_rad_sim_w"] == pytest.approx(te01[1]["p_rad_hybrid_pred_w"], rel=0.04)
    assert 1.0 < te01[2]["ratio_to_eps1"] < te01[1]["ratio_to_eps1"]
    assert te01[2]["ratio_to_quasistatic_formula"] > 1.05
    assert te01[2]["p_rad_sim_w"] < 0.15 * te01[0]["p_rad_sim_w"]
    assert te01[2]["p_rad_sim_w"] < 0.15 * te01[2]["p_rad_hybrid_pred_w"]

    # Task 6: Poynting energy bookkeeping P_in = P_rad + P_loss_core and Q_rad formula with 1/pi and F_1^2
    d6 = json.loads((h.DATA_DIR / "task_06_efficiency_q_and_convergence.json").read_text())
    l0 = d6["loss_sweep"][0]
    q_th_check = float(
        h.toroidal_efficiency_and_q_theory(
            h.F0_HZ,
            r_major_m=h.R_MAJOR_M,
            r_minor_m=h.R_MINOR_M,
            mu_real=l0["mu_real"],
            tan_delta_m=l0["tan_delta_m"],
        )["q_rad_toroid"]
    )
    assert l0["q_rad_theory"] == pytest.approx(q_th_check, rel=1e-4)
    assert abs(l0["q_rad_sim"] - l0["q_rad_theory"]) / l0["q_rad_theory"] < 0.03

    # Discrimination check & upstream hesitivity_efficiency_and_chu_q_theory mutation: omitting the 1/pi factor in Q_rad
    # and calling the unpatched downstream toroidal_efficiency_and_q_theory MUST fail the <3% Q_rad assertion.
    orig_q_theory = h.hesitivity_efficiency_and_chu_q_theory
    with monkeypatch.context() as mp:
        def _wrong_q_theory(*args, **kwargs):
            out = dict(orig_q_theory(*args, **kwargs))
            out["q_rad_toroid"] = np.asarray(out["q_rad_toroid"], dtype=float) * np.pi
            return out

        mp.setattr(h, "hesitivity_efficiency_and_chu_q_theory", _wrong_q_theory)
        q_mut = float(
            h.toroidal_efficiency_and_q_theory(
                h.F0_HZ,
                r_major_m=h.R_MAJOR_M,
                r_minor_m=h.R_MINOR_M,
                mu_real=l0["mu_real"],
                tan_delta_m=l0["tan_delta_m"],
            )["q_rad_toroid"]
        )
        assert not (abs(l0["q_rad_sim"] - q_mut) / q_mut < 0.03)

    for rec in d6["loss_sweep"]:
        assert rec["energy_closure_rel_err"] < 0.002
        assert rec["energy_closure_mutated_rel_err"] > 0.30
        assert not (rec["energy_closure_mutated_rel_err"] < 0.002)
        assert rec["q_ratio_over_chu_sim"] > 50.0

    # 3-level resolution study: all levels satisfy >= 4 cells across 2r and >= 20 cells/lambda_core;
    # because the binary drive staircase oscillates by +/-9%, |T_z^sim| and P_rad fail the stated < 2%
    # grid-convergence criterion (|Delta P_rad|/P_rad of last two levels < 2%) and are computed as NOT grid-converged.
    conv = d6["convergence"]
    assert len(conv["levels"]) == 3
    crit_rel_dp = float(conv.get("grid_convergence_criterion_rel_dp_max", 0.02))
    assert crit_rel_dp == pytest.approx(0.02, rel=1e-12)
    computed_is_conv = bool(abs(float(conv["p_rad_rel_change_fine_vs_med"])) < crit_rel_dp)
    assert conv["is_grid_converged"] is computed_is_conv
    assert conv["is_grid_converged"] is False
    for idx, lvl in enumerate(conv["levels"]):
        assert lvl["cells_across_2r"] >= 4.0 - 1e-9
        assert lvl["cells_per_lambda_core"] >= 20.0
        # Regression checks on the 4-6 cells/2r binary-staircase grids (discrete drive within 10%, dynamic |T_z| within 15%):
        assert conv["tz_disc_static_rel_err_levels"][idx] < 0.10
        assert conv["tz_rel_err_levels"][idx] < 0.15
        assert lvl["p_rad_self_consistency_rel_diff"] < 0.005
        assert lvl["energy_closure_rel_err"] < 0.001


def test_quantitative_summary_table_matches_json_and_fails_on_mutation(monkeypatch):
    """Verify that rendering does not mutate data/*.json and that the Quantitative Summary table & narrative match data/*.json."""
    py_path = TUT_DIR / "magnetic_toroid.py"
    ipynb_path = TUT_DIR / "magnetic_toroid.ipynb"
    summary_json_path = h.DATA_DIR / "magnetic_toroid_summary.json"

    # Verify recompute_json_postprocessing is removed and build_summary_json(write=False) does not mutate data/*.json
    assert not hasattr(h, "recompute_json_postprocessing")
    before_bytes = {p.name: p.read_bytes() for p in sorted(h.DATA_DIR.glob("*.json"))}
    _ = h.build_summary_json(data_dir=h.DATA_DIR, write=False)
    after_bytes = {p.name: p.read_bytes() for p in sorted(h.DATA_DIR.glob("*.json"))}
    assert before_bytes == after_bytes

    expected_md = h.build_quantitative_summary_markdown(data_dir=h.DATA_DIR).strip()
    py_table_md = _extract_markdown_summary_table_from_py(py_path).strip()
    ipynb_table_md = _extract_markdown_summary_table_from_ipynb(ipynb_path).strip()
    summary_json = json.loads(summary_json_path.read_text(encoding="utf-8"))

    assert py_table_md == expected_md
    assert ipynb_table_md == expected_md
    assert summary_json["quantitative_summary_markdown"].strip() == expected_md
    assert len(summary_json["quantitative_summary_rows"]) == 16

    # Also verify that every numerical narrative section generated from JSON via f-strings matches magnetic_toroid.py
    narr = h.build_narrative_markdown_from_json(data_dir=h.DATA_DIR)
    py_raw = py_path.read_text(encoding="utf-8")
    for sec_key in (
        "theory_4p2_countercases",
        "setup_mesh_bullets",
        "results_task_01",
        "results_task_02",
        "results_task_03",
        "results_task_04",
        "results_task_05",
        "results_task_06",
        "convergence_section",
        "quantitative_summary_table",
        "limits_and_failure_modes",
    ):
        commented_sec = "\n".join("# " + line if line else "#" for line in narr[sec_key].splitlines())
        assert commented_sec in py_raw, f"Narrative section {sec_key!r} in magnetic_toroid.py drifted from JSON f-string!"

    # Code-path mutation: if build_quantitative_summary_rows returns a corrupted simulation value,
    # build_quantitative_summary_markdown MUST disagree with the notebook table.
    orig_rows = h.build_quantitative_summary_rows(data_dir=h.DATA_DIR)
    corrupted_rows = [dict(r) for r in orig_rows]
    corrupted_rows[6]["simulation"] = "9.9999e-04 W (corrupted)"
    monkeypatch.setattr(h, "build_quantitative_summary_rows", lambda data_dir=h.DATA_DIR: corrupted_rows)
    mutated_md = h.build_quantitative_summary_markdown(data_dir=h.DATA_DIR).strip()
    assert not (py_table_md == mutated_md)


def test_cross_file_and_internal_json_consistency():
    """Verify internal consistency of task_02 and cross-file agreement across task_02, task_03, task_04, task_05, and task_06 to rel=1e-12."""
    d2 = json.loads((h.DATA_DIR / "task_02_closed_toroid.json").read_text())
    d3 = json.loads((h.DATA_DIR / "task_03_square_four_rods.json").read_text())
    d4 = json.loads((h.DATA_DIR / "task_04_break_the_loop.json").read_text())
    d5 = json.loads((h.DATA_DIR / "task_05_equivalence_limits.json").read_text())
    d6 = json.loads((h.DATA_DIR / "task_06_efficiency_q_and_convergence.json").read_text())

    # 1. Internal consistency of task_02_closed_toroid.json (Item 1):
    mom2 = d2["moments"]
    peq2 = d2["power_equivalence"]
    vsh2 = d2["vsh_toroid_8feed"]
    f2 = d2["frequency_scaling"]
    tz2_c = complex(mom2["tz_tot_am3"]["real"], mom2["tz_tot_am3"]["imag"])
    tz2_abs = float(abs(tz2_c))
    th2_check = h.toroidal_dipole_radiation_theory(h.F0_HZ, tz2_c, r_major_m=h.R_MAJOR_M)
    assert peq2["p_rad_theory_j1_w"] == pytest.approx(float(th2_check["p_rad_j1_w"]), rel=1e-12)
    assert peq2["p_rad_theory_point_w"] == pytest.approx(float(th2_check["p_rad_point_w"]), rel=1e-12)
    assert mom2["tz_sim_over_disc_residual_ratio"] == pytest.approx(
        tz2_abs / float(mom2["tz_disc_static_drive_am3"]), rel=1e-12
    )
    assert d2["marinov_quasistatic"]["tz_tot_sim_abs_am3"] == pytest.approx(tz2_abs, rel=1e-12)
    assert d2["feed_comparison"]["toroid_8_windings"]["tz_tot_abs_am3"] == pytest.approx(tz2_abs, rel=1e-12)
    assert d2["feed_comparison"]["toroid_8_windings"]["p_rad_w"] == pytest.approx(peq2["p_rad_n2f_w"], rel=1e-12)
    idx_f0 = int(np.argmin(np.abs(np.asarray(f2["freqs_ghz"], dtype=float) - (h.F0_HZ / 1e9))))
    assert f2["tz_tot_band_abs_am3"][idx_f0] == pytest.approx(tz2_abs, rel=1e-12)
    assert f2["p_rad_fixed_i_w"][idx_f0] == pytest.approx(peq2["p_rad_n2f_w"], rel=1e-12)
    assert f2["p_rad_theory_j1_band_w"][idx_f0] == pytest.approx(peq2["p_rad_theory_j1_w"], rel=1e-12)

    # 2. Cross-file consistency across task_04.cases.closed_8_feeds, task_06.convergence.levels[0] (dx_core = 2.5 mm),
    # task_02 (power_equivalence, vsh_toroid_8feed, moments), and task_05.winding_convergence[2] to rel=1e-12 (Item 2):
    c4_closed = d4["cases"]["closed_8_feeds"]
    l0_conv = d6["convergence"]["levels"][0]
    w8_t5 = next(r for r in d5["winding_convergence"] if r["n_windings"] == "8")
    assert l0_conv["dx_core_mm"] == pytest.approx(d4["dx_core_mm"], rel=1e-12)
    assert c4_closed["p_rad_n2f_w"] == pytest.approx(l0_conv["p_rad_n2f_w"], rel=1e-12)
    assert c4_closed["p_rad_n2f_w"] == pytest.approx(peq2["p_rad_n2f_w"], rel=1e-12)
    assert c4_closed["p_rad_n2f_w"] == pytest.approx(w8_t5["p_rad_w"], rel=1e-12)
    assert c4_closed["b1_over_a1"] == pytest.approx(l0_conv["b1_over_a1"], rel=1e-12)
    assert c4_closed["b1_over_a1"] == pytest.approx(vsh2["b1_over_a1"], rel=1e-12)
    assert c4_closed["b1_over_a1"] == pytest.approx(w8_t5["b1_over_a1"], rel=1e-12)
    assert c4_closed["p_a10_frac"] == pytest.approx(l0_conv["p_a10_frac"], rel=1e-12)
    assert c4_closed["p_a10_frac"] == pytest.approx(vsh2["p_a10_frac"], rel=1e-12)
    assert c4_closed["p_a10_frac"] == pytest.approx(w8_t5["p_a10_frac"], rel=1e-12)
    assert c4_closed["tz_tot_abs_am3"] == pytest.approx(l0_conv["tz_tot_abs_am3"], rel=1e-12)
    assert c4_closed["tz_tot_abs_am3"] == pytest.approx(tz2_abs, rel=1e-12)
    assert c4_closed["tz_tot_abs_am3"] == pytest.approx(w8_t5["tz_tot_abs_am3"], rel=1e-12)
    assert c4_closed["m_cancel_ratio"] == pytest.approx(mom2["m_cancel_ratio"], rel=1e-12)
    assert c4_closed["m_cancel_ratio"] == pytest.approx(w8_t5["m_cancel_ratio"], rel=1e-12)
    assert l0_conv["p_rad_flux_w"] == pytest.approx(peq2["p_rad_flux_w"], rel=1e-12)
    assert l0_conv["p_rad_theory_j1_w"] == pytest.approx(peq2["p_rad_theory_j1_w"], rel=1e-12)

    # Also verify 4-rod square at R_sq = 12.5 mm matches between task_04 and task_03 to rel=1e-12:
    c4_rods = d4["cases"]["four_rods_closed"]
    r3_0 = d3["records"][0]
    assert c4_rods["p_rad_n2f_w"] == pytest.approx(r3_0["p_rad_n2f_w"], rel=1e-12)
    assert c4_rods["b1_over_a1"] == pytest.approx(r3_0["b1_over_a1"], rel=1e-12)
    assert c4_rods["tz_tot_abs_am3"] == pytest.approx(r3_0["tz_tot_abs_am3"], rel=1e-12)
    assert c4_rods["m_cancel_ratio"] == pytest.approx(r3_0["m_cancel_ratio"], rel=1e-12)

    # Also verify internal consistency of task_05: core_te01_mode_sweep[0] (eps_core = 1.0) equals winding_convergence[3] (n_windings = "inf"):
    w_inf_t5 = next(r for r in d5["winding_convergence"] if r["n_windings"] == "inf")
    te0_t5 = d5["core_te01_mode_sweep"][0]
    assert te0_t5["tz_tot_abs_am3"] == pytest.approx(w_inf_t5["tz_tot_abs_am3"], rel=1e-12)
    assert te0_t5["p_rad_sim_w"] == pytest.approx(w_inf_t5["p_rad_w"], rel=1e-12)
    assert te0_t5["wall_s"] == pytest.approx(w_inf_t5["wall_s"], rel=1e-12)
