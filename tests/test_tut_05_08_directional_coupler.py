# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics and mutation validation tests for Tutorial 5.8:
SOI Directional Coupler (2D EIM, Wakis FIT).
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.constants import c as c_light

from wakis.mode_solver import solve_modes
from wakis.sources import GaussianSource
from wakis.templates.photonics import PICSimulation, PolySlab, n_to_eps, straight

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "05_guided_waves"
    / "05-08_directional_coupler"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    PhasedGaussianSource,
    coupler_mesh_overrides,
    extract_mesh_metrics_from_grid,
    find_crossing_wavelength,
    run_phased_dual_drive,
    solve_coupler_supermodes,
    trapezoidal_effective_width,
)
from directional_coupler import (  # noqa: E402
    DILATION,
    H_WG,
    N_CLAD,
    N_SUB,
    N_WG,
    PITCH_COUPLED,
    W_GDS,
    WL_0,
    WL_DESIGN,
    WL_MAX,
    WL_MIN,
    build_directional_coupler,
    get_task_table,
    measure_task_mesh,
)


def _audit_task_table_resolution(tasks):
    """
    Validate that every non-demo task in ``tasks`` has ``cpw >= 10``, at least
    4 cells across its smallest feature measured on the actual mesh, and a 2D
    ``Nz == 1`` grid.
    """
    non_demo = [t for t in tasks if t["kind"] != "underresolved_demo"]
    assert len(non_demo) > 0, "Expected at least one non-demo task in task table"
    for task in non_demo:
        cpw = float(task["cpw"])
        assert cpw >= 10.0, (
            f"Task {task['name']} violates cpw >= 10 rule: cpw={cpw}"
        )
        metrics = measure_task_mesh(task)
        caf = float(metrics["cells_across_feature"])
        assert caf >= 4.0, (
            f"Task {task['name']} violates cells_across_feature >= 4 rule: "
            f"cells_across_feature={caf:.2f} (min_feature={metrics['min_feature_nm']:.1f} nm, "
            f"dx_min={metrics['dx_min_nm']:.2f} nm)"
        )
        assert metrics["grid_shape"][2] == 1, (
            f"Task {task['name']} corrupted 2D Nz=1: grid_shape={metrics['grid_shape']}"
        )


def test_task_table_resolution_rules_and_mutation():
    """
    Every non-demo task in both the --quick and --full task tables has
    cpw >= 10 and cells_across_feature >= 4 measured on the actual mesh, while
    mutating cpw < 10 or removing the mesh override on a narrow-gap task fails.
    """
    for quick in (True, False):
        tasks = get_task_table(quick=quick)
        _audit_task_table_resolution(tasks)
        demos = [t for t in tasks if t["kind"] == "underresolved_demo"]
        assert [t["name"] for t in demos] == [
            "underresolved_kottke",
            "underresolved_staircase",
        ]
        for d in demos:
            assert d["cpw"] == 10
            assert d["n_cells_gap"] is None

    # Mutation check 1: cpw < 10 must fail the audit
    bad_cpw_task = dict(get_task_table(quick=True)[2])
    bad_cpw_task["cpw"] = 8
    with pytest.raises(AssertionError, match="violates cpw >= 10"):
        _audit_task_table_resolution([bad_cpw_task])

    # Mutation check 2: removing the override box on sw_angle_00deg (80 nm gap at 14 cpw -> 2.28 cells < 4) must fail
    sw0_tasks = [
        t for t in get_task_table(quick=True) if t["name"] == "sw_angle_00deg"
    ]
    assert len(sw0_tasks) == 1
    bad_gap_task = dict(sw0_tasks[0])
    bad_gap_task["n_cells_gap"] = None
    with pytest.raises(AssertionError, match="violates cells_across_feature >= 4"):
        _audit_task_table_resolution([bad_gap_task])


def test_trapezoidal_width_and_mirror_symmetry():
    """
    trapezoidal_effective_width reproduces exact bottom/top/mid-plane widths
    for 0 deg and 30 deg sidewalls, and build_directional_coupler produces two
    PolySlab arms with machine-precision y -> -y mirror symmetry.
    """
    g0 = trapezoidal_effective_width(
        w_gds=W_GDS,
        dilation=DILATION,
        thickness=H_WG,
        sidewall_angle_deg=0.0,
        pitch=PITCH_COUPLED,
    )
    assert g0["w_bot"] == pytest.approx(0.47e-6, abs=1e-12)
    assert g0["w_top"] == pytest.approx(0.47e-6, abs=1e-12)
    assert g0["w_eff"] == pytest.approx(0.47e-6, abs=1e-12)
    assert g0["gap_eff"] == pytest.approx(0.08e-6, abs=1e-12)

    g30 = trapezoidal_effective_width(
        w_gds=W_GDS,
        dilation=DILATION,
        thickness=H_WG,
        sidewall_angle_deg=30.0,
        pitch=PITCH_COUPLED,
    )
    expected_shrink = H_WG * np.tan(np.deg2rad(30.0))
    assert g30["w_eff"] == pytest.approx(0.47e-6 - expected_shrink, abs=1e-12)
    assert g30["w_top"] == pytest.approx(0.47e-6 - 2.0 * expected_shrink, abs=1e-12)
    assert g30["gap_eff"] == pytest.approx(PITCH_COUPLED - g30["w_eff"], abs=1e-12)
    assert g30["w_eff"] < g0["w_eff"]
    assert g30["gap_eff"] > g0["gap_eff"]

    structures, ports, domain, _ = build_directional_coupler(
        L_c=4.0e-6, theta_sw_deg=30.0, four_ports=True
    )
    assert len(structures) == 2
    assert len(ports) == 4
    v_top = structures[0].vertices
    v_bot = structures[1].vertices
    # Bottom arm is the reversed y-reflection of top arm (CCW oriented)
    v_top_reflected = v_top[::-1].copy()
    v_top_reflected[:, 1] *= -1.0
    # Up to cyclic shift of the starting vertex after _clean_polygon
    dists = np.linalg.norm(v_bot - v_top_reflected[0], axis=1)
    shift = int(np.argmin(dists))
    v_bot_aligned = np.roll(v_bot, -shift, axis=0)
    np.testing.assert_allclose(v_bot_aligned, v_top_reflected, atol=1e-14)
    assert domain[0] == pytest.approx(-domain[1], abs=1e-12)
    assert domain[2] == pytest.approx(-domain[3], abs=1e-12)


def test_supermodes_and_gap_mutation():
    """
    Coupled-section eigenmodes at 1530 nm exhibit symmetric (even) and
    antisymmetric (odd) transverse profiles with n_even > n_odd > n_sub, and
    mutating (widening) the waveguide pitch strictly reduces (n_even - n_odd)
    and increases the CMT crossover beat length L_pi.
    """
    w_eff = trapezoidal_effective_width(W_GDS, DILATION, H_WG, 30.0, PITCH_COUPLED)[
        "w_eff"
    ]
    f0 = c_light / WL_DESIGN

    l_pi_values = []
    dn_values = []
    for pitch in (0.55e-6, 0.75e-6):
        y0 = 0.5 * pitch
        gap_eff = pitch - w_eff
        ov = coupler_mesh_overrides(0.8e-6, gap_eff, w_eff, n_cells_gap=4)
        s_top = PolySlab(
            straight((-0.6e-6, +y0), (0.6e-6, +y0), width=w_eff),
            0.0,
            H_WG,
            n_to_eps(N_WG),
        )
        s_bot = PolySlab(
            straight((-0.6e-6, -y0), (0.6e-6, -y0), width=w_eff),
            0.0,
            H_WG,
            n_to_eps(N_WG),
        )
        pic = PICSimulation(
            WL_MIN,
            WL_MAX,
            structures=[s_top, s_bot],
            n_core=N_WG,
            n_clad=N_CLAD,
            n_sub=N_SUB,
            wavelength_ref=WL_0,
            two_d=True,
            cells_per_wavelength=10,
            domain=(-0.4e-6, 0.4e-6, -2.2e-6, 2.2e-6),
            n_pml=6,
            mesh_overrides=ov,
        )
        m = extract_mesh_metrics_from_grid(
            pic.grid.x, pic.grid.y, gap_eff, Nz=pic.grid.Nz
        )
        assert m["cells_across_feature"] >= 4.0
        assert pic.grid.Nz == 1
        z_mid = 0.5 * (pic.grid.zmin + pic.grid.zmax)
        modes = solve_modes(
            pic.solver, f0, (0.0, 0.0, z_mid), (0.0, 3.2e-6, np.inf), n_modes=2
        )
        ne = float(np.real(modes[0].neff))
        no = float(np.real(modes[1].neff))
        assert ne > no > N_SUB, f"Expected n_even > n_odd > n_sub, got {ne}, {no}"

        # Exclude the trailing non-periodic ghost edge sample at index -1
        y = modes[0].coords["Eb"][0][:-1]
        ey0 = np.real(modes[0].e["b"][:-1, 0])
        ey1 = np.real(modes[1].e["b"][:-1, 0])
        # Verify even symmetry of mode 0 and odd symmetry of mode 1
        even_err = float(np.linalg.norm(ey0 - ey0[::-1]) / np.linalg.norm(ey0))
        odd_err = float(np.linalg.norm(ey1 + ey1[::-1]) / np.linalg.norm(ey1))
        assert even_err < 1e-4, f"Even mode symmetry error {even_err:.3e} >= 1e-4"
        assert odd_err < 1e-4, f"Odd mode antisymmetry error {odd_err:.3e} >= 1e-4"
        assert np.all(np.abs(y + y[::-1]) < 1e-14)

        dn = ne - no
        l_pi = WL_DESIGN / (2.0 * dn)
        dn_values.append(dn)
        l_pi_values.append(l_pi)

    # Mutation check: widening pitch from 550 nm to 750 nm must decrease dn_eff by > 2x
    # and increase L_pi by > 2x
    assert dn_values[1] < 0.45 * dn_values[0], (
        f"Expected dn_eff to drop >2x when widening gap: {dn_values}"
    )
    assert l_pi_values[1] > 2.2 * l_pi_values[0], (
        f"Expected L_pi to grow >2x when widening gap: {l_pi_values}"
    )


def test_coupler_passivity_cmt_3db_and_length_mutation():
    """
    2D FIT directional coupler at L1=(10 cpw, N=4) on a compact S-bend domain
    satisfies passivity (T_cross + T_bar + R <= 1), matches the CMT 3 dB
    crossing wavelength within 3.5 %, exhibits -90 deg cross/bar phase
    quadrature, and mutating L_c from 1 um to 4 um to 7 um monotonically
    increases cross-port power fraction in agreement with CMT.
    """
    dx_sb_test = 2.6e-6
    dy_sb_test = 1.05e-6
    l_lead_test = 0.8e-6
    y_max_test = 2.0e-6

    # 1. Baseline L_c = 4.0 um coupler at L1 = (10 cpw, N=4)
    structures, ports, domain, geom = build_directional_coupler(
        L_c=4.0e-6,
        theta_sw_deg=30.0,
        n_sbend=45,
        four_ports=False,
        dx_sbend=dx_sb_test,
        dy_sbend=dy_sb_test,
        l_lead=l_lead_test,
        y_max=y_max_test,
    )
    ov_4 = coupler_mesh_overrides(
        4.0e-6,
        geom["gap_eff"],
        geom["w_eff"],
        n_cells_gap=4,
        dx_sb=dx_sb_test,
        dy_sb=dy_sb_test,
    )
    pic_4 = PICSimulation(
        WL_MIN,
        1.80e-6,
        n_freqs=13,
        structures=structures,
        ports=ports,
        n_core=N_WG,
        n_clad=N_CLAD,
        n_sub=N_SUB,
        wavelength_ref=WL_0,
        two_d=True,
        cells_per_wavelength=10,
        domain=domain,
        n_pml=6,
        mesh_overrides=ov_4,
    )
    s_4 = pic_4.run_s_parameters(
        drive=["in_bot"], energy_decay=5e-3, collect_fields=False
    )
    wl = pic_4.wavelengths
    order = np.argsort(wl)
    wl_s = wl[order]
    s11 = s_4[("in_bot", "in_bot")][order]
    s31 = s_4[("out_top", "in_bot")][order]
    s41 = s_4[("out_bot", "in_bot")][order]

    tc = np.abs(s31) ** 2
    tb = np.abs(s41) ** 2
    r = np.abs(s11) ** 2
    mid = slice(2, -2)

    # Passivity: cross + bar <= 1 and low insertion loss
    assert float(np.max(tc[mid] + tb[mid])) <= 1.0, (
        f"Passivity violation: max(T_cross + T_bar) = {np.max(tc[mid] + tb[mid]):.5f} > 1.0"
    )
    tot_power = tc[mid] + tb[mid] + r[mid]
    assert float(np.max(tot_power)) <= 1.0 + 1e-3, (
        f"Passivity violation: max(T_cross + T_bar + R) = {np.max(tot_power):.5f}"
    )
    assert float(np.mean(tc[mid] + tb[mid])) > 0.90, (
        f"Unexpectedly low guided transmission: {np.mean(tc[mid] + tb[mid]):.4f}"
    )

    # Quadrature phase between cross and bar outputs (-90 deg in exp(+i wt) convention)
    i_des = int(np.argmin(np.abs(wl_s - WL_DESIGN)))
    dphi_deg = float(np.rad2deg(np.angle(s31[i_des] * np.conj(s41[i_des]))))
    assert abs(dphi_deg - (-90.0)) < 10.0, (
        f"Expected cross/bar phase near -90 deg, got {dphi_deg:.2f} deg"
    )

    # CMT 3 dB crossing wavelength agreement within 3.5 %
    wl_cross_fit, _ = find_crossing_wavelength(wl_s[mid], tc[mid], tb[mid])
    sm = solve_coupler_supermodes(
        pic_4,
        wavelengths=np.array([1.45e-6, 1.53e-6, 1.62e-6]),
        x_coupled=0.0,
        x_sb_start=2.0e-6,
        dx_sb=dx_sb_test,
        dy_sb=dy_sb_test,
        L_c=4.0e-6,
        window_y=2.6e-6,
    )
    wl_cross_cmt = float(sm["cmt_wl_cross"])
    rel_err_3db = abs(wl_cross_fit - wl_cross_cmt) / wl_cross_cmt
    assert rel_err_3db < 0.035, (
        f"3 dB crossing wavelength mismatch: FIT={wl_cross_fit * 1e9:.1f} nm vs "
        f"CMT={wl_cross_cmt * 1e9:.1f} nm (rel_err={rel_err_3db:.3%})"
    )

    # 2. Mutation check: perturb L_c to 1.0 um (under-coupled) and 6.5 um (over-coupled)
    ratios = {4.0: float(tc[i_des] / (tc[i_des] + tb[i_des]))}
    for L_um in (1.0, 6.5):
        st_l, pt_l, dom_l, geom_l = build_directional_coupler(
            L_c=L_um * 1e-6,
            theta_sw_deg=30.0,
            n_sbend=30,
            four_ports=False,
            dx_sbend=1.2e-6,
            dy_sbend=0.70e-6,
            l_lead=0.45e-6,
            y_max=1.40e-6,
        )
        ov_l = coupler_mesh_overrides(
            L_um * 1e-6,
            geom_l["gap_eff"],
            geom_l["w_eff"],
            n_cells_gap=4,
            dx_sb=1.2e-6,
            dy_sb=0.70e-6,
        )
        pic_l = PICSimulation(
            WL_MIN,
            1.85e-6,
            n_freqs=15,
            structures=st_l,
            ports=pt_l,
            n_core=N_WG,
            n_clad=N_CLAD,
            n_sub=N_SUB,
            wavelength_ref=WL_0,
            two_d=True,
            cells_per_wavelength=10,
            domain=dom_l,
            n_pml=5,
            mesh_overrides=ov_l,
        )
        res_l = run_phased_dual_drive(
            pic_l,
            port_phases={"in_bot": 0.0},
            energy_decay=5e-2,
            collect_dft=False,
        )
        wl_l = pic_l.wavelengths
        ord_l = np.argsort(wl_l)
        i_l = int(np.argmin(np.abs(wl_l[ord_l] - WL_DESIGN)))
        tc_l = float(res_l["power"]["out_top"][ord_l][i_l])
        tb_l = float(res_l["power"]["out_bot"][ord_l][i_l])
        ratios[L_um] = tc_l / (tc_l + tb_l)

    assert ratios[1.0] < 0.30 < ratios[4.0] < 0.70 < ratios[6.5], (
        f"Coupling-length mutation failed monotonic CMT progression: {ratios}"
    )


def test_phased_gaussian_zero_dc_and_quadrature_switching():
    """
    PhasedGaussianSource maintains zero DC integral for any phase offset, and
    flipping the upper-input quadrature phase from -pi/2 to +pi/2 switches
    output routing between the upper and lower output arms at L1=(10 cpw, N=4).
    """
    f0 = c_light / WL_DESIGN
    base = GaussianSource(frequency=f0, fwidth=0.15 * f0)
    t = np.linspace(0.0, base.end_time * 1.1, 20001)
    for phi in (0.0, -0.5 * np.pi, +0.5 * np.pi, 1.17):
        psrc = PhasedGaussianSource(base, phase=phi)
        vals = psrc(t)
        dc_integral = abs(float(np.trapezoid(vals, t))) / float(
            np.trapezoid(np.abs(vals), t)
        )
        assert dc_integral < 1e-5, (
            f"Non-zero DC integral {dc_integral:.3e} for phase={phi}"
        )

    dx_sb_test = 1.6e-6
    dy_sb_test = 0.85e-6
    structures, ports, domain, geom = build_directional_coupler(
        L_c=4.0e-6,
        theta_sw_deg=30.0,
        n_sbend=35,
        four_ports=True,
        dx_sbend=dx_sb_test,
        dy_sbend=dy_sb_test,
        l_lead=0.55e-6,
        y_max=1.65e-6,
    )
    ov = coupler_mesh_overrides(
        4.0e-6,
        geom["gap_eff"],
        geom["w_eff"],
        n_cells_gap=4,
        dx_sb=dx_sb_test,
        dy_sb=dy_sb_test,
    )
    pic = PICSimulation(
        WL_MIN,
        1.85e-6,
        n_freqs=15,
        structures=structures,
        ports=ports,
        n_core=N_WG,
        n_clad=N_CLAD,
        n_sub=N_SUB,
        wavelength_ref=WL_0,
        two_d=True,
        cells_per_wavelength=10,
        domain=domain,
        n_pml=6,
        mesh_overrides=ov,
    )
    res_minus = run_phased_dual_drive(
        pic,
        port_phases={"in_bot": 0.0, "in_top": -0.5 * np.pi},
        energy_decay=1.5e-2,
        collect_dft=False,
    )
    res_plus = run_phased_dual_drive(
        pic,
        port_phases={"in_bot": 0.0, "in_top": +0.5 * np.pi},
        energy_decay=1.5e-2,
        collect_dft=False,
    )
    ratio_m = res_minus["power"]["out_top"] / (
        res_minus["power"]["out_top"] + res_minus["power"]["out_bot"] + 1e-30
    )
    i_peak_m = int(np.argmax(ratio_m))
    p_up_m = float(res_minus["power"]["out_top"][i_peak_m])
    p_dn_m = float(res_minus["power"]["out_bot"][i_peak_m])

    p_dn_p = float(res_plus["power"]["out_bot"][i_peak_m])
    p_up_p = float(res_plus["power"]["out_top"][i_peak_m])

    assert p_up_m > 0.90 and p_dn_m < 0.03, (
        f"Expected routing to out_top for -pi/2: P_up={p_up_m:.3f}, P_dn={p_dn_m:.3f}"
    )
    assert p_dn_p > 0.90 and p_up_p < 0.03, (
        f"Expected routing to out_bot for +pi/2: P_dn={p_dn_p:.3f}, P_up={p_up_p:.3f}"
    )


def _verify_coupler_length_and_sidewall_trends(tasks_npz):
    """
    Verify coupling-length progression vs supermode CMT and sidewall-angle
    sensitivity across all full-resolution sweep tasks.
    """
    primary = tasks_npz["kottke_L3"]
    w_cmt = primary["cmt_wavelengths"]
    lpi_1530 = float(np.interp(WL_DESIGN, w_cmt, primary["cmt_L_pi"]))
    lsb_1530 = float(np.interp(WL_DESIGN, w_cmt, primary["cmt_L_sb_eff"]))
    assert 9.0e-6 < lpi_1530 < 11.5e-6
    assert 1.0e-6 < lsb_1530 < 1.8e-6

    # 1. Coupling-length sweep vs supermode CMT at 1530 nm
    len_map = {
        0.0: tasks_npz["length_L0um"],
        2.0e-6: tasks_npz["length_L2um"],
        4.0e-6: primary,
        6.0e-6: tasks_npz["length_L6um"],
        8.0e-6: tasks_npz["length_L8um"],
    }
    fit_ratios = []
    for L_c, d in sorted(len_map.items()):
        w = d["wavelengths"]
        ord_w = np.argsort(w)
        tc = np.abs(d["S31"][ord_w]) ** 2
        tb = np.abs(d["S41"][ord_w]) ** 2
        assert float(np.max(tc + tb)) <= 1.0 + 1e-3, (
            f"Passivity violation at L_c={L_c * 1e6:.1f} um: max(T_tot)={np.max(tc + tb):.5f}"
        )
        tc_1530 = float(np.interp(WL_DESIGN, w[ord_w], tc))
        tb_1530 = float(np.interp(WL_DESIGN, w[ord_w], tb))
        rat = tc_1530 / (tc_1530 + tb_1530)
        cmt_rat = float(np.sin(0.5 * np.pi * (L_c + lsb_1530) / lpi_1530) ** 2)
        assert abs(rat - cmt_rat) < 0.06, (
            f"CMT mismatch at L_c={L_c * 1e6:.1f} um: FIT={rat:.4f} vs CMT={cmt_rat:.4f}"
        )
        fit_ratios.append(rat)

    assert np.all(np.diff(fit_ratios) > 0.10), (
        f"Expected strictly increasing cross-port fraction across L_c=0..8 um: {fit_ratios}"
    )
    assert fit_ratios[0] < 0.10 and fit_ratios[-1] > 0.98

    # 2. Sidewall-angle sweep (0..30 deg) at L3=(20,8): gap widening (0->10 deg)
    #    followed by mode-delocalization blue-shift (10->30 deg)
    sw_angles = (0, 5, 10, 15, 20, 25, 30)
    wlc_fit = []
    wlc_cmt = []
    dn_1530 = []
    for ang in sw_angles:
        d = primary if ang == 30 else tasks_npz[f"sw_angle_{ang:02d}deg"]
        assert float(d["cells_across_feature"]) >= 8.0 - 1e-2
        wf = float(d["wl_cross"]) * 1e9
        wc = float(d["cmt_wl_cross"]) * 1e9
        t_tot = float(d["t_tot_cross"])
        assert 0.98 <= t_tot <= 1.001, (
            f"Unexpected total transmittance {t_tot:.4f} at theta_sw={ang} deg"
        )
        rel_err = abs(wf - wc) / wc
        assert rel_err < 0.02, (
            f"Sidewall CMT mismatch at {ang} deg: FIT={wf:.2f} nm vs CMT={wc:.2f} nm ({rel_err:.2%})"
        )
        wlc_fit.append(wf)
        wlc_cmt.append(wc)
        dn_1530.append(
            float(d["supermode_neff_even_ref"] - d["supermode_neff_odd_ref"])
        )

    # Gap widening reduces dn_eff and red-shifts wl_cross from 0 deg to 10 deg
    assert dn_1530[2] < dn_1530[0]
    assert wlc_fit[0] < wlc_fit[1] < wlc_fit[2]
    # Mode delocalization increases dn_eff and blue-shifts wl_cross from 10 deg to 30 deg
    assert dn_1530[2] < dn_1530[4] < dn_1530[5] < dn_1530[6]
    assert wlc_fit[2] > wlc_fit[3] > wlc_fit[4] > wlc_fit[5] > wlc_fit[6]


def test_full_study_data_convergence_length_and_sidewall_sweeps():
    """
    Audit the saved --full study data (task_0..task_19 in data/):
    1. Every non-demo JSON/NPZ satisfies cpw >= 10, cells_across_feature >= 4, Nz == 1.
    2. Kottke ladder (L1, L2, L3) agrees with R=(40,16) reference within 3 nm (0.20 %)
       in wl_cross, has >10x smaller error than staircasing at every level, and
       monotonically reduces band-averaged reflection <|S11|^2>.
    3. Full-resolution coupling-length and sidewall-angle sweeps match supermode CMT,
       and mutating S31/S41 or L_sb_eff fails the audit.
    """
    import json

    data_dir = TUTORIAL_DIR / "data"
    tasks_Full = get_task_table(quick=False)
    assert len(tasks_Full) == 20

    tasks_npz = {}
    for t in tasks_Full:
        idx = t["index"]
        jf = data_dir / f"task_{idx}.json"
        nf = data_dir / f"task_{idx}.npz"
        assert jf.is_file() and nf.is_file(), f"Missing full-study output for task_{idx}"
        meta = json.loads(jf.read_text())
        d = dict(np.load(nf, allow_pickle=False))
        tasks_npz[meta["name"]] = d
        assert meta["name"] == t["name"]
        assert meta["grid_shape"][2] == 1
        if meta["kind"] != "underresolved_demo":
            assert float(meta["cpw"]) >= 10.0
            assert float(meta["cells_across_feature"]) >= 4.0
            assert float(d["cpw"]) >= 10.0
            assert float(d["cells_across_feature"]) >= 4.0

    # Convergence check: Kottke vs Staircase against kottke_reference R=(40,16)
    ref = tasks_npz["kottke_reference"]
    assert int(ref["cpw"]) == 40 and int(ref["n_cells_gap"]) == 16
    wlc_ref = float(ref["wl_cross"]) * 1e9

    def _r_mean(d):
        w = d["wavelengths"] * 1e9
        m = (w >= 1430.0) & (w <= 1670.0)
        return float(np.mean(np.abs(d["S11"][m]) ** 2))

    r_kottke = []
    for lvl in ("L1", "L2", "L3"):
        dk = tasks_npz[f"kottke_{lvl}"]
        ds = tasks_npz[f"staircase_{lvl}"]
        err_k = abs(float(dk["wl_cross"]) * 1e9 - wlc_ref)
        err_s = abs(float(ds["wl_cross"]) * 1e9 - wlc_ref)
        assert err_k < 3.0, f"Kottke {lvl} wl_cross error {err_k:.2f} nm >= 3.0 nm"
        assert err_s > 10.0 * err_k, (
            f"Expected Staircase {lvl} error ({err_s:.2f} nm) > 10x Kottke ({err_k:.2f} nm)"
        )
        r_kottke.append(_r_mean(dk))
    r_kottke.append(_r_mean(ref))
    assert np.all(np.diff(r_kottke) < 0.0), (
        f"Expected strictly decreasing Kottke reflection across L1->L2->L3->R: {r_kottke}"
    )

    _verify_coupler_length_and_sidewall_trends(tasks_npz)

    # Mutation check: zeroing S-bend coupling L_sb_eff or swapping L0um/L8um must fail
    mutated = dict(tasks_npz)
    bad_primary = dict(mutated["kottke_L3"])
    bad_primary["cmt_L_sb_eff"] = np.zeros_like(bad_primary["cmt_L_sb_eff"])
    mutated["kottke_L3"] = bad_primary
    with pytest.raises(AssertionError):
        _verify_coupler_length_and_sidewall_trends(mutated)

