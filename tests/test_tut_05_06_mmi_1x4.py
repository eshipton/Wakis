# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics and numerical validation tests for Tutorial 5.6:
1x4 Multimode Interference (MMI) Power Splitter in a Si-GaN-Si Slot Waveguide.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "05_guided_waves"
    / "05-06_mmi_1x4"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    equivalent_3layer_core_index,
    mmi_analytic_estimates,
    mmi_discretization_diagnostics,
    mmi_eme_length_sweep,
    slot_slab_neff,
    vertical_slot_mode,
)
from mmi_1x4 import (  # noqa: E402
    H_GAN,
    H_SI,
    H_STACK,
    L1,
    L2,
    L_LEAD_IN,
    L_LEAD_OUT,
    L_MMI_OPT,
    N_CORE_EQ,
    N_EFF_SLOT,
    N_GAN,
    N_SI,
    N_SIO2,
    PITCH,
    SWEEP_L_UM,
    SWEEP_L_UM_INIT,
    SWEEP_W_UM,
    SWEEP_W_UM_INIT,
    W1,
    W2,
    W_MMI_OPT,
    WL_MAX,
    WL_MIN,
    WL_REF,
    Y_OUT_CENTERS,
    build_mmi_1x4,
    get_task_table,
)
from wakis.templates.photonics import PICSimulation, slab_neff  # noqa: E402


def test_slot_eim_and_34_vertex_geometry():
    """
    Verify the 5-layer SiO2/Si/GaN/Si/SiO2 vertical-slot TM mode effective
    index, the dielectric-continuity E_z jump at the GaN slot interfaces, and
    the 34-vertex polygon area and mirror symmetry.
    """
    assert N_GAN < N_EFF_SLOT < N_SI
    assert N_EFF_SLOT == pytest.approx(2.984168, rel=1e-4)

    neff_check = slab_neff(
        N_CORE_EQ, N_SIO2, H_STACK, WL_REF, polarization="TE"
    )
    assert neff_check == pytest.approx(N_EFF_SLOT, rel=1e-6)

    vm = vertical_slot_mode(
        n_rail=N_SI,
        n_slot=N_GAN,
        n_clad=N_SIO2,
        h_rail=H_SI,
        h_slot=H_GAN,
        wavelength=WL_REF,
        z_span=1.6e-6,
        n_pts=1601,
    )
    z = vm["z"]
    ez = vm["ez"]
    # Just inside the GaN slot (|z| = 48 nm) vs just inside the Si rail (|z| = 52 nm),
    # |E_z| jumps by (n_Si / n_GaN)^2 = (3.48 / 2.305)^2 = 2.2794
    i_slot = int(np.argmin(np.abs(z - 48.0e-9)))
    i_rail = int(np.argmin(np.abs(z - 52.0e-9)))
    jump_ratio = float(ez[i_slot] / ez[i_rail])
    expected_jump = (N_SI / N_GAN) ** 2
    assert jump_ratio == pytest.approx(expected_jump, rel=2e-2)
    assert vm["stack_confinement"] > 0.90

    structures, ports, _ = build_mmi_1x4(L_MMI_OPT, W_MMI_OPT, include_mode1_port=True)
    assert len(structures) == 1
    assert len(ports) == 6
    verts = structures[0].vertices
    assert verts.shape == (34, 2)

    # Exact polygon area vs sum of constituent rectangles and trapezoids
    area_shoelace = 0.5 * abs(
        float(
            np.sum(
                verts[:, 0] * np.roll(verts[:, 1], -1)
                - np.roll(verts[:, 0], -1) * verts[:, 1]
            )
        )
    )
    area_exact = (
        L_LEAD_IN * W1
        + L1 * 0.5 * (W1 + W2)
        + L_MMI_OPT * W_MMI_OPT
        + 4.0 * L2 * 0.5 * (W2 + W1)
        + 4.0 * L_LEAD_OUT * W1
    )
    assert area_shoelace == pytest.approx(area_exact, rel=1e-12)


def test_self_imaging_beat_length_and_eme_sweep():
    """
    Compare the Soldano & Pennings (1995) analytical beat length L_pi and 4-fold
    imaging length L_MMI = 3*L_pi/16 with the 1D EME length sweep, and verify by
    mutation that widening W_MMI shifts L_opt upward as W_e^2.
    """
    est = mmi_analytic_estimates(
        w_mmi=W_MMI_OPT, n_r=N_EFF_SLOT, n_clad=N_SIO2, wavelength=WL_REF, n_splits=4
    )
    eme = mmi_eme_length_sweep(
        w_mmi=W_MMI_OPT,
        w_taper=W2,
        pitch=PITCH,
        n_r=N_EFF_SLOT,
        n_clad=N_SIO2,
        wavelength=WL_REF,
    )

    # Beat length L_pi = 4 * n_r * W_e^2 / (3 * lambda_0) agrees with exact m=0,1 mode beat length
    assert est["l_pi_tm"] == pytest.approx(eme["L_pi_exact"], rel=2e-2)
    # 4-fold symmetric imaging length 3*L_pi/16 agrees with non-paraxial EME optimal length within 7%
    assert est["l_mmi_tm"] == pytest.approx(eme["l_opt_eme"], rel=7e-2)
    assert float(np.max(eme["t_tot"])) > 0.94

    # Mutation check on W_MMI: widening from 4.90 um to 5.50 um increases L_opt and
    # degrades coupling at L_MMI = 11.15 um
    eme_wide = mmi_eme_length_sweep(
        w_mmi=5.50e-6,
        w_taper=W2,
        pitch=PITCH,
        n_r=N_EFF_SLOT,
        n_clad=N_SIO2,
        wavelength=WL_REF,
        lengths=np.linspace(10.0e-6, 16.0e-6, 241),
    )
    assert eme_wide["l_opt_eme"] > eme["l_opt_eme"] * 1.20
    i_design = int(np.argmin(np.abs(eme["lengths"] - L_MMI_OPT)))
    i_design_wide = int(np.argmin(np.abs(eme_wide["lengths"] - L_MMI_OPT)))
    assert eme_wide["t_tot"][i_design_wide] < eme["t_tot"][i_design] - 0.30


def test_mmi_1x4_fit_symmetry_passivity_and_length_mutation():
    """
    Full-wave 2D EIM FIT verification of the 1x4 MMI splitter:
    1. Pairwise mirror symmetry: |T1 - T4| < 1e-6 and |T2 - T3| < 1e-6.
    2. Passivity: T1 + T2 + T3 + T4 + R <= 1 + 1e-3 and 2D flux box ~ 1.0.
    3. 4-fold self-imaging intensity peaks in the four output arms and > 97% Mode 0 share.
    4. Mutation check: halving L_MMI to 5.58 um (2-fold image between output ports)
       reduces total guided output transmission by > 0.20 in 2D FIT.
    """
    structures, ports, domain = build_mmi_1x4(
        L_MMI_OPT, W_MMI_OPT, include_mode1_port=True
    )
    pic = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=21,
        structures=structures,
        ports=ports,
        n_core=N_CORE_EQ,
        n_clad=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=6,
        domain=domain,
        n_pml=8,
        cfln=0.85,
        field_wavelengths=(WL_REF,),
    )
    s_mat = pic.run_s_parameters(drive=["in"], energy_decay=5e-3, collect_fields=True)

    mid = slice(3, -3)
    t1 = np.abs(s_mat[("out1", "in")])[mid] ** 2
    t2 = np.abs(s_mat[("out2", "in")])[mid] ** 2
    t3 = np.abs(s_mat[("out3", "in")])[mid] ** 2
    t4 = np.abs(s_mat[("out4", "in")])[mid] ** 2
    r = np.abs(s_mat[("in", "in")])[mid] ** 2
    t_tot = t1 + t2 + t3 + t4

    # 1. Pairwise mirror symmetry (outer pair out1/out4, inner pair out2/out3)
    assert float(np.max(np.abs(t1 - t4))) < 1e-6
    assert float(np.max(np.abs(t2 - t3))) < 1e-6

    # 2. Passivity and 2D Poynting flux conservation
    assert float(np.max(t_tot + r)) <= 1.0 + 1e-3
    assert pic.flux_balance is not None
    np.testing.assert_allclose(pic.flux_balance["total_box"][mid], 1.0, atol=0.05)

    # 3. Mode 0 vs Mode 1 power share at outer output port (> 97% in Mode 0)
    i0 = int(np.argmin(np.abs(pic.wavelengths - WL_REF)))
    p_m0 = float(np.abs(s_mat[("out4", "in")][i0]) ** 2)
    p_m1 = float(np.abs(s_mat[("out4_m1", "in")][i0]) ** 2)
    mode0_share = p_m0 / max(p_m0 + p_m1, 1e-12)
    assert mode0_share > 0.97

    # 4. 4-fold self-imaging intensity profile across the four output channels
    fmap = pic.dft_fields[WL_REF]
    ix_img = int(np.argmin(np.abs(fmap["x"] - (L1 + L_MMI_OPT + L2 + 0.8e-6))))
    y_prof = fmap["y"]
    e2_prof = fmap["E2"][ix_img, :]
    gap = np.abs(y_prof) <= 0.15e-6
    for y_c in Y_OUT_CENTERS:
        win = np.abs(y_prof - y_c) <= 0.35e-6
        assert float(np.max(e2_prof[win])) > 2.2 * float(np.mean(e2_prof[gap]))

    # 5. Full-wave FIT mutation check: halving L_MMI to 5.58 um (2-fold image plane,
    # where peaks land at y = +-1.25 um in the oxide gaps between output tapers)
    # sharply reduces guided 4-port transmission
    structs_mut, ports_mut, dom_mut = build_mmi_1x4(
        0.5 * L_MMI_OPT, W_MMI_OPT, include_mode1_port=False
    )
    pic_mut = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=21,
        structures=structs_mut,
        ports=ports_mut,
        n_core=N_CORE_EQ,
        n_clad=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=6,
        domain=dom_mut,
        n_pml=8,
        cfln=0.85,
    )
    s_mut = pic_mut.run_s_parameters(
        drive=["in"], energy_decay=5e-3, collect_fields=False
    )
    t_tot_opt_1550 = float(
        sum(np.abs(s_mat[(f"out{k}", "in")][i0]) ** 2 for k in range(1, 5))
    )
    t_tot_mut_1550 = float(
        sum(np.abs(s_mut[(f"out{k}", "in")][i0]) ** 2 for k in range(1, 5))
    )
    imb_opt = abs(
        float(np.abs(s_mat[("out2", "in")][i0]) ** 2)
        - float(np.abs(s_mat[("out1", "in")][i0]) ** 2)
    )
    imb_mut = abs(
        float(np.abs(s_mut[("out2", "in")][i0]) ** 2)
        - float(np.abs(s_mut[("out1", "in")][i0]) ** 2)
    )
    assert t_tot_mut_1550 < t_tot_opt_1550 - 0.14
    assert imb_mut > 10.0 * imb_opt


def test_discretization_diagnostics_and_extended_sweep_table():
    """
    Verify staircase width quantization and numerical-dispersion diagnostics at
    10, 14, and 20 cpw, check that the extended --full task table preserves
    tasks 0..15 while placing the (5.00, 11.20) um 3x3 corner optimum in the
    interior of the 5x6 grid, and verify by mutation that refining dx by 2x
    reduces dispersion phase error 4x.
    """
    # 1. Staircase width quantization & Yee dispersion at 10, 14, 20 cpw
    d10 = mmi_discretization_diagnostics(
        dx=50.239234e-9, dy=50.000000e-9, ny=159, w_mmi=W_MMI_OPT, l_mmi=L_MMI_OPT
    )
    d14 = mmi_discretization_diagnostics(
        dx=35.852090e-9, dy=35.751295e-9, ny=214, w_mmi=W_MMI_OPT, l_mmi=L_MMI_OPT
    )
    d20 = mmi_discretization_diagnostics(
        dx=25.119617e-9, dy=25.090909e-9, ny=296, w_mmi=W_MMI_OPT, l_mmi=L_MMI_OPT
    )

    # At 10 cpw (dy = 50.00 nm), 98 Ey cells match 4.9000 um exactly, while 99 Ex nodes give 4.9500 um
    assert d10["n_ey"] == 98
    assert d10["w_stair_ey"] == pytest.approx(4.9000e-6, abs=1e-10)
    assert d10["w_stair_ex"] == pytest.approx(4.9500e-6, abs=1e-10)
    assert d10["phase_err_rad"] == pytest.approx(2.076, rel=1e-2)
    assert d10["image_shift_disp"] == pytest.approx(0.1716e-6, rel=1e-2)

    # At 20 cpw (dy = 25.09 nm), staircase Ey width error (-7.3 nm) is larger in magnitude than at 14 cpw (-2.1 nm)
    assert d14["n_ey"] == 137
    assert d20["n_ey"] == 195
    assert abs(d20["dw_ey"]) > 3.0 * abs(d14["dw_ey"])
    assert d20["w_stair_ey"] == pytest.approx(4.8927e-6, abs=1e-10)
    assert d20["dl_pi_ey"] == pytest.approx(-0.184e-6, rel=2e-2)

    # Mutation check: halving dx reduces quadratic dispersion phase error and image shift by 4x,
    # and perturbing w_mmi by +0.10 um increases n_ey by 4 cells at 20 cpw
    assert d20["phase_err_rad"] == pytest.approx(0.25 * d10["phase_err_rad"], rel=1e-3)
    assert d20["image_shift_disp"] == pytest.approx(
        0.25 * d10["image_shift_disp"], rel=1e-3
    )
    d20_wide = mmi_discretization_diagnostics(
        dx=25.119617e-9,
        dy=25.090909e-9,
        ny=300,
        w_mmi=W_MMI_OPT + 0.10e-6,
        l_mmi=L_MMI_OPT,
    )
    assert d20_wide["n_ey"] == d20["n_ey"] + 4
    assert d20_wide["w_stair_ey"] > d20["w_stair_ey"] + 0.095e-6

    # 2. Task table backward compatibility (tasks 0..15) and extended 5x6 sweep (tasks 16..36)
    tasks_quick = get_task_table(quick=True)
    tasks_full = get_task_table(quick=False)
    assert len(tasks_quick) == 16
    assert len(tasks_full) == 37
    for i in range(16):
        assert tasks_full[i]["index"] == i
        assert tasks_full[i]["name"] == tasks_quick[i]["name"]
    assert tasks_full[15]["name"] == "sweep_W5.00_L11.20"
    assert tasks_full[16]["name"] == "sweep_W4.80_L11.25"
    assert tasks_full[36]["name"] == "sweep_W5.20_L11.35"

    # The 3x3 corner optimum (W=5.00, L=11.20) is strictly interior to SWEEP_W_UM x SWEEP_L_UM
    assert min(SWEEP_W_UM) < 5.00 < max(SWEEP_W_UM)
    assert min(SWEEP_L_UM) < 11.20 < max(SWEEP_L_UM)
    assert len(SWEEP_W_UM_INIT) == 3 and len(SWEEP_L_UM_INIT) == 3

    # 3. Verify full 37-task dataset in data/: resolution, convergence, and interior 5x6 sweep optima
    data_dir = TUTORIAL_DIR / "data"
    by_name = {}
    for i in range(37):
        d = dict(np.load(data_dir / f"task_{i}.npz", allow_pickle=False))
        assert float(d["cpw"]) >= 10.0
        by_name[str(d["name"])] = d

    imb_mat = np.zeros((len(SWEEP_W_UM), len(SWEEP_L_UM)))
    el_mat = np.zeros((len(SWEEP_W_UM), len(SWEEP_L_UM)))
    for iw, w_um in enumerate(SWEEP_W_UM):
        for il, l_um in enumerate(SWEEP_L_UM):
            d_sw = by_name[f"sweep_W{w_um:.2f}_L{l_um:.2f}"]
            i0 = int(np.argmin(np.abs(d_sw["wavelengths"] - WL_REF)))
            t_out = float(np.abs(d_sw["S21"][i0]) ** 2)
            t_in = float(np.abs(d_sw["S31"][i0]) ** 2)
            imb_mat[iw, il] = abs(t_in - t_out)
            el_mat[iw, il] = -10.0 * np.log10(2.0 * (t_in + t_out))

    iw_imb, il_imb = np.unravel_index(int(np.argmin(imb_mat)), imb_mat.shape)
    iw_el, il_el = np.unravel_index(int(np.argmin(el_mat)), el_mat.shape)
    # Both global optima (W=5.00, L=11.30 min imbalance; W=5.00, L=11.20 min excess loss) are strictly interior
    assert 0 < iw_imb < len(SWEEP_W_UM) - 1 and 0 < il_imb < len(SWEEP_L_UM) - 1
    assert 0 < iw_el < len(SWEEP_W_UM) - 1 and 0 < il_el < len(SWEEP_L_UM) - 1
    assert (SWEEP_W_UM[iw_imb], SWEEP_L_UM[il_imb]) == (5.00, 11.30)
    assert (SWEEP_W_UM[iw_el], SWEEP_L_UM[il_el]) == (5.00, 11.20)
    assert imb_mat[iw_imb, il_imb] == pytest.approx(0.00271, rel=2e-2)
    assert el_mat[iw_el, il_el] == pytest.approx(0.325, rel=1e-2)

