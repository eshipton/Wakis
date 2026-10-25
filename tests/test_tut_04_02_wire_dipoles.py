# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Regression and physics validation tests for Tutorial 4.2 (Half-Wave Dipole and Quarter-Wave Monopole)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import wakis
from wakis.lumped import LumpedPort
from wakis.sources import GaussianSource

TUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "04_radiation_and_antennas"
    / "04-02_wire_dipoles"
)
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _helpers as h  # noqa: E402

DATA_DIR = TUT_DIR / "data"
FIG_DIR = TUT_DIR / "figures"


@pytest.fixture(scope="module", autouse=True)
def _ensure_tutorial_data_and_figures() -> None:
    """Ensure Tutorial 4.2 task archives and figures exist before running assertions."""
    missing_data = [
        name for name in h.TASKS if not (DATA_DIR / f"task_{name}.npz").exists()
    ]
    if missing_data:
        h.run_tasks(missing_data, DATA_DIR, quick=True)
    if not (FIG_DIR / "00a_setup_geometry.png").exists():
        h.generate_all_figures(DATA_DIR, FIG_DIR)


def test_setup_schematic_and_figures_exist() -> None:
    """Verify that the setup figures (00a/00b + 00_setup_table.md) and all Tutorial 4.2 figures exist."""
    expected = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "01_geometry_and_graded_mesh.png",
        "02_current_distribution_wire.png",
        "03_impedance_s11_and_shortening.png",
        "04_subcell_vs_resolved_convergence.png",
        "05_monopole_vs_dipole_image_theory.png",
        "06_near_field_maps_dipole_monopole.png",
        "07_far_field_3d_and_finite_ground.png",
        "08_wire_antenna_radiation.gif",
    ]
    for fname in expected:
        fpath = FIG_DIR / fname
        assert fpath.exists(), f"Missing Tutorial 4.2 figure: {fpath}"
        size_bytes = fpath.stat().st_size
        assert size_bytes > 10_000, f"Figure {fname} is unexpectedly small ({size_bytes} B)"
        if fname.endswith(".gif"):
            assert size_bytes < 5 * 1024 * 1024, f"GIF exceeds 5 MB limit ({size_bytes} B)"

    table_md = FIG_DIR / "00_setup_table.md"
    assert table_md.exists() and table_md.stat().st_size > 200, f"Missing setup table: {table_md}"

    from PIL import Image

    with Image.open(FIG_DIR / "08_wire_antenna_radiation.gif") as im:
        assert im.n_frames <= 60, f"Expected <= 60 GIF frames, got {im.n_frames}"
        assert im.size[0] <= 900, f"Expected GIF width <= 900 px, got {im.size[0]}"

    est = h.preflight_estimates()
    for key in (
        "graded_subcell",
        "graded_resolved",
        "uniform_subcell_cpw24",
        "uniform_resolved_4cell",
    ):
        assert key in est
        assert est[key]["cells"] > 0
    assert est["uniform_resolved_4cell"]["cells"] > 100 * est["graded_resolved"]["cells"]


def test_half_wave_dipole_impedance_current_and_pattern() -> None:
    """Validate half-wave dipole Z_in(f), resonant shortening, I(z) circulation, and Near2Far pattern."""
    d0 = np.load(DATA_DIR / "task_half_wave_thin_wire_radii.npz")
    freqs_ghz = np.asarray(d0["freqs_ghz"], dtype=float)
    idx_f0 = int(np.argmin(np.abs(freqs_ghz - 1.0)))

    # 1. Induced-EMF benchmark at L = lam0 / 2 (73.08 + j 42.51 Ohm)
    assert abs(float(d0["z_emf_real"][idx_f0]) - 73.08) < 0.10
    assert abs(float(d0["z_emf_imag"][idx_f0]) - 42.52) < 0.10

    # 2. Resonant length shortening L_res / lam0 < 0.50 and agreement with Hallen MoM
    l_res_fit = np.asarray(d0["l_res_fit_over_lam0"], dtype=float)
    l_res_mom = np.asarray(d0["l_res_mom_over_lam0"], dtype=float)
    assert np.all((l_res_fit > 0.44) & (l_res_fit < 0.50))
    assert l_res_fit[-1] > l_res_fit[0], "Thinner wire (larger Omega) must resonate closer to 0.50*lam0"
    np.testing.assert_allclose(l_res_fit, l_res_mom, rtol=0.025)

    # 3. Resonance-trimmed dipole (L = 0.47*lam0) reduces |X_in(f0)| and improves return loss
    x_untrimmed_f0 = abs(float(d0["z_fit_imag"][1, idx_f0]))
    x_trimmed_f0 = abs(float(d0["z_trim_fit_imag"][idx_f0]))
    assert x_trimmed_f0 < 0.5 * x_untrimmed_f0
    assert float(np.min(d0["s11_trim_73_db"])) < -15.0
    assert float(np.min(d0["s11_trim_50_db"])) < -11.0

    # 4. Wire current distribution I(z) from FIT magnetic circulation vs sinusoidal & MoM
    i_seg = np.asarray(d0["i_seg_per_v_real"], dtype=float) + 1j * np.asarray(
        d0["i_seg_per_v_imag"], dtype=float
    )
    i_norm = np.abs(i_seg / i_seg[len(i_seg) // 2])
    assert abs(i_norm[len(i_norm) // 2] - 1.0) < 1e-12
    assert i_norm[0] < 0.30 and i_norm[-1] < 0.30

    # 5. Far-field directivity D0 ~ 1.643 (2.15 dBi) and HPBW ~ 78 deg
    d0_fit = float(d0["d0_fit"])
    hpbw_deg = float(d0["hpbw_deg"])
    assert 1.58 <= d0_fit <= 1.72
    assert 74.0 <= hpbw_deg <= 82.0


def test_resolved_wire_and_monopole_image_theory_and_finite_ground() -> None:
    """Validate subcell vs. resolved 4x4 wire, monopole image theory, and finite ground diffraction."""
    d1 = np.load(DATA_DIR / "task_resolved_wire_vs_subcell_mesh.npz")
    d2 = np.load(DATA_DIR / "task_quarter_wave_monopole_infinite_gnd.npz")
    d3 = np.load(DATA_DIR / "task_finite_ground_plane_sweep.npz")

    # 1. Subcell thin wire agrees closely with 4x4-cell resolved PEC wire (~9.6% across band due to 3D flat end-cap capacitance),
    # whereas correction=False shifts X_in relative to the corrected subcell wire
    rel_diff_sub_vs_res = float(d1["rel_diff_res_vs_sub"])
    assert rel_diff_sub_vs_res < 0.12
    z_sub = np.asarray(d1["z_sub_real"]) + 1j * np.asarray(d1["z_sub_imag"])
    z_unc = np.asarray(d1["z_unc_real"]) + 1j * np.asarray(d1["z_unc_imag"])
    assert float(np.max(np.abs(z_unc.imag - z_sub.imag))) > 5.0

    # Axial segmentation convergence
    err_seg = np.asarray(d1["err_l2_seg"], dtype=float)
    assert err_seg[-1] < err_seg[0]

    # 2. Quarter-wave monopole over infinite PEC ground: Z_mono = 0.5 * Z_dip (<6% across 0.75-1.25 GHz) and D0_mono = 2 * D0_dip (+3 dB)
    image_z_err = float(d2["image_z_rel_err"])
    assert image_z_err < 0.06, f"Expected |2*Z_mono - Z_dip|/|Z_dip| < 6%, got {image_z_err:.4f}"
    d0_ratio_db = float(d2["d0_ratio_db"])
    assert abs(d0_ratio_db - 3.0103) < 0.15

    # 3. Finite PEC ground plane tilts main lobe upward (theta_max < 90 deg) and leaks power into lower hemisphere
    theta_max_deg = np.asarray(d3["theta_max_deg"], dtype=float)
    back_power_frac = np.asarray(d3["back_power_frac"], dtype=float)
    assert theta_max_deg[1] < 72.0 and theta_max_deg[2] < 72.0
    assert np.all(back_power_frac > 0.15)
    assert back_power_frac[0] > back_power_frac[1]


def test_analytical_helpers_and_mutation_check() -> None:
    """Verify analytical helpers and run a physics mutation check on grounded thin-wire termination."""
    omega = h.thickness_parameter_omega(0.15, 0.5e-3)
    assert abs(omega - 2.0 * np.log(300.0)) < 1e-12

    th = np.linspace(0.01, np.pi - 0.01, 181)
    d_dip = h.half_wave_analytic_directivity(th, mono_half_space=False)
    d_mono = h.half_wave_analytic_directivity(th, mono_half_space=True)
    assert abs(float(np.max(d_dip)) - 1.64092) < 1e-3
    assert abs(float(np.max(d_mono)) - 2.0 * 1.64092) < 2e-3
    assert np.all(d_mono[th > 0.5 * np.pi + 1e-6] == 0.0)
    assert np.any(d_dip[th > 0.5 * np.pi + 0.1] > 0.5)

    # Mutation check: omitting the z=0 PEC ground sheet registration before add_thin_wire
    # causes add_thin_wire to treat the ground junction as an open-space wire tip (deg=1),
    # doubling the base cell inductance and breaking image theory Z_mono = 0.5 * Z_dip.
    grid_dip, grid_mono, meta = h.build_image_theory_pair_grids(h.H_QUARTER_M, n_half=6, quick=True)
    dz_w = meta["dz_w"]
    src = GaussianSource(frequency=h.F0_HZ, fwidth=h.FWIDTH_HZ)

    # Grounded monopole WITH z=0 PEC sheet registered before add_thin_wire
    p_ok = LumpedPort(src_time=src, component="z", center=(0.0, 0.0, 0.5 * dz_w), size=(0.0, 0.0, 0.5 * dz_w))
    s_ok = wakis.SolverFIT3D(
        grid_mono, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    s_ok.add_sheet(plane=("z", 0.0), sigma_s="pec")
    s_ok.add_thin_wire([(0.0, 0.0, 0.0), (0.0, 0.0, h.H_QUARTER_M)], radius=1.0e-3, gap=p_ok)
    n_steps = int(np.ceil(2.8 / (h.F0_HZ * s_ok.dt)))
    s_ok.emsolve(n_steps, source=p_ok)
    z_ok = complex(p_ok.impedance_in([h.F0_HZ])[0])

    # Mutated monopole WITHOUT z=0 PEC sheet registered before add_thin_wire
    g_mut = wakis.GridFIT3D(
        x=np.asarray(grid_mono.x), y=np.asarray(grid_mono.y), z=np.asarray(grid_mono.z), use_mpi=False, verbose=0
    )
    p_mut = LumpedPort(src_time=src, component="z", center=(0.0, 0.0, 0.5 * dz_w), size=(0.0, 0.0, 0.5 * dz_w))
    s_mut = wakis.SolverFIT3D(
        g_mut, bc_low=["cpml", "cpml", "pec"], bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    s_mut.add_thin_wire([(0.0, 0.0, 0.0), (0.0, 0.0, h.H_QUARTER_M)], radius=1.0e-3, gap=p_mut)
    s_mut.emsolve(n_steps, source=p_mut)
    z_mut = complex(p_mut.impedance_in([h.F0_HZ])[0])

    # The mutated base-cell inductance inflates Im(Z_mono) by > 10 Ohm
    assert abs(z_mut.imag - z_ok.imag) > 10.0, (
        f"Expected ungrounded-tip mutation to shift monopole reactance by >10 Ohm, "
        f"got z_ok={z_ok:.2f}, z_mut={z_mut:.2f}"
    )
