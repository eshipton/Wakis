"""Tests for Tutorial 6.6: Valley-Hall Photonic Crystal & Topological Edge States."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wakis.lattices import (
    HexagonalLattice,
    PWE2D,
    dirac_berry_curvature,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = (
    REPO_ROOT / "tutorials" / "06_periodic_structures" / "06-06_valley_hall_edge_states"
)
SCRIPT_PATH = TUTORIAL_DIR / "valley_hall_edge_states.py"
NOTEBOOK_PATH = TUTORIAL_DIR / "valley_hall_edge_states.ipynb"
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"

if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as hlp  # noqa: E402


# ---------------------------------------------------------------------------
# Explicit assertion helpers (used for positive checks and mutation checks)
# ---------------------------------------------------------------------------


def _assert_unfolding_reproduces_primitive(
    f_unfolded: np.ndarray,
    f_primitive: np.ndarray,
    tol: float = 1.0e-5,
) -> None:
    """Assert that unfolded supercell bands match primitive-cell bands."""
    max_err = float(np.max(np.abs(f_unfolded - f_primitive)))
    assert max_err < tol, (
        f"Unfolded supercell bands deviate from primitive bands by {max_err:.3e} (> {tol:.3e})"
    )


def _assert_berry_sign_flip(
    c_pos: float,
    c_neg: float,
    c_zero: float,
    min_mag: float = 0.05,
    c_ref_pos: float | None = None,
    rtol_ref: float = 0.05,
) -> None:
    """Assert that valley Chern number flips sign under delta -> -delta, vanishes at delta = 0, and matches reference."""
    assert c_pos < -min_mag, f"Expected negative valley Chern at K for delta > 0, got {c_pos:+.4f}"
    assert c_neg > +min_mag, f"Expected positive valley Chern at K for delta < 0, got {c_neg:+.4f}"
    assert c_pos * c_neg < 0.0, "Valley Chern number did not flip sign with delta"
    assert abs(c_pos + c_neg) / max(abs(c_pos), 1e-12) < 0.05, (
        f"Valley Chern magnitudes are not antisymmetric: {c_pos:+.4f} vs {c_neg:+.4f}"
    )
    assert abs(c_zero) < 1.0e-3, (
        f"Inversion-symmetric (delta=0) control should have zero Berry flux, got {c_zero:+.4e}"
    )
    if c_ref_pos is not None:
        rel_err = abs(c_pos - c_ref_pos) / max(abs(c_ref_pos), 1e-12)
        assert rel_err < rtol_ref, (
            f"Patch Chern {c_pos:+.4f} deviates from reference {c_ref_pos:+.4f} by {rel_err:.2%} (>= {rtol_ref:.2%})"
        )


def _assert_bend_transmission_advantage(
    mean_t_valley: float,
    mean_t_trivial: float,
    min_valley_t: float = 0.75,
    min_ratio: float = 4.5,
    t_valley_arr: np.ndarray | None = None,
    t_trivial_arr: np.ndarray | None = None,
    disc_tol: float = 0.026,
) -> None:
    """Assert the bend/straight flux ratio <= 1 + disc_tol and the valley-Hall Z-bend advantage.

    The reference is the transmitted flux of the matched straight guide, so T_straight == 1
    holds by construction (those two asserts are identity checks, not validation).

    Discretization tolerance justification:
    Between the coarse (16x28, max T_VH = 0.9998, <T>_up = 0.9927, R_up = 15.09, R_gap = 6.07)
    and production (22x36, max T_VH = 1.0154, <T>_up = 1.0102, R_up = 14.08, R_gap = 6.45)
    waveguide grids, the grid-refinement shift on normalized transmission is |Delta T| = 0.0175
    (Roache GCI with safety factor Fs = 1.5 gives disc_tol = 1.5 * 0.0175 = 0.026) and the
    relative shift on the bend-to-straight ratio is 5.9%-7.2% (giving >= 30% GCI margin at
    min_ratio = 4.5 across the full bulk gap and min_ratio = 9.0 in the upper single-mode window).
    """
    assert mean_t_valley <= 1.0 + disc_tol, (
        f"Mean valley-Hall bend transmission {mean_t_valley:.4f} exceeds 1 + disc_tol ({1.0 + disc_tol:.4f})"
    )
    assert mean_t_trivial <= 1.0 + disc_tol, (
        f"Mean trivial bend transmission {mean_t_trivial:.4f} exceeds 1 + disc_tol ({1.0 + disc_tol:.4f})"
    )
    if t_valley_arr is not None:
        arr_v = np.asarray(t_valley_arr, dtype=float)
        assert float(np.min(arr_v)) >= 0.0, f"Negative transmission in valley bend: {np.min(arr_v):.4f}"
        assert float(np.max(arr_v)) <= 1.0 + disc_tol, (
            f"Peak valley-Hall bend transmission {np.max(arr_v):.4f} exceeds 1 + disc_tol ({1.0 + disc_tol:.4f})"
        )
    if t_trivial_arr is not None:
        arr_t = np.asarray(t_trivial_arr, dtype=float)
        assert float(np.min(arr_t)) >= 0.0, f"Negative transmission in trivial bend: {np.min(arr_t):.4f}"
        assert float(np.max(arr_t)) <= 1.0 + disc_tol, (
            f"Peak trivial bend transmission {np.max(arr_t):.4f} exceeds 1 + disc_tol ({1.0 + disc_tol:.4f})"
        )
    assert mean_t_valley >= min_valley_t, (
        f"Valley-Hall 120-deg Z-bend transmission {mean_t_valley:.3f} < {min_valley_t:.3f}"
    )
    assert mean_t_valley >= min_ratio * max(mean_t_trivial, 1e-12), (
        f"Valley-Hall bend transmission ({mean_t_valley:.3f}) does not exceed trivial bend "
        f"({mean_t_trivial:.3f}) by factor >= {min_ratio:.1f}"
    )


# ---------------------------------------------------------------------------
# 1. Supercell Unfolding & Dirac / Valley-Hall Dispersion Tests
# ---------------------------------------------------------------------------


def test_tutorial_unfolding_reproduces_primitive_bands_and_mutation() -> None:
    """Verify supercell unfolding along Gamma-M-K-Gamma and mutation failure under phase sign flip."""
    lat = HexagonalLattice(a=hlp.DEFAULT_A_M)
    path = lat.bz_path(["Gamma", "M", "K", "Gamma"], n_per_segment=4)
    r_a = hlp.DEFAULT_R0_OVER_A * (1.0 + hlp.DEFAULT_DELTA)
    r_b = hlp.DEFAULT_R0_OVER_A * (1.0 - hlp.DEFAULT_DELTA)

    pwe_prim = PWE2D(
        lat.a1,
        lat.a2,
        lat.honeycomb_primitive_sites(r_a, r_b, hlp.DEFAULT_EPS_ROD),
        eps_bg=1.0,
        n_pw=4,
        polarization="TM",
    )
    f_prim = pwe_prim.solve_path(path.k_points, n_bands=4, lattice_a=hlp.DEFAULT_A_M)["frequencies"]

    A1, A2 = lat.supercell_vectors
    pwe_sup = PWE2D(
        A1,
        A2,
        lat.honeycomb_supercell_sites(r_a, r_b, hlp.DEFAULT_EPS_ROD),
        eps_bg=1.0,
        n_pw=4,
        polarization="TM",
    )
    unf = pwe_sup.solve_path(
        path.k_points, n_bands=8, unfold_supercell=True, lattice_a=hlp.DEFAULT_A_M, phase_sign=1
    )
    _assert_unfolding_reproduces_primitive(unf["unfolded_frequencies"], f_prim, tol=1.0e-5)

    # Mutation check: flipping phase_sign = -1 in the primitive translation projector
    # selects the folded complement branches at M and K and must fail primitive comparison.
    unf_mut = pwe_sup.solve_path(
        path.k_points, n_bands=8, unfold_supercell=True, lattice_a=hlp.DEFAULT_A_M, phase_sign=-1
    )
    with pytest.raises(AssertionError):
        _assert_unfolding_reproduces_primitive(unf_mut["unfolded_frequencies"], f_prim, tol=1.0e-5)


def test_tutorial_berry_curvature_sign_flip_and_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Berry curvature & valley Chern sign reversal and code-path mutation failure."""
    lat = HexagonalLattice(a=hlp.DEFAULT_A_M)
    k_K = lat.high_symmetry_points()["K"]
    q_grid = np.linspace(-0.08 * 2.0 * np.pi, 0.08 * 2.0 * np.pi, 3)
    kx_g = k_K[0] + q_grid
    ky_g = k_K[1] + q_grid
    dq = float(q_grid[1] - q_grid[0])
    qc = 0.5 * (q_grid[:-1] + q_grid[1:])
    QX, QY = np.meshgrid(qc, qc, indexing="ij")

    # 1. Analytical massive Dirac theory
    om_d_pos = dirac_berry_curvature(QX, QY, v_d=0.17, delta_omega=+0.0344 * 2.0 * np.pi, valley=1)
    om_d_neg = dirac_berry_curvature(QX, QY, v_d=0.17, delta_omega=-0.0344 * 2.0 * np.pi, valley=1)
    c_d_pos = float(np.sum(om_d_pos) * dq * dq / (2.0 * np.pi))
    c_d_neg = float(np.sum(om_d_neg) * dq * dq / (2.0 * np.pi))
    _assert_berry_sign_flip(c_d_pos, c_d_neg, 0.0)

    # 2. 2D PWE Fukui-Hatsugai-Suzuki plaquette phase
    r0 = hlp.DEFAULT_R0_OVER_A
    d0 = hlp.DEFAULT_DELTA
    pwe_pos = PWE2D(
        lat.a1,
        lat.a2,
        lat.honeycomb_primitive_sites(r0 * (1.0 + d0), r0 * (1.0 - d0), hlp.DEFAULT_EPS_ROD),
        eps_bg=1.0,
        n_pw=4,
        polarization="TM",
    )
    pwe_neg = PWE2D(
        lat.a1,
        lat.a2,
        lat.honeycomb_primitive_sites(r0 * (1.0 - d0), r0 * (1.0 + d0), hlp.DEFAULT_EPS_ROD),
        eps_bg=1.0,
        n_pw=4,
        polarization="TM",
    )
    _, c_pos = pwe_pos.berry_curvature(kx_g, ky_g, band_index=0)
    _, c_neg = pwe_neg.berry_curvature(kx_g, ky_g, band_index=0)
    _assert_berry_sign_flip(c_pos, c_neg, 0.0, c_ref_pos=c_d_pos, rtol_ref=0.15)

    # Code-path mutation check: dropping complex conjugation in np.vdot inside
    # berry_curvature_from_envelopes breaks FHS gauge invariance and must fail the reference check.
    with monkeypatch.context() as m:
        m.setattr(np, "vdot", lambda a, b: np.sum(np.asarray(a) * np.asarray(b)))
        _, c_mut = pwe_pos.berry_curvature(kx_g, ky_g, band_index=0)
    with pytest.raises(AssertionError):
        _assert_berry_sign_flip(c_mut, c_neg, 0.0, c_ref_pos=c_pos, rtol_ref=0.05)


def test_tutorial_ribbon_domain_wall_mode_and_mutation() -> None:
    """Verify domain-wall and tuned trivial ribbon guided modes, derived windows, and mutation check."""
    rib = hlp.run_ribbon_dispersion_study(quick=True)
    m = rib["metrics"]
    arr = rib["arrays"]

    # Verify programmatically derived ribbon windows partition the bulk valley gap
    bg0, bg1 = m["bulk_gap_window"]
    lw0, lw1 = m["lower_window"]
    mg0, mg1 = m["ribbon_minigap_window"]
    uw0, uw1 = m["upper_single_mode_window"]
    assert 0.270 < bg0 < 0.275 and 0.304 < bg1 < 0.310
    assert bg0 == pytest.approx(lw0) and lw1 == pytest.approx(mg0)
    assert lw0 < lw1 < mg1 < uw0 < uw1 < bg1

    # Both valley-Hall and tuned trivial guided modes at projected K must lie inside the valley gap [bg0, bg1]
    assert bg0 < m["boundary_wall_freq_at_K_norm"] < m["edge_freq_at_K_norm"] < m["interior_wall_freq_at_K_norm"] < bg1
    assert m["edge_confinement_fraction"] > 0.80
    assert bg0 < m["trivial_freq_at_K_norm"] < bg1
    assert m["trivial_confinement_fraction"] > 0.80
    assert m["trivial_core_offset_over_a"] == pytest.approx(0.25 * np.sqrt(3.0), rel=1e-6)

    # Transverse profile must peak near the interior domain wall y = L_rib / 2
    y_over_a = arr["y_line_over_a"]
    prof_y = arr["edge_profile_y"]
    ly_rib = arr["ly_rib_over_a"]
    y_peak = float(y_over_a[int(np.argmax(prof_y))])
    assert abs(y_peak - 0.5 * ly_rib) < 0.8

    # Mutation check: a uniform delocalized bulk profile across the ribbon must fail confinement > 0.80
    prof_uniform = np.ones_like(prof_y)
    dist_mid = np.abs(y_over_a - 0.5 * ly_rib)
    dist_edge = np.minimum(y_over_a, ly_rib - y_over_a)
    mask_wall = (dist_mid < 1.2 * np.sqrt(3.0)) | (dist_edge < 1.2 * np.sqrt(3.0))
    conf_mut = float(np.sum(prof_uniform[mask_wall]) / np.sum(prof_uniform))
    with pytest.raises(AssertionError):
        assert conf_mut > 0.80


def test_tutorial_precomputed_json_table_and_bend_advantage() -> None:
    """Verify precomputed JSON metrics, Quantitative Summary table synchronization, and figures."""
    assert SCRIPT_PATH.exists(), f"Missing tutorial script {SCRIPT_PATH}"
    assert NOTEBOOK_PATH.exists(), f"Missing notebook {NOTEBOOK_PATH}"

    json_path = DATA_DIR / "valley_hall_results.json"
    npz_path = DATA_DIR / "valley_hall_arrays.npz"
    assert json_path.exists(), f"Missing {json_path}"
    assert npz_path.exists(), f"Missing {npz_path}"

    results = json.loads(json_path.read_text(encoding="utf-8"))
    npz_data = np.load(npz_path)
    b = results["bulk_and_berry"]
    w = results["waveguide_bend"]
    c = results["convergence"]

    # 1. Verify unfolding and FIT vs PWE accuracy
    assert b["unfolding_max_error"] < 1.0e-5
    assert b["fit_band1_rel_err"] < 0.01
    assert b["fit_band2_rel_err"] < 0.01

    # 2. Verify Berry curvature sign flip and < 5% magnitude match between 2D FIT and PWE
    _assert_berry_sign_flip(
        b["c_patch_pwe_K_pos"], b["c_patch_pwe_K_neg"], b["c_patch_fit_K_zero"]
    )
    _assert_berry_sign_flip(
        b["c_patch_fit_K_pos"],
        b["c_patch_fit_K_neg"],
        b["c_patch_fit_K_zero"],
        c_ref_pos=b["c_patch_pwe_K_pos"],
        rtol_ref=0.05,
    )
    # Short ring-down (n_steps=1600 -> -0.2382) must fail the < 5% PWE magnitude check
    with pytest.raises(AssertionError):
        _assert_berry_sign_flip(
            b["c_patch_fit_vs_nsteps"][0],
            b["c_patch_fit_K_neg"],
            b["c_patch_fit_K_zero"],
            c_ref_pos=b["c_patch_pwe_K_pos"],
            rtol_ref=0.05,
        )

    # 3. Identity check T_straight == 1 (by construction) and bend/straight ratio <= 1 + disc_tol
    # and 120-deg Z-bend valley-Hall advantage across both upper single-mode window and full bulk gap
    assert w["mean_transmission_valley_straight"] == pytest.approx(1.0, abs=1e-6)
    assert w["mean_transmission_trivial_straight"] == pytest.approx(1.0, abs=1e-6)
    t_vh_arr = np.asarray(npz_data["t_valley_bend"], dtype=float)
    t_tr_arr = np.asarray(npz_data["t_trivial_bend"], dtype=float)

    # Upper single-mode window [0.2915, 0.3033]: min_ratio = 9.0 (> 35% GCI margin below 14.08x)
    _assert_bend_transmission_advantage(
        w["mean_transmission_valley_bend"],
        w["mean_transmission_trivial_bend"],
        min_valley_t=0.85,
        min_ratio=9.0,
        t_valley_arr=t_vh_arr,
        t_trivial_arr=t_tr_arr,
    )
    # Full bulk valley gap [0.2715, 0.3059]: min_ratio = 4.5 (30% GCI margin below 6.45x)
    _assert_bend_transmission_advantage(
        w["mean_transmission_valley_bend_full_gap"],
        w["mean_transmission_trivial_bend_full_gap"],
        min_valley_t=0.75,
        min_ratio=4.5,
        t_valley_arr=t_vh_arr,
        t_trivial_arr=t_tr_arr,
    )
    # Mutation 1: swapping valley and trivial bend transmission must fail
    with pytest.raises(AssertionError):
        _assert_bend_transmission_advantage(
            w["mean_transmission_trivial_bend"],
            w["mean_transmission_valley_bend"],
        )
    # Mutation 2: unphysical T > 1 + disc_tol normalization (e.g. dividing by uncalibrated near-field flux) must fail
    with pytest.raises(AssertionError):
        _assert_bend_transmission_advantage(
            w["mean_transmission_valley_bend"],
            w["mean_transmission_trivial_bend"],
            t_valley_arr=1.08 * t_vh_arr,
            t_trivial_arr=t_tr_arr,
        )

    # 4. Verify grid resolution and convergence requirements (>= 20 cells/lambda, >= 4 cells/diam)
    assert c["cells_per_wavelength_finest"] >= 20.0
    assert c["cells_across_min_diameter_finest"] >= 4.0
    assert c["richardson_vs_pwe_rel_err"] < 0.005
    assert 1.3 < c["observed_order_p"] < 2.6

    # 5. Verify Quantitative Summary markdown table matches build_quantitative_summary_table(results)
    expected_table = hlp.build_quantitative_summary_table(results)
    assert results["summary_table_markdown"] == expected_table
    for val_str in (
        f"{b['dirac_freq_norm']:.4f}",
        f"{b['gap_bottom_pwe']:.4f}",
        f"{b['fit_band1_K']:.4f}",
        f"{w['mean_transmission_valley_bend']:.3f}",
        f"{w['mean_transmission_trivial_bend']:.3f}",
    ):
        assert val_str in expected_table

    # 6. Verify all setup and result figures and GIF exist
    for fname in (
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00a_setup_geometry_supercell.png",
        "00b_setup_mesh_and_excitation_supercell.png",
        "01_supercell_unfolding_and_dirac_gap.png",
        "02_berry_curvature_and_valley_chern.png",
        "03_ribbon_domain_wall_dispersion.png",
        "04_bend_transmission_and_convergence.png",
        "05_bend_field_maps.png",
        "06_valley_hall_bend_wavefront.gif",
    ):
        fpath = FIG_DIR / fname
        assert fpath.exists() and fpath.stat().st_size > 5_000, f"Missing or empty figure {fpath}"
