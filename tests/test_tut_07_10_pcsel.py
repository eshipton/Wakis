# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Verification and mutation tests for Tutorial 7.10:
Photonic-crystal surface-emitting laser (PCSEL): band-edge modes and 3D coupled-wave theory.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TUT_DIR = REPO_ROOT / "tutorials" / "07_light_sources" / "07-10_pcsel"
if str(TUT_DIR) not in sys.path:
    sys.path.insert(0, str(TUT_DIR))

import _helpers as h  # noqa: E402


def _validate_summary_table_against_json(md_text: str, summary: dict) -> None:
    """Validate that numbers in the Quantitative Summary markdown table match data/pcsel_summary.json."""
    m = re.search(
        r"## Quantitative Summary\s*\n(.*?)\n# ---",
        md_text,
        flags=re.DOTALL,
    )
    assert m is not None, "Could not locate Quantitative Summary section in pcsel.py."
    table_block = m.group(1)

    s1 = summary["slab_1d"]
    cwt = summary["cwt_infinite_r018"]
    fit = summary["fit_unit_cell_r018"]
    rich = summary["convergence"]["richardson"]
    r_rows = {round(r["r_over_a"], 2): r for r in summary["radius_sweep"]["fit_vs_cwt"]}
    dl = summary["double_lattice"]
    fin_rows = {int(round(r["L_um"])): r for r in summary["finite_cwt_sweep"]}
    f2d_8 = next(r for r in summary["finite_2d_fit"] if r["n_periods"] == 8)

    expected_tokens = [
        f"{s1['n_eff_exact']:.4f}",
        f"{s1['n_eff_fd']:.4f}",
        f"{s1['f_bragg_thz']:.2f}",
        f"{s1['gamma_phc'] * 100:.2f}",
        f"{s1['gamma_act'] * 100:.2f}",
        f"{s1['optical_theorem_lhs']:.6f}",
        f"{cwt['freq_A_thz']:.3f}",
        f"{fit['freq_A_thz']:.3f}",
        f"{rich['freq_A_thz']['richardson']:.3f}",
        f"{fit['Q_A']:.1f}",
        f"{fit['rel_err_cwt_vs_fit_A_pct']:.3f}",
        f"{cwt['freq_B_thz']:.3f}",
        f"{cwt['freq_B_with_longitudinal_thz']:.3f}",
        f"{fit['freq_B_thz']:.3f}",
        f"{rich['freq_B_thz']['richardson']:.3f}",
        f"{fit['Q_B']:.1f}",
        f"{fit['rel_err_cwt_vs_fit_B_pct']:.3f}",
        f"{fit['rel_err_cwt_longitudinal_vs_fit_B_pct']:.3f}",
        f"{cwt['delta_f_AB_thz']:.3f}",
        f"{cwt['delta_f_AB_with_longitudinal_thz']:.3f}",
        f"{fit['delta_f_AB_thz']:.3f}",
        f"{cwt['freq_C_thz']:.3f}",
        f"{fit['freq_CD_thz']:.3f}",
        f"{rich['freq_C_thz']['richardson']:.3f}",
        f"{fit['rel_err_cwt_vs_fit_CD_pct']:.3f}",
        f"{cwt['Q_C_dressed']:.1f}",
        f"{cwt['Q_C_unperturbed']:.1f}",
        f"{fit['Q_CD']:.1f}",
        f"{rich['Q_CD']['richardson']:.1f}",
        f"{rich['freq_A_thz']['rel_err_coarse_pct']:.3f}",
        f"{rich['freq_A_thz']['rel_err_medium_pct']:.3f}",
        f"{rich['freq_A_thz']['rel_err_fine_pct']:.3f}",
        f"{r_rows[0.15]['cwt_freq_CD_thz']:.3f}",
        f"{r_rows[0.15]['fit_freq_CD_thz']:.3f}",
        f"{r_rows[0.15]['cwt_Q_CD_dressed']:.1f}",
        f"{r_rows[0.15]['fit_Q_CD']:.1f}",
        f"{r_rows[0.15]['rel_err_Q_dressed_pct']:.2f}",
        f"{r_rows[0.26]['cwt_freq_CD_thz']:.3f}",
        f"{r_rows[0.26]['fit_freq_CD_thz']:.3f}",
        f"{r_rows[0.26]['cwt_Q_CD_dressed']:.1f}",
        f"{r_rows[0.26]['fit_Q_CD']:.1f}",
        f"{r_rows[0.26]['rel_err_Q_dressed_pct']:.1f}",
        f"{dl['opt_d_over_a']:.4f}",
        f"{dl['opt_r2_over_r1']:.3f}",
        f"{dl['single_abs_C21_cm']:.1f}",
        f"{dl['double_max_abs_C_offdiag_cm']:.1f}",
        f"{dl['double_alpha_v_inf_cm']:.2f}",
        f"{dl['crossover_L_um_vs_asym_single']:.1f}",
        f"{fin_rows[50]['single_matched_rel_margin_pct']:.2f}",
        f"{fin_rows[50]['double_rel_margin_pct']:.2f}",
        f"{fin_rows[50]['matched_rel_margin_enhancement_ratio']:.2f}",
        f"{fin_rows[500]['single_matched_rel_margin_pct']:.2f}",
        f"{fin_rows[500]['double_rel_margin_pct']:.1f}",
        f"{fin_rows[500]['matched_rel_margin_enhancement_ratio']:.1f}",
        f"{fin_rows[1000]['single_matched_rel_margin_pct']:.3f}",
        f"{fin_rows[1000]['double_rel_margin_pct']:.1f}",
        f"{fin_rows[1000]['matched_rel_margin_enhancement_ratio']:.1f}",
        f"{fin_rows[3000]['single_matched_rel_margin_pct']:.4f}",
        f"{fin_rows[3000]['double_rel_margin_pct']:.2f}",
        f"{fin_rows[3000]['matched_rel_margin_enhancement_ratio']:.1f}",
        f"{f2d_8['cwt_alpha_par_cm']:.0f}",
        f"{f2d_8['fit_alpha_par_cm']:.0f}",
    ]
    for tok in expected_tokens:
        if tok not in table_block:
            raise AssertionError(
                f"Expected JSON-derived value {tok!r} not found in Quantitative Summary table."
            )


def test_quantitative_summary_table_and_figures_exist():
    """Verify Quantitative Summary matches data/pcsel_summary.json, fails on mutation, and all figures exist."""
    summary_path = h.DATA_DIR / "pcsel_summary.json"
    assert summary_path.exists(), "Missing data/pcsel_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    nb_py = (TUT_DIR / "pcsel.py").read_text(encoding="utf-8")

    _validate_summary_table_against_json(nb_py, summary)

    # Mutation check: tampering with a numerical result in the Quantitative Summary table must fail validation
    mutated_nb = nb_py.replace("307.624", "999.999")
    with pytest.raises(AssertionError, match="not found in Quantitative Summary table"):
        _validate_summary_table_against_json(mutated_nb, summary)

    # Verify all setup tables, setup figures, result figures, and GIF exist and are non-empty
    for tname in ("00_setup_table.md", "00_setup_table_finite.md"):
        p_tab = h.FIG_DIR / tname
        assert p_tab.exists() and p_tab.stat().st_size > 200, f"Missing or empty {tname}"
        first_row = p_tab.read_text(encoding="utf-8").splitlines()[2]
        assert first_row in nb_py, f"Setup table {tname} not embedded in pcsel.py"

    for fname in (
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00a_setup_geometry_finite.png",
        "00b_setup_mesh_and_excitation_finite.png",
        "01_vertical_slab_and_greens_functions.png",
        "02_gamma_band_edge_modes_3d_fit.png",
        "02_band_edge_modes_animation.gif",
        "03_unit_cell_grid_convergence.png",
        "04_band_structure_and_radiative_q.png",
        "05_hole_radius_sweep_and_cwt_breakdown.png",
        "06_double_lattice_coupling_cancellation.png",
        "07_finite_size_pcsel_threshold_and_double_lattice.png",
    ):
        p_fig = h.FIG_DIR / fname
        assert p_fig.exists() and p_fig.stat().st_size > 10_000, f"Missing or empty figure {fname}"
    assert (h.FIG_DIR / "02_band_edge_modes_animation.gif").stat().st_size < 5 * 1024 * 1024


def test_production_summary_physics_and_convergence_invariants():
    """Verify mesh resolution rules (>=20 CPW, >=4 cells/hole), positive Q, convergence, and CWT vs FIT."""
    summary = json.loads((h.DATA_DIR / "pcsel_summary.json").read_text(encoding="utf-8"))

    # 1. Fine 3D FIT grid resolution rules: >= 20 cells/wavelength in n_max=3.52 and >= 4 cells across hole
    levels = summary["convergence"]["levels"]
    assert len(levels) == 3
    fine_meta = levels[-1]["meta"]
    assert fine_meta["cpw_max_index"] >= 20.0
    assert fine_meta["cells_across_hole_diameter"] >= 4.0

    # All radius-sweep 3D FIT points and 2D finite FIT runs must also satisfy >= 20 CPW and >= 4 cells/hole
    for row in summary["radius_sweep"]["fit_vs_cwt"]:
        assert row["cpw_max_index"] >= 20.0
        assert row["cells_per_hole_diameter"] >= 4.0

    f2d_list = summary["finite_2d_fit"]
    for row in f2d_list:
        assert row["cpw_max_index"] >= 20.0
        assert row["cells_per_hole_diameter"] >= 4.0
    # Verify monotonic frequency and Q_par progression across N_per = 8, 10, 12, 16
    f2d_freqs = [r["fit_freq_thz"] for r in f2d_list]
    f2d_qs = [r["fit_Q_par"] for r in f2d_list]
    assert all(f2d_freqs[i] < f2d_freqs[i + 1] for i in range(len(f2d_freqs) - 1))
    assert all(f2d_qs[i] < f2d_qs[i + 1] for i in range(len(f2d_qs) - 1))

    # All Bloch band-structure Q factors must be strictly positive (no negative harminv artifacts)
    for branch in ("gx", "gm"):
        for pt in summary["bands_fit"][branch]:
            assert all(q > 0.0 for q in pt["Q_factors"]), f"Negative Q in {branch} at k={pt['k_red']}: {pt['Q_factors']}"

    # 2. Monotonic convergence of all 4 band-edge frequencies and Q_CD toward Richardson asymptote
    rich = summary["convergence"]["richardson"]
    for key in ("freq_A_thz", "freq_B_thz", "freq_C_thz", "freq_D_thz", "Q_CD"):
        errs = [
            rich[key]["rel_err_coarse_pct"],
            rich[key]["rel_err_medium_pct"],
            rich[key]["rel_err_fine_pct"],
        ]
        assert errs[0] > errs[1] > errs[2] > 0.0, f"Non-monotonic convergence for {key}: {errs}"

    # 3. C4v symmetry, Noda sign convention, and BIC vs leaky doublet hierarchy at r/a = 0.18
    cwt = summary["cwt_infinite_r018"]
    fit = summary["fit_unit_cell_r018"]
    assert cwt["kappa_1d_cm"] > 1000.0
    assert np.isclose(cwt["kappa_2d_cm"], 40.886, rtol=1e-2)
    assert np.isclose(cwt["kappa_2d_with_longitudinal_cm"], 185.388, rtol=1e-2)
    assert fit["freq_A_thz"] < fit["freq_B_thz"] < fit["freq_CD_thz"]
    assert fit["c4v_degeneracy_rel_diff"] < 1e-5
    assert fit["Q_A"] > 50.0 * fit["Q_CD"]
    assert fit["Q_B"] > 50.0 * fit["Q_CD"]
    assert fit["rel_err_cwt_vs_fit_A_pct"] < 0.5
    assert fit["rel_err_cwt_vs_fit_B_pct"] < 0.5
    assert fit["rel_err_cwt_longitudinal_vs_fit_B_pct"] < 0.5
    assert fit["rel_err_cwt_vs_fit_CD_pct"] < 0.5
    assert np.isclose(cwt["delta_f_AB_thz"], 0.2348, rtol=1e-2)
    assert np.isclose(cwt["delta_f_AB_with_longitudinal_thz"], 1.0648, rtol=1e-2)
    assert fit["delta_f_AB_thz"] > cwt["delta_f_AB_with_longitudinal_thz"] > cwt["delta_f_AB_thz"]

    # 4. Radius sweep: CWT agrees on Q_CD at moderate fill factor (r/a=0.15) and breaks down at large fill factor (r/a=0.26)
    r_rows = {round(r["r_over_a"], 2): r for r in summary["radius_sweep"]["fit_vs_cwt"]}
    assert r_rows[0.15]["rel_err_Q_dressed_pct"] < 8.0
    assert r_rows[0.26]["rel_err_Q_dressed_pct"] > 50.0

    # 5. Double-lattice continuous cancellation root, crossover length, and asymptotic L^-2 scaling
    dl = summary["double_lattice"]
    assert dl["double_min_abs_C_offdiag_cm"] < 1e-6
    assert 10.0 < dl["double_alpha_v_inf_cm"] < 25.0
    assert np.isclose(dl["single_asym_alpha_v_inf_cm"], dl["double_alpha_v_inf_cm"], rtol=1e-6)
    assert 120.0 < dl["crossover_L_um_vs_asym_single"] < 180.0
    assert np.isclose(dl["second_crossover_L_um_vs_circ_single"], 1755.72, atol=1.0)
    assert np.isclose(dl["single_circ_sweep_edge_L_um"], 3000.0)
    assert "single_circ_opt_L_um" not in dl
    fin_rows = {int(round(r["L_um"])): r for r in summary["finite_cwt_sweep"]}
    assert fin_rows[50]["matched_rel_margin_enhancement_ratio"] < 1.0
    assert fin_rows[100]["matched_rel_margin_enhancement_ratio"] < 1.0
    assert fin_rows[200]["matched_rel_margin_enhancement_ratio"] > 1.5
    assert fin_rows[500]["matched_rel_margin_enhancement_ratio"] > 25.0
    assert fin_rows[1000]["matched_rel_margin_enhancement_ratio"] > 50.0
    assert fin_rows[3000]["matched_rel_margin_enhancement_ratio"] > 80.0
    assert np.isclose(fin_rows[2000]["unperturbed_rel_margin_enhancement_ratio"], 0.80, atol=0.01)
    assert np.isclose(fin_rows[3000]["unperturbed_rel_margin_enhancement_ratio"], 0.38, atol=0.01)

    slopes = summary["finite_scaling_slopes"]
    assert np.isclose(slopes["single_alpha_par_log_slopes"][-1], -3.0, atol=0.02)
    assert np.isclose(slopes["single_alpha_v_log_slopes"][-1], -2.0, atol=0.02)
    assert np.isclose(slopes["single_delta_alpha_th_log_slopes"][-1], -2.07, atol=0.05)


def test_live_quick_3d_fit_unit_cell_and_mutations(monkeypatch):
    """Run a fast 3D FIT unit-cell check (`nx=12`) and verify code-path mutation failure."""
    def _assert_unit_cell_physics(res_dict):
        m = res_dict["modes"]
        fA = m["A"]["freq_thz"]
        fB = m["B"]["freq_thz"]
        fC = m["C"]["freq_thz"]
        fD = m["D"]["freq_thz"]
        assert fA < fB < fC
        assert abs(fC - fD) / fC < 1e-5
        gap_thz = 0.5 * (fC + fD) - fA
        assert gap_thz > 3.5
        q_cd = 0.5 * (abs(m["C"]["Q"]) + abs(m["D"]["Q"]))
        assert abs(m["A"]["Q"]) > 10.0 * q_cd
        assert abs(m["B"]["Q"]) > 5.0 * q_cd

    res_phc = h.run_pcsel_unit_cell(
        stack=h.DEFAULT_STACK,
        nx=12,
        dz_core=18.0e-9,
        dz_clad=42.0e-9,
        z_box=900.0e-9,
        t_ring=150.0e-15,
    )
    _assert_unit_cell_physics(res_phc)

    # Code-path mutation check: zeroing _subpixel_circle_fill inside _helpers removes the air hole
    # from build_pcsel_unit_cell_solver, collapsing the PhC band gap f_CD - f_A and failing the check
    monkeypatch.setattr(
        h,
        "_subpixel_circle_fill",
        lambda xc, yc, dx, dy, radius, x0=0.0, y0=0.0, n_sub=8: np.zeros((len(xc), len(yc)), dtype=float),
    )
    res_mut = h.run_pcsel_unit_cell(
        stack=h.DEFAULT_STACK,
        nx=12,
        dz_core=18.0e-9,
        dz_clad=42.0e-9,
        z_box=900.0e-9,
        t_ring=150.0e-15,
    )
    with pytest.raises(AssertionError):
        _assert_unit_cell_physics(res_mut)
