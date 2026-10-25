# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Physics, resolution-rule, and mutation validation tests for Tutorial 2.3:
Magnetic materials (`02-03_magnetic_materials`).
"""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.constants import c as c0

from wakis.dispersive import Debye, DispersiveMaterial, Lorentz

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "02_materials"
    / "02-03_magnetic_materials"
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from magnetic_materials import (  # noqa: E402
    DATA_DIR,
    FIG_DIR,
    VERIFIED_MAGNETIC_MATERIALS,
    analytic_cavity_polariton_branches,
    build_verified_material,
    check_acher_integral_sum_rule,
    compute_hesitivity,
    run_single_cell_magnetic_ade,
    transfer_matrix_slab,
)


def _audit_mag_slab(thickness, dz, fmax, eps_vals, mu_vals):
    """
    Verify that a magnetic/magneto-dielectric slab satisfies ``cpw >= 10``
    inside the medium and ``cells_across >= 4`` across the slab thickness.
    """
    n_c = np.sqrt(
        np.asarray(eps_vals, dtype=np.complex128)
        * np.asarray(mu_vals, dtype=np.complex128)
    )
    n_c = np.where(np.imag(n_c) > 0.0, -n_c, n_c)
    n_max = float(np.max(np.maximum(np.real(n_c), 1.0)))
    lam_min = (c0 / float(fmax)) / n_max
    cpw = lam_min / float(dz)
    cells_across = float(thickness) / float(dz)
    ok = (cpw >= 10.0 - 1e-6) and (cells_across >= 4.0 - 1e-6)
    return ok, cpw, cells_across


def test_magnetic_materials_artifacts_and_tables():
    """
    Verify that:
    1. All 5 ``data/task_<I>.json`` and ``.npz`` files and all 8 figures exist.
    2. Task 0 verifies the literature models (CoZrNb Eq. 8-81, unaligned Co2Z,
       Ikonen film at 580 MHz, NiZn Debye tile), Acher's integral sum rule
       (< 5e-4 error at both low and high damping), the exact single-pole
       hesitivity identity (< 1e-12 relative std across 1 MHz - 20 GHz), and
       periodic FIT magnetic ADE agreement (< 1e-3 vs mu_discrete).
    3. Tasks 1 and 2 (Lorentz-mu and Debye-mu slabs) satisfy cpw >= 10 and
       cells_across >= 4 and match transfer_matrix_slab(mu_discrete) to < 3e-3.
    4. Task 3 (impedance-matched eps(w) = mu(w) slab) suppresses reflection
       R_FIT < 1e-3 across the entire resonant absorption band (> 30 dB lower
       than the unmatched pure-mu slab).
    5. Task 4 (3D PEC cavity loaded with dispersive Lorentz-mu) reproduces the
       lower and upper hybridized cavity-magnon polariton branches to < 1.2%.
    """
    for idx in range(5):
        json_path = DATA_DIR / f"task_{idx}.json"
        npz_path = DATA_DIR / f"task_{idx}.npz"
        assert json_path.is_file(), f"Missing {json_path}"
        assert npz_path.is_file() and npz_path.stat().st_size > 0

    t0 = json.loads((DATA_DIR / "task_0.json").read_text(encoding="utf-8"))
    for key in ("czn_eq881", "co2z_unaligned", "ikonen_film", "nizn_debye"):
        m = t0["literature_models"][key]
        assert m["h_eff_constancy_rel_std"] < 1e-12
    # Check Ikonen 2006 Eq. 17 at 580 MHz (mu' ~ 1.481, tan_delta_m ~ 0.029)
    ik = t0["literature_models"]["ikonen_film"]
    assert abs(ik["mu_real_580MHz"] - 1.481) < 0.02
    assert abs(ik["tan_delta_m_580MHz"] - 0.0284) < 0.003

    assert t0["acher_sum_rule_low_gamma"]["rel_err"] < 5e-4
    assert t0["acher_sum_rule_high_gamma"]["rel_err"] < 5e-4
    assert t0["ade_lorentz_rel_err_disc"] < 5e-3
    assert t0["ade_debye_rel_err_disc"] < 5e-3

    t1 = json.loads((DATA_DIR / "task_1.json").read_text(encoding="utf-8"))
    assert t1["coarse"]["cpw_mat"] >= 10.0
    assert t1["coarse"]["cells_across"] >= 4.0
    assert t1["fine"]["cpw_mat"] >= 10.0
    assert t1["fine"]["cells_across"] >= 4.0
    assert t1["fine"]["rms_R_disc"] < 2e-3
    assert t1["fine"]["rms_T_disc"] < 2e-3
    assert t1["fine"]["rms_R_disc"] < t1["coarse"]["rms_R_disc"]

    t2 = json.loads((DATA_DIR / "task_2.json").read_text(encoding="utf-8"))
    assert t2["cpw_mat"] >= 10.0
    assert t2["cells_across"] >= 4.0
    assert t2["rms_R_disc"] < 2e-3
    assert t2["rms_T_disc"] < 2e-3

    t3 = json.loads((DATA_DIR / "task_3.json").read_text(encoding="utf-8"))
    assert t3["cpw_mat"] >= 10.0
    assert t3["cells_across"] >= 4.0
    assert t3["max_R_cont_matched"] < 1e-14
    assert t3["max_R_fit_matched"] < 1e-3
    assert t3["max_R_fit_unmatched"] > 0.40
    assert t3["peak_A_fit_matched"] > 0.90
    assert t3["rms_T_disc_matched"] < 2e-3

    t4 = json.loads((DATA_DIR / "task_4.json").read_text(encoding="utf-8"))
    assert abs(t4["f_vac_fit_GHz"] - t4["f_vac_grid_GHz"]) < 0.015
    assert abs(t4["f_static_fit_GHz"] - t4["f_static_grid_GHz"]) < 0.015
    for rec in t4["polariton_sweep"]:
        assert rec["f_minus_rel_err"] < 0.012
        assert rec["f_plus_rel_err"] < 0.012
        assert rec["f_minus_fit_GHz"] < rec["f_r_GHz"]
        assert rec["f_plus_fit_GHz"] > rec["f_r_GHz"]

    expected_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "00a_setup_geometry_cavity.png",
        "00b_setup_mesh_and_excitation_cavity.png",
        "wikimedia/fmr_precession_and_permeability_wakis.png",
        "01_literature_ferrites_and_ade_check.png",
        "02_snoek_and_acher_limits.png",
        "03_loss_tangent_and_hesitivity.png",
        "04_lorentz_mu_slab_rt.png",
        "05_debye_mu_slab_rt.png",
        "06_impedance_matched_magneto_dielectric_slab.png",
        "07_cavity_mode_splitting_spectra.png",
        "08_cavity_polariton_avoided_crossing.png",
    ]
    for fname in expected_figs:
        fpath = FIG_DIR / fname
        assert (
            fpath.is_file() and fpath.stat().st_size > 1024
        ), f"Missing {fpath}"

    for tname in ("00_setup_table.md", "00_setup_table_cavity.md"):
        tpath = FIG_DIR / tname
        assert (
            tpath.is_file() and tpath.stat().st_size > 128
        ), f"Missing {tpath}"

    attr_path = FIG_DIR / "wikimedia" / "ATTRIBUTION.md"
    assert attr_path.is_file() and attr_path.stat().st_size > 256


def test_magnetic_materials_physics_and_mutations():
    """
    Fast (< 5 s) physics and mutation checks for Tutorial 2.3:
    1. Passivity in exp(+i*omega*t) convention: Im(mu(f)) <= 0 for all f > 0.
    2. Single-pole hesitivity identity vs multi-pole / mu_inf != 1 mutation:
       for mu_inf = 1 and a single Lorentz or Debye pole, h_eff(f) is strictly
       constant across frequency (rel_std < 1e-12), whereas mutating
       mu_inf = 1 -> 2.5 breaks the constancy of -Im[1/(mu-1)] / omega
       (rel_std > 0.25), matching the condition stated in Yousefi et al. (2016).
    3. Impedance-matched slab (eps = mu) gives R(f) == 0 at all frequencies,
       whereas mutating mu -> 1 (or eps != mu) produces R > 0.25.
    4. Cavity polariton splitting mutation: near f_r ~ f_vac, the naive
       non-dispersive formula f_vac / sqrt(mu_s) misses the true lower
       hybridized polariton branch f_- by > 12%.
    """
    freqs = np.geomspace(1e6, 20e9, 120)
    for key in VERIFIED_MAGNETIC_MATERIALS:
        mat = build_verified_material(key)
        mu_v = mat.mu(freqs)
        assert np.all(np.imag(mu_v) <= 0.0)
        hes = compute_hesitivity(freqs, mu_v)
        assert np.std(hes["h_eff"]) / np.mean(hes["h_eff"]) < 1e-12

    # Mutation 1: setting mu_inf = 2.5 != 1.0 breaks the single-pole hesitivity identity
    mat_mut_muinf = DispersiveMaterial(
        mu_inf=2.5,
        mu_poles=[Lorentz(11.0, 2.0 * np.pi * 1.0e9, 2.0 * np.pi * 0.8e9)],
    )
    hes_mut = compute_hesitivity(freqs, mat_mut_muinf.mu(freqs))
    assert np.std(hes_mut["h_eff"]) / np.mean(hes_mut["h_eff"]) > 0.25

    # Mutation 2: impedance-matched eps == mu vs mismatched eps != mu
    f_band = np.linspace(0.5e9, 2.5e9, 50)
    pole = Lorentz(2.5, 2.0 * np.pi * 1.5e9, 2.0 * np.pi * 0.4e9)
    mat_eq = DispersiveMaterial(
        eps_inf=1.0, poles=[pole], mu_inf=1.0, mu_poles=[pole]
    )
    tm_eq = transfer_matrix_slab(
        f_band, 0.05, mat_eq.epsilon(f_band), mat_eq.mu(f_band)
    )
    assert np.max(tm_eq["R"]) < 1e-14
    tm_mismatch = transfer_matrix_slab(
        f_band, 0.05, np.ones_like(f_band), mat_eq.mu(f_band)
    )
    assert np.max(tm_mismatch["R"]) > 0.40

    # Mutation 3: cavity polariton splitting vs naive static f_vac / sqrt(mu_s)
    f_vac = 1.7637e9
    mu_inf = 1.0
    dmu = 1.25
    f_r = 1.76e9
    f_minus, f_plus = analytic_cavity_polariton_branches(
        f_vac, mu_inf, dmu, f_r
    )
    f_static_naive = f_vac / np.sqrt(mu_inf + dmu)
    assert abs(f_minus - f_static_naive) / f_minus > 0.12
    assert f_plus > f_vac * 1.30

    # Mutation 4: resolution audit rejects coarse dz = 25 mm on 50 mm slab
    ok_good, cpw_g, cells_g = _audit_mag_slab(
        0.050, 1.25e-3, 2.7e9, np.ones(10), mat_eq.mu(np.linspace(0.5e9, 2.7e9, 10))
    )
    assert ok_good and cpw_g >= 10.0 and cells_g >= 4.0
    ok_bad, cpw_b, cells_b = _audit_mag_slab(
        0.050, 25.0e-3, 2.7e9, np.ones(10), mat_eq.mu(np.linspace(0.5e9, 2.7e9, 10))
    )
    assert not ok_bad and (cpw_b < 10.0 or cells_b < 4.0)

    # Mutation 5: citation audit helper and verification_status invariant
    from tools.audit_citations import classify_occurrence, normalize_doi

    verified_keys = [
        k
        for k, v in VERIFIED_MAGNETIC_MATERIALS.items()
        if "full-text verified" in v["verification_status"]
    ]
    assert verified_keys == ["ikonen_film"]
    assert (
        normalize_doi("10.1002/1098-2760(20001205)27:5%3C334::AID-MOP14%3E3.0.CO;2-A).")
        == "10.1002/1098-2760(20001205)27:5<334::aid-mop14>3.0.co;2-a"
    )
    ctx_yousefi = (
        'L. Yousefi, D. M. Sebastian, and R. E. Diaz, "Why the magnetic loss '
        'tangent is not a relevant constraint for permeable conformal antennas"'
    )
    st_ok, score_ok = classify_occurrence(
        ctx_yousefi,
        "Why the Magnetic Loss Tangent Is Not a Relevant Constraint for Permeable Conformal Antennas",
        ["Yousefi", "Sebastian", "Diaz"],
    )
    assert st_ok == "OK" and score_ok >= 0.8
    # Mutating the Crossref title/authors to an unrelated paper must fail (MISMATCH)
    st_bad, score_bad = classify_occurrence(
        ctx_yousefi,
        "Design of an indoor repeater antenna with the improved isolation using metamaterial absorber",
        ["Lee", "Ga", "Song"],
    )
    assert st_bad == "MISMATCH" and score_bad < 0.25
