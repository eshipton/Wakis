# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Tests for Tutorial 3.6: Localized surface plasmons of a metal cylinder
(`tutorials/03_diffraction_and_scattering/03-06_plasmonic_cylinder/plasmonic_cylinder.py`).

Verifies:
- Notebook/script sync (`tutorials/build.py --check`) and committed figures.
- All fitted Lorentz-Drude models (N = 1, 2, 3, 4, 6 and Rakic for Ag and Au)
  are passive (Im eps <= 0 under exp(+i*omega*t)) and stable at the run time
  steps (h = 4 nm and h = 2 nm), with monotonically decreasing fit RMS error
  as N increases.
- Error separation (model error vs FIT discretization error vs total error):
  * Total resonance wavelength error |dlam_nm| decreases from N = 1 to N >= 2
    (for Ag) and N >= 3 (for Au, due to the 2.4 eV interband threshold).
  * Discretization NRMS error decreases from h = 4 nm to h = 2 nm for every N.
  * Additive consistency: total shift = model shift + discretization shift.
  * Retardation red-shift and broadening from 2a = 40 nm to 2a = 80 nm.
  * Polarization selectivity: TE (in-plane E) excites the LSPR whereas TM does not.
- Single-thread CPU timing and memory cost:
  * Slowdown relative to the non-dispersive baseline (N0) is strictly monotone
    in N (0 < 1 < 2 < 3 < 4 < 6) across cylinder_conformal, cylinder_staircase,
    and bulk cases.
  * Auxiliary storage equals 16 * N bytes per dispersive capacitor (2 float64
    arrays per Lorentz/Drude pole).
- Mutation checks: perturbing pole parameters, damping sign, spectra, or
  timing order causes the corresponding verification predicates to fail.
- Live quick execution (`run_plasmonic_cylinder`) reproduces the Ag N = 2 LSPR
  within tolerance.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as c_light

from wakis.analytic.mie_cylinder import cylinder_efficiencies
from wakis.benchmarks.metrics import nrms, peak

REPO_ROOT = Path(__file__).resolve().parent.parent
TUT_DIR = REPO_ROOT / "tutorials" / "03_diffraction_and_scattering" / "03-06_plasmonic_cylinder"
DATA_DIR = TUT_DIR / "data"
FIG_DIR = TUT_DIR / "figures"
SCRIPT_PATH = TUT_DIR / "plasmonic_cylinder.py"

_CACHED_MOD = None


def _load_tutorial_module():
    global _CACHED_MOD
    if _CACHED_MOD is None:
        old_argv = sys.argv[:]
        sys.argv = [str(SCRIPT_PATH), "--plot"]
        try:
            spec = importlib.util.spec_from_file_location(
                "plasmonic_cylinder", SCRIPT_PATH
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _CACHED_MOD = mod
        finally:
            sys.argv = old_argv
    return _CACHED_MOD


def test_notebook_and_figures_present():
    """Verify tutorial script and notebook are in sync and all 5 figures exist."""
    pytest.importorskip("jupytext")
    spec = importlib.util.spec_from_file_location(
        "tut_build", REPO_ROOT / "tutorials" / "build.py"
    )
    build_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_mod)

    assert build_mod.check(SCRIPT_PATH), (
        "plasmonic_cylinder.ipynb is out of sync with plasmonic_cylinder.py"
    )

    expected_figs = [
        "eps_fits_ag_au.png",
        "qext_spectra_vs_mie.png",
        "error_decomposition_vs_poles.png",
        "cost_and_memory_vs_poles.png",
        "near_field_lspr.png",
    ]
    for fname in expected_figs:
        fpath = FIG_DIR / fname
        assert fpath.exists(), f"Missing tutorial figure: {fpath}"
        assert fpath.stat().st_size > 10_000, f"Figure too small / empty: {fpath}"


def test_fitted_models_passivity_stability_and_rms():
    """Verify all committed Ag and Au pole fits are passive, stable, and improve with N."""
    mod = _load_tutorial_module()
    models_data = json.loads((DATA_DIR / "models.json").read_text())

    # Skin depth check: >= 4 cells per minimum skin depth in Ag at both h = 4 nm and 2 nm
    ag_raw = models_data["metals"]["Ag"]["jc_raw"]
    lam_raw = np.array(ag_raw["lam_nm"], dtype=float)
    k_raw = np.array(ag_raw["k"], dtype=float)
    mask = (lam_raw >= 300.0) & (lam_raw <= 900.0)
    min_sd_nm = float(np.min(lam_raw[mask] / (2.0 * np.pi * k_raw[mask])))
    assert 20.0 <= min_sd_nm <= 25.0
    assert min_sd_nm / 4.0 >= 4.0
    assert min_sd_nm / 2.0 >= 8.0

    lam_m = np.linspace(300e-9, 900e-9, 241)
    freqs = c_light / lam_m
    dt_h4 = 4e-9 / (c_light * np.sqrt(3.0)) * 0.85
    dt_h2 = 2e-9 / (c_light * np.sqrt(3.0)) * 0.85

    pole_keys = ["N1", "N2", "N3", "N4", "N6"]
    for metal in ("Ag", "Au"):
        rms_dense = []
        rms_raw = []
        for k in pole_keys + ["Rakic"]:
            entry = models_data["metals"][metal]["models"][k]
            mat = mod.reconstruct_material(entry)
            assert entry["is_passive"] is True
            eps_w = mat.epsilon(freqs)
            assert np.all(eps_w.imag <= 1e-12), (
                f"{metal} {k} has gain: max Im(eps) = {np.max(eps_w.imag)}"
            )
            assert mat.is_stable(dt_h4, courant=0.85), f"{metal} {k} unstable at h=4 nm dt"
            assert mat.is_stable(dt_h2, courant=0.85), f"{metal} {k} unstable at h=2 nm dt"
            assert entry["dt_max_cfl085_as"] * 1e-18 > dt_h4 * 2.0
            if k in pole_keys:
                rms_dense.append(entry["fit_rms_rel_dense"])
                rms_raw.append(entry["fit_rms_rel_raw_jc"])

        # Fit RMS error must decrease monotonically from N=1 to N=6
        assert np.all(np.diff(rms_dense) < 0), (
            f"{metal} dense fit RMS not strictly decreasing with N: {rms_dense}"
        )
        assert np.all(np.diff(rms_raw) < 0), (
            f"{metal} raw J&C fit RMS not strictly decreasing with N: {rms_raw}"
        )


def test_error_separation_and_convergence():
    """Verify model/discretization/total error separation across N, resolution, and metal."""
    spec_data = json.loads((DATA_DIR / "spectra_and_errors.json").read_text())

    # 1. Ag 40 nm: total resonance wavelength error decreases from N=1 to N>=2 and N>=3
    ag40_models = spec_data["Ag_40nm"]["models"]

    for res_key in ("h_4nm", "h_2nm"):
        dlam_n1 = abs(
            ag40_models["N1"]["resolutions"][res_key]["total_error_vs_mie_jc"]["dlam_nm"]
        )
        nrms_n1 = ag40_models["N1"]["resolutions"][res_key]["total_error_vs_mie_jc"][
            "nrms"
        ]
        assert dlam_n1 > 5.0, f"Expected N=1 Drude Ag error > 5 nm ({res_key})"
        for k in ("N2", "N3", "N4", "N6"):
            m_k = ag40_models[k]
            res_k = m_k["resolutions"][res_key]
            dlam_k = abs(res_k["total_error_vs_mie_jc"]["dlam_nm"])
            nrms_k = res_k["total_error_vs_mie_jc"]["nrms"]
            assert dlam_k < 0.45 * dlam_n1, (
                f"Ag 40nm ({res_key}) {k} |dlam|={dlam_k:.2f} nm did not improve "
                f"by >2x over N1 ({dlam_n1:.2f} nm)"
            )
            assert nrms_k < 0.55 * nrms_n1, (
                f"Ag 40nm ({res_key}) {k} NRMS={nrms_k:.4f} did not improve "
                f"over N1 ({nrms_n1:.4f})"
            )

            # Verify exact additive error decomposition: total = model + disc
            mod_shift = m_k["model_error_vs_mie_jc"]["dlam_nm"]
            disc_shift = res_k["discretization_error_vs_mie_cont"]["dlam_nm"]
            tot_shift = res_k["total_error_vs_mie_jc"]["dlam_nm"]
            assert tot_shift == pytest.approx(mod_shift + disc_shift, abs=1e-4)

            q_jc = spec_data["Ag_40nm"]["mie_jc"]["q_peak"]
            q_mie = m_k["model_error_vs_mie_jc"]["q_peak"]
            q_fit = res_k["total_error_vs_mie_jc"]["q_peak"]
            mod_dpk = q_mie - q_jc
            disc_dpk = q_fit - q_mie
            tot_dpk = q_fit - q_jc
            assert tot_dpk == pytest.approx(mod_dpk + disc_dpk, abs=1e-6)
            assert m_k["model_error_vs_mie_jc"]["dq_rel"] == pytest.approx(
                mod_dpk / q_jc, abs=1e-5
            )
            assert res_k["total_error_vs_mie_jc"]["dq_rel"] == pytest.approx(
                tot_dpk / q_jc, abs=1e-5
            )

    # 2. Spatial convergence: h=2 nm discretization NRMS < h=4 nm for every model
    for k in ("N1", "N2", "N3", "N4", "N6", "Rakic"):
        res_dict = ag40_models[k]["resolutions"]
        nrms_disc_h4 = res_dict["h_4nm"]["discretization_error_vs_mie_cont"]["nrms"]
        nrms_disc_h2 = res_dict["h_2nm"]["discretization_error_vs_mie_cont"]["nrms"]
        assert nrms_disc_h2 < 0.65 * nrms_disc_h4, (
            f"Ag {k} discretization NRMS did not converge from h=4 nm ({nrms_disc_h4:.4f}) "
            f"to h=2 nm ({nrms_disc_h2:.4f})"
        )

    # 3. Au 40 nm: requires N >= 3 poles due to 2.4 eV interband transitions
    au40_models = spec_data["Au_40nm"]["models"]
    au_dlam = [
        abs(au40_models[k]["h_4nm"]["total_error_vs_mie_jc"]["dlam_nm"])
        for k in ("N1", "N2", "N3", "N6")
    ]
    assert au_dlam[0] > 80.0  # N=1 Drude misses by ~96 nm
    assert au_dlam[1] > 25.0  # N=2 still misses by ~33 nm
    assert au_dlam[2] < 10.0  # N=3 brings total error under 10 nm (~8.7 nm)
    assert au_dlam[3] < 2.0   # N=6 brings total error under 2 nm (~0.64 nm)
    assert au_dlam[0] > au_dlam[1] > au_dlam[2] > au_dlam[3]

    # 4. Retardation red-shift and radiation broadening (quasi-static -> 40 nm -> 80 nm)
    jc_40 = spec_data["Ag_40nm"]["mie_jc"]
    ret_80 = spec_data["Ag_80nm_retardation"]
    qs_40 = ret_80["quasi_static_40nm"]
    jc_80 = ret_80["mie_jc_80nm"]
    fit_80_n2 = ret_80["fit_n2_80nm_h4nm"]

    assert qs_40["lam_peak_nm"] < jc_40["lam_peak_nm"] < jc_80["q_ext_lam_peak_nm"] < jc_80["q_sca_lam_peak_nm"]
    assert jc_80["q_sca_lam_peak_nm"] - qs_40["lam_peak_nm"] > 9.0
    assert qs_40["fwhm_nm"] < jc_40["fwhm_nm"] < 0.5 * jc_80["q_ext_fwhm_nm"]
    assert fit_80_n2["nrms_vs_mie_n2"] < 0.05

    # 5. Polarization selectivity: TE has the plasmon resonance, TM does not
    lam_res = jc_40["lam_peak_nm"] * 1e-9
    x_res = 2.0 * np.pi * 20e-9 / lam_res
    m_res = np.sqrt(-1.08 + 0.25j)  # Ag near 341 nm
    q_te = cylinder_efficiencies(m_res, x_res, polarization="TE")[0]
    q_tm = cylinder_efficiencies(m_res, x_res, polarization="TM")[0]
    assert q_te > 4.0 and q_te / q_tm > 3.5


def test_timing_and_memory_monotone_in_poles():
    """Verify slowdown and memory per edge scale monotonically with pole count N."""
    timing = json.loads((DATA_DIR / "timing.json").read_text())

    assert "Intel Xeon" in timing["machine"]
    assert "N0" in timing["protocol"]["baseline_denominator"]

    ordered_keys = ["N0", "N1", "N2", "N3", "N4", "N6"]
    expected_n = [0, 1, 2, 3, 4, 6]

    for case_name in ("cylinder_conformal", "cylinder_staircase", "bulk"):
        entries = timing["cases"][case_name]["models"]
        slowdowns = [entries[k]["slowdown_vs_N0"] for k in ordered_keys]
        us_per_step = [entries[k]["median_us_per_step"] for k in ordered_keys]

        assert slowdowns[0] == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.diff(slowdowns) > 0), (
            f"{case_name} slowdown is not strictly monotone in N: {slowdowns}"
        )
        assert np.all(np.diff(us_per_step) > 0), (
            f"{case_name} median_us_per_step is not strictly monotone in N: {us_per_step}"
        )

        # Memory per cap must be exactly 16.0 * N bytes (2 float64 arrays per pole)
        for k, n_poles in zip(ordered_keys, expected_n):
            e = entries[k]
            assert e["n_poles"] == n_poles
            assert e["pole_aux_bytes_per_cap"] == pytest.approx(
                16.0 * n_poles, abs=1e-9
            )
            if n_poles > 0:
                assert e["cost_ns_per_disp_edge_per_pole"] > 0.0
                assert e["n_disp_edges"] > 0

    # Bulk case must have larger slowdown than cylinder cases for every N >= 1
    bulk_entries = timing["cases"]["bulk"]["models"]
    conf_entries = timing["cases"]["cylinder_conformal"]["models"]
    for k in ("N1", "N2", "N3", "N4", "N6"):
        assert bulk_entries[k]["slowdown_vs_N0"] > conf_entries[k]["slowdown_vs_N0"]


def test_mutation_checks_can_fail():
    """Mutation checks verifying that each validation predicate fails on perturbed data."""
    mod = _load_tutorial_module()
    models_data = json.loads((DATA_DIR / "models.json").read_text())
    spec_data = json.loads((DATA_DIR / "spectra_and_errors.json").read_text())
    timing = json.loads((DATA_DIR / "timing.json").read_text())

    lam_nm = np.array(spec_data["wavelengths_nm"], dtype=float)
    lam_m = lam_nm * 1e-9
    freqs = c_light / lam_m

    # Mutation 1: Flipping Lorentz/Drude damping sign breaks passivity (Im eps > 0)
    bad_entry = copy.deepcopy(models_data["metals"]["Ag"]["models"]["N2"])
    bad_entry["drude"][0]["gamma_rad_s"] = -abs(
        bad_entry["drude"][0]["gamma_rad_s"]
    )
    bad_mat = mod.reconstruct_material(bad_entry)
    assert not np.all(bad_mat.epsilon(freqs).imag <= 1e-12), (
        "Mutation check failed: negative damping should violate passivity (Im eps <= 0)"
    )

    # Mutation 2: Perturbing Ag N=2 plasma frequency by +5% ruins LSPR wavelength accuracy
    perturbed_entry = copy.deepcopy(models_data["metals"]["Ag"]["models"]["N2"])
    perturbed_entry["drude"][0]["omega_p_rad_s"] *= 1.05
    perturbed_mat = mod.reconstruct_material(perturbed_entry)
    eps_pert = perturbed_mat.epsilon(freqs)
    m_pert = np.sqrt(eps_pert)
    x_40 = 2.0 * np.pi * 20e-9 / lam_m
    qext_pert = np.array(
        [
            float(cylinder_efficiencies(m_pert[i], x_40[i], polarization="TE")[0])
            for i in range(len(lam_m))
        ]
    )
    pk_pert = peak(lam_nm, qext_pert)
    jc_ref = spec_data["Ag_40nm"]["mie_jc"]
    assert abs(pk_pert["lam_peak"] - jc_ref["lam_peak_nm"]) > 8.0, (
        "Mutation check failed: 5% omega_p perturbation must shift LSPR by > 8 nm"
    )

    # Mutation 3: Perturbing spectrum amplitude by +25% fails the NRMS accuracy check (< 7%)
    qext_jc = np.array(jc_ref["q_ext"], dtype=float)
    assert nrms(qext_jc, qext_jc) == pytest.approx(0.0, abs=1e-12)
    assert nrms(1.25 * qext_jc, qext_jc) > 0.075, (
        "Mutation check failed: 25% amplitude scaling must exceed 7% NRMS threshold"
    )

    # Mutation 4: Swapping two timing entries breaks monotonicity check
    conf_entries = timing["cases"]["cylinder_conformal"]["models"]
    mutated_slowdowns = [
        conf_entries[k]["slowdown_vs_N0"] for k in ("N0", "N1", "N3", "N2", "N4", "N6")
    ]
    assert not np.all(np.diff(mutated_slowdowns) > 0), (
        "Mutation check failed: swapped N2/N3 order must fail strict monotonicity"
    )


def test_quick_arm_execution():
    """Run a live 2D FIT simulation via run_plasmonic_cylinder and verify LSPR accuracy."""
    mod = _load_tutorial_module()
    models_data = json.loads((DATA_DIR / "models.json").read_text())
    spec_data = json.loads((DATA_DIR / "spectra_and_errors.json").read_text())

    mat_ag_n2 = mod.reconstruct_material(models_data["metals"]["Ag"]["models"]["N2"])
    res = mod.run_plasmonic_cylinder(
        mat_ag_n2, dx_nm=4.0, radius_nm=20.0, conformal="all", t_run_fs=35.0
    )
    lam_nm = np.array(spec_data["wavelengths_nm"], dtype=float)
    pk = peak(lam_nm, res["q_ext"])
    mie_n2_ref = np.array(
        spec_data["Ag_40nm"]["models"]["N2"]["mie_cont_q_ext"], dtype=float
    )
    err_nrms = float(nrms(res["q_ext"], mie_n2_ref))

    assert 340.0 <= pk["lam_peak"] <= 346.0
    assert 4.2 <= pk["height"] <= 5.2
    assert err_nrms < 0.07
