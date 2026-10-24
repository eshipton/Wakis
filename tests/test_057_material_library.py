# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Tests for the optical and RF dispersive materials library (`wakis.material_library`).

Verifies:
- Every library entry at >= 20 wavelengths across its stated validity range
  against its source formula (~1e-6 relative error in n) or source tabulated
  data (<= stated fit_error).
- Mutation check: perturbing one pole/model coefficient by 1% causes the
  accuracy check for that entry to fail.
- Passivity (Im eps <= 0) across the validity range for every entry.
- Existing `dispersive_lib` entries are unchanged numerically.
- Literature spot values with published citations.
- Metadata (`info`, `list_materials`) and `uniaxial_crystal` helper.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.constants import c as c_light

from wakis.dispersive import (
    EV,
    Debye,
    Drude,
    Lorentz,
    _rakic,
    _sellmeier,
    dispersive_lib,
    eps_from_nk,
)
from wakis.material_library import (
    REFERENCE_COURANT,
    REFERENCE_OPTICAL_DT,
    info,
    list_materials,
    uniaxial_crystal,
)

DATA_DIR = Path(__file__).parent / "data" / "refractiveindex_info"

#: Mapping of canonical material name -> refractiveindex.info YAML filename
YAML_SOURCES = {
    # Metals (Rakic 1998)
    "gold": "Au_Rakic-LD.yml",
    "silver": "Ag_Rakic-LD.yml",
    "aluminium": "Al_Rakic-LD.yml",
    "copper": "Cu_Rakic-LD.yml",
    "beryllium": "Be_Rakic-LD.yml",
    "chromium": "Cr_Rakic-LD.yml",
    "nickel": "Ni_Rakic-LD.yml",
    "palladium": "Pd_Rakic-LD.yml",
    "platinum": "Pt_Rakic-LD.yml",
    "titanium": "Ti_Rakic-LD.yml",
    "tungsten": "W_Rakic-LD.yml",
    # Metals (Johnson & Christy 1972)
    "gold jc": "Au_Johnson.yml",
    "gold jc wide": "Au_Johnson.yml",
    "silver jc": "Ag_Johnson.yml",
    "silver jc wide": "Ag_Johnson.yml",
    "copper jc": "Cu_Johnson.yml",
    "copper jc wide": "Cu_Johnson.yml",
    # Dielectrics, semiconductors, polymers
    "bk7": "BK7_SCHOTT.yml",
    "fused silica": "SiO2_Malitson.yml",
    "gaas": "GaAs_Skauli.yml",
    "si3n4": "Si3N4_Luke.yml",
    "si3n4 philipp": "Si3N4_Philipp.yml",
    "alas": "AlAs_Fern.yml",
    "inp": "InP_Pettit.yml",
    "germanium": "Ge_Icenogle.yml",
    "silicon": "Si_Salzberg.yml",
    "c-si": "Si_Green-1995.yml",
    "pmma": "polymethyl_methacrylate_Sultanova.yml",
    "polycarbonate": "polycarbonate_Sultanova.yml",
    "polystyrene": "polystyrene_Sultanova.yml",
    "cellulose": "cellulose_Sultanova.yml",
    "y2o3": "Y2O3_Nigara.yml",
    "yag": "Y3Al5O12_Zelmon.yml",
    "cdte": "CdTe_Marple.yml",
    "tio2 ald": "TiO2_Sarkar.yml",
    "caf2": "CaF2_Malitson.yml",
    "diamond": "C_Peter.yml",
    "znse": "ZnSe_Connolly.yml",
    "zns": "ZnS_Debenham.yml",
    "gap": "GaP_Bond.yml",
    # Uniaxial crystals
    "bab2o4 o": "BaB2O4_Tamosauskas-o.yml",
    "bab2o4 e": "BaB2O4_Tamosauskas-e.yml",
    "linbo3 o": "LiNbO3_Zelmon-o.yml",
    "linbo3 e": "LiNbO3_Zelmon-e.yml",
    "cawo4 o": "CaWO4_Bond-o.yml",
    "cawo4 e": "CaWO4_Bond-e.yml",
    "caco3 o": "CaCO3_Ghosh-o.yml",
    "caco3 e": "CaCO3_Ghosh-e.yml",
    "quartz o": "SiO2_Ghosh-o.yml",
    "quartz e": "SiO2_Ghosh-e.yml",
    "gan o": "GaN_Barker-o.yml",
    "gan e": "GaN_Barker-e.yml",
    "aln o": "AlN_Pastrnak-o.yml",
    "aln e": "AlN_Pastrnak-e.yml",
    "sapphire o": "Al2O3_Malitson-o.yml",
    "sapphire e": "Al2O3_Malitson-e.yml",
    "mgf2 o": "MgF2_Dodge-o.yml",
    "mgf2 e": "MgF2_Dodge-e.yml",
    "tio2 o": "TiO2_Devore-o.yml",
    "tio2 e": "TiO2_Devore-e.yml",
}

#: Rakic et al., Appl. Opt. 37, 5271 (1998) Table 1 & Table 2 parameters (eV)
RAKIC_TABLE = {
    "gold": (
        9.03,
        [0.760, 0.024, 0.010, 0.071, 0.601, 4.384],
        [0.053, 0.241, 0.345, 0.870, 2.494, 2.214],
        [0.000, 0.415, 0.830, 2.969, 4.304, 13.32],
    ),
    "silver": (
        9.01,
        [0.845, 0.065, 0.124, 0.011, 0.840, 5.646],
        [0.048, 3.886, 0.452, 0.065, 0.916, 2.419],
        [0.000, 0.816, 4.481, 8.185, 9.083, 20.29],
    ),
    "aluminium": (
        14.98,
        [0.523, 0.227, 0.050, 0.166, 0.030],
        [0.047, 0.333, 0.312, 1.351, 3.382],
        [0.000, 0.162, 1.544, 1.808, 3.473],
    ),
    "copper": (
        10.83,
        [0.575, 0.061, 0.104, 0.723, 0.638],
        [0.030, 0.378, 1.056, 3.213, 4.305],
        [0.000, 0.291, 2.957, 5.300, 11.18],
    ),
    "beryllium": (
        18.51,
        [0.084, 0.031, 0.140, 0.530, 0.130],
        [0.035, 1.664, 3.395, 4.454, 1.802],
        [0.000, 0.100, 1.032, 3.183, 4.604],
    ),
    "chromium": (
        10.75,
        [0.168, 0.151, 0.150, 1.149, 0.825],
        [0.047, 3.175, 1.305, 2.676, 1.335],
        [0.000, 0.121, 0.543, 1.970, 8.775],
    ),
    "nickel": (
        15.92,
        [0.096, 0.100, 0.135, 0.106, 0.729],
        [0.048, 4.511, 1.334, 2.178, 6.292],
        [0.000, 0.174, 0.582, 1.597, 6.089],
    ),
    "palladium": (
        9.72,
        [0.330, 0.649, 0.121, 0.638, 0.453],
        [0.008, 2.950, 0.555, 4.621, 3.236],
        [0.000, 0.336, 0.501, 1.659, 5.715],
    ),
    "platinum": (
        9.59,
        [0.333, 0.191, 0.659, 0.547, 3.576],
        [0.080, 0.517, 1.838, 3.668, 8.517],
        [0.000, 0.780, 1.314, 3.141, 9.249],
    ),
    "titanium": (
        7.29,
        [0.148, 0.899, 0.393, 0.187, 0.001],
        [0.082, 2.276, 2.518, 1.663, 1.762],
        [0.000, 0.777, 1.545, 2.509, 19.43],
    ),
    "tungsten": (
        13.22,
        [0.206, 0.054, 0.166, 0.706, 2.590],
        [0.064, 0.530, 1.281, 3.332, 5.836],
        [0.000, 1.004, 1.917, 3.580, 7.498],
    ),
}


def _nk_from_eps(eps):
    """Extract optical constants (n, k >= 0) from Wakis exp(+iwt) eps."""
    nk = np.sqrt(np.conj(eps))
    return nk.real, nk.imag


def _eval_yaml_formula(yml_data, lam_um):
    """Evaluate refractiveindex.info formula 1, 2, or 4 at `lam_um` [um]."""
    entry = yml_data["DATA"][0]
    ftype = entry["type"].strip()
    c = [float(x) for x in entry["coefficients"].split()]
    lam2 = np.asarray(lam_um, dtype=float) ** 2
    if ftype == "formula 1":
        n2 = 1.0 + c[0]
        for b, l_k in zip(c[1::2], c[2::2]):
            n2 = n2 + b * lam2 / (lam2 - l_k**2)
    elif ftype == "formula 2":
        n2 = 1.0 + c[0]
        for b, l2_k in zip(c[1::2], c[2::2]):
            n2 = n2 + b * lam2 / (lam2 - l2_k)
    elif ftype == "formula 4":
        n2 = c[0] + 0.0 * lam2
        for i in (1, 5):
            if i + 3 < len(c) and c[i] != 0.0:
                n2 = n2 + c[i] * (lam2 ** (c[i + 1] / 2.0)) / (
                    lam2 - c[i + 2] ** c[i + 3]
                )
    else:
        raise ValueError(f"Unsupported YAML formula type {ftype!r}")
    return np.sqrt(n2), np.zeros_like(n2)


def _eval_rakic_formula(name, lam_um):
    """Evaluate Rakic et al. (1998) Eq. (3) directly in eV at `lam_um` [um]."""
    omega_p, f_osc, gamma, omega_0 = RAKIC_TABLE[name]
    w_ev = (2 * np.pi * c_light / ( np.asarray(lam_um, dtype=float) * 1e-6)) / EV
    # In exp(+iwt): eps(w) = 1 - f0 wp^2 / (w^2 - i g0 w) + sum fk wp^2 / (wk^2 - w^2 + i gk w)
    eps = 1.0 - (f_osc[0] * omega_p**2) / (w_ev**2 - 1j * gamma[0] * w_ev)
    for fk, gk, wk in zip(f_osc[1:], gamma[1:], omega_0[1:]):
        eps = eps + (fk * omega_p**2) / (wk**2 - w_ev**2 + 1j * gk * w_ev)
    return _nk_from_eps(eps)


def _load_tabulated_nk(yml_data, lmin_um, lmax_um):
    """Load tabulated (lam_um, n, k) in [lmin_um, lmax_um] from YAML data."""
    if len(yml_data["DATA"]) == 2 and "tabulated n" in yml_data["DATA"][0]["type"]:
        arr_n = np.array(
            [
                [float(x) for x in r.split()]
                for r in yml_data["DATA"][0]["data"].strip().splitlines()
            ]
        )
        arr_k = np.array(
            [
                [float(x) for x in r.split()]
                for r in yml_data["DATA"][1]["data"].strip().splitlines()
            ]
        )
        mn = (arr_n[:, 0] >= lmin_um - 1e-9) & (arr_n[:, 0] <= lmax_um + 1e-9)
        mk = (arr_k[:, 0] >= lmin_um - 1e-9) & (arr_k[:, 0] <= lmax_um + 1e-9)
        lam_raw = arr_n[mn, 0]
        n_raw, k_raw = arr_n[mn, 1], arr_k[mk, 1]
    else:
        arr = np.array(
            [
                [float(x) for x in r.split()]
                for r in yml_data["DATA"][0]["data"].strip().splitlines()
            ]
        )
        m = (arr[:, 0] >= lmin_um - 1e-9) & (arr[:, 0] <= lmax_um + 1e-9)
        lam_raw = arr[m, 0]
        n_raw = arr[m, 1]
        k_raw = arr[m, 2] if arr.shape[1] > 2 else np.zeros_like(n_raw)
    if len(lam_raw) < 20:
        inv_lam = np.linspace(1.0 / lam_raw[-1], 1.0 / lam_raw[0], 25)[::-1]
        lam_eval = 1.0 / inv_lam
        n_eval = np.interp(lam_eval, lam_raw, n_raw)
        k_eval = np.interp(lam_eval, lam_raw, k_raw)
        return lam_eval, n_eval, k_eval
    return lam_raw, n_raw, k_raw


def _djordjevic_sarkar_eps(f, eps_r_f0, tand_f0, f0, w1=1e4, w2=1e12):
    """Wideband Djordjevic-Sarkar laminate permittivity in exp(+iwt)."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    w0 = 2 * np.pi * float(f0)
    ln_ratio = np.log(w2 / w1)
    term = np.log((w2 + 1j * w) / (w1 + 1j * w)) / ln_ratio
    term0 = np.log((w2 + 1j * w0) / (w1 + 1j * w0)) / ln_ratio
    d_eps = (eps_r_f0 * tand_f0) / (-term0.imag)
    eps_inf = eps_r_f0 - d_eps * term0.real
    return eps_inf + d_eps * term


def _check_entry_accuracy(name, material):
    """
    Return True if `material` satisfies the accuracy tolerance for `name`
    across >= 20 wavelengths in its validity range, False otherwise.
    """
    meta = info(name)
    lmin, lmax = meta["wavelength_range_um"]
    model = meta["model"]

    if model == "Sellmeier":
        yml_path = DATA_DIR / YAML_SOURCES[name]
        yml_data = yaml.safe_load(yml_path.read_text())
        lam_um = np.linspace(lmin, lmax, 25)
        assert lam_um.size >= 20
        f = c_light / (lam_um * 1e-6)
        n_ref, k_ref = _eval_yaml_formula(yml_data, lam_um)
        n_mod, k_mod = _nk_from_eps(material.epsilon(f))
        rel_n = np.max(np.abs(n_mod - n_ref) / n_ref)
        max_k = np.max(np.abs(k_mod - k_ref))
        return bool(rel_n <= 1e-6 and max_k <= 1e-12)

    if model == "Lorentz-Drude (Rakic)":
        lam_um = np.geomspace(lmin, lmax, 25)
        assert lam_um.size >= 20
        f = c_light / (lam_um * 1e-6)
        n_ref, k_ref = _eval_rakic_formula(name, lam_um)
        n_mod, k_mod = _nk_from_eps(material.epsilon(f))
        rel_n = np.max(np.abs(n_mod - n_ref) / n_ref)
        rel_k = np.max(np.abs(k_mod - k_ref) / k_ref)
        # Also verify agreement with the tabulated refractiveindex.info YAML file
        yml_data = yaml.safe_load((DATA_DIR / YAML_SOURCES[name]).read_text())
        lam_tab, n_tab, k_tab = _load_tabulated_nk(yml_data, lmin, lmax)
        n_tmod, k_tmod = _nk_from_eps(material.epsilon(c_light / (lam_tab * 1e-6)))
        tab_ok = (
            np.max(np.abs(n_tmod - n_tab) / n_tab) <= 5e-4
            and np.max(np.abs(k_tmod - k_tab) / k_tab) <= 5e-4
        )
        return bool(rel_n <= 1e-6 and rel_k <= 1e-6 and tab_ok)

    if model in ("Lorentz-Drude (fit)", "Lorentz (fit)"):
        yml_data = yaml.safe_load((DATA_DIR / YAML_SOURCES[name]).read_text())
        lam_um, n_ref, k_ref = _load_tabulated_nk(yml_data, lmin, lmax)
        assert lam_um.size >= 20
        f = c_light / (lam_um * 1e-6)
        eps_ref = eps_from_nk(n_ref, k_ref)
        eps_mod = material.epsilon(f)
        rel_rms = float(
            np.sqrt(np.mean(np.abs(eps_mod - eps_ref) ** 2 / np.abs(eps_ref) ** 2))
        )
        return bool(rel_rms <= meta["fit_error"])

    if model == "Debye (fit)":
        f = np.geomspace(0.1e9, 20e9, 25)
        assert f.size >= 20
        if name == "fr4":
            eps_ref = _djordjevic_sarkar_eps(f, 4.4, 0.02, 1e9)
        else:
            eps_ref = _djordjevic_sarkar_eps(f, 3.66, 0.0037, 10e9)
        eps_mod = material.epsilon(f)
        rel_rms = float(
            np.sqrt(np.mean(np.abs(eps_mod - eps_ref) ** 2 / np.abs(eps_ref) ** 2))
        )
        return bool(rel_rms <= meta["fit_error"])

    raise ValueError(f"Unhandled model type {model!r}")


@pytest.mark.parametrize("name", list_materials())
def test_every_entry_accuracy_and_mutation(name):
    """
    Every material entry matches its source formula (~1e-6 relative in n) or
    source tabulated data (<= stated fit_error) at >= 20 wavelengths across
    its validity range, AND perturbing one coefficient by 1% causes the check
    to fail.
    """
    mat = dispersive_lib[name]
    assert _check_entry_accuracy(name, mat), f"Accuracy check failed for {name!r}"

    # Mutation check: perturb dominant pole coefficient by +1% and verify failure
    lmin, lmax = info(name)["wavelength_range_um"]
    omega_band = 2 * np.pi * c_light / (np.geomspace(lmin, lmax, 25) * 1e-6)
    pole = max(mat.poles, key=lambda p: float(np.mean(np.abs(p.chi(omega_band)))))
    if isinstance(pole, Drude):
        old = pole.omega_p
        try:
            pole.omega_p = old * 1.01
            assert not _check_entry_accuracy(
                name, mat
            ), f"1% omega_p mutation did not fail for {name!r}"
        finally:
            pole.omega_p = old
    elif isinstance(pole, Lorentz):
        old = pole.omega_0
        try:
            pole.omega_0 = old * 1.01
            assert not _check_entry_accuracy(
                name, mat
            ), f"1% omega_0 mutation did not fail for {name!r}"
        finally:
            pole.omega_0 = old
    elif isinstance(pole, Debye):
        old = mat.eps_inf
        try:
            mat.eps_inf = old * 1.01
            assert not _check_entry_accuracy(
                name, mat
            ), f"1% eps_inf mutation did not fail for {name!r}"
        finally:
            mat.eps_inf = old


@pytest.mark.parametrize("name", list_materials())
def test_passivity_and_stability_metadata(name):
    """
    Every entry is passive (Im eps <= 0) over its validity range and across a
    broadband frequency sweep, and its recorded stability metadata matches
    `is_stable(5e-17, 0.5)`.
    """
    mat = dispersive_lib[name]
    meta = info(name)
    lmin, lmax = meta["wavelength_range_um"]
    f_band = c_light / (np.geomspace(lmin, lmax, 50) * 1e-6)
    eps_band = mat.epsilon(f_band)
    assert np.all(eps_band.imag <= 1e-15), f"Passivity violated in band for {name!r}"

    f_wide = np.geomspace(1e8, 1e17, 200)
    assert np.all(
        mat.epsilon(f_wide).imag <= 1e-15
    ), f"Passivity violated out of band for {name!r}"

    assert mat.is_stable(REFERENCE_OPTICAL_DT, REFERENCE_COURANT) == meta["stable_50as"]
    assert mat.is_stable(2.0e-17, REFERENCE_COURANT)


def test_existing_entries_unchanged():
    """
    The original seven `dispersive_lib` entries ('gold', 'silver',
    'aluminium', 'fused silica', 'silicon', 'fr4', 'ro4350b') remain
    numerically identical to their original definitions.
    """
    ref_gold = _rakic(
        "gold",
        9.03,
        [0.760, 0.024, 0.010, 0.071, 0.601, 4.384],
        [0.053, 0.241, 0.345, 0.870, 2.494, 2.214],
        [0.000, 0.415, 0.830, 2.969, 4.304, 13.32],
    )
    ref_silver = _rakic(
        "silver",
        9.01,
        [0.845, 0.065, 0.124, 0.011, 0.840, 5.646],
        [0.048, 3.886, 0.452, 0.065, 0.916, 2.419],
        [0.000, 0.816, 4.481, 8.185, 9.083, 20.29],
    )
    ref_al = _rakic(
        "aluminium",
        14.98,
        [0.523, 0.227, 0.050, 0.166, 0.030],
        [0.047, 0.333, 0.312, 1.351, 3.382],
        [0.000, 0.162, 1.544, 1.808, 3.473],
    )
    ref_sio2 = _sellmeier(
        "fused silica",
        [0.6961663, 0.4079426, 0.8974794],
        [0.0684043, 0.1162414, 9.896161],
    )
    ref_si = _sellmeier(
        "silicon",
        [10.6684293, 0.0030434748, 1.54133408],
        [0.301516485, 1.13475115, 1104.0],
    )
    f_opt = np.geomspace(30e12, 1500e12, 50)
    for key, ref in [
        ("gold", ref_gold),
        ("silver", ref_silver),
        ("aluminium", ref_al),
        ("fused silica", ref_sio2),
        ("silicon", ref_si),
    ]:
        np.testing.assert_allclose(
            dispersive_lib[key].epsilon(f_opt), ref.epsilon(f_opt), rtol=1e-15, atol=0
        )

    f_rf = np.geomspace(1e8, 20e9, 30)
    assert dispersive_lib["fr4"].eps_inf == pytest.approx(4.17903, rel=1e-15)
    assert dispersive_lib["ro4350b"].eps_inf == pytest.approx(3.64546, rel=1e-15)
    assert np.all(np.isfinite(dispersive_lib["fr4"].epsilon(f_rf)))
    assert np.all(np.isfinite(dispersive_lib["ro4350b"].epsilon(f_rf)))


def test_literature_spot_values():
    """
    Check spot values against cited literature numbers:
    - Fused silica n(1.55 um) = 1.4440 (Malitson, JOSA 55, 1205, 1965)
    - N-BK7 n_d(0.58756 um) = 1.51680 (SCHOTT Optical Glass Catalog)
    - Silicon n(1.55 um) = 3.4777 (Salzberg & Villa, JOSA 47, 244, 1957)
    - GaAs n(1.55 um) = 3.3702 (Skauli et al., JAP 94, 6447, 2003)
    - Si3N4 n(1.55 um) = 1.9963 (Luke et al., Opt. Lett. 40, 4823, 2015)
    - Sapphire n_o(0.5893 um) = 1.7682, n_e(0.5893 um) = 1.7600 (Malitson & Dodge 1972)
    - Calcite n_o(0.5893 um) = 1.6584, n_e(0.5893 um) = 1.4861 (Ghosh 1999)
    - Quartz n_o(0.5893 um) = 1.5443, n_e(0.5893 um) = 1.5534 (Ghosh 1999)
    - LiNbO3 n_o(1.064 um) = 2.2320, n_e(1.064 um) = 2.1555 (Zelmon et al. 1997)
    - YAG n(1.064 um) = 1.8147 (Zelmon et al., Appl. Opt. 37, 4933, 1998)
    - Diamond n(0.5893 um) = 2.4173 (Peter, Z. Phys. 15, 358, 1923)
    - PMMA n(0.5876 um) = 1.4906 (Sultanova et al., Acta Phys. Pol. A 116, 585, 2009)
    """
    spots = [
        ("fused silica", 1.55, 1.4440, 1e-4),
        ("bk7", 0.58756, 1.51680, 1e-4),
        ("silicon", 1.55, 3.4777, 1e-4),
        ("gaas", 1.55, 3.3702, 1e-4),
        ("si3n4", 1.55, 1.9963, 1e-4),
        ("sapphire o", 0.5893, 1.7682, 2e-4),
        ("sapphire e", 0.5893, 1.7600, 2e-4),
        ("caco3 o", 0.5893, 1.6584, 2e-4),
        ("caco3 e", 0.5893, 1.4861, 2e-4),
        ("quartz o", 0.5893, 1.5443, 2e-4),
        ("quartz e", 0.5893, 1.5534, 2e-4),
        ("linbo3 o", 1.064, 2.2320, 2e-4),
        ("linbo3 e", 1.064, 2.1555, 2e-4),
        ("yag", 1.064, 1.8147, 2e-4),
        ("diamond", 0.5893, 2.4173, 3e-4),
        ("pmma", 0.5876, 1.4906, 2e-4),
    ]
    for name, lam_um, n_expected, tol in spots:
        f = c_light / (lam_um * 1e-6)
        n_val, k_val = _nk_from_eps(dispersive_lib[name].epsilon(f))
        np.testing.assert_allclose(
            n_val, n_expected, atol=tol, err_msg=f"Spot check failed for {name}"
        )
        assert abs(k_val) < 1e-12


def test_info_and_uniaxial_helper():
    """Verify `info`, `list_materials`, and `uniaxial_crystal` behavior."""
    canon = list_materials()
    all_names = list_materials(include_aliases=True)
    assert len(canon) >= 60
    assert len(all_names) > len(canon)

    # Alias lookup returns identical canonical metadata and object
    assert dispersive_lib["au"] is dispersive_lib["gold"]
    assert dispersive_lib["sio2"] is dispersive_lib["fused silica"]
    assert dispersive_lib["bbo o"] is dispersive_lib["bab2o4 o"]
    assert info("Au")["name"] == "gold"
    assert info("N-BK7")["name"] == "bk7"
    assert "10.1364/AO.37.005271" in info("gold")["citation"]["reference"]

    with pytest.raises(KeyError):
        info("nonexistent_material_xyz")

    # Uniaxial helper returns (o, o, e) permutations along x, y, z
    o_mat = dispersive_lib["sapphire o"]
    e_mat = dispersive_lib["sapphire e"]
    assert uniaxial_crystal("sapphire", axis="z") == (o_mat, o_mat, e_mat)
    assert uniaxial_crystal("sapphire", axis="x") == (e_mat, o_mat, o_mat)
    assert uniaxial_crystal("al2o3", axis="y") == (o_mat, e_mat, o_mat)
    assert uniaxial_crystal("bbo", axis=0) == (
        dispersive_lib["bab2o4 e"],
        dispersive_lib["bab2o4 o"],
        dispersive_lib["bab2o4 o"],
    )

    with pytest.raises(KeyError):
        uniaxial_crystal("gold")
    with pytest.raises(ValueError):
        uniaxial_crystal("sapphire", axis="w")
