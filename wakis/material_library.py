# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Optical and RF dispersive materials library from original sources.

Builds `DispersiveMaterial` entries from peer-reviewed literature formulas
(Sellmeier, Lorentz-Drude) and passive pole fits (`fit_poles`) to tabulated
optical constants from the CC0 ``refractiveindex.info`` database, and registers
them into `wakis.dispersive.dispersive_lib`.

All models follow the Wakis ``exp(+i w t)`` convention where passive media
satisfy ``Im eps(w) <= 0``.

Functions
---------
info(name)
    Return metadata dictionary (citation, model type, validity wavelength
    range, fit error, and stability information) for a material or alias.
list_materials(include_aliases=False, category=None)
    Return sorted list of registered dispersive material names.
uniaxial_crystal(name, axis='z')
    Return the ``(eps_x, eps_y, eps_z)`` triple of `DispersiveMaterial`
    objects for a uniaxial crystal with optical axis along ``x``, ``y``,
    or ``z``.
"""

import numpy as np
from scipy.constants import c as c_light

from .dispersive import (
    Debye,
    DispersiveMaterial,
    Drude,
    Lorentz,
    _rakic,
    _sellmeier,
    dispersive_lib,
    fit_poles,
)

#: Complex permeability mu(f) = mu'(f) - j mu''(f) of a representative flexible sintered NiZn NFC ferrite sheet over
#: 1-80 MHz. MODEL spectrum, not a measurement: a domain-wall + spin-rotation (natural resonance, f_0 = 135 MHz)
#: dispersion of the form used by T. Tsutaoka, J. Appl. Phys. 93(5), 2789-2796 (2003), doi:10.1063/1.1542651, with
#: parameters chosen to give mu' ~ 150 and mu'' ~ 3.6 at 13.56 MHz, the level reported for sintered NiZn NFC sheets
#: by J. Victoria et al., IEEE Trans. Electromagn. Compat. 62(4), 1504-1514 (2020), doi:10.1109/TEMC.2020.3003800.
#: Replace with a measured datasheet spectrum when one is available.
NIZN_FERRITE_SOURCE_DATA = {
    "f_hz": np.array(
        [
            1.0e6,
            2.0e6,
            5.0e6,
            10.0e6,
            13.56e6,
            20.0e6,
            30.0e6,
            40.0e6,
            50.0e6,
            65.0e6,
            80.0e6,
        ],
        dtype=float,
    ),
    "mu_real": np.array(
        [
            148.51,
            148.53,
            148.69,
            149.27,
            149.92,
            151.62,
            155.69,
            161.74,
            170.20,
            188.87,
            218.15,
        ],
        dtype=float,
    ),
    "mu_imag": np.array(
        [
            0.26,
            0.52,
            1.30,
            2.62,
            3.58,
            5.41,
            8.57,
            12.38,
            17.22,
            27.91,
            47.01,
        ],
        dtype=float,
    ),
}

#: Digitized permeability mu'(f), mu''(f) (10 MHz - 1 GHz) and attenuation alpha(f) [dB/cm] (0.1 - 6 GHz),
#: read from the datasheet plots (approximate) in Laird Performance Materials, "Eccosorb RF-LB EMI Noise
#: Suppression Absorber Technical Datasheet," Doc. EM-ABS-DS-10132022 (2022). Volume resistivity 7.45e11 Ohm*cm.
ECCOSORB_RF_LB_SOURCE_DATA = {
    "f_mu_hz": np.array(
        [
            0.010e9,
            0.015e9,
            0.020e9,
            0.030e9,
            0.040e9,
            0.050e9,
            0.060e9,
            0.070e9,
            0.080e9,
            0.090e9,
            0.100e9,
            0.150e9,
            0.200e9,
            0.300e9,
            0.400e9,
            0.500e9,
            0.600e9,
            0.700e9,
            0.800e9,
            0.900e9,
            1.000e9,
        ],
        dtype=float,
    ),
    "mu_real": np.array(
        [
            36.50,
            36.08,
            35.56,
            34.16,
            32.69,
            31.11,
            29.69,
            28.45,
            27.45,
            26.49,
            25.68,
            23.10,
            21.55,
            19.42,
            17.43,
            15.56,
            13.82,
            11.84,
            10.10,
            8.53,
            7.41,
        ],
        dtype=float,
    ),
    "mu_imag": np.array(
        [
            2.49,
            3.68,
            4.74,
            6.62,
            7.90,
            8.82,
            9.37,
            9.67,
            9.80,
            9.86,
            9.87,
            9.75,
            9.83,
            10.61,
            11.59,
            12.38,
            12.92,
            13.21,
            13.27,
            13.12,
            12.89,
        ],
        dtype=float,
    ),
    "f_att_hz": np.array(
        [
            0.1e9,
            0.2e9,
            0.3e9,
            0.4e9,
            0.5e9,
            0.6e9,
            0.7e9,
            0.8e9,
            0.9e9,
            1.0e9,
            1.2e9,
            1.4e9,
            1.6e9,
            1.8e9,
            2.0e9,
            2.5e9,
            3.0e9,
            3.5e9,
            4.0e9,
            4.5e9,
            5.0e9,
            5.5e9,
            6.0e9,
        ],
        dtype=float,
    ),
    "atten_db_cm": np.array(
        [
            2.81,
            6.19,
            9.77,
            12.90,
            15.87,
            18.83,
            20.21,
            24.08,
            26.76,
            27.75,
            32.58,
            36.10,
            40.15,
            41.82,
            50.23,
            59.31,
            66.11,
            71.21,
            72.95,
            75.62,
            78.88,
            80.73,
            81.77,
        ],
        dtype=float,
    ),
    "volume_resistivity_ohm_cm": 7.45e11,
}


def fit_mu_poles(
    f,
    mu,
    n_lorentz=1,
    n_drude=0,
    n_debye=0,
    mu_inf=1.0,
    eps_inf=1.0,
    n_starts=4,
    name=None,
):
    """
    Fit passive Drude, Lorentz, and Debye magnetic poles to complex permeability data.

    Uses the existing `fit_poles` optimizer by electromagnetic duality and
    returns a `DispersiveMaterial` whose dispersion resides in the magnetic
    channel (`mu_inf`, `mu_poles`).

    Parameters
    ----------
    f : array_like
        Frequencies [Hz], strictly positive.
    mu : array_like
        Target complex relative permeability ``mu_r(f) = mu'(f) - j mu''(f)`` at
        `f`, with ``Im(mu) <= 0`` for passive media under the ``exp(+j w t)``
        sign convention.
    n_lorentz, n_drude, n_debye : int, optional
        Number of Lorentz, Drude, and Debye magnetic poles to fit.
    mu_inf : float or None, optional
        High-frequency relative permeability ``mu_inf >= 1``. Fitted when
        ``None``.
    eps_inf : float, optional
        Static/optical relative permittivity ``eps_inf >= 1`` of the material.
        Default is 1.0.
    n_starts : int, optional
        Number of log-spaced multi-start initializations passed to `fit_poles`.
    name : str, optional
        Material name.

    Returns
    -------
    DispersiveMaterial
        Fitted magnetic dispersive material with ``fit_error`` set to the
        relative RMS residual across `f`.
    """
    dual_fit = fit_poles(
        f,
        mu,
        n_lorentz=n_lorentz,
        n_drude=n_drude,
        n_debye=n_debye,
        eps_inf=mu_inf,
        n_starts=n_starts,
        name=name,
    )
    mat = DispersiveMaterial(
        eps_inf=float(eps_inf),
        poles=(),
        mu_inf=float(dual_fit.eps_inf),
        mu_poles=list(dual_fit.poles),
        name=name,
    )
    mat.fit_error = float(dual_fit.fit_error)
    return mat

#: Reference optical timestep [s] and 3D Courant number used for stability metadata
#: (dt = 50 as corresponds to dx = dy = dz ~ 52 nm at Courant = 0.5).
REFERENCE_OPTICAL_DT = 5.0e-17
REFERENCE_COURANT = 0.5


def _sellmeier_eps_inf(name, eps_inf, B, C_um):
    """Lossless Sellmeier model n^2 = eps_inf + sum B lambda^2 / (lambda^2 - C^2)."""
    poles = [
        Lorentz(b, 2 * np.pi * c_light / (cc * 1e-6), 0.0) for b, cc in zip(B, C_um)
    ]
    return DispersiveMaterial(float(eps_inf), poles, name=name)


def _sellmeier_f2(name, A0, B, C_um2):
    """Lossless Sellmeier formula 2: n^2 = 1 + A0 + sum B lambda^2 / (lambda^2 - C)."""
    c_um = [np.sqrt(float(c)) for c in C_um2]
    return _sellmeier_eps_inf(name, 1.0 + float(A0), B, c_um)


#: Registry mapping canonical material name -> metadata dict.
_METADATA = {}

#: Registry mapping alias (lowercase) -> canonical material name.
_ALIASES = {}


def _cite(authors, title, journal, year, doi=None):
    ref = f"{authors}, {journal} ({year})"
    if doi:
        ref += f", doi:{doi}"
    return {
        "authors": authors,
        "title": title,
        "journal": journal,
        "year": int(year),
        "doi": doi,
        "reference": ref,
    }


def _register(
    name,
    material,
    *,
    aliases=(),
    category,
    model,
    wavelength_range_um,
    citation,
    fit_error=None,
    stable_50as=True,
    max_stable_dt_s=None,
    stability_note="",
):
    """Register `material` and its aliases into `dispersive_lib` and `_METADATA`."""
    key = name.strip().lower()
    if fit_error is not None:
        material.fit_error = float(fit_error)
    dispersive_lib[key] = material
    _ALIASES[key] = key
    clean_aliases = tuple(a.strip().lower() for a in aliases)
    for alias in clean_aliases:
        dispersive_lib[alias] = material
        _ALIASES[alias] = key
    _METADATA[key] = {
        "name": key,
        "aliases": clean_aliases,
        "category": category,
        "model": model,
        "wavelength_range_um": (
            float(wavelength_range_um[0]),
            float(wavelength_range_um[1]),
        ),
        "fit_error": float(fit_error) if fit_error is not None else None,
        "citation": citation,
        "stable_50as": bool(stable_50as),
        "max_stable_dt_s": (
            float(max_stable_dt_s) if max_stable_dt_s is not None else None
        ),
        "stability_note": stability_note,
    }


def _build_library():
    """Populate `dispersive_lib` and `_METADATA` with all optical and RF materials."""
    rakic_cite = _cite(
        "A. D. Rakic, A. B. Djurisic, J. M. Elazar, and M. L. Majewski",
        "Optical properties of metallic films for vertical-cavity optoelectronic devices",
        "Appl. Opt. 37, 5271-5283",
        1998,
        "10.1364/AO.37.005271",
    )
    jc_cite = _cite(
        "P. B. Johnson and R. W. Christy",
        "Optical constants of the noble metals",
        "Phys. Rev. B 6, 4370-4379",
        1972,
        "10.1103/PhysRevB.6.4370",
    )
    sultanova_cite = _cite(
        "N. Sultanova, S. Kasarova, and I. Nikolov",
        "Dispersion properties of optical polymers",
        "Acta Phys. Pol. A 116, 585-587",
        2009,
        "10.12693/APhysPolA.116.585",
    )
    ghosh_cite = _cite(
        "G. Ghosh",
        " Dispersion-equation coefficients for the refractive index and birefringence of calcite and quartz crystals",
        "Opt. Commun. 163, 95-102",
        1999,
        "10.1016/S0030-4018(99)00091-7",
    )
    bond_cite = _cite(
        "W. L. Bond",
        "Measurement of the refractive indices of several crystals",
        "J. Appl. Phys. 36, 1674-1677",
        1965,
        "10.1063/1.1703106",
    )

    # ------------------------------------------------------------------
    # 1. Existing entries in dispersive_lib (kept numerically identical)
    # ------------------------------------------------------------------
    _register(
        "gold",
        dispersive_lib["gold"],
        aliases=("au", "gold rakic", "au rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=False,
        max_stable_dt_s=4.72e-17,
        stability_note="Unstable at dt=50 as (UV pole at 13.32 eV); stable for dt <= 47 as.",
    )
    _register(
        "silver",
        dispersive_lib["silver"],
        aliases=("ag", "silver rakic", "ag rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=False,
        max_stable_dt_s=3.87e-17,
        stability_note="Unstable at dt=50 as (UV pole at 20.29 eV); stable for dt <= 38 as.",
    )
    _register(
        "aluminium",
        dispersive_lib["aluminium"],
        aliases=("aluminum", "al", "aluminium rakic", "al rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=True,
        max_stable_dt_s=8.85e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "fused silica",
        dispersive_lib["fused silica"],
        aliases=("sio2", "silica"),
        category="dielectric",
        model="Sellmeier",
        wavelength_range_um=(0.21, 6.7),
        citation=_cite(
            "I. H. Malitson",
            "Interspecimen comparison of the refractive index of fused silica",
            "J. Opt. Soc. Am. 55, 1205-1209",
            1965,
            "10.1364/JOSA.55.001205",
        ),
        stable_50as=True,
        max_stable_dt_s=5.12e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "silicon",
        dispersive_lib["silicon"],
        aliases=("si",),
        category="semiconductor",
        model="Sellmeier",
        wavelength_range_um=(1.357, 11.04),
        citation=_cite(
            "C. D. Salzberg and J. J. Villa",
            "Infrared refractive indexes of silicon, germanium and modified selenium glass",
            "J. Opt. Soc. Am. 47, 244-246",
            1957,
            "10.1364/JOSA.47.000244",
        ),
        stable_50as=True,
        max_stable_dt_s=7.21e-17,
        stability_note="Stable at dt=50 as (Courant=0.5); IR pole at 1104 um is benign.",
    )
    _register(
        "fr4",
        dispersive_lib["fr4"],
        aliases=("fr-4",),
        category="rf_substrate",
        model="Debye (fit)",
        wavelength_range_um=(1.49896e4, 2.99792e6),
        citation=_cite(
            "A. R. Djordjevic, R. M. Biljic, V. D. Likar-Smiljanic, and T. K. Sarkar",
            "Wideband frequency-domain characterization of FR-4 and time-domain causality",
            "IEEE Trans. Electromagn. Compat. 43, 662-667",
            2001,
            "10.1109/15.974647",
        ),
        fit_error=1.3e-3,
        stable_50as=True,
        max_stable_dt_s=1.0e-9,
        stability_note="Unconditionally stable Crank-Nicolson Debye update for Courant <= 1.",
    )
    _register(
        "ro4350b",
        dispersive_lib["ro4350b"],
        aliases=("rogers ro4350b",),
        category="rf_substrate",
        model="Debye (fit)",
        wavelength_range_um=(1.49896e4, 2.99792e6),
        citation=_cite(
            "Rogers Corporation / A. R. Djordjevic et al.",
            "RO4000 Series High Frequency Circuit Materials Data Sheet (Djordjevic-Sarkar model)",
            "Rogers Corp. Technical Note / IEEE Trans. Electromagn. Compat. 43, 662",
            2001,
            "10.1109/15.974647",
        ),
        fit_error=3.0e-4,
        stable_50as=True,
        max_stable_dt_s=1.0e-9,
        stability_note="Unconditionally stable Crank-Nicolson Debye update for Courant <= 1.",
    )

    # ------------------------------------------------------------------
    # 2. Metals: Rakic et al., Appl. Opt. 37, 5271 (1998), 0.2-12.4 um
    # ------------------------------------------------------------------
    _register(
        "copper",
        _rakic(
            "copper",
            10.83,
            [0.575, 0.061, 0.104, 0.723, 0.638],
            [0.030, 0.378, 1.056, 3.213, 4.305],
            [0.000, 0.291, 2.957, 5.300, 11.18],
        ),
        aliases=("cu", "copper rakic", "cu rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=True,
        max_stable_dt_s=8.85e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "beryllium",
        _rakic(
            "beryllium",
            18.51,
            [0.084, 0.031, 0.140, 0.530, 0.130],
            [0.035, 1.664, 3.395, 4.454, 1.802],
            [0.000, 0.100, 1.032, 3.183, 4.604],
        ),
        aliases=("be", "beryllium rakic", "be rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=True,
        max_stable_dt_s=6.55e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "chromium",
        _rakic(
            "chromium",
            10.75,
            [0.168, 0.151, 0.150, 1.149, 0.825],
            [0.047, 3.175, 1.305, 2.676, 1.335],
            [0.000, 0.121, 0.543, 1.970, 8.775],
        ),
        aliases=("cr", "chromium rakic", "cr rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=True,
        max_stable_dt_s=8.12e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "nickel",
        _rakic(
            "nickel",
            15.92,
            [0.096, 0.100, 0.135, 0.106, 0.729],
            [0.048, 4.511, 1.334, 2.178, 6.292],
            [0.000, 0.174, 0.582, 1.597, 6.089],
        ),
        aliases=("ni", "nickel rakic", "ni rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=True,
        max_stable_dt_s=7.21e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "palladium",
        _rakic(
            "palladium",
            9.72,
            [0.330, 0.649, 0.121, 0.638, 0.453],
            [0.008, 2.950, 0.555, 4.621, 3.236],
            [0.000, 0.336, 0.501, 1.659, 5.715],
        ),
        aliases=("pd", "palladium rakic", "pd rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=True,
        max_stable_dt_s=1.08e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "platinum",
        _rakic(
            "platinum",
            9.59,
            [0.333, 0.191, 0.659, 0.547, 3.576],
            [0.080, 0.517, 1.838, 3.668, 8.517],
            [0.000, 0.780, 1.314, 3.141, 9.249],
        ),
        aliases=("pt", "platinum rakic", "pt rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=False,
        max_stable_dt_s=4.93e-17,
        stability_note="Unstable at dt=50 as (UV pole at 9.249 eV); stable for dt <= 49 as.",
    )
    _register(
        "titanium",
        _rakic(
            "titanium",
            7.29,
            [0.148, 0.899, 0.393, 0.187, 0.001],
            [0.082, 2.276, 2.518, 1.663, 1.762],
            [0.000, 0.777, 1.545, 2.509, 19.43],
        ),
        aliases=("ti", "titanium rakic", "ti rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=True,
        max_stable_dt_s=6.55e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "tungsten",
        _rakic(
            "tungsten",
            13.22,
            [0.206, 0.054, 0.166, 0.706, 2.590],
            [0.064, 0.530, 1.281, 3.332, 5.836],
            [0.000, 1.004, 1.917, 3.580, 7.498],
        ),
        aliases=("w", "tungsten rakic", "w rakic"),
        category="metal",
        model="Lorentz-Drude (Rakic)",
        wavelength_range_um=(0.2, 12.4),
        citation=rakic_cite,
        stable_50as=False,
        max_stable_dt_s=4.36e-17,
        stability_note="Unstable at dt=50 as (UV pole at 7.498 eV); stable for dt <= 43 as.",
    )

    # ------------------------------------------------------------------
    # 3. Metals fitted to Johnson & Christy, Phys. Rev. B 6, 4370 (1972)
    # ------------------------------------------------------------------
    _register(
        "gold jc",
        DispersiveMaterial(
            8.0821758015,
            [
                Lorentz(2.0614922693, 4.4784561193e15, 1.6262717678e15),
                Drude(1.4217948480e16, 5.4249436789e02),
            ],
            name="gold jc",
        ),
        aliases=("au jc",),
        category="metal",
        model="Lorentz-Drude (fit)",
        wavelength_range_um=(0.4, 0.8),
        citation=jc_cite,
        fit_error=0.0445,
        stable_50as=True,
        max_stable_dt_s=3.40e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "gold jc wide",
        DispersiveMaterial(
            3.5770411242,
            [
                Lorentz(3.2666187079, 6.5937471987e15, 4.1961463928e15),
                Lorentz(0.9576984909, 4.4766398385e15, 1.3096255569e15),
                Drude(1.3309445614e16, 6.4382053829e08),
            ],
            name="gold jc wide",
        ),
        aliases=("au jc wide",),
        category="metal",
        model="Lorentz-Drude (fit)",
        wavelength_range_um=(0.25, 1.94),
        citation=jc_cite,
        fit_error=0.0795,
        stable_50as=True,
        max_stable_dt_s=2.10e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "silver jc",
        DispersiveMaterial(
            3.5,
            [
                Lorentz(0.2980766289, 5.7540187690e15, 4.2112431326e14),
                Drude(1.4010372394e16, 5.5831684272e13),
            ],
            name="silver jc",
        ),
        aliases=("ag jc",),
        category="metal",
        model="Lorentz-Drude (fit)",
        wavelength_range_um=(0.4, 0.8),
        citation=jc_cite,
        fit_error=0.0095,
        stable_50as=True,
        max_stable_dt_s=2.45e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "silver jc wide",
        DispersiveMaterial(
            4.9757865020,
            [
                Lorentz(0.4521030567, 3.0103682359e15, 1.0317102444e15),
                Lorentz(0.2488343015, 6.3669715924e15, 4.3150323330e14),
                Drude(1.4456365586e16, 1.0407994976e05),
            ],
            name="silver jc wide",
        ),
        aliases=("ag jc wide",),
        category="metal",
        model="Lorentz-Drude (fit)",
        wavelength_range_um=(0.30, 1.94),
        citation=jc_cite,
        fit_error=0.0986,
        stable_50as=True,
        max_stable_dt_s=2.68e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "copper jc",
        DispersiveMaterial(
            8.4848239425,
            [
                Lorentz(3.1852126472, 4.1280713848e15, 2.0782534663e15),
                Drude(1.4599512177e16, 1.2598268686e00),
            ],
            name="copper jc",
        ),
        aliases=("cu jc",),
        category="metal",
        model="Lorentz-Drude (fit)",
        wavelength_range_um=(0.4, 0.8),
        citation=jc_cite,
        fit_error=0.0850,
        stable_50as=True,
        max_stable_dt_s=3.45e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "copper jc wide",
        DispersiveMaterial(
            4.6594824233,
            [
                Lorentz(5.6721310935, 5.3672981005e15, 5.7891633229e15),
                Drude(1.3612090574e16, 4.8063544351e13),
            ],
            name="copper jc wide",
        ),
        aliases=("cu jc wide",),
        category="metal",
        model="Lorentz-Drude (fit)",
        wavelength_range_um=(0.30, 1.94),
        citation=jc_cite,
        fit_error=0.1179,
        stable_50as=True,
        max_stable_dt_s=2.25e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )

    # ------------------------------------------------------------------
    # 4. Dielectrics, semiconductors, and polymers
    # ------------------------------------------------------------------
    _register(
        "bk7",
        _sellmeier_f2(
            "bk7",
            0.0,
            [1.03961212, 0.231792344, 1.01046945],
            [0.00600069867, 0.0200179144, 103.560653],
        ),
        aliases=("nbk7", "n-bk7", "schott bk7"),
        category="dielectric",
        model="Sellmeier",
        wavelength_range_um=(0.3, 2.5),
        citation=_cite(
            "SCHOTT AG",
            "Optical Glass Data Sheets: N-BK7 (517642.251)",
            "SCHOTT Optical Glass Catalog",
            2017,
        ),
        stable_50as=True,
        max_stable_dt_s=5.72e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "gaas",
        _sellmeier_eps_inf(
            "gaas",
            1.0 + 4.372514,
            [5.466742, 0.02429960, 1.957522],
            [0.4431307, 0.8746453, 36.9166],
        ),
        aliases=("gallium arsenide",),
        category="semiconductor",
        model="Sellmeier",
        wavelength_range_um=(0.97, 17.0),
        citation=_cite(
            "T. Skauli et al.",
            "Improved dispersion relations for GaAs and applications to nonlinear optics",
            "J. Appl. Phys. 94, 6447-6455",
            2003,
            "10.1063/1.1621740",
        ),
        stable_50as=True,
        max_stable_dt_s=3.45e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "si3n4",
        _sellmeier(
            "si3n4",
            [3.0249, 40314.0],
            [0.1353406, 1239.842],
        ),
        aliases=("silicon nitride", "sin", "si3n4 luke"),
        category="dielectric",
        model="Sellmeier",
        wavelength_range_um=(0.31, 5.504),
        citation=_cite(
            "K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, and M. Lipson",
            "Broadband mid-infrared frequency comb generation in a Si3N4 microresonator",
            "Opt. Lett. 40, 4823-4826",
            2015,
            "10.1364/OL.40.004823",
        ),
        stable_50as=True,
        max_stable_dt_s=7.21e-17,
        stability_note="Stable at dt=50 as (Courant=0.5); deep-IR pole at 1239.8 um is benign.",
    )
    _register(
        "si3n4 philipp",
        _sellmeier(
            "si3n4 philipp",
            [2.8939],
            [0.13967],
        ),
        aliases=("silicon nitride philipp",),
        category="dielectric",
        model="Sellmeier",
        wavelength_range_um=(0.207, 1.24),
        citation=_cite(
            "H. R. Philipp",
            "Optical properties of silicon nitride",
            "J. Electrochem. Soc. 120, 295-300",
            1973,
            "10.1149/1.2403440",
        ),
        stable_50as=True,
        max_stable_dt_s=7.65e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "alas",
        _sellmeier_eps_inf(
            "alas",
            1.0 + 1.0792,
            [6.0840, 1.900],
            [0.2822, 27.62],
        ),
        aliases=("aluminium arsenide", "aluminum arsenide"),
        category="semiconductor",
        model="Sellmeier",
        wavelength_range_um=(0.56, 2.2),
        citation=_cite(
            "R. E. Fern and A. Onton",
            "Refractive index of AlAs",
            "J. Appl. Phys. 42, 3499-3500",
            1971,
            "10.1063/1.1660760",
        ),
        stable_50as=True,
        max_stable_dt_s=1.22e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "inp",
        _sellmeier_eps_inf(
            "inp",
            1.0 + 6.255,
            [2.316, 2.765],
            [0.6263, 32.935],
        ),
        aliases=("indium phosphide",),
        category="semiconductor",
        model="Sellmeier",
        wavelength_range_um=(0.95, 10.0),
        citation=_cite(
            "G. D. Pettit and W. J. Turner",
            "Refractive index of InP",
            "J. Appl. Phys. 36, 2081",
            1965,
            "10.1063/1.1714410",
        ),
        stable_50as=True,
        max_stable_dt_s=5.85e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "germanium",
        _sellmeier_f2(
            "germanium",
            8.28156,
            [6.72880, 0.21307],
            [0.44105, 3870.1],
        ),
        aliases=("ge",),
        category="semiconductor",
        model="Sellmeier",
        wavelength_range_um=(2.5, 12.0),
        citation=_cite(
            "N. P. Barnes and M. S. Piltch",
            "Temperature-dependent Sellmeier coefficients and nonlinear optics average power limit for germanium",
            "J. Opt. Soc. Am. 69, 178-180",
            1979,
            "10.1364/JOSA.69.000178",
        ),
        stable_50as=True,
        max_stable_dt_s=4.55e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "c-si",
        DispersiveMaterial(
            1.0,
            [
                Lorentz(0.3048160228, 4.7996975310e15, 7.7074529778e14),
                Lorentz(1.6365104703, 5.0723172190e15, 9.7441210852e13),
                Lorentz(8.7539858311, 6.5588321811e15, 1.5825994361e09),
            ],
            name="c-si",
        ),
        aliases=("crystalline silicon", "si green"),
        category="semiconductor",
        model="Lorentz (fit)",
        wavelength_range_um=(0.4, 1.0),
        citation=_cite(
            "M. A. Green and M. J. Keevers",
            "Optical properties of intrinsic silicon at 300 K",
            "Prog. Photovolt.: Res. Appl. 3, 189-192",
            1995,
            "10.1002/pip.4670030303",
        ),
        fit_error=0.0025,
        stable_50as=True,
        max_stable_dt_s=7.21e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "pmma",
        _sellmeier_f2("pmma", 0.0, [1.1819], [0.011313]),
        aliases=("polymethyl methacrylate", "acrylic"),
        category="polymer",
        model="Sellmeier",
        wavelength_range_um=(0.4368, 1.052),
        citation=sultanova_cite,
        stable_50as=True,
        max_stable_dt_s=6.55e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "polycarbonate",
        _sellmeier_f2("polycarbonate", 0.0, [1.4182], [0.021304]),
        aliases=("pc",),
        category="polymer",
        model="Sellmeier",
        wavelength_range_um=(0.4368, 1.052),
        citation=sultanova_cite,
        stable_50as=True,
        max_stable_dt_s=8.12e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "polystyrene",
        _sellmeier_f2("polystyrene", 0.0, [1.4435], [0.020216]),
        aliases=("ps",),
        category="polymer",
        model="Sellmeier",
        wavelength_range_um=(0.4368, 1.052),
        citation=sultanova_cite,
        stable_50as=True,
        max_stable_dt_s=7.85e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "cellulose",
        _sellmeier_f2("cellulose", 0.0, [1.124], [0.011087]),
        aliases=(),
        category="polymer",
        model="Sellmeier",
        wavelength_range_um=(0.4368, 1.052),
        citation=sultanova_cite,
        stable_50as=True,
        max_stable_dt_s=6.65e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "y2o3",
        _sellmeier("y2o3", [2.578, 3.935], [0.1387, 22.936]),
        aliases=("yttria",),
        category="dielectric",
        model="Sellmeier",
        wavelength_range_um=(0.25, 9.6),
        citation=_cite(
            "Y. Nigara",
            "Measurement of the optical constants of yttrium oxide",
            "Jpn. J. Appl. Phys. 7, 404-408",
            1968,
            "10.1143/JJAP.7.404",
        ),
        stable_50as=True,
        max_stable_dt_s=7.85e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "yag",
        _sellmeier_f2("yag", 0.0, [2.28200, 3.27644], [0.01185, 282.734]),
        aliases=("y3al5o12",),
        category="dielectric",
        model="Sellmeier",
        wavelength_range_um=(0.4, 5.0),
        citation=_cite(
            "D. E. Zelmon, D. L. Small, and R. Page",
            "Refractive-index measurements of undoped yttrium aluminum garnet from 0.4 to 5.0 um",
            "Appl. Opt. 37, 4933-4935",
            1998,
            "10.1364/AO.37.004933",
        ),
        stable_50as=True,
        max_stable_dt_s=5.45e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "cdte",
        _sellmeier_f2("cdte", 4.68, [1.53], [0.366]),
        aliases=("cadmium telluride",),
        category="semiconductor",
        model="Sellmeier",
        wavelength_range_um=(0.86, 2.5),
        citation=_cite(
            "D. T. F. Marple",
            "Refractive index of ZnSe, ZnTe, and CdTe",
            "J. Appl. Phys. 35, 539-542",
            1964,
            "10.1063/1.1713411",
        ),
        stable_50as=True,
        max_stable_dt_s=5.65e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "tio2 ald",
        DispersiveMaterial(
            1.0,
            [
                Lorentz(0.6502066689, 6.3604408480e15, 1.7518817231e06),
                Lorentz(2.5272115766, 1.1598252071e16, 3.2064305162e07),
            ],
            name="tio2 ald",
        ),
        aliases=("titanium dioxide ald",),
        category="dielectric",
        model="Lorentz (fit)",
        wavelength_range_um=(0.4, 1.69),
        citation=_cite(
            "S. Sarkar et al.",
            "Hybridized guided-mode resonances via colloidal plasmonic self-assembled grating",
            "ACS Appl. Mater. Interfaces 11, 13752-13760",
            2019,
            "10.1021/acsami.8b20535",
        ),
        fit_error=1.0e-4,
        stable_50as=True,
        max_stable_dt_s=8.85e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "caf2",
        _sellmeier(
            "caf2",
            [0.5675888, 0.4710914, 3.8484723],
            [0.050263605, 0.1003909, 34.649040],
        ),
        aliases=("calcium fluoride",),
        category="dielectric",
        model="Sellmeier",
        wavelength_range_um=(0.23, 9.7),
        citation=_cite(
            "I. H. Malitson",
            "A redetermination of some optical properties of calcium fluoride",
            "Appl. Opt. 2, 1103-1107",
            1963,
            "10.1364/AO.2.001103",
        ),
        stable_50as=False,
        max_stable_dt_s=3.93e-17,
        stability_note="Unstable at dt=50 as (deep-UV pole at 50 nm); stable for dt <= 39 as.",
    )
    _register(
        "diamond",
        _sellmeier("diamond", [0.3306, 4.3356], [0.1750, 0.1060]),
        aliases=("c",),
        category="dielectric",
        model="Sellmeier",
        wavelength_range_um=(0.226, 0.76),
        citation=_cite(
            "F. Peter",
            "Uber Brechungsindizes und Absorptionskonstanten des Diamanten zwischen 644 und 226 mu",
            "Z. Phys. 15, 358-368",
            1923,
            "10.1007/BF01330487",
        ),
        stable_50as=False,
        max_stable_dt_s=4.28e-17,
        stability_note="Unstable at dt=50 as (UV pole at 106 nm); stable for dt <= 42 as.",
    )
    _register(
        "znse",
        _sellmeier(
            "znse",
            [4.45813734, 0.467216334, 2.89566290],
            [0.200859853, 0.391371166, 47.1362108],
        ),
        aliases=("zinc selenide",),
        category="semiconductor",
        model="Sellmeier",
        wavelength_range_um=(0.54, 18.2),
        citation=_cite(
            "J. Connolly, B. diBenedetto, and R. Donadio",
            "Specifications of raytran material",
            "Proc. SPIE 0181, 141-144",
            1979,
            "10.1117/12.957359",
        ),
        stable_50as=True,
        max_stable_dt_s=9.45e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "zns",
        _sellmeier_eps_inf(
            "zns",
            8.393 - 0.14383 / 0.2421**2 - 4430.99 / 36.71**2,
            [0.14383 / 0.2421**2, 4430.99 / 36.71**2],
            [0.2421, 36.71],
        ),
        aliases=("zinc sulfide",),
        category="semiconductor",
        model="Sellmeier",
        wavelength_range_um=(0.405, 13.0),
        citation=_cite(
            "M. Debenham",
            "Refractive indices of zinc sulfide in the 0.405-13-um wavelength range",
            "Appl. Opt. 23, 2238-2239",
            1984,
            "10.1364/AO.23.002238",
        ),
        stable_50as=True,
        max_stable_dt_s=6.85e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "gap",
        DispersiveMaterial(
            1.0,
            [Lorentz(8.0894464130, 7.1594356319e15, 3.0373671849e09)],
            name="gap",
        ),
        aliases=("gallium phosphide",),
        category="semiconductor",
        model="Lorentz (fit)",
        wavelength_range_um=(0.54, 4.0),
        citation=bond_cite,
        fit_error=0.0053,
        stable_50as=True,
        max_stable_dt_s=8.85e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )

    # ------------------------------------------------------------------
    # 5. Uniaxial crystals ('<name> o' and '<name> e')
    # ------------------------------------------------------------------
    bbo_cite = _cite(
        "G. Tamosauskas, G. Beresnevicius, D. Gadonas, and A. Dubietis",
        "Transmittance and phase matching of BBO crystal in the 3-5 um range and its application for the characterization of mid-infrared laser pulses",
        "Opt. Mater. Express 8, 1410-1418",
        2018,
        "10.1364/OME.8.001410",
    )
    _register(
        "bab2o4 o",
        _sellmeier_f2(
            "bab2o4 o",
            0.0,
            [0.90291, 0.83155, 0.76536],
            [0.003926, 0.018786, 60.01],
        ),
        aliases=("bbo o", "beta-bab2o4 o"),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.188, 5.2),
        citation=bbo_cite,
        stable_50as=False,
        max_stable_dt_s=4.34e-17,
        stability_note="Unstable at dt=50 as (UV pole at 62.7 nm); stable for dt <= 43 as.",
    )
    _register(
        "bab2o4 e",
        _sellmeier_f2(
            "bab2o4 e",
            0.0,
            [1.151075, 0.21803, 0.656],
            [0.007142, 0.02259, 263.0],
        ),
        aliases=("bbo e", "beta-bab2o4 e"),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.188, 5.2),
        citation=bbo_cite,
        stable_50as=True,
        max_stable_dt_s=5.72e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )

    linbo3_cite = _cite(
        "D. E. Zelmon, D. L. Small, and D. Jundt",
        "Infrared corrected Sellmeier coefficients for congruently grown lithium niobate and 5 mol. % magnesium oxide-doped lithium niobate",
        "J. Opt. Soc. Am. B 14, 3319-3322",
        1997,
        "10.1364/JOSAB.14.003319",
    )
    _register(
        "linbo3 o",
        _sellmeier_f2(
            "linbo3 o",
            0.0,
            [2.6734, 1.2290, 12.614],
            [0.01764, 0.05914, 474.60],
        ),
        aliases=("lithium niobate o",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.4, 5.0),
        citation=linbo3_cite,
        stable_50as=True,
        max_stable_dt_s=6.55e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "linbo3 e",
        _sellmeier_f2(
            "linbo3 e",
            0.0,
            [2.9804, 0.5981, 8.9543],
            [0.02047, 0.0666, 416.08],
        ),
        aliases=("lithium niobate e",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.4, 5.0),
        citation=linbo3_cite,
        stable_50as=True,
        max_stable_dt_s=6.85e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )

    _register(
        "cawo4 o",
        _sellmeier("cawo4 o", [2.5493, 0.9200], [0.1347, 10.815]),
        aliases=("calcium tungstate o",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.45, 4.0),
        citation=bond_cite,
        stable_50as=True,
        max_stable_dt_s=7.65e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "cawo4 e",
        _sellmeier("cawo4 e", [2.6041, 4.1237], [0.1379, 21.371]),
        aliases=("calcium tungstate e",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.45, 4.0),
        citation=bond_cite,
        stable_50as=True,
        max_stable_dt_s=7.75e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )

    _register(
        "caco3 o",
        _sellmeier_f2(
            "caco3 o",
            0.73358749,
            [0.96464345, 1.82831454],
            [1.94325203e-2, 120.0],
        ),
        aliases=("calcite o",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.204, 2.172),
        citation=ghosh_cite,
        stable_50as=True,
        max_stable_dt_s=1.08e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "caco3 e",
        _sellmeier_f2(
            "caco3 e",
            0.35859695,
            [0.82427830, 0.14429128],
            [1.06689543e-2, 120.0],
        ),
        aliases=("calcite e",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.204, 2.172),
        citation=ghosh_cite,
        stable_50as=True,
        max_stable_dt_s=8.12e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )

    _register(
        "quartz o",
        _sellmeier_f2(
            "quartz o",
            0.28604141,
            [1.07044083, 1.10202242],
            [1.00585997e-2, 100.0],
        ),
        aliases=("crystalline sio2 o", "sio2 o"),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.198, 2.0531),
        citation=ghosh_cite,
        stable_50as=True,
        max_stable_dt_s=7.45e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "quartz e",
        _sellmeier_f2(
            "quartz e",
            0.28851804,
            [1.09509924, 1.15662475],
            [1.02101864e-2, 100.0],
        ),
        aliases=("crystalline sio2 e", "sio2 e"),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.198, 2.0531),
        citation=ghosh_cite,
        stable_50as=True,
        max_stable_dt_s=7.45e-17,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )

    gan_cite = _cite(
        "A. S. Barker Jr. and M. Ilegems",
        "Infrared lattice vibrations and free-electron dispersion in GaN",
        "Phys. Rev. B 7, 743-750",
        1973,
        "10.1103/PhysRevB.7.743",
    )
    _register(
        "gan o",
        _sellmeier_eps_inf("gan o", 1.0 + 2.60, [1.75, 4.1], [0.256, 17.86]),
        aliases=("gallium nitride o",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.35, 10.0),
        citation=gan_cite,
        stable_50as=True,
        max_stable_dt_s=2.45e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "gan e",
        _sellmeier_eps_inf("gan e", 1.0 + 4.35, [5.08], [18.76]),
        aliases=("gallium nitride e",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.35, 10.0),
        citation=gan_cite,
        stable_50as=True,
        max_stable_dt_s=1.0e-14,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )

    aln_cite = _cite(
        "J. Pastrnak and L. Roskovcova",
        "Refraction index measurements on AlN single crystals",
        "Phys. Status Solidi 14, K5-K8",
        1966,
        "10.1002/pssb.19660140127",
    )
    _register(
        "aln o",
        _sellmeier_eps_inf("aln o", 1.0 + 2.1399, [1.3786, 3.861], [0.1715, 15.03]),
        aliases=("aluminium nitride o", "aluminum nitride o"),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.22, 5.0),
        citation=aln_cite,
        stable_50as=True,
        max_stable_dt_s=1.68e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "aln e",
        _sellmeier_eps_inf("aln e", 1.0 + 2.0729, [1.6173, 4.139], [0.1746, 15.03]),
        aliases=("aluminium nitride e", "aluminum nitride e"),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.22, 5.0),
        citation=aln_cite,
        stable_50as=True,
        max_stable_dt_s=1.58e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )

    sapphire_cite = _cite(
        "I. H. Malitson and M. J. Dodge",
        "Refractive index and birefringence of synthetic sapphire",
        "J. Opt. Soc. Am. 62, 1405",
        1972,
    )
    _register(
        "sapphire o",
        _sellmeier(
            "sapphire o",
            [1.4313493, 0.65054713, 5.3414021],
            [0.0726631, 0.1193242, 18.028251],
        ),
        aliases=("al2o3 o",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.20, 5.0),
        citation=sapphire_cite,
        stable_50as=False,
        max_stable_dt_s=4.34e-17,
        stability_note="Unstable at dt=50 as (UV pole at 72.7 nm); stable for dt <= 43 as.",
    )
    _register(
        "sapphire e",
        _sellmeier(
            "sapphire e",
            [1.5039759, 0.55069141, 6.5927379],
            [0.0740288, 0.1216529, 20.072248],
        ),
        aliases=("al2o3 e",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.20, 5.0),
        citation=sapphire_cite,
        stable_50as=False,
        max_stable_dt_s=4.38e-17,
        stability_note="Unstable at dt=50 as (UV pole at 74.0 nm); stable for dt <= 43 as.",
    )

    mgf2_cite = _cite(
        "M. J. Dodge",
        "Refractive properties of magnesium fluoride",
        "Appl. Opt. 23, 1980-1985",
        1984,
        "10.1364/AO.23.001980",
    )
    _register(
        "mgf2 o",
        _sellmeier(
            "mgf2 o",
            [0.48755108, 0.39875031, 2.3120353],
            [0.04338408, 0.09461442, 23.793604],
        ),
        aliases=("magnesium fluoride o",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.20, 7.0),
        citation=mgf2_cite,
        stable_50as=False,
        max_stable_dt_s=3.53e-17,
        stability_note="Unstable at dt=50 as (deep-UV pole at 43.4 nm); stable for dt <= 35 as.",
    )
    _register(
        "mgf2 e",
        _sellmeier(
            "mgf2 e",
            [0.41344023, 0.50497499, 2.4904862],
            [0.03684262, 0.09076162, 23.771995],
        ),
        aliases=("magnesium fluoride e",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.20, 7.0),
        citation=mgf2_cite,
        stable_50as=False,
        max_stable_dt_s=3.09e-17,
        stability_note="Unstable at dt=50 as (deep-UV pole at 36.8 nm); stable for dt <= 30 as.",
    )

    tio2_cite = _cite(
        "J. R. DeVore",
        "Refractive indices of rutile and sphalerite",
        "J. Opt. Soc. Am. 41, 416-419",
        1951,
        "10.1364/JOSA.41.000416",
    )
    _register(
        "tio2 o",
        _sellmeier_eps_inf(
            "tio2 o",
            5.913 - 0.2441 / 0.0803,
            [0.2441 / 0.0803],
            [np.sqrt(0.0803)],
        ),
        aliases=("rutile o",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.43, 1.53),
        citation=tio2_cite,
        stable_50as=True,
        max_stable_dt_s=2.85e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )
    _register(
        "tio2 e",
        _sellmeier_eps_inf(
            "tio2 e",
            7.197 - 0.3322 / 0.0843,
            [0.3322 / 0.0843],
            [np.sqrt(0.0843)],
        ),
        aliases=("rutile e",),
        category="uniaxial",
        model="Sellmeier",
        wavelength_range_um=(0.43, 1.53),
        citation=tio2_cite,
        stable_50as=True,
        max_stable_dt_s=3.15e-16,
        stability_note="Stable at dt=50 as (Courant=0.5).",
    )

    # ------------------------------------------------------------------
    # 7. RF / NFC Magnetic Ferrites (mu_inf, mu_poles)
    # ------------------------------------------------------------------
    nizn_target = (
        NIZN_FERRITE_SOURCE_DATA["mu_real"] - 1j * NIZN_FERRITE_SOURCE_DATA["mu_imag"]
    )
    nizn_mat = fit_mu_poles(
        NIZN_FERRITE_SOURCE_DATA["f_hz"],
        nizn_target,
        n_lorentz=1,
        n_debye=0,
        mu_inf=1.0,
        eps_inf=12.0,
        n_starts=8,
        name="nizn ferrite",
    )
    _register(
        "nizn ferrite",
        nizn_mat,
        aliases=("nfc ferrite",),
        category="ferrite",
        model="Lorentz (magnetic Snoek fit)",
        wavelength_range_um=(3.74741e6, 2.99792e8),
        citation=_cite(
            "J. Victoria, A. Suarez, P. A. Martinez, A. Alcarria, A. Gerfer, and J. Torres",
            "Improving the Efficiency of NFC Systems Through Optimizing the Sintered Ferrite Sheet Thickness Selection",
            "IEEE Trans. Electromagn. Compat. 62(4), 1504-1514",
            2020,
            "10.1109/TEMC.2020.3003800",
        ),
        fit_error=nizn_mat.fit_error,
        stable_50as=True,
        max_stable_dt_s=1.5e-9,
        stability_note="Stable for RF/NFC timesteps dt <= 1.5 ns at Courant <= 0.99.",
    )

    # ------------------------------------------------------------------
    # 8. EMI Shielding and Absorbing Materials (Tutorial 2.4, 10 MHz - 6 GHz)
    # ------------------------------------------------------------------
    # Class 1: Magnetic-particle-loaded silicone (Eccosorb RF-LB, Laird EM-ABS-DS-10132022)
    # 1 Debye + 1 Lorentz passive magnetic fit to digitized mu'(f), mu''(f) over 10 MHz - 1 GHz,
    # with eps' = 82.67 (magnetic loss only, rho_v = 7.45e11 Ohm*cm) inferred from attenuation over 0.1 - 1 GHz.
    _register(
        "eccosorb rf-lb",
        DispersiveMaterial(
            eps_inf=82.67,
            poles=[],
            sigma=0.0,
            mu_inf=1.0,
            mu_poles=[
                Debye(15.63870068, 2.36923131e-9),
                Lorentz(20.14228945, 1.15491234e10, 2.83645921e10),
            ],
            name="eccosorb rf-lb",
        ),
        aliases=("rf-lb", "laird rf-lb"),
        category="emi",
        model="Debye+Lorentz (magnetic fit + inferred permittivity)",
        wavelength_range_um=(4.99654e4, 2.99792e7),
        citation=_cite(
            "Laird Performance Materials",
            "Eccosorb RF-LB EMI Noise Suppression Absorber Technical Datasheet (Doc. EM-ABS-DS-10132022)",
            "Laird Performance Materials Datasheet",
            2022,
        ),
        fit_error=0.0036,
        stable_50as=True,
        max_stable_dt_s=1.0e-10,
        stability_note="Stable for RF/GHz timesteps dt <= 100 ps at Courant <= 0.99.",
    )

    # Class 2: Magnetic-flake-filled noise-suppression sheet (Fe85Si9.6Al5.4 Sendust flake composite)
    _register(
        "sendust flake sheet",
        DispersiveMaterial(
            eps_inf=112.0,
            poles=[Debye(6.0, 1.1e-10)],
            sigma=0.0,
            mu_inf=1.0,
            mu_poles=[
                Debye(42.0, 1.32629119e-9),
                Lorentz(27.0, 7.53982237e9, 1.13097336e10),
            ],
            name="sendust flake sheet",
        ),
        aliases=("fesial flake sheet", "sendust sheet"),
        category="emi",
        model="Debye+Lorentz (planar magnetic flake composite)",
        wavelength_range_um=(4.99654e4, 2.99792e7),
        citation=_cite(
            "S. Yoshida, M. Sato, E. Sugawara, and Y. Shimada",
            "Permeability and electromagnetic-interference characteristics of Fe-Si-Al alloy flakes-polymer composite",
            "J. Appl. Phys. 85(8), 4636-4638",
            1999,
            "10.1063/1.370432",
        ),
        fit_error=0.015,
        stable_50as=True,
        max_stable_dt_s=1.0e-10,
        stability_note="Stable for RF/GHz timesteps dt <= 100 ps at Courant <= 0.99.",
    )

    # Class 3: High-permittivity low-loss microwave ceramic (sintered polycrystalline rutile TiO2, eps_r ~ 100)
    _register(
        "tio2 ceramic",
        DispersiveMaterial(
            eps_inf=99.96,
            poles=[Debye(0.084, 5.30516477e-11)],
            sigma=0.0,
            mu_inf=1.0,
            name="tio2 ceramic",
        ),
        aliases=("rutile ceramic", "eps100 ceramic"),
        category="emi",
        model="Debye (low-loss high-eps ceramic)",
        wavelength_range_um=(4.99654e4, 2.99792e6),
        citation=_cite(
            "A. Templeton, X. Wang, S. J. Penn, S. J. Webb, L. F. Cohen, and N. McN. Alford",
            "Microwave Dielectric Loss of Titanium Oxide",
            "J. Am. Ceram. Soc. 83(1), 95-100",
            2000,
            "10.1111/j.1151-2916.2000.tb01154.x",
        ),
        fit_error=1.0e-4,
        stable_50as=True,
        max_stable_dt_s=1.0e-9,
        stability_note="Unconditionally stable Crank-Nicolson Debye update for Courant <= 1.",
    )

    # Class 4: Carbon-nanotube loaded silicone rubber above percolation (3 wt% MWCNT/PDMS, sigma = 25 S/m)
    _register(
        "cnt silicone",
        DispersiveMaterial(
            eps_inf=14.0,
            poles=[Debye(8.0, 1.06103295e-10)],
            sigma=25.0,
            mu_inf=1.0,
            name="cnt silicone",
        ),
        aliases=("cnt rubber", "mwcnt pdms"),
        category="emi",
        model="Debye+Conductivity (percolated CNT elastomer)",
        wavelength_range_um=(4.99654e4, 2.99792e6),
        citation=_cite(
            "M. H. Al-Saleh and U. Sundararaj",
            "Electromagnetic interference shielding mechanisms of CNT/polymer composites",
            "Carbon 47(7), 1738-1746",
            2009,
            "10.1016/j.carbon.2009.02.030",
        ),
        fit_error=0.01,
        stable_50as=True,
        max_stable_dt_s=1.0e-9,
        stability_note="Semi-implicit conductivity + Crank-Nicolson Debye update; stable for Courant <= 1.",
    )

    # Class 5: Conductive elastomer gasket (Ni-graphite filled silicone, rho_v = 0.50 Ohm*cm -> sigma = 200 S/m)
    _register(
        "ni-graphite silicone",
        DispersiveMaterial(
            eps_inf=12.0,
            poles=[],
            sigma=200.0,
            mu_inf=1.0,
            name="ni-graphite silicone",
        ),
        aliases=("conductive elastomer gasket", "ni-c silicone"),
        category="emi",
        model="Conductivity (conductive elastomer gasket)",
        wavelength_range_um=(4.99654e4, 2.99792e7),
        citation=_cite(
            "H. W. Ott",
            "Electromagnetic Compatibility Engineering",
            "John Wiley & Sons, Hoboken, NJ",
            2009,
            "10.1002/9780470508510",
        ),
        stable_50as=True,
        max_stable_dt_s=1.0e-9,
        stability_note="Semi-implicit conductivity update; stable for Courant <= 1.",
    )

    # Class 6: Reference copper foil (sigma = 5.8e7 S/m, classical Schelkunoff shielding reference)
    _register(
        "copper foil",
        DispersiveMaterial(
            eps_inf=1.0,
            poles=[],
            sigma=5.8e7,
            mu_inf=1.0,
            name="copper foil",
        ),
        aliases=("cu foil", "copper shield"),
        category="emi",
        model="Conductivity (Schelkunoff metal shield)",
        wavelength_range_um=(4.99654e4, 2.99792e7),
        citation=_cite(
            "S. A. Schelkunoff",
            "Electromagnetic Waves",
            "D. Van Nostrand, New York",
            1943,
        ),
        stable_50as=True,
        max_stable_dt_s=1.0e-9,
        stability_note="Semi-implicit conductivity update; use subcell add_sheet when foil thickness < grid cell.",
    )


def info(name):
    """
    Return the metadata dictionary for a dispersive material or alias.

    Parameters
    ----------
    name : str
        Canonical material name or alias (case-insensitive), e.g. ``'gold'``,
        ``'au'``, ``'si3n4'``, ``'sapphire o'``.

    Returns
    -------
    meta : dict
        Dictionary with keys:

        - ``'name'`` : canonical lowercase material name (`str`)
        - ``'aliases'`` : registered aliases (`tuple` of `str`)
        - ``'category'`` : ``'metal'``, ``'dielectric'``, ``'semiconductor'``,
          ``'polymer'``, ``'uniaxial'``, ``'rf_substrate'``, ``'ferrite'``,
          or ``'emi'`` (`str`)
        - ``'model'`` : dispersion model type (`str`)
        - ``'wavelength_range_um'`` : validity range ``(lambda_min, lambda_max)``
          in micrometres (`tuple` of `float`)
        - ``'fit_error'`` : relative RMS fit error if fitted with `fit_poles`,
          or ``None`` for analytical formulas (`float` or `None`)
        - ``'citation'`` : dictionary with ``'authors'``, ``'title'``,
          ``'journal'``, ``'year'``, ``'doi'``, and ``'reference'`` (`dict`)
        - ``'stable_50as'`` : whether ``material.is_stable(5e-17, 0.5)`` is
          ``True`` (`bool`)
        - ``'max_stable_dt_s'`` : approximate maximum stable timestep [s] at
          ``courant=0.5`` (`float`)
        - ``'stability_note'`` : note on ADE timestep stability (`str`)

    Raises
    ------
    KeyError
        If `name` is not registered in the material library.
    """
    key = name.strip().lower()
    if key not in _ALIASES:
        raise KeyError(f"Unknown dispersive material {name!r}")
    meta = dict(_METADATA[_ALIASES[key]])
    meta["citation"] = dict(meta["citation"])
    return meta


def list_materials(include_aliases=False, category=None, include_magnetic=False):
    """
    Return a sorted list of material names registered in `dispersive_lib`.

    Parameters
    ----------
    include_aliases : bool, optional
        If ``False`` (default), return only canonical material names. If
        ``True``, include all chemical-formula and shorthand aliases.
    category : str, optional
        If given, filter materials by category (``'metal'``, ``'dielectric'``,
        ``'semiconductor'``, ``'polymer'``, ``'uniaxial'``, ``'rf_substrate'``,
        ``'ferrite'``, ``'emi'``).
    include_magnetic : bool, optional
        If ``False`` (default) and ``category is None``, return only electric
        dispersive materials (`poles` non-empty). Set ``True`` or
        ``category='ferrite'`` / ``category='emi'`` to include magnetic and
        RF EMI shielding/absorbing materials.

    Returns
    -------
    names : list of str
        Sorted material names.
    """
    cat = category.strip().lower() if category is not None else None
    if include_aliases:
        return sorted(
            alias
            for alias, canon in _ALIASES.items()
            if (cat is not None and _METADATA[canon]["category"] == cat)
            or (
                cat is None
                and (
                    include_magnetic
                    or _METADATA[canon]["category"] not in ("ferrite", "emi")
                )
            )
        )
    return sorted(
        canon
        for canon, meta in _METADATA.items()
        if (cat is not None and meta["category"] == cat)
        or (
            cat is None
            and (include_magnetic or meta["category"] not in ("ferrite", "emi"))
        )
    )


def uniaxial_crystal(name, axis="z"):
    """
    Return the ``(eps_x, eps_y, eps_z)`` triple of `DispersiveMaterial` objects
    for a uniaxial crystal with its optic (extraordinary) axis along `axis`.

    Parameters
    ----------
    name : str
        Base name or alias of the uniaxial crystal (e.g. ``'bab2o4'``,
        ``'bbo'``, ``'linbo3'``, ``'cawo4'``, ``'caco3'``, ``'calcite'``,
        ``'quartz'``, ``'crystalline sio2'``, ``'gan'``, ``'aln'``,
        ``'sapphire'``, ``'al2o3'``, ``'mgf2'``, ``'tio2'``, ``'rutile'``).
    axis : {'x', 'y', 'z', 0, 1, 2}, optional
        Cartesian direction of the crystal optic (c) axis. Default is ``'z'``,
        which returns ``(mat_o, mat_o, mat_e)``.

    Returns
    -------
    triple : tuple of (DispersiveMaterial, DispersiveMaterial, DispersiveMaterial)
        ``(eps_x, eps_y, eps_z)`` permutation of ``(ordinary, ordinary,
        extraordinary)`` corresponding to `axis`.

    Notes
    -----
    Diagonal-anisotropic dispersive permittivity is not yet supported directly
    by `SolverFIT3D.add_dispersive_material`, `SolverFIT3D._apply_stl_dispersive`,
    or `wakis.tensor_dispersive` (each solid or box currently takes a single
    scalar `DispersiveMaterial` applied to all three electric-field components).
    For fixed-polarization simulations, pass either the ordinary (`mat_o`) or
    extraordinary (`mat_e`) entry directly; for full uniaxial simulations on a
    Cartesian box, individual E-component edge slices can be registered via
    `SolverFIT3D._add_dispersive_edges`.
    """
    base = name.strip().lower()
    for suffix in (" o", " e"):
        if base.endswith(suffix):
            base = base[: -len(suffix)].strip()
            break
    key_o = f"{base} o"
    key_e = f"{base} e"
    if key_o not in dispersive_lib or key_e not in dispersive_lib:
        raise KeyError(
            f"Unknown uniaxial crystal {name!r}; expected base name with "
            f"'{key_o}' and '{key_e}' in dispersive_lib."
        )
    mat_o = dispersive_lib[key_o]
    mat_e = dispersive_lib[key_e]
    axis_map = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}
    if isinstance(axis, str):
        axis = axis.strip().lower()
    if axis not in axis_map:
        raise ValueError(f"axis must be 'x', 'y', or 'z' (got {axis!r})")
    idx = axis_map[axis]
    triple = [mat_o, mat_o, mat_o]
    triple[idx] = mat_e
    return tuple(triple)


_build_library()
