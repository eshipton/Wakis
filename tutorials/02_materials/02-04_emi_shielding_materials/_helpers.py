# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Analytical transfer-matrix, Schelkunoff shielding decomposition, PEC-backed
reflection loss, Rozanov causality bound, Wakis 3D FIT (`SolverFIT3D`) runners,
and publication figure utilities for Tutorial 2.4 (`02-04_emi_shielding_materials`).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0
from scipy.optimize import minimize_scalar

from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.dispersive import Debye, DispersiveMaterial, Lorentz, dispersive_lib
from wakis.estimate import estimate, preflight
from wakis.field_monitors import DFTMonitor, FluxMonitor
from wakis.material_library import ECCOSORB_RF_LB_SOURCE_DATA, fit_mu_poles, info
from wakis.sources import CurrentSource, GaussianSource
from wakis.subcell import metal_sheet, sheet_transmission_reflection

Z0 = float(np.sqrt(mu0 / eps0))
NP_TO_DB = float(20.0 / np.log(10.0))  # 8.685889638 dB/Np

#: Canonical order and display metadata of the 6 EMI material classes
EMI_CLASSES: list[dict] = [
    {
        "key": "eccosorb rf-lb",
        "short": "Eccosorb RF-LB",
        "class_label": "Class 1 · Magnetic-particle silicone (Eccosorb RF-LB)",
        "mechanism": "Magnetic + dielectric absorption (matched |Z|)",
        "thicknesses_mm": [1.0, 2.0, 3.0],
        "quick_thicknesses_mm": [1.0, 2.0],
        "use_thin_sheet": False,
    },
    {
        "key": "sendust flake sheet",
        "short": "Sendust flake sheet",
        "class_label": "Class 2 · FeSiAl (Sendust) magnetic-flake sheet",
        "mechanism": "Planar-anisotropy magnetic absorption + reflection",
        "thicknesses_mm": [0.5, 1.0, 2.0],
        "quick_thicknesses_mm": [0.5, 1.0],
        "use_thin_sheet": False,
    },
    {
        "key": "tio2 ceramic",
        "short": "TiO2 ceramic (eps_r=100)",
        "class_label": "Class 3 · High-permittivity ceramic (rutile TiO2, eps_r ~ 100)",
        "mechanism": "Pure reactive reflection (Fabry-Perot etalon, A ~ 0)",
        "thicknesses_mm": [1.0, 2.0, 3.0],
        "quick_thicknesses_mm": [1.0, 2.0],
        "use_thin_sheet": False,
    },
    {
        "key": "cnt silicone",
        "short": "CNT silicone (3 wt%)",
        "class_label": "Class 4 · Percolated MWCNT/PDMS rubber (sigma = 25 S/m)",
        "mechanism": "Mixed conductive reflection + skin-depth absorption",
        "thicknesses_mm": [0.5, 1.0, 2.0],
        "quick_thicknesses_mm": [0.5, 1.0],
        "use_thin_sheet": False,
    },
    {
        "key": "ni-graphite silicone",
        "short": "Ni-graphite gasket",
        "class_label": "Class 5 · Conductive elastomer gasket (Ni-graphite, sigma = 200 S/m)",
        "mechanism": "Reflection-dominated + strong skin-depth attenuation",
        "thicknesses_mm": [0.5, 1.0, 1.5],
        "quick_thicknesses_mm": [0.5, 1.0],
        "use_thin_sheet": False,
    },
    {
        "key": "copper foil",
        "short": "Copper foil (subcell)",
        "class_label": "Class 6 · Reference copper foil (sigma = 5.8e7 S/m, F3 thin sheet)",
        "mechanism": "Extreme impedance-mismatch reflection + skin-effect SE_A",
        "thicknesses_mm": [1.0e-5, 5.0e-5, 2.0e-4],  # 10 nm, 50 nm, 200 nm metallized films
        "quick_thicknesses_mm": [1.0e-5, 5.0e-5],
        "use_thin_sheet": True,
    },
]


class SlabBox:
    """Lightweight box structure with ``bounds``, ``material``, and ``name`` for `wakis.schematic`."""

    def __init__(self, z_lo: float, z_hi: float, xy_half: float, material, name: str):
        self.z_lo = float(z_lo)
        self.z_hi = float(z_hi)
        self.xy_half = float(xy_half)
        self.material = material
        self.name = str(name)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([-self.xy_half, -self.xy_half, self.z_lo], dtype=float),
            np.array([self.xy_half, self.xy_half, self.z_hi], dtype=float),
        )


def load_eccosorb_rf_lb_csv(data_dir: Path) -> dict[str, np.ndarray]:
    """Load the digitized Eccosorb RF-LB permeability and attenuation CSV files from ``data_dir``.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``eccosorb_rf_lb_permeability.csv`` and
        ``eccosorb_rf_lb_attenuation.csv``.

    Returns
    -------
    data : dict of str to ndarray
        Dictionary with keys ``f_mu_hz``, ``f_mu_ghz``, ``mu_real``,
        ``mu_imag``, ``f_att_hz``, ``f_att_ghz``, and ``atten_db_cm``.
    """
    data_dir = Path(data_dir)
    perm_path = data_dir / "eccosorb_rf_lb_permeability.csv"
    att_path = data_dir / "eccosorb_rf_lb_attenuation.csv"
    if not perm_path.is_file() or not att_path.is_file():
        return {
            "f_mu_hz": ECCOSORB_RF_LB_SOURCE_DATA["f_mu_hz"].copy(),
            "f_mu_ghz": ECCOSORB_RF_LB_SOURCE_DATA["f_mu_hz"] * 1e-9,
            "mu_real": ECCOSORB_RF_LB_SOURCE_DATA["mu_real"].copy(),
            "mu_imag": ECCOSORB_RF_LB_SOURCE_DATA["mu_imag"].copy(),
            "f_att_hz": ECCOSORB_RF_LB_SOURCE_DATA["f_att_hz"].copy(),
            "f_att_ghz": ECCOSORB_RF_LB_SOURCE_DATA["f_att_hz"] * 1e-9,
            "atten_db_cm": ECCOSORB_RF_LB_SOURCE_DATA["atten_db_cm"].copy(),
        }

    f_mu_ghz, mu_r, mu_i = [], [], []
    with open(perm_path, encoding="utf-8") as fh:
        reader = csv.DictReader(line for line in fh if not line.lstrip().startswith("#"))
        for row in reader:
            f_mu_ghz.append(float(row["f_ghz"]))
            mu_r.append(float(row["mu_real"]))
            mu_i.append(float(row["mu_imag"]))

    f_att_ghz, att_db = [], []
    with open(att_path, encoding="utf-8") as fh:
        reader = csv.DictReader(line for line in fh if not line.lstrip().startswith("#"))
        for row in reader:
            f_att_ghz.append(float(row["f_ghz"]))
            att_db.append(float(row["attenuation_db_per_cm"]))

    f_mu_ghz_arr = np.asarray(f_mu_ghz, dtype=float)
    f_att_ghz_arr = np.asarray(f_att_ghz, dtype=float)
    return {
        "f_mu_hz": f_mu_ghz_arr * 1e9,
        "f_mu_ghz": f_mu_ghz_arr,
        "mu_real": np.asarray(mu_r, dtype=float),
        "mu_imag": np.asarray(mu_i, dtype=float),
        "f_att_hz": f_att_ghz_arr * 1e9,
        "f_att_ghz": f_att_ghz_arr,
        "atten_db_cm": np.asarray(att_db, dtype=float),
    }


def passive_refractive_index_and_impedance(
    eps_r: np.ndarray | complex | float,
    mu_r: np.ndarray | complex | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute passive complex refractive index ``n`` and normalized impedance ``z_rel = Z/Z0``.

    In Wakis's ``exp(+i*omega*t)`` convention, a passive medium has
    ``Im(eps_r) <= 0``, ``Im(mu_r) <= 0``, ``Re(n) >= 0``, ``Im(n) <= 0``,
    and ``Re(z_rel) >= 0`` so a forward wave ``exp(-i*k0*n*z)`` decays for ``z > 0``.

    Parameters
    ----------
    eps_r, mu_r : complex or array_like
        Complex relative permittivity and permeability in ``exp(+i*omega*t)``.

    Returns
    -------
    n : complex ndarray
        Complex refractive index ``sqrt(eps_r * mu_r)`` with ``Im(n) <= 0``.
    z_rel : complex ndarray
        Normalized wave impedance ``sqrt(mu_r / eps_r)`` with ``Re(z_rel) >= 0``.
    """
    eps_c = np.asarray(eps_r, dtype=np.complex128)
    mu_c = np.asarray(mu_r, dtype=np.complex128)
    n = np.sqrt(eps_c * mu_c)
    n = np.where((np.imag(n) > 0.0) | ((np.imag(n) == 0.0) & (np.real(n) < 0.0)), -n, n)
    z_rel = np.sqrt(mu_c / eps_c)
    z_rel = np.where(np.real(z_rel) < 0.0, -z_rel, z_rel)
    return n, z_rel


def attenuation_db_per_cm(
    f_hz: np.ndarray | float,
    eps_r: np.ndarray | complex | float,
    mu_r: np.ndarray | complex | float,
) -> np.ndarray:
    """Plane-wave power/amplitude attenuation rate ``alpha(f)`` in ``[dB/cm]``.

    Given ``E(z) = E0 * exp(-i * (omega/c0) * n(f) * z)`` with ``n = n' - i*kappa``
    (``kappa = -Im(n) >= 0``), the field amplitude decays as ``exp(-alpha_Np * z)``
    with ``alpha_Np = (omega/c0) * (-Im(sqrt(eps_r * mu_r)))`` [Np/m], corresponding
    to ``8.6859 * alpha_Np * 1e-2`` [dB/cm].

    Parameters
    ----------
    f_hz : float or array_like
        Frequency [Hz].
    eps_r, mu_r : complex or array_like
        Complex relative permittivity and permeability in ``exp(+i*omega*t)``.

    Returns
    -------
    alpha_db_cm : ndarray
        Attenuation rate [dB/cm].
    """
    omega = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
    n, _ = passive_refractive_index_and_impedance(eps_r, mu_r)
    alpha_np_m = (omega / c0) * (-np.imag(n))
    return NP_TO_DB * alpha_np_m * 1e-2


def infer_eps_real_from_attenuation(
    f_overlap_hz: np.ndarray,
    mu_overlap: np.ndarray,
    atten_overlap_db_cm: np.ndarray,
    tan_delta_e: float = 0.01,
) -> dict[str, float]:
    """Infer ``eps'`` over the 0.1-1.0 GHz overlap band from measured ``mu(f)`` and ``alpha(f)``.

    Minimizes the mean-squared residual between ``attenuation_db_per_cm(f, eps'*(1 - i*tan_delta_e), mu)``
    and the datasheet attenuation curve over ``f_overlap_hz``.

    Parameters
    ----------
    f_overlap_hz : ndarray
        Overlap frequencies [Hz] where both ``mu(f)`` and ``alpha(f)`` are measured.
    mu_overlap : complex ndarray
        Complex relative permeability ``mu'(f) - i*mu''(f)`` at ``f_overlap_hz``.
    atten_overlap_db_cm : ndarray
        Measured datasheet attenuation [dB/cm] at ``f_overlap_hz``.
    tan_delta_e : float, optional
        Assumed electric loss tangent ``eps'' / eps' >= 0``. Default is 0.01.

    Returns
    -------
    result : dict
        Dictionary with ``tan_delta_e``, ``eps_real``, ``eps_imag``,
        ``rms_overlap_db_cm``, and ``rel_rms_overlap``.
    """
    td = float(tan_delta_e)

    def obj(eps_r_val: float) -> float:
        eps_c = float(eps_r_val) * (1.0 - 1j * td)
        pred = attenuation_db_per_cm(f_overlap_hz, eps_c, mu_overlap)
        return float(np.mean((pred - atten_overlap_db_cm) ** 2))

    sol = minimize_scalar(obj, bounds=(1.0, 500.0), method="bounded")
    eps_real = float(sol.x)
    rms_db = float(np.sqrt(sol.fun))
    rel_rms = float(rms_db / np.sqrt(np.mean(atten_overlap_db_cm**2)))
    return {
        "tan_delta_e": td,
        "eps_real": eps_real,
        "eps_imag": eps_real * td,
        "rms_overlap_db_cm": rms_db,
        "rel_rms_overlap": rel_rms,
    }


def exact_slab_rt(
    f_hz: np.ndarray | float,
    eps_r: np.ndarray | complex | float,
    mu_r: np.ndarray | complex | float,
    d: float,
) -> dict[str, np.ndarray]:
    """Exact normal-incidence transfer-matrix reflection, transmission, and absorption for a slab in vacuum.

    Parameters
    ----------
    f_hz : float or array_like
        Frequency [Hz].
    eps_r, mu_r : complex or array_like
        Complex relative permittivity and permeability in ``exp(+i*omega*t)``.
    d : float
        Slab thickness [m].

    Returns
    -------
    out : dict of str to ndarray
        Dictionary with complex amplitudes ``r``, ``t`` and power fractions
        ``R = |r|^2``, ``T = |t|^2``, ``A = 1 - R - T``.
    """
    omega = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
    n, z_rel = passive_refractive_index_and_impedance(eps_r, mu_r)
    r12 = (z_rel - 1.0) / (z_rel + 1.0)
    P = np.exp(-1j * (omega / c0) * n * float(d))
    denom = 1.0 - (r12**2) * (P**2)
    r = r12 * (1.0 - P**2) / denom
    t = (1.0 - r12**2) * P / denom
    R = np.abs(r) ** 2
    T = np.abs(t) ** 2
    A = 1.0 - R - T
    return {"r": r, "t": t, "R": R, "T": T, "A": A, "n": n, "z_rel": z_rel, "r12": r12, "P": P}


def schelkunoff_decomposition(
    f_hz: np.ndarray | float,
    eps_r: np.ndarray | complex | float,
    mu_r: np.ndarray | complex | float,
    d: float,
) -> dict[str, np.ndarray]:
    """Decompose total shielding effectiveness ``SE = -10*log10|T|^2`` into Schelkunoff's three terms.

    From the exact slab transmission amplitude
    ``t = [(4 * z_rel) / (1 + z_rel)^2] * P / (1 - r12^2 * P^2)`` with
    ``P = exp(-i*(omega/c0)*n*d)`` and ``r12 = (z_rel - 1)/(z_rel + 1)``, taking
    ``-20*log10|t|`` yields the exact algebraic identity (Schelkunoff 1943):

        SE(f, d) = SE_R(f) + SE_A(f, d) + SE_M(f, d),

    where:
    * ``SE_R = 20*log10(|1 + z_rel|^2 / (4*|z_rel|))`` [dB] is the single-pass
      front+back impedance-mismatch reflection loss,
    * ``SE_A = -20*log10|P| = 8.6859 * (omega/c0) * (-Im(n)) * d`` [dB] is the
      internal absorption (attenuation) loss across thickness ``d``, and
    * ``SE_M = 20*log10|1 - r12^2 * P^2|`` [dB] is the multiple-reflection
      correction inside the slab.

    Parameters
    ----------
    f_hz : float or array_like
        Frequency [Hz].
    eps_r, mu_r : complex or array_like
        Complex relative permittivity and permeability in ``exp(+i*omega*t)``.
    d : float
        Slab thickness [m].

    Returns
    -------
    out : dict of str to ndarray
        Dictionary with ``SE_total_db``, ``SE_R_db``, ``SE_A_db``, ``SE_M_db``,
        ``R``, ``T``, and ``A``.
    """
    tm = exact_slab_rt(f_hz, eps_r, mu_r, d)
    z_rel = tm["z_rel"]
    r12 = tm["r12"]
    P = tm["P"]
    T = np.maximum(tm["T"], 1e-300)

    se_total = -10.0 * np.log10(T)
    se_r = np.broadcast_to(
        20.0 * np.log10(np.abs(1.0 + z_rel) ** 2 / np.maximum(4.0 * np.abs(z_rel), 1e-300)),
        np.shape(se_total),
    ).copy()
    se_a = -20.0 * np.log10(np.maximum(np.abs(P), 1e-300))
    se_m = 20.0 * np.log10(np.maximum(np.abs(1.0 - (r12**2) * (P**2)), 1e-300))
    return {
        "SE_total_db": se_total,
        "SE_R_db": se_r,
        "SE_A_db": se_a,
        "SE_M_db": se_m,
        "R": tm["R"],
        "T": tm["T"],
        "A": tm["A"],
    }


def exact_pec_backed_absorber(
    f_hz: np.ndarray | float,
    eps_r: np.ndarray | complex | float,
    mu_r: np.ndarray | complex | float,
    d: float,
) -> dict[str, np.ndarray]:
    """Exact normal-incidence reflection coefficient and reflection loss of a PEC-backed slab.

    In Wakis's ``exp(+i*omega*t)`` convention, a short-circuited transmission line
    of characteristic impedance ``Z0 * sqrt(mu_r / eps_r)`` and electrical length
    ``beta*d = (omega/c0)*n*d`` has normalized input impedance

        z_in(f, d) = i * sqrt(mu_r / eps_r) * tan((omega/c0) * n * d),

    and reflection coefficient ``Gamma(f, d) = (z_in - 1) / (z_in + 1)``, with
    reflection loss ``RL = -20*log10|Gamma| = -10*log10(R)`` [dB].

    Parameters
    ----------
    f_hz : float or array_like
        Frequency [Hz].
    eps_r, mu_r : complex or array_like
        Complex relative permittivity and permeability in ``exp(+i*omega*t)``.
    d : float
        Absorber thickness [m] backed by a PEC ground plane.

    Returns
    -------
    out : dict of str to ndarray
        Dictionary with ``z_in``, ``gamma``, ``R``, ``A``, and ``RL_db``.
    """
    omega = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
    n, z_rel = passive_refractive_index_and_impedance(eps_r, mu_r)
    z_in = 1j * z_rel * np.tan((omega / c0) * n * float(d))
    gamma = (z_in - 1.0) / (z_in + 1.0)
    R = np.abs(gamma) ** 2
    RL_db = -10.0 * np.log10(np.maximum(R, 1e-30))
    return {"z_in": z_in, "gamma": gamma, "R": R, "A": 1.0 - R, "RL_db": RL_db}


def evaluate_rozanov_bound(
    f_hz: np.ndarray,
    gamma: np.ndarray,
    mu_s: float,
    d: float,
) -> dict[str, float]:
    """Compare the wavelength-integrated reflection loss against Rozanov's causal bound.

    Rozanov (IEEE Trans. Antennas Propag. 48, 1230, 2000, doi:10.1109/8.884491)
    proved from Kramers-Kronig causality of a metal-backed slab of thickness ``d``
    and static relative permeability ``mu_s = lim_{f->0} Re(mu(f))`` that:

        | int_0^inf ln|Gamma(lambda)| d(lambda) | <= 2 * pi^2 * mu_s * d.

    Parameters
    ----------
    f_hz : ndarray
        Strictly increasing frequency array [Hz].
    gamma : complex ndarray
        Complex reflection coefficient ``Gamma(f)`` across ``f_hz``.
    mu_s : float
        Static relative permeability ``mu_s >= 1``.
    d : float
        Slab thickness [m].

    Returns
    -------
    out : dict of str to float
        Dictionary with ``integral_m``, ``rozanov_bound_m``, ``nonmag_bound_m``,
        ``utilization_fraction``, and ``exceeds_nonmag_bound``.
    """
    lam = c0 / np.asarray(f_hz, dtype=float)
    order = np.argsort(lam)
    lam_sorted = lam[order]
    abs_ln_gamma = -np.log(np.clip(np.abs(gamma[order]), 1e-15, 1.0))
    integral_m = float(np.trapezoid(abs_ln_gamma, lam_sorted))
    roz_bound_m = float(2.0 * (np.pi**2) * float(mu_s) * float(d))
    nonmag_bound_m = float(2.0 * (np.pi**2) * 1.0 * float(d))
    return {
        "integral_m": integral_m,
        "rozanov_bound_m": roz_bound_m,
        "nonmag_bound_m": nonmag_bound_m,
        "utilization_fraction": integral_m / max(roz_bound_m, 1e-30),
        "exceeds_nonmag_bound": bool(integral_m > nonmag_bound_m),
    }


def build_graded_z_grid(
    z_lo: float,
    z_hi: float,
    dz_slab: float,
    z_min_dom: float,
    z_max_dom: float,
    dz_vac: float,
    ratio: float = 1.16,
    collar_cells: int = 4,
) -> tuple[np.ndarray, float]:
    """Construct a locally refined 1D ``z`` node array with exact interface alignment.

    Places a uniform fine grid of spacing ``dz_actual <= dz_slab`` across the
    slab ``[z_lo, z_hi]`` plus a symmetric collar of ``collar_cells`` vacuum
    cells on both sides (so dual cell lengths across ``z_lo`` and ``z_hi`` are
    symmetric and second-order accurate), then grades geometrically by
    ``ratio`` per cell up to ``dz_vac`` in the outer vacuum regions.

    Parameters
    ----------
    z_lo, z_hi : float
        Slab lower and upper coordinates [m].
    dz_slab : float
        Target maximum cell size inside the slab [m].
    z_min_dom, z_max_dom : float
        Domain lower and upper coordinates [m].
    dz_vac : float
        Maximum cell size in the outer vacuum region [m].
    ratio : float, optional
        Geometric stretching factor between adjacent cells (default 1.16).
    collar_cells : int, optional
        Number of fine cells of size ``dz_actual`` kept outside each slab face.

    Returns
    -------
    z_nodes : ndarray
        Sorted 1D array of grid node positions [m].
    dz_actual : float
        Exact uniform cell size inside the slab [m].
    """
    d = float(z_hi - z_lo)
    n_slab = max(4, int(np.ceil(d / float(dz_slab) - 1e-9)))
    dz_actual = d / n_slab
    z_fine_lo = float(z_lo) - collar_cells * dz_actual
    z_fine_hi = float(z_hi) + collar_cells * dz_actual
    n_fine = n_slab + 2 * collar_cells
    z_fine = np.linspace(z_fine_lo, z_fine_hi, n_fine + 1)

    def _grade_side(z_start: float, z_stop: float) -> np.ndarray:
        pts = [z_start]
        dz = dz_actual
        direction = 1.0 if z_stop > z_start else -1.0
        while abs(z_stop - pts[-1]) > 1.25 * dz:
            dz = min(dz * ratio, float(dz_vac))
            pts.append(pts[-1] + direction * dz)
        rem = abs(z_stop - pts[-1])
        n_rem = max(1, int(np.ceil(rem / float(dz_vac))))
        tail = np.linspace(pts[-1], z_stop, n_rem + 1)[1:]
        return np.sort(np.concatenate([pts, tail]))

    z_left = _grade_side(z_fine_lo, z_min_dom)[:-1]
    z_right = _grade_side(z_fine_hi, z_max_dom)[1:]
    return np.concatenate([z_left, z_fine, z_right]), float(dz_actual)


def compute_slab_resolution_metrics(
    mat: DispersiveMaterial,
    d: float,
    freqs: np.ndarray,
    dz_slab: float,
    use_thin_sheet: bool = False,
) -> dict[str, float]:
    """Compute wavelength, feature, and skin-depth resolution metrics inside a slab."""
    fmax = float(np.max(freqs))
    lam0_min = float(c0 / fmax)
    if use_thin_sheet:
        return {
            "n_max": 1.0,
            "lam_mat_min_m": lam0_min,
            "skin_depth_min_m": float(np.sqrt(2.0 / (2.0 * np.pi * fmax * mu0 * max(mat.sigma, 1.0)))),
            "cpw_slab": float(lam0_min / dz_slab),
            "cells_across_slab": 1.0,  # subcell zero-thickness sheet
            "cells_per_skin_depth": np.inf,
        }
    eps_f = mat.epsilon(freqs)
    mu_f = mat.mu(freqs)
    n_f, _ = passive_refractive_index_and_impedance(eps_f, mu_f)
    n_abs_max = float(np.max(np.abs(n_f)))
    lam_mat_min = lam0_min / max(n_abs_max, 1.0)
    cpw_slab = lam_mat_min / float(dz_slab)
    cells_across = float(d) / float(dz_slab)
    if mat.sigma > 0.0:
        delta_min = float(np.sqrt(2.0 / (2.0 * np.pi * fmax * mu0 * mat.sigma)))
        cells_per_skin = delta_min / float(dz_slab)
    else:
        delta_min = np.inf
        cells_per_skin = np.inf
    return {
        "n_max": n_abs_max,
        "lam_mat_min_m": lam_mat_min,
        "skin_depth_min_m": delta_min,
        "cpw_slab": float(cpw_slab),
        "cells_across_slab": float(cells_across),
        "cells_per_skin_depth": float(cells_per_skin),
    }


def run_slab_rt_fit(
    material_name: str,
    d: float,
    freqs: np.ndarray,
    cpw_target: float = 22.0,
    graded: bool = True,
    use_thin_sheet: bool = False,
    record_fields: bool = False,
) -> dict:
    """Run paired 3D FIT (`SolverFIT3D`) normal-incidence slab reflection/transmission simulations.

    Enforces project resolution rules:
    * ``cpw_slab >= 20`` inside the slab using ``max|n(f)|`` (or ``cpw_target`` in a convergence sweep),
    * ``cells_across_slab >= 4`` across thickness ``d``,
    * ``cells_per_skin_depth >= 4`` for bulk conductive materials (``sigma > 0``),
    * ``solver.add_sheet(..., mode='thin')`` for subcell metal foils thinner than a grid cell.

    Parameters
    ----------
    material_name : str
        Material name in `wakis.dispersive.dispersive_lib`.
    d : float
        Slab thickness [m].
    freqs : ndarray
        Frequency array [Hz].
    cpw_target : float, optional
        Target minimum cells per wavelength inside the slab (default 22.0).
    graded : bool, optional
        If ``True`` (default), use local mesh refinement (`build_graded_z_grid`);
        if ``False``, use a uniform ``z`` grid of spacing ``dz_slab`` everywhere.
    use_thin_sheet : bool, optional
        If ``True``, model the layer as a zero-thickness subcell conductive sheet
        via ``solver.add_sheet`` (`wakis.subcell`).
    record_fields : bool, optional
        If ``True``, record 1D spatial profiles ``Ex(z, t)`` for snapshots and GIFs.

    Returns
    -------
    res : dict
        Simulation results, exact transfer-matrix reference, and grid metrics.
    """
    t0 = time.perf_counter()
    mat = dispersive_lib[material_name]
    freqs = np.asarray(freqs, dtype=float)
    fmin, fmax = float(freqs[0]), float(freqs[-1])
    f0 = 0.5 * (fmin + fmax)
    fwidth = 0.72 * (fmax - fmin)
    pulse = GaussianSource(f0, fwidth=fwidth, cutoff=3.6)

    dz_vac = (c0 / fmax) / max(float(cpw_target) * 1.25, 25.0)
    z_min_dom = -0.045
    z_max_dom = (0.0 if use_thin_sheet else float(d)) + 0.045

    if use_thin_sheet:
        dz_act = min(dz_vac, 1.5e-3)
        nz = int(round((z_max_dom - z_min_dom) / dz_act))
        z_nodes = np.linspace(z_min_dom, z_max_dom, nz + 1)
        iz_zero = int(np.argmin(np.abs(z_nodes - 0.0)))
        z_nodes = z_nodes - z_nodes[iz_zero]
        dz_act = float(z_nodes[1] - z_nodes[0])
    else:
        eps_f = mat.epsilon(freqs)
        mu_f = mat.mu(freqs)
        n_f, _ = passive_refractive_index_and_impedance(eps_f, mu_f)
        n_abs_max = float(np.max(np.abs(n_f)))
        lam_mat_min = (c0 / fmax) / max(n_abs_max, 1.0)
        dz_target = min(lam_mat_min / float(cpw_target), float(d) / max(6.0, 0.35 * float(cpw_target)))
        if mat.sigma > 0.0:
            delta_min = float(np.sqrt(2.0 / (2.0 * np.pi * fmax * mu0 * mat.sigma)))
            dz_target = min(dz_target, delta_min / max(4.5, 0.22 * float(cpw_target)))
        if graded:
            z_nodes, dz_act = build_graded_z_grid(
                0.0, float(d), dz_target, z_min_dom, z_max_dom, dz_vac
            )
        else:
            n_slab = max(4, int(np.ceil(float(d) / dz_target - 1e-9)))
            dz_act = float(d) / n_slab
            n_left = int(np.ceil(abs(z_min_dom) / dz_act))
            n_right = int(np.ceil(abs(z_max_dom - float(d)) / dz_act))
            z_nodes = np.arange(-n_left, n_slab + n_right + 1, dtype=float) * dz_act

    dxy = max(dz_act * 8.0, 2.0e-3)
    x_nodes = np.linspace(-dxy, dxy, 3)  # Nx = 2 so remove_edge_ghost keeps interior nodes
    y_nodes = np.linspace(-dxy, dxy, 3)  # Ny = 2

    z_hi_slab = 0.0 if use_thin_sheet else float(d)
    iz_src = int(np.argmin(np.abs(z_nodes - (-0.015))))
    iz_r = int(np.argmin(np.abs(z_nodes - (-0.006))))
    iz_t = int(np.argmin(np.abs(z_nodes - (z_hi_slab + 0.006))))
    z_src = float(z_nodes[iz_src])
    z_r = float(z_nodes[iz_r])
    z_t = float(z_nodes[iz_t])

    # High-Q dielectric etalon (TiO2 ceramic) needs longer ring-down for multiple internal reflections
    if material_name == "tio2 ceramic":
        t_ring = 4.5e-9
    elif material_name in ("cnt silicone", "ni-graphite silicone", "copper foil"):
        t_ring = 1.2e-9
    else:
        t_ring = 2.1e-9

    # 1. Empty vacuum reference run (only needs pulse passage + 0.45 ns)
    g0 = GridFIT3D(x=x_nodes, y=y_nodes, z=z_nodes, verbose=0)
    s0 = SolverFIT3D(
        g0,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=10,
        cfln=0.80,
        use_stl=False,
        verbose=0,
    )
    nt_ref = int(np.ceil((pulse.end_time + 0.45e-9) / s0.dt))
    nt_slab = int(np.ceil((pulse.end_time + t_ring) / s0.dt))
    est = estimate((g0.Nx, g0.Ny, g0.Nz), steps=nt_slab, cpml=True)
    preflight(est, "laptop")

    src0 = CurrentSource(
        pulse,
        component="x",
        center=(0.0, 0.0, z_src),
        size=(2.0 * dxy, 2.0 * dxy, 0.0),
        amplitude=1.0,
    )
    fr0 = FluxMonitor(freqs, center=(0.0, 0.0, z_r), size=(2.0 * dxy, 2.0 * dxy, 0.0))
    ft0 = FluxMonitor(freqs, center=(0.0, 0.0, z_t), size=(2.0 * dxy, 2.0 * dxy, 0.0))
    s0.emsolve(nt_ref, source=[src0], monitors=[fr0, ft0])
    pr0 = fr0.get_flux()
    pt0 = ft0.get_flux()

    # 2. Slab / thin-sheet run
    g1 = GridFIT3D(x=x_nodes, y=y_nodes, z=z_nodes, verbose=0)
    s1 = SolverFIT3D(
        g1,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=10,
        cfln=0.80,
        use_stl=False,
        verbose=0,
    )
    if use_thin_sheet:
        s1.add_sheet(
            plane="xy",
            position=0.0,
            sigma_s=mat.sigma,
            thickness=float(d),
            mode="thin",
        )
    else:
        s1.add_dispersive_material(
            mat,
            center=(0.0, 0.0, 0.5 * float(d)),
            size=(10.0 * dxy, 10.0 * dxy, float(d)),
        )

    src1 = CurrentSource(
        pulse,
        component="x",
        center=(0.0, 0.0, z_src),
        size=(2.0 * dxy, 2.0 * dxy, 0.0),
        amplitude=1.0,
    )
    fr1 = FluxMonitor(freqs, center=(0.0, 0.0, z_r), size=(2.0 * dxy, 2.0 * dxy, 0.0))
    ft1 = FluxMonitor(freqs, center=(0.0, 0.0, z_t), size=(2.0 * dxy, 2.0 * dxy, 0.0))

    frames = []
    frame_times = []
    if not record_fields:
        s1.emsolve(nt_slab, source=[src1], monitors=[fr1, ft1])
    else:
        t_max_anim = pulse.end_time + 1.0e-9
        nt_anim = min(nt_slab, int(np.ceil(t_max_anim / s1.dt)))
        stride = max(1, nt_anim // 32)
        step_counter = [0]

        def _rec_cb(solver, t_now):
            n = step_counter[0]
            step_counter[0] = n + 1
            if n <= nt_anim and n % stride == 0 and len(frames) < 32:
                ex_prof = np.asarray(solver.E[0, 0, :-1, "x"], dtype=float).copy()
                frames.append(ex_prof)
                frame_times.append(float(t_now))

        s1.emsolve(nt_slab, source=[src1], monitors=[fr1, ft1], callback=_rec_cb)

    pr1 = fr1.get_flux()
    pt1 = ft1.get_flux()
    R_fit = np.clip(1.0 - pr1 / pr0, 0.0, 1.0)
    T_fit = np.clip(pt1 / pt0, 1e-30, 1.0)
    A_fit = 1.0 - R_fit - T_fit
    SE_fit_db = -10.0 * np.log10(T_fit)

    if use_thin_sheet:
        _, _, T_ex, R_ex = sheet_transmission_reflection(
            np.full_like(freqs, mat.sigma * float(d)), f=freqs
        )
        A_ex = 1.0 - R_ex - T_ex
        SE_ex_db = -10.0 * np.log10(np.maximum(T_ex, 1e-300))
        R_disc, T_disc, A_disc = R_ex, T_ex, A_ex
    else:
        tm_cont = exact_slab_rt(freqs, mat.epsilon(freqs), mat.mu(freqs), float(d))
        tm_disc = exact_slab_rt(
            freqs,
            mat.epsilon_discrete(freqs, s1.dt),
            mat.mu_discrete(freqs, s1.dt),
            float(d),
        )
        R_ex, T_ex, A_ex = tm_cont["R"], tm_cont["T"], tm_cont["A"]
        R_disc, T_disc, A_disc = tm_disc["R"], tm_disc["T"], tm_disc["A"]
        SE_ex_db = -10.0 * np.log10(np.maximum(T_ex, 1e-300))

    metrics = compute_slab_resolution_metrics(
        mat, float(d), freqs, dz_act, use_thin_sheet=use_thin_sheet
    )
    rms_R = float(np.sqrt(np.mean((R_fit - R_ex) ** 2)))
    rms_T = float(np.sqrt(np.mean((T_fit - T_ex) ** 2)))
    rms_A = float(np.sqrt(np.mean((A_fit - A_ex) ** 2)))
    rms_R_disc = float(np.sqrt(np.mean((R_fit - R_disc) ** 2)))
    rms_T_disc = float(np.sqrt(np.mean((T_fit - T_disc) ** 2)))

    out = {
        "material": material_name,
        "d_m": float(d),
        "d_mm": float(d) * 1e3,
        "freqs": freqs,
        "dt": float(s1.dt),
        "Nz": int(g1.Nz),
        "n_cells": int(g1.Nx * g1.Ny * g1.Nz),
        "nt_ref": int(nt_ref),
        "nt_slab": int(nt_slab),
        "dz_slab_m": float(dz_act),
        "dz_vac_m": float(dz_vac),
        "z_src_m": z_src,
        "z_r_m": z_r,
        "z_t_m": z_t,
        "R_fit": R_fit,
        "T_fit": T_fit,
        "A_fit": A_fit,
        "SE_fit_db": SE_fit_db,
        "R_ex": R_ex,
        "T_ex": T_ex,
        "A_ex": A_ex,
        "SE_ex_db": SE_ex_db,
        "R_disc": R_disc,
        "T_disc": T_disc,
        "A_disc": A_disc,
        "rms_R": rms_R,
        "rms_T": rms_T,
        "rms_A": rms_A,
        "rms_R_disc": rms_R_disc,
        "rms_T_disc": rms_T_disc,
        "wall_s": float(time.perf_counter() - t0),
        **metrics,
    }
    if record_fields:
        out["z_nodes_m"] = z_nodes[:-1].copy()
        out["frames"] = np.asarray(frames, dtype=float)
        out["frame_times"] = np.asarray(frame_times, dtype=float)
    return out


def run_pec_backed_absorber_fit(
    material_name: str,
    d: float,
    freqs: np.ndarray,
    cpw_target: float = 22.0,
) -> dict:
    """Run 3D FIT (`SolverFIT3D`) simulation of a PEC-backed EMI absorber sheet.

    Places the absorber slab in ``z in [0, d]`` backed by ``bc_high = 'pec'`` at
    ``z = d`` and normalizes the net Poynting flux at ``z_r < 0`` against an
    incident-wave reference run.

    Parameters
    ----------
    material_name : str
        Material name in `wakis.dispersive.dispersive_lib`.
    d : float
        Absorber thickness [m].
    freqs : ndarray
        Frequency array [Hz].
    cpw_target : float, optional
        Minimum cells per internal wavelength at ``max(freqs)``.

    Returns
    -------
    res : dict
        Simulated and exact reflection loss ``RL(f)`` [dB] and reflectance ``R(f)``.
    """
    t0 = time.perf_counter()
    mat = dispersive_lib[material_name]
    freqs = np.asarray(freqs, dtype=float)
    fmin, fmax = float(freqs[0]), float(freqs[-1])
    f0 = 0.5 * (fmin + fmax)
    fwidth = 0.70 * (fmax - fmin)
    pulse = GaussianSource(f0, fwidth=fwidth, cutoff=3.6)

    eps_f = mat.epsilon(freqs)
    mu_f = mat.mu(freqs)
    n_f, _ = passive_refractive_index_and_impedance(eps_f, mu_f)
    n_abs_max = float(np.max(np.abs(n_f)))
    lam_mat_min = (c0 / fmax) / max(n_abs_max, 1.0)
    dz_target = min(lam_mat_min / float(cpw_target), float(d) / 10.0)
    dz_vac = (c0 / fmax) / 28.0

    n_slab = max(8, int(np.ceil(float(d) / dz_target - 1e-9)))
    dz_act = float(d) / n_slab
    z_slab = np.linspace(0.0, float(d), n_slab + 1)
    collar = 4 * dz_act
    z_col = np.linspace(-collar, 0.0, 5)[:-1]

    pts = [-collar]
    dz_g = dz_act
    z_min_dom = -0.045
    while abs(z_min_dom - pts[-1]) > 1.25 * dz_g:
        dz_g = min(dz_g * 1.16, dz_vac)
        pts.append(pts[-1] - dz_g)
    rem = abs(z_min_dom - pts[-1])
    n_rem = max(1, int(np.ceil(rem / dz_vac)))
    tail = np.linspace(pts[-1], z_min_dom, n_rem + 1)[1:]
    z_left = np.sort(np.concatenate([pts, tail]))[:-1]
    z_pec_nodes = np.concatenate([z_left, z_col, z_slab])

    dxy = max(dz_act * 8.0, 2.0e-3)
    xp = np.linspace(-dxy, dxy, 3)
    yp = np.linspace(-dxy, dxy, 3)
    iz_src = int(np.argmin(np.abs(z_pec_nodes - (-0.015))))
    iz_r = int(np.argmin(np.abs(z_pec_nodes - (-0.006))))
    z_src = float(z_pec_nodes[iz_src])
    z_r = float(z_pec_nodes[iz_r])

    # 1. Incident reference run (extend right boundary by 12 cells for CPML)
    z_right_pml = float(d) + np.arange(1, 13) * dz_act
    z_ref_nodes = np.concatenate([z_pec_nodes, z_right_pml])
    g0 = GridFIT3D(x=xp, y=yp, z=z_ref_nodes, verbose=0)
    s0 = SolverFIT3D(
        g0,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=10,
        cfln=0.80,
        use_stl=False,
        verbose=0,
    )
    nt_ref = int(np.ceil((pulse.end_time + 0.45e-9) / s0.dt))
    src0 = CurrentSource(
        pulse,
        component="x",
        center=(0.0, 0.0, z_src),
        size=(2.0 * dxy, 2.0 * dxy, 0.0),
        amplitude=1.0,
    )
    fr0 = FluxMonitor(freqs, center=(0.0, 0.0, z_r), size=(2.0 * dxy, 2.0 * dxy, 0.0))
    s0.emsolve(nt_ref, source=[src0], monitors=[fr0])
    pr0 = fr0.get_flux()

    # 2. PEC-backed absorber run (bc_high = 'pec' at z = d)
    g1 = GridFIT3D(x=xp, y=yp, z=z_pec_nodes, verbose=0)
    s1 = SolverFIT3D(
        g1,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "pec"],
        n_pml=10,
        cfln=0.80,
        use_stl=False,
        verbose=0,
    )
    s1.add_dispersive_material(
        mat,
        center=(0.0, 0.0, 0.5 * float(d)),
        size=(10.0 * dxy, 10.0 * dxy, float(d)),
    )
    nt_slab = int(np.ceil((pulse.end_time + 2.6e-9) / s1.dt))
    src1 = CurrentSource(
        pulse,
        component="x",
        center=(0.0, 0.0, z_src),
        size=(2.0 * dxy, 2.0 * dxy, 0.0),
        amplitude=1.0,
    )
    fr1 = FluxMonitor(freqs, center=(0.0, 0.0, z_r), size=(2.0 * dxy, 2.0 * dxy, 0.0))
    s1.emsolve(nt_slab, source=[src1], monitors=[fr1])
    pr1 = fr1.get_flux()

    R_fit = np.clip(1.0 - pr1 / pr0, 1e-16, 1.0)
    RL_fit_db = -10.0 * np.log10(R_fit)
    ex = exact_pec_backed_absorber(freqs, mat.epsilon(freqs), mat.mu(freqs), float(d))
    ex_disc = exact_pec_backed_absorber(
        freqs,
        mat.epsilon_discrete(freqs, s1.dt),
        mat.mu_discrete(freqs, s1.dt),
        float(d),
    )

    rms_R = float(np.sqrt(np.mean((R_fit - ex["R"]) ** 2)))
    rms_RL_db = float(np.sqrt(np.mean((RL_fit_db - ex["RL_db"]) ** 2)))
    metrics = compute_slab_resolution_metrics(mat, float(d), freqs, dz_act)

    return {
        "material": material_name,
        "d_m": float(d),
        "d_mm": float(d) * 1e3,
        "freqs": freqs,
        "dt": float(s1.dt),
        "Nz": int(g1.Nz),
        "nt_slab": int(nt_slab),
        "dz_slab_m": float(dz_act),
        "z_src_m": z_src,
        "z_r_m": z_r,
        "R_fit": R_fit,
        "RL_fit_db": RL_fit_db,
        "R_ex": ex["R"],
        "RL_ex_db": ex["RL_db"],
        "R_disc": ex_disc["R"],
        "RL_disc_db": ex_disc["RL_db"],
        "rms_R": rms_R,
        "rms_RL_db": rms_RL_db,
        "peak_RL_fit_db": float(np.max(RL_fit_db)),
        "peak_RL_ex_db": float(np.max(ex["RL_db"])),
        "f_peak_fit_ghz": float(freqs[int(np.argmax(RL_fit_db))] * 1e-9),
        "f_peak_ex_ghz": float(freqs[int(np.argmax(ex["RL_db"]))] * 1e-9),
        "wall_s": float(time.perf_counter() - t0),
        **metrics,
    }


def run_task_0(cfg: dict, data_dir: Path) -> dict:
    """Task 0: Eccosorb RF-LB datasheet pole fitting, eps' inference, 1-6 GHz extrapolation, and 6-class spectra."""
    ds = load_eccosorb_rf_lb_csv(data_dir)
    f_mu = ds["f_mu_hz"]
    mu_target = ds["mu_real"] - 1j * ds["mu_imag"]
    f_att = ds["f_att_hz"]
    att_target = ds["atten_db_cm"]

    # 1. Fit candidate passive magnetic pole models over 10 MHz - 1 GHz
    fit_2deb = fit_mu_poles(
        f_mu, mu_target, n_lorentz=0, n_debye=2, mu_inf=1.0, eps_inf=81.06, n_starts=8, name="rf_lb_2debye"
    )
    fit_1d1l = dispersive_lib["eccosorb rf-lb"]
    fit_2d1l = fit_mu_poles(
        f_mu, mu_target, n_lorentz=1, n_debye=2, mu_inf=1.0, eps_inf=81.06, n_starts=8, name="rf_lb_2debye_1lorentz"
    )

    mu_mod_1d1l = fit_1d1l.mu(f_mu)
    mu_mod_2deb = fit_2deb.mu(f_mu)
    mu_mod_2d1l = fit_2d1l.mu(f_mu)

    rel_rms_1d1l = float(np.sqrt(np.mean(np.abs(mu_mod_1d1l - mu_target) ** 2 / np.abs(mu_target) ** 2)))
    rel_rms_2deb = float(np.sqrt(np.mean(np.abs(mu_mod_2deb - mu_target) ** 2 / np.abs(mu_target) ** 2)))
    rel_rms_2d1l = float(np.sqrt(np.mean(np.abs(mu_mod_2d1l - mu_target) ** 2 / np.abs(mu_target) ** 2)))
    max_abs_dmu_real = float(np.max(np.abs(np.real(mu_mod_1d1l) - ds["mu_real"])))
    max_abs_dmu_imag = float(np.max(np.abs(-np.imag(mu_mod_1d1l) - ds["mu_imag"])))

    # 2. Infer eps' over the 0.1 - 1.0 GHz overlap band and evaluate sensitivity to tan_delta_e
    mask_ov = f_att <= 1.001e9
    f_ov = f_att[mask_ov]
    att_ov = att_target[mask_ov]
    mu_ov = fit_1d1l.mu(f_ov)

    tan_d_grid = np.array([0.00, 0.01, 0.02, 0.05, 0.10], dtype=float)
    eps_inferred_grid = np.zeros_like(tan_d_grid)
    rms_overlap_grid = np.zeros_like(tan_d_grid)
    for idx_td, td in enumerate(tan_d_grid):
        inf_res = infer_eps_real_from_attenuation(f_ov, mu_ov, att_ov, tan_delta_e=float(td))
        eps_inferred_grid[idx_td] = inf_res["eps_real"]
        rms_overlap_grid[idx_td] = inf_res["rms_overlap_db_cm"]

    # Dense sweep of eps' in [40, 130] for the sensitivity curve
    eps_sweep = np.linspace(40.0, 130.0, 181)
    rms_vs_eps_td0 = np.array(
        [
            float(np.sqrt(np.mean((attenuation_db_per_cm(f_ov, er, mu_ov) - att_ov) ** 2)))
            for er in eps_sweep
        ]
    )
    rms_vs_eps_td001 = np.array(
        [
            float(np.sqrt(np.mean((attenuation_db_per_cm(f_ov, er * (1.0 - 0.01j), mu_ov) - att_ov) ** 2)))
            for er in eps_sweep
        ]
    )
    rms_vs_eps_td005 = np.array(
        [
            float(np.sqrt(np.mean((attenuation_db_per_cm(f_ov, er * (1.0 - 0.05j), mu_ov) - att_ov) ** 2)))
            for er in eps_sweep
        ]
    )

    # 3. Extrapolate mu(f) from 1 GHz to 6 GHz and compare predicted attenuation vs datasheet
    f_wide = np.geomspace(1.0e7, 6.0e9, 240)
    mu_wide_1d1l = fit_1d1l.mu(f_wide)
    mu_wide_2deb = fit_2deb.mu(f_wide)
    mu_wide_2d1l = fit_2d1l.mu(f_wide)
    eps_wide_1d1l = fit_1d1l.epsilon(f_wide)

    att_wide_1d1l = attenuation_db_per_cm(f_wide, eps_wide_1d1l, mu_wide_1d1l)
    att_wide_2deb = attenuation_db_per_cm(f_wide, eps_wide_1d1l, mu_wide_2deb)
    att_wide_2d1l = attenuation_db_per_cm(f_wide, eps_wide_1d1l, mu_wide_2d1l)

    att_at_ds_1d1l = attenuation_db_per_cm(f_att, fit_1d1l.epsilon(f_att), fit_1d1l.mu(f_att))
    att_at_ds_2deb = attenuation_db_per_cm(f_att, fit_1d1l.epsilon(f_att), fit_2deb.mu(f_att))
    mask_extrap = f_att > 1.001e9
    rms_att_full_1d1l = float(np.sqrt(np.mean((att_at_ds_1d1l - att_target) ** 2)))
    rms_att_extrap_1d1l = float(np.sqrt(np.mean((att_at_ds_1d1l[mask_extrap] - att_target[mask_extrap]) ** 2)))
    rms_att_extrap_2deb = float(np.sqrt(np.mean((att_at_ds_2deb[mask_extrap] - att_target[mask_extrap]) ** 2)))

    # 4. Evaluate constitutive spectra across all 6 EMI material classes (10 MHz - 6 GHz)
    class_keys = [c["key"] for c in EMI_CLASSES]
    eps_6 = np.zeros((len(class_keys), f_wide.size), dtype=np.complex128)
    mu_6 = np.zeros((len(class_keys), f_wide.size), dtype=np.complex128)
    zrel_6 = np.zeros((len(class_keys), f_wide.size), dtype=np.complex128)
    atten_6 = np.zeros((len(class_keys), f_wide.size), dtype=float)
    for idx_c, key in enumerate(class_keys):
        m = dispersive_lib[key]
        ef = m.epsilon(f_wide)
        mf = m.mu(f_wide)
        _, zr = passive_refractive_index_and_impedance(ef, mf)
        eps_6[idx_c] = ef
        mu_6[idx_c] = mf
        zrel_6[idx_c] = zr
        atten_6[idx_c] = attenuation_db_per_cm(f_wide, ef, mf)

    return {
        "name": cfg["name"],
        "cpw": 24,
        "cells_across_feature": 10.0,
        "dx_min_mm": 0.10,
        "min_feature_mm": 1.00,
        "n_cells": 240,
        "n_steps": 240,
        "f_mu_hz": f_mu,
        "mu_real_ds": ds["mu_real"],
        "mu_imag_ds": ds["mu_imag"],
        "f_att_hz": f_att,
        "atten_ds_db_cm": att_target,
        "f_wide_hz": f_wide,
        "mu_wide_1d1l": mu_wide_1d1l,
        "mu_wide_2deb": mu_wide_2deb,
        "mu_wide_2d1l": mu_wide_2d1l,
        "eps_wide_1d1l": eps_wide_1d1l,
        "att_wide_1d1l": att_wide_1d1l,
        "att_wide_2deb": att_wide_2deb,
        "att_wide_2d1l": att_wide_2d1l,
        "att_at_ds_1d1l": att_at_ds_1d1l,
        "att_at_ds_2deb": att_at_ds_2deb,
        "rel_rms_1d1l": rel_rms_1d1l,
        "rel_rms_2deb": rel_rms_2deb,
        "rel_rms_2d1l": rel_rms_2d1l,
        "max_abs_dmu_real": max_abs_dmu_real,
        "max_abs_dmu_imag": max_abs_dmu_imag,
        "tan_d_grid": tan_d_grid,
        "eps_inferred_grid": eps_inferred_grid,
        "rms_overlap_grid": rms_overlap_grid,
        "eps_sweep": eps_sweep,
        "rms_vs_eps_td0": rms_vs_eps_td0,
        "rms_vs_eps_td001": rms_vs_eps_td001,
        "rms_vs_eps_td005": rms_vs_eps_td005,
        "rms_att_full_1d1l": rms_att_full_1d1l,
        "rms_att_extrap_1d1l": rms_att_extrap_1d1l,
        "rms_att_extrap_2deb": rms_att_extrap_2deb,
        "eps_6": eps_6,
        "mu_6": mu_6,
        "zrel_6": zrel_6,
        "atten_6": atten_6,
    }


def run_task_1(cfg: dict, data_dir: Path) -> dict:
    """Task 1: 3D FIT normal-incidence slab runs across all 6 EMI classes + Schelkunoff decomposition + snapshots."""
    quick = bool(cfg.get("quick", False))
    cpw_target = float(cfg.get("cpw", 20 if quick else 24))
    freqs = np.linspace(0.30e9, 2.50e9, 23 if quick else 29)

    n_classes = len(EMI_CLASSES)
    n_thick = 2 if quick else 3
    n_freq = freqs.size

    d_mm_table = np.zeros((n_classes, n_thick), dtype=float)
    R_fit_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)
    T_fit_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)
    A_fit_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)
    SE_fit_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)

    R_ex_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)
    T_ex_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)
    A_ex_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)
    SE_ex_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)
    SE_R_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)
    SE_A_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)
    SE_M_table = np.zeros((n_classes, n_thick, n_freq), dtype=float)

    rms_R_table = np.zeros((n_classes, n_thick), dtype=float)
    rms_T_table = np.zeros((n_classes, n_thick), dtype=float)
    rms_A_table = np.zeros((n_classes, n_thick), dtype=float)
    cpw_table = np.zeros((n_classes, n_thick), dtype=float)
    cells_across_table = np.zeros((n_classes, n_thick), dtype=float)
    cells_skin_table = np.zeros((n_classes, n_thick), dtype=float)

    total_cells = 0
    total_steps = 0
    min_cpw = np.inf
    min_cells_across = np.inf
    min_dx_mm = np.inf

    # Record field snapshots & GIF frames for 3 contrasting classes at d = 2.0 mm (or 1.0 mm)
    snap_runs = {}
    snap_targets = {
        "eccosorb rf-lb": 2.0e-3,
        "tio2 ceramic": 2.0e-3,
        "cnt silicone": 1.0e-3,
    }

    for ic, cinfo in enumerate(EMI_CLASSES):
        key = cinfo["key"]
        mat = dispersive_lib[key]
        use_sheet = bool(cinfo["use_thin_sheet"])
        thicks_mm = cinfo["quick_thicknesses_mm"] if quick else cinfo["thicknesses_mm"]
        for it, d_mm in enumerate(thicks_mm):
            d_m = float(d_mm) * 1e-3
            rec = (key in snap_targets) and (abs(d_m - snap_targets[key]) < 1e-6)
            res = run_slab_rt_fit(
                key,
                d_m,
                freqs,
                cpw_target=cpw_target,
                graded=True,
                use_thin_sheet=use_sheet,
                record_fields=rec,
            )
            if rec:
                snap_runs[key] = res

            d_mm_table[ic, it] = float(d_mm)
            R_fit_table[ic, it] = res["R_fit"]
            T_fit_table[ic, it] = res["T_fit"]
            A_fit_table[ic, it] = res["A_fit"]
            SE_fit_table[ic, it] = res["SE_fit_db"]
            R_ex_table[ic, it] = res["R_ex"]
            T_ex_table[ic, it] = res["T_ex"]
            A_ex_table[ic, it] = res["A_ex"]
            SE_ex_table[ic, it] = res["SE_ex_db"]

            sch = schelkunoff_decomposition(freqs, mat.epsilon(freqs), mat.mu(freqs), d_m)
            SE_R_table[ic, it] = sch["SE_R_db"]
            SE_A_table[ic, it] = sch["SE_A_db"]
            SE_M_table[ic, it] = sch["SE_M_db"]

            rms_R_table[ic, it] = res["rms_R"]
            rms_T_table[ic, it] = res["rms_T"]
            rms_A_table[ic, it] = res["rms_A"]
            cpw_table[ic, it] = res["cpw_slab"]
            cells_across_table[ic, it] = res["cells_across_slab"]
            cells_skin_table[ic, it] = res["cells_per_skin_depth"]

            total_cells += res["n_cells"]
            total_steps += res["nt_ref"] + res["nt_slab"]
            min_cpw = min(min_cpw, res["cpw_slab"])
            if not use_sheet:
                min_cells_across = min(min_cells_across, res["cells_across_slab"])
            min_dx_mm = min(min_dx_mm, res["dz_slab_m"] * 1e3)

    # Analytical Schelkunoff thickness sweep at f = 2.0 GHz across all 6 classes
    f_ref_se = 2.0e9
    d_sweep_mm = np.geomspace(0.05, 5.0, 80)
    SE_vs_d_total = np.zeros((n_classes, d_sweep_mm.size), dtype=float)
    SE_vs_d_R = np.zeros((n_classes, d_sweep_mm.size), dtype=float)
    SE_vs_d_A = np.zeros((n_classes, d_sweep_mm.size), dtype=float)
    SE_vs_d_M = np.zeros((n_classes, d_sweep_mm.size), dtype=float)
    for ic, cinfo in enumerate(EMI_CLASSES):
        mat = dispersive_lib[cinfo["key"]]
        e_f0 = mat.epsilon(np.array([f_ref_se]))[0]
        m_f0 = mat.mu(np.array([f_ref_se]))[0]
        for id_m, d_mm_val in enumerate(d_sweep_mm):
            # For copper foil, scale thickness in micrometers (0.05 um to 5.0 um)
            d_phys = (d_mm_val * 1e-6) if cinfo["use_thin_sheet"] else (d_mm_val * 1e-3)
            sch_d = schelkunoff_decomposition(np.array([f_ref_se]), e_f0, m_f0, d_phys)
            SE_vs_d_total[ic, id_m] = float(sch_d["SE_total_db"][0])
            SE_vs_d_R[ic, id_m] = float(sch_d["SE_R_db"][0])
            SE_vs_d_A[ic, id_m] = float(sch_d["SE_A_db"][0])
            SE_vs_d_M[ic, id_m] = float(sch_d["SE_M_db"][0])

    return {
        "name": cfg["name"],
        "cpw": int(round(min_cpw)),
        "cells_across_feature": float(min_cells_across),
        "dx_min_mm": float(min_dx_mm),
        "min_feature_mm": 0.50,
        "n_cells": int(total_cells),
        "n_steps": int(total_steps),
        "freqs": freqs,
        "d_mm_table": d_mm_table,
        "R_fit_table": R_fit_table,
        "T_fit_table": T_fit_table,
        "A_fit_table": A_fit_table,
        "SE_fit_table": SE_fit_table,
        "R_ex_table": R_ex_table,
        "T_ex_table": T_ex_table,
        "A_ex_table": A_ex_table,
        "SE_ex_table": SE_ex_table,
        "SE_R_table": SE_R_table,
        "SE_A_table": SE_A_table,
        "SE_M_table": SE_M_table,
        "rms_R_table": rms_R_table,
        "rms_T_table": rms_T_table,
        "rms_A_table": rms_A_table,
        "cpw_table": cpw_table,
        "cells_across_table": cells_across_table,
        "cells_skin_table": cells_skin_table,
        "max_rms_R": float(np.max(rms_R_table)),
        "max_rms_T": float(np.max(rms_T_table)),
        "d_sweep_mm": d_sweep_mm,
        "SE_vs_d_total": SE_vs_d_total,
        "SE_vs_d_R": SE_vs_d_R,
        "SE_vs_d_A": SE_vs_d_A,
        "SE_vs_d_M": SE_vs_d_M,
        "snap_rf_z_m": snap_runs["eccosorb rf-lb"]["z_nodes_m"],
        "snap_rf_frames": snap_runs["eccosorb rf-lb"]["frames"],
        "snap_rf_times": snap_runs["eccosorb rf-lb"]["frame_times"],
        "snap_tio2_z_m": snap_runs["tio2 ceramic"]["z_nodes_m"],
        "snap_tio2_frames": snap_runs["tio2 ceramic"]["frames"],
        "snap_tio2_times": snap_runs["tio2 ceramic"]["frame_times"],
        "snap_cnt_z_m": snap_runs["cnt silicone"]["z_nodes_m"],
        "snap_cnt_frames": snap_runs["cnt silicone"]["frames"],
        "snap_cnt_times": snap_runs["cnt silicone"]["frame_times"],
    }


def run_task_2(cfg: dict, data_dir: Path) -> dict:
    """Task 2: Metal-backed absorber (bc_high='pec') for Eccosorb RF-LB & Sendust flake sheet + Rozanov bound."""
    quick = bool(cfg.get("quick", False))
    cpw_target = float(cfg.get("cpw", 20 if quick else 24))
    freqs = np.linspace(0.30e9, 4.50e9, 25 if quick else 37)

    rf_thicks_mm = [1.0, 2.0, 3.0]
    sd_thicks_mm = [0.5, 1.0, 1.5]

    RL_rf_fit = np.zeros((3, freqs.size), dtype=float)
    RL_rf_ex = np.zeros((3, freqs.size), dtype=float)
    rms_RL_rf = np.zeros(3, dtype=float)

    RL_sd_fit = np.zeros((3, freqs.size), dtype=float)
    RL_sd_ex = np.zeros((3, freqs.size), dtype=float)
    rms_RL_sd = np.zeros(3, dtype=float)

    total_cells = 0
    total_steps = 0
    min_cpw = np.inf
    min_cells_across = np.inf
    min_dx_mm = np.inf

    for idx_d, d_mm in enumerate(rf_thicks_mm):
        res_rf = run_pec_backed_absorber_fit("eccosorb rf-lb", d_mm * 1e-3, freqs, cpw_target=cpw_target)
        RL_rf_fit[idx_d] = res_rf["RL_fit_db"]
        RL_rf_ex[idx_d] = res_rf["RL_ex_db"]
        rms_RL_rf[idx_d] = res_rf["rms_RL_db"]
        total_cells += res_rf["Nz"] * 4
        total_steps += res_rf["nt_slab"] * 2
        min_cpw = min(min_cpw, res_rf["cpw_slab"])
        min_cells_across = min(min_cells_across, res_rf["cells_across_slab"])
        min_dx_mm = min(min_dx_mm, res_rf["dz_slab_m"] * 1e3)

    for idx_d, d_mm in enumerate(sd_thicks_mm):
        res_sd = run_pec_backed_absorber_fit("sendust flake sheet", d_mm * 1e-3, freqs, cpw_target=cpw_target)
        RL_sd_fit[idx_d] = res_sd["RL_fit_db"]
        RL_sd_ex[idx_d] = res_sd["RL_ex_db"]
        rms_RL_sd[idx_d] = res_sd["rms_RL_db"]
        total_cells += res_sd["Nz"] * 4
        total_steps += res_sd["nt_slab"] * 2
        min_cpw = min(min_cpw, res_sd["cpw_slab"])
        min_cells_across = min(min_cells_across, res_sd["cells_across_slab"])
        min_dx_mm = min(min_dx_mm, res_sd["dz_slab_m"] * 1e3)

    # 2D map RL(f, d) for Eccosorb RF-LB + quarter-wave trajectory d_{lambda/4}(f)
    f_map = np.linspace(0.20e9, 6.00e9, 120)
    d_map_mm = np.linspace(0.30, 3.50, 90)
    mat_rf = dispersive_lib["eccosorb rf-lb"]
    eps_map = mat_rf.epsilon(f_map)
    mu_map = mat_rf.mu(f_map)
    n_map, _ = passive_refractive_index_and_impedance(eps_map, mu_map)
    d_qw_mm = (c0 / (4.0 * f_map * np.real(n_map))) * 1e3

    RL_map_rf = np.zeros((d_map_mm.size, f_map.size), dtype=float)
    for id_m, d_mm_val in enumerate(d_map_mm):
        RL_map_rf[id_m] = exact_pec_backed_absorber(f_map, eps_map, mu_map, d_mm_val * 1e-3)["RL_db"]

    # Rozanov bound evaluation over broadband frequency grid (10 MHz - 30 GHz)
    f_roz = np.geomspace(1.0e7, 30.0e9, 500)
    d_roz_mm = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
    mat_sd = dispersive_lib["sendust flake sheet"]
    mu_s_rf = float(np.real(mat_rf.mu(np.array([1.0e5]))[0]))
    mu_s_sd = float(np.real(mat_sd.mu(np.array([1.0e5]))[0]))

    roz_int_rf = np.zeros_like(d_roz_mm)
    roz_bound_rf = np.zeros_like(d_roz_mm)
    roz_int_sd = np.zeros_like(d_roz_mm)
    roz_bound_sd = np.zeros_like(d_roz_mm)
    roz_bound_nonmag = np.zeros_like(d_roz_mm)

    eps_roz_rf, mu_roz_rf = mat_rf.epsilon(f_roz), mat_rf.mu(f_roz)
    eps_roz_sd, mu_roz_sd = mat_sd.epsilon(f_roz), mat_sd.mu(f_roz)
    for idx_r, d_mm_val in enumerate(d_roz_mm):
        d_m = float(d_mm_val) * 1e-3
        g_rf = exact_pec_backed_absorber(f_roz, eps_roz_rf, mu_roz_rf, d_m)["gamma"]
        g_sd = exact_pec_backed_absorber(f_roz, eps_roz_sd, mu_roz_sd, d_m)["gamma"]
        ev_rf = evaluate_rozanov_bound(f_roz, g_rf, mu_s_rf, d_m)
        ev_sd = evaluate_rozanov_bound(f_roz, g_sd, mu_s_sd, d_m)
        roz_int_rf[idx_r] = ev_rf["integral_m"] * 1e3
        roz_bound_rf[idx_r] = ev_rf["rozanov_bound_m"] * 1e3
        roz_int_sd[idx_r] = ev_sd["integral_m"] * 1e3
        roz_bound_sd[idx_r] = ev_sd["rozanov_bound_m"] * 1e3
        roz_bound_nonmag[idx_r] = ev_rf["nonmag_bound_m"] * 1e3

    return {
        "name": cfg["name"],
        "cpw": int(round(min_cpw)),
        "cells_across_feature": float(min_cells_across),
        "dx_min_mm": float(min_dx_mm),
        "min_feature_mm": 0.50,
        "n_cells": int(total_cells),
        "n_steps": int(total_steps),
        "freqs": freqs,
        "rf_thicks_mm": np.asarray(rf_thicks_mm, dtype=float),
        "sd_thicks_mm": np.asarray(sd_thicks_mm, dtype=float),
        "RL_rf_fit": RL_rf_fit,
        "RL_rf_ex": RL_rf_ex,
        "rms_RL_rf": rms_RL_rf,
        "RL_sd_fit": RL_sd_fit,
        "RL_sd_ex": RL_sd_ex,
        "rms_RL_sd": rms_RL_sd,
        "max_rms_RL_db": float(max(np.max(rms_RL_rf), np.max(rms_RL_sd))),
        "f_map": f_map,
        "d_map_mm": d_map_mm,
        "d_qw_mm": d_qw_mm,
        "RL_map_rf": RL_map_rf,
        "d_roz_mm": d_roz_mm,
        "mu_s_rf": mu_s_rf,
        "mu_s_sd": mu_s_sd,
        "roz_int_rf_mm": roz_int_rf,
        "roz_bound_rf_mm": roz_bound_rf,
        "roz_int_sd_mm": roz_int_sd,
        "roz_bound_sd_mm": roz_bound_sd,
        "roz_bound_nonmag_mm": roz_bound_nonmag,
    }


def run_task_3(cfg: dict, data_dir: Path) -> dict:
    """Task 3: Resolution convergence study (cpw sweep) and locally refined z-mesh vs uniform baseline."""
    quick = bool(cfg.get("quick", False))
    cpw_levels = np.array([10, 14, 20, 28] if quick else [10, 14, 20, 28, 40], dtype=float)
    freqs = np.linspace(0.30e9, 2.50e9, 21)

    study_classes = [
        ("eccosorb rf-lb", 2.0e-3),
        ("tio2 ceramic", 2.0e-3),
        ("cnt silicone", 1.0e-3),
        ("ni-graphite silicone", 1.0e-3),
    ]
    rms_R_conv = np.zeros((len(study_classes), cpw_levels.size), dtype=float)
    rms_T_conv = np.zeros((len(study_classes), cpw_levels.size), dtype=float)
    dz_slab_conv_mm = np.zeros((len(study_classes), cpw_levels.size), dtype=float)
    nz_graded_conv = np.zeros((len(study_classes), cpw_levels.size), dtype=int)
    wall_graded_conv = np.zeros((len(study_classes), cpw_levels.size), dtype=float)

    total_cells = 0
    total_steps = 0

    for ic, (mname, d_m) in enumerate(study_classes):
        for il, cpw_val in enumerate(cpw_levels):
            r = run_slab_rt_fit(mname, d_m, freqs, cpw_target=float(cpw_val), graded=True)
            rms_R_conv[ic, il] = r["rms_R"]
            rms_T_conv[ic, il] = r["rms_T"]
            dz_slab_conv_mm[ic, il] = r["dz_slab_m"] * 1e3
            nz_graded_conv[ic, il] = r["Nz"]
            wall_graded_conv[ic, il] = r["wall_s"]
            total_cells += r["n_cells"]
            total_steps += r["nt_ref"] + r["nt_slab"]

    # Baseline comparison on Eccosorb RF-LB (d = 2.0 mm):
    # Compare locally refined z-mesh vs uniform z-mesh at the same slab resolutions
    cpw_base = np.array([10, 14, 20] if quick else [10, 14, 20, 28], dtype=float)
    rms_R_uniform = np.zeros_like(cpw_base)
    nz_uniform = np.zeros(cpw_base.size, dtype=int)
    wall_uniform = np.zeros_like(cpw_base)
    for il, cpw_val in enumerate(cpw_base):
        ru = run_slab_rt_fit("eccosorb rf-lb", 2.0e-3, freqs, cpw_target=float(cpw_val), graded=False)
        rms_R_uniform[il] = ru["rms_R"]
        nz_uniform[il] = ru["Nz"]
        wall_uniform[il] = ru["wall_s"]
        total_cells += ru["n_cells"]
        total_steps += ru["nt_ref"] + ru["nt_slab"]

    return {
        "name": cfg["name"],
        "cpw": int(cpw_levels[0]),
        "cells_across_feature": 6.0,
        "dx_min_mm": float(np.min(dz_slab_conv_mm)),
        "min_feature_mm": 1.00,
        "n_cells": int(total_cells),
        "n_steps": int(total_steps),
        "cpw_levels": cpw_levels,
        "rms_R_conv": rms_R_conv,
        "rms_T_conv": rms_T_conv,
        "dz_slab_conv_mm": dz_slab_conv_mm,
        "nz_graded_conv": nz_graded_conv,
        "wall_graded_conv": wall_graded_conv,
        "cpw_base": cpw_base,
        "rms_R_uniform": rms_R_uniform,
        "nz_uniform": nz_uniform,
        "wall_uniform": wall_uniform,
    }


def _build_schematic_objects(pec_backed: bool = False):
    """Build representative live Wakis solver, source, monitor, and structure objects for setup schematics."""
    d = 2.0e-3
    mat = dispersive_lib["eccosorb rf-lb"]
    freqs = np.linspace(0.30e9, 2.50e9, 25)
    f0 = 1.40e9
    fwidth = 1.00e9
    pulse = GaussianSource(f0, fwidth=fwidth, cutoff=3.6)

    dz_slab = 0.125e-3
    dz_vac = 1.8e-3
    dxy = 2.0e-3
    x_nodes = np.linspace(-dxy, dxy, 3)
    y_nodes = np.linspace(-dxy, dxy, 3)

    if not pec_backed:
        z_nodes, dz_act = build_graded_z_grid(0.0, d, dz_slab, -0.045, d + 0.045, dz_vac)
        grid = GridFIT3D(x=x_nodes, y=y_nodes, z=z_nodes, verbose=0)
        solver = SolverFIT3D(
            grid,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=10,
            cfln=0.80,
            use_stl=False,
            verbose=0,
        )
        solver.add_dispersive_material(mat, center=(0.0, 0.0, 0.5 * d), size=(10.0 * dxy, 10.0 * dxy, d))
        z_src = float(z_nodes[int(np.argmin(np.abs(z_nodes - (-0.015))))])
        z_r = float(z_nodes[int(np.argmin(np.abs(z_nodes - (-0.006))))])
        z_t = float(z_nodes[int(np.argmin(np.abs(z_nodes - (d + 0.006))))])
        src = CurrentSource(pulse, component="x", center=(0.0, 0.0, z_src), size=(2.0 * dxy, 2.0 * dxy, 0.0))
        mon_r = FluxMonitor(freqs, center=(0.0, 0.0, z_r), size=(2.0 * dxy, 2.0 * dxy, 0.0))
        mon_t = FluxMonitor(freqs, center=(0.0, 0.0, z_t), size=(2.0 * dxy, 2.0 * dxy, 0.0))
        slab_box = SlabBox(0.0, d, dxy, mat, "EMI Slab (d = 2.0 mm)")
        return solver, [src], [mon_r, mon_t], [slab_box], pulse, z_nodes, dz_act

    z_nodes_full, dz_act = build_graded_z_grid(0.0, d, dz_slab, -0.045, d + 0.045, dz_vac)
    z_pec_nodes = z_nodes_full[z_nodes_full <= d + 1e-12]
    grid = GridFIT3D(x=x_nodes, y=y_nodes, z=z_pec_nodes, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "pec"],
        n_pml=10,
        cfln=0.80,
        use_stl=False,
        verbose=0,
    )
    solver.add_dispersive_material(mat, center=(0.0, 0.0, 0.5 * d), size=(10.0 * dxy, 10.0 * dxy, d))
    z_src = float(z_pec_nodes[int(np.argmin(np.abs(z_pec_nodes - (-0.015))))])
    z_r = float(z_pec_nodes[int(np.argmin(np.abs(z_pec_nodes - (-0.006))))])
    src = CurrentSource(pulse, component="x", center=(0.0, 0.0, z_src), size=(2.0 * dxy, 2.0 * dxy, 0.0))
    mon_r = FluxMonitor(freqs, center=(0.0, 0.0, z_r), size=(2.0 * dxy, 2.0 * dxy, 0.0))
    slab_box = SlabBox(0.0, d, dxy, mat, "RF-LB Absorber (d = 2.0 mm)")
    return solver, [src], [mon_r], [slab_box], pulse, z_pec_nodes, dz_act


def _hide_1d_transverse_ticks(ax: plt.Axes) -> None:
    """Hide meaningless transverse axis ticks on quasi-1D `zx` schematics (Rule 6)."""
    ax.set_yticks([])
    ax.set_ylabel("Transverse x (periodic)")


def _save_setup_figure_pair(
    fig_dir: Path,
    prefix_a: str,
    prefix_b: str,
    title_a: str,
    title_b: str,
    pec_backed: bool = False,
) -> list[Path]:
    """Render the two uncluttered setup figures (00a geometry/BCs and 00b mesh + pulse spectrum)."""
    solver, sources, monitors, structures, pulse, z_nodes, dz_act = _build_schematic_objects(
        pec_backed=pec_backed
    )
    written: list[Path] = []

    with figstyle.apply_style():
        # 00a: Uncluttered geometry, materials, sources, monitors, and boundary conditions
        fig_a, ax_a = schematic.plot_schematic(
            solver,
            plane="zx",
            unit="mm",
            structures=structures,
            sources=sources,
            monitors=monitors,
            show_mesh=False,
            show_pulse_inset=False,
            show_wave_packet=False,
            title=title_a,
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.4,
        )
        ax_main = ax_a[0] if isinstance(ax_a, (list, tuple, np.ndarray)) else ax_a
        _hide_1d_transverse_ticks(ax_main)
        pa = fig_dir / f"{prefix_a}.png"
        figstyle.save(fig_a, pa, close=True)
        written.append(pa)

        # 00b: Mesh spacing + local refinement zoom + pulse waveform & spectrum (2 panels)
        fig_b, (ax_m, ax_p) = plt.subplots(
            1, 2, figsize=(figstyle.FULL_WIDTH_IN, 3.3), dpi=figstyle.DEFAULT_DPI
        )
        z_mid_mm = 0.5 * (z_nodes[:-1] + z_nodes[1:]) * 1e3
        dz_mm = np.diff(z_nodes) * 1e3
        st_m = figstyle.series_style(0, with_marker=True)
        ax_m.plot(z_mid_mm, dz_mm, markevery=max(1, len(z_mid_mm) // 18), label=r"Graded $\Delta z(z)$", **st_m)
        ax_m.axvspan(0.0, 2.0, color="#DCE8F2", alpha=0.65, label="Slab [0, 2.0 mm]")
        if pec_backed:
            ax_m.axvline(2.0, color="#222222", lw=2.2, ls="-", label="PEC ground (z = 2.0 mm)")
        ax_m.axhline(
            dz_act * 1e3,
            color="#D55E00",
            ls="--",
            lw=1.2,
            label=f"Slab dz = {dz_act * 1e3:.3f} mm (>= 24 cpw)",
        )
        ax_m.set(
            xlabel="Longitudinal position z [mm]",
            ylabel="Cell size dz [mm]",
            title="(a) Locally refined z-mesh & wave packet",
        )
        figstyle.spectrum_grid(ax_m)
        ax_m.legend(loc="upper right", frameon=False, fontsize=8.0)

        # Pulse spectrum |J_x(f)|^2 and time waveform inset
        t_arr = np.linspace(0.0, pulse.end_time * 1.15, 600)
        sig_t = np.array([float(np.real(pulse(t))) for t in t_arr])
        f_spec = np.linspace(0.05e9, 4.50e9, 300)
        dt_s = float(t_arr[1] - t_arr[0])
        spec_f = np.abs(np.sum(sig_t[:, None] * np.exp(-2j * np.pi * t_arr[:, None] * f_spec[None, :]), axis=0) * dt_s)
        spec_norm = (spec_f / max(float(np.max(spec_f)), 1e-30)) ** 2

        st_p = figstyle.series_style(1, with_marker=False)
        ax_p.plot(f_spec * 1e-9, spec_norm, label=r"Normalized power $|J_x(f)|^2$", **st_p)
        ax_p.axvspan(0.30, 2.50, color="#E6ECC8", alpha=0.50, label="Evaluated R/T band (0.3–2.5 GHz)")
        ax_p.set(
            xlabel="Frequency f [GHz]",
            ylabel="Normalized source spectrum |Jx(f)|^2",
            ylim=(0.0, 1.12),
            title=f"(b) {title_b}",
        )
        figstyle.spectrum_grid(ax_p)
        ax_p.legend(loc="upper right", frameon=False, fontsize=8.0)

        figstyle.finish(fig_b, auto_legend_below=False)
        pb = fig_dir / f"{prefix_b}.png"
        figstyle.save(fig_b, pb, close=True)
        written.append(pb)

    return written


def _save_wave_propagation_gif(t1: dict, out_path: Path, fps: int = 10) -> None:
    """Render an animated GIF (< 5 MB) comparing pulse propagation across 3 EMI slab classes."""
    z_rf = t1["snap_rf_z_m"] * 1e3
    fr_rf = t1["snap_rf_frames"]
    tm_rf = t1["snap_rf_times"] * 1e9

    z_ti = t1["snap_tio2_z_m"] * 1e3
    fr_ti = t1["snap_tio2_frames"]

    z_cn = t1["snap_cnt_z_m"] * 1e3
    fr_cn = t1["snap_cnt_frames"]

    n_frames = min(len(fr_rf), len(fr_ti), len(fr_cn))
    if n_frames == 0:
        return

    solver_rf, src_rf, mon_rf, st_rf, _, _, _ = _build_schematic_objects(pec_backed=False)
    vmax = max(
        float(np.max(np.abs(fr_rf))),
        float(np.max(np.abs(fr_ti))),
        float(np.max(np.abs(fr_cn))),
        1e-12,
    ) * 1.10

    pil_frames = []
    with figstyle.apply_style():
        for idx in range(n_frames):
            fig, axes = plt.subplots(1, 3, figsize=(figstyle.FULL_WIDTH_IN, 2.9), dpi=115, sharey=True)
            specs = [
                (axes[0], z_rf, fr_rf[idx], 2.0, "Eccosorb RF-LB (2.0 mm)", 0),
                (axes[1], z_ti, fr_ti[idx], 2.0, "TiO2 ceramic eps_r=100 (2.0 mm)", 1),
                (axes[2], z_cn, fr_cn[idx], 1.0, "CNT silicone 3 wt% (1.0 mm)", 2),
            ]
            for ax, z_mm, y_val, d_mm, label, s_idx in specs:
                st = figstyle.series_style(s_idx, with_marker=False)
                ax.plot(z_mm, y_val, label=r"$E_x(z, t)$", **st)
                slab_s = SlabBox(0.0, d_mm * 1e-3, 2.0e-3, dispersive_lib["eccosorb rf-lb"], label)
                schematic.overlay_geometry(
                    ax,
                    solver_rf,
                    plane="zx",
                    unit="mm",
                    orientation="1d",
                    structures=[slab_s],
                    sources=src_rf,
                    monitors=mon_rf,
                    show_boundaries=True,
                    show_materials=True,
                    show_sources=True,
                    show_monitors=True,
                )
                ax.set(
                    xlim=(-25.0, 25.0),
                    ylim=(-vmax, vmax),
                    xlabel="Position z [mm]",
                    title=f"{label}\n(t = {tm_rf[idx]:.2f} ns)",
                )
                figstyle.spectrum_grid(ax)
            axes[0].set_ylabel("Electric field Ex(z, t) [V/m]")
            figstyle.finish(fig, auto_legend_below=False)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            pil_frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE))
            plt.close(fig)

    if pil_frames:
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )


def render_all_figures(data_dir: Path, fig_dir: Path) -> list[Path]:
    """Read ``data_dir/task_*.npz`` and render all tutorial figures into ``fig_dir``."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tasks_data = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        tasks_data[str(d["name"])] = d

    required = (
        "eccosorb_rf_lb_characterization",
        "free_space_slab_6classes",
        "metal_backed_absorber_and_rozanov",
        "resolution_and_mesh_study",
    )
    for req in required:
        if req not in tasks_data:
            raise FileNotFoundError(f"Missing task {req!r} in {data_dir}. Run --quick or --full first.")

    t0 = tasks_data["eccosorb_rf_lb_characterization"]
    t1 = tasks_data["free_space_slab_6classes"]
    t2 = tasks_data["metal_backed_absorber_and_rozanov"]
    t3 = tasks_data["resolution_and_mesh_study"]

    written: list[Path] = []

    # 1. Setup figures: two uncluttered figures per distinct run (00a/00b free-space slab, 00c/00d PEC-backed)
    written.extend(
        _save_setup_figure_pair(
            fig_dir,
            "00a_free_space_slab_setup",
            "00b_free_space_slab_mesh_pulse",
            "Setup 1: Free-space EMI slab R/T & absorption (CPML ±z, periodic x/y)",
            "Gaussian plane-wave source (f0 = 1.4 GHz)",
            pec_backed=False,
        )
    )
    written.extend(
        _save_setup_figure_pair(
            fig_dir,
            "00c_metal_backed_absorber_setup",
            "00d_metal_backed_absorber_mesh_pulse",
            "Setup 2: Metal-backed EMI absorber reflection loss (CPML -z, PEC +z)",
            "Broadband pulse spectrum (0.3–4.5 GHz)",
            pec_backed=True,
        )
    )

    with figstyle.apply_style():
        # ------------------------------------------------------------------
        # Fig 01: Eccosorb RF-LB permeability fit, eps' inference & 1-6 GHz extrapolation (3 panels)
        # ------------------------------------------------------------------
        fig1, (ax1a, ax1b, ax1c) = plt.subplots(
            1, 3, figsize=(figstyle.FULL_WIDTH_IN, 3.4), dpi=figstyle.DEFAULT_DPI
        )
        f_mu_g = t0["f_mu_hz"] * 1e-9
        f_w_g = t0["f_wide_hz"] * 1e-9
        mu_1d1l = t0["mu_wide_1d1l"]
        mu_2deb = t0["mu_wide_2deb"]

        s0 = figstyle.series_style(0, with_marker=False)
        s1 = figstyle.series_style(1, with_marker=False)
        s2 = figstyle.series_style(2, with_marker=False)
        ax1a.semilogx(f_mu_g, t0["mu_real_ds"], "o", color=s0["color"], ms=4.2, label=r"Datasheet $\mu'(f)$")
        ax1a.semilogx(f_mu_g, t0["mu_imag_ds"], "s", color=s1["color"], ms=4.2, label=r"Datasheet $\mu''(f)$")
        ax1a.semilogx(f_w_g, np.real(mu_1d1l), "-", color=s0["color"], lw=1.8, label="1 Debye + 1 Lorentz (0.36%)")
        ax1a.semilogx(f_w_g, -np.imag(mu_1d1l), "-", color=s1["color"], lw=1.8)
        ax1a.semilogx(f_w_g, -np.imag(mu_2deb), "--", color=s2["color"], lw=1.3, label=r"2 Debye $\mu''$ (4.53%)")
        ax1a.axvline(1.0, color="#666666", ls=":", lw=1.1)
        ax1a.set(
            xlabel="Frequency f [GHz]",
            ylabel=r"Permeability $\mu'(f),\, \mu''(f)$",
            title=r"(a) RF-LB $\mu(f)$ fit & extrapolation",
        )
        figstyle.spectrum_grid(ax1a)
        ax1a.legend(loc="upper right", frameon=False, fontsize=7.5)

        # Panel (b): Inferred eps' sensitivity over 0.1 - 1.0 GHz overlap band
        eps_sw = t0["eps_sweep"]
        ax1b.plot(eps_sw, t0["rms_vs_eps_td0"], label=r"$\tan\delta_e = 0.00$ ($\varepsilon'=82.7$)", **s0)
        ax1b.plot(eps_sw, t0["rms_vs_eps_td001"], label=r"$\tan\delta_e = 0.01$ ($\varepsilon'=81.1$)", **s1)
        ax1b.plot(eps_sw, t0["rms_vs_eps_td005"], label=r"$\tan\delta_e = 0.05$ ($\varepsilon'=74.9$)", **s2)
        ax1b.axvline(81.06, color=s1["color"], ls=":", lw=1.2)
        ax1b.set(
            xlabel=r"Inferred relative permittivity $\varepsilon'$",
            ylabel="Overlap RMS error [dB/cm]",
            title=r"(b) $\varepsilon'$ inference (0.1–1.0 GHz)",
        )
        figstyle.spectrum_grid(ax1b)
        ax1b.legend(loc="upper right", frameon=False, fontsize=7.5)

        # Panel (c): Predicted vs datasheet attenuation 0.1 - 6.0 GHz
        f_att_g = t0["f_att_hz"] * 1e-9
        ax1c.plot(f_att_g, t0["atten_ds_db_cm"], "ko", ms=4.0, label="Datasheet attenuation")
        ax1c.plot(f_w_g, t0["att_wide_1d1l"], label="1 Debye + 1 Lorentz (extrap.)", **s0)
        ax1c.plot(f_w_g, t0["att_wide_2deb"], ls="--", color=s2["color"], lw=1.5, label="2 Debye (extrap. RMS 3.4 dB/cm)")
        ax1c.axvspan(0.1, 1.0, color="#D5ECE4", alpha=0.45, label="Overlap (0.1–1 GHz)")
        ax1c.set(
            xlim=(0.0, 6.1),
            ylim=(0.0, 105.0),
            xlabel="Frequency f [GHz]",
            ylabel=r"Attenuation $\alpha(f)$ [dB/cm]",
            title=r"(c) Attenuation prediction (0.1–6 GHz)",
        )
        figstyle.spectrum_grid(ax1c)
        ax1c.legend(loc="upper left", frameon=False, fontsize=7.5)

        figstyle.finish(fig1, auto_legend_below=False)
        p1 = fig_dir / "01_eccosorb_rf_lb_fit_and_attenuation.png"
        figstyle.save(fig1, p1, close=True)
        written.append(p1)

        # ------------------------------------------------------------------
        # Fig 02: Six EMI material classes constitutive spectra (3 panels)
        # ------------------------------------------------------------------
        fig2, (ax2a, ax2b, ax2c) = plt.subplots(
            1, 3, figsize=(figstyle.FULL_WIDTH_IN, 3.4), dpi=figstyle.DEFAULT_DPI
        )
        eps_6 = t0["eps_6"]
        mu_6 = t0["mu_6"]
        zrel_6 = t0["zrel_6"]

        for ic, cinfo in enumerate(EMI_CLASSES):
            st = figstyle.series_style(ic, with_marker=False)
            ax2a.loglog(f_w_g, np.abs(eps_6[ic]), label=cinfo["short"], **st)
            ax2c.semilogx(f_w_g, np.abs(zrel_6[ic]), label=cinfo["short"], **st)

        ax2a.set(
            xlabel="Frequency f [GHz]",
            ylabel=r"Permittivity magnitude $|\varepsilon(f)|$",
            title=r"(a) Electric response $|\varepsilon(f)|$",
        )
        figstyle.spectrum_grid(ax2a)
        ax2a.legend(loc="upper right", frameon=False, fontsize=7.0)

        # Panel (b): Magnetic permeability of Class 1 (RF-LB) & Class 2 (Sendust flake)
        st_rf = figstyle.series_style(0, with_marker=False)
        st_sd = figstyle.series_style(1, with_marker=False)
        ax2b.semilogx(f_w_g, np.real(mu_6[0]), "-", color=st_rf["color"], lw=1.8, label=r"RF-LB $\mu'(f)$")
        ax2b.semilogx(f_w_g, -np.imag(mu_6[0]), "--", color=st_rf["color"], lw=1.6, label=r"RF-LB $\mu''(f)$")
        ax2b.semilogx(f_w_g, np.real(mu_6[1]), "-", color=st_sd["color"], lw=1.8, label=r"Sendust $\mu'(f)$")
        ax2b.semilogx(f_w_g, -np.imag(mu_6[1]), "--", color=st_sd["color"], lw=1.6, label=r"Sendust $\mu''(f)$")
        ax2b.set(
            xlabel="Frequency f [GHz]",
            ylabel=r"Permeability $\mu'(f),\, \mu''(f)$",
            title=r"(b) Magnetic absorbers $\mu(f)$",
        )
        figstyle.spectrum_grid(ax2b)
        ax2b.legend(loc="upper right", frameon=False, fontsize=7.5)

        ax2c.axhline(1.0, color="#444444", ls=":", lw=1.1, label="Vacuum match |Z|/Z0 = 1")
        ax2c.set(
            ylim=(0.0, 1.05),
            xlabel="Frequency f [GHz]",
            ylabel=r"Normalized wave impedance $|Z(f)| / Z_0$",
            title=r"(c) Wave impedance $|Z(f)| / Z_0$",
        )
        figstyle.spectrum_grid(ax2c)

        figstyle.finish(fig2, auto_legend_below=False)
        p2 = fig_dir / "02_six_material_classes_dispersion.png"
        figstyle.save(fig2, p2, close=True)
        written.append(p2)

        # ------------------------------------------------------------------
        # Fig 03: Free-space slab |R|^2, |T|^2, and Absorbed fraction A vs frequency (3 panels)
        # ------------------------------------------------------------------
        fig3, (ax3a, ax3b, ax3c) = plt.subplots(
            1, 3, figsize=(figstyle.FULL_WIDTH_IN, 3.5), dpi=figstyle.DEFAULT_DPI
        )
        f_s_g = t1["freqs"] * 1e-9
        # Plot index 1 thickness for each class (e.g. 2.0 mm for RF-LB/TiO2, 1.0 mm for Sendust/CNT/Ni-C, 50 nm for Cu)
        it_show = 1
        for ic, cinfo in enumerate(EMI_CLASSES):
            d_val = float(t1["d_mm_table"][ic, it_show])
            d_str = f"{d_val * 1e3:.0f} nm" if cinfo["use_thin_sheet"] else f"{d_val:.1f} mm"
            lbl = f"{cinfo['short']} ({d_str})"
            st_line = figstyle.series_style(ic, with_marker=False)
            st_mk = figstyle.series_style(ic, with_marker=True)

            ax3a.plot(f_s_g, t1["R_ex_table"][ic, it_show], **st_line)
            ax3a.plot(
                f_s_g[::3],
                t1["R_fit_table"][ic, it_show, ::3],
                ls="none",
                color=st_mk["color"],
                marker=st_mk["marker"],
                ms=4.2,
                label=lbl,
            )

            ax3b.semilogy(f_s_g, np.maximum(t1["T_ex_table"][ic, it_show], 1e-10), **st_line)
            ax3b.semilogy(
                f_s_g[::3],
                np.maximum(t1["T_fit_table"][ic, it_show, ::3], 1e-10),
                ls="none",
                color=st_mk["color"],
                marker=st_mk["marker"],
                ms=4.2,
            )

            ax3c.plot(f_s_g, t1["A_ex_table"][ic, it_show], **st_line)
            ax3c.plot(
                f_s_g[::3],
                t1["A_fit_table"][ic, it_show, ::3],
                ls="none",
                color=st_mk["color"],
                marker=st_mk["marker"],
                ms=4.2,
            )

        ax3a.set(
            ylim=(-0.02, 1.05),
            xlabel="Frequency f [GHz]",
            ylabel=r"Power reflectance $|R(f)|^2$",
            title=r"(a) Reflectance $|R|^2$ (FIT vs. TM)",
        )
        ax3b.set(
            ylim=(1e-8, 1.5),
            xlabel="Frequency f [GHz]",
            ylabel=r"Power transmittance $|T(f)|^2$",
            title=r"(b) Transmittance $|T|^2$ (FIT vs. TM)",
        )
        ax3c.set(
            ylim=(-0.02, 0.85),
            xlabel="Frequency f [GHz]",
            ylabel=r"Absorbed fraction $A = 1 - |R|^2 - |T|^2$",
            title="(c) Absorbed power fraction A(f)",
        )
        for ax in (ax3a, ax3b, ax3c):
            figstyle.spectrum_grid(ax)
        ax3a.legend(loc="lower right", frameon=False, fontsize=6.8)

        figstyle.finish(fig3, auto_legend_below=False)
        p3 = fig_dir / "03_slab_rt_and_absorption_spectra.png"
        figstyle.save(fig3, p3, close=True)
        written.append(p3)

        # ------------------------------------------------------------------
        # Fig 04: Schelkunoff SE = SE_R + SE_A + SE_M decomposition (3 panels)
        # ------------------------------------------------------------------
        fig4, (ax4a, ax4b, ax4c) = plt.subplots(
            1, 3, figsize=(figstyle.FULL_WIDTH_IN, 3.5), dpi=figstyle.DEFAULT_DPI
        )
        for ic, cinfo in enumerate(EMI_CLASSES):
            st_line = figstyle.series_style(ic, with_marker=False)
            st_mk = figstyle.series_style(ic, with_marker=True)
            ax4a.plot(f_s_g, t1["SE_ex_table"][ic, it_show], **st_line)
            ax4a.plot(
                f_s_g[::3],
                t1["SE_fit_table"][ic, it_show, ::3],
                ls="none",
                color=st_mk["color"],
                marker=st_mk["marker"],
                ms=4.2,
                label=cinfo["short"],
            )

        ax4a.set(
            ylim=(0.0, 95.0),
            xlabel="Frequency f [GHz]",
            ylabel=r"Shielding effectiveness $\mathrm{SE} = -10\log_{10}|T|^2$ [dB]",
            title="(a) Total SE(f): 3D FIT vs. exact",
        )
        figstyle.spectrum_grid(ax4a)
        ax4a.legend(loc="upper left", frameon=False, fontsize=7.0)

        # Panel (b): Bar decomposition into SE_R, SE_A, SE_M near 2.0 GHz
        idx_2g = int(np.argmin(np.abs(f_s_g - 2.0)))
        x_pos = np.arange(len(EMI_CLASSES))
        se_r_vals = t1["SE_R_table"][:, it_show, idx_2g]
        se_a_vals = t1["SE_A_table"][:, it_show, idx_2g]
        se_m_vals = t1["SE_M_table"][:, it_show, idx_2g]
        w_bar = 0.26
        ax4b.bar(x_pos - w_bar, se_r_vals, width=w_bar, color="#0072B2", label=r"$\mathrm{SE}_R$ (reflection)")
        ax4b.bar(x_pos, se_a_vals, width=w_bar, color="#D55E00", label=r"$\mathrm{SE}_A$ (absorption)")
        ax4b.bar(x_pos + w_bar, se_m_vals, width=w_bar, color="#009E73", label=r"$\mathrm{SE}_M$ (mult. refl.)")
        ax4b.axhline(0.0, color="#444444", lw=0.8)
        ax4b.set_xticks(x_pos)
        ax4b.set_xticklabels(["RF-LB", "Sendust", "TiO2", "CNT", "Ni-C", "Cu"], rotation=25, fontsize=7.5)
        ax4b.set(
            ylabel="Schelkunoff contribution [dB]",
            title=f"(b) Decomposition at f = {f_s_g[idx_2g]:.1f} GHz",
        )
        figstyle.spectrum_grid(ax4b)
        ax4b.legend(loc="upper left", frameon=False, fontsize=7.2)

        # Panel (c): SE vs thickness at 2.0 GHz
        d_sw = t1["d_sweep_mm"]
        for ic in range(5):  # 5 bulk millimeter classes on common [0.05, 5.0] mm axis
            st_c = figstyle.series_style(ic, with_marker=False)
            ax4c.semilogx(d_sw, t1["SE_vs_d_total"][ic], label=EMI_CLASSES[ic]["short"], **st_c)
        ax4c.set(
            ylim=(0.0, 120.0),
            xlabel="Slab thickness d [mm]",
            ylabel="Total SE at 2.0 GHz [dB]",
            title="(c) SE(d) scaling at 2.0 GHz",
        )
        figstyle.spectrum_grid(ax4c)
        ax4c.legend(loc="upper left", frameon=False, fontsize=7.0)

        figstyle.finish(fig4, auto_legend_below=False)
        p4 = fig_dir / "04_schelkunoff_se_decomposition.png"
        figstyle.save(fig4, p4, close=True)
        written.append(p4)

        # ------------------------------------------------------------------
        # Fig 05: Metal-backed absorber RL(f, d) & Rozanov causality bound (3 panels)
        # ------------------------------------------------------------------
        fig5, (ax5a, ax5b, ax5c) = plt.subplots(
            1, 3, figsize=(figstyle.FULL_WIDTH_IN, 3.5), dpi=figstyle.DEFAULT_DPI
        )
        f_p_g = t2["freqs"] * 1e-9
        for idx_d, d_mm in enumerate(t2["rf_thicks_mm"]):
            st_l = figstyle.series_style(idx_d, with_marker=False)
            st_m = figstyle.series_style(idx_d, with_marker=True)
            ax5a.plot(f_p_g, t2["RL_rf_ex"][idx_d], **st_l)
            ax5a.plot(
                f_p_g[::2],
                t2["RL_rf_fit"][idx_d, ::2],
                ls="none",
                color=st_m["color"],
                marker=st_m["marker"],
                ms=4.2,
                label=f"RF-LB d = {d_mm:.1f} mm",
            )
        for idx_d, d_mm in enumerate(t2["sd_thicks_mm"]):
            st_l = figstyle.series_style(idx_d + 3, with_marker=False)
            st_m = figstyle.series_style(idx_d + 3, with_marker=True)
            ax5a.plot(f_p_g, t2["RL_sd_ex"][idx_d], ls="--", color=st_l["color"], lw=1.5)
            ax5a.plot(
                f_p_g[::2],
                t2["RL_sd_fit"][idx_d, ::2],
                ls="none",
                color=st_m["color"],
                marker=st_m["marker"],
                ms=4.0,
                label=f"Sendust d = {d_mm:.1f} mm",
            )
        ax5a.axhline(10.0, color="#666666", ls=":", lw=1.0, label="10 dB (90% abs.)")
        ax5a.set(
            xlabel="Frequency f [GHz]",
            ylabel=r"Reflection loss $\mathrm{RL} = -20\log_{10}|\Gamma|$ [dB]",
            title="(a) PEC-backed RL(f): FIT vs. exact",
        )
        figstyle.spectrum_grid(ax5a)
        ax5a.legend(loc="upper right", frameon=False, fontsize=6.6)

        # Panel (b): 2D RL(f, d) map + quarter-wave trajectory d = lambda_0 / (4 Re n)
        f_m_g = t2["f_map"] * 1e-9
        d_m_mm = t2["d_map_mm"]
        im5 = ax5b.pcolormesh(f_m_g, d_m_mm, t2["RL_map_rf"], shading="auto", cmap="viridis", vmin=0.0, vmax=22.0)
        ax5b.plot(f_m_g, t2["d_qw_mm"], "w--", lw=1.8, label=r"$d_{\lambda/4} = \lambda_0 / (4\operatorname{Re}\sqrt{\varepsilon\mu})$")
        fig5.colorbar(im5, ax=ax5b, pad=0.02, label="RL [dB]")
        ax5b.set(
            xlim=(float(f_m_g[0]), float(f_m_g[-1])),
            ylim=(float(d_m_mm[0]), float(d_m_mm[-1])),
            xlabel="Frequency f [GHz]",
            ylabel="Thickness d [mm]",
            title=r"(b) RF-LB $\lambda/4$ matching locus",
        )
        ax5b.legend(loc="upper right", frameon=True, facecolor="#222222", labelcolor="white", fontsize=7.0)

        # Panel (c): Rozanov bound comparison
        d_rz = t2["d_roz_mm"]
        s_r0 = figstyle.series_style(0, with_marker=True)
        s_r1 = figstyle.series_style(1, with_marker=True)
        ax5c.semilogy(d_rz, t2["roz_bound_rf_mm"], "--", color=s_r0["color"], lw=1.5, label=r"Rozanov bound $2\pi^2\mu_s d$ (RF-LB)")
        ax5c.semilogy(d_rz, t2["roz_int_rf_mm"], color=s_r0["color"], marker="o", lw=1.8, label=r"Integrated $\int|\ln|\Gamma||d\lambda$ (RF-LB)")
        ax5c.semilogy(d_rz, t2["roz_bound_sd_mm"], "--", color=s_r1["color"], lw=1.5, label=r"Rozanov bound $2\pi^2\mu_s d$ (Sendust)")
        ax5c.semilogy(d_rz, t2["roz_int_sd_mm"], color=s_r1["color"], marker="s", lw=1.8, label=r"Integrated $\int|\ln|\Gamma||d\lambda$ (Sendust)")
        ax5c.semilogy(d_rz, t2["roz_bound_nonmag_mm"], "k:", lw=1.3, label=r"Dielectric bound $2\pi^2 d$ ($\mu_s=1$)")
        ax5c.set(
            xlabel="Absorber thickness d [mm]",
            ylabel=r"Bandwidth integral $\int |\ln|\Gamma(\lambda)||\,d\lambda$ [mm]",
            title="(c) Rozanov (2000) causality bound",
        )
        figstyle.spectrum_grid(ax5c)
        ax5c.legend(loc="lower right", frameon=False, fontsize=6.5)

        figstyle.finish(fig5, auto_legend_below=False)
        p5 = fig_dir / "05_metal_backed_absorber_and_rozanov.png"
        figstyle.save(fig5, p5, close=True)
        written.append(p5)

        # ------------------------------------------------------------------
        # Fig 06: Resolution study & locally refined z-mesh vs uniform baseline (2 panels)
        # ------------------------------------------------------------------
        fig6, (ax6a, ax6b) = plt.subplots(
            1, 2, figsize=(figstyle.FULL_WIDTH_IN, 3.3), dpi=figstyle.DEFAULT_DPI
        )
        cpw_lv = t3["cpw_levels"]
        conv_labels = ["Eccosorb RF-LB (2 mm)", "TiO2 ceramic (2 mm)", "CNT silicone (1 mm)", "Ni-graphite gasket (1 mm)"]
        for ic, lbl in enumerate(conv_labels):
            st_c = figstyle.series_style(ic, with_marker=True)
            ax6a.loglog(cpw_lv, t3["rms_R_conv"][ic], label=lbl, **st_c)

        ref_line = t3["rms_R_conv"][0, 0] * (cpw_lv / cpw_lv[0]) ** (-2.0)
        ax6a.loglog(cpw_lv, ref_line, "k--", lw=1.2, label=r"$\mathcal{O}(\Delta z^2)$ reference")
        ax6a.axvline(20.0, color="#666666", ls=":", lw=1.1, label="20 cells/λ rule")
        ax6a.set(
            xlabel=r"Resolution inside slab $\mathrm{cpw}_{\mathrm{slab}}$ [cells/$\lambda_{\mathrm{mat}}$]",
            ylabel=r"Broadband RMS error $\mathrm{RMS}(|R_{\mathrm{FIT}}|^2 - |R_{\mathrm{exact}}|^2)$",
            title="(a) Grid convergence across EMI classes",
        )
        figstyle.spectrum_grid(ax6a)
        ax6a.legend(loc="upper right", frameon=False, fontsize=7.5)

        # Panel (b): Locally refined z-mesh vs uniform z-mesh baseline (error vs cell count Nz)
        n_base = len(t3["cpw_base"])
        st_gr = figstyle.series_style(0, with_marker=True)
        st_un = figstyle.series_style(1, with_marker=True)
        ax6b.loglog(
            t3["nz_graded_conv"][0, :n_base],
            t3["rms_R_conv"][0, :n_base],
            label="Locally refined graded z-mesh",
            **st_gr,
        )
        ax6b.loglog(
            t3["nz_uniform"],
            t3["rms_R_uniform"],
            ls="--",
            color=st_un["color"],
            marker=st_un["marker"],
            lw=1.6,
            label="Uniform z-mesh baseline (same dz_slab)",
        )
        ax6b.set(
            xlabel="Longitudinal cell count Nz [cells]",
            ylabel=r"RMS reflectance error (RF-LB, d = 2.0 mm)",
            title="(b) Local refinement vs. uniform baseline",
        )
        figstyle.spectrum_grid(ax6b)
        ax6b.legend(loc="upper right", frameon=False, fontsize=7.8)

        figstyle.finish(fig6, auto_legend_below=False)
        p6 = fig_dir / "06_resolution_convergence_study.png"
        figstyle.save(fig6, p6, close=True)
        written.append(p6)

        # ------------------------------------------------------------------
        # Fig 07: Pulse-slab interaction snapshots with overlay_geometry (3 panels)
        # ------------------------------------------------------------------
        fig7, axes7 = plt.subplots(
            1, 3, figsize=(figstyle.FULL_WIDTH_IN, 3.3), dpi=figstyle.DEFAULT_DPI, sharey=True
        )
        solver_rf, src_rf, mon_rf, _, _, _, _ = _build_schematic_objects(pec_backed=False)
        snap_specs = [
            (
                axes7[0],
                t1["snap_rf_z_m"] * 1e3,
                t1["snap_rf_frames"],
                t1["snap_rf_times"] * 1e9,
                2.0,
                "Eccosorb RF-LB (2 mm)",
                "(a) Magnetic absorber (RF-LB)",
            ),
            (
                axes7[1],
                t1["snap_tio2_z_m"] * 1e3,
                t1["snap_tio2_frames"],
                t1["snap_tio2_times"] * 1e9,
                2.0,
                "TiO2 eps_r=100 (2 mm)",
                "(b) Dielectric resonator (TiO2)",
            ),
            (
                axes7[2],
                t1["snap_cnt_z_m"] * 1e3,
                t1["snap_cnt_frames"],
                t1["snap_cnt_times"] * 1e9,
                1.0,
                "CNT silicone (1 mm)",
                "(c) Percolated conductor (CNT)",
            ),
        ]
        for ax, z_mm, fr_arr, t_ns, d_mm, s_name, p_title in snap_specs:
            n_f = len(fr_arr)
            i_early = max(0, int(0.32 * n_f))
            i_peak = max(1, int(0.50 * n_f))
            i_late = max(2, int(0.72 * n_f))
            for k_i, (f_idx, tag) in enumerate([(i_early, "incident"), (i_peak, "impinging"), (i_late, "trans./ring")]):
                st_k = figstyle.series_style(k_i, with_marker=False)
                ax.plot(z_mm, fr_arr[f_idx], label=f"t = {t_ns[f_idx]:.2f} ns ({tag})", **st_k)
            slab_s = SlabBox(0.0, d_mm * 1e-3, 2.0e-3, dispersive_lib["eccosorb rf-lb"], s_name)
            schematic.overlay_geometry(
                ax,
                solver_rf,
                plane="zx",
                unit="mm",
                orientation="1d",
                structures=[slab_s],
                sources=src_rf,
                monitors=mon_rf,
                show_boundaries=True,
                show_materials=True,
                show_sources=True,
                show_monitors=True,
            )
            ax.set(xlim=(-25.0, 25.0), xlabel="Position z [mm]", title=p_title)
            figstyle.spectrum_grid(ax)
            ax.legend(loc="lower left", frameon=False, fontsize=6.8)
        axes7[0].set_ylabel("Electric field Ex(z, t) [V/m]")

        figstyle.finish(fig7, auto_legend_below=False)
        p7 = fig_dir / "07_pulse_slab_interaction_snapshots.png"
        figstyle.save(fig7, p7, close=True)
        written.append(p7)

    # ------------------------------------------------------------------
    # Fig 08: Animated GIF of pulse interacting with the EMI slabs (< 5 MB)
    # ------------------------------------------------------------------
    p8 = fig_dir / "08_emi_slab_wave_propagation.gif"
    _save_wave_propagation_gif(t1, p8, fps=10)
    if p8.is_file():
        written.append(p8)

    return written
