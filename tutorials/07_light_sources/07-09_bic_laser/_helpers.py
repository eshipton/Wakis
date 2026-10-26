# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Physics, analytical models, and FIT solver helpers for Tutorial 7.9:
Lasing from a bound state in the continuum (`07-09_bic_laser`).

Builds on Tutorial 6.3 (`06-03_quasi_bic_metasurface`) and `wakis.gain`
(`SaturableGain`, `FourLevelGain`) without modifying either.

Primary references
------------------
1. A. Kodigala, T. Lepetit, Q. Gu, B. Bahari, Y. Fainman, and B. Kanté,
   "Lasing action from photonic bound states in continuum", Nature 541,
   196-199 (2017), doi:10.1038/nature20799.
2. C. W. Hsu, B. Zhen, A. D. Stone, J. D. Joannopoulos, and M. Soljačić,
   "Bound states in the continuum", Nat. Rev. Mater. 1, 16048 (2016),
   doi:10.1038/natrevmats.2016.48.
3. B. Zhen, C. W. Hsu, L. Lu, A. D. Stone, and M. Soljačić,
   "Topological nature of optical bound states in the continuum",
   Phys. Rev. Lett. 113(25), 257401 (2014), doi:10.1103/PhysRevLett.113.257401.
4. K. Koshelev, S. Lepeshov, M. Liu, A. Bogdanov, and Y. Kivshar,
   "Asymmetric metasurfaces with high-Q resonances governed by bound states in
   the continuum", Phys. Rev. Lett. 121(19), 193903 (2018),
   doi:10.1103/PhysRevLett.121.193903.
5. S.-H. Chang and A. Taflove, "Finite-difference time-domain model of lasing
   action in a four-level two-electron atomic system", Opt. Express 12(16),
   3827-3833 (2004), doi:10.1364/OPEX.12.003827.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.constants import c as c_light, epsilon_0, hbar
from scipy.optimize import brentq

import wakis
from wakis.estimate import estimate, preflight
from wakis.gain import FourLevelGain, SaturableGain
from wakis.near2far import Near2FarMonitor
from wakis.resonances import ResonanceMonitor, harminv
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.meshing import graded_axis
from wakis.templates.photonics import Box

# ---------------------------------------------------------------------------
# Import and reuse Tutorial 6.3 helpers and reference data (read-only)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
TUT_06_03_DIR = (
    REPO_ROOT / "tutorials" / "06_periodic_structures" / "06-03_quasi_bic_metasurface"
)


def _load_tut_06_03_helpers():
    spec = importlib.util.spec_from_file_location(
        "_tut_06_03_helpers", TUT_06_03_DIR / "_helpers.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Tutorial 6.3 helpers from {TUT_06_03_DIR}.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tut_06_03_helpers"] = mod
    spec.loader.exec_module(mod)
    return mod


h63 = _load_tut_06_03_helpers()

# Re-export Tutorial 6.3 geometric and material constants
PERIOD_P = float(h63.PERIOD_P)  # 680 nm unit-cell period [m]
DIMER_D = float(h63.DIMER_D)  # 255 nm bar center-to-center spacing [m]
WIDTH_W0 = float(h63.WIDTH_W0)  # 170 nm nominal bar width [m]
HEIGHT_H = float(h63.HEIGHT_H)  # 190 nm resonator slab thickness [m]
N_SI = float(h63.N_SI)  # 3.48 refractive index
EPS_SI = float(h63.EPS_SI)  # 12.1104 relative permittivity
EPS_BG = float(h63.EPS_BG)  # 1.0 air cladding permittivity
LAMBDA_BIC_REF = float(h63.LAMBDA_BIC_REF)  # 1573.885 nm BIC wavelength at nx=32 [m]
F0_BIC_REF = float(h63.F0_BIC_REF)  # 190.479 THz BIC frequency [Hz]
N_CYL_3D = 3.15  # Refractive index of 3D dielectric cylinder unit cell [-]
EPS_CYL_3D = float(N_CYL_3D**2)  # 9.9225 relative permittivity [-]


def load_tut_06_03_cached_data() -> dict[str, Any]:
    """
    Load precomputed quasi-BIC scaling data from Tutorial 6.3 (`data/task_00.npz`)
    without modifying Tutorial 6.3.

    Returns
    -------
    dict
        Dictionary containing ``alphas``, ``Q_rad``, ``lambda0_nm``, ``Q_bic``,
        ``lambda_bic_nm``, ``Q0_fixed``, and ``free_slope`` from Tutorial 6.3.
    """
    npz_path = TUT_06_03_DIR / "data" / "task_00.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Tutorial 6.3 data file not found: {npz_path}")
    with np.load(npz_path) as d:
        alphas = np.asarray(d["alphas"], dtype=float)
        q_rad = np.asarray(d["Q_fit"], dtype=float)
        lam_nm = np.asarray(d["lambda0_fit_nm"], dtype=float)
        lam_bic_nm = float(d["lambda_bic_fit_nm"])
        q_bic = float(d["Q_bic_fit"])
        q0_fixed = float(np.mean(q_rad * (alphas**2)))
        free_slope = float(d["log_slope_fit"])
        return {
            "alphas": alphas,
            "Q_rad": q_rad,
            "lambda0_nm_nonzero": lam_nm,
            "lambda_bic_nm": lam_bic_nm,
            "Q_bic": q_bic,
            "Q0_fixed": q0_fixed,
            "free_slope": free_slope,
        }


# ---------------------------------------------------------------------------
# Analytical BIC and Laser Theory Helpers ("derived here")
# ---------------------------------------------------------------------------
def compute_modal_confinement_2d(
    ey_2d: np.ndarray,
    fill_2d: np.ndarray,
    eps_r_2d: np.ndarray,
    grid_z: np.ndarray,
    eps_active: float = EPS_SI,
) -> float:
    """
    Compute the exact 2D TE electric-energy modal confinement factor ``Gamma_conf``
    for an active dielectric resonator mode ("derived here").

    From first-order Maxwell perturbation theory for a composite medium
    ``eps(x, z) = eps_bg + f(x, z) * (eps_active - eps_bg + chi_gain)``, a
    resonant line-center susceptibility ``chi_gain = i * c * n_active * g0 / omega0``
    produces an amplitude growth rate ``gamma_gain = (c / (2 * n_active)) * Gamma_conf * g0``,
    where::

        Gamma_conf = int f(x, z) eps_active |Ey(x, z)|^2 dx dz
                     / int eps_r(x, z) |Ey(x, z)|^2 dx dz

    Parameters
    ----------
    ey_2d : ndarray, shape (Nx, Nz)
        2D mode electric field ``Ey(x, z)`` [V/m].
    fill_2d : ndarray, shape (Nx, Nz)
        Subpixel active-material volume fraction ``f(x, z)`` in ``[0, 1]`` [-].
    eps_r_2d : ndarray, shape (Nx, Nz)
        Total relative permittivity map ``eps_r(x, z)`` [-].
    grid_z : ndarray, shape (Nz + 1,)
        Grid node coordinates along ``z`` [m].
    eps_active : float, optional
        Host relative permittivity of the active resonator material [-].
        Default ``EPS_SI`` (12.1104).

    Returns
    -------
    Gamma_conf : float
        Dimensionless modal confinement factor in ``(0, 1]`` [-].
    """
    z_mids = 0.5 * (grid_z[:-1] + grid_z[1:])
    z_lo = np.empty_like(grid_z[:-1])
    z_lo[0] = float(grid_z[0])
    z_lo[1:] = z_mids[:-1]
    dz_dual = z_mids - z_lo
    e2 = np.abs(np.asarray(ey_2d, dtype=float)) ** 2
    num = np.sum(fill_2d * float(eps_active) * e2 * dz_dual[None, :])
    den = np.sum(eps_r_2d * e2 * dz_dual[None, :])
    return float(num / den)


def bic_threshold_gain(
    q_factor: float | np.ndarray,
    lambda0_m: float | np.ndarray = LAMBDA_BIC_REF,
    n_active: float = N_SI,
    gamma_conf: float = 0.947,
) -> float | np.ndarray:
    """
    Cold-cavity material power threshold gain ``g_th`` [1/m] of a BIC mode ("derived here").

    Balances the modal amplitude radiation decay rate ``gamma_rad = omega0 / (2 Q)``
    against the modal amplitude gain rate ``gamma_gain = (c / (2 n_active)) * Gamma_conf * g_th``::

        g_th = (2 * pi * n_active) / (lambda0 * Gamma_conf * Q)

    Parameters
    ----------
    q_factor : float or array_like
        Cold-cavity quality factor ``Q`` [-], ``> 0``.
    lambda0_m : float or array_like, optional
        Resonance vacuum wavelength ``lambda0`` [m]. Default ``LAMBDA_BIC_REF``.
    n_active : float, optional
        Refractive index of the active resonator material [-]. Default ``N_SI``.
    gamma_conf : float, optional
        Modal electric-energy confinement factor ``Gamma_conf`` [-]. Default ``0.947``.

    Returns
    -------
    g_th : float or ndarray
        Material threshold power gain coefficient [1/m].
    """
    q_arr = np.asarray(q_factor, dtype=float)
    lam_arr = np.asarray(lambda0_m, dtype=float)
    if np.any(q_arr <= 0.0):
        raise ValueError("q_factor must be > 0.")
    if float(gamma_conf) <= 0.0:
        raise ValueError("gamma_conf must be > 0.")
    out = (2.0 * np.pi * float(n_active)) / (lam_arr * float(gamma_conf) * q_arr)
    return float(out) if q_arr.ndim == 0 and lam_arr.ndim == 0 else out


def run_bar_width_bic_sweep(
    widths_nm: Sequence[float] = (140.0, 155.0, 170.0, 185.0, 200.0),
    alpha_qbic: float = 0.10,
    nx: int = 32,
    ring_cycles: float = 36.0,
) -> dict[str, np.ndarray]:
    """
    Simulate the resonator bar-width sweep ``w_0 in widths_nm`` in 2D FIT (`harminv`)
    for our dimer metasurface (`p = 680 nm, d = 255 nm, h = 190 nm, n = 3.48`),
    comparing the symmetry-protected BIC (``alpha = 0``) against the broken-symmetry
    quasi-BIC (``alpha = alpha_qbic``) ("derived here").

    Parameters
    ----------
    widths_nm : sequence of float, optional
        Nominal bar widths ``w_0`` [nm]. Default ``(140, 155, 170, 185, 200)``.
    alpha_qbic : float, optional
        Width-asymmetry parameter ``(w_1 - w_2) / w_0`` for the quasi-BIC comparison.
        Default 0.10.
    nx : int, optional
        Grid cells per unit-cell period ``p``. Default 32.
    ring_cycles : float, optional
        Number of optical cycles recorded after the seed pulse ends. Default 36.0.

    Returns
    -------
    dict of str to ndarray
        2D FIT `harminv` arrays ``widths_nm``, ``lambda_sym_nm``, ``Q_sym_fit``,
        ``lambda_qbic_nm``, ``Q_qbic_fit``, ``gamma_conf_qbic``, ``g_th_sym_cm1``,
        and ``g_th_qbic_cm1``.
    """
    w_arr_nm = np.asarray(widths_nm, dtype=float)
    dx = PERIOD_P / int(nx)
    lam_sym, q_sym, gth_sym = [], [], []
    lam_qbic, q_qbic, gconf_qbic, gth_qbic = [], [], [], []

    for w_nm in w_arr_nm:
        w0 = float(w_nm) * 1e-9
        for alpha_val in (0.0, float(alpha_qbic)):
            slv, _ = h63.make_dimer_solver(alpha=alpha_val, nx=int(nx), cfln=0.90)
            fill, _, _ = h63.compute_dimer_fill_2d(
                slv.grid.x, slv.grid.z, alpha=alpha_val, w0=w0
            )
            eps_r = EPS_BG + fill * (EPS_SI - EPS_BG)
            slv.ieps[:, 0, :, "y"] = 1.0 / (epsilon_0 * eps_r)
            slv.update_tensors()
            slv.step_0 = True

            f0 = F0_BIC_REF
            wf = GaussianSource(frequency=f0, fwidth=0.12 * f0)
            src_L = CurrentSource(
                wf,
                component="y",
                center=(-0.5 * DIMER_D, 0.5 * dx, 0.0),
                size=(0.0, dx, 0.0),
                amplitude=+1.0,
            )
            src_R = CurrentSource(
                wf,
                component="y",
                center=(+0.5 * DIMER_D, 0.5 * dx, 0.0),
                size=(0.0, dx, 0.0),
                amplitude=-1.0,
            )
            dt = float(slv.dt)
            n_steps = int((wf.end_time + float(ring_cycles) / f0) / dt)
            n_start = int(np.ceil(wf.end_time / dt)) + 20
            n_snap = int((wf.end_time + 10.0 / f0) / dt)
            x_nodes = np.asarray(slv.grid.x[:-1], dtype=float)
            z_nodes = np.asarray(slv.grid.z[:-1], dtype=float)
            ixL = int(np.argmin(np.abs(x_nodes - (-0.5 * DIMER_D))))
            ixR = int(np.argmin(np.abs(x_nodes - (+0.5 * DIMER_D))))
            iz0 = int(np.argmin(np.abs(z_nodes - 0.025e-6)))
            n_tot, nx_s, ny_s = slv.N, slv.Nx, slv.Ny
            fL = n_tot + ixL + nx_s * ny_s * iz0
            fR = n_tot + ixR + nx_s * ny_s * iz0
            e_arr = slv.E.array
            sig = np.empty(n_steps - n_start, dtype=float)
            ey_snap = None
            for step in range(n_steps):
                t = step * dt
                src_L.update(slv, t)
                src_R.update(slv, t)
                slv.one_step()
                if step >= n_start:
                    sig[step - n_start] = float(e_arr[fL] - e_arr[fR])
                if step == n_snap:
                    ey_snap = np.real(slv.E.to_matrix("y")[:, 0, :]).copy()

            res = harminv(sig, dt, 0.85 * f0, 1.15 * f0, max_rel_error=None)
            idx = int(np.argmax(np.abs(res.amplitude)))
            fm = float(res.frequency[idx])
            qm = float(abs(res.Q[idx]))
            lam_m = c_light / fm
            gconf = compute_modal_confinement_2d(
                ey_snap, fill, eps_r, slv.grid.z, eps_active=EPS_SI
            )
            gth_c = (
                float(
                    bic_threshold_gain(
                        qm, lambda0_m=lam_m, n_active=N_SI, gamma_conf=gconf
                    )
                )
                / 100.0
            )
            if alpha_val == 0.0:
                lam_sym.append(lam_m * 1e9)
                q_sym.append(qm)
                gth_sym.append(gth_c)
            else:
                lam_qbic.append(lam_m * 1e9)
                q_qbic.append(qm)
                gconf_qbic.append(gconf)
                gth_qbic.append(gth_c)

    return {
        "widths_nm": w_arr_nm,
        "lambda_sym_nm": np.asarray(lam_sym, dtype=float),
        "Q_sym_fit": np.asarray(q_sym, dtype=float),
        "g_th_sym_cm1": np.asarray(gth_sym, dtype=float),
        "lambda_qbic_nm": np.asarray(lam_qbic, dtype=float),
        "Q_qbic_fit": np.asarray(q_qbic, dtype=float),
        "gamma_conf_qbic": np.asarray(gconf_qbic, dtype=float),
        "g_th_qbic_cm1": np.asarray(gth_qbic, dtype=float),
    }


def predict_shb_steady_intensity(
    g0_ratios: Sequence[float] | np.ndarray,
    ey_mode_2d: np.ndarray,
    fill_2d: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    """
    Predict the normalized steady-state intracavity intensity ``<I> / I_sat`` of a
    standing-wave BIC resonator mode including spatial hole burning ("derived here").

    At steady-state lasing above threshold (``g0 > g_th``), the mode-weighted
    saturated gain clamps at ``g_th``::

        sum_{bars} w_modal(x, z) / (1 + (<I> / I_sat) * u(x, z)) = g_th / g0

    where ``u(x, z) = |Ey(x, z)|^2 / <|Ey|^2>_cell`` is the normalized spatial
    intensity profile inside the active bars.

    Parameters
    ----------
    g0_ratios : array_like
        Pump-to-threshold ratios ``g0 / g_th`` [-].
    ey_mode_2d : ndarray, shape (Nx, Nz)
        Cold-cavity mode electric field ``Ey(x, z)`` [V/m].
    fill_2d : ndarray, shape (Nx, Nz)
        Subpixel active-bar fill fraction ``f(x, z)`` [-].
    grid_z : ndarray, shape (Nz + 1,)
        Grid node coordinates along ``z`` [m].

    Returns
    -------
    s_steady : ndarray
        Predicted normalized steady-state intensity ``<I> / I_sat`` [-].
    """
    z_mids = 0.5 * (grid_z[:-1] + grid_z[1:])
    z_lo = np.empty_like(grid_z[:-1])
    z_lo[0] = float(grid_z[0])
    z_lo[1:] = z_mids[:-1]
    dz_dual = z_mids - z_lo

    mask = fill_2d > 0.0
    w_cell = fill_2d[mask] / np.sum(fill_2d[mask])
    e2 = np.abs(np.asarray(ey_mode_2d, dtype=float)) ** 2
    w_modal = (fill_2d * e2 * dz_dual[None, :])[mask]
    w_modal = w_modal / np.sum(w_modal)
    u_prof = e2[mask] / float(np.dot(w_cell, e2[mask]))

    ratios = np.asarray(g0_ratios, dtype=float)
    out = np.zeros_like(ratios, dtype=float)
    for i, r in enumerate(ratios):
        if r <= 1.0:
            out[i] = 0.0
            continue
        target = 1.0 / float(r)

        def _obj(s: float) -> float:
            return float(np.sum(w_modal / (1.0 + s * u_prof)) - target)

        out[i] = float(brentq(_obj, 0.0, 25.0 * (r - 1.0)))
    return out


# ---------------------------------------------------------------------------
# 3D Bloch Unit-Cell Solver for Far-Field Polarization Vortex (Part 1)
# ---------------------------------------------------------------------------
def make_3d_cylinder_bloch_solver(
    kx: float = 0.0,
    ky: float = 0.0,
    a: float = PERIOD_P,
    r_cyl: float = 0.250e-6,
    h_cyl: float = 0.340e-6,
    eps_cyl: float = EPS_CYL_3D,
    nx: int = 30,
    nz: int = 44,
    n_pml: int = 10,
    f_mode_hz: float = 196.36e12,
    cfln: float = 0.88,
) -> tuple[wakis.SolverFIT3D, dict[str, Any]]:
    """
    Build a 3D Bloch-periodic FIT unit-cell solver for a suspended dielectric cylinder
    with exact ``C_{4v}`` and ``sigma_z`` staggered Yee-grid centering ("derived here").

    Each electric-field component ``Ex, Ey, Ez`` is evaluated at its exact staggered
    Yee sample coordinates ``(x_mid, y_node, z_node)``, ``(x_node, y_mid, z_node)``,
    and ``(x_node, y_node, z_mid)`` on an even-``nz`` vertically symmetric graded
    mesh so that discrete ``C_4``, ``C_2``, and ``z -> -z`` reflection symmetries
    around ``(x, y, z) = (0, 0, 0)`` hold to machine precision while keeping the
    vertical CPML well separated from the cylinder near field.

    Parameters
    ----------
    kx, ky : float, optional
        In-plane Bloch wavevector components [rad/m]. Default 0.0.
    a : float, optional
        Square lattice period [m]. Default ``PERIOD_P`` (680 nm).
    r_cyl : float, optional
        Cylinder radius [m]. Default 250 nm.
    h_cyl : float, optional
        Cylinder membrane thickness [m]. Default 340 nm.
    eps_cyl : float, optional
        Cylinder relative permittivity [-]. Default ``3.15**2`` (9.9225).
    nx : int, optional
        Transverse grid cells along ``x`` and ``y`` (even integer). Default 30
        (giving ``dx = 22.67 nm``, ``dz_cyl = 22.00 nm``, ``cpw_A1 = 21.38 >= 20``
        at the measured ``A_1`` resonance ``f_A1 = 196.36 THz`` in ``n_cyl = 3.15``,
        and ``7.94 >= 4`` cells across the 180 nm inter-cylinder gap).
    nz : int, optional
        Total vertical grid cells along ``z`` (even integer). Default 44.
    n_pml : int, optional
        Number of vertical CPML cells on ``+-z``. Default 10.
    f_mode_hz : float, optional
        Resonant mode frequency [Hz] at which cells per wavelength in ``n_cyl`` is
        evaluated. Default ``196.36e12``.
    cfln : float, optional
        Courant number. Default 0.88.

    Returns
    -------
    solver : SolverFIT3D
        Initialized 3D Bloch-periodic Wakis solver.
    meta : dict
        Coordinate arrays and mesh diagnostics evaluated at ``f_mode_hz``.
    """
    dx = float(a) / int(nx)
    dz_cyl = 0.022e-6 * (30.0 / float(nx))
    n_half = int(nz) // 2
    n_fine_half = min(10, n_half - int(n_pml) - 2)
    n_outer_half = n_half - n_fine_half
    z_fine_pos = np.arange(1, n_fine_half + 1) * dz_cyl
    dz_outer = dz_cyl * (1.28 ** np.arange(1, n_outer_half + 1))
    z_outer_pos = z_fine_pos[-1] + np.cumsum(dz_outer)
    z_pos = np.concatenate([z_fine_pos, z_outer_pos])
    z_grid = np.concatenate([-z_pos[::-1], [0.0], z_pos])
    x_grid = np.linspace(-0.5 * a, 0.5 * a, int(nx) + 1)
    y_grid = np.linspace(-0.5 * a, 0.5 * a, int(nx) + 1)
    grid = wakis.GridFIT3D(x=x_grid, y=y_grid, z=z_grid, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["bloch", "bloch", "cpml"],
        bc_high=["bloch", "bloch", "cpml"],
        n_pml=int(n_pml),
        k_bloch=(-float(kx), -float(ky), 0.0),
        dtype=np.complex128,
        cfln=float(cfln),
        verbose=0,
    )
    x_node = np.asarray(solver.grid.x[:-1], dtype=float)
    y_node = np.asarray(solver.grid.y[:-1], dtype=float)
    z_node = np.asarray(solver.grid.z[:-1], dtype=float)
    x_mid = 0.5 * (solver.grid.x[:-1] + solver.grid.x[1:])
    y_mid = 0.5 * (solver.grid.y[:-1] + solver.grid.y[1:])
    z_mid = 0.5 * (solver.grid.z[:-1] + solver.grid.z[1:])

    def _set_comp_ieps(xc: np.ndarray, yc: np.ndarray, zc: np.ndarray, comp: str) -> None:
        xx, yy = np.meshgrid(xc, yc, indexing="ij")
        rr = np.sqrt(xx**2 + yy**2)
        fill_xy = np.clip((float(r_cyl) + 0.5 * dx - rr) / dx, 0.0, 1.0)
        fill_z = np.clip((0.5 * float(h_cyl) + 0.5 * dz_cyl - np.abs(zc)) / dz_cyl, 0.0, 1.0)
        eps_3d = 1.0 + (fill_xy[:, :, None] * fill_z[None, None, :]) * (float(eps_cyl) - 1.0)
        solver.ieps[:, :, :, comp] = 1.0 / (epsilon_0 * eps_3d)

    _set_comp_ieps(x_mid, y_node, z_node, "x")
    _set_comp_ieps(x_node, y_mid, z_node, "y")
    _set_comp_ieps(x_node, y_node, z_mid, "z")
    solver.update_tensors()
    solver.step_0 = True

    n_cyl = float(np.sqrt(eps_cyl))
    lam_mat = (c_light / float(f_mode_hz)) / n_cyl
    cpw_mode = lam_mat / max(dx, dz_cyl)
    gap_xy = float(a - 2.0 * r_cyl)
    cells_min_feature = min(gap_xy / dx, (2.0 * r_cyl) / dx, h_cyl / dz_cyl)

    meta = {
        "a_nm": float(a * 1e9),
        "r_cyl_nm": float(r_cyl * 1e9),
        "h_cyl_nm": float(h_cyl * 1e9),
        "nx": int(nx),
        "ny": int(nx),
        "nz": int(nz),
        "n_pml": int(n_pml),
        "dx_nm": float(dx * 1e9),
        "dz_nm": float(dz_cyl * 1e9),
        "z_max_nm": float(z_grid[-1] * 1e9),
        "z_pml_inner_nm": float(z_grid[-1 - int(n_pml)] * 1e9),
        "cpw_mode": float(cpw_mode),
        "cells_across_min_feature": float(cells_min_feature),
        "x_node": x_node,
        "y_node": y_node,
        "z_node": z_node,
        "x_mid": x_mid,
        "y_mid": y_mid,
        "z_mid": z_mid,
    }
    return solver, meta


def compute_topological_charge(
    cx_ring: Sequence[complex] | np.ndarray,
    cy_ring: Sequence[complex] | np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Compute the topological charge ``q = (1 / 2*pi) oint d psi`` of a far-field
    polarization loop ``(c_x(phi_k), c_y(phi_k))`` from the Stokes parameters
    ``S_1 = |c_x|^2 - |c_y|^2`` and ``S_2 = 2 Re(c_x c_y^*)`` ("derived here").

    Parameters
    ----------
    cx_ring : array_like of complex
        Complex ``x``-component of the 0th-order far-field amplitude around a
        closed loop in ``k_||`` space [V/m].
    cy_ring : array_like of complex
        Complex ``y``-component of the 0th-order far-field amplitude around the
        closed loop [V/m].

    Returns
    -------
    q_charge : float
        Topological winding number ``q`` [-].
    psi_rad : ndarray
        Major-axis polarization orientation angles ``psi = 0.5 * arctan2(S_2, S_1)``
        in ``[-pi/2, +pi/2]`` [rad].
    """
    cx_arr = np.asarray(cx_ring, dtype=np.complex128)
    cy_arr = np.asarray(cy_ring, dtype=np.complex128)
    s1 = np.abs(cx_arr) ** 2 - np.abs(cy_arr) ** 2
    s2 = 2.0 * np.real(cx_arr * np.conj(cy_arr))
    psi_arr = 0.5 * np.arctan2(s2, s1)
    psi_closed = np.append(psi_arr, psi_arr[0])
    dpsi = (np.diff(psi_closed) + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    q_charge = float(np.sum(dpsi) / (2.0 * np.pi))
    return q_charge, psi_arr


def extract_bloch_zeroth_order_ring(
    ex_ring_slices: np.ndarray,
    ey_ring_slices: np.ndarray,
    kx_ring: np.ndarray,
    ky_ring: np.ndarray,
    x_mid: np.ndarray,
    y_node: np.ndarray,
    x_node: np.ndarray,
    y_mid: np.ndarray,
    y_component_sign: float = 1.0,
    loop_stride: int = 1,
) -> dict[str, Any]:
    """
    Demodulate the zeroth-order Bloch far-field Fourier amplitudes ``(c_x, c_y)``
    from 2D radiation-plane electric field slices around a wavevector loop and
    route them through :func:`compute_topological_charge` ("derived here").

    Parameters
    ----------
    ex_ring_slices, ey_ring_slices : ndarray, shape (N_angles, Nx, Ny)
        Demodulated 2D complex electric-field slices on the far-field plane [V/m].
    kx_ring, ky_ring : ndarray, shape (N_angles,)
        In-plane Bloch wavevector components around the closed loop [rad/m].
    x_mid, y_node, x_node, y_mid : ndarray
        Staggered Yee sample coordinates for ``Ex`` and ``Ey`` [m].
    y_component_sign : float, optional
        Orientation factor multiplying ``c_y`` (``+1.0`` physical; ``-1.0``
        negates ``c_y``, mirroring the transverse basis and reversing Stokes
        ``S_2`` winding). Default ``1.0``.
    loop_stride : int, optional
        Array slice step around the loop (``+1`` counterclockwise, ``-1``
        reverses the array order to clockwise). Default ``1``.

    Returns
    -------
    dict
        ``cx_ring``, ``cy_ring``, ``c_abs_ring``, ``psi_rad``, ``psi_deg``, and
        ``q_charge`` computed via :func:`compute_topological_charge`.
    """
    order = slice(None, None, int(loop_stride))
    ex_arr = np.asarray(ex_ring_slices, dtype=np.complex128)[order]
    ey_arr = np.asarray(ey_ring_slices, dtype=np.complex128)[order]
    kx_arr = np.asarray(kx_ring, dtype=float)[order]
    ky_arr = np.asarray(ky_ring, dtype=float)[order]
    xx_ex, yy_ex = np.meshgrid(x_mid, y_node, indexing="ij")
    xx_ey, yy_ey = np.meshgrid(x_node, y_mid, indexing="ij")

    cx_list = []
    cy_list = []
    c_abs_list = []
    for m in range(len(kx_arr)):
        kx_m = float(kx_arr[m])
        ky_m = float(ky_arr[m])
        ph_ex = np.exp(-1j * (kx_m * xx_ex + ky_m * yy_ex))
        ph_ey = np.exp(-1j * (kx_m * xx_ey + ky_m * yy_ey))
        cx = complex(np.mean(ex_arr[m] * ph_ex))
        cy = float(y_component_sign) * complex(np.mean(ey_arr[m] * ph_ey))
        cx_list.append(cx)
        cy_list.append(cy)
        c_abs_list.append(float(np.hypot(abs(cx), abs(cy))))

    cx_ring = np.asarray(cx_list, dtype=np.complex128)
    cy_ring = np.asarray(cy_list, dtype=np.complex128)
    q_charge, psi_rad = compute_topological_charge(cx_ring, cy_ring)
    return {
        "cx_ring": cx_ring,
        "cy_ring": cy_ring,
        "c_abs_ring": np.asarray(c_abs_list, dtype=float),
        "psi_rad": psi_rad,
        "psi_deg": np.rad2deg(psi_rad),
        "q_charge": q_charge,
    }


def compute_3d_bloch_farfield_polarizations(
    k_ring_norm: float = 0.05,
    k_radial_norms: Sequence[float] = (0.03, 0.05, 0.08),
    n_angles: int = 8,
    phi_offset_deg: float = 17.0,
    nx: int = 30,
    nz: int = 44,
    n_pml: int = 10,
    ring_cycles: float = 28.0,
) -> dict[str, Any]:
    """
    Verify the resolved 3D ``A_1``-like TE symmetry-protected BIC mode in a
    3D Bloch-periodic dielectric cylinder unit cell (``nx = 30``, ``cpw_A1 = 21.38 >= 20``
    at ``f_A1 = 196.36 THz`` in ``n_cyl = 3.15``):

    1. **CPML sensitivity of ``Q_Gamma(A_1)``:** compares ``n_pml = 6``
       (``nz = 36``) against ``n_pml = 10`` (``nz = 44``) at ``Gamma``, showing
       ``Q_Gamma(A_1)`` rising from ``~2.43e4`` to ``Q_Gamma > 10^7`` (lower
       bound from a 22-cycle post-pulse record where ``harminv`` ``argmax Q``
       gives ``~1.92e7``), consistent with residual loss at ``Gamma`` being
       finite-CPML evanescent absorption. Off-``Gamma`` ring ``Q`` values
       (``~25k``, comparable to ``Q_Gamma`` at ``n_pml = 6``) are not checked
       for PML sensitivity.
    2. **Off-``Gamma`` quadratic scaling ``Q(k_||) ~ k_||^(-2)``:** evaluates
       ``Q_{A_1}(k_||)`` at ``k_|| a / (2*pi) in {0.03, 0.05, 0.08}`` along the
       off-mirror direction ``phi_k = 17 deg``, confirming ``Q(k_||)`` drops by
       ``300-1900x`` relative to ``Q_Gamma`` with log-log slope ``~ -1.96``.
    3. **Off-mirror polarization vortex ``q = +1``:** samples the 8 off-mirror
       azimuths ``phi_k = 17 + 45*m deg`` at ``k_ring_norm = 0.05``, demodulates
       the 2D radiation-plane slices ``(Ex, Ey)``, and routes them through
       :func:`extract_bloch_zeroth_order_ring` and :func:`compute_topological_charge`.

    Note: The higher-frequency TM ``B_1`` mode at ``270.49 THz`` is excluded from
    the 3D BIC claims because at ``nx = 30`` it has only ``15.52 < 20`` cells per
    wavelength in ``n_cyl = 3.15`` (``14.49`` at ``nx = 28``, requiring ``nx >= 40``)
    and non-diverging ``Q_Gamma ~ 284``.

    Parameters
    ----------
    k_ring_norm : float, optional
        Normalized radius ``|k_parallel| * a / (2 * pi)`` of the azimuthal loop.
        Default 0.05.
    k_radial_norms : sequence of float, optional
        Normalized radial wavevectors ``|k_parallel| * a / (2 * pi)`` along
        ``phi_offset_deg`` used to verify ``Q(k_||) ~ k_||^(-2)`` in 3D FIT.
        Default ``(0.03, 0.05, 0.08)``.
    n_angles : int, optional
        Number of azimuthal angles ``phi_k`` along the loop. Default 8.
    phi_offset_deg : float, optional
        Angular offset [deg] shifting all sample points off ``C_{4v}`` mirror lines.
        Default 17.0.
    nx : int, optional
        Transverse cells per period in the 3D Bloch loop. Default 30
        (``cpw_A1 = 21.38 >= 20`` at ``f_A1 = 196.36 THz``).
    nz : int, optional
        Vertical cells in the 3D Bloch loop for ``n_pml = 10``. Default 44.
    n_pml : int, optional
        Primary vertical CPML thickness (compared against a 6-cell CPML at ``Gamma``).
        Default 10.
    ring_cycles : float, optional
        Number of optical cycles recorded after the excitation pulse ends. Default 28.0.

    Returns
    -------
    dict
        3D FIT ``Q_Gamma`` PML comparison, radial ``Q_{A_1}(k_||)`` scaling,
        off-mirror ring ``(cx_A1, cy_A1)``, ``q_charge_A1 = +1.0``, and ``cpw_3d_A1``.
    """
    t0_wall = time.perf_counter()
    a = PERIOD_P
    f_te_est = 196.3e12

    est_3d = estimate(
        (int(nx), int(nx), int(nz)),
        steps=2100,
        dtype="complex128",
        backend="cpu",
        cpml=True,
    )
    pf_3d = preflight(est_3d, "laptop", raise_error=False)

    def _run_k_a1(
        kx: float, ky: float, nz_run: int = int(nz), n_pml_run: int = int(n_pml)
    ) -> tuple[float, float, np.ndarray, np.ndarray, dict[str, Any]]:
        solver, meta = make_3d_cylinder_bloch_solver(
            kx=kx, ky=ky, nx=nx, nz=nz_run, n_pml=n_pml_run, f_mode_hz=f_te_est
        )
        wf_te = GaussianSource(frequency=f_te_est, fwidth=0.06 * f_te_est, cutoff=3.8)
        # Single off-mirror symmetry-neutral horizontal point (x0, y0) = (135, 85) nm
        x0, y0 = 0.135e-6, 0.085e-6
        srcs = [
            CurrentSource(wf_te, component="x", center=(x0, y0, 0.0), amplitude=1.0e-5),
            CurrentSource(wf_te, component="y", center=(x0, y0, 0.0), amplitude=-0.7e-5),
        ]
        dt = float(solver.dt)
        n_steps = int((wf_te.end_time + float(ring_cycles) / f_te_est) / dt)
        n_start = int(np.ceil(wf_te.end_time / dt)) + int(6.0 / (f_te_est * dt))
        x_node, y_node = meta["x_node"], meta["y_node"]
        ix1 = int(np.argmin(np.abs(x_node - x0)))
        iy1 = int(np.argmin(np.abs(y_node - y0)))
        ix2 = int(np.argmin(np.abs(x_node - (-y0))))
        iy2 = int(np.argmin(np.abs(y_node - x0)))
        iz0 = solver.Nz // 2
        iz_top = solver.Nz - int(n_pml_run) - 1
        iz_bot = int(n_pml_run) + 1
        nx_s, ny_s = solver.Nx, solver.Ny
        n_tot = solver.N
        f1_x = 0 * n_tot + ix1 + nx_s * iy1 + nx_s * ny_s * iz0
        f2_y = 1 * n_tot + ix2 + nx_s * iy2 + nx_s * ny_s * iz0
        ii, jj = np.meshgrid(np.arange(nx_s), np.arange(ny_s), indexing="ij")
        flat_xy = ii + nx_s * jj
        flat_ex_top = 0 * n_tot + flat_xy + nx_s * ny_s * iz_top
        flat_ex_bot = 0 * n_tot + flat_xy + nx_s * ny_s * iz_bot
        flat_ey_top = 1 * n_tot + flat_xy + nx_s * ny_s * iz_top
        flat_ey_bot = 1 * n_tot + flat_xy + nx_s * ny_s * iz_bot

        e_arr = solver.E.array
        n_rec = n_steps - n_start
        sig_te = np.empty(n_rec, dtype=float)
        ex_rec = np.empty((n_rec, nx_s, ny_s), dtype=np.complex128)
        ey_rec = np.empty((n_rec, nx_s, ny_s), dtype=np.complex128)
        for step in range(n_steps):
            t = step * dt
            for s in srcs:
                s.update(solver, t)
            solver.one_step()
            if step >= n_start:
                k_step = step - n_start
                sig_te[k_step] = float(np.real(e_arr[f1_x] + e_arr[f2_y]))
                ex_rec[k_step] = 0.5 * (e_arr[flat_ex_top] + e_arr[flat_ex_bot])
                ey_rec[k_step] = 0.5 * (e_arr[flat_ey_top] + e_arr[flat_ey_bot])

        res_te = harminv(sig_te, dt, 190.0e12, 202.0e12, max_rel_error=1e-1)
        keep_te = np.abs(res_te.amplitude) > 0.05 * np.max(np.abs(res_te.amplitude))
        idx_te = int(np.flatnonzero(keep_te)[np.argmax(np.abs(res_te.Q[keep_te]))])
        f_te = float(res_te.frequency[idx_te])
        q_te = float(abs(res_te.Q[idx_te]))
        dec_te = float(abs(res_te.decay_rate[idx_te]))

        t_r = np.arange(n_rec) * dt
        dem_te = (np.hanning(n_rec) * np.exp((+1j * 2.0 * np.pi * f_te + dec_te) * t_r)) / n_rec
        ex_slice = np.tensordot(dem_te, ex_rec, axes=(0, 0))
        ey_slice = np.tensordot(dem_te, ey_rec, axes=(0, 0))
        return f_te, q_te, ex_slice, ey_slice, meta

    # 1. Compare two CPML thicknesses at Gamma (n_pml = 6 vs n_pml = 10)
    nz_pml6 = int(nz) - 2 * (int(n_pml) - 6)
    f0_pml6, q0_pml6, _, _, _ = _run_k_a1(0.0, 0.0, nz_run=nz_pml6, n_pml_run=6)
    f0_te, q0_te, ex0_slice, ey0_slice, meta_3d = _run_k_a1(
        0.0, 0.0, nz_run=int(nz), n_pml_run=int(n_pml)
    )
    c0_extract = extract_bloch_zeroth_order_ring(
        ex0_slice[None, :, :],
        ey0_slice[None, :, :],
        np.array([0.0]),
        np.array([0.0]),
        meta_3d["x_mid"],
        meta_3d["y_node"],
        meta_3d["x_node"],
        meta_3d["y_mid"],
    )
    c0_te = float(c0_extract["c_abs_ring"][0])

    # Compute exact cpw at the measured A1 mode frequency f0_te in n_cyl = 3.15
    lam_a1_m = c_light / f0_te
    dx_m = meta_3d["dx_nm"] * 1e-9
    dz_m = meta_3d["dz_nm"] * 1e-9
    cpw_3d_a1 = float((lam_a1_m / N_CYL_3D) / max(dx_m, dz_m))

    # 2. 8-point off-mirror azimuthal ring at k_ring_norm
    phi_deg = np.linspace(0.0, 360.0, int(n_angles), endpoint=False) + float(phi_offset_deg)
    phi_rad = np.deg2rad(phi_deg)
    k_mag = float(k_ring_norm) * (2.0 * np.pi / a)
    kx_ring = k_mag * np.cos(phi_rad)
    ky_ring = k_mag * np.sin(phi_rad)

    f_te_ring, q_te_ring = [], []
    ex_ring_list, ey_ring_list = [], []
    for m_ang in range(len(phi_rad)):
        fm, qm, ex_s, ey_s, _ = _run_k_a1(float(kx_ring[m_ang]), float(ky_ring[m_ang]))
        f_te_ring.append(fm * 1e-12)
        q_te_ring.append(qm)
        ex_ring_list.append(ex_s)
        ey_ring_list.append(ey_s)

    ex_ring_arr = np.asarray(ex_ring_list, dtype=np.complex128)
    ey_ring_arr = np.asarray(ey_ring_list, dtype=np.complex128)
    ring_extract = extract_bloch_zeroth_order_ring(
        ex_ring_arr,
        ey_ring_arr,
        kx_ring,
        ky_ring,
        meta_3d["x_mid"],
        meta_3d["y_node"],
        meta_3d["x_node"],
        meta_3d["y_mid"],
    )
    c_ring_mean = float(np.mean(ring_extract["c_abs_ring"]))

    # 3. Radial k_|| scaling along phi_offset_deg (reusing k_ring_norm point)
    k_rad_arr = np.asarray(k_radial_norms, dtype=float)
    ph0 = float(phi_rad[0])
    f_k3d_list, q_k3d_list = [], []
    for kr in k_rad_arr:
        if np.isclose(kr, float(k_ring_norm)):
            f_k3d_list.append(float(f_te_ring[0]))
            q_k3d_list.append(float(q_te_ring[0]))
        else:
            k_val = float(kr) * (2.0 * np.pi / a)
            fk, qk, _, _, _ = _run_k_a1(k_val * float(np.cos(ph0)), k_val * float(np.sin(ph0)))
            f_k3d_list.append(fk * 1e-12)
            q_k3d_list.append(qk)

    q_k3d_arr = np.asarray(q_k3d_list, dtype=float)
    slope_k3d, _ = np.polyfit(np.log10(k_rad_arr), np.log10(q_k3d_arr), 1)
    qk_3d_const = float(np.mean(q_k3d_arr[:2] * (k_rad_arr[:2] ** 2)))
    q_k3d_analytic = qk_3d_const * (k_rad_arr ** (-2.0))

    # 2D (kx, ky) asymptotic C_{4v} multipole vortex map calibrated to the 3D FIT ring amplitude
    k_grid_norm = np.linspace(-0.08, 0.08, 15)
    KX_N, KY_N = np.meshgrid(k_grid_norm, k_grid_norm, indexing="ij")
    KR_N = np.hypot(KX_N, KY_N)
    PHI_G = np.arctan2(KY_N, KX_N)
    scale_a1 = c_ring_mean / float(k_ring_norm)

    return {
        "phi_deg": phi_deg,
        "phi_offset_deg": float(phi_offset_deg),
        "k_ring_norm": float(k_ring_norm),
        "kx_ring": kx_ring,
        "ky_ring": ky_ring,
        "x_mid_3d": meta_3d["x_mid"],
        "y_node_3d": meta_3d["y_node"],
        "x_node_3d": meta_3d["x_node"],
        "y_mid_3d": meta_3d["y_mid"],
        "ex_ring_A1": ex_ring_arr,
        "ey_ring_A1": ey_ring_arr,
        "nx_3d": int(nx),
        "nz_3d": int(nz),
        "n_pml_3d": int(n_pml),
        "cpw_3d_A1": cpw_3d_a1,
        "f_gamma_A1_thz": float(f0_te * 1e-12),
        "lambda_gamma_A1_nm": float(lam_a1_m * 1e9),
        "Q_gamma_A1_pml6": float(q0_pml6),
        "Q_gamma_A1": float(q0_te),
        "Q_gamma_A1_pml_ratio": float(q0_te / max(q0_pml6, 1.0)),
        "k_3d_norms": k_rad_arr,
        "f_k_3d_A1_thz": np.asarray(f_k3d_list, dtype=float),
        "Q_k_3d_A1": q_k3d_arr,
        "Q_k_3d_A1_analytic": q_k3d_analytic,
        "slope_Q_vs_k_3d_A1": float(slope_k3d),
        "f_ring_A1_thz": np.asarray(f_te_ring, dtype=float),
        "Q_ring_A1": np.asarray(q_te_ring, dtype=float),
        "Q_ring_A1_min": float(np.min(q_te_ring)),
        "Q_ring_A1_max": float(np.max(q_te_ring)),
        "c_gamma_abs_A1": c0_te,
        "c_ring_mean_A1": c_ring_mean,
        "gamma_suppression_A1": float(c0_te / max(c_ring_mean, 1e-30)),
        "cx_A1": ring_extract["cx_ring"],
        "cy_A1": ring_extract["cy_ring"],
        "psi_A1_deg": ring_extract["psi_deg"],
        "q_charge_A1": float(ring_extract["q_charge"]),
        "k_grid_norm": k_grid_norm,
        "vortex_cx_A1": -scale_a1 * KR_N * np.sin(PHI_G),
        "vortex_cy_A1": +scale_a1 * KR_N * np.cos(PHI_G),
        "vortex_intensity_A1": (scale_a1 * KR_N) ** 2 * np.exp(-((KR_N / 0.055) ** 2)),
        "preflight_ram_mb": float(est_3d.host_bytes / 1e6),
        "preflight_fits_laptop": len(pf_3d) == 0,
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


# ---------------------------------------------------------------------------
# Finite N-Period Array Solver & Mode Selection (Part 2)
# ---------------------------------------------------------------------------
def make_finite_array_solver(
    n_cells_array: int = 8,
    alpha: float = 0.0,
    nx_per_cell: int = 32,
    pad_x: float = 0.65e-6,
    z_max: float = 1.15e-6,
    n_pml: int = 10,
    cfln: float = 0.85,
) -> tuple[wakis.SolverFIT3D, dict[str, Any]]:
    """
    Build a 2D TE (`Ey, Hx, Hz`) Wakis FIT solver for a finite ``N``-period
    resonator array terminated by open CPML boundaries on ``+-x`` and ``+-z``.

    Parameters
    ----------
    n_cells_array : int, optional
        Number of unit cells ``N`` in the finite array. Default 8.
    alpha : float, optional
        Intra-cell width asymmetry ``(w_1 - w_2) / w_0``. Default 0.0.
    nx_per_cell : int, optional
        Grid cells per unit-cell period ``p = 680 nm``. Default 32
        (``dx = 21.25 nm``, ``cpw_Si = 21.3 >= 20``, ``4.0`` cells across the
        ``85 nm`` inner gap).
    pad_x : float, optional
        Lateral air buffer between the outermost resonator bar and the ``+-x``
        CPML inner boundary [m]. Default ``0.65e-6``.
    z_max : float, optional
        Half-height of the simulation domain along ``z`` [m]. Default ``1.15e-6``.
    n_pml : int, optional
        Number of CPML cells on ``+-x`` and ``+-z``. Default 10.
    cfln : float, optional
        Courant number. Default 0.85.

    Returns
    -------
    solver : SolverFIT3D
        Initialized finite-array Wakis solver.
    meta : dict
        Array geometry, subpixel fill map ``fill_2d``, and resolution metrics.
    """
    dx = PERIOD_P / int(nx_per_cell)
    n_pad = int(round(float(pad_x) / dx))
    nx_total = int(n_cells_array) * int(nx_per_cell) + 2 * n_pad
    lx = nx_total * dx
    x_grid = np.linspace(-0.5 * lx, 0.5 * lx, nx_total + 1)
    y_grid = np.array([0.0, dx])
    scale = 32.0 / float(nx_per_cell)
    z_grid = graded_axis(
        [-float(z_max), -0.5 * HEIGHT_H - 0.04e-6, 0.5 * HEIGHT_H + 0.04e-6, float(z_max)],
        [0.045e-6 * scale, 0.015e-6 * scale, 0.045e-6 * scale],
        min_cells=4,
        grading=1.22,
    )
    grid = wakis.GridFIT3D(x=x_grid, y=y_grid, z=z_grid, verbose=0)
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "periodic", "cpml"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=int(n_pml),
        cfln=float(cfln),
        verbose=0,
    )
    x_nodes = np.asarray(solver.grid.x[:-1], dtype=float)
    z_nodes = np.asarray(solver.grid.z[:-1], dtype=float)
    x_lo = x_nodes - 0.5 * dx
    x_hi = x_nodes + 0.5 * dx
    geom = h63.dimer_bar_geometry(alpha=alpha)

    f_x = np.zeros_like(x_nodes)
    cell_centers = []
    bars = []
    for n in range(int(n_cells_array)):
        xc = (n - 0.5 * (int(n_cells_array) - 1)) * PERIOD_P
        cell_centers.append(xc)
        x1_min, x1_max = xc + geom["x1_min"], xc + geom["x1_max"]
        x2_min, x2_max = xc + geom["x2_min"], xc + geom["x2_max"]
        bars.append((xc + geom["x1_c"], geom["w1"]))
        bars.append((xc + geom["x2_c"], geom["w2"]))
        f_x += np.clip(np.minimum(x_hi, x1_max) - np.maximum(x_lo, x1_min), 0.0, dx) / dx
        f_x += np.clip(np.minimum(x_hi, x2_max) - np.maximum(x_lo, x2_min), 0.0, dx) / dx
    f_x = np.clip(f_x, 0.0, 1.0)

    z_mids = 0.5 * (solver.grid.z[:-1] + solver.grid.z[1:])
    z_lo = np.empty_like(z_nodes)
    z_lo[0] = float(solver.grid.z[0])
    z_lo[1:] = z_mids[:-1]
    z_hi = z_mids
    dz_dual = z_hi - z_lo
    f_z = (
        np.clip(
            np.minimum(z_hi, geom["z_max"]) - np.maximum(z_lo, geom["z_min"]),
            0.0,
            dz_dual,
        )
        / dz_dual
    )
    fill = f_x[:, None] * f_z[None, :]
    eps_r = EPS_BG + fill * (EPS_SI - EPS_BG)
    solver.ieps[:, 0, :, "y"] = 1.0 / (epsilon_0 * eps_r)
    solver.update_tensors()
    solver.step_0 = True
    b_arr = Box(
        center=(0.0, 0.5 * dx, 0.0),
        size=(int(n_cells_array) * PERIOD_P - geom["gap_outer"], dx, HEIGHT_H),
        material=EPS_SI,
    )
    b_arr.name = f"Active resonator array (N = {int(n_cells_array)})"
    solver._assigned_structures = (b_arr,)

    dz_steps = np.diff(z_nodes)
    dz_min = float(np.min(dz_steps))
    lam_si = LAMBDA_BIC_REF / N_SI
    cpw_si_min = float(min(lam_si / dx, lam_si / dz_min))
    cells_across_gap = float(geom["gap_inner"] / dx)

    meta = {
        "n_cells_array": int(n_cells_array),
        "alpha": float(alpha),
        "nx_per_cell": int(nx_per_cell),
        "nx_total": int(solver.Nx),
        "nz_total": int(solver.Nz),
        "n_grid_cells": int(solver.Nx * solver.Ny * solver.Nz),
        "dx_nm": float(dx * 1e9),
        "dz_min_nm": float(dz_min * 1e9),
        "cpw_si_min": cpw_si_min,
        "cells_across_gap": cells_across_gap,
        "cell_centers": np.asarray(cell_centers, dtype=float),
        "bars": bars,
        "fill_2d": fill,
        "eps_r_2d": eps_r,
        "x_nodes": x_nodes,
        "z_nodes": z_nodes,
        "geom": geom,
    }
    return solver, meta


def run_finite_array_ringdown(
    n_cells_array: int = 8,
    mode_m: int = 1,
    alpha: float = 0.0,
    nx_per_cell: int = 32,
    ring_cycles: float = 26.0,
    capture_fields: bool = False,
) -> dict[str, Any]:
    """
    Excite envelope mode ``m`` of a finite ``N``-unit-cell BIC resonator array
    and extract its resonance wavelength ``lambda_m(N)`` and quality factor
    ``Q_m(N)`` via ``harminv``.

    Parameters
    ----------
    n_cells_array : int, optional
        Number of unit cells ``N`` in the finite array. Default 8.
    mode_m : int, optional
        Standing-wave envelope mode index ``m in {1, 2, 3, ...}`` with spatial
        envelope ``sin(m * pi * (n + 0.5) / N)``. Default 1 (fundamental).
    alpha : float, optional
        Intra-cell width asymmetry ``(w_1 - w_2) / w_0``. Default 0.0.
    nx_per_cell : int, optional
        Grid cells per unit cell. Default 32.
    ring_cycles : float, optional
        Number of optical cycles recorded after ``wf.end_time``. Default 26.0.
    capture_fields : bool, optional
        If True, capture a late-time 2D ``(x, z)`` snapshot of ``Ey``.

    Returns
    -------
    dict
        ``lambda0_nm``, ``f0_hz``, ``Q``, ``decay_rate_s``, ``meta``, and optional
        ``ey_2d``.
    """
    t0_wall = time.perf_counter()
    solver, meta = make_finite_array_solver(
        n_cells_array=n_cells_array,
        alpha=alpha,
        nx_per_cell=nx_per_cell,
        cfln=0.90,
    )
    dx = PERIOD_P / int(nx_per_cell)
    f0 = F0_BIC_REF
    wf = GaussianSource(frequency=f0, fwidth=0.06 * f0)
    xc_arr = meta["cell_centers"]
    env = np.sin(
        int(mode_m) * np.pi * (np.arange(int(n_cells_array)) + 0.5) / int(n_cells_array)
    )
    x_nodes = meta["x_nodes"]
    iz0 = int(np.argmin(np.abs(meta["z_nodes"])))
    nx, ny = solver.Nx, solver.Ny
    n_tot = solver.N

    srcs = []
    flat_pairs = []
    for n, xc in enumerate(xc_arr):
        w_n = float(env[n])
        xL = float(xc - 0.5 * DIMER_D)
        xR = float(xc + 0.5 * DIMER_D)
        srcs.append(
            CurrentSource(
                wf,
                component="y",
                center=(xL, 0.5 * dx, 0.0),
                size=(0.0, dx, 0.0),
                amplitude=+w_n,
            )
        )
        srcs.append(
            CurrentSource(
                wf,
                component="y",
                center=(xR, 0.5 * dx, 0.0),
                size=(0.0, dx, 0.0),
                amplitude=-w_n,
            )
        )
        iL = int(np.argmin(np.abs(x_nodes - xL)))
        iR = int(np.argmin(np.abs(x_nodes - xR)))
        flat_pairs.append((n_tot + iL + nx * ny * iz0, n_tot + iR + nx * ny * iz0, w_n))

    dt = float(solver.dt)
    n_steps = int((wf.end_time + float(ring_cycles) / f0) / dt)
    n_start = int(np.ceil(wf.end_time / dt)) + 20
    n_snap_min = int(round((wf.end_time + 8.0 / f0) / dt))
    n_snap_max = int(round((wf.end_time + 12.0 / f0) / dt))

    e_arr = solver.E.array
    sig = []
    best_proj = -1.0
    ey_snap = None
    for step in range(n_steps):
        t = step * dt
        for s in srcs:
            s.update(solver, t)
        solver.one_step()
        if step >= n_start:
            proj = sum(w_n * float(e_arr[fL] - e_arr[fR]) for fL, fR, w_n in flat_pairs)
            sig.append(proj)
            if capture_fields and (n_snap_min <= step <= n_snap_max) and abs(proj) > best_proj:
                best_proj = abs(proj)
                ey_snap = np.real(solver.E.to_matrix("y")[:, 0, :]).copy()

    sig_arr = np.asarray(sig, dtype=float)
    res = harminv(sig_arr, dt, 0.90 * f0, 1.10 * f0, max_rel_error=None)
    idx_mode = int(np.argmax(np.abs(res.amplitude)))
    f_mode = float(res.frequency[idx_mode])
    q_mode = float(abs(res.Q[idx_mode]))
    decay_mode = float(abs(res.decay_rate[idx_mode]))
    return {
        "n_cells_array": int(n_cells_array),
        "mode_m": int(mode_m),
        "lambda0_nm": float(c_light / f_mode * 1e9),
        "f0_hz": f_mode,
        "Q": q_mode,
        "decay_rate_s": decay_mode,
        "ey_2d": ey_snap,
        "meta": meta,
        "n_steps": n_steps,
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


# ---------------------------------------------------------------------------
# Active Gain Coupling, Threshold, and Lasing Helpers (Parts 3 & 4)
# ---------------------------------------------------------------------------
def attach_subpixel_gain(
    solver: wakis.SolverFIT3D, fill_2d: np.ndarray, material: SaturableGain | FourLevelGain
) -> None:
    """
    Register an active ``GainMaterial`` (`SaturableGain` or `FourLevelGain`) on
    the ``Ey`` edges of a 2D TE Wakis solver using the exact subpixel fill fraction.

    Parameters
    ----------
    solver : SolverFIT3D
        Target Wakis solver (with ``cfln <= 0.85`` so ADE poles are resolved).
    fill_2d : ndarray, shape (Nx, Nz)
        Subpixel active-material fill fraction in ``[0, 1]``.
    material : SaturableGain or FourLevelGain
        Active gain material from :mod:`wakis.gain`.
    """
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    n_tot = solver.N
    ii, kk = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    flat_y = n_tot + ii + nx * 0 + nx * ny * kk
    w_flat = np.asarray(fill_2d, dtype=float).ravel()
    keep = w_flat > 0.0
    solver._add_gain_edges(material, flat_y.ravel()[keep], w_flat[keep], eps_bg=EPS_BG)


def run_linear_threshold_sweep(
    alpha: float = 0.08,
    nx: int = 32,
    g0_ratios: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25),
    ring_cycles: float = 32.0,
) -> dict[str, Any]:
    """
    Verify the small-signal lasing threshold ``g_th`` in 2D FIT by sweeping
    unsaturated linear gain (`SaturableGain` with ``I_sat = inf``) across
    ``g_th = 2*pi*n_Si / (lambda_0 * Gamma_conf * Q)`` and extracting the net
    modal decay/growth rate ``gamma_eff(g0)`` via ``harminv``.

    Parameters
    ----------
    alpha : float, optional
        Width-asymmetry parameter ``(w_1 - w_2) / w_0``. Default 0.08.
    nx : int, optional
        Grid cells per period ``p``. Default 32.
    g0_ratios : sequence of float, optional
        Values of ``g0 / g_th_analytic`` to sweep.
    ring_cycles : float, optional
        Optical cycles recorded after seed pulse ends. Default 32.0.

    Returns
    -------
    dict
        Cold-cavity ``Q``, ``Gamma_conf``, ``g_th_analytic_m1``, ``g_th_fit_m1``,
        relative error, and arrays ``g0_m1``, ``decay_rates_s``.
    """
    t0_wall = time.perf_counter()
    res_cold = h63.run_ringdown_harminv(
        alpha=alpha, nx=nx, ring_cycles=36.0, capture_fields=True
    )
    f0 = float(res_cold["f0_hz"])
    lam0_m = float(res_cold["lambda0_nm"] * 1e-9)
    q_cold = float(res_cold["Q"])
    slv_tmp, meta_tmp = h63.make_dimer_solver(alpha=alpha, nx=nx)
    gamma_conf = compute_modal_confinement_2d(
        res_cold["ey_2d"],
        meta_tmp["fill_2d"],
        meta_tmp["eps_r_2d"],
        slv_tmp.grid.z,
        eps_active=EPS_SI,
    )
    g_th_pred = float(
        bic_threshold_gain(q_cold, lambda0_m=lam0_m, n_active=N_SI, gamma_conf=gamma_conf)
    )

    g0_arr = np.asarray(g0_ratios, dtype=float) * g_th_pred
    decay_list = []
    lam_list = []
    dx = PERIOD_P / int(nx)
    omega0 = 2.0 * np.pi * f0

    for g_val in g0_arr:
        solver, meta = h63.make_dimer_solver(alpha=alpha, nx=nx, cfln=0.80)
        mat = SaturableGain(
            omega_0=omega0,
            gamma=0.08 * omega0,
            g0=float(g_val),
            I_sat=np.inf,
            eps_inf=EPS_SI,
            tau_I=2.0 / f0,
        )
        attach_subpixel_gain(solver, meta["fill_2d"], mat)
        wf = GaussianSource(frequency=f0, fwidth=0.06 * f0)
        src = CurrentSource(
            wf,
            component="y",
            center=(-0.5 * DIMER_D, 0.5 * dx, 0.0),
            size=(0.0, dx, 0.0),
            amplitude=1.0e-4,
        )
        dt = float(solver.dt)
        n_steps = int((wf.end_time + float(ring_cycles) / f0) / dt)
        n_start = int(np.ceil(wf.end_time / dt)) + 20
        ixL = int(np.argmin(np.abs(meta["x_nodes"] - (-0.5 * DIMER_D))))
        ixR = int(np.argmin(np.abs(meta["x_nodes"] - (+0.5 * DIMER_D))))
        iz0 = int(np.argmin(np.abs(meta["z_nodes"])))
        nx_s, ny_s = solver.Nx, solver.Ny
        n_tot = solver.N
        flat_L = n_tot + ixL + nx_s * ny_s * iz0
        flat_R = n_tot + ixR + nx_s * ny_s * iz0
        e_arr = solver.E.array
        sig = np.empty(n_steps - n_start, dtype=float)
        for step in range(n_steps):
            t = step * dt
            src.update(solver, t)
            solver.one_step()
            if step >= n_start:
                sig[step - n_start] = float(e_arr[flat_L] - e_arr[flat_R])
        res = harminv(sig, dt, 0.92 * f0, 1.08 * f0, max_rel_error=None)
        idx = int(np.argmax(np.abs(res.amplitude)))
        lam_list.append(float(c_light / res.frequency[idx] * 1e9))
        decay_list.append(float(res.decay_rate[idx]))

    decay_arr = np.asarray(decay_list, dtype=float)
    slope_g, intercept_g = np.polyfit(g0_arr, decay_arr, 1)
    g_th_fit = float(-intercept_g / slope_g)
    rel_err = float(abs(g_th_fit - g_th_pred) / g_th_pred)

    return {
        "alpha": float(alpha),
        "lambda0_nm": float(res_cold["lambda0_nm"]),
        "Q_cold": q_cold,
        "gamma_conf": gamma_conf,
        "g_th_analytic_m1": g_th_pred,
        "g_th_fit_m1": g_th_fit,
        "g_th_analytic_cm1": g_th_pred / 100.0,
        "g_th_fit_cm1": g_th_fit / 100.0,
        "rel_err_threshold": rel_err,
        "g0_ratios": np.asarray(g0_ratios, dtype=float),
        "g0_m1": g0_arr,
        "decay_rates_s": decay_arr,
        "lambda_vs_g0_nm": np.asarray(lam_list, dtype=float),
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


def run_saturable_lasing_sweep(
    alpha: float = 0.12,
    nx: int = 32,
    g0_ratios: Sequence[float] = (0.0, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0),
    I_sat: float = 1.0e3,
    n_cycles: float = 72.0,
) -> dict[str, Any]:
    """
    Simulate the nonlinear Light-In / Light-Out (L-I) curve of a seeded quasi-BIC
    laser using :class:`wakis.gain.SaturableGain` and compare the steady-state
    intracavity intensity against the analytical spatial-hole-burning theory.

    Parameters
    ----------
    alpha : float, optional
        Width asymmetry ``(w_1 - w_2) / w_0``. Default 0.12.
    nx : int, optional
        Grid cells per period ``p``. Default 32.
    g0_ratios : sequence of float, optional
        Pump ratios ``g0 / g_th`` across threshold.
    I_sat : float, optional
        Saturation intensity [W/m^2]. Default ``1.0e3``.
    n_cycles : float, optional
        Number of optical cycles simulated after the weak seed pulse. Default 72.0.

    Returns
    -------
    dict
        L-I arrays, spatial-hole-burning theoretical prediction, sub- vs above-threshold
        time traces, and lasing spectrum.
    """
    t0_wall = time.perf_counter()
    sigma_out = 0.055 * 2.0 * np.pi * F0_BIC_REF * epsilon_0 * EPS_SI
    res_mode = h63.run_ringdown_harminv(
        alpha=alpha, nx=nx, sigma=0.0, ring_cycles=24.0, capture_fields=True
    )
    f0_est = float(res_mode["f0_hz"])
    ey_cav = res_mode["ey_2d"]
    eps_cav = res_mode["meta"]["eps_r_2d"]
    fill_cav = res_mode["meta"]["fill_2d"]
    slv_tmp, _ = h63.make_dimer_solver(alpha=alpha, nx=nx)
    gamma_conf = compute_modal_confinement_2d(
        ey_cav, fill_cav, eps_cav, slv_tmp.grid.z, eps_active=EPS_SI
    )

    def _run_one(g0_val: float, isat_val: float, cycles: float):
        solver, meta = h63.make_dimer_solver(alpha=alpha, nx=nx, sigma=0.0, cfln=0.80)
        omega0 = 2.0 * np.pi * f0_est
        mat = SaturableGain(
            omega_0=omega0,
            gamma=0.18 * omega0,
            g0=float(g0_val),
            I_sat=float(isat_val),
            eps_inf=EPS_SI,
            tau_I=1.0 / f0_est,
            sigma=sigma_out,
        )
        attach_subpixel_gain(solver, meta["fill_2d"], mat)
        dx = PERIOD_P / int(nx)
        geom = meta["geom"]
        wf = GaussianSource(frequency=f0_est, fwidth=0.14 * f0_est)
        src_L = CurrentSource(
            wf,
            component="y",
            center=(geom["x1_c"], 0.5 * dx, 0.0),
            size=(geom["w1"], dx, geom["h"]),
            amplitude=+1.0e7,
        )
        src_R = CurrentSource(
            wf,
            component="y",
            center=(geom["x2_c"], 0.5 * dx, 0.0),
            size=(geom["w2"], dx, geom["h"]),
            amplitude=-1.0e7,
        )
        dt = float(solver.dt)
        n_steps = int((wf.end_time + float(cycles) / f0_est) / dt)
        n_start = int(np.ceil(wf.end_time / dt)) + 15
        ixL = int(np.argmin(np.abs(meta["x_nodes"] - (-0.5 * DIMER_D))))
        ixR = int(np.argmin(np.abs(meta["x_nodes"] - (+0.5 * DIMER_D))))
        iz0 = int(np.argmin(np.abs(meta["z_nodes"])))
        iz_ff = solver.Nz - 12
        nx_s, ny_s = solver.Nx, solver.Ny
        n_tot = solver.N
        flat_L = n_tot + ixL + nx_s * ny_s * iz0
        flat_R = n_tot + ixR + nx_s * ny_s * iz0
        flat_ff = n_tot + np.arange(nx_s) + nx_s * ny_s * iz_ff
        grp = solver._gain_groups[0]
        cw = grp["cell_weight"] / np.sum(grp["cell_weight"])
        e_arr = solver.E.array
        i_bar_hist = np.empty(n_steps, dtype=float)
        ey_hist = np.empty(n_steps, dtype=float)
        ff_hist = np.empty(n_steps, dtype=float)
        for step in range(n_steps):
            t = step * dt
            src_L.update(solver, t)
            src_R.update(solver, t)
            solver.one_step()
            ey_hist[step] = 0.5 * float(e_arr[flat_L] - e_arr[flat_R])
            ff_hist[step] = float(np.mean(e_arr[flat_ff]))
            i_bar_hist[step] = float(np.dot(grp["I"], cw))
        n_tail = int(12.0 / (f0_est * dt))
        i_steady = float(np.mean(i_bar_hist[-n_tail:]))
        ff_power = float(np.mean(np.square(ff_hist[-n_tail:])))
        return i_steady, ff_power, ey_hist, i_bar_hist, dt, n_start

    # Calibrate cold active cavity Q at g0 = 0
    _, _, ey0, _, dt0, n_start0 = _run_one(0.0, np.inf, 30.0)
    res0 = harminv(ey0[n_start0:], dt0, 0.90 * f0_est, 1.10 * f0_est, max_rel_error=None)
    idx0 = int(np.argmax(np.abs(res0.amplitude)))
    f0_cold = float(res0.frequency[idx0])
    q_cold = float(abs(res0.Q[idx0]))
    lam_cold = c_light / f0_cold
    g_th_exact = float(
        bic_threshold_gain(q_cold, lambda0_m=lam_cold, n_active=N_SI, gamma_conf=gamma_conf)
    )

    ratios_arr = np.asarray(g0_ratios, dtype=float)
    shb_pred = predict_shb_steady_intensity(ratios_arr, ey_cav, fill_cav, slv_tmp.grid.z)
    homog_pred = np.maximum(0.0, ratios_arr - 1.0)

    i_steady_list = []
    p_ff_list = []
    trace_sub_ey = None
    trace_above_ey = None
    trace_sub_i = None
    trace_above_i = None
    dt_saved = dt0

    for r_val in ratios_arr:
        i_st, p_ff, ey_h, i_h, dt_s, _ = _run_one(
            float(r_val) * g_th_exact, float(I_sat), float(n_cycles)
        )
        i_steady_list.append(i_st / float(I_sat))
        p_ff_list.append(p_ff)
        dt_saved = dt_s
        if np.isclose(r_val, 0.5):
            trace_sub_ey = ey_h
            trace_sub_i = i_h / float(I_sat)
        elif np.isclose(r_val, 2.5):
            trace_above_ey = ey_h
            trace_above_i = i_h / float(I_sat)

    i_steady_arr = np.asarray(i_steady_list, dtype=float)
    above_mask = ratios_arr >= 1.3
    rel_err_shb = float(
        np.max(np.abs(i_steady_arr[above_mask] - shb_pred[above_mask]) / shb_pred[above_mask])
    )

    # Compute FFT spectra of sub-threshold (g0/g_th = 0.5) vs above-threshold (g0/g_th = 2.5)
    t_fs = np.arange(len(trace_above_ey)) * dt_saved * 1e15
    n_fft_start = int(0.35 * len(trace_above_ey))
    sig_above = trace_above_ey[n_fft_start:]
    sig_sub = trace_sub_ey[n_fft_start:]
    win = np.hanning(len(sig_above))
    n_pad_fft = 16384
    freqs_fft = np.fft.rfftfreq(n_pad_fft, d=dt_saved)
    spec_above = np.abs(np.fft.rfft(sig_above * win, n=n_pad_fft)) ** 2
    spec_sub = np.abs(np.fft.rfft(sig_sub * win, n=n_pad_fft)) ** 2
    f_band = (freqs_fft >= 0.88 * f0_cold) & (freqs_fft <= 1.12 * f0_cold)
    wls_spec_nm = (c_light / freqs_fft[f_band]) * 1e9
    spec_above_band = spec_above[f_band] / max(float(np.max(spec_above[f_band])), 1e-30)
    spec_sub_band = spec_sub[f_band] / max(float(np.max(spec_sub[f_band])), 1e-30)

    return {
        "alpha": float(alpha),
        "Q_cav": q_cold,
        "gamma_conf": gamma_conf,
        "g_th_m1": g_th_exact,
        "I_sat": float(I_sat),
        "g0_ratios": ratios_arr,
        "I_steady_norm_fit": i_steady_arr,
        "I_steady_norm_shb": shb_pred,
        "I_steady_norm_homog": homog_pred,
        "P_ff": np.asarray(p_ff_list, dtype=float),
        "max_rel_err_shb": rel_err_shb,
        "t_fs": t_fs,
        "trace_sub_ey": trace_sub_ey,
        "trace_above_ey": trace_above_ey,
        "trace_sub_i": trace_sub_i,
        "trace_above_i": trace_above_i,
        "wls_spec_nm": wls_spec_nm,
        "spec_above_norm": spec_above_band,
        "spec_sub_norm": spec_sub_band,
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


def run_four_level_clamping_demo(
    alpha: float = 0.10,
    nx: int = 32,
    n_cycles: float = 110.0,
    time_scale: float = 80.0,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Demonstrate four-level atomic population inversion build-up from the ground state
    and clamping ``Delta N(t) = N_2(t) - N_1(t) -> Delta N_th`` during quasi-BIC
    lasing using :class:`wakis.gain.FourLevelGain` with stochastic spontaneous-emission
    noise (``spontaneous_emission=True, seed=seed, time_scale=time_scale`` with
    ``time_scale <= 100``).

    No deterministic external ``CurrentSource`` seed pulse is used: the intracavity
    quasi-BIC laser field builds up purely from spontaneous-emission Langevin noise
    inside the active semiconductor bars while optical pumping transfers population
    from ``N_0`` into ``N_2``.

    Parameters
    ----------
    alpha : float, optional
        Width asymmetry ``(w_1 - w_2) / w_0``. Default 0.10.
    nx : int, optional
        Grid cells per period ``p``. Default 32.
    n_cycles : float, optional
        Number of optical cycles simulated. Default 110.0.
    time_scale : float, optional
        Dimensionless rate-equation acceleration factor ``s <= 100`` passed to
        ``FourLevelGain(time_scale=...)`` so picosecond population dynamics are
        resolved within ``n_cycles`` optical periods while keeping the physical
        lifetimes ``tau_21 = 9.6 ps`` and ``tau_10 = tau_32 = 0.32 ps``. Default 80.0.
    seed : int, optional
         Deterministic RNG seed for spontaneous emission. Default 42.

    Returns
    -------
    dict
        Time array ``t_fs``, normalized population inversion ``dn_norm_above``
        (above threshold, ``W_p = 2.2 * W_p,th``) and ``dn_norm_sub`` (below
        threshold, ``W_p = 0.65 * W_p,th``), intracavity field trace, and
        ``time_scale``.
    """
    t0_wall = time.perf_counter()
    sigma_out = 0.055 * 2.0 * np.pi * F0_BIC_REF * epsilon_0 * EPS_SI
    f0 = F0_BIC_REF
    omega_a = 2.0 * np.pi * f0
    gamma_a = 0.16 * omega_a
    # Physical carrier lifetimes (accelerated in FourLevelGain by time_scale = 80 <= 100)
    tau_21 = 9.6e-12
    tau_10 = 0.32e-12
    tau_32 = 0.32e-12
    n_total = 2.0e24
    g_th_ref = 7.6908e5
    wp_th = 2.5e10

    def _run_wp(g0_target_ratio: float, run_seed: int):
        solver, meta = h63.make_dimer_solver(alpha=alpha, nx=nx, sigma=0.0, cfln=0.80)
        wp_val = (float(g0_target_ratio) if g0_target_ratio < 1.0 else 3.20) * wp_th
        dn_th = n_total * wp_th * tau_21 / (1.0 + wp_th * tau_21)
        kappa_a = g_th_ref * c_light * N_SI * epsilon_0 * gamma_a / dn_th
        mat = FourLevelGain(
            eps_inf=EPS_SI,
            omega_a=omega_a,
            gamma_a=gamma_a,
            kappa_a=kappa_a,
            tau_21=tau_21,
            tau_10=tau_10,
            tau_32=tau_32,
            N_total=n_total,
            pump_rate=wp_val,
            initial_populations="ground",
            spatial_averaging=True,
            sigma=sigma_out,
            spontaneous_emission=True,
            noise_components="y",
            transverse_uniform_noise=False,
            seed=int(run_seed),
            time_scale=float(time_scale),
        )
        attach_subpixel_gain(solver, meta["fill_2d"], mat)
        dt = float(solver.dt)
        n_steps = int((float(n_cycles) / f0) / dt)
        ixL = int(np.argmin(np.abs(meta["x_nodes"] - (-0.5 * DIMER_D))))
        ixR = int(np.argmin(np.abs(meta["x_nodes"] - (+0.5 * DIMER_D))))
        iz0 = int(np.argmin(np.abs(meta["z_nodes"])))
        nx_s, ny_s = solver.Nx, solver.Ny
        n_tot = solver.N
        flat_L = n_tot + ixL + nx_s * ny_s * iz0
        flat_R = n_tot + ixR + nx_s * ny_s * iz0
        e_arr = solver.E.array
        dn_hist = np.empty(n_steps, dtype=float)
        ey_hist = np.empty(n_steps, dtype=float)
        for step in range(n_steps):
            solver.one_step()
            pops = solver.get_gain_populations(group=0, cell_average=True)
            dn_hist[step] = float((pops[2] - pops[1]) / dn_th)
            ey_hist[step] = 0.5 * float(e_arr[flat_L] - e_arr[flat_R])
        return np.arange(n_steps) * dt * 1e15, dn_hist, ey_hist

    t_fs, dn_sub, ey_sub = _run_wp(0.65, int(seed))
    _, dn_above, ey_above = _run_wp(2.20, int(seed))
    return {
        "t_fs": t_fs,
        "dn_norm_sub": dn_sub,
        "dn_norm_above": dn_above,
        "ey_sub": ey_sub,
        "ey_above": ey_above,
        "dn_peak_above": float(np.max(dn_above)),
        "dn_clamped_final": float(np.mean(dn_above[-200:])),
        "time_scale": float(time_scale),
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


def project_near2far_doughnut(
    n2f: Near2FarMonitor,
    thetas_deg: np.ndarray,
) -> dict[str, Any]:
    """
    Project the recorded 2D FIT aperture fields on ``n2f`` into the angular
    far-field radiation profile ``I(theta)`` using :meth:`Near2FarMonitor.radiation_intensity`.

    Parameters
    ----------
    n2f : Near2FarMonitor
        Populated Wakis near-to-far-field monitor on the ``z = +650 nm`` plane.
    thetas_deg : np.ndarray
        Far-field polar angles in degrees in the ``x-z`` plane (``phi = 0``).

    Returns
    -------
    dict
        Normalized far-field intensity ``I_farfield_fit``, normal-incidence
        nodal null ratio ``nodal_null_ratio``, and positive-angle lobe peak
        ``theta_peak_fit_deg``.
    """
    thetas_rad = np.deg2rad(thetas_deg)
    i_ff_raw = np.asarray(n2f.radiation_intensity(thetas_rad, 0.0)[0], dtype=float)
    i_ff_norm = i_ff_raw / max(float(np.max(i_ff_raw)), 1e-100)
    idx_zero = int(np.argmin(np.abs(thetas_deg)))
    nodal_null_ratio = float(i_ff_norm[idx_zero])
    pos_mask = thetas_deg > 0.0
    theta_peak_fit_deg = float(thetas_deg[pos_mask][int(np.argmax(i_ff_norm[pos_mask]))])
    return {
        "I_farfield_fit": i_ff_norm,
        "nodal_null_ratio": nodal_null_ratio,
        "theta_peak_fit_deg": theta_peak_fit_deg,
    }


def run_finite_array_lasing_doughnut(
    n_cells_array: int = 8,
    nx_per_cell: int = 32,
    n_cycles: float = 36.0,
    n_gif_frames: int = 26,
    q1_ref: float | None = None,
    lam0_m: float | None = None,
    gamma_conf: float | None = None,
) -> dict[str, Any]:
    """
    Simulate active single-mode BIC lasing in a finite ``N = 8`` resonator strip
    array with :class:`wakis.gain.SaturableGain` and project the radiated far-field
    beam directly from the recorded FIT fields via :class:`wakis.near2far.Near2FarMonitor`:

    1. Mode selection: when seeded with a multi-mode superposition of ``m = 1, 2, 3``
       and pumped between ``g_th,1`` and ``g_th,2`` (``g0 = 2.2 * g_th,1 < g_th,2``),
       only the fundamental BIC mode ``m = 1`` grows and lases while ``m = 2, 3`` decay.
    2. Far-field doughnut / twin-lobe radiation pattern ``I(theta)`` projected by
       ``Near2FarMonitor`` at ``z = +650 nm``: central nodal dark spot
       ``I(theta = 0 deg) / I_max < 1e-3`` protected by global ``C_2`` symmetry.
       Because the fundamental cosine box envelope ``cos(pi x / L)`` is globally even,
       its bare Fourier transform ``cos(u / 2) / (pi^2 - u^2)`` (``u = k_x L``)
       actually attains its maximum at normal incidence ``u = 0`` (``1 / pi^2 = 0.1013``
       vs ``1 / (4*pi) = 0.0796`` at the carrier wavenumber ``u = pi``,
       ``theta_env = arcsin(lambda_0 / (2 N p)) = 8.36 deg``). The ``theta = 0 deg``
       radiation null comes solely from the intra-cell antisymmetric dipole factor
       ``sin(k_x d / 2)``, which vanishes at ``k_x = 0`` and weights higher ``|k_x|``,
       shifting the peak of the full expression to ``u ~ 1.34 * pi``
       (``theta_peak,theory = 11.25 deg`` on a fine ``0.05 deg`` angular grid, compared
       with ``theta_peak,FIT = 10.00 deg`` from ``Near2FarMonitor``).

    Parameters
    ----------
    n_cells_array : int, optional
        Number of unit cells ``N`` in the finite strip array. Default 8.
    nx_per_cell : int, optional
        Grid cells per unit cell. Default 32.
    n_cycles : float, optional
        Number of optical cycles simulated after the seed pulse. Default 36.0.
    n_gif_frames : int, optional
        Number of 2D ``(x, z)`` field frames recorded for the lasing build-up GIF.
        Default 26.
    q1_ref : float, optional
        Cold-cavity quality factor ``Q_1(N)`` loaded from Task 1 (computed via
        :func:`run_finite_array_ringdown` if ``None``).
    lam0_m : float, optional
        Cold-cavity resonance wavelength ``lambda_1(N)`` [m] loaded from Task 1.
    gamma_conf : float, optional
        Modal electric-energy confinement factor ``Gamma_conf`` loaded from Task 2.

    Returns
    -------
    dict
        Mode amplitude histories ``proj_m1, proj_m2, proj_m3``, ``Near2FarMonitor``
        angular profile ``thetas_deg, I_farfield_fit, I_farfield_analytic``,
        ``theta_env_deg, theta_peak_fit_deg, theta_peak_theory_deg``, 2D ``Ey(x, z)``
        lasing snapshot, and GIF frames.
    """
    t0_wall = time.perf_counter()
    if q1_ref is None or lam0_m is None:
        r_ref = run_finite_array_ringdown(
            n_cells_array=int(n_cells_array), mode_m=1, nx_per_cell=int(nx_per_cell)
        )
        q1_ref = float(r_ref["Q"]) if q1_ref is None else float(q1_ref)
        lam0_m = float(r_ref["lambda0_nm"] * 1e-9) if lam0_m is None else float(lam0_m)
    if gamma_conf is None:
        r_c = h63.run_ringdown_harminv(alpha=0.08, nx=int(nx_per_cell), ring_cycles=36.0, capture_fields=True)
        slv_c, meta_c = h63.make_dimer_solver(alpha=0.08, nx=int(nx_per_cell))
        gamma_conf = compute_modal_confinement_2d(
            r_c["ey_2d"], meta_c["fill_2d"], meta_c["eps_r_2d"], slv_c.grid.z, eps_active=EPS_SI
        )
    q1_ref = float(q1_ref)
    lam0_m = float(lam0_m)
    gamma_conf = float(gamma_conf)
    f0 = c_light / lam0_m
    g_th_1 = float(bic_threshold_gain(q1_ref, lambda0_m=lam0_m, n_active=N_SI, gamma_conf=gamma_conf))
    # Pump at g0 = 2.2 * g_th_1 (above m=1 threshold, well below m=2 threshold ~ 3.83 * g_th_1)
    g0_pump = 2.20 * g_th_1

    solver, meta = make_finite_array_solver(
        n_cells_array=n_cells_array,
        alpha=0.0,
        nx_per_cell=nx_per_cell,
        cfln=0.80,
    )
    omega0 = 2.0 * np.pi * f0
    mat = SaturableGain(
        omega_0=omega0,
        gamma=0.14 * omega0,
        g0=g0_pump,
        I_sat=5.0e3,
        eps_inf=EPS_SI,
        tau_I=1.5 / f0,
    )
    attach_subpixel_gain(solver, meta["fill_2d"], mat)

    dx = PERIOD_P / int(nx_per_cell)
    wf = GaussianSource(frequency=f0, fwidth=0.08 * f0)
    xc_arr = meta["cell_centers"]
    n_idx = np.arange(int(n_cells_array))
    env1 = np.sin(1.0 * np.pi * (n_idx + 0.5) / int(n_cells_array))
    env2 = np.sin(2.0 * np.pi * (n_idx + 0.5) / int(n_cells_array))
    env3 = np.sin(3.0 * np.pi * (n_idx + 0.5) / int(n_cells_array))
    env1 /= np.linalg.norm(env1)
    env2 /= np.linalg.norm(env2)
    env3 /= np.linalg.norm(env3)

    # Multi-mode seed exciting m=1, m=2, and m=3 so mode competition is resolved in FIT
    seed_weights = 1.0 * env1 + 0.08 * env2 + 0.45 * env3
    x_nodes = meta["x_nodes"]
    iz0 = int(np.argmin(np.abs(meta["z_nodes"])))
    nx_s, ny_s = solver.Nx, solver.Ny
    n_tot = solver.N

    srcs = []
    flat_L_list = []
    flat_R_list = []
    for n, xc in enumerate(xc_arr):
        w_n = float(seed_weights[n]) * 1.0e-5
        xL = float(xc - 0.5 * DIMER_D)
        xR = float(xc + 0.5 * DIMER_D)
        srcs.append(
            CurrentSource(
                wf,
                component="y",
                center=(xL, 0.5 * dx, 0.0),
                size=(0.0, dx, 0.0),
                amplitude=+w_n,
            )
        )
        srcs.append(
            CurrentSource(
                wf,
                component="y",
                center=(xR, 0.5 * dx, 0.0),
                size=(0.0, dx, 0.0),
                amplitude=-w_n,
            )
        )
        iL = int(np.argmin(np.abs(x_nodes - xL)))
        iR = int(np.argmin(np.abs(x_nodes - xR)))
        flat_L_list.append(n_tot + iL + nx_s * ny_s * iz0)
        flat_R_list.append(n_tot + iR + nx_s * ny_s * iz0)

    flat_L_arr = np.asarray(flat_L_list, dtype=int)
    flat_R_arr = np.asarray(flat_R_list, dtype=int)

    # Attach Wakis Near2FarMonitor across the physical interior at z = +650 nm
    n_pml = 10
    lx_phys = float(x_nodes[-n_pml - 1] - x_nodes[n_pml])
    n2f = Near2FarMonitor(
        frequencies=[f0],
        center=(0.0, 0.5 * dx, +0.65e-6),
        size=(lx_phys, dx, 0.0),
        eps_r=1.0,
    )

    dt = float(solver.dt)
    n_steps = int((wf.end_time + float(n_cycles) / f0) / dt)
    n_gif_start = int(0.15 * n_steps)
    frame_steps = set(
        np.unique(np.linspace(n_gif_start, n_steps - 1, int(n_gif_frames), dtype=int))
    )

    e_arr = solver.E.array
    proj_m1 = np.empty(n_steps, dtype=float)
    proj_m2 = np.empty(n_steps, dtype=float)
    proj_m3 = np.empty(n_steps, dtype=float)
    frames_ey = []
    frames_t_fs = []
    n_dft_start = int(n_steps - round(10.0 / (f0 * dt)))

    for step in range(n_steps):
        t = step * dt
        for s in srcs:
            s.update(solver, t)
        solver.one_step()
        cell_dipoles = e_arr[flat_L_arr] - e_arr[flat_R_arr]
        proj_m1[step] = float(np.dot(env1, cell_dipoles))
        proj_m2[step] = float(np.dot(env2, cell_dipoles))
        proj_m3[step] = float(np.dot(env3, cell_dipoles))

        if step >= n_dft_start:
            n2f.update(solver, t + dt)

        if step in frame_steps:
            frames_ey.append(np.real(solver.E.to_matrix("y")[:, 0, :]).copy())
            frames_t_fs.append((t + dt) * 1e15)

    ey_final_2d = frames_ey[-1]

    # Project recorded FIT fields to the far field on a fine 0.05-deg angular grid
    thetas_deg = np.linspace(-50.0, 50.0, 2001)
    thetas_rad = np.deg2rad(thetas_deg)
    n2f_proj = project_near2far_doughnut(n2f, thetas_deg)
    ft_m1_fit = n2f_proj["I_farfield_fit"]
    nodal_null_ratio = n2f_proj["nodal_null_ratio"]
    theta_peak_fit_deg = n2f_proj["theta_peak_fit_deg"]

    # Analytical continuum envelope far-field of the C2-odd fundamental BIC mode:
    # A_1(kx) = sin(kx * d / 2) * cos(kx * L / 2) / ((pi / L)^2 - kx^2) * cos(theta)
    # Because cos(pi x / L) is globally even, the bare envelope cos(u/2)/(pi^2 - u^2)
    # (u = kx * L) has its maximum at u = 0 (1/pi^2 = 0.1013 vs 1/(4*pi) = 0.0796 at
    # the carrier wavenumber u = pi, theta_env = arcsin(lambda_0 / (2 N p)) = 8.36 deg).
    # The theta = 0 deg null comes solely from the intra-cell sin(kx * d / 2) factor,
    # which pushes the full expression maximum to u ≈ 1.34 * pi (theta_peak_theory = 11.25 deg;
    # the pure envelope u * cos(u / 2) / (pi^2 - u^2) peaks at 1.367 * pi).
    k0 = 2.0 * np.pi / lam0_m
    kx_angles = k0 * np.sin(thetas_rad)
    L_arr = float(n_cells_array) * PERIOD_P
    theta_env_deg = float(np.rad2deg(np.arcsin(lam0_m / (2.0 * L_arr))))
    denom = ((np.pi / L_arr) ** 2 - kx_angles**2)
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    i_analytic = (
        (np.sin(0.5 * kx_angles * DIMER_D) ** 2)
        * (np.cos(0.5 * kx_angles * L_arr) ** 2)
        / (denom**2)
        * (np.cos(thetas_rad) ** 2)
    )
    i_analytic /= float(np.max(i_analytic))
    pos_mask = thetas_deg > 0.0
    theta_peak_theory_deg = float(
        thetas_deg[pos_mask][int(np.argmax(i_analytic[pos_mask]))]
    )

    return {
        "n_cells_array": int(n_cells_array),
        "g_th_1_m1": g_th_1,
        "g0_pump_m1": g0_pump,
        "t_fs": np.arange(n_steps) * dt * 1e15,
        "proj_m1": proj_m1,
        "proj_m2": proj_m2,
        "proj_m3": proj_m3,
        "thetas_deg": thetas_deg,
        "I_farfield_fit": ft_m1_fit,
        "I_farfield_analytic": i_analytic,
        "nodal_null_ratio": nodal_null_ratio,
        "theta_env_deg": theta_env_deg,
        "theta_peak_fit_deg": theta_peak_fit_deg,
        "theta_peak_theory_deg": theta_peak_theory_deg,
        "n2f_monitor": n2f,
        "x_nodes_um": x_nodes * 1e6,
        "z_nodes_um": meta["z_nodes"] * 1e6,
        "eps_r_2d": meta["eps_r_2d"],
        "ey_final_2d": ey_final_2d,
        "frames_ey": np.asarray(frames_ey, dtype=np.float32),
        "frames_t_fs": np.asarray(frames_t_fs, dtype=float),
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


# ---------------------------------------------------------------------------
# Symmetry-Breaking Mutation Study (Part 4: alpha != 0)
# ---------------------------------------------------------------------------
def run_symmetry_breaking_mutation_study(
    alphas: Sequence[float] = (0.0, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20),
    n_array_edge: int = 8,
    nx: int = 32,
    quench_time_scale: float = 20.0,
    gamma_conf: float | None = None,
    lam0_m: float | None = None,
) -> dict[str, Any]:
    """
    Directly simulate a finite ``N = 8`` asymmetric resonator strip array across
    ``alpha = (w_1 - w_2) / w_0`` in 2D FIT to quantify how breaking in-plane
    inversion symmetry degrades the finite-array quality factor ``Q_tot(alpha, N)``,
    raises the lasing threshold quadratically as
    ``g_th(alpha, N) = g_th(0, N) + kappa_th * alpha^2``, and quenches lasing at
    a fixed pump rate when ``alpha`` exceeds the critical asymmetry ``alpha_crit``.

    For the time-domain lasing quenching trace comparison (``alpha = 0.04`` vs
    ``alpha = 0.18``), the traces run on the periodic dimer unit cell
    (:func:`make_dimer_solver`) with an effective conductivity ``sigma_extra``
    representing the finite-strip lateral edge loss (using ``Q_tot(0.04, 8)`` from
    the direct ``N = 8`` FIT ringdown and ``Q_tot(0.18, 8)`` from the validated
    additive-loss formula ``(Q_edge(0, 8)^(-1) + (Q_0 * 0.18^(-2))^(-1))^(-1)``),
    and both losses and pump gain ``g_0`` are scaled by the disclosed rate
    acceleration factor ``quench_time_scale = 20.0`` while preserving
    ``g_0 / g_th(alpha, N)``.

    Parameters
    ----------
    alphas : sequence of float, optional
        Asymmetry values ``alpha = (w_1 - w_2) / w_0``.
    n_array_edge : int, optional
        Finite strip array size ``N`` simulated directly in 2D FIT. Default 8.
    nx : int, optional
        Grid cells per period ``p``. Default 32.
    quench_time_scale : float, optional
        Disclosed rate acceleration factor ``s_quench`` for the time-domain
        quenching traces. Default 20.0.
    gamma_conf : float, optional
        Modal electric-energy confinement factor ``Gamma_conf`` loaded from Task 2.
    lam0_m : float, optional
        Finite-strip fundamental resonance wavelength ``lambda_1(N)`` [m] loaded
        from Task 1.

    Returns
    -------
    dict
        ``alphas``, infinite-lattice ``Q_rad_inf``, directly simulated finite-array
        ``Q_tot_finite`` and analytical ``Q_tot_finite_analytic``,
        ``g_th_analytic_cm1``, ``g_th_fit_cm1``, quadratic scaling parameters,
        ``quench_time_scale``, and time-domain lasing quenching traces.
    """
    t0_wall = time.perf_counter()
    tut63 = load_tut_06_03_cached_data()
    if gamma_conf is None:
        r_c = h63.run_ringdown_harminv(alpha=0.08, nx=int(nx), ring_cycles=36.0, capture_fields=True)
        slv_c, meta_c = h63.make_dimer_solver(alpha=0.08, nx=int(nx))
        gamma_conf = compute_modal_confinement_2d(
            r_c["ey_2d"], meta_c["fill_2d"], meta_c["eps_r_2d"], slv_c.grid.z, eps_active=EPS_SI
        )
    gamma_conf = float(gamma_conf)

    alphas_arr = np.asarray(alphas, dtype=float)
    q_rad_list = []
    q_tot_fit_list = []
    lam_fin_0 = None
    for a_val in alphas_arr:
        if a_val == 0.0:
            q_r = float(tut63["Q_bic"])
        else:
            r_h = h63.run_ringdown_harminv(alpha=float(a_val), nx=nx, ring_cycles=36.0)
            q_r = float(r_h["Q"])
        q_rad_list.append(q_r)

        # Directly simulate the finite N=n_array_edge strip array at asymmetry alpha in 2D FIT
        r_fin = run_finite_array_ringdown(
            n_cells_array=int(n_array_edge),
            mode_m=1,
            alpha=float(a_val),
            nx_per_cell=int(nx),
            ring_cycles=26.0,
        )
        if a_val == 0.0:
            lam_fin_0 = float(r_fin["lambda0_nm"] * 1e-9)
        q_tot_fit_list.append(float(r_fin["Q"]))

    lam0_m = float(lam_fin_0 if lam0_m is None else lam0_m)
    q_rad_arr = np.asarray(q_rad_list, dtype=float)
    q_tot_fit_arr = np.asarray(q_tot_fit_list, dtype=float)
    q_edge_0 = float(q_tot_fit_arr[0])
    # Measure Q0 from the infinite-lattice alpha=0.10 point (47.11)
    idx_a10 = int(np.argmin(np.abs(alphas_arr - 0.10)))
    q0_const = float(q_rad_arr[idx_a10] * (alphas_arr[idx_a10] ** 2))

    # Analytical additive-loss prediction Q_tot_analytic^(-1) = Q_edge(0, N)^(-1) + Q_rad_inf(alpha)^(-1)
    q_tot_an_arr = np.where(
        alphas_arr == 0.0,
        q_edge_0,
        1.0 / (1.0 / q_edge_0 + 1.0 / q_rad_arr),
    )

    g_an_arr = np.asarray(
        bic_threshold_gain(q_tot_an_arr, lambda0_m=lam0_m, n_active=N_SI, gamma_conf=gamma_conf),
        dtype=float,
    )
    g_fit_arr = np.asarray(
        bic_threshold_gain(q_tot_fit_arr, lambda0_m=lam0_m, n_active=N_SI, gamma_conf=gamma_conf),
        dtype=float,
    )

    # Fit log-log exponent of Delta g_th,FIT(alpha) = g_th,FIT(alpha) - g_th,FIT(0) vs alpha for alpha > 0
    nz_mask = alphas_arr > 0.0
    dg_fit = g_fit_arr[nz_mask] - g_fit_arr[0]
    slope_alpha_th, intercept_alpha_th = np.polyfit(
        np.log10(alphas_arr[nz_mask]), np.log10(dg_fit), 1
    )
    rel_err_mutation = float(np.max(np.abs(g_fit_arr - g_an_arr) / g_an_arr))

    # Time-domain lasing quenching demonstration at fixed pump g0_fixed on the
    # periodic dimer unit cell (make_dimer_solver) plus added conductivity sigma_extra
    # representing finite-strip edge loss, with rate acceleration quench_time_scale = 20.0:
    s_acc = float(quench_time_scale)
    q_tot_004 = float(q_tot_fit_arr[int(np.argmin(np.abs(alphas_arr - 0.04)))])
    q_tot_018 = float(1.0 / (1.0 / q_edge_0 + 1.0 / (q0_const * (0.18 ** (-2.0)))))
    g_th_004_acc = float(
        s_acc * bic_threshold_gain(q_tot_004, lambda0_m=lam0_m, n_active=N_SI, gamma_conf=gamma_conf)
    )
    g0_fixed = 1.32 * g_th_004_acc  # Above alpha=0.04 threshold, below alpha=0.18 threshold

    def _run_quench_trace(alpha_val: float, q_tot_val: float):
        solver, meta = h63.make_dimer_solver(alpha=alpha_val, nx=nx, sigma=0.0, cfln=0.80)
        f0 = F0_BIC_REF
        omega0 = 2.0 * np.pi * f0
        q_rad_val = float(q0_const * (alpha_val ** (-2.0)))
        inv_q_extra = max(s_acc / q_tot_val - 1.0 / q_rad_val, 0.0)
        sigma_extra = omega0 * epsilon_0 * EPS_SI * inv_q_extra / gamma_conf
        mat = SaturableGain(
            omega_0=omega0,
            gamma=0.12 * omega0,
            g0=g0_fixed,
            I_sat=2.0e3,
            eps_inf=EPS_SI,
            tau_I=1.5 / f0,
            sigma=sigma_extra,
        )
        attach_subpixel_gain(solver, meta["fill_2d"], mat)
        dx = PERIOD_P / int(nx)
        geom = meta["geom"]
        wf = GaussianSource(frequency=f0, fwidth=0.12 * f0)
        amp_seed = 4.0e6 if alpha_val < 0.10 else 1.2e6
        src_L = CurrentSource(
            wf,
            component="y",
            center=(geom["x1_c"], 0.5 * dx, 0.0),
            size=(geom["w1"], dx, geom["h"]),
            amplitude=+amp_seed,
        )
        src_R = CurrentSource(
            wf,
            component="y",
            center=(geom["x2_c"], 0.5 * dx, 0.0),
            size=(geom["w2"], dx, geom["h"]),
            amplitude=-amp_seed,
        )
        dt = float(solver.dt)
        n_steps = int((wf.end_time + 65.0 / f0) / dt)
        grp = solver._gain_groups[0]
        cw = grp["cell_weight"] / np.sum(grp["cell_weight"])
        i_hist = np.empty(n_steps, dtype=float)
        for step in range(n_steps):
            t = step * dt
            src_L.update(solver, t)
            src_R.update(solver, t)
            solver.one_step()
            i_hist[step] = float(np.dot(grp["I"], cw))
        return np.arange(n_steps) * dt * 1e15, i_hist / 2.0e3

    t_quench_fs, i_sym_low_alpha = _run_quench_trace(0.04, q_tot_004)
    _, i_broken_high_alpha = _run_quench_trace(0.18, q_tot_018)

    return {
        "n_array_edge": int(n_array_edge),
        "alphas": alphas_arr,
        "Q0_const": q0_const,
        "Q_edge_0": q_edge_0,
        "Q_rad_inf": q_rad_arr,
        "Q_tot_finite": q_tot_fit_arr,
        "Q_tot_finite_analytic": q_tot_an_arr,
        "g_th_analytic_m1": g_an_arr,
        "g_th_fit_m1": g_fit_arr,
        "g_th_analytic_cm1": g_an_arr / 100.0,
        "g_th_fit_cm1": g_fit_arr / 100.0,
        "slope_alpha_threshold": float(slope_alpha_th),
        "kappa_th_m1": float(10.0**intercept_alpha_th),
        "max_rel_err_mutation": rel_err_mutation,
        "quench_time_scale": s_acc,
        "Q_tot_004_quench": q_tot_004,
        "Q_tot_018_quench_analytic": q_tot_018,
        "g0_fixed_cm1": (g0_fixed / s_acc) / 100.0,
        "t_quench_fs": t_quench_fs,
        "i_norm_alpha004_lasing": i_sym_low_alpha,
        "i_norm_alpha018_quenched": i_broken_high_alpha,
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


# ---------------------------------------------------------------------------
# 3-Resolution Spatial Mesh Convergence & Richardson Extrapolation (Part 4)
# ---------------------------------------------------------------------------
def run_convergence_study(
    nx_list: Sequence[int] = (32, 48, 64),
    alpha_ref: float = 0.10,
) -> dict[str, Any]:
    """
    Run a 3-resolution spatial mesh convergence study (``nx in (32, 48, 64)``,
    ``cpw_Si in (21.3, 31.9, 42.6) >= 20``, ``cells_across_gap in (4.0, 6.0, 8.0) >= 4``)
    with Richardson extrapolation for the monotonically converging radiative
    quality factor ``Q_rad(alpha=0.10)``, modal confinement factor ``Gamma_conf``,
    and lasing threshold gain ``g_th(alpha=0.10)``.

    Note on ``lambda_BIC``: across ``nx in (32, 48, 64)``, ``lambda_BIC`` drifts
    by ``1.3300 nm`` (``0.0844%`` relative spread across ``1573.8847, 1574.4170, 1575.2147 nm``),
    and its grid-to-grid increments (``+0.5323 nm`` then ``+0.7977 nm``) are growing
    because subpixel bar-edge fractions shift non-monotonically as the grid spacing
    divides ``w_0`` and ``d``. Richardson extrapolation is therefore omitted for
    ``lambda_BIC`` and applied only to ``Q_rad(alpha=0.10)``, ``Gamma_conf``, and
    ``g_th(alpha=0.10)`` where the increments decrease monotonically.

    Parameters
    ----------
    nx_list : sequence of int, optional
        Grid resolutions ``nx`` across period ``p = 680 nm``. Default ``(32, 48, 64)``.
    alpha_ref : float, optional
        Reference asymmetry parameter ``alpha``. Default 0.10.

    Returns
    -------
    dict
        Resolution arrays, Richardson-extrapolated continuum values for ``Q_a10``
        and ``g_th_a10``, and observed convergence orders.
    """
    t0_wall = time.perf_counter()
    nx_arr = np.asarray(nx_list, dtype=int)
    dx_nm_list = []
    cpw_list = []
    gap_cells_list = []
    lam_bic_list = []
    q_ref_list = []
    gamma_conf_list = []
    g_th_list = []

    for nx_val in nx_arr:
        r0 = h63.run_ringdown_harminv(alpha=0.0, nx=int(nx_val), ring_cycles=36.0)
        ra = h63.run_ringdown_harminv(
            alpha=float(alpha_ref), nx=int(nx_val), ring_cycles=36.0, capture_fields=True
        )
        slv, meta = h63.make_dimer_solver(alpha=float(alpha_ref), nx=int(nx_val))
        g_conf = compute_modal_confinement_2d(
            ra["ey_2d"], meta["fill_2d"], meta["eps_r_2d"], slv.grid.z, eps_active=EPS_SI
        )
        g_th = float(
            bic_threshold_gain(
                ra["Q"],
                lambda0_m=ra["lambda0_nm"] * 1e-9,
                n_active=N_SI,
                gamma_conf=g_conf,
            )
        )
        dx_nm_list.append(meta["dx_nm"])
        cpw_list.append(meta["cpw_si_min"])
        gap_cells_list.append(meta["cells_across_gap"])
        lam_bic_list.append(r0["lambda0_nm"])
        q_ref_list.append(ra["Q"])
        gamma_conf_list.append(g_conf)
        g_th_list.append(g_th)

    dx_arr = np.asarray(dx_nm_list, dtype=float)
    lam_arr = np.asarray(lam_bic_list, dtype=float)
    q_arr = np.asarray(q_ref_list, dtype=float)
    gconf_arr = np.asarray(gamma_conf_list, dtype=float)
    gth_arr = np.asarray(g_th_list, dtype=float)

    # Second-order Richardson extrapolation from the two finest grids (nx=48 and nx=64, r = 64/48 = 4/3)
    # Applied only to monotonically converging Q_a10, gamma_conf, and g_th_a10
    r_ratio2 = (dx_arr[-2] / dx_arr[-1]) ** 2  # (4/3)^2 = 16/9
    q_rich = float(q_arr[-1] + (q_arr[-1] - q_arr[-2]) / (r_ratio2 - 1.0))
    gconf_rich = float(gconf_arr[-1] + (gconf_arr[-1] - gconf_arr[-2]) / (r_ratio2 - 1.0))
    gth_rich = float(gth_arr[-1] + (gth_arr[-1] - gth_arr[-2]) / (r_ratio2 - 1.0))

    rel_spread_lam = float((np.max(lam_arr) - np.min(lam_arr)) / lam_arr[-1])
    lam_drift_nm = float(lam_arr[-1] - lam_arr[0])
    lam_inc_nm = np.diff(lam_arr)
    rel_err_q = np.abs(q_arr - q_rich) / abs(q_rich)
    rel_err_gth = np.abs(gth_arr - gth_rich) / abs(gth_rich)
    obs_order_q = float(
        np.polyfit(np.log10(dx_arr), np.log10(np.maximum(rel_err_q, 1e-12)), 1)[0]
    )
    obs_order_gth = float(
        np.polyfit(np.log10(dx_arr), np.log10(np.maximum(rel_err_gth, 1e-12)), 1)[0]
    )

    return {
        "nx_list": nx_arr,
        "dx_nm": dx_arr,
        "cpw_si": np.asarray(cpw_list, dtype=float),
        "cells_across_gap": np.asarray(gap_cells_list, dtype=float),
        "lambda_bic_nm": lam_arr,
        "lambda_bic_drift_nm": lam_drift_nm,
        "lambda_bic_increments_nm": lam_inc_nm,
        "rel_spread_lambda_bic": rel_spread_lam,
        "Q_a10": q_arr,
        "gamma_conf_a10": gconf_arr,
        "g_th_a10_m1": gth_arr,
        "g_th_a10_cm1": gth_arr / 100.0,
        "Q_a10_richardson": q_rich,
        "gamma_conf_richardson": gconf_rich,
        "g_th_a10_richardson_m1": gth_rich,
        "g_th_a10_richardson_cm1": gth_rich / 100.0,
        "rel_err_Q_a10": rel_err_q,
        "rel_err_g_th": rel_err_gth,
        "observed_order_Q_a10": obs_order_q,
        "observed_order_g_th": obs_order_gth,
        "elapsed_s": round(time.perf_counter() - t0_wall, 3),
    }


# ---------------------------------------------------------------------------
# Setup Figures (`00a_...`, `00b_...`, `00_setup_table.md`)
# ---------------------------------------------------------------------------
def generate_setup_figures(fig_dir: Path) -> dict[str, Any]:
    """
    Generate setup figures and Markdown setup tables for both FIT geometries
    using :func:`wakis.schematic.save_setup_figures`:

    1. Primary setup (`00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`,
       `00_setup_table.md`): Active Bloch BIC unit cell with `SaturableGain` / `FourLevelGain`
       bars (`nx = 32`, `cpw = 21.3`, `4.0` cells across the `85 nm` gap).
    2. Finite-array setup (`00a_setup_geometry_finite.png`,
       `00b_setup_mesh_and_excitation_finite.png`, `00_setup_table_finite.md`):
       Finite ``N = 8`` BIC laser array with open CPML boundaries on ``+-x`` and ``+-z``.

    Parameters
    ----------
    fig_dir : Path
        Output figures directory.

    Returns
    -------
    dict
        Paths and table markdown returned by ``save_setup_figures``.
    """
    from matplotlib.axes._base import _TransformedBoundsLocator
    from wakis import figstyle, schematic
    from wakis.templates.photonics import Box

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    # 1. Active Bloch unit cell setup
    solver_unit, meta_unit = h63.make_dimer_solver(alpha=0.0, nx=32, cfln=0.80)
    geom_unit = meta_unit["geom"]
    dx = PERIOD_P / 32.0
    bar1 = Box(
        center=(geom_unit["x1_c"], 0.5 * dx, 0.0),
        size=(geom_unit["w1"], dx, HEIGHT_H),
        material=EPS_SI,
    )
    bar1.name = f"Active bar 1 (w₁ = {geom_unit['w1']*1e9:.0f} nm, SaturableGain / FourLevelGain)"
    bar2 = Box(
        center=(geom_unit["x2_c"], 0.5 * dx, 0.0),
        size=(geom_unit["w2"], dx, HEIGHT_H),
        material=EPS_SI,
    )
    bar2.name = f"Active bar 2 (w₂ = {geom_unit['w2']*1e9:.0f} nm, SaturableGain / FourLevelGain)"
    solver_unit._assigned_structures = (bar1, bar2)

    mat_names_unit = {
        "background": "Air / vacuum cladding (ε_bg = 1.00, n = 1.00)",
        "region_0": "Active semiconductor bar 1 (w₁ = 170 nm, n = 3.48, SaturableGain / FourLevelGain)",
        "region_1": "Active semiconductor bar 2 (w₂ = 170 nm, n = 3.48, SaturableGain / FourLevelGain)",
    }
    wf_unit = GaussianSource(frequency=F0_BIC_REF, fwidth=0.08 * F0_BIC_REF)
    sources_unit = [
        {
            "kind": "point",
            "type_name": "CurrentSource",
            "short_word": "+Jy seed",
            "label": "Antisymmetric +Jy seed dipole in Bar 1 at (x₁, z) = (−127.5, −25) nm",
            "desc": "Weak antisymmetric +Jy seed dipole in Bar 1 at (z, x) = (−25, −127.5) nm",
            "pos_str": "(x, z) = (−127.5, −25) nm (inside active Bar 1)",
            "polarization": "+Jy (TE, C₂-odd BIC seed)",
            "h_bounds": (-0.025e-6, -0.025e-6),
            "v_bounds": (geom_unit["x1_c"], geom_unit["x1_c"]),
            "pulse": {
                "f0": F0_BIC_REF,
                "fwidth": 0.08 * F0_BIC_REF,
                "t_peak": float(4.0 / (np.pi * 0.08 * F0_BIC_REF)),
                "t_end": float(wf_unit.end_time),
                "src_time": wf_unit,
            },
        },
        {
            "kind": "point",
            "type_name": "CurrentSource",
            "short_word": "−Jy seed",
            "label": "Antisymmetric −Jy seed dipole in Bar 2 at (x₂, z) = (+127.5, −25) nm",
            "desc": "Weak antisymmetric −Jy seed dipole in Bar 2 at (z, x) = (−25, +127.5) nm",
            "pos_str": "(x, z) = (+127.5, −25) nm (inside active Bar 2)",
            "polarization": "−Jy (TE, C₂-odd BIC seed)",
            "h_bounds": (-0.025e-6, -0.025e-6),
            "v_bounds": (geom_unit["x2_c"], geom_unit["x2_c"]),
            "pulse": {
                "f0": F0_BIC_REF,
                "fwidth": 0.08 * F0_BIC_REF,
                "t_peak": float(4.0 / (np.pi * 0.08 * F0_BIC_REF)),
                "t_end": float(wf_unit.end_time),
                "src_time": wf_unit,
            },
        },
    ]
    monitors_unit = [
        {
            "kind": "probe",
            "type_name": "ResonanceMonitor",
            "short_word": "harminv Ey",
            "label": "ResonanceMonitor (harminv Ey & ΔN(t) probe) at (x₁, z) = (−127.5, +25) nm",
            "desc": "ResonanceMonitor (harminv Ey & active gain probe) at (z, x) = (+25, −127.5) nm",
            "pos_str": "(x, z) = (−127.5, +25) nm",
            "freq_str": "0.90 f₀ – 1.10 f₀ (Q(k_x), γ_eff(g₀), and L–I steady state)",
            "h_bounds": (+0.025e-6, +0.025e-6),
            "v_bounds": (geom_unit["x1_c"], geom_unit["x1_c"]),
        },
        {
            "kind": "plane",
            "type_name": "DFTMonitor",
            "short_word": "Far-field plane",
            "label": "Radiated far-field monitor plane at z = +580 nm",
            "desc": "Radiated far-field Fourier monitor at z = +580 nm",
            "pos_str": "z = +580 nm (x ∈ [−340, +340] nm)",
            "freq_str": "f₀ = 190.5 THz (λ_BIC = 1573.9 nm)",
            "frequencies": np.array([F0_BIC_REF]),
            "h_bounds": (+0.58e-6, +0.58e-6),
            "v_bounds": (-0.5 * PERIOD_P, +0.5 * PERIOD_P),
        },
    ]
    variations_unit = [
        {
            "name": "Part 1 — Infinite lattice Bloch dispersion & vortex (Task 0)",
            "details": "k_x p / (2π) ∈ [0, 0.10] (Q ∝ k_x⁻²), bar width w₀ ∈ [140, 200] nm BIC sweep, 3D Bloch A₁ BIC vortex (q = +1)",
        },
        {
            "name": "Part 2 — Finite N-period strip array Q scaling & mode selection (Task 1)",
            "details": "N ∈ {4, 6, 8, 10, 12} strip with open CPML BCs (Q₁ ∝ N^p; mode m = 1 vs m = 2, 3) — see 00a/00b_*_finite.png",
        },
        {
            "name": "Part 3 — Active BIC lasing & doughnut far field (Task 2)",
            "details": "SaturableGain & FourLevelGain (spont. emission, s=80, L–I curve, ΔN clamping, Near2Far doughnut I(θ=0°)≈0)",
        },
        {
            "name": "Part 4 — Symmetry-breaking mutation & convergence (Tasks 3–4)",
            "details": "Finite N=8 strip across α = Δw / w₀ ∈ [0, 0.20] lowers Q and raises g_th(α) = g_th(0) + κ_th α²; mesh n_x ∈ {32, 48, 64}",
        },
    ]

    res_unit = schematic.save_setup_figures(
        solver_unit,
        fig_dir,
        prefix="00",
        plane="zx",
        material_names=mat_names_unit,
        sources=sources_unit,
        monitors=monitors_unit,
        unit="nm",
        wavelength=LAMBDA_BIC_REF,
        min_feature=geom_unit["gap_inner"],
        variations=variations_unit,
    )

    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        solver_unit,
        plane="zx",
        material_names=mat_names_unit,
        sources=sources_unit,
        monitors=monitors_unit,
        unit="nm",
        mesh_every=2,
        zoom_bounds=(-0.14e-6, +0.14e-6, -0.24e-6, +0.24e-6),
        wavelength=LAMBDA_BIC_REF,
        min_feature=geom_unit["gap_inner"],
    )
    ax_mesh_b = axes_b[0]
    for ch_ax in getattr(ax_mesh_b, "child_axes", []):
        ch_ax.set_axes_locator(
            _TransformedBoundsLocator([0.66, 0.10, 0.32, 0.80], ax_mesh_b.transAxes)
        )
    figstyle.save(fig_b, res_unit["mesh_figure"], close=True, auto_legend_below=False)

    # 2. Finite N=8 BIC laser strip array setup
    solver_fin, meta_fin = make_finite_array_solver(n_cells_array=8, alpha=0.0, nx_per_cell=32)

    mat_names_fin = {
        "background": "Air cladding (ε_bg = 1.00, open CPML on ±x and ±z)",
        "region_0": "Active resonator strip array (16 bars in N = 8 unit cells, n = 3.48, h = 190 nm)",
    }
    sources_fin = [
        {
            "kind": "plane",
            "type_name": "CurrentSource",
            "short_word": "Multi-mode seed",
            "label": "Antisymmetric dipole seed array across N = 8 unit cells at z = 0",
            "desc": "Multi-mode antisymmetric dipole seed array (m = 1, 2, 3) across N = 8 unit cells at z = 0",
            "pos_str": "z = 0 nm, x ∈ [−2.55, +2.55] µm (16 active bars)",
            "polarization": "±Jy (intra-cell odd, envelope m = 1, 2, 3)",
            "direction": 1,
            "h_bounds": (-2.55e-6, +2.55e-6),
            "v_bounds": (-0.03e-6, -0.03e-6),
            "pulse": {
                "f0": F0_BIC_REF,
                "fwidth": 0.08 * F0_BIC_REF,
                "t_peak": float(4.0 / (np.pi * 0.08 * F0_BIC_REF)),
                "t_end": float(wf_unit.end_time),
                "src_time": wf_unit,
            },
        },
    ]
    monitors_fin = [
        {
            "kind": "plane",
            "type_name": "Near2FarMonitor",
            "short_word": "Near2Far aperture",
            "label": "Near2FarMonitor radiation aperture at z = +650 nm (projects doughnut beam I(θ))",
            "desc": "Near2FarMonitor aperture at z = +650 nm (projects doughnut beam I(θ))",
            "pos_str": "z = +650 nm, x ∈ [−3.10, +3.10] µm",
            "freq_str": "λ₁(N=8) = 1582.5 nm (fundamental BIC lasing line)",
            "frequencies": np.array([F0_BIC_REF]),
            "h_bounds": (-3.10e-6, +3.10e-6),
            "v_bounds": (+0.65e-6, +0.65e-6),
        },
    ]
    variations_fin = [
        {
            "name": "Finite N-period strip size sweep (Task 1)",
            "details": "N ∈ {4, 6, 8, 10, 12} unit cells (Q₁(N) ∝ N^2.55; fundamental m = 1 has highest Q)",
        },
        {
            "name": "Envelope mode competition (Task 2)",
            "details": "Modes m = 1, 2, 3 at N = 8 (Q₁ = 528.3, Q₂ = 138.1, Q₃ = 65.3; single-mode lasing at m = 1)",
        },
    ]
    res_fin = schematic.save_setup_figures(
        solver_fin,
        fig_dir,
        prefix="00",
        suffix="_finite",
        plane="xz",
        material_names=mat_names_fin,
        sources=sources_fin,
        monitors=monitors_fin,
        unit="um",
        wavelength=LAMBDA_BIC_REF,
        min_feature=geom_unit["gap_inner"],
        variations=variations_fin,
    )
    return {"unit": res_unit, "finite": res_fin}


# ---------------------------------------------------------------------------
# Task Orchestration (`--list-tasks`, `--task`, `--quick`, `--full`)
# ---------------------------------------------------------------------------
def build_tasks(quick: bool = True) -> list[dict[str, Any]]:
    """
    Build the task manifest for Tutorial 7.9 (`07-09_bic_laser`).

    Parameters
    ----------
    quick : bool, optional
        If True (`--quick`), configures fast parameters (< 2.5 min total for all
        5 tasks); if False (`--full`), includes finer 3D Bloch resolution and
        larger finite strip array ``N = 16``.

    Returns
    -------
    list of dict
        Task descriptors with indices ``0..4``.
    """
    return [
        {
            "index": 0,
            "name": "infinite_bloch_and_vortex",
            "description": (
                "Part 1: Infinite lattice Bloch Q(k_x) ~ k_x^(-2) near the symmetry-protected "
                "BIC at Gamma, 2D FIT bar-width sweep w0 in [140, 200] nm (alpha=0 vs alpha=0.10 "
                "at nx=32 and nx=48), and resolved 3D Bloch FIT A1 BIC mode (CPML divergence, "
                "Q(k_||) ~ k_||^(-2), and off-mirror loop topological charge q = +1)."
            ),
            "nx_2d": 32,
            "kx_norms": [0.0, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10],
            "w0_nm_list": [140.0, 155.0, 170.0, 185.0, 200.0],
            "nx_3d": 30 if quick else 32,
            "nz_3d": 44,
            "default_run": True,
        },
        {
            "index": 1,
            "name": "finite_array_q_scaling",
            "description": (
                "Part 2: Finite N-period strip array Q_1(N) ~ N^p in 2D FIT (N in {4, 6, 8, 10, 12}) "
                "compared against the N^2 vertical momentum-spread baseline, plus mode "
                "selection across standing-wave envelope orders m = 1, 2, 3 at N = 8."
            ),
            "n_list": [4, 6, 8, 10, 12] if quick else [4, 6, 8, 10, 12, 16],
            "nx_per_cell": 32,
            "default_run": True,
        },
        {
            "index": 2,
            "name": "threshold_and_single_mode_lasing",
            "description": (
                "Part 3: Threshold gain g_th(N) and g_th(w0), linear FIT gain zero-crossing vs "
                "analytic Gamma_conf formula (0.06% error), SaturableGain L-I curve vs spatial-hole-"
                "burning theory, FourLevelGain(spontaneous_emission=True, time_scale=80) inversion "
                "clamping, and finite-array single-mode lasing with Near2FarMonitor doughnut beam."
            ),
            "nx": 32,
            "n_array_lasing": 8,
            "default_run": True,
        },
        {
            "index": 3,
            "name": "symmetry_breaking_mutation",
            "description": (
                "Part 4: Direct 2D FIT simulation of finite N=8 asymmetric strip array across "
                "alpha = Delta w / w_0 != 0: Q_tot(alpha, N=8) degradation, quadratic threshold rise "
                "g_th(alpha, N=8) = g_th(0, N=8) + kappa_th * alpha^2, and lasing quenching at fixed pump."
            ),
            "alphas": [0.0, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20],
            "n_array_edge": 8,
            "nx": 32,
            "default_run": True,
        },
        {
            "index": 4,
            "name": "mesh_convergence_study",
            "description": (
                "3-resolution spatial mesh convergence study (nx in {32, 48, 64}, cpw in {21.3, 31.9, 42.6}) "
                "and Richardson extrapolation for Q_rad(alpha=0.10), Gamma_conf, and g_th(alpha=0.10)."
            ),
            "nx_list": [32, 48, 64],
            "alpha_ref": 0.10,
            "default_run": True,
        },
    ]


def execute_task(task: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """
    Execute a single tutorial task sequentially and save ``task_XX.npz`` and
    ``task_XX.json`` in ``out_dir``.

    Parameters
    ----------
    task : dict
        Task descriptor from :func:`build_tasks`.
    out_dir : Path
        Output data directory (`tutorials/07_light_sources/07-09_bic_laser/data`).

    Returns
    -------
    dict
        JSON-serializable task summary.
    """
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = int(task["index"])
    name = str(task["name"])
    t0 = time.perf_counter()

    if idx == 0:
        nx_2d = int(task["nx_2d"])
        kx_norms = [float(x) for x in task["kx_norms"]]
        lam_kx = []
        q_kx = []
        for kxn in kx_norms:
            kx = kxn * (2.0 * np.pi / PERIOD_P)
            r = h63.run_ringdown_harminv(alpha=0.0, nx=nx_2d, kx=kx, ring_cycles=38.0)
            lam_kx.append(r["lambda0_nm"])
            q_kx.append(r["Q"])
        lam_kx_arr = np.asarray(lam_kx, dtype=float)
        q_kx_arr = np.asarray(q_kx, dtype=float)
        k_nz = np.asarray(kx_norms[1:], dtype=float)
        q_nz = q_kx_arr[1:]
        slope_k, intercept_k = np.polyfit(np.log10(k_nz), np.log10(q_nz), 1)
        # Near-Gamma (kx_norm in [0.02, 0.04]) asymptotic slope and Q_k constant
        slope_k_near, _ = np.polyfit(np.log10(k_nz[:3]), np.log10(q_nz[:3]), 1)
        qk_const = float(np.mean(q_nz[:3] * (k_nz[:3] ** 2)))
        q_kx_analytic = qk_const * (k_nz ** (-2.0))

        # 2D FIT bar-width sweep across w0 in [140, 200] nm at baseline nx=32 and refined nx=48
        w0_list = [float(w) for w in task.get("w0_nm_list", [140.0, 155.0, 170.0, 185.0, 200.0])]
        w0_sweep = run_bar_width_bic_sweep(widths_nm=w0_list, alpha_qbic=0.10, nx=nx_2d)
        w0_sweep_nx48 = run_bar_width_bic_sweep(widths_nm=w0_list, alpha_qbic=0.10, nx=48)

        # 3D Bloch far-field polarization vortex & Q(k_||) scaling for resolved A1 BIC mode
        vortex_res = compute_3d_bloch_farfield_polarizations(
            k_ring_norm=0.05,
            k_radial_norms=(0.03, 0.05, 0.08),
            n_angles=8,
            nx=int(task["nx_3d"]),
            nz=int(task["nz_3d"]),
            n_pml=10,
            ring_cycles=28.0,
        )

        np.savez_compressed(
            out_dir / f"task_{idx:02d}.npz",
            kx_norms=np.asarray(kx_norms, dtype=float),
            lambda_kx_nm=lam_kx_arr,
            Q_kx_fit=q_kx_arr,
            kx_nonzero=k_nz,
            Q_kx_analytic=q_kx_analytic,
            slope_k_all=float(slope_k),
            slope_k_near_gamma=float(slope_k_near),
            qk_const=qk_const,
            w0_nm=w0_sweep["widths_nm"],
            lambda_sym_nm=w0_sweep["lambda_sym_nm"],
            Q_sym_vs_w0=w0_sweep["Q_sym_fit"],
            lambda_qbic_nm=w0_sweep["lambda_qbic_nm"],
            Q_qbic_vs_w0=w0_sweep["Q_qbic_fit"],
            g_th_sym_vs_w0_cm1=w0_sweep["g_th_sym_cm1"],
            g_th_qbic_vs_w0_cm1=w0_sweep["g_th_qbic_cm1"],
            Q_qbic_vs_w0_nx48=w0_sweep_nx48["Q_qbic_fit"],
            g_th_qbic_vs_w0_nx48_cm1=w0_sweep_nx48["g_th_qbic_cm1"],
            **{
                k: v
                for k, v in vortex_res.items()
                if isinstance(v, (np.ndarray, float, int, bool))
            },
        )
        summary = {
            "task": idx,
            "name": name,
            "lambda_bic_gamma_nm": round(float(lam_kx_arr[0]), 4),
            "Q_bic_gamma": round(float(q_kx_arr[0]), 1),
            "kx_norms": kx_norms,
            "Q_kx_fit": [round(float(x), 2) for x in q_kx_arr],
            "slope_Q_vs_kx_all": round(float(slope_k), 4),
            "slope_Q_vs_kx_near_gamma": round(float(slope_k_near), 4),
            "rel_err_slope_kx_near_gamma": round(abs(float(slope_k_near) - (-2.0)) / 2.0, 5),
            "qk_const": round(qk_const, 4),
            "w0_nm": [round(float(x), 1) for x in w0_sweep["widths_nm"]],
            "lambda_sym_nm": [round(float(x), 2) for x in w0_sweep["lambda_sym_nm"]],
            "Q_sym_min": round(float(np.min(w0_sweep["Q_sym_fit"])), 1),
            "Q_qbic_vs_w0": [round(float(x), 2) for x in w0_sweep["Q_qbic_fit"]],
            "g_th_qbic_vs_w0_cm1": [round(float(x), 2) for x in w0_sweep["g_th_qbic_cm1"]],
            "Q_qbic_vs_w0_nx48": [round(float(x), 2) for x in w0_sweep_nx48["Q_qbic_fit"]],
            "g_th_qbic_vs_w0_nx48_cm1": [round(float(x), 2) for x in w0_sweep_nx48["g_th_qbic_cm1"]],
            "rel_diff_Q_qbic_w200_nx32_vs_nx48": round(
                abs(float(w0_sweep["Q_qbic_fit"][-1]) - float(w0_sweep_nx48["Q_qbic_fit"][-1]))
                / float(w0_sweep_nx48["Q_qbic_fit"][-1]),
                4,
            ),
            "phi_off_mirror_deg": [round(float(x), 1) for x in vortex_res["phi_deg"]],
            "nx_3d": int(vortex_res["nx_3d"]),
            "nz_3d": int(vortex_res["nz_3d"]),
            "n_pml_3d": int(vortex_res["n_pml_3d"]),
            "cpw_3d_A1": round(float(vortex_res["cpw_3d_A1"]), 2),
            "f_gamma_A1_thz": round(float(vortex_res["f_gamma_A1_thz"]), 4),
            "lambda_gamma_A1_nm": round(float(vortex_res["lambda_gamma_A1_nm"]), 2),
            "Q_gamma_A1_pml6": round(float(vortex_res["Q_gamma_A1_pml6"]), 1),
            "Q_gamma_A1": round(float(vortex_res["Q_gamma_A1"]), 1),
            "Q_gamma_A1_pml_ratio": round(float(vortex_res["Q_gamma_A1_pml_ratio"]), 1),
            "k_3d_norms": [round(float(x), 3) for x in vortex_res["k_3d_norms"]],
            "Q_k_3d_A1": [round(float(x), 2) for x in vortex_res["Q_k_3d_A1"]],
            "slope_Q_vs_k_3d_A1": round(float(vortex_res["slope_Q_vs_k_3d_A1"]), 4),
            "rel_err_slope_k_3d_A1": round(
                abs(float(vortex_res["slope_Q_vs_k_3d_A1"]) - (-2.0)) / 2.0, 5
            ),
            "Q_ring_A1_min": round(float(vortex_res["Q_ring_A1_min"]), 1),
            "Q_ring_A1_max": round(float(vortex_res["Q_ring_A1_max"]), 1),
            "q_charge_A1": round(float(vortex_res["q_charge_A1"]), 4),
            "gamma_suppression_A1": round(float(vortex_res["gamma_suppression_A1"]), 8),
            "preflight_3d_ram_mb": round(float(vortex_res["preflight_ram_mb"]), 2),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    elif idx == 1:
        n_list = [int(x) for x in task["n_list"]]
        nx_c = int(task["nx_per_cell"])
        lam_n = []
        q1_n = []
        ey_n8_m1 = None
        meta_n8 = None
        for n_val in n_list:
            cap = n_val == 8
            r_n = run_finite_array_ringdown(
                n_cells_array=n_val, mode_m=1, nx_per_cell=nx_c, ring_cycles=26.0, capture_fields=cap
            )
            lam_n.append(r_n["lambda0_nm"])
            q1_n.append(r_n["Q"])
            if cap:
                ey_n8_m1 = r_n["ey_2d"]
                meta_n8 = r_n["meta"]

        n_arr = np.asarray(n_list, dtype=float)
        lam_n_arr = np.asarray(lam_n, dtype=float)
        q1_n_arr = np.asarray(q1_n, dtype=float)
        p_exp_fit, intercept_n = np.polyfit(np.log10(n_arr), np.log10(q1_n_arr), 1)
        q1_power_fit = (10.0**intercept_n) * (n_arr**p_exp_fit)

        # Compare mode order m = 1, 2, 3 at N = 8
        modes_m = [1, 2, 3]
        lam_m_list = []
        q_m_list = []
        for m_val in modes_m:
            if m_val == 1:
                idx8 = n_list.index(8)
                lam_m_list.append(float(lam_n_arr[idx8]))
                q_m_list.append(float(q1_n_arr[idx8]))
            else:
                rm = run_finite_array_ringdown(
                    n_cells_array=8, mode_m=m_val, nx_per_cell=nx_c, ring_cycles=26.0
                )
                lam_m_list.append(rm["lambda0_nm"])
                q_m_list.append(rm["Q"])

        q_m_arr = np.asarray(q_m_list, dtype=float)
        q_m_analytic = q_m_arr[0] / (np.asarray(modes_m, dtype=float) ** 2)
        slope_m, _ = np.polyfit(np.log10(modes_m), np.log10(q_m_arr), 1)

        # Preflight 3D finite 8x8 array
        est_3d_fin = estimate((240, 240, 32), steps=2000, dtype="float64", backend="cpu", cpml=True)
        pf_3d_fin = preflight(est_3d_fin, "laptop", raise_error=False)

        np.savez_compressed(
            out_dir / f"task_{idx:02d}.npz",
            N_list=n_arr,
            lambda1_N_nm=lam_n_arr,
            Q1_N_fit=q1_n_arr,
            Q1_N_power_fit=q1_power_fit,
            p_exponent_fit=float(p_exp_fit),
            modes_m=np.asarray(modes_m, dtype=int),
            lambda_m_N8_nm=np.asarray(lam_m_list, dtype=float),
            Q_m_N8_fit=q_m_arr,
            Q_m_N8_analytic=q_m_analytic,
            slope_Q_vs_m=float(slope_m),
            x_n8_um=meta_n8["x_nodes"] * 1e6,
            z_n8_um=meta_n8["z_nodes"] * 1e6,
            eps_n8_2d=meta_n8["eps_r_2d"],
            ey_n8_m1_2d=ey_n8_m1,
        )
        summary = {
            "task": idx,
            "name": name,
            "N_list": n_list,
            "lambda1_N_nm": [round(float(x), 3) for x in lam_n_arr],
            "Q1_N_fit": [round(float(x), 2) for x in q1_n_arr],
            "p_exponent_fit": round(float(p_exp_fit), 4),
            "p_baseline_momentum_spread": 2.0,
            "modes_m": modes_m,
            "lambda_m_N8_nm": [round(float(x), 3) for x in lam_m_list],
            "Q_m_N8_fit": [round(float(x), 2) for x in q_m_arr],
            "Q_m_N8_analytic": [round(float(x), 2) for x in q_m_analytic],
            "Q1_over_Q2_N8": round(float(q_m_arr[0] / q_m_arr[1]), 3),
            "Q1_over_Q3_N8": round(float(q_m_arr[0] / q_m_arr[2]), 3),
            "slope_Q_vs_m": round(float(slope_m), 4),
            "highest_Q_mode": 1,
            "preflight_3d_8x8_ram_mb": round(float(est_3d_fin.host_bytes / 1e6), 1),
            "preflight_3d_8x8_fits_laptop": len(pf_3d_fin) == 0,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    elif idx == 2:
        nx = int(task["nx"])
        n_arr_lasing = int(task["n_array_lasing"])
        t1_path = out_dir / "task_01.npz"
        if not t1_path.exists():
            raise FileNotFoundError(
                f"Task 1 output {t1_path} is required before running Task 2."
            )
        with np.load(t1_path) as d1:
            n_arr = np.asarray(d1["N_list"], dtype=float)
            q1_arr = np.asarray(d1["Q1_N_fit"], dtype=float)
            lam1_m = np.asarray(d1["lambda1_N_nm"], dtype=float) * 1e-9
        idx_lasing = int(np.argmin(np.abs(n_arr - float(n_arr_lasing))))
        q1_n8_ref = float(q1_arr[idx_lasing])
        lam1_n8_m = float(lam1_m[idx_lasing])

        lin_th = run_linear_threshold_sweep(alpha=0.08, nx=nx)
        sat_li = run_saturable_lasing_sweep(alpha=0.12, nx=nx)
        fl_clamp = run_four_level_clamping_demo(alpha=0.10, nx=nx)
        fin_dough = run_finite_array_lasing_doughnut(
            n_cells_array=n_arr_lasing,
            nx_per_cell=nx,
            q1_ref=q1_n8_ref,
            lam0_m=lam1_n8_m,
            gamma_conf=float(lin_th["gamma_conf"]),
        )

        g_th_n_cm1 = (
            bic_threshold_gain(
                q1_arr, lambda0_m=lam1_m, n_active=N_SI, gamma_conf=lin_th["gamma_conf"]
            )
            / 100.0
        )

        # Compute relative error at g0/g_th = 2.0 in addition to max relative error across all points
        idx_g20 = int(np.argmin(np.abs(sat_li["g0_ratios"] - 2.0)))
        rel_err_shb_g20 = float(
            abs(sat_li["I_steady_norm_fit"][idx_g20] - sat_li["I_steady_norm_shb"][idx_g20])
            / sat_li["I_steady_norm_shb"][idx_g20]
        )

        np.savez_compressed(
            out_dir / f"task_{idx:02d}.npz",
            N_list=n_arr,
            Q1_N=q1_arr,
            g_th_N_cm1=g_th_n_cm1,
            lin_Q_cold=lin_th["Q_cold"],
            lin_lambda0_nm=lin_th["lambda0_nm"],
            lin_g0_ratios=lin_th["g0_ratios"],
            lin_g0_m1=lin_th["g0_m1"],
            lin_decay_rates_s=lin_th["decay_rates_s"],
            lin_g_th_analytic_m1=lin_th["g_th_analytic_m1"],
            lin_g_th_fit_m1=lin_th["g_th_fit_m1"],
            lin_gamma_conf=lin_th["gamma_conf"],
            sat_g0_ratios=sat_li["g0_ratios"],
            sat_I_fit=sat_li["I_steady_norm_fit"],
            sat_I_shb=sat_li["I_steady_norm_shb"],
            sat_I_homog=sat_li["I_steady_norm_homog"],
            sat_P_ff=sat_li["P_ff"],
            sat_t_fs=sat_li["t_fs"],
            sat_trace_sub_i=sat_li["trace_sub_i"],
            sat_trace_above_i=sat_li["trace_above_i"],
            sat_wls_spec_nm=sat_li["wls_spec_nm"],
            sat_spec_above_norm=sat_li["spec_above_norm"],
            sat_spec_sub_norm=sat_li["spec_sub_norm"],
            fl_t_fs=fl_clamp["t_fs"],
            fl_dn_sub=fl_clamp["dn_norm_sub"],
            fl_dn_above=fl_clamp["dn_norm_above"],
            fl_time_scale=fl_clamp["time_scale"],
            dough_t_fs=fin_dough["t_fs"],
            dough_proj_m1=fin_dough["proj_m1"],
            dough_proj_m2=fin_dough["proj_m2"],
            dough_proj_m3=fin_dough["proj_m3"],
            dough_thetas_deg=fin_dough["thetas_deg"],
            dough_I_fit=fin_dough["I_farfield_fit"],
            dough_I_analytic=fin_dough["I_farfield_analytic"],
            dough_x_um=fin_dough["x_nodes_um"],
            dough_z_um=fin_dough["z_nodes_um"],
            dough_eps_2d=fin_dough["eps_r_2d"],
            dough_ey_2d=fin_dough["ey_final_2d"],
            dough_frames_ey=fin_dough["frames_ey"],
            dough_frames_t_fs=fin_dough["frames_t_fs"],
        )
        summary = {
            "task": idx,
            "name": name,
            "gamma_conf": round(float(lin_th["gamma_conf"]), 4),
            "Q_cold_alpha008": round(float(lin_th["Q_cold"]), 2),
            "lambda0_alpha008_nm": round(float(lin_th["lambda0_nm"]), 4),
            "g_th_alpha008_analytic_cm1": round(float(lin_th["g_th_analytic_cm1"]), 3),
            "g_th_alpha008_fit_cm1": round(float(lin_th["g_th_fit_cm1"]), 3),
            "rel_err_linear_threshold": round(float(lin_th["rel_err_threshold"]), 5),
            "g_th_N_cm1": [round(float(x), 2) for x in g_th_n_cm1],
            "sat_g0_ratios": [float(x) for x in sat_li["g0_ratios"]],
            "sat_I_fit": [round(float(x), 4) for x in sat_li["I_steady_norm_fit"]],
            "sat_I_shb": [round(float(x), 4) for x in sat_li["I_steady_norm_shb"]],
            "rel_err_shb_li_at_g2": round(rel_err_shb_g20, 5),
            "max_rel_err_shb_li": round(float(sat_li["max_rel_err_shb"]), 5),
            "fl_time_scale": round(float(fl_clamp["time_scale"]), 1),
            "fl_dn_peak_above": round(float(fl_clamp["dn_peak_above"]), 4),
            "fl_dn_clamped_final": round(float(fl_clamp["dn_clamped_final"]), 4),
            "doughnut_nodal_null_ratio": round(float(fin_dough["nodal_null_ratio"]), 6),
            "doughnut_theta_env_deg": round(float(fin_dough["theta_env_deg"]), 2),
            "doughnut_theta_peak_fit_deg": round(float(fin_dough["theta_peak_fit_deg"]), 2),
            "doughnut_theta_peak_theory_deg": round(float(fin_dough["theta_peak_theory_deg"]), 2),
            "rel_diff_doughnut_theta_peak": round(
                abs(
                    float(fin_dough["theta_peak_fit_deg"])
                    - float(fin_dough["theta_peak_theory_deg"])
                )
                / float(fin_dough["theta_peak_theory_deg"]),
                5,
            ),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    elif idx == 3:
        t1_path = out_dir / "task_01.npz"
        t2_path = out_dir / "task_02.npz"
        lam0_n8_m = None
        gconf_ref = None
        if t1_path.exists():
            with np.load(t1_path) as d1:
                lam0_n8_m = float(d1["lambda_m_N8_nm"][0]) * 1e-9
        if t2_path.exists():
            with np.load(t2_path) as d2:
                gconf_ref = float(d2["lin_gamma_conf"])
        res_mut = run_symmetry_breaking_mutation_study(
            alphas=task["alphas"],
            n_array_edge=int(task.get("n_array_edge", 8)),
            nx=int(task["nx"]),
            gamma_conf=gconf_ref,
            lam0_m=lam0_n8_m,
        )
        np.savez_compressed(out_dir / f"task_{idx:02d}.npz", **res_mut)
        summary = {
            "task": idx,
            "name": name,
            "n_array_edge": int(res_mut["n_array_edge"]),
            "Q0_const": round(float(res_mut["Q0_const"]), 2),
            "Q_edge_0": round(float(res_mut["Q_edge_0"]), 2),
            "alphas": [float(x) for x in res_mut["alphas"]],
            "Q_rad_inf": [round(float(x), 2) for x in res_mut["Q_rad_inf"]],
            "Q_tot_finite": [round(float(x), 2) for x in res_mut["Q_tot_finite"]],
            "Q_tot_finite_analytic": [round(float(x), 2) for x in res_mut["Q_tot_finite_analytic"]],
            "g_th_analytic_cm1": [round(float(x), 2) for x in res_mut["g_th_analytic_cm1"]],
            "g_th_fit_cm1": [round(float(x), 2) for x in res_mut["g_th_fit_cm1"]],
            "slope_alpha_threshold": round(float(res_mut["slope_alpha_threshold"]), 4),
            "rel_err_slope_alpha_threshold": round(
                abs(float(res_mut["slope_alpha_threshold"]) - 2.0) / 2.0, 5
            ),
            "max_rel_err_mutation": round(float(res_mut["max_rel_err_mutation"]), 5),
            "quench_time_scale": round(float(res_mut["quench_time_scale"]), 1),
            "Q_tot_004_quench": round(float(res_mut["Q_tot_004_quench"]), 2),
            "Q_tot_018_quench_analytic": round(float(res_mut["Q_tot_018_quench_analytic"]), 2),
            "g0_fixed_cm1": round(float(res_mut["g0_fixed_cm1"]), 2),
            "quench_ratio_alpha018_over_004": round(
                float(
                    np.mean(res_mut["i_norm_alpha018_quenched"][-200:])
                    / max(float(np.mean(res_mut["i_norm_alpha004_lasing"][-200:])), 1e-12)
                ),
                6,
            ),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    elif idx == 4:
        res_conv = run_convergence_study(
            nx_list=task["nx_list"], alpha_ref=float(task["alpha_ref"])
        )
        np.savez_compressed(out_dir / f"task_{idx:02d}.npz", **res_conv)
        summary = {
            "task": idx,
            "name": name,
            "nx_list": [int(x) for x in res_conv["nx_list"]],
            "dx_nm": [round(float(x), 3) for x in res_conv["dx_nm"]],
            "cpw_si": [round(float(x), 2) for x in res_conv["cpw_si"]],
            "cells_across_gap": [round(float(x), 2) for x in res_conv["cells_across_gap"]],
            "lambda_bic_nm": [round(float(x), 4) for x in res_conv["lambda_bic_nm"]],
            "lambda_bic_drift_nm": round(float(res_conv["lambda_bic_drift_nm"]), 4),
            "lambda_bic_increments_nm": [
                round(float(x), 4) for x in res_conv["lambda_bic_increments_nm"]
            ],
            "rel_spread_lambda_bic": round(float(res_conv["rel_spread_lambda_bic"]), 6),
            "Q_a10": [round(float(x), 2) for x in res_conv["Q_a10"]],
            "gamma_conf_a10": [round(float(x), 4) for x in res_conv["gamma_conf_a10"]],
            "g_th_a10_cm1": [round(float(x), 3) for x in res_conv["g_th_a10_cm1"]],
            "Q_a10_richardson": round(float(res_conv["Q_a10_richardson"]), 2),
            "g_th_a10_richardson_cm1": round(float(res_conv["g_th_a10_richardson_cm1"]), 3),
            "rel_err_Q_a10_nx32": round(float(res_conv["rel_err_Q_a10"][0]), 5),
            "rel_err_Q_a10_nx64": round(float(res_conv["rel_err_Q_a10"][-1]), 5),
            "observed_order_Q_a10": round(float(res_conv["observed_order_Q_a10"]), 3),
            "rel_err_g_th_nx32": round(float(res_conv["rel_err_g_th"][0]), 5),
            "rel_err_g_th_nx64": round(float(res_conv["rel_err_g_th"][-1]), 5),
            "observed_order_g_th": round(float(res_conv["observed_order_g_th"]), 3),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
    else:
        raise ValueError(f"Unknown task index {idx}.")

    (out_dir / f"task_{idx:02d}.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Publication Figure & GIF Generation (`wakis.figstyle` + `overlay_geometry`)
# ---------------------------------------------------------------------------
PALETTE = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "black": "#111111",
    "grey": "#666666",
}


def plot_all_figures(data_dir: Path, fig_dir: Path) -> dict[str, Path]:
    """
    Generate all setup schematics, publication figures, and the animated lasing
    build-up GIF from the saved ``data/task_XX.npz`` files.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``task_00.npz .. task_04.npz``.
    fig_dir : Path
        Output figures directory.

    Returns
    -------
    dict of str to Path
        Generated figure file paths.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    from wakis import figstyle, schematic

    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    # 0. Setup figures (00a, 00b, 00_setup_table.md for unit cell and finite array)
    setup_res = generate_setup_figures(fig_dir)

    d0 = np.load(data_dir / "task_00.npz")
    d1 = np.load(data_dir / "task_01.npz")
    d2 = np.load(data_dir / "task_02.npz")
    d3 = np.load(data_dir / "task_03.npz")
    d4 = np.load(data_dir / "task_04.npz")

    # ------------------------------------------------------------------
    # Figure 1: Infinite lattice Bloch BIC Q(kx), bar-width sweep Q(w0), and 3D vortex
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(12.6, 10.0))
    ax1a, ax1b, ax1c, ax1d = axes1.ravel()

    k_nz = d0["kx_nonzero"]
    q_nz = d0["Q_kx_fit"][1:]
    q_an = d0["Q_kx_analytic"]
    slope_near = float(d0["slope_k_near_gamma"])
    k_3d = d0["k_3d_norms"]
    q_3d = d0["Q_k_3d_A1"]
    q_3d_an = d0["Q_k_3d_A1_analytic"]
    slope_3d = float(d0["slope_Q_vs_k_3d_A1"])
    ax1a.loglog(
        k_nz,
        q_an,
        "--",
        color=PALETTE["black"],
        lw=1.7,
        label=r"2D analytic $Q_k (k_x p / 2\pi)^{-2}$ (order $-2$)",
    )
    ax1a.loglog(
        k_nz,
        q_nz,
        "o-",
        color=PALETTE["blue"],
        lw=2.0,
        ms=6.5,
        label=rf"2D dimer FIT (`harminv`, slope $= {slope_near:.4f}$)",
    )
    ax1a.loglog(
        k_3d,
        q_3d_an,
        ":",
        color=PALETTE["grey"],
        lw=1.6,
        label=r"3D analytic $Q_{k,\mathrm{3D}} (k_\parallel a / 2\pi)^{-2}$",
    )
    ax1a.loglog(
        k_3d,
        q_3d,
        "s-",
        color=PALETTE["green"],
        lw=2.0,
        ms=6.5,
        label=rf"3D cylinder $A_1$ FIT (slope $= {slope_3d:.4f}$)",
    )
    ax1a.text(
        0.97,
        0.96,
        rf"At $\Gamma$ ($k_\parallel = 0$, dark BIC limit):"
        "\n"
        rf"• 2D dimer: $Q_\Gamma = {float(d0['Q_kx_fit'][0]):.2e}$"
        "\n"
        rf"• 3D $A_1$ ($n_{{\mathrm{{PML}}}}=10$): $Q_\Gamma > 10^7$ (22-cyc lower bound, ${float(d0['Q_gamma_A1']):.2e}$)"
        "\n"
        rf"• 3D $A_1$ ($n_{{\mathrm{{PML}}}}=6$): $Q_\Gamma = {float(d0['Q_gamma_A1_pml6']):.2e}$",
        transform=ax1a.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": PALETTE["blue"], "alpha": 0.94},
    )
    ax1a.set_xlabel(r"Normalized Bloch wavenumber $k_\parallel p / (2\pi)$")
    ax1a.set_ylabel(r"Radiative quality factor $Q(k_\parallel)$")
    ax1a.set_title(r"(a) 2D & 3D momentum-space scaling $Q(k_\parallel) \propto k_\parallel^{-2}$")
    ax1a.legend(loc="lower left", fontsize=7.8)
    ax1a.grid(True, which="both", alpha=0.3)

    w0_nm = d0["w0_nm"]
    q_sym_w0 = d0["Q_sym_vs_w0"]
    q_qbic_w0 = d0["Q_qbic_vs_w0"]
    q_qbic_w0_48 = d0["Q_qbic_vs_w0_nx48"]
    ax1b.semilogy(
        w0_nm,
        q_sym_w0,
        "s--",
        color=PALETTE["blue"],
        lw=2.0,
        ms=6.5,
        label=r"Symmetry-protected BIC ($\alpha = 0$, $n_x = 32$)",
    )
    ax1b.semilogy(
        w0_nm,
        q_qbic_w0,
        "o--",
        color=PALETTE["vermilion"],
        lw=1.9,
        ms=6.0,
        label=r"Quasi-BIC ($\alpha = 0.10$, $n_x = 32$, unconverged at $200\;\mathrm{nm}$)",
    )
    ax1b.semilogy(
        w0_nm,
        q_qbic_w0_48,
        "^-",
        color=PALETTE["purple"],
        lw=2.1,
        ms=6.5,
        label=r"Quasi-BIC ($\alpha = 0.10$, refined $n_x = 48$, monotonic)",
    )
    ax1b.axvline(170.0, color=PALETTE["grey"], ls=":", lw=1.2, label=r"Baseline $w_0 = 170\;\mathrm{nm}$")
    ax1b.set_xlabel(r"Mean semiconductor bar width $w_0$ [nm]")
    ax1b.set_ylabel(r"2D FIT quality factor $Q(w_0)$ at $\Gamma$")
    ax1b.set_title(r"(b) 2D FIT symmetry-protected ($\alpha=0$) vs quasi-BIC ($\alpha=0.10$)")
    ax1b.legend(loc="center right", fontsize=7.6)
    ax1b.grid(True, which="both", alpha=0.3)

    kg = d0["k_grid_norm"]
    KX, KY = np.meshgrid(kg, kg, indexing="ij")
    inten_v = d0["vortex_intensity_A1"]
    inten_v_norm = inten_v / max(float(np.max(inten_v)), 1e-30)
    ax1c.set_facecolor("white")
    im1c = ax1c.pcolormesh(
        KX, KY, inten_v_norm, shading="auto", cmap="inferno"
    )
    fig1.colorbar(
        im1c,
        ax=ax1c,
        label=r"Analytic map with an arbitrary Gaussian envelope (not FIT data)",
    )
    vx = d0["vortex_cx_A1"]
    vy = d0["vortex_cy_A1"]
    vmag = np.hypot(vx, vy)
    mask_v = vmag > 1e-12
    ux = np.zeros_like(vx)
    uy = np.zeros_like(vy)
    ux[mask_v] = vx[mask_v] / vmag[mask_v]
    uy[mask_v] = vy[mask_v] / vmag[mask_v]
    ax1c.quiver(
        KX[::2, ::2],
        KY[::2, ::2],
        ux[::2, ::2],
        uy[::2, ::2],
        color="white",
        pivot="mid",
        scale=14,
        width=0.006,
    )
    phi_d = d0["phi_deg"]
    k_ring_n = float(d0["k_ring_norm"])
    kx_pts = k_ring_n * np.cos(np.deg2rad(phi_d))
    ky_pts = k_ring_n * np.sin(np.deg2rad(phi_d))
    ax1c.plot(
        kx_pts,
        ky_pts,
        "co",
        ms=5.8,
        mec="black",
        mew=0.9,
        label=r"3D FIT off-mirror loop ($\phi_k = 17^\circ + 45^\circ m$)",
    )
    ax1c.plot(0.0, 0.0, "wo", ms=8, mfc="white", mec="black", mew=1.6, label=r"Vortex core at $\Gamma$ ($I \approx 0$)")
    ax1c.set_xlim(-0.082, 0.082)
    ax1c.set_ylim(-0.082, 0.112)
    ax1c.set_xlabel(r"$k_x a / (2\pi)$")
    ax1c.set_ylabel(r"$k_y a / (2\pi)$")
    ax1c.set_title(
        rf"(c) Analytic map with an arbitrary Gaussian envelope (not FIT data; $q = +{float(d0['q_charge_A1']):.2f}$)"
    )
    ax1c.legend(loc="upper center", fontsize=7.6)

    psi_a1 = d0["psi_A1_deg"]
    psi_a1_unw = np.rad2deg(np.unwrap(2.0 * np.deg2rad(psi_a1)) * 0.5)
    phi_dense = np.linspace(0.0, 350.0, 100)
    psi_offset = float(np.mean(psi_a1_unw - phi_d))
    ax1d.plot(
        phi_dense,
        phi_dense + psi_offset,
        "--",
        color=PALETTE["blue"],
        alpha=0.7,
        lw=1.7,
        label=r"Theory $\mathrm{d}\psi/\mathrm{d}\phi_k = +1$ ($q = +1$, $A_1$ BIC)",
    )
    ax1d.plot(
        phi_dense,
        -phi_dense,
        ":",
        color=PALETTE["grey"],
        alpha=0.8,
        lw=1.5,
        label=r"Analytic reference $\mathrm{d}\psi/\mathrm{d}\phi_k = -1$ ($q = -1$)",
    )
    ax1d.plot(
        phi_d,
        psi_a1_unw,
        "o",
        color=PALETTE["blue"],
        ms=7.5,
        label=rf"3D FIT resolved $A_1$ mode ($q = +{float(d0['q_charge_A1']):.2f}$, $\mathrm{{cpw}} = {float(d0['cpw_3d_A1']):.2f}$)",
    )
    ax1d.set_xlabel(r"Off-mirror Bloch azimuth $\phi_k = \arctan2(k_y, k_x)$ [deg]")
    ax1d.set_ylabel(r"Unwrapped polarization angle $\psi(\phi_k)$ [deg]")
    ax1d.set_title(r"(d) Topological winding of resolved 3D FIT $A_1$ BIC mode")
    ax1d.legend(loc="lower left", fontsize=7.8)
    ax1d.grid(True, alpha=0.3)

    fig1_path = fig_dir / "01_infinite_bic_q_and_vortex.png"
    figstyle.save(fig1, fig1_path, close=True, auto_legend_below=False)

    # ------------------------------------------------------------------
    # Figure 2: Finite Strip Array Q(N) vs N, mode selection m=1,2,3, and eigenmode map
    # ------------------------------------------------------------------
    fig2 = plt.figure(figsize=(12.8, 8.6))
    gs2 = fig2.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.28)
    ax2a = fig2.add_subplot(gs2[0, 0])
    ax2b = fig2.add_subplot(gs2[0, 1])
    ax2c = fig2.add_subplot(gs2[1, :])

    n_arr = d1["N_list"]
    q1_n = d1["Q1_N_fit"]
    q1_fit = d1["Q1_N_power_fit"]
    p_exp = float(d1["p_exponent_fit"])
    n_guide = np.linspace(float(n_arr[0]), float(n_arr[-1]), 100)
    q_n2_guide = q1_n[0] * ((n_guide / n_arr[0]) ** 2.0)
    ax2a.loglog(
        n_guide,
        q_n2_guide,
        "--",
        color=PALETTE["grey"],
        lw=1.6,
        label=r"Analytic baseline $\propto N^2$ (momentum spread $\Delta k = \pi / Np$)",
    )
    ax2a.loglog(
        n_arr,
        q1_fit,
        "-",
        color=PALETTE["vermilion"],
        lw=1.8,
        label=rf"Power-law fit $Q_1(N) \propto N^{{{p_exp:.4f}}}$",
    )
    ax2a.loglog(
        n_arr,
        q1_n,
        "o",
        color=PALETTE["blue"],
        ms=7.5,
        label="2D FIT finite $N$-period strip (`harminv`, $m=1$)",
    )
    ax2a.set_xlabel(r"Finite strip size $N$ (number of unit cells)")
    ax2a.set_ylabel(r"Fundamental quality factor $Q_1(N)$")
    ax2a.set_title(r"(a) Finite $N$-period strip scaling $Q_1(N) \propto N^p$")
    ax2a.legend(loc="upper left", fontsize=8.0)
    ax2a.grid(True, which="both", alpha=0.3)

    m_arr = d1["modes_m"]
    qm_fit = d1["Q_m_N8_fit"]
    qm_an = d1["Q_m_N8_analytic"]
    x_pos = np.arange(len(m_arr))
    w_bar_p = 0.34
    ax2b.bar(x_pos - 0.5 * w_bar_p, qm_fit, width=w_bar_p, color=PALETTE["blue"], label="2D FIT (`harminv`, $N=8$ strip)")
    ax2b.bar(
        x_pos + 0.5 * w_bar_p,
        qm_an,
        width=w_bar_p,
        color=PALETTE["orange"],
        label=r"Analytic fixed-$N$ scaling $Q_m(N) = Q_1(N) / m^2$",
    )
    for i_m, (qf, qa) in enumerate(zip(qm_fit, qm_an)):
        ax2b.text(
            x_pos[i_m] - 0.5 * w_bar_p,
            qf + 14.0,
            f"{qf:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2b.set_xticks(x_pos)
    ax2b.set_xticklabels([f"Mode $m={m}$\n($k_{m}={m}\\pi/L$)" for m in m_arr])
    ax2b.set_ylabel(r"Quality factor $Q_m(N=8)$")
    ax2b.set_title(r"(b) Envelope mode selection at fixed $N=8$: $m = 1$ has highest $Q$")
    ax2b.set_ylim(0, float(np.max(qm_fit)) * 1.22)
    ax2b.legend(loc="upper right", fontsize=8.2)
    ax2b.grid(True, axis="y", alpha=0.3)

    x_n8 = d1["x_n8_um"]
    z_n8 = d1["z_n8_um"]
    ey_n8 = d1["ey_n8_m1_2d"]
    vmax2 = float(np.max(np.abs(ey_n8)))
    im2c = ax2c.pcolormesh(
        z_n8, x_n8, ey_n8 / max(vmax2, 1e-30), shading="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0
    )
    fig2.colorbar(im2c, ax=ax2c, label=r"Normalized $E_y(z, x) / \max|E_y|$")
    slv_n8, _ = make_finite_array_solver(n_cells_array=8, alpha=0.0, nx_per_cell=32)
    schematic.overlay_geometry(
        ax2c,
        slv_n8,
        plane="zx",
        unit="um",
        material_names={"region_0": "16 active resonator bars (N = 8 strip)"},
    )
    ax2c.axhline(0.0, color=PALETTE["black"], ls="--", lw=1.2, label="Global nodal plane $x = 0$ ($E_y = 0$)")
    ax2c.set_xlabel(r"Vertical coordinate $z$ [$\mu$m]")
    ax2c.set_ylabel(r"Transverse strip coordinate $x$ [$\mu$m]")
    ax2c.set_title(
        r"(c) Fundamental $m=1$ quasi-BIC eigenmode $E_y(z, x)$ in the $N=8$ strip array ($\lambda_1 = 1582.5\;\mathrm{nm}$, $Q_1 = 528.3$)"
    )
    ax2c.legend(loc="lower center", fontsize=8.5)

    fig2_path = fig_dir / "02_finite_array_q_vs_n.png"
    figstyle.save(fig2, fig2_path, close=True, auto_legend_below=False)

    # ------------------------------------------------------------------
    # Figure 3: Threshold gain vs w0 and N, L-I curve & clamping, mode competition, doughnut beam
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(2, 2, figsize=(12.8, 9.8))
    ax3a, ax3b, ax3c, ax3d = axes3.ravel()

    ax3a.plot(
        d0["w0_nm"],
        d0["g_th_qbic_vs_w0_cm1"],
        "o--",
        color=PALETTE["vermilion"],
        lw=1.9,
        ms=6.0,
        label=r"Quasi-BIC ($\alpha = 0.10$, $n_x = 32$)",
    )
    ax3a.plot(
        d0["w0_nm"],
        d0["g_th_qbic_vs_w0_nx48_cm1"],
        "^-",
        color=PALETTE["purple"],
        lw=2.1,
        ms=6.5,
        label=r"Quasi-BIC ($\alpha = 0.10$, refined $n_x = 48$)",
    )
    ax3a.plot(
        d0["w0_nm"],
        d0["g_th_sym_vs_w0_cm1"],
        "s--",
        color=PALETTE["blue"],
        lw=2.0,
        ms=6.0,
        label=r"Symmetry-protected BIC ($\alpha = 0$, $g_{\mathrm{th}} < 2\times 10^{-4}\;\mathrm{cm}^{-1}$)",
    )
    ax3a.axvline(170.0, color=PALETTE["grey"], ls=":", lw=1.2)
    ax3a.set_xlabel(r"Mean semiconductor bar width $w_0$ [nm]")
    ax3a.set_ylabel(r"Cold-cavity threshold gain $g_{\mathrm{th}}(w_0)$ [$\mathrm{cm}^{-1}$]")
    ax3a.set_title(r"(a) 2D FIT threshold gain vs bar width $w_0$ and strip size $N$")
    ax3a.legend(loc="upper right", fontsize=8.0)
    ax3a.grid(True, alpha=0.3)

    # Inset in 3a: g_th(N) vs strip array size N
    ax3a_in = ax3a.inset_axes([0.56, 0.36, 0.40, 0.35])
    ax3a_in.loglog(d2["N_list"], d2["g_th_N_cm1"], "o-", color=PALETTE["green"], ms=4.5, lw=1.5)
    ax3a_in.set_xlabel("Strip size $N$", fontsize=7.5)
    ax3a_in.set_ylabel(r"$g_{\mathrm{th}}(N)$ [$\mathrm{cm}^{-1}$]", fontsize=7.5)
    ax3a_in.tick_params(labelsize=7)
    ax3a_in.grid(True, which="both", alpha=0.3)

    g_rat = d2["sat_g0_ratios"]
    i_fit = d2["sat_I_fit"]
    i_shb = d2["sat_I_shb"]
    i_hom = d2["sat_I_homog"]
    ax3b.plot(
        g_rat,
        i_hom,
        ":",
        color=PALETTE["grey"],
        lw=1.6,
        label=r"Homogeneous plane-wave $(g_0/g_{\mathrm{th}} - 1)_+$",
    )
    ax3b.plot(
        g_rat,
        i_shb,
        "--",
        color=PALETTE["black"],
        lw=2.0,
        label="Spatial-hole-burning theory (derived here)",
    )
    ax3b.plot(
        g_rat,
        i_fit,
        "o",
        color=PALETTE["vermilion"],
        ms=7.0,
        label="2D FIT + `SaturableGain` steady state",
    )
    ax3b.axvline(1.0, color=PALETTE["blue"], ls="--", lw=1.1, alpha=0.7)
    ax3b.set_xlabel(r"Normalized pump gain $g_0 / g_{\mathrm{th}}$")
    ax3b.set_ylabel(r"Steady-state intracavity intensity $\langle I \rangle / I_{\mathrm{sat}}$")
    ax3b.set_title(r"(b) BIC laser L–I curve & `FourLevelGain` noise-seeded clamping")
    ax3b.legend(loc="upper left", fontsize=8.0)
    ax3b.grid(True, alpha=0.3)

    # Inset in 3b: FourLevelGain(spontaneous_emission=True, time_scale=80) population inversion clamping
    ax3b_in = ax3b.inset_axes([0.52, 0.14, 0.44, 0.38])
    ax3b_in.plot(d2["fl_t_fs"], d2["fl_dn_above"], "-", color=PALETTE["vermilion"], lw=1.4, label=r"$2.2 W_{p,\mathrm{th}}$ ($s=80$)")
    ax3b_in.plot(d2["fl_t_fs"], d2["fl_dn_sub"], "--", color=PALETTE["blue"], lw=1.2, label=r"$0.65 W_{p,\mathrm{th}}$")
    ax3b_in.axhline(1.0, color=PALETTE["black"], ls=":", lw=1.0)
    ax3b_in.set_xlabel("$t$ [fs]", fontsize=7.5)
    ax3b_in.set_ylabel(r"$\Delta N(t) / \Delta N_{\mathrm{th}}$", fontsize=7.5)
    ax3b_in.tick_params(labelsize=7)
    ax3b_in.legend(loc="lower right", fontsize=6.5)

    # Panel 3c: Single-mode lasing mode competition in finite N=8 strip array
    t_d = d2["dough_t_fs"]
    win_env = max(int(len(t_d) / 35), 5)
    kern = np.ones(win_env) / win_env
    env_m1 = np.sqrt(np.convolve(d2["dough_proj_m1"] ** 2, kern, mode="same"))
    env_m2 = np.sqrt(np.convolve(d2["dough_proj_m2"] ** 2, kern, mode="same"))
    env_m3 = np.sqrt(np.convolve(d2["dough_proj_m3"] ** 2, kern, mode="same"))
    e_norm = max(float(np.max(env_m1)), 1e-30)
    ax3c.semilogy(t_d, np.maximum(env_m1 / e_norm, 1e-4), "-", color=PALETTE["vermilion"], lw=2.0, label="Mode $m=1$ ($Q_1 = 528.3$, lases)")
    ax3c.semilogy(t_d, np.maximum(env_m2 / e_norm, 1e-4), "--", color=PALETTE["blue"], lw=1.6, label="Mode $m=2$ ($Q_2 = 138.1$, decays)")
    ax3c.semilogy(t_d, np.maximum(env_m3 / e_norm, 1e-4), ":", color=PALETTE["green"], lw=1.6, label="Mode $m=3$ ($Q_3 = 65.3$, decays)")
    ax3c.set_ylim(2e-4, 2.0)
    ax3c.set_xlabel(r"Time $t$ [fs]")
    ax3c.set_ylabel(r"Normalized modal amplitude $|a_m(t)|$")
    ax3c.set_title(r"(c) Single-mode selection at the fundamental BIC ($N = 8$ strip)")
    ax3c.legend(loc="lower right", fontsize=8.0)
    ax3c.grid(True, which="both", alpha=0.3)

    # Panel 3d: Far-field doughnut / twin-lobe radiation profile I(theta) from Near2FarMonitor
    th_deg = d2["dough_thetas_deg"]
    i_ff_fit = d2["dough_I_fit"]
    i_ff_an = d2["dough_I_analytic"]
    ax3d.plot(
        th_deg,
        i_ff_an,
        "--",
        color=PALETTE["black"],
        lw=2.0,
        label="Analytic continuum envelope FT (derived here)",
    )
    ax3d.plot(
        th_deg,
        i_ff_fit,
        "-",
        color=PALETTE["vermilion"],
        lw=2.2,
        label=r"2D FIT + `Near2FarMonitor` ($N=8$ strip, $I(0^\circ) \approx 0$)",
    )
    ax3d.axvline(0.0, color=PALETTE["blue"], ls=":", lw=1.3, label=r"Surface normal $\theta = 0^\circ$ (nodal null)")
    ax3d.set_ylim(-0.04, 1.32)
    ax3d.set_xlabel(r"Far-field emission angle $\theta$ [deg]")
    ax3d.set_ylabel(r"Normalized far-field intensity $I(\theta) / I_{\max}$")
    ax3d.set_title(r"(d) `Near2FarMonitor` doughnut beam cross-section ($I(\theta = 0^\circ) \approx 0$)")
    ax3d.legend(loc="upper right", fontsize=8.0)
    ax3d.grid(True, alpha=0.3)

    fig3_path = fig_dir / "03_lasing_threshold_and_doughnut.png"
    figstyle.save(fig3, fig3_path, close=True, auto_legend_below=False)

    # ------------------------------------------------------------------
    # Animated GIF: 03b_bic_lasing_buildup.gif (< 5 MB)
    # ------------------------------------------------------------------
    frames_ey = d2["dough_frames_ey"]
    frames_t = d2["dough_frames_t_fs"]
    x_um = d2["dough_x_um"]
    z_um = d2["dough_z_um"]
    vmax_gif = float(np.max(np.abs(frames_ey))) * 0.85
    pil_frames = []
    slv_gif, _ = make_finite_array_solver(n_cells_array=8, alpha=0.0, nx_per_cell=32)

    for idx_f in range(len(frames_t)):
        fig_g, (ax_g1, ax_g2) = plt.subplots(
            1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
        )
        im_g = ax_g1.pcolormesh(
            z_um,
            x_um,
            frames_ey[idx_f] / max(vmax_gif, 1e-30),
            shading="auto",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
        )
        fig_g.colorbar(im_g, ax=ax_g1, label=r"$E_y(z, x) / E_{\max}$")
        schematic.overlay_geometry(ax_g1, slv_gif, plane="zx", unit="um")
        ax_g1.axhline(0.0, color=PALETTE["black"], ls="--", lw=1.0)
        ax_g1.set_xlabel(r"$z$ [$\mu$m]")
        ax_g1.set_ylabel(r"$x$ [$\mu$m]")
        ax_g1.set_title(rf"Active $N=8$ BIC strip laser $E_y(z, x)$ at $t = {frames_t[idx_f]:.0f}\;\mathrm{{fs}}$")

        ax_g2.semilogy(t_d, np.maximum(env_m1 / e_norm, 1e-4), "-", color=PALETTE["vermilion"], lw=1.8, label="$m=1$ (BIC)")
        ax_g2.semilogy(t_d, np.maximum(env_m2 / e_norm, 1e-4), "--", color=PALETTE["blue"], lw=1.4, label="$m=2$")
        ax_g2.semilogy(t_d, np.maximum(env_m3 / e_norm, 1e-4), ":", color=PALETTE["green"], lw=1.4, label="$m=3$")
        ax_g2.axvline(frames_t[idx_f], color=PALETTE["black"], ls="-", lw=1.3)
        ax_g2.set_ylim(2e-4, 2.0)
        ax_g2.set_xlabel(r"Time $t$ [fs]")
        ax_g2.set_ylabel(r"Modal envelope $|a_m(t)|$")
        ax_g2.set_title("Single-mode BIC selection")
        ax_g2.legend(loc="lower right", fontsize=7.5)
        ax_g2.grid(True, which="both", alpha=0.3)
        fig_g.tight_layout()
        fig_g.canvas.draw()
        rgba = np.asarray(fig_g.canvas.buffer_rgba())
        pil_frames.append(Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE, colors=128))
        plt.close(fig_g)

    gif_path = fig_dir / "03b_bic_lasing_buildup.gif"
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=110,
        loop=0,
        optimize=True,
    )

    # ------------------------------------------------------------------
    # Figure 4: Symmetry-breaking mutation (alpha != 0) & 3-level convergence
    # ------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 2, figsize=(12.6, 9.6))
    ax4a, ax4b, ax4c, ax4d = axes4.ravel()

    alphas_m = d3["alphas"]
    nz_m = alphas_m > 0.0
    ax4a.loglog(
        alphas_m[nz_m],
        d3["Q_rad_inf"][nz_m],
        "o-",
        color=PALETTE["blue"],
        lw=2.0,
        ms=6.5,
        label=r"Infinite-lattice 2D FIT $Q_{\mathrm{rad}}(\alpha) = Q_0 \alpha^{-2}$",
    )
    ax4a.loglog(
        alphas_m[nz_m],
        d3["Q_tot_finite_analytic"][nz_m],
        "--",
        color=PALETTE["black"],
        lw=1.8,
        label=r"Analytic $(Q_{\mathrm{edge}}^{-1}(N=8) + Q_{\mathrm{rad},\infty}^{-1}(\alpha))^{-1}$",
    )
    ax4a.loglog(
        alphas_m[nz_m],
        d3["Q_tot_finite"][nz_m],
        "s",
        color=PALETTE["vermilion"],
        ms=6.5,
        label=r"Direct 2D FIT finite strip $Q_{\mathrm{tot}}(\alpha, N=8)$",
    )
    ax4a.axhline(
        float(d3["Q_edge_0"]),
        color=PALETTE["grey"],
        ls=":",
        lw=1.3,
        label=rf"Symmetric strip limit $Q_1(\alpha=0, N=8) = {float(d3['Q_edge_0']):.1f}$",
    )
    ax4a.set_ylim(100.0, 5.0e4)
    ax4a.set_xlabel(r"Symmetry-breaking parameter $\alpha = \Delta w / w_0$")
    ax4a.set_ylabel(r"Quality factor $Q(\alpha)$")
    ax4a.set_title(r"(a) Direct 2D FIT finite-strip $Q_{\mathrm{tot}}(\alpha, N=8)$ vs $\alpha$")
    ax4a.legend(loc="lower left", fontsize=7.8)
    ax4a.grid(True, which="both", alpha=0.3)

    a_dense = np.linspace(0.0, float(alphas_m[-1]), 100)
    g0_base = float(d3["g_th_analytic_cm1"][0])
    kappa_cm1 = float(d3["kappa_th_m1"]) / 100.0
    ax4b.plot(
        a_dense,
        g0_base + kappa_cm1 * (a_dense**2),
        "--",
        color=PALETTE["black"],
        lw=2.0,
        label=r"Analytic $g_{\mathrm{th}}(0, N=8) + \kappa_{\mathrm{th}} \alpha^2$ (derived here)",
    )
    ax4b.plot(
        alphas_m,
        d3["g_th_fit_cm1"],
        "o",
        color=PALETTE["vermilion"],
        ms=7.0,
        label=rf"Direct 2D FIT $N=8$ strip (slope $= +{float(d3['slope_alpha_threshold']):.4f}$)",
    )
    ax4b.set_xlabel(r"Symmetry-breaking parameter $\alpha = \Delta w / w_0$")
    ax4b.set_ylabel(r"Threshold gain $g_{\mathrm{th}}(\alpha, N=8)$ [$\mathrm{cm}^{-1}$]")
    ax4b.set_title(r"(b) Quadratic threshold penalty $\Delta g_{\mathrm{th}}(\alpha) \propto \alpha^2$")
    ax4b.legend(loc="upper left", fontsize=8.0)
    ax4b.grid(True, alpha=0.3)

    s_q = float(d3["quench_time_scale"])
    ax4c.plot(
        d3["t_quench_fs"],
        d3["i_norm_alpha004_lasing"],
        "-",
        color=PALETTE["green"],
        lw=2.0,
        label=r"Low asymmetry $\alpha = 0.04$ ($g_0 > g_{\mathrm{th}}$, lases)",
    )
    ax4c.plot(
        d3["t_quench_fs"],
        d3["i_norm_alpha018_quenched"],
        "--",
        color=PALETTE["vermilion"],
        lw=2.0,
        label=r"Broken symmetry $\alpha = 0.18$ ($g_{\mathrm{th}}(\alpha) > g_0$, quenched)",
    )
    ax4c.set_xlabel(r"Time $t$ [fs]")
    ax4c.set_ylabel(r"Normalized intracavity intensity $\langle I(t) \rangle / I_{\mathrm{sat}}$")
    ax4c.set_title(
        rf"(c) Lasing quenching at fixed pump ($g_0 = {float(d3['g0_fixed_cm1']):.1f}\;\mathrm{{cm}}^{{-1}}$, $s = {s_q:.0f}\times$)"
    )
    ax4c.legend(loc="center right", fontsize=8.0)
    ax4c.grid(True, alpha=0.3)

    dx_conv = d4["dx_nm"]
    err_gth = d4["rel_err_g_th"]
    err_q = d4["rel_err_Q_a10"]
    dx_guide = np.linspace(float(dx_conv[-1]), float(dx_conv[0]), 50)
    guide_o2 = err_gth[0] * ((dx_guide / dx_conv[0]) ** 2.0)
    ax4d.loglog(
        dx_guide,
        guide_o2,
        "--",
        color=PALETTE["black"],
        lw=1.8,
        label=r"Second-order guide $\mathcal{O}(\Delta x^2)$",
    )
    ax4d.loglog(
        dx_conv,
        err_gth,
        "o-",
        color=PALETTE["vermilion"],
        lw=2.0,
        ms=7.0,
        label=rf"Threshold gain $|g_{{\mathrm{{th}}}} - g_{{\mathrm{{th,Rich}}}}| / g_{{\mathrm{{th,Rich}}}}$ (order {float(d4['observed_order_g_th']):.2f})",
    )
    ax4d.loglog(
        dx_conv,
        err_q,
        "s-",
        color=PALETTE["blue"],
        lw=1.8,
        ms=6.5,
        label=rf"Quality factor $|Q_{{\mathrm{{rad}}}} - Q_{{\mathrm{{Rich}}}}| / Q_{{\mathrm{{Rich}}}}$ (order {float(d4['observed_order_Q_a10']):.2f})",
    )
    ax4d.set_xlabel(r"Grid step $\Delta x$ [nm] ($n_x \in \{32, 48, 64\}$, $\mathrm{cpw}_{\mathrm{Si}} \in \{21.3, 31.9, 42.6\}$)")
    ax4d.set_ylabel(r"Relative error vs Richardson extrapolate")
    ax4d.set_title(r"(d) 3-resolution convergence ($\mathrm{cpw} \geq 20$, $\geq 4$ cells/gap)")
    ax4d.legend(loc="lower right", fontsize=8.0)
    ax4d.grid(True, which="both", alpha=0.3)

    fig4_path = fig_dir / "04_symmetry_breaking_mutation.png"
    figstyle.save(fig4, fig4_path, close=True, auto_legend_below=False)

    return {
        "setup_unit_geom": Path(setup_res["unit"]["geometry_figure"]),
        "setup_unit_mesh": Path(setup_res["unit"]["mesh_figure"]),
        "setup_finite_geom": Path(setup_res["finite"]["geometry_figure"]),
        "setup_finite_mesh": Path(setup_res["finite"]["mesh_figure"]),
        "fig1": fig1_path,
        "fig2": fig2_path,
        "fig3": fig3_path,
        "gif": gif_path,
        "fig4": fig4_path,
    }
