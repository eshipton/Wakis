# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Analytical Fresnel/Airy transfer-matrix formulas, Wakis FIT simulation
runners, and plotting utilities for Tutorial 1.2 (`01-02_slab_rt`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps0
from scipy.constants import mu_0 as mu0

from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.estimate import estimate, preflight
from wakis.field_monitors import DFTMonitor, FluxMonitor
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.formulas import slab_reflection_transmission
from wakis.templates.shapes import assign_materials, resolve_material

ETA0 = float(np.sqrt(mu0 / eps0))
LAM0 = 1.0e-6  # Reference free-space wavelength [m] (1.0 um)
F0 = c0 / LAM0  # Reference center frequency [Hz] (~299.79 THz)
N_SLAB = 2.0  # Dielectric slab refractive index (eps_r = 4.0)
EPS_SLAB = N_SLAB**2


class PlanarSlabZ:
    """Planar dielectric slab ``z in [z_lo, z_hi]`` supporting subpixel ``fill`` and ``normal``.

    Implements the ``fill(x, y, z, hx, hy, hz)`` and ``normal(x, y, z)``
    interface expected by ``wakis.templates.shapes.assign_materials`` so both
    Kottke (2008) anisotropic subpixel smoothing (arithmetic averaging for
    tangential ``Ex, Ey``, harmonic averaging for normal ``Ez``) and staircase
    rounding can be compared on off-grid interfaces.

    Parameters
    ----------
    z_lo, z_hi : float
        Lower and upper slab face coordinates along z [m].
    material : float or str
        Slab relative permittivity or material key.
    """

    def __init__(self, z_lo: float, z_hi: float, material: float | str):
        if z_hi <= z_lo:
            raise ValueError("z_hi must be > z_lo.")
        self.z_lo = float(z_lo)
        self.z_hi = float(z_hi)
        self.material = resolve_material(material)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-1.0, -1.0, self.z_lo]), np.array([1.0, 1.0, self.z_hi])

    def fill(self, x, y, z, hx, hy, hz, n_sub: int = 16) -> np.ndarray:
        lo = np.maximum(z - 0.5 * hz, self.z_lo)
        hi = np.minimum(z + 0.5 * hz, self.z_hi)
        return np.clip(hi - lo, 0.0, None) / np.maximum(hz, 1e-30)

    def normal(self, x, y, z) -> np.ndarray:
        n = np.zeros((x.size, 3), dtype=float)
        z_mid = 0.5 * (self.z_lo + self.z_hi)
        n[:, 2] = np.where(np.asarray(z).ravel() >= z_mid, 1.0, -1.0)
        return n


def fresnel_single_interface(
    n1: float,
    n2: float,
    theta_inc_rad: np.ndarray | float,
    polarization: str = "s",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact Fresnel amplitude and power coefficients at a planar ``n1 -> n2`` interface.

    Parameters
    ----------
    n1, n2 : float
        Incident and transmitted refractive indices [dimensionless].
    theta_inc_rad : float or array_like
        Incident angle from the interface normal [rad].
    polarization : {'s', 'p', 'te', 'tm'}, optional
        Wave polarization ('s'/'te' for E perpendicular to plane of incidence,
        'p'/'tm' for E in the plane of incidence). Default is 's'.

    Returns
    -------
    r, t : ndarray
        Complex amplitude reflection and transmission coefficients.
    R, T : ndarray
        Power reflectance ``|r|^2`` and transmittance ``(n2 cos(theta2)/(n1 cos(theta1))) |t|^2``.
    """
    th1 = np.atleast_1d(np.asarray(theta_inc_rad, dtype=float))
    c1 = np.cos(th1)
    s2 = (float(n1) / float(n2)) * np.sin(th1)
    c2 = np.sqrt(np.maximum(1.0 - s2**2, 0.0) + 0j)
    pol = polarization.lower()
    if pol in ("s", "te"):
        r = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
        t = (2.0 * n1 * c1) / (n1 * c1 + n2 * c2)
    elif pol in ("p", "tm"):
        r = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
        t = (2.0 * n1 * c1) / (n2 * c1 + n1 * c2)
    else:
        raise ValueError(f"Unknown polarization {polarization!r}.")
    R = np.abs(r) ** 2
    T = np.real(n2 * c2 / np.maximum(n1 * c1, 1e-30)) * (np.abs(t) ** 2)
    return r, t, R, T


def slab_rt_oblique(
    eps_r: complex | float,
    thickness: float,
    frequency: np.ndarray | float,
    theta_inc_rad: float = 0.0,
    polarization: str = "s",
    n_out: float = 1.0,
) -> dict[str, np.ndarray]:
    """Exact transfer-matrix (Airy) reflection and transmission of a dielectric slab.

    At normal incidence (``theta_inc_rad == 0``), ``R`` and ``T`` match
    ``wakis.templates.formulas.slab_reflection_transmission`` identically.

    Parameters
    ----------
    eps_r : float or complex
        Slab relative permittivity.
    thickness : float
        Slab physical thickness ``d`` [m].
    frequency : float or array_like
        Frequencies [Hz].
    theta_inc_rad : float, optional
        Angle of incidence in surrounding medium ``n_out`` [rad]. Default is 0.0.
    polarization : {'s', 'p', 'te', 'tm'}, optional
        Polarization state. Default is 's'.
    n_out : float, optional
        Surrounding medium refractive index. Default is 1.0.

    Returns
    -------
    dict
        ``{'r': r, 't': t, 'R': R, 'T': T}`` complex amplitudes and power spectra.
    """
    freqs = np.atleast_1d(np.asarray(frequency, dtype=float))
    n_slab = np.sqrt(complex(eps_r))
    if n_slab.imag > 0:
        n_slab = -n_slab
    c1 = np.cos(float(theta_inc_rad))
    s1 = np.sin(float(theta_inc_rad))
    c2 = np.sqrt(1.0 - (float(n_out) * s1 / n_slab) ** 2)

    pol = polarization.lower()
    if pol in ("s", "te"):
        q1 = float(n_out) * c1
        q2 = n_slab * c2
    elif pol in ("p", "tm"):
        q1 = float(n_out) / c1
        q2 = n_slab / c2
    else:
        raise ValueError(f"Unknown polarization {polarization!r}.")

    delta = (2.0 * np.pi * freqs / c0) * n_slab * c2 * float(thickness)
    m11 = np.cos(delta)
    m22 = np.cos(delta)
    m12 = 1j * np.sin(delta) / q2
    m21 = 1j * q2 * np.sin(delta)
    den = q1 * m11 + (q1**2) * m12 + m21 + q1 * m22
    r = (q1 * m11 + (q1**2) * m12 - m21 - q1 * m22) / den
    t = 2.0 * q1 / den
    R = np.abs(r) ** 2
    T = np.abs(t) ** 2
    return {"r": r, "t": t, "R": R, "T": T}


def run_fabry_perot_and_oblique_task(cfg: dict) -> dict:
    """Run Task 0: Broadband Fabry-Perot etalon spectra (FluxMonitor & DFT) and oblique Bloch s/p sweep."""
    cpw_slab = int(cfg["cpw"])
    dz = (LAM0 / N_SLAB) / cpw_slab
    n_pml = 10
    d_fp = 1.25 * (LAM0 / N_SLAB)  # 625 nm slab spanning multiple Fabry-Perot fringes
    z_lo = 1.00 * LAM0
    z_hi = z_lo + d_fp
    nz = int(round((z_hi + 0.85 * LAM0) / dz)) + 2 * n_pml
    x = np.array([0.0, dz])
    y = np.array([0.0, dz])
    z = np.arange(nz + 1) * dz

    z_src = float((n_pml + int(round(0.35 * LAM0 / dz))) * dz)
    z_refl = float((n_pml + int(round(0.68 * LAM0 / dz))) * dz)
    z_tran = float(z_hi + 0.35 * LAM0)

    freqs = np.linspace(0.60 * F0, 1.40 * F0, 41)
    pulse = GaussianSource(F0, fwidth=0.38 * F0, cutoff=4.5)

    # 1. Empty-domain reference run (incident flux & incident DFT field)
    g_ref = GridFIT3D(x=x, y=y, z=z, verbose=0)
    s_ref = SolverFIT3D(
        g_ref,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    nt = int(np.ceil((pulse.end_time + 18.0 * LAM0 / c0) / s_ref.dt))
    est = estimate((g_ref.Nx, g_ref.Ny, g_ref.Nz), steps=nt, cpml=True)
    preflight(est, "laptop")

    src_ref = CurrentSource(
        pulse,
        component="x",
        center=(0.5 * dz, 0.5 * dz, z_src),
        size=(dz, dz, 0.0),
        amplitude=1.0,
    )
    flux_r_ref = FluxMonitor(freqs, center=(0.5 * dz, 0.5 * dz, z_refl), size=(dz, dz, 0.0))
    flux_t_ref = FluxMonitor(freqs, center=(0.5 * dz, 0.5 * dz, z_tran), size=(dz, dz, 0.0))
    dft_r_ref = DFTMonitor(freqs, ["Ex"], center=(0.5 * dz, 0.5 * dz, z_refl), size=(0.0, 0.0, 0.0))
    dft_t_ref = DFTMonitor(freqs, ["Ex"], center=(0.5 * dz, 0.5 * dz, z_tran), size=(0.0, 0.0, 0.0))

    iz_r = int(np.argmin(np.abs(s_ref.z - z_refl)))
    iz_t = int(np.argmin(np.abs(s_ref.z - z_tran)))
    probe_r_ref = np.empty(nt, dtype=float)
    for n in range(nt):
        t = n * s_ref.dt
        src_ref.update(s_ref, t)
        s_ref.one_step()
        for m in (flux_r_ref, flux_t_ref, dft_r_ref, dft_t_ref):
            m.update(s_ref, t)
        probe_r_ref[n] = float(s_ref.E[0, 0, iz_r, "x"])

    # 2. Dielectric slab run (total flux & total DFT field + waterfall & movie frames)
    g_slab = GridFIT3D(x=x, y=y, z=z, verbose=0)
    s_slab = SolverFIT3D(
        g_slab,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    assign_materials(s_slab, [PlanarSlabZ(z_lo, z_hi, EPS_SLAB)], F0, averaging="kottke")
    src_slab = CurrentSource(
        pulse,
        component="x",
        center=(0.5 * dz, 0.5 * dz, z_src),
        size=(dz, dz, 0.0),
        amplitude=1.0,
    )
    flux_r_slab = FluxMonitor(freqs, center=(0.5 * dz, 0.5 * dz, z_refl), size=(dz, dz, 0.0))
    flux_t_slab = FluxMonitor(freqs, center=(0.5 * dz, 0.5 * dz, z_tran), size=(dz, dz, 0.0))
    dft_r_slab = DFTMonitor(freqs, ["Ex"], center=(0.5 * dz, 0.5 * dz, z_refl), size=(0.0, 0.0, 0.0))
    dft_t_slab = DFTMonitor(freqs, ["Ex"], center=(0.5 * dz, 0.5 * dz, z_tran), size=(0.0, 0.0, 0.0))

    probe_r_tot = np.empty(nt, dtype=float)
    probe_t_tot = np.empty(nt, dtype=float)
    wf_frames = []
    wf_times = []
    movie_frames = []
    movie_times = []
    stride_wf = max(1, nt // 150)
    stride_mov = max(1, nt // 48)

    for n in range(nt):
        t = n * s_slab.dt
        src_slab.update(s_slab, t)
        s_slab.one_step()
        for m in (flux_r_slab, flux_t_slab, dft_r_slab, dft_t_slab):
            m.update(s_slab, t)
        ex_z = np.asarray(s_slab.E[0, 0, :-1, "x"], dtype=float)
        probe_r_tot[n] = ex_z[iz_r]
        probe_t_tot[n] = ex_z[iz_t]
        if n % stride_wf == 0:
            wf_frames.append(ex_z.copy())
            wf_times.append(t)
        if n % stride_mov == 0 and len(movie_frames) < 34:
            movie_frames.append(ex_z.copy())
            movie_times.append(t)

    phi_r0 = flux_r_ref.get_flux()
    phi_t0 = flux_t_ref.get_flux()
    phi_r1 = flux_r_slab.get_flux()
    phi_t1 = flux_t_slab.get_flux()

    R_flux = 1.0 - phi_r1 / phi_r0
    T_flux = phi_t1 / phi_t0
    # Unnormalized raw ratio without subtracting reference run on reflection plane:
    R_wrong_no_sub = np.abs(phi_r1 / phi_r0)

    # Complex reflection & transmission amplitudes from DFT field subtraction
    e_r0 = dft_r_ref.get_field("Ex")[0][:, 0, 0, 0]
    e_t0 = dft_t_ref.get_field("Ex")[0][:, 0, 0, 0]
    e_r1 = dft_r_slab.get_field("Ex")[0][:, 0, 0, 0]
    e_t1 = dft_t_slab.get_field("Ex")[0][:, 0, 0, 0]
    R_dft = np.abs((e_r1 - e_r0) / e_r0) ** 2
    T_dft = np.abs(e_t1 / e_t0) ** 2

    R_exact, T_exact = slab_reflection_transmission(EPS_SLAB, d_fp, freqs)
    fp_rms_err = float(np.sqrt(np.mean((R_flux - R_exact) ** 2)))
    fp_energy_err = float(np.max(np.abs(R_flux + T_flux - 1.0)))

    # 3. Oblique-incidence Bloch s/TE and p/TM sweep (including Brewster's angle)
    d_ob = 0.375 * (LAM0 / N_SLAB)
    z_hi_ob = z_lo + d_ob
    z_tran_ob = float(z_hi_ob + 0.30 * LAM0)
    nz_ob = int(round((z_hi_ob + 0.70 * LAM0) / dz)) + 2 * n_pml
    z_ob = np.arange(nz_ob + 1) * dz
    brewster_deg = float(np.rad2deg(np.arctan(N_SLAB)))
    thetas_deg = np.array([0.0, 20.0, 35.0, 50.0, brewster_deg], dtype=float)
    R_ob_sim = {"s": np.zeros(len(thetas_deg)), "p": np.zeros(len(thetas_deg))}
    T_ob_sim = {"s": np.zeros(len(thetas_deg)), "p": np.zeros(len(thetas_deg))}
    R_ob_th = {"s": np.zeros(len(thetas_deg)), "p": np.zeros(len(thetas_deg))}
    T_ob_th = {"s": np.zeros(len(thetas_deg)), "p": np.zeros(len(thetas_deg))}

    for idx_th, th_deg in enumerate(thetas_deg):
        th_rad = float(np.deg2rad(th_deg))
        fw = 0.055 * F0 if th_deg > 55.0 else (0.11 * F0 if th_deg > 30.0 else 0.22 * F0)
        p_ob = GaussianSource(F0, fwidth=fw, cutoff=4.2)
        kx = (2.0 * np.pi / LAM0) * np.sin(th_rad)
        for pol, comp in (("s", "y"), ("p", "x")):
            g0 = GridFIT3D(x=x, y=y, z=z_ob, verbose=0)
            s0 = SolverFIT3D(
                g0,
                bc_low=["bloch", "periodic", "cpml"],
                bc_high=["bloch", "periodic", "cpml"],
                k_bloch=(kx, 0.0, 0.0),
                dtype=np.complex128,
                n_pml=n_pml,
                cfln=0.5,
                use_stl=False,
                verbose=0,
            )
            nt_ob = int(np.ceil((p_ob.end_time + 14.0 * LAM0 / (c0 * np.cos(th_rad) ** 2)) / s0.dt))
            src0 = CurrentSource(
                p_ob,
                component=comp,
                center=(0.5 * dz, 0.5 * dz, z_src),
                size=(dz, dz, 0.0),
                amplitude=1.0,
                amp_func=lambda X, Y, Z, kx=kx: np.exp(+1j * kx * X),
            )
            mr0 = FluxMonitor([F0], center=(0.5 * dz, 0.5 * dz, z_refl), size=(dz, dz, 0.0))
            mt0 = FluxMonitor([F0], center=(0.5 * dz, 0.5 * dz, z_tran_ob), size=(dz, dz, 0.0))
            s0.emsolve(nt_ob, source=[src0], monitors=[mr0, mt0])

            g1 = GridFIT3D(x=x, y=y, z=z_ob, verbose=0)
            s1 = SolverFIT3D(
                g1,
                bc_low=["bloch", "periodic", "cpml"],
                bc_high=["bloch", "periodic", "cpml"],
                k_bloch=(kx, 0.0, 0.0),
                dtype=np.complex128,
                n_pml=n_pml,
                cfln=0.5,
                use_stl=False,
                verbose=0,
            )
            assign_materials(s1, [PlanarSlabZ(z_lo, z_hi_ob, EPS_SLAB)], F0, averaging="kottke")
            src1 = CurrentSource(
                p_ob,
                component=comp,
                center=(0.5 * dz, 0.5 * dz, z_src),
                size=(dz, dz, 0.0),
                amplitude=1.0,
                amp_func=lambda X, Y, Z, kx=kx: np.exp(+1j * kx * X),
            )
            mr1 = FluxMonitor([F0], center=(0.5 * dz, 0.5 * dz, z_refl), size=(dz, dz, 0.0))
            mt1 = FluxMonitor([F0], center=(0.5 * dz, 0.5 * dz, z_tran_ob), size=(dz, dz, 0.0))
            s1.emsolve(nt_ob, source=[src1], monitors=[mr1, mt1])

            R_ob_sim[pol][idx_th] = float(1.0 - mr1.get_flux()[0] / mr0.get_flux()[0])
            T_ob_sim[pol][idx_th] = float(mt1.get_flux()[0] / mt0.get_flux()[0])
            th_dict = slab_rt_oblique(EPS_SLAB, d_ob, F0, th_rad, pol)
            R_ob_th[pol][idx_th] = float(th_dict["R"][0])
            T_ob_th[pol][idx_th] = float(th_dict["T"][0])

    cells_feat = d_ob / dz
    return {
        "name": cfg["name"],
        "cpw": cpw_slab,
        "averaging": "kottke",
        "dx_background_nm": (LAM0 / cpw_slab) * 1e9,
        "dx_min_nm": dz * 1e9,
        "min_feature_nm": d_ob * 1e9,
        "cells_across_feature": float(cells_feat),
        "n_cells": int(g_slab.Nx * g_slab.Ny * g_slab.Nz),
        "n_steps": int(2 * nt),
        "t_arr": np.arange(nt) * s_slab.dt,
        "probe_r_ref": probe_r_ref,
        "probe_r_tot": probe_r_tot,
        "probe_t_tot": probe_t_tot,
        "z_um": s_slab.z[:-1] * 1e6,
        "z_src_um": z_src * 1e6,
        "z_refl_um": z_refl * 1e6,
        "z_tran_um": z_tran * 1e6,
        "z_lo_um": z_lo * 1e6,
        "z_hi_um": z_hi * 1e6,
        "wf_frames": np.asarray(wf_frames, dtype=float),
        "wf_times": np.asarray(wf_times, dtype=float),
        "movie_frames": np.asarray(movie_frames, dtype=float),
        "movie_times": np.asarray(movie_times, dtype=float),
        "freqs": freqs,
        "R_flux": R_flux,
        "T_flux": T_flux,
        "R_dft": R_dft,
        "T_dft": T_dft,
        "R_exact": R_exact,
        "T_exact": T_exact,
        "R_wrong_no_sub": R_wrong_no_sub,
        "fp_rms_err": fp_rms_err,
        "fp_energy_err": fp_energy_err,
        "thetas_deg": thetas_deg,
        "R_ob_sim_s": R_ob_sim["s"],
        "T_ob_sim_s": T_ob_sim["s"],
        "R_ob_th_s": R_ob_th["s"],
        "T_ob_th_s": T_ob_th["s"],
        "R_ob_sim_p": R_ob_sim["p"],
        "T_ob_sim_p": T_ob_sim["p"],
        "R_ob_th_p": R_ob_th["p"],
        "T_ob_th_p": T_ob_th["p"],
        "brewster_Rp_sim": float(R_ob_sim["p"][-1]),
        "brewster_Tp_sim": float(T_ob_sim["p"][-1]),
        "brewster_Rs_sim": float(R_ob_sim["s"][-1]),
        "brewster_Rs_ex": float(R_ob_th["s"][-1]),
        "oblique_max_energy_err": float(
            max(
                np.max(np.abs(R_ob_sim["s"] + T_ob_sim["s"] - 1.0)),
                np.max(np.abs(R_ob_sim["p"] + T_ob_sim["p"] - 1.0)),
            )
        ),
    }


def run_cpml_reflection_task(cfg: dict) -> dict:
    """Run Task 1: Spurious reflection vs CPML layer count, grading exponent, Mur ABC, and PEC wall."""
    cpw = int(cfg["cpw"])
    dz = LAM0 / cpw
    freqs = np.linspace(0.75 * F0, 1.25 * F0, 25)
    pulse = GaussianSource(F0, fwidth=0.35 * F0, cutoff=4.5)

    nz_ref = 360
    z_ref_ax = np.arange(nz_ref + 1) * dz
    iz_offset = 140
    iz_src = 24
    iz_mon = 44
    iz_wall = 60

    x = np.array([0.0, dz])
    y = np.array([0.0, dz])
    g0 = GridFIT3D(x=x, y=y, z=z_ref_ax, verbose=0)
    s0 = SolverFIT3D(
        g0,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=12,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    nt = int(np.ceil((pulse.end_time + 2.8 * ((iz_wall - iz_src) * dz) / c0) / s0.dt))
    src0 = CurrentSource(
        pulse,
        component="x",
        center=(0.5 * dz, 0.5 * dz, (iz_offset + iz_src) * dz),
        size=(dz, dz, 0.0),
        amplitude=1.0,
    )
    m0 = DFTMonitor(
        freqs,
        ["Ex"],
        center=(0.5 * dz, 0.5 * dz, (iz_offset + iz_mon) * dz),
        size=(0.0, 0.0, 0.0),
    )
    t_trace_ref = np.empty(nt, dtype=float)
    for n in range(nt):
        t = n * s0.dt
        src0.update(s0, t)
        s0.one_step()
        m0.update(s0, t)
        t_trace_ref[n] = float(s0.E[0, 0, iz_offset + iz_mon, "x"])
    e_inc = m0.get_field("Ex")[0][:, 0, 0, 0]

    def _eval_bc(bc_hi: str, n_p: int, p_exp: int):
        nz_t = iz_wall + (n_p if bc_hi == "cpml" else 0)
        zt = np.arange(nz_t + 1) * dz
        gt = GridFIT3D(x=x, y=y, z=zt, verbose=0)
        bc_lo = "abc" if bc_hi == "abc" else "cpml"
        st = SolverFIT3D(
            gt,
            bc_low=["periodic", "periodic", bc_lo],
            bc_high=["periodic", "periodic", bc_hi],
            n_pml=n_p,
            pml_exp=p_exp,
            cfln=0.5,
            use_stl=False,
            verbose=0,
        )
        srct = CurrentSource(
            pulse,
            component="x",
            center=(0.5 * dz, 0.5 * dz, iz_src * dz),
            size=(dz, dz, 0.0),
            amplitude=1.0,
        )
        mt = DFTMonitor(
            freqs,
            ["Ex"],
            center=(0.5 * dz, 0.5 * dz, iz_mon * dz),
            size=(0.0, 0.0, 0.0),
        )
        tr = np.empty(nt, dtype=float)
        for n in range(nt):
            t = n * st.dt
            srct.update(st, t)
            st.one_step()
            mt.update(st, t)
            tr[n] = float(st.E[0, 0, iz_mon, "x"])
        e_tot = mt.get_field("Ex")[0][:, 0, 0, 0]
        r_pow = np.abs((e_tot - e_inc) / e_inc) ** 2
        r_db = 10.0 * np.log10(np.maximum(r_pow, 1e-16))
        return r_db, tr - t_trace_ref

    r_db_pec, tr_refl_pec = _eval_bc("pec", 8, 3)
    r_db_abc, tr_refl_abc = _eval_bc("abc", 8, 3)

    n_pml_list = np.array([4, 6, 8, 10, 12, 16], dtype=int)
    r_db_vs_npml_spec = np.empty((len(n_pml_list), len(freqs)), dtype=float)
    r_db_vs_npml_f0 = np.empty(len(n_pml_list), dtype=float)
    r_db_vs_npml_mean = np.empty(len(n_pml_list), dtype=float)
    tr_refl_cpml10 = np.zeros(nt, dtype=float)

    for idx_n, n_p in enumerate(n_pml_list):
        r_db, tr_d = _eval_bc("cpml", int(n_p), 3)
        r_db_vs_npml_spec[idx_n] = r_db
        r_db_vs_npml_f0[idx_n] = float(r_db[len(freqs) // 2])
        r_db_vs_npml_mean[idx_n] = float(10.0 * np.log10(np.mean(10.0 ** (r_db / 10.0))))
        if int(n_p) == 10:
            tr_refl_cpml10 = tr_d

    exp_list = np.array([2, 3, 4], dtype=int)
    r_db_vs_exp_f0 = np.empty(len(exp_list), dtype=float)
    r_db_vs_exp_spec = np.empty((len(exp_list), len(freqs)), dtype=float)
    for idx_e, p_e in enumerate(exp_list):
        r_db, _ = _eval_bc("cpml", 10, int(p_e))
        r_db_vs_exp_spec[idx_e] = r_db
        r_db_vs_exp_f0[idx_e] = float(r_db[len(freqs) // 2])

    return {
        "name": cfg["name"],
        "cpw": cpw,
        "averaging": "kottke",
        "dx_background_nm": dz * 1e9,
        "dx_min_nm": dz * 1e9,
        "min_feature_nm": (4.0 * dz) * 1e9,
        "cells_across_feature": 4.0,
        "n_cells": int(nz_ref),
        "n_steps": int(nt * 11),
        "t_arr": np.arange(nt) * s0.dt,
        "t_trace_ref": t_trace_ref,
        "tr_refl_pec": tr_refl_pec,
        "tr_refl_abc": tr_refl_abc,
        "tr_refl_cpml10": tr_refl_cpml10,
        "freqs": freqs,
        "r_db_pec": r_db_pec,
        "r_db_abc": r_db_abc,
        "n_pml_list": n_pml_list,
        "r_db_vs_npml_spec": r_db_vs_npml_spec,
        "r_db_vs_npml_f0": r_db_vs_npml_f0,
        "r_db_vs_npml_mean": r_db_vs_npml_mean,
        "exp_list": exp_list,
        "r_db_vs_exp_spec": r_db_vs_exp_spec,
        "r_db_vs_exp_f0": r_db_vs_exp_f0,
    }


def _eval_offgrid_slab(cpw_slab: int, averaging: str) -> dict:
    """Evaluate normal-incidence R(f), T(f) and permittivity profiles for an off-grid slab."""
    dz = (LAM0 / N_SLAB) / cpw_slab
    n_pml = max(10, cpw_slab // 3)
    d_slab = 0.737 * (LAM0 / N_SLAB)  # 368.5 nm off-grid thickness
    z_lo = 1.137 * LAM0
    z_hi = z_lo + d_slab
    nz = int(round(2.8 * LAM0 / dz)) + 2 * n_pml
    x = np.array([0.0, dz])
    y = np.array([0.0, dz])
    z = np.arange(nz + 1) * dz

    pulse = GaussianSource(F0, fwidth=0.35 * F0, cutoff=4.5)
    z_src = float((n_pml + int(round(0.35 * LAM0 / dz))) * dz)
    z_refl = float((n_pml + int(round(0.75 * LAM0 / dz))) * dz)
    z_tran = float((n_pml + int(round((z_hi - n_pml * dz + 0.35 * LAM0) / dz))) * dz)
    freqs = np.linspace(0.75 * F0, 1.25 * F0, 31)

    g_ref = GridFIT3D(x=x, y=y, z=z, verbose=0)
    s_ref = SolverFIT3D(
        g_ref,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    nt = int(np.ceil((pulse.end_time + 18.0 * LAM0 / c0) / s_ref.dt))
    src0 = CurrentSource(
        pulse,
        component="x",
        center=(0.5 * dz, 0.5 * dz, z_src),
        size=(dz, dz, 0.0),
        amplitude=1.0,
    )
    mr0 = FluxMonitor(freqs, center=(0.5 * dz, 0.5 * dz, z_refl), size=(dz, dz, 0.0))
    mt0 = FluxMonitor(freqs, center=(0.5 * dz, 0.5 * dz, z_tran), size=(dz, dz, 0.0))
    s_ref.emsolve(nt, source=[src0], monitors=[mr0, mt0])

    g = GridFIT3D(x=x, y=y, z=z, verbose=0)
    s = SolverFIT3D(
        g,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    assign_materials(s, [PlanarSlabZ(z_lo, z_hi, EPS_SLAB)], F0, averaging=averaging)

    ieps_x = np.asarray(s.ieps[0, 0, :-1, "x"], dtype=float)
    ieps_z = np.asarray(s.ieps[0, 0, :-1, "z"], dtype=float)
    eps_x_prof = 1.0 / (eps0 * np.maximum(ieps_x, 1e-30))
    eps_z_prof = 1.0 / (eps0 * np.maximum(ieps_z, 1e-30))
    z_prof_um = z[:-1] * 1e6
    z_mid_prof_um = s.z[:-1] * 1e6

    src1 = CurrentSource(
        pulse,
        component="x",
        center=(0.5 * dz, 0.5 * dz, z_src),
        size=(dz, dz, 0.0),
        amplitude=1.0,
    )
    mr1 = FluxMonitor(freqs, center=(0.5 * dz, 0.5 * dz, z_refl), size=(dz, dz, 0.0))
    mt1 = FluxMonitor(freqs, center=(0.5 * dz, 0.5 * dz, z_tran), size=(dz, dz, 0.0))
    s.emsolve(nt, source=[src1], monitors=[mr1, mt1])

    R_sim = 1.0 - mr1.get_flux() / mr0.get_flux()
    T_sim = mt1.get_flux() / mt0.get_flux()
    R_exact, T_exact = slab_reflection_transmission(EPS_SLAB, d_slab, freqs)
    rms_err = float(np.sqrt(np.mean((R_sim - R_exact) ** 2)))
    max_err = float(np.max(np.abs(R_sim - R_exact)))
    energy_err = float(np.max(np.abs(R_sim + T_sim - 1.0)))

    return {
        "dz": dz,
        "d_slab": d_slab,
        "z_lo": z_lo,
        "z_hi": z_hi,
        "n_cells": int(g.Nx * g.Ny * g.Nz),
        "n_steps": int(2 * nt),
        "freqs": freqs,
        "R_sim": R_sim,
        "T_sim": T_sim,
        "R_exact": R_exact,
        "T_exact": T_exact,
        "rms_err": rms_err,
        "max_err": max_err,
        "energy_err": energy_err,
        "z_prof_um": z_prof_um,
        "z_mid_prof_um": z_mid_prof_um,
        "eps_x_prof": eps_x_prof,
        "eps_z_prof": eps_z_prof,
    }


def run_subpixel_convergence_task(cfg: dict) -> dict:
    """Run an off-grid dielectric slab convergence task (Kottke vs staircase)."""
    cpw_slab = int(cfg["cpw"])
    avg = str(cfg["averaging"])
    res = _eval_offgrid_slab(cpw_slab, avg)

    demo_cpw = 0
    demo_rms_kottke = np.nan
    demo_rms_staircase = np.nan
    if str(cfg.get("level", "")) == "reference" and avg == "kottke":
        demo_cpw = 6
        demo_rms_kottke = float(_eval_offgrid_slab(demo_cpw, "kottke")["rms_err"])
        demo_rms_staircase = float(_eval_offgrid_slab(demo_cpw, "staircase")["rms_err"])

    return {
        "name": cfg["name"],
        "level": cfg["level"],
        "cpw": cpw_slab,
        "averaging": avg,
        "dx_background_nm": (LAM0 / cpw_slab) * 1e9,
        "dx_min_nm": res["dz"] * 1e9,
        "min_feature_nm": res["d_slab"] * 1e9,
        "cells_across_feature": float(res["d_slab"] / res["dz"]),
        "n_cells": res["n_cells"],
        "n_steps": res["n_steps"],
        "freqs": res["freqs"],
        "R_sim": res["R_sim"],
        "T_sim": res["T_sim"],
        "R_exact": res["R_exact"],
        "T_exact": res["T_exact"],
        "rms_err": res["rms_err"],
        "max_err": res["max_err"],
        "energy_err": res["energy_err"],
        "z_lo_um": res["z_lo"] * 1e6,
        "z_hi_um": res["z_hi"] * 1e6,
        "z_prof_um": res["z_prof_um"],
        "z_mid_prof_um": res["z_mid_prof_um"],
        "eps_x_prof": res["eps_x_prof"],
        "eps_z_prof": res["eps_z_prof"],
        "demo_cpw": demo_cpw,
        "demo_rms_kottke": demo_rms_kottke,
        "demo_rms_staircase": demo_rms_staircase,
    }


def _build_slab_setup_objects(
    t_fp: dict,
) -> tuple[SolverFIT3D, CurrentSource, FluxMonitor, FluxMonitor, PlanarSlabZ]:
    """Construct the actual Wakis solver, slab, source, and FluxMonitor objects (without running time steps)."""
    cpw_slab = int(t_fp["cpw"])
    dz = (LAM0 / N_SLAB) / cpw_slab
    n_pml = 10
    z_um = np.asarray(t_fp["z_um"], dtype=float)
    nz = len(z_um)
    x = np.array([0.0, dz])
    y = np.array([0.0, dz])
    z = np.arange(nz + 1) * dz

    grid = GridFIT3D(x=x, y=y, z=z, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=n_pml,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    z_lo = float(t_fp["z_lo_um"]) * 1e-6
    z_hi = float(t_fp["z_hi_um"]) * 1e-6
    slab = PlanarSlabZ(z_lo, z_hi, EPS_SLAB)
    assign_materials(solver, [slab], F0, averaging="kottke")

    pulse = GaussianSource(F0, fwidth=0.38 * F0, cutoff=4.5)
    z_src = float(t_fp["z_src_um"]) * 1e-6
    src = CurrentSource(
        pulse,
        component="x",
        center=(0.5 * dz, 0.5 * dz, z_src),
        size=(dz, dz, 0.0),
        amplitude=1.0,
    )
    freqs = np.asarray(t_fp["freqs"], dtype=float)
    z_refl = float(t_fp["z_refl_um"]) * 1e-6
    z_tran = float(t_fp["z_tran_um"]) * 1e-6
    flux_r = FluxMonitor(freqs, center=(0.5 * dz, 0.5 * dz, z_refl), size=(dz, dz, 0.0))
    flux_t = FluxMonitor(freqs, center=(0.5 * dz, 0.5 * dz, z_tran), size=(dz, dz, 0.0))
    return solver, src, flux_r, flux_t, slab


def _save_slab_movie_gif(
    solver: SolverFIT3D,
    src: CurrentSource,
    flux_r: FluxMonitor,
    flux_t: FluxMonitor,
    z_um: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    out_path: Path,
    fps: int = 10,
) -> None:
    """Render an animated GIF of a plane-wave pulse reflecting and transmitting at a dielectric slab."""
    if frames.size == 0:
        return
    vmax = max(float(np.max(np.abs(frames))) * 1.12, 1e-12)
    pil_frames = []
    with figstyle.apply_style():
        for idx in range(len(frames)):
            fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=130)
            st0 = figstyle.series_style(0, with_marker=False)
            ax.plot(z_um, frames[idx], label=r"$E_x(z, t)$ [V/m]", **st0)
            schematic.overlay_geometry(
                ax,
                solver,
                plane="zx",
                unit="µm",
                orientation="1d",
                material_names={"region_0": f"Slab (n={N_SLAB:.1f})"},
                sources=[src],
                monitors=[flux_r, flux_t],
                show_boundaries=True,
                show_materials=True,
                show_sources=True,
                show_monitors=True,
            )
            ax.set(
                xlim=(float(z_um[0]), float(z_um[-1])),
                ylim=(-vmax, vmax),
                xlabel="Position z [µm]",
                ylabel="Electric field Ex(z, t) [V/m]",
                title=f"Fabry-Pérot slab multiple reflections (t = {times[idx] * 1e15:.1f} fs)",
            )
            figstyle.spectrum_grid(ax)
            ax.legend(loc="lower right", frameon=False, fontsize=9.0)
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

    required = ("fabry_perot_and_oblique", "cpml_reflection")
    for req in required:
        if req not in tasks_data:
            raise FileNotFoundError(
                f"Missing task {req!r} in {data_dir}. Run --quick or --full first."
            )

    t_fp = tasks_data["fabry_perot_and_oblique"]
    t_pml = tasks_data["cpml_reflection"]
    kottke_tasks = sorted(
        [d for name, d in tasks_data.items() if name.startswith("kottke_cpw")],
        key=lambda x: int(x["cpw"]),
    )
    stair_tasks = sorted(
        [d for name, d in tasks_data.items() if name.startswith("staircase_cpw")],
        key=lambda x: int(x["cpw"]),
    )
    if not kottke_tasks or not stair_tasks:
        raise FileNotFoundError(f"Missing kottke_cpw* or staircase_cpw* tasks in {data_dir}.")

    solver, src, flux_r, flux_t, _ = _build_slab_setup_objects(t_fp)
    written: list[Path] = []

    with figstyle.apply_style():
        # ------------------------------------------------------------------
        # Fig 00a & 00b: Setup schematic (Figure A) + Mesh & excitation (Figure B) + table
        # ------------------------------------------------------------------
        old_p0 = fig_dir / "00_setup_schematic.png"
        if old_p0.exists():
            old_p0.unlink()

        d_slab_m = (float(t_fp["z_hi_um"]) - float(t_fp["z_lo_um"])) * 1e-6
        setup_res = schematic.save_setup_figures(
            solver,
            fig_dir,
            plane="zx",
            unit="µm",
            material_names={"region_0": f"Dielectric slab (n = {N_SLAB:.1f})"},
            sources=[src],
            monitors=[flux_r, flux_t],
            wavelength=LAM0,
            min_feature=d_slab_m,
            title_geometry="Tutorial 1.2 — Setup: dielectric slab, source, monitors & boundaries",
            title_mesh="Tutorial 1.2 — Discretization and excitation",
            variations=[
                {
                    "name": "Run 1a: Reference run (§1.1)",
                    "details": "Vacuum everywhere (ε_r = 1.00); records incident flux Φ_ref(f) and E_ref(z_r, t)",
                },
                {
                    "name": "Run 1b: Slab run (§1.1–1.2)",
                    "details": f"Dielectric slab ε_r = {EPS_SLAB:.2f} (n = {N_SLAB:.2f}, d = {d_slab_m * 1e6:.3f} µm); R = 1 − Φ_r/Φ_ref, T = Φ_t/Φ_ref",
                },
                {
                    "name": "Run 2: Oblique / Brewster scan (§1.3)",
                    "details": "2D (x, z) domain; Bloch ±x boundary (k_x = k₀ sin θ_i), θ_i ∈ [0°, 75°], glass n = 1.50 (θ_B = 56.31°)",
                },
                {
                    "name": "Run 3: Boundary benchmark (§1.4)",
                    "details": "Vacuum domain; +z test boundary ∈ {CPML (N_pml = 4, 8, 12, 16), Mur ABC, PEC}",
                },
                {
                    "name": "Run 4: Subpixel convergence (§1.5)",
                    "details": "Off-grid slab d = 0.655 µm; Kottke tensor vs staircase at cpw ∈ {10, 15, 20, 30, 40}",
                },
            ],
        )
        written.append(setup_res["geometry_figure"])
        written.append(setup_res["mesh_figure"])

        # ------------------------------------------------------------------
        # Fig 00c: Distinct setups for Oblique Bloch run (§1.3) & Boundary benchmark (§1.4)
        # ------------------------------------------------------------------
        dz_c = float(solver.z[1] - solver.z[0])
        g_obl = GridFIT3D(
            x=np.linspace(0.0, 1.0e-6, 17),
            y=np.array([0.0, dz_c]),
            z=np.asarray(solver.z, dtype=float),
            verbose=0,
        )
        s_obl = SolverFIT3D(
            g_obl,
            bc_low=["bloch", "periodic", "cpml"],
            bc_high=["bloch", "periodic", "cpml"],
            n_pml=10,
            k_bloch=[2.0 * np.pi / LAM0 * np.sin(np.deg2rad(56.31)), 0.0, 0.0],
            dtype=np.complex128,
            use_stl=False,
            verbose=0,
        )
        assign_materials(
            s_obl,
            [PlanarSlabZ(float(t_fp["z_lo_um"]) * 1e-6, float(t_fp["z_hi_um"]) * 1e-6, 1.5**2)],
            F0,
        )
        src_obl = CurrentSource(
            src.src_time,
            component="x",
            center=(0.5e-6, 0.5 * dz_c, float(t_fp["z_src_um"]) * 1e-6),
            size=(1.0e-6, dz_c, 0.0),
        )
        mon_obl_r = FluxMonitor(
            flux_r.frequencies,
            center=(0.5e-6, 0.5 * dz_c, float(t_fp["z_refl_um"]) * 1e-6),
            size=(1.0e-6, dz_c, 0.0),
        )

        s_bc = SolverFIT3D(
            solver.grid,
            bc_low=["periodic", "periodic", "abc"],
            bc_high=["periodic", "periodic", "abc"],
            use_stl=False,
            verbose=0,
        )
        fig_c, (ax_c0, ax_c1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=6.8,
            ncols=1,
            nrows=2,
            constrained_layout=False,
        )
        fig_c.subplots_adjust(left=0.14, right=0.88, top=0.91, bottom=0.14, hspace=1.55)
        schematic.plot_schematic(
            s_obl,
            plane="zx",
            ax=ax_c0,
            unit="µm",
            material_names={"region_0": "Glass slab (n = 1.5)"},
            sources=[src_obl],
            monitors=[mon_obl_r],
            title="(a) §1.3 Oblique Bloch-periodic run (k_x = k₀ sin θ_i, glass n = 1.5)",
        )
        schematic.plot_schematic(
            s_bc,
            plane="zx",
            ax=ax_c1,
            unit="µm",
            sources=[src],
            probes=[{"pos": (0.5 * dz_c, 0.5 * dz_c, float(t_fp["z_refl_um"]) * 1e-6), "label": "Echo probe z_p"}],
            title="(b) §1.4 +z boundary reflection benchmark (CPML vs Mur ABC vs PEC)",
        )
        p0c = fig_dir / "00c_setup_oblique_and_bc_runs.png"
        figstyle.save(fig_c, p0c, close=True, auto_legend_below=False)
        written.append(p0c)

        # ------------------------------------------------------------------
        # Fig 01: Two-run normalization setup, echo subtraction & space-time
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(figstyle.FULL_WIDTH_IN, 6.8), dpi=figstyle.DEFAULT_DPI)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=1.22, wspace=0.34)
        ax_top = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

        schematic.plot_schematic(
            solver,
            plane="zx",
            ax=ax_top,
            unit="µm",
            material_names={"region_0": f"Dielectric slab (n = {N_SLAB:.1f})"},
            sources=[src],
            monitors=[flux_r, flux_t],
            wavelength=LAM0,
            title=r"(a) Two-run setup: slab ($n=2.0$), source & FluxMonitors ($R = 1 - \Phi_r/\Phi_{r,\mathrm{ref}}$)",
        )

        # Panel (b): Time-domain incident vs total vs subtracted reflected echo train
        t_fs = t_fp["t_arr"] * 1e15
        e_ref = t_fp["probe_r_ref"]
        e_tot = t_fp["probe_r_tot"]
        e_refl = e_tot - e_ref
        m_t = t_fs <= 55.0
        st0 = figstyle.series_style(7, with_marker=False)
        st1 = figstyle.series_style(0, with_marker=False)
        st2 = figstyle.series_style(1, with_marker=False)
        ax1.plot(t_fs[m_t], e_tot[m_t], label=r"Total $E_{x,\mathrm{tot}}(z_r, t)$", **st0)
        ax1.plot(t_fs[m_t], e_ref[m_t], label=r"Reference $E_{x,\mathrm{ref}}(z_r, t)$", **st1)
        ax1.plot(
            t_fs[m_t],
            e_refl[m_t],
            label=r"Reflected $E_{\mathrm{tot}} - E_{\mathrm{ref}}$",
            **st2,
        )
        ax1.set(
            xlabel="Time t [fs]",
            ylabel="Field Ex(zr, t) [V/m]",
            title="(b) Reflected echo train",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="upper right", frameon=False, fontsize=8.5)

        # Panel (c): Space-time diagram Ex(z, t) with geometry overlay
        wf = t_fp["wf_frames"]
        wf_t = t_fp["wf_times"] * 1e15
        z_u = t_fp["z_um"]
        vm = max(float(np.percentile(np.abs(wf), 99.0)), 1e-12)
        im = ax2.pcolormesh(
            z_u,
            wf_t,
            wf,
            cmap=figstyle.CMAP_DIVERGING,
            vmin=-vm,
            vmax=vm,
            shading="auto",
        )
        schematic.overlay_geometry(
            ax2,
            solver,
            plane="zx",
            unit="µm",
            orientation="1d",
            material_names={"region_0": f"Slab (n={N_SLAB:.1f})"},
            sources=[src],
            monitors=[flux_r, flux_t],
            show_boundaries=True,
            show_materials=True,
            show_sources=True,
            show_monitors=True,
        )
        ax2.set(
            xlabel="Position z [µm]",
            ylabel="Time t [fs]",
            title="(c) Space-time bounces",
        )
        fig.colorbar(im, ax=ax2, label=r"$E_x(z, t)$ [V/m]")

        p1 = fig_dir / "01_two_run_normalization_and_setup.png"
        figstyle.save(fig, p1, close=True, auto_legend_below=False)
        written.append(p1)

        # ------------------------------------------------------------------
        # Fig 02: Fabry-Perot Airy fringes & Oblique Fresnel / Brewster angle
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.3,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.34)

        f_norm = t_fp["freqs"] / F0
        f_dense = np.linspace(0.60 * F0, 1.40 * F0, 300)
        r_ex_d, t_ex_d = slab_reflection_transmission(EPS_SLAB, 1.25 * (LAM0 / N_SLAB), f_dense)

        st_r = figstyle.series_style(1, with_marker=False)
        st_t = figstyle.series_style(2, with_marker=False)
        st_rf = figstyle.series_style(1, with_line=False)
        st_tf = figstyle.series_style(2, with_line=False)
        st_rd = figstyle.series_style(0, with_line=False)
        st_sum = figstyle.series_style(7, with_marker=False)

        ax0.plot(f_dense / F0, r_ex_d, label="Exact Airy R(f)", **st_r)
        ax0.plot(f_dense / F0, t_ex_d, label="Exact Airy T(f)", **st_t)
        ax0.plot(f_norm[::2], t_fp["R_flux"][::2], label="FluxMonitor R", **st_rf)
        ax0.plot(f_norm[::2], t_fp["T_flux"][::2], label="FluxMonitor T", **st_tf)
        ax0.plot(
            f_norm[::3],
            t_fp["R_dft"][::3],
            mfc="none",
            mew=1.3,
            label=r"DFT $|E_{\mathrm{refl}}/E_{\mathrm{ref}}|^2$",
            **st_rd,
        )
        ax0.plot(
            f_norm,
            t_fp["R_flux"] + t_fp["T_flux"],
            label="R + T = 1",
            **st_sum,
        )
        ax0.set(
            xlabel=r"Normalized frequency $f / f_0$",
            ylabel="Power R(f), T(f)",
            xlim=(0.60, 1.40),
            ylim=(-0.05, 1.15),
            title="(a) Fabry-Pérot fringes",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="center right", frameon=False, fontsize=8.5, ncol=2)

        # Panel (b): Oblique incidence s/TE and p/TM vs angle theta (including Brewster angle)
        th_dense = np.linspace(0.0, 78.0, 250)
        d_ob = 0.375 * (LAM0 / N_SLAB)
        r_s_curve = [
            float(slab_rt_oblique(EPS_SLAB, d_ob, F0, np.deg2rad(a), "s")["R"][0])
            for a in th_dense
        ]
        t_s_curve = [
            float(slab_rt_oblique(EPS_SLAB, d_ob, F0, np.deg2rad(a), "s")["T"][0])
            for a in th_dense
        ]
        r_p_curve = [
            float(slab_rt_oblique(EPS_SLAB, d_ob, F0, np.deg2rad(a), "p")["R"][0])
            for a in th_dense
        ]
        t_p_curve = [
            float(slab_rt_oblique(EPS_SLAB, d_ob, F0, np.deg2rad(a), "p")["T"][0])
            for a in th_dense
        ]

        st_rs = figstyle.series_style(1, with_marker=False)
        st_ts = figstyle.series_style(3, with_marker=False)
        st_rp = figstyle.series_style(0, with_marker=False)
        st_tp = figstyle.series_style(4, with_marker=False)
        ax1.plot(th_dense, r_s_curve, label="Exact s/TE R(θ)", **st_rs)
        ax1.plot(th_dense, t_s_curve, label="Exact s/TE T(θ)", **st_ts)
        ax1.plot(th_dense, r_p_curve, label="Exact p/TM R(θ)", **st_rp)
        ax1.plot(th_dense, t_p_curve, label="Exact p/TM T(θ)", **st_tp)

        th_pts = t_fp["thetas_deg"]
        ax1.plot(
            th_pts,
            t_fp["R_ob_sim_s"],
            label="Bloch s/TE R",
            **figstyle.series_style(1, with_line=False),
        )
        ax1.plot(
            th_pts,
            t_fp["T_ob_sim_s"],
            mfc="none",
            mew=1.3,
            **figstyle.series_style(3, with_line=False),
        )
        ax1.plot(
            th_pts,
            t_fp["R_ob_sim_p"],
            label="Bloch p/TM R",
            **figstyle.series_style(0, with_line=False),
        )
        ax1.plot(
            th_pts,
            t_fp["T_ob_sim_p"],
            mfc="none",
            mew=1.3,
            **figstyle.series_style(4, with_line=False),
        )

        brew_deg = float(np.rad2deg(np.arctan(N_SLAB)))
        ax1.axvline(
            brew_deg,
            color=figstyle.PALETTE[2],
            ls=":",
            lw=1.6,
            marker="none",
            label=f"Brewster θ_B = {brew_deg:.1f}°",
        )
        ax1.set(
            xlabel="Incidence angle θ [deg]",
            ylabel="Power R(θ), T(θ)",
            xlim=(0.0, 78.0),
            ylim=(-0.05, 1.12),
            title="(b) Oblique Bloch & Brewster",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="center left", frameon=False, fontsize=8.5, ncol=2)

        p2 = fig_dir / "02_fabry_perot_and_oblique_fresnel.png"
        figstyle.save(fig, p2, close=True, auto_legend_below=False)
        written.append(p2)

        # ------------------------------------------------------------------
        # Fig 03: Absorbing boundaries: CPML vs Mur ABC vs PEC wall
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(figstyle.FULL_WIDTH_IN, 5.8), dpi=figstyle.DEFAULT_DPI)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.52, wspace=0.34)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

        f_thz = t_pml["freqs"] * 1e-12
        ax0.plot(
            f_thz,
            t_pml["r_db_pec"],
            label="PEC wall (0.0 dB)",
            **figstyle.series_style(7, with_marker=False),
        )
        ax0.plot(
            f_thz,
            t_pml["r_db_abc"],
            label="Mur 1st-order ABC",
            **figstyle.series_style(5, with_marker=False),
        )
        n_list = t_pml["n_pml_list"]
        for idx_n, n_p in enumerate(n_list):
            if int(n_p) not in (4, 8, 10, 16):
                continue
            st_p = figstyle.series_style(idx_n, markevery=4)
            ax0.plot(
                f_thz,
                t_pml["r_db_vs_npml_spec"][idx_n],
                label=f"CPML (N={int(n_p)})",
                **st_p,
            )
        ax0.set(
            xlabel="Frequency f [THz]",
            ylabel=r"Reflection $10\log_{10}|r(f)|^2$ [dB]",
            ylim=(-105.0, 8.0),
            title="(a) Boundary reflection spectrum: PEC wall vs. Mur ABC vs. CPML",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="lower right", frameon=False, fontsize=8.5, ncol=3)

        # Panel (b): CPML reflection vs layer count n_pml and grading order pml_exp
        ax1.plot(
            n_list,
            t_pml["r_db_vs_npml_f0"],
            label=r"At $f_0$ (exp=3)",
            **figstyle.series_style(0),
        )
        ax1.plot(
            n_list,
            t_pml["r_db_vs_npml_mean"],
            label="Band mean (exp=3)",
            **figstyle.series_style(2),
        )
        for idx_e, p_e in enumerate(t_pml["exp_list"]):
            st_e = figstyle.series_style(idx_e + 1, with_line=False)
            ax1.plot(
                [10],
                [t_pml["r_db_vs_exp_f0"][idx_e]],
                label=f"N=10, exp={int(p_e)} ({t_pml['r_db_vs_exp_f0'][idx_e]:.0f} dB)",
                **st_e,
            )
        ax1.axhline(0.0, color=figstyle.INK, ls=":", lw=1.1, marker="none")
        ax1.set(
            xlabel="CPML layers N_pml",
            ylabel="Reflection [dB]",
            xlim=(3, 17),
            ylim=(-102.0, 6.0),
            title="(b) CPML thickness & order",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="upper right", frameon=False, fontsize=8.5)

        # Panel (c): Time-domain reflected residual
        t_p_fs = t_pml["t_arr"] * 1e15
        ax2.semilogy(
            t_p_fs,
            np.maximum(np.abs(t_pml["tr_refl_pec"]), 1e-16),
            label="PEC wall",
            **figstyle.series_style(7, with_marker=False),
        )
        ax2.semilogy(
            t_p_fs,
            np.maximum(np.abs(t_pml["tr_refl_abc"]), 1e-16),
            label="Mur ABC",
            **figstyle.series_style(5, with_marker=False),
        )
        ax2.semilogy(
            t_p_fs,
            np.maximum(np.abs(t_pml["tr_refl_cpml10"]), 1e-16),
            label="CPML (N=10, exp=3)",
            **figstyle.series_style(0, with_marker=False),
        )
        ax2.set(
            xlabel="Time t [fs]",
            ylabel=r"$|E_{\mathrm{tot}} - E_{\mathrm{ref}}|$ [V/m]",
            ylim=(1e-5, 1e3),
            title="(c) Time-domain echo",
        )
        figstyle.spectrum_grid(ax2)
        ax2.legend(loc="lower right", frameon=False, fontsize=8.5)

        p3 = fig_dir / "03_cpml_reflection_benchmark.png"
        figstyle.save(fig, p3, close=True, auto_legend_below=False)
        written.append(p3)

        # ------------------------------------------------------------------
        # Fig 04: Off-grid interface permittivity profiles & spectra (Kottke vs staircase)
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.3,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.30)

        dk_med = kottke_tasks[min(1, len(kottke_tasks) - 1)]
        ds_med = stair_tasks[min(1, len(stair_tasks) - 1)]
        cpw_m = int(dk_med["cpw"])
        z_lo_m = float(dk_med["z_lo_um"])
        z_hi_m = float(dk_med["z_hi_um"])

        ax0.axvspan(
            z_lo_m,
            z_hi_m,
            color="#3B82F6",
            alpha=0.16,
            label=f"Exact slab (n={N_SLAB:.1f})",
        )
        ax0.axvline(z_lo_m, color=figstyle.INK, ls=":", lw=1.1, marker="none")
        ax0.axvline(z_hi_m, color=figstyle.INK, ls=":", lw=1.1, marker="none")
        ax0.step(
            ds_med["z_prof_um"],
            ds_med["eps_x_prof"],
            where="mid",
            color=figstyle.PALETTE[1],
            lw=2.0,
            marker="none",
            label=r"Staircase $\varepsilon_{xx}(z)$",
        )
        ax0.plot(
            dk_med["z_prof_um"],
            dk_med["eps_x_prof"],
            label=r"Kottke $\langle\varepsilon\rangle_x$",
            **figstyle.series_style(2),
        )
        ax0.plot(
            dk_med["z_mid_prof_um"],
            dk_med["eps_z_prof"],
            label=r"Kottke $\langle\varepsilon^{-1}\rangle_z^{-1}$",
            **figstyle.series_style(0),
        )
        ax0.set(
            xlim=(z_lo_m - 0.12, z_hi_m + 0.12),
            ylim=(0.6, 4.5),
            xlabel="Position z [µm]",
            ylabel=r"Permittivity $\varepsilon_r(z)$",
            title=f"(a) Off-grid interfaces ({cpw_m} cpw)",
        )
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="lower center", frameon=False, fontsize=8.5)

        # Panel (b): Reflectance spectrum R(f) comparing Kottke vs Staircase vs Exact
        f_m = dk_med["freqs"] / F0
        ax1.plot(
            f_m,
            dk_med["R_exact"],
            label="Exact Airy R(f)",
            **figstyle.series_style(7, with_marker=False),
        )
        ax1.plot(
            f_m,
            ds_med["R_sim"],
            label=f"Staircase (RMS={float(ds_med['rms_err']):.3f})",
            **figstyle.series_style(1, markevery=3),
        )
        ax1.plot(
            f_m,
            dk_med["R_sim"],
            label=f"Kottke (RMS={float(dk_med['rms_err']):.4f})",
            **figstyle.series_style(2, markevery=3),
        )
        ax1.set(
            xlabel=r"Normalized frequency $f / f_0$",
            ylabel="Reflectance R(f)",
            title="(b) Off-grid fringe shift",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="lower left", frameon=False, fontsize=8.5)

        p4 = fig_dir / "04_subpixel_permittivity_and_spectra.png"
        figstyle.save(fig, p4, close=True, auto_legend_below=False)
        written.append(p4)

        # ------------------------------------------------------------------
        # Fig 05: Subpixel convergence O(dz^2) vs Staircase O(dz^1)
        # ------------------------------------------------------------------
        fig, (ax0, ax1) = figstyle.new_figure(
            width_in=figstyle.FULL_WIDTH_IN,
            height_in=3.3,
            ncols=2,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.30)

        cpw_k = np.array([int(d["cpw"]) for d in kottke_tasks], dtype=float)
        rms_k = np.array([float(d["rms_err"]) for d in kottke_tasks], dtype=float)
        cpw_s = np.array([int(d["cpw"]) for d in stair_tasks], dtype=float)
        rms_s = np.array([float(d["rms_err"]) for d in stair_tasks], dtype=float)

        slope_k = float(np.polyfit(np.log(cpw_k), np.log(rms_k), 1)[0])
        slope_s = float(np.polyfit(np.log(cpw_s), np.log(rms_s), 1)[0])

        ax0.loglog(
            cpw_s,
            rms_s,
            label=rf"Staircase $\mathcal{{O}}(\Delta z^1)$ ({slope_s:.2f})",
            **figstyle.series_style(1),
        )
        ax0.loglog(
            cpw_k,
            rms_k,
            label=rf"Kottke $\mathcal{{O}}(\Delta z^2)$ ({slope_k:.2f})",
            **figstyle.series_style(2),
        )

        ref_k = kottke_tasks[-1]
        if int(ref_k.get("demo_cpw", 0)) > 0:
            dc = float(ref_k["demo_cpw"])
            ax0.loglog(
                [dc],
                [float(ref_k["demo_rms_staircase"])],
                label=f"Under-resolved ({int(dc)} cpw)",
                **figstyle.series_style(5, with_line=False),
            )

        ax0.axvline(
            20.0,
            color=figstyle.PALETTE[0],
            ls=":",
            lw=1.5,
            marker="none",
            label="20 cpw in slab",
        )
        ax0.set(
            xlabel=r"Slab resolution $\mathrm{cpw}_{\mathrm{slab}}$",
            ylabel=r"RMS error $\mathrm{RMS}(R_{\mathrm{sim}} - R_{\mathrm{ex}})$",
            title="(a) Kottke vs. staircase",
        )
        ax0.set_xticks([6, 10, 14, 20, 28, 40, 60])
        ax0.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax0.xaxis.set_minor_formatter(mticker.NullFormatter())
        figstyle.spectrum_grid(ax0)
        ax0.legend(loc="lower left", frameon=False, fontsize=8.5)

        # Panel (b): Energy conservation residual |R + T - 1| across resolution
        en_k = np.array([max(float(d["energy_err"]), 1e-12) for d in kottke_tasks], dtype=float)
        en_s = np.array([max(float(d["energy_err"]), 1e-12) for d in stair_tasks], dtype=float)
        ax1.semilogy(
            cpw_k,
            en_k,
            label="Kottke |R + T − 1|",
            **figstyle.series_style(2),
        )
        ax1.semilogy(
            cpw_s,
            en_s,
            label="Staircase |R + T − 1|",
            **figstyle.series_style(1),
        )
        ax1.axhline(
            1e-3,
            color=figstyle.PALETTE[5],
            ls=":",
            lw=1.4,
            marker="none",
            label="0.1% tolerance",
        )
        ax1.set(
            xlabel=r"Slab resolution $\mathrm{cpw}_{\mathrm{slab}}$",
            ylabel=r"Max $|R(f) + T(f) - 1|$",
            ylim=(1e-10, 1e-2),
            title="(b) Energy balance R + T = 1",
        )
        figstyle.spectrum_grid(ax1)
        ax1.legend(loc="upper right", frameon=False, fontsize=8.5)

        p5 = fig_dir / "05_subpixel_convergence_study.png"
        figstyle.save(fig, p5, close=True, auto_legend_below=False)
        written.append(p5)

        # ------------------------------------------------------------------
        # Fig 06: Animated GIF of Fabry-Perot slab reflections
        # ------------------------------------------------------------------
        p6 = fig_dir / "06_slab_fabry_perot_etalon.gif"
        _save_slab_movie_gif(
            solver,
            src,
            flux_r,
            flux_t,
            t_fp["z_um"],
            t_fp["movie_frames"],
            t_fp["movie_times"],
            p6,
        )
        written.append(p6)

    return written
