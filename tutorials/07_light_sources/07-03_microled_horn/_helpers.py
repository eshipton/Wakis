# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Helper utilities for Tutorial 7.3: MicroLED with a gradient-index
micro-horn collimator (Luce, Alaee & Abass, Sci. Rep. 16, 7391, 2026).

This module implements:
1. Optical material models and published reference benchmarks from Luce et al. (2026).
2. Analytical baselines (Snell escape cone, 2D/3D Lambertian cone fractions,
   multilayer transfer-matrix reflectance, 2D line-dipole homogeneous power).
3. Solver kit-gap bridges for 2D FIT emitter simulations:
   - `Near2FarMonitor2D`: 2D cylindrical Green's function scaling and periodic-z
     edge-weight correction for `wakis.near2far.Near2FarMonitor`, including
     exact TE/TM polarization separation.
   - `assign_materials_2d_dispersive`: Kottke subpixel smoothing and staircasing
     for sloped dispersive (`DispersiveMaterial`) and dielectric `PolySlab`
     interfaces via `wakis.tensor_dispersive`.
   - `InterpolatedCurrentSource2D`: Bilinear (cloud-in-cell) source and field
     interpolation eliminating staggered-grid dipole position jitter across
     mesh resolutions and quantum-well position sweeps.
"""

import time

import numpy as np
from scipy.constants import c as c_light, epsilon_0, mu_0

from wakis.dispersive import DispersiveMaterial, Drude, dispersive_lib
from wakis.field_monitors import DFTMonitor, _flat_index
from wakis.gridFIT3D import GridFIT3D
from wakis.near2far import Near2FarMonitor
from wakis.solverFIT3D import SolverFIT3D
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates.meshing import graded_axis
from wakis.templates.photonics import PolySlab
from wakis.templates.shapes import (
    _dual_step,
    _periodic_fill,
    _periodic_normal,
    _periodic_weight,
    material_eps,
)
from wakis.tensor_dispersive import install, network_from_edges

# ---------------------------------------------------------------------------
# Physical & geometric constants (Luce et al., Sci. Rep. 16, 7391, 2026)
# ---------------------------------------------------------------------------
WL_MIN = 425e-9  # Short-wavelength band edge [m]
WL_MAX = 475e-9  # Long-wavelength band edge [m]
WL_0 = 450e-9  # Peak InGaN emission wavelength [m]
WL_FWHM = 25e-9  # Spectral full-width at half-maximum [m]

W_GAN = 0.60e-6  # Active GaN core width [m]
T_GAN = 0.30e-6  # Active GaN core thickness [m]
W_PASS = 0.20e-6  # Lateral SiO2 passivation width per side [m]
W_PIXEL = W_GAN + 2.0 * W_PASS  # Total bottom opening width (1.0 um) [m]
Y_QW_DEFAULT = 0.15e-6  # Default quantum-well plane y-coordinate [m]
H_HORN_DEFAULT = 3.00e-6  # Default micro-horn height [m]

# Standard literature refractive indices at lambda_0 = 450 nm
N_GAN = 2.45
N_SIO2 = 1.46
N_AIR = 1.00

# 6-layer discrete gradient-index (dGRIN) stack (bottom -> top, equal thickness)
DGRIN_STACK = (
    ("Nb2O5", 2.37),
    ("Si3N4", 2.05),
    ("Al2O3", 1.77),
    ("SiO2", 1.46),
    ("MgF2", 1.38),
    ("Air", 1.00),
)
DGRIN_INDICES = tuple(n for _, n in DGRIN_STACK)

# Published 2D FDTD benchmark values from Luce et al. (2026), p. 6, Fig. 3 & App. A.2
PAPER_2D_REFS = {
    "bare": {
        "label": "Bare reference (no horn)",
        "theta_deg": 90.0,
        "h_um": 0.0,
        "w_top_um": 1.00,
        "lee15_pct": 14.9,
        "lee90_pct": 55.9,
        "ratio_pct": round(100.0 * 14.9 / 55.9, 2),
        "enh15": 1.00,
        "citation": "Luce et al. (2026), p. 6 & App. A.2",
    },
    "air": {
        "label": "Air-filled µHorn",
        "theta_deg": 72.0,
        "h_um": 3.0,
        "w_top_um": 2.95,
        "lee15_pct": 27.5,
        "lee90_pct": 46.7,
        "ratio_pct": round(100.0 * 27.5 / 46.7, 2),
        "enh15": 1.85,
        "citation": "Luce et al. (2026), p. 6 & Fig. 3a,d",
    },
    "sio2": {
        "label": "SiO2-filled µHorn",
        "theta_deg": 64.0,
        "h_um": 3.0,
        "w_top_um": 3.93,
        "lee15_pct": 30.4,
        "lee90_pct": 65.1,
        "ratio_pct": round(100.0 * 30.4 / 65.1, 2),
        "enh15": 2.04,
        "citation": "Luce et al. (2026), p. 6 & Fig. 3b,e",
    },
    "dgrin": {
        "label": "6-layer dGRIN µHorn",
        "theta_deg": 60.0,
        "h_um": 3.0,
        "w_top_um": 4.46,
        "lee15_pct": 37.7,
        "lee90_pct": 75.3,
        "ratio_pct": round(100.0 * 37.7 / 75.3, 2),
        "enh15": 2.53,
        "citation": "Luce et al. (2026), p. 6 & Fig. 3c,f",
    },
}

# Published 3D cylindrical FDTD benchmark values from Luce et al. (2026), Table 1
PAPER_3D_TABLE1 = {
    "bare": {
        "lee15_pct": 2.3,
        "enh15": 1.00,
        "lee90_pct": 31.8,
        "enh90": 1.00,
        "ratio_pct": 7.0,
    },
    "air": {
        "lee15_pct": 11.6,
        "enh15": 5.04,
        "lee90_pct": 27.7,
        "enh90": 0.87,
        "ratio_pct": 42.0,
    },
    "sio2": {
        "lee15_pct": 11.7,
        "enh15": 5.09,
        "lee90_pct": 54.1,
        "enh90": 1.70,
        "ratio_pct": 22.0,
    },
    "gan": {
        "lee15_pct": 9.2,
        "enh15": 4.00,
        "lee90_pct": 66.5,
        "enh90": 2.09,
        "ratio_pct": 13.0,
    },
    "dgrin": {
        "lee15_pct": 19.9,
        "enh15": 8.65,
        "lee90_pct": 75.6,
        "enh90": 2.38,
        "ratio_pct": 26.0,
    },
    "cgrin": {
        "lee15_pct": 24.6,
        "enh15": 10.57,
        "lee90_pct": 80.0,
        "enh90": 2.52,
        "ratio_pct": 31.0,
    },
}


# ---------------------------------------------------------------------------
# Optical dispersion & analytical baselines
# ---------------------------------------------------------------------------
def make_ag_drude(wl_fit=WL_0, eps_inf=4.0):
    """
    Construct a 1-pole Drude dispersive silver model calibrated to Rakic Ag at `wl_fit`.
    """
    f_fit = c_light / float(wl_fit)
    w_fit = 2.0 * np.pi * f_fit
    eps_rakic = dispersive_lib["silver"].epsilon(f_fit)
    denom = float(eps_inf) - eps_rakic
    inv_d = 1.0 / denom
    gamma = -w_fit * inv_d.imag / inv_d.real
    omega_p = np.sqrt(w_fit**2 / inv_d.real)
    return DispersiveMaterial(
        eps_inf=float(eps_inf),
        poles=[Drude(omega_p=omega_p, gamma=gamma)],
        name="Ag_Drude_450nm",
    )


def ag_skin_depth(wavelength=WL_0, material=None):
    """
    Return electric-field and intensity skin depths `(delta_E, delta_I)` [m] of Ag.
    """
    mat = make_ag_drude() if material is None else material
    f = c_light / float(wavelength)
    w = 2.0 * np.pi * f
    eps = mat.epsilon(f)
    n_complex = np.sqrt(eps)
    kappa = abs(float(np.imag(n_complex)))
    delta_e = c_light / (w * kappa)
    return delta_e, 0.5 * delta_e


def led_gaussian_spectrum(wavelengths, wl_center=WL_0, fwhm=WL_FWHM):
    """
    Unnormalized Gaussian spectral lineshape S(lambda) of the InGaN active region.
    """
    wl = np.asarray(wavelengths, dtype=float)
    return np.exp(-4.0 * np.log(2.0) * ((wl - wl_center) / fwhm) ** 2)


def snell_critical_angle_deg(n_in=N_GAN, n_out=N_AIR):
    """Critical angle `arcsin(n_out / n_in)` [deg] for total internal reflection."""
    if n_out >= n_in:
        return 90.0
    return float(np.rad2deg(np.arcsin(float(n_out) / float(n_in))))


def fresnel_normal_transmission(n_in=N_GAN, n_out=N_AIR):
    """Normal-incidence Fresnel power transmission `4 n1 n2 / (n1 + n2)^2`."""
    return float(4.0 * n_in * n_out / ((n_in + n_out) ** 2))


def analytic_escape_cone_lee(n_in=N_GAN, n_out=N_AIR, dim=2, back_reflector=True):
    """
    Analytical planar escape-cone extraction efficiency (without waveguide/horn modes).
    """
    theta_c = np.deg2rad(snell_critical_angle_deg(n_in, n_out))
    t_fres = fresnel_normal_transmission(n_in, n_out)
    if dim == 2:
        cone_frac = theta_c / np.pi
    elif dim == 3:
        cone_frac = 0.5 * (1.0 - np.cos(theta_c))
    else:
        raise ValueError("dim must be 2 or 3.")
    factor = 2.0 if back_reflector else 1.0
    return float(factor * cone_frac * t_fres)


def lambertian_cone_fraction(theta_max_deg=15.0, dim=2):
    """
    Fraction of a Lambertian emitter's upper-hemisphere power within `+-theta_max_deg`.
    """
    th = np.deg2rad(float(theta_max_deg))
    if dim == 2:
        return float(np.sin(th))
    if dim == 3:
        return float(np.sin(th) ** 2)
    raise ValueError("dim must be 2 or 3.")


def tmm_normal_reflectance(
    wavelengths, n_in=N_GAN, n_out=N_AIR, layer_indices=(), layer_thicknesses=()
):
    """
    Normal-incidence power reflectance R(lambda) and transmittance T(lambda) via 2x2 TMM.
    """
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    k0 = 2.0 * np.pi / wl
    m11 = np.ones_like(wl, dtype=complex)
    m12 = np.zeros_like(wl, dtype=complex)
    m21 = np.zeros_like(wl, dtype=complex)
    m22 = np.ones_like(wl, dtype=complex)

    for n_lay, d_lay in zip(layer_indices, layer_thicknesses):
        delta = k0 * float(n_lay) * float(d_lay)
        c_d = np.cos(delta)
        s_d = -1j * np.sin(delta)
        n_c = float(n_lay)
        new11 = m11 * c_d + m12 * (s_d * n_c)
        new12 = m11 * (s_d / n_c) + m12 * c_d
        new21 = m21 * c_d + m22 * (s_d * n_c)
        new22 = m21 * (s_d / n_c) + m22 * c_d
        m11, m12, m21, m22 = new11, new12, new21, new22

    num = (m11 + m12 * n_out) * n_in - (m21 + m22 * n_out)
    den = (m11 + m12 * n_out) * n_in + (m21 + m22 * n_out)
    r = num / den
    R = np.abs(r) ** 2
    return R, 1.0 - R


def analytic_2d_line_dipole_power(frequencies, spectrum_f, dz, component="z"):
    """
    Analytical 2D homogeneous-medium power spectrum Phi_hom(f) [W s^2] of a unit point source.
    """
    freqs = np.asarray(frequencies, dtype=float)
    w = 2.0 * np.pi * freqs
    s2 = np.abs(np.asarray(spectrum_f)) ** 2
    c = component[-1].lower()
    denom = 4.0 * float(dz) if c == "z" else 8.0 * float(dz)
    return w * mu_0 * s2 / denom


def half_power_beamwidth_deg(angles_rad, pattern):
    """Full-width at half-maximum (FWHM) angular divergence [deg] of a 1D pattern."""
    ang_deg = np.rad2deg(np.asarray(angles_rad, dtype=float))
    pat = np.asarray(pattern, dtype=float)
    if pat.max() <= 0:
        return 0.0
    norm = pat / pat.max()
    above = norm >= 0.5
    if not np.any(above):
        return 0.0
    idx = np.flatnonzero(above)
    i0, i1 = idx[0], idx[-1]
    if i0 > 0:
        f0 = (0.5 - norm[i0 - 1]) / max(norm[i0] - norm[i0 - 1], 1e-30)
        a_left = ang_deg[i0 - 1] + f0 * (ang_deg[i0] - ang_deg[i0 - 1])
    else:
        a_left = ang_deg[0]
    if i1 < len(ang_deg) - 1:
        f1 = (0.5 - norm[i1]) / min(norm[i1 + 1] - norm[i1], -1e-30)
        a_right = ang_deg[i1] + f1 * (ang_deg[i1 + 1] - ang_deg[i1])
    else:
        a_right = ang_deg[-1]
    return float(a_right - a_left)


def directivity_2d_halfspace(angles_rad, pattern):
    """
    Upper-hemisphere 2D angular directivity `D_2D = pi * max(U) / int_{-pi/2}^{pi/2} U dtheta`.
    """
    ang = np.asarray(angles_rad, dtype=float)
    pat = np.asarray(pattern, dtype=float)
    total = float(np.trapezoid(pat, ang))
    if total <= 0:
        return 0.0
    return float(np.pi * np.max(pat) / total)


# ---------------------------------------------------------------------------
# Kit-gap helpers: 2D Near2FarMonitor, Bilinear Dipole, & Dispersive Kottke
# ---------------------------------------------------------------------------
def _bilinear_yee_weights_2d(grid, component, x0, y0):
    """
    Compute 2x2 bilinear (cloud-in-cell) interpolation indices and weights
    on the staggered 2D Yee grid for `component` in `('x', 'y', 'z')`.
    """
    c = "xyz".index(component[-1].lower())
    ux = 0.5 * (grid.x[:-1] + grid.x[1:]) if c == 0 else grid.x[:-1]
    uy = 0.5 * (grid.y[:-1] + grid.y[1:]) if c == 1 else grid.y[:-1]

    i0 = int(np.clip(np.searchsorted(ux, x0) - 1, 0, len(ux) - 2))
    j0 = int(np.clip(np.searchsorted(uy, y0) - 1, 0, len(uy) - 2))
    ax = float(np.clip((x0 - ux[i0]) / max(ux[i0 + 1] - ux[i0], 1e-30), 0.0, 1.0))
    ay = float(np.clip((y0 - uy[j0]) / max(uy[j0 + 1] - uy[j0], 1e-30), 0.0, 1.0))

    ix = np.array([i0, i0 + 1, i0, i0 + 1], dtype=np.int64)
    iy = np.array([j0, j0, j0 + 1, j0 + 1], dtype=np.int64)
    iz = np.zeros(4, dtype=np.int64)
    w_interp = np.array(
        [(1.0 - ax) * (1.0 - ay), ax * (1.0 - ay), (1.0 - ax) * ay, ax * ay],
        dtype=float,
    )
    return ix, iy, iz, w_interp


class InterpolatedCurrentSource2D(CurrentSource):
    """
    2D point `CurrentSource` using bilinear (cloud-in-cell) interpolation onto
    the staggered Yee grid so all polarizations `(Ex, Ey, Ez)` are centered at
    the exact continuous coordinate `(x0, y0)` without grid-snapping jitter.
    """

    def _set_edges(self, solver):
        c = "xyz".index(self.component)
        ix, iy, iz, w_interp = _bilinear_yee_weights_2d(
            solver.grid, self.component, self.center[0], self.center[1]
        )
        vol = np.ones(4, dtype=float)
        for d in range(3):
            widths = solver.grid.L if d == c else solver.grid.tL
            vol = vol * widths[ix, iy, iz, "xyz"[d]]
        weights = (self.amplitude * w_interp) / vol
        flat = _flat_index(solver, self.component, ix, iy, iz)
        self.flat_idx = solver.J.xp.asarray(flat)
        self.weights = solver.J.xp.asarray(weights, dtype=solver.J.dtype)
        self.J_old = solver.J.xp.zeros_like(self.weights)
        self.interp_weights = w_interp


class InterpolatedPointDFTMonitor2D:
    """
    Accumulate the bilinearly interpolated electric-field spectra at a continuous
    2D dipole location `(x0, y0)` matching `InterpolatedCurrentSource2D`.
    """

    def __init__(self, frequencies, components=("Ex", "Ey", "Ez"), center=(0.0, 0.0)):
        self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        self.components = tuple(components)
        self.center = (float(center[0]), float(center[1]))
        self.is_first_update = True

    def _initialize(self, solver):
        self._freqs = np.asarray(self.frequencies, dtype=float)
        self._entries = {}
        for comp in self.components:
            c_name = comp[-1].lower()
            ix, iy, iz, w_interp = _bilinear_yee_weights_2d(
                solver.grid, c_name, self.center[0], self.center[1]
            )
            flat = _flat_index(solver, c_name, ix, iy, iz)
            self._entries[comp] = {
                "flat": flat,
                "w": w_interp,
                "acc": np.zeros(self._freqs.size, dtype=np.complex128),
            }

    def update(self, solver, t):
        if self.is_first_update:
            self._initialize(solver)
            self.is_first_update = False
        kernel = solver.dt * np.exp(-2j * np.pi * self._freqs * (t + solver.dt))
        e_arr = solver.E.array
        for item in self._entries.values():
            val = float(np.dot(item["w"], e_arr[item["flat"]]))
            item["acc"] += kernel * val

    def get_spectrum(self, component):
        return self._entries[component]["acc"].copy()


class Near2FarMonitor2D(Near2FarMonitor):
    """
    2D (x, y) Near-to-Far-Field monitor for a 1-cell periodic-z FIT domain.
    """

    def __init__(self, frequencies, center_xy, size_xy, dz, **kwargs):
        super().__init__(
            frequencies=frequencies,
            center=(float(center_xy[0]), float(center_xy[1]), 0.0),
            size=(float(size_xy[0]), float(size_xy[1]), float(dz)),
            **kwargs,
        )
        self.faces = [f for f in self.faces if f[0] != 2]
        self.dz = float(dz)

    def _initialize(self, solver):
        orig_sz = self.size[2]
        self.size[2] = 3.0 * orig_sz
        super()._initialize(solver)
        self.size[2] = orig_sz

    def radiation_intensity_2d_te_tm(self, alpha):
        """
        Return `(U_TE, U_TM)` 2D angular radiation intensities [W s^2 / rad].
        """
        alpha = np.asarray(alpha, dtype=float)
        phi = 0.5 * np.pi - alpha
        theta = np.full_like(alpha, 0.5 * np.pi)
        ff = self.get_farfield(theta, phi)
        k = self.wavenumber[:, None] if ff["Etheta"].ndim > 1 else self.wavenumber
        scale = (2.0 * np.pi / (k * self.dz)) / self.impedance
        u_tm = (np.abs(ff["Etheta"]) ** 2) * scale
        u_te = (np.abs(ff["Ephi"]) ** 2) * scale
        return u_te, u_tm

    def radiation_intensity_2d(self, alpha):
        """Total 2D angular radiation intensity `U_TE + U_TM` [W s^2 / rad]."""
        u_te, u_tm = self.radiation_intensity_2d_te_tm(alpha)
        return u_te + u_tm


def assign_materials_2d_dispersive(solver, structures, frequency, averaging="kottke"):
    """
    Assign static, PEC, and dispersive (`DispersiveMaterial`) `Box`/`PolySlab`
    structures with Kottke subpixel smoothing or staircasing.
    """
    grid = solver.grid
    nx, ny, nz = grid.Nx, grid.Ny, grid.Nz
    N = nx * ny * nz
    tol = 1e-9 * min(np.min(grid.dx), np.min(grid.dy), np.min(grid.dz))
    nodes = (grid.x, grid.y, grid.z)
    centers = [0.5 * (u[:-1] + u[1:]) for u in nodes]
    prim_steps = [np.diff(u) for u in nodes]
    dual_steps = [
        _dual_step(nodes[d], solver.bc_low[d] in ("periodic", "bloch"))
        for d in range(3)
    ]
    shifts = [
        (-L, 0.0, L)
        if solver.bc_low[d] in ("periodic", "bloch")
        and (L := nodes[d][-1] - nodes[d][0]) > 0
        else (0.0,)
        for d in range(3)
    ]

    eps_static_all = np.ones(3 * N, dtype=float)
    f_disp_all = np.zeros(3 * N, dtype=float)
    p_disp_all = np.zeros(3 * N, dtype=float)
    eps_bg_under_disp = np.ones(3 * N, dtype=float)
    disp_mat = None

    for comp in range(3):
        coords_1d = [centers[d] if d == comp else nodes[d][:-1] for d in range(3)]
        sizes_1d = [prim_steps[d] if d == comp else dual_steps[d] for d in range(3)]
        X, Y, Z = np.meshgrid(*coords_1d, indexing="ij")
        SX, SY, SZ = np.meshgrid(*sizes_1d, indexing="ij")

        block_eps = np.ones((nx, ny, nz), dtype=float)
        block_f_disp = np.zeros((nx, ny, nz), dtype=float)
        block_p_disp = np.zeros((nx, ny, nz), dtype=float)
        block_eps_under = np.ones((nx, ny, nz), dtype=float)

        for shape in structures:
            if shape.material == "pec":
                w_raw = np.broadcast_to(
                    _periodic_weight(shape, (X, Y, Z), shifts, tol), X.shape
                )
                full_pec = w_raw >= 1.0 - 1e-12
                block_f_disp[full_pec] = 0.0
                block_p_disp[full_pec] = 0.0
                block_eps[full_pec] = np.inf
                continue

            is_disp = isinstance(shape.material, DispersiveMaterial)
            if is_disp:
                disp_mat = shape.material
                eps_shape = shape.material.eps_inf
            else:
                eps_shape = material_eps(shape.material, frequency)

            if hasattr(shape, "fill") and averaging != "staircase":
                w = np.broadcast_to(
                    _periodic_fill(shape, (X, Y, Z), (SX, SY, SZ), shifts, tol),
                    X.shape,
                )
            else:
                w_raw = np.broadcast_to(
                    _periodic_weight(shape, (X, Y, Z), shifts, tol), X.shape
                )
                w = (w_raw >= 0.5).astype(float) if averaging == "staircase" else w_raw

            full = w >= 1.0 - 1e-12
            part = (w > 1e-12) & (~full) & np.isfinite(block_eps)

            if is_disp:
                touched = full | part
                block_eps_under[touched] = np.where(
                    np.isfinite(block_eps[touched]), block_eps[touched], 1.0
                )
                block_f_disp[full] = 1.0
                block_p_disp[full] = 0.0
                if np.any(part):
                    wp = w[part]
                    block_f_disp[part] = wp
                    if averaging == "kottke" and hasattr(shape, "normal"):
                        n_vec = _periodic_normal(
                            shape, (X[part], Y[part], Z[part]), shifts, tol
                        )
                        block_p_disp[part] = n_vec[:, comp] ** 2
                    else:
                        block_p_disp[part] = 0.0
            else:
                block_f_disp[full] = 0.0
                block_p_disp[full] = 0.0
                block_eps[full] = eps_shape
                if np.any(part):
                    wp = w[part]
                    has_disp = block_f_disp[part] > 0
                    if np.any(has_disp):
                        block_f_disp[part] = (
                            np.maximum(0.0, 1.0 - wp) * block_f_disp[part]
                        )
                        block_eps_under[part] = eps_shape
                        if averaging == "kottke" and hasattr(shape, "normal"):
                            n_vec = _periodic_normal(
                                shape, (X[part], Y[part], Z[part]), shifts, tol
                            )
                            block_p_disp[part] = n_vec[:, comp] ** 2
                    else:
                        eps_b = block_eps[part]
                        if averaging == "kottke" and hasattr(shape, "normal"):
                            n_vec = _periodic_normal(
                                shape, (X[part], Y[part], Z[part]), shifts, tol
                            )
                            nc2 = n_vec[:, comp] ** 2
                            inv_eps = nc2 * (wp / eps_shape + (1.0 - wp) / eps_b) + (
                                1.0 - nc2
                            ) / (wp * eps_shape + (1.0 - wp) * eps_b)
                            block_eps[part] = 1.0 / inv_eps
                        else:
                            block_eps[part] = (1.0 - wp) * eps_b + wp * eps_shape

        sl = slice(comp * N, (comp + 1) * N)
        eps_static_all[sl] = block_eps.ravel(order="F")
        f_disp_all[sl] = block_f_disp.ravel(order="F")
        p_disp_all[sl] = block_p_disp.ravel(order="F")
        eps_bg_under_disp[sl] = block_eps_under.ravel(order="F")

    eps_eff = eps_static_all.copy()
    sel = np.flatnonzero((f_disp_all > 1e-12) & np.isfinite(eps_static_all))
    if sel.size > 0 and disp_mat is not None:
        fd = f_disp_all[sel]
        p = p_disp_all[sel]
        eb = eps_bg_under_disp[sel]
        einf_m = disp_mat.eps_inf
        inv_eps_d = p * (fd / einf_m + (1.0 - fd) / eb) + (1.0 - p) / (
            fd * einf_m + (1.0 - fd) * eb
        )
        eps_eff[sel] = 1.0 / inv_eps_d

    ieps = np.where(np.isinf(eps_eff), 0.0, 1.0 / (epsilon_0 * eps_eff))
    solver.ieps.fromarray(ieps)

    if sel.size > 0 and disp_mat is not None:
        fd = f_disp_all[sel]
        p = p_disp_all[sel]
        eb = eps_bg_under_disp[sel]
        einf_m = disp_mat.eps_inf

        edge_list, c_list, einf_list, s_list, mu_list = [], [], [], [], []
        count = 0
        tang = np.flatnonzero(1.0 - p > 1e-12)
        if tang.size > 0:
            edge_list.append(tang)
            c_list.append(1.0 - p[tang])
            einf_list.append(fd[tang] * einf_m + (1.0 - fd[tang]) * eb[tang])
            s_list.append(count + np.arange(tang.size))
            mu_list.append(fd[tang])
            count += tang.size

        ser = np.flatnonzero(p > 1e-12)
        if ser.size > 0:
            edge_list.append(ser)
            c_list.append(p[ser] * fd[ser])
            einf_list.append(np.full(ser.size, einf_m))
            s_list.append(count + np.arange(ser.size))
            mu_list.append(np.ones(ser.size))
            count += ser.size

        edge_arr = np.concatenate(edge_list)
        c_arr = np.concatenate(c_list)
        einf_arr = np.concatenate(einf_list)
        members = [(np.concatenate(s_list), np.concatenate(mu_list))]
        net = network_from_edges(
            solver, sel, edge_arr, c_arr, einf_arr, [disp_mat], members
        )
        install(solver, net)


# ---------------------------------------------------------------------------
# Geometry, mesh, and simulation runner
# ---------------------------------------------------------------------------
def _rect_polyslab(x0, x1, y0, y1, z_bounds, material):
    return PolySlab(
        vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        z_min=z_bounds[0],
        z_max=z_bounds[1],
        material=material,
    )


def build_microled_horn_structures(
    config="dgrin",
    h_horn=H_HORN_DEFAULT,
    theta_deg=60.0,
    x_max=2.8e-6,
    y_min=-0.30e-6,
    t_skin=0.20e-6,
    z_bounds=(-1e-6, 1e-6),
    ag_material=None,
    horn_ag_material=None,
    passivation_material=None,
):
    """
    Build the 2D microLED + tapered horn geometry of Luce et al. (2026), Fig. 2.

    All solids are represented as `PolySlab` polygons in `(x, y)` so both
    horizontal and sloped interfaces support exact subpixel `.fill` and `.normal`.
    """
    ag_mat = make_ag_drude() if ag_material is None else ag_material
    horn_ag_mat = ag_mat if horn_ag_material is None else horn_ag_material
    pass_mat = N_SIO2**2 if passivation_material is None else passivation_material
    y_base = T_GAN
    x_bot_half = 0.5 * W_PIXEL
    x_gan_half = 0.5 * W_GAN

    structures = [
        _rect_polyslab(-x_max, x_max, y_min, y_base, z_bounds, ag_mat),
        _rect_polyslab(-x_bot_half, x_bot_half, 0.0, y_base, z_bounds, pass_mat),
        _rect_polyslab(-x_gan_half, x_gan_half, 0.0, y_base, z_bounds, N_GAN**2),
    ]

    if config == "bare" or h_horn <= 0.0:
        return structures, y_base, W_PIXEL

    theta_rad = np.deg2rad(float(theta_deg))
    dx_horn = float(h_horn) / np.tan(theta_rad)
    w_top = W_PIXEL + 2.0 * dx_horn
    y_top = y_base + float(h_horn)
    x_top_half = 0.5 * w_top

    left_ag = PolySlab(
        vertices=[
            (-x_max, y_base),
            (-x_bot_half, y_base),
            (-x_top_half, y_top),
            (-x_max, y_top),
        ],
        z_min=z_bounds[0],
        z_max=z_bounds[1],
        material=horn_ag_mat,
    )
    right_ag = PolySlab(
        vertices=[
            (x_bot_half, y_base),
            (x_max, y_base),
            (x_max, y_top),
            (x_top_half, y_top),
        ],
        z_min=z_bounds[0],
        z_max=z_bounds[1],
        material=horn_ag_mat,
    )

    if config == "air":
        structures.extend([left_ag, right_ag])
    elif config == "sio2":
        horn_fill = PolySlab(
            vertices=[
                (-x_bot_half, y_base),
                (x_bot_half, y_base),
                (x_top_half, y_top),
                (-x_top_half, y_top),
            ],
            z_min=z_bounds[0],
            z_max=z_bounds[1],
            material=N_SIO2**2,
        )
        structures.extend([horn_fill, left_ag, right_ag])
    elif config == "dgrin":
        n_layers = len(DGRIN_INDICES)
        dh = float(h_horn) / n_layers
        for idx, n_lay in enumerate(DGRIN_INDICES):
            if n_lay == 1.0:
                continue
            y0 = y_base + idx * dh
            y1 = y_base + (idx + 1) * dh
            xb0 = x_bot_half + (idx * dh) / np.tan(theta_rad)
            xb1 = x_bot_half + ((idx + 1) * dh) / np.tan(theta_rad)
            layer_slab = PolySlab(
                vertices=[
                    (-xb0, y0),
                    (xb0, y0),
                    (xb1, y1),
                    (-xb1, y1),
                ],
                z_min=z_bounds[0],
                z_max=z_bounds[1],
                material=n_lay**2,
            )
            structures.append(layer_slab)
        structures.extend([left_ag, right_ag])
    else:
        raise ValueError(f"Unknown horn config {config!r}.")

    if t_skin is not None and t_skin > 0.0:
        x_pec_bot = x_bot_half + t_skin
        x_pec_top = x_top_half + t_skin
        y_pec_top = y_top - t_skin
        if y_pec_top > y_base and x_max > x_pec_top:
            left_pec = PolySlab(
                vertices=[
                    (-x_max, y_min),
                    (-x_pec_bot, y_min),
                    (-x_pec_bot, y_base),
                    (-x_pec_top, y_pec_top),
                    (-x_max, y_pec_top),
                ],
                z_min=z_bounds[0],
                z_max=z_bounds[1],
                material="pec",
            )
            right_pec = PolySlab(
                vertices=[
                    (x_pec_bot, y_min),
                    (x_max, y_min),
                    (x_max, y_pec_top),
                    (x_pec_top, y_pec_top),
                    (x_pec_bot, y_base),
                ],
                z_min=z_bounds[0],
                z_max=z_bounds[1],
                material="pec",
            )
            structures.extend([left_pec, right_pec])

    return structures, y_top, w_top


def build_uniform_aligned_grid(
    config="dgrin",
    h_horn=H_HORN_DEFAULT,
    theta_deg=60.0,
    dx=15e-9,
    x_half_min=2.50e-6,
    y_below=0.21e-6,
    y_above=0.48e-6,
    dz=None,
):
    """
    Construct a uniform 2D `(x, y)` grid whose nodes align with `x = 0` and `y = 0`.
    """
    dx = float(dx)
    dz = dx if dz is None else float(dz)
    y_top = T_GAN + (0.0 if (config == "bare" or h_horn <= 0.0) else float(h_horn))
    w_top = (
        W_PIXEL
        if (config == "bare" or h_horn <= 0.0)
        else W_PIXEL + 2.0 * float(h_horn) / np.tan(np.deg2rad(float(theta_deg)))
    )
    x_half_target = max(float(x_half_min), 0.5 * w_top + 0.36e-6)
    nx_half = int(np.ceil(x_half_target / dx))
    x_max = nx_half * dx
    x_nodes = np.linspace(-x_max, x_max, 2 * nx_half + 1)

    ny_below = max(8, int(np.round(float(y_below) / dx)))
    ny_above = int(np.ceil((y_top + float(y_above)) / dx))
    y_min = -ny_below * dx
    y_max = ny_above * dx
    y_nodes = np.linspace(y_min, y_max, ny_below + ny_above + 1)
    z_nodes = np.array([-0.5 * dz, 0.5 * dz])

    grid = GridFIT3D(x=x_nodes, y=y_nodes, z=z_nodes, verbose=0)
    info = {
        "dx": dx,
        "dx_skin": dx,
        "dz": dz,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "y_top": y_top,
        "w_top": w_top,
    }
    return grid, info


def build_graded_horn_grid(
    config="dgrin",
    h_horn=H_HORN_DEFAULT,
    theta_deg=60.0,
    y_qw=Y_QW_DEFAULT,
    dx=15e-9,
    dx_skin=None,
    x_half_min=2.50e-6,
    y_below=0.21e-6,
    y_above=0.48e-6,
    dz=0.5e-6,
    full_sidewall_skin=False,
):
    """
    Construct a 2D `(x, y)` `GridFIT3D` mesh using `wakis.templates.meshing.graded_axis`
    with nodes snapped to the device interfaces and local skin-depth refinement
    `dx_skin` around the silver (`Ag`) mirror and horn throat interfaces.

    Parameters
    ----------
    config : {'bare', 'air', 'sio2', 'dgrin'}
        Micro-horn configuration.
    h_horn : float
        Horn height [m].
    theta_deg : float
        Horn taper angle [deg] measured from horizontal.
    y_qw : float
        Quantum-well dipole plane height [m] above bottom Ag mirror (`y = 0`).
    dx : float
        Background cell size [m] inside GaN and dielectric layers.
    dx_skin : float or None
        Minimum cell size [m] across the Ag skin-depth zones (`None` -> `dx`).
    x_half_min : float
        Minimum half-width [m] of the simulation domain in `x`.
    y_below, y_above : float
        Vertical padding [m] below `y = 0` and above `y = y_top`.
    dz : float
        Periodic 1-cell thickness [m] in `z` (`dz = 0.5e-6` avoids restricting
        the 3D CFL timestep along the invariant `z` axis).
    full_sidewall_skin : bool
        If `True`, extend fine `dx_skin` across the entire sloped sidewall span
        (used in the staircase vs. Kottke resolution study).
    """
    dx = float(dx)
    dx_skin = float(dx if dx_skin is None else dx_skin)
    dz = float(dz)
    is_bare = config == "bare" or h_horn <= 0.0
    y_base = T_GAN
    y_top = y_base + (0.0 if is_bare else float(h_horn))
    w_top = (
        W_PIXEL
        if is_bare
        else W_PIXEL + 2.0 * float(h_horn) / np.tan(np.deg2rad(float(theta_deg)))
    )
    x_top_half = 0.5 * w_top
    x_half_target = max(float(x_half_min), x_top_half + 0.36e-6)
    nx_half = int(np.ceil(x_half_target / dx))
    x_max = nx_half * dx

    ny_below = max(8, int(np.round(float(y_below) / dx)))
    ny_above = int(np.ceil((y_top + float(y_above)) / dx))
    y_min = -ny_below * dx
    y_max = ny_above * dx

    if abs(dx_skin - dx) <= 1e-12:
        return build_uniform_aligned_grid(
            config=config,
            h_horn=h_horn,
            theta_deg=theta_deg,
            dx=dx,
            x_half_min=x_half_min,
            y_below=y_below,
            y_above=y_above,
            dz=dz,
        )

    w_skin_x = 0.06e-6
    if full_sidewall_skin and not is_bare:
        x_pts = [
            -x_max,
            -x_top_half - w_skin_x,
            -0.5 * W_PIXEL + w_skin_x,
            -0.5 * W_GAN,
            0.0,
            0.5 * W_GAN,
            0.5 * W_PIXEL - w_skin_x,
            x_top_half + w_skin_x,
            x_max,
        ]
        x_sz = [dx, dx_skin, dx, dx, dx, dx, dx_skin, dx]
    else:
        x_pts = [
            -x_max,
            -0.5 * W_PIXEL - w_skin_x,
            -0.5 * W_PIXEL,
            -0.5 * W_PIXEL + w_skin_x,
            -0.5 * W_GAN,
            0.0,
            0.5 * W_GAN,
            0.5 * W_PIXEL - w_skin_x,
            0.5 * W_PIXEL,
            0.5 * W_PIXEL + w_skin_x,
            x_max,
        ]
        x_sz = [dx, dx_skin, dx_skin, dx, dx, dx, dx, dx_skin, dx_skin, dx]
    x_nodes = graded_axis(x_pts, x_sz, min_cells=2, grading=1.35)

    w_skin_y = 0.03e-6
    y_mon = y_top + 0.15e-6
    if full_sidewall_skin and not is_bare:
        y_pts = [
            y_min,
            -0.06e-6,
            0.0,
            w_skin_y,
            float(y_qw),
            y_base - w_skin_y,
            y_base,
            y_top,
            y_mon,
            y_max,
        ]
        y_sz = [dx, dx_skin, dx_skin, dx, dx, dx_skin, dx_skin, dx, dx]
    else:
        y_pts = [
            y_min,
            -0.06e-6,
            0.0,
            w_skin_y,
            float(y_qw),
            y_base - w_skin_y,
            y_base,
            y_base + w_skin_y,
        ]
        y_sz = [dx, dx_skin, dx_skin, dx, dx, dx_skin, dx_skin]
        if not is_bare:
            if config == "dgrin":
                dh = float(h_horn) / len(DGRIN_INDICES)
                for k in range(1, len(DGRIN_INDICES)):
                    yk = y_base + k * dh
                    if yk > y_pts[-1] + 1e-9:
                        y_pts.append(yk)
                        y_sz.append(dx)
            if y_top > y_pts[-1] + 1e-9:
                y_pts.append(y_top)
                y_sz.append(dx)
        y_pts.extend([y_mon, y_max])
        y_sz.extend([dx, dx])
    y_nodes = graded_axis(y_pts, y_sz, min_cells=2, grading=1.35)
    z_nodes = np.array([-0.5 * dz, 0.5 * dz])

    grid = GridFIT3D(x=x_nodes, y=y_nodes, z=z_nodes, verbose=0)
    info = {
        "dx": dx,
        "dx_skin": dx_skin,
        "dz": dz,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "y_top": y_top,
        "w_top": w_top,
    }
    return grid, info


def compute_mesh_resolution_metrics(
    grid,
    config="dgrin",
    h_horn=H_HORN_DEFAULT,
    y_qw=Y_QW_DEFAULT,
    n_steps=0,
):
    """
    Measure background cell size, minimum cell size, cells per wavelength in GaN,
    smallest geometric feature cell count, and silver skin-depth resolution
    directly from the `GridFIT3D` node arrays `grid.x` and `grid.y`.

    Both skin-depth definitions used in the literature are reported:
    - Power / intensity skin depth: `delta_I = lambda_0 / (4 pi kappa) = 15.38 nm`
    - Electric-field skin depth: `delta_E = lambda_0 / (2 pi kappa) = 30.76 nm`
    """
    dxs_nm = np.diff(grid.x) * 1e9
    dys_nm = np.diff(grid.y) * 1e9
    dx_bg_nm = float(max(np.max(dxs_nm), np.max(dys_nm)))
    dx_min_nm = float(min(np.min(dxs_nm), np.min(dys_nm)))

    cpw_gan = float(((WL_0 / N_GAN) * 1e9) / dx_bg_nm)
    cpw_gan_min_wl = float(((WL_MIN / N_GAN) * 1e9) / dx_bg_nm)

    features = [
        (0.0, float(y_qw), grid.y),
        (float(y_qw), T_GAN, grid.y),
        (0.5 * W_GAN, 0.5 * W_PIXEL, grid.x),
        (-0.5 * W_GAN, 0.5 * W_GAN, grid.x),
    ]
    if config == "dgrin" and float(h_horn) > 0.0:
        dh = float(h_horn) / len(DGRIN_INDICES)
        for k in range(len(DGRIN_INDICES)):
            features.append((T_GAN + k * dh, T_GAN + (k + 1) * dh, grid.y))

    min_feature_nm = float(min((b - a) * 1e9 for a, b, _ in features))
    feat_counts = []
    for a, b, nodes in features:
        centers = 0.5 * (nodes[:-1] + nodes[1:])
        cnt = int(np.sum((centers >= a - 1e-12) & (centers <= b + 1e-12)))
        feat_counts.append((cnt, (b - a) * 1e9))
    cells_across_min_feat = float(
        min(c for c, w in feat_counts if abs(w - min_feature_nm) <= 1e-6)
    )
    cells_across_any_feat = float(min(c for c, _ in feat_counts))
    cells_across_feature = float(min(cells_across_min_feat, cells_across_any_feat))

    delta_e, delta_i = ag_skin_depth(WL_0)
    delta_e_nm = float(delta_e * 1e9)
    delta_i_nm = float(delta_i * 1e9)

    return {
        "dx_background_nm": np.float64(dx_bg_nm),
        "dx_min_nm": np.float64(dx_min_nm),
        "cpw_gan": np.float64(cpw_gan),
        "cpw_gan_min_wl": np.float64(cpw_gan_min_wl),
        "min_feature_nm": np.float64(min_feature_nm),
        "cells_across_feature": np.float64(cells_across_feature),
        "skin_depth_nm": np.float64(delta_i_nm),
        "skin_depth_E_nm": np.float64(delta_e_nm),
        "skin_depth_I_nm": np.float64(delta_i_nm),
        "cells_per_skin_depth": np.float64(delta_i_nm / dx_min_nm),
        "cells_per_skin_depth_E": np.float64(delta_e_nm / dx_min_nm),
        "cells_per_skin_depth_I": np.float64(delta_i_nm / dx_min_nm),
        "n_cells": np.int32(grid.Nx * grid.Ny * grid.Nz),
        "n_steps": np.int32(n_steps),
    }


def evaluate_horn_configuration(
    config="dgrin",
    h_horn=H_HORN_DEFAULT,
    theta_deg=60.0,
    y_qw=Y_QW_DEFAULT,
    dx=15e-9,
    dx_skin=None,
    averaging="kottke",
    dipole_xs=(0.0, 0.12e-6, 0.24e-6),
    dipole_weights=(1.0, 2.0, 2.0),
    n_freqs=21,
    n_angles=181,
    n_pml=8,
    run_extra_fs=None,
    collect_fields=False,
    record_movie=False,
    movie_frames=28,
    polarizations=("x", "y", "z"),
    x_half_min=2.50e-6,
    cfln=0.90,
    dz=0.5e-6,
    full_sidewall_skin=False,
    verbose_res=True,
    ag_material=None,
    horn_ag_material=None,
    passivation_material=None,
    y_mon_offset=0.16e-6,
    w_mon_override=None,
):
    """
    Run the 2D FIT multi-dipole, multi-polarization emission study for a µLED horn.
    """
    t0 = time.perf_counter()
    grid, info = build_graded_horn_grid(
        config=config,
        h_horn=h_horn,
        theta_deg=theta_deg,
        y_qw=y_qw,
        dx=dx,
        dx_skin=dx_skin,
        x_half_min=x_half_min,
        dz=dz,
        full_sidewall_skin=full_sidewall_skin,
    )
    dz = info["dz"]
    x_max = info["x_max"]
    y_min = info["y_min"]
    y_top = info["y_top"]
    w_top = info["w_top"]

    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=int(n_pml),
        cfln=float(cfln),
        verbose=0,
    )
    structures, _, _ = build_microled_horn_structures(
        config=config,
        h_horn=h_horn,
        theta_deg=theta_deg,
        x_max=x_max + 0.20e-6,
        y_min=y_min - 0.10e-6,
        z_bounds=(-1e-6, 1e-6),
        ag_material=ag_material,
        horn_ag_material=horn_ag_material,
        passivation_material=passivation_material,
    )
    f_center = c_light / WL_0
    assign_materials_2d_dispersive(solver, structures, f_center, averaging=averaging)

    lams_desc = np.linspace(WL_MIN, WL_MAX, int(n_freqs))
    freqs = c_light / lams_desc[::-1]
    wavelengths = c_light / freqs
    idx_f0 = int(np.argmin(np.abs(wavelengths - WL_0)))

    pulse = GaussianSource(frequency=f_center, fwidth=1.2e14, cutoff=4.0)
    S_f = pulse.fourier_transform(freqs)
    S_led = led_gaussian_spectrum(wavelengths, wl_center=WL_0, fwhm=WL_FWHM)
    inv_s2 = S_led / (np.abs(S_f) ** 2)

    alphas = np.linspace(-0.5 * np.pi, 0.5 * np.pi, int(n_angles))
    mask15 = np.abs(alphas) <= np.deg2rad(15.0)
    phi = 0.5 * np.pi - alphas
    theta = np.full_like(alphas, 0.5 * np.pi)

    if run_extra_fs is None:
        h_eff_um = 0.0 if (config == "bare" or h_horn <= 0.0) else float(h_horn) * 1e6
        run_extra_fs = 22.0 + 6.7 * h_eff_um
    Nt = int(np.ceil((pulse.end_time + float(run_extra_fs) * 1e-15) / solver.dt))
    y_mon = y_top + float(y_mon_offset)
    w_mon = (
        2.0 * (x_max - (n_pml + 2) * dx)
        if w_mon_override is None
        else float(w_mon_override)
    )

    dip_xs = [float(x) for x in dipole_xs]
    dip_ws = [float(w) for w in dipole_weights]
    n_pos = len(dip_xs)
    pols = tuple(p.lower() for p in polarizations)

    U_pos_pol = {
        c: np.zeros((n_pos, freqs.size, alphas.size), dtype=float) for c in pols
    }
    Phom_pos_pol = {c: np.zeros((n_pos, freqs.size), dtype=float) for c in pols}
    Pwrk_pos_pol = {c: np.zeros((n_pos, freqs.size), dtype=float) for c in pols}
    Pflux_pos_pol = {c: np.zeros((n_pos, freqs.size), dtype=float) for c in pols}

    Etheta_pos = np.zeros((n_pos, freqs.size, alphas.size), dtype=complex)
    Ephi_x_pos = np.zeros((n_pos, freqs.size, alphas.size), dtype=complex)
    scale_ff = None

    dft_payload = {}
    movie_payload = {}

    for m, (xd, wd) in enumerate(zip(dip_xs, dip_ws)):
        pos = (xd, float(y_qw), 0.0)

        # Pass 1: Simultaneous x (TE) and/or z (TM)
        run_xz = [c for c in ("x", "z") if c in pols]
        if run_xz:
            solver.reset_fields()
            if hasattr(solver, "dispersive_network"):
                solver.dispersive_network.reset()
            srcs = [
                InterpolatedCurrentSource2D(
                    src_time=pulse,
                    component=c,
                    center=pos,
                    size=(0.0, 0.0, 0.0),
                    amplitude=1.0,
                )
                for c in run_xz
            ]
            n2f_xz = Near2FarMonitor2D(
                freqs, (0.0, y_mon), (w_mon, 0.0), dz, eps_r=1.0, mu_r=1.0
            )
            pt_xz = InterpolatedPointDFTMonitor2D(
                frequencies=freqs,
                components=tuple(f"E{c}" for c in run_xz),
                center=(xd, float(y_qw)),
            )
            mons = [n2f_xz, pt_xz]

            do_fields_here = collect_fields and (m == 0)
            do_movie_here = record_movie and (m == 0)
            if do_fields_here:
                f0_arr = np.array([freqs[idx_f0]])
                full_dft = DFTMonitor(
                    frequencies=f0_arr,
                    components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                    center=(0.0, 0.5 * (y_min + info["y_max"]), 0.0),
                    size=(
                        2.0 * (x_max - n_pml * dx),
                        (info["y_max"] - y_min) - 2.0 * n_pml * dx,
                        0.0,
                    ),
                )
                mons.append(full_dft)

            if do_movie_here:
                stride = max(1, Nt // int(movie_frames))
                frames = []
                times_fs = []
                ix_lo, ix_hi = n_pml, grid.Nx - n_pml
                iy_lo, iy_hi = n_pml, grid.Ny - n_pml
                for step in range(Nt):
                    t = step * solver.dt
                    for s in srcs:
                        s.update(solver, t)
                    solver.one_step()
                    for mon in mons:
                        mon.update(solver, t)
                    if step % stride == 0 and len(frames) < int(movie_frames):
                        ez_slice = solver.E.to_matrix("z")[ix_lo:ix_hi, iy_lo:iy_hi, 0]
                        frames.append(np.asarray(ez_slice, dtype=np.float32).copy())
                        times_fs.append(float((t + solver.dt) * 1e15))
                solver.Nt = Nt
                movie_payload = {
                    "movie_x": solver.x[ix_lo:ix_hi].astype(np.float32),
                    "movie_y": solver.y[iy_lo:iy_hi].astype(np.float32),
                    "movie_times_fs": np.asarray(times_fs, dtype=np.float32),
                    "movie_frames": np.stack(frames, axis=0),
                }
            else:
                solver.emsolve(Nt, source=srcs, monitors=mons)

            ff = n2f_xz.get_farfield(theta, phi)
            scale = (2.0 * np.pi / (n2f_xz.wavenumber[:, None] * dz)) / n2f_xz.impedance
            scale_ff = scale
            flux_xz = n2f_xz.get_flux()

            if "z" in run_xz:
                Etheta_pos[m] = ff["Etheta"]
                u_z = (np.abs(ff["Etheta"]) ** 2) * scale
                if xd > 0:
                    u_z = 0.5 * (u_z + u_z[:, ::-1])
                Ez_pt = pt_xz.get_spectrum("Ez")
                U_pos_pol["z"][m] = u_z * inv_s2[:, None]
                Phom_pos_pol["z"][m] = (
                    analytic_2d_line_dipole_power(freqs, S_f, dz, "z") * inv_s2
                )
                Pwrk_pos_pol["z"][m] = (-np.real(Ez_pt * np.conj(S_f))) * inv_s2
                Pflux_pos_pol["z"][m] = (
                    np.trapezoid(u_z, alphas, axis=1) * inv_s2
                    if "x" in run_xz
                    else flux_xz * inv_s2
                )

            if "x" in run_xz:
                Ephi_x_pos[m] = ff["Ephi"]
                u_x = (np.abs(ff["Ephi"]) ** 2) * scale
                if xd > 0:
                    u_x = 0.5 * (u_x + u_x[:, ::-1])
                Ex_pt = pt_xz.get_spectrum("Ex")
                U_pos_pol["x"][m] = u_x * inv_s2[:, None]
                Phom_pos_pol["x"][m] = (
                    analytic_2d_line_dipole_power(freqs, S_f, dz, "x") * inv_s2
                )
                Pwrk_pos_pol["x"][m] = (-np.real(Ex_pt * np.conj(S_f))) * inv_s2
                Pflux_pos_pol["x"][m] = np.trapezoid(u_x, alphas, axis=1) * inv_s2

            if do_fields_here:
                s0 = S_f[idx_f0]
                Ex_m, coords_e = full_dft.get_field("Ex")
                Ey_m, _ = full_dft.get_field("Ey")
                Ez_m, _ = full_dft.get_field("Ez")
                Hx_m, _ = full_dft.get_field("Hx")
                Hy_m, _ = full_dft.get_field("Hy")
                Hz_m, _ = full_dft.get_field("Hz")
                raw = [
                    np.squeeze(a[0]) / s0
                    for a in (Ex_m, Ey_m, Ez_m, Hx_m, Hy_m, Hz_m)
                ]
                nx_m = min(a.shape[0] for a in raw)
                ny_m = min(a.shape[1] for a in raw)
                Ex_2d, Ey_2d, Ez_2d, Hx_2d, Hy_2d, Hz_2d = [
                    a[:nx_m, :ny_m] for a in raw
                ]
                Sx_2d = np.real(Ey_2d * np.conj(Hz_2d) - Ez_2d * np.conj(Hy_2d))
                Sy_2d = np.real(Ez_2d * np.conj(Hx_2d) - Ex_2d * np.conj(Hz_2d))
                E2_2d = np.abs(Ex_2d) ** 2 + np.abs(Ey_2d) ** 2 + np.abs(Ez_2d) ** 2
                dft_payload = {
                    "dft_x": coords_e[0][:nx_m].astype(np.float32),
                    "dft_y": coords_e[1][:ny_m].astype(np.float32),
                    "dft_Ez_real": np.real(Ez_2d).astype(np.float32),
                    "dft_Ex_real": np.real(Ex_2d).astype(np.float32),
                    "dft_E2": E2_2d.astype(np.float32),
                    "dft_Sx": Sx_2d.astype(np.float32),
                    "dft_Sy": Sy_2d.astype(np.float32),
                }

        # Pass 2: y (TE)
        if "y" in pols:
            solver.reset_fields()
            if hasattr(solver, "dispersive_network"):
                solver.dispersive_network.reset()
            src_y = InterpolatedCurrentSource2D(
                src_time=pulse,
                component="y",
                center=pos,
                size=(0.0, 0.0, 0.0),
                amplitude=1.0,
            )
            n2f_y = Near2FarMonitor2D(
                freqs, (0.0, y_mon), (w_mon, 0.0), dz, eps_r=1.0, mu_r=1.0
            )
            pt_y = InterpolatedPointDFTMonitor2D(
                frequencies=freqs,
                components=("Ey",),
                center=(xd, float(y_qw)),
            )
            solver.emsolve(Nt, source=[src_y], monitors=[n2f_y, pt_y])
            ff_y = n2f_y.get_farfield(theta, phi)
            scale = (2.0 * np.pi / (n2f_y.wavenumber[:, None] * dz)) / n2f_y.impedance
            u_y = (np.abs(ff_y["Ephi"]) ** 2) * scale
            if xd > 0:
                u_y = 0.5 * (u_y + u_y[:, ::-1])
            Ey_pt = pt_y.get_spectrum("Ey")
            U_pos_pol["y"][m] = u_y * inv_s2[:, None]
            Phom_pos_pol["y"][m] = (
                analytic_2d_line_dipole_power(freqs, S_f, dz, "y") * inv_s2
            )
            Pwrk_pos_pol["y"][m] = (-np.real(Ey_pt * np.conj(S_f))) * inv_s2
            Pflux_pos_pol["y"][m] = n2f_y.get_flux() * inv_s2

    w_arr_pos = np.asarray(dip_ws, dtype=float)
    w_norm = w_arr_pos / np.sum(w_arr_pos)
    n_pol = len(pols)

    U_pol = {c: np.einsum("m,mfa->fa", w_norm, U_pos_pol[c]) for c in pols}
    Phom_pol = {c: np.einsum("m,mf->f", w_norm, Phom_pos_pol[c]) for c in pols}
    Pwrk_pol = {c: np.einsum("m,mf->f", w_norm, Pwrk_pos_pol[c]) for c in pols}

    U_incoh = sum(U_pol[c] for c in pols) / n_pol
    Phom_tot = sum(Phom_pol[c] for c in pols) / n_pol
    Pwrk_tot = sum(Pwrk_pol[c] for c in pols) / n_pol

    U_ang = np.trapezoid(U_incoh, freqs, axis=0)
    Phom_int = float(np.trapezoid(Phom_tot, freqs))
    Pwrk_int = float(np.trapezoid(Pwrk_tot, freqs))

    rho_lee_rad = U_ang / Phom_int
    rho_lee_deg = 100.0 * rho_lee_rad * (np.pi / 180.0)

    lee15_pct = float(100.0 * np.trapezoid(rho_lee_rad[mask15], alphas[mask15]))
    lee90_pct = float(100.0 * np.trapezoid(rho_lee_rad, alphas))
    lee15_wrk_pct = float(
        100.0 * np.trapezoid(U_ang[mask15], alphas[mask15]) / Pwrk_int
    )
    lee90_wrk_pct = float(100.0 * np.trapezoid(U_ang, alphas) / Pwrk_int)
    purcell_factor = float(Pwrk_int / Phom_int)

    lee15_by_pol = {}
    lee90_by_pol = {}
    for c in pols:
        uc_ang = np.trapezoid(U_pol[c], freqs, axis=0)
        ph_c = float(np.trapezoid(Phom_pol[c], freqs))
        lee15_by_pol[c] = float(
            100.0 * np.trapezoid(uc_ang[mask15], alphas[mask15]) / ph_c
        )
        lee90_by_pol[c] = float(100.0 * np.trapezoid(uc_ang, alphas) / ph_c)
    lee15_eq_pct = float(np.mean([lee15_by_pol[c] for c in pols]))
    lee90_eq_pct = float(np.mean([lee90_by_pol[c] for c in pols]))

    conv_lee15 = np.zeros(n_pos, dtype=float)
    conv_lee90 = np.zeros(n_pos, dtype=float)
    conv_lee15_eq = np.zeros(n_pos, dtype=float)
    conv_lee90_eq = np.zeros(n_pos, dtype=float)
    conv_ndip_full = np.zeros(n_pos, dtype=np.int32)
    for k in range(1, n_pos + 1):
        wk = w_arr_pos[:k] / np.sum(w_arr_pos[:k])
        uk = sum(np.einsum("m,mfa->fa", wk, U_pos_pol[c][:k]) for c in pols) / n_pol
        pk = sum(np.einsum("m,mf->f", wk, Phom_pos_pol[c][:k]) for c in pols) / n_pol
        uk_ang = np.trapezoid(uk, freqs, axis=0)
        pk_int = float(np.trapezoid(pk, freqs))
        conv_lee15[k - 1] = (
            100.0 * np.trapezoid(uk_ang[mask15], alphas[mask15]) / pk_int
        )
        conv_lee90[k - 1] = 100.0 * np.trapezoid(uk_ang, alphas) / pk_int
        l15_k_pols = []
        l90_k_pols = []
        for c in pols:
            uck = np.einsum("m,mfa->fa", wk, U_pos_pol[c][:k])
            pck = np.einsum("m,mf->f", wk, Phom_pos_pol[c][:k])
            uck_ang = np.trapezoid(uck, freqs, axis=0)
            pck_int = float(np.trapezoid(pck, freqs))
            l15_k_pols.append(
                100.0 * np.trapezoid(uck_ang[mask15], alphas[mask15]) / pck_int
            )
            l90_k_pols.append(100.0 * np.trapezoid(uck_ang, alphas) / pck_int)
        conv_lee15_eq[k - 1] = float(np.mean(l15_k_pols))
        conv_lee90_eq[k - 1] = float(np.mean(l90_k_pols))
        conv_ndip_full[k - 1] = int(np.sum([1 if x == 0 else 2 for x in dip_xs[:k]]))

    if "z" in pols and scale_ff is not None:
        E_coh = np.zeros_like(Etheta_pos[0])
        n_full = 0.0
        for m, xd in enumerate(dip_xs):
            if xd == 0.0:
                E_coh += Etheta_pos[m]
                n_full += 1.0
            else:
                E_coh += Etheta_pos[m] + Etheta_pos[m, :, ::-1]
                n_full += 2.0
        u_coh_z = (np.abs(E_coh) ** 2 / n_full) * scale_ff * inv_s2[:, None]
        u_coh_ang = np.trapezoid(u_coh_z, freqs, axis=0)
        ph_z_int = float(np.trapezoid(Phom_pol["z"], freqs))
        rho_coh_z_deg = 100.0 * (u_coh_ang / ph_z_int) * (np.pi / 180.0)
        u_inc_z_ang = np.trapezoid(U_pol["z"], freqs, axis=0)
        rho_inc_z_deg = 100.0 * (u_inc_z_ang / ph_z_int) * (np.pi / 180.0)
    else:
        rho_coh_z_deg = rho_lee_deg.copy()
        rho_inc_z_deg = rho_lee_deg.copy()

    lee15_spectrum = (
        100.0 * np.trapezoid(U_incoh[:, mask15], alphas[mask15], axis=1) / Phom_tot
    )
    lee90_spectrum = 100.0 * np.trapezoid(U_incoh, alphas, axis=1) / Phom_tot

    res_metrics = compute_mesh_resolution_metrics(
        grid, config=config, h_horn=h_horn, y_qw=y_qw, n_steps=Nt
    )
    elapsed_s = float(time.perf_counter() - t0)
    if verbose_res:
        print(
            f"  [mesh] cfg={config} avg={averaging} "
            f"dx_bg={res_metrics['dx_background_nm']:.2f}nm ({res_metrics['cpw_gan']:.1f} cpw @450nm, "
            f"{res_metrics['cpw_gan_min_wl']:.1f} cpw @425nm), "
            f"dx_min={res_metrics['dx_min_nm']:.2f}nm, "
            f"min_feat={res_metrics['min_feature_nm']:.0f}nm ({res_metrics['cells_across_feature']:.0f} cells), "
            f"delta_I={res_metrics['skin_depth_nm']:.2f}nm ({res_metrics['cells_per_skin_depth']:.2f} c/delta_I, "
            f"{res_metrics['cells_per_skin_depth_E']:.2f} c/delta_E), "
            f"n_cells={int(res_metrics['n_cells'])}, n_steps={Nt}, "
            f"LEE15={lee15_pct:.2f}%, LEE90={lee90_pct:.2f}% ({elapsed_s:.1f}s)"
        )

    out = {
        "config": np.array(config),
        "averaging": np.array(averaging),
        "h_horn": np.float64(h_horn),
        "theta_deg": np.float64(theta_deg),
        "y_qw": np.float64(y_qw),
        "w_top": np.float64(w_top),
        "dx": np.float64(dx),
        "dx_skin": np.float64(info["dx_skin"]),
        **res_metrics,
        "grid_shape": np.array([grid.Nx, grid.Ny, grid.Nz], dtype=np.int32),
        "elapsed_s": np.float64(elapsed_s),
        "frequencies": freqs.astype(np.float64),
        "wavelengths": wavelengths.astype(np.float64),
        "led_spectrum": S_led.astype(np.float64),
        "angles_rad": alphas.astype(np.float64),
        "angles_deg": np.rad2deg(alphas).astype(np.float64),
        "rho_lee_deg": rho_lee_deg.astype(np.float64),
        "rho_coh_z_deg": rho_coh_z_deg.astype(np.float64),
        "rho_inc_z_deg": rho_inc_z_deg.astype(np.float64),
        "lee15_pct": np.float64(lee15_pct),
        "lee90_pct": np.float64(lee90_pct),
        "lee15_eq_pct": np.float64(lee15_eq_pct),
        "lee90_eq_pct": np.float64(lee90_eq_pct),
        "collimation_ratio_pct": np.float64(100.0 * lee15_pct / max(lee90_pct, 1e-12)),
        "lee15_wrk_pct": np.float64(lee15_wrk_pct),
        "lee90_wrk_pct": np.float64(lee90_wrk_pct),
        "purcell_factor": np.float64(purcell_factor),
        "fwhm_deg": np.float64(half_power_beamwidth_deg(alphas, rho_lee_deg)),
        "directivity_2d": np.float64(directivity_2d_halfspace(alphas, rho_lee_deg)),
        "lee15_pol_x": np.float64(lee15_by_pol.get("x", np.nan)),
        "lee15_pol_y": np.float64(lee15_by_pol.get("y", np.nan)),
        "lee15_pol_z": np.float64(lee15_by_pol.get("z", np.nan)),
        "lee90_pol_x": np.float64(lee90_by_pol.get("x", np.nan)),
        "lee90_pol_y": np.float64(lee90_by_pol.get("y", np.nan)),
        "lee90_pol_z": np.float64(lee90_by_pol.get("z", np.nan)),
        "lee15_spectrum": lee15_spectrum.astype(np.float64),
        "lee90_spectrum": lee90_spectrum.astype(np.float64),
        "conv_ndip_full": conv_ndip_full,
        "conv_lee15": conv_lee15,
        "conv_lee90": conv_lee90,
        "conv_lee15_eq": conv_lee15_eq,
        "conv_lee90_eq": conv_lee90_eq,
    }
    out.update(dft_payload)
    out.update(movie_payload)
    return out


def extract_2d_eps_map(
    config="dgrin",
    h_horn=H_HORN_DEFAULT,
    theta_deg=60.0,
    dx=15e-9,
    averaging="kottke",
):
    """
    Build the 2D grid and return `(x_nodes, y_nodes, eps_x, eps_y, eps_z, f_ag)`
    for visualizing the device cross-section and Kottke vs staircase sidewall mesh.
    """
    grid, info = build_uniform_aligned_grid(
        config=config, h_horn=h_horn, theta_deg=theta_deg, dx=dx
    )
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=8,
        cfln=0.5,
        verbose=0,
    )
    structures, y_top, w_top = build_microled_horn_structures(
        config=config,
        h_horn=h_horn,
        theta_deg=theta_deg,
        x_max=info["x_max"] + 0.20e-6,
        y_min=info["y_min"] - 0.10e-6,
        t_skin=None,
    )
    f_center = c_light / WL_0
    assign_materials_2d_dispersive(solver, structures, f_center, averaging=averaging)

    nx, ny, nz = grid.Nx, grid.Ny, grid.Nz
    N = nx * ny * nz
    ieps = solver.ieps.array
    with np.errstate(divide="ignore"):
        eps_all = np.where(ieps > 0, 1.0 / (epsilon_0 * ieps), np.inf)
    eps_x = eps_all[0:N].reshape((nx, ny, nz), order="F")[:, :, 0]
    eps_y = eps_all[N : 2 * N].reshape((nx, ny, nz), order="F")[:, :, 0]
    eps_z = eps_all[2 * N : 3 * N].reshape((nx, ny, nz), order="F")[:, :, 0]

    f_ag = np.zeros((nx, ny), dtype=float)
    if hasattr(solver, "dispersive_network"):
        net = solver.dispersive_network
        rows_z = net.rows[(net.rows >= 2 * N) & (net.rows < 3 * N)] - 2 * N
        f_flat = np.zeros(N, dtype=float)
        f_flat[rows_z] = 1.0
        f_ag = f_flat.reshape((nx, ny, nz), order="F")[:, :, 0]

    return {
        "x_nodes": grid.x,
        "y_nodes": grid.y,
        "x_centers": solver.x,
        "y_centers": solver.y,
        "eps_x": eps_x,
        "eps_y": eps_y,
        "eps_z": eps_z,
        "f_ag": f_ag,
        "y_top": y_top,
        "w_top": w_top,
    }
