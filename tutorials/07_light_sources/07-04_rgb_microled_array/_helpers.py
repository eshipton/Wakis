"""
Helper functions and verified material/geometry definitions for Tutorial 7.4:
RGB microLED array — incoherent averaging, collimation, and pixel crosstalk.

Reproduces the 2D FDTD baseline of:
L. Chen, Z. Li, P. Wang, S. Wu, W. Li, J. Wang, Y. Cao, M. Mortazavi,
L. Peng, and P. Wu, "Dual-Task Optimization Method for Inverse Design of
RGB Micro-LED Light Collimator", Nanomaterials 15(3), 190 (2025),
DOI: 10.3390/nano15030190 (PMC11820347).
"""

from __future__ import annotations

import time
import numpy as np
from scipy.constants import c as C0, epsilon_0 as EPS_0, mu_0 as MU_0

import wakis
from wakis.dispersive import DispersiveMaterial
from wakis.field_monitors import DFTMonitor
from wakis.sources import CurrentSource, GaussianSource, PlaneWave
from wakis.templates.shapes import Box, assign_materials

CHANNELS = ("B", "G", "R")

# Exact material parameters from Table 1 of Chen et al., Nanomaterials 15, 190 (2025)
RGB_MATERIALS = {
    "B": {
        "label": "Blue (457 nm)",
        "color": "#2b6cb0",
        "wavelength_m": 457e-9,
        "n_GaN": 2.49,
        "n_MQW": 2.51,
        "n_PI": 1.50,
        "n_SiO2": 1.465,
        "n_SiNx": 2.0475,
        "n_Si_real": 4.583,
        "n_Si_imag": 0.13,
        "x_center_m": -1.50e-6,
    },
    "G": {
        "label": "Green (527 nm)",
        "color": "#2f855a",
        "wavelength_m": 527e-9,
        "n_GaN": 2.38,
        "n_MQW": 2.41,
        "n_PI": 1.50,
        "n_SiO2": 1.4609,
        "n_SiNx": 2.0279,
        "n_Si_real": 4.215,
        "n_Si_imag": 0.048,
        "x_center_m": 0.00e-6,
    },
    "R": {
        "label": "Red (611 nm)",
        "color": "#c53030",
        "wavelength_m": 611e-9,
        "n_GaN": 2.35,
        "n_MQW": 2.37,
        "n_PI": 1.50,
        "n_SiO2": 1.4577,
        "n_SiNx": 2.0133,
        "n_Si_real": 3.906,
        "n_Si_imag": 0.022,
        "x_center_m": +1.50e-6,
    },
}

# Cross-sectional geometry from Section 2.1-2.3 and Figures 3a, 4a of Chen et al. (2025)
PIXEL_GEOMETRY = {
    "unit_pitch_m": 4.50e-6,  # Full RGB pixel pitch [m]
    "subpixel_pitch_m": 1.50e-6,  # Center-to-center subpixel spacing [m]
    "channel_width_m": 1.05e-6,  # Width of each R/G/B GaN/MQW mesa [m]
    "trench_width_m": 0.45e-6,  # Polyimide isolation gap between mesas [m]
    "t_pGaN_m": 0.20e-6,  # Top p-GaN thickness (y in [-0.20, 0.00] um) [m]
    "t_MQW_m": 0.20e-6,  # Active MQW thickness (y in [-0.40, -0.20] um) [m]
    "t_nGaN_m": 0.70e-6,  # Bottom n-GaN thickness (y in [-1.10, -0.40] um) [m]
    "y_top_m": 0.00e-6,  # Top surface of p-GaN [m]
    "y_mqw_center_m": -0.30e-6,  # Center of MQW (300 nm below y=0) [m]
    "y_si_top_m": -1.10e-6,  # Top of absorbing Silicon substrate [m]
    "collimator_height_m": 2.00e-6,  # Design region height above y=0 [m]
    "lambda_min_m": 457e-9,  # Shortest wavelength (Blue) [m]
    "n_max_active": 2.51,  # Max index in LED dielectric stack (Blue MQW)
    "n_max_silicon": 4.583,  # Max real index in Silicon substrate (Blue)
}

# Published reference metrics from Chen et al., Nanomaterials 15, 190 (2025)
# Section 3.1 (Fig. 3d), Section 3.2 (Fig. 4c,e,g), and Section 4 (Fig. S5)
PAPER_TARGETS = {
    "B": {
        "eta_20_uncol_pct": 30.29,
        "eta_20_col_pct": 65.60,
        "eta_20_filt_pct": 59.47,
        "lee_tot_uncol_pct": 26.63,
        "lee_tot_col_pct": 27.82,
        "lee_tot_filt_pct": 28.09,
        "oe_router_col_pct": 79.5,
    },
    "G": {
        "eta_20_uncol_pct": 34.84,
        "eta_20_col_pct": 55.94,
        "eta_20_filt_pct": 54.00,
        "lee_tot_uncol_pct": 26.53,
        "lee_tot_col_pct": 24.84,
        "lee_tot_filt_pct": 22.90,
        "oe_router_col_pct": 81.0,
    },
    "R": {
        "eta_20_uncol_pct": 35.83,
        "eta_20_col_pct": 57.70,
        "eta_20_filt_pct": 57.97,
        "lee_tot_uncol_pct": 23.98,
        "lee_tot_col_pct": 27.60,
        "lee_tot_filt_pct": 24.19,
        "oe_router_col_pct": 79.8,
    },
}


from wakis.templates.meshing import graded_axis, merge_points


def grid_spacing_from_cpw(cpw: float, n_ref: float | None = None) -> float:
    """Compute grid cell size dx from cells-per-wavelength in reference material.

    Parameters
    ----------
    cpw : float
        Cells per minimum wavelength ($\\lambda_{\\min} = 457\\;\\text{nm}$).
    n_ref : float or None, optional
        Reference refractive index. Defaults to ``PIXEL_GEOMETRY["n_max_active"]``
        ($n_{\\max,\\text{MQW}} = 2.51$).

    Returns
    -------
    float
        Grid spacing $\\Delta x = \\lambda_{\\min} / (\\text{cpw} \\cdot n_{\\text{ref}})$ [m].
    """
    n_val = PIXEL_GEOMETRY["n_max_active"] if n_ref is None else float(n_ref)
    return PIXEL_GEOMETRY["lambda_min_m"] / (float(cpw) * n_val)


def silicon_skin_depth(
    wavelength_m: float, k_imag: float, dx_m: float | None = None
) -> dict[str, float]:
    """Calculate optical penetration and field skin depths in lossy Silicon.

    We report both the $1/e$ power/intensity absorption skin depth
    $\\delta_P = \\lambda_0 / (4\\pi \\kappa)$ and the $1/e$ electric-field amplitude
    skin depth $\\delta_E = \\lambda_0 / (2\\pi \\kappa) = 2\\delta_P$. The canonical
    ``skin_depth_nm`` and ``cells_per_skin_depth`` keys use the stricter
    $\\delta = \\lambda_0 / (4\\pi \\kappa)$ definition.

    Parameters
    ----------
    wavelength_m : float
        Free-space wavelength $\\lambda_0$ [m].
    k_imag : float
        Imaginary part $\\kappa$ of the complex refractive index $n + i\\kappa$.
    dx_m : float or None, optional
        Actual mesh cell size $\\Delta y_{\\text{Si}}$ [m] inside Silicon.

    Returns
    -------
    dict of str to float
        Dictionary with ``delta_field_m`` ($\\lambda_0 / (2\\pi \\kappa)$ [m]),
        ``delta_power_m`` ($\\lambda_0 / (4\\pi \\kappa)$ [m]),
        ``skin_depth_nm`` ($\\delta_P$ [nm]), ``field_skin_depth_nm`` ($\\delta_E$ [nm]),
        and when ``dx_m`` is given, ``cells_per_skin_depth`` ($\\delta_P / \\Delta y_{\\text{Si}}$),
        ``cells_per_power_skin_depth``, and ``cells_per_field_skin_depth``.
    """
    delta_e = float(wavelength_m) / (2.0 * np.pi * float(k_imag))
    delta_p = 0.5 * delta_e
    out = {
        "delta_field_m": delta_e,
        "delta_power_m": delta_p,
        "skin_depth_nm": float(delta_p * 1e9),
        "field_skin_depth_nm": float(delta_e * 1e9),
    }
    if dx_m is not None and dx_m > 0:
        out["cells_per_field_skin_depth"] = delta_e / float(dx_m)
        out["cells_per_power_skin_depth"] = delta_p / float(dx_m)
        out["cells_per_skin_depth"] = delta_p / float(dx_m)
    return out


def lambertian_2d_density(theta_deg: np.ndarray) -> np.ndarray:
    """Normalized 2D Lambertian angular power ratio density $\\bar{P}_r(\\theta)$.

    Defined according to Equation (5) of Chen et al. (2025):
    $\\bar{P}_r(\\theta) = P(\\theta) / \\sum_{\\theta} P(\\theta)\\,\\Delta\\theta$
    where $P(\\theta) \\propto \\cos\\theta$ on $\\theta \\in [-90^\\circ, 90^\\circ]$.

    Parameters
    ----------
    theta_deg : np.ndarray
        Uniformly spaced far-field angles $\\theta$ [deg] in $[-90, 90]$.

    Returns
    -------
    np.ndarray
        Normalized angular power density $\\bar{P}_r(\\theta)$ [$\\text{deg}^{-1}$].
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    raw = np.clip(np.cos(np.deg2rad(theta_deg)), 0.0, None)
    dtheta = 180.0 / len(theta_deg)
    return raw / (np.sum(raw) * dtheta)


def lambertian_2d_cone_fraction(half_angle_deg: float = 20.0) -> float:
    """Exact 2D Lambertian fractional power within $\\pm\\theta_0$.

    In 2D:
    $\\eta_{\\text{Lamb,2D}}(\\theta_0) = \\frac{\\int_{-\\theta_0}^{\\theta_0} \\cos\\theta\\,d\\theta}{\\int_{-\\pi/2}^{\\pi/2} \\cos\\theta\\,d\\theta} = \\sin\\theta_0$.

    Parameters
    ----------
    half_angle_deg : float, optional
        Half-angle $\\theta_0$ [deg]. Default is 20.0.

    Returns
    -------
    float
        Fraction of power inside $|\\theta| \\le \\theta_0$ (dimensionless, in [0, 1]).
    """
    return float(np.sin(np.deg2rad(half_angle_deg)))


def lambertian_3d_cone_fraction(half_angle_deg: float = 20.0) -> float:
    """Exact 3D axisymmetric Lambertian fractional power within $\\pm\\theta_0$.

    In 3D:
    $\\eta_{\\text{Lamb,3D}}(\\theta_0) = \\frac{\\int_0^{\\theta_0} \\cos\\theta \\sin\\theta\\,d\\theta}{\\int_0^{\\pi/2} \\cos\\theta \\sin\\theta\\,d\\theta} = \\sin^2\\theta_0$.

    Parameters
    ----------
    half_angle_deg : float, optional
        Half-angle $\\theta_0$ [deg]. Default is 20.0.

    Returns
    -------
    float
        Fraction of power inside $\\theta \\le \\theta_0$ (dimensionless, in [0, 1]).
    """
    return float(np.sin(np.deg2rad(half_angle_deg)) ** 2)


def etendue_collimation_bound_2d(
    aperture_width_m: float, emitter_width_m: float, half_angle_deg: float = 20.0
) -> float:
    """Thermodynamic 2D étendue upper bound on collimation efficiency into $\\pm\\theta_0$.

    For a spatially incoherent 2D Lambertian emitter of width $W_{\\text{em}}$ radiating
    into air ($n=1$), passive optics of exit aperture $W_{\\text{ap}}$ conserving
    phase-space étendue $\\mathcal{E} = 2 n W \\sin\\theta$ satisfy:
    $\\eta_{\\max}(\\pm\\theta_0) = \\min(1,\\, (W_{\\text{ap}} / W_{\\text{em}}) \\sin\\theta_0)$.

    Parameters
    ----------
    aperture_width_m : float
        Exit aperture width $W_{\\text{ap}}$ [m].
    emitter_width_m : float
        Active emitter width $W_{\\text{em}}$ [m].
    half_angle_deg : float, optional
        Target half-angle $\\theta_0$ [deg]. Default is 20.0.

    Returns
    -------
    float
        Maximum passive coupling fraction in [0, 1].
    """
    ratio = (float(aperture_width_m) / float(emitter_width_m)) * np.sin(
        np.deg2rad(half_angle_deg)
    )
    return float(min(1.0, max(0.0, ratio)))


def escape_cone_2d_analytical(
    n_active: float,
    n_top: float = 1.0,
    n_bot_real: float = 4.2,
    n_bot_imag: float = 0.05,
) -> dict[str, float]:
    """Analytical 2D planar escape-cone and Fresnel extraction estimate.

    Parameters
    ----------
    n_active : float
        Refractive index of the active GaN/MQW slab.
    n_top : float, optional
        Refractive index of the top superstrate (1.0 for air).
    n_bot_real : float, optional
        Real part of the bottom Silicon substrate refractive index.
    n_bot_imag : float, optional
        Imaginary part of the bottom Silicon substrate refractive index.

    Returns
    -------
    dict of str to float
        Dictionary containing ``theta_c_deg`` (critical angle [deg]),
        ``single_cone_2d`` ($\\theta_c / \\pi$), `` single_cone_3d``
        ($\\frac{1}{2}(1 - \\cos\\theta_c)$), ``T_fresnel_avg`` (angle-averaged TM
        transmission inside escape cone), ``R_si_normal`` (normal-incidence
        GaN/Si reflectance), and ``lee_2d_planar_est`` (estimated 2D extraction
        efficiency including single bottom bounce).
    """
    sin_tc = min(1.0, float(n_top) / float(n_active))
    theta_c = np.arcsin(sin_tc)
    single_2d = float(theta_c / np.pi)
    single_3d = float(0.5 * (1.0 - np.cos(theta_c)))

    # Angle-averaged TM (in-plane E) Fresnel transmission inside [-theta_c, theta_c]
    thetas = np.linspace(0.0, theta_c * 0.999, 200)
    cos_i = np.cos(thetas)
    sin_t = (n_active / n_top) * np.sin(thetas)
    cos_t = np.sqrt(np.clip(1.0 - sin_t**2, 0.0, None))
    r_tm = (n_top * cos_i - n_active * cos_t) / (n_top * cos_i + n_active * cos_t)
    t_tm_avg = float(np.mean(1.0 - np.abs(r_tm) ** 2))

    # Normal-incidence reflectance at GaN / lossy-Si bottom interface
    n_si = complex(n_bot_real, n_bot_imag)
    r_si = abs((n_active - n_si) / (n_active + n_si)) ** 2

    # Planar top escape cone + bottom Si reflection + mesa sidewall escape via polyimide (n=1.50)
    lee_planar_only = single_2d * t_tm_avg * (1.0 + r_si)
    n_pi = PIXEL_GEOMETRY.get("n_PI", 1.50)
    theta_c_side = np.arcsin(min(1.0, n_pi / float(n_active)))
    sidewall_fraction = float((theta_c_side - theta_c) / np.pi) * t_tm_avg
    lee_mesa_est = lee_planar_only + sidewall_fraction
    return {
        "theta_c_deg": float(np.rad2deg(theta_c)),
        "single_cone_2d": single_2d,
        "single_cone_3d": single_3d,
        "T_fresnel_avg": t_tm_avg,
        "R_si_normal": float(r_si),
        "lee_2d_planar_only": float(lee_planar_only),
        "lee_2d_planar_est": float(lee_mesa_est),
    }


def random_phase_rse_theory(
    n_realizations: int | np.ndarray, n_dipoles: int = 20
) -> np.ndarray:
    """Analytical relative standard error (RSE) of Method B random-phase intensity.

    For $N$ equal-power incoherent dipoles superposed with i.i.d. uniform random
    phases $\\phi_m^{(k)} \\sim \\mathcal{U}(0, 2\\pi)$ across $K$ realizations,
    the pointwise angular intensity estimator
    $\\hat{I}^{(K)}(\\theta) = \\frac{1}{K}\\sum_{k=1}^K |\\sum_{m=1}^N E_m(\\theta) e^{i\\phi_m^{(k)}}|^2$
    is unbiased ($\\mathbb{E}[\\hat{I}^{(K)}] = \\sum_m |E_m|^2$) and has exact
    relative standard deviation:
    $\\text{RSE}(K, N) = \\sqrt{\\frac{1 - 1/N}{K}}$.

    Parameters
    ----------
    n_realizations : int or np.ndarray
        Number of random-phase realizations $K \\ge 1$.
    n_dipoles : int, optional
        Number of simultaneous dipoles $N$. Default is 20.

    Returns
    -------
    np.ndarray
        Theoretical relative standard error $\\sigma_I / I_{\\text{inc}}$.
    """
    k_arr = np.asarray(n_realizations, dtype=float)
    factor = max(0.0, 1.0 - 1.0 / float(n_dipoles))
    return np.sqrt(factor / k_arr)


def reset_solver_state(solver: wakis.SolverFIT3D) -> None:
    """Reset all dynamic fields, dispersive memory, and CPML auxiliary arrays.

    Addresses Wakis kit gap where ``SolverFIT3D.reset_fields()`` resets ``E``,
    ``H``, ``J``, and dispersive auxiliary fields, but leaves the CPML auxiliary
    arrays (``psiHa_*``, ``psiHb_*``, ``psiEa_*``, ``psiEb_*``) non-zero.

    Parameters
    ----------
    solver : wakis.SolverFIT3D
        Active FIT solver instance to reset in place.
    """
    solver.reset_fields()
    for name in dir(solver):
        if name.startswith(("psiHa_", "psiHb_", "psiEa_", "psiEb_")):
            arr = getattr(solver, name)
            if hasattr(arr, "fill"):
                arr.fill(0.0)


class RandomPhaseTimeProfile:
    """Gaussian-modulated carrier waveform with arbitrary phase offset $\\phi$.

    Used in Method B simultaneous multi-dipole incoherent simulations where each
    dipole $m$ carries an independent random phase $\\phi_m \\sim \\mathcal{U}(0, 2\\pi)$:
    $s_m(t) = A \\exp\\!\\left(-\\frac{(t - t_0)^2}{2\\sigma_t^2}\\right) \\cos\\!\\left(\\omega_0(t - t_0) + \\phi_m\\right)$.

    Parameters
    ----------
    frequency : float
        Carrier frequency $f_0$ [Hz].
    fwidth : float
        Gaussian bandwidth $\\Delta f$ [Hz] ($\\sigma_t = 1 / (2\\pi\\,\\Delta f)$).
    phase_rad : float, optional
        Carrier phase $\\phi_m$ [rad]. Default is 0.0.
    amplitude : float, optional
        Peak amplitude $A$. Default is 1.0.
    cutoff : float, optional
        Delay in units of $\\sigma_t$ ($t_0 = \\text{cutoff} \\cdot \\sigma_t$). Default is 4.0.
    """

    def __init__(
        self,
        frequency: float,
        fwidth: float,
        phase_rad: float = 0.0,
        amplitude: float = 1.0,
        cutoff: float = 4.0,
    ) -> None:
        self.frequency = float(frequency)
        self.fwidth = float(fwidth)
        self.phase_rad = float(phase_rad)
        self.amplitude = float(amplitude)
        self.cutoff = float(cutoff)
        self.sigma = 1.0 / (2.0 * np.pi * self.fwidth)
        self.t0 = self.cutoff * self.sigma
        self.end_time = 2.0 * self.t0

    def __call__(self, t: float) -> float:
        tau = float(t) - self.t0
        env = np.exp(-0.5 * (tau / self.sigma) ** 2)
        return float(
            self.amplitude
            * env
            * np.cos(2.0 * np.pi * self.frequency * tau + self.phase_rad)
        )


class MultiSource:
    """Composite source that updates multiple Wakis source objects at each step.

    Parameters
    ----------
    sources : list
        Sequence of Wakis source objects (e.g., ``CurrentSource`` instances).
    """

    def __init__(self, sources: list) -> None:
        self.sources = list(sources)

    def update(self, solver: wakis.SolverFIT3D, t: float) -> None:
        for src in self.sources:
            src.update(solver, t)


def make_flux_box_2d(
    freqs: list[float] | np.ndarray,
    center: tuple[float, float, float],
    width_x: float,
    height_y: float,
) -> wakis.FluxMonitor:
    """Create a 2D closed rectangular Poynting flux box in the $xy$-plane.

    Addresses Wakis kit gap where a 3D ``FluxMonitor`` with finite ``size[2]``
    includes $z$-faces and halves edge cell areas on $z$-boundaries. Setting
    ``size[2] = np.inf`` and restricting ``faces`` to in-plane normal axes
    ``(0, 1)`` yields an exact 4-sided 2D contour integral around the active region.

    Parameters
    ----------
    freqs : sequence of float
        Frequencies [Hz] at which to accumulate DFT fields.
    center : tuple of float
        Box center ``(x0, y0, z0)`` [m].
    width_x : float
        Box width along $x$ [m].
    height_y : float
        Box height along $y$ [m].

    Returns
    -------
    wakis.FluxMonitor
        Configured 4-face 2D ``FluxMonitor``.
    """
    mon = wakis.FluxMonitor(
        freqs, center=center, size=(float(width_x), float(height_y), np.inf)
    )
    mon.faces = [face for face in mon.faces if face[0] in (0, 1)]
    return mon


def project_near2far_2d(
    n2f_monitor: wakis.Near2FarMonitor,
    theta_deg: np.ndarray,
    dz_m: float,
    freq_idx: int = 0,
) -> dict[str, np.ndarray | float]:
    """Convert Wakis 3D ``Near2FarMonitor`` output on $z$-extruded line to 2D far field.

    For a 2D domain of thickness $\\Delta z$ (with ``Nz = 1`` and periodic $z$ BCs),
    integrating constant equivalent surface currents over $z \\in [-\\Delta z/2, \\Delta z/2]$
    in the $xy$-plane ($\\theta_{\\text{3D}} = \\pi/2$, $\\phi_{\\text{3D}} = \\pi/2 - \\theta$)
    relates the 2D cylindrical radiation intensity $U_{\\text{2D}}(\\theta)$ [W s$^2$/rad]
    to Wakis's 3D spherical ``radiation_intensity`` $U_{\\text{3D}}$ via:
    $U_{\\text{2D}}(\\theta) = \\frac{2\\pi}{k_0\\,\\Delta z}\\,U_{\\text{3D}}(\\theta_{\\text{3D}}=\\pi/2,\\,\\phi_{\\text{3D}}=\\pi/2 - \\theta)$.

    Parameters
    ----------
    n2f_monitor : wakis.Near2FarMonitor
        Completed ``Near2FarMonitor`` placed at a horizontal line $y = y_{\\text{mon}}$.
    theta_deg : np.ndarray
        Far-field angles $\\theta \\in [-90^\\circ, 90^\\circ]$ measured from $+y$ normal [deg].
    dz_m : float
        Out-of-plane cell thickness $\\Delta z$ [m].
    freq_idx : int, optional
        Frequency index in ``n2f_monitor.freqs``. Default is 0.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``u_2d`` : 2D angular radiation intensity $U_{\\text{2D}}(\\theta)$ [W s$^2$/rad]
        - ``p_ratio_density`` : normalized $\\bar{P}_r(\\theta)$ [$\\text{deg}^{-1}$] (Eq. 5)
        - ``eta_20_pct`` : percentage of far-field power within $|\\theta| \\le 20^\\circ$ [%]
        - ``power_rad_2d`` : integrated 2D far-field power $\\int_{-\\pi/2}^{\\pi/2} U_{\\text{2D}}\\,d\\theta$
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    theta_rad = np.deg2rad(theta_deg)
    theta_3d = np.full_like(theta_rad, 0.5 * np.pi)
    phi_3d = 0.5 * np.pi - theta_rad

    u_3d = n2f_monitor.radiation_intensity(theta_3d, phi_3d)[freq_idx]
    freqs_arr = getattr(n2f_monitor, "frequencies", getattr(n2f_monitor, "_freqs"))
    n_med = getattr(n2f_monitor, "n_medium", getattr(n2f_monitor, "_n_medium", 1.0))
    f0 = float(freqs_arr[freq_idx])
    k0 = 2.0 * np.pi * f0 / C0 * float(n_med)
    u_2d = np.asarray(u_3d, dtype=float) * (2.0 * np.pi / (k0 * float(dz_m)))

    return summarize_angular_distribution(theta_deg, u_2d)


def summarize_angular_distribution(
    theta_deg: np.ndarray, u_2d: np.ndarray, half_angle_deg: float = 20.0
) -> dict[str, np.ndarray | float]:
    """Compute Eq. (5) normalized density $\\bar{P}_r(\\theta)$ and $\\pm\\theta_0$ efficiency.

    Parameters
    ----------
    theta_deg : np.ndarray
        Uniform angle grid on $[-90^\\circ, 90^\\circ]$ [deg].
    u_2d : np.ndarray
        Unnormalized or 2D angular power distribution $P(\\theta) \\ge 0$.
    half_angle_deg : float, optional
        Collimation cone half-angle $\\theta_0$ [deg]. Default is 20.0.

    Returns
    -------
    dict
        Dictionary with ``u_2d``, ``p_ratio_density`` [$\\text{deg}^{-1}$],
        ``eta_20_pct`` [%], and ``power_rad_2d``.
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    u_2d = np.clip(np.asarray(u_2d, dtype=float), 0.0, None)
    dtheta_deg = 180.0 / len(theta_deg)
    total_sum = float(np.sum(u_2d))
    if total_sum > 0.0:
        p_ratio_density = u_2d / (total_sum * dtheta_deg)
        in_cone = np.abs(theta_deg) <= float(half_angle_deg)
        eta_pct = 100.0 * float(np.sum(u_2d[in_cone])) / total_sum
    else:
        p_ratio_density = np.zeros_like(u_2d)
        eta_pct = 0.0
    power_rad_2d = float(np.trapezoid(u_2d, np.deg2rad(theta_deg)))
    return {
        "u_2d": u_2d,
        "p_ratio_density": p_ratio_density,
        "eta_20_pct": eta_pct,
        "power_rad_2d": power_rad_2d,
    }


def mqw_dipole_grid(
    channel: str, nx_dip: int = 5, ny_dip: int = 4
) -> np.ndarray:
    """Generate uniform $(x_m, y_m)$ dipole coordinates inside a channel's MQW.

    In Chen et al. (2025), 20 dipoles per R/G/B channel represent spatially
    incoherent spontaneous emission across the $1.05\\;\\mu\\text{m} \\times 0.20\\;\\mu\\text{m}$
    MQW cross-section ($y \\in [-0.40, -0.20]\\;\\mu\\text{m}$).

    Parameters
    ----------
    channel : {"B", "G", "R"}
        Active subpixel color channel.
    nx_dip : int, optional
        Number of dipole columns along $x$. Default is 5.
    ny_dip : int, optional
        Number of dipole rows along $y$. Default is 4 ($5 \\times 4 = 20$).

    Returns
    -------
    np.ndarray
        Array of shape ``(nx_dip * ny_dip, 2)`` containing ``(x_m, y_m)`` [m].
    """
    x0 = RGB_MATERIALS[channel]["x_center_m"]
    w_ch = PIXEL_GEOMETRY["channel_width_m"]
    t_mqw = PIXEL_GEOMETRY["t_MQW_m"]
    y_bot = PIXEL_GEOMETRY["y_mqw_center_m"] - 0.5 * t_mqw

    xs = x0 - 0.5 * w_ch + (np.arange(nx_dip) + 0.5) / nx_dip * w_ch
    ys = y_bot + (np.arange(ny_dip) + 0.5) / ny_dip * t_mqw
    coords = [(float(x), float(y)) for y in ys for x in xs]
    return np.asarray(coords, dtype=float)


def build_rgb_array_mesh(
    channel: str = "G",
    cpw: float = 10.0,
    domain_x: tuple[float, float] = (-2.85e-6, 2.85e-6),
    domain_y: tuple[float, float] = (-1.38e-6, 0.58e-6),
    dz: float = 1.0e-6,
    collimator_domes: bool = False,
    min_cells_per_feature: int = 4,
    grading: float = 1.35,
) -> tuple[wakis.GridFIT3D, dict]:
    """Build 2D Wakis FIT graded mesh for the 3-pixel RGB microLED array.

    Uses ``wakis.templates.meshing.graded_axis`` with exact nodes snapped to all
    horizontal layer interfaces ($y = -1.10, -0.40, -0.20, 0.00\\;\\mu\\text{m}$) and
    vertical mesa/trench boundaries ($x_c \\pm W_{\\text{ch}}/2$). The active
    dielectric stack uses target cell size $\\Delta x_{\\text{bg}} = \\lambda_{\\min} / (\\text{cpw} \\cdot n_{\\max,\\text{MQW}})$,
    while the thin $200\\;\\text{nm}$ MQW and p-GaN layers and the high-index
    Silicon substrate ($n_{\\max,\\text{Si}} = 4.583$) use locally refined vertical
    cell size $\\Delta y_{\\min} = \\lambda_{\\min} / (\\text{cpw} \\cdot n_{\\max,\\text{Si}})$.

    Parameters
    ----------
    channel : {"B", "G", "R"}, optional
        Channel whose wavelength $\\lambda_0$ and Table 1 refractive indices are used.
    cpw : float, optional
        Cells per minimum wavelength in each material. Default is 10.0.
    domain_x : tuple of float, optional
        Simulation bounds ``(xmin, xmax)`` [m].
    domain_y : tuple of float, optional
        Simulation bounds ``(ymin, ymax)`` [m].
    dz : float, optional
        Out-of-plane cell thickness $\\Delta z$ [m] (with ``Nz = 1``). Default is 1.0e-6.
    collimator_domes : bool, optional
        If True, snap additional nodes for the tiered collimator in $y \\in [0.10, 1.10]\\;\\mu\\text{m}$.
    min_cells_per_feature : int, optional
        Minimum cells across any interval between structure faces. Default is 4.
    grading : float, optional
        Maximum adjacent cell-size grading ratio ($\\le 1.4$). Default is 1.35.

    Returns
    -------
    grid : wakis.GridFIT3D
        Constructed 2D FIT grid (``Nz = 1``).
    meta : dict
        Dictionary of mesh resolution, feature cell counts, and skin-depth metadata.
    """
    mat = RGB_MATERIALS[channel]
    wl = mat["wavelength_m"]
    f0 = C0 / wl

    dx_bg = grid_spacing_from_cpw(cpw, PIXEL_GEOMETRY["n_max_active"])
    dx_si = grid_spacing_from_cpw(cpw, PIXEL_GEOMETRY["n_max_silicon"])
    xmin, xmax = float(domain_x[0]), float(domain_x[1])
    ymin, ymax = float(domain_y[0]), float(domain_y[1])
    w_ch = PIXEL_GEOMETRY["channel_width_m"]

    x_faces = [xmin, xmax]
    for sub_ch in CHANNELS:
        x_sub = RGB_MATERIALS[sub_ch]["x_center_m"]
        for edge in (x_sub - 0.5 * w_ch, x_sub + 0.5 * w_ch):
            if xmin + 1e-9 < edge < xmax - 1e-9:
                x_faces.append(edge)
    x_points = merge_points(x_faces, tol=1e-12)
    x_sizes = [dx_bg] * (len(x_points) - 1)
    x_nodes = graded_axis(
        x_points, x_sizes, min_cells=min_cells_per_feature, grading=grading
    )

    y_faces = [ymin, -1.10e-6, -0.40e-6, -0.20e-6, 0.0, ymax]
    if collimator_domes:
        y_faces.extend(np.linspace(0.10e-6, 1.00e-6, 7).tolist())
        y_faces.append(1.10e-6)
    y_points = merge_points([p for p in y_faces if ymin <= p <= ymax], tol=1e-12)
    y_sizes = []
    for a, b in zip(y_points[:-1], y_points[1:]):
        mid = 0.5 * (a + b)
        if mid < -1.10e-6 or (-0.40e-6 <= mid <= 0.0):
            y_sizes.append(dx_si)
        else:
            y_sizes.append(dx_bg)
    y_nodes = graded_axis(
        y_points, y_sizes, min_cells=min_cells_per_feature, grading=grading
    )
    z_nodes = np.array([-0.5 * float(dz), 0.5 * float(dz)], dtype=float)

    grid = wakis.GridFIT3D(x=x_nodes, y=y_nodes, z=z_nodes, verbose=0)

    dx_phys = np.diff(x_nodes)
    dy_phys = np.diff(y_nodes)
    dx_bg_nm = float(max(np.max(dx_phys), np.max(dy_phys)) * 1e9)
    dx_min_nm = float(min(np.min(dx_phys), np.min(dy_phys)) * 1e9)

    mqw_mask = (y_nodes[:-1] >= -0.40e-6 - 1e-11) & (y_nodes[1:] <= -0.20e-6 + 1e-11)
    pgan_mask = (y_nodes[:-1] >= -0.20e-6 - 1e-11) & (y_nodes[1:] <= 0.0 + 1e-11)
    cells_across_feature = int(min(np.sum(mqw_mask), np.sum(pgan_mask)))

    trench_mask = (x_nodes[:-1] >= -0.975e-6 - 1e-11) & (
        x_nodes[1:] <= -0.525e-6 + 1e-11
    )
    cells_across_trench = int(np.sum(trench_mask))

    si_mask = y_nodes[1:] <= -1.10e-6 + 1e-11
    dy_si_m = float(np.max(dy_phys[si_mask])) if np.any(si_mask) else float(np.min(dy_phys))
    dy_mqw_m = float(np.max(dy_phys[mqw_mask])) if np.any(mqw_mask) else float(np.min(dy_phys))

    n_si = mat["n_Si_real"]
    k_si = mat["n_Si_imag"]
    skin = silicon_skin_depth(wl, k_si, dy_si_m)

    # GridFIT3D adds 1 edge ghost cell along non-periodic axes (x, y) and 1 along z
    # which SolverFIT3D removes along periodic z (so solver.Nx == grid.Nx, solver.Ny == grid.Ny)
    nx_solver = int(grid.Nx)
    ny_solver = int(grid.Ny)
    meta = {
        "channel": channel,
        "wavelength_m": wl,
        "f0_hz": f0,
        "cpw": float(cpw),
        "dx_m": float(np.mean(dx_phys)),
        "dy_m": float(np.mean(dy_phys)),
        "dz_m": float(dz),
        "dx_background_nm": dx_bg_nm,
        "dx_min_nm": dx_min_nm,
        "min_feature_nm": 200.0,
        "cells_across_feature": cells_across_feature,
        "cells_across_trench": cells_across_trench,
        "nx": nx_solver,
        "ny": ny_solver,
        "n_cells": int(nx_solver * ny_solver),
        "cpw_mqw": wl / (mat["n_MQW"] * dy_mqw_m),
        "cpw_mqw_x": wl / (mat["n_MQW"] * float(np.max(dx_phys))),
        "cpw_si": wl / (n_si * dy_si_m),
        **skin,
    }
    return grid, meta


def validate_mesh_resolution(meta: dict, min_cpw: float = 10.0) -> None:
    """Validate that a mesh metadata dictionary satisfies project resolution rules 1-3.

    Raises ``AssertionError`` if:
    1. ``cpw < min_cpw`` or ``dx_background_nm > lambda_min / (min_cpw * n_max_active)``
       or ``dx_min_nm > lambda_min / (min_cpw * n_max_silicon)``.
    2. ``cells_across_feature < 4`` (for the $200\\;\\text{nm}$ MQW / p-GaN layers) or
       ``cells_across_trench < 4`` (when the domain includes the $450\\;\\text{nm}$ trench).
    3. ``cells_per_skin_depth < 4.0`` in the lossy Silicon substrate.
    """
    assert float(meta["cpw"]) >= float(min_cpw) - 1e-9, (
        f"cpw={meta['cpw']} violates minimum {min_cpw} cpw"
    )
    max_bg_nm = grid_spacing_from_cpw(min_cpw, PIXEL_GEOMETRY["n_max_active"]) * 1e9
    max_si_nm = grid_spacing_from_cpw(min_cpw, PIXEL_GEOMETRY["n_max_silicon"]) * 1e9
    assert float(meta["dx_background_nm"]) <= max_bg_nm * 1.005, (
        f"dx_background_nm={meta['dx_background_nm']:.2f} exceeds {max_bg_nm:.2f} nm"
    )
    assert float(meta["dx_min_nm"]) <= max_si_nm * 1.005, (
        f"dx_min_nm={meta['dx_min_nm']:.2f} exceeds {max_si_nm:.2f} nm"
    )
    assert int(meta["cells_across_feature"]) >= 4, (
        f"cells_across_feature={meta['cells_across_feature']} < 4"
    )
    if int(meta.get("cells_across_trench", 4)) > 0:
        assert int(meta["cells_across_trench"]) >= 4, (
            f"cells_across_trench={meta['cells_across_trench']} < 4"
        )
    assert float(meta["cells_per_skin_depth"]) >= 4.0, (
        f"cells_per_skin_depth={meta['cells_per_skin_depth']:.2f} < 4.0"
    )


def build_rgb_array_solver(
    channel: str,
    cpw: float = 10.0,
    domain_x: tuple[float, float] = (-2.85e-6, 2.85e-6),
    domain_y: tuple[float, float] = (-1.38e-6, 0.58e-6),
    dz: float = 1.0e-6,
    cfln: float = 0.85,
    n_pml: int = 8,
    bc_x: str = "cpml",
    collimator_domes: bool = False,
) -> tuple[wakis.GridFIT3D, wakis.SolverFIT3D, dict]:
    """Build 2D Wakis FIT grid and solver for the 3-pixel RGB microLED array.

    Parameters
    ----------
    channel : {"B", "G", "R"}
        Channel whose wavelength $\\lambda_0$ and Table 1 refractive indices are used.
    cpw : float, optional
        Cells per minimum wavelength in each material. Default is 10.0.
    domain_x : tuple of float, optional
        Simulation bounds ``(xmin, xmax)`` [m].
    domain_y : tuple of float, optional
        Simulation bounds ``(ymin, ymax)`` [m].
    dz : float, optional
        Out-of-plane cell thickness $\\Delta z$ [m] (with ``Nz = 1``). Default is 1.0e-6.
    cfln : float, optional
        Courant factor (0.85 in 2D gives $0.85/\\sqrt{2} = 0.60 < 1/\\sqrt{2}$).
    n_pml : int, optional
        Number of CPML layers on $x$/$y$ boundaries. Default is 8.
    bc_x : {"cpml", "periodic"}, optional
        Boundary condition along $x$. Default is ``"cpml"``.
    collimator_domes : bool, optional
        If True, add a reference $\\text{SiN}_x$/$\\text{SiO}_2$ tiered lens structure
        above the pixel array ($y \\in [0, 1.15]\\;\\mu\\text{m}$). Default is False.

    Returns
    -------
    grid : wakis.GridFIT3D
        Constructed 2D FIT grid (``Nz = 1``).
    solver : wakis.SolverFIT3D
        Constructed and material-assigned 2D FIT solver.
    meta : dict
        Dictionary of resolution, wavelength, frequency, and skin-depth metadata.
    """
    grid, meta = build_rgb_array_mesh(
        channel=channel,
        cpw=cpw,
        domain_x=domain_x,
        domain_y=domain_y,
        dz=dz,
        collimator_domes=collimator_domes,
    )
    mat = RGB_MATERIALS[channel]
    f0 = meta["f0_hz"]
    omega0 = 2.0 * np.pi * f0
    xmin, xmax = float(domain_x[0]), float(domain_x[1])
    ymin = float(domain_y[0])

    # Complex refractive index of Silicon mapped to exact single-frequency Drude/conductive
    # permittivity: eps(omega0) = (n + i k)^2 = (n^2 - k^2) + i (2 n k)
    n_si = mat["n_Si_real"]
    k_si = mat["n_Si_imag"]
    eps_si_real = n_si**2 - k_si**2
    sigma_si = 2.0 * omega0 * EPS_0 * n_si * k_si
    si_dispersive = DispersiveMaterial(
        eps_inf=eps_si_real, sigma=sigma_si, poles=[], name=f"Si_{channel}"
    )

    w_span = (xmax - xmin) + 4.0e-6
    y_si_bot = ymin - 0.5e-6
    h_si = PIXEL_GEOMETRY["y_si_top_m"] - y_si_bot
    y_si_center = 0.5 * (PIXEL_GEOMETRY["y_si_top_m"] + y_si_bot)

    structs = [
        # Absorbing Silicon substrate (y <= -1.10 um)
        Box(
            center=(0.0, y_si_center, 0.0),
            size=(w_span, h_si, 10.0 * dz),
            material=si_dispersive,
        ),
        # Polyimide dielectric filler (y in [-1.10, 0.00] um)
        Box(
            center=(0.0, -0.55e-6, 0.0),
            size=(w_span, 1.10e-6, 10.0 * dz),
            material=mat["n_PI"] ** 2,
        ),
    ]

    # Three subpixel mesas: B (-1.5 um), G (0.0 um), R (+1.5 um)
    w_ch = PIXEL_GEOMETRY["channel_width_m"]
    for sub_ch in CHANNELS:
        x_sub = RGB_MATERIALS[sub_ch]["x_center_m"]
        # Bottom n-GaN (700 nm: y in [-1.10, -0.40] um)
        structs.append(
            Box(
                center=(x_sub, -0.75e-6, 0.0),
                size=(w_ch, 0.70e-6, 10.0 * dz),
                material=mat["n_GaN"] ** 2,
            )
        )
        # Middle active MQW (200 nm: y in [-0.40, -0.20] um)
        structs.append(
            Box(
                center=(x_sub, -0.30e-6, 0.0),
                size=(w_ch, 0.20e-6, 10.0 * dz),
                material=mat["n_MQW"] ** 2,
            )
        )
        # Top p-GaN (200 nm: y in [-0.20, 0.00] um)
        structs.append(
            Box(
                center=(x_sub, -0.10e-6, 0.0),
                size=(w_ch, 0.20e-6, 10.0 * dz),
                material=mat["n_GaN"] ** 2,
            )
        )

    if collimator_domes:
        # Reference SiO2/SiNx binary structure in y in [0, 1.10] um
        structs.append(
            Box(
                center=(0.0, 0.55e-6, 0.0),
                size=(w_span, 1.10e-6, 10.0 * dz),
                material=mat["n_SiO2"] ** 2,
            )
        )
        for sub_ch in CHANNELS:
            x_sub = RGB_MATERIALS[sub_ch]["x_center_m"]
            r_lens = 1.10e-6
            n_tiers = 6
            y_b, y_t = 0.10e-6, 1.00e-6
            dy_t = (y_t - y_b) / n_tiers
            for it in range(n_tiers):
                yc = y_b + (it + 0.5) * dy_t
                h_rel = yc - y_b
                w_t = 2.0 * np.sqrt(max(0.0, r_lens**2 - (h_rel + 0.35e-6) ** 2))
                if w_t > 0.1e-6:
                    structs.append(
                        Box(
                            center=(x_sub, yc, 0.0),
                            size=(min(w_t, 1.40e-6), dy_t, 10.0 * dz),
                            material=mat["n_SiNx"] ** 2,
                        )
                    )

    bc_low = [bc_x, "cpml", "periodic"]
    bc_high = [bc_x, "cpml", "periodic"]
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=bc_low,
        bc_high=bc_high,
        n_pml=n_pml,
        cfln=cfln,
        verbose=0,
    )
    assign_materials(solver, structs, f0, averaging="kottke")
    # Enforce 2D in-plane polarization (Ex, Ey, Hz; called TM in Chen et al. 2025)
    solver.ieps[:, :, :, "z"] = 0.0

    meta["nx"] = int(solver.Nx)
    meta["ny"] = int(solver.Ny)
    meta["n_cells"] = int(solver.Nx * solver.Ny)
    meta["dt_s"] = float(solver.dt)
    return grid, solver, meta


def _extract_inplane_phasors(
    field_mon: DFTMonitor, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return complex 2D arrays ``(Ex, Ey)`` of shape ``(Nx, Ny)`` at frequency 0."""
    ex = np.zeros((nx, ny), dtype=np.complex128)
    ey = np.zeros((nx, ny), dtype=np.complex128)
    ex_arr, _ = field_mon.get_field("Ex")
    sx, sy = min(nx, ex_arr.shape[1]), min(ny, ex_arr.shape[2])
    ex[:sx, :sy] = ex_arr[0, :sx, :sy, 0]
    if "Ey" in field_mon.components:
        ey_arr, _ = field_mon.get_field("Ey")
        sx_y, sy_y = min(nx, ey_arr.shape[1]), min(ny, ey_arr.shape[2])
        ey[:sx_y, :sy_y] = ey_arr[0, :sx_y, :sy_y, 0]
    return ex, ey


def run_single_channel_method_a(
    channel: str,
    cpw: float = 10.0,
    nx_dip: int = 5,
    ny_dip: int = 4,
    theta_deg: np.ndarray | None = None,
    record_fields: bool = True,
    tail_time_s: float = 18e-15,
) -> dict:
    """Run Method A ($N_{\\text{dip}}$ independent dipole simulations summed in power).

    Reproduces the incoherent multi-dipole evaluation of Chen et al. (2025),
    Section 2.4 & 3.2, and records per-dipole complex far-field and near-field
    phasors so that:
    1. Method A incoherent average ($I_{\\text{inc}} = \\sum_m |E_m|^2$) and
       wrong coherent field sum ($I_{\\text{coh}} = |\\sum_m E_m|^2$) can both be
       evaluated directly.
    2. Forward inter-pixel optical crosstalk matrix $C_{ij}^{\\text{emit}}$ is
       measured across all three subpixel exit apertures $W_{\\text{B}}, W_{\\text{G}}, W_{\\text{R}}$
       and MQW field monitors $M_{\\text{B}}, M_{\\text{G}}, M_{\\text{R}}$.

    Parameters
    ----------
    channel : {"B", "G", "R"}
        Active emitter channel to excite.
    cpw : float, optional
        Cells per wavelength in active MQW. Default is 10.0.
    nx_dip, ny_dip : int, optional
        Dipole grid dimensions inside the active MQW ($5 \\times 4 = 20$).
    theta_deg : np.ndarray or None, optional
        Far-field angle grid [deg]. Defaults to ``np.linspace(-90, 90, 181)``.
    record_fields : bool, optional
        Whether to record 2D complex field phasors ``Ex(x, y), Ey(x, y)`` over the
        cross-section. Default is True.
    tail_time_s : float, optional
        Post-pulse ringdown time [s]. Default is ``18e-15``.

    Returns
    -------
    dict
        Comprehensive dictionary with Method A incoherent results, coherent wrong-sum
        comparison, per-dipole convergence history, and inter-pixel crosstalk rows.
    """
    if theta_deg is None:
        theta_deg = np.linspace(-90.0, 90.0, 181)
    theta_deg = np.asarray(theta_deg, dtype=float)
    theta_rad = np.deg2rad(theta_deg)
    theta_3d = np.full_like(theta_rad, 0.5 * np.pi)
    phi_3d = 0.5 * np.pi - theta_rad

    t_start = time.perf_counter()
    grid, solver, meta = build_rgb_array_solver(channel=channel, cpw=cpw)
    f0 = meta["f0_hz"]
    dz = meta["dz_m"]
    k0 = 2.0 * np.pi * f0 / C0

    dip_coords = mqw_dipole_grid(channel, nx_dip=nx_dip, ny_dip=ny_dip)
    n_dip = len(dip_coords)

    src_time = GaussianSource(frequency=f0, fwidth=0.35 * f0, cutoff=4.0)
    nt = int((src_time.end_time + float(tail_time_s)) / solver.dt)

    x0_active = RGB_MATERIALS[channel]["x_center_m"]
    w_ch = PIXEL_GEOMETRY["channel_width_m"]
    p_sub = PIXEL_GEOMETRY["subpixel_pitch_m"]
    y_n2f = 0.25e-6
    w_n2f = 5.20e-6

    u_2d_per_dip = np.zeros((n_dip, len(theta_deg)), dtype=float)
    e_ff_per_dip = np.zeros((n_dip, len(theta_deg)), dtype=np.complex128)
    flux_mqw_per_dip = np.zeros(n_dip, dtype=float)
    flux_top_per_dip = np.zeros(n_dip, dtype=float)
    flux_win_per_dip = np.zeros((n_dip, 3), dtype=float)
    mqw_int_per_dip = np.zeros((n_dip, 3), dtype=float)

    nx, ny = meta["nx"], meta["ny"]
    intensity_inc_2d = np.zeros((nx, ny), dtype=float)
    ex_coh_sum = np.zeros((nx, ny), dtype=np.complex128)
    ey_coh_sum = np.zeros((nx, ny), dtype=np.complex128)

    # Indices of the 3 MQW line monitors M_B, M_G, M_R at y = -0.30 um
    x_cells = grid.x[:-1] + 0.5 * grid.dx
    y_cells = grid.y[:-1] + 0.5 * grid.dy
    j_mqw = int(np.argmin(np.abs(y_cells - PIXEL_GEOMETRY["y_mqw_center_m"])))
    mqw_x_masks = [
        np.abs(x_cells - RGB_MATERIALS[ch]["x_center_m"]) <= 0.5 * w_ch
        for ch in CHANNELS
    ]

    for m_idx, (xd, yd) in enumerate(dip_coords):
        reset_solver_state(solver)
        dip = CurrentSource(
            src_time,
            component="x",
            center=(float(xd), float(yd), 0.0),
            size=(0.0, 0.0, dz),
        )
        mqw_box = make_flux_box_2d(
            [f0],
            center=(x0_active, PIXEL_GEOMETRY["y_mqw_center_m"], 0.0),
            width_x=w_ch,
            height_y=0.22e-6,
        )
        top_n2f = wakis.Near2FarMonitor(
            [f0], center=(0.0, y_n2f, 0.0), size=(w_n2f, 0.0, np.inf)
        )
        win_monitors = [
            wakis.FluxMonitor(
                [f0],
                center=(RGB_MATERIALS[ch]["x_center_m"], y_n2f, 0.0),
                size=(p_sub, 0.0, np.inf),
            )
            for ch in CHANNELS
        ]
        monitors = [mqw_box, top_n2f, *win_monitors]
        f_mon = None
        if record_fields:
            f_mon = DFTMonitor(
            [f0],
            components=("Ex", "Ey"),
            center=(0.0, 0.0, 0.0),
            size=(np.inf, np.inf, np.inf),
        )
            monitors.append(f_mon)

        solver.emsolve(nt, source=dip, monitors=monitors)

        flux_mqw_per_dip[m_idx] = float(mqw_box.get_flux()[0])
        flux_top_per_dip[m_idx] = float(top_n2f.get_flux()[0])
        for c_idx, wm in enumerate(win_monitors):
            flux_win_per_dip[m_idx, c_idx] = max(0.0, float(wm.get_flux()[0]))

        # Extract complex far-field pattern E_phi(theta) and 2D radiation intensity
        ff = top_n2f.get_farfield(theta_3d, phi_3d)
        e_ff_per_dip[m_idx] = ff["Ephi"][0]
        u_3d = top_n2f.radiation_intensity(theta_3d, phi_3d)[0]
        u_2d_per_dip[m_idx] = np.asarray(u_3d, dtype=float) * (
            2.0 * np.pi / (k0 * dz)
        )

        if f_mon is not None:
            ex_2d, ey_2d = _extract_inplane_phasors(f_mon, meta["nx"], meta["ny"])
            intensity_inc_2d += np.abs(ex_2d) ** 2 + np.abs(ey_2d) ** 2
            ex_coh_sum += ex_2d
            ey_coh_sum += ey_2d
            for c_idx, mask in enumerate(mqw_x_masks):
                mqw_int_per_dip[m_idx, c_idx] = float(
                    np.sum(np.abs(ex_2d[mask, j_mqw]) ** 2 * grid.dx[mask])
                )

    runtime_s = time.perf_counter() - t_start
    u_2d_inc = np.sum(u_2d_per_dip, axis=0)
    ff_summary = summarize_angular_distribution(theta_deg, u_2d_inc)

    # Wrong coherent sum in far field: |sum_m E_phi,m|^2
    scale_ff = float(np.sum(u_2d_inc)) / max(
        1e-30, float(np.sum(np.sum( np.abs(e_ff_per_dip) ** 2, axis=0 )))
    )
    u_2d_coh = np.abs(np.sum(e_ff_per_dip, axis=0)) ** 2 * scale_ff
    ff_coh_summary = summarize_angular_distribution(theta_deg, u_2d_coh)

    # Cumulative convergence vs number of dipoles N = 1 ... n_dip
    eta_20_cum = np.zeros(n_dip, dtype=float)
    lee_tot_cum = np.zeros(n_dip, dtype=float)
    for n_cur in range(1, n_dip + 1):
        u_partial = np.sum(u_2d_per_dip[:n_cur], axis=0)
        eta_20_cum[n_cur - 1] = summarize_angular_distribution(
            theta_deg, u_partial
        )["eta_20_pct"]
        lee_tot_cum[n_cur - 1] = (
            100.0
            * float(np.sum(flux_top_per_dip[:n_cur]))
            / max(1e-30, float(np.sum(flux_mqw_per_dip[:n_cur])))
        )

    lee_tot_pct = float(lee_tot_cum[-1])
    win_flux_sum = np.sum(flux_win_per_dip, axis=0)
    win_crosstalk_row = win_flux_sum / max(1e-30, float(np.sum(win_flux_sum)))
    mqw_int_sum = np.sum(mqw_int_per_dip, axis=0)
    mqw_crosstalk_row = mqw_int_sum / max(1e-30, float(np.sum(mqw_int_sum)))

    intensity_coh_2d = np.abs(ex_coh_sum) ** 2 + np.abs(ey_coh_sum) ** 2

    return {
        "channel": channel,
        "meta": meta,
        "nt": nt,
        "n_steps": int(nt),
        "runtime_s": runtime_s,
        "dipole_coords_m": dip_coords,
        "x_cells_m": x_cells,
        "y_cells_m": y_cells,
        "theta_deg": theta_deg,
        "u_2d_per_dip": u_2d_per_dip,
        "e_ff_per_dip": e_ff_per_dip,
        "u_2d_inc": ff_summary["u_2d"],
        "p_ratio_density_inc": ff_summary["p_ratio_density"],
        "eta_20_inc_pct": ff_summary["eta_20_pct"],
        "u_2d_coh": ff_coh_summary["u_2d"],
        "p_ratio_density_coh": ff_coh_summary["p_ratio_density"],
        "eta_20_coh_pct": ff_coh_summary["eta_20_pct"],
        "lee_tot_pct": lee_tot_pct,
        "flux_mqw_per_dip": flux_mqw_per_dip,
        "flux_top_per_dip": flux_top_per_dip,
        "eta_20_cumulative_pct": eta_20_cum,
        "lee_tot_cumulative_pct": lee_tot_cum,
        "win_flux_sum": win_flux_sum,
        "win_crosstalk_row": win_crosstalk_row,
        "mqw_int_sum": mqw_int_sum,
        "mqw_crosstalk_row": mqw_crosstalk_row,
        "intensity_inc_2d": intensity_inc_2d,
        "intensity_coh_2d": intensity_coh_2d,
    }


def run_method_b_fdtd_realizations(
    channel: str = "G",
    cpw: float = 10.0,
    n_realizations: int = 8,
    nx_dip: int = 5,
    ny_dip: int = 4,
    theta_deg: np.ndarray | None = None,
    seed: int = 2025,
    tail_time_s: float = 18e-15,
) -> dict:
    """Run Method B in Wakis FDTD: $K$ realizations of $N$ simultaneous random-phase dipoles.

    In each realization $k \\in \\{0, \\dots, K-1\\}$, all $N_{\\text{dip}}$ dipoles inside
    the active MQW are excited simultaneously in a single FDTD run using
    ``RandomPhaseTimeProfile`` with independent phases $\\phi_m^{(k)} \\sim \\mathcal{U}(0, 2\\pi)$.

    Parameters
    ----------
    channel : {"B", "G", "R"}, optional
        Active channel. Default is ``"G"``.
    cpw : float, optional
        Cells per wavelength in active MQW. Default is 10.0.
    n_realizations : int, optional
        Number of independent FDTD random-phase runs $K$. Default is 8.
    nx_dip, ny_dip : int, optional
        Dipole grid dimensions ($5 \\times 4 = 20$).
    theta_deg : np.ndarray or None, optional
        Far-field angle grid [deg]. Defaults to ``np.linspace(-90, 90, 181)``.
    seed : int, optional
        Random number generator seed for reproducibility. Default is 2025.
    tail_time_s : float, optional
        Post-pulse ringdown time [s]. Default is ``18e-15``.

    Returns
    -------
    dict
        Dictionary containing the $K$ FDTD angular intensity realizations
        ``u_2d_realizations``, phases ``phases_rad``, runtime, and cumulative
        Method B estimates vs $K$.
    """
    if theta_deg is None:
        theta_deg = np.linspace(-90.0, 90.0, 181)
    theta_deg = np.asarray(theta_deg, dtype=float)
    theta_rad = np.deg2rad(theta_deg)
    theta_3d = np.full_like(theta_rad, 0.5 * np.pi)
    phi_3d = 0.5 * np.pi - theta_rad

    rng = np.random.default_rng(seed)
    dip_coords = mqw_dipole_grid(channel, nx_dip=nx_dip, ny_dip=ny_dip)
    n_dip = len(dip_coords)
    phases_rad = rng.uniform(0.0, 2.0 * np.pi, size=(n_realizations, n_dip))

    t_start = time.perf_counter()
    grid, solver, meta = build_rgb_array_solver(channel=channel, cpw=cpw)
    f0 = meta["f0_hz"]
    dz = meta["dz_m"]
    k0 = 2.0 * np.pi * f0 / C0

    ref_profile = RandomPhaseTimeProfile(frequency=f0, fwidth=0.35 * f0, phase_rad=0.0)
    nt = int((ref_profile.end_time + float(tail_time_s)) / solver.dt)

    u_2d_realizations = np.zeros((n_realizations, len(theta_deg)), dtype=float)
    for k_idx in range(n_realizations):
        reset_solver_state(solver)
        sources = []
        for m_idx, (xd, yd) in enumerate(dip_coords):
            prof = RandomPhaseTimeProfile(
                frequency=f0,
                fwidth=0.35 * f0,
                phase_rad=float(phases_rad[k_idx, m_idx]),
            )
            sources.append(
                CurrentSource(
                        prof,
                    component="x",
                    center=(float(xd), float(yd), 0.0),
                    size=(0.0, 0.0, dz),
                )
            )
        multi_src = MultiSource(sources)
        top_n2f = wakis.Near2FarMonitor(
            [f0], center=(0.0, 0.25e-6, 0.0), size=(5.20e-6, 0.0, np.inf)
        )
        solver.emsolve(nt, source=multi_src, monitors=[top_n2f])
        u_3d = top_n2f.radiation_intensity(theta_3d, phi_3d)[0]
        u_2d_realizations[k_idx] = np.asarray(u_3d, dtype=float) * (
            2.0 * np.pi / (k0 * dz)
        )

    runtime_s = time.perf_counter() - t_start
    return {
        "channel": channel,
        "meta": meta,
        "nt": int(nt),
        "n_steps": int(nt),
        "n_realizations": n_realizations,
        "n_dipoles": n_dip,
        "phases_rad": phases_rad,
        "theta_deg": theta_deg,
        "u_2d_realizations": u_2d_realizations,
        "runtime_s": runtime_s,
    }


def evaluate_method_b_statistics(
    e_ff_per_dip: np.ndarray,
    u_2d_inc: np.ndarray,
    theta_deg: np.ndarray,
    k_values: list[int] | np.ndarray = (1, 2, 4, 8, 16, 32, 64),
    n_trials: int = 400,
    seed: int = 42,
) -> dict:
    """Evaluate Monte Carlo bias and variance of Method B vs number of realizations $K$.

    Given the exact single-dipole complex far-field amplitudes $E_m(\\theta)$ from
    Method A, each Method B realization with random phases $\\phi_m^{(k)} \\sim \\mathcal{U}(0, 2\\pi)$
    produces angular intensity $I^{(k)}(\\theta) \\propto |\\sum_{m=1}^N E_m(\\theta) e^{i\\phi_m^{(k)}}|^2$.
    Averaging over $n_{\\text{trials}}$ independent Monte Carlo ensembles for each $K$
    quantifies:
    1. Pointwise angular relative RMSE vs exact analytical $\\sqrt{(1 - 1/N)/K}$.
    2. Cone-efficiency $\\eta_{\\pm 20^\\circ}^{(K)}$ standard deviation and bias vs $K$.

    Parameters
    ----------
    e_ff_per_dip : np.ndarray
        Complex far-field amplitudes of shape ``(N_dip, N_theta)``.
    u_2d_inc : np.ndarray
        Exact Method A incoherent angular intensity of shape ``(N_theta,)``.
    theta_deg : np.ndarray
        Angle grid [deg] of shape ``(N_theta,)``.
    k_values : sequence of int, optional
        Realization counts $K$ to evaluate.
    n_trials : int, optional
        Number of independent Monte Carlo trials per $K$. Default is 400.
    seed : int, optional
        Random seed. Default is 42.

    Returns
    -------
    dict
        Dictionary of empirical and theoretical RMSE, bias, and $\\eta_{\\pm 20^\\circ}$
        statistics as functions of $K$.
    """
    rng = np.random.default_rng(seed)
    e_ff = np.asarray(e_ff_per_dip, dtype=np.complex128)
    n_dip, n_theta = e_ff.shape
    u_exact = np.sum(np.abs(e_ff) ** 2, axis=0)
    eta_exact = summarize_angular_distribution(theta_deg, u_exact)["eta_20_pct"]
    in_20 = np.abs(theta_deg) <= 20.0

    k_arr = np.asarray(k_values, dtype=int)
    rse_empirical = np.zeros(len(k_arr), dtype=float)
    rel_bias_empirical = np.zeros(len(k_arr), dtype=float)
    eta_mean_vs_k = np.zeros(len(k_arr), dtype=float)
    eta_std_vs_k = np.zeros(len(k_arr), dtype=float)
    eta_single_ratio_bias_vs_k = np.zeros(len(k_arr), dtype=float)

    for idx, k_val in enumerate(k_arr):
        # Generate (n_trials, k_val, n_dip) random phases
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_trials, int(k_val), n_dip))
        weights = np.exp(1j * phases)  # (n_trials, k_val, n_dip)
        e_realizations = weights @ e_ff  # (n_trials, k_val, n_theta)
        i_realizations = np.abs(e_realizations) ** 2

        # Ensemble-averaged intensity across K realizations: (n_trials, n_theta)
        i_mean_k = np.mean(i_realizations, axis=1)

        # Pointwise relative RMSE and relative bias averaged over active angles
        mask_ang = u_exact > 0.1 * np.max(u_exact)
        rel_var = np.mean((i_mean_k[:, mask_ang] - u_exact[mask_ang]) ** 2, axis=0) / (
            u_exact[mask_ang] ** 2
        )
        rse_empirical[idx] = float(np.sqrt(np.mean(rel_var)))
        rel_bias = (np.mean(i_mean_k[:, mask_ang], axis=0) - u_exact[mask_ang]) / (
            u_exact[mask_ang]
        )
        rel_bias_empirical[idx] = float(np.mean( np.abs(rel_bias) ))

        # Proper ratio of K-averaged powers: sum_{|theta|<=20} I_mean_K / sum_theta I_mean_K
        num_k = np.sum(i_mean_k[:, in_20], axis=1)
        den_k = np.sum(i_mean_k, axis=1)
        eta_trials = 100.0 * num_k / np.maximum(1e-30, den_k)
        eta_mean_vs_k[idx] = float(np.mean(eta_trials))
        eta_std_vs_k[idx] = float(np.std(eta_trials, ddof=1))

        # Naive average of single-realization ratios (demonstrating Jensen ratio bias)
        eta_each = (
            100.0
            * np.sum(i_realizations[:, :, in_20], axis=2)
            / np.maximum(1e-30, np.sum(i_realizations, axis=2))
        )
        eta_single_ratio_bias_vs_k[idx] = float(np.mean(eta_each) - eta_exact)

    rse_theory = random_phase_rse_theory(k_arr, n_dipoles=n_dip)
    return {
        "k_values": k_arr,
        "n_dipoles": n_dip,
        "n_trials": n_trials,
        "eta_exact_pct": float(eta_exact),
        "rse_empirical": rse_empirical,
        "rse_theory": rse_theory,
        "rel_bias_empirical": rel_bias_empirical,
        "eta_mean_vs_k_pct": eta_mean_vs_k,
        "eta_std_vs_k_pct": eta_std_vs_k,
        "eta_bias_vs_k_pct": eta_mean_vs_k - float(eta_exact),
        "eta_naive_ratio_bias_pct": eta_single_ratio_bias_vs_k,
    }


def run_dual_task_color_routing_baseline(
    cpw: float = 10.0,
    angles_deg: tuple[float, ...] = (-15.0, -7.5, 0.0, 7.5, 15.0),
    tail_time_s: float = 18e-15,
) -> dict:
    """Evaluate Equation (4) dual-task plane-wave routing matrix $C_{ij}$ on baseline array.

    In Section 2.3 & 3.1 of Chen et al. (2025), plane waves at wavelengths
    $\\lambda_j \\in \\{\\lambda_{\\text{B}}, \\lambda_{\\text{G}}, \\lambda_{\\text{R}}\\}$
    impinge from above at angles $\\theta \\in \\{-15^\\circ, -7.5^\\circ, 0^\\circ, 7.5^\\circ, 15^\\circ\\}$,
    and three planar field monitors $M_i$ ($i \\in \\{\\text{B}, \\text{G}, \\text{R}\\}$)
    at the MQW center ($y = -0.30\\;\\mu\\text{m}$) record $c_{i,j} = \\int_{M_i} |E_x^j|^2\\,dx$.

    Parameters
    ----------
    cpw : float, optional
        Cells per minimum wavelength in MQW. Default is 10.0.
    angles_deg : tuple of float, optional
        Incident plane-wave angles [deg].
    tail_time_s : float, optional
        Post-pulse ringdown time [s]. Default is ``18e-15``.

    Returns
    -------
    dict
        Dictionary containing the raw and column-normalized $3 \\times 3$ matrix
        $C_{ij}(\\theta)$ for each angle and angle-averaged routing efficiency.
    """
    n_ang = len(angles_deg)
    c_raw = np.zeros((n_ang, 3, 3), dtype=float)
    c_norm = np.zeros((n_ang, 3, 3), dtype=float)

    w_ch = PIXEL_GEOMETRY["channel_width_m"]
    y_src = 0.35e-6
    meta_b = None
    nt_b = 0

    for j_idx, src_ch in enumerate(CHANNELS):
        grid, solver, meta = build_rgb_array_solver(channel=src_ch, cpw=cpw)
        f0 = meta["f0_hz"]
        wl = meta["wavelength_m"]
        k0 = 2.0 * np.pi / wl

        x_cells = grid.x[:-1] + 0.5 * grid.dx
        y_cells = grid.y[:-1] + 0.5 * grid.dy
        j_mqw = int(np.argmin(np.abs(y_cells - PIXEL_GEOMETRY["y_mqw_center_m"])))
        j_src = int(np.argmin(np.abs(y_cells - y_src)))
        mqw_masks = [
            np.abs(x_cells - RGB_MATERIALS[ch]["x_center_m"]) <= 0.5 * w_ch
            for ch in CHANNELS
        ]

        # Oblique line source across the top aperture x in [-2.25, +2.25] um
        src_mask_ix = np.where(np.abs(x_cells) <= 2.25e-6)[0]
        src_time = GaussianSource(frequency=f0, fwidth=0.35 * f0, cutoff=4.0)
        nt = int((src_time.end_time + float(tail_time_s)) / solver.dt)
        if src_ch == "B":
            meta_b = meta
            nt_b = int(nt)

        for a_idx, ang_deg in enumerate(angles_deg):
            reset_solver_state(solver)
            kx = k0 * np.sin(np.deg2rad(float(ang_deg)))
            sub_sources = []
            for ix in src_mask_ix:
                x_pos = float(x_cells[ix])
                prof = RandomPhaseTimeProfile(
                    frequency=f0,
                    fwidth=0.35 * f0,
                    phase_rad=float(kx * x_pos),
                )
                sub_sources.append(
                    CurrentSource(
                        prof,
                        component="x",
                        center=(x_pos, float(y_cells[j_src]), 0.0),
                        size=(0.0, 0.0, meta["dz_m"]),
                    )
                )
            f_mon = DFTMonitor(
            [f0],
            components=("Ex", "Ey"),
            center=(0.0, 0.0, 0.0),
            size=(np.inf, np.inf, np.inf),
        )
            solver.emsolve(nt, source=MultiSource(sub_sources), monitors=[f_mon])
            ex_2d, _ = _extract_inplane_phasors(f_mon, meta["nx"], meta["ny"])
            for i_idx, mask in enumerate(mqw_masks):
                c_raw[a_idx, i_idx, j_idx] = float(
                    np.sum(np.abs(ex_2d[mask, j_mqw]) ** 2 * grid.dx[mask])
                )
            col_sum = float(np.sum(c_raw[a_idx, :, j_idx]))
            if col_sum > 0.0:
                c_norm[a_idx, :, j_idx] = c_raw[a_idx, :, j_idx] / col_sum

    c_norm_mean = np.mean(c_norm, axis=0)
    return {
        "meta": meta_b,
        "nt": int(nt_b),
        "n_steps": int(nt_b),
        "angles_deg": np.asarray(angles_deg, dtype=float),
        "c_raw_per_angle": c_raw,
        "c_norm_per_angle": c_norm,
        "c_norm_mean": c_norm_mean,
        "oe_uncol_pct": 100.0 * np.diag(c_norm_mean),
        "ox_uncol_pct": 100.0 * (1.0 - np.diag(c_norm_mean)),
    }


def record_incoherent_Vs_coherent_movie_frames(
    channel: str = "G",
    cpw: float = 10.0,
    n_frames: int = 28,
    seed: int = 7,
) -> dict:
    """Record time-domain snapshots of $E_x(x, y, t)$ for a 20-dipole random-phase burst.

    Parameters
    ----------
    channel : {"B", "G", "R"}, optional
        Active channel to animate. Default is ``"G"``.
    cpw : float, optional
        Cells per minimum wavelength in MQW. Default is 10.0.
    n_frames : int, optional
        Number of animation frames to capture. Default is 28.
    seed : int, optional
        Random seed for dipole phases. Default is 7.

    Returns
    -------
    dict
        Dictionary with ``frames_ex`` of shape ``(n_frames, Nx, Ny)``,
        ``times_fs``, ``x_um``, and ``y_um``.
    """
    rng = np.random.default_rng(seed)
    grid, solver, meta = build_rgb_array_solver(channel=channel, cpw=cpw)
    f0 = meta["f0_hz"]
    dz = meta["dz_m"]
    dip_coords = mqw_dipole_grid(channel, nx_dip=5, ny_dip=4)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(dip_coords))

    sources = [
        CurrentSource(
            RandomPhaseTimeProfile(
                frequency=f0, fwidth=0.35 * f0, phase_rad=float(ph)
            ),
            component="x",
            center=(float(xd), float(yd), 0.0),
            size=(0.0, 0.0, dz),
        )
        for (xd, yd), ph in zip(dip_coords, phases)
    ]
    multi_src = MultiSource(sources)
    ref_prof = RandomPhaseTimeProfile(frequency=f0, fwidth=0.35 * f0)
    nt = int((ref_prof.end_time + 14e-15) / solver.dt)
    stride = max(1, nt // n_frames)

    frames_ex = []
    times_fs = []
    for step in range(nt):
        t = step * solver.dt
        multi_src.update(solver, t)
        solver.one_step()
        if step % stride == 0 and len(frames_ex) < n_frames:
            frames_ex.append(np.array(solver.E[:, :, 0, "x"], dtype=float))
            times_fs.append(float(t * 1e15))

    x_um = (grid.x[:-1] + 0.5 * grid.dx) * 1e6
    y_um = (grid.y[:-1] + 0.5 * grid.dy) * 1e6
    return {
        "channel": channel,
        "frames_ex": np.asarray(frames_ex, dtype=float),
        "times_fs": np.asarray(times_fs, dtype=float),
        "x_um": x_um,
        "y_um": y_um,
    }


def run_resolution_convergence_point(
    cpw: float,
    channel: str = "G",
    tail_time_s: float = 16e-15,
) -> dict:
    """Run a single-dipole resolution convergence point at the specified ``cpw``.

    Uses a compact 2D domain around the Green subpixel ($x \\in [-1.55, 1.55]\\;\\mu\\text{m}$,
    $y \\in [-1.32, 0.48]\\;\\mu\\text{m}$) with ``graded_axis`` snapped to all layer
    and mesa interfaces so that ``cpw`` from 10 up to the 28-cpw reference run
    probes spatial convergence of $\\text{LEE}_{\\text{tot}}$ and $\\eta_{\\pm 20^\\circ}$.

    Parameters
    ----------
    cpw : float
        Cells per minimum wavelength in each material ($\\ge 10$).
    channel : {"B", "G", "R"}, optional
        Active channel. Default is ``"G"``.
    tail_time_s : float, optional
        Post-pulse ringdown time [s]. Default is ``16e-15``.

    Returns
    -------
    dict
        Dictionary with ``cpw``, ``dx_nm``, ``dx_background_nm``, ``dx_min_nm``,
        ``min_feature_nm``, ``cells_across_feature``, ``skin_depth_nm``,
        ``cells_per_skin_depth``, ``n_cells``, ``n_steps``, ``lee_tot_pct``,
        ``eta_20_pct``, ``u_2d``, ``p_ratio_density``, and ``runtime_s``.
    """
    t0 = time.perf_counter()
    theta_deg = np.linspace(-90.0, 90.0, 181)
    grid, solver, meta = build_rgb_array_solver(
        channel=channel,
        cpw=cpw,
        domain_x=(-1.55e-6, 1.55e-6),
        domain_y=(-1.32e-6, 0.48e-6),
    )
    f0 = meta["f0_hz"]
    dz = meta["dz_m"]
    src_time = GaussianSource(frequency=f0, fwidth=0.35 * f0, cutoff=4.0)
    nt = int((src_time.end_time + float(tail_time_s)) / solver.dt)

    x0 = RGB_MATERIALS[channel]["x_center_m"]
    w_ch = PIXEL_GEOMETRY["channel_width_m"]
    dip = CurrentSource(
        src_time,
        component="x",
        center=(x0 + 0.12e-6, PIXEL_GEOMETRY["y_mqw_center_m"], 0.0),
        size=(0.0, 0.0, dz),
    )
    mqw_box = make_flux_box_2d(
        [f0],
        center=(x0, PIXEL_GEOMETRY["y_mqw_center_m"], 0.0),
        width_x=w_ch,
        height_y=0.22e-6,
    )
    top_n2f = wakis.Near2FarMonitor(
        [f0], center=(0.0, 0.22e-6, 0.0), size=(2.60e-6, 0.0, np.inf)
    )
    solver.emsolve(nt, source=dip, monitors=[mqw_box, top_n2f])

    flux_mqw = float(mqw_box.get_flux()[0])
    flux_top = float(top_n2f.get_flux()[0])
    lee_tot_pct = 100.0 * flux_top / max(1e-30, flux_mqw)
    ff_res = project_near2far_2d(top_n2f, theta_deg, dz_m=dz)
    runtime_s = time.perf_counter() - t0

    return {
        "cpw": float(cpw),
        "dx_nm": float(meta["dx_background_nm"]),
        "dx_background_nm": float(meta["dx_background_nm"]),
        "dx_min_nm": float(meta["dx_min_nm"]),
        "min_feature_nm": float(meta["min_feature_nm"]),
        "cells_across_feature": int(meta["cells_across_feature"]),
        "cells_across_trench": int(meta["cells_across_trench"]),
        "skin_depth_nm": float(meta["skin_depth_nm"]),
        "field_skin_depth_nm": float(meta["field_skin_depth_nm"]),
        "cells_per_skin_depth": float(meta["cells_per_skin_depth"]),
        "cells_per_field_skin_depth": float(meta["cells_per_field_skin_depth"]),
        "cells_per_power_skin_depth": float(meta["cells_per_power_skin_depth"]),
        "nx": int(meta["nx"]),
        "ny": int(meta["ny"]),
        "n_cells": int(meta["n_cells"]),
        "nt": int(nt),
        "n_steps": int(nt),
        "lee_tot_pct": float(lee_tot_pct),
        "eta_20_pct": float(ff_res["eta_20_pct"]),
        "theta_deg": theta_deg,
        "u_2d": ff_res["u_2d"],
        "p_ratio_density": ff_res["p_ratio_density"],
        "runtime_s": float(runtime_s),
    }

