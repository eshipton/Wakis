"""Helper utilities for Tutorial 7.1: Planar AlGaN UV LED (2D Wakis FIT).

Provides:
1. Analytic 2D and 3D escape-cone formulas (with and without Fresnel
   transmission and dipole radiation patterns) and the Beer-Lambert + Fresnel
   stratified model for the planar AlGaN deep-UV LED stack of Ryu (2014).
2. Subpixel Kottke tensor smoothing vs. staircase material assignment for
   lossy multilayer stacks in 2D (x-z) Wakis FIT simulations (`Ny = 1`,
   `bc_y = 'periodic'`) using `wakis.dispersive.DispersiveMaterial`.
3. Wakis-native dipole emission simulation (`CurrentSource`, `GaussianSource`,
   `DFTMonitor`, `FluxMonitor`, `Near2FarMonitor`), 2D cylindrical Green's
   function scaling of `Near2FarMonitor`, and incoherent vs. coherent
   multi-dipole averaging.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
from scipy.constants import c as C0, epsilon_0 as EPS0, mu_0 as MU0

import wakis
from wakis.dispersive import DispersiveMaterial
from wakis.field_monitors import DFTMonitor, FluxMonitor
from wakis.near2far import Near2FarMonitor
from wakis.sources import CurrentSource, GaussianSource
from wakis.templates import Box
from wakis.templates.meshing import graded_axis
from wakis.templates.shapes import assign_materials

ETA0 = float(np.sqrt(MU0 / EPS0))
# Beer-Lambert / field intensity skin depth in p-GaN at 280 nm (alpha = 170,000 cm^-1):
# delta = lambda_0 / (4 * pi * kappa) = 1 / alpha = 58.8235 nm
# (Note: 1/e electric-field amplitude decay length is 2 * delta = lambda_0 / (2 * pi * kappa) = 117.65 nm)
SKIN_DEPTH_PGAN_NM = float(1.0e7 / 170000.0)

# ---------------------------------------------------------------------------
# Published reference data from Ryu, Nanoscale Res. Lett. 9, 58 (2014)
# doi:10.1186/1556-276X-9-58
# ---------------------------------------------------------------------------
RYU_2014_REFERENCE = {
    "doi": "10.1186/1556-276X-9-58",
    "citation": "H.-Y. Ryu, Nanoscale Res. Lett. 9, 58 (2014)",
    "wavelength_nm": 280.0,
    "spectral_width_nm": 10.0,
    "layers": {
        "sapphire": {"n": 1.80, "alpha_cm": 0.0, "thickness_nm": None},
        "n_algan": {"n": 2.60, "alpha_cm": 10.0, "thickness_nm": 2000.0},
        "mqw": {"n": 2.60, "alpha_cm": 1000.0, "thickness_nm": 50.0},
        "p_algan": {"n": 2.60, "alpha_cm": 10.0, "thickness_nm": 50.0},
        "p_gan": {"n": 2.90, "alpha_cm": 170000.0, "thickness_nm_range": (0.0, 200.0)},
        "air": {"n": 1.00, "alpha_cm": 0.0, "thickness_nm": None},
    },
    # Reference points from Ryu (2014) Fig. 3 (planar LED):
    # Verified directly from the paper text ("Results and discussion"):
    # - At t_pGaN = 0 nm: LEE_TE ~ 4.0%, LEE_TM ~ 0.4% (ratio ~ 10:1)
    # - Exponential (log-linear) decrease with p-GaN thickness
    # - For t_pGaN > 80 nm: LEE_TE < 1.0%
    # - At t_pGaN = 100 nm: LEE_TM ~ 0.1% (and LEE_TE ~ 0.72%)
    "fig3_pgan_nm": np.array([0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0]),
    "fig3_lee_te_3d_pct": np.array([4.00, 2.60, 1.70, 1.10, 0.72, 0.31, 0.13]),
    "fig3_lee_tm_3d_pct": np.array([0.40, 0.28, 0.20, 0.14, 0.10, 0.048, 0.023]),
}


# ---------------------------------------------------------------------------
# 1. Analytic escape-cone & Fresnel formulas (2D vs 3D)
# ---------------------------------------------------------------------------
def escape_cone_angle(n_slab: float, n_ambient: float = 1.0) -> float:
    """Return the total internal reflection critical angle theta_c [rad].

    Parameters
    ----------
    n_slab : float
        Refractive index of the high-index emitting slab.
    n_ambient : float, optional
        Refractive index of the extraction medium (default 1.0 for air).

    Returns
    -------
    float
        Critical angle ``arcsin(n_ambient / n_slab)`` in radians.
    """
    if n_ambient >= n_slab:
        return float(0.5 * np.pi)
    return float(np.arcsin(n_ambient / n_slab))


def escape_cone_3d_nofresnel(
    n_slab: float | np.ndarray, n_ambient: float = 1.0
) -> np.ndarray:
    """Return the single-face 3D isotropic escape-cone fraction without Fresnel loss.

    In 3D, a cone of half-angle ``theta_c`` subtends solid angle
    ``Omega = 2 * pi * (1 - cos(theta_c))`` out of ``4 * pi`` steradians:

    .. math::
        \\eta_{\\text{3D, cone}} = \\frac{1 - \\cos\\theta_c}{2}
        \\approx \\frac{n_{\\text{amb}}^2}{4 n_{\\text{slab}}^2}
    """
    n_arr = np.asarray(n_slab, dtype=float)
    ratio = np.clip(n_ambient / n_arr, 0.0, 1.0)
    cos_tc = np.sqrt(1.0 - ratio**2)
    return 0.5 * (1.0 - cos_tc)


def escape_cone_2d_nofresnel(
    n_slab: float | np.ndarray, n_ambient: float = 1.0
) -> np.ndarray:
    """Return the single-face 2D isotropic escape-cone fraction without Fresnel loss.

    In 2D (x-z plane), rays within ``[-theta_c, +theta_c]`` of the +z surface
    normal subtend plane angle ``2 * theta_c`` out of ``2 * pi`` radians:

    .. math::
        \\eta_{\\text{2D, cone}} = \\frac{\\theta_c}{\\pi}
        = \\frac{\\arcsin(n_{\\text{amb}} / n_{\\text{slab}})}{\\pi}
        \\approx \\frac{n_{\\text{amb}}}{\\pi n_{\\text{slab}}}
    """
    n_arr = np.asarray(n_slab, dtype=float)
    ratio = np.clip(n_ambient / n_arr, 0.0, 1.0)
    return np.arcsin(ratio) / np.pi


def fresnel_transmission(
    theta_i: np.ndarray | float,
    n1: float,
    n2: float,
    pol: Literal["s", "p"] = "s",
) -> np.ndarray:
    """Compute Fresnel power transmittance T(theta_i) from medium n1 to n2."""
    th = np.asarray(theta_i, dtype=float)
    cos_i = np.cos(th)
    sin_t_sq = (n1 / n2) ** 2 * np.sin(th) ** 2
    T = np.zeros_like(th, dtype=float)
    mask = (sin_t_sq < 1.0) & (cos_i > 0.0)
    if not np.any(mask):
        return T
    cos_t = np.sqrt(1.0 - sin_t_sq[mask])
    ci = cos_i[mask]
    if pol == "s":
        r = (n1 * ci - n2 * cos_t) / (n1 * ci + n2 * cos_t)
    elif pol == "p":
        r = (n2 * ci - n1 * cos_t) / (n2 * ci + n1 * cos_t)
    else:
        raise ValueError(f"Unknown polarization {pol!r}; expected 's' or 'p'.")
    T[mask] = 1.0 - np.abs(r) ** 2
    return T


def escape_cone_2d_fresnel(
    n_slab: float,
    n_ambient: float = 1.0,
    dipole: Literal["y", "x", "z"] = "y",
    n_quad: int = 2001,
) -> float:
    """Compute exact 2D single-face extraction efficiency with Fresnel reflection.

    In a 2D (x-z) cross-section:
    - ``dipole='y'`` (out-of-plane line current J_y, s-polarized): isotropic
      2D angular power pattern ``p(theta) = 1 / (2*pi)``.
    - ``dipole='x'`` (in-plane horizontal dipole J_x, p-polarized in x-z,
      representing the TE optical mode E || QW plane): angular pattern
      ``p(theta) = cos^2(theta) / pi``.
    - ``dipole='z'`` (vertical dipole J_z || c-axis, p-polarized in x-z,
      representing the TM optical mode E || c-axis): angular pattern
      ``p(theta) = sin^2(theta) / pi``.
    """
    tc = escape_cone_angle(n_slab, n_ambient)
    th = np.linspace(-tc, tc, n_quad)
    if dipole == "y":
        T = fresnel_transmission(th, n_slab, n_ambient, pol="s")
        return float(np.trapezoid(T, th) / (2.0 * np.pi))
    if dipole == "x":
        T = fresnel_transmission(th, n_slab, n_ambient, pol="p")
        return float(np.trapezoid(T * np.cos(th) ** 2, th) / np.pi)
    if dipole == "z":
        T = fresnel_transmission(th, n_slab, n_ambient, pol="p")
        return float(np.trapezoid(T * np.sin(th) ** 2, th) / np.pi)
    raise ValueError(f"Unknown dipole orientation {dipole!r}; expected 'y', 'x', or 'z'.")


def escape_cone_3d_fresnel(
    n_slab: float,
    n_ambient: float = 1.0,
    dipole: Literal["te", "tm", "iso"] = "te",
    n_quad: int = 2001,
) -> float:
    """Compute exact 3D single-face extraction efficiency with Fresnel reflection."""
    tc = escape_cone_angle(n_slab, n_ambient)
    th = np.linspace(0.0, tc, n_quad)
    Ts = fresnel_transmission(th, n_slab, n_ambient, pol="s")
    Tp = fresnel_transmission(th, n_slab, n_ambient, pol="p")
    sin_th = np.sin(th)
    if dipole == "te":
        integrand = 0.375 * (Ts + Tp * np.cos(th) ** 2) * sin_th
        return float(np.trapezoid(integrand, th))
    if dipole == "tm":
        integrand = 0.75 * Tp * (sin_th**3)
        return float(np.trapezoid(integrand, th))
    if dipole == "iso":
        eta_te = escape_cone_3d_fresnel(n_slab, n_ambient, "te", n_quad)
        eta_tm = escape_cone_3d_fresnel(n_slab, n_ambient, "tm", n_quad)
        return float((2.0 * eta_te + eta_tm) / 3.0)
    raise ValueError(f"Unknown 3D dipole type {dipole!r}.")


def ryu_beer_lambert_transfer(
    t_pgan_nm: float | np.ndarray,
    n_algan: float = 2.60,
    n_pgan: float = 2.90,
    n_sapphire: float = 1.80,
    alpha_pgan_cm: float = 170000.0,
    alpha_mqw_cm: float = 1000.0,
    t_mqw_nm: float = 50.0,
    n_quad: int = 2001,
) -> dict[str, np.ndarray]:
    """Analytic angle-resolved Beer-Lambert + Fresnel model for Ryu (2014) stack."""
    t_arr = np.atleast_1d(np.asarray(t_pgan_nm, dtype=float))
    tc = escape_cone_angle(n_algan, 1.0)
    th1 = np.linspace(0.0, tc - 1e-7, n_quad)
    sin_th1 = np.sin(th1)
    cos_th1 = np.cos(th1)

    sin_th2 = (n_algan / n_pgan) * sin_th1
    cos_th2 = np.sqrt(1.0 - sin_th2**2)

    alpha_mqw_m = alpha_mqw_cm * 100.0
    atten_mqw = np.exp(-alpha_mqw_m * (0.5 * t_mqw_nm * 1e-9) / cos_th1)

    R_sap_s = 1.0 - fresnel_transmission(th1, n_algan, n_sapphire, pol="s")
    R_sap_p = 1.0 - fresnel_transmission(th1, n_algan, n_sapphire, pol="p")
    atten_mqw_full = np.exp(-alpha_mqw_m * (t_mqw_nm * 1e-9) / cos_th1)

    T12_s = fresnel_transmission(th1, n_algan, n_pgan, pol="s")
    T12_p = fresnel_transmission(th1, n_algan, n_pgan, pol="p")
    th2 = np.arcsin(sin_th2)
    T23_s = fresnel_transmission(th2, n_pgan, 1.0, pol="s")
    T23_p = fresnel_transmission(th2, n_pgan, 1.0, pol="p")

    T_direct_s = fresnel_transmission(th1, n_algan, 1.0, pol="s")
    T_direct_p = fresnel_transmission(th1, n_algan, 1.0, pol="p")

    alpha_pgan_m = alpha_pgan_cm * 100.0

    lee_2d_jx = np.zeros_like(t_arr)
    lee_2d_jy = np.zeros_like(t_arr)
    lee_2d_jz = np.zeros_like(t_arr)
    lee_3d_te = np.zeros_like(t_arr)
    lee_3d_tm = np.zeros_like(t_arr)

    for idx, t_nm in enumerate(t_arr):
        t_m = t_nm * 1e-9
        atten_pgan = np.exp(-alpha_pgan_m * t_m / cos_th2)
        w = 1.0 - np.exp(-t_nm / 8.0)
        Ts_eff = ((1.0 - w) * T_direct_s + w * T12_s * T23_s) * atten_pgan
        Tp_eff = ((1.0 - w) * T_direct_p + w * T12_p * T23_p) * atten_pgan

        mult_s = atten_mqw * (1.0 + R_sap_s * atten_mqw_full)
        mult_p = atten_mqw * (1.0 + R_sap_p * atten_mqw_full)

        Ts_tot = Ts_eff * mult_s
        Tp_tot = Tp_eff * mult_p

        lee_2d_jy[idx] = 2.0 * np.trapezoid(Ts_tot, th1) / (2.0 * np.pi)
        lee_2d_jx[idx] = 2.0 * np.trapezoid(Tp_tot * cos_th1**2, th1) / np.pi
        lee_2d_jz[idx] = 2.0 * np.trapezoid(Tp_tot * sin_th1**2, th1) / np.pi

        lee_3d_te[idx] = np.trapezoid(
            0.375 * (Ts_tot + Tp_tot * cos_th1**2) * sin_th1, th1
        )
        lee_3d_tm[idx] = np.trapezoid(0.75 * Tp_tot * (sin_th1**3), th1)

    return {
        "t_pgan_nm": t_arr,
        "lee_2d_jx": lee_2d_jx,
        "lee_2d_jy": lee_2d_jy,
        "lee_2d_jz": lee_2d_jz,
        "lee_3d_te": lee_3d_te,
        "lee_3d_tm": lee_3d_tm,
    }


# ---------------------------------------------------------------------------
# 2. Semi-infinite slab 2D Wakis FIT validation helper
# ---------------------------------------------------------------------------
def build_semi_infinite_mesh_2d(
    n_slab: float = 2.60,
    cpw: float = 12.0,
    lam0: float = 280e-9,
    lx: float = 3.2e-6,
    zmin: float = -0.45e-6,
    zmax: float = 0.35e-6,
    z_dip: float = -0.18e-6,
) -> dict:
    """Build the 2D (x-z) mesh for the semi-infinite slab validation without running the solver."""
    dx_target = lam0 / (cpw * n_slab)
    dy = 1.0e-6
    nx = max(24, int(np.ceil(lx / dx_target)))
    if nx % 2 == 1:
        nx += 1
    dx = lx / nx
    nz_neg = max(8, int(np.ceil(abs(zmin) / dx)))
    nz_pos = max(8, int(np.ceil(zmax / dx)))
    nz = nz_neg + nz_pos
    zmin_act, zmax_act = -nz_neg * dx, nz_pos * dx

    x_nodes = np.linspace(-0.5 * nx * dx, 0.5 * nx * dx, nx + 1)
    y_nodes = np.array([-0.5 * dy, 0.5 * dy])
    z_nodes = np.linspace(zmin_act, zmax_act, nz + 1)
    grid = wakis.GridFIT3D(x=x_nodes, y=y_nodes, z=z_nodes, verbose=0)

    zc = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    min_feature_nm = float(min(abs(z_dip), zmax_act, abs(zmin_act - z_dip)) * 1e9)
    cells_across_dip_gap = int(np.sum((zc >= z_dip) & (zc <= 0.0)))
    dx_bg_nm = float(np.max(grid.dx) * 1e9)
    dx_min_nm = float(min(np.min(grid.dx), np.min(grid.dz)) * 1e9)

    return {
        "grid": grid,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "nz": nz,
        "zmin_act": zmin_act,
        "zmax_act": zmax_act,
        "z_dip": z_dip,
        "dx_background_nm": dx_bg_nm,
        "dx_min_nm": dx_min_nm,
        "min_feature_nm": min_feature_nm,
        "cells_across_feature": cells_across_dip_gap,
        "skin_depth_nm": float(np.inf),
        "cells_per_skin_depth": float(np.inf),
        "n_cells": int(nx * nz),
    }


def run_semi_infinite_slab_2d(
    n_slab: float = 2.60,
    dipole_pol: Literal["y", "x", "z"] = "y",
    cpw: float = 12.0,
    lam0: float = 280e-9,
    lx: float = 3.2e-6,
    zmin: float = -0.45e-6,
    zmax: float = 0.35e-6,
) -> dict:
    """Validate 2D escape-cone extraction of a dipole in a semi-infinite slab.

    Uses `wakis.templates.Box` + `assign_materials`, `CurrentSource`,
    `GaussianSource`, `DFTMonitor`, and `FluxMonitor`.
    """
    t0 = time.perf_counter()
    f0 = C0 / lam0
    freqs = np.array([f0])
    mesh_info = build_semi_infinite_mesh_2d(
        n_slab=n_slab, cpw=cpw, lam0=lam0, lx=lx, zmin=zmin, zmax=zmax
    )
    grid = mesh_info["grid"]
    dx = mesh_info["dx"]
    dy = mesh_info["dy"]
    nx = mesh_info["nx"]
    zmin_act = mesh_info["zmin_act"]
    z_dip = mesh_info["z_dip"]

    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "periodic", "cpml"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        verbose=0,
    )
    slab = Box(
        center=(0.0, 0.0, 0.5 * zmin_act),
        size=(2.0 * lx, 10.0 * dy, abs(zmin_act)),
        material=n_slab**2,
    )
    assign_materials(solver, [slab], f0)

    src_time = GaussianSource(frequency=f0, fwidth=0.20 * f0)
    dip = CurrentSource(src_time, component=dipole_pol, center=(0.0, 0.0, z_dip))
    pt = DFTMonitor(freqs, components=["E" + dipole_pol], center=(0.0, 0.0, z_dip))

    z_ext = 2.0 * dx
    z_bot = zmin_act + 9.0 * dx
    w_ext = (nx - 18) * dx
    h_box = z_ext - z_bot
    z_mid = 0.5 * (z_ext + z_bot)

    flux_top = FluxMonitor(freqs, center=(0.0, 0.0, z_ext), size=(w_ext, np.inf, 0.0))
    flux_bot = FluxMonitor(freqs, center=(0.0, 0.0, z_bot), size=(w_ext, np.inf, 0.0))
    flux_r = FluxMonitor(
        freqs, center=(0.5 * w_ext, 0.0, z_mid), size=(0.0, np.inf, h_box)
    )
    flux_l = FluxMonitor(
        freqs, center=(-0.5 * w_ext, 0.0, z_mid), size=(0.0, np.inf, h_box)
    )

    Nt = int((src_time.end_time + 12e-15) / solver.dt)
    solver.emsolve(
        Nt,
        source=[dip],
        monitors=[pt, flux_top, flux_bot, flux_r, flux_l],
        stop_when_energy_decayed=1e-4,
    )

    S = src_time.fourier_transform(freqs)
    p_tot = float(-np.real(pt.get_field("E" + dipole_pol)[0].ravel()[0] * np.conj(S[0])))
    p_top = float(flux_top.get_flux()[0])
    p_bot = float(-flux_bot.get_flux()[0])
    p_sides = float(flux_r.get_flux()[0] - flux_l.get_flux()[0])
    p_box = p_top + p_bot + p_sides

    return {
        "n_slab": n_slab,
        "dipole_pol": dipole_pol,
        "cpw": cpw,
        "lee_top": p_top / p_tot,
        "lee_analytic": escape_cone_2d_fresnel(n_slab, 1.0, dipole=dipole_pol),
        "lee_cone_nofresnel": float(escape_cone_2d_nofresnel(n_slab, 1.0)),
        "box_over_inj": p_box / p_tot,
        "dx_background_nm": mesh_info["dx_background_nm"],
        "dx_min_nm": mesh_info["dx_min_nm"],
        "min_feature_nm": mesh_info["min_feature_nm"],
        "cells_across_feature": mesh_info["cells_across_feature"],
        "skin_depth_nm": mesh_info["skin_depth_nm"],
        "cells_per_skin_depth": mesh_info["cells_per_skin_depth"],
        "n_cells": mesh_info["n_cells"],
        "n_steps": int(pt.shape["E" + dipole_pol][0] if False else Nt),
        "wall_time_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# 3. Subpixel Kottke vs Staircase 2D Multilayer + DispersiveMaterial ADE
# ---------------------------------------------------------------------------
@dataclass
class LayerSpec:
    """Specification of a horizontal layer ``z in [z_min, z_max]``."""

    name: str
    z_min: float
    z_max: float
    n: float
    alpha_cm: float = 0.0

    def to_dispersive_material(self, lam0: float) -> DispersiveMaterial:
        """Create a Wakis `DispersiveMaterial` with complex index ``n + i*kappa``."""
        omega0 = 2.0 * np.pi * C0 / lam0
        alpha_m = self.alpha_cm * 100.0
        kappa = alpha_m * lam0 / (4.0 * np.pi)
        eps_inf = max(1.0, float(self.n**2 - kappa**2))
        sigma = float(2.0 * self.n * kappa * omega0 * EPS0)
        return DispersiveMaterial(eps_inf=eps_inf, sigma=sigma, name=self.name)


class _TimeTraceAndMovieRecorder:
    """Lightweight monitor compatible with `solver.emsolve(..., monitors=[...])`."""

    def __init__(
        self,
        ix_src: int,
        iz_src: int,
        ix_mid: int,
        iz_top: int,
        comp: str,
        dV: float,
        record_movie: bool = False,
        movie_stride: int = 25,
    ):
        self.ix_src = ix_src
        self.iz_src = iz_src
        self.ix_mid = ix_mid
        self.iz_top = iz_top
        self.comp = comp
        self.dV = dV
        self.record_movie = record_movie
        self.movie_stride = movie_stride
        self.step = 0
        self.time_fs: list[float] = []
        self.src_trace: list[float] = []
        self.top_trace: list[float] = []
        self.energy_trace: list[float] = []
        self.movie_frames: list[dict] = []

    def update(self, solver: wakis.SolverFIT3D, t: float) -> None:
        self.time_fs.append(float(t * 1e15))
        self.src_trace.append(
            float(solver.E[self.ix_src, 0, self.iz_src, self.comp])
        )
        self.top_trace.append(
            float(solver.E[self.ix_mid, 0, self.iz_top, self.comp])
        )
        if self.step % 8 == 0 or len(self.energy_trace) == 0:
            e_arr = solver.E.toarray()
            e_sq = float(np.dot(e_arr, e_arr))
            self.energy_trace.append(0.5 * EPS0 * e_sq * self.dV)
        else:
            self.energy_trace.append(self.energy_trace[-1])

        if self.record_movie and (self.step % self.movie_stride == 0):
            self.movie_frames.append(
                {
                    "step": self.step,
                    "t_fs": float(t * 1e15),
                    "field": solver.E.to_matrix(self.comp)[:, 0, :].copy(),
                }
            )
        self.step += 1


def assign_planar_stack_materials(
    solver: wakis.SolverFIT3D,
    layers: list[LayerSpec],
    lam0: float,
    averaging: Literal["kottke", "staircase"] = "kottke",
    dipole_xz: tuple[float, float] | None = None,
    source_clear_radius_m: float = 22e-9,
) -> dict[str, np.ndarray]:
    """Assign subpixel-smoothed (Kottke) or staircase permittivity & ADE conductivity.

    Supports both uniform and non-uniform (graded) ``z`` grids:
    - Tangential electric components (``E_x, E_y``) lie on primal ``z``-nodes
      ``z_node = grid.z[:-1]`` with dual cell ``[z_dual_lo[k], z_dual_hi[k]]``
      spanning between adjacent cell midpoints ``zc_edge``, so their effective
      relative permittivity is the dual-volume-weighted arithmetic average:
      ``eps_x = (1 - w_tan) * eps_bg + w_tan * eps_layer``.
    - Normal electric component (``E_z``) lies on primary ``z``-edges centered
      at ``zc_edge = 0.5 * (grid.z[:-1] + grid.z[1:])`` with edge length
      ``dz_edge = np.diff(grid.z)``, so its effective permittivity is the
      subcell harmonic average:
      ``1 / eps_z = w_nrm / eps_layer + (1 - w_nrm) / eps_bg``.
    - In ``staircase`` mode, fill weights ``w_tan`` and ``w_nrm`` are rounded
      to ``{0.0, 1.0}`` at ``w >= 0.5``, shifting interface tangential nodes by
      half a dual cell.
    - Absorbing slices (``sigma > 0``) are registered via Wakis's semi-implicit
      Crank-Nicolson ADE update (`solver._add_dispersive_edges`) with the exact
      slice effective ``(eps_eff, sigma_eff)`` and unit weight so interface
      permittivity is never double-counted. Inside the MQW active layer
      (``z < -25 nm``), a small ``source_clear_radius_m`` neighborhood around
      the point dipole is kept lossless so the 2D ``1/r^2`` quasi-static near
      field does not cause grid-divergent local Ohmic self-quenching.
    """
    if averaging not in ("kottke", "staircase"):
        raise ValueError(f"Unknown averaging mode {averaging!r}.")

    Nx, Ny, Nz = solver.Nx, solver.Ny, solver.Nz
    N = solver.N
    grid = solver.grid
    z_nodes = grid.z
    zc_edge = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    z_node = z_nodes[:-1]
    dz_edge = np.diff(z_nodes)

    z_dual_lo = np.empty(Nz, dtype=float)
    z_dual_hi = np.empty(Nz, dtype=float)
    z_dual_lo[0] = z_nodes[0] - 0.5 * dz_edge[0]
    z_dual_lo[1:] = zc_edge[:-1]
    z_dual_hi[:] = zc_edge
    dz_dual = z_dual_hi - z_dual_lo

    eps_x_1d = np.ones(Nz, dtype=float)
    eps_z_1d = np.ones(Nz, dtype=float)
    sigma_x_1d = np.zeros(Nz, dtype=float)
    sigma_z_1d = np.zeros(Nz, dtype=float)

    for lay in layers:
        mat = lay.to_dispersive_material(lam0)
        w_tan = np.clip(
            (np.minimum(z_dual_hi, lay.z_max) - np.maximum(z_dual_lo, lay.z_min))
            / dz_dual,
            0.0,
            1.0,
        )
        w_nrm = np.clip(
            (np.minimum(z_nodes[1:], lay.z_max) - np.maximum(z_nodes[:-1], lay.z_min))
            / dz_edge,
            0.0,
            1.0,
        )
        if averaging == "staircase":
            w_tan = np.where(w_tan >= 0.5, 1.0, 0.0)
            w_nrm = np.where(w_nrm >= 0.5, 1.0, 0.0)

        for k in np.flatnonzero(w_tan > 0.0):
            wk = float(w_tan[k])
            eps_x_1d[k] = (1.0 - wk) * eps_x_1d[k] + wk * mat.eps_inf
            sigma_x_1d[k] = (1.0 - wk) * sigma_x_1d[k] + wk * mat.sigma

        for k in np.flatnonzero(w_nrm > 0.0):
            wk = float(w_nrm[k])
            eps_z_1d[k] = 1.0 / (wk / mat.eps_inf + (1.0 - wk) / eps_z_1d[k])
            sigma_z_1d[k] = (1.0 - wk) * sigma_z_1d[k] + wk * mat.sigma

    eps_x = np.broadcast_to(eps_x_1d[None, None, :], (Nx, Ny, Nz))
    eps_z = np.broadcast_to(eps_z_1d[None, None, :], (Nx, Ny, Nz))

    ieps = solver.ieps.toarray().copy()
    ieps[0:N] = (1.0 / (EPS0 * eps_x)).ravel(order="F")
    ieps[N : 2 * N] = (1.0 / (EPS0 * eps_x)).ravel(order="F")
    ieps[2 * N : 3 * N] = (1.0 / (EPS0 * eps_z)).ravel(order="F")
    solver.ieps.fromarray(ieps)

    x_nodes = grid.x[:-1]
    xc_edge = 0.5 * (grid.x[:-1] + grid.x[1:])

    for c_idx, (eps_1d, sig_1d, z_arr, x_arr) in enumerate(
        (
            (eps_x_1d, sigma_x_1d, z_node, xc_edge),
            (eps_x_1d, sigma_x_1d, z_node, x_nodes),
            (eps_z_1d, sigma_z_1d, zc_edge, x_nodes),
        )
    ):
        for k in np.flatnonzero(sig_1d > 0.0):
            ix_valid = np.arange(Nx)
            if dipole_xz is not None and z_arr[k] < -25e-9 and source_clear_radius_m > 0.0:
                x0, z0 = dipole_xz
                if abs(z_arr[k] - z0) <= source_clear_radius_m:
                    ix_valid = np.flatnonzero(np.abs(x_arr - x0) > source_clear_radius_m)
            if ix_valid.size == 0:
                continue
            mat_k = DispersiveMaterial(
                eps_inf=float(eps_1d[k]),
                sigma=float(sig_1d[k]),
                name=f"slice_c{c_idx}_k{k}",
            )
            I, J, K = np.meshgrid(
                ix_valid, np.arange(Ny), np.array([k]), indexing="ij"
            )
            flat_idx = (I + J * Nx + K * Nx * Ny + c_idx * N).ravel()
            solver._add_dispersive_edges(
                mat_k, flat_idx, np.ones(flat_idx.size, dtype=float)
            )

    return {
        "z_node": z_node.copy(),
        "z_centers": zc_edge.copy(),
        "eps_x": eps_x_1d,
        "eps_z": eps_z_1d,
        "sigma_x": sigma_x_1d,
    }


def make_ryu_planar_layers(
    t_pgan_nm: float = 100.0,
    n_algan: float = 2.60,
    n_pgan: float = 2.90,
    n_sapphire: float = 1.80,
    alpha_pgan_cm: float = 170000.0,
    alpha_mqw_cm: float = 1000.0,
    alpha_algan_cm: float = 10.0,
    z_min: float = -0.85e-6,
    z_max: float = 0.40e-6,
    z_sapphire_top: float = -0.55e-6,
) -> tuple[list[LayerSpec], float]:
    """Construct the planar AlGaN UV LED stack from Ryu (2014).

    Coordinate convention:
    - ``z = 0`` is the ``p-AlGaN / p-GaN`` interface.
    - ``p-GaN`` contact layer spans ``z in [0, t_pgan]``.
    - ``p-AlGaN`` cladding (50 nm) spans ``z in [-50 nm, 0]``.
    - ``MQW`` active region (50 nm) spans ``z in [-100 nm, -50 nm]`` (center ``z = -75 nm``).
    - ``n-AlGaN`` cladding spans ``z in [z_sapphire_top, -100 nm]``.
    - ``Sapphire`` substrate spans ``z in [z_min, z_sapphire_top]``.
    - ``Air`` superstrate spans ``z in [t_pgan, z_max]``.
    """
    t_pgan = max(0.0, float(t_pgan_nm)) * 1e-9
    layers = [
        LayerSpec("sapphire", z_min, z_sapphire_top, n=n_sapphire, alpha_cm=0.0),
        LayerSpec("n_algan", z_sapphire_top, -100e-9, n=n_algan, alpha_cm=0.0),
        LayerSpec("mqw", -100e-9, -50e-9, n=n_algan, alpha_cm=alpha_mqw_cm),
        LayerSpec("p_algan", -50e-9, 0.0, n=n_algan, alpha_cm=alpha_algan_cm),
    ]
    if t_pgan > 0.0:
        layers.append(
            LayerSpec("p_gan", 0.0, t_pgan, n=n_pgan, alpha_cm=alpha_pgan_cm)
        )
    return layers, -75e-9


def build_planar_led_mesh_2d(
    layers: list[LayerSpec],
    z_dipole: float = -75e-9,
    lam0: float = 280e-9,
    n_max_ref: float = 2.90,
    cpw: float = 12.0,
    dz_fine_nm: float | None = None,
    min_cells_per_layer: int = 4,
    Lx: float = 2.8e-6,
    z_min: float = -0.85e-6,
    z_max: float = 0.40e-6,
    grading: float = 1.35,
) -> dict:
    """Build the 2D (x-z) graded mesh for the planar LED stack without running the solver.

    Places exact ``z`` nodes on every layer interface in ``layers``, the domain
    boundaries ``z_min, z_max``, and the dipole plane ``z_dipole`` via
    `wakis.templates.meshing.graded_axis`.
    """
    dx_bg_target = lam0 / (cpw * n_max_ref)
    dy = 1.0e-6
    nx = max(24, int(np.ceil(Lx / dx_bg_target)))
    if nx % 2 == 1:
        nx += 1
    x_nodes = np.linspace(-0.5 * Lx, 0.5 * Lx, nx + 1)
    y_nodes = np.array([-0.5 * dy, 0.5 * dy])
    dx_act = float(x_nodes[1] - x_nodes[0])

    # Default fine vertical cell size inside the active/cladding/contact stack:
    # resolves a 25 nm feature with >= min_cells_per_layer cells and scales with cpw
    if dz_fine_nm is None:
        n_per_25 = max(min_cells_per_layer, int(np.ceil(cpw / 2.5)))
        dz_fine_m = min(dx_bg_target, 25e-9 / max(1, n_per_25))
    else:
        dz_fine_m = float(dz_fine_nm) * 1e-9

    t_pgan_top = max([lay.z_max for lay in layers if lay.name == "p_gan"] + [0.0])

    pts = [z_min, z_max, z_dipole]
    for lay in layers:
        pts.extend([lay.z_min, lay.z_max])
    pts_sorted = np.unique(np.round(pts, 15))
    pts_sorted = pts_sorted[
        (pts_sorted >= z_min - 1e-12) & (pts_sorted <= z_max + 1e-12)
    ]

    sizes = []
    min_cells = []
    # When dz_fine_m > dx_bg_target (under-resolved demonstration grid), allow
    # the vertical background target to match dz_fine_m so graded_axis does not
    # artificially refine the p-GaN absorber from adjacent cladding intervals.
    dz_cladding_m = max(dx_bg_target, dz_fine_m)
    for a, b in zip(pts_sorted[:-1], pts_sorted[1:]):
        mid = 0.5 * (a + b)
        length = b - a
        if -105e-9 <= mid <= t_pgan_top + 5e-9:
            dz_t = dz_fine_m
            mc = max(min_cells_per_layer, int(np.round(length / dz_t)))
        else:
            dz_t = dz_cladding_m
            mc = max(2, int(np.ceil(length / dz_t)))
        sizes.append(dz_t)
        min_cells.append(mc)

    z_nodes = graded_axis(pts_sorted, sizes, min_cells=min_cells, grading=grading)
    grid = wakis.GridFIT3D(x=x_nodes, y=y_nodes, z=z_nodes, verbose=0)

    dz_arr = np.diff(z_nodes)
    zc = 0.5 * (z_nodes[:-1] + z_nodes[1:])

    # Measure minimum feature thickness and cells across every physical layer
    layer_thicknesses_nm = []
    layer_cell_counts = []
    for lay in layers:
        thick_nm = (lay.z_max - lay.z_min) * 1e9
        if thick_nm <= 0.0:
            continue
        layer_thicknesses_nm.append(thick_nm)
        cnt = int(np.sum((zc > lay.z_min) & (zc < lay.z_max)))
        layer_cell_counts.append(cnt)

    min_feature_nm = float(min(layer_thicknesses_nm))
    cells_across_feature = int(min(layer_cell_counts))

    # Measure actual cell size in the p-GaN absorber (or p-AlGaN if t_pGaN == 0)
    if t_pgan_top > 0.0:
        in_abs = (zc > 0.0) & (zc < t_pgan_top)
    else:
        in_abs = (zc > -50e-9) & (zc < 0.0)
    dz_abs_nm = float(np.mean(dz_arr[in_abs]) * 1e9)
    cells_per_skin = float(SKIN_DEPTH_PGAN_NM / dz_abs_nm)

    dx_bg_nm = float(dx_act * 1e9)
    dx_min_nm = float(min(dx_act, float(np.min(dz_arr))) * 1e9)

    return {
        "grid": grid,
        "dx_act": dx_act,
        "dy": dy,
        "nx": nx,
        "nz": int(len(z_nodes) - 1),
        "zmin_act": float(z_nodes[0]),
        "zmax_act": float(z_nodes[-1]),
        "dx_background_nm": dx_bg_nm,
        "dx_min_nm": dx_min_nm,
        "dz_absorber_nm": dz_abs_nm,
        "min_feature_nm": min_feature_nm,
        "cells_across_feature": cells_across_feature,
        "skin_depth_nm": SKIN_DEPTH_PGAN_NM,
        "cells_per_skin_depth": cells_per_skin,
        "n_cells": int(nx * (len(z_nodes) - 1)),
    }


def run_2d_dipole_simulation(
    layers: list[LayerSpec],
    z_dipole: float,
    x_dipole: float = 0.0,
    dipole_pol: Literal["x", "y", "z"] = "x",
    lam0: float = 280e-9,
    n_max_ref: float = 2.90,
    cpw: float = 12.0,
    dz_fine_nm: float | None = None,
    min_cells_per_layer: int = 4,
    Lx: float = 2.8e-6,
    z_min: float = -0.85e-6,
    z_max: float = 0.40e-6,
    averaging: Literal["kottke", "staircase"] = "kottke",
    extra_time_fs: float = 26.0,
    record_dft_2d: bool = False,
    record_movie: bool = False,
    movie_stride: int = 22,
    record_n2f: bool = True,
    n_theta_n2f: int = 181,
) -> dict:
    """Run a 2D (x-z) Wakis FIT dipole simulation on an interface-snapped graded z mesh."""
    t_wall0 = time.perf_counter()
    f0 = C0 / lam0
    omega0 = 2.0 * np.pi * f0
    freqs = np.array([f0])

    mesh_info = build_planar_led_mesh_2d(
        layers=layers,
        z_dipole=z_dipole,
        lam0=lam0,
        n_max_ref=n_max_ref,
        cpw=cpw,
        dz_fine_nm=dz_fine_nm,
        min_cells_per_layer=min_cells_per_layer,
        Lx=Lx,
        z_min=z_min,
        z_max=z_max,
    )
    grid = mesh_info["grid"]
    dx = mesh_info["dx_act"]
    dy = mesh_info["dy"]
    nx = mesh_info["nx"]
    nz = mesh_info["nz"]
    zmin_act = mesh_info["zmin_act"]
    zmax_act = mesh_info["zmax_act"]

    layers_adj = [
        LayerSpec(
            lay.name,
            zmin_act if i == 0 else lay.z_min,
            lay.z_max,
            lay.n,
            lay.alpha_cm,
        )
        for i, lay in enumerate(layers)
    ]

    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "periodic", "cpml"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        verbose=0,
    )

    mat_profiles = assign_planar_stack_materials(
        solver,
        layers_adj,
        lam0=lam0,
        averaging=averaging,
        dipole_xz=(x_dipole, z_dipole),
    )

    src_time = GaussianSource(frequency=f0, fwidth=0.18 * f0)
    dip = CurrentSource(
        src_time, component=dipole_pol, center=(x_dipole, 0.0, z_dipole)
    )
    pt_src = DFTMonitor(
        freqs, components=["E" + dipole_pol], center=(x_dipole, 0.0, z_dipole)
    )

    t_pgan_top = max([lay.z_max for lay in layers_adj if lay.name == "p_gan"] + [0.0])
    z_ext = t_pgan_top + max(30e-9, 3.0 * dx)
    dz_bot_pml = float(np.max(grid.dz[:10]))
    z_bot = zmin_act + 10.0 * dz_bot_pml
    w_ext = (nx - 20) * dx
    h_box = z_ext - z_bot
    z_mid = 0.5 * (z_ext + z_bot)

    n2f_top = Near2FarMonitor(
        freqs, center=(0.0, 0.0, z_ext), size=(w_ext, np.inf, 0.0), eps_r=1.0
    )
    flux_bot = FluxMonitor(
        freqs, center=(0.0, 0.0, z_bot), size=(w_ext, np.inf, 0.0)
    )
    flux_r = FluxMonitor(
        freqs, center=(0.5 * w_ext, 0.0, z_mid), size=(0.0, np.inf, h_box)
    )
    flux_l = FluxMonitor(
        freqs, center=(-0.5 * w_ext, 0.0, z_mid), size=(0.0, np.inf, h_box)
    )

    monitors = [pt_src, n2f_top, flux_bot, flux_r, flux_l]

    dft_plane = None
    if record_dft_2d:
        dft_plane = DFTMonitor(
            freqs,
            components=["Ex", "Ey", "Ez"],
            center=(0.0, 0.0, 0.5 * (zmin_act + zmax_act)),
            size=(nx * dx, 0.0, zmax_act - zmin_act),
        )
        monitors.append(dft_plane)

    ix_src = int(np.argmin(np.abs(solver.x - x_dipole)))
    iz_src = int(np.argmin(np.abs(solver.z - z_dipole)))
    iz_top = int(np.argmin(np.abs(solver.z - z_ext)))
    comp_plot = "x" if dipole_pol == "x" else ("z" if dipole_pol == "z" else "y")
    dV = float(dx * dy * float(np.mean(grid.dz)))

    recorder = _TimeTraceAndMovieRecorder(
        ix_src=ix_src,
        iz_src=iz_src,
        ix_mid=nx // 2,
        iz_top=iz_top,
        comp=comp_plot,
        dV=dV,
        record_movie=record_movie,
        movie_stride=movie_stride,
    )
    monitors.append(recorder)

    Nt = int((src_time.end_time + extra_time_fs * 1e-15) / solver.dt)
    solver.emsolve(
        Nt,
        source=[dip],
        monitors=monitors,
        stop_when_energy_decayed=1e-3,
    )

    S = src_time.fourier_transform(freqs)
    P_inj = float(
        -np.real(pt_src.get_field("E" + dipole_pol)[0].ravel()[0] * np.conj(S[0]))
    )
    P_inj = max(P_inj, 1e-99)

    Phi_top = float(n2f_top.get_flux()[0])
    Phi_bot = float(-flux_bot.get_flux()[0])
    Phi_sides = float(flux_r.get_flux()[0] - flux_l.get_flux()[0])
    Phi_total_out = Phi_top + Phi_bot + Phi_sides
    P_abs = max(0.0, P_inj - Phi_total_out)

    lee_top = float(Phi_top / P_inj)
    frac_bot = float(Phi_bot / P_inj)
    frac_sides = float(Phi_sides / P_inj)
    frac_abs = float(P_abs / P_inj)

    # Near2Far 2D angular distribution and 3D solid-angle-weighted estimate
    n2f_result = None
    lee_3d_est = None
    if record_n2f:
        theta_2d = np.linspace(-0.5 * np.pi + 1e-3, 0.5 * np.pi - 1e-3, n_theta_n2f)
        thetas_pos = np.abs(theta_2d)
        phis_pos = np.where(theta_2d >= 0.0, 0.0, np.pi)
        U_3d = np.asarray(
            n2f_top.radiation_intensity(thetas_pos, phis_pos)[0], dtype=float
        )
        k0 = omega0 / C0
        # 2D cylindrical Green's function Jacobian: dPhi_2D / dtheta = (2*pi / (k0 * dy)) * U_3D
        dPhi_2d_dtheta = (2.0 * np.pi / (k0 * dy)) * U_3d
        Phi_n2f_2d = float(np.trapezoid(dPhi_2d_dtheta, theta_2d))
        d_eta_dtheta = dPhi_2d_dtheta / P_inj

        # Axisymmetric 3D solid-angle projection via Snell's law:
        # Rays emerging in air at angle |theta_air| originated inside the active
        # layer (index n_src) at internal angle sin(theta_int) = sin|theta_air| / n_src.
        n_src_layer = float(np.sqrt(max(1.0, mat_profiles["eps_x"][iz_src])))
        sin_theta_int = np.sin(np.abs(theta_2d)) / n_src_layer
        # Exact ratio of 3D dipole solid-angle integrand (3/8 sin(theta_int) dtheta_int
        # on the symmetric [-theta_c, +theta_c] fold) to 2D dipole plane-angle integrand
        # (1/pi dtheta_int): pol_3d_factor = 3/8 = 0.375
        pol_3d_factor = 0.375
        jacobian_3d = (pol_3d_factor * np.pi) * sin_theta_int
        lee_3d_est = float(np.trapezoid(d_eta_dtheta * jacobian_3d, theta_2d))

        n2f_result = {
            "theta_rad": theta_2d,
            "theta_deg": np.degrees(theta_2d),
            "dPhi_2d_dtheta": dPhi_2d_dtheta,
            "d_eta_dtheta": d_eta_dtheta,
            "Phi_n2f_2d": Phi_n2f_2d,
            "Phi_n2f_aperture": Phi_top,
            "lee_n2f_2d": float(Phi_n2f_2d / P_inj),
            "lee_3d_est": lee_3d_est,
        }

    dft_2d = None
    if dft_plane is not None:
        Ex_raw = np.squeeze(dft_plane.get_field("Ex")[0])
        Ey_raw = np.squeeze(dft_plane.get_field("Ey")[0])
        Ez_raw = np.squeeze(dft_plane.get_field("Ez")[0])
        mx = min(nx, Ex_raw.shape[0], Ey_raw.shape[0], Ez_raw.shape[0], solver.x.size)
        mz = min(nz, Ex_raw.shape[1], Ey_raw.shape[1], Ez_raw.shape[1], solver.z.size)
        Ex_2d = Ex_raw[:mx, :mz]
        Ey_2d = Ey_raw[:mx, :mz]
        Ez_2d = Ez_raw[:mx, :mz]
        I_2d = np.abs(Ex_2d) ** 2 + np.abs(Ey_2d) ** 2 + np.abs(Ez_2d) ** 2
        dft_2d = {
            "x": solver.x[:mx].copy(),
            "z": solver.z[:mz].copy(),
            "Ex": Ex_2d,
            "Ey": Ey_2d,
            "Ez": Ez_2d,
            "I": I_2d,
        }
    for fr in recorder.movie_frames:
        fr["field"] = fr["field"][:nx, :nz]

    return {
        "cpw": cpw,
        "averaging": averaging,
        "dipole_pol": dipole_pol,
        "x_dipole": x_dipole,
        "z_dipole": z_dipole,
        "Nx": nx,
        "Nz": nz,
        "Nt": recorder.step,
        "dx_nm": mesh_info["dx_background_nm"],
        "dz_nm": mesh_info["dz_absorber_nm"],
        "dx_background_nm": mesh_info["dx_background_nm"],
        "dx_min_nm": mesh_info["dx_min_nm"],
        "min_feature_nm": mesh_info["min_feature_nm"],
        "cells_across_feature": mesh_info["cells_across_feature"],
        "skin_depth_nm": mesh_info["skin_depth_nm"],
        "cells_per_skin_depth": mesh_info["cells_per_skin_depth"],
        "n_cells": mesh_info["n_cells"],
        "n_steps": int(recorder.step),
        "P_inj": P_inj,
        "Phi_top": Phi_top,
        "Phi_bot": Phi_bot,
        "Phi_sides": Phi_sides,
        "Phi_total_out": Phi_total_out,
        "P_abs": P_abs,
        "lee_top": lee_top,
        "lee_3d_est": lee_3d_est,
        "frac_bot": frac_bot,
        "frac_sides": frac_sides,
        "frac_abs": frac_abs,
        "mat_profiles": mat_profiles,
        "time_fs": np.asarray(recorder.time_fs, dtype=float),
        "src_trace": np.asarray(recorder.src_trace, dtype=float),
        "top_trace": np.asarray(recorder.top_trace, dtype=float),
        "energy_trace": np.asarray(recorder.energy_trace, dtype=float),
        "dft_2d": dft_2d,
        "movie_frames": recorder.movie_frames,
        "n2f": n2f_result,
        "wall_time_s": time.perf_counter() - t_wall0,
    }


def incoherent_vs_coherent_combine(
    runs: list[dict], weights: np.ndarray | None = None
) -> dict:
    """Compare physically valid incoherent power averaging vs coherent field summation.

    Spontaneous emission events at different positions ``x_i`` or orthogonal
    polarizations have random, uncorrelated optical phases ``phi_i``. Averaging
    over ``<exp(i(phi_i - phi_j))> = delta_ij`` means intensities/powers add
    (incoherent sum), whereas summing complex fields before squaring (coherent
    sum) introduces unphysical interference fringes and violates power additivity.
    """
    n_runs = len(runs)
    if weights is None:
        w = np.ones(n_runs, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)

    cum_lee = np.zeros(n_runs, dtype=float)
    cum_w = 0.0
    cum_num = 0.0
    for i, r in enumerate(runs):
        cum_w += w[i]
        cum_num += w[i] * r["lee_top"]
        cum_lee[i] = cum_num / cum_w

    I_incoh = np.zeros_like(runs[0]["dft_2d"]["I"])
    Ex_coh = np.zeros_like(runs[0]["dft_2d"]["Ex"])
    Ey_coh = np.zeros_like(runs[0]["dft_2d"]["Ey"])
    Ez_coh = np.zeros_like(runs[0]["dft_2d"]["Ez"])
    d_eta_incoh = np.zeros_like(runs[0]["n2f"]["d_eta_dtheta"])

    w_norm = w / np.sum(w)
    for i, r in enumerate(runs):
        d = r["dft_2d"]
        p_norm = r["P_inj"]
        I_incoh += w_norm[i] * (d["I"] / p_norm)
        amp = np.sqrt(w_norm[i] / p_norm)
        Ex_coh += amp * d["Ex"]
        Ey_coh += amp * d["Ey"]
        Ez_coh += amp * d["Ez"]
        if r["n2f"] is not None:
            d_eta_incoh += w_norm[i] * r["n2f"]["d_eta_dtheta"]

    I_coh = np.abs(Ex_coh) ** 2 + np.abs(Ey_coh) ** 2 + np.abs(Ez_coh) ** 2

    return {
        "n_positions": np.arange(1, n_runs + 1),
        "cum_lee_incoh": cum_lee,
        "I_incoh": I_incoh,
        "I_coh": I_coh,
        "d_eta_incoh": d_eta_incoh,
    }


# ---------------------------------------------------------------------------
# 4. Schematic & Geometry Overlay Builders for `wakis.schematic`
# ---------------------------------------------------------------------------
def build_semi_infinite_schematic_objects(
    n_slab: float = 2.60,
    cpw: float = 20.0,
    lam0: float = 280e-9,
    lx: float = 3.6e-6,
    zmin: float = -0.45e-6,
    zmax: float = 0.35e-6,
    z_dip: float = -0.18e-6,
) -> dict:
    """Build live Wakis solver, Box structures, dipole source, and FluxMonitors
    for the Task 0 semi-infinite dielectric slab setup schematic.
    """
    f0 = C0 / lam0
    freqs = np.array([f0])
    mesh_info = build_semi_infinite_mesh_2d(
        n_slab=n_slab, cpw=cpw, lam0=lam0, lx=lx, zmin=zmin, zmax=zmax, z_dip=z_dip
    )
    grid = mesh_info["grid"]
    dx = mesh_info["dx"]
    dy = mesh_info["dy"]
    nx = mesh_info["nx"]
    zmin_act = mesh_info["zmin_act"]
    zmax_act = mesh_info["zmax_act"]

    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "periodic", "cpml"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        verbose=0,
    )
    w_dom = nx * dx
    slab = Box(
        center=(0.0, 0.0, 0.5 * zmin_act),
        size=(w_dom, dy, abs(zmin_act)),
        material=n_slab**2,
    )
    slab.name = f"Dielectric slab (n = {n_slab:.2f})"
    air = Box(
        center=(0.0, 0.0, 0.5 * zmax_act),
        size=(w_dom, dy, zmax_act),
        material=1.0,
    )
    air.name = "Air superstrate (n = 1.00)"

    src_time = GaussianSource(frequency=f0, fwidth=0.20 * f0)
    dip = CurrentSource(src_time, component="x", center=(0.0, 0.0, z_dip))

    z_ext = 2.0 * dx
    z_bot = zmin_act + 9.0 * dx
    w_ext = (nx - 18) * dx
    h_box = z_ext - z_bot
    z_mid = 0.5 * (z_ext + z_bot)

    flux_top = FluxMonitor(freqs, center=(0.0, 0.0, z_ext), size=(w_ext, np.inf, 0.0))
    flux_top.name = "Top extraction FluxMonitor"
    flux_bot = FluxMonitor(freqs, center=(0.0, 0.0, z_bot), size=(w_ext, np.inf, 0.0))
    flux_bot.name = "Bottom FluxMonitor"
    flux_r = FluxMonitor(
        freqs, center=(0.5 * w_ext, 0.0, z_mid), size=(0.0, np.inf, h_box)
    )
    flux_r.name = "Side FluxMonitor"
    flux_l = FluxMonitor(
        freqs, center=(-0.5 * w_ext, 0.0, z_mid), size=(0.0, np.inf, h_box)
    )
    flux_l.name = ""

    return {
        "solver": solver,
        "grid": grid,
        "mesh_info": mesh_info,
        "structures": [slab, air],
        "material_names": [slab.name, air.name],
        "sources": [dip],
        "monitors": [flux_top, flux_bot, flux_r, flux_l],
    }


def build_ryu_schematic_objects(
    t_pgan_nm: float = 50.0,
    cpw: float = 20.0,
    dz_fine_nm: float = 3.125,
    Lx: float = 3.6e-6,
    lam0: float = 280e-9,
    dipole_pol: Literal["x", "y", "z"] = "x",
    dipole_x_um: tuple[float, ...] = (0.0,),
) -> dict:
    """Build live Wakis solver, Box layer structures, dipole sources, and
    Near2Far / Flux monitors for the Ryu (2014) planar AlGaN UV LED stack.
    """
    f0 = C0 / lam0
    freqs = np.array([f0])
    layers, z_mqw = make_ryu_planar_layers(t_pgan_nm=t_pgan_nm)
    mesh_info = build_planar_led_mesh_2d(
        layers=layers,
        z_dipole=z_mqw,
        lam0=lam0,
        cpw=cpw,
        dz_fine_nm=dz_fine_nm,
        Lx=Lx,
    )
    grid = mesh_info["grid"]
    dx = mesh_info["dx_act"]
    dy = mesh_info["dy"]
    nx = mesh_info["nx"]
    zmin_act = mesh_info["zmin_act"]
    zmax_act = mesh_info["zmax_act"]
    w_dom = nx * dx

    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "periodic", "cpml"],
        bc_high=["cpml", "periodic", "cpml"],
        n_pml=8,
        verbose=0,
    )

    label_map = {
        "sapphire": "Sapphire (n = 1.80)",
        "n_algan": "n-AlGaN (n = 2.60)",
        "mqw": "MQW (50 nm, n = 2.60)",
        "p_algan": "p-AlGaN (50 nm, n = 2.60)",
        "p_gan": f"p-GaN ({int(round(t_pgan_nm))} nm, n = 2.90, α = 1.7e5/cm)",
    }

    structures: list[Box] = []
    mat_names: list[str] = []
    for i, lay in enumerate(layers):
        z0 = zmin_act if i == 0 else lay.z_min
        z1 = lay.z_max
        if z1 <= z0:
            continue
        b = Box(
            center=(0.0, 0.0, 0.5 * (z0 + z1)),
            size=(w_dom, dy, z1 - z0),
            material=lay.n**2,
        )
        b.name = label_map.get(lay.name, lay.name)
        structures.append(b)
        mat_names.append(b.name)

    t_pgan_top = max([lay.z_max for lay in layers if lay.name == "p_gan"] + [0.0])
    air_box = Box(
        center=(0.0, 0.0, 0.5 * (t_pgan_top + zmax_act)),
        size=(w_dom, dy, zmax_act - t_pgan_top),
        material=1.0,
    )
    air_box.name = "Air (n = 1.00)"
    structures.append(air_box)
    mat_names.append(air_box.name)

    src_time = GaussianSource(frequency=f0, fwidth=0.18 * f0)
    sources = [
        CurrentSource(
            src_time, component=dipole_pol, center=(float(x_u) * 1e-6, 0.0, z_mqw)
        )
        for x_u in dipole_x_um
    ]

    z_ext = t_pgan_top + max(30e-9, 3.0 * dx)
    dz_bot_pml = float(np.max(grid.dz[:10]))
    z_bot = zmin_act + 10.0 * dz_bot_pml
    w_ext = (nx - 20) * dx
    h_box = z_ext - z_bot
    z_mid = 0.5 * (z_ext + z_bot)

    n2f_top = Near2FarMonitor(
        freqs, center=(0.0, 0.0, z_ext), size=(w_ext, np.inf, 0.0), eps_r=1.0
    )
    n2f_top.name = "Top Near2FarMonitor"
    flux_bot = FluxMonitor(
        freqs, center=(0.0, 0.0, z_bot), size=(w_ext, np.inf, 0.0)
    )
    flux_bot.name = "Bottom FluxMonitor"
    flux_r = FluxMonitor(
        freqs, center=(0.5 * w_ext, 0.0, z_mid), size=(0.0, np.inf, h_box)
    )
    flux_r.name = "Side FluxMonitor"
    flux_l = FluxMonitor(
        freqs, center=(-0.5 * w_ext, 0.0, z_mid), size=(0.0, np.inf, h_box)
    )
    flux_l.name = ""

    return {
        "solver": solver,
        "grid": grid,
        "mesh_info": mesh_info,
        "layers": layers,
        "z_mqw": z_mqw,
        "t_pgan_top": t_pgan_top,
        "z_ext": z_ext,
        "z_bot": z_bot,
        "w_ext": w_ext,
        "structures": structures,
        "material_names": mat_names,
        "sources": sources,
        "monitors": [n2f_top, flux_bot, flux_r, flux_l],
    }


def tidy_schematic_axes(
    ax,
    x_dom_half_um: float = 1.8,
    z_min_um: float = -0.88,
    z_max_um: float = 0.68,
    is_semi: bool = False,
    fontsize: float = 8.0,
) -> None:
    """Expand vertical aspect ratio and declutter overlapping internal schematic text."""
    ax.set_aspect("auto")
    ax.set_ylim(z_min_um, z_max_um)
    drop_keys = (
        "FluxMonitor",
        "boundary:",
        "Out-of-plane",
        "MQW",
        "p-AlGaN",
        "p-GaN",
        "Air (",
        "Sapphire",
        "n-AlGaN",
        "Dielectric slab",
        "Air superstrate",
    )
    src_texts = []
    for t_obj in list(ax.texts):
        s = t_obj.get_text()
        if any(k in s for k in drop_keys):
            t_obj.remove()
        elif "CurrentSource" in s:
            src_texts.append(t_obj)
    if len(src_texts) > 1:
        for t_obj in src_texts[1:]:
            t_obj.remove()
        src_texts[0].set_text("Incoherent dipoles ($J_x$)")
        src_texts[0].set_position((-0.05, -0.15))
    bbox_kw = {
        "boxstyle": "round,pad=0.14",
        "facecolor": "white",
        "edgecolor": "#555555",
        "linewidth": 0.5,
        "alpha": 0.88,
    }
    ax.text(
        -0.94 * x_dom_half_um,
        0.31 if is_semi else 0.35,
        "±x, ±z: CPML (8 cells) | ±y: Periodic",
        ha="left",
        va="center",
        fontsize=fontsize - 0.5,
        color="#222222",
        bbox=bbox_kw,
        zorder=9,
    )
    if is_semi:
        ax.text(
            -0.92 * x_dom_half_um,
            0.14,
            "Air superstrate (n = 1.00)",
            ha="left",
            va="center",
            fontsize=fontsize,
            color="#111111",
            bbox=bbox_kw,
            zorder=9,
        )
        ax.text(
            -0.92 * x_dom_half_um,
            -0.31,
            "Dielectric slab (n = 2.60)",
            ha="left",
            va="center",
            fontsize=fontsize,
            color="#111111",
            bbox=bbox_kw,
            zorder=9,
        )
        ax.text(
            0.92 * x_dom_half_um,
            -0.31,
            "4-face FluxMonitor box",
            ha="right",
            va="center",
            fontsize=fontsize - 0.3,
            color="#0B6E4F",
            bbox=bbox_kw,
            zorder=9,
        )


def annotate_thin_led_layers(
    ax,
    t_pgan_nm: float,
    x_dom_half_um: float = 1.8,
    text_color: str = "#111111",
    box_alpha: float = 0.82,
    fontsize: float = 6.8,
    tidy_schematic: bool = False,
) -> None:
    """Place non-overlapping callout labels for the thin active/cladding/contact
    layers (MQW, p-AlGaN, p-GaN) and thick regions (Sapphire, n-AlGaN, Air) on an
    (x, z) plot in micrometres.
    """
    if tidy_schematic:
        tidy_schematic_axes(
            ax,
            x_dom_half_um=x_dom_half_um,
            z_min_um=-0.88,
            z_max_um=0.68,
            is_semi=False,
            fontsize=fontsize,
        )
    bbox_kw = {
        "boxstyle": "round,pad=0.16",
        "facecolor": "white",
        "edgecolor": "#555555",
        "linewidth": 0.5,
        "alpha": box_alpha,
    }
    x_left = -0.92 * x_dom_half_um
    x_right = 0.92 * x_dom_half_um
    t_um = max(0.0, float(t_pgan_nm)) * 1e-3

    # Thick regions on left side, well clear of the central dipole beam and CPML labels
    ax.text(
        x_left,
        0.26,
        "Air (n = 1.00)",
        ha="left",
        va="center",
        fontsize=fontsize,
        color=text_color,
        bbox=bbox_kw,
        zorder=9,
    )
    ax.text(
        x_left,
        -0.33,
        "n-AlGaN (n = 2.60)",
        ha="left",
        va="center",
        fontsize=fontsize,
        color=text_color,
        bbox=bbox_kw,
        zorder=9,
    )
    ax.text(
        x_left,
        -0.68,
        "Sapphire (n = 1.80)",
        ha="left",
        va="center",
        fontsize=fontsize,
        color=text_color,
        bbox=bbox_kw,
        zorder=9,
    )

    # Thin layers staggered on the right side with short leader lines
    thin_specs = [
        (-0.075, -0.22, x_right, "MQW (50 nm, n=2.60)"),
        (-0.025, -0.04, x_right, "p-AlGaN (50 nm)"),
    ]
    if t_pgan_nm > 0.0:
        thin_specs.append(
            (0.5 * t_um, max(0.14, t_um + 0.07), x_right, f"p-GaN ({int(round(t_pgan_nm))} nm, α=1.7e5/cm)")
        )

    for z_true, z_lbl, x_lbl, txt in thin_specs:
        ax.annotate(
            txt,
            xy=(x_lbl - 0.35, z_true),
            xytext=(x_lbl, z_lbl),
            ha="right",
            va="center",
            fontsize=fontsize,
            color=text_color,
            bbox=bbox_kw,
            arrowprops={
                "arrowstyle": "->",
                "color": text_color,
                "lw": 0.75,
                "shrinkA": 1,
                "shrinkB": 1,
            },
            zorder=9,
        )


def overlay_ryu_stack(
    ax,
    t_pgan_nm: float = 50.0,
    Lx_um: float = 3.6,
    dipole_pol: Literal["x", "y", "z"] = "x",
    dipole_x_um: tuple[float, ...] = (0.0,),
    line_color: str = "white",
    show_monitors: bool = True,
    show_escape_cone: bool = False,
    fontsize: float = 6.6,
) -> None:
    """Overlay full Ryu (2014) planar LED geometry on a 2D (x, z) field map or GIF frame.

    Calls `wakis.schematic.overlay_geometry` to draw all layer boundaries and
    CPML boundary shading/labels, then annotates the layer names (`Air`,
    `p-GaN`, `p-AlGaN`, `MQW`, `n-AlGaN`, `Sapphire`), dipole positions and
    orientation (`TE Jx` or `TM Jz`), and flux/Near2Far monitor lines.
    """
    from wakis import schematic

    obj = build_ryu_schematic_objects(
        t_pgan_nm=t_pgan_nm,
        cpw=12.0,
        dz_fine_nm=6.25,
        Lx=Lx_um * 1e-6,
        dipole_pol=dipole_pol,
        dipole_x_um=dipole_x_um,
    )
    schematic.overlay_geometry(
        ax,
        obj["solver"],
        plane="xz",
        unit="um",
        structures=obj["structures"],
        show_boundaries=True,
        show_materials=True,
        outline_color=line_color,
        label_materials=False,
    )

    x_half = 0.5 * Lx_um
    annotate_thin_led_layers(
        ax,
        t_pgan_nm=t_pgan_nm,
        x_dom_half_um=x_half,
        text_color="#111111",
        box_alpha=0.82,
        fontsize=fontsize,
    )

    # Monitors
    if show_monitors:
        z_ext_um = obj["z_ext"] * 1e6
        z_bot_um = obj["z_bot"] * 1e6
        w_half_um = 0.5 * obj["w_ext"] * 1e6
        ax.plot(
            [-w_half_um, w_half_um],
            [z_ext_um, z_ext_um],
            ls="-.",
            lw=1.1,
            color="#2ca25f",
            alpha=0.9,
            zorder=8,
        )
        ax.text(
            -w_half_um + 0.06,
            z_ext_um + 0.035,
            "Near2Far monitor",
            fontsize=fontsize - 0.3,
            color="#1b7837",
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=9,
        )
        ax.plot(
            [-w_half_um, w_half_um],
            [z_bot_um, z_bot_um],
            ls=":",
            lw=1.0,
            color="#756bb1",
            alpha=0.9,
            zorder=8,
        )

    # Escape cone
    if show_escape_cone:
        tc_rad = escape_cone_angle(2.60, 1.0)
        z_dip_um = -0.075
        z_cone_top = 0.36
        dx_cone = (z_cone_top - z_dip_um) * np.tan(tc_rad)
        for sgn in (-1.0, 1.0):
            ax.plot(
                [0.0, sgn * dx_cone],
                [z_dip_um, z_cone_top],
                ls=":",
                lw=1.2,
                color="#ffd92f" if line_color == "white" else "#d95f02",
                alpha=0.9,
                zorder=8,
            )
        ax.text(
            dx_cone + 0.04,
            0.29,
            f"θc = {np.degrees(tc_rad):.1f}°",
            fontsize=fontsize - 0.2,
            color="#111111",
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "#fff7bc",
                "edgecolor": "#d95f02",
                "linewidth": 0.5,
                "alpha": 0.85,
            },
            zorder=9,
        )

    # Dipole markers & polarization arrow
    z_dip_um = -0.075
    xs = [float(x) for x in dipole_x_um]
    if len(xs) > 1:
        # Mirror symmetric pairs for display if all non-negative
        all_xs = sorted({round(v, 3) for x in xs for v in (-x, x)})
    else:
        all_xs = xs
    ax.plot(
        all_xs,
        [z_dip_um] * len(all_xs),
        "o",
        color="#e41a1c",
         markeredgecolor="white",
        markeredgewidth=0.7,
        ms=5.2 if len(all_xs) == 1 else 4.2,
        zorder=10,
    )
    pol_lbl = (
        "TE dipole Jx"
        if dipole_pol == "x"
        else ("TM dipole Jz" if dipole_pol == "z" else "Dipole Jy")
    )
    if len(all_xs) > 1:
        pol_lbl = f"{len(all_xs)}× {pol_lbl}"
    ax.text(
        0.0,
        z_dip_um - 0.085,
        pol_lbl,
        ha="center",
        va="top",
        fontsize=fontsize - 0.2,
        fontweight="bold",
        color="#b30000",
        bbox={
            "boxstyle": "round,pad=0.13",
            "facecolor": "white",
            "edgecolor": "#b30000",
            "linewidth": 0.5,
            "alpha": 0.85,
        },
        zorder=10,
    )


class _RepCoord(np.ndarray):
    """Coordinate array that plots all representative ensemble markers while returning
    its scalar tag coordinate for comparisons and arithmetic offsets.
    """

    def __new__(cls, vals, tag_pos: float | None = None):
        obj = np.asarray(vals, dtype=float).view(cls)
        obj._tag_pos = None if tag_pos is None else float(tag_pos)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._tag_pos = getattr(obj, "_tag_pos", None)

    def __float__(self) -> float:
        if getattr(self, "_tag_pos", None) is not None:
            return float(self._tag_pos)
        return float(np.asarray(self).flat[0])

    def __mul__(self, other):
        res = np.asarray(self, dtype=float) * float(other)
        tp = None if getattr(self, "_tag_pos", None) is None else self._tag_pos * float(other)
        return _RepCoord(res, tag_pos=tp)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __lt__(self, other):
        return float(self) < float(other)

    def __gt__(self, other):
        return float(self) > float(other)

    def __add__(self, other):
        return float(self) + float(other)

    def __radd__(self, other):
        return float(other) + float(self)

    def __sub__(self, other):
        return float(self) - float(other)


def save_planar_led_setup_figures(
    fig_dir: Path,
    mode: str = "full",
) -> dict:
    """Render ``00a_setup_geometry.png``, ``00b_setup_mesh_and_excitation.png``, and
    ``00_setup_table.md`` for the Ryu (2014) planar AlGaN deep-UV LED stack via
    `wakis.schematic.save_setup_figures`.
    """
    from wakis import schematic

    cpw = 20.0 if mode == "full" else 12.0
    dz_fine_nm = 3.125 if mode == "full" else 6.25
    Lx = 3.6e-6 if mode == "full" else 2.8e-6

    obj = build_ryu_schematic_objects(
        t_pgan_nm=50.0,
        cpw=cpw,
        dz_fine_nm=dz_fine_nm,
        Lx=Lx,
        dipole_pol="x",
        dipole_x_um=(0.0, 0.25, 0.50, 0.75, 1.05),
    )
    solver = obj["solver"]
    structures = obj["structures"][:5]
    layer_labels = [
        "Sapphire (d_z = 300 nm, n = 1.80)",
        "n-AlGaN (d_z = 450 nm, n = 2.60)",
        "MQW active (d_z = 50 nm, n = 2.60, α = 10³ cm⁻¹)",
        "p-AlGaN (d_z = 50 nm, n = 2.60)",
        "p-GaN absorber (d_z = 50 nm, n = 2.90, α = 1.7×10⁵ cm⁻¹)",
    ]
    for st, lbl in zip(structures, layer_labels):
        st.name = lbl

    info0 = schematic.extract_setup_info(
        solver,
        plane="xz",
        structures=structures,
        sources=obj["sources"],
        monitors=obj["monitors"],
    )
    src_base = info0["sources"][0]

    rep_x = _RepCoord([0.0, -1.05e-6, -0.50e-6, 0.50e-6, 1.05e-6])
    rep_z = _RepCoord([-0.075e-6] * 5, tag_pos=-0.34e-6)

    src_ens = dict(src_base)
    src_ens["h_bounds"] = (rep_x, 0.0)
    src_ens["v_bounds"] = (rep_z, -0.075e-6)
    src_ens["short_word"] = "dipoles (9×2)"
    src_ens["label"] = (
        "MQW dipole ensemble: 9 sites × 2 pols (TE J_x, TM J_z) = 18 dipoles"
    )
    src_ens["desc"] = (
        "MQW dipole ensemble: 9 sites in [-1.05, 1.05] µm × 2 pols "
        "(TE J_x, TM J_z) = 18 dipoles (z = -75 nm)"
    )
    src_ens["polarization"] = "TE (J_x) & TM (J_z)"
    src_ens["pos_str"] = "z = -75 nm, 9 sites in x ∈ [-1.05, 1.05] µm (5 shown)"

    z_ext_nm = obj["z_ext"] * 1e9
    w_half_um = 0.5 * obj["w_ext"] * 1e6
    z_bot_um = obj["z_bot"] * 1e6

    mon_n2f = dict(info0["monitors"][0])
    mon_n2f["kind"] = "plane"
    mon_n2f["h_bounds"] = (-0.5 * obj["w_ext"], 0.5 * obj["w_ext"])
    mon_n2f["v_bounds"] = (0.21e-6, 0.21e-6)
    mon_n2f["short_word"] = "N2F top"
    mon_n2f["label"] = f"Top Near2FarMonitor in air (z = +{z_ext_nm:.0f} nm)"
    mon_n2f["desc"] = (
        f"Top Near2FarMonitor in air (z = +{z_ext_nm:.0f} nm, "
        f"x ∈ [-{w_half_um:.2f}, {w_half_um:.2f}] µm)"
    )
    mon_n2f["pos_str"] = (
        f"z = +{z_ext_nm:.0f} nm, x ∈ [-{w_half_um:.2f}, {w_half_um:.2f}] µm"
    )

    mon_bot = dict(info0["monitors"][1])
    mon_bot["kind"] = "plane"
    mon_bot["short_word"] = "substrate flux"
    mon_bot["label"] = (
        f"Bottom + lateral Poynting FluxMonitors (z_bot = {z_bot_um:.2f} µm)"
    )
    mon_bot["desc"] = (
        f"Bottom + lateral Poynting FluxMonitor box "
        f"(z_bot = {z_bot_um:.2f} µm, x = ±{w_half_um:.2f} µm)"
    )

    variations = [
        {
            "name": "Task 0 — Semi-infinite dielectric slab validation",
            "notes": (
                "n_slab ∈ {1.50, 2.00, 2.60} → Air (n = 1.00), z_dip = -180 nm, "
                "4-face FluxMonitor box vs exact 2D Fresnel integral"
            ),
        },
        {
            "name": "Task 1 — p-GaN thickness & AlGaN index sweeps (Ryu 2014 Figs. 3 & 7)",
            "notes": (
                "t_p-GaN ∈ {0, 25, 50, 100, 150, 200} nm (α = 1.7×10⁵ cm⁻¹, "
                "δ = 58.82 nm), n_AlGaN ∈ {2.50, 2.60, 2.70}"
            ),
        },
        {
            "name": "Task 2 — Incoherent dipole ensemble & Near2Far pattern",
            "notes": (
                "N_pos = 5 lateral positions x_i ∈ {0.0, 0.25, 0.50, 0.75, 1.05} µm "
                "(9 symmetric sites) × TE (J_x) / TM (J_z) = 18 dipoles"
            ),
        },
        {
            "name": "Tasks 3–6 — Kottke subpixel smoothing vs staircase convergence ladder",
            "notes": (
                "cpw ∈ {10 (demo, Δz = 25 nm), 14 (Δz = 4.17 nm), 20 (accepted, "
                "Δz = 3.125 nm, 18.8 cells/δ), 28 (reference, Δz = 2.50 nm, 23.5 cells/δ)}"
            ),
        },
    ]

    return schematic.save_setup_figures(
        solver,
        out_dir=fig_dir,
        plane="xz",
        unit="um",
        structures=structures,
        material_names={"background": "Air superstrate"},
        sources=[src_ens],
        monitors=[mon_n2f, mon_bot],
        wavelength=280e-9,
        min_feature=25e-9,
        title_geometry=(
            "Planar AlGaN Deep-UV LED (Ryu 2014 Stack): "
            "Geometry, Dipole Ensemble & Monitors"
        ),
        title_mesh=(
            "Planar AlGaN Deep-UV LED: Graded Thin-Layer Mesh & UV Dipole Excitation"
        ),
        variations=variations,
    )


