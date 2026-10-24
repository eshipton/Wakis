# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Physics, multi-layer RCWA, conformal subpixel averaging, and Wakis Floquet
helpers for Tutorial 6.5: Slanted / blazed in-coupler gratings.

Grounded in verified literature:
1. H. Wang and D. Ma, "Surface relief grating near-eye display waveguide
   design," arXiv:2307.01600 (2023), DOI: 10.48550/arXiv.2307.01600
   (Table 1: slant angle 25 deg, fill factor 66%, depth 415 nm, lambda=525 nm,
   n_wg=1.74).
2. T. Levola, "Diffractive optics for virtual reality displays," J. Soc. Inf.
   Display 14(5), 467-475 (2006), DOI: 10.1889/1.2206112; T. Levola and
   P. Laakkonen, "Replicated slanted gratings with a high refractive index
   coating for efficient compact coverage of the eye-box," Opt. Express 15(5),
   2067-2074 (2007), DOI: 10.1364/OE.15.002067.
3. A. Farjadpour et al., "Improving accuracy by subpixel smoothing in the
   finite-difference time domain," Opt. Lett. 31(20), 2972-2974 (2006);
   N. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
   anisotropic dielectric interfaces, and application to subpixel smoothing of
   discretized numerical methods," Phys. Rev. E 77, 036611 (2008).
"""

from __future__ import annotations

from collections.abc import Sequence
import time
from typing import Any

import numpy as np
from scipy.constants import c, epsilon_0

import wakis
from wakis.floquet import (
    AngleSweep,
    FloquetCell,
    FloquetMode,
    _run_single_Solver,
    grating_angles,
    rcwa_1d_grating,
    s_parameters_from_ports,
)
from wakis.sources import CurrentSource, GaussianSource

# ---------------------------------------------------------------------------
# Benchmark physical parameters (Wang & Ma arXiv:2307.01600, Table 1; Levola 2007)
# ---------------------------------------------------------------------------
WL_DESIGN_M = 525.0e-9  # Design green wavelength [m] (Wang & Ma, 2023)
PERIOD_INCOUPLER_M = 400.0e-9  # Grating period Lambda = 400 nm (couples m=-1 into TIR)
N_SUBSTRATE = 1.74  # N-LAF2 waveguide substrate index (Wang & Ma, 2023)
N_RIDGE = 2.35  # High-index TiO2 / ZrO2-composite ridge index
N_GROOVE = 1.00  # Air groove index
N_AIR = 1.00

# Wang & Ma (2023) Table 1 In-Coupler parameters:
WANG_SLANT_DEG = 25.0
WANG_FILL_FACTOR = 0.66
WANG_DEPTH_M = 415.0e-9
DEFAULT_DEPTH_M = 320.0e-9


def bragg_slant_angle_deg(
    theta_inc_deg: float,
    wavelength: float = WL_DESIGN_M,
    period: float = PERIOD_INCOUPLER_M,
    n_inc: float = N_AIR,
    n_sub: float = N_SUBSTRATE,
    order: int = -1,
) -> float:
    """Compute the geometric Bragg blaze/slant angle [deg] bisecting incident & diffracted rays.

    For an incident wave at angle ``theta_inc`` in medium ``n_inc`` and the ``m``-th
    transmitted order at angle ``theta_m`` inside ``n_sub``, the groove wall normal
    that specularly reflects the incident phase front into order ``m`` has slant
    angle ``alpha_Bragg`` with respect to the ``z``-axis given by:

    .. math::

        \\alpha_{\\mathrm{Bragg}} = \\frac{1}{2}\\left(|\\theta_m| - \\theta_{\\mathrm{inc}}\\right).
    """
    th_in = np.deg2rad(theta_inc_deg)
    sin_m = (n_inc * np.sin(th_in) + order * (wavelength / period)) / n_sub
    sin_m = float(np.clip(sin_m, -0.999, 0.999))
    th_m = float(np.arcsin(sin_m))
    return float(0.5 * np.rad2deg(abs(th_m) - th_in))


def rcwa_multilayer_1d(
    frequencies: Sequence[float] | np.ndarray,
    theta_inc: float,
    period: float,
    height: float,
    profile_type: str = "slanted",
    slant_deg: float = 25.0,
    fill_factor: float = 0.55,
    eps_ridge: float = N_RIDGE**2,
    eps_groove: float = 1.0,
    eps_below: float = 1.0,
    eps_above: float = N_SUBSTRATE**2,
    pol: str = "TE",
    n_harmonics: int = 12,
    n_slices: int = 28,
    n_x_fft: int = 1024,
    orders: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Multi-layer 1D Fourier Modal Method (RCWA) for binary, slanted, and blazed gratings.

    Uses enhanced transmittance-matrix recursion across ``n_slices`` staircase-free
    Fourier slices, matching :func:`wakis.floquet.rcwa_1d_grating` to machine
    precision for ``n_slices=1`` binary gratings and resolving arbitrary slanted
    or sawtooth blazed grooves.

    Parameters
    ----------
    frequencies : sequence of float
        Frequencies [Hz].
    theta_inc : float
        Incident polar angle [rad] in medium ``eps_below``.
    period : float
        Grating period ``Lambda`` along ``x`` [m].
    height : float
        Total grating groove depth ``d`` along ``z`` [m].
    profile_type : {'binary', 'slanted', 'blazed'}
        Tooth geometry:
        - ``'binary'``: vertical walls (``slant_deg = 0``), fill factor ``fill_factor``.
        - ``'slanted'``: parallelogram walls tilted by ``slant_deg`` from the ``z``-axis.
        - ``'blazed'``: right-angled triangular sawtooth profile of peak height ``height``.
    slant_deg : float, optional
        Slant angle ``alpha`` [deg] for ``profile_type='slanted'`` (default 25.0).
    fill_factor : float, optional
        Ridge duty cycle ``f`` in ``(0, 1)`` (default 0.55).
    eps_ridge, eps_groove, eps_below, eps_above : float
        Relative permittivities of ridge, groove, incident half-space, and substrate.
    pol : {'TE', 'TM'}
        Polarization state.
    n_harmonics : int, optional
        Truncation order ``M`` (total ``2*M + 1`` harmonics, default 12).
    n_slices : int, optional
        Number of horizontal RCWA slices through ``[0, height]`` (default 28).
    n_x_fft : int, optional
        Spatial sampling points per period for Fourier transform (default 1024).
    orders : sequence of int, optional
        Specific diffraction orders to return in ``R_orders`` and ``T_orders``.
    """
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    pol_u = pol.upper().strip()
    if pol_u in ("S", "TE"):
        pol_u = "TE"
    elif pol_u in ("P", "TM"):
        pol_u = "TM"
    else:
        raise ValueError(f"Unsupported polarization '{pol}'.")

    ptype = profile_type.lower().strip()
    if ptype == "binary":
        slant_rad = 0.0
        n_slices_eff = 1
    elif ptype == "slanted":
        slant_rad = float(np.deg2rad(slant_deg))
        n_slices_eff = 1 if abs(slant_deg) < 1e-9 else max(8, int(n_slices))
    elif ptype == "blazed":
        slant_rad = 0.0
        n_slices_eff = max(12, int(n_slices))
    else:
        raise ValueError(f"Unknown profile_type '{profile_type}'.")

    M = int(n_harmonics)
    N = 2 * M + 1
    m_vec = np.arange(-M, M + 1, dtype=int)
    G = 2.0 * np.pi / float(period)
    dz = float(height) / n_slices_eff
    z_centers = (np.arange(n_slices_eff) + 0.5) * dz
    x_grid = (np.arange(n_x_fft) + 0.5) / n_x_fft * float(period)

    # Build Toeplitz permittivity and inverse-permittivity matrices for each slice
    # Wakis spatial convention exp(-i*(k_x0 + m*G)*x) requires idx_diff = n - m
    idx_diff = np.arange(N)[None, :] - np.arange(N)[:, None]
    E_slices = []
    invE_slices = []
    for zc in z_centers:
        if ptype in ("binary", "slanted"):
            u = np.mod(
                (x_grid + zc * np.tan(slant_rad)) / float(period)
                + 0.5 * float(fill_factor),
                1.0,
            )
            in_ridge = u < float(fill_factor)
        else:
            # Sawtooth blazed grating: local fill at height zc is f * (1 - zc / height)
            u = np.mod(x_grid / float(period) + 0.5 * float(fill_factor), 1.0)
            local_fill = float(fill_factor) * (1.0 - zc / float(height))
            in_ridge = u < local_fill

        eps_x = np.where(in_ridge, float(eps_ridge), float(eps_groove))
        inv_eps_x = 1.0 / eps_x
        fft_eps = np.fft.fft(eps_x) / n_x_fft
        fft_inv = np.fft.fft(inv_eps_x) / n_x_fft
        E_slices.append(fft_eps[idx_diff % n_x_fft])
        invE_slices.append(fft_inv[idx_diff % n_x_fft])

    if orders is None:
        out_orders = m_vec.copy()
    else:
        out_orders = np.asarray(orders, dtype=int)

    n_inc = float(np.sqrt(eps_below))
    R_orders = np.full((freqs.size, out_orders.size), np.nan, dtype=float)
    T_orders = np.full((freqs.size, out_orders.size), np.nan, dtype=float)
    R_total = np.zeros(freqs.size, dtype=float)
    T_total = np.zeros(freqs.size, dtype=float)
    I_N = np.eye(N, dtype=complex)

    for jf, f in enumerate(freqs):
        k0 = 2.0 * np.pi * f / c
        kx_inc = k0 * n_inc * np.sin(theta_inc)
        kx_m = kx_inc + m_vec * G
        Kx = np.diag(kx_m / k0)

        kz1_sq = complex(eps_below) - (kx_m / k0) ** 2
        kz2_sq = complex(eps_above) - (kx_m / k0) ** 2
        kz1 = np.where(np.real(kz1_sq) >= 0.0, np.sqrt(kz1_sq + 0j), -1j * np.sqrt(-kz1_sq + 0j))
        kz2 = np.where(np.real(kz2_sq) >= 0.0, np.sqrt(kz2_sq + 0j), -1j * np.sqrt(-kz2_sq + 0j))

        if pol_u == "TE":
            Y1 = np.diag(1j * kz1)
            Y2 = np.diag(1j * kz2)
        else:
            Y1 = np.diag(1j * kz1 / eps_below)
            Y2 = np.diag(1j * kz2 / eps_above)

        G_prev = Y2.copy()
        layer_ops = []

        # Propagate impedance G_prev from top of grating (z=height, eps_above) to bottom (z=0, eps_below)
        for s_idx in range(n_slices_eff - 1, -1, -1):
            E_mat = E_slices[s_idx]
            invE_mat = invE_slices[s_idx]
            if pol_u == "TE":
                A_eig = Kx @ Kx - E_mat
                eigvals, W = np.linalg.eig(A_eig)
                q = np.sqrt(eigvals + 0j)
                q = np.where(np.real(q) + np.imag(q) < 0.0, -q, q)
                V = W @ np.diag(q)
            else:
                E_inv_rule = np.linalg.inv(invE_mat)
                B_mat = Kx @ np.linalg.solve(E_mat, Kx) - I_N
                A_eig = E_inv_rule @ B_mat
                eigvals, W = np.linalg.eig(A_eig)
                q = np.sqrt(eigvals + 0j)
                q = np.where(np.real(q) + np.imag(q) < 0.0, -q, q)
                V = np.linalg.solve(E_inv_rule, W) @ np.diag(q)

            X = np.diag(np.exp(-k0 * q * dz))
            S_top = np.linalg.solve(V + G_prev @ W, V - G_prev @ W)
            M1 = W @ (I_N + X @ S_top @ X)
            M2 = V @ (I_N - X @ S_top @ X)
            G_prev = np.linalg.solve(M1.T, M2.T).T
            layer_ops.append((W, X, S_top, M1, M2))

        layer_ops.reverse()
        _, _, _, M1_0, M2_0 = layer_ops[0]
        delta_0 = np.zeros(N, dtype=complex)
        delta_0[M] = 1.0
        rhs = 2.0 * Y1 @ delta_0
        f_0 = np.linalg.solve(Y1 @ M1_0 + M2_0, rhs)
        E_curr = M1_0 @ f_0
        r_vec = E_curr - delta_0

        for s_idx, (W_s, X_s, S_top_s, M1_s, _) in enumerate(layer_ops):
            f_s = f_0 if s_idx == 0 else np.linalg.solve(M1_s, E_curr)
            E_curr = W_s @ (X_s + S_top_s @ X_s) @ f_s
        t_vec = E_curr

        kz1_0 = float(np.real(kz1[M]))
        if kz1_0 <= 0.0:
            continue

        if pol_u == "TE":
            R_m = np.where(np.real(kz1_sq) > 0.0, (np.real(kz1) / kz1_0) * np.abs(r_vec) ** 2, np.nan)
            T_m = np.where(np.real(kz2_sq) > 0.0, (np.real(kz2) / kz1_0) * np.abs(t_vec) ** 2, np.nan)
        else:
            R_m = np.where(np.real(kz1_sq) > 0.0, (np.real(kz1) / kz1_0) * np.abs(r_vec) ** 2, np.nan)
            T_m = np.where(
                np.real(kz2_sq) > 0.0,
                (np.real(kz2 / eps_above) / (kz1_0 / eps_below)) * np.abs(t_vec) ** 2,
                np.nan,
            )

        R_total[jf] = float(np.nansum(R_m))
        T_total[jf] = float(np.nansum(T_m))
        for io, m_ord in enumerate(out_orders):
            idx = int(m_ord) + M
            if 0 <= idx < N:
                R_orders[jf, io] = R_m[idx]
                T_orders[jf, io] = T_m[idx]

    return {
        "orders": out_orders,
        "R_orders": R_orders,
        "T_orders": T_orders,
        "R_total": R_total,
        "T_total": T_total,
    }


class SlantedBlazedGratingCallback:
    """Picklable ``material_callback(solver)`` with Kottke conformal or staircase treatment.

    Implements both staircase point-sampling (``averaging='staircase'``) and
    Kottke/Farjadpour subpixel anisotropic permittivity smoothing
    (``averaging='kottke'``) using the exact local sidewall normal
    ``n_hat = (cos(alpha), 0, -sin(alpha))``:

    .. math::

        (\\tilde{\\varepsilon}^{-1})_{xx} = \\frac{\\sin^2\\alpha}{\\langle\\varepsilon\\rangle}
        + \\cos^2\\alpha\\,\\langle\\varepsilon^{-1}\\rangle, \\qquad
        (\\tilde{\\varepsilon}^{-1})_{zz} = \\frac{\\cos^2\\alpha}{\\langle\\varepsilon\\rangle}
        + \\sin^2\\alpha\\,\\langle\\varepsilon^{-1}\\rangle, \\qquad
        (\\tilde{\\varepsilon}^{-1})_{yy} = \\frac{1}{\\langle\\varepsilon\\rangle}.
    """

    def __init__(
        self,
        period: float,
        height: float,
        profile_type: str = "slanted",
        slant_deg: float = 25.0,
        fill_factor: float = 0.55,
        eps_ridge: float = N_RIDGE**2,
        eps_groove: float = 1.0,
        eps_below: float = 1.0,
        eps_above: float = N_SUBSTRATE**2,
        averaging: str = "kottke",
        n_sub: int = 16,
    ):
        self.period = float(period)
        self.height = float(height)
        self.profile_type = profile_type.lower().strip()
        self.slant_deg = float(slant_deg)
        self.slant_rad = 0.0 if self.profile_type == "binary" else float(np.deg2rad(slant_deg))
        self.fill_factor = float(fill_factor)
        self.eps_ridge = float(eps_ridge)
        self.eps_groove = float(eps_groove)
        self.eps_below = float(eps_below)
        self.eps_above = float(eps_above)
        self.averaging = averaging.lower().strip()
        self.n_sub = 1 if self.averaging == "staircase" else int(n_sub)

        if self.profile_type == "blazed":
            # Sawtooth hypotenuse normal angle
            hyp_angle = np.arctan2(self.height, self.fill_factor * self.period)
            self.nx_wall = float(np.sin(hyp_angle))
            self.nz_wall = float(np.cos(hyp_angle))
        else:
            self.nx_wall = float(np.cos(self.slant_rad))
            self.nz_wall = float(np.sin(self.slant_rad))

    def _point_eps(self, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
        in_above = zs >= self.height
        in_below = zs < 0.0
        in_gr = (~in_above) & (~in_below)
        if self.profile_type in ("binary", "slanted"):
            u = np.mod(
                (xs + zs * np.tan(self.slant_rad)) / self.period
                + 0.5 * self.fill_factor,
                1.0,
            )
            in_ridge = in_gr & (u < self.fill_factor)
        else:
            u = np.mod(xs / self.period + 0.5 * self.fill_factor, 1.0)
            local_fill = self.fill_factor * (1.0 - zs / self.height)
            in_ridge = in_gr & (u < local_fill)

        return np.where(
            in_above,
            self.eps_above,
            np.where(in_below, self.eps_below, np.where(in_ridge, self.eps_ridge, self.eps_groove)),
        )

    def __call__(self, solver) -> None:
        grid = solver.grid
        nodes = (grid.x, grid.y, grid.z)
        sub = (np.arange(self.n_sub) + 0.5) / self.n_sub - 0.5
        ox, oz = np.meshgrid(sub, sub, indexing="ij")
        ox, oz = ox.ravel(), oz.ravel()

        nx2 = self.nx_wall**2
        nz2 = self.nz_wall**2

        for ci, comp in enumerate("xyz"):
            pos = [
                (n[:-1] + n[1:]) / 2 if d == ci else n[:-1]
                for d, n in enumerate(nodes)
            ]
            dx_c = (
                (grid.x[1:] - grid.x[:-1])
                if ci == 0
                else np.full(grid.Nx, float(grid.dx[0]))
            )
            dz_c = (
                (grid.z[1:] - grid.z[:-1])
                if ci == 2
                else np.full(grid.Nz, float(grid.dz[0]))
            )
            X, _, Z = np.meshgrid(*pos, indexing="ij")
            DX, _, DZ = np.meshgrid(dx_c, [float(grid.dy[0])], dz_c, indexing="ij")

            xs = X[..., None] + DX[..., None] * ox
            zs = Z[..., None] + DZ[..., None] * oz
            eps_s = self._point_eps(xs, zs)

            if self.averaging == "staircase":
                inv_eps = 1.0 / eps_s[..., 0]
            else:
                eps_arith = np.mean(eps_s, axis=-1)
                inv_eps_harm = np.mean(1.0 / eps_s, axis=-1)
                is_horiz = (np.abs(Z) < DZ) | (np.abs(Z - self.height) < DZ)

                if ci == 0:
                    inv_wall = nz2 * (1.0 / eps_arith) + nx2 * inv_eps_harm
                    inv_eps = np.where(is_horiz, 1.0 / eps_arith, inv_wall)
                elif ci == 2:
                    inv_wall = nx2 * (1.0 / eps_arith) + nz2 * inv_eps_harm
                    inv_eps = np.where(is_horiz, inv_eps_harm, inv_wall)
                else:
                    inv_eps = 1.0 / eps_arith

            solver.ieps[:, :, :, comp] = inv_eps / epsilon_0


def build_slanted_floquet_cell(
    period: float = PERIOD_INCOUPLER_M,
    height: float = DEFAULT_DEPTH_M,
    wavelength_min: float = WL_DESIGN_M,
    profile_type: str = "slanted",
    slant_deg: float = 25.0,
    fill_factor: float = 0.55,
    eps_ridge: float = N_RIDGE**2,
    eps_groove: float = 1.0,
    eps_below: float = 1.0,
    eps_above: float = N_SUBSTRATE**2,
    averaging: str = "kottke",
    orders_x: Sequence[int] = (-2, -1, 0, 1, 2),
    cpw: int = 24,
    min_cells_per_feature: int = 4,
    n_pml: int = 14,
    run_time_cycles: float = 28.0,
) -> tuple[FloquetCell, dict[str, float]]:
    """Construct a 2D :class:`~wakis.floquet.FloquetCell` for a binary/slanted/blazed grating."""
    n_max = float(np.sqrt(max(eps_ridge, eps_groove, eps_below, eps_above)))
    lam_min_mat = float(wavelength_min) / n_max
    dx_cpw = lam_min_mat / float(cpw)

    min_w_x = float(period) * min(fill_factor, max(1.0 - fill_factor, 0.15))
    min_feat = min(min_w_x, float(height))
    dx_feat = min_feat / float(max(min_cells_per_feature, 1))

    dx_target = min(dx_cpw, dx_feat)
    nx = max(18, int(np.ceil(float(period) / dx_target)))
    dx = float(period) / nx

    dz_target = min(dx_cpw, float(height) / float(max(min_cells_per_feature, 6)))
    n_grating_z = max(6, int(np.ceil(float(height) / dz_target)))
    dz = float(height) / n_grating_z

    pad_below = max(1.15 * float(wavelength_min), 24 * dz)
    pad_above = max(1.15 * float(wavelength_min), 24 * dz)
    n_below = int(np.ceil(pad_below / dz))
    n_above = int(np.ceil(pad_above / dz))
    zmin = -n_below * dz
    zmax = float(height) + n_above * dz
    nz = n_below + n_grating_z + n_above

    z_source = -0.62 * pad_below
    z_refl = -0.32 * pad_below
    z_tran = float(height) + 0.36 * pad_above

    cb = SlantedBlazedGratingCallback(
        period=period,
        height=height,
        profile_type=profile_type,
        slant_deg=slant_deg,
        fill_factor=fill_factor,
        eps_ridge=eps_ridge,
        eps_groove=eps_groove,
        eps_below=eps_below,
        eps_above=eps_above,
        averaging=averaging,
    )
    cell = FloquetCell(
        Lx=float(period),
        Ly=dz,
        zmin=zmin,
        zmax=zmax,
        nx=nx,
        ny=1,
        nz=nz,
        z_source=z_source,
        z_refl=z_refl,
        z_tran=z_tran,
        z_ref_below=0.0,
        z_ref_above=float(height),
        eps_below=eps_below,
        eps_above=eps_above,
        orders_x=tuple(orders_x),
        orders_y=(0,),
        n_pml=n_pml,
        material_callback=cb,
        run_time_cycles=run_time_cycles,
    )
    mesh_info = {
        "nx": int(nx),
        "nz": int(nz),
        "n_cells_2d": int(nx * nz),
        "dx_nm": float(dx * 1e9),
        "dz_nm": float(dz * 1e9),
        "n_max": n_max,
        "cpw_x": float(lam_min_mat / dx),
        "cpw_z": float(lam_min_mat / dz),
        "min_feature_nm": float(min_feat * 1e9),
        "cells_per_min_feature_x": float(min_w_x / dx),
        "cells_per_grating_height_z": float(height / dz),
    }
    return cell, mesh_info


def run_single_grating_orders(
    cell: FloquetCell,
    frequency: float,
    theta_inc_rad: float = 0.0,
    polarization: str = "TE",
) -> dict[str, np.ndarray | float]:
    """Run a single 2D Floquet simulation and return R_m, T_m, and total energy."""
    f_arr = np.array([float(frequency)])
    k0 = 2.0 * np.pi * float(frequency) / c
    kx = k0 * cell.n_inc * float(np.sin(theta_inc_rad))
    k_par = abs(kx)
    phi = 0.0 if kx >= 0.0 else np.pi

    pb, pa = _run_single_Solver(
        cell, k_par=k_par, phi=phi, polarization=polarization, frequencies=f_arr
    )
    pol_u = "TE" if polarization.upper() in ("TE", "S") else "TM"
    a_inc = pb.mode_amplitude((1, 0, 0, pol_u), direction="in")[0]
    orders = np.asarray(cell.orders_x, dtype=int)
    r_ord = np.zeros(orders.size, dtype=float)
    t_ord = np.zeros(orders.size, dtype=float)

    for io, m in enumerate(orders):
        r_amp = pb.mode_amplitude((1, int(m), 0, pol_u), direction="out")[0]
        t_amp = pa.mode_amplitude((2, int(m), 0, pol_u), direction="out")[0]
        r_ord[io] = float(abs(r_amp / a_inc) ** 2) if np.isfinite(r_amp) else np.nan
        t_ord[io] = float(abs(t_amp / a_inc) ** 2) if np.isfinite(t_amp) else np.nan

    tot_e = float(np.nansum(r_ord) + np.nansum(t_ord))
    return {
        "orders": orders,
        "R_orders": r_ord,
        "T_orders": t_ord,
        "energy_total": tot_e,
    }


def capture_slanted_harmonic_field(
    cell: FloquetCell,
    frequency: float,
    theta_inc_rad: float = 0.0,
    polarization: str = "TE",
) -> dict[str, Any]:
    """Run a 2D Bloch FIT simulation and extract the complex steady-state field phasor."""
    grid = wakis.GridFIT3D(
        -0.5 * cell.Lx,
        0.5 * cell.Lx,
        -0.5 * cell.Ly,
        0.5 * cell.Ly,
        cell.zmin,
        cell.zmax,
        cell.nx,
        cell.ny,
        cell.nz,
        verbose=0,
    )
    k0 = 2.0 * np.pi * float(frequency) / c
    kx = k0 * cell.n_inc * float(np.sin(theta_inc_rad))
    k_bloch = (-kx, 0.0, 0.0)

    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["bloch", "bloch", "cpml"],
        bc_high=["bloch", "bloch", "cpml"],
        n_pml=cell.n_pml,
        k_bloch=k_bloch,
        dtype=np.complex128,
        cfln=cell.cfl,
        verbose=0,
    )
    if cell.material_callback is not None:
        cell.material_callback(solver)

    comp_src = "y" if polarization.upper() in ("TE", "S") else "x"
    src_prof = GaussianSource(frequency=float(frequency), fwidth=0.12 * float(frequency))
    sheet = CurrentSource(
        src_prof,
        component=comp_src,
        center=(0.0, 0.0, float(cell.z_source)),
        size=(float(cell.Lx), float(cell.Ly), 0.0),
        amp_func=lambda x, y, z: np.exp(-1j * kx * x),
    )

    t_total = float(src_prof.end_time) + 14.0 / float(frequency)
    n_steps = int(np.ceil(t_total / solver.dt))
    n_dft_steps = max(40, int(np.round(4.0 / (float(frequency) * solver.dt))))
    start_dft = max(0, n_steps - n_dft_steps)

    phasor = np.zeros((grid.Nx, grid.Nz), dtype=np.complex128)
    omega = 2.0 * np.pi * float(frequency)
    for step in range(n_steps):
        t = step * solver.dt
        sheet.update(solver, t)
        solver.one_step()
        if step >= start_dft:
            field_2d = np.asarray(solver.E[:, 0, :, comp_src], dtype=np.complex128)
            phasor += field_2d * np.exp(-1j * omega * t)
    phasor /= float(max(1, n_steps - start_dft))

    ieps_y = np.asarray(solver.ieps[:, 0, :, "y"], dtype=float)
    eps_map = 1.0 / np.maximum(ieps_y * epsilon_0, 1e-30)

    return {
        "x": np.asarray(grid.x[:-1], dtype=float),
        "z": np.asarray(grid.z[:-1], dtype=float),
        "field_phasor": phasor,
        "eps_map": eps_map,
        "kx": float(kx),
    }


def run_task_0_slant_and_profiles(
    quick: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Task 0: Slant angle sweep alpha in [0, 40] deg + Binary vs Blazed vs Slanted comparison."""
    t0 = time.perf_counter()
    cpw = 24 if quick else 36
    f0 = c / WL_DESIGN_M
    orders = (-2, -1, 0, 1, 2)
    im_m1 = orders.index(-1)
    im_0 = orders.index(0)
    im_p1 = orders.index(+1)

    slant_wakis_deg = (
        np.array([0.0, 12.0, 25.0, 35.0])
        if quick
        else np.linspace(0.0, 40.0, 9)
    )
    slant_rcwa_deg = np.linspace(0.0, 42.0, 43)

    rcwa_t_m1 = np.zeros_like(slant_rcwa_deg)
    rcwa_t_0 = np.zeros_like(slant_rcwa_deg)
    rcwa_t_p1 = np.zeros_like(slant_rcwa_deg)
    for ia, s_deg in enumerate(slant_rcwa_deg):
        rc = rcwa_multilayer_1d(
            [f0],
            0.0,
            PERIOD_INCOUPLER_M,
            DEFAULT_DEPTH_M,
            profile_type="slanted",
            slant_deg=float(s_deg),
            fill_factor=0.55,
            orders=orders,
        )
        rcwa_t_m1[ia] = rc["T_orders"][0, im_m1]
        rcwa_t_0[ia] = rc["T_orders"][0, im_0]
        rcwa_t_p1[ia] = rc["T_orders"][0, im_p1]

    wakis_t_m1 = np.zeros_like(slant_wakis_deg)
    wakis_t_0 = np.zeros_like(slant_wakis_deg)
    wakis_t_p1 = np.zeros_like(slant_wakis_deg)
    wakis_energy = np.zeros_like(slant_wakis_deg)
    rcwa_at_wakis_m1 = np.zeros_like(slant_wakis_deg)
    rcwa_at_wakis_p1 = np.zeros_like(slant_wakis_deg)

    for iw, s_deg in enumerate(slant_wakis_deg):
        cell_s, minfo_s = build_slanted_floquet_cell(
            period=PERIOD_INCOUPLER_M,
            height=DEFAULT_DEPTH_M,
            wavelength_min=WL_DESIGN_M,
            profile_type="slanted",
            slant_deg=float(s_deg),
            fill_factor=0.55,
            orders_x=orders,
            cpw=cpw,
        )
        res_w = run_single_grating_orders(cell_s, f0, theta_inc_rad=0.0, polarization="TE")
        wakis_t_m1[iw] = res_w["T_orders"][im_m1]
        wakis_t_0[iw] = res_w["T_orders"][im_0]
        wakis_t_p1[iw] = res_w["T_orders"][im_p1]
        wakis_energy[iw] = res_w["energy_total"]

        rc_w = rcwa_multilayer_1d(
            [f0],
            0.0,
            PERIOD_INCOUPLER_M,
            DEFAULT_DEPTH_M,
            profile_type="slanted",
            slant_deg=float(s_deg),
            fill_factor=0.55,
            orders=orders,
        )
        rcwa_at_wakis_m1[iw] = rc_w["T_orders"][0, im_m1]
        rcwa_at_wakis_p1[iw] = rc_w["T_orders"][0, im_p1]

    # Compare all 3 profiles (Binary alpha=0, Blazed sawtooth, Slanted alpha=25) at theta=0
    profiles_spec = [
        ("binary", 0.0, 0.50),
        ("blazed", 0.0, 0.85),
        ("slanted", 25.0, 0.55),
    ]
    prof_summary = {}
    npz_data: dict[str, np.ndarray] = {
        "orders": np.asarray(orders, dtype=int),
        "slant_wakis_deg": slant_wakis_deg,
        "slant_rcwa_deg": slant_rcwa_deg,
        "rcwa_t_m1": rcwa_t_m1,
        "rcwa_t_0": rcwa_t_0,
        "rcwa_t_p1": rcwa_t_p1,
        "wakis_t_m1": wakis_t_m1,
        "wakis_t_0": wakis_t_0,
        "wakis_t_p1": wakis_t_p1,
        "wakis_energy": wakis_energy,
        "rcwa_at_wakis_m1": rcwa_at_wakis_m1,
        "rcwa_at_wakis_p1": rcwa_at_wakis_p1,
    }

    for p_name, s_d, ff in profiles_spec:
        cell_p, _ = build_slanted_floquet_cell(
            period=PERIOD_INCOUPLER_M,
            height=DEFAULT_DEPTH_M,
            wavelength_min=WL_DESIGN_M,
            profile_type=p_name,
            slant_deg=s_d,
            fill_factor=ff,
            orders_x=orders,
            cpw=cpw,
        )
        rw = run_single_grating_orders(cell_p, f0, theta_inc_rad=0.0, polarization="TE")
        rc = rcwa_multilayer_1d(
            [f0],
            0.0,
            PERIOD_INCOUPLER_M,
            DEFAULT_DEPTH_M,
            profile_type=p_name,
            slant_deg=s_d,
            fill_factor=ff,
            orders=orders,
        )
        npz_data[f"prof_{p_name}_wakis_t"] = rw["T_orders"]
        npz_data[f"prof_{p_name}_rcwa_t"] = rc["T_orders"][0]
        npz_data[f"prof_{p_name}_wakis_r"] = rw["R_orders"]
        npz_data[f"prof_{p_name}_rcwa_r"] = rc["R_orders"][0]

        t_m1_val = float(rw["T_orders"][im_m1])
        t_p1_val = float(rw["T_orders"][im_p1])
        prof_summary[p_name] = {
            "T_m1_wakis": t_m1_val,
            "T_m1_rcwa": float(rc["T_orders"][0, im_m1]),
            "T_p1_wakis": t_p1_val,
            "T_p1_rcwa": float(rc["T_orders"][0, im_p1]),
            "asymmetry_ratio_m1_over_p1": float(t_m1_val / max(t_p1_val, 1e-12)),
            "max_abs_err_vs_rcwa": float(np.nanmax(np.abs(rw["T_orders"] - rc["T_orders"][0]))),
        }

    bragg_alpha = bragg_slant_angle_deg(0.0)
    summary = {
        "index": 0,
        "name": "slant_angle_and_profile_comparison",
        "cpw": cpw,
        "bragg_slant_angle_deg": bragg_alpha,
        "binary_symmetry_diff_at_0": float(abs(wakis_t_m1[0] - wakis_t_p1[0])),
        "peak_slanted_t_m1_wakis": float(np.max(wakis_t_m1)),
        "peak_slanted_t_m1_rcwa": float(np.max(rcwa_t_m1)),
        "max_slant_sweep_err_m1": float(np.max(np.abs(wakis_t_m1 - rcwa_at_wakis_m1))),
        "max_energy_residual": float(np.max(np.abs(wakis_energy - 1.0))),
        "profiles": prof_summary,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    return summary, npz_data


def run_task_1_depth_and_fill_sweeps(
    quick: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Task 1: Groove depth d and fill factor f sweeps + 2D RCWA design map."""
    t0 = time.perf_counter()
    cpw = 22 if quick else 32
    f0 = c / WL_DESIGN_M
    orders = (-2, -1, 0, 1, 2)
    im_m1 = orders.index(-1)
    im_0 = orders.index(0)
    im_p1 = orders.index(+1)

    # 1. Depth sweep at slant = 25 deg, fill = 0.66 (Wang & Ma Table 1)
    depths_w_nm = (
        np.array([140.0, 240.0, 330.0, 415.0])
        if quick
        else np.linspace(100.0, 440.0, 9)
    )
    depths_r_nm = np.linspace(80.0, 460.0, 39)

    rcwa_depth_m1 = np.zeros_like(depths_r_nm)
    rcwa_depth_0 = np.zeros_like(depths_r_nm)
    rcwa_depth_p1 = np.zeros_like(depths_r_nm)
    for idp, d_nm in enumerate(depths_r_nm):
        rc = rcwa_multilayer_1d(
            [f0],
            0.0,
            PERIOD_INCOUPLER_M,
            d_nm * 1e-9,
            profile_type="slanted",
            slant_deg=WANG_SLANT_DEG,
            fill_factor=WANG_FILL_FACTOR,
            orders=orders,
        )
        rcwa_depth_m1[idp] = rc["T_orders"][0, im_m1]
        rcwa_depth_0[idp] = rc["T_orders"][0, im_0]
        rcwa_depth_p1[idp] = rc["T_orders"][0, im_p1]

    wakis_depth_m1 = np.zeros_like(depths_w_nm)
    wakis_depth_0 = np.zeros_like(depths_w_nm)
    wakis_depth_p1 = np.zeros_like(depths_w_nm)
    rcwa_at_w_depth_m1 = np.zeros_like(depths_w_nm)
    for idp, d_nm in enumerate(depths_w_nm):
        cell_d, _ = build_slanted_floquet_cell(
            period=PERIOD_INCOUPLER_M,
            height=d_nm * 1e-9,
            wavelength_min=WL_DESIGN_M,
            profile_type="slanted",
            slant_deg=WANG_SLANT_DEG,
            fill_factor=WANG_FILL_FACTOR,
            orders_x=orders,
            cpw=cpw,
        )
        rw = run_single_grating_orders(cell_d, f0, theta_inc_rad=0.0, polarization="TE")
        wakis_depth_m1[idp] = rw["T_orders"][im_m1]
        wakis_depth_0[idp] = rw["T_orders"][im_0]
        wakis_depth_p1[idp] = rw["T_orders"][im_p1]
        rc_d = rcwa_multilayer_1d(
            [f0],
            0.0,
            PERIOD_INCOUPLER_M,
            d_nm * 1e-9,
            profile_type="slanted",
            slant_deg=WANG_SLANT_DEG,
            fill_factor=WANG_FILL_FACTOR,
            orders=orders,
        )
        rcwa_at_w_depth_m1[idp] = rc_d["T_orders"][0, im_m1]

    # 2. Fill factor sweep at slant = 25 deg, depth = 320 nm
    fills_w = (
        np.array([0.35, 0.50, 0.66, 0.75])
        if quick
        else np.linspace(0.25, 0.78, 9)
    )
    fills_r = np.linspace(0.20, 0.82, 32)
    rcwa_fill_m1 = np.zeros_like(fills_r)
    rcwa_fill_p1 = np.zeros_like(fills_r)
    for iff, ff in enumerate(fills_r):
        rc = rcwa_multilayer_1d(
            [f0],
            0.0,
            PERIOD_INCOUPLER_M,
            DEFAULT_DEPTH_M,
            profile_type="slanted",
            slant_deg=WANG_SLANT_DEG,
            fill_factor=float(ff),
            orders=orders,
        )
        rcwa_fill_m1[iff] = rc["T_orders"][0, im_m1]
        rcwa_fill_p1[iff] = rc["T_orders"][0, im_p1]

    wakis_fill_m1 = np.zeros_like(fills_w)
    wakis_fill_p1 = np.zeros_like(fills_w)
    rcwa_at_w_fill_m1 = np.zeros_like(fills_w)
    for iff, ff in enumerate(fills_w):
        cell_f, _ = build_slanted_floquet_cell(
            period=PERIOD_INCOUPLER_M,
            height=DEFAULT_DEPTH_M,
            wavelength_min=WL_DESIGN_M,
            profile_type="slanted",
            slant_deg=WANG_SLANT_DEG,
            fill_factor=float(ff),
            orders_x=orders,
            cpw=cpw,
        )
        rw = run_single_grating_orders(cell_f, f0, theta_inc_rad=0.0, polarization="TE")
        wakis_fill_m1[iff] = rw["T_orders"][im_m1]
        wakis_fill_p1[iff] = rw["T_orders"][im_p1]
        rc_f = rcwa_multilayer_1d(
            [f0],
            0.0,
            PERIOD_INCOUPLER_M,
            DEFAULT_DEPTH_M,
            profile_type="slanted",
            slant_deg=WANG_SLANT_DEG,
            fill_factor=float(ff),
            orders=orders,
        )
        rcwa_at_w_fill_m1[iff] = rc_f["T_orders"][0, im_m1]

    # 3. 2D RCWA design map over (fill_factor, depth_nm) at slant = 25 deg
    map_fills = np.linspace(0.25, 0.80, 15 if quick else 25)
    map_depths_nm = np.linspace(120.0, 440.0, 15 if quick else 25)
    map_t_m1 = np.zeros((map_depths_nm.size, map_fills.size))
    for idp, d_nm in enumerate(map_depths_nm):
        for iff, ff in enumerate(map_fills):
            rc_m = rcwa_multilayer_1d(
                [f0],
                0.0,
                PERIOD_INCOUPLER_M,
                d_nm * 1e-9,
                profile_type="slanted",
                slant_deg=WANG_SLANT_DEG,
                fill_factor=float(ff),
                n_harmonics=9,
                n_slices=18,
                orders=(-1,),
            )
            map_t_m1[idp, iff] = rc_m["T_orders"][0, 0]

    npz_data = {
        "depths_w_nm": depths_w_nm,
        "depths_r_nm": depths_r_nm,
        "rcwa_depth_m1": rcwa_depth_m1,
        "rcwa_depth_0": rcwa_depth_0,
        "rcwa_depth_p1": rcwa_depth_p1,
        "wakis_depth_m1": wakis_depth_m1,
        "wakis_depth_0": wakis_depth_0,
        "wakis_depth_p1": wakis_depth_p1,
        "fills_w": fills_w,
        "fills_r": fills_r,
        "rcwa_fill_m1": rcwa_fill_m1,
        "rcwa_fill_p1": rcwa_fill_p1,
        "wakis_fill_m1": wakis_fill_m1,
        "wakis_fill_p1": wakis_fill_p1,
        "map_fills": map_fills,
        "map_depths_nm": map_depths_nm,
        "map_t_m1": map_t_m1,
    }
    summary = {
        "index": 1,
        "name": "depth_and_fill_factor_sweeps",
        "cpw": cpw,
        "max_depth_sweep_err_m1": float(np.max(np.abs(wakis_depth_m1 - rcwa_at_w_depth_m1))),
        "max_fill_sweep_err_m1": float(np.max(np.abs(wakis_fill_m1 - rcwa_at_w_fill_m1))),
        "peak_2d_map_t_m1": float(np.max(map_t_m1)),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    return summary, npz_data


def run_task_2_oblique_and_broadband(
    quick: bool = True,
    processes: int = 2,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Task 2: Oblique incidence FOV sweeps (Binary, Blazed, Slanted) + broadband AngleSweep."""
    t0 = time.perf_counter()
    cpw = 21 if quick else 32
    n_k = 4 if quick else 9
    rtc = 21.0 if quick else 28.0
    f0 = np.array([c / WL_DESIGN_M])
    orders = (-2, -1, 0, 1, 2)
    im_m1 = orders.index(-1)

    angles_pos_deg = (
        np.array([0.0, 5.0, 10.0, 15.0])
        if quick
        else np.linspace(0.0, 16.0, 9)
    )
    angles_pos_rad = np.deg2rad(angles_pos_deg)

    npz_data: dict[str, np.ndarray] = {"angles_pos_deg": angles_pos_deg}
    oblique_metrics: dict[str, Any] = {}

    for p_name, s_d, ff in [
        ("binary", 0.0, 0.50),
        ("blazed", 0.0, 0.85),
        ("slanted", 25.0, 0.55),
    ]:
        cell_p, _ = build_slanted_floquet_cell(
            period=PERIOD_INCOUPLER_M,
            height=DEFAULT_DEPTH_M,
            wavelength_min=WL_DESIGN_M,
            profile_type=p_name,
            slant_deg=s_d,
            fill_factor=ff,
            orders_x=orders,
            cpw=cpw,
            run_time_cycles=rtc,
        )
        sweep_p = AngleSweep(cell_p, f0, angles_pos_rad, polarization="TE", n_k=n_k)
        res_p = sweep_p.run(processes=processes)
        out_p = res_p.resample()
        w_m1 = out_p["T_order"][:, 0, im_m1, 0]
        e_tot = out_p["R_total"][:, 0] + out_p["T_total"][:, 0]

        r_m1 = np.zeros_like(angles_pos_deg)
        for ia, th_r in enumerate(angles_pos_rad):
            rc = rcwa_multilayer_1d(
                f0,
                th_r,
                PERIOD_INCOUPLER_M,
                DEFAULT_DEPTH_M,
                profile_type=p_name,
                slant_deg=s_d,
                fill_factor=ff,
                orders=(-1,),
            )
            r_m1[ia] = rc["T_orders"][0, 0]

        npz_data[f"oblique_{p_name}_wakis_m1"] = w_m1
        npz_data[f"oblique_{p_name}_rcwa_m1"] = r_m1
        npz_data[f"oblique_{p_name}_energy"] = e_tot
        oblique_metrics[p_name] = {
            "mean_t_m1_wakis": float(np.mean(w_m1)),
            "mean_t_m1_rcwa": float(np.mean(r_m1)),
            "max_abs_err_vs_rcwa": float(np.max(np.abs(w_m1 - r_m1))),
            "max_energy_residual": float(np.max(np.abs(e_tot - 1.0))),
        }

    # Also run TM polarization for the slanted grating to compare TE vs TM
    cell_tm, _ = build_slanted_floquet_cell(
        period=PERIOD_INCOUPLER_M,
        height=DEFAULT_DEPTH_M,
        wavelength_min=WL_DESIGN_M,
        profile_type="slanted",
        slant_deg=25.0,
        fill_factor=0.55,
        orders_x=orders,
        cpw=cpw,
    )
    sweep_tm = AngleSweep(cell_tm, f0, angles_pos_rad, polarization="TM", n_k=n_k)
    out_tm = sweep_tm.run(processes=processes).resample()
    w_tm_m1 = out_tm["T_order"][:, 0, im_m1, 0]
    r_tm_m1 = np.zeros_like(angles_pos_deg)
    for ia, th_r in enumerate(angles_pos_rad):
        rc_tm = rcwa_multilayer_1d(
            f0,
            th_r,
            PERIOD_INCOUPLER_M,
            DEFAULT_DEPTH_M,
            profile_type="slanted",
            slant_deg=25.0,
            fill_factor=0.55,
            pol="TM",
            orders=(-1,),
        )
        r_tm_m1[ia] = rc_tm["T_orders"][0, 0]
    npz_data["oblique_slanted_tm_wakis_m1"] = w_tm_m1
    npz_data["oblique_slanted_tm_rcwa_m1"] = r_tm_m1

    # Broadband constant-angle sweep at theta_in = 8 deg across lambda in [490, 565] nm
    wl_bb_nm = np.linspace(490.0, 565.0, 7 if quick else 13)
    freqs_bb = c / (wl_bb_nm * 1e-9)
    theta_bb_deg = np.array([4.0, 8.0, 12.0])
    theta_bb_rad = np.deg2rad(theta_bb_deg)

    cell_bb, _ = build_slanted_floquet_cell(
        period=PERIOD_INCOUPLER_M,
        height=DEFAULT_DEPTH_M,
        wavelength_min=float(np.min(wl_bb_nm) * 1e-9),
        profile_type="slanted",
        slant_deg=25.0,
        fill_factor=0.55,
        orders_x=orders,
        cpw=cpw,
    )
    sweep_bb = AngleSweep(
        cell_bb, freqs_bb, theta_bb_rad, polarization="TE", n_k=6 if quick else 9
    )
    res_bb = sweep_bb.run(processes=processes)
    out_bb = res_bb.resample()
    out_bb_shifted = res_bb.resample(index_shift=1)

    # Index 1 corresponds to theta = 8.0 deg
    w_bb_8deg = out_bb["T_order"][1, :, im_m1, 0]
    w_bb_8deg_shifted = out_bb_shifted["T_order"][1, :, im_m1, 0]
    rc_bb_8deg = rcwa_multilayer_1d(
        freqs_bb,
        np.deg2rad(8.0),
        PERIOD_INCOUPLER_M,
        DEFAULT_DEPTH_M,
        profile_type="slanted",
        slant_deg=25.0,
        fill_factor=0.55,
        orders=(-1,),
    )["T_orders"][:, 0]

    npz_data["wl_bb_nm"] = wl_bb_nm
    npz_data["w_bb_8deg"] = w_bb_8deg
    npz_data["w_bb_8deg_shifted"] = w_bb_8deg_shifted
    npz_data["rc_bb_8deg"] = rc_bb_8deg

    summary = {
        "index": 2,
        "name": "oblique_incidence_and_broadband_sweep",
        "cpw": cpw,
        "profiles_oblique": oblique_metrics,
        "broadband_8deg_max_err": float(np.nanmax(np.abs(w_bb_8deg - rc_bb_8deg))),
        "broadband_8deg_shifted_err": float(np.nanmax(np.abs(w_bb_8deg_shifted - rc_bb_8deg))),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    return summary, npz_data


def run_task_3_staircase_vs_conformal(
    quick: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Task 3: Staircase vs Kottke Conformal resolution convergence + near-field maps."""
    t0 = time.perf_counter()
    cpw_levels = [16, 20, 26, 34] if quick else [16, 24, 32, 48]
    f0 = c / WL_DESIGN_M
    theta_test_rad = np.deg2rad(8.0)
    orders = (-2, -1, 0, 1, 2)
    im_m1 = orders.index(-1)

    rc_exact = rcwa_multilayer_1d(
        [f0],
        theta_test_rad,
        PERIOD_INCOUPLER_M,
        DEFAULT_DEPTH_M,
        profile_type="slanted",
        slant_deg=25.0,
        fill_factor=0.55,
        n_harmonics=18,
        n_slices=48,
        orders=orders,
    )
    t_exact = rc_exact["T_orders"][0]

    err_kottke_rms = []
    err_stair_rms = []
    err_kottke_m1 = []
    err_stair_m1 = []
    dx_nm_list = []

    for cpw in cpw_levels:
        cell_k, minfo = build_slanted_floquet_cell(
            period=PERIOD_INCOUPLER_M,
            height=DEFAULT_DEPTH_M,
            wavelength_min=WL_DESIGN_M,
            profile_type="slanted",
            slant_deg=25.0,
            fill_factor=0.55,
            averaging="kottke",
            orders_x=orders,
            cpw=cpw,
        )
        rw_k = run_single_grating_orders(cell_k, f0, theta_inc_rad=theta_test_rad, polarization="TE")

        cell_s, _ = build_slanted_floquet_cell(
            period=PERIOD_INCOUPLER_M,
            height=DEFAULT_DEPTH_M,
            wavelength_min=WL_DESIGN_M,
            profile_type="slanted",
            slant_deg=25.0,
            fill_factor=0.55,
            averaging="staircase",
            orders_x=orders,
            cpw=cpw,
        )
        rw_s = run_single_grating_orders(cell_s, f0, theta_inc_rad=theta_test_rad, polarization="TE")

        dx_nm_list.append(minfo["dx_nm"])
        err_kottke_m1.append(float(abs(rw_k["T_orders"][im_m1] - t_exact[im_m1])))
        err_stair_m1.append(float(abs(rw_s["T_orders"][im_m1] - t_exact[im_m1])))
        err_kottke_rms.append(float(np.sqrt(np.nanmean((rw_k["T_orders"] - t_exact) ** 2))))
        err_stair_rms.append(float(np.sqrt(np.nanmean((rw_s["T_orders"] - t_exact) ** 2))))

    # Capture steady-state 2D field phasor maps for Binary (alpha=0) vs Slanted (alpha=25) at theta=0
    cpw_field = 24 if quick else 32
    cell_bin_f, _ = build_slanted_floquet_cell(
        period=PERIOD_INCOUPLER_M,
        height=DEFAULT_DEPTH_M,
        wavelength_min=WL_DESIGN_M,
        profile_type="binary",
        slant_deg=0.0,
        fill_factor=0.50,
        orders_x=orders,
        cpw=cpw_field,
    )
    nf_bin = capture_slanted_harmonic_field(cell_bin_f, f0, theta_inc_rad=0.0, polarization="TE")

    cell_sl_f, _ = build_slanted_floquet_cell(
        period=PERIOD_INCOUPLER_M,
        height=DEFAULT_DEPTH_M,
        wavelength_min=WL_DESIGN_M,
        profile_type="slanted",
        slant_deg=25.0,
        fill_factor=0.55,
        orders_x=orders,
        cpw=cpw_field,
    )
    nf_sl = capture_slanted_harmonic_field(cell_sl_f, f0, theta_inc_rad=0.0, polarization="TE")

    npz_data = {
        "cpw_levels": np.asarray(cpw_levels, dtype=int),
        "dx_nm": np.asarray(dx_nm_list, dtype=float),
        "err_kottke_rms": np.asarray(err_kottke_rms, dtype=float),
        "err_stair_rms": np.asarray(err_stair_rms, dtype=float),
        "err_kottke_m1": np.asarray(err_kottke_m1, dtype=float),
        "err_stair_m1": np.asarray(err_stair_m1, dtype=float),
        "t_exact": t_exact,
        "bin_x": nf_bin["x"],
        "bin_z": nf_bin["z"],
        "bin_phasor": nf_bin["field_phasor"],
        "bin_eps": nf_bin["eps_map"],
        "bin_kx": np.array(nf_bin["kx"]),
        "sl_x": nf_sl["x"],
        "sl_z": nf_sl["z"],
        "sl_phasor": nf_sl["field_phasor"],
        "sl_eps": nf_sl["eps_map"],
        "sl_kx": np.array(nf_sl["kx"]),
    }
    summary = {
        "index": 3,
        "name": "staircase_vs_conformal_convergence",
        "cpw_levels": cpw_levels,
        "dx_nm": dx_nm_list,
        "err_kottke_rms": err_kottke_rms,
        "err_stair_rms": err_stair_rms,
        "kottke_finest_rms": float(err_kottke_rms[-1]),
        "stair_finest_rms": float(err_stair_rms[-1]),
        "kottke_improvement_factor": float(err_stair_rms[-1] / max(err_kottke_rms[-1], 1e-15)),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    return summary, npz_data
