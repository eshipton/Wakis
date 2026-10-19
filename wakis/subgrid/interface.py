# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plane-wave reflection of a straight coarse/fine interface (verification).

The leapfrog steady state at angular frequency w satisfies exactly

    (K^H W K - w_t^2 D) u = b,     w_t = 2 sin(w dt / 2) / dt,

so the reflection of the *time-domain* scheme at w can be computed in the
frequency domain. The interface x = x_i is periodic in y with the period of
one coarse cell, so a strip one coarse cell high with a Bloch phase
exp(-i k_y L_y) represents an infinite interface hit at angle theta. Both
ends of the strip carry a complex-stretched absorber; a line source in the
coarse region launches the wave and the coarse-grid samples between source
and interface are fitted to A e^{-i k_x x} + B e^{+i k_x x} with the exact
discrete wavenumber of the coarse grid, so R = |B / A| contains no coarse
numerical dispersion: only the interface and the fine-grid mismatch.
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as C0
from scipy.sparse.linalg import spsolve

from .mesh import QuadMesh
from .solver import SubgridFIT2D

__all__ = ["bloch_reflection", "dispersion_mismatch"]


def bloch_reflection(cells_per_wavelength, theta_deg, ratio=2, polarization="TM",
                     n_coarse=None, n_fine_cells=None, mutate=None, refine=True,
                     courant=0.5):  # fmt: skip
    """
    Reflection coefficient |R| of the coarse/fine interface.

    Parameters
    ----------
    cells_per_wavelength : float
        Coarse cells per vacuum wavelength.
    theta_deg : float
        Angle of incidence [deg].
    ratio : int
        Refinement ratio.
    n_coarse : int
        Coarse cells along x (half coarse, half refined); default: 5
        wavelengths plus an absorber of 1.5 wavelengths on each side.
    mutate : {None, 'inject'}
        'inject': hanging nodes copy their nearest coarse neighbour instead of
        interpolating (first-order interface; still symmetric and stable).
    refine : bool
        False: plain coarse strip (measures the absorber/fit floor).
    courant : float
        dt as a fraction of the 1D fine-grid limit, for w_t.

    Returns
    -------
    float
    """
    H = 1.0
    lam = cells_per_wavelength * H
    k0 = 2 * np.pi / lam
    n_abs = int(max(20, np.ceil(1.5 * lam / H)))  # absorber thickness
    if n_coarse is None:
        n_coarse = 2 * (n_abs + int(np.ceil(5 * lam / H)))
    x = H * np.arange(n_coarse + 1)
    xi = H * (n_coarse // 2)
    y = np.array([0.0, H, 2 * H])  # period: two coarse cells

    def ref(level, x0, x1, y0, y1):
        return x0 >= xi - 1e-12

    mesh = QuadMesh(x, y, ratio=ratio, max_level=1 if refine else 0,
                    refine=ref if refine else None, periodic_y=True)  # fmt: skip
    if mutate == "inject":
        mesh.P = _inject(mesh, 1.0).real.tocsr()
    s = SubgridFIT2D(mesh, polarization=polarization, cfl=0.5)
    theta = np.deg2rad(theta_deg)
    # time-discrete frequency: dt from the fine grid
    h = H / ratio if refine else H
    dt = courant * h / (C0 * np.sqrt(2))
    w = k0 * C0
    wt = 2 * np.sin(w * dt / 2) / dt
    kt = wt / C0
    ky = kt * np.sin(theta)
    # exact discrete kx of the coarse grid (5-point stencil)
    sx2 = (kt * H / 2) ** 2 - np.sin(ky * H / 2) ** 2
    kx = 2 / H * np.arcsin(np.sqrt(sx2))
    phase = np.exp(-1j * ky * 2 * H)
    G = mesh.gradient(phase)
    Pp = mesh.prolongation(phase) if mutate is None else _inject(mesh, phase)
    keep = np.isin(mesh.free, s.dof)
    K = (G @ Pp[:, keep]).tocsr()[s.edge_idx]
    # complex stretching at both ends (absorbers 20 coarse cells thick)
    T = n_abs * H
    edir = mesh.edge_dir[s.edge_idx]
    em = mesh.edge_mid[s.edge_idx]
    xn = mesh.nodes[s.dof, 0]

    def sx(v):
        d = np.maximum(T - v, 0) + np.maximum(v - (x[-1] - T), 0)
        return 1 - 1j * 6.0 * (d / T) ** 3

    W = s.W * np.where(edir == 0, 1 / sx(em[:, 0]), sx(em[:, 0]))
    D = s.D * sx(xn)
    if polarization == "TE":  # same scalar equation; node mass from mu
        pass
    A = (K.conj().T @ sp.diags(W) @ K - wt**2 * sp.diags(D)).tocsc()
    xs = T + 5 * H
    b = np.zeros(s.n_dof, complex)
    src = np.flatnonzero(np.isclose(xn, xs))
    b[src] = 1.0
    u = spsolve(A, b)
    yn = mesh.nodes[s.dof, 1]
    sel = (xn > xs + 2 * H) & (xn < xi - 5 * H) & np.isclose(yn, 0)
    X = xn[sel]
    M = np.stack([np.exp(-1j * kx * X), np.exp(1j * kx * X)], 1)
    (a_inc, b_ref), *_ = np.linalg.lstsq(M, u[sel], rcond=None)
    return float(np.abs(b_ref / a_inc))


def _inject(mesh, phase):
    """Mutation: hanging nodes copy one master (first-order interface)."""
    h, ab, t, wb = mesh._constraints
    t_inj = np.where(t >= 0.5, 1.0, 0.0)
    old = mesh._constraints
    mesh._constraints = (h, ab, t_inj, wb)
    try:
        return mesh.prolongation(phase)
    finally:
        mesh._constraints = old


def dispersion_mismatch(cells_per_wavelength, theta_deg, ratio=2, courant=0.5):
    """
    Reflection expected from the different numerical dispersion of the two
    grids alone (scalar wave: continuity of u and of its normal derivative
    with the discrete normal wavenumbers k_x,c and k_x,f).
    """
    H = 1.0
    k0 = 2 * np.pi / (cells_per_wavelength * H)
    h = H / ratio
    dt = courant * h / (C0 * np.sqrt(2))
    w = k0 * C0
    kt = 2 * np.sin(w * dt / 2) / dt / C0
    ky = kt * np.sin(np.deg2rad(theta_deg))

    def kx(hh):
        s2 = (kt * hh / 2) ** 2 - np.sin(ky * hh / 2) ** 2
        return 2 / hh * np.arcsin(np.sqrt(s2))

    # flux factor of the staggered difference: sin(kx h)/h for each grid
    zc = np.sin(kx(H) * H / 2) / (H / 2)
    zf = np.sin(kx(h) * h / 2) / (h / 2)
    return float(abs(zc - zf) / (zc + zf))
