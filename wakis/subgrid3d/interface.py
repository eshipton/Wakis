# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plane-wave reflection of a planar coarse/fine interface in 3D (verification).

As in the 2D `wakis.subgrid.interface`, the leapfrog steady state at angular
frequency w satisfies exactly

    (K^H W K - w_t^2 D) e = -i w_t i_src,    w_t = 2 sin(w dt / 2) / dt,

so the reflection of the time-domain scheme is computed in the frequency
domain. The domain is a column along x, Bloch-periodic in y and z with the
period of two coarse cells; the fine region x > x_i is a patch that spans the
whole y-z period. Both ends carry a complex-stretched absorber. A current
sheet in the coarse region launches the wave; the coarse samples between
source and interface are fitted to A e^{-i k_x x} + B e^{+i k_x x} with the
exact discrete k_x of the coarse Yee grid, so |B / A| has no coarse
dispersion in it.

Polarisations (plane of incidence xy, k_z = 0):

* ``'s'``: E = E_z (the 3D version of the 2D TM problem; z-invariant fields
  make the 3D scheme reduce exactly to `SubgridFIT2D` TM),
* ``'p'``: E in the xy plane (source J_y, fitted on E_y).
"""

import numpy as np
import scipy.sparse as sp
from scipy.constants import c as C0
from scipy.sparse.linalg import spsolve

from .mesh import SubgridMesh3D
from .solver import SubgridFIT3D

__all__ = ["bloch_reflection_3d", "reflection_1d_exact", "column_size"]


def _setup(cells_per_wavelength, ratio, refine):
    H = 1.0
    lam = cells_per_wavelength * H
    n_abs = int(max(20, np.ceil(1.5 * lam / H)))
    n_coarse = 2 * (n_abs + int(np.ceil(5 * lam / H)))
    return H, lam, n_abs, n_coarse


def column_size(cells_per_wavelength, ratio=2):
    """Number of free edges of the reflection column (for preflight)."""
    H, lam, n_abs, n = _setup(cells_per_wavelength, ratio, True)
    coarse = (n // 2) * 4 * 3
    fine = (n // 2) * ratio * (2 * ratio) ** 2 * 3
    return coarse + fine


def bloch_reflection_3d(cells_per_wavelength, theta_deg, ratio=2, pol="s", phi_deg=0.0,
                        mutate=None, refine=True, courant=0.5, bloch_source=True,
                        return_solver=False):  # fmt: skip
    """
    Reflection coefficient |R| of a planar 3D coarse/fine interface.

    Parameters
    ----------
    cells_per_wavelength : float
        Coarse cells per vacuum wavelength, N.
    theta_deg : float
        Angle of incidence from the interface normal (x) [deg].
    ratio : int
        Refinement ratio (2 or 3).
    pol : {'s', 'p'}
        's': E along z (source J_z, fit E_z); 'p': E in the plane of
        incidence (source J_y, fit E_y). With ``phi_deg != 0`` the plane of
        incidence is rotated about x and the 's' source/fit is kept on E_z.
    phi_deg : float
        Azimuth of the plane of incidence about x [deg] (0: xy plane).
    mutate : passed to `SubgridFIT3D`.
    refine : bool
        False: plain coarse column (absorber/fit floor).
    courant : float
        dt = courant * h_fine / (c sqrt(2)), the same definition as the 2D
        `bloch_reflection` (only enters through w_t).
    bloch_source : bool
        True: the source sheet carries the Bloch phase e^{-i(k_y y + k_z z)}
        (only the specular harmonic is launched). False: equal amplitudes on
        every source edge, exactly as the 2D `bloch_reflection` (the
        evanescent harmonic k_y + pi/H is also excited).

    Returns
    -------
    float, or (float, SubgridFIT3D) if `return_solver`.
    """
    H, lam, n_abs, n_coarse = _setup(cells_per_wavelength, ratio, refine)
    k0 = 2 * np.pi / lam
    x = H * np.arange(n_coarse + 1)
    xi = H * (n_coarse // 2)
    yz = np.array([0.0, H, 2 * H])
    h = H / ratio if refine else H
    dt = courant * h / (C0 * np.sqrt(2))
    w = k0 * C0
    wt = 2 * np.sin(w * dt / 2) / dt
    kt = wt / C0
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    ky = kt * np.sin(th) * np.cos(ph)
    kz = kt * np.sin(th) * np.sin(ph)
    sx2 = (kt * H / 2) ** 2 - np.sin(ky * H / 2) ** 2 - np.sin(kz * H / 2) ** 2
    kx = 2 / H * np.arcsin(np.sqrt(sx2))
    phase = (1.0, np.exp(-1j * ky * 2 * H), np.exp(-1j * kz * 2 * H))
    mesh = SubgridMesh3D(
        x, yz, yz, bc=[("pec", "pec"), "periodic", "periodic"], phase=phase
    )
    if refine:
        mesh.add_patch((n_coarse // 2, 0, 0), (n_coarse, 2, 2), ratio=ratio)
    T = n_abs * H

    def sx(v):
        d = np.maximum(T - v, 0) + np.maximum(v - (x[-1] - T), 0)
        return 1 - 1j * 6.0 * (d / T) ** 3

    def edge_scale(mid, direction):
        s = sx(mid[:, 0])
        return np.where(direction == 0, 1 / s, s)

    def face_scale(cen, normal):
        s = sx(cen[:, 0])
        return np.where(normal == 0, s, 1 / s)

    s = SubgridFIT3D(
        mesh, dt=dt, mutate=mutate, edge_scale=edge_scale, face_scale=face_scale
    )
    A = (s.KH @ sp.diags(s.W) @ s.K - wt**2 * sp.diags(s.D)).tocsc()
    xs = T + 5 * H
    comp = 2 if pol == "s" else 1
    em = s.edge_mid
    src = np.flatnonzero(
        (s.edge_block == 0) & (s.edge_dir == comp) & np.isclose(em[:, 0], xs)
    )
    i_all = np.zeros(mesh.n_edges, complex)
    if bloch_source:
        i_all[src] = np.exp(-1j * (ky * em[src, 1] + kz * em[src, 2]))
    else:  # equal amplitudes, as the 2D `bloch_reflection`
        i_all[src] = 1.0
    rhs = s.P.conj().T @ i_all
    e = spsolve(A, rhs)
    ea = s.P @ e
    sel = (
        (s.edge_block == 0)
        & (s.edge_dir == comp)
        & (em[:, 0] > xs + 2 * H)
        & (em[:, 0] < xi - 5 * H)
        & np.isclose(em[:, 1], 0.5 * H if comp == 1 else 0.0)
        & np.isclose(em[:, 2], 0.5 * H if comp == 2 else 0.0)
    )
    X = em[sel, 0]
    Mfit = np.stack([np.exp(-1j * kx * X), np.exp(1j * kx * X)], 1)
    (a_inc, b_ref), *_ = np.linalg.lstsq(Mfit, ea[sel], rcond=None)
    R = float(np.abs(b_ref / a_inc))
    return (R, s) if return_solver else R


def reflection_1d_exact(cells_per_wavelength, ratio=2, courant=0.5):
    """
    Exact |R| of the semi-discrete 1D FIT chain with a step H -> h = H/r at
    a node with trapezoidal mass (H + h)/2: the normal-incidence limit of the
    3D interface (fields uniform over the interface reduce it to this chain).

    Coarse ``u_j = A e^{-ikHj} + B e^{ikHj}`` (j <= 0), fine
    ``u_j = T e^{-i kappa h j}`` (j >= 0), node 0 shared; the interface row
    ``(u0 - u_-1)/H - (u1 - u0)/h = k_t^2 (H + h)/2 u0`` gives
    ``R = -(alpha - gamma)/(beta - gamma)``.
    """
    H = 1.0
    h = H / ratio
    k0 = 2 * np.pi / (cells_per_wavelength * H)
    dt = courant * h / (C0 * np.sqrt(2))
    w = k0 * C0
    kt = 2 * np.sin(w * dt / 2) / dt / C0
    k = 2 / H * np.arcsin(kt * H / 2)
    kap = 2 / h * np.arcsin(kt * h / 2)
    alpha = (1 - np.exp(1j * k * H)) / H
    beta = (1 - np.exp(-1j * k * H)) / H
    gamma = (np.exp(-1j * kap * h) - 1) / h + kt**2 * (H + h) / 2
    return float(np.abs((alpha - gamma) / (beta - gamma)))
