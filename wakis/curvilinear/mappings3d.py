# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
3D curvilinear hexahedral node mappings and analytic cavity resonance helpers
for `HexBlock3D` and `CurvilinearFIT3D`.

Mappings
--------
1. **Smoothly perturbed Cartesian box** (`perturbed_box_nodes`):
   Starts from a uniform Cartesian grid on ``[0, a] x [0, b] x [0, d]`` with
   spacings ``h_x = a / Nx``, ``h_y = b / Ny``, ``h_z = d / Nz`` and applies
   the smooth interior displacement

   .. math::

       \\delta x(x_0, y_0, z_0) &= \\text{amp} \\, h_x
           \\sin(\\pi x_0 / a) \\sin(\\pi y_0 / b) \\sin(\\pi z_0 / d), \\\\
       \\delta y(x_0, y_0, z_0) &= \\text{amp} \\, h_y
           \\sin(\\pi x_0 / a) \\sin(\\pi y_0 / b) \\sin(\\pi z_0 / d), \\\\
       \\delta z(x_0, y_0, z_0) &= \\text{amp} \\, h_z
           \\sin(\\pi x_0 / a) \\sin(\\pi y_0 / b) \\sin(\\pi z_0 / d).

   Because ``sin(0) = sin(pi) = 0``, the displacement vanishes identically on
   all six boundary planes, leaving every boundary node fixed and preserving
   the exact rectangular outer domain ``[0, a] x [0, b] x [0, d]``. For
   ``|amp| <= 0.3``, the mapping is smooth (``C^infinity``) with strictly
   positive Jacobian everywhere, and the lowest ``TE101`` cavity frequency
   converges at second order toward

   .. math::

       \\omega_{\\text{TE}_{101}} = c \\pi \\sqrt{\\frac{1}{a^2} + \\frac{1}{d^2}}.

2. **Circular cylinder via cube-to-disk elliptic map** (`cylinder_nodes`):
   Maps logical coordinates ``(u, v) in [-1, 1]^2`` and ``z in [0, L]`` to a
   circular cylinder of radius ``a`` and length ``L`` via

   .. math::

       x(u, v, z) = a u \\sqrt{1 - \\frac{v^2}{2}}, \\qquad
       y(u, v, z) = a v \\sqrt{1 - \\frac{u^2}{2}}, \\qquad
       z(u, v, z) = z.

   On ``u = +-1`` or ``v = +-1``, ``x^2 + y^2 = a^2 (1 - (1 - u^2)(1 - v^2)) = a^2``,
   so all four lateral logical faces map onto the cylinder wall ``r = a``.

   **Corner Jacobian derivation**:
   The partial derivatives in ``(u, v)`` are

   .. math::

       \\frac{\\partial x}{\\partial u} = a \\sqrt{1 - \\frac{v^2}{2}}, \\quad
       \\frac{\\partial x}{\\partial v} = -\\frac{a u v}{2 \\sqrt{1 - v^2 / 2}}, \\quad
       \\frac{\\partial y}{\\partial u} = -\\frac{a u v}{2 \\sqrt{1 - u^2 / 2}}, \\quad
       \\frac{\\partial y}{\\partial v} = a \\sqrt{1 - \\frac{u^2}{2}},

   giving the continuous Jacobian determinant

   .. math::

       J_{\\text{cont}}(u, v)
       = \\frac{\\partial(x, y, z)}{\\partial(u, v, z)}
       = a^2 \\frac{1 - \\frac{1}{2}(u^2 + v^2)}
                  {\\sqrt{\\left(1 - \\frac{u^2}{2}\\right)\\left(1 - \\frac{v^2}{2}\\right)}}.

   At the four logical corners ``(u, v) = (+-1, +-1)``,
   ``1 - (u^2 + v^2)/2 = 0`` while the denominator is ``1/2``, so
   ``J_cont(+-1, +-1) = 0``: the two logical boundary segments meet along the
   circle ``r = a`` at ``(+-a / sqrt(2), +-a / sqrt(2))`` with interior angle
   ``pi`` (colinear tangent vectors).

   On a finite ``Nx x Ny x Nz`` grid with ``du = 2 / Nx``, ``dv = 2 / Ny``,
   ``dz = L / Nz``, the trilinear `HexBlock3D` corner Jacobian at
   ``(u, v) = (+-1, +-1)`` is formed from the finite-difference secant chords
   along the boundary,

   .. math::

       \\Delta \\mathbf{x}_\\xi = a \\begin{pmatrix}
           \\Delta u / \\sqrt{2} \\\\ -\\delta(\\Delta u) \\\\ 0
       \\end{pmatrix}, \\qquad
       \\Delta \\mathbf{x}_\\eta = a \\begin{pmatrix}
           -\\delta(\\Delta v) \\\\ \\Delta v / \\sqrt{2} \\\\ 0
       \\end{pmatrix}, \\qquad
       \\Delta \\mathbf{x}_\\zeta = \\begin{pmatrix}
           0 \\\\ 0 \\\\ \\Delta z
       \\end{pmatrix},

   where ``delta(h) = sqrt(1 - (1 - h)^2 / 2) - 1 / sqrt(2)``. Their triple
   product gives the exact minimum discrete corner Jacobian over the block:

   .. math::

       J_{\\text{corner, min}}
       = a^2 \\Delta z \\left(
           \\frac{\\Delta u \\, \\Delta v}{2}
           - \\delta(\\Delta u) \\, \\delta(\\Delta v)
         \\right)
       = a^2 \\Delta z \\left(
           \\frac{3}{4} \\Delta u^3 + O(\\Delta u^4)
         \\right) > 0 \\quad (\\Delta u = \\Delta v).

   Because the secant chords subtend an angle strictly less than ``pi`` for any
   finite ``Nx, Ny``, ``J_corner,min > 0`` on every finite mesh, while the
   normalised corner determinant
   ``|det| / (||dx_xi|| ||dx_eta|| ||dx_zeta||)`` scales as ``O(1 / N) -> 0``
   toward the continuous corner value ``0``. For the ``TM010`` mode
   (``omega = c x_01 / a`` with ``J_0(x_01) = 0``), ``E_z`` vanishes on
   ``r = a`` including the singular corners and ``E_x = E_y = B_z = 0``, so
   second-order convergence is maintained. For the ``TE111`` mode
   (``omega = c sqrt((x'_11 / a)^2 + (pi / L)^2)`` with ``J'_1(x'_11) = 0``),
   ``E_r(a) != 0`` on the wall and at the singular corners, where the nearly
   colinear transverse dual/primal face vectors reduce the asymptotic convergence
   rate.

3. **Sphere via 3D elliptic cube-to-ball map** (`sphere_nodes`):
   Maps ``(u, v, w) in [-1, 1]^3`` to a ball of radius ``a`` via

   .. math::

       x &= a u \\sqrt{1 - \\frac{v^2}{2} - \\frac{w^2}{2} + \\frac{v^2 w^2}{3}}, \\\\
       y &= a v \\sqrt{1 - \\frac{w^2}{2} - \\frac{u^2}{2} + \\frac{w^2 u^2}{3}}, \\\\
       z &= a w \\sqrt{1 - \\frac{u^2}{2} - \\frac{v^2}{2} + \\frac{u^2 v^2}{3}},

   satisfying ``x^2 + y^2 + z^2 = a^2 [1 - (1 - u^2)(1 - v^2)(1 - w^2)] = a^2``
   on all six faces of ``[-1, 1]^3``. At the eight cube corners
   ``(|u|, |v|, |w|) = (1, 1, 1)`` and along the twelve cube edges, the
   continuous Jacobian vanishes because the logical boundary faces meet
   tangentially on the sphere ``r = a``; on a finite grid the normalised
   corner determinant remains positive and scales as ``O(1 / N^2)``.

   For a PEC sphere of radius ``a``, the resonant wavenumbers ``k = omega / c``
   with ``x = k a`` and angular degree ``l >= 1`` satisfy the radial
   characteristic equations

   .. math::

       \\frac{d}{dx}\\left[x j_l(x)\\right] = j_l(x) + x j_l'(x) = 0
       \\qquad \\text{and} \\qquad
       j_l(x) = 0,

   whose lowest ``l = 1`` roots are found by bracketing with
   `scipy.special.spherical_jn` and `scipy.optimize.brentq`. The ``j_1(x) = 0``
   mode has purely tangential electric field vanishing everywhere on ``r = a``
   (including at the eight singular corners) and converges at second order,
   whereas the ``d/dx[x j_1(x)] = 0`` mode has ``E_r(a) != 0`` at the boundary
   singularities.
"""

import numpy as np
from scipy.constants import c as C0
from scipy.optimize import brentq
from scipy.special import jn_zeros, jnp_zeros, spherical_jn

__all__ = [
    "box_te101_omega",
    "cylinder_continuous_jacobian",
    "cylinder_corner_jacobian_analytic",
    "cylinder_nodes",
    "cylinder_te111_omega",
    "cylinder_tm010_omega",
    "min_normalised_corner_det",
    "perturbed_box_nodes",
    "sphere_bessel_residual",
    "sphere_lowest_omegas",
    "sphere_lowest_roots",
    "sphere_nodes",
    "sphere_riccati_bessel_deriv_residual",
]


def _validate_counts(nx, ny, nz):
    nx, ny, nz = int(nx), int(ny), int(nz)
    if min(nx, ny, nz) < 1:
        raise ValueError(f"nx, ny, nz must be >= 1, got {(nx, ny, nz)}")
    return nx, ny, nz


def perturbed_box_nodes(nx, ny, nz, a=1.0, b=0.7, d=0.8, amp=0.3):
    """
    Generate node coordinates for a Cartesian box ``[0, a] x [0, b] x [0, d]``
    with a smooth sinusoidal interior displacement vanishing on all walls.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells along each logical axis.
    a, b, d : float, optional
        Box side lengths along x, y, and z [m].
    amp : float, optional
        Dimensionless displacement amplitude as a fraction of the local cell
        spacing ``(h_x, h_y, h_z)`` (default 0.3).

    Returns
    -------
    X : ndarray of shape (nx + 1, ny + 1, nz + 1, 3)
        Node coordinates [m].
    """
    nx, ny, nz = _validate_counts(nx, ny, nz)
    a, b, d = float(a), float(b), float(d)
    amp = float(amp)
    if min(a, b, d) <= 0.0:
        raise ValueError("box side lengths a, b, d must be positive")

    x0 = np.linspace(0.0, a, nx + 1)
    y0 = np.linspace(0.0, b, ny + 1)
    z0 = np.linspace(0.0, d, nz + 1)
    X = np.stack(np.meshgrid(x0, y0, z0, indexing="ij"), axis=-1)

    hx, hy, hz = a / nx, b / ny, d / nz
    env = (
        np.sin(np.pi * X[..., 0] / a)
        * np.sin(np.pi * X[..., 1] / b)
        * np.sin(np.pi * X[..., 2] / d)
    )
    X[..., 0] += amp * hx * env
    X[..., 1] += amp * hy * env
    X[..., 2] += amp * hz * env
    return X


def cylinder_nodes(nx, ny, nz, a=1.0, length=1.2):
    """
    Generate node coordinates for a circular cylinder of radius ``a`` and
    length ``length`` via the 2D elliptic cube-to-disk mapping extruded in ``z``.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells along logical axes ``(u, v, z)``.
    a : float, optional
        Cylinder radius [m] (default 1.0).
    length : float, optional
        Cylinder axial length ``L`` [m] (default 1.2).

    Returns
    -------
    X : ndarray of shape (nx + 1, ny + 1, nz + 1, 3)
        Node coordinates [m].
    """
    nx, ny, nz = _validate_counts(nx, ny, nz)
    a, length = float(a), float(length)
    if a <= 0.0 or length <= 0.0:
        raise ValueError("cylinder radius a and length must be positive")

    u = np.linspace(-1.0, 1.0, nx + 1)
    v = np.linspace(-1.0, 1.0, ny + 1)
    z = np.linspace(0.0, length, nz + 1)
    U, V, Z = np.meshgrid(u, v, z, indexing="ij")
    X = np.empty((nx + 1, ny + 1, nz + 1, 3), dtype=float)
    X[..., 0] = a * U * np.sqrt(1.0 - 0.5 * V * V)
    X[..., 1] = a * V * np.sqrt(1.0 - 0.5 * U * U)
    X[..., 2] = Z
    return X


def cylinder_continuous_jacobian(u, v, a=1.0):
    """
    Continuous Jacobian determinant ``d(x, y, z) / d(u, v, z)`` [m^2] of the
    elliptic cube-to-disk mapping ``x = a u sqrt(1 - v^2 / 2)``,
    ``y = a v sqrt(1 - u^2 / 2)``, ``z = z`` for ``(u, v) in [-1, 1]^2``.

    At the four corners ``(u, v) = (+-1, +-1)``, ``1 - (u^2 + v^2) / 2 = 0``
    and the continuous Jacobian vanishes identically.

    Parameters
    ----------
    u, v : array_like
        Logical coordinates in ``[-1, 1]``.
    a : float, optional
        Cylinder radius [m] (default 1.0).

    Returns
    -------
    ndarray or float
        Continuous Jacobian determinant [m^2].
    """
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    a = float(a)
    num = 1.0 - 0.5 * (u_arr * u_arr + v_arr * v_arr)
    den = np.sqrt((1.0 - 0.5 * u_arr * u_arr) * (1.0 - 0.5 * v_arr * v_arr))
    out = (a * a) * (num / den)
    return float(out) if out.ndim == 0 else out


def cylinder_corner_jacobian_analytic(nx, ny, nz, a=1.0, length=1.2):
    """
    Closed-form minimum trilinear corner Jacobian [m^3] of
    ``HexBlock3D(cylinder_nodes(nx, ny, nz, a=a, length=length))``, attained
    at the four singular corners ``(u, v) = (+-1, +-1)``:

    .. math::

        J_{\\text{corner, min}} = a^2 \\Delta z \\left(
            \\frac{\\Delta u \\, \\Delta v}{2}
            - \\delta(\\Delta u) \\, \\delta(\\Delta v)
        \\right),

    where ``du = 2 / nx``, ``dv = 2 / ny``, ``dz = length / nz``, and
    ``delta(h) = sqrt(1 - (1 - h)^2 / 2) - 1 / sqrt(2)``.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells along ``(u, v, z)``.
    a : float, optional
        Cylinder radius [m] (default 1.0).
    length : float, optional
        Cylinder axial length [m] (default 1.2).

    Returns
    -------
    float
        Analytic minimum trilinear corner Jacobian [m^3].
    """
    nx, ny, nz = _validate_counts(nx, ny, nz)
    a, length = float(a), float(length)
    du = 2.0 / nx
    dv = 2.0 / ny
    dz = length / nz
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    delta_u = np.sqrt(1.0 - 0.5 * (1.0 - du) ** 2) - inv_sqrt2
    delta_v = np.sqrt(1.0 - 0.5 * (1.0 - dv) ** 2) - inv_sqrt2
    return float((a * a) * dz * (0.5 * du * dv - delta_u * delta_v))


def sphere_nodes(nx, ny, nz, a=1.0):
    """
    Generate node coordinates for a sphere of radius ``a`` via the 3D elliptic
    cube-to-ball mapping on ``[-1, 1]^3``.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells along logical axes ``(u, v, w)``.
    a : float, optional
        Sphere radius [m] (default 1.0).

    Returns
    -------
    X : ndarray of shape (nx + 1, ny + 1, nz + 1, 3)
        Node coordinates [m].
    """
    nx, ny, nz = _validate_counts(nx, ny, nz)
    a = float(a)
    if a <= 0.0:
        raise ValueError("sphere radius a must be positive")

    u = np.linspace(-1.0, 1.0, nx + 1)
    v = np.linspace(-1.0, 1.0, ny + 1)
    w = np.linspace(-1.0, 1.0, nz + 1)
    U, V, W = np.meshgrid(u, v, w, indexing="ij")
    u2, v2, w2 = U * U, V * V, W * W

    X = np.empty((nx + 1, ny + 1, nz + 1, 3), dtype=float)
    X[..., 0] = (
        a * U * np.sqrt(np.maximum(1.0 - 0.5 * v2 - 0.5 * w2 + (v2 * w2) / 3.0, 0.0))
    )
    X[..., 1] = (
        a * V * np.sqrt(np.maximum(1.0 - 0.5 * w2 - 0.5 * u2 + (w2 * u2) / 3.0, 0.0))
    )
    X[..., 2] = (
        a * W * np.sqrt(np.maximum(1.0 - 0.5 * u2 - 0.5 * v2 + (u2 * v2) / 3.0, 0.0))
    )
    return X


def min_normalised_corner_det(block):
    """
    Minimum over all cells and all 8 cell corners of the normalised 3x3
    corner edge-vector determinant

    .. math::

        \\frac{|\\det(\\Delta \\mathbf{x}_\\xi, \\Delta \\mathbf{x}_\\eta, \\Delta \\mathbf{x}_\\zeta)|}
             {\\|\\Delta \\mathbf{x}_\\xi\\| \\, \\|\\Delta \\mathbf{x}_\\eta\\| \\, \\|\\Delta \\mathbf{x}_\\zeta\\|}.

    Equals ``1.0`` for an orthogonal Cartesian mesh and approaches ``0.0``
    at corners where the three logical edge directions become coplanar.

    Parameters
    ----------
    block : HexBlock3D
        Structured hexahedral block.

    Returns
    -------
    float
        Minimum normalised corner determinant in ``(0, 1]``.
    """
    X = block.X
    Nx, Ny, Nz = block.nc
    min_val = np.inf
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                dxi = X[1:, b : Ny + b, c : Nz + c] - X[:-1, b : Ny + b, c : Nz + c]
                deta = X[a : Nx + a, 1:, c : Nz + c] - X[a : Nx + a, :-1, c : Nz + c]
                dzeta = X[a : Nx + a, b : Ny + b, 1:] - X[a : Nx + a, b : Ny + b, :-1]
                det = np.abs(block.corner_jacobians[:, :, :, a, b, c])
                denom = (
                    np.linalg.norm(dxi, axis=-1)
                    * np.linalg.norm(deta, axis=-1)
                    * np.linalg.norm(dzeta, axis=-1)
                )
                min_val = min(min_val, float(np.min(det / denom)))
    return float(min_val)


def box_te101_omega(a=1.0, d=0.8):
    """
    Analytic ``TE101`` angular frequency ``omega = c * pi * sqrt(1/a^2 + 1/d^2)``
    [rad/s] of a rectangular PEC cavity with sides ``a`` and ``d``.

    Parameters
    ----------
    a, d : float, optional
        Side lengths [m].

    Returns
    -------
    float
        Angular frequency [rad/s].
    """
    return float(C0 * np.pi * np.hypot(1.0 / float(a), 1.0 / float(d)))


def cylinder_tm010_omega(a=1.0):
    """
    Analytic ``TM010`` angular frequency ``omega = c * x_01 / a`` [rad/s] of a
    circular PEC cylinder of radius ``a``, with ``x_01`` the first positive
    zero of ``J_0`` computed via `scipy.special.jn_zeros`.

    Parameters
    ----------
    a : float, optional
        Cylinder radius [m] (default 1.0).

    Returns
    -------
    float
        Angular frequency [rad/s].
    """
    x01 = float(jn_zeros(0, 1)[0])
    return float(C0 * x01 / float(a))


def cylinder_te111_omega(a=1.0, length=1.2):
    """
    Analytic ``TE111`` angular frequency
    ``omega = c * sqrt((x'_11 / a)^2 + (pi / L)^2)`` [rad/s] of a circular PEC
    cylinder of radius ``a`` and length ``length``, with ``x'_11`` the first
    positive zero of ``J'_1`` computed via `scipy.special.jnp_zeros`.

    Parameters
    ----------
    a : float, optional
        Cylinder radius [m] (default 1.0).
    length : float, optional
        Cylinder axial length ``L`` [m] (default 1.2).

    Returns
    -------
    float
        Angular frequency [rad/s].
    """
    x11p = float(jnp_zeros(1, 1)[0])
    return float(C0 * np.hypot(x11p / float(a), np.pi / float(length)))


def sphere_riccati_bessel_deriv_residual(x, l=1):
    """
    Characteristic residual ``d/dx [x j_l(x)] = j_l(x) + x j'_l(x)`` for a PEC
    sphere resonance.

    Parameters
    ----------
    x : float or array_like
        Dimensionless radial wavenumber ``k a``.
    l : int, optional
        Spherical harmonic degree ``l >= 1`` (default 1).

    Returns
    -------
    float or ndarray
        Value of ``j_l(x) + x j'_l(x)``.
    """
    return spherical_jn(l, x) + x * spherical_jn(l, x, derivative=True)


def sphere_bessel_residual(x, l=1):
    """
    Characteristic residual ``j_l(x)`` for a PEC sphere resonance.

    Parameters
    ----------
    x : float or array_like
        Dimensionless radial wavenumber ``k a``.
    l : int, optional
        Spherical harmonic degree ``l >= 1`` (default 1).

    Returns
    -------
    float or ndarray
        Value of ``j_l(x)``.
    """
    return spherical_jn(l, x)


def sphere_lowest_roots(l=1):
    """
    Compute the lowest positive roots ``(x_d_xjl, x_jl)`` of
    ``d/dx [x j_l(x)] = 0`` and ``j_l(x) = 0`` by bracketing with
    `scipy.optimize.brentq` and `scipy.special.spherical_jn`.

    For ``l >= 1``, ``x j_l(x)`` starts at ``0`` at ``x = 0``, increases to its
    first positive extremum ``x_d_xjl in (l, l + pi)``, and then crosses zero at
    ``x_jl in (x_d_xjl, l + 2 * pi)``.

    Parameters
    ----------
    l : int, optional
        Spherical harmonic degree ``l >= 1`` (default 1).

    Returns
    -------
    x_d_xjl : float
        Lowest positive root of ``d/dx [x j_l(x)] = 0``.
    x_jl : float
        Lowest positive root of ``j_l(x) = 0``.
    """
    l = int(l)
    if l < 1:
        raise ValueError(f"spherical degree l must be >= 1, got {l}")
    x_d_xjl = float(
        brentq(
            sphere_riccati_bessel_deriv_residual,
            float(l),
            float(l) + np.pi,
            args=(l,),
            xtol=1e-15,
            rtol=1e-15,
        )
    )
    x_jl = float(
        brentq(
            sphere_bessel_residual,
            x_d_xjl,
            float(l) + 2.0 * np.pi,
            args=(l,),
            xtol=1e-15,
            rtol=1e-15,
        )
    )
    return x_d_xjl, x_jl


def sphere_lowest_omegas(a=1.0, l=1):
    """
    Lowest two PEC sphere resonant angular frequencies
    ``(omega_d_xjl, omega_jl) = (c * x_d_xjl / a, c * x_jl / a)`` [rad/s] for
    degree ``l``.

    Parameters
    ----------
    a : float, optional
        Sphere radius [m] (default 1.0).
    l : int, optional
        Spherical harmonic degree ``l >= 1`` (default 1).

    Returns
    -------
    omega_d_xjl : float
        Angular frequency [rad/s] for the lowest root of ``d/dx [x j_l(x)] = 0``.
    omega_jl : float
        Angular frequency [rad/s] for the lowest root of ``j_l(x) = 0``.
    """
    a = float(a)
    if a <= 0.0:
        raise ValueError("sphere radius a must be positive")
    x1, x2 = sphere_lowest_roots(l=l)
    return float(C0 * x1 / a), float(C0 * x2 / a)
