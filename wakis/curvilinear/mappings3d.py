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
    "butterfly_disk_nodes",
    "cubed_sphere_ball_nodes",
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


def butterfly_disk_nodes(n_core, n_arc=None, nz=None, a=1.0, length=1.2):
    """
    Generate a 5-block butterfly (O-grid) conforming hexahedral mesh of a
    circular cylinder of radius ``a`` [m] and axial length ``length`` [m].

    Transfinite blend derivation
    ----------------------------
    The 2D cross-section of radius ``a`` [m] is partitioned into a central
    Cartesian square core ``[-b, b]^2`` of half-width
    ``b = alpha * a / sqrt(2)`` [m] (with ``alpha = 0.65``) and four
    surrounding curved blocks (East ``+x``, North ``+y``, West ``-x``, South
    ``-y``) whose outer arc corners lie at ``(+-a / sqrt(2), +-a / sqrt(2))`` [m]
    on the circle ``r = a`` along the four rays ``theta = +-pi/4, +-3*pi/4``.

    For each curved block ``m in {0, 1, 2, 3}`` with azimuthal logical
    coordinate ``s in [-1, 1]`` and radial blend coordinate ``t in [0, 1]``,
    the inner curve ``x_in,m(s)`` [m] is the corresponding side of the square
    ``[-b, b]^2`` (e.g. ``x_in,0(s) = (b, b * s)`` [m] on the East block) and
    the outer curve is the circular arc of radius ``a`` [m],
    ``x_out,m(s) = (a * cos(theta_m(s)), a * sin(theta_m(s)))`` [m] with
    ``theta_m(s) = m * pi / 2 + (pi / 4) * s`` [rad]. Transfinite linear
    blending in ``t`` extruded along ``z in [0, length]`` [m] gives

    .. math::

        \\mathbf{x}_m(s, t, z) = (1 - t) \\, \\mathbf{x}_{\\text{in}, m}(s)
            + t \\, \\mathbf{x}_{\\text{out}, m}(s) + (0, 0, z) \\quad [\\text{m}].

    Along the four inter-block seams ``s = +-1``, adjacent curved blocks share
    the straight radial line segment connecting ``(+-b, +-b, z)`` to
    ``(+-a / sqrt(2), +-a / sqrt(2), z)``, avoiding any centre singularity or
    boundary secant degeneracy (minimum scaled corner Jacobian
    ``1 / sqrt(2)`` independent of resolution).

    Boundary patches ``0..3`` label the four curved lateral arc walls and
    patches ``4, 5`` label the bottom (``z = 0``) and top (``z = length``) flat
    end caps. ``sharp_nodes`` marks only the top and bottom circular rims
    ``(r = a, z in {0, length})`` where the lateral wall meets the end caps at
    ``90`` degrees; the four axial arc junctions on ``r = a`` are smooth.

    Parameters
    ----------
    n_core : int
        Number of cells along each side of the central square core.
    n_arc : int, optional
        Number of radial cells in each of the 4 curved arc blocks (defaults to
        ``n_core``).
    nz : int, optional
        Number of axial cells along ``z`` (defaults to ``n_core``).
    a : float, optional
        Cylinder radius [m] (default 1.0).
    length : float, optional
        Cylinder axial length [m] (default 1.2).

    Returns
    -------
    HexMesh
        Conforming hexahedral mesh (also unpackable as
        ``nodes, hexes, boundary_patch, sharp_nodes``).
    """
    from .hexmesh import HexMesh

    if n_arc is None:
        n_arc = n_core
    if nz is None:
        nz = n_core
    n_core, n_arc, nz = _validate_counts(n_core, n_arc, nz)
    a, length = float(a), float(length)
    if a <= 0.0 or length <= 0.0:
        raise ValueError("cylinder radius a and length must be positive")

    alpha = 0.65
    b = alpha * a / np.sqrt(2.0)

    u = np.linspace(-1.0, 1.0, n_core + 1)
    U, V = np.meshgrid(u, u, indexing="ij")
    xy_core = np.stack([b * U, b * V], axis=-1).reshape(-1, 2)
    core_idx = np.arange((n_core + 1) ** 2, dtype=np.int64).reshape(
        n_core + 1, n_core + 1
    )

    ring0_idx = np.concatenate(
        [
            core_idx[n_core, :n_core],
            core_idx[n_core:0:-1, n_core],
            core_idx[0, n_core:0:-1],
            core_idx[:n_core, 0],
        ]
    )
    xy_in = xy_core[ring0_idx]

    s_seg = np.linspace(-1.0, 1.0, n_core + 1)[:-1]
    thetas = np.concatenate(
        [
            0.25 * np.pi * s_seg,
            0.5 * np.pi + 0.25 * np.pi * s_seg,
            np.pi + 0.25 * np.pi * s_seg,
            -0.5 * np.pi + 0.25 * np.pi * s_seg,
        ]
    )
    xy_out = a * np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)

    n_perim = 4 * n_core
    xy_rings = []
    ring_idx_list = [ring0_idx]
    next_id = int(xy_core.shape[0])
    for r in range(1, n_arc + 1):
        t = r / n_arc
        xy_rings.append((1.0 - t) * xy_in + t * xy_out)
        ring_idx_list.append(
            np.arange(next_id, next_id + n_perim, dtype=np.int64)
        )
        next_id += n_perim

    xy_2d = np.concatenate([xy_core] + xy_rings, axis=0)
    n_2d = int(xy_2d.shape[0])

    q_core = np.stack(
        [
            core_idx[:-1, :-1].ravel(),
            core_idx[1:, :-1].ravel(),
            core_idx[1:, 1:].ravel(),
            core_idx[:-1, 1:].ravel(),
        ],
        axis=-1,
    )

    q_arcs = []
    q_arc_sec = []
    for r in range(n_arc):
        r0 = ring_idx_list[r]
        r1 = ring_idx_list[r + 1]
        for p in range(n_perim):
            pn = (p + 1) % n_perim
            q_arcs.append([r0[p], r1[p], r1[pn], r0[pn]])
            q_arc_sec.append((r == n_arc - 1, p // n_core))
    q_arcs = np.asarray(q_arcs, dtype=np.int64)
    quads_2d = np.concatenate([q_core, q_arcs], axis=0)
    n_q2d = int(quads_2d.shape[0])
    n_core_q = int(q_core.shape[0])

    z = np.linspace(0.0, length, nz + 1)
    nodes = np.empty(((nz + 1) * n_2d, 3), dtype=float)
    for k in range(nz + 1):
        nodes[k * n_2d : (k + 1) * n_2d, :2] = xy_2d
        nodes[k * n_2d : (k + 1) * n_2d, 2] = z[k]

    hexes = np.empty((nz * n_q2d, 8), dtype=np.int64)
    local_patch = np.full((nz * n_q2d, 3, 2), -1, dtype=np.int64)
    for k in range(nz):
        h0 = k * n_q2d
        h1 = (k + 1) * n_q2d
        hexes[h0:h1, :4] = quads_2d + k * n_2d
        hexes[h0:h1, 4:] = quads_2d + (k + 1) * n_2d
        if k == 0:
            local_patch[h0:h1, 2, 0] = 4
        if k == nz - 1:
            local_patch[h0:h1, 2, 1] = 5
        for idx_a, (is_outer, sec) in enumerate(q_arc_sec):
            if is_outer:
                local_patch[h0 + n_core_q + idx_a, 0, 1] = sec

    sharp_nodes = np.zeros((nz + 1) * n_2d, dtype=bool)
    outer_ring_2d = ring_idx_list[-1]
    sharp_nodes[outer_ring_2d] = True
    sharp_nodes[nz * n_2d + outer_ring_2d] = True

    return HexMesh(
        nodes, hexes, boundary_patch=local_patch, sharp_nodes=sharp_nodes
    )


def cubed_sphere_ball_nodes(n, a=1.0):
    """
    Generate a 7-block cubed-sphere conforming hexahedral mesh of a ball of
    radius ``a`` [m] consisting of an ``n x n x n`` Cartesian cube core and
    six ``n x n x n`` radially blended gnomonic shells.

    Equidistant gnomonic projection and radial blend derivation
    -----------------------------------------------------------
    Let ``b = alpha * a / sqrt(3)`` [m] (with ``alpha = 0.56``) be the
    half-width of the inner Cartesian cube core ``[-b, b]^3``. Each of the six
    cube faces ``m in {0, ..., 5}`` (``+x, -x, +y, -y, +z, -z``) is
    parameterised by uniform logical coordinates ``(u, v) in [-1, 1]^2`` with
    cube-surface point ``p_m(u, v) in [-1, 1]^3`` satisfying
    ``max_i |p_{m, i}| = 1``.

    Under the equidistant gnomonic projection, ``p_m(u, v)`` is projected
    radially onto the sphere of radius ``a`` [m]:

    .. math::

        \\mathbf{x}_{\\text{in}, m}(u, v) = b \\, \\mathbf{p}_m(u, v) \\; [\\text{m}], \\qquad
        \\mathbf{x}_{\\text{out}, m}(u, v)
        = a \\, \\frac{\\mathbf{p}_m(u, v)}{\\|\\mathbf{p}_m(u, v)\\|}
        = \\frac{a}{\\sqrt{1 + u^2 + v^2}} \\, \\mathbf{p}_m(u, v) \\; [\\text{m}].

    Because ``x_in,m(u, v)`` and ``x_out,m(u, v)`` are strictly collinear along
    the ray ``p_m(u, v)`` for every ``(u, v) in [-1, 1]^2``, linear radial
    blending in ``t in [0, 1]`` yields

    .. math::

        \\mathbf{x}_m(u, v, t)
        = \\left[(1 - t) \\, b + t \\, \\frac{a}{\\sqrt{1 + u^2 + v^2}}\\right]
          \\mathbf{p}_m(u, v) \\quad [\\text{m}],

    whose radial edges remain straight rays through the origin and whose
    minimum scaled corner Jacobian equals ``1 / sqrt(3)`` at the eight inner
    cube corners ``(|u|, |v|, t) = (1, 1, 0)`` for all ``n`` (with non-growing
    corner reconstruction condition number as ``n`` increases, unlike the
    equiangular gnomonic projection where ``sec^2(pi u / 4)`` stretches the
    inner-to-outer transverse aspect ratio with ``n``).

    The outermost shell layer (``t = 1``) forms the spherical boundary with
    ``boundary_patch`` labels ``0..5`` for the six shells. Only the 8
    cube-corner nodes on ``r = a`` (where three shells meet at ``120`` degrees)
    are marked in ``sharp_nodes`` (24 sharp octants); the 12 cube-edge seams
    between shells are smooth.

    Parameters
    ----------
    n : int
        Number of cells along each cube edge and along the radial shell
        direction.
    a : float, optional
        Sphere radius [m] (default 1.0).

    Returns
    -------
    HexMesh
        Conforming hexahedral mesh (also unpackable as
        ``nodes, hexes, boundary_patch, sharp_nodes``).
    """
    from .hexmesh import HexMesh

    n, _, _ = _validate_counts(n, n, n)
    a = float(a)
    if a <= 0.0:
        raise ValueError("sphere radius a must be positive")

    alpha = 0.56
    b = alpha * a / np.sqrt(3.0)

    u = np.linspace(-1.0, 1.0, n + 1)
    U, V, W = np.meshgrid(u, u, u, indexing="ij")
    xyz_core = b * np.stack([U, V, W], axis=-1).reshape(-1, 3)
    core_idx = np.arange((n + 1) ** 3, dtype=np.int64).reshape(
        n + 1, n + 1, n + 1
    )

    h_core = np.stack(
        [
            core_idx[:-1, :-1, :-1].ravel(),
            core_idx[1:, :-1, :-1].ravel(),
            core_idx[1:, 1:, :-1].ravel(),
            core_idx[:-1, 1:, :-1].ravel(),
            core_idx[:-1, :-1, 1:].ravel(),
            core_idx[1:, :-1, 1:].ravel(),
            core_idx[1:, 1:, 1:].ravel(),
            core_idx[:-1, 1:, 1:].ravel(),
        ],
        axis=-1,
    )

    is_surf = np.zeros((n + 1, n + 1, n + 1), dtype=bool)
    is_surf[0, :, :] = is_surf[-1, :, :] = True
    is_surf[:, 0, :] = is_surf[:, -1, :] = True
    is_surf[:, :, 0] = is_surf[:, :, -1] = True
    surf_core_ids = core_idx[is_surf]
    n_surf = int(surf_core_ids.size)
    surf_map = -np.ones((n + 1) ** 3, dtype=np.int64)
    surf_map[surf_core_ids] = np.arange(n_surf, dtype=np.int64)

    p_surf = xyz_core[surf_core_ids] / b
    p_norm = np.linalg.norm(p_surf, axis=1, keepdims=True)
    xyz_out = a * (p_surf / p_norm)
    xyz_in = xyz_core[surf_core_ids]

    shell_layers = []
    layer_idx = [surf_core_ids]
    next_id = int(xyz_core.shape[0])
    for r in range(1, n + 1):
        t = r / n
        shell_layers.append((1.0 - t) * xyz_in + t * xyz_out)
        layer_idx.append(np.arange(next_id, next_id + n_surf, dtype=np.int64))
        next_id += n_surf

    nodes = np.concatenate([xyz_core] + shell_layers, axis=0)

    specs = [
        (
            0,
            core_idx[n, :-1, :-1],
            core_idx[n, 1:, :-1],
            core_idx[n, 1:, 1:],
            core_idx[n, :-1, 1:],
        ),
        (
            1,
            core_idx[0, :-1, :-1],
            core_idx[0, :-1, 1:],
            core_idx[0, 1:, 1:],
            core_idx[0, 1:, :-1],
        ),
        (
            2,
            core_idx[:-1, n, :-1],
            core_idx[:-1, n, 1:],
            core_idx[1:, n, 1:],
            core_idx[1:, n, :-1],
        ),
        (
            3,
            core_idx[:-1, 0, :-1],
            core_idx[1:, 0, :-1],
            core_idx[1:, 0, 1:],
            core_idx[:-1, 0, 1:],
        ),
        (
            4,
            core_idx[:-1, :-1, n],
            core_idx[1:, :-1, n],
            core_idx[1:, 1:, n],
            core_idx[:-1, 1:, n],
        ),
        (
            5,
            core_idx[:-1, :-1, 0],
            core_idx[:-1, 1:, 0],
            core_idx[1:, 1:, 0],
            core_idx[1:, :-1, 0],
        ),
    ]
    face_quads = []
    face_patches = []
    for p_id, c0, c1, c2, c3 in specs:
        q = np.stack(
            [
                surf_map[c0.ravel()],
                surf_map[c1.ravel()],
                surf_map[c2.ravel()],
                surf_map[c3.ravel()],
            ],
            axis=-1,
        )
        face_quads.append(q)
        face_patches.append(np.full(q.shape[0], p_id, dtype=np.int64))
    face_quads = np.concatenate(face_quads, axis=0)
    face_patches = np.concatenate(face_patches, axis=0)
    n_fq = int(face_quads.shape[0])

    h_shells = np.empty((n * n_fq, 8), dtype=np.int64)
    for r in range(n):
        L0 = layer_idx[r]
        L1 = layer_idx[r + 1]
        s0 = r * n_fq
        s1 = (r + 1) * n_fq
        h_shells[s0:s1, :4] = L0[face_quads]
        h_shells[s0:s1, 4:] = L1[face_quads]

    hexes = np.concatenate([h_core, h_shells], axis=0)
    local_patch = np.full((hexes.shape[0], 3, 2), -1, dtype=np.int64)
    outer_start = h_core.shape[0] + (n - 1) * n_fq
    local_patch[outer_start:, 2, 1] = face_patches

    is_corner_surf = (
        (np.abs(p_surf[:, 0]) == 1.0)
        & (np.abs(p_surf[:, 1]) == 1.0)
        & (np.abs(p_surf[:, 2]) == 1.0)
    )
    sharp_nodes = np.zeros(nodes.shape[0], dtype=bool)
    sharp_nodes[layer_idx[-1][is_corner_surf]] = True

    return HexMesh(
        nodes, hexes, boundary_patch=local_patch, sharp_nodes=sharp_nodes
    )
