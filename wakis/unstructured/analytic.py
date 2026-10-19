# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Exact resonances of layered circular and spherical cavities (references for
validating the unstructured FIT solvers).

All functions return wavenumbers k = omega / c [1/m] of the vacuum region,
sorted ascending and repeated by their degeneracy (2 for angular orders
n >= 1 in 2D, 2 n + 1 in 3D). Field continuity conditions:

* Rod (radius b, eps) in a circular PEC cavity (radius a). TM (Ez):
  Ez and dEz/dr continuous, Ez(a) = 0. TE (Hz): Hz and (1/eps) dHz/dr
  continuous, dHz/dr(a) = 0.
* Sphere (radius b, eps) in a spherical PEC cavity (radius a), with radial
  functions z_n(k r) (spherical Bessel). TE (transverse electric, E
  tangential to spheres): f and (r f)' continuous, f(a) = 0. TM: f and
  (r f)' / eps continuous, (r f)'(a) = 0.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import iv, ivp, jv, jvp, spherical_jn, spherical_yn, yv, yvp


def _roots(func, kmax, n_scan=4000, kmin=None):
    kmin = kmin if kmin is not None else kmax * 1e-4
    ks = np.linspace(kmin, kmax, n_scan)
    vals = np.array([func(k) for k in ks])
    out = []
    for i in np.flatnonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0):
        r = brentq(func, ks[i], ks[i + 1], xtol=1e-14 * kmax)
        # Reject poles (sign changes through infinity)
        if abs(func(r)) < 1e-6 * (1 + np.max(np.abs(vals[np.isfinite(vals)]))):
            out.append(r)
    return out


def _collect(per_order, n_modes, dim):
    roots = []
    for n, rs in per_order:
        deg = (1 if n == 0 else 2) if dim == 2 else 2 * n + 1
        for r in rs:
            roots += [r] * deg
    return np.sort(roots)[:n_modes]


def rod_in_cavity(a, b, eps, pol, n_modes=6, kmax=None, n_max=8):
    """
    Wavenumbers of a dielectric rod (radius b, rel. permittivity eps) in a
    circular PEC cavity of radius a.

    Parameters
    ----------
    a, b : float
        Cavity and rod radii [m].
    eps : float or callable
        Rod permittivity, or a function k -> eps(k) (e.g. Drude) [-].
    pol : {"TM", "TE"}
    n_modes : int
        Number of modes returned (with degeneracy).

    Returns
    -------
    k : ndarray
        Vacuum wavenumbers [1/m].
    """
    eps_of = eps if callable(eps) else (lambda k: eps)
    kmax = kmax or 12 / a

    def det(k, n):
        e = eps_of(k)
        if pol == "TM":
            A, B = yv(n, k * a), -jv(n, k * a)
            p_in = 1.0
        else:
            A, B = yvp(n, k * a), -jvp(n, k * a)
            p_in = 1.0 / e
        f_out = A * jv(n, k * b) + B * yv(n, k * b)
        df_out = k * (A * jvp(n, k * b) + B * yvp(n, k * b))
        # Interior logarithmic derivative k1 J_n'(k1 b) / J_n(k1 b); for
        # eps < 0 with modified Bessel functions (kappa I_n' / I_n)
        if e > 0:
            k1 = k * np.sqrt(e)
            f_in, df_in = jv(n, k1 * b), k1 * jvp(n, k1 * b)
        else:
            kap = k * np.sqrt(-e)
            f_in, df_in = iv(n, kap * b), kap * ivp(n, kap * b)
        return p_in * df_in * f_out - f_in * df_out

    per = [(n, _roots(lambda k: det(k, n), kmax)) for n in range(n_max)]
    return _collect(per, n_modes, 2)


def drude_rod_in_cavity_te(a, b, k_p, n=1, kmax=None):
    """
    TE (Hz) resonances of order `n` of a lossless Drude rod,
    eps(k) = 1 - (k_p / k)^2, in a circular PEC cavity.

    Returns all roots with angular order `n` below `kmax` (default 2 k_p);
    the surface-plasmon resonances lie below k_p / sqrt(2) ... k_p.

    Returns
    -------
    k : ndarray
        Vacuum wavenumbers [1/m] (not repeated by degeneracy).
    """
    kmax = kmax or 2 * k_p

    def det(k):
        e = 1 - (k_p / k) ** 2
        if abs(e) < 1e-12:
            e = 1e-12
        A, B = yvp(n, k * a), -jvp(n, k * a)
        f_out = A * jv(n, k * b) + B * yv(n, k * b)
        df_out = k * (A * jvp(n, k * b) + B * yvp(n, k * b))
        if e > 0:
            k1 = k * np.sqrt(e)
            f_in, df_in = jv(n, k1 * b), k1 * jvp(n, k1 * b)
        else:
            kap = k * np.sqrt(-e)
            f_in, df_in = iv(n, kap * b), kap * ivp(n, kap * b)
        # Normalize by f_in to avoid exponential growth of I_n
        return (df_in / e) * f_out / f_in - df_out

    return np.array(_roots(det, kmax, n_scan=20000, kmin=0.05 * k_p))


def _riccati(n, x):
    """psi = x j_n(x) and its derivative; chi = x y_n(x) and derivative."""
    j, jp = spherical_jn(n, x), spherical_jn(n, x, derivative=True)
    y, yp = spherical_yn(n, x), spherical_yn(n, x, derivative=True)
    return x * j, j + x * jp, x * y, y + x * yp


def sphere_in_cavity(a, b, eps, mode, n_modes=10, kmax=None, n_max=8):
    """
    Wavenumbers of a dielectric sphere (radius b, eps) in a spherical PEC
    cavity (radius a). ``b = 0`` (or eps = 1) gives the empty PEC sphere,
    whose TE roots are j_n(k a) = 0 and TM roots (k a j_n(k a))' = 0.

    Parameters
    ----------
    a, b : float
        Cavity and sphere radii [m].
    eps : float
        Relative permittivity of the sphere.
    mode : {"TE", "TM", "all"}
        Transverse electric, transverse magnetic, or both merged.
    n_modes : int
        Number of modes returned (each order n repeated 2n + 1 times).

    Returns
    -------
    k : ndarray
        Vacuum wavenumbers [1/m].
    """
    if mode == "all":
        k = np.concatenate(
            [sphere_in_cavity(a, b, eps, m, n_modes, kmax, n_max) for m in ("TE", "TM")]
        )
        return np.sort(k)[:n_modes]
    kmax = kmax or 10 / a
    m = np.sqrt(eps)

    def det(k, n):
        # Riccati-Bessel form: u = r f; inside psi(k1 r), outside
        # A psi(k r) + B chi(k r)
        pa, dpa, ca, dca = _riccati(n, k * a)
        if mode == "TE":
            A, B = ca, -pa  # u(a) = 0
        else:
            A, B = dca, -dpa  # u'(a) = 0
        if b == 0:
            return A * 0 + (pa if mode == "TE" else dpa)
        pb, dpb, cb, dcb = _riccati(n, k * b)
        u_out = A * pb + B * cb
        du_out = k * (A * dpb + B * dcb)
        pi, dpi, _, _ = _riccati(n, m * k * b)
        u_in, du_in = pi, m * k * dpi
        # TE: u, u' continuous; TM: u, u'/eps continuous
        w = 1.0 if mode == "TE" else 1.0 / eps
        return w * du_in * u_out - u_in * du_out

    per = [(n, _roots(lambda k: det(k, n), kmax)) for n in range(1, n_max)]
    return _collect(per, n_modes, 3)


def drude_filled_cavity(k_cavity, k_p):
    """
    Modes of a cavity completely filled with a lossless Drude plasma:
    k^2 = k_c^2 + k_p^2 for every empty-cavity wavenumber k_c [1/m].
    """
    return np.sqrt(np.asarray(k_cavity) ** 2 + k_p**2)


def line_source_field(r, k, current=1.0):
    """
    Ez phasor of an electric line current I (exp(+i w t) convention) in
    vacuum: Ez = -(w mu_0 I / 4) H0^(2)(k r).

    Parameters
    ----------
    r : array_like
        Distance [m].
    k : float
        Wavenumber [1/m].
    current : complex
        Line current phasor [A].
    """
    from scipy.constants import c, mu_0
    from scipy.special import hankel2

    omega = k * c
    return -omega * mu_0 * current / 4 * hankel2(0, k * np.asarray(r))


def _rod_guide_det(k, beta, n, a, b, eps):
    """
    Pole-free characteristic function of the hybrid modes of a dielectric rod
    (radius b, eps) coaxial in a circular PEC guide (radius a) at axial
    wavenumber beta, angular order n (see `rod_cavity_hybrid`).
    """
    from scipy.special import ivp, kv, kvp

    def inner(kap2, r):
        # f = J_n(kappa r) / kappa^n and d f / dr (entire in kappa^2)
        if kap2 >= 0:
            q = np.sqrt(kap2)
            if q == 0:
                q = 1e-300
            return jv(n, q * r) / q**n, q * jvp(n, q * r) / q**n
        q = np.sqrt(-kap2)
        return iv(n, q * r) / q**n, q * ivp(n, q * r) / q**n

    def outer(kap2, r):
        # F = J(kr)Y(ka) - Y(kr)J(ka) (Ez = 0 at a), Gh = dF/da (dHz/dr = 0
        # at a), both with their r-derivatives; continuous across kappa = 0
        if kap2 >= 0:
            q = max(np.sqrt(kap2), 1e-300)
            F = jv(n, q * r) * yv(n, q * a) - yv(n, q * r) * jv(n, q * a)
            dF = q * (jvp(n, q * r) * yv(n, q * a) - yvp(n, q * r) * jv(n, q * a))
            Gh = q * (jv(n, q * r) * yvp(n, q * a) - yv(n, q * r) * jvp(n, q * a))
            dGh = q * q * (jvp(n, q * r) * yvp(n, q * a) - yvp(n, q * r) * jvp(n, q * a))
            return F, dF, Gh, dGh
        q = np.sqrt(-kap2)
        s = -2 / np.pi
        F = s * (iv(n, q * r) * kv(n, q * a) - kv(n, q * r) * iv(n, q * a))
        dF = s * q * (ivp(n, q * r) * kv(n, q * a) - kvp(n, q * r) * iv(n, q * a))
        Gh = s * q * (iv(n, q * r) * kvp(n, q * a) - kv(n, q * r) * ivp(n, q * a))
        dGh = s * q * q * (ivp(n, q * r) * kvp(n, q * a) - kvp(n, q * r) * ivp(n, q * a))
        return F, dF, Gh, dGh

    k1, k2 = eps * k**2 - beta**2, k**2 - beta**2
    tiny = 1e-9 * k**2  # the functions are continuous through kappa = 0
    k1 = k1 if abs(k1) > tiny else tiny
    k2 = k2 if abs(k2) > tiny else tiny
    f, df = inner(k1, b)
    F, dF, G, dG = outer(k2, b)
    t1 = (beta * n / b) ** 2 * (k2 - k1) ** 2 * f**2 * F * G
    te = k2 * df * G - k1 * f * dG
    tm = eps * k2 * df * F - k1 * f * dF
    return t1 - k**2 * te * tm


def rod_cavity_hybrid(a, b, eps, height, n_modes=10, kmax=None, p_max=3, n_max=8):
    """
    Wavenumbers of a closed cylindrical PEC cavity (radius a, height h) with a
    coaxial dielectric rod (radius b, rel. permittivity eps) spanning its full
    height.

    The PEC end caps select standing waves with axial wavenumber
    beta_p = p pi / h. For p = 0 only TM (Ez) modes exist and k follows from
    `rod_in_cavity` (TM). For p >= 1 the rod couples Ez and Hz (hybrid
    HE/EH modes): k solves the 2 x 2 continuity determinant of E_phi and
    H_phi at r = b with Ez(a) = E_phi(a) = 0,

        (beta n / b)^2 (1/k1^2 - 1/k2^2)^2
            = k^2 (P/k1^2 - Q_H/k2^2)(eps P/k1^2 - Q_E/k2^2),

    k1^2 = eps k^2 - beta^2, k2^2 = k^2 - beta^2, P the logarithmic
    derivative of J_n(k1 r) at b and Q_E, Q_H those of the outer Ez and Hz
    radial functions; it is evaluated in a pole-free form with Bessel
    functions continued to modified Bessel functions for k^2 < 0. Note that
    k^2 = k_t^2 + beta^2 holds only for a homogeneous filling.

    Parameters
    ----------
    a, b : float
        Cavity and rod radii [m].
    eps : float
        Rod permittivity.
    height : float
        Cavity height h [m].
    n_modes : int
        Number of modes returned (with degeneracy 2 for n >= 1).
    kmax : float, optional
        Upper search limit [1/m] (default 8 / a).
    p_max : int
        Largest axial index.

    Returns
    -------
    k : ndarray
        Vacuum wavenumbers [1/m], sorted, repeated by degeneracy.
    """
    kmax = kmax or 8 / a
    roots = []
    for k in rod_in_cavity(a, b, eps, "TM", n_modes=200, kmax=kmax, n_max=n_max):
        roots.append(k)
    for p in range(1, p_max + 1):
        beta = p * np.pi / height
        if beta >= kmax:
            break
        for n in range(n_max):
            rs = _roots(
                lambda k: _rod_guide_det(k, beta, n, a, b, eps), kmax,
                n_scan=6000, kmin=beta / np.sqrt(eps) * (1 + 1e-9),
            )  # fmt: skip
            # discard the branch points kappa = 0 (k = beta, beta / sqrt(eps))
            rs = [
                r for r in rs
                if min(abs(r - beta), abs(r - beta / np.sqrt(eps))) > 1e-7 * beta
            ]  # fmt: skip
            roots += rs * (1 if n == 0 else 2)
    return np.sort(roots)[:n_modes]
