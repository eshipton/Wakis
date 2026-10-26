# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Vectorised relativistic particle pushers in SI units.

All momentum pushers advance the relativistic velocity-momentum
``u = gamma * v`` [m/s] from ``t_{n-1/2}`` to ``t_{n+1/2}`` given electric
and magnetic fields ``E^n`` [V/m] and ``B^n`` [T] at ``x^n``, followed by the
leapfrog position advance ``x^{n+1} = x^n + dt * u^{n+1/2} / gamma^{n+1/2}``.

Implemented schemes:
- ``"boris"``: Boris (1970); Birdsall & Langdon (Taylor & Francis / IOP, 1991), ISBN 978-0-7503-0117-6.
- ``"vay"``: J.-L. Vay, "Simulation of beams or plasmas crossing at
  relativistic velocity", *Phys. Plasmas* **15**, 056701 (2008),
  DOI: ``10.1063/1.2837054``.
- ``"hc"`` / ``"higuera_cary"``: A. V. Higuera and J. R. Cary,
  "Structure-preserving second-order integration of relativistic charged
  particle trajectories in electromagnetic fields", *Phys. Plasmas* **24**,
  052104 (2017), DOI: ``10.1063/1.4979989``.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c as c_light

_INV_C2 = 1.0 / (c_light * c_light)


def lorentz_gamma(ux, uy, uz, xp=np):
    """Compute Lorentz factor ``gamma = sqrt(1 + |u|^2 / c^2)`` from ``u = gamma * v`` [m/s]."""
    return xp.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * _INV_C2)


def push_boris(ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, q: float, m: float, dt: float, xp=np):
    """Advance ``u = gamma * v`` [m/s] over ``dt`` [s] using the relativistic Boris pusher.

    Parameters
    ----------
    ux, uy, uz : ndarray of float
        Normalised momentum components ``gamma * v`` [m/s] at ``t_{n-1/2}``.
    Ex, Ey, Ez : ndarray or float
        Electric field components [V/m] at ``x^n, t_n``.
    Bx, By, Bz : ndarray or float
        Magnetic flux density components [T] at ``x^n, t_n``.
    q : float
        Particle charge [C].
    m : float
        Particle rest mass [kg].
    dt : float
        Time step [s].
    xp : module, optional
        Array module (``numpy`` or ``cupy``). Default is ``numpy``.

    Returns
    -------
    ux_new, uy_new, uz_new : tuple of ndarray
        Updated momentum components ``gamma * v`` [m/s] at ``t_{n+1/2}``.
    """
    alpha = 0.5 * (q / m) * dt

    # First half electric impulse: u^- = u^{n-1/2} + (q dt / 2m) E^n
    umx = ux + alpha * Ex
    umy = uy + alpha * Ey
    umz = uz + alpha * Ez

    # Relativistic rotation vector t = (q dt / (2 m gamma^-)) B^n
    inv_gamma_m = 1.0 / xp.sqrt(1.0 + (umx * umx + umy * umy + umz * umz) * _INV_C2)
    tx = (alpha * Bx) * inv_gamma_m
    ty = (alpha * By) * inv_gamma_m
    tz = (alpha * Bz) * inv_gamma_m

    t_mag2 = tx * tx + ty * ty + tz * tz
    s_fac = 2.0 / (1.0 + t_mag2)
    sx = tx * s_fac
    sy = ty * s_fac
    sz = tz * s_fac

    # u' = u^- + u^- x t
    upx = umx + (umy * tz - umz * ty)
    upy = umy + (umz * tx - umx * tz)
    upz = umz + (umx * ty - umy * tx)

    # u^+ = u^- + u' x s, followed by second half electric impulse
    ux_new = umx + (upy * sz - upz * sy) + alpha * Ex
    uy_new = umy + (upz * sx - upx * sz) + alpha * Ey
    uz_new = umz + (upx * sy - upy * sx) + alpha * Ez
    return ux_new, uy_new, uz_new


def push_vay(ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, q: float, m: float, dt: float, xp=np):
    """Advance ``u = gamma * v`` [m/s] over ``dt`` [s] using the Vay (2008) pusher.

    Preserves the ``E x B`` drift velocity exactly at arbitrary Lorentz factor
    by centring the magnetic rotation with the average velocity
    ``(v^{n-1/2} + v^{n+1/2}) / 2``.

    References
    ----------
    J.-L. Vay, "Simulation of beams or plasmas crossing at relativistic
    velocity", *Phys. Plasmas* **15**, 056701 (2008),
    DOI: ``10.1063/1.2837054``.
    """
    alpha = 0.5 * (q / m) * dt

    inv_gamma_old = 1.0 / xp.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * _INV_C2)
    vx = ux * inv_gamma_old
    vy = uy * inv_gamma_old
    vz = uz * inv_gamma_old

    taux = alpha * Bx
    tauy = alpha * By
    tauz = alpha * Bz
    tau2 = taux * taux + tauy * tauy + tauz * tauz

    # u' = u^{n-1/2} + 2 alpha E^n + v^{n-1/2} x tau
    upx = ux + 2.0 * alpha * Ex + (vy * tauz - vz * tauy)
    upy = uy + 2.0 * alpha * Ey + (vz * taux - vx * tauz)
    upz = uz + 2.0 * alpha * Ez + (vx * tauy - vy * taux)

    up2_c2 = (upx * upx + upy * upy + upz * upz) * _INV_C2
    ustar_c = (upx * taux + upy * tauy + upz * tauz) / c_light
    sigma = 1.0 + up2_c2 - tau2

    gamma_new = xp.sqrt(
        0.5 * (sigma + xp.sqrt(sigma * sigma + 4.0 * (tau2 + ustar_c * ustar_c)))
    )
    inv_gamma_new = 1.0 / gamma_new

    tx = taux * inv_gamma_new
    ty = tauy * inv_gamma_new
    tz = tauz * inv_gamma_new
    s = 1.0 / (1.0 + tx * tx + ty * ty + tz * tz)
    up_dot_t = upx * tx + upy * ty + upz * tz

    ux_new = s * (upx + up_dot_t * tx + (upy * tz - upz * ty))
    uy_new = s * (upy + up_dot_t * ty + (upz * tx - upx * tz))
    uz_new = s * (upz + up_dot_t * tz + (upx * ty - upy * tx))
    return ux_new, uy_new, uz_new


def push_higuera_cary(
    ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, q: float, m: float, dt: float, xp=np
):
    """Advance ``u = gamma * v`` [m/s] over ``dt`` [s] using the Higuera-Cary (2017) pusher.

    Combines the volume-preserving Boris split with the Vay-like quadratic
    Lorentz factor at the rotation midpoint so both ``E x B`` drift and phase
    space volume are preserved.

    References
    ----------
    A. V. Higuera and J. R. Cary, "Structure-preserving second-order
    integration of relativistic charged particle trajectories in
    electromagnetic fields", *Phys. Plasmas* **24**, 052104 (2017),
    DOI: ``10.1063/1.4979989``.
    """
    alpha = 0.5 * (q / m) * dt

    # First half electric impulse: u^- = u^{n-1/2} + alpha E^n
    umx = ux + alpha * Ex
    umy = uy + alpha * Ey
    umz = uz + alpha * Ez

    taux = alpha * Bx
    tauy = alpha * By
    tauz = alpha * Bz
    tau2 = taux * taux + tauy * tauy + tauz * tauz

    gamma_m2 = 1.0 + (umx * umx + umy * umy + umz * umz) * _INV_C2
    ustar_c = (umx * taux + umy * tauy + umz * tauz) / c_light
    sigma = gamma_m2 - tau2

    gamma_hc = xp.sqrt(
        0.5 * (sigma + xp.sqrt(sigma * sigma + 4.0 * (tau2 + ustar_c * ustar_c)))
    )
    inv_gamma_hc = 1.0 / gamma_hc

    tx = taux * inv_gamma_hc
    ty = tauy * inv_gamma_hc
    tz = tauz * inv_gamma_hc

    t_mag2 = tx * tx + ty * ty + tz * tz
    s_fac = 2.0 / (1.0 + t_mag2)
    sx = tx * s_fac
    sy = ty * s_fac
    sz = tz * s_fac

    upx = umx + (umy * tz - umz * ty)
    upy = umy + (umz * tx - umx * tz)
    upz = umz + (umx * ty - umy * tx)

    ux_new = umx + (upy * sz - upz * sy) + alpha * Ex
    uy_new = umy + (upz * sx - upx * sz) + alpha * Ey
    uz_new = umz + (upx * sy - upy * sx) + alpha * Ez
    return ux_new, uy_new, uz_new


PUSHERS = {
    "boris": push_boris,
    "vay": push_vay,
    "hc": push_higuera_cary,
    "higuera_cary": push_higuera_cary,
    "higuera-cary": push_higuera_cary,
}


def get_pusher(name: str):
    """Return the momentum pusher function for ``name``."""
    key = str(name).strip().lower()
    if key not in PUSHERS:
        raise ValueError(
            f"Unknown pusher={name!r}; expected one of {sorted(PUSHERS.keys())}."
        )
    return PUSHERS[key]


def push_momentum(
    species,
    Ex,
    Ey,
    Ez,
    Bx,
    By,
    Bz,
    dt: float,
    pusher: str = "boris",
) -> None:
    """In-place momentum push ``u^{n-1/2} -> u^{n+1/2}`` for a :class:`Species`."""
    n = species.n_particles
    if n == 0:
        return
    fn = get_pusher(pusher)
    _CPU_CHUNK = 65536
    if species.xp is np and n > _CPU_CHUNK:
        ux_arr = species.ux
        uy_arr = species.uy
        uz_arr = species.uz
        for s0 in range(0, n, _CPU_CHUNK):
            s1 = min(s0 + _CPU_CHUNK, n)
            ux_c, uy_c, uz_c = fn(
                ux_arr[s0:s1],
                uy_arr[s0:s1],
                uz_arr[s0:s1],
                Ex[s0:s1] if np.ndim(Ex) > 0 else Ex,
                Ey[s0:s1] if np.ndim(Ey) > 0 else Ey,
                Ez[s0:s1] if np.ndim(Ez) > 0 else Ez,
                Bx[s0:s1] if np.ndim(Bx) > 0 else Bx,
                By[s0:s1] if np.ndim(By) > 0 else By,
                Bz[s0:s1] if np.ndim(Bz) > 0 else Bz,
                species.charge,
                species.mass,
                dt,
                xp=np,
            )
            ux_arr[s0:s1] = ux_c
            uy_arr[s0:s1] = uy_c
            uz_arr[s0:s1] = uz_c
        return
    ux_new, uy_new, uz_new = fn(
        species.ux,
        species.uy,
        species.uz,
        Ex,
        Ey,
        Ez,
        Bx,
        By,
        Bz,
        species.charge,
        species.mass,
        dt,
        xp=species.xp,
    )
    species.ux = ux_new
    species.uy = uy_new
    species.uz = uz_new


def push_position(species, dt: float) -> None:
    """In-place leapfrog position advance ``x^{n+1} = x^n + dt * u^{n+1/2} / gamma^{n+1/2}``."""
    n = species.n_particles
    if n == 0:
        return
    xp = species.xp
    _CPU_CHUNK = 65536
    if xp is np and n > _CPU_CHUNK:
        x_arr, y_arr, z_arr = species.x, species.y, species.z
        ux_arr, uy_arr, uz_arr = species.ux, species.uy, species.uz
        for s0 in range(0, n, _CPU_CHUNK):
            s1 = min(s0 + _CPU_CHUNK, n)
            ux_c = ux_arr[s0:s1]
            uy_c = uy_arr[s0:s1]
            uz_c = uz_arr[s0:s1]
            fac = dt / np.sqrt(1.0 + (ux_c * ux_c + uy_c * uy_c + uz_c * uz_c) * _INV_C2)
            x_arr[s0:s1] += fac * ux_c
            y_arr[s0:s1] += fac * uy_c
            z_arr[s0:s1] += fac * uz_c
        return
    ux = species.ux
    uy = species.uy
    uz = species.uz
    inv_gamma = 1.0 / xp.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) * _INV_C2)
    species.x = species.x + ( dt * inv_gamma) * ux
    species.y = species.y + (dt * inv_gamma) * uy
    species.z = species.z + (dt * inv_gamma) * uz


def integrate_trajectory(
    x0,
    u0,
    dt: float,
    n_steps: int,
    E_fn,
    B_fn,
    q: float,
    m: float,
    pusher: str = "boris",
    half_step_init: bool = True,
):
    """Integrate a single-particle or multi-particle trajectory in prescribed fields.

    Uses standard leapfrog centring: ``x`` at ``t_n = n * dt`` and
    ``u = gamma * v`` [m/s] at ``t_{n-1/2}``. When ``half_step_init=True``,
    ``u0`` is given synchronously at ``t = 0`` and is first back-pushed by
    ``-dt / 2`` to ``t_{-1/2}`` using ``E(x0, 0), B(x0, 0)``.

    Parameters
    ----------
    x0 : array-like, shape (3,) or (N, 3)
        Initial position [m] at ``t = 0``.
    u0 : array-like, shape (3,) or (N, 3)
        Initial relativistic momentum ``u = gamma * v`` [m/s] at ``t = 0``
        (if ``half_step_init=True``) or at ``t = -dt / 2``.
    dt : float
        Time step [s].
    n_steps : int
        Number of steps to integrate.
    E_fn, B_fn : callable or array-like of shape (3,)
        Either a constant 3-vector ``(Fx, Fy, Fz)`` or a callable
        ``F_fn(x, t) -> (Fx, Fy, Fz)``.
    q : float
        Particle charge [C].
    m : float
        Particle rest mass [kg].
    pusher : {"boris", "vay", "hc"}, optional
        Momentum pusher scheme. Default is ``"boris"``.
    half_step_init : bool, optional
        If True (default), back-push ``u0`` from ``t = 0`` to ``t = -dt / 2``.

    Returns
    -------
    t_hist : ndarray, shape (n_steps + 1,)
        Time stamps ``t_n = n * dt`` [s].
    x_hist : ndarray, shape (n_steps + 1, ...)
        Positions ``x^n`` [m] at ``t_n``.
    u_hist : ndarray, shape (n_steps + 1, ...)
        Half-step momenta ``u^{n-1/2}`` [m/s] (with ``u_hist[0] = u^{-1/2}``
        and ``u_hist[n] = u^{n-1/2}``).
    u_sync : ndarray, shape (n_steps + 1, ...)
        Time-centred momenta at ``t_n`` obtained by advancing ``u^{n-1/2}`` by
        ``+dt / 2`` at ``(x^n, t_n)``.
    """
    fn = get_pusher(pusher)
    x = np.asarray(x0, dtype=np.float64).copy()
    u = np.asarray(u0, dtype=np.float64).copy()

    def _eval_field(field_spec, pos, t_val):
        if callable(field_spec):
            f = np.asarray(field_spec(pos, t_val), dtype=np.float64)
        else:
            f = np.asarray(field_spec, dtype=np.float64)
        return f[..., 0], f[..., 1], f[..., 2]

    if half_step_init:
        Ex0, Ey0, Ez0 = _eval_field(E_fn, x, 0.0)
        Bx0, By0, Bz0 = _eval_field(B_fn, x, 0.0)
        ux, uy, uz = fn(
            u[..., 0],
            u[..., 1],
            u[..., 2],
            Ex0,
            Ey0,
            Ez0,
            Bx0,
            By0,
            Bz0,
            q,
            m,
            -0.5 * dt,
            xp=np,
        )
        u = np.stack([ux, uy, uz], axis=-1)

    t_hist = np.arange(n_steps + 1, dtype=np.float64) * dt
    x_hist = np.empty((n_steps + 1,) + x.shape, dtype=np.float64)
    u_hist = np.empty((n_steps + 1,) + u.shape, dtype=np.float64)
    u_sync = np.empty((n_steps + 1,) + u.shape, dtype=np.float64)

    x_hist[0] = x
    u_hist[0] = u

    for n in range(n_steps):
        t_n = n * dt
        Ex, Ey, Ez = _eval_field(E_fn, x, t_n)
        Bx, By, Bz = _eval_field(B_fn, x, t_n)

        # Synchronised momentum at t_n (half-step forward from u^{n-1/2})
        ux_s, uy_s, uz_s = fn(
            u[..., 0],
            u[..., 1],
            u[..., 2],
            Ex,
            Ey,
            Ez,
            Bx,
            By,
            Bz,
            q,
            m,
            0.5 * dt,
            xp=np,
        )
        u_sync[n] = np.stack([ux_s, uy_s, uz_s], axis=-1)

        # Full step momentum u^{n-1/2} -> u^{n+1/2}
        ux, uy, uz = fn(
            u[..., 0],
            u[..., 1],
            u[..., 2],
            Ex,
            Ey,
            Ez,
            Bx,
            By,
            Bz,
            q,
            m,
            dt,
            xp=np,
        )
        u = np.stack([ux, uy, uz], axis=-1)
        inv_gamma = 1.0 / np.sqrt(1.0 + np.sum(u * u, axis=-1, keepdims=True) * _INV_C2)
        x = x + dt * u * inv_gamma

        x_hist[n + 1] = x
        u_hist[n + 1] = u

    # Final synchronised momentum at t_{n_steps}
    t_end = n_steps * dt
    Ex, Ey, Ez = _eval_field(E_fn, x, t_end)
    Bx, By, Bz = _eval_field(B_fn, x, t_end)
    ux_s, uy_s, uz_s = fn(
        u[..., 0],
        u[..., 1],
        u[..., 2],
        Ex,
        Ey,
        Ez,
        Bx,
        By,
        Bz,
        q,
        m,
        0.5 * dt,
        xp=np,
    )
    u_sync[n_steps] = np.stack([ux_s, uy_s, uz_s], axis=-1)

    return t_hist, x_hist, u_hist, u_sync
