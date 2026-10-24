# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Dipole point sources, incoherent dipole ensembles, 2D Sommerfeld / CPS
half-space decay rates, and Bloch oblique plane-wave helpers for `SubgridFIT2D`.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0

__all__ = [
    "SubgridDipole",
    "homogeneous_dipole_power_2d",
    "run_incoherent_dipoles",
    "sommerfeld_2d_dipole_above_mirror",
    "add_bloch_plane_wave",
    "transfer_matrix_slab_2d",
]


class SubgridDipole:
    """
    2D point dipole source at `position = (x, y)` [m] for `SubgridFIT2D`.

    Parameters
    ----------
    position : sequence of float
        2D coordinates `(x, y)` [m].
    component : {'Ez', 'Ex', 'Ey', 'Hz', 'Hx', 'Hy'}, optional
        Excited field component. Default `'Ez'`.
    waveform : callable, optional
        Time waveform `q(t)` [dimensionless].
    moment : float or complex, optional
        2D dipole current moment `I * l` [A]. Default 1.0.
    phase : float, optional
        Phase offset [rad] applied to `moment` (as `exp(i * phase)` for complex
        solvers or used by `run_incoherent_dipoles`). Default 0.0.
    time_offset : float, optional
        Time shift [s] passed to `solver.add_current`. Default 0.0.
    """

    def __init__(
        self,
        position,
        component="Ez",
        waveform=None,
        moment=1.0,
        phase=0.0,
        time_offset=0.0,
    ):
        self.position = np.asarray(position, dtype=float)[:2]
        self.component = str(component)
        self.waveform = waveform
        self.moment = moment
        self.phase = float(phase)
        self.time_offset = float(time_offset)

    def register(self, solver, phase_override=None, time_offset_override=None):
        """Register this dipole on `solver` via `solver.add_current`."""
        ph = self.phase if phase_override is None else float(phase_override)
        t_off = (
            self.time_offset
            if time_offset_override is None
            else float(time_offset_override)
        )
        pts = self.position[None, :]
        if getattr(solver, "_is_complex", False):
            w = np.array([self.moment * np.exp(1j * ph)], dtype=complex)
            wf = self.waveform
        elif ph != 0.0 and self.waveform is not None:
            # For real solvers, apply phase shift via carrier frequency if present
            f0 = getattr(
                self.waveform, "frequency", getattr(self.waveform, "f0", None)
            )
            if f0 is not None and f0 > 0.0:
                t_off = t_off - ph / (2.0 * np.pi * float(f0))
                w = np.array([float(np.real(self.moment))], dtype=float)
                wf = self.waveform
            else:
                w = np.array([float(np.real(self.moment * np.exp(1j * ph)))], dtype=float)
                wf = self.waveform
        else:
            w = np.array([float(np.real(self.moment))], dtype=float)
            wf = self.waveform
        return solver.add_current(
            pts,
            w,
            waveform=wf,
            kind="points",
            component=self.component,
            time_offset=t_off,
        )


def homogeneous_dipole_power_2d(
    frequencies,
    waveform,
    dt=None,
    n_steps=None,
    polarization="TM",
    component=None,
    eps_r=1.0,
    moment=1.0,
    time_offset=0.0,
):
    """
    Exact 2D radiated power spectrum `P_0(f)` [W/m s^2] of a line dipole in a
    homogeneous lossless medium of relative permittivity `eps_r`.

    In 2D:

    * TM (`Ez` nodal current source): ``P_0(f) = (omega * mu_0 / 4) |moment * Q(f)|^2``;
    * TE (`Ex` or `Ey` in-plane electric dipole): ``P_0(f) = (omega * mu_0 / 8) |moment * Q(f)|^2``;
    * TE (`Hz` magnetic line source): ``P_0(f) = (omega * eps_0 * eps_r / 4) |moment * Q(f)|^2``.

    Parameters
    ----------
    frequencies : array_like
        Frequencies [Hz].
    waveform : callable
        Source waveform `q(t)`.
    dt : float, optional
        Timestep [s] for exact discrete Fourier transform of `q(t)`.
    n_steps : int, optional
        Number of timesteps for exact discrete Fourier transform of `q(t)`.
    polarization : {'TM', 'TE'}, optional
        Solver polarization. Default `'TM'`.
    component : str, optional
        Dipole component (`'Ez'`, `'Ex'`, `'Ey'`, `'Hz'`). Defaults to `'Ez'`
        for TM and `'Ex'` for TE.
    eps_r : float, optional
        Background relative permittivity. Default 1.0.
    moment : float or complex, optional
        Dipole moment `I * l` [A]. Default 1.0.
    time_offset : float, optional
        Time offset [s] added to `waveform`. Default 0.0.

    Returns
    -------
    ndarray
        Exact radiated power spectrum `P_0(f)` [W/m s^2].
    """
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    omega = 2.0 * np.pi * freqs
    if component is None:
        component = "Ez" if polarization == "TM" else "Ex"

    if dt is not None and n_steps is not None:
        n = np.arange(int(n_steps), dtype=float)
        # Nodal currents are sampled at (n + 0.5) * dt, edge currents at n * dt
        is_edge = component in ("Ex", "Ey", "Hx", "Hy")
        t_eval = (n if is_edge else (n + 0.5)) * float(dt) + float(time_offset)
        t_dft = (n + 0.5 if is_edge else (n + 1.0)) * float(dt)
        q_vals = np.asarray([waveform(ti) for ti in t_eval])
        Q = (
            float(dt)
            * np.exp(-2j * np.pi * freqs[:, None] * t_dft[None, :])
            @ q_vals
        )
    elif hasattr(waveform, "spectrum"):
        Q = waveform.spectrum(freqs)
    else:
        raise ValueError("Provide `dt` and `n_steps` or a waveform with `.spectrum(f)`.")

    amp2 = np.abs(moment * Q) ** 2
    if component == "Ez" or (polarization == "TM" and component not in ("Hx", "Hy")):
        return 0.25 * omega * MU0 * amp2
    if component in ("Ex", "Ey"):
        return 0.125 * omega * MU0 * amp2
    if component == "Hz":
        return 0.25 * omega * EPS0 * float(eps_r) * amp2
    return 0.125 * omega * EPS0 * float(eps_r) * amp2


def run_incoherent_dipoles(
    solver_factory,
    dipoles,
    monitor_factory,
    n_steps,
    mode="exact",
    n_realizations=16,
    seed=0,
    phi=None,
):
    """
    Evaluate ensemble-averaged flux and far-field intensity for mutually
    incoherent dipoles on `SubgridFIT2D`.

    Parameters
    ----------
    solver_factory : callable
        `solver_factory() -> SubgridFIT2D` returning a fresh solver instance.
    dipoles : sequence of SubgridDipole
        Dipole emitters in the ensemble.
    monitor_factory : callable
        `monitor_factory() -> SubgridFluxMonitor or SubgridNear2FarMonitor`.
    n_steps : int
        Number of time steps per run.
    mode : {'exact', 'random_phase'}, optional
        * `'exact'`: run one simulation per dipole and sum the quadratic power
          observables incoherently (`sum_k P_k`);
        * `'random_phase'`: run `n_realizations` simulations with all dipoles
          active simultaneously and independent uniform random phases
          `phi_k ~ U[0, 2 pi)`, averaging the quadratic observables.
    n_realizations : int, optional
        Number of Monte Carlo realizations when `mode='random_phase'`. Default 16.
    seed : int, optional
        Random number generator seed for `'random_phase'`. Default 0.
    phi : array_like, optional
        Observation angles [rad] if `monitor_factory` returns a
        `SubgridNear2FarMonitor`.

    Returns
    -------
    dict
        Keys `'flux'` (mean total flux spectrum), `'intensity'` (if `phi` is
        provided and monitor supports `far_field`), and `'per_run_flux'`.
    """
    dipoles = list(dipoles)
    if mode == "exact":
        flux_sum = None
        intensity_sum = None
        per_run = []
        for dip in dipoles:
            sim = solver_factory()
            dip.register(sim)
            mon = monitor_factory()
            for _ in range(int(n_steps)):
                t = sim.n * sim.dt
                sim.one_step()
                mon.update(sim, t)
            fl = mon.get_flux()
            per_run.append(fl)
            flux_sum = fl.copy() if flux_sum is None else (flux_sum + fl)
            if phi is not None and hasattr(mon, "far_field"):
                inten = mon.far_field(phi)["intensity"]
                intensity_sum = (
                    inten.copy() if intensity_sum is None else (intensity_sum + inten)
                )
        return dict(
            flux=flux_sum,
            intensity=intensity_sum,
            per_run_flux=np.asarray(per_run),
        )

    if mode in ("random_phase", "monte_carlo"):
        rng = np.random.default_rng(seed)
        flux_sum = None
        intensity_sum = None
        per_run = []
        n_real = int(n_realizations)
        for _ in range(n_real):
            sim = solver_factory()
            phases = rng.uniform(0.0, 2.0 * np.pi, size=len(dipoles))
            for dip, ph in zip(dipoles, phases):
                dip.register(sim, phase_override=float(ph))
            mon = monitor_factory()
            for _ in range(int(n_steps)):
                t = sim.n * sim.dt
                sim.one_step()
                mon.update(sim, t)
            fl = mon.get_flux()
            per_run.append(fl)
            flux_sum = fl.copy() if flux_sum is None else (flux_sum + fl)
            if phi is not None and hasattr(mon, "far_field"):
                inten = mon.far_field(phi)["intensity"]
                intensity_sum = (
                    inten.copy() if intensity_sum is None else (intensity_sum + inten)
                )
        return dict(
            flux=flux_sum / n_real,
            intensity=(None if intensity_sum is None else intensity_sum / n_real),
            per_run_flux=np.asarray(per_run),
        )

    raise ValueError(f"Unknown mode {mode!r}; expected 'exact' or 'random_phase'.")


def sommerfeld_2d_dipole_above_mirror(
    frequencies,
    distance,
    eps_mirror,
    orientation="Ex",
    eps_bg=1.0,
    n_quad=256,
):
    """
    Exact 2D Sommerfeld / Chance-Prock-Silbey (CPS) normalized decay rate
    `Gamma_tot / Gamma_0` for a 2D line dipole at height `distance` [m] above a
    half-space mirror with relative permittivity `eps_mirror`.

    The branch-point singularity at `u = k_|| / k_1 = 1` is removed analytically
    by substituting `u = sin(theta)` on `[0, 1]` and `u = sqrt(1 + q^2)` with
    Gauss-Legendre quadrature on `q = tan(tau)` on `[0, infinity)`.

    * `orientation='Ex'` (horizontal in-plane, TE / p-pol):
      ``Gamma_x / Gamma_{x,0} = 1 - (4 / pi) Re int_0^inf gamma_1 r_p(u) exp(2 i k_1 d gamma_1) du``;
    * `orientation='Ey'` (vertical in-plane, TE / p-pol):
      ``Gamma_y / Gamma_{y,0} = 1 + (4 / pi) Re int_0^inf (u^2 / gamma_1) r_p(u) exp(2 i k_1 d gamma_1) du``;
    * `orientation='Ez'` (horizontal out-of-plane, TM / s-pol):
      ``Gamma_z / Gamma_{z,0} = 1 + (2 / pi) Re int_0^inf (r_s(u) / gamma_1) exp(2 i k_1 d gamma_1) du``.

    Parameters
    ----------
    frequencies : float or array_like
        Frequencies [Hz].
    distance : float
        Height `d` [m] of the dipole above the mirror interface (`d > 0`).
    eps_mirror : str, complex, float, or DispersiveMaterial
        Mirror relative permittivity (or `'pec'`, or a `DispersiveMaterial`).
    orientation : {'Ex', 'Ey', 'Ez'}, optional
        Dipole orientation relative to an x-directed mirror interface (normal `y`).
        Default `'Ex'`.
    eps_bg : float, optional
        Upper half-space lossless relative permittivity. Default 1.0.
    n_quad : int, optional
        Number of Gauss-Legendre quadrature nodes per integration interval.
        Default 256.

    Returns
    -------
    float or ndarray
        Normalized total decay rate (Purcell factor) `Gamma_tot / Gamma_0`.
    """
    scalar_in = np.ndim(frequencies) == 0
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    d = float(distance)
    eb = float(eps_bg)
    k1 = 2.0 * np.pi * freqs * np.sqrt(eb) / C0

    is_pec = isinstance(eps_mirror, str) and eps_mirror.strip().lower() == "pec"
    if not is_pec:
        if hasattr(eps_mirror, "epsilon"):
            e2_raw = np.asarray(eps_mirror.epsilon(freqs), dtype=complex)
        else:
            e2_raw = np.broadcast_to(
                np.asarray(eps_mirror, dtype=complex), freqs.shape
            )
        # Convert to physics convention Im(eps_2) >= 0 for the Sommerfeld integral
        e2 = np.real(e2_raw) + 1j * np.abs(np.imag(e2_raw))

    nodes, weights = leggauss(int(n_quad))

    # --- Interval 1: propagating waves u = sin(theta), theta in [0, pi/2]
    # du = cos(theta) dtheta = gamma_1 dtheta
    theta = 0.25 * np.pi * (nodes + 1.0)  # (N,)
    w_theta = 0.25 * np.pi * weights
    u_prop = np.sin(theta)[None, :]  # (1, N)
    g1_prop = np.cos(theta)[None, :]  # (1, N)

    if is_pec:
        rs_prop = -np.ones_like(u_prop, dtype=complex)
        rp_prop = np.ones_like(u_prop, dtype=complex)
    else:
        ratio = (e2 / eb)[:, None]
        g2_prop = np.sqrt(ratio - u_prop**2 + 0j)
        g2_prop = np.where(np.imag(g2_prop) < 0, -g2_prop, g2_prop)
        rs_prop = (g1_prop - g2_prop) / (g1_prop + g2_prop)
        rp_prop = (ratio * g1_prop - g2_prop) / (ratio * g1_prop + g2_prop)

    phase_prop = np.exp(2j * k1[:, None] * d * g1_prop)
    if orientation == "Ez":
        # du / gamma_1 = dtheta
        int_prop = (2.0 / np.pi) * np.real(
            np.sum(rs_prop * phase_prop * w_theta[None, :], axis=1)
        )
    elif orientation == "Ex":
        # gamma_1 du = gamma_1^2 dtheta
        int_prop = -(4.0 / np.pi) * np.real(
            np.sum((g1_prop**2) * rp_prop * phase_prop * w_theta[None, :], axis=1)
        )
    elif orientation == "Ey":
        # (u^2 / gamma_1) du = u^2 dtheta
        int_prop = (4.0 / np.pi) * np.real(
            np.sum((u_prop**2) * rp_prop * phase_prop * w_theta[None, :], axis=1)
        )
    else:
        raise ValueError(f"Unknown orientation {orientation!r}; expected 'Ex', 'Ey', or 'Ez'.")

    # --- Interval 2: evanescent waves gamma_1 = i * q, q in [0, q_max]
    # u = sqrt(1 + q^2), du = (q / u) dq, exp(2 i k_1 d gamma_1) = exp(-2 k_1 d q)
    if is_pec:
        int_evan = np.zeros_like(int_prop)
    else:
        # Map s in [-1, 1] -> q = q_scale * (1 + s) / (1 - s)
        q_scale = 1.0 / max(2.0 * float(np.min(k1)) * d, 0.1)
        s = np.clip(nodes, -1.0 + 1e-12, 1.0 - 1e-12)
        q = q_scale * (1.0 + s) / (1.0 - s)
        dq = q_scale * 2.0 / ((1.0 - s) ** 2) * weights
        u_ev = np.sqrt(1.0 + q**2)[None, :]
        q_row = q[None, :]
        g1_ev = 1j * q_row
        ratio = (e2 / eb)[:, None]
        g2_ev = np.sqrt(ratio - u_ev**2 + 0j)
        g2_ev = np.where(np.imag(g2_ev) < 0, -g2_ev, g2_ev)
        rs_ev = (g1_ev - g2_ev) / (g1_ev + g2_ev)
        rp_ev = (ratio * g1_ev - g2_ev) / (ratio * g1_ev + g2_ev)
        decay = np.exp(-2.0 * k1[:, None] * d * q_row)
        du_dq = q_row / u_ev
        if orientation == "Ez":
            # Re( (rs / (i q)) * (q / u) ) = Im(rs) / u
            integrand = (np.imag(rs_ev) / u_ev) * decay
            int_evan = (2.0 / np.pi) * np.sum(integrand * dq[None, :], axis=1)
        elif orientation == "Ex":
            # Re( -(i q) rp * (q / u) ) = q^2 Im(rp) / u
            integrand = ((q_row**2) * np.imag(rp_ev) / u_ev) * decay
            int_evan = (4.0 / np.pi) * np.sum(integrand * dq[None, :], axis=1)
        else:
            # Re( (u^2 / (i q)) rp * (q / u) ) = u Im(rp)
            integrand = (u_ev * np.imag(rp_ev)) * decay
            int_evan = (4.0 / np.pi) * np.sum(
                integrand * (du_dq * 0.0 + 1.0) * dq[None, :], axis=1
            )

    res = 1.0 + int_prop + int_evan
    return float(res[0]) if scalar_in else res


def add_bloch_plane_wave(
    solver,
    x_src,
    src_time,
    theta_deg=0.0,
    amplitude=1.0,
    n_pts=None,
):
    """
    Register an oblique soft line source at `x = x_src` [m] in a `periodic_y`
    `SubgridFIT2D` solver with Bloch phase matching `theta_deg`.

    The spatial weight along `y` is ``exp(-i k_y y) * dy`` where
    ``k_y = -angle(solver.bloch_phase) / L_y``, exciting exclusively the
    `m = 0` Floquet harmonic at angle `theta_deg`.

    Parameters
    ----------
    solver : SubgridFIT2D
        Solver with `mesh.periodic_y == True`.
    x_src : float
        Source plane x-coordinate [m].
    src_time : callable
        Time waveform `q(t)`.
    theta_deg : float, optional
        Angle of incidence [deg] from `+x` (used if `solver.bloch_phase == 1`
        to compute `k_y = 0`). Default 0.0.
    amplitude : float or complex, optional
        Overall amplitude scale. Default 1.0.
    n_pts : int, optional
        Number of quadrature points across `L_y`.

    Returns
    -------
    int
        Index of the registered source in `solver.currents`.
    """
    m = solver.mesh
    if not m.periodic_y:
        raise ValueError("add_bloch_plane_wave requires `mesh.periodic_y == True`.")
    ymin, ymax = float(m.yc[0]), float(m.yc[-1])
    Ly = ymax - ymin
    ky = -float(np.angle(solver.bloch_phase)) / Ly
    if n_pts is None:
        c = m.cells
        h_min = float(np.min(c["y1"] - c["y0"]))
        n_pts = max(32, int(np.ceil(2.0 * Ly / h_min)))
    t = (np.arange(int(n_pts), dtype=float) + 0.5) / int(n_pts)
    yj = ymin + t * Ly
    xj = np.full_like(yj, float(x_src))
    pts = np.column_stack([xj, yj])
    dy = Ly / int(n_pts)
    weights = complex(amplitude) * dy * np.exp(-1j * ky * (yj - ymin))
    if not solver._is_complex and np.allclose(np.imag(weights), 0.0):
        weights = np.real(weights)
    comp = "Ez" if solver.pol == "TM" else "Hz"
    return solver.add_current(pts, weights, waveform=src_time, kind="points", component=comp)


def transfer_matrix_slab_2d(
    frequencies,
    thickness,
    eps_slab,
    theta_deg=0.0,
    polarization="TM",
    eps_bg=1.0,
):
    """
    Exact analytical reflectance `R` and transmittance `T` of a homogeneous 2D
    dielectric/dispersive slab of thickness `thickness` [m] at incidence angle
    `theta_deg` [deg].

    * `polarization='TM'` (`u = Ez`, out-of-plane E): s-polarization;
    * `polarization='TE'` (`u = Hz`, in-plane E): p-polarization.

    Returns
    -------
    dict
        Keys `'R'`, `'T'`, `'r'`, `'t'` (arrays of shape `(n_freq,)`).
    """
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    theta = np.deg2rad(float(theta_deg))
    eb = float(eps_bg)
    if hasattr(eps_slab, "epsilon"):
        es_raw = np.asarray(eps_slab.epsilon(freqs), dtype=complex)
    else:
        es_raw = np.broadcast_to(np.asarray(eps_slab, dtype=complex), freqs.shape)
    # Use physics convention Im(eps) >= 0 for standard exp(i k_x L) Airy formula
    es = np.real(es_raw) + 1j * np.abs(np.imag(es_raw))

    k0 = 2.0 * np.pi * freqs / C0
    n1 = np.sqrt(eb)
    sin_t = np.sin(theta)
    cos_t1 = np.cos(theta)
    kx1 = k0 * n1 * cos_t1
    ky = k0 * n1 * sin_t
    kx2 = np.sqrt(k0**2 * es - ky**2 + 0j)
    kx2 = np.where(np.imag(kx2) < 0, -kx2, kx2)

    if polarization == "TM":
        # s-pol (Ez out-of-plane)
        r12 = (kx1 - kx2) / (kx1 + kx2)
    elif polarization == "TE":
        # p-pol (Hz out-of-plane)
        r12 = (es * kx1 - eb * kx2) / (es * kx1 + eb * kx2)
    else:
        raise ValueError("polarization must be 'TM' or 'TE'.")

    t12 = 1.0 + r12
    t21 = 1.0 - r12
    phase = np.exp(1j * kx2 * float(thickness))
    denom = 1.0 - (r12**2) * (phase**2)
    r = r12 * (1.0 - phase**2) / denom
    t = (t12 * t21 * phase) / denom
    R = np.abs(r) ** 2
    T = np.abs(t) ** 2
    return dict(R=R, T=T, r=r, t=t)
