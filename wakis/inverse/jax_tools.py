# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""JAX bindings for adjoint simulations and design-density utilities.

JAX is an optional dependency (``pip install wakis[inverse]``); this module is
imported on first use of its functions through `wakis.inverse`. Enable double
precision with ``jax.config.update("jax_enable_x64", True)`` so gradients keep
the accuracy of the float64 solver.
"""

import jax
import jax.numpy as jnp
import numpy as np


def jax_simulation(sim):
    """
    Wrap an `AdjointSimulation` as a differentiable JAX function.

    Parameters
    ----------
    sim : AdjointSimulation
        Simulation to wrap.

    Returns
    -------
    callable
        Function rho -> {monitor: {output key: complex array}} with the
        structure of `sim.forward` (raw outputs; see `flux`,
        `mode_amplitudes`, `farfield`, `power_waves` and `diffraction_orders`).
        Its reverse-mode derivative (jax.grad, jax.vjp) runs `sim.vjp`;
        forward-mode (jax.jvp) is not defined.

    Notes
    -----
    The solver runs on the host through `jax.pure_callback`, so the function
    may be used inside `jax.jit`. Each evaluation runs a forward simulation;
    the gradient reuses the latest forward run when rho matches, and reruns
    it otherwise.
    """
    shapes = sim.output_shapes()  # also initializes sim.readouts
    cdtype = jnp.result_type(complex)
    out_spec = {
        name: {
            comp: jax.ShapeDtypeStruct(shape, cdtype) for comp, shape in comps.items()
        }
        for name, comps in shapes.items()
    }
    rho_shape = sim.design.shape

    def _forward(rho, record=True):
        result = sim.forward(np.asarray(rho, dtype=float), record=record)
        return {
            name: {comp: np.asarray(v, dtype=cdtype) for comp, v in comps.items()}
            for name, comps in result.items()
        }

    def _vjp(rho, cotangents):
        rho = np.asarray(rho, dtype=float)
        if (
            sim.rho is None
            or not np.array_equal(sim.rho, rho)
            or not getattr(sim, "_recorded", True)
        ):
            sim.forward(rho)
        # JAX cotangents of complex outputs are conj(dF/dRe + 1j dF/dIm)
        G = {
            name: {comp: np.conj(np.asarray(ct)) for comp, ct in comps.items()}
            for name, comps in cotangents.items()
        }
        return np.asarray(sim.vjp(G), dtype=jnp.result_type(float))

    @jax.custom_vjp
    def simulate(rho):
        # Primal evaluation without a gradient: record nothing
        return jax.pure_callback(lambda r: _forward(r, record=False), out_spec, rho)

    def simulate_fwd(rho):
        # Forward pass of reverse mode: keep what `vjp` needs
        return jax.pure_callback(_forward, out_spec, rho), rho

    def simulate_bwd(rho, cotangents):
        spec = jax.ShapeDtypeStruct(rho_shape, jnp.result_type(float))
        return (jax.pure_callback(_vjp, spec, rho, cotangents),)

    simulate.defvjp(simulate_fwd, simulate_bwd)
    return simulate


def _filter(rho, kernel):
    """Normalized convolution with edge padding, so a constant rho is kept."""
    kernel = kernel / jnp.sum(kernel)
    pad = [(k // 2, k // 2) for k in kernel.shape]
    padded = jnp.pad(rho, pad, mode="edge")
    return jax.scipy.signal.convolve(padded, kernel, mode="valid")


def _distance(radius, ndim):
    r = int(np.ceil(radius))
    axes = [np.arange(-r, r + 1, dtype=float)] * ndim
    grids = np.meshgrid(*axes, indexing="ij")
    return np.sqrt(sum(g**2 for g in grids))


def conic_filter(rho, radius):
    """
    Conic (linear hat) density filter (Bourdin 2001).

    Parameters
    ----------
    rho : array
        Density on the design grid; axes of length 1 are left unfiltered.
    radius : float
        Filter radius in design-grid pixels.

    Returns
    -------
    array
        Filtered density, same shape as rho.
    """
    squeezed = jnp.squeeze(rho)
    weights = np.maximum(0.0, 1.0 - _distance(radius, squeezed.ndim) / radius)
    return _filter(squeezed, jnp.asarray(weights)).reshape(rho.shape)


def gaussian_filter(rho, sigma):
    """
    Gaussian density filter, truncated at 3 sigma.

    Parameters
    ----------
    rho : array
        Density on the design grid; axes of length 1 are left unfiltered.
    sigma : float
        Standard deviation in design-grid pixels.

    Returns
    -------
    array
        Filtered density, same shape as rho.
    """
    squeezed = jnp.squeeze(rho)
    weights = np.exp(-0.5 * (_distance(3 * sigma, squeezed.ndim) / sigma) ** 2)
    return _filter(squeezed, jnp.asarray(weights)).reshape(rho.shape)


def tanh_projection(rho, beta, eta=0.5):
    """
    Smoothed Heaviside projection towards a binary design (Wang et al. 2011).

    rho_p = (tanh(beta eta) + tanh(beta (rho - eta)))
            / (tanh(beta eta) + tanh(beta (1 - eta)))

    Parameters
    ----------
    rho : array
        Density in [0, 1].
    beta : float
        Projection steepness; beta -> inf gives a step at eta.
    eta : float, optional
        Threshold. Default is 0.5.

    Returns
    -------
    array
        Projected density in [0, 1].
    """
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta))
    return num / (jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))


def _tanh_projection_any(rho, beta, eta):
    if np.isinf(beta):
        return jnp.where(rho > eta, 1.0, 0.0)
    return tanh_projection(rho, beta, eta)


def smoothed_projection(
    rho_f, beta, eta=0.5, spacing=1.0, radius=None, periodic=False, grad_norm=None
):
    """
    Subpixel-smoothed projection (SSP) of a filtered density (Hammond et
    al., Opt. Express 33, 33620 (2025), Eqs. 7, 10-12).

    Near the level set rho_f = eta, i.e. at signed distance
    d = (eta - rho_f) / |grad rho_f| below the smoothing radius R, the output
    is (1 - F) P(rho_f - R g F) + F P(rho_f + R g (1 - F)) with
    g = |grad rho_f|, F = F_2(d / R) the C^2 fill fraction of the paper's
    Eq. (10) and P the tanh projection; elsewhere it is P(rho_f). For
    beta = inf the result is binary except within R of interfaces, where it
    equals F(d), a smooth function of the interface position: the gradient
    stays finite and nonzero at beta = inf (unlike `tanh_projection`). NumPy
    twin with an explicit VJP: `wakis.inverse.projection`.

    Parameters
    ----------
    rho_f : array
        Filtered density on a regular grid.
    beta : float
        Projection steepness (> 0, may be inf).
    eta : float, optional
        Threshold. Default 0.5.
    spacing : float or sequence of float, optional
        Grid spacing per axis. Default 1 (pixels).
    radius : float, optional
        Smoothing radius R [units of spacing]. Default 0.55 times the
        largest spacing of the non-singleton axes.
    periodic : bool or sequence of bool, optional
        Periodic axes (wrapped central differences). Default False.
    grad_norm : array, optional
        |grad rho_f| [1 / units of spacing] if computed by the caller (e.g.
        from a filter evaluated beyond the grid); default: central
        differences of rho_f (one-sided at non-periodic ends, as
        jnp.gradient), axes of length 1 ignored.

    Returns
    -------
    array
        Projected density, same shape as rho_f.
    """
    from .projection import RADIUS_FACTOR

    rho_f = jnp.asarray(rho_f)
    nd = rho_f.ndim
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (nd,))
    periodic = np.broadcast_to(np.asarray(periodic, dtype=bool), (nd,))
    active = [a for a in range(nd) if rho_f.shape[a] > 1]
    if radius is None:
        radius = RADIUS_FACTOR * (max(spacing[a] for a in active) if active else 1.0)
    if grad_norm is None:
        q = jnp.zeros_like(rho_f)
        for a in active:
            if periodic[a]:
                diff = 0.5 * (jnp.roll(rho_f, -1, a) - jnp.roll(rho_f, 1, a))
            else:
                diff = jnp.gradient(rho_f, axis=a)
            q = q + (diff / spacing[a]) ** 2
    else:
        q = jnp.asarray(grad_norm) ** 2
    nonzero = q > 0
    g = jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, q, 1.0)), 1.0)
    d = (eta - rho_f) / g
    smooth = nonzero & (jnp.abs(d) < radius)
    x = jnp.where(smooth, d / radius, 0.0)
    F = 0.5 - 15 / 16 * x + 5 / 8 * x**3 - 3 / 16 * x**5
    minus = _tanh_projection_any(rho_f - radius * g * F, beta, eta)
    plus = _tanh_projection_any(rho_f + radius * g * (1 - F), beta, eta)
    return jnp.where(
        smooth, (1 - F) * minus + F * plus, _tanh_projection_any(rho_f, beta, eta)
    )


def flux(raw, monitor):
    """
    Flux spectrum of a `FluxMonitor` (or `Near2FarMonitor`) from raw outputs.

    Phi(f) = sum_i Re(E_i conj(H_i)) . weight_i, as `FluxMonitor.get_flux`.

    Parameters
    ----------
    raw : dict
        Raw outputs {'E<i>', 'H<i>'} of the monitor (from `jax_simulation`).
    monitor : FluxMonitor
        Initialized monitor, e.g. `AdjointSimulation.readouts[name]`.

    Returns
    -------
    array
        Phi(f) [W s^2], shape (n_freq,).
    """
    total = 0.0
    for i, term in enumerate(monitor._terms):
        weight = jnp.asarray(np.asarray(term["weight"]))
        total = total + jnp.real(raw[f"E{i}"] * jnp.conj(raw[f"H{i}"])) @ weight
    return total


def mode_amplitudes(raw, monitor, mode_index=0):
    """
    Forward and backward mode amplitudes of a `ModeMonitor` from raw outputs,
    as `ModeMonitor.get_amplitudes`:

        a+- = 1/4 sum (E x h^* +- e^* x H) . a dA

    Parameters
    ----------
    raw : dict
        Raw outputs {'E', 'H'} of shape (2, n_freq, n_b * n_c).
    monitor : ModeMonitor
        Initialized monitor, e.g. `AdjointSimulation.readouts[name]`. Its
        modes are solved on first use, on a solver without the design.
    mode_index : int, optional
        Mode number. Default 0.

    Returns
    -------
    a_plus, a_minus : array of complex
        Amplitudes [sqrt(W) s], shape (n_freq,).
    """
    if monitor.modes is None:
        monitor._solve_modes()
    if any(len(modes) <= mode_index for modes in monitor.modes):
        raise ValueError(f"Mode {mode_index} does not propagate at all frequencies.")
    plane = monitor.plane
    modes = [m[mode_index] for m in monitor.modes]

    def weights(profile, d, dA):
        return np.stack(
            [(np.conj(getattr(m, profile)[d]) * dA).ravel("F") for m in modes]
        )

    E, H = raw["E"], raw["H"]
    E_x_h = jnp.sum(
        E[0] * weights("h_mean", "c", plane.dA_b)
        - E[1] * weights("h_mean", "b", plane.dA_c),
        axis=-1,
    )
    e_x_H = jnp.sum(
        weights("e", "b", plane.dA_b) * H[1] - weights("e", "c", plane.dA_c) * H[0],
        axis=-1,
    )
    return 0.25 * (E_x_h + e_x_H), 0.25 * (E_x_h - e_x_H)


def farfield(raw, monitor, theta, phi, origin=None):
    """
    Far-field pattern of a `Near2FarMonitor` from raw outputs, as
    `Near2FarMonitor.get_farfield` without the radius factor.

    Parameters
    ----------
    raw : dict
        Raw outputs {'E<i>', 'H<i>'} of the monitor.
    monitor : Near2FarMonitor
        Initialized monitor, e.g. `AdjointSimulation.readouts[name]`.
    theta, phi : array_like
        Polar and azimuthal angles [rad], broadcast against each other.
    origin : tuple of float, optional
        Phase reference point [m]. Default is the monitor centre.

    Returns
    -------
    dict
        'Etheta', 'Ephi' [V s], shape (n_freq, *broadcast shape). The
        radiation intensity is (|Etheta|^2 + |Ephi|^2) / monitor.impedance.
    """
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    shape = theta.shape
    theta, phi = theta.ravel(), phi.ravel()
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    rhat = np.stack([st * cp, st * sp, ct], axis=1)
    theta_hat = np.stack([ct * cp, ct * sp, -st], axis=1)
    phi_hat = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)
    origin = monitor.center if origin is None else np.asarray(origin, dtype=float)
    k = monitor.wavenumber

    # Radiation vectors N, L of shape (n_freq, n_dir, 3), see Near2FarMonitor
    N = jnp.zeros((k.size, rhat.shape[0], 3), dtype=complex)
    L = jnp.zeros_like(N)
    for i, term in enumerate(monitor._terms):
        weight = np.asarray(term["weight"])
        X, Y, Z = np.meshgrid(*term["coords"], indexing="ij")
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        j_axis, m_axis = term["axes"]
        J = -weight * raw[f"H{i}"]
        M = -weight * raw[f"E{i}"]
        images = [(points, 1.0, 1.0)]
        if monitor._ground is not None:
            axis, position, _ = monitor._ground
            mirrored = points.copy()
            mirrored[:, axis] = 2 * position - points[:, axis]
            # PEC image: tangential J and normal M reversed
            sign_J = 1.0 if j_axis == axis else -1.0
            sign_M = -1.0 if m_axis == axis else 1.0
            images.append((mirrored, sign_J, sign_M))
        for pts, sign_J, sign_M in images:
            phase = np.exp(1j * k[:, None, None] * ((pts - origin) @ rhat.T)[None])
            N = N.at[:, :, j_axis].add(sign_J * jnp.einsum("fp,fpd->fd", J, phase))
            L = L.at[:, :, m_axis].add(sign_M * jnp.einsum("fp,fpd->fd", M, phase))

    N_perp = N - jnp.einsum("fdi,di->fd", N, rhat)[..., None] * rhat
    L_cross_r = jnp.cross(L, rhat[None])
    F = -1j * k[:, None, None] / (4 * np.pi) * (monitor.impedance * N_perp + L_cross_r)
    return {
        "Etheta": jnp.einsum("fdi,di->fd", F, theta_hat).reshape(-1, *shape),
        "Ephi": jnp.einsum("fdi,di->fd", F, phi_hat).reshape(-1, *shape),
    }


def power_waves(V, I, z0):
    """
    Power waves a = (V + z0 I) / (2 sqrt(z0)), b = (V - z0 I) / (2 sqrt(z0))
    (Kurokawa) from the raw 'V', 'I' outputs of a `LumpedReadout` of a port.

    S-parameters are ratios, e.g. S21 = b_2 / a_1 with port 1 driven and
    port 2 passive.

    Parameters
    ----------
    V, I : array of complex
        Port voltage [V s] and current into the structure [A s].
    z0 : float
        Real reference impedance [Ohm].

    Returns
    -------
    a, b : array of complex
        Incident and reflected power waves [sqrt(W) s].
    """
    root = 2 * np.sqrt(z0)
    return (V + z0 * I) / root, (V - z0 * I) / root


def diffraction_orders(raw, monitor, polarization="linear"):
    """
    Diffraction-order amplitudes of a `DiffractionMonitor` from raw outputs,
    as `DiffractionMonitor.get_amplitudes`.

    Parameters
    ----------
    raw : dict
        Raw outputs {'E0', 'H0', 'E1', 'H1'} of the monitor.
    monitor : DiffractionMonitor
        Initialized monitor, e.g. `AdjointSimulation.readouts[name]`.
    polarization : {'linear', 'circular'}, optional
        TE/TM ('up_s', 'up_p', 'down_s', 'down_p') or circular ('up_R',
        'up_L', 'down_R', 'down_L') amplitudes. Default 'linear'.

    Returns
    -------
    dict
        Amplitudes [sqrt(W) s] of shape (n_freq, n_m, n_n), indexed as
        `monitor.orders`; |a|^2 is the flux spectrum of the wave, so order
        efficiencies are |a|^2 divided by the incident flux.
    """
    from ..diffraction import circular_amplitudes, decompose_orders

    amps = decompose_orders(raw, monitor.geometry, jnp)
    amps.pop("net")
    if polarization == "circular":
        return circular_amplitudes(amps, monitor.geometry)
    if polarization != "linear":
        raise ValueError("polarization must be 'linear' or 'circular'.")
    return amps
