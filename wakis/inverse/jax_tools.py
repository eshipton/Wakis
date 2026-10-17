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
        Function rho -> {monitor: {component: complex array}} with the
        structure of `sim.forward`. Its reverse-mode derivative (jax.grad,
        jax.vjp) runs `sim.vjp`; forward-mode (jax.jvp) is not defined.

    Notes
    -----
    The solver runs on the host through `jax.pure_callback`, so the function
    may be used inside `jax.jit`. Each evaluation runs a forward simulation;
    the gradient reuses the latest forward run when rho matches, and reruns
    it otherwise.
    """
    shapes = sim.output_shapes()
    cdtype = jnp.result_type(complex)
    out_spec = {
        name: {
            comp: jax.ShapeDtypeStruct(shape, cdtype) for comp, shape in comps.items()
        }
        for name, comps in shapes.items()
    }
    rho_shape = sim.design.shape

    def _forward(rho):
        result = sim.forward(np.asarray(rho, dtype=float))
        return {
            name: {comp: np.asarray(v, dtype=cdtype) for comp, v in comps.items()}
            for name, comps in result.items()
        }

    def _vjp(rho, cotangents):
        rho = np.asarray(rho, dtype=float)
        if sim.rho is None or not np.array_equal(sim.rho, rho):
            sim.forward(rho)
        # JAX cotangents of complex outputs are conj(dF/dRe + 1j dF/dIm)
        G = {
            name: {comp: np.conj(np.asarray(ct)) for comp, ct in comps.items()}
            for name, comps in cotangents.items()
        }
        return np.asarray(sim.vjp(G), dtype=jnp.result_type(float))

    @jax.custom_vjp
    def simulate(rho):
        return jax.pure_callback(_forward, out_spec, rho)

    def simulate_fwd(rho):
        return simulate(rho), rho

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
