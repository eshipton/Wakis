"""Reference diffraction efficiencies of a lamellar grating computed with fmmax.

Generates the reference numbers of `tests/test_032_diffraction.py` with the
independent RCWA (Fourier modal method) solver fmmax
(https://github.com/facebookresearch/fmmax, MIT), as used by the invrs-gym
diffract challenges. fmmax is NOT a Wakis dependency; run this script in an
environment that has it, e.g. the invrs-gym virtual environment:

    python benchmarks/diffraction_fmmax_reference.py

Geometry (lengths in units of the free-space wavelength lambda0):
substrate eps = 2.25 (z < 0), grating layer 0 < z < 0.5 with teeth of
eps = 4 for 0 <= x < P/2 and air elsewhere, air above; period P = 1.5 along x
(1D grating, invariant along y). A plane wave is incident from the substrate
along +z at polar angle theta (azimuth 0, plane of incidence xz); TE has E
along y, TM has H along y. Efficiencies are the powers of the reflected
(substrate, -z) and transmitted (air, +z) orders over the incident power,
summed over the output polarizations as in invrs-gym `grating_efficiency`.
Order m has tangential wavevector k_inc + 2 pi m / P.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import fmmax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

PERIOD, THICKNESS = 1.5, 0.5
EPS_SUBSTRATE, EPS_TOOTH, EPS_AIR = 2.25, 4.0, 1.0


def efficiencies(theta_deg, n_orders=101, n_pixels=3000, wavelength=1.0):
    """Reflected and transmitted order efficiencies for TE and TM incidence."""
    lattice = fmmax.LatticeVectors(u=PERIOD * fmmax.X, v=fmmax.Y)
    m = np.arange(-(n_orders // 2), n_orders // 2 + 1)
    coeffs = np.stack([m, np.zeros_like(m)], axis=1)
    # Order (0, 0) first, as assumed by the amplitude indexing below
    coeffs = coeffs[np.argsort(np.abs(m), kind="stable")]
    expansion = fmmax.Expansion(basis_coefficients=coeffs)
    n = expansion.num_terms

    x = (np.arange(n_pixels) + 0.5) / n_pixels * PERIOD
    grating = np.where(x < PERIOD / 2, EPS_TOOTH, EPS_AIR)[:, None].astype(complex)
    wl = jnp.asarray(wavelength)
    k_in = fmmax.plane_wave_in_plane_wavevector(
        wavelength=wl,
        polar_angle=jnp.asarray(np.radians(theta_deg)),
        azimuthal_angle=jnp.asarray(0.0),
        permittivity=jnp.asarray(EPS_SUBSTRATE),
    )
    layers = [
        fmmax.eigensolve_isotropic_media(
            wavelength=wl,
            in_plane_wavevector=k_in,
            primitive_lattice_vectors=lattice,
            permittivity=jnp.asarray(eps),
            expansion=expansion,
            formulation=fmmax.Formulation.JONES_DIRECT,
        )
        for eps in (
            np.full((1, 1), EPS_SUBSTRATE, complex),
            grating,
            np.full((1, 1), EPS_AIR, complex),
        )
    ]
    s = fmmax.stack_s_matrix(layers, [jnp.asarray(0.0), THICKNESS, jnp.asarray(0.0)])

    # Forward (+z) incident amplitudes in the substrate: TE (index 0, E along y
    # for this plane of incidence) and TM (index n, H along y)
    incident = jnp.zeros((2 * n, 2), complex).at[0, 0].set(1.0).at[n, 1].set(1.0)
    inc_power, _ = fmmax.amplitude_poynting_flux(
        incident, jnp.zeros_like(incident), layers[0]
    )
    inc_power = jnp.sum(inc_power, axis=0)
    _, refl = fmmax.amplitude_poynting_flux(
        jnp.zeros_like(incident), s.s21 @ incident, layers[0]
    )
    trans, _ = fmmax.amplitude_poynting_flux(
        s.s11 @ incident, jnp.zeros_like(incident), layers[-1]
    )
    out = {}
    for p, pol in enumerate(("TE", "TM")):
        R = -(refl[:n, p] + refl[n:, p]) / inc_power[p]
        T = (trans[:n, p] + trans[n:, p]) / inc_power[p]
        orders = coeffs[:, 0]
        out[pol] = {
            "R": {int(o): float(r) for o, r in zip(orders, R) if abs(r) > 1e-12},
            "T": {int(o): float(t) for o, t in zip(orders, T) if abs(t) > 1e-12},
        }
    return out


if __name__ == "__main__":
    for theta in (0.0, 15.0):
        print(f"theta = {theta} deg (in the substrate), lambda0 = 1")
        results = {
            n_orders: efficiencies(theta, n_orders) for n_orders in (51, 101, 201)
        }
        for pol in ("TE", "TM"):
            for side in ("R", "T"):
                best = results[201][pol][side]
                line = ", ".join(
                    f"{m}: {best[m]:.5f}" for m in sorted(best) if best[m] > 1e-9
                )
                spread = max(
                    abs(results[k][pol][side].get(m, 0.0) - best[m])
                    for k in results
                    for m in best
                )
                print(
                    f"  {pol} {side}: {{{line}}}  (max change 51->201 orders {spread:.1e})"
                )
