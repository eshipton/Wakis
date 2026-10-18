# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Periodic unit cells (metasurfaces, frequency-selective surfaces, thin films)
under normal-incidence plane-wave illumination.

The cell is periodic in x and y (period p) and open along z (CPML). A sheet
current Kx spanning the cell below the structure launches an x-polarized plane
wave along +z. Flux planes between the source and the structure ('reflection')
and above it ('transmission') give, normalized by an empty-cell reference run
on the same mesh,

    T = Phi_t / Phi_t,ref,   R = 1 - Phi_r / Phi_r,ref

(the reflection plane sees the incident minus the reflected flux). These are
the specular (zeroth-order) values while the period is below the wavelength
(f < c / p, no grating lobes).
"""

import numpy as np
from scipy.constants import c as c_light

from ..field_monitors import FluxMonitor
from ..sources import CurrentSource
from . import meshing
from .shapes import Box, material_eps, resolve_material
from .simulation import Simulation


def unit_cell(period, structures, f_min, f_max, cells_per_wavelength=20, **kwargs):
    """
    Generic square unit cell with normal-incidence plane-wave excitation.

    Parameters
    ----------
    period : float or (float, float)
        Lattice period along x and y [m].
    structures : list of Box or Cylinder
        Contents of the cell, centred near z = 0, within the period.
    f_min, f_max : float
        Band [Hz].
    cells_per_wavelength : float, optional
        Mesh density at f_max. Default 20.
    **kwargs
        Passed to `Simulation` (e.g. `max_cell`, `padding`).

    Returns
    -------
    Simulation
        With flux monitors 'reflection' and 'transmission'. Run with
        `reference=True` and analyse with `reflection_transmission`.
    """
    px, py = (period, period) if np.isscalar(period) else period
    z_lo = min(s.bounds[0][2] for s in structures)
    z_hi = max(s.bounds[1][2] for s in structures)
    kwargs.setdefault("padding", meshing.padding(f_min, 0.5))
    sim = Simulation(
        f_min,
        f_max,
        structures,
        boundaries=(["periodic", "periodic", "cpml"], ["periodic", "periodic", "cpml"]),
        domain=(-px / 2, px / 2, -py / 2, py / 2, None, None),
        cells_per_wavelength=cells_per_wavelength,
        **kwargs,
    )
    pad = sim.padding
    sim.design = dict(period=(px, py))
    sim.add_source(
        CurrentSource(
            sim.pulse, component="x", center=(0, 0, z_lo - 0.6 * pad), size=(px, py, 0)
        )
    )
    planes = (("reflection", z_lo - 0.3 * pad), ("transmission", z_hi + 0.5 * pad))
    for name, z in planes:
        sim.add_monitor(name, FluxMonitor(sim.frequencies, (0, 0, z), (px, py, 0)))
    return sim


def dielectric_slab_cell(eps_r=4.0, thickness=10e-3, f_min=2e9, f_max=12e9, **kwargs):
    """
    Homogeneous dielectric slab (thin film) in a unit cell; reference
    solution: `formulas.slab_reflection_transmission` (Airy formulas).

    The transverse period is irrelevant for normal incidence and is set to two
    cells. Parameters are the slab permittivity (number or material key),
    thickness [m] and band [Hz]; **kwargs go to `unit_cell`.
    """
    eps_max = material_eps(resolve_material(eps_r), f_max)
    cells = kwargs.get("cells_per_wavelength", 20)
    period = 2 * meshing.cell_size(f_max, eps_max, cells)
    slab = Box((0, 0, thickness / 2), (period, period, thickness), eps_r)
    sim = unit_cell(period, [slab], f_min, f_max, **kwargs)
    sim.design.update(eps_r=eps_r, thickness=thickness)
    return sim


def patch_array_cell(period=15e-3, patch_size=12e-3, f_min=None, f_max=None, **kwargs):
    """
    Frequency-selective surface of square PEC patches (zero thickness) in
    free space. The array is transparent at low frequency and fully
    reflecting near its first resonance, slightly below c / period.

    Parameters
    ----------
    period : float, optional
        Lattice period [m]. Default 15 mm.
    patch_size : float, optional
        Side of the square patch [m]. Default 12 mm.
    f_min, f_max : float, optional
        Band [Hz]. Default 0.2 and 0.95 times c / period. Fields near the
        grating threshold decay slowly: run with a low `energy_decay`.
    **kwargs
        Passed to `unit_cell`.
    """
    f_wood = c_light / period
    f_min = 0.2 * f_wood if f_min is None else f_min
    f_max = 0.95 * f_wood if f_max is None else f_max
    patch = Box((0, 0, 0), (patch_size, patch_size, 0), "pec")
    sim = unit_cell(period, [patch], f_min, f_max, **kwargs)
    sim.design.update(patch_size=patch_size, grating_threshold=f_wood)
    return sim


def reflection_transmission(sim):
    """
    Specular power reflectance and transmittance of a unit-cell run made with
    `run(reference=True)`.

    Returns
    -------
    R, T : ndarray
        On `sim.frequencies`.
    """
    T = (
        sim.monitor("transmission").get_flux()
        / sim.reference_monitor("transmission").get_flux()
    )
    R = (
        1
        - sim.monitor("reflection").get_flux()
        / sim.reference_monitor("reflection").get_flux()
    )
    return R, T
