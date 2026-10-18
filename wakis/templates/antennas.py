# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Antenna templates: probe-fed rectangular patch and centre-fed dipole.
"""

import numpy as np

from ..lumped import LumpedPort
from ..near2far import Near2FarMonitor
from . import formulas, meshing
from .analysis import find_minimum, find_zero_crossings
from .shapes import Box, Cylinder, material_eps, resolve_material
from .simulation import Simulation


def patch_antenna(
    frequency=2.45e9,
    substrate=4.4,
    height=1.6e-3,
    length=None,
    width=None,
    feed_offset=None,
    z0=50.0,
    ground_margin=None,
    relative_bandwidth=0.4,
    cells_per_wavelength=20,
    substrate_cells=3,
    n_far_field=11,
    edge_correction=1 / 3,
    **simulation_kwargs,
):
    """
    Probe-fed rectangular microstrip patch antenna on a finite grounded
    substrate, designed with the transmission-line model.

    Geometry: ground plane (PEC sheet) at z = 0, substrate 0 < z < h, patch
    (PEC sheet) at z = h centred at the origin, resonant length L along x,
    width W along y. A coaxial probe is modelled by a `LumpedPort` along z
    across the substrate at x = -L/2 + feed_offset, y = 0.

    Parameters
    ----------
    frequency : float, optional
        Design frequency [Hz]. Default 2.45 GHz.
    substrate : float or str, optional
        Relative permittivity, or 'fr4' / 'ro4350b' (dispersive, lossy, from
        `dispersive_lib`). Default 4.4 (lossless FR-4).
    height : float, optional
        Substrate thickness [m]. Default 1.6 mm.
    length, width : float, optional
        Patch dimensions [m]. Default: `formulas.patch_design`.
    feed_offset : float, optional
        Feed distance from the radiating edge [m]. Default: the position where
        the cavity-model input resistance equals `z0`.
    z0 : float, optional
        Port reference impedance [Ohm]. Default 50.
    ground_margin : float, optional
        Ground plane and substrate extension beyond the patch on each side [m].
        Default max(6 h, lambda0 / 10).
    relative_bandwidth : float, optional
        Simulated band (f_max - f_min) / frequency. Default 0.4.
    cells_per_wavelength : float, optional
        Mesh density at f_max in each material. Default 20.
    substrate_cells : int, optional
        Minimum number of cells across the substrate. Default 3.
    n_far_field : int, optional
        Number of far-field frequencies across the band. Default 11.
    edge_correction : float, optional
        The meshed patch is made smaller by this fraction of the lateral cell
        size on each edge. A zero-thickness metal sheet ending on a mesh node
        behaves as if it extended about a third of a cell further (fringing
        field of the staircased edge); without the correction the resonance is
        lower by about 2 x 0.37 cells / L. Default 1/3, 0 disables it.
    **simulation_kwargs
        Passed to `Simulation`.

    Returns
    -------
    Simulation
        With a port, a Near2FarMonitor 'far_field' and the design values in
        `sim.design`. Analyse the run with `patch_results`.
    """
    material = resolve_material(substrate)
    eps_r = material_eps(material, frequency)
    design = formulas.patch_design(frequency, eps_r, height)
    width = design["width"] if width is None else width
    length = design["length"] if length is None else length
    edge_resistance = formulas.patch_edge_resistance(length, width, frequency)
    if feed_offset is None:
        feed_offset = formulas.patch_feed_offset(length, edge_resistance, z0)
    if ground_margin is None:
        ground_margin = max(6 * height, meshing.wavelength(frequency) / 10)
    design.update(
        width=width,
        length=length,
        feed_offset=feed_offset,
        edge_resistance=edge_resistance,
        resonance=formulas.patch_resonance(length, width, eps_r, height),
        directivity=formulas.patch_directivity(length, width, frequency),
        eps_r=eps_r,
    )

    half_bw = relative_bandwidth / 2
    cell = meshing.cell_size(frequency * (1 + half_bw), eps_r, cells_per_wavelength)
    if simulation_kwargs.get("max_cell") is not None:
        cell = min(cell, simulation_kwargs["max_cell"])
    trim = 2 * edge_correction * cell
    ground = (length + 2 * ground_margin, width + 2 * ground_margin)
    structures = [
        Box((0, 0, height / 2), (*ground, height), material),
        Box((0, 0, 0), (*ground, 0), "pec"),
        Box((0, 0, height), (length - trim, width - trim, 0), "pec"),
    ]
    simulation_kwargs.setdefault("min_cells_per_feature", substrate_cells)
    sim = Simulation(
        frequency * (1 - half_bw),
        frequency * (1 + half_bw),
        structures,
        cells_per_wavelength=cells_per_wavelength,
        **simulation_kwargs,
    )
    sim.design = design
    sim.add_source(
        LumpedPort(
            sim.pulse,
            component="z",
            center=(-length / 2 + feed_offset, 0, height / 2),
            size=(0, 0, height),
            impedance=z0,
        )
    )
    # Equivalence box half-way between the structure and the CPML
    gap = sim.padding / 2
    sim.add_monitor(
        "far_field",
        Near2FarMonitor(
            np.linspace(sim.f_min, sim.f_max, n_far_field),
            center=(0, 0, height / 2),
            size=(ground[0] + 2 * gap, ground[1] + 2 * gap, height + 2 * gap),
        ),
    )
    return sim


def patch_results(sim):
    """
    Key results of a patch run: resonance (|S11| minimum), S11 there, input
    impedance, -10 dB bandwidth and maximum directivity at the far-field
    frequency closest to resonance.

    Returns
    -------
    dict
        'resonance' [Hz], 's11_min_db' [dB], 'impedance' [Ohm] (at
        resonance), 'directivity' (linear), 'directivity_dbi',
        'resonance_error' (relative to the TLM formula), 'bandwidth' [Hz].
    """
    from .analysis import bandwidth, to_db

    # The port spectra can be evaluated at any frequency: resolve the narrow
    # resonance finely (patch bandwidths are a few percent)
    freqs = np.linspace(sim.f_min, sim.f_max, 1001)
    s11 = sim.s11(freqs)
    f_res, s11_min = find_minimum(freqs, to_db(s11))
    n2f = sim.monitor("far_field")
    i = int(np.argmin(np.abs(n2f.frequencies - f_res)))
    D_max, _ = n2f.max_directivity()
    return dict(
        resonance=f_res,
        resonance_error=f_res / sim.design["resonance"] - 1,
        s11_min_db=s11_min,
        impedance=complex(sim.input_impedance([f_res])[0]),
        bandwidth=bandwidth(freqs, s11),
        directivity=float(D_max[i]),
        directivity_dbi=float(10 * np.log10(D_max[i])),
        directivity_frequency=float(n2f.frequencies[i]),
    )


def dipole_antenna(
    frequency=1e9,
    length=None,
    radius=0.0,
    gap=None,
    z0=73.0,
    relative_bandwidth=0.6,
    cells_per_wavelength=40,
    n_far_field=11,
    **simulation_kwargs,
):
    """
    Centre-fed straight wire dipole along z with a lumped gap port.

    Parameters
    ----------
    frequency : float, optional
        Centre of the simulated band [Hz]. Default 1 GHz.
    length : float, optional
        Total length [m]. Default half a wavelength at `frequency`.
    radius : float, optional
        Wire radius [m]. Default 0: a PEC line of E edges (thin-wire limit
        whose effective radius is a fraction of the cell size).
    gap : float, optional
        Feed gap [m]. Default one cell.
    z0 : float, optional
        Port reference impedance [Ohm]. Default 73.
    relative_bandwidth : float, optional
        Simulated band (f_max - f_min) / frequency. Default 0.6.
    cells_per_wavelength : float, optional
        Mesh density at f_max. Default 40 (resonance length to ~1 %).
    n_far_field : int, optional
        Far-field frequencies across the band. Default 11.
    **simulation_kwargs
        Passed to `Simulation`.

    Returns
    -------
    Simulation
        With a port and a Near2FarMonitor 'far_field'. Analyse with
        `dipole_results`.
    """
    half_bw = relative_bandwidth / 2
    f_max = frequency * (1 + half_bw)
    if length is None:
        length = meshing.wavelength(frequency) / 2
    if gap is None:
        gap = meshing.cell_size(f_max, 1.0, cells_per_wavelength)
    arm = (length - gap) / 2
    structures = [
        Cylinder((0, 0, (gap + arm) / 2), radius, arm, "pec"),
        Cylinder((0, 0, -(gap + arm) / 2), radius, arm, "pec"),
    ]
    simulation_kwargs.setdefault("min_cells_per_feature", 1)
    sim = Simulation(
        frequency * (1 - half_bw),
        f_max,
        structures,
        cells_per_wavelength=cells_per_wavelength,
        **simulation_kwargs,
    )
    sim.design = dict(length=length, gap=gap, radius=radius)
    sim.add_source(LumpedPort(sim.pulse, component="z", size=(0, 0, gap), impedance=z0))
    box = 2 * radius + sim.padding
    sim.add_monitor(
        "far_field",
        Near2FarMonitor(
            np.linspace(sim.f_min, sim.f_max, n_far_field),
            size=(box, box, length + sim.padding),
        ),
    )
    return sim


def dipole_results(sim):
    """
    Key results of a dipole run: first resonance (Im Zin = 0 with positive
    slope), Zin there, length in wavelengths, and maximum directivity at the
    far-field frequency closest to resonance.
    """
    freqs = sim.frequencies
    z_in = sim.input_impedance(freqs)
    crossings = find_zero_crossings(freqs, z_in.imag)
    slope = np.interp(crossings + 1e-3 * (freqs[1] - freqs[0]), freqs, z_in.imag)
    rising = crossings[slope > 0]
    f_res = rising[0] if rising.size else np.nan
    n2f = sim.monitor("far_field")
    i = int(np.argmin(np.abs(n2f.frequencies - f_res)))
    D_max, _ = n2f.max_directivity()
    return dict(
        resonance=f_res,
        length_wavelengths=sim.design["length"] / meshing.wavelength(f_res),
        impedance=complex(sim.input_impedance([f_res])[0]),
        directivity=float(D_max[i]),
        directivity_dbi=float(10 * np.log10(D_max[i])),
        directivity_frequency=float(n2f.frequencies[i]),
    )
