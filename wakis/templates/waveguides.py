# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Guided-wave templates with mode ports: rectangular metallic waveguide (TE10)
and dielectric slab / strip waveguide (photonics).
"""

import numpy as np

from ..mode_solver import ModeMonitor, ModeSource
from . import formulas
from .shapes import Box
from .simulation import Simulation


def rectangular_waveguide(
    a=22.86e-3,
    b=10.16e-3,
    length=None,
    f_min=None,
    f_max=None,
    iris_width=None,
    iris_thickness=None,
    cells_across=20,
    **simulation_kwargs,
):
    """
    Section of PEC rectangular waveguide along z, excited in TE10 by a mode
    port, with an optional inductive iris (a symmetric diaphragm reducing the
    broad wall to `iris_width`) at z = 0.

    The broad wall a is along x, the narrow wall b along y. The PEC walls are
    the domain boundaries; the guide ends in CPML (matched).

    Parameters
    ----------
    a, b : float, optional
        Inner dimensions [m]. Default WR-90 (22.86 x 10.16 mm).
    length : float, optional
        Length between the CPML layers [m]. Default two guide wavelengths at
        the band centre.
    f_min, f_max : float, optional
        Band [Hz]. Default 1.25 and 1.9 times the TE10 cutoff (single-mode).
    iris_width : float, optional
        Opening of the inductive iris [m]. Default None (empty guide).
    iris_thickness : float, optional
        Iris thickness [m]. Default 0 (PEC sheet).
    cells_across : int, optional
        Cells across the broad wall. Default 20.
    **simulation_kwargs
        Passed to `Simulation`.

    Returns
    -------
    Simulation
        With a 'port1' mode monitor between the source and z = 0 and a
        'port2' monitor downstream. Analyse with `waveguide_s_parameters`.
    """
    fc = formulas.te10_cutoff(a)
    f_min = 1.25 * fc if f_min is None else f_min
    f_max = 1.9 * fc if f_max is None else f_max
    beta = formulas.te10_beta(a, 0.5 * (f_min + f_max)).real
    if length is None:
        length = 2 * (2 * np.pi / beta)
    structures = []
    if iris_width is not None:
        t = 0.0 if iris_thickness is None else iris_thickness
        jaw = (a - iris_width) / 2
        structures = [
            Box((jaw / 2, b / 2, 0), (jaw, b, t), "pec"),
            Box((a - jaw / 2, b / 2, 0), (jaw, b, t), "pec"),
        ]
    simulation_kwargs.setdefault("max_cell", a / cells_across)
    sim = Simulation(
        f_min,
        f_max,
        structures,
        boundaries=(["pec", "pec", "cpml"], ["pec", "pec", "cpml"]),
        domain=(0, a, 0, b, -length / 2, length / 2),
        **simulation_kwargs,
    )
    sim.design = dict(a=a, b=b, length=length, cutoff=fc)
    plane = (np.inf, np.inf, 0)
    z_src = -0.45 * length
    sim.add_source(
        ModeSource(
            sim.pulse, center=(0, 0, z_src), size=plane, mode_frequency=sim.f_center
        )
    )
    for name, z in (("port1", -0.25 * length), ("port2", 0.35 * length)):
        sim.add_monitor(name, ModeMonitor(sim.frequencies, (0, 0, z), plane))
    return sim


def waveguide_s_parameters(sim, port_in="port1", port_out="port2"):
    """
    S11 and S21 of a mode-port simulation, normalized by the injected mode
    amplitude A S(f) (ModeSource amplitude A = 1).

    S11 = a_-(port_in) / (A S),  S21 = a_+(port_out) / (A S). The phases are
    referred to the monitor planes.

    Returns
    -------
    S11, S21 : ndarray of complex
    """
    incident = sim.pulse.fourier_transform(sim.frequencies)
    _, a_minus = sim.monitor(port_in).get_amplitudes()
    a_plus, _ = sim.monitor(port_out).get_amplitudes()
    return a_minus / incident, a_plus / incident


def slab_waveguide(
    wavelength=1550e-9,
    thickness=220e-9,
    core="silicon",
    cladding=1.444**2,
    width=None,
    length=3e-6,
    relative_bandwidth=0.2,
    cells_per_wavelength=20,
    cladding_margin=None,
    **simulation_kwargs,
):
    """
    Straight dielectric waveguide along z with a fundamental-mode source and
    two mode monitors: a 2D slab (x: core thickness, invariant along y) or,
    with `width`, a 3D rectangular strip.

    Parameters
    ----------
    wavelength : float, optional
        Centre vacuum wavelength [m]. Default 1550 nm.
    thickness : float, optional
        Core thickness along x [m]. Default 220 nm.
    core, cladding : float or str, optional
        Relative permittivity or material key ('silicon', 'fused silica' are
        dispersive Sellmeier models). Default: silicon core (Sellmeier) and
        a constant-index SiO2 cladding, eps_r = 1.444**2. Numbers are
        relative permittivities (n**2).
    width : float, optional
        Core width along y [m] for a 3D strip. Default None (2D slab: one
        periodic cell along y).
    length : float, optional
        Length between the CPML layers [m]. Default 3 um.
    relative_bandwidth : float, optional
        Simulated band relative to the centre frequency. Default 0.2.
    cells_per_wavelength : float, optional
        Cells per wavelength in the core at f_max. Default 20.
    cladding_margin : float, optional
        Cladding thickness on each side of the core [m]. Default 0.6 wavelength.
    **simulation_kwargs
        Passed to `Simulation`.

    Returns
    -------
    Simulation
        With 'input' and 'output' mode monitors. Analyse with
        `slab_results`.
    """
    from scipy.constants import c as c_light

    f0 = c_light / wavelength
    margin = 0.6 * wavelength if cladding_margin is None else cladding_margin
    big = 100 * (length + wavelength)  # extends through the CPML
    core_size = (thickness, big if width is None else width, big)
    structures = [
        Box((0, 0, 0), (big, big, big), cladding),
        Box((0, 0, 0), core_size, core),
    ]
    half_x = thickness / 2 + margin
    if width is None:
        cell = wavelength / cells_per_wavelength / 3  # any size: invariant axis
        y_domain, y_bc = (0.0, cell), "periodic"
    else:
        half_y = width / 2 + margin
        y_domain, y_bc = (-half_y, half_y), "cpml"
    sim = Simulation(
        f0 * (1 - relative_bandwidth / 2),
        f0 * (1 + relative_bandwidth / 2),
        structures,
        boundaries=(["cpml", y_bc, "cpml"], ["cpml", y_bc, "cpml"]),
        domain=(-half_x, half_x, *y_domain, -length / 2, length / 2),
        cells_per_wavelength=cells_per_wavelength,
        **simulation_kwargs,
    )
    z_in, z_out = -0.3 * length, 0.4 * length
    sim.design = dict(wavelength=wavelength, thickness=thickness)
    plane = (np.inf, np.inf, 0)
    sim.add_source(ModeSource(sim.pulse, center=(0, 0, -0.4 * length), size=plane))
    for name, z in (("input", z_in), ("output", z_out)):
        sim.add_monitor(name, ModeMonitor(sim.frequencies, (0, 0, z), plane))
    return sim


def slab_results(sim):
    """
    Transmission |a_out|^2 / |a_in|^2 of the fundamental mode, the effective
    index from the phase advance between the monitors,
    neff = -unwrap(arg(a_out / a_in)) c / (2 pi f d), and the mode-solver
    effective index at the monitor frequencies.
    """
    from scipy.constants import c as c_light

    a_in, _ = sim.monitor("input").get_amplitudes()
    a_out, _ = sim.monitor("output").get_amplitudes()
    f = sim.frequencies
    phase = -np.unwrap(np.angle(a_out / a_in))
    # Distance between the grid planes the monitors snapped to
    d = sim.monitor("output").plane.position - sim.monitor("input").plane.position
    # Fix the 2 pi ambiguity with the mode solver index at the first frequency
    modes = sim.monitor("output").modes
    neff_modes = np.array([m[0].neff if m else np.nan for m in modes])
    k0d = 2 * np.pi * f * d / c_light
    phase += 2 * np.pi * np.round((neff_modes[0] * k0d[0] - phase[0]) / (2 * np.pi))
    return dict(
        transmission=np.abs(a_out) ** 2 / np.abs(a_in) ** 2,
        neff=phase / k0d,
        neff_modes=neff_modes,
    )
