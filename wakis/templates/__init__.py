# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
High-level design templates (CST/Antenna-Magus-style workflows without a GUI).

`Simulation` builds the grid, solver, sources and monitors from a declarative
description with automatic meshing, padding, CPML and run length. The template
functions return a configured `Simulation` for common devices; the `*_results`
helpers extract the usual engineering quantities. See docs/templates.md.
"""

from . import analysis, formulas, meshing
from .antennas import (
    dipole_antenna,
    dipole_results,
    patch_antenna,
    patch_results,
)
from .periodic import (
    dielectric_slab_cell,
    patch_array_cell,
    reflection_transmission,
    unit_cell,
)
from .shapes import Box, Cylinder
from .simulation import Simulation
from .sweeps import optimize, parameter_sweep
from .waveguides import (
    rectangular_waveguide,
    slab_results,
    slab_waveguide,
    waveguide_s_parameters,
)
