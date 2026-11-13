# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
3D Cartesian subgridding for FIT (milestone 1, CPU).

`SubgridMesh3D` nests refined boxes (ratio 2 or 3, balanced) inside a
non-uniform tensor grid; hanging edges on coarse faces take the Nedelec
trace of the coarse face (prolongation ``P_e``, restriction ``P_e^T``).
`SubgridFIT3D` runs the energy-conserving leapfrog on it with an exactly
transposable step. See ``docs/physicsguide_sections/subgrid3d.md``.
"""

from .block import CartesianBlock
from .mesh import Patch, SubgridMesh3D
from .solver import Box, SubgridFIT3D

__all__ = ["CartesianBlock", "Patch", "SubgridMesh3D", "SubgridFIT3D", "Box"]
