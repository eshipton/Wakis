# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Local Cartesian refinement (subgridding) for 2D FIT.

`QuadMesh` builds a coarse rectilinear grid with nested refined patches
(hanging nodes constrained by linear interpolation); `SubgridFIT2D` runs the
TM/TE leapfrog on it with a symmetric interface operator (provably stable
under a computed CFL limit, exactly transposable), sub-pixel material
averaging, PML, scattered-field plane waves and Drude/Lorentz ADE media.
"""

from .mesh import QuadMesh, refine_near
from .shapes import Circle, Geometry, Rect
from .solver import SubgridFIT2D

__all__ = ["QuadMesh", "refine_near", "Circle", "Rect", "Geometry", "SubgridFIT2D"]
