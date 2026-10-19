# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""High-order time-domain schemes: nodal DGTD on curved triangles
(`DGTD2D`) and the Cartesian (2,4) staggered scheme (`FDTD24`)."""

from .dgtd2d import DGTD2D, LSRK4, lsrk_stability
from .mesh2d import TriMesh, circles_mesh
from .nodal2d import ReferenceTriangle, reference_triangle

__all__ = [
    "DGTD2D", "LSRK4", "lsrk_stability", "TriMesh", "circles_mesh",
    "ReferenceTriangle", "reference_triangle",
]
