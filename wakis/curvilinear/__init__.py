# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Curvilinear (body-fitted, nonorthogonal) FIT in 2D on structured multi-block
quadrilateral meshes: O-grids around round objects, with a radial or
Cartesian PML. See the physics guide section "Curvilinear FIT".
"""

from .mesh import QuadMesh, ogrid_cartesian_mesh, ogrid_mesh, radial_nodes
from .solver2d import CurvilinearFIT2D, hodge_components

__all__ = [
    "CurvilinearFIT2D",
    "QuadMesh",
    "hodge_components",
    "ogrid_cartesian_mesh",
    "ogrid_mesh",
    "radial_nodes",
]
