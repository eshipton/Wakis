# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Unstructured-mesh Finite Integration Technique (triangles in 2D,
tetrahedra in 3D) with a circumcentric (Voronoi) dual grid.

See the physics guide section "Unstructured FIT" and
``docs/unstructured_roadmap.md``.
"""

from .estimate import estimate_memory
from .mesh import SimplexMesh
from .meshers import ball_mesh, disk_mesh, gmsh_ball_mesh, gmsh_mesh
from .solver2d import UnstructuredFIT2D
from .solver3d import UnstructuredFIT3D

__all__ = [
    "SimplexMesh",
    "UnstructuredFIT2D",
    "UnstructuredFIT3D",
    "estimate_memory",
    "ball_mesh",
    "disk_mesh",
    "gmsh_ball_mesh",
    "gmsh_mesh",
]
