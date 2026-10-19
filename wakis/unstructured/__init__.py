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
from .meshers import ball_mesh, disk_mesh, gmsh_ball_mesh, gmsh_mesh, graded_disk_mesh
from .polymesh import PolyMesh
from .polymeshers import hybrid_circles_mesh, voronoi_disk_mesh
from .polysolver2d import PolygonFIT2D
from .solver2d import UnstructuredFIT2D
from .solver3d import UnstructuredFIT3D
from .prism import PrismFIT3D, PrismMesh
from .subtri import SubTriFIT2D, subtri_circles_mesh

__all__ = [
    "SimplexMesh",
    "PolyMesh",
    "PolygonFIT2D",
    "SubTriFIT2D",
    "subtri_circles_mesh",
    "hybrid_circles_mesh",
    "voronoi_disk_mesh",
    "UnstructuredFIT2D",
    "UnstructuredFIT3D",
    "PrismMesh",
    "PrismFIT3D",
    "estimate_memory",
    "ball_mesh",
    "disk_mesh",
    "graded_disk_mesh",
    "gmsh_ball_mesh",
    "gmsh_mesh",
]
