# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Hybrid Cartesian / tetrahedral FIT (milestone 1, CPU): a tetrahedral box
patch inside a Cartesian grid, coupled by the hexahedral Nedelec trace on the
box surface (hanging tetrahedral edges). Energy conserving, exactly
transposable. See ``docs/physicsguide_sections/hybrid_cart_tet.md``.
"""

from .mass_mixed import mixed_partition, mixed_tet_mass, well_centred_tets
from .mesh import box_surface_check, delaunay_box_mesh, kuhn_box_mesh, refine_lines
from .solver import HybridFIT3D, MassSolver, box_trace

__all__ = [
    "HybridFIT3D",
    "MassSolver",
    "box_surface_check",
    "box_trace",
    "delaunay_box_mesh",
    "kuhn_box_mesh",
    "mixed_partition",
    "mixed_tet_mass",
    "refine_lines",
    "well_centred_tets",
]
