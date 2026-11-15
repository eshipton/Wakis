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
from .patch_quality import (
    consistent_spectrum_ratio,
    fixed_patch_nodes,
    normalised_tet_volume,
    smooth_patch,
    worst_normalised_volume,
)
from .solver import HybridFIT3D, MassSolver, box_trace

__all__ = [
    "HybridFIT3D",
    "MassSolver",
    "box_surface_check",
    "box_trace",
    "consistent_spectrum_ratio",
    "delaunay_box_mesh",
    "fixed_patch_nodes",
    "kuhn_box_mesh",
    "mixed_partition",
    "mixed_tet_mass",
    "normalised_tet_volume",
    "refine_lines",
    "smooth_patch",
    "well_centred_tets",
    "worst_normalised_volume",
]
