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

from .emitters import (
    SubgridDipole,
    add_bloch_plane_wave,
    homogeneous_dipole_power_2d,
    run_incoherent_dipoles,
    sommerfeld_2d_dipole_above_mirror,
    transfer_matrix_slab_2d,
)
from .geometry import (
    PolySlabGeometry,
    SubgridPlan,
    choose_refinement,
    combine_refinements,
    refine_boxes,
    refine_near_polygons,
    refine_skin_depth,
    skin_depth,
)
from .inverse import (
    SubgridAdjointSimulation,
    SubgridMaterialGrid,
    subgrid_mode_amplitudes,
)
from .mesh import QuadMesh, refine_near
from .monitors import (
    Near2FarMonitor2D,
    SubgridDFTMonitor,
    SubgridDiffractionMonitor,
    SubgridFluxMonitor,
    SubgridModeMonitor,
    SubgridNear2FarMonitor,
    SubgridResonanceMonitor,
)
from .pic import SubgridPICSimulation
from .ports import (
    PIC_POLARIZATION,
    SubgridMode,
    SubgridModeSource,
    polarization_mapping,
    solve_port_modes_2d,
)
from .shapes import Circle, Geometry, Rect
from .solver import SubgridFIT2D

__all__ = [
    "QuadMesh",
    "refine_near",
    "Circle",
    "Rect",
    "Geometry",
    "SubgridFIT2D",
    "PolySlabGeometry",
    "SubgridPlan",
    "refine_boxes",
    "refine_near_polygons",
    "combine_refinements",
    "choose_refinement",
    "skin_depth",
    "refine_skin_depth",
    "PIC_POLARIZATION",
    "polarization_mapping",
    "SubgridMode",
    "solve_port_modes_2d",
    "SubgridModeSource",
    "SubgridDFTMonitor",
    "SubgridModeMonitor",
    "SubgridFluxMonitor",
    "SubgridNear2FarMonitor",
    "Near2FarMonitor2D",
    "SubgridDiffractionMonitor",
    "SubgridResonanceMonitor",
    "SubgridPICSimulation",
    "SubgridDipole",
    "homogeneous_dipole_power_2d",
    "run_incoherent_dipoles",
    "sommerfeld_2d_dipole_above_mirror",
    "add_bloch_plane_wave",
    "transfer_matrix_slab_2d",
    "SubgridMaterialGrid",
    "SubgridAdjointSimulation",
    "subgrid_mode_amplitudes",
]
