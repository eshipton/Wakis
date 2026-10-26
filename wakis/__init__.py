# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

from . import (
    beam_fluid,
    boundaries,
    diffraction,
    dispersive,
    field,
    field_monitors,
    figstyle,
    floquet,
    gain,
    gain_semiconductor,
    geometry,
    gridFIT3D,
    inverse,
    logger,
    lumped,
    material_library,
    materials,
    near2far,
    pcsel_cwt,
    resonances,
    schematic,
    solverFIT3D,
    sources,
    sparameters,
    subcell,
    tfsf,
    time_varying,
    wakeSolver,
)
from ._version import __version__
from .beam_fluid import LinearizedBeam
from .diffraction import DiffractionMonitor
from .field import Field
from .field_monitors import DFTMonitor, FieldMonitor, FluxMonitor
from .floquet import AngleSweep, FloquetCell, FloquetMode, FloquetPort
from .gain import (
    FourLevelGain,
    GainMaterial,
    SaturableGain,
    SemiconductorGain,
    adiabatic_chirp,
    amplified_intensity,
    dbr_reflectance,
    fabry_perot_laser_output,
    fabry_perot_threshold_gain,
)
from .gridFIT3D import GridFIT3D
from .logger import Logger
from .lumped import LumpedElement, LumpedPort
from .mode_solver import ModeMonitor, ModeSource, solve_modes
from .near2far import Near2FarMonitor
from .pcsel_cwt import (
    PCSELLayerStack,
    compute_cwt_band_structure,
    compute_cwt_coupling_coefficients,
    double_lattice_coupling_scan,
    fourier_coefficients_phc,
    solve_finite_cwt,
    solve_infinite_cwt,
    solve_vertical_greens_functions,
    solve_vertical_slab_mode,
)
from .resonances import ResonanceMonitor, harminv
from .solverFIT3D import SolverFIT3D
from .subcell import (
    SheetMaterial,
    graphene_sheet,
    metal_sheet,
    resistive_sheet,
    salisbury_screen_reflection,
    sheet_transmission_reflection,
)
from .tfsf import TFSFSource
from .time_varying import (
    CosineModulation,
    SlabTimeProfile,
    StepProfile,
    TimeVaryingDrude,
    TimeVaryingMaterial,
)
from .wakeSolver import WakeSolver
