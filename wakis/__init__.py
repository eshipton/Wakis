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
    lattices,
    logger,
    lumped,
    material_library,
    materials,
    near2far,
    nonlinear,
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
from .lattices import (
    BZPath,
    CircularInclusion,
    HexagonalLattice,
    PWE2D,
    berry_curvature_from_envelopes,
    dirac_berry_curvature,
    fill_circular_inclusions,
    fit_berry_curvature_patch,
    fit_supercell_bands_at_k,
    translate_supercell_field,
    unfold_signal_pair,
    unfold_supercell_weight,
)
from .logger import Logger
from .lumped import LumpedElement, LumpedPort
from .mode_solver import ModeMonitor, ModeSource, solve_modes
from .near2far import Near2FarMonitor
from .nonlinear import (
    KerrMaterial,
    NonlinearMaterial,
    chi3_to_n2,
    effective_nonlinear_volume,
    field_from_intensity,
    kerr_cavity_frequency_shift,
    lle_cw_bistability,
    lle_soliton_analytic,
    n2_to_chi3,
    optical_intensity,
    solve_lle,
    spm_phase_shift,
    thg_conversion_efficiency,
)
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
