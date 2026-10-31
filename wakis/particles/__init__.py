# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Self-consistent electromagnetic particle-in-cell module for Wakis FIT."""

from .boundaries import BeamInjector, Collector, ParticleBoundaries
from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_STATE_ITEMS,
    CHECKPOINT_STATE_KEYS,
    CPML_PSI_ATTRS,
    compute_config_hash,
    compute_grid_hash,
    load_checkpoint,
    save_checkpoint,
)
from .diagnostics import (
    EmittanceProbe,
    EnergySpectrumMonitor,
    PhaseSpaceMonitor,
    PlaneCurrentProbe,
    TrackRecorder,
)
from .emission import (
    FowlerNordheimEmitter,
    SpaceChargeLimitedEmitter,
    ThermionicEmitter,
    child_langmuir_current_density,
    forbes_barrier_functions,
    fowler_nordheim_current_density,
    richardson_dushman_current_density,
    schottky_barrier_lowering,
)
from .mapping import LogicalGridMap
from .pushers import (
    integrate_trajectory,
    lorentz_gamma,
    push_boris,
    push_higuera_cary,
    push_momentum,
    push_position,
    push_vay,
)
from .shapes import (
    ParticleCFLError,
    bspline_weights_1d,
    deposit_charge,
    deposit_current_esirkepov,
    deposit_current_naive,
    gather_fields,
)
from .solver import (
    ParticleSolver,
    apply_binomial_filter_3d,
    cold_plasma_dispersion_discrete,
    compute_dual_divergence_d,
    gauss_residual,
)
from .species import Species

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CHECKPOINT_STATE_ITEMS",
    "CHECKPOINT_STATE_KEYS",
    "CPML_PSI_ATTRS",
    "BeamInjector",
    "Collector",
    "EmittanceProbe",
    "EnergySpectrumMonitor",
    "FowlerNordheimEmitter",
    "LogicalGridMap",
    "ParticleBoundaries",
    "ParticleCFLError",
    "ParticleSolver",
    "PhaseSpaceMonitor",
    "PlaneCurrentProbe",
    "SpaceChargeLimitedEmitter",
    "Species",
    "ThermionicEmitter",
    "TrackRecorder",
    "apply_binomial_filter_3d",
    "bspline_weights_1d",
    "child_langmuir_current_density",
    "cold_plasma_dispersion_discrete",
    "compute_config_hash",
    "compute_dual_divergence_d",
    "compute_grid_hash",
    "deposit_charge",
    "deposit_current_esirkepov",
    "deposit_current_naive",
    "forbes_barrier_functions",
    "fowler_nordheim_current_density",
    "gather_fields",
    "gauss_residual",
    "integrate_trajectory",
    "load_checkpoint",
    "lorentz_gamma",
    "push_boris",
    "push_higuera_cary",
    "push_momentum",
    "push_position",
    "push_vay",
    "richardson_dushman_current_density",
    "save_checkpoint",
    "schottky_barrier_lowering",
]
