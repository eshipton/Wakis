# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Self-consistent electromagnetic particle-in-cell module for Wakis FIT."""

from .boundaries import BeamInjector, Collector, ParticleBoundaries
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
    "BeamInjector",
    "Collector",
    "LogicalGridMap",
    "ParticleBoundaries",
    "ParticleCFLError",
    "ParticleSolver",
    "Species",
    "apply_binomial_filter_3d",
    "bspline_weights_1d",
    "cold_plasma_dispersion_discrete",
    "compute_dual_divergence_d",
    "deposit_charge",
    "deposit_current_esirkepov",
    "deposit_current_naive",
    "gather_fields",
    "gauss_residual",
    "integrate_trajectory",
    "lorentz_gamma",
    "push_boris",
    "push_higuera_cary",
    "push_momentum",
    "push_position",
    "push_vay",
]
