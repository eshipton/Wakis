# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Deprecated shim module; use `wakis.subgrid.photonic_circuit` instead."""

import warnings

from .photonic_circuit import SubgridPhotonicCircuit, SubgridPICSimulation

warnings.warn(
    "wakis.subgrid.pic is deprecated; use wakis.subgrid.photonic_circuit instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SubgridPhotonicCircuit", "SubgridPICSimulation"]
