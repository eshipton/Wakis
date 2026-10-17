# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Exact discrete adjoint and inverse-design tools.

The NumPy core (`AdjointSimulation`, `AdjointStepper`, `MaterialGrid`) has no
extra dependencies. The JAX bindings (`jax_simulation`, filters and
projection) are imported lazily and need ``pip install wakis[inverse]``.
"""

from .adjoint import AdjointSimulation, AdjointStepper
from .design import MaterialGrid

_JAX_NAMES = ("jax_simulation", "conic_filter", "gaussian_filter", "tanh_projection")


def __getattr__(name):
    if name in _JAX_NAMES:
        from . import jax_tools

        return getattr(jax_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
