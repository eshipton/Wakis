# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Shared benchmark suite for comparing meshes and schemes (see README.md).

>>> from wakis.benchmarks import get_case, list_cases, run
>>> case = get_case("B2-Au-TE")
>>> ref = case.reference            # exact spectra
>>> rec = run(case, my_arm, h)      # metrics + cost of one simulation
"""

from .cases import CASES, Scattering2DCase, get_case, list_cases
from .runner import Arm, cost_to_target, load_records, run, save_record

__all__ = [
    "Arm",
    "CASES",
    "Scattering2DCase",
    "cost_to_target",
    "get_case",
    "list_cases",
    "load_records",
    "run",
    "save_record",
]
