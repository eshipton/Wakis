# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Material library dictionary

Format (non-conductive)::

    {'material key': [eps_r, mu_r]}

Format (conductive)::

    {'material key': [eps_r, mu_r, sigma[S/m]]}

Material keys use lower case only. Permittivity and permeability are relative
to vacuum: eps = eps_r * eps_0 and mu = mu_r * mu_0. Conductivity is in S/m.
"""

import numpy as np

from .dispersive import (  # noqa: F401
    EV,
    Debye,
    DispersiveMaterial,
    Drude,
    Lorentz,
    dispersive_lib,
    eps_from_nk,
    fit_poles,
)
from .gain import (  # noqa: F401
    FourLevelGain,
    GainMaterial,
    SaturableGain,
    amplified_intensity,
    dbr_reflectance,
    fabry_perot_laser_output,
    fabry_perot_threshold_gain,
)
from .material_library import (  # noqa: F401
    info,
    list_materials,
    uniaxial_crystal,
)
from .subcell import (  # noqa: F401
    SheetMaterial,
    graphene_sheet,
    metal_sheet,
    resistive_sheet,
    salisbury_screen_reflection,
    sheet_transmission_reflection,
)

# fmt: off
material_lib = {
    'pec' : [np.inf, 1.],

    'vacuum' : [1.0, 1.0],

    'dielectric' : [10., 1.0],

    'lossy metal' : [10, 1.0, 10],

    'copper' : [5.8e+07, 1.0, 5.8e+07],

    'berillium' : [2.5e+07, 1.0, 2.5e+07],

    'stainless steel' : [1.4e+06, 1.0, 1.4e+06],
}

material_colors = {
    'pec' : 'white',

    'vacuum' : 'tab:blue',

    'dielectric' : 'tab:green',

    'lossy metal' : 'tab:orange',

    'copper' : [0.82745099, 0.698039, 0.49019599],

    'berillium' : [0.82745099, 0.698039, 0.49019599],

    'stainless steel' : 'silver',

    'martensite' : [0.,1.,1.],

    'other' : 'cyan',
}
# fmt: on
