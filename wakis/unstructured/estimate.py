# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Memory estimate of unstructured FIT runs (check before launching big runs).

Calibrated with measured peak RSS (``ru_maxrss``) of `ball_mesh` +
`UnstructuredFIT3D` runs on a 32-core x86-64 machine with NumPy/SciPy
(see the physics guide, "Unstructured FIT"):

* mesh + solver + time stepping: ~2.1 kB per tetrahedron (n_tets 1e3 ... 1e5);
* shift-invert eigen-solve (SuperLU factorization of K - sigma M on the
  free edges): fill-in grows super-linearly, ~``EIG_COEF * n_tets**EIG_EXP``
  bytes;
* 2D meshes: ~1 kB per triangle, sparse LU ~0.5 kB per triangle.

Delaunay tetrahedral meshes have nodes : edges : faces : tets ~ 1 : 6.5 :
11 : 5.5 (Euler), so edge/face counts follow from n_tets.
"""

import numpy as np

#: Python + NumPy/SciPy import baseline [bytes]
BASE = 0.2e9
#: Time-domain cost per tetrahedron [bytes]
TD_PER_TET = 2.1e3
#: Eigen-solve fill-in model coefficient and exponent (bytes = c n^e)
EIG_COEF = 20.0
EIG_EXP = 1.7
#: 2D cost per triangle (time domain and eigen) [bytes]
TD_PER_TRI = 1.0e3
EIG_PER_TRI = 0.5e3


def estimate_memory(mesh_or_n_cells, dim=3, eigen=False):
    """
    Estimated peak memory of an unstructured FIT run.

    Parameters
    ----------
    mesh_or_n_cells : SimplexMesh or int
        The mesh, or its number of cells (tetrahedra / triangles).
    dim : int, optional
        2 or 3 when a cell count is given.
    eigen : bool, optional
        Include a shift-invert eigen-solve (sparse LU fill-in).

    Returns
    -------
    float
        Estimated peak resident memory [bytes].

    Examples
    --------
    >>> estimate_memory(1e5) / 1e9  # GB, time domain, 100k tetrahedra
    0.41
    """
    if hasattr(mesh_or_n_cells, "n_cells"):
        n, dim = mesh_or_n_cells.n_cells, mesh_or_n_cells.dim
    else:
        n = float(mesh_or_n_cells)
    if dim == 2:
        return BASE + n * (TD_PER_TRI + (EIG_PER_TRI if eigen else 0.0))
    mem = BASE + TD_PER_TET * n
    if eigen:
        mem += EIG_COEF * n**EIG_EXP
    return float(np.round(mem, -7))
