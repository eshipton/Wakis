"""
Second-order gap-field reconstruction of the triangle arm
(`Triangles2DArm._patch_matrix`): a linear in-plane field E(x) = a + B x,
sampled as edge fluxes d_e = eps_0 eps_b L_e E(m_e) . n_e (exact for a linear
field), is recovered exactly at the gap points; the lowest-order
Raviart-Thomas pointwise reconstruction is not (mutation of the order).
"""

import numpy as np
from scipy.constants import epsilon_0 as eps_0

from wakis.benchmarks import get_case
from wakis.benchmarks.arms.triangles2d import Triangles2DArm
from wakis.unstructured import graded_disk_mesh


class _S:
    pass


def test_patch_reconstruction_is_exact_for_linear_fields():
    case = get_case("B3-g5")
    m = graded_disk_mesh([30e-9, 60e-9], 1.5e-9)
    m.cell_tags[:] = 0  # all host
    gpts = np.asarray(case.gap_points)
    P = Triangles2DArm._patch_matrix(m, _S(), gpts, case)
    a = np.array([1.0, -0.5])
    B = np.array([[3e7, -1e7], [2e7, 5e6]])  # [1/m]
    E = a + m.edge_midpoint @ B.T
    t = m.edge_tangent
    n_e = np.stack([t[:, 1], -t[:, 0]], 1)
    d = eps_0 * case.n_background**2 * m.edge_length * np.einsum("ij,ij->i", E, n_e)
    rec = P @ d
    ng = len(gpts)
    exact = a + gpts @ B.T
    err = np.abs(np.stack([rec[:ng], rec[ng:]], 1) - exact).max()
    assert err < 1e-10
    # lowest-order RT (Whitney x z) at the same points is only first order
    cell, W = m.whitney_edge(gpts)
    rt = np.array([[d[m.cell_edges[cell[i]]] @ W[i, :, 1], -(d[m.cell_edges[cell[i]]] @ W[i, :, 0])]
                   for i in range(ng)]) / (eps_0 * case.n_background**2)  # fmt: skip
    assert np.abs(rt - exact).max() > 1e3 * max(err, 1e-14)
