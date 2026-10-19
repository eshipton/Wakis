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


def test_arm_pml_options():
    """n_pml / pml_dr / pml_d / r_pml reach the solver's PML layer."""
    case = get_case("B3-g5")
    h = 4e-9
    _, s, h_out = Triangles2DArm(n_pml=40).build(case, h)
    cfg = s._pml_cfg
    assert abs(cfg["r0"] - max(case.r_absorber, max(case.circles) + 2 * h_out)) < 1e-15
    assert abs((cfg["r1"] - cfg["r0"]) - 40 * h_out) < 1e-12 * h_out
    _, s, _ = Triangles2DArm(n_pml=20, pml_d=80e-9, r_pml=150e-9).build(case, h)
    cfg = s._pml_cfg
    assert (
        abs(cfg["r0"] - 150e-9) < 1e-15 and abs(cfg["r1"] - cfg["r0"] - 80e-9) < 1e-15
    )
    assert s._pml.spokes.size > 0 and s._pml.chords.size > 0


def test_cartesian_bilinear_exact_for_linear_field_on_nonuniform_grid(monkeypatch):
    """
    `cartesian2d.bilinear` reproduces a linear field exactly at arbitrary
    points on a non-uniform tensor grid; a nearest-neighbour mutation does not.
    """
    import scipy.sparse as sp

    from wakis.benchmarks.arms import cartesian2d as c2d

    xs = np.array([-3.2, -1.7, -0.5, 0.3, 1.4, 2.9]) * 1e-8
    ys = np.array([-2.6, -1.1, 0.2, 1.5, 3.1]) * 1e-8
    xx, yy = np.meshgrid(xs, ys, indexing="xy")  # flat index i + nx * j
    c0, cx, cy = 1.3, 4e7, -2.5e7
    f_grid = (c0 + cx * xx + cy * yy).ravel()
    pts = np.array(
        [[-2.1, -1.8], [-0.9, 0.7], [0.0, 0.0], [0.85, -0.45], [2.2, 2.3]]
    ) * 1e-8
    exact = c0 + cx * pts[:, 0] + cy * pts[:, 1]

    P = c2d.bilinear(xs, ys, pts, len(xs))
    err = np.max(np.abs(P @ f_grid - exact))
    assert err < 1e-12

    def nearest_neighbour(xs, ys, points, nx):
        rows, cols, vals = [], [], []
        for p, (px, py) in enumerate(np.atleast_2d(points)):
            i = int(np.argmin(np.abs(xs - px)))
            j = int(np.argmin(np.abs(ys - py)))
            rows.append(p)
            cols.append(i + nx * j)
            vals.append(1.0)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(rows), nx * len(ys)))

    monkeypatch.setattr(c2d, "bilinear", nearest_neighbour)
    P_mut = c2d.bilinear(xs, ys, pts, len(xs))
    err_mut = np.max(np.abs(P_mut @ f_grid - exact))
    assert err_mut > 1e-1


def test_default_gap_patch_and_enh_rt_diagnostic():
    """
    Triangles2DArm defaults to gap_patch=True (e_gap = patch, e_gap_rt = RT,
    cost.gap_metric = 'patch', cost.n_pml set), and Scattering2DCase.evaluate
    reports enh_rt_peak and spectra['enh_rt'].
    """
    case = get_case("B3-g5")
    case.t_run = 2e-15  # short smoke window
    arm = Triangles2DArm(n_pml=6)
    assert arm.gap_patch is True
    res = arm.run(case, 4e-9)
    assert res["cost"]["n_pml"] == 6
    assert res["cost"]["gap_metric"] == "patch"
    assert "e_gap" in res and "e_gap_rt" in res
    assert not np.allclose(res["e_gap"], res["e_gap_rt"], atol=0, rtol=1e-3)

    ev = case.evaluate(res)
    assert "enh_rt_peak" in ev
    assert "enh_rt" in ev["spectra"]
    # Mutating e_gap_rt scales enh_rt by 4x and changes enh_rt_peak without affecting enh_peak
    res_mut = dict(res, e_gap_rt=2.0 * res["e_gap_rt"])
    ev_mut = case.evaluate(res_mut)
    assert ev_mut["enh_peak"] == ev["enh_peak"]
    assert np.allclose(ev_mut["spectra"]["enh_rt"], 4.0 * np.asarray(ev["spectra"]["enh_rt"]))
    assert abs(ev_mut["enh_rt_peak"] - ev["enh_rt_peak"]) > 0.02

