# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Part 3 verification for CFS-CPML with refined patches reaching into or through
the absorbing layer (`wakis.subgrid3d.pml`):
  - Semi-discrete generator ``G = cpml_generator(solver)`` vs the ``dt -> 0``
    limit of ``step_vec``, no-patch control spectrum, and eigenvector
    localisation of the ``sigma_sampling="fine"`` interface instability;
  - Hypothesis C1 (`sigma_sampling="coarse_cell"` and `"coarse_cell_dual"`):
    eliminates the ``5.443e-03 sigma_max`` low-frequency interface instability
    for patches reaching 1--2 cells into a 4-cell layer or spanning the
    waveguide, and preserves exact ``step_vec_T`` adjoint accuracy (<= 1e-15);
  - T1 full spectrum check across 2:1 and 3:1 patches reaching into and through
    layers along x, y, z and a two-layer corner at ``alpha / (2 pi) = 10 MHz``
    (strict xfail per Part 3 step 5 due to residual ``1.883e-04..4.111e-04
    sigma_max`` grazing cutoff coupling in ``x_into_3to1`` and
    ``x/y/z_through_2to1``).
"""

import numpy as np
import pytest
from scipy.linalg import eig, eigvals

from wakis.subgrid3d import SubgridLTSFIT3D, SubgridMesh3D
from wakis.subgrid3d.pml import cpml_generator, diagnose_cpml_eigenvector


def _make_cpml_solver(
    dims=(8, 5, 5),
    patch_cells=None,
    ratio=2,
    layer_axes=((0, 1, 4),),
    alpha_hz=10.0e6,
    sigma_sampling="fine",
    R0=1e-6,
    order=3.0,
    H=0.04,
    dt=None,
):
    axes = [np.linspace(0.0, n * H, n + 1) for n in dims]
    m = SubgridMesh3D(*axes)
    if patch_cells is not None:
        bounds = tuple((lo * H, hi * H) for lo, hi in patch_cells)
        m.add_box(bounds, ratio=ratio)
    th = np.zeros((3, 2))
    for d, side, n_cells in layer_axes:
        th[d, side] = n_cells * H
    return SubgridLTSFIT3D(
        m,
        lts=False,
        dt=dt,
        pml=dict(
            thickness=th,
            R0=R0,
            order=order,
            alpha_hz=alpha_hz,
            allow_patches=True,
            sigma_sampling=sigma_sampling,
        ),
    )


def _dot_defect(s, mutate=False):
    rng = np.random.default_rng(7)
    x = rng.standard_normal(s.n_state)
    y = rng.standard_normal(s.n_state)
    i = rng.standard_normal(s.mesh.n_edges)
    W = np.asarray(s._Wb)
    sb = np.sqrt(np.median(s.D) / np.median(W[W > 0]))
    x[: s._nb] *= sb
    y[: s._nb] *= sb
    fx = s.step_vec(x, i)
    if mutate:
        for p in s._pml:
            p["ah"] = 1.1 * p["ah"]
    xb, ib = s.step_vec_T(y)
    scale = np.linalg.norm(fx) * np.linalg.norm(y)
    return float(abs(fx @ y - (x @ xb + i @ ib)) / scale)


def test_cpml_generator_diagnosis_and_nopatch_control():
    """Verify the semi-discrete generator G against the dt -> 0 limit of
    step_vec, confirm the no-patch control <= 1e-7 sigma_max at 10 MHz,
    localise the unstable interface eigenvector under sigma_sampling='fine',
    confirm C1 ('coarse_cell') stabilises patches reaching 2 cells into the
    4-cell layer or spanning the domain, and verify step_vec_T <= 1e-15."""
    # 1. Verify (M(dt) - I) / dt -> G as dt -> 0 on a tiny patched mesh
    dt_small = 1.0e-15
    s_tiny = _make_cpml_solver(
        dims=(5, 4, 4),
        patch_cells=((1, 4), (1, 3), (1, 3)),
        ratio=2,
        layer_axes=((0, 1, 2),),
        alpha_hz=10.0e6,
        sigma_sampling="coarse_cell",
        dt=dt_small,
    )
    G_tiny = cpml_generator(s_tiny).toarray()
    n_st = s_tiny.n_state
    zero_i = np.zeros(s_tiny.mesh.n_edges)
    # Check action on a random state vector (since b is updated before e in
    # leapfrog, M(dt) = I + dt G + O(dt^2))
    rng = np.random.default_rng(11)
    v = rng.standard_normal(n_st)
    s_tiny_2 = _make_cpml_solver(
        dims=(5, 4, 4),
        patch_cells=((1, 4), (1, 3), (1, 3)),
        ratio=2,
        layer_axes=((0, 1, 2),),
        alpha_hz=10.0e6,
        sigma_sampling="coarse_cell",
        dt=1.0e-13,
    )
    g_fd1 = (s_tiny.step_vec(v, zero_i) - v) / 1.0e-15
    g_fd2 = (s_tiny_2.step_vec(v, zero_i) - v) / 1.0e-13
    err_fd1 = float(np.linalg.norm(g_fd1 - G_tiny @ v) / np.linalg.norm(G_tiny @ v))
    err_fd2 = float(np.linalg.norm(g_fd2 - G_tiny @ v) / np.linalg.norm(G_tiny @ v))
    assert err_fd1 < 1e-5 and err_fd2 < 1e-3

    # 2. No-patch control on 8 x 5 x 5 with 4-cell +x layer (n_state = 1393)
    s_nop_10m = _make_cpml_solver(
        dims=(8, 5, 5), patch_cells=None, layer_axes=((0, 1, 4),), alpha_hz=10.0e6
    )
    s_nop_0 = _make_cpml_solver(
        dims=(8, 5, 5), patch_cells=None, layer_axes=((0, 1, 4),), alpha_hz=0.0
    )
    smax = s_nop_10m.cpml.sigma_max
    nop_re_10m = float(eigvals(cpml_generator(s_nop_10m).toarray()).real.max() / smax)
    nop_re_0 = float(eigvals(cpml_generator(s_nop_0).toarray()).real.max() / smax)
    assert nop_re_10m <= 1e-7

    # 3. Diagnose unstable eigenvector on 8 x 5 x 5 with 2:1 patch (3,1,1)..(7,4,4)
    s_fine = _make_cpml_solver(
        dims=(8, 5, 5),
        patch_cells=((3, 7), (1, 4), (1, 4)),
        ratio=2,
        layer_axes=((0, 1, 4),),
        alpha_hz=10.0e6,
        sigma_sampling="fine",
    )
    vals_f, vecs_f = eig(cpml_generator(s_fine).toarray())
    diag = diagnose_cpml_eigenvector(s_fine, vals=vals_f, vecs=vecs_f)
    s_fine_0 = _make_cpml_solver(
        dims=(8, 5, 5),
        patch_cells=((3, 7), (1, 4), (1, 4)),
        ratio=2,
        layer_axes=((0, 1, 4),),
        alpha_hz=0.0,
        sigma_sampling="fine",
    )
    fine_re_0 = float(eigvals(cpml_generator(s_fine_0).toarray()).real.max() / smax)
    assert diag["re_over_sigma_max"] > 1e-3
    assert diag["interface_face_share_of_b"] > 0.90
    assert diag["interface_edge_share_of_e"] > 0.50

    # 4. Hypothesis C1 ('coarse_cell') on 2:1 patch 2 cells into 4-cell layer
    # and on a 2:1 tube spanning all x
    s_into_fine = _make_cpml_solver(
        dims=(8, 5, 5),
        patch_cells=((3, 6), (1, 4), (1, 4)),
        ratio=2,
        layer_axes=((0, 1, 4),),
        alpha_hz=10.0e6,
        sigma_sampling="fine",
    )
    s_into_c1 = _make_cpml_solver(
        dims=(8, 5, 5),
        patch_cells=((3, 6), (1, 4), (1, 4)),
        ratio=2,
        layer_axes=((0, 1, 4),),
        alpha_hz=10.0e6,
        sigma_sampling="coarse_cell",
    )
    into_re_fine = float(
        eigvals(cpml_generator(s_into_fine).toarray()).real.max() / smax
    )
    into_re_c1 = float(eigvals(cpml_generator(s_into_c1).toarray()).real.max() / smax)
    assert into_re_fine > 1e-3
    assert into_re_c1 <= 1e-6

    s_tube_c1 = _make_cpml_solver(
        dims=(6, 4, 4),
        patch_cells=((0, 6), (1, 3), (1, 3)),
        ratio=2,
        layer_axes=((0, 1, 3),),
        alpha_hz=10.0e6,
        sigma_sampling="coarse_cell",
    )
    tube_re_c1 = float(
        eigvals(cpml_generator(s_tube_c1).toarray()).real.max()
        / s_tube_c1.cpml.sigma_max
    )
    assert tube_re_c1 <= 1e-6

    # 5. Exact transpose dot-product test for coarse_cell and coarse_cell_dual
    s_dual = _make_cpml_solver(
        dims=(8, 5, 5),
        patch_cells=((3, 7), (1, 4), (1, 4)),
        ratio=2,
        layer_axes=((0, 1, 4),),
        alpha_hz=10.0e6,
        sigma_sampling="coarse_cell_dual",
    )
    dot_c1 = _dot_defect(s_into_c1)
    dot_dual = _dot_defect(s_dual)
    dot_mut = _dot_defect(s_into_c1, mutate=True)
    assert dot_c1 <= 1e-15 and dot_dual <= 1e-15
    assert dot_mut > 1e-8

    q0, q1, q2, q3 = diag["depth_quartile_share"]
    b0 = diag["block_energy_share"][0]
    b1 = diag["block_energy_share"][1]
    print(
        f"NUMBERS test_cpml_generator_diagnosis_and_nopatch_control: "
        f"sigma_max={smax:.3e} n_state_nop={s_nop_10m.n_state} n_state_fine={s_fine.n_state} "
        f"fd_err={err_fd1:.3e} nop_re_10m={nop_re_10m:.3e} nop_re_0={nop_re_0:.3e} "
        f"fine_re_10m={diag['re_over_sigma_max']:.3e} fine_re_0={fine_re_0:.3e} "
        f"fine_lam_re={diag['lambda_max'].real:.3e} fine_freq_hz={diag['freq_hz']:.3e} "
        f"blk0_share={b0:.4f} blk1_share={b1:.4f} "
        f"if_face_b={diag['interface_face_share_of_b']:.4f} "
        f"if_edge_e={diag['interface_edge_share_of_e']:.4f} "
        f"depth_q={q0:.3f},{q1:.3f},{q2:.3f},{q3:.3f} "
        f"into2_re_fine={into_re_fine:.3e} into2_re_c1={into_re_c1:.3e} "
        f"tube_re_c1={tube_re_c1:.3e} "
        f"dot_c1={dot_c1:.3e} dot_dual={dot_dual:.3e} dot_mut={dot_mut:.3e}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Part 3 step 5: C1 ('coarse_cell') eliminates the 5.443e-03 sigma_max "
        "interface instability for patches reaching into a 4-cell layer "
        "(x_into_2to1: 5.380e-09, y_into_2to1: 6.361e-09, z_into_2to1: 5.622e-09 "
        "<= 1e-6), x_through_3to1 (1.582e-09 <= 1e-6), and corner_xy_2to1 "
        "(4.222e-09 <= 1e-6), but 3D patches coupling a cutoff step into the "
        "deepest backing cells (x_into_3to1: 1.883e-04; x_through_2to1, "
        "y_through_2to1, z_through_2to1: 4.111e-04) retain 1.883e-04..4.111e-04 "
        "sigma_max > 1e-6 at 1.953e+09 Hz due to non-commuting transverse curl "
        "on side-wall faces and grazing cutoff coupling; requires Candidate C2 "
        "(uniaxial Hodge-mass lumping)."
    ),
)
def test_t1_cpml_patches_spectrum():
    """T1: max Re lambda(G) <= 1e-6 sigma_max across 2:1 and 3:1 patches
    reaching into and through layers along x, y, z and a corner where two
    layers overlap, at alpha / (2 pi) = 10 MHz."""
    cases = [
        # (name, dims, patch_cells, ratio, layer_axes)
        ("nopatch_x", (6, 4, 4), None, 2, ((0, 1, 3),)),
        ("x_into_2to1", (7, 5, 5), ((3, 5), (1, 4), (1, 4)), 2, ((0, 1, 4),)),
        ("x_through_2to1", (6, 4, 4), ((2, 6), (1, 3), (1, 3)), 2, ((0, 1, 3),)),
        ("y_into_2to1", (5, 7, 5), ((1, 4), (3, 5), (1, 4)), 2, ((1, 1, 4),)),
        ("y_through_2to1", (4, 6, 4), ((1, 3), (2, 6), (1, 3)), 2, ((1, 1, 3),)),
        ("z_into_2to1", (5, 5, 7), ((1, 4), (1, 4), (3, 5)), 2, ((2, 1, 4),)),
        ("z_through_2to1", (4, 4, 6), ((1, 3), (1, 3), (2, 6)), 2, ((2, 1, 3),)),
        ("x_into_3to1", (5, 3, 3), ((1, 3), (1, 2), (1, 2)), 3, ((0, 1, 3),)),
        ("x_through_3to1", (5, 3, 3), ((2, 5), (1, 2), (1, 2)), 3, ((0, 1, 3),)),
        (
            "corner_xy_2to1",
            (5, 5, 4),
            ((2, 5), (2, 5), (1, 3)),
            2,
            ((0, 1, 3), (1, 1, 3)),
        ),
    ]

    re_c1_10m = {}
    re_c1_0 = {}
    re_fine_10m = {}
    freq_c1_10m = {}

    for name, dims, patch_cells, ratio, layer_axes in cases:
        s_10m = _make_cpml_solver(
            dims=dims,
            patch_cells=patch_cells,
            ratio=ratio,
            layer_axes=layer_axes,
            alpha_hz=10.0e6,
            sigma_sampling="coarse_cell",
        )
        smax = s_10m.cpml.sigma_max
        vals_10m = eigvals(cpml_generator(s_10m).toarray())
        idx_10m = int(np.argmax(vals_10m.real))
        re_c1_10m[name] = float(vals_10m[idx_10m].real / smax)
        freq_c1_10m[name] = float(abs(vals_10m[idx_10m].imag) / (2 * np.pi))

        s_0 = _make_cpml_solver(
            dims=dims,
            patch_cells=patch_cells,
            ratio=ratio,
            layer_axes=layer_axes,
            alpha_hz=0.0,
            sigma_sampling="coarse_cell",
        )
        re_c1_0[name] = float(eigvals(cpml_generator(s_0).toarray()).real.max() / smax)

        if patch_cells is not None:
            s_f = _make_cpml_solver(
                dims=dims,
                patch_cells=patch_cells,
                ratio=ratio,
                layer_axes=layer_axes,
                alpha_hz=10.0e6,
                sigma_sampling="fine",
            )
            re_fine_10m[name] = float(
                eigvals(cpml_generator(s_f).toarray()).real.max() / smax
            )

    nop_10m = re_c1_10m["nopatch_x"]
    nop_0 = re_c1_0["nopatch_x"]
    patched_names = [c[0] for c in cases if c[2] is not None]
    worst_name = max(patched_names, key=lambda k: re_c1_10m[k])
    max_c1_10m = re_c1_10m[worst_name]
    c1_max_freq_hz = freq_c1_10m[worst_name]
    max_c1_0 = max(re_c1_0[k] for k in patched_names)
    failing_c1_10m = {k: re_c1_10m[k] for k in patched_names if re_c1_10m[k] > 1e-6}
    min_fail_c1_10m = min(failing_c1_10m.values())
    max_fine_10m = max(re_fine_10m[k] for k in patched_names)
    min_fine_10m = min(re_fine_10m[k] for k in patched_names)

    summary_10m = ",".join(f"{k}:{re_c1_10m[k]:.3e}" for k in patched_names)
    summary_fail = ",".join(f"{k}:{v:.3e}" for k, v in failing_c1_10m.items())
    summary_0 = ",".join(f"{k}:{re_c1_0[k]:.3e}" for k in patched_names)
    summary_fine = ",".join(f"{k}:{re_fine_10m[k]:.3e}" for k in patched_names)
    print(
        f"NUMBERS test_t1_cpml_patches_spectrum: "
        f"nop_10m={nop_10m:.3e} nop_0={nop_0:.3e} "
        f"min_fail_c1_10m={min_fail_c1_10m:.3e} max_c1_10m={max_c1_10m:.3e} "
        f"c1_max_freq_hz={c1_max_freq_hz:.3e} max_c1_0={max_c1_0:.3e} "
        f"min_fine_10m={min_fine_10m:.3e} max_fine_10m={max_fine_10m:.3e} "
        f"fail_c1_10m=[{summary_fail}] "
        f"c1_10m=[{summary_10m}] c1_0=[{summary_0}] fine_10m=[{summary_fine}]"
    )

    # No-patch control at 10 MHz must measure <= 1e-7 sigma_max
    assert nop_10m <= 1e-7
    # Mutation: present sampling ('fine') must fail T1 (> 1e-6 sigma_max)
    assert min_fine_10m > 1e-3
    # T1 criterion: max Re lambda(G) <= 1e-6 sigma_max across all patched cases
    assert max_c1_10m <= 1e-6
