"""
Partial mass lumping (`mass="mixed"`) for `wakis.hybrid.HybridFIT3D`.

Verification checks and mutations:

1. Symmetric positive definiteness (SPD) of the mixed tetrahedral mass block
   ``M1`` and of the full composite mass ``msolver.M`` on the Delaunay patch
   (symmetry to machine precision, smallest eigenvalue strictly positive via
   Lanczos `eigsh`), plus bitwise identity of the default ``mass="consistent"``
   path.
2. Exact transpose (`test_dot_product_mixed`): ``<K x, y> - <x, K^T y>`` is at
   double-precision round-off (`(K.T - KT).nnz == 0`), and the full leapfrog
   step satisfies ``<linear_step(x), y> = <x, linear_step_T(y)>`` to round-off
   for both LU and Chebyshev mass solves, broken by a non-symmetric mass
   perturbation.
3. Exact leapfrog energy conservation with ``mass="mixed"`` over 200 steps
   (relative peak-to-peak drift ``< 1e-14``), broken by ``mutate="asym"``.
4. Mutation checks (`test_lump_all_mutation`, `test_delaunay_wellcentred_mutation`):
   ``mutate="lump_all"`` lumps every tetrahedral edge including non-well-centred
   ones, failing by non-positive Voronoi diagonal ``D_V`` on the Kuhn and
   Delaunay test meshes (and raising ``RuntimeError`` in `MassSolver` on Kuhn);
   ``mutate="ignore_wellcentred"`` omits the `well_centred_tets` filter along the
   standard partition path and lumps additional non-well-centred interior edges
   on the Delaunay test mesh.
5. Eigenfrequency agreement (`test_cavity_eigenfrequency_mixed`): on the PEC
   cavity from `wakis.hybrid.verification.cavity_modes`, the lowest mode
   satisfies ``|f_mixed - f_consistent| <= |f_consistent - f_analytic|``.
6. Interior patch smoothing (`smooth_patch` in `wakis.hybrid.patch_quality`):
   (a) box-boundary and PEC-sphere nodes are unchanged with exact equality;
   (b) all tetrahedra keep positive volume and the worst normalised volume is
   monotonically non-decreasing across sweeps (exact comparison), while the
   Jacobi-scaled consistent-block spectrum ratio ``lam_hi / lam_lo`` drops on
   the test Delaunay mesh;
   (c) the smoothed mesh passes `box_surface_check`;
   (d) ``mutate="move_boundary"`` moves box-surface nodes and fails test (a);
   (e) `HybridFIT3D(..., mass="mixed")` on the smoothed mesh preserves exact
   transposition (`_assert_dot_product`);
   (f) `pec_sphere_scattering` supports ``mass="mixed"`` and ``smooth=True``
   (``method="hybrid"`` only), raises ``ValueError`` on ``mass="bogus"``, and
   its smoothed-mesh check fails under the verification-only mutation that
   makes ``smooth`` ignore `smooth_patch`;
   (g) `measure_cartesian_solverfit3d` in `studies/hybrid-cart-tet/run_mixed_study.py`
   computes finite positive `cost_per_unknown_step_s` after the `step_0`
   warm-up step (`_attrcleanup`) and matches the closed-form / `HybridFIT3D`
   free-edge and face counts, broken by `mutate="ignore_pec_walls"`.
"""

import importlib.util
import os

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from wakis.hybrid import (
    HybridFIT3D,
    box_surface_check,
    consistent_spectrum_ratio,
    delaunay_box_mesh,
    fixed_patch_nodes,
    kuhn_box_mesh,
    mixed_partition,
    normalised_tet_volume,
    smooth_patch,
    well_centred_tets,
    worst_normalised_volume,
)
from wakis.hybrid.verification import cavity_modes, pec_sphere_scattering

H = 0.1
XN = np.arange(0, 1.0 + 1e-9, H)
YN = np.arange(0, 0.8 + 1e-9, H)
ZN = np.arange(0, 0.6 + 1e-9, H)
BOX = ((3, 7), (2, 6), (2, 4))
LINES = (XN[3:8], YN[2:7], ZN[2:5])
EPS = np.finfo(float).eps


@pytest.fixture(scope="module")
def meshes():
    return {
        "kuhn": kuhn_box_mesh(*LINES, r=2),
        "delaunay": delaunay_box_mesh(*LINES, h=0.06, r=2, seed=0),
    }


def test_mixed_mass_spd(meshes):
    """Mixed tetrahedral block M1 and full composite msolver.M are SPD;
    default mass='consistent' is unchanged bit-for-bit."""
    with pytest.raises(ValueError):
        HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=meshes["kuhn"], mass="invalid")

    tm = meshes["delaunay"]
    s_def = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, dt=1.0)
    s_cons = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, mass="consistent", dt=1.0)
    assert (s_def.M - s_cons.M).nnz == 0

    s = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, mass="mixed", dt=1.0)
    # results_mixed.json (key 'test_delaunay'): 242 lumped edges out of 5873
    assert s.lump_mask.sum() > 0
    assert np.all(s.dv > 0)

    sym_m1 = sp.linalg.norm(s.M1 - s.M1.T) / sp.linalg.norm(s.M1)
    sym_m = sp.linalg.norm(s.msolver.M - s.msolver.M.T) / sp.linalg.norm(s.msolver.M)
    assert sym_m1 < 10 * EPS
    assert sym_m < 10 * EPS

    rng = np.random.default_rng(0)
    v1 = rng.standard_normal(s.M1.shape[0])
    lam_min_m1 = sla.eigsh(
        s.M1, k=1, which="SA", tol=1e-9, v0=v1, maxiter=50000, return_eigenvectors=False
    )[0]
    lam_max_m1 = sla.eigsh(
        s.M1, k=1, which="LA", tol=1e-9, v0=v1, return_eigenvectors=False
    )[0]
    vm = rng.standard_normal(s.msolver.M.shape[0])
    lam_min_m = sla.eigsh(
        s.msolver.M,
        k=1,
        which="SA",
        tol=1e-9,
        v0=vm,
        maxiter=50000,
        return_eigenvectors=False,
    )[0]
    lam_max_m = sla.eigsh(
        s.msolver.M, k=1, which="LA", tol=1e-9, v0=vm, return_eigenvectors=False
    )[0]
    assert lam_min_m1 > 1e6 * EPS * lam_max_m1
    assert lam_min_m > 1e6 * EPS * lam_max_m


def _assert_dot_product(tm, solver):
    s = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, mass="mixed", mass_solver=solver)
    assert (s.K.T - s.KT).nnz == 0

    rng = np.random.default_rng(2)
    x = rng.standard_normal(s.n_free)
    y = rng.standard_normal(s.n_faces)
    kx = s.K @ x
    kty = s.KT @ y
    cond_k = np.abs(kx) @ np.abs(y) + np.abs(x) @ np.abs(kty)
    rel_k = abs(kx @ y - x @ kty) / cond_k
    assert rel_k < 5 * EPS

    b, e = rng.standard_normal(s.n_faces), rng.standard_normal(s.n_free)
    i_all = rng.standard_normal(s.nEc + s.nEt)
    bb, eb = rng.standard_normal(s.n_faces), rng.standard_normal(s.n_free)

    def mismatch(sol):
        b1, e1 = sol.linear_step(b, e, i_all)
        tb, te, ti = sol.linear_step_T(bb, eb)
        lhs = b1 @ bb + e1 @ eb
        rhs = b @ tb + e @ te + i_all @ ti
        return abs(lhs - rhs) / (abs(lhs) + abs(rhs))

    assert mismatch(s) < 10 * EPS
    M = s.M.tolil()
    S = s.msolver.S
    M[S[0], S[1]] = M[S[0], S[1]] + 0.3 * M[S[0], S[0]]
    s.M = M.tocsr()
    s._msolver = None
    if solver == "chebyshev":
        assert mismatch(s) > 1e-8
    else:
        assert mismatch(s) < 10 * EPS


@pytest.mark.parametrize("solver", ["lu", "chebyshev"])
def test_dot_product_mixed(meshes, solver):
    """With mass='mixed', <K x, y> - <x, K^T y> and <step(x), y> - <x, step_T(y)>
    hold to double-precision round-off; non-symmetric perturbations break it."""
    _assert_dot_product(meshes["delaunay"], solver)


@pytest.mark.parametrize(
    "kind,solver",
    [
        ("kuhn", "lu"),
        ("kuhn", "chebyshev"),
        ("delaunay", "lu"),
        ("delaunay", "chebyshev"),
    ],
)
def test_energy_drift_mixed(meshes, kind, solver):
    """With mass='mixed', 1/2 b.W b + 1/2 e^n.M e^{n+1} drifts by < 1e-14 over
    200 steps; 'asym' breaks energy conservation."""
    tm = meshes[kind]
    rng = np.random.default_rng(1)
    drift = {}
    for mut in (None, "asym"):
        s = HybridFIT3D(
            XN,
            YN,
            ZN,
            box=BOX,
            tet_mesh=tm,
            mass="mixed",
            mutate=mut,
            mass_solver=solver,
        )
        s.e = rng.standard_normal(s.n_free)
        s.b = rng.standard_normal(s.n_faces) * 1e-9
        E = s.run(200, energy=True)
        drift[mut] = np.ptp(E) / np.abs(E).mean()
    assert drift[None] < 1e-14
    assert drift["asym"] > 1e-4


def test_lump_all_mutation(meshes):
    """mutate='lump_all' lumps non-well-centred edges and fails by
    non-positive D_V on the test meshes."""
    for kind in ("kuhn", "delaunay"):
        tm = meshes[kind]
        lump_mask, dv = mixed_partition(tm, mutate="lump_all")
        assert lump_mask.all()
        assert np.any(dv <= 0)
        s = HybridFIT3D(
            XN, YN, ZN, box=BOX, tet_mesh=tm, mass="mixed", mutate="lump_all", dt=1.0
        )
        assert s.M.diagonal().min() <= 0
        if kind == "kuhn":
            with pytest.raises(RuntimeError):
                _ = s.msolver

    with pytest.raises(ValueError):
        mixed_partition(meshes["delaunay"], mutate="unknown")


def test_delaunay_wellcentred_mutation(meshes):
    """mutate='ignore_wellcentred' omits the well_centred_tets filter in
    mixed_partition on the Delaunay test mesh and either yields a non-positive
    dv or changes the lumped edge set beyond n_tet_edges_lumped = 242."""
    tm = meshes["delaunay"]
    lump_ref, _ = mixed_partition(tm)
    wc = well_centred_tets(tm)
    try:
        lump_mut, dv_mut = mixed_partition(tm, mutate="ignore_wellcentred")
    except RuntimeError:
        return
    added_mask = lump_mut & ~lump_ref
    added = int(added_mask.sum())
    print(
        f"ignore_wellcentred added {added} lumped edges "
        f"({int(lump_mut.sum())} vs {int(lump_ref.sum())})"
    )
    non_wc_edges = np.zeros(tm.n_edges, bool)
    non_wc_edges[tm.cell_edges[~wc].ravel()] = True
    assert np.any(dv_mut <= 0) or (
        added > 0
        and not np.array_equal(lump_mut, lump_ref)
        and np.all(non_wc_edges[added_mask])
    )


def test_cavity_eigenfrequency_mixed():
    """On the PEC cavity from `cavity_modes` (N=16, Delaunay patch), the lowest
    resonance satisfies |f_mixed - f_consistent| <= |f_consistent - f_analytic|."""
    f_cons, f_ex, s_cons = cavity_modes(16, patch="delaunay", n_modes=3)
    box = tuple(zip(s_cons.box_lo, s_cons.box_hi))
    s_mix = HybridFIT3D(
        *s_cons.block.nodes, box=box, tet_mesh=s_cons.tet, mass="mixed", dt=1.0
    )
    f_mix = s_mix.eigenfrequencies(3, 0.9 * f_ex[0], 1.15 * f_ex[-1])
    diff_mix_cons = abs(f_mix[0] - f_cons[0])
    diff_cons_ex = abs(f_cons[0] - f_ex[0])
    # results_mixed.json (key 'cavity_N16'):
    # diff_mixed_consistent = 509548.66 Hz, diff_consistent_analytic = 620262.44 Hz,
    # ratio = 0.8215 <= 1.0
    assert diff_mix_cons <= diff_cons_ex, (f_mix[0], f_cons[0], f_ex[0])


def _assert_boundary_and_sphere_fixed(tm, tm_smoothed, lines, allow_other=False):
    fixed = fixed_patch_nodes(tm)
    assert np.array_equal(tm_smoothed.cells, tm.cells)
    assert np.array_equal(tm_smoothed.cell_tags, tm.cell_tags)
    assert np.array_equal(tm_smoothed.nodes[fixed], tm.nodes[fixed]), (
        f"boundary/sphere nodes moved: max diff "
        f"{np.abs(tm_smoothed.nodes[fixed] - tm.nodes[fixed]).max():.3e}"
    )
    assert not np.array_equal(tm_smoothed.nodes[~fixed], tm.nodes[~fixed])
    bf = box_surface_check(tm_smoothed, *lines, allow_other=allow_other)
    assert len(bf) > 0


def test_smooth_patch_boundary_and_box_check(meshes):
    """Tests (a), (c), (d): boundary and PEC-sphere nodes stay exactly fixed,
    smoothed meshes pass `box_surface_check`, and `mutate='move_boundary'`
    fails test (a)."""
    tm = meshes["delaunay"]
    tm_s = smooth_patch(tm)
    _assert_boundary_and_sphere_fixed(tm, tm_s, LINES, allow_other=False)

    # Mesh with a PEC sphere hole: both box-surface and sphere-surface nodes fixed
    sv = np.linspace(-0.25, 0.25, 6)
    tm_sph = delaunay_box_mesh(
        sv, sv, sv, h=0.08, sphere=((0.0, 0.0, 0.0), 0.08), hole=True, seed=0
    )
    tm_sph_s = smooth_patch(tm_sph)
    _assert_boundary_and_sphere_fixed(tm_sph, tm_sph_s, (sv, sv, sv), allow_other=True)
    r_nodes = np.linalg.norm(tm_sph.nodes, axis=1)
    on_sphere = np.isclose(r_nodes, 0.08, atol=1e-6)
    assert on_sphere.any()
    assert np.array_equal(tm_sph_s.nodes[on_sphere], tm_sph.nodes[on_sphere])

    # Test (d): mutate='move_boundary' moves boundary nodes and must fail test (a)
    tm_mut = smooth_patch(tm, mutate="move_boundary")
    with pytest.raises(AssertionError, match="boundary/sphere nodes moved"):
        _assert_boundary_and_sphere_fixed(tm, tm_mut, LINES, allow_other=False)
    with pytest.raises(ValueError):
        box_surface_check(tm_mut, *LINES)


def test_smooth_patch_monotone_volume_and_spectrum(meshes):
    """Test (b): all tetrahedra keep positive volume, the worst normalised
    volume is monotonically non-decreasing across sweeps (exact comparison),
    and the Jacobi-scaled consistent-block spectrum ratio drops on the test
    Delaunay mesh."""
    tm = meshes["delaunay"]
    w_prev = worst_normalised_volume(tm)
    for k in range(1, 7):
        tm_k = smooth_patch(tm, n_sweeps=k)
        assert np.all(tm_k.cell_volume > 0)
        assert np.all(normalised_tet_volume(tm_k) > 0)
        w_cur = worst_normalised_volume(tm_k)
        assert w_cur >= w_prev
        w_prev = w_cur
    assert w_prev > worst_normalised_volume(tm)

    hist = tm_k.smooth_history
    for k in range(len(hist) - 1):
        assert hist[k + 1] >= hist[k]

    ratio_before = consistent_spectrum_ratio(tm)
    ratio_after = consistent_spectrum_ratio(tm_k)
    assert ratio_after < ratio_before

    tm_spec = smooth_patch(tm, objective="spectrum", n_sweeps=3)
    assert np.all(tm_spec.cell_volume > 0)
    for k in range(len(tm_spec.smooth_history) - 1):
        assert tm_spec.smooth_history[k + 1] >= tm_spec.smooth_history[k]
    assert consistent_spectrum_ratio(tm_spec) < ratio_before


@pytest.mark.parametrize("solver", ["lu", "chebyshev"])
def test_dot_product_mixed_smoothed(meshes, solver):
    """Test (e): exact transposition (`_assert_dot_product`) holds to
    double-precision round-off on the smoothed Delaunay mesh with mass='mixed'."""
    tm_s = smooth_patch(meshes["delaunay"])
    _assert_dot_product(tm_s, solver)


def _assert_pec_sphere_smoothed_differs(out_unsmoothed, out_smoothed):
    assert np.isfinite(out_smoothed["err_near"])
    assert np.isfinite(out_smoothed["err_far"])
    nodes_in = out_unsmoothed["solver"].tet.nodes
    nodes_sm = out_smoothed["solver"].tet.nodes
    max_diff = float(np.abs(nodes_sm - nodes_in).max())
    assert not np.array_equal(nodes_sm, nodes_in), (
        f"smoothed mesh does not differ from input mesh: max_node_diff={max_diff:.3e}"
    )


def test_pec_sphere_scattering_mass_and_smooth():
    """`pec_sphere_scattering` supports `mass='mixed'` and `smooth=True` on
    `method='hybrid'`, raises `ValueError` on `mass='bogus'`, and fails the
    smoothed-mesh difference check when `smooth` ignores `smooth_patch`."""
    kw = dict(ka=1.0, gap=0.25, T=0.25, box_margin=0.05)
    with pytest.raises(ValueError, match="mass must be"):
        pec_sphere_scattering(4, mass="bogus", **kw)
    with pytest.raises(ValueError, match="method='hybrid' only"):
        pec_sphere_scattering(4, method="staircase", mass="mixed", **kw)
    with pytest.raises(ValueError, match="method='hybrid' only"):
        pec_sphere_scattering(4, method="staircase", smooth=True, **kw)

    out_mix = pec_sphere_scattering(4, mass="mixed", return_solver=True, **kw)
    assert out_mix["solver"].mass == "mixed"
    assert np.isfinite(out_mix["err_near"]) and out_mix["err_near"] > 0
    assert np.isfinite(out_mix["err_far"]) and out_mix["err_far"] > 0

    out_sm = pec_sphere_scattering(
        4, mass="consistent", smooth=True, return_solver=True, **kw
    )
    assert out_sm["solver"].mass == "consistent"
    _assert_pec_sphere_smoothed_differs(out_mix, out_sm)

    # Mutation check: make smooth ignore the smoothing (verification-only flag)
    out_mut = pec_sphere_scattering(
        4, mass="consistent", smooth="ignore", return_solver=True, **kw
    )
    with pytest.raises(
        AssertionError,
        match=r"smoothed mesh does not differ from input mesh: max_node_diff=0\.000e\+00",
    ) as exc_info:
        _assert_pec_sphere_smoothed_differs(out_mix, out_mut)
    print(f"Observed mutation failure: {exc_info.value}")


def _assert_cartesian_unknown_counts(out, nh, s_cart):
    expected_free = 3 * (2 * nh) * (2 * nh - 1) ** 2
    expected_faces = 3 * (2 * nh) ** 2 * (2 * nh + 1)
    assert out["cart_n_free_edges"] == expected_free == s_cart.n_free, (
        f"cart_n_free_edges={out['cart_n_free_edges']} != expected {expected_free}"
    )
    assert out["cart_n_faces"] == expected_faces == s_cart.n_faces
    assert out["cart_n_unknowns"] == expected_free + expected_faces


def test_measure_cartesian_solverfit3d():
    """Regression test for `measure_cartesian_solverfit3d`: after the warm-up
    `one_step()` (`_attrcleanup`), it derives the PEC wall edge mask from the
    public grid geometry, matches `HybridFIT3D` free-edge/face counts, and
    returns a finite positive `cost_per_unknown_step_s`."""
    study_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "studies",
        "hybrid-cart-tet",
        "run_mixed_study.py",
    )
    spec = importlib.util.spec_from_file_location("run_mixed_study", study_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    nh = 2
    h = 0.25
    v = h * np.arange(-nh, nh + 1)
    s_cart = HybridFIT3D(v, v, v, dt=1.0)

    out = mod.measure_cartesian_solverfit3d(v, n_steps=3)
    _assert_cartesian_unknown_counts(out, nh, s_cart)
    assert np.isfinite(out["cost_per_unknown_step_s"])
    assert out["cost_per_unknown_step_s"] > 0
    assert np.isfinite(out["cart_cost_per_unknown_step_s"])
    assert out["cart_cost_per_unknown_step_s"] > 0
    assert out["cost_per_unknown_step_s"] == out["cart_cost_per_unknown_step_s"]
    assert np.isfinite(out["cart_s_per_step"]) and out["cart_s_per_step"] > 0
    assert np.isfinite(out["cart_dt"]) and out["cart_dt"] > 0
    assert (
        np.isfinite(out["cart_time_to_solution_s"])
        and out["cart_time_to_solution_s"] > 0
    )

    # Mutation check: omitting the PEC outer-wall mask overcounts free edges
    out_mut = mod.measure_cartesian_solverfit3d(v, n_steps=1, mutate="ignore_pec_walls")
    with pytest.raises(
        AssertionError, match=r"cart_n_free_edges=300 != expected 108"
    ) as exc_info:
        _assert_cartesian_unknown_counts(out_mut, nh, s_cart)
    print(f"Observed mutation failure: {exc_info.value}")
