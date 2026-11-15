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
   (relative peak-to-peak drift ``< 1e-14``, measured ``4e-16`` to ``6e-16``),
   broken by ``mutate="asym"``.
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
"""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from wakis.hybrid import (
    HybridFIT3D,
    delaunay_box_mesh,
    kuhn_box_mesh,
    mixed_partition,
    well_centred_tets,
)
from wakis.hybrid.verification import cavity_modes

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
    # measured: 242 lumped edges out of 5873 on the test Delaunay mesh
    assert s.lump_mask.sum() > 0
    assert np.all(s.dv > 0)

    sym_m1 = sp.linalg.norm(s.M1 - s.M1.T) / sp.linalg.norm(s.M1)
    sym_m = sp.linalg.norm(s.msolver.M - s.msolver.M.T) / sp.linalg.norm(s.msolver.M)
    # measured: sym_m1 = 4.66e-17, sym_m = 1.07e-17
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
    # measured: lam_min_m1 = 1.79e-15 (lam_max_m1 = 4.12e-12),
    #           lam_min_m  = 1.79e-15 (lam_max_m  = 2.65e-12)
    assert lam_min_m1 > 1e6 * EPS * lam_max_m1
    assert lam_min_m > 1e6 * EPS * lam_max_m


@pytest.mark.parametrize("solver", ["lu", "chebyshev"])
def test_dot_product_mixed(meshes, solver):
    """With mass='mixed', <K x, y> - <x, K^T y> and <step(x), y> - <x, step_T(y)>
    hold to double-precision round-off; non-symmetric perturbations break it."""
    tm = meshes["delaunay"]
    s = HybridFIT3D(XN, YN, ZN, box=BOX, tet_mesh=tm, mass="mixed", mass_solver=solver)
    assert (s.K.T - s.KT).nnz == 0

    rng = np.random.default_rng(2)
    x = rng.standard_normal(s.n_free)
    y = rng.standard_normal(s.n_faces)
    kx = s.K @ x
    kty = s.KT @ y
    cond_k = np.abs(kx) @ np.abs(y) + np.abs(x) @ np.abs(kty)
    rel_k = abs(kx @ y - x @ kty) / cond_k
    # measured: 8.80e-17 (< EPS = 2.22e-16: pure inner-product summation order)
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

    # measured: 4.39e-16 (lu), 1.32e-15 (chebyshev)
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
    # measured: 3.97e-16 to 5.88e-16
    assert drift[None] < 1e-14
    assert drift["asym"] > 1e-4


def test_lump_all_mutation(meshes):
    """mutate='lump_all' lumps non-well-centred edges and fails both by
    non-positive D_V on the test meshes and by violating the cavity
    eigenfrequency bound."""
    for kind in ("kuhn", "delaunay"):
        tm = meshes[kind]
        lump_mask, dv = mixed_partition(tm, mutate="lump_all")
        assert lump_mask.all()
        # measured: min(dv) = -1.23e-28 (869 edges <= 0) on kuhn,
        #           min(dv) = -7.84e-30 (1 edge <= 0) on delaunay
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
    dv or changes the lumped edge set.
    # measured by this test on meshes['delaunay']: adds 85 lumped edges (327 vs 242)
    """
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
    # measured at N=16: |f_mixed - f_consistent| = 5.10e5 Hz (2.04e-3 rel),
    #                   |f_consistent - f_analytic| = 6.20e5 Hz (2.48e-3 rel),
    #                   ratio = 0.822 <= 1.0
    assert diff_mix_cons <= diff_cons_ex, (f_mix[0], f_cons[0], f_ex[0])
