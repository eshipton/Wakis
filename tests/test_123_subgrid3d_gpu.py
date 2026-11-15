# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Verification of fused GPU kernels for 3D Cartesian subgridding
(`wakis.subgrid3d.gpu_fused`: `FusedTables`, `FusedSubgridStepper`).

Checks G1-G6 (Brief W4-3c, docs/physicsguide_sections/subgrid3d_gpu.md):

* G1 (CPU): NumPy emulation of the edge-source table (`FusedTables.apply_K`)
  and its transpose (`FusedTables.apply_KH`) equals `BlockOps.K` and
  `BlockOps.KH` bit for bit (`np.array_equal`, 0 ULP) on `closed_mesh`
  (2:1 + nested 3:1), `_open_box` with CPML, and a 3:1 patch on a
  non-power-of-two `13 x 11 x 9` coarse grid; exact transpose dot-product
  test `<K e, y> == <e, K^H y>` to round-off (~1e-16).
  Mutations: `mutate="no_hanging"` -> relative 2-norm > 1e-3;
  `mutate="face_offset"` -> O(1).
* G2 (GPU): device `sg_faraday` (`apply_K`) and `sg_ampere` (`apply_KH`),
  one application, bitwise equal (`np.array_equal`, 0 ULP) to NumPy
  `BlockOps.K` and `BlockOps.KH`, with the same mutations on the device.
* G3 (GPU): 500 steps of `closed_mesh` global dt, `closed_mesh` LTS
  (2:1 + nested 3:1), `_open_box` with CPML global dt, and `_open_box` with
  CPML and LTS, plus the `13 x 11 x 9` 3:1 LTS mesh, each with one source
  and two probes with DFT: state (`b`, `e`, CPML `psi`, `phi`), probe time
  series and running DFTs are bitwise equal (`np.array_equal`, 0 ULP) to the
  NumPy `kernels="block"` run; full step transpose `step_vec_T` passes the
  dot-product test (< 1e-15).
  Mutation: `mutate="lts_kappa"` (Chebyshev buffer term `kappa_k g` dropped)
  -> error far above `_empirical_roundoff_tol`.
* G4 (GPU): CUDA-graph replay (`GraphRunner`) bitwise equals direct kernel
  launches (`graph_chunk=0`), and CuPy default memory pool `used_bytes()` is
  strictly constant (0 bytes allocated) over 100 steps.
* G5 (GPU): device `modified_energy` over 2000 LTS steps on `closed_mesh` at
  `0.95 dt_max^LTS` drifts by <= `50 eps sqrt(n)`.
* G6 (CPU+GPU): contract refusals (`dispersive_groups`, mismatched `n_state`,
  Bloch-periodic blocks, complex fields) raise `NotImplementedError`.
"""

import os
import types

import numpy as np
import pytest

from wakis.subgrid3d import Box, SubgridLTSFIT3D, SubgridMesh3D
from wakis.subgrid3d.gpu_fused import FusedSubgridStepper, FusedTables

EPS = np.finfo(float).eps


def _require_gpu():
    """Return `cupy` if a CUDA device is available; skip or fail per `WAKIS_REQUIRE_GPU`."""
    req = os.environ.get("WAKIS_REQUIRE_GPU") == "1"
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("getDeviceCount() == 0")
        return cp
    except Exception as exc:  # noqa: BLE001
        if req:
            pytest.fail(f"WAKIS_REQUIRE_GPU=1 but no CUDA device is available: {exc}")
        pytest.skip(f"no CUDA device: {exc}")


def closed_mesh():
    """Graded coarse grid, 2:1 patch with a nested 3:1 patch (copied from test_104)."""
    x = np.array([0, 0.1, 0.2, 0.3, 0.42, 0.55, 0.65, 0.75, 0.85, 1.0])
    m = SubgridMesh3D(x, np.linspace(0, 0.8, 9), np.linspace(0, 0.6, 7))
    p1 = m.add_patch((2, 2, 1), (7, 6, 5), ratio=2)
    m.add_patch((3, 2, 2), (7, 6, 6), ratio=3, parent=p1)
    return m


def closed_materials():
    """Inhomogeneous eps_r, mu_r, PEC and PMC boxes (copied from test_104)."""

    def eps(X, Y, Z):
        return np.where((X > 0.3) & (X < 0.55) & (Y > 0.1) & (Y < 0.5), 4.0, 1.0)

    def mu(X, Y, Z):
        return np.where((Z > 0.3) & (Y > 0.4), 2.0, 1.0)

    return {
        "eps_r": eps,
        "mu_r": mu,
        "pec": Box(((0.42, 0.55), (0.35, 0.45), (0.25, 0.35))),
        "pmc": Box(((0.75, 0.85), (0.1, 0.2), (0.0, 0.1))),
    }


def _open_box(
    patch=True, lts=True, kernels="csr", pml=True, R0=1e-6, alpha_hz=0.0, **kw
):
    """24^3 open box with a 2:1 patch and 6-cell CPML on all sides (copied from test_104)."""
    H, n = 0.04, 24
    x = np.arange(n + 1) * H
    m = SubgridMesh3D(x, x, x)
    if patch:
        m.add_box(((0.4, 0.68), (0.4, 0.6), (0.4, 0.6)), ratio=2)
    th = [(6 * H, 6 * H)] * 3
    return SubgridLTSFIT3D(
        m,
        lts=lts,
        kernels=kernels,
        dt_rtol=1e-2,
        pml={"thickness": th, "R0": R0, "alpha_hz": alpha_hz} if pml else None,
        **kw,
    )


def nonpow2_3to1_mesh():
    """Single 3:1 patch on a non-power-of-two 13 x 11 x 9 coarse grid."""
    x = np.linspace(0.0, 0.65, 14)
    y = np.linspace(0.0, 0.55, 12)
    z = np.linspace(0.0, 0.45, 10)
    m = SubgridMesh3D(x, y, z)
    m.add_patch((4, 3, 2), (9, 8, 7), ratio=3)
    return m


def _empirical_roundoff_tol(s, n_steps):
    """Empirical tolerance ``n u (m_K + m_KH) kappa_E`` on the relative 2-norm
    difference between block and CSR steps (copied from test_104)."""
    mK = np.diff(s.K.indptr).max()
    mKH = np.diff(s.KH.tocsr().indptr).max()
    rho = min(s.stability_ratio(), 0.999)
    return n_steps * EPS / 2 * (mK + mKH) * np.sqrt((1 + rho) / (1 - rho))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _max_ulp(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.array_equal(a, b):
        return 0
    ia = a.view(np.int64)
    ib = b.view(np.int64)
    ia = np.where(ia < 0, np.int64(-0x8000000000000000) - ia, ia)
    ib = np.where(ib < 0, np.int64(-0x8000000000000000) - ib, ib)
    return int(np.max(np.abs(ia - ib)))


# ----------------------------------------------------------------------- G1
def test_g1_cpu_tables():
    """G1 (CPU): NumPy emulation of the edge-source table (`apply_K`) and its
    transpose (`apply_KH`) equals `BlockOps.K` and `BlockOps.KH` bit for bit
    (`np.array_equal`, 0 ULP); mutations `no_hanging` and `face_offset` fail."""
    cases = [
        (
            "closed",
            SubgridLTSFIT3D(
                closed_mesh(), lts=False, cfl=0.8, kernels="block", **closed_materials()
            ),
        ),
        ("open_cpml", _open_box(lts=False, cfl=0.8, kernels="block")),
        (
            "nonpow2_3to1",
            SubgridLTSFIT3D(nonpow2_3to1_mesh(), lts=False, cfl=0.8, kernels="block"),
        ),
    ]
    rng = np.random.default_rng(101)
    max_ulp_k = 0
    max_ulp_kh = 0
    max_adj_def = 0.0
    mut_no_hang_k = {}
    mut_no_hang_kh = {}
    mut_face_off_k = {}
    mut_face_off_kh = {}

    for name, s in cases:
        ft = FusedTables(s)
        e = rng.standard_normal(s.n_free)
        y = rng.standard_normal(s.mesh.n_faces)

        k_ref = s.ops.K(e, np.empty(s.mesh.n_faces))
        k_cpu = ft.apply_K(e)
        assert np.array_equal(k_cpu, k_ref), f"G1 K not bitwise on {name}"
        max_ulp_k = max(max_ulp_k, _max_ulp(k_cpu, k_ref))

        kh_ref = s.ops.KH(y)
        kh_cpu = ft.apply_KH(y)
        assert np.array_equal(kh_cpu, kh_ref), f"G1 KH not bitwise on {name}"
        max_ulp_kh = max(max_ulp_kh, _max_ulp(kh_cpu, kh_ref))

        # Exact transpose dot-product check: <K e, y> == <e, KH y>
        lhs = float(np.dot(k_cpu, y))
        rhs = float(np.dot(e, kh_cpu))
        adj_def = abs(lhs - rhs) / (np.linalg.norm(k_cpu) * np.linalg.norm(y))
        assert adj_def < 1e-15, (name, adj_def)
        max_adj_def = max(max_adj_def, adj_def)

        # Mutation 1: no_hanging -> relative 2-norm > 1e-3
        ft_nh = FusedTables(s, mutate="no_hanging")
        mut_no_hang_k[name] = _rel(ft_nh.apply_K(e), k_ref)
        mut_no_hang_kh[name] = _rel(ft_nh.apply_KH(y), kh_ref)
        assert mut_no_hang_k[name] > 1e-3, (name, mut_no_hang_k[name])
        assert mut_no_hang_kh[name] > 1e-3, (name, mut_no_hang_kh[name])

        # Mutation 2: face_offset -> O(1)
        ft_fo = FusedTables(s, mutate="face_offset")
        mut_face_off_k[name] = _rel(ft_fo.apply_K(e), k_ref)
        mut_face_off_kh[name] = _rel(ft_fo.apply_KH(y), kh_ref)
        assert mut_face_off_k[name] > 0.5, (name, mut_face_off_k[name])
        assert mut_face_off_kh[name] > 0.5, (name, mut_face_off_kh[name])

    print(
        f"NUMBERS test_g1_cpu_tables: K_ulp={max_ulp_k} KH_ulp={max_ulp_kh} "
        f"dot_defect={max_adj_def:.2e} "
        f"mut_no_hanging_K_closed={mut_no_hang_k['closed']:.3e} "
        f"mut_no_hanging_KH_closed={mut_no_hang_kh['closed']:.3e} "
        f"mut_no_hanging_K_open={mut_no_hang_k['open_cpml']:.3e} "
        f"mut_no_hanging_K_3to1={mut_no_hang_k['nonpow2_3to1']:.3e} "
        f"mut_face_offset_K_closed={mut_face_off_k['closed']:.3e} "
        f"mut_face_offset_KH_closed={mut_face_off_kh['closed']:.3e}"
    )


# ----------------------------------------------------------------------- G2
def test_g2_gpu_operators():
    """G2 (GPU): fused device `apply_K` and `apply_KH` match NumPy `BlockOps`
    bitwise (`np.array_equal`, 0 ULP); device mutations `no_hanging` and
    `face_offset` produce the same measured errors as G1."""
    _require_gpu()
    cases = [
        (
            "closed",
            SubgridLTSFIT3D(
                closed_mesh(), lts=False, cfl=0.8, kernels="block", **closed_materials()
            ),
        ),
        ("open_cpml", _open_box(lts=False, cfl=0.8, kernels="block")),
        (
            "nonpow2_3to1",
            SubgridLTSFIT3D(nonpow2_3to1_mesh(), lts=False, cfl=0.8, kernels="block"),
        ),
    ]
    rng = np.random.default_rng(101)
    max_ulp_k = 0
    max_ulp_kh = 0
    max_adj_def = 0.0
    mut_nh_k = {}
    mut_nh_kh = {}
    mut_fo_k = {}
    mut_fo_kh = {}

    for name, s in cases:
        fs = FusedSubgridStepper(s, graph_chunk=0)
        e = rng.standard_normal(s.n_free)
        y = rng.standard_normal(s.mesh.n_faces)

        k_ref = s.ops.K(e, np.empty(s.mesh.n_faces))
        k_gpu = fs.apply_K(e).get()
        assert np.array_equal(k_gpu, k_ref), f"G2 GPU K not bitwise on {name}"
        max_ulp_k = max(max_ulp_k, _max_ulp(k_gpu, k_ref))
        assert _rel(k_gpu, k_ref) <= 4 * EPS

        kh_ref = s.ops.KH(y)
        kh_gpu = fs.apply_KH(y).get()
        assert np.array_equal(kh_gpu, kh_ref), f"G2 GPU KH not bitwise on {name}"
        max_ulp_kh = max(max_ulp_kh, _max_ulp(kh_gpu, kh_ref))
        assert _rel(kh_gpu, kh_ref) <= 4 * EPS

        # Exact transpose dot-product check on GPU outputs
        adj_def = abs(float(np.dot(k_gpu, y)) - float(np.dot(e, kh_gpu))) / (
            np.linalg.norm(k_gpu) * np.linalg.norm(y)
        )
        assert adj_def < 1e-15, (name, adj_def)
        max_adj_def = max(max_adj_def, adj_def)

        # Device mutations
        fs_nh = FusedSubgridStepper(s, graph_chunk=0, mutate="no_hanging")
        mut_nh_k[name] = _rel(fs_nh.apply_K(e).get(), k_ref)
        mut_nh_kh[name] = _rel(fs_nh.apply_KH(y).get(), kh_ref)
        assert mut_nh_k[name] > 1e-3
        assert mut_nh_kh[name] > 1e-3

        fs_fo = FusedSubgridStepper(s, graph_chunk=0, mutate="face_offset")
        mut_fo_k[name] = _rel(fs_fo.apply_K(e).get(), k_ref)
        mut_fo_kh[name] = _rel(fs_fo.apply_KH(y).get(), kh_ref)
        assert mut_fo_k[name] > 0.5
        assert mut_fo_kh[name] > 0.5

    print(
        f"NUMBERS test_g2_gpu_operators: K_ulp={max_ulp_k} KH_ulp={max_ulp_kh} "
        f"dot_defect={max_adj_def:.2e} "
        f"mut_no_hanging_K_closed={mut_nh_k['closed']:.3e} "
        f"mut_no_hanging_KH_closed={mut_nh_kh['closed']:.3e} "
        f"mut_face_offset_K_closed={mut_fo_k['closed']:.3e} "
        f"mut_face_offset_KH_closed={mut_fo_kh['closed']:.3e}"
    )


# ----------------------------------------------------------------------- G3
def test_g3_gpu_500_steps():
    """G3 (GPU): 500 steps of closed_mesh (global dt and LTS 2:1+3:1),
    _open_box with CPML (global dt and LTS), and nonpow2 13x11x9 3:1 LTS,
    each with one source and two probes with DFT: state, probe series and DFT
    match the NumPy `kernels='block'` run bitwise (0 ULP).
    Mutation `mutate='lts_kappa'` breaks agreement far above the empirical
    round-off tolerance."""
    _require_gpu()
    freqs = np.array([2.5e8, 4.0e8, 6.0e8])
    wf = lambda t: (
        np.exp(-(((t - 2.0e-9) / 0.5e-9) ** 2)) * np.sin(2 * np.pi * 4.0e8 * t)
    )

    builders = [
        (
            "closed_global",
            lambda: SubgridLTSFIT3D(
                closed_mesh(), lts=False, cfl=0.8, kernels="block", **closed_materials()
            ),
            ((0.35, 0.25, 0.25), 2),
            [((0.6, 0.4, 0.3), 2), ((0.25, 0.5, 0.35), 1)],
        ),
        (
            "closed_lts",
            lambda: SubgridLTSFIT3D(
                closed_mesh(),
                lts=True,
                cfl=0.95,
                dt_rtol=1e-2,
                kernels="block",
                **closed_materials(),
            ),
            ((0.35, 0.25, 0.25), 2),
            [((0.6, 0.4, 0.3), 2), ((0.25, 0.5, 0.35), 1)],
        ),
        (
            "open_cpml_global",
            lambda: _open_box(lts=False, cfl=0.8, kernels="block"),
            ((0.48, 0.5, 0.5), 2),
            [((0.56, 0.5, 0.5), 2), ((0.44, 0.48, 0.52), 1)],
        ),
        (
            "open_cpml_lts",
            lambda: _open_box(lts=True, dt=7.0e-11, kernels="block"),
            ((0.48, 0.5, 0.5), 2),
            [((0.56, 0.5, 0.5), 2), ((0.44, 0.48, 0.52), 1)],
        ),
        (
            "nonpow2_3to1_lts",
            lambda: SubgridLTSFIT3D(
                nonpow2_3to1_mesh(), lts=True, cfl=0.9, dt_rtol=1e-2, kernels="block"
            ),
            ((0.30, 0.25, 0.20), 2),
            [((0.35, 0.30, 0.25), 2), ((0.20, 0.20, 0.20), 1)],
        ),
    ]

    n_steps = 500
    max_state_ulp = 0
    max_probe_ulp = 0
    max_dft_ulp = 0
    tols = {}
    k_counts = {}

    for name, make_solver, src_spec, prb_specs in builders:
        s_ref = make_solver()
        s_gpu = make_solver()
        for s in (s_ref, s_gpu):
            s.add_current(src_spec[0], src_spec[1], wf)
            for pt, comp in prb_specs:
                s.add_probe(pt, comp, freqs=freqs)
        fs = FusedSubgridStepper(s_gpu, graph_chunk=64)
        k_counts[name] = fs.kernels_per_step()

        s_ref.run(n_steps)
        fs.run(n_steps)
        fs.synchronize()

        ref_state = np.concatenate(
            [s_ref.b, s_ref.e]
            + (s_ref.psi if s_ref.cpml is not None else [])
            + (s_ref.phi if s_ref.cpml is not None else [])
        )
        gpu_state = fs.state_host()
        assert np.array_equal(gpu_state, ref_state), f"G3 state not bitwise on {name}"
        assert np.array_equal(fs.e_host(), s_ref.e_host()), (
            f"G3 e not bitwise on {name}"
        )
        assert np.array_equal(fs.b_faces(), s_ref.b_faces()), (
            f"G3 b_faces not bitwise on {name}"
        )
        max_state_ulp = max(max_state_ulp, _max_ulp(gpu_state, ref_state))

        for pi in range(2):
            p_ref = np.asarray(s_ref.probes[pi]["values"], dtype=np.float64)
            p_gpu = fs.probe_signal(pi)
            assert np.array_equal(p_gpu, p_ref), f"G3 probe {pi} not bitwise on {name}"
            max_probe_ulp = max(max_probe_ulp, _max_ulp(p_gpu, p_ref))

            d_ref = np.asarray(s_ref.probes[pi]["dft"], dtype=np.complex128)
            d_gpu = fs.dft_data(pi)
            assert np.array_equal(d_gpu, d_ref), f"G3 DFT {pi} not bitwise on {name}"
            max_dft_ulp = max(
                max_dft_ulp,
                _max_ulp(d_gpu.real, d_ref.real),
                _max_ulp(d_gpu.imag, d_ref.imag),
            )

        tols[name] = _empirical_roundoff_tol(s_ref, n_steps)

    # Full step transpose dot-product check on closed_lts and open_cpml_lts
    dot_defects = {}
    for dname, dmake in [
        (
            "closed_lts",
            lambda: SubgridLTSFIT3D(
                closed_mesh(),
                lts=True,
                dt=1.0e-10,
                kernels="block",
                **closed_materials(),
            ),
        ),
        ("open_cpml_lts", lambda: _open_box(lts=True, dt=7.0e-11, kernels="block")),
    ]:
        sd = dmake()
        fsd = FusedSubgridStepper(sd, graph_chunk=0)
        rng = np.random.default_rng(2)
        x = rng.standard_normal(sd.n_state)
        y = rng.standard_normal(sd.n_state)
        i_all = rng.standard_normal(sd.mesh.n_edges)
        W = np.asarray(sd._Wb)
        sb = np.sqrt(np.median(sd.D) / np.median(W[W > 0]))
        x[: sd._nb] *= sb
        y[: sd._nb] *= sb
        fx = fsd.step_vec(x, i_all)
        assert np.array_equal(fx, sd.step_vec(x, i_all))
        xb, ib = fsd.step_vec_T(y)
        xb_ref, ib_ref = sd.step_vec_T(y)
        assert np.array_equal(xb, xb_ref) and np.array_equal(ib, ib_ref)
        scale = np.linalg.norm(fx) * np.linalg.norm(y)
        dot_defects[dname] = float(abs(fx @ y - (x @ xb + i_all @ ib)) / scale)
        assert dot_defects[dname] < 1e-15

    # Mutation: mutate="lts_kappa" (Chebyshev buffer term kappa_k g dropped)
    s_ref_lts = SubgridLTSFIT3D(
        closed_mesh(),
        lts=True,
        cfl=0.95,
        dt_rtol=1e-2,
        kernels="block",
        **closed_materials(),
    )
    s_mut_lts = SubgridLTSFIT3D(
        closed_mesh(), lts=True, dt=s_ref_lts.dt, kernels="block", **closed_materials()
    )
    for s in (s_ref_lts, s_mut_lts):
        s.add_current((0.35, 0.25, 0.25), 2, wf)
    fs_mut = FusedSubgridStepper(s_mut_lts, graph_chunk=64, mutate="lts_kappa")
    s_ref_lts.run(n_steps)
    fs_mut.run(n_steps)
    fs_mut.synchronize()
    err_kappa = _rel(fs_mut.e_host(), s_ref_lts.e_host())
    tol_lts = tols["closed_lts"]
    assert err_kappa > 100 * tol_lts and err_kappa > 1e-3, (err_kappa, tol_lts)

    print(
        f"NUMBERS test_g3_gpu_500_steps: state_ulp={max_state_ulp} "
        f"probe_ulp={max_probe_ulp} dft_ulp={max_dft_ulp} "
        f"tol_closed_global={tols['closed_global']:.3e} "
        f"tol_closed_lts={tol_lts:.3e} "
        f"tol_open_cpml_global={tols['open_cpml_global']:.3e} "
        f"tol_open_cpml_lts={tols['open_cpml_lts']:.3e} "
        f"dot_closed_lts={dot_defects['closed_lts']:.2e} "
        f"dot_open_cpml_lts={dot_defects['open_cpml_lts']:.2e} "
        f"kernels_global={k_counts['closed_global']} "
        f"kernels_2to1={k_counts['open_cpml_lts']} "
        f"kernels_3to1={k_counts['nonpow2_3to1_lts']} "
        f"kernels_2x3={k_counts['closed_lts']} "
        f"mut_lts_kappa_rel={err_kappa:.3e}"
    )


# ----------------------------------------------------------------------- G4
def test_g4_graph_and_zero_alloc():
    """G4 (GPU): CUDA-graph replay == direct kernel launches bit for bit, and
    CuPy memory pool `used_bytes()` stays strictly constant over 100 steps."""
    cp = _require_gpu()
    wf = lambda t: (
        np.exp(-(((t - 2.0e-9) / 0.5e-9) ** 2)) * np.sin(2 * np.pi * 4.0e8 * t)
    )
    freqs = np.array([2.5e8, 4.0e8, 6.0e8])

    s_graph = SubgridLTSFIT3D(
        closed_mesh(),
        lts=True,
        cfl=0.95,
        dt_rtol=1e-2,
        kernels="block",
        **closed_materials(),
    )
    s_direct = SubgridLTSFIT3D(
        closed_mesh(), lts=True, dt=s_graph.dt, kernels="block", **closed_materials()
    )
    for s in (s_graph, s_direct):
        s.add_current((0.35, 0.25, 0.25), 2, wf)
        s.add_probe((0.6, 0.4, 0.3), 2, freqs=freqs)
        s.add_probe((0.25, 0.5, 0.35), 1, freqs=freqs)

    fg = FusedSubgridStepper(s_graph, graph_chunk=32)
    fd = FusedSubgridStepper(s_direct, graph_chunk=0)

    # Warm up graph capture, then measure memory pool across 100 steps
    fg.run(32)
    fg.synchronize()
    pool = cp.get_default_memory_pool()
    used_before = int(pool.used_bytes())
    fg.run(100)
    fg.synchronize()
    used_after = int(pool.used_bytes())
    used_delta = used_after - used_before
    assert used_delta == 0, (
        f"expected 0 bytes allocated during 100 steps, got {used_delta}"
    )

    fd.run(132)
    fd.synchronize()

    assert np.array_equal(fg.state_host(), fd.state_host())
    assert np.array_equal(fg.probe_signal(0), fd.probe_signal(0))
    assert np.array_equal(fg.probe_signal(1), fd.probe_signal(1))
    assert np.array_equal(fg.dft_data(0), fd.dft_data(0))
    assert np.array_equal(fg.dft_data(1), fd.dft_data(1))

    ulp_state = _max_ulp(fg.state_host(), fd.state_host())
    print(
        f"NUMBERS test_g4_graph_and_zero_alloc: graph_vs_direct_ulp={ulp_state} "
        f"used_bytes_before={used_before} used_bytes_after={used_after} "
        f"used_bytes_delta={used_delta}"
    )


# ----------------------------------------------------------------------- G5
def test_g5_device_modified_energy():
    """G5 (GPU): device `modified_energy` over 2000 LTS steps on `closed_mesh`
    at `0.95 dt_max^LTS` drifts by <= `50 eps sqrt(n)`."""
    _require_gpu()
    n_steps = 2000
    s = SubgridLTSFIT3D(
        closed_mesh(),
        lts=True,
        cfl=0.95,
        dt_rtol=1e-2,
        kernels="block",
        **closed_materials(),
    )
    s.e[:] = np.random.default_rng(1).standard_normal(s.n_free)
    fs = FusedSubgridStepper(s, graph_chunk=64)
    E = fs.run(n_steps, energy=True)
    drift = float(np.abs(E / E[0] - 1.0).max())
    tol = float(50.0 * EPS * np.sqrt(n_steps))
    assert drift <= tol, (drift, tol)
    print(
        f"NUMBERS test_g5_device_modified_energy: n_steps={n_steps} "
        f"drift={drift:.3e} tol={tol:.3e} E0={E[0]:.6e}"
    )


# ----------------------------------------------------------------------- G6
def test_g6_refusals():
    """G6 (CPU+GPU): contract refusals (`dispersive_groups`, unknown `n_state`,
    Bloch-periodic blocks, complex dtype, unknown mutation) raise the expected
    exceptions on both `FusedTables` and `FusedSubgridStepper`."""
    s_base = SubgridLTSFIT3D(closed_mesh(), lts=False, cfl=0.8, kernels="block")

    # 1. non-empty dispersive_groups
    s_disp = types.SimpleNamespace(**s_base.__dict__)
    s_disp.dispersive_groups = [{"n": 1}]
    with pytest.raises(NotImplementedError, match="dispersive"):
        FusedTables(s_disp)

    # 2. mismatched n_state
    s_nstate = types.SimpleNamespace(**s_base.__dict__)
    s_nstate.n_state = len(s_base.b) + len(s_base.e) + 7
    with pytest.raises(NotImplementedError, match="n_state"):
        FusedTables(s_nstate)

    # 3. Bloch-periodic block
    x = np.linspace(0.0, 0.3, 4)
    m_per = SubgridMesh3D(
        x, x, x, bc=["periodic", "pec", "pec"], phase=(np.exp(0.3j), 1.0, 1.0)
    )
    s_per = SubgridLTSFIT3D(m_per, lts=False, cfl=0.8, kernels="csr")
    with pytest.raises(NotImplementedError, match="Bloch"):
        FusedTables(s_per)

    # 4. complex fields
    s_cplx = types.SimpleNamespace(**s_base.__dict__)
    s_cplx.dtype = complex
    with pytest.raises(NotImplementedError, match="real fields"):
        FusedTables(s_cplx)

    # 5. invalid mutation hook
    with pytest.raises(ValueError, match="unknown mutation"):
        FusedTables(s_base, mutate="nonexistent")

    gpu_checked = False
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            gpu_checked = True
            for bad_s, pat in [
                (s_disp, "dispersive"),
                (s_nstate, "n_state"),
                (s_per, "Bloch"),
                (s_cplx, "real fields"),
            ]:
                with pytest.raises(NotImplementedError, match=pat):
                    FusedSubgridStepper(bad_s)
    except ImportError:
        if os.environ.get("WAKIS_REQUIRE_GPU") == "1":
            pytest.fail("WAKIS_REQUIRE_GPU=1 but CuPy is not installed")

    print(f"NUMBERS test_g6_refusals: cpu_refusals=5 gpu_checked={int(gpu_checked)}")
