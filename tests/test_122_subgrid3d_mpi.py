# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Verification of in-process distributed-memory 3D subgridding
(`wakis.subgrid3d.mpi`, checks M1, M2, M3, M5 and distributed transpose).

Covers:
* M1: Partition ownership of active leaf cells, free edges and active faces
  for ``method in ("rcb", "rib", "blocks")`` and ``n_parts in (2, 3, 4)``,
  with weighted work imbalance reported.
* M2: Bitwise equality (`np.array_equal`) of `SubgridLocalCluster` against the
  serial solver over 120 steps for ``n_parts in (2, 3, 4)`` x
  ``method in ("rcb", "blocks")`` x cases (a)-(e) (fields `e`, `b`, CPML `psi`
  and `phi`, probe time series, and running DFTs), plus mutation checks
  ``_mutation="drop_halo"`` and ``_mutation="skip_lts_exchange"``.
* M3: Discrete energy `energy()` sum of per-rank partial inner products vs
  serial energy within Higham's summation bound ``2 * gamma_{n+2} * E``.
* M5: `NotImplementedError` refusals for ``backend="cupy"``, ``kernels="block"``,
  non-empty ``dispersive_groups``, and mismatched ``n_state``.
* Exact distributed step transpose `apply_T` dot-product test on cases (a)-(e)
  within ``(2 * gamma_m + 2 * gamma_N) |y|^T |F| |x|`` and transpose mutation.
"""

import copy

import numpy as np
import pytest

from wakis.subgrid3d import Box, SubgridFIT3D, SubgridLTSFIT3D, SubgridMesh3D
from wakis.subgrid3d.mpi import (
    UNIT_ROUNDOFF,
    SubgridLocalCluster,
    _entity_owners,
    partition_cells,
    partition_stats,
)


def gamma(n):
    """Higham (2002), eq. (3.5): ``n u / (1 - n u)`` for ``u = 2^-53``."""
    nu = float(n) * UNIT_ROUNDOFF
    return nu / (1.0 - nu)


class SoftPulse:
    """Picklable Gaussian-modulated sinusoid current waveform."""

    def __init__(self, f0=3.5e8, tau=1.5e-9, t0=4.0e-9):
        self.f0 = float(f0)
        self.tau = float(tau)
        self.t0 = float(t0)

    def __call__(self, t):
        s = (t - self.t0) / self.tau
        return float(np.exp(-(s * s)) * np.sin(2.0 * np.pi * self.f0 * t))


def closed_mesh():
    """Graded coarse grid, 2:1 patch with a nested 3:1 patch (as test_097)."""
    x = np.array([0, 0.1, 0.2, 0.3, 0.42, 0.55, 0.65, 0.75, 0.85, 1.0])
    m = SubgridMesh3D(x, np.linspace(0, 0.8, 9), np.linspace(0, 0.6, 7))
    p1 = m.add_patch((2, 2, 1), (7, 6, 5), ratio=2)
    m.add_patch((3, 2, 2), (7, 6, 6), ratio=3, parent=p1)
    return m


def closed_materials():
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


def _bloch_column():
    """Bloch-periodic complex mesh with a 2:1 patch (~13 coarse cells along x)."""
    H, n_coarse = 0.08, 13
    x = H * np.arange(n_coarse + 1)
    yz = np.array([0.0, H, 2 * H])
    phase = (1.0, np.exp(-1j * 0.7), np.exp(-1j * 0.4))
    m = SubgridMesh3D(
        x, yz, yz, bc=[("pec", "pec"), "periodic", "periodic"], phase=phase
    )
    m.add_patch((6, 0, 0), (13, 2, 2), ratio=2)
    return SubgridFIT3D(m, cfl=0.9)


def _seed_solver(s, seed, src_pt, probe_pts):
    """Populate random initial fields, a soft current source, and DFT probes."""
    rng = np.random.default_rng(seed)
    sb = float(np.sqrt(np.median(s.D.real) / np.median(s.W.real)))
    if s.dtype is complex or np.iscomplexobj(s.e):
        s.e[:] = rng.standard_normal(s.n_free) + 1j * rng.standard_normal(s.n_free)
        s.b[:] = sb * (
            rng.standard_normal(s.n_faces) + 1j * rng.standard_normal(s.n_faces)
        )
    else:
        s.e[:] = rng.standard_normal(s.n_free)
        s.b[:] = sb * rng.standard_normal(s.n_faces)
    if getattr(s, "cpml", None) is not None:
        for d in range(3):
            s.psi[d][:] = 0.1 * rng.standard_normal(s.psi[d].size)
            s.phi[d][:] = 0.1 * sb * rng.standard_normal(s.phi[d].size)
    s.add_current(src_pt, 2, SoftPulse(), amplitude=1.0)
    freqs = np.array([2.0e8, 3.5e8, 5.0e8])
    for pt, comp in probe_pts:
        s.add_probe(pt, comp, freqs=freqs)
    return s


def make_case(case_id):
    """
    Construct one of the five verification cases (a)-(e):
    (a) closed nested 2:1 + 3:1 graded mesh with eps/mu/PEC/PMC, global dt;
    (b) the same with LTS (2 levels, ratios [2, 3]);
    (c) open box with CPML and a 2:1 patch one coarse cell from the layer, global dt;
    (d) (c) with LTS;
    (e) Bloch-periodic complex mesh with a 2:1 patch, 13 coarse cells along x, global dt.
    """
    if case_id == "a":
        s = SubgridLTSFIT3D(closed_mesh(), lts=False, **closed_materials())
        return _seed_solver(
            s,
            seed=101,
            src_pt=(0.36, 0.31, 0.22),
            probe_pts=[((0.15, 0.25, 0.15), 2), ((0.51, 0.43, 0.33), 1)],
        )
    if case_id == "b":
        s = SubgridLTSFIT3D(closed_mesh(), lts=True, dt=1.0e-10, **closed_materials())
        return _seed_solver(
            s,
            seed=102,
            src_pt=(0.36, 0.31, 0.22),
            probe_pts=[((0.15, 0.25, 0.15), 2), ((0.51, 0.43, 0.33), 1)],
        )
    if case_id == "c":
        s = _open_box(lts=False, dt=3.5e-11)
        return _seed_solver(
            s,
            seed=103,
            src_pt=(0.48, 0.50, 0.50),
            probe_pts=[((0.32, 0.48, 0.48), 2), ((0.52, 0.50, 0.50), 2)],
        )
    if case_id == "d":
        s = _open_box(lts=True, dt=7.0e-11)
        return _seed_solver(
            s,
            seed=104,
            src_pt=(0.48, 0.50, 0.50),
            probe_pts=[((0.32, 0.48, 0.48), 2), ((0.52, 0.50, 0.50), 2)],
        )
    if case_id == "e":
        s = _bloch_column()
        return _seed_solver(
            s,
            seed=105,
            src_pt=(0.28, 0.06, 0.06),
            probe_pts=[((0.20, 0.06, 0.06), 2), ((0.72, 0.06, 0.06), 1)],
        )
    raise ValueError(f"unknown case {case_id!r}")


@pytest.fixture(scope="module")
def case_suite():
    """Build initial solver and 120-step serial reference snapshot for cases (a)-(e)."""
    suite = {}
    for cid in ("a", "b", "c", "d", "e"):
        s = make_case(cid)
        e0 = np.asarray(s.e).copy()
        b0 = np.asarray(s.b).copy()
        psi0 = [v.copy() for v in s.psi] if getattr(s, "cpml", None) is not None else []
        phi0 = [v.copy() for v in s.phi] if getattr(s, "cpml", None) is not None else []

        s.run(120)
        e_ref = np.asarray(s.e).copy()
        b_ref = np.asarray(s.b).copy()
        ref = {
            "e": e_ref,
            "b": b_ref,
            "psi": [v.copy() for v in s.psi]
            if getattr(s, "cpml", None) is not None
            else [],
            "phi": [v.copy() for v in s.phi]
            if getattr(s, "cpml", None) is not None
            else [],
            "probes": [
                np.asarray(s.probes[ip]["values"]).copy() for ip in range(len(s.probes))
            ],
            "dfts": [s.probes[ip]["dft"].copy() for ip in range(len(s.probes))],
            "energy": float(s.energy(e_ref, e_ref, b_ref)),
        }

        # Reset `s` to t = 0 so `SubgridLocalCluster(s, ...)` starts from initial state
        s.n = 0
        s.e[:] = e0
        s.b[:] = b0
        if getattr(s, "cpml", None) is not None:
            for d in range(3):
                s.psi[d][:] = psi0[d]
                s.phi[d][:] = phi0[d]
        for p in s.probes:
            p["values"].clear()
            if p["dft"] is not None:
                p["dft"][:] = 0.0

        suite[cid] = (s, ref)
    return suite


# ======================================================================== M1
def test_m1_partition_ownership(case_suite):
    """
    M1: Every active leaf cell is owned once, and every free edge and active
    face has a unique deterministic owner in ``0 .. n_parts - 1``.
    """
    imb_summary = {}
    for cid in ("a", "b", "c", "d", "e"):
        s0, _ = case_suite[cid]
        n_act = int(s0.active_cells.sum())
        for n_parts in (2, 3, 4):
            for method in ("rcb", "rib", "blocks"):
                parts = partition_cells(s0, n_parts, method=method, weights="work")
                assert parts.shape == (n_act,)
                assert parts.dtype == np.int64
                assert np.all((parts >= 0) & (parts < n_parts))

                owner_e, owner_f = _entity_owners(s0, parts, n_parts)
                assert owner_e.shape == (s0.n_free,)
                assert owner_f.shape == (s0.n_faces,)
                assert np.all((owner_e >= 0) & (owner_e < n_parts))
                assert np.all((owner_f >= 0) & (owner_f < n_parts))

                # Disjoint union of per-rank owned sets covers 0..N-1 exactly once
                all_e = np.concatenate(
                    [np.flatnonzero(owner_e == r) for r in range(n_parts)]
                )
                all_f = np.concatenate(
                    [np.flatnonzero(owner_f == r) for r in range(n_parts)]
                )
                assert np.array_equal(np.sort(all_e), np.arange(s0.n_free))
                assert np.array_equal(np.sort(all_f), np.arange(s0.n_faces))

                st = partition_stats(s0, parts=parts, n_parts=n_parts, weights="work")
                assert sum(st["cells"]) == n_act
                assert sum(st["owned_edges"]) == s0.n_free
                assert sum(st["owned_faces"]) == s0.n_faces
                assert st["imbalance"] >= 1.0 - 1e-15
                imb_summary[(cid, n_parts, method)] = st["imbalance"]

    # Weighted RCB/RIB balance work to within a single fine cell weight
    for cid in ("a", "b", "c", "d", "e"):
        for n_parts in (2, 3, 4):
            assert imb_summary[(cid, n_parts, "rcb")] < 1.05
            assert imb_summary[(cid, n_parts, "rib")] < 1.05

    print(
        "NUMBERS test_m1_partition_ownership: "
        f"case_a_p4_rcb={imb_summary[('a', 4, 'rcb')]:.4f} "
        f"case_a_p4_blocks={imb_summary[('a', 4, 'blocks')]:.4f} "
        f"case_b_p3_rcb={imb_summary[('b', 3, 'rcb')]:.4f} "
        f"case_b_p3_blocks={imb_summary[('b', 3, 'blocks')]:.4f} "
        f"case_b_p4_rcb={imb_summary[('b', 4, 'rcb')]:.4f} "
        f"case_b_p4_rib={imb_summary[('b', 4, 'rib')]:.4f} "
        f"case_b_p4_blocks={imb_summary[('b', 4, 'blocks')]:.4f} "
        f"case_d_p4_rcb={imb_summary[('d', 4, 'rcb')]:.4f} "
        f"case_d_p4_blocks={imb_summary[('d', 4, 'blocks')]:.4f}"
    )


# ======================================================================== M2 & M3
def test_m2_m3_local_cluster_bitwise_energy_and_mutations(case_suite):
    """
    M2: `SubgridLocalCluster` for ``n_parts in {2, 3, 4}`` x
    ``method in {"rcb", "blocks"}`` x cases (a)-(e) after 120 steps matches the
    serial run bit-for-bit (`np.array_equal`) on fields `e`, `b` (and CPML `psi`,
    `phi`), probe time series, and running DFTs.
    Mutations ``_mutation="drop_halo"`` and ``_mutation="skip_lts_exchange"``
    produce ``max|Delta e| > 1e-6 * max|e_ref|``.

    M3: `energy()` sum of per-rank partial inner products matches the serial
    dot product within Higham's bound ``2 * gamma_{n+2} * E``.
    """
    expected_exchanges = {"a": 2, "b": 12, "c": 2, "d": 4, "e": 2}
    max_energy_rel = 0.0
    max_energy_ratio_to_bound = 0.0
    energy_by_case = {}

    for cid in ("a", "b", "c", "d", "e"):
        s0, ref = case_suite[cid]
        e_ref = ref["e"]
        b_ref = ref["b"]
        E_ref = ref["energy"]
        n_terms = int(s0.n_free + s0.n_faces)
        e_bound = 2.0 * gamma(n_terms + 2) * E_ref
        case_max_de = 0.0

        for n_parts in (2, 3, 4):
            for method in ("rcb", "blocks"):
                lc = SubgridLocalCluster(s0, n_parts=n_parts, method=method)
                assert lc.exchange_counts_per_step() == expected_exchanges[cid]
                lc.run(120)
                st = lc.gather_state()
                assert np.array_equal(st["e"], e_ref), (cid, n_parts, method, "e")
                assert np.array_equal(st["b"], b_ref), (cid, n_parts, method, "b")
                if getattr(s0, "cpml", None) is not None:
                    for d in range(3):
                        assert np.array_equal(st["psi"][d], ref["psi"][d])
                        assert np.array_equal(st["phi"][d], ref["phi"][d])
                for ip in range(len(s0.probes)):
                    assert np.array_equal(lc.probe_signal(ip), ref["probes"][ip])
                    assert np.array_equal(lc.dft_data(ip), ref["dfts"][ip])

                # M3: Energy summation bound
                dE = abs(lc.energy() - E_ref)
                assert dE <= e_bound, (cid, n_parts, method, dE, e_bound)
                rel_e = dE / E_ref
                case_max_de = max(case_max_de, rel_e)
                max_energy_rel = max(max_energy_rel, rel_e)
                max_energy_ratio_to_bound = max(max_energy_ratio_to_bound, dE / e_bound)
        energy_by_case[cid] = (case_max_de, 2.0 * gamma(n_terms + 2))

    # Mutation 1: _mutation="drop_halo" on cases (a)-(e) with n_parts=3, method="rcb"
    drop_halo_rel = {}
    for cid in ("a", "b", "c", "d", "e"):
        s0, ref = case_suite[cid]
        e_ref = ref["e"]
        lc_mut = SubgridLocalCluster(s0, n_parts=3, method="rcb", _mutation="drop_halo")
        lc_mut.run(120)
        e_mut = lc_mut.gather("e")
        assert not np.array_equal(e_mut, e_ref)
        rel = float(np.max(np.abs(e_mut - e_ref)) / np.max(np.abs(e_ref)))
        assert rel > 1e-6, (cid, rel)
        drop_halo_rel[cid] = rel

    # Mutation 2: _mutation="skip_lts_exchange" on LTS cases (b) and (d)
    skip_lts_rel = {}
    for cid in ("b", "d"):
        s0, ref = case_suite[cid]
        e_ref = ref["e"]
        lc_mut = SubgridLocalCluster(
            s0, n_parts=3, method="rcb", _mutation="skip_lts_exchange"
        )
        lc_mut.run(120)
        e_mut = lc_mut.gather("e")
        assert not np.array_equal(e_mut, e_ref)
        rel = float(np.max(np.abs(e_mut - e_ref)) / np.max(np.abs(e_ref)))
        assert rel > 1e-6, (cid, rel)
        skip_lts_rel[cid] = rel

    print(
        "NUMBERS test_m2_local_cluster_bitwise_and_mutations: "
        "max_field_diff=0.0 max_probe_diff=0.0 max_dft_diff=0.0 "
        f"exchanges_global={expected_exchanges['a']} "
        f"exchanges_lts_2_3={expected_exchanges['b']} "
        f"exchanges_lts_2={expected_exchanges['d']} "
        f"drop_halo_a={drop_halo_rel['a']:.2e} "
        f"drop_halo_b={drop_halo_rel['b']:.2e} "
        f"drop_halo_c={drop_halo_rel['c']:.2e} "
        f"drop_halo_d={drop_halo_rel['d']:.2e} "
        f"drop_halo_e={drop_halo_rel['e']:.2e} "
        f"skip_lts_b={skip_lts_rel['b']:.2e} "
        f"skip_lts_d={skip_lts_rel['d']:.2e}"
    )
    print(
        "NUMBERS test_m3_energy_allreduce_higham: "
        f"max_rel_err={max_energy_rel:.2e} "
        f"max_bound_ratio={max_energy_ratio_to_bound:.3f} "
        f"case_a_rel={energy_by_case['a'][0]:.2e} "
        f"case_a_bound={energy_by_case['a'][1]:.2e} "
        f"case_b_rel={energy_by_case['b'][0]:.2e} "
        f"case_c_rel={energy_by_case['c'][0]:.2e} "
        f"case_c_bound={energy_by_case['c'][1]:.2e} "
        f"case_e_rel={energy_by_case['e'][0]:.2e} "
        f"case_e_bound={energy_by_case['e'][1]:.2e}"
    )


# ======================================================================== M5
def test_m5_unsupported_feature_refusals(case_suite):
    """
    M5: `SubgridLocalCluster` and `build_pieces` raise `NotImplementedError`
    with a clear message for ``backend="cupy"``, ``kernels="block"``, non-empty
    ``dispersive_groups``, or unknown ``n_state``.
    """
    s_base, _ = case_suite["a"]

    s_cupy = copy.copy(s_base)
    s_cupy.backend = "cupy"
    with pytest.raises(NotImplementedError, match="backend"):
        SubgridLocalCluster(s_cupy, n_parts=2)

    s_blk = SubgridLTSFIT3D(closed_mesh(), lts=False, kernels="block")
    with pytest.raises(NotImplementedError, match="kernels"):
        SubgridLocalCluster(s_blk, n_parts=2)

    s_disp = copy.copy(s_base)
    s_disp.dispersive_groups = [object()]
    with pytest.raises(NotImplementedError, match="dispersive"):
        SubgridLocalCluster(s_disp, n_parts=2)

    s_nstate = copy.copy(s_base)
    s_nstate.__dict__["n_state"] = int(s_base.b.size + s_base.e.size + 17)
    with pytest.raises(NotImplementedError, match="n_state"):
        SubgridLocalCluster(s_nstate, n_parts=2)

    print(
        "NUMBERS test_m5_unsupported_feature_refusals: "
        "refused_cupy=1 refused_block=1 refused_dispersive=1 refused_n_state=1"
    )


# ============================================================ Transpose check
def _abs_step_bound(s, x_vec, y_vec):
    """
    Upper bound on ``|y|^T |F| |x|`` and row-Accumulation depth ``m`` for the
    one-step linear operator ``F`` of solver `s`.
    """
    nb, ne = s.n_faces, s.n_free
    xb = np.abs(x_vec[:nb])
    xe = np.abs(x_vec[nb : nb + ne])
    np.abs(y_vec[:nb])
    np.abs(y_vec[nb : nb + ne])
    dt = float(s.dt)
    absK = abs(s.K)
    absKH = abs(s.KH)
    W = np.abs(s.W)
    Dinv = np.abs(s.Dinv)

    # |R_0| |xe|: under LTS, expand each Chebyshev polynomial with |coefficients|
    ze = xe.copy()
    depth_r = 1
    if getattr(s, "lts", None) is not None and s.lts.L > 0:
        for lv in range(s.lts.L, 0, -1):
            d = s.lts.levels[lv]
            abs_KF = abs(d.K_F)
            abs_KC = abs(d.K_C)
            abs_KHF = abs(d.KH_F)
            W_lv = np.abs(d.W)
            Dinv_lv = np.abs(d.Dinv)
            zF = ze[d.F].copy()
            zC = ze[d.C].copy()
            d_old = np.zeros_like(zF)
            dk = zF.copy()
            g = abs_KC @ zC
            for k in range(1, d.p):
                Xd = (
                    d.tau2
                    * Dinv_lv
                    * (abs_KHF @ (W_lv * (abs_KF @ dk + abs(d.kappa[k]) * g)))
                )
                d_new = (
                    2.0 * abs(d.delta) * dk
                    + 2.0 * abs(d.beta) * Xd
                    + 2.0 * abs(d.T[k]) * zF
                    + d_old
                )
                d_old, dk = dk, d_new
            ze[d.F] = zF + abs(d.out) * dk
            m_kf = int(np.diff(abs_KF.indptr).max())
            m_khf = int(np.diff(abs_KHF.indptr).max())
            depth_r += 4 * d.p * (m_kf + m_khf)

    b1 = xb + dt * (absK @ ze)
    o = nb + ne
    psi_1, phi_1 = [], []
    cpml = getattr(s, "cpml", None)
    if cpml is not None:
        for d in range(3):
            nf = cpml.pf[d].size
            xpsi = np.abs(x_vec[o : o + nf])
            o += nf
            psi_d = np.abs(cpml.bh[d]) * xpsi + np.abs(cpml.ah[d]) * (
                abs(cpml.Kf[d]) @ ze
            )
            b1[cpml.pf[d]] += dt * psi_d
            psi_1.append(psi_d)

    rhs = absKH @ (W * b1)
    if cpml is not None:
        for d in range(3):
            ne_d = cpml.pe[d].size
            xphi = np.abs(x_vec[o : o + ne_d])
            o += ne_d
            phi_d = np.abs(cpml.be[d]) * xphi + np.abs(cpml.ae[d]) * (
                abs(cpml.Ke[d]) @ (W * b1)
            )
            rhs[cpml.pe[d]] += phi_d
            phi_1.append(phi_d)

    e1 = xe + dt * Dinv * rhs
    f_abs = [b1, e1]
    if cpml is not None:
        f_abs += psi_1 + phi_1
    f_abs_vec = np.concatenate(f_abs)

    mK = int(np.diff(s.K.indptr).max())
    mKH = int(np.diff(s.KH.tocsr().indptr).max())
    m_depth = mK + mKH + depth_r + 6
    cond = float(np.abs(y_vec) @ f_abs_vec)
    return cond, m_depth


def test_distributed_transpose_dot_product(case_suite):
    """
    Exact transpose `apply_T` of distributed step `apply` satisfies
    ``|<apply(x), y> - <x, apply_T(y)>| <= (2 gamma_m + 2 gamma_N) |y|^T |F| |x|``
    across cases (a)-(e), and fails when one transpose entry on rank 1 is perturbed.
    """
    rel_by_case = {}
    ratio_by_case = {}

    for cid in ("a", "b", "c", "d", "e"):
        s0, _ = case_suite[cid]
        lc = SubgridLocalCluster(s0, n_parts=3, method="rcb")
        N = int(lc._layout_sizes()[-1])
        rng = np.random.default_rng(500 + ord(cid))
        if s0.dtype is complex:
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            y = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        else:
            x = rng.standard_normal(N)
            y = rng.standard_normal(N)

        Fx = lc.apply(x)
        FTy = lc.apply_T(y)
        # Algebraic bilinear inner product sum(Fx * y) == sum(x * FTy)
        lhs = np.sum(Fx * y)
        rhs = np.sum(x * FTy)
        diff = float(np.abs(lhs - rhs))
        cond, m_depth = _abs_step_bound(s0, x, y)
        bound = (2.0 * gamma(m_depth) + 2.0 * gamma(N)) * cond
        assert diff <= bound, (cid, diff, bound)
        rel_by_case[cid] = diff / cond
        ratio_by_case[cid] = diff / bound

    # Mutation: perturb KHT on rank 1 of case (b) -> transpose identity breaks
    s_b, _ = case_suite["b"]
    lc_mut = SubgridLocalCluster(s_b, n_parts=3, method="rcb")
    lc_mut.ranks[1].pc["KHT"].data[:] *= 1.1
    N_b = int(lc_mut._layout_sizes()[-1])
    rng = np.random.default_rng(500 + ord("b"))
    xb = rng.standard_normal(N_b)
    yb = rng.standard_normal(N_b)
    cond_b, _ = _abs_step_bound(s_b, xb, yb)
    diff_mut = float(
        np.abs(np.sum(lc_mut.apply(xb) * yb) - np.sum(xb * lc_mut.apply_T(yb)))
    )
    rel_mut = diff_mut / cond_b
    assert rel_mut > 1e-6

    print(
        "NUMBERS test_distributed_transpose_dot_product: "
        f"case_a_rel={rel_by_case['a']:.2e} "
        f"case_b_rel={rel_by_case['b']:.2e} "
        f"case_c_rel={rel_by_case['c']:.2e} "
        f"case_d_rel={rel_by_case['d']:.2e} "
        f"case_e_rel={rel_by_case['e']:.2e} "
        f"max_bound_ratio={max(ratio_by_case.values()):.3f} "
        f"mut_KT_rel={rel_mut:.2e}"
    )
