# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Real-MPI verification of `DistributedSubgridFIT3D` (`wakis.subgrid3d.mpi`, check M4).

Run with:
    mpirun -n 2 pytest -s tests/test_122_subgrid3d_mpi_run.py
    mpirun -n 4 --oversubscribe pytest -s tests/test_122_subgrid3d_mpi_run.py

When ``MPI.COMM_WORLD.Get_size() < 2`` (or `mpi4py` is unavailable), tests skip
unless ``WAKIS_REQUIRE_MPI=1`` is set in the environment, in which case they
fail loudly so missing MPI never passes silently. All assertions are collective
(verdict computed on rank 0, broadcast, and asserted on every rank).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pytest

from wakis.subgrid3d import Box, SubgridFIT3D, SubgridLTSFIT3D, SubgridMesh3D
from wakis.subgrid3d.mpi import UNIT_ROUNDOFF, DistributedSubgridFIT3D


def _get_mpi_comm():
    require = os.environ.get("WAKIS_REQUIRE_MPI", "0") == "1"
    try:
        from mpi4py import MPI
    except ImportError as exc:
        if require:
            pytest.fail(f"WAKIS_REQUIRE_MPI=1 but mpi4py is not installed: {exc}")
        pytest.skip("mpi4py is not installed")
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        if require:
            pytest.fail(
                f"WAKIS_REQUIRE_MPI=1 requires MPI size >= 2, got {comm.Get_size()}"
            )
        pytest.skip("run under mpirun -n P with P >= 2")
    return comm


def _collective_assert(comm, ok, msg=""):
    verdict = comm.bcast((bool(ok), str(msg)) if comm.Get_rank() == 0 else None, root=0)
    assert verdict[0], verdict[1]


def gamma(n):
    nu = float(n) * UNIT_ROUNDOFF
    return nu / (1.0 - nu)


class SoftPulse:
    """Picklable Gaussian-modulated sinusoid current waveform (for mode='scatter')."""

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
def mpi_cases():
    """Build initial solvers on all ranks and serial 120-step reference on rank 0."""
    comm = _get_mpi_comm()
    me = comm.Get_rank()
    suite = {}
    for cid in ("a", "b", "c", "d", "e"):
        s = make_case(cid)
        ref = None
        if me == 0:
            e0 = np.asarray(s.e).copy()
            b0 = np.asarray(s.b).copy()
            cpml = getattr(s, "cpml", None)
            psi0 = [v.copy() for v in s.psi] if cpml is not None else []
            phi0 = [v.copy() for v in s.phi] if cpml is not None else []

            s.run(120)
            e_ref = np.asarray(s.e).copy()
            b_ref = np.asarray(s.b).copy()
            ref = {
                "e": e_ref,
                "b": b_ref,
                "psi": [v.copy() for v in s.psi] if cpml is not None else [],
                "phi": [v.copy() for v in s.phi] if cpml is not None else [],
                "probes": [
                    np.asarray(s.probes[ip]["values"]).copy()
                    for ip in range(len(s.probes))
                ],
                "dfts": [s.probes[ip]["dft"].copy() for ip in range(len(s.probes))],
                "energy": float(s.energy(e_ref, e_ref, b_ref)),
            }

            s.n = 0
            s.e[:] = e0
            s.b[:] = b0
            if cpml is not None:
                for d in range(3):
                    s.psi[d][:] = psi0[d]
                    s.phi[d][:] = phi0[d]
            for p in s.probes:
                p["values"].clear()
                if p["dft"] is not None:
                    p["dft"][:] = 0.0
        comm.Barrier()
        suite[cid] = (s, ref)
    return comm, suite


def test_m4_distributed_mpi_bitwise_and_mutations(mpi_cases):
    """
    M4: `DistributedSubgridFIT3D` on cases (a)-(e), in both ``mode="replicated"``
    and ``mode="scatter"`` (with ``method="rcb"`` and ``method="blocks"``),
    matches the serial 120-step run on rank 0 bit-for-bit (`np.array_equal`),
    satisfies Higham's `energy()` Allreduce bound, and fails under
    ``_mutation="drop_halo"`` and ``_mutation="skip_lts_exchange"``.
    """
    comm, suite = mpi_cases
    me = comm.Get_rank()
    P = comm.Get_size()
    max_energy_rel = 0.0
    case_energy_rel = {}

    for cid in ("a", "b", "c", "d", "e"):
        s0, ref = suite[cid]
        c_rel = 0.0
        for mode in ("replicated", "scatter"):
            for method in ("rcb", "blocks") if mode == "replicated" else ("rcb",):
                s_arg = s0 if (mode == "replicated" or me == 0) else None
                dist = DistributedSubgridFIT3D(
                    s_arg, comm, method=method, mode=mode, timers=True
                )
                dist.run(120)
                st = dist.gather_state()
                p0 = dist.probe_signal(0)
                p1 = dist.probe_signal(1)
                d0 = dist.dft_data(0)
                d1 = dist.dft_data(1)
                E_mpi = dist.energy()

                ok = True
                msg = ""
                if me == 0:
                    if not np.array_equal(st["e"], ref["e"]):
                        ok, msg = False, f"{cid} {mode} {method}: e mismatch"
                    elif not np.array_equal(st["b"], ref["b"]):
                        ok, msg = False, f"{cid} {mode} {method}: b mismatch"
                    elif getattr(s0, "cpml", None) is not None and not all(
                        np.array_equal(st["psi"][d], ref["psi"][d])
                        and np.array_equal(st["phi"][d], ref["phi"][d])
                        for d in range(3)
                    ):
                        ok, msg = False, f"{cid} {mode} {method}: cpml mismatch"
                    elif not (
                        np.array_equal(p0, ref["probes"][0])
                        and np.array_equal(p1, ref["probes"][1])
                    ):
                        ok, msg = False, f"{cid} {mode} {method}: probe mismatch"
                    elif not (
                        np.array_equal(d0, ref["dfts"][0])
                        and np.array_equal(d1, ref["dfts"][1])
                    ):
                        ok, msg = False, f"{cid} {mode} {method}: dft mismatch"
                    else:
                        n_terms = int(s0.n_free + s0.n_faces)
                        bound = 2.0 * gamma(n_terms + 2) * ref["energy"]
                        dE = abs(E_mpi - ref["energy"])
                        rel_e = dE / ref["energy"]
                        c_rel = max(c_rel, rel_e)
                        max_energy_rel = max(max_energy_rel, rel_e)
                        if dE > bound:
                            ok, msg = (
                                False,
                                f"{cid} {mode} {method}: energy dE={dE} > {bound}",
                            )
                _collective_assert(comm, ok, msg)
        case_energy_rel[cid] = c_rel

    # Mutations under real MPI: drop_halo on (a)-(d) and skip_lts_exchange on (b)
    drop_rels = {}
    for cid in ("a", "b", "c", "d"):
        s0, ref = suite[cid]
        dist_mut = DistributedSubgridFIT3D(
            s0, comm, method="rcb", mode="replicated", _mutation="drop_halo"
        )
        dist_mut.run(120)
        e_mut = dist_mut.gather("e")
        ok = True
        msg = ""
        if me == 0:
            e_ref = ref["e"]
            rel = float(np.max(np.abs(e_mut - e_ref)) / np.max(np.abs(e_ref)))
            drop_rels[cid] = rel
            if np.array_equal(e_mut, e_ref) or rel <= 1e-6:
                ok, msg = False, f"drop_halo on {cid} did not fail: rel={rel}"
        _collective_assert(comm, ok, msg)

    s_b, ref_b = suite["b"]
    dist_lts_mut = DistributedSubgridFIT3D(
        s_b, comm, method="rcb", mode="replicated", _mutation="skip_lts_exchange"
    )
    dist_lts_mut.run(120)
    e_lts_mut = dist_lts_mut.gather("e")
    ok = True
    msg = ""
    skip_rel_b = 0.0
    if me == 0:
        e_ref = ref_b["e"]
        skip_rel_b = float(np.max(np.abs(e_lts_mut - e_ref)) / np.max(np.abs(e_ref)))
        if np.array_equal(e_lts_mut, e_ref) or skip_rel_b <= 1e-6:
            ok, msg = False, f"skip_lts_exchange on b did not fail: rel={skip_rel_b}"
    _collective_assert(comm, ok, msg)

    if me == 0:
        print(
            f"NUMBERS test_m4_distributed_mpi_p{P}: "
            f"P={P} max_field_diff=0.0 max_probe_diff=0.0 max_dft_diff=0.0 "
            f"max_energy_rel={max_energy_rel:.2e} "
            f"case_a_rel={case_energy_rel['a']:.2e} "
            f"case_b_rel={case_energy_rel['b']:.2e} "
            f"case_c_rel={case_energy_rel['c']:.2e} "
            f"case_d_rel={case_energy_rel['d']:.2e} "
            f"case_e_rel={case_energy_rel['e']:.2e} "
            f"drop_halo_a={drop_rels['a']:.2e} "
            f"drop_halo_b={drop_rels['b']:.2e} "
            f"drop_halo_c={drop_rels['c']:.2e} "
            f"drop_halo_d={drop_rels['d']:.2e} "
            f"skip_lts_b={skip_rel_b:.2e}"
        )
