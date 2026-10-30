# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Tests for checkpoint and restart of coupled field + particle runs (P7a)."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import m_e

import wakis.particles.checkpoint as ckpt_mod
from wakis import GridFIT3D, SolverFIT3D
from wakis.particles import (
    CHECKPOINT_STATE_KEYS,
    CPML_PSI_ATTRS,
    BeamInjector,
    Collector,
    ParticleSolver,
    Species,
    compute_config_hash,
)


def _detect_mpi_from_env() -> tuple[int, int]:
    """Detect MPI world size and rank from launcher environment variables without importing mpi4py.MPI."""
    size = 1
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
        val = os.environ.get(var)
        if val is not None and val.isdigit():
            size = max(size, int(val))
            break
    rank = 0
    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"):
        val = os.environ.get(var)
        if val is not None and val.isdigit():
            rank = int(val)
            break
    if size == 1 and "PMIX_RANK" in os.environ and os.environ["PMIX_RANK"].isdigit():
        size = max(size, rank + 1)
    return size, rank


_MPI_SIZE, _MPI_RANK = _detect_mpi_from_env()


def _get_mpi():
    """Lazily initialize and return ``mpi4py.MPI`` only when an MPI test needs it."""
    MPI = pytest.importorskip("mpi4py.MPI")
    if not MPI.Is_initialized():
        MPI.Init()
        atexit.register(
            lambda: (
                MPI.Finalize()
                if MPI.Is_initialized() and not MPI.Is_finalized()
                else None
            )
        )
    return MPI


if _MPI_SIZE > 1:
    # Isolate pytest's TempPathFactory root per MPI rank so concurrent ranks
    # never share /tmp/pytest-of-<user> or race when garbage-collecting old
    # numbered directories (garbage-<uuid>).
    _rank_temproot = (
        Path(tempfile.gettempdir()).resolve() / f"pytest-mpi-rank-{_MPI_RANK}"
    )
    _rank_temproot.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = str(_rank_temproot)

_serial_only = pytest.mark.skipif(
    _MPI_SIZE > 1,
    reason="Serial checkpoint test; skipped when MPI world size > 1",
)
_mpi_only = pytest.mark.skipif(
    _MPI_SIZE < 2,
    reason="MPI checkpoint test requires mpiexec -n >= 2",
)


@pytest.fixture(autouse=True)
def _mpi_isolate_tmp_and_sync(request: pytest.FixtureRequest):
    """Ensure per-rank pytest basetemp isolation and synchronize ranks between tests."""
    if _MPI_SIZE > 1:
        tpf = getattr(request.config, "_tmp_path_factory", None)
        if (
            tpf is not None
            and getattr(tpf, "_basetemp", None) is None
            and getattr(tpf, "_given_basetemp", None) is not None
        ):
            given = Path(tpf._given_basetemp)
            suffix = f"-rank{_MPI_RANK}"
            if not given.name.endswith(suffix):
                tpf._given_basetemp = given.parent / f"{given.name}{suffix}"
    yield
    if _MPI_SIZE > 1:
        MPI = _get_mpi()
        MPI.COMM_WORLD.Barrier()


def _clean_mpi_subprocess_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with OpenMPI/PMIx/PMI/OPAL variables removed."""
    mpi_prefixes = ("OMPI_", "PMIX_", "PMI_", "OPAL_", "PRTE_")
    return {k: v for k, v in os.environ.items() if not k.startswith(mpi_prefixes)}


def _find_rank_npz(ckpt_dir: Path, rank: int = 0) -> Path:
    """Return the path to the active token-suffixed rank .npz file in manifest.json."""
    manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    prefix = f"rank_{rank:06d}-"
    matches = [
        k for k in manifest["files"] if k.startswith(prefix) and k.endswith(".npz")
    ]
    assert len(matches) == 1, (
        f"Expected 1 rank archive for rank {rank}, found {matches}"
    )
    return ckpt_dir / matches[0]


def _mpi_tmp_dirs(tmp_path: Path, comm) -> tuple[Path, Path]:
    """Return ``(shared_root, rank_tmp)`` with per-rank suffix and rank-0 broadcast."""
    rank = int(comm.Get_rank())
    rank_tmp = tmp_path / f"rank_{rank:06d}"
    rank_tmp.mkdir(parents=True, exist_ok=True)
    shared_root = Path(comm.bcast(str(rank_tmp), root=0))
    comm.Barrier()
    return shared_root, rank_tmp


def _build_coupled_case(
    *,
    seed_initial_state: bool,
    nx: int = 14,
    ny: int = 12,
    nz: int = 16,
    xmax: float = 0.07,
    dt_factor: float = 0.45,
    use_gpu: int = 0,
    use_mpi: bool = False,
):
    """Construct the coupled CPML + injector + collector + sort_every test setup.

    Parameters
    ----------
    seed_initial_state : bool
        If True, populate initial asymmetric electromagnetic fields (extending
        into the CPML layers) and initial macroparticles (interior particles,
        a heavy boundary-crossing particle, and a collector-crossing particle).
        If False, return a fresh empty solver + particle solver ready for
        ``load_checkpoint``.
    """
    xmin, ymin, ymax, zmin, zmax = -xmax, -0.06, 0.06, 0.0, 0.16
    grid = GridFIT3D(
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        nx,
        ny,
        nz,
        use_mpi=use_mpi,
        verbose=0,
    )
    bc_low = ["cpml", "periodic", "pec"]
    bc_high = ["pec", "periodic", "cpml"]
    solver = SolverFIT3D(
        grid,
        bc_low=bc_low,
        bc_high=bc_high,
        use_stl=False,
        use_gpu=use_gpu,
        use_mpi=use_mpi,
        n_pml=4,
        verbose=0,
    )
    solver.dt = float(solver.dt * dt_factor)
    # Enable localized ohmic conductivity so solver.J_old and solver.step_0 are
    # active in the time-stepping loop.
    solver.sigma[5:8, 4:7, 6:10, "z"] = 2.0e-3
    solver.use_conductivity = True

    sp = Species(
        name="electrons",
        charge=-e_charge,
        mass=m_e,
        capacity=256,
        track_ids=True,
        use_gpu=bool(use_gpu),
    )

    col = Collector(
        name="anode_probe",
        bounds=(-0.03, 0.03, -0.03, 0.03, 0.095, 0.115),
    )

    dt_ref = float(solver.dt)
    inj = BeamInjector(
        species=sp,
        z0=0.015,
        current=lambda t, dt0=dt_ref: 2.5e-3 * (1.0 + 0.35 * (t / dt0)),
        gamma=1.08,
        profile="gaussian",
        radius=0.012,
        x0=0.002,
        y0=-0.001,
        n_macro_per_step=4,
        direction=1,
        thermal_u=4.0e5,
        seed=12345,
    )

    ps = ParticleSolver(
        solver,
        species=[sp],
        collectors=[col],
        injectors=[inj],
        shape=1,
        pusher="boris",
        deposit_scheme="esirkepov",
        sort_every=3,
    )

    if seed_initial_state:
        rank_offset = int(getattr(solver, "rank", 0)) if use_mpi else 0
        rng = np.random.default_rng(98765 + rank_offset)
        e_host = rng.normal(0.0, 1.5e4, size=solver.E.toarray().shape).astype(
            np.float64
        )
        h_host = rng.normal(0.0, 4.0e1, size=solver.H.toarray().shape).astype(
            np.float64
        )
        ckpt_mod._copy_into_target(solver.E.toarray(), e_host)
        ckpt_mod._copy_into_target(solver.H.toarray(), h_host)
        if use_mpi and int(getattr(solver, "size", 1)) > 1:
            solver._mpi_communicate(solver.E)
            solver._mpi_communicate(solver.H)

        # Seed:
        # 0..3: interior 3D moving particles (particle 2 at z=0.078 with +uz
        #       and particle 3 at z=0.082 with -uz cross the z=0.08 MPI slab
        #       boundary in opposite directions)
        # 4: heavy particle right next to x_high=xmax that gets absorbed on step 0
        #    (sets a large initial _particles_q_scale_ref and boundary_absorbed)
        # 5: particle just upstream of collector z=0.095 that gets absorbed by
        #    collector on step 0 or 1
        dx_cell = (xmax - xmin) / nx
        dz_cell = (zmax - zmin) / nz
        x_init = np.array(
            [
                -0.015,
                0.012,
                -0.008,
                0.018,
                xmax - 0.15 * dx_cell,
                0.005,
            ],
            dtype=np.float64,
        )
        y_init = np.array(
            [-0.010, 0.015, 0.022, -0.018, 0.004, -0.006],
            dtype=np.float64,
        )
        z_init = np.array(
            [
                0.045,
                0.062,
                0.0785,
                0.0815,
                0.065,
                0.095 - 0.15 * dz_cell,
            ],
            dtype=np.float64,
        )
        ux_init = np.array(
            [1.5e7, -1.2e7, 0.9e7, -1.8e7, 0.45 * c_light, 0.4e7],
            dtype=np.float64,
        )
        uy_init = np.array(
            [-1.1e7, 1.4e7, -1.6e7, 0.8e7, -0.3e7, 0.5e7],
            dtype=np.float64,
        )
        uz_init = np.array(
            [2.8e7, 2.2e7, 0.38 * c_light, -0.38 * c_light, 0.6e7, 0.45 * c_light],
            dtype=np.float64,
        )
        w_init = np.array(
            [1.2e3, 1.5e3, 0.9e3, 1.8e3, 5.0e8, 2.4e4],
            dtype=np.float64,
        )
        sp.add_particles(
            x=x_init,
            y=y_init,
            z=z_init,
            ux=ux_init,
            uy=uy_init,
            uz=uz_init,
            w=w_init,
        )
        solver.nstep = 5

    ps.attach()
    return solver, ps


def _extract_state_snapshot(ps: ParticleSolver) -> dict:
    """Extract a complete host snapshot of fields, particles, and diagnostics."""
    s = ps.solver
    if ps.on_gpu and getattr(ps.boundaries, "_gpu_stats_dirty", False):
        ps.boundaries._sync_gpu_stats()

    j_old_raw = getattr(s, "J_old", None)
    j_old_snap = (
        ckpt_mod._to_host_f64(j_old_raw).copy()
        if j_old_raw is not None
        else np.zeros_like(ckpt_mod._to_host_f64(s.J.toarray()))
    )
    j_prev_snap = (
        ckpt_mod._to_host_f64(ps._J_prev).copy() if ps._J_prev is not None else None
    )

    snap: dict = {
        "E": ckpt_mod._to_host_f64(s.E.toarray()).copy(),
        "H": ckpt_mod._to_host_f64(s.H.toarray()).copy(),
        "J": ckpt_mod._to_host_f64(s.J.toarray()).copy(),
        "J_old": j_old_snap,
        "J_particles": ckpt_mod._to_host_f64(ps.J_particles).copy(),
        "J_prev": j_prev_snap,
        "j_in_solver": bool(ps._j_in_solver),
        "solver_step_0": bool(getattr(s, "step_0", False)),
        "solver_nstep": int(getattr(s, "nstep", 0)),
        "q_bg": ckpt_mod._to_host_f64(s._particles_q_bg).copy(),
        "gauss_residual": float(ps.gauss_residual()),
        "n_particle_steps": int(ps.n_particle_steps),
        "initialized_half_step": bool(ps._initialized_half_step),
        "injected_charge": float(ps.injected_charge()),
        "injected_energy": float(ps.injected_energy()),
        "absorbed_charge": float(ps.absorbed_charge()),
        "absorbed_energy": float(ps.absorbed_energy()),
        "boundary_absorbed_charge": dict(ps.boundaries.absorbed_charge),
        "boundary_absorbed_energy": dict(ps.boundaries.absorbed_energy),
        "boundary_absorbed_count": dict(ps.boundaries.absorbed_count),
    }

    cpml = {}
    for attr in CPML_PSI_ATTRS:
        val = getattr(s, attr, None)
        if val is not None:
            cpml[attr] = ckpt_mod._to_host_f64(val).copy()
    snap["cpml"] = cpml

    sp_list = []
    for sp in ps.species:
        n = int(sp.n_particles)
        ids = ckpt_mod._to_host_array(sp.id).copy()
        order = np.argsort(ids, kind="stable")
        sp_list.append(
            {
                "n_particles": n,
                "next_id": int(sp._next_id),
                "u_is_synced": bool(sp.u_is_synced),
                "x_raw": ckpt_mod._to_host_f64(sp.x).copy(),
                "y_raw": ckpt_mod._to_host_f64(sp.y).copy(),
                "z_raw": ckpt_mod._to_host_f64(sp.z).copy(),
                "ux_raw": ckpt_mod._to_host_f64(sp.ux).copy(),
                "uy_raw": ckpt_mod._to_host_f64(sp.uy).copy(),
                "uz_raw": ckpt_mod._to_host_f64(sp.uz).copy(),
                "w_raw": ckpt_mod._to_host_f64(sp.w).copy(),
                "id_raw": ids,
                "x_sorted": ckpt_mod._to_host_f64(sp.x)[order].copy(),
                "y_sorted": ckpt_mod._to_host_f64(sp.y)[order].copy(),
                "z_sorted": ckpt_mod._to_host_f64(sp.z)[order].copy(),
                "ux_sorted": ckpt_mod._to_host_f64(sp.ux)[order].copy(),
                "uy_sorted": ckpt_mod._to_host_f64(sp.uy)[order].copy(),
                "uz_sorted": ckpt_mod._to_host_f64(sp.uz)[order].copy(),
                "w_sorted": ckpt_mod._to_host_f64(sp.w)[order].copy(),
                "id_sorted": ids[order].copy(),
            }
        )
    snap["species"] = sp_list

    col_list = []
    for col in ps.boundaries.collectors:
        if col.current_history:
            hist_arr = np.asarray(col.current_history, dtype=np.float64).reshape(-1, 2)
            t_arr = hist_arr[:, 0].copy()
            i_arr = hist_arr[:, 1].copy()
        else:
            t_arr = np.empty(0, dtype=np.float64)
            i_arr = np.empty(0, dtype=np.float64)
        col_list.append(
            {
                "absorbed_charge": float(col.absorbed_charge),
                "absorbed_energy": float(col.absorbed_energy),
                "absorbed_count": int(col.absorbed_count),
                "t_trace": t_arr,
                "i_trace": i_arr,
            }
        )
    snap["collectors"] = col_list
    return snap


def _fmt_exc(exc: BaseException) -> str:
    """Format an AssertionError message onto a single non-empty summary line."""
    return " | ".join(ln.strip() for ln in str(exc).splitlines() if ln.strip())[:160]


def _assert_snapshots_bitwise_equal(
    snap_a: dict,
    snap_b: dict,
    *,
    check_raw_particle_order: bool = True,
) -> None:
    """Assert that two snapshots from Run A and Run B are bitwise identical."""
    np.testing.assert_array_equal(snap_a["E"], snap_b["E"], err_msg="E mismatch")
    np.testing.assert_array_equal(snap_a["H"], snap_b["H"], err_msg="H mismatch")
    np.testing.assert_array_equal(snap_a["J"], snap_b["J"], err_msg="J mismatch")
    np.testing.assert_array_equal(
        snap_a["J_old"], snap_b["J_old"], err_msg="J_old mismatch"
    )
    np.testing.assert_array_equal(
        snap_a["J_particles"],
        snap_b["J_particles"],
        err_msg="J_particles mismatch",
    )
    if snap_a["J_prev"] is not None or snap_b["J_prev"] is not None:
        assert snap_a["J_prev"] is not None and snap_b["J_prev"] is not None
        np.testing.assert_array_equal(
            snap_a["J_prev"], snap_b["J_prev"], err_msg="J_prev mismatch"
        )
    assert snap_a["j_in_solver"] == snap_b["j_in_solver"], "j_in_solver mismatch"
    assert snap_a["solver_step_0"] == snap_b["solver_step_0"], "solver_step_0 mismatch"
    assert snap_a["solver_nstep"] == snap_b["solver_nstep"], (
        f"solver_nstep mismatch: {snap_a['solver_nstep']} vs {snap_b['solver_nstep']}"
    )
    np.testing.assert_array_equal(
        snap_a["q_bg"], snap_b["q_bg"], err_msg="q_bg mismatch"
    )
    assert snap_a["gauss_residual"] == snap_b["gauss_residual"], (
        f"gauss_residual mismatch: {snap_a['gauss_residual']} vs {snap_b['gauss_residual']}"
    )
    assert snap_a["n_particle_steps"] == snap_b["n_particle_steps"]
    assert snap_a["initialized_half_step"] == snap_b["initialized_half_step"]
    assert snap_a["injected_charge"] == snap_b["injected_charge"]
    assert snap_a["injected_energy"] == snap_b["injected_energy"]
    assert snap_a["absorbed_charge"] == snap_b["absorbed_charge"]
    assert snap_a["absorbed_energy"] == snap_b["absorbed_energy"]
    assert snap_a["boundary_absorbed_charge"] == snap_b["boundary_absorbed_charge"]
    assert snap_a["boundary_absorbed_energy"] == snap_b["boundary_absorbed_energy"]
    assert snap_a["boundary_absorbed_count"] == snap_b["boundary_absorbed_count"]

    assert set(snap_a["cpml"].keys()) == set(snap_b["cpml"].keys())
    for k in snap_a["cpml"]:
        np.testing.assert_array_equal(
            snap_a["cpml"][k], snap_b["cpml"][k], err_msg=f"CPML {k} mismatch"
        )

    assert len(snap_a["species"]) == len(snap_b["species"])
    for sp_a, sp_b in zip(snap_a["species"], snap_b["species"]):
        assert sp_a["n_particles"] == sp_b["n_particles"]
        assert sp_a["next_id"] == sp_b["next_id"]
        assert sp_a["u_is_synced"] == sp_b["u_is_synced"]
        # Unique IDs must hold
        assert len(np.unique(sp_b["id_sorted"])) == sp_b["n_particles"]
        for comp in ("x", "y", "z", "ux", "uy", "uz", "w", "id"):
            np.testing.assert_array_equal(
                sp_a[f"{comp}_sorted"],
                sp_b[f"{comp}_sorted"],
                err_msg=f"species sorted {comp} mismatch",
            )
            if check_raw_particle_order:
                np.testing.assert_array_equal(
                    sp_a[f"{comp}_raw"],
                    sp_b[f"{comp}_raw"],
                    err_msg=f"species raw {comp} mismatch",
                )

    assert len(snap_a["collectors"]) == len(snap_b["collectors"])
    for col_a, col_b in zip(snap_a["collectors"], snap_b["collectors"]):
        assert col_a["absorbed_charge"] == col_b["absorbed_charge"]
        assert col_a["absorbed_energy"] == col_b["absorbed_energy"]
        assert col_a["absorbed_count"] == col_b["absorbed_count"]
        np.testing.assert_array_equal(col_a["t_trace"], col_b["t_trace"])
        np.testing.assert_array_equal(col_a["i_trace"], col_b["i_trace"])


def run_subprocess_restart_worker(
    ckpt_dir: str,
    out_npz: str,
    n_rem_steps: int,
    use_gpu: int = 0,
) -> None:
    """Entry point executed inside a fresh Python subprocess for CPU/GPU restart tests."""
    if "mpi4py.MPI" in sys.modules:
        assert not sys.modules["mpi4py.MPI"].Is_initialized(), (
            "Restart subprocess must never initialize MPI"
        )
    if use_gpu:
        solver, ps = _build_gpu_high_contention_case(seed_initial_state=False)
    else:
        solver, ps = _build_coupled_case(seed_initial_state=False)
    orig_restore = ckpt_mod._restore_rank_arrays
    if os.environ.get("WAKIS_TEST_MUTATE_RESTORE") == "1":

        def _mutated_worker_restore(ps_target, data):
            orig_restore(ps_target, data)
            ps_target.solver.J.toarray()[0] += 1.0

        ckpt_mod._restore_rank_arrays = _mutated_worker_restore
    try:
        ps.load_checkpoint(ckpt_dir)
    finally:
        ckpt_mod._restore_rank_arrays = orig_restore
    for _ in range(n_rem_steps):
        solver.one_step()
    snap = _extract_state_snapshot(ps)
    if "mpi4py.MPI" in sys.modules:
        assert not sys.modules["mpi4py.MPI"].Is_initialized(), (
            "Restart subprocess must never initialize MPI"
        )
    sp0 = snap["species"][0]
    col0 = snap["collectors"][0]
    arrays = {
        "E": snap["E"],
        "H": snap["H"],
        "J": snap["J"],
        "J_old": snap["J_old"],
        "J_particles": snap["J_particles"],
        "q_bg": snap["q_bg"],
        "scalars_f64": np.array(
            [
                snap["gauss_residual"],
                snap["injected_charge"],
                snap["injected_energy"],
                snap["absorbed_charge"],
                snap["absorbed_energy"],
                col0["absorbed_charge"],
                col0["absorbed_energy"],
            ],
            dtype=np.float64,
        ),
        "scalars_i64": np.array(
            [
                snap["n_particle_steps"],
                int(snap["initialized_half_step"]),
                sp0["n_particles"],
                sp0["next_id"],
                int(sp0["u_is_synced"]),
                col0["absorbed_count"],
                snap["solver_nstep"],
                int(snap["solver_step_0"]),
                int(snap["j_in_solver"]),
            ],
            dtype=np.int64,
        ),
        "b_charge": np.asarray(
            list(snap["boundary_absorbed_charge"].values()), dtype=np.float64
        ),
        "b_energy": np.asarray(
            list(snap["boundary_absorbed_energy"].values()), dtype=np.float64
        ),
        "b_count": np.asarray(
            list(snap["boundary_absorbed_count"].values()), dtype=np.int64
        ),
        "col_t": col0["t_trace"],
        "col_i": col0["i_trace"],
    }
    if snap["J_prev"] is not None:
        arrays["J_prev"] = snap["J_prev"]
    for comp in ("x", "y", "z", "ux", "uy", "uz", "w", "id"):
        arrays[f"sp_{comp}_raw"] = sp0[f"{comp}_raw"]
        arrays[f"sp_{comp}_sorted"] = sp0[f"{comp}_sorted"]
    for k, v in snap["cpml"].items():
        arrays[f"cpml_{k}"] = v
    np.savez(out_npz, **arrays)


def _load_worker_snapshot(out_npz: Path, ref_snap: dict) -> dict:
    """Reconstruct a snapshot dictionary from ``run_subprocess_restart_worker`` output."""
    with np.load(out_npz, allow_pickle=False) as d:
        s_f64 = d["scalars_f64"]
        s_i64 = d["scalars_i64"]
        b_keys = list(ref_snap["boundary_absorbed_count"].keys())
        b_charge = {k: float(v) for k, v in zip(b_keys, d["b_charge"])}
        b_energy = {k: float(v) for k, v in zip(b_keys, d["b_energy"])}
        b_count = {k: int(v) for k, v in zip(b_keys, d["b_count"])}
        cpml = {k: d[f"cpml_{k}"].copy() for k in ref_snap["cpml"]}
        sp_entry = {
            "n_particles": int(s_i64[2]),
            "next_id": int(s_i64[3]),
            "u_is_synced": bool(s_i64[4]),
        }
        for comp in ("x", "y", "z", "ux", "uy", "uz", "w", "id"):
            sp_entry[f"{comp}_raw"] = d[f"sp_{comp}_raw"].copy()
            sp_entry[f"{comp}_sorted"] = d[f"sp_{comp}_sorted"].copy()
        col_entry = {
            "absorbed_charge": float(s_f64[5]),
            "absorbed_energy": float(s_f64[6]),
            "absorbed_count": int(s_i64[5]),
            "t_trace": d["col_t"].copy(),
            "i_trace": d["col_i"].copy(),
        }
        return {
            "E": d["E"].copy(),
            "H": d["H"].copy(),
            "J": d["J"].copy(),
            "J_old": d["J_old"].copy(),
            "J_particles": d["J_particles"].copy(),
            "J_prev": d["J_prev"].copy() if "J_prev" in d.files else None,
            "j_in_solver": bool(s_i64[8]),
            "solver_step_0": bool(s_i64[7]),
            "solver_nstep": int(s_i64[6]),
            "q_bg": d["q_bg"].copy(),
            "gauss_residual": float(s_f64[0]),
            "n_particle_steps": int(s_i64[0]),
            "initialized_half_step": bool(s_i64[1]),
            "injected_charge": float(s_f64[1]),
            "injected_energy": float(s_f64[2]),
            "absorbed_charge": float(s_f64[3]),
            "absorbed_energy": float(s_f64[4]),
            "boundary_absorbed_charge": b_charge,
            "boundary_absorbed_energy": b_energy,
            "boundary_absorbed_count": b_count,
            "cpml": cpml,
            "species": [sp_entry],
            "collectors": [col_entry],
        }


# ---------------------------------------------------------------------------
# (a) CPU bitwise restart in a new Python subprocess
# ---------------------------------------------------------------------------


@_serial_only
def test_cpu_bitwise_restart_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU restart after N steps in a fresh subprocess is bitwise equal to 2N steps."""
    n_half = 4
    n_total = 2 * n_half

    # Run A: uninterrupted 2N steps
    solver_a, ps_a = _build_coupled_case(seed_initial_state=True)
    for _ in range(n_total):
        solver_a.one_step()
    snap_a = _extract_state_snapshot(ps_a)

    # Sanity-check that every feature was exercised during the run
    assert len(snap_a["cpml"]) > 0
    assert any(np.any(v != 0.0) for v in snap_a["cpml"].values())
    assert snap_a["injected_charge"] != 0.0
    assert snap_a["absorbed_charge"] != 0.0
    assert snap_a["collectors"][0]["absorbed_count"] > 0

    # Run B part 1: N steps + save_checkpoint
    ckpt_dir = tmp_path / "ckpt_cpu"
    solver_b1, ps_b1 = _build_coupled_case(seed_initial_state=True)
    for _ in range(n_half):
        solver_b1.one_step()
    ps_b1.save_checkpoint(ckpt_dir)

    # Verify manifest and allow_pickle=False loadability
    manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert manifest["rank_count"] == 1
    assert manifest["species_names"] == ["electrons"]
    assert len(manifest["injector_names"]) == 1
    with np.load(_find_rank_npz(ckpt_dir, 0), allow_pickle=False) as zf:
        assert "field_E" in zf.files

    # Run B part 2: fresh SolverFIT3D + ParticleSolver in a NEW Python process
    out_npz = tmp_path / "subproc_out.npz"
    cmd = [
        sys.executable,
        "-c",
        (
            "try:\n    import mpi4py\n    mpi4py.rc.initialize = False\n    mpi4py.rc.finalize = False\nexcept ImportError:  # mpi4py is optional\n    pass\n"
            "import tests.test_089_particles_checkpoint as t; "
            f"t.run_subprocess_restart_worker({str(ckpt_dir)!r}, {str(out_npz)!r}, {n_half})"
        ),
    ]
    subprocess.run(
        cmd,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=_clean_mpi_subprocess_env(),
    )

    snap_sub = _load_worker_snapshot(out_npz, snap_a)
    _assert_snapshots_bitwise_equal(snap_a, snap_sub)

    # Verify in-place restoration preserves existing buffer pointers (SHOULD-FIX 11)
    solver_inp, ps_inp = _build_coupled_case(seed_initial_state=False)
    solver_inp.J_old = np.zeros_like(solver_inp.J.toarray())
    ptr_before = {
        "E": solver_inp.E.toarray().__array_interface__["data"][0],
        "H": solver_inp.H.toarray().__array_interface__["data"][0],
        "J": solver_inp.J.toarray().__array_interface__["data"][0],
        "J_old": solver_inp.J_old.__array_interface__["data"][0],
        "q_bg": solver_inp._particles_q_bg.__array_interface__["data"][0],
        "J_particles": ps_inp.J_particles.__array_interface__["data"][0],
    }
    ps_inp.load_checkpoint(ckpt_dir)
    ptr_after = {
        "E": solver_inp.E.toarray().__array_interface__["data"][0],
        "H": solver_inp.H.toarray().__array_interface__["data"][0],
        "J": solver_inp.J.toarray().__array_interface__["data"][0],
        "J_old": solver_inp.J_old.__array_interface__["data"][0],
        "q_bg": solver_inp._particles_q_bg.__array_interface__["data"][0],
        "J_particles": ps_inp.J_particles.__array_interface__["data"][0],
    }
    assert ptr_after == ptr_before, (
        f"In-place restore replaced buffer pointer(s): {ptr_before} vs {ptr_after}"
    )

    # Mutation check for test (a): perturbing a restored particle coordinate
    # on load must cause the bitwise comparison to fail.
    orig_restore = ckpt_mod._restore_rank_arrays

    def _mutated_restore(ps_target, data):
        orig_restore(ps_target, data)
        ps_target.species[0]._x[0] += 1.0e-9

    monkeypatch.setattr(ckpt_mod, "_restore_rank_arrays", _mutated_restore)
    solver_mut, ps_mut = _build_coupled_case(seed_initial_state=False)
    ps_mut.load_checkpoint(ckpt_dir)
    for _ in range(n_half):
        solver_mut.one_step()
    snap_mut = _extract_state_snapshot(ps_mut)
    with pytest.raises(AssertionError) as exc_info:
        _assert_snapshots_bitwise_equal(snap_a, snap_mut)
    print(f"[mutation test_a_coord] caught AssertionError: {_fmt_exc(exc_info.value)}")


# ---------------------------------------------------------------------------
# (d) Coverage mutation parametrized over CHECKPOINT_STATE_KEYS
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpu_reference_and_half_state():
    """Cache the uninterrupted 2N-step snapshot on CPU for parametrized test (d)."""
    n_half = 4
    n_total = 2 * n_half
    solver_a, ps_a = _build_coupled_case(seed_initial_state=True)
    for _ in range(n_total):
        solver_a.one_step()
    snap_a = _extract_state_snapshot(ps_a)
    return n_half, snap_a


@_serial_only
@pytest.mark.parametrize("omitted_key", CHECKPOINT_STATE_KEYS)
def test_checkpoint_state_coverage_mutation(
    omitted_key: str,
    cpu_reference_and_half_state,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping any item in CHECKPOINT_STATE_KEYS must be detected by test (a).

    Note: some omissions are detected by direct snapshot equality (e.g. step
    counters, sync flags, or cumulative boundary/injector/collector tallies)
    rather than by changed field or particle dynamics.
    """
    if omitted_key == "J_prev":
        pytest.skip("GPU-only state; covered by test_gpu_state_coverage_mutation")

    n_half, snap_a = cpu_reference_and_half_state

    orig_save_item = ckpt_mod._save_state_item
    orig_validate = ckpt_mod._validate_rank_archive
    remaining_keys = tuple(k for k in CHECKPOINT_STATE_KEYS if k != omitted_key)

    def _skip_one_save_item(arrays, key, particle_solver):
        if key == omitted_key:
            return
        orig_save_item(arrays, key, particle_solver)

    def _validate_without_omitted(particle_solver, data, required_keys=remaining_keys):
        return orig_validate(particle_solver, data, required_keys=remaining_keys)

    monkeypatch.setattr(ckpt_mod, "_save_state_item", _skip_one_save_item)
    monkeypatch.setattr(ckpt_mod, "_validate_rank_archive", _validate_without_omitted)

    ckpt_dir = tmp_path / f"ckpt_omit_{omitted_key}"
    solver_b1, ps_b1 = _build_coupled_case(seed_initial_state=True)
    for _ in range(n_half):
        solver_b1.one_step()
    ps_b1.save_checkpoint(ckpt_dir)

    solver_b2, ps_b2 = _build_coupled_case(seed_initial_state=False)
    ps_b2.load_checkpoint(ckpt_dir)
    for _ in range(n_half):
        solver_b2.one_step()
    snap_b = _extract_state_snapshot(ps_b2)

    with pytest.raises(AssertionError) as exc_info:
        _assert_snapshots_bitwise_equal(snap_a, snap_b)
    print(
        f"[mutation omit={omitted_key}] caught AssertionError: "
        f"{_fmt_exc(exc_info.value)}"
    )


# ---------------------------------------------------------------------------
# (e) Corrupt-file, grid-mismatch, rank-count, and unsupported-feature errors
# ---------------------------------------------------------------------------


def _apply_unsupported_feature(solver, feature_name: str) -> None:
    """Activate one unsupported auxiliary state attribute on ``solver``."""
    if feature_name == "_dispersive_groups":
        solver._dispersive_groups = [object()]
    elif feature_name == "_dispersive_h_groups":
        solver._dispersive_h_groups = [object()]
    elif feature_name == "dispersive_network":
        solver.dispersive_network = object()
    elif feature_name == "_gain_groups":
        solver._gain_groups = [object()]
    elif feature_name == "_nonlinear_groups":
        solver._nonlinear_groups = [object()]
    elif feature_name == "_time_varying_groups":
        solver._time_varying_groups = [object()]
    elif feature_name == "_time_varying_h_groups":
        solver._time_varying_h_groups = [object()]
    elif feature_name == "_time_varying_installed":
        solver._time_varying_installed = True
    elif feature_name == "injection_done":
        solver.injection_done = False
    elif feature_name == "E_trans":
        solver.E_trans.toarray()[0] = 1.0
    elif feature_name == "H_trans":
        solver.H_trans.toarray()[0] = 1.0
    elif feature_name == "M":
        solver.M = solver.J
    elif feature_name == "_beams":
        solver._beams = [object()]
    elif feature_name == "tensor_operator":
        solver.tensor_operator = object()
    else:
        raise ValueError(f"Unknown feature_name: {feature_name!r}")


UNSUPPORTED_SOLVER_FEATURES: tuple[str, ...] = (
    "_dispersive_groups",
    "_dispersive_h_groups",
    "dispersive_network",
    "_gain_groups",
    "_nonlinear_groups",
    "_time_varying_groups",
    "_time_varying_h_groups",
    "_time_varying_installed",
    "injection_done",
    "E_trans",
    "H_trans",
    "M",
    "_beams",
    "tensor_operator",
)


@_serial_only
@pytest.mark.parametrize("feature_name", UNSUPPORTED_SOLVER_FEATURES)
def test_unsupported_solver_features_refuse(
    feature_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save and load must raise NotImplementedError for unsupported solver features."""
    solver_clean, ps_clean = _build_coupled_case(seed_initial_state=True)
    solver_clean.one_step()
    valid_ckpt = tmp_path / "ckpt_clean"
    ps_clean.save_checkpoint(valid_ckpt)

    # 1. save_checkpoint refuses when unsupported feature is active
    solver_s, ps_s = _build_coupled_case(seed_initial_state=True)
    _apply_unsupported_feature(solver_s, feature_name)
    with pytest.raises(NotImplementedError):
        ps_s.save_checkpoint(tmp_path / f"ckpt_save_{feature_name}")

    # 2. load_checkpoint refuses when unsupported feature is active
    solver_l, ps_l = _build_coupled_case(seed_initial_state=False)
    _apply_unsupported_feature(solver_l, feature_name)
    with pytest.raises(NotImplementedError):
        ps_l.load_checkpoint(valid_ckpt)

    # Mutation check: disabling _check_supported_solver_features allows save to
    # proceed without raising NotImplementedError.
    if feature_name == "_dispersive_groups":
        monkeypatch.setattr(
            ckpt_mod, "_check_supported_solver_features", lambda _s: None
        )
        raised_not_impl = False
        try:
            ps_s.save_checkpoint(tmp_path / "ckpt_mut_unsup")
        except NotImplementedError:
            raised_not_impl = True
        with pytest.raises(AssertionError) as exc_info:
            assert raised_not_impl, (
                "save_checkpoint did not raise NotImplementedError for _dispersive_groups"
            )
        print(
            f"[mutation unsupported_feature] caught AssertionError: "
            f"{_fmt_exc(exc_info.value)}"
        )


@_serial_only
def test_checkpoint_error_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify clear exceptions on corrupt files, grid/config mismatch, and rank mismatch."""
    solver, ps = _build_coupled_case(seed_initial_state=True)
    solver.one_step()
    ckpt_dir = tmp_path / "ckpt_valid"
    ps.save_checkpoint(ckpt_dir)

    # 1. Missing checkpoint directory / manifest
    _, ps_fresh = _build_coupled_case(seed_initial_state=False)
    with pytest.raises(FileNotFoundError, match="manifest"):
        ps_fresh.load_checkpoint(tmp_path / "nonexistent_dir")

    # 2. Rank-count mismatch -> NotImplementedError, bad rank_count -> ValueError
    ckpt_rank_err = tmp_path / "ckpt_rank_err"
    ps.save_checkpoint(ckpt_rank_err)
    m_path = ckpt_rank_err / "manifest.json"
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    manifest["rank_count"] = 4
    m_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="rank_count"):
        ps_fresh.load_checkpoint(ckpt_rank_err)

    for bad_rc in ("not_an_int", True, None, 0, -2):
        manifest["rank_count"] = bad_rc
        m_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="rank_count"):
            ps_fresh.load_checkpoint(ckpt_rank_err)

    # 3. Unattached ParticleSolver on save_checkpoint -> RuntimeError
    grid_u = GridFIT3D(-0.07, 0.07, -0.06, 0.06, 0.0, 0.16, 14, 12, 16, verbose=0)
    solver_u = SolverFIT3D(grid_u, verbose=0)
    sp_u = Species(name="electrons", charge=-e_charge, mass=m_e, capacity=16)
    ps_u = ParticleSolver(solver_u, species=[sp_u])
    assert not ps_u._attached
    with pytest.raises(RuntimeError, match="attached"):
        ps_u.save_checkpoint(tmp_path / "ckpt_unattached")
    assert not ps_u._attached

    # 4. Grid/material/CPML mismatch (grid_hash) vs particle config mismatch (config_hash)
    _, ps_diff_nx = _build_coupled_case(seed_initial_state=False, nx=16)
    with pytest.raises(ValueError, match="grid_hash"):
        ps_diff_nx.load_checkpoint(ckpt_dir)

    _, ps_diff_bounds = _build_coupled_case(seed_initial_state=False, xmax=0.09)
    with pytest.raises(ValueError, match="grid_hash"):
        ps_diff_bounds.load_checkpoint(ckpt_dir)

    _, ps_diff_dt = _build_coupled_case(seed_initial_state=False, dt_factor=0.35)
    with pytest.raises(ValueError, match="grid_hash"):
        ps_diff_dt.load_checkpoint(ckpt_dir)

    # Changed permittivity epsilon (ieps) -> grid_hash ValueError
    solver_eps, ps_eps = _build_coupled_case(seed_initial_state=False)
    solver_eps.ieps.toarray()[0] *= 2.0
    with pytest.raises(ValueError, match="grid_hash"):
        ps_eps.load_checkpoint(ckpt_dir)

    # Changed CPML coefficient array (pml_b_E_x_low) -> grid_hash ValueError (SHOULD-FIX 8)
    solver_cpml, ps_cpml = _build_coupled_case(seed_initial_state=False)
    solver_cpml.pml_b_E_x_low[0] *= 0.5
    with pytest.raises(ValueError, match="grid_hash"):
        ps_cpml.load_checkpoint(ckpt_dir)

    # Changed species charge -> config_hash ValueError (NOTE 11)
    _, ps_q = _build_coupled_case(seed_initial_state=False)
    ps_q.species[0].charge *= 2.0
    with pytest.raises(ValueError, match="config_hash"):
        ps_q.load_checkpoint(ckpt_dir)

    # Changed injector parameter -> config_hash ValueError
    _, ps_inj = _build_coupled_case(seed_initial_state=False)
    ps_inj.injectors[0].radius *= 2.0
    with pytest.raises(ValueError, match="config_hash"):
        ps_inj.load_checkpoint(ckpt_dir)

    # Changed injector current(t) callable -> config_hash ValueError (SHOULD-FIX 8)
    _, ps_curr = _build_coupled_case(seed_initial_state=False)
    ps_curr.injectors[0].current = lambda t: 5.0e-3
    with pytest.raises(ValueError, match="config_hash"):
        ps_curr.load_checkpoint(ckpt_dir)

    # Changed E_ext callable -> config_hash ValueError (SHOULD-FIX 8)
    _, ps_eext = _build_coupled_case(seed_initial_state=False)
    ps_eext.E_ext = lambda x, y, z, t: (
        np.full_like(x, 100.0),
        np.zeros_like(y),
        np.zeros_like(z),
    )
    with pytest.raises(ValueError, match="config_hash"):
        ps_eext.load_checkpoint(ckpt_dir)

    # Changed B_ext constant -> config_hash ValueError (SHOULD-FIX 8)
    _, ps_bext = _build_coupled_case(seed_initial_state=False)
    ps_bext.B_ext = (0.0, 0.0, 0.05)
    with pytest.raises(ValueError, match="config_hash"):
        ps_bext.load_checkpoint(ckpt_dir)

    # Changed callable collector mask -> config_hash ValueError (SHOULD-FIX 8)
    _, ps_cmask = _build_coupled_case(seed_initial_state=False)
    ps_cmask.boundaries.collectors[0] = Collector(
        name="anode_probe",
        mask=lambda x, y, z: z > 0.12,
    )
    with pytest.raises(ValueError, match="config_hash"):
        ps_cmask.load_checkpoint(ckpt_dir)

    # Changed capacity_headroom -> config_hash ValueError (SHOULD-FIX 8)
    _, ps_cap = _build_coupled_case(seed_initial_state=False)
    ps_cap.capacity_headroom = 2.0
    with pytest.raises(ValueError, match="config_hash"):
        ps_cap.load_checkpoint(ckpt_dir)

    # Mutation check: bypassing compute_config_hash allows loading with changed species charge.
    saved_manifest = json.loads(
        (ckpt_dir / "manifest.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        ckpt_mod, "compute_config_hash", lambda _ps: saved_manifest["config_hash"]
    )
    raised_val_err = False
    try:
        ps_q.load_checkpoint(ckpt_dir)
    except ValueError:
        raised_val_err = True
    with pytest.raises(AssertionError) as exc_info:
        assert raised_val_err, (
            "load_checkpoint did not raise ValueError for changed species charge"
        )
    print(f"[mutation config_hash] caught AssertionError: {_fmt_exc(exc_info.value)}")
    monkeypatch.setattr(ckpt_mod, "compute_config_hash", compute_config_hash)

    # 5. CPML array set mismatch on load -> ValueError
    ckpt_cpml_err = tmp_path / "ckpt_cpml_err"
    ps.save_checkpoint(ckpt_cpml_err)
    npz_cpml = _find_rank_npz(ckpt_cpml_err, 0)
    with np.load(npz_cpml, allow_pickle=False) as zf_in:
        first_cpml_key = next(k for k in zf_in.files if k.startswith("cpml_"))
        cpml_dict = {k: zf_in[k] for k in zf_in.files if k != first_cpml_key}
    np.savez(npz_cpml, **cpml_dict)
    m_cpml_path = ckpt_cpml_err / "manifest.json"
    m_cpml = json.loads(m_cpml_path.read_text(encoding="utf-8"))
    m_cpml["files"][npz_cpml.name] = hashlib.sha256(npz_cpml.read_bytes()).hexdigest()
    m_cpml_path.write_text(json.dumps(m_cpml), encoding="utf-8")
    with pytest.raises(ValueError, match="CPML array set"):
        ps_fresh.load_checkpoint(ckpt_cpml_err)

    # 6. Refusal of object dtype on save -> ValueError
    orig_collect = ckpt_mod._collect_rank_arrays

    def _collect_with_object(ps_arg):
        arrs = orig_collect(ps_arg)
        arrs["bad_obj"] = np.array([{"pickle": True}], dtype=object)
        return arrs

    monkeypatch.setattr(ckpt_mod, "_collect_rank_arrays", _collect_with_object)
    with pytest.raises(ValueError, match="object array"):
        ps.save_checkpoint(tmp_path / "ckpt_obj_err")
    monkeypatch.setattr(ckpt_mod, "_collect_rank_arrays", orig_collect)

    # 7. No partial restore: validate all keys/shapes/dtypes BEFORE modifying
    #    any solver buffer
    ckpt_missing_key = tmp_path / "ckpt_missing_key"
    ps.save_checkpoint(ckpt_missing_key)
    npz_mk = _find_rank_npz(ckpt_missing_key, 0)
    with np.load(npz_mk, allow_pickle=False) as zf_mk:
        mk_dict = {k: zf_mk[k] for k in zf_mk.files if k != "boundary_absorbed_count"}
    np.savez(npz_mk, **mk_dict)
    m_mk_path = ckpt_missing_key / "manifest.json"
    m_mk = json.loads(m_mk_path.read_text(encoding="utf-8"))
    m_mk["files"][npz_mk.name] = hashlib.sha256(npz_mk.read_bytes()).hexdigest()
    m_mk_path.write_text(json.dumps(m_mk), encoding="utf-8")

    solver_guard, ps_guard = _build_coupled_case(seed_initial_state=False)
    sentinel_e = np.full_like(solver_guard.E.toarray(), -777.0)
    solver_guard.E.toarray()[...] = sentinel_e
    with pytest.raises(ValueError, match="boundary_absorbed_count"):
        ps_guard.load_checkpoint(ckpt_missing_key)
    np.testing.assert_array_equal(
        solver_guard.E.toarray(),
        sentinel_e,
        err_msg="Partial restore modified solver.E before archive validation completed",
    )

    # 8. Atomic write of the whole checkpoint (SHOULD-FIX 3):
    #    (a) Crash during np.savez leaves previous checkpoint intact and removes *.tmp-*
    #    (b) Crash AFTER rank_000000-<token>.npz is written/replaced on disk but
    #        BEFORE manifest.json is switched via os.replace leaves the previous
    #        checkpoint intact and bitwise loadable.
    ckpt_atomic = tmp_path / "ckpt_atomic"
    ps.save_checkpoint(ckpt_atomic)
    snap_ckpt1 = _extract_state_snapshot(ps)
    rank_file_ckpt1 = _find_rank_npz(ckpt_atomic, 0)

    # Advance solver by 1 step so the second save has distinct state
    solver.one_step()
    (ckpt_atomic / "stale_orphan.tmp-deadbeef").write_bytes(b"stale")

    orig_savez = np.savez

    def _crashing_savez(file, *args, **kwargs):
        file.write(b"PARTIAL_TMP_WRITE")
        raise OSError("simulated disk full during save_checkpoint")

    monkeypatch.setattr(np, "savez", _crashing_savez)
    with pytest.raises(OSError, match="simulated disk full"):
        ps.save_checkpoint(ckpt_atomic)
    monkeypatch.setattr(np, "savez", orig_savez)
    assert list(ckpt_atomic.glob("*.tmp-*")) == []

    # Simulate crash AFTER rank_000000-<token>.npz is moved into place,
    # right before os.replace switches manifest.json.
    orig_os_replace = os.replace

    def _crash_before_manifest_switch(src, dst):
        if Path(dst).name == "manifest.json":
            raise OSError("simulated crash after rank write before manifest switch")
        return orig_os_replace(src, dst)

    monkeypatch.setattr(os, "replace", _crash_before_manifest_switch)
    with pytest.raises(OSError, match="before manifest switch"):
        ps.save_checkpoint(ckpt_atomic)
    monkeypatch.setattr(os, "replace", orig_os_replace)

    assert list(ckpt_atomic.glob("*.tmp-*")) == []
    # Both the old rank file and the newly written rank file exist on disk,
    # while manifest.json still points to rank_file_ckpt1.
    assert rank_file_ckpt1.is_file()
    assert len(list(ckpt_atomic.glob("rank_000000-*.npz"))) == 2

    _, ps_atomic_load = _build_coupled_case(seed_initial_state=False)
    ps_atomic_load.load_checkpoint(ckpt_atomic)
    _assert_snapshots_bitwise_equal(snap_ckpt1, _extract_state_snapshot(ps_atomic_load))

    # A subsequent successful save switches manifest.json and deletes older-token rank files
    ps.save_checkpoint(ckpt_atomic)
    assert list(ckpt_atomic.glob("*.tmp-*")) == []
    active_rank_files = list(ckpt_atomic.glob("rank_000000-*.npz"))
    assert len(active_rank_files) == 1
    assert active_rank_files[0] != rank_file_ckpt1

    # Mutation check for SHOULD-FIX 3: if save_checkpoint did not use the
    # per-save token scheme and instead wrote the fixed name rank_000000.npz in
    # place before crashing at the manifest switch, loading the previous
    # checkpoint fails with a SHA-256 mismatch.
    ckpt_fixed = tmp_path / "ckpt_fixed_rank_name"
    ps.save_checkpoint(ckpt_fixed)
    tok_rf = _find_rank_npz(ckpt_fixed, 0)
    fixed_rf = ckpt_fixed / "rank_000000.npz"
    orig_os_replace(tok_rf, fixed_rf)
    m_fixed_path = ckpt_fixed / "manifest.json"
    m_fixed = json.loads(m_fixed_path.read_text(encoding="utf-8"))
    m_fixed["files"] = {"rank_000000.npz": m_fixed["files"][tok_rf.name]}
    m_fixed_path.write_text(json.dumps(m_fixed), encoding="utf-8")

    solver.one_step()

    def _fixed_rank_name_and_crash_before_manifest_switch(src, dst):
        dst_path = Path(dst)
        if dst_path.name == "manifest.json":
            raise OSError(
                "simulated crash after fixed-name rank_000000.npz write before manifest switch"
            )
        if dst_path.suffix == ".npz":
            return orig_os_replace(src, dst_path.parent / "rank_000000.npz")
        return orig_os_replace(src, dst)

    monkeypatch.setattr(
        os, "replace", _fixed_rank_name_and_crash_before_manifest_switch
    )
    with pytest.raises(OSError, match="before manifest switch"):
        ps.save_checkpoint(ckpt_fixed)
    monkeypatch.setattr(os, "replace", orig_os_replace)

    with pytest.raises(ValueError, match="SHA-256 mismatch") as exc_info:
        ps_fresh.load_checkpoint(ckpt_fixed)
    print(
        f"[mutation atomic_manifest_switch] caught ValueError: "
        f"{_fmt_exc(exc_info.value)}"
    )

    # 9. Native field dtype preserved without silent casting
    grid_f32 = GridFIT3D(-0.07, 0.07, -0.06, 0.06, 0.0, 0.16, 14, 12, 16, verbose=0)
    solver_f32 = SolverFIT3D(grid_f32, dtype=np.float32, verbose=0)
    sp_f32 = Species(name="electrons", charge=-e_charge, mass=m_e, capacity=16)
    ps_f32 = ParticleSolver(solver_f32, species=[sp_f32])
    ps_f32.attach()
    solver_f32.E.toarray()[...] = np.float32(3.25)
    ckpt_f32 = tmp_path / "ckpt_f32"
    ps_f32.save_checkpoint(ckpt_f32)
    with np.load(_find_rank_npz(ckpt_f32, 0), allow_pickle=False) as zf_f32:
        assert zf_f32["field_E"].dtype == np.float32

    solver_f32_b = SolverFIT3D(grid_f32, dtype=np.float32, verbose=0)
    sp_f32_b = Species(name="electrons", charge=-e_charge, mass=m_e, capacity=16)
    ps_f32_b = ParticleSolver(solver_f32_b, species=[sp_f32_b])
    ps_f32_b.attach()
    ps_f32_b.load_checkpoint(ckpt_f32)
    assert solver_f32_b.E.toarray().dtype == np.float32
    np.testing.assert_array_equal(
        solver_f32_b.E.toarray(),
        np.full_like(solver_f32_b.E.toarray(), np.float32(3.25)),
    )
    with pytest.raises(ValueError, match="Dtype mismatch"):
        ckpt_mod._copy_into_target(
            solver_f32_b.E.toarray(),
            np.zeros(solver_f32_b.E.toarray().shape, dtype=np.float64),
        )

    # 10. Corrupt .npz bytes -> SHA-256 mismatch ValueError
    ckpt_bad_hash = tmp_path / "ckpt_bad_hash"
    ps.save_checkpoint(ckpt_bad_hash)
    npz_file = _find_rank_npz(ckpt_bad_hash, 0)
    raw = bytearray(npz_file.read_bytes())
    raw[-10] ^= 0xFF
    npz_file.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ps_fresh.load_checkpoint(ckpt_bad_hash)

    # 11. Corrupt .npz payload with matching SHA-256 -> archive corruption ValueError
    ckpt_bad_zip = tmp_path / "ckpt_bad_zip"
    ps.save_checkpoint(ckpt_bad_zip)
    npz_zip = _find_rank_npz(ckpt_bad_zip, 0)
    npz_zip.write_bytes(b"NOT_A_VALID_NPZ_ARCHIVE")
    m_zip_path = ckpt_bad_zip / "manifest.json"
    m_zip = json.loads(m_zip_path.read_text(encoding="utf-8"))
    m_zip["files"][npz_zip.name] = hashlib.sha256(npz_zip.read_bytes()).hexdigest()
    m_zip_path.write_text(json.dumps(m_zip), encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt checkpoint archive"):
        ps_fresh.load_checkpoint(ckpt_bad_zip)

    # 12. Corrupt JSON in manifest.json -> ValueError
    ckpt_bad_json = tmp_path / "ckpt_bad_json"
    ps.save_checkpoint(ckpt_bad_json)
    (ckpt_bad_json / "manifest.json").write_text("{corrupt_json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt checkpoint manifest"):
        ps_fresh.load_checkpoint(ckpt_bad_json)

    # 13. Unsupported format_version -> ValueError
    ckpt_bad_ver = tmp_path / "ckpt_bad_ver"
    ps.save_checkpoint(ckpt_bad_ver)
    m_ver_path = ckpt_bad_ver / "manifest.json"
    m_ver = json.loads(m_ver_path.read_text(encoding="utf-8"))
    m_ver["format_version"] = 999
    m_ver_path.write_text(json.dumps(m_ver), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        ps_fresh.load_checkpoint(ckpt_bad_ver)


# ---------------------------------------------------------------------------
# (b) GPU restart with high-contention multi-block atomicAdd
# ---------------------------------------------------------------------------


def _require_gpu():
    """Skip test if CuPy or a CUDA device is unavailable."""
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA GPU available")
    except Exception as exc:  # noqa: BLE001 - capture whatever each rank raises to compare classes
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp


def _build_gpu_high_contention_case(
    *,
    seed_initial_state: bool,
    n_beam_particles: int = 120_000,
):
    """Build a GPU case with >= 1e5 particles concentrated across a few cells.

    Concentrating 120,000 macroparticles (~469 CUDA blocks of 256 threads) into
    a 4 x 4 x 4 interior cell subvolume (~1,875 particles per cell) exercises
    concurrent multi-block float64 ``atomicAdd`` contention on shared grid
    edges in ``deposit_esirkepov_kernel`` and ``deposit_charge_kernel``.
    """
    solver, ps = _build_coupled_case(seed_initial_state=False, use_gpu=1)
    dt_ref = float(solver.dt)
    ps.injectors[0].current = lambda t, dt0=dt_ref: 2.5e-7 * (1.0 + 0.35 * (t / dt0))
    sp = ps.species[0]
    if seed_initial_state:
        rng = np.random.default_rng(20261030)
        e_host = rng.normal(0.0, 1.5e4, size=solver.E.toarray().shape).astype(
            np.float64
        )
        h_host = rng.normal(0.0, 4.0e1, size=solver.H.toarray().shape).astype(
            np.float64
        )
        ckpt_mod._copy_into_target(solver.E.toarray(), e_host)
        ckpt_mod._copy_into_target(solver.H.toarray(), h_host)

        # Concentrate 120,000 particles into 4 x 4 x 4 interior cells with
        # strictly positive drift velocity components so all particles in a
        # cell deposit same-sign Esirkepov edge fluxes Delta W_{p,e} > 0, plus
        # one unit-weight collector-crossing particle absorbed on step 0.
        n_bulk = n_beam_particles - 1
        x_bulk = rng.uniform(-0.015, 0.015, size=n_bulk)
        y_bulk = rng.uniform(-0.015, 0.015, size=n_bulk)
        z_bulk = rng.uniform(0.050, 0.080, size=n_bulk)
        ux_bulk = rng.uniform(1.0e7, 2.0e7, size=n_bulk)
        uy_bulk = rng.uniform(1.0e7, 2.0e7, size=n_bulk)
        uz_bulk = rng.uniform(2.5e7, 4.0e7, size=n_bulk)
        w_bulk = rng.uniform(0.5, 1.5, size=n_bulk)

        dz_cell = (0.16 - 0.0) / 16
        x_all = np.concatenate([x_bulk, [0.005]])
        y_all = np.concatenate([y_bulk, [-0.006]])
        z_all = np.concatenate([z_bulk, [0.095 - 0.02 * dz_cell]])
        ux_all = np.concatenate([ux_bulk, [0.4e7]])
        uy_all = np.concatenate([uy_bulk, [0.5e7]])
        uz_all = np.concatenate([uz_bulk, [0.45 * c_light]])
        w_all = np.concatenate([w_bulk, [1.0]])

        sp.add_particles(
            x=x_all,
            y=y_all,
            z=z_all,
            ux=ux_all,
            uy=uy_all,
            uz=uz_all,
            w=w_all,
        )
        solver.nstep = 5
    return solver, ps


def _assert_gpu_one_step_roundoff_bound(
    snap_a: dict,
    snap_b: dict,
    ps_ref: ParticleSolver,
) -> tuple[bool, float, float]:
    """Assert 1-step post-restart GPU agreement within per-edge Higham Ch. 4 (§4.2) bound.

    During a single post-restart step starting from identical checkpointed
    fields E^N and H^{N-1/2}, ``gather_and_push_kernel`` has one CUDA thread per
    active particle and no inter-thread atomics, so particle phase-space arrays
    and H^{N+1/2} are identical. In ``deposit_esirkepov_kernel``, the n_e
    particles whose 3x3x3 S1 Esirkepov stencil overlaps edge e deposit flux
    contributions Delta W_{p,e} via hardware float64 ``atomicAdd``. By Higham
    (2002) Ch. 4 (§4.2, Eq. 4.4) with gamma_{n_e} = n_e*u / (1 - n_e*u) defined
    in §3.1 (u = 2^-53), two summation orders of n_e terms on edge e differ in
    current density J_e = W_e / (dt * tA_e) by at most:
        |J_e^(A) - J_e^(B)| <= 2 * gamma_{n_e} * |J_e^(A)| + floor,
    evaluated per edge e in vectorised form.

    Replacing sum_p |dW_{p,e}| by |J_e| is valid only when every contribution to an
    edge has the same sign: true for this setup (all velocity components positive,
    non-negative S1 Esirkepov weights, no sign flip within one step). A setup with
    counter-moving particles would make this bound too tight and fail spuriously.
    """
    u_eps = float(np.finfo(np.float64).eps) * 0.5  # u = 2^-53
    n_active = int(snap_a["species"][0]["n_particles"])
    assert n_active >= 100_000, f"Expected >= 1e5 active particles, got {n_active}"

    s = ps_ref.solver
    nx, ny, nz = int(s.Nx), int(s.Ny), int(s.Nz)
    sp0 = snap_a["species"][0]
    grid = s.grid
    xmin, xmax = float(grid.x[0]), float(grid.x[-1])
    ymin, ymax = float(grid.y[0]), float(grid.y[-1])
    zmin, zmax = float(grid.z[0]), float(grid.z[-1])
    ix = np.clip(
        np.floor((sp0["x_raw"] - xmin) / ((xmax - xmin) / nx)).astype(np.int64),
        0,
        nx - 1,
    )
    iy = np.clip(
        np.floor((sp0["y_raw"] - ymin) / ((ymax - ymin) / ny)).astype(np.int64),
        0,
        ny - 1,
    )
    iz = np.clip(
        np.floor((sp0["z_raw"] - zmin) / ((zmax - zmin) / nz)).astype(np.int64),
        0,
        nz - 1,
    )
    cell_counts = np.zeros((nx, ny, nz), dtype=np.int64)
    np.add.at(cell_counts, (ix, iy, iz), 1)
    pad_c = np.pad(cell_counts, 1, mode="wrap")
    n_3x3x3 = pad_c[:-2, :, :] + pad_c[1:-1, :, :] + pad_c[2:, :, :]
    n_3x3x3 = n_3x3x3[:, :-2, :] + n_3x3x3[:, 1:-1, :] + n_3x3x3[:, 2:, :]
    n_3x3x3 = n_3x3x3[:, :, :-2] + n_3x3x3[:, :, 1:-1] + n_3x3x3[:, :, 2:]
    n_e = np.tile(np.maximum(n_3x3x3.ravel(order="F"), 1).astype(np.float64), 3)
    gamma_ne = (n_e * u_eps) / (1.0 - n_e * u_eps)

    j_part_a = snap_a["J_particles"]
    j_part_b = snap_b["J_particles"]
    j_a = snap_a["J"]
    j_b = snap_b["J"]
    floor_dj = u_eps * float(np.max(np.abs(j_part_a)))
    bound_dj_part_edge = 2.0 * gamma_ne * np.abs(j_part_a) + floor_dj
    bound_dj_edge = (
        2.0 * gamma_ne * np.maximum(np.abs(j_part_a), np.abs(j_a)) + floor_dj
    )

    dj_part_edge = np.abs(j_part_a - j_part_b)
    dj_edge = np.abs(j_a - j_b)
    max_dj_part = float(np.max(dj_part_edge))
    max_dj = float(np.max(dj_edge))
    bound_dj = float(np.max(bound_dj_part_edge))
    assert bool(np.all(dj_part_edge <= bound_dj_part_edge)), (
        f"GPU 1-step per-edge J_particles diff (max={max_dj_part:.3e}) exceeds "
        f"per-edge Higham §4.2 bound (max={bound_dj:.3e})"
    )
    assert bool(np.all(dj_edge <= bound_dj_edge)), (
        f"GPU 1-step per-edge J diff (max={max_dj:.3e}) exceeds "
        f"per-edge Higham §4.2 bound (max={float(np.max(bound_dj_edge)):.3e})"
    )

    # Induced 1-step electric field difference per edge:
    # |Delta E_e^{N+1}| <= dt * |ieps_e| * bound_dj_e + 2 * u * |E_e^{N+1}| + floor_de.
    ieps_arr = np.abs(ckpt_mod._to_host_f64(s.ieps.toarray()))
    e_a = snap_a["E"]
    e_b = snap_b["E"]
    floor_de = 2.0 * u_eps * float(np.max(np.abs(e_a)))
    bound_de_edge = (
        float(s.dt) * ieps_arr * bound_dj_edge + 2.0 * u_eps * np.abs(e_a) + floor_de
    )
    de_edge = np.abs(e_a - e_b)
    max_de = float(np.max(de_edge))
    assert bool(np.all(de_edge <= bound_de_edge)), (
        f"GPU 1-step per-edge E diff (max={max_de:.3e}) exceeds induced "
        f"per-edge bound (max={float(np.max(bound_de_edge)):.3e})"
    )

    # H^{N+1/2} is updated before _mid_step_hook_gpu on step N, so it has no
    # dependence on step-N particle atomicAdd order.
    np.testing.assert_array_equal(snap_a["H"], snap_b["H"], err_msg="GPU H mismatch")
    np.testing.assert_array_equal(
        snap_a["q_bg"], snap_b["q_bg"], err_msg="GPU q_bg mismatch"
    )
    assert snap_a["n_particle_steps"] == snap_b["n_particle_steps"]
    assert snap_a["solver_nstep"] == snap_b["solver_nstep"]
    assert snap_a["solver_step_0"] == snap_b["solver_step_0"]
    assert snap_a["initialized_half_step"] == snap_b["initialized_half_step"]
    assert snap_a["j_in_solver"] == snap_b["j_in_solver"]
    assert snap_a["injected_charge"] == snap_b["injected_charge"]
    assert snap_a["injected_energy"] == snap_b["injected_energy"]
    assert snap_a["absorbed_charge"] == snap_b["absorbed_charge"]
    assert snap_a["absorbed_energy"] == snap_b["absorbed_energy"]
    assert snap_a["boundary_absorbed_count"] == snap_b["boundary_absorbed_count"]

    for k in snap_a["cpml"]:
        np.testing.assert_array_equal(
            snap_a["cpml"][k], snap_b["cpml"][k], err_msg=f"GPU CPML {k} mismatch"
        )

    for sp_a, sp_b in zip(snap_a["species"], snap_b["species"]):
        assert sp_a["n_particles"] == sp_b["n_particles"]
        assert sp_a["next_id"] == sp_b["next_id"]
        assert sp_a["u_is_synced"] == sp_b["u_is_synced"]
        for comp in ("x", "y", "z", "ux", "uy", "uz", "w", "id"):
            np.testing.assert_array_equal(
                sp_a[f"{comp}_sorted"],
                sp_b[f"{comp}_sorted"],
                err_msg=f"GPU 1-step species {comp} mismatch",
            )

    bitwise_equal = (
        max_dj_part == 0.0
        and max_dj == 0.0
        and max_de == 0.0
        and np.array_equal(snap_a["J_old"], snap_b["J_old"])
    )
    return bitwise_equal, max_dj_part, bound_dj


@_serial_only
def test_gpu_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU restart with >= 1e5 particles agrees within the 1-step Higham §4.2 bound."""
    _require_gpu()
    n_pre = 4
    n_post = 1

    # Run B1 on GPU: n_pre steps with 120,000 particles, save_checkpoint
    # (device -> host), then step B1 ONE more step to form the reference
    # starting from the exact checkpointed state.
    ckpt_dir = tmp_path / "ckpt_gpu"
    solver_b1, ps_b1 = _build_gpu_high_contention_case(seed_initial_state=True)
    for _ in range(n_pre):
        solver_b1.one_step()
    ps_b1.save_checkpoint(ckpt_dir)
    for _ in range(n_post):
        solver_b1.one_step()
    snap_b1 = _extract_state_snapshot(ps_b1)

    # Restore B2 from the checkpoint (host -> device) and step once.
    solver_b2, ps_b2 = _build_gpu_high_contention_case(seed_initial_state=False)
    solver_b2.J_old = solver_b2.J.xp.zeros_like(solver_b2.J.toarray())
    gpu_ptrs_before = {
        "E": int(solver_b2.E.toarray().data.ptr),
        "H": int(solver_b2.H.toarray().data.ptr),
        "J": int(solver_b2.J.toarray().data.ptr),
        "J_old": int(solver_b2.J_old.data.ptr),
        "q_bg": int(solver_b2._particles_q_bg.data.ptr),
        "J_particles": int(ps_b2.J_particles.data.ptr),
        "J_prev": int(ps_b2._J_prev.data.ptr),
    }
    ps_b2.load_checkpoint(ckpt_dir)
    gpu_ptrs_after = {
        "E": int(solver_b2.E.toarray().data.ptr),
        "H": int(solver_b2.H.toarray().data.ptr),
        "J": int(solver_b2.J.toarray().data.ptr),
        "J_old": int(solver_b2.J_old.data.ptr),
        "q_bg": int(solver_b2._particles_q_bg.data.ptr),
        "J_particles": int(ps_b2.J_particles.data.ptr),
        "J_prev": int(ps_b2._J_prev.data.ptr),
    }
    assert gpu_ptrs_after == gpu_ptrs_before, (
        f"GPU in-place restore replaced device pointer(s): {gpu_ptrs_before} vs {gpu_ptrs_after}"
    )
    assert ps_b2.on_gpu
    assert hasattr(ps_b2.J_particles, "get")
    assert hasattr(ps_b2.species[0]._x, "get")

    for _ in range(n_post):
        solver_b2.one_step()
    snap_b2 = _extract_state_snapshot(ps_b2)

    bitwise_eq, max_dj, bound_dj = _assert_gpu_one_step_roundoff_bound(
        snap_b1, snap_b2, ps_b1
    )
    print(
        f"[info test_b_gpu] n_particles={snap_b1['species'][0]['n_particles']}, "
        f"bitwise_equal={bitwise_eq}, max_dJ={max_dj:.3e}, bound_dJ={bound_dj:.3e}"
    )

    # Mutation check for test (b): perturbing a restored GPU field array on load
    # must fail the GPU round-off bound check.
    orig_restore = ckpt_mod._restore_rank_arrays

    def _mutated_gpu_restore(ps_target, data):
        orig_restore(ps_target, data)
        ps_target.solver.E.toarray()[0] += 1.0

    monkeypatch.setattr(ckpt_mod, "_restore_rank_arrays", _mutated_gpu_restore)
    solver_mut, ps_mut = _build_gpu_high_contention_case(seed_initial_state=False)
    ps_mut.load_checkpoint(ckpt_dir)
    for _ in range(n_post):
        solver_mut.one_step()
    snap_mut = _extract_state_snapshot(ps_mut)
    with pytest.raises(AssertionError) as exc_info:
        _assert_gpu_one_step_roundoff_bound(snap_b1, snap_mut, ps_b1)
    print(f"[mutation test_b_gpu_E] caught AssertionError: {_fmt_exc(exc_info.value)}")


@_serial_only
def test_gpu_restart_subprocess(tmp_path: Path) -> None:
    """Fresh-process GPU restart: save in-process, load and step once in a new Python process."""
    _require_gpu()
    n_pre = 2
    n_post = 1

    ckpt_dir = tmp_path / "ckpt_gpu_subproc"
    solver_b1, ps_b1 = _build_gpu_high_contention_case(seed_initial_state=True)
    for _ in range(n_pre):
        solver_b1.one_step()
    ps_b1.save_checkpoint(ckpt_dir)
    for _ in range(n_post):
        solver_b1.one_step()
    snap_b1 = _extract_state_snapshot(ps_b1)

    out_npz = tmp_path / "gpu_subproc_out.npz"
    cmd = [
        sys.executable,
        "-c",
        (
            "try:\n    import mpi4py\n    mpi4py.rc.initialize = False\n    mpi4py.rc.finalize = False\nexcept ImportError:  # mpi4py is optional\n    pass\n"
            "import tests.test_089_particles_checkpoint as t; "
            f"t.run_subprocess_restart_worker({str(ckpt_dir)!r}, {str(out_npz)!r}, {n_post}, use_gpu=1)"
        ),
    ]
    subprocess.run(
        cmd,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=_clean_mpi_subprocess_env(),
    )

    snap_b2 = _load_worker_snapshot(out_npz, snap_b1)
    bitwise_eq, max_dj, bound_dj = _assert_gpu_one_step_roundoff_bound(
        snap_b1, snap_b2, ps_b1
    )
    print(
        f"[info test_b_gpu_subproc] n_particles={snap_b1['species'][0]['n_particles']}, "
        f"bitwise_equal={bitwise_eq}, max_dJ={max_dj:.3e}, bound_dJ={bound_dj:.3e}"
    )

    # Mutation check: mutate restore inside the child subprocess via an
    # environment variable that wraps _restore_rank_arrays in the worker.
    out_npz_mut = tmp_path / "gpu_subproc_mut_out.npz"
    cmd_mut = [
        sys.executable,
        "-c",
        (
            "try:\n    import mpi4py\n    mpi4py.rc.initialize = False\n    mpi4py.rc.finalize = False\nexcept ImportError:  # mpi4py is optional\n    pass\n"
            "import tests.test_089_particles_checkpoint as t; "
            f"t.run_subprocess_restart_worker({str(ckpt_dir)!r}, {str(out_npz_mut)!r}, {n_post}, use_gpu=1)"
        ),
    ]
    env_mut = _clean_mpi_subprocess_env()
    env_mut["WAKIS_TEST_MUTATE_RESTORE"] = "1"
    subprocess.run(
        cmd_mut,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env_mut,
    )
    snap_mut = _load_worker_snapshot(out_npz_mut, snap_b1)
    with pytest.raises(AssertionError) as exc_info:
        _assert_gpu_one_step_roundoff_bound(snap_b1, snap_mut, ps_b1)
    print(
        f"[mutation test_b_gpu_subproc_J] caught AssertionError: "
        f"{_fmt_exc(exc_info.value)}"
    )


@_serial_only
@pytest.mark.parametrize("omitted_key", ("J_particles", "J_prev", "boundary_absorbed"))
def test_gpu_state_coverage_mutation(
    omitted_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting J_particles, J_prev, or boundary_absorbed on GPU must be detected."""
    _require_gpu()
    n_pre = 4
    n_post = 1

    orig_save_item = ckpt_mod._save_state_item
    orig_validate = ckpt_mod._validate_rank_archive
    remaining_keys = tuple(k for k in CHECKPOINT_STATE_KEYS if k != omitted_key)

    def _skip_gpu_save_item(arrays, key, particle_solver):
        if key == omitted_key:
            return
        orig_save_item(arrays, key, particle_solver)

    def _validate_without_omitted(particle_solver, data, required_keys=remaining_keys):
        return orig_validate(particle_solver, data, required_keys=remaining_keys)

    monkeypatch.setattr(ckpt_mod, "_save_state_item", _skip_gpu_save_item)
    monkeypatch.setattr(ckpt_mod, "_validate_rank_archive", _validate_without_omitted)

    ckpt_dir = tmp_path / f"ckpt_gpu_omit_{omitted_key}"
    solver_b1, ps_b1 = _build_gpu_high_contention_case(seed_initial_state=True)
    for _ in range(n_pre):
        solver_b1.one_step()
    snap_mid_b1 = _extract_state_snapshot(ps_b1)
    ps_b1.save_checkpoint(ckpt_dir)
    for _ in range(n_post):
        solver_b1.one_step()
    snap_b1 = _extract_state_snapshot(ps_b1)

    solver_b2, ps_b2 = _build_gpu_high_contention_case(seed_initial_state=False)
    ps_b2.load_checkpoint(ckpt_dir)
    snap_mid_b2 = _extract_state_snapshot(ps_b2)
    for _ in range(n_post):
        solver_b2.one_step()
    snap_b2 = _extract_state_snapshot(ps_b2)

    if omitted_key == "J_prev":
        # Omitting J_prev must be detected through changed 1-step field dynamics
        # (finalize_j_delta_gpu fails to subtract J_prev^{N-1/2} from solver.J,
        # altering both solver.J and solver.E on step N), not None vs zeros.
        assert not np.allclose(snap_b1["J"], snap_b2["J"])
        assert not np.allclose(snap_b1["E"], snap_b2["E"])
        with pytest.raises(AssertionError) as exc_info:
            _assert_gpu_one_step_roundoff_bound(snap_b1, snap_b2, ps_b1)
    else:
        with pytest.raises(AssertionError) as exc_info:
            _assert_gpu_one_step_roundoff_bound(snap_b1, snap_b2, ps_b1)
            _assert_snapshots_bitwise_equal(
                snap_mid_b1, snap_mid_b2, check_raw_particle_order=False
            )
    print(
        f"[mutation gpu_omit={omitted_key}] caught AssertionError: "
        f"{_fmt_exc(exc_info.value)}"
    )


# ---------------------------------------------------------------------------
# (c) MPI at -n 2 on CPU: restart with same rank count is bitwise equal,
#     plus collective MPI error propagation without hangs
# ---------------------------------------------------------------------------


@_mpi_only
def test_mpi_cpu_bitwise_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MPI (-n >= 2) CPU restart is bitwise equal on every rank to uninterrupted MPI run."""
    MPI = _get_mpi()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size < 2:
        pytest.skip("MPI checkpoint test requires mpiexec -n >= 2")

    shared_root, _ = _mpi_tmp_dirs(tmp_path, comm)
    ckpt_dir = shared_root / "ckpt_mpi_cpu"
    n_half = 4
    n_total = 2 * n_half

    # Run A: uninterrupted 2N steps under MPI
    solver_a, ps_a = _build_coupled_case(seed_initial_state=True, use_mpi=True)
    for _ in range(n_total):
        solver_a.one_step()
    snap_a = _extract_state_snapshot(ps_a)

    # Verify bidirectional migration across the z=0.08 slab interface occurred
    mig_total = int(
        comm.allreduce(ps_a._n_migrated_lo + ps_a._n_migrated_hi, op=MPI.SUM)
    )
    assert mig_total > 0

    # Run B part 1: N steps + save_checkpoint across all MPI ranks
    solver_b1, ps_b1 = _build_coupled_case(seed_initial_state=True, use_mpi=True)
    for _ in range(n_half):
        solver_b1.one_step()
    ps_b1.save_checkpoint(ckpt_dir)

    assert (ckpt_dir / "manifest.json").is_file()
    assert _find_rank_npz(ckpt_dir, rank).is_file()
    manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rank_count"] == size
    assert len(manifest["files"]) == size

    # Run B part 2: fresh MPI SolverFIT3D + ParticleSolver loads checkpoint and runs N steps
    solver_b2, ps_b2 = _build_coupled_case(seed_initial_state=False, use_mpi=True)
    ps_b2.load_checkpoint(ckpt_dir)
    for _ in range(n_half):
        solver_b2.one_step()
    snap_b = _extract_state_snapshot(ps_b2)

    _assert_snapshots_bitwise_equal(snap_a, snap_b)

    # Mutation check for test (c): perturbing rank 1's restored E field on load
    # must be detected after N coupled MPI steps.
    orig_restore = ckpt_mod._restore_rank_arrays

    def _mutated_mpi_restore(ps_target, data):
        orig_restore(ps_target, data)
        if int(getattr(ps_target.solver, "rank", 0)) == 1:
            ps_target.solver.E[7, 6, 4, "z"] += 1.0
            if ps_target.species[0].n_particles > 0:
                ps_target.species[0]._z[0] += 1.0e-5

    monkeypatch.setattr(ckpt_mod, "_restore_rank_arrays", _mutated_mpi_restore)
    solver_mut, ps_mut = _build_coupled_case(seed_initial_state=False, use_mpi=True)
    ps_mut.load_checkpoint(ckpt_dir)
    for _ in range(n_half):
        solver_mut.one_step()
    snap_mut = _extract_state_snapshot(ps_mut)
    local_failed = 0
    err_first_line = ""
    try:
        _assert_snapshots_bitwise_equal(snap_a, snap_mut)
    except AssertionError as exc:
        local_failed = 1
        err_first_line = _fmt_exc(exc)
    any_failed = int(comm.allreduce(local_failed, op=MPI.SUM))
    if local_failed:
        print(
            f"[mutation test_c_mpi_rank{rank}] caught AssertionError: {err_first_line}"
        )
    assert any_failed >= 1
    comm.Barrier()


def _assert_all_ranks_raised_same_class(comm, caught_exc: BaseException | None) -> str:
    """Assert every MPI rank raised an exception and all ranks raised the same class."""
    cls_name = type(caught_exc).__name__ if caught_exc is not None else "None"
    all_names = comm.allgather(cls_name)
    assert all(name != "None" for name in all_names), (
        f"Not all MPI ranks raised an exception: {all_names}"
    )
    assert len(set(all_names)) == 1, (
        f"MPI ranks raised mismatched exception classes: {all_names}"
    )
    return all_names[0]


@_mpi_only
def test_mpi_collective_error_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rank-local failures on save or load propagate original exception classes collectively."""
    MPI = _get_mpi()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size < 2:
        pytest.skip("MPI collective error test requires mpiexec -n >= 2")

    shared_root, _ = _mpi_tmp_dirs(tmp_path, comm)
    solver_src, ps_src = _build_coupled_case(seed_initial_state=True, use_mpi=True)
    solver_src.one_step()

    # Case (i): delete rank 1's .npz before load -> every rank raises FileNotFoundError
    ckpt_i = shared_root / "ckpt_mpi_err_i"
    ps_src.save_checkpoint(ckpt_i)
    if rank == 1:
        _find_rank_npz(ckpt_i, 1).unlink()
    comm.Barrier()

    _, ps_load_i = _build_coupled_case(seed_initial_state=False, use_mpi=True)
    caught_i: BaseException | None = None
    try:
        ps_load_i.load_checkpoint(ckpt_i)
    except Exception as exc:  # noqa: BLE001 - capture whatever each rank raises to compare classes
        caught_i = exc
    cls_i = _assert_all_ranks_raised_same_class(comm, caught_i)
    assert cls_i == "FileNotFoundError"

    # Case (ii): corrupt rank 0's .npz before load -> every rank raises ValueError
    ckpt_ii = shared_root / "ckpt_mpi_err_ii"
    ps_src.save_checkpoint(ckpt_ii)
    if rank == 0:
        npz_0 = _find_rank_npz(ckpt_ii, 0)
        raw_0 = bytearray(npz_0.read_bytes())
        raw_0[-10] ^= 0xFF
        npz_0.write_bytes(bytes(raw_0))
    comm.Barrier()

    _, ps_load_ii = _build_coupled_case(seed_initial_state=False, use_mpi=True)
    caught_ii: BaseException | None = None
    try:
        ps_load_ii.load_checkpoint(ckpt_ii)
    except Exception as exc:  # noqa: BLE001 - capture whatever each rank raises to compare classes
        caught_ii = exc
    cls_ii = _assert_all_ranks_raised_same_class(comm, caught_ii)
    assert cls_ii == "ValueError"

    # Case (iii): monkeypatch np.savez to raise OSError on rank 1 -> every rank raises OSError
    orig_savez = np.savez

    def _failing_savez_rank1(file, *args, **kwargs):
        if rank == 1:
            raise OSError("simulated np.savez failure on rank 1")
        return orig_savez(file, *args, **kwargs)

    monkeypatch.setattr(np, "savez", _failing_savez_rank1)
    ckpt_iii = shared_root / "ckpt_mpi_err_iii"
    caught_iii: BaseException | None = None
    try:
        ps_src.save_checkpoint(ckpt_iii)
    except Exception as exc:  # noqa: BLE001 - capture whatever each rank raises to compare classes
        caught_iii = exc
    cls_iii = _assert_all_ranks_raised_same_class(comm, caught_iii)
    assert cls_iii == "OSError"
    monkeypatch.setattr(np, "savez", orig_savez)

    # Case (iv): unsupported feature on rank 1 only -> every rank raises NotImplementedError
    solver_iv, ps_iv = _build_coupled_case(seed_initial_state=True, use_mpi=True)
    if rank == 1:
        solver_iv._dispersive_groups = [object()]
    caught_iv: BaseException | None = None
    try:
        ps_iv.save_checkpoint(shared_root / "ckpt_mpi_err_iv")
    except Exception as exc:  # noqa: BLE001 - capture whatever each rank raises to compare classes
        caught_iv = exc
    cls_iv = _assert_all_ranks_raised_same_class(comm, caught_iv)
    assert cls_iv == "NotImplementedError"

    # Case (v): rank 0 raises ValueError while rank 1 raises FileNotFoundError ->
    # failing ranks disagree on exception class, so every rank raises RuntimeError
    # listing each rank's error.
    ckpt_v = shared_root / "ckpt_mpi_err_v"
    ps_src.save_checkpoint(ckpt_v)
    if rank == 0:
        npz_v0 = _find_rank_npz(ckpt_v, 0)
        raw_v0 = bytearray(npz_v0.read_bytes())
        raw_v0[-10] ^= 0xFF
        npz_v0.write_bytes(bytes(raw_v0))
    elif rank == 1:
        _find_rank_npz(ckpt_v, 1).unlink()
    comm.Barrier()

    _, ps_load_v = _build_coupled_case(seed_initial_state=False, use_mpi=True)
    caught_v: BaseException | None = None
    try:
        ps_load_v.load_checkpoint(ckpt_v)
    except Exception as exc:  # noqa: BLE001 - capture whatever each rank raises to compare classes
        caught_v = exc
    cls_v = _assert_all_ranks_raised_same_class(comm, caught_v)
    assert cls_v == "RuntimeError"
    assert "rank 0 (ValueError)" in str(caught_v)
    assert "rank 1 (FileNotFoundError)" in str(caught_v)

    # Case (vi): edit rank_count in manifest.json -> every rank raises NotImplementedError
    ckpt_vi = shared_root / "ckpt_mpi_err_vi"
    ps_src.save_checkpoint(ckpt_vi)
    if rank == 0:
        m_vi_path = ckpt_vi / "manifest.json"
        m_vi = json.loads(m_vi_path.read_text(encoding="utf-8"))
        m_vi["rank_count"] = size + 2
        m_vi_path.write_text(json.dumps(m_vi), encoding="utf-8")
    comm.Barrier()

    _, ps_load_vi = _build_coupled_case(seed_initial_state=False, use_mpi=True)
    caught_vi: BaseException | None = None
    try:
        ps_load_vi.load_checkpoint(ckpt_vi)
    except Exception as exc:  # noqa: BLE001 - capture whatever each rank raises to compare classes
        caught_vi = exc
    cls_vi = _assert_all_ranks_raised_same_class(comm, caught_vi)
    assert cls_vi == "NotImplementedError"
    assert "rank_count" in str(caught_vi)

    # Mutation check: bypassing _checkpoint_collective_check causes rank 0 to
    # not raise when rank 1's .npz is missing, failing _assert_all_ranks_raised_same_class.
    _, ps_mut = _build_coupled_case(seed_initial_state=False, use_mpi=True)

    def _local_only_check(_ps, local_exc):
        if local_exc is not None:
            raise local_exc

    monkeypatch.setattr(ckpt_mod, "_checkpoint_collective_check", _local_only_check)

    class _NoBarrierComm:
        def __init__(self, real_comm):
            self._comm = real_comm

        def allgather(self, val):
            return self._comm.allgather(val)

        def Barrier(self):
            return None

    monkeypatch.setattr(ps_mut.solver, "comm", _NoBarrierComm(comm))
    caught_mut: BaseException | None = None
    try:
        ps_mut.load_checkpoint(ckpt_i)
    except Exception as exc:  # noqa: BLE001 - capture whatever each rank raises to compare classes
        caught_mut = exc
    with pytest.raises(AssertionError) as exc_info:
        _assert_all_ranks_raised_same_class(comm, caught_mut)
    if rank == 0:
        print(
            f"[mutation test_mpi_collective_err] caught AssertionError: "
            f"{_fmt_exc(exc_info.value)}"
        )
    comm.Barrier()


@_mpi_only
def test_mpi_state_coverage_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting a checkpoint state item under MPI (-n >= 2) must be detected."""
    MPI = _get_mpi()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size < 2:
        pytest.skip("MPI omission test requires mpiexec -n >= 2")

    shared_root, _ = _mpi_tmp_dirs(tmp_path, comm)
    n_half = 4
    n_total = 2 * n_half

    solver_a, ps_a = _build_coupled_case(seed_initial_state=True, use_mpi=True)
    for _ in range(n_total):
        solver_a.one_step()
    snap_a = _extract_state_snapshot(ps_a)

    omitted_key = "field_J"
    orig_save_item = ckpt_mod._save_state_item
    orig_validate = ckpt_mod._validate_rank_archive
    remaining_keys = tuple(k for k in CHECKPOINT_STATE_KEYS if k != omitted_key)

    def _skip_mpi_save_item(arrays, key, particle_solver):
        if key == omitted_key:
            return
        orig_save_item(arrays, key, particle_solver)

    def _validate_without_omitted(particle_solver, data, required_keys=remaining_keys):
        return orig_validate(particle_solver, data, required_keys=remaining_keys)

    monkeypatch.setattr(ckpt_mod, "_save_state_item", _skip_mpi_save_item)
    monkeypatch.setattr(ckpt_mod, "_validate_rank_archive", _validate_without_omitted)

    ckpt_dir = shared_root / f"ckpt_mpi_omit_{omitted_key}"
    solver_b1, ps_b1 = _build_coupled_case(seed_initial_state=True, use_mpi=True)
    for _ in range(n_half):
        solver_b1.one_step()
    ps_b1.save_checkpoint(ckpt_dir)

    solver_b2, ps_b2 = _build_coupled_case(seed_initial_state=False, use_mpi=True)
    ps_b2.load_checkpoint(ckpt_dir)
    for _ in range(n_half):
        solver_b2.one_step()
    snap_b = _extract_state_snapshot(ps_b2)

    local_failed = 0
    err_first_line = ""
    try:
        _assert_snapshots_bitwise_equal(snap_a, snap_b)
    except AssertionError as exc:
        local_failed = 1
        err_first_line = _fmt_exc(exc)
    any_failed = int(comm.allreduce(local_failed, op=MPI.MAX))
    if local_failed and rank == 0:
        print(
            f"[mutation mpi_omit={omitted_key}] caught AssertionError: {err_first_line}"
        )
    assert any_failed == 1
    comm.Barrier()
