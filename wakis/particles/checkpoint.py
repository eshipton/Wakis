# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Checkpoint and restart for coupled electromagnetic field and particle runs.

Why ``SolverFIT3D.save_state`` / ``load_state`` are not used directly
----------------------------------------------------------------------
``SolverFIT3D.save_state`` and ``SolverFIT3D.load_state``
(``wakis/solverFIT3D.py``) write a single HDF5 file containing only ``E``,
``H``, and ``J`` gathered to rank 0. They cannot be used for exact coupled
checkpoint/restart because:

1. They use HDF5 rather than one ``.npz`` per MPI rank (read with
   ``allow_pickle=False``) plus a JSON manifest with SHA-256 integrity checks.
2. Under MPI, ``save_state`` strips MPI ghost layers via
   ``mpi_gather_asField``, losing per-rank ghost state and requiring global
   gathers/scatters rather than rank-local files.
3. They omit the 24 convolutional CPML auxiliary memory arrays
   (``psiHa_*``, ``psiHb_*``, ``psiEa_*``, ``psiEb_*``), the solver first-step
   guard ``step_0`` (which, if left ``True`` on restart, zeroes boundary/MPI
   ghost planes via ``_set_ghosts_to_0()`` and resets ``J_old``), ``nstep``,
   and conductive ``J_old``.

Note on ``_H_before_update`` / ``_H_prev``
------------------------------------------
At the beginning of every ``SolverFIT3D.one_step()`` call,
``SolverFIT3D._capture_mid_step_H()`` copies ``solver.H.toarray()``
(:math:`\\mathbf{H}^{n-1/2}`) into ``solver._H_before_update`` before
``_update_H()`` advances :math:`\\mathbf{H}` to :math:`\\mathbf{H}^{n+1/2}`.
``ParticleSolver._mid_step_hook()`` then averages ``_H_before_update`` and
``solver.H`` to form the time-centred magnetic flux density
:math:`\\mathbf{B}^n = \\frac{1}{2}\\mu_0(\\mathbf{H}^{n-1/2} + \\mathbf{H}^{n+1/2})`.
Between complete time steps, ``_H_before_update`` and ``_H_prev`` are therefore
intra-step scratch buffers overwritten before first read on the next step, not
independent inter-step state. On ``load_checkpoint``, they are synchronized
from the restored ``solver.H`` buffer just as in ``ParticleSolver.attach()``.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .._version import __version__ as wakis_version

if TYPE_CHECKING:
    from .solver import ParticleSolver

CHECKPOINT_FORMAT_VERSION: int = 1

CPML_PSI_ATTRS: tuple[str, ...] = (
    "psiHa_z_low",
    "psiHb_y_low",
    "psiEa_z_low",
    "psiEb_y_low",
    "psiHa_x_low",
    "psiHb_z_low",
    "psiEa_x_low",
    "psiEb_z_low",
    "psiHa_y_low",
    "psiHb_x_low",
    "psiEa_y_low",
    "psiEb_x_low",
    "psiHa_z_high",
    "psiHb_y_high",
    "psiEa_z_high",
    "psiEb_y_high",
    "psiHa_x_high",
    "psiHb_z_high",
    "psiEa_x_high",
    "psiEb_z_high",
    "psiHa_y_high",
    "psiHb_x_high",
    "psiEa_y_high",
    "psiEb_x_high",
)

CPML_COEFF_ATTRS: tuple[str, ...] = tuple(
    f"pml_{coeff}_{field}_{axis}_{side}"
    for side in ("low", "high")
    for axis in ("x", "y", "z")
    for coeff in ("b", "c")
    for field in ("E", "H")
)

_KNOWN_EXC_CLASSES: dict[str, type[Exception]] = {
    "NotImplementedError": NotImplementedError,
    "ValueError": ValueError,
    "FileNotFoundError": FileNotFoundError,
    "RuntimeError": RuntimeError,
    "OSError": OSError,
    "MemoryError": MemoryError,
    "KeyError": KeyError,
    "TypeError": TypeError,
}

CHECKPOINT_STATE_KEYS: tuple[str, ...] = (
    "E",
    "H",
    "field_J",
    "field_J_old",
    "solver_step_0",
    "cpml_psi",
    "J_particles",
    "j_in_solver",
    "J_prev",
    "q_bg",
    "gauss_ref",
    "species_x",
    "species_y",
    "species_z",
    "species_ux",
    "species_uy",
    "species_uz",
    "species_w",
    "species_id",
    "species_n_particles",
    "species_next_id",
    "initialized_half_step",
    "species_u_is_synced",
    "n_particle_steps",
    "solver_nstep",
    "injector_rng_state",
    "injector_counters",
    "boundary_absorbed",
    "collector_state",
)

CHECKPOINT_STATE_ITEMS: tuple[str, ...] = CHECKPOINT_STATE_KEYS


def _check_supported_solver_features(solver) -> None:
    """Raise ``NotImplementedError`` if ``solver`` has unsupported auxiliary state."""
    if bool(getattr(solver, "_dispersive_groups", None)):
        raise NotImplementedError(
            "Checkpointing is not supported with dispersive electric media (_dispersive_groups)."
        )
    if bool(getattr(solver, "_dispersive_h_groups", None)):
        raise NotImplementedError(
            "Checkpointing is not supported with dispersive magnetic media (_dispersive_h_groups)."
        )
    if getattr(solver, "dispersive_network", None) is not None:
        raise NotImplementedError(
            "Checkpointing is not supported with a conformal dispersive_network."
        )
    if bool(getattr(solver, "_gain_groups", None)):
        raise NotImplementedError(
            "Checkpointing is not supported with active gain media (_gain_groups)."
        )
    if bool(getattr(solver, "_nonlinear_groups", None)):
        raise NotImplementedError(
            "Checkpointing is not supported with nonlinear media (_nonlinear_groups)."
        )
    if (
        bool(getattr(solver, "_time_varying_groups", None))
        or bool(getattr(solver, "_time_varying_h_groups", None))
        or bool(getattr(solver, "_time_varying_installed", False))
    ):
        raise NotImplementedError(
            "Checkpointing is not supported with time-varying media (_time_varying_groups)."
        )
    if not getattr(solver, "injection_done", True):
        raise NotImplementedError(
            "Checkpointing is not supported with active TF/SF injection (injection_done=False)."
        )
    for trans_attr in ("E_trans", "H_trans"):
        trans_field = getattr(solver, trans_attr, None)
        if trans_field is not None:
            arr = trans_field.toarray()
            if bool(trans_field.xp.any(arr != 0)):
                raise NotImplementedError(
                    f"Checkpointing is not supported with non-zero TF/SF auxiliary field {trans_attr}."
                )
    if getattr(solver, "M", None) is not None:
        raise NotImplementedError(
            "Checkpointing is not supported when magnetic current density M is present."
        )
    if bool(getattr(solver, "_beams", None)):
        raise NotImplementedError(
            "Checkpointing is not supported with cold-fluid beams (_beams)."
        )
    if getattr(solver, "tensor_operator", None) is not None:
        raise NotImplementedError(
            "Checkpointing is not supported with an off-diagonal tensor_operator."
        )


def _checkpoint_collective_check(
    particle_solver: ParticleSolver,
    local_exc: BaseException | None,
) -> None:
    """Propagate per-rank checkpoint exceptions collectively across all MPI ranks.

    Allgathers ``(type(exc).__name__, str(exc))`` across ``solver.comm``. When
    all failing ranks raise the same exception class, re-raises that original
    exception class on every rank. When failing ranks disagree on the exception
    class, raises ``RuntimeError`` listing each failing rank's error.
    """
    s = particle_solver.solver
    if not (
        getattr(particle_solver, "use_mpi", False) and int(getattr(s, "size", 1)) > 1
    ):
        if local_exc is not None:
            raise local_exc
        return

    entry = (
        (type(local_exc).__name__, str(local_exc))
        if local_exc is not None
        else ("", "")
    )
    all_entries = s.comm.allgather(entry)
    failing = [
        (r, cls_name, msg) for r, (cls_name, msg) in enumerate(all_entries) if cls_name
    ]
    if not failing:
        return

    failing_classes = {cls_name for _, cls_name, _ in failing}
    details = "; ".join(f"rank {r} ({cls_name}): {msg}" for r, cls_name, msg in failing)
    if len(failing_classes) == 1:
        only_cls_name = next(iter(failing_classes))
        exc_cls = _KNOWN_EXC_CLASSES.get(only_cls_name)
        if exc_cls is not None:
            if local_exc is not None and type(local_exc).__name__ == only_cls_name:
                raise local_exc
            raise exc_cls(f"MPI checkpoint failure ({details})")

    if local_exc is not None:
        raise RuntimeError(
            f"MPI checkpoint failure across ranks ({details})"
        ) from local_exc
    raise RuntimeError(f"MPI checkpoint failure across ranks ({details})")


def _to_host_array(arr) -> np.ndarray:
    """Return a contiguous host ``numpy.ndarray`` from a CPU or CuPy array."""
    if hasattr(arr, "get"):
        return np.ascontiguousarray(arr.get())
    return np.ascontiguousarray(np.asarray(arr))


def _to_host_f64(arr) -> np.ndarray:
    """Return a contiguous ``float64`` host ``numpy.ndarray``."""
    if hasattr(arr, "get"):
        return np.ascontiguousarray(arr.get(), dtype=np.float64)
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float64))


def _copy_into_target(target, host_arr: np.ndarray) -> None:
    """Copy ``host_arr`` in place into CPU or GPU buffer ``target`` without silent casting."""
    src = np.asarray(host_arr)
    if src.dtype != target.dtype:
        raise ValueError(
            f"Dtype mismatch during in-place restore: saved array has dtype={src.dtype}, "
            f"target buffer has dtype={target.dtype}."
        )
    if src.shape != target.shape:
        raise ValueError(
            f"Shape mismatch during in-place restore: saved array has shape={src.shape}, "
            f"target buffer has shape={target.shape}."
        )
    if hasattr(target, "get"):
        import cupy as cp

        cp.copyto(target, cp.asarray(src), casting="no")
    else:
        np.copyto(target, src, casting="no")


def _species_active_count(sp) -> int:
    """Return active particle count on host without altering GPU launch state."""
    if sp.on_gpu and sp._n_dirty_gpu and sp._d_n_particles is not None:
        return int(sp._d_n_particles[0].get())
    return int(sp._n_particles)


def _injector_names(particle_solver: ParticleSolver) -> list[str]:
    """Return canonical identifier list for attached injectors."""
    names: list[str] = []
    for idx, inj in enumerate(particle_solver.injectors):
        custom = getattr(inj, "name", None)
        if custom:
            names.append(str(custom))
        else:
            z0_val = float(getattr(inj, "z0", 0.0))
            names.append(f"{inj.species.name}:z0={z0_val:.6g}:{idx}")
    return names


def _callable_qualname(fn) -> str:
    """Return a stable qualified name string for a callable."""
    return str(
        getattr(
            fn,
            "__qualname__",
            getattr(fn, "__name__", type(fn).__qualname__),
        )
    )


def _sample_points_3d(solver) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 8 deterministic interior sample points across the grid domain."""
    grid = solver.grid
    x_arr = np.asarray(grid.x, dtype=np.float64)
    y_arr = np.asarray(grid.y, dtype=np.float64)
    z_nodes = getattr(grid, "Z_nodes", None)
    z_arr = np.asarray(z_nodes if z_nodes is not None else grid.z, dtype=np.float64)
    xs = x_arr[0] + (x_arr[-1] - x_arr[0]) * np.array([0.25, 0.75])
    ys = y_arr[0] + (y_arr[-1] - y_arr[0]) * np.array([0.25, 0.75])
    zs = z_arr[0] + (z_arr[-1] - z_arr[0]) * np.array([0.25, 0.75])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return xx.ravel(), yy.ravel(), zz.ravel()


def _fingerprint_ext_field(spec, particle_solver: ParticleSolver):
    """Return a JSON-serializable fingerprint of ``E_ext`` or ``B_ext``."""
    if spec is None:
        return None
    if not callable(spec):
        arr = np.asarray(spec, dtype=np.float64).ravel()
        return {
            "kind": "constant",
            "values": [float(arr[0]), float(arr[1]), float(arr[2])],
        }
    s = particle_solver.solver
    dt = float(s.dt)
    x_h, y_h, z_h = _sample_points_3d(s)
    xp = particle_solver.xp
    x_in = xp.asarray(x_h) if particle_solver.on_gpu else x_h
    y_in = xp.asarray(y_h) if particle_solver.on_gpu else y_h
    z_in = xp.asarray(z_h) if particle_solver.on_gpu else z_h
    n_pts = x_h.size
    samples: list[list[float]] = []
    for mult in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        t_val = float(mult * dt)
        try:
            fx, fy, fz = spec(x_in, y_in, z_in, t_val)
        except (TypeError, AttributeError, ValueError, NotImplementedError):  # host-only callable given device arrays
            fx, fy, fz = spec(x_h, y_h, z_h, t_val)
        fx_h = np.broadcast_to(_to_host_f64(fx), (n_pts,))
        fy_h = np.broadcast_to(_to_host_f64(fy), (n_pts,))
        fz_h = np.broadcast_to(_to_host_f64(fz), (n_pts,))
        samples.append([float(v) for v in np.concatenate([fx_h, fy_h, fz_h])])
    return {
        "kind": "callable",
        "qualname": _callable_qualname(spec),
        "samples": samples,
    }


def _fingerprint_collector_mask(mask_fn, particle_solver: ParticleSolver):
    """Return a JSON-serializable fingerprint of a callable collector ``mask``."""
    if mask_fn is None:
        return None
    s = particle_solver.solver
    grid = s.grid
    x_arr = np.asarray(grid.x, dtype=np.float64)
    y_arr = np.asarray(grid.y, dtype=np.float64)
    z_arr = np.asarray(grid.z, dtype=np.float64)
    if x_arr.size * y_arr.size * z_arr.size <= 8192:
        xc = 0.5 * (x_arr[:-1] + x_arr[1:]) if x_arr.size > 1 else x_arr
        yc = 0.5 * (y_arr[:-1] + y_arr[1:]) if y_arr.size > 1 else y_arr
        zc = 0.5 * (z_arr[:-1] + z_arr[1:]) if z_arr.size > 1 else z_arr
    else:
        xc = np.linspace(float(x_arr[0]), float(x_arr[-1]), 8)
        yc = np.linspace(float(y_arr[0]), float(y_arr[-1]), 8)
        zc = np.linspace(float(z_arr[0]), float(z_arr[-1]), 8)
    xx, yy, zz = np.meshgrid(xc, yc, zc, indexing="ij")
    xp = particle_solver.xp
    try:
        if particle_solver.on_gpu:
            res = mask_fn(xp.asarray(xx), xp.asarray(yy), xp.asarray(zz))
        else:
            res = mask_fn(xx, yy, zz)
    except (TypeError, AttributeError, ValueError, NotImplementedError):  # host-only callable given device arrays
        res = mask_fn(xx, yy, zz)
    mask_bool = np.ascontiguousarray(_to_host_array(res), dtype=np.uint8)
    return {
        "qualname": _callable_qualname(mask_fn),
        "shape": [int(d) for d in mask_bool.shape],
        "sha256": hashlib.sha256(mask_bool.tobytes()).hexdigest(),
    }


def compute_config_hash(particle_solver: ParticleSolver) -> str:
    """Compute a deterministic SHA-256 hex digest of the particle solver configuration.

    Hashes ``shape``, ``pusher``, ``deposit_scheme``, ``gather``, ``filter``,
    ``sort_every``, ``capacity_headroom``, ``E_ext``, ``B_ext``, each species'
    ``name``, ``charge``, ``mass``, ``dtype``, and ``track_ids``, each
    injector's parameters (``z0``, ``gamma0``, ``uz0``, ``profile``, ``radius``,
    ``x0``, ``y0``, ``n_macro_per_step``, ``rise_time``, ``direction``,
    ``thermal_u``, ``current`` qualified name and multi-time samples, and
    ``weight``), and each collector's ``name``, ``bounds``, ``plane``, and
    evaluated callable ``mask``. User-supplied callables (``E_ext``, ``B_ext``,
    injector ``current``, and collector ``mask``) are called at every
    :func:`save_checkpoint` and :func:`load_checkpoint` invocation and must
    therefore be cheap to evaluate and free of side effects. Note that
    user-supplied callables cannot be inspected symbolically without
    bytecode/closure serialization; two callables with the same ``__qualname__``
    that differ only outside the sampled points and times will produce the same
    fingerprint.

    Parameters
    ----------
    particle_solver : ParticleSolver
        Coupled particle solver instance.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    dt = float(particle_solver.solver.dt)
    ps_cfg = {
        "shape": int(particle_solver.shape),
        "pusher": str(particle_solver.pusher),
        "deposit_scheme": str(particle_solver.deposit_scheme),
        "gather": str(getattr(particle_solver, "gather", "galerkin")),
        "filter": getattr(particle_solver, "filter", None),
        "sort_every": int(getattr(particle_solver, "sort_every", 0)),
        "capacity_headroom": float(getattr(particle_solver, "capacity_headroom", 0.0)),
        "E_ext": _fingerprint_ext_field(
            getattr(particle_solver, "E_ext", None), particle_solver
        ),
        "B_ext": _fingerprint_ext_field(
            getattr(particle_solver, "B_ext", None), particle_solver
        ),
        "species": [
            {
                "name": str(sp.name),
                "charge": float(sp.charge),
                "mass": float(sp.mass),
                "dtype": np.dtype(sp.dtype).str,
                "track_ids": bool(sp.track_ids),
            }
            for sp in particle_solver.species
        ],
        "injectors": [
            {
                "species": str(inj.species.name),
                "z0": float(getattr(inj, "z0", 0.0)),
                "gamma0": float(getattr(inj, "gamma0", getattr(inj, "gamma", 0.0))),
                "uz0": float(getattr(inj, "uz0", 0.0)),
                "profile": str(getattr(inj, "profile", "")),
                "radius": float(getattr(inj, "radius", 0.0)),
                "x0": float(getattr(inj, "x0", 0.0)),
                "y0": float(getattr(inj, "y0", 0.0)),
                "n_macro_per_step": int(getattr(inj, "n_macro_per_step", 0)),
                "rise_time": float(getattr(inj, "rise_time", 0.0)),
                "direction": int(getattr(inj, "direction", 1)),
                "thermal_u": float(getattr(inj, "thermal_u", 0.0)),
                "current_qualname": (
                    _callable_qualname(inj.current)
                    if callable(getattr(inj, "current", None))
                    else None
                ),
                "current_samples": (
                    [
                        float(inj.current_at(m * dt))
                        for m in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0)
                    ]
                    if hasattr(inj, "current_at")
                    else None
                ),
                "weight": (
                    None if getattr(inj, "weight", None) is None else float(inj.weight)
                ),
            }
            for inj in particle_solver.injectors
        ],
        "collectors": [
            {
                "name": str(col.name),
                "bounds": (
                    [float(b) for b in col.bounds]
                    if getattr(col, "bounds", None) is not None
                    else None
                ),
                "plane": (
                    [int(col.plane[0]), float(col.plane[1]), int(col.plane[2])]
                    if getattr(col, "plane", None) is not None
                    else None
                ),
                "mask": _fingerprint_collector_mask(
                    getattr(col, "mask", None), particle_solver
                ),
            }
            for col in particle_solver.boundaries.collectors
        ],
    }
    h.update(json.dumps(ps_cfg, sort_keys=True).encode("utf-8"))
    local_digest = h.hexdigest()
    s = particle_solver.solver
    if getattr(particle_solver, "use_mpi", False) and int(getattr(s, "size", 1)) > 1:
        all_digests = s.comm.allgather(local_digest)
        comb = hashlib.sha256()
        for d in all_digests:
            comb.update(d.encode("ascii"))
            comb.update(b":")
        return comb.hexdigest()
    return local_digest


def compute_grid_hash(
    solver,
    particle_solver: ParticleSolver | None = None,
) -> str:
    """Compute a deterministic SHA-256 hex digest of grid, materials, and CPML.

    Hashes cell counts ``(Nx, Ny, Nz, NZ)``, domain bounds, node coordinate
    vectors ``(x, y, z)``, time step ``dt``, boundary condition specifiers
    ``bc_low`` / ``bc_high``, CPML profile parameters (``n_pml``, ``kappa_max``,
    ``alpha_max``, ``sigma_factor``, ``pml_exp``) and active CPML coefficient
    arrays (``pml_b_*``, ``pml_c_*``), field ``dtype``, and material tensors
    (``ieps``, ``imu``, ``sigma`` and any solid/STL masks). Kept separate from
    :func:`compute_config_hash` so a mismatch in ``load_checkpoint`` names
    either ``grid_hash`` or ``config_hash`` specifically. Under MPI, all
    per-rank digests are combined into a single global digest across ``comm``.

    Parameters
    ----------
    solver : SolverFIT3D
        Electromagnetic solver instance.
    particle_solver : ParticleSolver, optional
        Unused; retained for signature compatibility.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    _ = particle_solver
    h = hashlib.sha256()
    grid = solver.grid
    shape_tuple = (
        int(solver.Nx),
        int(solver.Ny),
        int(solver.Nz),
        int(getattr(solver, "NZ", solver.Nz)),
    )
    h.update(f"shape:{shape_tuple};".encode())

    x_arr = np.asarray(grid.x, dtype=np.float64)
    y_arr = np.asarray(grid.y, dtype=np.float64)
    z_arr = np.asarray(grid.z, dtype=np.float64)
    bounds = (
        float(x_arr[0]),
        float(x_arr[-1]),
        float(y_arr[0]),
        float(y_arr[-1]),
        float(z_arr[0]),
        float(z_arr[-1]),
    )
    h.update(np.asarray(bounds, dtype=np.float64).tobytes())
    h.update(x_arr.tobytes())
    h.update(y_arr.tobytes())
    h.update(z_arr.tobytes())
    h.update(np.asarray([float(solver.dt)], dtype=np.float64).tobytes())

    bc_low = tuple(str(b).strip().lower() for b in getattr(solver, "bc_low", ()))
    bc_high = tuple(str(b).strip().lower() for b in getattr(solver, "bc_high", ()))
    n_pml = int(getattr(solver, "n_pml", 0))
    cpml_params = (
        bool(getattr(solver, "activate_cpml", False)),
        n_pml,
        float(getattr(solver, "kappa_max", 0.0)),
        float(getattr(solver, "alpha_max", 0.0)),
        float(getattr(solver, "sigma_factor", 0.0)),
        float(getattr(solver, "pml_exp", 0.0)),
    )
    field_dtypes = (
        np.dtype(getattr(solver, "dtype", np.float64)).str,
        solver.E.toarray().dtype.str,
        solver.H.toarray().dtype.str,
        solver.J.toarray().dtype.str,
    )
    h.update(
        f"bc_low:{bc_low};bc_high:{bc_high};cpml:{cpml_params};dtypes:{field_dtypes};".encode()
    )

    for coeff_attr in CPML_COEFF_ATTRS:
        coeff_val = getattr(solver, coeff_attr, None)
        if coeff_val is not None:
            coeff_arr = _to_host_array(coeff_val)
            h.update(
                f"cpml_coeff:{coeff_attr}:{coeff_arr.dtype.str}:{coeff_arr.shape}:".encode()
            )
            h.update(coeff_arr.tobytes())

    for mat_name in ("ieps", "imu", "sigma"):
        mat_field = getattr(solver, mat_name, None)
        if mat_field is not None:
            mat_arr = _to_host_array(mat_field.toarray())
            h.update(
                f"mat:{mat_name}:{mat_arr.dtype.str}:{mat_arr.shape}:".encode()
            )
            h.update(mat_arr.tobytes())
    h.update(
        f"use_cond:{bool(getattr(solver, 'use_conductivity', False))};"
        f"use_stl:{bool(getattr(solver, 'use_stl', False))};".encode()
    )

    for mask_attr in ("solid", "stl_mask"):
        for owner in (solver, grid):
            mask_val = getattr(owner, mask_attr, None)
            if mask_val is not None:
                mask_arr = _to_host_array(mask_val)
                h.update(f"mask:{mask_attr}:{mask_arr.dtype.str}:".encode())
                h.update(mask_arr.tobytes())

    if bool(getattr(solver, "use_stl", False)):
        stl_solids = getattr(grid, "stl_solids", None)
        stl_materials = getattr(grid, "stl_materials", None)
        h.update(
            f"stl_solids:{stl_solids!r};stl_materials:{stl_materials!r};".encode()
        )
        pv_grid = getattr(grid, "grid", None)
        cell_data = getattr(pv_grid, "cell_data", None)
        if cell_data is not None:
            for k_cd in sorted(cell_data.keys()):
                cd_arr = np.ascontiguousarray(np.asarray(cell_data[k_cd]))
                h.update(f"cd:{k_cd}:{cd_arr.dtype.str}:".encode())
                h.update(cd_arr.tobytes())

    local_digest = h.hexdigest()
    if getattr(solver, "use_mpi", False) and int(getattr(solver, "size", 1)) > 1:
        all_digests = solver.comm.allgather(local_digest)
        comb = hashlib.sha256()
        for d in all_digests:
            comb.update(d.encode("ascii"))
            comb.update(b":")
        local_digest = comb.hexdigest()

    return local_digest


def _file_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _save_state_item(
    arrays: dict[str, np.ndarray],
    key: str,
    particle_solver: ParticleSolver,
) -> None:
    """Populate ``arrays`` with the host NumPy arrays for state item ``key``."""
    s = particle_solver.solver
    if key == "E":
        arrays["field_E"] = _to_host_array(s.E.toarray())
        return

    if key == "H":
        arrays["field_H"] = _to_host_array(s.H.toarray())
        return

    if key == "field_J":
        arrays["field_J"] = _to_host_array(s.J.toarray())
        return

    if key == "field_J_old":
        if (
            getattr(s, "use_conductivity", False)
            and getattr(s, "J_old", None) is not None
        ):
            arrays["field_J_old"] = _to_host_array(s.J_old)
            arrays["has_field_J_old"] = np.array([1], dtype=np.int64)
        else:
            arrays["has_field_J_old"] = np.array([0], dtype=np.int64)
        return

    if key == "solver_step_0":
        arrays["solver_step_0"] = np.array(
            [1 if getattr(s, "step_0", False) else 0], dtype=np.int64
        )
        return

    if key == "cpml_psi":
        arrays["has_cpml_psi"] = np.array([1], dtype=np.int64)
        for attr in CPML_PSI_ATTRS:
            val = getattr(s, attr, None)
            if val is not None:
                arrays[f"cpml_{attr}"] = _to_host_array(val)
        return

    if key == "J_particles":
        arrays["J_particles"] = _to_host_array(particle_solver.J_particles)
        return

    if key == "j_in_solver":
        arrays["j_in_solver"] = np.array(
            [1 if particle_solver._j_in_solver else 0], dtype=np.int64
        )
        return

    if key == "J_prev":
        if particle_solver._J_prev is not None:
            arrays["J_prev"] = _to_host_array(particle_solver._J_prev)
            arrays["has_J_prev"] = np.array([1], dtype=np.int64)
        else:
            arrays["has_J_prev"] = np.array([0], dtype=np.int64)
        return

    if key == "q_bg":
        q_bg = getattr(s, "_particles_q_bg", None)
        if q_bg is not None:
            arrays["q_bg"] = _to_host_array(q_bg)
        return

    if key == "gauss_ref":
        q_ref_raw = getattr(s, "_particles_q_scale_ref", None)
        has_q_ref = 0 if q_ref_raw is None else 1
        if q_ref_raw is None:
            q_ref_val = 0.0
        elif hasattr(q_ref_raw, "get"):
            q_ref_val = float(q_ref_raw.get())
        else:
            q_ref_val = float(q_ref_raw)
        arrays["gauss_q_scale_ref"] = np.array([q_ref_val], dtype=np.float64)
        arrays["gauss_has_q_scale_ref"] = np.array([has_q_ref], dtype=np.int64)
        arrays["gauss_init_counts"] = np.asarray(
            particle_solver._init_gauss_counts, dtype=np.int64
        )
        return

    if key in (
        "species_x",
        "species_y",
        "species_z",
        "species_ux",
        "species_uy",
        "species_uz",
        "species_w",
    ):
        attr = key.removeprefix("species_")
        for idx, sp in enumerate(particle_solver.species):
            n_act = _species_active_count(sp)
            arrays[f"sp_{idx}_{attr}"] = _to_host_array(getattr(sp, f"_{attr}")[:n_act])
        return

    if key == "species_id":
        for idx, sp in enumerate(particle_solver.species):
            n_act = _species_active_count(sp)
            arrays[f"sp_{idx}_track_ids"] = np.array(
                [1 if sp.track_ids else 0], dtype=np.int64
            )
            if sp.track_ids and sp._id is not None:
                arrays[f"sp_{idx}_id"] = _to_host_array(sp._id[:n_act]).astype(
                    np.int64, copy=False
                )
        return

    if key == "species_n_particles":
        counts = [_species_active_count(sp) for sp in particle_solver.species]
        arrays["species_n_particles"] = np.asarray(counts, dtype=np.int64)
        return

    if key == "species_next_id":
        next_ids = [int(sp._next_id) for sp in particle_solver.species]
        arrays["species_next_id"] = np.asarray(next_ids, dtype=np.int64)
        return

    if key == "initialized_half_step":
        arrays["initialized_half_step"] = np.array(
            [1 if particle_solver._initialized_half_step else 0], dtype=np.int64
        )
        return

    if key == "species_u_is_synced":
        arrays["species_u_is_synced"] = np.asarray(
            [1 if sp.u_is_synced else 0 for sp in particle_solver.species],
            dtype=np.int64,
        )
        return

    if key == "n_particle_steps":
        arrays["n_particle_steps"] = np.array(
            [int(particle_solver.n_particle_steps)], dtype=np.int64
        )
        return

    if key == "solver_nstep":
        arrays["solver_nstep"] = np.array(
            [int(getattr(s, "nstep", particle_solver.n_particle_steps))],
            dtype=np.int64,
        )
        return

    if key == "injector_rng_state":
        arrays["has_injector_rng_state"] = np.array([1], dtype=np.int64)
        for idx, inj in enumerate(particle_solver.injectors):
            st = inj.rng.bit_generator.state
            raw_bytes = json.dumps(st, sort_keys=True).encode("utf-8")
            arrays[f"inj_{idx}_rng_json_u8"] = np.frombuffer(
                raw_bytes, dtype=np.uint8
            ).copy()
        return

    if key == "injector_counters":
        for idx, inj in enumerate(particle_solver.injectors):
            arrays[f"inj_{idx}_counters_f64"] = np.array(
                [
                    float(inj._charge_accumulator),
                    float(inj.injected_charge),
                    float(inj.injected_energy),
                ],
                dtype=np.float64,
            )
            arrays[f"inj_{idx}_injected_count"] = np.array(
                [int(inj.injected_count)], dtype=np.int64
            )
        return

    if key == "boundary_absorbed":
        b_keys = list(particle_solver.boundaries._absorbed_charge.keys())
        arrays["boundary_absorbed_charge"] = np.asarray(
            [float(particle_solver.boundaries._absorbed_charge[k]) for k in b_keys],
            dtype=np.float64,
        )
        arrays["boundary_absorbed_energy"] = np.asarray(
            [float(particle_solver.boundaries._absorbed_energy[k]) for k in b_keys],
            dtype=np.float64,
        )
        arrays["boundary_absorbed_count"] = np.asarray(
            [int(particle_solver.boundaries._absorbed_count[k]) for k in b_keys],
            dtype=np.int64,
        )
        return

    if key == "collector_state":
        arrays["has_collector_state"] = np.array([1], dtype=np.int64)
        for idx, col in enumerate(particle_solver.boundaries.collectors):
            arrays[f"col_{idx}_scalars_f64"] = np.array(
                [float(col._absorbed_charge), float(col._absorbed_energy)],
                dtype=np.float64,
            )
            arrays[f"col_{idx}_count"] = np.array(
                [int(col._absorbed_count)], dtype=np.int64
            )
            if col.current_history:
                hist = np.asarray(col.current_history, dtype=np.float64).reshape(-1, 2)
            else:
                hist = np.empty((0, 2), dtype=np.float64)
            arrays[f"col_{idx}_history"] = hist
        return

    raise KeyError(f"Unknown checkpoint state key: {key!r}")


def _collect_rank_arrays(
    particle_solver: ParticleSolver,
) -> dict[str, np.ndarray]:
    """Collect all per-rank checkpoint arrays into a host dictionary."""
    arrays: dict[str, np.ndarray] = {
        "format_version": np.array([CHECKPOINT_FORMAT_VERSION], dtype=np.int64),
    }
    for key in CHECKPOINT_STATE_KEYS:
        _save_state_item(arrays, key, particle_solver)

    return arrays


def _read_npz_headers(
    data,
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Read ``(shape, dtype)`` from ``.npy`` headers in ``data.zip`` without loading array payloads."""
    headers: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    zf = getattr(data, "zip", None)
    if zf is not None:
        namelist = set(zf.namelist())
        for k_name in data.files:
            member = f"{k_name}.npy" if f"{k_name}.npy" in namelist else k_name
            with zf.open(member, "r") as fp:
                version = np.lib.format.read_magic(fp)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fp)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fp)
                else:
                    shape, _fortran, dtype = np.lib.format._read_array_header(
                        fp, version
                    )
            headers[k_name] = (tuple(int(d) for d in shape), np.dtype(dtype))
    else:
        for k_name in data.files:
            arr = data[k_name]
            headers[k_name] = (tuple(arr.shape), np.dtype(arr.dtype))
    return headers


def _validate_rank_archive(
    particle_solver: ParticleSolver,
    data,
    required_keys: tuple[str, ...] = CHECKPOINT_STATE_KEYS,
) -> None:
    """Validate that all required ``.npz`` keys exist with matching shapes and dtypes.

    Reads ``.npy`` headers directly from the zip entries (``np.lib.format.read_magic``
    and ``read_array_header_*``) before :func:`_restore_rank_arrays` so that a
    malformed or incomplete ``.npz`` archive raises ``ValueError`` before any
    solver or particle buffer is modified and without loading array payloads
    multiple times.
    """
    s = particle_solver.solver
    if not hasattr(data, "files"):
        raise ValueError("Archive does not expose '.files'.")
    headers = _read_npz_headers(data)
    files = set(headers.keys())

    for k_name, (_sh, dt) in headers.items():
        if dt == object or dt.hasobject:
            raise ValueError(
                f"Checkpoint archive contains disallowed object array {k_name!r}."
            )

    def _require_header(
        name: str,
        expected_shape: tuple[int, ...] | None,
        expected_dtype,
    ) -> tuple[tuple[int, ...], np.dtype]:
        if name not in headers:
            raise ValueError(f"Checkpoint archive is missing required array {name!r}.")
        sh, dt = headers[name]
        exp_dt = np.dtype(expected_dtype)
        if dt != exp_dt:
            raise ValueError(
                f"Checkpoint array {name!r} has dtype={dt}, expected {exp_dt}."
            )
        if expected_shape is not None and sh != expected_shape:
            raise ValueError(
                f"Checkpoint array {name!r} has shape={sh}, expected {expected_shape}."
            )
        return sh, dt

    _require_header("format_version", (1,), np.int64)
    ver_arr = data["format_version"]
    if int(ver_arr[0]) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Archive format_version={int(ver_arr[0])} does not match {CHECKPOINT_FORMAT_VERSION}."
        )

    req_set = set(required_keys)

    if "E" in req_set:
        _require_header("field_E", s.E.toarray().shape, s.E.toarray().dtype)
    if "H" in req_set:
        _require_header("field_H", s.H.toarray().shape, s.H.toarray().dtype)
    if "field_J" in req_set:
        _require_header("field_J", s.J.toarray().shape, s.J.toarray().dtype)
    if "field_J_old" in req_set:
        _require_header("has_field_J_old", (1,), np.int64)
        has_j_old_arr = data["has_field_J_old"]
        if bool(int(has_j_old_arr[0])) and getattr(s, "use_conductivity", False):
            _require_header("field_J_old", s.J.toarray().shape, s.J.toarray().dtype)
    if "solver_step_0" in req_set:
        _require_header("solver_step_0", (1,), np.int64)

    if "cpml_psi" in req_set:
        _require_header("has_cpml_psi", (1,), np.int64)
        expected_cpml = {
            f"cpml_{attr}": getattr(s, attr)
            for attr in CPML_PSI_ATTRS
            if getattr(s, attr, None) is not None
        }
        saved_cpml_keys = {k for k in files if k.startswith("cpml_")}
        if saved_cpml_keys != set(expected_cpml.keys()):
            raise ValueError(
                f"Checkpoint CPML array set {sorted(saved_cpml_keys)} does not match "
                f"active solver CPML array set {sorted(expected_cpml.keys())}."
            )
        for npz_key, target_psi in expected_cpml.items():
            _require_header(npz_key, target_psi.shape, target_psi.dtype)

    if "J_particles" in req_set:
        _require_header(
            "J_particles",
            particle_solver.J_particles.shape,
            particle_solver.J_particles.dtype,
        )
    if "j_in_solver" in req_set:
        _require_header("j_in_solver", (1,), np.int64)
    if "J_prev" in req_set:
        _require_header("has_J_prev", (1,), np.int64)
        has_prev_arr = data["has_J_prev"]
        if bool(int(has_prev_arr[0])) and particle_solver._J_prev is not None:
            _require_header(
                "J_prev",
                particle_solver._J_prev.shape,
                particle_solver._J_prev.dtype,
            )

    if "q_bg" in req_set:
        q_bg_target = getattr(s, "_particles_q_bg", None)
        exp_q_dtype = q_bg_target.dtype if q_bg_target is not None else np.float64
        _require_header("q_bg", (int(s.Nx), int(s.Ny), int(s.Nz)), exp_q_dtype)

    if "gauss_ref" in req_set:
        _require_header("gauss_q_scale_ref", (1,), np.float64)
        _require_header("gauss_has_q_scale_ref", (1,), np.int64)
        _require_header("gauss_init_counts", (len(particle_solver.species),), np.int64)

    counts: list[int] | None = None
    if "species_n_particles" in req_set:
        _require_header(
            "species_n_particles", (len(particle_solver.species),), np.int64
        )
        cnt_arr = data["species_n_particles"]
        counts = [int(c) for c in cnt_arr]
        if any(c < 0 for c in counts):
            raise ValueError(
                f"Negative particle count in species_n_particles: {counts}"
            )
    elif "species_n_particles" in files:
        counts = [int(c) for c in data["species_n_particles"]]

    for comp_key in (
        "species_x",
        "species_y",
        "species_z",
        "species_ux",
        "species_uy",
        "species_uz",
        "species_w",
    ):
        if comp_key in req_set:
            attr = comp_key.removeprefix("species_")
            for idx, sp in enumerate(particle_solver.species):
                exp_shape = (counts[idx],) if counts is not None else None
                _require_header(f"sp_{idx}_{attr}", exp_shape, sp.dtype)

    if "species_id" in req_set:
        for idx, sp in enumerate(particle_solver.species):
            _require_header(f"sp_{idx}_track_ids", (1,), np.int64)
            tr_arr = data[f"sp_{idx}_track_ids"]
            if bool(int(tr_arr[0])) != bool(sp.track_ids):
                raise ValueError(
                    f"Species {sp.name!r} track_ids mismatch in checkpoint."
                )
            if sp.track_ids:
                exp_shape = (counts[idx],) if counts is not None else None
                _require_header(f"sp_{idx}_id", exp_shape, np.int64)

    if "species_next_id" in req_set:
        _require_header("species_next_id", (len(particle_solver.species),), np.int64)
    if "initialized_half_step" in req_set:
        _require_header("initialized_half_step", (1,), np.int64)
    if "species_u_is_synced" in req_set:
        _require_header(
            "species_u_is_synced", (len(particle_solver.species),), np.int64
        )
    if "n_particle_steps" in req_set:
        _require_header("n_particle_steps", (1,), np.int64)
    if "solver_nstep" in req_set:
        _require_header("solver_nstep", (1,), np.int64)

    if "injector_rng_state" in req_set:
        _require_header("has_injector_rng_state", (1,), np.int64)
        for idx in range(len(particle_solver.injectors)):
            rng_sh, _ = _require_header(f"inj_{idx}_rng_json_u8", None, np.uint8)
            if len(rng_sh) != 1:
                raise ValueError(
                    f"inj_{idx}_rng_json_u8 must be 1D, got shape={rng_sh}."
                )
            rng_u8 = data[f"inj_{idx}_rng_json_u8"]
            st_obj = json.loads(
                bytes(np.asarray(rng_u8, dtype=np.uint8)).decode("utf-8")
            )
            if not isinstance(st_obj, dict) or "bit_generator" not in st_obj:
                raise ValueError(
                    f"Invalid BitGenerator state in inj_{idx}_rng_json_u8."
                )

    if "injector_counters" in req_set:
        for idx in range(len(particle_solver.injectors)):
            _require_header(f"inj_{idx}_counters_f64", (3,), np.float64)
            _require_header(f"inj_{idx}_injected_count", (1,), np.int64)

    if "boundary_absorbed" in req_set:
        n_bkeys = len(particle_solver.boundaries._absorbed_charge)
        _require_header("boundary_absorbed_charge", (n_bkeys,), np.float64)
        _require_header("boundary_absorbed_energy", (n_bkeys,), np.float64)
        _require_header("boundary_absorbed_count", (n_bkeys,), np.int64)

    if "collector_state" in req_set:
        _require_header("has_collector_state", (1,), np.int64)
        for idx in range(len(particle_solver.boundaries.collectors)):
            _require_header(f"col_{idx}_scalars_f64", (2,), np.float64)
            _require_header(f"col_{idx}_count", (1,), np.int64)
            hist_sh, _ = _require_header(f"col_{idx}_history", None, np.float64)
            if len(hist_sh) != 2 or hist_sh[1] != 2:
                raise ValueError(
                    f"col_{idx}_history must have shape (N, 2), got {hist_sh}."
                )


def _restore_rank_arrays(
    particle_solver: ParticleSolver,
    data,
) -> None:
    """Restore per-rank state in place from validated ``.npz`` archive ``data``."""
    s = particle_solver.solver
    files = set(data.files)

    if "solver_step_0" in files:
        saved_step_0 = bool(int(data["solver_step_0"][0]))
        if not saved_step_0 and getattr(s, "step_0", False):
            s.step_0 = False
            if hasattr(s, "_attrcleanup") and hasattr(s, "L"):
                s._attrcleanup()
            if getattr(s, "J_old", None) is None:
                s.J_old = s.J.xp.zeros_like(s.J.toarray())
        else:
            s.step_0 = saved_step_0

    if "field_J" in files:
        _copy_into_target(s.J.toarray(), data["field_J"])

    if "field_J_old" in files and getattr(s, "use_conductivity", False):
        if getattr(s, "J_old", None) is None:
            s.J_old = s.J.xp.zeros_like(s.J.toarray())
        _copy_into_target(s.J_old, data["field_J_old"])

    if "field_E" in files:
        _copy_into_target(s.E.toarray(), data["field_E"])

    if "field_H" in files:
        _copy_into_target(s.H.toarray(), data["field_H"])
        h_curr = s.H.toarray()
        if getattr(s, "_H_before_update", None) is not None:
            s.H.xp.copyto(s._H_before_update, h_curr)
        if particle_solver._H_prev is not None:
            particle_solver.xp.copyto(particle_solver._H_prev, h_curr)

    if "has_cpml_psi" in files:
        expected_cpml_keys = {
            f"cpml_{attr}"
            for attr in CPML_PSI_ATTRS
            if getattr(s, attr, None) is not None
        }
        saved_cpml_keys = {k for k in files if k.startswith("cpml_")}
        if saved_cpml_keys != expected_cpml_keys:
            raise ValueError(
                f"Checkpoint CPML array set {sorted(saved_cpml_keys)} does not match "
                f"active solver CPML array set {sorted(expected_cpml_keys)}."
            )
        for attr in CPML_PSI_ATTRS:
            npz_key = f"cpml_{attr}"
            if npz_key in expected_cpml_keys:
                _copy_into_target(getattr(s, attr), data[npz_key])

    if "J_particles" in files:
        _copy_into_target(particle_solver.J_particles, data["J_particles"])

    if "j_in_solver" in files:
        particle_solver._j_in_solver = bool(int(data["j_in_solver"][0]))

    if "J_prev" in files and particle_solver._J_prev is not None:
        _copy_into_target(particle_solver._J_prev, data["J_prev"])

    if "q_bg" in files:
        q_bg_arr = data["q_bg"]
        q_bg_existing = getattr(s, "_particles_q_bg", None)
        if (
            q_bg_existing is not None
            and q_bg_existing.shape == q_bg_arr.shape
            and q_bg_existing.dtype == q_bg_arr.dtype
        ):
            _copy_into_target(q_bg_existing, q_bg_arr)
        else:
            q_bg_host = np.asfortranarray(q_bg_arr)
            if particle_solver.on_gpu:
                s._particles_q_bg = particle_solver.xp.asarray(
                    q_bg_host, dtype=q_bg_host.dtype, order="F"
                )
            else:
                s._particles_q_bg = q_bg_host.copy(order="F")

    if "gauss_q_scale_ref" in files:
        has_ref = bool(int(data["gauss_has_q_scale_ref"][0]))
        s._particles_q_scale_ref = (
            float(data["gauss_q_scale_ref"][0]) if has_ref else None
        )
        particle_solver._init_gauss_counts = tuple(
            int(c) for c in data["gauss_init_counts"]
        )

    if "species_n_particles" in files:
        counts = [int(c) for c in data["species_n_particles"]]
        for idx, (sp, n_act) in enumerate(zip(particle_solver.species, counts)):
            if n_act > sp._capacity:
                sp._ensure_capacity(n_act)
                if sp.on_gpu:
                    sp._ensure_gpu_scratch()
            sp.n_particles = n_act
            for attr in ("x", "y", "z", "ux", "uy", "uz", "w"):
                k_arr = f"sp_{idx}_{attr}"
                buf = getattr(sp, f"_{attr}")
                if k_arr in files:
                    if n_act > 0:
                        _copy_into_target(buf[:n_act], data[k_arr])
                elif n_act > 0:
                    buf[:n_act].fill(0)
            if sp.track_ids and sp._id is not None:
                k_id = f"sp_{idx}_id"
                if k_id in files:
                    if n_act > 0:
                        _copy_into_target(sp._id[:n_act], data[k_id])
                elif n_act > 0:
                    sp._id[:n_act].fill(0)
    else:
        for sp in particle_solver.species:
            sp.n_particles = 0

    if "species_next_id" in files:
        for sp, nxt in zip(particle_solver.species, data["species_next_id"]):
            sp._next_id = int(nxt)

    if "initialized_half_step" in files:
        particle_solver._initialized_half_step = bool(
            int(data["initialized_half_step"][0])
        )

    if "species_u_is_synced" in files:
        for sp, u_sync in zip(particle_solver.species, data["species_u_is_synced"]):
            sp.u_is_synced = bool(int(u_sync))

    if "n_particle_steps" in files:
        particle_solver.n_particle_steps = int(data["n_particle_steps"][0])

    if "solver_nstep" in files:
        s.nstep = int(data["solver_nstep"][0])

    if "has_injector_rng_state" in files:
        for idx, inj in enumerate(particle_solver.injectors):
            k_rng = f"inj_{idx}_rng_json_u8"
            if k_rng in files:
                raw_bytes = bytes(np.asarray(data[k_rng], dtype=np.uint8))
                inj.rng.bit_generator.state = json.loads(raw_bytes.decode("utf-8"))

    for idx, inj in enumerate(particle_solver.injectors):
        k_cnt = f"inj_{idx}_counters_f64"
        if k_cnt in files:
            c_f64 = data[k_cnt]
            inj._charge_accumulator = float(c_f64[0])
            inj.injected_charge = float(c_f64[1])
            inj.injected_energy = float(c_f64[2])
            inj.injected_count = int(data[f"inj_{idx}_injected_count"][0])

    if "boundary_absorbed_charge" in files:
        b_keys = list(particle_solver.boundaries._absorbed_charge.keys())
        for k, q_v, e_v, c_v in zip(
            b_keys,
            data["boundary_absorbed_charge"],
            data["boundary_absorbed_energy"],
            data["boundary_absorbed_count"],
        ):
            particle_solver.boundaries._absorbed_charge[k] = float(q_v)
            particle_solver.boundaries._absorbed_energy[k] = float(e_v)
            particle_solver.boundaries._absorbed_count[k] = int(c_v)
        if particle_solver.on_gpu:
            particle_solver.boundaries._d_absorbed_charge.fill(0.0)
            particle_solver.boundaries._d_absorbed_energy.fill(0.0)
            particle_solver.boundaries._d_absorbed_count.fill(0)
            particle_solver.boundaries._gpu_stats_dirty = False

    if "has_collector_state" in files:
        for idx, col in enumerate(particle_solver.boundaries.collectors):
            sc = data[f"col_{idx}_scalars_f64"]
            col._absorbed_charge = float(sc[0])
            col._absorbed_energy = float(sc[1])
            col._absorbed_count = int(data[f"col_{idx}_count"][0])
            hist_arr = np.asarray(data[f"col_{idx}_history"], dtype=np.float64).reshape(
                -1, 2
            )
            col.current_history = [(float(row[0]), float(row[1])) for row in hist_arr]


def _cleanup_stale_tmp_files(ckpt_dir: Path) -> None:
    """Remove any leftover ``*.tmp-*`` files in ``ckpt_dir``."""
    if not ckpt_dir.is_dir():
        return
    for p in ckpt_dir.glob("*.tmp-*"):
        if p.is_file():
            p.unlink(missing_ok=True)


def _resolve_rank_filename_from_manifest(
    files_map: dict,
    rank: int,
) -> str:
    """Return the rank archive filename in ``files_map`` for ``rank``."""
    legacy_name = f"rank_{rank:06d}.npz"
    if legacy_name in files_map:
        return legacy_name
    prefix = f"rank_{rank:06d}-"
    matches = [
        k
        for k in files_map
        if isinstance(k, str) and k.startswith(prefix) and k.endswith(".npz")
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Checkpoint manifest is missing a unique file entry for rank {rank:06d}."
    )


def save_checkpoint(
    particle_solver: ParticleSolver,
    path: str | os.PathLike,
) -> Path:
    """Save coupled electromagnetic field and particle state to ``path``.

    Requires ``particle_solver`` to be attached (``ps.attach()`` already called).
    Writes one ``rank_<rank:06d>-<token>.npz`` archive per MPI rank (compatible
    with ``np.load(..., allow_pickle=False)``; no ``pickle`` is used anywhere)
    and a ``manifest.json`` file recording format and Wakis versions, grid and
    config hashes, rank count, species and injector names, and per-file SHA-256
    digests. Each rank writes ``rank_<rank:06d>-<token>.npz.tmp-<token>`` and
    renames it to ``rank_<rank:06d>-<token>.npz`` without overwriting any
    previous checkpoint's rank files; rank 0 then atomically switches
    ``manifest.json.tmp-<token>`` to ``manifest.json`` last via ``os.replace``
    and deletes rank files with older tokens. A crash before the manifest switch
    leaves the previous checkpoint intact (a crash between the rank rename and
    the manifest switch leaves the unreferenced new-token rank files on disk
    until the next successful save).

    Parameters
    ----------
    particle_solver : ParticleSolver
        Attached coupled particle solver.
    path : str or os.PathLike
        Directory path where the checkpoint files are written.

    Returns
    -------
    Path
        Resolved checkpoint directory path.

    Raises
    ------
    RuntimeError
        If ``particle_solver`` is not yet attached, or if MPI ranks fail with
        mismatched exception types.
    NotImplementedError
        If unsupported solver auxiliary features are active.
    ValueError
        If any collected state array has ``object`` dtype.
    """
    s = particle_solver.solver
    use_mpi = bool(particle_solver.use_mpi and int(getattr(s, "size", 1)) > 1)
    rank = int(getattr(s, "rank", 0)) if use_mpi else 0
    size = int(getattr(s, "size", 1)) if use_mpi else 1

    ckpt_dir = Path(path)
    local_exc: BaseException | None = None
    try:
        if not particle_solver._attached:
            raise RuntimeError(
                "ParticleSolver must be attached (call ps.attach()) before save_checkpoint()."
            )
        _check_supported_solver_features(s)
        if particle_solver.on_gpu and getattr(
            particle_solver.boundaries, "_gpu_stats_dirty", False
        ):
            particle_solver.boundaries._sync_gpu_stats()
        if rank == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _cleanup_stale_tmp_files(ckpt_dir)
    except Exception as exc:  # noqa: BLE001 - any local failure must reach the collective check so no rank hangs
        local_exc = exc
    _checkpoint_collective_check(particle_solver, local_exc)

    token = secrets.token_hex(6) if rank == 0 else ""
    if use_mpi:
        token = str(s.comm.bcast(token, root=0))
        s.comm.Barrier()

    grid_hash = compute_grid_hash(s)
    config_hash = compute_config_hash(particle_solver)
    rank_filename = f"rank_{rank:06d}-{token}.npz"
    rank_file = ckpt_dir / rank_filename
    tmp_rank_file = ckpt_dir / f"{rank_filename}.tmp-{token}"
    tmp_manifest_path = ckpt_dir / f"manifest.json.tmp-{token}"
    file_hash = ""

    local_exc = None
    try:
        arrays = _collect_rank_arrays(particle_solver)
        for arr_key, arr_val in arrays.items():
            if arr_val.dtype == object or arr_val.dtype.hasobject:
                raise ValueError(
                    f"Refusing to save object array {arr_key!r} (dtype={arr_val.dtype}); "
                    "checkpoints must not use pickle."
                )
        with tmp_rank_file.open("wb") as fh:
            np.savez(fh, **arrays)

        # Verify immediately that the written archive is readable without pickle
        with np.load(tmp_rank_file, allow_pickle=False) as verify_zf:
            hdrs = _read_npz_headers(verify_zf)
            if "format_version" not in hdrs:
                raise ValueError(
                    f"Written checkpoint archive {tmp_rank_file} is missing 'format_version'."
                )
        file_hash = _file_sha256(tmp_rank_file)
        os.replace(tmp_rank_file, rank_file)
    except Exception as exc:  # noqa: BLE001 - any local failure must reach the collective check so no rank hangs
        local_exc = exc
        for p_clean in (tmp_rank_file, rank_file):
            if p_clean.exists():
                p_clean.unlink(missing_ok=True)
    try:
        _checkpoint_collective_check(particle_solver, local_exc)
    except BaseException:
        for p_clean in (tmp_rank_file, rank_file):
            if p_clean.exists():
                p_clean.unlink(missing_ok=True)
        raise

    if use_mpi:
        file_entries = s.comm.gather((rank_filename, file_hash), root=0)
    else:
        file_entries = [(rank_filename, file_hash)]

    local_exc = None
    files_map: dict[str, str] = {}
    try:
        if rank == 0:
            files_map = {fname: fhash for fname, fhash in file_entries}
            manifest = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "wakis_version": str(wakis_version),
                "token": token,
                "grid_hash": grid_hash,
                "config_hash": config_hash,
                "rank_count": size,
                "species_names": [sp.name for sp in particle_solver.species],
                "injector_names": _injector_names(particle_solver),
                "collector_names": [
                    col.name for col in particle_solver.boundaries.collectors
                ],
                "state_keys": list(CHECKPOINT_STATE_KEYS),
                "files": files_map,
            }
            tmp_manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:  # noqa: BLE001 - any local failure must reach the collective check so no rank hangs
        local_exc = exc
        if tmp_manifest_path.exists():
            tmp_manifest_path.unlink(missing_ok=True)
    try:
        _checkpoint_collective_check(particle_solver, local_exc)
    except BaseException:
        if rank_file.exists():
            rank_file.unlink(missing_ok=True)
        if tmp_manifest_path.exists():
            tmp_manifest_path.unlink(missing_ok=True)
        raise

    # Switch manifest last so the checkpoint transition is atomic; any crash
    # prior to this os.replace leaves the previous manifest.json and its
    # rank_<r:06d>-<old_token>.npz files intact.
    local_exc = None
    try:
        if rank == 0:
            os.replace(tmp_manifest_path, ckpt_dir / "manifest.json")
    except Exception as exc:  # noqa: BLE001 - any local failure must reach the collective check so no rank hangs
        local_exc = exc
        if tmp_manifest_path.exists():
            tmp_manifest_path.unlink(missing_ok=True)
    _checkpoint_collective_check(particle_solver, local_exc)

    if use_mpi:
        s.comm.Barrier()

    if rank == 0:
        active_files = set(files_map.keys())
        for old_npz in ckpt_dir.glob("rank_*.npz"):
            if old_npz.name not in active_files and old_npz.is_file():
                old_npz.unlink(missing_ok=True)
        _cleanup_stale_tmp_files(ckpt_dir)

    if use_mpi:
        s.comm.Barrier()

    return ckpt_dir


def load_checkpoint(
    particle_solver: ParticleSolver,
    path: str | os.PathLike,
) -> None:
    """Restore coupled electromagnetic field and particle state from ``path``.

    Checks supported solver features and calls ``particle_solver.attach()`` if
    not yet attached, then verifies ``manifest.json``, format version, rank
    count, grid and config hashes, species/injector/collector names, per-file
    SHA-256 digests, and all required ``.npz`` array keys, shapes, and dtypes
    (read from ``.npy`` headers) before copying any checkpoint data into solver
    buffers.

    Parameters
    ----------
    particle_solver : ParticleSolver
        Target particle solver coupled to a :class:`~wakis.SolverFIT3D`.
    path : str or os.PathLike
        Checkpoint directory previously created by :func:`save_checkpoint`.

    Raises
    ------
    FileNotFoundError
        If ``manifest.json`` or a rank ``.npz`` file does not exist.
    NotImplementedError
        If the checkpoint rank count differs from the current solver rank count
        or if unsupported solver auxiliary features are active.
    ValueError
        If the manifest is malformed, the format version is unsupported, the
        grid/material/particle configuration hash does not match, a file
        SHA-256 digest does not match, or an ``.npz`` archive is corrupt or
        missing required keys/shapes/dtypes.
    RuntimeError
        Under MPI when failing ranks raise different exception classes.
    """
    s = particle_solver.solver
    local_exc: BaseException | None = None
    try:
        _check_supported_solver_features(s)
    except Exception as exc:  # noqa: BLE001 - any local failure must reach the collective check so no rank hangs
        local_exc = exc
    _checkpoint_collective_check(particle_solver, local_exc)

    if not particle_solver._attached:
        particle_solver.attach()

    use_mpi = bool(particle_solver.use_mpi and int(getattr(s, "size", 1)) > 1)
    rank = int(getattr(s, "rank", 0)) if use_mpi else 0
    size = int(getattr(s, "size", 1)) if use_mpi else 1

    current_grid_hash = compute_grid_hash(s)
    current_config_hash = compute_config_hash(particle_solver)
    ckpt_dir = Path(path)
    manifest_path = ckpt_dir / "manifest.json"
    rank_file: Path | None = None

    local_exc = None
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint manifest not found at {manifest_path}."
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise ValueError(
                f"Corrupt checkpoint manifest {manifest_path}: {exc}"
            ) from exc

        if not isinstance(manifest, dict):
            raise ValueError(  # noqa: TRY004 - a malformed manifest is corrupt data: documented ValueError
                f"Checkpoint manifest {manifest_path} must be a JSON object."
            )

        required_fields = (
            "format_version",
            "wakis_version",
            "grid_hash",
            "config_hash",
            "rank_count",
            "species_names",
            "injector_names",
            "files",
        )
        for field in required_fields:
            if field not in manifest:
                raise ValueError(
                    f"Checkpoint manifest {manifest_path} is missing required field {field!r}."
                )

        fmt_ver = manifest["format_version"]
        if isinstance(fmt_ver, bool) or fmt_ver != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format_version={fmt_ver!r}; "
                f"expected {CHECKPOINT_FORMAT_VERSION}."
            )

        raw_rank_count = manifest["rank_count"]
        if isinstance(raw_rank_count, bool) or not isinstance(
            raw_rank_count, (int, str)
        ):
            raise ValueError(  # noqa: TRY004 - a malformed manifest is corrupt data: documented ValueError
                f"Invalid rank_count={raw_rank_count!r} in checkpoint manifest; "
                "expected a positive integer."
            )
        try:
            saved_ranks = int(raw_rank_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid rank_count={raw_rank_count!r} in checkpoint manifest: {exc}"
            ) from exc
        if saved_ranks < 1:
            raise ValueError(
                f"Invalid rank_count={saved_ranks} in checkpoint manifest; must be >= 1."
            )

        if saved_ranks != size:
            raise NotImplementedError(
                f"Checkpoint was saved with rank_count={saved_ranks}, but current "
                f"solver has rank_count={size}. Restarting on a different rank "
                "count is not supported."
            )

        if manifest["grid_hash"] != current_grid_hash:
            raise ValueError(
                "Checkpoint grid_hash mismatch: the target grid shape, bounds, "
                "boundary conditions, materials, CPML parameters/coefficients, "
                "or dt do not match the saved checkpoint."
            )

        if manifest["config_hash"] != current_config_hash:
            raise ValueError(
                "Checkpoint config_hash mismatch: particle solver, species, "
                "injector, collector, or external field configuration does not "
                "match the saved checkpoint."
            )

        cur_species = [sp.name for sp in particle_solver.species]
        if list(manifest["species_names"]) != cur_species:
            raise ValueError(
                f"Checkpoint species_names {manifest['species_names']!r} do not "
                f"match current solver species {cur_species!r}."
            )

        cur_injectors = _injector_names(particle_solver)
        if list(manifest["injector_names"]) != cur_injectors:
            raise ValueError(
                f"Checkpoint injector_names {manifest['injector_names']!r} do not "
                f"match current solver injectors {cur_injectors!r}."
            )

        if "collector_names" in manifest:
            cur_collectors = [col.name for col in particle_solver.boundaries.collectors]
            if list(manifest["collector_names"]) != cur_collectors:
                raise ValueError(
                    f"Checkpoint collector_names {manifest['collector_names']!r} "
                    f"do not match current solver collectors {cur_collectors!r}."
                )

        files_map = manifest["files"]
        if not isinstance(files_map, dict) or len(files_map) != size:
            raise ValueError(
                f"Checkpoint manifest 'files' map is invalid for rank_count={size}."
            )

        rank_filename = _resolve_rank_filename_from_manifest(files_map, rank)
        rank_file = ckpt_dir / rank_filename

        if not rank_file.is_file():
            raise FileNotFoundError(f"Checkpoint rank file not found at {rank_file}.")

        actual_hash = _file_sha256(rank_file)
        expected_hash = str(files_map[rank_filename])
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for checkpoint file {rank_file}: "
                f"expected {expected_hash}, got {actual_hash}."
            )
    except Exception as exc:  # noqa: BLE001 - any local failure must reach the collective check so no rank hangs
        local_exc = exc
    _checkpoint_collective_check(particle_solver, local_exc)

    local_exc = None
    try:
        assert rank_file is not None
        try:
            with np.load(rank_file, allow_pickle=False) as data:
                _validate_rank_archive(particle_solver, data)
                _restore_rank_arrays(particle_solver, data)
        except Exception as exc:
            raise ValueError(f"Corrupt checkpoint archive {rank_file}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - any local failure must reach the collective check so no rank hangs
        local_exc = exc
    _checkpoint_collective_check(particle_solver, local_exc)

    if use_mpi:
        s.comm.Barrier()
