# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Resource estimates (peak memory, run time) for Wakis simulations, and a
preflight check against the target machine before launching a run.

All coefficients were measured with `benchmarks/calibrate_memory.py` on a GCP
g2-standard-16 VM (8-core Xeon @ 2.2 GHz, NVIDIA L4), each point in a fresh
process (peak RSS ``ru_maxrss`` after the setup and after 20 warmed-up steps):

* plain grids: vacuum box, PEC or CPML walls, float64/float32, CPU and GPU,
  32^3 to 160^3 cells;
* conformal grids (``conformal='pec'``, ``'all'``, ``'tensor'``): a sphere in
  a box, 32^3 to 160^3 cells; the setup (geometry, merged faces, Lanczos time
  step) has its own peak, and the per-step cost grows with the number of
  interface cells;
* dispersive edges (Drude/Lorentz poles) and the unstructured tetrahedral
  solver (`estimate_unstructured`).

Held-out sizes (not used in the fits) are reproduced within the errors
quoted in docs/performance.md. Monitors and adjoint storage are added
analytically. A safety factor (default 1.25) is applied to the memory.
"""

import argparse
from dataclasses import dataclass, field

import numpy as np

# Peak bytes = fixed + per_cell * cells  (host RAM; GPU memory for use_gpu)
_HOST = {  # (backend, cpml, dtype) -> (fixed bytes, bytes per cell)
    ("cpu", False, "float64"): (392e6, 1109.0),
    ("cpu", False, "float32"): (393e6, 813.0),
    ("cpu", True, "float64"): (428e6, 953.0),
    ("cpu", True, "float32"): (412e6, 601.0),
    ("gpu", False, "float64"): (583e6, 1081.0),
    ("gpu", False, "float32"): (538e6, 797.0),
    ("gpu", True, "float64"): (587e6, 940.0),
    ("gpu", True, "float32"): (566e6, 605.0),
}
_GPU = {
    (False, "float64"): 938.0,
    (False, "float32"): 714.0,
    (True, "float64"): 736.0,
    (True, "float32"): 457.0,
}
# Conformal grids (host RAM, float64; sphere in a box, 32^3 to 160^3 cells):
#   setup peak: fixed + per_cell * cells + per_interface * interface_cells
#   steady state: plain + per_cell * cells + per_interface * interface_cells
# With 1 to 4 Mcells the setup peak (geometry, merged faces, Lanczos time
# step) is the larger one.
_CONFORMAL_SETUP = {  # mode -> (fixed, bytes per cell, bytes per interface cell)
    "pec": (398e6, 1229.0, 9663.0),
    "all": (420e6, 1339.0, 1500.0),
    "tensor": (424e6, 1453.0, 0.0),
}
_CONFORMAL_RUN = {  # mode -> (bytes per cell, bytes per interface cell)
    "pec": (111.0, 10910.0),
    "all": (191.0, 4895.0),
    "tensor": (311.0, 3137.0),
}
_CONFORMAL_GPU_HOST = 290.0  # host bytes per cell added to the setup peak (GPU)
_CONFORMAL_GPU = {  # mode -> GPU bytes per cell on top of the plain solver
    "pec": 142.0,  # as 'all' (not measured separately)
    "all": 142.0,
    "tensor": 154.0,
}
# Dispersive edges (ADE state and per-step copies): steady-state bytes per
# edge, and the transient of `add_dispersive_material` on top of the steady
# solver (about 0.7 x its setup peak), which sets the peak for large fills
_DISPERSIVE_BYTES = 88.0
_DISPERSIVE_TRANSIENT = 160.0
_STEADY_FRACTION = 0.7

# Sustained throughput [cells * steps / s] of warmed-up float64 runs versus
# cell count, interpolated in log(cells) (constant beyond the ends). One CPU
# core (SciPy sparse), NVIDIA L4; 'mkl' from docs/performance.md (8 cores,
# not re-measured). Plain CPU grids slow down between 2.6 and 3 Mcells.
_THROUGHPUT = {  # (backend, cpml) -> ((cells, Mcells/s), ...)
    ("cpu", False): ((3.59e4, 25.5), (2.75e5, 19.4), (9.13e5, 16.1),
                     (2.15e6, 14.7), (4.17e6, 8.3)),
    ("cpu", True): ((3.59e4, 6.5), (2.75e5, 7.1), (9.13e5, 6.0),
                    (2.15e6, 7.3), (4.17e6, 6.8)),
    ("gpu", False): ((3.59e4, 204.0), (2.75e5, 738.0), (9.13e5, 418.0),
                     (2.15e6, 333.0), (4.17e6, 328.0)),
    ("gpu", True): ((2.15e6, 190.0), (4.17e6, 160.0)),
    ("mkl", False): ((2.62e5, 29.6), (2.10e6, 17.1), (7.08e6, 16.3)),
}  # fmt: skip
_THROUGHPUT[("mkl", True)] = _THROUGHPUT[("cpu", True)]
# Conformal grids on the CPU (sphere, all modes alike within the scatter)
_THROUGHPUT_CONFORMAL = ((3.59e4, 20.8), (2.75e5, 18.1), (9.13e5, 15.3),
                         (2.15e6, 13.7), (4.17e6, 13.5))  # fmt: skip
_CONFORMAL_GPU_STEP = {"pec": 1.15, "all": 1.15, "tensor": 1.18}  # x plain GPU
_FLOAT32_STEP = {"cpu": 0.62, "mkl": 0.62, "gpu": 0.60}  # time vs float64
_GPU_CPML_FLOOR = 7.1e-3  # s per step: CPML kernel launches (up to ~1.4 Mcells)
# Fused GPU step (SolverFIT3D(fused=True), studies/gpu-perf, L4, float64,
# 3D boxes): bandwidth-bound, no launch floor beyond ~20 us with CUDA graphs
_THROUGHPUT_FUSED = {  # cpml -> ((cells, Mcells/s), ...)
    False: ((2.75e5, 610.0), (2.15e6, 468.0)),
    True: ((2.75e5, 404.0),),
}
_GPU_FUSED_FLOOR = 2e-5  # s per step
_GPU_FUSED_CPML_BYTES = 380.0  # extra GPU bytes per cell (merged rows, psi)
_DISPERSIVE_STEP = (12e-9, 25e-9)  # s per dispersive edge per step: a + b n_poles
# Setup time [s] = a + b * cells + c * interface cells (grid with STL masks +
# solver; the conformal setup runs on the host also for GPU runs; measured
# with a 28 800-triangle sphere: the STL masks scale with the triangles)
_SETUP = {
    "plain": (0.0, 1.8e-6, 0.0),
    "pec": (0.03, 7.2e-6, 0.0),
    "all": (0.44, 4.3e-6, 2.2e-4),
    "tensor": (0.51, 4.8e-6, 2.6e-4),
}

_SETUP_GPU = {  # use_gpu: the stable time step runs on the GPU
    "plain": (0.05, 2.05e-6, 0.0),
    "pec": (0.03, 7.2e-6, 0.0),  # not measured: as on the CPU
    "all": (0.0, 3.7e-6, 1.8e-4),
    "tensor": (0.0, 3.5e-6, 2.4e-4),
}

# Unstructured tetrahedral FIT: `import wakis` + mesh, solver and stepping
# (ball_mesh + UnstructuredFIT3D, 2e4 to 4e5 tetrahedra)
_UNSTRUCTURED_BASE = 396e6  # bytes
_UNSTRUCTURED_PER_TET = 2523.0  # bytes
_UNSTRUCTURED_STEP = 52e-9  # s per tetrahedron per step

# Target machines: usable host RAM and GPU memory [bytes]
MACHINES = {
    "laptop": dict(ram=2e9, gpu=0),  # memguard cap for local runs
    "n2-standard-4": dict(ram=16e9, gpu=0),
    "n2-standard-8": dict(ram=32e9, gpu=0),
    "n2-standard-16": dict(ram=64e9, gpu=0),
    "n2-highmem-8": dict(ram=64e9, gpu=0),
    "n2-highmem-16": dict(ram=128e9, gpu=0),
    "g2-standard-16": dict(ram=64e9, gpu=24e9),
    "g2-standard-32": dict(ram=128e9, gpu=24e9),
}
_OS_RESERVE = 0.85  # fraction of RAM a job may use (OS, Python, page cache)


@dataclass
class Estimate:
    """
    Estimated resources of a simulation.

    Attributes
    ----------
    cells : int
        Grid cells (or tetrahedra).
    host_bytes, gpu_bytes : float
        Peak host RAM and GPU memory [bytes], including the safety factor.
    seconds : float
        Total wall time [s]: setup plus time stepping.
    breakdown : dict
        Memory contributions [bytes] (without the safety factor).
    setup_seconds, step_seconds : float
        Setup (grid, solver, conformal geometry and time step) and time per
        step [s].
    """

    cells: int
    host_bytes: float
    gpu_bytes: float
    seconds: float
    breakdown: dict = field(default_factory=dict)
    setup_seconds: float = 0.0
    step_seconds: float = 0.0

    def __str__(self):
        gb = 1e9
        lines = [
            f"cells: {self.cells:,}",
            f"peak host RAM: {self.host_bytes / gb:.2f} GB",
            f"peak GPU memory: {self.gpu_bytes / gb:.2f} GB",
            f"run time: {self.seconds / 60:.1f} min (setup {self.setup_seconds:.0f} s, "
            f"{self.step_seconds * 1e3:.2f} ms/step)",
        ]
        lines += [f"  {k}: {v / gb:.3f} GB" for k, v in self.breakdown.items()]
        return "\n".join(lines)


def interface_cells_from_area(area, cell_size, conformal="all"):
    """
    Typical number of interface cells of a conformal grid.

    Parameters
    ----------
    area : float
        Surface area of the solids [m^2].
    cell_size : float
        Cell size [m] (geometric mean of dx, dy, dz).
    conformal : {'pec', 'all', 'tensor'}
        Conformal mode.

    Returns
    -------
    int
        Cells holding an E edge whose length (PEC) or dual cell
        (dielectric) is cut by a surface: 1.1 area / cell_size^2 for PEC
        solids and 1.9 area / cell_size^2 for dielectrics (measured on
        spheres of 10 to 50 cells radius).
    """
    factor = 1.1 if conformal == "pec" else 1.9
    return int(factor * area / cell_size**2)


def _rate(table, cells):
    """Throughput [cells / s] interpolated in log(cells) from a table."""
    table = np.array(table)
    return np.interp(np.log(cells), np.log(table[:, 0]), table[:, 1]) * 1e6


def estimate(
    cells,
    steps=0,
    dtype="float64",
    backend="cpu",
    cpml=False,
    monitors=(),
    adjoint=None,
    dispersive_edges=0,
    n_poles=1,
    n_ranks=1,
    safety=1.25,
    conformal=False,
    interface_cells=None,
    fused=False,
):
    """
    Estimate peak memory and run time of a Wakis simulation.

    Parameters
    ----------
    cells : int or tuple of 3 int
        Number of grid cells, or (Nx, Ny, Nz).
    steps : int, optional
        Number of time steps (for the run time; adjoint runs count double).
    dtype : {'float64', 'float32', 'complex128', 'complex64'}, optional
        Field dtype; complex (Bloch) fields cost about twice the real ones.
    backend : {'cpu', 'mkl', 'gpu'}, optional
    cpml : bool, optional
        Whether any boundary is CPML.
    monitors : sequence of (n_frequencies, n_samples, n_components)
        DFT-type monitors; n_samples is the number of grid samples stored per
        component (e.g. the plane area in cells), complex128 accumulators.
    adjoint : dict, optional
        {'design_edges': int, 'store_bytes': 4 or 8}: stored E increments of an
        AdjointSimulation (steps x design edges) plus the adjoint state.
        Optional 'memory': 'checkpoint' with 'budget_gb' (or 'checkpoints'
        and 'block'): binomial checkpointing (`wakis.inverse.memory`) instead
        of storing; the breakdown then holds the snapshots and one block of
        rows, and the run time the recomputed steps (``adjoint_plan``).
    dispersive_edges, n_poles : int, optional
        E edges with dispersive materials (about 3 per cell inside the
        material) and poles per material.
    n_ranks : int, optional
        MPI ranks: memory and time are per rank.
    safety : float, optional
        Multiplier on the memory estimate.
    conformal : {False, True, 'pec', 'all', 'tensor'}, optional
        Conformal mode of the grid (`GridFIT3D(conformal=...)`; True is
        'all'). Adds the setup peak (geometry, merged faces, stable time
        step) and the per-step cost of the interface cells.
    interface_cells : int, optional
        Interface cells (see `interface_cells_from_area`); default
        ``2 * cells**(2/3)`` (a sphere of 0.6 times the domain width).
    fused : bool, optional
        GPU only: `SolverFIT3D(fused=True)` time step (measured throughput of
        the fused kernels instead of the CuPy path, no CPML launch floor; the
        CPML step keeps ~380 B per cell of extra GPU memory).

    Returns
    -------
    Estimate
    """
    if not isinstance(cells, (int, np.integer)):
        nx, ny, nz = cells
        cells = int(nx) * int(ny) * int(nz)
    cells = int(cells)
    mode = "all" if conformal is True else conformal
    if mode not in (False, None, "pec", "all", "tensor"):
        raise ValueError("conformal must be False, True, 'pec', 'all' or 'tensor'.")
    if mode and interface_cells is None:
        interface_cells = 2.0 * cells ** (2 / 3)
    n_iface = float(interface_cells or 0) / n_ranks
    complex_fields = dtype.startswith("complex")
    base_dtype = {"complex128": "float64", "complex64": "float32"}.get(dtype, dtype)
    per_rank = cells / n_ranks
    fixed, per_cell = _HOST[
        ("gpu" if backend == "gpu" else "cpu", bool(cpml), base_dtype)
    ]
    factor = 2.0 if complex_fields else 1.0
    breakdown = {"solver (host)": fixed + factor * per_cell * per_rank}
    gpu = (
        factor * _GPU[(bool(cpml), base_dtype)] * per_rank if backend == "gpu" else 0.0
    )
    if backend == "gpu" and fused and cpml and base_dtype == "float64":
        gpu += factor * _GPU_FUSED_CPML_BYTES * per_rank
    setup_peak = 0.0
    if mode:
        c_cell, c_iface = _CONFORMAL_RUN[mode]
        breakdown["conformal (host)"] = factor * (c_cell * per_rank + c_iface * n_iface)
        s_fixed, s_cell, s_iface = _CONFORMAL_SETUP[mode]
        if backend == "gpu":
            # Setup on the host, plus the CUDA runtime and host staging
            s_fixed += fixed - _HOST[("cpu", bool(cpml), base_dtype)][0]
            s_cell += _CONFORMAL_GPU_HOST
        setup_peak = s_fixed + factor * (s_cell * per_rank + s_iface * n_iface)
        if backend == "gpu":
            gpu += factor * _CONFORMAL_GPU[mode] * per_rank

    mon = sum(16.0 * nf * ns * nc for nf, ns, nc in monitors) / n_ranks
    breakdown["monitors"] = mon
    recompute = 0
    if adjoint:
        stored = (
            steps * adjoint["design_edges"] * adjoint.get("store_bytes", 8) / n_ranks
        )
        if adjoint.get("memory") == "checkpoint":
            plan = adjoint_plan(cells // n_ranks, steps, adjoint["design_edges"] // n_ranks,
                                adjoint.get("budget_gb"), dtype=dtype, cpml=cpml,
                                checkpoints=adjoint.get("checkpoints"),
                                block=adjoint.get("block"))  # fmt: skip
            stored, recompute = plan["bytes"], plan["recompute_steps"]
        state = (
            factor * per_cell * per_rank * 0.6
        )  # transposed operators + adjoint fields
        breakdown["adjoint stored increments"] = stored
        breakdown["adjoint state"] = state
    edges = dispersive_edges / n_ranks
    disp = _DISPERSIVE_BYTES * edges * factor
    breakdown["dispersive"] = disp

    host = sum(breakdown.values())
    if setup_peak > host:
        breakdown["conformal setup peak (host)"] = setup_peak
    # Filling many dispersive edges peaks above the steady solver
    disp_peak = (
        _STEADY_FRACTION * breakdown["solver (host)"]
        + _DISPERSIVE_TRANSIENT * edges * factor
    )
    host = max(host, setup_peak, disp_peak)
    if backend == "gpu":
        gpu += (
            mon
            + disp
            + (
                breakdown.get("adjoint state", 0)
                + breakdown.get("adjoint stored increments", 0)
            )
        )

    # Run time: setup (on the host) + steps
    work = per_rank * factor
    table = _THROUGHPUT[(backend, bool(cpml))]
    if mode and backend != "gpu" and not cpml:
        table = _THROUGHPUT_CONFORMAL
    step = work / _rate(table, work)
    if mode and backend == "gpu":
        step *= _CONFORMAL_GPU_STEP[mode]
    if base_dtype == "float32":
        step *= _FLOAT32_STEP[backend]
    if backend == "gpu" and fused and base_dtype == "float64":
        step = work / _rate(_THROUGHPUT_FUSED[bool(cpml)], work)
        if mode:
            step *= _CONFORMAL_GPU_STEP[mode]
        step = max(step, _GPU_FUSED_FLOOR)
    elif backend == "gpu" and cpml:
        step = max(step, _GPU_CPML_FLOOR)
    da, db = _DISPERSIVE_STEP
    step += edges * (da + db * n_poles) * factor
    sa, sb, sc = (_SETUP_GPU if backend == "gpu" else _SETUP)[mode or "plain"]
    setup = sa + sb * per_rank + sc * n_iface
    time_steps = steps * (2.2 if adjoint else 1.0) + recompute
    seconds = setup + time_steps * step
    return Estimate(cells, host * safety, gpu * safety, seconds, breakdown, setup, step)


def adjoint_plan(cells, steps, design_edges, budget_gb=None, dtype="float64",
                 cpml=True, checkpoints=None, block=None):  # fmt: skip
    """
    Checkpoint plan of an `AdjointSimulation` (memory='checkpoint').

    A solver snapshot holds E, H and J (9 values per cell; complex fields
    twice that) plus the CPML psi arrays (~10 % with CPML); a row holds one
    float64 value per design edge (complex: two).

    Parameters
    ----------
    cells, steps, design_edges : int
    budget_gb : float, optional
        Memory for snapshots and rows [GB]; the plan minimizes recomputation.
    checkpoints, block : int, optional
        Fixed settings instead of a budget.

    Returns
    -------
    dict
        mode, slots, block, bytes (snapshots + rows), recompute_steps,
        state_bytes, row_bytes.
    """
    from .inverse import memory as mem

    item = 16 if str(dtype).startswith("complex") else 8
    state = int(9 * item * cells * (1.1 if cpml else 1.0))
    row = item * int(design_edges)
    if checkpoints:
        if block is None:
            block = max(1, -(-(steps - 1) // mem.beta(checkpoints, 2)))
        n_blocks = -(-(steps - 1) // block)
        plan = dict(mode="checkpoint", slots=checkpoints, block=block,
                    bytes=checkpoints * state + (block + 1) * row,
                    recompute_steps=mem.revolve_cost(n_blocks - 1, checkpoints) * block)  # fmt: skip
    else:
        if budget_gb is None:
            raise ValueError("Give budget_gb or checkpoints.")
        plan = mem.plan_checkpoints(steps, state, row, budget_gb * 1e9)
    return dict(plan, state_bytes=state, row_bytes=row)


def estimate_unstructured(n_cells, steps=0, dim=3, eigen=False, safety=1.25):
    """
    Estimate peak memory and run time of an unstructured (simplex) FIT run.

    Time domain: ``import wakis`` (0.40 GB) plus 2.5 kB per tetrahedron for
    mesh, solver and stepping, and 52 ns per tetrahedron and step (measured,
    see the module docstring); an eigen-solve adds the sparse-LU fill-in
    ``EIG_COEF * n**EIG_EXP`` of `wakis.unstructured.estimate`. Triangles
    (dim=2) use `wakis.unstructured.estimate_memory` and half the step cost.

    Parameters
    ----------
    n_cells : int or SimplexMesh
        Tetrahedra (triangles for dim=2), or the mesh.
    steps : int, optional
        Leapfrog steps.
    dim : {2, 3}, optional
    eigen : bool, optional
        Include a shift-invert eigen-solve.
    safety : float, optional
        Multiplier on the memory estimate.

    Returns
    -------
    Estimate
        Without the mesh generation time (not modelled).
    """
    from .unstructured import estimate as unstructured

    if hasattr(n_cells, "n_cells"):
        dim = n_cells.dim
        n = int(n_cells.n_cells)
    else:
        n = int(n_cells)
    if dim == 2:
        mem = unstructured.estimate_memory(n, dim=2, eigen=eigen)
        step = 0.5 * _UNSTRUCTURED_STEP * n
    else:
        mem = _UNSTRUCTURED_BASE + _UNSTRUCTURED_PER_TET * n
        if eigen:
            mem += unstructured.EIG_COEF * n**unstructured.EIG_EXP
        step = _UNSTRUCTURED_STEP * n
    breakdown = {"unstructured (host)": mem}
    return Estimate(n, mem * safety, 0.0, steps * step, breakdown, 0.0, step)


def preflight(est, machine, raise_error=True):
    """
    Check an Estimate against a machine from `MACHINES` (or a dict with 'ram'
    and 'gpu' in bytes). Returns a list of problems (empty when it fits) and
    raises MemoryError when `raise_error` and it does not fit.
    """
    spec = MACHINES[machine] if isinstance(machine, str) else machine
    problems = []
    if est.host_bytes > _OS_RESERVE * spec["ram"]:
        problems.append(
            f"host RAM {est.host_bytes / 1e9:.1f} GB > {_OS_RESERVE:.0%} of "
            f"{spec['ram'] / 1e9:.0f} GB on {machine}"
        )
    if est.gpu_bytes > 0.9 * spec.get("gpu", 0):
        problems.append(
            f"GPU memory {est.gpu_bytes / 1e9:.1f} GB > 90% of "
            f"{spec.get('gpu', 0) / 1e9:.0f} GB on {machine}"
        )
    if problems and raise_error:
        raise MemoryError(
            "Simulation does not fit: " + "; ".join(problems) + f"\n{est}"
        )
    return problems


def main():
    ap = argparse.ArgumentParser(
        description="Estimate Wakis memory/run time and check a target machine."
    )
    ap.add_argument("--shape", required=True, help="Nx,Ny,Nz")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "mkl", "gpu"])
    ap.add_argument("--cpml", action="store_true")
    ap.add_argument("--conformal", default=None, choices=["pec", "all", "tensor"])
    ap.add_argument("--interface-cells", type=int, default=None)
    ap.add_argument("--dispersive-edges", type=int, default=0)
    ap.add_argument("--poles", type=int, default=1)
    ap.add_argument("--adjoint-design-edges", type=int, default=0)
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--fused", action="store_true", help="GPU fused step")
    ap.add_argument("--machine", default=None, help=", ".join(MACHINES))
    a = ap.parse_args()
    adj = {"design_edges": a.adjoint_design_edges} if a.adjoint_design_edges else None
    est = estimate(
        tuple(int(v) for v in a.shape.split(",")),
        a.steps,
        a.dtype,
        a.backend,
        a.cpml,
        adjoint=adj,
        dispersive_edges=a.dispersive_edges,
        n_poles=a.poles,
        n_ranks=a.ranks,
        conformal=a.conformal or False,
        interface_cells=a.interface_cells,
        fused=a.fused,
    )
    print(est)
    if a.machine:
        problems = preflight(est, a.machine, raise_error=False)
        print("FITS" if not problems else "DOES NOT FIT: " + "; ".join(problems))


if __name__ == "__main__":
    main()
