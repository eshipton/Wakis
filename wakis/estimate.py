# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Resource estimates (peak memory, run time) for Wakis simulations, and a
preflight check against the target machine before launching a run.

The base cost per cell was measured with `benchmarks/calibrate_memory.py` on a
GCP g2-standard-16 (NVIDIA L4) VM, 48 runs of 16^3 to 128^3 cells (peak RSS after
setup and 10 steps, fresh process per point); it is linear in the cell count.
Monitors, adjoint storage and dispersive auxiliary state are added analytically.
A safety factor (default 1.25) covers setup temporaries not seen in calibration.
"""

import argparse
from dataclasses import dataclass, field

# Peak bytes = fixed + per_cell * cells  (host RAM; GPU memory for use_gpu)
_HOST = {  # (backend, cpml, dtype) -> (fixed bytes, bytes per cell)
    ("cpu", False, "float64"): (393e6, 1105.0),
    ("cpu", False, "float32"): (394e6, 809.0),
    ("cpu", True, "float64"): (409e6, 969.0),
    ("cpu", True, "float32"): (396e6, 624.0),
    ("gpu", False, "float64"): (607e6, 1047.0),
    ("gpu", False, "float32"): (580e6, 753.0),
    ("gpu", True, "float64"): (574e6, 948.0),
    ("gpu", True, "float32"): (561e6, 620.0),
}
_GPU = {
    (False, "float64"): 935.0,
    (False, "float32"): 712.0,
    (True, "float64"): 738.0,
    (True, "float32"): 458.0,
}
# Sustained throughput [cells * steps / s], conservative (docs/performance.md and
# the calibration runs): one CPU core, 8-core MKL, NVIDIA L4
_THROUGHPUT = {"cpu": 8e6, "mkl": 16e6, "gpu": 75e6}

# Target machines: usable host RAM and GPU memory [bytes]
MACHINES = {
    "laptop": dict(ram=2e9, gpu=0),  # memguard cap for local runs
    "n2-standard-4": dict(ram=16e9, gpu=0),
    "n2-standard-8": dict(ram=32e9, gpu=0),
    "n2-standard-16": dict(ram=64e9, gpu=0),
    "n2-highmem-16": dict(ram=128e9, gpu=0),
    "g2-standard-16": dict(ram=64e9, gpu=24e9),
    "g2-standard-32": dict(ram=128e9, gpu=24e9),
}
_OS_RESERVE = 0.85  # fraction of RAM a job may use (OS, Python, page cache)


@dataclass
class Estimate:
    cells: int
    host_bytes: float
    gpu_bytes: float
    seconds: float
    breakdown: dict = field(default_factory=dict)

    def __str__(self):
        gb = 1e9
        lines = [
            f"cells: {self.cells:,}",
            f"peak host RAM: {self.host_bytes / gb:.2f} GB",
            f"peak GPU memory: {self.gpu_bytes / gb:.2f} GB",
            f"run time: {self.seconds / 60:.1f} min",
        ]
        lines += [f"  {k}: {v / gb:.3f} GB" for k, v in self.breakdown.items()]
        return "\n".join(lines)


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
    dispersive_edges, n_poles : int, optional
        Edges with dispersive materials and poles per material.
    n_ranks : int, optional
        MPI ranks: memory and time are per rank.
    safety : float, optional
        Multiplier on the memory estimate.

    Returns
    -------
    Estimate
    """
    if not isinstance(cells, int):
        nx, ny, nz = cells
        cells = int(nx) * int(ny) * int(nz)
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

    mon = sum(16.0 * nf * ns * nc for nf, ns, nc in monitors) / n_ranks
    breakdown["monitors"] = mon
    if adjoint:
        stored = (
            steps * adjoint["design_edges"] * adjoint.get("store_bytes", 8) / n_ranks
        )
        state = (
            factor * per_cell * per_rank * 0.6
        )  # transposed operators + adjoint fields
        breakdown["adjoint stored increments"] = stored
        breakdown["adjoint state"] = state
    disp = dispersive_edges * n_poles * 3 * 8.0 / n_ranks
    breakdown["dispersive"] = disp

    host = sum(breakdown.values())
    if backend == "gpu":
        gpu += (
            mon
            + disp
            + (
                breakdown.get("adjoint state", 0)
                + breakdown.get("adjoint stored increments", 0)
            )
        )
    time_steps = steps * (2.2 if adjoint else 1.0)
    seconds = time_steps * per_rank * factor / _THROUGHPUT[backend]
    return Estimate(cells, host * safety, gpu * safety, seconds, breakdown)


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
    ap.add_argument("--adjoint-design-edges", type=int, default=0)
    ap.add_argument("--ranks", type=int, default=1)
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
        n_ranks=a.ranks,
    )
    print(est)
    if a.machine:
        problems = preflight(est, a.machine, raise_error=False)
        print("FITS" if not problems else "DOES NOT FIT: " + "; ".join(problems))


if __name__ == "__main__":
    main()
