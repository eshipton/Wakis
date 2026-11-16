# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Strong-scaling and bitwise-correctness study for 3D subgridding over MPI
(`wakis.subgrid3d.mpi`).

Benchmarks `DistributedSubgridFIT3D` at ``P in {1, 2, 4, 8, 16}`` (or
``{1, 2, 4}`` with ``--quick``) on the two loaded-cavity problems of
`studies/subgrid3d/run_study_lts.py`:
* Part D (``case="big"``): PEC cavity ``1.0 x 0.8 x 0.6 m`` with an
  ``eps_r = 9`` dielectric cube ``[0.3, 0.6] x [0.2, 0.5] x [0.2, 0.5] m``
  and a 2:1 subgrid patch ``[0.2, 0.7] x [0.1, 0.6] x [0.1, 0.6] m`` (large
  patch holding most of the unknowns).
* Part E (``case="small"``): same cavity with a small ``eps_r = 9`` cube of
  side ``0.1 m`` and a 2:1 patch ``[0.4, 0.6] x [0.3, 0.5] x [0.2, 0.4] m``
  (0.8 % of the cavity volume).

For each ``(case, kind in {"global", "lts"}, method in {"rcb", "blocks"}, P)``,
records:
* serial `SubgridLTSFIT3D(kernels="csr")` baseline throughput on the same VM,
  same mesh and same ``dt``;
* bitwise equality (`np.array_equal`) of `e`, `b`, probes and running DFTs
  against the serial solver after the first 20 steps;
* `steps/s`, `ns` per unknown-step, `MPI.Wtime` compute and halo-exchange time
  per step, halo sizes, and messages per step.

Usage:
    mpirun -n 4 python studies/subgrid3d/mpi/scaling.py --quick --out DIR
    mpirun -n 16 --oversubscribe python studies/subgrid3d/mpi/scaling.py --out DIR
"""

import argparse
import gc
import json
import os
import sys
import time
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")

# Avoid loading CuPy/CUDA and unrelated top-level wakis dependencies on CPU MPI ranks
sys.modules.setdefault("cupy", None)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _pkg_name, _rel in [
    ("wakis", "wakis"),
    ("wakis.unstructured", os.path.join("wakis", "unstructured")),
]:
    if _pkg_name not in sys.modules:
        _mod = types.ModuleType(_pkg_name)
        _mod.__path__ = [os.path.join(REPO_ROOT, _rel)]
        sys.modules[_pkg_name] = _mod

import numpy as np
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
SUBGRID_STUDY_DIR = os.path.dirname(HERE)
if SUBGRID_STUDY_DIR not in sys.path:
    sys.path.insert(0, SUBGRID_STUDY_DIR)

from run_study_lts import CASES, make_solver

from wakis.subgrid3d.mpi import (
    DistributedSubgridFIT3D,
    partition_stats,
)


class StudyPulse:
    """Picklable Gaussian-modulated sinusoid source for the loaded cavity."""

    def __init__(self, f0=1.9e8):
        self.f0 = float(f0)
        self.tau = 2.0 / self.f0

    def __call__(self, t):
        s = (t - 4.0 * self.tau) / self.tau
        return float(np.exp(-(s * s)) * np.sin(2.0 * np.pi * self.f0 * t))


def _configure_solver(case, kind, h_coarse, ratio, dt=None):
    """Build and seed a deterministic loaded-cavity solver for `case` and `kind`."""
    s = make_solver(case, kind, h_coarse, ratio, dt=dt, dt_rtol=1e-2)
    rng = np.random.default_rng(
        1000 + (0 if case == "big" else 10) + (0 if kind == "global" else 1)
    )
    sb = float(np.sqrt(np.median(s.D.real) / np.median(s.W.real)))
    s.e[:] = rng.standard_normal(s.n_free)
    s.b[:] = sb * rng.standard_normal(s.n_faces)
    cfg = CASES[case]
    pulse = StudyPulse()
    for pt, comp in cfg["src"]:
        s.add_current(pt, comp, pulse, amplitude=1.0)
    freqs = np.array([1.5e8, 1.9e8, 2.5e8])
    for pt, comp in cfg["probes"]:
        s.add_probe(pt, comp, freqs=freqs)
    return s


def _reset_solver(s, e0, b0):
    s.n = 0
    s.t = 0.0
    s.e[:] = e0
    s.b[:] = b0
    for p in s.probes:
        p["values"].clear()
        if p["dft"] is not None:
            p["dft"][:] = 0.0


def _measure_serial_baseline(s, e0, b0, n_verify=20, n_bench=40):
    """
    Run 20-step serial reference for the bitwise check, then time `n_bench`
    steps of serial `SubgridLTSFIT3D(kernels="csr")` on rank 0.
    """
    _reset_solver(s, e0, b0)
    s.run(n_verify)
    e_ref = np.asarray(s.e).copy()
    b_ref = np.asarray(s.b).copy()
    probes_ref = [np.asarray(p["values"]).copy() for p in s.probes]
    dfts_ref = [p["dft"].copy() for p in s.probes]
    energy_ref = float(s.energy(e_ref, e_ref, b_ref))

    # Warm-up and timed serial baseline
    s.one_step()
    t0 = time.perf_counter()
    for _ in range(n_bench):
        s.one_step()
    elapsed = time.perf_counter() - t0
    s_per_step = elapsed / n_bench
    unknowns = int(s.n_free + s.n_faces)
    wps = int(s.lts.work_per_step()) if s.lts is not None else unknowns

    _reset_solver(s, e0, b0)
    return {
        "e": e_ref,
        "b": b_ref,
        "probes": probes_ref,
        "dfts": dfts_ref,
        "energy": energy_ref,
        "serial_s_per_step": float(s_per_step),
        "serial_steps_per_s": float(1.0 / s_per_step),
        "serial_ns_per_unknown_step": float(s_per_step * 1e9 / unknowns),
        "serial_ns_per_work_step": float(s_per_step * 1e9 / wps),
        "unknowns": unknowns,
        "work_per_step": wps,
        "n_free": int(s.n_free),
        "n_faces": int(s.n_faces),
        "active_cells": int(s.active_cells.sum()),
        "dt": float(s.dt),
    }


def _run_one_config(
    comm_sub, s, ref, case, kind, h_coarse, ratio, method, P, n_verify=20, n_bench=40
):
    """Run 20-step bitwise check and `n_bench`-step timing on `comm_sub` of size `P`."""
    me = comm_sub.Get_rank()
    dist = DistributedSubgridFIT3D(
        s, comm_sub, method=method, mode="replicated", timers=True
    )
    dist.run(n_verify)
    st = dist.gather_state()
    p0 = dist.probe_signal(0)
    p1 = dist.probe_signal(1)
    d0 = dist.dft_data(0)
    d1 = dist.dft_data(1)
    E_mpi = dist.energy()
    cstats = dist.comm_stats()

    bitwise_ok = False
    max_diff = 0.0
    energy_rel = 0.0
    if me == 0:
        e_ok = np.array_equal(st["e"], ref["e"])
        b_ok = np.array_equal(st["b"], ref["b"])
        p_ok = np.array_equal(p0, ref["probes"][0]) and np.array_equal(
            p1, ref["probes"][1]
        )
        d_ok = np.array_equal(d0, ref["dfts"][0]) and np.array_equal(d1, ref["dfts"][1])
        bitwise_ok = bool(e_ok and b_ok and p_ok and d_ok)
        max_diff = float(
            max(
                np.max(np.abs(st["e"] - ref["e"])),
                np.max(np.abs(st["b"] - ref["b"])),
            )
        )
        energy_rel = float(abs(E_mpi - ref["energy"]) / ref["energy"])

    # Reset timers and counters for the timed benchmark window
    dist.t = {"compute": 0.0, "exchange": 0.0}
    dist.bytes_sent = 0
    dist.messages_sent = 0
    comm_sub.Barrier()
    t0 = MPI.Wtime()
    for _ in range(n_bench):
        dist.step()
    comm_sub.Barrier()
    wall = MPI.Wtime() - t0

    comp_max = comm_sub.reduce(dist.t["compute"], op=MPI.MAX, root=0)
    exch_max = comm_sub.reduce(dist.t["exchange"], op=MPI.MAX, root=0)
    msgs_sum = comm_sub.reduce(dist.messages_sent, op=MPI.SUM, root=0)
    msgs_max = comm_sub.reduce(dist.messages_sent, op=MPI.MAX, root=0)
    bytes_sum = comm_sub.reduce(dist.bytes_sent, op=MPI.SUM, root=0)

    if me != 0:
        return None

    pstats = partition_stats(s, parts=dist.parts, n_parts=P, weights="work")
    halo_e = [cs.get("e", 0) for cs in cstats]
    halo_b = [cs.get("b", 0) for cs in cstats]
    halo_lts = [
        sum(v for k, v in cs.items() if k.startswith(("fc", "phi"))) for cs in cstats
    ]
    s_per_step = float(wall / n_bench)
    steps_per_s = float(n_bench / wall)
    comp_step = float(comp_max / n_bench)
    exch_step = float(exch_max / n_bench)
    unknowns = ref["unknowns"]
    wps = ref["work_per_step"]

    return {
        "case": case,
        "kind": kind,
        "h_coarse": float(h_coarse),
        "ratio": int(ratio),
        "method": method,
        "P": int(P),
        "unknowns": unknowns,
        "work_per_step": wps,
        "n_steps_verify": int(n_verify),
        "n_steps_bench": int(n_bench),
        "bitwise_20_steps": bitwise_ok,
        "max_abs_diff_20_steps": max_diff,
        "energy_rel_diff_20_steps": energy_rel,
        "serial_s_per_step": ref["serial_s_per_step"],
        "serial_steps_per_s": ref["serial_steps_per_s"],
        "serial_ns_per_unknown_step": ref["serial_ns_per_unknown_step"],
        "s_per_step": s_per_step,
        "steps_per_s": steps_per_s,
        "compute_s_per_step": comp_step,
        "exchange_s_per_step": exch_step,
        "comm_fraction": float(exch_step / max(comp_step + exch_step, 1e-30)),
        "ns_per_unknown_step": float(s_per_step * 1e9 / unknowns),
        "ns_per_work_step": float(s_per_step * 1e9 / wps),
        "speedup_vs_serial": float(ref["serial_s_per_step"] / s_per_step),
        "imbalance": float(pstats["imbalance"]),
        "cell_imbalance": float(pstats["cell_imbalance"]),
        "cut_faces": int(pstats["cut_faces"]),
        "exchanges_per_step": int(dist.exchange_counts_per_step()),
        "messages_per_step_sum": float(msgs_sum / n_bench),
        "messages_per_step_max": float(msgs_max / n_bench),
        "bytes_per_step_sum": float(bytes_sum / n_bench),
        "halo_e_max": int(max(halo_e)),
        "halo_e_sum": int(sum(halo_e)),
        "halo_b_max": int(max(halo_b)),
        "halo_b_sum": int(sum(halo_b)),
        "halo_lts_max": int(max(halo_lts)),
        "halo_lts_sum": int(sum(halo_lts)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small meshes and P in {1, 2, 4}; writes scaling_quick.json to --out.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for scaling.json or scaling_quick.json.",
    )
    args = parser.parse_args()

    world = MPI.COMM_WORLD
    W = world.Get_size()
    me = world.Get_rank()

    target_ps = [1, 2, 4] if args.quick else [1, 2, 4, 8, 16]
    p_list = [p for p in target_ps if p <= W]
    if not p_list:
        p_list = [W]
    elif W not in p_list:
        p_list.append(W)

    mesh_specs = (
        [("big", 0.1, 2), ("small", 0.05, 2)]
        if args.quick
        else [("big", 0.05, 2), ("small", 0.025, 2)]
    )
    n_bench = 25 if args.quick else 40

    cases_meta = {}
    runs = []

    for case, h_coarse, ratio in mesh_specs:
        cases_meta[case] = {"h_coarse": h_coarse, "ratio": ratio}
        for kind in ("global", "lts"):
            dt_val = None
            ref = None
            if me == 0:
                s0 = _configure_solver(case, kind, h_coarse, ratio, dt=None)
                e0 = np.asarray(s0.e).copy()
                b0 = np.asarray(s0.b).copy()
                ref = _measure_serial_baseline(s0, e0, b0, n_verify=20, n_bench=n_bench)
                dt_val = ref["dt"]
                cases_meta[case][kind] = {
                    "dt": ref["dt"],
                    "unknowns": ref["unknowns"],
                    "work_per_step": ref["work_per_step"],
                    "n_free": ref["n_free"],
                    "n_faces": ref["n_faces"],
                    "active_cells": ref["active_cells"],
                    "serial_steps_per_s": ref["serial_steps_per_s"],
                    "serial_s_per_step": ref["serial_s_per_step"],
                    "serial_ns_per_unknown_step": ref["serial_ns_per_unknown_step"],
                }
            dt_val = world.bcast(dt_val, root=0)
            if me != 0:
                s0 = _configure_solver(case, kind, h_coarse, ratio, dt=dt_val)
            else:
                e0_b = np.asarray(s0.e).copy()
                b0_b = np.asarray(s0.b).copy()

            for method in ("rcb", "blocks"):
                p1_s_per_step = None
                for P in p_list:
                    world.Barrier()
                    color = 0 if me < P else MPI.UNDEFINED
                    comm_sub = world.Split(color=color, key=me)
                    if comm_sub != MPI.COMM_NULL:
                        if me == 0:
                            _reset_solver(s0, e0_b, b0_b)
                        row = _run_one_config(
                            comm_sub,
                            s0,
                            ref,
                            case,
                            kind,
                            h_coarse,
                            ratio,
                            method,
                            P,
                            n_verify=20,
                            n_bench=n_bench,
                        )
                        comm_sub.Free()
                        gc.collect()
                        if me == 0 and row is not None:
                            if P == 1:
                                p1_s_per_step = row["s_per_step"]
                            row["speedup_vs_p1"] = (
                                float(p1_s_per_step / row["s_per_step"])
                                if p1_s_per_step
                                else float(ref["serial_s_per_step"] / row["s_per_step"])
                            )
                            runs.append(row)
                            assert row["bitwise_20_steps"], (
                                f"Bitwise check failed for {case} {kind} {method} P={P}"
                            )
                    world.Barrier()
            del s0, ref
            gc.collect()

    if me == 0:
        os.makedirs(args.out, exist_ok=True)
        fname = "scaling_quick.json" if args.quick else "scaling.json"
        out_path = os.path.join(args.out, fname)

        # Merge with existing runs if invoked once per P sequentially
        if os.path.exists(out_path):
            try:
                with open(out_path, encoding="utf-8") as fh:
                    prev = json.load(fh)
                by_key = {
                    (r["case"], r["kind"], r["method"], r["P"]): r
                    for r in prev.get("runs", [])
                }
                for r in runs:
                    by_key[(r["case"], r["kind"], r["method"], r["P"])] = r
                runs = sorted(
                    by_key.values(),
                    key=lambda r: (r["case"], r["kind"], r["method"], r["P"]),
                )
            except (OSError, json.JSONDecodeError):
                pass

        flat = {}
        for r in runs:
            prefix = f"{r['case']}_{r['kind']}_{r['method']}_p{r['P']}"
            flat[f"{prefix}_steps_per_s"] = r["steps_per_s"]
            flat[f"{prefix}_ns_per_unknown_step"] = r["ns_per_unknown_step"]
            flat[f"{prefix}_speedup"] = r["speedup_vs_serial"]
            flat[f"{prefix}_imbalance"] = r["imbalance"]
            flat[f"{prefix}_comm_fraction"] = r["comm_fraction"]
            flat[f"{prefix}_halo_e_sum"] = r["halo_e_sum"]
            flat[f"{prefix}_halo_lts_sum"] = r["halo_lts_sum"]

        payload = {
            "quick": bool(args.quick),
            "mpi_world_size": int(W),
            "p_list": sorted({r["P"] for r in runs}),
            "cases": cases_meta,
            "runs": runs,
            **flat,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        all_bitwise = all(r["bitwise_20_steps"] for r in runs)
        max_diff_all = max((r["max_abs_diff_20_steps"] for r in runs), default=0.0)
        summary_parts = [
            f"all_bitwise={int(all_bitwise)}",
            f"max_diff={max_diff_all:.1e}",
        ]
        for k in sorted(flat):
            if k.endswith(("_steps_per_s", "_imbalance")):
                summary_parts.append(f"{k}={flat[k]:.3f}")
        print("NUMBERS scaling: " + " ".join(summary_parts), flush=True)


if __name__ == "__main__":
    main()
