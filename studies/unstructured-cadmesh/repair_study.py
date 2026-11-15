"""
Topological flip repair study on the CAD benchmark meshes at cpw = 10.

    python repair_study.py          # writes results/repair.json and prints summary

Reuses the three CAD geometries (`pillbox`, `layered`, `muon`) and analytic /
Cartesian-extrapolated reference frequencies from `cadmesh_study.py` and
`plot_cadmesh.py`. For each geometry at cpw = 10, compares:
- the unimproved mesh before `improve_dt`,
- node smoothing alone (`improve_dt(..., repair=False)`),
- topological flip repair + node smoothing (`improve_dt(..., repair=True)`),
both on gmsh's Delaunay baseline (`optimize="none"`) and on gmsh's 3D quality
optimizer output (`optimize="gmsh"`).
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
from cadmesh_study import (
    C0,
    F_TARGET,
    HERE,
    N_MODES,
    RES,
    _layered_step,
    layered_exact,
    make_mesh,
    pillbox_exact,
)
from plot_cadmesh import muon_reference, nearest

from wakis.unstructured import UnstructuredFIT3D
from wakis.unstructured.quality import stability_report
from wakis.unstructured.repair import flip_to_delaunay


def guard_negative_count(solver):
    """Count faces with negative dual length coupling free edges (solver guard criterion)."""
    return int(np.sum((solver.Mnu_diag < 0) & (np.abs(solver.Cf).sum(1).A1 > 0)))


def reference_f0(case, mesh=None, materials=None):
    """Lowest reference eigenfrequency [Hz] used by `cadmesh_study.py`."""
    if case == "pillbox":
        return float(pillbox_exact()[0])
    if case == "layered":
        return float(layered_exact()[0])
    if case == "muon":
        try:
            return float(muon_reference()[0]["f_star"])
        except (OSError, KeyError, IndexError, ValueError):
            s_w = UnstructuredFIT3D(mesh, materials, hodge="whitney")
            f_w, _ = s_w.eigenmodes(N_MODES["muon"], f_target=F_TARGET["muon"])
            return float(f_w[0])
    raise ValueError(f"Unknown case {case!r}")


def _eval_solver(case, mesh, materials, f_ref):
    """Evaluate stability step and lowest eigenmode on `mesh`."""
    n_faces = int(mesh.faces.shape[0])
    s = UnstructuredFIT3D(mesh, materials, allow_negative_faces=True)
    n_neg = guard_negative_count(s)
    assert 0 <= n_neg <= n_faces, (
        f"Negative face count {n_neg} exceeds n_faces {n_faces}"
    )
    srep = stability_report(s)
    w = int(srep["worst_element"])
    w_edges = mesh.edge_length[mesh.cell_edges[w]]
    f, _ = s.eigenmodes(N_MODES[case], f_target=F_TARGET[case])
    f0 = float(nearest(f, f_ref))
    return {
        "n_tets": int(mesh.n_cells),
        "n_nodes": int(mesh.n_nodes),
        "n_faces": n_faces,
        "n_negative_faces": n_neg,
        "guard_bypassed": True,
        "voronoi_valid": bool(n_neg == 0),
        "dt_max": float(s.dt_max),
        "courant_h": float(s.dt_max * C0 / mesh.edge_length.mean()),
        "limit_element": w,
        "limit_radius_ratio": float(srep["worst_radius_ratio"]),
        "limit_min_dihedral": float(srep["worst_min_dihedral"]),
        "limit_edge_length": float(w_edges.mean()),
        "f0_hz": f0,
        "f_all_hz": [float(x) for x in f],
        "lowest_freq_err": float(f0 / f_ref - 1.0),
    }


def run_case(case, cpw=10, optimize="none", n_sweeps=10):
    """Run before / `repair=False` / `repair=True` comparisons for one CAD case."""
    t0 = time.perf_counter()
    m_before, info_before = make_mesh(case, cpw, optimize=optimize, improve=None)
    mats = info_before.materials
    f_ref = reference_f0(case, m_before, mats)

    before = _eval_solver(case, m_before, mats, f_ref)

    # Pure topological flip repair without node smoothing
    m_flip, flip_rep = flip_to_delaunay(m_before, mats)
    flip_only = _eval_solver(case, m_flip, mats, f_ref)
    flip_rep = dict(flip_rep)
    flip_rep["negative_faces_before"] = before["n_negative_faces"]
    flip_rep["negative_faces_after"] = flip_only["n_negative_faces"]
    flip_only["repair_report"] = flip_rep

    # Node smoothing without topological repair (repair=False baseline)
    m_no_rep, _ = make_mesh(
        case,
        cpw,
        optimize=optimize,
        improve={"n_sweeps": n_sweeps, "repair": False},
    )
    after_no_repair = _eval_solver(case, m_no_rep, mats, f_ref)

    # Topological repair + node smoothing + topological repair (repair=True)
    m_rep, info_rep = make_mesh(
        case,
        cpw,
        optimize=optimize,
        improve={"n_sweeps": n_sweeps, "repair": True},
    )
    after_repair = _eval_solver(case, m_rep, mats, f_ref)
    repair_entry = {}
    if info_rep.improve_history and "repair" in info_rep.improve_history[-1]:
        repair_entry = dict(info_rep.improve_history[-1]["repair"])
        repair_entry["negative_faces_before"] = before["n_negative_faces"]
        repair_entry["negative_faces_after"] = after_repair["n_negative_faces"]

    assert before["n_negative_faces"] <= before["n_faces"]
    assert flip_only["n_negative_faces"] <= flip_only["n_faces"]
    assert after_no_repair["n_negative_faces"] <= after_no_repair["n_faces"]
    assert after_repair["n_negative_faces"] <= after_repair["n_faces"]

    dt_gain_vs_no_repair = float(after_repair["dt_max"] / after_no_repair["dt_max"])
    dt_gain_vs_before = float(after_repair["dt_max"] / before["dt_max"])

    return {
        "case": case,
        "cpw": int(cpw),
        "optimize": optimize,
        "n_sweeps": int(n_sweeps),
        "f_ref_hz": f_ref,
        "guard_bypassed": True,
        "n_faces_before": before["n_faces"],
        "negative_faces_before": before["n_negative_faces"],
        "n_faces_flip_only": flip_only["n_faces"],
        "negative_faces_flip_only": flip_only["n_negative_faces"],
        "n_faces_after_repair_false": after_no_repair["n_faces"],
        "negative_faces_after_repair_false": after_no_repair["n_negative_faces"],
        "n_faces_after_repair_true": after_repair["n_faces"],
        "negative_faces_after_repair_true": after_repair["n_negative_faces"],
        "dt_max_before": before["dt_max"],
        "dt_max_flip_only": flip_only["dt_max"],
        "dt_max_repair_false": after_no_repair["dt_max"],
        "dt_max_repair_true": after_repair["dt_max"],
        "dt_gain_vs_repair_false": dt_gain_vs_no_repair,
        "dt_gain_vs_before": dt_gain_vs_before,
        "limit_element": after_repair["limit_element"],
        "limit_radius_ratio": after_repair["limit_radius_ratio"],
        "limit_min_dihedral": after_repair["limit_min_dihedral"],
        "limit_edge_length": after_repair["limit_edge_length"],
        "lowest_freq_err_before": before["lowest_freq_err"],
        "lowest_freq_err_repair_false": after_no_repair["lowest_freq_err"],
        "lowest_freq_err_repair_true": after_repair["lowest_freq_err"],
        "before": before,
        "flip_only": flip_only,
        "after_repair_false": after_no_repair,
        "after_repair_true": after_repair,
        "repair_report": repair_entry,
        "wall_s": float(time.perf_counter() - t0),
    }


def _git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE)
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main():
    _layered_step()
    cpw = 10
    records = []
    for optimize in ("none", "gmsh"):
        for case in ("pillbox", "layered", "muon"):
            records.append(run_case(case, cpw=cpw, optimize=optimize))

    primary = [r for r in records if r["optimize"] == "none"]
    payload = {
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "cpw": cpw,
        "cases": primary,
        "all_runs": records,
    }
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, "repair.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)

    header = (
        f"{'case':<10} {'opt':<6} {'neg/faces_before':>16} {'neg/faces_rep=F':>16} "
        f"{'neg/faces_rep=T':>16} {'dt_gain(T/F)':>13} {'err_rep=F':>12} {'err_rep=T':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        b_str = f"{r['negative_faces_before']}/{r['n_faces_before']}"
        f_str = f"{r['negative_faces_after_repair_false']}/{r['n_faces_after_repair_false']}"
        t_str = (
            f"{r['negative_faces_after_repair_true']}/{r['n_faces_after_repair_true']}"
        )
        print(
            f"{r['case']:<10} {r['optimize']:<6} "
            f"{b_str:>16} "
            f"{f_str:>16} "
            f"{t_str:>16} "
            f"{r['dt_gain_vs_repair_false']:>13.4f} "
            f"{r['lowest_freq_err_repair_false']:>+12.5e} "
            f"{r['lowest_freq_err_repair_true']:>+12.5e}"
        )


if __name__ == "__main__":
    main()
