"""Check for issue #91: longitudinal-to-transverse wake leakage on axis.

Reproduces the test-001 PEC cubic cavity with beam pipes on upstream ``main``
with source and test particle on the geometric axis (xs = ys = xt = yt = 0),
float64 precision, on CPU. Reports the dominant longitudinal impedance peak
frequency, |Z_parallel|, |Z_x|, and |Z_y| at that peak, max |WP_x| and
max |WP_y|, and the ratio |Z_x| / |Z_parallel| at the peak.
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pyvista as pv
from scipy.constants import c as c_light

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wakis import GridFIT3D, SolverFIT3D, WakeSolver


def get_git_head():
    """Return the current git commit SHA of the repository.

    Returns
    -------
    str
        Commit hash from ``git rev-parse HEAD``.
    """
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def run_check(args):
    """Run the on-axis cubic cavity wakefield simulation and extract metrics.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        JSON-serializable dictionary of inputs and impedance/wake metrics.
    """
    t0 = time.perf_counter()
    cells = 30 if (args.quick and args.cells is None) else (args.cells or 50)
    nz = 150 if args.nz is None else args.nz
    wakelength = (
        0.3 if (args.quick and args.wakelength is None) else (args.wakelength or 1.0)
    )
    sigmaz = args.sigmaz
    skip_cells = args.skip_cells

    # On main, GridFIT3D takes primal node counts (Nx, Ny, Nz) and builds
    # (Nx - 1, Ny - 1, Nz - 1) physical cells.
    nx_nodes = cells + 1
    ny_nodes = cells + 1
    nz_nodes = nz + 1

    stl_file = os.path.join(REPO_ROOT, "tests", "stl", "001_cubic_cavity.stl")
    surf = pv.read(stl_file)
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in surf.bounds)

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        contextlib.redirect_stdout(sys.stderr),
    ):
        grid = GridFIT3D(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            nx_nodes,
            ny_nodes,
            nz_nodes,
            stl_solids={"cavity": stl_file},
            stl_materials={"cavity": "vacuum"},
            stl_method=args.stl_method,
            verbose=0,
        )

        ez_file = os.path.join(tmpdir, "Ez.h5")
        wake = WakeSolver(
            wakelength=wakelength,
            q=1e-9,
            sigmaz=sigmaz,
            beta=1.0,
            xsource=0.0,
            ysource=0.0,
            xtest=0.0,
            ytest=0.0,
            save=False,
            results_folder=tmpdir,
            Ez_file=ez_file,
            skip_cells=skip_cells,
            verbose=0,
        )

        solver = SolverFIT3D(
            grid,
            wake,
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            use_stl=True,
            bg="pec",
            dtype=np.float64,
            use_gpu=False,
            verbose=0,
        )

        solver.wakesolve(wakelength=wakelength, save_J=False)

    z_abs = np.abs(wake.Z)
    peak_idx = int(np.argmax(z_abs))
    f_peak_hz = float(wake.f[peak_idx])
    z_par_peak = float(z_abs[peak_idx])
    zx_peak = float(np.abs(wake.Zx[peak_idx]))
    zy_peak = float(np.abs(wake.Zy[peak_idx]))
    max_wpx = float(np.max(np.abs(wake.WPx)))
    max_wpy = float(np.max(np.abs(wake.WPy)))
    max_wp_par = float(np.max(np.abs(wake.WP)))

    width_x = xmax - xmin
    width_y = ymax - ymin
    f_tm110_hz = float(0.5 * c_light * np.sqrt(1.0 / width_x**2 + 1.0 / width_y**2))

    # Count any interior asymmetry in the classified primal cavity mask.
    mask_asym = {}
    cavity_mask = grid.primal_masks["cavity"]
    for comp in "xyz":
        coords = grid.primal_coords[comp]
        arr = cavity_mask.to_matrix(comp)[tuple(slice(0, len(c)) for c in coords)]
        interior = (
            arr[:, 1:-1, 1:-1]
            if comp == "x"
            else arr[1:-1, :, 1:-1]
            if comp == "y"
            else arr[1:-1, 1:-1, 1:-1]
        )
        mask_asym[f"E_{comp}"] = {
            "x_flip_mismatches": int(np.sum(interior != interior[::-1, :, :])),
            "y_flip_mismatches": int(np.sum(interior != interior[:, ::-1, :])),
        }

    runtime_s = time.perf_counter() - t0
    return {
        "script": "checks/issue91_transverse_leak.py",
        "status": "ok",
        "git_head": get_git_head(),
        "runtime_s": runtime_s,
        "inputs": {
            "quick": bool(args.quick),
            "cells": int(cells),
            "nz": int(nz),
            "nodes": [int(nx_nodes), int(ny_nodes), int(nz_nodes)],
            "dx_m": float(grid.dx[0]),
            "dy_m": float(grid.dy[0]),
            "dz_m": float(grid.dz[0]),
            "sigmaz_m": float(sigmaz),
            "wakelength_m": float(wakelength),
            "skip_cells": int(skip_cells),
            "stl_method": args.stl_method,
            "dtype": "float64",
            "device": "cpu",
            "xs_m": 0.0,
            "ys_m": 0.0,
            "xt_m": 0.0,
            "yt_m": 0.0,
            "x_axis_node_m": float(grid.x[solver.ixt]),
            "y_axis_node_m": float(grid.y[solver.iyt]),
        },
        "f_peak_Hz": f_peak_hz,
        "f_peak_GHz": f_peak_hz * 1e-9,
        "f_tm110_closed_box_GHz": f_tm110_hz * 1e-9,
        "abs_Z_parallel_at_peak_Ohm": z_par_peak,
        "abs_Zx_at_peak_Ohm": zx_peak,
        "abs_Zy_at_peak_Ohm": zy_peak,
        "max_abs_WP_parallel_V_per_pC": max_wp_par,
        "max_abs_WPx_V_per_pC": max_wpx,
        "max_abs_WPy_V_per_pC": max_wpy,
        "ratio_abs_Zx_to_Z_parallel_at_peak": zx_peak / z_par_peak,
        "ratio_abs_Zy_to_Z_parallel_at_peak": zy_peak / z_par_peak,
        "primal_mask_interior_asymmetry": mask_asym,
    }


def parse_args(argv=None):
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Check issue #91 longitudinal-to-transverse wake leakage on main."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a shorter check (< 3 min on one CPU core).",
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=None,
        help="Transverse physical cells per axis (default: 50, or 30 with --quick).",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=150,
        help="Longitudinal physical cells (default: 150).",
    )
    parser.add_argument(
        "--wakelength",
        type=float,
        default=None,
        help="Wake length to simulate [m] (default: 1.0, or 0.3 with --quick).",
    )
    parser.add_argument(
        "--sigmaz",
        type=float,
        default=18.5e-3,
        help="Bunch rms length [m] (default: 18.5e-3).",
    )
    parser.add_argument(
        "--skip-cells",
        type=int,
        default=12,
        help="Number of longitudinal boundary cells skipped in wake integration.",
    )
    parser.add_argument(
        "--stl-method",
        choices=["legacy", "voxelize_rectilinear"],
        default="legacy",
        help="GridFIT3D STL classification method (default: legacy).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point printing a single JSON object to stdout."""
    args = parse_args(argv)
    result = run_check(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
