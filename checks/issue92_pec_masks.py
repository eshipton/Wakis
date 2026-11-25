"""Check for issue #92: per-component PEC masks and mirror symmetry on main.

Builds a symmetric PEC box STL inside a symmetric vacuum domain with
``SolverFIT3D`` and inspects ``solver.ieps`` for each electric field component
(``Ex``, ``Ey``, ``Ez``). Reports how many edges on the box surface or inside
it have ``ieps != 0`` (overall and on the low/high face along each axis), and
whether the ``ieps == 0`` pattern of each component is mirror-symmetric when
flipped along each axis. Evaluates two placements:
1. ``between_nodes``: box faces placed strictly between primal node planes.
2. ``on_nodes``: box faces placed exactly on primal node planes.
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wakis import GridFIT3D, SolverFIT3D


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


def evaluate_placement(tmpdir, cells, half_domain, half_box, stl_method, tag):
    """Build GridFIT3D and SolverFIT3D for one box placement and inspect ieps.

    Parameters
    ----------
    tmpdir : str
        Temporary directory for writing the box STL file.
    cells : int
        Number of physical cells per axis (primal node count is ``cells + 1``).
    half_domain : float
        Domain half-width [m]; domain is ``[-half_domain, half_domain]^3``.
    half_box : float
        PEC box half-width [m]; box is ``[-half_box, half_box]^3``.
    stl_method : str
        STL classification method (``'legacy'`` or ``'voxelize_rectilinear'``).
    tag : str
        Placement identifier used for the temporary STL filename.

    Returns
    -------
    dict
        Counts of ``ieps != 0`` edges inside/on the box and per-face, plus
        mirror-symmetry booleans for each electric field component.
    """
    stl_path = os.path.join(tmpdir, f"pec_box_{stl_method}_{tag}.stl")
    pv.Cube(
        bounds=(-half_box, half_box, -half_box, half_box, -half_box, half_box)
    ).triangulate().save(stl_path)

    n_nodes = cells + 1
    with contextlib.redirect_stdout(sys.stderr):
        grid = GridFIT3D(
            -half_domain,
            half_domain,
            -half_domain,
            half_domain,
            -half_domain,
            half_domain,
            n_nodes,
            n_nodes,
            n_nodes,
            stl_solids={"box": stl_path},
            stl_materials={"box": "pec"},
            stl_method=stl_method,
            verbose=0,
        )
        solver = SolverFIT3D(
            grid,
            use_stl=True,
            bg=[1.0, 1.0, 0.0],
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            dtype=np.float64,
            verbose=0,
        )

    tol = 1e-12
    # Outermost primal node plane lying within [-half_box, half_box]
    b_node = float(np.max(grid.x[grid.x <= half_box + tol]))

    components = {}
    for comp in "xyz":
        coords = grid.primal_coords[comp]
        physical_slice = tuple(slice(0, len(c)) for c in coords)
        ieps = solver.ieps.to_matrix(comp)[physical_slice]
        pec_mask = ieps == 0.0

        sym_axes = {
            ax: bool(np.array_equal(pec_mask, np.flip(pec_mask, axis=i)))
            for i, ax in enumerate("xyz")
        }
        sym_axes["all"] = bool(all(sym_axes.values()))

        X, Y, Z = np.meshgrid(*coords, indexing="ij")
        in_or_on_box = (
            (np.abs(X) <= half_box + tol)
            & (np.abs(Y) <= half_box + tol)
            & (np.abs(Z) <= half_box + tol)
        )
        strictly_inside_box = (
            (np.abs(X) < half_box - tol)
            & (np.abs(Y) < half_box - tol)
            & (np.abs(Z) < half_box - tol)
        )

        # Faces on the outermost enclosed node planes x/y/z = +-b_node
        # (equals +-half_box when the box faces lie on node planes).
        faces = {}
        stl_plane_edges = 0
        for ax_name, C in zip("xyz", (X, Y, Z)):
            low_node = in_or_on_box & (np.abs(C - (-b_node)) <= tol)
            high_node = in_or_on_box & (np.abs(C - b_node) <= tol)
            faces[ax_name] = {
                "low_total": int(low_node.sum()),
                "low_ieps_nonzero": int((low_node & ~pec_mask).sum()),
                "high_total": int(high_node.sum()),
                "high_ieps_nonzero": int((high_node & ~pec_mask).sum()),
            }
            low_geom = in_or_on_box & (np.abs(C - (-half_box)) <= tol)
            high_geom = in_or_on_box & (np.abs(C - half_box) <= tol)
            stl_plane_edges += int(low_geom.sum() + high_geom.sum())

        components[f"E_{comp}"] = {
            "physical_shape": list(ieps.shape),
            "total_pec_edges": int(pec_mask.sum()),
            "edges_in_or_on_box": int(in_or_on_box.sum()),
            "ieps_nonzero_in_or_on_box": int((in_or_on_box & ~pec_mask).sum()),
            "ieps_nonzero_strictly_inside_box": int(
                (strictly_inside_box & ~pec_mask).sum()
            ),
            "edges_on_stl_planes": int(stl_plane_edges),
            "faces_ieps_nonzero": faces,
            "ieps_zero_mirror_symmetric": sym_axes,
        }

    return {
        "half_box_m": float(half_box),
        "box_node_plane_m": b_node,
        "cell_size_m": float(grid.dx[0]),
        "components": components,
    }


def run_check(args):
    """Run the PEC box mask check for both placements and STL methods.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        JSON-serializable dictionary of results.
    """
    t0 = time.perf_counter()
    cells = 16 if (args.quick and args.cells is None) else (args.cells or 32)
    half_domain = 0.5
    dx = (2.0 * half_domain) / cells
    box_half_cells = cells // 4

    # Use dyadic-rational coordinates when cells is a power of two so STL
    # float32 vertex storage is exact and introduces no rounding shift.
    half_box_on_nodes = box_half_cells * dx
    half_box_between_nodes = (box_half_cells + 0.25) * dx

    with tempfile.TemporaryDirectory() as tmpdir:
        between_nodes = evaluate_placement(
            tmpdir,
            cells,
            half_domain,
            half_box_between_nodes,
            args.stl_method,
            "between_nodes",
        )
        on_nodes = evaluate_placement(
            tmpdir,
            cells,
            half_domain,
            half_box_on_nodes,
            args.stl_method,
            "on_nodes",
        )
        voxelize_on_nodes = None
        if args.stl_method == "legacy":
            vox = evaluate_placement(
                tmpdir,
                cells,
                half_domain,
                half_box_on_nodes,
                "voxelize_rectilinear",
                "on_nodes",
            )
            voxelize_on_nodes = {
                k: {
                    "ieps_nonzero_in_or_on_box": v["ieps_nonzero_in_or_on_box"],
                    "faces_ieps_nonzero": v["faces_ieps_nonzero"],
                    "ieps_zero_mirror_symmetric": v["ieps_zero_mirror_symmetric"],
                }
                for k, v in vox["components"].items()
            }

    runtime_s = time.perf_counter() - t0
    out = {
        "script": "checks/issue92_pec_masks.py",
        "status": "ok",
        "git_head": get_git_head(),
        "runtime_s": runtime_s,
        "inputs": {
            "quick": bool(args.quick),
            "cells": int(cells),
            "nodes": [int(cells + 1)] * 3,
            "domain_bounds_m": [-half_domain, half_domain],
            "dx_m": float(dx),
            "half_box_between_nodes_m": float(half_box_between_nodes),
            "half_box_on_nodes_m": float(half_box_on_nodes),
            "stl_method": args.stl_method,
        },
        "between_nodes": between_nodes,
        "on_nodes": on_nodes,
    }
    if voxelize_on_nodes is not None:
        out["on_nodes_voxelize_rectilinear_summary"] = voxelize_on_nodes
    return out


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
        description="Check issue #92 PEC box edge masks and mirror symmetry on main."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller grid (16 cells per axis instead of 32).",
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=None,
        help="Number of physical cells per axis (default: 32, or 16 with --quick).",
    )
    parser.add_argument(
        "--stl-method",
        choices=["legacy", "voxelize_rectilinear"],
        default="legacy",
        help="Primary GridFIT3D STL classification method (default: legacy).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point printing a single JSON object to stdout."""
    args = parse_args(argv)
    result = run_check(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
