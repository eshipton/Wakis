"""Quickstart (c): Wakefield and beam-coupling impedance of a lossy pillbox cavity."""

from pathlib import Path
import tempfile

import numpy as np
import pyvista as pv

from wakis import GridFIT3D, SolverFIT3D, WakeSolver

repo_root = Path(__file__).resolve().parents[2]
solid_cavity = str(repo_root / "examples" / "data" / "001_vacuum_cavity.stl")
solid_shell = str(repo_root / "examples" / "data" / "001_lossymetal_shell.stl")

solids = pv.read(solid_cavity) + pv.read(solid_shell)
Nx, Ny, Nz = 30, 30, 61

grid = GridFIT3D(
    *solids.bounds,
    Nx,
    Ny,
    Nz,
    stl_solids={"cavity": solid_cavity, "shell": solid_shell},
    stl_materials={"cavity": "vacuum", "shell": [30.0, 1.0, 30.0]},
    verbose=0,
)

add_space = 6
wakelength = 2.0  # simulated wake length [m]

with tempfile.TemporaryDirectory() as tmpdir:
    wake = WakeSolver(
        q=1e-9,            # bunch charge [C]
        sigmaz=10e-2,      # RMS bunch length [m]
        beta=1.0,          # relativistic velocity v/c
        xsource=0.0,
        ysource=0.0,
        xtest=0.0,
        ytest=0.0,
        add_space=add_space,
        wakelength=wakelength,
        results_folder=tmpdir + "/",
        Ez_file=tmpdir + "/Ez.h5",
        save=False,
        verbose=0,
    )

    solver = SolverFIT3D(
        grid,
        wake,
        bc_low=["pec", "pec", "pec"],
        bc_high=["pec", "pec", "pec"],
        use_stl=True,
        bg="pec",
        verbose=0,
    )

    solver.wakesolve(
        wakelength=wakelength,
        add_space=add_space,
        plot=False,
    )

    i_peak = int(np.argmax(np.abs(wake.Z)))
    print(f"Grid: {Nx}x{Ny}x{Nz}, wake length = {wakelength:.1f} m, samples = {len(wake.s)}")
    print(f"Min/Max longitudinal wake potential WP: [{wake.WP.min():.2f}, {wake.WP.max():.2f}] V/pC")
    print(f"Peak |Z_longitudinal|: {np.abs(wake.Z[i_peak]):.2f} Ohm at f = {wake.f[i_peak] * 1e-9:.3f} GHz")
