"""Quickstart (b): 2D Mie scattering by a dielectric cylinder with a TF/SF plane wave."""

from pathlib import Path
import tempfile

import numpy as np
import pyvista as pv
from scipy.constants import c as c0

import wakis
from wakis.analytic import cylinder_efficiencies
from wakis.sources import GaussianSource

# 1. Cylinder parameters: radius a = 1 um, refractive index n = 2.0 (eps_r = 4.0)
a_cyl = 1.0e-6
n_cyl = 2.0
res_per_um = 20.0  # 20 cells/um (same as tutorial 01 --quick)
dx = 1.0e-6 / res_per_um
nz = 4
n_pml = 10

# Box half-widths (in cells from the origin)
i_tf = int(round(1.25e-6 / dx))
i_flux = i_tf + 4
i_pml = i_flux + 12
n_half = i_pml + n_pml
half = n_half * dx
hz = (nz / 2) * dx

with tempfile.TemporaryDirectory() as tmpdir:
    stl_path = str(Path(tmpdir) / "cylinder.stl")
    pv.Cylinder(
        radius=a_cyl, height=10e-6, direction=(0, 0, 1), resolution=360, capping=True
    ).triangulate().save(stl_path)

    # 2. Build 2D periodic slab with conformal subpixel averaging
    grid = wakis.GridFIT3D(
        -half, half, -half, half, -hz, hz,
        2 * n_half, 2 * n_half, nz,
        stl_solids={"cylinder": stl_path},
        stl_materials={"cylinder": [n_cyl**2, 1.0]},
        conformal="all",
        verbose=0,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=n_pml,
        use_stl=True,
        verbose=0,
    )

    # 3. Broadband TF/SF plane wave (TM: E along z) and scattered-field FluxMonitor
    x_sim = np.linspace(2.0, 6.0, 21)
    freqs = x_sim * c0 / (2.0 * np.pi * a_cyl)
    f_center = 0.5 * (freqs[0] + freqs[-1])
    f_width = 0.25 * (freqs[-1] - freqs[0])

    pulse = GaussianSource(frequency=f_center, fwidth=f_width)
    src = wakis.TFSFSource(
        pulse,
        center=(0.0, 0.0, 0.0),
        size=(2.0 * i_tf * dx, 2.0 * i_tf * dx, None),
        direction="+x",
        polarization="z",
        amplitude=1.0,
    )
    lz = 2.0 * dx
    scat = wakis.FluxMonitor(freqs, size=(2.0 * i_flux * dx, 2.0 * i_flux * dx, lz))

    n_max = int(450e-15 / solver.dt)
    solver.emsolve(
        n_max,
        source=[src],
        monitors=[scat],
        stop_when_energy_decayed=(1e-6, 50),
    )

    # 4. Normalize scattered power per unit length by incident intensity and shadow width 2a
    phi_inc = src.incident_flux_density(freqs)
    q_sca_sim = scat.get_flux() / (lz * phi_inc) / (2.0 * a_cyl)
    q_sca_ref = cylinder_efficiencies(n_cyl, x_sim, "TM")[1]
    rel_rms = np.sqrt(np.mean((q_sca_sim - q_sca_ref) ** 2)) / np.sqrt(np.mean(q_sca_ref**2))

    print(f"Grid: {grid.Nx}x{grid.Ny}x{grid.Nz}, steps = {solver.Nt}/{n_max}")
    print(f"TM Q_sca relative RMS error across x in [2, 6]: {100.0 * rel_rms:.2f} %")
    for idx in (0, len(x_sim) // 2, -1):
        print(
            f"  x = {x_sim[idx]:.1f}: Q_sca(sim) = {q_sca_sim[idx]:.4f}, "
            f"Q_sca(Mie) = {q_sca_ref[idx]:.4f}"
        )
