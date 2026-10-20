"""Quickstart (a): 3D PEC rectangular cavity resonances with harminv."""

import numpy as np
from scipy.constants import c as c0

from wakis import GridFIT3D, ResonanceMonitor, SolverFIT3D
from wakis.sources import CurrentSource, GaussianSource

# 1. Define a 10 cm x 12 cm x 8 cm rectangular cavity on a 20 x 24 x 16 FIT grid.
#    By default (boundary_at_domain_edge=True), PEC walls sit exactly at [0, Lx/Ly/Lz].
Lx, Ly, Lz = 0.10, 0.12, 0.08
Nx, Ny, Nz = 20, 24, 16
grid = GridFIT3D(0.0, Lx, 0.0, Ly, 0.0, Lz, Nx, Ny, Nz, verbose=0)

solver = SolverFIT3D(
    grid,
    bc_low=["pec", "pec", "pec"],
    bc_high=["pec", "pec", "pec"],
    verbose=0,
)

# 2. Excite TM_mnp modes (Ez != 0 for p = 0, 1, ...) with an off-symmetry Ez dipole
pulse = GaussianSource(frequency=2.5e9, fwidth=1.5e9)
src = CurrentSource(
    pulse,
    center=(0.37 * Lx, 0.41 * Ly, 0.48 * Lz),
    component="z",
    amplitude=1.0,
)

# 3. Record Ez after the source ends (t_start = pulse.end_time) and run harminv
mon = ResonanceMonitor(
    component="Ez",
    center=(0.63 * Lx, 0.67 * Ly, 0.52 * Lz),
    t_start=pulse.end_time,
)

solver.emsolve(1500, source=[src], monitors=[mon])

# 4. Compare extracted modes against analytical TM_mnp frequencies
res = mon.harminv(1.5e9, 3.5e9)
f_tm110 = 0.5 * c0 * np.sqrt((1.0 / Lx) ** 2 + (1.0 / Ly) ** 2)
f_tm111 = 0.5 * c0 * np.sqrt((1.0 / Lx) ** 2 + (1.0 / Ly) ** 2 + (1.0 / Lz) ** 2)
f_tm120 = 0.5 * c0 * np.sqrt((1.0 / Lx) ** 2 + (2.0 / Ly) ** 2)
f_tm210 = 0.5 * c0 * np.sqrt((2.0 / Lx) ** 2 + (1.0 / Ly) ** 2)

print(f"Grid: {Nx}x{Ny}x{Nz} cells, dt = {solver.dt * 1e12:.3f} ps")
print(
    "Analytical TM_110, TM_111, TM_120, TM_210 [GHz]: "
    f"{f_tm110 * 1e-9:.4f}, {f_tm111 * 1e-9:.4f}, "
    f"{f_tm120 * 1e-9:.4f}, {f_tm210 * 1e-9:.4f}"
)
for f, amp, err in zip(res.frequency, np.abs(res.amplitude), res.error):
    print(f"  Extracted f = {f * 1e-9:.4f} GHz, |A| = {amp:.3e}, err = {err * 1e-6:.2e} MHz")
