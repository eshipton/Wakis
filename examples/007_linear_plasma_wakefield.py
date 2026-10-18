# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Linear plasma wakefield of a rigid relativistic bunch in a plasma-filled pipe.

A Gaussian bunch (`wakis.sources.Beam`, v = c) travels along a perfectly
conducting circular pipe of radius a filled with a cold, collisionless
plasma, modelled as a Drude medium (`wakis.plasma.cold_plasma`). Behind the
bunch the plasma electrons oscillate at w_p and leave an undamped wake with
wavelength lambda_p = 2 pi c / w_p. The longitudinal field E_z(r, xi) is
compared with linear cold-fluid theory (`wakis.plasma.linear_wake`):

    E_z = -(q k_p^2 / 2 pi eps_0) g(r) exp(-k_p^2 sigma_z^2 / 2) cos(k_p xi)

far behind the bunch, g(r) = K_0(k_p r) - K_0(k_p a) I_0(k_p r) / I_0(k_p a).

Three bunch lengths k_p sigma_z = 0.5, sqrt(2), 2 show the resonance of the
wake amplitude at k_p sigma_z = sqrt(2). The pipe is meshed with conformal
PEC cells; ``--staircase`` compares with the staircased wall.

Expected: RMS deviation from theory ~0.3 % for k_p sigma_z = sqrt(2) and 2
at 32 cells per lambda_p; for k_p sigma_z = 0.5 the transition radiation
of the vacuum-plasma entrance (stronger for short bunches) is still visible
2-3 lambda_p behind the bunch (2.3 % RMS; 0.4 % at 48 cells per lambda_p).

Run time (32 cells per lambda_p, 0.56 M cells, ~3200 steps per bunch length):
1 min 43 s for all three runs on an NVIDIA L4 (``--gpu``, mostly the
conformal setup on the CPU; 5 s of time stepping per run), peak RSS 2.0 GB
(`wakis.estimate`: 1.5 GB host + 0.7 GB GPU; the conformal geometry setup
adds ~0.5 GB). Without ``--gpu`` expect a few minutes per run.

Usage::

    python 007_linear_plasma_wakefield.py [--gpu] [--cells 32] [--staircase]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.constants import c

from wakis import GridFIT3D, SolverFIT3D
from wakis.estimate import estimate, preflight
from wakis.plasma import cold_plasma, linear_wake, wake_amplitude
from wakis.sources import Beam

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--cells", type=int, default=32, help="cells per lambda_p")
parser.add_argument("--staircase", action="store_true")
parser.add_argument("--machine", default="laptop", help="see wakis.estimate")
args = parser.parse_args()

# ---------- Plasma and pipe ----------
lambda_p = 1e-3  # plasma wavelength [m] (n_e = 1.1e21 m^-3)
k_p = 2 * np.pi / lambda_p  # plasma wavenumber [1/m]
omega_p = k_p * c
radius = 2.0 / k_p  # pipe radius a, k_p a = 2
length = 24 * lambda_p  # long pipe: entrance transients fall behind
q = 1e-9  # bunch charge [C]; the linear wake scales with q

h = lambda_p / args.cells  # cubic cells
n_half = int(np.ceil(radius / h)) + 2
Nz = int(round(length / h))
shape = (2 * n_half + 1, 2 * n_half + 1, Nz + 1)
est = estimate(
    shape,
    steps=int(1.25 * length / (0.5 * h / np.sqrt(3))),
    backend="gpu" if args.gpu else "cpu",
    dispersive_edges=3 * np.prod(shape),
)
print(est)
preflight(est, args.machine)

pipe = pv.Cylinder(
    center=(0, 0, length / 2), direction=(0, 0, 1), radius=radius,
    height=3 * length, resolution=720,
).triangulate()  # fmt: skip
grid = GridFIT3D(
    -n_half * h, n_half * h, -n_half * h, n_half * h, 0, length,
    2 * n_half, 2 * n_half, Nz,
    stl_solids={"pipe": pipe}, stl_materials={"pipe": "vacuum"},
    conformal=False if args.staircase else "pec", verbose=0,
)  # fmt: skip


def run(kps):
    """Send one bunch with k_p sigma_z = kps; return xi and E_z(r, xi)."""
    solver = SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg="pec", use_stl=True,
        use_gpu=args.gpu, verbose=0,
    )  # fmt: skip
    # Plasma everywhere except the first lambda_p: an ultra-relativistic
    # bunch leaving the PEC entrance wall in vacuum does not radiate; the
    # transition radiation at the plasma boundary trails the bunch (group
    # velocity < c) and is left behind in a long pipe.
    z0 = lambda_p - h / 4  # between the E samples of two cell layers
    solver.add_dispersive_material(
        cold_plasma(omega_p), center=(0, 0, z0 + 2 * length), size=(4 * length,) * 3
    )
    beam = Beam(q=q, sigmaz=kps / k_p, xsource=0.0, ysource=0.0)
    # stop when the bunch centre is one lambda_p before the exit wall
    Nt = int(((length - lambda_p - solver.z.min()) / c + beam.ti) / solver.dt)
    solver.emsolve(Nt, source=[beam])
    z_bunch = solver.z.min() + c * (Nt * solver.dt - beam.ti)
    xi = solver.z - z_bunch  # E_z lives on the z midpoints
    i0 = int(np.argmin(abs(grid.x)))
    ez = {}
    for di in range(1, int(radius / h)):  # interior nodes, not the cut-cell layer
        e = solver.E[i0 + di, i0, :, "z"]  # host array (also with use_gpu)
        ez[grid.x[i0 + di]] = np.asarray(e.get() if hasattr(e, "get") else e)
    return xi, ez


kps_list = [0.5, np.sqrt(2), 2.0]
results = {kps: run(kps) for kps in kps_list}

# ---------- Wake behind the bunch at r ~ a / 2 ----------
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for ax, kps in zip(axs, kps_list):
    xi, ez = results[kps]
    r = min(ez, key=lambda x: abs(x - radius / 2))
    th, _, _ = linear_wake(r, 0.0, xi, k_p, kps / k_p, q=q, radius=radius)
    ax.plot(xi / lambda_p, ez[r] / 1e6, label="Wakis")
    ax.plot(xi / lambda_p, th / 1e6, "k--", lw=1, label="linear theory")
    window = (xi < -3 * kps / k_p) & (xi > -2 * lambda_p)
    err = np.sqrt(np.mean((ez[r] - th)[window] ** 2)) / np.max(abs(th[window]))
    ax.set_title(
        f"$k_p\\sigma_z$ = {kps:.2f}, r = {r / radius:.2f} a: "
        f"RMS deviation {100 * err:.2f} % (window $-2\\lambda_p<\\xi<-3\\sigma_z$)",
        fontsize=9,
    )
    ax.set_ylabel("$E_z$ [MV/m]")
    ax.set_xlim(-6, 1)
axs[0].legend(loc="lower left")
axs[-1].set_xlabel(r"$\xi / \lambda_p$ = (z - z$_{bunch}$) / $\lambda_p$")
fig.tight_layout()
fig.savefig("plasma_wake_ez.png", dpi=90)

# ---------- Amplitude vs bunch length and radius ----------
# E_z = E_0 exp(-(k_p sigma_z)^2 / 2) with E_0 = q k_p^2 g / (2 pi eps_0). For a
# given bunch length sigma_z and a density scan (k_p a, k_p r fixed),
# E_z sigma_z^2 / (q g / 2 pi eps_0) = (k_p sigma_z)^2 exp(-(k_p sigma_z)^2 / 2),
# maximal (2 / e) at k_p sigma_z = sqrt(2) [Esarey et al. 2009].
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
kps_fine = np.linspace(0.2, 3.0, 200)
r_ref = min(results[kps_list[0]][1], key=lambda x: abs(x - radius / 2))
e0 = wake_amplitude(r_ref, k_p, 0.0, q, radius)
axs[0].plot(kps_fine, kps_fine**2 * np.exp(-(kps_fine**2) / 2), "k-", label="theory")
for kps in kps_list:
    xi, ez = results[kps]
    window = (xi < -3 * kps / k_p) & (xi > -2 * lambda_p)
    basis = np.column_stack([np.cos(k_p * xi[window]), np.sin(k_p * xi[window])])
    rs = np.array(sorted(ez))
    amps = np.array(
        [np.hypot(*np.linalg.lstsq(basis, ez[r][window], rcond=None)[0]) for r in rs]
    )
    a_ref = amps[np.argmin(abs(rs - r_ref))]
    axs[0].plot(kps, kps**2 * a_ref / e0, "o", color="C0")
    axs[1].plot(rs / radius, amps / 1e6, "o", label=f"Wakis $k_p\\sigma_z$={kps:.2f}")
    r_fine = np.linspace(0.05, 1, 200) * radius
    axs[1].plot(
        r_fine / radius, wake_amplitude(r_fine, k_p, kps / k_p, q, radius) / 1e6, "k-",
        lw=0.8,
    )  # fmt: skip
axs[0].axvline(np.sqrt(2), color="grey", ls=":")
axs[0].set_xlabel(r"$k_p\sigma_z$")
axs[0].set_ylabel(r"$(k_p\sigma_z)^2\,E_{z,\max}/E_0$ (fixed $\sigma_z$, r = a/2)")
axs[0].legend()
axs[1].set_xlabel("r / a")
axs[1].set_yscale("log")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("plasma_wake_amplitude.png", dpi=90)
plt.show()
