"""
Measured numerical phase velocity vs propagation angle, Yee vs TE/TM.

Periodic box (dx = dy = 2 dz), one standing Fourier mode per run (Ey of the
TE polarisation, k in the xz plane), omega from the two-term recurrence of a
point signal (as in tests/test_045_tete.py). Lines: Dohlus & Zagorodnov
eq. (27) for TE/TM and the Yee relation, at each scheme's own time step
(TE/TM: c dt = dz; Yee: 0.99 of its CFL limit). Writes
results/dispersion_vs_angle.json and figures/dispersion_vs_angle.png.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light

import wakis

D = np.array([2e-3, 2e-3, 1e-3])
NX, NZ = 24, 48


def measure(scheme, mx, mz, n_steps=80):
    grid = wakis.GridFIT3D(0, NX * D[0], 0, 2 * D[1], 0, NZ * D[2], NX, 2, NZ, verbose=0)
    s = wakis.SolverFIT3D(grid, bc_low=["periodic"] * 3, bc_high=["periodic"] * 3,
                          scheme=scheme, cfln=0.99, verbose=0)  # fmt: skip
    k = np.array([2 * np.pi * mx / (NX * D[0]), 0, 2 * np.pi * mz / (NZ * D[2])])
    i, j, kk = np.meshgrid(np.arange(NX), np.arange(2), np.arange(NZ), indexing="ij")
    s.one_step()
    s.E.fromarray(np.zeros_like(s.E.array))
    s.H.fromarray(np.zeros_like(s.H.array))
    s.E[:, :, :, "y"] = np.cos(k[0] * i * D[0] + k[2] * kk * D[2])
    f = []
    for _ in range(n_steps):
        s.one_step()
        f.append(s.E[1, 1, 1, "y"])
    f = np.array(f)
    cw = np.dot(f[1:-1], f[2:] + f[:-2]) / (2 * np.dot(f[1:-1], f[1:-1]))
    w = np.arccos(cw) / s.dt
    return k, w, s.dt


def theory(k, dt, scheme):
    K = 0.5 * k * D
    sx, sz = np.sin(K[0]) ** 2 / D[0] ** 2, np.sin(K[2]) ** 2 / D[2] ** 2
    if scheme == "tete":
        rhs = sz + sx * (1 - (c_light * dt / D[2]) ** 2 * np.sin(K[2]) ** 2)
    else:
        rhs = sz + sx
    return 2 / dt * np.arcsin(c_light * dt * np.sqrt(rhs))


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    out = {}
    # Lattice wave vectors (|k| varies: 9.6-16 cells per wavelength)
    pairs = [(0, 4), (1, 4), (2, 4), (2, 3), (3, 2), (3, 1), (3, 0)]
    for scheme in ("tete", "yee"):
        rows = []
        for mx, mz in pairs:
            k, w, dt = measure(scheme, mx, mz)
            kn = np.linalg.norm(k)
            rows.append(dict(angle=float(np.degrees(np.arctan2(k[0], k[2]))),
                             cells_per_wavelength=float(2 * np.pi / kn / D[2]),
                             vph=float(w / (c_light * kn)),
                             vph_theory=float(theory(k, dt, scheme) / (c_light * kn)),
                             dt=float(dt)))  # fmt: skip
        out[scheme] = rows
    with open("results/dispersion_vs_angle.json", "w") as f:
        json.dump(out, f, indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for scheme, col, lab in (("yee", "C0", "Yee (c dt = 0.99 CFL)"),
                             ("tete", "C3", "TE/TM (c dt = dz)")):  # fmt: skip
        r = out[scheme]
        ang = [x["angle"] for x in r]
        ax[0].plot(ang, [x["vph"] - 1 for x in r], "o", color=col, label=lab)
        ax[0].plot(ang, [x["vph_theory"] - 1 for x in r], "-", color=col, lw=0.8)
        ax[1].semilogy(ang, [abs(x["vph"] - x["vph_theory"]) + 1e-17 for x in r],
                       "s", color=col, label=lab)  # fmt: skip
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(xlabel="angle from z axis [deg]", ylabel=r"$v_{ph}/c - 1$",
              title="Measured (markers) vs dispersion relation (lines)")  # fmt: skip
    ax[0].legend()
    ax[1].set(xlabel="angle from z axis [deg]",
              ylabel=r"|measured - theory| ($v_{ph}/c$)",
              title="Agreement with eq. (27) / Yee relation")  # fmt: skip
    ax[1].legend()
    fig.suptitle(r"Periodic box, $\Delta x=\Delta y=2\Delta z$, lattice modes with $\lambda$ = 9.6-16 $\Delta z$")
    fig.tight_layout()
    fig.savefig("figures/dispersion_vs_angle.png", dpi=150)
    for s_ in out:
        print(s_, [(round(x["angle"]), f"{x['vph'] - 1:.3e}") for x in out[s_]])


if __name__ == "__main__":
    main()
