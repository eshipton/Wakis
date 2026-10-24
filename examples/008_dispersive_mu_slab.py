# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Plane-wave reflection and transmission through a dispersive magnetic (Lorentz
ferrite) slab, and complex permeability spectra with Snoek's resonance limit.

A broadband Gaussian plane wave (Ex, Hy) propagates along +z through a lossy
Lorentz-permeability slab (natural ferromagnetic resonance model of a soft
ferrite) bounded by periodic transverse boundaries and 15-cell CPML absorbers.
Power reflectance R(f) and transmittance T(f) spectra are extracted from two
FluxMonitor planes and compared with the exact 1D transfer-matrix solution
evaluated at the discrete ADE permeability ``material.mu_discrete(f, dt)``.

Produces two figures under ``docs/img/``:
  1. ``docs/img/dispersive_mu_lorentz_ferrite.png``: real and imaginary parts
     of mu(f) for the Lorentz ferrite model with its resonance f_r marked, and
     a family of NiZn ferrite curves illustrating Snoek's product
     (mu_s - 1) * f_r = const.
  2. ``docs/img/dispersive_mu_slab_rt.png``: FIT vs transfer-matrix R(f),
     T(f), and absorption A(f) = 1 - R - T across the magnetic resonance,
     together with the residual errors |dR| and |dT| at two grid resolutions
     (dz = 5 mm and 2.5 mm) demonstrating second-order convergence.

Usage::

    python examples/008_dispersive_mu_slab.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c0

from wakis import FluxMonitor, GridFIT3D, SolverFIT3D
from wakis.dispersive import DispersiveMaterial, Lorentz
from wakis.sources import CurrentSource, GaussianSource

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"


def transfer_matrix_slab(freqs, d, eps_r, mu_r):
    """Exact normal-incidence power R(f), T(f) of a homogeneous (eps_r, mu_r) slab."""
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    eps_r = np.asarray(eps_r, dtype=complex)
    mu_r = np.asarray(mu_r, dtype=complex)
    n = np.sqrt(eps_r * mu_r)
    n = np.where(np.imag(n) > 0, -n, n)
    eta = np.sqrt(mu_r / eps_r)
    eta = np.where(np.real(eta) < 0, -eta, eta)
    r12 = (eta - 1.0) / (eta + 1.0)
    phase = np.exp(-2j * omega * n * d / c0)
    r = r12 * (1.0 - phase) / (1.0 - (r12**2) * phase)
    t = (1.0 - r12**2) * np.exp(-1j * omega * n * d / c0) / (
        1.0 - (r12**2) * phase
    )
    return np.abs(r) ** 2, np.abs(t) ** 2


def run_ferrite_slab(ferrite, freqs, thickness=0.1, dz=5e-3):
    """Run slab + vacuum reference FIT simulations and return R(f), T(f), dt."""

    def _fluxes(mat):
        nz = int(round(1.2 / dz))
        grid = GridFIT3D(-0.01, 0.01, -0.01, 0.01, -0.6, 0.6, 2, 2, nz, verbose=0)
        solver = SolverFIT3D(
            grid,
            cfln=0.5,
            bc_low=["periodic", "periodic", "cpml"],
            bc_high=["periodic", "periodic", "cpml"],
            n_pml=15,
            verbose=0,
        )
        if mat is not None:
            solver.add_dispersive_material(
                mat, center=(0.0, 0.0, thickness / 2), size=(1.0, 1.0, thickness)
            )
        src_time = GaussianSource(frequency=1.0e9, fwidth=0.45e9)
        sheet = CurrentSource(
            src_time, component="x", center=(0.0, 0.0, -0.3), size=(0.02, 0.02, 0.0)
        )
        planes = [
            FluxMonitor(freqs, center=(0.0, 0.0, z0), size=(0.02, 0.02, 0.0))
            for z0 in (-0.15, thickness + 0.1)
        ]
        Nt = int((src_time.end_time + 20e-9) / solver.dt)
        solver.emsolve(Nt, source=sheet, monitors=planes)
        return [p.get_flux() for p in planes], solver.dt

    (below_ref, above_ref), _ = _fluxes(None)
    (below, above), dt = _fluxes(ferrite)
    return 1.0 - below / below_ref, above / above_ref, dt


def plot_lorentz_ferrite(ferrite, f_r, dt):
    """Plot Lorentz ferrite complex permeability and Snoek's product."""
    f = np.linspace(0.1e9, 3.0e9, 500)
    mu_cont = ferrite.mu(f)
    mu_disc = ferrite.mu_discrete(f, dt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax1.plot(f / 1e9, np.real(mu_cont), "C0-", lw=2.0, label=r"$\mu'(f)$ continuous")
    ax1.plot(
        f / 1e9,
        -np.imag(mu_cont),
        "C1-",
        lw=2.0,
        label=r"$-\mu''(f)$ continuous",
    )
    f_sub = f[::20]
    mu_d_sub = mu_disc[::20]
    ax1.plot(
        f_sub / 1e9,
        np.real(mu_d_sub),
        "C0o",
        ms=4,
        label=r"$\mu_d'(f, \Delta t)$ FIT",
    )
    ax1.plot(
        f_sub / 1e9,
        -np.imag(mu_d_sub),
        "C1s",
        ms=4,
        label=r"$-\mu_d''(f, \Delta t)$ FIT",
    )
    ax1.axvline(f_r / 1e9, color="k", ls="--", lw=1.1, label=f"$f_r = {f_r/1e9:.1f}$ GHz")
    ax1.axhline(ferrite.mu_inf, color="0.6", ls=":", lw=0.9)
    ax1.set_xlabel("Frequency $f$ [GHz]")
    ax1.set_ylabel("Relative permeability")
    ax1.set_title(
        r"Lorentz Ferrite Model ($\mu_\infty=2, \Delta\mu=2, f_r=1.3\ \mathrm{GHz}$)"
    )
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Snoek's law (mu_s - 1) * f_r = const = 3.9 GHz
    snoek_ghz = 3.9
    f_grid = np.linspace(0.05e9, 4.5e9, 600)
    grades = [
        (4.9, r"High-$\mu_s$ ferrite ($\mu_s=4.9, f_r=1.0$ GHz)", "C0"),
        (3.0, r"Mid-$\mu_s$ ferrite ($\mu_s=3.0, f_r=1.95$ GHz)", "C2"),
        (2.3, r"High-$f_r$ ferrite ($\mu_s=2.3, f_r=3.0$ GHz)", "C3"),
    ]
    for mu_s, label, color in grades:
        dmu = mu_s - 1.0
        fr_i = (snoek_ghz * 1e9) / dmu
        w0_i = 2.0 * np.pi * fr_i
        mat_i = DispersiveMaterial(
            mu_inf=1.0,
            mu_poles=[Lorentz(delta_eps=dmu, omega_0=w0_i, gamma=0.2 * w0_i)],
        )
        mu_i = mat_i.mu(f_grid)
        ax2.plot(f_grid / 1e9, np.real(mu_i), color=color, lw=1.8, label=label)
        ax2.plot(f_grid / 1e9, -np.imag(mu_i), color=color, ls="--", lw=1.3)
        ax2.axvline(fr_i / 1e9, color=color, ls=":", lw=0.9, alpha=0.8)

    ax2.set_xlabel("Frequency $f$ [GHz]")
    ax2.set_ylabel(r"$\mu'(f)$ (solid), $-\mu''(f)$ (dashed)")
    ax2.set_title(r"Snoek's limit (illustrative): $(\mu_s - 1)\, f_r = 3.9\ \mathrm{GHz}$")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = IMG_DIR / "dispersive_mu_lorentz_ferrite.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def plot_slab_rt(freqs, f_r, R_5, T_5, R_an_5, T_an_5, R_25, T_25, R_an_25, T_an_25):
    """Plot slab reflectance, transmittance, absorption, and resolution convergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    f_ghz = freqs / 1e9

    abs_tm = 1.0 - R_an_25 - T_an_25
    abs_fdtd = 1.0 - R_25 - T_25

    ax1.plot(f_ghz, R_an_25, "C0-", lw=1.8, label="$R(f)$ analytic TM")
    ax1.plot(f_ghz, R_25, "C0o", ms=4.5, label="$R(f)$ Wakis FIT")
    ax1.plot(f_ghz, T_an_25, "C1-", lw=1.8, label="$T(f)$ analytic TM")
    ax1.plot(f_ghz, T_25, "C1s", ms=4.5, label="$T(f)$ Wakis FIT")
    ax1.plot(f_ghz, abs_tm, "C2--", lw=1.5, label="$A = 1-R-T$ TM")
    ax1.plot(f_ghz, abs_fdtd, "C2^", ms=4.5, label="$A$ Wakis FIT")
    ax1.axvline(f_r / 1e9, color="k", ls=":", lw=1.0, label="$f_r$")
    ax1.set_xlabel("Frequency $f$ [GHz]")
    ax1.set_ylabel("Power fraction")
    ax1.set_title(
        r"Lorentz-$\mu$ Slab ($d = 10$ cm, $\Delta z = 2.5$ mm)", fontsize=10.5
    )
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(fontsize=8, ncol=2, loc="upper right")
    ax1.grid(True, alpha=0.3)

    dR_5 = np.abs(R_5 - R_an_5)
    dT_5 = np.abs(T_5 - T_an_5)
    dR_25 = np.abs(R_25 - R_an_25)
    dT_25 = np.abs(T_25 - T_an_25)

    ax2.semilogy(f_ghz, dR_5, "C0--o", ms=3.5, lw=1.2, label=r"$|\Delta R|$ ($\Delta z = 5$ mm)")
    ax2.semilogy(f_ghz, dR_25, "C0-o", ms=4.0, lw=1.6, label=r"$|\Delta R|$ ($\Delta z = 2.5$ mm)")
    ax2.semilogy(f_ghz, dT_5, "C1--s", ms=3.5, lw=1.2, label=r"$|\Delta T|$ ($\Delta z = 5$ mm)")
    ax2.semilogy(f_ghz, dT_25, "C1-s", ms=4.0, lw=1.6, label=r"$|\Delta T|$ ($\Delta z = 2.5$ mm)")
    ax2.axvline(f_r / 1e9, color="k", ls=":", lw=1.0)
    ax2.set_xlabel("Frequency $f$ [GHz]")
    ax2.set_ylabel("Absolute error vs transfer matrix")
    ax2.set_title(
        f"2nd-Order Convergence (max $|\\Delta R|={dR_25.max():.2e}, |\\Delta T|={dT_25.max():.2e}$)",
        fontsize=10,
    )
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = IMG_DIR / "dispersive_mu_slab_rt.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    f_r = 1.3e9
    w0 = 2.0 * np.pi * f_r
    ferrite = DispersiveMaterial(
        mu_inf=2.0,
        mu_poles=[Lorentz(delta_eps=2.0, omega_0=w0, gamma=2.0 * np.pi * 0.1e9)],
    )
    freqs = np.linspace(0.65e9, 1.45e9, 17)
    d = 0.10

    R_5, T_5, dt_5 = run_ferrite_slab(ferrite, freqs, thickness=d, dz=5e-3)
    R_an_5, T_an_5 = transfer_matrix_slab(freqs, d, 1.0, ferrite.mu_discrete(freqs, dt_5))

    R_25, T_25, dt_25 = run_ferrite_slab(ferrite, freqs, thickness=d, dz=2.5e-3)
    R_an_25, T_an_25 = transfer_matrix_slab(freqs, d, 1.0, ferrite.mu_discrete(freqs, dt_25))

    print(
        f"dz=5.0mm: max |dR|={np.max(np.abs(R_5 - R_an_5)):.4e}, "
        f"max |dT|={np.max(np.abs(T_5 - T_an_5)):.4e}"
    )
    print(
        f"dz=2.5mm: max |dR|={np.max(np.abs(R_25 - R_an_25)):.4e}, "
        f"max |dT|={np.max(np.abs(T_25 - T_an_25)):.4e}"
    )
    plot_lorentz_ferrite(ferrite, f_r, dt_25)
    plot_slab_rt(freqs, f_r, R_5, T_5, R_an_5, T_an_5, R_25, T_25, R_an_25, T_an_25)
