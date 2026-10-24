# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Subcell thin conducting/dispersive sheets and thin wires in 3D FIT.

Demonstrates:
  1. ``solver.add_sheet`` for a Salisbury radar-absorbing screen
     (R_s = eta_0 = 376.73 Ohm/sq placed d = lambda_0 / 4 in front of a PEC
     ground plane at f_0 = 3 GHz) and a dispersive Drude graphene sheet,
     compared against the exact thin-sheet transfer-matrix formulas.
  2. ``solver.add_thin_wire`` for a centre-fed half-wave dipole antenna of
     length L = 0.48 * lambda_0 at two subcell wire radii (a = dx / 10 and
     a = dx / 50) excited by a ``LumpedPort``, compared against the Hallen
     integral-equation Method of Moments reference ``hallen_dipole``.

Produces two figures under ``docs/img/``:
  1. ``docs/img/thin_sheet_salisbury_reflection.png``
  2. ``docs/img/thin_wire_dipole_impedance.png``

Usage::

    python examples/010_thin_sheets_and_wires.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c0

import wakis
from wakis.analytic.thin_wire import hallen_dipole
from wakis.lumped import LumpedPort
from wakis.sources import GaussianSource
from wakis.subcell import (
    ETA0,
    graphene_sheet,
    salisbury_screen_reflection,
    sheet_transmission_reflection,
)

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"


def run_1d_sheet(sheet_fn=None, f0=3.0e9, n_cells=48, dz=None, bc_high_z="cpml", nt=750):
    """Run a 1D plane-wave TEM scattering simulation along z."""
    lam0 = c0 / f0
    if dz is None:
        dz = lam0 / 48.0
    grid = wakis.GridFIT3D(
        -dz, dz, -dz, dz, 0.0, n_cells * dz, 2, 2, n_cells, use_mpi=False, verbose=0
    )
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", bc_high_z],
        n_pml=10,
        verbose=0,
    )
    if sheet_fn is not None:
        sheet_fn(solver)

    iz_src, iz_ref, iz_trn = 13, 18, n_cells - 14
    dt = solver.dt
    dz_src = float(grid.tL[0, 0, iz_src, "z"])
    t_arr = np.arange(nt) * dt
    ref_trace = np.zeros(nt)
    trn_trace = np.zeros(nt)
    for n in range(nt):
        t = t_arr[n]
        pulse = np.exp(-0.5 * ((t - 0.8e-9) / 0.18e-9) ** 2) * np.sin(2.0 * np.pi * f0 * t)
        solver.J[:, :, iz_src, "x"] = pulse / dz_src
        solver.one_step()
        ref_trace[n] = float(solver.E[0, 0, iz_ref, "x"])
        trn_trace[n] = float(solver.E[0, 0, iz_trn, "x"])
    return t_arr, ref_trace, trn_trace


def dtft(trace, t_arr, freqs):
    """Discrete-time Fourier transform of `trace` at `freqs` [Hz]."""
    f = np.atleast_1d(np.asarray(freqs, dtype=float))
    return np.sum(trace[:, None] * np.exp(-2j * np.pi * f[None, :] * t_arr[:, None]), axis=0)


def figure_salisbury_and_graphene():
    """Generate ``docs/img/thin_sheet_salisbury_reflection.png``."""
    f0 = 3.0e9
    lam0 = c0 / f0
    d = 0.25 * lam0
    n_spacer = 12
    dz = d / n_spacer
    n_cells = 48
    z_sheet = (n_cells - n_spacer) * dz

    freqs = np.linspace(1.5e9, 4.5e9, 61)
    t_arr, ref_inc, trn_inc = run_1d_sheet(
        sheet_fn=None, f0=f0, n_cells=n_cells, dz=dz, bc_high_z="cpml"
    )
    _, ref_sal, _ = run_1d_sheet(
        sheet_fn=lambda s: s.add_sheet(R_s=ETA0, plane=("z", z_sheet)),
        f0=f0,
        n_cells=n_cells,
        dz=dz,
        bc_high_z="pec",
    )
    _, ref_mut, _ = run_1d_sheet(
        sheet_fn=lambda s: s.add_sheet(
            R_s=ETA0, plane=("z", z_sheet), dual_length_factor=2.0
        ),
        f0=f0,
        n_cells=n_cells,
        dz=dz,
        bc_high_z="pec",
    )

    E_inc = dtft(ref_inc, t_arr, freqs)
    R_sal_fit = np.abs(dtft(ref_sal - ref_inc, t_arr, freqs) / E_inc) ** 2
    R_mut_fit = np.abs(dtft(ref_mut - ref_inc, t_arr, freqs) / E_inc) ** 2
    _, R_sal_ana = salisbury_screen_reflection(1.0 / ETA0, d, freqs)

    # Graphene Drude sheet in free space
    gr_base = graphene_sheet(E_F=0.35, tau=0.08e-9, T=300.0)
    gr_mat = gr_base.scaled((2.0 / ETA0) / gr_base.meta["sigma_dc"], name="graphene")
    z_mid = (n_cells // 2) * dz
    _, ref_gr, trn_gr = run_1d_sheet(
        sheet_fn=lambda s: s.add_sheet(material=gr_mat, plane=("z", z_mid)),
        f0=f0,
        n_cells=n_cells,
        dz=dz,
        bc_high_z="cpml",
    )
    E_inc_trn = dtft(trn_inc, t_arr, freqs)
    R_gr_fit = np.abs(dtft(ref_gr - ref_inc, t_arr, freqs) / E_inc) ** 2
    T_gr_fit = np.abs(dtft(trn_gr, t_arr, freqs) / E_inc_trn) ** 2
    A_gr_fit = 1.0 - R_gr_fit - T_gr_fit
    _, _, T_gr_ana, R_gr_ana = sheet_transmission_reflection(gr_mat, freqs)
    A_gr_ana = 1.0 - R_gr_ana - T_gr_ana

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax1.plot(freqs / 1e9, R_sal_ana, "k-", lw=2.0, label=r"Analytic Salisbury ($R_s=\eta_0$)")
    ax1.plot(
        freqs[::2] / 1e9,
        R_sal_fit[::2],
        "C0o",
        ms=4.5,
        label=r"3D FIT `add_sheet(R_s=eta0)`",
    )
    ax1.plot(
        freqs / 1e9,
        R_mut_fit,
        "C3--",
        lw=1.6,
        label=r"Mutated dual-length factor ($\times 2$)",
    )
    ax1.axvline(f0 / 1e9, color="0.5", ls=":", lw=1.0)
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel("Power reflectance $R(f)$")
    ax1.set_title(r"(a) Salisbury screen ($d = \lambda_0 / 4$ above PEC)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper center", fontsize=8.5)

    ax2.plot(freqs / 1e9, T_gr_ana, "C0-", lw=2.0, label="Analytic $T(f)$")
    ax2.plot(freqs / 1e9, R_gr_ana, "C1-", lw=2.0, label="Analytic $R(f)$")
    ax2.plot(freqs / 1e9, A_gr_ana, "C2-", lw=2.0, label="Analytic $A(f) = 1 - R - T$")
    ax2.plot(freqs[::3] / 1e9, T_gr_fit[::3], "C0o", ms=4.5, label="FIT $T(f)$")
    ax2.plot(freqs[::3] / 1e9, R_gr_fit[::3], "C1s", ms=4.5, label="FIT $R(f)$")
    ax2.plot(freqs[::3] / 1e9, A_gr_fit[::3], "C2^", ms=4.5, label="FIT $A(f)$")
    ax2.set_xlabel("Frequency [GHz]")
    ax2.set_ylabel("Power coefficient")
    ax2.set_title("(b) Dispersive Drude (graphene) sheet via surface ADE")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="right", fontsize=8.5, ncol=2)

    fig.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / "thin_sheet_salisbury_reflection.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    idx0 = int(np.argmin(np.abs(freqs - f0)))
    print(
        f"Saved {out} | Salisbury R(f0)={R_sal_fit[idx0]:.2e} "
        f"(mutated={R_mut_fit[idx0]:.3f}), max |dR|={np.max(np.abs(R_sal_fit - R_sal_ana)):.3e}"
    )


def run_dipole(radius_ratio, n_seg=15, correction=True):
    """Run a 3D FIT thin-wire dipole simulation for ``a = dx / radius_ratio``."""
    f0 = 1.0e9
    lam0 = c0 / f0
    length = 0.48 * lam0
    dz = length / n_seg
    dx = dy = dz
    nx = ny = 24
    nz = n_seg + 16
    grid = wakis.GridFIT3D(
        -nx * dx / 2,
        nx * dx / 2,
        -ny * dy / 2,
        ny * dy / 2,
        -nz * dz / 2,
        nz * dz / 2,
        nx,
        ny,
        nz,
        use_mpi=False,
        verbose=0,
    )
    ix, iy = nx // 2, ny // 2
    iz0 = (nz - n_seg) // 2
    iz1 = iz0 + n_seg
    iz_feed = (iz0 + iz1) // 2
    xw, yw = float(grid.x[ix]), float(grid.y[iy])
    z0, z1 = float(grid.z[iz0]), float(grid.z[iz1])
    z_feed = 0.5 * float(grid.z[iz_feed] + grid.z[iz_feed + 1])

    radius = dx / float(radius_ratio)
    port = LumpedPort(
        src_time=GaussianSource(frequency=f0, fwidth=0.5 * f0),
        component="z",
        center=(xw, yw, z_feed),
        size=(0.0, 0.0, 0.5 * dz),
        impedance=50.0,
    )
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=6, verbose=0
    )
    solver.add_thin_wire(
        [(xw, yw, z0), (xw, yw, z1)], radius=radius, correction=correction
    )
    solver.emsolve(int(40 * n_seg), source=port)
    freqs = np.linspace(0.85e9, 1.15e9, 121)
    Z_fit = port.impedance_in(freqs)
    Z_mom, _, _ = hallen_dipole(length, radius, freqs, n_segments=61, gap=dz)
    return freqs, Z_fit, Z_mom


def figure_thin_wire_dipole():
    """Generate ``docs/img/thin_wire_dipole_impedance.png``."""
    freqs, Z_fit_10, Z_mom_10 = run_dipole(10.0, n_seg=15, correction=True)
    _, Z_fit_50, Z_mom_50 = run_dipole(50.0, n_seg=15, correction=True)
    _, Z_uncorr, _ = run_dipole(50.0, n_seg=15, correction=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax1.plot(freqs / 1e9, Z_mom_10.real, "C0-", lw=2.0, label=r"Hallén MoM ($a = \Delta/10$)")
    ax1.plot(freqs / 1e9, Z_fit_10.real, "C0--", lw=2.0, label=r"3D FIT ($a = \Delta/10$)")
    ax1.plot(freqs / 1e9, Z_mom_50.real, "C1-", lw=2.0, label=r"Hallén MoM ($a = \Delta/50$)")
    ax1.plot(freqs / 1e9, Z_fit_50.real, "C1--", lw=2.0, label=r"3D FIT ($a = \Delta/50$)")
    ax1.plot(
        freqs / 1e9,
        Z_uncorr.real,
        "0.45",
        ls=":",
        lw=1.6,
        label=r"Uncorrected PEC edges (no $\ln(\Delta/a)$)",
    )
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel(r"Input resistance $R_{\mathrm{in}}(f)$ [$\Omega$]")
    ax1.set_title(r"(a) Thin-wire dipole input resistance ($L = 0.48\lambda_0$)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2.plot(freqs / 1e9, Z_mom_10.imag, "C0-", lw=2.0, label=r"Hallén MoM ($a = \Delta/10$)")
    ax2.plot(freqs / 1e9, Z_fit_10.imag, "C0--", lw=2.0, label=r"3D FIT ($a = \Delta/10$)")
    ax2.plot(freqs / 1e9, Z_mom_50.imag, "C1-", lw=2.0, label=r"Hallén MoM ($a = \Delta/50$)")
    ax2.plot(freqs / 1e9, Z_fit_50.imag, "C1--", lw=2.0, label=r"3D FIT ($a = \Delta/50$)")
    ax2.plot(
        freqs / 1e9,
        Z_uncorr.imag,
        "0.45",
        ls=":",
        lw=1.6,
        label=r"Uncorrected PEC edges",
    )
    ax2.axhline(0.0, color="0.5", ls=":", lw=1.0)
    ax2.set_xlabel("Frequency [GHz]")
    ax2.set_ylabel(r"Input reactance $X_{\mathrm{in}}(f)$ [$\Omega$]")
    ax2.set_title("(b) Thin-wire dipole input reactance & resonance shift")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / "thin_wire_dipole_impedance.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    f_res_10 = freqs[int(np.argmin(np.abs(Z_fit_10.imag)))] / 1e9
    f_res_50 = freqs[int(np.argmin(np.abs(Z_fit_50.imag)))] / 1e9
    f_mom_10 = freqs[int(np.argmin(np.abs(Z_mom_10.imag)))] / 1e9
    f_mom_50 = freqs[int(np.argmin(np.abs(Z_mom_50.imag)))] / 1e9
    print(
        f"Saved {out} | f_res(dx/10): FIT={f_res_10:.4f} GHz vs MoM={f_mom_10:.4f} GHz; "
        f"f_res(dx/50): FIT={f_res_50:.4f} GHz vs MoM={f_mom_50:.4f} GHz"
    )


if __name__ == "__main__":
    figure_salisbury_and_graphene()
    figure_thin_wire_dipole()
