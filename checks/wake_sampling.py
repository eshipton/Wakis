"""Check wake potential time sampling on a closed rectangular PEC cavity.

Simulates an ultrarelativistic (beta = 1) Gaussian bunch crossing a closed
rectangular PEC cavity of size a x b x d = 32 x 32 x 96 mm (whole domain,
beam entering and leaving through the PEC end walls) and compares the
longitudinal bunch loss factor computed three ways:
(a) ``WakeSolver`` on ``main`` (which samples ``Ez`` at ``int(ts / dt)``),
(b) linear time interpolation between the two bracketing stored ``Ez`` steps,
(c) analytic modal sum of the closed rectangular PEC cavity.
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import wakis


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


def box_modes(a, b, d, x0, y0, sigma, n_sigma=9.0):
    """Compute modal angular frequencies and loss factors of a closed PEC box.

    Ported from ``.ref/dfw_analytic_box.py.txt``. For a cavity
    ``[0, a] x [0, b] x [0, d]`` crossed at ``(x0, y0)`` by an
    ultrarelativistic bunch (``v = c``), only modes with longitudinal electric
    field ``E_z = sin(kx x) sin(ky y) cos(kz z)`` couple to the beam.

    Parameters
    ----------
    a, b, d : float
        Cavity dimensions along x, y, z [m].
    x0, y0 : float
        Transverse beam position in cavity coordinates ``[0, a] x [0, b]`` [m].
    sigma : float
        Bunch rms longitudinal length [m].
    n_sigma : float, optional
        Mode wavenumber cutoff ``k * sigma <= n_sigma``. Default is 9.0.

    Returns
    -------
    omega : ndarray
        Modal angular frequencies [rad/s].
    kn : ndarray
        Modal loss factors [V/C].
    """
    kmax = n_sigma / sigma
    m = np.arange(1, int(kmax * a / np.pi) + 1)
    n = np.arange(1, int(kmax * b / np.pi) + 1)
    p = np.arange(0, int(kmax * d / np.pi) + 1)
    sx = np.sin(m * np.pi * x0 / a)
    sy = np.sin(n * np.pi * y0 / b)
    m, sx = m[np.abs(sx) > 1e-12], sx[np.abs(sx) > 1e-12]
    n, sy = n[np.abs(sy) > 1e-12], sy[np.abs(sy) > 1e-12]
    m_grid, n_grid, p_grid = np.meshgrid(m, n, p, indexing="ij")
    sx_grid, sy_grid = np.meshgrid(sx, sy, indexing="ij")
    sx_grid, sy_grid = sx_grid[..., None], sy_grid[..., None]
    kx = m_grid * np.pi / a
    ky = n_grid * np.pi / b
    kz = p_grid * np.pi / d
    kt2 = kx**2 + ky**2
    k = np.sqrt(kt2 + kz**2)
    keep = k <= kmax
    k = k[keep]
    kz = kz[keep]
    kt2 = kt2[keep]
    sx_grid = np.broadcast_to(sx_grid, keep.shape)[keep]
    sy_grid = np.broadcast_to(sy_grid, keep.shape)[keep]
    p_grid = p_grid[keep]

    integral = 0.5 * (
        (np.exp(1j * (k + kz) * d) - 1.0) / (1j * (k + kz))
        + (np.exp(1j * (k - kz) * d) - 1.0) / (1j * (k - kz))
    )
    v2 = (sx_grid * sy_grid) ** 2 * np.abs(integral) ** 2
    energy = np.where(p_grid == 0, a * b * d / 4.0, a * b * d / 8.0 * k**2 / kt2)
    kn = v2 / (2.0 * epsilon_0 * energy)
    return k * c_light, kn


def analytic_loss_factor(omega, kn, sigma):
    """Compute the Gaussian-bunch analytic loss factor from modal parameters.

    Parameters
    ----------
    omega : ndarray
        Modal angular frequencies [rad/s].
    kn : ndarray
        Modal loss factors [V/C].
    sigma : float
        Bunch rms longitudinal length [m].

    Returns
    -------
    float
        Total loss factor ``sum k_n exp(-(omega * sigma / c)^2)`` [V/C].
    """
    return float(np.sum(kn * np.exp(-((omega * sigma / c_light) ** 2))))


def wake_potential_interp(ez_file, s, q, v, ti, skip_cells=0):
    """Compute longitudinal wake potential with linear time interpolation of Ez.

    Ported from ``.ref/dfw_wake_interp.py.txt`` and adapted to upstream
    ``main``, where ``Ez.h5`` stores longitudinal primal-edge midpoints in
    ``hf['z']`` and edge lengths in ``hf['dz']`` so the domain entrance is
    ``z[0] - dz[0] / 2`` (matching ``WakeSolver.calc_long_WP``).

    Parameters
    ----------
    ez_file : str
        Path to the ``Ez.h5`` file written by ``SolverFIT3D.wakesolve``.
    s : ndarray
        Wake coordinate vector [m].
    q : float
        Bunch charge [C].
    v : float
        Beam velocity [m/s].
    ti : float
        Beam injection time [s].
    skip_cells : int, optional
        Number of longitudinal boundary cells to skip at each end. Default is 0.

    Returns
    -------
    ndarray
        Longitudinal wake potential ``W(s)`` [V/pC] in Wakis sign convention
        (negative = energy loss).
    """
    with h5py.File(ez_file, "r") as hf:
        t = np.asarray(hf["t"], dtype=np.float64)
        z = np.asarray(hf["z"], dtype=np.float64)
        dz = np.asarray(hf["dz"], dtype=np.float64)
        keys = sorted(k for k in hf if isinstance(k, str) and k.startswith("#"))
        ez = np.array([np.asarray(hf[k], dtype=np.float64) for k in keys])

    if ez.ndim > 2:
        ez = ez[:, ez.shape[1] // 2, ez.shape[2] // 2, :]

    nt = ez.shape[0]
    dz_arr = dz if np.ndim(dz) else np.full(z.size, dz, dtype=np.float64)
    # Recover physical domain entrance from first edge midpoint and its half-width.
    zmin = float(z[0] - 0.5 * dz_arr[0])

    zz = slice(skip_cells, -skip_cells) if skip_cells else slice(None)
    z = z[zz]
    dzk = dz_arr[zz]
    ez = ez[:, zz]

    dt = float(t[1] - t[0])
    wp = np.zeros(len(s), dtype=np.float64)
    for k in range(z.size):
        ts = (z[k] + s) / v - zmin / v - t[0] + ti
        x = ts / dt
        i0 = np.floor(x).astype(int)
        w = x - i0
        ok = (i0 >= 0) & (i0 + 1 < nt)
        wp[ok] += ((1.0 - w[ok]) * ez[i0[ok], k] + w[ok] * ez[i0[ok] + 1, k]) * dzk[k]

    return wp / (q * 1e12)


def run_check(args):
    """Run the closed PEC box wake simulation and compare the three loss factors.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        JSON-serializable dictionary of loss factors and relative errors.
    """
    t0 = time.perf_counter()
    a = args.a
    b = args.b
    d = args.d
    sigmaz = args.sigmaz
    dz = 1.6e-3 if (args.quick and args.dz is None) else (args.dz or 0.8e-3)
    dx = args.rxy * dz
    dy = args.rxy * dz
    wakelength = (
        0.1 if (args.quick and args.wakelength is None) else (args.wakelength or 0.3)
    )

    nx_cells = round(a / dx)
    ny_cells = round(b / dy)
    nz_cells = round(d / dz)

    # On main, GridFIT3D takes primal node counts (Nx, Ny, Nz) = cells + 1.
    nx_nodes = nx_cells + 1
    ny_nodes = ny_cells + 1
    nz_nodes = nz_cells + 1

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        contextlib.redirect_stdout(sys.stderr),
    ):
        grid = wakis.GridFIT3D(
            -a / 2.0,
            a / 2.0,
            -b / 2.0,
            b / 2.0,
            0.0,
            d,
            nx_nodes,
            ny_nodes,
            nz_nodes,
            verbose=0,
        )
        ez_file = os.path.join(tmpdir, "Ez.h5")
        wake = wakis.WakeSolver(
            wakelength=wakelength,
            q=1e-9,
            sigmaz=sigmaz,
            beta=1.0,
            xsource=args.xsource,
            ysource=args.ysource,
            xtest=args.xsource,
            ytest=args.ysource,
            skip_cells=0,
            results_folder=tmpdir,
            Ez_file=ez_file,
            save=False,
            verbose=0,
        )
        wake.n_transverse_cells = 0

        solver = wakis.SolverFIT3D(
            grid,
            wake,
            cfln=args.cfln,
            bc_low=["pec", "pec", "pec"],
            bc_high=["pec", "pec", "pec"],
            bg=[1.0, 1.0, 0.0],
            dtype=np.float64,
            use_gpu=False,
            verbose=0,
        )
        solver.wakesolve(wakelength=wakelength, compute_plane="longitudinal")

        # (a) WakeSolver on main (samples Ez at int(ts / dt); negative = loss)
        k_wakis_signed = float(wake.calc_loss_factor(save=False))
        k_wakis = -k_wakis_signed

        # (b) Same saved Ez with linear interpolation in time
        s_arr = np.asarray(wake.s, dtype=np.float64)
        wp_interp = wake_potential_interp(
            ez_file, s_arr, wake.q, wake.v, wake.ti, skip_cells=0
        )
        lam = np.asarray(wake.lambdas, dtype=np.float64)
        k_interp_signed = float(
            np.trapezoid(wp_interp * lam, s_arr) / np.trapezoid(lam, s_arr)
        )
        k_interp = -k_interp_signed

    # (c) Analytic modal sum at the actual grid node used by the source
    x_actual = float(grid.x[solver.ixt])
    y_actual = float(grid.y[solver.iyt])
    omega, kn = box_modes(a, b, d, x_actual + a / 2.0, y_actual + b / 2.0, sigmaz)
    k_analytic = analytic_loss_factor(omega, kn, sigmaz) * 1e-12

    rel_err_wakis = abs(k_wakis - k_analytic) / k_analytic
    rel_err_interp = abs(k_interp - k_analytic) / k_analytic

    runtime_s = time.perf_counter() - t0
    return {
        "script": "checks/wake_sampling.py",
        "status": "ok",
        "git_head": get_git_head(),
        "runtime_s": runtime_s,
        "inputs": {
            "quick": bool(args.quick),
            "a_m": float(a),
            "b_m": float(b),
            "d_m": float(d),
            "sigmaz_m": float(sigmaz),
            "dz_m": float(grid.dz[0]),
            "dx_m": float(grid.dx[0]),
            "dy_m": float(grid.dy[0]),
            "cfln": float(args.cfln),
            "dt_s": float(solver.dt),
            "Nt": int(solver.Nt),
            "wakelength_m": float(wakelength),
            "cells": [int(nx_cells), int(ny_cells), int(nz_cells)],
            "nodes": [int(nx_nodes), int(ny_nodes), int(nz_nodes)],
            "xsource_m": float(x_actual),
            "ysource_m": float(y_actual),
            "box_coords_x0_y0_m": [
                float(x_actual + a / 2.0),
                float(y_actual + b / 2.0),
            ],
            "n_coupled_modes": int(omega.size),
        },
        "loss_factor_wakis_V_per_pC": k_wakis,
        "loss_factor_interp_V_per_pC": k_interp,
        "loss_factor_analytic_V_per_pC": k_analytic,
        "signed_loss_factor_wakis_V_per_pC": k_wakis_signed,
        "signed_loss_factor_interp_V_per_pC": k_interp_signed,
        "rel_error_wakis_vs_analytic": rel_err_wakis,
        "rel_error_interp_vs_analytic": rel_err_interp,
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
        description="Compare Wakis int(t/dt) wake sampling vs linear time interpolation and analytic box."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with coarser mesh (dz=1.6 mm) and shorter wakelength (0.1 m).",
    )
    parser.add_argument(
        "--a", type=float, default=0.032, help="Cavity width in x [m] (default: 0.032)."
    )
    parser.add_argument(
        "--b", type=float, default=0.032, help="Cavity width in y [m] (default: 0.032)."
    )
    parser.add_argument(
        "--d",
        type=float,
        default=0.096,
        help="Cavity length in z [m] (default: 0.096).",
    )
    parser.add_argument(
        "--sigmaz",
        type=float,
        default=3.2e-3,
        help="Bunch rms length [m] (default: 3.2e-3).",
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=None,
        help="Longitudinal cell size [m] (default: 0.8e-3, or 1.6e-3 with --quick).",
    )
    parser.add_argument(
        "--rxy",
        type=float,
        default=2.0,
        help="Transverse-to-longitudinal spacing ratio dx/dz = dy/dz (default: 2.0).",
    )
    parser.add_argument(
        "--cfln",
        type=float,
        default=0.5,
        help="CFL number for SolverFIT3D (default: 0.5, Wakis default).",
    )
    parser.add_argument(
        "--wakelength",
        type=float,
        default=None,
        help="Wake length [m] (default: 0.3, or 0.1 with --quick).",
    )
    parser.add_argument(
        "--xsource",
        type=float,
        default=0.0,
        help="Beam x position in domain [-a/2, a/2] [m] (default: 0.0, i.e. x0=a/2).",
    )
    parser.add_argument(
        "--ysource",
        type=float,
        default=0.0,
        help="Beam y position in domain [-b/2, b/2] [m] (default: 0.0, i.e. y0=b/2).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point printing a single JSON object to stdout."""
    args = parse_args(argv)
    result = run_check(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
