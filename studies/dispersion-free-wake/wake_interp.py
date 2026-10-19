"""
Longitudinal wake potential from a Wakis Ez.h5 file with linear time
interpolation.

Same integral and s-grid as `WakeSolver.calc_long_WP` / `calc_long_WP_3d`,
W(s) = (1/q) sum_k Ez(x_t, y_t, z_k, t_k(s)) dz_k with
t_k(s) = (z_k + s - z_min)/v + t_i, but Ez is interpolated linearly between
the two stored time steps instead of taken at index int(t/dt) (which
samples up to one time step early, a first-order error in dt unless t/dt is
an integer, as it is for TE/TM at c dt = dz).
"""

import h5py
import numpy as np


def wake_potential_interp(Ez_file, s, q, v, ti, skip_cells=0):
    """
    Returns W(s) [V/pC] with Wakis' sign convention (negative = loss).

    Parameters
    ----------
    Ez_file : str
        HDF5 file written by `SolverFIT3D.wakesolve` (keys '#00000' ...).
    s : ndarray
        Wake coordinate [m] (e.g. `WakeSolver.s`).
    q, v, ti : float
        Bunch charge [C], velocity [m/s] and injection time [s].
    """
    with h5py.File(Ez_file, "r") as hf:
        t = np.asarray(hf["t"])
        z = np.asarray(hf["z"])
        dz = np.asarray(hf["dz"])
        keys = sorted(k for k in hf.keys() if k.startswith("#"))
        Ez = np.array([np.asarray(hf[k]) for k in keys])  # (nt, ...)
    if Ez.ndim > 2:  # 3x3 transverse block: centre column
        Ez = Ez[:, Ez.shape[1] // 2, Ez.shape[2] // 2, :]
    nt = Ez.shape[0]
    zz = slice(skip_cells, -skip_cells) if skip_cells else slice(None)
    z, dzk, Ez = z[zz], (dz if np.ndim(dz) else np.full(z.size, dz))[zz], Ez[:, zz]
    dt = t[1] - t[0]
    zmin = np.min(np.asarray(z))
    W = np.zeros(len(s))
    for k in range(z.size):
        ts = (z[k] + s) / v - zmin / v - t[0] + ti
        x = ts / dt
        i0 = np.floor(x).astype(int)
        w = x - i0
        ok = (i0 >= 0) & (i0 + 1 < nt)
        W[ok] += ((1 - w[ok]) * Ez[i0[ok], k] + w[ok] * Ez[i0[ok] + 1, k]) * dzk[k]
    return W / (q * 1e12)
