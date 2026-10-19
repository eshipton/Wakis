# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Baseline arm: Cartesian FIT (`SolverFIT3D` on a one-cell periodic-z slab).

* geometry from STL cylinders, ``conformal=None`` (staircase masks) or
  ``'all'`` (diagonal subpixel averaging; dispersive cells through the
  passive sub-capacitor network of `wakis.tensor_dispersive`);
* rectilinear grid, uniform (h everywhere) or graded: h on the bounding box
  of the scatterers plus a 2 h band, geometric growth (`growth` per cell) to
  h_out = min(4 h, lambda_min,host / ppw_out), the CPML (``n_pml`` cells of
  the last size) behind the case's absorber radius;
* `TFSFSource` box around the scatterers (in the host, ``background_eps_r``),
  host filling the whole domain (``bg``);
* DFT of the scattered E_z (TM) or H_z (TE) on the case's extraction circles
  (bilinear interpolation of the Yee samples), the incident phasor at the
  origin from the TF/SF auxiliary line, and for gap cases the total in-plane
  E at the gap point (bilinear on the E_x / E_y staggered grids).

The resolution label is h [m] on the scatterers.
"""

import time

import numpy as np
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0

from ...cylinder_harmonics import circle_points
from ..runner import Arm

NM = 1e-9


def bilinear(xs, ys, points, nx):
    """Sparse bilinear interpolation from the tensor grid xs x ys (flat i + nx j)."""
    import scipy.sparse as sp

    rows, cols, vals = [], [], []
    for p, (px, py) in enumerate(np.atleast_2d(points)):
        i = int(np.clip(np.searchsorted(xs, px) - 1, 0, len(xs) - 2))
        j = int(np.clip(np.searchsorted(ys, py) - 1, 0, len(ys) - 2))
        tx = (px - xs[i]) / (xs[i + 1] - xs[i])
        ty = (py - ys[j]) / (ys[j + 1] - ys[j])
        for di, dj, w in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                          (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):  # fmt: skip
            rows.append(p)
            cols.append((i + di) + nx * (j + dj))
            vals.append(w)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(rows) // 4, nx * len(ys)))


def graded_axis(fine_half, h, h_out, r_out, n_pml, growth=1.2, graded=True):
    """
    Symmetric node coordinates: h on |x| <= fine_half (a node at 0), then
    geometric growth to h_out reaching r_out, then n_pml cells of the last
    size. ``graded=False``: h up to r_out.
    """
    n_core = int(np.ceil(fine_half / h - 1e-9))
    core = h * np.arange(n_core + 1)
    if graded and h_out > h:
        xs = [core[-1]]
        dx = h
        while xs[-1] < r_out:
            dx = min(h_out, dx * growth)
            xs.append(xs[-1] + dx)
        xs = np.array(xs)
        if len(xs) > 2:
            xs = core[-1] + (xs - core[-1]) * (r_out - core[-1]) / (xs[-1] - core[-1])
        x = np.unique(np.concatenate([core, xs]))
    else:
        x = h * np.arange(int(np.ceil(r_out / h - 1e-9)) + 1)
    hp = x[-1] - x[-2]
    x = np.concatenate([x, x[-1] + hp * np.arange(1, n_pml + 1)])
    return np.concatenate([-x[:0:-1], x])


class Cartesian2DArm(Arm):
    """
    Cartesian FIT arm for `Scattering2DCase` (see the module docstring).

    Parameters
    ----------
    conformal : {None, 'all', 'tensor'}
        None: staircase; 'all': diagonal subpixel averaging.
    graded : bool
        Graded rectilinear grid (else uniform).
    growth : float
        Cell-size growth ratio of the graded region.
    ratio : float
        Largest graded size in units of h (h_out = min(ratio h,
        lambda_min,host / ppw_out)), so that the whole mesh refines with h.
    ppw_out : float
        Cells per shortest host wavelength in the coarse region.
    n_pml : int
        CPML cells.
    cfl : float
    ladder : sequence of float, optional
        Resolutions h [m]; default from the case.
    """

    def __init__(self, conformal="all", graded=True, growth=1.2, ppw_out=20, ratio=4.0,
                 n_pml=10, cfl=0.95, ladder=None, name=None):  # fmt: skip
        self.conformal = conformal
        self.graded = graded
        self.growth = growth
        self.ppw_out = ppw_out
        self.ratio = ratio
        self.n_pml = n_pml
        self.cfl = cfl
        self.ladder = ladder
        kind = "staircase" if conformal is None else f"conformal-{conformal}"
        self.name = name or f"cart-{kind}-{'graded' if graded else 'uniform'}"

    def supports(self, case):
        return hasattr(case, "cylinders")

    def resolutions(self, case):
        if self.ladder is not None:
            return list(self.ladder)
        a = min(cy["radius"] for cy in case.cylinders)
        if a < 1e-6 * 0.5:  # nanoparticles: h = a / 5 ... a / 40
            return [a / q for q in (5, 7, 10, 14, 20, 28, 40)]
        lam_in = np.min(case.wavelengths) / 2.0  # n = 2 (B1) or PEC (B5)
        return [lam_in / r for r in (10, 14, 20, 28, 40)]

    # ------------------------------------------------------------------
    def build(self, case, h):
        import pyvista as pv

        import wakis

        nb2 = case.n_background**2
        lam_host = np.min(case.wavelengths) / case.n_background
        h_out = max(h, min(self.ratio * h, lam_host / self.ppw_out))
        xmin = min(cy["center"][0] - cy["radius"] for cy in case.cylinders)
        xmax = max(cy["center"][0] + cy["radius"] for cy in case.cylinders)
        ymax = max(abs(cy["center"][1]) + cy["radius"] for cy in case.cylinders)
        fx = max(abs(xmin), abs(xmax)) + 2 * h
        fy = ymax + 2 * h
        r_out = max(case.r_absorber, max(case.circles) + 3 * h_out)
        xn = graded_axis(fx, h, h_out, r_out, self.n_pml, self.growth, self.graded)
        yn = graded_axis(fy, h, h_out, r_out, self.n_pml, self.growth, self.graded)
        dz = 10 * h
        solids, mats = {}, {}
        for i, cy in enumerate(case.cylinders):
            key = f"cyl{i}"
            cyl = pv.Cylinder(
                center=(cy["center"][0], cy["center"][1], 0.0), radius=cy["radius"],
                height=100 * dz, direction=(0, 0, 1), resolution=720,
            )  # fmt: skip
            solids[key] = cyl.triangulate()
            m = case.materials[cy["material"]]
            mats[key] = "pec" if isinstance(m, str) else m
        grid = wakis.GridFIT3D(
            x=xn, y=yn, z=np.array([-dz / 2, dz / 2]), stl_solids=solids,
            stl_materials=mats, conformal=self.conformal or False, verbose=0,
        )  # fmt: skip
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"], n_pml=self.n_pml, use_stl=True,
            cfln=self.cfl, bg=[nb2, 1.0, 0.0], verbose=0,
        )  # fmt: skip
        # TF/SF box: scatterers + margin, corners inside the smallest circle
        margin = max(2 * h, 3 * NM)
        half = np.array([fx - 2 * h + margin, fy - 2 * h + margin])
        rmin = min(case.circles)
        # bilinear stencils of the circle points must stay outside the box
        xc = rmin / np.sqrt(2)
        h_loc = np.diff(xn)[np.searchsorted(xn, xc) - 1]
        if np.hypot(*half) > rmin - 1.5 * h_loc:
            raise ValueError(
                f"TF/SF box corner {np.hypot(*half):.3g} m too close to the "
                f"extraction circle {rmin:.3g} m at h = {h:.3g} m"
            )
        d = case.direction
        if np.isclose(d, 0.0):
            direction = "+x"
        elif np.isclose(d, np.pi / 2):
            direction = "+y"
        else:
            raise NotImplementedError("TF/SF along +x or +y only")
        if case.polarization == "TM":
            pol = "z"
        else:  # E = z x k
            pol = "y" if direction == "+x" else "x"
        src = wakis.TFSFSource(
            case.source(), size=(2 * half[0], 2 * half[1], None), direction=direction,
            polarization=pol, amplitude=1.0, background_eps_r=nb2,
        )  # fmt: skip
        src.prepare(solver)
        return grid, solver, src, xn, yn

    def run(self, case, h):
        t0 = time.time()
        grid, solver, src, xn, yn = self.build(case, h)
        gx, gy = np.asarray(grid.x), np.asarray(grid.y)
        Nx, Ny, N = solver.Nx, solver.Ny, solver.N
        xm = 0.5 * (gx[:Nx] + gx[1 : Nx + 1])
        ym = 0.5 * (gy[:Ny] + gy[1 : Ny + 1])
        if case.polarization == "TM":
            xs, ys, fld = gx[:Nx], gy[:Ny], "E"
        else:
            xs, ys, fld = xm, ym, "H"
        pts = np.vstack([circle_points(r, case.n_points)[0] for r in case.circles])
        P = bilinear(xs, ys, pts, Nx)
        gap = case.gap_point is not None
        if gap:
            gpts = case.gap_points
            Px = bilinear(xm, gy[:Ny], gpts, Nx)
            Py = bilinear(gx[:Nx], ym, gpts, Nx)
        # incident line sample at the origin (on the propagation axis)
        axis = 0 if src.direction[1] == "x" else 1
        coords = gx if axis == 0 else gy
        i0 = int(np.argmin(np.abs(coords[: (Nx if axis == 0 else Ny)])))
        setup_s = time.time() - t0
        dt = solver.dt
        shift = 0.0 if case.polarization == "TM" else -0.5 * dt
        freqs = case.freqs
        acc = np.zeros((len(freqs), len(pts)), complex)
        acc_e = np.zeros((len(freqs), 2 * len(case.gap_points) if gap else 2), complex)
        line, times = [], []
        off = 2 * N

        def cb(s, t):
            ts = t + dt + shift
            a = (s.E if fld == "E" else s.H).array[off : off + Nx * Ny]
            acc[:] += np.outer(np.exp(-2j * np.pi * freqs * ts), P @ a) * dt
            if gap:
                E = s.E.array
                ex = Px @ E[0 : Nx * Ny]
                ey = Py @ E[N : N + Nx * Ny]
                acc_e[:] += (
                    np.outer(np.exp(-2j * np.pi * freqs * (t + dt)), np.r_[ex, ey]) * dt
                )
            times.append(t + dt)
            line.append(float(np.asarray(src._e)[i0 + src._offset]))

        n_steps = int(case.t_run / dt)
        t1 = time.time()
        solver.emsolve(n_steps, source=[src], callback=cb)
        run_s = time.time() - t1
        if not np.all(np.isfinite(acc)):
            raise FloatingPointError("unstable")
        times, line = np.array(times), np.array(line)
        u0 = (np.exp(-2j * np.pi * np.outer(freqs, times)) @ line) * dt
        e0 = u0.copy()
        if case.polarization == "TE":
            eta = np.sqrt(mu_0 / (eps_0 * case.n_background**2))
            u0 = u0 / eta
            # TF/SF E = +x (+y travel) gives H_z = -E/eta: sign of z x k
            if src.direction == "+y":
                u0 = -u0
        n2d = (len(xn) - 1) * (len(yn) - 1)
        out = dict(
            samples=acc.reshape(len(freqs), len(case.circles), case.n_points),
            u0=u0, radii=list(case.circles),
            cost=dict(unknowns=3 * n2d, cells=n2d, steps=n_steps, dt=dt,
                      run_s=run_s, setup_s=setup_s, h=h,
                      h_max=float(np.diff(xn).max())),
        )  # fmt: skip
        if gap:
            ng = len(case.gap_points)
            out["e_gap"] = np.stack([acc_e[:, :ng], acc_e[:, ng:]], 2)
            out["e0_gap"] = e0
        return out
