# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Baseline arm for B4: Cartesian FIT (`SolverFIT3D`) dipole with a lumped port.

* domain: CPML from 0.21 lambda_max (transverse) and 0.14 lambda_max
  beyond the wire ends;
* rectilinear grid graded in x and y (h_t on |x|, |y| <= a + 2 h_t around the
  wire axis, which lies on a node line, growth 1.2 to lambda_min / 20) and
  uniform h_z = min(g, 4 h_t) along the wire (+5 cells), graded beyond;
* the two wire halves as PEC STL cylinders, staircase (``conformal=False``)
  or conformal PEC (``conformal='pec'``);
* `LumpedPort` (Z0 = 50 Ohm) on the z edges of the axis across the gap;
* CPML; `Near2FarMonitor` box for the broadside directivity.

The resolution label is h_t [m], the transverse cell at the wire.
"""

import time

import numpy as np
from scipy.constants import c as c_light

from ..runner import Arm
from .cartesian2d import graded_axis


def _axis_z(L, g, hz, h_out, r_out, n_pml, growth=1.2):
    # nodes at +-g/2 + k hz (the gap is a whole number of cells)
    half = L / 2 + 5 * hz
    n = int(np.ceil((half - g / 2) / hz))
    m = int(round(g / hz))
    core = np.r_[-g / 2 + hz * np.arange(m + 1), g / 2 + hz * np.arange(1, n + 1)]
    core = core[core >= -1e-12 * hz]
    xs, dx = [core[-1]], hz
    while xs[-1] < r_out:
        dx = min(h_out, dx * growth)
        xs.append(xs[-1] + dx)
    x = np.unique(np.concatenate([core, xs]))
    hp = x[-1] - x[-2]
    x = np.concatenate([x, x[-1] + hp * np.arange(1, n_pml + 1)])
    if x[0] > 1e-12 * hz:  # odd number of gap cells: no node at 0
        return np.concatenate([-x[::-1], x])
    return np.concatenate([-x[:0:-1], x])


class CartesianDipoleArm(Arm):
    def __init__(self, conformal=False, ladder=None, n_pml=10, cfl=0.95,
                 n2f=True, pad_xy=0.21, pad_z=0.14, graded=True, hz_ratio=4.0, ppw_out=20,
                 name=None):  # fmt: skip
        self.pad_xy, self.pad_z = pad_xy, pad_z
        self.graded = graded
        self.hz_ratio = hz_ratio
        self.ppw_out = ppw_out
        self.conformal = conformal
        self.ladder = ladder
        self.n_pml = n_pml
        self.cfl = cfl
        self.n2f = n2f
        kind = "staircase" if not conformal else f"conformal-{conformal}"
        self.name = name or f"cart-{kind}-{'graded' if graded else 'uniform'}"

    def supports(self, case):
        return case.name == "B4"

    def resolutions(self, case):
        if self.ladder is not None:
            return list(self.ladder)
        return [case.radius / q for q in (1, 1.5, 2, 3, 4)]

    def build(self, case, ht):
        import pyvista as pv

        import wakis

        L, a, g = case.length, case.radius, case.gap
        lam_min = c_light / case.freqs.max()
        h_out = lam_min / self.ppw_out if self.graded else ht
        hz = min(g, self.hz_ratio * ht) if self.graded else ht
        hz = g / np.ceil(g / hz - 1e-9)  # the gap is an integer number of cells
        lam_max = c_light / case.freqs.min()
        xn = graded_axis(a + 2 * ht, ht, h_out, self.pad_xy * lam_max, self.n_pml)
        zn = _axis_z(L, g, hz, h_out, L / 2 + self.pad_z * lam_max, self.n_pml)
        solids, mats = {}, {}
        # Staircase (legacy) masks mark a cell when any of its 8 nodes is
        # inside the solid, and its edges become PEC: both halves stop
        # 0.01 h_z short of the gap planes so that the gap cells stay free
        # (the physical gap is still g up to 2 % of a cell).
        for s, name in ((1, "top"), (-1, "bottom")):
            z_in = g / 2 + 0.01 * hz
            zc = s * (z_in + (L / 2 - z_in) / 2)
            cyl = pv.Cylinder(center=(0, 0, zc), radius=a, height=L / 2 - z_in,
                              direction=(0, 0, 1), resolution=180)  # fmt: skip
            solids[name] = cyl.triangulate()
            mats[name] = "pec"
        grid = wakis.GridFIT3D(
            x=xn, y=xn.copy(), z=zn, stl_solids=solids, stl_materials=mats,
            conformal=self.conformal, verbose=0,
        )  # fmt: skip
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=self.n_pml,
            use_stl=True, cfln=self.cfl, verbose=0,
        )  # fmt: skip
        port = wakis.LumpedPort(
            case.source(), component="z", center=(0, 0, 0), size=(0, 0, g),
            impedance=case.z0,
        )  # fmt: skip
        mons = []
        if self.n2f:
            n2f_f = np.linspace(case.freqs.min(), case.freqs.max(), 11)
            mon = wakis.Near2FarMonitor(
                n2f_f,
                center=(0, 0, 0),
                size=(
                    1.3 * self.pad_xy * lam_max,
                    1.3 * self.pad_xy * lam_max,
                    L + self.pad_z * lam_max,
                ),
            )
            mons.append(mon)
        return grid, solver, port, mons, dict(hz=hz, h_out=h_out)

    def run(self, case, ht):
        t0 = time.time()
        grid, solver, port, mons, info = self.build(case, ht)
        setup_s = time.time() - t0
        n_steps = int(case.t_run / solver.dt)
        t1 = time.time()
        solver.emsolve(n_steps, source=[port], monitors=mons)
        run_s = time.time() - t1
        z = port.impedance_in(case.freqs)
        out = dict(z_in=z)
        if mons:
            m = mons[0]
            out["directivity"] = np.asarray(m.directivity(np.pi / 2, 0.0)).ravel()
            out["d_freqs"] = np.asarray(m.frequencies)
        n = int(grid.Nx * grid.Ny * grid.Nz)
        out["cost"] = dict(unknowns=3 * n, cells=n, steps=n_steps, dt=solver.dt,
                           run_s=run_s, setup_s=setup_s, h=ht, **info)  # fmt: skip
        return out
