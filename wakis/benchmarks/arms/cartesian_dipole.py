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
* the two wire halves: ``conformal='edges'`` (recommended staircase: an E
  edge is PEC when its midpoint is in the wire, as `wakis.templates`),
  ``False`` (legacy STL cell masks: a cell touched by the wire becomes PEC,
  which thickens and lengthens the wire by up to a cell; kept as a
  documented failure mode) or ``'pec'`` (conformal PEC from the STL);
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


class HalfCapsule:
    """
    Half of a round-ended wire along z: a cylinder of radius `radius` from the
    flat gap face `z_flat` to `z_tip - sign * radius`, closed by a hemisphere
    of radius `radius` centred there so that the tip reaches `z_tip`.
    """

    def __init__(self, z_flat, z_tip, radius, material="pec"):
        self.z_flat = float(z_flat)
        self.z_tip = float(z_tip)
        self.radius = float(radius)
        self.material = material
        lo_z, hi_z = sorted((self.z_flat, self.z_tip))
        self.center = np.array([0.0, 0.0, 0.5 * (lo_z + hi_z)])
        self.bounds = (
            np.array([-self.radius, -self.radius, lo_z]),
            np.array([self.radius, self.radius, hi_z]),
        )

    def weight(self, x, y, z, tol):
        """
        Fill weight at broadcast coordinates (x, y, z): 1 inside, 0.5 on the
        surface within `tol`, 0 outside (same interface as `Cylinder`).
        """
        s = 1.0 if self.z_tip >= self.z_flat else -1.0
        u = s * z
        u_flat = s * self.z_flat
        u_cap = s * self.z_tip - self.radius
        rho = np.hypot(x, y)
        r_eff = np.hypot(rho, np.maximum(u - u_cap, 0.0))
        d = np.minimum(self.radius - r_eff, u - u_flat)
        return np.where(d > tol, 1.0, np.where(d >= -tol, 0.5, 0.0))


class CartesianDipoleArm(Arm):
    def __init__(self, conformal=False, ladder=None, n_pml=10, cfl=0.95,
                 n2f=True, pad_xy=0.21, pad_z=0.14, graded=True, hz_ratio=4.0, ppw_out=20,
                 use_gpu=False, port="axis", ends="flat",
                 name=None):  # fmt: skip
        if port not in ("axis", "sheet"):
            raise ValueError(f"port must be 'axis' or 'sheet', got {port!r}.")
        if ends not in ("flat", "round"):
            raise ValueError(f"ends must be 'flat' or 'round', got {ends!r}.")
        if ends == "round" and conformal != "edges":
            raise ValueError("ends='round' requires conformal='edges'.")
        self.pad_xy, self.pad_z = pad_xy, pad_z
        self.graded = graded
        self.hz_ratio = hz_ratio
        self.ppw_out = ppw_out
        self.use_gpu = use_gpu
        self.port = port
        self.ends = ends
        self.conformal = conformal
        self.ladder = ladder
        self.n_pml = n_pml
        self.cfl = cfl
        self.n2f = n2f
        kind = {False: "staircase-stl", "edges": "staircase-edges"}.get(
            conformal, f"conformal-{conformal}"
        )
        suffix = ("-sheet" if port == "sheet" else "") + (
            "-round" if ends == "round" else ""
        )
        self.name = name or f"cart-{kind}-{'graded' if graded else 'uniform'}{suffix}"

    def supports(self, case):
        return case.name.startswith("B4")

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
        if self.conformal == "edges":
            # Edge-wise staircase: an E edge is PEC when its midpoint lies in
            # (or on) the wire (wakis.templates.shapes.assign_materials).
            grid = wakis.GridFIT3D(x=xn, y=xn.copy(), z=zn, verbose=0)
        else:
            # STL solids. Legacy staircase masks mark a cell when any of its
            # 8 nodes is inside and zero its edges, which thickens the wire
            # by up to a cell; there both halves stop 0.01 h_z short of the
            # gap planes so that the gap cells stay free. Conformal PEC
            # resolves that sliver as a real 0.01 h_z series gap (a few
            # 100 Ohm capacitive), so there the halves end on the gap planes
            # (node planes: the port edges stay open, the wire edges closed).
            sliver = 0.0 if self.conformal else 0.01 * hz
            for s, name in ((1, "top"), (-1, "bottom")):
                z_in = g / 2 + sliver
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
            use_stl=bool(solids), cfln=self.cfl, use_gpu=self.use_gpu, verbose=0,
        )  # fmt: skip
        if self.conformal == "edges":
            from ...templates.shapes import Cylinder, assign_materials

            if self.ends == "round":
                halves = [HalfCapsule(s * g / 2, s * L / 2, a, "pec") for s in (1, -1)]
            else:
                arm_len = L / 2 - g / 2
                halves = [Cylinder((0, 0, s * (g / 2 + arm_len / 2)), a, arm_len, "pec")
                          for s in (1, -1)]  # fmt: skip
            assign_materials(solver, halves, case.freqs.mean())
        if self.port == "sheet":
            s = float(np.max(xn[xn <= a / np.sqrt(2) + 1e-12 * ht]))
            port_size = (2 * s, 2 * s, g)
        else:
            s = 0.0
            port_size = (0, 0, g)
        port = wakis.LumpedPort(
            case.source(), component="z", center=(0, 0, 0), size=port_size,
            impedance=case.z0,
        )  # fmt: skip
        port._set_edges(solver)
        (ix, iy, iz), (cx, cy, _) = grid.region_indices(
            "E", "z", port.center, port.size
        )
        ie = solver.ieps.toarray()
        ie = ie.get() if hasattr(ie, "get") else np.asarray(ie)
        ie_z = ie[2 * solver.N : 3 * solver.N].reshape(
            (solver.Nx, solver.Ny, solver.Nz), order="F"
        )
        assert np.all(ie_z[np.ix_(ix, iy, iz)] != 0)
        assert np.all(np.hypot(cx[:, None], cy[None, :]) <= a + 1e-12)
        assert np.all(ie_z[np.ix_(ix, iy, [iz[0] - 1, iz[-1] + 1])] == 0)
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
        return (
            grid,
            solver,
            port,
            mons,
            dict(hz=hz, h_out=h_out, port=self.port, ends=self.ends, port_s=s),
        )

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
                           run_s=run_s, setup_s=setup_s, h=ht,
                           device="gpu" if self.use_gpu else "cpu", **info)  # fmt: skip
        return out
