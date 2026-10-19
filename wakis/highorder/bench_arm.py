# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Benchmark-suite arm (`wakis.benchmarks.Arm`) for the high-order DGTD solver
(`wakis.highorder.DGTD2D`) on 2D scattering cases (B1, B2, B3, B5 without PEC).

* mesh: `circles_mesh`, element size h on the cylinders, growing linearly
  (rate `growth` per unit distance) to ``ratio * h`` (capped at
  lambda_min,host / `epw_out` elements per wavelength); for cases with a gap
  point the size is further limited to ``gap_h + growth_gap * distance`` to
  the gap point, so that the gap holds at least one element;
* curved (isoparametric) elements on every cylinder;
* boundary: 'abc' = disk with a curved first-order Silver-Mueller boundary
  at r = max(circles) + margin (only with de-embedding, symmetric cases),
  'pml' = square box with a CFS stretched-coordinate PML starting at
  ``case.r_absorber``;
* scattered-field plane wave (dispersive scatterers driven by the total
  field through their ADEs); all fields live at the same time levels (RK);
* gap E for B3: polynomial evaluation of (Ex, Ey) = -(vx, vy) at
  ``case.gap_points`` (centre + 8 points on a circle of radius g/4) plus the
  analytic incident field; ``e_gap`` has shape (n_freq, n_points, 2).

The resolution label is h [m] on the scatterers; the arm's degree is p.
"""

import time

import numpy as np

from ..benchmarks.runner import Arm
from ..cylinder_harmonics import circle_points
from .dgtd2d import DGTD2D
from .mesh2d import circles_mesh


class DGTD2DArm(Arm):
    def __init__(self, p=4, boundary="pml", ratio=3.0, growth=0.25, epw_out=None,
                 pml_thickness=None, curved=True, cfl=0.8, ladder=None, name=None):  # fmt: skip
        self.p = p
        self.boundary = boundary
        self.ratio = ratio
        self.growth = growth
        self.epw_out = epw_out if epw_out is not None else max(1.0, 0.6 * (p + 1) / 2)
        self.pml_thickness = pml_thickness
        self.curved = curved
        self.cfl = cfl
        self.ladder = ladder
        self.name = name or f"dg{p}-{boundary}" + ("" if curved else "-straight")

    def supports(self, case):
        return hasattr(case, "cylinders") and not any(
            isinstance(m, str) for m in case.materials.values()
        )

    def resolutions(self, case):
        if self.ladder is not None:
            return list(self.ladder)
        a = min(cy["radius"] for cy in case.cylinders)
        if a < 0.5e-6:
            return [a / q for q in (2, 3, 4, 6, 8)]
        lam_in = np.min(case.wavelengths) / 2.0
        return [lam_in / r for r in (1.5, 2, 3, 4, 6)]

    def _size(self, case, h):
        lam_host = np.min(case.wavelengths) / case.n_background
        h_out = max(h, min(self.ratio * h, lam_host / self.epw_out))
        cyl = case.cylinders
        gp = case.gap_point
        if gp is not None:
            gap = min(
                np.hypot(*(np.subtract(c1["center"], c2["center"])))
                - c1["radius"] - c2["radius"]
                for i, c1 in enumerate(cyl) for c2 in cyl[i + 1:]
            )  # fmt: skip
            h_gap = min(h, gap)
        g = self.growth

        def size(P):
            d = np.full(len(P), np.inf)
            for cy in cyl:
                r = np.hypot(P[:, 0] - cy["center"][0], P[:, 1] - cy["center"][1])
                d = np.minimum(d, np.maximum(r - cy["radius"], 0))
            s = np.minimum(h_out, h + g * d)
            if gp is not None:
                dg = np.hypot(P[:, 0] - gp[0], P[:, 1] - gp[1])
                s = np.minimum(s, h_gap + 0.5 * dg)
            return s

        return size, h_out

    def build(self, case, h):
        size, h_out = self._size(case, h)
        circles = [(cy["center"][0], cy["center"][1], cy["radius"]) for cy in case.cylinders]
        nb2 = case.n_background**2
        mats = {j + 1: case.materials[cy["material"]] for j, cy in enumerate(case.cylinders)}
        if self.boundary == "abc":
            r_out = max(case.circles) + max(h_out, 0.1 * max(case.circles))
            mesh = circles_mesh(circles, size, outer_circle=(0.0, 0.0, r_out))
            solver = DGTD2D(mesh, self.p, mats, polarization=case.polarization, bc="abc",
                            curved=self.curved, cfl=self.cfl, background_eps_r=nb2)  # fmt: skip
        else:
            lam_host = np.max(case.wavelengths) / case.n_background
            d = self.pml_thickness or max(2 * h_out, 0.25 * lam_host / 2)
            H = case.r_absorber + d
            mesh = circles_mesh(circles, size, box=(-H, H, -H, H))
            solver = DGTD2D(mesh, self.p, mats, polarization=case.polarization, bc="abc",
                            pml=dict(thickness=d, R0=1e-6, order=3), curved=self.curved,
                            cfl=self.cfl, background_eps_r=nb2)  # fmt: skip
        dr = case.direction
        solver.add_plane_wave(case.source(), direction=(np.cos(dr), np.sin(dr)))
        return mesh, solver, h_out

    def run(self, case, h):
        t0 = time.time()
        mesh, solver, h_out = self.build(case, h)
        pts = np.vstack([circle_points(r, case.n_points)[0] for r in case.circles])
        P = solver.interp_matrix(pts)
        gap = case.gap_point is not None
        if gap:
            gpts = np.atleast_2d(getattr(case, "gap_points", case.gap_point))
            Pg = solver.interp_matrix(gpts)
        setup_s = time.time() - t0
        dt = solver.dt
        freqs = case.freqs
        n_steps = int(case.t_run / dt)
        times = (np.arange(n_steps) + 1) * dt
        kern = np.exp(-2j * np.pi * np.outer(times, freqs))
        acc = np.zeros((len(freqs), len(pts)), complex)
        acc_e = None
        d = case.direction
        pvec = np.array([-np.sin(d), np.cos(d)])
        amp_e = 1.0  # incident E amplitude of add_plane_wave
        t1 = time.time()
        for n in range(n_steps):
            solver.one_step()
            acc += np.outer(kern[n], P @ solver.u.ravel()) * dt
            if gap:
                vx = solver._view(solver.X, "vx").ravel()
                vy = solver._view(solver.X, "vy").ravel()
                es = -np.stack([Pg @ vx, Pg @ vy], 1)  # (n_gap, 2); TE: v = -E
                ui = solver.incident_field(gpts, times[n])[0]  # (n_gap,)
                e_tot = es + (ui * (amp_e / solver.plane_wave["amp"]))[:, None] * pvec
                if acc_e is None:
                    acc_e = np.zeros((len(freqs),) + e_tot.shape, complex)
                acc_e += kern[n][:, None, None] * e_tot[None] * dt
            if n % 4000 == 0 and not np.all(np.isfinite(solver.X)):
                raise FloatingPointError("unstable")
        run_s = time.time() - t1
        u_inc0 = solver.incident_field(np.zeros((1, 2)), times)[:, 0]
        u0 = (kern.T @ u_inc0) * dt
        out = dict(
            samples=acc.reshape(len(freqs), len(case.circles), case.n_points),
            u0=u0, radii=list(case.circles),
            cost=dict(unknowns=int(solver.n_dofs), state=int(solver.n_state),
                      cells=int(solver.K), curved=int(solver.n_curved), p=self.p,
                      steps=n_steps, rhs_evals=5 * n_steps, dt=dt, run_s=run_s,
                      setup_s=setup_s, h=h, h_max=h_out),
        )  # fmt: skip
        if gap:
            out["e_gap"] = acc_e
            ui = solver.incident_field(np.zeros((1, 2)), times)[:, 0]
            ui = ui * (amp_e / solver.plane_wave["amp"])
            out["e0_gap"] = (kern.T @ ui) * dt
        return out
