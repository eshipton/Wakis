# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Polygonal-FIT arms (`PolygonFIT2D`, diagonal/orthogonal Hodge).

* ``hybrid``: `hybrid_circles_mesh` -- h_out squares in the bulk and h
  squares inside each cylinder, a conforming Delaunay triangle patch only
  around the interfaces; narrow gaps between cylinders refined to
  ``gap_cells`` triangles across (``h >= h_min``); square domain
  |x|, |y| < r_absorber, split-field PML of ``n_pml`` h_out squares behind,
  PEC behind the PML.
* ``voronoi``: `voronoi_disk_mesh` -- Voronoi polygons of staggered
  generator rings (generator spacing sqrt(3) h(r), Voronoi edges ~ h),
  mirrored rings straddling the interface; single centred cylinder only;
  first-order ABC at r_absorber + n_pml h_out (no PML on polygons).

Both use the scattered-field plane wave of `PolygonFIT2D.add_plane_wave`
(dispersive scatterers driven by the total field). Gap fields (TE) are
reconstructed from the edge fluxes of the triangle that contains the gap
point with the lowest-order Raviart-Thomas functions.

The resolution label is h [m] on the scatterers (same ladder and h_out rule
as the triangle baseline).
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0

from ...cylinder_harmonics import circle_points
from ..runner import Arm


def _rt_matrix(mesh, point):
    """(2, n_edges) map edge fluxes -> D at `point` (triangle cells only)."""
    c = int(mesh.locate([point])[0])
    nodes = mesh.cell(c)
    if len(nodes) != 3:
        raise NotImplementedError("gap point must lie in a triangle")
    X = mesh.nodes[nodes]
    T = np.array([X[1] - X[0], X[2] - X[0]]).T
    lam12 = np.linalg.solve(T, np.asarray(point) - X[0])
    lam = np.r_[1 - lam12.sum(), lam12]
    Ginv = np.linalg.inv(T)  # rows: gradients of lambda_1, lambda_2
    grads = np.vstack([-Ginv.sum(0), Ginv])  # (3, 2)
    k = slice(mesh.cell_ptr[c], mesh.cell_ptr[c + 1])
    rows, cols, vals = [], [], []
    loc = {n: i for i, n in enumerate(nodes)}
    for e in mesh.corner_edge[k]:
        a, b = mesh.edges[e]
        ia, ib = loc[a], loc[b]
        W = lam[ia] * grads[ib] - lam[ib] * grads[ia]  # Whitney 1-form, tangent a->b
        RT = np.array([W[1], -W[0]])  # flux function along n = t x z
        rows += [0, 1]
        cols += [e, e]
        vals += list(RT)
    return sp.csr_matrix((vals, (rows, cols)), shape=(2, mesh.n_edges))


class PolygonArm(Arm):
    """
    Polygonal FIT arm for `Scattering2DCase`.

    Parameters
    ----------
    kind : {'hybrid', 'voronoi'}
    growth, ratio, ppw_out : float
        Size field: h on the cylinders, geometric growth to
        h_out = max(h, min(ratio h, lambda_min,host / ppw_out)).
    n_pml : int
    gap_cells : float
        Cells across narrow gaps (hybrid only, None: off).
    cfl : float
    ladder : sequence of float, optional
    """

    def __init__(self, kind="hybrid", growth=1.2, ratio=4.0, ppw_out=20, n_pml=10,
                 gap_cells=4.0, cfl=0.95, ladder=None, name=None):  # fmt: skip
        self.kind = kind
        self.growth = growth
        self.ratio = ratio
        self.ppw_out = ppw_out
        self.n_pml = n_pml
        self.gap_cells = gap_cells
        self.cfl = cfl
        self.ladder = ladder
        self.name = name or f"poly-{kind}"

    def supports(self, case):
        if not hasattr(case, "cylinders") or any(
            isinstance(m, str) for m in case.materials.values()
        ):
            return False
        if self.kind == "voronoi":
            return len(case.cylinders) == 1 and np.allclose(case.cylinders[0]["center"], 0)
        return True

    def resolutions(self, case):
        if self.ladder is not None:
            return list(self.ladder)
        a = min(cy["radius"] for cy in case.cylinders)
        if a < 0.5e-6:
            return [a / q for q in (5, 7, 10, 14, 20, 28, 40)]
        lam_in = np.min(case.wavelengths) / 2.0
        return [lam_in / r for r in (10, 14, 20, 28, 40)]

    def build(self, case, h):
        from ...unstructured.polymeshers import hybrid_circles_mesh, voronoi_disk_mesh
        from ...unstructured.polysolver2d import PolygonFIT2D

        nb2 = case.n_background**2
        lam_host = np.min(case.wavelengths) / case.n_background
        h_out = max(h, min(self.ratio * h, lam_host / self.ppw_out))
        r0 = max(case.r_absorber, max(case.circles) + 2 * h_out)
        circles = [(*cy["center"], cy["radius"]) for cy in case.cylinders]
        ncyl = len(circles)
        if self.kind == "hybrid":
            gaps = []
            for i in range(ncyl):
                for j in range(i + 1, ncyl):
                    d = np.hypot(*np.subtract(circles[i][:2], circles[j][:2]))
                    gaps.append(d - circles[i][2] - circles[j][2])
            gap_cells = self.gap_cells if gaps else None
            h_min = min(gaps) / self.gap_cells if gaps and self.gap_cells else None
            mesh, info = hybrid_circles_mesh(
                circles, h, h_out, r0, growth=self.growth, pml_cells=self.n_pml,
                gap_cells=gap_cells, h_min=h_min,
            )  # fmt: skip
            L = info["half_width"]
            kw = dict(
                bc="pec", hodge="orthogonal",
                pml=dict(x=(-L, L), y=(-L, L), thickness=self.n_pml * h_out, eps_r=nb2),
            )  # fmt: skip
            mats = {ncyl: [nb2, 1.0]}
            for j, cy in enumerate(case.cylinders):
                mats[j] = case.materials[cy["material"]]
        else:
            a = case.cylinders[0]["radius"]
            s_in = np.sqrt(3) * h
            band = 2 * h

            def size(r):
                d = max(abs(r - a) - band, 0.0)
                return s_in * min(h_out / h, self.growth ** (d / h)) if r > a else s_in

            mesh = voronoi_disk_mesh([a, r0 + self.n_pml * h_out], size, collapse=0.1, lloyd=10)
            kw = dict(bc="abc", hodge="orthogonal", ortho_tol=np.inf)
            mats = {1: [nb2, 1.0], 0: case.materials[case.cylinders[0]["material"]]}
        # the explicit ADEs of the metal can limit dt below the leapfrog CFL
        # (von Neumann check of DispersiveMaterial.is_stable): lower cfl by
        # 15 % steps until stable (recorded in the cost as `cfl`)
        cfl = self.cfl
        for _ in range(20):
            try:
                solver = PolygonFIT2D(mesh, mats, polarization=case.polarization, cfl=cfl, **kw)
                break
            except ValueError as e:
                if "unstable" not in str(e):
                    raise
                cfl *= 0.85
        else:
            raise RuntimeError("no stable dt for the dispersive materials")
        self._cfl_used = cfl
        d = case.direction
        solver.add_plane_wave(case.source(), direction=(np.cos(d), np.sin(d)),
                              background_eps_r=nb2)  # fmt: skip
        return mesh, solver, h_out

    def run(self, case, h):
        t0 = time.time()
        mesh, solver, h_out = self.build(case, h)
        pts = np.vstack([circle_points(r, case.n_points)[0] for r in case.circles])
        P = solver._interp(pts)
        gap = case.gap_point is not None
        if gap:
            if case.polarization != "TE":
                raise NotImplementedError("gap field: TE only")
            Pg = _rt_matrix(mesh, case.gap_point) / (eps_0 * case.n_background**2)
        setup_s = time.time() - t0
        dt = solver.dt
        shift = 0.0 if case.polarization == "TM" else -0.5 * dt
        freqs = case.freqs
        acc = np.zeros((len(freqs), len(pts)), complex)
        acc_e = np.zeros((len(freqs), 2), complex)
        n_steps = int(case.t_run / dt)
        times = (np.arange(n_steps) + 1) * dt
        d = case.direction
        pvec = np.array([-np.sin(d), np.cos(d)])
        t1 = time.time()
        for n in range(n_steps):
            solver.one_step()
            acc += np.outer(np.exp(-2j * np.pi * freqs * (times[n] + shift)), P @ solver.u) * dt
            if gap:
                es = Pg @ solver.v
                einc = solver._incident_signal(
                    np.atleast_1d(np.asarray(case.gap_point) @ solver.plane_wave["k"]),
                    times[n],
                )[0]
                acc_e += np.outer(np.exp(-2j * np.pi * freqs * times[n]), es + einc * pvec) * dt
            if n % 4000 == 0 and not np.all(np.isfinite(solver.u)):
                raise FloatingPointError("unstable")
        run_s = time.time() - t1
        u_inc0 = solver.incident_field(np.zeros((1, 2)), times + shift)
        u0 = (np.exp(-2j * np.pi * np.outer(freqs, times + shift)) @ u_inc0) * dt
        out = dict(
            samples=acc.reshape(len(freqs), len(case.circles), case.n_points),
            u0=u0, radii=list(case.circles),
            cost=dict(unknowns=solver.n_unknowns, nodes=int(solver.free.size),
                      cells=int(mesh.n_cells), steps=n_steps, dt=dt, run_s=run_s,
                      setup_s=setup_s, h=h, h_max=h_out,
                      h_min=float(mesh.edge_length.min()), cfl=self._cfl_used,
                      cells_by_size=np.bincount(mesh.cell_size).tolist()),
        )  # fmt: skip
        if gap:
            out["e_gap"] = acc_e
            ui = solver._incident_signal(np.zeros(1), times)
            out["e0_gap"] = (np.exp(-2j * np.pi * np.outer(freqs, times)) @ ui) * dt
        return out
