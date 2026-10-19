# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Arm: Cartesian FIT with nested local refinement (`wakis.subgrid`).

* mesh: a uniform coarse grid of size H = h 2^L (the largest such size not
  above lambda_min,host / ppw_out and max_ratio h) refined by 2:1 levels:
  a cell of size s is split if it lies within ``band`` cells (of its own
  size) of a cylinder surface, or inside a cylinder while s > h. With
  ``gap_cells`` > 0, cells in a narrow gap between two cylinders are refined
  further until ``gap_cells`` cells span the local gap width (at most
  ``gap_levels`` extra levels): the local feature, not the surface size,
  sets the size there;
* `SubgridFIT2D`: symmetric hanging-node interface, sub-pixel averaging,
  scattered-field plane wave in the host, split-field PML (``n_pml`` coarse
  cells) behind the case's absorber radius, Drude/Lorentz ADE;
* optional two-level LTS (``lts=p``);
* DFT of the scattered node field (E_z or H_z, integer time levels) on the
  case circles (bilinear on the leaf cells), incident phasor at the origin at
  the same times; gap cases: total in-plane E at the gap point from the
  dual-edge voltages (half time levels).

The resolution label is h [m] on the scatterer surfaces.
"""

import time

import numpy as np

from ...cylinder_harmonics import circle_points
from ..runner import Arm


class Subgrid2DArm(Arm):
    """
    Subgridded Cartesian FIT arm for `Scattering2DCase`.

    Parameters
    ----------
    ppw_out : float
        Coarse cells per host wavelength (at the shortest wavelength).
    max_ratio : int
        Largest coarse/fine size ratio (power of 2).
    band : float
        Width of the refined shell around surfaces, in cells of each level.
    gap_cells : float
        Cells across a gap between two cylinders (0: no gap refinement).
    gap_levels : int
        Extra levels allowed for the gap refinement.
    gap_zone : float
        If > 0 (instead of `gap_cells`): every cell whose local gap width
        (sum of the distances to the two nearest surfaces) is below
        ``gap_zone`` times the smallest gap of the case is refined to
        h / 2**gap_levels: the facing surfaces and the gap get gap_levels
        more levels than the rest of the metal.
    n_pml : int
    cfl : float
    lts : int, optional
        Two-level LTS ratio.
    ladder : sequence of float, optional
    name : str, optional
    """

    def __init__(self, ppw_out=20, max_ratio=16, band=2.0, gap_cells=0, gap_levels=2,
                 gap_zone=0.0,
                 n_pml=10, cfl=0.95, lts=None, n_sub=8, ladder=None, name=None):  # fmt: skip
        self.ppw_out = ppw_out
        self.max_ratio = int(max_ratio)
        self.band = band
        self.gap_cells = gap_cells
        self.gap_levels = int(gap_levels)
        self.gap_zone = float(gap_zone)
        self.n_pml = n_pml
        self.cfl = cfl
        self.lts = lts
        self.n_sub = n_sub
        self.ladder = ladder
        self.name = name or ("sg" + (f"-gap{gap_cells:g}" if gap_cells else "")
                             + (f"-zone{gap_zone:g}L{gap_levels}" if gap_zone else "")
                             + (f"-lts{lts}" if lts else ""))  # fmt: skip

    def supports(self, case):
        return hasattr(case, "cylinders") and not any(
            isinstance(m, str) for m in case.materials.values()
        )

    def resolutions(self, case):
        if self.ladder is not None:
            return list(self.ladder)
        a = min(cy["radius"] for cy in case.cylinders)
        if a < 0.5e-6:
            return [a / q for q in (5, 7, 10, 14, 20, 28, 40)]
        lam_in = np.min(case.wavelengths) / 2.0
        return [lam_in / r for r in (10, 14, 20, 28, 40)]

    # ------------------------------------------------------------------
    def build(self, case, h):
        from ...subgrid import Circle, Geometry, QuadMesh, SubgridFIT2D

        lam_host = np.min(case.wavelengths) / case.n_background
        ratio = 1
        while 2 * ratio * h <= min(lam_host / self.ppw_out, self.max_ratio * h) * (
            1 + 1e-9
        ):
            ratio *= 2
        levels = int(round(np.log2(ratio)))
        H = ratio * h
        r0 = max(case.r_absorber, max(case.circles) + 2 * H)
        m_in = int(np.ceil(r0 / H - 1e-9))
        M = m_in + self.n_pml
        xc = H * np.arange(-M, M + 1)
        cyl = case.cylinders
        cen = np.array([c["center"] for c in cyl], float)
        rad = np.array([c["radius"] for c in cyl], float)
        multi = len(cyl) > 1
        gap_lv = self.gap_levels if ((self.gap_cells or self.gap_zone) and multi) else 0
        g_min = np.inf
        for i in range(len(cyl)):
            for j in range(i + 1, len(cyl)):
                g_min = min(g_min, np.hypot(*(cen[i] - cen[j])) - rad[i] - rad[j])
        max_level = levels + gap_lv

        def surf_dist(xm, ym):
            d = np.hypot(xm[..., None] - cen[:, 0], ym[..., None] - cen[:, 1]) - rad
            return d  # (..., n_cyl), negative inside

        def refine(level, x0, x1, y0, y1):
            s = x1 - x0
            xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            hd = 0.5 * np.hypot(s, y1 - y0)
            d = surf_dist(xm, ym)
            near = (np.abs(d) < (hd + self.band * s)[..., None]).any(-1)
            inside = (d < 0).any(-1)
            out = (near | inside) & (s > h * (1 + 1e-9))
            if gap_lv:
                # local gap width: distances to the two nearest surfaces
                ds = np.sort(np.maximum(np.abs(d) - hd[..., None], 0), -1)
                width = ds[..., 0] + ds[..., 1]
                h_gap = h / 2**gap_lv
                if self.gap_zone:
                    out |= (width < self.gap_zone * g_min) & (s > h_gap * (1 + 1e-9))
                else:
                    need = width / self.gap_cells
                    out |= s > np.maximum(need, h_gap) * (1 + 1e-9)
            return out

        mesh = QuadMesh(xc, xc.copy(), ratio=2, max_level=max_level, refine=refine)
        shapes = []
        for c in cyl:
            mat = case.materials[c["material"]]
            if not hasattr(mat, "poles"):
                mat = float(mat[0])
            shapes.append(Circle(c["center"], c["radius"], mat))
        # one Python object per material key so that groups are shared
        geo = Geometry(shapes, background_eps=case.n_background**2)
        box = (-m_in * H, m_in * H, -m_in * H, m_in * H)
        solver = SubgridFIT2D(mesh, geo, polarization=case.polarization,
                              pml=dict(box=box), cfl=self.cfl, n_sub=self.n_sub,
                              lts=self.lts)  # fmt: skip
        d = case.direction
        solver.add_plane_wave(case.source(), direction=(np.cos(d), np.sin(d)))
        return mesh, solver, H

    def run(self, case, h):
        t0 = time.time()
        mesh, solver, H = self.build(case, h)
        pts = np.vstack([circle_points(r, case.n_points)[0] for r in case.circles])
        P = solver.interpolation(pts)
        gap = case.gap_point is not None
        if gap:
            if case.polarization != "TE":
                raise ValueError("gap field implemented for TE")
            gp_case = getattr(case, "gap_points", None)
            gpts = np.atleast_2d(gp_case if gp_case is not None else [case.gap_point])
            Mx, My = solver.edge_field_matrices(gpts)
            ng = len(gpts)
        setup_s = time.time() - t0
        dt = solver.dt
        freqs = case.freqs
        acc = np.zeros((len(freqs), len(pts)), complex)
        acc_e = np.zeros((len(freqs), ng if gap else 1, 2), complex)
        n_steps = int(case.t_run / dt)
        times = (np.arange(n_steps) + 1) * dt
        d = case.direction
        pvec = np.array([-np.sin(d), np.cos(d)])  # z x k
        gp = gpts if gap else None
        t1 = time.time()
        for n in range(n_steps):
            solver.one_step()
            acc += np.outer(np.exp(-2j * np.pi * freqs * times[n]), P @ solver.u) * dt
            if gap:
                th = times[n] - 0.5 * dt  # g lives at t^{n+1/2}
                es = np.stack([Mx @ solver.g, My @ solver.g], 1)  # (ng, 2)
                einc = solver.eta_b * solver.incident(gp, th)[0]  # (ng,)
                et = es + einc[:, None] * pvec
                ph = np.exp(-2j * np.pi * freqs * th)
                acc_e += ph[:, None, None] * et[None] * dt
            if n % 4000 == 0 and not np.all(np.isfinite(solver.u)):
                raise FloatingPointError("unstable")
        run_s = time.time() - t1
        u_inc0 = solver.incident(np.zeros((1, 2)), times)[:, 0]
        u0 = (np.exp(-2j * np.pi * np.outer(freqs, times)) @ u_inc0) * dt
        lc = mesh.level_counts()
        out = dict(
            samples=acc.reshape(len(freqs), len(case.circles), case.n_points),
            u0=u0, radii=list(case.circles),
            cost=dict(unknowns=int(solver.n_unknowns), steps=n_steps, dt=dt,
                      run_s=run_s, setup_s=setup_s, cells=int(mesh.n_cells),
                      cells_per_level=lc.tolist(), h_min=float(np.min(
                          mesh.cells["x1"] - mesh.cells["x0"])), h_coarse=H,
                      lts=(None if solver.lts_info is None else dict(
                          p=int(self.lts),
                          fine_fraction=float(solver.lts_info["fine_fraction"]),
                          rho=float(solver.lts_info["rho"])))),
        )  # fmt: skip
        if gap:
            th = times - 0.5 * dt
            ui = (
                solver.eta_b * solver.incident(gp[:1], th)[:, 0]
            )  # amplitude at the centre
            out["e_gap"] = acc_e
            out["e0_gap"] = (np.exp(-2j * np.pi * np.outer(freqs, th)) @ ui) * dt
        return out
