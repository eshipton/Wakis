# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Quadtree-square + conforming-triangle FIT arm (`SubTriFIT2D`).

* mesh: `subtri_circles_mesh` -- 2:1-balanced quadtree of squares from coarse
  size ``H = 2**levels * h`` down to ``h`` on and inside each cylinder, with a
  narrow conforming Delaunay triangle band of half-width ``band * h`` around
  each circular interface and across narrow gaps (refined to ``gap_cells``
  triangles across, ``>= h_min``);
* solver: `SubTriFIT2D` (hanging-node prolongation ``P``, diagonal orthogonal
  Hodge, split-field PML of ``n_pml`` coarse ``H`` cells behind the absorber
  radius, PEC behind the PML);
* local time stepping: ``lts='auto'`` benchmarks ``p`` in ``lts_candidates``
  against the global leapfrog baseline and picks the stable configuration with
  the lowest unknown-weighted work rate ``lts_work_per_step() / dt``;
* gap fields (TE, B3): 9-point stencil ``case.gap_points`` reconstructed with
  the second-order least-squares edge-flux patch (``e_gap``) alongside the
  pointwise single-triangle Raviart-Thomas reconstruction (``e_gap_rt``).
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0

from ...cylinder_harmonics import circle_points
from ..runner import Arm
from .polygon2d import _rt_matrix

__all__ = ["SubTriArm"]


def _patch_matrix(mesh, gpts, case, k=12):
    """
    Sparse ``(2 * ng, n_edges)`` map from edge fluxes to ``E`` at ``gpts`` by a
    least-squares linear fit ``E(x) = a + B (x - x0)`` to
    ``E . n = d_e / (L_e eps_0 eps_b)`` on the ``k`` nearest host-only edges.
    """
    host_tag = len(case.cylinders)
    host_cell = mesh.cell_tags == host_tag
    bad = mesh.edge_cells[:, 1] < 0
    bad[mesh.corner_edge[~host_cell[mesh.corner_cell]]] = True
    cand = np.flatnonzero(~bad)
    e_nodes = mesh.nodes[mesh.edges[cand]]
    mid = 0.5 * (e_nodes[:, 0] + e_nodes[:, 1])
    vec = e_nodes[:, 1] - e_nodes[:, 0]
    elen = np.linalg.norm(vec, axis=1)
    t = vec / elen[:, None]
    n_e = np.stack([t[:, 1], -t[:, 0]], 1)  # flux normal t x z
    scale = 1.0 / (elen * eps_0 * case.n_background**2)
    ng = len(gpts)
    rows, cols, vals = [], [], []
    for i, x0 in enumerate(gpts):
        j = np.argsort(np.hypot(*(mid - x0).T))[:k]
        dx = mid[j] - x0
        ell = np.abs(dx).max()
        dx = dx / ell
        nx, ny = n_e[j, 0], n_e[j, 1]
        Amat = np.stack(
            [nx, ny, nx * dx[:, 0], nx * dx[:, 1], ny * dx[:, 0], ny * dx[:, 1]], 1
        )
        pinv = np.linalg.pinv(Amat)
        for comp, row in ((0, i), (1, ng + i)):
            rows += [row] * len(j)
            cols += list(cand[j])
            vals += list(pinv[comp] * scale[j])
    return sp.csr_matrix((vals, (rows, cols)), shape=(2 * ng, mesh.n_edges))


class SubTriArm(Arm):
    """
    Quadtree-square + conforming-triangle FIT arm for `Scattering2DCase`.

    Parameters
    ----------
    ratio : float
        Target coarse-to-fine size ratio ``H / h`` (default 4).
    ppw_out : float or None
        Cap target coarse size at ``lambda_min,host / ppw_out``; ``None`` leaves
        ``h_out_target = ratio * h`` uncapped.
    band : float
        Half-width of the removed finest-square annulus around each interface,
        in units of ``h`` (default 2.0).
    grade : float
        Quadtree grading margin in units of the current cell size (default 2.0).
    n_pml : int
        Number of coarse (size ``H``) PML square layers (default 10).
    gap_cells : float or None
        Minimum number of triangle cells across narrow inter-cylinder gaps
        (default 4.0).
    cfl : float
        Courant factor (default 0.95).
    lts : {"auto", None, 1} or int
        Local time stepping mode. ``"auto"`` tries each ``p`` in
        `lts_candidates` and selects the stable solver with the lowest work
        rate ``lts_work_per_step() / dt``.
    lts_candidates : sequence of int
        Candidate step ratios tested when ``lts="auto"`` (default ``(2, 3, 4)``).
    ladder : sequence of float, optional
        Custom resolution ladder [m].
    name : str
        Arm identifier (default ``"subtri-hybrid"``).
    """

    def __init__(
        self,
        ratio=4,
        ppw_out=20,
        band=2.0,
        grade=2.0,
        n_pml=10,
        gap_cells=4.0,
        cfl=0.95,
        lts="auto",
        lts_candidates=(2, 3, 4),
        ladder=None,
        name="subtri-hybrid",
    ):
        self.ratio = ratio
        self.ppw_out = ppw_out
        self.band = band
        self.grade = grade
        self.n_pml = n_pml
        self.gap_cells = gap_cells
        self.cfl = cfl
        self.lts = lts
        self.lts_candidates = tuple(lts_candidates)
        self.ladder = ladder
        self.name = name

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

    def build(self, case, h):
        from ...unstructured.subtri import SubTriFIT2D, subtri_circles_mesh

        nb2 = case.n_background**2
        lam_host = np.min(case.wavelengths) / case.n_background
        h_out_target = (
            self.ratio * h
            if self.ppw_out is None
            else max(h, min(self.ratio * h, lam_host / self.ppw_out))
        )
        levels = max(0, int(np.floor(np.log2(h_out_target / h) + 1e-9)))
        H = (2**levels) * h
        r0 = max(case.r_absorber, max(case.circles) + 2 * H)
        circles = [(*cy["center"], cy["radius"]) for cy in case.cylinders]
        ncyl = len(circles)
        gaps = []
        for i in range(ncyl):
            for j in range(i + 1, ncyl):
                d = np.hypot(*np.subtract(circles[i][:2], circles[j][:2]))
                gaps.append(d - circles[i][2] - circles[j][2])
        gap_cells = self.gap_cells if gaps else None
        h_min = min(gaps) / self.gap_cells if gaps and self.gap_cells else None
        mesh, P_mesh, info = subtri_circles_mesh(
            circles,
            h,
            levels,
            r0,
            band=self.band,
            grade=self.grade,
            pml_cells=self.n_pml,
            gap_cells=gap_cells,
            h_min=h_min,
        )
        L = info["half_width"]
        pml_kw = dict(x=(-L, L), y=(-L, L), thickness=self.n_pml * H, eps_r=nb2)
        mats = {ncyl: [nb2, 1.0]}
        for j, cy in enumerate(case.cylinders):
            mats[j] = case.materials[cy["material"]]

        cfl = self.cfl
        for _ in range(20):
            try:
                solver = SubTriFIT2D(
                    mesh,
                    P_mesh,
                    mats,
                    polarization=case.polarization,
                    pml=pml_kw,
                    cfl=cfl,
                    lts=None,
                )
                break
            except ValueError as e:
                if "unstable" not in str(e):
                    raise
                cfl *= 0.85
        else:
            raise RuntimeError("no stable dt for the dispersive materials")
        self._cfl_used = cfl
        dt_base = solver.dt
        best_rate = solver.n_unknowns / dt_base
        lts_search = {
            "1": dict(
                work_rate=float(best_rate),
                dt=float(dt_base),
                work_per_step=int(solver.n_unknowns),
            )
        }
        if self.lts == "auto":
            for p in self.lts_candidates:
                try:
                    cand = SubTriFIT2D(
                        mesh,
                        P_mesh,
                        mats,
                        polarization=case.polarization,
                        pml=pml_kw,
                        cfl=cfl,
                        lts=int(p),
                    )
                except ValueError as e:
                    lts_search[str(p)] = dict(skipped=str(e))
                    continue
                rate = cand.lts_work_per_step() / cand.dt
                lts_search[str(p)] = dict(
                    work_rate=float(rate),
                    dt=float(cand.dt),
                    dt_ratio=float(cand.dt / dt_base),
                    work_per_step=int(cand.lts_work_per_step()),
                    fine_fraction=float(cand.lts_info["fine_fraction"]),
                    rho=float(cand.lts_info["rho"]),
                )
                if rate < best_rate:
                    best_rate = rate
                    solver = cand
        elif self.lts not in (None, 1):
            solver = SubTriFIT2D(
                mesh,
                P_mesh,
                mats,
                polarization=case.polarization,
                pml=pml_kw,
                cfl=cfl,
                lts=self.lts,
            )
        self._dt_base = dt_base
        self._lts_search = lts_search
        d = case.direction
        solver.add_plane_wave(
            case.source(), direction=(np.cos(d), np.sin(d)), background_eps_r=nb2
        )
        return mesh, solver, info

    def run(self, case, h):
        t0 = time.time()
        mesh, solver, info = self.build(case, h)
        pts = np.vstack([circle_points(r, case.n_points)[0] for r in case.circles])
        P = solver._interp(pts)
        gap = case.gap_point is not None
        if gap:
            if case.polarization != "TE":
                raise NotImplementedError("gap field: TE only")
            gpts = case.gap_points
            ng = len(gpts)
            rts = [_rt_matrix(mesh, pt) for pt in gpts]
            Pg = sp.vstack(
                [*(r[0] for r in rts), *(r[1] for r in rts)], format="csr"
            ) / (eps_0 * case.n_background**2)
            Pp = _patch_matrix(mesh, gpts, case)
            gpos = gpts @ solver.plane_wave["k"]
        setup_s = time.time() - t0
        dt = solver.dt
        shift = 0.0 if case.polarization == "TM" else -0.5 * dt
        freqs = case.freqs
        acc = np.zeros((len(freqs), len(pts)), complex)
        acc_e = np.zeros((len(freqs), 2 * len(case.gap_points) if gap else 2), complex)
        acc_p = np.zeros_like(acc_e)
        n_steps = int(case.t_run / dt)
        times = (np.arange(n_steps) + 1) * dt
        d = case.direction
        pvec = np.array([-np.sin(d), np.cos(d)])
        t1 = time.time()
        for n in range(n_steps):
            solver.one_step()
            acc += (
                np.outer(np.exp(-2j * np.pi * freqs * (times[n] + shift)), P @ solver.u)
                * dt
            )
            if gap:
                es = Pg @ solver.v
                ep = Pp @ solver.v
                einc = solver._incident_signal(gpos, times[n])
                inc_vec = np.concatenate([einc * pvec[0], einc * pvec[1]])
                kern = np.exp(-2j * np.pi * freqs * times[n]) * dt
                acc_e += np.outer(kern, es + inc_vec)
                acc_p += np.outer(kern, ep + inc_vec)
            if n % 4000 == 0 and not np.all(np.isfinite(solver.u)):
                raise FloatingPointError("unstable")
        run_s = time.time() - t1
        u_inc0 = solver.incident_field(np.zeros((1, 2)), times + shift)
        u0 = (np.exp(-2j * np.pi * np.outer(freqs, times + shift)) @ u_inc0) * dt
        lts_cost = (
            None
            if solver.lts is None
            else dict(
                p=int(solver.lts.p),
                fine_fraction=float(solver.lts_info["fine_fraction"]),
                rho=float(solver.lts_info["rho"]),
                dt_ratio=float(dt / self._dt_base),
            )
        )
        work = int(n_steps * solver.lts_work_per_step())
        out = dict(
            samples=acc.reshape(len(freqs), len(case.circles), case.n_points),
            u0=u0,
            radii=list(case.circles),
            cost=dict(
                unknowns=int(solver.n_unknowns),
                nodes=int(solver.free.size),
                cells=int(mesh.n_cells),
                steps=int(n_steps),
                dt=float(dt),
                run_s=float(run_s),
                setup_s=float(setup_s),
                h=float(h),
                h_max=float(info["H"]),
                levels=int(info["levels"]),
                n_squares_per_level=list(info["n_squares"]),
                n_triangles=int(info["n_triangles"]),
                n_hanging=int(info["n_hanging"]),
                cfl=float(self._cfl_used),
                lts=lts_cost,
                lts_search=self._lts_search,
                work=work,
                gap_metric="patch",
            ),
        )
        if gap:
            out["e_gap"] = np.stack([acc_p[:, :ng], acc_p[:, ng:]], 2)
            out["e_gap_rt"] = np.stack([acc_e[:, :ng], acc_e[:, ng:]], 2)
            ui = solver._incident_signal(np.zeros(1), times)
            out["e0_gap"] = (np.exp(-2j * np.pi * np.outer(freqs, times)) @ ui) * dt
        return out
