# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Baseline arm: triangular FIT (`UnstructuredFIT2D`, Voronoi Hodge).

* mesh: `wakis.benchmarks.meshing.cluster_mesh`, h on and inside the
  cylinders, graded (growth 1.2 per cell up to h_out = min(4 h,
  lambda_min,host / ppw_out)) or uniform (h everywhere);
* scattered-field plane wave (`add_plane_wave`, dispersive scatterers driven
  by the total field, `wakis.unstructured.dispersive_planewave`);
* radial CFS-PML (``n_pml`` polar rings of h_out behind the case's absorber
  radius, PEC behind);
* DFT of the scattered node field on the extraction circles (linear
  interpolation), incident phasor at the origin from the analytic incident
  wave at the same time levels (TE: H_z at the half steps), gap E from the
  second-order least-squares patch reconstruction of the edge fluxes (with the
  pointwise Raviart-Thomas reconstruction kept as ``e_gap_rt``, TE).

The resolution label is h [m] on the scatterers.
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0 as eps_0

from ...cylinder_harmonics import circle_points
from ..meshing import cluster_mesh
from ..runner import Arm


class Triangles2DArm(Arm):
    """
    Triangle FIT arm for `Scattering2DCase` (see the module docstring).

    Parameters
    ----------
    graded : bool
    growth : float
    ratio : float
        Largest graded size in units of h (h_out = min(ratio h,
        lambda_min,host / ppw_out)), so that the whole mesh refines with h.
    ppw_out : float
    n_pml : int
    cfl : float
    hodge : {'voronoi', 'barycentric'}
    ladder : sequence of float, optional
    lts : int or dict, optional
        Local time stepping of `UnstructuredFIT2D` (e.g. 2).
    """

    def __init__(self, graded=True, growth=1.2, ppw_out=20, ratio=4.0, n_pml=10, cfl=0.95,
                 hodge="voronoi", ladder=None, name=None, lts=None, pml=None,
                 pml_dr=1.0, r_pml=None, gap_patch=True, pml_d=None):  # fmt: skip
        """
        pml : dict, optional
            Extra `UnstructuredFIT2D` PML options (order, R, kappa_max, alpha).
        pml_dr : float
            Radial spacing of the PML rings in units of h_out.
        pml_d : float, optional
            PML thickness [m] fixed in physical units (ring spacing pml_d /
            n_pml, overrides pml_dr).
        r_pml : float, optional
            PML start radius [m] (default: the case's absorber radius).
        gap_patch : bool
            When True (default), ``e_gap`` is the gap field from a
            least-squares linear fit to the edge fluxes D.n of the 12 nearest
            host edges (second order) and ``e_gap_rt`` is the pointwise
            lowest-order Raviart-Thomas reconstruction (diagnostic).
        """
        self.pml = dict(pml or {})
        self.pml_dr = pml_dr
        self.r_pml = r_pml
        self.gap_patch = gap_patch
        self.pml_d = pml_d
        self.graded = graded
        self.growth = growth
        self.ppw_out = ppw_out
        self.ratio = ratio
        self.n_pml = n_pml
        self.cfl = cfl
        self.hodge = hodge
        self.ladder = ladder
        self.lts = lts
        self.name = name or f"tri-{'graded' if graded else 'uniform'}" + (
            "-lts" if lts else ""
        )

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

    @staticmethod
    def _patch_matrix(mesh, solver, gpts, case, k=12):
        """
        Sparse (2 ng x n_edges) map edge fluxes -> E at the gap points by a
        least-squares linear fit E(x) = a + B (x - x0) to D.n / eps = d_e /
        (L_e eps_0 eps_b) on the k nearest edges whose cells are all host.
        """
        ce = mesh.cell_edges
        host_cell = mesh.cell_tags == 0
        bad = np.zeros(mesh.n_edges, bool)
        bad[ce[~host_cell].ravel()] = True
        cand = np.flatnonzero(~bad)
        mid = mesh.edge_midpoint[cand]
        t = mesh.edge_tangent[cand]
        n_e = np.stack([t[:, 1], -t[:, 0]], 1)  # flux normal t x z
        scale = 1.0 / (mesh.edge_length[cand] * eps_0 * case.n_background**2)
        ng = len(gpts)
        rows, cols, vals = [], [], []
        for i, x0 in enumerate(gpts):
            j = np.argsort(np.hypot(*(mid - x0).T))[:k]
            dx = mid[j] - x0
            ell = np.abs(dx).max()  # scale: well-conditioned fit
            dx = dx / ell
            nx, ny = n_e[j, 0], n_e[j, 1]
            # unknowns (ax, ay, bxx, bxy, byx, byy): E.n = nx (ax + bxx dx + bxy dy) + ny (ay + ...)
            Amat = np.stack(
                [nx, ny, nx * dx[:, 0], nx * dx[:, 1], ny * dx[:, 0], ny * dx[:, 1]], 1
            )
            pinv = np.linalg.pinv(Amat)  # (6, k)
            for comp, row in ((0, i), (1, ng + i)):
                rows += [row] * len(j)
                cols += list(cand[j])
                vals += list(pinv[comp] * scale[j])
        return sp.csr_matrix((vals, (rows, cols)), shape=(2 * ng, mesh.n_edges))

    def build(self, case, h):
        from ...unstructured import UnstructuredFIT2D

        lam_host = np.min(case.wavelengths) / case.n_background
        h_out = (
            max(h, min(self.ratio * h, lam_host / self.ppw_out)) if self.graded else h
        )
        r0 = max(self.r_pml or case.r_absorber, max(case.circles) + 2 * h_out)
        dr = self.pml_d / self.n_pml if self.pml_d else self.pml_dr * h_out
        mesh = cluster_mesh(
            case.cylinders, h, h_out, r0, growth=self.growth,
            pml_layers=self.n_pml, pml_dr=dr,
        )  # fmt: skip
        nb2 = case.n_background**2
        mats = {0: [nb2, 1.0], len(case.cylinders) + 1: [nb2, 1.0]}
        for j, cy in enumerate(case.cylinders):
            mats[j + 1] = case.materials[cy["material"]]
        solver = UnstructuredFIT2D(
            mesh, mats, polarization=case.polarization, bc="pec", hodge=self.hodge,
            cfl=self.cfl, lts=self.lts,
            pml=dict(dict(r0=r0, r1=r0 + self.n_pml * dr, kappa_max=1.0, alpha=0.0),
                     **self.pml),
        )  # fmt: skip
        d = case.direction
        solver.add_plane_wave(
            case.source(), direction=(np.cos(d), np.sin(d)), background_eps_r=nb2
        )
        return mesh, solver, h_out

    def run(self, case, h):
        t0 = time.time()
        mesh, solver, h_out = self.build(case, h)
        pts = np.vstack([circle_points(r, case.n_points)[0] for r in case.circles])
        P = solver._interp(pts)
        gap = case.gap_point is not None
        if gap:
            gpts = case.gap_points
            ng = len(gpts)
            cell, W = mesh.whitney_edge(gpts)
            rows, cols, vals = [], [], []
            for i in range(ng):  # Raviart-Thomas: W_e x z, rows (x_i, y_i)
                edges = mesh.cell_edges[cell[i]]
                rows += [i] * 3 + [ng + i] * 3
                cols += list(edges) * 2
                vals += list(W[i, :, 1]) + list(-W[i, :, 0])
            Pg = sp.csr_matrix((vals, (rows, cols)), shape=(2 * ng, mesh.n_edges)) / (
                eps_0 * case.n_background**2
            )
            gpos = gpts @ solver.plane_wave["k"]
            if self.gap_patch:
                Pp = self._patch_matrix(mesh, solver, gpts, case)
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
                einc = solver._incident_signal(gpos, times[n])
                et = es + np.concatenate([einc * pvec[0], einc * pvec[1]])
                kern = np.exp(-2j * np.pi * freqs * times[n]) * dt
                acc_e += np.outer(kern, et)
                if self.gap_patch:
                    ep = Pp @ solver.v + np.concatenate(
                        [einc * pvec[0], einc * pvec[1]]
                    )
                    acc_p += np.outer(kern, ep)
            if n % 4000 == 0 and not np.all(np.isfinite(solver.u)):
                raise FloatingPointError("unstable")
        run_s = time.time() - t1
        u_inc0 = solver.incident_field(np.zeros((1, 2)), times + shift)
        u0 = (np.exp(-2j * np.pi * np.outer(freqs, times + shift)) @ u_inc0) * dt
        out = dict(
            samples=acc.reshape(len(freqs), len(case.circles), case.n_points),
            u0=u0, radii=list(case.circles),
            cost=dict(unknowns=int(solver.free.size + mesh.n_edges),
                      nodes=int(solver.free.size), cells=int(mesh.n_cells),
                      steps=n_steps, dt=dt, run_s=run_s, setup_s=setup_s, h=h,
                      h_max=h_out, n_pml=self.n_pml,
                      gap_metric="patch" if self.gap_patch else "rt"),
        )  # fmt: skip
        if gap:
            rt = np.stack([acc_e[:, :ng], acc_e[:, ng:]], 2)
            if self.gap_patch:
                out["e_gap"] = np.stack([acc_p[:, :ng], acc_p[:, ng:]], 2)
                out["e_gap_rt"] = rt
            else:
                out["e_gap"] = rt
            ui = solver._incident_signal(np.zeros(1), times)  # amplitude s(t) at origin
            out["e0_gap"] = (np.exp(-2j * np.pi * np.outer(freqs, times)) @ ui) * dt
        return out
