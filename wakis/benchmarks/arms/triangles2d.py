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
  Raviart-Thomas reconstruction of the edge fluxes (TE).

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
                 hodge="voronoi", ladder=None, name=None, lts=None):  # fmt: skip
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

    def build(self, case, h):
        from ...unstructured import UnstructuredFIT2D

        lam_host = np.min(case.wavelengths) / case.n_background
        h_out = (
            max(h, min(self.ratio * h, lam_host / self.ppw_out)) if self.graded else h
        )
        r0 = max(case.r_absorber, max(case.circles) + 2 * h_out)
        mesh = cluster_mesh(
            case.cylinders, h, h_out, r0, growth=self.growth,
            pml_layers=self.n_pml, pml_dr=h_out,
        )  # fmt: skip
        nb2 = case.n_background**2
        mats = {0: [nb2, 1.0], len(case.cylinders) + 1: [nb2, 1.0]}
        for j, cy in enumerate(case.cylinders):
            mats[j + 1] = case.materials[cy["material"]]
        solver = UnstructuredFIT2D(
            mesh, mats, polarization=case.polarization, bc="pec", hodge=self.hodge,
            cfl=self.cfl, lts=self.lts,
            pml=dict(r0=r0, r1=r0 + self.n_pml * h_out, kappa_max=1.0, alpha=0.0),
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
        setup_s = time.time() - t0
        dt = solver.dt
        shift = 0.0 if case.polarization == "TM" else -0.5 * dt
        freqs = case.freqs
        acc = np.zeros((len(freqs), len(pts)), complex)
        acc_e = np.zeros((len(freqs), 2 * len(case.gap_points) if gap else 2), complex)
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
                acc_e += np.outer(np.exp(-2j * np.pi * freqs * times[n]), et) * dt
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
                      h_max=h_out),
        )  # fmt: skip
        if gap:
            out["e_gap"] = np.stack([acc_e[:, :ng], acc_e[:, ng:]], 2)
            ui = solver._incident_signal(np.zeros(1), times)  # amplitude s(t) at origin
            out["e0_gap"] = (np.exp(-2j * np.pi * np.outer(freqs, times)) @ ui) * dt
        return out
