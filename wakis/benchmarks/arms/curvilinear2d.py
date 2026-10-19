# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Arm: body-fitted O-grid curvilinear FIT (`wakis.curvilinear`).

* mesh: `ogrid_mesh` around a single cylinder centred on the origin
  (butterfly core inside ``core * a``, nonorthogonal corner Hodge; polar rings
  with the diagonal cylindrical Hodge elsewhere). Radial spacing ``h_r = h *
  radial`` inside the cylinder and in a band of 2 h outside it, then growth
  1.2 per cell up to h_out = max(h, min(ratio h, lambda_min,host / ppw_out));
  angular sectors n_theta = 4 ceil(2 pi a / (4 h)). The extraction circles,
  the absorber radius and the cylinder radius are node rings.
* scattered-field plane wave (dispersive scatterers through the total-field
  ADE), radial complex-stretching PML (``n_pml`` rings of h_out), PEC behind;
* DFT of the scattered node field on the extraction rings, resampled to the
  case's n_points angles by FFT (trigonometric interpolation), incident
  phasor at the origin at the same time levels (TE: H_z at the half steps).

Single centred cylinders only (B1, B2, B5 not supported: PEC with a plane
wave). The resolution label is h [m] (azimuthal spacing on the cylinder).
"""

import time

import numpy as np

from ..runner import Arm


def ring_resampler(mesh, free, radius, m_pts):
    """(free-local indices, real (m_pts, n_ring) matrix): trigonometric
    interpolation from the ring nodes to m_pts uniform angles from 0."""
    idx, th = mesh.ring(radius)
    loc = np.searchsorted(free, idx)
    if not np.all(free[loc] == idx):
        raise ValueError("ring nodes must be free")
    nr = len(idx)
    if not np.allclose(np.diff(th), 2 * np.pi / nr, rtol=1e-6):
        raise ValueError("ring nodes are not uniformly spaced")
    k = np.fft.fftfreq(nr, 1.0 / nr)
    keep = np.abs(k) < nr / 2
    tq = 2 * np.pi * np.arange(m_pts) / m_pts - th[0]
    T = np.exp(1j * np.outer(tq, k[keep])) @ np.fft.fft(np.eye(nr), axis=0)[keep] / nr
    return loc, T.real.copy()


class Curvilinear2DArm(Arm):
    """
    O-grid curvilinear FIT arm (see the module docstring).

    Parameters
    ----------
    radial : float
        Radial spacing near the cylinder surface in units of h.
    core : float
        Core radius / a.
    ratio, ppw_out, growth, n_pml, cfl
        As in `Triangles2DArm`.
    """

    def __init__(self, radial=1.0, core=0.9, ratio=4.0, ppw_out=20, growth=1.2, n_pml=10,
                 cfl=0.95, ladder=None, name=None):  # fmt: skip
        self.radial, self.core = radial, core
        self.ratio, self.ppw_out, self.growth = ratio, ppw_out, growth
        self.n_pml, self.cfl, self.ladder = n_pml, cfl, ladder
        self.name = name or ("ogrid" if radial == 1.0 else f"ogrid-r{radial:g}")

    def supports(self, case):
        cy = getattr(case, "cylinders", [])
        return (
            len(cy) == 1
            and np.allclose(cy[0]["center"], 0.0)
            and not any(isinstance(m, str) for m in case.materials.values())
        )

    def resolutions(self, case):
        if self.ladder is not None:
            return list(self.ladder)
        a = case.cylinders[0]["radius"]
        if a < 0.5e-6:
            return [a / q for q in (5, 7, 10, 14, 20, 28, 40)]
        lam_in = np.min(case.wavelengths) / 2.0
        return [lam_in / r for r in (10, 14, 20, 28, 40)]

    def build(self, case, h):
        from ...curvilinear import CurvilinearFIT2D, ogrid_mesh, radial_nodes

        a = case.cylinders[0]["radius"]
        lam_host = np.min(case.wavelengths) / case.n_background
        h_out = max(h, min(self.ratio * h, lam_host / self.ppw_out))
        hr = self.radial * h
        g = self.growth

        def size(r):  # hr within 2 h of the surface, linear growth (g - 1) away
            d = max(abs(r - a) - 2 * h, 0.0)
            return min(h if r < a else h_out, hr + (g - 1) * d)

        r0 = max(case.r_absorber, max(case.circles) + 2 * h_out)
        r1 = r0 + self.n_pml * h_out
        breaks = [self.core * a, a, *sorted(case.circles), r0, r1]
        radii = radial_nodes(breaks, lambda r: h_out if r >= r0 - 1e-15 else size(r))
        nt = int(4 * np.ceil(2 * np.pi * a / h / 4))
        mesh = ogrid_mesh(radii, nt, interfaces=[a])
        nb2 = case.n_background**2
        mats = {0: [nb2, 1.0], 1: case.materials[case.cylinders[0]["material"]]}
        solver = CurvilinearFIT2D(
            mesh, mats, polarization=case.polarization, bc="pec", cfl=self.cfl,
            pml=dict(kind="radial", r0=r0, r1=r1, v=2.99792458e8 / case.n_background),
        )  # fmt: skip
        d = case.direction
        solver.add_plane_wave(case.source(), direction=(np.cos(d), np.sin(d)),
                              background_eps_r=nb2)  # fmt: skip
        return mesh, solver, h_out

    def run(self, case, h):
        t0 = time.time()
        mesh, solver, h_out = self.build(case, h)
        rings = [ring_resampler(mesh, solver.free, r, case.n_points) for r in case.circles]
        locs = np.concatenate([r[0] for r in rings])
        setup_s = time.time() - t0
        dt = solver.dt
        shift = 0.0 if case.polarization == "TM" else -0.5 * dt
        freqs = case.freqs
        n_steps = int(case.t_run / dt)
        times = (np.arange(n_steps) + 1) * dt
        ker = np.exp(-2j * np.pi * np.outer(times + shift, freqs)) * dt
        acc = np.zeros((len(freqs), locs.size), complex)
        chunk = 256
        buf = np.zeros((chunk, locs.size))
        t1 = time.time()
        for n in range(n_steps):
            solver.one_step()
            buf[n % chunk] = solver.u[locs]
            if n % chunk == chunk - 1 or n == n_steps - 1:
                n0 = n - n % chunk
                acc += ker[n0 : n + 1].T @ buf[: n - n0 + 1]
            if n % 4000 == 0 and not np.all(np.isfinite(solver.u)):
                raise FloatingPointError("unstable")
        run_s = time.time() - t1
        parts, o = [], 0
        for loc, T in rings:
            parts.append(acc[:, o : o + loc.size] @ T.T)
            o += loc.size
        samples = np.stack(parts, axis=1)
        u_inc0 = solver.incident_field(np.zeros((1, 2)), times + shift)
        u0 = (np.exp(-2j * np.pi * np.outer(freqs, times + shift)) @ u_inc0) * dt
        return dict(
            samples=samples, u0=u0, radii=list(case.circles),
            cost=dict(unknowns=solver.n_unknowns, nodes=int(solver.free.size),
                      cells=int(mesh.n_cells), steps=n_steps, dt=dt, run_s=run_s,
                      setup_s=setup_s, h=h, h_max=h_out, n_theta=int(mesh.ring(case.circles[0])[0].size),
                      courant=float(solver.courant)),
        )  # fmt: skip
