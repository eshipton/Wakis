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


class BoxedCurvilinear2DArm(Arm):
    """
    Multi-block curvilinear FIT for a row of equal cylinders on the x axis
    (B3 dimers, also single cylinders): `boxed_cylinders_mesh` (core + polar
    rings inside each cylinder, a circle-to-box transition block of ``m_tr``
    layers, adjacent boxes tiling a rectangle, a graded rectilinear Cartesian
    block outside with a Cartesian PML). Extraction circles are sampled by
    bilinear interpolation on the Cartesian block; the gap field (TE) from
    the corner reconstruction of D in the cells around the gap node.

    The resolution label is h [m] (azimuthal spacing on the cylinders).
    """

    def __init__(self, radial=1.0, core=0.9, ratio=4.0, ppw_out=20, growth=1.2, n_pml=10,
                 cfl=0.95, box=1.25, ladder=None, name=None):  # fmt: skip
        self.radial, self.core, self.box = radial, core, box
        self.ratio, self.ppw_out, self.growth = ratio, ppw_out, growth
        self.n_pml, self.cfl, self.ladder = n_pml, cfl, ladder
        self.name = name or "ogrid-boxed"

    def supports(self, case):
        cy = getattr(case, "cylinders", [])
        if not cy or any(isinstance(m, str) for m in case.materials.values()):
            return False
        r = {c["radius"] for c in cy}
        xs = sorted(c["center"][0] for c in cy)
        return len(r) == 1 and all(c["center"][1] == 0 for c in cy) and np.allclose(xs, [-x for x in xs[::-1]])

    def resolutions(self, case):
        if self.ladder is not None:
            return list(self.ladder)
        a = case.cylinders[0]["radius"]
        return [a / q for q in (5, 7, 10, 14, 20, 28, 40)]

    def build(self, case, h):
        from ...curvilinear import CurvilinearFIT2D, boxed_cylinders_mesh, radial_nodes

        a = case.cylinders[0]["radius"]
        xs_c = sorted(c["center"][0] for c in case.cylinders)
        half = (xs_c[1] - xs_c[0]) / 2 if len(xs_c) > 1 else self.box * a
        gap = half - a  # circle-to-box distance on the box mid-sides
        lam_host = np.min(case.wavelengths) / case.n_background
        h_out = max(h, min(self.ratio * h, lam_host / self.ppw_out))
        hr = self.radial * h
        nt = int(8 * np.ceil(2 * np.pi * a / h / 8))
        radii_in = radial_nodes([self.core * a, a], lambda r: min(h, hr + 0.2 * max(a - 2 * h - r, 0)))
        m_tr = max(2, int(np.ceil(gap / hr)))
        hb = 2 * half / (nt // 4)
        g = self.growth
        r0 = max(case.r_absorber, max(case.circles) + 2 * h_out)
        r1 = r0 + self.n_pml * h_out

        def outer(start):
            return radial_nodes([start, r0, r1], lambda r: h_out if r >= r0 - 1e-15
                                else min(h_out, hb + (g - 1) * (r - start)))[1:]  # fmt: skip

        X0 = xs_c[-1] + half
        mesh = boxed_cylinders_mesh(xs_c, a, radii_in, nt, half, m_tr, outer(X0), outer(half))
        nb2 = case.n_background**2
        mats = {0: [nb2, 1.0], 1: case.materials[case.cylinders[0]["material"]]}
        solver = CurvilinearFIT2D(
            mesh, mats, polarization=case.polarization, bc="pec", cfl=self.cfl,
            pml=dict(kind="cartesian", r0=r0, r1=r1, v=2.99792458e8 / case.n_background),
        )  # fmt: skip
        d = case.direction
        solver.add_plane_wave(case.source(), direction=(np.cos(d), np.sin(d)),
                              background_eps_r=nb2)  # fmt: skip
        return mesh, solver, h_out

    @staticmethod
    def bilinear(mesh, free, points):
        import scipy.sparse as sp

        xs, ys = mesh.cartesian_axes
        scale = np.abs(mesh.nodes).max()
        key = {tuple(k): i for i, k in enumerate(np.round(mesh.nodes / (scale * 1e-10)).astype(np.int64))}
        loc = -np.ones(mesh.n_nodes, np.int64)
        loc[free] = np.arange(free.size)
        rows, cols, vals = [], [], []
        for p, (px, py) in enumerate(points):
            i = np.searchsorted(xs, px) - 1
            j = np.searchsorted(ys, py) - 1
            tx = (px - xs[i]) / (xs[i + 1] - xs[i])
            ty = (py - ys[j]) / (ys[j + 1] - ys[j])
            for di, dj, w in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                              (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):  # fmt: skip
                n = key[tuple(np.round(np.array([xs[i + di], ys[j + dj]]) / (scale * 1e-10)).astype(np.int64))]
                if loc[n] < 0:
                    raise ValueError("interpolation node is not free")
                rows.append(p)
                cols.append(loc[n])
                vals.append(w)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(points), free.size))

    @staticmethod
    def gap_operator(mesh, points):
        """Sparse (2 n_points, n_edges) map from edge fluxes to D_x (rows
        0..n-1) and D_y (rows n..2n-1) at `points`: the corner
        reconstructions of the containing (general) cell, blended with its
        inverse bilinear coordinates."""
        import scipy.sparse as sp

        from ...curvilinear import hodge_components

        hc = hodge_components(mesh)
        gen = np.flatnonzero(~mesh.is_polar)
        P = mesh.nodes[mesh.cells[gen]]
        rows_x, rows_y = [], []
        for p in points:
            cand = np.argsort(np.hypot(*(P.mean(1) - p).T))[:12]
            found = None
            for c in cand:
                X = P[c]
                xi = np.array([0.5, 0.5])
                for _ in range(30):  # Newton on the bilinear map
                    a, b = xi
                    f = ((1 - a) * (1 - b) * X[0] + a * (1 - b) * X[1] + a * b * X[2]
                         + (1 - a) * b * X[3] - p)  # fmt: skip
                    J = np.stack([(1 - b) * (X[1] - X[0]) + b * (X[2] - X[3]),
                                  (1 - a) * (X[3] - X[0]) + a * (X[2] - X[1])], 1)  # fmt: skip
                    xi = xi - np.linalg.solve(J, f)
                if np.all(xi > -1e-9) and np.all(xi < 1 + 1e-9):
                    found = (gen[c], xi)
                    break
            if found is None:
                raise ValueError(f"gap point {p} not in a general cell")
            cell, (a, b) = found
            w = [(1 - a) * (1 - b), a * (1 - b), a * b, (1 - a) * b]
            q = np.flatnonzero(hc["cell"] == cell)
            rx, ry = 0, 0
            for k in range(4):
                corner = mesh.nodes[mesh.cells[cell, k]]
                qk = q[np.argsort(np.hypot(*(hc["point"][q] - corner).T))[:2]]
                qx = qk[np.argmax(hc["direction"][qk][:, 0])]
                qy = qk[np.argmax(hc["direction"][qk][:, 1])]
                rx = rx + w[k] * hc["R"][qx]
                ry = ry + w[k] * hc["R"][qy]
            rows_x.append(rx)
            rows_y.append(ry)
        return sp.vstack(rows_x + rows_y).tocsr()

    def run(self, case, h):
        import scipy.sparse as sp
        from scipy.constants import epsilon_0 as eps_0

        from ...curvilinear import hodge_components
        from ...cylinder_harmonics import circle_points

        t0 = time.time()
        mesh, solver, h_out = self.build(case, h)
        pts = np.vstack([circle_points(r, case.n_points)[0] for r in case.circles])
        P = self.bilinear(mesh, solver.free, pts)
        gap = case.gap_point is not None
        if gap:
            if case.polarization != "TE":
                raise NotImplementedError("gap field: TE only")
            gpts = np.atleast_2d(case.gap_points)
            Pg = self.gap_operator(mesh, gpts) / (eps_0 * case.n_background**2)
            ng = len(gpts)
        setup_s = time.time() - t0
        dt = solver.dt
        shift = 0.0 if case.polarization == "TM" else -0.5 * dt
        freqs = case.freqs
        n_steps = int(case.t_run / dt)
        times = (np.arange(n_steps) + 1) * dt
        ker = np.exp(-2j * np.pi * np.outer(times + shift, freqs)) * dt
        ker_e = np.exp(-2j * np.pi * np.outer(times, freqs)) * dt
        acc = np.zeros((len(freqs), len(pts)), complex)
        acc_e = np.zeros((len(freqs), 2 * ng if gap else 2), complex)
        d = case.direction
        pvec = np.array([-np.sin(d), np.cos(d)])
        chunk = 256
        buf = np.zeros((chunk, len(pts)))
        bufe = np.zeros((chunk, acc_e.shape[1]))
        t1 = time.time()
        for n in range(n_steps):
            solver.one_step()
            buf[n % chunk] = P @ solver.u
            if gap:
                einc = solver._sig(gpts @ solver.plane_wave["k"], times[n])
                bufe[n % chunk] = Pg @ solver.v + np.concatenate([einc * pvec[0], einc * pvec[1]])
            if n % chunk == chunk - 1 or n == n_steps - 1:
                n0 = n - n % chunk
                acc += ker[n0 : n + 1].T @ buf[: n - n0 + 1]
                acc_e += ker_e[n0 : n + 1].T @ bufe[: n - n0 + 1]
            if n % 4000 == 0 and not np.all(np.isfinite(solver.u)):
                raise FloatingPointError("unstable")
        run_s = time.time() - t1
        u_inc0 = solver.incident_field(np.zeros((1, 2)), times + shift)
        u0 = (np.exp(-2j * np.pi * np.outer(freqs, times + shift)) @ u_inc0) * dt
        out = dict(
            samples=acc.reshape(len(freqs), len(case.circles), case.n_points), u0=u0,
            radii=list(case.circles),
            cost=dict(unknowns=solver.n_unknowns, nodes=int(solver.free.size),
                      cells=int(mesh.n_cells), steps=n_steps, dt=dt, run_s=run_s,
                      setup_s=setup_s, h=h, h_max=h_out, courant=float(solver.courant)),
        )  # fmt: skip
        if gap:
            out["e_gap"] = np.stack([acc_e[:, :ng], acc_e[:, ng:]], 2)
            ui = solver._sig(np.zeros(1), times)
            out["e0_gap"] = ker_e.T @ ui
        return out
