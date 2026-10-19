"""
Showcase: a dielectric cylinder (n = 2, a = 1 um) with a thin high-index
coating (n = 3.5, t = 20 nm), x = k a = 2..10, TM and TE, against the exact
coated-cylinder series (`wakis.analytic.coated_cylinder`).

Why this case: a layer much thinner than the wavelength-driven cell size must
be resolved (>= 2 cells across, otherwise the effective layer is wrong). A
body-fitted O-grid refines only *radially* (anisotropic polar cells, 10 nm x
h_in), while an isotropic triangle mesh must also refine along the layer and
a Cartesian grid must refine both axes over the whole cross section (or
average the layer sub-cell).

Arms
* ``ogrid``   -- `nfit_study.build('ogrid', res, shell=(t, eps))` (radial
  spacing min(h, t/2) in the shell, linear grading 0.2 per unit distance
  away from it), radial PML;
* ``cart``    -- graded rectilinear Cartesian, conformal 'all' sub-cell
  averaging of the shell (no refinement), compact geometry of the graded
  study (TF/SF box, CPML);
* ``tri``     -- `graded_disk_mesh([a, a + t, 1.65 um], size)` with the same
  isotropic size field h(r) = min(h_in, t/2 + 0.2 |r - shell|), 10 PML rings
  (radial CFS-PML of `UnstructuredFIT2D`, branch feature/tri-fixes), as the
  ``tri-graded-pml`` baseline of ``../mie-cylinder-graded``.

Usage::

    python shell_study.py list
    python shell_study.py run --shard I --nshards N --outdir DIR [--arms ogrid]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mie-cylinder-mesh"))
sys.path.insert(0, str(HERE))
import mie_mesh_study as S  # noqa: E402

A, N_CYL = S.A, S.N_CYL
T_SHELL = 20e-9
N_SHELL = 3.5
RES = [10, 14, 20, 28, 40]


def shell_size(r, h):
    """Isotropic target size h(r) around the shell, shared by both arms: the
    shell gets min(h, t/2); away from it the size grows linearly (0.2 per unit
    distance, ~1.2 per cell) from the same *optical* size hs n_shell / n(r)
    (core n = 2, vacuum outside) up to the graded-study field (h in the core
    and a 2 h band, then x1.2 per cell to n h)."""
    hs = min(h, T_SHELL / 2)
    if A <= r < A + T_SHELL * (1 - 1e-9):
        return hs
    if r < A:
        return min(h, hs * N_SHELL / N_CYL + 0.2 * (A - r))
    d = r - A - T_SHELL
    base = h if d <= 2 * h else min(N_CYL * h, h * 1.2 ** ((d - 2 * h) / h))
    return min(base, hs * N_SHELL + 0.2 * d)


def tasks(arms=None):
    return [(a, p, r) for r in RES[::-1] for a in (arms or ["ogrid", "tri", "cart"]) for p in ("TM", "TE")]


def reference(pol, freqs=S.FREQS):
    from wakis.analytic.coated_cylinder import coated_cylinder_coefficients

    k = 2 * np.pi * freqs / S.C0
    q, cs = [], []
    for kk in k:
        c = coated_cylinder_coefficients(N_CYL, N_SHELL, kk * A, kk * (A + T_SHELL), pol,
                                         n_terms=S.N_MAX)  # fmt: skip
        cs.append(c)
        q.append(2 / (kk * (A + T_SHELL)) * (abs(c[0]) ** 2 + 2 * np.sum(np.abs(c[1:]) ** 2)))
    return np.array(q), np.array(cs)


def extract(dft, u0, pol, radii):
    from wakis.cylinder_harmonics import (
        fit_harmonics,
        mie_coefficients_from_fit,
        project_harmonics,
    )

    k = 2 * np.pi * S.FREQS / S.C0
    F = project_harmonics(dft.reshape(len(S.FREQS), len(radii), S.M_PTS), S.N_MAX)
    alpha, gamma, resid = fit_harmonics(F, k, radii, S.N_MAX)
    c = mie_coefficients_from_fit(alpha, gamma, np.conj(u0), S.N_MAX)
    n = np.arange(-S.N_MAX, S.N_MAX + 1)
    x2 = k * (A + T_SHELL)
    keep = np.abs(n)[None, :] <= np.array([S.n_stop(x) + 4 for x in x2])[:, None]
    q = 2 / x2 * np.sum(np.abs(np.where(keep, c, 0)) ** 2, axis=1)
    try:
        q_ref = reference(pol)[0].tolist()
    except ImportError:  # baseline tree without the coated series: in plots
        q_ref = None
    q_bare = __import__("wakis.analytic", fromlist=["x"]).cylinder_efficiencies(N_CYL, S.X, pol)[1]
    refl = np.sqrt(np.sum(np.abs(gamma) ** 2, 1) / np.sum(np.abs(alpha) ** 2, 1))
    return dict(q=q.tolist(), q_ref=q_ref, q_bare=q_bare.tolist(),
                refl=refl.tolist(), resid=resid.tolist())  # fmt: skip


def run_ogrid(pol, res):
    import nfit_study as NS

    mats = {1: [N_CYL**2, 1.0], 2: [N_SHELL**2, 1.0]}
    out = NS.run_ogrid("ogrid", pol, res, materials=mats, shell=(T_SHELL, N_SHELL**2),
                       keep_time=True)  # fmt: skip
    radii = np.asarray(NS.GEOM["compact"]["radii"])
    res_ = extract(out.pop("dft"), out.pop("u0"), pol, radii)
    out.update(res_)
    return out


def run_tri(pol, res):
    from wakis.cylinder_harmonics import circle_points
    from wakis.unstructured import UnstructuredFIT2D, graded_disk_mesh

    radii = np.array([1.25e-6, 1.4e-6, 1.55e-6])
    r_out, n_pml = 1.65e-6, 10
    t0 = time.time()
    h = S.LAM_MIN_DIEL / res
    hs = min(h, T_SHELL / 2)
    h_out = N_CYL * h

    def size(r):
        return shell_size(r, h)

    dr = size(r_out)
    mesh = graded_disk_mesh([A, A + T_SHELL, r_out], size, pml_layers=n_pml, pml_dr=dr)
    solver = UnstructuredFIT2D(
        mesh, {0: [N_CYL**2, 1.0], 1: [N_SHELL**2, 1.0]}, polarization=pol, cfl=S.CFL,
        bc="pec", pml=dict(r0=r_out, r1=r_out + n_pml * dr),
    )  # fmt: skip
    solver.add_plane_wave(S.pulse())
    pts = np.vstack([circle_points(r, S.M_PTS)[0] for r in radii])
    P = S.tri_interp(solver, pts, h_out)
    t_setup = time.time() - t0
    dt = solver.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt
    n_max = int(S.T_RUN / dt)
    times = (np.arange(n_max) + 1) * dt + shift
    acc = np.zeros((len(S.FREQS), len(pts)), complex)
    t1 = time.time()
    for n in range(n_max):
        solver.one_step()
        acc += np.outer(np.exp(-2j * np.pi * S.FREQS * times[n]), P @ solver.u) * dt
        if n % 2000 == 0 and not np.all(np.isfinite(solver.u)):
            raise FloatingPointError("unstable")
    t_run = time.time() - t1
    u_inc0 = solver.incident_field(np.zeros((1, 2)), times)
    u0 = (np.exp(-2j * np.pi * np.outer(S.FREQS, times)) @ u_inc0) * dt
    out = extract(acc, u0, pol, radii)
    out.update(dt=dt, steps=n_max, nodes=int(solver.free.size),
               dofs=int(solver.free.size + mesh.n_edges), cells=int(mesh.n_cells),
               setup_s=t_setup, run_s=t_run, h=h)  # fmt: skip
    return out


def tube_stl(r_in, r_out, height, n=720):
    """Closed, outward-oriented triangulated tube r_in < r < r_out (STL)."""
    import pyvista as pv

    th = 2 * np.pi * np.arange(n) / n
    c, s = np.cos(th), np.sin(th)
    z0, z1 = -height / 2, height / 2
    P = np.concatenate([
        np.c_[r_out * c, r_out * s, np.full(n, z0)], np.c_[r_out * c, r_out * s, np.full(n, z1)],
        np.c_[r_in * c, r_in * s, np.full(n, z0)], np.c_[r_in * c, r_in * s, np.full(n, z1)],
    ])  # fmt: skip
    ob, ot, ib, it = 0, n, 2 * n, 3 * n
    F = []
    for j in range(n):
        k = (j + 1) % n
        F += [[ob + j, ob + k, ot + k], [ob + j, ot + k, ot + j]]  # outer
        F += [[ib + j, it + k, ib + k], [ib + j, it + j, it + k]]  # inner
        F += [[ot + j, ot + k, it + k], [ot + j, it + k, it + j]]  # top
        F += [[ob + j, ib + k, ob + k], [ob + j, ib + j, ib + k]]  # bottom
    F = np.array(F)
    return pv.PolyData(P, np.c_[np.full(len(F), 3), F].ravel())


def run_cart(pol, res):
    """Graded rectilinear Cartesian FIT, conformal 'all' (sub-cell averaging
    of the shell, no extra refinement), compact geometry of the graded study."""
    import pyvista as pv

    import wakis
    from wakis.cylinder_harmonics import circle_points

    sys.path.insert(0, str(HERE.parent / "mie-cylinder-graded"))
    import graded_study as G

    radii = np.array([1.72e-6, 1.86e-6, 2.0e-6])
    t0 = time.time()
    h = S.LAM_MIN_DIEL / res
    xn = G.axis_nodes(h, True, 2.12e-6, S.N_PML)
    dz = 10 * h
    core = pv.Cylinder(radius=A, height=10e-6, direction=(0, 0, 1), resolution=720)
    grid = wakis.GridFIT3D(
        x=xn, y=xn.copy(), z=np.array([-dz / 2, dz / 2]),
        stl_solids={"core": core.triangulate(), "shell": tube_stl(A, A + T_SHELL, 10e-6)},
        stl_materials={"core": [N_CYL**2, 1.0], "shell": [N_SHELL**2, 1.0]},
        conformal="all", verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml", "cpml", "periodic"], bc_high=["cpml", "cpml", "periodic"],
        n_pml=S.N_PML, use_stl=True, cfln=S.CFL, verbose=0,
    )  # fmt: skip
    src = wakis.TFSFSource(S.pulse(), size=(2.3e-6, 2.3e-6, None), direction="+x",
                           polarization={"TM": "z", "TE": "y"}[pol], amplitude=1.0)  # fmt: skip
    src.prepare(solver)
    gx, gy = np.asarray(grid.x), np.asarray(grid.y)
    Nx, Ny, N = solver.Nx, solver.Ny, solver.N
    if pol == "TM":
        xs, ys, field = gx[:Nx], gy[:Ny], "E"
    else:
        xs, ys, field = 0.5 * (gx[:Nx] + gx[1 : Nx + 1]), 0.5 * (gy[:Ny] + gy[1 : Ny + 1]), "H"
    off = 2 * N
    pts = np.vstack([circle_points(r, S.M_PTS)[0] for r in radii])
    P = S.bilinear(xs, ys, pts, Nx)
    i0 = int(np.argmin(np.abs(gx[:Nx])))
    t_setup = time.time() - t0
    dt = solver.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt
    acc = np.zeros((len(S.FREQS), len(pts)), complex)
    line, times = [], []

    def cb(s, t):
        ts = t + dt + shift
        a = (s.E if field == "E" else s.H).array[off : off + Nx * Ny]
        acc[:] += np.outer(np.exp(-2j * np.pi * S.FREQS * ts), P @ a) * dt
        times.append(t + dt)
        line.append(float(np.asarray(src._e)[i0 + src._offset]))

    n_max = int(S.T_RUN / dt)
    t1 = time.time()
    solver.emsolve(n_max, source=[src], callback=cb)
    t_run = time.time() - t1
    times, line = np.array(times), np.array(line)
    u0 = (np.exp(-2j * np.pi * np.outer(S.FREQS, times)) @ line) * dt
    if pol == "TE":
        u0 = u0 / np.sqrt(S.MU0 / S.EPS0)
    out = extract(acc, u0, pol, radii)
    n2d = (len(xn) - 1) ** 2
    out.update(dt=dt, steps=int(solver.Nt), nodes=int(n2d), dofs=int(3 * n2d), cells=int(n2d),
               setup_s=t_setup, run_s=t_run, h=h)  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=str(HERE / "data"))
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--t-run", type=float, default=1000.0, help="fs")
    a = ap.parse_args()
    S.T_RUN = a.t_run * 1e-15
    tl = tasks(a.arms)
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    mine = ([(x.split(":")[0], x.split(":")[1], int(x.split(":")[2])) for x in a.only]
            if a.only else tl[a.shard :: a.nshards])  # fmt: skip
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for arm, pol, res in mine:
        t0 = time.time()
        try:
            out = {"ogrid": run_ogrid, "tri": run_tri, "cart": run_cart}[arm](pol, res)
        except Exception as e:
            import traceback

            out = dict(error=repr(e), tb=traceback.format_exc())
        out.update(arm="shell-" + arm, pol=pol, res=res, wall_s=time.time() - t0)
        if out.get("q_ref") is None and "q" in out:
            out["q_ref"] = out["q"]  # placeholder, recomputed in plots
            out["q_ref_missing"] = True
        (Path(a.outdir) / f"shell-{arm}_{pol}_{res}.json").write_text(json.dumps(out))
        msg = out.get("error") or "rms {:.4f}".format(
            np.sqrt(np.mean((np.array(out["q"]) / np.array(out["q_ref"]) - 1) ** 2)))  # fmt: skip
        print(arm, pol, res, msg, out.get("dofs"), out.get("steps"), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
