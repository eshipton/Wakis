"""
Mie scattering by an infinite dielectric cylinder (n = 2, a = 1 um, x = ka in
2..10): triangular (unstructured) FIT versus the Cartesian FIT grid.

Arms (same target edge length h, same pulse, same run window, same
post-processing):

* ``staircase``  -- `SolverFIT3D`, STL cylinder, staircase cell masks;
* ``conformal``  -- `SolverFIT3D`, ``conformal='all'`` (diagonal subpixel
  averaging of eps);
* ``tensor``     -- `SolverFIT3D`, ``conformal='tensor'`` (full averaged
  inverse-permittivity tensor);
* ``tri-voronoi``/``tri-bary`` -- `UnstructuredFIT2D` on an
  interface-conforming Delaunay mesh (`disk_mesh`), Voronoi or barycentric
  node masses.

Excitation: Cartesian -- `TFSFSource` box (half width 1.25 um) on a thin
periodic-z slab, CPML; triangles -- scattered-field plane wave
(`UnstructuredFIT2D.add_plane_wave`), first-order ABC on the circle
r = 2.7 um (where the Cartesian CPML starts).

Extraction (identical for all arms): the scattered field (E_z for TM, H_z for
TE) is DFT'd on 3 circles r = 1.9, 2.2, 2.5 um (256 points each, linear
interpolation on the triangles / bilinear on the Cartesian samples),
projected on e^{i n theta}, fitted as outgoing H^(1) + incoming H^(2) waves;
c_n follows from the scattering-matrix relation (de-embeds the boundary
reflections) and Q_sca = (2/x) sum |c_n|^2. See `wakis.cylinder_harmonics`.
Cartesian flux-box Q_sca is a cross-check (``--flux`` runs, 4 cells along z).

Usage::

    python mie_mesh_study.py run [--quick] [--procs N] [--res 10 14 ...]
    python mie_mesh_study.py plot

Run locally only under ``bin/memguard`` with ``--quick``.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import mu_0 as MU0

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"

A = 1.0e-6  # cylinder radius [m]
N_CYL = 2.0
X = np.linspace(2.0, 10.0, 41)  # size parameters k0 a
FREQS = X * C0 / (2 * np.pi * A)
F_CENTER = 0.5 * (FREQS[0] + FREQS[-1])
F_WIDTH = 0.25 * (FREQS[-1] - FREQS[0])
X_SNAP = 6.0  # field maps
F_SNAP = X_SNAP * C0 / (2 * np.pi * A)
RADII = np.array([1.9e-6, 2.2e-6, 2.5e-6])  # extraction circles [m]
M_PTS = 256
N_MAX = 30  # harmonics fitted
R_OUT = 2.7e-6  # triangle ABC radius = Cartesian CPML start [m]
TF_HALF = 1.25e-6  # Cartesian TF/SF box half width [m]
N_PML = 10
T_RUN = 400e-15
DECAY = 1e-7
LAM_MIN_DIEL = 2 * np.pi * A / (X.max() * N_CYL)  # 0.314 um
RES = [10, 14, 20, 28, 40]  # cells per wavelength in the dielectric at x = 10
ARMS = ["staircase", "conformal", "tensor", "tri-voronoi", "tri-barycentric"]
POLS = ["TM", "TE"]
CFL = 0.95


def n_stop(x):
    return int(np.floor(x + 4.0 * x ** (1.0 / 3.0) + 2.0))


def pulse():
    from wakis.sources import GaussianSource

    return GaussianSource(F_CENTER, F_WIDTH)


def series(pol, x=X):
    from wakis.analytic import cylinder_coefficients, cylinder_efficiencies

    q = cylinder_efficiencies(N_CYL, x, pol)[1]
    c = [
        cylinder_coefficients(N_CYL, xx, pol, n_terms=N_MAX) for xx in np.atleast_1d(x)
    ]
    return q, np.array(c)


# --------------------------------------------------------------------------
# post-processing shared by all arms
# --------------------------------------------------------------------------
def extract(dft, u0, pol):
    """dft: (n_freq, n_radii * M) Wakis-convention DFT on the circles; u0:
    Wakis-convention incident phasor at the origin. Returns a dict."""
    from wakis.cylinder_harmonics import (
        fit_harmonics,
        mie_coefficients_from_fit,
        project_harmonics,
        q_sca_from_coefficients,
    )

    eps_b = 1.0
    k = 2 * np.pi * FREQS * np.sqrt(eps_b) / C0
    F = project_harmonics(dft.reshape(len(FREQS), len(RADII), M_PTS), N_MAX)
    alpha, gamma, resid = fit_harmonics(F, k, RADII, N_MAX)
    u0b = np.conj(u0)
    c = mie_coefficients_from_fit(alpha, gamma, u0b, N_MAX)
    c_naive = mie_coefficients_from_fit(alpha, gamma, u0b, N_MAX, deembed=False)
    n = np.arange(-N_MAX, N_MAX + 1)
    keep = np.abs(n)[None, :] <= np.array([n_stop(x) + 4 for x in X])[:, None]
    q = q_sca_from_coefficients(np.where(keep, c, 0), X)
    q_naive = q_sca_from_coefficients(np.where(keep, c_naive, 0), X)
    refl = np.sqrt(np.sum(np.abs(gamma) ** 2, 1) / np.sum(np.abs(alpha) ** 2, 1))
    # c_n versus c_-n (mirror symmetry of the problem)
    asym = np.abs(c[:, N_MAX + 1 :] - c[:, N_MAX - 1 :: -1]).max(1) / np.abs(c).max(1)
    q_ref, c_ref = series(pol)
    c_half = 0.5 * (c[:, N_MAX:] + c[:, N_MAX::-1])  # n = 0..N_MAX
    nn = np.arange(N_MAX + 1)
    w = np.where(nn > 0, 2.0, 1.0)
    # coefficient error, weighted like Q_sca
    c_err = np.sqrt(
        np.sum(w * np.abs(c_half - c_ref) ** 2, 1) / np.sum(w * np.abs(c_ref) ** 2, 1)
    )
    return dict(
        q=q.tolist(), q_naive=q_naive.tolist(), q_ref=q_ref.tolist(),
        refl=refl.tolist(), resid=resid.tolist(), asym=asym.tolist(),
        c_err=c_err.tolist(),
        c_re=c_half[:, :16].real.tolist(), c_im=c_half[:, :16].imag.tolist(),
        c_ref_re=c_ref[:, :16].real.tolist(), c_ref_im=c_ref[:, :16].imag.tolist(),
    )  # fmt: skip


def dft_kernel(t):
    return np.exp(-2j * np.pi * FREQS * t)


# --------------------------------------------------------------------------
# triangles
# --------------------------------------------------------------------------
def tri_interp(solver, points, h):
    """Sparse linear interpolation of free-node values at points."""
    import scipy.sparse as sp

    mesh = solver.mesh
    P = mesh.nodes[mesh.cells]  # (nc, 3, 2)
    rc = np.hypot(*P.mean(1).T)
    rows, cols, vals = [], [], []
    for r in np.unique(np.round(np.hypot(*points.T), 12)):
        sel = np.flatnonzero(np.abs(np.hypot(*points.T) - r) < 1e-9)
        cand = np.flatnonzero(np.abs(rc - r) < 2.5 * h)
        X0 = P[cand, 0]
        Am = np.transpose(P[cand, 1:] - X0[:, None], (0, 2, 1))
        Ainv = np.linalg.inv(Am)
        for i in sel:
            lam = np.einsum("nij,nj->ni", Ainv, points[i] - X0)
            lam = np.concatenate([1 - lam.sum(1, keepdims=True), lam], 1)
            j = np.argmax(lam.min(1))
            assert lam[j].min() > -1e-9, "point not found"
            rows += [i] * 3
            cols += list(mesh.cells[cand[j]])
            vals += list(lam[j])
    full = sp.csr_matrix((vals, (rows, cols)), shape=(len(points), mesh.n_nodes))
    return full[:, solver.free]


def run_tri(hodge, pol, res, maps=False):
    from wakis.cylinder_harmonics import circle_points
    from wakis.unstructured import UnstructuredFIT2D
    from wakis.unstructured.meshers import disk_mesh

    t0 = time.time()
    h = LAM_MIN_DIEL / res
    mesh = disk_mesh([A, R_OUT], h)
    solver = UnstructuredFIT2D(
        mesh, {0: [N_CYL**2, 1.0]}, polarization=pol, bc="abc", hodge=hodge, cfl=CFL
    )
    src = pulse()
    solver.add_plane_wave(src)
    pts = np.vstack([circle_points(r, M_PTS)[0] for r in RADII])
    P = tri_interp(solver, pts, h)
    t_setup = time.time() - t0
    dt = solver.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt  # Hz lives at half steps
    acc = np.zeros((len(FREQS), len(pts)), complex)
    acc_map = np.zeros(solver.free.size, complex) if maps else None
    n_max = int(T_RUN / dt)
    t_end = src.end_time + 2 * A * N_CYL / C0
    e_max, n = 0.0, 0
    times = []
    t1 = time.time()
    for n in range(n_max):
        solver.one_step()
        ts = (n + 1) * dt + shift
        times.append(ts)
        acc += np.outer(dft_kernel(ts), P @ solver.u) * dt
        if maps:
            acc_map += np.exp(-2j * np.pi * F_SNAP * ts) * solver.u * dt
        if n % 50 == 0:
            e = solver.energy()
            if not np.isfinite(e):
                raise FloatingPointError("unstable")
            e_max = max(e_max, e)
            e_last = e
            if ts > t_end and e < DECAY * e_max:
                break
    t_run = time.time() - t1
    times = np.array(times)
    u_inc0 = solver.incident_field(np.zeros((1, 2)), times)  # at the origin
    u0 = (np.exp(-2j * np.pi * np.outer(FREQS, times)) @ u_inc0) * dt
    out = extract(acc, u0, pol)
    n_free, n_edges = solver.free.size, mesh.n_edges
    out.update(
        arm="tri-" + hodge, pol=pol, res=res, h=h, dt=dt, steps=n + 1,
        nodes=int(n_free), dofs=int(n_free + n_edges), cells=int(mesh.n_cells),
        setup_s=t_setup, run_s=t_run, e_final=float(e_last / e_max),
        area_err=float(mesh.cell_volume[mesh.cell_tags == 0].sum() / (np.pi * A**2) - 1),
    )  # fmt: skip
    if maps:
        u0s = (np.exp(-2j * np.pi * F_SNAP * times) @ u_inc0) * dt
        np.savez_compressed(
            DATA / f"map_{out['arm']}_{pol}_{res}.npz",
            nodes=mesh.nodes, cells=mesh.cells, free=solver.free,
            u=acc_map, u0=u0s, tags=mesh.cell_tags,
        )  # fmt: skip
    return out


# --------------------------------------------------------------------------
# Cartesian
# --------------------------------------------------------------------------
def bilinear(xs, ys, points, nx):
    """Sparse bilinear interpolation from samples on the tensor grid xs x ys
    (flat index i + nx j) to points."""
    import scipy.sparse as sp

    rows, cols, vals = [], [], []
    for p, (px, py) in enumerate(points):
        i = np.searchsorted(xs, px) - 1
        j = np.searchsorted(ys, py) - 1
        tx = (px - xs[i]) / (xs[i + 1] - xs[i])
        ty = (py - ys[j]) / (ys[j + 1] - ys[j])
        for di, dj, w in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                          (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):  # fmt: skip
            rows.append(p)
            cols.append((i + di) + nx * (j + dj))
            vals.append(w)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(points), nx * len(ys)))


def run_cart(conf, pol, res, maps=False, flux=False):
    import pyvista as pv

    import wakis
    from wakis.cylinder_harmonics import circle_points

    t0 = time.time()
    h = LAM_MIN_DIEL / res
    nh = int(np.ceil(R_OUT / h - 1e-9)) + N_PML
    half = nh * h
    nz = 4 if flux else 1
    dz = h if flux else 10 * h  # z-invariant field: dz only enters the CFL
    cyl = pv.Cylinder(radius=A, height=10e-6, direction=(0, 0, 1), resolution=720)
    cyl = cyl.triangulate()
    grid = wakis.GridFIT3D(
        -half, half, -half, half, -nz * dz / 2, nz * dz / 2, 2 * nh, 2 * nh, nz,
        stl_solids={"cyl": cyl}, stl_materials={"cyl": [N_CYL**2, 1.0]},
        conformal={"staircase": False, "conformal": "all", "tensor": "tensor"}[conf],
        verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml", "cpml", "periodic"], bc_high=["cpml", "cpml", "periodic"],
        n_pml=N_PML, use_stl=True, cfln=CFL, verbose=0,
    )  # fmt: skip
    src = wakis.TFSFSource(
        pulse(), size=(2 * TF_HALF, 2 * TF_HALF, None), direction="+x",
        polarization={"TM": "z", "TE": "y"}[pol], amplitude=1.0,
    )  # fmt: skip
    src.prepare(solver)
    gx, gy = np.asarray(grid.x), np.asarray(grid.y)
    Nx, Ny, N = solver.Nx, solver.Ny, solver.N
    if pol == "TM":
        xs, ys, field, off = gx[:Nx], gy[:Ny], "E", 2 * N
    else:
        xs = 0.5 * (gx[:Nx] + gx[1 : Nx + 1])
        ys = 0.5 * (gy[:Ny] + gy[1 : Ny + 1])
        field, off = "H", 2 * N
    pts = np.vstack([circle_points(r, M_PTS)[0] for r in RADII])
    P = bilinear(xs, ys, pts, Nx)
    i0 = int(np.argmin(np.abs(gx[:Nx])))
    assert abs(gx[i0]) < 1e-12 * half
    monitors = []
    if flux:
        lz = 2 * dz
        r_f = 1.6e-6
        fm = wakis.FluxMonitor(FREQS, size=(2 * r_f, 2 * r_f, lz))
        monitors.append(fm)
    t_setup = time.time() - t0
    dt = solver.dt
    shift = 0.0 if pol == "TM" else -0.5 * dt
    acc = np.zeros((len(FREQS), len(pts)), complex)
    acc_map = np.zeros(Nx * Ny, complex) if maps else None
    line = []
    times = []

    def cb(s, t):
        ts = t + dt + shift
        a = (s.E if field == "E" else s.H).array[off : off + Nx * Ny]
        acc[:] += np.outer(dft_kernel(ts), P @ a) * dt
        if maps:
            acc_map[:] += np.exp(-2j * np.pi * F_SNAP * ts) * a * dt
        times.append(t + dt)
        line.append(float(np.asarray(src._e)[i0 + src._offset]))
        if len(times) % 200 == 0:
            e = float(s.field_energy())
            energy[0] = max(energy[0], e)
            energy[1] = e

    energy = [0.0, 0.0]

    t1 = time.time()
    solver.emsolve(
        int(T_RUN / dt), source=[src], monitors=monitors, callback=cb,
        stop_when_energy_decayed=(DECAY, 50),
    )  # fmt: skip
    t_run = time.time() - t1
    times, line = np.array(times), np.array(line)
    if not np.all(np.isfinite(acc)):
        raise FloatingPointError("unstable")
    u0 = (np.exp(-2j * np.pi * np.outer(FREQS, times)) @ line) * dt  # E_inc
    eta = np.sqrt(MU0 / EPS0)
    if pol == "TE":  # H_z,inc = +E_y,inc / eta for +x incidence
        u0 = u0 / eta
    out = extract(acc, u0, pol)
    out.update(
        arm=conf, pol=pol, res=res, h=h, dt=dt, steps=int(solver.Nt),
        nodes=int(4 * nh * nh), dofs=int(3 * 4 * nh * nh), cells=int(4 * nh * nh),
        setup_s=t_setup, run_s=t_run, nz=nz, e_final=energy[1] / energy[0],
    )  # fmt: skip
    if flux:
        phi = src.incident_flux_density(FREQS)
        out["q_flux"] = (fm.get_flux() / (lz * phi) / (2 * A)).tolist()
    if maps:
        u0s = (np.exp(-2j * np.pi * F_SNAP * times) @ line) * dt
        u0s = u0s / eta if pol == "TE" else u0s
        np.savez_compressed(
            DATA / f"map_{conf}_{pol}_{res}.npz", xs=xs, ys=ys,
            u=acc_map.reshape(Ny, Nx), u0=u0s,
        )  # fmt: skip
    return out


def task(args):
    global T_RUN
    arm, pol, res, maps, flux, t_run = args
    T_RUN = t_run
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    try:
        if arm.startswith("tri-"):
            out = run_tri(arm[4:], pol, res, maps)
        else:
            out = run_cart(arm, pol, res, maps, flux)
    except Exception as e:  # report, do not kill the pool
        import traceback

        out = dict(arm=arm, pol=pol, res=res, error=repr(e), tb=traceback.format_exc())
    out["flux_run"] = flux
    out["wall_s"] = time.time() - t0
    name = f"{arm}_{pol}_{res}{'_flux' if flux else ''}.json"
    (DATA / name).write_text(json.dumps(out))
    msg = out.get("error") or (
        f"q err max {np.max(np.abs(np.array(out['q']) / np.array(out['q_ref']) - 1)):.3e}"
    )
    print(
        f"{arm:12s} {pol} res {res:3d} flux={flux}: {msg}  ({out['wall_s']:.0f} s)",
        flush=True,
    )
    return out


def main_run(args):
    DATA.mkdir(exist_ok=True)
    t_run = args.t_run * 1e-15 if args.t_run else T_RUN
    tasks = []
    for res in args.res:
        for arm in args.arms:
            for pol in POLS:
                if arm == "tensor" and pol == "TM":
                    continue  # identical to 'conformal' (E_z never crosses)
                tasks.append((arm, pol, res, res == args.map_res, False, t_run))
    for res in args.flux_res:
        for arm in ("staircase", "conformal"):
            for pol in POLS:
                tasks.append((arm, pol, res, False, True, t_run))
    tasks.sort(key=lambda t: -t[2] * (4 if t[4] else 1))  # biggest first
    with Pool(args.procs) as pool:
        for _ in pool.imap_unordered(task, tasks):
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "plot"])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--res", type=int, nargs="+", default=RES)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--flux-res", type=int, nargs="*", default=[10, 20])
    ap.add_argument("--map-res", type=int, default=20)
    ap.add_argument("--t-run", type=float, default=None, help="fs")
    a = ap.parse_args()
    if a.mode == "run":
        main_run(a)
    else:
        from plot_study import main_plot

        main_plot()
