"""
Prism FIT vs Cartesian conformal FIT vs tetrahedral FIT at equal accuracy:
a closed PEC cylindrical cavity (radius a = 2 um, height H = 2 um) with a
coaxial dielectric rod (radius b = 1 um, n = 2) spanning its full height.

Reference: the 20 lowest resonances of `rod_cavity_hybrid` (TM_0 modes for
p = 0 and the hybrid HE/EH modes for p = 1, 2: exact Bessel determinant).

Every arm runs the same time-domain experiment: three broadband dipoles at
fixed off-symmetry points, six probes x 3 E components, T_RUN of ring-down;
harmonic inversion (`wakis.resonances.harminv`, FDM) of every probe signal
after the pulse; for each exact mode the nearest extracted frequency gives
its relative error, and the metric is the RMS over the 20 modes. The raw
leapfrog frequencies are used (time dispersion included, as a user gets);
the dispersion-corrected spatial error is stored too.

Arms (res = cells per minimum wavelength in the rod, h = lambda_min / res):

* prism        -- `disk_mesh` + ODT smoothing, uniform z (dz = h), PrismFIT3D
* prism-graded -- `graded_disk_mesh` (h in the rod and a 2 h band, x1.2
                  growth to 2 h), ODT smoothing, dz = h
* cart         -- SolverFIT3D, uniform h, conformal 'all' (PEC wall: bg pec
                  around a vacuum cylinder STL)
* cart-graded  -- rectilinear grid graded in x and y (h over |x| < b + 2h,
                  x1.2 growth to 2 h), dz = h
* tet          -- gmsh tetrahedra (size h), UnstructuredFIT3D with the Whitney
                  (consistent) Hodge: the Voronoi Hodge has negative dual
                  lengths on gmsh meshes and is unstable
* tet-graded   -- gmsh size field h in the rod, 2 h outside

Usage::

    python prism_study.py list
    python prism_study.py run --task I --outdir out
    python prism_study.py eig --res 10          # prism extraction check
"""

import argparse
import functools
import json
import os
import sys
import time

import numpy as np
from scipy.constants import c

A_CAV, B_ROD, H_CAV, EPS = 2e-6, 1e-6, 2e-6, 4.0
N_MODES = 20
T_RUN = 400 * B_ROD / c  # 1.33 ps
SRC_POS = np.array(
    [[0.43, -0.27, 0.31], [-1.21, 0.62, 0.77], [0.17, 1.37, 0.58]]
) * 1e-6 + np.array([0, 0, 0.0])
SRC_DIR = np.array([[0.3, 0.5, 1.0], [-0.7, 0.2, 0.6], [0.5, -0.8, 0.3]])
PRB_POS = np.array(
    [[0.61, 0.33, 0.53], [-0.38, -0.52, 0.29], [1.42, -0.31, 0.81],
     [-0.21, 1.53, 0.44], [0.07, -0.09, 0.66], [-1.15, -1.03, 0.37]]
) * 1e-6  # fmt: skip
# z of sources/probes: fractions of H (avoid the mid plane symmetry)
SRC_POS[:, 2] *= H_CAV / 1e-6
PRB_POS[:, 2] *= H_CAV / 1e-6

ARMS = ["prism", "prism-graded", "cart", "cart-graded", "tet", "tet-graded"]
RES = {
    "prism": [8, 11, 16, 22, 32, 45],
    "prism-graded": [8, 11, 16, 22, 32, 45],
    "cart": [8, 11, 16, 22, 32, 45, 64],
    "cart-graded": [8, 11, 16, 22, 32, 45, 64],
    "tet": [8, 11, 16],
    "tet-graded": [8, 11, 16],
}


@functools.lru_cache(maxsize=1)
def exact_k():
    from wakis.unstructured.analytic import rod_cavity_hybrid

    return rod_cavity_hybrid(A_CAV, B_ROD, EPS, H_CAV, n_modes=N_MODES, kmax=8 / A_CAV)


def band():
    k = exact_k()
    f = k * c / (2 * np.pi)
    return 0.9 * f[0], 1.04 * f[-1]


def h_of(res):
    fmax = band()[1]
    return c / (fmax * np.sqrt(EPS)) / res


def pulse():
    from wakis.sources import GaussianSource

    fmin, fmax = band()
    return GaussianSource(0.5 * (fmin + fmax), 0.45 * (fmax - fmin))


# ----------------------------------------------------------------- meshes
def size_field(h):
    def size(r):
        d = r - B_ROD - 2 * h
        return h if d <= 0 else min(2 * h, h * 1.2 ** (d / h))

    return size


def prism_solver(res, graded):
    from wakis.unstructured import PrismFIT3D, PrismMesh, disk_mesh, graded_disk_mesh
    from wakis.unstructured.smoothing import odt_smooth, radial_region

    h = h_of(res)
    radii = [B_ROD, A_CAV]
    m2 = graded_disk_mesh(radii, size_field(h)) if graded else disk_mesh(radii, h)
    m2 = odt_smooth(m2, radial_region(radii), 30)
    nz = int(np.ceil(H_CAV / h))
    pm = PrismMesh(m2, np.linspace(0, H_CAV, nz + 1), merge_tol=0.05)
    return PrismFIT3D(pm, {0: [EPS, 1.0]}, bc_xy="pec", bc_z="pec")


def graded_axis(h):
    """Nodes on [-a - h, a + h]: h for |x| < b + 2h, x1.2 growth to 2h."""
    xs = [0.0]
    x = 0.0
    lim = A_CAV + 1.5 * h
    while x < lim:
        d = x - B_ROD - 2 * h
        step = h if d <= 0 else min(2 * h, h * 1.2 ** (d / h))
        x += step
        xs.append(x)
    xs = np.array(xs)
    # small offset from the symmetry axis (generic cuts, as in the tests)
    return np.r_[-xs[:0:-1], xs] + 0.0123 * h


def cart_solver(res, graded):
    import pyvista as pv

    import wakis

    h = h_of(res)
    if graded:
        xn = graded_axis(h)
    else:
        n = int(np.ceil((2 * A_CAV + 3 * h) / h))
        xn = (np.arange(n + 1) - n / 2) * h + 0.0123 * h
    nz = int(np.ceil(H_CAV / h))
    zn = np.linspace(0, H_CAV, nz + 1)

    def cyl(r):
        return pv.Cylinder(
            center=(0, 0, H_CAV / 2), direction=(0, 0, 1), radius=r,
            height=3 * H_CAV, resolution=1440,
        ).triangulate()  # fmt: skip

    grid = wakis.GridFIT3D(
        x=xn, y=xn.copy(), z=zn,
        stl_solids={"cavity": cyl(A_CAV), "rod": cyl(B_ROD)},
        stl_materials={"cavity": "vacuum", "rod": [EPS, 1.0, 0.0]},
        conformal="all", verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg="pec",
        use_stl=True, verbose=0,
    )  # fmt: skip
    return solver, grid


def gmsh_cavity(h, graded, filename):
    from wakis.unstructured import gmsh_mesh

    def build(gmsh):
        occ = gmsh.model.occ
        outer = occ.addCylinder(0, 0, 0, 0, 0, H_CAV, A_CAV)
        rod = occ.addCylinder(0, 0, 0, 0, 0, H_CAV, B_ROD)
        out, _ = occ.fragment([(3, outer)], [(3, rod)])
        occ.synchronize()
        for _, vol in out:
            bb = gmsh.model.getBoundingBox(3, vol)
            tag = 1 if (bb[3] - bb[0]) > 1.5 * B_ROD * 2 else 0
            gmsh.model.addPhysicalGroup(3, [vol], tag + 1)
        if graded:
            size = size_field(h)

            def cb(dim, tag, x, y, z, lc):
                return float(size(np.hypot(x, y)))

            gmsh.model.mesh.setSizeCallback(cb)

    mesh = gmsh_mesh(build, 2 * h if graded else h, filename, dim=3)
    mesh.cell_tags -= 1
    return mesh


def tet_solver(res, graded, workdir):
    from wakis.unstructured import UnstructuredFIT3D

    h = h_of(res)
    mesh = gmsh_cavity(h, graded, os.path.join(workdir, f"cav_{res}_{graded}.msh"))
    # gmsh meshes are not well centred: the diagonal Voronoi Hodge has
    # negative dual lengths (unstable), so the consistent Whitney mass
    # matrices are used (one sparse LU solve per step)
    return UnstructuredFIT3D(mesh, {0: [EPS, 1.0]}, bc="pec", hodge="whitney")


# ----------------------------------------------------------------- extraction
def extract(signals, dt, t0):
    """
    Harminv every column of every signal.

    Returns a list of (frequency [Hz], decay [Hz], |amplitude|, signal index)
    for all fitted modes in the band.
    """
    from wakis.resonances import harminv

    fmin, fmax = band()
    # decimate to ~8 samples per period of fmax (keeps FDM cheap)
    step = max(1, int(1 / (8 * fmax * dt)))
    found = []
    k = 0
    for sig in signals:
        sig = np.asarray(sig)[::step]
        for j in range(sig.shape[1]):
            try:
                r = harminv(sig[:, j], step * dt, fmin, fmax, t0=t0)
            except Exception:
                k += 1
                continue
            a = np.abs(r.amplitude)
            for F, amp in zip(r.complex_frequency, a / a.max()):
                found.append((float(F.real), float(F.imag), float(amp), k))
            k += 1
    return found


def clusters(found, rel=1e-4, min_support=2, min_amp=1e-3):
    """
    Robust mode list: fitted frequencies of all signals grouped within
    `rel`; a mode needs fits in >= `min_support` signals with relative
    amplitude >= `min_amp`; its frequency is the amplitude-weighted mean.
    """
    pts = sorted((f, a, k) for f, g, a, k in found if a >= min_amp)
    groups, cur = [], []
    for p in pts:
        if cur and p[0] - cur[-1][0] > rel * p[0]:
            groups.append(cur)
            cur = []
        cur.append(p)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        if len({k for _, _, k in g}) >= min_support:
            f = np.array([p[0] for p in g])
            w = np.array([p[1] for p in g])
            out.append(float(np.average(f, weights=w)))
    return np.array(out)


def assign(f_ex, fc, dup_penalty=5e-3):
    """
    Optimal one-to-one matching (Hungarian) of the exact modes to the
    extracted clusters minimizing the summed |relative error|. Every cluster
    may also be used a second time at a cost `dup_penalty` (a degenerate pair
    that the discretization or the extraction did not split), so a missing
    partner does not force a match to a neighbouring mode.
    """
    from scipy.optimize import linear_sum_assignment

    cand = np.r_[fc, fc]
    cost = np.abs(cand[None, :] / f_ex[:, None] - 1)
    cost[:, fc.size :] += dup_penalty
    rows, cols = linear_sum_assignment(cost)
    near = np.empty_like(f_ex)
    near[rows] = cand[cols]
    return near


def score(found, dt):
    """
    RMS relative error of the 20 exact modes (with multiplicity), matched to
    the robust clusters of `clusters` by `assign`.
    """
    k = exact_k()
    f_ex = k * c / (2 * np.pi)
    fc = clusters(found) if (len(found) and isinstance(found[0], tuple)) else np.asarray(found)
    if fc.size == 0:
        return dict(rms=np.nan)
    near = assign(f_ex, fc)
    err = near / f_ex - 1
    # spatial (leapfrog-dispersion-free) frequency: sin(w dt / 2) = w_s dt / 2
    fs = 2 / (2 * np.pi * dt) * np.sin(np.pi * near * dt)
    err_s = fs / f_ex - 1
    return dict(
        rms=float(np.sqrt(np.mean(err**2))), max=float(np.abs(err).max()),
        rms_spatial=float(np.sqrt(np.mean(err_s**2))),
        err=err.tolist(), f_exact=f_ex.tolist(), f_found=near.tolist(),
        n_clusters=int(fc.size),
    )  # fmt: skip


# ----------------------------------------------------------------- runners
def run_prism(res, graded):
    t0 = time.time()
    s = prism_solver(res, graded)
    t_setup = time.time() - t0
    src = pulse()
    for p, d in zip(SRC_POS, SRC_DIR):
        s.add_point_dipole(p, d, src)
    probes = [s.add_probe(PRB_POS, comp) for comp in "xyz"]
    n_steps = int(T_RUN / s.dt)
    t1 = time.time()
    s.run(n_steps)
    t_run = time.time() - t1
    start = int(src.end_time / s.dt) + 1
    sig = [s.probe_signal(p)[start:] for p in probes]
    found = extract(sig, s.dt, (start + 1) * s.dt)
    out = score(found, s.dt)
    out["found"] = found
    out.update(
        unknowns=int(s.n_unknowns), free_edges=int(s.free.size),
        cells=int(s.mesh.n_cells), cells2d=int(s.mesh.c2.n_cells),
        nodes2d=int(s.mesh.c2.n_nodes), layers=int(s.mesh.K),
        merged=int(s.mesh.c2.n_removed), dt=s.dt, steps=n_steps,
        setup_s=t_setup, run_s=t_run, h=h_of(res),
    )  # fmt: skip
    return out


def run_cart(res, graded):
    from wakis.sources import CurrentSource

    t0 = time.time()
    solver, grid = cart_solver(res, graded)
    t_setup = time.time() - t0
    src = pulse()
    sources = [
        CurrentSource(src, component=comp, center=tuple(p), amplitude=float(d[i]))
        for p, d in zip(SRC_POS, SRC_DIR / np.linalg.norm(SRC_DIR, axis=1)[:, None])
        for i, comp in enumerate("xyz")
    ]
    x, y, z = (np.asarray(v) for v in (grid.x, grid.y, grid.z))
    idx = []
    for comp in "xyz":
        a = "xyz".index(comp)
        co = [x[:-1], y[:-1], z[:-1]]
        co[a] = 0.5 * ((x, y, z)[a][:-1] + (x, y, z)[a][1:])
        for p in PRB_POS:
            idx.append(tuple(int(np.argmin(np.abs(q - v))) for q, v in zip(co, p)) + (comp,))
    samples = []

    def cb(s, t):
        samples.append([float(s.E[i]) for i in idx])

    n_steps = int(T_RUN / solver.dt)
    t1 = time.time()
    solver.emsolve(n_steps, source=sources, callback=cb)
    t_run = time.time() - t1
    start = int(src.end_time / solver.dt) + 1
    sig = np.asarray(samples)[start:]
    found = extract([sig], solver.dt, (start + 1) * solver.dt)
    out = score(found, solver.dt)
    out["found"] = found
    Nx, Ny, Nz = len(x) - 1, len(y) - 1, len(z) - 1
    out.update(
        unknowns=int(6 * Nx * Ny * Nz), cells=int(Nx * Ny * Nz),
        dt=solver.dt, steps=n_steps, setup_s=t_setup, run_s=t_run, h=h_of(res),
    )  # fmt: skip
    return out


def run_tet(res, graded, workdir):
    t0 = time.time()
    s = tet_solver(res, graded, workdir)
    t_setup = time.time() - t0
    src = pulse()
    for p, d in zip(SRC_POS, SRC_DIR):
        s.add_point_dipole(p, d, src)
    probes = [s.add_probe(PRB_POS, k) for k in range(3)]
    n_steps = int(T_RUN / s.dt)
    t1 = time.time()
    s.run(n_steps)
    t_run = time.time() - t1
    start = int(src.end_time / s.dt) + 1
    sig = [s.probe_signal(p)[start:] for p in probes]
    found = extract(sig, s.dt, (start + 1) * s.dt)
    out = score(found, s.dt)
    out["found"] = found
    m = s.mesh
    out.update(
        unknowns=int(m.n_edges + m.n_faces), free_edges=int(s.free.size),
        massless=int(s.massless.size), cells=int(m.n_cells), dt=s.dt,
        steps=n_steps, setup_s=t_setup, run_s=t_run, h=h_of(res),
    )  # fmt: skip
    return out


# coarse resolutions added later (appended so that task indices stay stable)
EXTRA = [(a, r) for a in ("prism", "prism-graded", "cart", "cart-graded") for r in (5, 6)]
EXTRA += [(a, r) for a in ("tet", "tet-graded") for r in (4, 5, 6)]
EXTRA += [(a, r) for a in ("prism", "prism-graded", "cart", "cart-graded") for r in (4,)]


def tasks():
    return [(arm, r) for arm in ARMS for r in RES[arm]] + EXTRA


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run", "eig"])
    ap.add_argument("--task", type=int)
    ap.add_argument("--arm")
    ap.add_argument("--res", type=float)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for i, t in enumerate(tasks()):
            print(i, *t)
        return
    if args.cmd == "eig":
        # discrete eigenfrequencies of the prism operator (no extraction)
        s = prism_solver(args.res, (args.arm or "").endswith("graded"))
        f, _ = s.eigenmodes(N_MODES + 6, f_target=0.8 * band()[0])
        out = score(np.asarray(f), 1e-30)
        out.update(arm=(args.arm or "prism") + "-eig", res=args.res, f_eig=f.tolist())
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, f"eig_{args.arm or 'prism'}_{args.res:g}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        print(json.dumps({k: out[k] for k in ("rms", "max")}))
        return
    arm, res = (args.arm, args.res) if args.arm else tasks()[args.task]
    os.makedirs(args.outdir, exist_ok=True)
    graded = arm.endswith("graded")
    t0 = time.time()
    if arm.startswith("prism"):
        out = run_prism(res, graded)
    elif arm.startswith("cart"):
        out = run_cart(res, graded)
    else:
        out = run_tet(res, graded, args.outdir)
    out.update(arm=arm, res=res, wall_s=time.time() - t0, host=os.uname().nodename)
    fn = os.path.join(args.outdir, f"{arm}_{res:g}.json")
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1)
    print(arm, res, "rms %.3e" % out["rms"], "unknowns", out["unknowns"],
          "steps", out["steps"], "run %.1f s" % out["run_s"])  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
