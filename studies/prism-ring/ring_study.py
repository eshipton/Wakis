"""
Showcase: resonances of a silicon micro-ring (planar photonics), prism FIT
vs Cartesian conformal FIT.

Geometry: Si strip ring (eps = 12.1), radius R = 1.5 um (centre of the
strip), width w = 0.5 um, thickness t = 0.22 um, in a homogeneous oxide
cladding (eps = 2.1). The closed computational domain is a PEC cylinder of
radius R + w/2 + 1.25 um and |z| <= 1 um; the TE-like modes are even in z,
so only z >= 0 is simulated with a PMC plane at z = 0. The outer 0.45 um
are absorbing (prism: matched lossy annulus + first-order ABC laterally,
split-field z PML on top; Cartesian: CPML on the box faces, whose interior
ends where the prism absorber starts). The open resonator's modes (radiation
Q > 1e3) are identified by Q > 500; the evanescent tail at the absorbers
is ~e^-6 of its peak.

* prism  -- graded triangles in the plane (h_ring in the strip and a band
            around it, growing to 3 h_ring), quarter domain with the same
            symmetry planes as the Cartesian arm, z layers: dz_core = h in
            the strip, growing x1.25 to 0.1 um in the cladding.
* cart   -- SolverFIT3D, uniform h, conformal 'all', quarter domain (two
            runs: PMC/PMC on the planes x = 0, y = 0 selects the even
            azimuthal orders m, PMC/PEC the odd ones) and the z = 0 PMC plane.

A narrow-band pulse around 193 THz (1.55 um) excites the azimuthal orders
m ~ 13-16 of the fundamental TE-like mode; harmonic inversion of several
probes gives the resonance frequencies.

Usage::

    python ring_study.py list
    python ring_study.py run --task I --outdir out [--gpu]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.constants import c

EPS_SI, EPS_OX = 12.1, 2.1
R, W, T = 1.5e-6, 0.5e-6, 0.22e-6
R_IN, R_OUT = R - W / 2, R + W / 2
A_DOM = R_OUT + 1.25e-6
Z_TOP = 1.0e-6
D_ABS = 0.45e-6  # absorber thickness (lateral and top)
F0 = c / 1.55e-6
FMIN, FMAX = 0.90 * F0, 1.10 * F0
T_RUN = 1.0e-12

# probes / sources in the first quadrant, in and near the strip
SRC = [
    ((R * np.cos(0.37), R * np.sin(0.37), 0.03e-6), (np.cos(0.37), np.sin(0.37), 0.1)),
    ((R * np.cos(1.13), R * np.sin(1.13), 0.06e-6), (-np.sin(1.13), np.cos(1.13), 0.2)),
]
PRB = [
    (R * np.cos(p), R * np.sin(p), z)
    for p, z in [(0.21, 0.02e-6), (0.66, 0.05e-6), (0.97, 0.01e-6), (1.31, 0.07e-6)]
]

# symmetry planes x = 0, y = 0 of the quarter domain: equal types select the
# even azimuthal orders m (cos/sin partners are degenerate), mixed the odd m
SYM = {"even": ("pmc", "pmc"), "odd": ("pmc", "pec")}

ARMS = {
    "prism": [20, 15, 12, 9, 7, 5],  # h_ring [nm] in the plane
    "cart": [40, 30, 25, 20, 15, 12, 10],  # h [nm]
}


def pulse():
    from wakis.sources import GaussianSource

    return GaussianSource(F0, 0.12 * F0)


def z_levels(dz_core, dz_max=0.1e-6, growth=1.25):
    n_core = max(1, int(np.ceil(T / 2 / dz_core)))
    z = list(np.linspace(0, T / 2, n_core + 1))
    dz = T / 2 / n_core
    while z[-1] < Z_TOP - 1e-12:
        dz = min(dz * growth, dz_max)
        z.append(min(z[-1] + dz, Z_TOP))
    if Z_TOP - z[-2] < 0.3 * dz:  # avoid a sliver layer
        z.pop(-2)
    return np.array(z)


def quarter_mesh(full, size, radii):
    """
    Quarter (x >= 0, y >= 0) of a concentric-ring mesh with conforming
    symmetry edges: nodes closer than 0.35 h(r) to an axis are dropped,
    nodes on both axes (at every ring radius of the full mesh and at
    spacing h(r) in between) are added, and the convex quarter is
    re-triangulated (Delaunay) and ODT-smoothed with the boundary fixed.
    """
    from scipy.spatial import Delaunay

    from wakis.unstructured import SimplexMesh
    from wakis.unstructured.smoothing import odt_smooth, radial_region

    X = full.nodes
    r = np.hypot(*X.T)
    hr = np.array([size(q) for q in r])
    keep = (X[:, 0] > 0.35 * hr) & (X[:, 1] > 0.35 * hr)
    # axis stations: the ring radii of the full mesh (>= 8 nodes on a
    # radius) plus a fill at spacing h(r)
    rr, cnt = np.unique(np.round(r, 12), return_counts=True)
    st = list(rr[cnt >= 8]) + [0.0]
    q = 0.0
    while q < radii[-1]:
        q += size(q)
        st.append(min(q, radii[-1]))
    st = np.unique(np.round(np.array(st), 12))
    st = st[st <= radii[-1] * (1 + 1e-12)]
    # drop fill stations too close to a ring station
    ring = rr[cnt >= 8]
    ok = [x for x in st if x == 0 or np.min(np.abs(ring - x)) < 1e-15 or np.min(np.abs(ring - x)) > 0.4 * size(x)]
    st = np.array(ok)
    ax = np.r_[np.c_[st, 0 * st], np.c_[0 * st[1:], st[1:]]]
    nodes = np.r_[X[keep], ax]
    reg = radial_region(radii)
    tri = Delaunay(nodes).simplices
    P = nodes[tri]
    u, v = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    tri = tri[area > 1e-10 * area.max()]
    m = SimplexMesh(nodes, tri, reg(nodes[tri].mean(1)))
    return odt_smooth(m, reg, 30)


def prism_solver(h_nm, use_gpu=False, sym="even"):
    from wakis.unstructured import PrismFIT3D, PrismMesh, graded_disk_mesh
    from wakis.unstructured.smoothing import odt_smooth, radial_region

    h = h_nm * 1e-9
    band = 0.15e-6

    def size(r):
        d = max(R_IN - band - r, r - R_OUT - band, 0.0)
        return min(3 * h, h * 1.15 ** (d / h))

    radii = [R_IN, R_OUT, A_DOM]
    m2 = quarter_mesh(graded_disk_mesh(radii, size), size, radii)
    z = z_levels(h)
    sx, sy = SYM[sym]
    tol = 1e-3 * h

    def bc_xy(mid):
        return np.where(
            np.abs(mid[:, 1]) < tol, sx, np.where(np.abs(mid[:, 0]) < tol, sy, "abc")
        )

    def tags(tri, zm):
        return np.where((tri == 1) & (zm < T / 2), 1, 0)

    pm = PrismMesh(m2, z, tags=tags, merge_tol=0.15)
    n_pml = int(np.sum(pm.z_mid > Z_TOP - D_ABS))
    v = c / np.sqrt(EPS_OX)
    return PrismFIT3D(
        pm, {0: [EPS_OX, 1.0], 1: [EPS_SI, 1.0]}, bc_xy=bc_xy,
        bc_z=("pmc", "pec"), use_gpu=use_gpu,
        absorber_xy=dict(r0=A_DOM - D_ABS, r1=A_DOM, s_max=3 * v / D_ABS, order=2),
        pml_z=dict(n_low=0, n_high=n_pml, s_max=4 * v / D_ABS, order=3),
    )  # fmt: skip


def cart_solver(h_nm, sym, use_gpu=False):
    import pyvista as pv

    import wakis

    h = h_nm * 1e-9
    n = int(np.ceil(A_DOM / h))  # the CPML cells are the last n_pml cells
    x = np.arange(n + 1) * h
    nz = int(np.ceil(Z_TOP / h))
    z = np.arange(nz + 1) * h
    ring = (
        pv.Disc(inner=R_IN, outer=R_OUT, c_res=1440, r_res=2)
        .extrude((0, 0, T), capping=True)
        .translate((0, 0, -T / 2))
        .triangulate()
        .clean()
    )
    grid = wakis.GridFIT3D(
        x=x, y=x.copy(), z=z,
        stl_solids={"ring": ring}, stl_materials={"ring": [EPS_SI, 1.0, 0.0]},
        conformal="all", verbose=0,
    )  # fmt: skip
    n_pml = max(8, int(round(D_ABS / h)))
    solver = wakis.SolverFIT3D(
        grid, bc_low=[*SYM[sym], "pmc"], bc_high=["cpml", "cpml", "cpml"],
        bg=[EPS_OX, 1.0, 0.0], n_pml=n_pml, use_stl=True, use_gpu=use_gpu, verbose=0,
    )  # fmt: skip
    return solver, grid


def extract(sig, dt, t0):
    from wakis.resonances import harminv

    step = max(1, int(1 / (8 * FMAX * dt)))
    found = []
    for j in range(sig.shape[1]):
        r = harminv(sig[::step, j], step * dt, FMIN, FMAX, t0=t0)
        found += [
            (float(F.real), float(F.imag), float(abs(a)))
            for F, a in zip(r.complex_frequency, r.amplitude)
        ]
    return found


def run_prism(h_nm, sym, use_gpu):
    t0 = time.time()
    s = prism_solver(h_nm, use_gpu, sym)
    t_setup = time.time() - t0
    src = pulse()
    for p, d in SRC:
        s.add_point_dipole(p, d, src)
    probes = [s.add_probe(PRB, comp) for comp in "xy"]
    n = int(T_RUN / s.dt)
    t1 = time.time()
    s.run(n)
    if use_gpu:
        import cupy as cp

        cp.cuda.Device().synchronize()
    t_run = time.time() - t1
    start = int(src.end_time / s.dt) + 1
    sig = np.hstack([s.probe_signal(p) for p in probes])[start:]
    found = extract(sig, s.dt, (start + 1) * s.dt)
    return dict(
        found=found, unknowns=int(s.n_unknowns), cells=int(s.mesh.n_cells), sym=sym,
        cells2d=int(s.mesh.c2.n_cells), layers=int(s.mesh.K), dt=s.dt, steps=n,
        setup_s=t_setup, run_s=t_run, merged=int(s.mesh.c2.n_removed),
    )  # fmt: skip


def run_cart(h_nm, sym, use_gpu):
    from wakis.sources import CurrentSource

    t0 = time.time()
    solver, grid = cart_solver(h_nm, sym, use_gpu)
    t_setup = time.time() - t0
    src = pulse()
    sources = []
    for p, d in SRC:
        d = np.asarray(d) / np.linalg.norm(d)
        sources += [
            CurrentSource(src, component=k, center=tuple(p), amplitude=float(d[i]))
            for i, k in enumerate("xyz") if abs(d[i]) > 0
        ]  # fmt: skip
    xs, ys, zs = (np.asarray(v) for v in (grid.x, grid.y, grid.z))
    idx = []
    for comp in "xy":
        a = "xyz".index(comp)
        co = [xs[:-1], ys[:-1], zs[:-1]]
        co[a] = 0.5 * ((xs, ys, zs)[a][:-1] + (xs, ys, zs)[a][1:])
        for p in PRB:
            idx.append(tuple(int(np.argmin(np.abs(q - v))) for q, v in zip(co, p)) + (comp,))
    samples = []

    def cb(s, t):
        samples.append([float(s.E[i]) for i in idx])

    n = int(T_RUN / solver.dt)
    t1 = time.time()
    solver.emsolve(n, source=sources, callback=cb)
    t_run = time.time() - t1
    start = int(src.end_time / solver.dt) + 1
    sig = np.asarray(samples)[start:]
    found = extract(sig, solver.dt, (start + 1) * solver.dt)
    Nx, Ny, Nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    return dict(
        found=found, unknowns=int(6 * Nx * Ny * Nz), cells=int(Nx * Ny * Nz),
        dt=solver.dt, steps=n, setup_s=t_setup, run_s=t_run, sym=sym,
    )  # fmt: skip


def tasks():
    out = [("prism", h, s) for h in ARMS["prism"] for s in ("even", "odd")]
    out += [("cart", h, s) for h in ARMS["cart"] for s in ("even", "odd")]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("--task", type=int)
    ap.add_argument("--arm")
    ap.add_argument("--h", type=float)
    ap.add_argument("--sym", default="even")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for i, t in enumerate(tasks()):
            print(i, *t)
        return
    arm, h, sym = (args.arm, args.h, args.sym) if args.arm else tasks()[args.task]
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    out = run_prism(h, sym, args.gpu) if arm == "prism" else run_cart(h, sym, args.gpu)
    out.update(arm=arm, h_nm=h, gpu=args.gpu, wall_s=time.time() - t0,
               host=os.uname().nodename)  # fmt: skip
    tag = f"{arm}_{h:g}_{sym}" + ("_gpu" if args.gpu else "")
    with open(os.path.join(args.outdir, tag + ".json"), "w") as fh:
        json.dump(out, fh, indent=1)
    fs = sorted({round(f / 1e12, 3) for f, g, _ in out["found"] if g > 0 and f / (2 * g) > 500})
    print(tag, "unknowns", out["unknowns"], "steps", out["steps"],
          "run %.1f s" % out["run_s"], "f [THz]", fs)  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
