"""
CAD meshes and the tetrahedral time step of the unstructured FIT.

Parts (each writes results/<part>.json; run them on Cloud Batch):

    python cadmesh_study.py bounds   [n_procs]   # element bound tightness, mode localization
    python cadmesh_study.py improve  [n_procs]   # dt gain of improve_dt vs gmsh baselines
    python cadmesh_study.py muon_tet [n_procs]   # muon-cavity STEP: tet resonances + ringdown
    python cadmesh_study.py muon_cart H_MM [--gpu]  # Cartesian conformal-PEC reference at h [mm]
    python cadmesh_study.py figures              # figures/ from the JSON files

Geometries (all from CAD through `wakis.unstructured.cad`):

* pillbox: PEC cylinder a = L = 0.1 m (OCC primitive), analytic TM010,
  TE111 (x2), TM110 (x2).
* layered: the same pillbox with an eps = 4 layer 0 < z < 30 mm, written as
  STEP and re-imported; analytic TM_01 root (`layered_pillbox_tm0`).
* muon: tests/stl/006_muonCavity.stp (two vacuum half cells, mm), selected
  by STEP label; reference = Cartesian conformal-PEC FIT ring-down
  (`muon_cart`) and the tet convergence sequence.

Baselines for the dt gain (rule: name the denominator): the same CAD, same
f_max and cpw, gmsh Delaunay output without optimization ("none"), with
gmsh's default tetrahedral optimizer ("gmsh") and with Netgen's ("netgen");
ours = "none" followed by `improve_dt` (CAD-surface sliding). Every arm uses
the solver defaults (Voronoi Hodge, massless_tol = 1e-2, cfl 0.95).
"""

import json
import os
import sys
import time

import numpy as np
from scipy.constants import c as C0

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MUON = os.path.join(ROOT, "tests", "stl", "006_muonCavity.stp")
X01, X11, XP11 = 2.404825557695773, 3.831705970207512, 1.841183781340659

A_PB = L_PB = 0.1
T_LAY, EPS_LAY = 0.03, 4.0
F_MAX_PB = 1.5e9
F_MAX_MU = 1.0e9


def _dump(name, obj):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def _load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


# ------------------------------------------------------------------ geometries
def _pillbox(gmsh):
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L_PB, A_PB)


def _layered_step():
    import gmsh

    path = os.path.join(RES, "layered.step")
    os.makedirs(RES, exist_ok=True)
    if not os.path.exists(path):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, T_LAY, A_PB)
        gmsh.model.occ.addCylinder(0, 0, T_LAY, 0, 0, L_PB - T_LAY, A_PB)
        gmsh.model.occ.synchronize()
        tmp = path + f".{os.getpid()}.step"
        gmsh.write(tmp)
        gmsh.finalize()
        os.replace(tmp, path)  # atomic: workers never see a partial file
    return path


def pillbox_exact():
    k = [X01 / A_PB] + [np.hypot(XP11 / A_PB, np.pi / L_PB)] * 2 + [X11 / A_PB] * 2
    return np.array(k) * C0 / (2 * np.pi)


def layered_exact():
    from wakis.unstructured.analytic import layered_pillbox_tm0

    return layered_pillbox_tm0(A_PB, L_PB, T_LAY, EPS_LAY)[:1] * C0 / (2 * np.pi)


def make_mesh(case, cpw, optimize="none", improve=None, curvature=0):
    from wakis.unstructured.cad import mesh_cad

    if case == "pillbox":
        return mesh_cad(_pillbox, {"vac": (0, 0, L_PB / 2)}, {"vac": [1, 1]},
                        f_max=F_MAX_PB, cpw=cpw, optimize=optimize, improve=improve,
                        curvature_points=curvature)  # fmt: skip
    if case == "layered":
        return mesh_cad(_layered_step(),
                        {"diel": (0, 0, T_LAY / 2), "vac": (0, 0, (L_PB + T_LAY) / 2)},
                        {"diel": [EPS_LAY, 1], "vac": [1, 1]}, f_max=F_MAX_PB,
                        cpw=cpw, optimize=optimize, improve=improve,
                        curvature_points=curvature)  # fmt: skip
    if case == "muon":
        return mesh_cad(MUON, {"vacuum": r"Vacuum$"}, {"vacuum": [1, 1]},
                        f_max=F_MAX_MU, cpw=cpw, scale=1e-3, optimize=optimize,
                        improve=improve, curvature_points=curvature,
                        max_tets=3e6)  # fmt: skip
    raise ValueError(case)


F_TARGET = {"pillbox": 0.7e9, "layered": 0.5e9, "muon": 0.4e9}
N_MODES = {"pillbox": 5, "layered": 1, "muon": 4}


def step_time(s, n=200):
    """Wall time per leapfrog step [s] (median of 3 blocks of n/3 steps)."""
    s.reset()
    s.u[:] = np.random.default_rng(0).standard_normal(s.u.size) * 1e-6
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        s.run(n // 3)
        ts.append((time.perf_counter() - t) / (n // 3))
    s.reset()
    return float(np.median(ts))


# ------------------------------------------------------------------ part: bounds
def _bounds_task(args):
    case, cpw, hodge = args
    from wakis.unstructured import UnstructuredFIT3D, ball_mesh
    from wakis.unstructured.quality import (
        element_stability, stability_report, tet_quality, top_mode_localization)  # fmt: skip

    if case.startswith("ball"):
        h = float(case.split("_")[1])
        m, mats = ball_mesh([0.5, 1.0], h), {0: [4.0, 1.0], 1: [1.0, 1.0]}
    else:
        m, info = make_mesh(case, cpw)
        mats = info.materials
    s = UnstructuredFIT3D(m, mats, hodge=hodge)
    st = element_stability(s)
    loc = top_mode_localization(s, st)
    rep = stability_report(s, st)
    q = tet_quality(m)
    out = dict(case=case, cpw=cpw, hodge=hodge, **rep,
               lam_max=st["lam_max"], lam_bound=st["lam_bound"],
               lam_lower=st["lam_lower"], share_worst=loc["share_worst"],
               rank_of_peak=loc["rank_of_peak"])  # fmt: skip
    if hodge == "voronoi":
        out["lam_gershgorin"] = st["lam_gershgorin"]
    # per-element data for the histogram figure (normalized element steps)
    if case in ("pillbox", "muon") and cpw == 20:
        out["dt_T_over_dt"] = (st["dt_T"] / st["dt_max"]).tolist()
        out["radius_ratio"] = q["radius_ratio"].tolist()
        out["energy"] = loc["energy"].tolist()
    return out


def part_bounds(n_procs):
    from multiprocessing import Pool

    _layered_step()  # write the STEP once, before the workers start
    tasks = [(c, cpw, hd) for c in ("pillbox", "layered", "muon") for cpw in (10, 20)
             for hd in ("voronoi", "whitney")]  # fmt: skip
    tasks += [(f"ball_{h}", 0, hd) for h in (0.25, 0.15) for hd in ("voronoi", "whitney")]
    with Pool(n_procs) as p:
        res = p.map(_bounds_task, tasks)
    _dump("bounds", res)
    for r in res:
        print(r["case"], r["cpw"], r["hodge"], "tight %.3f lower %.4f share1%% %.3f"
              % (r["tightness"], np.sqrt(r["lam_lower"] / r["lam_max"]),
                 r["share_worst"]["0.01"] if "0.01" in r["share_worst"]
                 else r["share_worst"][0.01]))  # fmt: skip


# ------------------------------------------------------------------ part: improve
def _improve_task(args):
    case, cpw, variant = args
    from wakis.unstructured import UnstructuredFIT3D
    from wakis.unstructured.quality import element_stability

    t0 = time.perf_counter()
    if variant == "ours":
        m, info = make_mesh(case, cpw, "none", improve=dict(n_sweeps=10))
    else:
        m, info = make_mesh(case, cpw, variant)
    t_mesh = time.perf_counter() - t0
    d = m.diagnostics()
    out = dict(case=case, cpw=cpw, variant=variant, n_tets=m.n_cells, t_mesh=t_mesh,
               negative_dual_area=d["negative_dual_area"],
               negative_dual_length=d["negative_dual_length"],
               volume=float(m.cell_volume.sum()), h_mean=float(m.edge_length.mean()))  # fmt: skip
    try:
        s = UnstructuredFIT3D(m, info.materials)
    except ValueError as err:  # non-Delaunay: voronoi refused
        out["error"] = str(err)
        return out
    st = element_stability(s)
    out.update(dt_max=s.dt_max, courant_h=s.dt_max * C0 / m.edge_length.mean(),
               n_massless=int(s.massless.size), n_free=int(s.free.size),
               n_negative_faces=st["n_negative_faces"], tightness=st["tightness"],
               t_step=step_time(s))  # fmt: skip
    f, _ = s.eigenmodes(N_MODES[case], f_target=F_TARGET[case])
    out["f"] = f.tolist()
    if case == "pillbox":
        out["err"] = (f / pillbox_exact() - 1).tolist()
    elif case == "layered":
        out["err"] = (f[:1] / layered_exact() - 1).tolist()
    if info.get("improve_history"):
        out["history"] = info.improve_history
    return out


def part_improve(n_procs):
    from multiprocessing import Pool

    _layered_step()  # write the STEP once, before the workers start
    tasks = [(c, cpw, v) for c in ("pillbox", "layered", "muon") for cpw in (10, 15, 20)
             for v in ("none", "gmsh", "netgen", "ours")]  # fmt: skip
    with Pool(n_procs) as p:
        res = p.map(_improve_task, tasks)
    _dump("improve", res)
    for r in res:
        print(r["case"], r["cpw"], r["variant"], r["n_tets"],
              "c dt/h %.3f" % r.get("courant_h", np.nan),
              "massless", r.get("n_massless"), "neg", r.get("n_negative_faces"),
              "err", np.round(r.get("err", [np.nan]), 5)[:3], r.get("error", ""))  # fmt: skip


# ------------------------------------------------------------------ part: muon tet
def _muon_eig_task(args):
    cpw, variant, hodge = args
    from wakis.unstructured import UnstructuredFIT3D

    imp = dict(n_sweeps=10) if variant == "ours" else None
    m, info = make_mesh("muon", cpw, "none", improve=imp, curvature=48)
    s = UnstructuredFIT3D(m, info.materials, hodge=hodge)
    t = time.perf_counter()
    f, _ = s.eigenmodes(4, f_target=F_TARGET["muon"])
    return dict(cpw=cpw, variant=variant, hodge=hodge, n_tets=m.n_cells,
                h_mean=float(m.edge_length.mean()), volume=float(m.cell_volume.sum()),
                f=f.tolist(), dt_max=s.dt_max, t_eig=time.perf_counter() - t)  # fmt: skip


def _gauss_pulse(f0, bw):
    tau = 1 / (2 * np.pi * bw)
    t0 = 5 * tau

    def amp(t):
        return np.exp(-(((t - t0) / tau) ** 2) / 2) * np.sin(2 * np.pi * f0 * (t - t0))

    return amp, 2 * t0


def _muon_td_task(args):
    """Ring-down on the tet mesh (baseline vs improved): frequency and wall time."""
    cpw, variant = args
    from wakis.resonances import harminv
    from wakis.unstructured import UnstructuredFIT3D

    imp = dict(n_sweeps=10) if variant == "ours" else None
    m, info = make_mesh("muon", cpw, "none", improve=imp, curvature=48)
    s = UnstructuredFIT3D(m, info.materials)
    amp, t_end = _gauss_pulse(0.9e9, 0.5e9)
    s.add_point_dipole([0.031, 0.022, 0.011], [0.2, 0.3, 1.0], amp)
    probe = s.add_probe([[-0.043, 0.017, -0.013]], 2)
    t_rec = 60e-9
    n = int(np.ceil((t_end + t_rec) / s.dt))
    t = time.perf_counter()
    s.run(n)
    wall = time.perf_counter() - t
    x = s.probe_signal(probe)[:, 0]
    i0 = int(np.ceil(t_end / s.dt))
    stride = max(1, int(1 / (8 * 1.5e9 * s.dt)))
    sig = x[i0::stride]
    dts = stride * s.dt
    res = harminv(sig, dts, 0.5e9, 1.5e9, t0=(i0 + 1) * s.dt)
    f = np.asarray(res.frequency)
    # remove the leapfrog time dispersion: spatial (dt -> 0) frequency
    f_space = 2 / s.dt * np.sin(np.pi * f * s.dt) / (2 * np.pi)
    return dict(cpw=cpw, variant=variant, n_tets=m.n_cells, dt=s.dt, n_steps=n,
                h_mean=float(m.edge_length.mean()),
                wall=wall, wall_per_ns=wall / ((t_end + t_rec) * 1e9),
                f=f.tolist(), f_space=f_space.tolist(),
                amp=np.abs(np.asarray(res.amplitude)).tolist())  # fmt: skip


def part_muon_tet(n_procs):
    from multiprocessing import Pool

    eig = [(cpw, v, hd) for cpw in (10, 15, 20, 30, 40) for v in ("none", "ours")
           for hd in ("voronoi", "whitney")]  # fmt: skip
    eig = [t for t in eig if not (t[2] == "whitney" and t[1] == "ours")]
    td = [(cpw, v) for cpw in (15, 20) for v in ("none", "ours")]
    with Pool(n_procs) as p:
        r_eig = p.map_async(_muon_eig_task, eig)
        r_td = p.map_async(_muon_td_task, td)
        res = dict(eig=r_eig.get(), td=r_td.get())
    _dump("muon_tet", res)
    for r in res["eig"]:
        print("eig", r["cpw"], r["variant"], r["hodge"], r["n_tets"], np.round(np.array(r["f"]) / 1e9, 5))
    for r in res["td"]:
        print("td", r["cpw"], r["variant"], r["n_tets"], "dt %.3e wall/ns %.2f" % (r["dt"], r["wall_per_ns"]),
              np.round(np.array(r["f_space"]) / 1e9, 5))  # fmt: skip


# ------------------------------------------------------------------ part: Cartesian reference
def _vacuum_stl(path, margin):
    """STL of the muon vacuum (two half cells fused); bounds padded by margin [mm]."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        ents = occ.importShapes(MUON)
        occ.synchronize()
        vac = [(3, t) for d, t in ents if d == 3
               and gmsh.model.getEntityName(3, t).endswith("Vacuum")]  # fmt: skip
        others = [(3, t) for d, t in ents if d == 3 and (3, t) not in vac]
        occ.remove(others, recursive=True)
        occ.fuse(vac[:1], vac[1:])
        occ.synchronize()
        bb = np.array(gmsh.model.getBoundingBox(-1, -1))
        lo, hi = bb[:3] - margin, bb[3:] + margin
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 72)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 6.0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.model.mesh.generate(2)
        gmsh.write(path)
    finally:
        gmsh.finalize()
    return lo * 1e-3, hi * 1e-3


def part_muon_cart(h_mm, use_gpu=False):
    import pyvista as pv

    import wakis
    from wakis.resonances import ResonanceMonitor
    from wakis.sources import CurrentSource, GaussianSource

    h = h_mm * 1e-3
    stl = os.path.join(RES, "muon_vacuum.stl")
    os.makedirs(RES, exist_ok=True)
    lo, hi = _vacuum_stl(stl, 2 * h_mm)
    surf = pv.read(stl).scale(1e-3)
    n = np.round((hi - lo) / h).astype(int)
    grid = wakis.GridFIT3D(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], *n,
                           stl_solids={"cavity": surf}, stl_materials={"cavity": "vacuum"},
                           conformal="pec", verbose=0)  # fmt: skip
    solver = wakis.SolverFIT3D(grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3,
                               bg="pec", use_stl=True, use_gpu=use_gpu, verbose=0)  # fmt: skip
    pulse = GaussianSource(frequency=0.9e9, fwidth=0.5e9)
    src = CurrentSource(pulse, component="z", center=(0.031, 0.022, 0.011))
    probe = ResonanceMonitor("Ez", center=(-0.043, 0.017, -0.013), t_start=pulse.end_time)
    t_rec = 60e-9
    nt = int((pulse.end_time + t_rec) / solver.dt)
    t = time.perf_counter()
    solver.emsolve(nt, source=src, monitors=[probe])
    wall = time.perf_counter() - t
    res = probe.harminv(0.5e9, 1.5e9)
    f = np.asarray(res.frequency)
    f_space = 2 / solver.dt * np.sin(np.pi * f * solver.dt) / (2 * np.pi)
    out = dict(h_mm=h_mm, n_cells=int(np.prod(n)), dt=solver.dt, n_steps=nt, wall=wall,
               f=f.tolist(), f_space=f_space.tolist(),
               amp=np.abs(np.asarray(res.amplitude)).tolist())  # fmt: skip
    _dump(f"muon_cart_{h_mm:g}mm", out)
    print(out)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    part = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if part == "bounds":
        part_bounds(int(arg or os.cpu_count()))
    elif part == "improve":
        part_improve(int(arg or os.cpu_count()))
    elif part == "muon_tet":
        part_muon_tet(int(arg or os.cpu_count()))
    elif part == "muon_cart":
        part_muon_cart(float(arg), use_gpu="--gpu" in sys.argv)
    elif part == "figures":
        from plot_cadmesh import main

        main()
    else:
        raise SystemExit(__doc__)
