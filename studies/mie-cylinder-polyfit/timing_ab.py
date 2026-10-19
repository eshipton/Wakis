"""
Same-machine A/B timing (one uncontended Batch machine): ns per
(unknown x step) of

* UnstructuredFIT2D vs PolygonFIT2D on the same graded triangle mesh
  (res 28, plane wave with the tabulated source, ABC), TM and TE;
* PolygonFIT2D on the hybrid-disk and Voronoi meshes (res 28);
* the bare square-grid update: NumPy stencil (array slices) vs the sparse
  incidence update, 400 x 400 squares.

    python timing_ab.py run --outdir DIR     (Batch; --shard/--nshards ignored)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mie-cylinder-mesh"))
sys.path.insert(0, str(HERE.parent / "mie-cylinder-graded"))


def _time(solver, n=2000, warm=200, reps=3):
    solver.run(warm)
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        solver.run(n)
        best = min(best, (time.perf_counter() - t) / n)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--outdir", default=str(HERE / "data"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import graded_study as GS
    import polyfit_study as P

    from wakis.unstructured import UnstructuredFIT2D, graded_disk_mesh
    from wakis.unstructured.polymesh import PolyMesh
    from wakis.unstructured.polysolver2d import PolygonFIT2D

    res = 28
    h = P.S.LAM_MIN_DIEL / res
    tri = graded_disk_mesh([P.A, 2.7e-6], GS.size_field(h, True))
    out = {}
    for pol in ("TM", "TE"):
        a1 = UnstructuredFIT2D(tri, {0: [4.0, 1.0]}, polarization=pol, bc="abc", cfl=0.95)
        a1.add_plane_wave(P.S.pulse())
        n1 = a1.free.size + tri.n_edges
        out[f"tri-mesh UnstructuredFIT2D {pol}"] = _time(a1) / n1 * 1e9
        a2 = PolygonFIT2D(PolyMesh.from_simplex(tri), {0: [4.0, 1.0]}, polarization=pol, bc="abc", cfl=0.95)
        a2.add_plane_wave(P.S.pulse())
        out[f"tri-mesh PolygonFIT2D {pol}"] = _time(a2) / a2.n_unknowns * 1e9
        for kind in ("hybrid-disk", "hybrid", "voronoi"):
            mesh, info, kw, _ = P.build(kind, "std", res)
            s = PolygonFIT2D(mesh, {0: [4.0, 1.0]}, polarization=pol, cfl=0.95, **kw)
            s.add_plane_wave(P.S.pulse())
            out[f"{kind} PolygonFIT2D {pol}"] = _time(s) / s.n_unknowns * 1e9
    # bare square updates
    from wakis.unstructured.polymesh import PolyMesh as PM

    n = 400
    rng = np.random.default_rng(0)
    u = rng.random((n, n))
    vx, vy = rng.random((n, n - 1)), rng.random((n - 1, n))
    sx, sy = rng.random((n, n - 1)), rng.random((n - 1, n))
    dtm = 0.1 / (rng.random((n, n)) + 1)
    dt = 0.1

    def stencil():
        nonlocal u, vx, vy
        vx -= dt * (u[:, 1:] - u[:, :-1])
        vy -= dt * (u[1:, :] - u[:-1, :])
        fx, fy = sx * vx, sy * vy
        r = np.zeros_like(u)
        r[:, 1:] += fx
        r[:, :-1] -= fx
        r[1:, :] += fy
        r[:-1, :] -= fy
        u += dtm * r

    pm = PM.rectilinear(np.arange(n), np.arange(n))
    G = pm.G.tocsr()
    GT = G.T.tocsr()
    uu, v = rng.random(pm.n_nodes), rng.random(pm.n_edges)
    st, dm = rng.random(pm.n_edges), dtm.ravel()

    def sparse():
        nonlocal uu, v
        v -= dt * (G @ uu)
        uu += dm * (GT @ (st * v))

    for name, f in (("squares stencil", stencil), ("squares sparse", sparse)):
        f()
        best = np.inf
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(200):
                f()
            best = min(best, (time.perf_counter() - t) / 200)
        out[name] = best / (3 * n * n) * 1e9
    try:
        with open("/proc/cpuinfo") as fh:
            out["cpu"] = next(x.split(":")[1].strip() for x in fh if "model name" in x)
    except OSError:
        pass
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    (Path(a.outdir) / "timing_ab.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
