"""
Regression data of the conformal setup: metric arrays (open edge lengths,
face areas, volume fractions, interface normals), material arrays, the
merged-face operators, the tensor couplings and the stable time step, for a
few geometries. Save them with one version of the code and compare with
another (e.g. before and after a performance change)::

    python benchmarks/conformal_regression.py --save ref.npz --n 24
    python benchmarks/conformal_regression.py --compare ref.npz --n 24

Geometries: PEC-background sphere cavity ('all'), dielectric sphere
('tensor'), rotated PEC box ('pec'), tilted dielectric cylinder with a PEC
ball and a thin sheet ('all'). The comparison reports the largest relative
difference per array (max |a - b| / max |b|); the stable time step is an
iterative estimate (tolerance 1e-3) and is compared separately.
"""

import argparse

import numpy as np
import pyvista as pv

import wakis
from wakis.conformal import rectangle_sheet


def cases(n):
    lo, hi = -0.1, 0.1
    h = (hi - lo) / n
    res = 60
    c = (0.0071, -0.0043, 0.0029)
    box = pv.Box(bounds=(-0.05, 0.06, -0.04, 0.05, -0.06, 0.03)).triangulate()
    box = box.rotate_z(17.0, inplace=False).rotate_x(11.0, inplace=False)
    zs = lo + h * round((-0.05 - lo) / h)
    return {
        "sphere_pec": (
            dict(
                stl_solids={"cav": pv.Sphere(radius=0.083, center=c, theta_resolution=res, phi_resolution=res)},
                stl_materials={"cav": "vacuum"},
                conformal="all",
            ),
            "pec",
        ),
        "sphere_tensor": (
            dict(
                stl_solids={"core": pv.Sphere(radius=0.06, center=c, theta_resolution=res, phi_resolution=res)},
                stl_materials={"core": [4.0, 1.0, 0.0]},
                conformal="tensor",
            ),
            [1.0, 1.0, 0.0],
        ),
        "box": (
            dict(stl_solids={"box": box}, stl_materials={"box": "pec"}, conformal="pec"),
            [1.0, 1.0, 0.0],
        ),
        "mixed": (
            dict(
                stl_solids={
                    "ball": pv.Sphere(radius=0.043, center=(0.021, -0.013, 0.008), theta_resolution=res, phi_resolution=res),
                    "rod": pv.Cylinder(center=(-0.05, 0.03, 0), direction=(1, 1, 0.3), radius=0.02, height=0.07, resolution=res).triangulate(),
                },
                stl_materials={"ball": "pec", "rod": [3.0, 1.0, 0.0]},
                sheets={"plate": rectangle_sheet((0.0, 0.0, zs), (0.083, 0.061, 0.0))},
                conformal="all",
            ),
            [1.0, 1.0, 0.0],
        ),
    }  # fmt: skip


def arrays(n):
    """All regression arrays of all cases, as a flat dict name -> array."""
    out = {}
    for name, (kw, bg) in cases(n).items():
        grid = wakis.GridFIT3D(
            -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, n, n, n, verbose=0, **kw
        )
        solver = wakis.SolverFIT3D(
            grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg=bg, use_stl=True, verbose=0
        )  # fmt: skip
        geo = next(iter(grid._conformal_geometry.values()))
        for c in range(3):
            out[f"{name}/pec_length{c}"] = geo.pec_length[c]
            out[f"{name}/pec_area{c}"] = geo.pec_area[c]
            for key in geo.fractions:
                out[f"{name}/fraction_{key}{c}"] = geo.fractions[key][c]
                out[f"{name}/normal_{key}{c}"] = geo.normals[key][c]
        for i, (plane, p, q) in enumerate(geo.sheet_outlines):
            out[f"{name}/sheet{i}"] = np.concatenate(
                [np.ravel(plane), p.ravel(), q.ravel()]
            )
        out[f"{name}/L"] = solver.L.toarray()
        out[f"{name}/iA"] = solver.iA.toarray()
        out[f"{name}/ieps"] = np.asarray(solver.ieps.toarray())
        out[f"{name}/sigma"] = np.asarray(solver.sigma.toarray())
        out[f"{name}/ieps_factor"] = solver.conformal_ieps_factor
        for c, op in enumerate(solver.conformal_face_ops):
            op = op.tocoo()
            order = np.lexsort((op.col, op.row))
            out[f"{name}/face_op{c}"] = np.stack(
                [op.row[order], op.col[order], op.data[order]]
            )
        out[f"{name}/unstable"] = np.array([solver.conformal_unstable_faces])
        t = solver.tensor_operator
        if t is not None:
            out[f"{name}/tensor_pairs"] = np.stack(
                [t.pair_a, t.pair_b, np.real(t.pair_coef)]
            )
            Kp = t.Kp_host.tocoo()
            order = np.lexsort((Kp.col, Kp.row))
            out[f"{name}/tensor_Kp"] = np.stack(
                [Kp.row[order], Kp.col[order], np.real(Kp.data[order])]
            )
        out[f"{name}/dt"] = np.array([solver.dt])
    return out


def compare(ref, new):
    """Largest relative difference per array; returns (report, worst metric, worst dt)."""
    lines, worst, worst_dt = [], 0.0, 0.0
    for key in sorted(set(ref) | set(new)):
        if key not in ref or key not in new:
            lines.append(f"{key}: missing in {'new' if key in ref else 'reference'}")
            worst = np.inf
            continue
        a, b = np.asarray(new[key], float), np.asarray(ref[key], float)
        if a.shape != b.shape:
            lines.append(f"{key}: shape {a.shape} != {b.shape}")
            worst = np.inf
            continue
        scale = np.max(np.abs(b)) if b.size else 0.0
        diff = np.max(np.abs(a - b)) if a.size else 0.0
        rel = diff / scale if scale > 0 else diff
        if key.endswith("/dt"):
            worst_dt = max(worst_dt, rel)
        else:
            worst = max(worst, rel)
        exact = "bitwise" if np.array_equal(a, b) else f"{rel:.2e}"
        lines.append(f"{key}: {exact}")
    return "\n".join(lines), worst, worst_dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--save")
    ap.add_argument("--compare")
    a = ap.parse_args()
    data = arrays(a.n)
    if a.save:
        np.savez_compressed(a.save, **data)
        print(f"saved {len(data)} arrays to {a.save}")
    if a.compare:
        ref = dict(np.load(a.compare))
        report, worst, worst_dt = compare(ref, data)
        print(report)
        print(f"worst metric/material difference: {worst:.3e}; dt: {worst_dt:.3e}")


if __name__ == "__main__":
    main()
