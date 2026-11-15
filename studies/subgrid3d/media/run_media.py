# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
3D subgridding study for dispersive (Drude/Lorentz ADE) and conformal
(subpixel Kottke) media inside patches (`wakis.subgrid3d.media`).

Studies
-------
(i) Dispersive cube (Lorentz and lossless Drude) inside a 2:1 patch vs the
    named baseline `SubgridFIT3D` without patch at the same fine step h
    (which equals `SolverFIT3D` per subgrid3d.md §4):
    - `localized`: small dispersive cube (5.6% of cavity volume) where the
      2:1 patch reduces state unknowns and work per unit physical time.
    - `exterior_limited` (honest case where the patch does not help):
      a small vacuum-dominated cavity with a large 2:1 patch (70% of box
      volume) where exterior coarse dispersion dominates the resonance error
      while the global time step is still pinned to the fine patch limit.
(ii) Staircase vs subpixel Kottke permittivity averaging for a dielectric
     sphere (eps_r = 4.0) inside a 2:1 patch, reporting `n_sub` and comparing
     the default dual-volume moment normal against the analytic radial normal.

Usage
-----
    python studies/subgrid3d/media/run_media.py [--quick] --out DIR
"""

import argparse
import json
import os
import platform
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from wakis.dispersive import DispersiveMaterial, Drude, Lorentz  # noqa: E402
from wakis.subgrid3d import (  # noqa: E402
    Box,
    SubgridFIT3D,
    SubgridMesh3D,
    dispersive_eigenfrequencies,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

BOX_DISP = (1.0, 0.8, 0.6)
CASES_DISP = {
    "localized": dict(
        cube=((0.4, 0.6), (0.2, 0.6), (0.2, 0.4)),
        patch=((0.2, 0.8), (0.2, 0.6), (0.2, 0.4)),
        description="Localized dispersive cube (3.3% box volume) inside a 2:1 patch",
    ),
    "exterior_limited": dict(
        cube=((0.2, 0.8), (0.2, 0.6), (0.2, 0.4)),
        patch=((0.2, 0.8), (0.2, 0.6), (0.2, 0.4)),
        description=(
            "Exterior-dominated dispersive block (20% box volume, patch bound == block bound): "
            "hanging-edge interface sits directly on the dielectric jump while global dt is fine-limited"
        ),
    ),
}

BOX_SPHERE = (1.0, 0.85, 0.75)
SPHERE_CENTER = (0.48, 0.43, 0.37)
SPHERE_RADIUS = 0.22
SPHERE_EPS = 4.0
SPHERE_PATCH = ((0.22, 0.74), (0.17, 0.69), (0.12, 0.62))


def _make_dispersive_material(model):
    if model == "lorentz":
        return DispersiveMaterial(
            eps_inf=1.5,
            poles=[Lorentz(delta_eps=2.5, omega_0=2.0 * np.pi * 340e6, gamma=0.0)],
        )
    if model == "drude":
        return DispersiveMaterial(
            eps_inf=1.5,
            poles=[Drude(omega_p=2.0 * np.pi * 240e6, gamma=0.0)],
        )
    raise ValueError(f"Unknown model {model!r}")


def _axes_from_nx(box, nx):
    Lx, Ly, Lz = box
    ny = max(3, int(round(nx * Ly / Lx)))
    nz = max(3, int(round(nx * Lz / Lx)))
    return (
        np.linspace(0.0, Lx, nx + 1),
        np.linspace(0.0, Ly, ny + 1),
        np.linspace(0.0, Lz, nz + 1),
    )


def _snap_box_to_axes(bounds, axes):
    """Snap box bounds to the nearest interior coarse node coordinates."""
    snapped = []
    for (lo, hi), ax in zip(bounds, axes):
        i0 = max(1, min(len(ax) - 2, int(np.argmin(np.abs(ax - lo)))))
        i1 = max(i0 + 1, min(len(ax) - 1, int(np.argmin(np.abs(ax - hi)))))
        snapped.append((float(ax[i0]), float(ax[i1])))
    return tuple(snapped)


def _measure_step_wall_s(solver, n_steps=15):
    rng = np.random.default_rng(0)
    solver.e[:] = rng.standard_normal(solver.n_free) * 1e-3
    for _ in range(3):
        solver.one_step()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        solver.one_step()
    return float((time.perf_counter() - t0) / n_steps)


def _richardson_ref(h_vals, f_vals, order=2.0):
    idx = np.argsort(h_vals)
    h0, h1 = float(h_vals[idx[0]]), float(h_vals[idx[1]])
    f0, f1 = float(f_vals[idx[0]]), float(f_vals[idx[1]])
    r_p = (h1 / h0) ** order
    return float((r_p * f0 - f1) / (r_p - 1.0))


def run_dispersive_study(quick=False):
    nx_list = [5, 10] if quick else [5, 10, 15, 20]
    rows = []

    for case_name, cfg in CASES_DISP.items():
        cube_box = Box(cfg["cube"])

        for model in ("lorentz", "drude"):
            mat = _make_dispersive_material(model)
            f_guess = 140e6 if model == "lorentz" else 195e6

            for nx in nx_list:
                for arm in ("uniform_fine", "patch_r2", "uniform_coarse"):
                    eff_nx = 2 * nx if arm == "uniform_fine" else nx
                    axes = _axes_from_nx(BOX_DISP, eff_nx)
                    mesh = SubgridMesh3D(*axes)
                    if arm == "patch_r2":
                        mesh.add_box(_snap_box_to_axes(cfg["patch"], axes), ratio=2)

                    h_coarse = float(BOX_DISP[0] / nx)
                    h_fine = float(
                        BOX_DISP[0] / (2 * nx if arm != "uniform_coarse" else nx)
                    )
                    t_build = time.perf_counter()
                    solver = SubgridFIT3D(mesh, dispersive=[(cube_box, mat)], cfl=0.95)
                    build_s = float(time.perf_counter() - t_build)

                    freqs = dispersive_eigenfrequencies(solver, 2, f_guess)
                    f1 = float(freqs[0])
                    step_s = _measure_step_wall_s(solver, n_steps=10 if quick else 25)
                    n_pole_dofs = int(
                        sum(
                            len(g["poles"]) * int(g["idx"].size)
                            for g in solver.dispersive_groups
                        )
                    )
                    n_state = int(solver.n_faces + solver.n_free + 2 * n_pole_dofs)
                    work_per_ns = float(n_state / (solver.dt * 1e9))

                    row = dict(
                        case=case_name,
                        model=model,
                        arm=arm,
                        nx_coarse=int(nx),
                        h_coarse_m=h_coarse,
                        h_fine_m=h_fine,
                        n_free=int(solver.n_free),
                        n_faces=int(solver.n_faces),
                        n_pole_dofs=n_pole_dofs,
                        n_state=n_state,
                        dt_s=float(solver.dt),
                        dt_max_s=float(solver.dispersive_dt_max()),
                        f1_hz=f1,
                        sec_per_step=step_s,
                        build_s=build_s,
                        work_per_ns=work_per_ns,
                        wall_per_ns_s=float(step_s / (solver.dt * 1e9)),
                    )
                    rows.append(row)

    # Compute Richardson reference from uniform_fine for each (case, model)
    refs = {}
    for case_name in CASES_DISP:
        for model in ("lorentz", "drude"):
            uf = [
                r
                for r in rows
                if r["case"] == case_name
                and r["model"] == model
                and r["arm"] == "uniform_fine"
            ]
            f_star = _richardson_ref(
                [r["h_fine_m"] for r in uf], [r["f1_hz"] for r in uf], order=2.0
            )
            refs[f"{case_name}_{model}"] = f_star
            uf_by_nx = {r["nx_coarse"]: r["f1_hz"] for r in uf}
            for r in rows:
                if r["case"] == case_name and r["model"] == model:
                    r["f_ref_hz"] = f_star
                    r["rel_err_ref"] = float(abs(r["f1_hz"] - f_star) / f_star)
                    f_uf = uf_by_nx[r["nx_coarse"]]
                    r["rel_diff_uniform_fine"] = float(abs(r["f1_hz"] - f_uf) / f_uf)

    return dict(rows=rows, references_hz=refs)


def _sphere_eps_callable(X, Y, Z):
    cx, cy, cz = SPHERE_CENTER
    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    return np.where(r2 <= SPHERE_RADIUS**2, SPHERE_EPS, 1.0)


def _sphere_normal_callable(X, Y, Z):
    cx, cy, cz = SPHERE_CENTER
    dx = X - cx
    dy = Y - cy
    dz = Z - cz
    return np.stack([dx, dy, dz], axis=-1)


def run_sphere_study(quick=False):
    nx_list = [6, 8, 10] if quick else [6, 8, 10, 12, 16, 20]
    configs = (
        [
            ("staircase", 1, False),
            ("subpixel", 2, False),
            ("subpixel", 4, False),
            ("subpixel", 4, True),
        ]
        if quick
        else [
            ("staircase", 1, False),
            ("subpixel", 2, False),
            ("subpixel", 4, False),
            ("subpixel", 6, False),
            ("subpixel", 4, True),
        ]
    )

    f_guess = 185e6
    rows = []
    for nx in nx_list:
        axes = _axes_from_nx(BOX_SPHERE, nx)
        mesh = SubgridMesh3D(*axes)
        mesh.add_box(_snap_box_to_axes(SPHERE_PATCH, axes), ratio=2)
        h_coarse = float(BOX_SPHERE[0] / nx)
        h_fine = 0.5 * h_coarse

        for averaging, n_sub, use_analytic_normal in configs:
            t0 = time.perf_counter()
            solver = SubgridFIT3D(
                mesh,
                eps_r=_sphere_eps_callable,
                averaging=averaging,
                n_sub=n_sub,
                normal=_sphere_normal_callable if use_analytic_normal else None,
                dt=1e-11,
                check_stability=False,
            )
            freqs = solver.eigenfrequencies(4, f_guess)
            elapsed = float(time.perf_counter() - t0)
            label = (
                "staircase"
                if averaging == "staircase"
                else f"subpixel_nsub{n_sub}"
                + ("_exactN" if use_analytic_normal else "_momentN")
            )
            rows.append(
                dict(
                    label=label,
                    averaging=averaging,
                    n_sub=int(n_sub),
                    analytic_normal=bool(use_analytic_normal),
                    nx_coarse=int(nx),
                    h_coarse_m=h_coarse,
                    h_fine_m=h_fine,
                    n_free=int(solver.n_free),
                    freqs_hz=[float(v) for v in freqs],
                    wall_s=elapsed,
                )
            )

    # Reference: Richardson extrapolation on the finest subpixel_nsub4_exactN modes
    ref_arm = [r for r in rows if r["label"] == "subpixel_nsub4_exactN"]
    h_ref = [r["h_fine_m"] for r in ref_arm]
    f_star = [
        _richardson_ref(h_ref, [r["freqs_hz"][m] for r in ref_arm], order=2.0)
        for m in range(4)
    ]
    for r in rows:
        f_arr = np.asarray(r["freqs_hz"], float)
        ref_arr = np.asarray(f_star, float)
        rel_modes = np.abs(f_arr - ref_arr) / ref_arr
        r["rel_err_modes"] = [float(v) for v in rel_modes]
        r["rel_err_mean"] = float(np.mean(rel_modes))

    # Observed LS convergence orders
    orders = {}
    for label in sorted({r["label"] for r in rows}):
        sub = sorted(
            [r for r in rows if r["label"] == label], key=lambda x: x["nx_coarse"]
        )
        hs = np.array([r["h_fine_m"] for r in sub], float)
        es = np.array([r["rel_err_mean"] for r in sub], float)
        orders[label] = float(np.polyfit(np.log(hs), np.log(es), 1)[0])

    return dict(
        rows=rows,
        f_ref_hz=f_star,
        orders=orders,
        sphere=dict(
            box_m=list(BOX_SPHERE),
            center_m=list(SPHERE_CENTER),
            radius_m=SPHERE_RADIUS,
            eps_r=SPHERE_EPS,
            patch_m=[list(b) for b in SPHERE_PATCH],
        ),
    )


def _resolve_out_path(out_arg, quick):
    if quick:
        if not out_arg:
            raise ValueError("--quick requires an explicit --out DIR")
        abs_out = os.path.abspath(out_arg)
        forbidden = [
            os.path.join(REPO_ROOT, "results"),
            os.path.join(REPO_ROOT, "docs", "img"),
            HERE,
        ]
        for bad in forbidden:
            if abs_out == bad or abs_out.startswith(bad + os.sep):
                raise ValueError(f"--quick must not write into {bad}")
        if abs_out.endswith(".json"):
            return abs_out
        os.makedirs(abs_out, exist_ok=True)
        return os.path.join(abs_out, "media_quick.json")

    target_dir = os.path.abspath(out_arg) if out_arg else HERE
    if target_dir.endswith(".json"):
        return target_dir
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, "media.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Run fast subset (< 2 min)")
    ap.add_argument(
        "--out", default=None, help="Output directory (required for --quick)"
    )
    args = ap.parse_args()

    out_path = _resolve_out_path(args.out, args.quick)
    t0 = time.time()
    disp = run_dispersive_study(quick=args.quick)
    sphere = run_sphere_study(quick=args.quick)
    wall_s = float(time.time() - t0)

    payload = dict(
        meta=dict(
            host=platform.node(),
            cpus=os.cpu_count(),
            quick=bool(args.quick),
            numpy=np.__version__,
            wall_s=wall_s,
        ),
        dispersive=disp,
        sphere=sphere,
    )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Print grep-auditable summary line
    loc_lor = [
        r
        for r in disp["rows"]
        if r["case"] == "localized"
        and r["model"] == "lorentz"
        and r["arm"] == "patch_r2"
    ][-1]
    loc_uf = [
        r
        for r in disp["rows"]
        if r["case"] == "localized"
        and r["model"] == "lorentz"
        and r["arm"] == "uniform_fine"
    ][-1]
    sp_st = [r for r in sphere["rows"] if r["label"] == "staircase"][-1]
    sp_sub = [r for r in sphere["rows"] if r["label"] == "subpixel_nsub4_momentN"][-1]
    print(
        f"NUMBERS run_media: quick={args.quick} "
        f"loc_lor_patch_err={loc_lor['rel_err_ref']:.3e} "
        f"loc_lor_work_ratio={loc_uf['work_per_ns'] / loc_lor['work_per_ns']:.2f} "
        f"sphere_st_err={sp_st['rel_err_mean']:.3e} "
        f"sphere_sub4_err={sp_sub['rel_err_mean']:.3e} "
        f"wall_s={wall_s:.2f}"
    )


if __name__ == "__main__":
    main()
