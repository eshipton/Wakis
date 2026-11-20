# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
3D subgridding study for dispersive (Drude/Lorentz ADE) and conformal
(subpixel Kottke) media inside patches (`wakis.subgrid3d.media`).

Studies
-------
(i) Dispersive and subpixel planar slab crossing a 2:1 patch (`run_slab_study`):
    - D6: Lorentz and lossless Drude slab crossing a 2:1 patch across
      N_z in {9, 18, 27} vs the exact 1D shooting reference (`solve_ivp` +
      `brentq`) and vs the uniform fine `SubgridFIT3D` baseline (without patch
      at fine step h_z = H_z / 2, equal to `SolverFIT3D`), plus the unlumped
      interface mutation `mutate="lump_no_rowsum"`.
    - C2: Subpixel Kottke vs staircase vs `mutate="swap_projection"` convergence
      for a non-node-plane dielectric slab (eps_r = 4.0, z_s = 0.361 c) crossing
      a 2:1 patch across N_z in {9, 13, 17, 21, 27}.
(ii) Dispersive cube (Lorentz and lossless Drude) inside a 2:1 patch vs the
     named baseline `SubgridFIT3D` without patch at the same fine step h
     (which equals `SolverFIT3D` per subgrid3d.md §4):
     - `localized`: small dispersive cube (3.3% of cavity volume) where the
       2:1 patch reduces state unknowns and work per unit physical time.
     - `exterior_limited` (honest case where the patch does not help):
       a larger dispersive block (20% of box volume) whose boundary coincides
       with the 2:1 hanging-edge patch interface while the global time step is
       pinned to the fine patch limit.
(iii) Staircase vs subpixel Kottke permittivity averaging for a dielectric
      sphere (eps_r = 4.0) inside a 2:1 patch, reporting `n_sub` and comparing
      the default dual-volume moment normal against the analytic radial normal.

Usage
-----
    python studies/subgrid3d/media/run_media.py [--quick] --out DIR
"""

import argparse
import gc
import json
import os
import platform
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.constants import c as C0
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from wakis.dispersive import DispersiveMaterial, Drude, Lorentz
from wakis.subgrid3d import (
    Box,
    SubgridFIT3D,
    SubgridMesh3D,
    dispersive_eigenfrequencies,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

BOX_DISP = (1.0, 0.8, 0.6)
CASES_DISP = {
    "localized": {
        "cube": ((0.4, 0.6), (0.2, 0.6), (0.2, 0.4)),
        "patch": ((0.2, 0.8), (0.2, 0.6), (0.2, 0.4)),
        "description": "Localized dispersive cube (3.3% box volume) inside a 2:1 patch",
    },
    "exterior_limited": {
        "cube": ((0.2, 0.8), (0.2, 0.6), (0.2, 0.4)),
        "patch": ((0.2, 0.8), (0.2, 0.6), (0.2, 0.4)),
        "description": (
            "Exterior-dominated dispersive block (20% box volume, patch bound == block bound): "
            "hanging-edge interface sits directly on the dielectric jump while global dt is fine-limited"
        ),
    },
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
    ny = max(3, round(nx * Ly / Lx))
    nz = max(3, round(nx * Lz / Lx))
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


def _shoot_1d_slab_modes(
    a, b, c, z_s, eps_of_omega, n_modes=6, f_min=120e6, f_max=400e6
):
    """Compute the lowest `n_modes` exact 1D TE_z and TM_z resonances [Hz] of a PEC box
    [0, a] x [0, b] x [0, c] with slab 0 < z < z_s of permittivity `eps_of_omega(omega)`
    and vacuum z_s < z < c, via `solve_ivp` (rtol=1e-12) + `brentq`.
    """
    candidates = []
    w_grid = np.linspace(2 * np.pi * f_min, 2 * np.pi * f_max, 25)

    def res_te(omega, kt2):
        k02 = (omega / C0) ** 2
        eps1 = float(eps_of_omega(omega))
        q1 = eps1 * k02 - kt2
        q2 = k02 - kt2
        sol1 = solve_ivp(
            lambda _z, y: [y[1], -q1 * y[0]],
            (0.0, z_s),
            [0.0, 1.0],
            rtol=1e-12,
            atol=1e-14,
        )
        y_mid = [float(sol1.y[0, -1]), float(sol1.y[1, -1])]
        sol2 = solve_ivp(
            lambda _z, y: [y[1], -q2 * y[0]],
            (z_s, c),
            y_mid,
            rtol=1e-12,
            atol=1e-14,
        )
        return float(sol2.y[0, -1])

    def res_tm(omega, kt2):
        k02 = (omega / C0) ** 2
        eps1 = float(eps_of_omega(omega))
        sol1 = solve_ivp(
            lambda _z, y: [-(k02 - kt2 / eps1) * y[1], eps1 * y[0]],
            (0.0, z_s),
            [0.0, 1.0],
            rtol=1e-12,
            atol=1e-14,
        )
        y_mid = [float(sol1.y[0, -1]), float(sol1.y[1, -1])]
        sol2 = solve_ivp(
            lambda _z, y: [-(k02 - kt2) * y[1], y[0]],
            (z_s, c),
            y_mid,
            rtol=1e-12,
            atol=1e-14,
        )
        return float(sol2.y[0, -1])

    for m in range(3):
        for n in range(3):
            if m == 0 and n == 0:
                continue
            kt2 = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
            vals = [res_te(w, kt2) for w in w_grid]
            for i in range(len(w_grid) - 1):
                if vals[i] * vals[i + 1] <= 0:
                    w_root = brentq(
                        res_te,
                        w_grid[i],
                        w_grid[i + 1],
                        args=(kt2,),
                        xtol=1e-13,
                        rtol=1e-13,
                    )
                    candidates.append((w_root / (2 * np.pi), f"TE_{m}{n}"))
            if m >= 1 and n >= 1:
                vals = [res_tm(w, kt2) for w in w_grid]
                for i in range(len(w_grid) - 1):
                    if vals[i] * vals[i + 1] <= 0:
                        w_root = brentq(
                            res_tm,
                            w_grid[i],
                            w_grid[i + 1],
                            args=(kt2,),
                            xtol=1e-13,
                            rtol=1e-13,
                        )
                        candidates.append((w_root / (2 * np.pi), f"TM_{m}{n}"))

    candidates.sort(key=lambda item: item[0])
    freqs = np.array([item[0] for item in candidates[:n_modes]], float)
    labels = [item[1] for item in candidates[:n_modes]]
    return freqs, labels


def _analytic_empty_box_modes(a, b, c, n_modes=6):
    """Lowest `n_modes` analytic cavity frequencies [Hz] of an empty PEC box."""
    modes = []
    for m in range(4):
        for n in range(4):
            for p in range(4):
                f = 0.5 * C0 * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / c) ** 2)
                if m >= 1 and n >= 1 and p >= 0:
                    modes.append(f)
                if p >= 1 and (m >= 1 or n >= 1):
                    modes.append(f)
    return np.sort(modes)[:n_modes]


def run_slab_study(quick=False):
    """Run (D6) dispersive Lorentz/Drude slab and (C2) subpixel dielectric slab
    convergence across a 2:1 subgrid patch vs the 1D shooting reference and vs
    uniform fine `SubgridFIT3D` (`== SolverFIT3D`).
    """
    # --- Part A: D6 Lorentz and lossless Drude slab crossing a 2:1 patch ---
    a_d6, b_d6, c_d6 = 0.38, 0.31, 0.90
    z_s_d6 = 0.30
    n_modes_d6 = 3
    nz_d6 = [9, 18] if quick else [9, 18, 27]

    f_1d_vac, _ = _shoot_1d_slab_modes(
        a_d6, b_d6, c_d6, z_s_d6, lambda _w: 1.0, n_modes=6, f_min=300e6, f_max=850e6
    )
    f_ana_vac = _analytic_empty_box_modes(a_d6, b_d6, c_d6, n_modes=6)
    vac_1d_err = float(np.max(np.abs(f_1d_vac - f_ana_vac) / f_ana_vac))

    f0_lor = 520e6
    w0_lor = 2 * np.pi * f0_lor
    mat_lor = DispersiveMaterial(eps_inf=1.0, poles=[Lorentz(6.0, w0_lor, 0.0)])
    s_lor = 6.0 * w0_lor**2
    f_ref_lor, lbl_lor = _shoot_1d_slab_modes(
        a_d6,
        b_d6,
        c_d6,
        z_s_d6,
        lambda w: 1.0 + s_lor / (w0_lor**2 - w**2),
        n_modes=n_modes_d6,
        f_min=160e6,
        f_max=420e6,
    )

    wp_dru = 2 * np.pi * 380e6
    mat_dru = DispersiveMaterial(eps_inf=1.0, poles=[Drude(wp_dru, 0.0)])
    f_ref_dru, lbl_dru = _shoot_1d_slab_modes(
        a_d6,
        b_d6,
        c_d6,
        z_s_d6,
        lambda w: 1.0 - (wp_dru / w) ** 2,
        n_modes=n_modes_d6,
        f_min=400e6,
        f_max=750e6,
    )

    slab_box_d6 = Box(((-0.1, a_d6 + 0.1), (-0.1, b_d6 + 0.1), (-0.1, z_s_d6)))
    err_lor_1d, err_lor_fine, err_lor_mut, err_lor_mut_fine = [], [], [], []
    err_dru_1d, err_dru_fine, err_dru_mut, err_dru_mut_fine = [], [], [], []
    hz_coarse_d6, hz_fine_d6 = [], []

    for nz in nz_d6:
        s = nz // 9
        nx, ny = 3 * s, 3 * s
        x = np.linspace(0.0, a_d6, nx + 1)
        y = np.linspace(0.0, b_d6, ny + 1)
        z = np.linspace(0.0, c_d6, nz + 1)
        hz_coarse_d6.append(float(c_d6 / nz))
        hz_fine_d6.append(float(c_d6 / (2 * nz)))

        mesh_p = SubgridMesh3D(x, y, z)
        mesh_p.add_patch((1 * s, 1 * s, 1 * s), (2 * s, 2 * s, 5 * s), ratio=2)

        xf = np.linspace(0.0, a_d6, 2 * nx + 1)
        yf = np.linspace(0.0, b_d6, 2 * ny + 1)
        zf = np.linspace(0.0, c_d6, 2 * nz + 1)
        mesh_f = SubgridMesh3D(xf, yf, zf)

        slv_lp = SubgridFIT3D(
            mesh_p, dispersive=[(slab_box_d6, mat_lor)], dt=1e-11, check_stability=False
        )
        f_lp = dispersive_eigenfrequencies(slv_lp, n_modes_d6, f_ref_lor[0])
        del slv_lp
        gc.collect()

        slv_lf = SubgridFIT3D(
            mesh_f, dispersive=[(slab_box_d6, mat_lor)], dt=1e-11, check_stability=False
        )
        f_lf = dispersive_eigenfrequencies(slv_lf, n_modes_d6, f_ref_lor[0])
        del slv_lf
        gc.collect()

        slv_lm = SubgridFIT3D(
            mesh_p,
            dispersive=[(slab_box_d6, mat_lor)],
            dt=1e-11,
            check_stability=False,
            mutate="lump_no_rowsum",
        )
        f_lm = dispersive_eigenfrequencies(slv_lm, n_modes_d6, f_ref_lor[0])
        del slv_lm
        gc.collect()

        err_lor_1d.append(float(np.mean(np.abs(f_lp - f_ref_lor) / f_ref_lor)))
        err_lor_fine.append(float(np.mean(np.abs(f_lp - f_lf) / f_lf)))
        err_lor_mut.append(float(np.mean(np.abs(f_lm - f_ref_lor) / f_ref_lor)))
        err_lor_mut_fine.append(float(np.mean(np.abs(f_lm - f_lf) / f_lf)))

        f_guess_dru = 1.85 * f_ref_dru[0]
        slv_dp = SubgridFIT3D(
            mesh_p, dispersive=[(slab_box_d6, mat_dru)], dt=1e-11, check_stability=False
        )
        f_dp = dispersive_eigenfrequencies(slv_dp, n_modes_d6, f_guess_dru)
        del slv_dp
        gc.collect()

        slv_df = SubgridFIT3D(
            mesh_f, dispersive=[(slab_box_d6, mat_dru)], dt=1e-11, check_stability=False
        )
        f_df = dispersive_eigenfrequencies(slv_df, n_modes_d6, f_guess_dru)
        del slv_df
        gc.collect()

        slv_dm = SubgridFIT3D(
            mesh_p,
            dispersive=[(slab_box_d6, mat_dru)],
            dt=1e-11,
            check_stability=False,
            mutate="lump_no_rowsum",
        )
        f_dm = dispersive_eigenfrequencies(slv_dm, n_modes_d6, f_guess_dru)
        del slv_dm
        gc.collect()

        err_dru_1d.append(float(np.mean(np.abs(f_dp - f_ref_dru) / f_ref_dru)))
        err_dru_fine.append(float(np.mean(np.abs(f_dp - f_df) / f_df)))
        err_dru_mut.append(float(np.mean(np.abs(f_dm - f_ref_dru) / f_ref_dru)))
        err_dru_mut_fine.append(float(np.mean(np.abs(f_dm - f_df) / f_df)))

    def _orders(errs, nzs):
        return [
            float(np.log(errs[i] / errs[i + 1]) / np.log(nzs[i + 1] / nzs[i]))
            for i in range(len(nzs) - 1)
        ]

    # --- Part B: C2 subpixel vs staircase dielectric slab crossing a 2:1 patch ---
    a_c2, b_c2, c_c2 = 1.0, 0.7, 0.9
    r_c2 = 0.361
    z_s_c2 = r_c2 * c_c2
    nz_c2 = [9, 13, 17] if quick else [9, 13, 17, 21, 27]
    f_ref_c2, lbl_c2 = _shoot_1d_slab_modes(
        a_c2, b_c2, c_c2, z_s_c2, lambda _w: 4.0, n_modes=4, f_min=100e6, f_max=260e6
    )
    slab_box_c2 = Box(
        ((-0.1, a_c2 + 0.1), (-0.1, b_c2 + 0.1), (-0.1, z_s_c2)), value=4.0
    )

    err_sub, err_st, err_mut = [], [], []
    hz_coarse_c2, hz_fine_c2 = [], []
    for nz in nz_c2:
        nx = max(6, round(nz * 6 / 9))
        ny = max(6, round(nz * 6 / 9))
        x = np.linspace(0.0, a_c2, nx + 1)
        y = np.linspace(0.0, b_c2, ny + 1)
        z = np.linspace(0.0, c_c2, nz + 1)
        hz_coarse_c2.append(float(c_c2 / nz))
        hz_fine_c2.append(float(c_c2 / (2 * nz)))
        mesh = SubgridMesh3D(x, y, z)
        k1 = max(1, int(np.floor(r_c2 * nz)) - 1)
        k2 = min(nz - 1, int(np.ceil(r_c2 * nz)) + 1)
        mesh.add_patch((1, 1, k1), (nx // 2 + 1, ny // 2 + 1, k2), ratio=2)

        f_sub = SubgridFIT3D(
            mesh,
            eps_r=slab_box_c2,
            averaging="subpixel",
            dt=1e-11,
            check_stability=False,
        ).eigenfrequencies(4, f_ref_c2[0])
        f_st = SubgridFIT3D(
            mesh,
            eps_r=slab_box_c2,
            averaging="staircase",
            dt=1e-11,
            check_stability=False,
        ).eigenfrequencies(4, f_ref_c2[0])
        f_mut = SubgridFIT3D(
            mesh,
            eps_r=slab_box_c2,
            averaging="subpixel",
            mutate="swap_projection",
            dt=1e-11,
            check_stability=False,
        ).eigenfrequencies(4, f_ref_c2[0])
        gc.collect()

        err_sub.append(float(np.mean(np.abs(f_sub - f_ref_c2) / f_ref_c2)))
        err_st.append(float(np.mean(np.abs(f_st - f_ref_c2) / f_ref_c2)))
        err_mut.append(float(np.mean(np.abs(f_mut - f_ref_c2) / f_ref_c2)))

    pairwise_sub = _orders(err_sub, nz_c2)
    ls_sub = float(-np.polyfit(np.log(nz_c2), np.log(err_sub), 1)[0])
    ls_st = float(-np.polyfit(np.log(nz_c2), np.log(err_st), 1)[0])
    ls_mut = float(-np.polyfit(np.log(nz_c2), np.log(err_mut), 1)[0])

    return {
        "d6_dispersive": {
            "box_m": [a_d6, b_d6, c_d6],
            "z_s_m": z_s_d6,
            "patch_bounds_m": [
                [a_d6 / 3.0, 2.0 * a_d6 / 3.0],
                [b_d6 / 3.0, 2.0 * b_d6 / 3.0],
                [c_d6 / 9.0, 5.0 * c_d6 / 9.0],
            ],
            "nz_list": [int(v) for v in nz_d6],
            "hz_coarse_m": hz_coarse_d6,
            "hz_fine_m": hz_fine_d6,
            "vac_1d_err": vac_1d_err,
            "lbl_lor": lbl_lor,
            "lbl_dru": lbl_dru,
            "f_ref_lor_hz": [float(v) for v in f_ref_lor],
            "f_ref_dru_hz": [float(v) for v in f_ref_dru],
            "err_lor_1d": err_lor_1d,
            "err_lor_fine": err_lor_fine,
            "err_lor_mut": err_lor_mut,
            "err_lor_mut_fine": err_lor_mut_fine,
            "err_dru_1d": err_dru_1d,
            "err_dru_fine": err_dru_fine,
            "err_dru_mut": err_dru_mut,
            "err_dru_mut_fine": err_dru_mut_fine,
            "ord_lor_1d": _orders(err_lor_1d, nz_d6),
            "ord_lor_fine": _orders(err_lor_fine, nz_d6),
            "ord_lor_mut": _orders(err_lor_mut, nz_d6),
            "ord_lor_mut_fine": _orders(err_lor_mut_fine, nz_d6),
            "ord_dru_1d": _orders(err_dru_1d, nz_d6),
            "ord_dru_fine": _orders(err_dru_fine, nz_d6),
            "ord_dru_mut": _orders(err_dru_mut, nz_d6),
            "ord_dru_mut_fine": _orders(err_dru_mut_fine, nz_d6),
            "ls_ord_lor_1d": float(
                -np.polyfit(np.log(nz_d6), np.log(err_lor_1d), 1)[0]
            ),
            "ls_ord_lor_fine": float(
                -np.polyfit(np.log(nz_d6), np.log(err_lor_fine), 1)[0]
            ),
            "ls_ord_dru_1d": float(
                -np.polyfit(np.log(nz_d6), np.log(err_dru_1d), 1)[0]
            ),
            "ls_ord_dru_fine": float(
                -np.polyfit(np.log(nz_d6), np.log(err_dru_fine), 1)[0]
            ),
        },
        "c2_subpixel": {
            "box_m": [a_c2, b_c2, c_c2],
            "r_frac": r_c2,
            "z_s_m": z_s_c2,
            "eps_r": 4.0,
            "nz_list": [int(v) for v in nz_c2],
            "hz_coarse_m": hz_coarse_c2,
            "hz_fine_m": hz_fine_c2,
            "modes": lbl_c2,
            "f_ref_hz": [float(v) for v in f_ref_c2],
            "err_sub": err_sub,
            "err_st": err_st,
            "err_mut": err_mut,
            "pairwise_sub": pairwise_sub,
            "ls_sub": ls_sub,
            "ls_st": ls_st,
            "ls_mut": ls_mut,
        },
    }


def run_dispersive_study(quick=False):
    # nx in [5, 10, 15] keeps exact integer cell counts (5s, 4s, 3s) on BOX_DISP = (1.0, 0.8, 0.6)
    # and fits inside the 2000 MB memguard cap (uniform_fine at nx=15 has eff_nx=30, n_free=35,568,
    # ~885 MB peak RSS; nx=20 uniform_fine requires 2283 MB RSS in 3D SuperLU splu).
    nx_list = [5, 10] if quick else [5, 10, 15]
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

                    row = {
                        "case": case_name,
                        "model": model,
                        "arm": arm,
                        "nx_coarse": int(nx),
                        "h_coarse_m": h_coarse,
                        "h_fine_m": h_fine,
                        "n_free": int(solver.n_free),
                        "n_faces": int(solver.n_faces),
                        "n_pole_dofs": n_pole_dofs,
                        "n_state": n_state,
                        "dt_s": float(solver.dt),
                        "dt_max_s": float(solver.dispersive_dt_max()),
                        "f1_hz": f1,
                        "sec_per_step": step_s,
                        "build_s": build_s,
                        "work_per_ns": work_per_ns,
                        "wall_per_ns_s": float(step_s / (solver.dt * 1e9)),
                    }
                    rows.append(row)
                    del solver, mesh
                    gc.collect()

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

    return {"rows": rows, "references_hz": refs}


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
                {
                    "label": label,
                    "averaging": averaging,
                    "n_sub": int(n_sub),
                    "analytic_normal": bool(use_analytic_normal),
                    "nx_coarse": int(nx),
                    "h_coarse_m": h_coarse,
                    "h_fine_m": h_fine,
                    "n_free": int(solver.n_free),
                    "freqs_hz": [float(v) for v in freqs],
                    "wall_s": elapsed,
                }
            )
            del solver
            gc.collect()

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

    return {
        "rows": rows,
        "f_ref_hz": f_star,
        "orders": orders,
        "sphere": {
            "box_m": list(BOX_SPHERE),
            "center_m": list(SPHERE_CENTER),
            "radius_m": SPHERE_RADIUS,
            "eps_r": SPHERE_EPS,
            "patch_m": [list(b) for b in SPHERE_PATCH],
        },
    }


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
    slab = run_slab_study(quick=args.quick)
    disp = run_dispersive_study(quick=args.quick)
    sphere = run_sphere_study(quick=args.quick)
    wall_s = float(time.time() - t0)

    payload = {
        "meta": {
            "host": platform.node(),
            "cpus": os.cpu_count(),
            "quick": bool(args.quick),
            "numpy": np.__version__,
            "wall_s": wall_s,
        },
        "slab": slab,
        "dispersive": disp,
        "sphere": sphere,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Print grep-auditable summary line
    d6 = slab["d6_dispersive"]
    c2 = slab["c2_subpixel"]
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
        f"d6_lor_ls_ord={d6['ls_ord_lor_1d']:.3f} "
        f"d6_dru_ls_ord={d6['ls_ord_dru_1d']:.3f} "
        f"c2_sub_ls_ord={c2['ls_sub']:.3f} "
        f"loc_lor_patch_err={loc_lor['rel_err_ref']:.3e} "
        f"loc_lor_work_ratio={loc_uf['work_per_ns'] / loc_lor['work_per_ns']:.2f} "
        f"sphere_st_err={sp_st['rel_err_mean']:.3e} "
        f"sphere_sub4_err={sp_sub['rel_err_mean']:.3e} "
        f"wall_s={wall_s:.2f}"
    )


if __name__ == "__main__":
    main()
