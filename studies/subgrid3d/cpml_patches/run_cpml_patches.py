# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Study script for W4-3a Part 3: CFS-CPML with 3D subgrid patches reaching into
or through the absorbing layer.

Implements the Part F time-gated reflection benchmark of
``studies/subgrid3d/run_study_lts.py`` (probe 2.5 coarse cells from the layer
compared against a ~3.7x larger closed reference box, layers of 4--12 cells):
  1. ``nopatch``: uniform coarse grid without patch;
  2. ``patch_near``: 2:1 patch stopping one coarse cell outside the layer;
  3. ``patch_through_c1_alpha0``: 2:1 patch reaching through the layer with
     Candidate C1 sampling (``sigma_sampling="coarse_cell"``, ``alpha_hz=0``);
  4. ``patch_through_alpha1GHz``: 2:1 patch reaching through the layer with
     CFS shift ``alpha / (2 pi) = 1 GHz``.

Also records the semi-discrete generator spectrum ``G = cpml_generator(solver)``
and unstable eigenvector localisation for ``plot_cpml_patches.py``.
"""

import argparse
import gc
import json
import os
import platform
import time

import numpy as np
from scipy.linalg import eig, eigvals

from wakis.subgrid3d import SubgridLTSFIT3D, SubgridMesh3D
from wakis.subgrid3d.pml import cpml_generator, diagnose_cpml_eigenvector

H_F = 0.04


def _make_cpml_solver(
    dims=(8, 5, 5),
    patch_cells=None,
    ratio=2,
    layer_axes=((0, 1, 4),),
    alpha_hz=10.0e6,
    sigma_sampling="fine",
    R0=1e-6,
    order=3.0,
    H=H_F,
    dt=None,
):
    axes = [np.linspace(0.0, n * H, n + 1) for n in dims]
    m = SubgridMesh3D(*axes)
    if patch_cells is not None:
        bounds = tuple((lo * H, hi * H) for lo, hi in patch_cells)
        m.add_box(bounds, ratio=ratio)
    th = np.zeros((3, 2))
    for d, side, n_cells in layer_axes:
        th[d, side] = n_cells * H
    return SubgridLTSFIT3D(
        m,
        lts=False,
        dt=dt,
        pml={
            "thickness": th,
            "R0": R0,
            "order": order,
            "alpha_hz": alpha_hz,
            "allow_patches": True,
            "sigma_sampling": sigma_sampling,
        },
    )


def _box_f(n_int, n_pml, n_ext, patch, through, dt, pml_kw=None):
    """Interior ``n_int`` cells + ``n_pml`` layer cells each side (+ ``n_ext``
    cells of closed padding for the reference)."""
    n = n_int + 2 * n_pml + 2 * n_ext
    x = (np.arange(n + 1) - (n_pml + n_ext)) * H_F
    m = SubgridMesh3D(x, x, x)
    if patch:
        # Snap y, z bounds to coarse node planes near 0.4 L .. 0.6 L
        j0 = max(1, round(0.35 * n_int))
        j1 = min(n_int - 1, round(0.65 * n_int))
        i0 = max(1, n_int // 2)
        i1 = (n_int + n_pml - 1) if through else (n_int - 1)
        m.add_box(
            ((i0 * H_F, i1 * H_F), (j0 * H_F, j1 * H_F), (j0 * H_F, j1 * H_F)),
            ratio=2,
        )
    pml = None
    if n_pml and n_ext == 0:
        pml = {"thickness": [(n_pml * H_F, n_pml * H_F)] * 3, "R0": 1e-6}
        pml.update(pml_kw or {})
    return SubgridLTSFIT3D(m, lts=False, dt=dt, pml=pml)


_REF_CACHE = {}


def run_reflection_row(n_int, n_pml, variant):
    L = n_int * H_F
    patch = variant != "nopatch"
    through = variant in ("patch_through_c1_alpha0", "patch_through_alpha1GHz")
    pml_kw = None
    if variant == "patch_through_c1_alpha0":
        pml_kw = {
            "allow_patches": True,
            "sigma_sampling": "coarse_cell",
            "alpha_hz": 0.0,
        }
    elif variant == "patch_through_alpha1GHz":
        pml_kw = {
            "allow_patches": True,
            "sigma_sampling": "coarse_cell",
            "alpha_hz": 1.0e9,
        }

    s_open = _box_f(n_int, n_pml, 0, patch, through, None, pml_kw)
    dt = float(s_open.dt)
    f0 = 1.0e9
    tau = 0.5 / f0
    src_pt = (0.3 * L, 0.5 * L, 0.5 * L)
    prb = [
        (L - 2.5 * H_F, 0.45 * L, 0.55 * L),
        (0.5 * L + 0.5 * H_F, 0.5 * L, 0.5 * L),
    ]
    # Time gate before wall reflection from the 3.7x closed reference arrives
    t_end = 4.5e-9 if n_int < 20 else 8.0e-9
    pad_cells = max(n_pml + 10, round(1.35 * n_int))
    ext_cells = pad_cells - n_pml

    s_open.reset()
    s_open.sources, s_open.probes = [], []
    s_open.add_current(
        src_pt,
        2,
        lambda t: -(t - 4 * tau) / tau * np.exp(-(((t - 4 * tau) / tau) ** 2)),
    )
    for p in prb:
        s_open.add_probe(p, 2)
    s_open.run(int(t_end / dt))
    sig_open_arr = np.array([p["values"] for p in s_open.probes])
    del s_open
    gc.collect()

    ref_key = (
        n_int,
        pad_cells,
        patch,
        through,
        n_pml if through else 0,
        round(dt * 1e18, 6),
    )
    if ref_key not in _REF_CACHE:
        s_ref = _box_f(n_int, n_pml, ext_cells, patch, through, dt, pml_kw)
        s_ref.reset()
        s_ref.sources, s_ref.probes = [], []
        s_ref.add_current(
            src_pt,
            2,
            lambda t: -(t - 4 * tau) / tau * np.exp(-(((t - 4 * tau) / tau) ** 2)),
        )
        for p in prb:
            s_ref.add_probe(p, 2)
        s_ref.run(int(t_end / dt))
        _REF_CACHE[ref_key] = np.array([p["values"] for p in s_ref.probes])
        del s_ref
        gc.collect()
    sig_ref_arr = _REF_CACHE[ref_key]

    err = np.abs(sig_open_arr - sig_ref_arr).max(axis=1) / (
        np.abs(sig_ref_arr).max(axis=1) + 1e-30
    )
    return {
        "n_int": int(n_int),
        "n_pml": int(n_pml),
        "n_ref_total": int(n_int + 2 * n_pml + 2 * ext_cells),
        "variant": variant,
        "dt": float(dt),
        "err_probes": [float(x) for x in err],
        "err": float(err.max()),
        "t": (np.arange(sig_open_arr.shape[1]) * dt).tolist(),
        "sig_open": sig_open_arr[0].tolist(),
        "sig_ref": sig_ref_arr[0].tolist(),
    }


def run_spectrum_summary():
    """Compute semi-discrete generator spectrum and eigenvector diagnosis on
    an 8 x 5 x 5 coarse mesh with a 4-cell +x CPML layer and across the T1
    geometry matrix of `tests/test_121_subgrid3d_cpml_patches.py`."""
    dims = (8, 5, 5)
    axes = [np.linspace(0.0, n * H_F, n + 1) for n in dims]
    th = np.zeros((3, 2))
    th[0, 1] = 4 * H_F

    def _build(patch_cells, sampling, alpha_hz=10.0e6):
        m = SubgridMesh3D(*axes)
        if patch_cells is not None:
            m.add_box(tuple((a * H_F, b * H_F) for a, b in patch_cells), ratio=2)
        return SubgridLTSFIT3D(
            m,
            lts=False,
            pml={
                "thickness": th,
                "R0": 1e-6,
                "order": 3.0,
                "alpha_hz": alpha_hz,
                "allow_patches": True,
                "sigma_sampling": sampling,
            },
        )

    s_nop = _build(None, "fine")
    smax = s_nop.cpml.sigma_max
    re_nop = float(eigvals(cpml_generator(s_nop).toarray()).real.max() / smax)

    s_nop_a0 = _build(None, "fine", alpha_hz=0.0)
    re_nop_a0 = float(eigvals(cpml_generator(s_nop_a0).toarray()).real.max() / smax)

    s_near_a0 = _build(((1, 3), (1, 4), (1, 4)), "fine", alpha_hz=0.0)
    re_near_a0 = float(eigvals(cpml_generator(s_near_a0).toarray()).real.max() / smax)

    s_into_fine = _build(((3, 6), (1, 4), (1, 4)), "fine")
    re_into_fine = float(
        eigvals(cpml_generator(s_into_fine).toarray()).real.max() / smax
    )

    s_into_c1 = _build(((3, 6), (1, 4), (1, 4)), "coarse_cell")
    re_into_c1 = float(eigvals(cpml_generator(s_into_c1).toarray()).real.max() / smax)

    s_thru_fine = _build(((3, 7), (1, 4), (1, 4)), "fine")
    vals_tf, vecs_tf = eig(cpml_generator(s_thru_fine).toarray())
    diag_tf = diagnose_cpml_eigenvector(s_thru_fine, vals=vals_tf, vecs=vecs_tf)

    s_thru_c1 = _build(((3, 7), (1, 4), (1, 4)), "coarse_cell")
    re_thru_c1 = float(eigvals(cpml_generator(s_thru_c1).toarray()).real.max() / smax)

    s_thru_c1_a0 = _build(((3, 7), (1, 4), (1, 4)), "coarse_cell", alpha_hz=0.0)
    re_thru_c1_a0 = float(
        eigvals(cpml_generator(s_thru_c1_a0).toarray()).real.max() / smax
    )

    s_thru_a1g = _build(((3, 7), (1, 4), (1, 4)), "coarse_cell", alpha_hz=1.0e9)
    re_thru_a1g = float(eigvals(cpml_generator(s_thru_a1g).toarray()).real.max() / smax)

    # Full T1 geometry spectrum matrix (plus tube_2to1)
    t1_cases = [
        ("nopatch_x", (6, 4, 4), None, 2, ((0, 1, 3),)),
        ("x_into_2to1", (7, 5, 5), ((3, 5), (1, 4), (1, 4)), 2, ((0, 1, 4),)),
        ("y_into_2to1", (5, 7, 5), ((1, 4), (3, 5), (1, 4)), 2, ((1, 1, 4),)),
        ("z_into_2to1", (5, 5, 7), ((1, 4), (1, 4), (3, 5)), 2, ((2, 1, 4),)),
        ("tube_2to1", (6, 4, 4), ((0, 6), (1, 3), (1, 3)), 2, ((0, 1, 3),)),
        ("x_through_3to1", (5, 3, 3), ((2, 5), (1, 2), (1, 2)), 3, ((0, 1, 3),)),
        (
            "corner_xy_2to1",
            (5, 5, 4),
            ((2, 5), (2, 5), (1, 3)),
            2,
            ((0, 1, 3), (1, 1, 3)),
        ),
        ("x_into_3to1", (5, 3, 3), ((1, 3), (1, 2), (1, 2)), 3, ((0, 1, 3),)),
        ("x_through_2to1", (6, 4, 4), ((2, 6), (1, 3), (1, 3)), 2, ((0, 1, 3),)),
        ("y_through_2to1", (4, 6, 4), ((1, 3), (2, 6), (1, 3)), 2, ((1, 1, 3),)),
        ("z_through_2to1", (4, 4, 6), ((1, 3), (1, 3), (2, 6)), 2, ((2, 1, 3),)),
    ]
    t1_rows = []
    for name, c_dims, patch_cells, ratio, layer_axes in t1_cases:
        s_10m = _make_cpml_solver(
            dims=c_dims,
            patch_cells=patch_cells,
            ratio=ratio,
            layer_axes=layer_axes,
            alpha_hz=10.0e6,
            sigma_sampling="coarse_cell",
        )
        sm_c = float(s_10m.cpml.sigma_max)
        vals_10m = eigvals(cpml_generator(s_10m).toarray())
        idx_10m = int(np.argmax(vals_10m.real))
        re_c1_10m = float(vals_10m[idx_10m].real / sm_c)
        freq_c1_10m = float(abs(vals_10m[idx_10m].imag) / (2 * np.pi))

        s_0 = _make_cpml_solver(
            dims=c_dims,
            patch_cells=patch_cells,
            ratio=ratio,
            layer_axes=layer_axes,
            alpha_hz=0.0,
            sigma_sampling="coarse_cell",
        )
        re_c1_0 = float(eigvals(cpml_generator(s_0).toarray()).real.max() / sm_c)

        if patch_cells is not None:
            s_f = _make_cpml_solver(
                dims=c_dims,
                patch_cells=patch_cells,
                ratio=ratio,
                layer_axes=layer_axes,
                alpha_hz=10.0e6,
                sigma_sampling="fine",
            )
            re_fine_10m = float(
                eigvals(cpml_generator(s_f).toarray()).real.max() / sm_c
            )
        else:
            re_fine_10m = re_c1_10m

        t1_rows.append(
            {
                "geometry": name,
                "ratio": int(ratio),
                "re_c1_10m": re_c1_10m,
                "re_c1_0": re_c1_0,
                "re_fine_10m": re_fine_10m,
                "freq_c1_10m_hz": freq_c1_10m,
                "stable_c1_10m": bool(re_c1_10m <= 1e-6),
                "stable_fine_10m": bool(re_fine_10m <= 1e-6),
            }
        )
        gc.collect()

    return {
        "sigma_max": float(smax),
        "n_state": {
            "nopatch": int(s_nop.n_state),
            "patch_near": int(s_near_a0.n_state),
            "patch_through": int(s_thru_fine.n_state),
        },
        "re_over_smax": {
            "nopatch": re_nop,
            "patch_into_fine": re_into_fine,
            "patch_into_coarse_cell": re_into_c1,
            "patch_through_fine": float(diag_tf["re_over_sigma_max"]),
            "patch_through_coarse_cell": re_thru_c1,
        },
        "reflection_variants_re_over_smax": {
            "nopatch": re_nop_a0,
            "patch_near": re_near_a0,
            "patch_through_c1_alpha0": re_thru_c1_a0,
            "patch_through_alpha1GHz": re_thru_a1g,
        },
        "t1_geometries": t1_rows,
        "diagnosis_fine": {
            "lambda_re": float(diag_tf["lambda_max"].real),
            "lambda_im": float(diag_tf["lambda_max"].imag),
            "freq_hz": float(diag_tf["freq_hz"]),
            "block_energy_share": diag_tf["block_energy_share"],
            "interface_face_share_of_b": float(diag_tf["interface_face_share_of_b"]),
            "interface_edge_share_of_e": float(diag_tf["interface_edge_share_of_e"]),
            "interface_total_share": float(diag_tf["interface_total_share"]),
            "depth_quartile_share": diag_tf["depth_quartile_share"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", required=True, help="Output directory for JSON results")
    a = ap.parse_args()

    out_dir = os.path.abspath(a.out)
    if a.quick and (
        out_dir.endswith((os.sep + "results", os.sep + "img"))
        or (os.sep + "results" + os.sep) in out_dir
    ):
        raise ValueError("--quick must not write into results/ or docs/img/")
    os.makedirs(out_dir, exist_ok=True)
    fname = "cpml_patches_quick.json" if a.quick else "cpml_patches.json"
    out_path = os.path.join(out_dir, fname)

    t0 = time.time()
    n_int = 8 if a.quick else 20
    thick = [4, 6] if a.quick else [4, 6, 8, 12]
    variants = (
        "nopatch",
        "patch_near",
        "patch_through_c1_alpha0",
        "patch_through_alpha1GHz",
    )

    cpml_rows = []
    for n_pml in thick:
        for v in variants:
            cpml_rows.append(run_reflection_row(n_int, n_pml, v))

    spec = run_spectrum_summary()
    refl_re = spec["reflection_variants_re_over_smax"]
    for r in cpml_rows:
        re_val = float(refl_re[r["variant"]])
        r["re_over_smax"] = re_val
        r["spectrally_stable"] = bool(re_val <= 1e-6)
    wall_s = time.time() - t0

    res = {
        "meta": {
            "host": platform.node(),
            "quick": bool(a.quick),
            "numpy": np.__version__,
            "wall_s": float(wall_s),
        },
        "reflection": cpml_rows,
        "spectrum": spec,
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    by_var = {(r["n_pml"], r["variant"]): r["err"] for r in cpml_rows}
    n_hi = thick[-1]
    nst = spec["n_state"]
    print(
        f"NUMBERS run_cpml_patches: quick={a.quick} n_pml={n_hi} "
        f"sigma_max={spec['sigma_max']:.3e} "
        f"n_state_nopatch={nst['nopatch']} n_state_near={nst['patch_near']} n_state_through={nst['patch_through']} "
        f"err_nopatch={by_var[(n_hi, 'nopatch')]:.3e} "
        f"err_near={by_var[(n_hi, 'patch_near')]:.3e} "
        f"err_through_c1_a0={by_var[(n_hi, 'patch_through_c1_alpha0')]:.3e} "
        f"err_through_a1GHz={by_var[(n_hi, 'patch_through_alpha1GHz')]:.3e} "
        f"re_nopatch={refl_re['nopatch']:.3e} "
        f"re_near={refl_re['patch_near']:.3e} "
        f"re_through_c1_a0={refl_re['patch_through_c1_alpha0']:.3e} "
        f"re_through_a1GHz={refl_re['patch_through_alpha1GHz']:.3e} "
        f"re_into_fine={spec['re_over_smax']['patch_into_fine']:.3e} "
        f"re_into_c1={spec['re_over_smax']['patch_into_coarse_cell']:.3e} "
        f"re_through_fine={spec['re_over_smax']['patch_through_fine']:.3e} "
        f"re_through_c1={spec['re_over_smax']['patch_through_coarse_cell']:.3e} "
        f"wall_s={wall_s:.2f}"
    )


if __name__ == "__main__":
    main()
