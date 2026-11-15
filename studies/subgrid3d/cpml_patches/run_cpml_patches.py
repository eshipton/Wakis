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
import json
import os
import platform
import time

import numpy as np
from scipy.linalg import eig, eigvals

from wakis.subgrid3d import SubgridLTSFIT3D, SubgridMesh3D
from wakis.subgrid3d.pml import cpml_generator, diagnose_cpml_eigenvector

H_F = 0.04


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
        pml = dict(thickness=[(n_pml * H_F, n_pml * H_F)] * 3, R0=1e-6)
        pml.update(pml_kw or {})
    return SubgridLTSFIT3D(m, lts=False, dt=dt, pml=pml)


def run_reflection_row(n_int, n_pml, variant):
    L = n_int * H_F
    patch = variant != "nopatch"
    through = variant in ("patch_through_c1_alpha0", "patch_through_alpha1GHz")
    pml_kw = None
    if variant == "patch_through_c1_alpha0":
        pml_kw = dict(allow_patches=True, sigma_sampling="coarse_cell", alpha_hz=0.0)
    elif variant == "patch_through_alpha1GHz":
        pml_kw = dict(allow_patches=True, sigma_sampling="coarse_cell", alpha_hz=1.0e9)

    s_open = _box_f(n_int, n_pml, 0, patch, through, None, pml_kw)
    dt = s_open.dt
    f0 = 1.0e9
    tau = 0.5 / f0
    src_pt = (0.3 * L, 0.5 * L, 0.5 * L)
    prb = [
        (L - 2.5 * H_F, 0.45 * L, 0.55 * L),
        (0.5 * L + 0.5 * H_F, 0.5 * L, 0.5 * L),
    ]
    # Time gate before wall reflection from the 3.7x closed reference arrives
    t_end = 4.5e-9 if n_int < 20 else 8.0e-9
    ext_cells = 3 * n_int // 2 + n_pml
    sigs = []
    for ext in (0, ext_cells):
        s = (
            s_open
            if ext == 0
            else _box_f(n_int, n_pml, ext, patch, through, dt, pml_kw)
        )
        s.reset()
        s.sources, s.probes = [], []
        s.add_current(
            src_pt,
            2,
            lambda t: -(t - 4 * tau) / tau * np.exp(-(((t - 4 * tau) / tau) ** 2)),
        )
        for p in prb:
            s.add_probe(p, 2)
        s.run(int(t_end / dt))
        sigs.append(np.array([p["values"] for p in s.probes]))
    err = np.abs(sigs[0] - sigs[1]).max(axis=1) / (np.abs(sigs[1]).max(axis=1) + 1e-30)
    return {
        "n_int": int(n_int),
        "n_pml": int(n_pml),
        "n_ref_total": int(n_int + 2 * n_pml + 2 * ext_cells),
        "variant": variant,
        "dt": float(dt),
        "err_probes": [float(x) for x in err],
        "err": float(err.max()),
        "t": (np.arange(sigs[0].shape[1]) * dt).tolist(),
        "sig_open": sigs[0][0].tolist(),
        "sig_ref": sigs[1][0].tolist(),
    }


def run_spectrum_summary():
    """Compute semi-discrete generator spectrum and eigenvector diagnosis on
    an 8 x 5 x 5 coarse mesh with a 4-cell +x CPML layer."""
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
            pml=dict(
                thickness=th,
                R0=1e-6,
                order=3.0,
                alpha_hz=alpha_hz,
                allow_patches=True,
                sigma_sampling=sampling,
            ),
        )

    s_nop = _build(None, "fine")
    smax = s_nop.cpml.sigma_max
    re_nop = float(eigvals(cpml_generator(s_nop).toarray()).real.max() / smax)

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

    return {
        "sigma_max": float(smax),
        "re_over_smax": {
            "nopatch": re_nop,
            "patch_into_fine": re_into_fine,
            "patch_into_coarse_cell": re_into_c1,
            "patch_through_fine": float(diag_tf["re_over_sigma_max"]),
            "patch_through_coarse_cell": re_thru_c1,
        },
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
        out_dir.endswith(os.sep + "results")
        or (os.sep + "results" + os.sep) in out_dir
        or out_dir.endswith(os.sep + "img")
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
    print(
        f"NUMBERS run_cpml_patches: quick={a.quick} n_pml={n_hi} "
        f"err_nopatch={by_var[(n_hi, 'nopatch')]:.3e} "
        f"err_near={by_var[(n_hi, 'patch_near')]:.3e} "
        f"err_through_c1_a0={by_var[(n_hi, 'patch_through_c1_alpha0')]:.3e} "
        f"err_through_a1GHz={by_var[(n_hi, 'patch_through_alpha1GHz')]:.3e} "
        f"re_into_fine={spec['re_over_smax']['patch_into_fine']:.3e} "
        f"re_into_c1={spec['re_over_smax']['patch_into_coarse_cell']:.3e} "
        f"wall_s={wall_s:.2f}"
    )


if __name__ == "__main__":
    main()
