"""Falsifier: steady near-wall node charge of the wall-born m = 0 sheet beam (Gauss's law).

``wall_gather_model.deposit_node_profile`` derives the steady node charge a wall-born order-2
stream leaves next to the ``z = 0`` PEC wall, per interior node charge:

``c_k = dzeta * sum_{j >= 1} [W2(j dzeta - k) - s W2(-j dzeta - k)]``

(``s = 0`` masked, ``s = 1`` image), i.e. ``c_1 = 1 - 1/48 - dzeta/16 + dzeta^2/24`` (masked) and
``c_1 = 1 - 2/48`` (image). The ``j >= 1`` (the sheet born in the current step is not yet part of
``E^n``) and the cancellation of the ``dzeta`` terms in image mode are the risky parts: with
``j >= 0`` the masked deficit would be ``1/48 - dzeta/16``.

This script runs the drift-tunnel beam of ``examples/_space_charge_waves_common.py`` (m = 0,
constant current) for ``40 / dzeta`` steps and measures the node charge the field actually
holds, ``q_k = sum_i (S~ d~)_{i,k}`` (dual divergence of the FIT flux, i.e. Gauss's law on each
primal node plane), normalised by its mean over planes ``k = 10 .. 25``. It also deposits the
particle positions independently (masked deposit plus image charge) as a cross-check of the
field's charge. ``qm_scale`` small makes the space-charge velocity change negligible, so the
axial density is uniform apart from the derived near-wall deficit; ``qm_scale = 1`` shows the
additional depressed-density rise.

Run (Cloud Batch; laptop smoke only with ``--smoke``)::

    python studies/p5b-dcdep/near_wall_charge.py --grid coarse --wall-stencil image \
        --cfl-div 1 --qm-scale 1e-3

Prints one ``NWQ {json}`` line per run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "examples"), os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _space_charge_waves_common as s  # noqa: E402
from wall_gather_model import deposit_node_profile  # noqa: E402

from wakis import GridFIT3D, SolverFIT3D  # noqa: E402
from wakis.particles import BeamInjector, ParticleSolver, Species  # noqa: E402
from wakis.particles.solver import compute_dual_divergence_d  # noqa: E402


def run(grid: str, wall_stencil: str, cfl_div: float, qm_scale: float, smoke: bool) -> dict:
    nx, _ny, nz = s.COARSE_GRID if grid == "coarse" else s.FINE_GRID
    params = s.compute_analytic_parameters(qm_scale=qm_scale)
    a, b = s.TUNNEL_HALF_WIDTH_M, s.BEAM_HALF_THICKNESS_M
    ly, lz = s.PERIODIC_LY_M, s.TUNNEL_LENGTH_M
    cfln = s.CFL_NUMBER / float(cfl_div)
    g = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, 1, nz, verbose=0)
    dt, _ = s._integer_period_dt(g, cfln=cfln)
    solver = SolverFIT3D(
        g,
        cfln=cfln,
        dt=dt,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=False,
        verbose=0,
    )
    n_macro = 16 if nx == s.COARSE_GRID[0] else 32
    dz = lz / nz
    dzeta = float(params["u0"]) * dt / dz
    n_steps = 40 if smoke else int(np.ceil(40.0 / dzeta))
    sp = Species(
        name="electron",
        charge=-s.e_charge,
        mass=s.m_e / qm_scale,
        capacity=(n_steps + 16) * n_macro,
    )
    i0 = float(params["I0"])
    inj = BeamInjector(
        species=sp,
        z0=0.0,
        current=lambda t: i0,
        gamma=float(params["gamma0"]),
        profile="sheet_lattice",
        radius=b,
        x0=0.0,
        y0=0.0,
        n_macro_per_step=n_macro,
        seed=42,
    )
    ps = ParticleSolver(
        solver,
        species=[sp],
        injectors=[inj],
        shape=int(s.PARTICLE_SHAPE),
        filter=s.PARTICLE_FILTER,
        gather="galerkin",
        B_ext=(0.0, 0.0, s.GUIDE_BZ_TESLA),
        wall_stencil=wall_stencil,
    ).attach()
    t0 = time.time()
    n_avg = 0 if smoke else max(1, int(round(1.0 / dzeta)))  # average over ~1 cell of transit
    q_acc = None
    for step in range(n_steps):
        solver.one_step()
        if step >= n_steps - max(n_avg, 1):
            div = np.asarray(compute_dual_divergence_d(solver))  # (Nx, Ny, Nz) [C]
            q_plane = div[1:-1].sum(axis=(0, 1))  # interior x nodes (x-wall nodes hold induced charge)
            q_acc = q_plane if q_acc is None else q_acc + q_plane
    q_field = q_acc / max(n_avg, 1)
    # independent deposit of the particle positions at the last step
    from wakis.particles.shapes import deposit_charge

    xi, eta, zeta = ps.grid_map.to_logical(sp.x, sp.y, sp.z)
    qn = np.zeros((solver.Nx, solver.Ny, solver.Nz), order="F")
    deposit_charge(
        xi, eta, zeta, sp.charge * sp.w, solver.Nx, solver.Ny, solver.Nz,
        order=ps.shape, periodic=ps.boundaries.periodic, q_nodes=qn, xp=np,
    )
    ps._image_charge(xi, eta, zeta, sp.charge * sp.w, qn)
    q_part = qn[1:-1].sum(axis=(0, 1))
    div_last = np.asarray(compute_dual_divergence_d(solver))[1:-1].sum(axis=(0, 1))
    if os.environ.get("NWQ_DEBUG"):
        print("DBG div", div_last[:8], "\nDBG part", q_part[:8], "\nDBG qbg", np.asarray(solver._particles_q_bg).sum(axis=(0, 1))[:8])
    ref = slice(10, 26)
    c_field = q_field / np.mean(q_field[ref])
    c_part = q_part / np.mean(q_part[ref])
    c_pred = deposit_node_profile(wall_stencil, dzeta, 8)
    rec = {
        "grid": grid,
        "wall_stencil": wall_stencil,
        "cfl_div": cfl_div,
        "qm_scale": qm_scale,
        "dzeta": dzeta,
        "n_steps": n_steps,
        "n_avg": n_avg,
        "c_field_k1_7": c_field[1:8].tolist(),
        "c_part_k1_7": c_part[1:8].tolist(),
        "c_pred_k1_7": c_pred[1:8].tolist(),
        "deficit1_field": float(1.0 - c_field[1]),
        "deficit1_pred": float(1.0 - c_pred[1]),
        "deficit1_pred_if_j_from_0": float(1.0 - c_pred[1] - (dzeta / 8.0 if wall_stencil == "masked" else 0.0)),
        "field_vs_part_last_max_rel_k1_25": float(
            np.max(np.abs(div_last[1:26] - q_part[1:26])) / np.max(np.abs(q_part[1:26]))
        ),
        "far_flatness_rel_std_k10_25": float(np.std(q_field[ref]) / abs(np.mean(q_field[ref]))),
        "wall_s": time.time() - t0,
    }
    print("NWQ " + json.dumps(rec), flush=True)
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", choices=("coarse", "fine"), default="coarse")
    p.add_argument("--wall-stencil", choices=("masked", "image"), default="image")
    p.add_argument("--cfl-div", type=float, nargs="+", default=[1.0])
    p.add_argument("--qm-scale", type=float, nargs="+", default=[1e-3])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    for cd in args.cfl_div:
        for qm in args.qm_scale:
            run(args.grid, args.wall_stencil, cd, qm, args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
