"""Entrance diagnostic for the shape >= 2 DC-depression excess (P5 part B, V7).

Hypothesis: a macroparticle injected at the PEC wall z0 = 0 with B-spline order p >= 2 has
stencil weight on node 1 (one cell inside the vacuum; 1/8 for p = 2 at zeta = 0, 1/6 for p = 3).
The field never sees the particle being created, so the visible birth deposit stays behind as
an immobile compensating charge; the part on node 1 is not on the wall and accumulates every
step, giving a time-growing potential step at the entrance (p = 1 puts all weight on node 0).

Records, for an unmodulated (m = 0) DC beam on the coarse V7 grid:
  * per z-plane k = 0..K-1 (summed over x, y): dual divergence S~d~ (field-side charge),
    deposited particle charge q_nodes and the bookkeeping background q_bg, vs step;
  * the bulk mean axial velocity (z in [0.3, 0.7] Lz) vs step -> delta(t);
  * the final axial mean-u_z profile in the first cells.
Falsifier: if sum_xy S~d~ at node 1 minus q_nodes at node 1 stays constant in time for p = 2,
the hypothesis is wrong.

Usage: python studies/p5b-mscan/entrance_diag.py --shape 2 --out studies/p5b-mscan/out6
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
import _space_charge_waves_common as C  # noqa: E402
from scipy.constants import c as c_light  # noqa: E402
from scipy.constants import e as e_charge  # noqa: E402
from scipy.constants import m_e  # noqa: E402

from wakis import GridFIT3D, SolverFIT3D  # noqa: E402
from wakis.particles import BeamInjector, ParticleSolver, Species  # noqa: E402
from wakis.particles.mapping import LogicalGridMap  # noqa: E402
from wakis.particles.shapes import deposit_charge  # noqa: E402
from wakis.particles.solver import compute_dual_divergence_d  # noqa: E402

K_PLANES = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, default=2)
    ap.add_argument("--transits", type=float, default=2.0, help="run length in Lz/u0 transits")
    ap.add_argument("--every", type=int, default=200, help="record every N steps")
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    os.makedirs(a_.out, exist_ok=True)

    params = C.compute_analytic_parameters()
    nx, _, nz = C.COARSE_GRID
    a, b, ly, lz = C.TUNNEL_HALF_WIDTH_M, C.BEAM_HALF_THICKNESS_M, C.PERIODIC_LY_M, C.TUNNEL_LENGTH_M
    gamma0, u0, i0 = params["gamma0"], params["u0"], params["I0"]
    grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, 1, nz, verbose=0)
    dt, _ = C._integer_period_dt(grid, cfln=C.CFL_NUMBER)
    solver = SolverFIT3D(grid, cfln=C.CFL_NUMBER, dt=dt, bc_low=["pec", "periodic", "pec"],
                         bc_high=["pec", "periodic", "pec"], use_gpu=False, verbose=0)
    n_total = int(np.ceil(a_.transits * lz / (u0 * dt)))
    sp = Species(name="electron", charge=-e_charge, mass=m_e, capacity=int((lz / (u0 * dt) + 64) * 16 * 1.3))
    t_rise = C.RISE_PERIODS / C.DRIVE_FREQ_HZ

    def cur(t):
        return i0 * 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))

    inj = BeamInjector(species=sp, z0=0.0, current=cur, gamma=gamma0, profile="sheet_lattice",
                       radius=b, x0=0.0, y0=0.0, n_macro_per_step=16, seed=C.NOISE_SEEDS[0])
    ParticleSolver(solver, species=[sp], injectors=[inj], shape=a_.shape, filter=None,
                   gather="galerkin", B_ext=(0.0, 0.0, C.GUIDE_BZ_TESLA)).attach()
    gmap = LogicalGridMap.from_grid(solver, xp=np)
    dz = lz / nz
    q_plane_bulk = -i0 / u0 * dz  # beam charge per z-plane of one cell [C] (sign: electrons)

    rec = dict(step=[], t=[], divd=[], qn=[], qbg=[], u_bulk=[])
    t0 = time.time()
    for step in range(n_total):
        solver.one_step()
        if (step + 1) % a_.every == 0 or step == n_total - 1:
            n = sp.n_particles
            xi, eta, zeta = gmap.to_logical(sp.x, sp.y, sp.z)
            qn = np.zeros((solver.Nx, solver.Ny, solver.Nz))
            deposit_charge(xi, eta, zeta, sp.charge * sp.w, solver.Nx, solver.Ny, solver.Nz,
                           order=a_.shape, periodic=(False, True, False), q_nodes=qn, xp=np)
            divd = compute_dual_divergence_d(solver)
            qbg = getattr(solver, "_particles_q_bg", np.zeros_like(qn))
            zp = np.asarray(sp.z[:n])
            gam = np.sqrt(1.0 + (np.asarray(sp.ux[:n]) ** 2 + np.asarray(sp.uy[:n]) ** 2
                                 + np.asarray(sp.uz[:n]) ** 2) / c_light**2)
            vz = np.asarray(sp.uz[:n]) / gam
            sel = (zp > 0.3 * lz) & (zp < 0.7 * lz)
            rec["step"].append(step + 1)
            rec["t"].append((step + 1) * dt)
            rec["divd"].append(divd[:, :, :K_PLANES].sum(axis=(0, 1)).tolist())
            rec["qn"].append(qn[:, :, :K_PLANES].sum(axis=(0, 1)).tolist())
            rec["qbg"].append(np.asarray(qbg)[:, :, :K_PLANES].sum(axis=(0, 1)).tolist())
            rec["u_bulk"].append(float(np.mean(vz[sel])) if np.any(sel) else float("nan"))
    zp = np.asarray(sp.z[: sp.n_particles])
    gam = np.sqrt(1.0 + (np.asarray(sp.ux[: sp.n_particles]) ** 2 + np.asarray(sp.uy[: sp.n_particles]) ** 2
                         + np.asarray(sp.uz[: sp.n_particles]) ** 2) / c_light**2)
    vz = np.asarray(sp.uz[: sp.n_particles]) / gam
    cell = np.floor(zp / dz).astype(int)
    u_prof = [float(np.mean(vz[cell == k])) / u0 for k in range(16)]
    out = dict(shape=a_.shape, dt=dt, dz=dz, n_total=n_total, u0=u0, q_plane_bulk=q_plane_bulk,
               delta_disc=float(C.compute_discrete_dc_depression(nx, params, shape=a_.shape)),
               u_prof_first16=u_prof, wall_s=time.time() - t0, **rec)
    path = os.path.join(a_.out, f"entrance_p{a_.shape}.json")
    json.dump(out, open(path, "w"), indent=1)
    divd, qn = np.array(rec["divd"]), np.array(rec["qn"])
    print("shape", a_.shape, "steps", n_total, "wall", round(out["wall_s"]), "s")
    print("final (S~d~ - q_nodes) per plane / q_plane_bulk:", np.round((divd[-1] - qn[-1]) / q_plane_bulk, 4))
    print("delta(t) last 5:", [round(1 - u / u0, 6) for u in rec["u_bulk"][-5:]], "delta_disc", out["delta_disc"])
    print("u/u0 first 16 cells:", np.round(u_prof, 4))


if __name__ == "__main__":
    main()
