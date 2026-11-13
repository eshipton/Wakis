"""
Inverse design on a graded triangle mesh: focus a line source into the
shadow of a PEC cylinder with a dielectric annulus (2D TM, exact adjoint).

Geometry (vacuum wavelength lambda0 = 1 m, f0 = c / lambda0):

* PEC cylinder r < a = 0.5 lambda0 (its boundary is a conforming ring of
  triangle edges: no staircase);
* design annulus a < r < b = 1.25 lambda0, eps_r in [1, 4] per triangle,
  cone filter (radius 0.08 lambda0) and tanh projection with beta
  continuation 1, 4, 8, 16, 32;
* vacuum to R = 2.5 lambda0, then a radial CFS-PML of 8 polar-aligned
  rings (dr = 0.06 lambda0) backed by PEC;
* source: z line current at (-2, 0) lambda0, Gaussian-modulated sine at f0;
* objective: J = |Ez(f0)|^2 at the focus (+1.75, 0) lambda0 in the shadow,
  normalised by the bare cylinder (design annulus vacuum, same mesh, same
  source, same steps). This is the only baseline; no Cartesian run is made.

Mesh: 20 cells per lambda in eps = 4 (h = lambda0 / 40) inside the design
annulus and around the cylinder, graded (x1.2 per ring) to h = lambda0 / 20
in the outer vacuum and the PML.

Outputs (this folder): results.json and figures/*.png. Run on Cloud Batch:

    bin/batch-run uadj-demo WT studies/unstructured-adjoint -- \
        python studies/unstructured-adjoint/shadow_focus.py

``--quick``: coarse mesh and 2 iterations per stage (local smoke test).
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.constants import c
from scipy.optimize import minimize

from wakis.unstructured import UnstructuredFIT2D, graded_disk_mesh
from wakis.unstructured.adjoint import (
    CellDFTMonitor,
    CellMaterialGrid,
    UnstructuredAdjointSimulation,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LAM = 1.0
F0 = c / LAM
A_PEC, B_DES, R_OUT = 0.5, 1.25, 2.5
N_PML, DR_PML = 8, 0.06
SRC = np.array([-2.0, 0.0])
FOCUS = np.array([1.75, 0.0])
EPS_MAX = 4.0
BETAS = (1.0, 4.0, 8.0, 16.0, 32.0)


def build_mesh(cpw):
    """cpw: cells per wavelength in the eps = 4 design material."""
    h_f = LAM / (np.sqrt(EPS_MAX) * cpw)
    h_c = 2 * h_f

    def size(r):
        d = r - (B_DES + 0.1)
        return h_f if d <= 0 else min(h_c, h_f * 1.2 ** (d / h_f))

    mesh = graded_disk_mesh([A_PEC, B_DES, R_OUT], size, pml_layers=N_PML, pml_dr=DR_PML)
    return mesh, h_f, h_c


def source(t, tau=4 / F0):
    t0 = 3 * tau
    return np.exp(-(((t - t0) / tau) ** 2)) * np.sin(2 * np.pi * F0 * (t - t0))


def make_factory(mesh, dt=None, pec=True):
    def make():
        s = UnstructuredFIT2D(
            mesh,
            {0: "pec" if pec else [1.0, 1.0], 1: [1.0, 1.0], 2: [1.0, 1.0], 3: [1.0, 1.0]},
            polarization="TM",
            bc="pec",
            dt=dt,
            pml=dict(r0=R_OUT, r1=R_OUT + N_PML * DR_PML, R=1e-6),
        )
        s.add_point_source(SRC, source)
        return s

    return make


def setup(cpw, periods):
    mesh, h_f, h_c = build_mesh(cpw)
    dt = make_factory(mesh)().dt  # frozen for all designs (built at eps_min)
    make = make_factory(mesh, dt)
    Nt = int(np.ceil(periods / F0 / dt))
    cells = np.flatnonzero(mesh.cell_tags == 1)
    return mesh, make, dt, Nt, cells, h_f, h_c


# ------------------------------------------------------------ FD workers
_G = {}


def _fd_init(cpw, periods, radius, beta):
    mesh, make, dt, Nt, cells, _, _ = setup(cpw, periods)
    d = CellMaterialGrid(mesh, cells, 1.0, EPS_MAX, filter_radius=radius, beta=beta)
    _G["sim"] = UnstructuredAdjointSimulation(make, d, {"f": CellDFTMonitor(F0, [FOCUS])}, Nt)


def _fd_eval(rho):
    out = _G["sim"].forward(rho, record=False)
    return float(np.abs(out["f"]["dft"][0, 0]) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    cpw, periods, iters = (8, 30, 2) if args.quick else (20, 60, 12)
    radius = 0.08 * LAM
    t_start = time.time()
    mesh, make, dt, Nt, cells, h_f, h_c = setup(cpw, periods)
    print(f"mesh: {mesh.n_nodes} nodes, {mesh.n_cells} cells, {cells.size} design cells; "
          f"h {h_f:.4f}..{h_c:.4f} m, dt {dt:.3e} s, Nt {Nt}", flush=True)  # fmt: skip
    mon = {"f": CellDFTMonitor(F0, [FOCUS])}
    design = CellMaterialGrid(mesh, cells, 1.0, EPS_MAX, filter_radius=radius, beta=BETAS[0])
    sim = UnstructuredAdjointSimulation(make, design, mon, Nt)

    # Baseline: bare PEC cylinder (design annulus at eps = 1)
    X0 = sim.forward(np.zeros(design.shape), record=False)["f"]["dft"][0, 0]
    J0 = float(abs(X0) ** 2)

    def J_and_G(rho):
        X = sim.forward(rho)["f"]["dft"]
        J = float(np.abs(X[0, 0]) ** 2) / J0
        g = sim.vjp({"f": {"dft": 2 * X / J0}})
        return J, g

    # ---- gradient check at the start design (adjoint vs central FD, parallel)
    rho0 = np.full(design.shape, 0.5)
    J_start, g_start = J_and_G(rho0)
    rng = np.random.default_rng(0)
    order = np.argsort(-np.abs(g_start))
    probe = np.r_[order[:6], rng.choice(order[6:], 6, replace=False)]
    h = 1e-3
    jobs = []
    for k in probe:
        for hh in (h, 2 * h):
            for sgn in (1, -1):
                r = rho0.copy()
                r[k] += sgn * hh
                jobs.append(r)
    with Pool(min(args.procs, len(jobs)), _fd_init, (cpw, periods, radius, BETAS[0])) as pool:
        vals = np.array(pool.map(_fd_eval, jobs)).reshape(len(probe), 2, 2) / J0
    fd_h = (vals[:, 0, 0] - vals[:, 0, 1]) / (2 * h)
    fd_2h = (vals[:, 1, 0] - vals[:, 1, 1]) / (4 * h)
    fd_rich = (4 * fd_h - fd_2h) / 3
    gradcheck = dict(
        cells=probe.tolist(), adjoint=g_start[probe].tolist(), fd_h=fd_h.tolist(),
        fd_2h=fd_2h.tolist(), fd_richardson=fd_rich.tolist(), h=h,
        rel_err_fd=(np.abs(g_start[probe] - fd_h) / np.abs(fd_h)).tolist(),
        rel_err_richardson=(np.abs(g_start[probe] - fd_rich) / np.abs(fd_rich)).tolist(),
        trunc_estimate=(np.abs(fd_2h - fd_h) / 3 / np.abs(fd_h)).tolist(),
    )  # fmt: skip
    print("gradient check: max rel err vs FD %.2e (truncation est %.2e), vs Richardson %.2e"
          % (max(gradcheck["rel_err_fd"]), max(gradcheck["trunc_estimate"]),
             max(gradcheck["rel_err_richardson"])), flush=True)  # fmt: skip
    g_map0 = g_start.copy()

    # ---- optimization: L-BFGS-B on -log J per projection stage
    history, snapshots = [], {}
    rho = rho0.copy()
    for stage, beta in enumerate(BETAS):
        design.beta = beta

        def fun(r):
            J, g = J_and_G(r)
            history.append(dict(J=J, beta=beta, stage=stage))
            print(f"  beta {beta:5.1f}  it {len(history):3d}  J/J0 = {J:.4f}", flush=True)
            return -np.log(J), -g / J

        res = minimize(fun, rho, jac=True, method="L-BFGS-B", bounds=[(0, 1)] * rho.size,
                       options=dict(maxiter=iters, maxfun=iters + 5))  # fmt: skip
        rho = res.x
        snapshots[beta] = design.eps_r(rho).tolist()
    # ---- binarized final design (threshold projection)
    design.beta = np.inf
    eps_bin = design.eps_r(rho)
    X_bin = sim.forward(rho, record=False)["f"]["dft"][0, 0]
    J_bin = float(abs(X_bin) ** 2) / J0
    design.beta = BETAS[-1]
    J_final, g_final = J_and_G(rho)
    # checkpointed gradient of the final design: bitwise equal to the stored one
    ck = UnstructuredAdjointSimulation(make, design, mon, Nt, memory="checkpoint", checkpoints=4)
    ck.solver, ck.stepper = sim.solver, sim.stepper
    X = ck.forward(rho)["f"]["dft"]
    g_ck = ck.vjp({"f": {"dft": 2 * X / J0}})
    ckpt = dict(bitwise_equal=bool(np.array_equal(g_ck, g_final)), stats=ck.memory_stats,
                store_bytes=int(Nt * sim.stepper.dofs.size * 8))  # fmt: skip
    print("checkpoint gradient bitwise equal:", ckpt["bitwise_equal"], ck.memory_stats)

    # ---- guards: DFT window converged (2 Nt) and free-space scale (no cylinder)
    long = UnstructuredAdjointSimulation(make, design, mon, 2 * Nt)
    long.solver, long.stepper = sim.solver, sim.stepper
    J0_long = float(np.abs(long.forward(np.zeros(design.shape), record=False)["f"]["dft"][0, 0]) ** 2)
    design.beta = np.inf
    J_bin_long = float(np.abs(long.forward(rho, record=False)["f"]["dft"][0, 0]) ** 2) / J0_long
    design.beta = BETAS[-1]
    free = UnstructuredAdjointSimulation(make_factory(mesh, pec=False), design, mon, 2 * Nt)
    J_free = float(np.abs(free.forward(np.zeros(design.shape), record=False)["f"]["dft"][0, 0]) ** 2)
    guards = dict(J0_rel_change_2Nt=J0_long / J0 - 1, J_bin_2Nt=J_bin_long,
                  J_bin_rel_change_2Nt=J_bin_long / J_bin - 1,
                  shadow_depth_J0_over_Jfree=J0_long / J_free,
                  J_bin_over_Jfree=J_bin_long * J0_long / J_free)  # fmt: skip
    print("guards:", guards, flush=True)

    # ---- full-field DFT maps: bare cylinder and binarized design
    fields = {}
    full = {"all": CellDFTMonitor(F0)}
    for name, eps_cells in (("bare", np.ones(cells.size)), ("design", eps_bin)):
        bare = CellMaterialGrid(mesh, cells, 0.0, 1.0)  # eps = rho directly
        fs = UnstructuredAdjointSimulation(make, bare, full, Nt)
        Xf = fs.forward(eps_cells, record=False)["all"]["dft"][0]
        u = np.zeros(mesh.n_nodes, complex)
        u[fs.solver.free] = Xf
        fields[name] = u
    np.savez_compressed(os.path.join(HERE, "fields.npz"), nodes=mesh.nodes, cells=mesh.cells,
                        tags=mesh.cell_tags, design_cells=cells, bare=fields["bare"],
                        design=fields["design"], eps_bin=eps_bin, g_start=g_map0,
                        snap_betas=np.array(list(snapshots)),
                        snaps=np.array(list(snapshots.values())))  # fmt: skip
    results = dict(
        mesh=dict(n_nodes=mesh.n_nodes, n_cells=mesh.n_cells, n_design_cells=int(cells.size),
                  h_fine=h_f, h_coarse=h_c, cpw_eps4=cpw),
        dt=dt, Nt=Nt, periods=periods, f0=F0, filter_radius=radius, betas=list(BETAS),
        J0_abs=J0, J_start=J_start, J_final=J_final, J_binarized=J_bin,
        history=history, gradcheck=gradcheck, checkpoint=ckpt, guards=guards,
        wall_s=time.time() - t_start, quick=args.quick,
    )  # fmt: skip
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"J/J0: start {J_start:.3f}, final {J_final:.3f}, binarized {J_bin:.3f}; "
          f"{time.time() - t_start:.0f} s")  # fmt: skip


if __name__ == "__main__":
    main()
