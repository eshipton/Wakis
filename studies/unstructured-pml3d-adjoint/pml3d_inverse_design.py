"""
3D inverse design in an open domain: tetrahedral FIT + spherical CFS-PML + exact adjoint.

A z-directed point dipole radiates at f0 (vacuum wavelength LAM = 1 m). A
spherical design region (radius A_DES, centre at the origin, eps_r in
[1, EPS_MAX]) sits between the dipole (at x = -X_SRC) and a target point
(x = +X_TGT). Objective: maximise F = |E_z(target, f0)|^2 (running DFT over
the window). The domain is a tetrahedral ball truncated by the 3D spherical
CFS-PML; gradients come from the exact discrete adjoint
(`UnstructuredAdjointSimulation`, with the PML filter states in the reverse
sweep), checked against central finite differences on the full problem
before the optimisation starts.

Baselines (same mesh, same window): vacuum (rho = 0), uniform rho = 0.5
(the start) and the solid dielectric sphere (rho = 1, eps_r = EPS_MAX).

Usage
-----
    python pml3d_inverse_design.py run [n_iter]   # design.json, design.npz
    python pml3d_inverse_design.py plot           # docs/img/unstructured_pml3d_adjoint_*.png
"""

import json
import os
import sys
import time

import numpy as np
from scipy.constants import c as C0

HERE = os.path.dirname(os.path.abspath(__file__))
LAM = 1.0
F0 = C0 / LAM
H = LAM / 6  # 6 edges per vacuum wavelength, ~3 in eps_r = 4 (demo resolution)
R_BALL = 1.2
A_DES = 0.45
N_PML = 8
X_SRC, X_TGT = 0.85, 0.85
EPS_MAX = 4.0
W_PULSE = 3 / F0
T_PULSE = 3 * W_PULSE
T_END = T_PULSE + 3 * W_PULSE + 6 * LAM / C0


def amp(t):
    x = (t - T_PULSE) / W_PULSE
    return np.sin(2 * np.pi * F0 * (t - T_PULSE)) * np.exp(-(x**2))


def build():
    from wakis.unstructured import UnstructuredFIT3D, ball_mesh
    from wakis.unstructured.adjoint import CellDFTMonitor, CellMaterialGrid, UnstructuredAdjointSimulation
    from wakis.unstructured.pml3d import spherical_shell_mesh

    ball = ball_mesh([A_DES, R_BALL], H)
    radii = R_BALL + H * np.arange(1, N_PML + 1)
    mesh = spherical_shell_mesh(ball, radii)
    pml = dict(r0=R_BALL, r1=radii[-1], kappa_max=2.0, alpha=2 * np.pi * F0 / 10)
    dt = UnstructuredFIT3D(mesh, bc="pec", pml=pml).dt  # eps_min = 1 everywhere: stable for all rho

    def make():
        s = UnstructuredFIT3D(mesh, bc="pec", pml=pml, dt=dt)
        s.add_point_dipole([-X_SRC, 0.013, 0.007], [0, 0, 1], amp)
        return s

    cent = mesh.nodes[mesh.cells].mean(1)
    cells = np.flatnonzero(np.linalg.norm(cent, axis=1) < A_DES)
    design = CellMaterialGrid(mesh, cells, 1.0, EPS_MAX, filter_radius=1.5 * H)
    mon = CellDFTMonitor([F0], points=[[X_TGT, 0.0, 0.0]], component=2)
    Nt = int(np.ceil(T_END / dt))
    sim = UnstructuredAdjointSimulation(make, design, {"m": mon}, Nt)
    return mesh, sim, dt, Nt


def objective(out):
    X = out["m"]["dft"]
    return float(np.sum(np.abs(X) ** 2)), {"m": {"dft": 2 * X}}


def field_map(mesh, sim, rho, n=81):
    """E_z DFT at f0 on the plane z = 0 (|x|, |y| < R_BALL) and the eps map."""
    from wakis.unstructured.adjoint import CellDFTMonitor, UnstructuredAdjointSimulation

    g = np.linspace(-R_BALL * 0.98, R_BALL * 0.98, n)
    X, Y = np.meshgrid(g, g, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel(), 1e-4 + 0 * X.ravel()], 1)
    inside = np.linalg.norm(pts, axis=1) < R_BALL * 0.97
    mon = CellDFTMonitor([F0], points=pts[inside], component=2)
    s2 = UnstructuredAdjointSimulation(sim.make_solver, sim.design, {"map": mon}, sim.Nt)
    out = s2.forward(rho, record=False)["map"]["dft"][0]
    E = np.full(pts.shape[0], np.nan, complex)
    E[inside] = out
    cell, _ = mesh.locate(pts[inside])
    eps_c = sim.design.cell_eps(rho, sim.stepper.base_eps)
    epsm = np.full(pts.shape[0], np.nan)
    epsm[inside] = eps_c[cell]
    return g, E.reshape(n, n), epsm.reshape(n, n)


def run(n_iter=30):
    t0 = time.time()
    mesh, sim, dt, Nt = build()
    nd = sim.design.shape[0]
    print(f"mesh {mesh.n_cells} cells, {sim.solver.free.size} free edges, PML state "
          f"{sim.solver._pml.layout.n}, design {nd} cells, dt {dt:.3e} s, Nt {Nt}", flush=True)  # fmt: skip
    base = {}
    for name, val in (("vacuum", 0.0), ("uniform 0.5", 0.5), ("solid sphere", 1.0)):
        base[name] = objective(sim.forward(np.full(nd, val), record=False))[0]
        print("baseline", name, base[name], flush=True)
    rho = np.full(nd, 0.5)
    # gradient check on the full problem (central differences, h = 1e-3 and 2e-3)
    F, g = sim.value_and_grad(rho, objective)
    rng = np.random.default_rng(0)
    checks = []
    for k in rng.choice(nd, 3, replace=False):
        fd = []
        for hh in (1e-3, 2e-3):
            rp, rm = rho.copy(), rho.copy()
            rp[k] += hh
            rm[k] -= hh
            fd.append((objective(sim.forward(rp, record=False))[0]
                       - objective(sim.forward(rm, record=False))[0]) / (2 * hh))  # fmt: skip
        checks.append(dict(cell=int(k), adjoint=float(g[k]), fd=fd[0], trunc=abs(fd[1] - fd[0]) / 3))
        print("FD check", checks[-1], "rel", abs(g[k] - fd[0]) / abs(fd[0]), flush=True)
    # projected gradient ascent with Adam moments (design in [0, 1])
    hist, m1, m2 = [], np.zeros(nd), np.zeros(nd)
    lr, b1, b2 = 0.05, 0.9, 0.999
    rho_hist = [rho.copy()]
    for it in range(n_iter):
        if it:
            F, g = sim.value_and_grad(rho, objective)
        hist.append(F)
        gs = g / np.abs(g).max()
        m1 = b1 * m1 + (1 - b1) * gs
        m2 = b2 * m2 + (1 - b2) * gs**2
        step = lr * (m1 / (1 - b1 ** (it + 1))) / (np.sqrt(m2 / (1 - b2 ** (it + 1))) + 1e-8)
        rho = np.clip(rho + step, 0, 1)
        rho_hist.append(rho.copy())
        print(f"iter {it:3d}  F/F_vacuum = {F / base['vacuum']:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    F_final = objective(sim.forward(rho, record=False))[0]
    hist.append(F_final)
    maps = {}
    for name, r in (("start", np.full(nd, 0.5)), ("final", rho), ("vacuum", np.zeros(nd))):
        gx, E, epsm = field_map(mesh, sim, r)
        maps[name] = (E, epsm)
    out = dict(lam=LAM, h=H, R_ball=R_BALL, a_design=A_DES, n_pml=N_PML, eps_max=EPS_MAX,
               n_cells=int(mesh.n_cells), n_design=int(nd), dt=dt, Nt=Nt, baselines=base,
               history=hist, fd_checks=checks, wall_s=time.time() - t0)  # fmt: skip
    with open(os.path.join(HERE, "design.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(HERE, "design.npz"), grid=gx, rho=np.array(rho_hist),
                        **{f"E_{k}": v[0] for k, v in maps.items()},
                        **{f"eps_{k}": v[1] for k, v in maps.items()})  # fmt: skip
    print("done", out["history"][-1] / base["vacuum"], "wall", out["wall_s"])


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = os.path.join(HERE, "..", "..", "docs", "img")
    S = json.load(open(os.path.join(HERE, "design.json")))
    Z = np.load(os.path.join(HERE, "design.npz"))
    b = S["baselines"]
    Fv = b["vacuum"]
    hist = np.array(S["history"]) / Fv
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(np.arange(hist.size), hist, "o-", ms=3, label="adjoint design (projected Adam)")
    for (name, v), ls in zip(b.items(), (":", "--", "-.")):
        ax.axhline(v / Fv, color="0.4", ls=ls, lw=1, label=f"{name}: {v / Fv:.2f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$|E_z(\mathbf{x}_t, f_0)|^2$ / vacuum value")
    ax.set_title("Open-domain 3D design: field at a point behind a dielectric region")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(img, "unstructured_pml3d_adjoint_history.png"), dpi=150)
    plt.close(fig)

    g = Z["grid"]
    ext = [g[0], g[-1], g[0], g[-1]]
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 8.2), constrained_layout=True)
    vmax = np.nanmax(np.abs(Z["E_final"]))
    for col, key in enumerate(("start", "final")):
        im = axs[0, col].imshow(Z[f"eps_{key}"], origin="lower", extent=ext, cmap="viridis", vmin=1, vmax=S["eps_max"])
        axs[0, col].set_title(f"{key} design: $\\varepsilon_r$ on z = 0")
        fig.colorbar(im, ax=axs[0, col], shrink=0.8, label=r"$\varepsilon_r$")
        im = axs[1, col].imshow(np.real(Z[f"E_{key}"]), origin="lower", extent=ext, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax)  # fmt: skip
        axs[1, col].set_title(f"{key}: Re $\\hat E_z(f_0)$ on z = 0")
        fig.colorbar(im, ax=axs[1, col], shrink=0.8, label=r"Re $\hat E_z$ [V s/m]")
        for ax in axs[:, col]:
            ax.plot([-X_SRC], [0], "k*", ms=10)
            ax.plot([X_TGT], [0], "wo", mec="k", ms=7)
            ax.add_patch(plt.Circle((0, 0), A_DES, fill=False, ls="--", color="w"))
            ax.add_patch(plt.Circle((0, 0), R_BALL, fill=False, color="k", lw=0.8))
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
    fig.suptitle(f"Dipole (star) -> target (circle); domain truncated by a {S['n_pml']}-layer spherical PML at r = {S['R_ball']} m")
    fig.savefig(os.path.join(img, "unstructured_pml3d_adjoint_fields.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    else:
        plot()
