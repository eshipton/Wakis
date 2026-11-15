"""
WR-90 waveguide matching-section inverse design with tetrahedral ports (`wakis.unstructured.ports3d` + `adjoint`).

Setup
-----
* Waveguide: WR-90 (a = 22.86 mm, b = 10.16 mm), h = a/4, extruded along z
  with dz = 2.5 mm over 8 layers (L = 20 mm, 1056 tetrahedra, 96 massless
  diagonal edges).
* Ports: TE10 `ModalPort` at z = 0 (excited at f0 = 10 GHz, BW = 1.5 GHz) and
  at z = L (passive load g = Y_h(f0) / cos(pi f0 dt)).
* Dielectric obstacle: layers 4..6 (10.0 < z < 17.5 mm) have eps_r = 2.6.
* Design region: layers 1..3 (2.5 < z < 10.0 mm, 144 interior tetrahedra) with
  eps_T(rho) in [1.0, 2.6] via `CellMaterialGrid`.
* Objective: maximise J(rho) = |S21(f0)|^2 - |S11(f0)|^2 at f0 = 10 GHz using
  `PortDFTMonitor` and `UnstructuredAdjointSimulation`.

Usage
-----
    python studies/unstructured-ports3d/adjoint_wr90_match.py [out_json] [fig_dir]

Default `out_json` is `out/adjoint_wr90_match.json`; figures are written to
`studies/unstructured-ports3d/figures/`.
"""

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wakis import figstyle  # noqa: E402
from wakis.unstructured.adjoint import (  # noqa: E402
    CellMaterialGrid,
    UnstructuredAdjointSimulation,
    UnstructuredAdjointStepper,
)
from wakis.unstructured.mesh import SimplexMesh  # noqa: E402
from wakis.unstructured.pml3d import _snap_degenerate_duals  # noqa: E402
from wakis.unstructured.ports3d import (  # noqa: E402
    ExtrudedLine,
    ModalPort,
    PortAdjointHelper,
    PortDFTMonitor,
    PortPlane,
    attach_ports,
    extrude_section,
    rectangle_section,
)
from wakis.unstructured.solver3d import UnstructuredFIT3D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
A_WG, B_WG = 22.86e-3, 10.16e-3
F0 = 10.0e9
BW = 1.5e9
F_BAND = np.linspace(8.0e9, 12.0e9, 21)
EPS_LOAD = 2.6
DZ = 2.5e-3
NZ = 8
NT = 320


def pulse(f0=F0, bw=BW):
    sig = 1.0 / (2.0 * np.pi * bw)
    t0 = 4.5 * sig
    return lambda t: (
        np.exp(-(((t - t0) / sig) ** 2) / 2.0) * np.sin(2.0 * np.pi * f0 * (t - t0))
    )


def build_problem():
    h = A_WG / 4
    nodes, tris, tags = rectangle_section(A_WG, B_WG, h, wall=h / 2, seed=1)
    z = np.linspace(0.0, NZ * DZ, NZ + 1)
    m = extrude_section(nodes, tris, tags, z)
    zc = m.nodes[m.cells].mean(1)[:, 2]
    tg = m.cell_tags.copy()
    # Obstacle in layers 4..6 (10.0 < z < 17.5 mm)
    tg[(zc > z[4]) & (zc < z[7]) & (tg == 1)] = 2
    m = SimplexMesh(m.nodes, m.cells, tg)
    _snap_degenerate_duals(m)
    mats = {0: "pec", 2: [EPS_LOAD, 1.0]}
    eps_layers_base = np.ones(NZ)
    eps_layers_base[4:7] = EPS_LOAD

    s0 = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6)
    dt = s0.dt
    planes0 = [PortPlane(s0, (0, 0, zz), (0, 0, 1)) for zz in (z[0], z[-1])]
    kc, phi0 = planes0[0].te_modes(1)
    phi1 = planes0[1].te_modes(1)[1][:, 0]
    phis = [phi0[:, 0], phi1]
    kc = float(kc[0])
    line0 = ExtrudedLine(z, eps_layers_base, 1.0, kc, dt)

    def Y(f):
        return np.real(line0.port_admittance(f) / line0.cos_half(f))

    def make():
        s = UnstructuredFIT3D(m, mats, bc="pmc", massless_tol=1e-6, dt=dt)
        planes = [PortPlane(s, (0, 0, zz), (0, 0, 1)) for zz in (z[0], z[-1])]
        ports = [
            ModalPort(planes[0], phis[0], Y, f0=F0, i_src=pulse(), name="p1"),
            ModalPort(planes[1], phis[1], Y, f0=F0, name="p2"),
        ]
        attach_ports(s, ports)
        return s

    # Matching section: layers 1..3 (2.5 < z < 10.0 mm)
    design_cells = np.flatnonzero((zc > z[1]) & (zc < z[4]) & (m.cell_tags == 1))
    design = CellMaterialGrid(m, design_cells, 1.0, EPS_LOAD)
    return m, make, design, z, dt, kc, eps_layers_base


def dot_product_checks(st):
    s = st.solver
    cap = s.cap
    mnu = np.where(s.Mnu_diag > 0, s.Mnu_diag, 1.0)

    def multistep(n_steps, seed):
        rng = np.random.default_rng(seed)
        x = {
            "u": rng.standard_normal(cap.size) / np.sqrt(cap),
            "b": rng.standard_normal(mnu.size) / np.sqrt(mnu),
        }
        y = {
            "u": rng.standard_normal(cap.size) * np.sqrt(cap),
            "b": rng.standard_normal(mnu.size) * np.sqrt(mnu),
        }
        Lx = {k: v.copy() for k, v in x.items()}
        for _ in range(n_steps):
            Lx = st.linear_step(Lx)
        LTy = {k: v.copy() for k, v in y.items()}
        for _ in range(n_steps):
            LTy = st.linear_step_T(LTy)
        lhs, rhs = st.inner(y, Lx), st.inner(LTy, x)
        return float(abs(lhs - rhs) / (abs(lhs) + abs(rhs)))

    out = {f"multistep_k{k}": multistep(k, 5 * k) for k in (1, 5, 10)}
    orig_pre, orig_post = PortAdjointHelper.pre_step_T, PortAdjointHelper.post_step_T
    try:
        PortAdjointHelper.pre_step_T = lambda self, u1b: u1b
        PortAdjointHelper.post_step_T = lambda self, rb: np.zeros_like(rb)
        out["mutation_drop_port_T_k1"] = multistep(1, 1)
        PortAdjointHelper.pre_step_T = lambda self, u1b: (
            u1b
            + self.W @ (0.5 * self.g * (self._Sinv.T @ (self.W.T @ (self.dinv * u1b))))
        )
        PortAdjointHelper.post_step_T = lambda self, rb: (
            self.W @ (0.5 * self.g * (self.W.T @ rb))
        )
        out["mutation_flip_port_T_k1"] = multistep(1, 1)
    finally:
        PortAdjointHelper.pre_step_T = orig_pre
        PortAdjointHelper.post_step_T = orig_post
    return out


def make_Figures(res, fig_dir):
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    os.makedirs(fig_dir, exist_ok=True)
    with figstyle.apply_style():
        # Figure 1: Gradient vs FD & Optimization History
        fig, ax = figstyle.new_figure(7.0, 3.4, ncols=2)
        fds = res["fd_checks"]
        idx = np.arange(len(fds))
        err_adj = [abs(r["adjoint"] - r["fd_h"]) for r in fds]
        trunc = [r["trunc_est"] for r in fds]
        bound = [1.5 * r["trunc_est"] + r["roundoff"] for r in fds]
        err_drop = [abs(r["mut_drop_port_T"] - r["fd_h"]) for r in fds]
        err_half = [abs(r["mut_no_half_step"] - r["fd_h"]) for r in fds]

        s0 = figstyle.series_style(0)
        s1 = figstyle.series_style(1)
        s2 = figstyle.series_style(2)
        s3 = figstyle.series_style(3)
        ax[0].semilogy(
            idx,
            err_adj,
            marker=s0["marker"],
            color=s0["color"],
            ls="-",
            label=r"$|g_{\mathrm{adj}} - \mathrm{FD}(h)|$",
        )
        ax[0].semilogy(
            idx,
            trunc,
            marker=s1["marker"],
            color=s1["color"],
            ls="--",
            label=r"$\tau(h) = |\mathrm{FD}(2h) - \mathrm{FD}(h)|/3$",
        )
        ax[0].semilogy(
            idx,
            bound,
            color="k",
            ls=":",
            lw=1.1,
            label=r"$1.5\,\tau(h) + r(h)$ bound",
        )
        ax[0].semilogy(
            idx,
            err_drop,
            marker=s2["marker"],
            color=s2["color"],
            ls="-.",
            label="mutation: drop port $L^T$",
        )
        ax[0].semilogy(
            idx,
            err_half,
            marker=s3["marker"],
            color=s3["color"],
            ls=":",
            label=r"mutation: $t_{n+1}$ monitor",
        )
        ax[0].set_xticks(idx)
        ax[0].set_xticklabels([f"T{r['cell']}" for r in fds])
        ax[0].set(
            xlabel="sampled design tetrahedron",
            ylabel=r"gradient error $|g - \mathrm{FD}(h)|$ (1/$\rho$)",
            title="(a) Adjoint vs central FD ($h = 10^{-3}$)",
        )
        figstyle.spectrum_grid(ax[0])
        ax[0].legend(
            fontsize=6.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
        )

        hist = res["history"]
        it = np.arange(len(hist))
        s21_sq = [h["s21_power_f0"] for h in hist]
        s11_db = [h["s11_db_f0"] for h in hist]
        ax[1].plot(
            it,
            s21_sq,
            marker=s0["marker"],
            color=s0["color"],
            ls="-",
            label=r"$|S_{21}(f_0)|^2$",
        )
        ax1b = ax[1].twinx()
        ax1b.plot(
            it,
            s11_db,
            marker=s1["marker"],
            color=s1["color"],
            ls="--",
            label=r"$|S_{11}(f_0)|$ (dB)",
        )
        ax1b.set_ylabel(r"$|S_{11}(f_0)|$ (dB)", color=s1["color"])
        ax[1].set(
            xlabel="optimisation iteration",
            ylabel=r"transmission $|S_{21}(f_0)|^2$",
            ylim=(0.5, 1.03),
            title=r"(b) WR-90 match at $f_0 = 10$ GHz",
        )
        figstyle.spectrum_grid(ax[1])
        lines1, labels1 = ax[1].get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax[1].legend(
            lines1 + lines2,
            labels1 + labels2,
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
        )
        p1 = os.path.join(fig_dir, "adjoint_wr90_match_gradient_and_history.png")
        figstyle.save(fig, p1, close=True, auto_legend_below=False)

        # Figure 2: Broadband X-band spectra and axial permittivity profile
        fig, ax = figstyle.new_figure(7.0, 3.4, ncols=2)
        fg = np.array(res["freqs_hz"]) / 1e9
        ax[0].plot(
            fg,
            res["initial_s11_db"],
            color=s2["color"],
            ls="--",
            marker="none",
            label="unmatched $|S_{11}|$ (3D FIT)",
        )
        ax[0].plot(
            fg,
            res["optimized_s11_db"],
            color=s0["color"],
            ls="-",
            marker=s0["marker"],
            ms=3.5,
            markevery=2,
            label="optimised $|S_{11}|$ (3D FIT)",
        )
        ax[0].plot(
            fg,
            res["optimized_ladder_s11_db"],
            color="k",
            ls=":",
            marker="none",
            lw=1.2,
            label="optimised $|S_{11}|$ (ladder)",
        )
        ax[0].axvline(F0 / 1e9, color="#6b7280", ls=":", lw=0.9)
        ax[0].set(
            xlabel="frequency (GHz)",
            ylabel=r"reflection $|S_{11}|$ (dB)",
            title="(a) X-band reflection before/after match",
        )
        figstyle.spectrum_grid(ax[0])
        ax[0].legend(
            fontsize=6.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
        )

        z_mm = np.array(res["z_levels_m"]) * 1e3
        eps_init = np.array(res["eps_layers_initial"])
        eps_opt = np.array(res["eps_layers_optimized"])
        z_step = np.repeat(z_mm, 2)[1:-1]
        ax[1].plot(
            z_step,
            np.repeat(eps_init, 2),
            color=s2["color"],
            ls="--",
            marker="none",
            label=r"initial $\varepsilon_r(z)$",
        )
        ax[1].plot(
            z_step,
            np.repeat(eps_opt, 2),
            color=s0["color"],
            ls="-",
            marker="none",
            lw=2.0,
            label=r"optimised layer mean $\varepsilon_r(z)$",
        )
        zc_mm = np.array(res["design_cell_z_m"]) * 1e3
        ax[1].plot(
            zc_mm,
            res["design_cell_eps_final"],
            ls="none",
            marker="o",
            ms=2.8,
            color=s1["color"],
            alpha=0.7,
            label=r"tet cells $\varepsilon_T$",
        )
        ax[1].axvspan(
            z_mm[1], z_mm[4], color="#9ec5e8", alpha=0.25, label="design section"
        )
        ax[1].set(
            xlabel="axial position z (mm)",
            ylabel=r"relative permittivity $\varepsilon_r$",
            ylim=(0.85, EPS_LOAD + 0.25),
            title=r"(b) Axial permittivity profile $\varepsilon_r(z)$",
        )
        figstyle.spectrum_grid(ax[1])
        ax[1].legend(
            fontsize=6.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
        )
        p2 = os.path.join(fig_dir, "adjoint_wr90_match_spectra_and_profile.png")
        figstyle.save(fig, p2, close=True, auto_legend_below=False)
    return p1, p2


def main():
    t0 = time.time()
    out_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "..", "..", "out", "adjoint_wr90_match.json")
    )
    fig_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures")

    m, make, design, z, dt, kc, eps_layers_base = build_problem()
    mon_f0 = PortDFTMonitor([F0])
    sim = UnstructuredAdjointSimulation(make, design, {"p": mon_f0}, NT)
    st = sim.stepper
    nd = design.shape[0]
    zc_des = m.nodes[m.cells[design.cells]].mean(1)[:, 2]
    layer_masks = [(zc_des > z[l]) & (zc_des < z[l + 1]) for l in (1, 2, 3)]

    def obj_f0(out):
        V = out["p"]["dft"]
        S, vjp_s = mon_f0.s_column(V, sim.solver, NT, excite=0)
        val = float(np.abs(S[0, 1]) ** 2 - np.abs(S[0, 0]) ** 2)
        G_S = np.array([[-2.0 * S[0, 0], 2.0 * S[0, 1]]])
        return val, {"p": {"dft": vjp_s(G_S)}}

    def metrics_f0(rho):
        V = sim.forward(rho, record=False)["p"]["dft"]
        S, _ = mon_f0.s_column(V, sim.solver, NT, excite=0)
        s11 = float(np.abs(S[0, 0]))
        s21_sq = float(np.abs(S[0, 1]) ** 2)
        eps_c = design.eps_r(rho)
        return dict(
            objective=float(s21_sq - s11**2),
            s11_mag_f0=s11,
            s11_db_f0=float(20.0 * np.log10(max(s11, 1e-15))),
            s21_power_f0=s21_sq,
            layer_eps_mean=[float(eps_c[msk].mean()) for msk in layer_masks],
        )

    # 1. Dot-product checks
    dot_res = dot_product_checks(st)

    # 2. Gradient vs central finite differences at rho = 0.4
    rho_fd = np.full(nd, 0.4)
    F_fd, g_adj = sim.value_and_grad(rho_fd, obj_f0)
    orig_pre, orig_post = PortAdjointHelper.pre_step_T, PortAdjointHelper.post_step_T
    try:
        PortAdjointHelper.pre_step_T = lambda self, u1b: u1b
        PortAdjointHelper.post_step_T = lambda self, rb: np.zeros_like(rb)
        _, g_drop = sim.value_and_grad(rho_fd, obj_f0)
    finally:
        PortAdjointHelper.pre_step_T = orig_pre
        PortAdjointHelper.post_step_T = orig_post

    out_fwd = sim.forward(rho_fd, record=True)
    _, G_cot = obj_f0(out_fwd)
    mon_f0.half_step = False
    g_no_half = sim.vjp(G_cot)
    mon_f0.half_step = True

    rng = np.random.default_rng(0)
    # Pick 2 cells from each of the 3 matching layers (6 cells total)
    pick = np.concatenate(
        [rng.choice(np.flatnonzero(msk), 2, replace=False) for msk in layer_masks]
    )
    h_fd = 1e-3
    fd_checks = []
    for k in pick:
        fds = []
        for hh in (h_fd, 2.0 * h_fd):
            rp, rm = rho_fd.copy(), rho_fd.copy()
            rp[k] += hh
            rm[k] -= hh
            Fp = obj_f0(sim.forward(rp, record=False))[0]
            Fm = obj_f0(sim.forward(rm, record=False))[0]
            fds.append((Fp - Fm) / (2.0 * hh))
        trunc = abs(fds[1] - fds[0]) / 3.0
        roundoff = 1e3 * np.finfo(float).eps * abs(F_fd) / h_fd
        fd_checks.append(
            dict(
                cell=int(k),
                adjoint=float(g_adj[k]),
                fd_h=float(fds[0]),
                fd_2h=float(fds[1]),
                abs_err=float(abs(g_adj[k] - fds[0])),
                rel_err=float(abs(g_adj[k] - fds[0]) / abs(fds[0])),
                trunc_est=float(trunc),
                roundoff=float(roundoff),
                mut_drop_port_T=float(g_drop[k]),
                mut_drop_port_T_rel_err=float(abs(g_drop[k] - fds[0]) / abs(fds[0])),
                mut_no_half_step=float(g_no_half[k]),
                mut_no_half_step_rel_err=float(
                    abs(g_no_half[k] - fds[0]) / abs(fds[0])
                ),
            )
        )

    # 3. Optimisation: L-BFGS-B on per-cell rho in [0, 1] (starting from unmatched rho = 0)
    from scipy.optimize import minimize

    rho0 = np.zeros(nd)
    history = [metrics_f0(rho0)]

    def loss_and_grad(r):
        val, g = sim.value_and_grad(r, obj_f0)
        return -val, -g

    def cb(xk):
        history.append(metrics_f0(xk))

    opt = minimize(
        loss_and_grad,
        rho0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 1.0)] * nd,
        callback=cb,
        options=dict(maxiter=6, ftol=1e-9),
    )
    rho_opt = np.clip(opt.x, 0.0, 1.0)
    if len(history) == 1 or not np.allclose(rho_opt, rho0):
        # Ensure final point is recorded if callback didn't just record it
        m_end = metrics_f0(rho_opt)
        if abs(m_end["objective"] - history[-1]["objective"]) > 1e-12:
            history.append(m_end)

    # 4. Broadband X-band evaluation (8-12 GHz)
    mon_band = PortDFTMonitor(F_BAND)
    sim_band = UnstructuredAdjointSimulation(make, design, {"p": mon_band}, NT)
    V_init = sim_band.forward(rho0, record=False)["p"]["dft"]
    S_init, _ = mon_band.s_column(V_init, sim_band.solver, NT, excite=0)
    V_opt = sim_band.forward(rho_opt, record=False)["p"]["dft"]
    S_opt, _ = mon_band.s_column(V_opt, sim_band.solver, NT, excite=0)

    eps_layers_opt = eps_layers_base.copy()
    eps_c_opt = design.eps_r(rho_opt)
    for idx_l, msk in zip((1, 2, 3), layer_masks):
        eps_layers_opt[idx_l] = float(eps_c_opt[msk].mean())
    line_opt = ExtrudedLine(z, eps_layers_opt, 1.0, kc, dt)
    S_ladder_opt = line_opt.s_matrix(F_BAND)

    res = dict(
        a_m=A_WG,
        b_m=B_WG,
        dz_m=DZ,
        nz=NZ,
        length_m=float(z[-1]),
        n_cells=int(m.n_cells),
        n_free_edges=int(sim.solver.free.size),
        n_massless_edges=int(sim.solver.massless.size),
        n_design_cells=int(nd),
        dt_s=float(dt),
        nt=NT,
        f0_hz=F0,
        eps_load=EPS_LOAD,
        kc_te10_inv_m=kc,
        dot_product=dot_res,
        fd_checks=fd_checks,
        history=history,
        freqs_hz=F_BAND.tolist(),
        initial_s11_db=(
            20.0 * np.log10(np.maximum(np.abs(S_init[:, 0]), 1e-15))
        ).tolist(),
        initial_s21_power=(np.abs(S_init[:, 1]) ** 2).tolist(),
        optimized_s11_db=(
            20.0 * np.log10(np.maximum(np.abs(S_opt[:, 0]), 1e-15))
        ).tolist(),
        optimized_s21_power=(np.abs(S_opt[:, 1]) ** 2).tolist(),
        optimized_ladder_s11_db=(
            20.0 * np.log10(np.maximum(np.abs(S_ladder_opt[:, 0, 0]), 1e-15))
        ).tolist(),
        z_levels_m=z.tolist(),
        eps_layers_initial=eps_layers_base.tolist(),
        eps_layers_optimized=eps_layers_opt.tolist(),
        design_cell_z_m=zc_des.tolist(),
        design_cell_eps_final=eps_c_opt.tolist(),
        wall_s=float(time.time() - t0),
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)

    p1, p2 = make_Figures(res, fig_dir)
    summary = dict(
        out_json=out_path,
        figures=[p1, p2],
        n_cells=res["n_cells"],
        n_design_cells=res["n_design_cells"],
        dot_k1=dot_res["multistep_k1"],
        dot_k5=dot_res["multistep_k5"],
        dot_k10=dot_res["multistep_k10"],
        dot_mut_drop=dot_res["mutation_drop_port_T_k1"],
        dot_mut_flip=dot_res["mutation_flip_port_T_k1"],
        fd_max_rel_err=max(r["rel_err"] for r in fd_checks),
        fd_mut_drop_max_rel_err=max(r["mut_drop_port_T_rel_err"] for r in fd_checks),
        fd_mut_half_max_rel_err=max(r["mut_no_half_step_rel_err"] for r in fd_checks),
        initial_s11_db_f0=history[0]["s11_db_f0"],
        initial_s21_power_f0=history[0]["s21_power_f0"],
        final_s11_db_f0=history[-1]["s11_db_f0"],
        final_s21_power_f0=history[-1]["s21_power_f0"],
        layer_eps_final=history[-1]["layer_eps_mean"],
        wall_s=res["wall_s"],
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
