"""Figures and tables of the triangle-vs-Cartesian Mie cylinder study.

``python mie_mesh_study.py plot`` (or ``python plot_study.py``) reads
``data/*.json`` / ``data/map_*.npz`` and writes ``figures/*.png`` and
``data/summary.md``. The mesh figure builds the (small) res-10 meshes
locally.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import hankel1, jv  # noqa: E402

import mie_mesh_study as S  # noqa: E402

FIG = S.FIG
DATA = S.DATA
STYLE = {
    "staircase": dict(color="#D55E00", marker="s", label="Wakis FIT (Cartesian staircase)"),
    "conformal": dict(color="#E69F00", marker="D", label="Wakis FIT (Cartesian conformal 'all')"),
    "tensor": dict(color="#CC79A7", marker="v", label="Wakis FIT (Cartesian conformal 'tensor')"),
    "tri-voronoi": dict(color="#0072B2", marker="o", label="Wakis FIT (triangles, Voronoi Hodge)"),
    "tri-barycentric": dict(color="#009E73", marker="^", label="Wakis FIT (triangles, barycentric Hodge)"),
}  # fmt: skip
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 9})


def load():
    runs = []
    for f in sorted(DATA.glob("*.json")):
        d = json.loads(f.read_text())
        if "error" in d:
            print("FAILED:", f.name, d["error"])
            continue
        for k in ("q", "q_naive", "q_ref", "refl", "c_err", "resid", "asym"):
            d[k] = np.array(d[k])
        e = d["q"] / d["q_ref"] - 1
        d["err_rms"] = float(np.sqrt(np.mean(e**2)))
        d["err_max"] = float(np.max(np.abs(e)))
        lo = S.X <= 5.0 + 1e-9
        d["err_rms_lo"] = float(np.sqrt(np.mean(e[lo] ** 2)))
        en = d["q_naive"] / d["q_ref"] - 1
        d["err_rms_naive"] = float(np.sqrt(np.mean(en**2)))
        d["cerr_rms"] = float(np.sqrt(np.mean(d["c_err"] ** 2)))
        if "q_flux" in d:
            d["q_flux"] = np.array(d["q_flux"])
        runs.append(d)
    return runs


def main_runs(runs, arm=None, pol=None):
    out = [r for r in runs if not r.get("flux_run")]
    if arm:
        out = [r for r in out if r["arm"] == arm]
    if pol:
        out = [r for r in out if r["pol"] == pol]
    return sorted(out, key=lambda r: r["res"])


# --------------------------------------------------------------------------
def fig_meshes(res=10):
    import pyvista as pv  # noqa: F401

    import wakis
    from wakis.unstructured.meshers import disk_mesh

    h = S.LAM_MIN_DIEL / res
    win = (0.55e-6, 1.15e-6, -0.05e-6, 0.55e-6)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True)
    th = np.linspace(0, np.pi / 2, 200)
    maps = {}
    for conf in ("staircase", "all"):
        nh = int(np.ceil(S.R_OUT / h)) + S.N_PML
        half = nh * h
        cyl = pv.Cylinder(radius=S.A, height=10e-6, direction=(0, 0, 1), resolution=720)
        grid = wakis.GridFIT3D(
            -half, half, -half, half, -5 * h, 5 * h, 2 * nh, 2 * nh, 1,
            stl_solids={"cyl": cyl.triangulate()}, stl_materials={"cyl": [4.0, 1.0]},
            conformal=False if conf == "staircase" else conf, verbose=0,
        )  # fmt: skip
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"], n_pml=S.N_PML, use_stl=True,
            verbose=0,
        )  # fmt: skip
        gx = np.asarray(grid.x)[: solver.Nx + 1]
        gy = np.asarray(grid.y)[: solver.Ny + 1]
        ieps = solver.ieps.toarray().reshape(3, -1)
        Nx, Ny = solver.Nx, solver.Ny
        # E_x samples at (x_{i+1/2}, y_j): eps_xx seen by in-plane (TE) E
        ex = 1 / ieps[0].reshape(Nx, Ny, order="F") / S.EPS0
        maps[conf] = (gx, gy, ex)
    for ax, conf, title in (
        (axs[0], "staircase", "Cartesian staircase: eps of $E_x$ edges"),
        (
            axs[1],
            "all",
            "Cartesian conformal: averaged $\\varepsilon_{xx}$ of $E_x$ edges",
        ),
    ):
        gx, gy, ex = maps[conf]
        xm = 0.5 * (gx[:-1] + gx[1:])
        # E_x sample (i, j) sits at (x_{i+1/2}, y_j): draw it as a cell centred there
        xe = np.concatenate([[gx[0]], gx[1:-1], [gx[-1]]])
        ye = np.concatenate(
            [[gy[0] - h / 2], 0.5 * (gy[:-1] + gy[1:]), [gy[-1] + h / 2]]
        )
        im = ax.pcolormesh(
            xe[: len(xm) + 1] * 1e6, ye[: ex.shape[1] + 1] * 1e6, ex[: len(xm), :].T,
            cmap="viridis", vmin=1, vmax=4, shading="flat", edgecolors="w", linewidth=0.2,
        )  # fmt: skip
        ax.plot(np.cos(th), np.sin(th), "w--", lw=1.2, label="true interface r = a")
        ax.set_title(title)
    fig.colorbar(
        im, ax=axs[:2], label="relative permittivity seen by $E_x$", shrink=0.8
    )
    mesh = disk_mesh([S.A, S.R_OUT], h)
    ax = axs[2]
    tri = mesh.cells
    X = mesh.nodes * 1e6
    ax.tripcolor(X[:, 0], X[:, 1], tri, facecolors=np.where(mesh.cell_tags == 0, 4.0, 1.0),
                 cmap="viridis", vmin=1, vmax=4, edgecolors="w", linewidth=0.4)  # fmt: skip
    # Voronoi dual edges: edge midpoint -> circumcentre of each adjacent triangle
    ce = mesh.cell_edges
    seg = []
    for loc in range(3):
        m = mesh.edge_midpoint[ce[:, loc]] * 1e6
        c = mesh.cell_circumcenter * 1e6
        seg.append(np.stack([m, c], 1))
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(np.concatenate(seg), colors="k", linewidths=0.5,
                                     label="Voronoi dual edges"))  # fmt: skip
    ax.plot(np.cos(th), np.sin(th), "w--", lw=1.2)
    ax.set_title("Triangles (conforming), Voronoi dual in black")
    for ax in axs:
        ax.set_xlim(win[0] * 1e6, win[1] * 1e6)
        ax.set_ylim(win[2] * 1e6, win[3] * 1e6)
        ax.set_aspect("equal")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
    fig.suptitle(
        f"Meshes near the interface (a = 1 µm, n = 2), h = λ_min,diel/{res} = {h * 1e9:.0f} nm"
    )
    fig.savefig(FIG / "a_meshes.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def exact_scattered(x_grid, y_grid, pol, x=S.X_SNAP):
    """u_s / U0 (Wakis exp(+i w t) convention) on a grid [m]."""
    from wakis.analytic import cylinder_coefficients

    k = x / S.A
    c = cylinder_coefficients(S.N_CYL, x, pol, n_terms=40)
    n = np.arange(-40, 41)
    cn = c[np.abs(n)]
    X, Y = np.meshgrid(x_grid, y_grid)
    r, t = np.hypot(X, Y), np.arctan2(Y, X)
    out = np.zeros_like(X, dtype=complex)
    d = (jv(n, x) - cn * hankel1(n, x)) / jv(n, S.N_CYL * x)
    for nn, cc, dd in zip(n, cn, d):
        ang = (1j**nn) * np.exp(1j * nn * t)
        outside = -cc * hankel1(nn, k * np.maximum(r, 1e-12))
        inside = dd * jv(nn, S.N_CYL * k * r)
        out += ang * np.where(r > S.A, outside, inside)
    out -= np.where(r <= S.A, np.exp(1j * k * X), 0)
    return np.conj(out)


def fig_fields(res=20):
    import matplotlib.tri as mtri

    arms = ["staircase", "tensor", "tri-voronoi"]
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.2), constrained_layout=True)
    L = 2.6
    xg = np.linspace(-L, L, 401) * 1e-6
    th = np.linspace(0, 2 * np.pi, 300)
    for row, pol in enumerate(("TM", "TE")):
        for col, arm in enumerate(arms + ["exact"]):
            ax = axs[row, col]
            a = "conformal" if (arm == "tensor" and pol == "TM") else arm
            if arm == "exact":
                u = exact_scattered(xg, xg, pol)
                im = ax.pcolormesh(
                    xg * 1e6, xg * 1e6, u.real, cmap="RdBu_r", vmin=-1.2, vmax=1.2
                )
                ax.set_title(f"{pol}: exact series")
            else:
                f = DATA / f"map_{a}_{pol}_{res}.npz"
                if not f.exists():
                    ax.set_title(f"{a} {pol}: missing")
                    continue
                d = np.load(f)
                if a.startswith("tri"):
                    u = np.zeros(len(d["nodes"]), complex)
                    u[d["free"]] = d["u"]
                    u = u / d["u0"]
                    tr = mtri.Triangulation(
                        d["nodes"][:, 0] * 1e6, d["nodes"][:, 1] * 1e6, d["cells"]
                    )
                    im = ax.tripcolor(
                        tr,
                        u.real,
                        cmap="RdBu_r",
                        vmin=-1.2,
                        vmax=1.2,
                        shading="gouraud",
                    )
                else:
                    u = d["u"] / d["u0"]
                    # TF/SF: total field inside the box -> subtract the
                    # (analytic) incident wave to show the scattered field
                    X_, Y_ = np.meshgrid(d["xs"], d["ys"])
                    hh = S.LAM_MIN_DIEL / res
                    box = (np.abs(X_) <= S.TF_HALF + 0.25 * hh) & (
                        np.abs(Y_) <= S.TF_HALF + 0.25 * hh
                    )
                    u = u - box * np.exp(-1j * S.X_SNAP / S.A * X_)
                    im = ax.pcolormesh(d["xs"] * 1e6, d["ys"] * 1e6, u.real, cmap="RdBu_r",
                                       vmin=-1.2, vmax=1.2)  # fmt: skip
                ax.set_title(f"{pol}: {STYLE[a]['label']}", fontsize=7.5)
            ax.plot(np.cos(th), np.sin(th), "k-", lw=0.6)
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            ax.set_aspect("equal")
            ax.set_xlabel("x [µm]")
            ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=axs, shrink=0.7,
                 label="Re(u_scat / u_inc(0)),  u = E_z (TM) or H_z (TE)")  # fmt: skip
    h = S.LAM_MIN_DIEL / res
    fig.suptitle(
        f"Scattered field at x = ka = {S.X_SNAP:g}, h = {h * 1e9:.1f} nm "
        f"(λ_min,diel/{res}); plane wave along +x"
    )
    fig.savefig(FIG / "b_fields.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_qsca(runs, res=14):
    from wakis.analytic import cylinder_efficiencies

    xd = np.linspace(1.9, 10.1, 2000)
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, constrained_layout=True,
                            gridspec_kw=dict(height_ratios=[2, 1]))  # fmt: skip
    for col, pol in enumerate(("TM", "TE")):
        ax, ae = axs[0, col], axs[1, col]
        ax.plot(
            xd,
            cylinder_efficiencies(S.N_CYL, xd, pol)[1],
            "k-",
            lw=1.1,
            label="exact series",
        )
        for arm, st in STYLE.items():
            rr = [r for r in main_runs(runs, arm, pol) if r["res"] == res]
            if not rr:
                continue
            r = rr[0]
            ax.plot(S.X, r["q"], ls="none", ms=4, mfc="none", marker=st["marker"],
                    color=st["color"], label=st["label"])  # fmt: skip
            ae.plot(S.X, 100 * (r["q"] / r["q_ref"] - 1), "-", marker=st["marker"], ms=3,
                    color=st["color"], lw=0.8)  # fmt: skip
        ax.set_ylabel("$Q_{sca}$")
        ax.set_title(
            f"{pol} ({'E along the axis' if pol == 'TM' else 'E across the axis'})"
        )
        ae.axhline(0, color="k", lw=0.6)
        ae.set_ylabel("error vs series [%]")
        ae.set_xlabel("size parameter x = 2πa/λ")
        ax.legend(fontsize=7, loc="lower right")
    h = S.LAM_MIN_DIEL / res
    fig.suptitle(
        f"Dielectric cylinder n = 2, a = 1 µm: $Q_{{sca}}$ at h = {h * 1e9:.1f} nm "
        f"(λ_min in the dielectric / {res})"
    )
    fig.savefig(FIG / "c_qsca.png")
    plt.close(fig)


def fig_convergence(runs):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, pol in enumerate(("TM", "TE")):
        for arm, st in STYLE.items():
            rr = main_runs(runs, arm, pol)
            if not rr and not (arm == "tensor" and pol == "TM"):
                rr = []
            if not rr:
                continue
            res = np.array([r["res"] for r in rr])
            err = np.array([r["err_rms"] for r in rr])
            dofs = np.array([r["dofs"] for r in rr])
            t = np.array([r["dofs"] * r["steps"] for r in rr], float)
            kw = dict(color=st["color"], marker=st["marker"], ms=5, lw=1)
            axs[row, 0].loglog(res, err, label=st["label"], **kw)
            axs[row, 1].loglog(dofs, err, **kw)
            axs[row, 2].loglog(t, err, **kw)
            if arm == "tensor" and pol == "TM":
                continue
            for rp in sorted(set(S.RES) - set(res.tolist())):
                # pending run: open marker on the bottom axis at its resolution
                tr = axs[row, 0].get_xaxis_transform()
                axs[row, 0].plot(rp, 0.04, transform=tr, ls="none", marker=st["marker"],
                                 mfc="none", color=st["color"], ms=6, clip_on=False)  # fmt: skip
                axs[row, 0].text(rp, 0.08, "pending", transform=tr, fontsize=6,
                                 ha="center", color=st["color"])  # fmt: skip
        for col, (xl, slope, x0) in enumerate(
            (("cells per λ in the dielectric at x = 10  (λ_min,diel / h)", -2, 14),
             ("field unknowns (2D: Cartesian 3/cell; triangles nodes + edges)", -1, 1e5),
             ("work: unknowns × time steps", -2 / 3, 1e10))
        ):  # fmt: skip
            ax = axs[row, col]
            ax.set_xlabel(xl)
            ax.set_ylabel(f"{pol}: RMS rel. error of $Q_{{sca}}$ over x = 2 to 10")
            ax.grid(True, which="both", alpha=0.3)
            yl = ax.get_ylim()
            xl_ = ax.get_xlim()
            xx = np.geomspace(*xl_, 10)
            y0 = np.sqrt(yl[0] * yl[1])
            ax.loglog(xx, y0 * (xx / np.sqrt(xl_[0] * xl_[1])) ** slope, "k:", lw=1,
                      label=f"slope {slope:.2g} (2nd order)")  # fmt: skip
            ax.set_ylim(yl)
            ax.set_xlim(xl_)
            ax.legend(fontsize=6.5, loc="lower left" if col else "upper right")
    axs[0, 0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle(
        "Error vs resolution, unknowns and work (unknowns × steps)"
    )
    fig.savefig(FIG / "d_convergence.png")
    plt.close(fig)


def table(runs):
    lines = [
        "| arm | pol | res (λ_min,diel/h) | h [nm] | unknowns | dt [as] | steps | unknowns × steps | run [s] "
        "| RMS err | max err | RMS err x≤5 | RMS err no de-embed | c_n err RMS | refl. x=2 / 6 / 10 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pol in ("TM", "TE"):
        for arm in STYLE:
            for r in main_runs(runs, arm, pol):
                i2, i6 = 0, int(np.argmin(np.abs(S.X - 6)))
                lines.append(
                    f"| {arm} | {pol} | {r['res']} | {r['h'] * 1e9:.1f} | {r['dofs']:,} "
                    f"| {r['dt'] * 1e18:.1f} | {r['steps']} | {r['dofs'] * r['steps'] / 1e9:.1f}e9 | {r['run_s']:.0f} "
                    f"| {100 * r['err_rms']:.2f} % | {100 * r['err_max']:.2f} % | {100 * r['err_rms_lo']:.2f} % "
                    f"| {100 * r['err_rms_naive']:.2f} % | {100 * r['cerr_rms']:.1f} % "
                    f"| {r['refl'][i2]:.3f} / {r['refl'][i6]:.4f} / {r['refl'][-1]:.4f} |"
                )
    done = {(r["arm"], r["pol"], r["res"]) for r in main_runs(runs)}
    pend = [f"{a} {p} res {q}" for q in S.RES for p in ("TM", "TE") for a in STYLE
            if (a, p, q) not in done and not (a == "tensor" and p == "TM")]  # fmt: skip
    if pend:
        lines += ["", "Pending (still running): " + ", ".join(pend)]
    flux = [r for r in runs if r.get("flux_run")]
    if flux:
        lines += ["", "Flux-box cross-check (Cartesian, 4 cells along z):", "",
                  "| arm | pol | res | RMS err harmonics | RMS err flux box | RMS (harm - flux)/series |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        for r in sorted(flux, key=lambda r: (r["pol"], r["arm"], r["res"])):
            ef = r["q_flux"] / r["q_ref"] - 1
            dd = (r["q"] - r["q_flux"]) / r["q_ref"]
            lines.append(
                f"| {r['arm']} | {r['pol']} | {r['res']} | {100 * r['err_rms']:.2f} % "
                f"| {100 * np.sqrt(np.mean(ef**2)):.2f} % | {100 * np.sqrt(np.mean(dd**2)):.3f} % |"
            )
    txt = "\n".join(lines)
    (DATA / "summary.md").write_text(txt + "\n")
    print(txt)


def main_plot():
    FIG.mkdir(exist_ok=True)
    runs = load()
    table(runs)
    fig_qsca(runs)
    fig_convergence(runs)
    fig_fields()
    fig_meshes()


if __name__ == "__main__":
    main_plot()
