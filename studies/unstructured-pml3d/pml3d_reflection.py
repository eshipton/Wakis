"""
Spherical CFS-PML of `UnstructuredFIT3D`: reflection, baseline, stability.

Every case shares one tetrahedral ball (radius RB, edge h) and its radially
extruded shell, so the interior meshes of the test runs and of the reference
run are identical node by node; the reference continues the shell with
N_EXTRA plain vacuum layers before its own PML, far enough that nothing it
reflects reaches a probe inside the time window. The reflection is then the
probe-field difference test - reference, which removes the (common)
discretisation error of the interior exactly. All runs use the same dt.

Usage
-----
    python pml3d_reflection.py run [n_procs] [--quick]   # writes reflection.json, signals.npz
    python pml3d_reflection.py stability [n_steps]       # writes stability.json
    python pml3d_reflection.py plot                      # figures from the JSON files
"""

import json
import os
import sys
import time

import numpy as np
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0

HERE = os.path.dirname(os.path.abspath(__file__))
H = 1.0  # edge length [m] (the problem is scale free; lengths in units of h)
RB = 8.0  # ball radius
LAM0 = 16.0  # centre wavelength of the pulse
TAU = LAM0 / (np.pi * C0)  # Gaussian width [s]; spectrum peak of dI/dt at ~ LAM0
T0 = 4 * TAU
N_EXTRA = 40  # vacuum layers of the reference before its PML
N_REF_PML = 16
LAYERS = [4, 6, 10, 16]
A_OBL = 1.0  # source / probe depth below the PML surface (oblique cases)
PSI = [20, 40, 60, 90, 120, 150]  # angular source-probe separations [deg]


def amp(t):
    """Current moment I l(t) [A m]: derivative of a Gaussian (zero net charge)."""
    x = (t - T0) / TAU
    return -x * np.exp(-(x**2))


def dipole_field(points, t, p_dir):
    """
    Exact field of a point dipole p(t) = int I l dt in vacuum at `points`.

    E = [3 r^ (r^.p) - p] (p / r^3 + p' / (c r^2)) / (4 pi eps0)
        + [r^ (r^.p'') - p''] / (4 pi eps0 c^2 r), retarded at t - r / c,
    with p = (tau / 2) exp(-x^2), p' = I l, p'' = d(I l)/dt.
    """
    P = np.atleast_2d(points)
    r = np.linalg.norm(P, axis=1)
    rh = P / r[:, None]
    d = np.asarray(p_dir, float) / np.linalg.norm(p_dir)
    tr = np.asarray(t)[:, None] - r[None] / C0
    x = (tr - T0) / TAU
    g = np.exp(-(x**2))
    p0 = 0.5 * TAU * g
    p1 = -x * g
    p2 = (2 * x**2 - 1) * g / TAU
    rd = rh @ d  # (n,)
    near = 3 * rh[None] * rd[None, :, None] - d[None, None]
    far = rh[None] * rd[None, :, None] - d[None, None]
    E = (near * (p0 / r**3 + p1 / (C0 * r**2))[..., None] + far * (p2 / (C0**2 * r))[..., None])
    return E / (4 * np.pi * EPS0)


def build_mesh(n_extra, n_layers):
    from wakis.unstructured.pml3d import ball_with_pml_mesh

    return ball_with_pml_mesh(RB, H, n_layers, n_extra=n_extra)


def geometry(kind):
    """Source position, direction and probe points of a case."""
    if kind == "normal":
        # off-node source: the Whitney point dipole on a vertex puts its
        # charges on one arbitrary tetrahedron (strong near-field multipoles)
        src, d = np.array([0.31, 0.17, 0.23]), np.array([0.0, 0, 1])
        rp = RB - 1.0
        th = np.radians([90, 60, 30])
        probes = rp * np.stack([np.sin(th), 0 * th, np.cos(th)], 1)
        angles = np.zeros(len(th))
    else:
        rs = RB - A_OBL
        src, d = np.array([0, 0, rs]), np.array([0.0, 0, 1])  # radial dipole
        ps = np.radians(PSI)
        probes = rs * np.stack([np.sin(ps), 0 * ps, np.cos(ps)], 1)
        angles = np.degrees([incidence_angle(rs, RB, p) for p in ps])
    return src, d, probes, angles


def incidence_angle(rs, R, psi):
    """Incidence angle at the specular point (angle psi/2) for source and
    probe at radius rs separated by the angle psi on a sphere of radius R."""
    s = np.array([0.0, rs])
    m = R * np.array([np.sin(psi / 2), np.cos(psi / 2)])
    v = m - s
    return np.arccos(np.dot(v, m / R) / np.linalg.norm(v))


def window(kind, n_layers_max):
    """End time [s]: pulse end + longest path through the thickest PML."""
    r1 = RB + n_layers_max * H
    if kind == "normal":
        path = 2 * r1 - (RB - 1.0)
    else:
        path = 2 * (r1 - (RB - A_OBL)) + 2 * RB  # generous
    return 2 * T0 + path / C0


def run_case(case):
    """One solver run; returns probe signals (n_t, n_probe, 3)."""
    from wakis.unstructured import UnstructuredFIT3D, ball_mesh

    kind, label, n_layers, opts, dt, T = (
        case["kind"], case["label"], case["n_layers"], case["opts"], case["dt"], case["T"],
    )
    t0 = time.time()
    if label == "abc":
        m = ball_mesh([RB], H)
        s = UnstructuredFIT3D(m, bc="abc", dt=dt)
    elif label == "pec":
        m = ball_mesh([RB], H)
        s = UnstructuredFIT3D(m, bc="pec", dt=dt)
    else:
        n_extra = N_EXTRA if label == "ref" else 0
        m, cfg = build_mesh(n_extra, n_layers)
        s = UnstructuredFIT3D(m, bc="pec", dt=dt, pml=dict(cfg, **opts))
    if s.dt_max < dt:
        raise RuntimeError(f"{label}: dt {dt} above the limit {s.dt_max}")
    src, d, probes, _ = geometry(kind)
    s.add_point_dipole(src, d, amp)
    P = whitney_probe(s.mesh, probes)
    volt = s._pml.edge_voltages if s._pml is not None else (lambda: s.e)
    n = int(round(T / dt))
    sig = np.empty((n, len(probes), 3))
    for k in range(n):
        s.one_step()
        sig[k] = (P @ volt()).reshape(len(probes), 3)
    return dict(case, sig=sig, n_cells=s.mesh.n_cells, wall=time.time() - t0)


def whitney_probe(mesh, points):
    """Sparse (3 n_points, n_edges): Whitney interpolation of E from *all* edge
    voltages (`add_probe` omits the massless edges, which is not small)."""
    import scipy.sparse as sp

    cell, W = mesh.whitney_edge(points)
    rows = np.repeat(np.arange(len(cell)), 6)
    cols = mesh.cell_edges[cell].ravel()
    mats = [sp.csr_matrix((W[:, :, c].ravel(), (rows, cols)), shape=(len(cell), mesh.n_edges)) for c in range(3)]
    # interleave: row 3 i + c
    M = sp.vstack(mats).tocsr()
    perm = np.arange(3 * len(cell)).reshape(3, -1).T.ravel()
    return M[perm]


def common_dt():
    from wakis.unstructured import UnstructuredFIT3D, ball_mesh

    dts = []
    m, cfg = build_mesh(N_EXTRA, N_REF_PML)
    dts.append(UnstructuredFIT3D(m, bc="pec").dt_max)
    m = ball_mesh([RB], H)
    dts.append(UnstructuredFIT3D(m, bc="abc").dt_max)
    return 0.9 * min(dts)


def spectra(sig, dt, freqs):
    t = (np.arange(sig.shape[0]) + 1) * dt
    k = np.exp(-2j * np.pi * np.outer(freqs, t)) * dt
    return np.einsum("ft,tpc->fpc", k, sig)


def cases(dt, quick=False):
    out = []
    layers = [4, 10] if quick else LAYERS
    for kind in ("normal", "oblique"):
        T = window(kind, max(LAYERS))
        base = dict(kind=kind, dt=dt, T=T)
        out.append(dict(base, label="ref", n_layers=N_REF_PML, opts={}))
        out.append(dict(base, label="abc", n_layers=0, opts={}))
        out.append(dict(base, label="pec", n_layers=0, opts={}))
        for nl in layers:
            out.append(dict(base, label=f"pml{nl}", n_layers=nl, opts={}))
        # CFS variants at 10 layers
        a0 = 2 * np.pi * C0 / LAM0 / 10
        out.append(dict(base, label="pml10_cfs", n_layers=10, opts=dict(kappa_max=5.0, alpha=a0)))
        out.append(dict(base, label="pml10_alpha", n_layers=10, opts=dict(alpha=a0)))
    return out


def run(n_procs=1, quick=False):
    from multiprocessing import Pool

    dt = common_dt()
    cs = cases(dt, quick)
    t = time.time()
    if n_procs == 1:
        res = list(map(run_case, cs))
    else:
        with Pool(n_procs) as pool:
            res = pool.map(run_case, cs, chunksize=1)
    print(f"{len(cs)} runs in {time.time() - t:.0f} s on {n_procs} processes")
    freqs = C0 / np.linspace(32, 10, 23)  # lambda = 32 h .. 10 h
    summary = dict(h=H, RB=RB, lam0=LAM0, tau=TAU, dt=dt, N_extra=N_EXTRA,
                   lam_over_h=(C0 / freqs).tolist(), freqs=freqs.tolist(), cases={})  # fmt: skip
    npz = {}
    for kind in ("normal", "oblique"):
        R = [r for r in res if r["kind"] == kind]
        ref = [r for r in R if r["label"] == "ref"][0]
        src, d, probes, angles = geometry(kind)
        Sref = spectra(ref["sig"], dt, freqs)
        nref = np.linalg.norm(ref["sig"], axis=-1).max(0)  # per probe
        ent = {"probes": probes.tolist(), "angles_deg": angles.tolist(), "runs": {}}
        # Reference vs the exact dipole field (discretisation error of the interior)
        tt = (np.arange(ref["sig"].shape[0]) + 1) * dt
        Ex = dipole_field(probes - src, tt, d)
        ent["ref_vs_exact"] = (
            np.linalg.norm(ref["sig"] - Ex, axis=-1).max(0) / np.linalg.norm(Ex, axis=-1).max(0)
        ).tolist()
        npz[f"{kind}_t"] = tt
        npz[f"{kind}_exact"] = Ex
        for r in R:
            npz[f"{kind}_{r['label']}"] = r["sig"]
            if r["label"] == "ref":
                continue
            diff = r["sig"] - ref["sig"]
            Rt = np.linalg.norm(diff, axis=-1).max(0) / nref
            Sd = spectra(diff, dt, freqs)
            Rf = np.linalg.norm(Sd, axis=-1) / np.linalg.norm(Sref, axis=-1)  # (f, probe)
            ent["runs"][r["label"]] = dict(
                n_layers=r["n_layers"], opts=r["opts"], n_cells=r["n_cells"], wall_s=r["wall"],
                R_time=Rt.tolist(), R_freq=Rf.tolist(),
                R_time_max_dB=float(20 * np.log10(Rt.max())),
                R_freq_max_dB=float(20 * np.log10(Rf.max())),
            )  # fmt: skip
            print(kind, r["label"], "R_t", np.round(Rt, 6), "max R_f", Rf.max())
        summary["cases"][kind] = ent
    with open(os.path.join(HERE, "reflection.json"), "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(os.path.join(HERE, "signals.npz"), **npz)


def stability(n_steps=100000):
    """Long runs (no source after the pulse): alpha = 0 vs CFS alpha > 0."""
    from multiprocessing import Pool

    cfgs = [
        dict(label="pml10_alpha0", opts=dict()),
        dict(label="pml10_cfs", opts=dict(kappa_max=5.0, alpha=2 * np.pi * C0 / LAM0 / 10)),
        dict(label="pml10_alpha", opts=dict(alpha=2 * np.pi * C0 / LAM0 / 10)),
    ]
    for c in cfgs:
        c["n_steps"] = n_steps
    with Pool(len(cfgs)) as pool:
        res = pool.map(_long_run, cfgs)
    with open(os.path.join(HERE, "stability.json"), "w") as f:
        json.dump(dict(n_steps=n_steps, runs=res), f, indent=1)


def _long_run(c):
    from wakis.unstructured import UnstructuredFIT3D
    from wakis.unstructured.pml3d import ball_with_pml_mesh

    t0 = time.time()
    rb = 5.0
    m, cfg = ball_with_pml_mesh(rb, H, 10)
    s = UnstructuredFIT3D(m, bc="pec", pml=dict(cfg, **c["opts"]))
    s.add_point_dipole([0.3, 0.2, 0.1], [0.2, 0.3, 1.0], amp)
    p = s._pml
    every = 50
    t, wi, wt, umax = [], [], [], []
    for k in range(c["n_steps"]):
        rec = k % every == every - 1
        if rec:
            u_prev = s.u.copy()
        s.one_step()
        if rec:
            t.append(s.n * s.dt)
            wi.append(p.interior_energy(u_prev))
            wt.append(s.energy())
            umax.append(float(np.abs(s.u).max()))
    return dict(c, dt=s.dt, t=t, W_interior=wi, W_total=wt, u_max=umax,
                n_cells=m.n_cells, wall_s=time.time() - t0)  # fmt: skip


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_img = os.path.join(HERE, "..", "..", "docs", "img")
    with open(os.path.join(HERE, "reflection.json")) as f:
        S = json.load(f)
    lam = np.array(S["lam_over_h"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, kind, ip in ((axs[0], "normal", 0), (axs[1], "oblique", 2)):
        ent = S["cases"][kind]
        for lab, st in [("pec", "k:"), ("abc", "k--")] + [(f"pml{n}", "-") for n in LAYERS] + [
            ("pml10_cfs", "-.")
        ]:
            if lab not in ent["runs"]:
                continue
            Rf = np.array(ent["runs"][lab]["R_freq"])[:, ip]
            name = {"pec": "PEC wall at $r_0$", "abc": "1st-order Silver-Müller at $r_0$",
                    "pml10_cfs": r"10 layers, $\kappa_{max}=5$, $\alpha>0$"}.get(lab, f"PML, {lab[3:]} layers")  # fmt: skip
            ax.semilogy(lam, Rf, st, marker="o" if lab.startswith("pml") else None, ms=3, label=name)
        ax.axhline(1e-2, color="0.6", lw=0.8)
        ax.text(lam[-1], 1.15e-2, "-40 dB", color="0.4", fontsize=8)
        ax.set_xlabel(r"wavelength $\lambda / h$")
        ax.set_ylabel(r"$|\hat E_{test} - \hat E_{ref}| / |\hat E_{ref}|$")
        ang = ent["angles_deg"][ip]
        ax.set_title(
            f"{kind}: probe {ip} " + ("(normal incidence)" if kind == "normal" else f"(incidence {ang:.0f}°)")
        )
        ax.invert_xaxis()
        ax.grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Spherical CFS-PML on tetrahedra: reflection vs wavelength (ball radius {RB:g} h, dr = h)")
    fig.savefig(os.path.join(out_img, "unstructured_pml3d_reflection.png"), dpi=150)
    plt.close(fig)

    # reflection vs incidence angle and layers
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ent = S["cases"]["oblique"]
    ang = np.array(ent["angles_deg"])
    for lab in ["abc"] + [f"pml{n}" for n in LAYERS]:
        if lab in ent["runs"]:
            axs[0].semilogy(PSI, ent["runs"][lab]["R_time"], "o-" if lab != "abc" else "s--",
                            label=lab.replace("pml", "PML ") + (" layers" if lab != "abc" else " (Silver-Müller)"))  # fmt: skip
    axs[0].set_xticks(PSI, [f"{p}°\n{a:.0f}°" for p, a in zip(PSI, ang)])
    axs[0].set_xlabel(r"source-probe separation $\psi$ / incidence angle at the specular point")
    axs[0].set_ylabel(r"$\max_t |E_{test}-E_{ref}| / \max_t |E_{ref}|$")
    axs[0].legend(fontsize=7)
    axs[0].grid(alpha=0.3, which="both")
    nl = np.array(LAYERS)
    for kind, mk in (("normal", "o"), ("oblique", "s")):
        e = S["cases"][kind]["runs"]
        Rm = [max(e[f"pml{n}"]["R_time"]) for n in LAYERS if f"pml{n}" in e]
        axs[1].loglog(nl[: len(Rm)], Rm, mk + "-", label=f"{kind} (max over probes)")
    axs[1].set_xlabel("PML layers N (dr = h, R_design = 1e-8, m = 3)")
    axs[1].set_ylabel("time-domain reflection")
    axs[1].legend(fontsize=7)
    axs[1].grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(out_img, "unstructured_pml3d_angle_layers.png"), dpi=150)
    plt.close(fig)

    p = os.path.join(HERE, "stability.json")
    if os.path.exists(p):
        with open(p) as f:
            St = json.load(f)
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for r in St["runs"]:
            t = np.array(r["t"]) * C0 / H
            w = np.array(r["W_interior"])
            axs[0].semilogy(t, np.abs(w) / np.abs(w).max(), label=r["label"])
            u = np.array(r["u_max"])
            axs[1].semilogy(t, u / u.max(), label=r["label"])
        axs[0].set_xlabel("time  c t / h")
        axs[0].set_ylabel("interior energy / peak")
        axs[1].set_xlabel("time  c t / h")
        axs[1].set_ylabel(r"$\max |e|$ / peak (whole mesh incl. PML)")
        for ax in axs:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, which="both")
        fig.suptitle(f"Late-time behaviour, {St['n_steps']} steps (ball 5 h + 10 PML layers)")
        fig.savefig(os.path.join(out_img, "unstructured_pml3d_late_time.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 1, quick="--quick" in sys.argv)
    elif mode == "stability":
        stability(int(sys.argv[2]) if len(sys.argv) > 2 else 100000)
    else:
        plot()
