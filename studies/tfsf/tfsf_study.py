"""
TF/SF plane-wave study: discrete-exact `TFSFSource` versus continuum Huygens
sheets, and sigma_sca of a dielectric sphere against Mie theory.

Part A (leakage). The same TF/SF box is driven in four ways:

- ``tfsf``: `wakis.TFSFSource` (1D auxiliary FIT line, commutator corrections).
- ``sheets_discrete``: six face sheets J = n x H_inc, K = -n x E_inc (n the
  inward normal) on the E plane of each face and the H plane half a cell
  outside it, with values taken from the auxiliary line at their own grid
  positions and time levels. This is the commutator form written as sheets
  and must agree with ``tfsf`` to round-off.
- ``sheets_continuum``: the same sheets and registration (J on the face E
  plane carries H_inc of the dual plane half a cell outside, K on that dual
  plane carries E_inc of the face node; transverse faces use the samples'
  own positions) driven by the continuum incident field
  A s(t - k.(r - r0)/c) at the exact time levels. Error left: the
  grid-dispersion mismatch between the continuum wave and the wave the grid
  propagates.
- ``sheets_face``: as above, but the electric sheet of the faces normal to k
  carries H_inc of the face plane itself, i.e. both sheets are referenced to
  the face plane and the half cell between the E and H planes is ignored, as
  in a hand-built Meep source on a line (the user's Meep notebook). Error
  added: the half-cell offset, O(k h).

Leakage = power through a closed box in the scattered-field region divided by
the incident power on one box face.

Part B (Mie). sigma_sca(f) of an eps_r = 4 sphere (radius 3.5 cm) for
staircased and conformal spheres at several resolutions, against Mie theory.

Usage::

    python tfsf_study.py run  [--quick] [--procs N]   # writes results.json
    python tfsf_study.py plot                          # writes the figures

Wrap local runs in ``bin/memguard``. The full study runs on a VM (preflight
with ``wakis.estimate``); ``--quick`` is a small local smoke test.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F0 = 1.0e9  # leakage study centre frequency [Hz]
C0 = 299792458.0


def _levi(i, j, k):
    return int(np.sign((j - i) * (k - i) * (k - j)))


# --------------------------------------------------------------------------
# Huygens-box helper (study only, not a public API)
# --------------------------------------------------------------------------
class HuygensBox:
    """Face sheets J = n x H_inc and K = -n x E_inc on the box of a TFSFSource.

    Parameters
    ----------
    tfsf : wakis.TFSFSource
        Provides the box, the incident-wave parameters and (mode 'discrete')
        the auxiliary-line fields. Its own corrections are switched off.
    mode : {'discrete', 'continuum', 'face'}
    """

    def __init__(self, tfsf, mode):
        self.tfsf = tfsf
        self.mode = mode
        tfsf._apply_corrections = False
        self.ready = False

    def _setup(self, solver):
        from wakis.field import Field

        src = self.tfsf
        src.prepare(solver)
        grid = solver.grid
        nodes = (grid.x, grid.y, grid.z)
        N = (solver.Nx, solver.Ny, solver.Nz)
        Ntot = N[0] * N[1] * N[2]
        a, p, q = src.axis, src.p, src.q
        In, Im = src._in_node, src._in_mid
        L = grid.L.toarray().reshape(3, -1)
        tL = grid.tL.toarray().reshape(3, -1)

        def flat(I, J, K):
            return I + N[0] * J + N[0] * N[1] * K

        sheets = {"J": [], "M": []}
        for d in range(3):
            if src._idx[d] is None:
                continue
            for side, sigma_n in ((0, 1.0), (1, -1.0)):  # inward normal
                i_f = src._idx[d][side]
                face_xi = nodes[a][src._idx[a][side]] if d == a else None
                # Electric sheet J_r = sigma_n eps_{d q r} H_q on the face E plane
                if d != q:
                    r = 3 - d - q
                    axes = [None] * 3
                    axes[d] = np.array([i_f])
                    axes[r] = np.flatnonzero(Im[r])
                    axes[q] = np.flatnonzero(In[q])
                    I = np.meshgrid(*axes, indexing="ij")
                    f = flat(*I).ravel()
                    idx_a = I[a].ravel()
                    on_mid = a == r  # E_r sits at the mid along r
                    xi = _pos(nodes[a], idx_a, on_mid)
                    if d == a and self.mode == "continuum":
                        # H_inc of the dual sample half a cell outside the face
                        i_out = i_f - 1 if side == 0 else i_f
                        xi = np.full_like(
                            xi, 0.5 * (nodes[a][i_out] + nodes[a][i_out + 1])
                        )
                    w = sigma_n * _levi(d, q, r) / tL[d][f]
                    sheets["J"].append(
                        dict(flat=r * Ntot + f, w=w, xi=xi, line=idx_a, on_mid=on_mid)
                    )
                # Magnetic sheet K_r = -sigma_n eps_{d p r} E_p on the H plane outside
                if d != p:
                    r = 3 - d - p
                    i_h = i_f - 1 if side == 0 else i_f
                    axes = [None] * 3
                    axes[d] = np.array([i_h])
                    axes[r] = np.flatnonzero(In[r])
                    axes[p] = np.flatnonzero(Im[p])
                    I = np.meshgrid(*axes, indexing="ij")
                    f = flat(*I).ravel()
                    idx_a = I[a].ravel()
                    on_mid = a == d  # H_r sits at the mid along d (H plane)
                    xi = _pos(nodes[a], idx_a, on_mid)
                    if face_xi is not None:
                        # K on the dual plane outside carries E_inc of the face node
                        xi = np.full_like(xi, face_xi)
                    w = -sigma_n * _levi(d, p, r) / L[d][f]
                    sheets["M"].append(
                        dict(flat=r * Ntot + f, w=w, xi=xi, line=idx_a, on_mid=on_mid)
                    )
        self.sheets = {}
        for k, parts in sheets.items():
            self.sheets[k] = {
                key: np.concatenate([s[key] for s in parts])
                for key in ("flat", "w", "xi", "line")
            }
            self.sheets[k]["on_mid"] = np.concatenate(
                [np.full(s["flat"].size, s["on_mid"]) for s in parts]
            )
        if solver.M is None:
            solver.M = Field(
                solver.Nx,
                solver.Ny,
                solver.Nz,
                use_gpu=solver.use_gpu,
                dtype=solver.dtype,
            )
        self.old = {k: np.zeros(v["flat"].size) for k, v in self.sheets.items()}
        self.ready = True

    def _continuum_E(self, xi, t):
        src = self.tfsf
        tau = t - src.sign * (xi - src._xi0) / src.v
        return src.amplitude * np.asarray(src.src_time(tau), dtype=float)

    def update(self, solver, t):
        if not self.ready:
            self._setup(solver)
        src = self.tfsf
        M, J = self.sheets["M"], self.sheets["J"]
        lo = src._idx[src.axis][0]
        if self.mode == "discrete":
            off = src._offset
            e = np.asarray(src._e.get() if hasattr(src._e, "get") else src._e)
            # K of the faces normal to k sits on the dual plane half a cell
            # outside the face and carries E_inc of the face node (lo face:
            # dual index i_f - 1 -> node i_f); on the faces normal to H_inc,
            # K = H_a samples on the a-nodes carry E_inc of the same node
            shift = np.where(M["on_mid"] & (M["line"] == lo - 1), 1, 0)
            vals_M = e[M["line"] + off + shift]
        else:
            vals_M = self._continuum_E(M["xi"], t)
        new_M = M["w"] * vals_M
        solver.M.array[M["flat"]] += new_M - self.old["M"]
        self.old["M"] = new_M

        src.update(solver, t)  # advance the line (corrections switched off)

        if self.mode == "discrete":
            h = np.asarray(src._h.get() if hasattr(src._h, "get") else src._h)
            # J of the faces normal to k (E_p on the face node) carries H_inc
            # of the dual sample outside (lo face: i_f - 1, hi face: i_f); on
            # the faces normal to E_inc, J = E_a samples on the a-mids carry
            # H_inc of the same mid
            shift = np.where(~J["on_mid"] & (J["line"] == lo), -1, 0)
            vals_J = h[J["line"] + off + shift]
        else:
            vals_J = src.h_factor * self._continuum_E(J["xi"], t + solver.dt / 2)
        new_J = J["w"] * vals_J
        solver.J.array[J["flat"]] += new_J - self.old["J"]
        self.old["J"] = new_J


def _pos(nodes, idx, on_mid):
    return 0.5 * (nodes[idx] + nodes[idx + 1]) if on_mid else nodes[idx]


# --------------------------------------------------------------------------
# Part A: leakage
# --------------------------------------------------------------------------
def leakage_case(args):
    cells_per_lambda, mode, record_map = args
    import wakis
    from wakis.sources import GaussianSource

    lam = C0 / F0
    h = lam / cells_per_lambda
    n_box = int(round(0.5 * lam / h))  # TF box 0.5 lambda
    n_pml = 10
    n_gap = 6
    n = n_box + 2 * n_gap + 2 * n_pml
    half = n * h / 2
    grid = wakis.GridFIT3D(-half, half, -half, half, -half, half, n, n, n, verbose=0)
    solver = wakis.SolverFIT3D(
        grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=n_pml, verbose=0
    )
    st = GaussianSource(F0, 0.3 * F0, cutoff=8)
    src = wakis.TFSFSource(st, size=(n_box * h,) * 3, direction="+x", polarization="z")
    src.prepare(solver)
    b = src.box_bounds()[0][1]
    freqs = np.linspace(0.6, 1.4, 9) * F0
    scat = wakis.FluxMonitor(freqs, size=(2 * (b + 3 * h),) * 3)
    driver = src if mode == "tfsf" else HuygensBox(src, mode.replace("sheets_", ""))
    Nt = int((st.end_time + 3 * n * h / C0) / solver.dt)

    mask = src.total_field_mask(solver, "E", flat=True)
    peak = {"sf": 0.0, "tf": 0.0}
    snap = {}
    j0 = solver.Ny // 2

    def cb(s, t):
        E = s.E.array
        a = np.abs(E)
        peak["sf"] = max(peak["sf"], a[~mask].max())
        peak["tf"] = max(peak["tf"], a[mask].max())
        if record_map:
            Ez = E[2 * s.N : 3 * s.N].reshape(s.Nx, s.Ny, s.Nz, order="F")[:, j0, :]
            snap["max"] = np.maximum(snap.get("max", 0 * Ez), np.abs(Ez))

    t0 = time.time()
    solver.emsolve(Nt, source=[driver], monitors=[scat], callback=cb)
    wall = time.time() - t0
    inc = src.incident_flux_density(freqs) * (n_box * h) ** 2
    out = {
        "cells_per_lambda": cells_per_lambda,
        "mode": mode,
        "n": n,
        "Nt": Nt,
        "wall_s": wall,
        "freqs": freqs.tolist(),
        "leak_power": (scat.get_flux() / inc).tolist(),
        "leak_peak": peak["sf"] / peak["tf"],
    }
    if record_map:
        out["map"] = (snap["max"] / peak["tf"]).tolist()
        out["map_x"] = (grid.x[: solver.Nx]).tolist()  # Ez: nodes along x
        out["map_z"] = (
            0.5 * (grid.z[: solver.Nz] + grid.z[1 : solver.Nz + 1])
        ).tolist()  # mids along z
        out["box"] = [-b, b]
    return out


# --------------------------------------------------------------------------
# Part B: Mie
# --------------------------------------------------------------------------
def mie_qsca(m, x):
    """Mie scattering efficiency (Bohren & Huffman)."""
    try:
        from wakis.analytic.mie import mie_efficiencies

        return float(mie_efficiencies(m, x)[1])
    except Exception:
        pass
    from scipy.special import spherical_jn, spherical_yn

    n = np.arange(1, int(x + 4 * x ** (1 / 3) + 3) + 1)

    def psi(z):
        return z * spherical_jn(n, z)

    def dpsi(z):
        return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)

    def xi(z):
        return z * (spherical_jn(n, z) + 1j * spherical_yn(n, z))

    def dxi(z):
        hh = spherical_jn(n, z) + 1j * spherical_yn(n, z)
        dh = spherical_jn(n, z, True) + 1j * spherical_yn(n, z, True)
        return hh + z * dh

    mx = m * x
    an = (m * psi(mx) * dpsi(x) - psi(x) * dpsi(mx)) / (
        m * psi(mx) * dxi(x) - xi(x) * dpsi(mx)
    )
    bn = (psi(mx) * dpsi(x) - m * psi(x) * dpsi(mx)) / (
        psi(mx) * dxi(x) - m * xi(x) * dpsi(mx)
    )
    return float(2 / x**2 * np.sum((2 * n + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2)))


RADIUS = 0.035
EPS_SPHERE = 4.0


def mie_case(args):
    n, conformal = args
    import pyvista as pv

    import wakis
    from wakis.sources import GaussianSource

    length = 0.4
    h = length / n
    n_pml = max(8, int(round(0.06 / h)))
    sphere = pv.Sphere(radius=RADIUS, theta_resolution=120, phi_resolution=120)
    grid = wakis.GridFIT3D(
        -length / 2, length / 2, -length / 2, length / 2, -length / 2, length / 2,
        n, n, n,
        stl_solids={"s": sphere},
        stl_materials={"s": [EPS_SPHERE, 1.0, 0.0]},
        conformal=conformal,
        verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid,
        bc_low=["cpml"] * 3,
        bc_high=["cpml"] * 3,
        n_pml=n_pml,
        use_stl=True,
        verbose=0,
    )
    st = GaussianSource(1.5e9, 0.6e9, cutoff=8)
    src = wakis.TFSFSource(st, size=(0.12,) * 3, direction="+x", polarization="z")
    src.prepare(solver)
    b = src.box_bounds()[0][1]
    freqs = np.linspace(0.4e9, 2.5e9, 22)
    scat = wakis.FluxMonitor(freqs, size=(2 * (b + 3 * h),) * 3)
    tot = wakis.FluxMonitor(freqs, size=(2 * (b - h),) * 3)
    Nt = int(25e-9 / solver.dt)
    t0 = time.time()
    solver.emsolve(Nt, source=[src], monitors=[scat, tot])
    inc = src.incident_flux_density(freqs)
    return {
        "n": n,
        "conformal": bool(conformal),
        "h": h,
        "Nt": Nt,
        "wall_s": time.time() - t0,
        "freqs": freqs.tolist(),
        "sigma_sca": (scat.get_flux() / inc).tolist(),
        "net_tf_over_face": (tot.get_flux() / (inc * (2 * (b - h)) ** 2)).tolist(),
    }


# --------------------------------------------------------------------------
def run(quick, procs):
    if quick:
        leak = [
            (c, m, c == 10)
            for c in (10, 14)
            for m in ("tfsf", "sheets_discrete", "sheets_continuum", "sheets_face")
        ]
        mie = [(32, False), (32, True)]
    else:
        res = (10, 14, 20, 28, 40, 56)
        leak = [
            (c, m, c == 20)
            for c in res
            for m in ("tfsf", "sheets_discrete", "sheets_continuum", "sheets_face")
        ]
        mie = [(n, conf) for n in (32, 48, 64, 80, 96, 128) for conf in (False, True)]
    jobs = [("leak", j) for j in leak] + [("mie", j) for j in mie]
    # largest first for load balance
    jobs.sort(key=lambda j: -(j[1][0] if j[0] == "mie" else 2 * j[1][0]))
    with Pool(procs) as pool:
        results = pool.map(_dispatch, jobs, chunksize=1)
    out = {
        "leak": [r for k, r in results if k == "leak"],
        "mie": [r for k, r in results if k == "mie"],
    }
    with open(
        os.path.join(HERE, "results_quick.json" if quick else "results.json"), "w"
    ) as fh:
        json.dump(out, fh)
    for r in out["leak"]:
        print(
            f"leak {r['mode']:>17s} {r['cells_per_lambda']:3d} c/l: peak {r['leak_peak']:.2e} power@f0 {r['leak_power'][4]:.2e} ({r['wall_s']:.0f}s)"
        )
    for r in out["mie"]:
        print(f"mie n={r['n']} conformal={r['conformal']}: {r['wall_s']:.0f}s")


def _dispatch(job):
    kind, args = job
    return kind, (leakage_case(args) if kind == "leak" else mie_case(args))


def plot(quick):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(
        os.path.join(HERE, "results_quick.json" if quick else "results.json")
    ) as fh:
        res = json.load(fh)
    labels = {
        "tfsf": "TFSFSource (1D FIT line, commutator)",
        "sheets_discrete": "face sheets, discrete line values",
        "sheets_continuum": "face sheets, continuum field, exact registration",
        "sheets_face": "face sheets, continuum, J and K referenced to the face plane (Meep-style)",
    }
    colors = {
        "tfsf": "k",
        "sheets_discrete": "tab:green",
        "sheets_continuum": "tab:blue",
        "sheets_face": "tab:red",
    }
    floor = 1e-32

    # Figure 1: leakage maps
    maps = [r for r in res["leak"] if "map" in r]
    order = ["tfsf", "sheets_continuum", "sheets_face"]
    maps = sorted(
        [m for m in maps if m["mode"] in order], key=lambda m: order.index(m["mode"])
    )
    fig, axes = plt.subplots(
        1, len(maps), figsize=(4.6 * len(maps), 4.2), constrained_layout=True
    )
    for ax, m in zip(np.atleast_1d(axes), maps):
        x, z = np.array(m["map_x"]), np.array(m["map_z"])
        data = np.log10(np.maximum(np.array(m["map"]), 1e-17))
        im = ax.pcolormesh(
            x, z, data.T, cmap="magma", vmin=-16, vmax=0, shading="nearest"
        )
        b = m["box"]
        ax.plot(
            [b[0], b[1], b[1], b[0], b[0]], [b[0], b[0], b[1], b[1], b[0]], "c--", lw=1
        )
        ax.set_aspect("equal")
        ax.set_title(
            f"{labels[m['mode']]}\nmax SF/TF = {m['leak_peak']:.1e}", fontsize=8.5
        )
        ax.set_xlabel("x [m] (propagation)")
        ax.set_ylabel("z [m] (polarization)")
    fig.colorbar(im, ax=axes, label=r"$\log_{10}\,\max_t |E_z| / \max_t|E_z^{TF}|$")
    fig.suptitle(
        f"TF/SF leakage, empty domain, {maps[0]['cells_per_lambda']} cells per wavelength at {F0 / 1e9:.0f} GHz"
    )
    fig.savefig(os.path.join(HERE, "tfsf_leakage_map.png"), dpi=150)
    plt.close(fig)

    # Figure 2: leakage vs resolution and frequency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for mode in labels:
        rows = sorted(
            [r for r in res["leak"] if r["mode"] == mode],
            key=lambda r: r["cells_per_lambda"],
        )
        if not rows:
            continue
        cpl = np.array([r["cells_per_lambda"] for r in rows])
        p0 = np.array([max(abs(r["leak_power"][4]), floor) for r in rows])
        pk = np.array([max(r["leak_peak"], 1e-17) for r in rows])
        ax1.loglog(
            cpl, p0, "o-", color=colors[mode], label=labels[mode] + " (power, f0)"
        )
        ax1.loglog(cpl, pk**2, "s:", color=colors[mode], alpha=0.6)
        mid = [r for r in rows if r["cells_per_lambda"] == 20] or rows[:1]
        f = np.array(mid[0]["freqs"]) / 1e9
        ax2.semilogy(
            f,
            np.maximum(np.abs(mid[0]["leak_power"]), floor),
            "o-",
            color=colors[mode],
            label=labels[mode],
        )
    for mode, order_, ls in (("sheets_face", 2, "--"), ("sheets_continuum", 4, "-.")):
        rows = sorted(
            [r for r in res["leak"] if r["mode"] == mode],
            key=lambda r: r["cells_per_lambda"],
        )
        if rows:
            c = np.array(
                [rows[0]["cells_per_lambda"], rows[-1]["cells_per_lambda"]], float
            )
            y0 = abs(rows[0]["leak_power"][4]) * 3
            ax1.loglog(
                c,
                y0 * (c / c[0]) ** (-order_),
                "0.4",
                ls=ls,
                lw=1,
                label=f"slope -{order_}",
            )
    ax1.axhspan(1e-34, 1e-28, color="0.9", zorder=0)
    ax1.text(10.5, 3e-28, "float64 round-off", fontsize=7, color="0.4")
    ax1.set_xlabel("cells per wavelength at f0")
    ax1.set_ylabel("leaked power / incident power on a face")
    ax1.set_title(
        "Leakage vs resolution (circles: power at f0; squares: (peak |E| ratio)$^2$)",
        fontsize=9,
    )
    ax1.legend(fontsize=6.5, loc="center right")
    ax1.grid(True, which="both", alpha=0.3)
    ax2.set_xlabel("frequency [GHz]")
    ax2.set_ylabel("leaked power / incident power on a face")
    ax2.set_title("Leakage spectrum at 20 cells per wavelength (f0)", fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)
    fig.savefig(os.path.join(HERE, "tfsf_leakage_vs_resolution.png"), dpi=150)
    plt.close(fig)

    # Figure 3: Mie
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    rows = sorted(res["mie"], key=lambda r: (r["conformal"], r["n"]))
    f = np.array(rows[0]["freqs"])
    k = 2 * np.pi * f / C0
    fine_f = np.linspace(f[0], f[-1], 300)
    fk = 2 * np.pi * fine_f / C0
    area = np.pi * RADIUS**2
    q_mie = np.array([mie_qsca(np.sqrt(EPS_SPHERE), kk * RADIUS) for kk in fk])
    q_mie_s = np.array([mie_qsca(np.sqrt(EPS_SPHERE), kk * RADIUS) for kk in k])
    ax1.plot(fk * RADIUS, q_mie, "k-", lw=2, label="Mie")
    cmap = plt.get_cmap("viridis")
    conf_rows = [r for r in rows if r["conformal"]]
    for i, r in enumerate(conf_rows):
        ax1.plot(
            k * RADIUS,
            np.array(r["sigma_sca"]) / area,
            "o--",
            ms=3,
            color=cmap(i / max(1, len(conf_rows) - 1)),
            label=f"conformal, a/h = {RADIUS / r['h']:.1f}",
        )
    ax1.set_xlabel("size parameter ka")
    ax1.set_ylabel(r"$Q_{sca} = \sigma_{sca}/\pi a^2$")
    ax1.set_title(
        r"Dielectric sphere $\varepsilon_r$ = 4, TFSFSource, $\sigma_{sca} = \Phi_{scat}/\Phi_{inc}$",
        fontsize=9,
    )
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    sel = [np.argmin(np.abs(k * RADIUS - x)) for x in (0.5, 1.0, 1.5)]
    for conf, marker in ((False, "s"), (True, "o")):
        rr = [r for r in rows if r["conformal"] == conf]
        apr = np.array([RADIUS / r["h"] for r in rr])
        for j, s_ in enumerate(sel):
            err = np.array(
                [abs(r["sigma_sca"][s_] / area / q_mie_s[s_] - 1) for r in rr]
            )
            ax2.loglog(
                apr,
                err,
                marker + "-",
                color=f"C{j}",
                alpha=1.0 if conf else 0.5,
                label=f"{'conformal' if conf else 'staircase'}, ka={k[s_] * RADIUS:.2f}",
            )
    a_ = np.array([3.0, 12.0])
    ax2.loglog(a_, 0.3 * (a_ / 3) ** -1, "0.5", ls="--", lw=1, label="slope -1")
    ax2.loglog(a_, 0.1 * (a_ / 3) ** -2, "0.5", ls="-.", lw=1, label="slope -2")
    ax2.set_xlabel("cells per radius a/h")
    ax2.set_ylabel(r"$|\sigma_{sca}/\sigma_{Mie} - 1|$")
    ax2.set_title("Convergence to Mie theory", fontsize=9)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, which="both", alpha=0.3)
    fig.savefig(os.path.join(HERE, "tfsf_mie_convergence.png"), dpi=150)
    plt.close(fig)
    print("figures written to", HERE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "plot"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--procs", type=int, default=os.cpu_count())
    a = parser.parse_args()
    if a.action == "run":
        run(a.quick, a.procs)
    else:
        plot(a.quick)
