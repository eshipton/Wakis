"""
What sets the ~4.5e-4 "floor" of the tetrahedral spherical PML?

Hypotheses (open_questions.md, 2026-11-13) and the run that separates them:

H_ref    the reference (ball + 40 *vacuum* prism layers + PML) is not
         reflectionless: its tangentially coarsening shell reflects, and the
         echo arrives inside the window. Predicts: all PML variants agree with
         each other far below 4.5e-4 while every one differs from ref40 by
         ~4.5e-4; the difference ref40 - (thick PML) is spread over the time
         of flight through r0 < r < ~30 h, not concentrated at the backing echo.
H_back   the field that crosses the layer and reflects off the PEC backing
         (e.g. a quasi-static part that a PML does not attenuate). Predicts:
         |E(PEC backing) - E(PMC backing)| / 2 ~ the floor (the backing echo
         changes sign with the wall type).
H_grade  per-cell discretisation of the grading. Predicts: splitting every
         layer into two (dr = h/2, same thickness and profile) lowers it.
H_design the design reflection R = exp(-2 sigma_max d / ((m+1) c)) = 1e-8 is
         not reached. Predicts: R = 1e-12 lowers it.

All runs share one dt and the window of the original study (round trip
through 16 layers). Reflection metric as `pml3d_reflection.py`:
max_t |E_X - E_B| / max_t |E_B| per probe, for baselines B = ref40 and the
32-layer PML (whose backing echo arrives after the window).

Usage
-----
    FLOOR_TAG=_fixed python pml3d_floor.py run [n_procs]   # floor_fixed.json, floor_raw_fixed.npz
    python pml3d_floor.py plot            # docs/img/unstructured_pml3d_floor*.png
"""

import json
import os
import sys
import time

import numpy as np
from scipy.constants import c as C0

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pml3d_reflection as R  # noqa: E402

H, RB = R.H, R.RB
N_WINDOW = 16  # window of the original study
# "_before": massless decision per chord (as before the fix: more chords of a
# column turn massless at larger r); "_fixed": one decision per radial column
TAG = os.environ.get("FLOOR_TAG", "_fixed")


def _mesh(n_extra, n_layers, dr=None):
    from wakis.unstructured.pml3d import ball_with_pml_mesh

    m, cfg = ball_with_pml_mesh(RB, H, n_layers, dr=dr, n_extra=n_extra)
    if TAG == "_before":
        m.massless_area_factor = np.ones(m.n_edges)
    return m, cfg


CASES = {
    # label: (n_extra, n_layers, dr, bc, pml options)
    "ref40": (40, 16, None, "pec", {}),
    "pml4": (0, 4, None, "pec", {}),
    "pml6": (0, 6, None, "pec", {}),
    "pml10": (0, 10, None, "pec", {}),
    "pml16": (0, 16, None, "pec", {}),
    "pml24": (0, 24, None, "pec", {}),
    "pml32": (0, 32, None, "pec", {}),
    "pml10_pmc": (0, 10, None, "pmc", {}),
    "pml16_pmc": (0, 16, None, "pmc", {}),
    "pml10_half": (0, 20, 0.5 * H, "pec", {}),  # same thickness 10 h, dr = h / 2
    "pml16_R12": (0, 16, None, "pec", {"R": 1e-12}),
    "pml10_R12": (0, 10, None, "pec", {"R": 1e-12}),
    # thick (24 h) layers with other radial spacings: baselines for dr != h
    # and the tetrahedra -> prism transition (the shell geometry seen at r0)
    "half_thick": (0, 48, 0.5 * H, "pec", {}),
    "dr075_thick": (0, 32, 0.75 * H, "pec", {}),
    "dr15_thick": (0, 16, 1.5 * H, "pec", {}),
}
BASELINES = ("ref40", "pml32", "pml24", "half_thick")


def common_dt():
    from wakis.unstructured import UnstructuredFIT3D, ball_mesh

    dts = [UnstructuredFIT3D(ball_mesh([RB], H), bc="abc").dt_max]
    for lab in ("ref40", "pml10_half"):
        ne, nl, dr, bc, _ = CASES[lab]
        m, _ = _mesh(ne, nl, dr)
        dts.append(UnstructuredFIT3D(m, bc=bc).dt_max)
    return 0.9 * min(dts)


def run_case(args):
    from wakis.unstructured import UnstructuredFIT3D

    label, kind, dt = args
    ne, nl, dr, bc, opts = CASES[label]
    t0 = time.time()
    m, cfg = _mesh(ne, nl, dr)
    s = UnstructuredFIT3D(m, bc=bc, dt=dt, pml=dict(cfg, **opts))
    if s.dt_max < dt:
        raise RuntimeError(f"{label}: dt {dt} above the limit {s.dt_max}")
    src, d, probes, _ = R.geometry(kind)
    s.add_point_dipole(src, d, R.amp)
    P = R.whitney_probe(s.mesh, probes)
    n = int(round(R.window(kind, N_WINDOW) / dt))
    sig = np.empty((n, len(probes), 3))
    for k in range(n):
        s.one_step()
        sig[k] = (P @ s._pml.edge_voltages()).reshape(len(probes), 3)
    print(f"{kind} {label}: {m.n_cells} cells, {time.time() - t0:.0f} s", flush=True)
    return label, kind, sig, m.n_cells, time.time() - t0


def run(n_procs=1):
    from multiprocessing import Pool

    dt = common_dt()
    jobs = [(lab, kind, dt) for kind in ("normal", "oblique") for lab in CASES]
    # longest first
    jobs.sort(key=lambda j: -(CASES[j[0]][0] + CASES[j[0]][1]))
    t = time.time()
    with Pool(n_procs) as pool:
        res = pool.map(run_case, jobs, chunksize=1)
    print(f"{len(jobs)} runs in {time.time() - t:.0f} s on {n_procs} processes")
    sig = {(lab, kind): s for lab, kind, s, _, _ in res}
    meta = {(lab, kind): (nc, w) for lab, kind, _, nc, w in res}
    out = dict(dt=dt, h=H, RB=RB, window_layers=N_WINDOW, cases={})
    for kind in ("normal", "oblique"):
        _, _, probes, angles = R.geometry(kind)
        ent = dict(probes=probes.tolist(), angles_deg=list(map(float, angles)), runs={})
        for base in BASELINES:
            B = sig[(base, kind)]
            nb = np.linalg.norm(B, axis=-1).max(0)
            for lab in CASES:
                if lab == base:
                    continue
                X = sig[(lab, kind)]
                diff = np.linalg.norm(X - B, axis=-1)  # (n_t, n_probe)
                Rt = diff.max(0) / nb
                ent["runs"].setdefault(lab, {})[f"vs_{base}"] = Rt.tolist()
        for lab in CASES:
            ent["runs"].setdefault(lab, {})["n_cells"], ent["runs"][lab]["wall_s"] = meta[(lab, kind)]
        # backing echo: (PEC - PMC) / 2
        for n in (10, 16):
            diff = np.linalg.norm(sig[(f"pml{n}", kind)] - sig[(f"pml{n}_pmc", kind)], axis=-1) / 2
            nb = np.linalg.norm(sig[("pml32", kind)], axis=-1).max(0)
            ent[f"backing_echo_pml{n}"] = (diff.max(0) / nb).tolist()
        out["cases"][kind] = ent
        print(kind)
        for lab, e in ent["runs"].items():
            print(f"  {lab:12s}", {k: np.round(v, 7).tolist() for k, v in e.items() if k.startswith("vs")})
        print("  backing echo 10 / 16:", ent["backing_echo_pml10"], ent["backing_echo_pml16"])
    with open(os.path.join(HERE, f"floor{TAG}.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(HERE, f"floor_raw{TAG}.npz"), **{f"{k}__{lab}": v for (lab, k), v in sig.items()})


def _massless_per_ring():
    """Massless chords per shell ring, per chord vs per column (small mesh:
    ball 4 h + 30 layers, the geometry of tests/test_098)."""
    from wakis.unstructured import UnstructuredFIT3D
    from wakis.unstructured.pml3d import ball_with_pml_mesh

    m, cfg = ball_with_pml_mesh(4.0, H, 6, n_extra=24)
    rn = np.linalg.norm(m.nodes, axis=1)
    ra, rb = rn[m.edges].T
    radii = m.shell_radii[1:-1]
    out = {}
    for name in ("per column", "per chord"):
        if name == "per chord":
            m.massless_area_factor = np.ones(m.n_edges)
        ml = np.zeros(m.n_edges, bool)
        ml[UnstructuredFIT3D(m, bc="pec").massless] = True
        out[name] = [int(ml[(np.abs(ra - r) < 1e-6) & (np.abs(rb - r) < 1e-6)].sum()) for r in radii]
    return radii, out


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_img = os.path.join(HERE, "..", "..", "docs", "img")
    S = {k: json.load(open(os.path.join(HERE, f"floor_{k}.json"))) for k in ("before", "fixed")}
    Z = {k: np.load(os.path.join(HERE, f"floor_raw_{k}.npz")) for k in ("before", "fixed")}

    # 1. reflection vs layers, before / after the per-column massless decision
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True)
    Ns = [4, 6, 10, 16, 24, 32]
    for ax, kind in zip(axs, ("normal", "oblique")):
        for k, ls, mk in (("before", "--", "s"), ("fixed", "-", "o")):
            e = S[k]["cases"][kind]["runs"]
            y = [max(e[f"pml{n}"]["vs_ref40"]) if n != 32 else max(e["ref40"]["vs_pml32"]) for n in Ns[:-1]]
            ax.loglog(Ns[:-1], y, mk + ls, color="C0" if k == "fixed" else "0.5",
                      label=("massless per column (fixed)" if k == "fixed" else "massless per chord (before)")
                      + ": PML vs vacuum-shell ref40")  # fmt: skip
            ref_err = max(e["ref40"]["vs_pml32"])
            ax.axhline(ref_err, color="C3" if k == "fixed" else "0.6", lw=1, ls=":" if k == "fixed" else "-.",
                       label=f"ref40 vs PML32 ({k}): {ref_err:.1e}")  # fmt: skip
        e = S["fixed"]["cases"][kind]["runs"]
        ax.loglog([10, 16], [max(e["pml10_R12"]["vs_ref40"]), max(e["pml16_R12"]["vs_ref40"])], "^", color="C4",
                  ms=7, label=r"$R_{design}=10^{-12}$ (fixed)")  # fmt: skip
        ax.loglog([10], [max(e["pml10_half"]["vs_half_thick"])], "v", color="C2", ms=7,
                  label="10 h at dr = h/2 vs 24 h at dr = h/2 (fixed)")  # fmt: skip
        ax.loglog([10, 16], [max(S["fixed"]["cases"][kind][f"backing_echo_pml{n}"]) for n in (10, 16)], "x",
                  color="k", ms=8, label=r"backing echo $|E_{PEC}-E_{PMC}|/2$")  # fmt: skip
        ax.set_xlabel("PML layers N (dr = h, $R_{design} = 10^{-8}$, m = 3)")
        ax.set_ylabel(r"$\max_t |E_N - E_{ref}| / \max_t |E_{ref}|$ (worst probe)")
        ax.set_title(f"{kind} incidence (ball {RB:g} h)")
        ax.set_ylim(1e-8, 0.1)
        ax.set_xticks(Ns[:-1], [str(n) for n in Ns[:-1]])
        ax.minorticks_off()
        ax.grid(alpha=0.3, which="both")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=7)
    fig.suptitle("Tetrahedral spherical PML: the 4.5e-4 floor was the reference shell")
    fig.savefig(os.path.join(out_img, "unstructured_pml3d_floor_layers.png"), dpi=150)
    plt.close(fig)

    # 2. (a) time-resolved reference error, (b) massless chords per ring
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    dt = S["fixed"]["dt"]
    ip = 2
    for k, col in (("before", "0.5"), ("fixed", "C0")):
        z = Z[k]
        nb = np.linalg.norm(z["normal__pml32"], axis=-1)[:, ip].max()
        t = (np.arange(z["normal__pml32"].shape[0]) + 1) * dt * C0 / H
        d = np.linalg.norm(z["normal__ref40"] - z["normal__pml32"], axis=-1)[:, ip] / nb
        axs[0].semilogy(t, np.maximum(d, 1e-12), color=col, lw=1.2, label=f"ref40 - PML32 ({k})")
        d = np.linalg.norm(z["normal__pml10"] - z["normal__pml32"], axis=-1)[:, ip] / nb
        axs[0].semilogy(t, np.maximum(d, 1e-12), color=col, lw=0.8, ls="--", label=f"PML10 - PML32 ({k})")
    T0 = R.T0 * C0 / H
    rp = RB - 1.0
    for r, name in ((RB, "$r_0$"), (RB + 10, "$r_0$ + 10 h")):
        axs[0].axvline(T0 + 2 * r - rp, color="0.5", lw=0.7, ls=":")
        axs[0].text(T0 + 2 * r - rp, 2e-11, " echo from " + name, rotation=90, fontsize=7, color="0.3", va="bottom")
    axs[0].set_ylim(1e-11, 1e-2)
    axs[0].set_xlabel("time  c t / h")
    axs[0].set_ylabel(r"$|E_X - E_{PML32}|(t)\,/\,\max_t |E_{PML32}|$")
    axs[0].set_title(f"normal incidence, probe r = {rp:g} h, θ = 30°")
    axs[0].legend(fontsize=7)
    radii, cnt = _massless_per_ring()
    for name, mk in (("per chord", "s--"), ("per column", "o-")):
        axs[1].plot(radii, cnt[name], mk, ms=3, label=f"massless decision {name}")
    axs[1].set_xlabel("ring radius r / h (ball 4 h, 30 layers of dr = h)")
    axs[1].set_ylabel("massless chords on the ring")
    axs[1].set_title(r"area < tol $L^2$: area $\propto r\,dr$, $L^2 \propto r^2$")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(out_img, "unstructured_pml3d_floor_mechanism.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    else:
        plot()
