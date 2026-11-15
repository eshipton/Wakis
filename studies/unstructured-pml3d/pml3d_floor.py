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
    python pml3d_floor.py run [n_procs]   # floor.json, floor_signals.npz
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
    npz = {}
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
                npz[f"{kind}__{lab}__{base}"] = diff / nb
        for lab in CASES:
            ent["runs"].setdefault(lab, {})["n_cells"], ent["runs"][lab]["wall_s"] = meta[(lab, kind)]
        # backing echo: (PEC - PMC) / 2
        for n in (10, 16):
            diff = np.linalg.norm(sig[(f"pml{n}", kind)] - sig[(f"pml{n}_pmc", kind)], axis=-1) / 2
            nb = np.linalg.norm(sig[("pml32", kind)], axis=-1).max(0)
            ent[f"backing_echo_pml{n}"] = (diff.max(0) / nb).tolist()
            npz[f"{kind}__backing{n}"] = diff / nb
        npz[f"{kind}__t"] = (np.arange(sig[("pml32", kind)].shape[0]) + 1) * dt
        out["cases"][kind] = ent
        print(kind)
        for lab, e in ent["runs"].items():
            print(f"  {lab:12s}", {k: np.round(v, 7).tolist() for k, v in e.items() if k.startswith("vs")})
        print("  backing echo 10 / 16:", ent["backing_echo_pml10"], ent["backing_echo_pml16"])
    with open(os.path.join(HERE, f"floor{TAG}.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(HERE, f"floor_signals{TAG}.npz"), **npz)
    np.savez_compressed(os.path.join(HERE, f"floor_raw{TAG}.npz"), **{f"{k}__{l}": v for (l, k), v in sig.items()})


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_img = os.path.join(HERE, "..", "..", "docs", "img")
    S = json.load(open(os.path.join(HERE, "floor.json")))
    Z = np.load(os.path.join(HERE, "floor_signals.npz"))

    # 1. reflection vs layers against the two baselines
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    Ns = [4, 6, 10, 16, 24]
    for ax, kind in zip(axs, ("normal", "oblique")):
        e = S["cases"][kind]["runs"]
        for base, mk, name in (("ref40", "s--", "vs vacuum-shell reference (ref40)"),
                               ("pml32", "o-", "vs 32-layer PML (no echo in window)")):  # fmt: skip
            y = [max(e[f"pml{n}"][f"vs_{base}"]) for n in Ns if f"vs_{base}" in e[f"pml{n}"]]
            ax.loglog(Ns[: len(y)], y, mk, label=name)
        y_ref = max(e["ref40"]["vs_pml32"])
        ax.axhline(y_ref, color="C3", lw=1, ls=":", label=f"ref40 vs PML32: {y_ref:.1e}")
        ax.loglog([10], [max(e["pml10_half"]["vs_pml32"])], "v", color="C2", ms=8, label="10 h, dr = h/2 (20 layers)")
        ax.loglog([10, 16], [max(e["pml10_R12"]["vs_pml32"]), max(e["pml16_R12"]["vs_pml32"])], "^", color="C4",
                  ms=8, label=r"$R_{design}=10^{-12}$")  # fmt: skip
        ax.loglog([10, 16], [max(S["cases"][kind][f"backing_echo_pml{n}"]) for n in (10, 16)], "x", color="k",
                  ms=8, label="backing echo |E_PEC - E_PMC| / 2")  # fmt: skip
        ax.set_xlabel("PML layers N (dr = h unless noted)")
        ax.set_ylabel(r"$\max_t |E_N - E_B| / \max_t |E_B|$ (worst probe)")
        ax.set_title(f"{kind} incidence")
        ax.grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Tetrahedral spherical PML: the 4.5e-4 floor is the reference (ball {RB:g} h)")
    fig.savefig(os.path.join(out_img, "unstructured_pml3d_floor_layers.png"), dpi=150)
    plt.close(fig)

    # 2. time-resolved differences (normal incidence, probe 2)
    t = Z["normal__t"] * C0 / H
    ip = 2
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for key, lab in (("normal__ref40__pml32", "ref40 - PML32 (reference error)"),
                     ("normal__pml10__ref40", "PML10 - ref40"),
                     ("normal__pml10__pml32", "PML10 - PML32"),
                     ("normal__pml16__pml32", "PML16 - PML32"),
                     ("normal__backing10", "backing echo, 10 layers")):  # fmt: skip
        ax.semilogy(t, np.maximum(Z[key][:, ip], 1e-12), lw=1, label=lab)
    T0 = R.T0 * C0 / H
    rp = RB - 1.0
    for r, name in ((RB, "$r_0$"), (RB + 10, "$r_1$ (10)"), (RB + 16, "$r_1$ (16)")):
        ax.axvline(T0 + 2 * r - rp, color="0.5", lw=0.7, ls="--")
        ax.text(T0 + 2 * r - rp, 2e-11, " echo " + name, rotation=90, fontsize=7, color="0.3", va="bottom")
    ax.set_ylim(1e-11, 2)
    ax.set_xlabel("time  c t / h")
    ax.set_ylabel(r"$|E_X - E_B|(t) / \max_t |E_B|$")
    ax.set_title(f"Normal incidence, probe at r = {rp:g} h, θ = 30°: arrival times of the differences")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(out_img, "unstructured_pml3d_floor_time.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    else:
        plot()
