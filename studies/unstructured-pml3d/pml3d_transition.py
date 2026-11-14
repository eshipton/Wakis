"""
Absolute reflection of the tetrahedra -> prism-shell transition at r0.

The PML reflections of `pml3d_reflection.py` / `pml3d_floor.py` are measured
against references with the *same* shell, so the transition from the
unstructured ball to the radially extruded prism shell cancels. Different
radial spacings dr of the shell change the field in the ball by 0.4-3 %
(pml3d_floor.py), so the transition itself reflects. This script measures its
absolute size with the bare PEC ball as the calibration:

* a short pulse from a dipole near the centre, probes near the centre
  (r_p = 1.5 h): the wave returning from the sphere r0 arrives
  2 (r0 - r_p) / c after the direct pulse, separated from it;
* runs: PEC wall at r0 (discrete |R| = 1, lossless), first-order
  Silver-Mueller at r0, and the shell with a 24 h thick PML for several dr
  (its own reflection < 1e-4, its backing echo far outside the window);
* the time-gated echo ratio |E_echo(case)| / |E_echo(PEC)| is the reflection
  coefficient of the boundary as seen from the ball (same focusing geometry,
  same interior dispersion, same source model in every run).

Usage: python pml3d_transition.py run [n_procs] ; python pml3d_transition.py plot
"""

import json
import os
import sys
import time

import numpy as np
from scipy.constants import c as C0

HERE = os.path.dirname(os.path.abspath(__file__))
H, RB = 1.0, 12.0
LAM0 = 8.0
TAU = LAM0 / (np.pi * C0)
T0 = 4 * TAU
RP = 1.5
DRS = [0.5, 0.75, 1.0, 1.25, 1.5]
THICK = 24.0


def amp(t):
    x = (t - T0) / TAU
    return -x * np.exp(-(x**2))


def probes():
    th = np.radians([90, 45, 0])
    return RP * np.stack([np.sin(th), 0 * th, np.cos(th)], 1) + np.array([0.0, 0.05, 0.0])


def build(label, dt=None):
    from wakis.unstructured import UnstructuredFIT3D, ball_mesh
    from wakis.unstructured.pml3d import ball_with_pml_mesh

    if label in ("pec", "abc"):
        return UnstructuredFIT3D(ball_mesh([RB], H), bc=label, dt=dt)
    dr = float(label[2:])
    n = int(round(THICK / dr))
    m, cfg = ball_with_pml_mesh(RB, H, n, dr=dr)
    return UnstructuredFIT3D(m, bc="pec", dt=dt, pml=cfg)


def labels():
    return ["pec", "abc"] + [f"dr{d:g}" for d in DRS]


def common_dt():
    return 0.9 * min(build(lab).dt_max for lab in ("abc", f"dr{min(DRS):g}"))


def run_case(args):
    import sys as _s

    _s.path.insert(0, HERE)
    from pml3d_reflection import whitney_probe

    label, dt, n = args
    t0 = time.time()
    s = build(label, dt)
    s.add_point_dipole([0.31, 0.17, 0.23], [0, 0, 1], amp)
    P = whitney_probe(s.mesh, probes())
    volt = s._pml.edge_voltages if s._pml is not None else (lambda: s.e)
    sig = np.empty((n, 3, 3))
    for k in range(n):
        s.one_step()
        sig[k] = (P @ volt()).reshape(3, 3)
    print(label, s.mesh.n_cells, f"{time.time() - t0:.0f} s", flush=True)
    return label, sig


def gate(t):
    """Echo window: the returning wave between the direct pulse and the second pass."""
    a = T0 + (2 * RB - 2 * RP) / C0 - 3.5 * TAU
    b = T0 + (2 * RB + 2 * RP) / C0 + 3.5 * TAU
    return (t > a) & (t < b), (a, b)


def run(n_procs=7):
    from multiprocessing import Pool

    dt = common_dt()
    T = T0 + (2 * RB + 2 * RP) / C0 + 6 * TAU
    n = int(round(T / dt))
    with Pool(n_procs) as pool:
        res = dict(pool.map(run_case, [(lab, dt, n) for lab in labels()]))
    t = (np.arange(n) + 1) * dt
    g, (a, b) = gate(t)
    # direct pulse must be over when the gate opens
    direct_end = T0 + 3.5 * TAU + RP / C0
    pec = np.linalg.norm(res["pec"][g], axis=-1).max(0)
    out = dict(dt=dt, gate=[a, b], direct_end=direct_end, lam0=LAM0, rp=RP, rb=RB, thick=THICK, R={})
    for lab, sig in res.items():
        R = np.linalg.norm(sig[g], axis=-1).max(0) / pec
        out["R"][lab] = R.tolist()
        print(lab, np.round(R, 5))
    with open(os.path.join(HERE, "transition.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(HERE, "transition_signals.npz"), t=t, **res)


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = json.load(open(os.path.join(HERE, "transition.json")))
    Z = np.load(os.path.join(HERE, "transition_signals.npz"))
    t = Z["t"] * C0 / H
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ip = 0
    for lab, st in (("pec", "k-"), ("abc", "k--"), ("dr0.5", "C0-"), ("dr1", "C1-"), ("dr1.5", "C2-")):
        axs[0].semilogy(t, np.linalg.norm(Z[lab][:, ip], axis=-1) + 1e-12, st, lw=1,
                        label={"pec": "PEC wall at $r_0$", "abc": "Silver-Müller at $r_0$"}.get(lab, f"prism shell dr = {lab[2:]} h + PML"))  # fmt: skip
    a, b = np.array(S["gate"]) * C0 / H
    axs[0].axvspan(a, b, color="0.9", zorder=0, label="echo gate")
    axs[0].set_xlabel("time  c t / h")
    axs[0].set_ylabel(r"$|E|$ at the probe [V/m]")
    axs[0].set_ylim(1e-6 * np.abs(Z["pec"]).max(), 2 * np.abs(Z["pec"]).max())
    axs[0].legend(fontsize=7)
    drs = np.array(DRS)
    R = np.array([max(S["R"][f"dr{d:g}"]) for d in DRS])
    axs[1].plot(drs, R, "o-", label="tetrahedra -> prism shell (+ 24 h PML)")
    axs[1].axhline(max(S["R"]["abc"]), color="k", ls="--", lw=1, label="Silver-Müller at $r_0$")
    axs[1].set_xlabel("radial spacing of the shell dr / h (ball edge h)")
    axs[1].set_ylabel("echo / PEC echo (worst probe)")
    axs[1].set_yscale("log")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"Reflection of the ball -> shell transition at $r_0$ = {RB:g} h (pulse centre λ = {LAM0:g} h)")
    fig.savefig(os.path.join(HERE, "..", "..", "docs", "img", "unstructured_pml3d_transition.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
