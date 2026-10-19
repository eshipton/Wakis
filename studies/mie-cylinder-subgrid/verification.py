"""
Verification of the subgrid interface (figure e_verification.png and
data/verification.json): plane-wave reflection of a straight 2:1 / 3:1
interface vs angle and resolution (with the first-order 'inject' mutation),
cavity eigenfrequency convergence, and the discrete energy over 1e5 steps.

Small problems only (< 1 min on one core).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as sla  # noqa: E402
from scipy.constants import c as C0  # noqa: E402

from wakis.subgrid import Circle, Geometry, QuadMesh, Rect, SubgridFIT2D, refine_near  # noqa: E402
from wakis.subgrid.interface import _inject, bloch_reflection  # noqa: E402

HERE = Path(__file__).resolve().parent
NS = [10, 14, 20, 28, 40]
ANGLES = [0, 15, 30, 45, 60, 75]


def reflection():
    out = {}
    for th in ANGLES:
        out[f"r2_{th}"] = [bloch_reflection(n, th) for n in NS]
        out[f"r3_{th}"] = [bloch_reflection(n, th, ratio=3) for n in NS]
        out[f"inj_{th}"] = [bloch_reflection(n, th, mutate="inject") for n in NS]
        out[f"floor_{th}"] = [bloch_reflection(n, th, refine=False) for n in NS]
    return out


def eigen():
    a, b = 1.0, 0.7
    out = {}
    for pol in ("TM", "TE"):
        for mode in ("coarse", "patch", "inject", "fine"):
            errs, unk = [], []
            for N in (10, 20, 40, 80):
                NN = 2 * N if mode == "fine" else N
                x = np.linspace(0, a, NN + 1)
                y = np.linspace(0, b, int(round(NN * b / a)) + 1)
                if mode in ("patch", "inject"):
                    f = refine_near([lambda xm, ym, hd: (np.abs(xm - 0.37) < 0.18)
                                     & (np.abs(ym - 0.31) < 0.13)])  # fmt: skip
                    m = QuadMesh(x, y, ratio=2, max_level=1, refine=f)
                    if mode == "inject":
                        m.P = _inject(m, 1.0).real.tocsr()
                else:
                    m = QuadMesh(x, y)
                s = SubgridFIT2D(m, polarization=pol)
                K, D = s.stiffness()
                lam = np.sort(
                    sla.eigsh(K, k=8, M=sp.diags(D), sigma=-1.0, which="LM")[0]
                )
                lo = 1 if pol == "TM" else 0
                ex = sorted(C0 / 2 * np.hypot(i / a, j / b)
                            for i in range(lo, 8) for j in range(lo, 8) if i + j > 0)  # fmt: skip
                fr = np.sqrt(np.abs(lam)) / (2 * np.pi)
                fr = fr[fr > 1e3][:5]
                errs.append(float(np.abs(fr / np.array(ex[:5]) - 1).max()))
                unk.append(int(s.n_unknowns))
            out[f"{pol}_{mode}"] = dict(err=errs, unknowns=unk)
    return out


def energy():
    x = np.linspace(0, 1e-6, 17)
    y = np.linspace(0, 0.8e-6, 14)
    f = refine_near([
        lambda xm, ym, hd: np.hypot(xm - 0.45e-6, ym - 0.4e-6) < 0.25e-6,
        lambda xm, ym, hd: np.hypot(xm - 0.45e-6, ym - 0.4e-6) < 0.12e-6,
    ])  # fmt: skip
    m = QuadMesh(x, y, ratio=2, max_level=2, refine=f)
    geo = Geometry([Circle((0.5e-6, 0.42e-6), 0.2e-6, 4.0),
                    Rect(0.2e-6, 0.3e-6, 0.1e-6, 0.7e-6, 2.25)])  # fmt: skip
    out = {}
    for pol in ("TM", "TE"):
        for case in ("symmetric", "inject"):
            s = SubgridFIT2D(
                m, geo, polarization=pol, cfl=0.99 if case == "symmetric" else 0.5
            )
            step = s.one_step
            if case == "inject":
                keep = np.isin(m.free, s.dof)
                KQt = (
                    (m.gradient() @ _inject(m, 1.0).real[:, keep])
                    .tocsr()[s.edge_idx]
                    .T.tocsr()
                )

                def step(s=s, KQt=KQt):
                    f1 = s.f - s.dt * (s.K @ s.u)
                    s.u_prev, s.u = s.u, s.u + s.dt * (KQt @ (s.W * f1)) / s.D
                    s.f_prev, s.f = s.f, f1

            s.u = np.random.default_rng(0).standard_normal(s.n_dof)
            step()
            e0 = s.energy()
            n_, e_ = [], []
            for n in range(100_000):
                step()
                if n % 500 == 0:
                    e = s.energy()
                    n_.append(n)
                    e_.append(float(e / e0 - 1))
                    if not np.isfinite(e) or abs(e / e0) > 1e12:
                        break
            out[f"{pol}_{case}"] = dict(n=n_, drift=e_, unknowns=int(s.n_unknowns),
                                        levels=m.level_counts().tolist())  # fmt: skip
    return out


def plot(d):
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.9))
    ax = axs[0]
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(ANGLES)))
    for c, th in zip(cols, ANGLES):
        ax.loglog(NS, d["refl"][f"r2_{th}"], "o-", color=c, ms=4, label=f"{th}°")
    ax.loglog(NS, 2 * d["refl"]["r2_0"][0] * (NS[0] / np.array(NS)) ** 2, "k:", lw=0.8,
              label="order 2")  # fmt: skip
    ax.set_xlabel("coarse cells per wavelength")
    ax.set_ylabel("|R| of the 2:1 interface")
    ax.set_title("Wakis FIT (2:1 subgrid): reflection", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    ax = axs[1]
    for key, st, lab in (("r2", "o-", "2:1"), ("r3", "s--", "3:1"),
                         ("inj", "x:", "2:1 inject (mutation)"), ("floor", "^-", "no interface")):  # fmt: skip
        for th, c in ((30, "#0072B2"), (60, "#D55E00")):
            ax.loglog(NS, d["refl"][f"{key}_{th}"], st, color=c, ms=4,
                      label=f"{lab}, {th}°")  # fmt: skip
    ax.set_xlabel("coarse cells per wavelength")
    ax.set_ylabel("|R|")
    ax.set_title("ratio 3:1, first-order mutation, measurement floor", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    ax = axs[2]
    for pol, ls in (("TM", "-"), ("TE", "--")):
        for mode, c, mk in (("coarse", "#999999", "s"), ("patch", "k", "P"),
                            ("fine", "#56B4E9", "o"), ("inject", "#D55E00", "x")):  # fmt: skip
            e = d["eigen"][f"{pol}_{mode}"]
            ax.loglog(e["unknowns"], e["err"], ls, marker=mk, color=c, ms=4,
                      label=f"Wakis FIT ({mode}), {pol}")  # fmt: skip
    ax.set_xlabel("unknowns")
    ax.set_ylabel("max rel. error, 5 lowest modes")
    ax.set_title("PEC cavity 1 x 0.7 with off-centre patch", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    ax = axs[3]
    for key, c in (("TM_symmetric", "k"), ("TE_symmetric", "#0072B2"),
                   ("TM_inject", "#D55E00"), ("TE_inject", "#E69F00")):  # fmt: skip
        e = d["energy"][key]
        ax.semilogy(e["n"], np.maximum(np.abs(e["drift"]), 1e-17), color=c, lw=1,
                    label=key.replace("_", ", "))  # fmt: skip
    ax.set_xlabel("time step")
    ax.set_ylabel("|E^n / E^0 - 1|")
    ax.set_title("closed cavity, 3 levels, dielectrics on interfaces", fontsize=9)
    ax.legend(fontsize=6)
    for a in axs:
        a.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "e_verification.png")
    plt.close(fig)


if __name__ == "__main__":
    d = dict(refl=reflection(), eigen=eigen(), energy=energy())
    (HERE / "data" / "verification.json").write_text(json.dumps(d))
    plot(d)
    for k in ("r2_0", "r2_30", "r2_60", "inj_30", "floor_30"):
        print(k, ["%.2e" % v for v in d["refl"][k]])
    for k, v in d["eigen"].items():
        print(k, ["%.2e" % e for e in v["err"]])
    for k, v in d["energy"].items():
        print(k, "max drift %.2e" % np.max(np.abs(v["drift"])), "last n", v["n"][-1])
