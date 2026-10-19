"""
Reflection of the outer boundary of UnstructuredFIT2D: first-order ABC,
matched-loss absorber and the radial CFS-PML, for cylindrical waves.

An off-centre line source (excites harmonics |n| <= ~12) radiates in vacuum;
the field is DFT'd on circles r = 1.2/1.35/1.5 um and fitted as outgoing
H^(1) + incoming H^(2) waves (`wakis.cylinder_harmonics`); the reflection is
sqrt(sum |gamma_n|^2 / sum |alpha_n|^2) at each x = k a (a = 1 um). The
boundary (ABC, absorber start or PML start) is at r0 = 1.65 um.

    python pml_reflection.py run [--procs N]
    python pml_reflection.py plot
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.constants import c as C0

HERE = Path(__file__).resolve().parent
A, R0 = 1e-6, 1.65e-6
X = np.linspace(2, 10, 17)
F = X * C0 / (2 * np.pi * A)
CASES = [
    dict(name="ABC (1st order)", kind="abc", h=40e-9),
    dict(name="ABC (1st order)", kind="abc", h=20e-9),
    dict(name="matched-loss absorber, 0.8 um", kind="absorber", h=40e-9),
    dict(name="PML 5 cells", kind="pml", h=40e-9, nl=5),
    dict(name="PML 10 cells", kind="pml", h=40e-9, nl=10),
    dict(name="PML 20 cells", kind="pml", h=40e-9, nl=20),
    dict(name="PML 5 cells", kind="pml", h=20e-9, nl=5),
    dict(name="PML 10 cells", kind="pml", h=20e-9, nl=10),
    dict(name="PML 20 cells", kind="pml", h=20e-9, nl=20),
    dict(name="PML 10 cells, CFS kappa=3, alpha=2pi 20 THz", kind="pml", h=20e-9, nl=10,
         cfg=dict(kappa_max=3.0, alpha=2 * np.pi * 20e12)),
    dict(name="PML 10 cells (TE)", kind="pml", h=20e-9, nl=10, pol="TE"),
]  # fmt: skip


def run(case):
    from wakis.cylinder_harmonics import circle_points, fit_harmonics, project_harmonics
    from wakis.sources import GaussianSource
    from wakis.unstructured import UnstructuredFIT2D, graded_disk_mesh

    h, pol = case["h"], case.get("pol", "TM")
    if case["kind"] == "abc":
        m = graded_disk_mesh([A, R0], h)
        s = UnstructuredFIT2D(m, {}, polarization=pol, bc="abc")
    elif case["kind"] == "absorber":
        d = 0.8e-6
        m = graded_disk_mesh([A, R0, R0 + d], h)
        s = UnstructuredFIT2D(m, {}, polarization=pol, bc="abc",
                              absorber=dict(r0=R0, r1=R0 + d, s_max=6 * C0 / d))  # fmt: skip
    else:
        nl = case["nl"]
        m = graded_disk_mesh([A, R0], h, pml_layers=nl, pml_dr=h)
        s = UnstructuredFIT2D(m, {}, polarization=pol, bc="pec",
                              pml=dict(r0=R0, r1=R0 + nl * h, **case.get("cfg", {})))  # fmt: skip
    s.add_point_source(
        [0.4e-6, 0.1e-6], GaussianSource(F.mean(), 0.25 * (F[-1] - F[0]))
    )
    rc = np.array([1.2, 1.35, 1.5]) * 1e-6
    pts = np.vstack([circle_points(r, 64)[0] for r in rc])
    P = s._interp(pts)
    acc = np.zeros((F.size, len(pts)), complex)
    t0 = time.time()
    for i in range(int(400e-15 / s.dt)):
        s.one_step()
        acc += np.outer(np.exp(-2j * np.pi * F * (i + 1) * s.dt), P @ s.u)
    Fh = project_harmonics(acc.reshape(F.size, 3, 64), 12)
    al, ga, _ = fit_harmonics(Fh, 2 * np.pi * F / C0, rc, 12)
    r = np.sqrt((np.abs(ga) ** 2).sum(1) / (np.abs(al) ** 2).sum(1))
    out = dict(case, refl=r.tolist(), nodes=int(m.n_nodes), run_s=time.time() - t0)
    out.pop("cfg", None)
    return out


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((HERE / "reflection.json").read_text())
    fig, axs = plt.subplots(
        1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True
    )
    for ax, h in zip(axs, (40e-9, 20e-9)):
        for r in res:
            if abs(r["h"] - h) > 1e-12:
                continue
            ls = "--" if r["kind"] != "pml" else "-"
            lab = f"{r['name']} ({r['nodes']:,} nodes)"
            if r["kind"] == "pml":
                lab = f"{r['name']}, layer {r['nl'] * h * 1e9:.0f} nm = λ_max/{3.1416e-6 / (r['nl'] * h):.0f} ({r['nodes']:,} nodes)"
            ax.semilogy(X, r["refl"], ls, marker="o", ms=3, label=lab)
        ax.axhline(1e-3, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel("x = k a  (a = 1 µm; λ_max = 3.14 µm at x = 2)")
        ax.set_title(f"h = {h * 1e9:.0f} nm ({0.628e-6 / h:.0f} cells per λ at x = 10)")
        ax.legend(fontsize=6.5)
        ax.grid(True, which="both", alpha=0.3)
    axs[0].set_ylabel("Wakis FIT: |incoming| / |outgoing| (cylindrical harmonics)")
    fig.suptitle(
        "Outer-boundary reflection of UnstructuredFIT2D (TM unless noted), boundary at r = 1.65 µm"
    )
    fig.savefig(HERE / "pml_reflection.png", dpi=150)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        procs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        with Pool(procs) as p:
            res = p.map(run, CASES)
        (HERE / "reflection.json").write_text(json.dumps(res, indent=1))
        for r in res:
            print(
                r["name"], r["h"], r["nodes"], " ".join(f"{v:.1e}" for v in r["refl"])
            )
    plot()
