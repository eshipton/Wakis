"""
TM010 of the hollow of a PEC STL tube (vacuum background, PEC-only solids):
frequency error vs cells per radius for the staircase and conformal PEC.

Usage (each run < 10 s on a laptop)::

    python tube_study.py run --label conformal-fixed            # this branch
    python tube_study.py run --label staircase --staircase
    # pre-fix code: same script with PYTHONPATH pointing to a checkout of aa2293f
    PYTHONPATH=/path/to/aa2293f python tube_study.py run --label conformal-legacy-mask
    python tube_study.py plot

The run mode imports `_tube_tm010_error` style code inline so that it also
runs against the pre-fix package (which has no test_048).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
A = 0.01  # hollow radius [m]
X01 = 2.404825557695773
CENTER = np.array([0.00037, -0.00023])  # cuts the cells irregularly
LADDER = (4, 6, 8, 12, 16, 24, 32)


def tm010_error(n, conformal):
    """Relative TM010 error (spatial part) at n cells per radius."""
    import pyvista as pv
    from scipy.constants import c

    import wakis
    from wakis.sources import CurrentSource, GaussianSource

    h = A / n
    ncell = 2 * n + 6
    half = ncell * h / 2
    surf = (
        pv.CylinderStructured(
            radius=[A, A + 2 * h], height=10 * h, center=(*CENTER, 0),
            direction=(0, 0, 1), theta_resolution=360, z_resolution=2,
        ).extract_surface().triangulate()
    )  # fmt: skip
    grid = wakis.GridFIT3D(
        CENTER[0] - half, CENTER[0] + half, CENTER[1] - half, CENTER[1] + half,
        -h, h, ncell, ncell, 2, stl_solids={"tube": surf},
        stl_materials={"tube": "pec"}, conformal=conformal, verbose=0,
    )  # fmt: skip
    solver = wakis.SolverFIT3D(
        grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, use_stl=True, verbose=0
    )
    f0 = X01 * c / (2 * np.pi * A)
    pulse = GaussianSource(frequency=f0, fwidth=0.25 * f0)
    src = CurrentSource(
        pulse, component="z", center=(*(CENTER + np.array([0.13, 0.07]) * A), 0)
    )
    coords = [grid.x[:-1], grid.y[:-1], grid.z[:-1] + grid.dz / 2]
    point = (*(CENTER + np.array([-0.11, 0.17]) * A), 0.0)
    index = (*(int(np.argmin(abs(q - p))) for q, p in zip(coords, point)), "z")
    values = []

    class Probe:
        def update(self, solver, t):
            values.append(float(solver.E[index]))

    Nt = int((pulse.end_time + 30 / f0) / solver.dt)
    solver.emsolve(Nt, source=src, monitors=[Probe()])
    start = int(np.ceil(pulse.end_time / solver.dt))
    step = max(1, int(1 / (6 * f0 * solver.dt)))
    signal = np.array(values[start::step])
    dt = step * solver.dt
    # matrix pencil
    m_rows = len(signal) // 3
    Y = np.lib.stride_tricks.sliding_window_view(signal, m_rows + 1)
    _, s, vh = np.linalg.svd(Y, full_matrices=False)
    m = min(16, int(np.sum(s > 1e-9 * s[0])))
    v = vh[:m].T
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    amp = np.linalg.lstsq(np.vander(z, len(signal), increasing=True).T, signal,
                          rcond=None)[0]  # fmt: skip
    f, amp = np.angle(z) / (2 * np.pi * dt), np.abs(amp)
    keep = np.abs(f - f0) < 0.2 * f0
    f_res = f[keep][np.argmax(amp[keep])]
    f_space = 2 / solver.dt * np.sin(np.pi * f_res * solver.dt) / (2 * np.pi)
    return dict(n=n, err=float(f_space / f0 - 1), cells=int(solver.N), steps=Nt,
                dt=float(solver.dt))  # fmt: skip


def run(label, staircase):
    import wakis

    rows = [tm010_error(n, False if staircase else "pec") for n in LADDER]
    DATA.mkdir(exist_ok=True)
    import os

    code = os.environ.get("WAKIS_COMMIT", str(Path(wakis.__file__).parent))
    out = dict(label=label, wakis=code, rows=rows)
    (DATA / f"tube_{label}.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{label}: n={r['n']:3d} err={r['err']:+.4e}")


def mask_map():
    """ieps = 0 z edges in a cross-section of a one-cell-radius PEC cylinder."""
    import pyvista as pv

    import wakis

    h = 0.5e-3
    x = h * np.arange(-8, 9)
    cyl = pv.Cylinder(center=(0, 0, 0), radius=h, height=6e-3, direction=(0, 0, 1),
                      resolution=180).triangulate()  # fmt: skip
    out = {}
    for conf in (False, "pec"):
        g = wakis.GridFIT3D(x=x, y=x.copy(), z=x.copy(), stl_solids={"w": cyl},
                            stl_materials={"w": "pec"}, conformal=conf, verbose=0)  # fmt: skip
        s = wakis.SolverFIT3D(g, bc_low=["pec"] * 3, bc_high=["pec"] * 3,
                              use_stl=True, verbose=0)  # fmt: skip
        N = s.N
        ie = np.asarray(s.ieps.toarray())[2 * N :].reshape(
            (s.Nx, s.Ny, s.Nz), order="F"
        )
        L = np.asarray(s.L.toarray())[2 * N :].reshape((s.Nx, s.Ny, s.Nz), order="F")
        out[str(conf)] = dict(dead=(ie[:, :, 8] == 0).tolist(),
                              open=(L[:, :, 8] > 0).tolist())  # fmt: skip
    out["x_mm"] = (np.asarray(x) * 1e3).tolist()
    (DATA / "mask_map.json").write_text(json.dumps(out))


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    style = {
        "staircase": ("tab:gray", "s", "staircase (legacy cell mask)"),
        "conformal-legacy-mask": ("tab:red", "v", "conformal 'pec', before fix (aa2293f)"),
        "conformal-fixed": ("tab:blue", "o", "conformal 'pec', fixed"),
    }  # fmt: skip
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for label, (col, mk, name) in style.items():
        p = DATA / f"tube_{label}.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())["rows"]
        n = np.array([r["n"] for r in rows])
        e = np.abs([r["err"] for r in rows])
        ax.loglog(n, e, mk + "-", color=col, label=name)
    n = np.array([4, 32])
    ax.loglog(n, 0.12 * (4 / n), "k:", lw=1, label="order 1")
    ax.loglog(n, 0.01 * (4 / n) ** 2, "k--", lw=1, label="order 2")
    ax.set_xlabel("cells per radius a / h")
    ax.set_ylabel("|f / f_TM010 - 1|")
    ax.set_title("TM010 of a PEC STL tube in vacuum")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "tube_tm010_convergence.png", dpi=150)

    m = json.loads((DATA / "mask_map.json").read_text())
    x = np.array(m["x_mm"])
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.2), sharey=True)
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, key, title in zip(axs, ("False", "pec"),
                              ("legacy cell mask (conformal=False;\n'pec' kept these 12 before the fix)",
                               "conformal 'pec' (fixed)")):  # fmt: skip
        dead = np.array(m[key]["dead"])
        opn = np.array(m[key]["open"])
        X, Y = np.meshgrid(x, x, indexing="ij")
        sel = (np.abs(X) <= 2.0) & (np.abs(Y) <= 2.0)
        ax.scatter(
            X[sel & ~dead], Y[sel & ~dead], s=18, c="tab:blue", label="carries E_z"
        )
        bad = sel & dead & opn
        ok = sel & dead & ~opn
        ax.scatter(
            X[ok], Y[ok], s=40, c="k", marker="s", label="ieps = 0, closed (L = 0)"
        )
        ax.scatter(X[bad], Y[bad], s=50, c="tab:red", marker="x",
                   label="ieps = 0 but open (L > 0)")  # fmt: skip
        ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), "g-", lw=1.5, label="wire surface")
        ax.plot(*np.mean(np.c_[X[sel & dead], Y[sel & dead]], axis=0), "r+", ms=14)
        ax.set_title(f"{title}\n{int((sel & dead).sum())} dead z edges", fontsize=9)
        ax.set_xlabel("x [mm]")
        ax.set_aspect("equal")
    axs[0].set_ylabel("y [mm]")
    if key == "pec":
        axs[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("z edges in a cross-section of a PEC STL wire (a = h = 0.5 mm); + = centroid",
                 fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "wire_edge_mask.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "mask", "plot"])
    ap.add_argument("--label", default="conformal-fixed")
    ap.add_argument("--staircase", action="store_true")
    a = ap.parse_args()
    if a.mode == "run":
        run(a.label, a.staircase)
    elif a.mode == "mask":
        mask_map()
    else:
        plot()
    sys.exit(0)
