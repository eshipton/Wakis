"""
Figures of the unstructured-MPI study (reads out/correctness.json and out/scaling.json).

  python studies/unstructured-mpi/plot_figures.py

Writes into docs/img/:
  unstructured_mpi_correctness.png    distributed vs serial difference vs the round-off bound; partitions
  unstructured_mpi_strong_scaling.png speedup / efficiency vs ranks against the serial SciPy baseline
  unstructured_mpi_weak_scaling.png   time per step at fixed cells per rank
  unstructured_mpi_comm_volume.png    halo volume per step vs partition surface (cut facets)
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG = ROOT / "docs" / "img"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def correctness():
    path = HERE / "out" / "correctness.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    m = d["massless"]
    n = np.arange(1, d["steps"] + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    a = ax[0]
    a.semilogy(n, m["eps"], "k-", lw=2, label=r"derived bound $\varepsilon_n$ (agreement_bound)")
    a.semilogy(n, np.maximum(m["d_serial_lu"], 1e-300), "C0-", label="4 ranks vs serial (LU elimination)")
    a.semilogy(n, m["d_mutation"], "C3--", label="4 ranks, one halo entry dropped (mutation)")
    a.semilogy(n, m["norm"], "0.6", lw=1, label=r"$\|x_n\|_W$ (state norm)")
    a.text(0.02, 0.03, "4 ranks vs fused host step: 0 (bitwise) at every step\n"
           "plain mesh (no massless edges) vs one_step: 0 (bitwise)",
           transform=a.transAxes, fontsize=8.5, va="bottom",
           bbox=dict(fc="w", ec="0.7"))  # fmt: skip
    a.set_xlabel("time step n")
    a.set_ylabel(r"energy-norm difference $\|d_n\|_W$ [$\sqrt{\mathrm{J}}$]")
    a.set_title(f"3D, {m['n_cells']} tets, {m['n_massless']} massless edges, Debye + ABC")
    a.legend(fontsize=7.5, loc="center right")
    P = d["partition"]
    cent = np.asarray(P["centroids"])
    h = 0.22
    sl = np.abs(cent[:, 2]) < 0.5 * h
    for a, meth in zip(ax[1:], [k for k in ("metis", "rcb") if k in P["parts"]] + ["rib"]):
        if meth not in P["parts"]:
            a.axis("off")
            continue
        p = np.asarray(P["parts"][meth]["parts"])
        st = P["parts"][meth]["stats"]
        a.scatter(cent[sl, 0], cent[sl, 1], c=p[sl], cmap="tab10", vmin=0, vmax=9, s=9)
        a.set_aspect("equal")
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
        a.set_title(f"{meth.upper()}, 8 parts: {st['cut_facets']} cut faces,\n"
                    f"imbalance {st['imbalance']:.3f} (cells with |z| < h/2)", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "unstructured_mpi_correctness.png", dpi=150)
    plt.close(fig)


def _load():
    path = HERE / "out" / "scaling.json"
    return json.loads(path.read_text()) if path.exists() else []


def strong(res):
    rs = [r for r in res if r["study"] == "strong"]
    base = [r for r in rs if r["P"] == 0]
    dist = sorted([r for r in rs if r["P"] >= 1], key=lambda r: r["P"])
    if not base or not dist:
        return
    t0 = base[0]["t_step"]
    P = np.array([r["P"] for r in dist])
    t = np.array([r["t_step"] for r in dist])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    a = ax[0]
    a.loglog(P, t0 / t, "o-", label="vs serial one_step (LU elimination)")
    fused = [r for r in rs if r["P"] == -1]
    if fused:
        a.loglog(P, fused[0]["t_step"] / t, "s--", label="vs serial fused step (same arithmetic)")
    a.loglog(P, P, "k--", lw=1, label="ideal (P)")
    a.axvline(16, color="0.5", lw=0.8, ls=":")
    a.text(16, a.get_ylim()[0] * 1.3, " 16 physical cores", fontsize=8, color="0.4")
    a.set_xlabel("ranks P")
    a.set_ylabel("speedup of DistributedFIT (P ranks)")
    a.set_title(f"strong scaling, {dist[0]['n_cells']:,} tets "
                f"(serial {t0 * 1e3:.1f} ms/step)")  # fmt: skip
    a.legend(fontsize=8)
    a = ax[1]
    comp = np.array([r["t_compute_max"] for r in dist]) * 1e3
    exch = np.array([r["t_exchange_max"] for r in dist]) * 1e3
    x = np.arange(P.size)
    a.bar(x, comp, label="compute (max over ranks)")
    a.bar(x, exch, bottom=comp, label="halo exchange incl. wait (max)")
    a.plot(x, t * 1e3, "kx", label="wall time per step")
    a.set_xticks(x, [str(p) for p in P])
    a.set_yscale("log")
    a.set_xlabel("ranks P")
    a.set_ylabel("time per step [ms]")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "unstructured_mpi_strong_scaling.png", dpi=150)
    plt.close(fig)


def weak(res):
    rs = [r for r in res if r["study"] == "weak"]
    dist = sorted([r for r in rs if r["P"] >= 1], key=lambda r: r["P"])
    base = {round(r["h"], 6): r for r in rs if r["P"] == 0}
    basef = {round(r["h"], 6): r for r in rs if r["P"] == -1}
    if not dist:
        return
    P = np.array([r["P"] for r in dist])
    t = np.array([r["t_step"] for r in dist]) * 1e3
    tb = np.array([base[round(r["h"], 6)]["t_step"] * 1e3 if round(r["h"], 6) in base else np.nan
                   for r in dist])  # fmt: skip
    tf = np.array([basef[round(r["h"], 6)]["t_step"] * 1e3 if round(r["h"], 6) in basef else np.nan
                   for r in dist])  # fmt: skip
    cells = np.array([r["n_cells"] for r in dist])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    a = ax[0]
    a.semilogx(P, t, "o-", label="DistributedFIT, P ranks")
    a.semilogx(P, tb, "s--", label="serial one_step (LU elimination), same mesh")
    a.semilogx(P, tf, "^:", label="serial fused step, same mesh")
    a.set_xlabel("ranks P (mesh grows with P)")
    a.set_ylabel("time per step [ms]")
    a.set_yscale("log")
    a.set_title(f"weak scaling, ~{int(np.mean(cells / P)):,} tets per rank")
    a.legend(fontsize=8)
    a = ax[1]
    a.semilogx(P, t[0] / t, "o-", label="weak efficiency t(1)/t(P)")
    a.semilogx(P, (tf / t) / P, "^:", label="speedup / P vs serial fused step")
    a.axvline(16, color="0.5", lw=0.8, ls=":")
    a.set_ylim(0, 1.15)
    a.set_xlabel("ranks P")
    a.set_ylabel("efficiency")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "unstructured_mpi_weak_scaling.png", dpi=150)
    plt.close(fig)


def comm(res):
    rs = [r for r in res if r["P"] >= 2 and "partition" in r]
    if not rs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    a = ax[0]
    mk = {"metis": "o", "rcb": "s", "rib": "^", "auto": "o"}
    for meth in ("metis", "rcb", "rib", "auto"):
        sel = [r for r in rs if r.get("method", "auto") == meth]
        if not sel:
            continue
        cut = np.array([r["partition"]["cut_facets"] for r in sel])
        halo = np.array([r["halo_total"] for r in sel])
        a.loglog(cut, halo, mk[meth], ls="", label=f"{meth} ({sel[0]['study']})")
    allc = np.array([r["partition"]["cut_facets"] for r in rs], float)
    allh = np.array([r["halo_total"] for r in rs], float)
    k = np.sum(allc * allh) / np.sum(allc**2)
    xx = np.array([allc.min(), allc.max()])
    a.loglog(xx, k * xx, "k--", lw=1, label=f"fit: {k:.2f} doubles per cut face")
    a.set_xlabel("partition surface [cut faces]")
    a.set_ylabel("halo doubles received per step (all ranks)")
    a.legend(fontsize=8)
    a = ax[1]
    sel = sorted([r for r in rs if r["study"] == "partitioners"], key=lambda r: (r["P"], r["method"]))
    if sel:
        lab = [f"{r['method']}\nP={r['P']}" for r in sel]
        x = np.arange(len(sel))
        a.bar(x - 0.2, [r["partition"]["cut_facets"] / 1e3 for r in sel], 0.4, label="cut faces [1e3]")
        a2 = a.twinx()
        a2.plot(x + 0.2, [r["t_step"] * 1e3 for r in sel], "kD", label="time per step [ms]")
        a2.set_ylabel("time per step [ms]")
        a2.grid(False)
        a.set_xticks(x, lab, fontsize=8)
        a.set_ylabel("cut faces [thousands]")
        a.set_title("partitioners, strong-scaling mesh")
        a.legend(loc="upper left", fontsize=8)
        a2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "unstructured_mpi_comm_volume.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    correctness()
    res = _load()
    strong(res)
    weak(res)
    comm(res)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
