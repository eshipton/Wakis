"""
Field reflection of the radial CFS-PML at the B3 (Au nanodimer) scale.

Host n = 2, PML from r0 = 110 nm (the B3 absorber radius), band
lambda = 500..1000 nm (lambda_host = 250..500 nm, k r0 = 1.4..2.8), an
off-centre TE (H_z) or TM line source, harmonics fitted on the B3 circles
65/80/95 nm. Reflection = sqrt(sum_n |gamma_n|^2 / sum_n |alpha_n|^2) and
the n = 0, 1 ratios; this is the field reflection that enters the
(non-de-embedded) B3 gap enhancement linearly.

Variants (``key=value`` pairs): h_out [nm], n (rings), dr (ring spacing,
units of h_out), order, R, kappa, alpha (units of omega_min), buffer
(polar rings with sigma = 0 before the layer), r0 [nm], pol.

    python b3_pml_reflection.py run --shard I --nshards N --outdir DIR
    python b3_pml_reflection.py one h_out=12.5 n=10       # one variant (smoke)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.constants import c as C0

NM = 1e-9
LAM = np.linspace(500, 1000, 11) * NM
NB = 2.0
F = C0 / LAM
RC = np.array([65.0, 80.0, 95.0]) * NM

BASE = dict(h_out=12.5, n=10, dr=1.0, order=3, R=1e-8, kappa=1.0, alpha=0.0, buffer=0,
            r0=110.0, pol="TE", d=0.0)  # fmt: skip
# d > 0: PML thickness [nm] fixed in physical units, n rings of spacing d / n


def variants_fine():
    """Fine outer mesh (h_out = 2 nm, as B3 at h = 0.5 nm): thickness and profile."""
    v = []
    for h in (2.0, 4.0):
        v.append(dict(h_out=h))  # current: 10 rings of h_out
        v.append(dict(h_out=h, n=20))
        for d, n in ((40, 8), (60, 10), (60, 15), (80, 12), (80, 20), (120, 16), (120, 24), (200, 25)):
            v.append(dict(h_out=h, d=d, n=n))
        for R in (1e-4, 1e-6, 1e-10, 1e-12):
            v.append(dict(h_out=h, d=80, n=16, R=R))
        for m in (2, 4, 5):
            v.append(dict(h_out=h, d=80, n=16, order=m))
        for al in (0.05, 0.2, 0.5):
            v.append(dict(h_out=h, d=80, n=16, alpha=al))
        v.append(dict(h_out=h, d=80, n=16, kappa=2.0))
        v.append(dict(h_out=h, d=120, n=24, order=4, R=1e-10))
    return [dict(BASE, **x) for x in v]


def variants():
    v = []
    for h in (12.5, 6.25):
        v.append(dict(h_out=h))  # current B3 arm (ppw 20 / 40 at lambda_min)
        for n in (5, 20, 40):
            v.append(dict(h_out=h, n=n))
        for R in (1e-3, 1e-4, 1e-5, 1e-6, 1e-10):
            v.append(dict(h_out=h, R=R))
        for m in (2, 4):
            v.append(dict(h_out=h, order=m))
        v.append(dict(h_out=h, n=20, dr=0.5))
        v.append(dict(h_out=h, n=40, dr=0.5))
        v.append(dict(h_out=h, buffer=4))
        v.append(dict(h_out=h, r0=200.0))
        v.append(dict(h_out=h, alpha=0.3))
        v.append(dict(h_out=h, kappa=3.0))
        v.append(dict(h_out=h, n=20, R=1e-5))
        v.append(dict(h_out=h, n=20, R=1e-6, order=4))
        v.append(dict(h_out=h, n=30, R=1e-6, order=4))
        v.append(dict(h_out=h, n=20, dr=0.5, R=1e-5))
        v.append(dict(h_out=h, pol="TM"))
    v.append(dict(h_out=3.125))
    v.append(dict(h_out=3.125, n=20, R=1e-6, order=4))
    v.append(dict(h_out=12.5, n=0))  # bare PEC wall: must reflect ~1
    return [dict(BASE, **x) for x in v]


def run(var):
    from wakis.cylinder_harmonics import circle_points, fit_harmonics, project_harmonics
    from wakis.sources import GaussianSource
    from wakis.unstructured import UnstructuredFIT2D, graded_disk_mesh

    h = var["h_out"] * NM
    r0 = var["r0"] * NM
    dr = var["d"] * NM / var["n"] if var.get("d") else var["dr"] * h
    nb, nl = int(var["buffer"]), int(var["n"])
    t0 = time.time()
    m = graded_disk_mesh([40 * NM, r0], h, pml_layers=nb + nl, pml_dr=dr)
    pml = None
    if nl:
        r_start = r0 + nb * dr
        pml = dict(r0=r_start, r1=r_start + nl * dr, order=var["order"], R=var["R"],
                   kappa_max=var["kappa"], alpha=var["alpha"] * 2 * np.pi * F.min())  # fmt: skip
    mats = {k: [NB**2, 1.0] for k in range(3)}
    s = UnstructuredFIT2D(m, mats, polarization=var["pol"], bc="pec", pml=pml)
    src = GaussianSource(F.mean(), 0.3 * (F.max() - F.min()))
    s.add_point_source([30 * NM, 10 * NM], src)
    pts = np.vstack([circle_points(r, 64)[0] for r in RC])
    P = s._interp(pts)
    acc = np.zeros((F.size, len(pts)), complex)
    shift = 0.0 if var["pol"] == "TM" else -0.5 * s.dt
    n_steps = int(150e-15 / s.dt)
    for i in range(n_steps):
        s.one_step()
        acc += np.outer(np.exp(-2j * np.pi * F * ((i + 1) * s.dt + shift)), P @ s.u)
    Fh = project_harmonics(acc.reshape(F.size, 3, 64), 12)
    al, ga, _ = fit_harmonics(Fh, 2 * np.pi * F * NB / C0, RC, 12)
    R = np.sqrt((np.abs(ga) ** 2).sum(1) / (np.abs(al) ** 2).sum(1))
    r01 = np.abs(ga[:, 11:14]) / np.maximum(np.abs(al[:, 11:14]), 1e-300)
    return dict(var, refl=R.tolist(), refl_n0=r01[:, 1].tolist(), refl_n1=r01[:, 2].tolist(),
                refl_max=float(R.max()), nodes=int(m.n_nodes),
                unknowns=int(s.free.size + m.n_edges), steps=n_steps, run_s=time.time() - t0)  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "one", "list"])
    ap.add_argument("kv", nargs="*")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=str(Path(__file__).parent / "b3_reflection"))
    ap.add_argument("--set", default="coarse", choices=["coarse", "fine"])
    a = ap.parse_args()
    if a.mode == "list":
        for i, v in enumerate(variants() if a.set == "coarse" else variants_fine()):
            print(i, v)
        return
    if a.mode == "one":
        var = dict(BASE)
        for kv in a.kv:
            k, v = kv.split("=")
            var[k] = v if k == "pol" else float(v)
        out = run(var)
        print(
            json.dumps(
                {
                    k: (np.round(v, 5).tolist() if isinstance(v, list) else v)
                    for k, v in out.items()
                }
            )
        )
        return
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    vs = variants() if a.set == "coarse" else variants_fine()
    for i in range(a.shard, len(vs), a.nshards):
        out = run(vs[i])
        (Path(a.outdir) / f"refl_{i:03d}.json").write_text(json.dumps(out))
        print(i, vs[i], "max refl %.2e" % out["refl_max"], flush=True)


if __name__ == "__main__":
    sys.exit(main())
