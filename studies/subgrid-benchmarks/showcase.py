"""
Extra case where local refinement should pay off: a silicon nanogap dimer.

Two Si cylinders (n = 3.5, non-dispersive, a = 100 nm) with a 5 nm or 2 nm
gap in vacuum, TE with E along the dimer axis, incidence along +y,
lambda = 0.8-2.0 um. Metric: peak gap-centre enhancement |E/E0|^2 (about 54)
against the exact multi-cylinder T-matrix (`CylinderCluster`), plus Q_ext.

Why: the gap is 20-50x smaller than the cylinders and 400x smaller than the
wavelength. The graded baselines put their finest size h on the whole
surface of both cylinders (triangles) or on full rows and columns
(rectilinear); the subgrid puts h on the surfaces and h / 2^L only in the gap
zone (facing surfaces with local gap width < 2 g).

Arms: cart-conformal-all-graded, tri-graded (baselines of the suite) and
subgrid-2to1-zone2L3 (+ LTS p = 8: the dielectric is non-dispersive, so LTS
applies).

    python showcase.py list
    python showcase.py run --shard I --nshards N --outdir DIR
"""

import argparse
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NM = 1e-9
A_SI = 100 * NM
H_BASE = [20, 14, 10, 7, 5, 3.5, 2.5, 1.75, 1.25, 1.0]  # nm, baselines
H_SG = [20, 14, 10, 7, 5, 3.5, 2.5]  # nm, surface size (gap h / 8)
GAPS = [5, 2]


def make_case(gap_nm):
    from wakis.benchmarks.cases import Scattering2DCase

    g = gap_nm * NM
    xc = A_SI + g / 2
    return Scattering2DCase(
        name=f"S1-Si-g{gap_nm:g}",
        description=f"Si dimer a = 100 nm, gap {gap_nm:g} nm, vacuum, TE, E along the axis",
        cylinders=[dict(center=(-xc, 0.0), radius=A_SI, material="Si"),
                   dict(center=(xc, 0.0), radius=A_SI, material="Si")],
        materials={"Si": [3.5**2, 1.0]}, n_background=1.0, polarization="TE",
        direction=np.pi / 2, wavelengths=np.linspace(800, 2000, 121) * NM,
        circles=(330 * NM, 370 * NM, 410 * NM), n_points=64, r_absorber=450 * NM,
        t_run=150e-15, norm_width=4 * A_SI, gap_point=(0.0, 0.0), deembed=False,
        targets=dict(enh_peak=(0.05, 0.02)), primary="enh_peak",
    )  # fmt: skip


def arms():
    from wakis.benchmarks.arms.cartesian2d import Cartesian2DArm
    from wakis.benchmarks.arms.subgrid2d import Subgrid2DArm
    from wakis.benchmarks.arms.triangles2d import Triangles2DArm

    return {
        "cart-conformal-all-graded": Cartesian2DArm(conformal="all", graded=True),
        "tri-graded": Triangles2DArm(graded=True),
        "subgrid-2to1-zone2L3": Subgrid2DArm(gap_zone=2, gap_levels=3),
        "subgrid-2to1-zone2L3-lts8": Subgrid2DArm(gap_zone=2, gap_levels=3, lts=8),
    }


def tasks():
    out = []
    for g in GAPS:
        for arm in arms():
            ladder = H_SG if arm.startswith("subgrid") else H_BASE
            for h in ladder[::-1]:
                out.append((g, arm, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "run"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default=str(HERE / "data" / "S1"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    tl = tasks()
    if a.mode == "list":
        for i, t in enumerate(tl):
            print(i, *t)
        return
    from wakis.benchmarks import run, save_record

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if a.only:
        mine = [
            (float(x.split(":")[0]), x.split(":")[1], float(x.split(":")[2]))
            for x in a.only
        ]
    else:
        mine = tl[a.shard :: a.nshards]
    am = arms()
    for g, arm, h in mine:
        case = make_case(g)
        arm_obj = am[arm]
        arm_obj.name = arm
        rec = run(case, arm_obj, h * NM)
        save_record(rec, out / f"{case.name}__{arm}__{h:.3f}nm.json")
        m = rec.get("metrics", {})
        print(case.name, arm, h, rec.get("error") or
              f"enh_peak {m.get('enh_peak'):.4f} q_ext {m.get('q_ext_nrms'):.4f}",
              rec.get("cost", {}).get("unknowns"), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
