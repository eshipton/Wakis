"""publish_ho.py DIR COMMIT BATCH_JOB [--dry]: publish DGTD bench records to the shared store."""
import glob, json, sys
from wakis.benchmarks.results import publish

d, commit, job = sys.argv[1:4]
dry = "--dry" in sys.argv
ren = {"dg1-pml": "dgtd-p1", "dg2-pml": "dgtd-p2", "dg3-pml": "dgtd-p3", "dg4-pml": "dgtd-p4",
       "dg4-pml-straight": "dgtd-p4-straight"}
n = 0
for f in sorted(glob.glob(d + "/*.json")):
    r = json.load(open(f))
    if "metrics" not in r:
        print("skip", f, r.get("error", "")[:80]); continue
    r["arm"] = ren.get(r["arm"], r["arm"])
    r["commit"] = commit
    r["batch_job"] = job
    note = ("nodal DGTD (wakis.highorder.DGTD2D), curved isoparametric elements, upwind flux, LSRK(5,4), "
            "CFS-PML box; unknowns = 3 Np K field dofs (ADE/PML auxiliaries in cost.state); "
            "each step = 5 RHS evaluations (cost.rhs_evals)")
    if "straight" in r["arm"]:
        note += "; straight-sided (polygonal) interfaces: geometric-error control arm"
    if dry:
        print(r["case"], r["arm"], r["resolution"]); n += 1; continue
    publish(r, machine="n2-standard-2", notes=note); n += 1
print("published", n)
