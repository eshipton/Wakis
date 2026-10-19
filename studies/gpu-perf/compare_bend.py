"""Compare loss and gradient of two bend_iteration.py results.

  python compare_bend.py results/base_bend.json results/graph_bend.json
"""

import json
import sys

import numpy as np

a, b = (json.load(open(p)) for p in sys.argv[1:3])
ga, gb = np.asarray(a["grad"]), np.asarray(b["grad"])
out = {
    "loss_a": a["loss"],
    "loss_b": b["loss"],
    "loss_rel_diff": abs(a["loss"] - b["loss"]) / abs(a["loss"]),
    "grad_rel_max_diff": float(np.max(np.abs(ga - gb)) / np.max(np.abs(ga))),
    "grad_rel_l2_diff": float(np.linalg.norm(ga - gb) / np.linalg.norm(ga)),
    "time_a_s": a["value_and_grad_s"],
    "time_b_s": b["value_and_grad_s"],
    "speedup": a["value_and_grad_s"] / b["value_and_grad_s"],
}
print(json.dumps(out, indent=1))
