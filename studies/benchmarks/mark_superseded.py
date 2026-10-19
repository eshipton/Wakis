"""
Mark pre-m2 B3 records in the shared store as superseded by the m2 metric.

Downloads every JSON record under
``gs://cloud-code-511302-autoresearch/bench-results/B3-*/`` whose tag does not
end in ``_m2``, adds::

    "superseded_by_metric_change": "m2 (second-order gap reconstruction, 40-ring triangle PML)"

and uploads it back via ``gcloud storage cp``, printing each updated URI.
"""

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = "gs://cloud-code-511302-autoresearch/bench-results"
REASON = "m2 (second-order gap reconstruction, 40-ring triangle PML)"


BASELINE_ARMS = {
    "cart-conformal-all-graded", "cart-conformal-all-uniform",
    "cart-staircase-graded", "cart-staircase-uniform", "tri-graded", "tri-uniform",
}  # fmt: skip


def mark_superseded(root=ROOT):
    pattern = f"{root.rstrip('/')}/B3-*/*/*.json"
    res = subprocess.run(
        ["gcloud", "storage", "ls", pattern],
        check=True,
        capture_output=True,
        text=True,
    )
    uris = [
        line.strip()
        for line in res.stdout.splitlines()
        if line.strip().endswith(".json")
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "record.json"
        for uri in uris:
            tag = Path(uri).stem
            arm = Path(uri).parent.name
            # only this suite's baseline arms: other agents own their records
            if tag.endswith("_m2") or arm not in BASELINE_ARMS:
                continue
            subprocess.run(
                ["gcloud", "storage", "cp", uri, str(tmp), "--quiet"],
                check=True,
                capture_output=True,
            )
            rec = json.loads(tmp.read_text())
            rec["superseded_by_metric_change"] = REASON
            tmp.write_text(json.dumps(rec))
            subprocess.run(
                ["gcloud", "storage", "cp", str(tmp), uri, "--quiet"],
                check=True,
                capture_output=True,
            )
            print(uri, flush=True)


if __name__ == "__main__":
    mark_superseded()
