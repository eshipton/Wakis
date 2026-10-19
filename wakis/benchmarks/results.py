# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Shared results store of the benchmark suite.

Every record (one simulation: case x arm x resolution, from
`wakis.benchmarks.run`) is one JSON object at

    gs://cloud-code-511302-autoresearch/bench-results/<case>/<arm>/<tag>.json

with ``tag = f"{resolution_nm:.4f}nm"`` (resolution in nm, the arm's own
resolution label) unless given. Record schema (all arms):

* ``case``, ``arm``, ``resolution`` (arm label, usually h [m]);
* ``metrics``: the dict of ``case.evaluate`` (e.g. ``q_ext_nrms``,
  ``q_ext_lam_peak``, ``q_ext_height``, ``q_ext_fwhm``, ``enh_peak``,
  ``enh_lam`` ..., and ``spectra``);
* ``cost``: ``unknowns`` (field dofs), ``steps``, ``dt`` [s], ``run_s``
  (time-stepping wall time, one core, uncontended), ``setup_s``, plus
  arm-specific keys;
* ``t_run`` (DFT window [s]); records run with a longer window than the
  case default carry ``window_factor`` and the tag suffix ``_w<factor>``;
* ``wall_s``, ``cpu`` (machine model string), ``commit`` (git commit of
  the code that produced it), ``machine`` (e.g. 'n2-standard-2'),
  ``published`` (UTC ISO time), optional ``notes``.

Access goes through the gcloud CLI (no Python cloud dependency):
`publish(record)` uploads, `fetch(case)` returns a pandas DataFrame (one row
per record, metrics and cost flattened as ``m.<name>`` / ``c.<name>``),
`fetch_records(case)` the raw dicts. ``root`` may also be a local directory
(tests, offline use).
"""

import datetime
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = "gs://cloud-code-511302-autoresearch/bench-results"

__all__ = ["ROOT", "publish", "fetch", "fetch_records", "record_path"]


def _git_commit():
    try:
        here = Path(__file__).resolve().parent
        return subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except Exception:  # noqa: BLE001
        return os.environ.get("WAKIS_COMMIT", "unknown")


def record_path(record, root=ROOT, tag=None):
    """Store path of a record."""
    if tag is None:
        res = record["resolution"]
        tag = f"{float(res) * 1e9:.4f}nm" if isinstance(res, (int, float)) else str(res)
    return f"{root.rstrip('/')}/{record['case']}/{record['arm']}/{tag}.json"


def _is_gcs(root):
    return str(root).startswith("gs://")


def publish(record, root=ROOT, tag=None, machine=None, notes=None):
    """
    Upload one record (adds ``commit``, ``published``, ``machine``).

    Returns
    -------
    str
        The path written.
    """
    rec = dict(record)
    rec.setdefault("commit", _git_commit())
    rec["published"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if machine is not None:
        rec["machine"] = machine
    if notes is not None:
        rec["notes"] = notes
    path = record_path(rec, root, tag)
    text = json.dumps(rec)
    if _is_gcs(root):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(text)
        try:
            subprocess.run(["gcloud", "storage", "cp", f.name, path, "--quiet"],
                           check=True, capture_output=True)  # fmt: skip
        finally:
            os.unlink(f.name)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    return path


def fetch_records(case, arm=None, root=ROOT):
    """All records of `case` (optionally one `arm`) as a list of dicts."""
    if not _is_gcs(root):
        import glob

        pattern = f"{root.rstrip('/')}/{case}/{arm or '*'}/*.json"
        return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(pattern))]
    src = f"{root.rstrip('/')}/{case}" + (f"/{arm}" if arm else "")
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(["gcloud", "storage", "cp", "-r", src, d, "--quiet"],
                           capture_output=True, text=True)  # fmt: skip
        if r.returncode != 0:
            if (
                "matched no objects" in r.stderr.lower()
                or "not found" in r.stderr.lower()
            ):
                return []
            raise RuntimeError(f"gcloud failed (auth?): {r.stderr.strip()[:500]}")
        return [json.loads(p.read_text()) for p in sorted(Path(d).rglob("*.json"))]


def fetch(case, arm=None, root=ROOT):
    """
    Records of `case` as a pandas DataFrame: columns ``case``, ``arm``,
    ``resolution``, ``commit``, ``cpu``, ``m.<metric>`` (scalars only),
    ``c.<cost>`` and ``work`` (= unknowns x steps).
    """
    import pandas as pd

    rows = []
    for r in fetch_records(case, arm, root):
        row = {k: r.get(k) for k in ("case", "arm", "resolution", "commit", "cpu",
                                     "machine", "error")}  # fmt: skip
        for k, v in r.get("metrics", {}).items():
            if not isinstance(v, (dict, list)):
                row[f"m.{k}"] = v
        for k, v in r.get("cost", {}).items():
            row[f"c.{k}"] = v
        if "c.unknowns" in row and "c.steps" in row:
            row["work"] = row["c.unknowns"] * row["c.steps"]
        rows.append(row)
    return pd.DataFrame(rows)
