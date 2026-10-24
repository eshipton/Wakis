#!/usr/bin/env python3
# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# SPDX-License-Identifier: Apache-2.0

"""
Submit PIC tutorial parameter sweeps as a Google Cloud Batch array job.

Queries each tutorial script with ``--list-tasks`` to build a global task
manifest, packages the worktree, uploads to GCS, and submits a single array
job. Use ``--dry-run`` to inspect the manifest and Batch job JSON locally
without calling ``gcloud`` or ``gsutil``.

Usage
-----
    python tutorials/05_guided_waves/cloud/submit.py --dry-run tutorials/05_guided_waves/05-01_y_junction/y_junction.py
    python tutorials/05_guided_waves/cloud/submit.py --gpu --dry-run tutorials/05_guided_waves/05-01_y_junction/y_junction.py
"""

import argparse
import datetime as dt
import shlex
import re
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

PROJECT = os.environ.get("PROJECT", "cloud-code-511302")
REGION = os.environ.get("REGION", "us-central1")
BUCKET = os.environ.get("BUCKET", "cloud-code-511302-autoresearch")
IMAGE = os.environ.get("IMAGE", "python:3.11-slim")
SERVICE_ACCOUNT = os.environ.get(
    "SERVICE_ACCOUNT", "autoresearch-runner@cloud-code-511302.iam.gserviceaccount.com"
)


def _json_task_listing(out):
    """Parse a JSON ``--list-tasks`` output (a list, or a dict with ``tasks``); None if not JSON."""
    try:
        data = json.loads(out)
    except ValueError:
        return None
    tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        return None
    listing = []
    for pos, t in enumerate(tasks):
        t_idx = t.get("index", t.get("id", pos)) if isinstance(t, dict) else pos
        name = t.get("name", "") if isinstance(t, dict) else str(t)
        listing.append((int(t_idx), str(name)))
    return listing


def discover_tasks(script_paths, task_args=()):
    """Query each tutorial script with ``--list-tasks`` and return the manifest list.

    ``task_args`` (e.g. ``["--full"]``) are passed both to ``--list-tasks`` and, through the
    manifest, to every ``--task I`` run, so the listed table and the run table agree.
    """
    manifest = []
    for sp in script_paths:
        script = Path(sp).resolve()
        rel_script = script.relative_to(REPO).as_posix()
        tutorial_name = script.parent.name
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO)
        env["MPLBACKEND"] = "Agg"
        env["OMP_NUM_THREADS"] = "1"
        out = subprocess.check_output(
            [sys.executable, str(script), "--list-tasks", *task_args],
            cwd=REPO,
            env=env,
            text=True,
        )
        listing = _json_task_listing(out)
        if listing is not None:
            for t_idx, desc in listing:
                manifest.append(
                    {
                        "global_index": len(manifest),
                        "tutorial": tutorial_name,
                        "script": rel_script,
                        "task_index": t_idx,
                        "description": desc,
                        "task_args": list(task_args),
                    }
                )
            continue
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Total"):
                continue
            # Accepted line formats: "<idx>\t<name>\t<desc>", "<idx>: <desc>",
            # "<idx> | <name> | ..." and "[<idx>] <name> : <desc>".
            m = re.match(r"^\[?(\d+)\]?\s*(?:[\t|:]\s*)?(.*)$", line)
            if not m:
                continue
            t_idx = int(m.group(1))
            rest = m.group(2)
            parts = re.split(r"\t|\s+\|\s+|\s+:\s+", rest) if rest else []
            desc = "\t".join(p.strip() for p in parts if p.strip())
            manifest.append(
                {
                    "global_index": len(manifest),
                    "tutorial": tutorial_name,
                    "script": rel_script,
                    "task_index": t_idx,
                    "description": desc,
                    "task_args": list(task_args),
                }
            )
    return manifest


def collect_repo_files():
    """Return tracked files plus untracked files under wakis/, tutorials/ and studies/."""
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO, text=True
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "wakis", "tutorials", "studies"],
        cwd=REPO,
        text=True,
    ).splitlines()
    files = sorted(set(tracked + untracked))
    return [
        f
        for f in files
        if (REPO / f).is_file()
        and (f.startswith(("wakis/", "tutorials/", "studies/")) or "/" not in f)
        and not f.endswith((".png", ".gif", ".npz", ".vtk", ".h5"))
    ]


def build_tarball(dst, files):
    """Write ``repo.tar.gz`` containing ``files`` relative to ``REPO``."""
    with tarfile.open(dst, "w:gz") as tf:
        for rel in files:
            tf.add(REPO / rel, arcname=rel)


def make_job_spec(
    gcs_prefix,
    n_tasks,
    parallelism,
    gpu=False,
    max_Hours=2,
    spot=True,
):
    """Build the Google Cloud Batch job JSON specification."""
    if gpu:
        machine_type = "g2-standard-8"
        cpu_milli = 8000
        memory_mib = 32768
        omp_threads = "4"
        use_gpu = "1"
    else:
        machine_type = "n2-standard-4"
        cpu_milli = 4000
        memory_mib = 16384
        omp_threads = "4"
        use_gpu = "0"

    boot_cmd = (
        "set -e; "
        "apt-get update -qq && "
        "apt-get install -y -qq --no-install-recommends curl ca-certificates libgomp1 >/dev/null && "
        "python3 -m pip install -q google-cloud-storage && "
        'python3 -c "'
        "from google.cloud import storage; "
        f"b, _, p = '{gcs_prefix[5:]}'.partition('/'); "
        "storage.Client().bucket(b).blob(p + '/task.sh').download_to_filename('/tmp/task.sh')"
        '" && '
        "bash /tmp/task.sh"
    )

    instance_policy = {
        "machineType": machine_type,
        "provisioningModel": "SPOT" if spot else "STANDARD",
        "bootDisk": {"sizeGb": 50},
    }
    if gpu:
        instance_policy["accelerators"] = [{"type": "nvidia-l4", "count": 1}]

    allocation_instance = {"policy": instance_policy}
    if gpu:
        allocation_instance["installGpuDrivers"] = True

    return {
        "taskGroups": [
            {
                "taskCount": int(n_tasks),
                "parallelism": min(int(parallelism), int(n_tasks)),
                "taskSpec": {
                    "computeResource": {
                        "cpuMilli": cpu_milli,
                        "memoryMib": memory_mib,
                    },
                    "maxRunDuration": f"{int(max_Hours * 3600)}s",
                    "maxRetryCount": 2,
                    "environment": {
                        "variables": {
                            "GCS_PREFIX": gcs_prefix,
                            "OMP_NUM_THREADS": omp_threads,
                            "WAKIS_USE_GPU": use_gpu,
                        }
                    },
                    "runnables": [
                        {
                            "container": {
                                "imageUri": IMAGE,
                                "entrypoint": "/bin/bash",
                                "commands": ["-c", boot_cmd],
                            }
                        }
                    ],
                },
            }
        ],
        "allocationPolicy": {
            "location": {"allowedLocations": [f"regions/{REGION}"]},
            "instances": [allocation_instance],
            "serviceAccount": {"email": SERVICE_ACCOUNT},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "scripts",
        nargs="+",
        help="Tutorial script path(s) supporting --list-tasks and --task I --out DIR",
    )
    ap.add_argument("--run-id", default=None, help="Explicit run identifier")
    ap.add_argument(
        "--parallelism", type=int, default=16, help="Max concurrent Batch tasks"
    )
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="Use g2-standard-8 + nvidia-l4 instead of default n2-standard-4 CPU",
    )
    ap.add_argument(
        "--standard",
        action="store_true",
        help="Use STANDARD provisioning instead of SPOT",
    )
    ap.add_argument(
        "--max-hours", type=float, default=2.0, help="Per-task wall time limit [hours]"
    )
    ap.add_argument(
        "--task-args",
        default="",
        help='Extra arguments for --list-tasks and every --task run, e.g. "--full"',
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print manifest and Batch job JSON without uploading or calling gcloud",
    )
    args = ap.parse_args()

    manifest = discover_tasks(args.scripts, shlex.split(args.task_args))
    if not manifest:
        raise SystemExit("No tasks discovered from --list-tasks.")

    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or f"pic-{ts}"
    gcs_prefix = f"gs://{BUCKET}/pic-tutorials/{run_id}"
    files = collect_repo_files()
    job_spec = make_job_spec(
        gcs_prefix=gcs_prefix,
        n_tasks=len(manifest),
        parallelism=args.parallelism,
        gpu=args.gpu,
        max_Hours=args.max_hours,
        spot=not args.standard,
    )

    if args.dry_run:
        print(
            f"# Dry run: run_id={run_id}, repo_files={len(files)}, tasks={len(manifest)}"
        )
        print("# Manifest:")
        print(json.dumps(manifest, indent=2))
        print("# Cloud Batch Job Spec:")
        print(json.dumps(job_spec, indent=2))
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tar_path = tmp_path / "repo.tar.gz"
        manifest_path = tmp_path / "manifest.json"
        job_path = tmp_path / "job.json"

        build_tarball(tar_path, files)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        job_path.write_text(json.dumps(job_spec, indent=2))

        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                str(tar_path),
                str(manifest_path),
                str(HERE / "task.sh"),
                f"{gcs_prefix}/",
            ],
            check=True,
        )
        subprocess.run(
            [
                "gcloud",
                "batch",
                "jobs",
                "submit",
                run_id,
                f"--project={PROJECT}",
                f"--location={REGION}",
                f"--config={job_path}",
            ],
            check=True,
        )
    print(f"Submitted {run_id} ({len(manifest)} tasks) -> {gcs_prefix}")


if __name__ == "__main__":
    main()
