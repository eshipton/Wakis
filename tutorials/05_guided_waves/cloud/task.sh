#!/usr/bin/env bash
# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #
#
# SPDX-License-Identifier: Apache-2.0
#
# Cloud Batch worker entrypoint for PIC tutorials.
# Environment variables provided by submit.py / Cloud Batch:
#   BATCH_TASK_INDEX  0-based global task index
#   GCS_PREFIX        gs://<bucket>/pic-tutorials/<run_id>
#   OMP_NUM_THREADS   OpenMP thread count (default 4)
#   WAKIS_USE_GPU     "1" when running on an L4 GPU node, "0" otherwise

set -euo pipefail

idx="${BATCH_TASK_INDEX:-0}"
echo "=== PIC tutorial worker: global task ${idx} on $(hostname) at $(date -Is) ==="

export DEBIAN_FRONTEND=noninteractive
curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

mkdir -p /work /tmp/out
cd /work

python3 - <<'PY'
import os
from google.cloud import storage

prefix = os.environ["GCS_PREFIX"]
assert prefix.startswith("gs://")
bucket_name, _, blob_prefix = prefix[5:].partition("/")
bucket = storage.Client().bucket(bucket_name)
bucket.blob(f"{blob_prefix}/repo.tar.gz").download_to_filename("/work/repo.tar.gz")
bucket.blob(f"{blob_prefix}/manifest.json").download_to_filename("/work/manifest.json")
PY

tar xzf /work/repo.tar.gz -C /work

EXTRA_PKGS=""
if [[ "${WAKIS_USE_GPU:-0}" == "1" ]]; then
    EXTRA_PKGS="cupy-cuda12x"
fi

uv venv --python 3.11 /opt/venv -q
uv pip install --python /opt/venv/bin/python -q \
    numpy scipy matplotlib pyvista h5py tqdm numba pillow google-cloud-storage $EXTRA_PKGS
uv pip install --python /opt/venv/bin/python -q --no-deps -e /work

read -r TUTORIAL SCRIPT TASK_IDX TASK_ARGS < <(
    /opt/venv/bin/python - <<'PY'
import json, os
manifest = json.load(open("/work/manifest.json"))
entry = manifest[int(os.environ.get("BATCH_TASK_INDEX", "0"))]
print(entry["tutorial"], entry["script"], entry["task_index"], " ".join(entry.get("task_args", [])))
PY
)

echo "Running tutorial=${TUTORIAL} script=${SCRIPT} task=${TASK_IDX} args=${TASK_ARGS:-}"
export PYTHONPATH=/work
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# shellcheck disable=SC2086  # TASK_ARGS is a word list from the manifest
/opt/venv/bin/python "/work/${SCRIPT}" --task "${TASK_IDX}" --out /tmp/out ${TASK_ARGS:-}

/opt/venv/bin/python - "${TUTORIAL}" "${TASK_IDX}" <<'PY'
import glob, os, pathlib, sys
from google.cloud import storage

tutorial, task_idx = sys.argv[1], sys.argv[2]
prefix = os.environ["GCS_PREFIX"]
bucket_name, _, blob_prefix = prefix[5:].partition("/")
bucket = storage.Client().bucket(bucket_name)

# Each Batch task has its own VM, so everything under /tmp/out belongs to this task
# (tutorials name their outputs task_<i>.* or task_<ii>_<name>.*; keep subdirectories).
paths = sorted(p for p in pathlib.Path("/tmp/out").rglob("*") if p.is_file())
if not paths:
    sys.exit(f"task {task_idx}: no output files in /tmp/out")
for path in paths:
    name = path.relative_to("/tmp/out").as_posix()
    dst = f"{blob_prefix}/{tutorial}/{name}"
    bucket.blob(dst).upload_from_filename(path)
    print(f"Uploaded gs://{bucket_name}/{dst}")
PY

echo "=== Task ${idx} (${TUTORIAL} task_${TASK_IDX}) complete at $(date -Is) ==="
