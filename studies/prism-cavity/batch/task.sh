#!/usr/bin/env bash
# Cloud Batch task: one study task per machine (uncontended timing).
# Env: RUN_ID, BUCKET, STUDY (folder under studies/), SCRIPT, TASKS (space-separated task indices; task i = BATCH_TASK_INDEX-th entry)
set -uo pipefail
IDX="${BATCH_TASK_INDEX:-0}"
read -ra LIST <<< "${TASKS}"
TASK="${LIST[$IDX]}"
OUT="gs://${BUCKET}/sweeps/${RUN_ID}/task_${TASK}"
mkdir -p /work/out /work/repo
log() { printf '[%s] task %s: %s\n' "$(date -u +%H:%M:%S)" "$TASK" "$*" | tee -a /work/out/task.log; }
gcloud storage cp "gs://${BUCKET}/sweeps/${RUN_ID}/code.tgz" /work/code.tgz --quiet && tar xzf /work/code.tgz -C /work/repo
(apt-get update -qq && apt-get install -y -qq libgl1 libglu1-mesa libxft2 libxinerama1 libxcursor1) >> /work/out/task.log 2>&1 || log "apt failed (gmsh may not load)"
cd /work/repo
uv venv -q --python 3.12 .venv >> /work/out/task.log 2>&1 && uv pip install -q -e . pyvista gmsh meshio >> /work/out/task.log 2>&1 || log "install failed"
cd "studies/${STUDY}"
log "run on $(nproc) cpus, $(grep MemTotal /proc/meminfo)"
start=$(date +%s)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 400m /work/repo/.venv/bin/python "${SCRIPT}" run --task "$TASK" --outdir /work/out >> /work/out/task.log 2>&1
rc=$?
log "exit ${rc} after $(( $(date +%s) - start )) s"
gcloud storage cp -r /work/out/* "$OUT/" --quiet
exit $rc
