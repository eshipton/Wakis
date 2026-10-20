#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <phase>" >&2
  exit 1
fi

PHASE="$1"
PHASE_LC="$(echo "$PHASE" | tr '[:upper:]' '[:lower:]')"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
TAG="subtri-${PHASE_LC}-${STAMP}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(git -C "$HERE" rev-parse --show-toplevel)"

PY="/Users/erikshipton/computational-electromagnetics/wakis/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

N="$("$PY" "$HERE/subtri_bench.py" list --phase "$PHASE" | awk 'END {print $1}')"
if [[ -z "$N" || ! "$N" =~ ^[0-9]+$ || "$N" -le 0 ]]; then
  echo "Failed to determine positive task count for phase $PHASE (got: '$N')" >&2
  exit 1
fi

TMPDIR_BATCH="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_BATCH"' EXIT

CODE_TGZ="$TMPDIR_BATCH/code.tgz"
TASK_SH="$TMPDIR_BATCH/task.sh"
JOB_JSON="$TMPDIR_BATCH/job.json"
GCS_BASE="gs://cloud-code-511302-autoresearch/sweeps/${TAG}"

git -C "$WORKTREE" archive --format=tar.gz HEAD -o "$CODE_TGZ"

cat > "$TASK_SH" <<EOF
#!/usr/bin/env bash
set -uo pipefail
IDX="\${BATCH_TASK_INDEX:-0}"
OUT="gs://cloud-code-511302-autoresearch/sweeps/${TAG}/shard_\${IDX}"
mkdir -p /work/out /work/repo
gcloud storage cp "gs://cloud-code-511302-autoresearch/sweeps/${TAG}/code.tgz" /work/code.tgz --quiet && tar xzf /work/code.tgz -C /work/repo
cd /work/repo
uv venv -q --python 3.12 .venv >> /work/out/task.log 2>&1 && uv pip install -q -e . pyvista pandas >> /work/out/task.log 2>&1
cd studies/subtri-hybrid
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 330m /work/repo/.venv/bin/python subtri_bench.py run --phase ${PHASE} --shard "\$IDX" --nshards ${N} --outdir /work/out >> /work/out/task.log 2>&1
echo "exit \$?" >> /work/out/task.log
gcloud storage cp -r /work/out/* "\$OUT/" --quiet
EOF

gcloud storage cp "$CODE_TGZ" "${GCS_BASE}/code.tgz" --quiet
gcloud storage cp "$TASK_SH" "${GCS_BASE}/task.sh" --quiet

cat > "$JOB_JSON" <<EOF
{
  "taskGroups": [
    {
      "taskSpec": {
        "runnables": [
          {
            "container": {
              "imageUri": "us-central1-docker.pkg.dev/cloud-code-511302/autoresearch/probe:latest",
              "entrypoint": "/bin/bash",
              "commands": [
                "-c",
                "gcloud storage cp ${GCS_BASE}/task.sh /work/task.sh --quiet && bash /work/task.sh"
              ]
            }
          }
        ],
        "computeResource": {
          "cpuMilli": 2000,
          "memoryMib": 7000
        },
        "maxRunDuration": "20400s",
        "maxRetryCount": 2,
        "lifecyclePolicies": [
          {
            "action": "RETRY_TASK",
            "actionCondition": {
              "exitCodes": [50001]
            }
          }
        ]
      },
      "taskCount": ${N},
      "parallelism": ${N}
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "policy": {
          "machineType": "n2-standard-2",
          "provisioningModel": "SPOT",
          "bootDisk": {
            "sizeGb": 40
          }
        }
      }
    ],
    "serviceAccount": {
      "email": "autoresearch-runner@cloud-code-511302.iam.gserviceaccount.com"
    }
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  },
  "labels": {
    "autoresearch": "sweep",
    "tag": "subtri-${PHASE_LC}"
  }
}
EOF

gcloud batch jobs submit "$TAG" \
  --location us-central1 \
  --project cloud-code-511302 \
  --config "$JOB_JSON"

echo "$TAG"
