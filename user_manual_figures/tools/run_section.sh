#!/usr/bin/env bash
# Generate illustration candidates for one user-manual section on Cloud Batch:
#   bin/batch-run JOB WT user_manual_figures/section4/_candidates --hours 3 -- \
#     bash user_manual_figures/tools/run_section.sh section4 [extra driver args]
# Uses the job's service account through Vertex AI (no personal credentials).
set -euo pipefail
SECTION=$1; shift
PVA_COMMIT=e088a8fff74cc363b6897c0843631fff76484908   # google-research/papervizagent, Apache-2.0
WT=$(pwd)
git clone -q https://github.com/google-research/papervizagent.git /work/pva
git -C /work/pva checkout -q "$PVA_COMMIT"
uv venv -q --python 3.12 /work/pva-venv
uv pip install -q --python /work/pva-venv/bin/python -r /work/pva/requirements.txt huggingface_hub
# Reference diagrams for the Retriever agent, if the benchmark is published; else retrieval falls back to none.
/work/pva-venv/bin/python - <<'PY' || echo "PaperBananaBench not available; retrieval disabled"
from huggingface_hub import snapshot_download
snapshot_download("dwzhu/PaperBananaBench", repo_type="dataset", local_dir="/work/pva/data/PaperBananaBench")
PY
ls /work/pva/data/PaperBananaBench/diagram 2>/dev/null | head
export GOOGLE_CLOUD_PROJECT=cloud-code-511302 GOOGLE_CLOUD_LOCATION=global
/work/pva-venv/bin/python "$WT/user_manual_figures/tools/paperviz_generate.py" \
  --spec "$WT/user_manual_figures/$SECTION/specs/$SECTION.json" \
  --manual "$WT/user_manual.md" --pva-dir /work/pva \
  --out "$WT/user_manual_figures/$SECTION/_candidates" "$@"
