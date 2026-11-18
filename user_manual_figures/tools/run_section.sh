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
# Reference diagrams for the Retriever agent (PaperBananaBench, a zip on the HF Hub); without them retrieval falls back to none.
/work/pva-venv/bin/python - <<'PY' || echo "PaperBananaBench download failed; retrieval disabled"
from huggingface_hub import hf_hub_download
hf_hub_download("dwzhu/PaperBananaBench", "PaperBananaBench.zip", repo_type="dataset", local_dir="/work/pbb")
PY
if [ -f /work/pbb/PaperBananaBench.zip ]; then
  (cd /work/pbb && unzip -q PaperBananaBench.zip) || true
  REF=$(find /work/pbb -path '*diagram/ref.json' | head -1 || true)
  if [ -n "$REF" ]; then
    mkdir -p /work/pva/data && ln -sfn "$(dirname "$(dirname "$REF")")" /work/pva/data/PaperBananaBench
    echo "retrieval references: $REF"
  else
    echo "no diagram/ref.json in PaperBananaBench.zip; retrieval disabled"
  fi
fi
export GOOGLE_CLOUD_PROJECT=cloud-code-511302 GOOGLE_CLOUD_LOCATION=global
/work/pva-venv/bin/python "$WT/user_manual_figures/tools/paperviz_generate.py" \
  --spec "$WT/user_manual_figures/$SECTION/specs/$SECTION.json" \
  --manual "$WT/user_manual.md" --pva-dir /work/pva \
  --out "$WT/user_manual_figures/$SECTION/_candidates" "$@"
