#!/usr/bin/env bash
# bin/batch-run JOB WT user_manual_figures/SECTION/_refined -- bash user_manual_figures/tools/run_refine.sh SECTION
# Sources are the picked candidates in SECTION/picks/ (copied from _candidates/).
set -euo pipefail
SECTION=$1; shift
uv venv -q --python 3.12 /work/ref-venv && uv pip install -q --python /work/ref-venv/bin/python google-genai
export GOOGLE_CLOUD_PROJECT=cloud-code-511302
D=user_manual_figures/$SECTION
/work/ref-venv/bin/python user_manual_figures/tools/paperviz_refine.py --refine $D/specs/${SECTION}_refine.json \
  --spec $D/specs/$SECTION.json --candidates $D/picks --out $D/_refined "$@"
