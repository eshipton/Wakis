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
  /work/pva-venv/bin/python -c "import zipfile; zipfile.ZipFile('/work/pbb/PaperBananaBench.zip').extractall('/work/pbb')" || true
  REF=$(find /work/pbb -path '*diagram/ref.json' | head -1 || true)
  if [ -n "$REF" ]; then
    mkdir -p /work/pva/data && ln -sfn "$(dirname "$(dirname "$REF")")" /work/pva/data/PaperBananaBench
    echo "retrieval references: $REF"
    # Drop or repair reference entries whose image file is missing (some names differ only by Unicode form).
    /work/pva-venv/bin/python - "$REF" <<'PY'
import json, sys, unicodedata
from pathlib import Path
ref = Path(sys.argv[1]); root = ref.parent
norm = lambda s: unicodedata.normalize("NFKC", s).replace("\u2011", "-").replace("\u2010", "-")
files = {norm(str(p.relative_to(root))): str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
items = json.loads(ref.read_text(encoding="utf-8")); keep = []; fixed = 0
for it in items:
    rel = it.get("path_to_gt_image", "")
    if (root / rel).exists():
        keep.append(it)
    elif norm(rel) in files:
        it["path_to_gt_image"] = files[norm(rel)]; keep.append(it); fixed += 1
ref.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
print(f"ref.json: kept {len(keep)} of {len(items)} ({fixed} paths repaired)")
PY
  else
    echo "no diagram/ref.json in PaperBananaBench.zip; retrieval disabled"
  fi
fi
export GOOGLE_CLOUD_PROJECT=cloud-code-511302 GOOGLE_CLOUD_LOCATION=global
/work/pva-venv/bin/python "$WT/user_manual_figures/tools/paperviz_generate.py" \
  --spec "$WT/user_manual_figures/$SECTION/specs/$SECTION.json" \
  --manual "$WT/user_manual.md" --pva-dir /work/pva \
  --out "$WT/user_manual_figures/$SECTION/_candidates" "$@"
