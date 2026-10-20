#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <TAG> [phase]" >&2
  exit 1
fi

RAW_TAG="${1%/}"
TAG="${RAW_TAG##*/}"

if [[ $# -ge 2 ]]; then
  PHASE="$2"
else
  # Extract <phase> from subtri-<phase>-<stamp>
  PHASE="$(echo "$TAG" | sed -E 's/^subtri-([^-]+)-.*$/\1/' | tr '[:lower:]' '[:upper:]')"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/data/$PHASE"
mkdir -p "$DEST"

TMPDIR_COLLECT="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_COLLECT"' EXIT

GCS_BASE="gs://cloud-code-511302-autoresearch/sweeps/${TAG}"
gcloud storage cp -r "${GCS_BASE}/shard_*" "$TMPDIR_COLLECT/" --quiet

shopt -s nullglob
JSON_FILES=("$TMPDIR_COLLECT"/shard_*/*.json)
for f in "${JSON_FILES[@]}"; do
  cp "$f" "$DEST/"
done

TOTAL_JSON=("$DEST"/*.json)
echo "Collected ${#JSON_FILES[@]} shard JSON files into $DEST (total in directory: ${#TOTAL_JSON[@]})"

FAILURES=0
for log in "$TMPDIR_COLLECT"/shard_*/task.log; do
  SHARD_NAME="$(basename "$(dirname "$log")")"
  LAST_LINE="$(tail -n 1 "$log" || true)"
  if [[ "$LAST_LINE" != "exit 0" ]]; then
    FAILURES=$((FAILURES + 1))
    echo "=== Non-zero exit in ${SHARD_NAME} (${LAST_LINE}) ==="
    tail -n 25 "$log"
  fi
done

if [[ "$FAILURES" -eq 0 ]]; then
  echo "All shard task.log files ended with exit 0."
fi
