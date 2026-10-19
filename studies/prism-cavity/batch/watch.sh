#!/usr/bin/env bash
# watch.sh RUN_ID [interval]: poll a Batch job until it ends; fail loudly on auth errors.
RUN="$1"; INT="${2:-120}"
while true; do
  if ! err=$(gcloud auth print-access-token 2>&1 >/dev/null); then echo "!!! GCLOUD AUTH FAILURE (not completion): $err"; exit 2; fi
  out=$(gcloud batch jobs describe "$RUN" --location us-central1 --project cloud-code-511302 --format="value(status.state)" 2>&1)
  if echo "$out" | grep -qi "reauthentication\|auth login\|ERROR"; then echo "!!! GCLOUD ERROR: $out"; exit 2; fi
  echo "[$(date +%H:%M:%S)] $RUN $out"
  case "$out" in SUCCEEDED|FAILED|DELETION_IN_PROGRESS) echo "FINAL: $out"; exit 0;; esac
  sleep "$INT"
done
