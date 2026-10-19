#!/usr/bin/env bash
# Run the commands of a case list with N concurrent workers (GPU VM).
# Skips cases that are finished (results/<name>.json) or started by another
# runner (results/<name>_tmp or logs/<name>.log), so several runners can
# share a list. Failures go to logs/FAILED.txt.
# Usage: ./run_queue.sh cases_small.txt 4; ./run_queue.sh cases_large.txt 1
set -u
list=$1
workers=${2:-3}
mkdir -p results logs
export PATH="$HOME/wakis/.venv/bin:$PATH"
run_one() {
  cmd="$1"
  name=$(echo "$cmd" | sed -E 's/.*--out results\/([^ ]+).*/\1/')
  if [ -f "results/$name.json" ] || [ -d "results/${name}_tmp" ] || [ -f "logs/$name.log" ]; then
    return 0
  fi
  eval "$cmd" > "logs/$name.log" 2>&1 || echo "FAILED $name" >> logs/FAILED.txt
}
export -f run_one
grep -v '^\s*$' "$list" | xargs -d '\n' -P "$workers" -I{} bash -c 'run_one "$@"' _ {}
