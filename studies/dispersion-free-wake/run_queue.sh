#!/usr/bin/env bash
# Run the commands of a case list with N concurrent workers (GPU VM).
# Skips cases whose results/<name>.json exists. Logs to logs/<name>.log.
# Usage: ./run_queue.sh cases.txt 3
set -u
list=$1
workers=${2:-3}
mkdir -p results logs
export PATH="$HOME/wakis/.venv/bin:$PATH"
run_one() {
  cmd="$1"
  name=$(echo "$cmd" | sed -E 's/.*--out results\/([^ ]+).*/\1/')
  if [ -f "results/$name.json" ]; then return 0; fi
  if ! eval "$cmd" > "logs/$name.log" 2>&1; then
    echo "FAILED $name" >> logs/FAILED.txt
  fi
}
export -f run_one
grep -v '^\s*$' "$list" | xargs -d '\n' -P "$workers" -I{} bash -c 'run_one "$@"' _ {}
echo "queue done" >> logs/QUEUE_DONE.txt
