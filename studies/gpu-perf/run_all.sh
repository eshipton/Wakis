#!/bin/bash
# Run the benchmark cases on one GPU VM, one run at a time (uncontended).
#   bash run_all.sh <tag> <fused: none|0|1> [cases...]
# Writes results/<tag>_<case>_{gpu,cpu}.json and prof/<tag>_<case>_{api,kern}.csv
set -u
TAG=$1; FUSED=$2; shift 2
CASES=${@:-plain3d plain3d_big cpml3d tensor2d ade3d tfsf2d adjoint2d}
PY=${PY:-$HOME/venv/bin/python}
NSYS=/usr/local/cuda/bin/nsys
cd "$(dirname "$0")"
mkdir -p results prof
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
FLAG=""; [ "$FUSED" != none ] && FLAG="--fused $FUSED"
[ "${GRAPH:-0}" = 1 ] && FLAG="$FLAG --emsolve"
for c in $CASES; do
  echo "== $TAG $c"
  if [ "${BENCH:-1}" = 1 ]; then
    $PY bench.py --case $c --gpu $FLAG --steps 500 --out results/${TAG}_${c}_gpu.json > /dev/null || echo "FAILED gpu $c"
  fi
  if [ "${CPU:-0}" = 1 ]; then
    $PY bench.py --case $c $FLAG --steps 100 --warmup 10 --out results/${TAG}_${c}_cpu.json > /dev/null || echo "FAILED cpu $c"
  fi
  if [ "${PROFILE:-1}" = 1 ]; then
    $NSYS profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda -f true \
      -o prof/${TAG}_${c} $PY bench.py --case $c --gpu $FLAG --steps 100 --nsys > /dev/null 2>&1
    $NSYS stats -q --report cuda_api_sum --report cuda_gpu_kern_sum --format csv,csv prof/${TAG}_${c}.nsys-rep --force-export=true -o prof/${TAG}_${c} > /dev/null 2>&1
    rm -f prof/${TAG}_${c}.sqlite
  fi
done
echo DONE
