#!/bin/bash
# Run the coarse and fine TD settle scans in parallel (one process per core).
set -u
mkdir -p studies/lb-modal-prediction/out
OMP_NUM_THREADS=1 python studies/lb-modal-prediction/td_settle_scan.py 10 474 ${1:-4} studies/lb-modal-prediction/out/scan_10x474.npz > studies/lb-modal-prediction/out/scan_10x474.log 2>&1 &
OMP_NUM_THREADS=1 python studies/lb-modal-prediction/td_settle_scan.py 20 948 ${1:-4} studies/lb-modal-prediction/out/scan_20x948.npz > studies/lb-modal-prediction/out/scan_20x948.log 2>&1 &
wait
cat studies/lb-modal-prediction/out/*.log
