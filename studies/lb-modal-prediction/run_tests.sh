#!/bin/bash
# D21 validation: run the fast test_095, the fast test_087 and the LB-only slow D21 tests in parallel.
set -u
mkdir -p studies/lb-modal-prediction/out/tests
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m pytest -v -s -rA tests/test_095_multistream_prediction.py > studies/lb-modal-prediction/out/tests/t095.log 2>&1 &
python -m pytest -v -s -rA -m "not slow" tests/test_087_particles_vs_linearized_beam.py > studies/lb-modal-prediction/out/tests/t087_fast.log 2>&1 &
python -m pytest -v -s -rA -m slow -k "lb_d21" tests/test_087_particles_vs_linearized_beam.py > studies/lb-modal-prediction/out/tests/t087_d21_slow.log 2>&1 &
python -m pytest -v -rA tests/test_091_scw_cache.py > studies/lb-modal-prediction/out/tests/t091.log 2>&1 &
wait
rc=0
for f in studies/lb-modal-prediction/out/tests/*.log; do echo "== $f"; grep -E "^(PASSED|FAILED|ERROR)|passed|failed|error" "$f" | tail -40; grep -qE "[0-9]+ failed|[0-9]+ error" "$f" && rc=1; done
grep -h "D21\|LBD21\|control " studies/lb-modal-prediction/out/tests/*.log | head -80
exit $rc
