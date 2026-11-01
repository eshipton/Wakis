set -u
export OMP_NUM_THREADS=1
nvidia-smi -L || true
F="examples/_space_charge_waves_common.py tests/test_087_particles_vs_linearized_beam.py examples/024_particles_space_charge_waves.py"
echo "== ruff"; ruff check $F 2>&1 | tail -2; ruff format --check $F 2>&1 | tail -2
echo "== s_qs gone?"; grep -n "s_qs\|drop_k0" $F | head -5
echo "== 087 (full, incl. slow)"; python -m pytest -q -s -rs tests/test_087_particles_vs_linearized_beam.py > /tmp/t087.log 2>&1
grep -E '^(INJCHECK|H1CHECK|LBCHECK|DEPCHECK|MUTCHECK|DIAG|TERMS) ' /tmp/t087.log | sort -u | cut -c1-900
echo "== presence"
for p in "INJCHECK profile=sheet " "INJCHECK profile=sheet_lattice" "TERMS grid=coarse" "TERMS grid=fine" "LBCHECK grid=coarse" "LBCHECK grid=fine" "H1CHECK grid=coarse" "H1CHECK grid=fine" "DEPCHECK grid=coarse" "DEPCHECK grid=fine" "MUTCHECK id=M1" "MUTCHECK id=M2" "MUTCHECK id=M3" "DEPCHECK id=M3"; do grep -q "^$p" /tmp/t087.log && echo "PRESENT $p" || echo "MISSING $p"; done
for d in lb_ramp lb_parasitic particle_factorial modal_unit shear lb_outflow axial_vs_transverse A1 A2 A3 A4 A5 D3 D4 D5 D6; do grep -qE "^DIAG id=$d( |$)" /tmp/t087.log && echo "PRESENT DIAG $d" || echo "MISSING DIAG $d"; done
grep -E "status=ERROR" /tmp/t087.log | cut -c1-400
grep -E "^E  +[A-Za-z]*(Error|Exception)" /tmp/t087.log | sort -u | cut -c1-400
grep -E "^FAILED|passed|failed" /tmp/t087.log | grep -vE '^\s' | tail -12 | cut -c1-300
echo "== 079 080 090"; python -m pytest -q tests/test_079_particles_core.py tests/test_080_particles_gpu.py tests/test_090_linearized_beam_inflow_drive.py 2>&1 | tail -1
echo "== 083"; python -m pytest -q tests/test_083_particles_beams.py 2>&1 | tail -1
echo "== example smoke"; timeout 1800 python examples/024_particles_space_charge_waves.py --smoke 2>&1 | tail -5
echo "== untracked after smoke"; git status --short --untracked-files=all examples docs/img | head
echo "== diff stat"; git diff --stat 464ec32f HEAD
