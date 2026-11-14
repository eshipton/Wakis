# Ports on tetrahedral FIT (`wakis.unstructured.ports3d`)

Derivation and limits: `docs/physicsguide_sections/unstructured_ports3d.md`. Tests:
`tests/test_101_unstructured_ports3d.py`.

| file | what |
|---|---|
| `ports3d_study.py` | convergence/validation study (WR-90 straight guide and ε 2.2 slab, coax with 100 Ω sheet load, radial lumped ports); writes `out/results.json`; parallel over cases |
| `make_figures.py` | `docs/img/unstructured_ports3d_{setup,slab,convergence,lumped}.png` from `out/results.json` |
| `mutations.py` | runs the test file once per code mutation (monkeypatched) and reports which are caught; writes `out/mutations.json` |

Reproduce (Cloud Batch, from the repo root):

```bash
bin/batch-run w2ports-study-N wt-w2-ports studies/unstructured-ports3d/out --machine n2-standard-16 -- \
  bash -c "python studies/unstructured-ports3d/ports3d_study.py studies/unstructured-ports3d/out"
bin/batch-run w2ports-tests-N wt-w2-ports studies/unstructured-ports3d/out --machine n2-standard-8 -- \
  bash -c "python -m pytest -q tests/test_101_unstructured_ports3d.py && python studies/unstructured-ports3d/mutations.py studies/unstructured-ports3d/out/mutations.json"
```

Results (`w2ports-study-1`, `w2ports-tests-1`): see section 6 of the physics guide. In short, every
time-domain S-parameter equals the exact discrete ladder (`ExtrudedLine`) to 1e-12 (guide, coax) or 1e-9
(slab, truncation), and the deviation from the continuum converges at second order (TE10 cutoff, guide phase,
slab S11, coax Z0 and load reflection). Lumped ports equal the frequency-domain solve of the same discrete
equations to 1.3e-9.
