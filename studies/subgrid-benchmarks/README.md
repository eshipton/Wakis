# Subgridded Cartesian FIT on the plasmonic and nanogap benchmarks

**Question.** `wakis.subgrid` is 1.4–2.1× cheaper than the best graded baselines on the smooth dielectric Mie cylinder (`../mie-cylinder-subgrid`). Does local Cartesian refinement also pay off when the field concentrates on curved metal surfaces (B2, B3) or in a nanogap (S1)?

**Arm** `wakis.benchmarks.arms.subgrid2d.Subgrid2DArm`:
- **Mesh.** A coarse grid H = h·2^L (at most λ_host/20, or λ_host/40 for the `ppw40` arm), refined 2:1 down to h on and inside the metal, plus a band of 2 cells per level.
- **Gap refinement.**
  - `gap_zone = 2`, `gap_levels = L`: every cell whose local gap width (the sum of the distances to the two nearest surfaces) is below 2 g gets L more levels. That covers the gap and the facing surfaces.
  - `gap_cells = 4`: refine until 4 cells span the local gap.
- **Materials.** Rakić Au/Ag Drude–Lorentz poles through the ADE. In TE each sub-pixel column of a cut edge carries its own metal-segment field in series with the host, which is exact for a flat surface aligned with the grid. In TM the poles sit on the nodes, weighted by area fraction.
- **Excitation and boundary.** Scattered-field plane wave in the host, split-field PML on the coarse frame.
- **LTS.** Optional two-level LTS (`lts=p`).

Every record went to the shared store (`gs://cloud-code-511302-autoresearch/bench-results/<case>/<arm>/`) under the arm names `subgrid-2to1[-ppw40|-lts4|-gap4|-gap4-lts8|-zone2L1|-zone2L2|-zone2L3|...]`. Each record carries its commit and Batch job id. All runs used one n2-standard-2 spot machine per task (uncontended).

RESULTS

## Reproduce

```bash
python sg_bench.py run --phase B2|B2p|B3|B3z --shard I --nshards N --outdir out   # Cloud Batch
python showcase.py run --shard I --nshards 68 --outdir out                        # S1 (all arms)
python publish_sg.py --phase B2 --commit <sha> --job <batch job>                 # to the store
BENCH_ROOT=<local mirror of the store> python plot_bench.py                       # figures + data/summary_B2_B3.md
```
