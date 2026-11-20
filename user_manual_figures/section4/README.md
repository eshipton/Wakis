# User manual, Section 4 figures (FIT theory and method comparison)

One illustration per subsection of `user_manual.md` Section 4. Each figure was checked label by label against the manual text. Illustrations only: no plotted results.

| Figure | Subsection | Shows | Made by |
|---|---|---|---|
| `fig4_1_dual_cell_complex.png` | 4.1 | a primary cell and its dual: edge L_k pierces dual face Ã_k, dual edge L̃_m pierces face A_m; Faraday on one face as an exact ±1 sum | script `tools/section4_scripted.py` |
| `fig4_2_maxwell_grid_equations.png` | 4.2 | Maxwell grid equations (primary column, dual column, M_ε and M_ν links); unfolded cube showing ∂∂ = 0 ⇒ S C = 0 | script |
| `fig4_3_hodge_matrices.png` | 4.3 | four Hodge constructions: diagonal, conformal cut cell, tensor averaging, curvilinear | PaperVizAgent, candidate c1 + 2K re-render v0 |
| `fig4_4_leapfrog_cfl.png` | 4.4 | leapfrog time levels; exact amplification roots ξ for Δt²λ = 0.6, 2, 3.4, 4, 4.6; field norm of one mode below (Δt²λ = 3) and above (4.05) the limit | script |
| `fig4_5_fit_fdtd_dec.png` | 4.5 | FDTD, FIT and edge-element FEM reach one algebraic system; discrete exterior calculus ladder | PaperVizAgent c2 + re-render v1 |
| `fig4_6_fit_vs_fem.png` | 4.6 | Whitney (Galerkin) vs lumped mass pattern; acute pair (dual length > 0) and obtuse pair (dual length < 0, reversed); time vs frequency domain | script |
| `fig4_7_cem_landscape.png` | 4.7 | methods by what they discretize and by electrical size | PaperVizAgent c1 + re-render v0 |
| `fig4_8_operator_structure.png` | 4.8 | operator structure each method stores and solves | PaperVizAgent c2 + edit v2 (arrow removed) |
| `fig4_9_decision_guide.png` | 4.9 | decision tree | PaperVizAgent c1 + edit v2 (legend "solver in Wakis") |
| `fig4_10_failure_modes.png` | 4.10 | six failure modes with remedies | PaperVizAgent c1 + re-render v0 |

## How they were made
- Generated figures: PaperVizAgent (google-research/papervizagent, Apache-2.0, commit e088a8ff), Gemini 3.1 Pro (text) and Gemini 3 Pro Image through Vertex AI, from the briefs in `specs/section4.json` (`tools/run_section.sh`, Cloud Batch jobs pva-s4-full and pva-s4-c). Three candidates per figure, three critic rounds. The picked candidates are in `picks/`; corrections and 2K re-renders come from `specs/section4_refine.json` (`tools/run_refine.sh`, job pva-s4-refine), three variants each. Palette-quantized to 128 colours.
- Scripted figures: the generated candidates for 4.1, 4.2, 4.4 and 4.6 misplaced edges, arrows or circumcentres even after targeted corrections, so they are drawn from the geometry and mathematics: `python tools/section4_scripted.py section4`. The script asserts that the drawn dual lengths match ½L(cot α₁ + cot α₂), and prints that the leapfrog modified energy W is conserved both below and above the CFL limit (it is positive only below it).
- Correction found while scripting 4.4: the brief asked for "energy W grows above CFL". W is an exact invariant of every leapfrog step; above the limit it becomes indefinite while the field norm grows. The figure plots the field norm.
