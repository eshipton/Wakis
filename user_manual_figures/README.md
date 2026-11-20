# User manual figures

Illustrations for `user_manual.md`, one folder per section. Illustrations only: they explain concepts and quote
numbers from the manual text; they are not results.

- **Section 4:** see `section4/README.md`.
- **Sections 5–16 (47 figures):** briefs in `sectionN/specs/sectionN.json`. Each figure has a `route` and the
  manual lines it illustrates (`source_lines`). Every number in a brief was checked against the manual.
  - `generated` (27 figures): PaperVizAgent (google-research/papervizagent, Apache-2.0) with Gemini through Vertex AI
    (`tools/run_section.sh`); three candidates each, the picked one in `sectionN/picks/`. Figure 9.2 was corrected
    with `tools/run_refine.sh` (`section9/specs/section9_refine.json`): the CFS α profile now falls linearly from
    α_max at the inner face to zero at the wall, as in `wakis/boundaries.py`.
  - `scripted` (20 figures): matplotlib, drawn from the formulas in the manual or from the real `wakis` API, with
    asserts on the drawn mathematics: `tools/sections_5_6_7_8_scripted.py`, `tools/sections_9_10_11_12_scripted.py`,
    `tools/sections_13_14_15_16_scripted.py` (usage: `python <script> OUTROOT`).
- Final images are palette-quantized to 128 colours. `*.png` is gitignored in this repository, so figure files are
  added with `git add -f`.
