# Tutorial 8.8 · Reversed-Cherenkov Wakefields in an All-Metal Metamaterial (Wagon-Wheel) Structure

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/08_vacuum_electronics/08-08_mtm_wakefield_wagon_wheel/mtm_wakefield_wagon_wheel.ipynb)

Reproduces with Wakis's 3D finite-integration-technique (FIT) time-domain solver (`SolverFIT3D`, `WakeSolver`, subcell conformal PEC, `wakis.bloch`, and `ResonanceMonitor`) the all-metal wagon-wheel metamaterial reversed-Cherenkov wakefield experiment of X. Lu *et al.*, *Phys. Rev. Lett.* **122**, 014801 (2019).

## Files

- [`mtm_wakefield_wagon_wheel.py`](mtm_wakefield_wagon_wheel.py) — Single-source Jupytext percent script.
- [`mtm_wakefield_wagon_wheel.ipynb`](mtm_wakefield_wagon_wheel.ipynb) — Executed Jupyter notebook built via `python tutorials/build.py 08-08 --execute`.
- [`_helpers.py`](_helpers.py) — Geometry builders, 3D Bloch cold-test solver, finite-structure wakefield solver, two-bunch coherent superposition verifier, analytical backward-wave power formulas, and figure generators.
- [`generate_data.py`](generate_data.py) — CLI driver to regenerate `data/*.json`, `data/*.npz`, and `figures/*`.
- [`data/`](data/) — Saved simulation outputs and summary metrics (`summary_metrics.json`, `task_01_*.json` through `task_05_*.json`).
- [`figures/`](figures/) — Setup schematics (`00a`, `00b`, `00_setup_table.md`), dispersion, mesh convergence, wakefield length scaling, Poynting flux directionality, wave-packet snapshots (`06`), animated GIF (`07_reversed_cherenkov_wave_packet.gif`), two-bunch superposition (`08`), and energy balance (`09`).
