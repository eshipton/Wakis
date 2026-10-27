# Copyright 2026 Wakis contributors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate all Tutorial 8.7 data files (`data/*.json`, `data/*.npz`) and figures from scratch.

Usage
-----
Full production regeneration (run by deterministic cloud job)::

    python tutorials/08_vacuum_electronics/08-07_metamaterial_sws_reversed_cherenkov/regenerate.py

Quick verification run::

    python tutorials/08_vacuum_electronics/08-07_metamaterial_sws_reversed_cherenkov/regenerate.py --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parent
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

from _helpers import (  # noqa: E402
    DATA_DIR,
    FIG_DIR,
    render_all_figures,
    run_task_bwo_start_current,
    run_task_cold_test,
    run_task_reversed_cherenkov,
    run_task_sparam_retrieval,
)


def regenerate_all(quick: bool = False) -> dict[str, dict]:
    """Recompute every JSON/NPZ file in ``data/`` from scratch and render all figures.

    Parameters
    ----------
    quick : bool, optional
        If True, use reduced ring-down / post-wake durations for fast verification.
        Default is False (full production run).

    Returns
    -------
    outputs : dict[str, dict]
        Dictionary mapping task names to their freshly generated summary dictionaries.
    """
    t0 = time.perf_counter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cold = run_task_cold_test(quick=quick)
    sparam = run_task_sparam_retrieval(quick=quick)
    rcr = run_task_reversed_cherenkov(quick=quick)
    bwo = run_task_bwo_start_current(quick=quick)
    saved_figs = render_all_figures()

    total_wall_s = time.perf_counter() - t0
    peak_ram_mb = max(
        float(cold["preflight"]["host_mb"]),
        float(sparam["preflight"]["host_mb"]),
        float(rcr["preflight"]["host_mb"]),
        float(bwo["preflight"]["host_mb"]),
    )
    print(
        f"[regenerate.py] Completed (quick={quick}) in {total_wall_s:.2f} s | "
        f"peak preflight RAM = {peak_ram_mb:.2f} MB | "
        f"flux-plane asymmetry offsets {rcr['asymmetry_plane_offsets_cells']} cells -> "
        f"{[round(x, 2) for x in rcr['asymmetry_plane_values_db']]} dB "
        f"(spread = {rcr['asymmetry_plane_spread_db']:.2f} dB) | "
        f"rendered {len(saved_figs)} figures."
    )
    return {
        "cold": cold,
        "sparam": sparam,
        "rcr": rcr,
        "bwo": bwo,
    }


def main() -> None:
    """Parse CLI arguments and run full or quick regeneration."""
    parser = argparse.ArgumentParser(
        description="Regenerate Tutorial 8.7 data/*.json and figures/* from scratch."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced-duration quick verification instead of full production.",
    )
    args = parser.parse_args()
    regenerate_all(quick=bool(args.quick))


if __name__ == "__main__":
    main()
