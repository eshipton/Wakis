"""Driver script to regenerate all JSON/NPZ data files and figures for Tutorial 8.8.

Usage:
    # Full production regeneration:
    python3 tutorials/08_vacuum_electronics/08-08_mtm_wakefield_wagon_wheel/generate_data.py

    # Fast validation mode (--quick) into a target directory:
    python3 tutorials/08_vacuum_electronics/08-08_mtm_wakefield_wagon_wheel/generate_data.py --quick --data-dir /tmp/t88_quick
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    aggregate_all_tutorial_data,
    build_summary_metrics,
    generate_all_tutorial_data,
    generate_all_tutorial_figures,
    plot_mtm_setup_schematics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Tutorial 8.8 data and figure generation."""
    parser = argparse.ArgumentParser(
        description="Regenerate Tutorial 8.8 wagon-wheel metamaterial wakefield data files and figures."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced-resolution quick simulations for fast CI verification.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Recompute all derived/aggregate JSON quantities and figures from existing raw per-run data without running FIT simulations.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Subset of task numbers (1..6) to execute.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation and only write data/*.json and data/*.npz.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only rebuild summary_metrics.json and figures/* from existing task_0*.json files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HERE / "data",
        help="Directory for output JSON and NPZ files.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=HERE / "figures",
        help="Directory for output PNG, GIF, and Markdown setup tables.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the Tutorial 8.8 data and figure pipeline."""
    args = parse_args(argv)
    if args.aggregate_only:
        figs_dir = None if args.no_figures else args.figures_dir
        aggregate_all_tutorial_data(
            args.data_dir,
            figs_dir,
            source_data_dir=HERE / "data",
        )
        return 0

    if args.figures_only:
        req_files = [
            args.data_dir / "task_01_geometry_and_port_reflection.json",
            args.data_dir / "task_02_cold_test_and_convergence.json",
            args.data_dir / "task_03_wakefield_and_directionality.json",
            args.data_dir / "task_04_two_bunch_superposition.json",
            args.data_dir / "task_05_analytic_power_and_energy_balance.json",
        ]
        if all(f.is_file() for f in req_files):
            build_summary_metrics(args.data_dir)
        plot_mtm_setup_schematics(args.figures_dir)
        generate_all_tutorial_figures(args.data_dir, args.figures_dir)
        return 0

    figs_dir = None if args.no_figures else args.figures_dir
    generate_all_tutorial_data(
        args.data_dir,
        figs_dir,
        quick=args.quick,
        tasks=tuple(args.tasks),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
