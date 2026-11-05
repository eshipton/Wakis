# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""CLI for enumerating and running individual cached space-charge-wave simulation points.

Supports listing the complete, de-duplicated dependency-ordered simulation points
triggered through ``cached_linearized_beam_drift`` and ``_cached_particle_drift_impl``
by the slow ``tests/test_087_particles_vs_linearized_beam.py`` tests, and running
a single point by index so its result is persisted to ``CEM_SCW_CACHE_DIR``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._space_charge_waves_common import (
    COARSE_GRID,
    FINE_GRID,
    MODULATION_INDEX,
    NOISE_SEEDS,
    PARTICLE_SHAPE,
    _cached_particle_drift_impl,
    cached_linearized_beam_drift,
    scw_cache_key,
)


class _PointCollector:
    """Collect and de-duplicate `(cached_linearized_beam_drift, particle)` simulation points."""

    def __init__(self) -> None:
        self._lb_points: dict[tuple[int, int, float, float], dict[str, Any]] = {}
        self._particle_points: dict[
            tuple[int, int, str, int, float, float, int, str, int], dict[str, Any]
        ] = {}

    def add_linearized_beam(
        self,
        nx: int,
        nz: int,
        rho0_scale: float = 1.0,
        m_mod: float = MODULATION_INDEX,
    ) -> None:
        """Record a `cached_linearized_beam_drift` simulation point."""
        key = (int(nx), int(nz), float(rho0_scale), float(m_mod))
        if key not in self._lb_points:
            self._lb_points[key] = {
                "func": "cached_linearized_beam_drift",
                "args": {
                    "nx": key[0],
                    "nz": key[1],
                    "rho0_scale": key[2],
                    "m_mod": key[3],
                },
            }

    def add_particle(
        self,
        nx: int,
        nz: int,
        profile: str = "sheet_lattice",
        seed: int = 42,
        m_mod: float = MODULATION_INDEX,
        qm_scale: float = 1.0,
        n_macro_per_step: int | None = None,
        gather: str = "galerkin",
        shape: int = PARTICLE_SHAPE,
    ) -> None:
        """Record a `cached_particle_drift` simulation point and its linearized-beam dependency."""
        nx_i = int(nx)
        nz_i = int(nz)
        # _cached_particle_drift_impl always fetches the baseline linearized-beam reference first
        self.add_linearized_beam(nx_i, nz_i, rho0_scale=1.0, m_mod=MODULATION_INDEX)
        n_macro = (
            int(n_macro_per_step)
            if n_macro_per_step is not None
            else (16 if nx_i == COARSE_GRID[0] else 32)
        )
        eff_seed = int(NOISE_SEEDS[0]) if profile == "sheet_lattice" else int(seed)
        key = (
            nx_i,
            nz_i,
            str(profile),
            eff_seed,
            float(m_mod),
            float(qm_scale),
            n_macro,
            str(gather),
            int(shape),
        )
        if key not in self._particle_points:
            self._particle_points[key] = {
                "func": "particle",
                "args": {
                    "nx": key[0],
                    "nz": key[1],
                    "profile": key[2],
                    "seed": key[3],
                    "m_mod": key[4],
                    "qm_scale": key[5],
                    "n_macro": key[6],
                    "gather": key[7],
                    "shape": key[8],
                },
            }

    def add_baseline_pair(
        self,
        nx: int,
        nz: int,
        m_mod: float = MODULATION_INDEX,
        shape: int = PARTICLE_SHAPE,
    ) -> None:
        """Record the calls made by `cached_baseline_pair`."""
        self.add_linearized_beam(nx, nz, rho0_scale=1.0, m_mod=m_mod)
        self.add_particle(
            nx,
            nz,
            profile="sheet_lattice",
            seed=NOISE_SEEDS[0],
            m_mod=m_mod,
            qm_scale=1.0,
            shape=int(shape),
        )

    def add_m_extrapolation(
        self,
        nx: int,
        nz: int,
        include_m4: bool = True,
        shape: int = PARTICLE_SHAPE,
    ) -> None:
        """Record the calls made by `compute_m_extrapolation`."""
        self.add_baseline_pair(nx, nz, m_mod=MODULATION_INDEX, shape=int(shape))
        self.add_particle(
            nx,
            nz,
            profile="sheet_lattice",
            seed=NOISE_SEEDS[0],
            m_mod=0.5 * MODULATION_INDEX,
            qm_scale=1.0,
            shape=int(shape),
        )
        if include_m4:
            self.add_particle(
                nx,
                nz,
                profile="sheet_lattice",
                seed=NOISE_SEEDS[0],
                m_mod=0.25 * MODULATION_INDEX,
                qm_scale=1.0,
                shape=int(shape),
            )

    def add_empirical_noise(
        self,
        grid: tuple[int, int, int] = COARSE_GRID,
        m_mod: float = MODULATION_INDEX,
        n_macro_per_step: int | None = None,
        shape: int = PARTICLE_SHAPE,
    ) -> None:
        """Record the calls made by `measure_empirical_noise`."""
        nx, _, nz = grid
        n_macro = (
            int(n_macro_per_step)
            if n_macro_per_step is not None
            else (256 if nx == COARSE_GRID[0] else 512)
        )
        self.add_linearized_beam(nx, nz, rho0_scale=1.0, m_mod=m_mod)
        self.add_particle(
            nx,
            nz,
            profile="sheet_lattice",
            seed=NOISE_SEEDS[0],
            m_mod=m_mod,
            qm_scale=1.0,
            n_macro_per_step=n_macro,
            shape=int(shape),
        )
        for seed in NOISE_SEEDS:
            self.add_particle(
                nx,
                nz,
                profile="sheet",
                seed=seed,
                m_mod=m_mod,
                qm_scale=1.0,
                n_macro_per_step=n_macro,
                shape=int(shape),
            )

    def points(self) -> list[dict[str, Any]]:
        """Return all collected points with linearized-beam dependencies ordered first."""
        return [*self._lb_points.values(), *self._particle_points.values()]


def _points_test_two_resolution_convergence_and_sheet_ensemble(
    c: _PointCollector,
) -> None:
    """Enumerate cached simulation points for `test_two_resolution_convergence_and_sheet_ensemble`."""
    nx_c, _, nz_c = COARSE_GRID
    nx_f, _, nz_f = FINE_GRID
    c.add_baseline_pair(nx_c, nz_c)
    c.add_baseline_pair(nx_f, nz_f)
    c.add_empirical_noise(COARSE_GRID)
    c.add_empirical_noise(FINE_GRID)
    c.add_m_extrapolation(nx_c, nz_c, include_m4=True)
    c.add_m_extrapolation(nx_f, nz_f, include_m4=False)
    c.add_linearized_beam(nx_c, nz_c, rho0_scale=2.0, m_mod=MODULATION_INDEX)


def _points_test_axial_vs_transverse_refinement(c: _PointCollector) -> None:
    """Enumerate cached simulation points for `test_axial_vs_transverse_refinement`."""
    nx_c, _, nz_c = COARSE_GRID
    nx_f, _, nz_f = FINE_GRID
    m0 = MODULATION_INDEX
    m_half = 0.5 * MODULATION_INDEX

    def _add_meas(nx_g: int, nz_g: int, n_val: int, m_val: float) -> None:
        c.add_linearized_beam(nx_g, nz_g, rho0_scale=1.0, m_mod=m0)
        c.add_particle(
            nx_g,
            nz_g,
            profile="sheet_lattice",
            seed=42,
            m_mod=m_val,
            qm_scale=1.0,
            n_macro_per_step=n_val,
        )

    _add_meas(nx_c, nz_c, 16, m0)
    _add_meas(nx_c, nz_c, 16, m_half)
    _add_meas(nx_c, nz_c, 64, m0)
    _add_meas(nx_c, nz_c, 64, m_half)
    _add_meas(nx_f, nz_f, 32, m0)
    _add_meas(nx_f, nz_f, 32, m_half)
    _add_meas(nx_f, nz_f, 128, m0)
    _add_meas(nx_f, nz_f, 128, m_half)
    _add_meas(nx_c, nz_f, 16, m0)
    _add_meas(nx_f, nz_c, 32, m0)


def _points_test_optional_diagnostics(c: _PointCollector) -> None:
    """Enumerate cached simulation points for `test_optional_diagnostics`."""
    nx, _, nz = COARSE_GRID
    c.add_baseline_pair(nx, nz)
    # A3
    c.add_particle(
        nx, nz, profile="sheet", seed=42, m_mod=MODULATION_INDEX, qm_scale=1.0
    )
    # A5
    c.add_empirical_noise(COARSE_GRID, n_macro_per_step=16)
    c.add_particle(
        nx, nz, profile="sheet", seed=43, m_mod=MODULATION_INDEX, qm_scale=1.0
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=64,
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet",
        seed=43,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=64,
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet_lattice",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=64,
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=128,
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet_lattice",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=128,
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=256,
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet",
        seed=43,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=256,
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet_lattice",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=256,
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet_lattice",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=16,
        gather="momentum",
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=16,
        gather="momentum",
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet_lattice",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=128,
        gather="momentum",
    )
    c.add_particle(
        nx,
        nz,
        profile="sheet",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=128,
        gather="momentum",
    )
    # D3 (_cached_m3_disk_run calls cached_baseline_pair(nx, nz, shape=PARTICLE_SHAPE))
    c.add_baseline_pair(nx, nz, shape=PARTICLE_SHAPE)
    # D6
    c.add_particle(
        nx,
        nz,
        profile="sheet_lattice",
        seed=42,
        m_mod=MODULATION_INDEX,
        qm_scale=1.0,
        n_macro_per_step=64,
    )


TEST_ENUMERATORS: dict[str, Callable[[_PointCollector], None]] = {
    "test_two_resolution_convergence_and_sheet_ensemble": (
        _points_test_two_resolution_convergence_and_sheet_ensemble
    ),
    "test_axial_vs_transverse_refinement": _points_test_axial_vs_transverse_refinement,
    "test_optional_diagnostics": _points_test_optional_diagnostics,
}


def list_points(test_names: Sequence[str]) -> list[dict[str, Any]]:
    """Return the de-duplicated, dependency-ordered point list for `test_names`."""
    collector = _PointCollector()
    for name in test_names:
        if name not in TEST_ENUMERATORS:
            valid = ", ".join(sorted(TEST_ENUMERATORS))
            raise ValueError(f"Unknown test {name!r}; expected one of: {valid}")
        TEST_ENUMERATORS[name](collector)
    return collector.points()


def run_point(test_names: Sequence[str], index: int) -> tuple[str, float]:
    """Execute point `index` for `test_names` through the cached runner and return `(key, secs)`."""
    points = list_points(test_names)
    if index < 0 or index >= len(points):
        raise IndexError(f"Point index {index} out of range for {len(points)} points")
    pt = points[index]
    func = pt["func"]
    args = pt["args"]
    key = scw_cache_key(func, args)
    t0 = time.perf_counter()
    if func == "cached_linearized_beam_drift":
        cached_linearized_beam_drift(
            int(args["nx"]),
            int(args["nz"]),
            rho0_scale=float(args["rho0_scale"]),
            m_mod=float(args["m_mod"]),
        )
    elif func == "particle":
        # Ensure linearized-beam dependency is loaded from cache or computed
        cached_linearized_beam_drift(
            int(args["nx"]),
            int(args["nz"]),
            rho0_scale=1.0,
            m_mod=MODULATION_INDEX,
        )
        _cached_particle_drift_impl(
            int(args["nx"]),
            int(args["nz"]),
            str(args["profile"]),
            int(args["seed"]),
            float(args["m_mod"]),
            float(args["qm_scale"]),
            int(args["n_macro"]),
            str(args["gather"]),
            int(args["shape"]),
        )
    else:
        raise ValueError(f"Unsupported point func {func!r}")
    elapsed = time.perf_counter() - t0
    return key, elapsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List or run cached space-charge-wave simulation points."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Print JSON list of simulation points.")
    p_list.add_argument("tests", nargs="+", help="Test function names.")

    p_run = sub.add_parser("run", help="Run a single simulation point by index.")
    p_run.add_argument("tests", nargs="+", help="Test function names.")
    p_run.add_argument(
        "--index", "-i", type=int, required=True, help="Zero-based point index."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for `examples/scw_points.py`."""
    parser = _build_parser()
    parsed = parser.parse_args(argv)
    if parsed.cmd == "list":
        pts = list_points(parsed.tests)
        print(json.dumps(pts, indent=2), flush=True)
        return 0
    if parsed.cmd == "run":
        key, secs = run_point(parsed.tests, int(parsed.index))
        print(f"POINT i={int(parsed.index)} key={key} secs={secs:.3f}", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
