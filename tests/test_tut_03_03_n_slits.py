"""Tests for Tutorial 3.3: N-Slit Diffraction Gratings, Resolving Power, and Periodic Scaling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = (
    REPO_ROOT / "tutorials" / "03_diffraction_and_scattering" / "03-03_n_slits"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


helpers = _load_module("tut_03_03_helpers", TUTORIAL_DIR / "_helpers.py")


class TestNSlitAnalyticalModels:
    """Validate N-slit interference factor, (N-2) secondary maxima, and Rayleigh resolving power."""

    def test_interference_factor_principals_and_zeros(self):
        """Verify principal maxima at m*lambda/d and (N-1) exact zeros between orders."""
        lam = helpers.DEFAULT_WAVELENGTH
        d = 1.8 * lam
        a = 0.6 * lam

        # For N = 2, N-slit formula must match cos^2(pi * d * sin(theta) / lambda)
        theta = np.linspace(-1.2, 1.2, 241)
        i_n2 = helpers.n_slit_interference_factor(theta, 2, d, lam)
        i_cos2 = np.cos(np.pi * d * np.sin(theta) / lam) ** 2
        assert np.allclose(i_n2, i_cos2, rtol=1e-12, atol=1e-12)

        for n_slits in (3, 4, 6, 8):
            # Principal maxima m = -1, 0, +1 give unit interference factor
            orders, th_m = helpers.grating_equation_angles(d, lam)
            assert list(orders) == [-1, 0, 1]
            assert np.allclose(
                helpers.n_slit_interference_factor(th_m, n_slits, d, lam),
                1.0,
                rtol=1e-12,
            )

            # All (N - 1) intermediate zeros q = 1, ..., N - 1 vanish identically
            q_vals = np.arange(1, n_slits, dtype=float)
            th_zeros = np.arcsin((q_vals / n_slits) * (lam / d))
            u_zeros = helpers.n_slit_farfield_pattern(th_zeros, n_slits, a, d, lam)
            assert np.all(u_zeros < 1e-26)
            assert helpers.secondary_maxima_count(n_slits) == n_slits - 2

        with pytest.raises(ValueError, match="n_slits must be >= 1"):
            helpers.n_slit_interference_factor(theta, 0, d, lam)

    def test_peak_width_and_resolving_power_formulas(self):
        """Verify 1/N peak width scaling and R = |m| * N."""
        lam = helpers.DEFAULT_WAVELENGTH
        d = 1.8 * lam
        n_vals = [2, 4, 8]
        hwzms = [helpers.principal_peak_hwzm_sin_theta(n, d, lam) for n in n_vals]
        fwhms = [helpers.principal_peak_fwhm_sin_theta(n, d, lam) for n in n_vals]

        # Halving the width when doubling N
        assert np.isclose(hwzms[0] / hwzms[1], 2.0)
        assert np.isclose(hwzms[1] / hwzms[2], 2.0)
        assert np.isclose(fwhms[0] / fwhms[1], 2.0)

        assert helpers.rayleigh_resolving_power(6, 1) == 6
        assert helpers.rayleigh_resolving_power(6, -2) == 12
        with pytest.raises(ValueError, match="non-zero"):
            helpers.rayleigh_resolving_power(6, 0)

    def test_grid_discretization_rules(self):
        """Verify cpw >= 20 and >= 4 cells across every slit, inter-slit bar, and screen."""
        lam = helpers.DEFAULT_WAVELENGTH
        g = helpers.build_n_slit_grid(
            n_slits=5,
            slit_width=0.6 * lam,
            period=1.8 * lam,
            wavelength=lam,
            screen_thickness=0.14 * lam,
            cpw=22,
            min_slit_cells=6,
            min_barrier_cells=6,
            min_screen_cells=4,
        )
        assert g["cells_across_slit"] >= 6
        assert g["cells_across_bridge"] >= 6
        assert g["cells_across_screen"] >= 4
        assert np.allclose(g["x"], -g["x"][::-1], atol=1e-14)


class TestNSlitSavedBenchmarks:
    """Validate committed Tutorial 3.3 datasets and figures against grating theory."""

    def test_saved_data_and_figures_exist(self):
        """Ensure all expected .npz archives and .png figures exist and are non-empty."""
        for fname in (
            "n_slits_patterns.npz",
            "n_slits_resolving_power.npz",
            "n_slits_periodic_vs_finite.npz",
            "n_slits_resolution.npz",
        ):
            p = TUTORIAL_DIR / "data" / fname
            assert p.is_file() and p.stat().st_size > 100, f"Missing or empty {p}"

        for fig_name in (
            "00_setup_schematic.png",
            "n_slits_patterns_and_secondary_maxima.png",
            "n_slits_resolving_power_and_width.png",
            "n_slits_periodic_vs_finite_and_cost.png",
        ):
            p = TUTORIAL_DIR / "figures" / fig_name
            assert p.is_file() and p.stat().st_size > 2000, f"Missing or empty {p}"

    def test_patterns_secondary_maxima_and_width_scaling(self):
        """Verify L2 agreement, exact (N-2) secondary peak count, and 1/N width scaling."""
        d = np.load(TUTORIAL_DIR / "data" / "n_slits_patterns.npz")
        l2_errors = d["l2_errors"]
        n_sec_counted = d["n_sec_counted"]
        n_sec_expected = d["n_sec_expected"]
        hwzm_fit = d["hwzm_sin_fit"]
        hwzm_th = d["hwzm_sin_theory"]

        assert np.all(l2_errors < 0.075), f"L2 errors too large: {l2_errors}"
        assert np.array_equal(n_sec_counted, n_sec_expected)

        rel_hwzm = np.abs(hwzm_fit - hwzm_th) / hwzm_th
        assert np.all(rel_hwzm < 0.05), f"HWZM relative error too large: {rel_hwzm}"
        assert np.all(np.diff(hwzm_fit) < 0.0)

    def test_rayleigh_resolving_power_doublet(self):
        """Verify sub-Rayleigh, exact Rayleigh, and super-Rayleigh doublet."""
        d = np.load(TUTORIAL_DIR / "data" / "n_slits_resolving_power.npz")
        saddles = d["saddle_to_peak_fit"]
        saddles_th = d["saddle_to_peak_th"]

        # Strictly deeper dip as wavelength separation increases
        assert saddles[0] > saddles[1] > saddles[2]
        assert saddles[0] > 0.96
        assert 0.90 <= saddles[1] <= 0.97
        assert saddles[2] < 0.45
        assert np.allclose(saddles, saddles_th, atol=0.04)

    def test_periodic_vs_finite_and_convergence(self):
        """Verify finite-N lobe fractions approach periodic cell and mesh converges."""
        pat = np.load(TUTORIAL_DIR / "data" / "n_slits_patterns.npz")
        per = np.load(TUTORIAL_DIR / "data" / "n_slits_periodic_vs_finite.npz")
        res = np.load(TUTORIAL_DIR / "data" / "n_slits_resolution.npz")

        finite_largest_m0 = float(pat["lobe_fractions"][-1, 1])
        periodic_m0 = float(per["periodic_order_fractions"][1])
        assert abs(finite_largest_m0 - periodic_m0) < 0.02

        # Single periodic cell uses < 12% of the cells of the largest finite-N domain
        assert int(per["periodic_n_cells"]) < 0.12 * int(pat["n_cells"][-1])

        rel = res["rel_to_finest"]
        assert float(rel[-1]) == 0.0
        assert float(rel[0]) > float(rel[-2])
        assert float(rel[-2]) < 0.01


def test_live_n_slit_simulation_and_mutation():
    """Run a compact live N = 3 grating simulation and verify mutation sensitivity."""
    lam = helpers.DEFAULT_WAVELENGTH
    a = 0.6 * lam
    d = 1.8 * lam
    sim = helpers.run_n_slit_simulation(
        n_slits=3,
        slit_width=a,
        period=d,
        wavelength=lam,
        screen_thickness=0.14 * lam,
        polarization="Hz",
        with_screen=True,
        cpw=20,
        min_slit_cells=6,
        min_barrier_cells=6,
        min_screen_cells=4,
        x_half_span=None,
        y_upstream=1.2 * lam,
        y_downstream=2.0 * lam,
        record_2d_dft=False,
        n_cycles=11.5,
    )
    assert sim["cells_across_slit"] >= 6
    assert sim["cells_across_bridge"] >= 6
    assert sim["cells_across_screen"] >= 4
    assert 0.97 <= sim["parseval_ratio"] <= 1.03

    theta_deg = sim["theta_deg"]
    theta_rad = sim["theta_rad"]
    idx0 = int(np.argmin(np.abs(theta_deg)))
    pat_fit = sim["u2d"] / float(sim["u2d"][idx0])
    assert np.allclose(pat_fit, pat_fit[::-1], atol=1e-2)

    pat_th = helpers.n_slit_farfield_pattern(theta_rad, 3, a, d, lam, obliquity="rs2")
    mask = np.abs(theta_deg) <= 65.0
    rel_err = np.linalg.norm(pat_fit[mask] - pat_th[mask]) / np.linalg.norm(pat_th[mask])
    assert rel_err < 0.075

    # Mutation check: comparing the N = 3 FIT result against N = 6 theory must fail
    pat_mutated = helpers.n_slit_farfield_pattern(
        theta_rad, 6, a, d, lam, obliquity="rs2"
    )
    mutated_err = np.linalg.norm(pat_fit[mask] - pat_mutated[mask]) / np.linalg.norm(
        pat_mutated[mask]
    )
    assert mutated_err > 0.35
