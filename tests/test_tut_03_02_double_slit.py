"""Tests for Tutorial 3.2: Young's Double-Slit Interference and Slit Coupling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = (
    REPO_ROOT / "tutorials" / "03_diffraction_and_scattering" / "03-02_double_slit"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


helpers = _load_module("tut_03_02_helpers", TUTORIAL_DIR / "_helpers.py")


class TestDoubleSlitAnalyticalModels:
    """Validate analytical double-slit factorization, fringe spacing, and missing orders."""

    def test_product_law_and_missing_orders(self):
        """Verify cos^2 * sinc^2 factorization and exact missing-order indices."""
        lam = helpers.DEFAULT_WAVELENGTH
        a = 1.5 * lam
        for p_int in (2, 3, 4):
            d = p_int * a
            orders = helpers.missing_order_indices(a, d, max_order=3 * p_int)
            assert orders == [p_int, 2 * p_int, 3 * p_int]

            # At sin(theta) = p * lambda / d = lambda / a, interference factor is 1.0
            # while the single-slit envelope vanishes
            theta_miss = np.array([np.arcsin(lam / a)])
            assert np.isclose(helpers.interference_factor(theta_miss, d, lam)[0], 1.0, rtol=1e-12)
            assert helpers.single_slit_envelope(theta_miss, a, lam)[0] < 1e-28
            assert helpers.double_slit_farfield_pattern(theta_miss, a, d, lam)[0] < 1e-28

        # Rational non-integer d/a = 2.5 -> missing orders at multiples of 5
        assert helpers.missing_order_indices(a, 2.5 * a, max_order=12) == [5, 10]

        with pytest.raises(ValueError, match="Require 0 < slit_width < slit_sep"):
            helpers.missing_order_indices(a, a)

    def test_fringe_spacing_and_rs_line_symmetry(self):
        """Verify Delta(sin theta) = lambda / d and symmetry of 2D RS line profile."""
        lam = helpers.DEFAULT_WAVELENGTH
        a = 0.8 * lam
        d = 2.4 * lam
        assert np.isclose(helpers.fringe_spacing_sin_theta(d, lam), 1.0 / 2.4)
        with pytest.raises(ValueError, match="required"):
            helpers.fringe_spacing_sin_theta(0.0, lam)

        x = np.linspace(-4.0 * lam, 4.0 * lam, 201)
        prof = helpers.double_slit_rayleigh_sommerfeld_line(x, a, d, lam, distance=3.0 * lam)
        assert np.allclose(prof, prof[::-1], rtol=1e-12)
        # Central fringe at x = 0 is a local maximum
        assert prof[100] > prof[90]

    def test_grid_discretization_rules(self):
        """Check cpw >= 20 and >= 4 cells across each slit, central bridge, and screen."""
        lam = helpers.DEFAULT_WAVELENGTH
        g = helpers.build_double_slit_grid(
            slit_width=0.45 * lam,
            slit_sep=0.65 * lam,
            wavelength=lam,
            screen_thickness=0.18 * lam,
            cpw=22,
            min_slit_cells=6,
            min_barrier_cells=4,
            min_screen_cells=4,
        )
        assert g["cells_across_slit"] >= 6
        assert g["cells_across_bridge"] >= 4
        assert g["cells_across_screen"] >= 4
        assert np.allclose(g["x"], -g["x"][::-1], atol=1e-15)


class TestDoubleSlitSavedBenchmarks:
    """Validate committed Tutorial 3.2 datasets and figures against diffraction theory."""

    def test_saved_data_and_figures_exist(self):
        """Ensure all expected .npz archives and .png figures exist and are non-empty."""
        for fname in (
            "double_slit_missing_orders.npz",
            "double_slit_fringes.npz",
            "double_slit_coupling.npz",
            "double_slit_resolution.npz",
        ):
            p = TUTORIAL_DIR / "data" / fname
            assert p.is_file() and p.stat().st_size > 100, f"Missing or empty {p}"

        for fig_name in (
            "double_slit_envelope_and_missing_orders.png",
            "double_slit_fringe_spacing.png",
            "double_slit_coupling_and_convergence.png",
        ):
            p = TUTORIAL_DIR / "figures" / fig_name
            assert p.is_file() and p.stat().st_size > 2000, f"Missing or empty {p}"

    def test_envelope_bounding_and_missing_orders(self):
        """Check L2 agreement, envelope bounding, and missing-order suppression."""
        d = np.load(TUTORIAL_DIR / "data" / "double_slit_missing_orders.npz")
        l2_errors = d["l2_errors"]
        u_miss = d["missing_order_intensity"]
        u_prev = d["neighbor_order_intensity"]
        pats = d["patterns_fit"]
        env_fit = d["single_env_fit"]
        theta_deg = d["theta_deg"]

        assert np.all(l2_errors < 0.06), f"L2 errors too large: {l2_errors}"
        # Missing order is suppressed below 2.2% of peak and is < 1/7 of preceding bright order
        assert np.all(u_miss < 0.022)
        assert np.all(u_miss < u_prev / 7.0)

        # Inside |theta| <= 65 deg, double-slit pattern is bounded by the single-slit envelope (+5% tol)
        mask = np.abs(theta_deg) <= 65.0
        for k in range(pats.shape[0]):
            assert np.all(pats[k, mask] <= env_fit[mask] + 0.05)

    def test_fringe_spacing_scaling(self):
        """Verify measured Delta(sin theta) matches lambda / d within 4%."""
        d = np.load(TUTORIAL_DIR / "data" / "double_slit_fringes.npz")
        dsin_fit = d["dsin_fit"]
        dsin_th = d["dsin_theory"]

        rel_err = np.abs(dsin_fit - dsin_th) / dsin_th
        assert np.all(rel_err < 0.042), f"Fringe spacing relative error too high: {rel_err}"
        assert float(rel_err[-1]) < 0.005
        assert np.all(np.diff(dsin_fit) < 0.0)

    def test_inter_slit_coupling_and_convergence(self):
        """Verify H_z surface-wave coupling exceeds E_z coupling and mesh converges."""
        coup = np.load(TUTORIAL_DIR / "data" / "double_slit_coupling.npz")
        eta_ez = coup["eta_ez"]
        eta_hz = coup["eta_hz"]

        ptp_hz = float(np.max(eta_hz) - np.min(eta_hz))
        ptp_ez = float(np.max(eta_ez) - np.min(eta_ez))
        assert ptp_hz > 0.25
        assert ptp_hz > 1.5 * ptp_ez
        assert 0.90 <= float(np.mean(eta_hz)) <= 1.10
        assert 0.90 <= float(np.mean(eta_ez)) <= 1.10

        res = np.load(TUTORIAL_DIR / "data" / "double_slit_resolution.npz")
        rel = res["rel_to_finest"]
        assert float(rel[-1]) == 0.0
        assert float(rel[0]) > float(rel[-2])
        assert float(rel[-2]) < 0.01


def test_live_double_slit_simulation_and_mutation():
    """Run a compact live 2D FIT double-slit simulation and verify mutation sensitivity."""
    lam = helpers.DEFAULT_WAVELENGTH
    a = 1.4 * lam
    d = 2.8 * lam  # d / a = 2 -> missing order at m = +/- 2
    sim = helpers.run_double_slit_simulation(
        slit_width=a,
        slit_sep=d,
        wavelength=lam,
        screen_thickness=0.14 * lam,
        polarization="Hz",
        geometry_mode="double",
        cpw=20,
        min_slit_cells=6,
        min_barrier_cells=6,
        min_screen_cells=4,
        x_half_span=4.8 * lam,
        y_upstream=1.1 * lam,
        y_downstream=1.8 * lam,
        detector_distance=None,
        n_cycles=10.5,
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

    pat_th = helpers.double_slit_farfield_pattern(theta_rad, a, d, lam, obliquity="rs2")
    mask = np.abs(theta_deg) <= 65.0
    rel_err = np.linalg.norm(pat_fit[mask] - pat_th[mask]) / np.linalg.norm(pat_th[mask])
    assert rel_err < 0.06

    # Mutation check: perturbing slit separation d -> 1.4 * d must break agreement
    pat_mutated = helpers.double_slit_farfield_pattern(
        theta_rad, a, 1.4 * d, lam, obliquity="rs2"
    )
    mutated_err = np.linalg.norm(pat_fit[mask] - pat_mutated[mask]) / np.linalg.norm(
        pat_mutated[mask]
    )
    assert mutated_err > 0.45
