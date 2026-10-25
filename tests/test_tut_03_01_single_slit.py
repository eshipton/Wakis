"""Tests for Tutorial 3.1: Diffraction by a Single Slit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = (
    REPO_ROOT / "tutorials" / "03_diffraction_and_scattering" / "03-01_single_slit"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


helpers = _load_module("tut_03_01_helpers", TUTORIAL_DIR / "_helpers.py")


class TestSingleSlitAnalyticalModels:
    """Validate analytical Fraunhofer, Kirchhoff, Rayleigh-Sommerfeld, and Fresnel models."""

    def test_farfield_nulls_and_obliquity_hierarchy(self):
        """Verify exact diffraction zeros and RS-I < Kirchhoff < Fraunhofer (= RS-II) obliquity."""
        lam = 0.6e-6
        a = 2.0 * lam
        # First and second diffraction nulls at sin(theta) = +/- 0.5, +/- 1.0
        theta_null1 = np.arcsin(0.5)
        for fn in (
            helpers.fraunhofer_pattern,
            helpers.kirchhoff_obliquity_pattern,
            lambda th, w, wl: helpers.rayleigh_sommerfeld_farfield(th, w, wl, kind="I"),
            lambda th, w, wl: helpers.rayleigh_sommerfeld_farfield(th, w, wl, kind="II"),
        ):
            assert np.isclose(fn(np.array([0.0]), a, lam)[0], 1.0, rtol=1e-12)
            assert fn(np.array([theta_null1]), a, lam)[0] < 1e-28

        # At theta = 60 deg with a = 0.8*lambda (no null at 60 deg), check obliquity factors:
        # RS-I has cos^2(60 deg) = 0.25, Kirchhoff has ((1 + 0.5)/2)^2 = 0.5625, RS-II has 1.0
        a_narrow = 0.8 * lam
        th60 = np.array([np.deg2rad(60.0)])
        u_fraun = float(helpers.fraunhofer_pattern(th60, a_narrow, lam)[0])
        u_kirch = float(helpers.kirchhoff_obliquity_pattern(th60, a_narrow, lam)[0])
        u_rs1 = float(helpers.rayleigh_sommerfeld_farfield(th60, a_narrow, lam, kind="I")[0])
        u_rs2 = float(helpers.rayleigh_sommerfeld_farfield(th60, a_narrow, lam, kind="II")[0])

        assert np.isclose(u_rs2, u_fraun, rtol=1e-12)
        assert np.isclose(u_kirch / u_fraun, 0.5625, rtol=1e-12)
        assert np.isclose(u_rs1 / u_fraun, 0.25, rtol=1e-12)
        assert u_rs1 < u_kirch < u_rs2

        with pytest.raises(ValueError, match="Unknown Rayleigh-Sommerfeld kind"):
            helpers.rayleigh_sommerfeld_farfield(th60, a_narrow, lam, kind="III")

    def test_fresnel_cornu_and_rs_line_profiles(self):
        """Verify Fresnel number scaling, Cornu bright center, and symmetry."""
        lam = 0.6e-6
        a = 2.0 * lam
        assert np.isclose(helpers.fresnel_number(a, lam, distance=1.0 * lam), 1.0)
        assert np.isclose(helpers.fresnel_number(a, lam, distance=2.0 * lam), 0.5)

        with pytest.raises(ValueError, match="required"):
            helpers.fresnel_number(a, lam, distance=0.0)

        # At N_F = 0.50 (L = a^2 / (4 * lam * 0.50)), Cornu spiral has an on-axis bright focus > 1.55
        dist_focus = (a**2) / (4.0 * lam * 0.50)
        x = np.linspace(-2.0 * a, 2.0 * a, 201)
        prof_cornu = helpers.fresnel_cornu_profile(x, a, lam, dist_focus)
        prof_rs1 = helpers.rayleigh_sommerfeld_line_profile(x, a, lam, dist_focus, kind="I")

        assert np.allclose(prof_cornu, prof_cornu[::-1], rtol=1e-12)
        assert np.allclose(prof_rs1, prof_rs1[::-1], rtol=1e-12)
        assert prof_cornu[100] > 1.55
        assert prof_rs1[100] > 1.40

    def test_grid_discretization_rules(self):
        """Check cpw >= 20, >= 4 cells across slit and screen thickness, and x-symmetry."""
        lam = helpers.DEFAULT_WAVELENGTH
        g = helpers.build_single_slit_grid(
            slit_width=0.6 * lam,
            wavelength=lam,
            screen_thickness=0.15 * lam,
            cpw=24,
            min_slit_cells=6,
            min_screen_cells=4,
        )
        assert g["cells_across_slit"] >= 6
        assert g["cells_across_screen"] >= 4
        assert np.allclose(g["x"], -g["x"][::-1], atol=1e-15)


class TestSingleSlitSavedBenchmarks:
    """Validate committed tutorial datasets and figures against diffraction theory."""

    def test_saved_data_and_figures_exist(self):
        """Ensure all expected .npz archives and .png figures are present and non-empty."""
        for fname in (
            "single_slit_source.npz",
            "single_slit_fresnel.npz",
            "single_slit_farfield.npz",
            "single_slit_transmission.npz",
            "single_slit_resolution.npz",
        ):
            p = TUTORIAL_DIR / "data" / fname
            assert p.is_file() and p.stat().st_size > 100, f"Missing or empty {p}"

        for fig_name in (
            "00a_setup_geometry.png",
            "00b_setup_mesh_and_excitation.png",
            "single_slit_fresnel_and_source.png",
            "single_slit_farfield_patterns.png",
            "single_slit_transmission_and_convergence.png",
        ):
            p = TUTORIAL_DIR / "figures" / fig_name
            assert p.is_file() and p.stat().st_size > 2000, f"Missing or empty {p}"

        table_md = TUTORIAL_DIR / "figures" / "00_setup_table.md"
        assert table_md.is_file() and table_md.stat().st_size > 128, f"Missing {table_md}"

    def test_source_injection_and_parseval_identity(self):
        """Full-width plane-wave source has < 1% ripple vs > 15% for PML-truncated source."""
        d = np.load(TUTORIAL_DIR / "data" / "single_slit_source.npz")
        ripple_full = float(d["ripple_full"])
        ripple_trunc = float(d["ripple_trunc"])
        flux_diff = float(d["flux_field_rel_diff"])
        parseval = float(d["parseval_ratio_empty"])

        assert ripple_full < 0.01
        assert ripple_trunc > 0.15
        assert ripple_trunc > 20.0 * ripple_full
        assert flux_diff < 0.02
        assert 0.98 <= parseval <= 1.02

    def test_fresnel_nearfield_agreement(self):
        """Near-field FIT line profiles agree with 2D Rayleigh-Sommerfeld I across N_F."""
        d = np.load(TUTORIAL_DIR / "data" / "single_slit_fresnel.npz")
        err_rs1 = d["l2_errors_rs1"]
        nf_vals = d["nf_values"]
        fit_profs = d["fit_profiles"]
        x_oa = d["x_over_a"]

        assert np.all(err_rs1 < 0.16), f"Fresnel L2 errors too high: {err_rs1}"
        # At N_F = 0.5, on-axis intensity exhibits Fresnel focusing I(0)/I_inc > 1.15
        idx_nf05 = int(np.argmin(np.abs(nf_vals - 0.5)))
        idx_x0 = int(np.argmin(np.abs(x_oa)))
        assert float(fit_profs[idx_nf05, idx_x0]) > 1.15

    def test_farfield_obliquity_and_polarization_splitting(self):
        """Verify far-field agreement for wide slits and polarization divergence at a = 0.6*lambda."""
        d = np.load(TUTORIAL_DIR / "data" / "single_slit_farfield.npz")
        ratios = d["a_over_lam"]
        l2_ez_rs1 = d["l2_ez_vs_rs1"]
        l2_ez_fraun = d["l2_ez_vs_fraun"]
        parseval_ez = d["parseval_ez"]
        parseval_hz = d["parseval_hz"]

        # Wide slits (a/lambda = 4.0, 2.0): L2 error vs Fraunhofer < 7% and vs RS-I < 11%
        assert float(l2_ez_fraun[0]) < 0.05
        assert float(l2_ez_rs1[0]) < 0.06
        assert float(l2_ez_fraun[1]) < 0.07
        assert float(l2_ez_rs1[1]) < 0.11
        # For narrow slit (a/lambda = 0.6), RS-I (with cos^2 obliquity) is much closer
        # to E_z than obliquity-free Fraunhofer sinc^2 (10.2% vs 22.2%)
        assert float(l2_ez_rs1[-1]) < 0.55 * float(l2_ez_fraun[-1])
        assert np.all((parseval_ez > 0.97) & (parseval_ez < 1.03))
        assert np.all((parseval_hz > 0.95) & (parseval_hz < 1.05))

        # At a/lambda = 0.6 and |theta| = 55 deg, H_z (E perp slit) radiates much more strongly
        # than E_z (E || slit)
        theta_deg = d["theta_deg"]
        idx55 = int(np.argmin(np.abs(theta_deg - 55.0)))
        u_ez_55 = float(d["u2d_ez_norm"][-1, idx55])
        u_hz_55 = float(d["u2d_hz_norm"][-1, idx55])
        assert u_hz_55 > 1.5 * u_ez_55

    def test_transmission_cutoff_and_resolution_convergence(self):
        """Verify TE_10 waveguide cutoff for E_z vs TEM funneling for H_z, and mesh convergence."""
        trans = np.load(TUTORIAL_DIR / "data" / "single_slit_transmission.npz")
        ratios = trans["a_over_lam"]
        tau_ez_thick = trans["tau_ez_thick"]
        tau_hz_thick = trans["tau_hz_thick"]
        tau_ez_sheet = trans["tau_ez_sheet"]

        # Below cutoff (a/lambda < 0.42): thick PEC screen strongly attenuates E_z, while H_z > 0.95
        sub_mask = ratios < 0.42
        assert np.all(tau_ez_thick[sub_mask] < 0.16)
        assert np.all(tau_hz_thick[sub_mask] > 0.95)
        # Finite screen thickness attenuates sub-cutoff E_z strongly relative to t=0 sheet
        assert np.all(tau_ez_thick[sub_mask] < 0.25 * tau_ez_sheet[sub_mask])
        # Above a/lambda >= 1.4, both polarizations approach tau ~ 1
        wide_mask = ratios >= 1.4
        assert np.all(tau_ez_thick[wide_mask] > 0.85)
        assert np.all(tau_hz_thick[wide_mask] > 0.85)

        res = np.load(TUTORIAL_DIR / "data" / "single_slit_resolution.npz")
        err = res["l2_vs_rs1"]
        rel = res["rel_to_finest"]
        assert float(err[-1]) < float(err[0])
        assert float(rel[-1]) == 0.0
        assert float(rel[0]) < 0.02


def test_live_single_slit_simulation_and_mutation():
    """Run a compact live 2D FIT single-slit simulation and verify mutation failure sensitivity."""
    lam = helpers.DEFAULT_WAVELENGTH
    sim = helpers.run_single_slit_simulation(
        slit_width=1.8 * lam,
        wavelength=lam,
        screen_thickness=0.12 * lam,
        polarization="Ez",
        with_screen=True,
        cpw=20,
        min_slit_cells=6,
        min_screen_cells=4,
        x_half_span=4.0 * lam,
        y_upstream=1.1 * lam,
        y_downstream=1.8 * lam,
        n_cycles=10.0,
    )
    assert sim["cells_across_slit"] >= 6
    assert sim["cells_across_screen"] >= 4
    assert 0.97 <= sim["parseval_ratio"] <= 1.03

    theta_rad = sim["theta_rad"]
    theta_deg = sim["theta_deg"]
    idx0 = int(np.argmin(np.abs(theta_deg)))
    pat_fit = sim["u2d"] / float(sim["u2d"][idx0])
    # Symmetric pattern: U(-theta) == U(+theta)
    assert np.allclose(pat_fit, pat_fit[::-1], atol=1e-2)

    pat_fraun = helpers.fraunhofer_pattern(theta_rad, 1.8 * lam, lam)
    mask = np.abs(theta_deg) <= 65.0
    rel_err = np.linalg.norm(pat_fit[mask] - pat_fraun[mask]) / np.linalg.norm(pat_fraun[mask])
    assert rel_err < 0.12

    # Mutation check: if slit width in theoretical model is mutated by 1.5x, L2 error must fail threshold
    pat_mutated = helpers.fraunhofer_pattern(theta_rad, 2.7 * lam, lam)
    mutated_err = np.linalg.norm(pat_fit[mask] - pat_mutated[mask]) / np.linalg.norm(
        pat_mutated[mask]
    )
    assert mutated_err > 0.25
