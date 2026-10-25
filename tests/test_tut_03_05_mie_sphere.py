"""Tests for Tutorial 3.5: Mie Scattering by a Dielectric Sphere (3D TF/SF and Near-to-Far Transform)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.constants import c as c0

from wakis.analytic import (
    mie_differential_cross_section,
    mie_efficiencies,
    rayleigh_efficiencies,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = (
    REPO_ROOT / "tutorials" / "03_diffraction_and_scattering" / "03-05_mie_sphere"
)


class TestMieSphereAnalyticalAndSavedBenchmarks:
    """Validate 3D Mie sphere analytical limits and committed Tutorial 3.5 data and figures."""

    def test_saved_data_and_figures_exist(self):
        """Ensure setup schematic, all tutorial figures, and JSON dataset exist."""
        run_full_path = TUTORIAL_DIR / "data" / "run_full.json"
        assert run_full_path.is_file() and run_full_path.stat().st_size > 100

        for fig_name in (
            "00a_setup_geometry.png",
            "00b_setup_mesh_and_excitation.png",
            "tfsf_diagram.png",
            "mie_theory.png",
            "mesh_staircase_vs_conformal.png",
            "fields_xz.png",
            "fields_xz_log.png",
            "qsca_vs_mie.png",
            "far_field_vs_mie.png",
            "mie_movie.gif",
        ):
            p = TUTORIAL_DIR / "figures" / fig_name
            assert p.is_file() and p.stat().st_size > 1000, f"Missing or empty {p}"

        table_path = TUTORIAL_DIR / "figures" / "00_setup_table.md"
        assert table_path.is_file() and table_path.stat().st_size > 100, f"Missing or empty {table_path}"

        gif_path = TUTORIAL_DIR / "figures" / "mie_movie.gif"
        assert gif_path.stat().st_size < 5 * 1024 * 1024, "GIF exceeds 5 MB limit"

    def test_analytical_rayleigh_and_paradox_limits(self):
        """Verify x^4 Rayleigh scaling and large-x extinction paradox Q_sca -> 2."""
        x_small = np.array([0.02, 0.04])
        q_sca = mie_efficiencies(2.0, x_small)[1]
        q_ray, _ = rayleigh_efficiencies(2.0, x_small)
        assert np.allclose(q_sca, q_ray, rtol=3e-3)
        assert np.isclose(q_ray[1] / q_ray[0], 16.0, rtol=1e-12)

        x_large = np.linspace(180.0, 220.0, 41)
        q_mean = float(np.mean(mie_efficiencies(2.0, x_large)[1]))
        assert abs(q_mean - 2.0) < 0.08

    def test_saved_full_run_accuracy_farfield_and_mutation(self):
        """Verify 3D sphere Q_sca and Near2Far accuracy and mutation sensitivity."""
        d = json.loads((TUTORIAL_DIR / "data" / "run_full.json").read_text())
        x = np.asarray(d["x"], dtype=float)
        q_sca = np.asarray(d["q_sca"], dtype=float)
        q_abs = np.asarray(d["q_abs"], dtype=float)
        q_mie = mie_efficiencies(2.0, x)[1]

        rel_rms = float(np.sqrt(np.mean((q_sca - q_mie) ** 2) / np.mean(q_mie**2)))
        assert rel_rms < 0.04, f"Relative RMS error {rel_rms:.4f} >= 4%"
        assert float(np.max(np.abs(q_abs))) < 0.25

        # Far-field E and H plane patterns at x = 3, 6, 9 must agree within 1.5 dB RMS
        ff = d["far_field"]
        for key in ("x3_E", "x3_H", "x6_E", "x6_H", "x9_E", "x9_H"):
            assert ff[key]["rms_db"] < 1.5, f"{key} RMS dB {ff[key]['rms_db']} >= 1.5"
            assert ff[key]["rel_l2"] < 0.12

        # Mutation check: comparing against n = 1.5 sphere must fail both Q_sca and far field
        q_mut = mie_efficiencies(1.5, x)[1]
        mut_rms = float(np.sqrt(np.mean((q_sca - q_mut) ** 2) / np.mean(q_mut**2)))
        assert mut_rms > 0.35

        theta = np.linspace(0.0, np.pi, 181)
        r_sph = 1.0e-6
        lam_x3 = 2.0 * np.pi * r_sph / 3.0
        mie_mut_e = mie_differential_cross_section(1.5, r_sph, lam_x3, theta, 0.0)
        db_mut = 10.0 * np.log10(mie_mut_e / r_sph**2)
        db_sim = np.asarray(ff["x3_E"]["db_sim"], dtype=float)
        assert float(np.sqrt(np.mean((db_sim - db_mut) ** 2))) > 4.0

    def test_pml_gap_study_resolution_improvement(self):
        """Verify refining from 40 to 50 cells/um reduces Q_sca and far-field errors."""
        study_dir = REPO_ROOT / "studies" / "sphere-pml-gap" / "data"
        d40 = json.loads((study_dir / "uniform_gap1x.json").read_text())
        d50 = json.loads((study_dir / "uniform_gap1x_res50.json").read_text())

        assert d50["rel_rms_error"] < d40["rel_rms_error"]
        assert d50["far_field"]["x9_E"]["rms_db"] < d40["far_field"]["x9_E"]["rms_db"]
        assert d50["far_field"]["x9_H"]["rms_db"] < d40["far_field"]["x9_H"]["rms_db"]
