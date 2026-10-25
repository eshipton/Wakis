"""Tests for Tutorial 3.4: Scattering by a Dielectric Cylinder (2D Mie Theory, TF/SF, and Triangular FIT)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wakis.analytic import cylinder_efficiencies, cylinder_rayleigh_efficiencies
from wakis.sources import GaussianSource
from wakis.unstructured import UnstructuredFIT2D
from wakis.unstructured.meshers import disk_mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = (
    REPO_ROOT / "tutorials" / "03_diffraction_and_scattering" / "03-04_mie_cylinder"
)


class TestMieCylinderAnalyticalAndSavedBenchmarks:
    """Validate analytical cylinder limits and committed Tutorial 3.4 data and figures."""

    def test_saved_data_and_figures_exist(self):
        """Ensure setup schematic, all tutorial figures, and JSON datasets exist."""
        for fname in ("run_full.json", "convergence.json"):
            p = TUTORIAL_DIR / "data" / fname
            assert p.is_file() and p.stat().st_size > 100, f"Missing or empty {p}"

        for fig_name in (
            "00_setup_schematic.png",
            "tfsf_diagram.png",
            "cylinder_theory.png",
            "mesh_te_staircase_vs_conformal.png",
            "fields_xy.png",
            "fields_xy_te.png",
            "qsca_vs_series.png",
            "convergence.png",
            "qsca_triangles_vs_cartesian.png",
            "cylinder_movie.gif",
        ):
            p = TUTORIAL_DIR / "figures" / fig_name
            assert p.is_file() and p.stat().st_size > 1000, f"Missing or empty {p}"

        gif_path = TUTORIAL_DIR / "figures" / "cylinder_movie.gif"
        assert gif_path.stat().st_size < 5 * 1024 * 1024, "GIF exceeds 5 MB limit"

    def test_analytical_rayleigh_and_paradox_limits(self):
        """Verify thin-cylinder Rayleigh scaling and large-x extinction paradox Q -> 2."""
        x_small = np.array([0.005, 0.01])
        for pol in ("TM", "TE"):
            q_exact = cylinder_efficiencies(2.0, x_small, pol)[1]
            q_ray = cylinder_rayleigh_efficiencies(2.0, x_small, pol)
            assert np.allclose(q_exact, q_ray, rtol=3e-3)
            # Power-law check: Q_sca ~ x^3 for lossless cylinder
            assert np.isclose(q_ray[1] / q_ray[0], 8.0, rtol=1e-12)

        x_large = np.linspace(180.0, 220.0, 41)
        for pol in ("TM", "TE"):
            q_mean = float(np.mean(cylinder_efficiencies(2.0, x_large, pol)[1]))
            assert abs(q_mean - 2.0) < 0.08

    def test_saved_full_run_accuracy_and_mutation(self):
        """Verify TM and TE full-resolution Q_sca accuracy and mutation failure."""
        data = json.loads((TUTORIAL_DIR / "data" / "run_full.json").read_text())
        for pol in ("TM", "TE"):
            run = data[pol]
            x = np.asarray(run["x"], dtype=float)
            q_sca = np.asarray(run["q_sca"], dtype=float)
            q_abs = np.asarray(run["q_abs"], dtype=float)
            q_ref = cylinder_efficiencies(2.0, x, pol)[1]

            rel_rms = float(np.sqrt(np.mean((q_sca - q_ref) ** 2) / np.mean(q_ref**2)))
            assert rel_rms < 0.025, f"{pol} relative RMS error {rel_rms:.4f} >= 2.5%"
            assert float(np.max(np.abs(q_abs))) < 0.15

            # Mutation check: comparing against n = 1.5 cylinder must fail
            q_mut = cylinder_efficiencies(1.5, x, pol)[1]
            mut_rms = float(np.sqrt(np.mean((q_sca - q_mut) ** 2) / np.mean(q_mut**2)))
            assert mut_rms > 0.35

    def test_saved_convergence_study_conformal_vs_staircase(self):
        """Verify conformal meshing outperforms staircasing across resolutions."""
        study = json.loads((TUTORIAL_DIR / "data" / "convergence.json").read_text())
        runs = study["runs"]
        for pol in ("TM", "TE"):
            stair_60 = next(
                r
                for r in runs
                if r["pol"] == pol
                and r["conformal"] == "staircase"
                and r["resolution_per_um"] == 60
            )
            conf_60 = next(
                r
                for r in runs
                if r["pol"] == pol
                and r["conformal"] == ("all" if pol == "TM" else "tensor")
                and r["resolution_per_um"] == 60
            )
            assert conf_60["rel_rms_error"] < 0.5 * stair_60["rel_rms_error"]


def test_live_unstructured_cylinder_scattering_and_mutation():
    """Run a compact 2D unstructured triangle FIT cylinder simulation and check scattering."""
    a_cyl = 1.0e-6
    r_out = 2.4e-6
    h = 0.22e-6
    mesh = disk_mesh([a_cyl, r_out], h)

    # Scattering from n = 2 cylinder vs vacuum (n = 1) mutation
    solver_cyl = UnstructuredFIT2D(
        mesh, {0: [4.0, 1.0]}, polarization="TM", bc="abc", cfl=0.9
    )
    solver_vac = UnstructuredFIT2D(
        mesh, {0: [1.0, 1.0]}, polarization="TM", bc="abc", cfl=0.9
    )
    pulse = GaussianSource(frequency=1.5e14, fwidth=0.5e14)
    solver_cyl.add_plane_wave(pulse)
    solver_vac.add_plane_wave(pulse)

    for _ in range(80):
        solver_cyl.one_step()
        solver_vac.one_step()

    norm_cyl = float(np.linalg.norm(solver_cyl.u))
    norm_vac = float(np.linalg.norm(solver_vac.u))
    assert norm_cyl > 1e-3
    # Mutation check: with n = 1 (vacuum), scattered field must vanish identically
    assert norm_vac < 1e-12 * norm_cyl
