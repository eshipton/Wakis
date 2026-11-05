# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Fast unit and call-graph completeness tests for the SCW disk cache and `examples/scw_points.py`."""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples._space_charge_waves_common as scw
import tests.test_087_particles_vs_linearized_beam as t087
from examples import scw_points
from wakis import GridFIT3D, SolverFIT3D
from wakis.beam_fluid import LinearizedBeam


def _clear_all_scw_lru_caches() -> None:
    """Clear all in-memory `@lru_cache` wrappers in `_space_charge_waves_common` and `test_087`."""
    scw.cached_linearized_beam_drift.cache_clear()
    scw._cached_particle_drift_impl.cache_clear()
    scw.cached_baseline_pair.cache_clear()
    scw.compute_m_extrapolation.cache_clear()
    scw.measure_empirical_noise.cache_clear()
    t087._cached_m3_disk_run.cache_clear()


def _make_fake_simulators() -> tuple[Any, Any, dict[str, int], list[dict[str, Any]]]:
    """Create cheap deterministic fake `LinearizedBeam` and `ParticleSolver` runners."""
    params = scw.compute_analytic_parameters()
    counts = {"lb": 0, "part": 0}
    recorded: list[dict[str, Any]] = []
    beam_templates: dict[tuple[int, int], LinearizedBeam] = {}

    def _get_beam(nx: int, nz: int, rho0_scale: float, m_mod: float) -> LinearizedBeam:
        key = (int(nx), int(nz))
        if key not in beam_templates:
            a = scw.TUNNEL_HALF_WIDTH_M
            b = scw.BEAM_HALF_THICKNESS_M
            ly = scw.PERIODIC_LY_M
            lz = scw.TUNNEL_LENGTH_M
            # Use nz_grid=8 for fast SolverFIT3D construction, then set axial coordinates for nz
            grid = GridFIT3D(
                -a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, int(nx), 1, 8, verbose=0
            )
            solver = SolverFIT3D(
                grid,
                cfln=scw.CFL_NUMBER,
                bc_low=["pec", "periodic", "pec"],
                bc_high=["pec", "periodic", "pec"],
                use_gpu=False,
                verbose=0,
            )

            def _inflow_closure(t: float) -> float:
                return float(m_mod * np.sin(params["omega"] * t))

            beam = LinearizedBeam(
                u0=params["u_dep"],
                rho0=float(rho0_scale) * params["rho_dep"],
                gamma0=params["gamma_dep"],
                axis="z",
                half_thickness=b,
                inflow_rho1=_inflow_closure,
                solver=solver,
            )
            dz = lz / int(nz)
            m_edges = int(nz) - 1
            beam.M = m_edges
            beam.k_indices = np.arange(m_edges, dtype=int)
            beam.s_coords = (np.arange(m_edges, dtype=np.float64) + 0.5) * dz
            beam.ds_line = np.full(m_edges, dz, dtype=np.float64)
            beam.dA_perp = np.broadcast_to(
                beam.dA_perp[:1, :], (m_edges, beam.N_lines)
            ).copy()
            beam.dV = np.broadcast_to(beam.dV[:1, :], (m_edges, beam.N_lines)).copy()
            beam.flat_idx_2d = np.zeros((m_edges, beam.N_lines), dtype=np.int64)
            beam.flat_idx_host = beam.flat_idx_2d.ravel()
            beam.flat_idx = beam.flat_idx_host.copy()
            beam.w_axial = np.ones(m_edges, dtype=np.float64)
            beam.v1 = np.full((m_edges, beam.N_lines), 1.25, dtype=np.float64)
            beam.rho1 = np.full((m_edges, beam.N_lines), -2.5e-6, dtype=np.float64)
            beam.v1_prev = beam.v1.copy()
            beam.rho1_prev = beam.rho1.copy()
            beam.J1 = beam.rho0[:1, :] * beam.v1 + beam.u0 * beam.rho1
            beam.J_old = np.zeros(beam.flat_idx_host.size, dtype=np.float64)
            beam._step_count = 12
            beam_templates[key] = beam
        return beam_templates[key]

    def fake_run_lb(
        nx: int,
        nz: int,
        rho0_scale: float = 1.0,
        m_mod: float = scw.MODULATION_INDEX,
        rho0_profile: Any = None,
        outflow_ramp_m: float = scw.OUTFLOW_RAMP_M,
    ) -> dict[str, Any]:
        counts["lb"] += 1
        if rho0_profile is None and float(outflow_ramp_m) == float(scw.OUTFLOW_RAMP_M):
            recorded.append(
                {
                    "func": "cached_linearized_beam_drift",
                    "args": {
                        "nx": int(nx),
                        "nz": int(nz),
                        "rho0_scale": float(rho0_scale),
                        "m_mod": float(m_mod),
                    },
                }
            )
        beam = _get_beam(int(nx), int(nz), float(rho0_scale), float(m_mod))
        z = beam.s_coords.copy()
        bq = float(params["beta_q"] * np.sqrt(float(rho0_scale)))
        amp = float((float(m_mod) / scw.MODULATION_INDEX) * params["I1_0"])
        i1 = amp * np.cos(bq * z) * np.exp(-1j * params["beta_e"] * z)
        i1[0] = amp
        return {
            "beam": beam,
            "z_coords": z,
            "i1_lb": i1,
            "i2_lb": 1e-6 * i1,
            "dc_leakage_lb": 1e-16,
            "dirichlet_leak_lb": 1e-16,
            "dt": 1e-12,
            "n_spp": 100,
        }

    def fake_run_particle(
        nx: int,
        nz: int,
        beam_ref: LinearizedBeam,
        qm_scale: float = 1.0,
        seed: int = 42,
        m_mod: float = scw.MODULATION_INDEX,
        profile: str = "sheet_lattice",
        cfln: float = scw.CFL_NUMBER,
        n_macro_per_step: int | None = None,
        gather: str = "galerkin",
        shape: int = scw.PARTICLE_SHAPE,
        *,
        _from_cached: bool = False,
    ) -> dict[str, Any]:
        counts["part"] += 1
        nx_i = int(nx)
        nz_i = int(nz)
        n_macro = (
            int(n_macro_per_step)
            if n_macro_per_step is not None
            else (16 if nx_i == scw.COARSE_GRID[0] else 32)
        )
        if _from_cached:
            recorded.append(
                {
                    "func": "particle",
                    "args": {
                        "nx": nx_i,
                        "nz": nz_i,
                        "profile": str(profile),
                        "seed": int(seed),
                        "m_mod": float(m_mod),
                        "qm_scale": float(qm_scale),
                        "n_macro": n_macro,
                        "gather": str(gather),
                        "shape": int(shape),
                    },
                }
            )
        # Exercise beam_ref attributes the same way run_particle_beam_drift does
        assert hasattr(beam_ref, "flat_idx")
        assert hasattr(beam_ref, "dA_perp")
        assert hasattr(beam_ref, "s_coords")
        assert beam_ref.M == nz_i - 1

        z = beam_ref.s_coords.copy()
        seed_off = (float(seed) - 45.5) if profile == "sheet" else 0.0
        bq = float(
            params["beta_q"] * np.sqrt(float(qm_scale))
            + 1e-4 * seed_off
            + 0.1 * (float(m_mod) ** 2)
        )
        dz = scw.TUNNEL_LENGTH_M / nz_i
        amp_pred = float(
            (np.sinc(0.5 * params["beta_e"] * dz / np.pi) ** int(shape))
            * np.cos(0.5 * params["beta_q"] * dz)
        )
        amp = max(abs(float(m_mod)), scw.MODULATION_INDEX) * params["I0"] * amp_pred
        i1 = amp * np.cos(bq * z) * np.exp(-1j * params["beta_e"] * z)
        i1[0] = amp
        u_z_mean = float(params["u_dep"] * (1.0 - 1e-6 * seed_off))
        return {
            "i1_part": i1,
            "i2_part": 1e-5 * i1,
            "dc_leakage_part": 1e-16,
            "dirichlet_leak_part": 1e-16,
            "u_z_mean": u_z_mean,
            "rho_dc_mean": float(params["rho_dep"]),
            "u_z_profile": np.full(nz_i, u_z_mean, dtype=np.float64),
            "rho_dc_profile": np.full(nz_i, params["rho_dep"], dtype=np.float64),
            "vz_mean_coh": u_z_mean,
            "sigma_vz_mean": 1.0,
            "sigma_vz_over_u": 1e-6,
            "gamma_mean": float(params["gamma_dep"]),
            "gamma_pred_mean": float(params["gamma_dep"]),
            "gamma_diff_mean": 0.0,
            "vz_coh_profile": np.full(nz_i, u_z_mean, dtype=np.float64),
            "sigma_vz_profile": np.zeros(nz_i, dtype=np.float64),
            "gamma_profile": np.full(nz_i, params["gamma_dep"], dtype=np.float64),
            "gamma_pred_profile": np.full(nz_i, params["gamma_dep"], dtype=np.float64),
            "dt": 1e-12,
            "n_spp": 100,
            "n_settle": 100,
            "iz_hist_dec": np.zeros(
                (600, len(range(0, beam_ref.M, 8))), dtype=np.float64
            ),
        }

    return fake_run_lb, fake_run_particle, counts, recorded


def test_disk_cache_round_trip_and_cli_run(tmp_path, monkeypatch, capsys):
    """Verify round-trip pickle/rebuild through `CEM_SCW_CACHE_DIR` and `scw_points.py run`."""
    _clear_all_scw_lru_caches()
    monkeypatch.setenv("CEM_SCW_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CEM_SCW_CACHE_STRICT", raising=False)

    fake_lb, fake_part, counts, _ = _make_fake_simulators()
    monkeypatch.setattr(scw, "run_linearized_beam_drift", fake_lb)
    monkeypatch.setattr(
        scw,
        "run_particle_beam_drift",
        lambda *a, **kw: fake_part(*a, _from_cached=True, **kw),
    )

    # First call computes and writes to disk
    lb1 = scw.cached_linearized_beam_drift(10, 474, rho0_scale=1.0, m_mod=0.015)
    part1 = scw.cached_particle_drift(
        10, 474, profile="sheet_lattice", seed=42, m_mod=0.015
    )
    assert counts["lb"] == 1
    assert counts["part"] == 1

    # Confirm the raw beam object has an unpicklable closure, proving custom serialization was used
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        pickle.dumps(lb1["beam"], protocol=5)

    pkl_files = sorted(tmp_path.glob("*.pkl"))
    assert len(pkl_files) == 2

    # Clear LRU caches and re-query: must load from disk without invoking the simulator
    _clear_all_scw_lru_caches()
    lb2 = scw.cached_linearized_beam_drift(10, 474, rho0_scale=1.0, m_mod=0.015)
    part2 = scw.cached_particle_drift(
        10, 474, profile="sheet_lattice", seed=42, m_mod=0.015
    )
    assert counts["lb"] == 1
    assert counts["part"] == 1

    for k in (
        "z_coords",
        "i1_lb",
        "i2_lb",
        "dc_leakage_lb",
        "dirichlet_leak_lb",
        "dt",
        "n_spp",
    ):
        np.testing.assert_allclose(lb2[k], lb1[k], rtol=0.0, atol=0.0)
    assert isinstance(lb2["beam"], LinearizedBeam)
    np.testing.assert_array_equal(lb2["beam"].flat_idx_host, lb1["beam"].flat_idx_host)
    np.testing.assert_allclose(
        lb2["beam"].dA_perp, lb1["beam"].dA_perp, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        lb2["beam"].s_coords, lb1["beam"].s_coords, rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(lb2["beam"].keep_lines, lb1["beam"].keep_lines)
    np.testing.assert_allclose(
        lb2["beam"]._w_perp_eff, lb1["beam"]._w_perp_eff, rtol=0.0, atol=0.0
    )

    for k, v in part1.items():
        np.testing.assert_allclose(part2[k], v, rtol=0.0, atol=0.0)

    # Verify CLI `run` command hits the disk cache and prints `POINT i=<I> key=<key> secs=<t>`
    _clear_all_scw_lru_caches()
    capsys.readouterr()
    rc = scw_points.main(["run", "test_optional_diagnostics", "--index", "1"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    expected_key = scw.scw_cache_key(
        "particle",
        scw_points.list_points(["test_optional_diagnostics"])[1]["args"],
    )
    assert out.startswith(f"POINT i=1 key={expected_key} secs=")
    assert counts["lb"] == 1
    assert counts["part"] == 1


def test_strict_mode_miss_and_hit(tmp_path, monkeypatch):
    """Verify `CEM_SCW_CACHE_STRICT=1` raises `RuntimeError` on miss and succeeds on hit."""
    _clear_all_scw_lru_caches()
    monkeypatch.setenv("CEM_SCW_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CEM_SCW_CACHE_STRICT", "1")

    fake_lb, fake_part, counts, _ = _make_fake_simulators()
    monkeypatch.setattr(scw, "run_linearized_beam_drift", fake_lb)
    monkeypatch.setattr(
        scw,
        "run_particle_beam_drift",
        lambda *a, **kw: fake_part(*a, _from_cached=True, **kw),
    )

    # Miss on empty cache must raise RuntimeError naming function and arguments
    with pytest.raises(
        RuntimeError, match=r"cached_linearized_beam_drift\(10, 474, 1\.0, 0\.015\)"
    ):
        scw.cached_linearized_beam_drift(10, 474, rho0_scale=1.0, m_mod=0.015)
    assert counts["lb"] == 0

    # Populate cache with strict mode disabled
    monkeypatch.setenv("CEM_SCW_CACHE_STRICT", "0")
    _clear_all_scw_lru_caches()
    scw.cached_particle_drift(10, 474, profile="sheet", seed=44, m_mod=0.015)
    assert counts["lb"] == 1
    assert counts["part"] == 1

    # Re-enable strict mode after clearing LRU caches: hit must not raise or re-simulate
    monkeypatch.setenv("CEM_SCW_CACHE_STRICT", "1")
    _clear_all_scw_lru_caches()
    res = scw.cached_particle_drift(10, 474, profile="sheet", seed=44, m_mod=0.015)
    assert "i1_part" in res
    assert counts["lb"] == 1
    assert counts["part"] == 1

    # Uncached seed in strict mode must raise RuntimeError naming _cached_particle_drift_impl and seed 45
    with pytest.raises(RuntimeError, match=r"_cached_particle_drift_impl.*45"):
        scw.cached_particle_drift(10, 474, profile="sheet", seed=45, m_mod=0.015)
    assert counts["part"] == 1


def test_source_fingerprint_changes_on_file_edit(tmp_path, monkeypatch):
    """Verify `scw_source_fingerprint` and `scw_cache_key` change when a fingerprinted file changes."""
    f1 = tmp_path / "part_a.py"
    f2 = tmp_path / "part_b.py"
    f1.write_bytes(b"x = 1\n")
    f2.write_bytes(b"y = 2\n")

    monkeypatch.setattr(scw, "_SCW_FINGERPRINT_FILES", [f2, f1])
    fp1 = scw.scw_source_fingerprint()
    k1 = scw.scw_cache_key("cached_linearized_beam_drift", (10, 474, 1.0, 0.015))

    # Modifying one byte in f2 must change both the source fingerprint and cache key
    f2.write_bytes(b"y = 3\n")
    fp2 = scw.scw_source_fingerprint()
    k2 = scw.scw_cache_key("cached_linearized_beam_drift", (10, 474, 1.0, 0.015))
    assert fp1 != fp2
    assert k1 != k2

    # Argument sensitivity and eff_seed normalisation for sheet_lattice vs sheet
    k_lat_42 = scw.scw_cache_key(
        "particle", (10, 474, "sheet_lattice", 42, 0.015, 1.0, 16, "galerkin", 2)
    )
    k_lat_99 = scw.scw_cache_key(
        "particle", (10, 474, "sheet_lattice", 99, 0.015, 1.0, 16, "galerkin", 2)
    )
    k_sheet_42 = scw.scw_cache_key(
        "particle", (10, 474, "sheet", 42, 0.015, 1.0, 16, "galerkin", 2)
    )
    k_sheet_43 = scw.scw_cache_key(
        "particle", (10, 474, "sheet", 43, 0.015, 1.0, 16, "galerkin", 2)
    )
    assert k_lat_42 == k_lat_99
    assert k_sheet_42 != k_sheet_43
    assert k_lat_42 != k_sheet_42


def test_unset_env_performs_no_file_io(monkeypatch):
    """Verify `CEM_SCW_CACHE_DIR` absent performs zero disk/pickle I/O."""
    _clear_all_scw_lru_caches()
    monkeypatch.delenv("CEM_SCW_CACHE_DIR", raising=False)
    monkeypatch.delenv("CEM_SCW_CACHE_STRICT", raising=False)

    fake_lb, fake_part, counts, _ = _make_fake_simulators()
    monkeypatch.setattr(scw, "run_linearized_beam_drift", fake_lb)
    monkeypatch.setattr(
        scw,
        "run_particle_beam_drift",
        lambda *a, **kw: fake_part(*a, _from_cached=True, **kw),
    )

    # Pre-warm the dummy beam before blocking open / pickle.dump
    fake_lb(10, 474, rho0_scale=1.0, m_mod=scw.MODULATION_INDEX)
    counts["lb"] = 0

    def _forbidden_io(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "File or pickle I/O invoked when CEM_SCW_CACHE_DIR is unset"
        )

    monkeypatch.setattr(builtins, "open", _forbidden_io)
    monkeypatch.setattr(pickle, "dump", _forbidden_io)
    monkeypatch.setattr(pickle, "load", _forbidden_io)

    res_lb = scw.cached_linearized_beam_drift(
        10, 474, rho0_scale=1.0, m_mod=scw.MODULATION_INDEX
    )
    res_part = scw.cached_particle_drift(10, 474)
    assert "i1_lb" in res_lb
    assert "i1_part" in res_part
    assert counts["lb"] == 1
    assert counts["part"] == 1


def test_enumeration_completeness_and_strict_replay(tmp_path, monkeypatch):
    """Verify `scw_points.py list` matches the recorded call graph of the three slow tests and passes strict replay."""
    _clear_all_scw_lru_caches()
    monkeypatch.delenv("CEM_SCW_CACHE_DIR", raising=False)
    monkeypatch.delenv("CEM_SCW_CACHE_STRICT", raising=False)

    fake_lb, fake_part, _counts, recorded = _make_fake_simulators()
    monkeypatch.setattr(scw, "run_linearized_beam_drift", fake_lb)
    monkeypatch.setattr(
        scw,
        "run_particle_beam_drift",
        lambda *a, **kw: fake_part(*a, _from_cached=True, **kw),
    )
    monkeypatch.setattr(t087, "run_linearized_beam_drift", fake_lb)
    monkeypatch.setattr(
        t087,
        "run_particle_beam_drift",
        lambda *a, **kw: fake_part(*a, _from_cached=False, **kw),
    )

    # Stub the expensive Nelder-Mead / brentq post-processing helpers (none of which call the simulators)
    def _fast_fit_envelope(
        z_coords: np.ndarray,
        i1_phasor: np.ndarray,
        beta_q_guess: float,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> dict[str, Any]:
        bq = float(beta_q_guess) * (1.0 + 1e-9)
        kc = 700.0
        a_cap = complex(float(np.max(np.abs(i1_phasor))))
        return {
            "beta_q": bq,
            "k_c": kc,
            "A": a_cap,
            "B": 0.0j,
            "a_f": 0.5 * a_cap,
            "a_s": 0.5 * a_cap,
            "rss": 1e-24,
            "sigma_beta_q": 1e-12,
            "k_f": kc - bq,
            "k_s": kc + bq,
            "z0": 0.0,
            "as_over_af": 1.0,
            "b_over_a": 0.0j,
        }

    def _fast_fit_4wave(
        z_coords: np.ndarray,
        i1_phasor: np.ndarray,
        beta_q_guess: float,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> dict[str, Any]:
        bq = float(beta_q_guess) * (1.0 + 1e-9)
        return {
            "beta_q_2": bq,
            "beta_q_4": bq,
            "k_c_4": 700.0,
            "delta_beta_q": 0.0,
            "rel_delta_beta_q": 0.0,
            "bwd_over_fwd": 1e-12,
            "bwd_f_over_fwd_f": 1e-12,
            "bwd_s_over_fwd_s": 1e-12,
            "rss_4": 1e-24,
        }

    def _fast_fit_parasitic(
        z_coords: np.ndarray,
        i1_phasor: np.ndarray,
        nx: int,
        nz: int,
        params: dict[str, float],
        omega_p_scale: float = 1.0,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> dict[str, Any]:
        bq = float(params["beta_q"] * omega_p_scale) * (1.0 + 1e-9)
        return {
            "beta_q_2": bq,
            "beta_q_para": bq,
            "k_c_para": 700.0,
            "kp_minus": 6500.0,
            "kp_plus": 6400.0,
            "disp_resid_minus": 1e-15,
            "disp_resid_plus": 1e-15,
            "delta_beta_q": 0.0,
            "rel_delta_beta_q": 0.0,
            "ap_over_afwd": 1e-12,
            "rss_para": 1e-24,
        }

    def _fast_discrete_modal_prediction(
        code: str,
        nx: int,
        nz: int,
        params: dict[str, float],
        line_x: np.ndarray | None = None,
        line_w: np.ndarray | None = None,
        entrance_profile: np.ndarray | None = None,
        include_shear: bool = True,
        include_sz: bool = True,
        use_omega_f: bool = True,
        drop_highest_mode: bool = False,
        win_shift: float = 0.0,
        omega_p_scale: float = 1.0,
        return_details: bool = False,
        shape: int = scw.PARTICLE_SHAPE,
    ) -> float | dict[str, Any]:
        bq = float(params["beta_q"] * omega_p_scale) * (1.0 + 1e-9)
        if not return_details:
            return bq
        dz = scw.TUNNEL_LENGTH_M / int(nz)
        z_eval = (np.arange(int(nz), dtype=np.float64) + 0.5) * dz
        i1_model = (
            params["I1_0"]
            * np.cos(bq * z_eval)
            * np.exp(-1j * params["beta_e"] * z_eval)
        )
        return {
            "beta_q": bq,
            "evals": np.array([0.32, 0.07, 0.02, 0.01], dtype=np.float64),
            "bq_modes": np.array([bq], dtype=np.float64),
            "kc_modes": np.array([params["beta_e"]], dtype=np.float64),
            "c_modes": np.array([1.0], dtype=np.float64),
            "z_coords": z_eval,
            "i1_model": i1_model,
            "fit": _fast_fit_envelope(z_eval, i1_model, bq),
        }

    for mod in (scw, t087):
        monkeypatch.setattr(mod, "fit_beta_q_from_envelope", _fast_fit_envelope)
        monkeypatch.setattr(mod, "fit_four_wave_reflection", _fast_fit_4wave)
        monkeypatch.setattr(mod, "fit_lb_parasitic_waves", _fast_fit_parasitic)
        monkeypatch.setattr(
            mod, "discrete_modal_prediction", _fast_discrete_modal_prediction
        )

    test_funcs = [
        (
            "test_two_resolution_convergence_and_sheet_ensemble",
            t087.test_two_resolution_convergence_and_sheet_ensemble,
        ),
        (
            "test_axial_vs_transverse_refinement",
            t087.test_axial_vs_transverse_refinement,
        ),
        ("test_optional_diagnostics", t087.test_optional_diagnostics),
    ]

    def _point_set(
        pts: list[dict[str, Any]],
    ) -> set[tuple[str, tuple[tuple[str, Any], ...]]]:
        return {(p["func"], tuple(sorted(p["args"].items()))) for p in pts}

    # 1. Check per-test enumeration completeness against the recorded call graph
    for tname, tfun in test_funcs:
        _clear_all_scw_lru_caches()
        recorded.clear()
        try:
            tfun()
        except AssertionError:
            pass
        rec_set = _point_set(recorded)
        list_pts = scw_points.list_points([tname])
        list_set = _point_set(list_pts)
        print(
            f"\nENUM_CHECK test={tname} recorded_size={len(rec_set)} listed_size={len(list_set)}",
            flush=True,
        )
        assert len(list_pts) == len(list_set), (
            f"Duplicate points in list_points([{tname!r}])"
        )
        assert rec_set == list_set, (
            f"Point mismatch for {tname}: "
            f"missing={rec_set - list_set}, extra={list_set - rec_set}"
        )

    # 2. Check combined CLI `list` output and dependency ordering across all three tests
    all_names = [tname for tname, _ in test_funcs]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = scw_points.main(["list", *all_names])
    assert rc == 0
    cli_pts = json.loads(buf.getvalue())

    _clear_all_scw_lru_caches()
    recorded.clear()
    for _, tfun in test_funcs:
        try:
            tfun()
        except AssertionError:
            pass
    rec_all_set = _point_set(recorded)
    cli_all_set = _point_set(cli_pts)
    print(
        f"ENUM_CHECK test=ALL recorded_size={len(rec_all_set)} listed_size={len(cli_all_set)}",
        flush=True,
    )
    assert len(cli_pts) == len(cli_all_set)
    assert rec_all_set == cli_all_set

    # Verify dependency ordering: every particle point's LB (nx, nz, 1.0, MODULATION_INDEX) precedes it
    seen_lb: set[tuple[int, int, float, float]] = set()
    for pt in cli_pts:
        args = pt["args"]
        if pt["func"] == "cached_linearized_beam_drift":
            seen_lb.add(
                (
                    int(args["nx"]),
                    int(args["nz"]),
                    float(args["rho0_scale"]),
                    float(args["m_mod"]),
                )
            )
        else:
            dep = (int(args["nx"]), int(args["nz"]), 1.0, float(scw.MODULATION_INDEX))
            assert dep in seen_lb, (
                f"Particle point {pt} appears before LB dependency {dep}"
            )

    # 3. Populate cache strictly via `run_point` for each listed point, then replay all 3 tests under CEM_SCW_CACHE_STRICT=1
    monkeypatch.setenv("CEM_SCW_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CEM_SCW_CACHE_STRICT", "0")
    _clear_all_scw_lru_caches()
    for idx in range(len(cli_pts)):
        scw_points.run_point(all_names, idx)

    pkl_files = list(tmp_path.glob("*.pkl"))
    assert len(pkl_files) == len(cli_pts)

    monkeypatch.setenv("CEM_SCW_CACHE_STRICT", "1")
    _clear_all_scw_lru_caches()
    recorded.clear()
    for _, tfun in test_funcs:
        try:
            tfun()
        except AssertionError:
            pass
    # Zero cached-wrapper simulator calls during strict replay
    assert len(recorded) == 0

    # 4. Mutation check: deleting one cache file must cause strict replay to fail with RuntimeError
    pkl_files[-1].unlink()
    _clear_all_scw_lru_caches()
    with pytest.raises(RuntimeError, match="SCW disk cache miss"):
        for _, tfun in test_funcs:
            try:
                tfun()
            except AssertionError:
                pass
