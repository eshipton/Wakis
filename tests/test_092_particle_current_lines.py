# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Fast tests for `_particle_current_lines` and `run_particle_beam_drift(..., meas_lines=...)`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples._space_charge_waves_common as scw
from wakis import GridFIT3D, SolverFIT3D


@pytest.fixture(scope="module")
def small_beam_and_solver():
    """Construct a small `(nx=10, ny=1, nz=8)` reference `LinearizedBeam` and its `SolverFIT3D`."""
    beam_ref = scw._build_reference_linearized_beam(10, 8)
    return beam_ref, beam_ref._solver


def test_particle_current_lines_lb_keep_identical(small_beam_and_solver):
    """Verify `mode='lb_keep'` returns arrays identical (`np.array_equal`) to `beam_ref.flat_idx_2d` and `beam_ref.dA_perp`."""
    beam_ref, solver = small_beam_and_solver
    flat_keep, dA_keep = scw._particle_current_lines(beam_ref, solver, "lb_keep")

    assert flat_keep.shape == (beam_ref.M, beam_ref.N_lines)
    assert dA_keep.shape == (beam_ref.M, beam_ref.N_lines)
    assert np.array_equal(flat_keep, beam_ref.flat_idx_2d)
    assert np.array_equal(dA_keep, beam_ref.dA_perp)


def test_particle_current_lines_all_superset_and_ieps(small_beam_and_solver):
    """Verify `mode='all'` is a strict superset of `lb_keep`, has no `ieps == 0` edges, and matches `dA_perp` on shared lines."""
    beam_ref, solver = small_beam_and_solver
    flat_keep, dA_keep = scw._particle_current_lines(beam_ref, solver, "lb_keep")
    flat_all, dA_all = scw._particle_current_lines(beam_ref, solver, "all")

    assert flat_all.shape[0] == beam_ref.M
    assert dA_all.shape == flat_all.shape
    assert flat_all.shape[1] > flat_keep.shape[1]

    keep_set = set(flat_keep.ravel().tolist())
    all_set = set(flat_all.ravel().tolist())
    assert keep_set < all_set

    ieps_host = np.asarray(
        solver.ieps.array.get() if solver.use_gpu else solver.ieps.array
    )
    assert np.all(ieps_host[flat_all] != 0.0)

    # Identify shared transverse lines by matching first-row flat indices
    shared_cols = [
        int(np.flatnonzero(flat_all[0, :] == col_idx)[0]) for col_idx in flat_keep[0, :]
    ]
    assert np.array_equal(flat_all[:, shared_cols], flat_keep)
    assert np.array_equal(dA_all[:, shared_cols], dA_keep)

    # Verify that zeroing ieps on one transverse line outside lb_keep excludes that line
    extra_cols = [c for c in range(flat_all.shape[1]) if c not in shared_cols]
    assert len(extra_cols) > 0
    pec_line_edges = flat_all[:, extra_cols[0]]
    orig_ieps = ieps_host[pec_line_edges].copy()
    try:
        if solver.use_gpu:
            solver.ieps.array[pec_line_edges] = 0.0
        else:
            solver.ieps.array[pec_line_edges] = 0.0
        flat_after, _ = scw._particle_current_lines(beam_ref, solver, "all")
        ieps_after = np.asarray(
            solver.ieps.array.get() if solver.use_gpu else solver.ieps.array
        )
        assert flat_after.shape == (beam_ref.M, flat_all.shape[1] - 1)
        assert np.all(ieps_after[flat_after] != 0.0)
        assert set(pec_line_edges.tolist()).isdisjoint(set(flat_after.ravel().tolist()))
    finally:
        solver.ieps.array[pec_line_edges] = orig_ieps


def test_particle_current_lines_synthetic_j_and_mutation(small_beam_and_solver):
    """Verify synthetic J gives equal sums when supported on `lb_keep` and differs when an outer line is active."""
    beam_ref, solver = small_beam_and_solver
    flat_keep, dA_keep = scw._particle_current_lines(beam_ref, solver, "lb_keep")
    flat_all, dA_all = scw._particle_current_lines(beam_ref, solver, "all")

    j_vec = np.zeros(3 * solver.N, dtype=np.float64)
    k_prof = np.arange(1, beam_ref.M + 1, dtype=np.float64)[:, None]
    l_prof = np.arange(1, flat_keep.shape[1] + 1, dtype=np.float64)[None, :]
    j_vec[flat_keep] = 2.5e3 * k_prof * l_prof

    sum_keep = np.sum(j_vec[flat_keep] * dA_keep, axis=1)
    sum_all = np.sum(j_vec[flat_all] * dA_all, axis=1)
    assert np.array_equal(sum_all, sum_keep)

    # Add current on one transverse line outside lb_keep (mutation check: lb_keep misses it)
    keep_first_row = set(flat_keep[0, :].tolist())
    outside_cols = [
        c for c in range(flat_all.shape[1]) if int(flat_all[0, c]) not in keep_first_row
    ]
    assert len(outside_cols) >= 2
    out_col = outside_cols[0]
    j_extra = -4.0e2 * np.arange(1, beam_ref.M + 1, dtype=np.float64)
    j_vec[flat_all[:, out_col]] = j_extra

    sum_keep_after = np.sum(j_vec[flat_keep] * dA_keep, axis=1)
    sum_all_after = np.sum(j_vec[flat_all] * dA_all, axis=1)

    assert np.array_equal(sum_keep_after, sum_keep)
    assert not np.allclose(sum_all_after, sum_keep_after)
    expected_delta = j_extra * dA_all[:, out_col]
    np.testing.assert_allclose(
        sum_all_after - sum_keep_after, expected_delta, rtol=0.0, atol=1e-15
    )


def test_run_particle_beam_drift_invalid_meas_lines_raises(
    small_beam_and_solver, monkeypatch
):
    """Verify `run_particle_beam_drift(..., meas_lines=...)` validates `meas_lines` before stepping and integrates all vs `lb_keep` lines."""
    beam_ref, solver = small_beam_and_solver

    with pytest.raises(ValueError, match="meas_lines"):
        scw._particle_current_lines(beam_ref, solver, "bogus")

    with monkeypatch.context() as m:

        def _forbidden(*args, **kwargs):
            raise AssertionError(
                "GridFIT3D or SolverFIT3D stepping should not be called"
            )

        m.setattr(scw, "compute_analytic_parameters", _forbidden)
        m.setattr(scw, "GridFIT3D", _forbidden)
        m.setattr(scw, "SolverFIT3D", _forbidden)
        m.setattr(SolverFIT3D, "__init__", _forbidden)
        m.setattr(GridFIT3D, "__init__", _forbidden)

        with pytest.raises(ValueError, match="meas_lines"):
            scw.run_particle_beam_drift(10, 8, beam_ref=beam_ref, meas_lines="bogus")

    # Minimal 6-step run (n_settle=0, n_spp=1) comparing default meas_lines='all' vs 'lb_keep'
    dt, _ = scw._integer_period_dt(solver.grid)
    with monkeypatch.context() as m:
        m.setattr(
            beam_ref,
            "s_coords",
            np.full_like(beam_ref.s_coords, 0.5 * scw.TUNNEL_LENGTH_M),
        )
        m.setattr(scw, "_integer_period_dt", lambda grid, cfln=scw.CFL_NUMBER: (dt, 1))
        m.setattr(scw, "settle_steps", lambda *a, **kw: 0)
        # Bitwise equality below needs a deterministic summation order: GPU current deposition
        # is not order-deterministic (1 of 3 A100 runs differed in the last bit), so run on CPU.
        m.setattr(scw, "USE_GPU", False)
        res_all = scw.run_particle_beam_drift(10, 8, beam_ref=beam_ref)
        res_keep = scw.run_particle_beam_drift(
            10, 8, beam_ref=beam_ref, meas_lines="lb_keep"
        )

    assert res_all["i1_part"].shape == (beam_ref.M,)
    assert res_keep["i1_part"].shape == (beam_ref.M,)
    # On nx=10 with 16 shape-2 sheet_lattice lines, 5/512 of the current lands on
    # lines x = +-1.2 mm outside lb_keep, so step-1 I_z(lb_keep) / I_z(all) == 507/512
    iz0_all = float(res_all["iz_hist_dec"][0, 0])
    iz0_keep = float(res_keep["iz_hist_dec"][0, 0])
    assert abs(iz0_all) > abs(iz0_keep) > 0.0
    np.testing.assert_allclose(iz0_keep / iz0_all, 507.0 / 512.0, rtol=1e-13, atol=0.0)
    # Unchanged DC/moment diagnostics between meas_lines='all' and 'lb_keep'
    for key in ("u_z_mean", "rho_dc_mean", "vz_mean_coh", "gamma_mean"):
        assert res_all[key] == res_keep[key]
