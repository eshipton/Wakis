# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Per-line decomposition of the DC-depression gap at m = 0.

Runs the planar drift-tunnel `ParticleSolver` + `BeamInjector` setup from
``examples/_space_charge_waves_common.py`` (`run_particle_beam_drift`) and
``tests/test_087_particles_vs_linearized_beam.py``
(`test_two_resolution_convergence_and_sheet_ensemble`) with ``profile="sheet_lattice"``,
``qm_scale=1.0``, and ``m_mod=0.0`` (pure DC; `BeamInjector` and
`ParticleSolver` accept ``m_mod=0.0`` directly, giving constant DC current
``I(t) = I0``).

Over the settled interior cells ``z in [0.2 * Lz, 0.8 * Lz]`` and the first RF
period of the measurement window (``n_spp`` steps starting at ``n_settle``),
records per injected line ``l in {0, ..., n_macro - 1}``:

1. ``N_l`` (time-averaged physical charge magnitude [C] in the interior window),
   ``mean_x_l`` [m], and ``rms_dx_l = rms(x - x_{l,inj})`` [m];
2. ``mean_gamma_l`` [-], ``mean_vz_l`` [m/s], and
   ``mean_u_perp2_c2_l = <(u_x^2 + u_y^2) / c^2>_l`` [-];
3. ``phi_sim_l`` [V]: simulated transverse potential depression magnitude
   ``-|phi(x, z)|`` sampled at each particle's ``x`` with
   ``transverse_weight_matrix`` (same potential gather as the library's
   ``gamma_pred``), charge-weighted mean over the line's interior particles;
4. Model per-line ``phi_l`` [V], ``gamma_l`` [-], ``u_l`` [m/s] from
   ``compute_discrete_dc_depression(..., weighting="charge", return_details=True)``.

Decomposes ``delta_meas - delta_charge_model`` into:

- ``(A)`` Simulated DC potential at the lines vs the model's 1D discrete Poisson
  potential ``phi_l`` (replacing model ``phi_l`` by ``phi_sim_l`` in the model's
  ``gamma(phi)`` and ``u(gamma)`` and recomputing the harmonic mean; reported both
  directly from ``x_{l,inj}`` as ``A_direct`` and net of line shift ``(C)`` as
  ``A`` so ``A + C = A_direct``);
- ``(B)`` Transverse kinetic energy (``u_x, u_y`` nonzero) reducing ``v_z`` below
  ``c * sqrt(1 - 1 / gamma^2)`` (``v_z`` from ``mean_gamma_l`` with vs without
  ``mean_u_perp2_c2_l``, plus within-line Jensen residual ``jensen_vz``);
- ``(C)`` Line movement in ``x`` (model 1D discrete Poisson potential re-evaluated
  at the measured ``mean_x_l``, plus fixed-node-potential gather ``C_gather``);
- ``(D)`` Measured ``u_z_mean`` line weighting ``N_l`` vs the model's steady-state
  continuity line weighting ``rho_l / u_l``;
- Plus ``energy_residual`` from ``mean_gamma_l - gamma_pred(phi_sim_l)`` so the
  full telescoping chain closes to ``total = delta_meas - delta_charge_model`` to
  machine precision while ``closure_abcd = (A + B + C + D) - total`` reports the
  non-ABCD remainder ``-(energy_residual + jensen_vz)``.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import epsilon_0, m_e

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "examples"))

import _space_charge_waves_common as s

from wakis import GridFIT3D, SolverFIT3D
from wakis.particles import BeamInjector, ParticleSolver, Species

SELF_CHECK_TOL = 1.0e-15
DEFAULT_SEED = int(
    inspect.signature(s.run_particle_beam_drift).parameters["seed"].default
)


class _LineTrackingBeamInjector(BeamInjector):
    """`BeamInjector` subclass that records each particle's line index from ``x`` at injection."""

    def __init__(self, *args: Any, line_x: np.ndarray, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._line_x = np.asarray(line_x, dtype=np.float64).copy()
        self.line_id_by_pid = np.empty(0, dtype=np.int64)
        self.line_id_by_order = np.empty(0, dtype=np.int64)
        self.max_inj_x_err_m = 0.0

    def inject(
        self,
        t: float,
        dt: float,
        step_index: int | None = None,
        *,
        owning: bool = True,
    ) -> int:
        """Inject macroparticles and bin their injection ``x`` to the nearest ``line_x``."""
        sp = self.species
        n_before = int(sp.n_particles)
        pid_before = int(getattr(sp, "_next_id", n_before))
        n_inj = super().inject(t, dt, step_index=step_index, owning=owning)
        if n_inj > 0:
            x_raw = sp.x.get() if hasattr(sp.x, "get") else sp.x
            x_new = np.asarray(x_raw[n_before : n_before + n_inj], dtype=np.float64)
            line_idx = np.argmin(
                np.abs(x_new[:, None] - self._line_x[None, :]), axis=1
            ).astype(np.int64)
            err = float(np.max(np.abs(x_new - self._line_x[line_idx])))
            self.max_inj_x_err_m = max(self.max_inj_x_err_m, err)

            pid_end = pid_before + n_inj
            if pid_end > len(self.line_id_by_pid):
                new_cap = max(pid_end, 2 * max(len(self.line_id_by_pid), n_inj))
                grown = np.empty(new_cap, dtype=np.int64)
                grown[: len(self.line_id_by_pid)] = self.line_id_by_pid
                self.line_id_by_pid = grown
            if getattr(sp, "track_ids", False) and sp.id is not None:
                id_raw = sp.id.get() if hasattr(sp.id, "get") else sp.id
                pids = np.asarray(id_raw[n_before : n_before + n_inj], dtype=np.int64)
                self.line_id_by_pid[pids] = line_idx
            else:
                self.line_id_by_pid[pid_before:pid_end] = line_idx
            self.line_id_by_order = np.concatenate([self.line_id_by_order, line_idx])
        return n_inj


def _gamma_from_phi(
    phi_depression_v: np.ndarray,
    gamma0: float,
    mass_eff: float,
) -> np.ndarray:
    """Evaluate relativistic Lorentz factor ``gamma = gamma0 - e * phi / (m_eff * c^2)`` [-].

    Parameters
    ----------
    phi_depression_v : np.ndarray
        Positive potential depression magnitude ``|phi_l| = -phi(x_l)`` [V].
    gamma0 : float
        Injection Lorentz factor [-].
    mass_eff : float
        Effective species mass ``m_e / qm_scale`` [kg].

    Returns
    -------
    np.ndarray
        Per-line Lorentz factor [-].
    """
    return gamma0 - (e_charge / (mass_eff * (c_light**2))) * np.asarray(
        phi_depression_v, dtype=np.float64
    )


def _vz_from_gamma(
    gamma_arr: np.ndarray,
    u_perp2_c2_arr: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Evaluate axial velocity ``c * sqrt(1 - (1 + (u_x^2 + u_y^2)/c^2) / gamma^2)`` [m/s].

    Parameters
    ----------
    gamma_arr : np.ndarray
        Per-line Lorentz factor [-].
    u_perp2_c2_arr : np.ndarray or float, optional
        Per-line dimensionless transverse proper velocity squared
        ``(u_x^2 + u_y^2) / c^2`` [-]. Default is ``0.0``.

    Returns
    -------
    np.ndarray
        Per-line axial velocity [m/s].
    """
    g2 = np.asarray(gamma_arr, dtype=np.float64) ** 2
    up2 = np.asarray(u_perp2_c2_arr, dtype=np.float64)
    return c_light * np.sqrt(np.maximum(1.0 - (1.0 + up2) / g2, 0.0))


def _delta_weighted(
    v_arr: np.ndarray,
    w_arr: np.ndarray,
    u0: float,
) -> float:
    """Return weighted relative velocity depression ``1 - sum(w_l * v_l) / (u0 * sum(w_l))`` [-]."""
    v = np.asarray(v_arr, dtype=np.float64)
    w = np.asarray(w_arr, dtype=np.float64)
    return float(1.0 - float(np.sum(w * v) / (float(u0) * np.sum(w))))


def _delta_harmonic(
    v_arr: np.ndarray,
    rho_l: np.ndarray,
    u0: float,
) -> float:
    """Return charge-weighted harmonic-mean depression ``1 - sum(w_l * v_l) / (u0 * sum(w_l))`` with ``w_l = rho_l / v_l`` [-]."""
    v = np.asarray(v_arr, dtype=np.float64)
    w_l = np.asarray(rho_l, dtype=np.float64) / v
    return _delta_weighted(v, w_l, u0)


def _solve_model_node_potential(
    nx: int,
    params: dict[str, float],
    model_det: dict[str, Any],
) -> np.ndarray:
    """Return the converged 1D discrete Poisson node potential magnitude ``phi_node`` [V] of ``model_det``."""
    a = s.TUNNEL_HALF_WIDTH_M
    b = s.BEAM_HALF_THICKNESS_M
    dx = 2.0 * a / int(nx)
    n_int = int(nx) - 1
    lap = (
        np.diag(np.full(n_int, -2.0))
        + np.diag(np.ones(n_int - 1), 1)
        + np.diag(np.ones(n_int - 1), -1)
    ) / (dx**2)
    u0 = float(params["u0"])
    rho0 = abs(float(params["rho0"]))
    rho_l = np.asarray(model_det["rho_l"], dtype=np.float64)
    u_l = np.asarray(model_det["u_l"], dtype=np.float64)
    w_mat = np.asarray(model_det["w_mat"], dtype=np.float64)
    rho_dep_l = rho_l * (u0 / u_l)
    w_node_dep = np.sum(rho_dep_l[:, None] * w_mat, axis=0) * (2.0 * b / dx)
    phi_node = np.zeros(int(nx) + 1, dtype=np.float64)
    phi_node[1:-1] = np.linalg.solve(lap, -rho0 * w_node_dep[1:-1] / epsilon_0)
    return phi_node


def decompose_dc_depression(
    nx: int,
    params: dict[str, float],
    model_det: dict[str, Any],
    N_l: np.ndarray,
    mean_x_l: np.ndarray,
    rms_dx_l: np.ndarray,
    mean_gamma_l: np.ndarray,
    mean_vz_l: np.ndarray,
    mean_u_perp2_c2_l: np.ndarray,
    phi_sim_l: np.ndarray,
    shape: int = s.PARTICLE_SHAPE,
) -> dict[str, Any]:
    """Decompose ``delta_meas - delta_charge_model`` into contributions ``(A)-(D)`` from per-line arrays.

    Parameters
    ----------
    nx : int
        Transverse cell count across ``[-a, a]`` [-].
    params : dict of str to float
        Output of ``compute_analytic_parameters``.
    model_det : dict of str to Any
        Output of ``compute_discrete_dc_depression(nx, params, weighting="charge", return_details=True)``.
    N_l : np.ndarray
        Measured per-line charge magnitude [C] in the interior window.
    mean_x_l : np.ndarray
        Measured per-line charge-weighted mean transverse position ``<x>_l`` [m].
    rms_dx_l : np.ndarray
        Measured per-line transverse RMS displacement ``rms(x - x_{l,inj})`` [m].
    mean_gamma_l : np.ndarray
        Measured per-line charge-weighted mean Lorentz factor ``<gamma>_l`` [-].
    mean_vz_l : np.ndarray
        Measured per-line charge-weighted mean axial velocity ``<v_z>_l`` [m/s].
    mean_u_perp2_c2_l : np.ndarray
        Measured per-line mean dimensionless transverse proper velocity squared
        ``<(u_x^2 + u_y^2) / c^2>_l`` [-].
    phi_sim_l : np.ndarray
        Measured per-line mean simulated potential depression magnitude
        ``<-phi_sim(x_p, z_p)>_l`` [V] (same sign convention as ``model_det["phi_l"]``).
    shape : int, optional
        B-spline particle shape order ``p`` [-]. Default is ``PARTICLE_SHAPE``.

    Returns
    -------
    dict of str to Any
        Dictionary containing the per-line arrays, intermediate stage depressions,
        contributions ``A``, ``A_direct``, ``B``, ``B_total``, ``C``, ``C_gather``,
        ``D``, ``D_vs_w_model``, ``energy_residual``, ``jensen_vz``, ``total``,
        ``sum_abcd``, ``closure_abcd``, ``sum_parts``, and ``closure``.
    """
    u0 = float(params["u0"])
    gamma0 = float(params["gamma0"])
    qm_scale = float(params.get("qm_scale", 1.0))
    mass_eff = m_e / qm_scale

    x_inj_l = np.asarray(model_det["x_sheet"], dtype=np.float64)
    rho_l = np.asarray(model_det["rho_l"], dtype=np.float64)
    phi_model_l = np.asarray(model_det["phi_l"], dtype=np.float64)
    gamma_model_l = np.asarray(model_det["gamma_l"], dtype=np.float64)
    u_model_l = np.asarray(model_det["u_l"], dtype=np.float64)
    delta_charge_model = float(model_det["delta_charge"])

    N_arr = np.asarray(N_l, dtype=np.float64)
    xbar_arr = np.asarray(mean_x_l, dtype=np.float64)
    rms_dx_arr = np.asarray(rms_dx_l, dtype=np.float64)
    gamma_meas_arr = np.asarray(mean_gamma_l, dtype=np.float64)
    vz_meas_arr = np.asarray(mean_vz_l, dtype=np.float64)
    uperp2_arr = np.asarray(mean_u_perp2_c2_l, dtype=np.float64)
    phi_sim_arr = np.asarray(phi_sim_l, dtype=np.float64)

    # Stage 0: Model at x_{l,inj}
    delta_0 = delta_charge_model

    # Stage C: Model potential re-evaluated at measured mean x_l
    det_xbar = s.compute_discrete_dc_depression(
        int(nx),
        params,
        line_x=xbar_arr,
        line_w=rho_l,
        weighting="charge",
        return_details=True,
        shape=int(shape),
    )
    phi_model_xbar_l = np.asarray(det_xbar["phi_l"], dtype=np.float64)
    gamma_model_xbar_l = np.asarray(det_xbar["gamma_l"], dtype=np.float64)
    u_model_xbar_l = np.asarray(det_xbar["u_l"], dtype=np.float64)
    delta_c = float(det_xbar["delta_charge"])

    # Fixed-node-potential gather variant of (C)
    phi_node_model = _solve_model_node_potential(int(nx), params, model_det)
    w_mat_xbar = np.asarray(
        s.transverse_weight_matrix(xbar_arr, int(nx), int(shape)),
        dtype=np.float64,
    )
    phi_model_gather_xbar_l = np.abs(w_mat_xbar @ phi_node_model)
    gamma_model_gather_xbar_l = _gamma_from_phi(
        phi_model_gather_xbar_l, gamma0, mass_eff
    )
    u_model_gather_xbar_l = _vz_from_gamma(gamma_model_gather_xbar_l, 0.0)
    delta_c_gather = _delta_harmonic(u_model_gather_xbar_l, rho_l, u0)

    # Stage A: Replace model phi_l by phi_sim,l in model's gamma/u and recompute harmonic mean
    gamma_sim_l = _gamma_from_phi(phi_sim_arr, gamma0, mass_eff)
    u_sim_l = _vz_from_gamma(gamma_sim_l, 0.0)
    delta_a = _delta_harmonic(u_sim_l, rho_l, u0)

    # Stage gamma (no transverse energy): v_z from measured mean_gamma_l with u_perp^2 = 0
    vz_gamma_noperp_l = _vz_from_gamma(gamma_meas_arr, 0.0)
    delta_gamma_noperp = _delta_harmonic(vz_gamma_noperp_l, rho_l, u0)

    # Stage B (with transverse energy): v_z from measured mean_gamma_l and mean_u_perp2_c2_l
    vz_gamma_withperp_l = _vz_from_gamma(gamma_meas_arr, uperp2_arr)
    delta_gamma_withperp = _delta_harmonic(vz_gamma_withperp_l, rho_l, u0)

    # Stage vz_hm: Measured per-line mean_vz_l with model harmonic-mean weighting rho_l / v_{z,l}
    delta_vz_hm = _delta_harmonic(vz_meas_arr, rho_l, u0)
    w_model_l = rho_l / u_model_l
    delta_vz_w_model = _delta_weighted(vz_meas_arr, w_model_l, u0)

    # Stage D (measured): Measured per-line mean_vz_l with measured charge weights N_l
    delta_meas = _delta_weighted(vz_meas_arr, N_arr, u0)
    u_z_mean_from_lines = float(np.sum(N_arr * vz_meas_arr) / np.sum(N_arr))

    # Contributions
    part_c = float(delta_c - delta_0)
    part_c_gather = float(delta_c_gather - delta_0)
    part_a_direct = float(delta_a - delta_0)
    part_a_net = float(delta_a - delta_c)
    part_energy = float(delta_gamma_noperp - delta_a)
    part_b = float(delta_gamma_withperp - delta_gamma_noperp)
    part_jensen = float(delta_vz_hm - delta_gamma_withperp)
    part_b_total = float(delta_vz_hm - delta_gamma_noperp)
    part_d = float(delta_meas - delta_vz_hm)
    part_d_vs_w_model = float(delta_meas - delta_vz_w_model)

    total = float(delta_meas - delta_0)
    sum_abcd = float(part_a_net + part_b + part_c + part_d)
    closure_abcd = float(sum_abcd - total)
    sum_abcd_direct = float(part_a_direct + part_b + part_c + part_d)
    closure_abcd_direct = float(sum_abcd_direct - total)
    sum_parts = float(part_a_net + part_b + part_c + part_d + part_energy + part_jensen)
    closure = float(sum_parts - total)

    w_meas_norm = N_arr / np.sum(N_arr)
    w_model_norm = w_model_l / np.sum(w_model_l)
    w_hm_meas_l = rho_l / vz_meas_arr
    w_hm_meas_norm = w_hm_meas_l / np.sum(w_hm_meas_l)

    gamma_mean_meas = float(np.sum(w_meas_norm * gamma_meas_arr))
    gamma_pred_mean_meas = float(np.sum(w_meas_norm * gamma_sim_l))

    return {
        "u0": u0,
        "gamma0": gamma0,
        "u_z_mean_from_lines": u_z_mean_from_lines,
        "delta_charge_model": delta_charge_model,
        "delta_flux_model": float(model_det["delta_flux"]),
        "delta_density_model": float(model_det["delta_density"]),
        "delta_meas": delta_meas,
        "total": total,
        "A": part_a_net,
        "A_direct": part_a_direct,
        "B": part_b,
        "B_total": part_b_total,
        "C": part_c,
        "C_gather": part_c_gather,
        "D": part_d,
        "D_vs_w_model": part_d_vs_w_model,
        "energy_residual": part_energy,
        "jensen_vz": part_jensen,
        "sum_abcd": sum_abcd,
        "closure_abcd": closure_abcd,
        "abs_closure_abcd": abs(closure_abcd),
        "sum_abcd_direct": sum_abcd_direct,
        "closure_abcd_direct": closure_abcd_direct,
        "sum_parts": sum_parts,
        "closure": closure,
        "abs_closure": abs(closure),
        "gamma_mean": gamma_mean_meas,
        "gamma_pred_mean": gamma_pred_mean_meas,
        "gamma_diff_mean": float(gamma_mean_meas - gamma_pred_mean_meas),
        "stages": {
            "delta_0_model": delta_0,
            "delta_C_xbar": delta_c,
            "delta_C_gather": delta_c_gather,
            "delta_A_phi_sim": delta_a,
            "delta_gamma_noperp": delta_gamma_noperp,
            "delta_B_withperp": delta_gamma_withperp,
            "delta_vz_hm": delta_vz_hm,
            "delta_vz_w_model": delta_vz_w_model,
            "delta_D_meas": delta_meas,
        },
        "per_line": {
            "x_inj_m": x_inj_l.tolist(),
            "rho_l": rho_l.tolist(),
            "N_l_C": N_arr.tolist(),
            "w_meas_norm": w_meas_norm.tolist(),
            "w_model_norm": w_model_norm.tolist(),
            "w_hm_meas_norm": w_hm_meas_norm.tolist(),
            "mean_x_m": xbar_arr.tolist(),
            "mean_dx_m": (xbar_arr - x_inj_l).tolist(),
            "rms_dx_m": rms_dx_arr.tolist(),
            "mean_gamma": gamma_meas_arr.tolist(),
            "gamma_pred_sim": gamma_sim_l.tolist(),
            "gamma_diff": (gamma_meas_arr - gamma_sim_l).tolist(),
            "mean_vz_m_s": vz_meas_arr.tolist(),
            "mean_u_perp2_c2": uperp2_arr.tolist(),
            "phi_sim_V": phi_sim_arr.tolist(),
            "phi_model_V": phi_model_l.tolist(),
            "phi_diff_V": (phi_sim_arr - phi_model_l).tolist(),
            "phi_model_xbar_V": phi_model_xbar_l.tolist(),
            "phi_model_gather_xbar_V": phi_model_gather_xbar_l.tolist(),
            "gamma_model": gamma_model_l.tolist(),
            "gamma_model_xbar": gamma_model_xbar_l.tolist(),
            "u_model_m_s": u_model_l.tolist(),
            "u_model_xbar_m_s": u_model_xbar_l.tolist(),
            "u_sim_m_s": u_sim_l.tolist(),
            "vz_gamma_noperp_m_s": vz_gamma_noperp_l.tolist(),
            "vz_gamma_withperp_m_s": vz_gamma_withperp_l.tolist(),
        },
    }


def assert_self_check_zero(decomp: dict[str, Any], tol: float = SELF_CHECK_TOL) -> None:
    """Assert that every part and the total in ``decomp`` are zero within ``tol``."""
    keys_to_check = (
        "A",
        "A_direct",
        "B",
        "B_total",
        "C",
        "C_gather",
        "D",
        "D_vs_w_model",
        "energy_residual",
        "jensen_vz",
        "total",
        "sum_abcd",
        "closure_abcd",
        "sum_parts",
        "closure",
    )
    for k in keys_to_check:
        val = float(decomp[k])
        if abs(val) > float(tol):
            raise AssertionError(
                f"Self-check failed: |{k}| = {abs(val):.6e} > tol = {float(tol):.6e}"
            )


def run_self_check(
    tol: float = SELF_CHECK_TOL,
    run_mutation_check: bool = True,
) -> dict[str, Any]:
    """Run synthetic model-identity self-check and single-channel mutation checks.

    Feeds ``decompose_dc_depression`` synthetic per-line arrays equal to the
    model's own values (``phi_sim = phi_model``, ``x = line_x``, no transverse
    energy, ``N_l = rho_l / u_model_l``) on both coarse (``nx=10``) and fine
    (``nx=20``) grids and asserts every part and the total are zero within
    ``1e-15``. Also verifies that perturbing each channel ``(A)-(D)`` causes
    :func:`assert_self_check_zero` to fail and is isolated to that channel.
    """
    params = s.compute_analytic_parameters(qm_scale=1.0)
    results: dict[str, Any] = {}

    for label, (nx, _, _) in (("coarse", s.COARSE_GRID), ("fine", s.FINE_GRID)):
        det = s.compute_discrete_dc_depression(
            nx,
            params,
            weighting="charge",
            return_details=True,
            shape=s.PARTICLE_SHAPE,
        )
        x_inj = np.asarray(det["x_sheet"], dtype=np.float64)
        rho_l = np.asarray(det["rho_l"], dtype=np.float64)
        phi_model = np.asarray(det["phi_l"], dtype=np.float64)
        gamma_model = np.asarray(det["gamma_l"], dtype=np.float64)
        u_model = np.asarray(det["u_l"], dtype=np.float64)
        N_model = rho_l / u_model
        zeros = np.zeros_like(x_inj)

        dec = decompose_dc_depression(
            nx=nx,
            params=params,
            model_det=det,
            N_l=N_model,
            mean_x_l=x_inj,
            rms_dx_l=zeros,
            mean_gamma_l=gamma_model,
            mean_vz_l=u_model,
            mean_u_perp2_c2_l=zeros,
            phi_sim_l=phi_model,
            shape=s.PARTICLE_SHAPE,
        )
        assert_self_check_zero(dec, tol=tol)
        max_abs_part = max(
            abs(float(dec[k]))
            for k in (
                "A",
                "A_direct",
                "B",
                "B_total",
                "C",
                "C_gather",
                "D",
                "D_vs_w_model",
                "energy_residual",
                "jensen_vz",
                "total",
                "closure_abcd",
                "closure",
            )
        )

        mut_summary: dict[str, Any] = {}
        if run_mutation_check:
            mass_eff = m_e / float(params.get("qm_scale", 1.0))
            gamma0 = float(params["gamma0"])
            eps_mut = float(s.MODULATION_INDEX)

            # Mutation A: perturb phi_sim_l -> must fail zero check and isolate to A
            phi_mut_a = phi_model * (1.0 + eps_mut)
            gamma_mut_a = _gamma_from_phi(phi_mut_a, gamma0, mass_eff)
            u_mut_a = _vz_from_gamma(gamma_mut_a, 0.0)
            dec_a = decompose_dc_depression(
                nx=nx,
                params=params,
                model_det=det,
                N_l=rho_l / u_mut_a,
                mean_x_l=x_inj,
                rms_dx_l=zeros,
                mean_gamma_l=gamma_mut_a,
                mean_vz_l=u_mut_a,
                mean_u_perp2_c2_l=zeros,
                phi_sim_l=phi_mut_a,
                shape=s.PARTICLE_SHAPE,
            )
            failed_a = False
            try:
                assert_self_check_zero(dec_a, tol=tol)
            except AssertionError:
                failed_a = True
            if not failed_a or abs(dec_a["A"] - dec_a["total"]) > tol:
                raise RuntimeError("Mutation check A failed to isolate channel A.")
            mut_summary["mut_A"] = {
                "failed_as_expected": failed_a,
                "A_minus_total": float(dec_a["A"] - dec_a["total"]),
                "abs_closure_abcd": float(dec_a["abs_closure_abcd"]),
            }

            # Mutation B: nonzero transverse kinetic energy -> must fail zero check and isolate to B
            up2_mut_b = np.full_like(x_inj, eps_mut**2)
            vz_mut_b = _vz_from_gamma(gamma_model, up2_mut_b)
            dec_b = decompose_dc_depression(
                nx=nx,
                params=params,
                model_det=det,
                N_l=rho_l / vz_mut_b,
                mean_x_l=x_inj,
                rms_dx_l=zeros,
                mean_gamma_l=gamma_model,
                mean_vz_l=vz_mut_b,
                mean_u_perp2_c2_l=up2_mut_b,
                phi_sim_l=phi_model,
                shape=s.PARTICLE_SHAPE,
            )
            failed_b = False
            try:
                assert_self_check_zero(dec_b, tol=tol)
            except AssertionError:
                failed_b = True
            if not failed_b or abs(dec_b["B"] - dec_b["total"]) > tol:
                raise RuntimeError("Mutation check B failed to isolate channel B.")
            mut_summary["mut_B"] = {
                "failed_as_expected": failed_b,
                "B_minus_total": float(dec_b["B"] - dec_b["total"]),
                "abs_closure_abcd": float(dec_b["abs_closure_abcd"]),
            }

            # Mutation C: shift mean_x_l -> must fail zero check and isolate to C
            x_mut_c = x_inj * (1.0 - eps_mut)
            det_mut_c = s.compute_discrete_dc_depression(
                nx,
                params,
                line_x=x_mut_c,
                line_w=rho_l,
                weighting="charge",
                return_details=True,
                shape=s.PARTICLE_SHAPE,
            )
            u_mut_c = np.asarray(det_mut_c["u_l"], dtype=np.float64)
            dec_c = decompose_dc_depression(
                nx=nx,
                params=params,
                model_det=det,
                N_l=rho_l / u_mut_c,
                mean_x_l=x_mut_c,
                rms_dx_l=np.abs(x_mut_c - x_inj),
                mean_gamma_l=np.asarray(det_mut_c["gamma_l"], dtype=np.float64),
                mean_vz_l=u_mut_c,
                mean_u_perp2_c2_l=zeros,
                phi_sim_l=np.asarray(det_mut_c["phi_l"], dtype=np.float64),
                shape=s.PARTICLE_SHAPE,
            )
            failed_c = False
            try:
                assert_self_check_zero(dec_c, tol=tol)
            except AssertionError:
                failed_c = True
            if not failed_c or abs(dec_c["C"] - dec_c["total"]) > tol:
                raise RuntimeError("Mutation check C failed to isolate channel C.")
            mut_summary["mut_C"] = {
                "failed_as_expected": failed_c,
                "C_minus_total": float(dec_c["C"] - dec_c["total"]),
                "abs_closure_abcd": float(dec_c["abs_closure_abcd"]),
            }

            # Mutation D: uniform weight instead of rho_l / u_model_l -> must fail zero check and isolate to D
            dec_d = decompose_dc_depression(
                nx=nx,
                params=params,
                model_det=det,
                N_l=rho_l.copy(),
                mean_x_l=x_inj,
                rms_dx_l=zeros,
                mean_gamma_l=gamma_model,
                mean_vz_l=u_model,
                mean_u_perp2_c2_l=zeros,
                phi_sim_l=phi_model,
                shape=s.PARTICLE_SHAPE,
            )
            failed_d = False
            try:
                assert_self_check_zero(dec_d, tol=tol)
            except AssertionError:
                failed_d = True
            if not failed_d or abs(dec_d["D"] - dec_d["total"]) > tol:
                raise RuntimeError("Mutation check D failed to isolate channel D.")
            mut_summary["mut_D"] = {
                "failed_as_expected": failed_d,
                "D_minus_total": float(dec_d["D"] - dec_d["total"]),
                "abs_closure_abcd": float(dec_d["abs_closure_abcd"]),
            }

        results[label] = {
            "nx": int(nx),
            "n_macro": len(x_inj),
            "model_details_keys": list(det.keys()),
            "A": float(dec["A"]),
            "B": float(dec["B"]),
            "C": float(dec["C"]),
            "D": float(dec["D"]),
            "energy_residual": float(dec["energy_residual"]),
            "jensen_vz": float(dec["jensen_vz"]),
            "total": float(dec["total"]),
            "closure_abcd": float(dec["closure_abcd"]),
            "closure": float(dec["closure"]),
            "max_abs_part": float(max_abs_part),
            "tol": float(tol),
            "passed": True,
            "mutation_checks": mut_summary,
        }

    print(f"SELF_CHECK {json.dumps(results, sort_keys=True)}", flush=True)
    return results


def run_dc_decomposition_grid(
    grid_label: str,
    *,
    nx_override: int | None = None,
    nz_override: int | None = None,
    n_settle_override: int | None = None,
    n_window_override: int | None = None,
    lz_scale_with_nz: bool = False,
    force_cpu: bool = False,
    m_mod: float = 0.0,
) -> dict[str, Any]:
    """Run ``sheet_lattice`` drift simulation at ``m_mod`` and decompose per-line DC depression.

    Replicates the setup and measurement loop of ``run_particle_beam_drift`` in
    ``examples/_space_charge_waves_common.py`` and
    ``test_two_resolution_convergence_and_sheet_ensemble`` in
    ``tests/test_087_particles_vs_linearized_beam.py`` (``profile="sheet_lattice"``,
    ``seed=DEFAULT_SEED``, ``qm_scale=1.0``, ``shape=PARTICLE_SHAPE``).

    Parameters
    ----------
    grid_label : {"coarse", "fine", "smoke"}
        Grid preset selector.
    nx_override, nz_override : int, optional
        Optional cell-count overrides (used for the tiny CPU smoke check).
    n_settle_override, n_window_override : int, optional
        Optional step-count overrides (used for the tiny CPU smoke check).
    lz_scale_with_nz : bool, optional
        If ``True``, scale ``lz = TUNNEL_LENGTH_M * (nz / COARSE_GRID[2])`` so
        ``dz`` matches ``COARSE_GRID`` during a tiny reduced-``nz`` smoke check.
        Default is ``False``.
    force_cpu : bool, optional
        If ``True``, force ``use_gpu=False`` even if CuPy/CUDA is available.
        Default is ``False``.
    m_mod : float, optional
        Fractional AC modulation index ``m``. Default is ``0.0`` (pure DC).

    Returns
    -------
    dict of str to Any
        Full per-line measurement and ``(A)-(D)`` decomposition dictionary,
        also printed as ``DCDEC {json}`` to stdout.
    """
    if grid_label == "coarse":
        nx_default, _, nz_default = s.COARSE_GRID
    elif grid_label == "fine":
        nx_default, _, nz_default = s.FINE_GRID
    elif grid_label == "smoke":
        nx_default, _, nz_default = s.COARSE_GRID[0], 1, 16
    else:
        raise ValueError(
            f"Unknown grid_label={grid_label!r}; expected 'coarse', 'fine', or 'smoke'."
        )

    nx = int(nx_override) if nx_override is not None else int(nx_default)
    nz = int(nz_override) if nz_override is not None else int(nz_default)
    use_gpu = bool(s.USE_GPU and not force_cpu)

    qm_scale = 1.0
    seed = DEFAULT_SEED
    profile = "sheet_lattice"
    cfln = s.CFL_NUMBER
    shape = s.PARTICLE_SHAPE
    gather = "galerkin"

    params = s.compute_analytic_parameters(qm_scale=qm_scale)
    gamma0 = float(params["gamma0"])
    omega = float(params["omega"])
    i0_dc = float(params["I0"])
    a = s.TUNNEL_HALF_WIDTH_M
    b = s.BEAM_HALF_THICKNESS_M
    ly = s.PERIODIC_LY_M
    lz = (
        float(s.TUNNEL_LENGTH_M) * (float(nz) / float(s.COARSE_GRID[2]))
        if lz_scale_with_nz
        else float(s.TUNNEL_LENGTH_M)
    )
    t_rise = s.RISE_PERIODS / s.DRIVE_FREQ_HZ

    n_macro = 16 if nx == s.COARSE_GRID[0] else 32
    model_det = s.compute_discrete_dc_depression(
        nx,
        params,
        weighting="charge",
        return_details=True,
        shape=shape,
    )
    details_keys = list(model_det.keys())
    print(
        f"MODEL_DETAILS grid={grid_label} nx={nx} keys={details_keys}",
        flush=True,
    )
    x_sheet = np.asarray(model_det["x_sheet"], dtype=np.float64)

    # Build reference LinearizedBeam on the same grid (for s_coords and k_indices)
    if not lz_scale_with_nz:
        beam_ref = s._build_reference_linearized_beam(
            nx, nz, rho0_scale=1.0, m_mod=s.MODULATION_INDEX
        )
        z_coords = np.asarray(beam_ref.s_coords, dtype=np.float64).copy()
        del beam_ref
    else:
        z_edges_tmp = np.linspace(0.0, lz, nz + 1)
        z_coords = 0.5 * (z_edges_tmp[:-1] + z_edges_tmp[1:])

    grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, 1, nz, verbose=0)
    dt, n_spp = s._integer_period_dt(grid, cfln=cfln)
    solver = SolverFIT3D(
        grid,
        cfln=cfln,
        dt=dt,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=use_gpu,
        verbose=0,
    )

    if n_settle_override is not None:
        n_settle = int(n_settle_override)
    elif grid_label == "smoke":
        n_settle = int(np.ceil(lz / (float(params["u0"]) * dt))) + nx
    else:
        n_settle = int(s.settle_steps("particles", nx, nz, params, dt, shape=shape))

    if n_window_override is not None:
        n_window = int(n_window_override)
    elif grid_label == "smoke":
        n_window = int(nx)
    else:
        n_window = int(n_spp)
    n_total = n_settle + n_window

    est_cap = int((n_total + n_macro) * n_macro)
    sp = Species(
        name="electron",
        charge=-e_charge,
        mass=m_e / qm_scale,
        capacity=est_cap,
        track_ids=True,
    )

    def current_modulation(t: float) -> float:
        ramp_t = 0.5 * (1.0 - np.cos(np.pi * min(t / t_rise, 1.0)))
        return i0_dc * (1.0 + float(m_mod) * ramp_t * np.sin(omega * t))

    inj = _LineTrackingBeamInjector(
        species=sp,
        z0=0.0,
        current=current_modulation,
        gamma=gamma0,
        profile=profile,
        radius=b,
        x0=0.0,
        y0=0.0,
        n_macro_per_step=n_macro,
        seed=seed,
        line_x=x_sheet,
    )
    # Verify at setup that BeamInjector._sheet_lattice_x matches model x_sheet
    x_inj_check = inj._sheet_lattice_x(n_macro)
    max_inj_x_err = float(np.max(np.abs(x_inj_check - x_sheet)))
    if max_inj_x_err > SELF_CHECK_TOL * b:
        raise RuntimeError(
            f"Injector lattice x mismatch vs model x_sheet: {max_inj_x_err:.3e} m"
        )

    ps = ParticleSolver(
        solver,
        species=[sp],
        injectors=[inj],
        shape=int(shape),
        filter=s.PARTICLE_FILTER,
        gather=gather,
        B_ext=(0.0, 0.0, s.GUIDE_BZ_TESLA),
    ).attach()

    dx = 2.0 * a / nx
    z_edges = np.linspace(0.0, lz, nz + 1)
    x_node_frac = np.arange(nx + 1, dtype=np.float64)[:, None] / float(nx)
    interior = (z_coords >= 0.2 * lz) & (z_coords <= 0.8 * lz)

    # Cell-binned accumulators (matching run_particle_beam_drift lines 3332-3334)
    w_sum_dc = np.zeros(nz, dtype=np.float64)
    wv_sum_dc = np.zeros(nz, dtype=np.float64)
    n_dc_samples = 0

    # Per-line accumulators over interior cells in the first RF period of the window
    w_line_sum = np.zeros(n_macro, dtype=np.float64)
    wx_line_sum = np.zeros(n_macro, dtype=np.float64)
    wdx2_line_sum = np.zeros(n_macro, dtype=np.float64)
    wgamma_line_sum = np.zeros(n_macro, dtype=np.float64)
    wvz_line_sum = np.zeros(n_macro, dtype=np.float64)
    wuperp2_line_sum = np.zeros(n_macro, dtype=np.float64)
    wphi_line_sum = np.zeros(n_macro, dtype=np.float64)

    for step in range(n_total):
        solver.one_step()
        if step >= n_settle:
            k = step - n_settle
            if k < n_window:
                # Deposited-field transverse potential phi(x, z) with PEC phi(+-a) = 0
                ex_raw = solver.E[:, :, :, "x"]
                ex_host = np.asarray(
                    ex_raw.get() if hasattr(ex_raw, "get") else ex_raw,
                    dtype=np.float64,
                )
                ex_cell = 0.5 * (ex_host[:nx, 0, :nz] + ex_host[:nx, 0, 1 : nz + 1])
                phi_nodes = np.zeros((nx + 1, nz), dtype=np.float64)
                phi_nodes[1:, :] = -dx * np.cumsum(ex_cell, axis=0)
                phi_nodes -= x_node_frac * phi_nodes[-1:, :]

                xp = np.asarray(
                    sp.x.get() if hasattr(sp.x, "get") else sp.x, dtype=np.float64
                )
                zp = np.asarray(
                    sp.z.get() if hasattr(sp.z, "get") else sp.z, dtype=np.float64
                )
                ux_p = np.asarray(
                    sp.ux.get() if hasattr(sp.ux, "get") else sp.ux, dtype=np.float64
                )
                uy_p = np.asarray(
                    sp.uy.get() if hasattr(sp.uy, "get") else sp.uy, dtype=np.float64
                )
                uz_p = np.asarray(
                    sp.uz.get() if hasattr(sp.uz, "get") else sp.uz, dtype=np.float64
                )
                wp = np.asarray(
                    sp.w.get() if hasattr(sp.w, "get") else sp.w, dtype=np.float64
                )

                u_perp2_c2_p = (ux_p**2 + uy_p**2) / (c_light**2)
                gamma_p = np.sqrt(1.0 + u_perp2_c2_p + (uz_p / c_light) ** 2)
                vz_p = uz_p / gamma_p
                bin_idx = np.clip(np.digitize(zp, z_edges) - 1, 0, nz - 1)

                w_k = np.bincount(bin_idx, weights=wp, minlength=nz)
                wv_k = np.bincount(bin_idx, weights=wp * vz_p, minlength=nz)
                w_sum_dc += w_k
                wv_sum_dc += wv_k
                n_dc_samples += 1

                in_int = interior[bin_idx]
                if np.any(in_int):
                    x_i = xp[in_int]
                    w_i = wp[in_int]
                    b_i = bin_idx[in_int]
                    if getattr(sp, "track_ids", False) and sp.id is not None:
                        idp = np.asarray(
                            sp.id.get() if hasattr(sp.id, "get") else sp.id,
                            dtype=np.int64,
                        )
                        line_i = inj.line_id_by_pid[idp[in_int]]
                    else:
                        line_i = inj.line_id_by_order[-len(xp) :][in_int]
                    up2_i = u_perp2_c2_p[in_int]
                    gamma_i = gamma_p[in_int]
                    vz_i = vz_p[in_int]

                    w_mat_i = np.asarray(
                        s.transverse_weight_matrix(x_i, nx, shape),
                        dtype=np.float64,
                    )
                    phi_signed_i = np.sum(w_mat_i * phi_nodes[:, b_i].T, axis=1)
                    phi_sim_i = -phi_signed_i
                    dx_inj_i = x_i - x_sheet[line_i]

                    w_line_sum += np.bincount(line_i, weights=w_i, minlength=n_macro)
                    wx_line_sum += np.bincount(
                        line_i, weights=w_i * x_i, minlength=n_macro
                    )
                    wdx2_line_sum += np.bincount(
                        line_i, weights=w_i * (dx_inj_i**2), minlength=n_macro
                    )
                    wgamma_line_sum += np.bincount(
                        line_i, weights=w_i * gamma_i, minlength=n_macro
                    )
                    wvz_line_sum += np.bincount(
                        line_i, weights=w_i * vz_i, minlength=n_macro
                    )
                    wuperp2_line_sum += np.bincount(
                        line_i, weights=w_i * up2_i, minlength=n_macro
                    )
                    wphi_line_sum += np.bincount(
                        line_i, weights=w_i * phi_sim_i, minlength=n_macro
                    )

    max_inj_x_err_runtime = float(inj.max_inj_x_err_m)
    sp_charge_abs = abs(float(sp.charge))
    del ps, inj, sp, solver, grid
    if use_gpu and s.cp is not None:
        gc.collect()
        s.cp.get_default_memory_pool().free_all_blocks()
        s.cp.get_default_pinned_memory_pool().free_all_blocks()

    if np.any(w_line_sum <= 0.0):
        raise RuntimeError(
            f"Grid {grid_label!r} had empty interior line bins: {w_line_sum.tolist()}"
        )

    w_mean_cell = w_sum_dc / max(n_dc_samples, 1)
    wv_mean_cell = wv_sum_dc / max(n_dc_samples, 1)
    u_z_mean_cells = float(
        np.sum(wv_mean_cell[interior]) / np.sum(w_mean_cell[interior])
    )
    delta_meas_cells = float((params["u0"] - u_z_mean_cells) / params["u0"])

    # Per-line charge N_l [C] (time-averaged over the n_dc_samples steps in the interior)
    N_l = sp_charge_abs * (w_line_sum / max(n_dc_samples, 1))
    mean_x_l = wx_line_sum / w_line_sum
    rms_dx_l = np.sqrt(wdx2_line_sum / w_line_sum)
    mean_gamma_l = wgamma_line_sum / w_line_sum
    mean_vz_l = wvz_line_sum / w_line_sum
    mean_u_perp2_c2_l = wuperp2_line_sum / w_line_sum
    phi_sim_l = wphi_line_sum / w_line_sum

    decomp = decompose_dc_depression(
        nx=nx,
        params=params,
        model_det=model_det,
        N_l=N_l,
        mean_x_l=mean_x_l,
        rms_dx_l=rms_dx_l,
        mean_gamma_l=mean_gamma_l,
        mean_vz_l=mean_vz_l,
        mean_u_perp2_c2_l=mean_u_perp2_c2_l,
        phi_sim_l=phi_sim_l,
        shape=shape,
    )

    out = {
        "grid": grid_label,
        "nx": nx,
        "nz": nz,
        "n_macro": n_macro,
        "m_mod": float(m_mod),
        "m_mod_zero_supported": True,
        "use_gpu": use_gpu,
        "dt": float(dt),
        "n_spp": int(n_spp),
        "n_settle": int(n_settle),
        "n_window": int(n_window),
        "n_dc_samples": int(n_dc_samples),
        "lz_m": float(lz),
        "max_inj_x_err_m": max_inj_x_err_runtime,
        "model_details_keys": details_keys,
        "u_z_mean_cells": u_z_mean_cells,
        "delta_meas_cells": delta_meas_cells,
        "cells_vs_lines_delta_diff": float(delta_meas_cells - decomp["delta_meas"]),
        **decomp,
    }
    print(
        f"CLOSURE grid={grid_label} nx={nx} nz={nz} "
        f"total={decomp['total']:.12e} "
        f"sum_abcd={decomp['sum_abcd']:.12e} "
        f"|closure_abcd|={decomp['abs_closure_abcd']:.12e} "
        f"sum_parts={decomp['sum_parts']:.12e} "
        f"|closure|={decomp['abs_closure']:.12e}",
        flush=True,
    )
    print(f"DCDEC {json.dumps(out, sort_keys=True)}", flush=True)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Per-line decomposition of the DC-depression gap at m = 0."
    )
    parser.add_argument(
        "--grid",
        choices=("coarse", "fine", "both", "smoke"),
        default="both",
        help="Grid to run: 'coarse' (10x474), 'fine' (20x948), 'both', or 'smoke' (tiny CPU check).",
    )
    parser.add_argument(
        "--self-check-only",
        action="store_true",
        help="Run only the synthetic model-identity self-check and mutation checks.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even if GPU is available.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=None,
        help="Optional transverse cell count override (for smoke testing).",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=None,
        help="Optional axial cell count override (for smoke testing).",
    )
    parser.add_argument(
        "--n-settle",
        type=int,
        default=None,
        help="Optional settle step count override (for smoke testing).",
    )
    parser.add_argument(
        "--n-window",
        type=int,
        default=None,
        help="Optional measurement window step count override (for smoke testing).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``studies/p5b-dcdep/dc_decomp.py``."""
    args = parse_args(argv)
    run_self_check(tol=SELF_CHECK_TOL, run_mutation_check=True)
    if args.self_check_only:
        return 0

    if args.grid == "smoke":
        run_dc_decomposition_grid(
            "smoke",
            nx_override=args.nx,
            nz_override=args.nz,
            n_settle_override=args.n_settle,
            n_window_override=args.n_window,
            lz_scale_with_nz=True,
            force_cpu=True,
            m_mod=0.0,
        )
        return 0

    grids = ("coarse", "fine") if args.grid == "both" else (args.grid,)
    for g in grids:
        run_dc_decomposition_grid(
            g,
            nx_override=args.nx,
            nz_override=args.nz,
            n_settle_override=args.n_settle,
            n_window_override=args.n_window,
            force_cpu=bool(args.cpu),
            m_mod=0.0,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
