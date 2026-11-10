# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Exact discrete first-pass prediction of the `LinearizedBeam` ``beta_q`` (Decision D21).

The legacy `LinearizedBeam` prediction
``_space_charge_waves_common.discrete_modal_prediction("lb", ...)`` synthesises
``I1(z) = sum_j c_j cos(beta_q,j z) exp(-i k_c,j z)`` (transverse modes frozen at
``beta_e``) with roots of ``Omega_tan - u0 sin(k dz)/dz = +- (omega_p/gamma^1.5) sqrt(lambda_j(k))``.
Against the LB's own discrete scheme it makes three approximations (derivation in
``first_principles.md``, "LinearizedBeam discrete space-charge waves (D21)"):

1. **Crank-Nicolson time symbol.** With ``E^n`` and ``(v1, rho1, J1)^{n+1/2}`` the
   leapfrog + CN step has the exact symbol ``(Omega_f - u0 K c_h)^2 = (omega_p^2/gamma^3) lambda_j``
   (``Omega_f = (2/dt) sin(omega dt/2)``, ``c_h = cos(omega dt/2)``, ``K = sin(k dz)/dz``),
   i.e. ``Omega_tan - u0 K = +- omega_p sqrt(lambda_j) / (gamma^1.5 c_h)``. The legacy
   roots omit ``1/c_h`` (``beta_q`` low by ``~1/c_h - 1``).
2. **cos-form amplitudes** (equal fast/slow amplitudes per mode, frozen eigenvectors):
   the two-wave fit turns the wrong higher-mode amplitudes into a ``beta_q`` bias.
3. **No entrance wall.** The ``z = 0`` PEC face is ignored; its even ``J_z`` image changes
   the entrance amplitudes of every transverse mode.

This module replaces the model by the *time-harmonic form of the discrete scheme itself*:
``E^n = E_hat z^n``, ``H^{n+1/2} = H_hat z^{n+1/2}``, ``(v1, rho1, J1)^{n+1/2} = (.)_hat z^{n+1/2}``,
``z = exp(i omega dt)``. The `SolverFIT3D` leapfrog (``H -= dt A E``; ``E += dt (B H - ieps J)``)
and the `LinearizedBeam` CN step become one sparse system assembled from the solver's
``A = tDsiDmuiDaC``, ``B = itDaiDepsDstC``, ``ieps`` and the beam's ``M_+-``, ``D_s``,
coupling arrays and inflow-ghost coefficient (:func:`lb_exact_phasor`; one TD step of its
solution reproduces it to ~1e-15).

**First pass, not steady state.** The centred ``D_s`` supports a parasitic branch
``k' ~ pi/dz - k`` with *negative* group velocity. The outflow taper/upwind closure
reflects the physical waves into it; it reaches the entrance only after
``t_reach0 = z_ramp / v_phys + z_ramp / v_para`` (:func:`lb_branch_group_velocities`), later than the end of the
production DFT window, so the measured ``I1`` is the *first-pass* state (no outflow
reflection) plus a parasitic front in the downstream part of the window. The full
steady state (multi-bounce) differs by ~1e-4 in ``beta_q`` and is not what the runs
measure. :func:`lb_first_pass_phasor` builds the first-pass state by replacing the
outflow taper with a long full-density beam extension ending in a C-infinity sponge
(field-coupled region unchanged up to ``z_ramp``).

Limits & Failure Modes
----------------------
- Valid only while ``t_end < t_reach0`` (checked: ``first_pass_valid``); later windows
  see the outflow reflection re-reflected at the entrance (``beta_q`` swings ~1e-3).
  The front timing uses drive-frequency group velocities: off-frequency turn-on
  components (and CN's exponentially small implicit precursors) are not bounded; the
  TD settle scan (``studies/lb-modal-prediction``) shows the plateau up to 1.25x settle.
- The downstream replacement changes the near field at the window end by
  ``<= EPS_WINDOW`` (by construction of ``d_w``): bias ``s_tail = eps_w_bias``.
- The parasitic front inside the window is bounded, not modelled (``s_front``):
  linearised (Gauss-Newton) fit sensitivity, step-front envelope, front amplitude
  ``<= 2 x`` the steady parasitic amplitude (passive loop, ``|L| <= 1``).
- ``rho0_scale`` scales ``rho0`` but not the inflow drive (as the M2 mutation test does).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.constants import c as c_light
from scipy.optimize import brentq
from scipy.signal import firwin
from scipy.sparse import bmat, diags, eye, kron
from scipy.sparse import csr_matrix as sp_csr
from scipy.sparse.linalg import splu

from examples._space_charge_waves_common import (
    BEAM_HALF_THICKNESS_M,
    CFL_NUMBER,
    DRIVE_FREQ_HZ,
    EPS_WINDOW,
    MODULATION_INDEX,
    OUTFLOW_RAMP_M,
    PERIODIC_LY_M,
    TUNNEL_HALF_WIDTH_M,
    TUNNEL_LENGTH_M,
    _grid_gamma1_disc_and_dw,
    _integer_period_dt,
    compute_analytic_parameters,
    fit_beta_q_from_envelope,
    fit_window,
    settle_steps,
)
from wakis import GridFIT3D, SolverFIT3D
from wakis.beam_fluid import LinearizedBeam

# First-pass construction: downstream beam extension [m] (= 200 coarse cells) ending in a
# C-infinity sponge of FP_SPONGE_NP nepers one-way; the convergence check doubles both.
FP_EXTENSION_M = 200 * TUNNEL_LENGTH_M / 474
FP_SPONGE_NP = 10.0
# Parasitic-band separation: Blackman FIR low-pass, cutoff pi / (2 dz), 41 taps
# (half-length 20 cells < d_w on both grids).
PARA_FIR_TAPS = 41
S_ROUND = 1.0e-12


def _build_cpu_beam(
    nx: int,
    nz: int,
    rho0_scale: float = 1.0,
    outflow_ramp_m: float = OUTFLOW_RAMP_M,
    bc_z_low: str = "pec",
    entrance_gap_cells: int = 0,
    extension_cells: int = 0,
) -> tuple[SolverFIT3D, LinearizedBeam, float]:
    """Build the CPU `SolverFIT3D` + `LinearizedBeam` pair of ``run_linearized_beam_drift``.

    ``entrance_gap_cells > 0`` prepends beam-free vacuum cells upstream of the entrance
    (the PEC wall moves to ``z = -gap``: "no wall image" control). ``extension_cells > 0``
    replaces the outflow taper by that many extra full-density beam cells beyond ``Lz``.
    """
    params = compute_analytic_parameters()
    a = TUNNEL_HALF_WIDTH_M
    ly = PERIODIC_LY_M
    lz = TUNNEL_LENGTH_M
    dz = lz / int(nz)
    n_gap = int(entrance_gap_cells)
    n_ext = int(extension_cells)
    n_ramp = round(float(outflow_ramp_m) / dz) if n_ext == 0 else 0
    grid = GridFIT3D(
        -a,
        a,
        -0.5 * ly,
        0.5 * ly,
        -n_gap * dz,
        lz + n_ext * dz,
        int(nx),
        1,
        int(nz) + n_gap + n_ext,
        verbose=0,
    )
    dt, _ = _integer_period_dt(grid)
    solver = SolverFIT3D(
        grid,
        cfln=CFL_NUMBER,
        dt=dt,
        bc_low=["pec", "periodic", str(bc_z_low)],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=False,
        verbose=0,
    )
    beam = LinearizedBeam(
        u0=params["u_dep"],
        rho0=float(rho0_scale) * params["rho_dep"],
        gamma0=params["gamma_dep"],
        axis="z",
        half_thickness=BEAM_HALF_THICKNESS_M,
        ramp_cells=n_ramp,
        inflow_rho1=lambda t: 0.0,
        s_min=0.0 if n_gap > 0 else None,
        solver=solver,
    )
    return solver, beam, float(dt)


def _cinf_ramp(n: int) -> np.ndarray:
    """C-infinity 0 -> 1 profile on ``n`` cells (values at ``s = 1/n .. 1``)."""
    s_r = np.arange(1, n + 1, dtype=np.float64) / n
    with np.errstate(divide="ignore", over="ignore"):
        f1 = np.where(s_r > 0, np.exp(-1.0 / np.maximum(s_r, 1e-300)), 0.0)
        f2 = np.where(s_r < 1, np.exp(-1.0 / np.maximum(1.0 - s_r, 1e-300)), 0.0)
    return f1 / (f1 + f2)


def lb_exact_phasor(
    nx: int,
    nz: int,
    rho0_scale: float = 1.0,
    m_mod: float = MODULATION_INDEX,
    outflow_ramp_m: float = OUTFLOW_RAMP_M,
    bc_z_low: str = "pec",
    entrance_gap_cells: int = 0,
    extension_cells: int = 0,
    sponge_np: float = 0.0,
    sponge_cells: int = 0,
) -> dict[str, Any]:
    """Solve the time-harmonic discrete FIT + `LinearizedBeam` scheme for ``I1_hat(z)`` [A].

    Parameters
    ----------
    nx, nz : int
        Transverse and axial cell counts of the production grid.
    rho0_scale : float, optional
        Multiplier on ``rho0`` (the inflow drive stays anchored at ``rho_dep``).
    m_mod : float, optional
        Modulation index ``m`` (the solve is linear in ``m``).
    outflow_ramp_m : float, optional
        Outflow taper length [m] (ignored when ``extension_cells > 0``).
    bc_z_low : str, optional
        Entrance-face field boundary (``"pec"`` as in the runs; ``"pmc"``: wrong-wall control).
    entrance_gap_cells : int, optional
        Beam-free vacuum cells upstream of the entrance ("no wall image" control).
    extension_cells : int, optional
        Full-density beam cells added beyond ``Lz`` in place of the outflow taper.
    sponge_np, sponge_cells : float, int, optional
        Beam-variable damping ``nu(z) = nu_max p(z)`` (C-infinity ``p`` from 0 to 1) on
        the last ``sponge_cells`` beam edges, added to the ``v1`` and ``rho1`` rows, with
        one-way amplitude attenuation ``int nu dz / u0 = sponge_np`` nepers.

    Returns
    -------
    dict of str to Any
        ``z_coords`` [m], ``i1`` [A] (complex phasor with the DFT convention
        ``(2/N) sum_n exp(-i omega t_n) I(t_n)``), ``e_hat``, ``v1``, ``rho1``, ``j1``,
        ``dt`` [s], ``c_half`` = ``cos(omega dt / 2)``, ``n_unknowns`` and
        ``solve_resid`` (scaled residual after iterative refinement).
    """
    params = compute_analytic_parameters()
    omega = float(params["omega"])
    solver, beam, dt = _build_cpu_beam(
        nx,
        nz,
        rho0_scale,
        outflow_ramp_m,
        bc_z_low=bc_z_low,
        entrance_gap_cells=entrance_gap_cells,
        extension_cells=extension_cells,
    )

    a_op = sp_csr(solver.tDsiDmuiDaC)
    b_op = sp_csr(solver.itDaiDepsDstC)
    ieps = np.asarray(solver.ieps.toarray(), dtype=np.float64)
    n_e = a_op.shape[1]
    zh = np.exp(0.5j * omega * dt)  # z^{1/2}
    two_is = zh - 1.0 / zh  # 2 i sin(omega dt / 2)
    c_half = 0.5 * (zh + 1.0 / zh)  # cos(omega dt / 2)

    m_ax = int(beam.M)
    n_l = int(beam.N_lines)
    n_b = m_ax * n_l
    flat = np.asarray(beam.flat_idx_host, dtype=np.int64)  # row-major (M, N_lines)
    sel = sp_csr(
        (np.ones(n_b), (np.arange(n_b), flat)), shape=(n_b, n_e), dtype=np.float64
    )
    i_l = eye(n_l, format="csr")

    # Field rows: (z - 1) E = dt z^{1/2} (B H - ieps J), H = -dt A E / (z^{1/2} - z^{-1/2})
    #   => [two_is^2 + dt^2 B A] E + two_is dt ieps P^T J = 0
    k_ee = (two_is**2) * eye(n_e, format="csr") + (dt**2) * (b_op @ a_op)
    ieps_pt = diags(ieps) @ sel.T

    mp_v = kron(sp_csr(beam.Mp_v), i_l)
    mm_v = kron(sp_csr(beam.Mm_v), i_l)
    mp_r = kron(sp_csr(beam.Mp_r), i_l)
    mm_r = kron(sp_csr(beam.Mm_r), i_l)
    ds_k = kron(sp_csr(beam.Ds), i_l)
    alpha = np.asarray(beam._alpha_E, dtype=np.float64).ravel()
    beta_v = np.asarray(beam._beta_v, dtype=np.float64).ravel()
    c_v = np.asarray(beam._c_v, dtype=np.float64).ravel()
    c_rho = np.asarray(beam._c_rho, dtype=np.float64).ravel()

    if sponge_np > 0.0 and int(sponge_cells) > 0:
        n_r = int(sponge_cells)
        prof = np.zeros(m_ax)
        prof[m_ax - n_r :] = _cinf_ramp(n_r)
        dz_b = float(np.mean(beam.ds_line))
        nu_max = float(sponge_np) * abs(beam.u0) / (np.sum(prof) * dz_b)
        nu_k = kron(diags(0.5 * dt * nu_max * prof), i_l)
        mp_v, mm_v = mp_v + nu_k, mm_v - nu_k
        mp_r, mm_r = mp_r + nu_k, mm_r - nu_k

    # Beam rows: (M+ z^{1/2} - M- z^{-1/2}) v = alpha_E (P E)
    k_vv = zh * mp_v - mm_v / zh
    k_ve = -diags(alpha) @ sel
    # (M+ z^{1/2} - M- z^{-1/2}) rho = beta_v D_s (c_h v) + ghost(t^n)
    k_rr = zh * mp_r - mm_r / zh
    k_rv = -c_half * (diags(beta_v) @ ds_k)
    # J^{n+1/2} = c_v v + c_rho rho  -> field rows
    k_ev = two_is * dt * (ieps_pt @ diags(c_v))
    k_er = two_is * dt * (ieps_pt @ diags(c_rho))
    big = bmat(
        [[k_ee, k_ev, k_er], [k_ve, k_vv, None], [None, k_rv, k_rr]], format="csc"
    ).astype(np.complex128)

    # Inflow drive m rho_dep sin(omega t^n) = Re(-i m rho_dep z^n); the drive time level
    # only sets a global phase of the linear response, which beta_q does not see.
    rho_in_hat = -1j * float(m_mod) * float(params["rho_dep"])
    rhs = np.zeros(big.shape[0], dtype=np.complex128)
    k_in = int(beam._k_in)
    g_in = (
        np.asarray(beam._inflow_ghost_coeff, dtype=np.float64)
        * np.asarray(beam._w_perp_eff, dtype=np.float64)[k_in, :]
        * rho_in_hat
    )
    off_r = n_e + n_b
    rhs[off_r + k_in * n_l : off_r + (k_in + 1) * n_l] = g_in

    # Diagonal equilibration (entries span ~1e-4 .. 1e8) + LU + iterative refinement
    r_scale = 1.0 / np.maximum(
        np.asarray(abs(big).max(axis=1).todense()).ravel(), 1e-300
    )
    big_r = diags(r_scale) @ big
    c_scale = 1.0 / np.maximum(
        np.asarray(abs(big_r).max(axis=0).todense()).ravel(), 1e-300
    )
    big_s = (big_r @ diags(c_scale)).tocsc()
    rhs_s = r_scale * rhs
    lu = splu(big_s)
    y = lu.solve(rhs_s)
    for _ in range(4):
        y = y + lu.solve(rhs_s - big_s @ y)
    sol = c_scale * y
    solve_resid = float(np.linalg.norm(big_s @ y - rhs_s) / np.linalg.norm(rhs_s))

    v_hat = sol[n_e : n_e + n_b].reshape(m_ax, n_l)
    r_hat = sol[n_e + n_b :].reshape(m_ax, n_l)
    j_hat = np.asarray(beam._c_v) * v_hat + np.asarray(beam._c_rho) * r_hat
    i1 = np.sum(j_hat * beam.dA_perp, axis=1)
    return {
        "z_coords": np.asarray(beam.s_coords, dtype=np.float64).copy(),
        "i1": i1,
        "e_hat": sol[:n_e],
        "v1": v_hat,
        "rho1": r_hat,
        "j1": j_hat,
        "dt": dt,
        "c_half": float(np.real(c_half)),
        "n_unknowns": int(big.shape[0]),
        "solve_resid": solve_resid,
    }


def lb_first_pass_phasor(
    nx: int,
    nz: int,
    rho0_scale: float = 1.0,
    extension_m: float = FP_EXTENSION_M,
    sponge_np: float = FP_SPONGE_NP,
    **kwargs: Any,
) -> dict[str, Any]:
    """First-pass ``I1_hat(z)`` on the production edges ``z_k < Lz`` [A].

    The outflow taper is replaced by ``extension_m`` of full-density beam whose whole
    length is a C-infinity sponge of ``sponge_np`` nepers (one way): physical waves leave
    without reflecting into the upstream parasitic branch.
    """
    n_ext = int(round(float(extension_m) / (TUNNEL_LENGTH_M / int(nz))))
    res = lb_exact_phasor(
        nx,
        nz,
        rho0_scale=rho0_scale,
        extension_cells=n_ext,
        sponge_np=sponge_np,
        sponge_cells=n_ext,
        **kwargs,
    )
    m_run = int(np.sum(res["z_coords"] < TUNNEL_LENGTH_M))
    out = dict(res)
    out["z_coords"] = res["z_coords"][:m_run]
    out["i1"] = res["i1"][:m_run]
    out["n_ext"] = n_ext
    return out


def _lb_line_weights(nx: int) -> np.ndarray:
    """Interior-node top-hat fill fractions ``w_l`` of the sheet beam (as `LinearizedBeam`)."""
    a = TUNNEL_HALF_WIDTH_M
    b = BEAM_HALF_THICKNESS_M
    dx = 2.0 * a / nx
    x = -a + dx * np.arange(1, nx)
    lo = np.maximum(x - 0.5 * dx, -b)
    hi = np.minimum(x + 0.5 * dx, b)
    return np.clip(hi - lo, 0.0, None) / dx


def lb_branch_group_velocities(
    nx: int, nz: int, params: dict[str, float]
) -> dict[str, Any]:
    """Group velocities [m/s] of the discrete LB beam waves at the drive frequency ``omega``.

    Each branch obeys ``Omega_f(omega) - u0 K(k) c_h(omega) = s S_j(k, omega)``, ``s = +-1``,
    with ``Omega_f = (2/dt) sin(omega dt/2)``, ``c_h = cos(omega dt/2)``, ``K = sin(k dz)/dz``,
    ``S_j = sqrt(sigma lambda_j)`` (``sigma = omega_p^2/gamma^3``; ``lambda_j`` the eigenvalues
    of ``tau_Y^2 (-L_x + tau_Y^2)^{-1} diag(w)``, ``tau_Y^2 = kappa^2 - (Omega_f/c)^2``).
    Implicit differentiation gives
    ``v_g = (u0 c_h cos(k dz) + s S_k) / (c_h + u0 K (dt/2) sin(omega dt/2) - s S_omega)``.
    Physical roots lie in ``(Omega_f/c, pi/(2 dz))`` (``v_g > 0``), parasitic roots in
    ``(pi/(2 dz), pi/dz)`` (``cos(k dz) < 0``, ``v_g < 0``). Returns ``v_phys_max`` (largest
    physical ``v_g``, all modes and both signs) and ``v_para_max`` (largest parasitic ``|v_g|``).
    """
    a = TUNNEL_HALF_WIDTH_M
    dx = 2.0 * a / nx
    dz = TUNNEL_LENGTH_M / nz
    dt, _ = _integer_period_dt(
        GridFIT3D(
            -a,
            a,
            -0.5 * PERIODIC_LY_M,
            0.5 * PERIODIC_LY_M,
            0.0,
            TUNNEL_LENGTH_M,
            int(nx),
            1,
            int(nz),
            verbose=0,
        )
    )
    omega = float(params["omega"])
    u0 = float(params["u_dep"])
    sigma = float(params["omega_p"] ** 2 / params["gamma_dep"] ** 3)
    w = _lb_line_weights(nx)
    n_int = nx - 1
    lap = (
        np.diag(np.full(n_int, -2.0))
        + np.diag(np.ones(n_int - 1), 1)
        + np.diag(np.ones(n_int - 1), -1)
    ) / dx**2
    sw = np.sqrt(w)
    n_modes = int(np.sum(w > 0))

    def s_j(k: float, om: float, j: int) -> float:
        om_f = (2.0 / dt) * np.sin(0.5 * om * dt)
        kap = (2.0 / dz) * np.sin(0.5 * k * dz)
        t2 = kap**2 - (om_f / c_light) ** 2
        g = t2 * np.linalg.inv(-lap + t2 * np.eye(n_int))
        ev = np.sort(np.linalg.eigvalsh((sw[:, None] * g) * sw[None, :]))[::-1]
        return float(np.sqrt(max(sigma * ev[j], 0.0)))

    def resid(k: float, om: float, j: int, s: float) -> float:
        om_f = (2.0 / dt) * np.sin(0.5 * om * dt)
        return (
            om_f - u0 * np.sin(k * dz) / dz * np.cos(0.5 * om * dt) - s * s_j(k, om, j)
        )

    c_h = float(np.cos(0.5 * omega * dt))
    om_f0 = (2.0 / dt) * np.sin(0.5 * omega * dt)
    k_light = om_f0 / c_light
    out: dict[str, Any] = {"phys": [], "para": []}
    for j in range(n_modes):
        for s in (-1.0, 1.0):
            for band, lo, hi in (
                ("phys", 1.001 * k_light, 0.5 * np.pi / dz),
                ("para", 0.5 * np.pi / dz, (1.0 - 1e-9) * np.pi / dz),
            ):
                k_r = float(brentq(resid, lo, hi, args=(omega, j, s), xtol=1e-12))
                hk = 1e-6 * k_r
                s_k = (s_j(k_r + hk, omega, j) - s_j(k_r - hk, omega, j)) / (2 * hk)
                hw = 1e-6 * omega
                s_w = (s_j(k_r, omega + hw, j) - s_j(k_r, omega - hw, j)) / (2 * hw)
                num = u0 * c_h * np.cos(k_r * dz) + s * s_k
                den = (
                    c_h
                    + u0 * np.sin(k_r * dz) / dz * 0.5 * dt * np.sin(0.5 * omega * dt)
                    - s * s_w
                )
                out[band].append((j, s, k_r, float(num / den)))
    out["v_phys_max"] = float(max(v for *_, v in out["phys"]))
    out["v_para_max"] = float(max(abs(v) for *_, v in out["para"]))
    return out


def _gn_beta_sensitivity(
    z: np.ndarray, fit_res: dict[str, Any], mask: np.ndarray
) -> np.ndarray:
    """Gauss-Newton sensitivity ``g`` with ``d beta_q = Re(sum conj(g) d I1)`` on ``mask`` [rad/m/A]."""
    zm = z[mask]
    k_c, beta_q = float(fit_res["k_c"]), float(fit_res["beta_q"])
    a_f, a_s = complex(fit_res["a_f"]), complex(fit_res["a_s"])
    u_f = np.exp(-1j * (k_c - beta_q) * zm)
    u_s = np.exp(-1j * (k_c + beta_q) * zm)
    f_model = a_f * u_f + a_s * u_s
    jac = np.column_stack(
        [
            u_f,
            1j * u_f,
            u_s,
            1j * u_s,
            -1j * zm * f_model,
            1j * zm * (a_f * u_f - a_s * u_s),
        ]
    )
    j_r = np.vstack([np.real(jac), np.imag(jac)])
    row = np.linalg.solve(j_r.T @ j_r, j_r.T)[5]
    n = len(zm)
    g = np.zeros(len(z), dtype=np.complex128)
    g[mask] = row[:n] + 1j * row[n:]
    return g


def parasitic_band(i1: np.ndarray, taps: int = PARA_FIR_TAPS) -> np.ndarray:
    """High-band part ``|k| > pi/(2 dz)`` of ``I1(z)`` (``I1 - FIR_lowpass(I1)``)."""
    h = firwin(int(taps), 0.5, window="blackman")
    return i1 - np.convolve(i1, h, mode="same")


@lru_cache(maxsize=16)
def lb_first_pass_prediction(
    nx: int, nz: int, rho0_scale: float = 1.0
) -> dict[str, Any]:
    """D21 prediction ``beta_q`` [rad/m] of the production `LinearizedBeam` run and its derived slack.

    ``slack = s_fp + s_tail + s_front + s_round`` with

    - ``s_fp``: first-pass construction convergence, ``|beta(L_ext, Np) - beta(2 L_ext, 2 Np)| / beta``;
    - ``s_tail = EPS_WINDOW (1/gamma1_disc) / (z_max - z_min)``: the replaced downstream
      structure is seen in the window only through the evanescent tail ``<= EPS_WINDOW``;
    - ``s_front = (1 + rho_fit) max_{z_e} |sum_k conj(g_k) 2 P_k w(z_k; z_e)| / beta``: a
      parasitic front of amplitude ``<= 2 |P|`` (``P`` = steady parasitic band of the exact
      scheme; passive loop ``|L| <= 1``), any constant phase, whose DFT envelope is the
      ramp ``w = clip((z - z_e) / L_front, 0, 1)`` (the front moves
      ``L_front = v_para T_dft`` upstream during the window; any extra packet width is a
      convex mixture of such ramps), with end position ``z_e >= max(z_min, z_front_min)``,
      ``z_front_min = z_ramp - v_para (t_end - z_ramp / v_phys)`` (drive-frequency group
      velocities of :func:`lb_branch_group_velocities`; reflection no earlier than at ``z_ramp``);
      ``g`` is the Gauss-Newton sensitivity of the two-wave fit and ``rho_fit`` the
      relative fit residual (neglected curvature term);
    - ``s_round = 1e-12``.

    Also returns ``first_pass_valid = (z_front_min > 0)`` (no parasitic wave has reached
    the entrance before the end of the DFT window) and the legacy/diagnostic variants.
    """
    params = compute_analytic_parameters()
    z_min, z_max = fit_window(params)
    omega_p_scale = float(np.sqrt(rho0_scale))

    fp = lb_first_pass_phasor(nx, nz, rho0_scale=rho0_scale)
    fp2 = lb_first_pass_phasor(
        nx,
        nz,
        rho0_scale=rho0_scale,
        extension_m=2 * FP_EXTENSION_M,
        sponge_np=2 * FP_SPONGE_NP,
    )
    z = fp["z_coords"]
    fit_fp = fit_beta_q_from_envelope(
        z, fp["i1"], params["beta_q"], z_min=z_min, z_max=z_max
    )
    bq = float(fit_fp["beta_q"])
    bq2 = float(
        fit_beta_q_from_envelope(
            z, fp2["i1"], params["beta_q"], z_min=z_min, z_max=z_max
        )["beta_q"]
    )
    s_fp = abs(bq2 - bq) / bq

    gamma1_disc, _, _ = _grid_gamma1_disc_and_dw(int(nx), int(nz), params["omega"])
    s_tail = float(EPS_WINDOW * (1.0 / gamma1_disc) / (z_max - z_min))

    # Timing of the production run (run_linearized_beam_drift)
    dt = float(fp["dt"])
    n_spp = int(round(1.0 / (DRIVE_FREQ_HZ * dt)))
    n_settle = settle_steps("lb", nx, nz, params, dt, omega_p_scale=omega_p_scale)
    t_end = float((n_settle + 6 * n_spp) * dt)
    vg = lb_branch_group_velocities(nx, nz, params)
    v_ph, v_pa = vg["v_phys_max"], vg["v_para_max"]
    z_ramp = float(TUNNEL_LENGTH_M - OUTFLOW_RAMP_M)
    # earliest reflection at z_ramp, reached no earlier than z_ramp / v_ph after the turn-on
    z_front_min = float(z_ramp - v_pa * (t_end - z_ramp / v_ph))
    t_reach0 = float(z_ramp / v_ph + z_ramp / v_pa)

    exact = lb_exact_phasor(nx, nz, rho0_scale=rho0_scale)
    para = parasitic_band(exact["i1"])
    mask = (z >= z_min) & (z <= z_max)
    g = _gn_beta_sensitivity(z, fit_fp, mask)
    rho_fit = float(
        np.sqrt(fit_fp["rss"] / np.real(np.vdot(fp["i1"][mask], fp["i1"][mask])))
    )
    contrib = np.conj(g) * 2.0 * para
    # During the DFT window (T_dft = 6 periods) the front moves upstream by
    # L_front = v_para T_dft, so the DFT phasor sees the envelope
    # w(z) = clip((z - z_e) / L_front, 0, 1), z_e = front position at the window end.
    l_front = float(v_pa * 6 * n_spp * dt)
    z_e_all = z[mask & (z >= max(z_min, z_front_min) - l_front)]
    if len(z_e_all):
        w_env = np.clip((z[None, :] - z_e_all[:, None]) / l_front, 0.0, 1.0)
        x_env = w_env @ contrib
        s_front = float((1.0 + rho_fit) * np.max(np.abs(x_env)) / bq)
    else:
        s_front = 0.0
    contrib_step = np.cumsum(contrib[::-1])[::-1]
    sel = mask & (z >= max(z_min, z_front_min))
    s_front_step = float(
        (1.0 + rho_fit)
        * (np.max(np.abs(contrib_step[sel])) if np.any(sel) else 0.0)
        / bq
    )

    slack = float(s_fp + s_tail + s_front + S_ROUND)
    return {
        "bq_pred": bq,
        "bq_pred_fp2": bq2,
        "s_fp": float(s_fp),
        "s_tail": s_tail,
        "s_front": s_front,
        "s_front_step": s_front_step,
        "l_front": l_front,
        "s_round": S_ROUND,
        "slack": slack,
        "rho_fit": rho_fit,
        "t_end": t_end,
        "t_reach0": t_reach0,
        "v_phys_max": v_ph,
        "v_para_max": v_pa,
        "z_front_min": z_front_min,
        "first_pass_valid": bool(z_front_min > 0.0),
        "n_settle": int(n_settle),
        "bq_exact_steady": float(
            fit_beta_q_from_envelope(
                z, exact["i1"], params["beta_q"], z_min=z_min, z_max=z_max
            )["beta_q"]
        ),
        "para_amp_max": float(np.max(np.abs(para[mask])) / params["I1_0"]),
        "z_coords": z,
        "i1_fp": fp["i1"],
        "c_half": float(fp["c_half"]),
    }


def lb_error_decomposition(nx: int, nz: int) -> dict[str, float]:
    """Split ``beta_fp / beta_legacy - 1`` into time symbol, cos-form amplitudes and entrance wall.

    - ``d_symbol``: legacy modal with roots ``omega_p -> omega_p / c_h`` minus legacy;
    - ``d_wall``: first-pass with PEC at ``z = 0`` minus first-pass with the wall moved
      ``n_gap`` cells upstream (converged in ``n_gap``);
    - ``d_amplitude``: first-pass without wall minus legacy-with-``c_h`` (cos-form amplitude error).
    All relative to the legacy prediction; they sum to ``beta_fp / beta_legacy - 1``.
    """
    from examples._space_charge_waves_common import discrete_modal_prediction

    params = compute_analytic_parameters()
    z_min, z_max = fit_window(params)
    fp = lb_first_pass_prediction(nx, nz)
    c_h = fp["c_half"]
    b_leg = float(discrete_modal_prediction("lb", nx, nz, params))
    b_leg_ch = float(
        discrete_modal_prediction("lb", nx, nz, params, omega_p_scale=1.0 / c_h)
    )
    n_gap = int(round(0.02 / (TUNNEL_LENGTH_M / nz)))  # 20 mm >> 1/gamma1 = 1.3 mm
    nw = lb_first_pass_phasor(nx, nz, entrance_gap_cells=n_gap)
    b_nowall = float(
        fit_beta_q_from_envelope(
            nw["z_coords"], nw["i1"], params["beta_q"], z_min=z_min, z_max=z_max
        )["beta_q"]
    )
    b_fp = fp["bq_pred"]
    return {
        "bq_legacy": b_leg,
        "bq_legacy_ch": b_leg_ch,
        "bq_fp_nowall": b_nowall,
        "bq_fp": b_fp,
        "bq_exact_steady": fp["bq_exact_steady"],
        "d_symbol": b_leg_ch / b_leg - 1.0,
        "d_amplitude": (b_nowall - b_leg_ch) / b_leg,
        "d_wall": (b_fp - b_nowall) / b_leg,
        "d_total": b_fp / b_leg - 1.0,
        "d_multibounce": fp["bq_exact_steady"] / b_fp - 1.0,
        "c_half_inv_minus_1": 1.0 / c_h - 1.0,
    }
