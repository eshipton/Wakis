"""3D path-potential consistency and step-by-step energy identity study (Part B).

Starting from ``dc_decomp.py``'s ``m = 0`` cold sheet-beam setup and settling
for the same number of transits:

(a) Saves the electric and magnetic fields on the full grid at one settled
    step and computes, for each sheet lattice line :math:`x_l` and interior
    axial coordinate :math:`z`:
    - ``phi_trans``: transverse line integral :math:`-\\int_{-a}^x E_x\\,\\mathrm{d}x'`
      from the ``x = -a`` PEC wall (both raw on primal nodes and with the
      linear drift correction / cell averaging of ``dc_decomp.py``),
    - ``phi_axial``: axial line integral :math:`-\\int_0^z E_z\\,\\mathrm{d}z'`
      along ``z`` from the ``z = 0`` emitter PEC wall at each ``x`` node,
    - both potentials gathered at each lattice line :math:`x_l` with the
      particle's order-:math:`p` B-spline weights,
    - ``y``-direction consistency diagnostics (printing ``Ny`` and the ``y``
      boundary conditions),
    - per-line ``max`` and ``mean`` relative discrepancy
      :math:`|\\phi_{\\text{axial}} - \\phi_{\\text{trans}}| / |\\phi|`,
    - interior discrete Faraday residual
      :math:`\\max|\\nabla_h \\times \\mathbf{E}| / (\\max|\\mathbf{E}| / \\Delta z)`
      and interior :math:`\\max|\\mathbf{H}|`.

(b) Tracks 3 particles per lattice line across one transit and evaluates the
    discrete time-variation sum step by step:

    .. math::

        S_p = \\frac{q}{m c^2} \\sum_{m > n_b}
        \\bigl[\\Phi^m(\\mathbf{x}_p^m) - \\Phi^{m-1}(\\mathbf{x}_p^m)\\bigr]

    with :math:`\\Phi` computed from the order-:math:`p` gathered axial integral
    of :math:`E_z`, reporting :math:`S_p` beside each particle's interior
    Lorentz factor excess
    :math:`\\gamma_{\\text{int}} - \\gamma_0 + (q / m c^2)\\Phi(\\mathbf{x}_{\\text{int}})`.

Includes a fast synthetic self-check on a tiny grid with a curl-free analytic
discrete potential verifying ``phi_axial == phi_trans`` to ``1e-12`` relative,
together with a mutation check that perturbs a single interior edge and verifies
that the check fails as expected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as e_charge
from scipy.constants import m_e

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")
for _p in (_REPO_ROOT, _EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _space_charge_waves_common as s

from wakis import GridFIT3D, SolverFIT3D
from wakis.particles import BeamInjector, ParticleSolver, Species
from wakis.particles.pushers import get_pusher
from wakis.particles.shapes import (
    _gather_component_3d,
    bspline_weights_1d,
    gather_fields,
)

GRID_MAP: dict[str, tuple[int, int, int]] = {
    "coarse": s.COARSE_GRID,
    "fine": s.FINE_GRID,
}


def _to_host(arr) -> np.ndarray:
    """Return a host ``numpy.ndarray`` copy of ``arr``."""
    if hasattr(arr, "get"):
        return np.asarray(arr.get(), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def gather_node_potential_3d(
    phi_nodes: np.ndarray,
    ps: ParticleSolver,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Gather a 3D primal-node potential array at ``(x, y, z)`` with order-``p`` weights.

    Parameters
    ----------
    phi_nodes : numpy.ndarray
        Node potential array of shape ``(Nx, Ny, Nz)`` in [V].
    ps : ParticleSolver
        Attached particle solver.
    x, y, z : numpy.ndarray
        Particle coordinates in [m].

    Returns
    -------
    numpy.ndarray
        Gathered potential values at each particle position in [V].
    """
    solver = ps.solver
    shape = int(ps.shape)
    xi, eta, zeta = ps.grid_map.to_logical(x, y, z)
    ix0, wx = bspline_weights_1d(xi, shape, staggered=False, xp=np)
    iy0, wy = bspline_weights_1d(eta, shape, staggered=False, xp=np)
    iz0, wz = bspline_weights_1d(zeta, shape, staggered=False, xp=np)
    periodic = (
        bool(ps.boundaries.periodic[0]),
        bool(ps.boundaries.periodic[1]),
        False,
    )
    return _gather_component_3d(
        phi_nodes.ravel(order="F"),
        int(solver.Nx),
        int(solver.Ny),
        int(solver.Nz),
        ix0,
        wx,
        iy0,
        wy,
        iz0,
        wz,
        periodic,
        np,
    )


def build_axial_node_potential(
    ez_host: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    dz: float,
) -> np.ndarray:
    """Build node potential ``phi_axial(x_i, y_j, z_k) = -int_0^{z_k} E_z dz'``.

    Parameters
    ----------
    ez_host : numpy.ndarray
        Host ``z``-edge electric field array of shape ``(Nx, Ny, Nz)`` in [V/m].
    nx, ny, nz : int
        Physical cell counts along ``x``, ``y``, and ``z``.
    dz : float
        Axial cell size in [m].

    Returns
    -------
    numpy.ndarray
        Node potential array of shape ``(Nx, Ny, Nz)`` in [V] with
        ``phi_axial[:, :, 0] == 0.0``.
    """
    phi_axial = np.zeros_like(ez_host, dtype=np.float64, order="F")
    phi_axial[: nx + 1, :ny, 1 : nz + 1] = -np.cumsum(
        ez_host[: nx + 1, :ny, :nz] * dz, axis=2
    )
    return phi_axial


def build_transverse_node_potential(
    ex_host: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    *,
    subtract_wall_drift: bool = False,
) -> np.ndarray:
    """Build node potential ``phi_trans(x_i, y_j, z_k) = -int_{-a}^{x_i} E_x dx'``.

    Parameters
    ----------
    ex_host : numpy.ndarray
        Host ``x``-edge electric field array of shape ``(Nx, Ny, Nz)`` in [V/m].
    nx, ny, nz : int
        Physical cell counts along ``x``, ``y``, and ``z``.
    dx : float
        Transverse ``x`` cell size in [m].
    subtract_wall_drift : bool, optional
        If True, subtract a linear ramp in ``x`` so ``phi_trans[nx, :, :] == 0``
        at ``x = +a`` as in ``dc_decomp.py``. Default is ``False``.

    Returns
    -------
    numpy.ndarray
        Node potential array of shape ``(Nx, Ny, Nz)`` in [V] with
        ``phi_trans[0, :, :] == 0.0``.
    """
    phi_trans = np.zeros_like(ex_host, dtype=np.float64, order="F")
    phi_trans[1 : nx + 1, :ny, : nz + 1] = -np.cumsum(
        ex_host[:nx, :ny, : nz + 1] * dx, axis=0
    )
    if subtract_wall_drift:
        x_frac = np.linspace(0.0, 1.0, nx + 1, dtype=np.float64)[:, None, None]
        phi_trans[: nx + 1, :ny, : nz + 1] -= (
            x_frac * phi_trans[nx : nx + 1, :ny, : nz + 1]
        )
    return phi_trans


def analyze_static_potentials_and_curl(
    solver: SolverFIT3D,
    ps: ParticleSolver,
    nx: int,
    ny: int,
    nz: int,
    a: float,
    ly: float,
    lz: float,
    x_sheet: np.ndarray,
    z_window: tuple[float, float],
) -> dict:
    """Perform Part (a): compare ``phi_trans`` vs ``phi_axial``, curl E, and H.

    Parameters
    ----------
    solver : SolverFIT3D
        Electromagnetic solver instance.
    ps : ParticleSolver
        Attached particle solver.
    nx, ny, nz : int
        Physical cell counts along ``x``, ``y``, and ``z``.
    a, ly, lz : float
        Tunnel half-width, ``y`` period, and tunnel length in [m].
    x_sheet : numpy.ndarray
        Transverse coordinates of the sheet lattice lines in [m].
    z_window : tuple of float
        Interior axial analysis window ``(z_lo, z_hi)`` in [m].

    Returns
    -------
    dict
        Dictionary of Part (a) diagnostics.
    """
    dx = float(2.0 * a / nx)
    dy = float(ly / max(ny, 1))
    dz = float(lz / nz)
    shape = int(ps.shape)

    ex_host = _to_host(solver.E.to_matrix("x"))
    ey_host = _to_host(solver.E.to_matrix("y"))
    ez_host = _to_host(solver.E.to_matrix("z"))
    hx_host = _to_host(solver.H.to_matrix("x"))
    hy_host = _to_host(solver.H.to_matrix("y"))
    hz_host = _to_host(solver.H.to_matrix("z"))

    bc_y_low = str(solver.bc_low[1])
    bc_y_high = str(solver.bc_high[1])
    ny_solver = int(solver.Ny)
    print(
        f"Y_CONSISTENCY Ny={ny_solver} (ny_phys={ny}) "
        f"bc_y=({bc_y_low!r}, {bc_y_high!r})"
    )

    phi_axial_nodes = build_axial_node_potential(ez_host, nx, ny, nz, dz)
    phi_trans_nodes = build_transverse_node_potential(
        ex_host, nx, ny, nz, dx, subtract_wall_drift=False
    )
    phi_trans_drift_nodes = build_transverse_node_potential(
        ex_host, nx, ny, nz, dx, subtract_wall_drift=True
    )

    # y-direction consistency check
    max_abs_ey = float(np.max(np.abs(ey_host[: nx + 1, :ny, : nz + 1])))
    if ny > 1:
        phi_y_int = -np.cumsum(ey_host[: nx + 1, :ny, : nz + 1] * dy, axis=1)
        max_phi_y_variation = float(
            np.max(
                np.abs(
                    phi_axial_nodes[: nx + 1, :ny, : nz + 1]
                    - np.mean(
                        phi_axial_nodes[: nx + 1, :ny, : nz + 1],
                        axis=1,
                        keepdims=True,
                    )
                )
            )
        )
        max_phi_y_integral = float(np.max(np.abs(phi_y_int)))
    else:
        max_phi_y_variation = 0.0
        max_phi_y_integral = float(max_abs_ey * dy)

    z_lo, z_hi = float(z_window[0]), float(z_window[1])
    k_lo = max(1, int(np.floor(z_lo / dz)))
    k_hi = min(nz - 1, int(np.ceil(z_hi / dz)))
    z_int_nodes = np.asarray(solver.grid.z[k_lo : k_hi + 1], dtype=np.float64)
    n_k_int = int(z_int_nodes.size)

    # Transverse order-p B-spline weights for each lattice line x_l
    xi_lines = (x_sheet - (-a)) / dx
    ix_l, wx_l = bspline_weights_1d(xi_lines, shape, staggered=False, xp=np)
    n_lines = int(x_sheet.size)

    phi_trans_line = np.zeros((n_lines, n_k_int), dtype=np.float64)
    phi_trans_drift_line = np.zeros((n_lines, n_k_int), dtype=np.float64)
    phi_axial_line = np.zeros((n_lines, n_k_int), dtype=np.float64)
    phi_trans_gat3d = np.zeros((n_lines, n_k_int), dtype=np.float64)
    phi_axial_gat3d = np.zeros((n_lines, n_k_int), dtype=np.float64)

    for l_idx in range(n_lines):
        i_base = int(ix_l[l_idx])
        for m in range(shape + 1):
            w_m = float(wx_l[m][l_idx])
            i_nod = i_base + m
            if 0 <= i_nod <= nx:
                phi_trans_line[l_idx, :] += (
                    w_m * phi_trans_nodes[i_nod, 0, k_lo : k_hi + 1]
                )
                phi_trans_drift_line[l_idx, :] += (
                    w_m * phi_trans_drift_nodes[i_nod, 0, k_lo : k_hi + 1]
                )
                phi_axial_line[l_idx, :] += (
                    w_m * phi_axial_nodes[i_nod, 0, k_lo : k_hi + 1]
                )

        x_rep = np.full(n_k_int, float(x_sheet[l_idx]), dtype=np.float64)
        y_rep = np.zeros(n_k_int, dtype=np.float64)
        phi_trans_gat3d[l_idx, :] = gather_node_potential_3d(
            phi_trans_nodes, ps, x_rep, y_rep, z_int_nodes
        )
        phi_axial_gat3d[l_idx, :] = gather_node_potential_3d(
            phi_axial_nodes, ps, x_rep, y_rep, z_int_nodes
        )

    line_max_rel_diff = []
    line_mean_rel_diff = []
    line_max_rel_diff_3d = []
    line_mean_rel_diff_3d = []
    line_max_rel_diff_drift = []
    for l_idx in range(n_lines):
        denom = max(float(np.max(np.abs(phi_trans_line[l_idx, :]))), 1e-30)
        diff_nod = np.abs(phi_axial_line[l_idx, :] - phi_trans_line[l_idx, :])
        line_max_rel_diff.append(float(np.max(diff_nod) / denom))
        line_mean_rel_diff.append(float(np.mean(diff_nod) / denom))

        denom_3d = max(float(np.max(np.abs(phi_trans_gat3d[l_idx, :]))), 1e-30)
        diff_3d = np.abs(phi_axial_gat3d[l_idx, :] - phi_trans_gat3d[l_idx, :])
        line_max_rel_diff_3d.append(float(np.max(diff_3d) / denom_3d))
        line_mean_rel_diff_3d.append(float(np.mean(diff_3d) / denom_3d))

        diff_drift = np.abs(phi_axial_line[l_idx, :] - phi_trans_drift_line[l_idx, :])
        line_max_rel_diff_drift.append(float(np.max(diff_drift) / denom))

    # Discrete curl of E on interior cells [0, nx) x [0, ny) x [k_lo, k_hi)
    curl_y = (
        ex_host[:nx, :ny, k_lo + 1 : k_hi + 1] - ex_host[:nx, :ny, k_lo:k_hi]
    ) / dz - (ez_host[1 : nx + 1, :ny, k_lo:k_hi] - ez_host[:nx, :ny, k_lo:k_hi]) / dx

    if ny > 1:
        ez_next_y = np.roll(ez_host[:nx, :ny, k_lo:k_hi], -1, axis=1)
        ex_next_y = np.roll(ex_host[:nx, :ny, k_lo:k_hi], -1, axis=1)
        curl_x = (ez_next_y - ez_host[:nx, :ny, k_lo:k_hi]) / dy - (
            ey_host[:nx, :ny, k_lo + 1 : k_hi + 1] - ey_host[:nx, :ny, k_lo:k_hi]
        ) / dz
        curl_z = (
            ey_host[1 : nx + 1, :ny, k_lo:k_hi] - ey_host[:nx, :ny, k_lo:k_hi]
        ) / dx - (ex_next_y - ex_host[:nx, :ny, k_lo:k_hi]) / dy
    else:
        curl_x = (
            -(ey_host[:nx, :ny, k_lo + 1 : k_hi + 1] - ey_host[:nx, :ny, k_lo:k_hi])
            / dz
        )
        curl_z = (
            ey_host[1 : nx + 1, :ny, k_lo:k_hi] - ey_host[:nx, :ny, k_lo:k_hi]
        ) / dx

    curl_mag = np.sqrt(curl_x * curl_x + curl_y * curl_y + curl_z * curl_z)
    e_mag_int = np.sqrt(
        ex_host[:nx, :ny, k_lo:k_hi] ** 2
        + ey_host[:nx, :ny, k_lo:k_hi] ** 2
        + ez_host[:nx, :ny, k_lo:k_hi] ** 2
    )
    h_mag_int = np.sqrt(
        hx_host[:nx, :ny, k_lo:k_hi] ** 2
        + hy_host[:nx, :ny, k_lo:k_hi] ** 2
        + hz_host[:nx, :ny, k_lo:k_hi] ** 2
    )
    max_curl_e = float(np.max(curl_mag))
    max_e_int = max(float(np.max(e_mag_int)), 1e-30)
    rel_curl_e = float(max_curl_e / (max_e_int / dz))
    max_h_int = float(np.max(h_mag_int))

    # Node-level interior difference across all physical nodes
    node_denom = max(
        float(np.max(np.abs(phi_trans_nodes[: nx + 1, :ny, k_lo : k_hi + 1]))),
        1e-30,
    )
    node_max_rel_diff = float(
        np.max(
            np.abs(
                phi_axial_nodes[: nx + 1, :ny, k_lo : k_hi + 1]
                - phi_trans_nodes[: nx + 1, :ny, k_lo : k_hi + 1]
            )
        )
        / node_denom
    )

    return {
        "Ny": ny_solver,
        "ny_phys": int(ny),
        "bc_y": [bc_y_low, bc_y_high],
        "max_abs_ey": max_abs_ey,
        "max_phi_y_variation": max_phi_y_variation,
        "max_phi_y_integral": max_phi_y_integral,
        "node_max_rel_diff": node_max_rel_diff,
        "line_max_rel_diff": line_max_rel_diff,
        "line_mean_rel_diff": line_mean_rel_diff,
        "line_max_rel_diff_3d": line_max_rel_diff_3d,
        "line_mean_rel_diff_3d": line_mean_rel_diff_3d,
        "line_max_rel_diff_drift": line_max_rel_diff_drift,
        "overall_max_rel_diff_line": float(np.max(line_max_rel_diff)),
        "overall_mean_rel_diff_line": float(np.mean(line_mean_rel_diff)),
        "overall_max_rel_diff_3d": float(np.max(line_max_rel_diff_3d)),
        "overall_mean_rel_diff_3d": float(np.mean(line_mean_rel_diff_3d)),
        "max_curl_E": max_curl_e,
        "max_E_interior": max_e_int,
        "rel_curl_E_over_E_dz": rel_curl_e,
        "max_H_interior": max_h_int,
    }


def track_transit_identity(
    solver: SolverFIT3D,
    ps: ParticleSolver,
    beam: Species,
    nx: int,
    ny: int,
    nz: int,
    lz: float,
    dt: float,
    gamma0: float,
    x_sheet: np.ndarray,
    n_transit: int,
    z_eval_target: float,
    n_samples_per_line: int = 3,
) -> dict:
    """Perform Part (b): track ``n_samples_per_line`` particles per line for one transit.

    At each step :math:`m`, before ``solver.one_step()`` advances particles
    from :math:`\\mathbf{x}^m \\to \\mathbf{x}^{m+1}` using :math:`\\mathbf{E}^m`,
    accumulates

    .. math::

        S_p \\mathrel{+}= \\frac{q}{m c^2}
        \\bigl[\\Phi_{\\text{axial}}^m(\\mathbf{x}_p^m)
        - \\Phi_{\\text{axial}}^{m-1}(\\mathbf{x}_p^m)\\bigr]

    for every tracked particle with birth step :math:`n_b(p) < m` until it
    reaches ``z_eval_target`` in the interior.

    Parameters
    ----------
    solver : SolverFIT3D
        Settled electromagnetic solver.
    ps : ParticleSolver
        Attached particle solver.
    beam : Species
        Beam particle species with ``track_ids=True``.
    nx, ny, nz : int
        Physical cell counts along ``x``, ``y``, and ``z``.
    lz : float
        Tunnel length in [m].
    dt : float
        Time step size in [s].
    gamma0 : float
        Injected Lorentz factor at the emitter wall (dimensionless).
    x_sheet : numpy.ndarray
        Transverse sheet lattice line coordinates in [m].
    n_transit : int
        Number of steps in one beam transit across the tunnel.
    z_eval_target : float
        Interior axial position at which each tracked particle's identity is
        evaluated in [m].
    n_samples_per_line : int, optional
        Number of tracked particles per lattice line. Default is ``3``.

    Returns
    -------
    dict
        Summary and per-particle identity records.
    """
    dz = float(lz / nz)
    q = float(beam.charge)
    m_mass = float(beam.mass)
    q_over_mc2 = q / (m_mass * c_light * c_light)
    n_lines = int(x_sheet.size)

    # Space the 3 birth steps so all 3 batches reach z_eval_target within n_transit steps
    birth_stride = max(1, int(n_transit // max(4 * n_samples_per_line, 6)))
    birth_steps = [k * birth_stride for k in range(n_samples_per_line)]

    # Map pid -> tracking record
    tracked: dict[int, dict] = {}
    phi_prev_nodes: np.ndarray | None = None

    fn_pusher = get_pusher(ps.pusher)
    B_ext = (0.0, 0.0, float(s.GUIDE_BZ_TESLA))

    for step_rel in range(n_transit):
        ez_curr = _to_host(solver.E.to_matrix("z"))
        phi_curr_nodes = build_axial_node_potential(ez_curr, nx, ny, nz, dz)

        # Accumulate Phi^m(x^m) - Phi^{m-1}(x^m) for active tracked particles (m > n_b)
        active_pids = [
            pid
            for pid, rec in tracked.items()
            if not rec["completed"] and step_rel > rec["birth_step_rel"]
        ]
        if active_pids and phi_prev_nodes is not None:
            n_alive = int(beam.n_particles)
            ids_h = _to_host(beam.id[:n_alive]).astype(np.int64)
            x_h = _to_host(beam.x[:n_alive])
            y_h = _to_host(beam.y[:n_alive])
            z_h = _to_host(beam.z[:n_alive])
            ux_h = _to_host(beam.ux[:n_alive])
            uy_h = _to_host(beam.uy[:n_alive])
            uz_h = _to_host(beam.uz[:n_alive])

            id_to_slot = {int(ids_h[idx]): idx for idx in range(n_alive)}
            slots = [id_to_slot[pid] for pid in active_pids if pid in id_to_slot]
            valid_pids = [pid for pid in active_pids if pid in id_to_slot]

            if slots:
                slot_arr = np.asarray(slots, dtype=np.int64)
                x_sub = x_h[slot_arr]
                y_sub = y_h[slot_arr]
                z_sub = z_h[slot_arr]

                phi_m = gather_node_potential_3d(
                    phi_curr_nodes, ps, x_sub, y_sub, z_sub
                )
                phi_m_prev = gather_node_potential_3d(
                    phi_prev_nodes, ps, x_sub, y_sub, z_sub
                )
                d_phi = phi_m - phi_m_prev

                for k_p, pid in enumerate(valid_pids):
                    rec = tracked[pid]
                    rec["sum_dphi_over_mc2"] += float(q_over_mc2 * d_phi[k_p])
                    rec["n_accum_steps"] += 1
                    z_cur = float(z_sub[k_p])
                    if z_cur >= z_eval_target or step_rel == n_transit - 1:
                        slot_i = int(slot_arr[k_p])
                        u2_raw = float(
                            ux_h[slot_i] ** 2 + uy_h[slot_i] ** 2 + uz_h[slot_i] ** 2
                        )
                        gamma_raw = float(np.sqrt(1.0 + u2_raw / (c_light * c_light)))
                        phi_val = float(phi_m[k_p])
                        excess_raw = gamma_raw - gamma0 + q_over_mc2 * phi_val

                        # Synchronise u^{m-1/2} -> u^m by +0.5 * dt at x^m
                        xi_p, eta_p, zeta_p = ps.grid_map.to_logical(
                            x_sub[k_p : k_p + 1],
                            y_sub[k_p : k_p + 1],
                            z_sub[k_p : k_p + 1],
                        )
                        ex_h = _to_host(solver.E.to_matrix("x"))
                        ey_h = _to_host(solver.E.to_matrix("y"))
                        hx_h = _to_host(solver.H.to_matrix("x"))
                        hy_h = _to_host(solver.H.to_matrix("y"))
                        hz_h = _to_host(solver.H.to_matrix("z"))
                        E_tup_h = (
                            ex_h.ravel(order="F"),
                            ey_h.ravel(order="F"),
                            ez_curr.ravel(order="F"),
                        )
                        from scipy.constants import mu_0

                        B_tup_h = (
                            (mu_0 * hx_h).ravel(order="F"),
                            (mu_0 * hy_h).ravel(order="F"),
                            (mu_0 * hz_h).ravel(order="F"),
                        )
                        Ex_g, Ey_g, Ez_g, Bx_g, By_g, Bz_g = gather_fields(
                            xi_p,
                            eta_p,
                            zeta_p,
                            E_tup_h,
                            B_tup_h,
                            int(solver.Nx),
                            int(solver.Ny),
                            int(solver.Nz),
                            order=int(ps.shape),
                            gather=ps.gather,
                            periodic=ps.boundaries.periodic,
                            xp=np,
                        )
                        ux_s, uy_s, uz_s = fn_pusher(
                            ux_h[slot_i : slot_i + 1],
                            uy_h[slot_i : slot_i + 1],
                            uz_h[slot_i : slot_i + 1],
                            Ex_g,
                            Ey_g,
                            Ez_g,
                            Bx_g + B_ext[0],
                            By_g + B_ext[1],
                            Bz_g + B_ext[2],
                            q,
                            m_mass,
                            0.5 * dt,
                            xp=np,
                        )
                        u2_sync = float(ux_s[0] ** 2 + uy_s[0] ** 2 + uz_s[0] ** 2)
                        gamma_sync = float(np.sqrt(1.0 + u2_sync / (c_light * c_light)))
                        excess_sync = gamma_sync - gamma0 + q_over_mc2 * phi_val

                        rec["z_eval"] = z_cur
                        rec["x_eval"] = float(x_sub[k_p])
                        rec["phi_axial_eval"] = phi_val
                        rec["gamma_excess_raw"] = excess_raw
                        rec["gamma_excess_sync"] = excess_sync
                        rec["residual_raw"] = excess_raw - rec["sum_dphi_over_mc2"]
                        rec["residual_sync"] = excess_sync - rec["sum_dphi_over_mc2"]
                        rec["completed"] = True

        solver.one_step()
        n_after = int(beam.n_particles)

        if step_rel in birth_steps and n_after >= n_lines:
            sample_idx = birth_steps.index(step_rel)
            # Newly injected batch of n_lines particles lies at the tail of beam
            ids_tail = _to_host(beam.id[n_after - n_lines : n_after]).astype(np.int64)
            x_tail = _to_host(beam.x[n_after - n_lines : n_after])
            for k_l in range(n_lines):
                pid = int(ids_tail[k_l])
                line_idx = int(np.argmin(np.abs(x_sheet - float(x_tail[k_l]))))
                tracked[pid] = {
                    "pid": pid,
                    "line_idx": line_idx,
                    "sample_idx": int(sample_idx),
                    "x_line": float(x_sheet[line_idx]),
                    "birth_step_rel": int(step_rel),
                    "n_accum_steps": 0,
                    "sum_dphi_over_mc2": 0.0,
                    "z_eval": 0.0,
                    "x_eval": 0.0,
                    "phi_axial_eval": 0.0,
                    "gamma_excess_raw": 0.0,
                    "gamma_excess_sync": 0.0,
                    "residual_raw": 0.0,
                    "residual_sync": 0.0,
                    "completed": False,
                }

        phi_prev_nodes = phi_curr_nodes

    records_list = sorted(
        tracked.values(), key=lambda r: (r["line_idx"], r["sample_idx"])
    )
    for rec in records_list:
        print(
            f"  TRACK line={rec['line_idx']:2d} sample={rec['sample_idx']} "
            f"x={rec['x_line'] * 1e3:+6.2f}mm z_eval={rec['z_eval']:.4f}m "
            f"sum_dPhi/mc2={rec['sum_dphi_over_mc2']:+.6e} "
            f"excess_raw={rec['gamma_excess_raw']:+.6e} "
            f"excess_sync={rec['gamma_excess_sync']:+.6e} "
            f"residual_sync={rec['residual_sync']:+.6e}"
        )

    per_line_mean_sum_dphi = []
    per_line_mean_excess_raw = []
    per_line_mean_excess_sync = []
    for l_idx in range(n_lines):
        line_recs = [
            r for r in records_list if r["line_idx"] == l_idx and r["completed"]
        ]
        if line_recs:
            per_line_mean_sum_dphi.append(
                float(np.mean([r["sum_dphi_over_mc2"] for r in line_recs]))
            )
            per_line_mean_excess_raw.append(
                float(np.mean([r["gamma_excess_raw"] for r in line_recs]))
            )
            per_line_mean_excess_sync.append(
                float(np.mean([r["gamma_excess_sync"] for r in line_recs]))
            )

    return {
        "n_tracked_particles": len(records_list),
        "n_completed_particles": sum(1 for r in records_list if r["completed"]),
        "per_line_mean_sum_dphi_over_mc2": per_line_mean_sum_dphi,
        "per_line_mean_gamma_excess_raw": per_line_mean_excess_raw,
        "per_line_mean_gamma_excess_sync": per_line_mean_excess_sync,
        "overall_mean_sum_dphi_over_mc2": float(np.mean(per_line_mean_sum_dphi)),
        "overall_mean_gamma_excess_raw": float(np.mean(per_line_mean_excess_raw)),
        "overall_mean_gamma_excess_sync": float(np.mean(per_line_mean_excess_sync)),
    }


def run_synthetic_self_check() -> dict:
    """Run the fast synthetic curl-free self-check and mutation check on a tiny grid.

    Verifies that for an exact discrete gradient field :math:`\\mathbf{E} = -\\nabla_h \\phi`
    vanishing on all PEC walls:
    1. ``phi_axial == phi_trans`` to ``< 1e-12`` relative on nodes and when gathered
       with order-``p`` B-spline weights at every lattice line,
    2. the discrete interior curl satisfies ``max |curl E| / (max |E| / dz) < 1e-12``,
    3. perturbing a single interior ``E_z`` edge breaks curl-freeness and causes the
       ``1e-12`` relative path-independence check to fail (mutation check).

    Returns
    -------
    dict
        Self-check diagnostic dictionary.
    """
    nx, ny, nz = 6, 1, 16
    a = float(s.TUNNEL_HALF_WIDTH_M)
    b = float(s.BEAM_HALF_THICKNESS_M)
    ly = float(s.PERIODIC_LY_M)
    lz = 0.04
    cfln = float(s.CFL_NUMBER)
    shape = int(s.PARTICLE_SHAPE)

    grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, ny, nz, verbose=0)
    dt, _ = s._integer_period_dt(grid, cfln=cfln)
    solver = SolverFIT3D(
        grid,
        cfln=cfln,
        dt=dt,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=False,
        verbose=0,
    )
    sp = Species(name="beam", charge=-e_charge, mass=m_e, capacity=16, track_ids=True)
    ps = ParticleSolver(
        solver,
        [sp],
        B_ext=(0.0, 0.0, float(s.GUIDE_BZ_TESLA)),
        shape=shape,
        gather="galerkin",
        track_energy=False,
    ).attach()

    x_nodes = np.asarray(solver.grid.x[: nx + 1], dtype=np.float64)
    z_nodes = np.asarray(solver.grid.z[: nz + 1], dtype=np.float64)
    phi_x = -200.0 * (1.0 - (x_nodes / a) ** 2)
    phi_x[0] = 0.0
    phi_x[-1] = 0.0
    z_prof = np.sin(np.pi * z_nodes / lz)
    z_prof[0] = 0.0
    z_prof[-1] = 0.0

    phi_exact = np.zeros(
        (int(solver.Nx), int(solver.Ny), int(solver.Nz)),
        dtype=np.float64,
        order="F",
    )
    phi_exact[: nx + 1, :ny, : nz + 1] = phi_x[:, None, None] * z_prof[None, None, :]

    dx = float(2.0 * a / nx)
    dz = float(lz / nz)
    ex_mat = np.zeros_like(phi_exact)
    ez_mat = np.zeros_like(phi_exact)
    ex_mat[:nx, :ny, : nz + 1] = (
        -(phi_exact[1 : nx + 1, :ny, : nz + 1] - phi_exact[:nx, :ny, : nz + 1]) / dx
    )
    ez_mat[: nx + 1, :ny, :nz] = (
        -(phi_exact[: nx + 1, :ny, 1 : nz + 1] - phi_exact[: nx + 1, :ny, :nz]) / dz
    )

    solver.E.from_matrix(ex_mat, "x")
    solver.E.from_matrix(np.zeros_like(ex_mat), "y")
    solver.E.from_matrix(ez_mat, "z")

    n_lines = 4
    k_idx = np.arange(n_lines, dtype=np.float64)
    x_sheet = b * (2.0 * (k_idx + 0.5) / float(n_lines) - 1.0)
    z_window = (0.25 * lz, 0.75 * lz)

    res_clean = analyze_static_potentials_and_curl(
        solver, ps, nx, ny, nz, a, ly, lz, x_sheet, z_window
    )
    max_rel_clean = float(
        max(
            res_clean["node_max_rel_diff"],
            res_clean["overall_max_rel_diff_line"],
            res_clean["overall_max_rel_diff_3d"],
        )
    )
    if max_rel_clean >= 1e-12:
        raise AssertionError(
            f"Synthetic curl-free self-check failed: max_rel_diff={max_rel_clean:.3e} >= 1e-12"
        )
    if res_clean["rel_curl_E_over_E_dz"] >= 1e-12:
        raise AssertionError(
            f"Synthetic curl-free curl(E) check failed: "
            f"rel_curl={res_clean['rel_curl_E_over_E_dz']:.3e} >= 1e-12"
        )

    # Mutation check: perturb one interior z-edge so curl E != 0 and verify failure
    ez_mut = ez_mat.copy()
    ez_mut[nx // 2, 0, nz // 3] += 10.0 / dz
    solver.E.from_matrix(ez_mut, "z")
    res_mut = analyze_static_potentials_and_curl(
        solver, ps, nx, ny, nz, a, ly, lz, x_sheet, z_window
    )
    max_rel_mut = float(res_mut["overall_max_rel_diff_line"])
    if max_rel_mut < 1e-12:
        raise AssertionError(
            f"Mutation check failed to detect non-curl-free perturbation: "
            f"max_rel_mut={max_rel_mut:.3e}"
        )

    # Verify Part (b) tracking on the tiny grid with a linearly ramping potential
    # phi^m = (1 + alpha_ramp * m) * phi_exact so sum_{m > n_b} [Phi^m - Phi^{m-1}]
    # is non-zero and exercises the full trajectory identity pipeline.
    params = s.compute_analytic_parameters(qm_scale=1.0)
    gamma0 = float(params["gamma0"])
    beta0 = float(np.sqrt(1.0 - 1.0 / (gamma0 * gamma0)))
    n_lines = 4
    solver.E.from_matrix(ex_mat, "x")
    solver.E.from_matrix(np.zeros_like(ex_mat), "y")
    solver.E.from_matrix(ez_mat, "z")
    solver.one_step = solver._one_step
    solver.step_0 = False
    solver._update_H = lambda: None
    alpha_ramp = 2.0e-3
    step_counter = [0]

    def _ramp_update_e() -> None:
        step_counter[0] += 1
        scale = 1.0 + alpha_ramp * float(step_counter[0])
        solver.E.from_matrix(ex_mat * scale, "x")
        solver.E.from_matrix(ez_mat * scale, "z")

    solver._update_E = _ramp_update_e
    solver._mid_step_fn = None
    solver._particle_solver = None

    beam_tiny = Species(
        name="beam",
        charge=-e_charge,
        mass=m_e,
        capacity=1024,
        track_ids=True,
    )
    inj_tiny = BeamInjector(
        species=beam_tiny,
        z0=0.0,
        radius=b,
        current=float(params["I0"]),
        gamma=gamma0,
        n_macro_per_step=n_lines,
        profile="sheet_lattice",
        seed=1,
    )
    ps_tiny = ParticleSolver(
        solver,
        [beam_tiny],
        injectors=[inj_tiny],
        B_ext=(0.0, 0.0, float(s.GUIDE_BZ_TESLA)),
        shape=shape,
        gather="galerkin",
        track_energy=False,
    ).attach()

    n_transit_tiny = int(np.ceil(lz / (beta0 * c_light * dt)))
    part_b_tiny = track_transit_identity(
        solver,
        ps_tiny,
        beam_tiny,
        nx,
        ny,
        nz,
        lz,
        dt,
        gamma0,
        x_sheet,
        n_transit_tiny,
        z_eval_target=0.5 * lz,
        n_samples_per_line=3,
    )
    if part_b_tiny["n_completed_particles"] != 3 * n_lines:
        raise AssertionError(
            f"Expected {3 * n_lines} completed tracked particles in tiny self-check, "
            f"got {part_b_tiny['n_completed_particles']}"
        )

    summary = {
        "status": "PASS",
        "tiny_grid": [nx, ny, nz],
        "Ny": res_clean["Ny"],
        "bc_y": res_clean["bc_y"],
        "clean_node_max_rel_diff": res_clean["node_max_rel_diff"],
        "clean_line_max_rel_diff": res_clean["overall_max_rel_diff_line"],
        "clean_3d_max_rel_diff": res_clean["overall_max_rel_diff_3d"],
        "clean_rel_curl_E": res_clean["rel_curl_E_over_E_dz"],
        "mutated_line_max_rel_diff": max_rel_mut,
        "mutated_rel_curl_E": res_mut["rel_curl_E_over_E_dz"],
        "tiny_tracked_completed": part_b_tiny["n_completed_particles"],
        "tiny_mean_sum_dphi_over_mc2": part_b_tiny["overall_mean_sum_dphi_over_mc2"],
        "tiny_mean_gamma_excess_sync": part_b_tiny["overall_mean_gamma_excess_sync"],
    }
    print(f"SYNTHETIC_SELF_CHECK {json.dumps(summary, sort_keys=True)}")
    return summary


def run_production_grid(
    grid_name: str,
    *,
    use_gpu: bool,
    settle_transits: float = 3.0,
) -> dict:
    """Run Part (a) and Part (b) on a settled ``m = 0`` beam for ``grid_name``.

    Parameters
    ----------
    grid_name : {"coarse", "fine"}
        Target resolution preset.
    use_gpu : bool
        Whether to run on GPU via CuPy.
    settle_transits : float, optional
        Number of transits to settle before Part (a) and Part (b). Default ``3.0``.

    Returns
    -------
    dict
        Combined ``PATH`` result dictionary.
    """
    nx, ny, nz = GRID_MAP[grid_name]
    params = s.compute_analytic_parameters(qm_scale=1.0)
    gamma0 = float(params["gamma0"])
    i0_dc = float(params["I0"])
    u0 = float(params["u0"])
    a = float(s.TUNNEL_HALF_WIDTH_M)
    b = float(s.BEAM_HALF_THICKNESS_M)
    ly = float(s.PERIODIC_LY_M)
    lz = float(s.TUNNEL_LENGTH_M)
    cfln = float(s.CFL_NUMBER)
    shape = int(s.PARTICLE_SHAPE)
    n_macro = 16 if nx == s.COARSE_GRID[0] else 32

    det = s.compute_discrete_dc_depression(
        nx, params, weighting="charge", return_details=True, shape=shape
    )
    x_sheet = np.asarray(det["x_sheet"], dtype=np.float64)

    grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, ny, nz, verbose=0)
    dt, _ = s._integer_period_dt(grid, cfln=cfln)
    solver = SolverFIT3D(
        grid,
        cfln=cfln,
        dt=dt,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=use_gpu,
        verbose=0,
    )

    n_transit = int(np.ceil(lz / (u0 * dt)))
    n_settle_default = int(s.settle_steps("particles", nx, nz, params, dt, shape=shape))
    n_settle = (
        n_settle_default
        if settle_transits is None
        else max(n_settle_default, round(float(settle_transits) * n_transit))
    )
    cap = int((n_settle + 2 * n_transit + 64) * n_macro)

    beam = Species(
        name="electron",
        charge=-e_charge,
        mass=m_e,
        capacity=cap,
        track_ids=True,
    )
    inj = BeamInjector(
        species=beam,
        z0=0.0,
        radius=b,
        current=i0_dc,
        gamma=gamma0,
        n_macro_per_step=n_macro,
        profile="sheet_lattice",
        seed=42,
        rise_time=0.0,
        velocity_modulation=None,
    )
    ps = ParticleSolver(
        solver,
        [beam],
        injectors=[inj],
        B_ext=(0.0, 0.0, float(s.GUIDE_BZ_TESLA)),
        shape=shape,
        filter=s.PARTICLE_FILTER,
        gather="galerkin",
        track_energy=False,
    ).attach()

    for _ in range(n_settle):
        solver.one_step()

    z_window = (
        float(s.FIT_WINDOW_FRAC[0] * lz),
        float(s.FIT_WINDOW_FRAC[1] * lz),
    )
    part_a = analyze_static_potentials_and_curl(
        solver, ps, nx, ny, nz, a, ly, lz, x_sheet, z_window
    )
    part_b = track_transit_identity(
        solver,
        ps,
        beam,
        nx,
        ny,
        nz,
        lz,
        dt,
        gamma0,
        x_sheet,
        n_transit,
        z_eval_target=0.5 * lz,
        n_samples_per_line=3,
    )

    out = {
        "grid": grid_name,
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "dt": float(dt),
        "n_settle": int(n_settle),
        "n_transit": int(n_transit),
        "use_gpu": bool(use_gpu),
        "part_a": part_a,
        "part_b": part_b,
    }
    print(f"PATH {json.dumps(out, sort_keys=True)}")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``path_identity.py``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        choices=("coarse", "fine", "both"),
        default="coarse",
        help="Grid preset to run (default: coarse).",
    )
    parser.add_argument(
        "--self-check-only",
        action="store_true",
        help="Run only the fast synthetic curl-free self-check and exit.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even if a GPU is available.",
    )
    parser.add_argument(
        "--settle-transits",
        type=float,
        default=3.0,
        help="Number of transits to settle before analysis (default: 3.0).",
    )
    args = parser.parse_args(argv)

    run_synthetic_self_check()
    if args.self_check_only:
        return 0

    use_gpu = False if args.cpu else bool(s.USE_GPU)
    grids = ("coarse", "fine") if args.grid == "both" else (args.grid,)
    for g_name in grids:
        run_production_grid(
            g_name,
            use_gpu=use_gpu,
            settle_transits=float(args.settle_transits),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
