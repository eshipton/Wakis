"""Single-particle frozen-potential study for wall-born energy excess (Part B).

Builds the same coarse grid, ``SolverFIT3D``, and ``ParticleSolver`` setup as
``run_particle_beam_drift`` (``nx=10, ny=1, nz=474``, ``shape=2``,
``gather="galerkin"``), but injects a single macroparticle of unit weight
(one physical electron, and with the field update frozen so the particle never
perturbs ``E`` or ``H``) from the ``z = 0`` PEC wall at an interior sheet
lattice line ``x_l``.

The electric field is set from a prescribed discrete node potential

.. math::

    \\phi(x_i, z_k) = \\phi_x(x_i)\\,\\bigl(1 - e^{-z_k / L}\\bigr)

vanishing on all PEC walls (``x = \\pm a``, ``z = 0``, and ``z = L_z``), with
:math:`\\mathbf{E} = -\\nabla_h \\phi` on the primal Yee edges and
:math:`E_y = 0`. The field advance is frozen by replacing ``solver._update_H``
and ``solver._update_E`` with no-ops while keeping ``solver.one_step =
solver._one_step`` so ``ParticleSolver._mid_step_hook`` executes normally.

For ``dt``, ``dt / 2``, and ``dt / 4``, the particle is pushed past ``z > 5 L``
and its relativistic energy invariant

.. math::

    \\Delta \\gamma = \\gamma - \\gamma_0
    + \\frac{q}{m c^2}\\,\\Phi_{\\text{gathered}}(x, z)

is evaluated with the same order-:math:`p` nodal B-spline weights used by the
PEC-wall birth kick, both with the birth kick enabled and disabled.
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

import wakis.particles.solver as psolver_mod
from wakis import GridFIT3D, SolverFIT3D
from wakis.particles import BeamInjector, ParticleSolver, Species
from wakis.particles.pushers import get_pusher
from wakis.particles.shapes import (
    _gather_component_3d,
    bspline_weights_1d,
    gather_fields,
)


WALL_STENCIL = "masked"  # ParticleSolver wall_stencil (CLI --wall-stencil)


def gather_node_scalar(
    node_arr: np.ndarray,
    ps: ParticleSolver,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Gather a 3D node-centred scalar field at ``(x, y, z)`` with order-``p`` B-splines.

    Uses the exact nodal B-spline weights ``bspline_weights_1d(..., staggered=False)``
    and non-periodic wall masking ``periodic=(False, True, False)`` employed by
    :meth:`ParticleSolver._apply_pec_wall_birth_kick`.

    Parameters
    ----------
    node_arr : numpy.ndarray
        Node array of shape ``(Nx, Ny, Nz)`` in [V].
    ps : ParticleSolver
        Attached particle solver providing ``grid_map`` and ``shape``.
    x, y, z : numpy.ndarray
        Particle coordinates in [m].

    Returns
    -------
    numpy.ndarray
        Gathered scalar values at each particle position in [V].
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
    flat = node_arr.ravel(order="F")
    out = _gather_component_3d(
        flat,
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
    wi = getattr(ps, "_wall_images", None)
    if wi is not None:
        # wall_stencil="image": odd potential images (sign (-1)^{|S|}).
        for S, idx, wall in wi.image_sets(([xi], [eta], [zeta]), xp=np):
            m = wi._mirror((xi, eta, zeta), S, idx, wall)
            jx, vx = bspline_weights_1d(m[0], shape, staggered=False, xp=np)
            jy, vy = bspline_weights_1d(m[1], shape, staggered=False, xp=np)
            jz, vz = bspline_weights_1d(m[2], shape, staggered=False, xp=np)
            out[idx] += ((-1.0) ** len(S)) * _gather_component_3d(
                flat, int(solver.Nx), int(solver.Ny), int(solver.Nz),
                jx, vx, jy, vy, jz, vz, periodic, np,
            )
    return out


def build_prescribed_potential_and_edges(
    solver: SolverFIT3D,
    nx: int,
    nz: int,
    a: float,
    lz: float,
    phi_min: float,
    L_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct nodal potential ``phi(x, z)`` and Yee edge fields ``E = -grad phi``.

    Parameters
    ----------
    solver : SolverFIT3D
        Electromagnetic solver on the ``(nx, 1, nz)`` grid.
    nx, nz : int
        Number of physical cells along ``x`` and ``z``.
    a : float
        Tunnel half-width in [m].
    lz : float
        Tunnel length in [m].
    phi_min : float
        Peak potential at ``x = 0`` in [V] (negative for electron space-charge well).
    L_decay : float
        Axial sheath decay length :math:`L` in [m].

    Returns
    -------
    phi_nodes : numpy.ndarray
        Node potential array of shape ``(Nx, Ny, Nz)`` in [V].
    ex_mat : numpy.ndarray
        Primal ``x``-edge electric field array of shape ``(Nx, Ny, Nz)`` in [V/m].
    ez_mat : numpy.ndarray
        Primal ``z``-edge electric field array of shape ``(Nx, Ny, Nz)`` in [V/m].
    """
    x_nodes = np.asarray(solver.grid.x[: nx + 1], dtype=np.float64)
    z_nodes = np.asarray(solver.grid.z[: nz + 1], dtype=np.float64)

    phi_x = phi_min * (1.0 - (x_nodes / a) ** 2)
    phi_x[0] = 0.0
    phi_x[-1] = 0.0

    z_prof = 1.0 - np.exp(-z_nodes / L_decay)
    z_prof[0] = 0.0
    z_prof[-1] = 0.0

    phi_nodes = np.zeros(
        (int(solver.Nx), int(solver.Ny), int(solver.Nz)),
        dtype=np.float64,
        order="F",
    )
    phi_nodes[: nx + 1, 0, : nz + 1] = phi_x[:, None] * z_prof[None, :]

    dx = float(2.0 * a / nx)
    dz = float(lz / nz)
    ex_mat = np.zeros_like(phi_nodes)
    ez_mat = np.zeros_like(phi_nodes)
    ex_mat[:nx, 0, : nz + 1] = (
        -(phi_nodes[1 : nx + 1, 0, : nz + 1] - phi_nodes[:nx, 0, : nz + 1]) / dx
    )
    ez_mat[: nx + 1, 0, :nz] = (
        -(phi_nodes[: nx + 1, 0, 1 : nz + 1] - phi_nodes[: nx + 1, 0, :nz]) / dz
    )
    return phi_nodes, ex_mat, ez_mat


def compute_predicted_birth_kick(
    ps: ParticleSolver,
    ez_mat: np.ndarray,
    dz: float,
    x_inj: float,
    q: float,
    m: float,
    w: float = 1.0,
) -> tuple[float, float, float]:
    """Compute predicted birth kick ``sum_k W_k Psi_k`` at ``(x_inj, 0, 0)``.

    Parameters
    ----------
    ps : ParticleSolver
        Attached particle solver.
    ez_mat : numpy.ndarray
        Primal ``z``-edge electric field array of shape ``(Nx, Ny, Nz)`` in [V/m].
    dz : float
        Axial cell size in [m].
    x_inj : float
        Transverse birth coordinate in [m].
    q : float
        Particle charge in [C].
    m : float
        Particle rest mass in [kg].
    w : float, optional
        Macroparticle weight (dimensionless). Default is ``1.0``.

    Returns
    -------
    psi_birth : float
        Gathered wall-normal potential integral :math:`\\sum_k W_k \\Psi_k` in [V].
    dgamma_pred : float
        Predicted Lorentz factor increment :math:`(q / m c^2) \\sum_k W_k \\Psi_k`
        (dimensionless).
    dK_pred_J : float
        Predicted kinetic energy increment :math:`q w \\sum_k W_k \\Psi_k` in [J].
    """
    solver = ps.solver
    shape = int(ps.shape)
    n_slab = min(shape + 2, int(solver.Nz))
    psi_slab = np.zeros(
        (int(solver.Nx), int(solver.Ny), n_slab), dtype=np.float64, order="F"
    )
    np.cumsum(ez_mat[:, :, : n_slab - 1] * dz, axis=2, out=psi_slab[:, :, 1:n_slab])

    xi_b, eta_b, zeta_b = ps.grid_map.to_logical(
        np.array([x_inj], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )
    ix0, wx = bspline_weights_1d(xi_b, shape, staggered=False, xp=np)
    iy0, wy = bspline_weights_1d(eta_b, shape, staggered=False, xp=np)
    iz0, wz = bspline_weights_1d(zeta_b, shape, staggered=False, xp=np)
    psi_birth = float(
        _gather_component_3d(
            psi_slab.ravel(order="F"),
            int(solver.Nx),
            int(solver.Ny),
            n_slab,
            ix0,
            wx,
            iy0,
            wy,
            iz0,
            wz,
            (False, True, False),
            np,
        )[0]
    )
    dK_pred_J = float(q * w * psi_birth)
    dgamma_pred = float((q * psi_birth) / (m * c_light * c_light))
    return psi_birth, dgamma_pred, dK_pred_J


def eval_particle_excess(
    sp: Species,
    ps: ParticleSolver,
    phi_nodes: np.ndarray,
    B_zero: tuple[np.ndarray, np.ndarray, np.ndarray],
    B_ext: tuple[float, float, float],
    gamma0: float,
    dt: float,
) -> tuple[float, float, float, float]:
    """Evaluate raw and time-synchronised energy excess at current particle state.

    Parameters
    ----------
    sp : Species
        Particle species containing 1 particle.
    ps : ParticleSolver
        Attached particle solver.
    phi_nodes : numpy.ndarray
        Nodal potential array of shape ``(Nx, Ny, Nz)`` in [V].
    B_zero : tuple of numpy.ndarray
        Zero magnetic field arrays for ``gather_fields``.
    B_ext : tuple of float
        Uniform external magnetic field ``(Bx, By, Bz)`` in [T].
    gamma0 : float
        Injected Lorentz factor at the wall (dimensionless).
    dt : float
        Time step size in [s].

    Returns
    -------
    excess_raw : float
        Unsynchronised excess :math:`\\gamma(t_{n-1/2}) - \\gamma_0 + (q/mc^2)\\Phi(x^n)`.
    excess_sync : float
        Synchronised excess :math:`\\gamma(t_n) - \\gamma_0 + (q/mc^2)\\Phi(x^n)` after
        advancing :math:`\\mathbf{u}^{n-1/2} \\to \\mathbf{u}^n` by ``+0.5 * dt`` at
        :math:`\\mathbf{x}^n`.
    phi_gat : float
        Gathered potential :math:`\\Phi_{\\text{gathered}}(\\mathbf{x}^n)` in [V].
    zeta_cur : float
        Logical axial coordinate :math:`\\zeta = z / \\Delta z` (dimensionless).
    """
    solver = ps.solver
    shape = int(ps.shape)
    x_arr = sp.x[:1]
    y_arr = sp.y[:1]
    z_arr = sp.z[:1]
    xi, eta, zeta = ps.grid_map.to_logical(x_arr, y_arr, z_arr)
    phi_gat = float(gather_node_scalar(phi_nodes, ps, x_arr, y_arr, z_arr)[0])

    q = float(sp.charge)
    m = float(sp.mass)
    q_over_mc2 = q / (m * c_light * c_light)

    u2_raw = float(sp.ux[0] ** 2 + sp.uy[0] ** 2 + sp.uz[0] ** 2)
    gamma_raw = float(np.sqrt(1.0 + u2_raw / (c_light * c_light)))
    excess_raw = gamma_raw - gamma0 + q_over_mc2 * phi_gat

    Ex_g, Ey_g, Ez_g, Bx_g, By_g, Bz_g = gather_fields(
        xi,
        eta,
        zeta,
        solver.E,
        B_zero,
        int(solver.Nx),
        int(solver.Ny),
        int(solver.Nz),
        order=shape,
        gather=ps.gather,
        periodic=ps.boundaries.periodic,
        xp=np,
    )
    corr = ps._image_field(xi, eta, zeta, solver.E, B_zero)
    if corr is not None:
        Ex_g, Ey_g, Ez_g = Ex_g + corr[0], Ey_g + corr[1], Ez_g + corr[2]
        Bx_g, By_g, Bz_g = Bx_g + corr[3], By_g + corr[4], Bz_g + corr[5]
    fn = get_pusher(ps.pusher)
    ux_s, uy_s, uz_s = fn(
        sp.ux[:1],
        sp.uy[:1],
        sp.uz[:1],
        Ex_g + float(B_ext[0]) * 0.0,
        Ey_g + float(B_ext[1]) * 0.0,
        Ez_g,
        Bx_g + float(B_ext[0]),
        By_g + float(B_ext[1]),
        Bz_g + float(B_ext[2]),
        q,
        m,
        0.5 * dt,
        xp=np,
    )
    u2_sync = float(ux_s[0] ** 2 + uy_s[0] ** 2 + uz_s[0] ** 2)
    gamma_sync = float(np.sqrt(1.0 + u2_sync / (c_light * c_light)))
    excess_sync = gamma_sync - gamma0 + q_over_mc2 * phi_gat
    return excess_raw, excess_sync, phi_gat, float(zeta[0])


def run_single_particle_case(
    solver: SolverFIT3D,
    phi_nodes: np.ndarray,
    ex_mat: np.ndarray,
    ez_mat: np.ndarray,
    dt_base: float,
    dt_div: int,
    x_inj: float,
    line_idx: int,
    gamma0: float,
    L_decay: float,
    n_base_steps: int,
    enable_kick: bool,
) -> dict:
    """Run one single-particle trajectory at ``dt = dt_base / dt_div``.

    Parameters
    ----------
    solver : SolverFIT3D
        Frozen-field electromagnetic solver on the coarse grid.
    phi_nodes, ex_mat, ez_mat : numpy.ndarray
        Prescribed nodal potential and edge electric field arrays.
    dt_base : float
        Coarse nominal time step in [s].
    dt_div : int
        Time step subdivision factor (``1``, ``2``, ``4``, ...).
    x_inj : float
        Transverse birth position in [m].
    line_idx : int
        Sheet lattice line index.
    gamma0 : float
        Injection Lorentz factor at the emitter wall (dimensionless).
    L_decay : float
        Axial sheath decay scale :math:`L` in [m].
    n_base_steps : int
        Number of nominal ``dt_base`` steps to integrate (total steps
        ``n_base_steps * dt_div``).
    enable_kick : bool
        Whether ``wakis.particles.solver._ENABLE_PEC_WALL_KICK`` is enabled.

    Returns
    -------
    dict
        Trajectory diagnostic dictionary.
    """
    dt = float(dt_base / dt_div)
    solver.dt = dt
    solver._mid_step_fn = None
    solver._particle_solver = None

    prev_kick_flag = psolver_mod._ENABLE_PEC_WALL_KICK
    psolver_mod._ENABLE_PEC_WALL_KICK = bool(enable_kick)
    try:
        sp = Species(
            name="beam",
            charge=-e_charge,
            mass=m_e,
            capacity=16,
            track_ids=True,
        )
        # current = e_charge / dt emits 1 macroparticle of weight w = 1.0 per step
        inj = BeamInjector(
            species=sp,
            x0=x_inj,
            y0=0.0,
            z0=0.0,
            radius=0.0,
            current=e_charge / dt,
            gamma=gamma0,
            n_macro_per_step=1,
            profile="lattice",
            seed=1,
        )
        B_ext = (0.0, 0.0, float(s.GUIDE_BZ_TESLA))
        ps = ParticleSolver(
            solver,
            [sp],
            injectors=[inj],
            B_ext=B_ext,
            shape=int(s.PARTICLE_SHAPE),
            gather="galerkin",
            track_energy=True,
            wall_stencil=WALL_STENCIL,
        ).attach()

        dz = float(s.TUNNEL_LENGTH_M / s.COARSE_GRID[2])
        psi_birth, dgamma_kick_pred, dK_kick_pred_J = compute_predicted_birth_kick(
            ps, ez_mat, dz, x_inj, float(sp.charge), float(sp.mass), w=1.0
        )
        if WALL_STENCIL == "image":
            # Image stencil: no birth kick (the odd image makes sum_k W_k Psi_k vanish).
            psi_birth, dgamma_kick_pred, dK_kick_pred_J = 0.0, 0.0, 0.0

        # Verify birth state at z = 0 before any push
        phi_birth_gat = float(
            gather_node_scalar(
                phi_nodes,
                ps,
                np.array([x_inj], dtype=np.float64),
                np.array([0.0], dtype=np.float64),
                np.array([0.0], dtype=np.float64),
            )[0]
        )
        q_over_mc2 = float(sp.charge) / (float(sp.mass) * c_light * c_light)
        gamma_birth_after_kick = gamma0 + (dgamma_kick_pred if enable_kick else 0.0)
        birth_excess = gamma_birth_after_kick - gamma0 + q_over_mc2 * phi_birth_gat

        B_zero = (
            np.zeros_like(ex_mat).ravel(order="F"),
            np.zeros_like(ex_mat).ravel(order="F"),
            np.zeros_like(ex_mat).ravel(order="F"),
        )

        n_total_steps = int(n_base_steps * dt_div)
        step_pre_coarse = 7 * dt_div - 1
        step_pre_kink_sub = 8 * dt_div - 2
        step_after_kink = 8 * dt_div - 1
        excess_before_kink_sync = 0.0
        excess_pre_kink_sub_sync = 0.0
        excess_after_kink_sync = 0.0
        zeta_pre_kink_sub = 0.0
        zeta_after_kink = 0.0

        for step in range(n_total_steps):
            solver.one_step()
            if step == 0:
                ps.injectors.clear()
            if step == step_pre_coarse:
                _, excess_before_kink_sync, _, zeta_cur = eval_particle_excess(
                    sp, ps, phi_nodes, B_zero, B_ext, gamma0, dt
                )
                if step_pre_kink_sub == step_pre_coarse:
                    excess_pre_kink_sub_sync = excess_before_kink_sync
                    zeta_pre_kink_sub = zeta_cur
            elif step == step_pre_kink_sub:
                _, excess_pre_kink_sub_sync, _, zeta_pre_kink_sub = (
                    eval_particle_excess(sp, ps, phi_nodes, B_zero, B_ext, gamma0, dt)
                )
            elif step == step_after_kink:
                _, excess_after_kink_sync, _, zeta_after_kink = eval_particle_excess(
                    sp, ps, phi_nodes, B_zero, B_ext, gamma0, dt
                )

        excess_raw, excess_sync, phi_gat_final, zeta_final = eval_particle_excess(
            sp, ps, phi_nodes, B_zero, B_ext, gamma0, dt
        )
        wall_kick_J = float(ps.wall_kick_energy())
        kink_jump = excess_after_kink_sync - excess_pre_kink_sub_sync
        return {
            "dt_div": int(dt_div),
            "dt": dt,
            "n_steps": n_total_steps,
            "line_idx": int(line_idx),
            "x_inj": float(x_inj),
            "x_final": float(sp.x[0]),
            "z_final": float(sp.z[0]),
            "zeta_final": zeta_final,
            "z_over_L": float(sp.z[0] / L_decay),
            "phi_gathered": phi_gat_final,
            "phi_birth_gathered": phi_birth_gat,
            "psi_birth": psi_birth,
            "dgamma_kick_pred": dgamma_kick_pred,
            "dK_kick_pred_J": dK_kick_pred_J,
            "wall_kick_energy_J": wall_kick_J,
            "birth_excess": birth_excess,
            "excess_before_kink_sync": excess_before_kink_sync,
            "excess_after_kink_sync": excess_after_kink_sync,
            "zeta_before_kink": zeta_pre_kink_sub,
            "zeta_after_kink": zeta_after_kink,
            "kink_jump_sync": kink_jump,
            "excess_smooth_interior_sync": excess_sync - kink_jump,
            "excess_raw": excess_raw,
            "excess_sync": excess_sync,
        }
    finally:
        psolver_mod._ENABLE_PEC_WALL_KICK = prev_kick_flag


def main(argv: list[str] | None = None) -> int:
    """Run the single-particle frozen-potential study and print ``SP1`` lines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dt-divs",
        type=str,
        default="1,2,4",
        help="Comma-separated dt subdivision factors (default: 1,2,4).",
    )
    parser.add_argument(
        "--line-idx",
        type=int,
        default=None,
        help="Sheet lattice line index (default: N_MACRO_PER_STEP // 2).",
    )
    parser.add_argument(
        "--decay-length-m",
        type=float,
        default=None,
        help="Axial sheath decay length L in [m] (default: tunnel half-width a).",
    )
    parser.add_argument(
        "--wall-stencil",
        choices=("masked", "image"),
        default="masked",
        help="ParticleSolver wall_stencil (default: masked, the released behaviour).",
    )
    args = parser.parse_args(argv)
    global WALL_STENCIL
    WALL_STENCIL = str(args.wall_stencil)
    print(f"WALL_STENCIL {WALL_STENCIL}", flush=True)

    dt_divs = [int(tok.strip()) for tok in args.dt_divs.split(",") if tok.strip()]
    nx, _ny, nz = s.COARSE_GRID
    params = s.compute_analytic_parameters(qm_scale=1.0)
    gamma0 = float(params["gamma0"])
    beta0 = float(np.sqrt(1.0 - 1.0 / (gamma0 * gamma0)))
    a = float(s.TUNNEL_HALF_WIDTH_M)
    ly = float(s.PERIODIC_LY_M)
    lz = float(s.TUNNEL_LENGTH_M)
    cfln = float(s.CFL_NUMBER)
    shape = int(s.PARTICLE_SHAPE)
    dx = float(2.0 * a / nx)
    dz = float(lz / nz)

    det = s.compute_discrete_dc_depression(
        nx, params, weighting="charge", return_details=True, shape=shape
    )
    x_sheet = np.asarray(det["x_sheet"], dtype=np.float64)
    line_idx = int(len(x_sheet) // 2) if args.line_idx is None else int(args.line_idx)
    x_inj = float(x_sheet[line_idx])
    phi_min = -float(np.max(det["phi_l"]))
    L_decay = float(a) if args.decay_length_m is None else float(args.decay_length_m)

    grid = GridFIT3D(-a, a, -0.5 * ly, 0.5 * ly, 0.0, lz, nx, 1, nz, verbose=0)
    dt_base, _ = s._integer_period_dt(grid, cfln=cfln)

    # Choose n_base_steps so z_final > 5.0 * L_decay (well inside z > 5 L)
    v_z0 = beta0 * c_light
    n_base_steps = max(int(np.ceil(5.45 * L_decay / (v_z0 * dt_base))), 360)

    solver = SolverFIT3D(
        grid,
        cfln=cfln,
        dt=dt_base,
        bc_low=["pec", "periodic", "pec"],
        bc_high=["pec", "periodic", "pec"],
        use_gpu=False,
        verbose=0,
    )
    phi_nodes, ex_mat, ez_mat = build_prescribed_potential_and_edges(
        solver, nx, nz, a, lz, phi_min, L_decay
    )
    solver.E.from_matrix(ex_mat, "x")
    solver.E.from_matrix(np.zeros_like(ex_mat), "y")
    solver.E.from_matrix(ez_mat, "z")

    # Freeze field update while keeping SolverFIT3D._one_step mid-step hook active
    solver.one_step = solver._one_step
    solver.step_0 = False
    solver._update_H = lambda: None
    solver._update_E = lambda: None

    e_initial_snap = solver.E.toarray().copy()

    # Transverse gathered Ez_0 and Ez_1 at x_inj for analytical zeta=0.5 knot formula
    xi_inj = (x_inj - (-a)) / dx
    ix0, wx = bspline_weights_1d(np.array([xi_inj]), shape, staggered=False, xp=np)
    ez0_gat = sum(
        float(wx[m][0]) * float(ez_mat[int(ix0[0]) + m, 0, 0]) for m in range(shape + 1)
    )
    ez1_gat = sum(
        float(wx[m][0]) * float(ez_mat[int(ix0[0]) + m, 0, 1]) for m in range(shape + 1)
    )

    records = []
    for dt_div in dt_divs:
        r_on = run_single_particle_case(
            solver,
            phi_nodes,
            ex_mat,
            ez_mat,
            dt_base,
            dt_div,
            x_inj,
            line_idx,
            gamma0,
            L_decay,
            n_base_steps,
            enable_kick=True,
        )
        r_off = run_single_particle_case(
            solver,
            phi_nodes,
            ex_mat,
            ez_mat,
            dt_base,
            dt_div,
            x_inj,
            line_idx,
            gamma0,
            L_decay,
            n_base_steps,
            enable_kick=False,
        )

        # Verify field remained strictly frozen
        max_dE = float(np.max(np.abs(solver.E.toarray() - e_initial_snap)))
        max_H = float(np.max(np.abs(solver.H.toarray())))
        if max_dE != 0.0 or max_H != 0.0:
            raise AssertionError(
                f"Field was not frozen: max_dE={max_dE:.3e}, max_H={max_H:.3e}"
            )
        if r_on["z_over_L"] <= 5.0:
            raise AssertionError(
                f"Particle did not reach z > 5 L: z/L={r_on['z_over_L']:.3f}"
            )

        # Analytical knot-jump prediction at zeta = 0.5
        d_zeta = float(r_on["zeta_after_kink"] - r_on["zeta_before_kink"])
        theta = float((r_on["zeta_after_kink"] - 0.5) / d_zeta)
        jump_field = (2.0 * ez0_gat - ez1_gat) if WALL_STENCIL == "masked" else (ez0_gat - ez1_gat)
        kink_jump_pred = float(
            (e_charge * dz / (m_e * c_light * c_light))
            * 0.5
            * theta
            * (1.0 - theta)
            * (d_zeta**2)
            * jump_field
        )

        # Kick-disabled checks:
        # 1. At birth (z = 0): birth_excess_off == -dgamma_kick_pred == -(q/mc^2) sum_k W_k Psi_k
        #    and wall_kick_energy_J == q * sum_k W_k Psi_k to 1e-12 relative.
        pred_dgamma = r_on["dgamma_kick_pred"]
        pred_dK_J = r_on["dK_kick_pred_J"]
        birth_rel_err_dgamma = abs(r_off["birth_excess"] - (-pred_dgamma)) / max(
            abs(pred_dgamma), 1e-30
        )
        birth_rel_err_J = abs(r_on["wall_kick_energy_J"] - pred_dK_J) / max(
            abs(pred_dK_J), 1e-30
        )
        # 2. At z > 5 L after subtracting the trajectory leapfrog error excess_on:
        #    (excess_off_sync - excess_on_sync) vs -dgamma_kick_pred
        interior_kick_diff_sync = r_off["excess_sync"] - r_on["excess_sync"]
        interior_kick_rel_err = abs(interior_kick_diff_sync - (-pred_dgamma)) / max(
            abs(pred_dgamma), 1e-30
        )

        if birth_rel_err_dgamma >= 1e-12 or birth_rel_err_J >= 1e-12:
            raise AssertionError(
                f"Birth kick relative error exceeded 1e-12: "
                f"rel_dgamma={birth_rel_err_dgamma:.3e}, rel_J={birth_rel_err_J:.3e}"
            )

        rec = {
            "dt_div": int(dt_div),
            "dt": r_on["dt"],
            "n_steps": r_on["n_steps"],
            "x_inj": r_on["x_inj"],
            "z_final": r_on["z_final"],
            "L_decay": L_decay,
            "z_over_L": r_on["z_over_L"],
            "phi_gathered": r_on["phi_gathered"],
            "excess_kick_on_raw": r_on["excess_raw"],
            "excess_kick_on_sync": r_on["excess_sync"],
            "excess_before_kink_sync": r_on["excess_before_kink_sync"],
            "kink_jump_sync": r_on["kink_jump_sync"],
            "kink_jump_pred": kink_jump_pred,
            "kink_theta": theta,
            "excess_smooth_interior_sync": r_on["excess_smooth_interior_sync"],
            "excess_kick_off_raw": r_off["excess_raw"],
            "excess_kick_off_sync": r_off["excess_sync"],
            "birth_excess_kick_on": r_on["birth_excess"],
            "birth_excess_kick_off": r_off["birth_excess"],
            "pred_birth_kick_dgamma": pred_dgamma,
            "pred_birth_kick_J": pred_dK_J,
            "wall_kick_energy_J": r_on["wall_kick_energy_J"],
            "birth_kick_rel_err_dgamma": birth_rel_err_dgamma,
            "birth_kick_rel_err_J": birth_rel_err_J,
            "interior_kick_diff_rel_err": interior_kick_rel_err,
        }
        records.append(rec)
        print(f"SP1 {json.dumps(rec, sort_keys=True)}")

    # Verify O(dt^2) ~4x scaling on the smooth region before the zeta=0.5 knot
    # and on the smooth interior contribution after subtracting the zeta=0.5 knot jump
    if len(records) >= 2:
        for idx in range(len(records) - 1):
            r0 = records[idx]
            r1 = records[idx + 1]
            if WALL_STENCIL == "image":
                # Constant gathered E_z in the first half cell: the pre-knot excess is round-off.
                print(
                    f"PREKNOT dt/{r0['dt_div']}: |excess_before_kink_sync| = "
                    f"{abs(r0['excess_before_kink_sync']):.3e}; knot jump {r0['kink_jump_sync']:+.6e} "
                    f"(pred {r0['kink_jump_pred']:+.6e})"
                )
                continue
            ratio_pre = r0["excess_before_kink_sync"] / r1["excess_before_kink_sync"]
            ratio_smooth = (
                r0["excess_smooth_interior_sync"] / r1["excess_smooth_interior_sync"]
            )
            print(
                f"SCALING dt/{r0['dt_div']} -> dt/{r1['dt_div']}: "
                f"raw_ratio={r0['excess_kick_on_raw'] / r1['excess_kick_on_raw']:.4f} "
                f"sync_ratio={r0['excess_kick_on_sync'] / r1['excess_kick_on_sync']:.4f} "
                f"pre_kink_ratio={ratio_pre:.4f} "
                f"smooth_interior_ratio={ratio_smooth:.4f}"
            )
            if not (3.8 <= ratio_pre <= 4.2):
                raise AssertionError(
                    f"Pre-kink O(dt^2) scaling ratio out of [3.8, 4.2]: {ratio_pre:.4f}"
                )

    if WALL_STENCIL == "image":
        if any(abs(r["birth_excess_kick_off"]) > 0.0 or r["wall_kick_energy_J"] != 0.0 for r in records):
            raise AssertionError("image stencil: nonzero birth potential or birth kick")
        print("SELF_CHECK PASS (image): zero birth potential and zero birth kick.")
        return 0
    # Mutation check: perturbing the predicted birth kick by 1e-6 relative MUST fail
    # the 1e-12 relative check.
    mutated_pred = records[0]["pred_birth_kick_dgamma"] * (1.0 + 1e-6)
    mutated_rel_err = abs(records[0]["birth_excess_kick_off"] - (-mutated_pred)) / abs(
        mutated_pred
    )
    if mutated_rel_err < 1e-12:
        raise AssertionError(
            "Mutation check failed to detect 1e-6 birth kick perturbation."
        )
    print(
        f"SELF_CHECK PASS: frozen field, z > 5 L, birth kick rel_err="
        f"{records[0]['birth_kick_rel_err_dgamma']:.3e} (< 1e-12), "
        f"mutation_rel_err={mutated_rel_err:.3e} (> 1e-12)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
