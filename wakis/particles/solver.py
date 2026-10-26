# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Self-consistent electromagnetic particle-in-cell coupler (`ParticleSolver`) and diagnostics.

Couples one or more :class:`~wakis.particles.Species` (and optional
:class:`~wakis.particles.BeamInjector` instances) to :class:`~wakis.SolverFIT3D`
via the ``mid_step`` hook executed between the magnetic-field update
``H^{n-1/2} -> H^{n+1/2}`` and the electric-field update ``E^n -> E^{n+1}``.

Time-level centring per step ``n``:
1. ``B^n = mu_0 * (H^{n-1/2} + H^{n+1/2}) / 2 + B_ext(x^n, t_n)``.
2. Gather ``E^n + E_ext(x^n, t_n)`` and ``B^n`` at ``x^n``.
3. Advance ``u^{n-1/2} -> u^{n+1/2}`` (Boris, Vay, or Higuera-Cary).
4. Advance ``x^{n+1} = x^n + dt * u^{n+1/2} / gamma^{n+1/2}``.
5. Truncate boundary-crossing paths and deposit dual-face charges ``W_e``
   via the 3D Esirkepov scheme; convert to primal-edge current density
   ``J_e^{n+1/2} = W_e * itA_e / dt``.
6. Replace the particle share of ``solver.J`` (subtract ``J_particles^{n-1/2}``, add ``J_particles^{n+1/2}``) so external
   ``CurrentSource`` objects can share the same edges.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.constants import mu_0

from .boundaries import Collector, ParticleBoundaries
from .mapping import LogicalGridMap
from .pushers import get_pusher, push_momentum, push_position
from .shapes import (
    deposit_charge,
    deposit_current_esirkepov,
    deposit_current_naive,
    gather_fields,
)
from .species import Species

_CPU_CHUNK = 65536


def _cpu_chunk_ranges(n_p: int):
    """Yield ``(s0, s1)`` slice bounds of size at most ``_CPU_CHUNK``."""
    return (
        (s0, min(s0 + _CPU_CHUNK, n_p))
        for s0 in range(0, n_p, _CPU_CHUNK)
    )


def apply_binomial_filter_3d(
    Wx_flat,
    Wy_flat,
    Wz_flat,
    Nx: int,
    Ny: int,
    Nz: int,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    compensate: bool = True,
    xp=np,
):
    """Apply a 3D separable binomial (1-2-1)/4 filter (with optional compensation) to edge fluxes.

    On uniform periodic grids, applying the identical translation-invariant
    stencil to ``(Wx, Wy, Wz)`` commutes with the discrete divergence ``S_tilde``.
    """

    def _filter_1d_axis(arr_3d, axis: int, is_per: bool, alpha: float):
        if arr_3d.shape[axis] < 3:
            return arr_3d
        if is_per:
            minus = xp.roll(arr_3d, 1, axis=axis)
            plus = xp.roll(arr_3d, -1, axis=axis)
            return (1.0 - 2.0 * alpha) * arr_3d + alpha * (minus + plus)
        out = arr_3d.copy()
        sl_mid = [slice(None), slice(None), slice(None)]
        sl_m = [slice(None), slice(None), slice(None)]
        sl_p = [slice(None), slice(None), slice(None)]
        sl_mid[axis] = slice(1, -1)
        sl_m[axis] = slice(0, -2)
        sl_p[axis] = slice(2, None)
        out[tuple(sl_mid)] = (1.0 - 2.0 * alpha) * arr_3d[tuple(sl_mid)] + alpha * (
            arr_3d[tuple(sl_m)] + arr_3d[tuple(sl_p)]
        )
        return out

    alphas = (0.25, -1.0 / 6.0) if compensate else (0.25,)
    out_flats = []
    for W_flat in (Wx_flat, Wy_flat, Wz_flat):
        arr = W_flat.reshape((Nx, Ny, Nz), order="F")
        for alpha in alphas:
            for ax in range(3):
                arr = _filter_1d_axis(arr, ax, periodic[ax], alpha)
        out_flats.append(arr.reshape(-1, order="F"))
    return out_flats[0], out_flats[1], out_flats[2]


def compute_dual_divergence_d(solver):
    """Compute dual divergence ``(S_tilde d_tilde)_{i,j,k}`` [C] on primal nodes.

    In FIT, the dual electric flux on primal edge ``e`` is
    ``d_tilde_e = eps_e * E_e * A_tilde_e = E_e / (ieps_e * itA_e)`` for active
    edges (``ieps_e > 0``) and ``0`` on PEC edges (``ieps_e == 0``).
    The dual divergence at primal node ``(i, j, k)`` is:
    ``(d_x[i,j,k] - d_x[i-1,j,k]) + (d_y[i,j,k] - d_y[i,j-1,k]) + (d_z[i,j,k] - d_z[i,j,k-1])``.
    """
    xp = getattr(solver.E, "xp", np)
    Nx, Ny, Nz = solver.Nx, solver.Ny, solver.Nz

    bc_low = getattr(solver, "bc_low", ("periodic", "periodic", "periodic"))
    bc_high = getattr(solver, "bc_high", ("periodic", "periodic", "periodic"))
    periodic = tuple(
        str(bc_low[a]).strip().lower() in ("periodic", "bloch")
        and str(bc_high[a]).strip().lower() in ("periodic", "bloch")
        for a in range(3)
    )

    itA_field = getattr(solver, "itA", solver.grid.itA)
    flux_comps = []
    for comp in ("x", "y", "z"):
        E_c = solver.E.to_matrix(comp)
        ieps_c = solver.ieps.to_matrix(comp)
        itA_c = xp.asarray(itA_field.to_matrix(comp), dtype=np.float64)
        denom = ieps_c * itA_c
        active = denom > 0.0
        d_c = xp.where(active, E_c / xp.where(active, denom, 1.0), 0.0)
        flux_comps.append(d_c)

    dx_f, dy_f, dz_f = flux_comps

    if periodic[0]:
        div_x = dx_f - xp.roll(dx_f, 1, axis=0)
    else:
        div_x = xp.zeros_like(dx_f)
        div_x[1:, :, :] = dx_f[1:, :, :] - dx_f[:-1, :, :]
        div_x[0, :, :] = dx_f[0, :, :]

    if periodic[1]:
        div_y = dy_f - xp.roll(dy_f, 1, axis=1)
    else:
        div_y = xp.zeros_like(dy_f)
        div_y[:, 1:, :] = dy_f[:, 1:, :] - dy_f[:, :-1, :]
        div_y[:, 0, :] = dy_f[:, 0, :]

    if periodic[2]:
        div_z = dz_f - xp.roll(dz_f, 1, axis=2)
    else:
        div_z = xp.zeros_like(dz_f)
        div_z[:, :, 1:] = dz_f[:, :, 1:] - dz_f[:, :, :-1]
        div_z[:, :, 0] = dz_f[:, :, 0]

    return div_x + div_y + div_z


def _interior_vacuum_node_mask(solver):
    """Boolean mask of primal nodes ``(i, j, k)`` strictly in the active vacuum region."""
    xp = getattr(solver.E, "xp", np)
    Nx, Ny, Nz = solver.Nx, solver.Ny, solver.Nz
    mask = xp.ones((Nx, Ny, Nz), dtype=bool)

    bc_low = getattr(solver, "bc_low", ("periodic", "periodic", "periodic"))
    bc_high = getattr(solver, "bc_high", ("periodic", "periodic", "periodic"))
    edge_ghosts = getattr(getattr(solver, "grid", None), "edge_ghosts", (0, 0, 0))
    n_pml = int(getattr(solver, "n_pml", 10))
    shape_order = int(getattr(solver, "_particles_shape_order", 1))
    pad = (shape_order + 1) // 2 + 1

    for ax, n_ax in enumerate((Nx, Ny, Nz)):
        low_s = str(bc_low[ax]).strip().lower()
        high_s = str(bc_high[ax]).strip().lower()
        is_per = low_s in ("periodic", "bloch") and high_s in ("periodic", "bloch")
        if is_per:
            continue

        n_phys = n_ax - int(edge_ghosts[ax])
        i_start = (n_pml + pad) if low_s in ("pml", "cpml") else 1
        i_end = (n_phys - n_pml - pad + 1) if high_s in ("pml", "cpml") else n_phys

        sl_lo = [slice(None), slice(None), slice(None)]
        sl_lo[ax] = slice(0, min(i_start, n_ax))
        mask[tuple(sl_lo)] = False

        if i_end < n_ax:
            sl_hi = [slice(None), slice(None), slice(None)]
            sl_hi[ax] = slice(max(0, i_end), n_ax)
            mask[tuple(sl_hi)] = False

    # Exclude nodes adjacent to any internal PEC edge (ieps == 0)
    for ax, comp in enumerate(("x", "y", "z")):
        pec_edge = solver.ieps.to_matrix(comp) == 0.0
        if bool(xp.any(pec_edge)):
            mask &= ~pec_edge
            shifted = xp.zeros_like(pec_edge)
            sl_dst = [slice(None), slice(None), slice(None)]
            sl_src = [slice(None), slice(None), slice(None)]
            sl_dst[ax] = slice(1, None)
            sl_src[ax] = slice(0, -1)
            shifted[tuple(sl_dst)] = pec_edge[tuple(sl_src)]
            mask &= ~shifted

    return mask


def gauss_residual(solver, species, order: int | None = None) -> float:
    """Compute the relative discrete Gauss's law residual ``max|S_tilde d_tilde - q| / max|q|``.

    Evaluates ``S_tilde (eps E A_tilde) - (q_nodes + q_bg)`` on all interior
    vacuum nodes, normalised by ``max(|q_nodes|, |q_0|)``. Here ``q_bg`` is the
    immobile background node charge recorded when :class:`ParticleSolver` was
    initialized (zero if ``E(0)`` satisfied Poisson's equation or if the domain
    started empty; ``-q_nodes(0)`` if the simulation started from neutral
    equilibrium ``E(0) = 0`` with ``q_nodes(0) != 0``).

    Parameters
    ----------
    solver : SolverFIT3D
        Electromagnetic FIT solver instance.
    species : Species or sequence of Species
        Active particle species.
    order : int, optional
        B-spline shape order ``p``. Defaults to ``solver._particles_shape_order``
        if attached, else ``1``.

    Returns
    -------
    float
        Dimensionless max-norm Gauss residual in the interior vacuum region.
    """
    xp = getattr(solver.E, "xp", np)
    ps_att = getattr(solver, "_particle_solver", None)
    if (
        ps_att is not None
        and getattr(ps_att, "_d_cfl_error", None) is not None
        and int(ps_att._d_cfl_error[0].get()) > 0
    ):
        raise ValueError(
            "Particle crossed more than 1 cell in a single step on GPU "
            "(|db| > 1 violates the Esirkepov single-cell CFL assumption)."
        )
    if isinstance(species, Species):
        species_list = [species]
    else:
        species_list = list(species)

    if order is None:
        order = int(getattr(solver, "_particles_shape_order", 1))

    gmap = getattr(solver, "_particles_grid_map", None)
    if gmap is None:
        gmap = LogicalGridMap.from_grid(solver, xp=xp)

    bc_low = getattr(solver, "bc_low", ("periodic", "periodic", "periodic"))
    bc_high = getattr(solver, "bc_high", ("periodic", "periodic", "periodic"))
    periodic = tuple(
        str(bc_low[a]).strip().lower() in ("periodic", "bloch")
        and str(bc_high[a]).strip().lower() in ("periodic", "bloch")
        for a in range(3)
    )

    q_nodes = xp.zeros((solver.Nx, solver.Ny, solver.Nz), dtype=np.float64)
    for sp in species_list:
        if sp.n_particles == 0:
            continue
        xi, eta, zeta = gmap.to_logical(sp.x, sp.y, sp.z)
        qw = sp.charge * sp.w
        deposit_charge(
            xi,
            eta,
            zeta,
            qw,
            solver.Nx,
            solver.Ny,
            solver.Nz,
            order=order,
            periodic=periodic,
            q_nodes=q_nodes,
            xp=xp,
        )

    div_d = compute_dual_divergence_d(solver)
    q_bg = getattr(solver, "_particles_q_bg", None)
    if q_bg is not None:
        q_tot = q_nodes + q_bg
    else:
        q_tot = q_nodes

    vac_mask = _interior_vacuum_node_mask(solver)
    if not bool(xp.any(vac_mask)):
        raise ValueError(
            "gauss_residual: no interior vacuum nodes outside the CPML and the shape-stencil "
            "margin; enlarge the domain (the residual on CPML nodes is not a conservation check)."
        )

    res = xp.abs((div_d - q_tot)[vac_mask])
    max_res = float(xp.max(res)) if res.size > 0 else 0.0

    q_scale = float(xp.max(xp.abs(q_nodes[vac_mask]))) if res.size > 0 else 0.0
    q_ref = float(getattr(solver, "_particles_q_scale_ref", 0.0))
    denom = max(q_scale, q_ref, 1e-30)
    return max_res / denom


def cold_plasma_dispersion_discrete(
    omega_p0: float,
    dt: float,
    dz: float,
    k: float,
    order: int = 1,
    gather: str = "galerkin",
) -> float:
    """Birdsall-Langdon finite-grid, finite-``dt`` cold plasma oscillation frequency [rad/s].

    For a 1D cold plasma with continuum frequency
    ``omega_p0 = sqrt(n_0 q^2 / (eps_0 m))``, wavenumber ``k`` along ``z``,
    Esirkepov current deposition of B-spline order ``p = order`` (which acts on
    staggered edges with effective shape ``S^{(p-1)}`` and Fourier symbol
    ``exp(-i k_m dz / 2) * sinc^p(k_m dz / 2)``), and field gather of staggered
    order ``p_g`` (``p - 1`` for ``"galerkin"`` and ``p`` for ``"momentum"``,
    with conjugate symbol ``exp(+i k_m dz / 2) * sinc^{p_g + 1}(k_m dz / 2)``),
    summing the product of deposit and gather symbols over Brillouin aliases
    ``k_m = k + 2 * pi * m / dz`` yields:

    .. math::

        \\kappa_{\\mathrm{grid}}^2(\\theta)
        = \\sum_{m=-\\infty}^{\\infty}
          \\left[\\operatorname{sinc}\\!\\left(\\frac{\\theta}{2} + \\pi m\\right)\\right]^{p + p_g + 1},
        \\qquad \\theta = k \\, \\Delta z,
        \\qquad s = \\sin\\!\\left(\\frac{k \\, \\Delta z}{2}\\right),

    which evaluates in closed form via the partial-fraction expansions of
    ``csc^2(theta/2)`` (even exponent ``2p`` for ``"galerkin"``) and
    ``csc(theta/2)`` (odd exponent ``2p + 1`` for ``"momentum"``) to:
    - ``p = 1, "galerkin"`` (``N = 2``): ``kappa^2 = 1``
    - ``p = 1, "momentum"`` (``N = 3``): ``kappa^2 = 1 - 0.5 * s^2``
    - ``p = 2, "galerkin"`` (``N = 4``): ``kappa^2 = 1 - (2/3) * s^2``
    - ``p = 2, "momentum"`` (``N = 5``): ``kappa^2 = 1 - (5/6) * s^2 + (1/24) * s^4``

    Combined with the leapfrog time integrator, the discrete frequency is:

    .. math::

        \\frac{2}{\\Delta t} \\sin\\!\\left(\\frac{\\omega_p \\Delta t}{2}\\right)
        = \\omega_{p0} \\, \\kappa_{\\mathrm{grid}}(k \\, \\Delta z).
    """
    mode = str(gather).strip().lower()
    is_galerkin = mode in ("galerkin", "energy", "energy-conserving")
    s2 = float(np.sin(0.5 * k * dz) ** 2)

    if order == 1 and is_galerkin:
        kappa2 = 1.0
    elif order == 1 and not is_galerkin:
        kappa2 = 1.0 - 0.5 * s2
    elif order == 2 and is_galerkin:
        kappa2 = 1.0 - (2.0 / 3.0) * s2
    elif order == 2 and not is_galerkin:
        kappa2 = 1.0 - (5.0 / 6.0) * s2 + (1.0 / 24.0) * (s2 * s2)
    else:
        # Direct alias summation for higher orders: sinc(theta/2 + pi*m)**(p + p_g + 1)
        p_g = (order - 1) if is_galerkin else order
        exp_tot = order + p_g + 1
        theta = k * dz
        m = np.arange(-80, 81, dtype=np.float64)
        arg = 0.5 * theta + np.pi * m
        sinc = np.where(np.abs(arg) < 1e-14, 1.0, np.sin(arg) / arg)
        kappa2 = float(np.sum(sinc**exp_tot))

    omega_eff = omega_p0 * np.sqrt(max(0.0, kappa2))
    arg_sin = 0.5 * omega_eff * dt
    if arg_sin >= 1.0:
        raise ValueError(
            f"Cold plasma leapfrog stability limit exceeded: omega_eff * dt / 2 = {arg_sin:.4f} >= 1."
        )
    return float((2.0 / dt) * np.arcsin(arg_sin))


class ParticleSolver:
    """Self-consistent electromagnetic particle-in-cell coupler for :class:`~wakis.SolverFIT3D`.

    Parameters
    ----------
    solver : SolverFIT3D
        Electromagnetic FIT solver instance.
    species : Species or sequence of Species
        One or more macroparticle species.
    injectors : sequence of BeamInjector, optional
        Optional particle injectors executed each step.
    collectors : sequence of Collector, optional
        Optional user collector regions.
    pusher : {"boris", "vay", "hc"}, optional
        Relativistic momentum pusher. Default is ``"boris"``.
    shape : {1, 2, 3}, optional
        B-spline particle shape order ``p`` (1: CIC, 2: TSC, 3: PCS).
        Default is ``1``.
    gather : {"galerkin", "momentum"}, optional
        Field interpolation mode. Default is ``"galerkin"``.
    filter : bool or str or None, optional
        Optional binomial current filter (``"binomial"`` or ``True``).
        Default is ``None``.
    sort_every : int, optional
        Cell-sorting period in time steps (``0`` disables periodic sorting).
        Default is ``50``.
    B_ext : array-like of shape (3,) or callable, optional
        External magnetic flux density [T], either a uniform 3-vector
        ``(Bx0, By0, Bz0)`` or callable ``B_ext(x, y, z, t) -> (Bx, By, Bz)``.
    E_ext : array-like of shape (3,) or callable, optional
        External electric field [V/m], either a uniform 3-vector
        ``(Ex0, Ey0, Ez0)`` or callable ``E_ext(x, y, z, t) -> (Ex, Ey, Ez)``.
    solid_mask : ndarray of bool, shape (Nx, Ny, Nz), optional
        Optional explicit solid/PEC cell mask.
    deposit_scheme : {"esirkepov", "naive"}, optional
        Current deposition scheme. Default is ``"esirkepov"`` (exact charge
        conservation). ``"naive"`` is provided for V2 mutation testing.
    profile : bool, optional
        If True, record per-step CPU stage wall-clock times
        (``_last_push_gather_s`` and ``_last_deposit_s``) via
        :func:`time.perf_counter` inside :meth:`_mid_step_hook`. Default is
        ``False`` (zero timing calls per step).
    """

    def __init__(
        self,
        solver,
        species,
        injectors=None,
        collectors=None,
        pusher: str = "boris",
        shape: int = 1,
        gather: str = "galerkin",
        filter=None,
        sort_every: int = 50,
        B_ext=None,
        E_ext=None,
        solid_mask=None,
        deposit_scheme: str = "esirkepov",
        profile: bool = False,
    ):
        if getattr(solver, "use_mpi", False):
            raise NotImplementedError(
                "ParticleSolver with use_mpi=True is not implemented in milestone M2 "
                "(scheduled for milestone M3)."
            )

        self.solver = solver
        self.on_gpu = bool(getattr(solver, "use_gpu", False))
        if self.on_gpu:
            from .species import _HAS_CUPY, cp

            if not _HAS_CUPY:
                raise ImportError(
                    "[!] CuPy could not be imported; cannot use_gpu=True in ParticleSolver."
                )
            self.xp = cp
        else:
            self.xp = getattr(solver.E, "xp", np)
            if self.xp is not np:
                self.on_gpu = True

        if isinstance(species, Species):
            self.species = [species]
        else:
            self.species = list(species)
        for sp in self.species:
            if self.on_gpu and not sp.on_gpu:
                sp.to_gpu()
            elif not self.on_gpu and sp.on_gpu:
                sp.to_cpu()

        self.injectors = list(injectors) if injectors is not None else []
        get_pusher(pusher)
        self.pusher = str(pusher).strip().lower()

        if int(shape) not in (1, 2, 3):
            raise ValueError(f"Unsupported shape={shape!r}; expected 1, 2, or 3.")
        self.shape = int(shape)

        gather_norm = str(gather).strip().lower()
        if gather_norm not in (
            "galerkin",
            "energy",
            "energy-conserving",
            "momentum",
            "momentum-conserving",
        ):
            raise ValueError(
                f"Unknown gather={gather!r}; expected 'galerkin' or 'momentum'."
            )
        self.gather = gather_norm
        self._p_stag = (
            (self.shape - 1)
            if self.gather in ("galerkin", "energy", "energy-conserving")
            else self.shape
        )

        self.filter = filter
        self.sort_every = int(sort_every)
        self.B_ext = B_ext
        self.E_ext = E_ext

        dep_norm = str(deposit_scheme).strip().lower()
        if dep_norm not in ("esirkepov", "naive"):
            raise ValueError(
                f"Unknown deposit_scheme={deposit_scheme!r}; expected 'esirkepov' or 'naive'."
            )
        self.deposit_scheme = dep_norm

        self.grid_map = LogicalGridMap.from_grid(solver, xp=self.xp)
        self.boundaries = ParticleBoundaries.from_solver(
            solver, collectors=collectors, solid_mask=solid_mask
        )

        n_cells = solver.Nx * solver.Ny * solver.Nz
        self.J_particles = self.xp.zeros(3 * n_cells, dtype=np.float64)
        self._B_n_buf = self.xp.zeros(3 * n_cells, dtype=np.float64)
        if self.on_gpu:
            self._J_prev = self.xp.zeros(3 * n_cells, dtype=np.float64)
            self._Wx_buf = self.xp.zeros(n_cells, dtype=np.float64)
            self._Wy_buf = self.xp.zeros(n_cells, dtype=np.float64)
            self._Wz_buf = self.xp.zeros(n_cells, dtype=np.float64)
        else:
            self._J_prev = None
            self._Wx_buf = self.J_particles[:n_cells]
            self._Wy_buf = self.J_particles[n_cells : 2 * n_cells]
            self._Wz_buf = self.J_particles[2 * n_cells : 3 * n_cells]
        self._q0_buf = self.xp.zeros(
            (solver.Nx, solver.Ny, solver.Nz), dtype=np.float64, order="F"
        )
        itA_raw = getattr(solver, "itA", solver.grid.itA).toarray()
        self._itA_arr = (
            self.xp.asarray(itA_raw, dtype=np.float64)
            if self.on_gpu or itA_raw.dtype != np.float64
            else itA_raw
        )
        self._H_prev = None
        self._d_cfl_error = (
            self.xp.zeros(1, dtype=np.int32) if self.on_gpu else None
        )
        self._dx_min = float(
            min(
                np.min(np.asarray(solver.grid.dx)),
                np.min(np.asarray(solver.grid.dy)),
                np.min(np.asarray(solver.grid.dz)),
            )
        )
        self._attached_dt = None
        self._attached = False
        self._initialized_half_step = False
        self._init_gauss_counts = ()
        self.n_particle_steps = 0
        self._j_in_solver = False
        self.profile = bool(profile)
        self._last_push_gather_s = 0.0
        self._last_deposit_s = 0.0

    def _check_cfl_dt(self, dt: float) -> None:
        """Verify host-side relativistic single-cell CFL ``c * dt <= min_cell_size``."""
        from scipy.constants import c as c_light

        if c_light * dt > self._dx_min * (1.0 + 1e-12):
            raise ValueError(
                f"ParticleSolver requires c * dt <= min_cell_size so relativistic "
                f"particles (|v| < c) cannot cross more than one cell per step "
                f"(Esirkepov |b1 - b0| <= 1 assumption), got c * dt = {c_light * dt:.4e} m "
                f"> min_cell_size = {self._dx_min:.4e} m."
            )
        self._attached_dt = dt

    def reset_cfl_error(self) -> None:
        """Reset the GPU device CFL violation counter ``_d_cfl_error`` to zero."""
        if self._d_cfl_error is not None:
            self._d_cfl_error.fill(0)

    def reset_gauss_reference(self) -> None:
        """Re-record the initial Gauss's law background charge ``q_bg = S_tilde d_tilde - q_nodes``.

        Call this method if ``solver.E`` or particle positions/weights are
        modified in-place after :meth:`attach` (when particle counts are
        unchanged so automatic step-0 re-recording does not trigger).
        """
        self._record_initial_gauss_state()
        self._init_gauss_counts = tuple(sp._n_launch_bound for sp in self.species)

    def attach(self) -> "ParticleSolver":
        """Register the particle ``mid_step`` hook on ``self.solver`` and initialise Gauss bookkeeping.

        Raises
        ------
        NotImplementedError
            If ``solver.use_mpi`` is True.
        ValueError
            If ``solver.one_step`` is not a hooked step variant
            (``_one_step``, ``_one_step_cpml``, ``_fused_step``, or their MKL/ABC
            variants), or if ``c * solver.dt > min_cell_size`` (violating the
            single-cell Esirkepov CFL assumption ``|b1 - b0| <= 1``).
        """
        s = self.solver
        if getattr(s, "use_mpi", False):
            raise NotImplementedError(
                "ParticleSolver with use_mpi=True is not implemented in milestone M2."
            )

        hooked_steps = [
            fn
            for fn in (
                getattr(s, "_one_step", None),
                getattr(s, "_one_step_cpml", None),
                getattr(s, "_one_step_mkl", None),
                getattr(s, "_one_step_cpml_mkl", None),
                getattr(s, "_fused_step", None),
            )
            if fn is not None
        ]
        step_fn = getattr(s, "one_step", None)
        abc_fn = getattr(s, "_one_step_with_abc", None)
        if step_fn == abc_fn and abc_fn is not None:
            is_hooked = getattr(s, "_one_step_backend", None) in hooked_steps
        else:
            is_hooked = step_fn in hooked_steps
        if not is_hooked:
            step_name = getattr(step_fn, "__name__", repr(step_fn))
            raise ValueError(
                f"ParticleSolver.attach() requires a SolverFIT3D step variant that "
                f"invokes mid_step hooks (_one_step, _one_step_cpml, _fused_step, or their MKL/ABC variants), but "
                f"solver.one_step is {step_name!r}, which does not execute particle hooks."
            )

        self._dx_min = min(
            float(self.xp.min(self.grid_map.dx)),
            float(self.xp.min(self.grid_map.dy)),
            float(self.xp.min(self.grid_map.dz)),
        )
        self._check_cfl_dt(float(s.dt))
        self.reset_cfl_error()

        s._particles_grid_map = self.grid_map
        s._particles_shape_order = self.shape
        s._particle_solver = self

        if self.on_gpu:
            for sp in self.species:
                sp._ensure_gpu_scratch()
            self.boundaries._ensure_gpu_buffers()
            # Pre-compile the CUDA RawModule during attach() so step 0 does not compile
            from .gpu_kernels import _get_module

            _get_module()

        self.reset_gauss_reference()
        H_arr = s.H.toarray()
        if self.on_gpu:
            self._H_prev = H_arr.copy()
            s._H_before_update = H_arr.copy()
        else:
            self._H_prev = None
            np.copyto(self._B_n_buf, H_arr)
            s._H_before_update = self._B_n_buf

        if not self._attached:
            s.register_mid_step(self._mid_step_hook)
            self._attached = True
        return self

    def _record_initial_gauss_state(self) -> None:
        """Capture initial background charge ``q_bg = S_tilde d_tilde(0) - q_nodes(0)``."""
        s = self.solver
        xp = self.xp
        if self._q0_buf is None:
            self._q0_buf = xp.zeros(
                (s.Nx, s.Ny, s.Nz), dtype=np.float64, order="F"
            )
        q0 = self._q0_buf
        q0.fill(0.0)
        for sp in self.species:
            if sp._n_launch_bound == 0:
                continue
            if self.on_gpu:
                from .gpu_kernels import deposit_charge_gpu

                n_l = sp._n_launch_bound
                xi, eta, zeta = self.grid_map.to_logical(
                    sp._x[:n_l], sp._y[:n_l], sp._z[:n_l]
                )
                deposit_charge_gpu(
                    xi,
                    eta,
                    zeta,
                    sp._w[:n_l],
                    sp.charge,
                    s.Nx,
                    s.Ny,
                    s.Nz,
                    order=self.shape,
                    periodic=self.boundaries.periodic,
                    q_nodes=q0,
                    d_n_particles=sp._d_n_particles if sp._n_dirty_gpu else None,
                    n_launch=n_l,
                )
            else:
                xi, eta, zeta = self.grid_map.to_logical(sp.x, sp.y, sp.z)
                deposit_charge(
                    xi,
                    eta,
                    zeta,
                    sp.charge * sp.w,
                    s.Nx,
                    s.Ny,
                    s.Nz,
                    order=self.shape,
                    periodic=self.boundaries.periodic,
                    q_nodes=q0,
                    xp=xp,
                )
        div_d0 = compute_dual_divergence_d(s)
        q_bg = getattr(s, "_particles_q_bg", None)
        if q_bg is not None and q_bg.shape == q0.shape and q_bg.flags.f_contiguous:
            q_bg[...] = div_d0 - q0
        else:
            s._particles_q_bg = xp.asarray(div_d0 - q0, dtype=np.float64, order="F")
        if self.on_gpu:
            s._particles_q_scale_ref = xp.max(xp.abs(q0))
        else:
            vac_mask = _interior_vacuum_node_mask(s)
            q_max = (
                float(xp.max(xp.abs(q0[vac_mask]))) if bool(xp.any(vac_mask)) else 0.0
            )
            s._particles_q_scale_ref = max(q_max, float(xp.max(xp.abs(q0))), 0.0)

    def _eval_ext_field(self, spec, x, y, z, t: float):
        """Evaluate external uniform or callable 3-vector field at ``(x, y, z, t)``."""
        if spec is None:
            return 0.0, 0.0, 0.0
        if callable(spec):
            fx, fy, fz = spec(x, y, z, t)
            return fx, fy, fz
        arr = np.asarray(spec, dtype=np.float64).ravel()
        return float(arr[0]), float(arr[1]), float(arr[2])

    def _ext_field_args_gpu(self, spec, x, y, z, t: float):
        """Split external field into uniform host tuple and optional per-particle device arrays."""
        if spec is None:
            return (0.0, 0.0, 0.0), None
        if callable(spec):
            fx, fy, fz = spec(x, y, z, t)
            xp = self.xp
            n_l = x.size
            return (0.0, 0.0, 0.0), (
                xp.broadcast_to(xp.asarray(fx, dtype=np.float64), (n_l,)).copy(),
                xp.broadcast_to(xp.asarray(fy, dtype=np.float64), (n_l,)).copy(),
                xp.broadcast_to(xp.asarray(fz, dtype=np.float64), (n_l,)).copy(),
            )
        arr = np.asarray(spec, dtype=np.float64).ravel()
        return (float(arr[0]), float(arr[1]), float(arr[2])), None

    def _initialize_species_half_step(self, sp: Species, B_n_tuple, t_n: float) -> None:
        """Back-push ``u^0 -> u^{-1/2}`` by ``-dt / 2`` on step 0 if ``sp.u_is_synced``."""
        if not sp.u_is_synced or sp._n_launch_bound == 0:
            return
        s = self.solver
        if self.on_gpu:
            from .gpu_kernels import launch_gather_and_push_gpu

            n_l = sp._n_launch_bound
            E_uni, E_arr = self._ext_field_args_gpu(
                self.E_ext, sp._x[:n_l], sp._y[:n_l], sp._z[:n_l], t_n
            )
            B_uni, B_arr = self._ext_field_args_gpu(
                self.B_ext, sp._x[:n_l], sp._y[:n_l], sp._z[:n_l], t_n
            )
            launch_gather_and_push_gpu(
                sp,
                self.grid_map,
                s.E.toarray(),
                self._B_n_buf,
                order=self.shape,
                p_stag=self._p_stag,
                periodic=self.boundaries.periodic,
                dt=-0.5 * s.dt,
                pusher=self.pusher,
                advance_pos=False,
                E_ext_uniform=E_uni,
                B_ext_uniform=B_uni,
                E_ext_arrays=E_arr,
                B_ext_arrays=B_arr,
            )
            sp.u_is_synced = False
            return

        if self.xp is np and sp.n_particles > _CPU_CHUNK:
            from .pushers import get_pusher

            fn = get_pusher(self.pusher)
            for s0, s1 in _cpu_chunk_ranges(sp.n_particles):
                x_c, y_c, z_c = sp.x[s0:s1], sp.y[s0:s1], sp.z[s0:s1]
                xi_c, eta_c, zeta_c = self.grid_map.to_logical(x_c, y_c, z_c)
                Ex_c, Ey_c, Ez_c, Bx_c, By_c, Bz_c = gather_fields(
                    xi_c,
                    eta_c,
                    zeta_c,
                    s.E,
                    B_n_tuple,
                    s.Nx,
                    s.Ny,
                    s.Nz,
                    order=self.shape,
                    gather=self.gather,
                    periodic=self.boundaries.periodic,
                    xp=self.xp,
                )
                Ex_e, Ey_e, Ez_e = self._eval_ext_field(self.E_ext, x_c, y_c, z_c, t_n)
                Bx_e, By_e, Bz_e = self._eval_ext_field(self.B_ext, x_c, y_c, z_c, t_n)
                ux_c, uy_c, uz_c = fn(
                    sp.ux[s0:s1],
                    sp.uy[s0:s1],
                    sp.uz[s0:s1],
                    Ex_c + Ex_e,
                    Ey_c + Ey_e,
                    Ez_c + Ez_e,
                    Bx_c + Bx_e,
                    By_c + By_e,
                    Bz_c + Bz_e,
                    sp.charge,
                    sp.mass,
                    -0.5 * s.dt,
                    xp=self.xp,
                )
                sp.ux[s0:s1] = ux_c
                sp.uy[s0:s1] = uy_c
                sp.uz[s0:s1] = uz_c
            sp.u_is_synced = False
            return

        xi0, eta0, zeta0 = self.grid_map.to_logical(sp.x, sp.y, sp.z)
        Ex, Ey, Ez, Bx, By, Bz = gather_fields(
            xi0,
            eta0,
            zeta0,
            s.E,
            B_n_tuple,
            s.Nx,
            s.Ny,
            s.Nz,
            order=self.shape,
            gather=self.gather,
            periodic=self.boundaries.periodic,
            xp=self.xp,
        )
        Ex_e, Ey_e, Ez_e = self._eval_ext_field(self.E_ext, sp.x, sp.y, sp.z, t_n)
        Bx_e, By_e, Bz_e = self._eval_ext_field(self.B_ext, sp.x, sp.y, sp.z, t_n)
        push_momentum(
            sp,
            Ex + Ex_e,
            Ey + Ey_e,
            Ez + Ez_e,
            Bx + Bx_e,
            By + By_e,
            Bz + Bz_e,
            dt=-0.5 * s.dt,
            pusher=self.pusher,
        )
        sp.u_is_synced = False

    def _compute_centred_B(self, H_prev, H_curr):
        """Return time-centred magnetic flux density ``B^n = 0.5 * mu_0 * (H^{n-1/2} + H^{n+1/2})`` [T].

        Assumes vacuum permeability ``mu = mu_0`` in the active particle domain
        on both CPU and GPU paths.
        """
        if self.on_gpu and H_prev.shape == self._B_n_buf.shape:
            from .gpu_kernels import compute_centred_B_gpu

            compute_centred_B_gpu(H_prev, H_curr, self._B_n_buf)
            return self._B_n_buf
        if self.xp is np and H_prev.shape == self._B_n_buf.shape:
            np.add(H_prev, H_curr, out=self._B_n_buf)
            self._B_n_buf *= 0.5 * mu_0
            return self._B_n_buf
        return (0.5 * mu_0) * (H_prev + H_curr)

    def _mid_step_hook_gpu(self, s) -> None:
        """Asynchronous GPU execution of one particle-in-cell step.

        On the standard Esirkepov fast path (periodic or plane/box collectors,
        uniform external fields), all CUDA kernels execute on the current stream
        without device-to-host synchronization, and non-sorting steps perform
        zero CuPy memory-pool allocations. Periodic cell sorting
        (``sort_every > 0``) calls ``cupy.argsort`` on sort steps, which reuses
        cached blocks in the CuPy memory pool without growing
        ``pool.total_bytes()`` or synchronizing the host. Optional paths
        (``BeamInjector``, custom-callable ``Collector(mask=...)``,
        ``deposit_scheme="naive"``, ``filter="binomial"``, and callable
        ``E_ext``/``B_ext``) perform host synchronizations or temporary device
        allocations as documented in the Physics Guide.
        """
        from .gpu_kernels import (
            deposit_charge_gpu,
            deposit_current_esirkepov_gpu,
            finalize_j_delta_gpu,
            launch_boundary_prepare_paths_gpu,
            launch_gather_and_push_gpu,
            launch_periodic_wrap_gpu,
        )

        xp = self.xp
        dt = float(s.dt)
        if dt != self._attached_dt:
            self._check_cfl_dt(dt)
        t_n = float(self.n_particle_steps * dt)
        Nx, Ny, Nz = s.Nx, s.Ny, s.Nz
        N = Nx * Ny * Nz

        H_curr = s.H.toarray()
        H_prev = getattr(s, "_H_before_update", None)
        if H_prev is None:
            H_prev = self._H_prev if self._H_prev is not None else H_curr
            B_n_flat = self._compute_centred_B(H_prev, H_curr)
            if self._H_prev is None or self._H_prev.shape != H_curr.shape:
                self._H_prev = H_curr.copy()
            else:
                xp.copyto(self._H_prev, H_curr)
        else:
            B_n_flat = self._compute_centred_B(H_prev, H_curr)
        B_n_tuple = (B_n_flat[:N], B_n_flat[N : 2 * N], B_n_flat[2 * N : 3 * N])

        if not self._initialized_half_step:
            if (
                tuple(sp._n_launch_bound for sp in self.species)
                != self._init_gauss_counts
            ):
                self._record_initial_gauss_state()
            for sp in self.species:
                self._initialize_species_half_step(sp, B_n_tuple, t_n)
            self._initialized_half_step = True

        for inj in self.injectors:
            n_before = inj.species.n_particles
            n_inj = inj.inject(t_n, dt)
            if n_inj > 0:
                sp_inj = inj.species
                sp_inj._ensure_gpu_scratch()
                sl_new = slice(n_before, sp_inj.n_particles)
                xi_i, eta_i, zeta_i = self.grid_map.to_logical(
                    sp_inj._x[sl_new], sp_inj._y[sl_new], sp_inj._z[sl_new]
                )
                deposit_charge_gpu(
                    xi_i,
                    eta_i,
                    zeta_i,
                    sp_inj._w[sl_new],
                    -sp_inj.charge,
                    Nx,
                    Ny,
                    Nz,
                    order=self.shape,
                    periodic=self.boundaries.periodic,
                    q_nodes=s._particles_q_bg,
                )
                s._particles_q_scale_ref = abs(inj.injected_charge)
                for col in self.boundaries.collectors:
                    col.current_history.append((float(t_n + dt), 0.0))

        Wx_tot = self._Wx_buf
        Wy_tot = self._Wy_buf
        Wz_tot = self._Wz_buf
        Wx_tot.fill(0.0)
        Wy_tot.fill(0.0)
        Wz_tot.fill(0.0)

        E_flat = s.E.toarray()
        all_periodic_no_obs = (
            all(self.boundaries.periodic)
            and self.boundaries.solid_mask is None
            and not self.boundaries.collectors
        )
        has_custom_mask_col = any(
            col.mask is not None for col in self.boundaries.collectors
        )

        for sp in self.species:
            n_l = sp._n_launch_bound
            if n_l == 0:
                continue

            if (
                self.sort_every > 0
                and (self.n_particle_steps > 0)
                and (self.n_particle_steps % self.sort_every == 0)
            ):
                sp.sort_by_cell(self.grid_map)

            E_uni, E_arr = self._ext_field_args_gpu(
                self.E_ext, sp._x[:n_l], sp._y[:n_l], sp._z[:n_l], t_n
            )
            B_uni, B_arr = self._ext_field_args_gpu(
                self.B_ext, sp._x[:n_l], sp._y[:n_l], sp._z[:n_l], t_n
            )

            launch_gather_and_push_gpu(
                sp,
                self.grid_map,
                E_flat,
                B_n_flat,
                order=self.shape,
                p_stag=self._p_stag,
                periodic=self.boundaries.periodic,
                dt=dt,
                pusher=self.pusher,
                advance_pos=True,
                E_ext_uniform=E_uni,
                B_ext_uniform=B_uni,
                E_ext_arrays=E_arr,
                B_ext_arrays=B_arr,
            )

            if all_periodic_no_obs:
                xi1_dep = sp._xi1_scratch
                eta1_dep = sp._eta1_scratch
                zeta1_dep = sp._zeta1_scratch
                keep_mask_i32 = None
            elif not has_custom_mask_col:
                xi1_dep, eta1_dep, zeta1_dep, keep_mask_i32 = (
                    launch_boundary_prepare_paths_gpu(
                        self.boundaries,
                        sp,
                        order=self.shape,
                        q_bg_nodes=s._particles_q_bg,
                    )
                )
            else:
                n_act = sp.n_particles
                xi1_dep, eta1_dep, zeta1_dep, keep_mask_b = (
                    self.boundaries.prepare_step_paths(
                        sp,
                        sp._x0_scratch[:n_act],
                        sp._y0_scratch[:n_act],
                        sp._z0_scratch[:n_act],
                        sp._xi0_scratch[:n_act],
                        sp._eta0_scratch[:n_act],
                        sp._zeta0_scratch[:n_act],
                        sp._xi1_scratch[:n_act],
                        sp._eta1_scratch[:n_act],
                        sp._zeta1_scratch[:n_act],
                        t=t_n + dt,
                    )
                )
                sp._keep_mask_scratch[:n_act] = keep_mask_b.astype(np.int32)
                keep_mask_i32 = sp._keep_mask_scratch

            d_n_ptr = sp._d_n_particles if sp._n_dirty_gpu else None
            if self.deposit_scheme == "esirkepov":
                deposit_current_esirkepov_gpu(
                    sp._xi0_scratch,
                    sp._eta0_scratch,
                    sp._zeta0_scratch,
                    xi1_dep,
                    eta1_dep,
                    zeta1_dep,
                    sp._w,
                    sp.charge,
                    Nx,
                    Ny,
                    Nz,
                    order=self.shape,
                    periodic=self.boundaries.periodic,
                    Wx_out=Wx_tot,
                    Wy_out=Wy_tot,
                    Wz_out=Wz_tot,
                    d_n_particles=d_n_ptr,
                    n_launch=n_l,
                    d_cfl_error=self._d_cfl_error,
                )
            else:
                n_act = sp.n_particles
                Wx_n, Wy_n, Wz_n = deposit_current_naive(
                    sp._xi0_scratch[:n_act],
                    sp._eta0_scratch[:n_act],
                    sp._zeta0_scratch[:n_act],
                    xi1_dep[:n_act],
                    eta1_dep[:n_act],
                    zeta1_dep[:n_act],
                    sp.charge * sp._w[:n_act],
                    Nx,
                    Ny,
                    Nz,
                    order=self.shape,
                    periodic=self.boundaries.periodic,
                    xp=xp,
                )
                Wx_tot += Wx_n
                Wy_tot += Wy_n
                Wz_tot += Wz_n

            if keep_mask_i32 is not None:
                sp._compact_async_gpu(keep_mask_i32)
            launch_periodic_wrap_gpu(self.boundaries, sp)

        if self.filter:
            Wx_tot, Wy_tot, Wz_tot = apply_binomial_filter_3d(
                Wx_tot,
                Wy_tot,
                Wz_tot,
                Nx,
                Ny,
                Nz,
                periodic=self.boundaries.periodic,
                compensate=(self.filter != "uncompensated"),
                xp=xp,
            )

        finalize_j_delta_gpu(
            Wx_tot,
            Wy_tot,
            Wz_tot,
            self._itA_arr,
            self.J_particles,
            self._J_prev,
            s.J.array,
            1.0 / dt,
            N,
        )
        self.n_particle_steps += 1

    def _mid_step_hook(self, solver=None) -> None:
        """Execute one self-consistent particle-in-cell step between the H and E field updates."""
        s = self.solver if solver is None else solver
        if self.on_gpu:
            self._mid_step_hook_gpu(s)
            return
        xp = self.xp
        dt = float(s.dt)
        if dt != self._attached_dt:
            self._check_cfl_dt(dt)
        t_n = float(self.n_particle_steps * dt)
        Nx, Ny, Nz = s.Nx, s.Ny, s.Nz
        N = Nx * Ny * Nz

        # Time-centred magnetic flux density B^n = mu_0 * (H^{n-1/2} + H^{n+1/2}) / 2.
        # SolverFIT3D._capture_mid_step_H() copies H^{n-1/2} into s._H_before_update
        # once per step prior to _update_H(), avoiding a second H copy here.
        do_profile = self.profile
        t_pg_acc = 0.0
        t_dep_acc = 0.0
        if do_profile:
            t0_b = time.perf_counter()
        H_curr = s.H.toarray()
        H_prev = getattr(s, "_H_before_update", None)
        if H_prev is None:
            H_prev = self._H_prev if self._H_prev is not None else H_curr
            B_n_flat = self._compute_centred_B(H_prev, H_curr)
            if self._H_prev is None or self._H_prev.shape != H_curr.shape:
                self._H_prev = H_curr.copy()
            else:
                np.copyto(self._H_prev, H_curr)
        else:
            B_n_flat = self._compute_centred_B(H_prev, H_curr)
        B_n_tuple = (B_n_flat[:N], B_n_flat[N : 2 * N], B_n_flat[2 * N : 3 * N])
        if do_profile:
            t_pg_acc += time.perf_counter() - t0_b

        if not self._initialized_half_step:
            if (
                tuple(sp._n_launch_bound for sp in self.species)
                != self._init_gauss_counts
            ):
                self._record_initial_gauss_state()
            for sp in self.species:
                self._initialize_species_half_step(sp, B_n_tuple, t_n)
            self._initialized_half_step = True

        # Inject new macroparticles at emitter planes before the step advance.
        # Record the equal-and-opposite emitter image charge in _particles_q_bg so
        # discrete Gauss's law accounts for charge emitted into the domain.
        for inj in self.injectors:
            n_before = inj.species.n_particles
            n_inj = inj.inject(t_n, dt)
            if n_inj > 0:
                sp_inj = inj.species
                sl_new = slice(n_before, sp_inj.n_particles)
                xi_i, eta_i, zeta_i = self.grid_map.to_logical(
                    sp_inj.x[sl_new], sp_inj.y[sl_new], sp_inj.z[sl_new]
                )
                q_inj_nodes = xp.zeros((Nx, Ny, Nz), dtype=np.float64)
                deposit_charge(
                    xi_i,
                    eta_i,
                    zeta_i,
                    sp_inj.charge * sp_inj.w[sl_new],
                    Nx,
                    Ny,
                    Nz,
                    order=self.shape,
                    periodic=self.boundaries.periodic,
                    q_nodes=q_inj_nodes,
                    xp=xp,
                )
                # Record fixed opposite charge -q_inj in _particles_q_bg at the
                # injection plane z0. When z0 lies on a PEC boundary face (e.g.
                # zmin with bc="pec"), this represents the physical wall image
                # charge where tangential E = 0; if z0 lies inside the vacuum
                # interior, -q_inj remains at z0 as a fixed immobile space-charge
                # sheet balancing the emitted beam charge.
                s._particles_q_bg = s._particles_q_bg - q_inj_nodes
                if getattr(s, "_particles_q_scale_ref", 0.0) == 0.0:
                    s._particles_q_scale_ref = abs(inj.injected_charge)

        if do_profile:
            t0_zero = time.perf_counter()
        if self._j_in_solver:
            s.J.array -= self.J_particles
        Wx_tot = self._Wx_buf
        Wy_tot = self._Wy_buf
        Wz_tot = self._Wz_buf
        Wx_tot.fill(0.0)
        Wy_tot.fill(0.0)
        Wz_tot.fill(0.0)
        if do_profile:
            t_dep_acc += time.perf_counter() - t0_zero

        for sp in self.species:
            if sp.n_particles == 0:
                continue

            if self.sort_every > 0 and (self.n_particle_steps > 0) and (
                self.n_particle_steps % self.sort_every == 0
            ):
                sp.sort_by_cell(self.grid_map)

            if (
                xp is np
                and sp.n_particles > _CPU_CHUNK
                and all(self.boundaries.periodic)
                and self.boundaries.solid_mask is None
                and not self.boundaries.collectors
                and self.deposit_scheme == "esirkepov"
            ):
                from .pushers import _INV_C2, get_pusher

                fn = get_pusher(self.pusher)
                E_tup = (s.E.field_x, s.E.field_y, s.E.field_z)
                n_p = sp.n_particles
                for s0, s1 in _cpu_chunk_ranges(n_p):
                    if do_profile:
                        t_c0 = time.perf_counter()
                    x0_c = sp._x[s0:s1].copy()
                    y0_c = sp._y[s0:s1].copy()
                    z0_c = sp._z[s0:s1].copy()
                    xi0_c, eta0_c, zeta0_c = self.grid_map.to_logical(x0_c, y0_c, z0_c)
                    Ex_c, Ey_c, Ez_c, Bx_c, By_c, Bz_c = gather_fields(
                        xi0_c,
                        eta0_c,
                        zeta0_c,
                        E_tup,
                        B_n_tuple,
                        Nx,
                        Ny,
                        Nz,
                        order=self.shape,
                        gather=self.gather,
                        periodic=self.boundaries.periodic,
                        xp=xp,
                    )
                    Ex_e, Ey_e, Ez_e = self._eval_ext_field(
                        self.E_ext, x0_c, y0_c, z0_c, t_n
                    )
                    Bx_e, By_e, Bz_e = self._eval_ext_field(
                        self.B_ext, x0_c, y0_c, z0_c, t_n
                    )
                    ux_c, uy_c, uz_c = fn(
                        sp._ux[s0:s1],
                        sp._uy[s0:s1],
                        sp._uz[s0:s1],
                        Ex_c + Ex_e,
                        Ey_c + Ey_e,
                        Ez_c + Ez_e,
                        Bx_c + Bx_e,
                        By_c + By_e,
                        Bz_c + Bz_e,
                        sp.charge,
                        sp.mass,
                        dt,
                        xp=xp,
                    )
                    sp._ux[s0:s1] = ux_c
                    sp._uy[s0:s1] = uy_c
                    sp._uz[s0:s1] = uz_c
                    fac = dt / np.sqrt(
                        1.0 + (ux_c * ux_c + uy_c * uy_c + uz_c * uz_c) * _INV_C2
                    )
                    sp._x[s0:s1] = x0_c + fac * ux_c
                    sp._y[s0:s1] = y0_c + fac * uy_c
                    sp._z[s0:s1] = z0_c + fac * uz_c
                    xi1_c, eta1_c, zeta1_c = self.grid_map.to_logical(
                        sp._x[s0:s1], sp._y[s0:s1], sp._z[s0:s1]
                    )
                    del (
                        x0_c,
                        y0_c,
                        z0_c,
                        Ex_c,
                        Ey_c,
                        Ez_c,
                        Bx_c,
                        By_c,
                        Bz_c,
                        ux_c,
                        uy_c,
                        uz_c,
                        fac,
                    )
                    if do_profile:
                        t_c1 = time.perf_counter()
                        t_pg_acc += t_c1 - t_c0

                    deposit_current_esirkepov(
                        xi0_c,
                        eta0_c,
                        zeta0_c,
                        xi1_c,
                        eta1_c,
                        zeta1_c,
                        sp.charge * sp._w[s0:s1],
                        Nx,
                        Ny,
                        Nz,
                        order=self.shape,
                        periodic=self.boundaries.periodic,
                        Wx_out=Wx_tot,
                        Wy_out=Wy_tot,
                        Wz_out=Wz_tot,
                        xp=xp,
                    )
                    if do_profile:
                        t_dep_acc += time.perf_counter() - t_c1
                if do_profile:
                    t_w0 = time.perf_counter()
                self.boundaries.apply_periodic_wrap(sp)
                if do_profile:
                    t_dep_acc += time.perf_counter() - t_w0
                continue

            if do_profile:
                t_g0 = time.perf_counter()
            x0 = sp.x.copy()
            y0 = sp.y.copy()
            z0 = sp.z.copy()
            xi0, eta0, zeta0 = self.grid_map.to_logical(x0, y0, z0)

            # 1. Gather E^n and B^n at x^n
            Ex, Ey, Ez, Bx, By, Bz = gather_fields(
                xi0,
                eta0,
                zeta0,
                s.E,
                B_n_tuple,
                Nx,
                Ny,
                Nz,
                order=self.shape,
                gather=self.gather,
                periodic=self.boundaries.periodic,
                xp=xp,
            )
            Ex_e, Ey_e, Ez_e = self._eval_ext_field(self.E_ext, x0, y0, z0, t_n)
            Bx_e, By_e, Bz_e = self._eval_ext_field(self.B_ext, x0, y0, z0, t_n)

            # 2. Push u^{n-1/2} -> u^{n+1/2}
            push_momentum(
                sp,
                Ex + Ex_e,
                Ey + Ey_e,
                Ez + Ez_e,
                Bx + Bx_e,
                By + By_e,
                Bz + Bz_e,
                dt=dt,
                pusher=self.pusher,
            )

            # 3. Advance x^{n+1} = x^n + dt * u^{n+1/2} / gamma^{n+1/2}
            push_position(sp, dt=dt)
            xi1, eta1, zeta1 = self.grid_map.to_logical(sp.x, sp.y, sp.z)
            if do_profile:
                t_g1 = time.perf_counter()
                t_pg_acc += t_g1 - t_g0

            # 4. Boundary truncation (absorb at domain walls / CPML / PEC / collectors)
            xi1_dep, eta1_dep, zeta1_dep, keep_mask = (
                self.boundaries.prepare_step_paths(
                    sp, x0, y0, z0, xi0, eta0, zeta0, xi1, eta1, zeta1, t=t_n + dt
                )
            )

            # 5. Deposit dual-face charges W from xi0 -> xi1_dep
            qw = sp.charge * sp.w
            if self.deposit_scheme == "esirkepov":
                deposit_current_esirkepov(
                    xi0,
                    eta0,
                    zeta0,
                    xi1_dep,
                    eta1_dep,
                    zeta1_dep,
                    qw,
                    Nx,
                    Ny,
                    Nz,
                    order=self.shape,
                    periodic=self.boundaries.periodic,
                    Wx_out=Wx_tot,
                    Wy_out=Wy_tot,
                    Wz_out=Wz_tot,
                    xp=xp,
                )
            else:
                Wx_n, Wy_n, Wz_n = deposit_current_naive(
                    xi0,
                    eta0,
                    zeta0,
                    xi1_dep,
                    eta1_dep,
                    zeta1_dep,
                    qw,
                    Nx,
                    Ny,
                    Nz,
                    order=self.shape,
                    periodic=self.boundaries.periodic,
                    xp=xp,
                )
                Wx_tot += Wx_n
                Wy_tot += Wy_n
                Wz_tot += Wz_n

            # 6. Remove absorbed particles (recording their deposited surface
            # charge at the absorption point in _particles_q_bg). Because the
            # Esirkepov path stops at xi1_dep and the particle is removed from
            # the active list, adding +q_abs to _particles_q_bg represents the
            # collected charge remaining as fixed space charge at the wall or
            # collector surface.
            absorbed_mask = ~keep_mask
            if bool(xp.any(absorbed_mask)):
                q_abs_nodes = xp.zeros((Nx, Ny, Nz), dtype=np.float64)
                deposit_charge(
                    xi1_dep[absorbed_mask],
                    eta1_dep[absorbed_mask],
                    zeta1_dep[absorbed_mask],
                    qw[absorbed_mask],
                    Nx,
                    Ny,
                    Nz,
                    order=self.shape,
                    periodic=self.boundaries.periodic,
                    q_nodes=q_abs_nodes,
                    xp=xp,
                )
                s._particles_q_bg = s._particles_q_bg + q_abs_nodes
            sp.keep(keep_mask)
            self.boundaries.apply_periodic_wrap(sp)
            if do_profile:
                t_dep_acc += time.perf_counter() - t_g1

        if do_profile:
            t_fin0 = time.perf_counter()
        if self.filter:
            Wx_tot, Wy_tot, Wz_tot = apply_binomial_filter_3d(
                Wx_tot,
                Wy_tot,
                Wz_tot,
                Nx,
                Ny,
                Nz,
                periodic=self.boundaries.periodic,
                compensate=(self.filter != "uncompensated"),
                xp=xp,
            )

        # Convert dual-face charge W_e [C] to primal-edge current density
        # J_e = W_e / (dt * A_tilde_e) = W_e * itA_e / dt [A/m^2]
        inv_dt = 1.0 / dt
        itA_arr = self._itA_arr
        if self.filter:
            self.J_particles[:N] = Wx_tot
            self.J_particles[N : 2 * N] = Wy_tot
            self.J_particles[2 * N : 3 * N] = Wz_tot
        self.J_particles *= itA_arr
        self.J_particles *= inv_dt
        s.J.array += self.J_particles
        self._j_in_solver = True
        if do_profile:
            t_dep_acc += time.perf_counter() - t_fin0
            self._last_push_gather_s = t_pg_acc
            self._last_deposit_s = t_dep_acc
        else:
            self._last_push_gather_s = 0.0
            self._last_deposit_s = 0.0
        self.n_particle_steps += 1

    # --- Diagnostics ---------------------------------------------------------

    def kinetic_energy(self, synchronised: bool = False) -> float:
        """Total relativistic kinetic energy [J] across all species.

        Parameters
        ----------
        synchronised : bool, optional
            If False (default), evaluates kinetic energy directly from the stored
            half-step proper velocities ``u^{n-1/2}``.
            If True, half-pushes ``u^{n-1/2} -> u^n`` by ``+dt / 2`` on a
            temporary copy using ``E^n`` and the time-centred magnetic field
            ``B^n = 0.5 * mu_0 * (H^{n-1/2} + H^{n+1/2})`` (where
            ``H^{n+1/2} = H^{n-1/2} - dt * tDsiDmuiDaC * E^n`` from Faraday's
            law when called between full ``one_step()`` calls) so that the
            evaluated kinetic energy is second-order time-centred at ``t_n``.
            Note: the diagnostic Faraday half-step ``H^{n+1/2}`` ignores CPML
            convolutional ``psi`` terms and magnetic current sources ``M``.
        """
        if not synchronised:
            return float(sum(sp.kinetic_energy() for sp in self.species))

        s = self.solver
        Nx, Ny, Nz = s.Nx, s.Ny, s.Nz
        N = Nx * Ny * Nz
        H_curr = s.H.toarray()
        if hasattr(s, "tDsiDmuiDaC") and self.n_particle_steps > 0:
            H_next = H_curr - s.dt * (s.tDsiDmuiDaC * s.E.toarray())
        else:
            H_next = H_curr
        B_n_flat = self._compute_centred_B(H_curr, H_next)
        B_n_tuple = (B_n_flat[:N], B_n_flat[N : 2 * N], B_n_flat[2 * N : 3 * N])
        t_n = float(self.n_particle_steps * s.dt)
        fn = get_pusher(self.pusher)

        ke_tot = 0.0
        for sp in self.species:
            if sp.n_particles == 0:
                continue
            if sp.u_is_synced:
                ke_tot += sp.kinetic_energy()
                continue
            xi, eta, zeta = self.grid_map.to_logical(sp.x, sp.y, sp.z)
            Ex, Ey, Ez, Bx, By, Bz = gather_fields(
                xi,
                eta,
                zeta,
                s.E,
                B_n_tuple,
                Nx,
                Ny,
                Nz,
                order=self.shape,
                gather=self.gather,
                periodic=self.boundaries.periodic,
                xp=self.xp,
            )
            Ex_e, Ey_e, Ez_e = self._eval_ext_field(self.E_ext, sp.x, sp.y, sp.z, t_n)
            Bx_e, By_e, Bz_e = self._eval_ext_field(self.B_ext, sp.x, sp.y, sp.z, t_n)
            ux_s, uy_s, uz_s = fn(
                sp.ux,
                sp.uy,
                sp.uz,
                Ex + Ex_e,
                Ey + Ey_e,
                Ez + Ez_e,
                Bx + Bx_e,
                By + By_e,
                Bz + Bz_e,
                sp.charge,
                sp.mass,
                0.5 * s.dt,
                xp=self.xp,
            )
            u2 = ux_s * ux_s + uy_s * uy_s + uz_s * uz_s
            from scipy.constants import c as c_light

            g = self.xp.sqrt(1.0 + u2 / (c_light * c_light))
            ke_tot += float(self.xp.sum(sp.w * sp.mass * (u2 / (g + 1.0))))
        return ke_tot

    def total_charge(self) -> float:
        """Total active macroparticle charge [C] on the grid across all species."""
        return float(sum(sp.total_charge() for sp in self.species))

    def absorbed_charge(self) -> float:
        """Total charge [C] absorbed by all boundaries and collectors."""
        return self.boundaries.total_absorbed_charge()

    def absorbed_energy(self) -> float:
        """Total kinetic energy [J] absorbed by all boundaries and collectors."""
        return self.boundaries.total_absorbed_energy()

    def injected_charge(self) -> float:
        """Total charge [C] injected by all attached :class:`BeamInjector` instances."""
        return float(sum(inj.injected_charge for inj in self.injectors))

    def injected_energy(self) -> float:
        """Total kinetic energy [J] injected by all attached :class:`BeamInjector` instances."""
        return float(sum(inj.injected_energy for inj in self.injectors))

    def field_energy(self) -> float:
        """Return total electromagnetic field energy [J] from ``solver.field_energy()``."""
        return float(self.solver.field_energy())

    def gauss_residual(self) -> float:
        """Return relative discrete Gauss's law residual ``max|S_tilde d_tilde - q| / max|q|``."""
        return gauss_residual(self.solver, self.species, order=self.shape)
