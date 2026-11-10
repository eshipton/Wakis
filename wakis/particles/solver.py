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
from scipy.constants import c as c_light
from scipy.constants import mu_0

from .boundaries import Collector, ParticleBoundaries
from .mapping import LogicalGridMap
from .pushers import get_pusher, push_momentum, push_position
from .shapes import (
    ParticleCFLError,
    _gather_component_3d,
    bspline_weights_1d,
    deposit_charge,
    deposit_current_esirkepov,
    deposit_current_naive,
    gather_fields,
)
from .species import Species

_CPU_CHUNK = 65536
_ENABLE_PEC_WALL_CLOSURE: bool = True
_ENABLE_PEC_WALL_KICK: bool = True


def _pec_wall_closure_1d(
    q_vis,
    axis: int,
    side: str,
    k_w: int | None = None,
    *,
    event: str = "inject",
    xp=np,
):
    """Compute the 1D normal-axis PEC wall-closure edge charge flux ``dW_n`` [C].

    Moves all visible stencil charge ``q_vis`` of a particle creation or
    absorption event on a PEC domain wall onto the wall node ``k_w`` along
    normal ``axis`` via a 1D telescoping cumulative sum applied column by
    column in the two transverse indices:

    - Low face (``k_w = 0``), injection:
      ``dW_n[k] = +sum_{k' > k} q_vis[k']`` for ``k >= 0``
      (and ``-sum_{k' > k} q_vis[k']`` for absorption).
    - High face (``k_w = N_n``), absorption:
      ``dW_n[k] = +sum_{k' <= k} q_vis[k']`` for ``0 <= k < k_w``
      (and ``-sum_{k' <= k} q_vis[k']`` for injection).

    Parameters
    ----------
    q_vis : ndarray of float, shape (Nx, Ny, Nz)
        Order-``p`` nodal charge deposit of the event's particles [C].
    axis : {0, 1, 2}
        Wall normal axis index (``0`` for ``x``, ``1`` for ``y``, ``2`` for ``z``).
    side : {"low", "high"}
        Domain face side along ``axis``.
    k_w : int or None, optional
        Wall node index along ``axis`` (``0`` for ``"low"``, ``N_n`` for
        ``"high"``). Defaults to ``0`` for ``"low"`` and
        ``q_vis.shape[axis] - 1`` for ``"high"``.
    event : {"inject", "absorb"}, optional
        Event type (``"inject"`` or ``"absorb"``). Default is ``"inject"``.
    xp : module, optional
        Array module (``numpy`` or ``cupy``). Default is ``numpy``.

    Returns
    -------
    dW_n : ndarray of float, shape (Nx, Ny, Nz)
        Normal-axis edge charge flux increment [C] to add to ``W_n``.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"Unsupported axis={axis!r}; expected 0, 1, or 2.")
    if side not in ("low", "high"):
        raise ValueError(f"Unsupported side={side!r}; expected 'low' or 'high'.")
    if event not in ("inject", "absorb"):
        raise ValueError(
            f"Unsupported event={event!r}; expected 'inject' or 'absorb'."
        )

    dW_n = xp.zeros_like(q_vis)
    if not _ENABLE_PEC_WALL_CLOSURE:
        return dW_n

    n_ax = int(q_vis.shape[axis])
    if side == "low":
        k_wall = 0 if k_w is None else int(k_w)
        if k_wall != 0:
            raise ValueError(f"Low-face wall node index k_w must be 0, got {k_wall}.")
        if n_ax > 1:
            sl_src = [slice(None), slice(None), slice(None)]
            sl_dst = [slice(None), slice(None), slice(None)]
            sl_src[axis] = slice(1, n_ax)
            sl_dst[axis] = slice(0, n_ax - 1)
            tail = xp.flip(
                xp.cumsum(xp.flip(q_vis[tuple(sl_src)], axis=axis), axis=axis),
                axis=axis,
            )
            dW_n[tuple(sl_dst)] = tail if event == "inject" else (0.0 - tail)
        return dW_n

    k_wall = (n_ax - 1) if k_w is None else int(k_w)
    if k_wall < 0 or k_wall >= n_ax:
        raise ValueError(
            f"High-face wall node index k_w={k_wall} out of bounds for size {n_ax}."
        )
    if k_wall > 0:
        sl_head = [slice(None), slice(None), slice(None)]
        sl_head[axis] = slice(0, k_wall)
        head = xp.cumsum(q_vis[tuple(sl_head)], axis=axis)
        dW_n[tuple(sl_head)] = head if event == "absorb" else (0.0 - head)
    return dW_n


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


def _interior_vacuum_node_mask(solver, include_wall_planes: bool = False):
    """Boolean mask of primal nodes ``(i, j, k)`` strictly in the active vacuum region.

    Parameters
    ----------
    solver : SolverFIT3D
        Electromagnetic FIT solver instance.
    include_wall_planes : bool, optional
        If False (default), skip a B-spline stencil margin ``(p + 1) // 2 + 1``
        next to CPML inner faces; non-CPML faces (such as PEC domain walls)
        already evaluate all interior vacuum node planes ``1..N-1`` by default.
        If True, shrink the CPML margin to ``1`` node.
    """
    xp = getattr(solver.E, "xp", np)
    Nx, Ny, Nz = solver.Nx, solver.Ny, solver.Nz
    mask = xp.ones((Nx, Ny, Nz), dtype=bool)

    bc_low = getattr(solver, "bc_low", ("periodic", "periodic", "periodic"))
    bc_high = getattr(solver, "bc_high", ("periodic", "periodic", "periodic"))
    n_pml = int(getattr(solver, "n_pml", 10))
    shape_order = int(getattr(solver, "_particles_shape_order", 1))
    pad = 1 if include_wall_planes else ((shape_order + 1) // 2 + 1)

    for ax, n_ax in enumerate((Nx, Ny, Nz)):
        low_s = str(bc_low[ax]).strip().lower()
        high_s = str(bc_high[ax]).strip().lower()
        is_per = low_s in ("periodic", "bloch") and high_s in ("periodic", "bloch")
        if is_per:
            continue

        # On any non-periodic axis, SolverFIT3D places the high-side boundary
        # node at index n_ax - 1 (both when boundary_at_domain_edge=True and False).
        n_phys = n_ax - 1
        i_start = (n_pml + pad) if low_s in ("pml", "cpml") else 1
        if high_s == "mpi":
            i_end = n_ax - 1
        elif high_s in ("pml", "cpml"):
            i_end = n_phys - n_pml - pad + 1
        else:
            i_end = n_phys

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


def _mpi_sum_node_guards(solver, q_nodes) -> None:
    """Add MPI ghost node-charge layers of ``q_nodes`` into neighbour owned layers."""
    if not getattr(solver, "use_mpi", False) or getattr(solver, "size", 1) <= 1:
        return
    from mpi4py import MPI

    comm = solver.comm
    rank = int(solver.rank)
    size = int(solver.size)
    k_own_start = 1 if rank > 0 else 0
    k_own_end = k_own_start + int(solver.NZ // size)
    is_gpu = hasattr(q_nodes, "get")
    q_h = q_nodes.get() if is_gpu else np.asarray(q_nodes)
    nx, ny, nz_buf = q_h.shape

    reqs = []
    recv_from_lo = None
    recv_from_hi = None
    if rank > 0:
        send_lo = np.ascontiguousarray(q_h[:, :, 0])
        recv_from_lo = np.empty((nx, ny), dtype=q_h.dtype)
        reqs.append(comm.Irecv(recv_from_lo, source=rank - 1, tag=30))
        reqs.append(comm.Isend(send_lo, dest=rank - 1, tag=31))
    if rank < size - 1:
        send_hi = np.ascontiguousarray(q_h[:, :, k_own_end])
        recv_from_hi = np.empty((nx, ny), dtype=q_h.dtype)
        reqs.append(comm.Irecv(recv_from_hi, source=rank + 1, tag=31))
        reqs.append(comm.Isend(send_hi, dest=rank + 1, tag=30))
    if reqs:
        MPI.Request.Waitall(reqs)
    if recv_from_hi is not None:
        q_h[:, :, k_own_end - 1] += recv_from_hi
    if recv_from_lo is not None:
        q_h[:, :, 1] += recv_from_lo

    reqs2 = []
    recv_from_lo2 = None
    if rank > 0:
        recv_from_lo2 = np.empty((nx, ny), dtype=q_h.dtype)
        reqs2.append(comm.Irecv(recv_from_lo2, source=rank - 1, tag=32))
    if rank < size - 1:
        if nz_buf > k_own_end + 1:
            send_hi2 = np.ascontiguousarray(q_h[:, :, k_own_end + 1])
        else:
            send_hi2 = np.zeros((nx, ny), dtype=q_h.dtype)
        reqs2.append(comm.Isend(send_hi2, dest=rank + 1, tag=32))
    if reqs2:
        MPI.Request.Waitall(reqs2)
    if recv_from_lo2 is not None and q_h.shape[2] > 2:
        q_h[:, :, 2] += recv_from_lo2

    if is_gpu:
        if hasattr(q_nodes, "set"):
            q_nodes.set(q_h)
        else:
            q_nodes[...] = solver.E.xp.asarray(q_h)


def gauss_residual(
    solver,
    species,
    order: int | None = None,
    include_wall_planes: bool = False,
) -> float:
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
    include_wall_planes : bool, optional
        If False (default), skip a B-spline stencil margin ``(p + 1) // 2 + 1``
        next to CPML inner interfaces; non-CPML faces (such as PEC domain walls)
        already evaluate all interior vacuum node planes ``1..N-1`` by default.
        If True, shrink the CPML margin to ``1`` node.

    Returns
    -------
    float
        Dimensionless max-norm Gauss residual in the interior vacuum region.
    """
    xp = getattr(solver.E, "xp", np)
    ps_att = getattr(solver, "_particle_solver", None)
    if ps_att is not None and getattr(ps_att, "_d_cfl_error", None) is not None:
        cfl_err = int(ps_att._d_cfl_error[0].get())
        if getattr(solver, "use_mpi", False) and getattr(solver, "size", 1) > 1:
            from mpi4py import MPI

            cfl_err = int(solver.comm.allreduce(cfl_err, op=MPI.MAX))
        if cfl_err > 0:
            raise ParticleCFLError(
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
        wi = getattr(ps_att, "_wall_images", None) if ps_att is not None else None
        if wi is not None:
            wi.deposit_charge(
                xi, eta, zeta, qw, solver.Nx, solver.Ny, solver.Nz, periodic, q_nodes, xp=xp
            )

    if getattr(solver, "use_mpi", False) and getattr(solver, "size", 1) > 1:
        _mpi_sum_node_guards(solver, q_nodes)

    div_d = compute_dual_divergence_d(solver)
    q_bg = getattr(solver, "_particles_q_bg", None)
    if q_bg is not None:
        if getattr(solver, "use_mpi", False) and getattr(solver, "size", 1) > 1:
            q_bg_sum = q_bg.copy()
            _mpi_sum_node_guards(solver, q_bg_sum)
            q_tot = q_nodes[:, :, : solver.Nz] + q_bg_sum[:, :, : solver.Nz]
        else:
            q_tot = q_nodes + q_bg[:, :, : solver.Nz]
    else:
        q_tot = q_nodes[:, :, : solver.Nz]

    if include_wall_planes:
        vac_mask = _interior_vacuum_node_mask(solver, include_wall_planes=True)
    else:
        vac_mask = _interior_vacuum_node_mask(solver)
    if not bool(xp.any(vac_mask)):
        raise ValueError(
            "gauss_residual: no interior vacuum nodes outside the CPML and the shape-stencil "
            "margin; enlarge the domain (the residual on CPML nodes is not a conservation check)."
        )

    res = xp.abs((div_d - q_tot)[vac_mask])
    max_res = float(xp.max(res)) if res.size > 0 else 0.0

    q_scale = (
        float(xp.max(xp.abs(q_nodes[:, :, : solver.Nz][vac_mask])))
        if res.size > 0
        else 0.0
    )
    q_ref = float(getattr(solver, "_particles_q_scale_ref", 0.0))
    denom = max(q_scale, q_ref, 1e-30)
    if getattr(solver, "use_mpi", False) and getattr(solver, "size", 1) > 1:
        from mpi4py import MPI

        max_res = float(solver.comm.allreduce(max_res, op=MPI.MAX))
        denom = float(solver.comm.allreduce(denom, op=MPI.MAX))
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


class _CollectiveMemoryError(RuntimeError, MemoryError):
    """Out-of-memory error raised collectively across all MPI ranks (is both a RuntimeError and a MemoryError)."""


class _EnergyTrackingDisabledError(RuntimeError, ValueError):
    """Raised when energy-balance diagnostics are queried with ``track_energy=False``."""


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
        If True, record per-step stage wall-clock times
        (``_last_push_gather_s``, ``_last_deposit_s``, ``_last_guard_reduce_s``,
        and ``_last_migrate_s``) via :func:`time.perf_counter`. Default is
        ``False`` (zero timing calls per step).
    particles_comm : {"auto", "nccl", "mpi", "none"} or None, optional
        Multi-GPU communication backend when ``solver.use_mpi`` and
        ``solver.use_gpu`` are both True. ``"nccl"`` uses direct GPU-to-GPU
        ``cupy.cuda.nccl`` point-to-point exchanges and raises a collective
        ``RuntimeError`` if NCCL is unavailable; ``"mpi"`` stages through
        pinned host buffers; ``"none"`` is used when MPI is disabled or
        ``solver.size == 1``. When ``None`` or ``"auto"`` (default), ``"nccl"``
        is selected collectively if all ranks have a valid NCCL communicator,
        falling back to ``"mpi"`` with a ``RuntimeWarning`` when unavailable
        (or ``"none"`` when ``not solver.use_mpi or solver.size == 1``).
    capacity_headroom : float or None, optional
        Fractional buffer capacity headroom reserved at :meth:`attach` (e.g.
        ``0.05`` reserves ``max(ceil(1.05 * n_particles), n_particles + 64)``
        macroparticle slots) so incoming MPI-migrated particles do not trigger
        mid-step buffer reallocations. Defaults to ``0.05`` when
        ``use_mpi=True`` and ``solver.size > 1``, and ``0.0`` otherwise.
    track_energy : bool, optional
        If True, allocate discrete FIT energy-balance weight and history buffers
        and accumulate per-step field work so :meth:`field_work` and
        :meth:`energy_balance` can be evaluated. Default is ``False`` (zero extra
        buffer allocations and zero per-step energy-tracking passes, leaving the
        default time-stepping path bitwise unchanged). When ``track_energy=True``,
        the memory cost per grid cell is
        :math:`\\text{cost} = 9 \\times 3 \\times \\text{sizeof}(\\text{float64}) = 27 \\times 8 = 216\\;\\text{bytes/cell}`
        across the nine ``3 * Nx * Ny * Nz`` ``float64`` edge arrays
        (``_m_eps_arr``, ``_m_mu_arr``, ``_vol_E_fw``, ``_E_prev_fw``,
        ``_fw_scratch``, ``_fw_accum_vec``, ``_E_init_fw``, ``_H_prev_init_fw``,
        ``_H_curr_init_fw``), and ``0`` bytes/cell when ``track_energy=False``.

    Attributes
    ----------
    n_particle_steps : int
        Public counter of completed self-consistent particle steps. Used to seed
        each :class:`BeamInjector`'s deterministic per-step :class:`numpy.random.SeedSequence`
        and exposed for checkpoint/restart persistence.

    Notes
    -----
    Under MPI domain decomposition along ``z`` (``use_mpi=True``):

    - Only ``shape=1`` (CIC) and ``filter=None`` are supported because
      ``GridFIT3D`` provides a single 1-cell MPI ghost layer.
    - The global cell count ``Nz`` must be divisible by ``solver.size`` (so
      ``GridFIT3D`` does not truncate ``NZ = Nz - Nz % size``) and must satisfy
      ``solver.NZ // solver.size >= 2`` so that the two high Esirkepov guard
      planes ``k_own_end`` and ``k_own_end + 1`` land inside the upper
      neighbour's owned slab ``[1, 1 + NZ // size)``.
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
        particles_comm: str | None = None,
        capacity_headroom: float | None = None,
        track_energy: bool = False,
        wall_stencil: str = "masked",
    ):
        self.solver = solver
        self.track_energy = bool(track_energy)
        self.use_mpi = bool(getattr(solver, "use_mpi", False))
        if self.use_mpi and not bool(getattr(solver.grid, "use_mpi", False)):
            raise ValueError(
                "SolverFIT3D has use_mpi=True, but grid.use_mpi is False; "
                "set use_mpi=True on GridFIT3D as well."
            )
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

        mpi_multi = self.use_mpi and int(getattr(solver, "size", 1)) > 1
        if capacity_headroom is None:
            self.capacity_headroom = 0.05 if mpi_multi else 0.0
        else:
            if float(capacity_headroom) < 0.0:
                raise ValueError(
                    f"capacity_headroom must be >= 0, got {capacity_headroom}."
                )
            self.capacity_headroom = float(capacity_headroom)

        pc_norm = (
            "auto"
            if particles_comm is None
            else str(particles_comm).strip().lower()
        )
        if pc_norm not in ("auto", "nccl", "mpi", "none"):
            raise ValueError(
                f"Unknown particles_comm={particles_comm!r}; expected 'auto', 'nccl', 'mpi', or 'none'."
            )
        if pc_norm == "auto":
            if not mpi_multi:
                self.particles_comm = "none"
            elif self.on_gpu:
                nccl_comm = (
                    solver._ensure_nccl_comm()
                    if hasattr(solver, "_ensure_nccl_comm")
                    else None
                )
                if nccl_comm is not None:
                    self.particles_comm = "nccl"
                else:
                    import warnings

                    warnings.warn(
                        "NCCL communicator is unavailable across all MPI ranks; "
                        "falling back to particles_comm='mpi'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self.particles_comm = "mpi"
            else:
                self.particles_comm = "mpi"
        elif pc_norm == "nccl":
            nccl_comm = (
                solver._ensure_nccl_comm()
                if (mpi_multi and self.on_gpu and hasattr(solver, "_ensure_nccl_comm"))
                else None
            )
            if nccl_comm is None:
                raise RuntimeError(
                    "particles_comm='nccl' was requested explicitly, but a collective "
                    "NCCL communicator is unavailable across all MPI ranks "
                    "(NCCL requires use_gpu=True and one rank per GPU on each node, "
                    "local_size <= n_gpus; use particles_comm='mpi' when ranks share GPUs)."
                )
            self.particles_comm = "nccl"
        elif pc_norm == "mpi":
            self.particles_comm = "mpi" if mpi_multi else "none"
        else:
            if mpi_multi:
                raise ValueError(
                    "particles_comm='none' is only valid when use_mpi=False or solver.size == 1."
                )
            self.particles_comm = "none"

        if self.use_mpi:
            solver.particles_comm = self.particles_comm

        if int(shape) not in (1, 2, 3):
            raise ValueError(f"Unsupported shape={shape!r}; expected 1, 2, or 3.")
        self.shape = int(shape)
        if self.use_mpi and self.shape != 1:
            n_req_lo = self.shape // 2 + 1
            n_req_hi = (self.shape + 1) // 2 + 1
            raise ValueError(
                f"ParticleSolver with use_mpi=True supports only shape=1 (CIC) because "
                f"shape={self.shape} requires {n_req_lo} low and {n_req_hi} high guard layers, "
                f"while GridFIT3D provides 1 MPI ghost layer."
            )
        if self.use_mpi and bool(filter):
            raise NotImplementedError(
                "Binomial current filtering (filter != None) is not supported with "
                "use_mpi=True because the two-pass compensated stencil requires two "
                "halo z-planes whereas GridFIT3D provides 1 MPI ghost layer."
            )
        if self.use_mpi:
            mpi_size = int(getattr(solver, "size", 1))
            nz_global = int(getattr(solver, "NZ", solver.Nz))
            nz_input = int(getattr(solver.grid, "_Nz_input", nz_global))
            if nz_input % mpi_size != 0:
                raise ValueError(
                    f"ParticleSolver with use_mpi=True requires global Nz ({nz_input}) "
                    f"to be divisible by solver.size ({mpi_size}) so GridFIT3D does not "
                    f"truncate the z domain (NZ={nz_global})."
                )
            if nz_global // mpi_size < 2:
                raise ValueError(
                    f"ParticleSolver with use_mpi=True requires at least 2 owned "
                    f"z-cells per MPI rank (solver.NZ // solver.size >= 2) so the "
                    f"two high Esirkepov guard planes land inside the neighbour's "
                    f"owned slab, got NZ={nz_global} and size={mpi_size}."
                )

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
        self._has_pec_domain_wall = self._compute_has_pec_domain_wall()
        solver._particles_solid_mask = self.boundaries.solid_mask

        ws_norm = str(wall_stencil).strip().lower()
        if ws_norm not in ("masked", "image"):
            raise ValueError(
                f"Unknown wall_stencil={wall_stencil!r}; expected 'masked' or 'image'."
            )
        self.wall_stencil = ws_norm
        self._wall_images = None
        if ws_norm == "image":
            if self.use_mpi and int(getattr(solver, "size", 1)) > 1:
                raise NotImplementedError(
                    "wall_stencil='image' is not implemented with MPI domain decomposition."
                )
            if self.deposit_scheme != "esirkepov":
                raise ValueError("wall_stencil='image' requires deposit_scheme='esirkepov'.")
            from .wall_images import WallImages

            wi = WallImages(
                tuple(float(v) for v in self.boundaries.xi_min),
                tuple(float(v) for v in self.boundaries.xi_max),
                self.boundaries.periodic,
                self.boundaries.bc_low,
                self.boundaries.bc_high,
                self.shape,
            )
            self._wall_images = wi if wi.active else None

        if self.use_mpi:
            rank = int(solver.rank)
            size = int(solver.size)
            nz_own = int(solver.NZ // size)
            self._k_own_start = 1 if rank > 0 else 0
            self._k_own_end = self._k_own_start + nz_own
            self._z_own_min = float(
                getattr(
                    solver.grid,
                    "z_own_min",
                    solver.grid.z[self._k_own_start],
                )
            )
            self._z_own_max = float(
                getattr(
                    solver.grid,
                    "z_own_max",
                    solver.grid.z[self._k_own_end],
                )
            )
            self._z_ghost_lo = float(solver.grid.z[0])
            n_phys_z = solver.Nz - int(solver.grid.edge_ghosts[2])
            self._z_ghost_hi = float(solver.grid.z[n_phys_z])
            self._Nz_dep = solver.Nz + (1 if (size > 1 and rank < size - 1) else 0)
            self._filter_species_to_owned_slab()
        else:
            self._k_own_start = 0
            self._k_own_end = int(solver.Nz)
            self._z_own_min = float(solver.grid.z[0])
            self._z_own_max = float(solver.grid.z[-1])
            self._z_ghost_lo = self._z_own_min
            self._z_ghost_hi = self._z_own_max
            self._Nz_dep = int(solver.Nz)

        n_cells = solver.Nx * solver.Ny * solver.Nz
        n_cells_dep = solver.Nx * solver.Ny * self._Nz_dep
        self.J_particles = self.xp.zeros(3 * n_cells, dtype=np.float64)
        self._B_n_buf = self.xp.zeros(3 * n_cells, dtype=np.float64)
        if self.on_gpu or (self.use_mpi and int(getattr(solver, "size", 1)) > 1):
            self._J_prev = (
                self.xp.zeros(3 * n_cells, dtype=np.float64) if self.on_gpu else None
            )
            self._Wx_buf = self.xp.zeros(n_cells_dep, dtype=np.float64)
            self._Wy_buf = self.xp.zeros(n_cells_dep, dtype=np.float64)
            self._Wz_buf = self.xp.zeros(n_cells_dep, dtype=np.float64)
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
        self._m_eps_arr = None
        self._m_mu_arr = None
        self._vol_E_fw = None
        if self.track_energy:
            self._E_prev_fw = self.xp.zeros(3 * n_cells, dtype=np.float64)
            self._fw_scratch = self.xp.zeros(3 * n_cells, dtype=np.float64)
            self._fw_accum_vec = self.xp.zeros(3 * n_cells, dtype=np.float64)
            self._W_cl = self.xp.zeros(3 * n_cells, dtype=np.float64)
            self._J_cl = self.xp.zeros(3 * n_cells, dtype=np.float64)
            self._cl_work_vec = self.xp.zeros(3 * n_cells, dtype=np.float64)
            self._E_init_fw = self.xp.zeros(3 * n_cells, dtype=np.float64)
            self._H_prev_init_fw = self.xp.zeros(3 * n_cells, dtype=np.float64)
            self._H_curr_init_fw = self.xp.zeros(3 * n_cells, dtype=np.float64)
            self._init_energy_weights()
        else:
            self._E_prev_fw = None
            self._fw_scratch = None
            self._fw_accum_vec = None
            self._W_cl = None
            self._J_cl = None
            self._cl_work_vec = None
            self._E_init_fw = None
            self._H_prev_init_fw = None
            self._H_curr_init_fw = None
        self._fw_pending = False
        self._cl_pending = False
        self._cl_step_active = False
        self._fw_dt = 0.0
        self._has_step0_field_snap = False
        self._K_particles_0 = 0.0
        self._K_wall_kick = 0.0
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
        if self.use_mpi and int(getattr(solver, "size", 1)) > 1:
            from mpi4py import MPI

            self._dx_min = float(
                solver.comm.allreduce(self._dx_min, op=MPI.MIN)
            )
        self._attached_dt = None
        self._attached = False
        self._initialized_half_step = False
        self._init_gauss_counts = ()
        self.n_particle_steps = 0
        self.diagnostics: list = []
        self._j_in_solver = False
        self._local_cfl_err = None
        self._local_step_exc = None
        self._enable_step_status_piggyback = True
        self._n_migrated_lo = 0
        self._n_migrated_hi = 0
        self.profile = bool(profile)
        self._last_push_gather_s = 0.0
        self._last_deposit_s = 0.0
        self._last_guard_reduce_s = 0.0
        self._last_migrate_s = 0.0
        self._gpu_mpi_ready = False
        self._gpu_mig_pack_cap = 0
        self._gpu_mig_recv_lo_cap = 0
        self._gpu_mig_recv_hi_cap = 0
        self._cpu_guard_key = None
        self._cpu_guard_send_lo = None
        self._cpu_guard_recv_lo = None
        self._cpu_guard_send_hi = None
        self._cpu_guard_recv_hi = None
        if self.on_gpu and self.use_mpi and int(getattr(solver, "size", 1)) > 1:
            local_exc = None
            try:
                self._ensure_gpu_mpi_buffers()
            except Exception as exc:
                local_exc = exc
            self._collective_check(local_exc)

    def _collective_check(self, local_exc: BaseException | None) -> None:
        """Propagate a per-rank buffer allocation exception collectively across all MPI ranks."""
        s = self.solver
        if not (self.use_mpi and int(getattr(s, "size", 1)) > 1):
            if local_exc is not None:
                raise local_exc
            return
        from mpi4py import MPI

        if local_exc is None:
            code = 0
        elif isinstance(local_exc, MemoryError):
            code = 4
        elif isinstance(local_exc, ParticleCFLError):
            code = 1
        elif isinstance(local_exc, ValueError):
            code = 2
        else:
            code = 3
        max_code = int(s.comm.allreduce(code, op=MPI.MAX))
        if max_code == 0:
            return
        if max_code == 1:
            if local_exc is not None:
                raise ParticleCFLError(str(local_exc)) from local_exc
            raise ParticleCFLError(
                "A peer MPI rank violated the single-cell CFL condition during a collective particle operation."
            )
        if max_code == 2:
            if local_exc is not None:
                raise ValueError(str(local_exc)) from local_exc
            raise ValueError(
                "A peer MPI rank raised ValueError during a collective particle buffer allocation."
            )
        if max_code == 4:
            if local_exc is not None:
                raise _CollectiveMemoryError(str(local_exc)) from local_exc
            raise _CollectiveMemoryError(
                "A peer MPI rank ran out of memory during a collective particle buffer allocation."
            )
        if local_exc is not None:
            raise RuntimeError(str(local_exc)) from local_exc
        raise RuntimeError(
            "A peer MPI rank failed during a collective particle buffer allocation."
        )

    def _piggyback_step_status(
        self, cfl_err: int, local_exc: BaseException | None
    ) -> int:
        """Encode single-cell CFL violation and per-rank step exception into one allreduce integer.

        Status codes (reduced via ``MPI.MAX``):
        - ``0``: success
        - ``1``: :class:`ParticleCFLError` (or ``cfl_err > 0``)
        - ``2``: non-CFL :class:`ValueError`
        - ``3``: other non-OOM :class:`Exception`
        - ``4``: :class:`MemoryError` (including ``cupy.cuda.memory.OutOfMemoryError``)
        """
        if not self._enable_step_status_piggyback:
            return 1 if cfl_err > 0 else 0
        if local_exc is not None:
            if isinstance(local_exc, MemoryError):
                return 4
            if isinstance(local_exc, ParticleCFLError):
                return 1
            if isinstance(local_exc, ValueError):
                return 2
            return 3
        return 1 if cfl_err > 0 else 0

    def _check_step_cfl_and_status(self, cfl_err: int) -> None:
        """Perform the single per-step collective check for CFL violations and per-rank exceptions."""
        from mpi4py import MPI

        comm = self.solver.comm
        local_exc = self._local_step_exc
        self._local_step_exc = None
        self._local_cfl_err = None
        if local_exc is None and cfl_err > 0:
            local_exc = ParticleCFLError(
                "Particle migrated across more than 1 cell in a single step "
                "(violates single-cell CFL / no multi-hop assumption)."
            )
        code = self._piggyback_step_status(cfl_err, local_exc)
        max_code = int(comm.allreduce(code, op=MPI.MAX))
        if max_code == 0:
            return
        if max_code == 1:
            msg = (
                str(local_exc)
                if isinstance(local_exc, ParticleCFLError)
                else (
                    "Particle migrated across more than 1 cell in a single step "
                    "(violates single-cell CFL / no multi-hop assumption)."
                )
            )
            if local_exc is not None:
                raise ParticleCFLError(msg) from local_exc
            raise ParticleCFLError(msg)
        if max_code == 2:
            if local_exc is not None:
                raise ValueError(str(local_exc)) from local_exc
            raise ValueError(
                "A peer MPI rank raised ValueError during a collective particle step."
            )
        if max_code == 4:
            if local_exc is not None:
                raise _CollectiveMemoryError(str(local_exc)) from local_exc
            raise _CollectiveMemoryError(
                "A peer MPI rank ran out of memory during a collective particle step."
            )
        if local_exc is not None:
            raise RuntimeError(str(local_exc)) from local_exc
        raise RuntimeError(
            "A peer MPI rank failed during a collective particle step."
        )

    def _ensure_gpu_mpi_buffers(
        self,
        pack_cap: int = 4096,
        recv_lo_cap: int = 4096,
        recv_hi_cap: int = 4096,
    ) -> None:
        """Allocate or grow device and pinned-host buffers for GPU MPI guard reduction and migration."""
        if not (self.on_gpu and self.use_mpi):
            return
        import cupyx

        xp = self.xp
        nxy = int(self.solver.Nx * self.solver.Ny)
        if not self._gpu_mpi_ready:
            self._d_guard_send_lo = xp.zeros((3, nxy), dtype=np.float64)
            self._d_guard_recv_lo = xp.zeros((3, 2, nxy), dtype=np.float64)
            self._d_guard_send_hi = xp.zeros((3, 2, nxy), dtype=np.float64)
            self._d_guard_recv_hi = xp.zeros((3, nxy), dtype=np.float64)

            self._h_guard_send_lo = cupyx.empty_pinned((3, nxy), dtype=np.float64)
            self._h_guard_recv_lo = cupyx.empty_pinned((3, 2, nxy), dtype=np.float64)
            self._h_guard_send_hi = cupyx.empty_pinned((3, 2, nxy), dtype=np.float64)
            self._h_guard_recv_hi = cupyx.empty_pinned((3, nxy), dtype=np.float64)

            # counts[0..5]: n_send_lo, n_send_hi, n_stay, n_cfl_err, n_recv_lo, n_recv_hi
            self._d_mig_counts = xp.zeros(6, dtype=np.int32)
            self._h_mig_counts = cupyx.empty_pinned(6, dtype=np.int32)
            self._h_mig_recv_cnt_lo = cupyx.empty_pinned(1, dtype=np.int32)
            self._h_mig_recv_cnt_hi = cupyx.empty_pinned(1, dtype=np.int32)
            self._gpu_mpi_ready = True

        if pack_cap > self._gpu_mig_pack_cap:
            new_cap = max(pack_cap, self._gpu_mig_pack_cap * 2, 4096)
            self._d_mig_send_lo = xp.empty(8 * new_cap, dtype=np.float64)
            self._d_mig_send_hi = xp.empty(8 * new_cap, dtype=np.float64)
            self._h_mig_send_lo = cupyx.empty_pinned(8 * new_cap, dtype=np.float64)
            self._h_mig_send_hi = cupyx.empty_pinned(8 * new_cap, dtype=np.float64)
            self._gpu_mig_pack_cap = new_cap

        if recv_lo_cap > self._gpu_mig_recv_lo_cap:
            new_lo = max(recv_lo_cap, self._gpu_mig_recv_lo_cap * 2, 4096)
            self._d_mig_recv_lo = xp.empty(8 * new_lo, dtype=np.float64)
            self._h_mig_recv_lo = cupyx.empty_pinned(8 * new_lo, dtype=np.float64)
            self._gpu_mig_recv_lo_cap = new_lo

        if recv_hi_cap > self._gpu_mig_recv_hi_cap:
            new_hi = max(recv_hi_cap, self._gpu_mig_recv_hi_cap * 2, 4096)
            self._d_mig_recv_hi = xp.empty(8 * new_hi, dtype=np.float64)
            self._h_mig_recv_hi = cupyx.empty_pinned(8 * new_hi, dtype=np.float64)
            self._gpu_mig_recv_hi_cap = new_hi

    def _check_cfl_dt(self, dt: float) -> None:
        """Verify host-side relativistic single-cell CFL ``c * dt <= min_cell_size``."""
        from scipy.constants import c as c_light

        if c_light * dt > self._dx_min * (1.0 + 1e-12):
            raise ParticleCFLError(
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

    def _owned_z_mask(self, z_arr):
        """Return boolean mask of ``z_arr`` coordinates inside this rank's owned z-slab."""
        s = self.solver
        if not self.use_mpi or int(getattr(s, "size", 1)) <= 1:
            return self.xp.ones(z_arr.shape, dtype=bool)
        rank = int(s.rank)
        size = int(s.size)
        z_lo = self._z_own_min
        z_hi = self._z_own_max
        if rank == 0:
            return z_arr < z_hi
        if rank == size - 1:
            return z_arr >= z_lo
        return (z_arr >= z_lo) & (z_arr < z_hi)

    def _owns_z_coord(self, z_val: float) -> bool:
        """Return True if scalar physical coordinate ``z_val`` [m] belongs to this rank's owned z-slab."""
        s = self.solver
        if not self.use_mpi or int(getattr(s, "size", 1)) <= 1:
            return True
        rank = int(s.rank)
        size = int(s.size)
        if rank == 0:
            return float(z_val) < self._z_own_max
        if rank == size - 1:
            return float(z_val) >= self._z_own_min
        return self._z_own_min <= float(z_val) < self._z_own_max

    def _filter_species_to_owned_slab(self) -> None:
        """Filter each species in-place so each MPI rank retains only particles in ``[z_lo, z_hi)``."""
        if not self.use_mpi or int(getattr(self.solver, "size", 1)) <= 1:
            return
        for sp in self.species:
            if sp.n_particles == 0:
                continue
            mask = self._owned_z_mask(sp.z)
            if not bool(self.xp.all(mask)):
                sp.keep(mask)

    @staticmethod
    def _accumulate_guard_plane(dst, src) -> None:
        """Add received guard-layer current ``src`` into owned layer ``dst``."""
        dst += src

    @staticmethod
    def _accumulate_low_guard_plane(dst, src) -> None:
        """Add low guard-layer current ``src`` (from rank ``r + 1``'s plane ``0``) into owned plane ``dst``."""
        dst += src

    def _pack_second_high_guard_plane(
        self, send_hi, N: int, nxy: int, *, use_w_buf: bool = True
    ) -> None:
        """Pack the second high guard plane ``k = k_own_end + 1`` into ``send_hi[:, 1, :]``."""
        if use_w_buf and self._Wx_buf.size >= N + nxy:
            send_hi[0, 1, :] = self._Wx_buf[N : N + nxy]
            send_hi[1, 1, :] = self._Wy_buf[N : N + nxy]
            send_hi[2, 1, :] = self._Wz_buf[N : N + nxy]
        else:
            send_hi[:, 1, :] = 0.0

    def _ensure_cpu_mpi_guard_buffers(self, nxy: int, dtype) -> None:
        """Allocate or resize reusable host buffers for CPU MPI guard-layer exchanges."""
        key = (int(nxy), np.dtype(dtype))
        if self._cpu_guard_key != key:
            self._cpu_guard_send_lo = np.empty((3, nxy), dtype=dtype)
            self._cpu_guard_recv_lo = np.empty((3, 2, nxy), dtype=dtype)
            self._cpu_guard_send_hi = np.empty((3, 2, nxy), dtype=dtype)
            self._cpu_guard_recv_hi = np.empty((3, nxy), dtype=dtype)
            self._cpu_guard_key = key

    def _mpi_sum_guards(self, J_part=None) -> None:
        """Add MPI guard-layer current deposits into neighbour owned layers.

        In Esirkepov current deposition with ``p = 1``, particles owned by rank ``r``
        (with ``zeta^n in [k_own_start, k_own_end)``) deposit into local ``z``-planes
        ``[0, k_own_end + 1]``. Plane ``0`` (when ``r > 0``) is a low guard plane that
        belongs to rank ``r - 1``'s owned plane ``k_own_end - 1``; planes
        ``k_own_end`` and ``k_own_end + 1`` (when ``r < size - 1``) are high guard
        planes that belong to rank ``r + 1``'s owned planes ``1`` and ``2``.
        No ghost refresh of ``J_particles`` is required after summing into the
        owned planes because the FIT electric-field update ``_update_E`` is
        edge-local in ``J`` and is immediately followed by ``_mpi_communicate(E)``,
        which overwrites the 1-cell ghost layers of ``E`` from the neighbours'
        owned boundary planes.
        """
        s = self.solver
        if not self.use_mpi or int(getattr(s, "size", 1)) <= 1:
            return
        from mpi4py import MPI

        buf = self.J_particles if J_part is None else J_part
        comm = s.comm
        rank = int(s.rank)
        size = int(s.size)
        nxy = int(s.Nx * s.Ny)
        N = int(nxy * s.Nz)
        k_own_end = self._k_own_end

        is_gpu = hasattr(buf, "get")
        if is_gpu:
            self._mpi_sum_guards_gpu(buf)
            return
        buf_h = buf
        view = buf_h.reshape((3, s.Nz, nxy))
        self._ensure_cpu_mpi_guard_buffers(nxy, buf_h.dtype)

        send_lo = None
        recv_from_lo = None
        if rank > 0:
            send_lo = self._cpu_guard_send_lo
            send_lo[...] = view[:, 0, :]
            recv_from_lo = self._cpu_guard_recv_lo

        send_hi = None
        recv_from_hi = None
        if rank < size - 1:
            send_hi = self._cpu_guard_send_hi
            send_hi[:, 0, :] = view[:, k_own_end, :]
            self._pack_second_high_guard_plane(
                send_hi, N, nxy, use_w_buf=(buf is self.J_particles)
            )
            recv_from_hi = self._cpu_guard_recv_hi

        reqs = []
        if rank > 0:
            reqs.append(comm.Irecv(recv_from_lo, source=rank - 1, tag=20))
            reqs.append(comm.Isend(send_lo, dest=rank - 1, tag=21))
        if rank < size - 1:
            reqs.append(comm.Irecv(recv_from_hi, source=rank + 1, tag=21))
            reqs.append(comm.Isend(send_hi, dest=rank + 1, tag=20))
        if reqs:
            MPI.Request.Waitall(reqs)

        if recv_from_hi is not None:
            self._accumulate_low_guard_plane(view[:, k_own_end - 1, :], recv_from_hi)
        if recv_from_lo is not None:
            self._accumulate_guard_plane(view[:, 1, :], recv_from_lo[:, 0, :])
            if s.Nz > 2:
                self._accumulate_guard_plane(view[:, 2, :], recv_from_lo[:, 1, :])

    def _mpi_sum_guards_gpu(self, buf) -> None:
        """GPU additive guard-layer reduction via NCCL or pinned-host MPI.

        Notes
        -----
        When ``self.particles_comm == "nccl"``, the guard-layer exchange
        executes directly GPU-to-GPU via ``cupy.cuda.nccl`` on the current CUDA
        stream with zero device-to-host synchronization. When
        ``self.particles_comm == "mpi"``, planes are staged through
        pre-allocated pinned host buffers (``cupyx.empty_pinned``). No ghost
        refresh of ``J_particles`` is required because ``_update_E`` is
        edge-local in ``J`` and ``_mpi_communicate(E)`` immediately overwrites
        the ghost planes of ``E``.
        """
        from mpi4py import MPI

        s = self.solver
        comm = s.comm
        rank = int(s.rank)
        size = int(s.size)
        nxy = int(s.Nx * s.Ny)
        N = int(nxy * s.Nz)
        k_own_end = self._k_own_end
        xp = self.xp
        if not self._gpu_mpi_ready:
            local_exc = None
            try:
                self._ensure_gpu_mpi_buffers()
            except Exception as exc:
                local_exc = exc
            self._collective_check(local_exc)

        view = buf.reshape((3, s.Nz, nxy))
        use_nccl = self.particles_comm == "nccl"
        nccl_comm = (
            s._ensure_nccl_comm()
            if (use_nccl and hasattr(s, "_ensure_nccl_comm"))
            else None
        )

        if rank > 0:
            self._d_guard_send_lo[...] = view[:, 0, :]
        if rank < size - 1:
            self._d_guard_send_hi[:, 0, :] = view[:, k_own_end, :]
            self._pack_second_high_guard_plane(
                self._d_guard_send_hi, N, nxy, use_w_buf=(buf is self.J_particles)
            )

        if nccl_comm is not None:
            from cupy.cuda import nccl

            stream_ptr = xp.cuda.Stream.null.ptr
            nccl.groupStart()
            try:
                if rank > 0:
                    nccl_comm.send(
                        self._d_guard_send_lo.data.ptr,
                        3 * nxy,
                        nccl.NCCL_FLOAT64,
                        rank - 1,
                        stream_ptr,
                    )
                    nccl_comm.recv(
                        self._d_guard_recv_lo.data.ptr,
                        6 * nxy,
                        nccl.NCCL_FLOAT64,
                        rank - 1,
                        stream_ptr,
                    )
                if rank < size - 1:
                    nccl_comm.send(
                        self._d_guard_send_hi.data.ptr,
                        6 * nxy,
                        nccl.NCCL_FLOAT64,
                        rank + 1,
                        stream_ptr,
                    )
                    nccl_comm.recv(
                        self._d_guard_recv_hi.data.ptr,
                        3 * nxy,
                        nccl.NCCL_FLOAT64,
                        rank + 1,
                        stream_ptr,
                    )
            finally:
                nccl.groupEnd()
        else:
            if rank > 0:
                self._d_guard_send_lo.get(out=self._h_guard_send_lo)
            if rank < size - 1:
                self._d_guard_send_hi.get(out=self._h_guard_send_hi)
            xp.cuda.Stream.null.synchronize()

            reqs = []
            if rank > 0:
                reqs.append(
                    comm.Irecv(self._h_guard_recv_lo, source=rank - 1, tag=20)
                )
                reqs.append(
                    comm.Isend(self._h_guard_send_lo, dest=rank - 1, tag=21)
                )
            if rank < size - 1:
                reqs.append(
                    comm.Irecv(self._h_guard_recv_hi, source=rank + 1, tag=21)
                )
                reqs.append(
                    comm.Isend(self._h_guard_send_hi, dest=rank + 1, tag=20)
                )
            if reqs:
                MPI.Request.Waitall(reqs)

            if rank > 0:
                self._d_guard_recv_lo.set(self._h_guard_recv_lo)
            if rank < size - 1:
                self._d_guard_recv_hi.set(self._h_guard_recv_hi)

        if rank < size - 1:
            self._accumulate_low_guard_plane(
                view[:, k_own_end - 1, :], self._d_guard_recv_hi
            )
        if rank > 0:
            self._accumulate_guard_plane(
                view[:, 1, :], self._d_guard_recv_lo[:, 0, :]
            )
            if s.Nz > 2:
                self._accumulate_guard_plane(
                    view[:, 2, :], self._d_guard_recv_lo[:, 1, :]
                )

    def _mpi_migrate_particles_gpu(self, sp: Species) -> None:
        """GPU particle migration across MPI z-slabs via NCCL or pinned-host MPI.

        Notes
        -----
        Active particles are classified and packed into 8-word contiguous device
        buffers (``_d_mig_send_lo``, ``_d_mig_send_hi``) by
        ``classify_and_pack_migration_kernel``. On the NCCL path
        (``self.particles_comm == "nccl"``), neighbour send counts are exchanged
        directly into ``_d_mig_counts[4:6]`` via ``nccl_comm.send`` / ``nccl_comm.recv``,
        after which a single 24-byte transfer to pinned host memory
        (``_h_mig_counts``) retrieves ``(n_send_lo, n_send_hi, n_stay, n_cfl_err,
        n_recv_lo, n_recv_hi)`` so the host can issue the dynamic-length NCCL
        payload calls and update ``sp.n_particles``. On the ``"mpi"`` fallback path,
        counts and payloads are staged through pre-allocated pinned host buffers
        (``cupyx.empty_pinned``).
        """
        from mpi4py import MPI

        from .gpu_kernels import classify_and_pack_migration_gpu, unpack_migration_gpu

        s = self.solver
        comm = s.comm
        rank = int(s.rank)
        size = int(s.size)
        has_lo = rank > 0
        has_hi = rank < size - 1
        xp = self.xp
        if not self._gpu_mpi_ready and self._local_step_exc is None:
            try:
                self._ensure_gpu_mpi_buffers()
            except Exception as exc:
                if not self._enable_step_status_piggyback:
                    raise
                self._local_step_exc = exc

        if self._local_step_exc is None:
            try:
                classify_and_pack_migration_gpu(
                    sp,
                    has_lo=has_lo,
                    has_hi=has_hi,
                    z_own_min=self._z_own_min,
                    z_own_max=self._z_own_max,
                    z_ghost_lo=self._z_ghost_lo,
                    z_ghost_hi=self._z_ghost_hi,
                    d_counts=self._d_mig_counts,
                    d_send_lo=self._d_mig_send_lo,
                    d_send_hi=self._d_mig_send_hi,
                    pack_cap=self._gpu_mig_pack_cap,
                )
                if self._d_cfl_error is not None:
                    self._d_mig_counts[3] += self._d_cfl_error[0]
            except Exception as exc:
                if not self._enable_step_status_piggyback:
                    raise
                self._local_step_exc = exc
        elif self._gpu_mpi_ready:
            self._d_mig_counts.fill(0)

        use_nccl = self.particles_comm == "nccl"
        nccl_comm = (
            s._ensure_nccl_comm()
            if (use_nccl and hasattr(s, "_ensure_nccl_comm"))
            else None
        )

        if nccl_comm is not None:
            from cupy.cuda import nccl

            stream_ptr = xp.cuda.Stream.null.ptr
            nccl.groupStart()
            try:
                if has_lo:
                    nccl_comm.send(
                        self._d_mig_counts.data.ptr,
                        1,
                        nccl.NCCL_INT32,
                        rank - 1,
                        stream_ptr,
                    )
                    nccl_comm.recv(
                        self._d_mig_counts.data.ptr + 16,
                        1,
                        nccl.NCCL_INT32,
                        rank - 1,
                        stream_ptr,
                    )
                if has_hi:
                    nccl_comm.send(
                        self._d_mig_counts.data.ptr + 4,
                        1,
                        nccl.NCCL_INT32,
                        rank + 1,
                        stream_ptr,
                    )
                    nccl_comm.recv(
                        self._d_mig_counts.data.ptr + 20,
                        1,
                        nccl.NCCL_INT32,
                        rank + 1,
                        stream_ptr,
                    )
            finally:
                nccl.groupEnd()

            self._d_mig_counts.get(out=self._h_mig_counts)
            xp.cuda.Stream.null.synchronize()
            if self._local_step_exc is None:
                n_send_lo = int(self._h_mig_counts[0])
                n_send_hi = int(self._h_mig_counts[1])
                n_stay = int(self._h_mig_counts[2])
                n_cfl_err = int(self._h_mig_counts[3])
            else:
                n_send_lo = 0
                n_send_hi = 0
                n_stay = sp.n_particles
                n_cfl_err = 0
            n_recv_lo = int(self._h_mig_counts[4]) if has_lo else 0
            n_recv_hi = int(self._h_mig_counts[5]) if has_hi else 0
        else:
            if self._local_step_exc is None and self._gpu_mpi_ready:
                self._d_mig_counts.get(out=self._h_mig_counts)
                xp.cuda.Stream.null.synchronize()
                n_send_lo = int(self._h_mig_counts[0])
                n_send_hi = int(self._h_mig_counts[1])
                n_stay = int(self._h_mig_counts[2])
                n_cfl_err = int(self._h_mig_counts[3])
            else:
                self._h_mig_counts[:] = 0
                n_send_lo = 0
                n_send_hi = 0
                n_stay = sp.n_particles
                n_cfl_err = 0

            self._h_mig_recv_cnt_lo[0] = 0
            self._h_mig_recv_cnt_hi[0] = 0
            reqs_cnt = []
            if has_lo:
                reqs_cnt.append(
                    comm.Irecv(self._h_mig_recv_cnt_lo, source=rank - 1, tag=10)
                )
                reqs_cnt.append(
                    comm.Isend(self._h_mig_counts[0:1], dest=rank - 1, tag=11)
                )
            if has_hi:
                reqs_cnt.append(
                    comm.Irecv(self._h_mig_recv_cnt_hi, source=rank + 1, tag=11)
                )
                reqs_cnt.append(
                    comm.Isend(self._h_mig_counts[1:2], dest=rank + 1, tag=10)
                )
            if reqs_cnt:
                MPI.Request.Waitall(reqs_cnt)
            n_recv_lo = int(self._h_mig_recv_cnt_lo[0])
            n_recv_hi = int(self._h_mig_recv_cnt_hi[0])

        max_send = max(n_send_lo, n_send_hi)
        n_add = n_recv_lo + n_recv_hi
        n_new = n_stay + n_add
        if self._local_step_exc is None and n_cfl_err == 0:
            try:
                if max_send > self._gpu_mig_pack_cap:
                    self._ensure_gpu_mpi_buffers(
                        pack_cap=max_send,
                        recv_lo_cap=n_recv_lo,
                        recv_hi_cap=n_recv_hi,
                    )
                    classify_and_pack_migration_gpu(
                        sp,
                        has_lo=has_lo,
                        has_hi=has_hi,
                        z_own_min=self._z_own_min,
                        z_own_max=self._z_own_max,
                        z_ghost_lo=self._z_ghost_lo,
                        z_ghost_hi=self._z_ghost_hi,
                        d_counts=self._d_mig_counts,
                        d_send_lo=self._d_mig_send_lo,
                        d_send_hi=self._d_mig_send_hi,
                        pack_cap=self._gpu_mig_pack_cap,
                    )
                elif (
                    n_recv_lo > self._gpu_mig_recv_lo_cap
                    or n_recv_hi > self._gpu_mig_recv_hi_cap
                ):
                    self._ensure_gpu_mpi_buffers(
                        pack_cap=self._gpu_mig_pack_cap,
                        recv_lo_cap=n_recv_lo,
                        recv_hi_cap=n_recv_hi,
                    )
                if n_send_lo > 0 or n_send_hi > 0:
                    sp._compact_async_gpu(sp._keep_mask_scratch)
                sp.n_particles = n_stay
                if n_add > 0:
                    sp._ensure_capacity(n_new)
            except Exception as exc:
                if not self._enable_step_status_piggyback:
                    raise
                self._local_step_exc = exc

        self._check_step_cfl_and_status(n_cfl_err)
        self._n_migrated_lo += n_send_lo
        self._n_migrated_hi += n_send_hi

        if nccl_comm is not None:
            from cupy.cuda import nccl

            stream_ptr = xp.cuda.Stream.null.ptr
            nccl.groupStart()
            try:
                if has_lo:
                    if n_send_lo > 0:
                        nccl_comm.send(
                            self._d_mig_send_lo.data.ptr,
                            8 * n_send_lo,
                            nccl.NCCL_FLOAT64,
                            rank - 1,
                            stream_ptr,
                        )
                    if n_recv_lo > 0:
                        nccl_comm.recv(
                            self._d_mig_recv_lo.data.ptr,
                            8 * n_recv_lo,
                            nccl.NCCL_FLOAT64,
                            rank - 1,
                            stream_ptr,
                        )
                if has_hi:
                    if n_send_hi > 0:
                        nccl_comm.send(
                            self._d_mig_send_hi.data.ptr,
                            8 * n_send_hi,
                            nccl.NCCL_FLOAT64,
                            rank + 1,
                            stream_ptr,
                        )
                    if n_recv_hi > 0:
                        nccl_comm.recv(
                            self._d_mig_recv_hi.data.ptr,
                            8 * n_recv_hi,
                            nccl.NCCL_FLOAT64,
                            rank + 1,
                            stream_ptr,
                        )
            finally:
                nccl.groupEnd()
        else:
            if n_send_lo > 0:
                self._d_mig_send_lo[: 8 * n_send_lo].get(
                    out=self._h_mig_send_lo[: 8 * n_send_lo]
                )
            if n_send_hi > 0:
                self._d_mig_send_hi[: 8 * n_send_hi].get(
                    out=self._h_mig_send_hi[: 8 * n_send_hi]
                )
            if n_send_lo > 0 or n_send_hi > 0:
                xp.cuda.Stream.null.synchronize()

            reqs_pay = []
            if has_lo:
                if n_recv_lo > 0:
                    reqs_pay.append(
                        comm.Irecv(
                            self._h_mig_recv_lo[: 8 * n_recv_lo],
                            source=rank - 1,
                            tag=12,
                        )
                    )
                if n_send_lo > 0:
                    reqs_pay.append(
                        comm.Isend(
                            self._h_mig_send_lo[: 8 * n_send_lo],
                            dest=rank - 1,
                            tag=13,
                        )
                    )
            if has_hi:
                if n_recv_hi > 0:
                    reqs_pay.append(
                        comm.Irecv(
                            self._h_mig_recv_hi[: 8 * n_recv_hi],
                            source=rank + 1,
                            tag=13,
                        )
                    )
                if n_send_hi > 0:
                    reqs_pay.append(
                        comm.Isend(
                            self._h_mig_send_hi[: 8 * n_send_hi],
                            dest=rank + 1,
                            tag=12,
                        )
                    )
            if reqs_pay:
                MPI.Request.Waitall(reqs_pay)

            if n_recv_lo > 0:
                self._d_mig_recv_lo[: 8 * n_recv_lo].set(
                    self._h_mig_recv_lo[: 8 * n_recv_lo]
                )
            if n_recv_hi > 0:
                self._d_mig_recv_hi[: 8 * n_recv_hi].set(
                    self._h_mig_recv_hi[: 8 * n_recv_hi]
                )

        if n_add > 0:
            if n_recv_lo > 0:
                unpack_migration_gpu(sp, self._d_mig_recv_lo, n_recv_lo, n_stay)
            if n_recv_hi > 0:
                unpack_migration_gpu(
                    sp, self._d_mig_recv_hi, n_recv_hi, n_stay + n_recv_lo
                )
            sp.n_particles = n_new

    def _mpi_migrate_particles(self, sp: Species) -> None:
        """Pack and exchange particles that crossed ``[z_lo, z_hi)`` with z-neighbours."""
        s = self.solver
        if not self.use_mpi or int(getattr(s, "size", 1)) <= 1:
            return
        if sp.on_gpu:
            self._mpi_migrate_particles_gpu(sp)
            return
        from mpi4py import MPI

        comm = s.comm
        rank = int(s.rank)
        size = int(s.size)
        z_lo = self._z_own_min
        z_hi = self._z_own_max

        cfl_err = 1 if self._local_cfl_err is not None else 0
        self._local_cfl_err = None
        n_send_lo = 0
        n_send_hi = 0
        send_lo_f = np.empty((7, 0), dtype=np.float64)
        send_lo_id = np.empty(0, dtype=np.int64)
        send_hi_f = np.empty((7, 0), dtype=np.float64)
        send_hi_id = np.empty(0, dtype=np.int64)

        if self._local_step_exc is None:
            try:
                n_p = sp.n_particles
                if n_p > 0:
                    z_h = sp.z
                    leave_lo = (
                        (z_h < z_lo) if rank > 0 else np.zeros(n_p, dtype=bool)
                    )
                    leave_hi = (
                        (z_h >= z_hi)
                        if rank < size - 1
                        else np.zeros(n_p, dtype=bool)
                    )
                    if rank > 0 and bool(np.any(leave_lo)):
                        if bool(np.any(z_h[leave_lo] < self._z_ghost_lo)):
                            cfl_err = 1
                    if rank < size - 1 and bool(np.any(leave_hi)):
                        if bool(np.any(z_h[leave_hi] >= self._z_ghost_hi)):
                            cfl_err = 1
                else:
                    leave_lo = np.zeros(0, dtype=bool)
                    leave_hi = np.zeros(0, dtype=bool)

                if cfl_err == 0:
                    n_send_lo = int(np.sum(leave_lo))
                    n_send_hi = int(np.sum(leave_hi))

                    def _pack(mask, count):
                        if count == 0:
                            return (
                                np.empty((7, 0), dtype=np.float64),
                                np.empty(0, dtype=np.int64),
                            )
                        f_buf = np.empty((7, count), dtype=np.float64)
                        f_buf[0] = sp.x[mask]
                        f_buf[1] = sp.y[mask]
                        f_buf[2] = sp.z[mask]
                        f_buf[3] = sp.ux[mask]
                        f_buf[4] = sp.uy[mask]
                        f_buf[5] = sp.uz[mask]
                        f_buf[6] = sp.w[mask]
                        id_buf = (
                            sp.id[mask].copy()
                            if sp.track_ids
                            else np.empty(0, dtype=np.int64)
                        )
                        return f_buf, id_buf

                    send_lo_f, send_lo_id = _pack(leave_lo, n_send_lo)
                    send_hi_f, send_hi_id = _pack(leave_hi, n_send_hi)

                    if n_send_lo > 0 or n_send_hi > 0:
                        stay_mask = ~(leave_lo | leave_hi)
                        sp.keep(stay_mask)
            except Exception as exc:
                if not self._enable_step_status_piggyback:
                    raise
                self._local_step_exc = exc
                n_send_lo = 0
                n_send_hi = 0

        # Exchange counts first
        cnt_send_lo = np.array([n_send_lo], dtype=np.int64)
        cnt_send_hi = np.array([n_send_hi], dtype=np.int64)
        cnt_recv_lo = np.zeros(1, dtype=np.int64)
        cnt_recv_hi = np.zeros(1, dtype=np.int64)

        reqs_cnt = []
        if rank > 0:
            reqs_cnt.append(comm.Irecv(cnt_recv_lo, source=rank - 1, tag=10))
            reqs_cnt.append(comm.Isend(cnt_send_lo, dest=rank - 1, tag=11))
        if rank < size - 1:
            reqs_cnt.append(comm.Irecv(cnt_recv_hi, source=rank + 1, tag=11))
            reqs_cnt.append(comm.Isend(cnt_send_hi, dest=rank + 1, tag=10))
        if reqs_cnt:
            MPI.Request.Waitall(reqs_cnt)

        n_recv_lo = int(cnt_recv_lo[0])
        n_recv_hi = int(cnt_recv_hi[0])
        n_add = n_recv_lo + n_recv_hi
        n_old = sp.n_particles
        n_new = n_old + n_add

        recv_lo_f = None
        recv_hi_f = None
        recv_lo_id = None
        recv_hi_id = None
        if self._local_step_exc is None and cfl_err == 0:
            try:
                recv_lo_f = np.empty((7, n_recv_lo), dtype=np.float64)
                recv_hi_f = np.empty((7, n_recv_hi), dtype=np.float64)
                recv_lo_id = (
                    np.empty(n_recv_lo, dtype=np.int64)
                    if sp.track_ids
                    else np.empty(0, dtype=np.int64)
                )
                recv_hi_id = (
                    np.empty(n_recv_hi, dtype=np.int64)
                    if sp.track_ids
                    else np.empty(0, dtype=np.int64)
                )
                if n_add > 0:
                    sp._ensure_capacity(n_new)
            except Exception as exc:
                if not self._enable_step_status_piggyback:
                    raise
                self._local_step_exc = exc

        self._check_step_cfl_and_status(cfl_err)
        self._n_migrated_lo += n_send_lo
        self._n_migrated_hi += n_send_hi

        reqs_pay = []
        if rank > 0:
            if n_recv_lo > 0:
                reqs_pay.append(comm.Irecv(recv_lo_f, source=rank - 1, tag=12))
                if sp.track_ids:
                    reqs_pay.append(comm.Irecv(recv_lo_id, source=rank - 1, tag=14))
            if n_send_lo > 0:
                reqs_pay.append(comm.Isend(send_lo_f, dest=rank - 1, tag=13))
                if sp.track_ids:
                    reqs_pay.append(comm.Isend(send_lo_id, dest=rank - 1, tag=15))
        if rank < size - 1:
            if n_recv_hi > 0:
                reqs_pay.append(comm.Irecv(recv_hi_f, source=rank + 1, tag=13))
                if sp.track_ids:
                    reqs_pay.append(comm.Irecv(recv_hi_id, source=rank + 1, tag=15))
            if n_send_hi > 0:
                reqs_pay.append(comm.Isend(send_hi_f, dest=rank + 1, tag=12))
                if sp.track_ids:
                    reqs_pay.append(comm.Isend(send_hi_id, dest=rank + 1, tag=14))
        if reqs_pay:
            MPI.Request.Waitall(reqs_pay)

        if n_add > 0:
            if n_recv_lo > 0:
                sl_lo = slice(n_old, n_old + n_recv_lo)
                sp._x[sl_lo] = recv_lo_f[0]
                sp._y[sl_lo] = recv_lo_f[1]
                sp._z[sl_lo] = recv_lo_f[2]
                sp._ux[sl_lo] = recv_lo_f[3]
                sp._uy[sl_lo] = recv_lo_f[4]
                sp._uz[sl_lo] = recv_lo_f[5]
                sp._w[sl_lo] = recv_lo_f[6]
                if sp.track_ids:
                    sp._id[sl_lo] = recv_lo_id
            if n_recv_hi > 0:
                sl_hi = slice(n_old + n_recv_lo, n_new)
                sp._x[sl_hi] = recv_hi_f[0]
                sp._y[sl_hi] = recv_hi_f[1]
                sp._z[sl_hi] = recv_hi_f[2]
                sp._ux[sl_hi] = recv_hi_f[3]
                sp._uy[sl_hi] = recv_hi_f[4]
                sp._uz[sl_hi] = recv_hi_f[5]
                sp._w[sl_hi] = recv_hi_f[6]
                if sp.track_ids:
                    sp._id[sl_hi] = recv_hi_id
            sp.n_particles = n_new

    def _mpi_pre_push_hook(self, sp: Species) -> None:
        """Hook called before particle push (no-op in normal execution; used for mutation testing)."""
        pass

    def _mpi_post_push_hook(self, sp: Species) -> None:
        """Hook called after particle push and deposit to migrate leavers across MPI slabs."""
        if self.use_mpi and int(getattr(self.solver, "size", 1)) > 1:
            self._mpi_migrate_particles(sp)

    def mpi_gather_species(
        self, sp: Species | int = 0, sort_by_id: bool = True, root: int | None = 0
    ) -> Species | None:
        """Gather a :class:`Species` across all MPI ranks into a single CPU :class:`Species`.

        Parameters
        ----------
        sp : Species or int, optional
            Species instance or index in ``self.species``. Default is ``0``.
        sort_by_id : bool, optional
            If True (default) and ``sp.track_ids`` is True, sort the gathered
            particles by ascending ``id``.
        root : int or None, optional
            Root rank to gather onto (returns ``None`` on other ranks), or
            ``None`` to ``allgather`` onto every rank. Default is ``0``.

        Returns
        -------
        Species or None
            Global CPU :class:`Species` on ``root`` (or all ranks if ``root is None``).
        """
        sp_obj = self.species[sp] if isinstance(sp, int) else sp
        is_gpu = sp_obj.on_gpu
        n_p = sp_obj.n_particles
        f_local = np.empty((7, n_p), dtype=np.float64)
        if n_p > 0:
            f_local[0] = sp_obj.x.get() if is_gpu else sp_obj.x
            f_local[1] = sp_obj.y.get() if is_gpu else sp_obj.y
            f_local[2] = sp_obj.z.get() if is_gpu else sp_obj.z
            f_local[3] = sp_obj.ux.get() if is_gpu else sp_obj.ux
            f_local[4] = sp_obj.uy.get() if is_gpu else sp_obj.uy
            f_local[5] = sp_obj.uz.get() if is_gpu else sp_obj.uz
            f_local[6] = sp_obj.w.get() if is_gpu else sp_obj.w
        id_local = (
            (sp_obj.id.get() if is_gpu else sp_obj.id.copy())
            if (sp_obj.track_ids and n_p > 0)
            else np.empty(0, dtype=np.int64)
        )

        s = self.solver
        if not self.use_mpi or int(getattr(s, "size", 1)) <= 1:
            f_list = [f_local]
            id_list = [id_local]
            is_target = True
        elif root is None:
            f_list = s.comm.allgather(f_local)
            id_list = s.comm.allgather(id_local) if sp_obj.track_ids else None
            is_target = True
        else:
            f_list = s.comm.gather(f_local, root=root)
            id_list = (
                s.comm.gather(id_local, root=root) if sp_obj.track_ids else None
            )
            is_target = int(s.rank) == int(root)

        if not is_target:
            return None

        f_all = (
            np.concatenate(f_list, axis=1)
            if f_list
            else np.empty((7, 0), dtype=np.float64)
        )
        n_tot = f_all.shape[1]
        out = Species(
            name=sp_obj.name,
            charge=sp_obj.charge,
            mass=sp_obj.mass,
            capacity=max(n_tot, 1),
            track_ids=sp_obj.track_ids,
        )
        out.u_is_synced = sp_obj.u_is_synced
        if n_tot > 0:
            id_all = (
                np.concatenate(id_list)
                if (sp_obj.track_ids and id_list is not None)
                else None
            )
            if sort_by_id and id_all is not None:
                perm = np.argsort(id_all, kind="mergesort")
                f_all = f_all[:, perm]
                id_all = id_all[perm]
            out._x[:n_tot] = f_all[0]
            out._y[:n_tot] = f_all[1]
            out._z[:n_tot] = f_all[2]
            out._ux[:n_tot] = f_all[3]
            out._uy[:n_tot] = f_all[4]
            out._uz[:n_tot] = f_all[5]
            out._w[:n_tot] = f_all[6]
            if sp_obj.track_ids and id_all is not None:
                out._id[:n_tot] = id_all
            out.n_particles = n_tot
        return out

    def reset_gauss_reference(self) -> None:
        """Re-record the initial Gauss's law background charge ``q_bg = S_tilde d_tilde - q_nodes``.

        Call this method if ``solver.E`` or particle positions/weights are
        modified in-place after :meth:`attach` (when particle counts are
        unchanged so automatic step-0 re-recording does not trigger).
        """
        if self.use_mpi:
            local_exc = None
            try:
                self._filter_species_to_owned_slab()
            except Exception as exc:
                local_exc = exc
            self._collective_check(local_exc)
        self._record_initial_gauss_state()
        self._init_gauss_counts = tuple(sp._n_launch_bound for sp in self.species)

    def attach(self) -> "ParticleSolver":
        """Register the particle ``mid_step`` hook on ``self.solver`` and initialise Gauss bookkeeping.

        Raises
        ------
        ValueError
            If ``solver.one_step`` is not a hooked step variant
            (``_one_step``, ``_one_step_cpml``, ``_mpi_one_step``, ``_fused_step``,
            or their MKL/ABC variants), or if ``c * solver.dt > min_cell_size``
            (violating the single-cell Esirkepov CFL assumption ``|b1 - b0| <= 1``).
        """
        s = self.solver

        hooked_steps = [
            fn
            for fn in (
                getattr(s, "_one_step", None),
                getattr(s, "_one_step_cpml", None),
                getattr(s, "_one_step_mkl", None),
                getattr(s, "_one_step_cpml_mkl", None),
                getattr(s, "_mpi_one_step", None),
                getattr(s, "_mpi_one_step_mkl", None),
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
                f"invokes mid_step hooks (_one_step, _one_step_cpml, _mpi_one_step, _fused_step, or their MKL/ABC variants), but "
                f"solver.one_step is {step_name!r}, which does not execute particle hooks."
            )

        self._dx_min = min(
            float(self.xp.min(self.grid_map.dx)),
            float(self.xp.min(self.grid_map.dy)),
            float(self.xp.min(self.grid_map.dz)),
        )
        if self.use_mpi and int(getattr(s, "size", 1)) > 1:
            from mpi4py import MPI

            self._dx_min = float(s.comm.allreduce(self._dx_min, op=MPI.MIN))
            s._mpi_communicate(s.E)
            s._mpi_communicate(s.H)
        self._check_cfl_dt(float(s.dt))
        self.reset_cfl_error()

        s._particles_grid_map = self.grid_map
        s._particles_shape_order = self.shape
        s._particle_solver = self

        for inj in self.injectors:
            local_exc = None
            try:
                inj._bind(s)
            except Exception as exc:  # noqa: BLE001
                local_exc = exc
            self._collective_check(local_exc)

        local_exc = None
        try:
            if self.capacity_headroom > 0.0:
                for sp in self.species:
                    n_cur = int(sp._n_launch_bound if self.on_gpu else sp.n_particles)
                    if n_cur > 0:
                        target_cap = max(
                            int(np.ceil(n_cur * (1.0 + self.capacity_headroom))),
                            n_cur + 64,
                        )
                        sp._ensure_capacity(target_cap, exact=True)

            if self.on_gpu:
                all_periodic_no_obs = (
                    all(self.boundaries.periodic)
                    and self.boundaries.solid_mask is None
                    and not self.boundaries.collectors
                )
                need_boundary = not all_periodic_no_obs
                need_sort = self.sort_every > 0
                need_tmp = (
                    not all_periodic_no_obs
                    or need_sort
                    or (self.use_mpi and int(getattr(s, "size", 1)) > 1)
                )
                for sp in self.species:
                    sp._ensure_gpu_scratch(
                        need_tmp=need_tmp,
                        need_boundary=need_boundary,
                        need_sort=need_sort,
                    )
                self.boundaries._ensure_gpu_buffers()
                # Pre-compile the CUDA RawModule during attach() so step 0 does not compile
                from .gpu_kernels import _get_module

                _get_module()
        except Exception as exc:
            local_exc = exc
        self._collective_check(local_exc)

        self.reset_gauss_reference()
        self._has_pec_domain_wall = self._compute_has_pec_domain_wall()
        local_exc = None
        try:
            if self.track_energy:
                if self._E_prev_fw is None:
                    n_cells = s.Nx * s.Ny * s.Nz
                    self._E_prev_fw = self.xp.zeros(3 * n_cells, dtype=np.float64)
                    self._fw_scratch = self.xp.zeros(3 * n_cells, dtype=np.float64)
                    self._fw_accum_vec = self.xp.zeros(3 * n_cells, dtype=np.float64)
                    self._E_init_fw = self.xp.zeros(3 * n_cells, dtype=np.float64)
                    self._H_prev_init_fw = self.xp.zeros(3 * n_cells, dtype=np.float64)
                    self._H_curr_init_fw = self.xp.zeros(3 * n_cells, dtype=np.float64)
                if self._W_cl is None:
                    n_cells = s.Nx * s.Ny * s.Nz
                    self._W_cl = self.xp.zeros(3 * n_cells, dtype=np.float64)
                    self._J_cl = self.xp.zeros(3 * n_cells, dtype=np.float64)
                    self._cl_work_vec = self.xp.zeros(3 * n_cells, dtype=np.float64)
                self._init_energy_weights()
                self._fw_accum_vec.fill(0.0)
                self._W_cl.fill(0.0)
                self._J_cl.fill(0.0)
                self._cl_work_vec.fill(0.0)
            self._fw_pending = False
            self._cl_pending = False
            self._cl_step_active = False
            self._fw_dt = 0.0
            self._has_step0_field_snap = False
            H_arr = s.H.toarray()
            if self.on_gpu:
                self._H_prev = H_arr.copy()
                s._H_before_update = H_arr.copy()
            else:
                self._H_prev = None
                np.copyto(self._B_n_buf, H_arr)
                s._H_before_update = self._B_n_buf
        except Exception as exc:
            local_exc = exc
        self._collective_check(local_exc)
        if self.track_energy:
            self._K_particles_0 = self.kinetic_energy(synchronised=True)
        else:
            self._K_particles_0 = 0.0
        self._K_wall_kick = 0.0

        if not self._attached:
            s.register_mid_step(self._mid_step_hook)
            self._attached = True
        return self

    def _init_energy_weights(self) -> None:
        """Compute discrete FIT energy weights ``M_eps``, ``M_mu``, and edge dual-volume ``V_E``."""
        s = self.solver

        def _to_host_f64(arr_obj):
            raw = arr_obj.toarray() if hasattr(arr_obj, "toarray") else arr_obj
            if hasattr(raw, "get"):
                return np.asarray(raw.get(), dtype=np.float64)
            return np.asarray(raw, dtype=np.float64)

        s_grid = getattr(s, "grid", None)
        L_obj = getattr(s_grid, "L", None)
        if L_obj is None:
            L_obj = getattr(s, "L", None)
        tL_obj = getattr(s_grid, "tL", None)
        if tL_obj is None:
            tL_obj = getattr(s, "tL", None)
        iA_obj = getattr(s, "iA", None)
        if iA_obj is None:
            iA_obj = getattr(s_grid, "iA", None)
        itA_obj = getattr(s, "itA", None)
        if itA_obj is None:
            itA_obj = getattr(s_grid, "itA", None)
        ieps_obj = getattr(s, "ieps", None)
        imu_obj = getattr(s, "imu", None)
        missing_ops = [
            name
            for name, obj in (
                ("L", L_obj),
                ("tL", tL_obj),
                ("iA", iA_obj),
                ("itA", itA_obj),
                ("ieps", ieps_obj),
                ("imu", imu_obj),
            )
            if obj is None
        ]
        if missing_ops:
            raise ValueError(
                "ParticleSolver energy diagnostics require FIT metric and material "
                f"operators (L, tL, iA, itA, ieps, imu); missing: {', '.join(missing_ops)}."
            )

        L_h = _to_host_f64(L_obj)
        tL_h = _to_host_f64(tL_obj)
        iA_h = _to_host_f64(iA_obj)
        itA_h = _to_host_f64(itA_obj)
        ieps_h = _to_host_f64(ieps_obj)
        imu_h = _to_host_f64(imu_obj)

        act_e = (ieps_h != 0.0) & (itA_h != 0.0) & (L_h != 0.0)
        m_eps_h = np.where(act_e, L_h / np.where(act_e, ieps_h * itA_h, 1.0), 0.0)
        act_h = (imu_h != 0.0) & (iA_h != 0.0) & (tL_h != 0.0)
        m_mu_h = np.where(act_h, tL_h / np.where(act_h, imu_h * iA_h, 1.0), 0.0)
        vol_E_h = m_eps_h * ieps_h

        if self.use_mpi and int(getattr(s, "size", 1)) > 1:
            nxy = int(s.Nx * s.Ny)
            k0, k1 = self._k_own_start, self._k_own_end
            for arr_h in (m_eps_h, m_mu_h, vol_E_h):
                v3 = arr_h.reshape((3, s.Nz, nxy))
                if k0 > 0:
                    v3[:, :k0, :] = 0.0
                if k1 < s.Nz:
                    v3[:, k1:, :] = 0.0

        self._m_eps_arr = (
            self.xp.asarray(m_eps_h, dtype=np.float64) if self.on_gpu else m_eps_h
        )
        self._m_mu_arr = (
            self.xp.asarray(m_mu_h, dtype=np.float64) if self.on_gpu else m_mu_h
        )
        self._vol_E_fw = (
            self.xp.asarray(vol_E_h, dtype=np.float64) if self.on_gpu else vol_E_h
        )

    def _step_e_bar_for_work(self, E_prev, E_curr, out) -> None:
        """Compute time-centred electric field ``0.5 * (E^n + E^{n+1})`` into ``out`` in-place."""
        self.xp.add(E_prev, E_curr, out=out)
        out *= 0.5

    def _flush_pending_field_work(self, E_curr) -> None:
        """Accumulate ``dt * J^{n+1/2} * E_bar^{n+1/2}`` into ``_fw_accum_vec`` (and ``_cl_work_vec``) in-place."""
        if not self._fw_pending:
            return
        self._step_e_bar_for_work(self._E_prev_fw, E_curr, self._fw_scratch)
        if self._cl_pending:
            self._cl_work_vec += (self._fw_scratch * self._J_cl) * self._fw_dt
            self._cl_pending = False
        self._fw_scratch *= self.J_particles
        self._fw_scratch *= self._fw_dt
        self._fw_accum_vec += self._fw_scratch
        self._fw_pending = False

    def _finalize_closure_current(self, dt: float) -> None:
        """Finalize per-step PEC wall-closure current ``_J_cl`` from ``_W_cl``."""
        s = self.solver
        use_mpi_multi = self.use_mpi and int(getattr(s, "size", 1)) > 1
        if use_mpi_multi:
            active = bool(
                _ENABLE_PEC_WALL_CLOSURE
                and (
                    self._has_pec_domain_wall
                    or any(inj.emitter_kind == "pec_wall" for inj in self.injectors)
                )
            )
        else:
            active = self._cl_step_active
        if not active:
            self._cl_pending = False
            return

        xp = self.xp
        Nx, Ny, Nz = s.Nx, s.Ny, s.Nz
        N = Nx * Ny * Nz
        if use_mpi_multi:
            self._mpi_sum_guards(self._W_cl)

        if self.filter:
            if self.use_mpi:
                raise NotImplementedError(
                    "Binomial current filtering is not supported with use_mpi=True."
                )
            Wx_cl, Wy_cl, Wz_cl = apply_binomial_filter_3d(
                self._W_cl[:N],
                self._W_cl[N : 2 * N],
                self._W_cl[2 * N : 3 * N],
                Nx,
                Ny,
                Nz,
                periodic=self.boundaries.periodic,
                compensate=(self.filter != "uncompensated"),
                xp=xp,
            )
            self._W_cl[:N] = Wx_cl
            self._W_cl[N : 2 * N] = Wy_cl
            self._W_cl[2 * N : 3 * N] = Wz_cl

        xp.multiply(self._W_cl, self._itA_arr, out=self._J_cl)
        self._J_cl *= 1.0 / dt
        self._cl_pending = True

    def _backpush_injected_slice(
        self,
        sp: Species,
        sl: slice,
        xi_i,
        eta_i,
        zeta_i,
        E_field,
        B_n_tuple,
        t_n: float,
        dt: float,
    ) -> None:
        """Back-push newly injected particles ``u(t_n) -> u(t_n - dt/2)`` prior to the step's full momentum push."""
        s = self.solver
        xp = self.xp
        Ex, Ey, Ez, Bx, By, Bz = gather_fields(
            xi_i,
            eta_i,
            zeta_i,
            E_field,
            B_n_tuple,
            s.Nx,
            s.Ny,
            s.Nz,
            order=self.shape,
            gather=self.gather,
            periodic=self.boundaries.periodic,
            xp=xp,
        )
        Ex, Ey, Ez, Bx, By, Bz = self._add_image_field(
            (Ex, Ey, Ez, Bx, By, Bz),
            self._image_field(xi_i, eta_i, zeta_i, E_field, B_n_tuple),
        )
        x_s = sp._x[sl]
        y_s = sp._y[sl]
        z_s = sp._z[sl]
        Ex_e, Ey_e, Ez_e = self._eval_ext_field(self.E_ext, x_s, y_s, z_s, t_n)
        Bx_e, By_e, Bz_e = self._eval_ext_field(self.B_ext, x_s, y_s, z_s, t_n)
        fn = get_pusher(self.pusher)
        ux_b, uy_b, uz_b = fn(
            sp._ux[sl],
            sp._uy[sl],
            sp._uz[sl],
            Ex + Ex_e,
            Ey + Ey_e,
            Ez + Ez_e,
            Bx + Bx_e,
            By + By_e,
            Bz + Bz_e,
            sp.charge,
            sp.mass,
            -0.5 * dt,
            xp=xp,
        )
        sp._ux[sl] = ux_b
        sp._uy[sl] = uy_b
        sp._uz[sl] = uz_b

    def _compute_has_pec_domain_wall(self) -> bool:
        """Return True on all MPI ranks if any global domain boundary face is PEC."""
        s = self.solver
        has_pec = any(
            b in ("pec", "electric")
            for b in (*self.boundaries.bc_low, *self.boundaries.bc_high)
        )
        global_bc_z = getattr(s, "_global_bc_z", None)
        if global_bc_z is not None:
            has_pec = has_pec or any(
                str(b).strip().lower() in ("pec", "electric") for b in global_bc_z
            )
        if self.use_mpi and int(getattr(s, "size", 1)) > 1:
            from mpi4py import MPI

            has_pec = bool(s.comm.allreduce(bool(has_pec), op=MPI.LOR))
        return bool(has_pec)

    def _pec_wall_node_index(self, axis: int, side: str) -> int:
        """Return the logical primal node index ``k_w`` of the domain wall on ``(axis, side)``."""
        if side == "low":
            return round(self.boundaries.xi_min[axis])
        n_ax = (self.solver.Nx, self.solver.Ny, self.solver.Nz)[axis]
        return min(round(self.boundaries.xi_max[axis]), n_ax - 1)

    def _injector_pec_wall_face(self, inj) -> tuple[int, str]:
        """Return ``(axis, side)`` for a ``pec_wall`` injector.

        Raises
        ------
        ValueError
            If the injector axis, side, direction, or plane coordinate does not
            correspond to a domain boundary wall emitting into the domain.
        """
        raw_axis = getattr(inj, "axis", 2)
        if isinstance(raw_axis, bool) or int(raw_axis) not in (0, 1, 2):
            raise ValueError(
                f"Unsupported pec_wall injector axis={raw_axis!r}; expected 0, 1, or 2."
            )
        axis = int(raw_axis)
        side = getattr(inj, "side", None)
        if side is not None:
            if side not in ("low", "high"):
                raise ValueError(
                    f"Unsupported pec_wall injector side={side!r}; expected 'low' or 'high'."
                )
            return axis, str(side)
        if axis != 2 or not hasattr(inj, "z0"):
            raise ValueError(
                "pec_wall injector without an explicit side='low'/'high' is only "
                f"supported for z-axis injectors with z0; got axis={axis!r}."
            )
        z0 = float(inj.z0)
        z_lo = float(getattr(self.solver.grid, "zmin", self.grid_map.zmin))
        z_hi = float(getattr(self.solver.grid, "zmax", self.grid_map.zmax))
        span = max(abs(z_hi - z_lo), 1.0)
        tol = 1e-12 * span
        direction = getattr(inj, "direction", None)
        if abs(z0 - z_lo) <= tol:
            if direction is not None and int(direction) != 1:
                raise ValueError(
                    f"pec_wall injector at zmin={z_lo:g} must emit into the domain "
                    f"(direction=+1); got direction={direction!r}."
                )
            return (2, "low")
        if abs(z0 - z_hi) <= tol:
            if direction is not None and int(direction) != -1:
                raise ValueError(
                    f"pec_wall injector at zmax={z_hi:g} must emit into the domain "
                    f"(direction=-1); got direction={direction!r}."
                )
            return (2, "high")
        raise ValueError(
            f"pec_wall injector z0={z0:g} does not lie on zmin={z_lo:g} or zmax={z_hi:g}; "
            "use emitter_kind='interior_sheet' for interior emission planes."
        )

    def _check_mpi_pec_wall_closure(
        self, q_vis, axis: int, side: str, k_w: int
    ) -> None:
        """Validate that a PEC wall closure on ``axis == 2`` lies within this rank's slab and guard layers."""
        s = self.solver
        if not self.use_mpi or int(getattr(s, "size", 1)) <= 1 or axis != 2:
            return
        rank = int(s.rank)
        size = int(s.size)
        xp = self.xp
        if side == "low":
            if rank != 0:
                raise RuntimeError(
                    f"PEC low-z wall closure executed on non-owning MPI rank {rank}."
                )
            k_lim = int(self._k_own_end) + 2
            if k_lim < q_vis.shape[2] and bool(xp.any(q_vis[:, :, k_lim:] != 0.0)):
                raise ValueError(
                    "PEC low-z wall closure stencil extends beyond the MPI guard layers "
                    f"(owned slab ends at k={self._k_own_end}, shape={self.shape})."
                )
        else:
            if rank != size - 1:
                raise RuntimeError(
                    f"PEC high-z wall closure executed on non-owning MPI rank {rank}."
                )
            if int(self._k_own_start) > 1 and bool(
                xp.any(q_vis[:, :, : int(self._k_own_start) - 1] != 0.0)
            ):
                raise ValueError(
                    "PEC high-z wall closure stencil extends beyond the low MPI guard layer."
                )

    def _apply_pec_wall_event(
        self,
        q_vis,
        axis: int,
        side: str,
        *,
        event: str,
        W_tuple: tuple,
        q_bg,
    ) -> None:
        """Apply PEC wall-closure current to ``W_tuple[axis]`` and record surface charge on ``k_w`` in ``q_bg``."""
        xp = self.xp
        k_w = self._pec_wall_node_index(axis, side)
        self._check_mpi_pec_wall_closure(q_vis, axis, side, k_w)

        sl_wall = [slice(None), slice(None), slice(None)]
        sl_wall[axis] = k_w
        sl_wall_t = tuple(sl_wall)
        # B-spline symmetry at the wall node k_w implies the masked exterior
        # ghost weight equals the interior tail weight sum_{k != k_w} q_vis[k],
        # so the full particle charge at the wall is 2 * sum(q_vis) - q_vis[k_w].
        q_wall = 2.0 * xp.sum(q_vis, axis=axis) - q_vis[sl_wall_t]
        if event == "inject":
            q_bg[sl_wall_t] -= q_wall
        else:
            q_bg[sl_wall_t] += q_wall

        if not _ENABLE_PEC_WALL_CLOSURE:
            return

        dW_n = _pec_wall_closure_1d(q_vis, axis, side, k_w=k_w, event=event, xp=xp)
        W_n = W_tuple[axis]
        N = q_vis.size
        dW_flat = dW_n.ravel(order="F")
        W_n[:N] += dW_flat
        if self.track_energy and self._W_cl is not None:
            self._W_cl[axis * N : (axis + 1) * N] += dW_flat
            self._cl_step_active = True

    def _check_wall_kick_error(self, *, periodic: bool = False) -> None:
        """Synchronise and check the device-side PEC-wall birth-kick flag ``_d_wall_kick_error``.

        When ``periodic=True``, the device flag is synchronised to the host only
        when ``self.n_particle_steps % check_every == 0`` (where ``check_every``
        defaults to ``64`` and may be overridden via ``self._wall_kick_check_every``)
        to avoid a per-step device-to-host synchronisation on GPU. When
        ``periodic=False``, the flag is synchronised and checked unconditionally.
        """
        err_flag = getattr(self, "_d_wall_kick_error", None)
        if err_flag is None:
            return
        if periodic:
            check_every = int(getattr(self, "_wall_kick_check_every", 64))
            if check_every <= 0 or (self.n_particle_steps % check_every) != 0:
                return
        if bool(err_flag):
            err_flag.fill(False)
            raise ValueError(
                "PEC-wall birth kick resulted in gamma' < 1 (particle cannot leave the wall)."
            )

    def _apply_pec_wall_birth_kick(
        self,
        sp_inj: Species,
        sl_new: slice,
        xi_i,
        eta_i,
        zeta_i,
        E_field,
        axis: int,
        side: str,
    ) -> None:
        """Apply the PEC-wall birth kick for ``shape >= 2`` and accumulate ``_K_wall_kick``.

        For shape order :math:`p \\ge 2`, builds the node potential integral
        :math:`\\Psi` of the normal electric field :math:`E_n^n` from the PEC
        wall node :math:`k_w` along ``axis`` on the wall stencil slab
        :math:`k_w \\dots k_w \\pm (p + 1)` (:math:`\\Psi[k_w] = 0`), gathers
        :math:`\\mathrm{d}K_i = q_i w_i \\sum_{\\text{nodes}} W^{(p)}(\\mathbf{x}_i)\\,\\Psi_{\\text{node}}`,
        and updates each injected particle's birth momentum via
        :math:`\\gamma'_i = \\gamma_i + \\mathrm{d}K_i / (w_i m c^2)`:

        - For moving particles (:math:`|\\mathbf{u}_i| > 0`), scales
          :math:`\\mathbf{u}_i` at constant direction so that
          :math:`|\\mathbf{u}'_i| / c = \\sqrt{\\gamma_i'^2 - 1}`.
        - For particles born at rest (:math:`|\\mathbf{u}_i| = 0`), applies the
          kick along the inward wall normal (:math:`+\\hat{\\mathbf{n}}` on
          ``side="low"``, :math:`-\\hat{\\mathbf{n}}` on ``side="high"``) with
          :math:`|\\mathbf{u}'_i| / c = \\sqrt{\\gamma_i'^2 - 1}`.

        To avoid per-step device-to-host synchronisations on GPU, ``_K_wall_kick``
        is accumulated as a 0-D device scalar (synchronised only when read, e.g.
        in :meth:`wall_kick_energy`), and the :math:`\\gamma'_i < 1` condition is
        OR-accumulated into the 0-D device flag ``_d_wall_kick_error`` and
        checked via :meth:`_check_wall_kick_error` every
        ``_wall_kick_check_every`` steps (default ``64``, including step ``0``).
        """
        if self.shape < 2 or not _ENABLE_PEC_WALL_KICK:
            return
        if self._wall_images is not None:
            # Image stencil: the effective potential sum_k W_k Psi_k of a particle on the wall
            # is identically zero (odd image), so the D17 kick vanishes.
            return
        s = self.solver
        xp = self.xp
        Nx, Ny, Nz = int(s.Nx), int(s.Ny), int(s.Nz)
        N = Nx * Ny * Nz
        E_arr = E_field.toarray() if hasattr(E_field, "toarray") else E_field
        E_n = E_arr[axis * N : (axis + 1) * N].reshape((Nx, Ny, Nz), order="F")
        k_w = self._pec_wall_node_index(axis, side)
        n_ax = int(E_n.shape[axis])
        dl = self.grid_map._deltas[axis]

        slab_shape = [Nx, Ny, Nz]
        dl_shape = [1, 1, 1]
        if side == "low":
            k_slab_start = k_w
            k_stop = min(n_ax, k_w + self.shape + 2)
            n_slab = max(1, k_stop - k_w)
            slab_shape[axis] = n_slab
            Psi_slab = xp.zeros(tuple(slab_shape), dtype=np.float64, order="F")
            if k_stop > k_w + 1:
                sl_edges = [slice(None), slice(None), slice(None)]
                sl_nodes = [slice(None), slice(None), slice(None)]
                sl_edges[axis] = slice(k_w, k_stop - 1)
                sl_nodes[axis] = slice(1, n_slab)
                dl_shape[axis] = k_stop - 1 - k_w
                dl_sub = dl[k_w : k_stop - 1].reshape(tuple(dl_shape))
                Psi_slab[tuple(sl_nodes)] = xp.cumsum(
                    E_n[tuple(sl_edges)] * dl_sub, axis=axis
                )
        elif side == "high":
            k_start = max(0, k_w - (self.shape + 1))
            k_slab_start = k_start
            n_slab = max(1, k_w - k_start + 1)
            slab_shape[axis] = n_slab
            Psi_slab = xp.zeros(tuple(slab_shape), dtype=np.float64, order="F")
            if k_w > k_start:
                sl_edges = [slice(None), slice(None), slice(None)]
                sl_nodes = [slice(None), slice(None), slice(None)]
                sl_edges[axis] = slice(k_start, k_w)
                sl_nodes[axis] = slice(0, k_w - k_start)
                dl_shape[axis] = k_w - k_start
                dl_sub = dl[k_start:k_w].reshape(tuple(dl_shape))
                Psi_slab[tuple(sl_nodes)] = -xp.flip(
                    xp.cumsum(
                        xp.flip(E_n[tuple(sl_edges)] * dl_sub, axis=axis),
                        axis=axis,
                    ),
                    axis=axis,
                )
        else:
            raise ValueError(f"Unsupported side={side!r}; expected 'low' or 'high'.")

        ix_nod, wx_nod = bspline_weights_1d(xi_i, self.shape, staggered=False, xp=xp)
        iy_nod, wy_nod = bspline_weights_1d(eta_i, self.shape, staggered=False, xp=xp)
        iz_nod, wz_nod = bspline_weights_1d(zeta_i, self.shape, staggered=False, xp=xp)
        i_nod = [ix_nod, iy_nod, iz_nod]
        if k_slab_start != 0:
            i_nod[axis] = i_nod[axis] - k_slab_start
        periodic_slab = list(self.boundaries.periodic)
        periodic_slab[axis] = False
        psi_p = _gather_component_3d(
            Psi_slab.ravel(order="F"),
            slab_shape[0],
            slab_shape[1],
            slab_shape[2],
            i_nod[0],
            wx_nod,
            i_nod[1],
            wy_nod,
            i_nod[2],
            wz_nod,
            tuple(periodic_slab),
            xp,
        )

        w_sl = sp_inj._w[sl_new]
        dK = (sp_inj.charge * w_sl) * psi_p
        ux_sl = sp_inj._ux[sl_new]
        uy_sl = sp_inj._uy[sl_new]
        uz_sl = sp_inj._uz[sl_new]
        u2 = ux_sl * ux_sl + uy_sl * uy_sl + uz_sl * uz_sl
        gamma = xp.sqrt(1.0 + u2 / (c_light * c_light))
        mc2 = float(sp_inj.mass) * (c_light * c_light)
        w_mc2 = w_sl * mc2
        dgamma = xp.where(w_sl != 0.0, dK / xp.where(w_sl != 0.0, w_mc2, 1.0), 0.0)
        gamma_prime = gamma + dgamma
        err_flag = getattr(self, "_d_wall_kick_error", None)
        if not isinstance(err_flag, xp.ndarray):
            err_flag = xp.zeros((), dtype=bool)
            self._d_wall_kick_error = err_flag
        err_flag |= xp.any(gamma_prime < 1.0)
        self._check_wall_kick_error(periodic=True)
        denom = xp.sqrt(gamma * gamma - 1.0)
        u_prime_over_c = xp.sqrt(xp.maximum(gamma_prime * gamma_prime - 1.0, 0.0))
        scale = xp.where(
            denom > 0.0,
            u_prime_over_c / xp.where(denom > 0.0, denom, 1.0),
            1.0,
        )
        inward_sign = 1.0 if side == "low" else -1.0
        u_rest_n = (inward_sign * c_light) * u_prime_over_c
        ux_new = ux_sl * scale
        uy_new = uy_sl * scale
        uz_new = uz_sl * scale
        if axis == 0:
            ux_new = xp.where(denom > 0.0, ux_new, u_rest_n)
        elif axis == 1:
            uy_new = xp.where(denom > 0.0, uy_new, u_rest_n)
        else:
            uz_new = xp.where(denom > 0.0, uz_new, u_rest_n)
        sp_inj._ux[sl_new] = ux_new
        sp_inj._uy[sl_new] = uy_new
        sp_inj._uz[sl_new] = uz_new
        if not isinstance(self._K_wall_kick, xp.ndarray):
            k_arr = xp.zeros((), dtype=np.float64)
            k_arr += self._K_wall_kick
            self._K_wall_kick = k_arr
        self._K_wall_kick += xp.sum(dK)

    def _record_absorbed_particles(
        self,
        xi1_dep,
        eta1_dep,
        zeta1_dep,
        qw,
        absorbed_mask,
        hit_code,
        W_tuple: tuple,
        q_bg,
    ) -> None:
        """Record absorbed particle charge in ``q_bg`` and apply PEC wall closure for PEC domain faces."""
        s = self.solver
        xp = self.xp
        Nx, Ny, Nz = s.Nx, s.Ny, s.Nz
        if self._q0_buf is None or self._q0_buf.shape != (Nx, Ny, Nz):
            self._q0_buf = xp.zeros((Nx, Ny, Nz), dtype=np.float64, order="F")
        q_abs_nodes = self._q0_buf

        if hit_code is None or hit_code.shape != absorbed_mask.shape:
            q_abs_nodes.fill(0.0)
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
            self._image_charge(
                xi1_dep[absorbed_mask],
                eta1_dep[absorbed_mask],
                zeta1_dep[absorbed_mask],
                qw[absorbed_mask],
                q_abs_nodes,
            )
            q_bg += q_abs_nodes
            return

        rem_mask = absorbed_mask.copy()
        faces = (
            (0, "low", 0, self.boundaries.bc_low[0]),
            (0, "high", 1, self.boundaries.bc_high[0]),
            (1, "low", 2, self.boundaries.bc_low[1]),
            (1, "high", 3, self.boundaries.bc_high[1]),
            (2, "low", 4, self.boundaries.bc_low[2]),
            (2, "high", 5, self.boundaries.bc_high[2]),
        )
        for axis_f, side_f, code_f, bc_f in faces:
            if bc_f not in ("pec", "electric"):
                continue
            face_mask = rem_mask & (hit_code == code_f)
            if not bool(xp.any(face_mask)):
                continue
            rem_mask = rem_mask & ~face_mask
            q_abs_nodes.fill(0.0)
            deposit_charge(
                xi1_dep[face_mask],
                eta1_dep[face_mask],
                zeta1_dep[face_mask],
                qw[face_mask],
                Nx,
                Ny,
                Nz,
                order=self.shape,
                periodic=self.boundaries.periodic,
                q_nodes=q_abs_nodes,
                xp=xp,
            )
            self._image_charge(
                xi1_dep[face_mask],
                eta1_dep[face_mask],
                zeta1_dep[face_mask],
                qw[face_mask],
                q_abs_nodes,
            )
            self._apply_pec_wall_event(
                q_abs_nodes,
                axis=axis_f,
                side=side_f,
                event="absorb",
                W_tuple=W_tuple,
                q_bg=q_bg,
            )

        if bool(xp.any(rem_mask)):
            q_abs_nodes.fill(0.0)
            deposit_charge(
                xi1_dep[rem_mask],
                eta1_dep[rem_mask],
                zeta1_dep[rem_mask],
                qw[rem_mask],
                Nx,
                Ny,
                Nz,
                order=self.shape,
                periodic=self.boundaries.periodic,
                q_nodes=q_abs_nodes,
                xp=xp,
            )
            self._image_charge(
                xi1_dep[rem_mask],
                eta1_dep[rem_mask],
                zeta1_dep[rem_mask],
                qw[rem_mask],
                q_abs_nodes,
            )
            q_bg += q_abs_nodes

    def _record_initial_gauss_state(self) -> None:
        """Capture initial background charge ``q_bg = S_tilde d_tilde(0) - q_nodes(0)``."""
        s = self.solver
        xp = self.xp
        local_exc = None
        try:
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
                    out_scratch = (
                        (
                            sp._xi0_scratch[:n_l],
                            sp._eta0_scratch[:n_l],
                            sp._zeta0_scratch[:n_l],
                        )
                        if sp._gpu_scratch_cap >= n_l
                        else None
                    )
                    xi, eta, zeta = self.grid_map.to_logical(
                        sp._x[:n_l],
                        sp._y[:n_l],
                        sp._z[:n_l],
                        out=out_scratch,
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
                    if self._wall_images is not None:
                        n_act = int(sp.n_particles)
                        self._image_charge(
                            xi[:n_act],
                            eta[:n_act],
                            zeta[:n_act],
                            sp.charge * sp._w[:n_act],
                            q0,
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
                    self._image_charge(xi, eta, zeta, sp.charge * sp.w, q0)
        except Exception as exc:
            local_exc = exc
        self._collective_check(local_exc)
        if self.use_mpi and int(getattr(s, "size", 1)) > 1:
            _mpi_sum_node_guards(s, q0)
        local_exc = None
        try:
            div_d0 = compute_dual_divergence_d(s)
            q_bg = getattr(s, "_particles_q_bg", None)
            if q_bg is not None and q_bg.shape == q0.shape and q_bg.flags.f_contiguous:
                q_bg[...] = div_d0 - q0
            else:
                s._particles_q_bg = xp.asarray(
                    div_d0 - q0, dtype=np.float64, order="F"
                )
            if self.use_mpi and int(getattr(s, "size", 1)) > 1:
                rank = int(s.rank)
                size = int(s.size)
                if rank > 0:
                    s._particles_q_bg[:, :, 0] = 0.0
                if rank < size - 1:
                    s._particles_q_bg[:, :, self._k_own_end :] = 0.0
            if self.on_gpu:
                s._particles_q_scale_ref = xp.max(xp.abs(q0))
            else:
                vac_mask = _interior_vacuum_node_mask(s)
                q_max = (
                    float(xp.max(xp.abs(q0[vac_mask])))
                    if bool(xp.any(vac_mask))
                    else 0.0
                )
                s._particles_q_scale_ref = max(q_max, float(xp.max(xp.abs(q0))), 0.0)
        except Exception as exc:
            local_exc = exc
        self._collective_check(local_exc)
        if self.use_mpi and int(getattr(s, "size", 1)) > 1:
            from mpi4py import MPI

            if self.on_gpu:
                q_loc = float(s._particles_q_scale_ref.get())
                s._particles_q_scale_ref = float(
                    s.comm.allreduce(q_loc, op=MPI.MAX)
                )
            else:
                s._particles_q_scale_ref = float(
                    s.comm.allreduce(s._particles_q_scale_ref, op=MPI.MAX)
                )

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

    def _image_field(self, xi, eta, zeta, E_field, B_tuple):
        """Image-stencil part of the gathered ``(E, B)`` at logical positions, or ``None``.

        Non-``None`` only with ``wall_stencil="image"`` and particles within the stencil margin
        of a planar PEC domain face (see :mod:`wakis.particles.wall_images`).
        """
        wi = self._wall_images
        if wi is None:
            return None
        s = self.solver
        return wi.field_correction(
            xi,
            eta,
            zeta,
            E_field,
            B_tuple,
            s.Nx,
            s.Ny,
            s.Nz,
            self.gather,
            self.boundaries.periodic,
            xp=self.xp,
        )

    def _image_charge(self, xi, eta, zeta, qw, q_nodes) -> None:
        """Add the image charges [C] of particles at logical ``(xi, eta, zeta)`` to ``q_nodes``."""
        wi = self._wall_images
        if wi is None:
            return
        s = self.solver
        wi.deposit_charge(
            xi,
            eta,
            zeta,
            qw,
            s.Nx,
            s.Ny,
            s.Nz,
            self.boundaries.periodic,
            q_nodes,
            xp=self.xp,
        )

    @staticmethod
    def _add_image_field(fields6, corr):
        """Return ``fields6 + corr`` component-wise (``fields6`` unchanged if ``corr`` is None)."""
        if corr is None:
            return tuple(fields6)
        return tuple(f + c for f, c in zip(fields6, corr))

    def _gpu_ext_arrays_with_images(self, sp, n_l, E_arr, B_arr, E_flat, B_flat):
        """Merge the image-stencil field into the per-particle external-field arrays (GPU)."""
        if self._wall_images is None:
            return E_arr, B_arr
        xp = self.xp
        s = self.solver
        N = s.Nx * s.Ny * s.Nz
        n_act = min(int(sp.n_particles), int(n_l))
        if n_act == 0:
            return E_arr, B_arr
        xi, eta, zeta = self.grid_map.to_logical(
            sp._x[:n_act], sp._y[:n_act], sp._z[:n_act]
        )
        corr = self._image_field(
            xi,
            eta,
            zeta,
            (E_flat[:N], E_flat[N : 2 * N], E_flat[2 * N : 3 * N]),
            (B_flat[:N], B_flat[N : 2 * N], B_flat[2 * N : 3 * N]),
        )
        if corr is None:
            return E_arr, B_arr
        full = []
        for c in corr:
            a = xp.zeros(int(n_l), dtype=np.float64)
            a[:n_act] = c
            full.append(a)
        E_new = tuple(full[:3]) if E_arr is None else tuple(a + b for a, b in zip(E_arr, full[:3]))
        B_new = tuple(full[3:]) if B_arr is None else tuple(a + b for a, b in zip(B_arr, full[3:]))
        return E_new, B_new

    def _initialize_species_half_step(self, sp: Species, B_n_tuple, t_n: float) -> None:
        """Back-push ``u^0 -> u^{-1/2}`` by ``-dt / 2`` on step 0 if ``sp.u_is_synced``."""
        if not sp.u_is_synced:
            return
        if sp._n_launch_bound == 0:
            sp.u_is_synced = False
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
            E_arr, B_arr = self._gpu_ext_arrays_with_images(
                sp, n_l, E_arr, B_arr, s.E.toarray(), self._B_n_buf
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
                Ex_c, Ey_c, Ez_c, Bx_c, By_c, Bz_c = self._add_image_field(
                    (Ex_c, Ey_c, Ez_c, Bx_c, By_c, Bz_c),
                    self._image_field(xi_c, eta_c, zeta_c, s.E, B_n_tuple),
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
        Ex, Ey, Ez, Bx, By, Bz = self._add_image_field(
            (Ex, Ey, Ez, Bx, By, Bz),
            self._image_field(xi0, eta0, zeta0, s.E, B_n_tuple),
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

        do_profile = self.profile
        t_pg_acc = 0.0
        t_dep_acc = 0.0
        t_grd_acc = 0.0
        t_mig_acc = 0.0
        if do_profile:
            xp.cuda.Stream.null.synchronize()
            t0_b = time.perf_counter()

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
        if self.track_energy:
            E_flat = s.E.toarray()
            if self._fw_pending:
                self._flush_pending_field_work(E_flat)
            xp.copyto(self._E_prev_fw, E_flat)
            self._fw_dt = dt
        if do_profile:
            xp.cuda.Stream.null.synchronize()
            t_pg_acc += time.perf_counter() - t0_b

        use_mpi_multi = self.use_mpi and int(getattr(s, "size", 1)) > 1
        self._local_step_exc = None
        for inj in self.injectors:
            if inj.emitter_kind is None:
                local_exc = None
                try:
                    inj._bind(s)
                except Exception as exc:  # noqa: BLE001
                    local_exc = exc
                self._collective_check(local_exc)
        Wx_tot = self._Wx_buf
        Wy_tot = self._Wy_buf
        Wz_tot = self._Wz_buf
        try:
            if not self._initialized_half_step:
                if self.track_energy:
                    xp.copyto(self._E_init_fw, E_flat)
                    xp.copyto(self._H_prev_init_fw, H_prev)
                    xp.copyto(self._H_curr_init_fw, H_curr)
                    self._has_step0_field_snap = True
                if (
                    tuple(sp._n_launch_bound for sp in self.species)
                    != self._init_gauss_counts
                ):
                    if self.use_mpi:
                        self._filter_species_to_owned_slab()
                    if not use_mpi_multi:
                        self._record_initial_gauss_state()
                    self._init_gauss_counts = tuple(
                        sp._n_launch_bound for sp in self.species
                    )
                for sp in self.species:
                    self._initialize_species_half_step(sp, B_n_tuple, t_n)
                self._initialized_half_step = True

            if do_profile:
                xp.cuda.Stream.null.synchronize()
                t0_zero = time.perf_counter()
            Wx_tot.fill(0.0)
            Wy_tot.fill(0.0)
            Wz_tot.fill(0.0)
            if self.track_energy:
                self._W_cl.fill(0.0)
                self._cl_step_active = False
            if do_profile:
                xp.cuda.Stream.null.synchronize()
                t_dep_acc += time.perf_counter() - t0_zero

            for inj in self.injectors:
                if self.use_mpi and not self._owns_z_coord(inj.z0):
                    inj.inject(t_n, dt, step_index=self.n_particle_steps, owning=False)
                    continue
                n_before = inj.species.n_particles
                n_inj = inj.inject(t_n, dt, step_index=self.n_particle_steps)
                if n_inj > 0:
                    sp_inj = inj.species
                    sl_new = slice(n_before, sp_inj.n_particles)
                    xi_i, eta_i, zeta_i = self.grid_map.to_logical(
                        sp_inj._x[sl_new], sp_inj._y[sl_new], sp_inj._z[sl_new]
                    )
                    if self._q0_buf is None or self._q0_buf.shape != (
                        Nx,
                        Ny,
                        Nz,
                    ):
                        self._q0_buf = xp.zeros(
                            (Nx, Ny, Nz), dtype=np.float64, order="F"
                        )
                    q_inj_nodes = self._q0_buf
                    q_inj_nodes.fill(0.0)
                    deposit_charge_gpu(
                        xi_i,
                        eta_i,
                        zeta_i,
                        sp_inj._w[sl_new],
                        sp_inj.charge,
                        Nx,
                        Ny,
                        Nz,
                        order=self.shape,
                        periodic=self.boundaries.periodic,
                        q_nodes=q_inj_nodes,
                    )
                    self._image_charge(
                        xi_i,
                        eta_i,
                        zeta_i,
                        sp_inj.charge * sp_inj._w[sl_new],
                        q_inj_nodes,
                    )
                    if inj.emitter_kind == "pec_wall":
                        axis_inj, side_inj = self._injector_pec_wall_face(inj)
                        self._apply_pec_wall_event(
                            q_inj_nodes,
                            axis=axis_inj,
                            side=side_inj,
                            event="inject",
                            W_tuple=(Wx_tot, Wy_tot, Wz_tot),
                            q_bg=s._particles_q_bg,
                        )
                        self._apply_pec_wall_birth_kick(
                            sp_inj,
                            sl_new,
                            xi_i,
                            eta_i,
                            zeta_i,
                            s.E,
                            axis=axis_inj,
                            side=side_inj,
                        )
                    else:
                        s._particles_q_bg -= q_inj_nodes
                    self._backpush_injected_slice(
                        sp_inj,
                        sl_new,
                        xi_i,
                        eta_i,
                        zeta_i,
                        s.E,
                        B_n_tuple,
                        t_n,
                        dt,
                    )
                    s._particles_q_scale_ref = abs(inj.injected_charge)
                    for col in self.boundaries.collectors:
                        col.current_history.append((float(t_n + dt), 0.0))
        except Exception as exc:
            if not (use_mpi_multi and self._enable_step_status_piggyback):
                raise
            self._local_step_exc = exc

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
            t_g1 = None
            if self._local_step_exc is None:
                try:
                    self._mpi_pre_push_hook(sp)
                    n_l = sp._n_launch_bound
                    if n_l > 0:
                        if (
                            self.sort_every > 0
                            and (self.n_particle_steps > 0)
                            and (self.n_particle_steps % self.sort_every == 0)
                        ):
                            sp.sort_by_cell(self.grid_map)

                        if do_profile:
                            xp.cuda.Stream.null.synchronize()
                            t_g0 = time.perf_counter()

                        E_uni, E_arr = self._ext_field_args_gpu(
                            self.E_ext, sp._x[:n_l], sp._y[:n_l], sp._z[:n_l], t_n
                        )
                        B_uni, B_arr = self._ext_field_args_gpu(
                            self.B_ext, sp._x[:n_l], sp._y[:n_l], sp._z[:n_l], t_n
                        )
                        E_arr, B_arr = self._gpu_ext_arrays_with_images(
                            sp, n_l, E_arr, B_arr, E_flat, B_n_flat
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
                        if do_profile:
                            xp.cuda.Stream.null.synchronize()
                            t_g1 = time.perf_counter()
                            t_pg_acc += t_g1 - t_g0

                        if all_periodic_no_obs:
                            xi1_dep = sp._xi1_scratch
                            eta1_dep = sp._eta1_scratch
                            zeta1_dep = sp._zeta1_scratch
                            keep_mask_i32 = None
                        elif not has_custom_mask_col and self._wall_images is None:
                            use_cl_gpu = bool(
                                self.track_energy
                                and _ENABLE_PEC_WALL_CLOSURE
                                and self._W_cl is not None
                            )
                            xi1_dep, eta1_dep, zeta1_dep, keep_mask_i32 = (
                                launch_boundary_prepare_paths_gpu(
                                    self.boundaries,
                                    sp,
                                    order=self.shape,
                                    q_bg_nodes=s._particles_q_bg,
                                    Wx_out=Wx_tot,
                                    Wy_out=Wy_tot,
                                    Wz_out=Wz_tot,
                                    enable_closure=_ENABLE_PEC_WALL_CLOSURE,
                                    W_cl_out=self._W_cl if use_cl_gpu else None,
                                )
                            )
                            if (
                                use_cl_gpu
                                and self._has_pec_domain_wall
                                and self.boundaries._poll_pec_wall_absorbed_gpu()
                            ):
                                self._cl_step_active = True
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
                            absorbed_b = ~keep_mask_b
                            if bool(xp.any(absorbed_b)):
                                self._record_absorbed_particles(
                                    xi1_dep,
                                    eta1_dep,
                                    zeta1_dep,
                                    sp.charge * sp._w[:n_act],
                                    absorbed_b,
                                    getattr(self.boundaries, "_last_hit_code", None),
                                    (Wx_tot, Wy_tot, Wz_tot),
                                    s._particles_q_bg,
                                )
                            sp._keep_mask_scratch[:n_act] = keep_mask_b.astype(
                                np.int32
                            )
                            keep_mask_i32 = sp._keep_mask_scratch

                        d_n_ptr = (
                            sp._d_n_particles if sp._n_dirty_gpu else None
                        )
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
                                self._Nz_dep,
                                order=self.shape,
                                periodic=self.boundaries.periodic,
                                Wx_out=Wx_tot,
                                Wy_out=Wy_tot,
                                Wz_out=Wz_tot,
                                d_n_particles=d_n_ptr,
                                n_launch=n_l,
                                d_cfl_error=self._d_cfl_error,
                            )
                            if self._wall_images is not None:
                                n_act = int(sp.n_particles)
                                self._wall_images.deposit_current(
                                    sp._xi0_scratch[:n_act],
                                    sp._eta0_scratch[:n_act],
                                    sp._zeta0_scratch[:n_act],
                                    xi1_dep[:n_act],
                                    eta1_dep[:n_act],
                                    zeta1_dep[:n_act],
                                    sp.charge * sp._w[:n_act],
                                    Nx,
                                    Ny,
                                    self._Nz_dep,
                                    self.boundaries.periodic,
                                    (Wx_tot, Wy_tot, Wz_tot),
                                    xp=xp,
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
                                self._Nz_dep,
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
                except Exception as exc:
                    if not (
                        use_mpi_multi and self._enable_step_status_piggyback
                    ):
                        raise
                    self._local_step_exc = exc

            if do_profile:
                xp.cuda.Stream.null.synchronize()
                t_m0 = time.perf_counter()
                if t_g1 is not None:
                    t_dep_acc += t_m0 - t_g1
            self._mpi_post_push_hook(sp)
            if do_profile:
                xp.cuda.Stream.null.synchronize()
                t_mig_acc += time.perf_counter() - t_m0

        if do_profile:
            xp.cuda.Stream.null.synchronize()
            t_fin0 = time.perf_counter()
        if self.use_mpi and int(getattr(s, "size", 1)) > 1:
            self.J_particles[:N] = Wx_tot[:N]
            self.J_particles[N : 2 * N] = Wy_tot[:N]
            self.J_particles[2 * N : 3 * N] = Wz_tot[:N]
            if do_profile:
                xp.cuda.Stream.null.synchronize()
                t_gr0 = time.perf_counter()
                t_dep_acc += t_gr0 - t_fin0
            self._mpi_sum_guards(self.J_particles)
            if do_profile:
                xp.cuda.Stream.null.synchronize()
                t_fin0 = time.perf_counter()
                t_grd_acc += t_fin0 - t_gr0
            Wx_tot = self.J_particles[:N]
            Wy_tot = self.J_particles[N : 2 * N]
            Wz_tot = self.J_particles[2 * N : 3 * N]

        if self.filter:
            if self.use_mpi:
                raise NotImplementedError(
                    "Binomial current filtering is not supported with use_mpi=True."
                )
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
        if self.track_energy:
            self._finalize_closure_current(dt)
        if do_profile:
            xp.cuda.Stream.null.synchronize()
            t_dep_acc += time.perf_counter() - t_fin0
            self._last_push_gather_s = t_pg_acc
            self._last_deposit_s = t_dep_acc
            self._last_guard_reduce_s = t_grd_acc
            self._last_migrate_s = t_mig_acc
        else:
            self._last_push_gather_s = 0.0
            self._last_deposit_s = 0.0
            self._last_guard_reduce_s = 0.0
            self._last_migrate_s = 0.0
        if self.track_energy:
            self._fw_pending = True
        self.n_particle_steps += 1
        self._run_diagnostics(t_n + dt, dt)

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
        t_grd_acc = 0.0
        t_mig_acc = 0.0
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
        if self.track_energy:
            E_flat = s.E.toarray()
            if self._fw_pending:
                self._flush_pending_field_work(E_flat)
            xp.copyto(self._E_prev_fw, E_flat)
            self._fw_dt = dt
        if do_profile:
            t_pg_acc += time.perf_counter() - t0_b

        use_mpi_multi = self.use_mpi and int(getattr(s, "size", 1)) > 1
        self._local_step_exc = None
        for inj in self.injectors:
            if inj.emitter_kind is None:
                local_exc = None
                try:
                    inj._bind(s)
                except Exception as exc:  # noqa: BLE001
                    local_exc = exc
                self._collective_check(local_exc)
        Wx_tot = self._Wx_buf
        Wy_tot = self._Wy_buf
        Wz_tot = self._Wz_buf
        try:
            if not self._initialized_half_step:
                if self.track_energy:
                    xp.copyto(self._E_init_fw, E_flat)
                    xp.copyto(self._H_prev_init_fw, H_prev)
                    xp.copyto(self._H_curr_init_fw, H_curr)
                    self._has_step0_field_snap = True
                if (
                    tuple(sp._n_launch_bound for sp in self.species)
                    != self._init_gauss_counts
                ):
                    if self.use_mpi:
                        self._filter_species_to_owned_slab()
                    if not use_mpi_multi:
                        self._record_initial_gauss_state()
                    self._init_gauss_counts = tuple(
                        sp._n_launch_bound for sp in self.species
                    )
                for sp in self.species:
                    self._initialize_species_half_step(sp, B_n_tuple, t_n)
                self._initialized_half_step = True

            if do_profile:
                t0_zero = time.perf_counter()
            if self._j_in_solver:
                s.J.array -= self.J_particles
            Wx_tot.fill(0.0)
            Wy_tot.fill(0.0)
            Wz_tot.fill(0.0)
            if self.track_energy:
                self._W_cl.fill(0.0)
                self._cl_step_active = False
            if do_profile:
                t_dep_acc += time.perf_counter() - t0_zero

            # Inject new macroparticles at emitter planes after W_tot is zeroed so
            # PEC-wall injectors can add the normal-axis wall-closure current directly
            # into W_tot for this step.
            for inj in self.injectors:
                if self.use_mpi and not self._owns_z_coord(inj.z0):
                    inj.inject(t_n, dt, step_index=self.n_particle_steps, owning=False)
                    continue
                n_before = inj.species.n_particles
                n_inj = inj.inject(t_n, dt, step_index=self.n_particle_steps)
                if n_inj > 0:
                    sp_inj = inj.species
                    sl_new = slice(n_before, sp_inj.n_particles)
                    xi_i, eta_i, zeta_i = self.grid_map.to_logical(
                        sp_inj.x[sl_new], sp_inj.y[sl_new], sp_inj.z[sl_new]
                    )
                    if self._q0_buf is None or self._q0_buf.shape != (
                        Nx,
                        Ny,
                        Nz,
                    ):
                        self._q0_buf = xp.zeros(
                            (Nx, Ny, Nz), dtype=np.float64, order="F"
                        )
                    q_inj_nodes = self._q0_buf
                    q_inj_nodes.fill(0.0)
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
                    self._image_charge(
                        xi_i,
                        eta_i,
                        zeta_i,
                        sp_inj.charge * sp_inj.w[sl_new],
                        q_inj_nodes,
                    )
                    if inj.emitter_kind == "pec_wall":
                        axis_inj, side_inj = self._injector_pec_wall_face(inj)
                        self._apply_pec_wall_event(
                            q_inj_nodes,
                            axis=axis_inj,
                            side=side_inj,
                            event="inject",
                            W_tuple=(Wx_tot, Wy_tot, Wz_tot),
                            q_bg=s._particles_q_bg,
                        )
                        self._apply_pec_wall_birth_kick(
                            sp_inj,
                            sl_new,
                            xi_i,
                            eta_i,
                            zeta_i,
                            s.E,
                            axis=axis_inj,
                            side=side_inj,
                        )
                    else:
                        s._particles_q_bg -= q_inj_nodes
                    self._backpush_injected_slice(
                        sp_inj,
                        sl_new,
                        xi_i,
                        eta_i,
                        zeta_i,
                        s.E,
                        B_n_tuple,
                        t_n,
                        dt,
                    )
                    if getattr(s, "_particles_q_scale_ref", 0.0) == 0.0:
                        s._particles_q_scale_ref = abs(inj.injected_charge)
        except Exception as exc:
            if not (use_mpi_multi and self._enable_step_status_piggyback):
                raise
            self._local_step_exc = exc

        for sp in self.species:
            t_g1 = None
            if self._local_step_exc is None:
                try:
                    self._mpi_pre_push_hook(sp)
                    if sp.n_particles > 0:
                        if (
                            self.sort_every > 0
                            and (self.n_particle_steps > 0)
                            and (self.n_particle_steps % self.sort_every == 0)
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
                                xi0_c, eta0_c, zeta0_c = (
                                    self.grid_map.to_logical(x0_c, y0_c, z0_c)
                                )
                                Ex_c, Ey_c, Ez_c, Bx_c, By_c, Bz_c = (
                                    gather_fields(
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
                                    1.0
                                    + (ux_c * ux_c + uy_c * uy_c + uz_c * uz_c)
                                    * _INV_C2
                                )
                                sp._x[s0:s1] = x0_c + fac * ux_c
                                sp._y[s0:s1] = y0_c + fac * uy_c
                                sp._z[s0:s1] = z0_c + fac * uz_c
                                xi1_c, eta1_c, zeta1_c = (
                                    self.grid_map.to_logical(
                                        sp._x[s0:s1],
                                        sp._y[s0:s1],
                                        sp._z[s0:s1],
                                    )
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
                                    self._Nz_dep,
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
                        Ex, Ey, Ez, Bx, By, Bz = self._add_image_field(
                            (Ex, Ey, Ez, Bx, By, Bz),
                            self._image_field(xi0, eta0, zeta0, s.E, B_n_tuple),
                        )
                        Ex_e, Ey_e, Ez_e = self._eval_ext_field(
                            self.E_ext, x0, y0, z0, t_n
                        )
                        Bx_e, By_e, Bz_e = self._eval_ext_field(
                            self.B_ext, x0, y0, z0, t_n
                        )

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
                        xi1, eta1, zeta1 = self.grid_map.to_logical(
                            sp.x, sp.y, sp.z
                        )
                        if do_profile:
                            t_g1 = time.perf_counter()
                            t_pg_acc += t_g1 - t_g0

                        # 4. Boundary truncation (absorb at domain walls / CPML / PEC / collectors)
                        xi1_dep, eta1_dep, zeta1_dep, keep_mask = (
                            self.boundaries.prepare_step_paths(
                                sp,
                                x0,
                                y0,
                                z0,
                                xi0,
                                eta0,
                                zeta0,
                                xi1,
                                eta1,
                                zeta1,
                                t=t_n + dt,
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
                                self._Nz_dep,
                                order=self.shape,
                                periodic=self.boundaries.periodic,
                                Wx_out=Wx_tot,
                                Wy_out=Wy_tot,
                                Wz_out=Wz_tot,
                                xp=xp,
                            )
                            if self._wall_images is not None:
                                self._wall_images.deposit_current(
                                    xi0,
                                    eta0,
                                    zeta0,
                                    xi1_dep,
                                    eta1_dep,
                                    zeta1_dep,
                                    qw,
                                    Nx,
                                    Ny,
                                    self._Nz_dep,
                                    self.boundaries.periodic,
                                    (Wx_tot, Wy_tot, Wz_tot),
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
                                self._Nz_dep,
                                order=self.shape,
                                periodic=self.boundaries.periodic,
                                xp=xp,
                            )
                            Wx_tot += Wx_n
                            Wy_tot += Wy_n
                            Wz_tot += Wz_n

                        # 6. Remove absorbed particles (recording their deposited surface
                        # charge at the absorption point in _particles_q_bg, and adding the
                        # normal-axis PEC wall-closure current when absorbed on a PEC domain wall).
                        absorbed_mask = ~keep_mask
                        if bool(xp.any(absorbed_mask)):
                            self._record_absorbed_particles(
                                xi1_dep,
                                eta1_dep,
                                zeta1_dep,
                                qw,
                                absorbed_mask,
                                getattr(self.boundaries, "_last_hit_code", None),
                                (Wx_tot, Wy_tot, Wz_tot),
                                s._particles_q_bg,
                            )
                        sp.keep(keep_mask)
                        self.boundaries.apply_periodic_wrap(sp)
                except Exception as exc:
                    if not (
                        use_mpi_multi and self._enable_step_status_piggyback
                    ):
                        raise
                    self._local_step_exc = exc

            if do_profile:
                t_m0 = time.perf_counter()
                if t_g1 is not None:
                    t_dep_acc += t_m0 - t_g1
            self._mpi_post_push_hook(sp)
            if do_profile:
                t_mig_acc += time.perf_counter() - t_m0

        if do_profile:
            t_fin0 = time.perf_counter()
        if self.use_mpi and int(getattr(s, "size", 1)) > 1:
            self.J_particles[:N] = Wx_tot[:N]
            self.J_particles[N : 2 * N] = Wy_tot[:N]
            self.J_particles[2 * N : 3 * N] = Wz_tot[:N]
            if do_profile:
                t_gr0 = time.perf_counter()
                t_dep_acc += t_gr0 - t_fin0
            self._mpi_sum_guards(self.J_particles)
            if do_profile:
                t_fin0 = time.perf_counter()
                t_grd_acc += t_fin0 - t_gr0
            Wx_tot = self.J_particles[:N]
            Wy_tot = self.J_particles[N : 2 * N]
            Wz_tot = self.J_particles[2 * N : 3 * N]

        if self.filter:
            if self.use_mpi:
                raise NotImplementedError(
                    "Binomial current filtering is not supported with use_mpi=True."
                )
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
        if self.track_energy:
            self._finalize_closure_current(dt)
        if do_profile:
            t_dep_acc += time.perf_counter() - t_fin0
            self._last_push_gather_s = t_pg_acc
            self._last_deposit_s = t_dep_acc
            self._last_guard_reduce_s = t_grd_acc
            self._last_migrate_s = t_mig_acc
        else:
            self._last_push_gather_s = 0.0
            self._last_deposit_s = 0.0
            self._last_guard_reduce_s = 0.0
            self._last_migrate_s = 0.0
        if self.track_energy:
            self._fw_pending = True
        self.n_particle_steps += 1
        self._run_diagnostics(t_n + dt, dt)

    # --- Diagnostics ---------------------------------------------------------

    def add_diagnostic(self, diag):
        """Attach a particle diagnostic monitor to this solver.

        Parameters
        ----------
        diag : object
            Diagnostic monitor instance from :mod:`wakis.particles.diagnostics`.

        Returns
        -------
        object
            The attached diagnostic ``diag``.
        """
        if hasattr(diag, "_bind"):
            diag._bind(self)
        self.diagnostics.append(diag)
        return diag

    def _run_diagnostics(self, t: float, dt: float) -> None:
        """Advance all attached diagnostic monitors at the end of a particle step."""
        step = self.n_particle_steps - 1
        for diag in self.diagnostics:
            if getattr(diag, "_always_record", False) or step % diag.every == 0:
                diag._record(self, t, dt, step)

    def _staggered_field_energy(self, E_arr=None, H_prev=None, H_next=None) -> float:
        """Return half-step-staggered discrete FIT electromagnetic energy [J].

        Evaluates

        .. math::

            W^n = \\frac{1}{2}\\left(
                \\mathbf{e}^n \\cdot \\mathbf{M}_\\varepsilon \\mathbf{e}^n
                + \\mathbf{h}^{n-1/2} \\cdot \\mathbf{M}_\\mu \\mathbf{h}^{n+1/2}
            \\right),

        where :math:`\\mathbf{h}^{n+1/2} = \\mathbf{h}^{n-1/2} - \\Delta t\\,
        \\mathbf{M}_\\mu^{-1}\\tilde{\\mathbf{L}}\\mathbf{C}\\mathbf{L}\\mathbf{e}^n`.
        Unlike :meth:`field_energy` (which evaluates the un-staggered quadratic
        form :math:`\\frac{1}{2}(\\mathbf{e}^n \\cdot \\mathbf{M}_\\varepsilon \\mathbf{e}^n
        + \\mathbf{h}^{n-1/2} \\cdot \\mathbf{M}_\\mu \\mathbf{h}^{n-1/2})` via
        ``solver.field_energy()``), this staggered bilinear form satisfies the
        exact discrete Poynting theorem over each leapfrog step. Under MPI
        (``use_mpi=True`` and ``solver.size > 1``), this method is collective
        across ``solver.comm`` (via halo communication for :math:`\\mathbf{h}^{n+1/2}`
        and ``MPI.SUM`` reduction). Convolutional CPML auxiliary :math:`\\boldsymbol{\\psi}`
        fields are not included in :math:`W^n`.
        """
        s = self.solver
        xp = self.xp
        if E_arr is None:
            E_arr = s.E.toarray()
        if H_prev is None:
            H_prev = s.H.toarray()
        if H_next is None:
            if hasattr(s, "tDsiDmuiDaC"):
                H_next = H_prev - float(s.dt) * (s.tDsiDmuiDaC * E_arr)
                if self.use_mpi and int(getattr(s, "size", 1)) > 1:
                    from ..field import Field

                    H_tmp = Field(s.Nx, s.Ny, s.Nz, use_gpu=s.use_gpu, dtype=s.dtype)
                    H_tmp.fromarray(H_next)
                    s._mpi_communicate(H_tmp)
                    H_next = H_tmp.toarray()
            else:
                H_next = H_prev
        w_e = 0.5 * float(xp.sum(self._m_eps_arr * E_arr * E_arr))
        w_h = 0.5 * float(xp.sum(self._m_mu_arr * H_prev * H_next))
        w_tot = w_e + w_h
        if self.use_mpi and int(getattr(s, "size", 1)) > 1:
            from mpi4py import MPI

            w_tot = float(s.comm.allreduce(w_tot, op=MPI.SUM))
        return w_tot

    def field_work(self) -> float:
        """Cumulative work [J] done by the electromagnetic field on the particle current.

        Returns the discrete time integral accumulated over all completed steps:

        .. math::

            W_{\\text{work}} = \\sum_{m=0}^{n-1} \\Delta t \\sum_{e} V_{E,e}\\, J_e^{m+1/2}
            \\frac{E_e^m + E_e^{m+1}}{2},

        where :math:`V_{E,e} = L_e \\tilde{A}_e` is the FIT edge dual-cell volume,
        :math:`J_e^{m+1/2}` is the deposited particle current density on primal
        edge :math:`e`, and :math:`E_e^m, E_e^{m+1}` are the electric field edge
        values before and after step :math:`m`'s Ampère–Maxwell update. Under MPI
        (``use_mpi=True`` and ``solver.size > 1``), this method is collective
        across ``solver.comm`` (``MPI.SUM`` reduction).

        Returns
        -------
        float
            Cumulative discrete field work [J].

        Raises
        ------
        RuntimeError, ValueError
            If the :class:`ParticleSolver` was initialized with
            ``track_energy=False`` (the default).
        """
        if not self.track_energy:
            raise _EnergyTrackingDisabledError(
                "ParticleSolver.field_work() requires track_energy=True."
            )
        if self._fw_pending:
            self._flush_pending_field_work(self.solver.E.toarray())
        w_loc = float(self.xp.sum(self._fw_accum_vec * self._vol_E_fw))
        if self.use_mpi and int(getattr(self.solver, "size", 1)) > 1:
            from mpi4py import MPI

            w_loc = float(self.solver.comm.allreduce(w_loc, op=MPI.SUM))
        return w_loc

    def wall_closure_work(self) -> float:
        """Cumulative work [J] done by the electromagnetic field on PEC wall-closure currents.

        Returns the discrete time integral accumulated over all completed steps:

        .. math::

            W_{\\text{wall\\_closure}} = \\sum_{m=0}^{n-1} \\Delta t \\sum_{e} V_{E,e}\\,
            J_{\\text{cl},e}^{m+1/2} \\frac{E_e^m + E_e^{m+1}}{2},

        where :math:`V_{E,e} = L_e \\tilde{A}_e` is the FIT edge dual-cell volume
        and :math:`J_{\\text{cl},e}^{m+1/2} = W_{\\text{cl},e}^{m+1/2} / (\\Delta t\\, \\tilde{A}_e)`
        is the PEC wall-closure current density on primal edge :math:`e`. By this
        sign convention, :math:`W_{\\text{wall\\_closure}} > 0` means the
        electromagnetic field did positive work on the wall-closure current
        (energy delivered from the field to the wall/emitter), and
        :math:`W_{\\text{wall\\_closure}} < 0` means the wall-closure current
        delivered energy to the field. Under MPI (``use_mpi=True`` and
        ``solver.size > 1``), this method is collective across ``solver.comm``
        (``MPI.SUM`` reduction).

        Returns
        -------
        float
            Cumulative discrete wall-closure field work [J].

        Raises
        ------
        RuntimeError, ValueError
            If the :class:`ParticleSolver` was initialized with
            ``track_energy=False`` (the default).
        """
        if not self.track_energy:
            raise _EnergyTrackingDisabledError(
                "ParticleSolver.wall_closure_work() requires track_energy=True."
            )
        if self._fw_pending:
            self._flush_pending_field_work(self.solver.E.toarray())
        w_loc = float(self.xp.sum(self._cl_work_vec * self._vol_E_fw))
        if self.use_mpi and int(getattr(self.solver, "size", 1)) > 1:
            from mpi4py import MPI

            w_loc = float(self.solver.comm.allreduce(w_loc, op=MPI.SUM))
        return w_loc

    def wall_kick_energy(self) -> float:
        """Cumulative kinetic-energy change [J] applied at PEC-wall birth kicks.

        Returns the sum over all completed PEC-wall injection events (for shape
        order :math:`p \\ge 2`) of

        .. math::

            K_{\\text{wall\\_kick}} = \\sum_{\\text{events}} \\sum_{i} \\Delta K_i,
            \\qquad
            \\Delta K_i = q_i w_i \\sum_{\\mathbf{k}} W^{(p)}(\\mathbf{x}_i - \\mathbf{x}_{\\mathbf{k}})\\, \\Psi_{\\mathbf{k}},

        where :math:`\\Psi_{\\mathbf{k}}` is the discrete line integral of the
        normal electric field :math:`E_n^m` from the PEC wall node to node
        :math:`\\mathbf{k}` at birth step :math:`m`. By this sign convention,
        :math:`K_{\\text{wall\\_kick}} > 0` when the birth kick increases the
        injected particles' kinetic energy relative to ``"K_injected"``, and
        :math:`K_{\\text{wall\\_kick}} < 0` when the birth kick reduces it
        (paying the electrostatic work against the wall-normal field at birth).
        Under MPI (``use_mpi=True`` and ``solver.size > 1``), this method is
        collective across ``solver.comm`` (``MPI.SUM`` reduction).

        Returns
        -------
        float
            Cumulative PEC-wall birth-kick kinetic energy change [J].

        Raises
        ------
        ValueError
            If any birth kick since the last periodic check produced
            :math:`\\gamma' < 1` (deferred device flag, see
            :meth:`_check_wall_kick_error`).
        """
        self._check_wall_kick_error(periodic=False)
        k_loc = float(self._K_wall_kick)
        if self.use_mpi and int(getattr(self.solver, "size", 1)) > 1:
            from mpi4py import MPI

            k_loc = float(self.solver.comm.allreduce(k_loc, op=MPI.SUM))
        return k_loc

    def energy_balance(self) -> dict[str, float]:
        """Return discrete electromagnetic and particle energy-balance terms [J].

        Notes
        -----
        - Under MPI (``use_mpi=True`` and ``solver.size > 1``),
          :meth:`energy_balance` is a collective operation across ``solver.comm``
          and must be called by all ranks simultaneously.
        - ``"W_field"`` uses the half-step-staggered discrete FIT bilinear form
          :meth:`_staggered_field_energy` rather than the un-staggered
          :meth:`field_energy` (:math:`\\frac{1}{2}(\\mathbf{e}^n \\cdot \\mathbf{M}_\\varepsilon \\mathbf{e}^n
          + \\mathbf{h}^{n-1/2} \\cdot \\mathbf{M}_\\mu \\mathbf{h}^{n-1/2})`), and
          does not include convolutional CPML auxiliary :math:`\\boldsymbol{\\psi}`
          energy terms (which act as an artificial absorber outside the lossless
          FIT curl identity).
        - ``"W_wall_closure"`` (:meth:`wall_closure_work`) uses the same sign
          convention as ``"field_work"``: positive when the electromagnetic field
          does positive work on the PEC wall-closure current (delivering energy
          from the field to the wall/emitter), so it is subtracted alongside
          ``"W_field"`` and ``"K_particles"`` in ``"residual"``.
        - ``"K_wall_kick"`` (:meth:`wall_kick_energy`) is the cumulative kinetic
          energy change :math:`\\sum \\Delta K_i` applied to newly injected
          macroparticles at PEC-wall emitters for shape order :math:`p \\ge 2`
          (positive when the birth kick adds kinetic energy to the particles
          beyond the pre-kick ``"K_injected"``, negative when the particle pays
          work against the wall-normal electric field at birth), so it is added
          to ``"K_injected"`` in ``"residual"``.

        Returns
        -------
        dict of str to float
            Dictionary containing:

            - ``"W_field"`` : half-step-staggered discrete FIT field energy
              :math:`\\frac{1}{2}(\\mathbf{e}^n \\cdot \\mathbf{M}_\\varepsilon \\mathbf{e}^n
              + \\mathbf{h}^{n-1/2} \\cdot \\mathbf{M}_\\mu \\mathbf{h}^{n+1/2})` [J]
              (excluding CPML :math:`\\boldsymbol{\\psi}` auxiliary terms; compare
              :meth:`field_energy` for the un-staggered quadratic form).
            - ``"K_particles"`` : synchronised relativistic kinetic energy
              :math:`K_{\\text{particles}}^n` [J] at :math:`t_n`.
            - ``"K_injected"`` : cumulative pre-kick kinetic energy [J] injected
              by all attached :class:`BeamInjector` instances.
            - ``"K_absorbed"`` : cumulative kinetic energy [J] absorbed by all
              boundaries and :class:`Collector` instances.
            - ``"field_work"`` : cumulative discrete field-to-particle work
              :math:`\\sum \\Delta t \\sum_{\\text{edges}} V_E J^{m+1/2} (E^m + E^{m+1})/2` [J].
            - ``"W_wall_closure"`` : cumulative work done by the electromagnetic
              field on the PEC wall-closure current
              :math:`\\sum \\Delta t \\sum_{\\text{edges}} V_E J_{\\text{cl}}^{m+1/2} (E^m + E^{m+1})/2`
              [J] (positive when energy is delivered from the field to the
              wall/emitter).
            - ``"K_wall_kick"`` : cumulative kinetic-energy change
              :math:`\\sum \\Delta K_i` [J] applied at PEC-wall birth kicks for
              shape order :math:`p \\ge 2`.
            - ``"residual"`` : total energy closure residual
              :math:`(K_{\\text{injected}} + K_{\\text{particles}}^0 + W_{\\text{field}}^0)
              - K_{\\text{absorbed}} - K_{\\text{particles}} - W_{\\text{field}}
              - W_{\\text{wall\\_closure}} + K_{\\text{wall\\_kick}}` [J].

        Raises
        ------
        RuntimeError, ValueError
            If the :class:`ParticleSolver` was initialized with
            ``track_energy=False`` (the default).
        """
        if not self.track_energy:
            raise _EnergyTrackingDisabledError(
                "ParticleSolver.energy_balance() requires track_energy=True."
            )
        w_field = self._staggered_field_energy()
        if self._has_step0_field_snap:
            w_field_0 = self._staggered_field_energy(
                self._E_init_fw, self._H_prev_init_fw, self._H_curr_init_fw
            )
        else:
            w_field_0 = w_field
        k_particles = self.kinetic_energy(synchronised=True)
        k_injected = self.injected_energy()
        k_absorbed = self.absorbed_energy()
        fw = self.field_work()
        w_wall_cl = self.wall_closure_work()
        k_wall_kick = self.wall_kick_energy()
        e_init = self._K_particles_0 + w_field_0
        residual = (
            (k_injected + e_init)
            - k_absorbed
            - k_particles
            - w_field
            - w_wall_cl
            + k_wall_kick
        )
        return {
            "W_field": w_field,
            "K_particles": k_particles,
            "K_injected": k_injected,
            "K_absorbed": k_absorbed,
            "field_work": fw,
            "W_wall_closure": w_wall_cl,
            "K_wall_kick": k_wall_kick,
            "residual": residual,
        }

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
            ke_tot = float(sum(sp.kinetic_energy() for sp in self.species))
            if self.use_mpi and int(getattr(self.solver, "size", 1)) > 1:
                from mpi4py import MPI

                ke_tot = float(self.solver.comm.allreduce(ke_tot, op=MPI.SUM))
            return ke_tot

        s = self.solver
        Nx, Ny, Nz = s.Nx, s.Ny, s.Nz
        N = Nx * Ny * Nz
        H_curr = s.H.toarray()
        if hasattr(s, "tDsiDmuiDaC") and self.n_particle_steps > 0:
            H_next = H_curr - s.dt * (s.tDsiDmuiDaC * s.E.toarray())
            if self.use_mpi and int(getattr(s, "size", 1)) > 1:
                from ..field import Field

                H_tmp = Field(Nx, Ny, Nz, use_gpu=s.use_gpu, dtype=s.dtype)
                H_tmp.fromarray(H_next)
                s._mpi_communicate(H_tmp)
                H_next = H_tmp.toarray()
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
        if self.use_mpi and int(getattr(s, "size", 1)) > 1:
            from mpi4py import MPI

            ke_tot = float(s.comm.allreduce(ke_tot, op=MPI.SUM))
        return ke_tot

    def total_charge(self) -> float:
        """Total active macroparticle charge [C] on the grid across all species."""
        q_tot = float(sum(sp.total_charge() for sp in self.species))
        if self.use_mpi and int(getattr(self.solver, "size", 1)) > 1:
            from mpi4py import MPI

            q_tot = float(self.solver.comm.allreduce(q_tot, op=MPI.SUM))
        return q_tot

    def absorbed_charge(self, name: str | None = None) -> float:
        """Total charge [C] absorbed by all boundaries/collectors (or by ``name``)."""
        if name is None:
            q_abs = self.boundaries.total_absorbed_charge()
        else:
            q_abs = float(self.boundaries.absorbed_charge[name])
        if self.use_mpi and int(getattr(self.solver, "size", 1)) > 1:
            from mpi4py import MPI

            q_abs = float(self.solver.comm.allreduce(q_abs, op=MPI.SUM))
        return q_abs

    def absorbed_energy(self, name: str | None = None) -> float:
        """Total kinetic energy [J] absorbed by all boundaries/collectors (or by ``name``)."""
        if name is None:
            e_abs = self.boundaries.total_absorbed_energy()
        else:
            e_abs = float(self.boundaries.absorbed_energy[name])
        if self.use_mpi and int(getattr(self.solver, "size", 1)) > 1:
            from mpi4py import MPI

            e_abs = float(self.solver.comm.allreduce(e_abs, op=MPI.SUM))
        return e_abs

    def injected_charge(self) -> float:
        """Total charge [C] injected by all attached :class:`BeamInjector` instances."""
        q_inj = float(sum(inj.injected_charge for inj in self.injectors))
        if self.use_mpi and int(getattr(self.solver, "size", 1)) > 1:
            from mpi4py import MPI

            q_inj = float(self.solver.comm.allreduce(q_inj, op=MPI.SUM))
        return q_inj

    def injected_energy(self) -> float:
        """Total kinetic energy [J] injected by all attached :class:`BeamInjector` instances."""
        e_inj = float(sum(inj.injected_energy for inj in self.injectors))
        if self.use_mpi and int(getattr(self.solver, "size", 1)) > 1:
            from mpi4py import MPI

            e_inj = float(self.solver.comm.allreduce(e_inj, op=MPI.SUM))
        return e_inj

    def field_energy(self) -> float:
        """Return un-staggered electromagnetic field energy [J] from ``solver.field_energy()``.

        Evaluates :math:`\\frac{1}{2}(\\mathbf{e}^n \\cdot \\mathbf{M}_\\varepsilon \\mathbf{e}^n
        + \\mathbf{h}^{n-1/2} \\cdot \\mathbf{M}_\\mu \\mathbf{h}^{n-1/2})` at the raw
        leapfrog storage levels. For the half-step-staggered discrete FIT field
        energy :math:`\\frac{1}{2}(\\mathbf{e}^n \\cdot \\mathbf{M}_\\varepsilon \\mathbf{e}^n
        + \\mathbf{h}^{n-1/2} \\cdot \\mathbf{M}_\\mu \\mathbf{h}^{n+1/2})` used in the
        exact discrete Poynting theorem, see :meth:`_staggered_field_energy` and
        the ``"W_field"`` entry of :meth:`energy_balance`.
        """
        return float(self.solver.field_energy())

    def gauss_residual(self, include_wall_planes: bool = False) -> float:
        """Return relative discrete Gauss's law residual ``max|S_tilde d_tilde - q| / max|q|``.

        Parameters
        ----------
        include_wall_planes : bool, optional
            If False (default), skip a B-spline stencil margin ``(p + 1) // 2 + 1``
            next to CPML inner interfaces; non-CPML faces (such as PEC domain walls)
            already evaluate all interior vacuum node planes ``1..N-1`` by default.
            If True, shrink the CPML margin to ``1`` node.
        """
        return gauss_residual(
            self.solver,
            self.species,
            order=self.shape,
            include_wall_planes=include_wall_planes,
        )

    def save_checkpoint(self, path):
        """Save coupled electromagnetic field and particle state to ``path``."""
        from .checkpoint import save_checkpoint

        return save_checkpoint(self, path)

    def load_checkpoint(self, path) -> None:
        """Restore coupled electromagnetic field and particle state from ``path``."""
        from .checkpoint import load_checkpoint

        load_checkpoint(self, path)
