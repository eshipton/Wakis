# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""B-spline particle shape functions, field gathering, and Esirkepov current deposition.

All shape evaluations and current depositions operate in logical (index)
coordinates ``(xi, eta, zeta)`` provided by :class:`~wakis.particles.LogicalGridMap`,
ensuring that discrete charge conservation ``S_tilde d_tilde = q`` holds to
machine precision on both uniform and non-uniform (graded) tensor-product
meshes.

References
----------
- C. K. Birdsall and A. B. Langdon, *Plasma Physics via Computer Simulation*
  (Taylor & Francis / IOP Publishing, 1991), Ch. 8, ISBN 978-0-7503-0117-6.
- J. Villasenor and O. Buneman, "Rigorous charge conservation for local
  electromagnetic field solvers", *Comput. Phys. Commun.* **69**, 306 (1992),
  DOI: ``10.1016/0010-4655(92)90169-Y``.
- T. Zh. Esirkepov, "Exact charge conservation scheme for Particle-in-Cell
  simulation with an arbitrary form-factor", *Comput. Phys. Commun.* **135**,
  144 (2001), DOI: ``10.1016/S0010-4655(00)00228-9``.
"""

from __future__ import annotations

import numpy as np


class ParticleCFLError(ValueError):
    """Raised when a particle step violates the single-cell Esirkepov CFL condition."""


def bspline_weights_1d(xi, order: int, staggered: bool = False, xp=np):
    """Evaluate 1D B-spline weights of ``order`` in ``{0, 1, 2, 3}``.

    Parameters
    ----------
    xi : ndarray of float, shape (N_p,)
        1D logical coordinates in ``[0, N_axis]``.
    order : {0, 1, 2, 3}
        B-spline polynomial order (0: NGP, 1: CIC, 2: TSC, 3: PCS).
    staggered : bool, optional
        If False (default), evaluate weights on primal nodes ``i``.
        If True, evaluate weights on staggered cell centres ``i + 0.5``
        (indexed by cell ``i``).
    xp : module, optional
        Array module (``numpy`` or ``cupy``). Default is ``numpy``.

    Returns
    -------
    base_idx : ndarray of int64, shape (N_p,)
        Starting grid index ``i_0`` of the ``(order + 1)``-point stencil.
    weights : tuple of ndarray, length ``order + 1``
        Tuple ``(w_0, ..., w_order)`` of shape ``(N_p,)`` summing to 1.
    """
    u = xi - 0.5 if staggered else xi

    if order == 0:
        i0 = xp.floor(u + 0.5).astype(np.int64)
        return i0, (xp.ones_like(u, dtype=np.float64),)

    if order == 1:
        i0 = xp.floor(u).astype(np.int64)
        d = u - i0.astype(np.float64)
        return i0, (1.0 - d, d)

    if order == 2:
        ic = xp.floor(u + 0.5).astype(np.int64)
        d = u - ic.astype(np.float64)
        w0 = 0.5 * (0.5 - d) * (0.5 - d)
        w1 = 0.75 - d * d
        w2 = 0.5 * (0.5 + d) * (0.5 + d)
        return ic - 1, (w0, w1, w2)

    if order == 3:
        ifloor = xp.floor(u).astype(np.int64)
        d = u - ifloor.astype(np.float64)
        omd = 1.0 - d
        d2 = d * d
        d3 = d2 * d
        w0 = (1.0 / 6.0) * omd * omd * omd
        w1 = (1.0 / 6.0) * (4.0 - 6.0 * d2 + 3.0 * d3)
        w2 = (1.0 / 6.0) * (1.0 + 3.0 * d + 3.0 * d2 - 3.0 * d3)
        w3 = (1.0 / 6.0) * d3
        return ifloor - 1, (w0, w1, w2, w3)

    raise ValueError(f"Unsupported B-spline order={order!r}; expected 0, 1, 2, or 3.")


def _wrap_or_mask_1d(idx, n_cells: int, periodic: bool, xp=np):
    """Return ``(wrapped_idx, valid_mask)`` for 1D index array ``idx``."""
    if periodic:
        return idx % n_cells, None
    valid = (idx >= 0) & (idx < n_cells)
    return xp.clip(idx, 0, n_cells - 1), valid


def _gather_component_3d(
    field_flat,
    Nx: int,
    Ny: int,
    Nz: int,
    ix0,
    wx,
    iy0,
    wy,
    iz0,
    wz,
    periodic: tuple[bool, bool, bool],
    xp=np,
):
    """Interpolate a single 3D field component from flat Fortran array ``field_flat``."""
    n_p = ix0.size
    out = xp.zeros(n_p, dtype=np.float64)
    if n_p == 0:
        return out

    stride_y = Nx
    stride_z = Nx * Ny

    # Precompute wrapped indices and validity masks along each axis
    ix_list = []
    vx_list = []
    for ox in range(len(wx)):
        ix_m, vx_m = _wrap_or_mask_1d(ix0 + ox, Nx, periodic[0], xp=xp)
        ix_list.append(ix_m)
        vx_list.append(vx_m)

    iy_list = []
    vy_list = []
    for oy in range(len(wy)):
        iy_m, vy_m = _wrap_or_mask_1d(iy0 + oy, Ny, periodic[1], xp=xp)
        iy_list.append(iy_m * stride_y)
        vy_list.append(vy_m)

    iz_list = []
    vz_list = []
    for oz in range(len(wz)):
        iz_m, vz_m = _wrap_or_mask_1d(iz0 + oz, Nz, periodic[2], xp=xp)
        iz_list.append(iz_m * stride_z)
        vz_list.append(vz_m)

    for oz, wz_m in enumerate(wz):
        iz_off = iz_list[oz]
        vz_m = vz_list[oz]
        for oy, wy_m in enumerate(wy):
            wyz = wy_m * wz_m
            iyz_off = iy_off = iy_list[oy] + iz_off
            vy_m = vy_list[oy]
            if vz_m is not None and vy_m is not None:
                vyz_m = vy_m & vz_m
            elif vz_m is not None:
                vyz_m = vz_m
            else:
                vyz_m = vy_m

            for ox, wx_m in enumerate(wx):
                w = wx_m * wyz
                idx = ix_list[ox] + iyz_off
                vx_m = vx_list[ox]
                if vyz_m is not None and vx_m is not None:
                    v = vx_m & vyz_m
                elif vyz_m is not None:
                    v = vyz_m
                else:
                    v = vx_m
                if v is not None:
                    w = xp.where(v, w, 0.0)
                out += w * field_flat[idx]

    return out


def gather_fields(
    xi,
    eta,
    zeta,
    E_field,
    B_field,
    Nx: int,
    Ny: int,
    Nz: int,
    order: int = 1,
    gather: str = "galerkin",
    periodic: tuple[bool, bool, bool] = (True, True, True),
    xp=np,
):
    """Gather ``(Ex, Ey, Ez)`` and ``(Bx, By, Bz)`` at logical positions ``(xi, eta, zeta)``.

    Parameters
    ----------
    xi, eta, zeta : ndarray of float, shape (N_p,)
        Particle logical coordinates.
    E_field, B_field : wakis.Field or tuple of 3 ndarray
        Electric field [V/m] on primal edges and magnetic flux density [T]
        on primal faces (dual edges). Each may be a :class:`wakis.Field` or a
        tuple ``(Fx_flat, Fy_flat, Fz_flat)`` of 1D Fortran-ordered arrays of
        length ``N = Nx * Ny * Nz``.
    Nx, Ny, Nz : int
        Number of grid cells along x, y, z.
    order : {1, 2, 3}, optional
        Particle shape order ``p``. Default is 1 (CIC).
    gather : {"galerkin", "momentum"}, optional
        Interpolation scheme:
        - ``"galerkin"`` (default, energy-conserving): uses order ``p - 1``
          along staggered axes and order ``p`` along nodal axes.
        - ``"momentum"`` (momentum-conserving): uses order ``p`` along all axes.
    periodic : tuple of 3 bool, optional
        Periodicity flags for ``(x, y, z)``. Default is ``(True, True, True)``.
    xp : module, optional
        Array module. Default is ``numpy``.

    Returns
    -------
    Ex, Ey, Ez, Bx, By, Bz : tuple of 6 ndarray, each shape (N_p,)
        Interpolated electric [V/m] and magnetic [T] field components.
    """
    mode = str(gather).strip().lower()
    if mode in ("galerkin", "energy", "energy-conserving"):
        p_stag = order - 1
    elif mode in ("momentum", "momentum-conserving"):
        p_stag = order
    else:
        raise ValueError(
            f"Unknown gather={gather!r}; expected 'galerkin' or 'momentum'."
        )

    N = Nx * Ny * Nz
    if hasattr(E_field, "toarray"):
        E_arr = E_field.toarray()
        Ex_f, Ey_f, Ez_f = E_arr[:N], E_arr[N : 2 * N], E_arr[2 * N : 3 * N]
    else:
        Ex_f, Ey_f, Ez_f = E_field

    if hasattr(B_field, "toarray"):
        B_arr = B_field.toarray()
        Bx_f, By_f, Bz_f = B_arr[:N], B_arr[N : 2 * N], B_arr[2 * N : 3 * N]
    else:
        Bx_f, By_f, Bz_f = B_field

    if xp is not np:
        from .gpu_kernels import gather_fields_gpu

        return gather_fields_gpu(
            xi,
            eta,
            zeta,
            Ex_f,
            Ey_f,
            Ez_f,
            Bx_f,
            By_f,
            Bz_f,
            Nx,
            Ny,
            Nz,
            order=order,
            p_stag=p_stag,
            periodic=periodic,
        )

    n_p = xi.size
    _CPU_CHUNK = 65536
    if n_p > _CPU_CHUNK:
        Ex = np.empty(n_p, dtype=np.float64)
        Ey = np.empty(n_p, dtype=np.float64)
        Ez = np.empty(n_p, dtype=np.float64)
        Bx = np.empty(n_p, dtype=np.float64)
        By = np.empty(n_p, dtype=np.float64)
        Bz = np.empty(n_p, dtype=np.float64)
        E_tup = (Ex_f, Ey_f, Ez_f)
        B_tup = (Bx_f, By_f, Bz_f)
        for s0 in range(0, n_p, _CPU_CHUNK):
            s1 = min(s0 + _CPU_CHUNK, n_p)
            ex_c, ey_c, ez_c, bx_c, by_c, bz_c = gather_fields(
                xi[s0:s1],
                eta[s0:s1],
                zeta[s0:s1],
                E_tup,
                B_tup,
                Nx,
                Ny,
                Nz,
                order=order,
                gather=gather,
                periodic=periodic,
                xp=xp,
            )
            Ex[s0:s1] = ex_c
            Ey[s0:s1] = ey_c
            Ez[s0:s1] = ez_c
            Bx[s0:s1] = bx_c
            By[s0:s1] = by_c
            Bz[s0:s1] = bz_c
        return Ex, Ey, Ez, Bx, By, Bz

    # Precompute nodal (order p) and staggered (order p_stag) 1D weights
    ix_nod, wx_nod = bspline_weights_1d(xi, order, staggered=False, xp=xp)
    iy_nod, wy_nod = bspline_weights_1d(eta, order, staggered=False, xp=xp)
    iz_nod, wz_nod = bspline_weights_1d(zeta, order, staggered=False, xp=xp)

    ix_stg, wx_stg = bspline_weights_1d(xi, p_stag, staggered=True, xp=xp)
    iy_stg, wy_stg = bspline_weights_1d(eta, p_stag, staggered=True, xp=xp)
    iz_stg, wz_stg = bspline_weights_1d(zeta, p_stag, staggered=True, xp=xp)

    # Primal edges: Ex at (i+1/2, j, k), Ey at (i, j+1/2, k), Ez at (i, j, k+1/2)
    Ex = _gather_component_3d(
        Ex_f, Nx, Ny, Nz, ix_stg, wx_stg, iy_nod, wy_nod, iz_nod, wz_nod, periodic, xp
    )
    Ey = _gather_component_3d(
        Ey_f, Nx, Ny, Nz, ix_nod, wx_nod, iy_stg, wy_stg, iz_nod, wz_nod, periodic, xp
    )
    Ez = _gather_component_3d(
        Ez_f, Nx, Ny, Nz, ix_nod, wx_nod, iy_nod, wy_nod, iz_stg, wz_stg, periodic, xp
    )

    # Primal faces (dual edges):
    # Bx at (i, j+1/2, k+1/2), By at (i+1/2, j, k+1/2), Bz at (i+1/2, j+1/2, k)
    Bx = _gather_component_3d(
        Bx_f, Nx, Ny, Nz, ix_nod, wx_nod, iy_stg, wy_stg, iz_stg, wz_stg, periodic, xp
    )
    By = _gather_component_3d(
        By_f, Nx, Ny, Nz, ix_stg, wx_stg, iy_nod, wy_nod, iz_stg, wz_stg, periodic, xp
    )
    Bz = _gather_component_3d(
        Bz_f, Nx, Ny, Nz, ix_stg, wx_stg, iy_stg, wy_stg, iz_nod, wz_nod, periodic, xp
    )

    return Ex, Ey, Ez, Bx, By, Bz


def _accumulate_flat(target_flat, idx, weights, n_total: int, xp=np):
    """Vectorised scatter-add of ``weights`` at ``idx`` into ``target_flat``."""
    if idx.size == 0:
        return
    if xp is np:
        target_flat += np.bincount(idx, weights=weights, minlength=n_total)
    else:
        xp.add.at(target_flat, idx, weights)


def _accumulate_flat_chunks(target_flat, idx_chunks, w_chunks, n_total: int, xp=np):
    """Concatenate stencil index/weight chunks and call ``_accumulate_flat`` once."""
    if not idx_chunks:
        return
    _accumulate_flat(
        target_flat,
        xp.concatenate(idx_chunks),
        xp.concatenate(w_chunks),
        n_total,
        xp=xp,
    )


def deposit_charge(
    xi,
    eta,
    zeta,
    qw,
    Nx: int,
    Ny: int,
    Nz: int,
    order: int = 1,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    q_nodes=None,
    xp=np,
):
    """Deposit macroparticle charges ``qw = q * w`` [C] onto primal nodes (dual cells).

    Parameters
    ----------
    xi, eta, zeta : ndarray of float, shape (N_p,)
        Particle logical coordinates.
    qw : ndarray of float, shape (N_p,)
        Macroparticle charges ``q * w`` [C].
    Nx, Ny, Nz : int
        Grid cell counts.
    order : {1, 2, 3}, optional
        B-spline shape order ``p``. Default is 1 (CIC).
    periodic : tuple of 3 bool, optional
        Periodicity flags for ``(x, y, z)``.
    q_nodes : ndarray of float, shape (Nx, Ny, Nz), optional
        Pre-allocated Fortran-ordered output array to accumulate into.
    xp : module, optional
        Array module. Default is ``numpy``.

    Returns
    -------
    q_nodes : ndarray of float, shape (Nx, Ny, Nz)
        Deposited node charges [C] in Fortran order.
    """
    if xp is not np:
        from .gpu_kernels import deposit_charge_gpu

        return deposit_charge_gpu(
            xi,
            eta,
            zeta,
            qw,
            1.0,
            Nx,
            Ny,
            Nz,
            order=order,
            periodic=periodic,
            q_nodes=q_nodes,
        )

    n_total = Nx * Ny * Nz
    if q_nodes is None:
        q_flat = xp.zeros(n_total, dtype=np.float64)
    else:
        q_flat = q_nodes.reshape(-1, order="F").copy()

    if xi.size == 0:
        out = q_flat.reshape((Nx, Ny, Nz), order="F")
        if q_nodes is not None:
            q_nodes[...] = out
            return q_nodes
        return out

    _CPU_CHUNK = 65536
    if xp is np and xi.size > _CPU_CHUNK:
        out = q_flat.reshape((Nx, Ny, Nz), order="F")
        for s0 in range(0, xi.size, _CPU_CHUNK):
            s1 = min(s0 + _CPU_CHUNK, xi.size)
            deposit_charge(
                xi[s0:s1],
                eta[s0:s1],
                zeta[s0:s1],
                qw[s0:s1] if np.ndim(qw) > 0 else qw,
                Nx,
                Ny,
                Nz,
                order=order,
                periodic=periodic,
                q_nodes=out,
                xp=xp,
            )
        if q_nodes is not None:
            q_nodes[...] = out
            return q_nodes
        return out

    ix0, wx = bspline_weights_1d(xi, order, staggered=False, xp=xp)
    iy0, wy = bspline_weights_1d(eta, order, staggered=False, xp=xp)
    iz0, wz = bspline_weights_1d(zeta, order, staggered=False, xp=xp)

    stride_y = Nx
    stride_z = Nx * Ny

    ix_list, vx_list = [], []
    for ox in range(order + 1):
        im, vm = _wrap_or_mask_1d(ix0 + ox, Nx, periodic[0], xp=xp)
        ix_list.append(im)
        vx_list.append(vm)

    iy_list, vy_list = [], []
    for oy in range(order + 1):
        im, vm = _wrap_or_mask_1d(iy0 + oy, Ny, periodic[1], xp=xp)
        iy_list.append(im * stride_y)
        vy_list.append(vm)

    iz_list, vz_list = [], []
    for oz in range(order + 1):
        im, vm = _wrap_or_mask_1d(iz0 + oz, Nz, periodic[2], xp=xp)
        iz_list.append(im * stride_z)
        vz_list.append(vm)

    n_p = xi.size
    n_sten = (order + 1) ** 3
    idx_buf = xp.empty((n_sten, n_p), dtype=np.int64)
    w_buf = xp.empty((n_sten, n_p), dtype=np.float64)
    k = 0
    for oz, wz_m in enumerate(wz):
        iz_off = iz_list[oz]
        vz_m = vz_list[oz]
        for oy, wy_m in enumerate(wy):
            q_yz = qw * (wy_m * wz_m)
            iyz_off = iy_list[oy] + iz_off
            vy_m = vy_list[oy]
            if vz_m is not None and vy_m is not None:
                vyz_m = vy_m & vz_m
            elif vz_m is not None:
                vyz_m = vz_m
            else:
                vyz_m = vy_m

            for ox, wx_m in enumerate(wx):
                vx_m = vx_list[ox]
                if vyz_m is not None and vx_m is not None:
                    v = vx_m & vyz_m
                elif vyz_m is not None:
                    v = vyz_m
                else:
                    v = vx_m
                if xp is np:
                    np.add(ix_list[ox], iyz_off, out=idx_buf[k])
                    np.multiply(wx_m, q_yz, out=w_buf[k])
                    if v is not None:
                        w_buf[k] *= v
                else:
                    idx_buf[k] = ix_list[ox] + iyz_off
                    w = wx_m * q_yz
                    if v is not None:
                        w = xp.where(v, w, 0.0)
                    w_buf[k] = w
                k += 1

    _accumulate_flat(q_flat, idx_buf.ravel(), w_buf.ravel(), n_total, xp=xp)

    out = q_flat.reshape((Nx, Ny, Nz), order="F")
    if q_nodes is not None:
        q_nodes[...] = out
        return q_nodes
    return out


def _esirkepov_1d_window(xi0, xi1, order: int, xp=np):
    """Build 1D Esirkepov shape arrays ``(S0, S1, dS, F)`` on the minimal ``(p + 2)``-node window.

    For each particle, the window starts at ``i_base = min(b0, b1)`` where
    ``b0`` and ``b1`` are the starting node indices of the B-spline of ``order``
    at ``xi0`` and ``xi1``. Because ``|b1 - b0| <= 1``, the union of the
    supports of ``S0`` and ``S1`` spans at most ``order + 2`` nodes
    ``i_base .. i_base + order + 1`` and ``order + 1`` primal edges.

    Returns
    -------
    i_base : ndarray of int64, shape (N_p,)
        Window starting node index ``min(b0, b1)``.
    S0, S1, dS : list of ndarray, length ``order + 2``
        Old weights, new weights, and ``dS = S1 - S0`` at ``i_base + m``.
    F : list of ndarray, length ``order + 1``
        1D edge charge fluxes ``F[m] = -sum_{k=0}^m dS[k]`` on primal edges
        ``i_base + m`` so that ``F[m] - F[m - 1] = -dS[m]`` holds identically.
    """
    n_p = xi0.size
    m_win = order + 2
    b0, w0 = bspline_weights_1d(xi0, order, staggered=False, xp=xp)
    b1, w1 = bspline_weights_1d(xi1, order, staggered=False, xp=xp)

    db = b1 - b0
    if n_p > 0 and bool(xp.any((db < -1) | (db > 1))):
        raise ParticleCFLError(
            "Particle crossed more than 1 cell in a single step "
            "(|db| > 1 violates the Esirkepov single-cell CFL assumption)."
        )

    i_base = xp.minimum(b0, b1)
    zeros = xp.zeros(n_p, dtype=np.float64)

    if n_p == 0 or not bool(xp.any(db)):
        S0 = list(w0) + [zeros]
        S1 = list(w1) + [zeros]
    else:
        c_m = db == -1
        c_p = db == 1
        w0_0 = list(w0) + [zeros]
        w0_1 = [zeros] + list(w0)
        S0 = [xp.where(c_m, w0_1[m], w0_0[m]) for m in range(m_win)]

        w1_0 = list(w1) + [zeros]
        w1_1 = [zeros] + list(w1)
        S1 = [xp.where(c_p, w1_1[m], w1_0[m]) for m in range(m_win)]

    dS = [S1[m] - S0[m] for m in range(m_win)]

    # Cumulative edge fluxes F[m] = -sum_{k=0}^m dS[k] for m = 0 .. order.
    # For the last edge m = order, use +dS[order + 1] so F[order + 1] = 0
    # holds without floating-point summation drift.
    F = []
    acc = -dS[0]
    F.append(acc)
    for m in range(1, order):
        acc = acc - dS[m]
        F.append(acc)
    if order >= 1:
        F.append(dS[order + 1])

    return i_base, S0, S1, dS, F


def deposit_current_esirkepov(
    xi0,
    eta0,
    zeta0,
    xi1,
    eta1,
    zeta1,
    qw,
    Nx: int,
    Ny: int,
    Nz: int,
    order: int = 1,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    Wx_out=None,
    Wy_out=None,
    Wz_out=None,
    xp=np,
):
    """Deposit dual-face charge fluxes ``(Wx, Wy, Wz)`` [C] via the 3D Esirkepov scheme.

    The deposited dual-face charges ``W_e`` satisfy the exact discrete
    continuity identity on any tensor-product grid:
    ``(S_tilde W)_{i,j,k} = -(q^{n+1}_{i,j,k} - q^n_{i,j,k})``.
    To obtain the FIT primal-edge current density ``J_e`` [A/m^2], multiply by
    ``itA_e / dt`` (where ``itA_e = 1 / A_tilde_e`` is the inverse dual face area).

    Parameters
    ----------
    xi0, eta0, zeta0 : ndarray of float, shape (N_p,)
        Logical coordinates at ``t_n``.
    xi1, eta1, zeta1 : ndarray of float, shape (N_p,)
        Logical coordinates at ``t_{n+1}`` (unwrapped relative to ``t_n``, i.e.
        along the continuous straight path ``xi0 -> xi1``).
    qw : ndarray of float, shape (N_p,)
        Macroparticle charges ``q * w`` [C].
    Nx, Ny, Nz : int
        Grid cell counts.
    order : {1, 2, 3}, optional
        B-spline shape order ``p``. Default is 1 (CIC).
    periodic : tuple of 3 bool, optional
        Periodicity flags for ``(x, y, z)``.
    Wx_out, Wy_out, Wz_out : ndarray of float, shape (Nx * Ny * Nz,), optional
        Optional flat output buffers to accumulate into.
    xp : module, optional
        Array module. Default is ``numpy``.

    Returns
    -------
    Wx, Wy, Wz : tuple of 3 ndarray, each shape (Nx * Ny * Nz,)
        Flat Fortran-ordered dual-face charge fluxes [C] on x, y, z primal edges.
    """
    if xp is not np:
        from .gpu_kernels import deposit_current_esirkepov_gpu

        return deposit_current_esirkepov_gpu(
            xi0,
            eta0,
            zeta0,
            xi1,
            eta1,
            zeta1,
            qw,
            1.0,
            Nx,
            Ny,
            Nz,
            order=order,
            periodic=periodic,
            Wx_out=Wx_out,
            Wy_out=Wy_out,
            Wz_out=Wz_out,
        )

    n_total = Nx * Ny * Nz
    Wx = xp.zeros(n_total, dtype=np.float64) if Wx_out is None else Wx_out
    Wy = xp.zeros(n_total, dtype=np.float64) if Wy_out is None else Wy_out
    Wz = xp.zeros(n_total, dtype=np.float64) if Wz_out is None else Wz_out

    n_p = xi0.size
    if n_p == 0:
        return Wx, Wy, Wz

    _CPU_CHUNK = 65536
    if n_p > _CPU_CHUNK:
        for s0 in range(0, n_p, _CPU_CHUNK):
            s1 = min(s0 + _CPU_CHUNK, n_p)
            deposit_current_esirkepov(
                xi0[s0:s1],
                eta0[s0:s1],
                zeta0[s0:s1],
                xi1[s0:s1],
                eta1[s0:s1],
                zeta1[s0:s1],
                qw[s0:s1],
                Nx,
                Ny,
                Nz,
                order=order,
                periodic=periodic,
                Wx_out=Wx,
                Wy_out=Wy,
                Wz_out=Wz,
                xp=xp,
            )
        return Wx, Wy, Wz

    m_win = order + 2
    m_flux = order + 1

    ix_base, Sx0, Sx1, dSx, Fx = _esirkepov_1d_window(xi0, xi1, order, xp=xp)
    iy_base, Sy0, Sy1, dSy, Fy = _esirkepov_1d_window(eta0, eta1, order, xp=xp)
    iz_base, Sz0, Sz1, dSz, Fz = _esirkepov_1d_window(zeta0, zeta1, order, xp=xp)

    stride_y = Nx
    stride_z = Nx * Ny

    ix_list, vx_list = [], []
    for mx in range(m_win):
        im, vm = _wrap_or_mask_1d(ix_base + mx, Nx, periodic[0], xp=xp)
        ix_list.append(im)
        vx_list.append(vm)

    iy_list, vy_list = [], []
    for my in range(m_win):
        im, vm = _wrap_or_mask_1d(iy_base + my, Ny, periodic[1], xp=xp)
        iy_list.append(im * stride_y)
        vy_list.append(vm)

    iz_list, vz_list = [], []
    for mz in range(m_win):
        im, vm = _wrap_or_mask_1d(iz_base + mz, Nz, periodic[2], xp=xp)
        iz_list.append(im * stride_z)
        vz_list.append(vm)

    one_third = 1.0 / 3.0
    n_sten = m_flux * m_win * m_win
    idx_buf = xp.empty((n_sten, n_p), dtype=np.int64)
    w_buf = xp.empty((n_sten, n_p), dtype=np.float64)

    # --- 1. Deposit Wx (mx in 0..m_flux-1, my in 0..m_win-1, mz in 0..m_win-1) ---
    qFx = [qw * Fx[mx] for mx in range(m_flux)]
    del Fx
    k = 0
    for mz in range(m_win):
        iz_off = iz_list[mz]
        vz_m = vz_list[mz]
        sz0_m, dsz_m = Sz0[mz], dSz[mz]
        for my in range(m_win):
            sy0_m, dsy_m = Sy0[my], dSy[my]
            wyz = (
                sy0_m * sz0_m
                + 0.5 * (sy0_m * dsz_m + dsy_m * sz0_m)
                + one_third * dsy_m * dsz_m
            )
            iyz_off = iy_list[my] + iz_off
            vy_m = vy_list[my]
            vyz = (
                (vy_m & vz_m)
                if (vy_m is not None and vz_m is not None)
                else (vz_m if vy_m is None else vy_m)
            )
            for mx in range(m_flux):
                vx_m = vx_list[mx]
                v = (
                    (vx_m & vyz)
                    if (vx_m is not None and vyz is not None)
                    else (vyz if vx_m is None else vx_m)
                )
                if xp is np:
                    np.add(ix_list[mx], iyz_off, out=idx_buf[k])
                    np.multiply(qFx[mx], wyz, out=w_buf[k])
                    if v is not None:
                        w_buf[k] *= v
                else:
                    idx_buf[k] = ix_list[mx] + iyz_off
                    w = qFx[mx] * wyz
                    if v is not None:
                        w = xp.where(v, w, 0.0)
                    w_buf[k] = w
                k += 1
    del qFx
    _accumulate_flat(Wx, idx_buf.ravel(), w_buf.ravel(), n_total, xp=xp)

    # --- 2. Deposit Wy (mx in 0..m_win-1, my in 0..m_flux-1, mz in 0..m_win-1) ---
    qFy = [qw * Fy[my] for my in range(m_flux)]
    del Fy
    k = 0
    for mz in range(m_win):
        iz_off = iz_list[mz]
        vz_m = vz_list[mz]
        sz0_m, dsz_m = Sz0[mz], dSz[mz]
        for mx in range(m_win):
            sx0_m, dsx_m = Sx0[mx], dSx[mx]
            wxz = (
                sx0_m * sz0_m
                + 0.5 * (sx0_m * dsz_m + dsx_m * sz0_m)
                + one_third * dsx_m * dsz_m
            )
            ixz_off = ix_list[mx] + iz_off
            vx_m = vx_list[mx]
            vxz = (
                (vx_m & vz_m)
                if (vx_m is not None and vz_m is not None)
                else (vz_m if vx_m is None else vx_m)
            )
            for my in range(m_flux):
                vy_m = vy_list[my]
                v = (
                    (vy_m & vxz)
                    if (vy_m is not None and vxz is not None)
                    else (vxz if vy_m is None else vy_m)
                )
                if xp is np:
                    np.add(ixz_off, iy_list[my], out=idx_buf[k])
                    np.multiply(qFy[my], wxz, out=w_buf[k])
                    if v is not None:
                        w_buf[k] *= v
                else:
                    idx_buf[k] = ixz_off + iy_list[my]
                    w = qFy[my] * wxz
                    if v is not None:
                        w = xp.where(v, w, 0.0)
                    w_buf[k] = w
                k += 1
    del qFy, Sz0, dSz
    _accumulate_flat(Wy, idx_buf.ravel(), w_buf.ravel(), n_total, xp=xp)

    # --- 3. Deposit Wz (mx in 0..m_win-1, my in 0..m_win-1, mz in 0..m_flux-1) ---
    qFz = [qw * Fz[mz] for mz in range(m_flux)]
    del Fz
    k = 0
    for my in range(m_win):
        iy_off = iy_list[my]
        vy_m = vy_list[my]
        sy0_m, dsy_m = Sy0[my], dSy[my]
        for mx in range(m_win):
            sx0_m, dsx_m = Sx0[mx], dSx[mx]
            wxy = (
                sx0_m * sy0_m
                + 0.5 * (sx0_m * dsy_m + dsx_m * sy0_m)
                + one_third * dsx_m * dsy_m
            )
            ixy_off = ix_list[mx] + iy_off
            vx_m = vx_list[mx]
            vxy = (
                (vx_m & vy_m)
                if (vx_m is not None and vy_m is not None)
                else (vy_m if vx_m is None else vx_m)
            )
            for mz in range(m_flux):
                vz_m = vz_list[mz]
                v = (
                    (vz_m & vxy)
                    if (vz_m is not None and vxy is not None)
                    else (vxy if vz_m is None else vz_m)
                )
                if xp is np:
                    np.add(ixy_off, iz_list[mz], out=idx_buf[k])
                    np.multiply(qFz[mz], wxy, out=w_buf[k])
                    if v is not None:
                        w_buf[k] *= v
                else:
                    idx_buf[k] = ixy_off + iz_list[mz]
                    w = qFz[mz] * wxy
                    if v is not None:
                        w = xp.where(v, w, 0.0)
                    w_buf[k] = w
                k += 1
    del qFz, Sx0, dSx, Sy0, dSy, ix_list, vx_list, iy_list, vy_list, iz_list, vz_list
    _accumulate_flat(Wz, idx_buf.ravel(), w_buf.ravel(), n_total, xp=xp)

    return Wx, Wy, Wz


def esirkepov_edge_abs_sum_and_counts(
    xi0,
    eta0,
    zeta0,
    xi1,
    eta1,
    zeta1,
    qw,
    Nx: int,
    Ny: int,
    Nz: int,
    order: int = 1,
    periodic: tuple[bool, bool, bool] = (True, True, True),
):
    """Compute exact per-edge ``sum_k |W_{e,k}|`` [C] and non-zero stencil contribution counts ``n(e)`` [-].

    For each primal edge ``e`` in ``(Wx, Wy, Wz)``, accumulates the sum of
    magnitudes ``|w_{e,k}|`` and the integer number ``n(e)`` of non-zero
    stencil contributions deposited by ``deposit_current_esirkepov``. By
    Higham's floating-point summation theorem (Theorem 3.1), each summation
    order of ``n(e)`` terms is within ``gamma_{n(e)-1} * sum_{k in C(e)} |w_{e,k}|``
    of the exact sum, so the difference between any two summation orders (such as
    GPU ``atomicAdd`` vs CPU sequential ``bincount``) on edge ``e`` satisfies:

        |W_e^GPU - W_e^CPU| <= 2 * gamma_{n(e)-1} * sum_{k in C(e)} |w_{e,k}|,

    where ``gamma_m = (m * u) / (1 - m * u)`` (with ``gamma_0 = 0`` for
    ``n(e) <= 1``) and ``u = 2^-53`` is the IEEE-754 ``float64`` unit roundoff
    (``0.5 * eps_mach``).
    """
    n_total = Nx * Ny * Nz
    abs_Wx = np.zeros(n_total, dtype=np.float64)
    abs_Wy = np.zeros(n_total, dtype=np.float64)
    abs_Wz = np.zeros(n_total, dtype=np.float64)
    cnt_Wx = np.zeros(n_total, dtype=np.int64)
    cnt_Wy = np.zeros(n_total, dtype=np.int64)
    cnt_Wz = np.zeros(n_total, dtype=np.int64)

    n_p = xi0.size
    if n_p == 0:
        return (abs_Wx, abs_Wy, abs_Wz), (cnt_Wx, cnt_Wy, cnt_Wz)

    m_win = order + 2
    m_flux = order + 1

    ix_base, Sx0, _, dSx, Fx = _esirkepov_1d_window(xi0, xi1, order, xp=np)
    iy_base, Sy0, _, dSy, Fy = _esirkepov_1d_window(eta0, eta1, order, xp=np)
    iz_base, Sz0, _, dSz, Fz = _esirkepov_1d_window(zeta0, zeta1, order, xp=np)

    stride_y = Nx
    stride_z = Nx * Ny

    ix_list, vx_list = [], []
    for mx in range(m_win):
        im, vm = _wrap_or_mask_1d(ix_base + mx, Nx, periodic[0], xp=np)
        ix_list.append(im)
        vx_list.append(vm)

    iy_list, vy_list = [], []
    for my in range(m_win):
        im, vm = _wrap_or_mask_1d(iy_base + my, Ny, periodic[1], xp=np)
        iy_list.append(im * stride_y)
        vy_list.append(vm)

    iz_list, vz_list = [], []
    for mz in range(m_win):
        im, vm = _wrap_or_mask_1d(iz_base + mz, Nz, periodic[2], xp=np)
        iz_list.append(im * stride_z)
        vz_list.append(vm)

    one_third = 1.0 / 3.0
    n_sten = m_flux * m_win * m_win
    idx_buf = np.empty((n_sten, n_p), dtype=np.int64)
    w_buf = np.empty((n_sten, n_p), dtype=np.float64)

    def _accum_abs_and_cnt(abs_out, cnt_out):
        idx_r = idx_buf.ravel()
        w_r = np.abs(w_buf.ravel())
        nz = w_r != 0.0
        if np.any(nz):
            abs_out += np.bincount(idx_r[nz], weights=w_r[nz], minlength=n_total)
            cnt_out += np.bincount(idx_r[nz], minlength=n_total)

    qFx = [qw * Fx[mx] for mx in range(m_flux)]
    k = 0
    for mz in range(m_win):
        iz_off = iz_list[mz]
        vz_m = vz_list[mz]
        sz0_m, dsz_m = Sz0[mz], dSz[mz]
        for my in range(m_win):
            sy0_m, dsy_m = Sy0[my], dSy[my]
            wyz = (
                sy0_m * sz0_m
                + 0.5 * (sy0_m * dsz_m + dsy_m * sz0_m)
                + one_third * dsy_m * dsz_m
            )
            iyz_off = iy_list[my] + iz_off
            vy_m = vy_list[my]
            vyz = (
                (vy_m & vz_m)
                if (vy_m is not None and vz_m is not None)
                else (vz_m if vy_m is None else vy_m)
            )
            for mx in range(m_flux):
                vx_m = vx_list[mx]
                v = (
                    (vx_m & vyz)
                    if (vx_m is not None and vyz is not None)
                    else (vyz if vx_m is None else vx_m)
                )
                np.add(ix_list[mx], iyz_off, out=idx_buf[k])
                np.multiply(qFx[mx], wyz, out=w_buf[k])
                if v is not None:
                    w_buf[k] *= v
                k += 1
    _accum_abs_and_cnt(abs_Wx, cnt_Wx)

    qFy = [qw * Fy[my] for my in range(m_flux)]
    k = 0
    for mz in range(m_win):
        iz_off = iz_list[mz]
        vz_m = vz_list[mz]
        sz0_m, dsz_m = Sz0[mz], dSz[mz]
        for mx in range(m_win):
            sx0_m, dsx_m = Sx0[mx], dSx[mx]
            wxz = (
                sx0_m * sz0_m
                + 0.5 * (sx0_m * dsz_m + dsx_m * sz0_m)
                + one_third * dsx_m * dsz_m
            )
            ixz_off = ix_list[mx] + iz_off
            vx_m = vx_list[mx]
            vxz = (
                (vx_m & vz_m)
                if (vx_m is not None and vz_m is not None)
                else (vz_m if vx_m is None else vx_m)
            )
            for my in range(m_flux):
                vy_m = vy_list[my]
                v = (
                    (vy_m & vxz)
                    if (vy_m is not None and vxz is not None)
                    else (vxz if vy_m is None else vy_m)
                )
                np.add(ixz_off, iy_list[my], out=idx_buf[k])
                np.multiply(qFy[my], wxz, out=w_buf[k])
                if v is not None:
                    w_buf[k] *= v
                k += 1
    _accum_abs_and_cnt(abs_Wy, cnt_Wy)

    qFz = [qw * Fz[mz] for mz in range(m_flux)]
    k = 0
    for my in range(m_win):
        iy_off = iy_list[my]
        vy_m = vy_list[my]
        sy0_m, dsy_m = Sy0[my], dSy[my]
        for mx in range(m_win):
            sx0_m, dsx_m = Sx0[mx], dSx[mx]
            wxy = (
                sx0_m * sy0_m
                + 0.5 * (sx0_m * dsy_m + dsx_m * sy0_m)
                + one_third * dsx_m * dsy_m
            )
            ixy_off = ix_list[mx] + iy_off
            vx_m = vx_list[mx]
            vxy = (
                (vx_m & vy_m)
                if (vx_m is not None and vy_m is not None)
                else (vy_m if vx_m is None else vx_m)
            )
            for mz in range(m_flux):
                vz_m = vz_list[mz]
                v = (
                    (vz_m & vxy)
                    if (vz_m is not None and vxy is not None)
                    else (vxy if vz_m is None else vz_m)
                )
                np.add(ixy_off, iz_list[mz], out=idx_buf[k])
                np.multiply(qFz[mz], wxy, out=w_buf[k])
                if v is not None:
                    w_buf[k] *= v
                k += 1
    _accum_abs_and_cnt(abs_Wz, cnt_Wz)

    return (abs_Wx, abs_Wy, abs_Wz), (cnt_Wx, cnt_Wy, cnt_Wz)


def deposit_current_naive(
    xi0,
    eta0,
    zeta0,
    xi1,
    eta1,
    zeta1,
    qw,
    Nx: int,
    Ny: int,
    Nz: int,
    order: int = 1,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    xp=np,
):
    """Non-charge-conserving midpoint ``J = q v`` deposition (for V2 mutation tests).

    Evaluates ``W_a = qw * (xi1_a - xi0_a) * S(xi_mid)`` at the midpoint
    ``xi_mid = 0.5 * (xi0 + xi1)`` using staggered edge weights along axis ``a``
    and nodal weights along the two transverse axes. Because ``S(xi_mid) * d_xi``
    does not telescope to ``S(xi1) - S(xi0)``, discrete continuity is violated
    at ``O(d_xi^2)`` per step and the Gauss residual grows Secularly.
    """
    n_total = Nx * Ny * Nz
    Wx = xp.zeros(n_total, dtype=np.float64)
    Wy = xp.zeros(n_total, dtype=np.float64)
    Wz = xp.zeros(n_total, dtype=np.float64)

    if xi0.size == 0:
        return Wx, Wy, Wz

    xi_m = 0.5 * (xi0 + xi1)
    eta_m = 0.5 * (eta0 + eta1)
    zeta_m = 0.5 * (zeta0 + zeta1)

    dxi = xi1 - xi0
    deta = eta1 - eta0
    dzeta = zeta1 - zeta0

    ix_nod, wx_nod = bspline_weights_1d(xi_m, order, staggered=False, xp=xp)
    iy_nod, wy_nod = bspline_weights_1d(eta_m, order, staggered=False, xp=xp)
    iz_nod, wz_nod = bspline_weights_1d(zeta_m, order, staggered=False, xp=xp)

    ix_stg, wx_stg = bspline_weights_1d(xi_m, order, staggered=True, xp=xp)
    iy_stg, wy_stg = bspline_weights_1d(eta_m, order, staggered=True, xp=xp)
    iz_stg, wz_stg = bspline_weights_1d(zeta_m, order, staggered=True, xp=xp)

    stride_y = Nx
    stride_z = Nx * Ny

    def _scatter_comp(W_flat, q_vel, ix0, wx, iy0, wy, iz0, wz):
        idx_c, w_c = [], []
        for oz, wz_m in enumerate(wz):
            iz_w, vz_m = _wrap_or_mask_1d(iz0 + oz, Nz, periodic[2], xp=xp)
            iz_off = iz_w * stride_z
            for oy, wy_m in enumerate(wy):
                iy_w, vy_m = _wrap_or_mask_1d(iy0 + oy, Ny, periodic[1], xp=xp)
                iyz_off = iy_w * stride_y + iz_off
                for ox, wx_m in enumerate(wx):
                    ix_w, vx_m = _wrap_or_mask_1d(ix0 + ox, Nx, periodic[0], xp=xp)
                    idx = ix_w + iyz_off
                    w = q_vel * (wx_m * wy_m * wz_m)
                    if vx_m is not None:
                        w = xp.where(vx_m, w, 0.0)
                    if vy_m is not None:
                        w = xp.where(vy_m, w, 0.0)
                    if vz_m is not None:
                        w = xp.where(vz_m, w, 0.0)
                    idx_c.append(idx)
                    w_c.append(w)
        _accumulate_flat_chunks(W_flat, idx_c, w_c, n_total, xp=xp)

    _scatter_comp(Wx, qw * dxi, ix_stg, wx_stg, iy_nod, wy_nod, iz_nod, wz_nod)
    _scatter_comp(Wy, qw * deta, ix_nod, wx_nod, iy_stg, wy_stg, iz_nod, wz_nod)
    _scatter_comp(Wz, qw * dzeta, ix_nod, wx_nod, iy_nod, wy_nod, iz_stg, wz_stg)

    return Wx, Wy, Wz
