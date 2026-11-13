"""Image-charge (reflected) particle stencils at planar PEC domain walls.

With ``ParticleSolver(..., wall_stencil="image")`` the part of a shape-``p`` B-spline stencil that
falls outside a planar PEC domain face is folded back into the domain with the PEC image symmetry
instead of being dropped (``wall_stencil="masked"``, the default). For a wall normal to axis ``a``
at logical coordinate ``w``, a particle at ``x`` with charge ``q`` has an image at
``x' = R_a x`` (``x'_a = 2 w - x_a``) with charge ``-q``:

* charge and current: the image path is deposited with the standard (masked) charge and Esirkepov
  current deposits. Each deposit satisfies discrete continuity on every interior node, so Gauss's
  law stays exact; on the wall node and on the wall-adjacent edge the folded values are
  ``rho_k - rho_{-k}`` (odd), ``J_n,k + J_n,-k-1`` (even) and ``J_t,k - J_t,-k`` (odd).
* field gather: the transpose of that deposit, i.e. the masked gather plus the masked gather at the
  image point with ``E_n`` even and ``E_t`` odd (``B_n`` odd, ``B_t`` even, the PEC image of the
  magnetic field). Field work on the folded current therefore equals ``q`` times the path integral
  of the gathered field, as for the interior Galerkin pair.

Consequences (derivation in ``first_principles.md``, "image wall stencil"):

* A particle on the wall has no visible charge and no effective potential: the D17 birth/death
  closure current and the birth kick are identically zero, and none is applied.
* For ``shape = 2`` the gathered normal field in the first half cell is constant (``E_g = E_0``)
  instead of ``(1/2 + zeta) E_0``; the slope jump at ``zeta = 1/2`` drops from ``E_1 - 2 E_0`` to
  ``E_1 - E_0``, which turns the coherent first-cell leapfrog energy error of wall-born particles
  from first order into second order in ``dz``.

Several walls (edges and corners of the box) are handled with the products of the single
reflections (charge sign ``(-1)^{|S|}`` for the set ``S`` of reflected axes).
"""

from __future__ import annotations

import itertools

import numpy as np

from .shapes import deposit_charge, deposit_current_esirkepov, gather_fields

_PEC_BCS = ("pec", "electric")


class WallImages:
    """Image-stencil bookkeeping for the planar PEC faces of a ``ParticleSolver`` domain.

    Parameters
    ----------
    wall_lo, wall_hi : tuple of 3 float
        Logical coordinates of the low and high domain faces per axis (primal nodes).
    periodic : tuple of 3 bool
        Periodic axes (never imaged).
    bc_low, bc_high : tuple of 3 str
        Parsed boundary conditions per axis.
    order : int
        Particle shape order ``p``.
    """

    def __init__(self, wall_lo, wall_hi, periodic, bc_low, bc_high, order: int):
        self.order = int(order)
        # Distance [cells] inside which a stencil (nodal half-width (p + 1) / 2) or one step of
        # path (< 1 cell, Esirkepov CFL) can reach across the wall.
        self.margin = 0.5 * (self.order + 1) + 1.0
        self.faces = []  # (axis, side, wall logical coordinate)
        self.axis_faces = {}
        for axis in range(3):
            if periodic[axis]:
                continue
            for side, bc in (("low", bc_low[axis]), ("high", bc_high[axis])):
                if str(bc).strip().lower() in _PEC_BCS:
                    w = float(wall_lo[axis] if side == "low" else wall_hi[axis])
                    if abs(w - round(w)) > 1e-9:
                        raise ValueError(
                            "wall_stencil='image' needs the PEC face on a primal node plane; "
                            f"axis {axis} {side} face is at logical {w}."
                        )
                    w = float(round(w))
                    self.faces.append((axis, side, w))
                    self.axis_faces.setdefault(axis, []).append((side, w))
        for axis, fl in self.axis_faces.items():
            n_cells = float(wall_hi[axis]) - float(wall_lo[axis])
            # A stencil (width p + 1) plus one step of path must not reach both faces, or second
            # order images (reflections in both faces of one axis) would be needed.
            if len(fl) == 2 and n_cells < self.order + 3.0:
                raise ValueError(
                    f"wall_stencil='image' needs at least {self.order + 3} cells between the two "
                    f"PEC faces along axis {axis}; got {n_cells:g}."
                )
        self.active = bool(self.faces)

    # ------------------------------------------------------------------ helpers
    def _axis_near(self, axis: int, coords, xp):
        """Return ``(near_mask, wall_coord_per_particle)`` for one axis (``None`` if no face)."""
        fl = self.axis_faces.get(axis)
        if not fl:
            return None, None
        if len(fl) == 2:
            # Image across the nearer face only (the other face is out of stencil reach).
            w_lo = fl[0][1] if fl[0][0] == "low" else fl[1][1]
            w_hi = fl[1][1] if fl[0][0] == "low" else fl[0][1]
            mid = 0.5 * (w_lo + w_hi)
            low_side = coords[0] < mid
            m_lo = low_side & (coords[0] < w_lo + self.margin)
            m_hi = (~low_side) & (coords[0] > w_hi - self.margin)
            for c in coords[1:]:
                m_lo = m_lo | (low_side & (c < w_lo + self.margin))
                m_hi = m_hi | ((~low_side) & (c > w_hi - self.margin))
            return m_lo | m_hi, xp.where(low_side, w_lo, w_hi)
        near = None
        wcoord = None
        for side, w in fl:
            if side == "low":
                m = coords[0] < w + self.margin
                for c in coords[1:]:
                    m = m | (c < w + self.margin)
            else:
                m = coords[0] > w - self.margin
                for c in coords[1:]:
                    m = m | (c > w - self.margin)
            if near is None:
                near = m
                wcoord = xp.where(m, w, 0.0)
            else:
                wcoord = xp.where(m, w, wcoord)
                near = near | m
        return near, wcoord

    def image_sets(self, coords_by_axis, xp=np):
        """Yield ``(S, idx, wall)`` for every non-empty set ``S`` of reflected axes.

        Parameters
        ----------
        coords_by_axis : sequence of 3 sequences of arrays
            For each axis, the logical coordinate arrays that decide proximity (e.g. path start and
            end). A particle is imaged across axis ``a`` if any of them is within ``margin``.

        Yields
        ------
        S : tuple of int
            Reflected axes.
        idx : array of int
            Particle indices with images for ``S``.
        wall : dict
            ``{axis: wall_coordinate_array_for_idx}``.
        """
        near = {}
        wall = {}
        for axis in self.axis_faces:
            m, w = self._axis_near(axis, coords_by_axis[axis], xp)
            if m is not None and bool(xp.any(m)):
                near[axis] = m
                wall[axis] = w
        axes = sorted(near)
        for r in range(1, len(axes) + 1):
            for S in itertools.combinations(axes, r):
                m = near[S[0]]
                for a in S[1:]:
                    m = m & near[a]
                idx = xp.nonzero(m)[0]
                if idx.size == 0:
                    continue
                yield S, idx, {a: wall[a][idx] for a in S}

    @staticmethod
    def _mirror(coords, S, idx, wall):
        out = []
        for axis in range(3):
            c = coords[axis][idx]
            if axis in S:
                c = 2.0 * wall[axis] - c
            out.append(c)
        return out

    # --------------------------------------------------------------- deposits
    def deposit_charge(self, xi, eta, zeta, qw, Nx, Ny, Nz, periodic, q_nodes, xp=np):
        """Add the image charges of particles at ``(xi, eta, zeta)`` to ``q_nodes`` [C]."""
        if not self.active or xi.size == 0:
            return
        coords = (xi, eta, zeta)
        qw = xp.broadcast_to(xp.asarray(qw, dtype=np.float64), xi.shape)
        for S, idx, wall in self.image_sets(([xi], [eta], [zeta]), xp=xp):
            m = self._mirror(coords, S, idx, wall)
            sign = -1.0 if (len(S) % 2) else 1.0
            deposit_charge(
                m[0], m[1], m[2], sign * qw[idx], Nx, Ny, Nz,
                order=self.order, periodic=periodic, q_nodes=q_nodes, xp=xp,
            )

    def deposit_current(
        self, xi0, eta0, zeta0, xi1, eta1, zeta1, qw, Nx, Ny, Nz, periodic, W_out, xp=np
    ):
        """Add the Esirkepov dual-face charges ``W`` [C] of the image paths to ``W_out``."""
        if not self.active or xi0.size == 0:
            return
        c0 = (xi0, eta0, zeta0)
        c1 = (xi1, eta1, zeta1)
        qw = xp.broadcast_to(xp.asarray(qw, dtype=np.float64), xi0.shape)
        for S, idx, wall in self.image_sets(([xi0, xi1], [eta0, eta1], [zeta0, zeta1]), xp=xp):
            m0 = self._mirror(c0, S, idx, wall)
            m1 = self._mirror(c1, S, idx, wall)
            sign = -1.0 if (len(S) % 2) else 1.0
            deposit_current_esirkepov(
                m0[0], m0[1], m0[2], m1[0], m1[1], m1[2], sign * qw[idx], Nx, Ny, Nz,
                order=self.order, periodic=periodic,
                Wx_out=W_out[0], Wy_out=W_out[1], Wz_out=W_out[2], xp=xp,
            )

    # ----------------------------------------------------------------- gather
    def field_correction(self, xi, eta, zeta, E_field, B_field, Nx, Ny, Nz, gather, periodic, xp=np):
        """Return the image part of the gathered fields, or ``None`` if no particle is near a wall.

        Returns
        -------
        tuple of 6 arrays or None
            ``(dEx, dEy, dEz, dBx, dBy, dBz)`` of shape ``xi.shape`` [V/m, T], to be added to the
            masked gather.
        """
        if not self.active or xi.size == 0:
            return None
        coords = (xi, eta, zeta)
        out = None
        for S, idx, wall in self.image_sets(([xi], [eta], [zeta]), xp=xp):
            m = self._mirror(coords, S, idx, wall)
            g = gather_fields(
                m[0], m[1], m[2], E_field, B_field, Nx, Ny, Nz,
                order=self.order, gather=gather, periodic=periodic, xp=xp,
            )
            if out is None:
                out = [xp.zeros(xi.shape, dtype=np.float64) for _ in range(6)]
            s_e = -1.0 if (len(S) % 2) else 1.0
            for c in range(3):
                flip = -1.0 if c in S else 1.0
                out[c][idx] += (s_e * flip) * g[c]  # E: normal even, tangential odd
                out[3 + c][idx] += flip * g[3 + c]  # B: normal odd, tangential even
        return None if out is None else tuple(out)
