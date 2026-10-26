# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Macroparticle species container (`Species`) with a GPU-ready
Structure-of-Arrays (SoA) memory layout, amortised capacity growth,
boolean-mask stream compaction, and cell-index sorting.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import m_e

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:  # pragma: no cover - CPU-only installs
    cp = None
    _HAS_CUPY = False


class Species:
    """
    Structure-of-Arrays (SoA) macroparticle container for a single species.

    Positions ``(x, y, z)`` are stored at integer time levels ``t_n`` in ``[m]``
    and relativistic proper velocities ``u = gamma * v = (ux, uy, uz)`` are
    stored at half-integer time levels ``t_{n +- 1/2}`` in ``[m/s]``, with
    Lorentz factor ``gamma = sqrt(1 + |u|^2 / c^2)``. Each macroparticle
    carries weight ``w`` (number of real physical particles represented by the
    macroparticle), so its total charge is ``q * w`` ``[C]`` and its total rest
    mass is ``m * w`` ``[kg]``.

    Parameters
    ----------
    name : str, optional
        Species identifier. Default is ``"electron"``.
    charge : float, optional
        Single-particle physical charge ``q`` [C]. Default is ``-e``
        (CODATA elementary charge).
    mass : float, optional
        Single-particle physical rest mass ``m`` [kg]. Default is ``m_e``
        (CODATA electron mass).
    capacity : int, optional
        Initial pre-allocated buffer capacity (number of macroparticles).
        Buffers grow by a factor of 2 when exceeded. Default is ``1024``.
    track_ids : bool, optional
        If True, maintain a unique 64-bit integer identifier ``id`` per
        macroparticle. Default is True.
    use_gpu : bool, optional
        Store particle buffers on the GPU via CuPy (`xp = cupy`). Default is
        False (`xp = numpy`).
    dtype : numpy dtype, optional
        Floating-point precision of the phase-space and weight arrays.
        Default is ``np.float64``.

    Attributes
    ----------
    name : str
        Species identifier.
    charge, q : float
        Single-particle charge [C].
    mass, m : float
        Single-particle rest mass [kg].
    n_particles : int
        Current number of active macroparticles in the species.
    xp : module
        Array backend (`numpy` or `cupy`).
    """

    _FLOAT_FIELDS = ("_x", "_y", "_z", "_ux", "_uy", "_uz", "_w")

    def __init__(
        self,
        name: str = "electron",
        charge: float = -q_e,
        mass: float = m_e,
        capacity: int = 1024,
        track_ids: bool = True,
        use_gpu: bool = False,
        dtype=np.float64,
    ):
        if float(mass) <= 0.0:
            raise ValueError(f"Species mass must be > 0, got {mass}.")
        if int(capacity) < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}.")

        self.name = str(name)
        self.charge = float(charge)
        self.q = self.charge
        self.mass = float(mass)
        self.m = self.mass
        self.track_ids = bool(track_ids)
        self.on_gpu = bool(use_gpu)
        self.dtype = np.dtype(dtype)

        if self.on_gpu:
            if not _HAS_CUPY:
                raise ImportError(
                    "[!] CuPy could not be imported; cannot use_gpu=True in Species."
                )
            self.xp = cp
        else:
            self.xp = np

        self._capacity = max(int(capacity), 16)
        self._n_particles = 0
        self._n_launch_bound = 0
        self._n_dirty_gpu = False
        self._next_id = 0
        self._gpu_scratch_cap = 0

        if self.on_gpu:
            self._d_n_particles = self.xp.zeros(1, dtype=np.int32)
            self._d_n_next = self.xp.zeros(1, dtype=np.int32)
        else:
            self._d_n_particles = None
            self._d_n_next = None

        for attr in self._FLOAT_FIELDS:
            setattr(self, attr, self.xp.empty(self._capacity, dtype=self.dtype))
        if self.track_ids:
            self._id = self.xp.empty(self._capacity, dtype=np.int64)
        else:
            self._id = None

        # Set to False once the initial half-step back-push u^0 -> u^{-1/2} has been applied
        self.u_is_synced = True

    @property
    def n_particles(self) -> int:
        """Current number of active macroparticles in the species."""
        if self._n_dirty_gpu and self._d_n_particles is not None:
            self._n_particles = int(self._d_n_particles[0].get())
            self._n_launch_bound = self._n_particles
            self._n_dirty_gpu = False
        return self._n_particles

    @n_particles.setter
    def n_particles(self, val: int) -> None:
        v = int(val)
        self._n_particles = v
        self._n_launch_bound = v
        self._n_dirty_gpu = False
        if self.on_gpu and getattr(self, "_d_n_particles", None) is not None:
            self._d_n_particles.fill(v)

    def __len__(self) -> int:
        return self.n_particles

    @property
    def capacity(self) -> int:
        """Current allocated buffer capacity [macroparticles]."""
        return self._capacity

    def _ensure_gpu_scratch(self) -> None:
        """Pre-allocate device scratch buffers for zero-allocation GPU kernels."""
        if not self.on_gpu or self._gpu_scratch_cap >= self._capacity:
            return
        cap = self._capacity
        xp = self.xp
        for attr in self._FLOAT_FIELDS:
            setattr(self, f"{attr}_tmp", xp.empty(cap, dtype=self.dtype))
        if self.track_ids:
            self._id_tmp = xp.empty(cap, dtype=np.int64)
        else:
            self._id_tmp = None

        for name in (
            "_x0_scratch",
            "_y0_scratch",
            "_z0_scratch",
            "_xi0_scratch",
            "_eta0_scratch",
            "_zeta0_scratch",
            "_xi1_scratch",
            "_eta1_scratch",
            "_zeta1_scratch",
            "_xi1_dep_scratch",
            "_eta1_dep_scratch",
            "_zeta1_dep_scratch",
        ):
            setattr(self, name, xp.empty(cap, dtype=self.dtype))

        self._keep_mask_scratch = xp.ones(cap, dtype=np.int32)
        self._scan_dst_idx = xp.zeros(cap, dtype=np.int32)
        n_blocks = max(1, (cap + 255) // 256)
        self._scan_block_sums = xp.zeros(n_blocks, dtype=np.int32)
        self._scan_block_offsets = xp.zeros(n_blocks, dtype=np.int32)
        self._cell_idx_scratch = xp.zeros(cap, dtype=np.int64)
        # Pre-warm CuPy stable argsort workspace in the memory pool
        _ = xp.argsort(self._cell_idx_scratch[: max(self._n_launch_bound, 2)], kind="stable")
        self._gpu_scratch_cap = cap

    def to_gpu(self) -> "Species":
        """Migrate particle buffers to the GPU (`xp = cupy`) in place."""
        if self.on_gpu:
            return self
        if not _HAS_CUPY:
            raise ImportError(
                "[!] CuPy could not be imported; cannot migrate Species to GPU."
            )
        n = self.n_particles
        self.on_gpu = True
        self.xp = cp
        self._d_n_particles = cp.full(1, n, dtype=np.int32)
        self._d_n_next = cp.zeros(1, dtype=np.int32)
        for attr in self._FLOAT_FIELDS:
            old = getattr(self, attr)
            buf = cp.empty(self._capacity, dtype=self.dtype)
            if n > 0:
                buf[:n] = cp.asarray(old[:n], dtype=self.dtype)
            setattr(self, attr, buf)
        if self.track_ids:
            old_id = self._id
            buf_id = cp.empty(self._capacity, dtype=np.int64)
            if n > 0:
                buf_id[:n] = cp.asarray(old_id[:n], dtype=np.int64)
            self._id = buf_id
        self._gpu_scratch_cap = 0
        return self

    def to_cpu(self) -> "Species":
        """Migrate particle buffers to the host CPU (`xp = numpy`) in place."""
        if not self.on_gpu:
            return self
        n = self.n_particles
        self.on_gpu = False
        self.xp = np
        self._d_n_particles = None
        self._d_n_next = None
        self._n_dirty_gpu = False
        self._gpu_scratch_cap = 0
        for attr in self._FLOAT_FIELDS:
            old = getattr(self, attr)
            buf = np.empty(self._capacity, dtype=self.dtype)
            if n > 0:
                buf[:n] = cp.asnumpy(old[:n])
            setattr(self, attr, buf)
        if self.track_ids:
            old_id = self._id
            buf_id = np.empty(self._capacity, dtype=np.int64)
            if n > 0:
                buf_id[:n] = cp.asnumpy(old_id[:n])
            self._id = buf_id
        return self

    def _ensure_capacity(self, required: int) -> None:
        """Grow SoA buffers with amortised doubling when ``required > capacity``."""
        if required <= self._capacity:
            return
        new_cap = max(self._capacity * 2, int(required), 16)
        n = self.n_particles
        for attr in self._FLOAT_FIELDS:
            old = getattr(self, attr)
            buf = self.xp.empty(new_cap, dtype=self.dtype)
            if n > 0:
                buf[:n] = old[:n]
            setattr(self, attr, buf)
        if self.track_ids:
            old_id = self._id
            buf_id = self.xp.empty(new_cap, dtype=np.int64)
            if n > 0:
                buf_id[:n] = old_id[:n]
            self._id = buf_id
        self._capacity = new_cap
        if self.on_gpu and self._gpu_scratch_cap > 0:
            self._ensure_gpu_scratch()

    # --- Active slice views --------------------------------------------------

    @property
    def x(self):
        """Active particle x positions [m], shape ``(n_particles,)``."""
        return self._x[: self.n_particles]

    @x.setter
    def x(self, val):
        self._x[: self.n_particles] = val

    @property
    def y(self):
        """Active particle y positions [m], shape ``(n_particles,)``."""
        return self._y[: self.n_particles]

    @y.setter
    def y(self, val):
        self._y[: self.n_particles] = val

    @property
    def z(self):
        """Active particle z positions [m], shape ``(n_particles,)``."""
        return self._z[: self.n_particles]

    @z.setter
    def z(self, val):
        self._z[: self.n_particles] = val

    @property
    def ux(self):
        """Active particle x proper velocities ``gamma * vx`` [m/s]."""
        return self._ux[: self.n_particles]

    @ux.setter
    def ux(self, val):
        self._ux[: self.n_particles] = val

    @property
    def uy(self):
        """Active particle y proper velocities ``gamma * vy`` [m/s]."""
        return self._uy[: self.n_particles]

    @uy.setter
    def uy(self, val):
        self._uy[: self.n_particles] = val

    @property
    def uz(self):
        """Active particle z proper velocities ``gamma * vz`` [m/s]."""
        return self._uz[: self.n_particles]

    @uz.setter
    def uz(self, val):
        self._uz[: self.n_particles] = val

    @property
    def w(self):
        """Active macroparticle weights (real particles per macro) [-]."""
        return self._w[: self.n_particles]

    @w.setter
    def w(self, val):
        self._w[: self.n_particles] = val

    @property
    def id(self):
        """Unique 64-bit integer macroparticle IDs, or ``None`` if disabled."""
        if not self.track_ids:
            return None
        return self._id[: self.n_particles]

    # --- Relativistic kinematics & diagnostics -------------------------------

    @property
    def gamma(self):
        """Relativistic Lorentz factor ``sqrt(1 + |u|^2 / c^2)`` [-]."""
        if self.n_particles == 0:
            return self.xp.empty(0, dtype=self.dtype)
        u2 = self.ux**2 + self.uy**2 + self.uz**2
        return self.xp.sqrt(1.0 + u2 / (c_light**2))

    @property
    def vx(self):
        """Physical x velocity ``ux / gamma`` [m/s]."""
        return self.ux / self.gamma

    @property
    def vy(self):
        """Physical y velocity ``uy / gamma`` [m/s]."""
        return self.uy / self.gamma

    @property
    def vz(self):
        """Physical z velocity ``uz / gamma`` [m/s]."""
        return self.uz / self.gamma

    def kinetic_energy_per_particle(self):
        """
        Relativistic kinetic energy ``w * (gamma - 1) * m * c^2`` of each
        macroparticle [J].

        Uses the numerically stable expression
        ``w * m * |u|^2 / (gamma + 1)`` to avoid catastrophic cancellation at
        non-relativistic speeds ``|u| << c``.

        Returns
        -------
        ndarray
            Kinetic energy per macroparticle [J], shape ``(n_particles,)``.
        """
        if self.n_particles == 0:
            return self.xp.empty(0, dtype=self.dtype)
        u2 = self.ux**2 + self.uy**2 + self.uz**2
        g = self.xp.sqrt(1.0 + u2 / (c_light**2))
        return self.w * self.mass * (u2 / (g + 1.0))

    def kinetic_energy(self) -> float:
        """
        Total relativistic kinetic energy ``sum_p w_p (gamma_p - 1) m c^2`` [J].

        Returns
        -------
        float
            Total kinetic energy [J].
        """
        if self.n_particles == 0:
            return 0.0
        return float(self.xp.sum(self.kinetic_energy_per_particle()))

    def total_charge(self) -> float:
        """
        Total physical charge ``q * sum_p w_p`` carried by active particles [C].

        Returns
        -------
        float
            Total species charge [C].
        """
        if self.n_particles == 0:
            return 0.0
        return float(self.charge * self.xp.sum(self.w))

    def total_macro_weight(self) -> float:
        """Total number of real particles ``sum_p w_p`` [-]."""
        if self.n_particles == 0:
            return 0.0
        return float(self.xp.sum(self.w))

    # --- Particle insertion, compaction, and sorting -------------------------

    def add_particles(
        self,
        x,
        y,
        z,
        ux=0.0,
        uy=0.0,
        uz=0.0,
        w=1.0,
        *,
        vx=None,
        vy=None,
        vz=None,
    ) -> int:
        """
        Append macroparticles to the species buffers.

        Provide either proper velocities ``(ux, uy, uz)`` [m/s] (where
        ``u = gamma * v``) or physical velocities ``(vx, vy, vz)`` [m/s]
        (with ``|v| < c``). Scalar inputs are broadcast to the length of ``x``.

        Parameters
        ----------
        x, y, z : array_like or float
            Initial particle coordinates [m].
        ux, uy, uz : array_like or float, optional
            Initial proper velocities ``gamma * v`` [m/s]. Default is ``0.0``.
        w : array_like or float, optional
            Macroparticle weights (real particles per macroparticle) [-].
            Default is ``1.0``.
        vx, vy, vz : array_like or float, optional
            Physical velocities [m/s]. If any of ``vx, vy, vz`` is provided,
            ``(ux, uy, uz)`` are computed from ``v / sqrt(1 - |v|^2 / c^2)``.

        Returns
        -------
        int
            Number of macroparticles added.
        """
        xp = self.xp
        x_arr = xp.atleast_1d(xp.asarray(x, dtype=self.dtype)).ravel()
        m_add = int(x_arr.size)
        if m_add == 0:
            return 0

        y_arr = xp.broadcast_to(xp.asarray(y, dtype=self.dtype), (m_add,))
        z_arr = xp.broadcast_to(xp.asarray(z, dtype=self.dtype), (m_add,))
        w_arr = xp.broadcast_to(xp.asarray(w, dtype=self.dtype), (m_add,))

        if vx is not None or vy is not None or vz is not None:
            vx_a = xp.broadcast_to(
                xp.asarray(0.0 if vx is None else vx, dtype=self.dtype), (m_add,)
            )
            vy_a = xp.broadcast_to(
                xp.asarray(0.0 if vy is None else vy, dtype=self.dtype), (m_add,)
            )
            vz_a = xp.broadcast_to(
                xp.asarray(0.0 if vz is None else vz, dtype=self.dtype), (m_add,)
            )
            v2 = vx_a**2 + vy_a**2 + vz_a**2
            if bool(xp.any(v2 >= c_light**2)):
                raise ValueError("Physical velocity magnitude |v| must be < c.")
            gamma_a = 1.0 / xp.sqrt(1.0 - v2 / (c_light**2))
            ux_arr = gamma_a * vx_a
            uy_arr = gamma_a * vy_a
            uz_arr = gamma_a * vz_a
        else:
            ux_arr = xp.broadcast_to(xp.asarray(ux, dtype=self.dtype), (m_add,))
            uy_arr = xp.broadcast_to(xp.asarray(uy, dtype=self.dtype), (m_add,))
            uz_arr = xp.broadcast_to(xp.asarray(uz, dtype=self.dtype), (m_add,))

        n0 = self.n_particles
        n1 = n0 + m_add
        self._ensure_capacity(n1)

        self._x[n0:n1] = x_arr
        self._y[n0:n1] = y_arr
        self._z[n0:n1] = z_arr
        self._ux[n0:n1] = ux_arr
        self._uy[n0:n1] = uy_arr
        self._uz[n0:n1] = uz_arr
        self._w[n0:n1] = w_arr

        if self.track_ids:
            self._id[n0:n1] = xp.arange(
                self._next_id, self._next_id + m_add, dtype=np.int64
            )
            self._next_id += m_add

        self.n_particles = n1
        return m_add

    def _compact_async_gpu(self, keep_mask_i32) -> None:
        """Stream-compact active GPU particles without host synchronization."""
        from .gpu_kernels import compact_species_gpu

        compact_species_gpu(self, keep_mask_i32)

    def keep(self, keep_mask) -> int:
        """
        Stream-compact active particles in place using a boolean ``keep_mask``.

        Parameters
        ----------
        keep_mask : ndarray of bool, shape ``(n_particles,)``
            Boolean mask selecting particles to retain (``True`` = keep,
            ``False`` = remove).

        Returns
        -------
        int
            Number of particles removed.
        """
        n = self.n_particles
        if n == 0:
            return 0
        if self.on_gpu:
            mask = self.xp.asarray(keep_mask)
            if mask.shape != (n,):
                raise ValueError(
                    f"keep_mask shape {mask.shape} does not match n_particles={n}."
                )
            self._ensure_gpu_scratch()
            self._keep_mask_scratch[:n] = mask.astype(np.int32)
            self._compact_async_gpu(self._keep_mask_scratch)
            return n - self.n_particles

        mask = self.xp.asarray(keep_mask, dtype=bool)
        if mask.shape != (n,):
            raise ValueError(
                f"keep_mask shape {mask.shape} does not match n_particles={n}."
            )
        n_keep = int(self.xp.sum(mask))
        if n_keep == n:
            return 0
        if n_keep > 0:
            for attr in self._FLOAT_FIELDS:
                buf = getattr(self, attr)
                buf[:n_keep] = buf[:n][mask]
            if self.track_ids:
                self._id[:n_keep] = self._id[:n][mask]
        self.n_particles = n_keep
        return n - n_keep

    def remove(self, remove_mask) -> int:
        """
        Remove particles where ``remove_mask`` is True via stream compaction.

        Parameters
        ----------
        remove_mask : ndarray of bool, shape ``(n_particles,)``
            Boolean mask selecting particles to remove.

        Returns
        -------
        int
            Number of particles removed.
        """
        if self.n_particles == 0:
            return 0
        return self.keep(~self.xp.asarray(remove_mask, dtype=bool))

    def clear(self) -> None:
        """Remove all active particles while retaining allocated buffers."""
        self.n_particles = 0

    def reorder(self, perm) -> None:
        """
        Permute active particles in place according to index array ``perm``.

        Parameters
        ----------
        perm : ndarray of int, shape ``(n_particles,)``
            Permutation of ``0 .. n_particles - 1``.
        """
        n = self.n_particles
        if n <= 1:
            return
        if self.xp is np and n > 65536:
            tmp = np.empty(n, dtype=self.dtype)
            for attr in self._FLOAT_FIELDS:
                buf = getattr(self, attr)
                np.take(buf[:n], perm, out=tmp)
                np.copyto(buf[:n], tmp)
            del tmp
            if self.track_ids:
                tmp_id = np.empty(n, dtype=np.int64)
                np.take(self._id[:n], perm, out=tmp_id)
                np.copyto(self._id[:n], tmp_id)
                del tmp_id
            return
        for attr in self._FLOAT_FIELDS:
            buf = getattr(self, attr)
            buf[:n] = buf[:n][perm]
        if self.track_ids:
            self._id[:n] = self._id[:n][perm]

    def sort_by_cell(self, grid_map) -> None:
        """
        Sort active particles by their linear 3D cell index for cache locality.

        Parameters
        ----------
        grid_map : LogicalGridMap
            Grid coordinate mapper providing ``linear_cell_index(x, y, z)``.
        """
        if self.on_gpu:
            from .gpu_kernels import sort_species_by_cell_gpu

            sort_species_by_cell_gpu(self, grid_map)
            return
        if self.n_particles <= 1:
            return
        cell_idx = grid_map.linear_cell_index(self.x, self.y, self.z)
        perm = self.xp.argsort(cell_idx, kind="stable")
        del cell_idx
        if self.xp is np and 65536 < self.n_particles < 2**31:
            perm = perm.astype(np.int32)
        self.reorder(perm)

    def copy(self) -> Species:
        """Return a deep copy of this ``Species`` instance."""
        out = Species(
            name=self.name,
            charge=self.charge,
            mass=self.mass,
            capacity=max(self._capacity, self.n_particles),
            track_ids=self.track_ids,
            use_gpu=self.on_gpu,
            dtype=self.dtype,
        )
        n = self.n_particles
        out.n_particles = n
        out._next_id = self._next_id
        out.u_is_synced = self.u_is_synced
        if n > 0:
            for attr in self._FLOAT_FIELDS:
                getattr(out, attr)[:n] = getattr(self, attr)[:n]
            if self.track_ids:
                out._id[:n] = self._id[:n]
        return out
