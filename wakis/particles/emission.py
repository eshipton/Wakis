# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Cathode surface emission models for ``wakis.particles``:

* :class:`SpaceChargeLimitedEmitter` — Gauss-law surface emission converging to
  the Child–Langmuir space-charge-limited current density.
* :class:`ThermionicEmitter` — Richardson–Dushman thermionic emission with
  Schottky image-charge barrier lowering and optional space-charge limiting.
* :class:`FowlerNordheimEmitter` — Cold field emission in the Murphy–Good
  formulation using the Forbes (2006) approximations for the Schottky–Nordheim
  barrier functions :math:`v(f)` and :math:`t(f)`.

All emitter classes implement the :class:`~wakis.particles.boundaries.InjectorProtocol`
interface expected by ``ParticleSolver(injectors=[...])`` (matching
:class:`~wakis.particles.BeamInjector`):
``_bind(solver)``, ``inject(t, dt, step_index=None, *, owning=True) -> int``,
``species``, ``z0``, ``emitter_kind = "pec_wall"``, ``injected_charge``,
``injected_energy``, ``state_dict()``, and ``load_state_dict(state)``.

Time-level and GPU execution notes
----------------------------------
* **Electric-field time level**: ``ParticleSolver._mid_step_hook`` (CPU) and
  ``ParticleSolver._mid_step_hook_gpu`` (GPU) invoke ``inj.inject(t_n, dt)`` at
  the beginning of step ``n``, after ``SolverFIT3D`` has advanced
  :math:`\\mathbf{B}^{n-1/2} \\to \\mathbf{B}^{n+1/2}` and **before** the
  particle field gather, relativistic Boris push, boundary absorption,
  Esirkepov current deposition, and Ampere update
  :math:`\\mathbf{E}^n \\to \\mathbf{E}^{n+1}`. Each emitter therefore reads the
  electric field :math:`\\mathbf{E}^n` at integer time level
  :math:`t_n = n\\,\\Delta t`.
* **GPU step path and host-synchronization accounting**: On the GPU
  (``solver.use_gpu=True``), ``inject(t, dt)`` injects a fixed number of
  macroparticles per active emitting node per step
  (``n_emit = n_emitting_cells * n_macro_per_cell``) and computes their weights
  entirely on the device from the local normal electric field, adding **no
  device-to-host synchronization of its own** (``cp.ndarray.get`` is never
  called by the emitter, and ``injected_charge``, ``injected_energy``, and
  ``voltage_source_work`` are accumulated in 0-D CuPy arrays wrapped by
  :class:`_DeviceScalar`). Note, however, that the solver call site in
  ``ParticleSolver._mid_step_hook_gpu`` (``wakis/particles/solver.py`` ~l.2373)
  queries ``n_before = inj.species.n_particles``, which synchronizes the
  device-side survivor count ``_d_n_particles[0].get()`` when
  ``species._n_dirty_gpu`` is ``True`` after boundary compaction on the
  preceding step; once ``species.n_particles`` is clean, a full
  ``solver.one_step()`` with ``inject(t, dt)`` executes without any host
  synchronization.
* **Zero-weight particles and stream compaction**: Nodes with zero or retarding
  surface field (:math:`E_{\\text{pull}} \\le 0`) receive weight ``w = 0.0``.
  On CPU (``solver.use_gpu=False``), ``inject(t, dt)`` skips ``add_particles``
  and returns ``0`` whenever every weight is zero (e.g. when :math:`\\mathbf{E} \\equiv 0`),
  preventing zero-weight particle pile-up. On GPU (``solver.use_gpu=True``), to
  avoid a host synchronization checking ``any(w > 0)``, ``inject(t, dt)``
  appends ``n_emit_per_step`` macroparticles with ``w = 0.0``. Because stream
  compaction (``Species.keep`` on CPU and ``compact_species_gpu`` on GPU)
  filters strictly by ``alive != 0`` and not by ``w == 0.0``, zero-weight GPU
  macroparticles that remain inside the domain (for example, stationary at
  :math:`z = z_{\\min}` with ``thermal_u == 0`` when :math:`E_z \\equiv 0`) are
  retained by compaction and incur the per-particle field-gather, Boris-push,
  and Esirkepov current-deposition memory bandwidth and arithmetic cost on
  subsequent steps while contributing identically zero to :math:`\\mathbf{J}`
  and :math:`\\rho`. If the local field is strictly retarding
  (:math:`-\\operatorname{sgn}(q_s) E_z < 0`), the particle is pushed across
  :math:`z < z_{\\min}` on the next step and removed by the absorbing boundary
  check.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0, h, hbar, m_e, pi
from scipy.constants import k as k_B

from .mapping import LogicalGridMap
from .shapes import _wrap_or_mask_1d, bspline_weights_1d
from .species import Species

try:
    import cupy as cp
except ImportError:  # pragma: no cover - CPU-only installs
    cp = None


# Universal Richardson constant A_0 = 4 * pi * m_e * k_B^2 * e / h^3 [A / (m^2 K^2)]
# Reference: S. Dushman, Phys. Rev. 21, 623 (1923), doi:10.1103/PhysRev.21.623
RICHARDSON_A0 = (4.0 * pi * m_e * (k_B**2) * q_e) / (h**3)

# First Fowler-Nordheim constant a_FN = e^3 / (16 * pi^2 * hbar * eV) = e^2 / (16 * pi^2 * hbar) [A eV V^-2]
# and second Fowler-Nordheim constant b_FN = 4 * sqrt(2 * m_e * e) / (3 * hbar) [eV^(-3/2) V m^-1]
# (obtained from the pure-SI exponent 4 * sqrt(2 * m_e) * phi_J^(3/2) / (3 * hbar * e * F) with phi_J = e * phi_eV)
# Reference: E. L. Murphy and R. H. Good, Jr., Phys. Rev. 102, 1464 (1956), doi:10.1103/PhysRev.102.1464;
# R. G. Forbes, Appl. Phys. Lett. 89, 113122 (2006), doi:10.1063/1.2354582
FN_A_CONST = (q_e**2) / (16.0 * (pi**2) * hbar)
FN_B_CONST = (4.0 * np.sqrt(2.0 * m_e * q_e)) / (3.0 * hbar)

# Schottky constant c_S^2 = e^3 / (4 * pi * eps_0) [J^2 m / V] and in [eV^2 m / V]
SCHOTTKY_PREFACTOR_SI = (q_e**3) / (4.0 * pi * epsilon_0)
SCHOTTKY_PREFACTOR_EV2 = q_e / (4.0 * pi * epsilon_0)


def _xp_for(arr):
    """Return ``cupy`` if ``arr`` is a CuPy array, else ``numpy``."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


class _DeviceScalar:
    """
    Lightweight wrapper around a 0-D CuPy scalar that keeps ``abs(x)`` on the
    device during ``ParticleSolver._mid_step_hook_gpu`` while converting to a
    Python ``float`` on demand when inspected from host code.
    """

    __slots__ = ("_d",)

    def __init__(self, d_arr):
        self._d = d_arr

    def get(self) -> float:
        """Synchronize and return the scalar value as a host ``float``."""
        return float(self._d.get())

    def __abs__(self) -> _DeviceScalar:
        return _DeviceScalar(cp.abs(self._d))

    def __neg__(self) -> _DeviceScalar:
        return _DeviceScalar(-self._d)

    def __pos__(self) -> _DeviceScalar:
        return self

    def __float__(self) -> float:
        return float(self._d.get())

    def __int__(self) -> int:
        return int(float(self._d.get()))

    def __array__(self, dtype=None):
        val = np.asarray(float(self._d.get()))
        return val if dtype is None else val.astype(dtype)

    def __repr__(self) -> str:
        return repr(float(self._d.get()))

    def __str__(self) -> str:
        return str(float(self._d.get()))

    def __format__(self, format_spec: str) -> str:
        return format(float(self._d.get()), format_spec)

    def __eq__(self, other) -> bool:
        return float(self) == float(other)

    def __lt__(self, other) -> bool:
        return float(self) < float(other)

    def __le__(self, other) -> bool:
        return float(self) <= float(other)

    def __gt__(self, other) -> bool:
        return float(self) > float(other)

    def __ge__(self, other) -> bool:
        return float(self) >= float(other)

    def __add__(self, other) -> float:
        return float(self) + float(other)

    def __radd__(self, other) -> float:
        return float(other) + float(self)

    def __sub__(self, other) -> float:
        return float(self) - float(other)

    def __rsub__(self, other) -> float:
        return float(other) - float(self)

    def __mul__(self, other) -> float:
        return float(self) * float(other)

    def __rmul__(self, other) -> float:
        return float(other) * float(self)

    def __truediv__(self, other) -> float:
        return float(self) / float(other)

    def __rtruediv__(self, other) -> float:
        return float(other) / float(self)


def child_langmuir_current_density(
    voltage: float,
    gap: float,
    charge: float = -q_e,
    mass: float = m_e,
) -> float:
    """
    Analytical 1D non-relativistic Child–Langmuir space-charge-limited current
    density [A/m^2].

    .. math::

        J_{\\text{CL}} = \\frac{4 \\varepsilon_0}{9}
        \\sqrt{\\frac{2 |q|}{m}}\\,\\frac{V^{3/2}}{d^2}

    References: C. D. Child, Phys. Rev. (Series I) 32, 492 (1911),
    doi:10.1103/PhysRevSeriesI.32.492; I. Langmuir, Phys. Rev. 2, 450 (1913),
    doi:10.1103/PhysRev.2.450.

    Parameters
    ----------
    voltage : float
        Anode-to-cathode gap voltage :math:`V > 0` [V].
    gap : float
        Planar diode gap separation :math:`d > 0` [m].
    charge : float, optional
        Single-particle species charge :math:`q` [C]. Default is ``-e``.
    mass : float, optional
        Single-particle species rest mass :math:`m` [kg]. Default is ``m_e``.

    Returns
    -------
    float
        Steady-state space-charge-limited current density magnitude
        :math:`J_{\\text{CL}}` [A/m^2].
    """
    if float(voltage) < 0.0:
        raise ValueError(f"voltage must be >= 0, got {voltage}.")
    if float(gap) <= 0.0:
        raise ValueError(f"gap must be > 0, got {gap}.")
    if float(mass) <= 0.0:
        raise ValueError(f"mass must be > 0, got {mass}.")
    return float(
        (4.0 * epsilon_0 / 9.0)
        * np.sqrt(2.0 * abs(float(charge)) / float(mass))
        * (float(voltage) ** 1.5)
        / (float(gap) ** 2)
    )


def schottky_barrier_lowering(E_field):
    """
    Schottky image-charge work-function lowering
    :math:`\\Delta W_S = \\sqrt{e^3 F / (4 \\pi \\varepsilon_0)}` [J].

    For zero or retarding surface fields (:math:`F \\le 0`), the barrier
    lowering is zero.

    References: W. Schottky, Phys. Z. 15, 872 (1914); E. L. Murphy and
    R. H. Good, Jr., Phys. Rev. 102, 1464 (1956), doi:10.1103/PhysRev.102.1464.

    Parameters
    ----------
    E_field : array_like or float
        Normal pulling surface electric field magnitude :math:`F` [V/m].

    Returns
    -------
    ndarray or float
        Barrier lowering :math:`\\Delta W_S` [J].
    """
    xp = _xp_for(E_field)
    F_arr = xp.asarray(E_field, dtype=np.float64)
    F_pos = xp.maximum(F_arr, 0.0)
    dW = xp.sqrt(SCHOTTKY_PREFACTOR_SI * F_pos)
    if np.ndim(E_field) == 0 and xp is np:
        return float(dW)
    return dW


def richardson_dushman_current_density(
    E_field,
    temperature: float,
    work_function_eV: float,
    lambda_R: float = 1.0,
):
    """
    Richardson–Dushman thermionic current density with Schottky barrier
    lowering [A/m^2].

    .. math::

        J_{\\text{RD}}(F, T) = \\lambda_R A_0 T^2
        \\exp\\!\\left(-\\frac{W - \\Delta W_S(F)}{k_B T}\\right),
        \\qquad \\Delta W_S(F) = \\sqrt{\\frac{e^3 \\max(F, 0)}{4\\pi\\varepsilon_0}}

    References: O. W. Richardson, *The Emission of Electricity from Hot Bodies*
    (Longmans, Green and Co., London, 1916); S. Dushman, Phys. Rev. 21, 623
    (1923), doi:10.1103/PhysRev.21.623; W. Schottky, Phys. Z. 15, 872 (1914).

    Parameters
    ----------
    E_field : array_like or float
        Normal pulling electric field :math:`F` at the cathode surface [V/m].
    temperature : float
        Cathode absolute temperature :math:`T > 0` [K].
    work_function_eV : float
        Material work function :math:`W / e > 0` [eV].
    lambda_R : float, optional
        Material-specific Richardson correction factor :math:`\\lambda_R > 0`
        [-]. Default is ``1.0``.

    Returns
    -------
    ndarray or float
        Thermionic emission current density :math:`J_{\\text{RD}}` [A/m^2].
    """
    T = float(temperature)
    W_eV = float(work_function_eV)
    lam = float(lambda_R)
    if T <= 0.0:
        raise ValueError(f"temperature must be > 0 K, got {temperature}.")
    if W_eV <= 0.0:
        raise ValueError(f"work_function_eV must be > 0 eV, got {work_function_eV}.")
    if lam <= 0.0:
        raise ValueError(f"lambda_R must be > 0, got {lambda_R}.")

    xp = _xp_for(E_field)
    F_arr = xp.asarray(E_field, dtype=np.float64)
    dW_S = schottky_barrier_lowering(F_arr)
    W_J = W_eV * q_e
    # Effective barrier W_eff = max(W - dW_S, 0) in [J]
    W_eff = xp.maximum(W_J - dW_S, 0.0)
    J = (lam * RICHARDSON_A0 * (T**2)) * xp.exp(-W_eff / (k_B * T))
    if np.ndim(E_field) == 0 and xp is np:
        return float(J)
    return J


def forbes_barrier_functions(f):
    """
    Forbes (2006) analytical approximations for the Murphy–Good /
    Schottky–Nordheim barrier functions :math:`v(f)` and :math:`t(f)`.

    For dimensionless scaled barrier field
    :math:`f = (e^3 / (4\\pi\\varepsilon_0)) F / \\phi^2 \\in [0, 1]`:

    .. math::

        v(f) = 1 - f + \\frac{1}{6}\\,f \\ln f, \\qquad
        t(f) = 1 + \\frac{f}{9} - \\frac{1}{18}\\,f \\ln f

    with exact limiting values :math:`\\lim_{f \\to 0^+} v(f) = 1`,
    :math:`\\lim_{f \\to 0^+} t(f) = 1`, and at the barrier top :math:`f = 1`:
    :math:`v(1) = 0`, :math:`t(1) = 10/9`. For over-barrier inputs
    :math:`f \\ge 1` (where the Schottky–Nordheim barrier peak drops to or
    below the Fermi level and WKB tunneling ceases to apply), a
    :class:`RuntimeWarning` is emitted on host inputs and :math:`f` is clamped
    to :math:`[0, 1]` so :math:`v(f) \\ge 0` and :math:`t(f) \\le 10/9`.

    Reference: R. G. Forbes, Appl. Phys. Lett. 89, 113122 (2006),
    doi:10.1063/1.2354582; R. G. Forbes and J. H. B. Deane, Proc. R. Soc. A
    463, 2907 (2007), doi:10.1098/rspa.2007.0030.

    Parameters
    ----------
    f : array_like or float
        Dimensionless scaled barrier electric field :math:`f \\ge 0` [-].

    Returns
    -------
    v : ndarray or float
        Barrier shape factor :math:`v(f)` in the Fowler–Nordheim exponent [-].
    t : ndarray or float
        Barrier shape factor :math:`t(f)` in the Fowler–Nordheim prefactor [-].
    """
    xp = _xp_for(f)
    f_arr = xp.asarray(f, dtype=np.float64)
    if xp is np and bool(np.any(f_arr >= 1.0)):
        warnings.warn(
            "Scaled barrier field f >= 1 reaches or exceeds the Schottky-Nordheim "
            "barrier top where the WKB tunneling approximation ceases to apply; "
            "clamping f to [0, 1] (v = 0, t = 10/9).",
            RuntimeWarning,
            stacklevel=2,
        )
    f_pos = xp.clip(f_arr, 0.0, 1.0)
    # Safely evaluate f * ln(f) with the exact limit 0 as f -> 0^+
    f_safe = xp.where(f_pos > 0.0, f_pos, 1.0)
    f_ln_f = xp.where(f_pos > 0.0, f_pos * xp.log(f_safe), 0.0)

    v = 1.0 - f_pos + (1.0 / 6.0) * f_ln_f
    t = 1.0 + (1.0 / 9.0) * f_pos - (1.0 / 18.0) * f_ln_f

    if np.ndim(f) == 0 and xp is np:
        return float(v), float(t)
    return v, t


def fowler_nordheim_current_density(
    E_field,
    work_function_eV: float,
    beta_fe: float = 1.0,
):
    """
    Murphy–Good cold field-emission current density [A/m^2] using the Forbes
    (2006) approximations for :math:`v(f)` and :math:`t(f)`.

    With local barrier electric field :math:`F_{\\text{loc}} = \\beta_{\\text{FE}} F`
    and scaled field :math:`f = (e^3 / (4\\pi\\varepsilon_0)) F_{\\text{loc}} / \\phi^2`:

    .. math::

        J_{\\text{FN}}(F) = \\frac{a_{\\text{FN}}\\,F_{\\text{loc}}^2}
        {\\phi_{\\text{eV}}\\,t(f)^2}
        \\exp\\!\\left(-\\frac{b_{\\text{FN}}\\,\\phi_{\\text{eV}}^{3/2}}
        {F_{\\text{loc}}}\\,v(f)\\right)

    For :math:`F \\le 0`, :math:`J_{\\text{FN}} = 0`.

    References: R. H. Fowler and L. Nordheim, Proc. R. Soc. Lond. A 119, 173
    (1928), doi:10.1098/rspa.1928.0091; E. L. Murphy and R. H. Good, Jr.,
    Phys. Rev. 102, 1464 (1956), doi:10.1103/PhysRev.102.1464; R. G. Forbes,
    Appl. Phys. Lett. 89, 113122 (2006), doi:10.1063/1.2354582.

    Parameters
    ----------
    E_field : array_like or float
        Macroscopic normal pulling electric field :math:`F` [V/m].
    work_function_eV : float
        Work function :math:`\\phi_{\\text{eV}} > 0` [eV].
    beta_fe : float, optional
        Local field-enhancement factor :math:`\\beta_{\\text{FE}} > 0` [-].
        Default is ``1.0``.

    Returns
    -------
    ndarray or float
        Field-emission current density :math:`J_{\\text{FN}}` [A/m^2].
    """
    phi_eV = float(work_function_eV)
    beta = float(beta_fe)
    if phi_eV <= 0.0:
        raise ValueError(f"work_function_eV must be > 0 eV, got {work_function_eV}.")
    if beta <= 0.0:
        raise ValueError(f"beta_fe must be > 0, got {beta_fe}.")

    xp = _xp_for(E_field)
    F_arr = xp.asarray(E_field, dtype=np.float64)
    F_loc = beta * xp.maximum(F_arr, 0.0)

    # Scaled barrier field f = (e / (4*pi*eps_0)) * F_loc / phi_eV^2
    f = (SCHOTTKY_PREFACTOR_EV2 / (phi_eV**2)) * F_loc
    v_f, t_f = forbes_barrier_functions(f)
    # Clamp v(f) >= 0 above barrier top (f > 1) so the tunneling exponent does not invert
    v_pos = xp.maximum(v_f, 0.0)

    F_safe = xp.where(F_loc > 0.0, F_loc, 1.0)
    exponent = xp.where(
        F_loc > 0.0,
        -(FN_B_CONST * (phi_eV**1.5) * v_pos) / F_safe,
        -np.inf,
    )
    prefactor = (FN_A_CONST * (F_loc**2)) / (phi_eV * (t_f**2))
    J = xp.where(F_loc > 0.0, prefactor * xp.exp(exponent), 0.0)

    if np.ndim(E_field) == 0 and xp is np:
        return float(J)
    return J


class _SurfaceEmitterBase:
    """
    Base class for planar PEC cathode surface emitters on ``z0 = grid.zmin``.

    Implements :class:`~wakis.particles.boundaries.InjectorProtocol` called by
    :class:`~wakis.particles.ParticleSolver`:
    ``_bind(solver)``, ``inject(t, dt, step_index=None, *, owning=True) -> int``,
    ``species``, ``z0``, ``emitter_kind = "pec_wall"``, ``injected_charge``,
    ``injected_energy``, ``state_dict()``, and ``load_state_dict(state)``.
    """

    def __init__(
        self,
        solver,
        species: Species,
        *,
        n_macro_per_cell: int = 1,
        thermal_u: float = 0.0,
        x_bounds: tuple[float, float] | None = None,
        y_bounds: tuple[float, float] | None = None,
        radius: float | None = None,
        center: tuple[float, float] = (0.0, 0.0),
        voltage: float | None = None,
        maintain_voltage: bool = False,
        t_start: float = 0.0,
        t_end: float = float("inf"),
    ):
        if not isinstance(species, Species):
            raise TypeError("species must be a Species instance.")
        if solver is None or not hasattr(solver, "grid") or not hasattr(solver, "E"):
            raise TypeError("solver must be a SolverFIT3D instance with grid and E.")
        if int(n_macro_per_cell) < 1:
            raise ValueError(f"n_macro_per_cell must be >= 1, got {n_macro_per_cell}.")
        if float(thermal_u) < 0.0:
            raise ValueError(f"thermal_u must be >= 0, got {thermal_u}.")
        if x_bounds is not None and radius is not None:
            raise ValueError(
                "Specify either rectangular bounds (x_bounds/y_bounds) or disk radius, not both."
            )
        if y_bounds is not None and radius is not None:
            raise ValueError(
                "Specify either rectangular bounds (x_bounds/y_bounds) or disk radius, not both."
            )
        if radius is not None and float(radius) <= 0.0:
            raise ValueError(f"radius must be > 0, got {radius}.")
        if x_bounds is not None and float(x_bounds[0]) > float(x_bounds[1]):
            raise ValueError(f"x_bounds must satisfy xmin <= xmax, got {x_bounds}.")
        if y_bounds is not None and float(y_bounds[0]) > float(y_bounds[1]):
            raise ValueError(f"y_bounds must satisfy ymin <= ymax, got {y_bounds}.")

        self.species = species
        self.n_macro_per_cell = int(n_macro_per_cell)
        self.thermal_u = float(thermal_u)
        self.x_bounds = (
            (float(x_bounds[0]), float(x_bounds[1])) if x_bounds is not None else None
        )
        self.y_bounds = (
            (float(y_bounds[0]), float(y_bounds[1])) if y_bounds is not None else None
        )
        self.radius = float(radius) if radius is not None else None
        self.center = (float(center[0]), float(center[1]))
        self.voltage = float(voltage) if voltage is not None else None
        self._maintain_voltage = bool(maintain_voltage)
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.emitter_kind: str | None = None

        # Cumulative diagnostics (device-resident on GPU to avoid host syncs)
        self._injected_charge_host = 0.0
        self._injected_energy_host = 0.0
        self._voltage_source_work_host = 0.0
        self._d_injected_charge = None
        self._d_injected_energy = None
        self._d_voltage_source_work = None
        self.injected_count = 0
        self.step_index = 0
        self._V_target = None
        self._k_read = 0

        self._bind(solver)

    def _bind(self, solver):
        """Validate and bind cathode grid/field buffers on ``solver`` and set ``emitter_kind = 'pec_wall'``."""
        if solver is None or not hasattr(solver, "grid") or not hasattr(solver, "E"):
            raise TypeError("solver must be a SolverFIT3D instance with grid and E.")

        self._use_mpi = bool(
            getattr(solver, "use_mpi", False) or getattr(solver.grid, "use_mpi", False)
        )
        if self._use_mpi and (self.voltage is not None or self._maintain_voltage):
            raise NotImplementedError(
                "Gap voltage initialization (voltage != None) and gap voltage "
                "clamping (maintain_voltage=True) require the global longitudinal "
                "domain [ZMIN, ZMAX] and a global reduction of E_z across MPI "
                "z-slabs, which are not implemented under use_mpi=True."
            )

        bc_z_low = str(solver.bc_low[2]).lower()
        if self._use_mpi and bc_z_low == "mpi":
            global_bc_z = getattr(solver, "_global_bc_z", None)
            if global_bc_z is not None:
                bc_z_low = str(global_bc_z[0]).lower()
            else:
                logged_bc_low = getattr(
                    getattr(solver, "logger", None), "solver", {}
                ).get("bc_low")
                if logged_bc_low is not None and len(logged_bc_low) >= 3:
                    bc_z_low = str(logged_bc_low[2]).lower()
                elif (
                    int(getattr(solver, "size", 1)) > 1
                    and getattr(solver, "comm", None) is not None
                ):
                    comm = solver.comm
                    size = int(solver.size)
                    raw_z_low = (
                        str(comm.bcast(solver.bc_low[2], root=0)).strip().lower()
                    )
                    raw_z_high = (
                        str(comm.bcast(solver.bc_high[2], root=size - 1))
                        .strip()
                        .lower()
                    )
                    solver._global_bc_z = (raw_z_low, raw_z_high)
                    bc_z_low = raw_z_low
        if bc_z_low not in ("pec", "electric"):
            raise ValueError(
                f"Cathode face zmin must have a PEC boundary condition "
                f"(bc_low[2]='pec'), got {solver.bc_low[2]!r}."
            )

        self.solver = solver
        species = self.species
        grid = solver.grid
        if self._use_mpi and getattr(grid, "ZMIN", None) is not None:
            self.z0 = float(grid.ZMIN)
        else:
            self.z0 = float(getattr(grid, "global_zmin", grid.zmin))

        self._use_gpu = bool(getattr(solver, "use_gpu", False) or species.on_gpu)
        self.xp = cp if self._use_gpu else np
        self.grid_map = LogicalGridMap.from_grid(solver, xp=self.xp)

        edge_ghosts = tuple(int(g) for g in getattr(grid, "edge_ghosts", (0, 0, 0)))
        x_full = np.asarray(grid.x, dtype=np.float64)
        y_full = np.asarray(grid.y, dtype=np.float64)
        z_full = np.asarray(grid.z, dtype=np.float64)
        x_nodes = x_full[: len(x_full) - edge_ghosts[0]]
        y_nodes = y_full[: len(y_full) - edge_ghosts[1]]
        z_nodes = z_full[: len(z_full) - edge_ghosts[2]]
        dx = np.diff(x_nodes)
        dy = np.diff(y_nodes)
        dz = np.diff(z_nodes)

        self._Nx_phys = int(dx.size)
        self._Ny_phys = int(dy.size)
        self._Nz_phys = int(dz.size)
        self._Nx_nodes = int(solver.Nx)
        self._Ny_nodes = int(solver.Ny)
        self._x_min = float(x_nodes[0])
        self._y_min = float(y_nodes[0])
        self._z_min = float(z_nodes[0])
        self._z_max = float(z_nodes[-1])
        self._gap_d = float(self._z_max - self._z_min)
        self._dx0 = float(dx[0])
        self._dy0 = float(dy[0])
        self._dz0 = float(dz[0])

        bc_low = getattr(solver, "bc_low", ("periodic", "periodic", "pec"))
        bc_high = getattr(solver, "bc_high", ("periodic", "periodic", "pec"))
        self._periodic_xy = (
            str(bc_low[0]).strip().lower() in ("periodic", "bloch")
            and str(bc_high[0]).strip().lower() in ("periodic", "bloch"),
            str(bc_low[1]).strip().lower() in ("periodic", "bloch")
            and str(bc_high[1]).strip().lower() in ("periodic", "bloch"),
        )

        # Pair primary longitudinal edge E_z[i, j, 0] at transverse primal node
        # (x_i, y_j) with its FIT dual face area A_tilde_z[i, j] = 1 / itA_z[i, j, 0].
        itA_field = getattr(solver, "itA", grid.itA)
        itAz_raw = itA_field.to_matrix("z")[: self._Nx_nodes, : self._Ny_nodes, 0]
        if hasattr(itAz_raw, "get"):
            itAz_0 = np.asarray(itAz_raw.get(), dtype=np.float64)
        else:
            itAz_0 = np.asarray(itAz_raw, dtype=np.float64)
        area_2d = np.where(itAz_0 > 0.0, 1.0 / np.where(itAz_0 > 0.0, itAz_0, 1.0), 0.0)

        # Build transverse patch mask over primal nodes / dual cells (Nx_nodes, Ny_nodes)
        x_node_1d = x_full[: self._Nx_nodes]
        y_node_1d = y_full[: self._Ny_nodes]
        xx_n, yy_n = np.meshgrid(x_node_1d, y_node_1d, indexing="ij")
        mask = area_2d > 0.0
        if self.x_bounds is not None:
            mask &= (xx_n >= self.x_bounds[0]) & (xx_n <= self.x_bounds[1])
        if self.y_bounds is not None:
            mask &= (yy_n >= self.y_bounds[0]) & (yy_n <= self.y_bounds[1])
        if self.radius is not None:
            r2 = (xx_n - self.center[0]) ** 2 + (yy_n - self.center[1]) ** 2
            mask &= r2 <= (self.radius**2)

        ii, jj = np.nonzero(mask)
        self._cell_i_cpu = ii.astype(np.int64)
        self._cell_j_cpu = jj.astype(np.int64)
        self.n_emitting_cells = int(ii.size)
        self.n_emit_per_step = int(self.n_emitting_cells * self.n_macro_per_cell)

        cell_areas = area_2d[ii, jj]
        self._cell_areas_cpu = cell_areas.astype(np.float64)
        self._total_domain_area = float(np.sum(area_2d))

        # Emit macroparticles at the cathode primal nodes (x_i, y_j, z0) so that
        # nodal CIC charge deposition and E_z[i, j, 0] dual-face geometry are consistent.
        if self.n_emitting_cells > 0:
            x_n = x_node_1d[ii]
            y_n = y_node_1d[jj]
            m = self.n_macro_per_cell
            if m == 1:
                x_emit = x_n.copy()
                y_emit = y_n.copy()
            else:
                x_emit = np.repeat(x_n, m)
                y_emit = np.repeat(y_n, m)
            z_emit = np.full(self.n_emit_per_step, self.z0, dtype=np.float64)
        else:
            x_emit = np.empty(0, dtype=np.float64)
            y_emit = np.empty(0, dtype=np.float64)
            z_emit = np.empty(0, dtype=np.float64)

        self._cell_i = self.xp.asarray(self._cell_i_cpu)
        self._cell_j = self.xp.asarray(self._cell_j_cpu)
        self._cell_areas = self.xp.asarray(self._cell_areas_cpu)
        self._domain_area_2d = self.xp.asarray(area_2d, dtype=np.float64)
        self._dz_vec = self.xp.asarray(dz, dtype=np.float64)
        self._x_emit = self.xp.asarray(x_emit, dtype=species.dtype)
        self._y_emit = self.xp.asarray(y_emit, dtype=species.dtype)
        self._z_emit = self.xp.asarray(z_emit, dtype=species.dtype)
        self._ux_emit = self.xp.zeros(self.n_emit_per_step, dtype=species.dtype)
        self._uy_emit = self.xp.zeros(self.n_emit_per_step, dtype=species.dtype)
        self._uz_emit = self.xp.full(
            self.n_emit_per_step, self.thermal_u, dtype=species.dtype
        )
        self._q0_grid = self.xp.zeros(
            (self._Nx_nodes, self._Ny_nodes), dtype=np.float64
        )

        if self._use_gpu:
            if self._d_injected_charge is None:
                self._d_injected_charge = cp.asarray(
                    self._injected_charge_host, dtype=np.float64
                )
            if self._d_injected_energy is None:
                self._d_injected_energy = cp.asarray(
                    self._injected_energy_host, dtype=np.float64
                )
            if self._d_voltage_source_work is None:
                self._d_voltage_source_work = cp.asarray(
                    self._voltage_source_work_host, dtype=np.float64
                )

        if self._maintain_voltage:
            bc_x_lo = str(solver.bc_low[0]).lower()
            bc_x_hi = str(solver.bc_high[0]).lower()
            bc_y_lo = str(solver.bc_low[1]).lower()
            bc_y_hi = str(solver.bc_high[1]).lower()
            bc_z_hi = str(solver.bc_high[2]).lower()
            if (
                bc_x_lo not in ("periodic", "bloch")
                or bc_x_hi not in ("periodic", "bloch")
                or bc_y_lo not in ("periodic", "bloch")
                or bc_y_hi not in ("periodic", "bloch")
                or bc_z_hi not in ("pec", "electric")
            ):
                raise ValueError(
                    "maintain_voltage=True models an ideal planar voltage source "
                    "and requires periodic transverse boundaries (bc_low/high[0, 1]='periodic') "
                    "and a PEC anode at zmax (bc_high[2]='pec') so a spatially uniform "
                    f"longitudinal shift dEz does not introduce curl(E); got "
                    f"bc_low={solver.bc_low!r}, bc_high={solver.bc_high!r}."
                )
            self._check_no_embedded_conductors()

        if self.voltage is not None and self._gap_d > 0.0:
            # Initialize uniform vacuum diode electric field E_z = -V / d if E_z is currently zero
            Ez_3d = self._ez_3d_view()
            Ez_view = Ez_3d[: self._Nx_nodes, : self._Ny_nodes, : self._Nz_phys]
            if not bool(self.xp.any(Ez_view != 0.0)):
                Ez_view[...] = -self.voltage / self._gap_d

        self.emitter_kind = "pec_wall"
        return self

    def state_dict(self) -> dict:
        """Return a serialisable dictionary of mutable emitter restart state.

        Returns
        -------
        dict
            Dictionary containing ``injected_charge`` [C], ``injected_energy``
            [J], ``injected_count``, ``voltage_source_work`` [J], ``_V_target``
            [V], ``step_index``, ``_next_id`` (from ``self.species._next_id``),
            and ``emitter_kind``.
        """
        return {
            "injected_charge": float(self.injected_charge),
            "injected_energy": float(self.injected_energy),
            "injected_count": int(self.injected_count),
            "voltage_source_work": float(self.voltage_source_work),
            "_V_target": (None if self._V_target is None else float(self._V_target)),
            "step_index": int(self.step_index),
            "_next_id": int(getattr(self.species, "_next_id", 0)),
            "emitter_kind": self.emitter_kind,
        }

    def load_state_dict(self, state: dict):
        """Restore mutable emitter restart state from ``state`` in-place.

        Parameters
        ----------
        state : dict
            Dictionary previously produced by :meth:`state_dict`.

        Returns
        -------
        _SurfaceEmitterBase
            ``self``.
        """
        self.injected_charge = float(state["injected_charge"])
        self.injected_energy = float(state["injected_energy"])
        self.injected_count = int(state["injected_count"])
        if "voltage_source_work" in state:
            self.voltage_source_work = float(state["voltage_source_work"])
        if "_V_target" in state:
            v_tgt = state["_V_target"]
            if v_tgt is None:
                self._V_target = None
            elif self._use_gpu:
                self._V_target = self.xp.asarray(float(v_tgt), dtype=np.float64)
            else:
                self._V_target = float(v_tgt)
        self.step_index = int(state["step_index"])
        if "_next_id" in state and hasattr(self.species, "_next_id"):
            self.species._next_id = int(state["_next_id"])
        if "emitter_kind" in state:
            self.emitter_kind = state["emitter_kind"]
        return self

    def _check_no_embedded_conductors(self) -> None:
        """Raise ``ValueError`` if STL solids or solid conductors are present in the gap."""
        solver = self.solver
        grid = solver.grid
        ieps_field = getattr(solver, "ieps", None)
        has_internal_pec = False
        if ieps_field is not None:
            ieps_z = ieps_field.to_matrix("z")[
                : self._Nx_nodes, : self._Ny_nodes, : self._Nz_phys
            ]
            has_internal_pec = bool(self.xp.any(ieps_z == 0.0))
        if (
            bool(getattr(solver, "use_stl", False))
            or bool(getattr(grid, "use_stl", False))
            or bool(getattr(grid, "stl_solids", None))
            or bool(getattr(grid, "sheets", None))
            or bool(getattr(solver, "use_conductors", False))
            or has_internal_pec
        ):
            raise ValueError(
                "maintain_voltage=True cannot be used when embedded STL solids "
                "(use_stl=True / stl_solids) or solid conductors (use_conductors=True "
                "or internal PEC edges ieps_z == 0) are present, because a spatially "
                "uniform longitudinal shift dEz would also shift fields inside "
                "embedded conductors."
            )

    @property
    def gap_d(self) -> float:
        """Global longitudinal gap separation :math:`d = z_{\\max} - z_{\\min}` [m]."""
        if self._use_mpi:
            raise NotImplementedError(
                "gap_d is local to the rank's z-slab under use_mpi=True; "
                "global gap length requires MPI domain metadata."
            )
        return self._gap_d

    def _ez_3d_view(self):
        """Return a zero-copy 3D view ``(Nx, Ny, Nz)`` of ``solver.E``'s z-component on ``xp``."""
        E_field = self.solver.E
        N = E_field.N
        return E_field.xp.reshape(
            E_field.array[2 * N : 3 * N],
            (E_field.Nx, E_field.Ny, E_field.Nz),
            order="F",
        )

    @property
    def injected_charge(self):
        """Cumulative physical charge [C] emitted into the domain."""
        if self._use_gpu:
            return _DeviceScalar(self._d_injected_charge)
        return self._injected_charge_host

    @injected_charge.setter
    def injected_charge(self, val: float) -> None:
        if self._use_gpu:
            if isinstance(val, _DeviceScalar):
                self._d_injected_charge = val._d
            else:
                self._d_injected_charge.fill(float(val))
        else:
            self._injected_charge_host = float(val)

    @property
    def injected_energy(self):
        """Cumulative kinetic energy [J] of emitted macroparticles."""
        if self._use_gpu:
            return _DeviceScalar(self._d_injected_energy)
        return self._injected_energy_host

    @injected_energy.setter
    def injected_energy(self, val: float) -> None:
        if self._use_gpu:
            if isinstance(val, _DeviceScalar):
                self._d_injected_energy = val._d
            else:
                self._d_injected_energy.fill(float(val))
        else:
            self._injected_energy_host = float(val)

    @property
    def voltage_source_work(self):
        """
        Cumulative electromagnetic work [J] performed by the ideal voltage source
        (``maintain_voltage=True``) via return current :math:`J_{\\text{ext}} = \\varepsilon_0\\,\\Delta E_z / \\Delta t`.
        """
        if self._use_gpu:
            return _DeviceScalar(self._d_voltage_source_work)
        return self._voltage_source_work_host

    @voltage_source_work.setter
    def voltage_source_work(self, val: float) -> None:
        if self._use_gpu:
            if isinstance(val, _DeviceScalar):
                self._d_voltage_source_work = val._d
            else:
                self._d_voltage_source_work.fill(float(val))
        else:
            self._voltage_source_work_host = float(val)

    def _enforce_gap_voltage(self) -> None:
        """
        Enforce fixed anode-to-cathode potential difference
        :math:`-\\int_{z_{\\min}}^{z_{\\max}} E_z\\,\\mathrm{d}z = V` across a
        planar periodic-transverse PEC diode by applying a single spatially
        uniform longitudinal field increment

        .. math::

            \\Delta E_z = -\\frac{V - V_{\\text{curr}}}{d},
            \\qquad V_{\\text{curr}} = -\\frac{1}{A_{\\text{domain}}}
            \\int_{\\Omega} E_z\\,\\mathrm{d}V.

        Because :math:`\\Delta E_z` is independent of :math:`(x, y, z)`, it adds
        identically zero :math:`\\nabla \\times \\mathbf{E}` and leaves discrete
        Gauss's law unchanged at all interior nodes
        :math:`1 \\le k \\le N_z - 1`. At the PEC boundary nodes :math:`k = 0`
        (cathode) and :math:`k = N_z` (anode), :math:`\\Delta E_z` modifies the
        boundary electric flux by
        :math:`\\pm \\varepsilon_0 \\Delta E_z \\tilde{A}_z` (representing the surface
        charge supplied by the external voltage source); accordingly,
        ``gauss_residual(solver, species)`` evaluates discrete Gauss's law on
        interior vacuum nodes (:math:`1 \\le k \\le N_z - 2` of the
        ``Nz``-sized solver array, a subset of :math:`1 \\le k \\le N_z - 1`) and
        does not check the cathode boundary node :math:`k = 0`. Physically,
        :math:`\\Delta E_z` models an ideal external voltage source supplying
        uniform return current density
        :math:`J_{\\text{ext}} = \\varepsilon_0\\,\\Delta E_z / \\Delta t` and
        delivering electrostatic work
        :math:`\\delta W_{\\text{src}} = -\\varepsilon_0 A_{\\text{domain}}\\,\\Delta E_z\\,\\tfrac{1}{2}(V + V_{\\text{curr}}) = \\Delta W_{\\text{field}}`
        per step.
        """
        if not self._maintain_voltage or self._gap_d <= 0.0:
            return
        if self._use_mpi:
            raise NotImplementedError(
                "maintain_voltage=True requires a global longitudinal reduction "
                "of E_z across MPI ranks and is not implemented under use_mpi=True."
            )
        if bool(getattr(self.solver, "use_stl", False)) or bool(
            getattr(self.solver, "use_conductors", False)
        ):
            self._check_no_embedded_conductors()
        Ez_3d = self._ez_3d_view()
        Ez_phys = Ez_3d[: self._Nx_nodes, : self._Ny_nodes, : self._Nz_phys]
        V_col = -self.xp.sum(Ez_phys * self._dz_vec[None, None, :], axis=2)
        V_curr = self.xp.sum(V_col * self._domain_area_2d) / self._total_domain_area
        if self._V_target is None:
            if self.voltage is not None:
                self._V_target = float(self.voltage)
            else:
                self._V_target = V_curr.copy() if self._use_gpu else float(V_curr)
        dEz = -(self._V_target - V_curr) / self._gap_d
        Ez_phys += dEz
        dW_src = (
            -epsilon_0 * self._total_domain_area * dEz * 0.5 * (self._V_target + V_curr)
        )
        if self._use_gpu:
            self._d_voltage_source_work += dW_src
        else:
            self._voltage_source_work_host += float(dW_src)

    def _cathode_node_charge(self):
        """
        Compute the linear Cloud-in-Cell charge :math:`q_0[i, j]` [C] currently
        assigned to cathode primal node :math:`k = 0` by active particles in the
        first longitudinal cell :math:`\\zeta_p \\in [0, 1)` using
        ``grid_map.to_logical`` and first-order B-spline nodal weights:

        .. math::

            q_0[i, j] = \\sum_{p:\\,\\zeta_p \\in [0, 1)} q_s w_p
            S_1(\\xi_p - i)\\,S_1(\\eta_p - j)\\,(1 - \\zeta_p).
        """
        psolver = getattr(self.solver, "_particle_solver", None)
        shape_order = int(
            getattr(
                psolver,
                "shape",
                getattr(self.solver, "_particles_shape_order", 1),
            )
        )
        if shape_order != 1:
            raise NotImplementedError(
                "Cathode surface emission boundary pillbox charge q_0 is "
                f"implemented only for first-order (CIC, shape=1) particle "
                f"deposition, got shape={shape_order}."
            )

        self._q0_grid.fill(0.0)
        n = self.species.n_particles
        if n == 0:
            return self._q0_grid[self._cell_i, self._cell_j]

        xp = self.xp
        x_p = self.species._x[:n]
        y_p = self.species._y[:n]
        z_p = self.species._z[:n]
        w_p = self.species._w[:n]
        xi, eta, zeta = self.grid_map.to_logical(x_p, y_p, z_p)
        in_first = (zeta >= 0.0) & (zeta < 1.0)
        q_contrib = xp.where(in_first, self.species.charge * w_p * (1.0 - zeta), 0.0)

        if self._Nx_nodes == 1 and self._Ny_nodes == 1:
            self._q0_grid[0, 0] = xp.sum(q_contrib)
        else:
            ix0, wx = bspline_weights_1d(xi, 1, staggered=False, xp=xp)
            iy0, wy = bspline_weights_1d(eta, 1, staggered=False, xp=xp)
            for ox in range(2):
                ix_m, vx_m = _wrap_or_mask_1d(
                    ix0 + ox, self._Nx_nodes, self._periodic_xy[0], xp=xp
                )
                for oy in range(2):
                    iy_m, vy_m = _wrap_or_mask_1d(
                        iy0 + oy, self._Ny_nodes, self._periodic_xy[1], xp=xp
                    )
                    w_xy = wx[ox] * wy[oy]
                    if vx_m is not None:
                        w_xy = xp.where(vx_m, w_xy, 0.0)
                    if vy_m is not None:
                        w_xy = xp.where(vy_m, w_xy, 0.0)
                    q_val = q_contrib * w_xy
                    xp.add.at(self._q0_grid, (ix_m, iy_m), q_val)

        return self._q0_grid[self._cell_i, self._cell_j]

    def _read_pulling_surface_field(self):
        """
        Read the inward-pulling normal electric field :math:`E_{\\text{pull}}`
        [V/m] on each active cathode node at integer time level :math:`t_n`.

        By Gauss's law on the boundary half-cell pillbox
        :math:`[z_{\\min}, z_{\\min} + \\tfrac{1}{2}\\Delta z_0]` around cathode
        primal node :math:`k = 0` with FIT dual face area
        :math:`\\tilde{A}_z[i, j] = 1 / \\mathrm{itA}_z[i, j, 0]`:

        .. math::

            E_z(z_{\\min}^+) = E_z[i, j, k_{\\text{read}}]
            - \\frac{q_0[i, j]}{\\varepsilon_0 \\tilde{A}_z[i, j]}

        where :math:`k_{\\text{read}} = 0` is the first primary longitudinal edge
        midpoint :math:`z_{\\min} + \\tfrac{1}{2}\\Delta z_0`. For a species of
        charge :math:`q_s`, the field pulling particles into the domain
        (:math:`+z` direction) is :math:`-\\operatorname{sgn}(q_s) E_z(z_{\\min}^+)`.
        """
        self._enforce_gap_voltage()
        Ez_3d = self._ez_3d_view()
        Ez_grid = Ez_3d[:, :, self._k_read]
        Ez_cell = Ez_grid[self._cell_i, self._cell_j].astype(np.float64, copy=False)
        q0_cell = self._cathode_node_charge()
        Ez_surf = Ez_cell - q0_cell / (epsilon_0 * self._cell_areas)

        q_sign = 1.0 if self.species.charge >= 0.0 else -1.0
        E_normal = q_sign * Ez_surf
        return self.xp.maximum(E_normal, 0.0)

    def _compute_emitted_charge(self, E_pull, dt: float):
        """Return positive physical charge magnitude ``|Q_cell|`` [C] per active node."""
        raise NotImplementedError

    def inject(
        self,
        t: float,
        dt: float,
        step_index: int | None = None,
        *,
        owning: bool = True,
    ) -> int:
        """
        Emit macroparticles from the cathode surface for step ``[t, t + dt]``.

        Parameters
        ----------
        t : float
            Current integer time level :math:`t_n` [s].
        dt : float
            Time step :math:`\\Delta t` [s].
        step_index : int or None, optional
            Solver particle step index ``n_particle_steps``. Defaults to
            ``self.step_index``.
        owning : bool, optional
            If False (non-owning MPI rank), advances ``self.step_index`` (and
            ``self.species._next_id`` when ``track_ids=True``) without
            appending local particles.

        Returns
        -------
        int
            Number of macroparticles appended to ``self.species`` (``0`` on a
            non-owning MPI rank or on CPU when every emitted weight is zero, or
            ``n_emitting_cells * n_macro_per_cell`` when active emission occurs
            or on GPU to avoid a host synchronization).
        """
        step_idx = int(self.step_index if step_index is None else step_index)
        self.step_index = step_idx + 1

        if t < self.t_start or t >= self.t_end or self.n_emit_per_step == 0:
            return 0

        if not owning:
            if self.species.track_ids:
                self.species._next_id += self.n_emit_per_step
            return 0

        xp = self.xp
        E_pull = self._read_pulling_surface_field()
        q_cell = xp.where(
            E_pull > 0.0,
            xp.maximum(self._compute_emitted_charge(E_pull, float(dt)), 0.0),
            0.0,
        )

        q_mag = abs(self.species.charge)
        w_cell = q_cell / (q_mag * float(self.n_macro_per_cell))
        if self.n_macro_per_cell == 1:
            w_emit = w_cell.astype(self.species.dtype, copy=False)
        else:
            w_emit = xp.repeat(w_cell, self.n_macro_per_cell).astype(
                self.species.dtype, copy=False
            )

        # On CPU, skip appending macroparticles when every weight is zero
        # (e.g. when E == 0 or the entire cathode is retarding) to prevent
        # zero-weight particle pile-up at z = zmin. The skipped slots still
        # consume their ids, as on non-owning ranks and on the GPU (which
        # always appends), so serial and MPI runs share one id stream.
        if not self._use_gpu and not bool(np.any(w_emit > 0.0)):
            if self.species.track_ids:
                self.species._next_id += self.n_emit_per_step
            return 0

        self.species.add_particles(
            self._x_emit,
            self._y_emit,
            self._z_emit,
            ux=self._ux_emit,
            uy=self._uy_emit,
            uz=self._uz_emit,
            w=w_emit,
        )
        self.injected_count += self.n_emit_per_step

        # Update cumulative charge and kinetic energy without host synchronization
        w_sum = xp.sum(w_emit)
        dq = self.species.charge * w_sum
        if self.thermal_u > 0.0:
            gamma0 = float(np.sqrt(1.0 + (self.thermal_u / c_light) ** 2))
            ke_single = self.species.mass * (self.thermal_u**2) / (gamma0 + 1.0)
        else:
            ke_single = 0.0
        dE = ke_single * w_sum

        if self._use_gpu:
            self._d_injected_charge += dq
            self._d_injected_energy += dE
        else:
            self._injected_charge_host += float(dq)
            self._injected_energy_host += float(dE)

        return self.n_emit_per_step


class SpaceChargeLimitedEmitter(_SurfaceEmitterBase):
    """
    Gauss-law space-charge-limited surface emitter on ``z0 = grid.zmin``.

    At each step, emits into each active transverse cathode dual cell :math:`(i, j)`
    the charge that drives the normal electric field immediately outside the
    PEC cathode surface to zero:

    .. math::

        |q_{\\text{cell}}[i, j]| = \\begin{cases}
        \\varepsilon_0 E_n[i, j] \\tilde{A}_z[i, j], & E_n[i, j] > 0, \\\\
        0, & E_n[i, j] \\le 0,
        \\end{cases}

    where :math:`E_n[i, j] = -\\operatorname{sgn}(q_s)\\,E_z(x_i, y_j, z_{\\min}^+)`
    is the pulling surface normal electric field reconstructed from the first
    cell's FIT edge field :math:`E_z[i, j, 0]` and the cathode node charge
    :math:`q_0[i, j]` via the boundary half-cell Gauss pillbox with FIT dual
    face area :math:`\\tilde{A}_z[i, j] = 1 / \\mathrm{itA}_z[i, j, 0]`.

    Because :math:`E_z(z_{\\min}^+) \\to 0` is enforced at every step, the 1D
    steady state across a planar gap :math:`d` at voltage :math:`V` converges to
    the analytical Child–Langmuir space-charge-limited current density
    :math:`J_{\\text{CL}} = (4\\varepsilon_0 / 9)\\sqrt{2|q_s|/m_s}\\,V^{3/2}/d^2`
    with potential profile :math:`\\phi(z) = V (z / d)^{4/3}` and space-charge
    density :math:`\\rho(z) \\propto z^{-2/3}` with two-term **spatial** cathode
    boundary-layer error
    :math:`e(\\Delta z) = a(\\Delta z / d)^{2/3} + b(\\Delta z / d)`.
    Beyond the first cell (:math:`z \\ge \\Delta z`), Child–Langmuir flow with
    zero entry field and entry energy :math:`|q_s| V_0 = \\frac{1}{2} m_s v_1^2`
    satisfies
    :math:`{\\phi'}^2 = (4J/\\varepsilon_0)\\sqrt{m_s/(2|q_s|)}(\\tilde{\\phi}^{1/2} - V_0^{1/2})`,
    giving :math:`J / J_{\\text{CL}} \\approx 1 + 3\\sqrt{V_0 / V} = 1 + 3\\,v_1 / v_{\\max}`
    with :math:`v_1 / v_{\\max} = (\\Delta z / d)^{2/3}`; because
    :math:`E_z \\propto z^{1/3}` near the cathode, the gathered field across the
    first cell normalized by :math:`E_z(\\Delta z)` (:math:`(\\frac{1}{2} + \\zeta)E_z[0]`
    on :math:`[0, 1/2)` then linear to :math:`E_z[1]` with
    :math:`E_z[0]/E_z(\\Delta z) = 2^{-1/3}` and
    :math:`E_z[1]/E_z(\\Delta z) = (3/2)^{1/3}`) has a :math:`\\Delta z`-independent
    shape, so :math:`\\delta v_1 / v_1` is constant across resolutions and
    :math:`e_{2/3} = 3(\\delta v_1 / v_1)(\\Delta z / d)^{2/3}` (equivalently
    :math:`(E_c d / V)^2 \\propto (\\Delta z / d)^{2/3}` from a residual first-cell
    field error :math:`E_c \\propto E_z(\\Delta z) \\propto (\\Delta z / d)^{1/3} V / d`;
    the order is derived while the sign and coefficient :math:`a` remain fitted
    because feeding the exact Child–Langmuir field through the gather gives a
    velocity deficit whereas the self-consistent bias is positive, with fitted
    :math:`a \\approx 0.17` corresponding to :math:`\\delta v_1 / v_1 \\approx 6\\%`).

    References: C. D. Child, Phys. Rev. (Series I) 32, 492 (1911),
    doi:10.1103/PhysRevSeriesI.32.492; I. Langmuir, Phys. Rev. 2, 450 (1913),
    doi:10.1103/PhysRev.2.450; C. K. Birdsall and A. B. Langdon, *Plasma
    Physics via Computer Simulation* (Taylor & Francis, 1991), Ch. 16;
    J. J. Watrous, J. W. Luginsland, and G. E. Sasser III, Phys. Plasmas 8, 289
    (2001), doi:10.1063/1.1329152.

    Parameters
    ----------
    solver : SolverFIT3D
        Electromagnetic FIT solver providing the grid and electric field ``E``.
    species : Species
        Target macroparticle species (e.g. electrons with ``charge = -e``).
    n_macro_per_cell : int, optional
        Number of macroparticles injected per active transverse node per step.
        Default is ``1``.
    thermal_u : float, optional
        Initial longitudinal proper velocity :math:`u_z = \\gamma v_z \\ge 0`
        [m/s]. Default is ``0.0`` (cold emission from rest).
    x_bounds, y_bounds : tuple of (float, float) or None, optional
        Optional rectangular emission patch bounds ``(xmin, xmax)`` and
        ``(ymin, ymax)`` [m].
    radius : float or None, optional
        Optional circular disk emission patch radius [m] around ``center``.
    center : tuple of (float, float), optional
        Transverse center ``(x0, y0)`` [m] for a circular disk patch.
        Default is ``(0.0, 0.0)``.
    voltage : float or None, optional
        Optional anode-to-cathode gap voltage :math:`V` [V] across
        ``[zmin, zmax]``. When provided and ``solver.E`` is zero, initializes
        :math:`E_z = -V / d`.
    maintain_voltage : bool, optional
        If True, clamp the gap voltage to :math:`V` at each step via an ideal
        external voltage source with return current
        :math:`J_{\\text{ext}} = \\varepsilon_0\\,\\Delta E_z / \\Delta t` and
        cumulative work tracked in ``voltage_source_work``. Default is ``False``.
    t_start, t_end : float, optional
        Active emission time window ``[t_start, t_end)`` [s].
    """

    def _compute_emitted_charge(self, E_pull, dt: float):
        return epsilon_0 * E_pull * self._cell_areas


class ThermionicEmitter(_SurfaceEmitterBase):
    """
    Richardson–Dushman thermionic surface emitter with Schottky barrier
    lowering on ``z0 = grid.zmin``.

    Emits current density

    .. math::

        J_{\\text{RD}}(F, T) = \\lambda_R A_0 T^2
        \\exp\\!\\left(-\\frac{W - \\Delta W_S(F)}{k_B T}\\right),
        \\qquad \\Delta W_S(F) = \\sqrt{\\frac{e^3 F}{4\\pi\\varepsilon_0}}

    into each active cathode node where the pulling normal electric field is
    :math:`F = E_{\\text{pull}} > 0`, and emits **zero** where
    :math:`F = E_{\\text{pull}} \\le 0` (so thermal emission against a retarding
    field and virtual-cathode potential-minimum formation :math:`\\phi_{\\min} < 0`
    are not modelled). When ``space_charge_limit=True``, the emitted charge per
    node per step is capped by the Gauss-law charge
    :math:`\\varepsilon_0 F \\tilde{A}_z`.

    Parameters
    ----------
    solver : SolverFIT3D
        Electromagnetic FIT solver providing the grid and electric field ``E``.
    species : Species
        Target macroparticle species.
    temperature : float
        Cathode absolute temperature :math:`T > 0` [K].
    work_function_eV : float
        Material zero-field work function :math:`W / e > 0` [eV].
    lambda_R : float, optional
        Material-specific Richardson constant correction factor
        :math:`\\lambda_R > 0` [-]. Default is ``1.0``.
    space_charge_limit : bool, optional
        If True, cap emitted charge per cell per step by the Gauss-law space-charge
        limit :math:`\\varepsilon_0 E_n A_{\\text{cell}}`. Default is ``False``.
    n_macro_per_cell : int, optional
        Number of macroparticles injected per active cell per step. Default is ``1``.
    thermal_u : float, optional
        Initial longitudinal proper velocity :math:`u_z \\ge 0` [m/s].
        Default is ``0.0``.
    x_bounds, y_bounds : tuple of (float, float) or None, optional
        Optional rectangular patch bounds [m].
    radius : float or None, optional
        Optional disk patch radius [m].
    center : tuple of (float, float), optional
        Disk patch center ``(x0, y0)`` [m]. Default is ``(0.0, 0.0)``.
    voltage : float or None, optional
        Optional gap voltage :math:`V` [V].
    maintain_voltage : bool, optional
        Whether to clamp fixed gap voltage across ``[zmin, zmax]`` via an ideal
        voltage source. Default is ``False``.
    t_start, t_end : float, optional
        Active emission time window ``[t_start, t_end)`` [s].
    """

    def __init__(
        self,
        solver,
        species: Species,
        temperature: float,
        work_function_eV: float,
        *,
        lambda_R: float = 1.0,
        space_charge_limit: bool = False,
        n_macro_per_cell: int = 1,
        thermal_u: float = 0.0,
        x_bounds: tuple[float, float] | None = None,
        y_bounds: tuple[float, float] | None = None,
        radius: float | None = None,
        center: tuple[float, float] = (0.0, 0.0),
        voltage: float | None = None,
        maintain_voltage: bool = False,
        t_start: float = 0.0,
        t_end: float = float("inf"),
    ):
        if float(temperature) <= 0.0:
            raise ValueError(f"temperature must be > 0 K, got {temperature}.")
        if float(work_function_eV) <= 0.0:
            raise ValueError(
                f"work_function_eV must be > 0 eV, got {work_function_eV}."
            )
        if float(lambda_R) <= 0.0:
            raise ValueError(f"lambda_R must be > 0, got {lambda_R}.")

        self.temperature = float(temperature)
        self.work_function_eV = float(work_function_eV)
        self.lambda_R = float(lambda_R)
        self.space_charge_limit = bool(space_charge_limit)

        super().__init__(
            solver,
            species,
            n_macro_per_cell=n_macro_per_cell,
            thermal_u=thermal_u,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            radius=radius,
            center=center,
            voltage=voltage,
            maintain_voltage=maintain_voltage,
            t_start=t_start,
            t_end=t_end,
        )

    def current_density(self, E_field):
        """Evaluate Richardson–Dushman current density [A/m^2] at field ``E_field`` [V/m]."""
        return richardson_dushman_current_density(
            E_field,
            temperature=self.temperature,
            work_function_eV=self.work_function_eV,
            lambda_R=self.lambda_R,
        )

    def _compute_emitted_charge(self, E_pull, dt: float):
        J_rd = self.current_density(E_pull)
        q_rd = J_rd * self._cell_areas * float(dt)
        if self.space_charge_limit:
            q_gauss = epsilon_0 * E_pull * self._cell_areas
            return self.xp.minimum(q_rd, q_gauss)
        return q_rd


class FowlerNordheimEmitter(_SurfaceEmitterBase):
    """
    Murphy–Good cold field-emission surface emitter on ``z0 = grid.zmin`` using
    the Forbes (2006) approximations for :math:`v(f)` and :math:`t(f)`.

    With local barrier field :math:`F_{\\text{loc}} = \\beta_{\\text{FE}} E_{\\text{pull}}`
    and scaled barrier field
    :math:`f = (e^3 / (4\\pi\\varepsilon_0)) F_{\\text{loc}} / \\phi^2`:

    .. math::

        J_{\\text{FN}}(F) = \\frac{a_{\\text{FN}}\\,F_{\\text{loc}}^2}
        {\\phi_{\\text{eV}}\\,t(f)^2}
        \\exp\\!\\left(-\\frac{b_{\\text{FN}}\\,\\phi_{\\text{eV}}^{3/2}}
        {F_{\\text{loc}}}\\,v(f)\\right),

    .. math::

        v(f) = 1 - f + \\frac{1}{6}\\,f \\ln f, \\qquad
        t(f) = 1 + \\frac{f}{9} - \\frac{1}{18}\\,f \\ln f.

    When ``space_charge_limit=True``, the emitted charge per cell per step is
    capped by the Gauss-law charge :math:`\\varepsilon_0 E_{\\text{pull}} A_{\\text{cell}}`.

    Parameters
    ----------
    solver : SolverFIT3D
        Electromagnetic FIT solver providing the grid and electric field ``E``.
    species : Species
        Target macroparticle species.
    work_function_eV : float
        Material work function :math:`\\phi_{\\text{eV}} > 0` [eV].
    beta_fe : float, optional
        Field-enhancement factor :math:`\\beta_{\\text{FE}} > 0` [-].
        Default is ``1.0``.
    space_charge_limit : bool, optional
        If True, cap emitted charge per cell per step by the Gauss-law space-charge
        limit :math:`\\varepsilon_0 E_n A_{\\text{cell}}`. Default is ``False``.
    n_macro_per_cell : int, optional
        Number of macroparticles injected per active cell per step. Default is ``1``.
    thermal_u : float, optional
        Initial longitudinal proper velocity :math:`u_z \\ge 0` [m/s].
        Default is ``0.0``.
    x_bounds, y_bounds : tuple of (float, float) or None, optional
        Optional rectangular patch bounds [m].
    radius : float or None, optional
        Optional disk patch radius [m].
    center : tuple of (float, float), optional
        Disk patch center ``(x0, y0)`` [m]. Default is ``(0.0, 0.0)``.
    voltage : float or None, optional
        Optional gap voltage :math:`V` [V].
    maintain_voltage : bool, optional
        Whether to clamp fixed gap voltage across ``[zmin, zmax]`` via an ideal
        voltage source. Default is ``False``.
    t_start, t_end : float, optional
        Active emission time window ``[t_start, t_end)`` [s].
    """

    def __init__(
        self,
        solver,
        species: Species,
        work_function_eV: float,
        *,
        beta_fe: float = 1.0,
        space_charge_limit: bool = False,
        n_macro_per_cell: int = 1,
        thermal_u: float = 0.0,
        x_bounds: tuple[float, float] | None = None,
        y_bounds: tuple[float, float] | None = None,
        radius: float | None = None,
        center: tuple[float, float] = (0.0, 0.0),
        voltage: float | None = None,
        maintain_voltage: bool = False,
        t_start: float = 0.0,
        t_end: float = float("inf"),
    ):
        if float(work_function_eV) <= 0.0:
            raise ValueError(
                f"work_function_eV must be > 0 eV, got {work_function_eV}."
            )
        if float(beta_fe) <= 0.0:
            raise ValueError(f"beta_fe must be > 0, got {beta_fe}.")

        self.work_function_eV = float(work_function_eV)
        self.beta_fe = float(beta_fe)
        self.space_charge_limit = bool(space_charge_limit)

        super().__init__(
            solver,
            species,
            n_macro_per_cell=n_macro_per_cell,
            thermal_u=thermal_u,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            radius=radius,
            center=center,
            voltage=voltage,
            maintain_voltage=maintain_voltage,
            t_start=t_start,
            t_end=t_end,
        )

    def current_density(self, E_field):
        """Evaluate Murphy–Good / Forbes (2006) current density [A/m^2] at ``E_field`` [V/m]."""
        return fowler_nordheim_current_density(
            E_field,
            work_function_eV=self.work_function_eV,
            beta_fe=self.beta_fe,
        )

    def _compute_emitted_charge(self, E_pull, dt: float):
        J_fn = self.current_density(E_pull)
        q_fn = J_fn * self._cell_areas * float(dt)
        if self.space_charge_limit:
            q_gauss = epsilon_0 * E_pull * self._cell_areas
            return self.xp.minimum(q_fn, q_gauss)
        return q_fn
