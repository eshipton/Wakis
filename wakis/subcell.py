# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Subcell models for thin conducting sheets and thin wires in ``SolverFIT3D``.

Thin sheets
-----------
A zero-thickness sheet on an axis-aligned primary grid plane :math:`w = w_{k_0}`
(normal axis :math:`w \\in \\{x, y, z\\}`) carries the tangential surface
current density

.. math::

    \\mathbf{J}_s(\\omega) = \\sigma_s(\\omega)\\, \\mathbf{E}_t(\\omega)
    \\quad [\\text{A/m}].

In the Finite Integration Technique (FIT), integral Ampère's law on the dual
facet :math:`\\tilde{A}_{u,e} = \\tilde{L}_{v,e}\\tilde{L}_{w,e}` pierced by a
tangential primary edge :math:`e` along :math:`u` reads

.. math::

    \\oint_{\\partial\\tilde{A}_{u,e}} \\mathbf{H}\\cdot d\\tilde{\\mathbf{s}}
    = \\frac{d}{dt} \\iint_{\\tilde{A}_{u,e}} \\varepsilon E_u\\, d\\tilde{A}
    + \\int_{\\tilde{A}_{u,e} \\cap \\text{sheet}} J_{s,u}\\, d\\tilde{\\ell}_v.

The sheet current crossing :math:`\\tilde{A}_{u,e}` is
:math:`I_{s,e} = J_{s,u}\\,(f_e\\tilde{L}_{v,e})`, where :math:`\\tilde{L}_{v,e}`
is the in-plane dual-edge length crossing :math:`e` and :math:`f_e \\in [0, 1]`
is the in-plane area fraction of the sheet around :math:`e`. Because the
discrete :math:`\\mathbf{E}` update divides the dual curl by the dual facet area
:math:`\\tilde{A}_{u,e} = (\\text{itA}_{u,e})^{-1}`, the sheet adds an effective
current density

.. math::

    J_{\\text{eff},e}
    = \\frac{I_{s,e}}{\\tilde{A}_{u,e}}
    = \\sigma_s E_{u,e}\\,\\left(f_e\\,\\tilde{L}_{v,e}\\,\\text{itA}_{u,e}\\right)
    = \\sigma_s E_{u,e}\\,\\frac{f_e}{\\tilde{L}_{w,e}},

where :math:`\\tilde{L}_{w,e}` is the normal dual-edge length crossing the
sheet. For a full-plane sheet (:math:`f_e = 1`) on a uniform grid, this matches
the volume-conductivity ratio :math:`\\sigma_s / \\Delta w` **only** on the
in-plane tangential edges; a naive cell-based volume assignment fails by
polluting the normal electric field :math:`E_w`, smearing tangential currents
over two grid planes, and using the primary cell width :math:`\\Delta w_k`
instead of the dual length :math:`\\tilde{L}_{w,k} = (\\Delta w_{k-1} + \\Delta w_k)/2`
on non-uniform meshes or domain boundaries.

Time integration uses the semi-implicit Crank-Nicolson / auxiliary differential
equation (ADE) update of :mod:`wakis.dispersive`, which is unconditionally
stable for any passive :math:`\\sigma_s \\ge 0`, supports Drude/Lorentz/Debye
surface dispersion (graphene intraband conductivity and multi-pole skin-effect
surface impedance), and is linear and exactly transposed by
:class:`~wakis.inverse.AdjointStepper`.

Thin wires
----------
An axis-aligned wire of subcell radius :math:`a \\ll \\Delta` along primary
edges is modelled by the contour-path / in-cell inductance formulation of
Umashankar, Beker & Taflove [1, 2] expressed in symmetric FIT Hodge form [3]:

* On the wire primary edges :math:`e_w(i_0, j_0, k)`, :math:`E_w = 0`
  (:math:`\\varepsilon^{-1} = 0`) for a PEC wire, or a semi-implicit conductive
  update with conductance per unit length :math:`(\\pi a^2 \\sigma_w)` for a
  lossy wire, except on driven/loaded gap segments where a
  :class:`~wakis.lumped.LumpedPort` or :class:`~wakis.lumped.LumpedElement` is
  attached.
* Assuming the quasi-static :math:`1/r` radial variation of azimuthal
  :math:`H_\\phi` and radial :math:`E_r` for :math:`r \\in [a, \\Delta]`, the
  magnetic flux through the four primary faces circulating around each wire
  segment is scaled by :math:`m = \\frac{1}{2}\\ln(\\Delta / a)`, so the dual-edge
  inverse permeability Hodge entries are divided by :math:`m`
  (:math:`\\mu_{\\text{eff}} = m\\mu`).
* Dual-cell charge conservation on the radial edges attached to a wire node of
  degree :math:`d` (where :math:`d = 2` at interior and bend nodes, :math:`d = 1`
  at wire ends, and :math:`d \\ge 3` at junctions) scales the inverse
  permittivity on the non-wire incident edges by :math:`w_{\\text{node}} = (2/d)m`
  (:math:`\\varepsilon_{\\text{eff}} = \\varepsilon / w_{\\text{node}}`), with
  paired permeability scaling on the adjacent faces so the Courant limit is
  preserved at the standard Yee timestep.

References
----------
[1] K. R. Umashankar, B. Beker, and A. Taflove, "Calculation and experimental
    validation of induced currents on coupled wires in an arbitrary shaped
    cavity," *IEEE Trans. Antennas Propag.*, vol. 35, no. 11, pp. 1248-1257,
    1987.
[2] A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
    Finite-Difference Time-Domain Method*, 3rd ed., Artech House, 2005, ch. 10.
[3] T. Noda and S. Yokoyama, "Thin wire representation in finite difference
    time domain surge simulation," *IEEE Trans. Power Del.*, vol. 17, no. 3,
    pp. 840-846, 2002.
[4] G. W. Hanson, "Dyadic Green's functions and guided surface waves for a
    surface conductivity model of graphene," *J. Appl. Phys.*, vol. 103,
    064302, 2008.
"""

import pathlib
import warnings

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import e as q_e
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import hbar
from scipy.constants import k as k_B
from scipy.constants import mu_0
from scipy.optimize import lsq_linear
from scipy.sparse import diags

from .dispersive import (
    Debye,
    DispersiveMaterial,
    Drude,
    Lorentz,
    _ade_coefficients,
    _restrict_group,
)


def _host(arr):
    """Return a host NumPy array from a NumPy or CuPy array."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)

ETA0 = np.sqrt(mu_0 / eps_0)
_AXES = ("x", "y", "z")
_PLANE_TO_NORMAL = {
    "x": 0,
    "y": 1,
    "z": 2,
    "yz": 0,
    "zy": 0,
    "xz": 1,
    "zx": 1,
    "xy": 2,
    "yx": 2,
}
_INPLANE_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def _scale_operator_rows(mat, scale, xp):
    """Multiply row ``r`` of sparse matrix ``mat`` in-place by ``scale[r]``."""
    if mat is None:
        return
    if getattr(mat, "format", "csr") == "csc":
        mat.data *= scale[mat.indices]
    elif xp is np:
        mat.data *= np.repeat(scale, np.diff(mat.indptr))
    else:
        rows = xp.searchsorted(mat.indptr, xp.arange(mat.nnz), side="right") - 1
        mat.data *= scale[rows]


def _scale_pole(pole, factor):
    """Scale susceptibility of a `Drude`, `Lorentz`, or `Debye` pole by `factor` >= 0."""
    factor = float(factor)
    if isinstance(pole, Drude):
        return Drude(omega_p=pole.omega_p * np.sqrt(factor), gamma=pole.gamma)
    if isinstance(pole, Lorentz):
        return Lorentz(
            delta_eps=pole.delta_eps * factor, omega_0=pole.omega_0, gamma=pole.gamma
        )
    if isinstance(pole, Debye):
        return Debye(delta_eps=pole.delta_eps * factor, tau=pole.tau)
    raise TypeError(f"Unsupported pole type: {type(pole)!r}.")


class SheetMaterial(DispersiveMaterial):
    """
    Surface-conductivity / sheet-impedance material model for a zero-thickness sheet.

    Stores the sheet parameters as a :class:`~wakis.dispersive.DispersiveMaterial`
    with ``eps_inf = 1 + chi_s_inf`` (where ``chi_s_inf`` [m] is optional
    high-frequency sheet susceptibility, default 0), ``sigma`` equal to the
    instantaneous sheet conductivity :math:`\\sigma_{s,\\infty}` [S], and poles
    representing the dispersive surface susceptibility
    :math:`\\chi_s(\\omega) = \\sigma_{s,\\text{disp}}(\\omega) / (j\\omega\\varepsilon_0)` [m].

    Sign convention
    ---------------
    Wakis uses the engineering phasor convention :math:`\\exp(+j\\omega t)`
    (:math:`\\partial_t \\leftrightarrow +j\\omega`), in which a passive Drude
    sheet with relaxation time :math:`\\tau = 1/\\gamma` has

    .. math::

        \\sigma_s^{(+)}(\\omega)
        = \\frac{\\sigma_{s,0}}{1 + j\\omega\\tau}
        = \\frac{-j\\,e^2 E_F}{\\pi\\hbar^2(\\omega - j/\\tau)}.

    In the physics convention :math:`\\exp(-i\\omega t)`
    (:math:`\\partial_t \\leftrightarrow -i\\omega`), the same causal response is
    written as the complex conjugate

    .. math::

        \\sigma_s^{(-)}(\\omega)
        = \\frac{\\sigma_{s,0}}{1 - i\\omega\\tau}
        = \\frac{i\\,e^2 E_F}{\\pi\\hbar^2(\\omega + i/\\tau)}
        = \\left[\\sigma_s^{(+)}(\\omega)\\right]^*.

    Both conventions have :math:`\\operatorname{Re}\\sigma_s(\\omega) \\ge 0` for
    passive media.
    """

    def __init__(
        self,
        poles=(),
        sigma_s=0.0,
        chi_s_inf=0.0,
        name="",
        meta=None,
    ):
        super().__init__(
            eps_inf=1.0 + float(chi_s_inf),
            poles=poles,
            sigma=float(sigma_s),
            name=name,
        )
        self.meta = {} if meta is None else dict(meta)

    def surface_conductivity(self, f, convention="+"):
        """
        Complex surface conductivity :math:`\\sigma_s(f)` [S].

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz] (must be non-negative).
        convention : {'+', '-'}, optional
            Phasor time-dependence convention: ``'+'`` for :math:`\\exp(+j\\omega t)`
            (Wakis default) or ``'-'`` for :math:`\\exp(-i\\omega t)` (physics).

        Returns
        -------
        complex or ndarray of complex
            Surface conductivity :math:`\\sigma_s(f)` [S].
        """
        scalar = np.ndim(f) == 0
        f_arr = np.atleast_1d(np.asarray(f, dtype=float))
        omega = 2.0 * np.pi * f_arr
        sig = np.full(f_arr.shape, self.sigma, dtype=complex)
        if self.eps_inf != 1.0:
            sig = sig + 1j * omega * eps_0 * (self.eps_inf - 1.0)
        for p in self.poles:
            if isinstance(p, Drude):
                # chi(w) = -wp^2 / (w^2 - j gamma w) => j w eps_0 chi(w) = eps_0 wp^2 / (gamma + j w)
                sig = sig + (eps_0 * p.omega_p**2) / (p.gamma + 1j * omega)
            else:
                sig = sig + 1j * omega * eps_0 * p.susceptibility(omega)
        if convention in ("-", "exp(-iwt)", "physics"):
            sig = np.conj(sig)
        elif convention not in ("+", "exp(+jwt)", "wakis"):
            raise ValueError(f"convention must be '+' or '-', got {convention!r}.")
        return sig[0] if scalar else sig

    def sheet_impedance(self, f, convention="+"):
        """
        Complex sheet impedance :math:`Z_s(f) = 1 / \\sigma_s(f)` [:math:`\\Omega`/sq].
        """
        return 1.0 / self.surface_conductivity(f, convention=convention)

    def scaled(self, factor, name=None):
        """Return a copy of this :class:`SheetMaterial` with all conductivities and poles scaled by `factor`."""
        factor = float(factor)
        return SheetMaterial(
            poles=[_scale_pole(p, factor) for p in self.poles],
            sigma_s=self.sigma * factor,
            chi_s_inf=(self.eps_inf - 1.0) * factor,
            name=self.name if name is None else name,
            meta=self.meta,
        )


def resistive_sheet(sigma_s=None, *, R_s=None, name="resistive_sheet"):
    """
    Create a frequency-independent resistive sheet with :math:`\\sigma_s = 1 / R_s`.

    Parameters
    ----------
    sigma_s : float, optional
        Sheet conductivity [S]. Mutually exclusive with `R_s`.
    R_s : float, optional
        Sheet resistance [:math:`\\Omega`/sq], with :math:`\\sigma_s = 1 / R_s`.
    name : str, optional
        Material label.

    Returns
    -------
    SheetMaterial
    """
    if (sigma_s is None) == (R_s is None):
        raise ValueError("Provide exactly one of `sigma_s` [S] or `R_s` [Ohm/sq].")
    if R_s is not None:
        if R_s < 0:
            raise ValueError(f"R_s must be >= 0, got {R_s}.")
        sigma_s = np.inf if R_s == 0 else 1.0 / float(R_s)
    if sigma_s < 0:
        raise ValueError(f"sigma_s must be >= 0, got {sigma_s}.")
    return SheetMaterial(
        poles=(),
        sigma_s=float(sigma_s),
        name=name,
        meta={"model": "resistive", "R_s": (0.0 if np.isinf(sigma_s) else 1.0 / sigma_s if sigma_s > 0 else np.inf)},
    )


def graphene_sheet(E_F=0.2, tau=0.5e-12, T=0.0, name="graphene"):
    """
    Intraband Drude surface-conductivity model of a graphene sheet [4].

    In the :math:`\\exp(-i\\omega t)` physics convention,

    .. math::

        \\sigma_s^{(-)}(\\omega)
        = \\frac{i\\,e^2 E_{F,\\text{eff}}}{\\pi\\hbar^2(\\omega + i/\\tau)},

    and in Wakis's :math:`\\exp(+j\\omega t)` convention,

    .. math::

        \\sigma_s^{(+)}(\\omega)
        = \\frac{-j\\,e^2 E_{F,\\text{eff}}}{\\pi\\hbar^2(\\omega - j/\\tau)}
        = \\frac{\\sigma_{s,0}}{1 + j\\omega\\tau},
        \\qquad
        \\sigma_{s,0} = \\frac{e^2 E_{F,\\text{eff}}\\,\\tau}{\\pi\\hbar^2},

    with :math:`E_{F,\\text{eff}} = |E_F|` at :math:`T = 0` and
    :math:`E_{F,\\text{eff}} = 2 k_B T \\ln(2\\cosh(|E_F|/(2 k_B T)))` at finite
    temperature :math:`T > 0`.

    Parameters
    ----------
    E_F : float, optional
        Fermi energy (chemical potential :math:`\\mu_c`) in eV if :math:`|E_F| > 10^{-12}`
        (or in Joules if :math:`|E_F| \\le 10^{-12}`). Default is 0.2 eV.
    tau : float, optional
        Momentum relaxation time :math:`\\tau` [s]. Default is 0.5 ps.
    T : float, optional
        Temperature [K]. Default is 0 (zero-temperature limit).
    name : str, optional
        Material label.

    Returns
    -------
    SheetMaterial
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}.")
    if T < 0:
        raise ValueError(f"T must be >= 0, got {T}.")
    ef_val = abs(float(E_F))
    ef_joule = ef_val * q_e if ef_val > 1e-12 else ef_val
    if T > 0:
        x = ef_joule / (2.0 * k_B * T)
        # Stable evaluation of 2 k_B T ln(2 cosh(x)) = 2 k_B T (x + ln(1 + exp(-2x)))
        ef_eff = 2.0 * k_B * T * (x + np.log1p(np.exp(-2.0 * x)))
    else:
        ef_eff = ef_joule
    gamma = 1.0 / float(tau)
    omega_p = np.sqrt(q_e**2 * ef_eff / (np.pi * hbar**2 * eps_0))
    sigma_dc = eps_0 * omega_p**2 * tau
    return SheetMaterial(
        poles=[Drude(omega_p=omega_p, gamma=gamma)],
        sigma_s=0.0,
        name=name,
        meta={
            "model": "graphene_drude",
            "E_F_J": ef_joule,
            "E_F_eV": ef_joule / q_e,
            "tau": float(tau),
            "T": float(T),
            "sigma_dc": float(sigma_dc),
        },
    )


def metal_sheet(
    sigma,
    thickness=None,
    *,
    mu_r=1.0,
    mode="auto",
    f0=None,
    f_range=None,
    n_poles=3,
    sides=1,
    name="metal_sheet",
):
    """
    Thin metal trace or skin-effect surface-impedance sheet model.

    Supports both physical regimes of a conducting film of bulk conductivity
    :math:`\\sigma` [S/m], permeability :math:`\\mu = \\mu_r\\mu_0`, and
    thickness :math:`t` [m] relative to the skin depth
    :math:`\\delta(\\omega) = \\sqrt{2 / (\\omega\\mu\\sigma)}`:

    * **Thin limit** (:math:`t \\ll \\delta`, ``mode='thin'``): uniform current
      across the thickness gives a constant sheet conductivity
      :math:`\\sigma_s = \\sigma t` (sheet resistance :math:`R_s = 1/(\\sigma t)`).
    * **Thick / skin-effect regime** (:math:`t \\gg \\delta`, ``mode='sibc'``):
      the Leontovich surface impedance is

      .. math::

          Z_s(\\omega) = \\frac{1 + j}{\\sigma\\,\\delta(\\omega)}
          = (1 + j)\\sqrt{\\frac{\\omega\\mu}{2\\sigma}}
          \\quad (\\exp(+j\\omega t)\\text{ convention}),

      corresponding to :math:`\\sigma_s(\\omega) = \\text{sides} / Z_s(\\omega)
      \\propto (j\\omega)^{-1/2}`. Because :math:`\\sqrt{\\omega}` is not a single
      rational pole, two passive ADE realizations are provided:

      1. **Narrowband (centre-frequency) model** (when ``f0`` is given and
         ``f_range is None`` or ``n_poles == 1``): a single Drude pole with
         :math:`\\gamma = \\omega_0 = 2\\pi f_0` and
         :math:`\\sigma_{s,0} = \\text{sides}\\,\\sigma\\delta(\\omega_0)`,
         giving :math:`\\sigma_s(\\omega) = \\sigma_{s,0} / (1 + j\\omega/\\omega_0)`.
         This is exact at :math:`f = f_0` and accurate to :math:`< 5\\%` in
         :math:`|Z_s|` over :math:`[0.82 f_0, 1.22 f_0]` (:math:`\\pm 20\\%`
         fractional bandwidth).
      2. **Wideband multi-pole Stieltjes-Drude fit** (when ``f_range=(fmin, fmax)``
         is given with ``n_poles >= 2``): approximates
         :math:`\\sigma_s(\\omega) = \\sigma_\\infty + \\sum_{k=1}^K r_k / (\\gamma_k + j\\omega)`
         with :math:`\\sigma_\\infty, r_k \\ge 0` via nonnegative least squares on
         log-spaced relaxation rates :math:`\\gamma_k`, guaranteeing strict
         passivity (:math:`\\operatorname{Re}\\sigma_s(\\omega) > 0` for all
         :math:`\\omega`) and :math:`\\sim 1\\%` accuracy across an octave or
         3:1 mm-wave band.

    Parameters
    ----------
    sigma : float
        Bulk metal conductivity :math:`\\sigma` [S/m] (e.g. ``5.8e7`` for copper).
    thickness : float, optional
        Trace thickness :math:`t` [m]. Required for ``mode='thin'`` or ``mode='auto'``.
    mu_r : float, optional
        Relative magnetic permeability :math:`\\mu_r`. Default is 1.0.
    mode : {'auto', 'thin', 'sibc'}, optional
        Regime selection. ``'auto'`` compares ``thickness`` against the skin
        depth at ``f0`` (or geometric centre of ``f_range``), selecting
        ``'thin'`` when :math:`t \\le \\delta` (or when no frequency is given)
        and ``'sibc'`` when :math:`t > \\delta`.
    f0 : float, optional
        Centre frequency [Hz] for the narrowband skin-effect model (or if
        ``f_range is None`` and ``n_poles > 1``, fits over
        ``(0.5 * f0, 1.5 * f0)``).
    f_range : tuple of (float, float), optional
        Frequency band ``(fmin, fmax)`` [Hz] for the multi-pole skin-effect fit.
    n_poles : int, optional
        Number of Drude poles for ``mode='sibc'`` (1 for narrowband at ``f0``,
        default 3 when ``f_range`` is provided).
    sides : int, optional
        Number of skin-effect conducting surfaces in parallel (1 for single-sided
        surface impedance :math:`Z_s = (1+j)/(\\sigma\\delta)`, 2 for a two-sided
        sheet immersed in dielectric). Default is 1.
    name : str, optional
        Material label.

    Returns
    -------
    SheetMaterial
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    if mu_r <= 0:
        raise ValueError(f"mu_r must be > 0, got {mu_r}.")
    mu = float(mu_r) * mu_0
    mode = mode.lower()
    if mode not in ("auto", "thin", "dc", "sibc", "skin"):
        raise ValueError(f"mode must be 'auto', 'thin' or 'sibc', got {mode!r}.")

    f_ref = f0
    if f_ref is None and f_range is not None:
        f_ref = float(np.sqrt(f_range[0] * f_range[1]))

    if mode == "auto":
        if thickness is not None and f_ref is None:
            mode = "thin"
        elif thickness is None and f_ref is not None:
            mode = "sibc"
        elif thickness is not None and f_ref is not None:
            delta_ref = np.sqrt(2.0 / (2.0 * np.pi * f_ref * mu * sigma))
            mode = "thin" if thickness < delta_ref else "sibc"
        else:
            raise ValueError("Provide `thickness` (for thin film) or `f0`/`f_range` (for SIBC).")

    if mode in ("thin", "dc"):
        if thickness is None or thickness <= 0:
            raise ValueError("thickness must be > 0 for mode='thin'.")
        sig_s = float(sigma) * float(thickness)
        return SheetMaterial(
            poles=(),
            sigma_s=sig_s,
            name=name,
            meta={
                "model": "metal_thin",
                "sigma": float(sigma),
                "thickness": float(thickness),
                "R_s": 1.0 / sig_s,
            },
        )

    # mode in ("sibc", "skin"): Z_s(w) = (1 + j) / (sigma * delta(w))
    if f_range is None and f0 is None:
        raise ValueError("mode='sibc' requires `f0` or `f_range=(fmin, fmax)`.")

    if f_range is None or int(n_poles) == 1:
        if f0 is None:
            f0 = float(np.sqrt(f_range[0] * f_range[1]))
        if f0 <= 0:
            raise ValueError(f"f0 must be > 0, got {f0}.")
        w0 = 2.0 * np.pi * float(f0)
        delta0 = np.sqrt(2.0 / (w0 * mu * sigma))
        sig_s0 = float(sides) * float(sigma) * delta0
        wp = np.sqrt(sig_s0 * w0 / eps_0)
        return SheetMaterial(
            poles=[Drude(omega_p=wp, gamma=w0)],
            sigma_s=0.0,
            name=name,
            meta={
                "model": "metal_sibc_narrowband",
                "sigma": float(sigma),
                "mu_r": float(mu_r),
                "f0": float(f0),
                "delta_f0": float(delta0),
                "sides": int(sides),
                "validity_band_5pct": (0.82 * float(f0), 1.22 * float(f0)),
            },
        )

    fmin, fmax = float(f_range[0]), float(f_range[1])
    if fmin <= 0 or fmax <= fmin:
        raise ValueError(f"f_range=(fmin, fmax) must satisfy 0 < fmin < fmax, got {f_range}.")
    n_poles = int(n_poles)
    if n_poles < 1:
        raise ValueError(f"n_poles must be >= 1, got {n_poles}.")

    f_grid = np.geomspace(fmin, fmax, max(60, 16 * n_poles))
    w_grid = 2.0 * np.pi * f_grid
    delta_grid = np.sqrt(2.0 / (w_grid * mu * sigma))
    # Exact sheet conductivity: sides / Z_s(w) = sides * sigma * delta(w) / (1 + j)
    target = float(sides) * float(sigma) * delta_grid / (1.0 + 1j)

    gammas = np.geomspace(0.35 * w_grid[0], 2.8 * w_grid[-1], n_poles)
    basis = np.column_stack(
        [np.ones_like(w_grid, dtype=complex)]
        + [1.0 / (g + 1j * w_grid) for g in gammas]
    )
    w_norm = 1.0 / np.abs(target)
    A_mat = np.vstack([(basis * w_norm[:, None]).real, (basis * w_norm[:, None]).imag])
    b_vec = np.concatenate([(target * w_norm).real, (target * w_norm).imag])
    sol = lsq_linear(A_mat, b_vec, bounds=(0.0, np.inf))
    sig_inf = float(sol.x[0])
    poles = [
        Drude(omega_p=np.sqrt(rk / eps_0), gamma=float(gk))
        for rk, gk in zip(sol.x[1:], gammas)
        if rk > 0
    ]
    mat = SheetMaterial(
        poles=poles,
        sigma_s=sig_inf,
        name=name,
        meta={
            "model": "metal_sibc_multipole",
            "sigma": float(sigma),
            "mu_r": float(mu_r),
            "f_range": (fmin, fmax),
            "n_poles": n_poles,
            "sides": int(sides),
            "validity_band": (fmin, fmax),
        },
    )
    fit_sig = mat.surface_conductivity(f_grid)
    mat.fit_error = float(np.max(np.abs(fit_sig - target) / np.abs(target)))
    mat.meta["max_rel_error"] = mat.fit_error
    return mat


def sheet_transmission_reflection(sigma_s, f=None, theta=0.0, polarization="te", eta=ETA0):
    """
    Analytic transmission and reflection coefficients of a zero-thickness sheet
    in a homogeneous medium of wave impedance :math:`\\eta`.

    At normal incidence (:math:`\\theta = 0`), the complex amplitude coefficients
    are

    .. math::

        t = \\frac{1}{1 + \\eta\\sigma_s / 2},
        \\qquad
        r = -\\frac{\\eta\\sigma_s / 2}{1 + \\eta\\sigma_s / 2} = t - 1,

    with power transmittance :math:`T = |t|^2`, reflectance :math:`R = |r|^2`,
    and absorptance :math:`A = 1 - R - T = \\eta\\operatorname{Re}(\\sigma_s)|t|^2`.

    Parameters
    ----------
    sigma_s : float, complex, array_like, or SheetMaterial
        Sheet conductivity [S] or a :class:`SheetMaterial` evaluated at `f`.
    f : float or array_like, optional
        Frequency [Hz] (required when `sigma_s` is a :class:`SheetMaterial`).
    theta : float, optional
        Incidence angle [rad]. Default is 0.
    polarization : {'te', 'tm'}, optional
        Polarization for oblique incidence. Default is ``'te'``.
    eta : float, optional
        Background wave impedance [:math:`\\Omega`]. Default is :math:`\\eta_0`.

    Returns
    -------
    t, r, T, R : complex or ndarray
        Complex amplitude transmission ``t``, reflection ``r``, and power
        coefficients ``T = |t|^2``, ``R = |r|^2``.
    """
    if isinstance(sigma_s, DispersiveMaterial):
        if f is None:
            raise ValueError("`f` is required when `sigma_s` is a SheetMaterial.")
        if hasattr(sigma_s, "surface_conductivity"):
            sig = sigma_s.surface_conductivity(f)
        else:
            w = 2.0 * np.pi * np.asarray(f, dtype=float)
            sig = 1j * w * eps_0 * (sigma_s.epsilon(f) - 1.0)
    else:
        sig = np.asarray(sigma_s)
    cos_t = np.cos(theta)
    pol = polarization.lower()
    if pol == "te":
        half_norm = 0.5 * eta * sig / cos_t
    elif pol == "tm":
        half_norm = 0.5 * eta * sig * cos_t
    else:
        raise ValueError(f"polarization must be 'te' or 'tm', got {polarization!r}.")
    t = 1.0 / (1.0 + half_norm)
    r = -half_norm / (1.0 + half_norm)
    return t, r, np.abs(t) ** 2, np.abs(r) ** 2


def salisbury_screen_reflection(sigma_s, d, f, eta=ETA0):
    """
    Analytic normal-incidence reflection of a Salisbury screen (a sheet of
    conductivity :math:`\\sigma_s` placed a distance :math:`d` in front of a PEC
    ground plane, in :math:`\\exp(+j\\omega t)` convention):

    .. math::

        Y_{\\text{in}}(\\omega) = \\sigma_s(\\omega) - j\\frac{\\cot(k d)}{\\eta},
        \\qquad
        r(\\omega) = \\frac{1 - \\eta Y_{\\text{in}}(\\omega)}{1 + \\eta Y_{\\text{in}}(\\omega)},
        \\qquad
        R(\\omega) = |r(\\omega)|^2.

    For :math:`\\sigma_s = 1/\\eta` (:math:`377\\,\\Omega`/sq) and
    :math:`d = \\lambda_0 / 4` (:math:`\\cot(k_0 d) = 0`), :math:`r(f_0) = 0`.
    """
    f_arr = np.asarray(f, dtype=float)
    if isinstance(sigma_s, DispersiveMaterial):
        sig = (
            sigma_s.surface_conductivity(f_arr)
            if hasattr(sigma_s, "surface_conductivity")
            else 1j * 2.0 * np.pi * f_arr * eps_0 * (sigma_s.epsilon(f_arr) - 1.0)
        )
    else:
        sig = sigma_s
    k = 2.0 * np.pi * f_arr / c_light
    y_norm = eta * sig - 1j / np.tan(k * d)
    r = (1.0 - y_norm) / (1.0 + y_norm)
    return r, np.abs(r) ** 2


def _coerce_sheet_spec(spec, *, R_s=None, thickness=None, mode="auto", f0=None, f_range=None, n_poles=3):
    """Convert a single component's sheet specification to ``'pec'``, ``None``, or a ``SheetMaterial``."""
    if R_s is not None and spec is None:
        if R_s == 0:
            return "pec"
        return resistive_sheet(R_s=R_s)
    if spec is None:
        raise ValueError("Provide `sigma_s`, `material`, or `R_s` for the sheet.")
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "pec":
            return "pec"
        raise ValueError(f"Unknown sheet material string {spec!r}; expected 'pec'.")
    if isinstance(spec, DispersiveMaterial):
        return spec
    if isinstance(spec, (Drude, Lorentz, Debye)):
        return SheetMaterial(poles=[spec], sigma_s=0.0)
    if isinstance(spec, (list, tuple)) and all(isinstance(p, (Drude, Lorentz, Debye)) for p in spec):
        return SheetMaterial(poles=list(spec), sigma_s=0.0)
    if np.isscalar(spec):
        val = float(spec)
        if val < 0:
            raise ValueError(f"sigma_s must be >= 0, got {val}.")
        if np.isinf(val):
            return "pec"
        if thickness is not None or mode in ("thin", "sibc", "skin"):
            if val == 0.0:
                return None
            return metal_sheet(
                val,
                thickness=thickness,
                mode=mode,
                f0=f0,
                f_range=f_range,
                n_poles=n_poles,
            )
        if val == 0.0:
            return None
        return resistive_sheet(sigma_s=val)
    raise TypeError(f"Unsupported sheet conductivity/material specification: {spec!r}.")


def _point_in_triangles_2d(pts_u, pts_v, tri_uv):
    """Boolean mask of 2D points ``(pts_u, pts_v)`` inside any triangle of ``tri_uv`` (shape ``(M, 3, 2)``)."""
    if tri_uv.size == 0:
        return np.zeros(pts_u.shape, dtype=bool)
    p = np.stack([pts_u.ravel(), pts_v.ravel()], axis=-1)[:, None, :]  # (P, 1, 2)
    a = tri_uv[None, :, 0, :]
    b = tri_uv[None, :, 1, :]
    c = tri_uv[None, :, 2, :]
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = np.sum(v0 * v0, axis=-1)
    dot01 = np.sum(v0 * v1, axis=-1)
    dot02 = np.sum(v0 * v2, axis=-1)
    dot11 = np.sum(v1 * v1, axis=-1)
    dot12 = np.sum(v1 * v2, axis=-1)
    denom = dot00 * dot11 - dot01 * dot01
    valid = np.abs(denom) > 1e-30
    inv = np.divide(1.0, denom, out=np.zeros_like(denom), where=valid)
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    tol = -1e-9
    inside = valid & (u >= tol) & (v >= tol) & (u + v <= 1.0 - tol)
    return np.any(inside, axis=1).reshape(pts_u.shape)


class SubcellMixin:
    """
    ``SolverFIT3D`` mixin providing subcell thin-sheet (:meth:`add_sheet`) and
    thin-wire (:meth:`add_thin_wire`) models.
    """

    def _parse_sheet_geometry(self, sheet_or_box, plane, normal, position, center, size):
        """
        Resolve sheet geometry into ``(axis_w, k0, bounds_uv, tri_uv)``.

        Returns
        -------
        axis_w : int
            Normal axis index in ``(0, 1, 2)``.
        k0 : int
            Grid node plane index along `axis_w`.
        bounds_uv : tuple of (float, float, float, float)
            In-plane bounding box ``(u_min, u_max, v_min, v_max)``.
        tri_uv : ndarray or None
            2D triangle vertices ``(M, 3, 2)`` when defined from a surface mesh.
        """
        nodes = (self.grid.x, self.grid.y, self.grid.z)
        tri_uv = None
        mesh_obj = None

        # Handle plane=("z", z0) tuple shorthand
        if isinstance(plane, (tuple, list)) and len(plane) == 2 and isinstance(plane[0], str):
            plane, position = plane[0], float(plane[1])

        # Inspect sheet_or_box
        if sheet_or_box is not None:
            if isinstance(sheet_or_box, (str, pathlib.Path)):
                import pyvista as pv

                mesh_obj = pv.read(str(sheet_or_box))
            elif hasattr(sheet_or_box, "faces") and hasattr(sheet_or_box, "points"):
                mesh_obj = sheet_or_box
            elif isinstance(sheet_or_box, dict):
                center = sheet_or_box.get("center", center)
                size = sheet_or_box.get("size", size)
                plane = sheet_or_box.get("plane", plane)
                normal = sheet_or_box.get("normal", normal)
                position = sheet_or_box.get("position", position)
            elif hasattr(sheet_or_box, "center") and hasattr(sheet_or_box, "size"):
                center = sheet_or_box.center
                size = sheet_or_box.size
            elif hasattr(sheet_or_box, "bounds"):
                b = np.asarray(sheet_or_box.bounds, dtype=float)
                if b.size == 6:
                    center = 0.5 * (b[0::2] + b[1::2])
                    size = b[1::2] - b[0::2]
            elif isinstance(sheet_or_box, (tuple, list, np.ndarray)):
                arr = np.asarray(sheet_or_box, dtype=float)
                if arr.shape == (2, 3):
                    center, size = arr[0], arr[1]
                elif arr.shape == (6,):
                    center = 0.5 * (arr[0::2] + arr[1::2])
                    size = arr[1::2] - arr[0::2]
                else:
                    raise ValueError(
                        "sheet_or_box tuple/array must have shape (2, 3) as (center, size) "
                        f"or (6,) as bounds, got {arr.shape}."
                    )
            else:
                raise TypeError(f"Unsupported sheet_or_box type: {type(sheet_or_box)!r}.")

        # Determine normal axis
        axis_w = None
        spec_norm = normal if normal is not None else plane
        if spec_norm is not None:
            if isinstance(spec_norm, str):
                key = spec_norm.strip().lower()
                if key not in _PLANE_TO_NORMAL:
                    raise ValueError(
                        f"Unknown plane/normal {spec_norm!r}; expected 'x', 'y', 'z', 'xy', 'xz', or 'yz'."
                    )
                axis_w = _PLANE_TO_NORMAL[key]
            else:
                vec = np.asarray(spec_norm, dtype=float)
                nz = np.flatnonzero(np.abs(vec) > 1e-12)
                if vec.shape != (3,) or nz.size != 1:
                    raise ValueError(
                        f"Sheet normal vector must be axis-aligned (x, y, or z), got {spec_norm!r}."
                    )
                axis_w = int(nz[0])

        n_xyz = (self.Nx, self.Ny, self.Nz)
        ghosts = getattr(self.grid, "edge_ghosts", [0, 0, 0])
        dom_hi = tuple(float(nodes[d][n_xyz[d] - ghosts[d]]) for d in range(3))

        if mesh_obj is not None:
            from .conformal import _triangles

            tris_3d = _triangles(mesh_obj)
            b = np.asarray(mesh_obj.bounds, dtype=float)
            spans = b[1::2] - b[0::2]
            if axis_w is None:
                axis_w = int(np.argmin(spans))
            d_min = float(np.min(np.diff(nodes[axis_w])))
            if spans[axis_w] > 1e-2 * d_min:
                raise ValueError(
                    f"STL/PolyData sheet must be planar and axis-aligned along {'xyz'[axis_w]}, "
                    f"but its thickness along {'xyz'[axis_w]} is {spans[axis_w]:.3e} m."
                )
            if position is None:
                position = 0.5 * (b[2 * axis_w] + b[2 * axis_w + 1])
            u_ax, v_ax = _INPLANE_AXES[axis_w]
            bounds_uv = (b[2 * u_ax], b[2 * u_ax + 1], b[2 * v_ax], b[2 * v_ax + 1])
            tri_uv = tris_3d[:, :, [u_ax, v_ax]]
        else:
            if center is not None:
                center = np.asarray(center, dtype=float)
            if size is not None:
                size = np.asarray(size, dtype=float)
                if np.any(size < 0):
                    raise ValueError("Sheet size must be non-negative.")
            if axis_w is None:
                if size is not None:
                    zeros = np.flatnonzero(np.isclose(size, 0.0, atol=1e-15))
                    if zeros.size == 1:
                        axis_w = int(zeros[0])
                if axis_w is None:
                    raise ValueError(
                        "Specify `plane`/`normal` ('x', 'y', or 'z') or pass a `size` with one zero component."
                    )
            if position is None:
                if center is not None:
                    position = float(center[axis_w])
                else:
                    position = float(0.5 * (nodes[axis_w][0] + dom_hi[axis_w]))
            u_ax, v_ax = _INPLANE_AXES[axis_w]
            if center is None or size is None:
                bounds_uv = (
                    float(nodes[u_ax][0]),
                    dom_hi[u_ax],
                    float(nodes[v_ax][0]),
                    dom_hi[v_ax],
                )
            else:
                bounds_uv = (
                    float(center[u_ax] - 0.5 * size[u_ax]),
                    float(center[u_ax] + 0.5 * size[u_ax]),
                    float(center[v_ax] - 0.5 * size[v_ax]),
                    float(center[v_ax] + 0.5 * size[v_ax]),
                )

        # Locate grid plane k0 along axis_w
        n_w = n_xyz[axis_w]
        coords_w = nodes[axis_w][:n_w]
        dw_min = float(np.min(np.diff(nodes[axis_w])))
        if position < nodes[axis_w][0] - 0.5 * dw_min or position > dom_hi[axis_w] + 0.5 * dw_min:
            raise ValueError(
                f"Sheet position {'xyz'[axis_w]}={position} lies outside the grid domain "
                f"[{nodes[axis_w][0]}, {dom_hi[axis_w]}]."
            )
        k0 = int(np.argmin(np.abs(coords_w - position)))
        if abs(coords_w[k0] - position) > 1e-3 * dw_min:
            warnings.warn(
                f"Thin sheet at {'xyz'[axis_w]}={position:.6g} m does not lie on a primary grid "
                f"plane; snapping to nearest plane {'xyz'[axis_w]}[{k0}]={coords_w[k0]:.6g} m.",
                UserWarning,
                stacklevel=3,
            )
        return axis_w, k0, bounds_uv, tri_uv

    def _sheet_component_edges(self, axis_w, k0, comp_axis, other_axis, a_min, a_max, b_min, b_max, tri_uv=None):
        """
        Compute flat E indices and in-plane coverage fractions ``f_e in [0, 1]``
        for tangential component `comp_axis` on grid plane ``axis_w = k0``.
        """
        nodes = (self.grid.x, self.grid.y, self.grid.z)
        n_xyz = (self.Nx, self.Ny, self.Nz)
        ghosts = getattr(self.grid, "edge_ghosts", [0, 0, 0])
        n_a = n_xyz[comp_axis]
        n_b = n_xyz[other_axis]
        coords_a = nodes[comp_axis]
        coords_b = nodes[other_axis]

        # Valid primary edges along comp_axis (exclude ghost edge beyond domain on non-periodic axes)
        max_ia = n_a - ghosts[comp_axis]
        ia_all = np.arange(max_ia)
        a_lo = coords_a[ia_all]
        a_hi = coords_a[ia_all + 1]
        da = a_hi - a_lo
        frac_a = np.clip((np.minimum(a_hi, a_max) - np.maximum(a_lo, a_min)) / da, 0.0, 1.0)

        # Dual intervals along other_axis around nodes ib = 0 .. n_b - 1
        max_ib = n_b
        ib_all = np.arange(max_ib)
        db_min = float(np.min(np.diff(coords_b)))
        tol_b = 1e-6 * db_min
        b_dom_hi = float(coords_b[n_b - ghosts[other_axis]])
        full_b = (b_min <= coords_b[0] + tol_b) and (b_max >= b_dom_hi - tol_b)
        if full_b:
            frac_b = np.ones(max_ib, dtype=float)
        else:
            b_mid = 0.5 * (coords_b[:-1] + coords_b[1:])
            b_dual_lo = np.empty(max_ib, dtype=float)
            b_dual_hi = np.empty(max_ib, dtype=float)
            b_dual_lo[0] = coords_b[0]
            b_dual_lo[1:] = b_mid[: max_ib - 1]
            b_dual_hi[:-1] = b_mid[: max_ib - 1]
            b_dual_hi[-1] = b_dom_hi
            span_b = np.maximum(b_dual_hi - b_dual_lo, 1e-30)
            frac_b = np.clip(
                (np.minimum(b_dual_hi, b_max) - np.maximum(b_dual_lo, b_min)) / span_b,
                0.0,
                1.0,
            )

        ia_grid, ib_grid = np.meshgrid(ia_all, ib_all, indexing="ij")
        frac_grid = frac_a[:, None] * frac_b[None, :]

        if tri_uv is not None:
            # Subcell quadrature (4x4 per edge cell) for arbitrary planar triangle mesh
            u_ax, _ = _INPLANE_AXES[axis_w]
            q_pts = (np.arange(4) + 0.5) / 4.0
            qa = a_lo[:, None] + da[:, None] * q_pts[None, :]  # (n_ia, 4)
            b_mid = 0.5 * (coords_b[:-1] + coords_b[1:])
            b_dual_lo = np.r_[coords_b[0], b_mid[: max_ib - 1]]
            b_dual_hi = np.r_[b_mid[: max_ib - 1], coords_b[max_ib - 1]]
            qb = b_dual_lo[:, None] + (b_dual_hi - b_dual_lo)[:, None] * q_pts[None, :]  # (n_ib, 4)
            QA, QB = np.meshgrid(qa.ravel(), qb.ravel(), indexing="ij")
            if comp_axis == u_ax:
                inside = _point_in_triangles_2d(QA, QB, tri_uv)
            else:
                inside = _point_in_triangles_2d(QB, QA, tri_uv)
            inside = inside.reshape(ia_all.size, 4, ib_all.size, 4)
            frac_grid = inside.mean(axis=(1, 3))

        keep = frac_grid > 0
        ia_sel = ia_grid[keep]
        ib_sel = ib_grid[keep]
        frac_sel = frac_grid[keep]

        ijk = [None, None, None]
        ijk[axis_w] = np.full(ia_sel.size, k0, dtype=int)
        ijk[comp_axis] = ia_sel
        ijk[other_axis] = ib_sel
        flat_3d = np.ravel_multi_index((ijk[0], ijk[1], ijk[2]), n_xyz, order="F")
        flat_idx = flat_3d + comp_axis * self.N

        # FIT integral factor: (f_e * tL_other) * itA_comp = f_e / tL_normal
        tL_other = self.grid.tL.toarray()[flat_3d + other_axis * self.N]
        itA_comp = self.grid.itA.toarray()[flat_idx]
        dual_weight = frac_sel * tL_other * itA_comp
        return flat_idx, dual_weight, frac_sel

    def _register_sheet_group(self, material, idx, weight):
        """
        Attach a zero-thickness sheet ADE group on flat E indices ``idx`` with
        dual-length weights ``weight = f_e * tL_inplane * itA`` [1/m].
        """
        if idx.size == 0:
            return
        if getattr(self, "scheme", "yee") == "tete":
            raise NotImplementedError("scheme='tete' does not support thin sheets.")

        ieps = np.asarray(_host(self.ieps.array))
        keep = (weight > 0) & (ieps[idx] != 0)
        idx, weight = idx[keep], weight[keep]
        if idx.size == 0:
            return

        # Check stability using the maximum effective pole weight on the grid
        w_max = float(np.max(weight))
        scaled_poles = [_scale_pole(p, w_max) for p in material.poles]
        eff_mat = DispersiveMaterial(
            eps_inf=max(1.0 + w_max * (material.eps_inf - 1.0), 1e-6),
            poles=scaled_poles,
            sigma=w_max * material.sigma,
            name=material.name or "sheet",
        )
        self._check_dispersive_stable(eff_mat)

        if not hasattr(self, "_dispersive_groups"):
            self._dispersive_groups = []
            fused = getattr(self, "_fused_step", None)
            if (
                fused is not None
                and self.one_step is fused
                and np.dtype(self.dtype) == np.float64
            ):
                from .gpu_fused import ADEHooks

                hooks = ADEHooks(self)
                fused.pre.append(hooks.pre)
                fused.post.append(hooks.post)
            elif (
                getattr(self, "activate_abc", False)
                and self.one_step == self._one_step_with_abc
            ):
                self._one_step_nondispersive = self._one_step_backend
                self._one_step_backend = self._one_step_dispersive
            else:
                self._one_step_nondispersive = self.one_step
                self.one_step = self._one_step_dispersive

        # Later assignments on the same edges override earlier groups
        for group in self._dispersive_groups:
            old = ~np.isin(group["idx_host"], idx)
            if not old.all():
                _restrict_group(group, old, self.E.xp)

        eps_s = 1.0 / (eps_0 * ieps[idx])
        # Zero-thickness sheet adds surface susceptibility without displacing bulk eps_s
        eps_e = eps_s + weight * (material.eps_inf - 1.0)
        lorentz, debye = _ade_coefficients(material, self.dt)
        beta_sum = weight * sum(beta for _, beta in debye)
        s_hat = weight * material.sigma * self.dt / eps_0 / 2.0
        xp = self.E.xp
        zeros = xp.zeros(idx.size, dtype=self.dtype)
        group = {
            "material": material,
            "idx_host": idx,
            "idx": xp.asarray(idx),
            "weight": xp.asarray(weight, dtype=self.dtype),
            "eps_s": xp.asarray(eps_s, dtype=self.dtype),
            "a1": xp.asarray(eps_e - s_hat - beta_sum, dtype=self.dtype),
            "den": xp.asarray(eps_e + s_hat + beta_sum, dtype=self.dtype),
            "lorentz": [(*cf, zeros.copy(), zeros.copy()) for cf in lorentz],
            "debye": [(*cf, zeros.copy()) for cf in debye],
        }
        self._dispersive_groups = [
            g for g in self._dispersive_groups if g["idx_host"].size
        ] + [group]

    def _set_pec_edges(self, idx):
        """Zero tangential ``ieps`` and ``E`` on flat E indices ``idx`` and update operators."""
        idx = np.unique(np.asarray(idx, dtype=int))
        if idx.size == 0:
            return
        xp = self.E.xp
        idx_dev = xp.asarray(idx)
        self.ieps.array[idx_dev] = 0.0
        self.E.array[idx_dev] = 0.0
        if not self.activate_cpml and hasattr(self, "itDaiDepsDstC") and self.itDaiDepsDstC is not None:
            scale = xp.ones(3 * self.N, dtype=self.dtype)
            scale[idx_dev] = 0.0
            _scale_operator_rows(self.itDaiDepsDstC, scale, xp)
            if hasattr(self, "iDeps"):
                self.iDeps = diags(_host(self.ieps.toarray()), dtype=self.dtype)

    def add_sheet(
        self,
        sheet_or_box=None,
        sigma_s=None,
        plane=None,
        *,
        normal=None,
        position=None,
        center=None,
        size=None,
        material=None,
        R_s=None,
        thickness=None,
        mode="auto",
        f0=None,
        f_range=None,
        n_poles=3,
        dual_length_factor=None,
    ):
        """
        Add a zero-thickness conducting or dispersive sheet on a grid plane.

        In FIT integral form, on every tangential primary :math:`\\mathbf{E}` edge
        :math:`e` lying in the sheet, the surface current
        :math:`\\mathbf{J}_s = \\sigma_s(\\omega)\\mathbf{E}_t` adds a current
        :math:`I_{s,e} = \\sigma_s E_e\\,(f_e \\tilde{L}_{\\text{in-plane},e})`
        through the dual facet :math:`\\tilde{A}_e`, which divides by
        :math:`\\tilde{A}_e` to give the weight
        :math:`f_e \\tilde{L}_{\\text{in-plane},e} / \\tilde{A}_e = f_e / \\tilde{L}_{\\perp,e}`.

        Parameters
        ----------
        sheet_or_box : tuple, dict, Sheet, Box, pyvista.PolyData, str or Path, optional
            Sheet geometry: ``(center, size)`` tuple, object with ``center`` and
            ``size``, bounds ``(xmin, xmax, ymin, ymax, zmin, zmax)``, planar
            ``pyvista.PolyData`` mesh, or path to a planar ``.stl`` file. If
            omitted, use `center`/`size` or `plane`/`normal` + `position` for a
            full-plane sheet.
        sigma_s : float, str, SheetMaterial, DispersiveMaterial, Pole, tuple or dict, optional
            Sheet conductivity :math:`\\sigma_s` [S], ``'pec'`` / ``np.inf`` for
            a PEC sheet, a :class:`SheetMaterial` / :class:`DispersiveMaterial`
            (e.g. from :func:`graphene_sheet` or :func:`metal_sheet`), or a
            2-tuple ``(sigma_u, sigma_v)`` / dict ``{'x': ..., 'y': ...}`` for
            anisotropic in-plane surface conductivity. When `thickness` or
            ``mode='sibc'`` is given, a scalar `sigma_s` is interpreted as bulk
            metal conductivity :math:`\\sigma` [S/m].
        plane, normal : str, tuple, or array_like, optional
            Sheet plane or normal axis (``'x'``, ``'y'``, ``'z'``, ``'xy'``,
            ``'xz'``, ``'yz'``, ``('z', z0)``, or a unit normal vector).
        position : float, optional
            Coordinate [m] of the sheet along its normal axis.
        center, size : tuple of float, optional
            Box centre and extent ``(x, y, z)`` [m] when `sheet_or_box` is not
            passed positionally.
        material : SheetMaterial, DispersiveMaterial, str, or float, optional
            Alias for `sigma_s`.
        R_s : float or tuple, optional
            Sheet resistance [:math:`\\Omega`/sq], :math:`R_s = 1 / \\sigma_s`.
        thickness : float, optional
            Metal trace thickness :math:`t` [m] for :func:`metal_sheet`.
        mode : {'auto', 'thin', 'sibc'}, optional
            Regime for thin metal traces (see :func:`metal_sheet`).
        f0 : float, optional
            Centre frequency [Hz] for narrowband skin-effect surface impedance.
        f_range : tuple of (float, float), optional
            Frequency range ``(fmin, fmax)`` [Hz] for multi-pole skin-effect fit.
        n_poles : int, optional
            Number of poles for multi-pole skin-effect fit. Default is 3.
        dual_length_factor : float, optional
            Multiplier on the dual-edge length crossing the sheet (default 1.0;
            used for testing/mutation verification of the FIT integral factor).

        Returns
        -------
        dict
            Summary dictionary with normal axis, plane index, edge indices, and
            applied models.
        """
        if not self.step_0:
            raise RuntimeError("Call add_sheet before starting time-stepping.")
        spec = material if material is not None else sigma_s
        axis_w, k0, (u_min, u_max, v_min, v_max), tri_uv = self._parse_sheet_geometry(
            sheet_or_box, plane, normal, position, center, size
        )
        u_ax, v_ax = _INPLANE_AXES[axis_w]
        u_name, v_name = _AXES[u_ax], _AXES[v_ax]

        # Resolve isotropic vs anisotropic (u, v) specifications
        kw = dict(thickness=thickness, mode=mode, f0=f0, f_range=f_range, n_poles=n_poles)
        if isinstance(spec, dict):
            spec_u = _coerce_sheet_spec(spec.get(u_name, 0.0), **kw)
            spec_v = _coerce_sheet_spec(spec.get(v_name, 0.0), **kw)
        elif (
            isinstance(spec, (tuple, list))
            and len(spec) == 2
            and not all(isinstance(p, (Drude, Lorentz, Debye)) for p in spec)
        ):
            spec_u = _coerce_sheet_spec(spec[0], **kw)
            spec_v = _coerce_sheet_spec(spec[1], **kw)
        elif isinstance(R_s, (tuple, list)) and len(R_s) == 2:
            spec_u = _coerce_sheet_spec(None, R_s=R_s[0], **kw)
            spec_v = _coerce_sheet_spec(None, R_s=R_s[1], **kw)
        else:
            spec_u = spec_v = _coerce_sheet_spec(spec, R_s=R_s, **kw)

        dl_factor = (
            float(getattr(self, "_sheet_dual_length_factor", 1.0))
            if dual_length_factor is None
            else float(dual_length_factor)
        )

        idx_u, w_u, _ = self._sheet_component_edges(
            axis_w, k0, u_ax, v_ax, u_min, u_max, v_min, v_max, tri_uv=tri_uv
        )
        idx_v, w_v, _ = self._sheet_component_edges(
            axis_w, k0, v_ax, u_ax, v_min, v_max, u_min, u_max, tri_uv=tri_uv
        )
        w_u = w_u * dl_factor
        w_v = w_v * dl_factor

        if spec_u is spec_v:
            idx_all = np.concatenate([idx_u, idx_v])
            w_all = np.concatenate([w_u, w_v])
            if spec_u == "pec":
                self._set_pec_edges(idx_all)
            elif spec_u is not None:
                self._register_sheet_group(spec_u, idx_all, w_all)
        else:
            for sp_c, idx_c, w_c in ((spec_u, idx_u, w_u), (spec_v, idx_v, w_v)):
                if sp_c == "pec":
                    self._set_pec_edges(idx_c)
                elif sp_c is not None:
                    self._register_sheet_group(sp_c, idx_c, w_c)

        info = {
            "normal": _AXES[axis_w],
            "plane_index": k0,
            "position": float((self.grid.x, self.grid.y, self.grid.z)[axis_w][k0]),
            "idx_u": idx_u,
            "idx_v": idx_v,
            "material_u": spec_u,
            "material_v": spec_v,
        }
        if not hasattr(self, "_sheets"):
            self._sheets = []
        self._sheets.append(info)
        return info

    def add_thin_wire(
        self,
        points,
        radius,
        *,
        material="pec",
        sigma=None,
        R_per_m=None,
        gap=None,
        correction=True,
    ):
        """
        Add a straight or axis-aligned polyline thin wire of subcell radius
        :math:`a < \\Delta / 2` along primary grid edges.

        Applies the contour-path / in-cell inductance correction [1-3] in
        symmetric FIT Hodge form:

        * Around every wire segment along axis :math:`w` at transverse grid node
          :math:`(u_i, v_j)`, the four circulating magnetic field components
          (:math:`H_u(i, j, k)`, :math:`H_u(i, j-1, k)`, :math:`H_v(i, j, k)`,
          :math:`H_v(i-1, j, k)`) have their inverse permeability ``imu``
          divided by :math:`m = \\frac{1}{2}\\ln(\\Delta_\\perp / a)`.
        * At each wire node of degree :math:`d` (:math:`d = 2` at interior and
          bend nodes, :math:`d = 1` at wire ends, :math:`d \\ge 3` at junctions),
          the non-wire incident primary :math:`\\mathbf{E}` edges have their
          inverse permittivity ``ieps`` multiplied by :math:`(2/d)m` (with
          paired permeability scaling on the adjacent faces to preserve the
          vacuum Courant limit at the standard timestep).
        * On the wire primary edges themselves, ``ieps = 0`` for a PEC wire, or
          a semi-implicit conductive ADE update with conductance per unit length
          :math:`G' = \\pi a^2 \\sigma = 1 / R_{\\text{per\\_m}}` for a lossy
          wire, except on gap edges excluded via `gap` (or automatically
          re-opened when a :class:`~wakis.lumped.LumpedPort` or
          :class:`~wakis.lumped.LumpedElement` is placed on the wire).

        Parameters
        ----------
        points : sequence of array_like
            Sequence of 3D vertices ``[(x0, y0, z0), (x1, y1, z1), ...]`` [m]
            defining a straight wire (2 points) or an axis-aligned polyline
            (:math:`\\ge 2` points). Each segment must be parallel to the x, y,
            or z axis.
        radius : float
            Wire radius :math:`a` [m]. Must satisfy :math:`0 < a < \\Delta_\\perp / 2`.
        material : {'pec', 'lossy'}, optional
            Wire material type. Default is ``'pec'``.
        sigma : float, optional
            Bulk wire conductivity :math:`\\sigma` [S/m] for a lossy wire,
            giving DC resistance per unit length :math:`R' = 1 / (\\pi a^2 \\sigma)`.
        R_per_m : float, optional
            Resistance per unit length :math:`R'` [:math:`\\Omega`/m] for a lossy
            wire.
        gap : LumpedElement, LumpedPort, str, float, tuple, or sequence, optional
            Feed/load gap specification on the wire where ``ieps`` is left at its
            background value while keeping the thin-wire magnetic Hodge correction
            around the gap:
            - a :class:`~wakis.lumped.LumpedPort` or :class:`~wakis.lumped.LumpedElement`
              (or list thereof),
            - ``'center'`` / ``'centre'`` to leave the middle segment of the wire open,
            - a float coordinate along the wire axis, or
            - a ``(center, size)`` box tuple.
        correction : bool, optional
            If ``True`` (default), apply the :math:`\\ln(\\Delta/a)` subcell
            Hodge correction; if ``False``, only zero ``ieps`` on the wire edges
            (used for mutation testing).

        Returns
        -------
        dict
            Summary dictionary with the wire segments, node degrees, and flat
            indices of modified ``E`` and ``H`` edges.
        """
        if not self.step_0:
            raise RuntimeError("Call add_thin_wire before starting time-stepping.")
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
            raise ValueError(
                f"`points` must be an (M, 3) array of at least 2 3D points, got shape {pts.shape}."
            )
        radius = float(radius)
        if radius <= 0:
            raise ValueError(f"Wire radius must be > 0, got {radius}.")

        nodes = (self.grid.x, self.grid.y, self.grid.z)
        n_xyz = (self.Nx, self.Ny, self.Nz)

        # Trace polyline segments into a list of unit grid edges (axis_w, i_x, i_y, i_z)
        wire_edges = []
        seen_edges = set()
        node_degrees = {}

        for seg_idx in range(pts.shape[0] - 1):
            p0, p1 = pts[seg_idx], pts[seg_idx + 1]
            diff = p1 - p0
            active = np.flatnonzero(np.abs(diff) > 1e-12)
            if active.size != 1:
                raise ValueError(
                    f"Wire segment {seg_idx} from {p0.tolist()} to {p1.tolist()} must be "
                    "aligned with the x, y, or z grid axis."
                )
            axis_w = int(active[0])
            u_ax, v_ax = _INPLANE_AXES[axis_w]

            iu = int(np.argmin(np.abs(nodes[u_ax][: n_xyz[u_ax]] - p0[u_ax])))
            iv = int(np.argmin(np.abs(nodes[v_ax][: n_xyz[v_ax]] - p0[v_ax])))
            w_lo, w_hi = (p0[axis_w], p1[axis_w]) if p0[axis_w] <= p1[axis_w] else (p1[axis_w], p0[axis_w])
            k_start = int(np.argmin(np.abs(nodes[axis_w][: n_xyz[axis_w]] - w_lo)))
            k_stop = int(np.argmin(np.abs(nodes[axis_w][: n_xyz[axis_w]] - w_hi)))
            if k_stop <= k_start:
                raise ValueError(
                    f"Wire segment {seg_idx} along {'xyz'[axis_w]} spans zero grid edges "
                    f"(k_start={k_start}, k_stop={k_stop})."
                )

            du = float(nodes[u_ax][iu + 1] - nodes[u_ax][iu])
            dv = float(nodes[v_ax][iv + 1] - nodes[v_ax][iv])
            d_perp = min(du, dv)
            if radius >= 0.5 * d_perp:
                raise ValueError(
                    f"Thin-wire radius a={radius:.4g} m must be < 0.5 * cell_size "
                    f"({0.5 * d_perp:.4g} m along {'xyz'[u_ax]}/{'xyz'[v_ax]})."
                )

            for k in range(k_start, k_stop):
                ijk = [0, 0, 0]
                ijk[axis_w] = k
                ijk[u_ax] = iu
                ijk[v_ax] = iv
                edge_key = (axis_w, ijk[0], ijk[1], ijk[2])
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                wire_edges.append(edge_key)

                n0 = [ijk[0], ijk[1], ijk[2]]
                n1 = [ijk[0], ijk[1], ijk[2]]
                n1[axis_w] = k + 1
                t0, t1 = tuple(n0), tuple(n1)
                node_degrees[t0] = node_degrees.get(t0, 0) + 1
                node_degrees[t1] = node_degrees.get(t1, 0) + 1

        # Determine gap edges (left open for lumped port / element)
        gap_flat_set = set()
        if gap is not None:
            gaps = gap if isinstance(gap, (list, tuple)) and not (
                len(gap) == 2 and np.asarray(gap[0]).shape == (3,)
            ) else [gap]
            for g in gaps:
                if isinstance(g, str) and g.strip().lower() in ("center", "centre", "mid", "middle"):
                    mid_edge = wire_edges[len(wire_edges) // 2]
                    c_ax, ix, iy, iz = mid_edge
                    gap_flat_set.add(
                        int(np.ravel_multi_index((ix, iy, iz), n_xyz, order="F") + c_ax * self.N)
                    )
                elif hasattr(g, "component") and hasattr(g, "center") and hasattr(g, "size"):
                    c_ax = _AXES.index(g.component)
                    ind, _ = self.grid.region_indices("E", g.component, g.center, g.size)
                    II, JJ, KK = np.meshgrid(*ind, indexing="ij")
                    fl = (c_ax * self.N + II + self.Nx * JJ + self.Nx * self.Ny * KK).ravel()
                    gap_flat_set.update(int(x) for x in fl)
                elif isinstance(g, (tuple, list)) and len(g) == 2 and np.asarray(g[0]).shape == (3,):
                    c_center, c_size = np.asarray(g[0], float), np.asarray(g[1], float)
                    for c_ax in range(3):
                        ind, _ = self.grid.region_indices("E", _AXES[c_ax], c_center, c_size)
                        II, JJ, KK = np.meshgrid(*ind, indexing="ij")
                        fl = (c_ax * self.N + II + self.Nx * JJ + self.Nx * self.Ny * KK).ravel()
                        gap_flat_set.update(int(x) for x in fl)
                elif np.isscalar(g):
                    pos_g = float(g)
                    # Find wire edge whose midpoint is closest to pos_g along its axis
                    best_edge = min(
                        wire_edges,
                        key=lambda e: abs(
                            0.5 * (nodes[e[0]][e[1 + e[0]]] + nodes[e[0]][e[1 + e[0]] + 1]) - pos_g
                        ),
                    )
                    c_ax, ix, iy, iz = best_edge
                    gap_flat_set.add(
                        int(np.ravel_multi_index((ix, iy, iz), n_xyz, order="F") + c_ax * self.N)
                    )
                else:
                    raise ValueError(f"Unsupported gap specification: {g!r}.")

        use_corr = bool(correction) and bool(getattr(self, "_thin_wire_correction_enabled", True))
        ieps_host = _host(self.ieps.array).copy()
        imu_host = _host(self.imu.array).copy()
        ieps_new = ieps_host.copy()
        imu_scale = np.ones_like(imu_host)

        if not hasattr(self, "_thin_wire_pec_edges"):
            self._thin_wire_pec_edges = {}

        # Determine if wire is lossy or PEC
        is_lossy = (sigma is not None) or (R_per_m is not None) or (str(material).lower() not in ("pec",))
        if is_lossy:
            if R_per_m is None:
                if sigma is None or sigma <= 0:
                    raise ValueError("Lossy thin wire requires `sigma > 0` [S/m] or `R_per_m > 0` [Ohm/m].")
                R_per_m = 1.0 / (np.pi * radius**2 * float(sigma))
            elif R_per_m <= 0:
                raise ValueError(f"R_per_m must be > 0, got {R_per_m}.")

        wire_flat_all = []
        wire_flat_active = []
        for axis_w, ix, iy, iz in wire_edges:
            flat_e = int(np.ravel_multi_index((ix, iy, iz), n_xyz, order="F") + axis_w * self.N)
            wire_flat_all.append(flat_e)
            if flat_e not in gap_flat_set:
                wire_flat_active.append(flat_e)

            if not use_corr:
                continue

            u_ax, v_ax = _INPLANE_AXES[axis_w]
            ijk = [ix, iy, iz]
            iu, iv = ijk[u_ax], ijk[v_ax]
            du_plus = float(nodes[u_ax][iu + 1] - nodes[u_ax][iu])
            du_minus = float(nodes[u_ax][iu] - nodes[u_ax][iu - 1]) if iu > 0 else du_plus
            dv_plus = float(nodes[v_ax][iv + 1] - nodes[v_ax][iv])
            dv_minus = float(nodes[v_ax][iv] - nodes[v_ax][iv - 1]) if iv > 0 else dv_plus

            # 4 circulating H faces around wire segment:
            # H_u at (iu, iv, kw) and (iu, iv - 1, kw) have width dv along v_ax
            for jv, dv_loc in ((iv, dv_plus), (iv - 1, dv_minus)):
                if 0 <= jv < n_xyz[v_ax]:
                    h_ijk = list(ijk)
                    h_ijk[v_ax] = jv
                    h_flat = int(np.ravel_multi_index(h_ijk, n_xyz, order="F") + u_ax * self.N)
                    m_loc = 0.5 * np.log(dv_loc / radius)
                    imu_scale[h_flat] = min(imu_scale[h_flat], 1.0 / m_loc)

            # H_v at (iu, iv, kw) and (iu - 1, iv, kw) have width du along u_ax
            for ju, du_loc in ((iu, du_plus), (iu - 1, du_minus)):
                if 0 <= ju < n_xyz[u_ax]:
                    h_ijk = list(ijk)
                    h_ijk[u_ax] = ju
                    h_flat = int(np.ravel_multi_index(h_ijk, n_xyz, order="F") + v_ax * self.N)
                    m_loc = 0.5 * np.log(du_loc / radius)
                    imu_scale[h_flat] = min(imu_scale[h_flat], 1.0 / m_loc)

        wire_flat_set = set(wire_flat_all)

        # Node radial & tip capacitance scaling: w_node = (2 / degree) * m
        if use_corr:
            for (nx, ny, nz), deg in node_degrees.items():
                n_ijk = (nx, ny, nz)
                dx_loc = float(nodes[0][min(nx, n_xyz[0] - 1) + 1] - nodes[0][min(nx, n_xyz[0] - 1)])
                dy_loc = float(nodes[1][min(ny, n_xyz[1] - 1) + 1] - nodes[1][min(ny, n_xyz[1] - 1)])
                dz_loc = float(nodes[2][min(nz, n_xyz[2] - 1) + 1] - nodes[2][min(nz, n_xyz[2] - 1)])
                d_loc = (dx_loc, dy_loc, dz_loc)

                # Incident primary E edges at node (nx, ny, nz): along each axis c,
                # edges (nx, ny, nz) and (nx - delta_c, ny, nz) touch this node.
                for c_ax in range(3):
                    p_ax, q_ax = _INPLANE_AXES[c_ax]
                    d_perp = 0.5 * (d_loc[p_ax] + d_loc[q_ax])
                    m_node = 0.5 * np.log(d_perp / radius)
                    w_node = (2.0 / float(deg)) * m_node
                    for offset in (0, -1):
                        e_ijk = [nx, ny, nz]
                        e_ijk[c_ax] += offset
                        if not (0 <= e_ijk[c_ax] < n_xyz[c_ax] - 1):
                            continue
                        e_flat = int(np.ravel_multi_index(e_ijk, n_xyz, order="F") + c_ax * self.N)
                        if e_flat in wire_flat_set:
                            continue
                        if ieps_host[e_flat] == 0.0:
                            continue
                        ieps_new[e_flat] = ieps_host[e_flat] * w_node
                        # Pair the 4 primary faces containing this scaled E edge so
                        # mu_eff * eps_eff >= mu_0 * eps_0 (preserving the CFL limit)
                        for f_ax, o_ax in ((p_ax, q_ax), (q_ax, p_ax)):
                            for f_off in (0, -1):
                                f_ijk = list(e_ijk)
                                f_ijk[o_ax] += f_off
                                if 0 <= f_ijk[o_ax] < n_xyz[o_ax]:
                                    f_flat = int(
                                        np.ravel_multi_index(f_ijk, n_xyz, order="F") + f_ax * self.N
                                    )
                                    imu_scale[f_flat] = min(imu_scale[f_flat], 1.0 / w_node)

        # Apply PEC or lossy wire conductivity on active wire segments
        if not is_lossy:
            B = getattr(self, "itDaiDepsDstC", None) if not self.activate_cpml else None
            for flat_e in wire_flat_active:
                b_row = None
                if B is not None and getattr(B, "format", "csr") == "csr":
                    s0, s1 = int(B.indptr[flat_e]), int(B.indptr[flat_e + 1])
                    b_row = _host(B.data[s0:s1]).copy()
                self._thin_wire_pec_edges[flat_e] = (float(ieps_host[flat_e]), b_row)
                ieps_new[flat_e] = 0.0

        # Commit ieps and imu changes and update solver operators
        xp = self.E.xp
        mod_e = np.flatnonzero(ieps_new != ieps_host)
        if mod_e.size:
            self.ieps.array[xp.asarray(mod_e)] = xp.asarray(ieps_new[mod_e], dtype=self.dtype)
            pec_e = mod_e[ieps_new[mod_e] == 0.0]
            if pec_e.size:
                self.E.array[xp.asarray(pec_e)] = 0.0
            if not self.activate_cpml and hasattr(self, "itDaiDepsDstC") and self.itDaiDepsDstC is not None:
                scale_e = np.ones(3 * self.N, dtype=self.dtype)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_e = np.where(
                        ieps_host[mod_e] != 0.0, ieps_new[mod_e] / ieps_host[mod_e], 0.0
                    )
                scale_e[mod_e] = ratio_e
                _scale_operator_rows(self.itDaiDepsDstC, xp.asarray(scale_e), xp)
                if hasattr(self, "iDeps"):
                    self.iDeps = diags(_host(self.ieps.toarray()), dtype=self.dtype)

        mod_h = np.flatnonzero(imu_scale != 1.0)
        if mod_h.size:
            imu_new = imu_host * imu_scale
            self.imu.array[xp.asarray(mod_h)] = xp.asarray(imu_new[mod_h], dtype=self.dtype)
            if not self.activate_cpml and hasattr(self, "tDsiDmuiDaC") and self.tDsiDmuiDaC is not None:
                scale_h = np.ones(3 * self.N, dtype=self.dtype)
                scale_h[mod_h] = imu_scale[mod_h]
                _scale_operator_rows(self.tDsiDmuiDaC, xp.asarray(scale_h), xp)
                if hasattr(self, "iDmu"):
                    self.iDmu = diags(_host(self.imu.toarray()), dtype=self.dtype)

        # Register semi-implicit conductance on lossy wire segments
        if is_lossy and wire_flat_active:
            idx_lossy = np.asarray(wire_flat_active, dtype=int)
            itA_wire = self.grid.itA.toarray()[idx_lossy]
            lossy_mat = DispersiveMaterial(1.0, [], sigma=1.0 / float(R_per_m), name="lossy_wire")
            self._register_sheet_group(lossy_mat, idx_lossy, itA_wire)

        info = {
            "points": pts,
            "radius": radius,
            "edges": wire_edges,
            "wire_flat_idx": np.asarray(wire_flat_all, dtype=int),
            "active_flat_idx": np.asarray(wire_flat_active, dtype=int),
            "gap_flat_idx": np.asarray(sorted(gap_flat_set), dtype=int),
            "modified_h_idx": mod_h,
            "modified_e_idx": mod_e,
            "node_degrees": node_degrees,
            "is_lossy": is_lossy,
            "R_per_m": R_per_m,
        }
        if not hasattr(self, "_thin_wires"):
            self._thin_wires = []
        self._thin_wires.append(info)
        return info

    def _reopen_thin_wire_edges(self, flat_indices):
        """Restore background ``ieps`` on thin-wire edges where a lumped element is attached."""
        pec_map = getattr(self, "_thin_wire_pec_edges", None)
        if not pec_map:
            return
        xp = self.E.xp
        B = getattr(self, "itDaiDepsDstC", None) if not self.activate_cpml else None
        reopened = False
        for flat_e in np.asarray(flat_indices, dtype=int).ravel():
            entry = pec_map.pop(int(flat_e), None)
            if entry is None:
                continue
            ieps_val, b_row = entry
            self.ieps.array[int(flat_e)] = ieps_val
            if B is not None and b_row is not None and getattr(B, "format", "csr") == "csr":
                s0, s1 = int(B.indptr[int(flat_e)]), int(B.indptr[int(flat_e) + 1])
                B.data[s0:s1] = xp.asarray(b_row, dtype=self.dtype)
            reopened = True
        if reopened and not self.activate_cpml and hasattr(self, "iDeps"):
            self.iDeps = diags(_host(self.ieps.toarray()), dtype=self.dtype)

