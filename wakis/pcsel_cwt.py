# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Three-dimensional coupled-wave theory (3D-CWT) for photonic-crystal
surface-emitting lasers (PCSELs).

This module implements the 3D coupled-wave formalism developed by Noda and
collaborators for square-lattice TE-polarized PCSELs:

- 1D vertical TE slab waveguide solver for the fundamental guided profile
  :math:`\\Theta_0(z)`, effective refractive index :math:`n_{\\text{eff}}`, and
  layer optical confinement factors :math:`\\Gamma_{\\text{PhC}}, \\Gamma_{\\text{act}}`.
- 2D in-plane Fourier permittivity harmonics :math:`\\xi_{m,n}` in the
  photonic-crystal (PhC) layer for single circular air holes, shifted
  double-lattice hole pairs, and general 2D unit-cell permittivity maps.
- 1D vertical Green's function boundary-value solvers for the :math:`(0,0)`
  radiative diffraction order (with outgoing Sommerfeld radiation boundary
  conditions) and higher-order :math:`(m,n)` evanescent diffraction waves.
- Construction of the :math:`4\\times 4` 3D-CWT coupling matrix
  :math:`\\mathbf{C} = \\mathbf{C}_{\\text{rad}} + \\mathbf{C}_{1\\text{D}} + \\mathbf{C}_{2\\text{D}}`
  for the four basic waves :math:`\\mathbf{v} = (R_x, S_x, R_y, S_y)^T`,
  infinite-crystal :math:`\\Gamma`-point and :math:`\\mathbf{k}_\\parallel` band-edge
  eigenmodes (modes A, B, C, D), and double-lattice Hermitian/non-Hermitian
  coupling cancellation analysis.
- 2D finite-size PCSEL eigenproblem solver on :math:`[-L/2, L/2]^2` using
  Gauss--Lobatto--Legendre (GLL) Summation-By-Parts (SBP) collocation with
  upwind open boundary conditions, yielding threshold gain
  :math:`\\alpha_{\\text{th}} = \\alpha_v + \\alpha_\\parallel` and spatial mode
  envelopes.

References
----------
- K. Sakai, E. Miyai, and S. Noda, "Coupled-wave theory for square-lattice
  photonic crystal lasers with TE polarization," *IEEE J. Quantum Electron.*
  **46**(5), 788--795 (2010), doi:10.1109/JQE.2009.2037597.
- Y. Liang, C. Peng, K. Sakai, S. Iwahashi, and S. Noda, "Three-dimensional
  coupled-wave model for square-lattice photonic crystal lasers with transverse
  electric polarization: A general approach," *Phys. Rev. B* **84**(19),
  195119 (2011), doi:10.1103/PhysRevB.84.195119.
- Y. Liang, C. Peng, K. Sakai, S. Iwahashi, and S. Noda, "Three-dimensional
  coupled-wave analysis for square-lattice photonic crystal surface emitting
  lasers with transverse-electric polarization: finite-size effects,"
  *Opt. Express* **20**(14), 15945--15961 (2012), doi:10.1364/OE.20.015945.
- M. Yoshida, M. De Zoysa, K. Ishizaki, Y. Tanaka, M. Kawasaki, R. Hatsuda,
  B. Song, J. Gelleta, and S. Noda, "Double-lattice photonic-crystal resonators
  enabling high-brightness semiconductor lasers with symmetric
  narrow-divergence beams," *Nat. Mater.* **18**, 121--128 (2019),
  doi:10.1038/s41563-018-0242-y.
- T. Inoue, M. Yoshida, J. Gelleta, K. Izumi, K. Yoshida, K. Ishizaki,
  M. De Zoysa, and S. Noda, "General recipe to realize photonic-crystal
  surface-emitting lasers with 100-W-to-1-kW single-mode operation,"
  *Nat. Commun.* **13**, 3262 (2022), doi:10.1038/s41467-022-30910-7.
- M. Yoshida, S. Katsuno, T. Inoue, J. Gelleta, K. Izumi, M. De Zoysa,
  K. Ishizaki, and S. Noda, "High-brightness scalable continuous-wave
  single-mode photonic-crystal laser," *Nature* **618**, 727--732 (2023),
  doi:10.1038/s41586-023-06059-8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre
from scipy.constants import c as C_LIGHT
from scipy.linalg import eigh_tridiagonal, solve_banded
from scipy.optimize import brentq, root
from scipy.special import j1

__all__ = [
    "PCSELLayerStack",
    "compute_cwt_band_structure",
    "compute_cwt_coupling_coefficients",
    "double_lattice_coupling_scan",
    "fourier_coefficients_phc",
    "solve_finite_cwt",
    "solve_infinite_cwt",
    "solve_vertical_greens_functions",
    "solve_vertical_slab_mode",
]


@dataclass(frozen=True)
class PCSELLayerStack:
    """Vertical epitaxial layer stack and unit-cell parameters of a PCSEL.

    The vertical domain :math:`z \\in [z_{\\min}, z_{\\max}]` consists of:

    - Lower cladding (:math:`z < z_{\\text{act,min}}`): refractive index ``n_clad``.
    - Active / waveguide layer (:math:`z_{\\text{act,min}} \\le z < z_{\\text{act,max}}`):
      refractive index ``n_active``.
    - Photonic-crystal layer (:math:`z_{\\text{phc,min}} \\le z \\le z_{\\text{phc,max}}`):
      host semiconductor index ``n_phc_host`` perforated by holes of index
      ``n_hole``.
    - Upper cladding (:math:`z > z_{\\text{phc,max}}`): refractive index ``n_clad``.

    Parameters
    ----------
    a : float
        Square-lattice period in meters (m).
    r_over_a : float
        Primary circular hole radius normalized by the lattice period
        (:math:`0 \\le r/a < 0.5`).
    n_clad : float
        Refractive index of the upper and lower cladding layers.
    n_active : float
        Refractive index of the active / waveguide core layer.
    n_phc_host : float
        Refractive index of the semiconductor host in the PhC layer.
    n_hole : float, optional
        Refractive index inside the etched holes (default ``1.0`` for air).
    z_act_min : float, optional
        Lower z-boundary of the active layer in meters (default ``-150e-9``).
    z_act_max : float, optional
        Upper z-boundary of the active layer in meters (default ``-30e-9``).
    z_phc_min : float, optional
        Lower z-boundary of the PhC layer in meters (default ``-30e-9``).
    z_phc_max : float, optional
        Upper z-boundary of the PhC layer in meters (default ``150e-9``).
    z_min : float, optional
        Lower truncation coordinate of the 1D vertical solver domain in meters
        (default ``-1400e-9``).
    z_max : float, optional
        Upper truncation coordinate of the 1D vertical solver domain in meters
        (default ``1400e-9``).
    r2_over_a : float, optional
        Secondary circular hole radius normalized by ``a`` for a double-lattice
        PCSEL (default ``0.0`` for a single-lattice PCSEL).
    dx_over_a : float, optional
        In-plane x-separation between the primary and secondary holes normalized
        by ``a`` (default ``0.25``).
    dy_over_a : float, optional
        In-plane y-separation between the primary and secondary holes normalized
        by ``a`` (default ``0.25``).
    """

    a: float = 290e-9
    r_over_a: float = 0.18
    n_clad: float = 3.26
    n_active: float = 3.52
    n_phc_host: float = 3.46
    n_hole: float = 1.0
    z_act_min: float = -150e-9
    z_act_max: float = -30e-9
    z_phc_min: float = -30e-9
    z_phc_max: float = 150e-9
    z_min: float = -1400e-9
    z_max: float = 1400e-9
    r2_over_a: float = 0.0
    dx_over_a: float = 0.25
    dy_over_a: float = 0.25

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"Lattice constant a must be positive, got {self.a}.")
        if not (0.0 <= self.r_over_a < 0.5):
            raise ValueError(f"r_over_a must be in [0, 0.5), got {self.r_over_a}.")
        if not (0.0 <= self.r2_over_a < 0.5):
            raise ValueError(f"r2_over_a must be in [0, 0.5), got {self.r2_over_a}.")
        if not (self.z_min < self.z_act_min < self.z_act_max <= self.z_phc_min < self.z_phc_max < self.z_max):
            raise ValueError(
                "Vertical coordinates must satisfy z_min < z_act_min < z_act_max <= z_phc_min < z_phc_max < z_max."
            )

    @property
    def fill_factor(self) -> float:
        """Total in-plane areal fill factor :math:`f = \\pi (r_1^2 + r_2^2) / a^2`."""
        return float(np.pi * (self.r_over_a**2 + self.r2_over_a**2))

    @property
    def eps_clad(self) -> float:
        """Relative permittivity of the cladding layers."""
        return float(self.n_clad**2)

    @property
    def eps_active(self) -> float:
        """Relative permittivity of the active layer."""
        return float(self.n_active**2)

    @property
    def eps_phc_host(self) -> float:
        """Relative permittivity of the PhC semiconductor host."""
        return float(self.n_phc_host**2)

    @property
    def eps_hole(self) -> float:
        """Relative permittivity inside the etched holes."""
        return float(self.n_hole**2)

    @property
    def eps_phc_avg(self) -> float:
        """Zero-order average permittivity :math:`\\varepsilon_0` in the PhC layer."""
        f = self.fill_factor
        return float((1.0 - f) * self.eps_phc_host + f * self.eps_hole)


def fourier_coefficients_phc(
    stack: PCSELLayerStack,
    m_max: int = 3,
    eps_2d: np.ndarray | None = None,
    align_radiative_center: bool = True,
) -> dict[tuple[int, int], complex]:
    """Compute 2D Fourier coefficients :math:`\\xi_{m,n}` of :math:`\\varepsilon(x,y)` in the PhC layer.

    The relative permittivity in the PhC unit cell :math:`[-a/2, a/2]^2` is
    expanded as

    .. math::

        \\varepsilon(x, y) = \\varepsilon_0 + \\sum_{(m,n) \\neq (0,0)}
        \\xi_{m,n} e^{-i(m G_0 x + n G_0 y)}, \\qquad G_0 = \\frac{2\\pi}{a},

    where

    .. math::

        \\xi_{m,n} = \\frac{1}{a^2} \\iint_{[-a/2,a/2]^2}
        \\varepsilon(x,y) e^{+i(m G_0 x + n G_0 y)}\\,\\mathrm{d}x\\,\\mathrm{d}y.

    For circular holes of radii :math:`r_1, r_2` separated by
    :math:`(d_x, d_y)`, when ``align_radiative_center=True`` (default), the
    unit-cell coordinate origin :math:`(x_0, y_0)` is placed at the radiative
    symmetry center where the first-order harmonics :math:`\\xi_{1,0}` and
    :math:`\\xi_{0,1}` are real and negative (the canonical CWT reference frame
    of Sakai *et al.* 2010, Yoshida *et al.* 2019, and Inoue *et al.* 2022).

    Parameters
    ----------
    stack : PCSELLayerStack
        PCSEL layer stack and unit-cell specification.
    m_max : int, optional
        Maximum absolute harmonic order :math:`|m|, |n| \\le m_{\\max}` (default ``3``).
    eps_2d : numpy.ndarray, optional
        Optional 2D permittivity array sampled on a uniform grid over
        :math:`[-a/2, a/2)^2`. When provided, :math:`\\xi_{m,n}` is computed by
        2D discrete Fourier transform instead of the analytical Bessel formula.
    align_radiative_center : bool, optional
        If ``True`` (default), aligns the unit-cell phase origin with the
        fundamental radiative center where :math:`\\xi_{1,0}` and :math:`\\xi_{0,1}`
        are real and negative.

    Returns
    -------
    dict of {(int, int): complex}
        Mapping from harmonic index pair ``(m, n)`` to complex Fourier
        coefficient :math:`\\xi_{m,n}`.
    """
    if m_max < 1:
        raise ValueError(f"m_max must be >= 1, got {m_max}.")

    coeffs: dict[tuple[int, int], complex] = {}
    if eps_2d is not None:
        arr = np.asarray(eps_2d, dtype=np.complex128)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("eps_2d must be a square 2D array.")
        n_grid = arr.shape[0]
        coords = -0.5 + np.arange(n_grid) / n_grid
        for m in range(-m_max, m_max + 1):
            phase_x = np.exp(2j * np.pi * m * coords)
            for n in range(-m_max, m_max + 1):
                phase_y = np.exp(2j * np.pi * n * coords)
                val = np.mean(arr * np.outer(phase_x, phase_y))
                coeffs[(m, n)] = complex(val)
        return coeffs

    delta_eps = stack.eps_hole - stack.eps_phc_host
    r1 = stack.r_over_a
    r2 = stack.r2_over_a
    x1, y1 = 0.0, 0.0
    x2, y2 = float(stack.dx_over_a), float(stack.dy_over_a)

    raw: dict[tuple[int, int], complex] = {}
    for m in range(-m_max, m_max + 1):
        for n in range(-m_max, m_max + 1):
            if m == 0 and n == 0:
                raw[(0, 0)] = complex(stack.eps_phc_avg)
                continue
            rho = float(np.hypot(m, n))
            val = 0.0j
            if r1 > 0.0:
                amp1 = delta_eps * (r1 / rho) * float(j1(2.0 * np.pi * rho * r1))
                phase1 = np.exp(2j * np.pi * (m * x1 + n * y1))
                val += amp1 * phase1
            if r2 > 0.0:
                amp2 = delta_eps * (r2 / rho) * float(j1(2.0 * np.pi * rho * r2))
                phase2 = np.exp(2j * np.pi * (m * x2 + n * y2))
                val += amp2 * phase2
            raw[(m, n)] = complex(val)

    if align_radiative_center and r2 > 0.0:
        phi_x = float(np.angle(-raw[(1, 0)]))
        phi_y = float(np.angle(-raw[(0, 1)]))
        for (m, n), val in raw.items():
            coeffs[(m, n)] = complex(val * np.exp(-1j * (m * phi_x + n * phi_y)))
    else:
        coeffs = raw

    return coeffs


def _exact_slab_dispersion_residual(
    n_eff: float,
    beta0: float,
    stack: PCSELLayerStack,
    eps_phc_eff: float,
) -> float:
    """Transfer-matrix characteristic residual for the 1D 4-layer TE slab waveguide."""
    k0 = beta0 / n_eff
    gamma_c = np.sqrt(max(beta0**2 - k0**2 * stack.eps_clad, 0.0))
    k_act_sq = k0**2 * stack.eps_active - beta0**2
    k_phc_sq = k0**2 * eps_phc_eff - beta0**2
    t_act = stack.z_act_max - stack.z_act_min
    t_phc = stack.z_phc_max - stack.z_phc_min

    def layer_matrix(k_sq: float, thickness: float) -> np.ndarray:
        if k_sq >= 0.0:
            kz = np.sqrt(k_sq)
            if kz * thickness < 1e-12:
                return np.array([[1.0, thickness], [-k_sq * thickness, 1.0]])
            c_val = np.cos(kz * thickness)
            s_val = np.sin(kz * thickness)
            return np.array([[c_val, s_val / kz], [-kz * s_val, c_val]])
        alpha = np.sqrt(-k_sq)
        ch_val = np.cosh(alpha * thickness)
        sh_val = np.sinh(alpha * thickness)
        return np.array([[ch_val, sh_val / alpha], [alpha * sh_val, ch_val]])

    m_tot = layer_matrix(k_phc_sq, t_phc) @ layer_matrix(k_act_sq, t_act)
    e_top = m_tot[0, 0] + m_tot[0, 1] * gamma_c
    de_top = m_tot[1, 0] + m_tot[1, 1] * gamma_c
    return float(de_top + gamma_c * e_top)


def solve_vertical_slab_mode(
    stack: PCSELLayerStack,
    nz: int = 2001,
    eps_phc_override: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Solve the 1D vertical TE slab waveguide problem at :math:`\\beta_0 = 2\\pi/a`.

    Solves the self-consistent 1D Helmholtz eigenproblem

    .. math::

        \\left[\\frac{\\mathrm{d}^2}{\\mathrm{d}z^2}
        + k_0^2 \\varepsilon_{\\text{slab}}(z)\\right] \\Theta_0(z)
        = \\beta_0^2 \\Theta_0(z),
        \\qquad k_0 = \\frac{\\beta_0}{n_{\\text{eff}}},

    for the fundamental :math:`\\text{TE}_0` guided vertical profile
    :math:`\\Theta_0(z)` normalized such that
    :math:`\\int_{z_{\\min}}^{z_{\\max}} |\\Theta_0(z)|^2\\,\\mathrm{d}z = 1`.

    Parameters
    ----------
    stack : PCSELLayerStack
        PCSEL vertical layer stack.
    nz : int, optional
        Number of uniform grid points across :math:`[z_{\\min}, z_{\\max}]`
        (default ``2001``).
    eps_phc_override : float, optional
        Optional effective PhC permittivity :math:`\\varepsilon_{\\text{PhC,eff}}`
        used to evaluate mode-dressed vertical confinement (default ``None``,
        which uses the zero-order average :math:`\\varepsilon_0`).

    Returns
    -------
    dict
        Dictionary containing ``z``, ``dz``, ``eps_z``, ``theta0``, ``n_eff``,
        ``n_eff_exact``, ``k0``, ``beta0``, ``f_bragg``, ``gamma_phc``, and
        ``gamma_act``.
    """
    if nz < 201:
        raise ValueError(f"nz must be >= 201, got {nz}.")

    beta0 = 2.0 * np.pi / stack.a
    eps_phc = float(stack.eps_phc_avg if eps_phc_override is None else eps_phc_override)

    z = np.linspace(stack.z_min, stack.z_max, nz)
    dz = float(z[1] - z[0])
    eps_z = np.full(nz, stack.eps_clad, dtype=np.float64)
    act_mask = (z >= stack.z_act_min) & (z < stack.z_act_max)
    phc_mask = (z >= stack.z_phc_min) & (z <= stack.z_phc_max)
    eps_z[act_mask] = stack.eps_active
    eps_z[phc_mask] = eps_phc

    n_eff = 0.5 * (stack.n_clad + stack.n_active)
    off_diag = np.full(nz - 1, 1.0 / dz**2, dtype=np.float64)
    eigvec = np.zeros(nz, dtype=np.float64)
    for _ in range(12):
        k0 = beta0 / n_eff
        diag = -2.0 / dz**2 + k0**2 * eps_z
        eigvals, eigvecs = eigh_tridiagonal(
            diag,
            off_diag,
            select="i",
            select_range=(nz - 1, nz - 1),
        )
        beta_sq = float(eigvals[0])
        eigvec = eigvecs[:, 0]
        n_eff = float(np.sqrt(beta_sq) / k0)

    if eigvec[np.argmax(np.abs(eigvec))] < 0:
        eigvec = -eigvec
    norm = float(np.sqrt(np.sum(eigvec**2) * dz))
    theta0 = eigvec / norm

    gamma_phc = float(np.sum(theta0[phc_mask] ** 2) * dz)
    gamma_act = float(np.sum(theta0[act_mask] ** 2) * dz)

    n_min = stack.n_clad + 1e-7
    n_max = max(stack.n_active, np.sqrt(max(eps_phc, stack.eps_clad))) - 1e-7
    n_eff_exact = float(
        brentq(
            _exact_slab_dispersion_residual,
            n_min,
            n_max,
            args=(beta0, stack, eps_phc),
            xtol=1e-12,
        )
    )

    k0 = beta0 / n_eff
    f_bragg = float(C_LIGHT * k0 / (2.0 * np.pi))

    return {
        "z": z,
        "dz": dz,
        "eps_z": eps_z,
        "theta0": theta0,
        "n_eff": n_eff,
        "n_eff_exact": n_eff_exact,
        "k0": float(k0),
        "beta0": float(beta0),
        "f_bragg": f_bragg,
        "gamma_phc": gamma_phc,
        "gamma_act": gamma_act,
        "phc_mask": phc_mask,
        "act_mask": act_mask,
    }


def solve_vertical_greens_functions(
    stack: PCSELLayerStack,
    slab: dict[str, float | np.ndarray] | None = None,
    m_max: int = 3,
) -> dict[str, complex | float | np.ndarray | dict]:
    """Solve the 1D vertical Green's function boundary-value problems for 3D-CWT.

    Uses :func:`scipy.linalg.solve_banded` on the tridiagonal Helmholtz systems
    for the :math:`(0,0)` radiative order (with outgoing Sommerfeld BCs) and all
    higher-order :math:`(m,n)` evanescent diffraction waves with
    :math:`s = m^2 + n^2 > 1` up to :math:`|m|, |n| \\le m_{\\max}`. By discrete
    summation by parts, the radiative overlap
    :math:`\\mathcal{I}_{\\text{rad}} = \\int_{\\text{PhC}} \\Theta_0(z)\\Phi_{\\text{rad}}(z)\\,\\mathrm{d}z`
    satisfies the exact optical theorem

    .. math::

        -\\text{Im}(\\mathcal{I}_{\\text{rad}}) = \\frac{n_{\\text{clad}}}{k_0}
        \\left(|\\Phi_{\\text{rad}}(z_{\\max})|^2 + |\\Phi_{\\text{rad}}(z_{\\min})|^2\\right) > 0

    to machine precision.

    Parameters
    ----------
    stack : PCSELLayerStack
        PCSEL vertical layer stack.
    slab : dict, optional
        Output dictionary from :func:`solve_vertical_slab_mode`.
    m_max : int, optional
        Maximum Fourier harmonic order for evanescent diffraction modes
        (default ``3``).

    Returns
    -------
    dict
        Dictionary containing ``phi_rad``, ``phi_11``, ``i_rad``, ``i_11``,
        ``chi_rad``, ``chi_11``, ``phi_evan``, ``i_evan``, ``chi_evan``,
        ``optical_theorem_lhs``, and ``optical_theorem_rhs``.
    """
    if slab is None:
        slab = solve_vertical_slab_mode(stack)

    z = np.asarray(slab["z"], dtype=np.float64)
    dz = float(slab["dz"])
    eps_z = np.asarray(slab["eps_z"], dtype=np.float64)
    theta0 = np.asarray(slab["theta0"], dtype=np.float64)
    phc_mask = np.asarray(slab["phc_mask"], dtype=bool)
    k0 = float(slab["k0"])
    beta0 = float(slab["beta0"])
    gamma_phc = float(slab["gamma_phc"])
    nz = len(z)

    rhs = np.zeros(nz, dtype=np.complex128)
    rhs[phc_mask] = -(k0**2) * theta0[phc_mask]

    # 1. Radiative (0,0) tridiagonal system in banded storage (3, nz)
    kz_clad = k0 * stack.n_clad
    ab_rad = np.zeros((3, nz), dtype=np.complex128)
    ab_rad[0, 1:] = 1.0 / dz**2
    ab_rad[0, 1] = 2.0 / dz**2
    ab_rad[1, :] = -2.0 / dz**2 + k0**2 * eps_z
    ab_rad[1, 0] = (-2.0 - 2j * dz * kz_clad) / dz**2 + k0**2 * eps_z[0]
    ab_rad[1, -1] = (-2.0 - 2j * dz * kz_clad) / dz**2 + k0**2 * eps_z[-1]
    ab_rad[2, :-1] = 1.0 / dz**2
    ab_rad[2, -2] = 2.0 / dz**2

    phi_rad = solve_banded((1, 1), ab_rad, rhs)
    i_rad = complex(np.sum(theta0[phc_mask] * phi_rad[phc_mask]) * dz)
    chi_rad = i_rad / gamma_phc

    opt_lhs = float(-np.imag(i_rad))
    opt_rhs = float((kz_clad / k0**2) * (np.abs(phi_rad[0]) ** 2 + np.abs(phi_rad[-1]) ** 2))

    # 2. Higher-order evanescent (m, n) systems indexed by s = m^2 + n^2 > 1
    s_values = sorted({m * m + n * n for m in range(-m_max, m_max + 1) for n in range(-m_max, m_max + 1) if m * m + n * n > 1})
    if 2 not in s_values:
        s_values = [2, *s_values]

    phi_evan: dict[int, np.ndarray] = {}
    i_evan: dict[int, float] = {}
    chi_evan: dict[int, float] = {}
    rhs_real = np.real(rhs)

    for s in s_values:
        k_mn_sq = float(s) * beta0**2
        gamma_mn = np.sqrt(max(k_mn_sq - k0**2 * stack.eps_clad, 0.0))
        ab_ev = np.zeros((3, nz), dtype=np.float64)
        ab_ev[0, 1:] = 1.0 / dz**2
        ab_ev[0, 1] = 2.0 / dz**2
        ab_ev[1, :] = -2.0 / dz**2 + k0**2 * eps_z - k_mn_sq
        ab_ev[1, 0] = (-2.0 - 2.0 * dz * gamma_mn) / dz**2 + k0**2 * eps_z[0] - k_mn_sq
        ab_ev[1, -1] = (-2.0 - 2.0 * dz * gamma_mn) / dz**2 + k0**2 * eps_z[-1] - k_mn_sq
        ab_ev[2, :-1] = 1.0 / dz**2
        ab_ev[2, -2] = 2.0 / dz**2

        phi_s = solve_banded((1, 1), ab_ev, rhs_real)
        i_s = float(np.sum(theta0[phc_mask] * phi_s[phc_mask]) * dz)
        phi_evan[s] = phi_s
        i_evan[s] = i_s
        chi_evan[s] = i_s / gamma_phc

    return {
        "phi_rad": phi_rad,
        "phi_11": phi_evan[2],
        "i_rad": i_rad,
        "i_11": i_evan[2],
        "chi_rad": chi_rad,
        "chi_11": chi_evan[2],
        "phi_evan": phi_evan,
        "i_evan": i_evan,
        "chi_evan": chi_evan,
        "optical_theorem_lhs": opt_lhs,
        "optical_theorem_rhs": opt_rhs,
    }


def compute_cwt_coupling_coefficients(
    stack: PCSELLayerStack,
    m_max: int = 3,
    eps_2d: np.ndarray | None = None,
    slab: dict[str, float | np.ndarray] | None = None,
    greens: dict[str, complex | float | np.ndarray | dict] | None = None,
    include_longitudinal: bool = False,
    bragg_phase_rad: float = 0.0,
) -> dict[str, complex | float | np.ndarray | dict]:
    """Assemble the 3D-CWT :math:`4\\times 4` coupling matrix :math:`\\mathbf{C}`.

    For the basic wave vector :math:`\\mathbf{v} = (R_x, S_x, R_y, S_y)^T` with
    in-plane wavevectors :math:`\\beta_0(m_\\alpha, n_\\alpha) \\in \\{\\beta_0(\\pm 1,0), \\beta_0(0,\\pm 1)\\}`
    and TE electric-field unit vectors
    :math:`\\hat{\\mathbf{e}}_0 = \\hat{\\mathbf{e}}_1 = \\hat{\\mathbf{y}}` and
    :math:`\\hat{\\mathbf{e}}_2 = \\hat{\\mathbf{e}}_3 = \\hat{\\mathbf{x}}`
    (Liang *et al.*, *Phys. Rev. B* **84**, 195119 (2011)), the 3D coupled-wave
    matrix at the :math:`\\Gamma` point is

    .. math::

        \\mathbf{C} = \\mathbf{C}_{\\text{rad}} + \\mathbf{C}_{1\\text{D}} + \\mathbf{C}_{2\\text{D}},

    where:

    - :math:`(\\mathbf{C}_{1\\text{D}})_{\\alpha\\beta} = -\\frac{k_0^2}{2\\beta_0}\\Gamma_{\\text{PhC}}(\\hat{\\mathbf{e}}_\\alpha\\cdot\\hat{\\mathbf{e}}_\\beta)\\xi_{m_\\alpha - m_\\beta, n_\\alpha - n_\\beta}`:
      direct 1D backward Bragg coupling via :math:`\\xi_{\\pm 2,0}` and :math:`\\xi_{0,\\pm 2}`
      (:math:`\\kappa_{1\\text{D}} = -\\frac{k_0^2}{2\\beta_0}\\Gamma_{\\text{PhC}}\\xi_{2,0} > 0` for a
      circular air hole at the origin).
    - :math:`(\\mathbf{C}_{\\text{rad}})_{\\alpha\\beta} = -\\frac{k_0^2}{2\\beta_0}\\mathcal{I}_{\\text{rad}}(\\hat{\\mathbf{e}}_\\alpha\\cdot\\hat{\\mathbf{e}}_\\beta)\\xi_{m_\\alpha, n_\\alpha}\\xi_{-m_\\beta, -n_\\beta}`:
      complex vertical radiative coupling mediated by the :math:`(0,0)` diffraction order.
    - :math:`(\\mathbf{C}_{2\\text{D}}^{(T)})_{\\alpha\\beta} = -\\frac{k_0^2}{2\\beta_0}\\sum_{m^2+n^2 > 1} \\mathcal{I}_{m,n}(\\hat{\\mathbf{e}}_\\alpha\\cdot\\hat{\\mathbf{t}}_{m,n})(\\hat{\\mathbf{t}}_{m,n}\\cdot\\hat{\\mathbf{e}}_\\beta)\\xi_{m_\\alpha - m, n_\\alpha - n}\\xi_{m - m_\\beta, n - n_\\beta}`:
      higher-order 2D coupling mediated by transverse TE evanescent diffraction waves with
      in-plane polarization unit vectors :math:`\\hat{\\mathbf{t}}_{m,n} = (n\\hat{\\mathbf{x}} - m\\hat{\\mathbf{y}})/\\sqrt{m^2+n^2}`
      (including two-step orthogonal coupling :math:`R_x \\to (\\pm 1,\\pm 1) \\to R_y, S_y`
      proportional to :math:`\\xi_{0,\\pm 1}\\xi_{\\pm 1,0}`).
    - :math:`(\\mathbf{C}_{2\\text{D}}^{(L)})_{\\alpha\\beta} = +\\frac{k_0^2}{2\\beta_0}\\frac{\\Gamma_{\\text{PhC}}}{\\varepsilon_{\\text{avg}}}\\sum_{m^2+n^2 > 1} (\\hat{\\mathbf{e}}_\\alpha\\cdot\\hat{\\mathbf{l}}_{m,n})(\\hat{\\mathbf{l}}_{m,n}\\cdot\\hat{\\mathbf{e}}_\\beta)\\xi_{m_\\alpha - m, n_\\alpha - n}\\xi_{m - m_\\beta, n - n_\\beta}`:
      an ad hoc local depolarisation term :math:`\\propto \\xi\\xi / \\varepsilon_{\\text{avg}}`
      (not Liang *et al.*'s :math:`\\zeta_{m,n}` form) along
      :math:`\\hat{\\mathbf{l}}_{m,n} = (m\\hat{\\mathbf{x}} + n\\hat{\\mathbf{y}})/\\sqrt{m^2+n^2}`,
      where :math:`\\varepsilon_{\\text{avg}} = \\xi_{0,0}` is ``stack.eps_phc_avg``
      (included in ``C_2D`` when ``include_longitudinal=True``).

    In the 1D sub-block for :math:`(R_x, S_x)`, the total :math:`180^\\circ`
    coupling entries are decomposed in the radiative-center frame following
    Yoshida *et al.* (2019) and Inoue *et al.* (2022) as

    .. math::

        C_{12} = R + i(\\mu + I), \\qquad C_{21} = R + i(\\mu - I),

    where :math:`R + iI = \\frac{1}{2}(C_{12} + C_{21}^*)` is the total Hermitian
    backward coupling (so :math:`R` already contains :math:`\\operatorname{Re}(\\kappa_{1\\text{D}})`
    plus the real radiative and higher-order 1D shifts) and
    :math:`i\\mu = \\frac{1}{2}(C_{12} - C_{21}^*)` is the off-diagonal
    non-Hermitian radiation-mediated coupling.

    Parameters
    ----------
    stack : PCSELLayerStack
        PCSEL layer stack and hole geometry.
    m_max : int, optional
        Maximum diffraction order :math:`|m|, |n| \\le m_{\\max}` in the higher-order
        evanescent sum (default ``3``).
    eps_2d : numpy.ndarray, optional
        Optional 2D permittivity array in the PhC unit cell.
    slab : dict, optional
        Precomputed output from :func:`solve_vertical_slab_mode`.
    greens : dict, optional
        Precomputed output from :func:`solve_vertical_greens_functions`.
    include_longitudinal : bool, optional
        If ``True``, adds an ad hoc local depolarisation term
        :math:`\\propto \\xi\\xi / \\varepsilon_{\\text{avg}}` (not Liang *et al.*'s
        :math:`\\zeta_{m,n}` form), ``C_2D_longitudinal``, into ``C_2D`` and ``C``
        (default ``False``).
    bragg_phase_rad : float, optional
        Optional synthetic Hermitian phase shift :math:`\\phi_{\\text{Bragg}}`
        (in radians) applied directly to :math:`\\xi_{\\pm 2,0}` and
        :math:`\\xi_{0,\\pm 2}` in ``C_1D`` (a synthetic phase on
        :math:`\\xi_{\\pm 2,0}, \\xi_{0,\\pm 2}` tuned to match a target
        :math:`\\alpha_{v,0}`, not a physical hole geometry; default ``0.0``).

    Returns
    -------
    dict
        Dictionary containing ``C`` (in :math:`\\text{m}^{-1}`), ``C_cm`` (in
        :math:`\\text{cm}^{-1}`), ``C_rad``, ``C_1D``, ``C_2D``,
        ``C_2D_transverse``, ``C_2D_longitudinal``, ``xi``, scalar
        coupling constants ``kappa_1d``, ``kappa_2d``, ``kappa_2d_longitudinal``,
        ``kappa_rad`` (in :math:`\\text{cm}^{-1}`), and the Hermitian /
        non-Hermitian components ``R_cm``, ``I_cm``, ``mu_cm``, ``C12_cm``,
        ``C21_cm``.
    """
    if slab is None:
        slab = solve_vertical_slab_mode(stack)
    max_s_needed = 2 * m_max * m_max
    if (
        greens is None
        or "chi_evan" not in greens
        or max(greens["chi_evan"].keys(), default=0) < max_s_needed
    ):
        greens = solve_vertical_greens_functions(stack, slab=slab, m_max=m_max)
    # Compute Fourier harmonics up to m_max + 1 so (m_alpha - m, n_alpha - n) is in range
    xi = fourier_coefficients_phc(stack, m_max=m_max + 1, eps_2d=eps_2d)

    k0 = float(slab["k0"])
    beta0 = float(slab["beta0"])
    gamma_phc = float(slab["gamma_phc"])
    pref = (k0**2 / (2.0 * beta0)) * gamma_phc
    chi_rad = complex(greens["chi_rad"])
    chi_evan = greens["chi_evan"]

    # 1. Direct 1D Bragg coupling from (m,n) = (\pm 2, 0) and (0, \pm 2) in Noda's sign convention
    phase_p = np.exp(1j * float(bragg_phase_rad))
    phase_m = np.exp(-1j * float(bragg_phase_rad))
    c1d_x_12 = -pref * xi[(2, 0)] * phase_p
    c1d_x_21 = -pref * xi[(-2, 0)] * phase_m
    c1d_y_34 = -pref * xi[(0, 2)] * phase_p
    c1d_y_43 = -pref * xi[(0, -2)] * phase_m

    c_1d = np.array(
        [
            [0.0j, c1d_x_12, 0.0j, 0.0j],
            [c1d_x_21, 0.0j, 0.0j, 0.0j],
            [0.0j, 0.0j, 0.0j, c1d_y_34],
            [0.0j, 0.0j, c1d_y_43, 0.0j],
        ],
        dtype=np.complex128,
    )

    # 2. Radiative (0,0) coupling via (\pm 1, 0) for Ey and (0, \pm 1) for Ex
    rad_scale = -pref * chi_rad
    u_x = np.array([xi[(1, 0)], xi[(-1, 0)]], dtype=np.complex128)
    v_x = np.array([xi[(-1, 0)], xi[(1, 0)]], dtype=np.complex128)
    u_y = np.array([xi[(0, 1)], xi[(0, -1)]], dtype=np.complex128)
    v_y = np.array([xi[(0, -1)], xi[(0, 1)]], dtype=np.complex128)

    c_rad = np.zeros((4, 4), dtype=np.complex128)
    c_rad[0:2, 0:2] = rad_scale * np.outer(u_x, v_x)
    c_rad[2:4, 2:4] = rad_scale * np.outer(u_y, v_y)

    # 3. Higher-order 2D evanescent coupling (Liang et al., Phys. Rev. B 84, 195119 (2011))
    # Basic waves: (R_x, S_x, R_y, S_y) with wavevectors (m_a, n_a) and TE unit vectors e_a
    basic_mn = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    e_vecs = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    c_2d_trans = np.zeros((4, 4), dtype=np.complex128)
    c_2d_long = np.zeros((4, 4), dtype=np.complex128)
    scale_long = pref / stack.eps_phc_avg
    for m in range(-m_max, m_max + 1):
        for n in range(-m_max, m_max + 1):
            s = m * m + n * n
            if s <= 1:
                continue
            inv_sqrt_s = 1.0 / np.sqrt(float(s))
            t_mn = np.array([float(n), float(-m)], dtype=np.float64) * inv_sqrt_s
            l_mn = np.array([float(m), float(n)], dtype=np.float64) * inv_sqrt_s
            pol_proj_t = e_vecs @ t_mn
            pol_proj_l = e_vecs @ l_mn
            scale_mn = -pref * float(chi_evan[s])
            xi_left = [xi[(basic_mn[a][0] - m, basic_mn[a][1] - n)] for a in range(4)]
            xi_right = [xi[(m - basic_mn[b][0], n - basic_mn[b][1])] for b in range(4)]
            u_mn_t = np.array([pol_proj_t[a] * xi_left[a] for a in range(4)], dtype=np.complex128)
            v_mn_t = np.array([pol_proj_t[b] * xi_right[b] for b in range(4)], dtype=np.complex128)
            u_mn_l = np.array([pol_proj_l[a] * xi_left[a] for a in range(4)], dtype=np.complex128)
            v_mn_l = np.array([pol_proj_l[b] * xi_right[b] for b in range(4)], dtype=np.complex128)
            c_2d_trans += scale_mn * np.outer(u_mn_t, v_mn_t)
            c_2d_long += scale_long * np.outer(u_mn_l, v_mn_l)

    c_2d = c_2d_trans + c_2d_long if include_longitudinal else c_2d_trans
    c_total = c_rad + c_1d + c_2d
    c_cm = c_total * 1e-2

    # Radiative-center phase gauge where xi_{1,0} is negative real:
    # Let phi_1 = arg(-xi_{1,0}); under R_x -> R_x e^{i phi_1}, S_x -> S_x e^{-i phi_1},
    # C12 -> C12 e^{-2i phi_1} = R + i*(mu + I) and C21 -> C21 e^{+2i phi_1} = R + i*(mu - I).
    phi_1 = float(np.angle(-xi[(1, 0)]))
    c12_rot = complex(c_cm[0, 1] * np.exp(-2j * phi_1))
    c21_rot = complex(c_cm[1, 0] * np.exp(2j * phi_1))
    h12_rot = 0.5 * (c12_rot + np.conj(c21_rot))
    ah12_rot = (c12_rot - np.conj(c21_rot)) / 2j

    r_cm = float(np.real(h12_rot))
    i_cm = float(np.imag(h12_rot))
    mu_cm = float(np.real(ah12_rot))
    mu_diag_cm = float(np.imag(c_rad[0, 0]) * 1e-2)
    mu_12_cm = complex(c_rad[0, 1] * 1e-2)
    mu_21_cm = complex(c_rad[1, 0] * 1e-2)

    return {
        "C": c_total,
        "C_cm": c_cm,
        "C_rad": c_rad,
        "C_1D": c_1d,
        "C_2D": c_2d,
        "C_2D_transverse": c_2d_trans,
        "C_2D_longitudinal": c_2d_long,
        "xi": xi,
        "slab": slab,
        "greens": greens,
        "kappa_1d": complex(c_1d[0, 1] * 1e-2),
        "kappa_2d": complex(c_2d[0, 2] * 1e-2),
        "kappa_2d_transverse": complex(c_2d_trans[0, 2] * 1e-2),
        "kappa_2d_longitudinal": complex(c_2d_long[0, 2] * 1e-2),
        "kappa_rad": mu_diag_cm,
        "R_cm": r_cm,
        "I_cm": i_cm,
        "mu_cm": mu_cm,
        "mu_12_cm": mu_12_cm,
        "mu_21_cm": mu_21_cm,
        "C12_cm": complex(c_cm[0, 1]),
        "C21_cm": complex(c_cm[1, 0]),
    }


def solve_infinite_cwt(
    stack: PCSELLayerStack,
    kx: float = 0.0,
    ky: float = 0.0,
    cwt_data: dict | None = None,
    dressed_vertical_profile: bool = False,
    include_longitudinal: bool = False,
) -> dict[str, dict | np.ndarray | float]:
    """Solve the :math:`4\\times 4` infinite-crystal 3D-CWT eigenproblem at :math:`(k_x, k_y)`.

    Parameters
    ----------
    stack : PCSELLayerStack
        PCSEL layer stack and hole geometry.
    kx, ky : float, optional
        In-plane Bloch wavevector components relative to :math:`\\Gamma` in
        :math:`\\text{m}^{-1}` (default ``0.0``).
    cwt_data : dict, optional
        Precomputed output from :func:`compute_cwt_coupling_coefficients`.
    dressed_vertical_profile : bool, optional
        If ``True``, recomputes the vertical slab profile :math:`\\Theta_\\nu(z)`
        for the radiative modes C/D using the mode-dressed effective PhC
        permittivity :math:`\\varepsilon_{\\text{PhC,CD}} = \\varepsilon_0 - |\\xi_{2,0}|`
        to capture high-fill-factor vertical profile expulsion (default ``False``).
    include_longitudinal : bool, optional
        If ``True`` (and ``cwt_data`` is ``None``), includes the ad hoc local
        depolarisation term :math:`\\propto \\xi\\xi / \\varepsilon_{\\text{avg}}`
        (not Liang *et al.*'s :math:`\\zeta_{m,n}` form) in ``C_2D``
        (default ``False``).

    Returns
    -------
    dict
        Dictionary mapping ``"A"``, ``"B"``, ``"C"``, ``"D"`` to mode properties
        plus ``frequencies_hz`` and ``Q_factors`` arrays ordered ``[A, B, C, D]``.
    """
    if cwt_data is None:
        cwt_data = compute_cwt_coupling_coefficients(stack, include_longitudinal=include_longitudinal)

    slab = cwt_data["slab"]
    beta0 = float(slab["beta0"])
    f_bragg = float(slab["f_bragg"])
    c_mat = np.asarray(cwt_data["C"], dtype=np.complex128)

    k_diag = np.diag(np.array([kx, -kx, ky, -ky], dtype=np.complex128))
    h_mat = k_diag + c_mat
    evals, evecs = np.linalg.eig(h_mat)

    ref_vecs = np.array(
        [
            [0.5, -0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0],
            [0.0, 0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
        ],
        dtype=np.complex128,
    )
    labels = ["A", "B", "C", "D"]
    irreps = {"A": "A2", "B": "B2", "C": "E(1)", "D": "E(2)"}

    overlaps = np.abs(ref_vecs.conj() @ evecs) ** 2
    assigned: dict[str, int] = {}
    used: set[int] = set()
    for r_idx, lbl in enumerate(["A", "B"]):
        scores = overlaps[r_idx].copy()
        for u in used:
            scores[u] = -1.0
        best = int(np.argmax(scores))
        assigned[lbl] = best
        used.add(best)

    rem = [i for i in range(4) if i not in used]
    if abs(evals[rem[0]] - evals[rem[1]]) < 1e-6 * beta0 and abs(kx) + abs(ky) == 0.0:
        sub = evecs[:, rem]
        q_orth, _ = np.linalg.qr(sub)
        v_c = q_orth @ (q_orth.conj().T @ ref_vecs[2])
        v_c /= np.linalg.norm(v_c)
        v_d = q_orth @ (q_orth.conj().T @ ref_vecs[3])
        v_d /= np.linalg.norm(v_d)
        evecs[:, rem[0]] = v_c
        evecs[:, rem[1]] = v_d
        assigned["C"] = rem[0]
        assigned["D"] = rem[1]
    else:
        if np.real(evals[rem[0]]) <= np.real(evals[rem[1]]):
            assigned["C"] = rem[0]
            assigned["D"] = rem[1]
        else:
            assigned["C"] = rem[1]
            assigned["D"] = rem[0]

    dressed_cwt_cd = None
    if dressed_vertical_profile:
        xi_20 = abs(cwt_data["xi"][(2, 0)])
        eps_cd = max(stack.eps_phc_avg - xi_20, stack.eps_clad + 0.05)
        slab_cd = solve_vertical_slab_mode(stack, eps_phc_override=eps_cd)
        dressed_cwt_cd = compute_cwt_coupling_coefficients(stack, slab=slab_cd)

    modes_out: dict[str, dict] = {}
    freqs_hz = np.zeros(4, dtype=np.float64)
    q_factors = np.zeros(4, dtype=np.float64)

    for idx_mode, lbl in enumerate(labels):
        col = assigned[lbl]
        ev = complex(evals[col])
        vec = evecs[:, col]

        if dressed_vertical_profile and lbl in ("C", "D") and dressed_cwt_cd is not None:
            c_cd = np.asarray(dressed_cwt_cd["C"], dtype=np.complex128)
            h_cd = k_diag + c_cd
            ev = complex(vec.conj().T @ (h_cd @ vec) / (vec.conj().T @ vec))

        delta_m = float(np.real(ev))
        alpha_v_m = float(2.0 * np.imag(ev))
        if abs(alpha_v_m) < 1e-9:
            alpha_v_m = 0.0
        freq_hz = float(f_bragg * (1.0 + delta_m / beta0))
        q_val = float(np.inf if abs(alpha_v_m) < 1e-6 else beta0 / alpha_v_m)

        modes_out[lbl] = {
            "label": lbl,
            "irrep": irreps[lbl],
            "freq_hz": freq_hz,
            "freq_thz": freq_hz * 1e-12,
            "delta_m": delta_m,
            "delta_cm": delta_m * 1e-2,
            "alpha_v_m": alpha_v_m,
            "alpha_v_cm": alpha_v_m * 1e-2,
            "Q": q_val,
            "eigvec": vec,
        }
        freqs_hz[idx_mode] = freq_hz
        q_factors[idx_mode] = q_val

    return {
        "modes": modes_out,
        "A": modes_out["A"],
        "B": modes_out["B"],
        "C": modes_out["C"],
        "D": modes_out["D"],
        "frequencies_hz": freqs_hz,
        "Q_factors": q_factors,
        "f_bragg_hz": f_bragg,
        "beta0": beta0,
    }


def compute_cwt_band_structure(
    stack: PCSELLayerStack,
    k_Red: np.ndarray,
    direction: str = "GX",
    cwt_data: dict | None = None,
) -> dict[str, np.ndarray]:
    """Compute the 4-band 3D-CWT dispersion along :math:`\\Gamma\\text{--X}` or :math:`\\Gamma\\text{--M}`.

    Parameters
    ----------
    stack : PCSELLayerStack
        PCSEL layer stack and unit-cell specification.
    k_Red : numpy.ndarray
        1D array of reduced wavevector magnitudes :math:`k a / (2\\pi)` relative
        to :math:`\\Gamma` (e.g., ``np.linspace(0, 0.06, 25)``).
    direction : {"GX", "GM"}, optional
        Brillouin-zone high-symmetry direction from :math:`\\Gamma` (default ``"GX"``).
    cwt_data : dict, optional
        Precomputed output from :func:`compute_cwt_coupling_coefficients`.

    Returns
    -------
    dict
        Dictionary with keys ``k_red``, ``frequencies_thz`` of shape ``(len(k_Red), 4)``,
        ``Q_factors`` of shape ``(len(k_Red), 4)``, and ``alpha_v_cm`` of shape
        ``(len(k_Red), 4)``.
    """
    if cwt_data is None:
        cwt_data = compute_cwt_coupling_coefficients(stack)

    k_arr = np.asarray(k_Red, dtype=np.float64)
    beta0 = 2.0 * np.pi / stack.a
    n_pts = len(k_arr)
    freqs_thz = np.zeros((n_pts, 4), dtype=np.float64)
    q_arr = np.zeros((n_pts, 4), dtype=np.float64)
    alpha_cm = np.zeros((n_pts, 4), dtype=np.float64)

    for idx, kr in enumerate(k_arr):
        k_val = kr * beta0
        if direction.upper() == "GX":
            kx, ky = k_val, 0.0
        elif direction.upper() == "GM":
            kx, ky = k_val, k_val
        else:
            raise ValueError(f"Unsupported direction {direction!r}; expected 'GX' or 'GM'.")

        res = solve_infinite_cwt(stack, kx=kx, ky=ky, cwt_data=cwt_data)
        f_list = [res[m]["freq_thz"] for m in ("A", "B", "C", "D")]
        q_list = [res[m]["Q"] for m in ("A", "B", "C", "D")]
        a_list = [res[m]["alpha_v_cm"] for m in ("A", "B", "C", "D")]
        order = np.argsort(f_list)
        freqs_thz[idx] = np.array(f_list)[order]
        q_arr[idx] = np.array(q_list)[order]
        alpha_cm[idx] = np.array(a_list)[order]

    return {
        "k_red": k_arr,
        "direction": direction.upper(),
        "frequencies_thz": freqs_thz,
        "Q_factors": q_arr,
        "alpha_v_cm": alpha_cm,
    }


def double_lattice_coupling_scan(
    base_stack: PCSELLayerStack,
    d_over_a_grid: np.ndarray,
    r2_over_r1_grid: np.ndarray,
    m_max: int = 3,
) -> dict[str, np.ndarray | float | PCSELLayerStack | dict]:
    """Scan double-lattice hole separation and radius ratio for 1D feedback cancellation.

    In the radiative-center frame where :math:`\\xi_{1,0}` is real and negative,
    the off-diagonal :math:`180^\\circ` coupling entries between :math:`R_x` and
    :math:`S_x` are

    .. math::

        C_{12} = R(d, r_2/r_1) + i\\left[\\mu(d, r_2/r_1) + I(d, r_2/r_1)\\right], \\qquad
        C_{21} = R(d, r_2/r_1) + i\\left[\\mu(d, r_2/r_1) - I(d, r_2/r_1)\\right].

    When :math:`d \\approx a/4`, the :math:`(2,0)` Bragg harmonic phase shifts by
    :math:`\\pi`, so the real Hermitian coupling :math:`R \\to 0` while the
    imaginary Hermitian coupling :math:`I \\propto r_1^2 - r_2^2` can balance the
    non-Hermitian radiative coupling :math:`\\mu`. The continuous root solved here
    (:math:`R = 0` and :math:`|I| = \\mu`, so :math:`C_{12} = 0` while
    :math:`|C_{21}| = 2\\mu`) corresponds to unidirectional :math:`180^\\circ`
    coupling—a non-Hermitian exceptional point of the 1D counter-propagating
    sub-block where the net geometric backward feedback :math:`\\sqrt{C_{12} C_{21}} = 0`
    vanishes while 2D coupling and diagonal radiation sustain a finite
    infinite-lattice surface-emission rate :math:`\\alpha_{v,A}^\\infty > 0`
    (cf. the near-cancellation regime of Yoshida *et al.*, *Nat. Mater.* **18**,
    121 (2019); Inoue *et al.*, *Nat. Commun.* **13**, 3262 (2022)).

    For each :math:`r_2 / r_1`, the vertical slab mode and Green's functions are
    recomputed self-consistently to account for the change in average PhC
    permittivity :math:`\\varepsilon_{\\text{avg}}(r_1, r_2)`. After evaluating the
    2D grid, the exact continuous cancellation root where :math:`R = 0` and
    :math:`|I| = \\mu` is found via :func:`scipy.optimize.root`.

    Parameters
    ----------
    base_stack : PCSELLayerStack
        Baseline PCSEL stack defining :math:`r_1/a = \\text{base\\_stack.r\\_over\\_a}`
        and epitaxial layer parameters.
    d_over_a_grid : numpy.ndarray
        1D array of normalized hole shifts :math:`d/a`.
    r2_over_r1_grid : numpy.ndarray
        1D array of hole radius ratios :math:`r_2 / r_1`.
    m_max : int, optional
        Maximum diffraction order in the 3D-CWT coupling matrix (default ``3``).

    Returns
    -------
    dict
        Dictionary containing 2D arrays ``R_cm``, ``I_cm``, ``mu_cm``,
        ``abs_C12_cm``, ``abs_C21_cm``, ``min_feedback_cm`` of shape
        ``(len(d_over_a_grid), len(r2_over_r1_grid))`` and the continuous
        cancellation root parameters ``opt_d_over_a``, ``opt_r2_over_r1``,
        ``min_abs_C21_cm``, ``opt_alpha_v_inf_cm``, ``opt_stack``, and ``opt_cwt``.
    """
    d_arr = np.asarray(d_over_a_grid, dtype=np.float64)
    ratio_arr = np.asarray(r2_over_r1_grid, dtype=np.float64)
    nd, nr = len(d_arr), len(ratio_arr)

    r_mat = np.zeros((nd, nr), dtype=np.float64)
    i_mat = np.zeros((nd, nr), dtype=np.float64)
    mu_mat = np.zeros((nd, nr), dtype=np.float64)
    c12_mat = np.zeros((nd, nr), dtype=np.float64)
    c21_mat = np.zeros((nd, nr), dtype=np.float64)

    def _make_dl_stack(d_val: float, ratio: float) -> PCSELLayerStack:
        return PCSELLayerStack(
            a=base_stack.a,
            r_over_a=base_stack.r_over_a,
            n_clad=base_stack.n_clad,
            n_active=base_stack.n_active,
            n_phc_host=base_stack.n_phc_host,
            n_hole=base_stack.n_hole,
            z_act_min=base_stack.z_act_min,
            z_act_max=base_stack.z_act_max,
            z_phc_min=base_stack.z_phc_min,
            z_phc_max=base_stack.z_phc_max,
            z_min=base_stack.z_min,
            z_max=base_stack.z_max,
            r2_over_a=base_stack.r_over_a * float(ratio),
            dx_over_a=float(d_val),
            dy_over_a=float(d_val),
        )

    # Precompute vertical slab & Green's functions per r2/r1 (independent of d/a)
    slab_by_r: list[dict] = []
    greens_by_r: list[dict] = []
    for j_r, ratio in enumerate(ratio_arr):
        st_r = _make_dl_stack(0.25, float(ratio))
        sl_r = solve_vertical_slab_mode(st_r)
        gr_r = solve_vertical_greens_functions(st_r, slab=sl_r, m_max=m_max)
        slab_by_r.append(sl_r)
        greens_by_r.append(gr_r)

    for i_d, d_val in enumerate(d_arr):
        for j_r, ratio in enumerate(ratio_arr):
            st = _make_dl_stack(float(d_val), float(ratio))
            cwt = compute_cwt_coupling_coefficients(
                st,
                m_max=m_max,
                slab=slab_by_r[j_r],
                greens=greens_by_r[j_r],
            )
            r_mat[i_d, j_r] = float(cwt["R_cm"])
            i_mat[i_d, j_r] = float(cwt["I_cm"])
            mu_mat[i_d, j_r] = float(cwt["mu_cm"])
            c12_mat[i_d, j_r] = float(abs(cwt["C12_cm"]))
            c21_mat[i_d, j_r] = float(abs(cwt["C21_cm"]))

    min_fb = np.minimum(c12_mat, c21_mat)
    flat_idx = int(np.argmin(min_fb))
    opt_i, opt_j = np.unravel_index(flat_idx, min_fb.shape)
    d_init = float(d_arr[opt_i])
    r_init = float(ratio_arr[opt_j])
    target_is_c12 = bool(c12_mat[opt_i, opt_j] <= c21_mat[opt_i, opt_j])

    def _residual(p: np.ndarray) -> np.ndarray:
        dv = float(np.clip(p[0], 0.15, 0.40))
        rv = float(np.clip(p[1], 0.40, 0.99))
        st_p = _make_dl_stack(dv, rv)
        sl_p = solve_vertical_slab_mode(st_p)
        gr_p = solve_vertical_greens_functions(st_p, slab=sl_p, m_max=m_max)
        cwt_p = compute_cwt_coupling_coefficients(st_p, m_max=m_max, slab=sl_p, greens=gr_p)
        r_v = float(cwt_p["R_cm"])
        i_v = float(cwt_p["I_cm"])
        mu_v = float(cwt_p["mu_cm"])
        return np.array([r_v, mu_v + i_v if target_is_c12 else mu_v - i_v], dtype=np.float64)

    sol = root(_residual, np.array([d_init, r_init], dtype=np.float64), tol=1e-11)
    if sol.success and 0.15 <= sol.x[0] <= 0.40 and 0.40 <= sol.x[1] <= 0.99:
        opt_d = float(sol.x[0])
        opt_r = float(sol.x[1])
    else:
        opt_d = d_init
        opt_r = r_init

    opt_stack = _make_dl_stack(opt_d, opt_r)
    opt_slab = solve_vertical_slab_mode(opt_stack)
    opt_greens = solve_vertical_greens_functions(opt_stack, slab=opt_slab, m_max=m_max)
    opt_cwt = compute_cwt_coupling_coefficients(opt_stack, m_max=m_max, slab=opt_slab, greens=opt_greens)
    opt_inf = solve_infinite_cwt(opt_stack, cwt_data=opt_cwt)
    opt_min_c = float(min(abs(opt_cwt["C12_cm"]), abs(opt_cwt["C21_cm"])))

    return {
        "d_over_a": d_arr,
        "r2_over_r1": ratio_arr,
        "R_cm": r_mat,
        "I_cm": i_mat,
        "mu_cm": mu_mat,
        "abs_C12_cm": c12_mat,
        "abs_C21_cm": c21_mat,
        "min_feedback_cm": min_fb,
        "grid_opt_d_over_a": d_init,
        "grid_opt_r2_over_r1": r_init,
        "grid_min_abs_C21_cm": float(min_fb[opt_i, opt_j]),
        "opt_d_over_a": opt_d,
        "opt_r2_over_r1": opt_r,
        "min_abs_C21_cm": opt_min_c,
        "opt_max_abs_C_cm": float(max(abs(opt_cwt["C12_cm"]), abs(opt_cwt["C21_cm"]))),
        "opt_alpha_v_inf_cm": float(opt_inf["A"]["alpha_v_cm"]),
        "opt_stack": opt_stack,
        "opt_cwt": opt_cwt,
    }


def _gll_differentiation_1d(n_poly: int, length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D Gauss--Lobatto--Legendre (GLL) nodes, differentiation matrix, and weights on ``[-L/2, L/2]``."""
    if n_poly < 3:
        raise ValueError(f"n_poly must be >= 3, got {n_poly}.")

    c_leg = np.zeros(n_poly + 1, dtype=np.float64)
    c_leg[n_poly] = 1.0
    dc_leg = legendre.legder(c_leg)
    roots = np.real(legendre.legroots(dc_leg))
    x_ref = np.concatenate(([-1.0], np.sort(roots), [1.0]))
    p_n = legendre.legval(x_ref, c_leg)
    w_ref = 2.0 / (n_poly * (n_poly + 1) * p_n**2)

    dx_mat = x_ref[:, None] - x_ref[None, :]
    d_ref = np.outer(p_n, 1.0 / p_n) / (dx_mat + np.eye(n_poly + 1))
    d_ref -= np.diag(np.sum(d_ref, axis=1))
    d_ref[0, 0] = -n_poly * (n_poly + 1) / 4.0
    d_ref[n_poly, n_poly] = n_poly * (n_poly + 1) / 4.0

    x = 0.5 * length * x_ref
    w = 0.5 * length * w_ref
    d_mat = (2.0 / length) * d_ref
    return x, d_mat, w


def solve_finite_cwt(
    c_matrix: np.ndarray,
    length: float,
    n_poly: int = 8,
    tau_sat: float = 1.0,
) -> dict[str, np.ndarray | float | dict]:
    """Solve the 2D finite-size PCSEL CWT eigenvalue problem on :math:`[-L/2, L/2]^2`.

    Parameters
    ----------
    c_matrix : numpy.ndarray
        :math:`4\\times 4` complex coupling matrix :math:`\\mathbf{C}` (in the
        reciprocal units of ``length``).
    length : float
        Lateral side length :math:`L` of the square PCSEL cavity.
    n_poly : int, optional
        GLL polynomial degree per spatial direction (default ``8``).
    tau_sat : float, optional
        Upwind SAT penalty parameter (:math:`\\tau_{\\text{SAT}} \\ge 1/2`, default
        ``1.0``).

    Returns
    -------
    dict
        Dictionary containing ``x``, ``w``, ``evals``, ``alpha_th``, ``alpha_v``,
        ``alpha_par``, ``power_balance_error``, ``fundamental_A``, ``second_A``,
        ``fundamental_B``, and ``delta_alpha_th``.
    """
    c_mat = np.asarray(c_matrix, dtype=np.complex128)
    if c_mat.shape != (4, 4):
        raise ValueError(f"c_matrix must have shape (4, 4), got {c_mat.shape}.")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}.")
    if tau_sat < 0.5:
        raise ValueError(f"tau_sat must be >= 0.5 for SBP stability, got {tau_sat}.")

    x, d_1d, w = _gll_differentiation_1d(n_poly, length)
    n1 = n_poly + 1
    m_grid = n1 * n1

    e0 = np.zeros((n1, n1), dtype=np.float64)
    e0[0, 0] = 1.0 / w[0]
    en = np.zeros((n1, n1), dtype=np.float64)
    en[n_poly, n_poly] = 1.0 / w[n_poly]

    dx_plus = 1j * d_1d + 1j * tau_sat * e0
    dx_minus = -1j * d_1d + 1j * tau_sat * en
    eye_1d = np.eye(n1, dtype=np.complex128)
    eye_2d = np.eye(m_grid, dtype=np.complex128)

    h_mat = np.zeros((4 * m_grid, 4 * m_grid), dtype=np.complex128)
    h_mat[0 * m_grid : 1 * m_grid, 0 * m_grid : 1 * m_grid] = np.kron(dx_plus, eye_1d)
    h_mat[1 * m_grid : 2 * m_grid, 1 * m_grid : 2 * m_grid] = np.kron(dx_minus, eye_1d)
    h_mat[2 * m_grid : 3 * m_grid, 2 * m_grid : 3 * m_grid] = np.kron(eye_1d, dx_plus)
    h_mat[3 * m_grid : 4 * m_grid, 3 * m_grid : 4 * m_grid] = np.kron(eye_1d, dx_minus)

    for a in range(4):
        for b in range(4):
            if c_mat[a, b] != 0.0j:
                h_mat[a * m_grid : (a + 1) * m_grid, b * m_grid : (b + 1) * m_grid] += c_mat[a, b] * eye_2d

    evals, evecs = np.linalg.eig(h_mat)

    w2 = np.kron(w, w)
    w4 = np.tile(w2, 4)
    norms = np.real(np.sum(w4[:, None] * (np.abs(evecs) ** 2), axis=0))

    c_rad_skew = (c_mat - c_mat.conj().T) / (2j)
    c_in = 2.0 * tau_sat - 1.0

    u_basis = np.array(
        [
            [0.5, -0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0],
            [0.0, 0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
        ],
        dtype=np.complex128,
    )
    env_fund = np.outer(np.cos(np.pi * x / length), np.cos(np.pi * x / length)).ravel()
    env_fund /= np.sqrt(np.sum(w2 * env_fund**2))

    n_modes = len(evals)
    alpha_th = 2.0 * np.imag(evals)
    alpha_v = np.zeros(n_modes, dtype=np.float64)
    alpha_par = np.zeros(n_modes, dtype=np.float64)
    family_idx = np.zeros(n_modes, dtype=int)
    fund_overlap = np.zeros(n_modes, dtype=int)

    for m in range(n_modes):
        v4 = evecs[:, m].reshape(4, m_grid) / np.sqrt(norms[m])
        av = 2.0 * float(np.real(np.sum(np.conj(v4) * (c_rad_skew @ v4) * w2[None, :])))
        alpha_v[m] = av

        v_grid = v4.reshape(4, n1, n1)
        ap = float(
            np.sum(w * (np.abs(v_grid[0, n_poly, :]) ** 2 + c_in * np.abs(v_grid[0, 0, :]) ** 2))
            + np.sum(w * (np.abs(v_grid[1, 0, :]) ** 2 + c_in * np.abs(v_grid[1, n_poly, :]) ** 2))
            + np.sum(w * (np.abs(v_grid[2, :, n_poly]) ** 2 + c_in * np.abs(v_grid[2, :, 0]) ** 2))
            + np.sum(w * (np.abs(v_grid[3, :, 0]) ** 2 + c_in * np.abs(v_grid[3, :, n_poly]) ** 2))
        )
        alpha_par[m] = ap

        proj = u_basis.conj() @ v4
        chan_energy = np.real(np.sum(w2[None, :] * (np.abs(proj) ** 2), axis=1))
        fam = int(np.argmax(chan_energy))
        family_idx[m] = fam

    pb_error = float(np.max(np.abs(alpha_th - (alpha_v + alpha_par))))

    def _extract_mode_info(idx_m: int) -> dict:
        v4 = evecs[:, idx_m].reshape(4, n1, n1) / np.sqrt(norms[idx_m])
        intensity = np.sum(np.abs(v4) ** 2, axis=0)
        return {
            "index": int(idx_m),
            "delta": float(np.real(evals[idx_m])),
            "alpha_th": float(alpha_th[idx_m]),
            "alpha_v": float(alpha_v[idx_m]),
            "alpha_par": float(alpha_par[idx_m]),
            "family": ("A", "B", "C", "D")[family_idx[idx_m]],
            "fields": v4,
            "intensity": intensity,
        }

    idx_a = np.where((family_idx == 0) & (alpha_th > 0.0))[0]
    idx_a = idx_a[np.argsort(alpha_th[idx_a])]
    idx_b = np.where((family_idx == 1) & (alpha_th > 0.0))[0]
    idx_b = idx_b[np.argsort(alpha_th[idx_b])]

    idx_all = np.where(((family_idx == 0) | (family_idx == 1)) & (alpha_th > 0.0))[0]
    idx_all = idx_all[np.argsort(alpha_th[idx_all])]

    fund_a = _extract_mode_info(int(idx_a[0]))
    sec_a = _extract_mode_info(int(idx_a[1]))
    fund_b = _extract_mode_info(int(idx_b[0]))

    lowest = float(alpha_th[idx_all[0]])
    second_lowest = float(alpha_th[idx_all[1]])
    delta_alpha_th = second_lowest - lowest

    return {
        "x": x,
        "w": w,
        "evals": evals,
        "alpha_th": alpha_th,
        "alpha_v": alpha_v,
        "alpha_par": alpha_par,
        "family_idx": family_idx,
        "power_balance_error": pb_error,
        "fundamental_A": fund_a,
        "second_A": sec_a,
        "fundamental_B": fund_b,
        "lowest_alpha_th": lowest,
        "second_lowest_alpha_th": second_lowest,
        "delta_alpha_th": delta_alpha_th,
    }
