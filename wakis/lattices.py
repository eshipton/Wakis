# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Triangular and honeycomb photonic-crystal lattices on a Cartesian FIT grid.

This module provides:

1. **Hexagonal Bravais and honeycomb lattice geometry** (:class:`HexagonalLattice`)
   represented on a Cartesian Finite Integration Technique (FIT) mesh via the
   centred-rectangular supercell of dimensions :math:`L_x = a` and
   :math:`L_y = \\sqrt{3}\\,a` containing two primitive lattice points at
   :math:`(0, 0)` and :math:`\\boldsymbol{\\tau} = \\mathbf{a}_2 = (a/2, \\sqrt{3}a/2)`.
2. **Bloch boundary phases and Brillouin-zone paths** (:meth:`HexagonalLattice.bloch_phases`,
   :meth:`HexagonalLattice.bz_path`) following the Wakis :math:`\\exp(+i\\omega t)` time
   and :math:`\\psi_{\\mathbf{k}}(\\mathbf{r} + \\mathbf{L}) = \\exp(+i\\mathbf{k}\\cdot\\mathbf{L})\\psi_{\\mathbf{k}}(\\mathbf{r})`
   spatial Bloch convention.
3. **Supercell band unfolding** (:func:`unfold_supercell_weight`,
   :func:`unfold_signal_pair`, :func:`translate_supercell_field`) projecting
   centred-rectangular supercell eigenstates and time-domain probe pairs onto
   the primitive hexagonal Brillouin zone via the internal translation eigenvalue
   :math:`\\hat{T}_{\\boldsymbol{\\tau}}\\psi_{\\mathbf{k}}(\\mathbf{r}) = \\exp(+i\\mathbf{k}\\cdot\\boldsymbol{\\tau})\\psi_{\\mathbf{k}}(\\mathbf{r})`.
4. **Independent 2D plane-wave-expansion (PWE) solver** (:class:`PWE2D`) written
   from Ho, Chan & Soukoulis (*Phys. Rev. Lett.* **65**, 3152, 1990), Plihal &
   Maradudin (*Phys. Rev. B* **44**, 8565, 1991), and Joannopoulos et al.
   (*Photonic Crystals: Molding the Flow of Light*, 2nd ed., 2008).
5. **Gauge-invariant Berry curvature and valley Chern numbers**
   (:func:`berry_curvature_from_envelopes`, :func:`fit_berry_curvature_patch`,
   :func:`dirac_berry_curvature`) on discretized :math:`\\mathbf{k}`-meshes via the
   Fukui-Hatsugai-Suzuki plaquette phase (*J. Phys. Soc. Jpn.* **74**, 1674, 2005).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0
from scipy.linalg import eigh, inv, sqrtm
from scipy.special import j1

from wakis.gridFIT3D import GridFIT3D
from wakis.resonances import harminv
from wakis.solverFIT3D import SolverFIT3D


@dataclass(frozen=True)
class CircularInclusion:
    """Circular dielectric rod or air hole in a 2D photonic crystal.

    Parameters
    ----------
    x : float
        Centre :math:`x`-coordinate [m].
    y : float
        Centre :math:`y`-coordinate [m].
    radius : float
        Inclusion radius :math:`r` [m]. Must be non-negative.
    eps_in : float
        Relative permittivity :math:`\\varepsilon_{\\mathrm{in}}` inside the circle
        (dimensionless).
    label : str, optional
        Optional sublattice or domain tag (e.g. ``"A"`` or ``"B"``).
    """

    x: float
    y: float
    radius: float
    eps_in: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.radius < 0.0:
            raise ValueError(f"Inclusion radius must be non-negative, got {self.radius}.")
        if self.eps_in <= 0.0:
            raise ValueError(f"Inclusion eps_in must be positive, got {self.eps_in}.")

    @property
    def center(self) -> tuple[float, float, float]:
        """3D centre coordinates ``(x, y, 0.0)`` [m] for :mod:`wakis.schematic`."""
        return (self.x, self.y, 0.0)

    @property
    def axis(self) -> int:
        """Cylinder axis index ``2`` (z-axis) for :mod:`wakis.schematic`."""
        return 2

    @property
    def material(self) -> float:
        """Relative permittivity :math:`\\varepsilon_{\\mathrm{in}}` for :mod:`wakis.schematic`."""
        return self.eps_in

    @property
    def name(self) -> str:
        """Human-readable label for :mod:`wakis.schematic`."""
        return self.label if self.label else f"Rod/Hole (eps={self.eps_in:g})"


@dataclass(frozen=True)
class BZPath:
    """Piecewise-linear path through the 2D Brillouin zone.

    Attributes
    ----------
    k_points : numpy.ndarray
        Wavevectors along the path, shape ``(n_k, 2)`` [rad/m].
    s : numpy.ndarray
        Cumulative arc length in reciprocal space, shape ``(n_k,)`` [rad/m].
    tick_positions : numpy.ndarray
        Arc-length coordinates of the high-symmetry vertices [rad/m].
    tick_labels : tuple of str
        Labels of the high-symmetry vertices (e.g. ``("Gamma", "M", "K", "Gamma")``).
    segment_indices : tuple of int
        Indices in ``k_points`` where each high-symmetry vertex lies.
    """

    k_points: np.ndarray
    s: np.ndarray
    tick_positions: np.ndarray
    tick_labels: tuple[str, ...]
    segment_indices: tuple[int, ...]


class HexagonalLattice:
    """Triangular and honeycomb 2D photonic-crystal lattice geometry.

    The primitive triangular Bravais lattice is spanned by

    .. math::

        \\mathbf{a}_1 = a(1, 0), \\qquad
        \\mathbf{a}_2 = a\\left(\\frac{1}{2}, \\frac{\\sqrt{3}}{2}\\right),

    with primitive cell area :math:`A_{\\mathrm{prim}} = \\frac{\\sqrt{3}}{2}a^2`.
    On an orthogonal Cartesian FIT grid, the lattice is represented by its
    centred-rectangular supercell

    .. math::

        \\mathbf{A}_1 = \\mathbf{a}_1 = (a, 0), \\qquad
        \\mathbf{A}_2 = 2\\mathbf{a}_2 - \\mathbf{a}_1 = (0, \\sqrt{3}\\,a),

    of area :math:`A_{\\mathrm{sup}} = \\sqrt{3}\\,a^2 = 2 A_{\\mathrm{prim}}`, which
    contains two Bravais lattice points per supercell separated by the internal
    primitive translation :math:`\\boldsymbol{\\tau} = \\mathbf{a}_2 = (a/2, \\sqrt{3}a/2)`.

    Parameters
    ----------
    a : float, optional
        Lattice constant :math:`a` [m]. Default is ``1.0``.
    """

    def __init__(self, a: float = 1.0) -> None:
        if a <= 0.0:
            raise ValueError(f"Lattice constant a must be positive, got {a}.")
        self.a = float(a)

    @property
    def a1(self) -> np.ndarray:
        """First primitive Bravais vector :math:`\\mathbf{a}_1 = a(1, 0)` [m]."""
        return np.array([self.a, 0.0], dtype=float)

    @property
    def a2(self) -> np.ndarray:
        """Second primitive Bravais vector :math:`\\mathbf{a}_2 = a(1/2, \\sqrt{3}/2)` [m]."""
        return np.array([0.5 * self.a, 0.5 * np.sqrt(3.0) * self.a], dtype=float)

    @property
    def b1(self) -> np.ndarray:
        """First primitive reciprocal vector :math:`\\mathbf{b}_1 = \\frac{2\\pi}{a}(1, -1/\\sqrt{3})` [rad/m]."""
        return (2.0 * np.pi / self.a) * np.array([1.0, -1.0 / np.sqrt(3.0)], dtype=float)

    @property
    def b2(self) -> np.ndarray:
        """Second primitive reciprocal vector :math:`\\mathbf{b}_2 = \\frac{2\\pi}{a}(0, 2/\\sqrt{3})` [rad/m]."""
        return (2.0 * np.pi / self.a) * np.array([0.0, 2.0 / np.sqrt(3.0)], dtype=float)

    @property
    def supercell_size(self) -> tuple[float, float]:
        """Centred-rectangular supercell dimensions :math:`(L_x, L_y) = (a, \\sqrt{3}a)` [m]."""
        return (self.a, float(np.sqrt(3.0) * self.a))

    @property
    def supercell_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Orthogonal supercell vectors :math:`\\mathbf{A}_1 = (a, 0)` and :math:`\\mathbf{A}_2 = (0, \\sqrt{3}a)` [m]."""
        lx, ly = self.supercell_size
        return (np.array([lx, 0.0], dtype=float), np.array([0.0, ly], dtype=float))

    @property
    def tau(self) -> np.ndarray:
        """Internal primitive translation :math:`\\boldsymbol{\\tau} = (a/2, \\sqrt{3}a/2)` [m]."""
        return self.a2.copy()

    @property
    def g_fold(self) -> np.ndarray:
        """Supercell folding reciprocal vector :math:`\\mathbf{G}_{\\mathrm{fold}} = (0, 2\\pi/(\\sqrt{3}a))` [rad/m].

        Satisfies :math:`\\exp(i\\mathbf{G}_{\\mathrm{fold}}\\cdot\\boldsymbol{\\tau}) = -1`.
        """
        return np.array([0.0, 2.0 * np.pi / (np.sqrt(3.0) * self.a)], dtype=float)

    def high_symmetry_points(self) -> dict[str, np.ndarray]:
        """High-symmetry wavevectors of the hexagonal Brillouin zone [rad/m].

        Returns
        -------
        dict of str to numpy.ndarray
            Dictionary containing ``"Gamma"``, ``"M"``, ``"K"``, ``"K'"``, and
            ``"M_y"`` in Cartesian reciprocal coordinates :math:`(k_x, k_y)` [rad/m].
        """
        a = self.a
        return {
            "Gamma": np.array([0.0, 0.0], dtype=float),
            "M": np.array([np.pi / a, np.pi / (np.sqrt(3.0) * a)], dtype=float),
            "K": np.array([4.0 * np.pi / (3.0 * a), 0.0], dtype=float),
            "K'": np.array([-4.0 * np.pi / (3.0 * a), 0.0], dtype=float),
            "M_y": np.array([0.0, 2.0 * np.pi / (np.sqrt(3.0) * a)], dtype=float),
        }

    def bz_path(
        self,
        labels: Sequence[str] = ("Gamma", "M", "K", "Gamma"),
        n_per_segment: int = 20,
    ) -> BZPath:
        """Construct a piecewise-linear path through the hexagonal Brillouin zone.

        Parameters
        ----------
        labels : sequence of str, optional
            Sequence of high-symmetry point names. Default is
            ``("Gamma", "M", "K", "Gamma")``.
        n_per_segment : int, optional
            Number of intervals per segment between consecutive vertices.
            Must be at least 1. Default is ``20``.

        Returns
        -------
        BZPath
            Sampled wavevectors, cumulative arc-length parameter ``s``, and
            high-symmetry tick positions/labels.
        """
        if len(labels) < 2:
            raise ValueError("At least two high-symmetry labels are required.")
        if n_per_segment < 1:
            raise ValueError(f"n_per_segment must be >= 1, got {n_per_segment}.")
        hs = self.high_symmetry_points()
        for name in labels:
            if name not in hs:
                raise ValueError(f"Unknown high-symmetry point {name!r}; valid keys: {list(hs)}.")

        k_list: list[np.ndarray] = []
        s_list: list[float] = []
        tick_pos: list[float] = [0.0]
        seg_idx: list[int] = [0]
        s_curr = 0.0

        for seg in range(len(labels) - 1):
            k_start = hs[labels[seg]]
            k_end = hs[labels[seg + 1]]
            dk_vec = k_end - k_start
            seg_len = float(np.linalg.norm(dk_vec))
            n_pts = n_per_segment + 1 if seg == len(labels) - 2 else n_per_segment
            for idx in range(n_pts):
                frac = idx / float(n_per_segment)
                k_list.append(k_start + frac * dk_vec)
                s_list.append(s_curr + frac * seg_len)
            s_curr += seg_len
            tick_pos.append(s_curr)
            seg_idx.append((seg + 1) * n_per_segment)

        return BZPath(
            k_points=np.asarray(k_list, dtype=float),
            s=np.asarray(s_list, dtype=float),
            tick_positions=np.asarray(tick_pos, dtype=float),
            tick_labels=tuple(labels),
            segment_indices=tuple(seg_idx),
        )

    def bloch_wavevector(self, k: Sequence[float]) -> tuple[float, float, float]:
        """Return the 3D Bloch wavevector ``(kx, ky, 0.0)`` [rad/m] for :class:`SolverFIT3D`."""
        k_arr = np.asarray(k, dtype=float).ravel()
        if k_arr.size < 2:
            raise ValueError(f"Expected at least 2 wavevector components, got {k_arr.size}.")
        return (float(k_arr[0]), float(k_arr[1]), 0.0)

    def bloch_phases(self, k: Sequence[float]) -> tuple[complex, complex]:
        """Bloch phase factors :math:`(\\phi_x, \\phi_y)` across the rectangular supercell faces.

        For a wavevector :math:`\\mathbf{k} = (k_x, k_y)` in the hexagonal Brillouin
        zone, the centred-rectangular supercell of size :math:`(L_x, L_y) = (a, \\sqrt{3}a)`
        carries the boundary phases

        .. math::

            \\phi_x = \\exp(+i k_x a), \\qquad \\phi_y = \\exp(+i k_y \\sqrt{3}\\,a),

        such that :math:`F(x + a, y) = \\phi_x F(x, y)` and
        :math:`F(x, y + \\sqrt{3}a) = \\phi_y F(x, y)`.

        Parameters
        ----------
        k : sequence of float
            In-plane wavevector :math:`(k_x, k_y)` [rad/m].

        Returns
        -------
        tuple of complex
            ``(phi_x, phi_y)`` on the :math:`x` and :math:`y` faces of the supercell.
        """
        kx, ky, _ = self.bloch_wavevector(k)
        lx, ly = self.supercell_size
        return (complex(np.exp(1j * kx * lx)), complex(np.exp(1j * ky * ly)))

    def triangular_primitive_sites(
        self,
        radius: float,
        eps_in: float,
        offset: Sequence[float] = (0.0, 0.0),
    ) -> list[CircularInclusion]:
        """Single inclusion of a primitive triangular Bravais cell."""
        return [
            CircularInclusion(
                x=float(offset[0]),
                y=float(offset[1]),
                radius=float(radius),
                eps_in=float(eps_in),
                label="A",
            )
        ]

    def triangular_supercell_sites(
        self,
        radius: float,
        eps_in: float,
        origin: Sequence[float] | None = None,
    ) -> list[CircularInclusion]:
        """Two inclusions of a triangular lattice inside the :math:`a \\times \\sqrt{3}a` supercell.

        Parameters
        ----------
        radius : float
            Hole or rod radius :math:`r` [m].
        eps_in : float
            Inclusion relative permittivity :math:`\\varepsilon_{\\mathrm{in}}`.
        origin : sequence of float, optional
            Position of the first Bravais site in the supercell [m]. Defaults to
            :math:`(a/4, \\sqrt{3}a/4)` so both sites lie strictly inside the box
            boundaries.

        Returns
        -------
        list of CircularInclusion
            Two inclusions at :math:`\\mathbf{r}_0` and
            :math:`(\\mathbf{r}_0 + \\boldsymbol{\\tau}) \\bmod (L_x, L_y)`.
        """
        lx, ly = self.supercell_size
        if origin is None:
            x0, y0 = 0.0, 0.0
        else:
            x0, y0 = float(origin[0]) % lx, float(origin[1]) % ly
        tx, ty = self.tau
        return [
            CircularInclusion(x=x0, y=y0, radius=float(radius), eps_in=float(eps_in), label="1"),
            CircularInclusion(
                x=float((x0 + tx) % lx),
                y=float((y0 + ty) % ly),
                radius=float(radius),
                eps_in=float(eps_in),
                label="2",
            ),
        ]

    def honeycomb_primitive_sites(
        self,
        r_a: float,
        r_b: float,
        eps_in: float,
    ) -> list[CircularInclusion]:
        """Two sublattice sites (:math:`A, B`) in the primitive honeycomb unit cell.

        Placed at :math:`\\mathbf{d}_A = \\frac{1}{3}(\\mathbf{a}_1 + \\mathbf{a}_2) = (a/2, a/(2\\sqrt{3}))`
        and :math:`\\mathbf{d}_B = \\frac{2}{3}(\\mathbf{a}_1 + \\mathbf{a}_2) = (a, a/\\sqrt{3})`.
        """
        d_a = (self.a1 + self.a2) / 3.0
        d_b = 2.0 * (self.a1 + self.a2) / 3.0
        return [
            CircularInclusion(x=float(d_a[0]), y=float(d_a[1]), radius=float(r_a), eps_in=float(eps_in), label="A"),
            CircularInclusion(x=float(d_b[0]), y=float(d_b[1]), radius=float(r_b), eps_in=float(eps_in), label="B"),
        ]

    def honeycomb_supercell_sites(
        self,
        r_a: float,
        r_b: float,
        eps_in: float,
    ) -> list[CircularInclusion]:
        """Four sublattice sites (:math:`A_1, A_2, B_1, B_2`) in the :math:`a \\times \\sqrt{3}a` supercell.

        Sublattice :math:`A` sits at :math:`(a/2, L_y/6)` and :math:`(0, 2L_y/3)`;
        sublattice :math:`B` sits at :math:`(a/2, 5L_y/6)` and :math:`(0, L_y/3)`.
        When :math:`r_A = r_B`, the crystal possesses :math:`C_{6v}` symmetry and
        hosts massless Dirac cones at :math:`K` and :math:`K'`. When :math:`r_A \\neq r_B`,
        spatial inversion is broken (:math:`C_{6v} \\to C_{3v}`), opening a
        valley-Hall photonic bandgap.
        """
        lx, ly = self.supercell_size
        return [
            CircularInclusion(x=0.5 * lx, y=ly / 6.0, radius=float(r_a), eps_in=float(eps_in), label="A1"),
            CircularInclusion(x=0.0, y=2.0 * ly / 3.0, radius=float(r_a), eps_in=float(eps_in), label="A2"),
            CircularInclusion(x=0.5 * lx, y=5.0 * ly / 6.0, radius=float(r_b), eps_in=float(eps_in), label="B1"),
            CircularInclusion(x=0.0, y=ly / 3.0, radius=float(r_b), eps_in=float(eps_in), label="B2"),
        ]

    def generate_lattice_inclusions(
        self,
        n1_range: Sequence[int],
        n2_range: Sequence[int],
        basis: Sequence[ tuple[float, float, float, float, str] | CircularInclusion ],
        origin: Sequence[float] = (0.0, 0.0),
        clip_bounds: tuple[float, float, float, float] | None = None,
    ) -> list[CircularInclusion]:
        """Place holes or rods by primitive Bravais vectors :math:`n_1\\mathbf{a}_1 + n_2\\mathbf{a}_2`.

        Parameters
        ----------
        n1_range, n2_range : sequence of int
            Integer lattice indices :math:`n_1` and :math:`n_2`.
        basis : sequence of CircularInclusion or tuple
            Basis inclusions relative to each Bravais lattice site.
        origin : sequence of float, optional
            Global Cartesian shift :math:`(x_0, y_0)` [m]. Default is ``(0.0, 0.0)``.
        clip_bounds : tuple of float, optional
            Optional bounding box ``(xmin, xmax, ymin, ymax)`` [m]. Only inclusions
            whose centres fall inside this box are returned.

        Returns
        -------
        list of CircularInclusion
            Generated inclusions across the lattice.
        """
        ox, oy = float(origin[0]), float(origin[1])
        a1, a2 = self.a1, self.a2
        basis_list: list[CircularInclusion] = []
        for item in basis:
            if isinstance(item, CircularInclusion):
                basis_list.append(item)
            else:
                bx, by, br, beps, blabel = item
                basis_list.append(CircularInclusion(float(bx), float(by), float(br), float(beps), str(blabel)))

        out: list[CircularInclusion] = []
        for n1 in n1_range:
            for n2 in n2_range:
                r_vec = ox + n1 * a1[0] + n2 * a2[0], oy + n1 * a1[1] + n2 * a2[1]
                for b in basis_list:
                    cx = r_vec[0] + b.x
                    cy = r_vec[1] + b.y
                    if clip_bounds is not None:
                        xmin, xmax, ymin, ymax = clip_bounds
                        if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
                            continue
                    out.append(CircularInclusion(x=cx, y=cy, radius=b.radius, eps_in=b.eps_in, label=b.label))
        return out

    def ribbon_supercell_sites(
        self,
        n_cells_y: int,
        r_0: float,
        delta_lower: float,
        delta_upper: float,
        eps_in: float,
    ) -> tuple[list[CircularInclusion], tuple[float, float]]:
        """Construct a valley-Hall domain-wall ribbon supercell of size :math:`a \\times N_y\\sqrt{3}a`.

        In a periodic supercell along :math:`y`, two domain walls are present:
        one at the interior interface :math:`y = L_{\\mathrm{rib}}/2` separating the
        lower domain (:math:`y < L_{\\mathrm{rib}}/2`, asymmetry ``delta_lower``) from
        the upper domain (:math:`y > L_{\\mathrm{rib}}/2`, asymmetry ``delta_upper``),
        and its partner at the periodic boundary :math:`y = 0 \\equiv L_{\\mathrm{rib}}`.

        Parameters
        ----------
        n_cells_y : int
            Total number of :math:`\\sqrt{3}a` supercell rows along :math:`y` (even integer >= 4).
        r_0 : float
            Mean rod radius :math:`r_0` [m].
        delta_lower, delta_upper : float
            Dimensionless radius asymmetry :math:`\\delta` in the lower and upper halves,
            where :math:`r_A = r_0(1 + \\delta)` and :math:`r_B = r_0(1 - \\delta)`.
        eps_in : float
            Rod relative permittivity :math:`\\varepsilon_{\\mathrm{in}}`.

        Returns
        -------
        inclusions : list of CircularInclusion
            All rods in the :math:`a \\times (n_{\\mathrm{cells},y}\\sqrt{3}a)` ribbon supercell.
        ribbon_size : tuple of float
            ``(Lx, Ly_ribbon)`` [m].
        """
        if n_cells_y < 2 or n_cells_y % 2 != 0:
            raise ValueError(f"n_cells_y must be an even integer >= 2, got {n_cells_y}.")
        lx, ly_unit = self.supercell_size
        ly_ribbon = n_cells_y * ly_unit
        inclusions: list[CircularInclusion] = []
        half = n_cells_y // 2
        for iy in range(n_cells_y):
            delta = float(delta_lower if iy < half else delta_upper)
            r_a = float(r_0 * (1.0 + delta))
            r_b = float(r_0 * (1.0 - delta))
            y_base = iy * ly_unit
            for site in self.honeycomb_supercell_sites(r_a, r_b, eps_in):
                inclusions.append(
                    CircularInclusion(
                        x=site.x,
                        y=y_base + site.y,
                        radius=site.radius,
                        eps_in=site.eps_in,
                        label=f"{site.label}_row{iy}",
                    )
                )
        return inclusions, (lx, ly_ribbon)

    def domain_wall_bend_inclusions(
        self,
        nx_cells: int,
        ny_cells: int,
        r_0: float,
        delta: float,
        eps_in: float,
        kind: str = "valley_bend",
        defect_radius: float = 0.0,
    ) -> tuple[list[CircularInclusion], np.ndarray, np.ndarray]:
        """Build a 2D photonic-crystal waveguide with a sharp :math:`120^\\circ` Z-bend or straight guide.

        The domain spans :math:`[0, N_x a] \\times [0, N_y \\sqrt{3}a]`. The waveguide path
        :math:`y_{\\mathrm{wall}}(x)` enters horizontally at :math:`y_1 = 0.35 L_y` from
        :math:`x = 0` to :math:`x_1 = 0.38 L_x`, turns by :math:`120^\\circ` along the
        primitive lattice direction :math:`\\mathbf{a}_2 = (a/2, \\sqrt{3}a/2)` with slope
        :math:`dy/dx = \\sqrt{3}` up to :math:`y_2 = 0.65 L_y`, and exits horizontally
        to :math:`x = L_x`.

        Parameters
        ----------
        nx_cells : int
            Number of :math:`a` cells along :math:`x`.
        ny_cells : int
            Number of :math:`\\sqrt{3}a` cells along :math:`y`.
        r_0 : float
            Mean rod radius [m].
        delta : float
            Dimensionless valley-Hall inversion asymmetry :math:`\\delta`.
        eps_in : float
            Rod relative permittivity :math:`\\varepsilon_{\\mathrm{in}}`.
        kind : {"valley_bend", "valley_straight", "trivial_bend", "trivial_straight"}, optional
            Waveguide topology and geometry. Default is ``"valley_bend"``.
        defect_radius : float, optional
            Reduced rod radius along the waveguide core for ``"trivial_bend"`` and
            ``"trivial_straight"`` guides [m]. Default is ``0.0`` (missing-row defect).

        Returns
        -------
        inclusions : list of CircularInclusion
            All dielectric rods in the simulation domain.
        path_x : numpy.ndarray
            :math:`x`-coordinates of the waveguide centreline [m].
        path_y : numpy.ndarray
            :math:`y`-coordinates of the waveguide centreline [m].
        """
        if kind not in {"valley_bend", "valley_straight", "trivial_bend", "trivial_straight"}:
            raise ValueError(f"Unknown waveguide kind {kind!r}.")
        lx_unit, ly_unit = self.supercell_size
        lx_total = nx_cells * lx_unit
        ly_total = ny_cells * ly_unit

        # C3-compatible -60-deg Z-bend with x1 at a half-integer supercell column (i1 + 0.5)*a
        # so (y_wall(x) + sqrt(3)*x)/Ly = 0.5 (mod 1) on the diagonal arm, placing sublattice B
        # at u = 1/3*Ly below the wall and sublattice A at u = 2/3*Ly above the wall (identical
        # small-rod B-A interface on all three segments).
        m1 = max(int(round(0.60 * ny_cells)), 2)
        m2 = min(int(round(0.40 * ny_cells)), m1 - 1)
        if m2 < 1:
            y1 = 0.65 * ly_total
            y2 = 0.35 * ly_total
        else:
            y1 = float(m1) * ly_unit
            y2 = float(m2) * ly_unit
        dy_bend = y1 - y2
        dx_bend = dy_bend / np.sqrt(3.0)
        i1 = int(round(0.5 * (nx_cells - round(dx_bend / lx_unit)))) - 1
        x1 = (float(max(1, i1)) + 0.5) * lx_unit
        x2 = x1 + dx_bend

        def wall_y(x_val: float) -> float:
            if "straight" in kind:
                return y1
            if x_val <= x1:
                return y1
            if x_val >= x2:
                return y2
            return y1 - np.sqrt(3.0) * (x_val - x1)

        path_x = np.linspace(0.0, lx_total, 200)
        path_y = np.array([wall_y(xv) for xv in path_x], dtype=float)

        inclusions: list[CircularInclusion] = []
        base_sites = self.honeycomb_supercell_sites(r_0, r_0, eps_in)
        core_half_width = 0.48 * self.a

        for ix in range(nx_cells):
            for iy in range(ny_cells):
                x_base = ix * lx_unit
                y_base = iy * ly_unit
                for s in base_sites:
                    cx = x_base + s.x
                    cy = y_base + s.y
                    yw = wall_y(cx)
                    is_a = s.label.startswith("A")

                    if kind.startswith("valley"):
                        # Domain wall flips the sign of delta across y = wall_y(x)
                        loc_delta = + abs(delta) if cy < yw else -abs(delta)
                        if delta < 0:
                            loc_delta = -loc_delta
                        rad = r_0 * (1.0 + loc_delta) if is_a else r_0 * (1.0 - loc_delta)
                        dom_tag = "lower" if cy < yw else "upper"
                        inclusions.append(
                            CircularInclusion(x=cx, y=cy, radius=float(rad), eps_in=float(eps_in), label=dom_tag)
                        )
                    else:
                        # Trivial PhC waveguide: uniform sublattices away from the core,
                        # and missing/reduced rods within core_half_width of the path
                        dist = abs(cy - yw)
                        if "bend" in kind and x1 <= cx <= x2:
                            # Perpendicular distance to the 60-degree segment (slope sqrt(3)) is |cy - yw| / 2
                            dist = 0.5 * abs(cy - yw)
                        if dist <= core_half_width:
                            if defect_radius > 0.0:
                                inclusions.append(
                                    CircularInclusion(
                                        x=cx, y=cy, radius=float(defect_radius), eps_in=float(eps_in), label="core"
                                    )
                                )
                        else:
                            rad = r_0 * (1.0 + abs(delta)) if is_a else r_0 * (1.0 - abs(delta))
                            inclusions.append(
                                CircularInclusion(x=cx, y=cy, radius=float(rad), eps_in=float(eps_in), label="cladding")
                            )

        return inclusions, path_x, path_y


def translate_supercell_field(
    field_xy: np.ndarray,
    k: Sequence[float] | None = None,
    a: float = 1.0,
    is_envelope: bool = False,
) -> np.ndarray:
    """Translate a 2D field on the :math:`a \\times \\sqrt{3}a` supercell by :math:`\\boldsymbol{\\tau} = (a/2, \\sqrt{3}a/2)`.

    For a periodic Bloch envelope :math:`u_{\\mathbf{k}}(x, y)` on :math:`[0, L_x) \\times [0, L_y)`,
    translation by :math:`(L_x/2, L_y/2)` is a pure cyclic half-grid shift. For a
    full Bloch state :math:`\\psi_{\\mathbf{k}}(x, y) = u_{\\mathbf{k}}(x, y) \\exp(i\\mathbf{k}\\cdot\\mathbf{r})`,
    wrapping across :math:`x = L_x` or :math:`y = L_y` multiplies the wrapped half
    of the array by the supercell Bloch phase :math:`\\exp(i k_x L_x)` or :math:`\\exp(i k_y L_y)`.

    Parameters
    ----------
    field_xy : numpy.ndarray
        Complex 2D field array of even shape ``(nx, ny)``.
    k : sequence of float, optional
        Wavevector :math:`(k_x, k_y)` [rad/m]. Required when ``is_envelope=False``.
    a : float, optional
        Lattice constant :math:`a` [m]. Default is ``1.0``.
    is_envelope : bool, optional
        If ``True``, ``field_xy`` is the cell-periodic envelope :math:`u_{\\mathbf{k}}(\\mathbf{r})`.
        If ``False`` (default), ``field_xy`` is the full Bloch field :math:`\\psi_{\\mathbf{k}}(\\mathbf{r})`.

    Returns
    -------
    numpy.ndarray
        Translated field :math:`\\psi_{\\mathbf{k}}(\\mathbf{r} + \\boldsymbol{\\tau})` of shape ``(nx, ny)``.
    """
    arr = np.asarray(field_xy, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"field_xy must be 2D, got shape {arr.shape}.")
    nx, ny = arr.shape
    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError(f"Supercell grid dimensions (nx, ny) must be even, got ({nx}, {ny}).")

    shifted = np.roll(np.roll(arr, -nx // 2, axis=0), -ny // 2, axis=1)
    if is_envelope:
        return shifted
    if k is None:
        raise ValueError("Wavevector k must be provided when is_envelope=False.")
    kx, ky = float(k[0]), float(k[1])
    lx = float(a)
    ly = float(np.sqrt(3.0) * a)
    # Entries with ix >= nx//2 wrapped around x -> x + Lx; iy >= ny//2 wrapped around y -> y + Ly
    shifted[nx // 2 :, :] *= np.exp(1j * kx * lx)
    shifted[:, ny // 2 :] *= np.exp(1j * ky * ly)
    return shifted


def unfold_supercell_weight(
    field_xy: np.ndarray,
    k: Sequence[float],
    a: float = 1.0,
    eps_xy: np.ndarray | None = None,
    is_envelope: bool = False,
    phase_sign: int = 1,
) -> float:
    """Compute the primitive-zone unfolding spectral weight :math:`W_{\\mathbf{k}} \\in [0, 1]` of a supercell mode.

    On the :math:`a \\times \\sqrt{3}a` centred-rectangular supercell, true primitive
    Bloch modes at wavevector :math:`\\mathbf{k}` obey the internal translation
    symmetry

    .. math::

        \\psi_{\\mathbf{k}}(\\mathbf{r} + \\boldsymbol{\\tau})
        = \\exp(+i\\mathbf{k}\\cdot\\boldsymbol{\\tau})\\,\\psi_{\\mathbf{k}}(\\mathbf{r}),

    giving weight :math:`W_{\\mathbf{k}} = 1`, whereas modes folded from
    :math:`\\mathbf{k} + \\mathbf{G}_{\\mathrm{fold}}` satisfy
    :math:`\\psi(\\mathbf{r} + \\boldsymbol{\\tau}) = -\\exp(+i\\mathbf{k}\\cdot\\boldsymbol{\\tau})\\psi(\\mathbf{r})`
    and give :math:`W_{\\mathbf{k}} = 0`.

    Parameters
    ----------
    field_xy : numpy.ndarray
        2D supercell eigenfield of even shape ``(nx, ny)``.
    k : sequence of float
        Target wavevector :math:`(k_x, k_y)` [rad/m].
    a : float, optional
        Lattice constant :math:`a` [m]. Default is ``1.0``.
    eps_xy : numpy.ndarray, optional
        Optional 2D permittivity weight :math:`\\varepsilon(x, y)`.
    is_envelope : bool, optional
        Whether ``field_xy`` is already the periodic envelope :math:`u_{\\mathbf{k}}`.
    phase_sign : int, optional
        Sign of the projection phase :math:`\\exp(i \\cdot \\mathrm{phase\\_sign} \\cdot \\mathbf{k}\\cdot\\boldsymbol{\\tau})`.
        Default is ``+1`` (physical Wakis convention). Setting ``phase_sign=-1``
        projects onto the folded branch instead.

    Returns
    -------
    float
        Unfolding spectral weight in ``[0.0, 1.0]``.
    """
    arr = np.asarray(field_xy, dtype=np.complex128)
    shifted = translate_supercell_field(arr, k=k, a=a, is_envelope=is_envelope)
    kx, ky = float(k[0]), float(k[1])
    tau_x = 0.5 * a
    tau_y = 0.5 * np.sqrt(3.0) * a
    if is_envelope:
        # For u_k(r), the Bloch factor exp(i k . tau) is already factored out when phase_sign == +1
        rel_phase = complex(np.exp(1j * (phase_sign - 1) * (kx * tau_x + ky * tau_y)))
    else:
        rel_phase = complex(np.exp(-1j * phase_sign * (kx * tau_x + ky * tau_y)))

    proj = 0.5 * (arr + rel_phase * shifted)
    w_mat = np.ones_like(arr, dtype=float) if eps_xy is None else np.asarray(eps_xy, dtype=float)
    num = float(np.sum(w_mat * np.abs(proj) ** 2))
    den = 0.5 * float(np.sum(w_mat * (np.abs(arr) ** 2 + np.abs(shifted) ** 2)))
    if den <= 0.0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def unfold_signal_pair(
    s1: np.ndarray,
    s2: np.ndarray,
    k: Sequence[float],
    delta_r: Sequence[float],
    phase_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Unfold a pair of time-domain probe signals recorded at :math:`\\boldsymbol{\\tau}`-paired points.

    Given time series :math:`s_1(t) = \\psi_{\\mathbf{k}}(\\mathbf{r}_1, t)` and
    :math:`s_2(t) = \\psi_{\\mathbf{k}}(\\mathbf{r}_2, t)` at two points in the
    centred-rectangular supercell separated by :math:`\\Delta\\mathbf{r} = \\mathbf{r}_2 - \\mathbf{r}_1`
    (modulo supercell translations, equal to :math:`\\pm\\boldsymbol{\\tau}`), the
    linear combinations

    .. math::

        s_{\\pm}(t) = \\frac{1}{2}\\Bigl(s_1(t) \\pm \\exp(-i\\,\\mathrm{phase\\_sign}\\,\\mathbf{k}\\cdot\\Delta\\mathbf{r})\\,s_2(t)\\Bigr)

    project onto the unfolded primitive branch (:math:`s_+`) and the folded
    supercell branch (:math:`s_-`).

    Parameters
    ----------
    s1, s2 : numpy.ndarray
        Complex time-series arrays of identical shape.
    k : sequence of float
        Wavevector :math:`(k_x, k_y)` [rad/m].
    delta_r : sequence of float
        In-box displacement vector :math:`(x_2 - x_1, y_2 - y_1)` [m].
    phase_sign : int, optional
        Sign of the Bloch projection phase (default ``+1``).

    Returns
    -------
    s_unfolded, s_folded : numpy.ndarray
        Projected time signals for the primitive and folded branches.
    """
    a1 = np.asarray(s1, dtype=np.complex128)
    a2 = np.asarray(s2, dtype=np.complex128)
    kx, ky = float(k[0]), float(k[1])
    dx, dy = float(delta_r[0]), float(delta_r[1])
    phase = np.exp(-1j * phase_sign * (kx * dx + ky * dy))
    s_plus = 0.5 * (a1 + phase * a2)
    s_minus = 0.5 * (a1 - phase * a2)
    return s_plus, s_minus


class PWE2D:
    """Two-dimensional plane-wave-expansion (PWE) photonic band solver.

    Solves the 2D frequency-domain Maxwell eigenvalue equations in a general
    2D periodic unit cell spanned by :math:`(\\mathbf{a}_1, \\mathbf{a}_2)`:

    * **TM polarization** (:math:`E_z` out-of-plane, :math:`H_x, H_y` in-plane):

      .. math::

          \\sum_{\\mathbf{G}'} \\varepsilon^{-1/2}_{\\mathbf{G},\\mathbf{G}''}
          |\\mathbf{k} + \\mathbf{G}''|^2
          \\varepsilon^{-1/2}_{\\mathbf{G}'',\\mathbf{G}'}\\,\\tilde{\\phi}_{\\mathbf{G}'}
          = \\left(\\frac{\\omega}{c}\\right)^2 \\tilde{\\phi}_{\\mathbf{G}},

    * **TE polarization** (:math:`H_z` out-of-plane, :math:`E_x, E_y` in-plane):

      .. math::

          \\sum_{\\mathbf{G}'} (\\mathbf{k} + \\mathbf{G})\\cdot(\\mathbf{k} + \\mathbf{G}')\\,
          \\eta_{\\mathbf{G},\\mathbf{G}'}\\,h_{\\mathbf{G}'}
          = \\left(\\frac{\\omega}{c}\\right)^2 h_{\\mathbf{G}},

    where the dielectric Fourier matrix :math:`\\varepsilon_{\\mathbf{G},\\mathbf{G}'} = \\varepsilon(\\mathbf{G}-\\mathbf{G}')`
    is evaluated analytically via Bessel functions :math:`J_1(|\\mathbf{q}|r)` for
    circular inclusions and inverted using the Ho-Chan-Soukoulis inverse rule
    :math:`\\eta = \\varepsilon^{-1}`.

    Parameters
    ----------
    a1, a2 : sequence of float
        2D Bravais lattice vectors spanning the unit cell [m].
    inclusions : sequence of CircularInclusion
        Circular inclusions inside the unit cell.
    eps_bg : float, optional
        Background relative permittivity :math:`\\varepsilon_{\\mathrm{bg}}`. Default is ``1.0``.
    n_pw : int, optional
        Target half-width of the reciprocal-lattice shell. A circular cutoff
        :math:`|\\mathbf{G}| \\le (n_{\\mathrm{pw}} + 0.15)\\,\\max(|\\mathbf{b}_1|, |\\mathbf{b}_2|)`
        is used when ``circular_cutoff=True`` to preserve exact :math:`C_6` point-group
        symmetry. Default is ``5``.
    polarization : {"TM", "TE"}, optional
        Field polarization. Default is ``"TM"``.
    inverse_rule : bool, optional
        Whether to use the Ho-Chan-Soukoulis matrix inverse :math:`\\eta = \\varepsilon^{-1}`
        instead of the direct Fourier coefficients of :math:`1/\\varepsilon(\\mathbf{r})`.
        Default is ``True``.
    circular_cutoff : bool, optional
        Whether to truncate reciprocal-lattice vectors by a circular norm bound.
        Default is ``True``.
    """

    def __init__(
        self,
        a1: Sequence[float],
        a2: Sequence[float],
        inclusions: Sequence[CircularInclusion],
        eps_bg: float = 1.0,
        n_pw: int = 5,
        polarization: str = "TM",
        inverse_rule: bool = True,
        circular_cutoff: bool = True,
    ) -> None:
        if eps_bg <= 0.0:
            raise ValueError(f"eps_bg must be positive, got {eps_bg}.")
        pol = polarization.upper()
        if pol not in {"TM", "TE"}:
            raise ValueError(f"polarization must be 'TM' or 'TE', got {polarization!r}.")

        self.a1 = np.asarray(a1, dtype=float).ravel()[:2]
        self.a2 = np.asarray(a2, dtype=float).ravel()[:2]
        area = float(abs(self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]))
        if area <= 0.0:
            raise ValueError("Lattice vectors a1 and a2 must be linearly independent.")
        self.cell_area = area
        self.inclusions = list(inclusions)
        self.eps_bg = float(eps_bg)
        self.n_pw = int(n_pw)
        self.polarization = pol
        self.inverse_rule = bool(inverse_rule)
        self.circular_cutoff = bool(circular_cutoff)

        # Reciprocal basis b1, b2 satisfying a_i . b_j = 2*pi*delta_ij
        mat_a = np.column_stack([self.a1, self.a2])
        mat_b = 2.0 * np.pi * inv(mat_a).T
        self.b1 = mat_b[:, 0]
        self.b2 = mat_b[:, 1]

        self.g_vectors = self._build_g_vectors()
        self.n_g = int(self.g_vectors.shape[0])
        self.eps_matrix, self.eta_matrix, self.inv_sqrt_eps = self._build_dielectric_matrices()

    def _build_g_vectors(self) -> np.ndarray:
        a1_norm = float(np.linalg.norm(self.a1))
        a2_norm = float(np.linalg.norm(self.a2))
        g_max = (self.n_pw + 0.15) * (2.0 * np.pi / a1_norm)
        m1_max = max(1, int(np.ceil(a1_norm * g_max / (2.0 * np.pi))) + 1)
        m2_max = max(1, int(np.ceil(a2_norm * g_max / (2.0 * np.pi))) + 1)
        g_list: list[np.ndarray] = []
        for m1 in range(-m1_max, m1_max + 1):
            for m2 in range(-m2_max, m2_max + 1):
                g_vec = m1 * self.b1 + m2 * self.b2
                if self.circular_cutoff:
                    if np.linalg.norm(g_vec) <= g_max * (1.0 + 1e-12):
                        g_list.append(g_vec)
                else:
                    if abs(m1) <= self.n_pw and abs(m2) <= self.n_pw:
                        g_list.append(g_vec)
        g_arr = np.asarray(g_list, dtype=float)
        norms = np.linalg.norm(g_arr, axis=1)
        order = np.argsort(norms, kind="mergesort")
        return g_arr[order]

    def _build_dielectric_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        diff_x = self.g_vectors[:, None, 0] - self.g_vectors[None, :, 0]
        diff_y = self.g_vectors[:, None, 1] - self.g_vectors[None, :, 1]
        q_norm = np.hypot(diff_x, diff_y)

        eps_mat = np.eye(self.n_g, dtype=np.complex128) * self.eps_bg
        eta_direct = np.eye(self.n_g, dtype=np.complex128) * (1.0 / self.eps_bg)

        for inc in self.inclusions:
            if inc.radius <= 0.0:
                continue
            f_fill = (np.pi * inc.radius**2) / self.cell_area
            qr = q_norm * inc.radius
            form = np.ones_like(qr)
            nz = qr > 1e-12
            form[nz] = 2.0 * j1(qr[nz]) / qr[nz]
            phase = np.exp(-1j * (diff_x * inc.x + diff_y * inc.y))
            kernel = f_fill * form * phase
            eps_mat += (inc.eps_in - self.eps_bg) * kernel
            eta_direct += (1.0 / inc.eps_in - 1.0 / self.eps_bg) * kernel

        eps_mat = 0.5 * (eps_mat + eps_mat.conj().T)
        eta_mat = inv(eps_mat) if self.inverse_rule else 0.5 * (eta_direct + eta_direct.conj().T)
        eta_mat = 0.5 * (eta_mat + eta_mat.conj().T)
        inv_sqrt_eps = np.asarray(sqrtm(eta_mat), dtype=np.complex128)
        inv_sqrt_eps = 0.5 * (inv_sqrt_eps + inv_sqrt_eps.conj().T)
        return eps_mat, eta_mat, inv_sqrt_eps

    def solve_k(
        self,
        k: Sequence[float],
        n_bands: int = 8,
        return_vectors: bool = False,
        normalize_a: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Compute photonic eigenfrequencies at a single wavevector :math:`\\mathbf{k}`.

        Parameters
        ----------
        k : sequence of float
            In-plane wavevector :math:`(k_x, k_y)` [rad/m].
        n_bands : int, optional
            Number of lowest bands to return. Default is ``8``.
        return_vectors : bool, optional
            If ``True``, also return the Fourier eigenvectors of shape ``(n_g, n_bands)``.
        normalize_a : float, optional
            If provided, return dimensionless frequencies :math:`\\omega a / (2\\pi c)`.
            If ``None``, uses :math:`a = |\\mathbf{a}_1|`.

        Returns
        -------
        freqs : numpy.ndarray
            Dimensionless eigenfrequencies :math:`\\omega a / (2\\pi c)` of shape ``(n_bands,)``.
        eigvecs : numpy.ndarray, optional
            Fourier coefficients :math:`u_{\\mathbf{G}}` of the periodic envelope
            (returned when ``return_vectors=True``).
        """
        kx, ky = float(k[0]), float(k[1])
        a_norm = float(np.linalg.norm(self.a1) if normalize_a is None else normalize_a)
        kg_x = kx + self.g_vectors[:, 0]
        kg_y = ky + self.g_vectors[:, 1]

        if self.polarization == "TM":
            kg2 = kg_x**2 + kg_y**2
            # H = eps^{-1/2} |k+G|^2 eps^{-1/2}
            op = self.inv_sqrt_eps * kg2[None, :] @ self.inv_sqrt_eps
        else:
            dot_kg = np.outer(kg_x, kg_x) + np.outer(kg_y, kg_y)
            op = dot_kg * self.eta_matrix

        op = 0.5 * (op + op.conj().T)
        n_keep = min(int(n_bands), self.n_g)
        evals, evecs = eigh(op, subset_by_index=(0, n_keep - 1))
        evals = np.clip(np.real(evals), 0.0, None)
        freqs = np.sqrt(evals) * (a_norm / (2.0 * np.pi))

        if not return_vectors:
            return freqs

        if self.polarization == "TM":
            # Convert from eps^{1/2} E_z basis back to E_z Fourier coefficients
            u_vecs = self.inv_sqrt_eps @ evecs
        else:
            u_vecs = evecs
        return freqs, u_vecs

    def solve_path(
        self,
        k_points: np.ndarray,
        n_bands: int = 8,
        unfold_supercell: bool = False,
        lattice_a: float | None = None,
        phase_sign: int = 1,
    ) -> dict[str, np.ndarray]:
        """Compute band structure along a sequence of wavevectors.

        Parameters
        ----------
        k_points : numpy.ndarray
            Wavevectors of shape ``(n_k, 2)`` [rad/m].
        n_bands : int, optional
            Number of bands to compute at each :math:`\\mathbf{k}`. Default is ``8``.
        unfold_supercell : bool, optional
            If ``True``, compute the primitive-zone unfolding weight :math:`W_n(\\mathbf{k})`
            for each supercell band using the internal translation
            :math:`\\boldsymbol{\\tau} = (a/2, \\sqrt{3}a/2)`.
        lattice_a : float, optional
            Lattice constant :math:`a` [m] for frequency normalization and unfolding.
        phase_sign : int, optional
            Sign of the unfolding phase factor (default ``+1``).

        Returns
        -------
        dict of str to numpy.ndarray
            Dictionary containing ``"frequencies"`` of shape ``(n_k, n_bands)``,
            and when ``unfold_supercell=True``, ``"weights"`` of shape ``(n_k, n_bands)``
            and ``"unfolded_frequencies"`` containing the primitive bands selected by
            greatest unfolding weight.
        """
        k_arr = np.asarray(k_points, dtype=float)
        a_val = float(np.linalg.norm(self.a1) if lattice_a is None else lattice_a)
        n_k = k_arr.shape[0]
        n_keep = min(int(n_bands), self.n_g)
        n_solve = min(max(n_keep + 6, 2 * n_keep), self.n_g) if unfold_supercell else n_keep
        freqs_all = np.zeros((n_k, n_keep), dtype=float)
        weights_all = np.ones((n_k, n_keep), dtype=float)
        n_prim = n_keep // 2
        unfolded = np.zeros((n_k, n_prim), dtype=float)

        tau = np.array([0.5 * a_val, 0.5 * np.sqrt(3.0) * a_val], dtype=float)
        # For u_k(r) = sum_G c_G exp(i G . r), u_k(r + tau) has Fourier coefficients c_G exp(i G . tau)
        g_tau_phase = np.exp(1j * (self.g_vectors @ tau))

        for ik in range(n_k):
            k_vec = k_arr[ik]
            if unfold_supercell:
                f_k, v_k = self.solve_k(k_vec, n_bands=n_solve, return_vectors=True, normalize_a=a_val)
                freqs_all[ik] = f_k[:n_keep]
                w_solve = np.zeros(n_solve, dtype=float)
                # When phase_sign == +1, primitive modes satisfy u_k(r + tau) = +u_k(r).
                # When phase_sign != +1, an extra factor exp(i*(phase_sign - 1)*(k . tau)) is applied.
                extra = np.exp(1j * (phase_sign - 1) * float(np.dot(k_vec[:2], tau)))
                for ib in range(n_solve):
                    c_vec = v_k[:, ib]
                    c_shift = c_vec * g_tau_phase * extra
                    if self.polarization == "TM":
                        # Measure norm in the physical dielectric metric <c | eps | c>
                        ec = self.eps_matrix @ c_vec
                        es = self.eps_matrix @ c_shift
                        c_proj = 0.5 * (c_vec + c_shift)
                        ep = self.eps_matrix @ c_proj
                        num = float(np.real(np.vdot(c_proj, ep)))
                        den = 0.5 * float(np.real(np.vdot(c_vec, ec) + np.vdot(c_shift, es)))
                    else:
                        c_proj = 0.5 * (c_vec + c_shift)
                        num = float(np.vdot(c_proj, c_proj).real)
                        den = float(np.vdot(c_vec, c_vec).real)
                    w_solve[ib] = np.clip(num / max(den, 1e-30), 0.0, 1.0)
                weights_all[ik] = w_solve[:n_keep]
                prim_candidates = np.where(w_solve > 0.45)[0]
                if prim_candidates.size >= n_prim:
                    idx_keep = prim_candidates[:n_prim]
                else:
                    idx_keep = np.argsort(-w_solve)[:n_prim]
                unfolded[ik] = np.sort(f_k[idx_keep])
            else:
                freqs_all[ik] = self.solve_k(k_vec, n_bands=n_keep, return_vectors=False, normalize_a=a_val)

        out: dict[str, np.ndarray] = {"frequencies": freqs_all}
        if unfold_supercell:
            out["weights"] = weights_all
            out["unfolded_frequencies"] = unfolded
        return out

    def reconstruct_field(
        self,
        eigvec: np.ndarray,
        nx: int,
        ny: int,
        k: Sequence[float] | None = None,
        bloch: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct a 2D spatial eigenfield on a regular grid over the unit cell.

        Parameters
        ----------
        eigvec : numpy.ndarray
            Fourier coefficient vector of shape ``(n_g,)``.
        nx, ny : int
            Grid resolution along :math:`\\mathbf{a}_1` and :math:`\\mathbf{a}_2`.
        k : sequence of float, optional
            Wavevector :math:`(k_x, k_y)` [rad/m], required when ``bloch=True``.
        bloch : bool, optional
            If ``True``, return the full Bloch wave :math:`\\psi_{\\mathbf{k}}(\\mathbf{r}) = u_{\\mathbf{k}}(\\mathbf{r})e^{i\\mathbf{k}\\cdot\\mathbf{r}}`.
            If ``False`` (default), return the cell-periodic envelope :math:`u_{\\mathbf{k}}(\\mathbf{r})`.

        Returns
        -------
        x_grid, y_grid, field : numpy.ndarray
            2D Cartesian coordinate arrays and complex field of shape ``(nx, ny)``.
        """
        s1 = np.arange(nx, dtype=float) / float(nx)
        s2 = np.arange(ny, dtype=float) / float(ny)
        S1, S2 = np.meshgrid(s1, s2, indexing="ij")
        X = S1 * self.a1[0] + S2 * self.a2[0]
        Y = S1 * self.a1[1] + S2 * self.a2[1]

        phase = np.exp(1j * (self.g_vectors[:, 0, None, None] * X[None, :, :] + self.g_vectors[:, 1, None, None] * Y[None, :, :]))
        field = np.tensordot(np.asarray(eigvec, dtype=np.complex128), phase, axes=(0, 0))
        if bloch:
            if k is None:
                raise ValueError("Wavevector k is required when bloch=True.")
            field = field * np.exp(1j * (float(k[0]) * X + float(k[1]) * Y))
        return X, Y, field

    def berry_curvature(
        self,
        kx_grid: Sequence[float],
        ky_grid: Sequence[float],
        band_index: int = 0,
        phase_sign: int = 1,
    ) -> tuple[np.ndarray, float]:
        """Compute the gauge-invariant FHS Berry curvature on a 2D :math:`(k_x, k_y)` mesh.

        Uses the Fukui-Hatsugai-Suzuki plaquette formula (*J. Phys. Soc. Jpn.* **74**,
        1674, 2005) on the normalized eigenstates :math:`|u_{\\mathbf{k}}\\rangle` of
        the symmetrized Maxwell operator:

        .. math::

            F_{ij} = \\mathrm{Im}\\,\\ln\\Bigl(
                \\langle u_{i,j}|u_{i+1,j}\\rangle
                \\langle u_{i+1,j}|u_{i+1,j+1}\\rangle
                \\langle u_{i+1,j+1}|u_{i,j+1}\\rangle
                \\langle u_{i,j+1}|u_{i,j}\\rangle
            \\Bigr), \\qquad
            \\Omega_{ij} = \\frac{F_{ij}}{\\Delta k_x\\,\\Delta k_y}.

        Parameters
        ----------
        kx_grid, ky_grid : sequence of float
            1D uniform wavevector grids along :math:`k_x` and :math:`k_y` [rad/m].
        band_index : int, optional
            Zero-based index of the band. Default is ``0``.
        phase_sign : int, optional
            Sign applied to the Berry phase (default ``+1``; set ``-1`` for mutation testing).

        Returns
        -------
        omega : numpy.ndarray
            Plaquette Berry curvature :math:`\\Omega(k_x, k_y)` of shape ``(len(kx_grid)-1, len(ky_grid)-1)`` [:math:`\\mathrm{m}^2`].
        c_valley : float
            Integrated patch Chern number :math:`C_{\\mathrm{patch}} = \\frac{1}{2\\pi}\\sum_{i,j} F_{ij}` (dimensionless).
        """
        kx_arr = np.asarray(kx_grid, dtype=float)
        ky_arr = np.asarray(ky_grid, dtype=float)
        if kx_arr.size < 2 or ky_arr.size < 2:
            raise ValueError("kx_grid and ky_grid must each have at least 2 points.")
        dkx = float(kx_arr[1] - kx_arr[0])
        dky = float(ky_arr[1] - ky_arr[0])

        nkx, nky = kx_arr.size, ky_arr.size
        u_grid = np.zeros((nkx, nky, self.n_g), dtype=np.complex128)
        n_req = int(band_index) + 1
        for i, kx in enumerate(kx_arr):
            for j, ky in enumerate(ky_arr):
                kg_x = kx + self.g_vectors[:, 0]
                kg_y = ky + self.g_vectors[:, 1]
                if self.polarization == "TM":
                    kg2 = kg_x**2 + kg_y**2
                    op = self.inv_sqrt_eps * kg2[None, :] @ self.inv_sqrt_eps
                else:
                    dot_kg = np.outer(kg_x, kg_x) + np.outer(kg_y, kg_y)
                    op = dot_kg * self.eta_matrix
                op = 0.5 * (op + op.conj().T)
                _, evecs = eigh(op, subset_by_index=(0, n_req - 1))
                u_grid[i, j, :] = evecs[:, band_index]

        return berry_curvature_from_envelopes(u_grid, dkx, dky, phase_sign=phase_sign)


def berry_curvature_from_envelopes(
    u_grid: np.ndarray,
    dk_x: float,
    dk_y: float,
    phase_sign: int = 1,
) -> tuple[np.ndarray, float]:
    """Compute gauge-invariant Fukui-Hatsugai-Suzuki Berry curvature from periodic envelopes.

    Parameters
    ----------
    u_grid : numpy.ndarray
        Complex array of periodic Bloch states of shape ``(nkx, nky, n_dof)``.
    dk_x, dk_y : float
        Wavevector grid spacings :math:`\\Delta k_x, \\Delta k_y` [rad/m].
    phase_sign : int, optional
        Sign multiplier for the link phase (default ``+1``).

    Returns
    -------
    omega : numpy.ndarray
        Berry curvature on plaquette centres, shape ``(nkx - 1, nky - 1)`` [:math:`\\mathrm{m}^2`].
    c_patch : float
        Integrated Chern number over the patch :math:`\\frac{1}{2\\pi}\\int \\Omega\\,d^2k`.
    """
    arr = np.asarray(u_grid, dtype=np.complex128)
    if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"u_grid must have shape (nkx>=2, nky>=2, n_dof), got {arr.shape}.")
    nkx, nky, _ = arr.shape
    omega = np.zeros((nkx - 1, nky - 1), dtype=float)
    flux_sum = 0.0
    area_k = float(dk_x * dk_y)

    for i in range(nkx - 1):
        for j in range(nky - 1):
            u00 = arr[i, j]
            u10 = arr[i + 1, j]
            u11 = arr[i + 1, j + 1]
            u01 = arr[i, j + 1]
            if phase_sign >= 0:
                u_links = (
                    np.vdot(u00, u10)
                    * np.vdot(u10, u11)
                    * np.vdot(u11, u01)
                    * np.vdot(u01, u00)
                )
            else:
                u_links = np.conj(
                    np.vdot(u00, u10)
                    * np.vdot(u10, u11)
                    * np.vdot(u11, u01)
                    * np.vdot(u01, u00)
                )
            f_ij = float(np.angle(u_links))
            omega[i, j] = f_ij / area_k
            flux_sum += f_ij

    return omega, float(flux_sum / (2.0 * np.pi))


def dirac_berry_curvature(
    qx: np.ndarray,
    qy: np.ndarray,
    v_d: float,
    delta_omega: float,
    valley: int = 1,
) -> np.ndarray:
    """Analytical Berry curvature of a gapped 2D massive Dirac cone around :math:`K` or :math:`K'`.

    Near valley :math:`\\tau_v = \\pm 1` (:math:`+1` for :math:`K`, :math:`-1` for :math:`K'`),
    the two-band effective Hamiltonian is

    .. math::

        H_{\\mathrm{eff}}(\\mathbf{q}) = \\omega_D + v_D(\\tau_v q_x \\sigma_x + q_y \\sigma_y) + m\\sigma_z,

    with mass :math:`m = \\Delta\\omega_K / 2`. The lower-band Berry curvature is

    .. math::

        \\Omega_1(\\mathbf{q}) = -\\frac{\\tau_v v_D^2 m}{2\\left(v_D^2 (q_x^2 + q_y^2) + m^2\\right)^{3/2}},

    whose integral over the infinite :math:`\\mathbf{q}`-plane yields the quantized
    half-integer valley Chern number :math:`C_v = -\\frac{1}{2}\\,\\tau_v\\,\\mathrm{sgn}(m)`.

    Parameters
    ----------
    qx, qy : numpy.ndarray
        Wavevector displacements :math:`\\mathbf{q} = \\mathbf{k} - \\mathbf{K}_{\\tau_v}` [rad/m].
    v_d : float
        Dirac group velocity :math:`v_D` [m/s] (or dimensionless if :math:`\\Delta\\omega` is dimensionless).
    delta_omega : float
        Signed Dirac bandgap :math:`\\Delta\\omega_K = 2m` in the same frequency units as :math:`v_D |\\mathbf{q}|`.
    valley : {1, -1}, optional
        Valley index :math:`+1` for :math:`K` and :math:`-1` for :math:`K'`. Default is ``1``.

    Returns
    -------
    numpy.ndarray
        Analytical lower-band Berry curvature :math:`\\Omega_1(\\mathbf{q})` [:math:`\\mathrm{m}^2`].
    """
    if valley not in {1, -1}:
        raise ValueError(f"valley must be +1 or -1, got {valley}.")
    qx_arr = np.asarray(qx, dtype=float)
    qy_arr = np.asarray(qy, dtype=float)
    m = 0.5 * float(delta_omega)
    q2 = qx_arr**2 + qy_arr**2
    denom = (float(v_d) ** 2 * q2 + m**2) ** 1.5
    return -float(valley) * (float(v_d) ** 2 * m) / np.maximum(2.0 * denom, 1e-30)


def fill_circular_inclusions(
    solver: SolverFIT3D,
    inclusions: Sequence[CircularInclusion],
    eps_bg: float = 1.0,
    n_sub: int = 8,
    periodic_wrap: bool = True,
) -> np.ndarray:
    """Populate ``solver.ieps`` with subpixel-averaged circular inclusions.

    Uses :math:`n_{\\mathrm{sub}} \\times n_{\\mathrm{sub}}` quadrature over each
    Cartesian cell to compute the area fill fraction :math:`f_{i,j} \\in [0, 1]` and
    the effective permittivity :math:`\\varepsilon_{\\mathrm{eff}} = f\\,\\varepsilon_{\\mathrm{in}} + (1 - f)\\varepsilon_{\\mathrm{bg}}`,
    followed by ``solver.update_tensors("ieps")``.

    Parameters
    ----------
    solver : SolverFIT3D
        Initialized Wakis 3D FIT solver.
    inclusions : sequence of CircularInclusion
        Circular rods or holes to place on the mesh.
    eps_bg : float, optional
        Background relative permittivity. Default is ``1.0``.
    n_sub : int, optional
        Subpixel quadrature subdivisions per axis. Default is ``8``.
    periodic_wrap : bool, optional
        If ``True`` (default), wrap inclusion distances using minimum-image
        convention across the domain :math:`(L_x, L_y)`.

    Returns
    -------
    numpy.ndarray
        2D relative permittivity map ``eps_eff`` of shape ``(Nx, Ny)``.
    """
    if eps_bg <= 0.0:
        raise ValueError(f"eps_bg must be positive, got {eps_bg}.")
    if n_sub < 1:
        raise ValueError(f"n_sub must be >= 1, got {n_sub}.")

    nx, ny = solver.Nx, solver.Ny
    x_nodes = np.asarray(solver.grid.x[:nx], dtype=float)
    y_nodes = np.asarray(solver.grid.y[:ny], dtype=float)
    lx = float(solver.grid.xmax - solver.grid.xmin)
    ly = float(solver.grid.ymax - solver.grid.ymin)
    dx = lx / float(nx)
    dy = ly / float(ny)

    sub = (np.arange(n_sub, dtype=float) + 0.5) / float(n_sub) - 0.5
    ox, oy = np.meshgrid(sub * dx, sub * dy, indexing="ij")
    XC, YC = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    SX = XC[:, :, None, None] + ox[None, None, :, :]
    SY = YC[:, :, None, None] + oy[None, None, :, :]

    eps_sub = np.full(SX.shape, float(eps_bg), dtype=float)
    for inc in inclusions:
        if inc.radius <= 0.0:
            continue
        if periodic_wrap:
            ddx = (SX - inc.x + 0.5 * lx) % lx - 0.5 * lx
            ddy = (SY - inc.y + 0.5 * ly) % ly - 0.5 * ly
        else:
            ddx = SX - inc.x
            ddy = SY - inc.y
        mask = (ddx**2 + ddy**2) <= inc.radius**2
        eps_sub[mask] = float(inc.eps_in)

    eps_eff = eps_sub.mean(axis=(2, 3))
    inv_eps = 1.0 / (eps_eff * epsilon_0)
    for iz in range(solver.Nz):
        solver.ieps[:, :, iz, "x"] = inv_eps
        solver.ieps[:, :, iz, "y"] = inv_eps
        solver.ieps[:, :, iz, "z"] = inv_eps
    solver.update_tensors("ieps")
    return eps_eff


def fit_supercell_bands_at_k(
    lattice: HexagonalLattice,
    inclusions: Sequence[CircularInclusion],
    k: Sequence[float],
    polarization: str = "TM",
    nx: int = 24,
    ny: int = 42,
    eps_bg: float = 1.0,
    f_min: float = 0.12,
    f_max: float = 0.55,
    n_steps: int = 2400,
    n_sub: int = 8,
    phase_sign: int = 1,
) -> dict[str, np.ndarray]:
    """Extract unfolded and folded photonic bands at wavevector :math:`\\mathbf{k}` via 2D FIT + ``harminv``.

    Runs :class:`SolverFIT3D` on the :math:`a \\times \\sqrt{3}a` centred-rectangular
    supercell with Bloch boundary conditions at :math:`\\mathbf{k} = (k_x, k_y, 0)`,
    excites the cell at two :math:`\\boldsymbol{\\tau}`-related probe pairs, unfolds
    the recorded time series with :func:`unfold_signal_pair`, and extracts the
    resonant eigenfrequencies with :func:`wakis.resonances.harminv`.

    Parameters
    ----------
    lattice : HexagonalLattice
        Hexagonal lattice instance.
    inclusions : sequence of CircularInclusion
        Supercell inclusions inside :math:`[0, a) \\times [0, \\sqrt{3}a)`.
    k : sequence of float
        Wavevector :math:`(k_x, k_y)` [rad/m].
    polarization : {"TM", "TE"}, optional
        ``"TM"`` (:math:`E_z` out-of-plane) or ``"TE"`` (:math:`H_z` out-of-plane).
        Default is ``"TM"``.
    nx, ny : int, optional
        Supercell grid resolution (must be even). Default is ``(24, 42)``.
    eps_bg : float, optional
        Background relative permittivity. Default is ``1.0``.
    f_min, f_max : float, optional
        Dimensionless frequency search window :math:`\\omega a / (2\\pi c)`.
    n_steps : int, optional
        Total number of FIT time steps. Default is ``2400``.
    n_sub : int, optional
        Subpixel permittivity averaging order. Default is ``8``.
    phase_sign : int, optional
        Sign of the :math:`\\boldsymbol{\\tau}` unfolding phase (default ``+1``).

    Returns
    -------
    dict of str to numpy.ndarray
        Dictionary with keys ``"unfolded_frequencies"``, ``"folded_frequencies"``,
        ``"s_unfolded"``, and ``"s_folded"``.
    """
    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError(f"Supercell (nx, ny) must be even, got ({nx}, {ny}).")
    pol = polarization.upper()
    if pol not in {"TM", "TE"}:
        raise ValueError(f"polarization must be 'TM' or 'TE', got {polarization!r}.")

    a = lattice.a
    lx, ly = lattice.supercell_size
    lz = lx / float(nx)
    kx, ky, kz = lattice.bloch_wavevector(k)

    grid = GridFIT3D(0.0, lx, 0.0, ly, 0.0, lz, nx, ny, 1, use_mpi=False, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["bloch", "bloch", "periodic"],
        bc_high=["bloch", "bloch", "periodic"],
        k_bloch=(kx, ky, kz),
        dtype=np.complex128,
        verbose=0,
    )
    fill_circular_inclusions(solver, inclusions, eps_bg=eps_bg, n_sub=n_sub, periodic_wrap=True)

    # Choose two low-symmetry points in cell 1 (near sublattices A1 and B1) and their exact tau-images in cell 2
    p1_ix, p1_iy = max(1, int(round(0.44 * nx))), max(1, int(round(0.15 * ny)))
    p2_ix, p2_iy = (p1_ix + nx // 2) % nx, (p1_iy + ny // 2) % ny

    q1_ix, q1_iy = max(1, int(round(0.46 * nx))), max(1, int(round(0.82 * ny)))
    q2_ix, q2_iy = (q1_ix + nx // 2) % nx, (q1_iy + ny // 2) % ny

    dx_p = float(grid.x[p2_ix] - grid.x[p1_ix])
    dy_p = float(grid.y[p2_iy] - grid.y[p1_iy])
    dx_q = float(grid.x[q2_ix] - grid.x[q1_ix])
    dy_q = float(grid.y[q2_iy] - grid.y[q1_iy])

    f0_hz = 0.5 * (f_min + f_max) * c_light / a
    df_hz = max(0.20, 0.65 * (f_max - f_min)) * c_light / a
    t_arr = np.arange(n_steps, dtype=float) * solver.dt
    tau_p = 1.0 / (np.pi * df_hz)
    t_peak = 2.6 * tau_p
    n_drive = min(n_steps // 3, int(round(5.5 * tau_p / solver.dt)))
    pulse = np.exp(-((t_arr - t_peak) / tau_p) ** 2) * np.cos(2.0 * np.pi * f0_hz * (t_arr - t_peak))

    # Drive mainly at (p1_ix, p1_iy) and (q1_ix, q1_iy) so both unfolded and folded subspaces are excited
    s_p1 = np.zeros(n_steps, dtype=np.complex128)
    s_p2 = np.zeros(n_steps, dtype=np.complex128)
    s_q1 = np.zeros(n_steps, dtype=np.complex128)
    s_q2 = np.zeros(n_steps, dtype=np.complex128)

    for n in range(n_steps):
        if n < n_drive:
            val = pulse[n]
            if pol == "TM":
                solver.J[p1_ix, p1_iy, 0, "z"] = val
                solver.J[q1_ix, q1_iy, 0, "z"] = 0.7 * val
            else:
                solver.J[p1_ix, p1_iy, 0, "x"] = val
                solver.J[p1_ix, p1_iy, 0, "y"] = -0.8 * val
                solver.J[q1_ix, q1_iy, 0, "x"] = 0.6 * val
                solver.J[q1_ix, q1_iy, 0, "y"] = 0.5 * val
        elif n == n_drive:
            solver.J[p1_ix, p1_iy, 0, "x"] = 0.0
            solver.J[p1_ix, p1_iy, 0, "y"] = 0.0
            solver.J[p1_ix, p1_iy, 0, "z"] = 0.0
            solver.J[q1_ix, q1_iy, 0, "x"] = 0.0
            solver.J[q1_ix, q1_iy, 0, "y"] = 0.0
            solver.J[q1_ix, q1_iy, 0, "z"] = 0.0

        solver.one_step()
        if pol == "TM":
            s_p1[n] = solver.E[p1_ix, p1_iy, 0, "z"]
            s_p2[n] = solver.E[p2_ix, p2_iy, 0, "z"]
            s_q1[n] = solver.E[q1_ix, q1_iy, 0, "z"]
            s_q2[n] = solver.E[q2_ix, q2_iy, 0, "z"]
        else:
            s_p1[n] = solver.H[p1_ix, p1_iy, 0, "z"]
            s_p2[n] = solver.H[p2_ix, p2_iy, 0, "z"]
            s_q1[n] = solver.H[q1_ix, q1_iy, 0, "z"]
            s_q2[n] = solver.H[q2_ix, q2_iy, 0, "z"]

    sp_unf, sp_fold = unfold_signal_pair(s_p1[n_drive:], s_p2[n_drive:], (kx, ky), (dx_p, dy_p), phase_sign=phase_sign)
    sq_unf, sq_fold = unfold_signal_pair(s_q1[n_drive:], s_q2[n_drive:], (kx, ky), (dx_q, dy_q), phase_sign=phase_sign)
    sig_unf = sp_unf + 0.7 * sq_unf
    sig_fold = sp_fold + 0.7 * sq_fold

    fmin_hz = f_min * c_light / a
    fmax_hz = f_max * c_light / a

    def _collect_poles(signals: list[np.ndarray]) -> np.ndarray:
        vals: list[float] = []
        for sig in signals:
            if np.max(np.abs(sig)) <= 1e-15:
                continue
            res = harminv(
                np.real(sig), solver.dt, fmin=fmin_hz, fmax=fmax_hz, max_rel_error=5e-2, min_rel_amplitude=1e-4
            )
            good = res.frequency[np.abs(res.Q) > 20.0] * a / c_light
            for f_val in good:
                if not any(abs(float(f_val) - existing) < 2.5e-3 for existing in vals):
                    vals.append(float(f_val))
        return np.asarray(sorted(vals), dtype=float)

    f_unf = _collect_poles([sp_unf, sq_unf, sig_unf])
    f_fold = _collect_poles([sp_fold, sq_fold, sig_fold])
    return {
        "unfolded_frequencies": f_unf,
        "folded_frequencies": f_fold,
        "s_unfolded": sig_unf,
        "s_folded": sig_fold,
    }


def fit_berry_curvature_patch(
    lattice: HexagonalLattice,
    r_0: float = 0.18,
    delta: float = 0.12,
    eps_rod: float = 12.0,
    eps_bg: float = 1.0,
    k_center: Sequence[float] | None = None,
    dk_frac: float = 0.08,
    n_k: int = 3,
    nx: int = 18,
    ny: int = 30,
    n_steps: int = 6000,
    phase_sign: int = 1,
    k_offset: Sequence[float] | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute the 2D FIT Berry curvature patch and valley Chern sign around :math:`K` or :math:`K'`.

    For each wavevector :math:`\\mathbf{k}_{i,j}` on an :math:`n_k \\times n_k` grid
    centred on ``k_center`` (default :math:`K = (4\\pi/(3a), 0)`), runs a 2D
    :class:`SolverFIT3D` time-domain simulation on the :math:`a \\times \\sqrt{3}a`
    supercell, extracts the lower-band Bloch mode during the post-pulse ring-down
    via a Hanning-windowed Fourier projection, weights by :math:`\\sqrt{\\varepsilon(\\mathbf{r})}`,
    and evaluates the gauge-invariant FHS plaquette Berry curvature.

    Parameters
    ----------
    lattice : HexagonalLattice
        Hexagonal lattice instance.
    r_0 : float, optional
        Mean rod radius in units of :math:`a` (if ``lattice.a == 1``) or [m]. Default is ``0.18 * a``.
    delta : float, optional
        Dimensionless inversion-breaking asymmetry :math:`\\delta`, where
        :math:`r_A = r_0(1 + \\delta)` and :math:`r_B = r_0(1 - \\delta)`. Default is ``0.12``.
    eps_rod : float, optional
        Relative permittivity of the dielectric rods. Default is ``12.0``.
    eps_bg : float, optional
        Background relative permittivity. Default is ``1.0``.
    k_center : sequence of float, optional
        Centre of the :math:`\\mathbf{k}`-patch [rad/m]. Defaults to :math:`K = (4\\pi/(3a), 0)`.
    dk_frac : float, optional
        Half-width of the patch in units of :math:`2\\pi/a`. Default is ``0.08``.
    n_k : int, optional
        Number of :math:`\\mathbf{k}` points per axis (:math:`n_k \\ge 2`). Default is ``3``.
    nx, ny : int, optional
        Supercell grid dimensions (``nx`` even, ``ny`` divisible by 6 recommended so
        all four honeycomb sublattice sites lie on exact grid nodes). Default is ``(18, 30)``.
    n_steps : int, optional
        Total time steps per :math:`\\mathbf{k}` point. Default is ``6000`` (~120 :math:`a/c`
        ring-down to resolve the lower valley band cleanly from the upper band).
    phase_sign : int, optional
        Sign multiplier for the FHS link phase (default ``+1``).
    k_offset : sequence of float, optional
        Optional wavevector shift :math:`(\\delta k_x, \\delta k_y)` [rad/m] applied to the
        patch centre. When ``None`` and ``delta == 0``, the patch is automatically offset
        away from the degenerate Dirac point at :math:`K` and off the :math:`k_y = 0` mirror
        line so no grid node sits on the degeneracy.

    Returns
    -------
    dict
        Dictionary containing ``"kx_grid"``, ``"ky_grid"``, ``"omega"``, and
        ``"c_patch"``.
    """
    a = lattice.a
    lx, ly = lattice.supercell_size
    lz = lx / float(nx)
    if k_center is None:
        kc = lattice.high_symmetry_points()["K"].copy()
    else:
        kc = np.asarray(k_center, dtype=float).ravel()[:2].copy()

    dk = float(dk_frac) * (2.0 * np.pi / a)
    if k_offset is not None:
        off = np.asarray(k_offset, dtype=float).ravel()[:2]
        kc = kc + off
        ks_offset = np.linspace(-dk, dk, int(n_k))
    elif abs(delta) < 1e-12:
        # Offset the patch away from the degenerate Dirac point at K and break ky <-> -ky symmetry
        kc = kc + np.array([-1.6 * dk, +0.35 * dk], dtype=float)
        ks_offset = np.linspace(-0.5 * dk, 0.5 * dk, int(n_k))
    else:
        ks_offset = np.linspace(-dk, dk, int(n_k))
    kx_grid = kc[0] + ks_offset
    ky_grid = kc[1] + ks_offset

    r_base = float(r_0)
    r_a = r_base * (1.0 + float(delta))
    r_b = r_base * (1.0 - float(delta))
    inclusions = lattice.honeycomb_supercell_sites(r_a, r_b, eps_rod)

    # Sublattice excitation weights: lower band is localized on the larger-radius sublattice
    if delta > 0.0:
        w_a, w_b = 1.0, 0.0
    elif delta < 0.0:
        w_a, w_b = 0.0, 1.0
    else:
        w_a, w_b = 1.0, 1.0

    f0_norm = 0.252
    f0 = f0_norm * c_light / a
    df = 0.015 * c_light / a

    ix_a1 = int(round((0.5 * lx) / (lx / nx))) % nx
    iy_a1 = int(round((ly / 6.0) / (ly / ny))) % ny
    ix_a2 = (ix_a1 + nx // 2) % nx
    iy_a2 = (iy_a1 + ny // 2) % ny

    ix_b1 = int(round((0.5 * lx) / (lx / nx))) % nx
    iy_b1 = int(round((5.0 * ly / 6.0) / (ly / ny))) % ny
    ix_b2 = (ix_b1 + nx // 2) % nx
    iy_b2 = (iy_b1 + ny // 2) % ny

    u_grid = np.zeros((n_k, n_k, nx * ny), dtype=np.complex128)

    for i, kx in enumerate(kx_grid):
        for j, ky in enumerate(ky_grid):
            grid = GridFIT3D(0.0, lx, 0.0, ly, 0.0, lz, nx, ny, 1, use_mpi=False, verbose=0)
            solver = SolverFIT3D(
                grid,
                bc_low=["bloch", "bloch", "periodic"],
                bc_high=["bloch", "bloch", "periodic"],
                k_bloch=(float(kx), float(ky), 0.0),
                dtype=np.complex128,
                verbose=0,
            )
            eps_eff = fill_circular_inclusions(solver, inclusions, eps_bg=eps_bg, n_sub=6, periodic_wrap=True)

            t_arr = np.arange(n_steps, dtype=float) * solver.dt
            tau_p = 1.0 / (np.pi * df)
            t_peak = 2.5 * tau_p
            nt_drive = int(min(n_steps // 2, round(5.0 * tau_p / solver.dt)))
            pulse = np.exp(-((t_arr - t_peak) / tau_p) ** 2) * np.cos(2.0 * np.pi * f0 * (t_arr - t_peak))

            ph_a1 = w_a * np.exp(1j * (kx * grid.x[ix_a1] + ky * grid.y[iy_a1]))
            ph_a2 = w_a * np.exp(1j * (kx * grid.x[ix_a2] + ky * grid.y[iy_a2]))
            ph_b1 = w_b * np.exp(1j * (kx * grid.x[ix_b1] + ky * grid.y[iy_b1]))
            ph_b2 = w_b * np.exp(1j * (kx * grid.x[ix_b2] + ky * grid.y[iy_b2]))

            nt_ring = n_steps - nt_drive
            win = np.hanning(nt_ring)
            acc = np.zeros((nx, ny), dtype=np.complex128)

            for n in range(n_steps):
                if n < nt_drive:
                    val = pulse[n]
                    solver.J[ix_a1, iy_a1, 0, "z"] = val * ph_a1
                    solver.J[ix_a2, iy_a2, 0, "z"] = val * ph_a2
                    solver.J[ix_b1, iy_b1, 0, "z"] = val * ph_b1
                    solver.J[ix_b2, iy_b2, 0, "z"] = val * ph_b2
                elif n == nt_drive:
                    solver.J[ix_a1, iy_a1, 0, "z"] = 0.0
                    solver.J[ix_a2, iy_a2, 0, "z"] = 0.0
                    solver.J[ix_b1, iy_b1, 0, "z"] = 0.0
                    solver.J[ix_b2, iy_b2, 0, "z"] = 0.0

                solver.one_step()
                if n >= nt_drive:
                    m = n - nt_drive
                    t_now = (n + 1) * solver.dt
                    kern = win[m] * np.exp(-2j * np.pi * f0 * t_now)
                    acc += kern * solver.E[:, :, 0, "z"]

            X, Y = np.meshgrid(grid.x[:nx], grid.y[:ny], indexing="ij")
            u_env = acc * np.exp(-1j * (kx * X + ky * Y)) * np.sqrt(eps_eff)
            u_env = 0.5 * (u_env + translate_supercell_field(u_env, is_envelope=True))
            u_vec = u_env.ravel()
            norm_u = float(np.linalg.norm(u_vec))
            if norm_u > 0.0:
                u_vec = u_vec / norm_u
            u_grid[i, j, :] = u_vec

    dk_step = float(ks_offset[1] - ks_offset[0])
    omega, c_patch = berry_curvature_from_envelopes(u_grid, dk_step, dk_step, phase_sign=phase_sign)
    return {
        "kx_grid": kx_grid,
        "ky_grid": ky_grid,
        "omega": omega,
        "c_patch": c_patch,
    }


__all__ = [
    "BZPath",
    "CircularInclusion",
    "HexagonalLattice",
    "PWE2D",
    "berry_curvature_from_envelopes",
    "dirac_berry_curvature",
    "fill_circular_inclusions",
    "fit_berry_curvature_patch",
    "fit_supercell_bands_at_k",
    "translate_supercell_field",
    "unfold_signal_pair",
    "unfold_supercell_weight",
]
