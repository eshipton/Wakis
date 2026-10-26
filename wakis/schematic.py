# Copyright 2026 Wakis contributors
# SPDX-License-Identifier: Apache-2.0

"""Clean simulation-setup schematics, discretization/excitation views, and setup tables for Wakis.

This module inspects live Wakis solver, grid, mesh, material, source, and
monitor objects to produce:

* **Figure A** (:func:`plot_schematic`, ``00a_setup_geometry.png``): the
  computational domain drawn to scale, materials as filled regions, boundaries
  shown graphically (CPML as a hatched band, PEC as a thick dark edge,
  periodic/Bloch as dashed edges with matching arrows) with one short tag per
  face placed outside the domain box, sources and monitors as distinct symbols
  with a short numbered tag (at most 6 labels inside the axes), one dimension
  arrow per physical axis, and all parameter details in a compact legend below
  the axes.
* **Figure B** (:func:`plot_mesh_and_excitation`,
  ``00b_setup_mesh_and_excitation.png``): up to 3 panels showing (1) the
  computational mesh with every *n*-th line and a 1× zoom inset at the smallest
  feature annotated with :math:`\\Delta`, cells per wavelength in the
  highest-index material, and cells across the smallest feature; (2) the wave
  packet in space at one instant along the propagation axis with domain/CPML
  shading; and (3) the source pulse time trace and spectrum with :math:`f_0`,
  bandwidth (stating ``-3 dB`` or ``1/e``), and shaded simulated band.
* **Setup table** (:func:`setup_table`): a Markdown table summarizing domain
  size and cell counts, every boundary face, materials (:math:`\\varepsilon_r`,
  :math:`\\mu_r`, model), sources, monitors, and discretization metrics.
* **Convenience saver** (:func:`save_setup_figures`): writes ``00a`` and ``00b``
  and returns the Markdown setup table string.
* **Field-map overlays** (:func:`overlay_geometry` and :func:`draw_mesh`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from scipy.constants import c as c_light
from scipy.constants import epsilon_0 as eps_0

from . import figstyle

__all__ = [
    "choose_length_unit",
    "draw_mesh",
    "extract_setup_info",
    "in_axes_texts",
    "overlay_geometry",
    "plot_mesh_and_excitation",
    "plot_schematic",
    "save_setup_figures",
    "setup_table",
]

THIN_SPACE: str = "\u2009"
SCHEMATIC_MIN_PT: float = 9.2

_UNIT_SCALES: dict[str, float] = {
    "nm": 1e9,
    "µm": 1e6,
    "um": 1e6,
    "mm": 1e3,
    "cm": 1e2,
    "m": 1.0,
}

_BC_SHORT_TAG: dict[str, str] = {
    "pec": "PEC",
    "electric": "PEC",
    "dirichlet": "PEC",
    "pmc": "PMC",
    "magnetic": "PMC",
    "periodic": "periodic",
    "bloch": "Bloch",
    "cpml": "CPML",
    "pml": "PML",
    "abc": "Mur ABC",
    "mur": "Mur ABC",
    "foextrap": "Mur ABC",
    "silver-muller": "ABC",
    "symmetry": "symmetry",
}

_BC_DISPLAY: dict[str, str] = {
    "pec": "PEC",
    "electric": "PEC",
    "dirichlet": "PEC",
    "pmc": "PMC",
    "magnetic": "PMC",
    "periodic": "Periodic",
    "bloch": "Bloch",
    "cpml": "CPML",
    "pml": "PML",
    "abc": "Mur ABC",
    "mur": "Mur ABC",
    "foextrap": "Mur ABC",
    "silver-muller": "Silver–Müller ABC",
    "symmetry": "Symmetry",
}

_MATERIAL_PALETTE: tuple[str, ...] = (
    "#DCE8F2",  # soft sky blue
    "#F5E4C8",  # warm sand
    "#D5ECE4",  # pale teal
    "#EED8E4",  # muted rose
    "#E4E2F0",  # soft lavender
    "#E6ECC8",  # pale olive
)


def choose_length_unit(span_m: float, unit: str = "auto") -> tuple[str, float]:
    """Select a human-readable length unit and scale factor from SI meters.

    Parameters
    ----------
    span_m : float
        Characteristic domain size in meters [m].
    unit : {'auto', 'nm', 'µm', 'um', 'mm', 'cm', 'm'}, optional
        Requested unit or ``'auto'`` (default) to pick sensible engineering
        units based on ``span_m``.

    Returns
    -------
    unit_label : str
        Upright display unit string (``'nm'``, ``'µm'``, ``'mm'``, ``'cm'``,
        or ``'m'``).
    scale : float
        Multiplier converting meters [m] to ``unit_label``.
    """
    if unit != "auto":
        if unit not in _UNIT_SCALES:
            raise ValueError(
                f"Unknown length unit {unit!r}; expected 'auto' or one of "
                f"{sorted(_UNIT_SCALES)}."
            )
        label = "µm" if unit == "um" else unit
        return label, _UNIT_SCALES[unit]

    span = abs(float(span_m))
    if span <= 0.0:
        return "m", 1.0
    if span < 1e-6 * (1.0 - 1e-6):
        return "nm", 1e9
    if span < 1e-3 * (1.0 - 1e-6):
        return "µm", 1e6
    if span < 1.0 * (1.0 - 1e-6):
        return "mm", 1e3
    return "m", 1.0


def _format_length(value_m: float, unit_label: str = "auto", sep: str = THIN_SPACE) -> str:
    """Format a length in meters [m] with thin space and upright engineering unit."""
    u, sc = choose_length_unit(abs(value_m), unit=unit_label)
    val = float(value_m) * sc
    if abs(val - round(val)) < 1e-3 * max(1.0, abs(val)):
        return f"{val:.0f}{sep}{u}"
    if abs(val) >= 10.0:
        return f"{val:.1f}{sep}{u}"
    return f"{val:.2f}{sep}{u}"


def _choose_freq_unit(freq_hz: float) -> tuple[str, float]:
    """Return ``(unit_label, scale_from_hz)`` for a frequency in Hz [1/s]."""
    f = abs(float(freq_hz))
    if f >= 1e12:
        return "THz", 1e-12
    if f >= 1e9:
        return "GHz", 1e-9
    if f >= 1e6:
        return "MHz", 1e-6
    if f >= 1e3:
        return "kHz", 1e-3
    return "Hz", 1.0


def _format_freq(freq_hz: float, sep: str = THIN_SPACE) -> str:
    """Format a frequency in Hz [1/s] with thin space and upright unit."""
    u, sc = _choose_freq_unit(freq_hz)
    val = float(freq_hz) * sc
    if abs(val) >= 100.0:
        return f"{val:.1f}{sep}{u}"
    if abs(val) >= 10.0:
        return f"{val:.2f}{sep}{u}"
    return f"{val:.2g}{sep}{u}"


def _choose_time_unit(time_s: float) -> tuple[str, float]:
    """Return ``(unit_label, scale_from_s)`` for a time duration in seconds [s]."""
    t = abs(float(time_s))
    if t <= 0.0:
        return "s", 1.0
    if t < 1e-12:
        return "fs", 1e15
    if t < 1e-9:
        return "ps", 1e12
    if t < 1e-6:
        return "ns", 1e9
    if t < 1e-3:
        return "µs", 1e6
    if t < 1.0:
        return "ms", 1e3
    return "s", 1.0


def _format_time(time_s: float, sep: str = THIN_SPACE) -> str:
    """Format a time in seconds [s] with thin space and upright unit."""
    u, sc = _choose_time_unit(time_s)
    val = float(time_s) * sc
    if abs(val) >= 10.0:
        return f"{val:.1f}{sep}{u}"
    return f"{val:.2f}{sep}{u}"


def _host_array(arr: Any) -> np.ndarray:
    """Convert a NumPy or CuPy array/Field to a host NumPy array."""
    if hasattr(arr, "get"):
        return np.asarray(arr.get())
    if hasattr(arr, "toarray"):
        out = arr.toarray()
        return np.asarray(out.get() if hasattr(out, "get") else out)
    return np.asarray(arr)


def _is_bloch_axis(raw_bc: Any, axis: int, solver: Any) -> bool:
    """Return True if ``axis`` uses Bloch-periodic boundary conditions."""
    if raw_bc is None:
        return False
    key = str(raw_bc).strip().lower()
    if key == "bloch":
        return True
    if key != "periodic" or solver is None:
        return False
    bloch_axes = getattr(solver, "_bloch_axes", None)
    if bloch_axes is not None and axis < len(bloch_axes) and bloch_axes[axis]:
        return True
    k_bloch = getattr(solver, "k_bloch", None)
    return bool(
        k_bloch is not None and axis < len(k_bloch) and float(k_bloch[axis]) != 0.0
    )


def _short_bc_tag(raw_bc: Any, axis: int, solver: Any) -> str:
    """Return a 1-2 word boundary tag for placement outside a domain face."""
    if raw_bc is None:
        return "PEC"
    key = str(raw_bc).strip().lower()
    if _is_bloch_axis(raw_bc, axis, solver):
        return "Bloch"
    return _BC_SHORT_TAG.get(key, str(raw_bc).strip().upper())


def _format_bc_label(
    raw_bc: Any,
    axis: int,
    solver: Any,
    face_thickness_m: float = 0.0,
) -> str:
    """Format a boundary condition into a detailed table/legend description."""
    if raw_bc is None:
        return "PEC"
    key = str(raw_bc).strip().lower()
    if _is_bloch_axis(raw_bc, axis, solver):
        axis_name = "xyz"[axis]
        k_bloch = getattr(solver, "k_bloch", None)
        kv = (
            float(k_bloch[axis])
            if (k_bloch is not None and axis < len(k_bloch))
            else 0.0
        )
        return f"Bloch (k_{axis_name} = {kv:.3g}{THIN_SPACE}rad/m)"
    base = _BC_DISPLAY.get(key, str(raw_bc).strip().upper())
    if key in ("cpml", "pml"):
        n_pml = int(getattr(solver, "n_pml", 0) or 0)
        if n_pml > 0 and face_thickness_m > 0.0:
            return f"{base} ({n_pml} cells, {_format_length(face_thickness_m)})"
        if n_pml > 0:
            return f"{base} ({n_pml} cells)"
    return base


def _resolve_grid_and_solver(obj: Any) -> tuple[Any, Any]:
    """Return ``(solver_or_none, grid_or_mesh)`` from a solver or grid input."""
    if hasattr(obj, "grid") and hasattr(obj.grid, "Nx"):
        return obj, obj.grid
    if hasattr(obj, "mesh"):
        return obj, obj.mesh
    return None, obj


def _axis_physical_nodes(grid: Any, axis: int) -> np.ndarray:
    """Return physical grid node coordinates [m] along ``axis`` (0=x, 1=y, 2=z)."""
    name = "xyz"[axis]
    nodes = np.asarray(getattr(grid, name), dtype=float)
    edge_ghosts = getattr(grid, "edge_ghosts", [0, 0, 0])
    if axis < len(edge_ghosts) and edge_ghosts[axis]:
        return nodes[:-1]
    max_attr = f"{name}max"
    if hasattr(grid, max_attr):
        hi = float(getattr(grid, max_attr))
        if hi > nodes[-1] + 1e-15:
            return np.append(nodes, hi)
    return nodes


def _describe_material(mat: Any, name_hint: str | None = None) -> dict[str, Any]:
    """Return normalized material metadata and legend/table strings."""
    if isinstance(mat, str):
        low = mat.strip().lower()
        if low == "pec":
            name = name_hint or "PEC"
            return {
                "name": name,
                "label": f"{name}, PEC (σ → ∞)",
                "eps_r": np.inf,
                "mu_r": 1.0,
                "sigma": np.inf,
                "model": "PEC",
            }
        if low in ("vacuum", "air"):
            name = name_hint or mat.capitalize()
            return {
                "name": name,
                "label": f"{name}, ε = 1.00",
                "eps_r": 1.0,
                "mu_r": 1.0,
                "sigma": 0.0,
                "model": "Isotropic",
            }
        name = name_hint or mat
        return {
            "name": name,
            "label": name,
            "eps_r": 1.0,
            "mu_r": 1.0,
            "sigma": 0.0,
            "model": "Custom",
        }

    cls_name = type(mat).__name__
    if hasattr(mat, "poles") or cls_name in (
        "Drude",
        "Lorentz",
        "Debye",
        "DispersiveMaterial",
    ):
        eps_inf = float(getattr(mat, "eps_inf", 1.0))
        model = getattr(mat, "name", None) or cls_name
        name = name_hint or str(model)
        return {
            "name": name,
            "label": f"{name}, {cls_name} (ε_∞ = {eps_inf:.2f})",
            "eps_r": eps_inf,
            "mu_r": 1.0,
            "sigma": 0.0,
            "model": cls_name,
        }

    if isinstance(mat, Iterable):
        vals = list(mat)
        eps_r = float(vals[0]) if len(vals) >= 1 else 1.0
        mu_r = float(vals[1]) if len(vals) >= 2 else 1.0
        sigma = float(vals[2]) if len(vals) >= 3 else 0.0
    elif isinstance(mat, (int, float, np.floating)):
        eps_r = float(mat)
        mu_r = 1.0
        sigma = 0.0
    else:
        name = name_hint or str(mat)
        return {
            "name": name,
            "label": name,
            "eps_r": 1.0,
            "mu_r": 1.0,
            "sigma": 0.0,
            "model": cls_name,
        }

    if np.isinf(eps_r):
        name = name_hint or "PEC"
        return {
            "name": name,
            "label": f"{name}, PEC",
            "eps_r": np.inf,
            "mu_r": mu_r,
            "sigma": np.inf,
            "model": "PEC",
        }

    name = name_hint or ("Vacuum" if abs(eps_r - 1.0) < 1e-6 and abs(mu_r - 1.0) < 1e-6 else "Dielectric")
    parts = [f"ε = {eps_r:.2f}"]
    if abs(mu_r - 1.0) > 1e-6:
        parts.append(f"μ = {mu_r:.2f}")
    if sigma > 0.0:
        parts.append(f"σ = {sigma:g}{THIN_SPACE}S/m")
    prop_str = ", ".join(parts)
    label = f"{name}, {prop_str}" if "ε" not in name else name
    return {
        "name": name,
        "label": label,
        "eps_r": eps_r,
        "mu_r": mu_r,
        "sigma": sigma,
        "model": "Lossy dielectric" if sigma > 0.0 else "Isotropic dielectric",
    }


def _extract_eps_map_2d(
    solver: Any,
    grid: Any,
    h_idx: int,
    v_idx: int,
    n_idx: int,
    cut_index: int | None = None,
) -> np.ndarray | None:
    """Extract a 2D relative permittivity map ``eps_r[Nh, Nv]`` on a cut plane."""
    if solver is None or not hasattr(solver, "ieps"):
        return None
    ieps = solver.ieps
    if not hasattr(ieps, "field_x"):
        return None
    shape = (int(grid.Nx), int(grid.Ny), int(grid.Nz))
    edge_ghosts = getattr(grid, "edge_ghosts", [0, 0, 0])
    phys_slices = [
        slice(0, shape[d] - (1 if d < len(edge_ghosts) and edge_ghosts[d] else 0))
        for d in range(3)
    ]
    comps = []
    for comp in ("x", "y", "z"):
        arr = _host_array(ieps[:, :, :, comp]).reshape(shape, order="F")
        arr = arr[tuple(phys_slices)]
        comps.append(arr)
    ieps_mean = np.mean(comps, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_r = np.where(ieps_mean <= 0.0, np.inf, 1.0 / (ieps_mean * eps_0))
    n_len = eps_r.shape[n_idx]
    idx = (n_len // 2) if cut_index is None else int(np.clip(cut_index, 0, n_len - 1))
    slc = [slice(None), slice(None), slice(None)]
    slc[n_idx] = idx
    plane_arr = eps_r[tuple(slc)]
    remaining = [d for d in range(3) if d != n_idx]
    if remaining != [h_idx, v_idx]:
        plane_arr = plane_arr.T
    return plane_arr


def _extract_contiguous_1d_slabs(
    nodes_h: np.ndarray,
    eps_profile: np.ndarray,
    bg_eps: float = 1.0,
    material_names: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Detect contiguous material intervals along 1D ``nodes_h`` from ``eps_r``."""
    regions: list[dict[str, Any]] = []
    # Ignore isolated subpixel boundary cells by finding plateau permittivity values
    diff = np.abs(eps_profile - bg_eps) > 1e-2
    if not np.any(diff):
        return regions

    n_cells = len(eps_profile)
    i = 0
    region_idx = 0
    while i < n_cells:
        if not diff[i]:
            i += 1
            continue
        j = i + 1
        while j < n_cells and diff[j]:
            j += 1
        seg = eps_profile[i:j]
        val = float(np.median(seg)) if len(seg) >= 3 else float( np.max(seg) )
        h0 = float(nodes_h[i])
        h1 = float(nodes_h[min(j, len(nodes_h) - 1)])
        region_key = f"region_{region_idx}"
        hint = None
        if material_names:
            hint = (
                material_names.get(region_key)
                or material_names.get(f"eps={val:g}")
                or (
                    next(iter(material_names.values()))
                    if len(material_names) == 1 and region_idx == 0
                    else None
                )
            )
        desc = _describe_material(val, name_hint=hint)
        regions.append(
            {
                "kind": "slab_1d",
                "h_bounds": (h0, h1),
                "eps_r": desc["eps_r"],
                "mu_r": desc["mu_r"],
                "sigma": desc["sigma"],
                "model": desc["model"],
                "name": desc["name"],
                "label": desc["label"],
                "color": "#B0B7BF"
                if np.isinf(val)
                else _MATERIAL_PALETTE[region_idx % len(_MATERIAL_PALETTE)],
            }
        )
        region_idx += 1
        i = j
    return regions


def _slice_to_bounds(
    slc_or_int: Any,
    axis_nodes: np.ndarray,
    axis_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Convert an integer index or ``slice`` on ``axis_nodes`` to ``(low, high)`` [m]."""
    n = len(axis_nodes)
    if slc_or_int is None or slc_or_int == slice(None):
        return axis_bounds
    if isinstance(slc_or_int, (int, np.integer)):
        idx = int(slc_or_int)
        if idx < 0:
            idx = max(0, n - 1 + idx)
        idx = int(np.clip(idx, 0, n - 1))
        val = float(axis_nodes[idx])
        return (val, val)
    if isinstance(slc_or_int, slice):
        start, stop, _ = slc_or_int.indices(max(1, n - 1))
        lo = float(axis_nodes[int(np.clip(start, 0, n - 1))])
        hi = float(axis_nodes[int(np.clip(stop, 0, n - 1))])
        return (min(lo, hi), max(lo, hi))
    return axis_bounds


def _center_size_to_bounds(
    obj: Any,
    nodes: list[np.ndarray],
    bounds: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Extract 3D ``[(x0, x1), (y0, y1), (z0, z1)]`` bounds [m]."""
    if hasattr(obj, "center") and hasattr(obj, "size"):
        c = np.asarray(obj.center, dtype=float)
        raw_s = obj.size
        out: list[tuple[float, float]] = []
        for d in range(3):
            sd = None if (raw_s is None or raw_s[d] is None) else float(raw_s[d])
            if sd is None or sd >= (bounds[d][1] - bounds[d][0]):
                out.append(bounds[d])
            else:
                lo = max(bounds[d][0], float(c[d] - 0.5 * sd))
                hi = min(bounds[d][1], float(c[d] + 0.5 * sd))
                out.append((lo, hi))
        return out
    return [
        _slice_to_bounds(getattr(obj, "xs", None), nodes[0], bounds[0]),
        _slice_to_bounds(getattr(obj, "ys", None), nodes[1], bounds[1]),
        _slice_to_bounds(getattr(obj, "zs", None), nodes[2], bounds[2]),
    ]


def _parse_direction_vec(direction: Any) -> np.ndarray:
    """Convert a direction string (``'+z'``, ``'-x'``) or 3-vector to a unit 3-vector."""
    if isinstance(direction, str):
        s = direction.strip().lower()
        sign = -1.0 if s.startswith("-") else 1.0
        ax = s[-1]
        v = np.zeros(3)
        if ax in "xyz":
            v["xyz".index(ax)] = sign
        else:
            v[2] = 1.0
        return v
    arr = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else np.array([0.0, 0.0, 1.0])


def _extract_pulse_metadata(src_time: Any) -> dict[str, Any] | None:
    """Extract center frequency, bandwidth, and callable waveform from ``src_time``."""
    if src_time is None:
        return None
    f0 = getattr(src_time, "frequency", getattr(src_time, "freq", None))
    fwidth = getattr(src_time, "fwidth", None)
    cutoff = float(getattr(src_time, "cutoff", 5.0))
    if f0 is None:
        return None
    f0 = float(f0)
    fwidth = float(fwidth) if fwidth is not None else 0.2 * f0
    t_peak = getattr(src_time, "t0", getattr(src_time, "peak_time", None))
    if t_peak is None:
        t_peak = cutoff / (np.pi * max(fwidth, 1e-30))
    t_end = getattr(src_time, "end_time", 2.0 * float(t_peak))
    return {
        "f0": f0,
        "fwidth": fwidth,
        "t_peak": float(t_peak),
        "t_end": float(t_end),
        "src_time": src_time,
    }


def _parse_source(
    src: Any,
    nodes: list[np.ndarray],
    bounds: list[tuple[float, float]],
    h_idx: int,
    v_idx: int,
    n_idx: int,
) -> dict[str, Any]:
    """Normalize a Wakis source object into a 2D cut-plane schematic record."""
    if isinstance(src, Mapping):
        return dict(src)

    cls_name = type(src).__name__
    src_time = getattr(src, "src_time", None)
    pulse = _extract_pulse_metadata(src_time)
    if pulse is None and hasattr(src, "f"):
        f0 = float(getattr(src, "f"))
        pulse = {
            "f0": f0,
            "fwidth": 0.2 * f0,
            "t_peak": 2.5 / max(f0, 1.0),
            "t_end": 5.0 / max(f0, 1.0),
            "src_time": None,
        }

    # 1. TFSFSource
    if cls_name == "TFSFSource" or (
        hasattr(src, "direction") and (hasattr(src, "polarization") or hasattr(src, "pol"))
    ):
        s_bounds = _center_size_to_bounds(src, nodes, bounds)
        direction = _parse_direction_vec(getattr(src, "direction", "+z"))
        pol_raw = getattr(src, "polarization", getattr(src, "pol", "x"))
        pol = _parse_direction_vec(pol_raw)
        h_vec = np.cross(direction, pol)
        h_ax = "xyz"[h_idx]
        pos_str = f"{h_ax} ∈ [{_format_length(s_bounds[h_idx][0])}, {_format_length(s_bounds[h_idx][1])}]"
        pol_char = "xyz"[int(np.argmax(np.abs(pol)))]
        return {
            "kind": "tfsf",
            "type_name": "TFSFSource",
            "short_word": "TFSF",
            "desc": f"TFSF box E_{pol_char} (TFSFSource, {pos_str})",
            "label": getattr(src, "label", None) or f"TFSFSource ({pos_str})",
            "pos_str": pos_str,
            "polarization": f"E_{pol_char}",
            "h_bounds": s_bounds[h_idx],
            "v_bounds": s_bounds[v_idx],
            "k_vec": (float(direction[h_idx]), float(direction[v_idx]), float(direction[n_idx])),
            "e_vec": (float(pol[h_idx]), float(pol[v_idx]), float(pol[n_idx])),
            "h_vec": (float(h_vec[h_idx]), float(h_vec[v_idx]), float(h_vec[n_idx])),
            "pulse": pulse,
        }

    # 2. LumpedPort / LumpedElement
    if cls_name in ("LumpedPort", "LumpedElement") or hasattr(src, "R") or hasattr(src, "Z0"):
        s_bounds = _center_size_to_bounds(src, nodes, bounds)
        comp = str(getattr(src, "component", "z")).lower()[-1]
        comp_idx = "xyz".index(comp) if comp in "xyz" else 2
        e_vec3 = [0.0, 0.0, 0.0]
        e_vec3[comp_idx] = 1.0
        r_ohm = getattr(src, "Z0", getattr(src, "R", getattr(src, "impedance", None)))
        h_ax = "xyz"[h_idx]
        pos_str = f"{h_ax} = {_format_length(0.5 * sum(s_bounds[h_idx]))}"
        ohm_str = f"{float(r_ohm):g}{THIN_SPACE}Ω, " if r_ohm is not None else ""
        return {
            "kind": "lumped",
            "type_name": cls_name,
            "short_word": "port",
            "desc": f"lumped port E_{comp} ({cls_name}, {ohm_str}{pos_str})",
            "label": getattr(src, "label", None) or f"{cls_name} ({ohm_str}{pos_str})",
            "pos_str": pos_str,
            "polarization": f"E_{comp}",
            "h_bounds": s_bounds[h_idx],
            "v_bounds": s_bounds[v_idx],
            "component": f"E{comp}",
            "e_vec": (e_vec3[h_idx], e_vec3[v_idx], e_vec3[n_idx]),
            "pulse": pulse,
        }

    # 3. ModeSource
    if cls_name == "ModeSource" or hasattr(src, "mode_index"):
        normal_d = int(getattr(src, "normal", 2))
        idx_pos = int(getattr(src, "index", 0))
        pos_m = float(nodes[normal_d][int(np.clip(idx_pos, 0, len(nodes[normal_d]) - 1))])
        s_bounds = list(bounds)
        s_bounds[normal_d] = (pos_m, pos_m)
        sign = 1.0 if getattr(src, "direction", "+") != "-" else -1.0
        k3 = [0.0, 0.0, 0.0]
        k3[normal_d] = sign
        m_idx = int(getattr(src, "mode_index", 0))
        pos_str = f"{'xyz'[normal_d]} = {_format_length(pos_m)}"
        return {
            "kind": "sheet",
            "type_name": "ModeSource",
            "short_word": f"mode {m_idx}",
            "desc": f"mode {m_idx} (ModeSource, {pos_str})",
            "label": getattr(src, "label", None) or f"ModeSource (mode {m_idx})",
            "pos_str": pos_str,
            "polarization": f"mode {m_idx}",
            "h_bounds": s_bounds[h_idx],
            "v_bounds": s_bounds[v_idx],
            "k_vec": (k3[h_idx], k3[v_idx], k3[n_idx]),
            "e_vec": (0.0, 1.0, 0.0),
            "h_vec": (0.0, 0.0, 1.0),
            "pulse": pulse,
        }

    # 4. Beam / LinearizedBeam
    if cls_name in ("Beam", "LinearizedBeam") or hasattr(src, "sigmaz") or hasattr(src, "v0"):
        x0 = float(getattr(src, "xsource", getattr(src, "x0", 0.5 * sum(bounds[0]))))
        y0 = float(getattr(src, "ysource", getattr(src, "y0", 0.5 * sum(bounds[1]))))
        coords3 = [x0, y0, 0.5 * sum(bounds[2])]
        s_bounds = list(bounds)
        for d in (0, 1):
            s_bounds[d] = (coords3[d], coords3[d])
        k3 = [0.0, 0.0, 1.0]
        beta = getattr(src, "beta", None)
        pos_str = f"x = {_format_length(x0)}, y = {_format_length(y0)}"
        beta_str = f"β = {float(beta):.2f}" if beta is not None else "particle beam"
        return {
            "kind": "beam",
            "type_name": cls_name,
            "short_word": "beam",
            "desc": f"beam ({cls_name}, {beta_str})",
            "label": getattr(src, "label", None) or f"{cls_name} ({beta_str})",
            "pos_str": pos_str,
            "polarization": "J_z",
            "h_bounds": s_bounds[h_idx],
            "v_bounds": s_bounds[v_idx],
            "k_vec": (k3[h_idx], k3[v_idx], k3[n_idx]),
            "pulse": pulse,
        }

    # 5. CurrentSource / MagneticCurrentSource / Dipole
    s_bounds = _center_size_to_bounds(src, nodes, bounds)
    comp_raw = str(getattr(src, "component", "Ex"))
    is_mag = cls_name == "MagneticCurrentSource" or comp_raw.startswith(("M", "H"))
    pol_char = comp_raw[-1].lower() if comp_raw[-1].lower() in "xyz" else "x"
    pol_idx = "xyz".index(pol_char)
    e_vec3 = [0.0, 0.0, 0.0]
    e_vec3[pol_idx] = 1.0

    sheet_axes = [d for d in range(3) if abs(s_bounds[d][1] - s_bounds[d][0]) < 1e-12]
    prop_idx = sheet_axes[0] if len(sheet_axes) == 1 else 2
    k3 = [0.0, 0.0, 0.0]
    k3[prop_idx] = 1.0
    h3 = np.cross(k3, e_vec3)

    h_span = abs(s_bounds[h_idx][1] - s_bounds[h_idx][0])
    v_span = abs(s_bounds[v_idx][1] - s_bounds[v_idx][0])
    kind = "point" if (h_span < 1e-12 and v_span < 1e-12) else "sheet"
    if len(sheet_axes) == 1:
        ax_c = "xyz"[sheet_axes[0]]
        pos_str = f"{ax_c} = {_format_length(s_bounds[sheet_axes[0]][0])}"
    else:
        pos_str = (
            f"{'xyz'[h_idx]} = {_format_length(0.5 * sum(s_bounds[h_idx]))}, "
            f"{'xyz'[v_idx]} = {_format_length(0.5 * sum(s_bounds[v_idx]))}"
        )
    curr_sym = f"{'M' if is_mag else 'J'}_{pol_char}"
    role_word = "dipole" if kind == "point" else "current sheet"
    return {
        "kind": kind,
        "type_name": cls_name,
        "short_word": curr_sym,
        "desc": f"{role_word} {curr_sym} ({cls_name}, {pos_str})",
        "label": getattr(src, "label", None) or f"{role_word} {curr_sym} ({pos_str})",
        "pos_str": pos_str,
        "polarization": f"{curr_sym} ({'H' if is_mag else 'E'}_{pol_char})",
        "component": f"{'M' if is_mag else 'E'}{pol_char}",
        "h_bounds": s_bounds[h_idx],
        "v_bounds": s_bounds[v_idx],
        "k_vec": (float(k3[h_idx]), float(k3[v_idx]), float(k3[n_idx])),
        "e_vec": (float(e_vec3[h_idx]), float(e_vec3[v_idx]), float(e_vec3[n_idx])),
        "h_vec": (float(h3[h_idx]), float(h3[v_idx]), float(h3[n_idx])),
        "pulse": pulse,
    }


def _extract_monitor_freqs(mon: Any) -> np.ndarray | None:
    """Return a 1D array of monitor frequencies [Hz] if present."""
    for attr in ("frequencies", "freqs", "f"):
        val = getattr(mon, attr, None)
        if val is not None:
            arr = np.atleast_1d(np.asarray(val, dtype=float))
            if arr.size > 0:
                return arr
    return None


def _format_freq_band(freqs: np.ndarray | None) -> str:
    """Summarize a monitor frequency array for tables and legends."""
    if freqs is None or freqs.size == 0:
        return "time-domain"
    if freqs.size == 1:
        return f"f = {_format_freq(float(freqs[0]))}"
    f_min, f_max = float(np.min(freqs)), float(np.max(freqs))
    u, sc = _choose_freq_unit(f_max)
    return f"{freqs.size} freqs in [{f_min * sc:.1f}, {f_max * sc:.1f}]{THIN_SPACE}{u}"


def _parse_monitor(
    mon: Any,
    index: int,
    nodes: list[np.ndarray],
    bounds: list[tuple[float, float]],
    h_idx: int,
    v_idx: int,
    n_idx: int,
) -> dict[str, Any]:
    """Normalize a Wakis monitor object into a 2D cut-plane schematic record."""
    if isinstance(mon, Mapping):
        return dict(mon)

    cls_name = type(mon).__name__
    freqs = _extract_monitor_freqs(mon)
    freq_str = _format_freq_band(freqs)

    if cls_name == "Near2FarMonitor" or (hasattr(mon, "faces") and cls_name != "FluxMonitor"):
        faces = getattr(mon, "faces", [])
        h_vals, v_vals = [], []
        for f in faces:
            fb = _center_size_to_bounds(f, nodes, bounds)
            h_vals.extend(fb[h_idx])
            v_vals.extend(fb[v_idx])
        hb = (min(h_vals), max(h_vals)) if h_vals else bounds[h_idx]
        vb = (min(v_vals), max(v_vals)) if v_vals else bounds[v_idx]
        pos_str = f"{'xyz'[h_idx]} ∈ [{_format_length(hb[0])}, {_format_length(hb[1])}]"
        return {
            "kind": "box",
            "type_name": cls_name,
            "short_word": "N2F",
            "desc": f"near-to-far box ({cls_name}, {pos_str})",
            "label": getattr(mon, "label", None) or "Near2FarMonitor",
            "pos_str": pos_str,
            "freq_str": freq_str,
            "frequencies": freqs,
            "h_bounds": hb,
            "v_bounds": vb,
        }

    if hasattr(mon, "normal") and hasattr(mon, "index") and not hasattr(mon, "center"):
        normal_d = int(getattr(mon, "normal", 2))
        idx_pos = int(getattr(mon, "index", 0))
        pos_m = float(nodes[normal_d][int(np.clip(idx_pos, 0, len(nodes[normal_d]) - 1))])
        m_bounds = list(bounds)
        m_bounds[normal_d] = (pos_m, pos_m)
        ax_c = "xyz"[normal_d]
        pos_str = f"{ax_c} = {_format_length(pos_m)}"
        short_w = "flux" if "Flux" in cls_name else ("mode" if "Mode" in cls_name else "mon")
        return {
            "kind": "plane",
            "type_name": cls_name,
            "short_word": short_w,
            "desc": f"{short_w} monitor ({cls_name}, {pos_str})",
            "label": getattr(mon, "label", None) or f"{cls_name} ({pos_str})",
            "pos_str": pos_str,
            "freq_str": freq_str,
            "frequencies": freqs,
            "h_bounds": m_bounds[h_idx],
            "v_bounds": m_bounds[v_idx],
            "normal_axis": normal_d,
        }

    m_bounds = _center_size_to_bounds(mon, nodes, bounds)
    plane_axes = [d for d in range(3) if abs(m_bounds[d][1] - m_bounds[d][0]) < 1e-12]
    h_span = abs(m_bounds[h_idx][1] - m_bounds[h_idx][0])
    v_span = abs(m_bounds[v_idx][1] - m_bounds[v_idx][0])
    kind = "probe" if (h_span < 1e-12 and v_span < 1e-12) else "plane"
    h_ax = "xyz"[h_idx]
    v_ax = "xyz"[v_idx]
    if abs(m_bounds[h_idx][1] - m_bounds[h_idx][0]) < 1e-12 and abs(m_bounds[v_idx][1] - m_bounds[v_idx][0]) < 1e-12:
        pos_str = f"{h_ax} = {_format_length(m_bounds[h_idx][0])}"
    elif abs(m_bounds[h_idx][1] - m_bounds[h_idx][0]) < 1e-12:
        pos_str = f"{h_ax} = {_format_length(m_bounds[h_idx][0])}"
    elif len(plane_axes) == 1 and plane_axes[0] != n_idx:
        ax_c = "xyz"[plane_axes[0]]
        pos_str = f"{ax_c} = {_format_length(m_bounds[plane_axes[0]][0])}"
    elif h_span > 1e-12 and v_span > 1e-12 and abs(bounds[v_idx][1] - bounds[v_idx][0]) >= 0.15 * abs(bounds[h_idx][1] - bounds[h_idx][0]):
        pos_str = (
            f"{h_ax} ∈ [{_format_length(m_bounds[h_idx][0])}, {_format_length(m_bounds[h_idx][1])}], "
            f"{v_ax} ∈ [{_format_length(m_bounds[v_idx][0])}, {_format_length(m_bounds[v_idx][1])}]"
        )
    else:
        pos_str = f"{h_ax} ∈ [{_format_length(m_bounds[h_idx][0])}, {_format_length(m_bounds[h_idx][1])}]"
    short_w = "flux" if "Flux" in cls_name else ("DFT" if "DFT" in cls_name else "mon")
    return {
        "kind": kind,
        "type_name": cls_name,
        "short_word": short_w,
        "desc": f"{short_w} monitor ({cls_name}, {pos_str})",
        "label": getattr(mon, "label", None) or f"{cls_name} ({pos_str})",
        "pos_str": pos_str,
        "freq_str": freq_str,
        "frequencies": freqs,
        "h_bounds": m_bounds[h_idx],
        "v_bounds": m_bounds[v_idx],
    }


def _parse_probe(
    probe: Any,
    index: int,
    bounds: list[tuple[float, float]],
    h_idx: int,
    v_idx: int,
) -> dict[str, Any]:
    """Normalize a user probe specification into a schematic monitor record."""
    if isinstance(probe, Mapping):
        pos = probe.get("pos", probe.get("position", (0.0, 0.0, 0.0)))
        lbl = str(probe.get("label", f"Probe {index + 1}"))
    elif isinstance(probe, (int, float, np.floating)):
        pos = [0.5 * sum(bounds[0]), 0.5 * sum(bounds[1]), 0.5 * sum(bounds[2])]
        pos[h_idx] = float(probe)
        lbl = f"Probe {index + 1}"
    else:
        pos = list(probe)
        if len(pos) == 2:
            pos = [pos[0], pos[1], 0.0]
        lbl = f"Probe {index + 1}"

    coords3 = [
        float(pos[0]) if len(pos) > 0 else 0.5 * sum(bounds[0]),
        float(pos[1]) if len(pos) > 1 else 0.5 * sum(bounds[1]),
        float(pos[2]) if len(pos) > 2 else 0.5 * sum(bounds[2]),
    ]
    h_val = coords3[h_idx]
    v_val = coords3[v_idx]
    h_ax = "xyz"[h_idx]
    v_ax = "xyz"[v_idx]
    if abs(bounds[v_idx][1] - bounds[v_idx][0]) >= 0.15 * abs(bounds[h_idx][1] - bounds[h_idx][0]):
        pos_str = f"({h_ax}, {v_ax}) = ({_format_length(h_val)}, {_format_length(v_val)})"
    else:
        pos_str = f"{h_ax} = {_format_length(h_val)}"
    return {
        "kind": "probe",
        "type_name": "Probe",
        "short_word": "probe",
        "desc": f"probe ({lbl})" if ("(" in lbl or pos_str in lbl) else f"probe ({lbl}, {pos_str})",
        "label": lbl,
        "pos_str": pos_str,
        "freq_str": "time-domain",
        "frequencies": None,
        "h_bounds": (h_val, h_val),
        "v_bounds": (v_val, v_val),
    }


def extract_setup_info(
    solver_or_grid: Any,
    plane: str = "auto",
    cut_coord: float | None = None,
    structures: Sequence[Any] | None = None,
    material_names: Mapping[str, str] | None = None,
    sources: Any = None,
    monitors: Any = None,
    probes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Extract domain, boundary, material, source, and monitor metadata.

    Parameters
    ----------
    solver_or_grid : SolverFIT3D, GridFIT3D, SubgridFIT2D, UnstructuredFIT2D, or mesh
        Wakis solver, grid, or mesh instance to inspect.
    plane : {'auto', 'xy', 'xz', 'yz', 'zx', 'zy'}, optional
        2D cut plane to project onto.
    cut_coord : float or None, optional
        Coordinate [m] along the normal axis for 3D cuts.
    structures : sequence of shape objects, optional
        Explicit geometric shapes if not already attached to ``solver``.
    material_names : mapping of str to str, optional
        Dictionary mapping solid keys or ``'region_i'`` to human-readable
        material names (e.g. ``{'region_0': 'SiO₂'}``).
    sources : source object or sequence of source objects, optional
        Source(s) to include if not stored on ``solver.source``.
    monitors : monitor object or sequence of monitor objects, optional
        Monitor(s) to include if not stored on ``solver.monitors``.
    probes : sequence of tuple or dict, optional
        Point probes not stored as monitor objects.

    Returns
    -------
    info : dict
        Normalized dictionary describing the cut plane, bounds, boundaries,
        PML thickness, materials, sources, and monitors.
    """
    solver, grid = _resolve_grid_and_solver(solver_or_grid)
    material_names = dict(material_names or {})

    is_2d_mesh = not hasattr(grid, "Nz") and (
        hasattr(grid, "xc")
        or hasattr(grid, "Lx")
        or (hasattr(grid, "nodes") and getattr(grid, "dim", 2) == 2)
        or hasattr(grid, "vertices")
    )
    if plane == "auto":
        if is_2d_mesh:
            plane_str = "xy"
        else:
            nx = int(getattr(grid, "Nx", 1))
            ny = int(getattr(grid, "Ny", 1))
            nz = int(getattr(grid, "Nz", 1))
            if nx <= 2 and ny <= 2 and nz > 2:
                plane_str = "zx"
            elif ny <= 2 and nx > 2 and nz > 2:
                plane_str = "xz"
            elif nz <= 2 and nx > 2 and ny > 2:
                plane_str = "xy"
            else:
                plane_str = "xz"
    else:
        plane_str = plane.strip().lower()

    if len(plane_str) != 2 or any(c not in "xyz" for c in plane_str) or plane_str[0] == plane_str[1]:
        raise ValueError(
            f"Invalid cut plane {plane!r}; expected 'auto', 'xy', 'xz', 'yz', 'zx', or 'zy'."
        )

    h_axis_char, v_axis_char = plane_str[0], plane_str[1]
    n_axis_char = next(c for c in "xyz" if c not in plane_str)
    h_idx = "xyz".index(h_axis_char)
    v_idx = "xyz".index(v_axis_char)
    n_idx = "xyz".index(n_axis_char)

    if hasattr(grid, "x") and hasattr(grid, "y") and hasattr(grid, "z"):
        nodes = [_axis_physical_nodes(grid, d) for d in range(3)]
        bounds = [(float(n[0]), float(n[-1])) for n in nodes]
        cell_counts = [max(1, len(n) - 1) for n in nodes]
    elif hasattr(grid, "xc") and hasattr(grid, "yc"):
        xc = np.asarray(grid.xc, dtype=float)
        yc = np.asarray(grid.yc, dtype=float)
        nodes = [xc, yc, np.array([0.0, 1.0])]
        bounds = [(float(xc[0]), float(xc[-1])), (float(yc[0]), float(yc[-1])), (0.0, 1.0)]
        cell_counts = [max(1, len(xc) - 1), max(1, len(yc) - 1), 1]
    elif hasattr(grid, "Lx") and hasattr(grid, "Ly"):
        lx, ly = float(grid.Lx), float(grid.Ly)
        nx = int(getattr(grid, "Nx", 16))
        ny = int(getattr(grid, "Ny", 16))
        nodes = [
            np.linspace(0.0, lx, nx + 1),
            np.linspace(0.0, ly, ny + 1),
            np.array([0.0, 1.0]),
        ]
        bounds = [(0.0, lx), (0.0, ly), (0.0, 1.0)]
        cell_counts = [nx, ny, 1]
    elif hasattr(grid, "nodes") or hasattr(grid, "vertices"):
        verts = np.asarray(getattr(grid, "nodes", getattr(grid, "vertices", None)), dtype=float)
        dim = verts.shape[1]
        b_min = verts.min(axis=0)
        b_max = verts.max(axis=0)
        if dim == 2:
            bounds = [
                (float(b_min[0]), float(b_max[0])),
                (float(b_min[1]), float(b_max[1])),
                (0.0, 1.0),
            ]
        else:
            bounds = [(float(b_min[d]), float(b_max[d])) for d in range(3)]
        nodes = [np.linspace(b[0], b[1], 11) for b in bounds]
        cell_counts = [int(getattr(grid, "n_cells", 10)), int(getattr(grid, "n_cells", 10)), 1]
    else:
        raise TypeError(
            f"Cannot extract domain geometry from object of type {type(solver_or_grid).__name__}."
        )

    n_bounds = bounds[n_idx]
    cut_val = 0.5 * (n_bounds[0] + n_bounds[1]) if cut_coord is None else float(cut_coord)

    bc_low_raw = list(getattr(solver, "bc_low", ["pec", "pec", "pec"]))
    bc_high_raw = list(getattr(solver, "bc_high", ["pec", "pec", "pec"]))
    if solver is None and hasattr(solver_or_grid, "bc"):
        bc_val = getattr(solver_or_grid, "bc")
        bc_low_raw = [bc_val] * 3
        bc_high_raw = [bc_val] * 3
    elif solver is not None and hasattr(solver, "bc") and not hasattr(solver, "bc_low"):
        bc_val = getattr(solver, "bc")
        bc_low_raw = [bc_val] * 3
        bc_high_raw = [bc_val] * 3

    n_pml = int(getattr(solver, "n_pml", 0) or 0)
    pml_thickness: dict[str, float] = {}
    bc_labels: dict[str, str] = {}
    bc_tags: dict[str, str] = {}
    for d, ax_c in enumerate("xyz"):
        n_arr = nodes[d]
        low_bc = str(bc_low_raw[d]).lower() if d < len(bc_low_raw) else "pec"
        high_bc = str(bc_high_raw[d]).lower() if d < len(bc_high_raw) else "pec"
        thick_low = (
            float(n_arr[min(n_pml, len(n_arr) - 1)] - n_arr[0])
            if low_bc in ("cpml", "pml") and n_pml > 0 and len(n_arr) > 1
            else 0.0
        )
        thick_high = (
            float(n_arr[-1] - n_arr[max(0, len(n_arr) - 1 - n_pml)])
            if high_bc in ("cpml", "pml") and n_pml > 0 and len(n_arr) > 1
            else 0.0
        )
        pml_thickness[f"{ax_c}_low"] = thick_low
        pml_thickness[f"{ax_c}_high"] = thick_high
        bc_labels[f"{ax_c}_low"] = _format_bc_label(bc_low_raw[d], d, solver, thick_low)
        bc_labels[f"{ax_c}_high"] = _format_bc_label(bc_high_raw[d], d, solver, thick_high)
        bc_tags[f"{ax_c}_low"] = _short_bc_tag(bc_low_raw[d], d, solver)
        bc_tags[f"{ax_c}_high"] = _short_bc_tag(bc_high_raw[d], d, solver)

    materials_list: list[dict[str, Any]] = []
    bg_spec = getattr(solver, "bg", [1.0, 1.0, 0.0])
    bg_eps = float(bg_spec[0]) if isinstance(bg_spec, Sequence) else 1.0
    bg_name = material_names.get("background", "Vacuum" if abs(bg_eps - 1.0) < 1e-6 else "Background")
    bg_desc = _describe_material(bg_spec, name_hint=bg_name)

    all_structs = list(structures or ())
    if not all_structs and solver is not None:
        all_structs = list(getattr(solver, "_assigned_structures", ()))
    for idx, st in enumerate(all_structs):
        cls_n = type(st).__name__
        hint = getattr(st, "name", None) or material_names.get(f"region_{idx}")
        if hint is None and len(material_names) == 1 and idx == 0:
            hint = next(iter(material_names.values()))
        desc = _describe_material(getattr(st, "material", 1.0), name_hint=hint)
        color = (
            "#B0B7BF"
            if str(getattr(st, "material", "")).lower() == "pec" or np.isinf(desc["eps_r"])
            else _MATERIAL_PALETTE[idx % len(_MATERIAL_PALETTE)]
        )
        if cls_n == "Box" or (hasattr(st, "bounds") and not hasattr(st, "radius")):
            if hasattr(st, "bounds"):
                low, high = (np.asarray(b, dtype=float) for b in st.bounds)
            else:
                low = np.asarray(st.low, dtype=float)
                high = np.asarray(st.high, dtype=float)
            h0 = float(max(bounds[h_idx][0], low[h_idx]))
            h1 = float(min(bounds[h_idx][1], high[h_idx]))
            v0 = float(max(bounds[v_idx][0], low[v_idx]))
            v1 = float(min(bounds[v_idx][1], high[v_idx]))
            if h1 > h0 and v1 > v0:
                materials_list.append(
                    {
                        "kind": "rect",
                        "h_bounds": (h0, h1),
                        "v_bounds": (v0, v1),
                        **desc,
                        "color": color,
                    }
                )
        elif cls_n == "Cylinder" or hasattr(st, "radius"):
            axis_d = int(getattr(st, "axis", 2))
            center = np.asarray(getattr(st, "center", [0.0, 0.0, 0.0]), dtype=float)
            radius = float(st.radius)
            if axis_d == n_idx:
                materials_list.append(
                    {
                        "kind": "circle",
                        "center": (float(center[h_idx]), float(center[v_idx])),
                        "radius": radius,
                        "h_bounds": (float(center[h_idx] - radius), float(center[h_idx] + radius)),
                        "v_bounds": (float(center[v_idx] - radius), float(center[v_idx] + radius)),
                        **desc,
                        "color": color,
                    }
                )
            else:
                height = float(getattr(st, "height", np.inf))
                h0 = float(center[h_idx] - (radius if h_idx != axis_d else 0.5 * height))
                h1 = float(center[h_idx] + (radius if h_idx != axis_d else 0.5 * height))
                v0 = float(center[v_idx] - (radius if v_idx != axis_d else 0.5 * height))
                v1 = float(center[v_idx] + (radius if v_idx != axis_d else 0.5 * height))
                h0, h1 = max(bounds[h_idx][0], h0), min(bounds[h_idx][1], h1)
                v0, v1 = max(bounds[v_idx][0], v0), min(bounds[v_idx][1], v1)
                if h1 > h0 and v1 > v0:
                    materials_list.append(
                        {
                            "kind": "rect",
                            "h_bounds": (h0, h1),
                            "v_bounds": (v0, v1),
                            **desc,
                            "color": color,
                        }
                    )
        elif hasattr(st, "vertices"):
            verts = np.asarray(st.vertices, dtype=float)
            materials_list.append(
                {
                    "kind": "polygon",
                    "vertices": verts,
                    "h_bounds": (float(verts[:, 0].min()), float(verts[:, 0].max())),
                    "v_bounds": (float(verts[:, 1].min()), float(verts[:, 1].max())),
                    **desc,
                    "color": color,
                }
            )

    stl_solids = getattr(grid, "stl_solids", None) or {}
    stl_materials = getattr(grid, "stl_materials", None) or {}
    for idx, key in enumerate(stl_solids):
        mat_spec = stl_materials.get(key, [1.0, 1.0, 0.0])
        hint = material_names.get(key, str(key))
        desc = _describe_material(mat_spec, name_hint=hint)
        color = _MATERIAL_PALETTE[(len(materials_list) + idx) % len(_MATERIAL_PALETTE)]
        if hasattr(grid, "grid") and key in getattr(grid.grid, "cell_data", {}):
            mask = np.reshape(
                np.asarray(grid.grid.cell_data[key]),
                (int(grid.Nx), int(grid.Ny), int(grid.Nz)),
            )
            occ = np.any(mask > 0, axis=n_idx)
            if remaining := [d for d in range(3) if d != n_idx]:
                if remaining != [h_idx, v_idx]:
                    occ = occ.T
            if np.any(occ):
                hi = np.where(np.any(occ, axis=1))[0]
                vi = np.where(np.any(occ, axis=0))[0]
                h_n, v_n = nodes[h_idx], nodes[vertical_idx := v_idx]
                h0, h1 = float(h_n[hi[0]]), float(h_n[min(hi[-1] + 1, len(h_n) - 1)])
                v0, v1 = float(v_n[vi[0]]), float(v_n[min(vi[-1] + 1, len(v_n) - 1)])
                materials_list.append(
                    {
                        "kind": "rect",
                        "h_bounds": (h0, h1),
                        "v_bounds": (v0, v1),
                        **desc,
                        "color": color,
                        "mask_2d": occ,
                    }
                )

    eps_map_2d = _extract_eps_map_2d(solver, grid, h_idx, v_idx, n_idx)
    if not materials_list and eps_map_2d is not None:
        profile_h = np.median(eps_map_2d, axis=1)
        slabs = _extract_contiguous_1d_slabs(
            nodes[h_idx], profile_h, bg_eps=bg_eps, material_names=material_names
        )
        for s in slabs:
            s["kind"] = "rect"
            s["v_bounds"] = bounds[v_idx]
            materials_list.append(s)
        if not materials_list and np.any(np.abs(eps_map_2d - bg_eps) > 1e-2):
            diff_mask = np.abs(eps_map_2d - bg_eps) > 1e-2
            vals = np.unique(np.round(eps_map_2d[diff_mask], 3))
            for idx, val in enumerate(vals[:6]):
                m = (
                    np.isinf(eps_map_2d)
                    if np.isinf(val)
                    else np.abs(eps_map_2d - val) < 0.05 * max(1.0, abs(val))
                )
                if not np.any(m):
                    continue
                hi = np.where(np.any(m, axis=1))[0]
                vi = np.where(np.any(m, axis=0))[0]
                h_n, v_n = nodes[h_idx], nodes[v_idx]
                h0, h1 = float(h_n[hi[0]]), float(h_n[min(hi[-1] + 1, len(h_n) - 1)])
                v0, v1 = float(v_n[vi[0]]), float(v_n[min(vi[-1] + 1, len(v_n) - 1)])
                hint = material_names.get(f"region_{idx}") or material_names.get(f"eps={val:g}")
                desc = _describe_material(val, name_hint=hint)
                materials_list.append(
                    {
                        "kind": "rect",
                        "h_bounds": (h0, h1),
                        "v_bounds": (v0, v1),
                        **desc,
                        "color": "#B0B7BF"
                        if np.isinf(val)
                        else _MATERIAL_PALETTE[idx % len(_MATERIAL_PALETTE)],
                        "mask_2d": m,
                    }
                )

    raw_sources = sources if sources is not None else getattr(solver, "source", None)
    if raw_sources is None:
        raw_sources = []
    elif not isinstance(raw_sources, (list, tuple)):
        raw_sources = [raw_sources]
    for b in getattr(solver, "_beams", ()):
        if b not in raw_sources:
            raw_sources.append(b)

    parsed_sources = [
        _parse_source(src, nodes, bounds, h_idx, v_idx, n_idx) for src in raw_sources
    ]

    raw_monitors = monitors if monitors is not None else getattr(solver, "monitors", None)
    if raw_monitors is None:
        raw_monitors = []
    elif not isinstance(raw_monitors, (list, tuple)):
        raw_monitors = [raw_monitors]

    parsed_monitors = [
        _parse_monitor(mon, i, nodes, bounds, h_idx, v_idx, n_idx)
        for i, mon in enumerate(raw_monitors)
    ]
    if probes:
        for i, p in enumerate(probes):
            parsed_monitors.append(_parse_probe(p, i, bounds, h_idx, v_idx))

    return {
        "solver": solver,
        "grid": grid,
        "is_2d_mesh": is_2d_mesh,
        "plane": plane_str,
        "h_axis": h_axis_char,
        "v_axis": v_axis_char,
        "n_axis": n_axis_char,
        "h_idx": h_idx,
        "v_idx": v_idx,
        "n_idx": n_idx,
        "cut_coord": cut_val,
        "nodes": nodes,
        "bounds": bounds,
        "h_bounds": bounds[h_idx],
        "v_bounds": bounds[v_idx],
        "cell_counts": cell_counts,
        "bc_labels": bc_labels,
        "bc_tags": bc_tags,
        "bc_low_raw": bc_low_raw,
        "bc_high_raw": bc_high_raw,
        "n_pml": n_pml,
        "pml_thickness": pml_thickness,
        "bg_label": bg_desc["label"],
        "bg_name": bg_desc["name"],
        "bg_eps": bg_eps,
        "materials": materials_list,
        "eps_map_2d": eps_map_2d,
        "sources": parsed_sources,
        "monitors": parsed_monitors,
    }


def _text_halo(width: float = 2.6) -> list[Any]:
    """White stroke path-effect behind schematic labels for contrast."""
    return [pe.withStroke(linewidth=width, foreground=figstyle.PAPER), pe.Normal()]


def in_axes_texts(ax: Axes) -> list[Text]:
    """Return visible ``ax.texts`` placed inside the domain/axes box (excluding outer boundary/dimension tags)."""
    out: list[Text] = []
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    for t in ax.texts:
        if not t.get_visible() or not t.get_text().strip():
            continue
        role = getattr(t, "_schematic_role", None)
        if role is not None:
            if role == "in_axes":
                out.append(t)
            continue
        if t.get_transform() == ax.transData:
            tx, ty = t.get_position()
            if (x0 - 1e-9 <= float(tx) <= x1 + 1e-9) and (y0 - 1e-9 <= float(ty) <= y1 + 1e-9):
                out.append(t)
        elif t.get_transform() == ax.transAxes:
            tx, ty = t.get_position()
            if 0.0 <= float(tx) <= 1.0 and 0.0 <= float(ty) <= 1.0:
                out.append(t)
    return out


def _prune_out_of_bounds_ticks(ax: Axes) -> None:
    """Restrict major ticks on ``ax`` to its visible ``xlim`` and ``ylim`` so off-axis tick labels do not exist."""
    if not ax.axison or not getattr(ax, "_is_schematic_axes", False):
        return
    if ax.xaxis.get_visible() and len(ax.get_xticks()) > 0:
        x0, x1 = sorted(ax.get_xlim())
        tol_x = 1e-6 * max(1.0, abs(x1 - x0))
        valid_x = [float(x) for x in ax.get_xticks() if x0 - tol_x <= float(x) <= x1 + tol_x]
        ax.set_xticks(valid_x)
    if ax.yaxis.get_visible() and len(ax.get_yticks()) > 0:
        y0, y1 = sorted(ax.get_ylim())
        tol_y = 1e-6 * max(1.0, abs(y1 - y0))
        valid_y = [float(y) for y in ax.get_yticks() if y0 - tol_y <= float(y) <= y1 + tol_y]
        ax.set_yticks(valid_y)


def _pad_bbox(bb: Bbox, pad_px: float = 1.5) -> Bbox:
    """Expand a pixel bounding box by ``pad_px`` on all sides."""
    return Bbox.from_extents(bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px)


def _resolve_text_overlaps(fig: Figure) -> None:
    """Ensure every text artist on ``fig`` is >= ``SCHEMATIC_MIN_PT`` and no two text bounding boxes intersect."""
    for ax in fig.axes:
        _prune_out_of_bounds_ticks(ax)
    for t in fig.findobj(Text):
        if float(t.get_fontsize()) < SCHEMATIC_MIN_PT:
            t.set_fontsize(SCHEMATIC_MIN_PT)

    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return

    for ax in fig.axes:
        if not ax.texts:
            continue
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        dh = max(x1 - x0, 1e-12)
        dv = max(y1 - y0, 1e-12)

        # Fixed structural texts on this axes (title, axis labels, tick labels)
        fixed_bboxes: list[Bbox] = []
        for st in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if st is not None and st.get_visible() and st.get_text().strip():
                fixed_bboxes.append(_pad_bbox(st.get_window_extent(renderer), 1.5))
        for tl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if tl.get_visible() and tl.get_text().strip():
                fixed_bboxes.append(_pad_bbox(tl.get_window_extent(renderer), 1.0))

        placed_bboxes: list[Bbox] = list(fixed_bboxes)
        for t in ax.texts:
            if not t.get_visible() or not t.get_text().strip():
                continue
            bb = t.get_window_extent(renderer)
            if not any(bb.overlaps(other) for other in placed_bboxes):
                placed_bboxes.append(_pad_bbox(bb, 1.5))
                continue

            role = getattr(t, "_schematic_role", "in_axes")
            orig_x, orig_y = t.get_position()
            found = False
            if role == "in_axes" and t.get_transform() == ax.transData:
                y_fracs = (0.78, 0.56, 0.34, 0.18, 0.86, 0.44, 0.66, 0.24)
                x_shifts = (0.0, 0.06, -0.06, 0.12, -0.12, 0.18, -0.18, 0.24, -0.24)
                for dx_f in x_shifts:
                    for yf in y_fracs:
                        cand_x = float(np.clip(orig_x + dx_f * dh, x0 + 0.05 * dh, x1 - 0.05 * dh))
                        cand_y = float(y0 + yf * dv)
                        t.set_position((cand_x, cand_y))
                        cand_bb = t.get_window_extent(renderer)
                        if not any(cand_bb.overlaps(other) for other in placed_bboxes):
                            placed_bboxes.append(_pad_bbox(cand_bb, 1.5))
                            found = True
                            break
                    if found:
                        break
            elif t.get_transform() == ax.transData:
                steps_y = (1, 2, 3, 4) if (role == "dimension" and orig_y >= y1) else (0, 1, -1, 2, -2, 3, -3, 4, -4)
                for dx_f in (0.0, 0.08, -0.08, 0.16, -0.16):
                    for step in steps_y:
                        if dx_f == 0.0 and step == 0:
                            continue
                        dy = step * 0.06 * dv
                        t.set_position((orig_x + dx_f * dh, orig_y + dy))
                        cand_bb = t.get_window_extent(renderer)
                        if not any(cand_bb.overlaps(other) for other in placed_bboxes):
                            placed_bboxes.append(_pad_bbox(cand_bb, 1.5))
                            found = True
                            break
                    if found:
                        break
            if not found:
                placed_bboxes.append(_pad_bbox(t.get_window_extent(renderer), 1.0))

        # Ensure legend below ax does not overlap ax, x-axis label, or bottom boundary tag
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            ax_bb = ax.get_window_extent(renderer)
            lowest_y0 = ax_bb.y0
            if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text().strip():
                lowest_y0 = min(lowest_y0, ax.xaxis.label.get_window_extent(renderer).y0)
            for tl in ax.get_xticklabels():
                if tl.get_visible() and tl.get_text().strip():
                    lowest_y0 = min(lowest_y0, tl.get_window_extent(renderer).y0)
            for t in ax.texts:
                if t.get_visible() and getattr(t, "_schematic_role", "") == "boundary_bottom":
                    lowest_y0 = min(lowest_y0, t.get_window_extent(renderer).y0)
            leg_bb = leg.get_window_extent(renderer)
            if leg_bb.y1 > lowest_y0 - 6.0:
                ax_h_px = max(ax_bb.height, 1.0)
                y_anchor = (lowest_y0 - 10.0 - ax_bb.y0) / ax_h_px
                leg.set_bbox_to_anchor((0.5, float(min(y_anchor, -0.22))), transform=ax.transAxes)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()


def _discretization_metrics(
    info: dict[str, Any],
    wavelength: float | None = None,
    freq: float | None = None,
    min_feature: float | None = None,
) -> dict[str, Any]:
    """Compute min cell size, highest refractive index, cpw in n_max, and cells across smallest feature."""
    grid = info["grid"]
    solver = info["solver"]
    h_idx, v_idx = info["h_idx"], info["v_idx"]
    h_nodes = info["nodes"][h_idx]
    v_nodes = info["nodes"][v_idx]
    nv_cells = info["cell_counts"][v_idx]

    if hasattr(grid, "cells") and isinstance(grid.cells, dict) and "x0" in grid.cells:
        c = grid.cells
        dx_m = np.asarray(c["x1"], dtype=float) - np.asarray(c["x0"], dtype=float)
        dy_m = np.asarray(c["y1"], dtype=float) - np.asarray(c["y0"], dtype=float)
        min_cell_m = float(min(np.min(dx_m), np.min(dy_m)))
        max_cell_m = float(max(np.max(dx_m), np.max(dy_m)))
    elif (hasattr(grid, "nodes") or hasattr(grid, "vertices")) and (
        hasattr(grid, "edges") or hasattr(grid, "tri_vertices")
    ):
        verts = np.asarray(getattr(grid, "nodes", getattr(grid, "vertices", None)), dtype=float)
        if hasattr(grid, "edges"):
            edges = np.asarray(grid.edges, dtype=int)
            edge_lens = np.linalg.norm(verts[edges[:, 1]] - verts[edges[:, 0]], axis=1)
            min_cell_m = float(np.min(edge_lens)) if len(edge_lens) else 1.0
            max_cell_m = float(np.max(edge_lens)) if len(edge_lens) else 1.0
        else:
            min_cell_m = (
                float(np.min(grid.dual_edge_Len)) if hasattr(grid, "dual_edge_Len") else 1.0
            )
            max_cell_m = min_cell_m
    else:
        dh_min = float(np.min(np.diff(h_nodes))) if len(h_nodes) > 1 else 1.0
        dv_min = float(np.min(np.diff(v_nodes))) if len(v_nodes) > 1 else 1.0
        min_cell_m = min(dh_min, dv_min) if nv_cells > 2 else dh_min
        max_cell_m = max(dh_min, dv_min) if nv_cells > 2 else dh_min

    if wavelength is None:
        if freq is None and info["sources"]:
            for s in info["sources"]:
                if s.get("pulse") and s["pulse"].get("f0"):
                    freq = float(s["pulse"]["f0"])
                    break
        if freq is None and info["monitors"]:
            for m in info["monitors"]:
                m_freqs = m.get("frequencies")
                if m_freqs is not None and len(m_freqs) > 0:
                    freq = float(np.mean(m_freqs))
                    break
        if freq is None and solver is not None and getattr(solver, "fmax", None):
            freq = float(solver.fmax)
        if freq is not None and freq > 0.0:
            wavelength = c_light / float(freq)

    n_max = float(np.sqrt(max(1.0, info["bg_eps"])))
    for m in info["materials"]:
        eps_val = m.get("eps_r")
        mu_val = m.get("mu_r", 1.0)
        if eps_val is not None and np.isfinite(eps_val) and eps_val > 0.0:
            n_max = max(n_max, float(np.sqrt(float(eps_val) * float(mu_val))))

    lam0_m = float(wavelength) if wavelength is not None else None
    lam_med_m = (lam0_m / n_max) if lam0_m is not None else None
    cpw_nmax = (
        (lam_med_m / min_cell_m)
        if (lam_med_m is not None and np.isfinite(min_cell_m) and min_cell_m > 0.0)
        else None
    )
    cpw_bg = (
        (lam0_m / min_cell_m)
        if (lam0_m is not None and np.isfinite(min_cell_m) and min_cell_m > 0.0)
        else None
    )

    feat_m = float(min_feature) if min_feature is not None else None
    feat_name = "feature"
    if feat_m is None:
        candidates: list[tuple[float, str]] = []
        h_span_m = info["h_bounds"][1] - info["h_bounds"][0]
        v_span_m = info["v_bounds"][1] - info["v_bounds"][0]
        for m in info["materials"]:
            if m["kind"] == "circle":
                candidates.append((2.0 * float(m["radius"]), m["name"]))
            else:
                dh_m = abs(m["h_bounds"][1] - m["h_bounds"][0])
                dv_m = abs(m["v_bounds"][1] - m["v_bounds"][0])
                if 0.0 < dh_m < 0.98 * h_span_m:
                    candidates.append((dh_m, m["name"]))
                if nv_cells > 2 and 0.0 < dv_m < 0.98 * v_span_m:
                    candidates.append((dv_m, m["name"]))
        if candidates:
            feat_m, feat_name = min(candidates, key=lambda item: item[0])
        elif lam_med_m is not None:
            feat_m = lam_med_m
            feat_name = "λ_min"
        else:
            feat_m = info["h_bounds"][1] - info["h_bounds"][0]
            feat_name = f"L_{info['h_axis']}"

    cells_across_feat = (
        (feat_m / min_cell_m)
        if (feat_m is not None and np.isfinite(min_cell_m) and min_cell_m > 0.0)
        else None
    )
    return {
        "min_cell_m": min_cell_m,
        "max_cell_m": max_cell_m,
        "n_max": n_max,
        "wavelength_m": lam0_m,
        "wavelength_med_m": lam_med_m,
        "cpw_nmax": cpw_nmax,
        "cpw_bg": cpw_bg,
        "min_feature_m": feat_m,
        "min_feature_name": feat_name,
        "cells_across_feature": cells_across_feat,
    }


def draw_mesh(
    ax: Axes,
    solver_or_grid: Any,
    plane: str = "auto",
    unit: str = "m",
    every: int | None = None,
    wavelength: float | None = None,
    freq: float | None = None,
    zoom_inset: bool = False,
    zoom_bounds: tuple[float, float, float, float] | None = None,
    color: str = "#6B7280",
    alpha: float = 0.42,
    linewidth: float = 0.5,
    show_cpw: bool = True,
) -> dict[str, Any]:
    """Overlay the computational mesh and optional resolution badge on ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (already scaled in ``unit``).
    solver_or_grid : solver, grid, or mesh object
        The Wakis solver, grid, or mesh to draw.
    plane : {'auto', 'xy', 'xz', 'yz', 'zx', 'zy'}, optional
        2D cut plane.
    unit : {'auto', 'nm', 'µm', 'um', 'mm', 'cm', 'm'}, optional
        Length unit of ``ax``. Default is ``'m'``.
    every : int or None, optional
        Stride for Cartesian grid lines.
    wavelength : float or None, optional
        Reference free-space wavelength :math:`\\lambda_0` [m].
    freq : float or None, optional
        Reference frequency :math:`f_0` [Hz].
    zoom_inset : bool, optional
        If ``True``, add a 1× full-resolution zoom inset around the smallest
        feature or material boundary.
    zoom_bounds : tuple of 4 floats ``(h0, h1, v0, v1)`` [m], optional
        Region in meters [m] for the ``zoom_inset``.
    color : str, optional
        Grid line color.
    alpha : float, optional
        Grid line opacity.
    linewidth : float, optional
        Grid line width [pt].
    show_cpw : bool, optional
        Whether to draw a compact ``cells/λ`` badge on ``ax``.

    Returns
    -------
    meta : dict
        Dictionary with ``'stride_h'``, ``'stride_v'``, ``'min_cell_m'``,
        ``'cells_per_wavelength'``, and ``'cells_across_feature'``.
    """
    info = extract_setup_info(solver_or_grid, plane=plane)
    grid = info["grid"]
    h_idx, v_idx = info["h_idx"], info["v_idx"]
    h_span = info["h_bounds"][1] - info["h_bounds"][0]
    v_span = info["v_bounds"][1] - info["v_bounds"][0]
    _, scale = choose_length_unit(max(h_span, v_span), unit=unit)

    disc = _discretization_metrics(info, wavelength=wavelength, freq=freq)
    min_cell_m = disc["min_cell_m"]
    stride_h, stride_v = 1, 1

    if hasattr(grid, "cells") and isinstance(grid.cells, dict) and "x0" in grid.cells:
        c = grid.cells
        x0s = np.asarray(c["x0"], dtype=float) * scale
        x1s = np.asarray(c["x1"], dtype=float) * scale
        y0s = np.asarray(c["y0"], dtype=float) * scale
        y1s = np.asarray(c["y1"], dtype=float) * scale
        segments = [
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            for x0, x1, y0, y1 in zip(x0s, x1s, y0s, y1s)
        ]
        lc = LineCollection(segments, colors=color, linewidths=linewidth, alpha=alpha, zorder=2.5)
        ax.add_collection(lc)
    elif (hasattr(grid, "nodes") or hasattr(grid, "vertices")) and (
        hasattr(grid, "edges") or hasattr(grid, "tri_vertices")
    ):
        verts = np.asarray(getattr(grid, "nodes", getattr(grid, "vertices", None)), dtype=float)
        if hasattr(grid, "edges"):
            edges = np.asarray(grid.edges, dtype=int)
            pts_2d = verts[:, [h_idx, v_idx]] if verts.shape[1] >= 3 else verts[:, :2]
            segs = pts_2d[edges] * scale
            lc = LineCollection(segs, colors=color, linewidths=linewidth, alpha=alpha, zorder=2.5)
            ax.add_collection(lc)
        elif hasattr(grid, "tri_vertices"):
            tv = np.asarray(grid.tri_vertices, dtype=float) * scale
            tris = np.asarray(grid.triangles, dtype=int)
            segs = [[tv[t[0]], tv[t[1]], tv[t[2]], tv[t[0]]] for t in tris]
            lc = LineCollection(segs, colors=color, linewidths=linewidth, alpha=alpha, zorder=2.5)
            ax.add_collection(lc)
    else:
        h_nodes = info["nodes"][h_idx]
        v_nodes = info["nodes"][v_idx]
        nh = max(1, len(h_nodes) - 1)
        nv = max(1, len(v_nodes) - 1)
        stride_h = int(every) if every is not None else max(1, int(np.ceil(nh / 30)))
        stride_v = int(every) if every is not None else max(1, int(np.ceil(nv / 18)))

        h_scaled = h_nodes * scale
        v_scaled = v_nodes * scale
        v0, v1 = v_scaled[0], v_scaled[-1]
        h0, h1 = h_scaled[0], h_scaled[-1]

        segs = [[(float(x), float(v0)), (float(x), float(v1))] for x in h_scaled[::stride_h]] + [
            [(float(h0), float(y)), (float(h1), float(y))] for y in v_scaled[::stride_v]
        ]
        lc = LineCollection(
            segs, colors=color, linewidths=linewidth, alpha=alpha, zorder=2.5, label="_mesh_lines"
        )
        ax.add_collection(lc)

        dh_min = float(np.min(np.diff(h_nodes))) if len(h_nodes) > 1 else 1.0
        dv_min = float(np.min(np.diff(v_nodes))) if len(v_nodes) > 1 else 1.0

        if zoom_inset and (stride_h > 1 or stride_v > 1):
            if zoom_bounds is not None:
                zh0, zh1, zv0, zv1 = zoom_bounds
            else:
                if info["materials"]:
                    m0 = info["materials"][0]
                    if m0["kind"] == "circle":
                        z_center_h = m0["center"][0] + m0["radius"]
                        z_center_v = m0["center"][1]
                    else:
                        z_center_h = m0["h_bounds"][0]
                        z_center_v = 0.5 * sum(info["v_bounds"])
                else:
                    z_center_h = 0.5 * sum(info["h_bounds"])
                    z_center_v = 0.5 * sum(info["v_bounds"])
                half_h = min(4.0 * dh_min, 0.12 * h_span)
                half_v = min(4.0 * dv_min, 0.25 * v_span) if nv > 2 else 0.35 * v_span
                zh0, zh1 = z_center_h - half_h, z_center_h + half_h
                zv0, zv1 = z_center_v - half_v, z_center_v + half_v

            ins = ax.inset_axes([0.68, 0.52, 0.29, 0.42])
            ins._is_inset = True
            ins.set_in_layout(False)
            ins.set_facecolor(figstyle.PAPER)
            for sp in ("top", "right", "left", "bottom"):
                ins.spines[sp].set_visible(True)
                ins.spines[sp].set_color("#334155")
                ins.spines[sp].set_linewidth(0.85)
            for mat in info["materials"]:
                if mat["kind"] == "rect":
                    mh0, mh1 = mat["h_bounds"][0] * scale, mat["h_bounds"][1] * scale
                    mv0, mv1 = mat["v_bounds"][0] * scale, mat["v_bounds"][1] * scale
                    ins.add_patch(
                        Rectangle(
                            (mh0, mv0),
                            mh1 - mh0,
                            mv1 - mv0,
                            facecolor=mat["color"],
                            edgecolor="#1E293B",
                            lw=0.9,
                            alpha=0.75,
                        )
                    )
                elif mat["kind"] == "circle":
                    cx, cy = mat["center"][0] * scale, mat["center"][1] * scale
                    rad = mat["radius"] * scale
                    ins.add_patch(
                        Circle(
                            (cx, cy),
                            rad,
                            facecolor=mat["color"],
                            edgecolor="#1E293B",
                            lw=0.9,
                            alpha=0.75,
                        )
                    )
                elif mat["kind"] == "polygon":
                    verts = np.asarray(mat["vertices"], dtype=float) * scale
                    ins.add_patch(
                        Polygon(
                            verts,
                            closed=True,
                            facecolor=mat["color"],
                            edgecolor="#1E293B",
                            lw=0.9,
                            alpha=0.75,
                        )
                    )
            h_sub = h_nodes[(h_nodes >= zh0 - dh_min) & (h_nodes <= zh1 + dh_min)] * scale
            v_sub = v_nodes[(v_nodes >= zv0 - dv_min) & (v_nodes <= zv1 + dv_min)] * scale
            for x in h_sub:
                ins.axvline(x, color="#475569", lw=0.65, alpha=0.75)
            for y in v_sub:
                ins.axhline(y, color="#475569", lw=0.65, alpha=0.75)
            ins.set_xlim(zh0 * scale, zh1 * scale)
            ins.set_ylim(zv0 * scale, zv1 * scale)
            ins.set_xticks([])
            ins.set_yticks([])
            ins_t = ins.text(
                0.5,
                0.88,
                f"1× zoom (Δ{info['h_axis']} = {_format_length(dh_min)})",
                transform=ins.transAxes,
                ha="center",
                va="top",
                fontsize=7.6,
                fontweight="bold",
                color=figstyle.INK,
                path_effects=_text_halo(2.4),
            )
            ins_t._schematic_role = "inset"

    cpw = disc["cpw_nmax"]
    if show_cpw and cpw is not None:
        stride_txt = f", 1/{stride_h} shown" if stride_h > 1 else ""
        badge = f"Δ{info['h_axis']} = {_format_length(min_cell_m)} ({cpw:.1f} cells/λ{stride_txt})"
        t_cpw = ax.text(
            1.02,
            0.0,
            badge,
            transform=ax.transAxes,
            fontsize=9.2,
            color=figstyle.INK,
            va="bottom",
            ha="left",
            clip_on=False,
            zorder=9,
        )
        t_cpw._schematic_role = "mesh_badge"

    return {
        "stride_h": stride_h,
        "stride_v": stride_v,
        "min_cell_m": min_cell_m,
        "cells_per_wavelength": cpw,
        "cells_across_feature": disc["cells_across_feature"],
        "n_max": disc["n_max"],
    }


def overlay_geometry(
    ax: Axes,
    solver_or_grid: Any,
    plane: str = "auto",
    unit: str = "m",
    orientation: str = "2d",
    structures: Sequence[Any] | None = None,
    material_names: Mapping[str, str] | None = None,
    sources: Any = None,
    monitors: Any = None,
    probes: Sequence[Any] | None = None,
    show_boundaries: bool = True,
    show_materials: bool = True,
    show_sources: bool = False,
    show_monitors: bool = False,
    fill_materials: bool = False,
    material_alpha: float = 0.14,
    outline_color: str = "#1a1a1a",
    label_materials: bool = True,
) -> dict[str, Any]:
    """Overlay material outlines, short labels, and boundary/CPML regions onto ``ax``."""
    info = extract_setup_info(
        solver_or_grid,
        plane=plane,
        structures=structures,
        material_names=material_names,
        sources=sources,
        monitors=monitors,
        probes=probes,
    )
    h_span = info["h_bounds"][1] - info["h_bounds"][0]
    v_span = info["v_bounds"][1] - info["v_bounds"][0]
    _, scale = choose_length_unit(max(h_span, v_span), unit=unit)

    h0, h1 = info["h_bounds"][0] * scale, info["h_bounds"][1] * scale
    v0, v1 = info["v_bounds"][0] * scale, info["v_bounds"][1] * scale
    h_ax = info["h_axis"]
    v_ax = info["v_axis"]
    is_2d_spatial = orientation == "2d"

    if show_materials:
        for idx, mat in enumerate(info["materials"]):
            mh0, mh1 = mat["h_bounds"][0] * scale, mat["h_bounds"][1] * scale
            short_mat_lbl = mat["name"]
            if is_2d_spatial:
                if mat["kind"] == "circle":
                    cx, cy = mat["center"][0] * scale, mat["center"][1] * scale
                    rad = mat["radius"] * scale
                    patch = Circle(
                        (cx, cy),
                        rad,
                        facecolor=mat["color"] if fill_materials else "none",
                        edgecolor=outline_color,
                        lw=1.3,
                        ls="-",
                        alpha=material_alpha if fill_materials else 0.9,
                        zorder=6,
                    )
                    ax.add_patch(patch)
                    lx, ly = cx, cy
                elif mat["kind"] == "polygon":
                    verts = np.asarray(mat["vertices"], dtype=float) * scale
                    patch = Polygon(
                        verts,
                        closed=True,
                        facecolor=mat["color"] if fill_materials else "none",
                        edgecolor=outline_color,
                        lw=1.3,
                        alpha=material_alpha if fill_materials else 0.9,
                        zorder=6,
                    )
                    ax.add_patch(patch)
                    lx, ly = float(verts[:, 0].mean()), float(verts[:, 1].mean())
                else:
                    mv0, mv1 = mat["v_bounds"][0] * scale, mat["v_bounds"][1] * scale
                    if fill_materials:
                        ax.add_patch(
                            Rectangle(
                                (mh0, mv0),
                                mh1 - mh0,
                                mv1 - mv0,
                                facecolor=mat["color"],
                                edgecolor="none",
                                alpha=material_alpha,
                                zorder=5,
                            )
                        )
                    ax.add_patch(
                        Rectangle(
                            (mh0, mv0),
                            mh1 - mh0,
                            mv1 - mv0,
                            facecolor="none",
                            edgecolor=outline_color,
                            lw=1.3,
                            ls="-",
                            zorder=6,
                        )
                    )
                    lx, ly = 0.5 * (mh0 + mh1), mv0 + 0.82 * (mv1 - mv0)
                if label_materials:
                    ax.text(
                        lx,
                        ly,
                        short_mat_lbl,
                        ha="center",
                        va="center",
                        fontsize=9.2,
                        fontweight="bold",
                        color=figstyle.INK,
                        path_effects=_text_halo(2.6),
                        zorder=8,
                    )
            else:
                if fill_materials:
                    ax.axvspan(
                        mh0,
                        mh1,
                        color=mat["color"],
                        alpha=max(material_alpha, 0.22),
                        zorder=1,
                    )
                ax.axvline(mh0, color=outline_color, lw=1.1, ls="-", alpha=0.8, zorder=5)
                ax.axvline(mh1, color=outline_color, lw=1.1, ls="-", alpha=0.8, zorder=5)
                if label_materials:
                    trans = ax.get_xaxis_transform()
                    ax.text(
                        0.5 * (mh0 + mh1),
                        0.84 - 0.08 * (idx % 2),
                        short_mat_lbl,
                        transform=trans,
                        ha="center",
                        va="top",
                        fontsize=9.2,
                        fontweight="bold",
                        color=figstyle.INK,
                        path_effects=_text_halo(2.5),
                        zorder=8,
                    )

    if show_boundaries:
        th_h_low = info["pml_thickness"][f"{h_ax}_low"] * scale
        th_h_high = info["pml_thickness"][f"{h_ax}_high"] * scale
        tag_h_low = info["bc_tags"][f"{h_ax}_low"]
        tag_h_high = info["bc_tags"][f"{h_ax}_high"]

        if is_2d_spatial:
            th_v_low = info["pml_thickness"][f"{v_ax}_low"] * scale
            th_v_high = info["pml_thickness"][f"{v_ax}_high"] * scale
            if th_h_low > 0.0:
                ax.add_patch(
                    Rectangle(
                        (h0, v0),
                        th_h_low,
                        v1 - v0,
                        facecolor="#CBD5E1",
                        edgecolor="#64748B",
                        hatch="///",
                        alpha=0.28,
                        lw=0.8,
                        zorder=4,
                    )
                )
            if th_h_high > 0.0:
                ax.add_patch(
                    Rectangle(
                        (h1 - th_h_high, v0),
                        th_h_high,
                        v1 - v0,
                        facecolor="#CBD5E1",
                        edgecolor="#64748B",
                        hatch="///",
                        alpha=0.28,
                        lw=0.8,
                        zorder=4,
                    )
                )
            if th_v_low > 0.0:
                ax.add_patch(
                    Rectangle(
                        (h0, v0),
                        h1 - h0,
                        th_v_low,
                        facecolor="#CBD5E1",
                        edgecolor="#64748B",
                        hatch="\\\\\\",
                        alpha=0.28,
                        lw=0.8,
                        zorder=4,
                    )
                )
            if th_v_high > 0.0:
                ax.add_patch(
                    Rectangle(
                        (h0, v1 - th_v_high),
                        h1 - h0,
                        th_v_high,
                        facecolor="#CBD5E1",
                        edgecolor="#64748B",
                        hatch="\\\\\\",
                        alpha=0.28,
                        lw=0.8,
                        zorder=4,
                    )
                )
            ax.text(
                0.015,
                0.96,
                f"−{h_ax}: {tag_h_low}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.2,
                color=figstyle.INK,
                path_effects=_text_halo(2.5),
                zorder=8,
            )
            ax.text(
                0.985,
                0.96,
                f"+{h_ax}: {tag_h_high}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9.2,
                color=figstyle.INK,
                path_effects=_text_halo(2.5),
                zorder=8,
            )
        else:
            if th_h_low > 0.0:
                ax.axvspan(
                    h0,
                    h0 + th_h_low,
                    facecolor="#CBD5E1",
                    edgecolor="#64748B",
                    hatch="///",
                    alpha=0.25,
                    zorder=2,
                )
            if th_h_high > 0.0:
                ax.axvspan(
                    h1 - th_h_high,
                    h1,
                    facecolor="#CBD5E1",
                    edgecolor="#64748B",
                    hatch="///",
                    alpha=0.25,
                    zorder=2,
                )
            trans = ax.get_xaxis_transform()
            ax.text(
                h0 + 0.015 * (h1 - h0),
                0.96,
                tag_h_low,
                transform=trans,
                ha="left",
                va="top",
                fontsize=9.2,
                color="#334155",
                path_effects=_text_halo(2.5),
                zorder=8,
            )
            ax.text(
                h1 - 0.015 * (h1 - h0),
                0.96,
                tag_h_high,
                transform=trans,
                ha="right",
                va="top",
                fontsize=9.2,
                color="#334155",
                path_effects=_text_halo(2.5),
                zorder=8,
            )

    if show_sources:
        for src in info["sources"]:
            sh0, sh1 = src["h_bounds"][0] * scale, src["h_bounds"][1] * scale
            if abs(sh1 - sh0) < 1e-9 * scale:
                ax.axvline(sh0, color="#D55E00", lw=1.2, ls="--", alpha=0.85, zorder=6)
                trans = ax.get_xaxis_transform()
                ax.text(
                    sh0,
                    0.06,
                    " Source",
                    transform=trans,
                    ha="left",
                    va="bottom",
                    fontsize=9.2,
                    color="#B04A00",
                    path_effects=_text_halo(2.5),
                    zorder=8,
                )
    if show_monitors:
        for mon in info["monitors"]:
            mh0, mh1 = mon["h_bounds"][0] * scale, mon["h_bounds"][1] * scale
            if abs(mh1 - mh0) < 1e-9 * scale:
                ax.axvline(mh0, color="#009E73", lw=1.1, ls=":", alpha=0.85, zorder=6)

    return info


def _draw_boundary_edge(
    ax: Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    tag: str,
) -> None:
    """Draw a domain face edge graphically according to its boundary tag."""
    x0, y0 = p0
    x1, y1 = p1
    low_tag = tag.lower()
    if low_tag == "pec":
        ax.plot([x0, x1], [y0, y1], color=figstyle.INK, lw=3.2, ls="-", solid_capstyle="butt", zorder=7)
    elif low_tag == "pmc":
        ax.plot([x0, x1], [y0, y1], color="#334155", lw=2.4, ls="-.", solid_capstyle="butt", zorder=7)
    elif low_tag in ("periodic", "bloch"):
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#334155",
            lw=1.6,
            ls=(0, (4.0, 2.5)),
            zorder=7,
        )
        # Matching periodicity arrow along the edge midpoint
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        dx, dy = x1 - x0, y1 - y0
        ax.annotate(
            "",
            xy=(mx + 0.04 * dx, my + 0.04 * dy),
            xytext=(mx - 0.04 * dx, my - 0.04 * dy),
            arrowprops={"arrowstyle": "-|>", "color": "#334155", "lw": 1.2, "mutation_scale": 9},
            zorder=7.5,
        )
    elif "abc" in low_tag or "mur" in low_tag:
        ax.plot([x0, x1], [y0, y1], color="#475569", lw=2.0, ls=":", zorder=7)
    else:
        ax.plot([x0, x1], [y0, y1], color=figstyle.INK, lw=1.6, ls="-", zorder=7)


def _draw_schematic_panel(
    ax: Axes,
    info: dict[str, Any],
    unit: str = "auto",
    show_mesh: bool = False,
    mesh_every: int | None = None,
    mesh_zoom: bool = False,
    show_pulse_inset: bool = False,
    show_wave_packet: bool = False,
    wave_packet_center: float | None = None,
    wavelength: float | None = None,
    freq: float | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Render Figure A (clean geometry, boundaries, sources & monitors) onto ``ax``."""
    h_ax, v_ax = info["h_axis"], info["v_axis"]
    h_span_m = info["h_bounds"][1] - info["h_bounds"][0]
    v_span_m = info["v_bounds"][1] - info["v_bounds"][0]
    unit_label, scale = choose_length_unit(max(h_span_m, v_span_m), unit=unit)

    h0, h1 = info["h_bounds"][0] * scale, info["h_bounds"][1] * scale
    v0, v1 = info["v_bounds"][0] * scale, info["v_bounds"][1] * scale
    dh = max(h1 - h0, 1e-12)
    dv = max(v1 - v0, 1e-12)

    nv_cells = info["cell_counts"][info["v_idx"]]
    is_quasi_1d = nv_cells <= 2 and v_span_m < 0.15 * h_span_m

    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    # 1. Background domain fill
    ax.add_patch(
        Rectangle(
            (h0, v0),
            dh,
            dv,
            facecolor="#F8FAFC",
            edgecolor="none",
            zorder=1,
        )
    )
    legend_handles.append(
        Rectangle((0, 0), 1, 1, facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.9)
    )
    legend_labels.append(info["bg_label"])

    # 2. Material regions (filled regions to scale; details in legend below axes)
    seen_mat_labels: set[str] = set()
    for mat in info["materials"]:
        mh0, mh1 = mat["h_bounds"][0] * scale, mat["h_bounds"][1] * scale
        if mat["kind"] == "circle":
            cx, cy = mat["center"][0] * scale, mat["center"][1] * scale
            rad = mat["radius"] * scale
            patch = Circle(
                (cx, cy),
                rad,
                facecolor=mat["color"],
                edgecolor="#1E293B",
                lw=1.3,
                alpha=0.88,
                zorder=3,
            )
            ax.add_patch(patch)
            if "r =" in mat["label"].lower():
                mat_leg = mat["label"]
            else:
                mat_leg = f"{mat['label']} (r = {_format_length(mat['radius'])})"
        elif mat["kind"] == "polygon":
            verts = np.asarray(mat["vertices"], dtype=float) * scale
            patch = Polygon(
                verts,
                closed=True,
                facecolor=mat["color"],
                edgecolor="#1E293B",
                lw=1.3,
                alpha=0.88,
                zorder=3,
            )
            ax.add_patch(patch)
            mat_leg = mat["label"]
        else:
            mv0, mv1 = mat["v_bounds"][0] * scale, mat["v_bounds"][1] * scale
            patch = Rectangle(
                (mh0, mv0),
                mh1 - mh0,
                mv1 - mv0,
                facecolor=mat["color"],
                edgecolor="#1E293B",
                lw=1.3,
                alpha=0.88,
                zorder=3,
            )
            ax.add_patch(patch)
            thick_m = (
                abs(mat["h_bounds"][1] - mat["h_bounds"][0])
                if is_quasi_1d
                else min(
                    abs(mat["h_bounds"][1] - mat["h_bounds"][0]),
                    abs(mat["v_bounds"][1] - mat["v_bounds"][0]),
                )
            )
            if "(w =" in mat["label"] or "(d =" in mat["label"]:
                mat_leg = mat["label"]
            else:
                mat_leg = f"{mat['label']} (d = {_format_length(thick_m)})"

        if mat_leg not in seen_mat_labels:
            seen_mat_labels.add(mat_leg)
            legend_handles.append(
                Rectangle((0, 0), 1, 1, facecolor=mat["color"], edgecolor="#1E293B", lw=1.0)
            )
            legend_labels.append(mat_leg)

    # 3. Optional mesh overlay (kept for backward compatibility when show_mesh=True)
    if show_mesh:
        draw_mesh(
            ax,
            info["solver"] if info["solver"] is not None else info["grid"],
            plane=info["plane"],
            unit=unit_label,
            every=mesh_every,
            wavelength=wavelength,
            freq=freq,
            zoom_inset=mesh_zoom,
            show_cpw=True,
        )

    # 4. CPML hatched bands and graphical boundary edges
    th_h_low = info["pml_thickness"][f"{h_ax}_low"] * scale
    th_h_high = info["pml_thickness"][f"{h_ax}_high"] * scale
    th_v_low = info["pml_thickness"][f"{v_ax}_low"] * scale
    th_v_high = info["pml_thickness"][f"{v_ax}_high"] * scale
    has_pml = any(t > 0.0 for t in (th_h_low, th_h_high, th_v_low, th_v_high))

    if th_h_low > 0.0:
        ax.add_patch(
            Rectangle(
                (h0, v0),
                th_h_low,
                dv,
                facecolor="#CBD5E1",
                edgecolor="#475569",
                hatch="///",
                alpha=0.45,
                lw=0.9,
                zorder=4,
            )
        )
    if th_h_high > 0.0:
        ax.add_patch(
            Rectangle(
                (h1 - th_h_high, v0),
                th_h_high,
                dv,
                facecolor="#CBD5E1",
                edgecolor="#475569",
                hatch="///",
                alpha=0.45,
                lw=0.9,
                zorder=4,
            )
        )
    if th_v_low > 0.0:
        ax.add_patch(
            Rectangle(
                (h0, v0),
                dh,
                th_v_low,
                facecolor="#CBD5E1",
                edgecolor="#475569",
                hatch="\\\\\\",
                alpha=0.45,
                lw=0.9,
                zorder=4,
            )
        )
    if th_v_high > 0.0:
        ax.add_patch(
            Rectangle(
                (h0, v1 - th_v_high),
                dh,
                th_v_high,
                facecolor="#CBD5E1",
                edgecolor="#475569",
                hatch="\\\\\\",
                alpha=0.45,
                lw=0.9,
                zorder=4,
            )
        )
    if has_pml:
        pml_m = max(
            info["pml_thickness"][f"{h_ax}_low"],
            info["pml_thickness"][f"{h_ax}_high"],
            info["pml_thickness"][f"{v_ax}_low"],
            info["pml_thickness"][f"{v_ax}_high"],
        )
        legend_handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor="#CBD5E1",
                edgecolor="#475569",
                hatch="///",
                lw=0.9,
            )
        )
        legend_labels.append(
            f"CPML ({info['n_pml']} cells, {_format_length(pml_m)})"
            if info["n_pml"] > 0
            else "CPML absorbing layer"
        )

    tag_h_low = info["bc_tags"][f"{h_ax}_low"]
    tag_h_high = info["bc_tags"][f"{h_ax}_high"]
    tag_v_low = info["bc_tags"][f"{v_ax}_low"]
    tag_v_high = info["bc_tags"][f"{v_ax}_high"]

    _draw_boundary_edge(ax, (h0, v0), (h0, v1), tag_h_low)
    _draw_boundary_edge(ax, (h1, v0), (h1, v1), tag_h_high)
    _draw_boundary_edge(ax, (h0, v0), (h1, v0), tag_v_low)
    _draw_boundary_edge(ax, (h0, v1), (h1, v1), tag_v_high)

    # 5. ONE short boundary tag per face placed OUTSIDE the domain box
    t_left = ax.text(
        h0 - 0.018 * dh,
        v0 + 0.50 * dv,
        tag_h_low,
        ha="right",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1E293B",
        clip_on=False,
        zorder=9,
    )
    t_left._schematic_role = "boundary_left"

    t_right = ax.text(
        h1 + 0.018 * dh,
        v0 + (0.50 if is_quasi_1d else 0.25) * dv,
        tag_h_high,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#1E293B",
        clip_on=False,
        zorder=9,
    )
    t_right._schematic_role = "boundary_right"

    t_top = ax.text(
        h0 + 0.50 * dh,
        v1 + 0.04 * dv,
        tag_v_high,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#1E293B",
        clip_on=False,
        zorder=9,
    )
    t_top._schematic_role = "boundary_top"

    t_bot = ax.text(
        h0 + 0.50 * dh,
        v0 - 0.04 * dv,
        tag_v_low,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#1E293B",
        clip_on=False,
        zorder=9,
    )
    t_bot._schematic_role = "boundary_bottom"

    # 6. Dimension arrow per physical axis placed OUTSIDE the domain box
    y_arrow = v1 + 0.145 * dv
    ax.annotate(
        "",
        xy=(h1, y_arrow),
        xytext=(h0, y_arrow),
        arrowprops={"arrowstyle": "<->", "color": figstyle.INK, "lw": 1.15},
        annotation_clip=False,
        zorder=9,
    )
    t_dim_h = ax.text(
        0.5 * (h0 + h1),
        y_arrow + 0.02 * dv,
        _format_length(h_span_m, unit_label=unit_label),
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=figstyle.INK,
        clip_on=False,
        zorder=9,
    )
    t_dim_h._schematic_role = "dimension"

    if not is_quasi_1d:
        x_arrow = h1 + 0.16 * dh
        ax.annotate(
            "",
            xy=(x_arrow, v1),
            xytext=(x_arrow, v0),
            arrowprops={"arrowstyle": "<->", "color": figstyle.INK, "lw": 1.15},
            annotation_clip=False,
            zorder=9,
        )
        t_dim_v = ax.text(
            x_arrow + 0.02 * dh,
            0.72 * v1 + 0.28 * v0,
            _format_length(v_span_m, unit_label=unit_label),
            ha="left",
            va="center",
            rotation=90,
            fontsize=9.5,
            fontweight="bold",
            color=figstyle.INK,
            clip_on=False,
            zorder=9,
        )
        t_dim_v._schematic_role = "dimension"

    # 7. Sources & monitors as distinct symbols with short numbered tags (<= 6 inside axes)
    src_color = "#D55E00"
    mon_color = "#009E73"
    y_lanes = (0.76, 0.26, 0.52, 0.84, 0.38, 0.16)
    item_num = 0
    in_axes_count = 0
    max_in_axes = 6

    for s_idx, src in enumerate(info["sources"]):
        item_num += 1
        sh0, sh1 = src["h_bounds"][0] * scale, src["h_bounds"][1] * scale
        sv0, sv1 = src["v_bounds"][0] * scale, src["v_bounds"][1] * scale
        kind = src.get("kind", "sheet")
        short_w = src.get("short_word", "source")
        short_tag = f"{item_num} · {short_w}"
        leg_desc = f"{item_num} · {src.get('desc', src.get('label', 'source'))}"

        if kind == "tfsf":
            ax.add_patch(
                Rectangle(
                    (sh0, sv0),
                    sh1 - sh0,
                    sv1 - sv0,
                    facecolor="none",
                    edgecolor=src_color,
                    lw=1.7,
                    ls="--",
                    zorder=6,
                )
            )
            tx, ty = 0.5 * (sh0 + sh1), v0 + y_lanes[(item_num - 1) % len(y_lanes)] * dv
            leg_handle = plt.Line2D([0], [0], color=src_color, lw=1.8, ls="--")
        elif kind == "point":
            yv_pt = sv0 if not is_quasi_1d else v0 + 0.50 * dv
            ax.plot(
                sh0,
                yv_pt,
                marker="*",
                linestyle="None",
                markersize=10.5,
                color=src_color,
                markeredgecolor=figstyle.INK,
                markeredgewidth=0.6,
                zorder=8,
            )
            # Tag next to the marker itself, not in a fixed lane (which can fall in another layer).
            tx, ty = sh0 + 0.03 * dh, yv_pt + (0.08 if yv_pt < v0 + 0.85 * dv else -0.08) * dv
            leg_handle = plt.Line2D(
                [0],
                [0],
                color=src_color,
                marker="*",
                linestyle="None",
                markersize=9.0,
                markeredgecolor=figstyle.INK,
            )
        elif kind == "lumped":
            ax.plot([sh0, sh1], [sv0, sv1], color=src_color, lw=2.5, marker="none", zorder=7)
            ax.plot(
                0.5 * (sh0 + sh1),
                0.5 * (sv0 + sv1),
                marker="s",
                linestyle="None",
                markersize=6.5,
                color=src_color,
                zorder=8,
            )
            tx = 0.5 * (sh0 + sh1) + 0.025 * dh
            ty = v0 + y_lanes[(item_num - 1) % len(y_lanes)] * dv
            leg_handle = plt.Line2D([0], [0], color=src_color, lw=2.2, marker="s", markersize=6.0)
        elif kind == "beam":
            y_b = sv0 if abs(sv1 - sv0) < 1e-9 * scale else 0.5 * (v0 + v1)
            ax.annotate(
                "",
                xy=(h1 - max(th_h_high, 0.04 * dh), y_b),
                xytext=(h0 + max(th_h_low, 0.04 * dh), y_b),
                arrowprops={"arrowstyle": "-|>", "color": src_color, "lw": 1.8, "ls": "-."},
                zorder=7,
            )
            # Tag sits on the beam line itself (a fixed lane can land next to another horizontal object).
            tx, ty = h0 + 0.30 * dh, y_b + 0.07 * dv
            leg_handle = plt.Line2D([0], [0], color=src_color, lw=1.8, ls="-.")
        else:
            ax.plot([sh0, sh1], [sv0, sv1], color=src_color, lw=2.3, ls="-", marker="none", zorder=7)
            if s_idx == 0:
                k_vec = src.get("k_vec", (1.0, 0.0, 0.0))
                if abs(k_vec[0]) > 0.1 or abs(k_vec[1]) > 0.1:
                    ax.annotate(
                        "",
                        xy=(sh0 + 0.065 * dh * k_vec[0], v0 + 0.50 * dv + 0.12 * dv * k_vec[1]),
                        xytext=(sh0, v0 + 0.50 * dv),
                        arrowprops={"arrowstyle": "-|>", "color": src_color, "lw": 1.6},
                        zorder=8,
                    )
            tx = sh0 + 0.025 * dh
            ty = v0 + y_lanes[(item_num - 1) % len(y_lanes)] * dv
            leg_handle = plt.Line2D([0], [0], color=src_color, lw=2.3, ls="-")

        if in_axes_count < max_in_axes:
            t_src = ax.text(
                float(np.clip(tx, h0 + 0.06 * dh, h1 - 0.06 * dh)),
                ty,
                short_tag,
                color=src_color,
                fontsize=9.5,
                fontweight="bold",
                ha="center",
                va="center",
                path_effects=_text_halo(2.8),
                zorder=9,
            )
            t_src._schematic_role = "in_axes"
            in_axes_count += 1

        legend_handles.append(leg_handle)
        legend_labels.append(leg_desc)

    # Group excess monitors if total sources + monitors would exceed 6 in-axes tags
    rem_slots = max(0, max_in_axes - in_axes_count)
    n_mons = len(info["monitors"])
    for m_idx, mon in enumerate(info["monitors"]):
        item_num += 1
        mh0, mh1 = mon["h_bounds"][0] * scale, mon["h_bounds"][1] * scale
        mv0, mv1 = mon["v_bounds"][0] * scale, mon["v_bounds"][1] * scale
        kind = mon.get("kind", "plane")
        short_w = mon.get("short_word", "mon")
        short_tag = f"{item_num} · {short_w}"
        leg_desc = f"{item_num} · {mon.get('desc', mon.get('label', 'monitor'))}"

        if kind == "box":
            ax.add_patch(
                Rectangle(
                    (mh0, mv0),
                    mh1 - mh0,
                    mv1 - mv0,
                    facecolor="none",
                    edgecolor=mon_color,
                    lw=1.5,
                    ls=":",
                    zorder=6,
                )
            )
            tx = 0.5 * (mh0 + mh1)
            ty = v0 + y_lanes[(item_num - 1) % len(y_lanes)] * dv
            leg_handle = plt.Line2D([0], [0], color=mon_color, lw=1.6, ls=":")
        elif kind == "probe":
            yv = mv0 if not is_quasi_1d else v0 + 0.48 * dv
            ax.plot(
                mh0,
                yv,
                marker="o",
                linestyle="None",
                markersize=6.8,
                color=mon_color,
                markeredgecolor=figstyle.INK,
                markeredgewidth=0.7,
                zorder=8,
            )
            if is_quasi_1d:
                tx, ty = mh0, yv - 0.18 * dv
            else:
                tx = mh0 + 0.03 * dh
                ty = yv + (0.075 if yv < v0 + 0.85 * dv else -0.075) * dv
            leg_handle = plt.Line2D(
                [0],
                [0],
                color=mon_color,
                marker="o",
                linestyle="None",
                markersize=6.5,
                markeredgecolor=figstyle.INK,
            )
        else:
            yv_line = v0 + 0.64 * dv if (is_quasi_1d and abs(mh1 - mh0) > 1e-9 * scale) else None
            if yv_line is not None:
                ax.plot([mh0, mh1], [yv_line, yv_line], color=mon_color, lw=1.9, ls="--", zorder=7)
                tx = 0.5 * (mh0 + mh1)
                ty = v0 + y_lanes[(item_num - 1) % len(y_lanes)] * dv
            elif abs(mh1 - mh0) > 1e-9 * scale and abs(mv1 - mv0) > 1e-9 * scale:
                # 2D area monitor in the cut plane: draw its bounding rectangle
                ax.add_patch(
                    Rectangle(
                        (mh0, mv0),
                        mh1 - mh0,
                        mv1 - mv0,
                        facecolor="none",
                        edgecolor=mon_color,
                        lw=1.7,
                        ls="--",
                        zorder=6,
                    )
                )
                tx = mh0 + 0.22 * (mh1 - mh0)
                ty = mv1 - 0.06 * dv
            else:
                ax.plot([mh0, mh1], [mv0, mv1], color=mon_color, lw=1.9, ls="--", zorder=7)
                tx = mh0
                ty = v0 + y_lanes[(item_num - 1) % len(y_lanes)] * dv
            if yv_line is None and abs(mh1 - mh0) > 1e-9 * scale and abs(mv1 - mv0) < 1e-9 * scale:
                # Horizontal line monitor: tag it at its own height, near the middle of the line.
                tx, ty = 0.5 * (mh0 + mh1), mv0 + 0.07 * dv
            leg_handle = plt.Line2D([0], [0], color=mon_color, lw=1.9, ls="--")

        if m_idx < rem_slots:
            if m_idx == rem_slots - 1 and n_mons > rem_slots:
                short_tag = f"{item_num}–{item_num + n_mons - 1 - m_idx} · {short_w}"
            t_mon = ax.text(
                float(np.clip(tx, h0 + 0.06 * dh, h1 - 0.06 * dh)),
                ty,
                short_tag,
                color="#006B4E",
                fontsize=9.5,
                fontweight="bold",
                ha="center",
                va="center",
                path_effects=_text_halo(2.8),
                zorder=9,
            )
            t_mon._schematic_role = "in_axes"
            in_axes_count += 1

        legend_handles.append(leg_handle)
        legend_labels.append(leg_desc)

    # 8. Exact domain bounds on axes so in-axes == in-domain; outward spines for whitespace
    ax._is_schematic_axes = True
    ax.figure._wakis_custom_layout = True
    ax.set_xlim(h0, h1)
    ax.set_ylim(v0, v1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 20))

    if is_quasi_1d:
        ax.set_aspect("auto")
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.spines["left"].set_visible(False)
    else:
        ax.set_aspect("equal", adjustable="box")
        ax.spines["left"].set_position(("outward", 48))
        ax.set_ylabel(f"{v_ax} [{unit_label}]", fontsize=10.5)

    ax.set_xlabel(f"{h_ax} [{unit_label}]", fontsize=10.5)
    if title:
        ax.set_title(title, pad=40, fontsize=11.5, fontweight="bold")

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32 if is_quasi_1d else -0.26),
            ncol=min(2, len(legend_handles)),
            frameon=False,
            fontsize=9.5,
            handlelength=1.8,
            columnspacing=1.4,
        )

    return info


def plot_schematic(
    solver_or_grid: Any,
    plane: str | Sequence[str] = "auto",
    ax: Axes | Sequence[Axes] | None = None,
    cut_coord: float | None = None,
    structures: Sequence[Any] | None = None,
    material_names: Mapping[str, str] | None = None,
    sources: Any = None,
    monitors: Any = None,
    probes: Sequence[Any] | None = None,
    unit: str = "auto",
    show_mesh: bool = False,
    mesh_every: int | None = None,
    mesh_zoom: bool = False,
    show_pulse_inset: bool = False,
    show_wave_packet: bool = False,
    wave_packet_center: float | None = None,
    wavelength: float | None = None,
    freq: float | None = None,
    include_3d: bool = False,
    title: str | None = None,
    width_in: float = figstyle.FULL_WIDTH_IN,
    height_in: float | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    """Draw Figure A: clean setup schematic of geometry, materials, sources, monitors, and boundaries.

    Parameters
    ----------
    solver_or_grid : SolverFIT3D, GridFIT3D, SubgridFIT2D, UnstructuredFIT2D, or mesh
        Wakis solver, grid, or mesh object.
    plane : str or sequence of str, optional
        Cut plane (``'auto'``, ``'xy'``, ``'xz'``, ``'yz'``, ``'zx'``, ``'zy'``)
        or sequence of planes for multi-plane 3D views.
    ax : matplotlib.axes.Axes or sequence of Axes, optional
        Target axes. If ``None``, creates a new figure.
    cut_coord : float or None, optional
        Normal-axis coordinate [m] for 3D cuts.
    structures : sequence of shape objects, optional
        Explicit geometric shapes if not attached to ``solver``.
    material_names : mapping of str to str, optional
        Dictionary mapping solid keys or ``'region_i'`` to display names.
    sources : source object or sequence of source objects, optional
        Simulation source(s).
    monitors : monitor object or sequence of monitor objects, optional
        Simulation monitor(s).
    probes : sequence of float, tuple, or dict, optional
        Point probe locations [m] or dicts ``{'pos': ..., 'label': ...}``.
    unit : {'auto', 'nm', 'µm', 'um', 'mm', 'cm', 'm'}, optional
        Display length unit. Default is ``'auto'``.
    show_mesh : bool, optional
        If ``True``, overlay the computational mesh on the geometry.
    mesh_every : int or None, optional
        Grid line stride when ``show_mesh=True``.
    mesh_zoom : bool, optional
        If ``True`` and ``show_mesh=True``, include a 1× mesh zoom inset.
    show_pulse_inset : bool, optional
        Retained for backward compatibility (pulse details now live in
        :func:`plot_mesh_and_excitation`).
    show_wave_packet : bool, optional
        Retained for backward compatibility (wave packet now lives in
        :func:`plot_mesh_and_excitation`).
    wave_packet_center : float or None, optional
        Center coordinate [m] for the wave packet.
    wavelength : float or None, optional
        Reference wavelength [m].
    freq : float or None, optional
        Reference frequency [Hz].
    include_3d : bool, optional
        If ``True``, add a 3D isometric box panel next to the 2D cut.
    title : str or None, optional
        Panel title.
    width_in : float, optional
        Figure width in inches [in].
    height_in : float or None, optional
        Figure height in inches [in].

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing Figure A.
    axes : matplotlib.axes.Axes or numpy.ndarray of Axes
        The schematic axes.
    """
    planes = [plane] if isinstance(plane, str) else list(plane)
    n_panels = len(planes) + (1 if include_3d else 0)

    created_fig = False
    if ax is None:
        h_in = height_in if height_in is not None else (3.1 if n_panels == 1 else 3.4)
        fig, axes = figstyle.new_figure(
            width_in=width_in,
            height_in=h_in,
            ncols=n_panels,
            nrows=1,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.12, right=0.88, top=0.76, bottom=0.30, wspace=0.34)
        created_fig = True
    elif isinstance(ax, Axes):
        fig = ax.figure
        axes = ax
    else:
        axes = np.asarray(list(ax), dtype=object)
        fig = axes.flat[0].figure

    ax_list = [axes] if isinstance(axes, Axes) else list(np.asarray(axes).flat)

    last_info = None
    for idx, pl in enumerate(planes):
        cur_ax = ax_list[idx]
        info = extract_setup_info(
            solver_or_grid,
            plane=pl,
            cut_coord=cut_coord,
            structures=structures,
            material_names=material_names,
            sources=sources,
            monitors=monitors,
            probes=probes,
        )
        panel_title = (
            title
            if (title is not None and len(planes) == 1)
            else (
                f"{title} ({info['plane'].upper()} cut)"
                if title
                else f"Setup: geometry, sources, monitors & boundaries ({info['plane'].upper()} plane)"
            )
        )
        _draw_schematic_panel(
            cur_ax,
            info,
            unit=unit,
            show_mesh=show_mesh,
            mesh_every=mesh_every,
            mesh_zoom=mesh_zoom,
            show_pulse_inset=show_pulse_inset,
            show_wave_packet=show_wave_packet,
            wave_packet_center=wave_packet_center,
            wavelength=wavelength,
            freq=freq,
            title=panel_title,
        )
        last_info = info

    if include_3d and last_info is not None and len(ax_list) >= len(planes) + 1:
        pos = ax_list[-1].get_position()
        fig.delaxes(ax_list[-1])
        ax3d = fig.add_subplot(1, n_panels, n_panels, projection="3d", position=pos)
        b = last_info["bounds"]
        u_lbl, sc = choose_length_unit(
            max(b[0][1] - b[0][0], b[1][1] - b[1][0], b[2][1] - b[2][0]), unit=unit
        )
        x0, x1 = b[0][0] * sc, b[0][1] * sc
        y0, y1 = b[1][0] * sc, b[1][1] * sc
        z0, z1 = b[2][0] * sc, b[2][1] * sc
        edges_3d = [
            ([x0, x1], [y0, y0], [z0, z0]),
            ([x0, x1], [y1, y1], [z0, z0]),
            ([x0, x1], [y0, y0], [z1, z1]),
            ([x0, x1], [y1, y1], [z1, z1]),
            ([x0, x0], [y0, y1], [z0, z0]),
            ([x1, x1], [y0, y1], [z0, z0]),
            ([x0, x0], [y0, y1], [z1, z1]),
            ([x1, x1], [y0, y1], [z1, z1]),
            ([x0, x0], [y0, y0], [z0, z1]),
            ([x1, x1], [y0, y0], [z0, z1]),
            ([x0, x0], [y1, y1], [z0, z1]),
            ([x1, x1], [y1, y1], [z0, z1]),
        ]
        for ex, ey, ez in edges_3d:
            ax3d.plot(ex, ey, ez, color=figstyle.INK, lw=1.1)
        ax3d.set_xlabel(f"x [{u_lbl}]", fontsize=9.5)
        ax3d.set_ylabel(f"y [{u_lbl}]", fontsize=9.5)
        ax3d.set_zlabel(f"z [{u_lbl}]", fontsize=9.5)
        ax3d.set_title("3D domain view", fontsize=10.5)
        ax_list[-1] = ax3d

    if created_fig:
        figstyle.finish(fig, auto_legend_below=False)
    _resolve_text_overlaps(fig)
    return fig, axes


def plot_mesh_and_excitation(
    solver_or_grid: Any,
    plane: str = "auto",
    cut_coord: float | None = None,
    structures: Sequence[Any] | None = None,
    material_names: Mapping[str, str] | None = None,
    sources: Any = None,
    monitors: Any = None,
    probes: Sequence[Any] | None = None,
    unit: str = "auto",
    mesh_every: int | None = None,
    zoom_bounds: tuple[float, float, float, float] | None = None,
    wave_packet_center: float | None = None,
    wavelength: float | None = None,
    freq: float | None = None,
    min_feature: float | None = None,
    bandwidth_def: str = "-3dB",
    title: str | None = None,
    width_in: float = figstyle.FULL_WIDTH_IN,
    height_in: float | None = None,
) -> tuple[Figure, np.ndarray]:
    """Draw Figure B: Discretization (mesh & 1× zoom) and excitation (wave packet & pulse).

    Renders up to 3 primary panels:

    1. **Computational mesh**: whole domain with every *n*-th line and a 1× zoom
       inset at the smallest feature, annotated with :math:`\\Delta`, cells per
       wavelength in the highest-index material (:math:`n_{\\max}`), and cells
       across the smallest feature.
    2. **Wave packet in space**: spatial waveform and Gaussian envelope at one
       instant along the propagation axis with its spatial extent marked and
       domain/material/CPML regions lightly shaded for reference.
    3. **Source pulse & spectrum**: time trace :math:`s(t)` and Fourier spectrum
       :math:`|S(f)|` with :math:`f_0` and bandwidth (stating ``-3 dB`` or
       ``1/e``) marked and the simulated monitor frequency band shaded.

    Parameters
    ----------
    solver_or_grid : SolverFIT3D, GridFIT3D, SubgridFIT2D, UnstructuredFIT2D, or mesh
        Wakis solver, grid, or mesh object.
    plane : str, optional
        2D cut plane. Default is ``'auto'``.
    cut_coord : float or None, optional
        Normal-axis coordinate [m] for 3D cuts.
    structures : sequence of shape objects, optional
        Explicit geometric shapes if not attached to ``solver``.
    material_names : mapping of str to str, optional
        Dictionary mapping solid keys or ``'region_i'`` to display names.
    sources : source object or sequence of source objects, optional
        Simulation source(s).
    monitors : monitor object or sequence of monitor objects, optional
        Simulation monitor(s).
    probes : sequence, optional
        Point probe locations.
    unit : str, optional
        Display length unit. Default is ``'auto'``.
    mesh_every : int or None, optional
        Grid line stride for the full-domain mesh view.
    zoom_bounds : tuple of 4 floats, optional
        Custom ``(h0, h1, v0, v1)`` [m] bounds for the 1× mesh zoom inset.
    wave_packet_center : float or None, optional
        Center coordinate [m] for the spatial wave packet.
    wavelength : float or None, optional
        Reference free-space wavelength [m].
    freq : float or None, optional
        Reference center frequency [Hz].
    min_feature : float or None, optional
        Smallest geometric feature size [m].
    bandwidth_def : {'-3dB', '1/e'}, optional
        Bandwidth definition marked on the spectrum (``'-3dB'`` FWHM or
        ``'1/e'`` amplitude width). Default is ``'-3dB'``.
    title : str or None, optional
        Optional figure suptitle.
    width_in : float, optional
        Figure width [in]. Default is ``7.0``.
    height_in : float or None, optional
        Figure height [in]. Default is ``5.4``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing Figure B.
    axes : numpy.ndarray of Axes
        Array ``[ax_mesh, ax_packet, ax_spectrum]`` of the 3 primary panels.
    """
    info = extract_setup_info(
        solver_or_grid,
        plane=plane,
        cut_coord=cut_coord,
        structures=structures,
        material_names=material_names,
        sources=sources,
        monitors=monitors,
        probes=probes,
    )
    disc = _discretization_metrics(
        info, wavelength=wavelength, freq=freq, min_feature=min_feature
    )

    h_ax, v_ax = info["h_axis"], info["v_axis"]
    h_span_m = info["h_bounds"][1] - info["h_bounds"][0]
    v_span_m = info["v_bounds"][1] - info["v_bounds"][0]
    unit_label, scale = choose_length_unit(max(h_span_m, v_span_m), unit=unit)
    h0, h1 = info["h_bounds"][0] * scale, info["h_bounds"][1] * scale
    v0, v1 = info["v_bounds"][0] * scale, info["v_bounds"][1] * scale
    dh = max(h1 - h0, 1e-12)
    dv = max(v1 - v0, 1e-12)

    nv_cells = info["cell_counts"][info["v_idx"]]
    is_quasi_1d = nv_cells <= 2 and v_span_m < 0.15 * h_span_m

    h_in = height_in if height_in is not None else 5.5
    figstyle.apply_style()
    fig = plt.figure(figsize=(float(width_in), float(h_in)), dpi=figstyle.DEFAULT_DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.05], hspace=0.58, wspace=0.32)
    ax_mesh = fig.add_subplot(gs[0, :])
    ax_pkt = fig.add_subplot(gs[1, 0])
    ax_spec = fig.add_subplot(gs[1, 1])

    # --- Panel (a): Computational mesh + 1x zoom inset at smallest feature ---
    ax_mesh.add_patch(Rectangle((h0, v0), dh, dv, facecolor="#F8FAFC", edgecolor="none", zorder=1))
    for mat in info["materials"]:
        if mat["kind"] == "rect":
            mh0, mh1 = mat["h_bounds"][0] * scale, mat["h_bounds"][1] * scale
            mv0, mv1 = mat["v_bounds"][0] * scale, mat["v_bounds"][1] * scale
            ax_mesh.add_patch(
                Rectangle(
                    (mh0, mv0),
                    mh1 - mh0,
                    mv1 - mv0,
                    facecolor=mat["color"],
                    edgecolor="#1E293B",
                    lw=1.1,
                    alpha=0.75,
                    zorder=2,
                )
            )
        elif mat["kind"] == "circle":
            cx, cy = mat["center"][0] * scale, mat["center"][1] * scale
            rad = mat["radius"] * scale
            ax_mesh.add_patch(
                Circle(
                    (cx, cy),
                    rad,
                    facecolor=mat["color"],
                    edgecolor="#1E293B",
                    lw=1.1,
                    alpha=0.75,
                    zorder=2,
                )
            )
        elif mat["kind"] == "polygon":
            verts = np.asarray(mat["vertices"], dtype=float) * scale
            ax_mesh.add_patch(
                Polygon(
                    verts,
                    closed=True,
                    facecolor=mat["color"],
                    edgecolor="#1E293B",
                    lw=1.1,
                    alpha=0.75,
                    zorder=2,
                )
            )
    th_h_low = info["pml_thickness"][f"{h_ax}_low"] * scale
    th_h_high = info["pml_thickness"][f"{h_ax}_high"] * scale
    th_v_low = info["pml_thickness"][f"{v_ax}_low"] * scale
    th_v_high = info["pml_thickness"][f"{v_ax}_high"] * scale
    if th_h_low > 0.0:
        ax_mesh.add_patch(
            Rectangle(
                (h0, v0),
                th_h_low,
                dv,
                facecolor="#CBD5E1",
                edgecolor="#64748B",
                hatch="///",
                alpha=0.35,
                lw=0.8,
                zorder=2,
            )
        )
    if th_h_high > 0.0:
        ax_mesh.add_patch(
            Rectangle(
                (h1 - th_h_high, v0),
                th_h_high,
                dv,
                facecolor="#CBD5E1",
                edgecolor="#64748B",
                hatch="///",
                alpha=0.35,
                lw=0.8,
                zorder=2,
            )
        )
    if not is_quasi_1d and th_v_low > 0.0:
        ax_mesh.add_patch(
            Rectangle(
                (h0, v0),
                dh,
                th_v_low,
                facecolor="#CBD5E1",
                edgecolor="#64748B",
                hatch="\\\\\\",
                alpha=0.35,
                lw=0.8,
                zorder=2,
            )
        )
    if not is_quasi_1d and th_v_high > 0.0:
        ax_mesh.add_patch(
            Rectangle(
                (h0, v1 - th_v_high),
                dh,
                th_v_high,
                facecolor="#CBD5E1",
                edgecolor="#64748B",
                hatch="\\\\\\",
                alpha=0.35,
                lw=0.8,
                zorder=2,
            )
        )

    mesh_meta = draw_mesh(
        ax_mesh,
        info["solver"] if info["solver"] is not None else info["grid"],
        plane=info["plane"],
        unit=unit_label,
        every=mesh_every,
        wavelength=disc["wavelength_m"],
        zoom_inset=True,
        zoom_bounds=zoom_bounds,
        show_cpw=False,
    )
    ax_mesh.add_patch(
        Rectangle((h0, v0), dh, dv, facecolor="none", edgecolor=figstyle.INK, lw=1.4, zorder=6)
    )
    ax_mesh.set_xlim(h0, h1)
    ax_mesh.set_ylim(v0, v1)
    if is_quasi_1d:
        ax_mesh.set_aspect("auto")
        ax_mesh.set_yticks([])
        ax_mesh.set_ylabel("")
        ax_mesh.spines["left"].set_visible(False)
    else:
        ax_mesh.set_aspect("equal", adjustable="box")
        ax_mesh.set_ylabel(f"{v_ax} [{unit_label}]", fontsize=10.0)
    ax_mesh.set_xlabel(f"{h_ax} [{unit_label}]", fontsize=10.0)

    min_cell_str = _format_length(disc["min_cell_m"])
    cpw_str = (
        f"{disc['cpw_nmax']:.1f} cells/λ (n_max = {disc['n_max']:.2f})"
        if disc["cpw_nmax"] is not None
        else f"N_{h_ax} = {info['cell_counts'][info['h_idx']]} cells"
    )
    feat_str = (
        f"{disc['cells_across_feature']:.1f} cells / {disc['min_feature_name']} "
        f"({_format_length(disc['min_feature_m'])})"
        if disc["cells_across_feature"] is not None and disc["min_feature_m"] is not None
        else ""
    )
    stride_note = (
        f"every {mesh_meta['stride_h']}th line"
        if mesh_meta["stride_h"] > 1
        else "all lines"
    )
    ax_mesh.set_title(
        f"(a) Mesh ({stride_note}): Δ{h_ax} = {min_cell_str} · {cpw_str}"
        + (f" · {feat_str}" if feat_str else ""),
        fontsize=10.2,
        fontweight="bold",
    )

    # --- Extract pulse parameters for Panels (b) and (c) ---
    pulse = None
    for s in info["sources"]:
        if s.get("pulse") is not None:
            pulse = s["pulse"]
            break
    f0 = float(pulse["f0"]) if pulse is not None else (
        float(freq)
        if freq is not None
        else (c_light / float(disc["wavelength_m"]) if disc["wavelength_m"] else 300e12)
    )
    fwidth = float(pulse["fwidth"]) if pulse is not None else 0.35 * f0
    lam0_m = c_light / max(f0, 1.0)
    sigma_z_m = c_light / (np.pi * max(fwidth, 1e-30))
    dz_pulse_m = 2.0 * sigma_z_m

    # --- Panel (b): Wave packet in space at one instant along propagation axis ---
    if th_h_low > 0.0:
        ax_pkt.axvspan(
            h0, h0 + th_h_low, facecolor="#CBD5E1", edgecolor="#64748B", hatch="///", alpha=0.35, label="CPML"
        )
    if th_h_high > 0.0:
        ax_pkt.axvspan(
            h1 - th_h_high, h1, facecolor="#CBD5E1", edgecolor="#64748B", hatch="///", alpha=0.35
        )
    for m_idx, mat in enumerate(info["materials"]):
        mh0, mh1 = mat["h_bounds"][0] * scale, mat["h_bounds"][1] * scale
        ax_pkt.axvspan(
            mh0,
            mh1,
            color=mat["color"],
            alpha=0.55,
            label=mat["name"] if m_idx == 0 else None,
        )

    if wave_packet_center is not None:
        wc = float(wave_packet_center) * scale
    else:
        wc = 0.5 * (h0 + th_h_low + h1 - th_h_high)

    z_grid = np.linspace(h0, h1, 500)
    sigma_scaled = max(sigma_z_m * scale, 0.04 * dh)
    env = np.exp(-0.5 * ((z_grid - wc) / sigma_scaled) ** 2)
    k_vis = 2.0 * np.pi / max(lam0_m * scale, 0.04 * dh)
    carrier = env * np.cos(k_vis * (z_grid - wc))

    ax_pkt.plot(z_grid, carrier, color="#0072B2", lw=1.6, marker="none", label="Wave packet E(z)")
    ax_pkt.plot(z_grid, env, color="#0072B2", lw=1.0, ls=":", marker="none", alpha=0.85, label="±Envelope")
    ax_pkt.plot(z_grid, -env, color="#0072B2", lw=1.0, ls=":", marker="none", alpha=0.85)

    w_lo = max(h0 + 0.02 * dh, wc - sigma_scaled)
    w_hi = min(h1 - 0.02 * dh, wc + sigma_scaled)
    ax_pkt.annotate(
        "",
        xy=(w_hi, -1.16),
        xytext=(w_lo, -1.16),
        arrowprops={"arrowstyle": "<->", "color": "#0072B2", "lw": 1.2},
    )
    t_wp = ax_pkt.text(
        0.5 * (w_lo + w_hi),
        -1.24,
        f"2σ_z = {_format_length(dz_pulse_m)} (λ₀ = {_format_length(lam0_m)})",
        ha="center",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#005A8D",
        path_effects=_text_halo(2.4),
    )
    t_wp._schematic_role = "in_axes"

    ax_pkt.set_xlim(h0, h1)
    ax_pkt.set_ylim(-1.55, 1.35)
    ax_pkt.set_xlabel(f"{h_ax} [{unit_label}]", fontsize=10.0)
    ax_pkt.set_ylabel("Normalized E", fontsize=10.0)
    ax_pkt.set_title("(b) Wave packet (1/e extent)", fontsize=10.0, fontweight="bold")
    figstyle.spectrum_grid(ax_pkt)
    ax_pkt.legend(loc="upper right", frameon=False, fontsize=9.2, ncol=2)

    # --- Panel (c): Source pulse spectrum |S(f)| + time-trace inset s(t) ---
    f_unit, f_sc = _choose_freq_unit(f0)
    f_arr = np.linspace(max(0.05 * f0, f0 - 2.4 * fwidth), f0 + 2.4 * fwidth, 400)
    src_time_obj = pulse.get("src_time") if pulse is not None else None
    if src_time_obj is not None and hasattr(src_time_obj, "fourier_transform"):
        s_f = np.abs(np.asarray(src_time_obj.fourier_transform(f_arr), dtype=complex))
    else:
        sigma_f = fwidth / (2.0 * np.sqrt(2.0 * np.log(2.0))) if fwidth > 0 else 0.15 * f0
        s_f = (f_arr / max(f0, 1.0)) * np.exp(-0.5 * ((f_arr - f0) / max(sigma_f, 1e-30)) ** 2)
    s_f_norm = s_f / max(float(np.max(s_f)), 1e-30)

    # Bandwidth calculation (-3 dB FWHM or 1/e amplitude)
    bw_key = bandwidth_def.strip().lower()
    thresh = (1.0 / np.e) if "e" in bw_key else (1.0 / np.sqrt(2.0))
    bw_label_name = "1/e" if "e" in bw_key else "−3 dB"
    above = np.where(s_f_norm >= thresh)[0]
    if above.size >= 2:
        f_bw_lo = float(f_arr[above[0]])
        f_bw_hi = float(f_arr[above[-1]])
    else:
        f_bw_lo = f0 - 0.5 * fwidth
        f_bw_hi = f0 + 0.5 * fwidth
    bw_hz = f_bw_hi - f_bw_lo

    # Simulated monitor band
    mon_freqs_all = [
        m["frequencies"] for m in info["monitors"] if m.get("frequencies") is not None and len(m["frequencies"]) > 0
    ]
    if mon_freqs_all:
        all_f = np.concatenate(mon_freqs_all)
        f_sim_lo = float(np.min(all_f))
        f_sim_hi = float(np.max(all_f))
    else:
        f_sim_lo, f_sim_hi = f_bw_lo, f_bw_hi

    if f_sim_hi > f_sim_lo:
        ax_spec.axvspan(
            f_sim_lo * f_sc,
            f_sim_hi * f_sc,
            color="#D1FAE5",
            alpha=0.65,
            label="Simulated band",
        )

    ax_spec.plot(
        f_arr * f_sc,
        s_f_norm,
        color="#D55E00",
        lw=1.9,
        marker="none",
        label="Spectrum |S(f)|",
    )
    ax_spec.axvline(
        f0 * f_sc,
        color=figstyle.INK,
        ls="--",
        lw=1.2,
        label=f"f₀ = {_format_freq(f0)}",
    )
    ax_spec.annotate(
        "",
        xy=(f_bw_hi * f_sc, thresh),
        xytext=(f_bw_lo * f_sc, thresh),
        arrowprops={"arrowstyle": "<->", "color": "#0072B2", "lw": 1.3},
    )
    t_bw = ax_spec.text(
        0.5 * (f_bw_lo + f_bw_hi) * f_sc,
        0.30,
        f"Δf ({bw_label_name}) = {_format_freq(bw_hz)}",
        ha="center",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color="#005A8D",
        path_effects=_text_halo(2.5),
    )
    t_bw._schematic_role = "in_axes"

    # Companion time-trace inset s(t)
    t_end = float(pulse["t_end"]) if pulse is not None else 8.0 / max(f0, 1.0)
    t_unit, t_sc = _choose_time_unit(t_end)
    t_arr = np.linspace(0.0, t_end, 300)
    if src_time_obj is not None and callable(src_time_obj):
        s_t = np.asarray(src_time_obj(t_arr), dtype=float)
    else:
        t0_p = 0.5 * t_end
        sig_t = 1.0 / (np.pi * max(fwidth, 1e-30))
        s_t = np.exp(-0.5 * ((t_arr - t0_p) / sig_t) ** 2) * np.cos(2.0 * np.pi * f0 * (t_arr - t0_p))
    s_t_norm = s_t / max(float(np.max(np.abs(s_t))), 1e-30)

    ins_t = ax_spec.inset_axes([0.65, 0.64, 0.32, 0.33])
    ins_t._is_inset = True
    ins_t.set_in_layout(False)
    ins_t.set_facecolor(figstyle.PAPER)
    for sp in ("top", "right"):
        ins_t.spines[sp].set_visible(False)
    ins_t.plot(t_arr * t_sc, s_t_norm, color="#0072B2", lw=1.2, marker="none")
    ins_t.axhline(0.0, color="#94A3B8", ls=":", lw=0.7)
    ins_t.set_xlim(0.0, t_end * t_sc)
    ins_t.set_ylim(-1.15, 1.15)
    ins_t.set_yticks([])
    ins_t.tick_params(axis="x", labelsize=9.2)
    ins_lbl = ins_t.text(
        0.5,
        0.92,
        f"s(t) [{t_unit}]",
        transform=ins_t.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        color=figstyle.INK,
        path_effects=_text_halo(2.2),
    )
    ins_lbl._schematic_role = "inset"

    ax_spec.set_xlim(float(f_arr[0] * f_sc), float(f_arr[-1] * f_sc))
    ax_spec.set_ylim(0.0, 1.32)
    ax_spec.set_xlabel(f"Frequency f [{f_unit}]", fontsize=10.0)
    ax_spec.set_ylabel("Normalized |S(f)|", fontsize=10.0)
    ax_spec.set_title(
        f"(c) Pulse & spectrum ({bw_label_name} BW)",
        fontsize=10.0,
        fontweight="bold",
    )
    figstyle.spectrum_grid(ax_spec)
    ax_spec.legend(loc="upper left", frameon=False, fontsize=9.2)

    if title:
        fig.suptitle(title, y=0.99, fontsize=11.5, fontweight="bold")

    figstyle.finish(fig, max_panels=3, auto_legend_below=False)
    _resolve_text_overlaps(fig)
    return fig, np.array([ax_mesh, ax_pkt, ax_spec], dtype=object)


def setup_table(
    solver_or_grid: Any,
    plane: str = "auto",
    cut_coord: float | None = None,
    structures: Sequence[Any] | None = None,
    material_names: Mapping[str, str] | None = None,
    sources: Any = None,
    monitors: Any = None,
    probes: Sequence[Any] | None = None,
    unit: str = "auto",
    wavelength: float | None = None,
    freq: float | None = None,
    min_feature: float | None = None,
    variations: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Return a Markdown table summarizing the complete simulation setup.

    Lists domain size and cell counts, boundary conditions on every face,
    materials (:math:`\\varepsilon_r`, :math:`\\mu_r`, model, and bounds),
    sources (type, position, polarization, pulse :math:`f_0` and
    :math:`\\Delta f`), monitors/probes (type, position, frequencies), and
    discretization (:math:`\\Delta`, :math:`n_{\\max}`, cells per wavelength,
    and cells across the smallest feature).

    Parameters
    ----------
    solver_or_grid : SolverFIT3D, GridFIT3D, SubgridFIT2D, UnstructuredFIT2D, or mesh
        Wakis solver, grid, or mesh object.
    plane : str, optional
        2D cut plane. Default is ``'auto'``.
    cut_coord : float or None, optional
        Normal-axis coordinate [m] for 3D cuts.
    structures : sequence of shape objects, optional
        Explicit geometric shapes if not attached to ``solver``.
    material_names : mapping of str to str, optional
        Dictionary mapping solid keys or ``'region_i'`` to display names.
    sources : source object or sequence of source objects, optional
        Simulation source(s).
    monitors : monitor object or sequence of monitor objects, optional
        Simulation monitor(s).
    probes : sequence, optional
        Point probe locations.
    unit : str, optional
        Display length unit. Default is ``'auto'``.
    wavelength : float or None, optional
        Reference wavelength [m].
    freq : float or None, optional
        Reference frequency [Hz].
    min_feature : float or None, optional
        Smallest geometric feature size [m].
    variations : sequence of dict, optional
        Additional run variations to append to the table (each dict with
        ``'name'`` and ``'details'`` or parameter keys).

    Returns
    -------
    table_md : str
        Formatted GitHub-flavored Markdown table.
    """
    info = extract_setup_info(
        solver_or_grid,
        plane=plane,
        cut_coord=cut_coord,
        structures=structures,
        material_names=material_names,
        sources=sources,
        monitors=monitors,
        probes=probes,
    )
    disc = _discretization_metrics(
        info, wavelength=wavelength, freq=freq, min_feature=min_feature
    )
    b = info["bounds"]
    cc = info["cell_counts"]
    lx_m, ly_m, lz_m = b[0][1] - b[0][0], b[1][1] - b[1][0], b[2][1] - b[2][0]

    rows: list[tuple[str, str, str]] = []
    if info["is_2d_mesh"]:
        rows.append(
            (
                "Domain",
                "Size & bounds",
                f"Lx × Ly = {_format_length(lx_m)} × {_format_length(ly_m)} "
                f"(x ∈ [{_format_length(b[0][0])}, {_format_length(b[0][1])}], "
                f"y ∈ [{_format_length(b[1][0])}, {_format_length(b[1][1])}])",
            )
        )
        rows.append(("Domain", "Grid cells", f"Nx × Ny = {cc[0]} × {cc[1]} ({cc[0] * cc[1]} cells)"))
        face_axes = ("x", "y")
    else:
        rows.append(
            (
                "Domain",
                "Size & bounds",
                f"Lx × Ly × Lz = {_format_length(lx_m)} × {_format_length(ly_m)} × {_format_length(lz_m)} "
                f"(z ∈ [{_format_length(b[2][0])}, {_format_length(b[2][1])}])",
            )
        )
        rows.append(
            (
                "Domain",
                "Grid cells",
                f"Nx × Ny × Nz = {cc[0]} × {cc[1]} × {cc[2]} ({cc[0] * cc[1] * cc[2]} cells)",
            )
        )
        face_axes = ("x", "y", "z")

    # Discretization row
    cpw_txt = (
        f"{disc['cpw_nmax']:.1f} cells/λ (in n_max = {disc['n_max']:.2f})"
        if disc["cpw_nmax"] is not None
        else "—"
    )
    feat_txt = (
        f"{disc['cells_across_feature']:.1f} cells across {disc['min_feature_name']} "
        f"({_format_length(disc['min_feature_m'])})"
        if disc["cells_across_feature"] is not None and disc["min_feature_m"] is not None
        else "—"
    )
    rows.append(
        (
            "Discretization",
            f"Δ{info['h_axis']} & resolution",
            f"Δ_min = {_format_length(disc['min_cell_m'])}, {cpw_txt}, {feat_txt}",
        )
    )

    # Every boundary face listed explicitly
    for ax_c in face_axes:
        rows.append(("Boundary", f"-{ax_c} face", info["bc_labels"][f"{ax_c}_low"]))
        rows.append(("Boundary", f"+{ax_c} face", info["bc_labels"][f"{ax_c}_high"]))

    # Materials
    rows.append(("Material", "Background", f"{info['bg_label']} (isotropic)"))
    h_ax = info["h_axis"]
    for idx, mat in enumerate(info["materials"], start=1):
        mh0, mh1 = mat["h_bounds"]
        thick_m = abs(mh1 - mh0)
        d_suffix = (
            ""
            if ("(d =" in mat["name"] or "(w =" in mat["name"])
            else f" (d = {_format_length(thick_m)})"
        )
        rows.append(
            (
                "Material",
                f"Region {idx}: {mat['name']}",
                f"ε_r = {mat['eps_r']:.2f}, μ_r = {mat['mu_r']:.2f}, model = {mat['model']}, "
                f"{h_ax} ∈ [{_format_length(mh0)}, {_format_length(mh1)}]{d_suffix}",
            )
        )

    # Sources (numbered matching Figure A)
    item_num = 0
    for src in info["sources"]:
        item_num += 1
        p = src.get("pulse")
        pulse_str = (
            f"f₀ = {_format_freq(p['f0'])}, Δf = {_format_freq(p['fwidth'])}"
            if p is not None
            else "continuous / custom"
        )
        rows.append(
            (
                "Source",
                f"{item_num} · {src['type_name']}",
                f"pos: {src.get('pos_str', '—')}, pol: {src.get('polarization', '—')}, pulse: {pulse_str}",
            )
        )

    # Monitors & probes (continuing numbering matching Figure A)
    for mon in info["monitors"]:
        item_num += 1
        rows.append(
            (
                "Monitor",
                f"{item_num} · {mon['type_name']}",
                f"{mon.get('label', mon['type_name'])}, pos: {mon.get('pos_str', '—')}, "
                f"freqs: {mon.get('freq_str', 'time-domain')}",
            )
        )

    if variations:
        for var in variations:
            if isinstance(var, Mapping):
                v_name = str(var.get("name", "Run variation"))
                v_det = str(
                    var.get("details", ", ".join(f"{k}={v}" for k, v in var.items() if k != "name"))
                )
            elif isinstance(var, Sequence) and not isinstance(var, str) and len(var) >= 2:
                v_name, v_det = str(var[0]), str(var[1])
            else:
                v_name, v_det = "Run variation", str(var)
            rows.append(("Variation", v_name, v_det))

    lines = [
        "| Category | Component / Face | Specification |",
        "| :--- | :--- | :--- |",
    ]
    for cat, comp, spec in rows:
        lines.append(f"| {cat} | {comp} | {spec} |")
    return "\n".join(lines)


def save_setup_figures(
    solver_or_grid: Any,
    out_dir: str | Path,
    *,
    prefix: str = "00",
    suffix: str = "",
    plane: str | Sequence[str] = "auto",
    cut_coord: float | None = None,
    structures: Sequence[Any] | None = None,
    material_names: Mapping[str, str] | None = None,
    sources: Any = None,
    monitors: Any = None,
    probes: Sequence[Any] | None = None,
    unit: str = "auto",
    wavelength: float | None = None,
    freq: float | None = None,
    min_feature: float | None = None,
    bandwidth_def: str = "-3dB",
    title_geometry: str | None = None,
    title_mesh: str | None = None,
    variations: Sequence[Mapping[str, Any]] | None = None,
    dpi: int = figstyle.DEFAULT_DPI,
    write_table: bool = True,
) -> dict[str, Any]:
    """Render and save Figure A (``00a``), Figure B (``00b``), and the Markdown setup table.

    Parameters
    ----------
    solver_or_grid : SolverFIT3D, GridFIT3D, SubgridFIT2D, UnstructuredFIT2D, or mesh
        Wakis solver, grid, or mesh object.
    out_dir : str or Path
        Output directory for ``00a_setup_geometry.png`` and
        ``00b_setup_mesh_and_excitation.png``.
    prefix : str, optional
        Filename prefix. Default is ``'00'``.
    suffix : str, optional
        Optional filename suffix before ``.png`` (e.g. ``'_oblique'``).
    plane : str or sequence of str, optional
        Cut plane. Default is ``'auto'``.
    cut_coord : float or None, optional
        Normal-axis coordinate [m] for 3D cuts.
    structures : sequence of shape objects, optional
        Explicit geometric shapes if not attached to ``solver``.
    material_names : mapping of str to str, optional
        Dictionary mapping solid keys or ``'region_i'`` to display names.
    sources : source object or sequence of source objects, optional
        Simulation source(s).
    monitors : monitor object or sequence of monitor objects, optional
        Simulation monitor(s).
    probes : sequence, optional
        Point probe locations.
    unit : str, optional
        Display length unit. Default is ``'auto'``.
    wavelength : float or None, optional
        Reference wavelength [m].
    freq : float or None, optional
        Reference frequency [Hz].
    min_feature : float or None, optional
        Smallest geometric feature size [m].
    bandwidth_def : {'-3dB', '1/e'}, optional
        Bandwidth definition for Figure B. Default is ``'-3dB'``.
    title_geometry : str or None, optional
        Title for Figure A (``00a``).
    title_mesh : str or None, optional
        Title for Figure B (``00b``).
    variations : sequence of dict, optional
        Additional run variations included in the Markdown setup table.
    dpi : int, optional
        Output image resolution [dots/in]. Default is ``200``.
    write_table : bool, optional
        If ``True``, also write ``<prefix>_setup_table<suffix>.md`` in ``out_dir``.

    Returns
    -------
    result : dict
        Dictionary with keys ``'geometry_figure'``, ``'mesh_figure'``,
        ``'table'``, and ``'table_path'``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    path_a = out_path / f"{prefix}a_setup_geometry{suffix}.png"
    path_b = out_path / f"{prefix}b_setup_mesh_and_excitation{suffix}.png"
    path_md = out_path / f"{prefix}_setup_table{suffix}.md"

    fig_a, _ = plot_schematic(
        solver_or_grid,
        plane=plane,
        cut_coord=cut_coord,
        structures=structures,
        material_names=material_names,
        sources=sources,
        monitors=monitors,
        probes=probes,
        unit=unit,
        wavelength=wavelength,
        freq=freq,
        title=title_geometry,
    )
    figstyle.save(fig_a, path_a, dpi=dpi, close=True, auto_legend_below=False)

    primary_plane = plane if isinstance(plane, str) else list(plane)[0]
    fig_b, _ = plot_mesh_and_excitation(
        solver_or_grid,
        plane=primary_plane,
        cut_coord=cut_coord,
        structures=structures,
        material_names=material_names,
        sources=sources,
        monitors=monitors,
        probes=probes,
        unit=unit,
        wavelength=wavelength,
        freq=freq,
        min_feature=min_feature,
        bandwidth_def=bandwidth_def,
        title=title_mesh,
    )
    figstyle.save(fig_b, path_b, dpi=dpi, close=True, auto_legend_below=False)

    md_table = setup_table(
        solver_or_grid,
        plane=primary_plane,
        cut_coord=cut_coord,
        structures=structures,
        material_names=material_names,
        sources=sources,
        monitors=monitors,
        probes=probes,
        unit=unit,
        wavelength=wavelength,
        freq=freq,
        min_feature=min_feature,
        variations=variations,
    )
    if write_table:
        path_md.write_text(md_table + "\n", encoding="utf-8")

    return {
        "geometry_figure": path_a,
        "mesh_figure": path_b,
        "table": md_table,
        "table_path": path_md if write_table else None,
    }
