# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# SPDX-License-Identifier: Apache-2.0         #
# ########################################### #

"""Publication and textbook figure style for Wakis (`wakis.figstyle`).

Provides a high-contrast, colour-blind-safe light journal style designed for
computational electromagnetics figures and animations:

* White background (``#ffffff``), near-black ink (``#1a1a1a``), and sans-serif
  typography (``Arial`` when installed, falling back to ``DejaVu Sans`` /
  ``Liberation Sans``).
* Literal print/display sizing at ``200 dpi``: full-width figures at
  ``7.0 in``, half-width figures at ``3.4 in``, axis labels at ``11 pt``,
  tick labels and legends at ``10 pt``, panel titles at ``11.5 pt``, and a hard
  floor of ``8.5 pt`` enforced by :func:`finish`.
* Clean axes with top and right spines removed, outward ticks, no grid by
  default (light dotted grid optional for spectra via :func:`spectrum_grid`),
  and frameless legends placed below panels when they would cover data.
* High-contrast Okabe–Ito categorical palette where every series receives a
  distinct colour, line style, and marker.
* Standardized field colormaps: ``RdBu_r`` with symmetric colour limits for
  diverging real fields, ``inferno`` / ``magma`` for field magnitudes and
  intensities, and a light grey ramp (``wakis_eps``) for permittivity maps so
  that geometry and annotation overlays remain legible.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.text import Text

__all__ = [
    "FULL_WIDTH_IN",
    "HALF_WIDTH_IN",
    "DEFAULT_DPI",
    "INK",
    "PAPER",
    "PT_LABEL",
    "PT_TICK",
    "PT_LEGEND",
    "PT_TITLE",
    "MIN_PT",
    "MIN_CONTRAST_RATIO",
    "OKABE_ITO",
    "PALETTE",
    "LINESTYLES",
    "MARKERS",
    "CMAP_DIVERGING",
    "CMAP_MAGNITUDE",
    "CMAP_INTENSITY",
    "CMAP_EPS",
    "apply_style",
    "new_figure",
    "finish",
    "save",
    "series_style",
    "symmetric_limits",
    "spectrum_grid",
    "place_legend_below",
    "panel_label",
    "contrast_ratio",
    "darken_for_contrast",
]

# --- Display geometry & typography constants --------------------------------
FULL_WIDTH_IN: float = 7.0
HALF_WIDTH_IN: float = 3.4
DEFAULT_DPI: int = 200

PAPER: str = "#ffffff"
INK: str = "#1a1a1a"

PT_LABEL: float = 11.0
PT_TICK: float = 10.0
PT_LEGEND: float = 10.0
PT_TITLE: float = 11.5
MIN_PT: float = 8.5
MIN_CONTRAST_RATIO: float = 4.5

# --- Okabe-Ito colour-blind-safe categorical palette ------------------------
# Ordered for high contrast against white; the yellow slot (#F0E442 in raw
# Okabe-Ito has 1.3:1 contrast on white) uses darkened ochre (#8C6D00) so
# lines and markers remain visible on a white page.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#8C6D00",  # dark gold / ochre
    "#1A1A1A",  # near-black
)
PALETTE: tuple[str, ...] = OKABE_ITO

LINESTYLES: tuple[Any, ...] = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1.5)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (2, 1)),
    (0, (4, 1, 1, 1)),
)

MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

# --- Standard colormaps for electromagnetic fields & materials --------------
CMAP_DIVERGING: str = "RdBu_r"
CMAP_MAGNITUDE: str = "inferno"
CMAP_INTENSITY: str = "magma"
CMAP_EPS: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "wakis_eps",
    ["#ffffff", "#eef1f5", "#d8dee6", "#b8c2cc"],
    N=256,
)

if "wakis_eps" not in plt.colormaps():
    mpl.colormaps.register(CMAP_EPS)


def _select_sans_font() -> list[str]:
    """Return the preferred sans-serif font stack starting with Arial if installed."""
    candidates = ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica"]
    available = {f.name for f in fm.fontManager.ttflist}
    ordered = [name for name in candidates if name in available]
    for name in candidates:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _relative_luminance(color: Any) -> float:
    """Compute WCAG 2.1 relative luminance in ``[0, 1]`` for a matplotlib colour."""
    r, g, b = to_rgb(color)
    channels = []
    for c in (r, g, b):
        if c <= 0.04045:
            channels.append(c / 12.92)
        else:
            channels.append(((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast_ratio(color: Any, background: Any = PAPER) -> float:
    """Compute the WCAG 2.1 contrast ratio between ``color`` and ``background``.

    Parameters
    ----------
    color : str or tuple
        Foreground colour specification.
    background : str or tuple, optional
        Background colour specification. Default is ``'#ffffff'``.

    Returns
    -------
    float
        Contrast ratio in ``[1.0, 21.0]`` [dimensionless].
    """
    l1 = _relative_luminance(color)
    l2 = _relative_luminance(background)
    lighter, darker = max(l1, l2), min(l1, l2)
    return float((lighter + 0.05) / (darker + 0.05))


def darken_for_contrast(
    color: Any,
    min_contrast: float = MIN_CONTRAST_RATIO,
    background: Any = PAPER,
) -> str:
    """Darken ``color`` preserving hue until its contrast against ``background`` >= ``min_contrast``.

    Parameters
    ----------
    color : str or tuple
        Input matplotlib colour.
    min_contrast : float, optional
        Minimum WCAG contrast ratio [dimensionless]. Default is 4.5.
    background : str or tuple, optional
        Background colour. Default is ``'#ffffff'``.

    Returns
    -------
    str
        Hex colour string ``'#rrggbb'`` meeting the contrast requirement.
    """
    try:
        rgb = np.array(to_rgb(color), dtype=float)
    except ValueError:
        return INK
    if contrast_ratio(rgb, background) >= min_contrast:
        return to_hex(rgb)

    lo, hi = 0.0, 1.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if contrast_ratio(rgb * mid, background) >= min_contrast + 0.05:
            lo = mid
        else:
            hi = mid
    return to_hex(rgb * lo)


def series_style(
    index: int,
    *,
    with_marker: bool = True,
    with_line: bool = True,
    lw: float = 1.9,
    ms: float = 5.2,
    markevery: int | None = None,
) -> dict[str, Any]:
    """Return a distinct ``(color, linestyle, marker)`` dictionary for series ``index``.

    Parameters
    ----------
    index : int
        Zero-based series index.
    with_marker : bool, optional
        Include the marker key. Default is True.
    with_line : bool, optional
        Include a connecting line (``linestyle='none'`` when False). Default is True.
    lw : float, optional
        Line width [pt]. Default is 1.9.
    ms : float, optional
        Marker size [pt]. Default is 5.2.
    markevery : int or None, optional
        Stride between markers along the curve.

    Returns
    -------
    dict
        Keyword arguments suitable for :meth:`matplotlib.axes.Axes.plot`.
    """
    i = int(index)
    style: dict[str, Any] = {
        "color": PALETTE[i % len(PALETTE)],
        "linestyle": LINESTYLES[i % len(LINESTYLES)] if with_line else "none",
        "linewidth": float(lw),
    }
    if with_marker:
        style["marker"] = MARKERS[i % len(MARKERS)]
        style["markersize"] = float(ms)
        if markevery is not None:
            style["markevery"] = int(markevery)
    else:
        style["marker"] = "none"
    return style


def symmetric_limits(values: np.ndarray, percentile: float | None = None) -> tuple[float, float]:
    """Return symmetric colour limits ``(-vmax, +vmax)`` for diverging field maps.

    Parameters
    ----------
    values : array_like
        Real field array.
    percentile : float, optional
        If provided (e.g. ``99.0``), compute ``vmax`` from ``np.nanpercentile(|values|, percentile)``.
        Default uses the exact maximum ``np.nanmax(|values|)``.

    Returns
    -------
    vmin, vmax : tuple of float
        Symmetric limits ``(-vmax, +vmax)`` in the units of ``values``.
    """
    arr = np.abs(np.asarray(values, dtype=float))
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(finite, percentile) if percentile is not None else np.max(finite))
    if vmax <= 0.0:
        vmax = 1.0
    return -vmax, vmax


class _StyleContext:
    """Context manager returned by :func:`apply_style` that restores previous ``rcParams`` on exit."""

    def __init__(self, previous_rc: dict[str, Any]) -> None:
        self._previous_rc = previous_rc

    def __enter__(self) -> _StyleContext:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        mpl.rcParams.update(self._previous_rc)


def _style_rcparams() -> dict[str, Any]:
    """Build the dictionary of matplotlib ``rcParams`` for the Wakis book style."""
    font_stack = _select_sans_font()
    prop_cycle = mpl.cycler(
        color=list(PALETTE),
        linestyle=list(LINESTYLES),
        marker=list(MARKERS),
    )
    return {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.edgecolor": PAPER,
        "figure.dpi": DEFAULT_DPI,
        "savefig.dpi": DEFAULT_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "font.family": font_stack,  # a list of names enables per-glyph fallback (e.g. Arial lacks ∈, ⁻)
        "font.sans-serif": font_stack,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": PT_LABEL,
        "axes.labelsize": PT_LABEL,
        "axes.titlesize": PT_TITLE,
        "figure.titlesize": PT_TITLE,
        "xtick.labelsize": PT_TICK,
        "ytick.labelsize": PT_TICK,
        "legend.fontsize": PT_LEGEND,
        "legend.title_fontsize": PT_LEGEND,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": "#cbd5e1",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "lines.linewidth": 1.9,
        "lines.markersize": 5.2,
        "axes.prop_cycle": prop_cycle,
        "image.cmap": CMAP_DIVERGING,
    }


def apply_style() -> _StyleContext:
    """Apply the Wakis publication figure style to ``matplotlib.rcParams``.

    Can be called directly as ``apply_style()`` or used as a ``with apply_style():``
    context manager that restores the previous ``rcParams`` upon exit.

    Returns
    -------
    _StyleContext
        Context manager restoring the previous ``rcParams`` on ``__exit__``.
    """
    prev = dict(mpl.rcParams)
    if "wakis_eps" not in plt.colormaps():
        mpl.colormaps.register(CMAP_EPS)
    mpl.rcParams.update(_style_rcparams())
    return _StyleContext(prev)


def new_figure(
    width_in: float = FULL_WIDTH_IN,
    height_in: float | None = None,
    ncols: int = 1,
    nrows: int = 1,
    **subplots_kw: Any,
) -> tuple[plt.Figure, Any]:
    """Create a styled matplotlib figure at its literal display width and ``200 dpi``.

    Parameters
    ----------
    width_in : float, optional
        Figure width [in] (e.g. ``7.0`` for full width, ``3.4`` for half width).
        Default is ``7.0``.
    height_in : float, optional
        Figure height [in]. If ``None``, defaults to ``2.8 * nrows`` (or ``3.1``
        for a single row).
    ncols : int, optional
        Number of subplot columns. Default is 1.
    nrows : int, optional
        Number of subplot rows. Default is 1.
    **subplots_kw
        Additional keyword arguments forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure at ``200 dpi``.
    axes : matplotlib.axes.Axes or ndarray of Axes
        Subplot axes returned by :func:`matplotlib.pyplot.subplots`.
    """
    apply_style()
    if "width" in subplots_kw:
        width_in = float(subplots_kw.pop("width"))
    if "height" in subplots_kw:
        height_in = float(subplots_kw.pop("height"))
    if height_in is None:
        height_in = 3.0 if int(nrows) == 1 else 2.65 * int(nrows)
    dpi = int(subplots_kw.pop("dpi", DEFAULT_DPI))
    return plt.subplots(
        nrows=int(nrows),
        ncols=int(ncols),
        figsize=(float(width_in), float(height_in)),
        dpi=dpi,
        **subplots_kw,
    )


def spectrum_grid(ax: plt.Axes) -> None:
    """Enable a subtle light dotted grid suitable for spectral or convergence plots."""
    ax.grid(True, which="major", linestyle=":", color="#cbd5e1", linewidth=0.7, alpha=0.8)


def place_legend_below(ax: plt.Axes, ncol: int | None = None, y_offset: float = -0.22, **kwargs: Any) -> Any:
    """Place a frameless legend centered below ``ax`` so it never covers plotted data."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if ncol is None:
        ncol = min(len(labels), 3)
    leg = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, float(y_offset)),
        ncol=int(ncol),
        frameon=False,
        fontsize=kwargs.pop("fontsize", PT_LEGEND),
        **kwargs,
    )
    return leg


def panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.08) -> Text:
    """Add a bold panel label (e.g. ``'(a)'`` or ``'A'``) at the top-left of ``ax``."""
    return ax.text(
        float(x),
        float(y),
        str(label),
        transform=ax.transAxes,
        fontsize=PT_TITLE,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
    )


def _is_auxiliary_axes(ax: plt.Axes) -> bool:
    """Return True if ``ax`` is a colorbar, inset, or twin axis rather than a primary panel."""
    if ax.get_label() == "<colorbar>" or getattr(ax, "_is_inset", False):
        return True
    if getattr(ax, "_colorbar", None) is not None:
        return True
    # Detect narrow colorbar axes created by fig.colorbar
    pos = ax.get_position()
    if pos.width < 0.04 or pos.height < 0.04:
        return True
    return False


def _legend_covers_data(ax: plt.Axes, leg: Any, renderer: Any) -> bool:
    """Check whether an in-axes legend bounding box overlaps plotted line vertices."""
    if leg is None or not leg.get_visible() or not ax.lines:
        return False
    try:
        leg_bbox = leg.get_window_extent(renderer)
    except Exception:
        return False
    # Check if legend is already placed outside the axes box
    ax_bbox = ax.get_window_extent(renderer)
    if (
        leg_bbox.y1 <= ax_bbox.y0 + 2.0
        or leg_bbox.y0 >= ax_bbox.y1 - 2.0
        or leg_bbox.x1 <= ax_bbox.x0 + 2.0
        or leg_bbox.x0 >= ax_bbox.x1 - 2.0
    ):
        return False

    for line in ax.lines:
        if not line.get_visible():
            continue
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if xdata.size == 0 or ydata.size == 0:
            continue
        pts = np.column_stack([xdata, ydata])
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.size == 0:
            continue
        disp = ax.transData.transform(pts)
        if np.any(leg_bbox.containsPoints(disp)):
            return True
    return False


def finish(
    fig: plt.Figure,
    *,
    max_panels: int = 3,
    auto_legend_below: bool = False,
) -> plt.Figure:
    """Apply journal finishing rules to ``fig`` before saving or displaying.

    Enforces:

    1. **Panel count check**: warns with :class:`UserWarning` if the figure has
       more than ``max_panels`` (default 3) primary panels.
    2. **Minimum font size**: raises every text artist (labels, titles, ticks,
       legends, annotations, colorbars) to at least ``MIN_PT`` (``8.5 pt``).
    3. **Contrast enforcement**: darkens any text colour whose WCAG 2.1 contrast
       ratio against white is below ``4.5:1``.
    4. **White halo on image/overlay annotations**: applies a white stroke halo
       (:func:`matplotlib.patheffects.withStroke`) behind annotation text on
       axes containing images, meshes, or patches so labels remain readable over
       colour maps and material fills.
    5. **Frameless legends & marker spacing**: removes legend frames and thins
       default markers on dense curves (> 25 samples) so they remain distinct.
    6. **Tight layout**: adjusts subplot spacing so labels and legends are not
       clipped.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure to finish.
    max_panels : int, optional
        Maximum recommended number of primary panels per figure. Default is 3.
    auto_legend_below : bool, optional
        If True, automatically move in-axes legends that overlap data curves
        below their panel. Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure ``fig``.
    """
    fig.set_facecolor(PAPER)
    content_axes = [ax for ax in fig.axes if not _is_auxiliary_axes(ax)]
    if len(content_axes) > int(max_panels):
        warnings.warn(
            f"Figure has {len(content_axes)} panels (> {max_panels}); "
            "split into at most 3 panels per figure for readability.",
            UserWarning,
            stacklevel=2,
        )

    halo_effect = [pe.withStroke(linewidth=2.5, foreground=PAPER), pe.Normal()]
    structural_texts: set[int] = set()

    for ax in fig.axes:
        is_aux = _is_auxiliary_axes(ax)
        if not is_aux and ax.axison:
            # Keep top/right spines off on standard plots; frame full-bleed 2D images thinly
            has_2d_map = bool(ax.images) or any(
                c.__class__.__name__ in ("QuadMesh", "PolyCollection")
                for c in ax.collections
            )
            if has_2d_map:
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor(INK)
                    sp.set_linewidth(0.8)
            else:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_edgecolor(INK)
                ax.spines["left"].set_edgecolor(INK)

        # Space out cycled markers on dense curves so they do not form a solid bar
        for line in ax.lines:
            mk = line.get_marker()
            ls = line.get_linestyle()
            if (
                mk not in ("", "none", "None", None)
                and ls not in ("", "none", "None", None)
                and line.get_markevery() is None
            ):
                n_pts = len(line.get_xdata())
                if n_pts > 25:
                    line.set_markevery(max(1, n_pts // 10))

        for t_obj in (
            ax.title,
            ax._left_title,
            ax._right_title,
            ax.xaxis.label,
            ax.yaxis.label,
            ax.xaxis.get_offset_text(),
            ax.yaxis.get_offset_text(),
        ):
            structural_texts.add(id(t_obj))

        for axis, axis_name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            tick_labels = axis.get_ticklabels(which="both")
            for tl in tick_labels:
                structural_texts.add(id(tl))
                fs = max(float(tl.get_fontsize()), MIN_PT)
                tl.set_fontsize(fs)
                tl.set_color(darken_for_contrast(tl.get_color(), MIN_CONTRAST_RATIO, PAPER))
            if tick_labels:
                ax.tick_params(
                    axis=axis_name,
                    which="both",
                    labelsize=max(float(tick_labels[0].get_fontsize()), MIN_PT),
                )

        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(False)
            if leg.get_title() is not None:
                structural_texts.add(id(leg.get_title()))
            for lt in leg.get_texts():
                structural_texts.add(id(lt))

    for leg in fig.legends:
        leg.set_frame_on(False)
        if leg.get_title() is not None:
            structural_texts.add(id(leg.get_title()))
        for lt in leg.get_texts():
            structural_texts.add(id(lt))

    for t in fig.findobj(Text):
        txt = t.get_text()
        if not txt and id(t) in structural_texts:
            continue
        if float(t.get_fontsize()) < MIN_PT:
            t.set_fontsize(MIN_PT)
        t.set_color(darken_for_contrast(t.get_color(), MIN_CONTRAST_RATIO, PAPER))

        # Apply white halo to in-axes annotations on images/maps/overlays
        if id(t) not in structural_texts and t.axes is not None and txt.strip():
            ax_t = t.axes
            has_bg_art = bool(ax_t.images or ax_t.collections or ax_t.patches or ax_t.lines)
            if has_bg_art and not t.get_path_effects():
                t.set_path_effects(halo_effect)

    if auto_legend_below:
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax in content_axes:
                leg = ax.get_legend()
                if leg is not None and _legend_covers_data(ax, leg, renderer):
                    place_legend_below(ax)
                    for lt in ax.get_legend().get_texts():
                        lt.set_fontsize(max(float(lt.get_fontsize()), MIN_PT))
                        lt.set_color(darken_for_contrast(lt.get_color(), MIN_CONTRAST_RATIO, PAPER))
        except Exception:
            pass

    if not fig.get_constrained_layout() and not getattr(fig, "_wakis_custom_layout", False):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
        except Exception:
            pass

    return fig


def save(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    close: bool = False,
    auto_legend_below: bool = True,
) -> Path:
    """Run :func:`finish` on ``fig`` and save it as a ``200 dpi`` PNG file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Destination file path (if no suffix is given, ``.png`` is appended).
    dpi : int, optional
        Output resolution [dots/in]. Default is ``200``.
    close : bool, optional
        Close ``fig`` after saving. Default is False.
    auto_legend_below : bool, optional
        Forwarded to :func:`finish`. Default is True.

    Returns
    -------
    Path
        Resolved path of the written image file.
    """
    finish(fig, auto_legend_below=auto_legend_below)
    out = Path(path)
    if not out.suffix:
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), facecolor=PAPER, edgecolor=PAPER, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out
