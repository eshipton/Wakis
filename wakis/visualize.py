# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""
Headless-friendly visualization helpers shared by the tutorials.

The interactive plotting methods of `GridFIT3D` and `SolverFIT3D`
(``plot_solids``, ``plot2D``, ``plot3D``, ``inspect``; see `wakis.plotting`)
are built around a desktop window and sliders. This module adds the static,
reproducible pictures a tutorial or a paper needs, rendered off screen on a
VM, in Colab or in CI:

* `setup_rendering` -- off-screen PyVista (xvfb/OSMesa) and the static
  notebook backend, so images are stored in the notebook;
* `plot_scene` -- STL solids plus labelled translucent boxes (TF/SF box,
  monitors, CPML shell, source plane);
* `plot_material_slice` -- the per-edge permittivity seen by the solver on a
  slice (staircase versus conformal averaging), with the grid lines and the
  true surface;
* `plot_field_slice`, `field_slice` -- DFT-monitor or snapshot slices with
  the boxes overlaid;
* `plot_isosurfaces` -- 3D isosurfaces of a field magnitude;
* `SliceRecorder` and `save_animation` -- a time-domain movie (GIF) of one
  field component on a plane, recorded as an `emsolve` monitor.

All lengths are in metres; the plotting functions take a ``length_unit``
(default micrometres) for the axis labels.
"""

import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

__all__ = [
    "setup_rendering",
    "box_bounds",
    "cpml_inner_bounds",
    "plot_scene",
    "plot_material_slice",
    "edge_permittivity",
    "field_slice",
    "plot_field_slice",
    "plot_isosurfaces",
    "SliceRecorder",
    "save_animation",
    "save_plotter",
]

LENGTH_UNITS = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
_UNIT_LABEL = {"m": "m", "mm": "mm", "um": "µm", "nm": "nm"}
_AXES = "xyz"

# Colours of the standard tutorial boxes (colour-blind safe, Okabe-Ito)
BOX_STYLES = {
    "tfsf": dict(color="#E69F00", label="TF/SF box"),
    "flux": dict(color="#0072B2", label="flux box"),
    "cpml": dict(color="#999999", label="CPML"),
    "source": dict(color="#D55E00", label="source plane"),
    "monitor": dict(color="#009E73", label="monitor"),
}


def _in_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:
        return False


def _start_xvfb(display=":99", screen="1920x1080x24"):
    """Start an X virtual frame buffer and point DISPLAY at it."""
    import subprocess
    import time

    lock = f"/tmp/.X{display.lstrip(':')}-lock"
    if not os.path.exists(lock):
        subprocess.Popen(
            ["Xvfb", display, "-screen", "0", screen, "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
    os.environ["DISPLAY"] = display
    return f"Xvfb on {display}"


def setup_rendering(notebook_backend="static", force_off_screen=None):
    """
    Configure PyVista for headless, reproducible rendering.

    On Linux without a display (a VM, Colab, CI) an X virtual frame buffer is
    started when ``Xvfb`` is installed; otherwise PyVista must be built with
    OSMesa/EGL (``vtk-osmesa`` wheels), which needs no server. Plotters are
    then created off screen. In a notebook the ``static`` backend embeds PNG
    images in the cell outputs, so the executed notebook shows them without
    a running kernel.

    Parameters
    ----------
    notebook_backend : str, optional
        PyVista Jupyter backend used inside notebooks (default 'static').
    force_off_screen : bool, optional
        Override the automatic choice (True on headless Linux, in notebooks
        and when the environment variable ``WAKIS_OFF_SCREEN`` is set).

    Returns
    -------
    str
        A short description of the configuration chosen.
    """
    notes = []
    headless = sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
    if headless and shutil.which("Xvfb"):
        notes.append(_start_xvfb())
    elif headless:
        notes.append("no display: VTK falls back to OSMesa/EGL if available")
    off = force_off_screen
    if off is None:
        off = headless or _in_notebook() or bool(os.environ.get("WAKIS_OFF_SCREEN"))
    pv.OFF_SCREEN = bool(off)
    notes.append(f"off_screen={pv.OFF_SCREEN}")
    if _in_notebook():
        try:
            pv.set_jupyter_backend(notebook_backend)
            notes.append(f"jupyter backend={notebook_backend}")
        except Exception as err:  # pragma: no cover
            notes.append(f"jupyter backend unchanged ({err})")
    try:
        import matplotlib

        if headless and not _in_notebook():
            matplotlib.use("Agg")
    except Exception:  # pragma: no cover
        pass
    return ", ".join(notes)


def box_bounds(center, size):
    """(xmin, xmax, ymin, ymax, zmin, zmax) of a box given centre and size [m]."""
    c = np.asarray(center, float)
    s = np.asarray(size, float)
    return tuple(v for a in range(3) for v in (c[a] - s[a] / 2, c[a] + s[a] / 2))


def cpml_inner_bounds(solver):
    """
    Bounds of the region inside the CPML layers of a solver [m].

    Faces that are not CPML keep the domain bound.
    """
    nodes = (solver.x, solver.y, solver.z)
    n = solver.n_pml
    out = []
    for a in range(3):
        lo = nodes[a][n] if str(solver.bc_low[a]).lower() == "cpml" else nodes[a][0]
        hi = (
            nodes[a][-1 - n]
            if str(solver.bc_high[a]).lower() == "cpml"
            else nodes[a][-1]
        )
        out += [lo, hi]
    return tuple(out)


def _split_spec(spec):
    """(bounds as a flat 6-tuple, style or None) from a box specification.

    Accepts flat bounds (xmin, xmax, ...), nested ((xmin, xmax), ...) as
    returned by `TFSFSource.box_bounds`, or (bounds, style).
    """
    try:
        flat = np.asarray(spec, dtype=float).ravel()
        if flat.size == 6:
            return tuple(flat), None
    except (TypeError, ValueError):
        pass
    bounds, style = spec
    return tuple(np.asarray(bounds, dtype=float).ravel()), style


def plot_scene(
    grid,
    boxes=None,
    planes=None,
    cpml=None,
    solids=True,
    solid_color="#56B4E9",
    clip_normal="-y",
    window_size=(1000, 800),
    length_unit="um",
    show_grid=False,
    camera=None,
    off_screen=None,
):
    """
    3D view of the simulation set-up: solids, boxes and planes.

    Parameters
    ----------
    grid : GridFIT3D
        Grid with the STL solids (``grid.stl_solids``).
    boxes : dict, optional
        ``{name: bounds or (bounds, style)}`` drawn as translucent boxes with
        an edge outline and a label. ``style`` is a key of `BOX_STYLES` or a
        dict(color=..., label=...). Names matching a `BOX_STYLES` key use that
        style by default.
    planes : dict, optional
        ``{name: (center, normal_axis, size)}`` drawn as opaque-ish rectangles
        (e.g. the source entry plane).
    cpml : tuple of 2 bounds, optional
        (outer, inner) bounds of the absorbing shell, drawn as a wireframe.
    solids : bool, optional
        Draw the STL solids (default True).
    clip_normal : str or None, optional
        Clip the solids (e.g. '-y' shows the inside of the sphere); None
        draws them whole.
    show_grid : bool, optional
        Add the grid lines on the outer domain faces.
    camera : list or str, optional
        PyVista camera position.
    off_screen : bool, optional
        Default True (render to an image with `save_plotter`); False opens
        an interactive window with ``pl.show()``.

    Returns
    -------
    pyvista.Plotter
        Call ``save_plotter(pl, path)`` or ``pl.show()``.
    """
    scale = 1.0 / LENGTH_UNITS[length_unit]
    off = True if off_screen is None else off_screen
    pl = pv.Plotter(off_screen=off, window_size=list(window_size))
    pl.set_background("white")
    outline = pv.Box(bounds=np.array(grid.grid.bounds) * scale)
    pl.add_mesh(outline, style="wireframe", color="black", line_width=1)
    if show_grid:
        pl.add_mesh(
            grid.grid.scale(scale, inplace=False).extract_surface(),
            style="wireframe",
            color="grey",
            opacity=0.15,
        )
    if solids:
        for key in grid.stl_solids:
            surf = grid.read_stl(key).scale(scale, inplace=False)
            if clip_normal:
                surf = surf.clip(normal=clip_normal, origin=surf.center)
            pl.add_mesh(surf, color=solid_color, smooth_shading=True, specular=0.4)
    labels, label_pts, label_cols = [], [], []
    if cpml is not None:
        outer, inner = cpml
        style = BOX_STYLES["cpml"]
        for b in (outer, inner):
            pl.add_mesh(
                pv.Box(bounds=np.array(b) * scale),
                style="wireframe",
                color=style["color"],
                line_width=2,
            )
        pts = np.array(outer).reshape(3, 2)
        label_pts.append([pts[0, 1], pts[1, 0], pts[2, 1]])
        labels.append(style["label"])
        label_cols.append(style["color"])
    for name, spec in (boxes or {}).items():
        bounds, style = _split_spec(spec)
        if style is None:
            style = BOX_STYLES.get(name, dict(color="#CC79A7", label=name))
        elif isinstance(style, str):
            style = BOX_STYLES[style]
        style = {"label": name, **style}
        box = pv.Box(bounds=np.array(bounds, float) * scale)
        pl.add_mesh(box, color=style["color"], opacity=0.12)
        pl.add_mesh(box, style="wireframe", color=style["color"], line_width=3)
        b = np.array(bounds, float).reshape(3, 2) * scale
        label_pts.append([b[0, 1], b[1, 0], b[2, 1]])
        labels.append(style["label"])
        label_cols.append(style["color"])
    for name, (center, axis, size) in (planes or {}).items():
        style = BOX_STYLES.get(name, BOX_STYLES["source"])
        a = _AXES.index(axis)
        normal = np.zeros(3)
        normal[a] = 1
        u, v = [i for i in range(3) if i != a]
        rect = pv.Plane(
            center=np.array(center) * scale,
            direction=normal,
            i_size=size[0] * scale,
            j_size=size[1] * scale,
        )
        pl.add_mesh(rect, color=style["color"], opacity=0.5)
        label_pts.append(np.array(center) * scale)
        labels.append(style.get("label", name))
        label_cols.append(style["color"])
    if labels:
        pl.add_legend(
            [[text, col] for text, col in zip(labels, label_cols)],
            bcolor="white",
            border=True,
            size=(0.22, 0.04 + 0.04 * len(labels)),
            loc="upper right",
            face="rectangle",
        )
    pl.add_axes(xlabel="x", ylabel="y", zlabel="z")
    pl.show_bounds(
        xtitle=f"x [{length_unit}]",
        ytitle=f"y [{length_unit}]",
        ztitle=f"z [{length_unit}]",
        color="black",
        font_size=12,
    )
    if camera is not None:
        pl.camera_position = camera
    else:
        pl.camera_position = "xz"
        pl.camera.azimuth = 35
        pl.camera.elevation = 25
        pl.camera.zoom(1.1)
    return pl


def save_plotter(pl, path, transparent=False, close=True):
    """Render a PyVista plotter to a PNG file and return the image array."""
    img = pl.screenshot(path, transparent_background=transparent, return_img=True)
    if close:
        pl.close()
    return img


def _nodes(obj):
    return (np.asarray(obj.x), np.asarray(obj.y), np.asarray(obj.z))


def edge_permittivity(solver, component, axis="z", position=0.0):
    """
    Relative permittivity seen by one E-edge component on a grid plane.

    ``eps_r = 1 / (eps_0 ieps)`` of each edge, where ``ieps`` is the diagonal
    inverse permittivity used by the update (staircase: the cell mask value;
    ``conformal='all'``: the volume-averaged value of the dual cell; with
    ``conformal='tensor'`` the diagonal part only, the off-diagonal coupling
    is held separately by the tensor operator).

    Parameters
    ----------
    solver : SolverFIT3D
    component : {'x', 'y', 'z'}
        Edge direction.
    axis : {'x', 'y', 'z'}
        Normal of the slice plane.
    position : float
        Coordinate of the plane [m]; the nearest grid node is used.

    Returns
    -------
    eps_r : ndarray
        2D array indexed (u, v), the two in-plane axes in xyz order.
    coords : tuple of 2 ndarray
        Edge-centre coordinates along u and v [m].
    """
    from scipy.constants import epsilon_0

    nodes = _nodes(solver)
    a = _AXES.index(axis)
    k = int(np.argmin(np.abs(nodes[a] - position)))
    ieps = solver.ieps
    if getattr(ieps, "on_gpu", False):
        ieps = ieps.copy()
        ieps.from_gpu()
    mat = np.asarray(ieps.to_matrix(component))
    sl = [slice(None)] * 3
    sl[a] = k
    vals = mat[tuple(sl)]
    with np.errstate(divide="ignore"):
        eps_r = 1.0 / (epsilon_0 * vals)
    c = _AXES.index(component)
    u, v = [i for i in range(3) if i != a]
    coords = []
    for d in (u, v):
        x = nodes[d]
        if d == c:  # edge midpoints along its own direction
            x = 0.5 * (x[:-1] + x[1:])
            x = np.append(x, np.nan)  # ghost edge beyond the last node
        coords.append(x)
    return eps_r, tuple(coords)


def plot_material_slice(
    solver,
    axis="z",
    position=0.0,
    window=None,
    surface=None,
    ax=None,
    vmin=1.0,
    vmax=None,
    cmap="viridis",
    length_unit="um",
    title=None,
    line_width=None,
    colorbar=True,
):
    """
    Draw the in-plane E edges of a slice coloured by their permittivity.

    Each primary-grid edge in the plane is drawn as a coloured segment, so a
    staircased surface shows edges switching from vacuum to the solid, while
    conformal averaging shows intermediate values along the true surface.
    Grid lines are drawn in light grey and the true surface section in black.

    Parameters
    ----------
    solver : SolverFIT3D
    axis : {'x', 'y', 'z'}
        Normal of the slice.
    position : float
        Plane coordinate [m] (nearest grid node).
    window : tuple (umin, umax, vmin, vmax), optional
        Zoom window [m].
    surface : pyvista.PolyData or str, optional
        Surface whose section is overlaid (an STL key of the grid or a mesh).
    ax : matplotlib Axes, optional
    vmin, vmax : float, optional
        Colour range of eps_r (default 1 to the maximum).
    line_width : float, optional
        Edge segment width in points (default scales with the zoom).

    Returns
    -------
    matplotlib.collections.LineCollection
        The coloured edges (use it for a shared colour bar).
    """
    scale = 1.0 / LENGTH_UNITS[length_unit]
    unit = _UNIT_LABEL[length_unit]
    a = _AXES.index(axis)
    u, v = [i for i in range(3) if i != a]
    nodes = _nodes(solver)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    segs, vals = [], []
    for comp in (u, v):
        eps_r, (cu, cv) = edge_permittivity(solver, _AXES[comp], axis, position)
        hu, hv = (
            (np.diff(nodes[u]).min() / 2, 0)
            if comp == u
            else (0, np.diff(nodes[v]).min() / 2)
        )
        UU, VV = np.meshgrid(cu, cv, indexing="ij")
        ok = np.isfinite(UU) & np.isfinite(VV) & np.isfinite(eps_r)
        if window is not None:
            ok &= (UU >= window[0]) & (UU <= window[1])
            ok &= (VV >= window[2]) & (VV <= window[3])
        p0 = np.stack([UU[ok] - hu, VV[ok] - hv], -1) * scale
        p1 = np.stack([UU[ok] + hu, VV[ok] + hv], -1) * scale
        segs.append(np.stack([p0, p1], 1))
        vals.append(eps_r[ok])
    segs, vals = np.concatenate(segs), np.concatenate(vals)
    if vmax is None:
        vmax = max(np.nanmax(vals), vmin + 1e-9)
    if line_width is None:
        span = (window[1] - window[0]) if window else (nodes[u][-1] - nodes[u][0])
        n_cells = span / np.diff(nodes[u]).min()
        line_width = float(np.clip(70 / n_cells, 0.5, 6))
    # Grid lines
    lo_u, hi_u = (window[0], window[1]) if window else (nodes[u][0], nodes[u][-1])
    lo_v, hi_v = (window[2], window[3]) if window else (nodes[v][0], nodes[v][-1])
    for x in nodes[u][(nodes[u] >= lo_u) & (nodes[u] <= hi_u)]:
        ax.plot(
            [x * scale] * 2,
            [lo_v * scale, hi_v * scale],
            color="0.85",
            lw=0.5,
            zorder=0,
        )
    for y in nodes[v][(nodes[v] >= lo_v) & (nodes[v] <= hi_v)]:
        ax.plot(
            [lo_u * scale, hi_u * scale],
            [y * scale] * 2,
            color="0.85",
            lw=0.5,
            zorder=0,
        )
    lc = LineCollection(segs, cmap=cmap, linewidths=line_width, capstyle="butt")
    lc.set_array(vals)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)
    if surface is not None:
        if isinstance(surface, str):
            surface = solver.grid.read_stl(surface)
        k = int(np.argmin(np.abs(nodes[a] - position)))
        normal = np.zeros(3)
        normal[a] = 1
        origin = np.zeros(3)
        origin[a] = nodes[a][k]
        cut = surface.slice(normal=normal, origin=origin)
        if cut.n_points:
            lines = cut.lines.reshape(-1, 3)[:, 1:]
            pts = cut.points[:, [u, v]] * scale
            ax.add_collection(
                LineCollection(pts[lines], colors="k", linewidths=1.2, zorder=3)
            )
    ax.set_xlim(lo_u * scale, hi_u * scale)
    ax.set_ylim(lo_v * scale, hi_v * scale)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{_AXES[u]} [{unit}]")
    ax.set_ylabel(f"{_AXES[v]} [{unit}]")
    if title:
        ax.set_title(title)
    if colorbar:
        plt.colorbar(lc, ax=ax, label=r"edge $\varepsilon_r$", shrink=0.8)
    return lc


def field_slice(field, axis="y", position=0.0, component="x", solver=None):
    """
    2D slice of a field component on a grid plane.

    Parameters
    ----------
    field : DFTMonitor, Field or ndarray
        * `DFTMonitor`: ``component`` like 'Ex'; returns all monitored
          frequencies, shape (n_freq, nu, nv);
        * `wakis.Field` (e.g. ``solver.E``): ``component`` in 'x', 'y', 'z'
          (needs ``solver`` for the coordinates);
        * ndarray (nx, ny, nz) with ``solver`` for coordinates.
    axis : {'x', 'y', 'z'}
        Normal of the plane.
    position : float
        Plane coordinate [m] (nearest sample).

    Returns
    -------
    values : ndarray
        (nu, nv) or (n_freq, nu, nv).
    coords : tuple of 2 ndarray
        Sample coordinates along the two in-plane axes [m].
    """
    a = _AXES.index(axis)
    u, v = [i for i in range(3) if i != a]
    if hasattr(field, "get_field"):
        values, coords = field.get_field(component)
        k = int(np.argmin(np.abs(np.asarray(coords[a]) - position)))
        sl = [slice(None)] * 4
        sl[a + 1] = k
        return values[tuple(sl)], (np.asarray(coords[u]), np.asarray(coords[v]))
    if solver is None:
        raise ValueError("solver is needed for the coordinates of a Field/array.")
    nodes = _nodes(solver)
    if hasattr(field, "to_matrix"):
        arr = field.to_matrix(component)
        arr = arr.get() if hasattr(arr, "get") else np.asarray(arr)
        c = _AXES.index(component[-1])
    else:
        arr, c = np.asarray(field), None
    coords = []
    for d in range(3):
        x = nodes[d]
        if c is not None and d == c:
            x = np.append(0.5 * (x[:-1] + x[1:]), x[-1])
        coords.append(x[: arr.shape[d]])
    k = int(np.argmin(np.abs(coords[a] - position)))
    sl = [slice(None)] * 3
    sl[a] = k
    return arr[tuple(sl)], (coords[u], coords[v])


def _draw_boxes(ax, boxes, axis, scale):
    a = _AXES.index(axis)
    u, v = [i for i in range(3) if i != a]
    for name, spec in (boxes or {}).items():
        bounds, style = _split_spec(spec)
        if style is None:
            style = BOX_STYLES.get(name, dict(color="#CC79A7", label=name))
        elif isinstance(style, str):
            style = BOX_STYLES[style]
        b = np.array(bounds, float).reshape(3, 2) * scale
        ax.add_patch(
            Rectangle(
                (b[u, 0], b[v, 0]),
                b[u, 1] - b[u, 0],
                b[v, 1] - b[v, 0],
                fill=False,
                ec=style["color"],
                lw=1.5,
                ls="--",
                label=style.get("label", name),
            )
        )


def plot_field_slice(
    values,
    coords,
    kind="real",
    ax=None,
    boxes=None,
    axis="y",
    surface=None,
    cmap=None,
    vmax=None,
    length_unit="um",
    title=None,
    colorbar_label=None,
    log_floor=None,
):
    """
    Plot a 2D field slice with labelled axes, the set-up boxes and the
    scatterer outline.

    Parameters
    ----------
    values : ndarray (nu, nv), complex or real
    coords : tuple of 2 ndarray
        Coordinates along u and v [m] (as returned by `field_slice`).
    kind : {'real', 'abs', 'abs2', 'log'}
        Re(F) with a diverging colormap (symmetric limits), |F|, |F|^2 or
        log10|F| with a perceptual colormap.
    boxes : dict, optional
        As in `plot_scene`; outlines on the plane.
    axis : {'x', 'y', 'z'}
        Normal of the plane (to place the boxes and the surface section).
    surface : (center, radius) or pyvista.PolyData, optional
        Scatterer outline: a circle for a sphere, or a mesh section.
    log_floor : float, optional
        Lower limit of the log scale relative to the maximum (e.g. 1e-4).

    Returns
    -------
    matplotlib image (QuadMesh)
    """
    scale = 1.0 / LENGTH_UNITS[length_unit]
    unit = _UNIT_LABEL[length_unit]
    a = _AXES.index(axis)
    u, v = [i for i in range(3) if i != a]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    values = np.asarray(values)
    if kind == "real":
        data = np.real(values)
        vmax = vmax or np.nanmax(np.abs(data))
        kw = dict(cmap=cmap or "RdBu_r", vmin=-vmax, vmax=vmax)
    elif kind == "abs":
        data = np.abs(values)
        kw = dict(cmap=cmap or "magma", vmin=0, vmax=vmax or np.nanmax(data))
    elif kind == "abs2":
        data = np.abs(values) ** 2
        kw = dict(cmap=cmap or "magma", vmin=0, vmax=vmax or np.nanmax(data))
    elif kind == "log":
        data = np.log10(np.abs(values) + 1e-300)
        top = np.nanmax(data) if vmax is None else np.log10(vmax)
        floor = top + np.log10(log_floor) if log_floor else np.nanmin(data)
        kw = dict(cmap=cmap or "magma", vmin=floor, vmax=top)
    else:
        raise ValueError("kind must be 'real', 'abs', 'abs2' or 'log'.")
    cu, cv = (np.asarray(c) * scale for c in coords)
    im = ax.pcolormesh(cu, cv, data.T, shading="nearest", rasterized=True, **kw)
    _draw_boxes(ax, boxes, axis, scale)
    if surface is not None:
        if isinstance(surface, tuple):
            center, radius = surface
            ax.add_patch(
                plt.Circle(
                    (center[u] * scale, center[v] * scale),
                    radius * scale,
                    fill=False,
                    ec="k",
                    lw=1,
                )
            )
        else:
            normal = np.zeros(3)
            normal[a] = 1
            cut = surface.slice(normal=normal, origin=surface.center)
            if cut.n_points:
                lines = cut.lines.reshape(-1, 3)[:, 1:]
                pts = cut.points[:, [u, v]] * scale
                ax.add_collection(LineCollection(pts[lines], colors="k", linewidths=1))
    ax.set_aspect("equal")
    ax.set_xlabel(f"{_AXES[u]} [{unit}]")
    ax.set_ylabel(f"{_AXES[v]} [{unit}]")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label=colorbar_label or "", shrink=0.85)
    return im


def plot_isosurfaces(
    values,
    coords,
    levels=(0.3, 0.6),
    surfaces=None,
    cmap="magma",
    length_unit="um",
    window_size=(900, 700),
    off_screen=None,
    clip_normal=None,
    scalar_label="|E| / max",
):
    """
    3D isosurfaces of a non-negative scalar field (e.g. |E| of a DFT monitor).

    Parameters
    ----------
    values : ndarray (nx, ny, nz)
        Real scalar samples, normalized internally to the maximum.
    coords : sequence of 3 ndarray
        Sample coordinates [m].
    levels : sequence of float
        Iso-levels as fractions of the maximum.
    surfaces : list of pyvista.PolyData, optional
        Solids drawn as translucent context.

    Returns
    -------
    pyvista.Plotter
    """
    scale = 1.0 / LENGTH_UNITS[length_unit]
    data = np.asarray(values, float)
    data = data / np.nanmax(data)
    X, Y, Z = np.meshgrid(*(np.asarray(c) * scale for c in coords), indexing="ij")
    sg = pv.StructuredGrid(X, Y, Z)
    sg.point_data[scalar_label] = data.ravel(order="F")
    off = True if off_screen is None else off_screen
    pl = pv.Plotter(off_screen=off, window_size=list(window_size))
    pl.set_background("white")
    iso = sg.contour(isosurfaces=list(levels), scalars=scalar_label)
    if clip_normal and iso.n_points:
        iso = iso.clip(normal=clip_normal, origin=(0, 0, 0))
    if iso.n_points:
        pl.add_mesh(
            iso,
            scalars=scalar_label,
            cmap=cmap,
            clim=[0, 1],
            opacity=0.6,
            smooth_shading=True,
            scalar_bar_args=dict(title=scalar_label, color="black"),
        )
    for s in surfaces or []:
        pl.add_mesh(s.scale(scale, inplace=False), color="#56B4E9", opacity=0.2)
    pl.add_mesh(sg.outline(), color="black")
    pl.add_axes()
    pl.camera_position = "xz"
    pl.camera.azimuth = 35
    pl.camera.elevation = 25
    return pl


class SliceRecorder:
    """
    Record one field component on a plane every few time steps.

    Pass it in the ``monitors`` list of `SolverFIT3D.emsolve`; it stores
    host copies of the slice (cheap: one plane) for `save_animation`.

    Parameters
    ----------
    component : str
        'Ex', 'Hy', ... (E, H or J).
    axis : {'x', 'y', 'z'}
        Normal of the plane.
    position : float
        Plane coordinate [m].
    every : int
        Record every `every` steps.
    """

    def __init__(self, component="Ex", axis="y", position=0.0, every=10):
        self.component = component
        self.axis = axis
        self.position = position
        self.every = int(every)
        self.frames, self.times = [], []
        self._n = 0
        self.coords = None

    def update(self, solver, t):
        if self._n % self.every == 0:
            field = getattr(solver, self.component[0])
            values, coords = field_slice(
                field, self.axis, self.position, self.component[1], solver
            )
            self.frames.append(np.array(values, copy=True))
            self.times.append(t + solver.dt)
            self.coords = coords
        self._n += 1


def save_animation(
    recorder,
    path,
    boxes=None,
    surface=None,
    fps=15,
    vmax=None,
    length_unit="um",
    time_unit=1e-15,
    time_label="fs",
    dpi=80,
    title=None,
    max_frames=120,
):
    """
    Write a GIF of a `SliceRecorder` with fixed symmetric colour limits.

    Parameters
    ----------
    recorder : SliceRecorder
    path : str
        Output .gif path.
    vmax : float, optional
        Colour limit (default: 0.5 x the largest |value| of all frames, so
        the weaker scattered waves stay visible).
    max_frames : int
        Frames are sub-sampled to at most this number (file size).

    Returns
    -------
    str
        The path written.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = recorder.frames
    idx = np.unique(
        np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)
    )
    if vmax is None:
        vmax = 0.5 * max(np.abs(f).max() for f in frames) or 1.0
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = plot_field_slice(
        frames[idx[0]],
        recorder.coords,
        kind="real",
        ax=ax,
        boxes=boxes,
        axis=recorder.axis,
        surface=surface,
        vmax=vmax,
        length_unit=length_unit,
        colorbar_label=f"{recorder.component} [arb.]",
    )
    label = ax.set_title("")
    fig.tight_layout()

    def draw(i):
        im.set_array(np.real(frames[i]).T.ravel())
        t = recorder.times[i] / time_unit
        label.set_text(f"{title + '  ' if title else ''}t = {t:.1f} {time_label}")
        return (im, label)

    anim = FuncAnimation(fig, draw, frames=idx, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path
