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
    "SIM_LABEL",
    "THEORY_LABEL",
    "MESH_STYLES",
    "plot_sim_vs_theory",
    "setup_rendering",
    "box_bounds",
    "cpml_inner_bounds",
    "plot_scene",
    "plot_material_slice",
    "edge_permittivity",
    "field_slice",
    "plot_field_slice",
    "plot_material_plane",
    "plot_field_plane",
    "plot_isosurfaces",
    "SliceRecorder",
    "save_animation",
    "save_plotter",
]

LENGTH_UNITS = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
_UNIT_LABEL = {"m": "m", "mm": "mm", "um": "µm", "nm": "nm"}
_AXES = "xyz"

# Shared figure conventions (tutorials and studies): simulation = markers
# (no line), theory/reference = lines (no markers), with these legend words
SIM_LABEL = "Wakis (FIT)"
THEORY_LABEL = "exact series"
# Mesh / material-model colours and markers (Okabe-Ito, colour-blind safe)
MESH_STYLES = {
    "staircase": dict(color="#D55E00", marker="o", label="Cartesian, staircase"),
    "all": dict(color="#0072B2", marker="s", label="Cartesian, conformal ('all')"),
    "tensor": dict(color="#009E73", marker="^", label="Cartesian, tensor averaging"),
    "triangles": dict(
        color="#CC79A7", marker="D", label="triangles (unstructured FIT)"
    ),
}


def plot_sim_vs_theory(ax, x_sim, y_sim, x_theory, y_theory, sim_label=None,
                       theory_label=THEORY_LABEL, color="#D55E00", marker="o",
                       ms=3.5, theory_color="k"):  # fmt: skip
    """
    Plot a simulated curve (markers only) over a theory curve (line only)
    with unambiguous legend entries; returns the two artists.
    """
    th = ax.plot(
        x_theory, y_theory, "-", color=theory_color, lw=1.1, label=theory_label
    )
    sim = ax.plot(x_sim, y_sim, ls="none", marker=marker, ms=ms, color=color,
                  label=sim_label or SIM_LABEL)  # fmt: skip
    return sim[0], th[0]


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


def _start_xvfb(screen="1920x1080x24"):
    """
    Start a private X virtual frame buffer on a free display and point
    DISPLAY at it. Displays already in use (lock files, possibly owned by
    other users or `xvfb-run`) are skipped.
    """
    import subprocess
    import time

    for n in range(99, 200):
        if os.path.exists(f"/tmp/.X{n}-lock") or os.path.exists(f"/tmp/.X11-unix/X{n}"):
            continue
        proc = subprocess.Popen(
            ["Xvfb", f":{n}", "-screen", "0", screen, "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
        if proc.poll() is None:  # still running: the server is up
            os.environ["DISPLAY"] = f":{n}"
            return f"Xvfb on :{n}"
    return "Xvfb could not be started"


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
    azimuth=35.0,
    elevation=25.0,
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
        ``{name: (center, normal_axis, (size_u, size_v))}`` drawn as
        translucent rectangles (e.g. the source entry plane); the sizes are
        along the two other axes in xyz order.
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
        PyVista camera position (overrides azimuth/elevation).
    azimuth, elevation : float, optional
        Camera rotation from the default x-z view [deg].
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
        # size = extents along the two in-plane axes, in xyz order
        b = np.repeat(np.array(center, float), 2).reshape(3, 2)
        b[u] += np.array([-0.5, 0.5]) * size[0]
        b[v] += np.array([-0.5, 0.5]) * size[1]
        rect = pv.Box(bounds=tuple(b.ravel() * scale))
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
        pl.camera.azimuth = azimuth
        pl.camera.elevation = elevation
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
    components="in-plane",
):
    """
    Draw the E edges of a slice coloured by their permittivity.

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
    components : {'in-plane', 'normal', 'all'}
        Which edges to draw: the two in-plane components as segments, the
        component normal to the slice as squares on the nodes (e.g. E_z of a
        2D TM simulation in the z = 0 plane), or both.

    Returns
    -------
    matplotlib.collections.Collection
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
    in_plane = (u, v) if components in ("in-plane", "all") else ()
    for comp in in_plane:
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
    normal = None
    if components in ("normal", "all"):
        eps_n, (cu, cv) = edge_permittivity(solver, _AXES[a], axis, position)
        UU, VV = np.meshgrid(cu, cv, indexing="ij")
        ok = np.isfinite(eps_n)
        if window is not None:
            ok &= (UU >= window[0]) & (UU <= window[1])
            ok &= (VV >= window[2]) & (VV <= window[3])
        normal = (UU[ok] * scale, VV[ok] * scale, eps_n[ok])
    if in_plane:
        segs, vals = np.concatenate(segs), np.concatenate(vals)
    else:
        segs, vals = np.zeros((0, 2, 2)), normal[2]
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
    if in_plane:
        lc.set_array(vals)
        lc.set_clim(vmin, vmax)
        ax.add_collection(lc)
    if normal is not None:
        lc = ax.scatter(
            normal[0], normal[1], c=normal[2], cmap=cmap, vmin=vmin, vmax=vmax,
            marker="s", s=(1.6 * line_width) ** 2, edgecolors="none", zorder=2,
        )  # fmt: skip
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


def _cell_bounds(centers):
    """Cell boundaries around sorted sample centres (half spacing at the ends)."""
    c = np.asarray(centers, float)
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def plot_material_plane(
    solver,
    component="z",
    axis="z",
    position=0.0,
    window=None,
    surfaces=None,
    vmin=1.0,
    vmax=None,
    cmap="viridis",
    length_unit="um",
    window_size=(800, 760),
    title=None,
    off_screen=None,
):
    """
    PyVista top view of the permittivity seen by one E component on a grid
    plane: each edge is drawn as the dual cell around it, with the primary
    grid lines and the true surface section on top.

    For a 2D TM simulation (E_z) use ``component='z', axis='z'``: every node
    of the plane carries one E_z edge. For TE use 'x' or 'y'.

    Parameters
    ----------
    solver : SolverFIT3D
    component : {'x', 'y', 'z'}
        E-edge direction.
    axis, position : str, float
        Normal and position of the plane [m].
    window : tuple (umin, umax, vmin, vmax), optional
        Zoom window [m].
    surfaces : list of pyvista.PolyData, optional
        Solids whose section is overlaid.

    Returns
    -------
    pyvista.Plotter
    """
    scale = 1.0 / LENGTH_UNITS[length_unit]
    a = _AXES.index(axis)
    u, v = [i for i in range(3) if i != a]
    eps, (cu, cv) = edge_permittivity(solver, component, axis, position)
    ok_u, ok_v = np.isfinite(cu), np.isfinite(cv)
    cu, cv, eps = cu[ok_u], cv[ok_v], eps[np.ix_(ok_u, ok_v)]
    if window is not None:
        su = (cu >= window[0]) & (cu <= window[1])
        sv = (cv >= window[2]) & (cv <= window[3])
        cu, cv, eps = cu[su], cv[sv], eps[np.ix_(su, sv)]
    bu, bv = _cell_bounds(cu) * scale, _cell_bounds(cv) * scale
    UU, VV = np.meshgrid(bu, bv, indexing="ij")
    pts = [None, None, None]
    pts[u], pts[v] = UU, VV
    pts[a] = np.full_like(UU, position * scale)
    cells = pv.StructuredGrid(*pts)
    cells.cell_data["eps_r"] = eps.ravel(order="F")
    off = True if off_screen is None else off_screen
    pl = pv.Plotter(off_screen=off, window_size=list(window_size))
    pl.set_background("white")
    pl.add_mesh(
        cells,
        scalars="eps_r",
        cmap=cmap,
        clim=[vmin, vmax or float(np.nanmax(eps))],
        scalar_bar_args=dict(title=f"eps_r of E{component} edges", color="black"),
    )
    # Primary grid lines inside the window
    nodes = _nodes(solver)
    nu, nv = nodes[u], nodes[v]
    lo_u, hi_u = bu[0] / scale, bu[-1] / scale
    lo_v, hi_v = bv[0] / scale, bv[-1] / scale
    nu = nu[(nu >= lo_u) & (nu <= hi_u)]
    nv = nv[(nv >= lo_v) & (nv <= hi_v)]
    lift = np.zeros(3)
    lift[a] = 1e-3 * (hi_u - lo_u) * scale
    segs = []
    for x in nu:
        for y0, y1 in ((lo_v, hi_v),):
            p0, p1 = np.zeros(3), np.zeros(3)
            p0[u], p0[v], p1[u], p1[v] = x, y0, x, y1
            segs += [p0, p1]
    for y in nv:
        p0, p1 = np.zeros(3), np.zeros(3)
        p0[u], p0[v], p1[u], p1[v] = lo_u, y, hi_u, y
        segs += [p0, p1]
    if segs:
        segs = np.array(segs) * scale
        segs[:, a] = position * scale
        pl.add_mesh(pv.line_segments_from_points(segs + lift), color="white",
                    line_width=1, opacity=0.6)  # fmt: skip
    normal = np.zeros(3)
    normal[a] = 1.0
    origin = np.zeros(3)
    origin[a] = position * scale
    for surf in surfaces or []:
        cut = surf.scale(scale, inplace=False).slice(normal=normal, origin=origin)
        if cut.n_points:
            pl.add_mesh(
                cut.translate(2 * lift, inplace=False), color="red", line_width=4
            )
    if title:
        pl.add_text(title, font_size=14, color="black")
    pl.view_vector(normal, viewup=np.eye(3)[v])
    pl.reset_camera(bounds=cells.bounds)
    return pl


def plot_field_plane(
    values,
    coords,
    axis="z",
    position=0.0,
    kind="real",
    surfaces=None,
    boxes=None,
    cmap=None,
    vmax=None,
    warp=0.0,
    length_unit="um",
    window_size=(900, 760),
    scalar_label="field",
    off_screen=None,
    camera=None,
):
    """
    PyVista rendering of a 2D field slice as a coloured plane (optionally
    warped into a relief), with the solids' sections and the set-up boxes.

    Parameters
    ----------
    values : ndarray (nu, nv)
        Complex or real samples (as returned by `field_slice`).
    coords : tuple of 2 ndarray
        Coordinates along the in-plane axes [m].
    axis, position : str, float
        Normal and position of the plane [m].
    kind : {'real', 'abs'}
        Re(F) with a diverging map, or |F| with a perceptual map.
    surfaces : list of pyvista.PolyData, optional
        Solids; their section with the plane is drawn as a black line.
    boxes : dict, optional
        As in `plot_scene`, drawn as outlines on the plane.
    warp : float, optional
        Relief height of the maximum value as a fraction of the plane
        width (0: flat).
    camera : list or str, optional
        PyVista camera position (default: oblique for a relief, top view
        otherwise).

    Returns
    -------
    pyvista.Plotter
    """
    scale = 1.0 / LENGTH_UNITS[length_unit]
    a = _AXES.index(axis)
    u, v = [i for i in range(3) if i != a]
    data = np.real(values) if kind == "real" else np.abs(values)
    vmax = vmax or float(np.nanmax(np.abs(data)))
    cu, cv = (np.asarray(c) * scale for c in coords)
    UU, VV = np.meshgrid(cu, cv, indexing="ij")
    pts = [None, None, None]
    pts[u], pts[v] = UU, VV
    pts[a] = np.full_like(UU, position * scale)
    sg = pv.StructuredGrid(*pts)
    sg.point_data[scalar_label] = data.ravel(order="F")
    width = max(cu.max() - cu.min(), cv.max() - cv.min())
    if warp:
        normal = np.zeros(3)
        normal[a] = 1.0
        sg.point_data["_w"] = np.clip(data.ravel(order="F"), -vmax, vmax)
        sg = sg.warp_by_scalar("_w", factor=warp * width / vmax, normal=normal)
    off = True if off_screen is None else off_screen
    pl = pv.Plotter(off_screen=off, window_size=list(window_size))
    pl.set_background("white")
    clim = [-vmax, vmax] if kind == "real" else [0, vmax]
    pl.add_mesh(
        sg,
        scalars=scalar_label,
        cmap=cmap or ("RdBu_r" if kind == "real" else "magma"),
        clim=clim,
        smooth_shading=True,
        scalar_bar_args=dict(title=scalar_label, color="black", vertical=True),
    )
    origin = np.zeros(3)
    origin[a] = position * scale
    normal = np.zeros(3)
    normal[a] = 1.0
    lift = normal * (1e-3 * width + (warp * width if warp else 0))
    for surf in surfaces or []:
        cut = surf.scale(scale, inplace=False).slice(normal=normal, origin=origin)
        if cut.n_points:
            pl.add_mesh(cut.translate(lift, inplace=False), color="black", line_width=3)
    for name, spec in (boxes or {}).items():
        bounds, style = _split_spec(spec)
        if style is None:
            style = BOX_STYLES.get(name, dict(color="#CC79A7"))
        elif isinstance(style, str):
            style = BOX_STYLES[style]
        b = np.array(bounds, float).reshape(3, 2) * scale
        corners = []
        for cu_, cv_ in ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)):
            p = origin.copy()
            p[u], p[v] = b[u, cu_], b[v, cv_]
            corners.append(p + lift)
        pl.add_mesh(pv.lines_from_points(np.array(corners)), color=style["color"],
                    line_width=3)  # fmt: skip
    if camera is not None:
        pl.camera_position = camera
    elif warp:  # oblique view from above the plane
        pl.view_vector(normal - 1.3 * np.eye(3)[v] - 0.5 * np.eye(3)[u], viewup=normal)
    else:
        pl.view_vector(normal, viewup=np.eye(3)[v])
        pl.camera.zoom(1.25)
    pl.add_axes()
    return pl


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
