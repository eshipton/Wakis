# Copyright 2026 Wakis contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for wakis.figstyle and wakis.schematic (Figure A, Figure B, setup_table, and mutation checks)."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from wakis import DFTMonitor, FluxMonitor, GridFIT3D, SolverFIT3D, TFSFSource, figstyle, schematic
from wakis.lumped import LumpedPort
from wakis.sources import CurrentSource, GaussianSource
from wakis.subgrid import QuadMesh, SubgridFIT2D
from wakis.templates.shapes import Box, assign_materials
from wakis.unstructured import UnstructuredFIT2D, disk_mesh


def _collect_axes_texts(ax) -> list[str]:
    """Collect all visible text strings (including title, labels, legend, and inset child_axes) on an Axes."""
    out = [t.get_text() for t in ax.texts if t.get_text()]
    out.append(ax.get_title())
    out.append(ax.get_xlabel())
    out.append(ax.get_ylabel())
    leg = ax.get_legend()
    if leg is not None:
        out.extend(t.get_text() for t in leg.get_texts())
    for child in getattr(ax, "child_axes", ()):
        out.extend(_collect_axes_texts(child))
    return [s for s in out if s]


def _assert_no_overlapping_texts(fig) -> None:
    """Assert that no two visible text bounding boxes intersect anywhere on the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                texts.append(t)
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if t is not None and t.get_visible() and t.get_text().strip():
                texts.append(t)
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            for t in leg.get_texts():
                if t.get_visible() and t.get_text().strip():
                    texts.append(t)

    boxes = [(t.get_text().replace("\n", " "), t.get_window_extent(renderer)) for t in texts]
    for i in range(len(boxes)):
        s1, b1 = boxes[i]
        if b1.width <= 0 or b1.height <= 0:
            continue
        for j in range(i + 1, len(boxes)):
            s2, b2 = boxes[j]
            if b2.width <= 0 or b2.height <= 0:
                continue
            assert not b1.overlaps(b2), f"Text bounding boxes overlap: {s1!r} ({b1}) vs {s2!r} ({b2})"


def test_figstyle_apply_finish_and_save(tmp_path):
    """Verify rcParams, min font enforcement, WCAG contrast darkening, halo, >3 panel warning, and 200 dpi save."""
    figstyle.apply_style()
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["axes.grid"] is False
    assert float(matplotlib.rcParams["axes.labelsize"]) == pytest.approx(11.0)
    assert float(matplotlib.rcParams["xtick.labelsize"]) == pytest.approx(10.0)

    # Context manager restores state and each series gets distinct color, linestyle, and marker
    with figstyle.apply_style():
        fig, ax = figstyle.new_figure(width_in=figstyle.HALF_WIDTH_IN, height_in=2.4)
        x = np.linspace(0, 1, 8)
        for idx in range(4):
            st = figstyle.series_style(idx)
            ax.plot(x, x + idx, label=f"Series {idx}", **st)
        # Add a tiny low-contrast text on an image to verify finish() fixes both font size and contrast
        ax.imshow(np.zeros((6, 6)), extent=(0, 1, 0, 5), cmap=figstyle.CMAP_DIVERGING, vmin=-1, vmax=1)
        tiny_txt = ax.text(0.5, 2.5, "Tiny low-contrast note", fontsize=5.0, color="#DDDDDD")
        figstyle.finish(fig)

        assert tiny_txt.get_fontsize() >= figstyle.MIN_PT
        assert figstyle.contrast_ratio(tiny_txt.get_color(), figstyle.PAPER) >= 4.5
        assert any(isinstance(eff, pe.Stroke) for eff in tiny_txt.get_path_effects())

        out_png = tmp_path / "test_style.png"
        figstyle.save(fig, out_png)
        plt.close(fig)

    assert out_png.exists() and out_png.stat().st_size > 1000
    with Image.open(out_png) as im:
        dpi_x, _ = im.info.get("dpi", (200, 200))
        assert round(dpi_x) == figstyle.DEFAULT_DPI

    # > 3 panels emits a UserWarning
    fig4, _ = figstyle.new_figure(ncols=2, nrows=2)
    with pytest.warns(UserWarning, match="panels"):
        figstyle.finish(fig4, max_panels=3)
    plt.close(fig4)


def test_schematic_2d_subgrid_and_unstructured():
    """Verify plot_schematic and draw_mesh on 2D quadtree subgrid and unstructured triangle meshes."""
    qmesh = QuadMesh(
        np.linspace(0.0, 0.04, 9),
        np.linspace(0.0, 0.03, 7),
        ratio=2,
        max_level=1,
        refine=lambda lev, x0, x1, y0, y1: (x1 > 0.01) & (x0 < 0.025) & (y1 > 0.01) & (y0 < 0.02),
    )
    sub_solver = SubgridFIT2D(qmesh)

    fig, ax = schematic.plot_schematic(
        sub_solver,
        show_mesh=True,
        freq=5e9,
        probes=[{"pos": (0.02, 0.015, 0.0), "label": "Center probe"}],
        title="2D Subgrid Cavity",
    )
    texts = "\n".join(_collect_axes_texts(ax))
    assert "PEC" in texts
    assert "Center probe" in texts
    assert len(schematic.in_axes_texts(ax)) <= 6
    _assert_no_overlapping_texts(fig)
    plt.close(fig)

    # Unstructured 2D SimplexMesh
    smesh = disk_mesh(radii=[0.03], h=0.015)
    u_solver = UnstructuredFIT2D(smesh, bc="pmc")
    fig2, ax2 = schematic.plot_schematic(u_solver, show_mesh=True, freq=3e9)
    texts2 = "\n".join(_collect_axes_texts(ax2))
    assert "PMC" in texts2
    assert len(schematic.in_axes_texts(ax2)) <= 6
    _assert_no_overlapping_texts(fig2)
    plt.close(fig2)


def test_schematic_3d_fit_and_mutation():
    """Verify Figure A layout rules (<=6 in-axes labels, no overlap, legend below, 1D axis hidden, mutation)."""
    grid = GridFIT3D(0.0, 0.01, 0.0, 0.01, 0.0, 0.20, 1, 1, 40, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=6,
        verbose=0,
    )
    slab = Box(center=(0.005, 0.005, 0.105), size=(0.01, 0.01, 0.05), material=4.0)
    assign_materials(solver, [slab], frequency=3e9)

    pulse = GaussianSource(frequency=3e9, fwidth=1.5e9)
    src = CurrentSource(
        src_time=pulse,
        component="Ex",
        center=(0.005, 0.005, 0.05),
        size=(0.01, 0.01, 0.0),
    )
    tfsf = TFSFSource(
        src_time=pulse,
        center=(0.005, 0.005, 0.105),
        size=(0.01, 0.01, 0.09),
        direction="+z",
        polarization="x",
    )
    port = LumpedPort(
        src_time=pulse,
        component="x",
        center=(0.005, 0.005, 0.075),
        size=(0.01, 0.0, 0.0),
        impedance=50.0,
    )
    mon_flux = FluxMonitor(
        frequencies=np.array([3e9]),
        center=(0.005, 0.005, 0.16),
        size=(0.01, 0.01, 0.0),
    )
    mon_dft = DFTMonitor(
        frequencies=np.array([3e9]),
        components=("Ex",),
        center=(0.005, 0.005, 0.04),
        size=(0.01, 0.01, 0.0),
    )

    fig, ax = schematic.plot_schematic(
        solver,
        material_names={"region_0": "Silica slab"},
        sources=[src, tfsf, port],
        monitors=[mon_flux, mon_dft],
        probes=[{"pos": (0.005, 0.005, 0.16), "label": "Probe A"}],
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)

    # 1. Label count inside the axes <= 6
    inside_texts = schematic.in_axes_texts(ax)
    assert len(inside_texts) <= 6, f"Expected <= 6 in-axes texts, got {len(inside_texts)}: {[t.get_text() for t in inside_texts]}"
    geom_inside = [
        t
        for t in ax.texts
        if t.get_visible()
        and t.get_text().strip()
        and ax_bbox.contains(
            0.5 * (t.get_window_extent(renderer).x0 + t.get_window_extent(renderer).x1),
            0.5 * (t.get_window_extent(renderer).y0 + t.get_window_extent(renderer).y1),
        )
    ]
    assert len(geom_inside) <= 6

    # 2. Boundary face tags ("CPML", "periodic") are placed OUTSIDE the domain box
    bc_face_texts = [t for t in ax.texts if t.get_text().strip() in ("CPML", "periodic")]
    assert len(bc_face_texts) == 4
    for bt in bc_face_texts:
        bb = bt.get_window_extent(renderer)
        cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
        assert not ax_bbox.contains(cx, cy), f"Boundary tag {bt.get_text()!r} must be outside the domain box"

    # 3. No two text bounding boxes intersect
    _assert_no_overlapping_texts(fig)

    # 4. Legend is below the axes and contains material + source/monitor details
    leg = ax.get_legend()
    assert leg is not None
    leg_bbox = leg.get_window_extent(renderer)
    assert leg_bbox.y1 <= ax_bbox.y0, f"Legend top ({leg_bbox.y1:.1f}) must be below axes bottom ({ax_bbox.y0:.1f})"
    leg_str = "\n".join(t.get_text() for t in leg.get_texts())
    assert "Silica slab, ε = 4.00" in leg_str
    assert "CPML" in leg_str and "6 cells" in leg_str
    assert "CurrentSource" in leg_str
    assert "TFSF" in leg_str
    assert "LumpedPort" in leg_str
    assert "FluxMonitor" in leg_str
    assert "DFTMonitor" in leg_str
    assert "Probe A" in leg_str

    # 5. Meaningless transverse axis of 1D domain is hidden; thin space + upright unit used
    assert len(ax.get_yticks()) == 0
    assert ax.get_ylabel() == ""
    texts_before = "\n".join(_collect_axes_texts(ax))
    assert "200\u2009mm" in texts_before

    # 6. All text font sizes >= 9 pt
    for t in ax.texts + leg.get_texts():
        if t.get_text().strip():
            assert t.get_fontsize() >= 9.0
    plt.close(fig)

    # 7. Setup table before mutation lists every boundary face
    table_before = schematic.setup_table(
        solver,
        material_names={"region_0": "Silica slab"},
        sources=[src, tfsf, port],
        monitors=[mon_flux, mon_dft],
        probes=[{"pos": (0.005, 0.005, 0.16), "label": "Probe A"}],
    )
    for face in ("-x face", "+x face", "-y face", "+y face", "-z face", "+z face"):
        assert f"| Boundary | {face} |" in table_before
    assert "| Boundary | -z face | CPML" in table_before
    assert "| Boundary | +z face | CPML" in table_before

    # 8. Mutation check: mutate solver z boundary from 'cpml' to 'pec'/'pmc' and verify both tag & table change
    solver.bc_low[2] = "pec"
    solver.bc_high[2] = "pmc"
    fig_mut, ax_mut = schematic.plot_schematic(solver, sources=[src], monitors=[mon_flux])
    texts_after = "\n".join(_collect_axes_texts(ax_mut))
    _assert_no_overlapping_texts(fig_mut)
    plt.close(fig_mut)

    table_after = schematic.setup_table(solver, sources=[src], monitors=[mon_flux])

    # Pre-mutation expectations must fail on mutated outputs
    with pytest.raises(AssertionError):
        assert "CPML" in texts_after
    with pytest.raises(AssertionError):
        assert "| Boundary | -z face | CPML" in table_after

    assert "PEC" in texts_after
    assert "PMC" in texts_after
    assert "| Boundary | -z face | PEC |" in table_after
    assert "| Boundary | +z face | PMC |" in table_after

    # Also test overlay_geometry in 1D and 2D modes
    fig_ov, (ax_1d, ax_2d) = figstyle.new_figure(ncols=2, nrows=1)
    solver.bc_low[2] = "cpml"
    solver.bc_high[2] = "cpml"
    schematic.overlay_geometry(
        ax_1d,
        solver,
        orientation="1d",
        material_names={"region_0": "Silica slab"},
        fill_materials=True,
        show_sources=True,
        show_monitors=True,
        sources=[src],
        monitors=[mon_flux],
    )
    schematic.overlay_geometry(
        ax_2d,
        solver,
        orientation="2d",
        material_names={"region_0": "Silica slab"},
    )
    assert any("Silica slab" in t or "4" in t for t in _collect_axes_texts(ax_1d))
    assert any("CPML" in t for t in _collect_axes_texts(ax_2d))
    plt.close(fig_ov)


def test_mesh_and_excitation_and_save_setup_figures(tmp_path):
    """Verify Figure B (plot_mesh_and_excitation), setup_table, and save_setup_figures."""
    grid = GridFIT3D(0.0, 0.01, 0.0, 0.01, 0.0, 0.25, 1, 1, 250, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=10,
        verbose=0,
    )
    slab = Box(center=(0.005, 0.005, 0.115), size=(0.01, 0.01, 0.03), material=4.0)
    assign_materials(solver, [slab], frequency=5e9)

    pulse = GaussianSource(frequency=5e9, fwidth=3e9)
    src = CurrentSource(
        src_time=pulse,
        component="Ex",
        center=(0.005, 0.005, 0.05),
        size=(0.01, 0.01, 0.0),
    )
    mon_dft = DFTMonitor(
        frequencies=np.linspace(2e9, 8e9, 25),
        components=("Ex", "Hy"),
        center=(0.005, 0.005, 0.19),
        size=(0.01, 0.01, 0.0),
    )
    mon_dft.label = "Transmission DFT"

    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        solver,
        material_names={"region_0": "SiO₂ slab"},
        sources=[src],
        monitors=[mon_dft],
    )
    assert len(axes_b) == 3
    all_b_texts = "\n".join(
        "\n".join(_collect_axes_texts(a)) for a in fig_b.axes if a.get_visible()
    )
    assert "Δz" in all_b_texts
    assert "cells/λ" in all_b_texts
    assert "extent" in all_b_texts
    assert "−3 dB" in all_b_texts
    assert "1/e" in all_b_texts
    assert "s(t)" in all_b_texts
    _assert_no_overlapping_texts(fig_b)
    plt.close(fig_b)

    res = schematic.save_setup_figures(
        solver,
        tmp_path,
        material_names={"region_0": "SiO₂ slab"},
        sources=[src],
        monitors=[mon_dft],
        variations=[("Reference run", "Vacuum (ε_r = 1)"), ("Slab run", "SiO₂ slab (ε_r = 4.00)")],
    )
    path_a = res["geometry_figure"]
    path_b = res["mesh_figure"]
    md_table = res["table"]
    assert path_a.name == "00a_setup_geometry.png" and path_a.exists()
    assert path_b.name == "00b_setup_mesh_and_excitation.png" and path_b.exists()
    assert (tmp_path / "00_setup_table.md").exists()
    assert "SiO₂ slab" in md_table
    assert "Transmission DFT" in md_table
    assert "| Variation | Reference run |" in md_table
