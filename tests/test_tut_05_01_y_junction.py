# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Validation, non-square graded mesh comparison, setup schematic, and mutation tests
for Tutorial 5.1: `tutorials/05_guided_waves/05-01_y_junction/`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TUTORIAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "tutorials"
    / "05_guided_waves"
    / "05-01_y_junction"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

hlp = _load_module("_tut_05_01_helpers", TUTORIAL_DIR / "_helpers.py")
yj = _load_module("_tut_05_01_y_junction", TUTORIAL_DIR / "y_junction.py")


def _assert_task_resolution_rules(meta: dict, full_mode: bool = False) -> None:
    """Enforce project grid resolution rules on a Y-junction task metadata dictionary."""
    min_cpw = 10
    assert int(meta["cpw"]) >= min_cpw, (
        f"Rule 1 violated: cpw={meta['cpw']} < {min_cpw}"
    )
    assert float(meta["cells_across_feature"]) >= 4.0 - 1e-6, (
        f"Rule 2 violated: cells_across_feature={meta['cells_across_feature']} < 4.0"
    )
    assert float(meta["dy_tip_nm"]) <= float(meta["min_feature_nm"]) / 4.0 + 1e-6, (
        f"Rule 2 violated: dy_tip_nm={meta['dy_tip_nm']} > min_feature_nm/4"
    )
    if full_mode and meta["name"] in (
        "kottke_fine",
        "kottke_reference",
        "kottke_nonuniform",
    ):
        assert int(meta["cpw"]) >= 20, (
            f"Rule 1 (--full): production task {meta['name']} must have >= 20 cpw"
        )


def test_y_junction_geometry_and_nonuniform_mesh_builder():
    """Verify Y-junction mirror symmetry and graded non-square Cartesian mesh properties."""
    structures, ports = yj.build_y_junction()
    assert len(structures) == 1
    assert [p.name for p in ports] == ["in", "out1", "out2"]

    slab = structures[0]
    y_test = np.linspace(-1.2e-6, 1.2e-6, 25)
    x_test = np.full_like(y_test, 2.5e-6)
    z_test = np.zeros_like(y_test)
    h_test = np.full_like(y_test, 20e-9)
    f_pos = slab.fill(x_test, y_test, z_test, h_test, h_test, h_test, n_sub=8)
    f_neg = slab.fill(x_test, -y_test, z_test, h_test, h_test, h_test, n_sub=8)
    np.testing.assert_allclose(f_pos, f_neg, atol=1e-12)

    circuit_u = yj.PhotonicCircuit(
        yj.WL_MIN,
        yj.WL_MAX,
        structures=structures,
        ports=ports,
        n_core=yj.N_SI,
        n_clad=yj.N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=28,
        domain=yj.DOMAIN_2D,
        n_pml=10,
    )
    circuit_nu = hlp.NonUniformYJunctionCircuit(
        yj.WL_MIN,
        yj.WL_MAX,
        structures=structures,
        ports=ports,
        n_core=yj.N_SI,
        n_clad=yj.N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=28,
        domain=yj.DOMAIN_2D,
        n_pml=10,
    )
    m_u = hlp.measure_y_junction_mesh(circuit_u.grid)
    m_nu = hlp.measure_y_junction_mesh(circuit_nu.grid)

    # Upgraded non-square graded mesh cuts cell count by > 30% while increasing tip-gap resolution
    assert m_nu["n_cells"] < 0.70 * m_u["n_cells"]
    assert m_nu["cells_across_feature"] >= 11.5
    assert m_nu["cells_across_feature"] > m_u["cells_across_feature"]
    assert m_nu["aspect_ratio_max_core"] > 1.35
    assert m_nu["dy_clad_max_nm"] > 45.0

    # Adjacent cell-size grading ratio <= 1.30 on both axes
    dx = np.diff(circuit_nu.grid.x)
    dy = np.diff(circuit_nu.grid.y)
    assert float(np.max(np.maximum(dx[1:] / dx[:-1], dx[:-1] / dx[1:]))) <= 1.30
    assert float(np.max(np.maximum(dy[1:] / dy[:-1], dy[:-1] / dy[1:]))) <= 1.30


def test_committed_full_study_data_and_figures():
    """Verify committed data/task_0..7.{json,npz} and figures/00..14 satisfy accuracy and grid rules."""
    full_tasks = yj.get_task_table(quick=False)
    assert len(full_tasks) == 8

    metas = {}
    datas = {}
    for t_cfg in full_tasks:
        idx = int(t_cfg["index"])
        json_p = TUTORIAL_DIR / "data" / f"task_{idx}.json"
        npz_p = TUTORIAL_DIR / "data" / f"task_{idx}.npz"
        assert json_p.is_file(), f"Missing {json_p}"
        assert npz_p.is_file(), f"Missing {npz_p}"
        meta = json.loads(json_p.read_text())
        assert meta["task_index"] == idx
        assert meta["name"] == t_cfg["name"]
        _assert_task_resolution_rules(meta, full_mode=True)
        metas[meta["name"]] = meta
        datas[meta["name"]] = dict(np.load(npz_p, allow_pickle=False))

    # 1. Primary uniform baseline (`kottke_fine`) symmetry, passivity, and flux conservation
    d_fine = datas["kottke_fine"]
    wl_nm = d_fine["wavelengths"] * 1e9
    mask = (wl_nm >= 1505.0) & (wl_nm <= 1595.0)
    t1 = np.abs(d_fine["S21"][mask]) ** 2
    t2 = np.abs(d_fine["S31"][mask]) ** 2
    r = np.abs(d_fine["S11"][mask]) ** 2
    assert float(np.max(np.abs(t1 - t2))) < 1e-6
    assert float(np.max(t1 + t2 + r)) <= 1.001
    assert 0.90 < float(np.mean(t1 + t2)) < 0.93
    np.testing.assert_allclose(d_fine["flux_total_box"][mask], 1.0, atol=0.02)

    # 2. Kottke subpixel smoothing beats staircasing at 20 and 28 cpw against 56-cpw reference
    def _t_1550(d):
        i0 = int(np.argmin(np.abs(d["wavelengths"] - yj.WL_REF)))
        return float(np.abs(d["S21"][i0]) ** 2 + np.abs(d["S31"][i0]) ** 2)

    t_ref = _t_1550(datas["kottke_reference"])
    err_k20 = abs(_t_1550(datas["kottke_medium"]) - t_ref)
    err_s20 = abs(_t_1550(datas["staircase_medium"]) - t_ref)
    err_k28 = abs(_t_1550(datas["kottke_fine"]) - t_ref)
    err_s28 = abs(_t_1550(datas["staircase_fine"]) - t_ref)
    assert err_k20 < 0.5 * err_s20
    assert err_k28 < 0.3 * err_s28

    # 3. Upgraded non-square graded mesh (`kottke_nonuniform`, task_7) matches reference at -34% cells
    d_nu = datas["kottke_nonuniform"]
    m_nu = metas["kottke_nonuniform"]
    m_fine = metas["kottke_fine"]
    t_nu = _t_1550(d_nu)
    assert m_nu["n_cells"] < 0.70 * m_fine["n_cells"]
    assert abs(t_nu - t_ref) < 3.0e-4
    el_nu_db = -10.0 * np.log10(t_nu)
    el_ref_db = -10.0 * np.log10(t_ref)
    assert abs(el_nu_db - el_ref_db) < 0.002

    # 4. All 17 figures + setup table exist (including 00a/00b setup figures and mesh upgrade comparison)
    expected_figs = [
        "00a_setup_geometry.png",
        "00b_setup_mesh_and_excitation.png",
        "01_y_junction_layout.png",
        "02_eim_vertical_stack.png",
        "03_junction_width_profile.png",
        "04_permittivity_mesh_zoom.png",
        "04b_mesh_upgrade_comparison.png",
        "05_port_mode_profiles.png",
        "06_port_neff_dispersion.png",
        "07_field_intensity_1550.png",
        "08_field_real_ey_1550.png",
        "09_pulse_propagation.gif",
        "10_sparameter_spectra_db.png",
        "11_excess_loss_and_energy_budget.png",
        "12_kottke_vs_staircase_convergence.png",
        "13_resolution_spectra_comparison.png",
        "14_port_time_trace.png",
    ]
    for fname in expected_figs:
        fpath = TUTORIAL_DIR / "figures" / fname
        assert fpath.is_file() and fpath.stat().st_size > 1000, (
            f"Missing or empty figure {fpath}"
        )
    table_md = TUTORIAL_DIR / "figures" / "00_setup_table.md"
    assert table_md.is_file() and table_md.stat().st_size > 200, (
        f"Missing or empty setup table {table_md}"
    )
    gif_size = (TUTORIAL_DIR / "figures" / "09_pulse_propagation.gif").stat().st_size
    assert gif_size < 5 * 1024 * 1024, f"GIF size {gif_size} >= 5 MB"


def test_mutation_checks_provably_fail():
    """Mutation checks: broken mirror symmetry, staircase error, or coarse tip-gap cell size MUST fail."""
    d_fine = dict(np.load(TUTORIAL_DIR / "data" / "task_4.npz", allow_pickle=False))
    d_stair = dict(np.load(TUTORIAL_DIR / "data" / "task_5.npz", allow_pickle=False))
    d_ref = dict(np.load(TUTORIAL_DIR / "data" / "task_6.npz", allow_pickle=False))

    # Mutation 1: Asymmetry in S21 vs S31 MUST fail the mirror-symmetry audit (< 1e-6)
    t1 = np.abs(d_fine["S21"]) ** 2
    t2_mut = np.abs(1.02 * d_fine["S31"]) ** 2
    with pytest.raises(AssertionError):
        assert float(np.max(np.abs(t1 - t2_mut))) < 1e-6

    # Mutation 2: Staircase at 28 cpw MUST fail the Kottke / graded-mesh accuracy bound (< 3.5e-4)
    i0 = int(np.argmin(np.abs(d_ref["wavelengths"] - yj.WL_REF)))
    t_ref = float(np.abs(d_ref["S21"][i0]) ** 2 + np.abs(d_ref["S31"][i0]) ** 2)
    t_stair = float(np.abs(d_stair["S21"][i0]) ** 2 + np.abs(d_stair["S31"][i0]) ** 2)
    with pytest.raises(AssertionError):
        assert abs(t_stair - t_ref) < 3.5e-4

    # Mutation 3: Grid resolution rules MUST reject cpw < 10 or < 4 cells across the 200 nm tip gap
    with pytest.raises(AssertionError, match="Rule 1"):
        _assert_task_resolution_rules(
            {
                "name": "bad_cpw",
                "cpw": 8,
                "cells_across_feature": 5.0,
                "dy_tip_nm": 40.0,
                "min_feature_nm": 200.0,
            }
        )
    with pytest.raises(AssertionError, match="Rule 2"):
        _assert_task_resolution_rules(
            {
                "name": "bad_tip_gap",
                "cpw": 14,
                "cells_across_feature": 3.2,
                "dy_tip_nm": 62.5,
                "min_feature_nm": 200.0,
            }
        )
