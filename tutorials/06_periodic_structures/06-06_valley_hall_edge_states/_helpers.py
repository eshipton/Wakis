# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Physics, 2D FIT simulation runners, and plotting helpers for Tutorial 6.6.

Covers:
1. Honeycomb photonic-crystal representation on a Cartesian FIT grid via its
   centred-rectangular supercell (:math:`a \\times \\sqrt{3}a`), Bloch boundary
   conditions, and band unfolding back onto the primitive hexagonal Brillouin zone.
2. Massless Dirac cone at :math:`K` (:math:`\\delta = 0`) and valley-Hall bandgap
   opening (:math:`\\delta \\neq 0`) with inversion-broken sublattice rod radii
   :math:`r_A = r_0(1 + \\delta), r_B = r_0(1 - \\delta)`.
3. Gauge-invariant Fukui-Hatsugai-Suzuki Berry curvature :math:`\\Omega_1(\\mathbf{k})`
   and valley Chern sign reversal (:math:`\\delta \\to -\\delta` and :math:`K \\leftrightarrow K'`)
   computed from both 2D PWE and 2D FIT (:class:`~wakis.solverFIT3D.SolverFIT3D`)
   fields and compared against the analytical massive Dirac model.
4. Supercell ribbon band structure :math:`\\omega(k_x)` of a :math:`\\mathrm{VPC}_+ / \\mathrm{VPC}_-`
   domain wall showing valley-momentum-locked topological kink states traversing
   the bulk photonic bandgap.
5. 2D FIT transmission comparison through a sharp :math:`120^\\circ` Z-bend
   waveguide vs a trivial photonic-crystal line-defect waveguide with mode/flux
   monitors, 3-resolution grid convergence with Richardson extrapolation, setup
   schematics (``00a``/``00b``), and an animated wavefront GIF.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0

from wakis import GridFIT3D, SolverFIT3D, figstyle, schematic
from wakis.estimate import estimate, preflight
from wakis.field_monitors import DFTMonitor, FluxMonitor
from wakis.lattices import (
    CircularInclusion,
    HexagonalLattice,
    PWE2D,
    dirac_berry_curvature,
    fill_circular_inclusions,
    fit_berry_curvature_patch,
    fit_supercell_bands_at_k,
)
from wakis.sources import CurrentSource, GaussianSource

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# Default valley-Hall photonic crystal parameters (normalized lattice constant a = 1.0 m)
DEFAULT_A_M: float = 1.0
DEFAULT_R0_OVER_A: float = 0.18
DEFAULT_DELTA: float = 0.12
DEFAULT_EPS_ROD: float = 12.0
DEFAULT_EPS_BG: float = 1.0
DEFAULT_F_MAX_NORM: float = 0.34
# Tuned core rod radius r_def = 0.65 * r_0 = 0.117 a so the trivial PhC waveguide
# hosts a single guided mode spanning the bulk valley bandgap [0.2715, 0.3059].
DEFAULT_DEFECT_RADIUS_OVER_A: float = 0.65 * DEFAULT_R0_OVER_A


def audit_honeycomb_resolution(
    nx_per_a: int,
    ny_per_supercell: int,
    a: float = DEFAULT_A_M,
    r_0: float = DEFAULT_R0_OVER_A,
    delta: float = DEFAULT_DELTA,
    eps_rod: float = DEFAULT_EPS_ROD,
    f_max_norm: float = DEFAULT_F_MAX_NORM,
    min_cpw: float = 20.0,
    min_cells_feature: float = 4.0,
) -> dict[str, float]:
    """Audit grid resolution against wavelength-in-dielectric and smallest-feature rules.

    Verifies:
    1. :math:`\\lambda_{\\min} / \\max(\\Delta x, \\Delta y) \\ge \\mathrm{min\\_cpw}` inside
       the highest-index rod (:math:`n_{\\max} = \\sqrt{\\varepsilon_{\\mathrm{rod}}}`).
    2. :math:`2 r_{\\min} / \\max(\\Delta x, \\Delta y) \\ge \\mathrm{min\\_cells\\_feature}`
       across the smaller sublattice rod diameter :math:`2 r_B = 2 r_0(1 - |\\delta|)`.

    Parameters
    ----------
    nx_per_a : int
        Number of cells along :math:`x` per lattice constant :math:`a`.
    ny_per_supercell : int
        Number of cells along :math:`y` per supercell height :math:`\\sqrt{3}a`.
    a : float, optional
        Lattice constant [m]. Default is ``1.0``.
    r_0 : float, optional
        Mean rod radius [m]. Default is ``0.18``.
    delta : float, optional
        Magnitude of radius asymmetry :math:`\\delta`. Default is ``0.12``.
    eps_rod : float, optional
        Relative permittivity of the rods. Default is ``12.0``.
    f_max_norm : float, optional
        Maximum dimensionless frequency :math:`\\omega a / (2\\pi c)`. Default is ``0.34``.
    min_cpw : float, optional
        Minimum required cells per wavelength in the highest-index medium. Default is ``20.0``.
    min_cells_feature : float, optional
        Minimum required cells across the smallest rod diameter. Default is ``4.0``.

    Returns
    -------
    dict of str to float
        Resolution metrics ``dx``, ``dy``, ``cells_per_wavelength``, and
        ``cells_across_min_diameter``.

    Raises
    ------
    ValueError
        If any resolution criterion is violated.
    """
    if nx_per_a <= 0 or ny_per_supercell <= 0:
        raise ValueError("Grid cell counts must be positive.")
    dx = float(a) / float(nx_per_a)
    dy = float(np.sqrt(3.0) * a) / float(ny_per_supercell)
    h_max = max(dx, dy)

    n_max = float(np.sqrt(eps_rod))
    lambda_min_m = float(a) / (float(f_max_norm) * n_max)
    cpw = lambda_min_m / h_max

    d_min_m = 2.0 * float(r_0) * (1.0 - abs(float(delta)))
    cells_feat = d_min_m / h_max

    if cpw < min_cpw:
        raise ValueError(
            f"Grid under-resolved in dielectric: {cpw:.2f} cells/lambda < required {min_cpw:.1f}."
        )
    if cells_feat < min_cells_feature:
        raise ValueError(
            f"Smallest rod diameter under-resolved: {cells_feat:.2f} cells < required {min_cells_feature:.1f}."
        )

    return {
        "dx": dx,
        "dy": dy,
        "lambda_min_dielectric_m": lambda_min_m,
        "min_rod_diameter_m": d_min_m,
        "cells_per_wavelength": float(cpw),
        "cells_across_min_diameter": float(cells_feat),
    }


def _preflight_fit(
    nx: int,
    ny: int,
    n_steps: int,
    dtype: str = "float64",
    cpml: bool = False,
) -> dict[str, float]:
    """Run Wakis memory and runtime preflight check before a 2D FIT run."""
    est = estimate((nx, ny, 1), steps=n_steps, dtype=dtype, backend="cpu", cpml=cpml)
    preflight(est, {"ram": 2.0e9, "gpu": 0.0})
    return {
        "cells": int(est.cells),
        "ram_mb": float(est.host_bytes / 1e6),
        "estimated_seconds": float(est.seconds),
    }


def generate_setup_artifacts(quick: bool = False) -> dict[str, Any]:
    """Generate 00a/00b setup figures and Markdown setup tables for supercell & bend."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lat = HexagonalLattice(a=DEFAULT_A_M)
    lx_s, ly_s = lat.supercell_size

    # 1. Supercell Bloch setup (28 x 48 cells)
    nx_s, ny_s = 28, 48
    lz_s = lx_s / float(nx_s)
    k_K = lat.high_symmetry_points()["K"]
    grid_s = GridFIT3D(0.0, lx_s, 0.0, ly_s, 0.0, lz_s, nx_s, ny_s, 1, use_mpi=False, verbose=0)
    solver_s = SolverFIT3D(
        grid_s,
        bc_low=["bloch", "bloch", "periodic"],
        bc_high=["bloch", "bloch", "periodic"],
        k_bloch=(float(k_K[0]), float(k_K[1]), 0.0),
        dtype=np.complex128,
        verbose=0,
    )
    r_a = DEFAULT_R0_OVER_A * (1.0 + DEFAULT_DELTA)
    r_b = DEFAULT_R0_OVER_A * (1.0 - DEFAULT_DELTA)
    inc_s = lat.honeycomb_supercell_sites(r_a, r_b, DEFAULT_EPS_ROD)
    fill_circular_inclusions(solver_s, inc_s, eps_bg=DEFAULT_EPS_BG, n_sub=6, periodic_wrap=True)

    f0_s = 0.288 * c_light / DEFAULT_A_M
    df_s = 0.08 * c_light / DEFAULT_A_M
    src_s = CurrentSource(
        GaussianSource(f0_s, df_s),
        component="z",
        center=(0.5 * lx_s, ly_s / 6.0, 0.5 * lz_s),
        size=(0.0, 0.0, lz_s),
    )
    probes_s = [(0.44 * lx_s, 0.15 * ly_s, 0.5 * lz_s), (0.94 * lx_s, 0.65 * ly_s, 0.5 * lz_s)]
    min_diam = 2.0 * r_b

    res_supercell = schematic.save_setup_figures(
        solver_s,
        FIG_DIR,
        prefix="00",
        suffix="_supercell",
        plane="xy",
        structures=inc_s,
        material_names={"background": "Air (eps_r=1)", "region_0": "Si rod A (r_A)", "region_2": "Si rod B (r_B)"},
        sources=[src_s],
        probes=probes_s,
        unit="m",
        freq=f0_s,
        min_feature=min_diam,
        title_geometry="Supercell setup: centred-rectangular honeycomb cell (XY plane)",
        title_mesh="Supercell discretization & excitation spectrum",
    )

    # 2. Sharp 120-degree Z-bend waveguide setup
    nx_cells, ny_cells = 9, 7
    nx_cell_res = 18 if quick else 22
    ny_cell_res = 30 if quick else 36
    nx_b = nx_cells * nx_cell_res
    ny_b = ny_cells * ny_cell_res
    lx_b = nx_cells * lx_s
    ly_b = ny_cells * ly_s
    lz_b = 10.0 * lx_s

    grid_b = GridFIT3D(0.0, lx_b, 0.0, ly_b, 0.0, lz_b, nx_b, ny_b, 1, use_mpi=False, verbose=0)
    solver_b = SolverFIT3D(
        grid_b,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=6,
        dtype=np.float64,
        verbose=0,
    )
    inc_b, path_x, path_y = lat.domain_wall_bend_inclusions(
        nx_cells, ny_cells, DEFAULT_R0_OVER_A, DEFAULT_DELTA, DEFAULT_EPS_ROD, kind="valley_bend"
    )
    fill_circular_inclusions(solver_b, inc_b, eps_bg=DEFAULT_EPS_BG, n_sub=4, periodic_wrap=False)

    y_in = float(path_y[0])
    y_out = float(path_y[-1])
    f_freqs = np.linspace(0.280, 0.290, 11) * c_light / DEFAULT_A_M
    mon_in = FluxMonitor(f_freqs, center=(2.5 * DEFAULT_A_M, y_in, 0.5 * lz_b), size=(0.0, 2.2 * DEFAULT_A_M, lz_b))
    mon_out = FluxMonitor(
        f_freqs, center=((nx_cells - 1.5) * DEFAULT_A_M, y_out, 0.5 * lz_b), size=(0.0, 2.2 * DEFAULT_A_M, lz_b)
    )

    f0_b = 0.285 * c_light / DEFAULT_A_M
    df_b = 0.016 * c_light / DEFAULT_A_M
    src_b = CurrentSource(
        GaussianSource(f0_b, df_b),
        component="z",
        center=(1.5 * DEFAULT_A_M, y_in, 0.5 * lz_b),
        size=(0.0, 1.0 * DEFAULT_A_M, lz_b),
    )
    res_bend = schematic.save_setup_figures(
        solver_b,
        FIG_DIR,
        prefix="00",
        suffix="",
        plane="xy",
        structures=inc_b[:24],  # Representative unit-cell rods for clean legend, plus full eps map in solver
        material_names={"background": "Air (eps_r=1)", "region_0": "High-index rods (eps_r=12)"},
        sources=[src_b],
        monitors=[mon_in, mon_out],
        unit="m",
        freq=f0_b,
        min_feature=min_diam,
        title_geometry="Valley-Hall 120-deg Z-bend waveguide setup (XY plane)",
        title_mesh="Waveguide mesh, spatial wave packet & pulse spectrum",
    )

    # Also copy 00_setup_table.md to tutorial root if needed
    root_table = HERE / "00_setup_table.md"
    root_table.write_text(res_bend["table"] + "\n", encoding="utf-8")

    return {
        "supercell_table_path": str(res_supercell["table_path"].relative_to(HERE)),
        "bend_table_path": str(res_bend["table_path"].relative_to(HERE)),
    }


def run_bulk_and_berry_study(quick: bool = False) -> dict[str, Any]:
    """Compute honeycomb Dirac dispersion, supercell unfolding, gap scaling, and Berry curvature."""
    lat = HexagonalLattice(a=DEFAULT_A_M)
    hs = lat.high_symmetry_points()
    k_K = hs["K"]
    k_Kp = hs["K'"]
    A1, A2 = lat.supercell_vectors

    n_seg = 8 if quick else 14
    n_pw = 5 if quick else 7
    path = lat.bz_path(("Gamma", "M", "K", "Gamma"), n_per_segment=n_seg)

    r0 = DEFAULT_R0_OVER_A
    delta0 = DEFAULT_DELTA
    eps_r = DEFAULT_EPS_ROD

    # Primitive vs centred-rectangular supercell PWE for symmetric (delta=0) and valley-Hall (delta=0.12)
    pwe_sym_prim = PWE2D(lat.a1, lat.a2, lat.honeycomb_primitive_sites(r0, r0, eps_r), eps_bg=1.0, n_pw=n_pw, polarization="TM")
    pwe_val_prim = PWE2D(
        lat.a1,
        lat.a2,
        lat.honeycomb_primitive_sites(r0 * (1.0 + delta0), r0 * (1.0 - delta0), eps_r),
        eps_bg=1.0,
        n_pw=n_pw,
        polarization="TM",
    )
    pwe_val_sup = PWE2D(
        A1,
        A2,
        lat.honeycomb_supercell_sites(r0 * (1.0 + delta0), r0 * (1.0 - delta0), eps_r),
        eps_bg=1.0,
        n_pw=n_pw,
        polarization="TM",
    )

    bands_sym_prim = pwe_sym_prim.solve_path(path.k_points, n_bands=4)["frequencies"]
    bands_val_prim = pwe_val_prim.solve_path(path.k_points, n_bands=4)["frequencies"]
    sup_val_res = pwe_val_sup.solve_path(path.k_points, n_bands=8, unfold_supercell=True, lattice_a=DEFAULT_A_M)

    unfolding_max_error = float(np.max(np.abs(sup_val_res["unfolded_frequencies"] - bands_val_prim)))

    # Dirac cone parameters at K
    fk_sym = pwe_sym_prim.solve_k(k_K, n_bands=2)
    f_dirac = float(0.5 * (fk_sym[0] + fk_sym[1]))
    dirac_degeneracy_split = float(abs(fk_sym[1] - fk_sym[0]))

    dq = 0.015 * (2.0 * np.pi / DEFAULT_A_M)
    fk_dq = pwe_sym_prim.solve_k(k_K + np.array([dq, 0.0]), n_bands=2)
    v_d_over_c = float(0.5 * (fk_dq[1] - fk_dq[0]) * (2.0 * np.pi / DEFAULT_A_M) / dq)

    # Valley-Hall bandgap at delta = 0.12
    fk_val = pwe_val_prim.solve_k(k_K, n_bands=2)
    gap_bot = float(fk_val[0])
    gap_top = float(fk_val[1])
    gap_width = float(gap_top - gap_bot)
    gap_mid = float(0.5 * (gap_bot + gap_top))
    gap_ratio = float(gap_width / gap_mid)

    # Gap opening sweep vs asymmetry delta
    deltas = np.linspace(-0.18, 0.18, 9 if quick else 13)
    gaps_pwe = []
    for d in deltas:
        p_d = PWE2D(
            lat.a1,
            lat.a2,
            lat.honeycomb_primitive_sites(r0 * (1.0 + d), r0 * (1.0 - d), eps_r),
            eps_bg=1.0,
            n_pw=5 if quick else 6,
            polarization="TM",
        )
        fk_d = p_d.solve_k(k_K, n_bands=2)
        gaps_pwe.append(float(np.sign(d) * (fk_d[1] - fk_d[0]) if abs(d) > 1e-12 else 0.0))
    gaps_pwe_arr = np.asarray(gaps_pwe, dtype=float)
    gap_slope_pwe = float(np.dot(deltas, gaps_pwe_arr) / np.dot(deltas, deltas))

    # 2D FIT validation of band 1 and band 2 at K for delta = 0.12
    nx_fit = 22 if quick else 28
    ny_fit = 36 if quick else 48
    nt_fit = int(round(82.0 * nx_fit))
    pre_bulk = _preflight_fit(nx_fit, ny_fit, nt_fit, dtype="complex128", cpml=False)
    t0_fit = time.perf_counter()
    inc_hc_sup = lat.honeycomb_supercell_sites(r0 * (1.0 + delta0), r0 * (1.0 - delta0), eps_r)
    fit_k_both = fit_supercell_bands_at_k(
        lat, inc_hc_sup, k_K, polarization="TM", nx=nx_fit, ny=ny_fit, eps_bg=1.0, f_min=0.22, f_max=0.35, n_steps=nt_fit
    )
    wall_bulk_fit = float(time.perf_counter() - t0_fit)

    unf_k = fit_k_both["unfolded_frequencies"]
    f_fit_b1 = float(unf_k[np.argmin(np.abs(unf_k - gap_bot))])
    f_fit_b2 = float(unf_k[np.argmin(np.abs(unf_k - gap_top))])
    fit_gap_width = float(f_fit_b2 - f_fit_b1)

    # Berry curvature on k-patch around K and K' (PWE, 2D FIT, and analytical massive Dirac)
    dk_patch = 0.08 * (2.0 * np.pi / DEFAULT_A_M)
    nk_pwe_patch = 11 if quick else 15
    q_pwe = np.linspace(-dk_patch, dk_patch, nk_pwe_patch)

    pwe_neg_prim = PWE2D(
        lat.a1,
        lat.a2,
        lat.honeycomb_primitive_sites(r0 * (1.0 - delta0), r0 * (1.0 + delta0), eps_r),
        eps_bg=1.0,
        n_pw=5,
        polarization="TM",
    )
    om_pwe_k_pos, c_pwe_k_pos = pwe_val_prim.berry_curvature(k_K[0] + q_pwe, k_K[1] + q_pwe, band_index=0)
    om_pwe_k_neg, c_pwe_k_neg = pwe_neg_prim.berry_curvature(k_K[0] + q_pwe, k_K[1] + q_pwe, band_index=0)
    om_pwe_kp_pos, c_pwe_kp_pos = pwe_val_prim.berry_curvature(k_Kp[0] + q_pwe, k_Kp[1] + q_pwe, band_index=0)

    # Analytical massive Dirac Berry curvature on the same patch
    qc_pwe = 0.5 * (q_pwe[:-1] + q_pwe[1:])
    QX, QY = np.meshgrid(qc_pwe, qc_pwe, indexing="ij")
    om_dirac_k_pos = dirac_berry_curvature(QX, QY, v_d=v_d_over_c, delta_omega=gap_width * 2.0 * np.pi, valley=+1)
    dq_step = float(q_pwe[1] - q_pwe[0])
    c_dirac_k_pos = float(np.sum(om_dirac_k_pos) * (dq_step**2) / (2.0 * np.pi))

    # Full-BZ 2D Berry curvature map from PWE for visualization
    nk_bz = 15 if quick else 21
    kx_bz = np.linspace(-1.5 * np.pi / DEFAULT_A_M, 1.5 * np.pi / DEFAULT_A_M, nk_bz)
    ky_bz = np.linspace(-1.5 * np.pi / DEFAULT_A_M, 1.5 * np.pi / DEFAULT_A_M, nk_bz)
    om_bz_map, _ = pwe_val_prim.berry_curvature(kx_bz, ky_bz, band_index=0)

    # 2D FIT Berry curvature patch around K for delta = +0.12, -0.12, 0.0 (n_steps = 6000)
    # plus a ring-down duration convergence sweep n_steps in [1600, 3000, 6000, 10000]
    t0_berry_fit = time.perf_counter()
    berry_nsteps_list = [1600, 3000, 6000] if quick else [1600, 3000, 6000, 10000]
    c_patch_vs_nsteps = []
    fit_berry_pos = None
    for n_st in berry_nsteps_list:
        res_st = fit_berry_curvature_patch(
            lat, r_0=r0, delta=+delta0, eps_rod=eps_r, dk_frac=0.08, n_k=3, nx=18, ny=30, n_steps=n_st
        )
        c_patch_vs_nsteps.append(float(res_st["c_patch"]))
        if n_st == 6000:
            fit_berry_pos = res_st
    assert fit_berry_pos is not None
    fit_berry_neg = fit_berry_curvature_patch(
        lat, r_0=r0, delta=-delta0, eps_rod=eps_r, dk_frac=0.08, n_k=3, nx=18, ny=30, n_steps=6000
    )
    fit_berry_zero = fit_berry_curvature_patch(
        lat, r_0=r0, delta=0.0, eps_rod=eps_r, dk_frac=0.08, n_k=3, nx=18, ny=30, n_steps=6000
    )
    wall_berry_fit = float(time.perf_counter() - t0_berry_fit)
    c_fit_pos_val = float(fit_berry_pos["c_patch"])
    c_fit_neg_val = float(fit_berry_neg["c_patch"])
    c_fit_zero_val = float(fit_berry_zero["c_patch"])
    c_fit_rel_err = float(abs(c_fit_pos_val - float(c_pwe_k_pos)) / abs(float(c_pwe_k_pos)))

    return {
        "metrics": {
            "dirac_freq_norm": f_dirac,
            "dirac_degeneracy_split": dirac_degeneracy_split,
            "dirac_velocity_over_c": v_d_over_c,
            "unfolding_max_error": unfolding_max_error,
            "gap_bottom_pwe": gap_bot,
            "gap_top_pwe": gap_top,
            "gap_width_pwe": gap_width,
            "gap_mid_pwe": gap_mid,
            "gap_midgap_ratio_pwe": gap_ratio,
            "gap_slope_vs_delta": gap_slope_pwe,
            "fit_band1_K": f_fit_b1,
            "fit_band2_K": f_fit_b2,
            "fit_gap_width_K": fit_gap_width,
            "fit_band1_rel_err": float(abs(f_fit_b1 - gap_bot) / gap_bot),
            "fit_band2_rel_err": float(abs(f_fit_b2 - gap_top) / gap_top),
            "c_patch_dirac_K_pos": c_dirac_k_pos,
            "c_patch_pwe_K_pos": float(c_pwe_k_pos),
            "c_patch_pwe_K_neg": float(c_pwe_k_neg),
            "c_patch_pwe_Kp_pos": float(c_pwe_kp_pos),
            "c_patch_fit_K_pos": c_fit_pos_val,
            "c_patch_fit_K_neg": c_fit_neg_val,
            "c_patch_fit_K_zero": c_fit_zero_val,
            "c_patch_fit_rel_err_vs_pwe": c_fit_rel_err,
            "berry_nsteps_sweep": berry_nsteps_list,
            "c_patch_fit_vs_nsteps": c_patch_vs_nsteps,
            "wall_bulk_fit_s": wall_bulk_fit,
            "wall_berry_fit_s": wall_berry_fit,
            "preflight_bulk_ram_mb": pre_bulk["ram_mb"],
        },
        "arrays": {
            "path_s": path.s,
            "path_ticks": path.tick_positions,
            "bands_sym_prim": bands_sym_prim,
            "bands_val_prim": bands_val_prim,
            "bands_val_sup_folded": sup_val_res["frequencies"],
            "bands_val_sup_unfolded": sup_val_res["unfolded_frequencies"],
            "deltas": deltas,
            "gaps_pwe": gaps_pwe_arr,
            "q_pwe": q_pwe,
            "om_pwe_k_pos": om_pwe_k_pos,
            "om_pwe_k_neg": om_pwe_k_neg,
            "om_pwe_kp_pos": om_pwe_kp_pos,
            "om_dirac_k_pos": om_dirac_k_pos,
            "kx_bz": kx_bz,
            "ky_bz": ky_bz,
            "om_bz_map": om_bz_map,
            "fit_berry_kx": np.asarray(fit_berry_pos["kx_grid"]),
            "fit_berry_om_pos": np.asarray(fit_berry_pos["omega"]),
            "fit_berry_om_neg": np.asarray(fit_berry_neg["omega"]),
            "berry_nsteps_sweep": np.asarray(berry_nsteps_list, dtype=int),
            "c_patch_fit_vs_nsteps": np.asarray(c_patch_vs_nsteps, dtype=float),
        },
    }


def run_ribbon_dispersion_study(
    quick: bool = False,
    bulk_gap: tuple[float, float] = (0.2715, 0.3059),
) -> dict[str, Any]:
    """Compute projected 1D band structures of both valley-Hall and tuned trivial ribbon supercells."""
    lat = HexagonalLattice(a=DEFAULT_A_M)
    n_cells_y = 6
    r0 = DEFAULT_R0_OVER_A
    delta = DEFAULT_DELTA
    eps_r = DEFAULT_EPS_ROD
    r_def = DEFAULT_DEFECT_RADIUS_OVER_A

    inclusions, (lx_rib, ly_rib) = lat.ribbon_supercell_sites(
        n_cells_y=n_cells_y, r_0=r0, delta_lower=+delta, delta_upper=-delta, eps_in=eps_r
    )
    a1_rib = np.array([lx_rib, 0.0], dtype=float)
    a2_rib = np.array([0.0, ly_rib], dtype=float)

    # Also build the tuned trivial line-defect ribbon supercell (defect_radius = 0.65 * r_0 = 0.117 a)
    _, ly_unit = lat.supercell_size
    yw_mid = 0.5 * ly_rib
    base_sites = lat.honeycomb_supercell_sites(r0, r0, eps_r)
    inclusions_triv: list[CircularInclusion] = []
    for iy in range(n_cells_y):
        for s in base_sites:
            cx = s.x
            cy = iy * ly_unit + s.y
            if abs(cy - yw_mid) <= 0.48 * DEFAULT_A_M:
                inclusions_triv.append(CircularInclusion(x=cx, y=cy, radius=r_def, eps_in=eps_r, label="core"))
            else:
                is_a = s.label.startswith("A")
                rad = r0 * (1.0 + abs(delta)) if is_a else r0 * (1.0 - abs(delta))
                inclusions_triv.append(CircularInclusion(x=cx, y=cy, radius=rad, eps_in=eps_r, label="cladding"))

    n_pw_rib = 3 if quick else 4
    pwe_rib = PWE2D(a1_rib, a2_rib, inclusions, eps_bg=1.0, n_pw=n_pw_rib, polarization="TM")
    pwe_triv = PWE2D(a1_rib, a2_rib, inclusions_triv, eps_bg=1.0, n_pw=n_pw_rib, polarization="TM")

    nk_x = 17 if quick else 25
    kx_vals = np.linspace(0.0, 2.0 * np.pi / DEFAULT_A_M, nk_x)
    kx_over_2pi = kx_vals * DEFAULT_A_M / (2.0 * np.pi)
    n_bands_rib = 4 * n_cells_y + 4
    freqs_rib = np.zeros((nk_x, n_bands_rib), dtype=float)
    freqs_triv_rib = np.zeros((nk_x, n_bands_rib), dtype=float)
    wall_confinement = np.zeros((nk_x, n_bands_rib), dtype=float)
    triv_confinement = np.zeros((nk_x, n_bands_rib), dtype=float)

    # Evaluate transverse confinement near the two domain walls (y = ly_rib/2 and y = 0/ly_rib)
    ny_grid = 12 * n_cells_y
    y_line = (np.arange(ny_grid, dtype=float) + 0.5) * (ly_rib / ny_grid)
    dist_wall_mid = np.abs(y_line - 0.5 * ly_rib)
    dist_wall_edge = np.minimum(y_line, ly_rib - y_line)
    mask_wall = (dist_wall_mid < 1.2 * np.sqrt(3.0) * DEFAULT_A_M) | (dist_wall_edge < 1.2 * np.sqrt(3.0) * DEFAULT_A_M)
    mask_mid_only = dist_wall_mid < 1.2 * np.sqrt(3.0) * DEFAULT_A_M

    edge_profile_y = np.zeros(ny_grid, dtype=float)
    triv_profile_y = np.zeros(ny_grid, dtype=float)
    k_K_proj = 2.0 * np.pi / (3.0 * DEFAULT_A_M)
    ik_target = int(np.argmin(np.abs(kx_vals - k_K_proj)))
    idx_triv_band = 2 * n_cells_y - 1

    for ik, kx in enumerate(kx_vals):
        fk, vk = pwe_rib.solve_k((kx, 0.0), n_bands=n_bands_rib, return_vectors=True, normalize_a=DEFAULT_A_M)
        freqs_rib[ik] = fk
        for ib in range(n_bands_rib):
            _, _, field_xy = pwe_rib.reconstruct_field(vk[:, ib], 8, ny_grid, k=(kx, 0.0), bloch=False)
            prof_y = np.mean(np.abs(field_xy) ** 2, axis=0)
            wall_confinement[ik, ib] = float(np.sum(prof_y[mask_wall]) / max(np.sum(prof_y), 1e-30))
            if ik == ik_target and ib == 2 * n_cells_y:
                # Store mid-wall localized mode profile (small-rod interior domain wall at y = ly_rib / 2)
                edge_profile_y = prof_y / max(float(np.max(prof_y)), 1e-30)

        fk_tr, vk_tr = pwe_triv.solve_k((kx, 0.0), n_bands=n_bands_rib, return_vectors=True, normalize_a=DEFAULT_A_M)
        freqs_triv_rib[ik] = fk_tr
        for ib in range(n_bands_rib):
            _, _, fxy_tr = pwe_triv.reconstruct_field(vk_tr[:, ib], 8, ny_grid, k=(kx, 0.0), bloch=False)
            prof_tr = np.mean(np.abs(fxy_tr) ** 2, axis=0)
            triv_confinement[ik, ib] = float(np.sum(prof_tr[mask_mid_only]) / max(np.sum(prof_tr), 1e-30))
            if ik == ik_target and ib == idx_triv_band:
                triv_profile_y = prof_tr / max(float(np.max(prof_tr)), 1e-30)

    # Identify the two topological edge-state branches inside the bulk gap (bands 2*n_cells_y - 1 and 2*n_cells_y):
    # - Mode 2 at k_x = K_x is localized on the interior small-rod domain wall (y = L_rib / 2, our physical waveguide wall)
    # - Mode 1 at k_x = K_x is localized on the periodic-boundary large-rod domain wall (y = 0 == L_rib)
    # Near k_x a / (2*pi) = 0.375 the two branches anticross across the finite ribbon, opening a mini-gap [max(b1), min(b2)].
    idx_edge_1 = 2 * n_cells_y - 1
    idx_edge_2 = 2 * n_cells_y
    b1 = freqs_rib[:, idx_edge_1]
    b2 = freqs_rib[:, idx_edge_2]
    b_triv = freqs_triv_rib[:, idx_triv_band]

    half_mask = kx_over_2pi <= 0.5001
    idx_half_end = int(np.argmin(np.abs(kx_over_2pi - 0.5)))
    gap_bot, gap_top = float(bulk_gap[0]), float(bulk_gap[1])

    f_mid_wall_at_K = float(b2[ik_target])
    f_bnd_wall_at_K = float(b1[ik_target])
    f_edge_at_K = float(0.5 * (f_bnd_wall_at_K + f_mid_wall_at_K))
    max_edge_confinement = float(np.max(wall_confinement[ik_target, [idx_edge_1, idx_edge_2]]))
    f_triv_at_K = float(b_triv[ik_target])
    triv_conf_at_K = float(triv_confinement[ik_target, idx_triv_band])

    # Derive windows directly from the computed ribbon bands (no hard-coded bounds):
    minigap_bot = float(np.max(b1[half_mask]))
    minigap_top = float(np.min(b2[half_mask]))
    lower_win_bot = float(max(gap_bot, np.min(b_triv[half_mask])))
    lower_win_top = minigap_bot
    upper_win_bot = float(b2[idx_half_end])
    upper_win_top = float(min(b2[0], gap_top, np.max(b_triv[half_mask])))

    return {
        "metrics": {
            "n_cells_y": n_cells_y,
            "edge_freq_at_K_norm": f_edge_at_K,
            "interior_wall_freq_at_K_norm": f_mid_wall_at_K,
            "boundary_wall_freq_at_K_norm": f_bnd_wall_at_K,
            "edge_confinement_fraction": max_edge_confinement,
            "trivial_defect_radius_over_a": r_def,
            "trivial_freq_at_K_norm": f_triv_at_K,
            "trivial_confinement_fraction": triv_conf_at_K,
            "trivial_core_offset_over_a": float(0.25 * np.sqrt(3.0)),
            "bulk_gap_window": [gap_bot, gap_top],
            "lower_window": [lower_win_bot, lower_win_top],
            "ribbon_minigap_window": [minigap_bot, minigap_top],
            "upper_single_mode_window": [upper_win_bot, upper_win_top],
        },
        "arrays": {
            "kx_over_2pi": kx_over_2pi,
            "freqs_rib": freqs_rib,
            "freqs_triv_rib": freqs_triv_rib,
            "wall_confinement": wall_confinement,
            "triv_confinement": triv_confinement,
            "y_line_over_a": y_line / DEFAULT_A_M,
            "edge_profile_y": edge_profile_y,
            "triv_profile_y": triv_profile_y,
            "ly_rib_over_a": float(ly_rib / DEFAULT_A_M),
        },
    }


def _run_single_waveguide_fit(
    kind: str,
    nx_cell_res: int,
    ny_cell_res: int,
    f_norm_arr: np.ndarray,
    f_dft_norm: float = 0.2929,
    record_snapshots: bool = False,
) -> dict[str, Any]:
    """Run one 2D FIT waveguide simulation sequentially and return flux spectra & fields."""
    lat = HexagonalLattice(a=DEFAULT_A_M)
    lx_s, ly_s = lat.supercell_size
    nx_cells, ny_cells = 9, 7
    nx = nx_cells * nx_cell_res
    ny = ny_cells * ny_cell_res
    lx = nx_cells * lx_s
    ly = ny_cells * ly_s
    lz = 10.0 * lx_s

    n_steps = int(round(260.0 * nx_cell_res))
    pre = _preflight_fit(nx, ny, n_steps, dtype="float64", cpml=True)

    t0 = time.perf_counter()
    grid = GridFIT3D(0.0, lx, 0.0, ly, 0.0, lz, nx, ny, 1, use_mpi=False, verbose=0)
    solver = SolverFIT3D(
        grid,
        bc_low=["cpml", "cpml", "periodic"],
        bc_high=["cpml", "cpml", "periodic"],
        n_pml=6,
        dtype=np.float64,
        verbose=0,
    )

    inclusions, path_x, path_y = lat.domain_wall_bend_inclusions(
        nx_cells=nx_cells,
        ny_cells=ny_cells,
        r_0=DEFAULT_R0_OVER_A,
        delta=DEFAULT_DELTA,
        eps_in=DEFAULT_EPS_ROD,
        kind=kind,
        defect_radius=DEFAULT_DEFECT_RADIUS_OVER_A,
    )
    eps_map = fill_circular_inclusions(solver, inclusions, eps_bg=DEFAULT_EPS_BG, n_sub=4, periodic_wrap=False)

    # In the trivial line-defect waveguide (uniform delta = +0.12 cladding with B1 and A1 shrunk to r_def = 0.117 a),
    # the donor defect mode is centered on the A1-B2 chain at y_w(x) + 0.25 * L_y = y_w(x) + 0.433 a.
    y_offset = 0.25 * ly_s if kind.startswith("trivial") else 0.0
    path_y_mode = np.asarray(path_y, dtype=float) + y_offset
    y_in = float(path_y_mode[0])
    y_out = float(path_y_mode[-1])

    f_hz_arr = f_norm_arr * c_light / DEFAULT_A_M
    f0_norm = 0.289
    f0_hz = f0_norm * c_light / DEFAULT_A_M
    df_hz = 0.018 * c_light / DEFAULT_A_M
    f_dft_hz = float(f_dft_norm) * c_light / DEFAULT_A_M

    # Input and output flux monitors across the waveguide channel (x_out = 7.2 a, upstream of the output end-sponge)
    mon_in = FluxMonitor(f_hz_arr, center=(2.5 * DEFAULT_A_M, y_in, 0.5 * lz), size=(0.0, 2.2 * DEFAULT_A_M, lz))
    mon_out = FluxMonitor(
        f_hz_arr, center=(7.2 * DEFAULT_A_M, y_out, 0.5 * lz), size=(0.0, 2.2 * DEFAULT_A_M, lz)
    )
    mon_dft = DFTMonitor([f_dft_hz], components=("Ez",), center=(0.5 * lx, 0.5 * ly, 0.5 * lz), size=(lx, ly, 0.0))

    # Soft line source across the waveguide channel at x = 1.5 a (aligned with A1/B1 vertical bond)
    x_nodes = np.asarray(grid.x[:nx], dtype=float)
    y_nodes = np.asarray(grid.y[:ny], dtype=float)
    ix_src = int(round((1.5 * DEFAULT_A_M) / (lx / nx)))
    sigma_y = 0.35 * DEFAULT_A_M
    trans_profile = np.exp(-0.5 * ((y_nodes - y_in) / sigma_y) ** 2)

    # Matched (E, H) adiabatic end-sponge in the x < 1.1 a and x > 7.5 a termination zones to suppress
    # Bloch-impedance reflections where the periodic photonic crystal meets the homogeneous CPML.
    damp_x = np.ones(nx, dtype=float)
    for i, xv in enumerate(x_nodes):
        if xv < 1.1 * DEFAULT_A_M:
            damp_x[i] = 1.0 - 0.03 * ((1.1 * DEFAULT_A_M - xv) / (1.1 * DEFAULT_A_M)) ** 3
        elif xv > 7.5 * DEFAULT_A_M:
            damp_x[i] = 1.0 - 0.03 * ((xv - 7.5 * DEFAULT_A_M) / (lx - 7.5 * DEFAULT_A_M)) ** 3
    damp_2d = damp_x[:, None]

    t_arr = np.arange(n_steps, dtype=float) * solver.dt
    tau_p = 1.0 / (np.pi * df_hz)
    t_peak = 2.5 * tau_p
    pulse = np.exp(-((t_arr - t_peak) / tau_p) ** 2) * np.sin(2.0 * np.pi * f0_hz * (t_arr - t_peak))

    snapshots = []
    snap_start = int(round(0.18 * n_steps))
    snap_end = int(round(0.76 * n_steps))
    snap_stride = max(1, (snap_end - snap_start) // 26)

    for n in range(n_steps):
        t_now = n * solver.dt
        solver.J[ix_src, :ny, 0, "z"] = pulse[n] * trans_profile
        solver.one_step()
        solver.E[:nx, :ny, 0, "z"] *= damp_2d
        solver.H[:nx, :ny, 0, "x"] *= damp_2d
        solver.H[:nx, :ny, 0, "y"] *= damp_2d
        mon_in.update(solver, t_now)
        mon_out.update(solver, t_now)
        mon_dft.update(solver, t_now)
        if record_snapshots and (snap_start <= n <= snap_end) and ((n - snap_start) % snap_stride == 0) and len(snapshots) < 26:
            snapshots.append(np.asarray(solver.E[:nx, :ny, 0, "z"], dtype=float).copy())

    wall_s = float(time.perf_counter() - t0)
    flux_in = np.asarray(mon_in.get_flux(), dtype=float)
    flux_out = np.asarray(mon_out.get_flux(), dtype=float)
    ez_dft = mon_dft.get_field("Ez")[0][0, :nx, :ny, 0]

    return {
        "f_norm": f_norm_arr,
        "flux_in": flux_in,
        "flux_out": flux_out,
        "ez_dft": ez_dft,
        "eps_map": eps_map,
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "path_x": path_x,
        "path_y": path_y_mode,
        "inclusions": inclusions,
        "snapshots": np.asarray(snapshots, dtype=float) if snapshots else np.zeros((0, nx, ny)),
        "wall_seconds": wall_s,
        "ram_mb": pre["ram_mb"],
    }


def run_waveguide_bend_comparison(
    quick: bool = False,
    rib_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare straight and 120-deg Z-bend transmission in valley-Hall vs tuned trivial PhC waveguides."""
    if rib_metrics is None:
        rib_metrics = run_ribbon_dispersion_study(quick=True)["metrics"]

    bulk_gap_win = rib_metrics["bulk_gap_window"]
    lower_win = rib_metrics["lower_window"]
    minigap_win = rib_metrics["ribbon_minigap_window"]
    upper_win = rib_metrics["upper_single_mode_window"]
    f_dft_norm = float(rib_metrics["interior_wall_freq_at_K_norm"])

    f_norm_arr = np.linspace(float(bulk_gap_win[0]), float(bulk_gap_win[1]), 19)
    nx_res = 16 if quick else 22
    ny_res = 28 if quick else 36

    # Run the 4 waveguide simulations strictly ONE AT A TIME
    res_vh_straight = _run_single_waveguide_fit(
        "valley_straight", nx_res, ny_res, f_norm_arr, f_dft_norm=f_dft_norm, record_snapshots=False
    )
    res_vh_bend = _run_single_waveguide_fit(
        "valley_bend", nx_res, ny_res, f_norm_arr, f_dft_norm=f_dft_norm, record_snapshots=True
    )
    res_tr_straight = _run_single_waveguide_fit(
        "trivial_straight", nx_res, ny_res, f_norm_arr, f_dft_norm=f_dft_norm, record_snapshots=False
    )
    res_tr_bend = _run_single_waveguide_fit(
        "trivial_bend", nx_res, ny_res, f_norm_arr, f_dft_norm=f_dft_norm, record_snapshots=False
    )

    # Normalize the output flux of each waveguide at x_out = 7.2 a to the incident guided power
    # delivered to the identical plane x_out = 7.2 a of the matched straight reference waveguide
    # with the identical source and end-sponge (so straight reference T_straight == 1 and bend T_bend <= 1).
    ref_vh_out = np.abs(res_vh_straight["flux_out"])
    ref_tr_out = np.abs(res_tr_straight["flux_out"])

    t_vh_straight = np.abs(res_vh_straight["flux_out"]) / ref_vh_out
    t_vh_bend = np.abs(res_vh_bend["flux_out"]) / ref_vh_out
    t_tr_straight = np.abs(res_tr_straight["flux_out"]) / ref_tr_out
    t_tr_bend = np.abs(res_tr_bend["flux_out"]) / ref_tr_out

    f_norm = res_vh_bend["f_norm"]
    mask_upper = (f_norm >= upper_win[0]) & (f_norm <= upper_win[1])
    mask_lower = (f_norm >= lower_win[0]) & (f_norm <= lower_win[1])

    # Headline single-mode window metrics (derived upper single-mode window [upper_win[0], upper_win[1]])
    mean_t_vh_str = float(np.mean(t_vh_straight[mask_upper]))
    mean_t_tr_str = float(np.mean(t_tr_straight[mask_upper]))
    mean_t_vh = float(np.mean(t_vh_bend[mask_upper]))
    mean_t_tr = float(np.mean(t_tr_bend[mask_upper]))
    peak_t_vh = float(np.max(t_vh_bend))
    peak_t_tr = float(np.max(t_tr_bend))
    bend_to_str_vh = float(mean_t_vh / mean_t_vh_str)
    bend_to_str_tr = float(mean_t_tr / mean_t_tr_str)
    suppression_db = float(10.0 * np.log10(mean_t_vh / mean_t_tr))

    # Lower window and full-bulk-gap averages
    mean_t_vh_lower = float(np.mean(t_vh_bend[mask_lower]))
    mean_t_tr_lower = float(np.mean(t_tr_bend[mask_lower]))
    suppression_lower_db = float(10.0 * np.log10(mean_t_vh_lower / mean_t_tr_lower))

    mean_t_vh_full = float(np.mean(t_vh_bend))
    mean_t_tr_full = float(np.mean(t_tr_bend))
    suppression_full_db = float(10.0 * np.log10(mean_t_vh_full / mean_t_tr_full))

    return {
        "metrics": {
            "nx_per_a": nx_res,
            "ny_per_supercell": ny_res,
            "trivial_defect_radius_over_a": DEFAULT_DEFECT_RADIUS_OVER_A,
            "dft_freq_norm": f_dft_norm,
            "bulk_gap_window": [float(bulk_gap_win[0]), float(bulk_gap_win[1])],
            "lower_window": [float(lower_win[0]), float(lower_win[1])],
            "ribbon_minigap_window": [float(minigap_win[0]), float(minigap_win[1])],
            "upper_single_mode_window": [float(upper_win[0]), float(upper_win[1])],
            "mean_transmission_valley_straight": mean_t_vh_str,
            "mean_transmission_trivial_straight": mean_t_tr_str,
            "mean_transmission_valley_bend": mean_t_vh,
            "mean_transmission_trivial_bend": mean_t_tr,
            "bend_to_straight_valley": bend_to_str_vh,
            "bend_to_straight_trivial": bend_to_str_tr,
            "peak_transmission_valley_bend": peak_t_vh,
            "peak_transmission_trivial_bend": peak_t_tr,
            "bend_transmission_advantage_db": suppression_db,
            "mean_transmission_valley_bend_lower": mean_t_vh_lower,
            "mean_transmission_trivial_bend_lower": mean_t_tr_lower,
            "bend_transmission_advantage_lower_db": suppression_lower_db,
            "mean_transmission_valley_bend_full_gap": mean_t_vh_full,
            "mean_transmission_trivial_bend_full_gap": mean_t_tr_full,
            "bend_transmission_advantage_full_gap_db": suppression_full_db,
            "wall_vh_straight_s": res_vh_straight["wall_seconds"],
            "wall_vh_bend_s": res_vh_bend["wall_seconds"],
            "wall_tr_straight_s": res_tr_straight["wall_seconds"],
            "wall_tr_bend_s": res_tr_bend["wall_seconds"],
            "ram_mb_waveguide": res_vh_bend["ram_mb"],
        },
        "arrays": {
            "f_norm": f_norm,
            "t_valley_straight": t_vh_straight,
            "t_valley_bend": t_vh_bend,
            "t_trivial_straight": t_tr_straight,
            "t_trivial_bend": t_tr_bend,
            "ez_dft_valley_bend": res_vh_bend["ez_dft"],
            "ez_dft_trivial_bend": res_tr_bend["ez_dft"],
            "eps_map_valley_bend": res_vh_bend["eps_map"],
            "eps_map_trivial_bend": res_tr_bend["eps_map"],
            "x_nodes": res_vh_bend["x_nodes"],
            "y_nodes": res_vh_bend["y_nodes"],
            "path_x": res_vh_bend["path_x"],
            "path_y": res_vh_bend["path_y"],
            "path_y_trivial": res_tr_bend["path_y"],
            "snapshots_valley_bend": res_vh_bend["snapshots"],
        },
        "inclusions_valley": res_vh_bend["inclusions"],
        "inclusions_trivial": res_tr_bend["inclusions"],
    }


def run_convergence_study(quick: bool = False) -> dict[str, Any]:
    """Run a 3-resolution 2D FIT convergence study and Richardson extrapolation."""
    lat = HexagonalLattice(a=DEFAULT_A_M)
    k_K = lat.high_symmetry_points()["K"]
    r0 = DEFAULT_R0_OVER_A
    delta0 = DEFAULT_DELTA
    eps_r = DEFAULT_EPS_ROD

    pwe_ref = PWE2D(
        lat.a1,
        lat.a2,
        lat.honeycomb_primitive_sites(r0 * (1.0 + delta0), r0 * (1.0 - delta0), eps_r),
        eps_bg=1.0,
        n_pw=9,
        polarization="TM",
    )
    fk_ref = pwe_ref.solve_k(k_K, n_bands=2)
    f1_pwe_ref = float(fk_ref[0])

    # Three grids where ny is divisible by 6 so all four honeycomb sites lie on exact grid nodes
    grids = [(12, 24), (18, 30), (24, 42)] if quick else [(18, 30), (24, 42), (30, 54)]
    inc_sup = lat.honeycomb_supercell_sites(r0 * (1.0 + delta0), r0 * (1.0 - delta0), eps_r)

    nx_list = []
    f1_vals = []
    wall_list = []
    ram_list = []

    for gx, gy in grids:
        nt = int(round(85.0 * gx))
        pre = _preflight_fit(gx, gy, nt, dtype="complex128", cpml=False)
        t0 = time.perf_counter()
        res = fit_supercell_bands_at_k(
            lat, inc_sup, k_K, polarization="TM", nx=gx, ny=gy, eps_bg=1.0, f_min=0.22, f_max=0.285, n_steps=nt
        )
        wall_list.append(float(time.perf_counter() - t0))
        ram_list.append(pre["ram_mb"])
        f_pick = float(res["unfolded_frequencies"][np.argmin(np.abs(res["unfolded_frequencies"] - f1_pwe_ref))])
        nx_list.append(int(gx))
        f1_vals.append(f_pick)

    # Audit finest grid against the >= 20 cells/lambda and >= 4 cells/diameter requirements
    audit_finest = audit_honeycomb_resolution(nx_list[-1], grids[-1][1])

    h1 = 1.0 / float(nx_list[-2])
    h2 = 1.0 / float(nx_list[-1])
    r2 = (h1 / h2) ** 2
    f1_rich = float((r2 * f1_vals[-1] - f1_vals[-2]) / (r2 - 1.0))
    rel_errors = [float(abs(v - f1_rich) / f1_rich) for v in f1_vals]

    # Observed order from coarse and medium vs Richardson extrapolate
    h0 = 1.0 / float(nx_list[0])
    if rel_errors[0] > 1e-12 and rel_errors[-1] > 1e-12:
        p_obs = float(np.log(rel_errors[0] / rel_errors[-1]) / np.log(h0 / h2))
    else:
        p_obs = 2.0

    return {
        "metrics": {
            "grids_nx": nx_list,
            "grids_ny": [g[1] for g in grids],
            "f1_K_values": f1_vals,
            "f1_K_richardson": f1_rich,
            "f1_K_pwe_ref": f1_pwe_ref,
            "richardson_vs_pwe_rel_err": float(abs(f1_rich - f1_pwe_ref) / f1_pwe_ref),
            "rel_errors_vs_richardson": rel_errors,
            "observed_order_p": p_obs,
            "cells_per_wavelength_finest": audit_finest["cells_per_wavelength"],
            "cells_across_min_diameter_finest": audit_finest["cells_across_min_diameter"],
            "wall_seconds_per_grid": wall_list,
            "ram_mb_per_grid": ram_list,
        }
    }


def plot_all_figures(
    bulk_data: dict[str, Any],
    rib_data: dict[str, Any],
    bend_data: dict[str, Any],
    conv_data: dict[str, Any],
) -> dict[str, str]:
    """Generate all tutorial figures and the animated wavefront GIF in figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    b_arr = bulk_data["arrays"]
    b_met = bulk_data["metrics"]
    r_arr = rib_data["arrays"]
    w_arr = bend_data["arrays"]
    w_met = bend_data["metrics"]
    c_met = conv_data["metrics"]

    # ------------------------------------------------------------------
    # Figure 1: Supercell band unfolding & Dirac-to-Valley-Hall gap scaling
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes1[0]
    s_path = b_arr["path_s"]
    ticks = b_arr["path_ticks"]
    for ib in range(8):
        ax.plot(
            s_path,
            b_arr["bands_val_sup_folded"][:, ib],
            ":",
            color="#94a3b8",
            lw=1.0,
            label="Supercell folded" if ib == 0 else None,
        )
    for ib in range(4):
        ax.plot(
            s_path,
            b_arr["bands_sym_prim"][:, ib],
            "--",
            color="#64748b",
            lw=1.4,
            label=r"Primitive $\delta=0$ (Dirac)" if ib == 0 else None,
        )
        ax.plot(
            s_path,
            b_arr["bands_val_sup_unfolded"][:, ib],
            "-",
            color="#0072B2",
            lw=2.0,
            label=r"Unfolded supercell $\delta=0.12$" if ib == 0 else None,
        )
    ax.axhspan(b_met["gap_bottom_pwe"], b_met["gap_top_pwe"], color="#E69F00", alpha=0.25, label="Valley-Hall gap")
    ax.plot(
        [ticks[2], ticks[2]],
        [b_met["fit_band1_K"], b_met["fit_band2_K"]],
        "D",
        color="#D55E00",
        ms=6.5,
        label="2D FIT + harminv at $K$",
    )
    for xp in ticks:
        ax.axvline(xp, color="#cbd5e1", lw=0.8, zorder=0)
    ax.set_xticks(ticks)
    ax.set_xticklabels([r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"])
    ax.set_xlim(s_path[0], s_path[-1])
    ax.set_ylim(0.0, 0.46)
    ax.set_ylabel(r"Normalized frequency $\omega a / (2\pi c)$")
    ax.set_title(r"(a) Supercell unfolding & valley-Hall bandgap")
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes1[1]
    deltas = b_arr["deltas"]
    gaps = b_arr["gaps_pwe"]
    slope = b_met["gap_slope_vs_delta"]
    ax.plot(deltas, gaps, "o", color="#0072B2", ms=5.8, label="PWE gap splitting at $K$")
    ax.plot(deltas, slope * deltas, "-", color="#D55E00", lw=1.8, label=rf"Linear Dirac mass ($\alpha = {slope:.3f}$)")
    ax.plot([+DEFAULT_DELTA, -DEFAULT_DELTA], [+b_met["fit_gap_width_K"], -b_met["fit_gap_width_K"]], "s", color="#009E73", ms=6.5, label="2D FIT + harminv")
    ax.axhline(0.0, color="#94a3b8", lw=0.8, ls=":")
    ax.axvline(0.0, color="#94a3b8", lw=0.8, ls=":")
    ax.set_xlabel(r"Inversion asymmetry $\delta = (r_A - r_B)/(2 r_0)$")
    ax.set_ylabel(r"Signed valley gap $\mathrm{sgn}(\delta)\,\Delta\omega_K a / (2\pi c)$")
    ax.set_title(r"(b) Valley gap opening vs sublattice asymmetry $\delta$")
    ax.legend(loc="upper left", fontsize=8.5)

    fig1.tight_layout()
    p1 = FIG_DIR / "01_supercell_unfolding_and_dirac_gap.png"
    figstyle.save(fig1, p1)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: Berry curvature map and valley Chern sign flip (PWE, FIT, Dirac)
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes2[0]
    kx_bz = b_arr["kx_bz"] / (2.0 * np.pi)
    ky_bz = b_arr["ky_bz"] / (2.0 * np.pi)
    kx_c = 0.5 * (kx_bz[:-1] + kx_bz[1:])
    ky_c = 0.5 * (ky_bz[:-1] + ky_bz[1:])
    om_map = b_arr["om_bz_map"]
    vmax = float(np.percentile(np.abs(om_map), 98))
    im = ax.pcolormesh(kx_c, ky_c, om_map.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig2.colorbar(im, ax=ax, label=r"$\Omega_1(\mathbf{k}) / a^2$")
    # Overlay hexagonal first Brillouin zone boundary
    angles = np.deg2rad(np.arange(0, 361, 60))
    r_hex = 2.0 / 3.0
    ax.plot(r_hex * np.cos(angles), r_hex * np.sin(angles), "k--", lw=1.4, label="Hexagonal 1st BZ")
    ax.plot([2.0 / 3.0], [0.0], "o", color="#0072B2", ms=6, label=r"$K$ valley ($\Omega_1 < 0$)")
    ax.plot([-2.0 / 3.0], [0.0], "s", color="#D55E00", ms=6, label=r"$K'$ valley ($\Omega_1 > 0$)")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x a / (2\pi)$")
    ax.set_ylabel(r"$k_y a / (2\pi)$")
    ax.set_title(r"(a) Lower-band Berry curvature $\Omega_1(\mathbf{k})$ ($\delta=+0.12$)")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes2[1]
    q_pwe = b_arr["q_pwe"]
    qc = 0.5 * (q_pwe[:-1] + q_pwe[1:]) / (2.0 * np.pi)
    mid = b_arr["om_pwe_k_pos"].shape[1] // 2
    ax.plot(
        qc,
        b_arr["om_pwe_k_pos"][:, mid],
        "o-",
        color="#0072B2",
        lw=1.8,
        label=rf"PWE at $K$ ($\delta=+0.12, C_K={b_met['c_patch_pwe_K_pos']:+.4f}$)",
    )
    ax.plot(
        qc,
        b_arr["om_pwe_k_neg"][:, mid],
        "s-",
        color="#D55E00",
        lw=1.8,
        label=rf"PWE at $K$ ($\delta=-0.12, C_K={b_met['c_patch_pwe_K_neg']:+.4f}$)",
    )
    ax.plot(
        qc,
        b_arr["om_dirac_k_pos"][:, mid],
        "--",
        color="#009E73",
        lw=1.5,
        label=rf"Massive Dirac ($C_K={b_met['c_patch_dirac_K_pos']:+.4f}$)",
    )
    fit_kx = b_arr["fit_berry_kx"]
    fit_qc = (0.5 * (fit_kx[:-1] + fit_kx[1:]) - (4.0 * np.pi / 3.0)) / (2.0 * np.pi)
    ax.plot(
        fit_qc,
        np.mean(b_arr["fit_berry_om_pos"], axis=1),
        "D",
        color="#0072B2",
        ms=6.5,
        label=rf"2D FIT ($N=6000, \delta=+0.12, C_K={b_met['c_patch_fit_K_pos']:+.4f}$)",
    )
    ax.plot(
        fit_qc,
        np.mean(b_arr["fit_berry_om_neg"], axis=1),
        "D",
        color="#D55E00",
        ms=6.5,
        label=rf"2D FIT ($N=6000, \delta=-0.12, C_K={b_met['c_patch_fit_K_neg']:+.4f}$)",
    )
    ax.axhline(0.0, color="#94a3b8", lw=0.8, ls=":")
    ax.set_xlabel(r"Wavevector offset $(k_x - K_x) a / (2\pi)$")
    ax.set_ylabel(r"Berry curvature $\Omega_1(k_x, 0) / a^2$")
    ax.set_title(r"(b) Valley Berry curvature sign flip ($\delta \leftrightarrow -\delta$)")
    ax.legend(loc="lower right", fontsize=8.0)

    fig2.tight_layout()
    p2 = FIG_DIR / "02_berry_curvature_and_valley_chern.png"
    figstyle.save(fig2, p2)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Figure 3: Ribbon supercell projected band structure & edge-mode localization
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes3[0]
    kx_norm = r_arr["kx_over_2pi"]
    freqs_rib = r_arr["freqs_rib"]
    freqs_triv_rib = r_arr["freqs_triv_rib"]
    r_met = rib_data["metrics"]
    n_cells_y = int(r_met["n_cells_y"])
    idx_e1 = 2 * n_cells_y - 1
    idx_e2 = 2 * n_cells_y
    for ib in range(freqs_rib.shape[1]):
        if ib in (idx_e1, idx_e2):
            continue
        ax.plot(kx_norm, freqs_rib[:, ib], "-", color="#94a3b8", lw=1.0, label="Projected bulk bands" if ib == 0 else None)
    ax.plot(
        kx_norm,
        freqs_rib[:, idx_e1],
        "o-",
        color="#D55E00",
        lw=2.0,
        ms=3.8,
        label="Ribbon mode 1 ($y=0$ wall at $K$)",
    )
    ax.plot(
        kx_norm,
        freqs_rib[:, idx_e2],
        "s--",
        color="#0072B2",
        lw=2.0,
        ms=3.8,
        label="Ribbon mode 2 ($y=L_y/2$ wall at $K$)",
    )
    ax.plot(
        kx_norm,
        freqs_triv_rib[:, idx_e1],
        "^:",
        color="#009E73",
        lw=2.0,
        ms=4.0,
        label=r"Tuned trivial guide ($r_{\mathrm{def}}=0.65 r_0$)",
    )
    up_win = r_met["upper_single_mode_window"]
    mg_win = r_met["ribbon_minigap_window"]
    ax.axhspan(b_met["gap_bottom_pwe"], b_met["gap_top_pwe"], color="#E69F00", alpha=0.14, label="Bulk valley bandgap")
    ax.axhspan(
        up_win[0],
        up_win[1],
        color="#009E73",
        alpha=0.18,
        label=rf"Upper single-mode $[{up_win[0]:.4f}, {up_win[1]:.4f}]$",
    )
    ax.axhspan(
        mg_win[0],
        mg_win[1],
        color="#CC79A7",
        alpha=0.35,
        label=rf"Ribbon mini-gap $[{mg_win[0]:.4f}, {mg_win[1]:.4f}]$",
    )
    ax.axvline(1.0 / 3.0, color="#64748b", lw=0.9, ls=":")
    ax.axvline(2.0 / 3.0, color="#64748b", lw=0.9, ls=":")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.22, 0.35)
    ax.set_xlabel(r"Projected wavevector $k_x a / (2\pi)$")
    ax.set_ylabel(r"Normalized frequency $\omega a / (2\pi c)$")
    ax.set_title(r"(a) Ribbon dispersion & derived single-mode windows")
    ax.legend(loc="lower right", fontsize=7.2)

    ax = axes3[1]
    y_over_a = r_arr["y_line_over_a"]
    prof_y = r_arr["edge_profile_y"]
    triv_prof_y = r_arr["triv_profile_y"]
    ly_rib = float(r_arr["ly_rib_over_a"])
    triv_off = float(r_met.get("trivial_core_offset_over_a", 0.25 * np.sqrt(3.0)))
    ax.plot(y_over_a, prof_y, "-", color="#0072B2", lw=2.0, label=r"Valley-Hall $|E_z(y)|^2$ at $k_x = K_x$")
    ax.plot(y_over_a, triv_prof_y, "--", color="#009E73", lw=1.8, label=r"Tuned trivial $|E_z(y)|^2$ at $k_x = K_x$")
    ax.axvline(0.5 * ly_rib, color="#D55E00", lw=1.6, ls="--", label=r"Domain wall ($y = L_y/2$)")
    ax.axvline(
        0.5 * ly_rib + triv_off,
        color="#009E73",
        lw=1.3,
        ls=":",
        label=rf"Trivial defect axis ($y = L_y/2 + {triv_off:.2f}a$)",
    )
    ax.axvspan(0.0, 0.5 * ly_rib, color="#e0f2fe", alpha=0.35, label=r"$\mathrm{VPC}_+$ ($\delta = +0.12$)")
    ax.axvspan(0.5 * ly_rib, ly_rib, color="#fef3c7", alpha=0.35, label=r"$\mathrm{VPC}_-$ ($\delta = -0.12$)")
    ax.set_xlim(0.0, ly_rib)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel(r"Transverse coordinate $y / a$")
    ax.set_ylabel(r"Normalized intensity $\langle |E_z|^2 \rangle_x$")
    ax.set_title(r"(b) Transverse mode profiles at $k_x = K_x$")
    ax.legend(loc="upper right", fontsize=7.5)

    fig3.tight_layout()
    p3 = FIG_DIR / "03_ribbon_domain_wall_dispersion.png"
    figstyle.save(fig3, p3)
    plt.close(fig3)

    # ------------------------------------------------------------------
    # Figure 4: 120-deg Z-bend transmission spectra across full bulk gap & grid convergence
    # ------------------------------------------------------------------
    fig4, axes4 = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ax = axes4[0]
    f_norm = w_arr["f_norm"]
    lo_win = w_met["lower_window"]
    mg_win = w_met["ribbon_minigap_window"]
    up_win = w_met["upper_single_mode_window"]
    ax.axhline(
        1.0,
        color="#475569",
        ls="--",
        lw=1.5,
        label=r"Straight reference guide ($T_{\mathrm{straight}} \equiv 1.00$)",
    )
    ax.plot(
        f_norm,
        w_arr["t_valley_bend"],
        "o-",
        color="#0072B2",
        lw=2.2,
        ms=4.8,
        label=(
            rf"Valley-Hall $120^\circ$ Z-bend ($\langle T \rangle_{{\mathrm{{up}}}} = {w_met['mean_transmission_valley_bend']:.3f}$, "
            rf"$\langle T \rangle_{{\mathrm{{gap}}}} = {w_met['mean_transmission_valley_bend_full_gap']:.3f}$)"
        ),
    )
    ax.plot(
        f_norm,
        w_arr["t_trivial_bend"],
        "s-",
        color="#D55E00",
        lw=2.2,
        ms=4.8,
        label=(
            rf"Trivial PhC $120^\circ$ bend ($\langle T \rangle_{{\mathrm{{up}}}} = {w_met['mean_transmission_trivial_bend']:.3f}$, "
            rf"$\langle T \rangle_{{\mathrm{{gap}}}} = {w_met['mean_transmission_trivial_bend_full_gap']:.3f}$)"
        ),
    )
    ax.axvspan(
        lo_win[0],
        lo_win[1],
        color="#56B4E9",
        alpha=0.14,
        label=rf"Lower window $[{lo_win[0]:.4f}, {lo_win[1]:.4f}]$",
    )
    ax.axvspan(
        mg_win[0],
        mg_win[1],
        color="#CC79A7",
        alpha=0.30,
        label=rf"Ribbon mini-gap $[{mg_win[0]:.4f}, {mg_win[1]:.4f}]$",
    )
    ax.axvspan(
        up_win[0],
        up_win[1],
        color="#009E73",
        alpha=0.18,
        label=rf"Upper single-mode $[{up_win[0]:.4f}, {up_win[1]:.4f}]$",
    )
    ax.set_xlim(f_norm[0], f_norm[-1])
    ax.set_ylim(0.0, 1.12)
    ax.set_xlabel(r"Normalized frequency $\omega a / (2\pi c)$ across bulk gap")
    ax.set_ylabel(r"Normalized transmission $\Phi_{\mathrm{out}}^{\mathrm{bend}}(\omega) / \Phi_{\mathrm{out}}^{\mathrm{straight}}(\omega)$")
    ax.set_title(r"(a) $120^\circ$ Z-bend transmission across full bulk valley gap")
    ax.legend(loc="center left", fontsize=7.2)

    # Panel (b): 3-resolution grid convergence with O(dx^2) guide
    ax = axes4[1]
    nx_c = np.asarray(c_met["grids_nx"], dtype=float)
    err_c = np.asarray(c_met["rel_errors_vs_richardson"], dtype=float)
    guide_c = err_c[0] * (nx_c / nx_c[0]) ** (-2.0)
    ax.loglog(nx_c, err_c, "o-", color="#0072B2", lw=2.0, ms=6.0, label=rf"2D FIT vs Richardson ($p={c_met['observed_order_p']:.2f}$)")
    ax.loglog(nx_c, guide_c, "k--", lw=1.4, label=r"$\mathcal{O}(\Delta x^2)$ second-order guide")
    ax.set_xlabel(r"Grid resolution $N_x$ [cells / $a$]")
    ax.set_ylabel(r"Relative error $|\omega_{N_x} - \omega_{\mathrm{Rich}}| / \omega_{\mathrm{Rich}}$")
    ax.set_title(r"(b) 2D FIT grid convergence & Richardson order")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5)

    fig4.tight_layout()
    p4 = FIG_DIR / "04_bend_transmission_and_convergence.png"
    figstyle.save(fig4, p4)
    plt.close(fig4)

    # ------------------------------------------------------------------
    # Figure 5: 2D steady-state field maps |Ez(x,y)| around the 120-deg Z-bend
    # with overlay_geometry
    # ------------------------------------------------------------------
    fig5, axes5 = plt.subplots(1, 2, figsize=(11.8, 4.6))
    x_n = w_arr["x_nodes"]
    y_n = w_arr["y_nodes"]
    ez_vh = np.abs(w_arr["ez_dft_valley_bend"])
    ez_tr = np.abs(w_arr["ez_dft_trivial_bend"])
    norm_scale = max(float(np.percentile(ez_vh[x_n > 2.4 * DEFAULT_A_M, :], 99.0)), 1e-12)
    path_y_triv = w_arr.get("path_y_trivial", w_arr["path_y"])

    for idx_p, (ax, field_abs, inc_list, py_curve, path_lbl, title_str) in enumerate(
        [
            (
                axes5[0],
                ez_vh / norm_scale,
                bend_data["inclusions_valley"],
                w_arr["path_y"],
                "Valley domain-wall path",
                r"(a) Valley-Hall domain-wall $120^\circ$ Z-bend $|E_z|$",
            ),
            (
                axes5[1],
                ez_tr / norm_scale,
                bend_data["inclusions_trivial"],
                path_y_triv,
                "Trivial defect channel axis",
                r"(b) Tuned trivial PhC defect $120^\circ$ bend $|E_z|$",
            ),
        ]
    ):
        im5 = ax.pcolormesh(x_n, y_n, np.clip(field_abs.T, 0.0, 1.0), cmap="inferno", vmin=0.0, vmax=1.0, shading="auto")
        fig5.colorbar(im5, ax=ax, label=r"Normalized $|E_z(x, y)|$")
        # Overlay rod circles and waveguide path
        for inc in inc_list:
            ax.add_patch(
                mpatches.Circle((inc.x, inc.y), inc.radius, facecolor="none", edgecolor="#cbd5e1", lw=0.45, alpha=0.65)
            )
        ax.plot(w_arr["path_x"], py_curve, "--", color="#38bdf8", lw=1.6, label=path_lbl)
        # Mark CPML boundaries with overlay_geometry on a lightweight GridFIT3D
        g_ov = GridFIT3D(0.0, float(x_n[-1]), 0.0, float(y_n[-1]), 0.0, 0.1, 30, 21, 1, use_mpi=False, verbose=0)
        s_ov = SolverFIT3D(g_ov, bc_low=["cpml", "cpml", "periodic"], bc_high=["cpml", "cpml", "periodic"], n_pml=2, verbose=0)
        schematic.overlay_geometry(ax, s_ov, plane="xy", unit="m", label_materials=(idx_p == 0))
        ax.set_xlim(0.0, float(x_n[-1]))
        ax.set_ylim(0.0, float(y_n[-1]))
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [m] ($a = 1$ m)")
        ax.set_ylabel(r"$y$ [m]")
        ax.set_title(title_str)
        leg = ax.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.94)
        leg.set_frame_on(True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#cbd5e1")
        leg.get_frame().set_alpha(0.94)
        for txt in leg.get_texts():
            txt.set_color("#111827")

    fig5.tight_layout()
    p5 = FIG_DIR / "05_bend_field_maps.png"
    figstyle.save(fig5, p5)
    # Ensure legend box remains visible on dark inferno background after figstyle.save
    for ax in axes5:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(True)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("#cbd5e1")
            leg.get_frame().set_alpha(0.94)
            for txt in leg.get_texts():
                txt.set_color("#111827")
    fig5.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close(fig5)

    # ------------------------------------------------------------------
    # Figure 6 (GIF): Animated wavefront rounding the 120-deg Z-bend
    # ------------------------------------------------------------------
    snaps = w_arr["snapshots_valley_bend"]
    gif_path = FIG_DIR / "06_valley_hall_bend_wavefront.gif"
    if snaps.shape[0] > 0:
        fig_g, ax_g = plt.subplots(figsize=(6.4, 4.2), dpi=110)
        v_lim = max(float(np.percentile(np.abs(snaps[:, x_n > 2.4 * DEFAULT_A_M, :]), 99.0)), 1e-6)
        mesh_g = ax_g.pcolormesh(
            x_n, y_n, snaps[0].T, cmap="RdBu_r", vmin=-v_lim, vmax=v_lim, shading="auto"
        )
        fig_g.colorbar(mesh_g, ax=ax_g, label=r"$E_z(x, y, t)$ [arb. units]")
        for inc in bend_data["inclusions_valley"]:
            ax_g.add_patch(
                mpatches.Circle((inc.x, inc.y), inc.radius, facecolor="none", edgecolor="#475569", lw=0.4, alpha=0.55)
            )
        ax_g.plot(w_arr["path_x"], w_arr["path_y"], "k--", lw=1.2, label="Valley domain wall")
        ax_g.set_aspect("equal")
        ax_g.set_xlabel(r"$x / a$")
        ax_g.set_ylabel(r"$y / a$")
        title_g = ax_g.set_title("Valley-Hall domain-wall mode rounding 120-deg Z-bend (frame 1)")
        ax_g.legend(loc="lower right", fontsize=8.5)
        fig_g.tight_layout()

        def _update_frame(frame_idx: int):
            mesh_g.set_array(snaps[frame_idx].T.ravel())
            title_g.set_text(
                f"Valley-Hall domain-wall mode rounding 120-deg Z-bend (frame {frame_idx + 1}/{snaps.shape[0]})"
            )
            return (mesh_g, title_g)

        anim = animation.FuncAnimation(fig_g, _update_frame, frames=snaps.shape[0], interval=90, blit=False)
        anim.save(gif_path, writer=animation.PillowWriter(fps=11))
        plt.close(fig_g)

    return {
        "fig1": str(p1.relative_to(HERE)),
        "fig2": str(p2.relative_to(HERE)),
        "fig3": str(p3.relative_to(HERE)),
        "fig4": str(p4.relative_to(HERE)),
        "fig5": str(p5.relative_to(HERE)),
        "gif": str(gif_path.relative_to(HERE)),
    }


def build_quantitative_summary_table(results: dict[str, Any]) -> str:
    """Build the Markdown Quantitative Summary table from the saved JSON dictionary."""
    b = results["bulk_and_berry"]
    r = results["ribbon"]
    w = results["waveguide_bend"]
    c = results["convergence"]
    up_w = w["upper_single_mode_window"]
    gap_w = w["bulk_gap_window"]

    lines = [
        "| Quantity | Symbol | Theory / PWE Reference | 2D FIT Result | Relative Error / Comparison |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| Dirac point frequency ($\\delta = 0$) | $\\omega_D a / (2\\pi c)$ | `{b['dirac_freq_norm']:.4f}` | `{b['dirac_freq_norm']:.4f}` | Degeneracy split `{b['dirac_degeneracy_split']:.2e}` |",
        f"| Dirac group velocity ($\\delta = 0$) | $v_D / c$ | `{b['dirac_velocity_over_c']:.4f}` | — | Linear cone slope near $K$ |",
        f"| Supercell unfolding error | $\\max|\\omega_{{\\mathrm{{sup}}}} - \\omega_{{\\mathrm{{prim}}}}|$ | `0.0000` | `{b['unfolding_max_error']:.2e}` | Exact primitive-zone projection |",
        f"| Lower band edge at $K$ ($\\delta = 0.12$) | $\\omega_1(K) a / (2\\pi c)$ | `{b['gap_bottom_pwe']:.4f}` | `{b['fit_band1_K']:.4f}` | `{100.0 * b['fit_band1_rel_err']:.2f}%` (Richardson `{c['f1_K_richardson']:.4f}`) |",
        f"| Upper band edge at $K$ ($\\delta = 0.12$) | $\\omega_2(K) a / (2\\pi c)$ | `{b['gap_top_pwe']:.4f}` | `{b['fit_band2_K']:.4f}` | `{100.0 * b['fit_band2_rel_err']:.2f}%` |",
        f"| Valley-Hall bandgap width ($\\delta = 0.12$) | $\\Delta\\omega_K a / (2\\pi c)$ | `{b['gap_width_pwe']:.4f}` | `{b['fit_gap_width_K']:.4f}` | Midgap ratio `{100.0 * b['gap_midgap_ratio_pwe']:.2f}%` |",
        f"| Patch valley Chern number ($K, \\delta = +0.12$) | $C_K^{{\\mathrm{{patch}}}}(+\\delta)$ | `{b['c_patch_pwe_K_pos']:+.4f}` | `{b['c_patch_fit_K_pos']:+.4f}` | `{100.0 * b['c_patch_fit_rel_err_vs_pwe']:.2f}%` vs PWE (Dirac `{b['c_patch_dirac_K_pos']:+.4f}`) |",
        f"| Patch valley Chern number ($K, \\delta = -0.12$) | $C_K^{{\\mathrm{{patch}}}}(-\\delta)$ | `{b['c_patch_pwe_K_neg']:+.4f}` | `{b['c_patch_fit_K_neg']:+.4f}` | Sign flip $C(-\\delta) = -C(+\\delta)$; $\\delta=0$: `{b['c_patch_fit_K_zero']:+.2e}` |",
        f"| Interior-wall & trivial mode at $k_x = K_x$ | $\\omega(K_x) a / (2\\pi c)$ | Valley `{r['interior_wall_freq_at_K_norm']:.4f}` | Trivial `{r['trivial_freq_at_K_norm']:.4f}` | Confinement `{100.0 * r['edge_confinement_fraction']:.1f}%` vs `{100.0 * r['trivial_confinement_fraction']:.1f}%` |",
        f"| Straight reference transmission | $\\langle T_{{\\mathrm{{straight}}}} \\rangle$ | Valley `{w['mean_transmission_valley_straight']:.3f}` | Trivial `{w['mean_transmission_trivial_straight']:.3f}` | Matched straight reference ($T \\le 1$) |",
        f"| $120^\\circ$ bend transmission (upper single-mode $[{up_w[0]:.4f}, {up_w[1]:.4f}]$) | $\\langle T_{{\\mathrm{{bend}}}} \\rangle_{{\\mathrm{{up}}}}$ | Valley `{w['mean_transmission_valley_bend']:.3f}` | Trivial `{w['mean_transmission_trivial_bend']:.3f}` | Bend ratio `+{w['bend_transmission_advantage_db']:.2f} dB` |",
        f"| $120^\\circ$ bend transmission (full bulk gap $[{gap_w[0]:.4f}, {gap_w[1]:.4f}]$) | $\\langle T_{{\\mathrm{{bend}}}} \\rangle_{{\\mathrm{{gap}}}}$ | Valley `{w['mean_transmission_valley_bend_full_gap']:.3f}` | Trivial `{w['mean_transmission_trivial_bend_full_gap']:.3f}` | Full-gap ratio `+{w['bend_transmission_advantage_full_gap_db']:.2f} dB` |",
        f"| Grid convergence order | $p_{{\\mathrm{{obs}}}}$ | `2.00` | `{c['observed_order_p']:.2f}` | Finest grid `{c['cells_per_wavelength_finest']:.1f}` cells/$\\lambda$ |",
    ]
    return "\n".join(lines)


def run_all_simulations(quick: bool = False) -> dict[str, Any]:
    """Run all Tutorial 6.6 simulations sequentially, save JSON/NPZ, and render figures."""
    t_total = time.perf_counter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    setup_info = generate_setup_artifacts(quick=quick)
    bulk_res = run_bulk_and_berry_study(quick=quick)
    rib_res = run_ribbon_dispersion_study(
        quick=quick,
        bulk_gap=(
            float(bulk_res["metrics"]["gap_bottom_pwe"]),
            float(bulk_res["metrics"]["gap_top_pwe"]),
        ),
    )
    bend_res = run_waveguide_bend_comparison(quick=quick, rib_metrics=rib_res["metrics"])
    conv_res = run_convergence_study(quick=quick)
    fig_paths = plot_all_figures(bulk_res, rib_res, bend_res, conv_res)

    results = {
        "parameters": {
            "a_m": DEFAULT_A_M,
            "r0_over_a": DEFAULT_R0_OVER_A,
            "delta": DEFAULT_DELTA,
            "eps_rod": DEFAULT_EPS_ROD,
            "eps_bg": DEFAULT_EPS_BG,
        },
        "setup": setup_info,
        "bulk_and_berry": bulk_res["metrics"],
        "ribbon": rib_res["metrics"],
        "waveguide_bend": bend_res["metrics"],
        "convergence": conv_res["metrics"],
        "figures": fig_paths,
        "total_wall_seconds": float(time.perf_counter() - t_total),
    }
    results["summary_table_markdown"] = build_quantitative_summary_table(results)

    json_path = DATA_DIR / "valley_hall_results.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    npz_path = DATA_DIR / "valley_hall_arrays.npz"
    np.savez_compressed(
        npz_path,
        **bulk_res["arrays"],
        **rib_res["arrays"],
        **bend_res["arrays"],
    )
    return results
