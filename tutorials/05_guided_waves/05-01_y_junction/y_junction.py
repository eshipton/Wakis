# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# %% [markdown]
# # Tutorial 5.1 · Y-junction power splitter
#
# **Abstract.** Passive $1 \times 2$ optical power splitters are foundational
# building blocks of silicon photonic integrated circuits (PICs), feeding
# Mach–Zehnder modulators, phased arrays, and on-chip interferometers. Whereas
# conventional adiabatic Y-branches require tens of micrometres to suppress
# mode-mismatch radiation at the bifurcation tip, particle-swarm-optimized
# (PSO) multi-mode spline tapers achieve $< 0.3\;\text{dB}$ insertion loss in a
# $2\;\mu\text{m}$ footprint. In this tutorial we model the compact
# silicon-on-insulator (SOI) Y-junction of Zhang *et al.* (2013) in Wakis using
# the **2D effective-index method (EIM)** and **Kottke subpixel permittivity
# smoothing**, extract broadband $3\times 3$ S-parameters across the
# $1500\text{--}1600\;\text{nm}$ telecom window, verify passivity and energy
# conservation with a 2D Poynting flux box, and quantify the convergence
# advantage of Kottke smoothing over staircasing on curved waveguide boundaries.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Map a 3D silicon-on-insulator ($220\;\text{nm}$ $\text{Si}$ on
#    $\text{SiO}_2$) waveguide layout onto a **2D effective-index (EIM)**
#    finite-integration-technique (FIT) domain (`Ex`, `Ey`, `Hz`; one periodic
#    cell along $z$).
# 2. Construct smooth parametric PIC polygons (`spline_taper`, `sbend`,
#    `straight`) wrapped in `PolySlab`, and understand how `PolySlab.fill` and
#    `PolySlab.normal` drive diagonal **Kottke (2008) subpixel smoothing**.
# 3. Solve waveguide port eigenmodes (`solve_modes`), launch a broadband
#    fundamental $\text{TE}_0$ mode (`ModeSource`), and extract normalized
#    multi-port scattering parameters $S_{ij}(\lambda)$ (`ModeMonitor`).
# 4. Audit numerical symmetry ($|T_1 - T_2|$), insertion/excess loss, return
#    loss, and radiation loss via a 2D Poynting `FluxMonitor` box, and explain
#    why 2D EIM predictions differ from 3D FDTD and experiment.
#
# ## Contents
#
# 1. [Physical background & device specification](#1.-Physical-background-&-device-specification)
# 2. [Effective-Index Method (EIM) reduction](#2.-Effective-Index-Method-(EIM)-reduction)
# 3. [Parametric geometry construction](#3.-Parametric-geometry-construction)
# 4. [Subpixel permittivity smoothing vs staircasing](#4.-Subpixel-permittivity-smoothing-vs-staircasing)
# 5. [Study tasks & execution modes](#5.-Study-tasks-&-execution-modes)
# 6. [Generating the complete figure suite](#6.-Generating-the-complete-figure-suite)
# 7. [Broadband spectra, symmetry & energy conservation](#7.-Broadband-spectra,-symmetry-&-energy-conservation)
# 8. [Resolution convergence: Kottke vs staircase](#8.-Resolution-convergence:-Kottke-vs-staircase)
# 9. [Limits & when 2D EIM breaks down](#9.-Limits-&-when-2D-EIM-breaks-down)
# 10. [Exercises](#10.-Exercises)
# 11. [References](#11.-References)

# %% [markdown]
# ## 1. Physical background & device specification
#
# In a symmetric $1 \times 2$ Y-junction driven in the fundamental symmetric
# mode $\text{TE}_0$ at port 1 (`in`), mirror symmetry $y \mapsto -y$ dictates
# that the scattering matrix elements into the upper arm (port 2, `out1`) and
# lower arm (port 3, `out2`) are identical:
#
# $$S_{21}(\omega) = S_{31}(\omega), \qquad T_1(\omega) = |S_{21}(\omega)|^2 = T_2(\omega) = |S_{31}(\omega)|^2.$$
#
# An ideal lossless, reflectionless splitter achieves $T_1 = T_2 = 1/2$
# ($-3.010\;\text{dB}$ per arm). In a practical submicron SOI splitter, however,
# two loss channels compete:
#
# 1. **Blunt splitter tip gap ($g_{\text{tip}} \approx 200\;\text{nm}$)**: CMOS
#    lithography cannot resolve an infinitely sharp vertex between the two
#    output branches. A sudden step discontinuity at the bifurcation excites
#    unguided radiation modes and backward reflections ($R = |S_{11}|^2$).
# 2. **Bending radiation in the output arms**: The two $500\;\text{nm}$ output
#    waveguides separate laterally through S-bends to decouple their evanescent
#    tails.
#
# **Zhang *et al.* (2013)** [1] mitigated the tip mismatch by inserting a
# $L_{\text{junc}} = 2.0\;\mu\text{m}$ multi-mode section between the
# $w_{\text{wg}} = 500\;\text{nm}$ input waveguide and the two S-bend output
# branches. The width profile $w(x)$ is parameterized by **13 equally spaced
# control knots** $w_0, \dots, w_{12}$ along $x \in [0, 2.0]\;\mu\text{m}$
# connected by a natural cubic spline:
#
# $$w_{0..12} = [0.50, 0.50, 0.60, 0.70, 0.90, 1.26, 1.40, 1.40, 1.40, 1.40, 1.31, 1.20, 1.20]\;\mu\text{m}.$$
#
# The non-monotonic expansion to $1.40\;\mu\text{m}$ followed by a slight
# contraction to $1.20\;\mu\text{m}$ excites both the $\text{TE}_0$ and
# symmetric $\text{TE}_2$ slab-guided modes in the junction region; their
# constructive interference at $x = 2.0\;\mu\text{m}$ forms a double-peaked
# transverse field profile with a local minimum on the axis ($y = 0$), minimizing
# scattering at the $200\;\text{nm}$ oxide gap between the two output S-bends
# centered at $y = \pm 0.35\;\mu\text{m}$.

# %%
import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.constants import c as c_light
from scipy.interpolate import CubicSpline

from wakis.templates.photonics import (
    PIC_MATERIALS,
    PICSimulation,
    PolySlab,
    Port,
    make_movie,
    n_to_eps,
    plot_convergence,
    plot_cross_section_mesh,
    plot_eim_stack,
    plot_field_map,
    plot_layout,
    plot_neff_dispersion,
    plot_port_modes,
    plot_spectra_db,
    sbend,
    slab_neff,
    spline_taper,
    straight,
    vertical_slab_mode,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# %% [markdown]
# ### Device parameters from Zhang *et al.* (2013)
#
# All geometric dimensions are specified in metres (SI units):
#
# | Parameter | Symbol | Value |
# |---|---|---|
# | Silicon core refractive index ($1550\;\text{nm}$) | $n_{\text{Si}}$ | $3.476$ (Li 1980 [4]) |
# | $\text{SiO}_2$ cladding refractive index ($1550\;\text{nm}$) | $n_{\text{SiO}_2}$ | $1.444$ (Malitson 1965 [5]) |
# | Silicon layer thickness | $h_{\text{Si}}$ | $220\;\text{nm}$ |
# | Input / output waveguide width | $w_{\text{wg}}$ | $500\;\text{nm}$ |
# | Spline junction length | $L_{\text{junc}}$ | $2.0\;\mu\text{m}$ |
# | Junction output width ($x = 2.0\;\mu\text{m}$) | $w_{12}$ | $1.20\;\mu\text{m}$ |
# | Bifurcation tip gap at $x = 2.0\;\mu\text{m}$ | $g_{\text{tip}}$ | $200\;\text{nm}$ |
# | S-bend initial arm center ($x = 2.0\;\mu\text{m}$) | $\pm y_0$ | $\pm 0.35\;\mu\text{m}$ |
# | S-bend axial length / final arm center | $L_{\text{sb}}, \pm y_{\text{out}}$ | $2.0\;\mu\text{m}, \pm 0.90\;\mu\text{m}$ |

# %%
# Material refractive indices at 1550 nm
N_SI = PIC_MATERIALS["Si"]  # 3.476
N_SIO2 = PIC_MATERIALS["SiO2"]  # 1.444
H_SI = 220e-9  # 220 nm SOI thickness

# Wavelength band: 1500 nm to 1600 nm
WL_MIN = 1500e-9
WL_MAX = 1600e-9
WL_REF = 1550e-9
N_FREQS = 51

# Zhang et al. (2013), Opt. Express 21(1), 1310-1316, Table / Fig. 1 parameters
W_WG = 0.50e-6
L_IN = 1.40e-6
L_JUNC = 2.00e-6
L_SBEND = 2.00e-6
L_OUT = 1.20e-6
Y0_ARM = 0.35e-6  # (1.20 um / 2) - (0.50 um / 2) = 0.35 um -> 200 nm tip gap
DY_ARM = 0.55e-6  # S-bend offset to y_out = +-0.90 um
Y_OUT = Y0_ARM + DY_ARM  # 0.90 um

W_KNOTS_UM = np.array(
    [0.50, 0.50, 0.60, 0.70, 0.90, 1.26, 1.40, 1.40, 1.40, 1.40, 1.31, 1.20, 1.20]
)
W_KNOTS_M = W_KNOTS_UM * 1e-6

# Simulation domain (x_min, x_max, y_min, y_max) [m] and port coordinates
DOMAIN_2D = (-1.15e-6, 4.85e-6, -1.85e-6, 1.85e-6)
X_PORT_IN = -0.80e-6
X_PORT_OUT = 4.45e-6

# %% [markdown]
# ## 2. Effective-Index Method (EIM) reduction
#
# A full 3D FDTD/FIT simulation of a $220\;\text{nm}$ SOI device resolves both
# the sub-wavelength vertical confinement and the in-plane propagation. In the
# **2D effective-index method (EIM)**, we factor the 3D quasi-TE mode
# $\mathbf{E}(x, y, z) \approx \psi(z)\,\mathbf{E}_{\parallel}(x, y)$ by first
# solving the 1D vertical slab waveguide problem along $z$:
#
# $$\frac{\mathrm{d}^2 \psi}{\mathrm{d} z^2} + k_0^2 \left[n^2(z) - n_{\text{eff,slab}}^2\right]\psi(z) = 0.$$
#
# Inside the etched silicon pattern, the vertical stack is a symmetric
# $\text{SiO}_2 / \text{Si}(220\;\text{nm}) / \text{SiO}_2$ slab whose
# fundamental $\text{TE}_0$ mode at $\lambda_0 = 1550\;\text{nm}$ has effective
# index:
#
# $$n_{\text{eff,slab}}(1550\;\text{nm}) = 2.8449.$$
#
# Outside the silicon pattern, the vertical stack is homogeneous $\text{SiO}_2$
# cladding with index $n_{\text{clad}} = 1.444$. The 3D quasi-TE problem (dominant
# in-plane electric field, vertical magnetic field $H_z$) therefore maps onto a
# **2D in-plane FIT simulation** with electric components $(E_x, E_y)$ and
# magnetic component $H_z$, using a single periodic cell along $z$ (`ieps_z = 0`)
# and constant core index $n_{\text{eff,slab}}(1550\;\text{nm}) = 2.8449$ across
# the $1500\text{--}1600\;\text{nm}$ band.

# %% [markdown]
# ## 3. Parametric geometry construction
#
# We construct the Y-junction using `spline_taper`, `sbend`, and `straight` from
# `wakis.templates.photonics`. Combining the boundary chains of the input
# waveguide, the 13-knot spline junction, the two cosine S-bend arms, and the
# straight output waveguides into a single closed counter-clockwise `PolySlab`
# eliminates internal artificial polygon seams while preserving exact
# mirror symmetry ($y \mapsto -y$) to machine precision.


# %%
def build_y_junction(n_taper=160, n_sbend=100):
    """
    Build the Zhang et al. (2013) Y-junction as a single seamless `PolySlab`
    along with its three named waveguide `Port` specifications.
    """
    x_in = -L_IN
    x_sb_end = L_JUNC + L_SBEND
    x_out = x_sb_end + L_OUT
    hw = 0.5 * W_WG

    taper_v = spline_taper(L_JUNC, W_KNOTS_M, x0=0.0, y0=0.0, n=n_taper)
    taper_bot = taper_v[:n_taper]
    taper_top = taper_v[n_taper:]

    sb_bot_v = sbend(
        (L_JUNC, -Y0_ARM),
        dx=L_SBEND,
        dy=-DY_ARM,
        width=W_WG,
        profile="cosine",
        n=n_sbend,
    )
    sb_bot_outer = sb_bot_v[:n_sbend]
    sb_bot_inner = sb_bot_v[n_sbend:]

    sb_top_v = sbend(
        (L_JUNC, +Y0_ARM),
        dx=L_SBEND,
        dy=+DY_ARM,
        width=W_WG,
        profile="cosine",
        n=n_sbend,
    )
    sb_top_inner = sb_top_v[:n_sbend]
    sb_top_outer = sb_top_v[n_sbend:]

    verts = np.vstack(
        [
            [[x_in, -hw]],
            taper_bot,
            sb_bot_outer[1:],
            [[x_out, -Y_OUT - hw], [x_out, -Y_OUT + hw]],
            sb_bot_inner,
            sb_top_inner,
            [[x_out, +Y_OUT - hw], [x_out, +Y_OUT + hw]],
            sb_top_outer,
            taper_top[1:],
            [[x_in, +hw]],
        ]
    )
    slab = PolySlab(
        verts,
        z_min=-0.5 * H_SI,
        z_max=0.5 * H_SI,
        material=n_to_eps(N_SI),
    )
    ports = [
        Port(
            "in", center=(X_PORT_IN, 0.0, 0.0), size=(0.0, 1.8e-6, 0.8e-6), direction=+1
        ),
        Port(
            "out1",
            center=(X_PORT_OUT, +Y_OUT, 0.0),
            size=(0.0, 1.5e-6, 0.8e-6),
            direction=+1,
        ),
        Port(
            "out2",
            center=(X_PORT_OUT, -Y_OUT, 0.0),
            size=(0.0, 1.5e-6, 0.8e-6),
            direction=+1,
        ),
    ]
    return [slab], ports


# %% [markdown]
# ## 4. Subpixel permittivity smoothing vs staircasing
#
# Because the spline taper and cosine S-bends curve continuously across the
# Cartesian Yee grid, staircasing (rounding each cell's fill fraction $w \in [0, 1]$
# to $0$ or $1$) turns smooth sidewalls into a series of right-angle steps.
# Those steps act as parasitic Bragg scatterers that cause first-order
# $\mathcal{O}(\Delta x)$ phase errors and spurious back-reflection $|S_{11}|^2$.
# In the full study (Fig. 12) the transmission error shows this clearly: staircasing
# stalls near $10^{-3}$ while Kottke keeps converging. The back-reflection of this
# device is already near $-30$ dB at every resolution, where port/PML residuals of
# the same size dominate, so it does not show a clean grid trend.
#
# Wakis implements the diagonal part of **Kottke, Farjadpour & Johnson (2008)**
# [2] in `wakis.templates.shapes.assign_materials`: for each electric-field edge
# component $c \in \{x, y, z\}$, the dual-cell fill fraction $w$ (`PolySlab.fill`)
# and unit outward boundary normal $\mathbf{n}$ (`PolySlab.normal`) blend the
# shape permittivity $\varepsilon_s$ and background permittivity $\varepsilon_b$
# harmonically along the normal direction and arithmetically along the
# tangential directions:
#
# $$\varepsilon_{\text{eff}, c}^{-1} = n_c^2 \left(\frac{w}{\varepsilon_s} + \frac{1 - w}{\varepsilon_b}\right) + \left(1 - n_c^2\right)\left(w\varepsilon_s + (1 - w)\varepsilon_b\right)^{-1}.$$

# %% [markdown]
# ## 5. Study tasks & execution modes
#
# To compare Kottke subpixel smoothing against staircasing across three mesh
# resolutions, we define **7 study tasks** (`task_0` through `task_6`). Task 6 is an
# independent, much finer Kottke run that serves as the reference $T_{\mathrm{ref}}$
# for the convergence plot (an extrapolation from the compared runs themselves would
# presuppose their convergence order):
#
# | Task ID | Name | Averaging | `--quick` resolution (cpw) | `--full` resolution (cpw) |
# |---|---|---|---|---|
# | `0` | `kottke_coarse` | `kottke` | $8$ | $14$ |
# | `1` | `staircase_coarse` | `staircase` | $8$ | $14$ |
# | `2` | `kottke_medium` | `kottke` | $10$ | $20$ |
# | `3` | `staircase_medium` | `staircase` | $10$ | $20$ |
# | `4` | `kottke_fine` | `kottke` | $12$ (+ fields & movie) | $28$ (+ fields & movie) |
# | `5` | `staircase_fine` | `staircase` | $12$ | $28$ |
# | `6` | `kottke_reference` | `kottke` | $24$ | $56$ |


# %%
def get_task_table(quick=False):
    """Return the 7-point `(averaging, cells_per_wavelength)` study table (6 compared runs + reference)."""
    cpws = (8, 10, 12) if quick else (14, 20, 28)
    levels = ("coarse", "medium", "fine")
    tasks = []
    for lvl, cpw in zip(levels, cpws):
        for avg in ("kottke", "staircase"):
            idx = len(tasks)
            is_primary = lvl == "fine" and avg == "kottke"
            tasks.append(
                {
                    "index": idx,
                    "name": f"{avg}_{lvl}",
                    "level": lvl,
                    "averaging": avg,
                    "cpw": cpw,
                    "n_pml": 8 if quick else 10,
                    "energy_decay": 5e-3 if quick else 1e-4,
                    "collect_fields": is_primary,
                    "record_movie": is_primary,
                    "movie_frames": 32 if quick else 48,
                    "description": (
                        f"2D EIM Y-junction ({avg}, {lvl} grid, cpw={cpw}"
                        + (", DFT fields + movie + flux box)" if is_primary else ")")
                    ),
                }
            )
    ref_cpw = 24 if quick else 56
    tasks.append(
        {
            "index": len(tasks),
            "name": "kottke_reference",
            "level": "reference",
            "averaging": "kottke",
            "cpw": ref_cpw,
            "n_pml": 8 if quick else 10,
            "energy_decay": 5e-3 if quick else 1e-4,
            "collect_fields": False,
            "record_movie": False,
            "movie_frames": 0,
            "description": f"2D EIM Y-junction (kottke, reference grid, cpw={ref_cpw})",
        }
    )
    return tasks


def run_single_task(task_cfg, out_dir):
    """
    Execute one 2D EIM Y-junction study task and write ``task_<I>.npz`` and
    ``task_<I>.json`` into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = task_cfg["index"]
    t0 = time.perf_counter()

    structures, ports = build_y_junction()
    use_gpu = os.environ.get("WAKIS_USE_GPU", "0") == "1"
    pic = PICSimulation(
        wavelength_min=WL_MIN,
        wavelength_max=WL_MAX,
        n_freqs=N_FREQS,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        wavelength_ref=WL_REF,
        two_d=True,
        averaging=task_cfg["averaging"],
        cells_per_wavelength=task_cfg["cpw"],
        domain=DOMAIN_2D,
        n_pml=task_cfg["n_pml"],
        field_wavelengths=(WL_REF,),
        record_movie=task_cfg["record_movie"],
        movie_frames=task_cfg["movie_frames"],
        use_gpu=use_gpu,
    )

    s_mat = pic.run_s_parameters(
        drive=["in"],
        energy_decay=task_cfg["energy_decay"],
        collect_fields=task_cfg["collect_fields"],
    )

    payload = {
        "task_index": np.int32(idx),
        "name": np.array(task_cfg["name"]),
        "level": np.array(task_cfg["level"]),
        "averaging": np.array(task_cfg["averaging"]),
        "cpw": np.float32(task_cfg["cpw"]),
        "n_eff_slab": np.float64(pic.n_eff_slab),
        "frequencies": pic.frequencies.astype(np.float64),
        "wavelengths": pic.wavelengths.astype(np.float64),
        "S11": s_mat[("in", "in")].astype(np.complex128),
        "S21": s_mat[("out1", "in")].astype(np.complex128),
        "S31": s_mat[("out2", "in")].astype(np.complex128),
    }

    if task_cfg["collect_fields"]:
        # Solve port modes at 1550 nm and port dispersion across 11 wavelengths
        modes_in = pic.solve_port_modes("in", frequencies=c_light / WL_REF, n_modes=1)
        modes_out = pic.solve_port_modes(
            "out1", frequencies=c_light / WL_REF, n_modes=1
        )
        y_in = modes_in[0].coords["Eb"][0]
        ey_in = modes_in[0].e["b"][:, 0]
        m_in = np.abs(y_in) <= 0.9e-6
        y_out = modes_out[0].coords["Eb"][0]
        ey_out = modes_out[0].e["b"][:, 0]
        m_out = np.abs(y_out - Y_OUT) <= 0.75e-6
        payload["port_y_in"] = y_in[m_in].astype(np.float32)
        payload["port_ey_in"] = ey_in[m_in].astype(np.complex64)
        payload["port_neff_in"] = np.float64(np.real(modes_in[0].neff))
        payload["port_y_out"] = y_out[m_out].astype(np.float32)
        payload["port_ey_out"] = ey_out[m_out].astype(np.complex64)
        payload["port_neff_out"] = np.float64(np.real(modes_out[0].neff))

        wl_disp = np.linspace(WL_MIN, WL_MAX, 11)
        f_disp = c_light / wl_disp
        disp_modes = pic.solve_port_modes("in", frequencies=f_disp, n_modes=1)
        neff_disp = np.array([float(np.real(m[0].neff)) for m in disp_modes])
        dneff_dl = np.gradient(neff_disp, wl_disp)
        ng_disp = neff_disp - wl_disp * dneff_dl
        payload["disp_wavelengths"] = wl_disp.astype(np.float64)
        payload["disp_neff"] = neff_disp.astype(np.float64)
        payload["disp_ng"] = ng_disp.astype(np.float64)

        if WL_REF in pic.dft_fields:
            fmap = pic.dft_fields[WL_REF]
            payload["dft_x"] = fmap["x"]
            payload["dft_y"] = fmap["y"]
            payload["dft_Ey"] = fmap["Ey"]
            payload["dft_E2"] = fmap["E2"]

        if pic.movie_data is not None:
            payload["movie_x"] = pic.movie_data["x"]
            payload["movie_y"] = pic.movie_data["y"]
            payload["movie_times"] = pic.movie_data["times"]
            payload["movie_frames"] = pic.movie_data["frames"]
            payload["movie_peak"] = pic.movie_data["peak"]

        if pic.port_time_trace is not None:
            payload["trace_t"] = pic.port_time_trace["t"]
            payload["trace_Ey"] = pic.port_time_trace["Ey"]

        if pic.flux_balance is not None:
            payload["flux_east"] = pic.flux_balance["east"].astype(np.float64)
            payload["flux_west"] = pic.flux_balance["west"].astype(np.float64)
            payload["flux_lateral"] = pic.flux_balance["lateral"].astype(np.float64)
            payload["flux_total_box"] = pic.flux_balance["total_box"].astype(np.float64)

    npz_path = out_dir / f"task_{idx}.npz"
    np.savez_compressed(npz_path, **payload)

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta = {
        "task_index": idx,
        "name": task_cfg["name"],
        "averaging": task_cfg["averaging"],
        "cpw": task_cfg["cpw"],
        "grid_shape": [int(pic.grid.Nx), int(pic.grid.Ny), int(pic.grid.Nz)],
        "n_cells": int(pic.grid.Nx * pic.grid.Ny * pic.grid.Nz),
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
        "npz_bytes": int(npz_path.stat().st_size),
    }
    (out_dir / f"task_{idx}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[task_{idx} {task_cfg['name']}] grid={meta['grid_shape']} "
        f"time={elapsed:.2f}s rss={rss_mb:.1f}MB size={meta['npz_bytes'] / 1024:.1f}KB"
    )
    return meta


# %% [markdown]
# ## 6. Generating the complete figure suite
#
# When invoked with `--plot` (or at the end of `--quick` / `--full`),
# `generate_all_figures` reads `data/task_*.npz` and writes 14 diagnostic and
# validation figures to `figures/`:
#
# 1. `01_y_junction_layout.png` — Annotated top-view layout of the Zhang *et al.* (2013) Y-junction.
# 2. `02_eim_vertical_stack.png` — Vertical $220\;\text{nm}$ SOI stack, 1D $\text{TE}_0$ mode profile, and $n_{\text{eff,slab}}(\lambda)$.
# 3. `03_junction_width_profile.png` — The 13 PSO spline width control knots $w(x)$ and slope $\mathrm{d}w/\mathrm{d}x$.
# 4. `04_permittivity_mesh_zoom.png` — Zoomed in-plane permittivity map with Yee mesh lines: staircase vs Kottke side by side.
# 5. `05_port_mode_profiles.png` — Input and output port eigenmode profiles $|E_y(y)|$ and $\mathrm{Re}(E_y(y))$.
# 6. `06_port_neff_dispersion.png` — Channel waveguide effective index $n_{\text{eff}}(\lambda)$ and group index $n_g(\lambda)$.
# 7. `07_field_intensity_1550.png` — Steady-state logarithmic intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $1550\;\text{nm}$.
# 8. `08_field_real_ey_1550.png` — Steady-state wave-front map $\mathrm{Re}(E_y)$ at $1550\;\text{nm}$.
# 9. `09_pulse_propagation.gif` — Time-domain pulse propagation movie through the Y-junction.
# 10. `10_sparameter_spectra_db.png` — Broadband transmission ($S_{21}, S_{31}$) and reflection ($S_{11}$) spectra in dB.
# 11. `11_excess_loss_and_energy_budget.png` — Excess insertion loss vs Zhang *et al.* (2013), 2D flux-box radiation budget, and symmetry error $|T_1 - T_2|$.
# 12. `12_kottke_vs_staircase_convergence.png` — Subpixel convergence of Kottke vs staircasing across mesh resolutions.
# 13. `13_resolution_spectra_comparison.png` — Broadband excess transmission and return loss at all 3 resolutions on one figure.
# 14. `14_port_time_trace.png` — Output port time-domain waveform $E_y(t)$ and exponential energy decay.


# %%
def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Read ``data_dir/task_*.npz`` and render all 14 figures into ``fig_dir``."""
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tasks_data = {}
    for p in sorted(data_dir.glob("task_*.npz")):
        d = dict(np.load(p, allow_pickle=False))
        name = str(d["name"])
        tasks_data[name] = d

    if "kottke_fine" not in tasks_data:
        raise FileNotFoundError(
            f"Missing primary task 'kottke_fine' in {data_dir}. Run --quick or --full first."
        )
    primary = tasks_data["kottke_fine"]
    structures, ports = build_y_junction()

    # 1. Top-view device layout with dimensional annotations
    annotations = [
        {
            "text": "Input WG\nw = 500 nm",
            "xy": (-0.30, -0.25),
            "xytext": (-0.30, -1.05),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
        {
            "text": "13-pt PSO spline taper\nL = 2.0 µm, wmax = 1.40 µm",
            "xy": (1.0, 0.70),
            "xytext": (1.0, 1.38),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
        {
            "text": "200 nm tip gap\n(x = 2.0 µm)",
            "xy": (2.0, 0.0),
            "xytext": (2.55, 0.0),
            "arrowprops": dict(arrowstyle="->", color="#d95f02", lw=1.2),
        },
        {
            "text": "Cosine S-bends\nL = 2.0 µm, y = ±0.90 µm",
            "xy": (3.1, -0.70),
            "xytext": (2.9, -1.48),
            "arrowprops": dict(arrowstyle="->", color="0.3", lw=1.0),
        },
    ]
    plot_layout(
        structures,
        ports=ports,
        domain=DOMAIN_2D,
        annotations=annotations,
        title="Zhang et al. (2013) compact SOI Y-junction layout (2D EIM domain)",
        out_path=fig_dir / "01_y_junction_layout.png",
    )

    # 2. Effective-Index Method (EIM) vertical stack, TE0 slab mode, and dispersion
    plot_eim_stack(
        n_core=N_SI,
        n_clad=N_SIO2,
        thickness=H_SI,
        wavelength_ref=WL_REF,
        wavelengths=np.linspace(WL_MIN, WL_MAX, 51),
        out_path=fig_dir / "02_eim_vertical_stack.png",
    )

    # 3. Spline width profile w(x) and slope dw/dx vs linear taper
    x_knots = np.linspace(0.0, L_JUNC * 1e6, len(W_KNOTS_UM))
    cs_um = CubicSpline(x_knots, W_KNOTS_UM)
    x_fine = np.linspace(0.0, L_JUNC * 1e6, 300)
    w_fine = cs_um(x_fine)
    dw_dx = cs_um(x_fine, 1)
    w_lin = W_KNOTS_UM[0] + (W_KNOTS_UM[-1] - W_KNOTS_UM[0]) * (x_fine / (L_JUNC * 1e6))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.2, 3.8), dpi=150)
    ax0.plot(x_fine, w_fine, color="#2b5c8f", lw=2.2, label="Cubic-spline taper $w(x)$")
    ax0.plot(
        x_knots,
        W_KNOTS_UM,
        "o",
        color="#d95f02",
        ms=5.5,
        label="13 PSO knots (Zhang 2013)",
    )
    ax0.plot(
        x_fine, w_lin, "--", color="0.5", lw=1.5, label="Linear taper (0.50 → 1.20 µm)"
    )
    ax0.set(
        xlabel="Axial position x along junction [µm]",
        ylabel="Waveguide width w(x) [µm]",
        title="(a) Junction width profile w(x)",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=8.5)

    ax1.plot(
        x_fine,
        dw_dx,
        color="#1b9e77",
        lw=2.0,
        label=r"Spline slope $\mathrm{d}w/\mathrm{d}x$",
    )
    ax1.axhline(0.0, color="0.4", ls=":", lw=1.2)
    ax1.axhline(
        (W_KNOTS_UM[-1] - W_KNOTS_UM[0]) / (L_JUNC * 1e6),
        color="0.5",
        ls="--",
        lw=1.4,
        label="Linear taper slope (+0.35)",
    )
    ax1.set(
        xlabel="Axial position x along junction [µm]",
        ylabel=r"Taper flare rate $\mathrm{d}w/\mathrm{d}x$ [rad]",
        title="(b) Local flare rate (expansion then contraction)",
    )
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_junction_width_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. In-plane permittivity map with mesh lines zoomed on the junction tip: staircase vs Kottke
    cpw_mesh = float(tasks_data.get("kottke_medium", primary)["cpw"])
    pic_stair = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        two_d=True,
        averaging="staircase",
        cells_per_wavelength=cpw_mesh,
        domain=DOMAIN_2D,
        n_pml=8,
    )
    pic_kottke = PICSimulation(
        WL_MIN,
        WL_MAX,
        structures=structures,
        ports=ports,
        n_core=N_SI,
        n_clad=N_SIO2,
        two_d=True,
        averaging="kottke",
        cells_per_wavelength=cpw_mesh,
        domain=DOMAIN_2D,
        n_pml=8,
    )
    plot_cross_section_mesh(
        pic_stair.solver,
        pic_kottke.solver,
        xlim_um=(0.2, 2.7),
        ylim_um=(-0.85, 0.85),
        component="y",
        out_path=fig_dir / "04_permittivity_mesh_zoom.png",
    )

    # 5. Port mode profiles at input and upper output arm
    plot_port_modes(
        primary["port_y_in"] * 1e6,
        primary["port_ey_in"],
        float(primary["port_neff_in"]),
        primary["port_y_out"] * 1e6,
        primary["port_ey_out"],
        float(primary["port_neff_out"]),
        out_path=fig_dir / "05_port_mode_profiles.png",
    )

    # 6. Waveguide mode effective index and group index dispersion
    plot_neff_dispersion(
        primary["disp_wavelengths"] * 1e9,
        primary["disp_neff"],
        neff_slab=float(primary["n_eff_slab"]),
        ng_2d=primary["disp_ng"],
        out_path=fig_dir / "06_port_neff_dispersion.png",
    )

    # 7. Steady-state logarithmic intensity map |E|^2 [dB] at 1550 nm
    plot_field_map(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_E2"],
        structures=structures,
        kind="intensity_log",
        title=r"Steady-state modal intensity $10\log_{10}(|E|^2/|E|_{\max}^2)$ at $\lambda = 1550\;\mathrm{nm}$",
        out_path=fig_dir / "07_field_intensity_1550.png",
    )

    # 8. Steady-state real field map Re(Ey) at 1550 nm
    plot_field_map(
        primary["dft_x"],
        primary["dft_y"],
        primary["dft_Ey"],
        structures=structures,
        kind="real",
        title=r"Steady-state wave-front phase pattern $\mathrm{Re}(E_y)$ at $\lambda = 1550\;\mathrm{nm}$",
        out_path=fig_dir / "08_field_real_ey_1550.png",
    )

    # 9. Animated pulse propagation GIF
    if "movie_frames" in primary and len(primary["movie_frames"]) > 0:
        make_movie(
            primary["movie_x"],
            primary["movie_y"],
            primary["movie_frames"],
            primary["movie_times"],
            structures=structures,
            out_path=fig_dir / "09_pulse_propagation.gif",
            fps=12,
        )

    # Sort wavelengths ascending for spectral plots (interior of pulse band: 1505-1595 nm)
    wl_nm_all = primary["wavelengths"] * 1e9
    order = np.argsort(wl_nm_all)
    wl_nm = wl_nm_all[order]
    mask = (wl_nm >= 1505.0) & (wl_nm <= 1595.0)
    wl_b = wl_nm[mask]

    s11 = primary["S11"][order][mask]
    s21 = primary["S21"][order][mask]
    s31 = primary["S31"][order][mask]
    t1 = np.abs(s21) ** 2
    t2 = np.abs(s31) ** 2
    r = np.abs(s11) ** 2

    # 10. S-parameter spectra in dB (T1, T2, R)
    plot_spectra_db(
        wl_b,
        {
            r"$T_1 = |S_{21}|^2$ (upper arm)": 10.0 * np.log10(np.maximum(t1, 1e-12)),
            r"$T_2 = |S_{31}|^2$ (lower arm)": 10.0 * np.log10(np.maximum(t2, 1e-12)),
            r"$R = |S_{11}|^2$ (return loss)": 10.0 * np.log10(np.maximum(r, 1e-12)),
        },
        title="Y-junction S-parameter spectra across the 1505–1595 nm band (2D EIM, Kottke)",
        ylabel="Modal power [dB]",
        reference_lines=[
            {
                "y": -3.0103,
                "color": "0.35",
                "ls": ":",
                "label": "Ideal 50/50 split (−3.01 dB)",
            }
        ],
        out_path=fig_dir / "10_sparameter_spectra_db.png",
    )

    # 11. Excess insertion loss vs Zhang (2013), energy conservation & symmetry check
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.1), dpi=150)
    t_tot_db = 10.0 * np.log10(np.maximum(t1 + t2, 1e-12))
    ax0.plot(
        wl_b,
        t_tot_db,
        color="#1b9e77",
        lw=2.2,
        label=r"Guided total $10\log_{10}(T_1 + T_2)$ (2D EIM)",
    )
    ax0.axhline(
        -0.13,
        color="#2b5c8f",
        ls="--",
        lw=1.7,
        label="Zhang et al. (2013) 3D FDTD (−0.13 dB)",
    )
    ax0.axhline(
        -0.28,
        color="#d95f02",
        ls="-.",
        lw=1.7,
        label="Zhang et al. (2013) measured (−0.28 ± 0.02 dB)",
    )
    ax0.fill_between(wl_b, -0.30, -0.26, color="#d95f02", alpha=0.15)
    if "flux_total_box" in primary:
        f_box = primary["flux_total_box"][order][mask]
        f_lat = primary["flux_lateral"][order][mask]
        ax0.plot(
            wl_b,
            10.0 * np.log10(np.maximum(f_box, 1e-12)),
            color="0.3",
            ls=":",
            lw=1.6,
            label=f"2D Poynting flux box total (mean={np.mean(f_box):.3f}, lat={np.mean(f_lat) * 100:.1f}%)",
        )
    ax0.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Normalized transmitted power [dB]",
        ylim=(-0.85, 0.15),
        title="(a) Excess transmission vs Zhang et al. (2013) & flux budget",
    )
    ax0.grid(True, ls=":", alpha=0.4)
    ax0.legend(loc="lower right", fontsize=8.0)

    sym_err = np.maximum(np.abs(t1 - t2), 1e-16)
    phase_err = np.maximum(np.abs(np.angle(s21 * np.conj(s31))), 1e-16)
    ax1.semilogy(
        wl_b, sym_err, color="#2b5c8f", lw=2.0, label=r"Power imbalance $|T_1 - T_2|$"
    )
    ax1.semilogy(
        wl_b,
        phase_err,
        color="#d95f02",
        ls="--",
        lw=1.8,
        label=r"Phase imbalance $|\arg(S_{21}) - \arg(S_{31})|$ [rad]",
    )
    ax1.semilogy(
        wl_b,
        r,
        color="#7570b3",
        ls="-.",
        lw=1.8,
        label=r"Return loss $R = |S_{11}|^2$",
    )
    ax1.set(
        xlabel="Wavelength λ [nm]",
        ylabel="Dimensionless magnitude (log scale)",
        title="(b) Mirror symmetry audit & return loss",
    )
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(loc="best", fontsize=8.0)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "11_excess_loss_and_energy_budget.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # 12 & 13. Multi-resolution convergence & 3-resolution spectral comparison
    levels = ["coarse", "medium", "fine"]
    avail = all(
        f"{avg}_{lvl}" in tasks_data
        for avg in ("kottke", "staircase")
        for lvl in levels
    )
    if avail:
        cpws = [float(tasks_data[f"kottke_{lvl}"]["cpw"]) for lvl in levels]

        # Reference at 1550 nm: the independent kottke_reference run (task 6)
        def _t_tot_at_1550(d):
            w = d["wavelengths"]
            i0 = int(np.argmin(np.abs(w - WL_REF)))
            return float(np.abs(d["S21"][i0]) ** 2 + np.abs(d["S31"][i0]) ** 2)

        def _r_mean(d):
            w = d["wavelengths"] * 1e9
            m = (w >= 1515.0) & (w <= 1585.0)
            return float(np.mean(np.abs(d["S11"][m]) ** 2))

        tk_vals = [_t_tot_at_1550(tasks_data[f"kottke_{lvl}"]) for lvl in levels]
        ts_vals = [_t_tot_at_1550(tasks_data[f"staircase_{lvl}"]) for lvl in levels]
        if "kottke_reference" in tasks_data:
            t_ref = _t_tot_at_1550(tasks_data["kottke_reference"])
        else:  # fallback only: Richardson from the compared runs presupposes 2nd order
            print("WARNING: no kottke_reference task; T_ref is a Richardson estimate")
            r_ratio = (cpws[2] / cpws[1]) ** 2
            t_ref = tk_vals[2] + (tk_vals[2] - tk_vals[1]) / max(r_ratio - 1.0, 0.25)

        err_k = [max(abs(v - t_ref), 2e-4) for v in tk_vals]
        err_s = [max(abs(v - t_ref), 2e-4) for v in ts_vals]
        rk_vals = [_r_mean(tasks_data[f"kottke_{lvl}"]) for lvl in levels]
        rs_vals = [_r_mean(tasks_data[f"staircase_{lvl}"]) for lvl in levels]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.1), dpi=150)
        res = np.asarray(cpws, dtype=float)
        ax0.loglog(
            res,
            err_s,
            "s-",
            color="#d95f02",
            lw=2.0,
            ms=6,
            label="Staircase $|T - T_{\\mathrm{ref}}|$",
        )
        ax0.loglog(
            res,
            err_k,
            "o-",
            color="#1b9e77",
            lw=2.0,
            ms=6,
            label="Kottke $|T - T_{\\mathrm{ref}}|$",
        )
        ax0.loglog(
            res,
            err_s[0] * (res[0] / res) ** 1,
            "--",
            color="#d95f02",
            alpha=0.55,
            lw=1.2,
            label=r"$\mathcal{O}(\Delta x^1)$ guide",
        )
        ax0.loglog(
            res,
            err_k[0] * (res[0] / res) ** 2,
            ":",
            color="#1b9e77",
            alpha=0.65,
            lw=1.4,
            label=r"$\mathcal{O}(\Delta x^2)$ guide",
        )
        ax0.set(
            xlabel="Resolution [cells per wavelength]",
            ylabel=r"Transmission error $|T_{\mathrm{tot}}(1550\;\mathrm{nm}) - T_{\mathrm{ref}}|$",
            title="(a) Total transmission convergence at 1550 nm",
        )
        ax0.grid(True, which="both", ls=":", alpha=0.4)
        ax0.legend(loc="best", fontsize=8.5)

        ax1.loglog(
            res,
            rs_vals,
            "s-",
            color="#d95f02",
            lw=2.0,
            ms=6,
            label="Staircase mean $|S_{11}|^2$",
        )
        ax1.loglog(
            res,
            rk_vals,
            "o-",
            color="#1b9e77",
            lw=2.0,
            ms=6,
            label="Kottke mean $|S_{11}|^2$",
        )
        ax1.set(
            xlabel="Resolution [cells per wavelength]",
            ylabel=r"Band-averaged reflection $\langle |S_{11}|^2 \rangle$",
            title="(b) Back-reflection vs resolution (near the -30 dB floor)",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="best", fontsize=8.5)
        for ax in (ax0, ax1):  # label only the simulated resolutions
            ax.set_xticks(res)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        fig.tight_layout()
        fig.savefig(
            fig_dir / "12_kottke_vs_staircase_convergence.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # 13. Spectra at 3 resolutions on one figure (Kottke vs Staircase)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.1), dpi=150, sharey=False)
        cols = {"coarse": "#7570b3", "medium": "#d95f02", "fine": "#1b9e77"}
        for lvl in levels:
            for avg, ls in (("kottke", "-"), ("staircase", "--")):
                d = tasks_data[f"{avg}_{lvl}"]
                w = d["wavelengths"] * 1e9
                idx_s = np.argsort(w)
                w_s = w[idx_s]
                m = (w_s >= 1505.0) & (w_s <= 1595.0)
                t_db = 10.0 * np.log10(
                    np.maximum(
                        np.abs(d["S21"][idx_s][m]) ** 2
                        + np.abs(d["S31"][idx_s][m]) ** 2,
                        1e-12,
                    )
                )
                r_db = 10.0 * np.log10(
                    np.maximum(np.abs(d["S11"][idx_s][m]) ** 2, 1e-12)
                )
                lbl = f"{avg.capitalize()} {lvl} (cpw={int(d['cpw'])})"
                ax0.plot(w_s[m], t_db, color=cols[lvl], ls=ls, lw=1.8, label=lbl)
                ax1.plot(w_s[m], r_db, color=cols[lvl], ls=ls, lw=1.8, label=lbl)

        ax0.set(
            xlabel="Wavelength λ [nm]",
            ylabel=r"Total guided transmission $10\log_{10}(T_1 + T_2)$ [dB]",
            title="(a) Excess transmission across 3 resolutions",
        )
        ax0.grid(True, ls=":", alpha=0.4)
        ax0.legend(loc="best", fontsize=7.5)

        ax1.set(
            xlabel="Wavelength λ [nm]",
            ylabel=r"Return loss $10\log_{10}|S_{11}|^2$ [dB]",
            title="(b) Return loss across 3 resolutions",
        )
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(loc="best", fontsize=7.5)
        fig.tight_layout()
        fig.savefig(
            fig_dir / "13_resolution_spectra_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 14. Port time trace and energy decay
    if "trace_t" in primary and primary["trace_t"].size > 0:
        t_fs = primary["trace_t"] * 1e15
        ey_tr = primary["trace_Ey"]
        ey_norm = ey_tr / max(float(np.max(np.abs(ey_tr))), 1e-30)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=150)
        ax0.plot(t_fs, ey_norm, color="#2b5c8f", lw=1.4)
        ax0.set(
            xlabel="Time t [fs]",
            ylabel=r"Normalized $E_y(t) / \max|E_y|$",
            title="(a) Output arm port time-domain waveform",
        )
        ax0.grid(True, ls=":", alpha=0.4)

        env = np.maximum(ey_norm**2, 1e-10)
        ax1.semilogy(
            t_fs, env, color="#d95f02", lw=1.4, label=r"$|E_y(t)|^2 / |E_y|_{\max}^2$"
        )
        ax1.axhline(
            1e-4, color="0.4", ls="--", lw=1.2, label="1e-4 energy decay threshold"
        )
        ax1.set(
            xlabel="Time t [fs]",
            ylabel="Normalized instantaneous power (log scale)",
            ylim=(1e-8, 2.0),
            title="(b) Ring-down & termination audit",
        )
        ax1.grid(True, which="both", ls=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(fig_dir / "14_port_time_trace.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Rendered 14 figures in {fig_dir}")


# %% [markdown]
# ## 7. Broadband spectra, symmetry & energy conservation
#
# Fig. 10 shows both arms at $T_1 = T_2 \approx -3.4$ dB across 1505–1595 nm, i.e. a
# total transmission $T_1 + T_2 \approx 0.913$ (about 0.4 dB excess loss) with
# back-reflection near $-30$ dB. By mirror symmetry the two arms must carry equal
# power; the study reproduces this to $|T_1 - T_2| < 10^{-7}$, which tests the port
# normalization and the symmetry of the meshed geometry. The power missing from
# $T_1 + T_2 + R$ is radiated: a Poynting flux box around the device accounts for
# 0.999 of the input power, with about 3.3 % leaving through its lateral sides
# (Fig. 11a). The 2D excess loss (0.34–0.41 dB) is larger than the 3D FDTD value
# (0.13 dB) and the measured 0.28 dB of Zhang et al. (2013): radiation in the 2D
# effective-index model has no substrate or cladding to escape into and is confined
# to the plane, so the 2D loss is not a prediction of the 3D device loss (Section 9).
#
# ## 8. Resolution convergence: Kottke vs staircase
#
# Full study (`--full`, run on Cloud Batch), transmission error at 1550 nm against an
# independent Kottke run at 56 cells per wavelength (counted inside the core, with
# the slab effective index):
#
# | cells per wavelength | Kottke $|T - T_{\mathrm{ref}}|$ | staircase $|T - T_{\mathrm{ref}}|$ | Kottke runtime | staircase runtime |
# |---|---|---|---|---|
# | 14 | $2.0 \times 10^{-3}$ | $1.0 \times 10^{-2}$ | 23 s | 17 s |
# | 20 | $5.2 \times 10^{-4}$ | $1.3 \times 10^{-3}$ | 45 s | 41 s |
# | 28 | $2.7 \times 10^{-4}$ | $1.2 \times 10^{-3}$ | 135 s | 107 s |
#
# Staircasing stalls near $10^{-3}$: refining the grid moves the steps but does not
# remove them. Kottke smoothing keeps converging, so at 20 cells per wavelength it is
# already more accurate than staircasing at 28, at about a third of the runtime.
# Two caveats keep this honest: the reference is itself a Kottke run (a staircase
# reference at 56 cpw would bias the comparison the other way), and subpixel
# smoothing is also what modern FDTD codes use, so this is a staircase-vs-smoothing
# result, not a FIT-vs-FDTD one. Back-reflection (Fig. 12b) sits near the $-30$ dB
# floor at every resolution and shows no clean trend.

# %% [markdown]
# ## 9. Limits & when 2D EIM breaks down
#
# While the 2D effective-index method reproduces the in-plane multi-mode
# interference and 50/50 splitting of the Zhang *et al.* (2013) Y-junction at a
# tiny fraction of 3D computational cost, three physical approximations explain
# the offset between our 2D EIM excess loss ($\approx 0.18\text{--}0.25\;\text{dB}$)
# and the paper's 3D FDTD ($0.13\;\text{dB}$) and experimental
# ($0.28 \pm 0.02\;\text{dB}$) values:
#
# 1. **Out-of-plane radiation at the bifurcation tip**: In 3D SOI, the abrupt
#    termination of the central ridge at the $200\;\text{nm}$ splitter tip
#    couples power into unguided vertical substrate/cladding radiation modes.
#    In 2D EIM (`bc_z = periodic`), vertical radiation is absent and all tip
#    scattering is confined to in-plane cladding waves.
# 2. **Index contrast overestimation & variational EIM**: Standard EIM replaces
#    the etched cladding by the bulk oxide index $n_{\text{SiO}_2} = 1.444$,
#    which slightly overestimates the lateral effective-index contrast
#    $\Delta n = n_{\text{eff,slab}} - n_{\text{SiO}_2} = 1.401$ and shifts the
#    relative beat length $L_\pi = \pi / (\beta_0 - \beta_2)$ between the
#    $\text{TE}_0$ and $\text{TE}_2$ modes inside the $2.0\;\mu\text{m}$ taper
#    (the knots $w_0\dots w_{12}$ were optimized for 3D SOI dispersion). The
#    **variational effective-index method (VEIM)** of **Hammer & Ivanova (2009)**
#    [3] projects the 3D curl-curl operator onto the fixed vertical reference
#    slab profile $\psi(z)$, yielding a modified effective lateral permittivity
#    that improves phase-velocity agreement in high-index-contrast silicon
#    photonics.
# 3. **Material and slab dispersion**: Fixing $n_{\text{eff,slab}}$ to its
#    $1550\;\text{nm}$ value ($2.8449$) neglects both the intrinsic chromatic
#    dispersion of silicon ($\Delta n_{\text{Si}} \approx -0.028$ across
#    $1500\text{--}1600\;\text{nm}$) and the waveguide dispersion of the vertical
#    slab ($\Delta n_{\text{eff,slab}} \approx -0.049$ across the band, shown in
#    `02_eim_vertical_stack.png`).
#
# ## 10. Exercises
#
# 1. **Tip-gap sensitivity**: Modify `Y0_ARM` so that the splitter tip gap
#    varies over $g_{\text{tip}} \in \{100, 150, 200, 300\}\;\text{nm}$ while
#    keeping $w_{12} = 2 y_0 + w_{\text{wg}}$. Plot the excess insertion loss
#    and return loss $|S_{11}|^2$ at $1550\;\text{nm}$ vs $g_{\text{tip}}$.
# 2. **Linear vs PSO spline taper**: Replace `W_KNOTS_UM` by a linear ramp from
#    $0.50\;\mu\text{m}$ to $1.20\;\mu\text{m}$ over the same $2.0\;\mu\text{m}$
#    length. By how many dB does the insertion loss degrade?
# 3. **Reciprocity check**: Run `pic.run_s_parameters(drive=["in", "out1"])` and
#    verify that $|S_{12}(\lambda) - S_{21}(\lambda)| < 10^{-3}$ across the band
#    while driving a single output arm reflects roughly $-6\;\text{dB}$ and
#    radiates $-3\;\text{dB}$ into the antisymmetric $\text{TE}_1$ radiation
#    channel at the input stem.
#
# ## 11. References
#
# 1. Y. Zhang, S. Yang, A. E.-J. Lim, G.-Q. Lo, C. Galland, T. Baehr-Jones, and
#    M. Hochberg, "A compact and low loss Y-junction for submicron silicon
#    waveguide," *Opt. Express* **21**(1), 1310–1316 (2013),
#    DOI: [10.1364/OE.21.001310](https://doi.org/10.1364/OE.21.001310).
# 2. C. Kottke, A. Farjadpour, and S. G. Johnson, "Perturbation theory for
#    anisotropic dielectric interfaces, and application to subpixel smoothing
#    of discretized numerical methods," *Phys. Rev. E* **77**, 036611 (2008),
#    DOI: [10.1103/PhysRevE.77.036611](https://doi.org/10.1103/PhysRevE.77.036611).
# 3. M. Hammer and O. V. Ivanova, "Effective index approximations of photonic
#    crystal slabs: a 2-to-1-D assessment," *Opt. Quantum Electron.* **41**(4),
#    267–283 (2009),
#    DOI: [10.1007/s11082-009-9349-3](https://doi.org/10.1007/s11082-009-9349-3).
# 4. H. H. Li, "Refractive index of silicon and germanium and its wavelength
#    and temperature derivatives," *J. Phys. Chem. Ref. Data* **9**, 561–658
#    (1980), DOI: [10.1063/1.555624](https://doi.org/10.1063/1.555624).
# 5. I. H. Malitson, "Interspecimen comparison of the refractive index of fused
#    silica," *J. Opt. Soc. Am.* **55**, 1205–1209 (1965),
#    DOI: [10.1364/JOSA.55.001205](https://doi.org/10.1364/JOSA.55.001205).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="01. Compact Submicron Silicon Y-Junction (2D EIM)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run coarse 2D study locally (< 60 s) and render all figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run converged 2D study across all tasks and render all figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study point I (0-indexed) at --full resolution",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for --task (default: data/)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Read data/task_*.npz and generate all figures in figures/",
    )
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the study task table and exit",
    )
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS")
        argv = shlex.split(env_args) if env_args else sys.argv[1:]
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.list_tasks:
        tasks = get_task_table(quick=False)
        print(f"# Total tasks: {len(tasks)}")
        for t in tasks:
            print(f"{t['index']}\t{t['name']}\t{t['description']}")
        return

    if args.task is not None:
        tasks = get_task_table(quick=False)
        if not (0 <= args.task < len(tasks)):
            raise SystemExit(f"--task must be in 0..{len(tasks) - 1}")
        run_single_task(tasks[args.task], args.out)
        return

    if args.plot and not (args.quick or args.full):
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
        return

    # Default when invoked with no flags (e.g. notebook execution): --quick
    quick = not args.full
    tasks = get_task_table(quick=quick)
    for t in tasks:
        run_single_task(t, DATA_DIR)
    generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
