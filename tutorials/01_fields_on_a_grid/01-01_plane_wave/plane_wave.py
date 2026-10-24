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
# Copyright (c) CERN, 2026.                   #
# ########################################### #

# %% [markdown]
# # Tutorial 1.1 · Plane wave in a box
#
# **Welcome to the Wakis tutorial book.** This is the first tutorial a new user
# reads: it introduces both the **fundamental numerical physics** of the
# **Finite Integration Technique (FIT)** on a staggered Yee grid and the **core
# Wakis solver workflow** (`GridFIT3D`, `SolverFIT3D`, `GaussianSource`,
# `CurrentSource`, `DFTMonitor`, `TFSFSource`, and `wakis.estimate`), step by
# step, with short runnable code cells and pointers to later chapters where each
# feature is applied to waveguides, antennas, scattering, and accelerators.
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. Formulate **Maxwell's integral equations** on the staggered primary/dual
#    **FIT grid** (Yee 1966 [1]; Weiland 1977 [2]) and verify the topological
#    curl duality $\widetilde{\mathbf{C}} = \mathbf{C}^T$ and discrete
#    leapfrog time stepping in Wakis.
# 2. Estimate peak RAM and wall-clock runtime before launching a simulation
#    using `wakis.estimate.estimate` and `preflight`.
# 3. Excite a broadband plane wave using a soft electric current sheet
#    (`CurrentSource`) driven by a zero-DC derivative `GaussianSource`, sample
#    fields in time and frequency (`DFTMonitor`), and perform **source
#    normalization** (dividing by the analytic source spectrum $S(f)$; Oskooi
#    *et al.* 2010 [4], Oskooi & Johnson 2013 [5]).
# 4. Inject a unidirectional plane wave inside a bounded region with
#    `TFSFSource` and verify machine-precision ($\sim 10^{-16}$) cancellation in
#    the exterior scattered-field region.
# 5. Derive the 3D **Courant–Friedrichs–Lewy (CFL) stability limit**
#    $S = c_0 \Delta t \sqrt{\Delta x^{-2} + \Delta y^{-2} + \Delta z^{-2}} \le 1$,
#    safely demonstrate the exponential Nyquist checkerboard blow-up just above
#    $S = 1$, and match the measured growth rate per step against the exact von
#    Neumann eigenvalue $|\lambda_{\max}(S)| = \left(S + \sqrt{S^2 - 1}\right)^2$.
# 6. Measure **1D and 2D numerical dispersion** $v_p(\text{cpw}, \phi)/c_0$ as a
#    function of **cells per wavelength ($\text{cpw}$)** and **propagation angle
#    ($\phi$)**, confirm $\mathcal{O}(\text{cpw}^{-2})$ second-order convergence
#    against the exact discrete dispersion relation, and justify the **20 cpw
#    rule** and the project's mandatory grid resolution rules.
#
# ## Where each feature is used next in the book
#
# | Solver feature introduced here | Physical role in Tutorial 1.1 | Where it is used later |
# |---|---|---|
# | `GridFIT3D`, `SolverFIT3D`, `estimate`, `preflight` | Staggered primary/dual grid, leapfrog update, and RAM/runtime preflight | Every chapter (Ch. 1–9) |
# | `GaussianSource` & `fourier_transform(f)` | Zero-DC broadband pulse $s(t)$ and analytic spectrum $S(f)$ | Ch. 1.2 (slab spectra), Ch. 2 (dispersive media), Ch. 3–7 |
# | `CurrentSource` (soft source) + `DFTMonitor` | Transparent current sheet & source-normalized transfer function $E(f)/S(f)$ | Ch. 4.1 (dipole Purcell factor near mirrors & cavities), Ch. 7 (SLEDs) |
# | `TFSFSource` | Unidirectional plane-wave box with round-off exterior leakage | Ch. 3.4 & 3.5 (2D cylinder & 3D sphere Lorenz–Mie scattering) |
# | `bc_low`/`bc_high = 'cpml'` & `'bloch'` | Convolutional PML absorption and Floquet–Bloch phase-shifted periodicity | Ch. 1.2 (CPML reflection & oblique incidence), Ch. 6.1 & 6.2 (Bragg gratings & SWS) |
#
# ---
#
# ## 1. Simulation setup schematic & Maxwell's equations on the FIT grid
#
# ### 1.0 Where is the box, how do we launch the plane wave, and what is measured?
#
# Before writing a single update equation, let us inspect the physical and
# numerical layout of our simulation box (`wakis.schematic.plot_schematic`,
# `wakis.schematic.setup_table`, and `wakis.schematic.plot_mesh_and_excitation`):
#
# ![Tutorial 1.1 Figure A — geometry, sources, monitors, and boundaries](figures/00a_setup_geometry.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 35.7 nm × 35.7 nm × 4.71 µm (z ∈ [0 m, 4.71 µm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 1 × 1 × 132 (132 cells) |
# | Discretization | Δz & resolution | Δ_min = 35.7 nm, 28.0 cells/λ (in n_max = 1.00), 28.0 cells across λ_min (1 µm) |
# | Boundary | -x face | Periodic |
# | Boundary | +x face | Periodic |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | CPML (10 cells, 357 nm) |
# | Boundary | +z face | CPML (10 cells, 357 nm) |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Source | 1 · CurrentSource | pos: z = 1.38 µm, pol: J_x (E_x), pulse: f₀ = 299.8 THz, Δf = 104.9 THz |
# | Source | 2 · TFSFSource | pos: z ∈ [1.14 µm, 3.57 µm], pol: E_x, pulse: f₀ = 299.8 THz, Δf = 104.9 THz |
# | Monitor | 3 · DFTMonitor | DFTMonitor (z ∈ [1.98 µm, 3.48 µm]), pos: z ∈ [1.98 µm, 3.48 µm], freqs: 25 freqs in [224.8, 374.7] THz |
# | Monitor | 4 · Probe | Probe z₁ (1.98 µm), pos: z = 1.98 µm, freqs: time-domain |
# | Monitor | 5 · Probe | Probe z₂ (3.48 µm), pos: z = 3.48 µm, freqs: time-domain |
# | Variation | Soft-source run (§1.2) | CurrentSource J_x sheet at z = 1.50 µm; bilateral ±z radiation |
# | Variation | TFSF run (§1.2) | TFSFSource box z ∈ [1.12 µm, 3.38 µm]; unilateral +z injection |
# | Variation | Dispersion sweep (§1.3) | cpw ∈ {10, 20, 40, 80} cells/λ₀; periodic z boundary |
# | Variation | Courant sweep (§1.4) | CFL S ∈ {0.50, 0.99, 1.005, 1.02} |
#
# ![Tutorial 1.1 Figure B — discretization and excitation](figures/00b_setup_mesh_and_excitation.png)
#
# Every component in Figure A, the setup table, and Figure B is extracted
# directly from the live `GridFIT3D`, `SolverFIT3D`, `CurrentSource`,
# `TFSFSource`, and `DFTMonitor` objects:
#
# * **Where is the box?** The computational domain spans $z \in [0, 4.71]\;\mu\text{m}$
#   along the propagation axis ($N_z = 132$ cells of size $\Delta z = 35.7\;\text{nm}$,
#   corresponding to $28\;\text{cells}/\lambda_0$ at $\lambda_0 = 1.00\;\mu\text{m}$).
#   Transversally, the box is a single cell thick ($x, y \in [0, 35.7]\;\text{nm}$,
#   $N_x = N_y = 1$) filled with vacuum ($\varepsilon_r = 1, \mu_r = 1$).
# * **What are the boundary conditions on each face?** The four transverse faces
#   ($\pm x$ and $\pm y$) use exact **Periodic** boundary conditions, enforcing
#   $\partial_x = \partial_y = 0$ so that any $x$-polarized excitation is an
#   exact, diffraction-free 1D TEM plane wave $(\mathbf{E} = E_x\hat{\mathbf{x}},\,
#   \mathbf{H} = H_y\hat{\mathbf{y}},\,\mathbf{k} = +k_0\hat{\mathbf{z}})$. The
#   longitudinal ends ($-z$ and $+z$) are terminated by **10-cell ($357\;\text{nm}$)
#   Convolutional PML (CPML)** absorbing layers (hatched regions) that absorb
#   outgoing waves without reflection.
# * **How are we setting the wave?** A transparent ("soft") electric surface-current
#   sheet `CurrentSource` at $z_0 = 1.38\;\mu\text{m}$ injects an $x$-polarized
#   current density $J_x(t) = J_0\,s(t)\,\delta(z - z_0)$ driven by a zero-DC
#   `GaussianSource` pulse centered at $f_0 = c_0/\lambda_0 = 299.79\;\text{THz}$
#   with bandwidth $\Delta f = 0.35\,f_0 = 104.93\;\text{THz}$. For unidirectional
#   comparison, a Total-Field / Scattered-Field (`TFSFSource`) box spans
#   $z \in [1.14, 3.54]\;\mu\text{m}$.
# * **Where are the extents of the wave?** Because the source is pulsed in time
#   with duration $\tau_p \sim 1/(\pi \Delta f)$, at any instant $t$ the radiated
#   wave is a localized spatial wave packet of length
#   $\Delta z_{\text{pulse}} \approx 2 c_0 / (\pi \Delta f) \approx 1.82\;\mu\text{m}$
#   ($\approx 1.82\,\lambda_0$, blue wave packet drawn inside the box), rather than
#   an infinite sinusoidal train.
# * **Where are the probes and monitors?** Downstream of the source, a line
#   `DFTMonitor` and two time-domain point probes at $z_1 = 1.98\;\mu\text{m}$ and
#   $z_2 = 3.48\;\mu\text{m}$ record $E_x(z, t)$ and accumulate the running Fourier
#   transform $E_x(z, f)$ and $H_y(z, f)$.
#
# ### 1.1 Primary and dual grids, grid voltages, and facet fluxes
#
# Unlike finite-difference methods that approximate point derivatives, Weiland's
# **Finite Integration Technique (FIT)** [2] discretizes **Maxwell's equations in
# integral form** directly on a pair of dual cell complexes: a primary Cartesian
# grid $G$ with nodes $(x_i, y_j, z_k)$ and a dual grid $\widetilde{G}$ whose
# vertices lie at the cell centers of $G$ (matching Yee's staggered lattice [1]):
#
# $$\oint_{\partial A_n} \mathbf{E}\cdot\mathrm{d}\mathbf{s} = -\frac{\mathrm{d}}{\mathrm{d}t}\iint_{A_n} \mathbf{B}\cdot\mathrm{d}\mathbf{A},
# \qquad
# \oint_{\partial \widetilde{A}_m} \mathbf{H}\cdot\mathrm{d}\tilde{\mathbf{s}} = \frac{\mathrm{d}}{\mathrm{d}t}\iint_{\widetilde{A}_m} \mathbf{D}\cdot\mathrm{d}\widetilde{\mathbf{A}} + \iint_{\widetilde{A}_m} \mathbf{J}\cdot\mathrm{d}\widetilde{\mathbf{A}}.$$
#
# Defining electric grid voltages $\widehat{\mathbf{e}}_i = \int_{L_i} \mathbf{E}\cdot\mathrm{d}\mathbf{s} = L_i E_i$
# along primary edges $\mathbf{D}_s = \operatorname{diag}(\mathbf{L})$, magnetic
# voltages $\widehat{\mathbf{h}}_j = \tilde{L}_j H_j$ along dual edges
# $\widetilde{\mathbf{D}}_s = \operatorname{diag}(\tilde{\mathbf{L}})$, and facet
# fluxes $\widehat{\mathbf{b}} = \mathbf{D}_A \mathbf{B}$,
# $\widehat{\mathbf{d}} = \widetilde{\mathbf{D}}_A \mathbf{D}$,
# $\widehat{\mathbf{j}} = \widetilde{\mathbf{D}}_A \mathbf{J}$, Stoke's theorem on
# every face yields the **exact topological Maxwell grid equations**:
#
# $$\mathbf{C}\,\widehat{\mathbf{e}} = -\frac{\mathrm{d}\widehat{\mathbf{b}}}{\mathrm{d}t},
# \qquad
# \widetilde{\mathbf{C}}\,\widehat{\mathbf{h}} = \frac{\mathrm{d}\widehat{\mathbf{d}}}{\mathrm{d}t} + \widehat{\mathbf{j}},
# \qquad
# \mathbf{S}\,\widehat{\mathbf{b}} = \mathbf{0},
# \qquad
# \widetilde{\mathbf{S}}\,\widehat{\mathbf{d}} = \mathbf{q}.$$
#
# The incidence matrices $\mathbf{C}, \widetilde{\mathbf{C}}, \mathbf{S}, \widetilde{\mathbf{S}}$
# contain only $\{-1, 0, +1\}$ and are completely free of metric approximation
# error. Because a primary edge pierces the corresponding dual face with matching
# right-hand orientation, the dual curl is the exact transpose of the primary curl:
#
# $$\widetilde{\mathbf{C}} = \mathbf{C}^T, \qquad \mathbf{S}\,\mathbf{C} = \mathbf{0}, \qquad \widetilde{\mathbf{S}}\,\widetilde{\mathbf{C}} = \mathbf{0}.$$
#
# All metric lengths, areas, and material properties ($\varepsilon, \mu$) enter
# through the diagonal constitutive matrices:
#
# $$\mathbf{A} = \mathbf{D}_\mu^{-1}\mathbf{D}_A^{-1}\mathbf{C}\mathbf{D}_s, \qquad
# \mathbf{B} = \widetilde{\mathbf{D}}_\varepsilon^{-1}\widetilde{\mathbf{D}}_A^{-1}\widetilde{\mathbf{C}}\widetilde{\mathbf{D}}_s.$$
#
# ### 1.2 Second-order leapfrog update and soft current sources
#
# `SolverFIT3D` advances $\mathbf{E}^n$ at integer time steps $t_n = n\Delta t$
# and $\mathbf{H}^{n+1/2}$ at half-integer time steps $t_{n+1/2} = (n + \frac{1}{2})\Delta t$:
#
# $$\mathbf{H}^{n+1/2} = \mathbf{H}^{n-1/2} - \Delta t\,\mathbf{A}\,\mathbf{E}^n,
# \qquad
# \mathbf{E}^{n+1} = \mathbf{E}^n + \Delta t\left(\mathbf{B}\,\mathbf{H}^{n+1/2} - \mathbf{D}_\varepsilon^{-1}\mathbf{J}^{n+1/2}\right).$$
#
# Note that `CurrentSource` evaluates the impressed current $\mathbf{J}^{n+1/2}$ at
# $t_n + \Delta t/2$, centering the forcing term in the electric-field update from
# $t_n$ to $t_{n+1}$ to preserve $\mathcal{O}(\Delta t^2)$ temporal accuracy.

# %%
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time

import numpy as np
from scipy.constants import c as c0

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _helpers import (  # noqa: E402
    ETA0,
    F0,
    LAM0,
    courant_amplification_factor,
    discrete_dispersion_1d,
    discrete_dispersion_2d_angle,
    render_all_figures,
    run_courant_stability_task,
    run_dispersion_task,
    run_soft_source_and_tfsf_task,
    sheet_radiated_field_1d,
)
from wakis import GridFIT3D, SolverFIT3D, TFSFSource  # noqa: E402
from wakis.estimate import estimate, preflight  # noqa: E402
from wakis.field_monitors import DFTMonitor  # noqa: E402
from wakis.sources import CurrentSource, GaussianSource  # noqa: E402

DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

# Run the short interactive step-by-step walkthrough cells when executed in a
# notebook or without CLI batch flags (--list-tasks, --plot, --task).
_RUN_WALKTHROUGH = (__name__ == "__main__") and not any(
    flag in sys.argv[1:] for flag in ("--list-tasks", "--plot", "--task", "-h", "--help")
)

# %% [markdown]
# ---
#
# ## 2. Step-by-step walkthrough of Wakis solver features
#
# ### Step 1 · Building `GridFIT3D` and `SolverFIT3D`, checking $\widetilde{\mathbf{C}} = \mathbf{C}^T$, and preflight estimation
#
# Every Wakis simulation begins by defining a grid (`GridFIT3D`) and a solver
# (`SolverFIT3D`). Before running a large simulation, `wakis.estimate.estimate`
# and `preflight` predict the peak host RAM, GPU memory, and wall-clock run time
# against a target machine profile (`"laptop"` caps RAM at $2\;\text{GB}$).
#
# We also adhere to the three project-wide **grid resolution rules** throughout
# this book:
#
# 1. **Wavelength rule**: $\ge 10$ cells per wavelength ($\text{cpw}$) in the
#    **highest-index material** ($n_{\max}$) in `--quick` mode, and $\ge 20\;\text{cpw}$
#    in production (`--full`) runs: $\Delta x \le \lambda_{\min} / (n_{\max}\,\text{cpw})$.
# 2. **Smallest-feature rule**: $\ge 4$ grid cells across the smallest geometric
#    feature of the structure.
# 3. **Skin-depth rule**: $\ge 4$ grid cells per electromagnetic skin depth
#    $\delta_e$ in lossy or plasmonic conductors.

# %%
if _RUN_WALKTHROUGH:
    dz_demo = LAM0 / 20  # 20 cells per wavelength in vacuum (n_max = 1.0) -> dz = 50 nm
    grid_demo = GridFIT3D(
        x=np.array([0.0, dz_demo]),
        y=np.array([0.0, dz_demo]),
        z=np.arange(81) * dz_demo,
        verbose=0,
    )
    solver_demo = SolverFIT3D(
        grid_demo,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=8,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    # Discrete vector identity S @ C == 0 (div(curl E) == 0) on the topological FIT grid
    from scipy.sparse import hstack as sp_hstack

    sc_demo = sp_hstack([solver_demo.Px, solver_demo.Py, solver_demo.Pz]) @ solver_demo.C
    duality_err = float(np.max(np.abs(sc_demo.data))) if sc_demo.nnz else 0.0
    est_demo = estimate((grid_demo.Nx, grid_demo.Ny, grid_demo.Nz), steps=600, cpml=True)
    preflight(est_demo, "laptop")
    print(
        f"Step 1: Grid shape=({grid_demo.Nx}, {grid_demo.Ny}, {grid_demo.Nz}), "
        f"dz={dz_demo * 1e9:.1f} nm (20 cpw), dt={solver_demo.dt * 1e18:.2f} as, "
        f"||S @ C||={duality_err:.1e}, est RAM={est_demo.host_bytes / 1e6:.0f} MB"
    )

# %% [markdown]
# ### Step 2 · Soft `CurrentSource`, zero-DC `GaussianSource`, `DFTMonitor`, and source normalization
#
# Why do we use a **soft current source** (`CurrentSource`) and a **time-derivative
# Gaussian pulse** (`GaussianSource`)?
#
# * **Soft vs. hard sources**: A "hard" source overwrites $\mathbf{E}$ at the
#   source plane ($E_x(z_0, t) \leftarrow s(t)$), acting like a perfect electric
#   conductor (PEC) mirror to any wave reflected back toward the source. A **soft
#   source** (`CurrentSource`) instead adds an impressed current density $\mathbf{J}$
#   to Ampère's law, so counter-propagating waves pass through the source plane
#   completely unperturbed (Oskooi & Johnson 2013 [5]).
# * **Zero-DC derivative pulse**: A plain Gaussian-modulated cosine has a small
#   nonzero DC Fourier component $S(f=0) \ne 0$. By the continuity equation
#   $\nabla\cdot\mathbf{J} + \partial_t\rho = 0$, a nonzero time integral
#   $\int_{-\infty}^{\infty} J(t)\,\mathrm{d}t \ne 0$ leaves behind a permanent
#   static charge distribution $\rho(\mathbf{r})$ whose electrostatic field never
#   radiates away and contaminates frequency-domain monitors. `GaussianSource`
#   uses the normalized time derivative:
#
#   $$s(t) = \frac{1}{2\pi f_0}\frac{\mathrm{d}}{\mathrm{d}t}\left[e^{-(t-t_0)^2/(2\sigma_t^2)}\sin(2\pi f_0(t-t_0))\right], \qquad \int_{-\infty}^{\infty} s(t)\,\mathrm{d}t = 0,$$
#
#   and provides its exact analytical Fourier transform
#   $S(f) = \int_{-\infty}^{\infty} s(t) e^{-i 2\pi f t}\,\mathrm{d}t$ via
#   `pulse.fourier_transform(f)`.
# * **Source normalization**: Because a linear time-invariant Maxwell system
#   multiplies each spectral component by the pulse spectrum $S(f)$, dividing a
#   `DFTMonitor` field $E_x(z, f)$ by $S(f)$ removes the pulse shape and phase
#   offset $e^{-i 2\pi f t_0}$, yielding the intrinsic **Green's / transfer
#   function** of the structure (Oskooi *et al.* 2010 [4]; Oskooi & Johnson 2013 [5]).
#   For a uniform current sheet $J_0 = 1\;\text{A/m}$ in 1D, the exact discrete
#   transfer function is:
#
#   $$\frac{E_x(z, f)}{S(f)} = -\frac{\eta_0}{2}\,\frac{\cos(\omega\Delta t / 2)}{\cos(k_{\mathrm{num}}\Delta z / 2)}\,J_0\,e^{-i k_{\mathrm{num}}|z - z_0|}.$$

# %%
if _RUN_WALKTHROUGH:
    pulse_demo = GaussianSource(F0, fwidth=0.35 * F0, cutoff=4.5)
    src_demo = CurrentSource(
        pulse_demo,
        component="x",
        center=(0.5 * dz_demo, 0.5 * dz_demo, 20 * dz_demo),
        size=(dz_demo, dz_demo, 0.0),
        amplitude=1.0,
    )
    freqs_demo = np.linspace(0.80 * F0, 1.20 * F0, 9)
    mon_demo = DFTMonitor(
        freqs_demo,
        ["Ex"],
        center=(0.5 * dz_demo, 0.5 * dz_demo, 42 * dz_demo),
        size=(0.0, 0.0, 20 * dz_demo),
    )
    nt_demo = int(np.ceil((pulse_demo.end_time + 2.5 * (80 * dz_demo) / c0) / solver_demo.dt))
    solver_demo.emsolve(nt_demo, source=[src_demo], monitors=[mon_demo])

    ex_dft_demo, coords_demo = mon_demo.get_field("Ex")
    s_f_demo = pulse_demo.fourier_transform(freqs_demo)
    ex_norm_demo = ex_dft_demo[:, 0, 0, :] / s_f_demo[:, None]
    ex_th_demo = sheet_radiated_field_1d(
        freqs_demo, dz_demo, solver_demo.dt, z_dist=coords_demo[2] - 20 * dz_demo
    )
    rel_err_demo = float(np.max(np.abs(np.abs(ex_norm_demo) - np.abs(ex_th_demo)) / np.abs(ex_th_demo)))
    print(
        f"Step 2: Source-normalized |Ex(z, f) / S(f)| matches exact discrete sheet "
        f"impedance to max relative error = {rel_err_demo:.2e}"
    )

# %% [markdown]
# ### Step 3 · Unidirectional plane-wave injection with `TFSFSource`
#
# A single electric current sheet radiates symmetrically into both $+z$ and $-z$.
# In scattering problems (such as Mie scattering by cylinders and spheres in
# **Tutorials 3.4 and 3.5**), we often want a plane wave confined strictly to an
# interior **total-field (TF)** box while the exterior **scattered-field (SF)**
# region contains *only* waves scattered by an object inside the box.
# `TFSFSource` evolves an auxiliary 1D FIT line with the exact longitudinal edge
# lengths and time step of the 3D grid, applying discrete commutator corrections
# on the TF/SF boundary so that the incident wave cancels to machine round-off
# ($\sim 10^{-16}$) outside an empty box.

# %%
if _RUN_WALKTHROUGH:
    grid_tf_demo = GridFIT3D(
        x=np.array([0.0, dz_demo]),
        y=np.array([0.0, dz_demo]),
        z=np.arange(81) * dz_demo,
        verbose=0,
    )
    solver_tf_demo = SolverFIT3D(
        grid_tf_demo,
        bc_low=["periodic", "periodic", "cpml"],
        bc_high=["periodic", "periodic", "cpml"],
        n_pml=8,
        cfln=0.5,
        use_stl=False,
        verbose=0,
    )
    tfsf_demo = TFSFSource(
        pulse_demo,
        center=(0.5 * dz_demo, 0.5 * dz_demo, 40 * dz_demo),
        size=(None, None, 40 * dz_demo),
        direction="+z",
        polarization="x",
        amplitude=1.0,
    )
    nt_tf_peak = int(round((pulse_demo.t0 + 1.0 * LAM0 / c0) / solver_tf_demo.dt))
    solver_tf_demo.emsolve(nt_tf_peak, source=[tfsf_demo])
    ex_tf_profile = np.abs(solver_tf_demo.E[0, 0, :-1, "x"])
    print(
        f"Step 3: TFSFSource interior |Ex|_max={np.max(ex_tf_profile[22:58]):.4f} V/m, "
        f"exterior SF leakage={np.max(ex_tf_profile[10:18]):.2e} V/m"
    )


# %% [markdown]
# ### Step 4 · The Courant–Friedrichs–Lewy (CFL) stability limit
#
# Substituting a discrete plane-wave harmonic ansatz
# $(\mathbf{E}^n, \mathbf{H}^{n+1/2}) \propto \lambda^n e^{i(k_x x + k_y y + k_z z)}$
# into the source-free leapfrog update yields the characteristic polynomial for
# the per-step amplification factor $\lambda = e^{i\omega\Delta t}$:
#
# $$\lambda^2 - 2\left(1 - 2\zeta^2(\mathbf{k})\right)\lambda + 1 = 0,
# \qquad
# \zeta(\mathbf{k}) = c_0\Delta t\sqrt{
#   \frac{\sin^2(k_x\Delta x / 2)}{\Delta x^2}
#   + \frac{\sin^2(k_y\Delta y / 2)}{\Delta y^2}
#   + \frac{\sin^2(k_z\Delta z / 2)}{\Delta z^2}
# }.$$
#
# Over the entire 3D Brillouin zone $k_a \in [-\pi/\Delta a, \pi/\Delta a]$, the
# spatial radical attains its maximum at the **Nyquist corner**
# $\mathbf{k}_{\mathrm{Nyq}} = (\pi/\Delta x, \pi/\Delta y, \pi/\Delta z)$, where
# $\sin^2(k_a\Delta a / 2) = 1$ and $\zeta(\mathbf{k}_{\mathrm{Nyq}}) = S$:
#
# $$S \equiv c_0\Delta t\sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}
# = \frac{\Delta t}{\Delta t_{\max}}.$$
#
# * **Subcritical ($S \le 1$)**: $\zeta(\mathbf{k}) \le 1$ for all $\mathbf{k}$,
#   so $\lambda_\pm = (1 - 2\zeta^2) \pm 2i\zeta\sqrt{1 - \zeta^2}$ lie on the
#   unit circle $|\lambda_\pm| = 1$ (exact discrete energy conservation, no
#   artificial damping).
# * **Supercritical ($S > 1$)**: At the Nyquist corner $\zeta = S > 1$, the
#   discriminant becomes positive and the two roots become real and reciprocal,
#   $\lambda_\pm = -\left(S \pm \sqrt{S^2 - 1}\right)^2$, so the unstable Nyquist
#   checkerboard mode $(-1)^{i+j+k+n}$ blows up exponentially at every step:
#
#   $$|\lambda_{\max}(S)| = \left(S + \sqrt{S^2 - 1}\right)^2 = 2S^2 - 1 + 2S\sqrt{S^2 - 1} > 1.$$
#
# In `SolverFIT3D`, `cfln` sets $S = \text{cfln}$ (default `cfln = 0.5`). Let us
# compare a stable run ($S = 0.99$) against a slightly supercritical run
# ($S = 1.02$), stopping safely before floating-point overflow:

# %%
if _RUN_WALKTHROUGH:
    for s_val in (0.99, 1.02):
        g_cfl = GridFIT3D(
            x=np.arange(7) * dz_demo,
            y=np.arange(7) * dz_demo,
            z=np.arange(25) * dz_demo,
            verbose=0,
        )
        sol_cfl = SolverFIT3D(
            g_cfl,
            bc_low=["periodic", "periodic", "periodic"],
            bc_high=["periodic", "periodic", "periodic"],
            cfln=s_val,
            use_stl=False,
            verbose=0,
        )
        src_cfl = CurrentSource(
            pulse_demo,
            component="x",
            center=(3 * dz_demo, 3 * dz_demo, 12 * dz_demo),
            size=(0.0, 0.0, 0.0),
            amplitude=dz_demo**2,
        )
        peaks = []
        for n in range(120):
            src_cfl.update(sol_cfl, n * sol_cfl.dt)
            sol_cfl.one_step()
            pk = float(np.max(np.abs(sol_cfl.E.to_matrix("x"))))
            peaks.append(pk)
            if pk > 1e14:
                break
        if s_val > 1.0:
            g_meas = float(np.exp(np.polyfit(np.arange(12), np.log(peaks[-12:]), 1)[0]))
            g_exact = float(courant_amplification_factor(s_val)[0])
            print(
                f"Step 4 (S={s_val:.2f} > 1, UNSTABLE): stopped safely at step {len(peaks)}, "
                f"measured |lambda|={g_meas:.4f} vs exact {g_exact:.4f}"
            )
        else:
            print(
                f"Step 4 (S={s_val:.2f} <= 1, STABLE): {len(peaks)} steps completed, "
                f"bounded peak |Ex|={peaks[-1]:.2f} V/m"
            )

# %% [markdown]
# ### Step 5 · Numerical dispersion in 1D & 2D and the 20 cpw rule
#
# For stable Courant numbers ($S \le 1$), setting $\lambda = e^{i\omega\Delta t}$
# in $\sin(\omega\Delta t/2) = \zeta(\mathbf{k})$ gives the **3D discrete
# dispersion relation** on the Yee/FIT grid (Taflove & Hagness 2005 [3]):
#
# $$\left(\frac{2}{c_0\Delta t}\sin\frac{\omega\Delta t}{2}\right)^2
# = \left(\frac{2}{\Delta x}\sin\frac{k_x\Delta x}{2}\right)^2
# + \left(\frac{2}{\Delta y}\sin\frac{k_y\Delta y}{2}\right)^2
# + \left(\frac{2}{\Delta z}\sin\frac{k_z\Delta z}{2}\right)^2.$$
#
# For a 2D plane wave propagating at angle $\phi$ on a square grid
# ($\Delta x = \Delta y = \Delta = \lambda_0 / \text{cpw}$) with
# $(k_x, k_y) = k_{\mathrm{num}}(\phi)(\cos\phi, \sin\phi)$, a Taylor expansion
# for $\text{cpw} \gg 1$ yields the **numerical phase velocity**:
#
# $$\frac{v_p(\text{cpw}, \phi)}{c_0} = \frac{\omega}{c_0\,k_{\mathrm{num}}(\phi)}
# = 1 - \frac{\pi^2}{6\,\text{cpw}^2}\left(\cos^4\phi + \sin^4\phi - S_{1\mathrm{D}}^2\right) + \mathcal{O}(\text{cpw}^{-4}),$$
#
# where $S_{1\mathrm{D}} = c_0\Delta t / \Delta$. Two key physical consequences
# follow immediately:
#
# 1. **Second-order accuracy ($\mathcal{O}(\text{cpw}^{-2})$)** and **grid
#    anisotropy**: Because $\cos^4\phi + \sin^4\phi = 1 - \frac{1}{2}\sin^2(2\phi)$
#    equals $1$ along the grid axes ($\phi = 0^\circ, 90^\circ$) and $\frac{1}{2}$
#    along the grid diagonals ($\phi = 45^\circ$), numerical waves travel slightly
#    slower than $c_0$ and lag most along the coordinate axes.
# 2. **The 20 cells-per-wavelength (cpw) rule**: At $\text{cpw} = 10$, the axial
#    phase-velocity error is $|1 - v_p/c_0| \approx 1.28\%$ ($\approx 4.6^\circ$
#    phase lag per wavelength of propagation). Doubling the resolution to
#    $\text{cpw} = 20$ reduces the error by $4\times$ to $\approx 0.32\%$ (at
#    $S_{1\mathrm{D}} \approx 0.29$) or $\approx 0.13\%$ (at $S_{1\mathrm{D}} = 0.5$),
#    keeping cumulative phase errors small across multi-wavelength photonic and RF
#    domains. In media with refractive index $n > 1$, the wavelength shrinks to
#    $\lambda = \lambda_0 / n$, so $\text{cpw}$ **must always be counted in the
#    highest-index material** ($n_{\max}$), as we will see in **Tutorial 1.2**.

# %% [markdown]
# ---
#
# ## 3. Study tasks & execution modes
#
# `plane_wave.py` supports `--quick`, `--full`, `--task I`, `--plot`, and
# `--list-tasks`:
#
# * **Task 0 (`soft_source_and_tfsf`)**: Soft `CurrentSource` sheet, `GaussianSource`
#   zero-DC pulse, `DFTMonitor` source normalization $E_x(z, f)/S(f)$, `TFSFSource`
#   machine-precision cancellation, and `wakis.estimate` preflight check.
# * **Task 1 (`courant_stability`)**: 3D Courant stability sweep across
#   $S \in [0.50, 1.05]$ verifying bounded energy for $S \le 1$ and exact von
#   Neumann exponential growth $|\lambda_{\max}(S)|$ for $S > 1$.
# * **Tasks 2..N (`dispersion_cpw*`)**: 1D broadband and 2D angular numerical
#   dispersion study across $\text{cpw} \in \{10, 14, 20, 28\}$ in `--quick` mode
#   (6 tasks total) and $\text{cpw} \in \{10, 14, 20, 28, 40, 60\}$ in `--full`
#   mode (8 tasks total), plus a clearly labelled under-resolved $\text{cpw} = 6$
#   warning demonstration stored on the reference task.


# %%
def get_task_table(quick: bool = False) -> list[dict]:
    """Return the study task configurations for ``--quick`` (6 tasks) or ``--full`` (8 tasks)."""
    cpw_src = 20 if quick else 28
    cpw_levels = [
        ("coarse", 10, False),
        ("medium", 14, True),
        ("standard", 20, True),
        ("reference" if quick else "fine", 28, False),
    ]
    if not quick:
        cpw_levels.extend(
            [
                ("very_fine", 40, False),
                ("reference", 60, False),
            ]
        )

    tasks = [
        {
            "index": 0,
            "name": "soft_source_and_tfsf",
            "kind": "source_tfsf",
            "cpw": cpw_src,
            "description": (
                f"1D soft CurrentSource sheet, DFT source normalization & TFSFSource ({cpw_src} cpw)"
            ),
        },
        {
            "index": 1,
            "name": "courant_stability",
            "kind": "courant",
            "cpw": 20,
            "description": "3D Courant stability sweep S in [0.50, 1.05] vs von Neumann growth rate (20 cpw)",
        },
    ]
    for idx_offset, (lvl, cpw_val, save_2d) in enumerate(cpw_levels, start=2):
        tasks.append(
            {
                "index": idx_offset,
                "name": f"dispersion_cpw{cpw_val}",
                "kind": "dispersion",
                "level": lvl,
                "cpw": cpw_val,
                "save_field_2d": save_2d,
                "description": (
                    f"1D broadband & 2D angular numerical dispersion at {cpw_val} cpw ({lvl})"
                ),
            }
        )
    return tasks


def run_single_task(task_cfg: dict, out_dir: Path = DATA_DIR) -> dict:
    """Execute one study task and write ``task_I.npz`` and ``task_I.json`` to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = int(task_cfg["index"])
    kind = str(task_cfg["kind"])
    t0 = time.perf_counter()

    if kind == "source_tfsf":
        payload = run_soft_source_and_tfsf_task(task_cfg)
    elif kind == "courant":
        payload = run_courant_stability_task(task_cfg)
    elif kind == "dispersion":
        payload = run_dispersion_task(task_cfg)
    else:
        raise ValueError(f"Unknown task kind {kind!r}")

    npz_path = out_dir / f"task_{idx}.npz"
    np.savez_compressed(npz_path, **payload)

    elapsed = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    meta = {
        "task_index": idx,
        "name": str(task_cfg["name"]),
        "kind": kind,
        "cpw": int(task_cfg["cpw"]),
        "dx_background_nm": round(float(payload["dx_background_nm"]), 3),
        "dx_min_nm": round(float(payload["dx_min_nm"]), 3),
        "min_feature_nm": round(float(payload["min_feature_nm"]), 3),
        "cells_across_feature": round(float(payload["cells_across_feature"]), 3),
        "skin_depth_nm": None,
        "cells_per_skin_depth": None,
        "n_cells": int(payload["n_cells"]),
        "n_steps": int(payload["n_steps"]),
        "elapsed_s": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
        "npz_bytes": int(npz_path.stat().st_size),
    }
    (out_dir / f"task_{idx}.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[task_{idx} {meta['name']}] cpw={meta['cpw']} "
        f"dx={meta['dx_min_nm']:.2f}nm cells={meta['n_cells']} "
        f"steps={meta['n_steps']} time={elapsed:.2f}s rss={rss_mb:.1f}MB"
    )
    return meta


def generate_all_figures(data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR) -> list[Path]:
    """Read ``data_dir/task_*.npz`` and render all 6 figures into ``fig_dir``."""
    return render_all_figures(data_dir=data_dir, fig_dir=fig_dir)


# %% [markdown]
# ---
#
# ## 4. Quantitative results & resolution convergence
#
# ### Measured `--full` dispersion & stability benchmarks
#
# | Task | Name | cpw | $\Delta x\;[\text{nm}]$ | Axial $v_p(0^\circ)/c_0$ | Axial error $|1 - v_p/c_0|$ | Diagonal $v_p(45^\circ)/c_0$ | Diagonal error $|1 - v_p/c_0|$ | Max $|v_{p,\text{sim}} - v_{p,\text{th}}|/c_0$ (1D / 2D) |
# |---:|---|---:|---:|---:|---:|---:|---:|---:|
# | *(demo)* | `underresolved_cpw6` | 6 | 166.67 | 0.953670 | **4.633%** | 0.981158 | **1.884%** | — |
# | `2` | `dispersion_cpw10` | 10 | 100.00 | 0.984171 | 1.583% | 0.993606 | 0.639% | $1.32\times 10^{-4}$ / $5.65\times 10^{-4}$ |
# | `3` | `dispersion_cpw14` | 14 | 71.43 | 0.992011 | 0.799% | 0.996644 | 0.336% | $2.83\times 10^{-5}$ / $1.79\times 10^{-4}$ |
# | `4` | `dispersion_cpw20` (**20 cpw rule**) | 20 | 50.00 | 0.996118 | **0.388%** | 0.998372 | **0.163%** | $7.67\times 10^{-6}$ / $9.19\times 10^{-5}$ |
# | `5` | `dispersion_cpw28` | 28 | 35.71 | 0.998033 | 0.197% | 0.999173 | 0.083% | $4.37\times 10^{-6}$ / $4.91\times 10^{-5}$ |
# | `6` | `dispersion_cpw40` | 40 | 25.00 | 0.999033 | 0.097% | 0.999594 | 0.041% | $4.37\times 10^{-6}$ / $2.32\times 10^{-5}$ |
# | `7` | `dispersion_cpw60` | 60 | 16.67 | 0.999566 | 0.043% | 0.999819 | 0.018% | $4.31\times 10^{-6}$ / $1.48\times 10^{-5}$ |
#
# * **Empirical convergence order** across $\text{cpw} \in [10, 60]$: slope
#   $= -2.01$ along the grid axes ($\phi = 0^\circ$) and $-1.99$ along the grid
#   diagonals ($\phi = 45^\circ$), confirming second-order $\mathcal{O}(\text{cpw}^{-2})$
#   convergence.
# * **Source normalization & `TFSFSource` (`task_0`, $28\;\text{cpw}$)**:
#   $|E_x(z, f)/S(f)|$ matches the exact discrete sheet impedance to
#   $8.31\times 10^{-4}$ ($0.083\%$), `GaussianSource` relative DC integral is
#   $6.10\times 10^{-6}$, and `TFSFSource` exterior scattered-field leakage is
#   $1.10\times 10^{-15}$.
# * **Courant stability (`task_1`, $20\;\text{cpw}$)**: All subcritical runs
#   ($S \in \{0.50, 0.80, 0.95, 0.99\}$) remain bounded ($|\lambda_{\mathrm{sim}}| \approx 1$),
#   while supercritical runs ($S \in \{1.005, 1.010, 1.020, 1.050\}$) blow up with
#   measured amplification factors $|\lambda_{\mathrm{sim}}| = \{1.22128, 1.32526, 1.48441, 1.86228\}$
#   matching the exact von Neumann roots $|\lambda_{\max}(S)| = \{1.22130, 1.32658, 1.49083, 1.87733\}$
#   to $1.9\times 10^{-5}\text{--}8.0\times 10^{-3}$.
#
# Running `plane_wave.py --full` produces the 6 diagnostic figures in `figures/`:
#
# 1. `01_fit_grid_and_source_pulse.png` — Staggered primary/dual FIT grid and
#    leapfrog time levels ($\widetilde{\mathbf{C}} = \mathbf{C}^T$), zero-DC
#    `GaussianSource` waveform ($\int s(t)\,\mathrm{d}t = 0$), and analytical
#    `fourier_transform(f)` vs. discrete-time DFT.
# 2. `02_soft_source_and_normalization.png` — Space-time propagation $E_x(z, t)$
#    from a soft `CurrentSource` sheet, source-normalized transfer impedance
#    $|E_x(z, f) / S(f)|$ matching the exact discrete sheet impedance to $< 0.2\%$,
#    and unidirectional `TFSFSource` injection with $< 10^{-15}$ exterior leakage.
# 3. `03_courant_stability_limit.png` — Subcritical stability ($S \le 1$) vs.
#    supercritical Nyquist checkerboard blow-up ($S > 1$), matching the exact von
#    Neumann amplification root $|\lambda_{\max}(S)| = (S + \sqrt{S^2 - 1})^2$.
# 4. `04_numerical_dispersion_1d_and_2d.png` — 1D broadband phase velocity
#    $v_p(f)/c_0$ and 2D directional phase velocity $v_p(\text{cpw}, \phi)/c_0$
#    from both 2D cylindrical wave ray-phase unwrapping and 2D Bloch eigenmodes
#    compared against the exact analytical discrete dispersion relation.
# 5. `05_resolution_convergence_20cpw_rule.png` — Log-log convergence of axial
#    ($\phi = 0^\circ$) and diagonal ($\phi = 45^\circ$) phase-velocity error
#    $|1 - v_p/c_0| \propto \text{cpw}^{-2}$ across $\text{cpw} \in [10, 60]$
#    (plus the under-resolved $6\;\text{cpw}$ warning point) and cumulative phase
#    drift vs. propagation distance $L/\lambda_0$.
# 6. `06_plane_wave_propagation.gif` — Animated time-domain propagation of
#    $E_x(z, t)$ and $\eta_0 H_y(z, t)$ absorbed by $\pm z$ CPML boundaries.
#
# ---
#
# ## 5. Limits & failure modes
#
# 1. **Under-resolving the wavelength ($\text{cpw} < 10$) or forgetting $n_{\max}$**:
#    At $\text{cpw} = 6$, the 2D axial phase-velocity error exceeds $3.7\%$
#    ($> 13^\circ$ phase error per wavelength), and above the grid cutoff
#    $\lambda < 2\Delta x$ ($\text{cpw} < 2$), $\sin(\omega\Delta t/2) > S$ makes
#    $k_{\mathrm{num}}$ complex so waves become evanescent on the grid and cannot
#    propagate at all. Always compute $\text{cpw}$ inside the highest-index
#    material ($n_{\max}$), and use $\ge 20\;\text{cpw}$ whenever phase coherence
#    over several wavelengths matters.
# 2. **Exceeding the 3D Courant limit ($S > 1$)**: Even if a wave is uniform
#    transversally ($k_x = k_y = 0$), round-off noise on a 3D grid excites the
#    3D Nyquist corner $(\pi/\Delta x, \pi/\Delta y, \pi/\Delta z)$. Whenever
#    $c_0\Delta t\sqrt{\Delta x^{-2} + \Delta y^{-2} + \Delta z^{-2}} > 1$, the
#    Nyquist mode grows exponentially by $(S + \sqrt{S^2-1})^2$ per step and
#    destroys the simulation within tens of steps. On graded meshes, the smallest
#    cell $\min(\Delta x, \Delta y, \Delta z)$ controls $\Delta t_{\max}$.
# 3. **Truncating `DFTMonitor` before the pulse leaves the domain or omitting
#    source normalization**: Stopping the time-stepping loop while field energy
#    remains in the domain convolves the spectrum with a $\operatorname{sinc}(\omega T)$
#    window, and omitting division by `pulse.fourier_transform(f)` leaves the
#    Gaussian envelope and turn-on phase $e^{-i\omega t_0}$ multiplied into the
#    measured fields.
# 4. **Dividing by $S(f)$ outside the source bandwidth**: Far in the spectral
#    tails ($|f - f_0| \gtrsim 3\,\text{fwidth}$), $|S(f)|$ decays exponentially
#    toward numerical noise, so $E(f)/S(f)$ amplifies round-off and finite-time
#    truncation errors. Always place `DFTMonitor` frequencies within
#    $|f - f_0| \lesssim 1.5\,\text{fwidth}$.
#
# ---
#
# ## 6. Exercises
#
# 1. **Magic time step in 1D ($S_{1\mathrm{D}} = 1$) vs. 3D Courant limit**: On a
#    1D grid where $c_0\Delta t/\Delta z = 1$, verify from
#    `discrete_dispersion_1d` that $\sin(\omega\Delta t/2) = \sin(k_{\mathrm{num}}\Delta z/2)$
#    gives $v_p = c_0$ with zero numerical dispersion! Why can $c_0\Delta t/\Delta x = 1$
#    never be used on a uniform 2D or 3D grid without triggering the Courant
#    instability?
# 2. **Huygens unidirectional source from paired electric and magnetic sheets**:
#    Combine a `CurrentSource` ($J_x = J_0$) and a `MagneticCurrentSource`
#    ($M_y = \eta_0 J_0$) at the same plane $z_0$ and verify that the $-z$ wave is
#    canceled while the $+z$ wave doubles in amplitude. Why does `TFSFSource`
#    achieve much lower backward leakage ($10^{-16}$ vs. $\sim 10^{-3}$) on a
#    staggered grid? (Hint: $E_x$ and $H_y$ are staggered by $\Delta z/2$ and
#    $\Delta t/2$.)
# 3. ** Courant number dependence of 2D dispersion**: Sweep `cfln` from $0.1$ to
#    $0.95$ at fixed $\text{cpw} = 20$ and plot $|1 - v_p(\phi)/c_0|$. Show why
#    operating near the stability limit (`cfln` $\approx 0.85\text{--}0.95$)
#    both speeds up the simulation and reduces numerical dispersion.
#
# ---
#
# ## 7. References
#
# 1. K. S. Yee, "Numerical solution of initial boundary value problems involving
#    Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.*
#    **14**(3), 302–307 (1966),
#    doi:[10.1109/TAP.1966.1138693](https://doi.org/10.1109/TAP.1966.1138693).
# 2. T. Weiland, "A discretization method for the solution of Maxwell's equations
#    for six-component fields," *Electronics and Communications (AEÜ)* **31**(3),
#    116–120 (1977); see also T. Weiland, "Time domain electromagnetic field
#    computation with finite difference methods," *Int. J. Numer. Model.* **9**(4),
#    295–319 (1996),
#    doi:[10.1002/(SICI)1099-1204(199607)9:4<295::AID-JNM240>3.0.CO;2-8](https://doi.org/10.1002/(SICI)1099-1204(199607)9:4<295::AID-JNM240>3.0.CO;2-8).
# 3. A. Taflove and S. C. Hagness, *Computational Electrodynamics: The
#    Finite-Difference Time-Domain Method*, 3rd ed. (Artech House, Norwood, MA,
#    2005), Ch. 4 ("Numerical Dispersion and Stability").
# 4. A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos, and
#    S. G. Johnson, "Meep: A flexible free-software package for electromagnetic
#    simulations by the FDTD method," *Comput. Phys. Commun.* **181**(3), 687–702
#    (2010), doi:[10.1016/j.cpc.2009.11.008](https://doi.org/10.1016/j.cpc.2009.11.008).
# 5. A. Oskooi and S. G. Johnson, "Electromagnetic Wave Source Conditions," Ch. 4
#    in A. Taflove, A. Oskooi, and S. G. Johnson (eds.), *Advances in FDTD
#    Computational Electrodynamics: Photonics and Nanotechnology* (Artech House,
#    2013), [arXiv:1301.5366](https://arxiv.org/abs/1301.5366).
# 6. J.-P. Berenger, "A perfectly matched layer for the absorption of
#    electromagnetic waves," *J. Comput. Phys.* **114**(2), 185–200 (1994),
#    doi:[10.1006/jcph.1994.1159](https://doi.org/10.1006/jcph.1994.1159).
# 7. J. A. Roden and S. D. Gedney, "Convolutional PML (CPML): An efficient FDTD
#    implementation of the CFS-PML for arbitrary media," *Microw. Opt. Technol.
#    Lett.* **27**(5), 334–339 (2000),
#    doi:[10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A](https://doi.org/10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A).
# 8. R. Courant, K. Friedrichs, and H. Lewy, "Über die partiellen
#    Differenzengleichungen der mathematischen Physik," *Math. Ann.* **100**(1),
#    32–74 (1928), doi:[10.1007/BF01448839](https://doi.org/10.1007/BF01448839).
# 9. M. Clemens and T. Weiland, "Discrete electromagnetism with the Finite
#    Integration Technique," *Prog. Electromagn. Res. (PIER)* **32**, 65–87
#    (2001), doi:[10.2528/PIER00080103](https://doi.org/10.2528/PIER00080103).


# %%
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Tutorial 1.1: Plane wave in a box (1D & 2D FIT)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run fast 6-task study and render all 6 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run converged 8-task study (up to 60 cpw) and render all 6 figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study task I (0-indexed)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output directory for task data (default: data/)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Read data/task_*.npz and generate all 6 figures in figures/",
    )
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the study task table and exit",
    )
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS")
        if env_args is not None:
            argv = shlex.split(env_args)
        elif "ipykernel" in sys.modules:
            argv = ["--quick"]
        else:
            argv = sys.argv[1:]
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.list_tasks:
        tasks = get_task_table(quick=bool(args.quick))
        print(f"# Total tasks: {len(tasks)}")
        for t in tasks:
            print(f"{t['index']}\t{t['name']}\t{t['description']}")
        return

    if args.task is not None:
        tasks = get_task_table(quick=bool(args.quick))
        if not (0 <= args.task < len(tasks)):
            raise SystemExit(f"--task must be in 0..{len(tasks) - 1}")
        run_single_task(tasks[args.task], args.out)
        return

    if args.plot:
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
        return

    quick = not args.full
    tasks = get_task_table(quick=quick)
    for t in tasks:
        run_single_task(t, DATA_DIR)
    generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
