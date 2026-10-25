# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial 9.1 · Adjoint Gradients for Time-Domain Maxwell Solvers
#
# **Chapter 9 — Inverse Design**
#
# Gradient-based topology and shape optimization of photonic and RF structures
# requires differentiating a scalar figure of merit $F(\boldsymbol{\rho}) \in \mathbb{R}$
# with respect to $N_p = 10^3\text{--}10^7$ design parameters $\boldsymbol{\rho} \in [0, 1]^{N_p}$.
# Computing $\nabla_{\boldsymbol{\rho}} F$ by central finite differences requires $2 N_p$
# full Maxwell simulations, which is intractable beyond a few dozen parameters.
# The **adjoint method** computes the exact gradient with respect to *all* $N_p$
# parameters in **one forward run plus one backward (transpose) run**, at a cost
# independent of $N_p$.
#
# In electromagnetics and nanophotonics, the adjoint inverse-design formalism builds
# on optimal control and aerodynamic design (Giles & Pierce, 2000) and was established
# for subwavelength photonic structures by Owen D. Miller and collaborators
# ([millergroup.yale.edu](https://millergroup.yale.edu)): Miller's PhD thesis (Miller, 2012)
# derived general shape and topology derivatives of Maxwell's equations by pairing the
# forward field with an adjoint wavefield driven by the figure of merit; Lalau-Keraly,
# Bhargava, Miller & Yablonovitch (2013) applied level-set and boundary-perturbation
# adjoint shape optimization to dielectric waveguide splitters and couplers; Ganapati,
# Miller & Yablonovitch (2014) used electromagnetic adjoint optimization to design
# light-trapping surface textures for subwavelength-thick solar cells; and Miller &
# Yablonovitch (2015) synthesized the general inverse optical design framework, later
# reviewed across nanophotonics by Molesky et al. (2018) and cast in operator
# automatic-differentiation form by Hughes et al. (2019).
#
# ### Learning goals
#
# 1. **Discretize-then-differentiate vs. continuous adjoint**: derive the exact
#    discrete transpose of Wakis's leapfrog Finite Integration Technique (FIT)
#    time-stepper (`AdjointSimulation`, `AdjointStepper`) and verify that it satisfies
#    the operator dot-product identity $\langle \mathbf{U}\mathbf{x}, \mathbf{y}\rangle = \langle \mathbf{x}, \mathbf{U}^H\mathbf{y}\rangle$
#    to floating-point round-off ($\sim 10^{-16}\text{--}10^{-15}$) across PEC, lossy,
#    CPML, dispersive (Drude/Lorentz/Debye), anisotropic tensor, Bloch, and
#    lumped-circuit regimes—and see how mutating a single transposed coefficient
#    breaks the identity by more than 10 orders of magnitude.
# 2. **Adjoint vs. central finite differences**: sweep the finite-difference step
#    size $h \in [10^{-10}, 10^{-1}]$ to trace the classic V-curve separating
#    $O(h^2)$ Taylor truncation error from $O(\varepsilon_{\text{mach}}/h)$ cancellation
#    error, and verify node-by-node gradient parity across random directions.
# 3. **Density parameterization (`MaterialGrid`)**: chain gradients through trilinear
#    edge interpolation $\mathbf{P}$, spatial conic filtering (`conic_filter`),
#    smoothed Heaviside projection (`tanh_projection`, $\beta$-continuation), and
#    harmonic/arithmetic subpixel tensor smoothing (`MaterialGrid(..., smoothing="tensor")`).
# 4. **The four-rung adjoint memory ladder (`wakis.inverse`)**: benchmark memory
#    bytes and wall-clock time across `store` (`float64`/`float32`), binomial
#    `checkpoint` (Griewank & Walther's `revolve` dynamic program), lossless
#    time-reversal with recorded 1-cell CPML interface strips (`reverse`), and
#    band-limited `nyquist` decimation following Park, Miller & Chung (2026).
# 5. **Live 2D topology optimization**: optimize a $90^\circ$ dielectric waveguide
#    bend in under a minute using $\beta$-continuation ($\beta = 2 \to 4 \to 8$),
#    tracking the objective trajectory, binarization, broadband transmission
#    spectrum, steady-state wavefields, and design-evolution animation.
#
# ### Simulation setup
#
# ![Setup geometry](figures/00a_setup_geometry.png)
# ![Setup mesh and excitation](figures/00b_setup_mesh_and_excitation.png)
#
# **Initial ($\bar{\rho} = 0.50$) and final optimized ($\beta = 8$) design-region setup variants:**
#
# ![Initial setup geometry](figures/00a_setup_geometry_initial.png)
# ![Initial setup mesh and excitation](figures/00b_setup_mesh_and_excitation_initial.png)
# ![Final setup geometry](figures/00a_setup_geometry_final.png)
# ![Final setup mesh and excitation](figures/00b_setup_mesh_and_excitation_final.png)
#
# A $15.7\text{ cm} \times 15.7\text{ cm}$ 2D $\text{TM}_z$ ($E_z, H_x, H_y$) domain with 8-cell CPML boundaries launches a broadband Gaussian $J_z$ line source ($f_0 = 3.00\text{ GHz}, \lambda_0 = 10.0\text{ cm}$) in the horizontal dielectric input waveguide ($\varepsilon_r = 6.00, w = 2.14\text{ cm}$) into a central $16\times 16$-cell hatched design region $\Omega_d$ ($5.71\text{ cm} \times 5.71\text{ cm}$, $\varepsilon_r(\bar{\rho}) \in [1.00, 6.00]$) to maximize guided transmission at the orthogonal output `DFTMonitor`.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 157 mm × 157 mm × 35.7 mm (z ∈ [-17.8 mm, 17.8 mm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 44 × 44 × 1 (1936 cells) |
# | Discretization | Δx & resolution | Δ_min = 3.57 mm, 11.4 cells/λ (in n_max = 2.45), 6.0 cells across feature (21.4 mm) |
# | Boundary | -x face | CPML (8 cells, 28.6 mm) |
# | Boundary | +x face | CPML (8 cells, 28.6 mm) |
# | Boundary | -y face | CPML (8 cells, 28.6 mm) |
# | Boundary | +y face | CPML (8 cells, 28.6 mm) |
# | Boundary | -z face | Periodic |
# | Boundary | +z face | Periodic |
# | Material | Background | Vacuum cladding (ε = 1.00, n = 1.00) (isotropic) |
# | Material | Region 1: Input waveguide (ε = 6.00, n = 2.45) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-78.5 mm, -28.6 mm] (d = 50 mm) |
# | Material | Region 2: Output waveguide (ε = 6.00, n = 2.45) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-10.7 mm, 10.7 mm] (d = 21.4 mm) |
# | Material | Region 3: Design region Ω_d (16×16 cells, ε ∈ [1.00, 6.00]) | ε_r = 6.00, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-28.6 mm, 28.6 mm] (d = 57.1 mm) |
# | Source | 1 · CurrentSource | pos: x = -3.93 cm, y ∈ [-1.07, 1.07] cm, pol: E_z (TM_z, J_z), pulse: f₀ = 3 GHz, Δf = 2 GHz |
# | Monitor | 2 · DFTMonitor | Objective DFTMonitor (guided output E_z), pos: x ∈ [-1.07, 1.07] cm, y = 3.93 cm, freqs: 2.55–3.45 GHz (11 freqs, f₀ = 3.00 GHz) |
# | Variation | Initial design (_initial) | Uniform greyscale ρ = 0.50 (ε_r = 3.50) across 16 × 16 design box Ω_d (5.71 cm × 5.71 cm) |
# | Variation | Final design (_final) | β-continuation optimized 90° bend (β = 2 → 4 → 8, ε_r ∈ [1.00, 6.00]) stored in data/task_4_optimization.npz |
# | Variation | Memory ladder (task_3) | store (float64/float32), checkpoint (revolve DP), reverse (CPML strip recording), nyquist (decimation d ∈ {4, 8, 16}) |
#
# ---
#
# ## Physics & Numerical Method
#
# ### 1. Exact discrete adjoint of the leapfrog FIT time-stepper
#
# On a primary/dual Cartesian grid pair $(G, \tilde{G})$, Wakis advances the
# electric field $\mathbf{E}$, magnetic field $\mathbf{H}$, conductive current
# $\mathbf{J}$, and auxiliary boundary/dispersive/lumped states (`cpml`,
# `dispersive`, `lumped`) through $N_t$ leapfrog steps $n = 0, \dots, N_t - 1$:
#
# $$
# \mathbf{x}^{(n+1)} = \mathbf{U}^{(n)}(\boldsymbol{\rho})\,\mathbf{x}^{(n)} + \mathbf{s}^{(n)}, \qquad n = 0, \dots, N_t - 1, \qquad \mathbf{x}^{(0)} = \mathbf{0},
# $$
#
# where each sub-update inside $\mathbf{U}^{(n)}$ is an elementary in-place rule:
#
# 1. **Half-step $\mathbf{B}$ / $\mathbf{H}$ curl update**:
#    $\mathbf{B}^{(n+1/2)} = \mathbf{B}^{(n-1/2)} - \Delta t\,\mathbf{C}\,\mathbf{E}^{(n)}$,
#    followed by magnetic CPML memory accumulation $\boldsymbol{\psi}_H^{(n+1/2)} = \mathbf{b}_H \odot \boldsymbol{\psi}_H^{(n-1/2)} + \mathbf{c}_H \odot (\mathbf{C}_\perp \mathbf{E}^{(n)})$
#    and $\mathbf{H}^{(n+1/2)} = \mu_0^{-1}\mathbf{M}_\mu^{-1}\mathbf{B}^{(n+1/2)}$.
# 2. **Full-step $\mathbf{D}$ curl update**:
#    $\Delta \mathbf{D}^{(n)} = \Delta t\bigl(\tilde{\mathbf{C}}\,\mathbf{H}^{(n+1/2)} - \mathbf{J}^{(n)} - \mathbf{J}_{\text{src}}^{(n)}\bigr)$
#    plus electric CPML auxiliary injection, and $\mathbf{D}^{(n+1)} = \mathbf{D}^{(n)} + \Delta \mathbf{D}^{(n)}$.
# 3. **Material constitutive step (where $\boldsymbol{\rho}$ enters)**:
#    with diagonal or symmetric $3\times 3$ inverse relative permittivity
#    $\mathbf{W}(\boldsymbol{\rho}) = \boldsymbol{\varepsilon}_r(\boldsymbol{\rho})^{-1}$,
#
# $$
# \mathbf{E}^{(n+1)} = \mathbf{E}^{(n)} + \varepsilon_0^{-1}\mathbf{W}(\boldsymbol{\rho})\,\Delta \mathbf{D}^{(n)}.
# $$
#
# Introducing Lagrange multipliers $\boldsymbol{\lambda}^{(n)}$ for any scalar
# objective $F\bigl(\{\mathbf{E}^{(n)}, \mathbf{H}^{(n)}\}\bigr)$ yields the
# **reverse-time discrete adjoint recursion** (`AdjointStepper.step`):
#
# $$
# \boldsymbol{\lambda}^{(n)} = \bigl(\mathbf{U}^{(n)}(\boldsymbol{\rho})\bigr)^H \boldsymbol{\lambda}^{(n+1)} + \left(\frac{\partial F}{\partial \mathbf{x}^{(n)}}\right)^{\!H}, \qquad n = N_t - 1, \dots, 0.
# $$
#
# - **Why discretize-then-differentiate is exact to round-off**: in the
#   *continuous-adjoint* approach, half-step temporal staggering ($\Delta t / 2$),
#   spatial Yee staggering ($\Delta x / 2$), and stretched-coordinate CPML recursions
#   do not commute with continuous integration by parts, leaving an $O(\Delta t, \Delta x)$
#   gradient floor ($10^{-3}\text{--}10^{-2}$). Transposing the discrete update
#   $\mathbf{U}^{(n)}$ directly matches the exact derivative of the discrete
#   simulation to floating-point round-off ($\sim 10^{-16}\text{--}10^{-15}$).
#
# ---
#
# ### 2. Adjoint monitor taps and sensitivity assembly
#
# A `DFTMonitor` accumulates $\hat{E}_e(f_k) = \sum_{n=0}^{N_t - 1} \Delta t\, e^{-2\pi i f_k (n + 1)\Delta t}\, E_e^{(n+1)}$.
# Given cotangents $G_{E,e}(f_k)$ satisfying $\delta F = \operatorname{Re}\langle \mathbf{G}_E, \delta \hat{\mathbf{E}}\rangle$,
# the backward sweep injects at reverse step $n$:
#
# $$
# \frac{\partial F}{\partial E_e^{(n+1)}} = \Delta t \sum_k \operatorname{Re}\!\left\{ G_{E,e}(f_k)\, e^{2\pi i f_k (n + 1)\Delta t} \right\},
# \qquad
# \frac{\partial F}{\partial W_{e,ab}} = \sum_{n=0}^{N_t - 1} \lambda_{D,e,a,\text{raw}}^{(n+1)}\,\Delta D_{e,b}^{(n)},
# $$
#
# where $\lambda_{D,e,\text{raw}}^{(n+1)} = \varepsilon_0^{-1}\lambda_{E,e}^{(n+1)}$ is read
# right before the transposed material step. For isotropic edges ($W_e = \varepsilon_{r,e}^{-1}$),
# $\partial F / \partial \varepsilon_{r,e} = -\varepsilon_{r,e}^{-2}(\partial F / \partial W_e)$.
#
# ---
#
# ### 3. Density parameterization (`MaterialGrid`, filter, $\tanh$ projection, tensor SSP)
#
# 1. **Conic filter** (`conic_filter`, Bourdin 2001):
#    $\tilde{\rho}_{i,j} = \sum_{p,q} K_{p,q}\,\rho_{\text{raw},i-p,j-q} \big/ \sum_{p,q} K_{p,q}$
#    with $K_{p,q} = \max(0, 1 - \sqrt{p^2+q^2}/R)$.
# 2. **Smoothed Heaviside projection** (`tanh_projection`, Wang, Lazarov & Sigmund 2011):
#
# $$
# \bar{\rho}(\tilde{\rho}; \beta, \eta) = \frac{\tanh(\beta\eta) + \tanh\!\bigl(\beta(\tilde{\rho} - \eta)\bigr)}{\tanh(\beta\eta) + \tanh\!\bigl(\beta(1 - \eta)\bigr)}, \qquad
# \frac{\partial \bar{\rho}}{\partial \tilde{\rho}} = \frac{\beta\,\operatorname{sech}^2\!\bigl(\beta(\tilde{\rho} - \eta)\bigr)}{\tanh(\beta\eta) + \tanh\!\bigl(\beta(1 - \eta)\bigr)}.
# $$
#
# 3. **Subpixel-smoothed tensor parameterization** (`MaterialGrid(..., smoothing="tensor")`,
#    Farjadpour et al. 2006): combines parallel arithmetic averaging $\langle\varepsilon\rangle$
#    and normal harmonic averaging $\langle\varepsilon^{-1}\rangle$ via the density-gradient
#    unit normal $\hat{\mathbf{n}} = \nabla\bar{\rho} / \|\nabla\bar{\rho}\|$,
#    $\mathbf{W}_e(\bar{\boldsymbol{\rho}}) = \langle\varepsilon_e\rangle^{-1}\mathbf{I} + (\langle\varepsilon_e^{-1}\rangle - \langle\varepsilon_e\rangle^{-1})\hat{\mathbf{n}}_e\hat{\mathbf{n}}_e^T$.
#
# ---
#
# ### 4. The four-rung adjoint memory ladder (`wakis.inverse`)
#
# | Rung | `sim.memory` | Peak memory bytes | Extra forward steps | Gradient accuracy vs. `store` |
# | :--- | :--- | :--- | :--- | :--- |
# | **1. Store** | `"store"` (`float64` / `float32`) | $N_t \times N_{\text{design}} \times (8\text{ or }4)\text{ B}$ | $0$ | Reference ($0$ / $\sim 10^{-8}$ for `f32`) |
# | **2. Checkpoint** | `"checkpoint"` ($s$ slots, block $b$) | $s \times B_{\text{state}} + b \times N_{\text{design}} \times 8\text{ B}$ | $C(m, s)\,b$ via Griewank–Walther `revolve` DP | **Bitwise identical** ($0$) |
# | **3. Reverse** | `"reverse"` (lossless interior + CPML strip) | $2 B_{\text{state}} + N_t \times N_{\text{interface}} \times 8\text{ B}$ | $0$ (1 reverse pass) | Round-off ($\sim 10^{-15}$) |
# | **4. Nyquist** | `"nyquist"` (decimation $d$) | $\lceil N_t / d\rceil \times N_{\text{design}} \times 8\text{ B}$ | $0$ | Band-limited ($10^{-4}\text{--}10^{-3}$) |
#
# Rungs 1–3 are exact: `"checkpoint"` follows Griewank & Walther's (2000) optimal
# binomial `revolve` schedule, and `"reverse"` steps the lossless interior backward
# from the final state using recorded 1-cell CPML interface strips (Tang et al., 2023;
# Mahlau et al., 2026). Rung 4 (`"nyquist"`) follows Park, Miller & Chung (2026), who
# showed that for band-limited time-domain objectives, storing forward design-edge
# fields at the Nyquist sampling interval of the excitation/monitor bandwidth and
# reconstructing intermediate steps with windowed sinc interpolation reduces adjoint
# FDTD memory by more than two orders of magnitude with controlled gradient error.

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tracemalloc

import matplotlib
import numpy as np
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS_0
from scipy.signal import convolve2d

import wakis
from wakis.inverse import AdjointSimulation, AdjointStepper, MaterialGrid
from wakis.inverse.memory import revolve_cost
from wakis.lumped import LumpedElement, LumpedPort
from wakis.materials import Debye, DispersiveMaterial, Drude, Lorentz
from wakis.sources import CurrentSource, GaussianSource

TUTORIAL_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"
STUDY_DIR = TUTORIAL_DIR.parents[2] / "studies" / "adjoint-memory"

F0 = 3.0e9
LAM0 = C0 / F0


def get_task_table(quick: bool = False) -> list[dict]:
    """Return the task table for Tutorial 9.1 with verified resolution metrics."""
    cpw_bend = 28.0 / float(np.sqrt(6.0))  # 11.43 >= 10 cells per minimum wavelength
    return [
        {
            "index": 0,
            "name": "dot_product_and_mutations",
            "description": (
                "Operator transpose dot-product identity <U x, y> = <x, U^H y> "
                "across 6 physics regimes (PEC, lossy, CPML, dispersive, tensor, "
                "lumped) plus transpose & reverse-boundary mutation checks."
            ),
            "cpw": 12.0,
            "cells_across_feature": 4.0,
        },
        {
            "index": 1,
            "name": "fd_vcurve_and_random_dirs",
            "description": (
                "Adjoint gradient vs. central finite differences on a 2D design block: "
                "step-size sweep h in [1e-10, 1e-1] (truncation O(h^2) vs. round-off "
                "O(eps/h) V-curve), float32 floor, and node-by-node parity."
            ),
            "cpw": cpw_bend,
            "cells_across_feature": 4.0,
            "n_grid": 28 if quick else 32,
            "n_design": 6 if quick else 8,
            "Nt": 90 if quick else 120,
        },
        {
            "index": 2,
            "name": "sensitivity_and_parametrization",
            "description": (
                "2D 90-degree waveguide bend forward field Ez, adjoint field lambda_Ez, "
                "raw/filtered/projected sensitivity maps (conic_filter + tanh_projection), "
                "and subpixel interface shift sensitivity (isotropic vs. tensor smoothing)."
            ),
            "cpw": cpw_bend,
            "cells_across_feature": 6.0,
            "n_grid": 44 if quick else 52,
            "n_design": 16 if quick else 20,
            "Nt": 360 if quick else 460,
        },
        {
            "index": 3,
            "name": "memory_ladder_benchmark",
            "description": (
                "Live benchmark of store (f64/f32), checkpoint (revolve DP), auto, "
                "reverse (CPML interface recording), and nyquist rungs on the 2D bend, "
                "plus reference scaling curves up to Nt = 64,000 and 3D metalens."
            ),
            "cpw": cpw_bend,
            "cells_across_feature": 6.0,
            "n_grid": 40 if quick else 48,
            "n_design": 16 if quick else 20,
            "Nt": 180 if quick else 300,
        },
        {
            "index": 4,
            "name": "toy_bend_optimization",
            "description": (
                "Multi-stage topology optimization (beta = 2 -> 4 -> 8) of a 2D "
                "90-degree dielectric waveguide bend maximizing guided transmission, "
                "with broadband spectrum and design-evolution GIF."
            ),
            "cpw": cpw_bend,
            "cells_across_feature": 6.0,
            "n_grid": 44 if quick else 50,
            "n_design": 16 if quick else 18,
            "Nt": 420 if quick else 520,
            "n_iters": 24 if quick else 32,
        },
    ]


def conic_kernel_2d(radius_cells: float) -> np.ndarray:
    """Build normalized 2D conic filter kernel matching ``wakis.inverse.jax_tools.conic_filter``."""
    r = int(np.ceil(radius_cells))
    if r <= 0:
        return np.ones((1, 1), dtype=np.float64)
    ax = np.arange(-r, r + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    k = np.maximum(0.0, 1.0 - np.hypot(xx, yy) / float(radius_cells))
    return k / float(k.sum())


def conic_filter_np(
    rho: np.ndarray, radius_cells: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply 2D conic filter with boundary-weight normalization and return ``(rho_tilde, K, den)``."""
    k = conic_kernel_2d(radius_cells)
    if k.size == 1:
        return rho.copy(), k, np.ones(rho.shape[:2], dtype=np.float64)
    r2 = np.asarray(rho[:, :, 0] if rho.ndim == 3 else rho, dtype=np.float64)
    den = convolve2d(np.ones_like(r2), k, mode="same")
    out2 = convolve2d(r2, k, mode="same") / den
    out = out2[:, :, None] if rho.ndim == 3 else out2
    return out, k, den


def conic_filter_vjp_np(g_tilde: np.ndarray, k: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Exact discrete transpose (VJP) of :func:`conic_filter_np`."""
    if k.size == 1:
        return g_tilde.copy()
    g2 = np.asarray(g_tilde[:, :, 0] if g_tilde.ndim == 3 else g_tilde, dtype=np.float64)
    g_raw2 = convolve2d(g2 / den, k, mode="same")
    return g_raw2[:, :, None] if g_tilde.ndim == 3 else g_raw2


def tanh_projection_np(
    rho_tilde: np.ndarray, beta: float = 8.0, eta: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed Heaviside projection (Wang, Lazarov & Sigmund 2011) and its exact derivative."""
    b = float(beta)
    e = float(eta)
    if b <= 0.0:
        return rho_tilde.copy(), np.ones_like(rho_tilde, dtype=np.float64)
    th_be = np.tanh(b * e)
    th_b1 = np.tanh(b * (1.0 - e))
    denom = th_be + th_b1
    u = np.tanh(b * (rho_tilde - e))
    rho_bar = (th_be + u) / denom
    dbar_dtilde = b * (1.0 - u**2) / denom
    return rho_bar, dbar_dtilde


def make_bend_problem(
    n_grid: int = 44,
    n_design: int = 16,
    Nt: int = 420,
    n_pml: int = 8,
    cpl: int = 28,
    eps_wg: float = 6.0,
    smoothing: str | None = None,
    include_plane_monitor: bool = False,
    freqs: np.ndarray | list[float] | None = None,
) -> tuple[AdjointSimulation, dict]:
    """Construct a 2D 90-degree dielectric waveguide bend problem in Wakis."""
    dx = LAM0 / float(cpl)
    dz = 10.0 * dx
    L = n_grid * dx
    w = 6.0 * dx
    D = n_design * dx
    freq_arr = np.atleast_1d(np.asarray([F0] if freqs is None else freqs, dtype=np.float64))

    def make_solver():
        grid = wakis.GridFIT3D(
            -L / 2, L / 2, -L / 2, L / 2, -dz / 2, dz / 2, n_grid, n_grid, 1, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"],
            n_pml=n_pml,
            verbose=0,
        )
        for center, size in (
            ((-L / 4, 0.0, 0.0), (L / 2, w, 0.0)),
            ((0.0, L / 4, 0.0), (w, L / 2, 0.0)),
        ):
            guide = MaterialGrid(center, size, (1, 1, 1), eps_min=eps_wg, eps_max=eps_wg)
            guide.apply(solver, np.ones((1, 1, 1)))
        return solver

    design = MaterialGrid(
        (0.0, 0.0, 0.0),
        (D, D, 0.0),
        (n_design, n_design, 1),
        eps_min=1.0,
        eps_max=eps_wg,
        smoothing=smoothing,
    )
    x_src = -L / 2 + (n_pml + 3) * dx
    y_prb = L / 2 - (n_pml + 3) * dx
    src = CurrentSource(
        GaussianSource(F0, 0.65 * F0),
        component="z",
        center=(x_src, 0.0, 0.0),
        size=(0.0, w, 0.0),
    )
    monitors = {
        "probe": wakis.DFTMonitor(
            freq_arr, components=["Ez"], center=(0.0, y_prb, 0.0), size=(w, 0.0, 0.0)
        )
    }
    if include_plane_monitor:
        monitors["plane"] = wakis.DFTMonitor(
            [F0], components=["Ez"], center=(0.0, 0.0, 0.0), size=(L, L, 0.0)
        )
    sim = AdjointSimulation(make_solver, [src], monitors, design, Nt)
    meta = {
        "n_grid": n_grid,
        "n_design": n_design,
        "Nt": Nt,
        "n_pml": n_pml,
        "cpl": cpl,
        "dx": dx,
        "dz": dz,
        "L": L,
        "w": w,
        "D": D,
        "eps_wg": eps_wg,
        "x_src": x_src,
        "y_prb": y_prb,
    }
    return sim, meta


# %% [markdown]
# ---
#
# ## Section 1 · Operator Transpose Dot-Product Test & Mutation Checks (`task_0`)

# %%
def _build_dot_product_case(case: str) -> tuple[wakis.SolverFIT3D, list]:
    """Create a 3D solver and optional lumped elements for a given physics regime."""
    grid = wakis.GridFIT3D(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 12, 10, 10, verbose=0)
    elements: list = []
    if case == "pec":
        solver = wakis.SolverFIT3D(
            grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg=[1.8, 1.0, 0.0], verbose=0
        )
    elif case == "pec-lossy":
        solver = wakis.SolverFIT3D(
            grid, bc_low=["pec"] * 3, bc_high=["pec"] * 3, bg=[2.0, 1.0, 0.05], verbose=0
        )
    elif case == "cpml":
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["cpml"] * 3,
            bc_high=["cpml"] * 3,
            bg=[2.0, 1.0, 0.04],
            n_pml=3,
            verbose=0,
        )
    elif case == "dispersive-cpml":
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=3, cfln=0.5, verbose=0
        )
        mat = DispersiveMaterial(
            2.0,
            [
                Lorentz(1.5, 2 * np.pi * 3.5e9, 2 * np.pi * 0.4e9),
                Drude(2 * np.pi * 2.0e9, 2 * np.pi * 0.5e9),
                Debye(0.8, 1e-10),
            ],
            sigma=0.02,
        )
        solver.add_dispersive_material(mat, center=(0.0, 0.0, 0.0), size=(0.08,) * 3)
    elif case == "tensor-cpml":
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=3, verbose=0
        )
        mg = MaterialGrid(
            center=(0.0, 0.0, 0.0),
            size=(0.1, 0.1, 0.1),
            shape=(5, 5, 5),
            eps_min=1.0,
            eps_max=4.5,
            smoothing="tensor",
        )
        u = np.linspace(-1, 1, 5)
        X, Y, Z = np.meshgrid(u, u, u, indexing="ij")
        rho = np.clip(0.5 + 0.4 * np.tanh(3.0 * (0.5 - np.sqrt(X**2 + Y**2 + Z**2))), 0.05, 0.95)
        mg.apply(solver, rho)
    elif case == "lumped-cpml":
        solver = wakis.SolverFIT3D(
            grid, bc_low=["cpml"] * 3, bc_high=["cpml"] * 3, n_pml=3, verbose=0
        )
        gap = dict(component="z", size=(0.0, 0.0, 0.05))
        elements = [
            LumpedElement(center=(-0.03, 0.0, 0.0), R=50.0, L=2e-9, C=1e-12, **gap),
            LumpedPort(None, center=(0.03, -0.02, 0.0), impedance=75.0, **gap),
        ]
    else:
        raise ValueError(f"Unknown case {case!r}")
    return solver, elements


def _forward_step(solver: wakis.SolverFIT3D, elements: list, t: float = 0.0) -> None:
    for el in elements:
        el.update(solver, t)
    solver.one_step()
    for el in elements:
        el.post_step(solver, t)


def run_task_0_dot_product(cfg: dict) -> dict:
    """Run Task 0: operator dot-product test across 6 physics regimes + 2 mutation checks."""
    cases = ["pec", "pec-lossy", "cpml", "dispersive-cpml", "tensor-cpml", "lumped-cpml"]
    rng = np.random.default_rng(2026)

    case_results = []
    for case in cases:
        solver, elements = _build_dot_product_case(case)
        _forward_step(solver, elements)  # initialize operators and ghost masks
        stepper = AdjointStepper(solver, elements)
        template = stepper.zero_state()
        keys = list(template)

        x = {k: rng.standard_normal(v.shape) for k, v in template.items()}
        y = {k: rng.standard_normal(v.shape) for k, v in template.items()}
        stepper.set_state(x)
        _forward_step(solver, elements)
        Ux = stepper.get_state()
        Uty = stepper.step(y)

        lhs = float(stepper.inner(Ux, y).real)
        rhs = float(stepper.inner(x, Uty).real)
        scale = float(
            stepper.inner({k: np.abs(v) for k, v in Ux.items()}, {k: np.abs(v) for k, v in y.items()}).real
        )
        rel_err = abs(lhs - rhs) / max(scale, 1e-30)
        case_results.append(
            {
                "case": case,
                "lhs": lhs,
                "rhs": rhs,
                "abs_err": abs(lhs - rhs),
                "rel_err": rel_err,
            }
        )

    # Mutation Check A: Perturb one transposed constitutive coupling entry by delta
    solver_m, elems_m = _build_dot_product_case("tensor-cpml")
    _forward_step(solver_m, elems_m)
    stepper_m = AdjointStepper(solver_m, elems_m)
    tmpl_m = stepper_m.zero_state()
    x_m = {k: rng.standard_normal(v.shape) for k, v in tmpl_m.items()}
    y_m = {k: rng.standard_normal(v.shape) for k, v in tmpl_m.items()}
    stepper_m.set_state(x_m)
    _forward_step(solver_m, elems_m)
    Ux_m = stepper_m.get_state()
    Uty_clean = stepper_m.step(y_m)
    lhs_m = float(stepper_m.inner(Ux_m, y_m).real)
    scale_m = float(
        stepper_m.inner({k: np.abs(v) for k, v in Ux_m.items()}, {k: np.abs(v) for k, v in y_m.items()}).real
    )

    mutation_deltas = [0.0, 1e-6, 1e-4, 1e-2]
    mutation_transpose_errs = []
    for delta in mutation_deltas:
        Uty_mut = {k: v.copy() for k, v in Uty_clean.items()}
        if delta > 0.0:
            # Perturb the transposed E-field accumulator
            Uty_mut["E"] = Uty_mut["E"] + delta * np.roll(y_m["E"], 1)
        rhs_m = float(stepper_m.inner(x_m, Uty_mut).real)
        mutation_transpose_errs.append(abs(lhs_m - rhs_m) / max(scale_m, 1e-30))

    # Mutation Check B: Corrupt one recorded CPML interface layer step in memory="reverse"
    sim_rev, _ = make_bend_problem(n_grid=28, n_design=8, Nt=70, n_pml=6, cpl=28)
    rho_test = rng.uniform(0.25, 0.75, sim_rev.design.shape)
    sim_rev.memory = "store"
    out_rev = sim_rev.forward(rho_test)
    G_rev = {"probe": {"Ez": 2.0e18 * out_rev["probe"]["Ez"]}}
    g_store = sim_rev.vjp(G_rev)

    sim_rev.memory = "reverse"
    sim_rev.forward(rho_test)
    g_rev_clean = sim_rev.vjp(G_rev)
    rev_clean_err = float(np.linalg.norm(g_rev_clean - g_store) / np.linalg.norm(g_store))

    sim_rev.forward(rho_test)
    rec_h = np.asarray(sim_rev._rung.rec["H_layer"])
    nz_steps = np.flatnonzero(np.any(rec_h != 0, axis=1))
    corrupt_step = int(nz_steps[len(nz_steps) // 2])
    sim_rev._rung.corrupt = (corrupt_step, "H_layer")
    g_rev_corrupt = sim_rev.vjp(G_rev)
    rev_corrupt_err = float(np.linalg.norm(g_rev_corrupt - g_store) / np.linalg.norm(g_store))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_data = {
        "cases": case_results,
        "mutation_deltas": mutation_deltas,
        "mutation_transpose_errs": mutation_transpose_errs,
        "reverse_clean_rel_err": rev_clean_err,
        "reverse_corrupt_rel_err": rev_corrupt_err,
    }
    with open(DATA_DIR / "task_0_dot_product.json", "w") as f:
        json.dump(out_data, f, indent=2)
    return out_data


# %% [markdown]
# ---
#
# ## Section 2 · Adjoint vs. Central Finite Differences & the Step-Size V-Curve (`task_1`)

# %%
def run_task_1_fd_vcurve(cfg: dict) -> dict:
    """Run Task 1: central finite-difference step sweep V-curve and node-by-node parity."""
    n_grid = int(cfg["n_grid"])
    n_design = int(cfg["n_design"])
    Nt = int(cfg["Nt"])

    sim, _ = make_bend_problem(
        n_grid=n_grid, n_design=n_design, Nt=Nt, n_pml=6, cpl=28, smoothing=None
    )
    rng = np.random.default_rng(101)
    rho0 = rng.uniform(0.30, 0.70, sim.design.shape)
    scale = 1.0e20

    def eval_obj_and_cotangent(sim_obj: AdjointSimulation, rho_arr: np.ndarray) -> tuple[float, dict]:
        out = sim_obj.forward(rho_arr)
        ez = out["probe"]["Ez"]
        val = float(scale * np.sum(np.abs(ez) ** 2))
        cot = {"probe": {"Ez": 2.0 * scale * ez}}
        return val, cot

    sim.memory = "store"
    sim.store_dtype = np.float64
    _, cot0 = eval_obj_and_cotangent(sim, rho0)
    g_adj_f64 = sim.vjp(cot0)

    sim.store_dtype = np.float32
    _, cot0_f32 = eval_obj_and_cotangent(sim, rho0)
    g_adj_f32 = sim.vjp(cot0_f32)
    sim.store_dtype = np.float64

    direction = rng.standard_normal(rho0.shape)
    direction /= float(np.linalg.norm(direction))
    d_adj_f64 = float(np.sum(g_adj_f64 * direction))
    d_adj_f32 = float(np.sum(g_adj_f32 * direction))

    h_values = np.logspace(-10, -1, 15)
    fd_dir_values = np.zeros_like(h_values)
    rel_err_f64 = np.zeros_like(h_values)
    rel_err_f32 = np.zeros_like(h_values)

    for idx, h in enumerate(h_values):
        fp, _ = eval_obj_and_cotangent(sim, rho0 + h * direction)
        fm, _ = eval_obj_and_cotangent(sim, rho0 - h * direction)
        fd_val = (fp - fm) / (2.0 * h)
        fd_dir_values[idx] = fd_val
        rel_err_f64[idx] = abs(fd_val - d_adj_f64) / max(abs(d_adj_f64), 1e-30)
        rel_err_f32[idx] = abs(fd_val - d_adj_f32) / max(abs(d_adj_f32), 1e-30)

    h_opt = 1.0e-5
    step_i = max(1, n_design // 3)
    sub_indices = [
        (i, j, 0)
        for i in range(1, n_design - 1, step_i)[:3]
        for j in range(1, n_design - 1, step_i)[:3]
    ]
    fd_nodes_iso = np.zeros(len(sub_indices))
    adj_nodes_iso = np.zeros(len(sub_indices))
    for k, idx_tuple in enumerate(sub_indices):
        pert = np.zeros_like(rho0)
        pert[idx_tuple] = h_opt
        fp, _ = eval_obj_and_cotangent(sim, rho0 + pert)
        fm, _ = eval_obj_and_cotangent(sim, rho0 - pert)
        fd_nodes_iso[k] = (fp - fm) / (2.0 * h_opt)
        adj_nodes_iso[k] = g_adj_f64[idx_tuple]

    # Tensor subpixel smoothing in 2D TE polarization (Ex, Ey in the x-y plane)
    dx = LAM0 / 28.0
    L_te = n_grid * dx

    def make_te_solver():
        grid = wakis.GridFIT3D(
            -L_te / 2, L_te / 2, -L_te / 2, L_te / 2, -5 * dx, 5 * dx, n_grid, n_grid, 1, verbose=0
        )
        return wakis.SolverFIT3D(
            grid,
            bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"],
            n_pml=6,
            verbose=0,
        )

    src_te = CurrentSource(
        GaussianSource(F0, 0.65 * F0), component="x", center=(-0.22 * L_te, 0.02 * L_te, 0.0)
    )
    mon_te = wakis.DFTMonitor(
        [F0], components=["Ex", "Ey"], center=(0.22 * L_te, -0.02 * L_te, 0.0)
    )
    des_te = MaterialGrid(
        center=(0.0, 0.0, 0.0),
        size=(n_design * dx, n_design * dx, 0.0),
        shape=(n_design, n_design, 1),
        eps_min=1.0,
        eps_max=4.5,
        smoothing="tensor",
    )
    sim_tensor = AdjointSimulation(make_te_solver, [src_te], {"probe": mon_te}, des_te, Nt=Nt)
    u = np.linspace(-1.0, 1.0, n_design)
    X, Y = np.meshgrid(u, u, indexing="ij")
    rho_te = np.clip(0.5 + 0.4 * np.tanh(3.0 * (0.55 - np.hypot(X - 0.1, Y + 0.08))), 0.05, 0.95)[
        :, :, None
    ]

    def eval_te(rho_arr: np.ndarray) -> tuple[float, dict]:
        out = sim_tensor.forward(rho_arr)
        ex, ey = out["probe"]["Ex"], out["probe"]["Ey"]
        val = float(scale * np.sum(np.abs(ex) ** 2 + 0.5 * np.abs(ey) ** 2))
        return val, {"probe": {"Ex": 2.0 * scale * ex, "Ey": scale * ey}}

    _, cot_t = eval_te(rho_te)
    g_adj_tensor = sim_tensor.vjp(cot_t)
    fd_nodes_tensor = np.zeros(len(sub_indices))
    adj_nodes_tensor = np.zeros(len(sub_indices))
    for k, idx_tuple in enumerate(sub_indices):
        pert = np.zeros_like(rho_te)
        pert[idx_tuple] = h_opt
        fp, _ = eval_te(rho_te + pert)
        fm, _ = eval_te(rho_te - pert)
        fd_nodes_tensor[k] = (fp - fm) / (2.0 * h_opt)
        adj_nodes_tensor[k] = g_adj_tensor[idx_tuple]

    iso_node_rel_err = float(
        np.linalg.norm(fd_nodes_iso - adj_nodes_iso) / np.linalg.norm(fd_nodes_iso)
    )
    tensor_node_rel_err = float(
        np.linalg.norm(fd_nodes_tensor - adj_nodes_tensor) / np.linalg.norm(fd_nodes_tensor)
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_1_fd_vcurve.npz",
        h_values=h_values,
        fd_dir_values=fd_dir_values,
        d_adj_f64=np.array(d_adj_f64),
        d_adj_f32=np.array(d_adj_f32),
        rel_err_f64=rel_err_f64,
        rel_err_f32=rel_err_f32,
        fd_nodes_iso=fd_nodes_iso,
        adj_nodes_iso=adj_nodes_iso,
        fd_nodes_tensor=fd_nodes_tensor,
        adj_nodes_tensor=adj_nodes_tensor,
        iso_node_rel_err=np.array(iso_node_rel_err),
        tensor_node_rel_err=np.array(tensor_node_rel_err),
    )
    return {
        "min_rel_err_f64": float(np.min(rel_err_f64)),
        "f32_floor": float(np.linalg.norm(g_adj_f32 - g_adj_f64) / np.linalg.norm(g_adj_f64)),
        "iso_node_rel_err": iso_node_rel_err,
        "tensor_node_rel_err": tensor_node_rel_err,
    }


# %% [markdown]
# ---
#
# ## Section 3 · 2D Sensitivity Map, Filtering, $\tanh$ Projection & Tensor Smoothing (`task_2`)

# %%
def run_task_2_sensitivity_and_parametrization(cfg: dict) -> dict:
    """Run Task 2: 2D bend forward/adjoint fields, sensitivity maps, filter/projection, and tensor smoothing."""
    n_grid = int(cfg["n_grid"])
    n_design = int(cfg["n_design"])
    Nt = int(cfg["Nt"])

    sim, meta = make_bend_problem(
        n_grid=n_grid,
        n_design=n_design,
        Nt=Nt,
        n_pml=8,
        cpl=28,
        smoothing=None,
        include_plane_monitor=True,
    )

    ax = np.linspace(-1.0, 1.0, n_design)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    rho_raw = np.clip(0.50 + 0.32 * np.exp(-2.5 * (xx + yy) ** 2) - 0.15 * (xx - yy), 0.0, 1.0)[
        :, :, None
    ]

    radius_cells = 2.4
    beta_val = 8.0
    rho_tilde, k_filt, den_filt = conic_filter_np(rho_raw, radius_cells)
    rho_bar, dbar_dtilde = tanh_projection_np(rho_tilde, beta=beta_val, eta=0.5)

    scale = 1.0e14
    out = sim.forward(rho_bar)
    ez_probe = out["probe"]["Ez"]
    ez_plane = out["plane"]["Ez"][0, :, :, 0]
    g_bar = sim.vjp({"probe": {"Ez": 2.0 * scale * ez_probe}})
    g_tilde = g_bar * dbar_dtilde
    g_raw = conic_filter_vjp_np(g_tilde, k_filt, den_filt)

    L = meta["L"]
    w = meta["w"]
    y_prb = meta["y_prb"]
    src_adj = CurrentSource(
        GaussianSource(F0, 0.65 * F0),
        component="z",
        center=(0.0, y_prb, 0.0),
        size=(w, 0.0, 0.0),
    )
    mon_adj_plane = wakis.DFTMonitor(
        [F0], components=["Ez"], center=(0.0, 0.0, 0.0), size=(L, L, 0.0)
    )
    sim_adj_wave = AdjointSimulation(
        sim.make_solver, [src_adj], {"plane": mon_adj_plane}, sim.design, Nt
    )
    out_adj_wave = sim_adj_wave.forward(rho_bar)
    lam_ez_plane = out_adj_wave["plane"]["Ez"][0, :, :, 0]

    betas_sweep = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    x_proj = np.linspace(0.0, 1.0, 201)
    proj_curves = np.zeros((len(betas_sweep), len(x_proj)))
    proj_derivs = np.zeros((len(betas_sweep), len(x_proj)))
    binarization_vs_beta = np.zeros(len(betas_sweep))
    for i, b in enumerate(betas_sweep):
        c_val, d_val = tanh_projection_np(x_proj, beta=float(b), eta=0.5)
        proj_curves[i] = c_val
        proj_derivs[i] = d_val
        rb_b, _ = tanh_projection_np(rho_tilde, beta=float(b), eta=0.5)
        binarization_vs_beta[i] = float(1.0 - np.mean(4.0 * rb_b * (1.0 - rb_b)))

    # Subpixel interface shift study in 2D TE polarization
    shifts = np.linspace(-0.35, 0.35, 9)
    dir_deriv_iso = np.zeros_like(shifts)
    dir_deriv_tensor = np.zeros_like(shifts)
    dx = meta["dx"]
    L_sub = 28 * dx

    def make_sub_solver():
        grid = wakis.GridFIT3D(
            -L_sub / 2, L_sub / 2, -L_sub / 2, L_sub / 2, -5 * dx, 5 * dx, 28, 28, 1, verbose=0
        )
        return wakis.SolverFIT3D(
            grid,
            bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"],
            n_pml=6,
            verbose=0,
        )

    src_sub = CurrentSource(
        GaussianSource(F0, 0.65 * F0), component="x", center=(-0.22 * L_sub, 0.0, 0.0)
    )
    mon_sub = wakis.DFTMonitor([F0], components=["Ex", "Ey"], center=(0.22 * L_sub, 0.0, 0.0))
    des_iso = MaterialGrid(
        (0.0, 0.0, 0.0), (8 * dx, 8 * dx, 0.0), (8, 8, 1), eps_min=1.0, eps_max=5.0, smoothing=None
    )
    des_ten = MaterialGrid(
        (0.0, 0.0, 0.0), (8 * dx, 8 * dx, 0.0), (8, 8, 1), eps_min=1.0, eps_max=5.0, smoothing="tensor"
    )
    sim_sub_iso = AdjointSimulation(make_sub_solver, [src_sub], {"probe": mon_sub}, des_iso, Nt=90)
    sim_sub_ten = AdjointSimulation(make_sub_solver, [src_sub], {"probe": mon_sub}, des_ten, Nt=90)

    ax8 = np.linspace(-1.0, 1.0, 8)
    x8, y8 = np.meshgrid(ax8, ax8, indexing="ij")
    normal_coord = 0.8 * x8 + 0.6 * y8

    for idx, s_shift in enumerate(shifts):
        rho_s, drho_ds = tanh_projection_np(
            0.5 + 0.5 * (normal_coord - s_shift)[:, :, None], beta=5.0, eta=0.5
        )
        drho_ds = -0.5 * drho_ds
        out_i = sim_sub_iso.forward(rho_s)
        g_i = sim_sub_iso.vjp(
            {"probe": {"Ex": 2.0 * scale * out_i["probe"]["Ex"], "Ey": 2.0 * scale * out_i["probe"]["Ey"]}}
        )
        dir_deriv_iso[idx] = float(np.sum(g_i * drho_ds))

        out_t = sim_sub_ten.forward(rho_s)
        g_t = sim_sub_ten.vjp(
            {"probe": {"Ex": 2.0 * scale * out_t["probe"]["Ex"], "Ey": 2.0 * scale * out_t["probe"]["Ey"]}}
        )
        dir_deriv_tensor[idx] = float(np.sum(g_t * drho_ds))

    solver_snap = sim.make_solver()
    sim.design.apply(solver_snap, rho_bar)
    eps_z_plane = 1.0 / (solver_snap.ieps.to_matrix("z")[:, :, 0] * EPS_0)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_2_sensitivity.npz",
        rho_raw=rho_raw[:, :, 0],
        rho_tilde=rho_tilde[:, :, 0],
        rho_bar=rho_bar[:, :, 0],
        g_bar=g_bar[:, :, 0],
        g_tilde=g_tilde[:, :, 0],
        g_raw=g_raw[:, :, 0],
        ez_plane_real=np.real(ez_plane).astype(np.float32),
        ez_plane_abs=np.abs(ez_plane).astype(np.float32),
        lam_ez_plane_real=np.real(lam_ez_plane).astype(np.float32),
        lam_ez_plane_abs=np.abs(lam_ez_plane).astype(np.float32),
        eps_z_plane=eps_z_plane.astype(np.float32),
        betas_sweep=betas_sweep,
        x_proj=x_proj,
        proj_curves=proj_curves,
        proj_derivs=proj_derivs,
        binarization_vs_beta=binarization_vs_beta,
        shifts=shifts,
        dir_deriv_iso=dir_deriv_iso,
        dir_deriv_tensor=dir_deriv_tensor,
        L=np.array(meta["L"]),
        D=np.array(meta["D"]),
        w=np.array(meta["w"]),
    )
    return {
        "grad_bar_norm": float(np.linalg.norm(g_bar)),
        "grad_raw_norm": float(np.linalg.norm(g_raw)),
        "binarization_vs_beta": binarization_vs_beta.tolist(),
    }


# %% [markdown]
# ---
#
# ## Section 4 · The Four-Rung Adjoint Memory Ladder (`task_3`)

# %%
def _naive_halving_cost(m: int, s: int) -> float:
    """Forward block evaluations (recording included) for naive midpoint halving with s slots."""
    if m <= 1:
        return float(m)
    if s <= 1:
        return float(m * (m + 1) // 2)
    k = m // 2
    return float(k) + _naive_halving_cost(m - k, s - 1) + _naive_halving_cost(k, s)


def run_task_3_memory_ladder(cfg: dict) -> dict:
    """Run Task 3: live benchmark of all 4 memory ladder rungs + revolve DP curve + reference scaling data."""
    n_grid = int(cfg["n_grid"])
    n_design = int(cfg["n_design"])
    Nt = int(cfg["Nt"])

    sim, _ = make_bend_problem(n_grid=n_grid, n_design=n_design, Nt=Nt, n_pml=6, cpl=28)
    rng = np.random.default_rng(7)
    rho = rng.uniform(0.3, 0.7, sim.design.shape)

    configs = [
        ("store_f64", "store", {"store_dtype": np.float64}),
        ("store_f32", "store", {"store_dtype": np.float32}),
        ("ckpt_s2", "checkpoint", {"checkpoints": 2}),
        ("ckpt_s4", "checkpoint", {"checkpoints": 4}),
        ("ckpt_s8", "checkpoint", {"checkpoints": 8}),
        ("ckpt_s6_b1", "checkpoint", {"checkpoints": 6, "block": 1}),
        ("auto_1.5MB", "auto", {"budget_gb": 0.0015}),
        ("reverse", "reverse", {}),
        ("nyquist_d4", "nyquist", {"decimation": 4}),
        ("nyquist_d8", "nyquist", {"decimation": 8}),
        ("nyquist_d16", "nyquist", {"decimation": 16}),
    ]

    live_results = []
    g_ref = None
    for label, mode, kw in configs:
        sim.memory = mode
        sim.checkpoints = kw.get("checkpoints")
        sim.block = kw.get("block")
        sim.decimation = kw.get("decimation")
        sim.memory_budget_gb = kw.get("budget_gb")
        sim.store_dtype = kw.get("store_dtype", np.float64)

        tracemalloc.start()
        t0 = time.perf_counter()
        out = sim.forward(rho)
        t1 = time.perf_counter()
        G = {"probe": {"Ez": 2.0e18 * out["probe"]["Ez"]}}
        grad = sim.vjp(G)
        t2 = time.perf_counter()
        host_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        if g_ref is None:
            g_ref = grad.copy()
        rel_err = float(np.linalg.norm(grad - g_ref) / np.linalg.norm(g_ref))
        max_rel_err = float(np.max(np.abs(grad - g_ref)) / np.max(np.abs(g_ref)))
        bitwise = bool(np.array_equal(grad, g_ref))
        stats = dict(sim.memory_stats)
        live_results.append(
            {
                "label": label,
                "mode": mode,
                "resolved_mode": str(stats.get("mode", mode)),
                "peak_bytes": int(stats["peak_bytes"]),
                "host_peak_bytes": int(host_peak),
                "recompute_steps": int(stats["recompute_steps"]),
                "t_forward_s": float(t1 - t0),
                "t_backward_s": float(t2 - t1),
                "t_total_s": float(t2 - t0),
                "grad_rel_err": rel_err,
                "grad_max_rel_err": max_rel_err,
                "bitwise": bitwise,
            }
        )

    m_vals = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    revolve_dp_table = {}
    for s_val in (2, 3, 4, 6):
        revolve_dp_table[str(s_val)] = {
            "m": m_vals,
            "revolve_extra": [int(revolve_cost(m, s_val)) for m in m_vals],
            "halving_extra": [int(_naive_halving_cost(m, s_val)) for m in m_vals],
        }

    ref_study = {}
    for name in ("bend_cpu.json", "bend_gpu.json", "roundoff.json", "projection.json"):
        p = STUDY_DIR / name
        if p.exists():
            with open(p) as f:
                ref_study[name.replace(".json", "")] = json.load(f)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "meta": {
            "n_grid": n_grid,
            "n_design": n_design,
            "Nt": Nt,
            "design_edges": int(sim.design.flat_idx.size),
        },
        "live_results": live_results,
        "revolve_dp_table": revolve_dp_table,
        "reference_study": ref_study,
    }
    with open(DATA_DIR / "task_3_memory_ladder.json", "w") as f:
        json.dump(out_payload, f, indent=2)
    return out_payload


# %% [markdown]
# ---
#
# ## Section 5 · Live 2D Waveguide-Bend Topology Optimization (`task_4`)

# %%
def run_task_4_toy_optimization(cfg: dict) -> dict:
    """Run Task 4: multi-stage topology optimization of a 2D 90-degree waveguide bend."""
    n_grid = int(cfg["n_grid"])
    n_design = int(cfg["n_design"])
    Nt = int(cfg["Nt"])
    n_iters = int(cfg["n_iters"])

    sim, meta = make_bend_problem(n_grid=n_grid, n_design=n_design, Nt=Nt, n_pml=8, cpl=28)
    radius_cells = 2.0
    k_filt = conic_kernel_2d(radius_cells)
    den_filt = convolve2d(np.ones((n_design, n_design)), k_filt, mode="same")

    s1 = n_iters // 3
    s2 = n_iters // 3
    s3 = n_iters - s1 - s2
    beta_schedule = np.array([2.0] * s1 + [4.0] * s2 + [8.0] * s3, dtype=np.float64)

    rho_raw = np.full(sim.design.shape, 0.5, dtype=np.float64)
    objectives = np.zeros(n_iters, dtype=np.float64)
    grad_norms = np.zeros(n_iters, dtype=np.float64)
    binarization = np.zeros(n_iters, dtype=np.float64)
    snapshots = np.zeros((n_iters, n_design, n_design), dtype=np.float16)
    grad_snapshots = np.zeros((n_iters, n_design, n_design), dtype=np.float16)

    for it in range(n_iters):
        beta_val = float(beta_schedule[it])
        rho_tilde, _, _ = conic_filter_np(rho_raw, radius_cells)
        rho_bar, dbar_dtilde = tanh_projection_np(rho_tilde, beta=beta_val, eta=0.5)

        out = sim.forward(rho_bar)
        ez = out["probe"]["Ez"]
        obj_val = float(np.sum(np.abs(ez) ** 2))
        objectives[it] = obj_val
        binarization[it] = float(1.0 - np.mean(4.0 * rho_bar * (1.0 - rho_bar)))
        snapshots[it] = rho_bar[:, :, 0].astype(np.float16)

        g_bar = sim.vjp({"probe": {"Ez": 2.0 * ez}})
        g_tilde = g_bar * dbar_dtilde
        g_raw = conic_filter_vjp_np(g_tilde, k_filt, den_filt)
        grad_norms[it] = float(np.linalg.norm(g_raw))
        g_norm_map = g_raw[:, :, 0] / (float(np.max(np.abs(g_raw))) + 1e-30)
        grad_snapshots[it] = g_norm_map.astype(np.float16)

        step_lr = 0.14 if it < s1 + s2 else 0.10
        rho_raw = np.clip(rho_raw + step_lr * g_norm_map[:, :, None], 0.0, 1.0)

    rho_tilde_fin, _, _ = conic_filter_np(rho_raw, radius_cells)
    rho_bar_fin, _ = tanh_projection_np(rho_tilde_fin, beta=float(beta_schedule[-1]), eta=0.5)
    rho_binary_fin = (rho_tilde_fin >= 0.5).astype(np.float64)

    freqs_spec = np.linspace(0.85 * F0, 1.15 * F0, 11)
    L = meta["L"]
    w = meta["w"]
    dz = meta["dz"]
    n_pml = meta["n_pml"]
    x_src = meta["x_src"]

    def make_straight_solver():
        grid = wakis.GridFIT3D(
            -L / 2, L / 2, -L / 2, L / 2, -dz / 2, dz / 2, n_grid, n_grid, 1, verbose=0
        )
        solver = wakis.SolverFIT3D(
            grid,
            bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"],
            n_pml=n_pml,
            verbose=0,
        )
        guide = MaterialGrid((0.0, 0.0, 0.0), (L, w, 0.0), (1, 1, 1), eps_min=6.0, eps_max=6.0)
        guide.apply(solver, np.ones((1, 1, 1)))
        return solver

    src_spec = CurrentSource(
        GaussianSource(F0, 0.65 * F0),
        component="z",
        center=(x_src, 0.0, 0.0),
        size=(0.0, w, 0.0),
    )
    mon_straight = wakis.DFTMonitor(
        freqs_spec, components=["Ez"], center=(-x_src, 0.0, 0.0), size=(0.0, w, 0.0)
    )
    design_dummy = MaterialGrid((0.0, 0.0, 0.0), (w, w, 0.0), (1, 1, 1), eps_min=6.0, eps_max=6.0)
    sim_straight = AdjointSimulation(
        make_straight_solver, [src_spec], {"probe": mon_straight}, design_dummy, Nt
    )
    out_straight = sim_straight.forward(np.ones((1, 1, 1)))
    spec_straight = np.sum(np.abs(out_straight["probe"]["Ez"]) ** 2, axis=(1, 2, 3))

    sim_eval, _ = make_bend_problem(
        n_grid=n_grid,
        n_design=n_design,
        Nt=Nt,
        n_pml=n_pml,
        cpl=28,
        include_plane_monitor=True,
        freqs=freqs_spec,
    )
    rho_init = np.full(sim.design.shape, 0.5, dtype=np.float64)
    out_init = sim_eval.forward(rho_init)
    spec_init = np.sum(np.abs(out_init["probe"]["Ez"]) ** 2, axis=(1, 2, 3))
    ez_plane_init = out_init["plane"]["Ez"][0, :, :, 0]

    out_opt = sim_eval.forward(rho_bar_fin)
    spec_opt = np.sum(np.abs(out_opt["probe"]["Ez"]) ** 2, axis=(1, 2, 3))
    ez_plane_opt = out_opt["plane"]["Ez"][0, :, :, 0]

    out_bin = sim_eval.forward(rho_binary_fin)
    spec_bin = np.sum(np.abs(out_bin["probe"]["Ez"]) ** 2, axis=(1, 2, 3))

    s_init = sim_eval.make_solver()
    sim_eval.design.apply(s_init, rho_init)
    eps_plane_init = 1.0 / (s_init.ieps.to_matrix("z")[:, :, 0] * EPS_0)

    s_opt = sim_eval.make_solver()
    sim_eval.design.apply(s_opt, rho_bar_fin)
    eps_plane_opt = 1.0 / (s_opt.ieps.to_matrix("z")[:, :, 0] * EPS_0)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "task_4_optimization.npz",
        objectives=objectives,
        grad_norms=grad_norms,
        beta_schedule=beta_schedule,
        binarization=binarization,
        snapshots=snapshots,
        grad_snapshots=grad_snapshots,
        rho_init=rho_init[:, :, 0].astype(np.float32),
        rho_bar_fin=rho_bar_fin[:, :, 0].astype(np.float32),
        rho_binary_fin=rho_binary_fin[:, :, 0].astype(np.float32),
        freqs_spec=freqs_spec,
        spec_straight=spec_straight,
        spec_init=spec_init,
        spec_opt=spec_opt,
        spec_bin=spec_bin,
        ez_plane_init_real=np.real(ez_plane_init).astype(np.float32),
        ez_plane_init_abs=np.abs(ez_plane_init).astype(np.float32),
        ez_plane_opt_real=np.real(ez_plane_opt).astype(np.float32),
        ez_plane_opt_abs=np.abs(ez_plane_opt).astype(np.float32),
        eps_plane_init=eps_plane_init.astype(np.float32),
        eps_plane_opt=eps_plane_opt.astype(np.float32),
        L=np.array(L),
        D=np.array(meta["D"]),
        w=np.array(w),
    )
    return {
        "obj_init": float(objectives[0]),
        "obj_final": float(objectives[-1]),
        "improvement_ratio": float(objectives[-1] / max(objectives[0], 1e-30)),
        "final_binarization": float(binarization[-1]),
    }


# %% [markdown]
# ---
#
# ## Section 6 · Visualization & Figure Generation

# %%
def _plot_00_setup_schematics(fig_dir: Path) -> None:
    """Generate 00a/00b setup figures and 00_setup_table.md (base, _initial, _final) from stored data."""
    from matplotlib.patches import Rectangle
    from wakis import figstyle, schematic
    from wakis.templates.shapes import Box

    d4 = np.load(DATA_DIR / "task_4_optimization.npz")
    rho_init = np.asarray(d4["rho_init"], dtype=np.float64)
    rho_final = np.asarray(d4["rho_bar_fin"], dtype=np.float64)
    freqs_spec = np.asarray(d4["freqs_spec"], dtype=np.float64)
    n_design = int(rho_init.shape[0])
    n_grid = int(d4["eps_plane_init"].shape[0] - 1)

    sim, meta = make_bend_problem(
        n_grid=n_grid,
        n_design=n_design,
        Nt=420,
        n_pml=8,
        cpl=28,
        eps_wg=6.0,
        include_plane_monitor=False,
        freqs=freqs_spec,
    )
    L = float(meta["L"])
    D = float(meta["D"])
    w = float(meta["w"])
    dx = float(meta["dx"])
    dz = float(meta["dz"])
    eps_wg = float(meta["eps_wg"])
    x_src = float(meta["x_src"])
    y_prb = float(meta["y_prb"])

    pulse_obj = GaussianSource(F0, 0.65 * F0)
    sources_s = [
        {
            "kind": "sheet",
            "type_name": "CurrentSource",
            "short_word": "J_z src",
            "label": "Input J_z CurrentSource (TM_z E_z, +x)",
            "desc": f"Input J_z line source at x = {x_src * 1e2:.2f} cm, y ∈ [{-0.5 * w * 1e2:.2f}, {0.5 * w * 1e2:.2f}] cm (+x)",
            "pos_str": f"x = {x_src * 1e2:.2f} cm, y ∈ [{-0.5 * w * 1e2:.2f}, {0.5 * w * 1e2:.2f}] cm",
            "polarization": "E_z (TM_z, J_z)",
            "h_bounds": (x_src, x_src),
            "v_bounds": (-0.5 * w, 0.5 * w),
            "e_vec": (0.0, 0.0, 1.0),
            "k_vec": (1.0, 0.0, 0.0),
            "h_vec": (0.0, -1.0, 0.0),
            "pulse": schematic._extract_pulse_metadata(pulse_obj),
        }
    ]
    monitors_s = [
        {
            "kind": "plane",
            "type_name": "DFTMonitor",
            "short_word": "obj DFT",
            "label": "Objective DFTMonitor (guided output E_z)",
            "desc": f"Objective DFTMonitor (E_z) at y = {y_prb * 1e2:.2f} cm, x ∈ [{-0.5 * w * 1e2:.2f}, {0.5 * w * 1e2:.2f}] cm",
            "pos_str": f"x ∈ [{-0.5 * w * 1e2:.2f}, {0.5 * w * 1e2:.2f}] cm, y = {y_prb * 1e2:.2f} cm",
            "frequencies": freqs_spec,
            "freq_str": f"{freqs_spec[0] * 1e-9:.2f}–{freqs_spec[-1] * 1e-9:.2f} GHz ({len(freqs_spec)} freqs, f₀ = 3.00 GHz)",
            "h_bounds": (-0.5 * w, 0.5 * w),
            "v_bounds": (y_prb, y_prb),
        }
    ]
    variations = [
        {
            "name": "Initial design (_initial)",
            "details": "Uniform greyscale ρ = 0.50 (ε_r = 3.50) across 16 × 16 design box Ω_d (5.71 cm × 5.71 cm)",
        },
        {
            "name": "Final design (_final)",
            "details": "β-continuation optimized 90° bend (β = 2 → 4 → 8, ε_r ∈ [1.00, 6.00]) stored in data/task_4_optimization.npz",
        },
        {
            "name": "Memory ladder (task_3)",
            "details": "store (float64/float32), checkpoint (revolve DP), reverse (CPML strip recording), nyquist (decimation d ∈ {4, 8, 16})",
        },
    ]

    variants = [
        (
            "_initial",
            rho_init,
            3.50,
            "Design region Ω_d (initial ρ = 0.50, ε = 3.50)",
            "Design region Ω_d\n(initial ρ = 0.50)",
            "2D 90° Waveguide Bend Adjoint Setup — Initial Design (ρ = 0.50)",
            "Discretization & Excitation — Initial Design (ρ = 0.50)",
        ),
        (
            "_final",
            rho_final,
            eps_wg,
            "Design region Ω_d (final optimized, ε ∈ [1.00, 6.00])",
            "Design region Ω_d\n(final optimized)",
            "2D 90° Waveguide Bend Adjoint Setup — Final Optimized Design (β = 8)",
            "Discretization & Excitation — Final Optimized Design (β = 8)",
        ),
        (
            "",
            rho_final,
            eps_wg,
            "Design region Ω_d (16×16 cells, ε ∈ [1.00, 6.00])",
            "Design region Ω_d\n(hatched, 5.71 cm × 5.71 cm)",
            "2D 90° Dielectric Waveguide Bend — Adjoint Inverse-Design Setup",
            "Discretization (Uniform 28 cells/λ₀ Mesh & 1× Zoom) and Excitation",
        ),
    ]

    zoom_bounds = (-0.5 * D - 2.0 * dx, -0.5 * D + 4.0 * dx, -0.5 * w - 1.5 * dx, 0.5 * w + 1.5 * dx)

    for suffix, rho_2d, des_eps, des_struct_name, des_box_lbl, title_a, title_b in variants:
        solver = sim.make_solver()
        sim.design.apply(solver, rho_2d[:, :, None])
        structs = [
            Box(
                center=(-0.25 * (L + D), 0.0, 0.0),
                size=(0.5 * (L - D), w, dz),
                material=eps_wg,
            ),
            Box(
                center=(0.0, 0.25 * (L + D), 0.0),
                size=(w, 0.5 * (L - D), dz),
                material=eps_wg,
            ),
            Box(
                center=(0.0, 0.0, 0.0),
                size=(D, D, dz),
                material=des_eps,
            ),
        ]
        mat_names = {
            "background": "Vacuum cladding (ε = 1.00, n = 1.00)",
            "region_0": "Input waveguide (ε = 6.00, n = 2.45)",
            "region_1": "Output waveguide (ε = 6.00, n = 2.45)",
            "region_2": des_struct_name,
        }
        res = schematic.save_setup_figures(
            solver,
            fig_dir,
            prefix="00",
            suffix=suffix,
            plane="xy",
            structures=structs,
            material_names=mat_names,
            sources=sources_s,
            monitors=monitors_s,
            unit="cm",
            wavelength=LAM0,
            min_feature=w,
            title_geometry=title_a,
            title_mesh=title_b,
            variations=variations,
        )

        # Enhance 00a: render design array inside hatched design box and place tags cleanly on their objects
        fig_a, ax_a = schematic.plot_schematic(
            solver,
            plane="xy",
            structures=structs[:2],
            material_names=mat_names,
            sources=sources_s,
            monitors=monitors_s,
            unit="cm",
            wavelength=LAM0,
            title=title_a,
            width_in=7.4,
            height_in=4.8,
        )
        d_cm = D * 1e2
        w_cm = w * 1e2
        eps_des = 1.0 + (eps_wg - 1.0) * rho_2d.T
        ax_a.imshow(
            eps_des,
            origin="lower",
            extent=[-0.5 * d_cm, 0.5 * d_cm, -0.5 * d_cm, 0.5 * d_cm],
            cmap="Blues",
            vmin=1.0,
            vmax=eps_wg,
            alpha=0.82,
            zorder=3.2,
            interpolation="nearest",
        )
        ax_a.add_patch(
            Rectangle(
                (-0.5 * d_cm, -0.5 * d_cm),
                d_cm,
                d_cm,
                facecolor="none",
                edgecolor="#6D28D9",
                hatch="xx",
                lw=1.6,
                ls="--",
                zorder=5.4,
            )
        )
        ax_a.text(
            0.0,
            -0.5 * d_cm - 0.95,
            des_box_lbl,
            ha="center",
            va="top",
            fontsize=8.6,
            fontweight="bold",
            color="#5B21B6",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#F5F3FF", edgecolor="#6D28D9", lw=0.8, alpha=0.94),
            zorder=9,
        )
        # Position numbered tags and bottom boundary tag cleanly without overlaps
        for txt_obj in ax_a.texts:
            t_str = txt_obj.get_text()
            role = getattr(txt_obj, "_schematic_role", "")
            if t_str.startswith("1 ·"):
                txt_obj.set_position((-3.25, 0.5 * w_cm + 0.45))
                txt_obj.set_ha("right")
                txt_obj.set_va("bottom")
            elif t_str.startswith("2 ·"):
                txt_obj.set_position((0.5 * w_cm + 0.35, y_prb * 1e2))
                txt_obj.set_ha("left")
                txt_obj.set_va("center")
            elif role == "boundary_bottom":
                txt_obj.set_position((-3.2, -0.5 * L * 1e2 - 0.35))
                txt_obj.set_ha("center")
                txt_obj.set_va("top")
        ax_a.spines["bottom"].set_position(("outward", 24))

        leg = ax_a.get_legend()
        if leg is not None:
            handles = list(leg.legend_handles)
            labels = [t.get_text() for t in leg.get_texts()]
            handles.insert(
                2,
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="#DBEAFE",
                    edgecolor="#6D28D9",
                    hatch="xx",
                    lw=1.2,
                    ls="--",
                ),
            )
            labels.insert(2, f"{des_struct_name} (d = {d_cm:.2f} cm)")
            ax_a.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.28),
                ncol=2,
                frameon=False,
                fontsize=8.8,
                handlelength=1.8,
                columnspacing=1.2,
            )
        figstyle.save(fig_a, res["geometry_figure"], close=True, auto_legend_below=False)

        # Enhance 00b: overlay design array and hatched design region box on mesh panel & zoom inset
        fig_b, axes_b = schematic.plot_mesh_and_excitation(
            solver,
            plane="xy",
            structures=structs[:2],
            material_names=mat_names,
            sources=sources_s,
            monitors=monitors_s,
            unit="cm",
            zoom_bounds=zoom_bounds,
            wavelength=LAM0,
            min_feature=w,
            title=title_b,
            width_in=7.4,
            height_in=5.6,
        )
        ax_mesh = axes_b[0]
        ax_mesh.imshow(
            eps_des,
            origin="lower",
            extent=[-0.5 * d_cm, 0.5 * d_cm, -0.5 * d_cm, 0.5 * d_cm],
            cmap="Blues",
            vmin=1.0,
            vmax=eps_wg,
            alpha=0.72,
            zorder=2.1,
            interpolation="nearest",
        )
        ax_mesh.add_patch(
            Rectangle(
                (-0.5 * d_cm, -0.5 * d_cm),
                d_cm,
                d_cm,
                facecolor="none",
                edgecolor="#6D28D9",
                hatch="xx",
                lw=1.3,
                ls="--",
                zorder=3.2,
            )
        )
        for child_ax in ax_mesh.child_axes:
            child_ax.set_axes_locator(None)
            child_ax.set_position([0.68, 0.56, 0.24, 0.28])
            child_ax.imshow(
                eps_des,
                origin="lower",
                extent=[-0.5 * d_cm, 0.5 * d_cm, -0.5 * d_cm, 0.5 * d_cm],
                cmap="Blues",
                vmin=1.0,
                vmax=eps_wg,
                alpha=0.72,
                zorder=1.6,
                interpolation="nearest",
            )
            child_ax.add_patch(
                Rectangle(
                    (-0.5 * d_cm, -0.5 * d_cm),
                    d_cm,
                    d_cm,
                    facecolor="none",
                    edgecolor="#6D28D9",
                    hatch="xx",
                    lw=1.1,
                    ls="--",
                    zorder=2.2,
                )
            )
            for ins_txt in child_ax.texts:
                ins_txt.set_fontsize(8.4)
                ins_txt.set_position((0.5, 0.93))
        figstyle.save(fig_b, res["mesh_figure"], close=True, auto_legend_below=False)


def _plot_01_schematic(fig_dir: Path) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(
        6.0,
        4.9,
        "Discrete Adjoint of the Leapfrog FIT Time-Stepper & Density Parameterization Chain",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
    )

    boxes_top = [
        (0.4, 3.3, 2.0, 1.1, "#e8f1fa", "#2b5c8f", r"Latent $\boldsymbol{\rho}_{\mathrm{raw}}$" + "\n(Unconstrained)"),
        (3.0, 3.3, 2.1, 1.1, "#e8f1fa", "#2b5c8f", r"Conic Filter $K_R$" + "\n" + r"$\tilde{\boldsymbol{\rho}} = (K_R * \boldsymbol{\rho}_{\mathrm{raw}})/\mathrm{den}$"),
        (5.7, 3.3, 2.2, 1.1, "#e8f1fa", "#2b5c8f", r"Projection $\beta, \eta$" + "\n" + r"$\bar{\boldsymbol{\rho}} = \mathrm{tanh\_proj}(\tilde{\boldsymbol{\rho}})$"),
        (8.5, 3.3, 3.1, 1.1, "#fff3e0", "#b26a00", r"MaterialGrid $\mathbf{P}$" + "\n" + r"$\mathbf{W}(\bar{\boldsymbol{\rho}}) = \boldsymbol{\varepsilon}_r(\bar{\boldsymbol{\rho}})^{-1}$"),
    ]
    for x0, y0, w, h, fc, ec, txt in boxes_top:
        rect = mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.08", facecolor=fc, edgecolor=ec, lw=1.6
        )
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=9.2)

    for x_start, x_end in ((2.45, 2.95), (5.15, 5.65), (7.95, 8.45)):
        ax.annotate(
            "",
            xy=(x_end, 3.85),
            xytext=(x_start, 3.85),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#2b5c8f"),
        )

    fwd_box = mpatches.FancyBboxPatch(
        (0.4, 0.5),
        5.1,
        2.1,
        boxstyle="round,pad=0.1",
        facecolor="#edf7ed",
        edgecolor="#2e7d32",
        lw=1.6,
    )
    ax.add_patch(fwd_box)
    ax.text(
        2.95,
        2.25,
        r"Forward Leapfrog ($n = 0 \to N_t - 1$)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#1b5e20",
    )
    ax.text(
        2.95,
        1.35,
        r"1. $\mathbf{B}^{(n+1/2)} = \mathbf{B}^{(n-1/2)} - \Delta t\,\mathbf{C}\mathbf{E}^{(n)}$ (+ CPML $\boldsymbol{\psi}_H$)"
        + "\n"
        + r"2. $\Delta \mathbf{D}^{(n)} = \Delta t\,(\tilde{\mathbf{C}}\mathbf{H}^{(n+1/2)} - \mathbf{J}^{(n)} - \mathbf{J}_{\mathrm{src}}^{(n)})$"
        + "\n"
        + r"3. $\mathbf{E}^{(n+1)} = \mathbf{E}^{(n)} + \varepsilon_0^{-1}\mathbf{W}(\bar{\boldsymbol{\rho}})\Delta\mathbf{D}^{(n)}$  $\longrightarrow$ DFT $\hat{\mathbf{E}}(f_k)$",
        ha="center",
        va="center",
        fontsize=8.8,
    )

    bwd_box = mpatches.FancyBboxPatch(
        (6.5, 0.5),
        5.1,
        2.1,
        boxstyle="round,pad=0.1",
        facecolor="#fce8e6",
        edgecolor="#c62828",
        lw=1.6,
    )
    ax.add_patch(bwd_box)
    ax.text(
        9.05,
        2.25,
        r"Backward Transpose ($n = N_t - 1 \to 0$)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#b71c1c",
    )
    ax.text(
        9.05,
        1.35,
        r"1. Inject monitor tap: $\boldsymbol{\lambda}_E += \Delta t\,\mathrm{Re}\{\mathbf{G}_E e^{2\pi i f_k t_{n+1}}\}$"
        + "\n"
        + r"2. Accumulate: $\partial F / \partial W_e += \varepsilon_0^{-1}\lambda_{E,e}^{(n+1)}\Delta D_e^{(n)}$"
        + "\n"
        + r"3. Transpose step: $\boldsymbol{\lambda}^{(n)} = (\mathbf{U}^{(n)})^T \boldsymbol{\lambda}^{(n+1)}$",
        ha="center",
        va="center",
        fontsize=8.8,
    )

    ax.annotate(
        r"$\Delta \mathbf{D}^{(n)}$ (store / checkpoint / reverse)",
        xy=(6.45, 1.55),
        xytext=(5.55, 1.55),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#444444",
    )
    ax.annotate(
        "",
        xy=(6.45, 1.45),
        xytext=(5.55, 1.45),
        arrowprops=dict(arrowstyle="->", lw=1.6, color="#555555"),
    )
    ax.annotate(
        r"VJP pullback $\partial F / \partial \boldsymbol{\rho}_{\mathrm{raw}}$",
        xy=(1.4, 3.25),
        xytext=(9.0, 2.65),
        arrowprops=dict(
            arrowstyle="->", lw=1.5, ls="--", color="#c62828", connectionstyle="arc3,rad=-0.12"
        ),
        fontsize=9.0,
        color="#b71c1c",
        fontweight="bold",
    )

    fig.savefig(fig_dir / "01_adjoint_pipeline_schematic.png", dpi=180)
    plt.close(fig)


def _plot_02_dot_product(fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    with open(DATA_DIR / "task_0_dot_product.json") as f:
        d0 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    ax = axes[0]
    labels = [c["case"] for c in d0["cases"]]
    errs = [max(c["rel_err"], 1e-17) for c in d0["cases"]]
    x = np.arange(len(labels))
    bars = ax.bar(x, errs, color="#1f77b4", edgecolor="#0d3b66", width=0.6)
    ax.axhline(
        np.finfo(np.float64).eps,
        color="#555555",
        ls="--",
        lw=1.2,
        label=r"$\varepsilon_{\mathrm{mach}} \approx 2.2\times 10^{-16}$",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel(r"Relative error $|\langle \mathbf{U}\mathbf{x}, \mathbf{y}\rangle - \langle \mathbf{x}, \mathbf{U}^T\mathbf{y}\rangle| / \mathrm{scale}$")
    ax.set_title("(a) Exact Transpose Dot-Product Test Across Regimes")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.8)
    for b, v in zip(bars, errs):
        ax.text(
            b.get_x() + b.get_width() / 2, v * 1.6, f"{v:.1e}", ha="center", va="bottom", fontsize=7.8
        )

    ax2 = axes[1]
    mut_labels = [
        "Unmutated\ntranspose",
        "Transpose mut.\n" + r"$\delta = 10^{-6}$",
        "Transpose mut.\n" + r"$\delta = 10^{-4}$",
        "Transpose mut.\n" + r"$\delta = 10^{-2}$",
        "Clean\nreverse",
        "Corrupted\nH_layer",
    ]
    mut_vals = [max(v, 1e-17) for v in d0["mutation_transpose_errs"]] + [
        max(d0["reverse_clean_rel_err"], 1e-17),
        max(d0["reverse_corrupt_rel_err"], 1e-17),
    ]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#8c1d40", "#2ca02c", "#d62728"]
    x2 = np.arange(len(mut_labels))
    bars2 = ax2.bar(x2, mut_vals, color=colors, edgecolor="#222222", width=0.6)
    ax2.axhline(1e-12, color="#b71c1c", ls=":", lw=1.3, label="Pass/fail threshold ($10^{-12}$)")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-17, 1.0)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(mut_labels, fontsize=8.5)
    ax2.set_ylabel("Relative error (log scale)")
    ax2.set_title("(b) Mutation Falsifiability: Perturbed Transpose & Corrupted Record")
    ax2.grid(True, which="both", axis="y", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.8)
    for b, v in zip(bars2, mut_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2, v * 1.8, f"{v:.1e}", ha="center", va="bottom", fontsize=7.8
        )

    fig.savefig(fig_dir / "02_dot_product_and_mutations.png", dpi=180)
    plt.close(fig)


def _plot_03_fd_vcurve(fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    d1 = np.load(DATA_DIR / "task_1_fd_vcurve.npz")
    h_vals = d1["h_values"]
    err_f64 = d1["rel_err_f64"]
    err_f32 = d1["rel_err_f32"]
    fd_iso = d1["fd_nodes_iso"]
    adj_iso = d1["adj_nodes_iso"]
    fd_ten = d1["fd_nodes_tensor"]
    adj_ten = d1["adj_nodes_tensor"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.loglog(
        h_vals, err_f64, "o-", color="#1f77b4", lw=1.8, label="Adjoint (float64 increments) vs. central FD"
    )
    ax.loglog(
        h_vals, err_f32, "s--", color="#ff7f0e", lw=1.5, label="Adjoint (float32 increments) vs. central FD"
    )

    h_right = h_vals[h_vals >= 1e-5]
    c_trunc = err_f64[h_vals == h_right[2]][0] / (h_right[2] ** 2)
    ax.loglog(
        h_right, c_trunc * h_right**2, ":", color="#2ca02c", lw=1.4, label=r"Truncation slope $\propto h^2$"
    )

    h_left = h_vals[h_vals <= 1e-5]
    c_round = err_f64[h_vals == h_left[2]][0] * h_left[2]
    ax.loglog(
        h_left,
        c_round / h_left,
        "-.",
        color="#d62728",
        lw=1.4,
        label=r"Cancellation slope $\propto \varepsilon_{\mathrm{mach}} / h$",
    )

    ax.set_xlabel("Central finite-difference step size $h$")
    ax.set_ylabel(r"Relative error $|\delta_{\mathrm{FD}}(h) - \nabla F \cdot \mathbf{d}| / |\nabla F \cdot \mathbf{d}|$")
    ax.set_title("(a) Step-Size Sweep V-Curve (Truncation vs. Round-Off)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.2, loc="upper left")

    ax2 = axes[1]
    all_vals = np.concatenate([fd_iso, adj_iso, fd_ten, adj_ten])
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.08 * (vmax - vmin + 1e-30)
    line_x = np.linspace(vmin - pad, vmax + pad, 100)
    ax2.plot(line_x, line_x, "k--", lw=1.2, label="1:1 exact agreement")
    ax2.scatter(
        fd_iso,
        adj_iso,
        s=52,
        color="#1f77b4",
        marker="o",
        label=f"Isotropic (rel. L2 err = {float(d1['iso_node_rel_err']):.2e})",
    )
    ax2.scatter(
        fd_ten,
        adj_ten,
        s=52,
        color="#d62728",
        marker="^",
        label=f"Tensor SSP (rel. L2 err = {float(d1['tensor_node_rel_err']):.2e})",
    )
    ax2.set_xlabel(r"Central finite difference $\partial F / \partial \rho_k$ at $h^* = 10^{-5}$")
    ax2.set_ylabel(r"Single-run adjoint gradient $(\nabla_{\boldsymbol{\rho}} F)_k$")
    ax2.set_title("(b) Node-by-Node Gradient Parity (Isotropic & Tensor SSP)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5, loc="upper left")

    fig.savefig(fig_dir / "03_fd_step_sweep_vcurve.png", dpi=180)
    plt.close(fig)


def _plot_04_density_filter_projection(fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    d2 = np.load(DATA_DIR / "task_2_sensitivity.npz")
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)

    titles_top = [
        r"(a) Raw density $\boldsymbol{\rho}_{\mathrm{raw}}$",
        r"(b) Conic-filtered $\tilde{\boldsymbol{\rho}}$ ($R=2.4\,\mathrm{px}$)",
        r"(c) Projected $\bar{\boldsymbol{\rho}}$ ($\beta=8, \eta=0.5$)",
    ]
    for ax, arr, ttl in zip(axes[0], (d2["rho_raw"], d2["rho_tilde"], d2["rho_bar"]), titles_top):
        im = ax.imshow(arr.T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(ttl, fontsize=10.2)
        ax.set_xlabel("Design cell $i_x$")
        ax.set_ylabel("Design cell $i_y$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_b0 = axes[1, 0]
    x_proj = d2["x_proj"]
    betas = d2["betas_sweep"]
    cmap_b = plt.cm.plasma(np.linspace(0.1, 0.9, len(betas)))
    for c_col, b_val, curve in zip(cmap_b, betas, d2["proj_curves"]):
        ax_b0.plot(x_proj, curve, color=c_col, lw=1.7, label=rf"$\beta = {int(b_val)}$")
    ax_b0.axvline(0.5, color="#666666", ls=":", lw=1.0)
    ax_b0.set_xlabel(r"Filtered density $\tilde{\rho}$")
    ax_b0.set_ylabel(r"Projected density $\bar{\rho}(\tilde{\rho}; \beta, \eta)$")
    ax_b0.set_title("(d) Smoothed Heaviside Projection Family")
    ax_b0.grid(True, alpha=0.3)
    ax_b0.legend(fontsize=7.8, ncol=2)

    ax_b1 = axes[1, 1]
    for c_col, b_val, dcurve in zip(cmap_b[:5], betas[:5], d2["proj_derivs"][:5]):
        ax_b1.plot(x_proj, dcurve, color=c_col, lw=1.6, label=rf"$\beta = {int(b_val)}$")
    ax_b1.set_xlabel(r"Filtered density $\tilde{\rho}$")
    ax_b1.set_ylabel(r"Chain-rule factor $\partial \bar{\rho} / \partial \tilde{\rho}$")
    ax_b1.set_title(r"(e) Projection Jacobian $\partial\bar{\rho}/\partial\tilde{\rho}$")
    ax_b1.grid(True, alpha=0.3)
    ax_b1.legend(fontsize=8.0)

    ax_b2 = axes[1, 2]
    shifts = d2["shifts"]
    ax_b2.plot(
        shifts,
        d2["dir_deriv_iso"] / np.max(np.abs(d2["dir_deriv_iso"])),
        "o--",
        color="#1f77b4",
        lw=1.6,
        label='smoothing=None',
    )
    ax_b2.plot(
        shifts,
        d2["dir_deriv_tensor"] / np.max(np.abs(d2["dir_deriv_tensor"])),
        "s-",
        color="#d62728",
        lw=1.8,
        label='smoothing="tensor" (anisotropic SSP)',
    )
    ax_b2.set_xlabel("Subpixel wall shift $s / D$")
    ax_b2.set_ylabel("Normalized shape sensitivity $dF/ds$")
    ax_b2.set_title("(f) Subpixel Interface Shift Sensitivity")
    ax_b2.grid(True, alpha=0.3)
    ax_b2.legend(fontsize=8.2)

    fig.savefig(fig_dir / "04_density_filter_projection.png", dpi=180)
    plt.close(fig)


def _plot_05_sensitivity_map(fig_dir: Path) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    d2 = np.load(DATA_DIR / "task_2_sensitivity.npz")
    L = float(d2["L"]) * 1e2
    D = float(d2["D"]) * 1e2
    ext = [-L / 2, L / 2, -L / 2, L / 2]
    ext_des = [-D / 2, D / 2, -D / 2, D / 2]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    ez_re = d2["ez_plane_real"].T
    vmax_e = float(np.max(np.abs(ez_re)) + 1e-30)
    im0 = ax.imshow(ez_re, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax_e, vmax=vmax_e)
    ax.contour(
        d2["eps_z_plane"].T, levels=[3.5], extent=ext, colors="k", linewidths=1.0, linestyles="--"
    )
    ax.add_patch(
        mpatches.Rectangle((-D / 2, -D / 2), D, D, fill=False, edgecolor="#222222", lw=1.5)
    )
    ax.set_title(r"(a) Forward DFT Wavefield $\mathrm{Re}\{\hat{E}_z(x,y; f_0)\}$")
    ax.set_xlabel("$x$ [cm]")
    ax.set_ylabel("$y$ [cm]")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    lam_re = d2["lam_ez_plane_real"].T
    vmax_l = float(np.max(np.abs(lam_re)) + 1e-30)
    im1 = ax.imshow(lam_re, origin="lower", extent=ext, cmap="PuOr_r", vmin=-vmax_l, vmax=vmax_l)
    ax.contour(
        d2["eps_z_plane"].T, levels=[3.5], extent=ext, colors="k", linewidths=1.0, linestyles="--"
    )
    ax.add_patch(
        mpatches.Rectangle((-D / 2, -D / 2), D, D, fill=False, edgecolor="#222222", lw=1.5)
    )
    ax.set_title(r"(b) Adjoint Wavefield $\mathrm{Re}\{\hat{\lambda}_{E,z}(x,y; f_0)\}$")
    ax.set_xlabel("$x$ [cm]")
    ax.set_ylabel("$y$ [cm]")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    gb = d2["g_bar"].T
    vmax_gb = float(np.max(np.abs(gb)) + 1e-30)
    im2 = ax.imshow(
        gb, origin="lower", extent=ext_des, cmap="seismic", vmin=-vmax_gb, vmax=vmax_gb
    )
    ax.set_title(r"(c) Physical Sensitivity $\partial F / \partial \bar{\boldsymbol{\rho}}$ (Design Box)")
    ax.set_xlabel("$x$ [cm]")
    ax.set_ylabel("$y$ [cm]")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    gr = d2["g_raw"].T
    vmax_gr = float(np.max(np.abs(gr)) + 1e-30)
    im3 = ax.imshow(
        gr, origin="lower", extent=ext_des, cmap="seismic", vmin=-vmax_gr, vmax=vmax_gr
    )
    ax.set_title(r"(d) Filtered + Projected Pullback $\partial F / \partial \boldsymbol{\rho}_{\mathrm{raw}}$")
    ax.set_xlabel("$x$ [cm]")
    ax.set_ylabel("$y$ [cm]")
    fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(fig_dir / "05_sensitivity_gradient_map.png", dpi=180)
    plt.close(fig)


def _plot_06_memory_ladder_live(fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    with open(DATA_DIR / "task_3_memory_ladder.json") as f:
        d3 = json.load(f)
    res = d3["live_results"]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)

    ax = axes[0]
    mode_colors = {
        "store": "#1f77b4",
        "checkpoint": "#ff7f0e",
        "auto": "#9467bd",
        "reverse": "#2ca02c",
        "nyquist": "#d62728",
    }
    for r in res:
        kb = r["peak_bytes"] / 1024.0
        t_ms = r["t_total_s"] * 1000.0
        col = mode_colors.get(r["mode"], "#333333")
        ax.scatter(kb, t_ms, s=68, color=col, edgecolor="#111111", zorder=4)
        ax.annotate(r["label"], (kb, t_ms), textcoords="offset points", xytext=(5, 4), fontsize=7.8)
    ax.set_xscale("log")
    ax.set_xlabel("Predicted rung peak memory `sim.memory_stats['peak_bytes']` [KiB]")
    ax.set_ylabel("Total forward + backward wall time [ms]")
    ax.set_title(
        f"(a) Live 2D Bend Memory vs. Wall Time ($N_t={d3['meta']['Nt']}$, $N_e={d3['meta']['design_edges']}$)"
    )
    ax.grid(True, which="both", alpha=0.3)

    ax2 = axes[1]
    dp = d3["revolve_dp_table"]
    palette = {"2": "#d62728", "3": "#ff7f0e", "4": "#1f77b4", "6": "#2ca02c"}
    for s_key, col in palette.items():
        m_arr = np.array(dp[s_key]["m"], dtype=float)
        rev_arr = np.array(dp[s_key]["revolve_extra"], dtype=float) / m_arr
        hal_arr = np.array(dp[s_key]["halving_extra"], dtype=float) / m_arr
        ax2.plot(m_arr, rev_arr, "o-", color=col, lw=1.8, label=f"revolve DP ($s={s_key}$)")
        ax2.plot(
            m_arr, hal_arr, "--", color=col, lw=1.2, alpha=0.65, label=f"naive halving ($s={s_key}$)"
        )
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel(r"Number of time blocks $m = \lceil N_t / b \rceil$")
    ax2.set_ylabel("Recompute ratio $C(m, s) / m$ (extra forward passes)")
    ax2.set_title("(b) Griewank–Walther `revolve` DP vs. Naive Midpoint Halving")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7.6, ncol=2)

    fig.savefig(fig_dir / "06_memory_ladder_measured.png", dpi=180)
    plt.close(fig)


def _plot_07_memory_scaling_and_roundoff(fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    with open(DATA_DIR / "task_3_memory_ladder.json") as f:
        d3 = json.load(f)
    ref = d3.get("reference_study", {})

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), constrained_layout=True)

    ax0 = axes[0]
    if "bend_cpu" in ref:
        bcpu = ref["bend_cpu"]["results"]
        for r in bcpu:
            mb = r["rung_peak_bytes"] / 1e6
            t_tot = r["t_forward"] + r["t_backward"]
            mode = r["mode"]
            col = {
                "store": "#1f77b4",
                "checkpoint": "#ff7f0e",
                "auto": "#9467bd",
                "reverse": "#2ca02c",
                "nyquist": "#d62728",
            }.get(mode, "k")
            ax0.scatter(mb, t_tot, s=55, color=col, edgecolor="#111111", zorder=4)
        ax0.set_xscale("log")
        ax0.set_xlabel("Rung memory `peak_bytes` [MB]")
        ax0.set_ylabel("Forward + Backward wall time [s]")
        ax0.set_title(r"(a) $313\times 313$ Bend, $N_t = 20\,000$ (CPU)")
        ax0.grid(True, which="both", alpha=0.3)

    ax1 = axes[1]
    if "roundoff" in ref:
        ro = ref["roundoff"]
        for bc_type, col, mk, lbl in (
            ("cpml", "#1f77b4", "o-", "Open CPML + recorded interface strip"),
            ("pec", "#2ca02c", "s--", "Closed lossless PEC cavity (resonant)"),
        ):
            sub = [x for x in ro if x["bc"] == bc_type]
            nt_arr = [x["Nt"] for x in sub]
            err_arr = [x["rel_err"] for x in sub]
            ax1.loglog(nt_arr, err_arr, mk, color=col, lw=1.7, label=lbl)
        ax1.axhline(1e-14, color="#666666", ls=":", lw=1.1, label="$10^{-14}$ reference line")
        ax1.set_ylim(1e-16, 1e-12)
        ax1.set_xlabel("Number of leapfrog steps $N_t$")
        ax1.set_ylabel(
            r"Relative error $\|\mathbf{g}_{\mathrm{rev}} - \mathbf{g}_{\mathrm{store}}\|_\infty / \|\mathbf{g}_{\mathrm{store}}\|_\infty$"
        )
        ax1.set_title(r"(b) `reverse` Round-Off Stability ($N_t \leq 64\,000$)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(fontsize=7.8, loc="upper left")

    ax2 = axes[2]
    if "projection" in ref:
        proj = ref["projection"]
        prob_styles = [
            ("ceviche_waveguide_bend", "#1f77b4", r"2D Bend ($313^2, N_t=56.6\mathrm{k}$)"),
            ("ceviche_wdm", "#ff7f0e", r"2D WDM ($825\times 761, N_t=56.6\mathrm{k}$)"),
            ("metalens_3d", "#d62728", r"3D Metalens ($220^2\times 100, N_t=4\mathrm{k}$)"),
        ]
        for key, col, lbl in prob_styles:
            if key not in proj:
                continue
            p_info = proj[key]
            Nt_p = float(p_info["Nt"])
            curve = p_info["ladder"]["checkpoint_curve"]
            gb_arr = [c["bytes"] / 1e9 for c in curve]
            extra_ratio = [c["extra"] / Nt_p for c in curve]
            ax2.plot(gb_arr, extra_ratio, "o-", color=col, lw=1.6, ms=4, label=lbl)
            rev_gb = p_info["ladder"]["reverse"]["bytes"] / 1e9
            ax2.scatter([rev_gb], [0.0], marker="*", s=130, color=col, edgecolor="k", zorder=5)
        ax2.set_xscale("log")
        ax2.set_xlabel(r"Peak memory budget [GB] ($\star$ = `reverse` rung)")
        ax2.set_ylabel(r"Extra forward passes $N_{\mathrm{recompute}} / N_t$")
        ax2.set_title("(c) Projected Memory Ladder (2D & 3D Challenges)")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(fontsize=7.5, loc="upper right")

    fig.savefig(fig_dir / "07_memory_scaling_and_roundoff.png", dpi=180)
    plt.close(fig)


def _plot_08_optimization_convergence(fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    d4 = np.load(DATA_DIR / "task_4_optimization.npz")
    objs = d4["objectives"]
    betas = d4["beta_schedule"]
    bin_idx = d4["binarization"]
    freqs_ghz = d4["freqs_spec"] * 1e-9
    spec_str = d4["spec_straight"]
    spec_ini = d4["spec_init"]
    spec_opt = d4["spec_opt"]
    spec_bin = d4["spec_bin"]

    iters = np.arange(len(objs))
    norm_objs = objs / float(objs[0])

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), constrained_layout=True)

    ax0 = axes[0]
    ax0.plot(
        iters,
        norm_objs,
        "o-",
        color="#1f77b4",
        lw=2.0,
        ms=4.5,
        label=r"$F(\bar{\boldsymbol{\rho}}^{(k)}) / F(\bar{\boldsymbol{\rho}}^{(0)})$",
    )
    unique_betas = np.unique(betas)
    stage_colors = ["#e8f4f8", "#fff3e0", "#fce8e6", "#ede7f6"]
    for idx_b, b_val in enumerate(unique_betas):
        mask = np.where(betas == b_val)[0]
        i0, i1 = int(mask[0]), int(mask[-1])
        ax0.axvspan(
            i0 - 0.4,
            i1 + 0.4,
            color=stage_colors[idx_b % len(stage_colors)],
            alpha=0.55,
            label=rf"Stage $\beta = {int(b_val)}$",
        )
    ax0.set_xlabel("Optimization iteration $k$")
    ax0.set_ylabel("Normalized output guided power $F^{(k)} / F^{(0)}$")
    ax0.set_title(r"(a) Objective Trajectory & $\beta$-Continuation")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8.0, loc="lower right")

    ax1 = axes[1]
    ax1.plot(
        iters,
        bin_idx * 100.0,
        "s-",
        color="#2ca02c",
        lw=1.8,
        ms=4.0,
        label=r"Binarization $1 - 4\langle\bar{\rho}(1-\bar{\rho})\rangle$ [%]",
    )
    ax1.set_xlabel("Optimization iteration $k$")
    ax1.set_ylabel("Binarization index [%]", color="#2ca02c")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1_r = ax1.twinx()
    ax1_r.semilogy(
        iters,
        d4["grad_norms"] / float(d4["grad_norms"][0]),
        "^--",
        color="#d62728",
        lw=1.4,
        ms=4.0,
        label=r"Relative $\|\nabla_{\boldsymbol{\rho}_{\mathrm{raw}}} F\|_2$",
    )
    ax1_r.set_ylabel(
        r"Normalized gradient norm $\|\nabla F\|_2 / \|\nabla F^{(0)}\|_2$", color="#d62728"
    )
    ax1.set_title("(b) Binarization Progression & Gradient Norm")

    ax2 = axes[2]
    t_init_db = 10.0 * np.log10(np.maximum(spec_ini / spec_str, 1e-6))
    t_opt_db = 10.0 * np.log10(np.maximum(spec_opt / spec_str, 1e-6))
    t_bin_db = 10.0 * np.log10(np.maximum(spec_bin / spec_str, 1e-6))
    ax2.plot(
        freqs_ghz, t_init_db, "o--", color="#7f7f7f", lw=1.6, label=r"Initial uniform ($\bar{\rho}=0.5$)"
    )
    ax2.plot(
        freqs_ghz, t_opt_db, "s-", color="#1f77b4", lw=2.0, label=r"Optimized projected ($\beta=8$)"
    )
    ax2.plot(
        freqs_ghz,
        t_bin_db,
        "^:",
        color="#d62728",
        lw=1.7,
        label=r"Thresholded binary ($\bar{\rho} \in \{0,1\}$)",
    )
    ax2.axvline(F0 * 1e-9, color="#222222", ls=":", lw=1.2, label=r"$f_0 = 3.0\,\mathrm{GHz}$")
    ax2.set_xlabel("Frequency $f$ [GHz]")
    ax2.set_ylabel(r"Relative transmission $10\log_{10}(P_{\mathrm{out}}/P_{\mathrm{straight}})$ [dB]")
    ax2.set_title("(c) Broadband Bend Transmission Spectrum")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7.8, loc="lower left")

    fig.savefig(fig_dir / "08_toy_optimization_convergence.png", dpi=180)
    plt.close(fig)


def _plot_09_initial_vs_optimized_fields(fig_dir: Path) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    d4 = np.load(DATA_DIR / "task_4_optimization.npz")
    L = float(d4["L"]) * 1e2
    D = float(d4["D"]) * 1e2
    ext = [-L / 2, L / 2, -L / 2, L / 2]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)

    vmax_re = float(
        max(np.max(np.abs(d4["ez_plane_init_real"])), np.max(np.abs(d4["ez_plane_opt_real"]))) + 1e-30
    )
    vmax_int = float(
        max(np.max(d4["ez_plane_init_abs"] ** 2), np.max(d4["ez_plane_opt_abs"] ** 2)) + 1e-30
    )

    rows = [
        (
            0,
            r"Initial Uniform Greyscale ($\bar{\rho}=0.5$)",
            d4["eps_plane_init"].T,
            d4["ez_plane_init_real"].T,
            (d4["ez_plane_init_abs"].T) ** 2,
        ),
        (
            1,
            r"Optimized $90^\circ$ Waveguide Bend",
            d4["eps_plane_opt"].T,
            d4["ez_plane_opt_real"].T,
            (d4["ez_plane_opt_abs"].T) ** 2,
        ),
    ]
    for r_idx, row_label, eps_map, ez_re, ez_int in rows:
        ax0 = axes[r_idx, 0]
        im0 = ax0.imshow(eps_map, origin="lower", extent=ext, cmap="Greys", vmin=1.0, vmax=6.0)
        ax0.add_patch(
            mpatches.Rectangle(
                (-D / 2, -D / 2), D, D, fill=False, edgecolor="#d62728", lw=1.5, ls="--"
            )
        )
        ax0.set_title(f"{row_label}\n" + r"Permittivity $\varepsilon_r(x,y)$", fontsize=10.0)
        ax0.set_xlabel("$x$ [cm]")
        ax0.set_ylabel("$y$ [cm]")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = axes[r_idx, 1]
        im1 = ax1.imshow(
            ez_re, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax_re, vmax=vmax_re
        )
        ax1.contour(eps_map, levels=[3.5], extent=ext, colors="k", linewidths=0.9, linestyles=":")
        ax1.add_patch(
            mpatches.Rectangle((-D / 2, -D / 2), D, D, fill=False, edgecolor="#222222", lw=1.2)
        )
        ax1.set_title(
            f"{row_label}\n" + r"Steady-State $\mathrm{Re}\{\hat{E}_z(f_0)\}$", fontsize=10.0
        )
        ax1.set_xlabel("$x$ [cm]")
        ax1.set_ylabel("$y$ [cm]")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = axes[r_idx, 2]
        im2 = ax2.imshow(
            ez_int / vmax_int, origin="lower", extent=ext, cmap="inferno", vmin=0.0, vmax=1.0
        )
        ax2.contour(eps_map, levels=[3.5], extent=ext, colors="w", linewidths=0.8, linestyles=":")
        ax2.add_patch(
            mpatches.Rectangle((-D / 2, -D / 2), D, D, fill=False, edgecolor="cyan", lw=1.2)
        )
        ax2.set_title(
            f"{row_label}\n" + r"Normalized Intensity $|\hat{E}_z(f_0)|^2$", fontsize=10.0
        )
        ax2.set_xlabel("$x$ [cm]")
        ax2.set_ylabel("$y$ [cm]")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.savefig(fig_dir / "09_initial_vs_optimized_fields.png", dpi=180)
    plt.close(fig)


def _plot_10_design_evolution_gif(fig_dir: Path) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt

    d4 = np.load(DATA_DIR / "task_4_optimization.npz")
    snaps = d4["snapshots"].astype(np.float32)
    gsnaps = d4["grad_snapshots"].astype(np.float32)
    objs = d4["objectives"]
    betas = d4["beta_schedule"]
    norm_objs = objs / float(objs[0])
    n_iters = len(objs)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), constrained_layout=True)

    im_rho = axes[0].imshow(
        1.0 + 5.0 * snaps[0].T, origin="lower", cmap="Greys", vmin=1.0, vmax=6.0
    )
    axes[0].set_title(r"Permittivity $\varepsilon_r(\bar{\boldsymbol{\rho}}^{(k)})$")
    axes[0].set_xlabel("Design cell $i_x$")
    axes[0].set_ylabel("Design cell $i_y$")
    fig.colorbar(im_rho, ax=axes[0], fraction=0.046, pad=0.04)

    im_grad = axes[1].imshow(gsnaps[0].T, origin="lower", cmap="seismic", vmin=-1.0, vmax=1.0)
    axes[1].set_title(r"Normalized Pullback $\nabla_{\boldsymbol{\rho}_{\mathrm{raw}}} F$")
    axes[1].set_xlabel("Design cell $i_x$")
    axes[1].set_ylabel("Design cell $i_y$")
    fig.colorbar(im_grad, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].plot(np.arange(n_iters), norm_objs, "-", color="#cccccc", lw=1.5)
    (line_prog,) = axes[2].plot([0], [norm_objs[0]], "o-", color="#1f77b4", lw=2.0)
    axes[2].set_xlim(0, n_iters - 1)
    axes[2].set_ylim(0.8, float(np.max(norm_objs)) * 1.1)
    axes[2].set_xlabel("Iteration $k$")
    axes[2].set_ylabel("Normalized Transmission $F^{(k)} / F^{(0)}$")
    axes[2].grid(True, alpha=0.3)
    title_txt = axes[2].set_title(
        f"Iter 0/{n_iters - 1} (beta={int(betas[0])}): {norm_objs[0]:.2f}x"
    )

    def _update(frame: int):
        im_rho.set_data(1.0 + 5.0 * snaps[frame].T)
        im_grad.set_data(gsnaps[frame].T)
        line_prog.set_data(np.arange(frame + 1), norm_objs[: frame + 1])
        title_txt.set_text(
            f"Iter {frame}/{n_iters - 1} (beta={int(betas[frame])}): {norm_objs[frame]:.2f}x"
        )
        return im_rho, im_grad, line_prog, title_txt

    anim = FuncAnimation(fig, _update, frames=n_iters, interval=180, blit=False)
    anim.save(fig_dir / "10_design_evolution.gif", writer=PillowWriter(fps=6), dpi=110)
    plt.close(fig)


def plot_all(fig_dir: Path | None = None) -> list[Path]:
    """Render all tutorial figures and setup schematics from saved data in ``data/``."""
    target_dir = FIG_DIR if fig_dir is None else Path(fig_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    _plot_00_setup_schematics(target_dir)
    _plot_01_schematic(target_dir)
    _plot_02_dot_product(target_dir)
    _plot_03_fd_vcurve(target_dir)
    _plot_04_density_filter_projection(target_dir)
    _plot_05_sensitivity_map(target_dir)
    _plot_06_memory_ladder_live(target_dir)
    _plot_07_memory_scaling_and_roundoff(target_dir)
    _plot_08_optimization_convergence(target_dir)
    _plot_09_initial_vs_optimized_fields(target_dir)
    _plot_10_design_evolution_gif(target_dir)
    return sorted(target_dir.glob("*"))


# %% [markdown]
# ---
#
# ## Section 7 · Results, Limits & Failure Modes, and Verified References
#
# ### How Good Can a Design Be? Fundamental and Computational Bounds
#
# Because gradient-based topology optimization is non-convex and converges to local
# optima, a natural question is **how close a discovered structure comes to the best
# performance physically achievable** within a given design footprint and material
# susceptibility $\chi = \varepsilon_r - 1$:
#
# - **Global conservation bounds in absorptive media**: Miller et al. (2016) derived
#   shape-independent upper bounds on absorption, scattering, extinction, and near-field
#   local density of states by enforcing optical-theorem energy conservation on the
#   induced polarization currents.
# - **Broadband Kramers–Kronig bounds**: Shim, Fan, Johnson & Miller (2019) used
#   causality and passivity in the complex frequency plane to bound near-field optical
#   response integrated over any finite bandwidth via a single complex-frequency
#   evaluation.
# - **Local conservation-law bounds**: Kuang & Miller (2020) generalized global optical
#   theorems to an infinite family of spatially localized power-conservation constraints,
#   yielding tighter computational upper bounds on light–matter interactions via
#   Lagrangian duality.
# - **Sparse QCQP relaxations of differential Maxwell equations**: Gertler, Kuang,
#   Christie, Li & Miller (2025) showed that differential-equation formulations of
#   photonic design problems (such as waveguide bends, splitters, and metasurfaces)
#   are sparse quadratically constrained quadratic programs (QCQPs) whose semidefinite
#   relaxations provide rigorous global performance upper bounds against which local
#   adjoint-optimized designs can be benchmarked.
#
# ### Limits & Failure Modes
#
# 1. **High-$Q$ resonances and truncated DFT windows**:
#    A `DFTMonitor` accumulates over a finite horizon $N_t \Delta t$. If an
#    intermediate topology traps a high-$Q$ cavity mode with ring-down time
#    $\tau = 2Q/\omega_0 \gg N_t \Delta t$, the discrete adjoint gives the exact
#    derivative of the *truncated* sum rather than the CW steady state. Increase
#    $N_t$ or optimize across multiple frequencies to damp narrowband ringing.
# 2. **Conductive or dispersive interiors in `memory="reverse"`**:
#    Time-reversing the interior leapfrog requires a lossless, nondispersive
#    interior ($\sigma = 0$, no damped poles). When interior loss or dispersion is
#    present, `AdjointSimulation` raises a `ValueError` for `memory="reverse"` and
#    selects `checkpoint` when `memory="auto"`.
# 3. **Premature $\beta$-projection saturation**:
#    Because $\partial \bar{\rho} / \partial \tilde{\rho} \propto \beta\,\operatorname{sech}^2(\beta(\tilde{\rho}-\eta))$
#    vanishes exponentially at large $\beta$ for $\tilde{\rho}$ away from $\eta$,
#    starting at $\beta \ge 32$ stalls gradient ascent. Use a continuation
#    schedule ($\beta = 2 \to 4 \to 8 \to 16 \to 32$).
# 4. **Nyquist decimation (`memory="nyquist"`, Park, Miller & Chung 2026) on broadband transients**:
#    Decimation $d$ assumes $\Delta \mathbf{D}^{(n)}$ has negligible spectral
#    content above $1 / (2 d \Delta t)$. Impulsive sources or high-frequency
#    transients alias into $10^{-3}\text{--}10^{-1}$ gradient errors; use
#    `checkpoint` or `reverse` when quasi-Newton line searches require machine
#    precision.
#
# ### Verified References
#
# 1. M. B. Giles and N. A. Pierce, "An introduction to the adjoint approach to design,"
#    *Flow, Turbulence and Combustion* **65**, 393–415 (2000),
#    DOI: [`10.1023/A:1011430410075`](https://doi.org/10.1023/A:1011430410075).
# 2. O. D. Miller, "Photonic Design: From Fundamental Solar Cell Physics to Computational Inverse Design",
#    PhD thesis, UC Berkeley (2012), arXiv:[`1308.0212`](https://arxiv.org/abs/1308.0212).
# 3. C. M. Lalau-Keraly, S. Bhargava, O. D. Miller and E. Yablonovitch, "Adjoint shape optimization applied to
#    electromagnetic design", *Opt. Express* **21**, 21693 (2013),
#    DOI: [`10.1364/OE.21.021693`](https://doi.org/10.1364/OE.21.021693).
# 4. V. Ganapati, O. D. Miller and E. Yablonovitch, "Light Trapping Textures Designed by Electromagnetic Optimization for
#    Subwavelength Thick Solar Cells", *IEEE J. Photovoltaics* **4**, 175–182 (2014),
#    DOI: [`10.1109/JPHOTOV.2013.2280340`](https://doi.org/10.1109/JPHOTOV.2013.2280340).
# 5. O. D. Miller and E. Yablonovitch, "Inverse Optical Design", in *Encyclopedia of Applied and Computational
#    Mathematics*, pp. 729–732 (Springer, 2015),
#    DOI: [`10.1007/978-3-540-70529-1_45`](https://doi.org/10.1007/978-3-540-70529-1_45).
# 6. S. Molesky, Z. Lin, A. Y. Piggott, W. Jin, J. Vučković, and A. W. Rodriguez,
#    "Inverse design in nanophotonics," *Nature Photonics* **12**, 659–670 (2018),
#    DOI: [`10.1038/s41566-018-0246-9`](https://doi.org/10.1038/s41566-018-0246-9).
# 7. T. W. Hughes, I. A. D. Williamson, M. Minkov, and S. Fan, "Forward-mode
#    differentiation of Maxwell's equations," *ACS Photonics* **6**, 3010–3016 (2019),
#    arXiv:[`1908.10507`](https://arxiv.org/abs/1908.10507),
#    DOI: [`10.1021/acsphotonics.9b01238`](https://doi.org/10.1021/acsphotonics.9b01238).
# 8. B. Bourdin, "Filters in topology optimization," *International Journal for
#    Numerical Methods in Engineering* **50**, 2143–2158 (2001),
#    DOI: [`10.1002/nme.116`](https://doi.org/10.1002/nme.116).
# 9. F. Wang, B. S. Lazarov, and O. Sigmund, "On projection methods, convergence and
#    robust formulations in topology optimization," *Structural and Multidisciplinary
#    Optimization* **43**, 767–784 (2011),
#    DOI: [`10.1007/s00158-010-0602-y`](https://doi.org/10.1007/s00158-010-0602-y).
# 10. A. Farjadpour, D. Roundy, A. Rodriguez, M. Ibanescu, P. Bermel, J. D. Joannopoulos,
#     S. G. Johnson, and G. W. Burr, "Improving accuracy by subpixel smoothing in the
#     finite-difference time domain," *Optics Letters* **31**, 2972–2974 (2006),
#     DOI: [`10.1364/OL.31.002972`](https://doi.org/10.1364/OL.31.002972).
# 11. A. Griewank and A. Walther, "Algorithm 799: revolve: an implementation of
#     checkpointing for the reverse or adjoint mode of computational differentiation,"
#     *ACM Transactions on Mathematical Software* **26**, 19–45 (2000),
#     DOI: [`10.1145/347837.347846`](https://doi.org/10.1145/347837.347846).
# 12. R. Tang, J. Lim, M. Ossiander, X. Yin, and F. Capasso, "Time Reversal
#     Differentiation of FDTD for Photonic Inverse Design," *ACS Photonics* **10**,
#     4140–4150 (2023),
#     DOI: [`10.1021/acsphotonics.3c00694`](https://doi.org/10.1021/acsphotonics.3c00694).
# 13. Y. Mahlau, F. Schubert, K. Bethmann, R. Caspary, A. B. Fallahkhair, M. M.ermann,
#     and B. Rosenhahn, "FDTDX: High-performance open-source FDTD in JAX,"
#     arXiv:[`2412.12360`](https://arxiv.org/abs/2412.12360); *Journal of Open Source
#     Software* **11**, 8912 (2026),
#     DOI: [`10.21105/joss.08912`](https://doi.org/10.21105/joss.08912).
# 14. M. Park, O. D. Miller and H. Chung, "Nyquist-Sampled Time-Domain Adjoint FDTD for Memory-Efficient Broadband
#     Nanophotonic Inverse Design", arXiv:[`2607.08159`](https://arxiv.org/abs/2607.08159) (2026).
# 15. O. D. Miller, A. G. Polimeridis, M. T. Homer Reid, C. W. Hsu, B. G. DeLacy, J. D. Joannopoulos, M. Soljačić and
#     S. G. Johnson, "Fundamental limits to optical response in absorptive systems", *Opt. Express* **24**, 3329 (2016),
#     DOI: [`10.1364/OE.24.003329`](https://doi.org/10.1364/OE.24.003329).
# 16. H. Shim, L. Fan, S. G. Johnson and O. D. Miller, "Fundamental Limits to Near-Field Optical Response over Any
#     Bandwidth", *Phys. Rev. X* **9**, 011043 (2019),
#     DOI: [`10.1103/PhysRevX.9.011043`](https://doi.org/10.1103/PhysRevX.9.011043).
# 17. Z. Kuang and O. D. Miller, "Computational Bounds to Light–Matter Interactions via Local Conservation Laws",
#     *Phys. Rev. Lett.* **125**, 263607 (2020),
#     DOI: [`10.1103/PhysRevLett.125.263607`](https://doi.org/10.1103/PhysRevLett.125.263607).
# 18. L. Gertler, Z. Kuang, C. Christie, H. Li and O. D. Miller, "Many photonic design problems are sparse QCQPs",
#     *Sci. Adv.* **11**, eadl3237 (2025),
#     DOI: [`10.1126/sciadv.adl3237`](https://doi.org/10.1126/sciadv.adl3237).
# 19. M. F. Schubert, A. K. C. Cheung, I. A. D. Williamson, A. Spyra, and D. H. Alexander,
#     "invrs-gym: a toolkit for nanophotonic inverse design research,"
#     arXiv:[`2410.24132`](https://arxiv.org/abs/2410.24132) (2024).

# %%
TASK_RUNNERS = {
    0: run_task_0_dot_product,
    1: run_task_1_fd_vcurve,
    2: run_task_2_sensitivity_and_parametrization,
    3: run_task_3_memory_ladder,
    4: run_task_4_toy_optimization,
}


def run_tasks(quick: bool = False, task_index: int | None = None) -> dict[int, dict]:
    """Execute all tasks (or a single task by index) and return their summaries."""
    table = get_task_table(quick=quick)
    selected = table if task_index is None else [t for t in table if t["index"] == task_index]
    if not selected:
        raise ValueError(f"Unknown task index {task_index!r}; valid indices are 0..{len(table) - 1}.")
    summaries = {}
    for cfg in selected:
        idx = int(cfg["index"])
        t0 = time.perf_counter()
        res = TASK_RUNNERS[idx](cfg)
        dt_s = time.perf_counter() - t0
        summaries[idx] = {"name": cfg["name"], "elapsed_s": round(dt_s, 3), "summary": res}
        print(f"[task {idx}: {cfg['name']}] completed in {dt_s:.2f} s", flush=True)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tutorial 9.1: Adjoint gradients for time-domain Maxwell solvers."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run fast validation tasks (< 1 min).")
    mode.add_argument("--full", action="store_true", help="Run full-resolution tasks.")
    parser.add_argument("--task", type=int, default=None, help="Run only task index I.")
    parser.add_argument("--out", type=Path, default=None, help="Override output data directory.")
    parser.add_argument("--plot", action="store_true", help="Regenerate figures from data/ only.")
    parser.add_argument("--list-tasks", action="store_true", help="Print task table as JSON and exit.")
    args = parser.parse_args(argv)

    global DATA_DIR
    if args.out is not None:
        DATA_DIR = Path(args.out)

    quick = not args.full
    if args.list_tasks:
        print(json.dumps(get_task_table(quick=quick), indent=2))
        return 0

    matplotlib.use("Agg")
    if args.plot:
        figs = plot_all()
        print(f"Generated {len(figs)} figures in {FIG_DIR}")
        return 0

    run_tasks(quick=quick, task_index=args.task)
    if args.task is None:
        figs = plot_all()
        print(f"Generated {len(figs)} figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
