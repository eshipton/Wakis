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

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/07_light_sources/07-09_bic_laser/bic_laser.ipynb)
#
# # Tutorial 7.9 · Lasing from a bound state in the continuum
# <h3 align="center">Infinite-lattice Bloch BICs ($Q \propto k^{-2}$ and 3D $A_1$ topological polarization vortex $q = +1$), finite $N$-period strip scaling ($Q_1 \propto N^p$), active threshold &amp; single-mode lasing (`wakis.gain`), `Near2FarMonitor` doughnut far-field beam, and symmetry-breaking quenching ($\alpha \ne 0$)</h3>
# <p align="center"><i>Erik Shipton &middot; Wakis tutorial series (fork of CERN's Wakis)</i></p>
#
# ---
#
# ### Abstract
#
# **Photonic bound states in the continuum (BICs) trap electromagnetic energy inside
# an open radiation continuum without mirrors, enabling ultracompact surface-emitting
# lasers with tailored topological far fields [Kodigala *et al.*, *Nature* 541, 196
# (2017); Hsu *et al.*, *Nat. Rev. Mater.* 1, 16048 (2016)]. Building directly on
# Tutorial 6.3 (`06-03_quasi_bic_metasurface`) and Wakis's time-domain auxiliary
# differential equation (ADE) gain module (`wakis.gain`), this tutorial models the
# complete physics of a BIC laser from the infinite Bloch lattice to nonlinear
# single-mode lasing in a finite $N$-period resonator strip array:**
#
# 1. **Infinite lattice (Bloch unit cell):** We demonstrate a symmetry-protected BIC
#    at the $\Gamma$ point ($\lambda_{\mathrm{BIC}} = 1573.8847\;\mathrm{nm}$,
#    $Q_\Gamma = 1.573 \times 10^9$), verify the off-normal momentum-space radiation
#    scaling $Q(k_x) \propto k_x^{-2}$ (fitted near-$\Gamma$ slope $-1.9879$, $0.61\%$
#    from $-2.0000$), sweep our dimer bar width $w_0 \in \{140, 155, 170, 185, 200\}\;\mathrm{nm}$
#    in 2D FIT to show that the symmetric ($\alpha = 0$) BIC remains radiation-free
#    ($Q_{\mathrm{sym}} \ge 8.55 \times 10^8$) across the entire geometric tuning range
#    whereas the $\alpha = 0.10$ quasi-BIC radiates with $Q_{\mathrm{qBIC}}(w_0; n_x = 48) \in [3093.51, 8504.60]$
#    ($[2313.20, 12672.61]$ at $n_x = 32$, where $w_0 = 200\;\mathrm{nm}$ is unconverged by $49\%$),
#    and verify a resolved 3D cylinder $A_1$-like
#    TE BIC ($f_{A_1} = 196.3574\;\mathrm{THz}$, $\lambda_{A_1} = 1526.77\;\mathrm{nm}$,
#    $\mathrm{cpw}_{A_1} = 21.38 \ge 20$ at $n_x = n_y = 30$) whose $\Gamma$-point
#    quality factor increases from $Q_\Gamma(n_{\mathrm{PML}}=6) = 24345.8$ to
#    $Q_\Gamma > 10^7$ (lower bound from a 22-cycle record; `harminv` value $19245231.8$ at
#    $n_{\mathrm{PML}}=10$, $790.5\times$, consistent with near-field CPML absorption), scales off-$\Gamma$ as
#    $Q(k_\parallel) \propto k_\parallel^{-2}$ with log–log slope $-1.9560$ ($2.20\%$ from
#    $-2.0000$), and winds around an off-mirror wavevector loop ($\phi_k = 17^\circ + 45^\circ m$)
#    with quantized topological charge $q_{A_1} = +1.0000$ via `compute_topological_charge`
#    [Zhen *et al.*, *Phys. Rev. Lett.* 113, 257401 (2014)].
# 2. **Finite $N$-period strip arrays:** Terminating the 2D lattice along $x$ into a
#    finite strip array of $N \in \{4, 6, 8, 10, 12\}$ unit cells (periodic along $y$,
#    open CPML boundaries on $\pm x$ and $\pm z$) introduces momentum spread
#    $\Delta k = \pi / (N p)$ and lateral edge radiation, giving $Q_1(N) \propto N^p$
#    with measured $p = 2.5518$ compared against the $N^2$ vertical momentum-spread
#    baseline ($p_{\mathrm{baseline}} = 2.0$). Across standing-wave envelope orders
#    $m \in \{1, 2, 3\}$ at fixed $N = 8$, the fundamental single-lobe mode ($m = 1$,
#    $k_1 = \pi/(N p)$ closest to $\Gamma$) achieves the highest quality factor
#    ($Q_1 = 528.34$ vs $Q_2 = 138.07$ and $Q_3 = 65.28$, matching the fixed-$N$ scaling
#    $Q_m(N) \approx Q_1(N)/m^2$ with $Q_1/Q_2 = 3.827$, $4.33\%$ from $4.000$).
# 3. **Threshold & single-mode lasing (`wakis.gain` & `wakis.near2far`):** Coupling
#    `SaturableGain` and `FourLevelGain(spontaneous_emission=True, seed=42, time_scale=80.0)`
#    into the active semiconductor resonators, we verify the cold-cavity threshold
#    identity $g_{\mathrm{th}} = 2\pi n_{\mathrm{cyl}} / (\lambda_0 \Gamma_{\mathrm{conf}} Q)$
#    to within $0.06\%$ in 2D FIT (`19.929 cm⁻¹` vs analytic `19.940 cm⁻¹`), reproduce
#    the nonlinear Light-In / Light-Out (L–I) curve against our spatial-hole-burning
#    integral theory ($0.06\%$ relative error at $g_0/g_{\mathrm{th}} = 2.0$, $1.09\%$
#    maximum relative error across all above-threshold points) and noise-seeded
#    four-level population inversion clamping ($\Delta N(t)/\Delta N_{\mathrm{th}} \to 1.0924$
#    from peak $1.5763$), and project the recorded 2D FIT lasing fields with
#    `Near2FarMonitor` to verify a dark normal-incidence nodal null
#    ($I(0^\circ)/I_{\max} = 6.51 \times 10^{-4}$) and twin radiation lobes at
#    $\theta_{\mathrm{peak,FIT}} = \pm 10.05^\circ$ (vs continuum theory $\pm 11.25^\circ$,
#    $10.67\%$ relative difference).
# 4. **Symmetry-breaking mutation ($\alpha \ne 0$) & convergence:** Directly simulating
#    the finite $N = 8$ strip array in 2D FIT across width asymmetries
#    $\alpha = \Delta w / w_0 \in \{0.0, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20\}$, we show
#    that breaking $\mathcal{C}_2$ symmetry degrades the finite-strip quality factor from
#    $Q_{\mathrm{tot}}(0, 8) = 528.34$ to $Q_{\mathrm{tot}}(0.20, 8) = 362.64$
#    (matching $Q_{\mathrm{tot}}^{-1}(\alpha, N) \approx Q_{\mathrm{edge}}^{-1}(0, N) + Q_0^{-1}\alpha^2$
#    with $Q_0 = 47.11$ to within $0.71\%$ maximum error), raises the threshold gain
#    quadratically as $g_{\mathrm{th}}(\alpha, N) = g_{\mathrm{th}}(0, N) + \kappa_{\mathrm{th}}\alpha^2$
#    (fitted log–log slope $+1.9970$, $0.15\%$ from $+2.0000$), and quenches lasing at a
#    fixed pump rate, backed by a 3-resolution spatial mesh convergence study
#    ($n_x \in \{32, 48, 64\}$, $\mathrm{cpw}_{\mathrm{Si}} \in \{21.28, 31.92, 42.57\} \ge 20$)
#    with Richardson extrapolation for $Q_{\mathrm{rad}}(\alpha=0.10)$ and $g_{\mathrm{th}}(\alpha=0.10)$.
#
# ### Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. **derive** why a symmetry-protected BIC at $\Gamma$ ($\mathbf{k}_\parallel = \mathbf{0}$)
#    has radiative quality factor $Q(k_\parallel) \propto k_\parallel^{-2}$ for small
#    off-normal wavevectors and carries a conserved integer topological charge
#    $q = \frac{1}{2\pi}\oint_C \mathrm{d}\psi(\mathbf{k}_\parallel) \in \mathbb{Z}$
#    in its far-field polarization vortex;
# 2. **contrast** in 2D FIT a symmetry-protected BIC ($\alpha = 0$, whose radiative
#    channel at $\Gamma$ remains forbidden across continuous changes in mean bar width
#    $w_0 \in [140, 200]\;\mathrm{nm}$) with an asymmetric quasi-BIC ($\alpha = 0.10$,
#    whose radiative quality factor $Q_{\mathrm{qBIC}}(w_0)$ and threshold gain vary
#    strongly with geometry);
# 3. **explain** how truncating an infinite 2D BIC lattice into a finite $N$-period
#    strip array quantizes the envelope wavevector at $k_m = m \pi / (N p)$, yielding
#    $Q_1(N) \propto N^p$ across strip sizes $N$ while scaling at fixed $N$ as
#    $Q_m(N) \approx Q_1(N) / m^2$ so that the fundamental $m = 1$ envelope mode always
#    exhibits the lowest lasing threshold;
# 4. **couple** active gain media (`wakis.gain.SaturableGain` and noise-seeded
#    `FourLevelGain(spontaneous_emission=True, seed=42, time_scale=80.0)`) to a high-$Q$
#    dielectric metasurface in `SolverFIT3D`, verify the threshold gain formula
#    $g_{\mathrm{th}} = 2\pi n / (\lambda_0 \Gamma_{\mathrm{conf}} Q)$, derive the
#    spatial-hole-burning steady-state L–I curve, and project the far-field doughnut
#    beam using `wakis.near2far.Near2FarMonitor`;
# 5. **predict and verify** directly in 2D FIT how breaking in-plane inversion symmetry
#    ($\alpha = \Delta w / w_0 \ne 0$) in a finite $N = 8$ strip array degrades
#    $Q_{\mathrm{tot}}(\alpha, N)$, induces a quadratic lasing threshold penalty
#    $\Delta g_{\mathrm{th}}(\alpha, N) \propto \alpha^2$, and quenches lasing.
#
# ### Contents
#
# 0. [Set-up & command-line modes (`--quick`, `--full`, `--task`, `--list-tasks`, `--plot`)](#0-set-up)
# 1. [Theory: BIC dispersion, topological charge, finite-strip scaling, and active lasing threshold](#1-theory)
# 2. [Simulation setup, graded mesh, and 3D preflight (`wakis.estimate`)](#2-setup)
# 3. [Results — Part 1: Infinite lattice ($Q \propto k^{-2}$, bar-width BIC sweep, and 3D polarization vortex)](#3-infinite-lattice)
# 4. [Results — Part 2: Finite $N$-period strip arrays ($Q_1 \propto N^p$ scaling and mode selection)](#4-finite-arrays)
# 5. [Results — Part 3: Lasing threshold, single-mode lasing (`wakis.gain`), and `Near2FarMonitor` doughnut beam](#5-threshold-and-lasing)
# 6. [Results — Part 4: Symmetry-breaking mutation ($\alpha \ne 0$) and mesh convergence](#6-mutation-and-convergence)
# 7. [Quantitative summary](#7-quantitative-summary)
# 8. [Limits & failure modes](#8-limits-and-failure-modes)
# 9. [References](#9-references)

# %% [markdown]
# <a id="0-set-up"></a>
# ## 0. Set-up & Command-Line Modes
#
# Locally, run this tutorial inside the project virtual environment (always wrapped
# with `memguard`):
#
# ```bash
# # Fast verification & figure generation (sequential FIT tasks, < 2000 MB RAM):
# /root/computational-electromagnetics/bin/memguard -m 2000 -- \
#   /root/computational-electromagnetics/wakis/.venv/bin/python \
#   tutorials/07_light_sources/07-09_bic_laser/bic_laser.py --quick
#
# # Full resolution & extended finite-array sweep:
# /root/computational-electromagnetics/bin/memguard -m 2000 -- \
#   /root/computational-electromagnetics/wakis/.venv/bin/python \
#   tutorials/07_light_sources/07-09_bic_laser/bic_laser.py --full
#
# # Regenerate figures and summary table from cached data/task_*.npz:
# /root/computational-electromagnetics/bin/memguard -m 2000 -- \
#   /root/computational-electromagnetics/wakis/.venv/bin/python \
#   tutorials/07_light_sources/07-09_bic_laser/bic_laser.py --plot
# ```

# %%
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run("apt-get -qq install -y xvfb > /dev/null", shell=True, check=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/eshipton/Wakis.git@main",
        ],
        check=True,
    )

TUTORIAL_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import numpy as np  # noqa: E402
import _helpers as h  # noqa: E402

DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# <a id="1-theory"></a>
# ## 1. Theory: BIC Dispersion, Topological Charge, Finite-Size Scaling, and Lasing Threshold
#
# ### 1.1 Symmetry-Protected BICs at $\Gamma$ and $Q(k_\parallel) \propto k_\parallel^{-2}$ (Derived Here)
#
# Consider a periodic dielectric slab or metasurface with lattice period $p < \lambda_0$
# (sub-diffractive regime, so only the zeroth-order $(m_x, m_y) = (0, 0)$ diffraction
# channel lies inside the free-space light cone $|\mathbf{k}_\parallel| < k_0 = \omega/c$).
# By Bloch's theorem, the electric field of a quasi-normal mode at in-plane wavevector
# $\mathbf{k}_\parallel = (k_x, k_y)$ takes the form
#
# $$
# \mathbf{E}_{\mathbf{k}_\parallel}(\mathbf{r}_\parallel, z)
# = e^{i \mathbf{k}_\parallel \cdot \mathbf{r}_\parallel}\,\mathbf{u}_{\mathbf{k}_\parallel}(\mathbf{r}_\parallel, z),
# $$
#
# where $\mathbf{u}_{\mathbf{k}_\parallel}(\mathbf{r}_\parallel + \mathbf{R}, z) = \mathbf{u}_{\mathbf{k}_\parallel}(\mathbf{r}_\parallel, z)$
# is lattice-periodic. The complex radiation amplitude $\mathbf{c}(\mathbf{k}_\parallel) = (c_x(\mathbf{k}_\parallel), c_y(\mathbf{k}_\parallel))$
# into the outgoing $(0,0)$ plane wave in the upper half-space ($z > h/2$) is given by
# the unit-cell spatial average of the transverse Bloch envelope:
#
# $$
# \mathbf{c}(\mathbf{k}_\parallel)
# = \frac{1}{A_{\mathrm{cell}}} \iint_{\mathrm{cell}}
#   \mathbf{u}_{\mathbf{k}_\parallel,\parallel}(\mathbf{r}_\parallel, z_{\mathrm{ff}})\,\mathrm{d}^2 r_\parallel
# = \frac{1}{A_{\mathrm{cell}}} \iint_{\mathrm{cell}}
#   e^{-i \mathbf{k}_\parallel \cdot \mathbf{r}_\parallel}\,\mathbf{E}_{\mathbf{k}_\parallel,\parallel}(\mathbf{r}_\parallel, z_{\mathrm{ff}})\,\mathrm{d}^2 r_\parallel.
# $$
#
# When the structure possesses two-fold in-plane rotational symmetry $\mathcal{C}_2$
# ($\varepsilon(-\mathbf{r}_\parallel, z) = \varepsilon(\mathbf{r}_\parallel, z)$),
# eigenmodes at the $\Gamma$ point ($\mathbf{k}_\parallel = \mathbf{0}$) are simultaneous
# eigenstates of $\mathcal{C}_2$ with parity $\eta \in \{+1, -1\}$. For any mode whose
# transverse electric field is $\mathcal{C}_2$-odd ($\mathbf{u}_{\mathbf{0},\parallel}(-\mathbf{r}_\parallel, z) = -\mathbf{u}_{\mathbf{0},\parallel}(\mathbf{r}_\parallel, z)$,
# such as the anti-symmetric dimer mode of Tutorial 6.3 or the $A_1, A_2, B_1, B_2$ TM/TE
# cylinder modes in a $\mathcal{C}_{4v}$ square lattice), the unit-cell integral vanishes
# identically at $\mathbf{k}_\parallel = \mathbf{0}$:
#
# $$
# \mathbf{c}(\mathbf{0}) = \mathbf{0} \quad \implies \quad \gamma_{\mathrm{rad}}(\mathbf{0}) = 0, \qquad Q_{\mathrm{rad}}(\mathbf{0}) = \infty.
# $$
#
# **Perturbative $k_\parallel$ expansion (derived here):** For a small off-normal Bloch
# wavevector $|\mathbf{k}_\parallel| p \ll 1$ with symmetric geometry ($\alpha = 0$),
# expanding the Bloch phase factor $e^{-i \mathbf{k}_\parallel \cdot \mathbf{r}_\parallel} = 1 - i\,\mathbf{k}_\parallel \cdot \mathbf{r}_\parallel + \mathcal{O}(k_\parallel^2)$
# against the $\mathcal{C}_2$-odd unperturbed mode $\mathbf{u}_{\mathbf{0},\parallel}$
# gives a leading-order dipole-moment radiation amplitude linear in $\mathbf{k}_\parallel$:
#
# $$
# \mathbf{c}(\mathbf{k}_\parallel)
# = -\frac{i}{A_{\mathrm{cell}}} \iint_{\mathrm{cell}}
#   (\mathbf{k}_\parallel \cdot \mathbf{r}_\parallel)\,\mathbf{u}_{\mathbf{0},\parallel}(\mathbf{r}_\parallel, z_{\mathrm{ff}})\,\mathrm{d}^2 r_\parallel
#   + \mathcal{O}(k_\parallel^3)
# \propto \mathbf{k}_\parallel.
# $$
#
# In in-plane 1D variation along $x$ for the dimer pair at $x_{1,2} = \mp d/2$ with
# $+E_y$ in Bar 1 and $-E_y$ in Bar 2, $c_y(k_x) \propto e^{+i k_x d/2} - e^{-i k_x d/2} = 2 i \sin(k_x d / 2) \approx i k_x d$.
# Because the radiated power flux scales as $P_{\mathrm{rad}}(\mathbf{k}_\parallel) \propto |\mathbf{c}(\mathbf{k}_\parallel)|^2 \propto |\mathbf{k}_\parallel|^2$,
# the radiative quality factor diverges inversely with the square of the Bloch wavevector:
#
# $$
# Q_{\mathrm{rad}}(k_x) = Q_k \left(\frac{k_x p}{2\pi}\right)^{-2} \qquad (k_x \to 0).
# $$
#
# ---
#
# ### 1.2 Symmetry-Protected vs Quasi-BICs Under Geometric Bar-Width Tuning (Derived Here)
#
# In periodic resonator lattices [Hsu *et al.*, *Nat. Rev. Mater.* **1**, 16048 (2016);
# Kodigala *et al.*, *Nature* **541**, 196 (2017)], bound states in the continuum fall
# into two distinct classes:
#
# 1. **Symmetry-protected BICs at $\Gamma$ ($\alpha = 0$):** Because the unit-cell
#    far-field integral $\mathbf{c}(\mathbf{0}) = \mathbf{0}$ vanishes purely by
#    $\mathcal{C}_2$ parity mismatch against outgoing normal-incidence plane waves,
#    varying a continuous $\mathcal{C}_2$-preserving geometric parameter — such as the
#    mean semiconductor bar width $w_0 \in [140, 200]\;\mathrm{nm}$ in our dimer
#    metasurface (period $p = 680\;\mathrm{nm}$, center-to-center spacing $d = 255\;\mathrm{nm}$,
#    slab thickness $h = 190\;\mathrm{nm}$, refractive index $n = 3.48$) — shifts the
#    resonant wavelength $\lambda_{\mathrm{BIC}}(w_0)$ continuously without opening the
#    zeroth-order radiation channel ($Q_{\mathrm{sym}}(w_0) = \infty$ in the lossless
#    continuum limit, and $Q_{\mathrm{sym}}(w_0) \ge 8.55 \times 10^8$ in 2D FIT).
# 2. **Asymmetric quasi-BICs ($\alpha = (w_1 - w_2)/w_0 \ne 0$) and accidental BICs:**
#    When $\mathcal{C}_2$ symmetry is broken by a fixed relative bar-width asymmetry
#    $\alpha = 0.10$, the net unit-cell dipole moment $c_y(\mathbf{0}; w_0, \alpha) \propto \alpha\,\mathcal{I}(w_0)$
#    couples to outgoing plane waves with a radiative quality factor $Q_{\mathrm{qBIC}}(w_0, \alpha)$
#    and lasing threshold gain $g_{\mathrm{th}}(w_0, \alpha)$ that depend strongly on the
#    resonator width $w_0$ through the internal Fabry–Pérot / Mie overlap integral
#    $\mathcal{I}(w_0)$. (In 3D cylindrical lattices where the exact etched radius and
#    thickness are tuned simultaneously [Kodigala *et al.* 2017], destructive interference
#    between two modes of the same symmetry can also produce an accidental
#    Friedrich–Wintgen BIC at a single optimal radius; here we simulate our explicitly
#    specified 2D FIT dimer geometry across $w_0 \in [140, 200]\;\mathrm{nm}$ without
#    making unverified numeric comparisons to proprietary experimental dimensions.)
#
# ---
#
# ### 1.3 Topological-Charge Picture: Polarization Vortex Around $\Gamma$ (Derived Here)
#
# As established by Zhen *et al.* (*Phys. Rev. Lett.* **113**, 257401, 2014), the
# projected far-field polarization vector $\mathbf{c}(\mathbf{k}_\parallel) = (c_x(\mathbf{k}_\parallel), c_y(\mathbf{k}_\parallel))$
# defines a 2D vector field on the Brillouin zone whose major-axis orientation angle is
# $\psi(\mathbf{k}_\parallel) = \frac{1}{2}\arctan2(S_2, S_1)$ in terms of the Stokes
# parameters $S_1 = |c_x|^2 - |c_y|^2$ and $S_2 = 2\,\mathrm{Re}(c_x c_y^*)$.
# A BIC occurs precisely at a vortex core where $\mathbf{c}(\mathbf{k}_{\mathrm{BIC}}) = \mathbf{0}$
# ($S_1 = S_2 = 0$), so $\psi$ is undefined at $\mathbf{k}_{\mathrm{BIC}}$ and winds by
# an integer multiple of $2\pi$ along any closed counterclockwise loop $C$ enclosing
# the BIC:
#
# $$
# q = \frac{1}{2\pi} \oint_C \nabla_{\mathbf{k}_\parallel} \psi(\mathbf{k}_\parallel) \cdot \mathrm{d}\mathbf{k}_\parallel \in \mathbb{Z}.
# $$
#
# For a $\mathcal{C}_{4v}$-symmetric dielectric cylinder unit cell near $\Gamma$
# ($\mathbf{k}_\parallel = k_\parallel(\cos\phi_k, \sin\phi_k)$) (**derived here**):
# - **$A_1$-like TE mode ($C_4 = +1$):** In-plane far-field polarization winds in the
#   same sense as the wavevector azimuth $\phi_k$, $\mathrm{d}\psi / \mathrm{d}\phi_k > 0$,
#   giving topological charge **$q = +1$**.
# - **Analytical $C_4 = -1$ reference ($B_1$-like symmetry):** In the analytical
#   $\mathcal{C}_{4v}$ vortex expansion (`analytic_c4v_vortex_field`), a mode with
#   $C_4 = -1$ winds in the opposite sense to $\phi_k$, $\mathrm{d}\psi / \mathrm{d}\phi_k < 0$,
#   giving reference charge $q = -1$.
#
# To verify the 3D symmetry-protected BIC and its topological vortex in 3D FIT without
# symmetry bias while respecting the $\ge 20\;\mathrm{cells}/\lambda_{\mathrm{mat}}$ rule
# at the actual mode frequency, we configure `make_3d_cylinder_bloch_solver` at
# $n_x = n_y = 30$, $n_z = 44$ ($n_{\mathrm{PML}} = 10$, $\Delta x = \Delta y = 22.67\;\mathrm{nm}$,
# $\Delta z_{\mathrm{cyl}} = 22.00\;\mathrm{nm}$) and excite the unit cell using a single
# off-mirror dipole at $(x_s, y_s) = (135, 85)\;\mathrm{nm}$ ($x_s \ne y_s \ne 0$).
# `harminv` identifies the resolved $A_1$-like TE BIC at $f_{A_1} = 196.3574\;\mathrm{THz}$
# ($\lambda_{A_1} = 1526.77\;\mathrm{nm}$), giving
# $\mathrm{cpw}_{A_1} = \lambda_{A_1} / (n_{\mathrm{cyl}}\,\max(\Delta x, \Delta y, \Delta z_{\mathrm{cyl}})) = 21.38 \ge 20$
# inside $n_{\mathrm{cyl}} = 3.15$. (By contrast, a higher-frequency TM resonance near
# $270.49\;\mathrm{THz}$ has $\lambda \approx 1108.3\;\mathrm{nm}$, which yields only
# $14.5\text{--}15.5\;\mathrm{cells}/\lambda < 20$ at $n_x \le 30$ and leaks with low
# $Q_\Gamma \approx 284 \approx Q_{\mathrm{ring}}$; we therefore restrict our 3D FIT
# BIC and topological-charge claims strictly to the resolved $A_1$ mode.) Comparing two
# CPML configurations at $\Gamma$ (`n_pml = 6, nz = 36` vs `n_pml = 10, nz = 44`), where the
# ringdown record covers $\approx 22$ optical cycles after the pulse and `harminv` selects the
# highest-$Q$ candidate ($\arg\max Q$), shows that $Q_\Gamma(A_1)$ increases from $24345.8$ to
# $Q_\Gamma > 10^7$ (lower bound from a 22-cycle record; `harminv` value $19245231.8$, $790.5\times$),
# consistent with residual $\Gamma$-point loss being dominated by near-field CPML absorption.
# Stepping off $\Gamma$ along $\phi_k = 17^\circ$ at $k_\parallel a / (2\pi) \in \{0.03, 0.05, 0.08\}$
# is consistent with $Q_{A_1}(k_\parallel) \propto k_\parallel^{-2}$ ($Q = [69899.56,\, 25480.82,\, 10266.02]$,
# log–log slope $-1.9560$; the off-$\Gamma$ ring $Q$ values were not checked for PML sensitivity,
# with $Q_\Gamma$ at $n_{\mathrm{PML}} = 6 \approx 24346 \approx \text{ring } Q \approx 25\mathrm{k}$),
# and sampling the far-field Fourier amplitudes on the 8-point
# off-mirror loop $\phi_k = 17^\circ + 45^\circ m$ via `extract_bloch_zeroth_order_ring`
# (which delegates directly to `compute_topological_charge`) yields $q_{A_1} = +1.0000$.
#
# ---
#
# ### 1.4 Finite $N$-Period Strip Arrays: Scaling $Q_1(N) \propto N^p$ vs Fixed-$N$ Mode Order $Q_m(N) \approx Q_1(N)/m^2$ (Derived Here)
#
# Truncating the 2D periodic lattice along $x$ to a finite strip array of $N$ unit cells
# of total length $L = N p$ (periodic along $y$, open CPML boundaries on $\pm x$ and
# $\pm z$) quantizes the slowly varying Bloch envelope into standing-wave box modes:
#
# $$
# \psi_m(x_n) = \sqrt{\frac{2}{N}}\sin\!\left(\frac{m \pi (n + 1/2)}{N}\right), \qquad k_m = \frac{m \pi}{N p}, \qquad m = 1, 2, 3, \dots
# $$
#
# In Fourier space, the envelope spectrum $|\tilde{\psi}_m(k_x)|^2$ is concentrated at
# $k_x = \pm k_m = \pm m \pi / (N p)$. Weighting the infinite-lattice radiative decay
# rate $\gamma_{\mathrm{rad}}(k_x) = \frac{\omega_0}{2 Q_k}\left(\frac{k_x p}{2\pi}\right)^2$
# by the envelope momentum distribution gives the vertical momentum-spread baseline
# (**derived here**):
#
# $$
# Q_{\mathrm{vert}, m}(N) \approx Q_{\mathrm{rad}}(k_m) = \frac{4}{\pi^2}\,Q_k \left(\frac{N}{m}\right)^2.
# $$
#
# In a finite strip array with open lateral boundaries, two distinct scalings hold:
# 1. **Across strip sizes $N$ at fixed fundamental order $m = 1$:** In an open finite
#    strip terminated by air and CPML boundaries at $x = \pm L/2$, total modal loss sums
#    out-of-plane radiative leakage ($\gamma_{\mathrm{vert}} \propto k_1^2 \propto N^{-2}$)
#    and in-plane guided-slab escape at the strip ends ($\gamma_{\mathrm{lat}} \propto v_g (1 - R_{\mathrm{edge}}) / (N p)$).
#    Because the band-edge group velocity $v_g(k_1) = |\partial \omega / \partial k_x|_{k_1} \propto k_1 \propto N^{-1}$
#    vanishes linearly toward the quadratic band extremum at $\Gamma$ while the
#    impedance-mismatched end reflectivity $R_{\mathrm{edge}}(k_1)$ and evanescent mode
#    tail vary across small array sizes $N \in [4, 12]$, the combined quality factor
#    scales with an effective pre-asymptotic power law $Q_1(N) \propto N^p$ with measured
#    $p = 2.5518$ across $N \in \{4, 6, 8, 10, 12\}$ (intermediate between the pure
#    vertical momentum-spread baseline $p_{\mathrm{baseline}} = 2.0$ and the $N^3$
#    band-edge Fabry–Pérot cavity regime).
# 2. **Across envelope mode orders $m = 1, 2, 3$ at fixed strip size $N$:** Because the
#    envelope wavenumber $k_m = m \pi / (N p)$ is proportional to $m$ and the near-$\Gamma$
#    radiative loss rate scales as $k_m^2 \propto m^2$, the quality factors of higher-order
#    envelope modes at a *fixed* $N$ follow (**derived here**):
#
# $$
# Q_m(N) \approx \frac{Q_1(N)}{m^2} \qquad (\text{at fixed } N), \quad\text{so}\quad Q_1(N) \approx 4\,Q_2(N) \approx 9\,Q_3(N).
# $$
#
# Thus **the fundamental $m = 1$ envelope mode (whose dominant wavevector $k_1 = \pi / (N p)$
# lies closest to the BIC at $\Gamma$) always possesses the highest quality factor and lowest lasing threshold**.
#
# ---
#
# ### 1.5 Active Gain, Lasing Threshold, Doughnut Far-Field Beam, and Symmetry Breaking (Derived Here)
#
# #### Cold-cavity threshold gain $g_{\mathrm{th}}$
#
# When the resonator bars of refractive index $n_{\mathrm{cyl}} = \sqrt{\varepsilon_{\mathrm{cyl}}}$
# and volume fraction $f(\mathbf{r})$ are doped with an active gain medium (`SaturableGain`
# or `FourLevelGain` in `wakis.gain`) of line-center power gain coefficient $g_0\;[\mathrm{m}^{-1}]$,
# the resonant active susceptibility is $\chi_{\mathrm{gain}}(\omega_0) = +i\,\frac{c\,n_{\mathrm{cyl}}\,g_0}{\omega_0}$
# (in Wakis's $e^{+i\omega t}$ convention). By first-order Maxwell perturbation theory
# on the composite permittivity $\varepsilon(\mathbf{r}) = \varepsilon_{\mathrm{bg}} + f(\mathbf{r})(\varepsilon_{\mathrm{cyl}} - \varepsilon_{\mathrm{bg}} + \chi_{\mathrm{gain}})$,
# the modal amplitude growth rate is (**derived here**):
#
# $$
# \gamma_{\mathrm{gain}}(g_0) = \frac{c}{2 n_{\mathrm{cyl}}}\,\Gamma_{\mathrm{conf}}\,g_0,
# \qquad
# \Gamma_{\mathrm{conf}} = \frac{\iint_{\mathrm{cell}} f(x, z)\,\varepsilon_{\mathrm{cyl}}\,|E_y(x, z)|^2\,\mathrm{d}x\,\mathrm{d}z}{\iint_{\mathrm{cell}} \varepsilon_r(x, z)\,|E_y(x, z)|^2\,\mathrm{d}x\,\mathrm{d}z}.
# $$
#
# Lasing threshold occurs when $\gamma_{\mathrm{eff}}(g_0) = \gamma_{\mathrm{rad}} - \gamma_{\mathrm{gain}}(g_0) = 0$
# with $\gamma_{\mathrm{rad}} = \omega_0 / (2 Q)$, yielding the exact cold-cavity threshold
# identity (**derived here**):
#
# $$
# g_{\mathrm{th}} = \frac{2\pi n_{\mathrm{cyl}}}{\lambda_0\,\Gamma_{\mathrm{conf}}\,Q}.
# $$
#
# #### Spatial-hole-burning L–I curve and noise-seeded `FourLevelGain` clamping
#
# Above threshold ($g_0 > g_{\mathrm{th}}$) in a saturable gain medium
# $g(\mathbf{r}) = \frac{g_0}{1 + I(\mathbf{r}) / I_{\mathrm{sat}}}$, the standing-wave
# intensity profile $I(\mathbf{r}) = \langle I \rangle\,u(\mathbf{r})$ (normalized so
# $\langle u(\mathbf{r}) \rangle_{\mathrm{bars}} = 1$) saturates preferentially at the
# antinodes inside each resonator bar. Steady-state oscillation requires the
# mode-weighted saturated gain to clamp at $g_{\mathrm{th}}$ (**derived here**):
#
# $$
# \iint_{\mathrm{bars}} \frac{w_{\mathrm{modal}}(\mathbf{r})}{1 + (\langle I \rangle_{\mathrm{steady}} / I_{\mathrm{sat}})\,u(\mathbf{r})}\,\mathrm{d}x\,\mathrm{d}z
# = \frac{g_{\mathrm{th}}}{g_0},
# \qquad
# w_{\mathrm{modal}}(\mathbf{r}) = \frac{f(\mathbf{r})\,\varepsilon_{\mathrm{cyl}}\,|E_y(\mathbf{r})|^2}{\iint_{\mathrm{bars}} f\,\varepsilon_{\mathrm{cyl}}\,|E_y|^2\,\mathrm{d}A}.
# $$
#
# In the microscopic four-level atomic rate-equation model (`FourLevelGain` in `wakis.gain`
# [Chang & Taflove, *Opt. Express* **12**, 3827 (2004); Andreasen & Cao, *J. Lightwave Technol.* **27**, 4530 (2009)]),
# lasing builds up self-consistently from zero initial field via fluctuation–dissipation
# Langevin spontaneous-emission noise (`spontaneous_emission=True, seed=42`) using a
# documented rate-equation time-scale factor `time_scale = 80.0` ($\le 100$, accelerating
# the picosecond atomic population lifetimes $\tau_{21}, \tau_{32}, \tau_{10}$ and pump
# rate $W_p$ by $s = 80\times$ while keeping $\Delta N_{\mathrm{th}}$ invariant).
#
# #### Doughnut far-field radiation profile via `Near2FarMonitor`
#
# Projecting the tangential steady-state lasing fields $(E_y(x, z_{\mathrm{ff}}), H_x(x, z_{\mathrm{ff}}))$
# recorded on Wakis's `Near2FarMonitor` aperture at $z_{\mathrm{ff}} = +650\;\mathrm{nm}$
# across the finite $N = 8$ strip ($L = N p = 5.44\;\mu\mathrm{m}$), the continuum
# far-field radiation amplitude of the $\mathcal{C}_2$-odd fundamental mode $m = 1$ is
# the product of the intra-cell antisymmetric dipole factor $2i\sin(k_x d / 2)$ and the
# fundamental cosine box-envelope Fourier transform (**derived here**):
#
# $$
# A_1(\theta) \propto \sin\!\left(\frac{k_0 d \sin\theta}{2}\right)
# \frac{\cos\!\left(\tfrac{1}{2} k_0 L \sin\theta\right)}{(\pi/L)^2 - (k_0 \sin\theta)^2}\,\cos\theta.
# $$
#
# With $u = k_x L = k_0 L \sin\theta$, the bare cosine-envelope Fourier transform
# $\hat{A}_{\mathrm{env}}(u) \propto \cos(u/2)/(\pi^2 - u^2)$ is **even** in $u$ and
# attains its global maximum at normal incidence $u = 0$
# ($|\hat{A}_{\mathrm{env}}(0)| \propto 1/\pi^2 \approx 0.1013$ vs
# $|\hat{A}_{\mathrm{env}}(\pm\pi)| \propto 1/(4\pi) \approx 0.0796$ at the standing-wave
# carrier wavevector $k_x = \pm\pi/L$, $\theta_{\mathrm{env}} = \arcsin(\lambda_0 / (2 N p)) = 8.36^\circ$).
# Therefore, **the normal-incidence radiation null $I(\theta = 0^\circ) = 0$ arises
# solely from the intra-cell antisymmetric dipole interference factor
# $\sin(k_x d / 2) = \sin(\tfrac{1}{2} k_0 d \sin\theta)$**. Multiplying the bare
# envelope by $\sin(k_x d / 2) \approx (d / 2L)\,u$ suppresses the $u = 0$ maximum to
# zero and shifts the twin radiation lobe peaks outward beyond $u = \pi$ to the root of
# $\frac{\mathrm{d}}{\mathrm{d}u}\!\left[\frac{u \cos(u/2)}{\pi^2 - u^2}\right] = 0$
# at $u \approx 1.367\,\pi$ for the pure envelope $u\cos(u/2)/(\pi^2 - u^2)$ and
# $u \approx 1.34\,\pi$ (full expression) when including $\sin(k_x d/2)$ and $\cos\theta$
# (**derived here**), yielding a continuum analytical lobe peak
# at **$\theta_{\mathrm{peak,theory}} = \pm 11.25^\circ$** on a fine $0.05^\circ$ angular
# grid (compared with $\theta_{\mathrm{peak,FIT}} = \pm 10.05^\circ$ from `Near2FarMonitor`,
# a $10.67\%$ difference; attributing this gap to an evanescent-tail broader envelope of
# the open-strip supermode relative to the hard-wall $\cos(\pi x/L)$ continuum box ansatz
# is an uncomputed hypothesis).
#
# #### Symmetry-breaking mutation ($\alpha \ne 0$) in a finite $N$-period strip
#
# Reusing Tutorial 6.3's intra-cell width asymmetry $\alpha = (w_1 - w_2)/w_0$,
# breaking $\mathcal{C}_2$ symmetry opens a direct normal-incidence radiation channel
# with $Q_{\mathrm{rad},\infty}(\alpha) = Q_0\,\alpha^{-2}$ ($Q_0 = 47.11$). Directly
# simulating the finite $N = 8$ strip array across $\alpha \in [0, 0.20]$ in 2D FIT,
# the total quality factor combines lateral edge leakage $Q_{\mathrm{edge}}(0, N)$ and
# bulk asymmetry radiation as $Q_{\mathrm{tot}}^{-1}(\alpha, N) \approx Q_{\mathrm{edge}}^{-1}(0, N) + Q_0^{-1}\alpha^2$,
# which raises the threshold gain quadratically (**derived here**):
#
# $$
# g_{\mathrm{th}}(\alpha, N) = g_{\mathrm{th}}(0, N) + \kappa_{\mathrm{th}}\,\alpha^2,
# \qquad
# \kappa_{\mathrm{th}} = \frac{2\pi n_{\mathrm{cyl}}}{\lambda_0\,\Gamma_{\mathrm{conf}}\,Q_0}.
# $$

# %% [markdown]
# <a id="2-setup"></a>
# ## 2. Simulation Setup, Graded Mesh, and 3D Preflight (`wakis.estimate`)
#
# The setup figures (`figures/00a_setup_geometry.png`, `figures/00b_setup_mesh_and_excitation.png`,
# `figures/00a_setup_geometry_finite.png`, and `figures/00b_setup_mesh_and_excitation_finite.png`),
# generated directly from the live Wakis `SolverFIT3D` and `GridFIT3D` objects via
# `wakis.schematic.save_setup_figures`, illustrate both the infinite-lattice active Bloch
# unit cell and the finite $N = 8$ active BIC laser strip array before any time-stepping run:
#
# - **Primary Setup (Active Bloch Unit Cell):** Two semiconductor resonator bars
#   ($p = 680\;\mathrm{nm}$, $d = 255\;\mathrm{nm}$, $w_0 = 170\;\mathrm{nm}$,
#   $h = 190\;\mathrm{nm}$, $n = 3.48$, inner air gap $g_{\mathrm{in}} = 85\;\mathrm{nm}$)
#   populated with subpixel `SaturableGain` or `FourLevelGain` (`wakis.gain`), excited
#   by a weak antisymmetric $\pm J_y$ dipole seed pair (or by `FourLevelGain(spontaneous_emission=True)`
#   Langevin noise) and monitored by an intracavity `ResonanceMonitor` (`harminv`) and a
#   far-field `DFTMonitor` at $z = +580\;\mathrm{nm}$. At baseline resolution $n_x = 32$
#   ($\Delta x = 21.25\;\mathrm{nm}$, graded $\Delta z_{\min} = 15.0\;\mathrm{nm}$),
#   the mesh provides $21.28 \ge 20\;\mathrm{cells}/\lambda_{\mathrm{mat}}$ horizontally,
#   $30.15 \ge 20\;\mathrm{cells}/\lambda_{\mathrm{mat}}$ vertically, and $4.00 \ge 4$
#   cells across the $85\;\mathrm{nm}$ gap.
# - **Finite-Strip Setup ($N = 8$ Active BIC Laser Strip Array):** Sixteen active resonator
#   bars across $N = 8$ unit cells ($L = 8 p = 5.44\;\mu\mathrm{m}$, periodic along $y$)
#   surrounded by $0.65\;\mu\mathrm{m}$ lateral air buffers and $10$-cell CPML boundaries
#   on $\pm x$ and $\pm z$, seeded with a multi-mode envelope ($m = 1, 2, 3$) and
#   monitored across the $z = +650\;\mathrm{nm}$ `Near2FarMonitor` radiation aperture to
#   project the far-field doughnut beam cross-section $I(\theta)$.
#
# ![Setup Geometry (Active Bloch BIC Unit Cell)](figures/00a_setup_geometry.png)
# ![Setup Mesh and Excitation (Active Bloch BIC Unit Cell)](figures/00b_setup_mesh_and_excitation.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 680 nm × 21.2 nm × 2.60 µm (z ∈ [-1.30 µm, 1.30 µm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 32 × 1 × 74 (2368 cells) |
# | Discretization | Δz & resolution | Δ_min = 15 nm, 30.2 cells/λ (in n_max = 3.48), 5.7 cells across feature (85 nm) |
# | Boundary | -x face | Periodic |
# | Boundary | +x face | Periodic |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | CPML (10 cells, 448 nm) |
# | Boundary | +z face | CPML (10 cells, 448 nm) |
# | Material | Background | Air / vacuum cladding (ε_bg = 1.00, n = 1.00) (isotropic) |
# | Material | Region 1: Active bar 1 (w₁ = 170 nm, SaturableGain / FourLevelGain) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
# | Material | Region 2: Active bar 2 (w₂ = 170 nm, SaturableGain / FourLevelGain) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
# | Source | 1 · CurrentSource | pos: (x, z) = (−127.5, −25) nm (inside active Bar 1), pol: +Jy (TE, C₂-odd BIC seed), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
# | Source | 2 · CurrentSource | pos: (x, z) = (+127.5, −25) nm (inside active Bar 2), pol: −Jy (TE, C₂-odd BIC seed), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
# | Monitor | 3 · ResonanceMonitor | ResonanceMonitor (harminv Ey & ΔN(t) probe) at (x₁, z) = (−127.5, +25) nm, pos: (x, z) = (−127.5, +25) nm, freqs: 0.90 f₀ – 1.10 f₀ (Q(k_x), γ_eff(g₀), and L–I steady state) |
# | Monitor | 4 · DFTMonitor | Radiated far-field monitor plane at z = +580 nm, pos: z = +580 nm (x ∈ [−340, +340] nm), freqs: f₀ = 190.5 THz (λ_BIC = 1573.9 nm) |
# | Variation | Part 1 — Infinite lattice Bloch dispersion & vortex (Task 0) | k_x p / (2π) ∈ [0, 0.10] (Q ∝ k_x⁻²), bar width w₀ ∈ [140, 200] nm BIC sweep, 3D Bloch A₁ BIC vortex (q = +1) |
# | Variation | Part 2 — Finite N-period strip array Q scaling & mode selection (Task 1) | N ∈ {4, 6, 8, 10, 12} strip with open CPML BCs (Q₁ ∝ N^p; mode m = 1 vs m = 2, 3) — see 00a/00b_*_finite.png |
# | Variation | Part 3 — Active BIC lasing & doughnut far field (Task 2) | SaturableGain & FourLevelGain (spont. emission, s=80, L–I curve, ΔN clamping, Near2Far doughnut I(θ=0°)≈0) |
# | Variation | Part 4 — Symmetry-breaking mutation & convergence (Tasks 3–4) | Finite N=8 strip across α = Δw / w₀ ∈ [0, 0.20] lowers Q and raises g_th(α) = g_th(0) + κ_th α²; mesh n_x ∈ {32, 48, 64} |
#
# ![Setup Geometry (Finite N = 8 Active BIC Laser Strip Array)](figures/00a_setup_geometry_finite.png)
# ![Setup Mesh and Excitation (Finite N = 8 Active BIC Laser Strip Array)](figures/00b_setup_mesh_and_excitation_finite.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 6.76 µm × 21.2 nm × 2.30 µm (z ∈ [-1.15 µm, 1.15 µm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 318 × 1 × 68 (21624 cells) |
# | Discretization | Δx & resolution | Δ_min = 15 nm, 30.2 cells/λ (in n_max = 3.48), 5.7 cells across feature (85 nm) |
# | Boundary | -x face | CPML (10 cells, 212.5 nm) |
# | Boundary | +x face | CPML (10 cells, 212.5 nm) |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | CPML (10 cells, 441 nm) |
# | Boundary | +z face | CPML (10 cells, 441 nm) |
# | Material | Background | Air cladding (ε_bg = 1.00, open CPML on ±x and ±z) (isotropic) |
# | Material | Region 1: Active resonator array (N = 8) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, x ∈ [-2.59 µm, 2.59 µm] (d = 5.18 µm) |
# | Source | 1 · CurrentSource | pos: z = 0 nm, x ∈ [−2.55, +2.55] µm (16 active bars), pol: ±Jy (intra-cell odd, envelope m = 1, 2, 3), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
# | Monitor | 2 · Near2FarMonitor | Near2FarMonitor radiation aperture at z = +650 nm (projects doughnut beam I(θ)), pos: z = +650 nm, x ∈ [−3.10, +3.10] µm, freqs: λ₁(N=8) = 1582.5 nm (fundamental BIC lasing line) |
# | Variation | Finite N-period strip size sweep (Task 1) | N ∈ {4, 6, 8, 10, 12} unit cells (Q₁(N) ∝ N^2.55; fundamental m = 1 has highest Q) |
# | Variation | Envelope mode competition (Task 2) | Modes m = 1, 2, 3 at N = 8 (Q₁ = 528.3, Q₂ = 138.1, Q₃ = 65.3; single-mode lasing at m = 1) |

# %% [markdown]
# <a id="3-infinite-lattice"></a>
# ## 3. Results — Part 1: Infinite Lattice ($Q \propto k^{-2}$, Bar-Width BIC Sweep, and 3D Polarization Vortex)
#
# Figure `01_infinite_bic_q_and_vortex.png` summarizes the infinite-lattice Bloch
# unit-cell properties from Task 0:
#
# - **Momentum-space scaling $Q(k_x) \propto k_x^{-2}$ near $\Gamma$ (Panel a):**
#   At $\Gamma$ ($k_x = 0$), the symmetric ($\alpha = 0$) dimer mode has
#   $\lambda_{\mathrm{BIC}} = 1573.8847\;\mathrm{nm}$ and $Q(\Gamma) = 1.573 \times 10^9$
#   ($1572905115.2$). Stepping off normal incidence along
#   $k_x p / (2\pi) \in \{0.02, 0.03, 0.04, 0.06, 0.08, 0.10\}$ in 2D Bloch FIT (`harminv`),
#   $Q(k_x)$ drops from $15320.13$ at $k_x p / (2\pi) = 0.02$ to $655.13$ at
#   $k_x p / (2\pi) = 0.10$, following a near-$\Gamma$ log–log power-law slope of
#   **$-1.9879$** ($0.61\%$ relative error against the exact perturbative exponent
#   $-2.0000$, with asymptotic constant $Q_k = Q(k_x)(k_x p / 2\pi)^2 = 6.1529$).
# - **2D FIT symmetry-protected ($\alpha = 0$) vs quasi-BIC ($\alpha = 0.10$) across bar width $w_0$ (Panel b):**
#   Sweeping the mean semiconductor bar width across $w_0 \in \{140.0, 155.0, 170.0, 185.0, 200.0\}\;\mathrm{nm}$
#   in 2D FIT (`run_bar_width_bic_sweep`), the resonant wavelength tunes from
#   $\lambda_{\mathrm{sym}} = 1503.41\;\mathrm{nm}$ to $1629.93\;\mathrm{nm}$ while the
#   symmetric ($\alpha = 0$) mode remains dark at $\Gamma$ for every bar width
#   ($Q_{\mathrm{sym}}(w_0) \ge 8.55 \times 10^8$, minimum $855060054.8$). Under fixed
#   relative asymmetry $\alpha = 0.10$ on the baseline $n_x = 32$ mesh ($\Delta x = 21.25\;\mathrm{nm}$),
#   $Q_{\mathrm{qBIC}}(w_0)$ rises from $2313.20$ at $w_0 = 140\;\mathrm{nm}$
#   ($g_{\mathrm{th}} = 67.25\;\mathrm{cm}^{-1}$) to $4710.72$ at $170\;\mathrm{nm}$
#   ($g_{\mathrm{th}} = 31.16\;\mathrm{cm}^{-1}$) and $4978.66$ at $185\;\mathrm{nm}$
#   ($g_{\mathrm{th}} = 28.82\;\mathrm{cm}^{-1}$), before jumping $2.55\times$ to $12672.61$
#   at $w_0 = 200\;\mathrm{nm}$ ($g_{\mathrm{th}} = 11.09\;\mathrm{cm}^{-1}$).
#   $Q_{\mathrm{qBIC}}$ at $n_x = 32$ is not converged ($12672.61$ vs $8504.60$ at
#   $n_x = 48$ for $w_0 = 200\;\mathrm{nm}$, $49\%$); the $n_x = 48$ sweep is monotonic
#   ($Q_{\mathrm{qBIC}}(w_0; n_x = 48) = [3093.51,\, 3592.13,\, 4690.80,\, 6259.99,\, 8504.60]$,
#   $g_{\mathrm{th}} = [50.31,\, 41.95,\, 31.28,\, 22.93,\, 16.54]\;\mathrm{cm}^{-1}$);
#   the cause of the $n_x = 32$ jump is unexplained.
# - **3D FIT $A_1$ BIC at $\Gamma$, $Q(k_\parallel) \propto k_\parallel^{-2}$ scaling, and polarization vortex around $\Gamma$ (Panels a, c–d):**
#   Using our 3D Bloch-periodic cylinder unit cell (`make_3d_cylinder_bloch_solver`,
#   $n_x = n_y = 30$, $n_z = 44$, $n_{\mathrm{PML}} = 10$), we excite the unit cell with
#   a single off-mirror dipole at $(x_s, y_s) = (135, 85)\;\mathrm{nm}$ and resolve the
#   $A_1$-like TE BIC at $f_{A_1} = 196.3574\;\mathrm{THz}$ ($\lambda_{A_1} = 1526.77\;\mathrm{nm}$),
#   where the spatial resolution evaluated at the actual mode frequency in $n_{\mathrm{cyl}} = 3.15$
#   is **$\mathrm{cpw}_{A_1} = 21.38 \ge 20$** (whereas the higher-frequency TM resonance
#   at $270.49\;\mathrm{THz}$ has only $14.5\text{--}15.5\;\mathrm{cells}/\lambda < 20$
#   and $Q_\Gamma \approx 284$, and is therefore excluded). Because the 3D ringdown record
#   covers only $\approx 22$ optical cycles after the pulse and `harminv` picks the
#   highest-$Q$ candidate ($\arg\max Q$), the $\Gamma$-point $Q$ is a lower-bound-type
#   estimate: comparing two CPML thicknesses at $\Gamma$ shows that the 3D quality factor
#   increases from **$Q_\Gamma(n_{\mathrm{PML}}=6) = 24345.8$** ($N_z = 36$) to
#   **$Q_\Gamma > 10^7$ (lower bound from a 22-cycle record; `harminv` value $19245231.8$
#   at $n_{\mathrm{PML}}=10$, $N_z = 44$, a $790.5\times$ increase)**, consistent with
#   residual $\Gamma$-point loss being near-field evanescent CPML absorption. Stepping off
#   $\Gamma$ along $\phi_k = 17^\circ$ across $k_\parallel a / (2\pi) \in \{0.03, 0.05, 0.08\}$,
#   the 3D quality factor drops as $Q_{A_1}(k_\parallel) = [69899.56,\, 25480.82,\, 10266.02]$
#   with a fitted log–log slope of **$-1.9560$** ($2.20\%$ relative error against $-2.0000$,
#   Panel a). Around the 8-point off-mirror azimuth loop at $k_{\mathrm{ring}} = 0.05\,(2\pi/a)$
#   ($\phi_k \in \{17^\circ, 62^\circ, 107^\circ, 152^\circ, 197^\circ, 242^\circ, 287^\circ, 332^\circ\}$,
#   $Q_{\mathrm{ring}}(A_1) \in [24614.6,\, 25483.7]$; the off-$\Gamma$ ring $Q$ values were
#   not checked for PML sensitivity, with $Q_\Gamma$ at $n_{\mathrm{PML}} = 6 \approx 24346 \approx \text{ring } Q \approx 25\mathrm{k}$),
#   the $\Gamma$-point zeroth-order radiation intensity is suppressed by
#   $|\mathbf{c}_{\mathrm{3D}}(\mathbf{0})|^2 / \langle|\mathbf{c}_{\mathrm{3D}}|^2\rangle_{\mathrm{ring}} = 1.8 \times 10^{-7}$,
#   and routing the extracted Bloch ring amplitudes through `extract_bloch_zeroth_order_ring`
#   and `compute_topological_charge` yields a quantized topological charge
#   **$q_{A_1} = +1.0000$** [Zhen *et al.*, *Phys. Rev. Lett.* **113**, 257401 (2014)].
#   Panel (c) displays the analytic map with an arbitrary Gaussian envelope
#   $\exp(-(k_\parallel/0.055)^2)$ (not FIT data) overlaid with the 8
#   off-mirror 3D FIT sampling points, while Panel (d) plots the unwrapped 3D FIT
#   polarization angle $\psi(\phi_k)$ alongside the $+1$ theory slope and $-1$ analytical reference.
#
# ![Infinite Lattice BIC Q(k), Bar-Width BIC Sweep, and 3D Polarization Vortex](figures/01_infinite_bic_q_and_vortex.png)

# %% [markdown]
# <a id="4-finite-arrays"></a>
# ## 4. Results — Part 2: Finite $N$-Period Strip Arrays ($Q_1 \propto N^p$ Scaling and Mode Selection)
#
# Figure `02_finite_array_q_vs_n.png` shows the 2D FIT finite-strip results from Task 1:
#
# - **Finite $N$-period strip quality factor $Q_1(N) \propto N^p$ (Panel a):** Across
#   finite strip arrays of $N \in \{4, 6, 8, 10, 12\}$ symmetric ($\alpha = 0$) unit
#   cells (periodic along $y$ and terminated by open CPML boundaries on $\pm x$ and
#   $\pm z$), the fundamental $m = 1$ BIC resonance shifts monotonically toward the
#   infinite-lattice wavelength $\lambda_{\mathrm{BIC}} = 1573.8847\;\mathrm{nm}$
#   ($\lambda_1 = 1604.753, 1588.630, 1582.472, 1579.497, 1577.841\;\mathrm{nm}$) while its
#   quality factor grows from $Q_1(4) = 90.08$ to $Q_1(12) = 1482.50$ with a fitted
#   power-law exponent **$p = 2.5518$**, compared directly against the $N^2$ vertical
#   momentum-spread baseline ($p_{\mathrm{baseline}} = 2.0$, $\Delta k = \pi/(N p)$).
#   The steeper pre-asymptotic exponent $p = 2.5518 > 2.0$ arises because total loss in
#   the open strip combines vertical radiative leakage ($\propto k_1^2 \propto N^{-2}$)
#   with lateral guided-slab escape at $x = \pm L/2$, where the vanishing band-edge group
#   velocity $v_g(k_1) \propto k_1 \propto N^{-1}$ and size-dependent end-facet
#   reflectivity steepen the scaling between $N^2$ and $N^3$ across $N \in [4, 12]$.
# - **Mode selection across standing-wave envelope orders $m = 1, 2, 3$ at fixed $N = 8$ (Panel b):**
#   At fixed strip size $N = 8$, exciting standing-wave envelopes
#   $\psi_m(x_n) = \sin(m \pi (n+1/2)/N)$ for $m \in \{1, 2, 3\}$ yields
#   $Q_1(8) = 528.34$, $Q_2(8) = 138.07$, and $Q_3(8) = 65.28$ (log–log slope $-1.9068$
#   vs $m$, matching the fixed-$N$ law $Q_m(N) \approx Q_1(N)/m^2$ with ratios
#   $Q_1/Q_2 = 3.827$, $4.33\%$ from $4.000$, and $Q_1/Q_3 = 8.093$). Because
#   $k_1 = \pi/(N p)$ lies closest to the radiation null at $\Gamma$, **mode $m = 1$
#   always has the highest $Q$**.
# - **2D $(x, z)$ fundamental BIC eigenmode map (Panel c):** The $m = 1$ eigenmode
#   displays an intra-cell odd ($\pm E_y$) oscillation modulated by a smooth single-lobe
#   $\sin(\pi(x+L/2)/L)$ envelope, producing a global nodal plane $E_y(x=0, z) = 0$
#   along the surface normal.
#
# ![Finite Strip Array Q(N) Scaling, Envelope Mode Selection, and Eigenmode Map](figures/02_finite_array_q_vs_n.png)

# %% [markdown]
# <a id="5-threshold-and-lasing"></a>
# ## 5. Results — Part 3: Lasing Threshold, Single-Mode Lasing (`wakis.gain`), and `Near2FarMonitor` Doughnut Beam
#
# Figure `03_lasing_threshold_and_doughnut.png` and the animated field movie
# `03b_bic_lasing_buildup.gif` present the active lasing simulations from Task 2:
#
# - **Linear threshold gain $g_{\mathrm{th}}$ vs $w_0$, vs $N$, and 2D FIT zero-crossing (Panel a):**
#   From the cold-cavity eigenmode at $\alpha = 0.08$ ($Q = 7356.63$, $\lambda_0 = 1573.69\;\mathrm{nm}$),
#   our perturbation integral gives $\Gamma_{\mathrm{conf}} = 0.9472$ and an analytical
#   threshold gain $g_{\mathrm{th,analytic}} = 19.940\;\mathrm{cm}^{-1}$ ($1993.98\;\mathrm{m}^{-1}$).
#   Sweeping linear gain (`SaturableGain` with $I_{\mathrm{sat}} = \infty$) in 2D FIT
#   yields a strictly linear reduction of the modal decay rate $\gamma_{\mathrm{eff}}(g_0)$
#   that crosses zero at **$g_{\mathrm{th,FIT}} = 19.929\;\mathrm{cm}^{-1}$** ($1992.87\;\mathrm{m}^{-1}$,
#   **$0.056\%$ relative error**, $0.06\%$ rounded). Across finite strip sizes
#   $N \in \{4, 6, 8, 10, 12\}$, the fundamental BIC threshold drops by $16.2\times$
#   from $1596.96\;\mathrm{cm}^{-1}$ at $N = 4$ to $98.69\;\mathrm{cm}^{-1}$ at $N = 12$
#   (inset), while across bar widths $w_0 \in [140, 200]\;\mathrm{nm}$ at $\alpha = 0.10$
#   the threshold drops monotonically on the refined $n_x = 48$ grid from
#   $50.31\;\mathrm{cm}^{-1}$ to $16.54\;\mathrm{cm}^{-1}$ ($67.25\;\mathrm{cm}^{-1}$ to
#   $11.09\;\mathrm{cm}^{-1}$ at $n_x = 32$).
# - **Nonlinear `SaturableGain` L–I curve and noise-seeded `FourLevelGain` inversion clamping (Panel b):**
#   Sweeping $g_0 / g_{\mathrm{th}} \in [0.0, 3.0]$ with `SaturableGain`, the intracavity
#   intensity vanishes below threshold ($g_0 < g_{\mathrm{th}}$) and grows linearly
#   above threshold, matching our spatial-hole-burning integral theory to within
#   **$0.06\%$ relative error at $g_0 / g_{\mathrm{th}} = 2.0$** ($1.09\%$ maximum relative
#   error across all above-threshold points; $\langle I \rangle / I_{\mathrm{sat}} = 0.9044$
#   in FIT vs $0.9039$ in SHB theory at $g_0 / g_{\mathrm{th}} = 2.0$). With
#   `FourLevelGain(spontaneous_emission=True, seed=42, time_scale=80.0)` ($s = 80 \le 100$)
#   starting from zero initial electromagnetic field, Langevin spontaneous-emission
#   noise initiates lasing above threshold ($W_p = 2.2\,W_{p,\mathrm{th}}$): the
#   normalized population inversion $\Delta N(t) / \Delta N_{\mathrm{th}}$ peaks at
#   $1.5763$ before stimulated emission clamps it to **$1.0924$** ($9.24\%$ from unity).
# - **Single-mode selection & `Near2FarMonitor` doughnut far-field beam (Panels c–d and GIF):**
#   Seeding the $N = 8$ active strip array with a multi-mode superposition of $m = 1, 2, 3$
#   at $g_0 = 2.2\,g_{\mathrm{th},1} < g_{\mathrm{th},2}$, only the fundamental $m = 1$
#   BIC mode grows and saturates while $m = 2, 3$ decay. Projecting the recorded 2D FIT
#   tangential fields at $z = +650\;\mathrm{nm}$ to the far field on a fine $0.05^\circ$
#   angular grid using Wakis's `Near2FarMonitor` (`project_near2far_doughnut`), the
#   far-field intensity $I(\theta)$ exhibits a deep dark nodal null along the surface normal
#   (**$I(0^\circ) / I_{\max} = 0.000651$**, i.e. $6.51 \times 10^{-4}$, enforced by the
#   intra-cell $\sin(k_x d / 2)$ factor) and symmetric twin lobes peaking at
#   **$\theta_{\mathrm{peak,FIT}} = \pm 10.05^\circ$** (compared with the standing-wave
#   carrier angle $\theta_{\mathrm{env}} = \arcsin(\lambda_0/(2 N p)) = 8.36^\circ$ and
#   the continuum analytical lobe peak $\theta_{\mathrm{peak,theory}} = \pm 11.25^\circ$ at
#   $u \approx 1.34\,\pi$ (full expression; pure envelope $u\cos(u/2)/(\pi^2 - u^2)$ peaks at
#   $1.367\,\pi$), a $10.67\%$ relative difference; attributing this gap to an evanescent-tail
#   broader envelope of the open-strip supermode relative to a hard-wall box cosine is an
#   uncomputed hypothesis).
#
# ![Lasing Threshold, L-I Curve, Single-Mode Selection, and Doughnut Far-Field Beam](figures/03_lasing_threshold_and_doughnut.png)
# ![Animated Time-Domain Build-Up of Single-Mode BIC Lasing in the N = 8 Strip Array](figures/03b_bic_lasing_buildup.gif)

# %% [markdown]
# <a id="6-mutation-and-convergence"></a>
# ## 6. Results — Part 4: Symmetry-Breaking Mutation ($\alpha \ne 0$) and Mesh Convergence
#
# Figure `04_symmetry_breaking_mutation.png` presents the direct finite-strip symmetry-breaking
# mutation study (Task 3) and the 3-resolution spatial mesh convergence study (Task 4):
#
# - **Direct 2D FIT symmetry breaking ($\alpha \ne 0$) in the finite $N = 8$ strip lowers $Q_{\mathrm{tot}}$ and raises $g_{\mathrm{th}}(\alpha, N=8)$ quadratically (Panels a–b):**
#   Directly simulating the finite $N = 8$ strip array (`run_finite_array_ringdown`) across
#   Tutorial 6.3's bar-width asymmetry $\alpha = \Delta w / w_0 \in \{0.0, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20\}$,
#   the infinite-lattice radiative quality factor collapses as $Q_{\mathrm{rad},\infty}(\alpha) = Q_0 \alpha^{-2}$
#   ($Q_0 = 47.11$), degrading the directly measured finite-strip quality factor from
#   $Q_{\mathrm{tot}}(0, 8) = 528.34$ to $Q_{\mathrm{tot}}(0.20, 8) = 362.64$ (matching
#   the additive-loss formula $(Q_{\mathrm{edge}}^{-1}(0, 8) + Q_{\mathrm{rad},\infty}^{-1}(\alpha))^{-1} = 365.20$
#   to within $0.71\%$ maximum relative error across all $\alpha$). Consequently, the
#   finite-strip threshold gain rises quadratically from $g_{\mathrm{th}}(0, 8) = 276.10\;\mathrm{cm}^{-1}$
#   to $g_{\mathrm{th}}(0.20, 8) = 402.26\;\mathrm{cm}^{-1}$ with a fitted log–log slope of
#   **$+1.9970$** for $\Delta g_{\mathrm{th}}(\alpha, 8) = g_{\mathrm{th}}(\alpha, 8) - g_{\mathrm{th}}(0, 8)$
#   ($0.15\%$ relative error from $+2.0000$).
# - **Lasing quenching under symmetry breaking (Panel c):** To integrate long-time active
#   saturation traces efficiently, `run_symmetry_breaking_mutation_study` evolves the
#   periodic dimer unit cell (`make_dimer_solver`) augmented with an effective bulk
#   conductivity $\sigma_{\mathrm{extra}}(\alpha) = \varepsilon_0 \varepsilon_{\mathrm{Si}} \omega_0 / Q_{\mathrm{tot}}(\alpha, 8)$
#   that reproduces the total finite-strip loss rate ($Q_{\mathrm{tot}}(0.04, 8) = 518.81$
#   from the direct finite-strip `harminv` sweep, and analytic
#   $Q_{\mathrm{tot}}(0.18, 8) = (Q_{\mathrm{edge}}(0, 8)^{-1} + (Q_0 / 0.18^2)^{-1})^{-1} = 387.52$
#   from $Q_{\mathrm{rad},\infty}(0.18) = Q_0 / 0.18^2 = 1453.9$) together with an
#   explicitly disclosed rate acceleration factor `quench_time_scale = 20.0` ($65$
#   simulated FIT cycles representing $1300$ physical optical cycles at invariant
#   $g_0 / g_{\mathrm{th}}(\alpha, N=8)$). At fixed unscaled pump gain
#   $g_0 = 371.15\;\mathrm{cm}^{-1}$, the low-asymmetry cavity ($\alpha = 0.04$,
#   $g_{\mathrm{th}} = 281.17\;\mathrm{cm}^{-1} < g_0$) lases into a bright steady state
#   (`i_norm_alpha004_lasing`), whereas mutating the asymmetry to $\alpha = 0.18$ raises
#   the threshold above the pump ($g_{\mathrm{th}}(0.18, 8) > g_0$) and quenches the
#   intracavity intensity to `quench_ratio_alpha018_over_004 = 0.023096` ($2.31\%$) of
#   the lasing level.
# - **3-resolution spatial mesh convergence (Panel d):** Across $n_x \in \{32, 48, 64\}$
#   ($\Delta x \in \{21.25, 14.167, 10.625\}\;\mathrm{nm}$, $\mathrm{cpw}_{\mathrm{Si}} \in \{21.28, 31.92, 42.57\} \ge 20$,
#   and $\{4.0, 6.0, 8.0\} \ge 4$ cells across the $85\;\mathrm{nm}$ inner gap),
#   $Q_{\mathrm{rad}}(\alpha=0.10) \in \{4710.65, 4690.77, 4683.88\}$ and
#   $g_{\mathrm{th}}(\alpha=0.10) \in \{31.149, 31.261, 31.283\}\;\mathrm{cm}^{-1}$
#   converge monotonically toward their Richardson extrapolates
#   ($Q_{\mathrm{Rich}} = 4675.03$, $g_{\mathrm{th,Rich}} = 31.311\;\mathrm{cm}^{-1}$)
#   with relative errors of only $0.76\%$ ($Q$) and $0.52\%$ ($g_{\mathrm{th}}$) at
#   baseline $n_x = 32$, and $0.19\%$ ($Q$) and $0.09\%$ ($g_{\mathrm{th}}$) at
#   $n_x = 64$ (observed orders $2.009$ and $2.558$). Across the same three grids,
#   $\lambda_{\mathrm{BIC}} \in \{1573.8847, 1574.4170, 1575.2147\}\;\mathrm{nm}$ shifts
#   by $1.33\;\mathrm{nm}$ total (`lambda_bic_drift_nm = 1.33`, relative spread
#   `rel_spread_lambda_bic = 0.000844`, i.e. $0.0844\%$), with successive grid-refinement
#   increments of $+0.5324\;\mathrm{nm}$ ($32 \to 48$) and $+0.7976\;\mathrm{nm}$
#   ($48 \to 64$); Richardson extrapolation is omitted for $\lambda_{\mathrm{BIC}}$
#   because these wavelength increments are non-shrinking due to non-monotonic subpixel
#   bar-edge volume fractions as cell boundaries shift relative to the fixed bar edges.
#
# ![Symmetry-Breaking Mutation and 3-Resolution Mesh Convergence](figures/04_symmetry_breaking_mutation.png)

# %% [markdown]
# <a id="7-quantitative-summary"></a>
# ## 7. Quantitative Summary
#
# Every numerical value in the table below is loaded directly from `data/summary.json`
# (and verified automatically by `tests/test_074_bic_laser.py`):
#
# | Quantity | Symbol | FIT / Measured | Analytic / Reference | Rel. Error | Source / Task |
# | :--- | :--- | :--- | :--- | :--- | :--- |
# | Infinite-lattice BIC wavelength at $\Gamma$ ($n_x = 32$) | $\lambda_{\mathrm{BIC}}$ | 1573.8847 nm | 1575.2147 nm ($n_x = 64$) | 0.08% | Task 0 / Task 4 |
# | Symmetry-protected 2D quality factor at $\Gamma$ ($\alpha = 0$) | $Q_\Gamma$ | 1572905115.2 | $\infty$ (dark BIC) | — | Task 0 |
# | 2D momentum-space log–log slope ($Q(k_x) \propto k_x^{s_k}$) | $s_k$ | -1.9879 | -2.0000 | 0.61% | Task 0 |
# | Symmetric BIC minimum $Q$ across $w_0 \in [140, 200]\;\mathrm{nm}$ | $Q_{\mathrm{sym,min}}$ | 855060054.8 | $\infty$ ($\mathcal{C}_2$ pinned) | — | Task 0 |
# | 3D cylinder $A_1$ BIC $Q_\Gamma$ ($n_{\mathrm{PML}} = 10$ vs $6$, consistent with PML limit) | $Q_{\Gamma,\mathrm{3D}}(A_1)$ | $Q_\Gamma > 10^7$ (lower bound from a 22-cycle record; 19245231.8) | 24345.8 ($n_{\mathrm{PML}} = 6$, $790.5\times$) | — | Task 0 |
# | 3D cylinder $A_1$ BIC off-$\Gamma$ slope ($Q(k_\parallel) \propto k_\parallel^{s_{\mathrm{3D}}}$) | $s_{\mathrm{3D}}$ | -1.9560 | -2.0000 | 2.20% | Task 0 |
# | 3D FIT topological charge ($A_1$-like TE BIC, $\mathrm{cpw}_{A_1} = 21.38$) | $q_{A_1}$ | +1.0000 | +1.0000 | 0.00% | Task 0 |
# | Finite $N = 8$ strip mode-selection ratio | $Q_1(8) / Q_2(8)$ | 3.827 | 4.000 ($m^{-2}$ law) | 4.33% | Task 1 |
# | Finite $N$-period strip scaling exponent ($Q_1(N) \propto N^p$) | $p$ | 2.5518 | 2.0 ($N^2$ momentum-spread baseline) | — | Task 1 |
# | Linear threshold gain at $\alpha = 0.08$ ($\Gamma_{\mathrm{conf}} = 0.9472$) | $g_{\mathrm{th}}$ | 19.929 $\mathrm{cm}^{-1}$ | 19.940 $\mathrm{cm}^{-1}$ | 0.06% | Task 2 |
# | Saturated L–I steady intensity at $g_0 / g_{\mathrm{th}} = 2.0$ | $\langle I \rangle / I_{\mathrm{sat}}$ | 0.9044 | 0.9039 (SHB theory) | 0.06% | Task 2 |
# | Noise-seeded `FourLevelGain` inversion clamping ($s = 80$) | $\Delta N / \Delta N_{\mathrm{th}}$ | 1.0924 | 1.0000 | 9.24% | Task 2 |
# | `Near2FarMonitor` doughnut normal-incidence null ($\theta = 0^\circ$) | $I(0^\circ) / I_{\max}$ | 0.000651 | 0.000000 | — | Task 2 |
# | `Near2FarMonitor` doughnut lobe peak angle ($N = 8$ strip) | $\theta_{\mathrm{peak}}$ | 10.05 deg | 11.25 deg | 10.67% | Task 2 |
# | Symmetry-breaking threshold exponent ($\Delta g_{\mathrm{th}} \propto \alpha^{s_\alpha}$) | $s_\alpha$ | +1.9970 | +2.0000 | 0.15% | Task 3 |
# | Dynamic lasing quenching ratio $\langle I \rangle(\alpha=0.18)/\langle I \rangle(\alpha=0.04)$ | $R_{\mathrm{quench}}$ | 0.0231 | $< 0.05$ (quenched) | — | Task 3 |
# | Baseline $n_x = 32$ quality factor $Q(\alpha=0.10)$ vs Richardson | $Q_{\mathrm{rad}}(32)$ | 4710.65 | 4675.03 (Rich.) | 0.76% | Task 4 |
# | Baseline $n_x = 32$ threshold gain $g_{\mathrm{th}}(\alpha=0.10)$ vs Richardson | $g_{\mathrm{th}}(32)$ | 31.149 $\mathrm{cm}^{-1}$ | 31.311 $\mathrm{cm}^{-1}$ (Rich.) | 0.52% | Task 4 |

# %% [markdown]
# <a id="8-limits-and-failure-modes"></a>
# ## 8. Limits & Failure Modes
#
# 1. **2D $N$-period strip geometry vs full 3D $N \times N$ finite array and rate-equation time scaling (`time_scale`):**
#    To keep sequential FIT runs fast and well within laptop memory, all finite-array
#    simulations in Tasks 1–3 (`run_finite_array_ringdown`, `run_finite_array_lasing_doughnut`,
#    and `run_symmetry_breaking_mutation_study`) model a 2D $N$-period strip array that is
#    finite with open CPML boundaries along $x$ and $z$ ($N_x \times 1 \times N_z$) and
#    periodic along $y$. Consequently, `Near2FarMonitor` projects the 1D azimuthal
#    cross-section $I(\theta)$ in the $(x, z)$ plane (producing twin lobes with a dark
#    nodal null $I(0^\circ)/I_{\max} = 6.51 \times 10^{-4}$) rather than a 2D annular
#    ring in $(\theta_x, \theta_y)$. Likewise, to reach steady-state relaxation-oscillation
#    clamping within $\le 140$ optical cycles, `run_four_level_clamping_demo` uses
#    `FourLevelGain(..., time_scale=80.0)` ($\le 100$) and the time-domain quenching
#    trace in `run_symmetry_breaking_mutation_study` uses `quench_time_scale = 20.0`;
#    both scale rates uniformly so $\Delta N_{\mathrm{th}}$ and $g_0 / g_{\mathrm{th}}$
#    are invariant, while slightly broadening the quasi-steady relaxation envelope.
# 2. **Staircase quantization blindness at small asymmetry $\alpha$ (Counter-case 1):**
#    In our perturbative threshold law $g_{\mathrm{th}}(\alpha, N) = g_{\mathrm{th}}(0, N) + \kappa_{\mathrm{th}}\alpha^2$
#    where $\Delta w = \alpha w_0 / 2 \sim 1.7\text{--}8.5\;\mathrm{nm}$ is smaller than
#    the grid cell size $\Delta x = 21.25\;\mathrm{nm}$, subpixel volume-fraction averaging
#    on the Yee dual cell is mandatory. **Counter-case:** If staircase (`averaging="staircase"`)
#    thresholding is used instead of subpixel averaging at $n_x = 32$, any width
#    perturbation $\Delta w / 2 = \alpha w_0 / 4 < \Delta x / 2$ ($\alpha < 0.25$) fails
#    to cross a grid midpoint, leaving the discrete permittivity identical to $\alpha = 0$
#    and erroneously predicting $Q_{\mathrm{rad},\infty}(\alpha) \sim 10^9$ and zero
#    radiative threshold penalty across $\alpha \in [0.04, 0.20]$ (as shown in Tutorial
#    6.3's Task 4 staircase failure test).
# 3. **Multi-mode breakthrough and spatial hole burning at high pump rates $g_0 > g_{\mathrm{th},2}$ (Counter-case 2):**
#    Single-mode lasing in a finite $N$-period BIC strip array relies on operating in the
#    pump window $g_{\mathrm{th},1}(N) < g_0 < g_{\mathrm{th},2}(N) \approx 4\,g_{\mathrm{th},1}(N)$
#    where only the fundamental $m = 1$ envelope mode experiences net gain.
#    **Counter-case:** When the pump rate exceeds $g_0 \gtrsim 3.8\,g_{\mathrm{th},1}(N)$,
#    spatial hole burning by the $m = 1$ mode (which peaks at the strip center $x = 0$
#    and vanishes near the edges $x = \pm L/2$) leaves unsaturated gain near the strip
#    periphery where the two-lobe $m = 2$ mode peaks. Because the $m = 2$ envelope is
#    globally *even* under $x \mapsto -x$ (putting its intra-cell odd dipoles in phase
#    across the two halves of the strip), breakthrough of the $m = 2$ mode radiates
#    constructively at $\theta = 0^\circ$ and fills in the central dark null of the
#    doughnut beam.
# 4. **Yee-grid staggering of $\mathcal{C}_{4v}$ symmetry in 3D Bloch simulations (Counter-case 3):**
#    In a 3D Yee/FIT cell, $E_x$, $E_y$, and $E_z$ live on distinct staggered edge
#    locations $(x_{\mathrm{mid}}, y_{\mathrm{node}}, z_{\mathrm{node}})$,
#    $(x_{\mathrm{node}}, y_{\mathrm{mid}}, z_{\mathrm{node}})$, and
#    $(x_{\mathrm{node}}, y_{\mathrm{node}}, z_{\mathrm{mid}})$. **Counter-case:** If a
#    single cell-corner coordinate array $(x_{\mathrm{node}}, y_{\mathrm{node}})$ is
#    assigned to all three permittivity components $\varepsilon_x, \varepsilon_y, \varepsilon_z$,
#    the discrete cylinder is shifted by $-\Delta x / 2$ along $x$ for $E_x$ and
#    $-\Delta y / 2$ along $y$ for $E_y$, breaking discrete $\mathcal{C}_2$ symmetry by
#    $\alpha_{\mathrm{grid}} \sim \Delta x / (2 R) \approx 0.04$ and producing spurious
#    radiation at $\Gamma$ that corrupts the polarization vortex core.

# %% [markdown]
# <a id="9-references"></a>
# ## 9. References
#
# 1. A. Kodigala, T. Lepetit, Q. Gu, B. Bahari, Y. Fainman, and B. Kanté,
#    "Lasing action from photonic bound states in continuum," *Nature* **541**,
#    196–199, doi:10.1038/nature20799 (2017).
# 2. C. W. Hsu, B. Zhen, A. D. Stone, J. D. Joannopoulos, and M. Soljačić,
#    "Bound states in the continuum," *Nature Reviews Materials* **1**, 16048,
#    doi:10.1038/natrevmats.2016.48 (2016).
# 3. B. Zhen, C. W. Hsu, L. Lu, A. D. Stone, and M. Soljačić,
#    "Topological nature of optical bound states in the continuum," *Physical Review
#    Letters* **113**(25), 257401, doi:10.1103/PhysRevLett.113.257401 (2014).
# 4. K. Koshelev, S. Lepeshov, M. Liu, A. Bogdanov, and Y. Kivshar,
#    "Asymmetric metasurfaces with high-Q resonances governed by bound states in the
#    continuum," *Physical Review Letters* **121**(19), 193903,
#    doi:10.1103/PhysRevLett.121.193903 (2018).
# 5. S.-H. Chang and A. Taflove, "Finite-difference time-domain model of lasing action
#    in a four-level two-electron atomic system," *Optics Express* **12**(16),
#    3827–3833, doi:10.1364/OPEX.12.003827 (2004).
# 6. J. Andreasen, H. Cao, A. Taflove, P. Kumar, and C. Qi, "Finite-difference
#    time-domain simulation of thermal noise in open cavities," *Physical Review A*
#    **77**(2), 023810, doi:10.1103/PhysRevA.77.023810 (2008).

# %%
def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Tutorial 7.9: Lasing from a bound state in the continuum."
    )
    parser.add_argument("--quick", action="store_true", help="Run fast validation tasks (default).")
    parser.add_argument("--full", action="store_true", help="Run full resolution tasks.")
    parser.add_argument("--plot", action="store_true", help="Regenerate figures from cached data/.")
    parser.add_argument("--list-tasks", action="store_true", help="Print task manifest as JSON.")
    parser.add_argument("--task", type=int, default=None, help="Run a single task index (0..4).")
    parser.add_argument("--out", type=str, default=str(DATA_DIR), help="Output data directory.")
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS", "").split()
        args, _ = parser.parse_known_args(env_args if env_args else None)
    else:
        args, _ = parser.parse_known_args(argv)

    quick = not bool(args.full)
    tasks = h.build_tasks(quick=quick)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list_tasks:
        print(json.dumps(tasks, indent=2))
        return {"tasks": tasks}

    if args.task is not None:
        t_sel = next((t for t in tasks if int(t["index"]) == int(args.task)), None)
        if t_sel is None:
            raise ValueError(f"Task index {args.task} not found in 0..{len(tasks)-1}.")
        res = h.execute_task(t_sel, out_dir)
        print(json.dumps(res, indent=2))
        return res

    if not args.plot:
        for t_info in tasks:
            if not t_info.get("default_run", True):
                continue
            npz_file = out_dir / f"task_{int(t_info['index']):02d}.npz"
            json_file = out_dir / f"task_{int(t_info['index']):02d}.json"
            if quick and npz_file.exists() and json_file.exists():
                continue
            h.execute_task(t_info, out_dir)

    summary_all: dict[str, Any] = {}
    for t_info in tasks:
        idx = int(t_info["index"])
        jf = out_dir / f"task_{idx:02d}.json"
        if jf.exists():
            t_data = json.loads(jf.read_text())
            if idx == 0 and "Q_qbic_vs_w0" in t_data and "Q_qbic_vs_w0_nx48" in t_data:
                t_data["rel_diff_Q_qbic_w200_nx32_vs_nx48"] = round(
                    abs(float(t_data["Q_qbic_vs_w0"][-1]) - float(t_data["Q_qbic_vs_w0_nx48"][-1]))
                    / float(t_data["Q_qbic_vs_w0_nx48"][-1]),
                    4,
                )
                jf.write_text(json.dumps(t_data, indent=2))
            summary_all[f"task_{idx:02d}"] = t_data
    (out_dir / "summary.json").write_text(json.dumps(summary_all, indent=2))

    h.plot_all_figures(out_dir, FIG_DIR)
    print(f"Tutorial 7.9 completed. Summary written to {out_dir / 'summary.json'}.")
    return summary_all


if __name__ == "__main__":
    main()
