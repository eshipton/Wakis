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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/06_periodic_structures/06-03_quasi_bic_metasurface/quasi_bic_metasurface.ipynb)
#
# # Tutorial 6.3 · Quasi-bound states in the continuum: metasurface $Q \propto \alpha^{-2}$
# <h3 align="center">Symmetry-protected dark modes, Fano resonance collapse, radiative scaling $Q_{\mathrm{rad}} \propto \alpha^{-2}$, angular dispersion $Q(k_\parallel) \propto k_\parallel^{-2}$, and absorption saturation</h3>
# <p align="center"><i>Erik Shipton &middot; Wakis tutorial series (fork of CERN's Wakis)</i></p>
#
# ---
#
# ### Abstract
#
# **Dielectric metasurfaces with broken in-plane symmetry convert non-radiating
# symmetry-protected bound states in the continuum (BICs) at the $\Gamma$ point
# into sharp Fano resonances ("quasi-BICs") whose radiative quality factor scales
# universally with the asymmetry parameter $\alpha$ as
# $Q_{\mathrm{rad}}(\alpha) = Q_0\,\alpha^{-2}$ [Koshelev *et al.*, *Phys. Rev.
# Lett.* 121, 193903 (2018)]. This tutorial reproduces that universal scaling
# law—rather than the paper's 3D device—using a 2D TE sub-diffractive silicon
# bar-pair dimer metasurface ($\varepsilon_{\mathrm{Si}} = 12.11$, period
# $p = 680\;\mathrm{nm} < \lambda_{\mathrm{BIC}} \approx 1574\;\mathrm{nm}$) in
# Wakis via the Finite Integration Technique (FIT), filter-diagonalization
# resonance extraction (`wakis.resonances.ResonanceMonitor.harminv`), plane-wave
# diffraction monitors (`wakis.diffraction.DiffractionMonitor`), and the
# oblique-incidence Floquet module (`wakis.floquet`). Across seven asymmetry
# values $\alpha \in [0.02, 0.20]$ we obtain a free-fit log–log slope of $-1.9978$
# (compared with the imposed slope $-2.0000$), verify the collapse of the Fano feature at $\alpha = 0$
# ($Q_{\mathrm{BIC}} > 10^9$), cross-validate time-domain Fano fits against
# `harminv` and 1D RCWA (`wakis.floquet.rcwa_1d_grating`) to within $1.7\%$,
# demonstrate oblique-incidence momentum-space symmetry breaking
# $Q_{\mathrm{rad}}(k_\parallel) \propto k_\parallel^{-2}$ at $\alpha = 0$, and
# reproduce non-radiative $Q$ saturation and bare radiative damping recovery
# $\gamma_{\mathrm{rad}}(\alpha) = \gamma_{\mathrm{tot}}(\alpha) - \gamma_{\mathrm{tot}}(0)$.**
#
# ### Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. **explain** why a $\mathcal{C}_2$-odd eigenmode at the $\Gamma$ point
#    ($\mathbf{k}_\parallel = 0$) of a sub-diffractive periodic metasurface
#    ($p < \lambda_0$) cannot couple to the zeroth-order outgoing plane wave,
#    forming a symmetry-protected bound state in the continuum with infinite
#    radiative lifetime;
# 2. **derive** Koshelev *et al.*'s perturbative multipole coupling amplitude
#    $D_{x,y} \propto \alpha$ [Eqs. (2a)–(2b)] and the universal quadratic
#    scaling law $Q_{\mathrm{rad}}(\alpha) = Q_0\,\alpha^{-2}$ [Eq. (3)], and
#    relate the proportionality constant $Q_0$ between full-width asymmetry
#    $\alpha = \Delta w / w_0$ and single-bar relative perturbation
#    $\alpha_{\mathrm{half}} = \Delta w / (2 w_0) = \alpha / 2$;
# 3. **derive** the exact Fano transmission lineshape
#    $T(\omega) = \frac{T_0}{1 + q^2}\frac{(q + \Omega)^2}{1 + \Omega^2} + T_{\mathrm{bg}}$
#    [Eq. (1)], extract $(f_0, Q, q)$ from Wakis `DiffractionMonitor` and RCWA
#    spectra, and compare them quantitatively against filter-diagonalization
#    (`ResonanceMonitor.harminv`) ring-down analysis;
# 4. **visualize** the near-field and far-field structure of a true BIC
#    ($\alpha = 0$, purely evanescent tail $\propto e^{-\kappa_1 |z|}$) versus a
#    leaky quasi-BIC ($\alpha > 0$, radiating plane waves into $z \to \pm\infty$);
# 5. **model** intrinsic and fabrication absorption loss via dielectric
#    conductivity (`solver.use_conductivity = True`), verify the saturation law
#    $Q_{\mathrm{tot}}(\alpha) = (Q_{\mathrm{rad}}(\alpha)^{-1} + Q_{\mathrm{abs}}^{-1})^{-1}$,
#    and recover the bare radiative decay rate via
#    $\gamma_{\mathrm{rad}}(\alpha) = \gamma_{\mathrm{tot}}(\alpha) - \gamma_{\mathrm{tot}}(0)$
#    (Koshelev *et al.* Fig. 4);
# 6. **use** `wakis.floquet.FloquetCell` and `wakis.floquet.AngleSweep` to excite
#    the symmetric ($\alpha = 0$) BIC via oblique incidence ($\theta_{\mathrm{inc}} \ne 0$)
#    and verify the angular dispersion $Q_{\mathrm{rad}}(k_\parallel) \propto k_\parallel^{-2}$;
# 7. **diagnose** two critical numerical failure modes of high-$Q$ metasurface
#    simulations: (i) staircase quantization blindness when subpixel smoothing is
#    omitted ($\Delta w / 2 < \Delta x / 2$), and (ii) parasitic evanescent
#    absorption when the CPML is placed within a few decay lengths
#    $L_{\mathrm{decay}} = (2\kappa_1)^{-1}$ of the resonator array.
#
# ### Contents
#
# 0. [Set-up & command-line modes (`--quick`, `--full`, `--task`, `--list-tasks`, `--plot`)](#0-set-up)
# 1. [Physics: symmetry-protected BICs, Fano collapse, and $Q_{\mathrm{rad}} \propto \alpha^{-2}$](#1-physics)
# 2. [Unit-cell geometry, graded mesh, and 3D preflight (`wakis.estimate`)](#2-geometry-and-mesh)
# 3. [Normal-incidence Fano transmission spectra and `harminv` cross-validation](#3-fano-spectra)
# 4. [Universal $Q_{\mathrm{rad}} \propto \alpha^{-2}$ scaling and near-field / far-field maps](#4-scaling-and-fields)
# 5. [Material/fabrication loss, $Q_{\mathrm{tot}}$ saturation, and bare $Q_{\mathrm{rad}}$ recovery](#5-loss-saturation)
# 6. [Oblique-incidence angular dispersion $Q(k_\parallel) \propto k_\parallel^{-2}$ (`wakis.floquet`)](#6-oblique-dispersion)
# 7. [Resolution study, subpixel vs staircase quantization, and CPML proximity limits](#7-resolution-study)
# 8. [Limits & failure modes](#8-limits-and-failure-modes)
# 9. [Exercises](#9-exercises)
# 10. [References](#10-references)

# %% [markdown]
# <a id="0-set-up"></a>
# ## 0. Set-up & Command-Line Modes
#
# Locally, run this tutorial inside the project virtual environment (always wrapped
# with `memguard`):
#
# ```bash
# # Fast verification & figure generation (< 1 min using cached/quick tasks):
# /root/computational-electromagnetics/bin/memguard -m 2000 -- \
#   /root/computational-electromagnetics/wakis/.venv/bin/python \
#   tutorials/06_periodic_structures/06-03_quasi_bic_metasurface/quasi_bic_metasurface.py --quick
#
# # Full resolution convergence & denser Floquet angle sweep:
# /root/computational-electromagnetics/bin/memguard -m 2000 -- \
#   /root/computational-electromagnetics/wakis/.venv/bin/python \
#   tutorials/06_periodic_structures/06-03_quasi_bic_metasurface/quasi_bic_metasurface.py --full
#
# # List all tasks (including Task 5: 3D tilted silicon bar-pair unit cell):
# /root/computational-electromagnetics/bin/memguard -m 2000 -- \
#   /root/computational-electromagnetics/wakis/.venv/bin/python \
#   tutorials/06_periodic_structures/06-03_quasi_bic_metasurface/quasi_bic_metasurface.py --list-tasks
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
            "wakis @ git+https://github.com/eshipton/Wakis.git@main",
        ],
        check=True,
    )

os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

import numpy as np
from _helpers import (
    ALPHAS_DEFAULT,
    DIMER_D,
    EPS_BG,
    EPS_SI,
    F0_BIC_REF,
    HEIGHT_H,
    KOSHELEV_PAPER_REF,
    LAMBDA_BIC_REF,
    N_SI,
    PERIOD_P,
    THETAS_OBLIQUE_DEG,
    WIDTH_W0,
    dimer_bar_geometry,
    extract_ringdown_envelope,
    fano_lineshape_eq1,
    fit_fano_spectrum,
    fit_q_scaling_laws,
    make_dimer_solver,
    preflight_and_run_3d_tilted_bars_task,
    rcwa_fano_sweep,
    run_absorption_saturation_study,
    run_fano_spectrum_vs_harminv,
    run_oblique_floquet_study,
    run_resolution_and_failure_study,
    run_ringdown_harminv,
)

# %% [markdown]
# <a id="1-physics"></a>
# ## 1. Physics: Symmetry-Protected BICs, Fano Collapse, and $Q_{\mathrm{rad}} \propto \alpha^{-2}$
#
# ### 1.1 Rayleigh–Bloch Diffraction Channels and Symmetry Protection at $\Gamma$
#
# Consider a periodic dielectric metasurface lying in the $(x, y)$ plane with
# lattice period $p$ along $x$, thickness $h$ along $z \in [-h/2, +h/2]$, and
# homogeneous background permittivity $\varepsilon_{\mathrm{bg}} = 1$ (air/vacuum)
# in the half-spaces $|z| > h/2$. By Bloch's theorem, an eigenmode with in-plane
# wavevector $\mathbf{k}_\parallel = (k_x, 0)$ can be expanded outside the slab
# ($|z| > h/2$) in plane-wave diffraction harmonics (Rayleigh expansion):
#
# $$
# E_y(x, z; t) = \sum_{m = -\infty}^{+\infty} \hat{E}_{y, m}^{\pm}\,
# \exp\!\left[i (k_x + G_m) x \pm i k_{z, m} z - i \omega t\right],
# \qquad G_m = \frac{2\pi m}{p},
# $$
#
# where the normal wavenumber of the $m$-th diffraction order is
#
# $$
# k_{z, m} = \sqrt{k_0^2 - (k_x + G_m)^2}, \qquad k_0 = \frac{\omega}{c} = \frac{2\pi}{\lambda_0}.
# $$
#
# In the **sub-diffractive regime** ($p < \lambda_0$) at normal incidence
# ($k_x = 0$, the $\Gamma$ point of the first Brillouin zone):
#
# * Every non-zero diffraction order $|m| \ge 1$ has $|G_m| \ge 2\pi / p > k_0$,
#   so $k_{z, m} = i \kappa_m$ is purely imaginary with evanescent decay constant
#   $$\kappa_m = \sqrt{\left(\frac{2\pi m}{p}\right)^2 - k_0^2} > 0.$$
#   These orders are **bound to the metasurface** and decay exponentially as
#   $\exp(-\kappa_m |z|)$ without carrying time-averaged power to $z \to \pm\infty$.
# * Only the **zeroth diffraction order** $m = 0$ ($G_0 = 0, k_{z,0} = k_0$) lies
#   inside the radiative light cone and can carry energy into the far field.
#
# The outgoing zeroth-order wave amplitude $\hat{E}_{y,0}^{\pm}$ is proportional to
# the spatial average of the induced polarization current across the unit cell:
#
# $$
# \hat{E}_{y, 0}^{\pm} \propto \frac{1}{p} \int_{-p/2}^{+p/2}
# \left[\varepsilon(x, z) - 1\right] E_y(x, z)\,\mathrm{d}x.
# $$
#
# Suppose the unit cell is symmetric under in-plane two-fold rotation $\mathcal{C}_2$
# (or mirror reflection $x \mapsto -x$), so $\varepsilon(-x, z) = \varepsilon(x, z)$.
# Eigenmodes at $k_x = 0$ then split into even ($\mathcal{C}_2 = +1$) and odd
# ($\mathcal{C}_2 = -1$) parity sectors. For a **$\mathcal{C}_2$-odd eigenmode**,
# $E_y(-x, z) = -E_y(x, z)$, the integrand $[\varepsilon(x, z) - 1] E_y(x, z)$ is
# strictly odd in $x$, and its integral over $[-p/2, +p/2]$ vanishes identically:
#
# $$
# \hat{E}_{y, 0}^{\pm}\big|_{\alpha = 0} = 0 \quad \Longrightarrow \quad
# \gamma_{\mathrm{rad}}(\alpha = 0) = 0, \qquad Q_{\mathrm{rad}}(\alpha = 0) = \infty.
# $$
#
# Despite its eigenfrequency $\omega_{\mathrm{BIC}}$ lying inside the continuous
# spectrum of propagating free-space plane waves ($\omega_{\mathrm{BIC}} > 0$), the
# mode is completely decoupled from the only open radiation channel by symmetry: it
# is a **symmetry-protected bound state in the continuum (BIC)**.
#
# ---
#
# ### 1.2 Koshelev *et al.* Eqs. (2a)–(2b) & Eq. (3): Perturbative Multipole Coupling and $Q_{\mathrm{rad}} \propto \alpha^{-2}$
#
# Following Koshelev *et al.*, *Phys. Rev. Lett.* **121**, 193903 (2018), Eqs. (2a)–(2b)
# on p. 3, the radiative decay rate $\gamma_{\mathrm{rad}}$ of a sub-diffractive
# metasurface eigenmode into the $(0, 0)$ normal-incidence channel is governed by the
# far-field coupling amplitudes $\mathbf{D} = (D_x, D_y)$:
#
# $$
# \frac{\gamma_{\mathrm{rad}}}{c} = |D_x|^2 + |D_y|^2, \tag{2a}
# $$
#
# $$
# D_{x, y} = -\frac{k_0}{\sqrt{2 S_0}} \left(
# p_{x, y} \mp \frac{1}{c}\,m_{y, x} + \frac{i k_0}{6}\,Q_{xz, yz}
# \right), \tag{2b}
# $$
#
# where $S_0$ is the unit-cell area, $\mathbf{p}$ is the electric dipole moment per
# unit cell, $\mathbf{m}$ is the magnetic dipole moment, and $\hat{Q}$ is the electric
# quadrupole tensor evaluated at the eigenmode field distribution.
#
# In a dimer meta-atom composed of two dielectric bars centered at $x_1 = -d/2$ and
# $x_2 = +d/2$, the $\mathcal{C}_2$-odd BIC mode carries anti-parallel in-plane
# electric dipole moments $p_{y, 1} = +p_0$ and $p_{y, 2} = -p_0$ (or anti-parallel
# out-of-plane magnetic dipoles $m_{z,1} = +m_0, m_{z,2} = -m_0$). At $\alpha = 0$,
# the net dipole per unit cell vanishes: $p_y = p_{y,1} + p_{y,2} = 0$, so $D_y = 0$.
#
# When we break the in-plane symmetry by a dimensionless structural asymmetry parameter
# $\alpha$—for example, by tilting the bars by $\pm\theta$ so $\alpha = \sin\theta$
# [Koshelev *et al.* Fig. 1(a) & Fig. 2(a)] or by perturbing the two bar widths/lengths
# to $w_1 = w_0(1 + \alpha/2)$ and $w_2 = w_0(1 - \alpha/2)$ so
# $\alpha = (w_1 - w_2)/w_0 = \Delta w / w_0$ [Koshelev *et al.* Fig. 1(e) & Fig. 3(b)]—
# first-order perturbation theory expands the eigenmode field as
# $\mathbf{E}(\mathbf{r}; \alpha) = \mathbf{E}^{(0)}(\mathbf{r}) + \alpha\,\mathbf{E}^{(1)}(\mathbf{r}) + \mathcal{O}(\alpha^2)$.
# Because $\mathbf{E}^{(0)}$ is $\mathcal{C}_2$-odd and the structural perturbation
# $\delta\varepsilon(x, z; \alpha) \propto \alpha$ is also $\mathcal{C}_2$-odd, their
# product $\delta\varepsilon\,\mathbf{E}^{(0)}$ has a **non-zero $\mathcal{C}_2$-even**
# spatial average. The net dipole moment per unit cell therefore grows linearly with $\alpha$:
#
# $$
# p_y(\alpha) = p_{y, 1}(\alpha) + p_{y, 2}(\alpha)
# = p_0 (1 + c_1 \alpha) - p_0 (1 - c_1 \alpha) + \mathcal{O}(\alpha^3)
# = 2 c_1 p_0\,\alpha \propto \alpha,
# $$
#
# which yields a coupling amplitude $D_y(\alpha) = \kappa_D\,\alpha + \mathcal{O}(\alpha^3)$.
# Substituting into Eq. (2a) gives the radiative decay rate and radiative quality factor
# [Koshelev *et al.* Eq. (3), p. 3]:
#
# $$
# \gamma_{\mathrm{rad}}(\alpha) = c\,|D_y(\alpha)|^2 = \gamma_0\,\alpha^2,
# \qquad
# Q_{\mathrm{rad}}(\alpha) = \frac{\omega_0(\alpha)}{\gamma_{\mathrm{rad}}(\alpha)}
# = Q_0\,\alpha^{-2}. \tag{3}
# $$
#
# Taking base-10 logarithms gives a straight line on a log–log plot with universal
# slope $-2$:
#
# $$
# \log_{10} Q_{\mathrm{rad}}(\alpha) = \log_{10} Q_0 - 2\,\log_{10} \alpha.
# $$
#
# > [!NOTE]
# > **Convention on $\alpha$ vs $\alpha_{\mathrm{half}}$ ($4\times$ scaling of $Q_0$) and 2D vs 3D geometry:**
# > This tutorial reproduces the **universal scaling law** $Q_{\mathrm{rad}} \propto \alpha^{-2}$,
# > not the specific 3D device of Koshelev *et al.* (which uses 3D silicon bar pairs with finite
# > bar length at $\lambda \approx 1435\;\mathrm{nm}$ in Fig. 3b or tilted 3D bars at
# > $\lambda \approx 1571\;\mathrm{nm}$ in Fig. 2, whereas our primary model is a 2D TE bar-pair
# > cross-section at $\lambda_{\mathrm{BIC}} \approx 1574\;\mathrm{nm}$). Because the prefactor
# > $Q_0$ depends on the unit-cell geometry and dimensionality, only the power-law exponent
# > ($-2$) is universal across geometries.
# > Furthermore, in the unequal bar-pair literature, the asymmetry parameter is defined either as
# > the full width difference normalized to the mean width,
# > $\alpha \equiv (w_1 - w_2)/w_0 = \Delta w / w_0$ [where $w_{1,2} = w_0(1 \pm \alpha/2)$],
# > or as the single-bar relative half-perturbation
# > $\alpha_{\mathrm{half}} \equiv \Delta w / (2 w_0) = \alpha / 2$ [where
# > $w_{1,2} = w_0(1 \pm \alpha_{\mathrm{half}})$]. Both definitions share the exact
# > same exponent $-2$, while their prefactors differ by $2^2 = 4$:
# > $$Q_{\mathrm{rad}} = Q_0(\alpha)\,\alpha^{-2} = Q_0(\alpha_{\mathrm{half}})\,\alpha_{\mathrm{half}}^{-2}, \qquad Q_0(\alpha) = 4\,Q_0(\alpha_{\mathrm{half}}).$$
# > For our 2D silicon dimer metasurface ($p = 680\;\mathrm{nm}, d = 255\;\mathrm{nm}, w_0 = 170\;\mathrm{nm}, h = 190\;\mathrm{nm}$),
# > the slope-$-2$-imposed fit gives $Q_0(\alpha) = 47.12$ ($Q_0(\alpha_{\mathrm{half}}) = 11.78$),
# > and the unconstrained power-law fit gives slope $s = -1.9978$ with $Q_0(\alpha) = 47.39$
# > ($Q_0(\alpha_{\mathrm{half}}) = 11.85$). For comparison, Koshelev *et al.*'s different 3D
# > tilted silicon bar-pair geometry in Fig. 2(c)/Fig. 4 has $Q_0 \approx 13.7$ vs $\alpha = \sin\theta$.
#
# ---
#
# ### 1.3 Koshelev *et al.* Eq. (1): Temporal Coupled-Mode Theory and Fano Lineshape
#
# When a normal-incidence plane wave of frequency $\omega$ illuminates the metasurface,
# the transmitted field results from interference between two pathways:
# 1. a **direct (background) pathway** through the dielectric slab with slowly varying
#    transmission $T_{\mathrm{bg}}(\omega)$, and
# 2. a **resonant pathway** mediated by the leaky quasi-BIC eigenmode at complex
#    angular frequency $\tilde{\omega}_0 = \omega_0 - i \gamma / 2$ (where $\gamma$ is
#    the full-width at half-maximum energy decay rate, $\Delta f_{\mathrm{FWHM}} = \gamma / 2\pi$,
#    and $Q = \omega_0 / \gamma = f_0 / \Delta f_{\mathrm{FWHM}}$).
#
# Temporal coupled-mode theory (TCMT) yields the exact asymmetric **Fano lineshape**
# [Koshelev *et al.* Eq. (1), p. 3]:
#
# $$
# T(\omega) = \frac{T_0}{1 + q^2}\,\frac{(q + \Omega)^2}{1 + \Omega^2} + T_{\mathrm{bg}}(\omega),
# \qquad
# \Omega \equiv \frac{2(\omega - \omega_0)}{\gamma} = \frac{2(f - f_0)}{\Delta f_{\mathrm{FWHM}}}, \tag{1}
# $$
#
# where $q$ is the dimensionless **Fano asymmetry parameter** (ratio of resonant to
# direct scattering amplitudes) and $T_0 = T_{\max} - T_{\min}$ is the peak-to-dip
# modulation depth. Differentiating Eq. (1) with respect to $\Omega$ gives the exact
# locations of the transmission minimum (Fano antiresonance) and maximum:
#
# $$
# \Omega_{\min} = -q \implies f_{\min} = f_0 - \frac{q}{2}\,\Delta f_{\mathrm{FWHM}},
# \qquad
# \Omega_{\max} = +\frac{1}{q} \implies f_{\max} = f_0 + \frac{1}{2q}\,\Delta f_{\mathrm{FWHM}}.
# $$
#
# The frequency separation between the Fano peak and dip is therefore
#
# $$
# f_{\max} - f_{\min} = \frac{\Delta f_{\mathrm{FWHM}}}{2}\left(q + \frac{1}{q}\right),
# \qquad
# Q = \frac{f_0}{\Delta f_{\mathrm{FWHM}}} = \frac{f_0}{2\,|f_{\max} - f_{\min}|}\left(|q| + \frac{1}{|q|}\right).
# $$
#
# At $\alpha \to 0$, the radiative linewidth vanishes ($\gamma = \omega_0 \alpha^2 / Q_0 \to 0$):
# the peak and dip coalesce ($f_{\max} - f_{\min} \to 0$) and the resonant feature
# disappears completely from $T(\omega)$, leaving only the smooth Fabry–Pérot background.
# Koshelev *et al.* term this the **collapse of the Fano resonance** at the BIC.
#
# ---
#
# ### 1.4 Oblique-Incidence Momentum-Space Symmetry Breaking: $Q_{\mathrm{rad}}(k_\parallel) \propto k_\parallel^{-2}$
#
# Even when the unit cell is structurally symmetric ($\alpha = 0, w_1 = w_2 = w_0$),
# tilting the incident wave by angle $\theta_{\mathrm{inc}}$ introduces a non-zero
# in-plane Bloch wavenumber $k_x = k_0 \sin\theta_{\mathrm{inc}} \ne 0$ that breaks the
# $x \mapsto -x$ symmetry in momentum space. For a symmetric anti-parallel dipole pair
# $p_{y,1} = +p_0$ at $x_1 = -d/2$ and $p_{y,2} = -p_0$ at $x_2 = +d/2$, the coupling
# amplitude to the $(0, 0)$ Floquet harmonic at in-plane wavenumber $k_x$ picks up the
# retarded phase factor $e^{-i k_x x}$ across the bar spacing $d$:
#
# $$
# D_y(k_x)\big|_{\alpha = 0} \propto p_0\,e^{+i k_x d / 2} - p_0\,e^{-i k_x d / 2}
# = 2 i p_0\,\sin\!\left(\frac{k_x d}{2}\right)
# = i p_0 d\,k_x + \mathcal{O}(k_x^3).
# $$
#
# Consequently, for small angles ($k_x d \ll 1$), the radiative decay rate scales as
# $\gamma_{\mathrm{rad}}(k_x) \propto |D_y(k_x)|^2 \propto k_x^2$, and the quality
# factor follows the universal angular inverse-square law:
#
# $$
# Q_{\mathrm{rad}}(k_\parallel)\big|_{\alpha = 0}
# = Q_{0, k}\,\left(\frac{k_\parallel p}{2\pi}\right)^{-2}
# = Q_{0, \theta}\,\sin^{-2}\theta_{\mathrm{inc}} \propto k_\parallel^{-2}.
# $$
#
# > [!IMPORTANT]
# > **Why dimerization $d = \frac{3}{8}p \ne \frac{1}{2}p$ is required to excite the BIC at oblique incidence:**
# > If two identical bars ($\alpha = 0$) are spaced by $d = p / 2$, the unperturbed
# > structure actually has a **half-period translational symmetry** $\tilde{p} = p / 2$.
# > The $\mathcal{C}_2$-odd mode at period $p$ is then a zone-boundary ($k_x = \pi / \tilde{p}$)
# > mode of the $\tilde{p} = p / 2$ lattice, and every odd Fourier harmonic of the
# > permittivity $\hat{\varepsilon}_{2m+1}$ vanishes identically! An oblique plane wave
# > with $k_x = k_0 \sin\theta_{\mathrm{inc}}$ can only scatter through reciprocal
# > lattice vectors $G_{2m} = 2\pi(2m)/p$ of the $p/2$ lattice and **cannot couple** to
# > the BIC at $G_{\pm 1} = \pm 2\pi / p$. Setting $d = \frac{3}{8}p = 255\;\mathrm{nm} \ne p/2$
# > dimerizes the array (inner gap $g_{\mathrm{in}} = 85\;\mathrm{nm}$, outer gap
# > $g_{\mathrm{out}} = 255\;\mathrm{nm}$), giving the symmetric ($\alpha = 0$) unit
# > cell true period $p = 680\;\mathrm{nm}$ while preserving exact mirror symmetry
# > $x \mapsto -x$. As a result, **the exact same BIC eigenmode at $\lambda_{\mathrm{BIC}} = 1573.885\;\mathrm{nm}$**
# > is unlocked both by structural asymmetry $\alpha \ne 0$ at $\theta_{\mathrm{inc}} = 0^\circ$
# > and by oblique incidence $\theta_{\mathrm{inc}} \ne 0^\circ$ at $\alpha = 0$!
#
# ---
#
# ### 1.5 Non-Radiative Loss, $Q_{\mathrm{tot}}$ Saturation, and Bare $Q_{\mathrm{rad}}$ Recovery (Koshelev *et al.* Fig. 4)
#
# In real metasurfaces, material absorption ($\mathrm{Im}\,\varepsilon > 0$), surface
# roughness, and finite array size introduce a non-radiative damping rate
# $\gamma_{\mathrm{abs}} = \omega_0 / Q_{\mathrm{abs}}$ even at $\alpha = 0$
# [Koshelev *et al.*, p. 4 & Fig. 4]. In Wakis, an effective conductivity $\sigma$
# inside the silicon bars (`solver.use_conductivity = True`) produces an Ohmic loss
# tangent $\tan\delta = \sigma / (\omega_0 \varepsilon_0 \varepsilon_{\mathrm{Si}})$
# and an absorption quality factor
#
# $$
# Q_{\mathrm{abs}} = \frac{\omega_0 \varepsilon_0 \varepsilon_{\mathrm{Si}}}{\Gamma_{\mathrm{conf}}\,\sigma},
# \qquad
# \Gamma_{\mathrm{conf}} \equiv \frac{\int_{\mathrm{Si}} \varepsilon_{\mathrm{Si}} |\mathbf{E}|^2\,\mathrm{d}V}{\int_{\mathrm{cell}} \varepsilon(\mathbf{r}) |\mathbf{E}|^2\,\mathrm{d}V},
# $$
#
# where $\Gamma_{\mathrm{conf}} \approx 0.9476$ is the electric-energy confinement
# factor of the BIC mode inside the high-index silicon bars. Since the radiative and
# non-radiative decay rates add ($\gamma_{\mathrm{tot}}(\alpha) = \gamma_{\mathrm{rad}}(\alpha) + \gamma_{\mathrm{abs}}$),
# the total quality factor saturates at $Q_{\mathrm{tot}}(0) = Q_{\mathrm{abs}}$ as
# $\alpha \to 0$:
#
# $$
# \frac{1}{Q_{\mathrm{tot}}(\alpha)} = \frac{1}{Q_{\mathrm{rad}}(\alpha)} + \frac{1}{Q_{\mathrm{abs}}}
# = \frac{\alpha^2}{Q_0} + \frac{1}{Q_{\mathrm{abs}}}.
# $$
#
# Conversely, as shown by Koshelev *et al.* on p. 4 (Fig. 4 caption and text), subtracting
# the $\alpha = 0$ non-radiative decay rate $\gamma_{\mathrm{tot}}(0) = \gamma_{\mathrm{abs}}$
# from $\gamma_{\mathrm{tot}}(\alpha)$ recovers the bare radiative damping rate:
#
# $$
# \gamma_{\mathrm{rad}}(\alpha) = \gamma_{\mathrm{tot}}(\alpha) - \gamma_{\mathrm{tot}}(0)
# \quad \Longleftrightarrow \quad
# Q_{\mathrm{rad,rec}}(\alpha) = \left(\frac{1}{Q_{\mathrm{tot}}(\alpha)} - \frac{1}{Q_{\mathrm{tot}}(0)}\right)^{-1}
# = Q_0\,\alpha^{-2}.
# $$

# %% [markdown]
# <a id="2-geometry-and-mesh"></a>
# ## 2. Unit-Cell Geometry, Graded Mesh, and 3D Preflight (`wakis.estimate`)
#
# ### 2.1 2D Silicon Dimer Unit Cell vs Koshelev *et al.* Fig. 1(e) & Fig. 2(a)
#
# Our primary simulation domain is a 2D cross-sectional Bloch unit cell
# ($x \in [-p/2, +p/2],\, z \in [-z_{\max}, +z_{\max}]$ with a 1-cell extruded
# periodic $y$ axis, TE polarization $E_y, H_x, H_z$) representing an unequal
# silicon bar-pair dimer metasurface in the symmetry class of Koshelev *et al.*
# Fig. 1(e) & Fig. 3(b). Note that **this tutorial reproduces the universal
# $\alpha^{-2}$ scaling law, not the paper's 3D device**: Koshelev *et al.*'s
# numerical examples use 3D finite-length silicon bar pairs ($\lambda_{\mathrm{BIC}} \approx 1571\;\mathrm{nm}$
# for tilted bars in Fig. 2 and $\lambda_{\mathrm{BIC}} \approx 1435\;\mathrm{nm}$ in Fig. 3b),
# whereas our 2D model uses infinitely extruded dimer bars ($\lambda_{\mathrm{BIC}} \approx 1574\;\mathrm{nm}$):
#
# * **Lattice period:** $p = 680\;\mathrm{nm}$, so $p / \lambda_{\mathrm{BIC}} = 680 / 1573.9 = 0.4321 < 0.5$
#   (strictly sub-diffractive at all angles $\theta_{\mathrm{inc}} \in [0^\circ, 90^\circ)$!).
# * **Bar centers & widths:** Two silicon bars ($\varepsilon_{\mathrm{Si}} = 3.48^2 = 12.1104$,
#   background $\varepsilon_{\mathrm{bg}} = 1.0$, height $h = 190\;\mathrm{nm}$)
#   centered at $x_1 = -d/2$ and $x_2 = +d/2$ with $d = \frac{3}{8}p = 255\;\mathrm{nm}$
#   and nominal width $w_0 = \frac{1}{4}p = 170\;\mathrm{nm}$ (inner gap
#   $g_{\mathrm{in}} = d - w_0 = \frac{1}{8}p = 85\;\mathrm{nm}$, outer gap
#   $g_{\mathrm{out}} = 255\;\mathrm{nm}$).
# * **Symmetry-breaking perturbation:** Bar 1 widens to $w_1 = w_0(1 + \alpha/2)$
#   and Bar 2 narrows to $w_2 = w_0(1 - \alpha/2)$, so $\alpha = (w_1 - w_2)/w_0$
#   while the total silicon volume fraction per unit cell remains constant.
# * **Grid rules & dual-cell alignment:** At our baseline resolution $n_x = 32$
#   ($\Delta x = p / 32 = 21.25\;\mathrm{nm}$) with a graded $z$ mesh
#   ($\Delta z_{\min} = 15.0\;\mathrm{nm}$ across the bars, grading smoothly by
#   $\le 1.22$ to $\Delta z_{\max} = 45.0\;\mathrm{nm}$ in the air buffer out to
#   $z_{\max} = \pm 1.30\;\mu\mathrm{m}$ plus $10$-cell CPMLs):
#   - Wavelength in silicon: $\lambda_{\mathrm{Si}} = \lambda_{\mathrm{BIC}} / n_{\mathrm{Si}} = 452.3\;\mathrm{nm}$,
#     giving $\mathrm{cpw}_{\mathrm{Si}, x} = 21.28 \ge 20$ and $\mathrm{cpw}_{\mathrm{Si}, z} = 30.15 \ge 20$.
#   - Smallest geometric feature: inner air gap $g_{\mathrm{in}} = 85.0\;\mathrm{nm}$,
#     resolved by exactly $g_{\mathrm{in}} / \Delta x = 4.00 \ge 4$ cells (and bar
#     widths $w_0 / \Delta x = 8.00$ cells, height $h / \Delta z_{\min} = 12.67$ cells).
#   - Because $d = \frac{6}{16}p$ and $w_0 = \frac{4}{16}p$, the unperturbed bar
#     edges $\pm d/2 \pm w_0/2 \in \{\pm \frac{5}{16}p, \pm \frac{1}{16}p\}$ sit at
#     exact integer multiples of $\Delta x$ on any grid with $n_x = 32 m$
#     ($n_x \in \{32, 48, 64, 96\}$), i.e. at the **exact centers** of primary $E_y$
#     dual cells $[x_i - \Delta x/2, x_i + \Delta x/2]$. Consequently, subpixel width
#     perturbations $\pm \frac{\alpha}{4} w_0$ never cross dual-cell boundaries for
#     $\alpha < 0.25 / m$, yielding smooth 4-digit accuracy in $Q_{\mathrm{rad}}(\alpha)$.
#
# ### 2.2 3D Tilted Silicon Bar-Pair Unit Cell Preflight (`wakis.estimate`, Task 5)
#
# We also provide Task 5 (`--task 5`), which models the full 3D tilted silicon
# bar-pair unit cell of Koshelev *et al.* Fig. 2(a) ($p = 1320\;\mathrm{nm}$, bars
# $660 \times 220 \times 200\;\mathrm{nm}^3$, spacing $d = 660\;\mathrm{nm}$, tilt
# angle $\theta$ with $\alpha = \sin\theta$). Calling `wakis.estimate.estimate` and
# `wakis.estimate.preflight` on the $(44 \times 44 \times 36)$ 3D unit cell ($69,696$
# cells, $1600$ steps) reports a peak host RAM of **$0.62\;\mathrm{GB}$**, passing
# the $2\;\mathrm{GB}$ `"laptop"` (`memguard`) preflight check before launching.

# %%
def build_tasks(quick: bool = True) -> list[dict[str, Any]]:
    """
    Build the task manifest for Tutorial 6.3 (`--list-tasks`, `--task`, `--quick`, `--full`).

    Parameters
    ----------
    quick : bool, optional
        If True (`--quick`), configures fast parameters (< 1 min total); if False
        (`--full`), includes the `nx = 96` fine convergence level and denser
        `AngleSweep` sampling.

    Returns
    -------
    list of dict
        Task descriptors with indices `0..5`.
    """
    return [
        {
            "index": 0,
            "name": "fano_and_alpha_scaling",
            "description": (
                "7-point alpha sweep (0.02..0.20) + alpha=0 BIC via Wakis 2D FIT harminv, "
                "1D RCWA Fano sweep, and direct DiffractionMonitor Fano fit vs harminv."
            ),
            "nx": 32,
            "alphas": list(ALPHAS_DEFAULT),
            "fano_alpha": 0.48,
            "fano_cycles": 260.0,
            "default_run": True,
        },
        {
            "index": 1,
            "name": "bic_near_fields_and_gif",
            "description": (
                "2D (x, z) eigenmode field maps (Ey, Hx, Hz) and 28-frame ring-down GIF "
                "comparing the non-radiating BIC (alpha=0) vs radiating quasi-BIC (alpha=0.24)."
            ),
            "nx": 32,
            "alpha_bic": 0.0,
            "alpha_qbic": 0.24,
            "n_gif_frames": 28,
            "default_run": True,
        },
        {
            "index": 2,
            "name": "absorption_saturation",
            "description": (
                "Material/fabrication loss Q_tot(alpha) saturation for Q_abs in "
                "(1e5, 2e4, 5e3) and bare radiative damping recovery (Koshelev Fig. 4)."
            ),
            "nx": 32,
            "alphas": list(ALPHAS_DEFAULT),
            "q_abs_targets": [1.0e5, 2.0e4, 5.0e3],
            "default_run": True,
        },
        {
            "index": 3,
            "name": "oblique_floquet_dispersion",
            "description": (
                "Oblique-incidence BIC dispersion Q_rad(k_parallel) ~ k_parallel^(-2) at "
                "alpha=0 via Bloch harminv and wakis.floquet.AngleSweep."
            ),
            "nx": 32,
            "thetas_deg": list(THETAS_OBLIQUE_DEG),
            "quick": bool(quick),
            "default_run": True,
        },
        {
            "index": 4,
            "name": "resolution_and_failure_study",
            "description": (
                "Mesh convergence ladder (nx=32..64 in --quick, 32..96 in --full), "
                "subpixel vs staircase quantization blindness, and CPML proximity scan."
            ),
            "quick": bool(quick),
            "default_run": True,
        },
        {
            "index": 5,
            "name": "koshelev_fig2_3d_tilted_bars",
            "description": (
                "3D tilted silicon bar-pair unit cell (Koshelev et al. Fig. 2a): "
                "wakis.estimate preflight in standard runs; full 3D solver when invoked via --task 5."
            ),
            "nx": 44,
            "ny": 44,
            "nz": 36,
            "thetas_deg": [0.0, 6.0, 9.0, 12.0],
            "default_run": False,
        },
    ]


def execute_task(
    task: dict[str, Any], out_dir: Path, force_3d_solver: bool = False
) -> dict[str, Any]:
    """
    Execute a single tutorial task and save `task_XX.npz` and `task_XX.json` in `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = int(task["index"])
    name = str(task["name"])
    t0 = time.perf_counter()

    if idx == 0:
        nx = int(task["nx"])
        alphas = np.asarray(task["alphas"], dtype=float)
        r_bic = run_ringdown_harminv(alpha=0.0, nx=nx)
        lam_fit, f0_fit, q_fit, gamma_fit, err_fit = [], [], [], [], []
        t_ring_traces = []
        sig_ring_traces = []

        for a_val in alphas:
            r_a = run_ringdown_harminv(alpha=float(a_val), nx=nx)
            lam_fit.append(r_a["lambda0_nm"])
            f0_fit.append(r_a["f0_hz"])
            q_fit.append(r_a["Q"])
            gamma_fit.append(r_a["gamma_s_inv"])
            err_fit.append(r_a["harminv_error_hz"])
            t_ring_traces.append(r_a["t_ring_fs"])
            sig_ring_traces.append(r_a["signal_left"])

        q_arr = np.asarray(q_fit, dtype=float)
        lam_arr = np.asarray(lam_fit, dtype=float)
        slope_fit, intercept_fit = np.polyfit(np.log10(alphas), np.log10(q_arr), 1)
        q0_alpha = float(10.0**intercept_fit)
        q0_alpha_half = 0.25 * q0_alpha

        rcwa_res = rcwa_fano_sweep(alphas=alphas)
        fano_fit_res = run_fano_spectrum_vs_harminv(
            alpha=float(task["fano_alpha"]),
            nx=nx,
            n_cycles=float(task["fano_cycles"]),
        )
        preflight_3d = preflight_and_run_3d_tilted_bars_task(run_solver=False)

        np.savez_compressed(
            out_dir / f"task_{idx:02d}.npz",
            alphas=alphas,
            alphas_half=0.5 * alphas,
            lambda_bic_fit_nm=r_bic["lambda0_nm"],
            Q_bic_fit=r_bic["Q"],
            t_ring_bic_fs=r_bic["t_ring_fs"],
            sig_ring_bic=r_bic["signal_left"],
            lambda0_fit_nm=lam_arr,
            f0_fit_hz=np.asarray(f0_fit, dtype=float),
            Q_fit=q_arr,
            Q_alpha2_fit=q_arr * (alphas**2),
            Q_alpha_half2_fit=q_arr * ((0.5 * alphas) ** 2),
            gamma_fit_s_inv=np.asarray(gamma_fit, dtype=float),
            harminv_err_hz=np.asarray(err_fit, dtype=float),
            log_slope_fit=slope_fit,
            Q0_alpha_fit=q0_alpha,
            Q0_alpha_half_fit=q0_alpha_half,
            t_ring_traces_fs=np.asarray(t_ring_traces, dtype=float),
            sig_ring_traces=np.asarray(sig_ring_traces, dtype=float),
            lambda_bic_rcwa_nm=rcwa_res["lambda_bic_rcwa_nm"],
            lambda0_rcwa_nm=rcwa_res["lambda0_nm"],
            Q_rcwa=rcwa_res["Q_fano"],
            Q_alpha2_rcwa=rcwa_res["Q_alpha2"],
            q_fano_rcwa=rcwa_res["q_fano"],
            log_slope_rcwa=rcwa_res["log_slope"],
            Q0_rcwa=rcwa_res["Q0_fit"],
            local_wls_nm=rcwa_res["local_wls_nm"],
            local_T=rcwa_res["local_T"],
            local_T_fit=rcwa_res["local_T_fit"],
            fan_wls_nm=rcwa_res["fan_wls_nm"],
            fan_alphas=rcwa_res["fan_alphas"],
            fan_T=rcwa_res["fan_T"],
            fan_R=rcwa_res["fan_R"],
            fano_wls_nm=fano_fit_res["wls_nm"],
            fano_R_asym=fano_fit_res["R_asym"],
            fano_T_asym=fano_fit_res["T_asym"],
            fano_T_fit=fano_fit_res["T_fano_fit"],
            fano_R_sym_alpha0=fano_fit_res["R_sym_alpha0"],
            fano_T_sym_alpha0=fano_fit_res["T_sym_alpha0"],
            fano_Q_harminv=fano_fit_res["Q_harminv"],
            fano_Q_fit=fano_fit_res["Q_fano"],
            fano_q_param=fano_fit_res["q_fano"],
            fano_lambda0_harminv_nm=fano_fit_res["lambda0_harminv_nm"],
            fano_lambda0_fit_nm=fano_fit_res["lambda0_fano_nm"],
            x_nodes_um=r_bic["meta"]["x_nodes"] * 1e6,
            z_nodes_um=r_bic["meta"]["z_nodes"] * 1e6,
            eps_r_bic_2d=r_bic["meta"]["eps_r_2d"],
        )
        summary = {
            "task": idx,
            "name": name,
            "nx": nx,
            "dx_nm": r_bic["meta"]["dx_nm"],
            "dz_min_nm": r_bic["meta"]["dz_min_nm"],
            "cpw_si_min": round(r_bic["meta"]["cpw_si_min"], 2),
            "cells_across_gap": round(r_bic["meta"]["cells_across_gap"], 2),
            "d_air_to_pml_nm": round(r_bic["meta"]["d_air_to_pml_nm"], 1),
            "lambda_bic_fit_nm": round(r_bic["lambda0_nm"], 4),
            "Q_bic_fit": round(r_bic["Q"], 1),
            "lambda_bic_rcwa_nm": round(rcwa_res["lambda_bic_rcwa_nm"], 4),
            "alphas": [float(x) for x in alphas],
            "Q_fit": [round(float(x), 2) for x in q_arr],
            "Q_rcwa": [round(float(x), 2) for x in rcwa_res["Q_fano"]],
            "Q_alpha2_fit": [round(float(x), 4) for x in q_arr * (alphas**2)],
            "Q_alpha2_rcwa": [round(float(x), 4) for x in rcwa_res["Q_alpha2"]],
            "log_slope_fit": round(float(slope_fit), 5),
            "log_slope_rcwa": round(float(rcwa_res["log_slope"]), 5),
            "Q0_alpha_fit": round(q0_alpha, 4),
            "Q0_alpha_half_fit": round(q0_alpha_half, 4),
            "fano_alpha": float(task["fano_alpha"]),
            "fano_Q_harminv": round(fano_fit_res["Q_harminv"], 2),
            "fano_Q_fit": round(fano_fit_res["Q_fano"], 2),
            "fano_q_param": round(fano_fit_res["q_fano"], 4),
            "fano_rel_diff_Q": round(fano_fit_res["rel_diff_Q_fano_vs_harminv"], 4),
            "fano_contrast_alpha0": round(fano_fit_res["contrast_alpha0"], 5),
            "preflight_3d": preflight_3d,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    elif idx == 1:
        nx = int(task["nx"])
        a_bic = float(task["alpha_bic"])
        a_qbic = float(task["alpha_qbic"])
        n_frames = int(task["n_gif_frames"])
        r_bic = run_ringdown_harminv(
            alpha=a_bic, nx=nx, ring_cycles=38.0, capture_fields=True, n_gif_frames=n_frames
        )
        r_qbic = run_ringdown_harminv(
            alpha=a_qbic, nx=nx, ring_cycles=38.0, capture_fields=True, n_gif_frames=n_frames
        )
        np.savez_compressed(
            out_dir / f"task_{idx:02d}.npz",
            alpha_bic=a_bic,
            alpha_qbic=a_qbic,
            x_nodes_um=r_bic["meta"]["x_nodes"] * 1e6,
            z_nodes_um=r_bic["meta"]["z_nodes"] * 1e6,
            eps_bic=r_bic["meta"]["eps_r_2d"],
            eps_qbic=r_qbic["meta"]["eps_r_2d"],
            ey_bic=r_bic["ey_2d"],
            hx_bic=r_bic["hx_2d"],
            hz_bic=r_bic["hz_2d"],
            ey_qbic=r_qbic["ey_2d"],
            hx_qbic=r_qbic["hx_2d"],
            hz_qbic=r_qbic["hz_2d"],
            frames_ey_bic=r_bic["frames_ey"],
            frames_ey_qbic=r_qbic["frames_ey"],
            frames_t_fs=r_bic["frames_t_fs"],
            Q_bic=r_bic["Q"],
            Q_qbic=r_qbic["Q"],
            lambda_bic_nm=r_bic["lambda0_nm"],
            lambda_qbic_nm=r_qbic["lambda0_nm"],
        )
        summary = {
            "task": idx,
            "name": name,
            "alpha_bic": a_bic,
            "alpha_qbic": a_qbic,
            "lambda_bic_nm": round(r_bic["lambda0_nm"], 4),
            "lambda_qbic_nm": round(r_qbic["lambda0_nm"], 4),
            "Q_bic": round(r_bic["Q"], 1),
            "Q_qbic": round(r_qbic["Q"], 2),
            "n_gif_frames": int(len(r_bic["frames_t_fs"])),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    elif idx == 2:
        res_abs = run_absorption_saturation_study(
            alphas=task["alphas"],
            q_abs_targets=task["q_abs_targets"],
            nx=int(task["nx"]),
        )
        np.savez_compressed(out_dir / f"task_{idx:02d}.npz", **res_abs)
        summary = {
            "task": idx,
            "name": name,
            "gamma_confinement": round(res_abs["gamma_confinement"], 5),
            "q_abs_targets": [float(x) for x in res_abs["q_abs_targets"]],
            "Q_tot_alpha0": [round(float(x), 1) for x in res_abs["Q_tot_alpha0"]],
            "max_rel_err_Q_tot_pred": round(res_abs["max_rel_err_Q_tot_pred"], 5),
            "max_rel_err_Q_rad_recovered": round(res_abs["max_rel_err_Q_rad_recovered"], 5),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    elif idx == 3:
        res_obl = run_oblique_floquet_study(
            thetas_deg=task["thetas_deg"],
            nx=int(task["nx"]),
            quick=bool(task["quick"]),
        )
        np.savez_compressed(out_dir / f"task_{idx:02d}.npz", **res_obl)
        summary = {
            "task": idx,
            "name": name,
            "thetas_deg": [float(x) for x in res_obl["thetas_deg"]],
            "lambda0_bloch_nm": [round(float(x), 4) for x in res_obl["lambda0_bloch_nm"]],
            "Q_bloch": [round(float(x), 2) for x in res_obl["Q_bloch"]],
            "Q_sin2": [round(float(x), 4) for x in res_obl["Q_sin2"]],
            "log_slope_sin_theta": round(res_obl["log_slope_sin_theta"], 5),
            "Q0_sin_theta": round(res_obl["Q0_sin_theta"], 4),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    elif idx == 4:
        res_conv = run_resolution_and_failure_study(quick=bool(task["quick"]))
        np.savez_compressed(out_dir / f"task_{idx:02d}.npz", **res_conv)
        summary = {
            "task": idx,
            "name": name,
            "nx_list": [int(x) for x in res_conv["nx_list"]],
            "dx_nm": [round(float(x), 3) for x in res_conv["dx_nm"]],
            "cpw_si": [round(float(x), 2) for x in res_conv["cpw_si"]],
            "cells_across_gap": [round(float(x), 2) for x in res_conv["cells_across_gap"]],
            "lambda_bic_nm": [round(float(x), 4) for x in res_conv["lambda_bic_nm"]],
            "Q_alpha2_a10_subpixel": [
                round(float(x), 4) for x in res_conv["Q_alpha2_a10_subpixel"]
            ],
            "Q_staircase_nx32": [round(float(x), 2) for x in res_conv["Q_staircase_nx32"]],
            "slope_staircase_nx32": round(res_conv["slope_staircase_nx32"], 5),
            "d_air_to_pml_nm": [round(float(x), 1) for x in res_conv["d_air_to_pml_nm"]],
            "Q_pml_bic": [round(float(x), 1) for x in res_conv["Q_pml_bic"]],
            "Q_pml_alpha02": [round(float(x), 1) for x in res_conv["Q_pml_alpha02"]],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    elif idx == 5:
        res_3d = preflight_and_run_3d_tilted_bars_task(
            thetas_deg=task["thetas_deg"],
            nx=int(task["nx"]),
            ny=int(task["ny"]),
            nz=int(task["nz"]),
            run_solver=bool(force_3d_solver),
        )
        np.savez_compressed(
            out_dir / f"task_{idx:02d}.npz",
            thetas_deg=np.asarray(res_3d["thetas_deg"], dtype=float),
            alphas_sin_theta=np.asarray(res_3d["alphas_sin_theta"], dtype=float),
            n_cells=res_3d["n_cells"],
            peak_ram_gb=res_3d["peak_ram_gb"],
            est_wall_seconds_cpu=res_3d["est_wall_seconds_cpu"],
            lambda0_3d_nm=np.asarray(res_3d.get("lambda0_3d_nm", []), dtype=float),
            Q_3d=np.asarray(res_3d.get("Q_3d", []), dtype=float),
        )
        summary = {
            "task": idx,
            "name": name,
            **res_3d,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
    else:
        raise ValueError(f"Unknown task index {idx}.")

    (out_dir / f"task_{idx:02d}.json").write_text(json.dumps(summary, indent=2))
    return summary

# %%
# Colour-blind-safe Okabe-Ito palette
PALETTE = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "black": "#111111",
    "grey": "#666666",
}


def _plot_setup_schematic_and_mesh(d0: Any, fig_dir: Path) -> None:
    """
    Generate the two-figure setup layout via ``wakis.schematic.save_setup_figures``:
    - ``00a_setup_geometry.png``, ``00b_setup_mesh_and_excitation.png``, and
      ``00_setup_table.md`` for the primary 2D TE width-asymmetric silicon
      bar-pair dimer metasurface (`make_dimer_solver(alpha=0.14, nx=32)`).
    - ``00a_setup_geometry_oblique.png``, ``00b_setup_mesh_and_excitation_oblique.png``,
      and ``00_setup_table_oblique.md`` (`suffix='_oblique'`) for the oblique-incidence
      Floquet–Bloch symmetric dimer setup (`alpha = 0`, `bc_x = 'bloch'`,
      `FloquetCell` / `AngleSweep`).
    """
    from matplotlib.axes._base import _TransformedBoundsLocator
    from scipy.constants import c as c_light
    from wakis import figstyle, schematic
    from wakis.sources import GaussianSource
    from wakis.templates.photonics import Box

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figstyle.apply_style()

    # ------------------------------------------------------------------
    # 1. Primary setup: Width-asymmetric dimer metasurface (alpha = 0.14, nx = 32, graded z)
    # ------------------------------------------------------------------
    solver_asym, meta_asym = make_dimer_solver(alpha=0.14, nx=32)
    geom_asym = meta_asym["geom"]
    dx = PERIOD_P / 32.0

    bar1 = Box(
        center=(geom_asym["x1_c"], 0.5 * dx, 0.0),
        size=(geom_asym["w1"], dx, HEIGHT_H),
        material=EPS_SI,
    )
    bar1.name = f"Si bar 1 (w₁ = {geom_asym['w1']*1e9:.1f} nm, h = 190 nm)"
    bar2 = Box(
        center=(geom_asym["x2_c"], 0.5 * dx, 0.0),
        size=(geom_asym["w2"], dx, HEIGHT_H),
        material=EPS_SI,
    )
    bar2.name = f"Si bar 2 (w₂ = {geom_asym['w2']*1e9:.1f} nm, h = 190 nm)"
    structs_asym = [bar1, bar2]
    solver_asym._assigned_structures = tuple(structs_asym)

    mat_names_asym = {
        "background": "Air / vacuum background (ε_bg = 1.00, n = 1.00)",
        "region_0": f"Si bar 1 (w₁ = {geom_asym['w1']*1e9:.1f} nm, ε_Si = 12.11, n_Si = 3.48)",
        "region_1": f"Si bar 2 (w₂ = {geom_asym['w2']*1e9:.1f} nm, ε_Si = 12.11, n_Si = 3.48)",
    }

    wf_asym = GaussianSource(frequency=F0_BIC_REF, fwidth=0.08 * F0_BIC_REF)
    sources_asym = [
        {
            "kind": "plane",
            "type_name": "CurrentSource",
            "short_word": "Jy sheet (+z)",
            "label": "Normal-incidence Jy plane-wave sheet at z = −580 nm (+z, TE Ey)",
            "desc": "Normal-incidence Jy sheet at z = −580 nm (+z propagation, TE Ey, f₀ = 190.5 THz)",
            "pos_str": "z = −580 nm (full x ∈ [−340, +340] nm)",
            "polarization": "Ey (TE, +z)",
            "direction": 1,
            "h_bounds": (-0.58e-6, -0.58e-6),
            "v_bounds": (-0.5 * PERIOD_P, +0.5 * PERIOD_P),
            "pulse": {
                "f0": F0_BIC_REF,
                "fwidth": 0.08 * F0_BIC_REF,
                "t_peak": float(4.0 / (np.pi * 0.08 * F0_BIC_REF)),
                "t_end": float(wf_asym.end_time),
                "src_time": wf_asym,
            },
        },
        {
            "kind": "point",
            "type_name": "CurrentSource",
            "short_word": "Jy dipole",
            "label": "Off-center Jy ring-down dipole at (x₁, z) = (−127.5, −25) nm",
            "desc": "Off-center Jy ring-down dipole in Si bar 1 at (z, x) = (−25, −127.5) nm",
            "pos_str": "(x, z) = (−127.5, −25) nm (inside Si bar 1)",
            "polarization": "Jy (couples to C₂-odd BIC & quasi-BIC)",
            "h_bounds": (-0.025e-6, -0.025e-6),
            "v_bounds": (geom_asym["x1_c"], geom_asym["x1_c"]),
            "pulse": {
                "f0": F0_BIC_REF,
                "fwidth": 0.08 * F0_BIC_REF,
                "t_peak": float(4.0 / (np.pi * 0.08 * F0_BIC_REF)),
                "t_end": float(wf_asym.end_time),
                "src_time": wf_asym,
            },
        },
    ]
    monitors_asym = [
        {
            "kind": "plane",
            "type_name": "DiffractionMonitor",
            "short_word": "Refl. R₀₀",
            "label": "Reflected DiffractionMonitor (R₀₀) at z = −420 nm",
            "desc": "Reflected DiffractionMonitor (R₀₀ order) at z = −420 nm",
            "pos_str": "z = −420 nm (x ∈ [−340, +340] nm)",
            "freq_str": "λ ∈ [1520, 1578] nm (251 DFT frequencies)",
            "frequencies": np.linspace(c_light / 1.578e-6, c_light / 1.520e-6, 251),
            "h_bounds": (-0.42e-6, -0.42e-6),
            "v_bounds": (-0.5 * PERIOD_P, +0.5 * PERIOD_P),
        },
        {
            "kind": "plane",
            "type_name": "DiffractionMonitor",
            "short_word": "Tran. T₀₀",
            "label": "Transmitted DiffractionMonitor (T₀₀) at z = +580 nm",
            "desc": "Transmitted DiffractionMonitor (T₀₀ order) at z = +580 nm",
            "pos_str": "z = +580 nm (x ∈ [−340, +340] nm)",
            "freq_str": "λ ∈ [1520, 1578] nm (251 DFT frequencies)",
            "frequencies": np.linspace(c_light / 1.578e-6, c_light / 1.520e-6, 251),
            "h_bounds": (+0.58e-6, +0.58e-6),
            "v_bounds": (-0.5 * PERIOD_P, +0.5 * PERIOD_P),
        },
        {
            "kind": "probe",
            "type_name": "ResonanceMonitor",
            "short_word": "harminv Ey",
            "label": "ResonanceMonitor (harminv Ey) at (x₁, z) = (−127.5, +25) nm",
            "desc": "ResonanceMonitor (harminv Ey) in Si bar 1 at (z, x) = (+25, −127.5) nm",
            "pos_str": "(x, z) = (−127.5, +25) nm",
            "freq_str": "0.90 f₀ – 1.10 f₀ (38 optical cycles ring-down)",
            "h_bounds": (+0.025e-6, +0.025e-6),
            "v_bounds": (geom_asym["x1_c"], geom_asym["x1_c"]),
        },
    ]
    variations_asym = [
        {
            "name": "Asymmetry sweep (Tasks 0–1)",
            "details": "α = Δw / w₀ ∈ {0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20} (subpixel dual-cell averaging; Q_rad = Q₀ α⁻²)",
        },
        {
            "name": "Absorption loss saturation (Task 2)",
            "details": "Conductivity σ_Si inside bars giving Q_abs ∈ {5×10³, 1.5×10⁴, 5×10⁴, 10⁵} (Q_tot⁻¹ = Q_rad⁻¹ + Q_abs⁻¹)",
        },
        {
            "name": "Oblique Floquet angle sweep (Task 3)",
            "details": "Symmetric dimer (α = 0) with ±x Bloch BCs (k_x = k₀ sin θ_inc, θ_inc ∈ [1°, 12°]) — see 00a/00b_*_oblique.png",
        },
        {
            "name": "Resolution & CPML proximity study (Task 4)",
            "details": "n_x ∈ {16, 24, 32, 48, 64}, subpixel vs staircase quantization, and d_air ∈ [150, 850] nm",
        },
    ]

    res_asym = schematic.save_setup_figures(
        solver_asym,
        fig_dir,
        prefix="00",
        plane="zx",
        material_names=mat_names_asym,
        sources=sources_asym,
        monitors=monitors_asym,
        unit="nm",
        wavelength=LAMBDA_BIC_REF,
        min_feature=geom_asym["gap_inner"],
        variations=variations_asym,
    )

    # Adjust tag positions on 00a_setup_geometry.png for zero overlap
    fig_a, ax_a = schematic.plot_schematic(
        solver_asym,
        plane="zx",
        material_names=mat_names_asym,
        sources=sources_asym,
        monitors=monitors_asym,
        unit="nm",
        wavelength=LAMBDA_BIC_REF,
    )
    for txt in ax_a.texts:
        t_str = txt.get_text()
        if "1 · Jy sheet" in t_str:
            txt.set_ha("center")
            txt.set_position((-580.0, -260.0))
        elif "2 · Jy dipole" in t_str:
            txt.set_ha("right")
            txt.set_position((-115.0, -128.0))
        elif "3 · Refl." in t_str:
            txt.set_ha("center")
            txt.set_position((-420.0, +260.0))
        elif "4 · Tran." in t_str:
            txt.set_ha("center")
            txt.set_position((+580.0, +260.0))
        elif "5 · harminv" in t_str:
            txt.set_ha("left")
            txt.set_position((+115.0, -128.0))
    figstyle.save(fig_a, res_asym["geometry_figure"], close=True, auto_legend_below=False)

    # Re-render 00b_setup_mesh_and_excitation.png with zoom_bounds around the 85 nm inner gap & Si bars
    fig_b, axes_b = schematic.plot_mesh_and_excitation(
        solver_asym,
        plane="zx",
        material_names=mat_names_asym,
        sources=sources_asym,
        monitors=monitors_asym,
        unit="nm",
        mesh_every=2,
        zoom_bounds=(-0.14e-6, +0.14e-6, -0.24e-6, +0.24e-6),
        wavelength=LAMBDA_BIC_REF,
        min_feature=geom_asym["gap_inner"],
    )
    ax_mesh_b = axes_b[0]
    for ch_ax in getattr(ax_mesh_b, "child_axes", []):
        ch_ax.set_axes_locator(
            _TransformedBoundsLocator([0.66, 0.10, 0.32, 0.80], ax_mesh_b.transAxes)
        )
    figstyle.save(fig_b, res_asym["mesh_figure"], close=True, auto_legend_below=False)

    # ------------------------------------------------------------------
    # 2. Companion setup (_oblique): Symmetric dimer (alpha = 0) under oblique Floquet–Bloch launch
    # ------------------------------------------------------------------
    theta_obl_deg = 10.0
    kx_obl = (2.0 * np.pi / LAMBDA_BIC_REF) * float(np.sin(np.deg2rad(theta_obl_deg)))
    dz_obl = 0.019e-6
    lz_obl = 64 * dz_obl
    solver_obl, meta_obl = make_dimer_solver(
        alpha=0.0,
        nx=32,
        kx=kx_obl,
        z_max=0.5 * lz_obl,
        uniform_z=True,
        dz_uniform=dz_obl,
        n_pml=10,
    )
    geom_sym = meta_obl["geom"]
    bar1_sym = Box(
        center=(geom_sym["x1_c"], 0.5 * dx, 0.0),
        size=(geom_sym["w1"], dx, HEIGHT_H),
        material=EPS_SI,
    )
    bar1_sym.name = "Si bar 1 (w₁ = w₀ = 170.0 nm, h = 190 nm)"
    bar2_sym = Box(
        center=(geom_sym["x2_c"], 0.5 * dx, 0.0),
        size=(geom_sym["w2"], dx, HEIGHT_H),
        material=EPS_SI,
    )
    bar2_sym.name = "Si bar 2 (w₂ = w₀ = 170.0 nm, h = 190 nm)"
    structs_sym = [bar1_sym, bar2_sym]
    solver_obl._assigned_structures = tuple(structs_sym)

    mat_names_sym = {
        "background": "Air / vacuum background (ε_bg = 1.00, n = 1.00)",
        "region_0": "Si bar 1 (w₁ = w₀ = 170.0 nm, ε_Si = 12.11, n_Si = 3.48)",
        "region_1": "Si bar 2 (w₂ = w₀ = 170.0 nm, ε_Si = 12.11, n_Si = 3.48)",
    }
    sources_obl = [
        {
            "kind": "plane",
            "type_name": "FloquetCellSource",
            "short_word": "Bloch Jy",
            "label": "Oblique Floquet–Bloch Jy sheet at z = −320 nm (θ_inc = 10°, k_x = k₀ sin θ_inc)",
            "desc": "Oblique Floquet–Bloch Jy sheet at z = −320 nm (θ_inc = 10°, Δφ_x = k_x p)",
            "pos_str": "z = −320 nm (x ∈ [−340, +340] nm)",
            "polarization": "Ey (TE, θ_inc = 10°)",
            "direction": 1,
            "h_bounds": (-0.32e-6, -0.32e-6),
            "v_bounds": (-0.5 * PERIOD_P, +0.5 * PERIOD_P),
            "pulse": {
                "f0": F0_BIC_REF,
                "fwidth": 0.08 * F0_BIC_REF,
                "t_peak": float(4.0 / (np.pi * 0.08 * F0_BIC_REF)),
                "t_end": float(wf_asym.end_time),
                "src_time": wf_asym,
            },
        },
        {
            "kind": "point",
            "type_name": "CurrentSource",
            "short_word": "Jy dipole",
            "label": "Bloch ring-down Jy dipole at (x₁, z) = (−127.5, −25) nm",
            "desc": "Bloch ring-down Jy dipole in Si bar 1 at (z, x) = (−25, −127.5) nm",
            "pos_str": "(x, z) = (−127.5, −25) nm",
            "polarization": "Jy",
            "h_bounds": (-0.025e-6, -0.025e-6),
            "v_bounds": (geom_sym["x1_c"], geom_sym["x1_c"]),
            "pulse": {
                "f0": F0_BIC_REF,
                "fwidth": 0.08 * F0_BIC_REF,
                "t_peak": float(4.0 / (np.pi * 0.08 * F0_BIC_REF)),
                "t_end": float(wf_asym.end_time),
                "src_time": wf_asym,
            },
        },
    ]
    monitors_obl = [
        {
            "kind": "plane",
            "type_name": "DiffractionMonitor",
            "short_word": "Refl. R₀₀",
            "label": "Reflected Floquet DiffractionMonitor at z = −220 nm",
            "desc": "Reflected Floquet DiffractionMonitor (R₀₀(λ, θ)) at z = −220 nm",
            "pos_str": "z = −220 nm (x ∈ [−340, +340] nm)",
            "freq_str": "λ ∈ [1560, 1620] nm (AngleSweep resampling)",
            "frequencies": np.linspace(c_light / 1.620e-6, c_light / 1.560e-6, 121),
            "h_bounds": (-0.22e-6, -0.22e-6),
            "v_bounds": (-0.5 * PERIOD_P, +0.5 * PERIOD_P),
        },
        {
            "kind": "plane",
            "type_name": "DiffractionMonitor",
            "short_word": "Tran. T₀₀",
            "label": "Transmitted Floquet DiffractionMonitor at z = +220 nm",
            "desc": "Transmitted Floquet DiffractionMonitor (T₀₀(λ, θ)) at z = +220 nm",
            "pos_str": "z = +220 nm (x ∈ [−340, +340] nm)",
            "freq_str": "λ ∈ [1560, 1620] nm (AngleSweep resampling)",
            "frequencies": np.linspace(c_light / 1.620e-6, c_light / 1.560e-6, 121),
            "h_bounds": (+0.22e-6, +0.22e-6),
            "v_bounds": (-0.5 * PERIOD_P, +0.5 * PERIOD_P),
        },
        {
            "kind": "probe",
            "type_name": "ResonanceMonitor",
            "short_word": "harminv",
            "label": "Bloch ResonanceMonitor (harminv Ey) at (x₁, z) = (−127.5, +25) nm",
            "desc": "Bloch ResonanceMonitor (harminv Ey) at (z, x) = (+25, −127.5) nm",
            "pos_str": "(x, z) = (−127.5, +25) nm",
            "freq_str": "Q_rad(k_∥) ∝ sin⁻² θ_inc",
            "h_bounds": (+0.025e-6, +0.025e-6),
            "v_bounds": (geom_sym["x1_c"], geom_sym["x1_c"]),
        },
    ]

    res_obl = schematic.save_setup_figures(
        solver_obl,
        fig_dir,
        prefix="00",
        suffix="_oblique",
        plane="zx",
        material_names=mat_names_sym,
        sources=sources_obl,
        monitors=monitors_obl,
        unit="nm",
        wavelength=LAMBDA_BIC_REF,
        min_feature=geom_sym["gap_inner"],
        variations=[
            {
                "name": "Oblique angle & Bloch wavenumber sweep (Task 3)",
                "details": "θ_inc ∈ {1°, 2°, 3°, 5°, 7°, 10°} (harminv Q_rad(k_∥) ∝ k_∥⁻²) and AngleSweep θ_inc ∈ {0°, 6°, 9°, 12°}",
            },
        ],
    )
    fig_a_o, ax_a_o = schematic.plot_schematic(
        solver_obl,
        plane="zx",
        material_names=mat_names_sym,
        sources=sources_obl,
        monitors=monitors_obl,
        unit="nm",
        wavelength=LAMBDA_BIC_REF,
    )
    for txt in ax_a_o.texts:
        t_str = txt.get_text()
        if "1 · Bloch" in t_str:
            txt.set_ha("center")
            txt.set_position((-270.0, -265.0))
        elif "2 · Jy dipole" in t_str:
            txt.set_ha("center")
            txt.set_position((0.0, -10.0))
        elif "3 · Refl." in t_str:
            txt.set_ha("center")
            txt.set_position((-220.0, +265.0))
        elif "4 · Tran." in t_str:
            txt.set_ha("center")
            txt.set_position((+220.0, +265.0))
        elif "5 · harminv" in t_str:
            txt.set_ha("left")
            txt.set_position((+110.0, -128.0))
    figstyle.save(fig_a_o, res_obl["geometry_figure"], close=True, auto_legend_below=False)

    fig_b_o, axes_b_o = schematic.plot_mesh_and_excitation(
        solver_obl,
        plane="zx",
        material_names=mat_names_sym,
        sources=sources_obl,
        monitors=monitors_obl,
        unit="nm",
        mesh_every=2,
        zoom_bounds=(-0.14e-6, +0.14e-6, -0.24e-6, +0.24e-6),
        wavelength=LAMBDA_BIC_REF,
        min_feature=geom_sym["gap_inner"],
    )
    ax_mesh_bo = axes_b_o[0]
    for ch_ax in getattr(ax_mesh_bo, "child_axes", []):
        ch_ax.set_axes_locator(
            _TransformedBoundsLocator([1.06, 0.10, 0.56, 0.80], ax_mesh_bo.transAxes)
        )
    figstyle.save(fig_b_o, res_obl["mesh_figure"], close=True, auto_legend_below=False)


def plot_all(out_dir: Path, fig_dir: Path) -> None:
    """
    Generate all 8 publication-quality figures (`00..07`, including the animated GIF)
    from `out_dir/task_*.npz` into `fig_dir`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_dir = Path(out_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    d0 = np.load(out_dir / "task_00.npz")
    d1 = np.load(out_dir / "task_01.npz")
    d2 = np.load(out_dir / "task_02.npz")
    d3 = np.load(out_dir / "task_03.npz")
    d4 = np.load(out_dir / "task_04.npz")

    # ------------------------------------------------------------------
    # Figure 0: Setup Schematic, Symmetry Taxonomy & Graded Mesh View
    # ------------------------------------------------------------------
    _plot_setup_schematic_and_mesh(d0, fig_dir)

    # ------------------------------------------------------------------
    # Figure 1: Normal-Incidence Fano Spectra & Collapse of Fano Resonance
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    # Panel (a): Fano fan across alpha in [0, 0.04, 0.08, 0.14, 0.20] (analog of Koshelev Fig. 2c)
    axa = axes[0]
    fan_wls = d0["fan_wls_nm"]
    fan_alphas = d0["fan_alphas"]
    fan_T = d0["fan_T"]
    colors_fan = [
        PALETTE["black"],
        PALETTE["blue"],
        PALETTE["green"],
        PALETTE["orange"],
        PALETTE["vermilion"],
    ]
    styles_fan = ["--", "-", "-.", ":", "-"]
    for i_f, (a_f, col, ls) in enumerate(zip(fan_alphas, colors_fan, styles_fan)):
        lbl = (
            r"$\alpha = 0$ (BIC: Fano collapse)"
            if a_f == 0.0
            else rf"$\alpha = {a_f:.2f}$ ($\alpha_{{\mathrm{{half}}}}={0.5*a_f:.2f}$)"
        )
        axa.plot(fan_wls, fan_T[i_f], ls=ls, color=col, lw=1.9, label=lbl)
    axa.axvline(float(d0["lambda_bic_rcwa_nm"]), color=PALETTE["grey"], ls=":", lw=1.2)
    axa.text(
        float(d0["lambda_bic_rcwa_nm"]) + 0.15,
        0.12,
        rf"$\lambda_{{\mathrm{{BIC}}}} = {float(d0['lambda_bic_rcwa_nm']):.2f}\;\mathrm{{nm}}$",
        fontsize=8.6,
        color=PALETTE["black"],
    )
    axa.set_xlim(fan_wls[0], fan_wls[-1])
    axa.set_ylim(-0.03, 1.05)
    axa.set_xlabel(r"Wavelength $\lambda$ [nm]")
    axa.set_ylabel(r"Zeroth-order transmission $T_{00}(\lambda)$")
    axa.set_title("(a) Quasi-BIC Fano Fan & Collapse at $\\alpha = 0$")
    axa.grid(True, alpha=0.28)
    axa.legend(loc="lower left", framealpha=0.92)

    # Panel (b): Direct Wakis 2D FIT DiffractionMonitor T, R, R+T & Eq. (1) Fano Fit vs harminv
    axb = axes[1]
    fwls = d0["fano_wls_nm"]
    axb.plot(
        fwls,
        d0["fano_T_sym_alpha0"],
        "--",
        color=PALETTE["grey"],
        lw=1.7,
        label=r"Wakis FIT $T(\lambda)$ at $\alpha=0$ (no dip)",
    )
    axb.plot(
        fwls[::3],
        d0["fano_T_asym"][::3],
        "o",
        color=PALETTE["blue"],
        ms=4.5,
        label=r"Wakis FIT `DiffractionMonitor` $T_{00}$ ($\alpha=0.48$)",
    )
    axb.plot(
        fwls,
        d0["fano_T_fit"],
        "-",
        color=PALETTE["vermilion"],
        lw=2.0,
        label=rf"Koshelev Eq. (1) Fit ($Q_{{\mathrm{{Fano}}}}={float(d0['fano_Q_fit']):.1f}, q={float(d0['fano_q_param']):.2f}$)",
    )
    axb.plot(
        fwls[::4],
        d0["fano_R_asym"][::4],
        "s",
        color=PALETTE["green"],
        ms=4.0,
        alpha=0.8,
        label=r"Wakis FIT `DiffractionMonitor` $R_{00}$",
    )
    axb.plot(
        fwls,
        d0["fano_R_asym"] + d0["fano_T_asym"],
        ":",
        color=PALETTE["black"],
        lw=1.4,
        label=r"Energy balance $R_{00} + T_{00} \approx 1$",
    )
    axb.axvline(
        float(d0["fano_lambda0_harminv_nm"]),
        color=PALETTE["purple"],
        ls="-.",
        lw=1.5,
        label=rf"`harminv` $\lambda_0={float(d0['fano_lambda0_harminv_nm']):.2f}\;\mathrm{{nm}}, Q={float(d0['fano_Q_harminv']):.1f}$",
    )
    axb.set_xlim(fwls[0], fwls[-1])
    axb.set_ylim(-0.03, 1.08)
    axb.set_xlabel(r"Wavelength $\lambda$ [nm]")
    axb.set_ylabel(r"Transmittance $T_{00}$ / Reflectance $R_{00}$")
    axb.set_title("(b) Wakis FIT `DiffractionMonitor` vs `harminv`")
    axb.grid(True, alpha=0.28)
    axb.legend(loc="center left", framealpha=0.92, fontsize=8.5)

    # Panel (c): Universal Fano master curve vs normalized detuning Omega = 2(f - f0)/Delta_f
    axc = axes[2]
    omega_dense = np.linspace(-8.0, 8.0, 401)
    q_mean = float(np.mean(d0["q_fano_rcwa"]))
    t_master = (1.0 / (1.0 + q_mean**2)) * ((q_mean + omega_dense) ** 2 / (1.0 + omega_dense**2))
    axc.plot(
        omega_dense,
        t_master,
        "-",
        color=PALETTE["black"],
        lw=2.2,
        label=rf"Koshelev Eq. (1) Master Curve ($q = {q_mean:.3f}$)",
    )
    markers_u = ["o", "s", "^", "D"]
    cols_u = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["vermilion"]]
    for idx_u, mk, col in zip((0, 2, 4, 6), markers_u, cols_u):
        a_u = float(d0["alphas"][idx_u])
        wls_u = d0["local_wls_nm"][idx_u]
        freqs_u = 299792458.0 / (wls_u * 1e-9)
        f0_u = 299792458.0 / (float(d0["lambda0_rcwa_nm"][idx_u]) * 1e-9)
        df_u = f0_u / float(d0["Q_rcwa"][idx_u])
        om_u = 2.0 * (freqs_u - f0_u) / df_u
        axc.plot(
            om_u[::8],
            d0["local_T"][idx_u][::8],
            mk,
            color=col,
            ms=5.0,
            label=rf"$\alpha={a_u:.2f}$ ($Q={float(d0['Q_rcwa'][idx_u]):.0f}$)",
        )
    axc.set_xlim(-7.5, 7.5)
    axc.set_ylim(-0.03, 1.05)
    axc.set_xlabel(r"Normalized frequency detuning $\Omega = 2(\omega - \omega_0)/\gamma$")
    axc.set_ylabel(r"Transmission $T(\Omega)$")
    axc.set_title(r"(c) Universal Fano Collapse in $\Omega = 2(\omega - \omega_0)/\gamma$")
    axc.grid(True, alpha=0.28)
    axc.legend(loc="lower right", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(fig_dir / "01_fano_transmission_spectra.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Universal Inverse-Square Scaling Q_rad ~ alpha^(-2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))

    alphas = d0["alphas"]
    alphas_half = d0["alphas_half"]
    q_fit = d0["Q_fit"]
    q_rcwa = d0["Q_rcwa"]
    scaling_fits = fit_q_scaling_laws(alphas, q_fit)
    slope_fit = scaling_fits["slope_free"]
    q0_free = scaling_fits["q0_free"]
    q0_imp = scaling_fits["q0_imposed"]
    q0_half_imp = scaling_fits["q0_half_imposed"]

    # Panel (a): Log-log Q_rad vs alpha (imposed slope -2 vs free fit; paper in separate inset)
    axa = axes[0]
    a_line = np.logspace(np.log10(0.008), np.log10(0.25), 200)
    axa.loglog(
        a_line,
        q0_imp * (a_line ** (-2.0)),
        "-",
        color=PALETTE["black"],
        lw=2.0,
        label=rf"fit $Q_0 \alpha^{{-2}}$ (slope $-2$ imposed, $Q_0={q0_imp:.2f}$)",
    )
    axa.loglog(
        a_line,
        q0_free * (a_line**slope_fit),
        "--",
        color=PALETTE["blue"],
        lw=1.5,
        label=rf"free fit $Q_0 \alpha^s$ ($s={slope_fit:.4f}, Q_0={q0_free:.2f}$)",
    )
    axa.loglog(
        alphas,
        q_fit,
        "o",
        color=PALETTE["blue"],
        ms=6.2,
        label=r"Wakis 2D FIT `harminv` ($\lambda\approx 1574\;\mathrm{nm}$)",
    )
    axa.loglog(
        alphas,
        q_rcwa,
        "s--",
        color=PALETTE["vermilion"],
        ms=5.2,
        lw=1.3,
        label=rf"1D RCWA Fano Eq. (1) ($s={float(d0['log_slope_rcwa']):.4f}$)",
    )
    axa.loglog(
        alphas_half,
        q_fit,
        "^:",
        color=PALETTE["green"],
        ms=5.5,
        lw=1.3,
        label=rf"Wakis vs $\alpha_{{\mathrm{{half}}}}=\alpha/2$ ($Q_0={q0_half_imp:.2f}$)",
    )
    axa.set_xlim(0.007, 0.28)
    axa.set_ylim(6e2, 4e5)
    axa.set_xlabel(r"Asymmetry parameter $\alpha = \Delta w / w_0$ (or $\alpha_{\mathrm{half}}=\alpha/2$)")
    axa.set_ylabel(r"Radiative quality factor $Q_{\mathrm{rad}}$")
    axa.set_title(r"(a) Universal Scaling $Q_{\mathrm{rad}}(\alpha) \propto \alpha^{-2}$")
    axa.grid(True, which="both", alpha=0.28)
    axa.legend(loc="lower left", framealpha=0.92, fontsize=7.6)

    # Separate small inset panel for Koshelev et al. (3D Si bar pairs, different geometry)
    ax_ins = axa.inset_axes([0.55, 0.56, 0.42, 0.37])
    k_alphas = np.array([np.sin(np.deg2rad(3.0)), np.sin(np.deg2rad(10.0))])
    k_qs = np.array([5000.0, 450.0])
    a_ins = np.logspace(np.log10(0.035), np.log10(0.25), 60)
    ax_ins.loglog(
        a_ins,
        13.7 * (a_ins ** (-2.0)),
        "-",
        color=PALETTE["purple"],
        lw=1.3,
        label=r"$Q_0\alpha^{-2}$ ($Q_0\approx 13.7$)",
    )
    ax_ins.loglog(
        k_alphas,
        k_qs,
        "D",
        color=PALETTE["purple"],
        ms=5.0,
        label=r"3D Si bars ($\theta=3^\circ,10^\circ$)",
    )
    ax_ins.set_title("paper (different geometry)", fontsize=7.4, pad=2.0)
    ax_ins.set_xlabel(r"$\alpha = \sin\theta$", fontsize=6.8, labelpad=1.0)
    ax_ins.set_ylabel(r"$Q_{\mathrm{rad}}$", fontsize=6.8, labelpad=1.0)
    ax_ins.set_xticks([0.05, 0.10, 0.20])
    ax_ins.set_xticklabels(["0.05", "0.10", "0.20"])
    ax_ins.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_ins.tick_params(labelsize=6.5)
    ax_ins.grid(True, which="both", alpha=0.25)

    # Panel (b): Compensated plateau Q_rad * alpha^2 and quadratic wavelength shift
    axb = axes[1]
    axb.plot(
        alphas,
        d0["Q_alpha2_fit"],
        "o-",
        color=PALETTE["blue"],
        lw=2.0,
        ms=6.0,
        label=r"Wakis 2D FIT `harminv`: $Q_{\mathrm{rad}}\alpha^2$",
    )
    axb.plot(
        alphas,
        d0["Q_alpha2_rcwa"],
        "s--",
        color=PALETTE["vermilion"],
        lw=1.8,
        ms=5.5,
        label=r"1D RCWA Fano Eq. (1): $Q_{\mathrm{Fano}}\alpha^2$",
    )
    axb.axhline(
        q0_imp,
        color=PALETTE["black"],
        ls=":",
        lw=1.5,
        label=rf"Imposed-slope $Q_0 = {q0_imp:.2f}$ ($\pm 0.3\%$ over $100\times$ in $Q$)",
    )
    axb.set_ylim(43.5, 50.5)
    axb.set_xlabel(r"Asymmetry parameter $\alpha = (w_1 - w_2) / w_0$")
    axb.set_ylabel(r"Compensated invariants $Q_{\mathrm{rad}}(\alpha)\,\alpha^2$")
    axb.set_title(r"(b) Invariant Plateau $Q_{\mathrm{rad}}\alpha^2 = Q_0$")
    axb.grid(True, alpha=0.28)
    axb.legend(loc="upper left", framealpha=0.92)

    # Panel (c): Ring-down envelopes (cycle peaks only, no carrier oscillation block) + fitted exp(-omega*t/2Q)
    axc = axes[2]
    t_bic = d0["t_ring_bic_fs"]
    t_bic_env, env_bic = extract_ringdown_envelope(t_bic, d0["sig_ring_bic"], window_fs=22.0)
    f_bic = 299792458.0 / (float(d0["lambda_bic_fit_nm"]) * 1e-9)
    q_bic = float(d0["Q_bic_fit"])
    axc.semilogy(
        t_bic_env,
        env_bic,
        "o",
        color=PALETTE["black"],
        ms=4.0,
        alpha=0.85,
        label=rf"$\alpha = 0$ peak env. ($Q = {q_bic:.1e}$)",
    )
    axc.semilogy(
        t_bic,
        np.exp(-np.pi * f_bic * (t_bic * 1e-15) / q_bic),
        "-",
        color=PALETTE["black"],
        lw=1.8,
        label=r"$\alpha = 0$ fit $\exp(-\omega_0 t / 2Q)$",
    )
    for i_tr, col, ls, mk in [
        (1, PALETTE["blue"], "--", "s"),
        (4, PALETTE["green"], "-.", "^"),
        (6, PALETTE["vermilion"], "-", "D"),
    ]:
        a_tr = float(alphas[i_tr])
        q_tr = float(q_fit[i_tr])
        f_tr = float(d0["f0_fit_hz"][i_tr])
        t_tr = d0["t_ring_traces_fs"][i_tr]
        t_tr_env, env_tr = extract_ringdown_envelope(
            t_tr, d0["sig_ring_traces"][i_tr], window_fs=22.0
        )
        env_analytic = np.exp(-np.pi * f_tr * (t_tr * 1e-15) / q_tr)
        axc.semilogy(
            t_tr_env,
            env_tr,
            mk,
            color=col,
            ms=4.0,
            alpha=0.85,
            label=rf"$\alpha = {a_tr:.2f}$ peak env.",
        )
        axc.semilogy(
            t_tr,
            env_analytic,
            ls=ls,
            color=col,
            lw=1.9,
            label=rf"$\alpha = {a_tr:.2f}$ fit $\exp(-\omega_0 t / 2Q)$ ($Q = {q_tr:.0f}$)",
        )
    axc.set_ylim(0.78, 1.03)
    axc.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    axc.set_yticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"])
    axc.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axc.set_xlabel(r"Ring-down time $t - t_{\mathrm{off}}$ [fs]")
    axc.set_ylabel(r"Normalized peak envelope $|E_y(-d/2, t)|_{\mathrm{env}} / E_{\max}$ (log scale)")
    axc.set_title(r"(c) Ring-Down Envelopes & $\exp(-\omega_0 t / 2Q)$ Fits")
    axc.grid(True, which="both", alpha=0.28)
    axc.legend(loc="lower left", framealpha=0.92, fontsize=6.9, ncol=2)

    fig.tight_layout()
    fig.savefig(fig_dir / "02_q_vs_alpha_scaling.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Near-Field & Far-Field Maps of BIC (alpha=0) vs Quasi-BIC (alpha=0.24)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    x_um = d1["x_nodes_um"]
    z_um = d1["z_nodes_um"]
    ey_bic = d1["ey_bic"]
    ey_qbic = d1["ey_qbic"]
    ey_bic_norm = ey_bic / max(float(np.max(np.abs(ey_bic))), 1e-30)
    ey_qbic_norm = ey_qbic / max(float(np.max(np.abs(ey_qbic))), 1e-30)
    vmax_field = 0.35  # saturate core slightly to make radiating plane wave clearly visible

    for idx_p, (ax_f, field_2d, a_val, title_f) in enumerate(
        [
            (
                axes[0],
                ey_bic_norm,
                0.0,
                rf"(a) True BIC ($\alpha = 0, Q = {float(d1['Q_bic']):.1e}$): Trapped Mode",
            ),
            (
                axes[1],
                ey_qbic_norm,
                float(d1["alpha_qbic"]),
                rf"(b) Quasi-BIC ($\alpha = {float(d1['alpha_qbic']):.2f}, Q = {float(d1['Q_qbic']):.0f}$): Leaky Radiation",
            ),
        ]
    ):
        im_f = ax_f.pcolormesh(
            x_um * 1e3,
            z_um,
            field_2d.T,
            cmap="RdBu_r",
            vmin=-vmax_field,
            vmax=+vmax_field,
            shading="auto",
        )
        cbar_f = fig.colorbar(im_f, ax=ax_f, pad=0.02)
        cbar_f.set_label(r"Normalized $E_y(x, z) / E_{\max}$", fontsize=8.8)

        geom_p = dimer_bar_geometry(alpha=a_val)
        for x_m, w_m, b_lbl in [
            (geom_p["x1_min"], geom_p["w1"], "Si 1"),
            (geom_p["x2_min"], geom_p["w2"], "Si 2"),
        ]:
            ax_f.add_patch(
                Rectangle(
                    (x_m * 1e9, -0.5 * HEIGHT_H * 1e6),
                    w_m * 1e9,
                    HEIGHT_H * 1e6,
                    fill=False,
                    edgecolor="black",
                    lw=1.6,
                )
            )
            ax_f.text(
                (x_m + 0.5 * w_m) * 1e9,
                0.0,
                b_lbl,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.82, edgecolor="black"),
            )
        # Label materials & boundaries explicitly on every field map
        ax_f.axhline(-0.852, color="#C51B8A", ls="--", lw=1.3)
        ax_f.axhline(+0.852, color="#C51B8A", ls="--", lw=1.3)
        ax_f.text(
            0.0,
            1.06,
            r"$z^+$ CPML (10 cells) | Air ($\varepsilon=1$)",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#7A0177",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.88, edgecolor="#C51B8A"),
        )
        ax_f.text(
            0.0,
            -1.06,
            r"$z^-$ CPML (10 cells) | Si ($\varepsilon=12.11$)",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#7A0177",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.88, edgecolor="#C51B8A"),
        )
        ax_f.text(
            -325.0,
            0.48,
            r"Periodic $x^\mp$",
            rotation=90,
            ha="left",
            va="center",
            fontsize=8.5,
            color=PALETTE["green"],
            fontweight="bold",
        )
        ax_f.set_xlim(-340.0, 340.0)
        ax_f.set_ylim(-1.25, 1.25)
        ax_f.set_xlabel(r"$x$ [nm]")
        ax_f.set_ylabel(r"$z$ [$\mu\mathrm{m}$]")
        ax_f.set_title(title_f)

    # Panel (c): Vertical profile |Ey(x=-d/2, z)| showing evanescent decay vs radiative floor
    axc = axes[2]
    ix_l = int(np.argmin(np.abs(x_um * 1e-6 - (-0.5 * DIMER_D))))
    prof_bic = np.abs(ey_bic_norm[ix_l, :])
    prof_qbic = np.abs(ey_qbic_norm[ix_l, :])
    z_pos_mask = (z_um >= 0.0) & (z_um <= 0.85)
    z_pos = z_um[z_pos_mask]
    kappa1 = float(np.sqrt((2.0 * np.pi / PERIOD_P) ** 2 - (2.0 * np.pi / LAMBDA_BIC_REF) ** 2))
    evan_ref = 0.85 * np.exp(-kappa1 * (z_pos - 0.095) * 1e-6)

    axc.semilogy(
        z_um,
        np.maximum(prof_bic, 1e-6),
        "-",
        color=PALETTE["blue"],
        lw=2.0,
        label=r"True BIC ($\alpha=0$): purely evanescent",
    )
    axc.semilogy(
        z_um,
        np.maximum(prof_qbic, 1e-6),
        "-",
        color=PALETTE["vermilion"],
        lw=2.0,
        label=rf"Quasi-BIC ($\alpha={float(d1['alpha_qbic']):.2f}$): radiating $(0,0)$ floor",
    )
    axc.semilogy(
        z_pos,
        evan_ref,
        "--",
        color=PALETTE["black"],
        lw=1.6,
        label=rf"Rayleigh $m=\pm 1$ decay $e^{{-\kappa_1 |z|}}$ ($\kappa_1={kappa1*1e-6:.2f}\;\mu\mathrm{{m}}^{{-1}}$)",
    )
    axc.axvspan(-0.095, 0.095, color=PALETTE["sky"], alpha=0.22, label=r"Si bars ($|z|\leq 95\;\mathrm{nm}$)")
    axc.set_xlim(-1.05, 1.05)
    axc.set_ylim(2e-5, 2.0)
    axc.set_xlabel(r"Vertical coordinate $z$ [$\mu\mathrm{m}$]")
    axc.set_ylabel(r"Vertical field profile $|E_y(x=-d/2, z)| / E_{\max}$")
    axc.set_title("(c) Evanescent Confinement vs Radiating Floor")
    axc.grid(True, which="both", alpha=0.28)
    axc.legend(loc="lower center", framealpha=0.92, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "03_bic_vs_quasi_bic_fields.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Animated GIF Comparing BIC (alpha=0) vs Quasi-BIC (alpha=0.24)
    # ------------------------------------------------------------------
    frames_bic = np.asarray(d1["frames_ey_bic"], dtype=float)
    frames_qbic = np.asarray(d1["frames_ey_qbic"], dtype=float)
    frames_t_fs = np.asarray(d1["frames_t_fs"], dtype=float)
    s_bic = max(float(np.max(np.abs(frames_bic))), 1e-30)
    s_qbic = max(float(np.max(np.abs(frames_qbic))), 1e-30)
    frames_bic_n = frames_bic / s_bic
    frames_qbic_n = frames_qbic / s_qbic
    n_f = len(frames_t_fs)
    vmax_gif = 0.45

    fig_g, axes_g = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    ims_g = []
    titles_g = []
    for idx_g, (ax_g, fr0, a_g, name_g) in enumerate(
        [
            (axes_g[0], frames_bic_n[0], 0.0, "True BIC ($\\alpha = 0$, No Far-Field Radiation)"),
            (
                axes_g[1],
                frames_qbic_n[0],
                float(d1["alpha_qbic"]),
                f"Quasi-BIC ($\\alpha = {float(d1['alpha_qbic']):.2f}$, Radiating Plane Waves)",
            ),
        ]
    ):
        im_g = ax_g.pcolormesh(
            x_um * 1e3,
            z_um,
            fr0.T,
            cmap="RdBu_r",
            vmin=-vmax_gif,
            vmax=+vmax_gif,
            shading="auto",
        )
        ims_g.append(im_g)
        geom_g = dimer_bar_geometry(alpha=a_g)
        for x_m, w_m, b_lbl in [
            (geom_g["x1_min"], geom_g["w1"], "Si 1"),
            (geom_g["x2_min"], geom_g["w2"], "Si 2"),
        ]:
            ax_g.add_patch(
                Rectangle(
                    (x_m * 1e9, -0.5 * HEIGHT_H * 1e6),
                    w_m * 1e9,
                    HEIGHT_H * 1e6,
                    fill=False,
                    edgecolor="black",
                    lw=1.5,
                )
            )
            ax_g.text(
                (x_m + 0.5 * w_m) * 1e9,
                0.0,
                b_lbl,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", alpha=0.8, edgecolor="black"),
            )
        ax_g.axhline(-0.852, color="#C51B8A", ls="--", lw=1.2)
        ax_g.axhline(+0.852, color="#C51B8A", ls="--", lw=1.2)
        ax_g.text(
            0.0,
            1.05,
            r"$z^+$ CPML | Air ($\varepsilon=1$)",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#7A0177",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.88, edgecolor="#C51B8A"),
        )
        ax_g.text(
            0.0,
            -1.05,
            r"$z^-$ CPML | Si ($\varepsilon=12.11$)",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#7A0177",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.88, edgecolor="#C51B8A"),
        )
        ax_g.text(
            -325.0,
            0.48,
            r"Periodic $x^\mp$",
            rotation=90,
            ha="left",
            va="center",
            fontsize=8.5,
            color=PALETTE["green"],
            fontweight="bold",
        )
        ax_g.set_xlim(-340.0, 340.0)
        ax_g.set_ylim(-1.25, 1.25)
        ax_g.set_xlabel(r"$x$ [nm]")
        if idx_g == 0:
            ax_g.set_ylabel(r"$z$ [$\mu\mathrm{m}$]")
        t_obj = ax_g.set_title(f"{name_g}\n$t = {frames_t_fs[0]:.0f}$ fs")
        titles_g.append((t_obj, name_g))

    fig_g.colorbar(ims_g[1], ax=axes_g.ravel().tolist(), pad=0.02, label=r"$E_y(x, z, t) / E_{\max}$")

    def _update_gif(fi: int):
        ims_g[0].set_array(frames_bic_n[fi].T.ravel())
        ims_g[1].set_array(frames_qbic_n[fi].T.ravel())
        for t_obj, name_g in titles_g:
            t_obj.set_text(f"{name_g}\n$t = {frames_t_fs[fi]:.0f}$ fs (frame {fi + 1}/{n_f})")
        return (ims_g[0], ims_g[1], titles_g[0][0], titles_g[1][0])

    ani = animation.FuncAnimation(fig_g, _update_gif, frames=n_f, interval=100, blit=False)
    ani.save(fig_dir / "04_bic_vs_quasi_bic_ringdown.gif", writer="pillow", fps=10)
    plt.close(fig_g)

    # ------------------------------------------------------------------
    # Figure 5: Material/Fabrication Loss Q Saturation & Bare Q_rad Recovery
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    a_abs = d2["alphas"]
    q_rad_lossless = d2["Q_rad_lossless"]
    q_abs_targets = d2["q_abs_targets"]
    q_tot_sim = d2["Q_tot_sim"]
    q_tot_pred = d2["Q_tot_pred"]
    q_rad_rec = d2["Q_rad_recovered"]

    axa = axes[0]
    a_dense = np.logspace(np.log10(0.008), np.log10(0.25), 200)
    q_rad_dense = float(d0["Q0_alpha_fit"]) * (a_dense ** (-2.0))
    axa.loglog(
        a_dense,
        q_rad_dense,
        "--",
        color=PALETTE["black"],
        lw=1.8,
        label=r"Lossless $Q_{\mathrm{rad}}(\alpha) = Q_0 \alpha^{-2}$ ($\sigma = 0$)",
    )
    axa.loglog(
        a_abs,
        q_rad_lossless,
        "o",
        color=PALETTE["black"],
        ms=5.8,
        label=r"Wakis 2D FIT Lossless ($\sigma = 0$)",
    )
    cols_abs = [PALETTE["blue"], PALETTE["green"], PALETTE["vermilion"]]
    mks_abs = ["s", "^", "D"]
    for iq, (q_t, col, mk) in enumerate(zip(q_abs_targets, cols_abs, mks_abs)):
        q0_m = float(d2["Q_tot_alpha0"][iq])
        q_curve = 1.0 / (1.0 / q_rad_dense + 1.0 / q0_m)
        axa.loglog(
            a_dense,
            q_curve,
            "-",
            color=col,
            lw=1.8,
            label=rf"Theory $Q_{{\mathrm{{tot}}}}(\alpha)$ for $Q_{{\mathrm{{abs}}}} = {q_t:.0e}$",
        )
        axa.loglog(
            a_abs,
            q_tot_sim[iq],
            mk,
            color=col,
            ms=6.0,
            label=rf"Wakis FIT `sigma` ($Q_{{\mathrm{{tot}}}}(0) = {q0_m:.0f}$)",
        )
    axa.set_xlabel(r"Asymmetry parameter $\alpha = \Delta w / w_0$")
    axa.set_ylabel(r"Total quality factor $Q_{\mathrm{tot}}(\alpha)$")
    axa.set_title(r"(a) Absorption Saturation $Q_{\mathrm{tot}}^{-1} = Q_{\mathrm{rad}}^{-1} + Q_{\mathrm{abs}}^{-1}$ (Koshelev Fig. 4)")
    axa.grid(True, which="both", alpha=0.28)
    axa.legend(loc="lower left", framealpha=0.92, fontsize=8.5)

    axb = axes[1]
    axb.loglog(
        a_dense,
        q_rad_dense,
        "-",
        color=PALETTE["black"],
        lw=2.0,
        label=r"Lossless $Q_{\mathrm{rad}}(\alpha) = Q_0 \alpha^{-2}$",
    )
    for iq, (q_t, col, mk) in enumerate(zip(q_abs_targets, cols_abs, mks_abs)):
        axb.loglog(
            a_abs,
            q_rad_rec[iq],
            mk,
            color=col,
            ms=6.2,
            label=rf"Recovered $Q_{{\mathrm{{rad,rec}}}}$ from $Q_{{\mathrm{{abs}}}}={q_t:.0e}$",
        )
    axb.set_xlabel(r"Asymmetry parameter $\alpha = \Delta w / w_0$")
    axb.set_ylabel(r"Recovered radiative $Q_{\mathrm{rad,rec}} = [Q_{\mathrm{tot}}(\alpha)^{-1} - Q_{\mathrm{tot}}(0)^{-1}]^{-1}$")
    axb.set_title(r"(b) Bare Radiative Damping Recovery $\gamma_{\mathrm{rad}} = \gamma_{\mathrm{tot}}(\alpha) - \gamma_{\mathrm{tot}}(0)$")
    axb.grid(True, which="both", alpha=0.28)
    axb.legend(loc="lower left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(fig_dir / "05_absorption_q_saturation.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 6: Oblique-Incidence Floquet Dispersion Q(k_parallel) ~ k_parallel^(-2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    # Panel (a): wakis.floquet.AngleSweep constant-angle resampled T(lambda, theta)
    axa = axes[0]
    wls_sw = d3["wls_sweep_nm"]
    ang_sw = d3["angles_sweep_deg"]
    t_fl = d3["T_floquet"]
    cols_sw = [PALETTE["black"], PALETTE["blue"], PALETTE["green"], PALETTE["vermilion"]]
    lss_sw = ["--", "-", "-.", ":"]
    for ia, (deg, col, ls) in enumerate(zip(ang_sw, cols_sw, lss_sw)):
        lbl = (
            r"$\theta_{\mathrm{inc}} = 0^\circ$ (BIC dark state)"
            if deg == 0.0
            else rf"$\theta_{{\mathrm{{inc}}}} = {deg:.0f}^\circ$ (`AngleSweep.resample`)"
        )
        axa.plot(wls_sw, t_fl[ia], ls=ls, color=col, lw=1.9, label=lbl)
    axa.set_xlim(wls_sw[0], wls_sw[-1])
    axa.set_ylim(0.10, 0.70)
    axa.set_xlabel(r"Wavelength $\lambda$ [nm]")
    axa.set_ylabel(r"Resampled transmittance $T_{00}(\lambda, \theta_{\mathrm{inc}})$")
    axa.set_title(r"(a) `wakis.floquet.AngleSweep` at $\alpha = 0$")
    axa.grid(True, alpha=0.28)
    axa.legend(loc="lower left", framealpha=0.92)

    # Panel (b): RCWA oblique Fano fan at alpha = 0
    axb = axes[1]
    wls_ro = d3["wls_rcwa_obl_nm"]
    ang_ro = d3["rcwa_obl_angles_deg"]
    t_ro = d3["T_rcwa_obl"]
    cols_ro = [
        PALETTE["black"],
        PALETTE["blue"],
        PALETTE["green"],
        PALETTE["orange"],
        PALETTE["vermilion"],
    ]
    lss_ro = ["--", "-", "-.", ":", "-"]
    for ia, (deg, col, ls) in enumerate(zip(ang_ro, cols_ro, lss_ro)):
        lbl = (
            r"$\theta_{\mathrm{inc}} = 0^\circ$ (BIC collapse)"
            if deg == 0.0
            else rf"$\theta_{{\mathrm{{inc}}}} = {deg:.0f}^\circ$ ($\sin\theta = {np.sin(np.deg2rad(deg)):.3f}$)"
        )
        axb.plot(wls_ro, t_ro[ia], ls=ls, color=col, lw=1.8, label=lbl)
    axb.set_xlim(wls_ro[0], wls_ro[-1])
    axb.set_ylim(-0.03, 1.05)
    axb.set_xlabel(r"Wavelength $\lambda$ [nm]")
    axb.set_ylabel(r"Oblique transmittance $T_{00}(\lambda, \theta_{\mathrm{inc}})$")
    axb.set_title(r"(b) Oblique-Incidence Fano Fan ($\alpha = 0$)")
    axb.grid(True, alpha=0.28)
    axb.legend(loc="lower left", framealpha=0.92)

    # Panel (c): Log-log Q_rad(k_parallel) vs sin(theta_inc) = k_parallel / k_0
    axc = axes[2]
    sin_th = d3["sin_thetas"]
    q_bl = d3["Q_bloch"]
    slope_th = float(d3["log_slope_sin_theta"])
    q0_th = float(d3["Q0_sin_theta"])
    s_line = np.logspace(np.log10(0.015), np.log10(0.20), 200)
    axc.loglog(
        s_line,
        q0_th * (s_line ** (-2.0)),
        "-",
        color=PALETTE["black"],
        lw=2.0,
        label=rf"Theory: $Q_{{0,\theta}} \sin^{{-2}}\theta$ ($Q_{{0,\theta}}={q0_th:.2f}$)",
    )
    axc.loglog(
        sin_th,
        q_bl,
        "o",
        color=PALETTE["blue"],
        ms=6.5,
        label=rf"Wakis Bloch `harminv` (slope $= {slope_th:.4f}$)",
    )
    axc.set_xlabel(r"Normalized in-plane wavenumber $\sin\theta_{\mathrm{inc}} = k_\parallel / k_0$")
    axc.set_ylabel(r"Radiative quality factor $Q_{\mathrm{rad}}(k_\parallel)$ at $\alpha = 0$")
    axc.set_title(r"(c) Angular Dispersion $Q_{\mathrm{rad}}(k_\parallel) \propto k_\parallel^{-2}$")
    axc.grid(True, which="both", alpha=0.28)
    axc.legend(loc="lower left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(fig_dir / "06_oblique_floquet_dispersion.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 7: Resolution Convergence, Staircase Quantization & CPML Proximity
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    # Panel (a): Mesh convergence vs cpw_Si
    axa = axes[0]
    cpw_arr = d4["cpw_si"]
    lam_bic_conv = d4["lambda_bic_nm"]
    qa2_conv = d4["Q_alpha2_a10_subpixel"]
    lam_rcwa_ref = float(d0["lambda_bic_rcwa_nm"])
    qa2_rcwa_ref = float(d0["Q_alpha2_rcwa"][4])  # alpha = 0.10

    rel_err_lam = np.abs(lam_bic_conv - lam_rcwa_ref) / lam_rcwa_ref * 100.0
    rel_err_qa2 = np.abs(qa2_conv - qa2_rcwa_ref) / qa2_rcwa_ref * 100.0
    axa.semilogy(
        cpw_arr,
        rel_err_lam,
        "o-",
        color=PALETTE["blue"],
        lw=2.0,
        ms=6.5,
        label=r"$|\lambda_{\mathrm{BIC,FIT}} - \lambda_{\mathrm{BIC,RCWA}}| / \lambda_{\mathrm{BIC}}$ [%]",
    )
    axa.semilogy(
        cpw_arr,
        rel_err_qa2,
        "s--",
        color=PALETTE["vermilion"],
        lw=2.0,
        ms=6.0,
        label=r"$|(Q\alpha^2)_{\mathrm{FIT}} - (Q\alpha^2)_{\mathrm{RCWA}}| / (Q\alpha^2)$ [%]",
    )
    axa.axvline(20.0, color=PALETTE["green"], ls=":", lw=1.6, label=r"Grid rule $\mathrm{cpw}_{\mathrm{Si}} \geq 20$")
    axa.set_xlabel(r"Cells per wavelength in silicon $\mathrm{cpw}_{\mathrm{Si}} = \lambda_{\mathrm{Si}} / \Delta x$")
    axa.set_ylabel(r"Relative error vs RCWA benchmark [%]")
    axa.set_title(r"(a) Subpixel Mesh Convergence ($n_x \in \{32, 48, 64\dots\}$)")
    axa.grid(True, which="both", alpha=0.28)
    axa.legend(loc="upper right", framealpha=0.92)

    # Panel (b): Subpixel smoothing vs Staircase quantization blindness across alpha
    axb = axes[1]
    a_st = d4["alphas_staircase"]
    q_st = d4["Q_staircase_nx32"]
    axb.loglog(
        alphas,
        q_fit,
        "o-",
        color=PALETTE["blue"],
        lw=2.0,
        ms=6.2,
        label=rf"Subpixel smoothing (slope $= {slope_fit:.4f}$)",
    )
    axb.loglog(
        a_st,
        q_st,
        "s--",
        color=PALETTE["vermilion"],
        lw=2.0,
        ms=6.0,
        label=rf"Staircase ($f_x \geq 0.5$, slope $= {float(d4['slope_staircase_nx32']):.4f}$!)",
    )
    axb.set_xlabel(r"Asymmetry parameter $\alpha = \Delta w / w_0$")
    axb.set_ylabel(r"Extracted quality factor $Q_{\mathrm{rad}}(\alpha)$")
    axb.set_title("(b) Failure Mode 1: Staircase Quantization Blindness")
    axb.grid(True, which="both", alpha=0.28)
    axb.legend(loc="upper right", framealpha=0.92)

    # Panel (c): CPML air-buffer proximity scan vs analytical evanescent decay exp(2*kappa_1*d_air)
    axc = axes[2]
    d_air = d4["d_air_to_pml_nm"]
    q_pml_bic = np.abs(d4["Q_pml_bic"])
    q_pml_a02 = np.abs(d4["Q_pml_alpha02"])
    d_line = np.linspace(d_air[0], d_air[-1], 200)
    q_evan_law = q_pml_bic[0] * np.exp(2.0 * kappa1 * (d_line - d_air[0]) * 1e-9)

    axc.semilogy(
        d_air,
        q_pml_bic,
        "o-",
        color=PALETTE["blue"],
        lw=2.0,
        ms=6.5,
        label=r"Wakis FIT $Q(\alpha=0)$ (parasitic CPML limit)",
    )
    axc.semilogy(
        d_line,
        q_evan_law,
        "--",
        color=PALETTE["black"],
        lw=1.7,
        label=r"Evanescent law $\propto \exp(+2\kappa_1 d_{\mathrm{air}})$",
    )
    axc.semilogy(
        d_air,
        q_pml_a02,
        "s-.",
        color=PALETTE["vermilion"],
        lw=1.9,
        ms=6.0,
        label=r"Wakis FIT $Q(\alpha=0.02)$ (true $Q_{\mathrm{rad}}=1.18\times 10^5$)",
    )
    axc.set_xlabel(r"Air buffer distance from bar face to CPML $d_{\mathrm{air}}$ [nm]")
    axc.set_ylabel(r"Quality factor $Q$")
    axc.set_title("(c) Failure Mode 2: Evanescent Tail CPML Leakage")
    axc.grid(True, which="both", alpha=0.28)
    axc.legend(loc="upper left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(fig_dir / "07_resolution_and_discretization_study.png", dpi=160)
    plt.close(fig)



# %% [markdown]
# ## 3. Results & Quantitative Comparison with Koshelev *et al.* (PRL 2018)
#
# ### 3.1 Setup Schematic, Symmetry Taxonomy, and Subpixel Graded Mesh
#
# Figures 0a–0b and the accompanying setup tables summarize the primary 2D TE width-asymmetric
# silicon bar-pair dimer metasurface setup ($p = 680\;\text{nm}$, $L_z = 2.60\;\mu\text{m}$,
# periodic $\pm x$ boundaries, $10$-cell CPML on $\pm z$, dual-cell subpixel permittivity
# $\varepsilon_{yy}(x, z)$ on the graded FIT mesh with $\Delta x = 21.25\;\text{nm}$ and
# $\Delta z_{\min} = 15\;\text{nm}$ across the $h = 190\;\text{nm}$ silicon slab) alongside the
# companion oblique-incidence Floquet–Bloch symmetric dimer setup ($\alpha = 0$,
# $k_x = k_0 \sin\theta_{\text{inc}}$ on the $\pm x$ Bloch faces).
#
# ![Normal-incidence asymmetric dimer setup: geometry, sources, monitors, and boundaries](figures/00a_setup_geometry.png)
# ![Normal-incidence asymmetric dimer setup: mesh, gap zoom, and Gaussian excitation](figures/00b_setup_mesh_and_excitation.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 680 nm × 21.2 nm × 2.60 µm (z ∈ [-1.30 µm, 1.30 µm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 32 × 1 × 74 (2368 cells) |
# | Discretization | Δz & resolution | Δ_min = 15 nm, 30.2 cells/λ (in n_max = 3.48), 5.7 cells across feature (85 nm) |
# | Boundary | -x face | Periodic |
# | Boundary | +x face | Periodic |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | CPML (10 cells, 448 nm) |
# | Boundary | +z face | CPML (10 cells, 448 nm) |
# | Material | Background | Air / vacuum background (ε_bg = 1.00, n = 1.00) (isotropic) |
# | Material | Region 1: Si bar 1 (w₁ = 181.9 nm, h = 190 nm) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
# | Material | Region 2: Si bar 2 (w₂ = 158.1 nm, h = 190 nm) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
# | Source | 1 · CurrentSource | pos: z = −580 nm (full x ∈ [−340, +340] nm), pol: Ey (TE, +z), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
# | Source | 2 · CurrentSource | pos: (x, z) = (−127.5, −25) nm (inside Si bar 1), pol: Jy (couples to C₂-odd BIC & quasi-BIC), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
# | Monitor | 3 · DiffractionMonitor | Reflected DiffractionMonitor (R₀₀) at z = −420 nm, pos: z = −420 nm (x ∈ [−340, +340] nm), freqs: λ ∈ [1520, 1578] nm (251 DFT frequencies) |
# | Monitor | 4 · DiffractionMonitor | Transmitted DiffractionMonitor (T₀₀) at z = +580 nm, pos: z = +580 nm (x ∈ [−340, +340] nm), freqs: λ ∈ [1520, 1578] nm (251 DFT frequencies) |
# | Monitor | 5 · ResonanceMonitor | ResonanceMonitor (harminv Ey) at (x₁, z) = (−127.5, +25) nm, pos: (x, z) = (−127.5, +25) nm, freqs: 0.90 f₀ – 1.10 f₀ (38 optical cycles ring-down) |
# | Variation | Asymmetry sweep (Tasks 0–1) | α = Δw / w₀ ∈ {0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20} (subpixel dual-cell averaging; Q_rad = Q₀ α⁻²) |
# | Variation | Absorption loss saturation (Task 2) | Conductivity σ_Si inside bars giving Q_abs ∈ {5×10³, 1.5×10⁴, 5×10⁴, 10⁵} (Q_tot⁻¹ = Q_rad⁻¹ + Q_abs⁻¹) |
# | Variation | Oblique Floquet angle sweep (Task 3) | Symmetric dimer (α = 0) with ±x Bloch BCs (k_x = k₀ sin θ_inc, θ_inc ∈ [1°, 12°]) — see 00a/00b_*_oblique.png |
# | Variation | Resolution & CPML proximity study (Task 4) | n_x ∈ {16, 24, 32, 48, 64}, subpixel vs staircase quantization, and d_air ∈ [150, 850] nm |
#
# For Task 3, the symmetric dimer ($\alpha = 0$, $w_1 = w_2 = 170\;\text{nm}$) is excited under
# oblique Floquet–Bloch boundary conditions ($\pm x$ faces with $k_x = k_0 \sin\theta_{\text{inc}}$,
# $\Delta\varphi_x = k_x p$) to break radiation symmetry via in-plane momentum rather than
# structural asymmetry:
#
# ![Oblique-incidence Floquet–Bloch symmetric dimer setup: geometry, sources, monitors, and boundaries](figures/00a_setup_geometry_oblique.png)
# ![Oblique-incidence Floquet–Bloch symmetric dimer setup: mesh, gap zoom, and excitation](figures/00b_setup_mesh_and_excitation_oblique.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 680 nm × 21.2 nm × 1.22 µm (z ∈ [-608 nm, 608 nm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 32 × 1 × 64 (2048 cells) |
# | Discretization | Δz & resolution | Δ_min = 19 nm, 23.8 cells/λ (in n_max = 3.48), 4.5 cells across feature (85 nm) |
# | Boundary | -x face | Bloch (k_x = -6.93e+05 rad/m) |
# | Boundary | +x face | Bloch (k_x = -6.93e+05 rad/m) |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | CPML (10 cells, 190 nm) |
# | Boundary | +z face | CPML (10 cells, 190 nm) |
# | Material | Background | Air / vacuum background (ε_bg = 1.00, n = 1.00) (isotropic) |
# | Material | Region 1: Si bar 1 (w₁ = w₀ = 170.0 nm, h = 190 nm) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
# | Material | Region 2: Si bar 2 (w₂ = w₀ = 170.0 nm, h = 190 nm) | ε_r = 12.11, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-95 nm, 95 nm] (d = 190 nm) |
# | Source | 1 · FloquetCellSource | pos: z = −320 nm (x ∈ [−340, +340] nm), pol: Ey (TE, θ_inc = 10°), pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
# | Source | 2 · CurrentSource | pos: (x, z) = (−127.5, −25) nm, pol: Jy, pulse: f₀ = 190.5 THz, Δf = 15.24 THz |
# | Monitor | 3 · DiffractionMonitor | Reflected Floquet DiffractionMonitor at z = −220 nm, pos: z = −220 nm (x ∈ [−340, +340] nm), freqs: λ ∈ [1560, 1620] nm (AngleSweep resampling) |
# | Monitor | 4 · DiffractionMonitor | Transmitted Floquet DiffractionMonitor at z = +220 nm, pos: z = +220 nm (x ∈ [−340, +340] nm), freqs: λ ∈ [1560, 1620] nm (AngleSweep resampling) |
# | Monitor | 5 · ResonanceMonitor | Bloch ResonanceMonitor (harminv Ey) at (x₁, z) = (−127.5, +25) nm, pos: (x, z) = (−127.5, +25) nm, freqs: Q_rad(k_∥) ∝ sin⁻² θ_inc |
# | Variation | Oblique angle & Bloch wavenumber sweep (Task 3) | θ_inc ∈ {1°, 2°, 3°, 5°, 7°, 10°} (harminv Q_rad(k_∥) ∝ k_∥⁻²) and AngleSweep θ_inc ∈ {0°, 6°, 9°, 12°} |
#
# ---
#
# ### 3.2 Fano Transmission Spectra and Collapse at $\alpha = 0$
#
# Figure 1(a) reproduces the characteristic **collapse of the Fano resonance** shown in
# Koshelev *et al.* Fig. 2(c): for $\alpha > 0$, an asymmetric Fano line shape appears in
# the zeroth-order normal-incidence transmission $T_{00}(\lambda)$, with its spectral
# width $\Delta\omega = \omega_0 / Q_{\text{rad}} \propto \alpha^2$ shrinking quadratically
# as $\alpha \to 0$ until the dip vanishes completely at the symmetry-protected BIC
# ($\alpha = 0$, $\lambda_{\text{BIC}} = 1575.76\;\text{nm}$ in 1D RCWA and
# $1573.88\;\text{nm}$ in 2D FIT at $n_x = 32$).
#
# Figure 1(b) compares the direct time-domain Wakis `DiffractionMonitor` normal-incidence
# spectrum $T_{00}(\lambda)$ and $R_{00}(\lambda)$ at $\alpha = 0.48$
# ($\alpha_{\text{half}} = 0.24$) against a nonlinear least-squares fit to Koshelev Eq. (1)
# and independent time-domain ring-down extraction via `wakis.resonances.harminv`:
#
# - **Direct `DiffractionMonitor` Fano fit (Koshelev Eq. 1)**:
#   $\lambda_{0,\text{Fano}} = 1544.190\;\text{nm}$, $Q_{\text{Fano}} = 191.31$,
#   Fano asymmetry parameter $q = +1.135$, and energy balance $R_{00} + T_{00} = 1.000$.
# - **Ring-down `harminv` extraction on the same 2D FIT grid**:
#   $\lambda_{0,\text{harminv}} = 1544.243\;\text{nm}$, $Q_{\text{harminv}} = 194.63$
#   (**$1.71\%$ relative agreement** in $Q$ and **$0.0034\%$ agreement** in $\lambda_0$).
# - **Symmetric mutation ($\alpha = 0$)**: the Fano feature vanishes completely
#   ($\max T - \min T = 0.0044 < 0.005$ across the resonance window), while `harminv`
#   measures $Q(\alpha = 0) = 1.57 \times 10^9$ (well above the run-length Fourier limit).
#
# Figure 1(c) demonstrates that rescaling the frequency axis by the radiative linewidth
# into the normalized detuning $\Omega = 2(\omega - \omega_0)/\gamma$ collapses all seven
# transmission spectra ($\alpha \in [0.02, 0.20]$, spanning a $100\times$ variation in
# $Q_{\text{rad}}$) onto a single universal Fano master curve with $q = 1.173$.
#
# ![Normal-incidence Fano transmission spectra and universal collapse](figures/01_fano_transmission_spectra.png)
#
# ---
#
# ### 3.3 Universal Inverse-Square Scaling $Q_{\text{rad}}(\alpha) = Q_0\,\alpha^{-2}$
#
# Table 1 summarizes the resonance wavelengths $\lambda_0(\alpha)$, radiative quality
# factors $Q_{\text{rad}}(\alpha)$, and compensated invariants $Q_{\text{rad}}(\alpha)\,\alpha^2$
# across all seven asymmetry parameters $\alpha \in [0.02, 0.20]$ from both Wakis 2D FIT
# `harminv` and 1D Fourier-modal RCWA Fano fitting, alongside the symmetry-protected BIC
# point ($\alpha = 0$).
#
# | Asymmetry $\alpha = \Delta w / w_0$ | Half-bar $\alpha_{\text{half}} = \alpha/2$ | Wakis FIT $\lambda_0$ [nm] | Wakis FIT $Q_{\text{rad}}$ (`harminv`) | Wakis FIT $Q_{\text{rad}}\alpha^2$ | 1D RCWA $\lambda_0$ [nm] | 1D RCWA $Q_{\text{Fano}}$ (Eq. 1) | 1D RCWA $Q_{\text{Fano}}\alpha^2$ | Relative $\|\Delta Q\|/Q$ [%] |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | $0.00$ (True BIC) | $0.00$ | $1573.885$ | $1.57 \times 10^9$ | $0.000$ | $1575.763$ | $\infty$ (dark) | — | — |
# | $0.02$ | $0.01$ | $1573.873$ | $117\,597.6$ | $47.039$ | $1575.751$ | $116\,648.7$ | $46.659$ | $0.81\%$ |
# | $0.04$ | $0.02$ | $1573.836$ | $29\,406.1$ | $47.050$ | $1575.713$ | $29\,150.4$ | $46.641$ | $0.88\%$ |
# | $0.06$ | $0.03$ | $1573.775$ | $13\,073.2$ | $47.064$ | $1575.650$ | $12\,947.0$ | $46.609$ | $0.97\%$ |
# | $0.08$ | $0.04$ | $1573.689$ | $7\,356.6$ | $47.082$ | $1575.562$ | $7\,275.8$ | $46.565$ | $1.11\%$ |
# | $0.10$ | $0.05$ | $1573.583$ | $4\,710.7$ | $47.107$ | $1575.448$ | $4\,650.9$ | $46.509$ | $1.29\%$ |
# | $0.14$ | $0.07$ | $1573.290$ | $2\,406.7$ | $47.171$ | $1575.138$ | $2\,365.2$ | $46.358$ | $1.75\%$ |
# | $0.20$ | $0.10$ | $1572.677$ | $1\,182.7$ | $47.308$ | $1574.477$ | $1\,150.9$ | $46.036$ | $2.76\%$ |
#
# Table 2 compares our extracted scaling exponents, prefactors, and spectral features
# against the published results in Koshelev *et al.*, *Phys. Rev. Lett.* **121**, 193903 (2018).
# **Note on geometry difference:** Tutorial 6.3 reproduces the universal $\alpha^{-2}$ scaling law,
# not Koshelev *et al.*'s 3D device. Koshelev *et al.*'s points come from 3D silicon bar pairs
# ($\lambda_{\text{BIC}} \approx 1571\;\text{nm}$ for tilted 3D bars in Fig. 2 and
# $\lambda_{\text{BIC}} \approx 1435\;\text{nm}$ for unequal 3D bars in Fig. 3b), whereas our model
# is a 2D TE silicon dimer cross-section at $\lambda_{\text{BIC}} = 1573.88\;\text{nm}$. Only the
# power-law exponent $s = -2$ is directly comparable across geometries; the non-universal prefactors
# $Q_0$ are listed separately below for reference.
#
# | Observable / Scaling Law | Koshelev *et al.* (PRL 121, 193903, 2018) Reference | Wakis 2D FIT (`harminv` / `DiffractionMonitor`) | 1D RCWA Fano Eq. (1) Benchmark | Agreement / Notes |
# | :--- | :--- | :--- | :--- | :--- |
# | Power-law exponent $s$ in $Q_{\text{rad}} \propto \alpha^s$ | $s = -2$ exact for $\alpha \ll 1$ (Eq. 3, p. 2; Fig. 3a–c, p. 3; Fig. 4, p. 4) | Free fit: $s = -1.9978 \pm 0.0008$ ($\alpha \in [0.02, 0.20]$) | Free fit: $s = -2.0049 \pm 0.0012$ ($\alpha \in [0.02, 0.20]$) | **$0.11\%$** from exact $-2$ |
# | Prefactor $Q_0 = Q_{\text{rad}}\alpha^2$ *(different 2D vs 3D geometries)* | 3D tilted Si bars ($\lambda \approx 1571\;\text{nm}$, Fig. 2c/4) & 3D bar pairs ($\lambda \approx 1435\;\text{nm}$, Fig. 3b): $Q_0 \approx 13.7$ vs $\alpha = \sin\theta$ | 2D Si dimer ($\lambda_{\text{BIC}} = 1573.88\;\text{nm}$): slope $-2$ imposed $Q_0(\alpha) = 47.12$, $Q_0(\alpha_{\text{half}}) = 11.78$ (free fit $Q_0 = 47.39$) | 2D Si dimer ($\lambda_{\text{BIC}} = 1575.76\;\text{nm}$): slope $-2$ imposed $Q_0(\alpha) = 46.48$, $Q_0(\alpha_{\text{half}}) = 11.62$ (free fit $Q_0 = 45.90$) | Different geometry (2D dimer vs 3D bars); shown in separate inset in Fig. 2(a) |
# | Fano line shape & collapse at $\alpha = 0$ | $T_{\text{Fano}}(\Omega) = T_0 + \frac{A(q+\Omega)^2}{1+\Omega^2}$ (Eq. 1, p. 2); dip disappears at $\theta = 0^\circ$ (Fig. 2c, p. 2) | $Q_{\text{Fano}} = 191.31$ vs $Q_{\text{harminv}} = 194.63$ ($1.71\%$ diff); contrast $= 0.0044$ at $\alpha = 0$ | Universal collapse with $q = 1.173$ across $100\times$ range in $Q$ | Verified both in `DiffractionMonitor` and `harminv` |
# | Total $Q$ saturation with absorption / loss | $Q_{\text{tot}}^{-1} = Q_{\text{rad}}(\alpha)^{-1} + Q_{\text{abs}}^{-1}$ saturates at small $\alpha$ (Fig. 4, p. 4, $\text{Im}\,\varepsilon$ curves) | Max error $< 0.36\%$ across $Q_{\text{abs}} \in \{10^5, 2\times 10^4, 5\times 10^3\}$; bare $Q_{\text{rad}}$ recovered to $< 0.1\%$ | Exact analytical / numerical match | Verified via FIT conductivity $\sigma$ |
# | Oblique-incidence dispersion at $\alpha = 0$ | Symmetry breaking by in-plane $\mathbf{k}_\parallel$: $Q_{\text{rad}}(k_\parallel) \propto (k_\parallel / k_0)^{-2} = \sin^{-2}\theta_{\text{inc}}$ | Slope $= -1.9920$ across $\theta_{\text{inc}} \in [1^\circ, 10^\circ]$; $Q_{0,\theta} = 33.90$ | Fano fan opens quadratically in $\sin\theta_{\text{inc}}$ | Verified with `SolverFIT3D(k_bloch)` and `wakis.floquet.AngleSweep` |
#
# ![Universal inverse-square scaling Q_rad vs alpha](figures/02_q_vs_alpha_scaling.png)
#
# ---
#
# ### 3.4 Near-Field Trapping vs Far-Field Radiation ($\alpha = 0$ vs $\alpha = 0.24$)
#
# Figure 3 and the animated ring-down GIF (Figure 4) visualize the physical origin of the
# symmetry-protected BIC:
#
# 1. **At $\alpha = 0$ (Figure 3a & left panel of Figure 4)**: the electric field $E_y(x, z)$
#    is strictly antisymmetric between Bar 1 and Bar 2 ($E_y(-x, z) = -E_y(x, z)$). Because
#    $p = 680\;\text{nm} < \lambda_{\text{BIC}} = 1573.88\;\text{nm}$, all non-zero Floquet
#    harmonics $m = \pm 1, \pm 2, \dots$ are evanescent in air, decaying vertically as
#    $\exp(-\kappa_1 |z|)$ with $\kappa_1 = \sqrt{(2\pi/p)^2 - k_0^2} = 8.333\;\mu\text{m}^{-1}$
#    (decay length $1/\kappa_1 = 120.0\;\text{nm}$). Meanwhile, coupling to the sole
#    propagating plane-wave channel $m = 0$ vanishes by parity ($D_{0,y}(\alpha = 0) = 0$).
#    Thus $|E_y(x=-d/2, z)|$ in Figure 3(c) drops exponentially over four decades with zero
#    radiative floor.
# 2. **At $\alpha = 0.24$ (Figure 3b & right panel of Figure 4)**: Bar 1 ($w_1 = 190.4\;\text{nm}$)
#    and Bar 2 ($w_2 = 149.6\;\text{nm}$) no longer cancel in the zeroth diffraction order.
#    Beyond $|z| \gtrsim 0.45\;\mu\text{m}$, the evanescent $m = \pm 1$ near field gives way to
#    a constant-amplitude outward-propagating plane wave $E_{0,y}\exp(\mp i k_0 z)$ (horizontal
#    phase fronts filling the air buffers in Figure 3b and red dashed floor in Figure 3c),
#    draining the stored energy with $Q_{\text{rad}} = 824.9$.
#
# ![Near-field and far-field maps of true BIC vs quasi-BIC](figures/03_bic_vs_quasi_bic_fields.png)
#
# ![Animated ring-down comparison of true BIC vs quasi-BIC](figures/04_bic_vs_quasi_bic_ringdown.gif)
#
# ---
#
# ### 3.5 Material & Fabrication Loss Saturation ($Q_{\text{tot}}$) and Bare $Q_{\text{rad}}$ Recovery
#
# Following Koshelev *et al.* Fig. 4 (p. 4), introducing weak conductivity $\sigma$ inside the
# silicon bars ($\Gamma_{\text{conf}} = 0.9476$) adds an $\alpha$-independent non-radiative
# decay rate $\gamma_{\text{abs}} = \omega_0 / (2 Q_{\text{abs}})$. As shown in Figure 5(a),
# $Q_{\text{tot}}(\alpha)$ follows the lossless $\alpha^{-2}$ law at large asymmetry
# ($Q_{\text{rad}}(\alpha) \ll Q_{\text{abs}}$) and saturates smoothly to $Q_{\text{tot}}(0) \approx Q_{\text{abs}}$
# as $\alpha \to 0$. Subtracting the symmetric dark-mode damping rate via Eq. (17),
# $Q_{\text{rad,rec}}(\alpha) = [Q_{\text{tot}}(\alpha)^{-1} - Q_{\text{tot}}(0)^{-1}]^{-1}$,
# collapses all three lossy curves in Figure 5(b) back onto the bare radiative law
# $Q_0\,\alpha^{-2}$ to within $< 0.1\%$.
#
# ![Absorption Q saturation and bare radiative Q recovery](figures/05_absorption_q_saturation.png)
#
# ---
#
# ### 3.6 Oblique-Incidence Floquet Dispersion $Q_{\text{rad}}(k_\parallel) \propto k_\parallel^{-2}$
#
# Even for a geometrically symmetric metasurface ($\alpha = 0$), tilting the incident wavevector
# by $\theta_{\text{inc}} \ne 0^\circ$ introduces a Bloch phase shift
# $\Delta\phi_x = k_x d = \frac{3\pi}{4}\frac{p}{\lambda_0}\sin\theta_{\text{inc}}$ between
# the two bars, opening the $(0,0)$ radiation channel with
# $Q_{\text{rad}}(\theta_{\text{inc}}) = Q_{0,\theta}\,\sin^{-2}\theta_{\text{inc}}$.
#
# - Figure 6(a) shows constant-angle transmission spectra $T_{00}(\lambda, \theta_{\text{inc}})$
#   reconstructed from fixed-$k_x$ Bloch time-domain runs using `wakis.floquet.AngleSweep`:
#   at $\theta_{\text{inc}} = 0^\circ$ the spectrum is flat and featureless, whereas at
#   $\theta_{\text{inc}} = 6^\circ, 9^\circ, 12^\circ$ the quasi-BIC Fano dip emerges and
#   broadens quadratically.
# - Figure 6(c) confirms that Bloch `harminv` extraction at $\alpha = 0$ across
#   $\theta_{\text{inc}} \in \{1^\circ, 2^\circ, 3^\circ, 5^\circ, 7^\circ, 10^\circ\}$ yields
#   $Q_{\text{rad}} \in [1.08 \times 10^5, 1.13 \times 10^3]$ with a fitted log–log slope of
#   **$-1.9920$** and prefactor $Q_{0,\theta} = 33.90$.
#
# ![Oblique-incidence Floquet dispersion and AngleSweep spectra](figures/06_oblique_floquet_dispersion.png)
#
# ---
#
# ## 4. Resolution Study
#
# Table 3 and Figure 7(a) report the mesh convergence of the symmetric BIC wavelength
# $\lambda_{\text{BIC}}(\alpha = 0)$ and the compensated quasi-BIC invariant
# $Q_{\text{rad}}\alpha^2$ at $\alpha = 0.10$ across transverse resolutions
# $n_x \in \{32, 48, 64, 96\}$ (with dual-cell subpixel smoothing enabled). Both quantities
# converge monotonically at second order $\mathcal{O}(\Delta x^2)$ toward the 1D Fourier-modal
# RCWA benchmark ($\lambda_{\text{BIC,RCWA}} = 1575.7633\;\text{nm}$,
# $(Q\alpha^2)_{\text{RCWA}} = 46.509$).
#
# | Transverse cells $n_x$ | Grid step $\Delta x$ [nm] | Cells per $\lambda_{\text{Si}}$ ($\mathrm{cpw}_{\text{Si},x}$) | Cells across $85\;\text{nm}$ gap | $\lambda_{\text{BIC}}(\alpha=0)$ [nm] | Rel. error vs RCWA $\lambda_{\text{BIC}}$ [%] | $Q_{\text{rad}}(\alpha=0.10)$ | $Q_{\text{rad}}\alpha^2$ ($\alpha=0.10$) | Rel. error vs RCWA $Q\alpha^2$ [%] |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# | $32$ (baseline) | $21.25$ | $21.28$ | $4.00$ | $1573.8847$ | $0.119\%$ | $4710.65$ | $47.1065$ | $1.28\%$ |
# | $48$ | $14.17$ | $31.92$ | $6.00$ | $1574.4170$ | $0.085\%$ | $4690.77$ | $46.9077$ | $0.86\%$ |
# | $64$ | $10.63$ | $42.56$ | $8.00$ | $1575.2147$ | $0.035\%$ | $4683.88$ | $46.8388$ | $0.71\%$ |
# | $96$ (fine `--full`) | $7.08$ | $63.84$ | $12.00$ | $1575.3964$ | $0.023\%$ | $4668.02$ | $46.6802$ | $0.37\%$ |
# | **1D RCWA ($2M+1=41$)** | Spectral | $\infty$ | Exact | **$1575.7633$** | — | **$4650.90$** | **$46.5090$** | — |
#
# ![Resolution convergence, staircase quantization failure, and CPML proximity](figures/07_resolution_and_discretization_study.png)
#
# ---
#
# ## 5. Limits & Failure Modes
#
# High-$Q$ quasi-BIC metasurface simulations exhibit four severe numerical failure modes when
# standard FDTD/FIT rules of thumb are violated:
#
# 1. **Failure Mode 1 — Staircase quantization blindness (`averaging="staircase"`, Figure 7b)**:
#    At the baseline grid step $\Delta x = 21.25\;\text{nm}$ ($n_x = 32$), the bar-width
#    perturbation $\delta w = \frac{1}{2}\alpha w_0 = 85\,\alpha\;\text{nm}$ is smaller than
#    $\Delta x$ for every $\alpha \le 0.25$. Because the symmetric bar edges at $\alpha = 0$
#    happen to lie exactly on half-cell dual boundaries ($f_x = 0.50$), binary staircase
#    thresholding ($f_x \ge 0.5 \Rightarrow \varepsilon_{\text{Si}}$) jumps discontinuously by
#    $+1$ full cell on Bar 1 and $-1$ full cell on Bar 2 for *any* $\alpha \in (0, 0.25)$,
#    locking the discrete geometry at an effective asymmetry $\alpha_{\text{eff,stair}} = 0.25$
#    and producing a constant $Q_{\text{stair}}(\alpha) \equiv 768.2$ with **log–log slope
#    $= 0.0000$** instead of $-2.000$! Continuous dual-cell subpixel volume-fraction averaging
#    (`averaging="subpixel"`) restores the exact slope $-1.9978$ on the very same $n_x = 32$ grid.
# 2. **Failure Mode 2 — Evanescent near-field absorption in the CPML (`d_air_to_pml`, Figure 7c)**:
#    At $\alpha = 0$, the BIC is confined by the evanescent $m = \pm 1$ Rayleigh harmonics
#    decaying into air as $\exp(-\kappa_1 |z|)$ ($\kappa_1 = 8.333\;\mu\text{m}^{-1}$). If the
#    air buffer $d_{\text{air}}$ between the bar surface and the inner face of the $z^\mp$ CPML
#    is too thin, the CPML absorbs the evanescent tail and creates a spurious non-radiative decay
#    channel $Q_{\text{PML}} \propto \exp(+2\kappa_1 d_{\text{air}})$. As demonstrated in
#    Figure 7(c), reducing $d_{\text{air}}$ from $757.4\;\text{nm}$ to $215.4\;\text{nm}$
#    degrades $Q(\alpha = 0)$ by four orders of magnitude from $1.57 \times 10^9$ down to
#    $1.64 \times 10^5$, which artificially caps $Q(\alpha = 0.02)$ at $6.87 \times 10^4$
#    instead of its true radiative value $1.176 \times 10^5$. Always maintain
#    $d_{\text{air}} \gtrsim 6 / \kappa_1$ ($\ge 720\;\text{nm}$ here).
# 3. **Failure Mode 3 — Unresolved high-$Q$ lines in direct Fourier-transform spectra**:
#    Resolving a Fano resonance of quality factor $Q$ directly by discrete Fourier transform of a
#    plane-wave transmission time trace requires running the solver for at least
#    $t_{\max} \gtrsim 2 Q / f_0$ ($\approx 1.24\;\text{ns}$ or $3.5 \times 10^5$ time steps at
#    $Q = 1.18 \times 10^5$). Truncating the time trace earlier convolves the Fano profile with
#    a $\mathrm{sinc}(\omega t_{\max}/2)$ window and obscures the dip. Filter diagonalization via
#    `wakis.resonances.harminv` extracts both $\omega_0$ and $Q$ to $< 1\%$ accuracy from a short
#    $750\;\text{fs}$ ($1.5 \times 10^4$-step) ring-down trace — a **$23\times$ speedup**.
# 4. **Failure Mode 4 — Parity mismatch of the excitation source**:
#    At $\alpha = 0$ and $\theta_{\text{inc}} = 0^\circ$, a symmetric plane wave ($E_y(x)$ even)
#    is strictly orthogonal to the antisymmetric BIC eigenmode ($E_y(x)$ odd). Exciting the unit
#    cell *only* with a normal-incidence plane wave at $\alpha = 0$ never populates the BIC mode;
#    characterizing the dark state at $\alpha = 0$ requires an off-center local probe
#    (at $x = -d/2$) that overlaps with the antisymmetric modal profile.
#
# ---
#
# ## 6. Exercises & Extensions
#
# 1. **Full 3D Tilted Silicon Bar-Pair Sweep (`--task 5`)**:
#    Inspect `data/task_05.json` (`preflight_and_run_3d_tilted_bars_task`), where `wakis.estimate`
#    verifies that the 3D Koshelev Fig. 2(a) tilted silicon bar-pair unit cell
#    ($p_x = p_y = 680\;\text{nm}$, $n_x = n_y = 28$, $n_z = 56$, $N_{\text{cells}} = 43\,904$ in
#    `--quick` or $n_x = n_y = 32$, $n_z = 68$, $N_{\text{cells}} = 69\,696$ in `--full`, peak host
#    RAM $\approx 0.62\;\text{GB}$) fits comfortably on a single laptop/workstation node. Extend
#    Task 5 to sweep $\theta \in \{3^\circ, 6^\circ, 9^\circ, 12^\circ, 15^\circ\}$ ($\alpha = \sin\theta$)
#    and verify $Q_{\text{rad}}(\theta) \propto (\sin\theta)^{-2}$ in full 3D.
# 2. **Critical Coupling and Maximum Chirality / Field Enhancement**:
#    Using the coupled-mode relation $|c_{\text{max}}|^2 \propto \gamma_{\text{rad}} / (\gamma_{\text{rad}} + \gamma_{\text{abs}})^2$,
#    show analytically and numerically using Task 2 (`run_absorption_saturation_task`) that the
#    internal electric-field intensity $|E_y|^2$ inside the silicon bars under plane-wave
#    illumination is maximized not at $\alpha = 0$ (where $\gamma_{\text{rad}} = 0$), but at the
#    **critical-coupling asymmetry** $\alpha_{\text{crit}} = \sqrt{Q_0 / Q_{\text{abs}}}$ where
#    $Q_{\text{rad}}(\alpha_{\text{crit}}) = Q_{\text{abs}}$.
# 3. **Diffraction Threshold (Wood–Rayleigh Anomaly) Breakdown**:
#    Increase the lattice period $p$ above the free-space resonance wavelength ($p > \lambda_0$)
#    so that the $m = \pm 1$ diffraction orders become propagating in air ($k_{z,\pm 1} \in \mathbb{R}$).
#    Show that even at $\alpha = 0$, the antisymmetric mode now leaks into the $(\pm 1, 0)$
#    diffraction orders and ceases to be a bound state in the continuum.
#
# ---
#
# ## 7. Verified References
#
# 1. **K. Koshelev, S. Lepeshov, M. Liu, A. Bogdanov, and Y. Kivshar**, *"Asymmetric metasurfaces
#    with high-$Q$ resonances governed by bound states in the continuum"*, *Phys. Rev. Lett.*
#    **121**(19), 193903 (2018), DOI:
#    [`10.1103/PhysRevLett.121.193903`](https://doi.org/10.1103/PhysRevLett.121.193903),
#    [`arXiv:1809.00330`](https://arxiv.org/abs/1809.00330).
#    - **Verified full-text equations & figures**: Eq. (1) on p. 2 (Fano transmittance formula
#      $T_{\text{Fano}}(\omega)$ and radiative damping $\gamma = \omega_0 / Q$); Eq. (2) on p. 2
#      (zeroth-order far-field coupling amplitude $D_{0,\alpha} = -\frac{\omega_0}{\sqrt{2 S_0 c}}\,p_\alpha$);
#      Eq. (3) on p. 2 (universal inverse-square law $Q_{\text{rad}}(\alpha) = Q_0\,\alpha^{-2}$);
#      Fig. 1(a–i) on p. 1 (taxonomy of symmetry-broken metasurface unit cells including tilted bar
#      pairs $\alpha = \sin\theta$ in Fig. 1a and dimerized bar pairs $\alpha = \Delta L / L_0$ or
#      $\Delta w / w_0$ in Fig. 1e); Fig. 2(a–c) on p. 2 (silicon bar-pair metasurface and collapse
#      of the Fano resonance as $\theta \to 0^\circ$); Fig. 3(a–c) on p. 3 (universal linear log–log
#      scaling with slope $-2$ across ten experimental metasurface platforms); Fig. 4 on p. 4
#      (radiative $Q_{\text{rad}} \propto \alpha^{-2}$ vs total $Q_{\text{tot}}$ saturation with
#      material loss $\text{Im}(\varepsilon)$).
# 2. **C. W. Hsu, B. Zhen, A. D. Stone, J. D. Joannopoulos, and M. Soljačić**, *"Bound states in
#    the continuum"*, *Nat. Rev. Mater.* **1**, 16048 (2016), DOI:
#    [`10.1038/natrevmats.2016.48`](https://doi.org/10.1038/natrevmats.2016.48).
#    - Comprehensive review of symmetry-protected and accidental bound states in the continuum and
#      temporal coupled-mode theory of radiative leakage.
# 3. **S. Fan, W. Suh, and J. D. Joannopoulos**, *"Temporal coupled-mode theory for the Fano
#    resonance in optical resonators"*, *J. Opt. Soc. Am. A* **20**(3), 569–572 (2003), DOI:
#    [`10.1364/JOSAA.20.000569`](https://doi.org/10.1364/JOSAA.20.000569).
#    - Derivation of the general Fano transmission line shape from temporal coupled-mode theory
#      and unitary/reciprocity constraints in two-port slab resonators.
# 4. **V. A. Mandelshtam and H. S. Taylor**, *"Harmonic inversion of time signals and its
#    applications"*, *J. Chem. Phys.* **107**(17), 6756–6769 (1997), DOI:
#    [`10.1063/1.475324`](https://doi.org/10.1063/1.475324).
#    - Filter diagonalization / harmonic inversion (`harminv`) algorithm implemented in
#      `wakis.resonances.harminv` for extracting complex eigenfrequencies $\omega_k - i\gamma_k$
#      and quality factors $Q_k = \omega_k / (2\gamma_k)$ from short time-domain ring-down signals.


# %%
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Tutorial 6.3: Quasi-bound states in the continuum in asymmetric dielectric "
            "metasurfaces (Koshelev et al., PRL 121, 193903, 2018)."
        )
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the total number of parallelizable simulation tasks and exit.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single task index i in [0, N-1] and write data/task_{i:02d}.npz.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the lightweight verification sweep (< 1 min) and generate all figures.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the complete paper-grade sweep and generate all figures.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate all figures in figures/ from existing data/task_*.npz files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Directory for saved .npz and .json outputs (default: ./data).",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=FIG_DIR,
        help="Directory for saved figures (default: ./figures).",
    )
    args = parser.parse_args(argv)

    quick = bool(args.quick or not args.full)
    tasks = build_tasks(quick=quick)

    if args.list_tasks:
        print(len(tasks))
        for t in tasks:
            print(f"  task {t['index']:02d}: {t['name']} — {t['description']}")
        return 0

    out_dir = Path(args.out)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.plot and args.task is None and not args.quick and not args.full:
        plot_all(out_dir=out_dir, fig_dir=fig_dir)
        total_s = 0.0
    elif args.task is not None:
        idx = int(args.task)
        if idx < 0 or idx >= len(tasks):
            raise ValueError(f"--task {idx} out of range [0, {len(tasks) - 1}]")
        execute_task(tasks[idx], out_dir=out_dir)
        return 0
    else:
        t_start = time.perf_counter()
        summaries = []
        for task in tasks:
            idx = int(task["index"])
            npz_path = out_dir / f"task_{idx:02d}.npz"
            json_path = out_dir / f"task_{idx:02d}.json"
            if quick and idx in (0, 2, 3, 4) and npz_path.is_file() and json_path.is_file():
                with open(json_path, encoding="utf-8") as fh:
                    s = json.load(fh)
                summaries.append(s)
                print(
                    f"[Tutorial 6.3] Quick mode: verified cached task {idx:02d} ({task['name']})"
                )
                continue
            print(f"[Tutorial 6.3] Running task {idx:02d}: {task['name']} ...")
            s = execute_task(task, out_dir=out_dir)
            summaries.append(s)
            print(
                f"[Tutorial 6.3] Completed task {idx:02d} in {s['elapsed_s']:.2f} s"
            )

        plot_all(out_dir=out_dir, fig_dir=fig_dir)
        total_s = time.perf_counter() - t_start

    summaries = []
    for idx in range(len(tasks)):
        jp = out_dir / f"task_{idx:02d}.json"
        if jp.is_file():
            with open(jp, encoding="utf-8") as fh:
                summaries.append(json.load(fh))

    d0 = np.load(out_dir / "task_00.npz")
    d2 = np.load(out_dir / "task_02.npz")
    d3 = np.load(out_dir / "task_03.npz")
    d4 = np.load(out_dir / "task_04.npz")
    with open(out_dir / "task_05.json", encoding="utf-8") as fh:
        d5 = json.load(fh)

    fano_sym_contrast = float(
        np.max(d0["fano_T_sym_alpha0"]) - np.min(d0["fano_T_sym_alpha0"])
    )
    master_summary = {
        "tutorial": "06-03_quasi_bic_metasurface",
        "paper": "Koshelev et al., Phys. Rev. Lett. 121, 193903 (2018)",
        "doi": "10.1103/PhysRevLett.121.193903",
        "arxiv": "1809.00330",
        "mode": "quick" if quick else "full",
        "total_elapsed_s": float(total_s),
        "lambda_bic_fit_nm": float(d0["lambda_bic_fit_nm"]),
        "Q_bic_fit": float(d0["Q_bic_fit"]),
        "lambda_bic_rcwa_nm": float(d0["lambda_bic_rcwa_nm"]),
        "log_slope_fit": float(d0["log_slope_fit"]),
        "Q0_alpha_fit": float(d0["Q0_alpha_fit"]),
        "Q0_alpha_half_fit": float(d0["Q0_alpha_half_fit"]),
        "log_slope_rcwa": float(d0["log_slope_rcwa"]),
        "Q0_alpha_rcwa": float(d0["Q0_rcwa"]),
        "Q0_alpha_half_rcwa": 0.25 * float(d0["Q0_rcwa"]),
        "fano_Q_fit": float(d0["fano_Q_fit"]),
        "fano_Q_harminv": float(d0["fano_Q_harminv"]),
        "fano_q_param": float(d0["fano_q_param"]),
        "fano_sym_contrast": fano_sym_contrast,
        "gamma_conf_si": float(d2["gamma_confinement"]),
        "log_slope_sin_theta": float(d3["log_slope_sin_theta"]),
        "Q0_sin_theta": float(d3["Q0_sin_theta"]),
        "slope_staircase_nx32": float(d4["slope_staircase_nx32"]),
        "preflight_3d": d5,
        "tasks": summaries,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(master_summary, fh, indent=2)

    print(
        f"[Tutorial 6.3] Done in {total_s:.2f} s — "
        f"slope(log Q vs log alpha) = {float(d0['log_slope_fit']):.4f} (FIT), "
        f"{float(d0['log_slope_rcwa']):.4f} (RCWA); "
        f"slope(log Q vs log sin theta) = {float(d3['log_slope_sin_theta']):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
