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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/04_radiation_and_antennas/04-06_nfc_coils/nfc_coils.ipynb)
#
# # Tutorial 4.6 · NFC reader and tag coils at 13.56 MHz (near-field inductive coupling)
# <h3 align="center">Quasi-magnetostatic loop coupling, capacitive resonance tuning, and flexible NiZn ferrite shielding over metal in 3D FIT</h3>
#
# ---
#
# ### Abstract
#
# **Near-Field Communication (NFC) at $f_0 = 13.56\;\text{MHz}$ ($\lambda_0 = 22.11\;\text{m}$) operates deep inside the reactive magnetic near-field zone ($r \ll \lambda_0 / (2\pi) = 3.52\;\text{m}$), transferring power and data across centimetre gaps by inductive flux linkage between two electrically small planar spiral coils. In this tutorial we model a representative ISO/IEC 14443 reader antenna ($66 \times 66\;\text{mm}$, $N_1 = 3$ turns) and an ID-1 card-size tag coil ($78 \times 48\;\text{mm}$ inside the $85.6 \times 54.0\;\text{mm}$ card outline, $N_2 = 3$ turns) in `SolverFIT3D`. We benchmark the simulated self-inductances $L_1, L_2$ against Wheeler's formula [1], Mohan et al.'s modified Wheeler and current-sheet approximations [2], and Grover's strip-GMD summation [3], and validate the mutual inductance $M(d, \Delta x)$ and coupling coefficient $k = M / \sqrt{L_1 L_2}$ against the numerical Neumann double integral over the exact 3D trace segments. Using `LumpedPort`, `LumpedElement`, and `s_matrix`, we model copper skin-depth trace losses ($\delta_{\text{Cu}} \approx 17.7\;\mu\text{m}$ at $13.56\;\text{MHz}$), series/parallel capacitive resonance tuning, and two-port link efficiency $\eta_{\max}$ (Schormans et al. [4]). Finally, we compare three smartphone/reader mounting configurations—free space, direct mounting on a PEC battery/chassis plate, and shielding with a commercial flexible sintered NiZn ferrite sheet (`dispersive_lib["nizn ferrite"]`, fitted to complex permeability data $\mu'(f) - j\mu''(f)$ [5, 6])—and verify the thin-sheet $\mu' \cdot t$ equivalence principle and its physical breakdown limits.**
#
# ### Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. **Explain** why $13.56\;\text{MHz}$ NFC is a quasi-magnetostatic inductive link ($k_0 r \sim 10^{-2} \ll 1$, $|\mathbf{H}| \propto r^{-3}$), and compute planar spiral self-inductance from the **Modified Wheeler** [1, 2], **Current-Sheet** [2], and **Grover GMD** [3] formulas.
# 2. **Evaluate** the exact **Neumann double line integral** numerically for arbitrary 3D multi-turn rectangular spiral traces to obtain the analytical reference $M(d, \Delta x)$ and coupling factor $k = M / \sqrt{L_1 L_2}$.
# 3. **Model** copper skin-depth resistance ($\delta_{\text{Cu}} = 1 / \sqrt{\pi f_0 \mu_0 \sigma_{\text{Cu}}} \approx 17.71\;\mu\text{m}$ at $13.56\;\text{MHz}$), unloaded and loaded coil quality factors $Q_1, Q_2$, capacitive tuning $C_i = 1 / (\omega_0^2 L_i)$ with `LumpedElement`, two-port $S$/$Z$-parameters with `LumpedPort` and `s_matrix`, and the **Schormans et al. (2018)** maximum link efficiency $\eta_{\max} = k^2 Q_1 Q_2 / (1 + \sqrt{1 + k^2 Q_1 Q_2})^2$ [4].
# 4. **Formulate** a practical explicit 3D FIT time-domain strategy for multiscale RF problems ($\lambda_0 = 22.1\;\text{m}$ vs millimetre traces), combining a compact graded near-field domain, broadband port-impedance damping, and optional background permittivity scaling ($\varepsilon_r = s \gg 1$), and identify when $s$-scaling fails due to shifted self-resonance $f_{\text{SR}}(s) \propto s^{-1/2}$.
# 5. **Fit and simulate** a magnetic dispersive material (`DispersiveMaterial` with `mu_inf`, `mu_poles` via `fit_mu_poles`) representing a typical flexible NiZn NFC ferrite sheet (a model spectrum of the form in [6], tuned to the $\mu' \approx 150, \mu'' \approx 3.6$ at $13.56\;\text{MHz}$ reported for such sheets [5]; not a product measurement), explain image eddy currents in a metal chassis, contrast NiZn versus MnZn ferrites via Snoek's law [7], and validate the thin-sheet $(\mu' - 1) t$ equivalence across multiple $(\mu', t)$ pairs.
#
# ### Contents
#
# 1. [Physics of near-field inductive links at 13.56 MHz](#1.-Physics-of-near-field-inductive-links-at-13.56-MHz)
# 2. [Representative ISO/IEC 14443 reader & ID-1 card tag geometry](#2.-Representative-ISO/IEC-14443-reader-&-ID-1-card-tag-geometry)
# 3. [Time-domain strategy, preflight estimation & permittivity scaling](#3.-Time-domain-strategy,-preflight-estimation-&-permittivity-scaling)
# 4. [Ferrite shielding over metal & magnetic dispersive model](#4.-Ferrite-shielding-over-metal-&-magnetic-dispersive-model)
# 5. [3D FIT solver pipeline & study task table](#5.-3D-FIT-solver-pipeline-&-study-task-table)
# 6. [Generating the complete 11-figure & animation suite](#6.-Generating-the-complete-11-figure-&-animation-suite)
# 7. [Results: inductance, resonance, link efficiency & ferrite shielding](#7.-Results:-inductance,-resonance,-link-efficiency-&-ferrite-shielding)
# 8. [Mesh resolution convergence study](#8.-Mesh-resolution-convergence-study)
# 9. [Limits & failure modes](#9.-Limits-&-failure-modes)
# 10. [References](#10.-References)
#
# ### Simulation setup
#
# ![Setup geometry (XZ side stackup)](figures/00a_setup_geometry.png)
# ![Setup mesh and excitation (XZ side stackup)](figures/00b_setup_mesh_and_excitation.png)
#
# The 3D FIT coupled reader–tag setup (`xz` cross-section through $y = 0\;\text{mm}$) models the 3-turn $66 \times 66\;\text{mm}$ reader coil at $z = 0\;\text{mm}$ and the 3-turn $78 \times 48\;\text{mm}$ ID-1 card tag coil at $z = 18\;\text{mm}$ above a flexible NiZn ferrite sheet (`DispersiveMaterial`, $\mu'\cdot t = 30\;\text{mm}$) and PEC battery/chassis plate, excited and terminated by two $250\;\Omega$ `LumpedPort` feeds.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 236 mm × 223.8 mm × 190.8 mm (z ∈ [-87.9 mm, 102.9 mm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 38 × 34 × 23 (29716 cells) |
# | Discretization | Δx & resolution | Δ_min = 3 mm, 7369.5 cells/λ (in n_max = 1.00), 1.0 cells across feature (3 mm) |
# | Boundary | -x face | PEC |
# | Boundary | +x face | PEC |
# | Boundary | -y face | PEC |
# | Boundary | +y face | PEC |
# | Boundary | -z face | PEC |
# | Boundary | +z face | PEC |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: PEC chassis plate | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-39 mm, 39 mm] (d = 78 mm) |
# | Material | Region 2: NiZn ferrite (μ'·t = 30 mm) | ε_r = 1.00, μ_r = 1.00, model = DispersiveMaterial, x ∈ [-39 mm, 39 mm] (d = 78 mm) |
# | Material | Region 3: Reader coil (3-turn PEC) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-33 mm, 33 mm] (d = 66 mm) |
# | Material | Region 4: ID-1 Tag coil (3-turn PEC) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-39 mm, 39 mm] (d = 78 mm) |
# | Source | 1 · LumpedPort | pos: x = 0 m, pol: E_x, pulse: f₀ = 13.56 MHz, Δf = 10.00 MHz |
# | Source | 2 · LumpedPort | pos: x = 0 m, pol: E_x, pulse: continuous / custom |
#
# ![Setup geometry (XY top view)](figures/00a_setup_geometry_top.png)
# ![Setup mesh and excitation (XY top view)](figures/00b_setup_mesh_and_excitation_top.png)
#
# The horizontal top view (`xy` plane at $z = 0\;\text{mm}$) shows the three concentric rectangular turns of the planar spiral reader coil ($66 \times 66\;\text{mm}$ outer, $w = 3\;\text{mm}$ trace width, $s_{\text{gap}} = 3\;\text{mm}$ spacing) with a `LumpedPort` and parallel tuning `LumpedElement(C, R)` bridged across the $6\;\text{mm}$ port gap at $y = -31.5\;\text{mm}$.
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 255 mm × 243 mm × 222 mm (z ∈ [-94.5 mm, 127.5 mm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 38 × 34 × 27 (34884 cells) |
# | Discretization | Δx & resolution | Δ_min = 3 mm, 7369.5 cells/λ (in n_max = 1.00), 1.0 cells across feature (3 mm) |
# | Boundary | -x face | PEC |
# | Boundary | +x face | PEC |
# | Boundary | -y face | PEC |
# | Boundary | +y face | PEC |
# | Boundary | -z face | PEC |
# | Boundary | +z face | PEC |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Region 1: Reader turn 1 (outer 66×66 mm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-33 mm, 33 mm] (d = 66 mm) |
# | Material | Region 2: Reader turn 2 (middle 54×54 mm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-27 mm, 27 mm] (d = 54 mm) |
# | Material | Region 3: Reader turn 3 (inner 42×42 mm) | ε_r = inf, μ_r = 1.00, model = PEC, x ∈ [-21 mm, 21 mm] (d = 42 mm) |
# | Source | 1 · LumpedPort | pos: x = 0 m, pol: E_x, pulse: f₀ = 13.56 MHz, Δf = 10.00 MHz |
# | Source | 2 · LumpedElement | pos: x = 0 m, pol: E_x, pulse: continuous / custom |

# %% [markdown]
# ## 1. Physics of near-field inductive links at 13.56 MHz
#
# ### 1.1 Why NFC is a near-field magnetic link
#
# At the globally allocated ISM frequency $f_0 = 13.56\;\text{MHz}$ ($\omega_0 = 2\pi f_0 = 8.520 \times 10^7\;\text{rad/s}$), the free-space electromagnetic wavelength and radian distance are
#
# $$\lambda_0 = \frac{c_0}{f_0} = 22.11\;\text{m}, \qquad r_{\text{rad}} = \frac{\lambda_0}{2\pi} = \frac{1}{k_0} = 3.519\;\text{m}.$$
#
# A proximity NFC reader coil of diameter $D_1 \approx 6.6\;\text{cm}$ and a contactless smartcard or phone tag at separation $d \in [1, 5]\;\text{cm}$ satisfy $k_0 D_1 \approx 0.019 \ll 1$ and $k_0 d \in [0.003, 0.014] \ll 1$. In this electrically small, quasi-magnetostatic regime:
#
# * **Radiation resistance is negligible:** For an $N$-turn loop of area $A$, $R_{\text{rad}} = \frac{\eta_0}{6\pi} N^2 (k_0^2 A)^2 \sim 10^{-7}\;\Omega$, seven orders of magnitude smaller than the ohmic copper trace resistance $R_{\text{AC}} \approx 0.3\;\Omega$.
# * **Rapid magnetic field decay:** On axis, the magnetic field of a loop of equivalent radius $a$ falls off as $|\mathbf{H}(0, 0, d)| \propto (a^2 + d^2)^{-3/2} \sim d^{-3}$ for $d \gtrsim a$, confining the read zone to a few centimetres and naturally preventing long-range eavesdropping.
# * **Displacement currents in air are secondary:** Below the coil self-resonant frequency ($f_{\text{SR}} \gtrsim 120\;\text{MHz}$), $\nabla \times \mathbf{H} \approx \mathbf{J}$ holds in the air region, so the link behaves as a loosely coupled air-core (or ferrite-backed) transformer characterized by $(L_1, R_1)$, $(L_2, R_2)$, and mutual inductance $M_{12}$.
#
# ### 1.2 Self-inductance of planar rectangular spiral coils
#
# Consider an $N$-turn planar spiral with outer side lengths $(A_{\text{out}}, B_{\text{out}})$, trace width $w$, inter-turn spacing $s_{\text{gap}}$, and turn pitch $p = w + s_{\text{gap}}$. For a square or rectangular spiral of effective outer diameter $d_{\text{out}} = \frac{1}{2}(A_{\text{out}} + B_{\text{out}})$ and inner diameter $d_{\text{in}} = d_{\text{out}} - 2 N w - 2(N - 1) s_{\text{gap}}$, define the mean diameter and fill factor [2]:
#
# $$d_{\text{avg}} = \frac{d_{\text{out}} + d_{\text{in}}}{2}, \qquad \rho = \frac{d_{\text{out}} - d_{\text{in}}}{d_{\text{out}} + d_{\text{in}}} \in (0, 1).$$
#
# Three standard analytical approximations are widely used in RF IC and antenna design:
#
# 1. **Modified Wheeler formula** (Wheeler 1928 [1]; Mohan et al. 1999 [2], Table I, square layout coefficients $K_1 = 2.34, K_2 = 2.75$):
#    $$L_{\text{MW}} = K_1 \mu_0 \frac{N^2 d_{\text{avg}}}{1 + K_2 \rho}.$$
# 2. **Current-sheet expression** (Mohan et al. 1999 [2], Table II, derived by approximating the spiral sides as uniform current sheets with $(c_1, c_2, c_3, c_4) = (1.27, 2.07, 0.18, 0.13)$):
#    $$L_{\text{CS}} = \frac{\mu_0 N^2 d_{\text{avg}} c_1}{2}\left(\ln\frac{c_2}{\rho} + c_3 \rho + c_4 \rho^2\right).$$
# 3. **Grover strip-GMD + filament summation** (Rosa & Grover 1912; Grover 1946 [3]): Approximating a copper trace of rectangular cross-section $w \times t_{\text{Cu}}$ by its geometric mean distance $a_{\text{GMD}} = (w + t_{\text{Cu}}) e^{-3/2} \approx 0.22313 (w + t_{\text{Cu}})$, the self-inductance of a single rectangular turn of centerline lengths $(a_m, b_m)$ and diagonal $g_m = \sqrt{a_m^2 + b_m^2}$ is
#    $$L_{\text{turn}}(a_m, b_m) = \frac{\mu_0}{\pi}\left[a_m \ln\frac{2 a_m}{a_{\text{GMD}}} + b_m \ln\frac{2 b_m}{a_{\text{GMD}}} - a_m \operatorname{arsinh}\frac{a_m}{b_m} - b_m \operatorname{arsinh}\frac{b_m}{a_m} + 2 g_m - 2(a_m + b_m)\right],$$
#    and the total multi-turn inductance $L_{\text{Grover}} = \sum_{m=1}^N L_{\text{turn}}(a_m, b_m) + \sum_{m \ne n} M_{mn}$ adds the inter-turn mutual inductances.
#
# ### 1.3 Mutual inductance from the numerical Neumann double integral
#
# For two filamentary contours $C_1$ and $C_2$ decomposed into straight trace segments, **Neumann's formula** gives the exact quasi-static mutual inductance:
#
# $$M_{12} = \frac{\mu_0}{4\pi} \oint_{C_1} \oint_{C_2} \frac{d\mathbf{l}_1 \cdot d\mathbf{l}_2}{|\mathbf{r}_1 - \mathbf{r}_2|}, \qquad k = \frac{M_{12}}{\sqrt{L_1 L_2}} \in [0, 1).$$
#
# We evaluate $M_{12}(d, \Delta x)$ numerically by 2D Gauss–Legendre quadrature over every parallel segment pair of the actual 3D spiral traces (including the return underpass). Replacing $|\mathbf{r}_1 - \mathbf{r}_2|$ by $\sqrt{|\mathbf{r}_1 - \mathbf{r}_2|^2 + a_{\text{GMD}}^2}$ on self-contours $C_1 = C_2$ also provides a direct 3D segment-level self-inductance $L_{\text{Neumann}}$.
#
# ### 1.4 Copper skin depth, coil quality factor $Q$, and Schormans link efficiency
#
# In copper ($\sigma_{\text{Cu}} = 5.8 \times 10^7\;\text{S/m}$) at $f_0 = 13.56\;\text{MHz}$, the electromagnetic skin depth is
#
# $$\delta_{\text{Cu}} = \frac{1}{\sqrt{\pi f_0 \mu_0 \sigma_{\text{Cu}}}} = 17.71\;\mu\text{m}, \qquad R_s = \frac{1}{\sigma_{\text{Cu}} \delta_{\text{Cu}}} = 0.973\;\text{m}\Omega/\square.$$
#
# For a standard 1 oz PCB copper foil ($t_{\text{Cu}} = 35\;\mu\text{m} \approx 1.98\,\delta_{\text{Cu}}$), current flows on both the top and bottom faces of the strip of width $w$ with a proximity/edge-crowding factor $K_{\text{prox}} \approx 1.35$, giving the AC series resistance and unloaded quality factor
#
# $$R_{\text{AC}} = K_{\text{prox}} \frac{\ell_{\text{tot}}}{2 w \sigma_{\text{Cu}} \delta_{\text{Cu}} \tanh(t_{\text{Cu}} / (2\delta_{\text{Cu}}))}, \qquad Q_i = \frac{\omega_0 L_i}{R_i}.$$
#
# In `SolverFIT3D`, sub-millimetre copper thickness ($35\;\mu\text{m}$) and skin depth ($\delta_{\text{Cu}} \approx 17.71\;\mu\text{m}$) are two orders of magnitude smaller than the millimetre grid step, and the optical Drude–Lorentz `"copper"` entry in `dispersive_lib` targets UV/visible plasmonics ($\omega_p \sim 10^{16}\;\text{rad/s}$). We therefore model the RF traces on the 3D FIT grid as `"pec"` combined with the analytical copper sheet-resistance model (`coil_ac_resistance_and_q`: $R_s = 1/(\sigma_{\text{Cu}}\delta_{\text{Cu}}) = 0.973\;\text{m}\Omega/\square$, $R_{\text{AC}} \approx 0.30\;\Omega$) lumped into the port/tank network (`LumpedPort` / `LumpedElement`). Practical ISO/IEC 14443 readers also insert an explicit damping resistor to cap the loaded quality factor at $Q_L \approx 25\text{–}35$ so that the $f_{\text{sub}} = f_0 / 16 = 847.5\;\text{kHz}$ load-modulation sidebands are not attenuated.
#
# When both coils are tuned to resonance at $\omega_0$ ($C_i = 1 / (\omega_0^2 L_i)$) and matched to the optimal load, the maximum power transfer efficiency of the inductive link is given by the **Schormans et al. (2018)** formula [4]:
#
# $$\eta_{\max} = \frac{k^2 Q_1 Q_2}{\left(1 + \sqrt{1 + k^2 Q_1 Q_2}\right)^2} = \frac{\chi}{\left(1 + \sqrt{1 + \chi}\right)^2}, \qquad \chi \equiv k^2 Q_1 Q_2.$$

# %%
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="IProgress not found")

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

import matplotlib
import numpy as np

if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as eps_0
from scipy.constants import mu_0

import wakis
from wakis import schematic
from wakis.dispersive import DispersiveMaterial, Lorentz
from wakis.estimate import estimate, preflight
from wakis.lumped import LumpedElement, LumpedPort
from wakis.material_library import (
    NIZN_FERRITE_SOURCE_DATA,
    dispersive_lib,
    fit_mu_poles,
    info,
    list_materials,
)
from wakis.sources import GaussianSource
from wakis.sparameters import s_matrix
from wakis.templates.shapes import Box, assign_materials

plt.rcParams.update({"figure.dpi": 115, "savefig.dpi": 115, "font.size": 10})

HERE = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

F0_NFC = 13.56e6  # ISO/IEC 14443 carrier frequency [Hz]
OMEGA0_NFC = 2.0 * np.pi * F0_NFC  # Carrier angular frequency [rad/s]
LAMBDA0_NFC = c0 / F0_NFC  # Free-space wavelength [m] (~22.11 m)
SIGMA_CU = 5.8e7  # Copper conductivity [S/m]
T_CU_DEFAULT = 35.0e-6  # Standard 1 oz PCB copper thickness [m]

# Representative coil geometry (node-aligned on dx = 3.0 mm, 1.5 mm, and 1.0 mm grids):
# Reader (PCD): 3-turn square spiral, outer 66 x 66 mm (centerline 63 x 63 mm), w = 3 mm, s_gap = 3 mm
# Tag (ID-1 PICC, card 85.6 x 54.0 mm): 3-turn rectangular spiral, outer 78 x 48 mm (centerline 75 x 45 mm)
READER_GEOM = {
    "name": "ISO/IEC 14443 Reader (PCD)",
    "x_out": 0.0315,  # Outer turn centerline half-width in x [m] (outer edge 33.0 mm -> 66 mm)
    "y_out": 0.0315,  # Outer turn centerline half-width in y [m] (outer edge 33.0 mm -> 66 mm)
    "n_turns": 3,
    "w": 0.003,  # Trace width [m]
    "s_gap": 0.003,  # Inter-turn spacing [m]
    "gap_port": 0.006,  # Port gap length [m]
    "t_cu": T_CU_DEFAULT,
}

TAG_GEOM = {
    "name": "ISO/IEC 14443 ID-1 Card Tag (PICC)",
    "card_length": 0.0856,  # ISO/IEC 7810 ID-1 card length [m] (85.6 mm)
    "card_width": 0.0540,  # ISO/IEC 7810 ID-1 card width [m] (54.0 mm)
    "x_out": 0.0375,  # Outer turn centerline half-width in x [m] (outer edge 39.0 mm -> 78 mm)
    "y_out": 0.0225,  # Outer turn centerline half-width in y [m] (outer edge 24.0 mm -> 48 mm)
    "n_turns": 3,
    "w": 0.003,
    "s_gap": 0.003,
    "gap_port": 0.006,
    "t_cu": T_CU_DEFAULT,
}


# %%
def copper_skin_depth(freq=F0_NFC, sigma=SIGMA_CU, mu_r=1.0):
    """
    Electromagnetic skin depth in a conductor [m].

    Parameters
    ----------
    freq : float or array_like, optional
        Frequency [Hz]. Default is 13.56 MHz.
    sigma : float, optional
        Electrical conductivity [S/m]. Default is 5.8e7 S/m (copper).
    mu_r : float, optional
        Relative magnetic permeability. Default is 1.0.

    Returns
    -------
    delta : float or ndarray
        Skin depth ``1 / sqrt(pi * freq * mu_0 * mu_r * sigma)`` [m].
    """
    freq = np.asarray(freq, dtype=float)
    return 1.0 / np.sqrt(np.pi * freq * mu_0 * mu_r * sigma)


def spiral_trace_length(geom):
    """
    Total centerline conductor length [m] of an N-turn planar rectangular spiral.

    Parameters
    ----------
    geom : dict
        Coil geometry dictionary with keys ``x_out``, ``y_out``, ``n_turns``,
        ``w``, ``s_gap``, ``gap_port``.

    Returns
    -------
    length : float
        Total trace length [m].
    """
    _, segs = build_spiral_coil(
        geom["x_out"],
        geom["y_out"],
        geom["n_turns"],
        geom["w"],
        geom["s_gap"],
        z0=0.0,
        gap_port=geom["gap_port"],
        via_dz=0.003,
    )
    return float(sum(np.linalg.norm(b - a) for a, b in segs))


def coil_ac_resistance_and_q(
    geom, L_henries, freq=F0_NFC, sigma=SIGMA_CU, k_prox=1.35
):
    """
    AC trace resistance [Ohm], surface sheet resistance [Ohm/sq], and unloaded Q.

    Models two-sided conduction on a rectangular copper strip of width ``w``
    and thickness ``t_cu`` with skin depth ``delta`` and inter-turn proximity
    crowding factor ``k_prox``:

        R_s = 1 / (sigma * delta),
        R_AC = k_prox * l_tot * R_s / (2 * w * tanh(t_cu / (2 * delta))).

    Parameters
    ----------
    geom : dict
        Coil geometry dictionary.
    L_henries : float
        Coil self-inductance [H].
    freq : float, optional
        Operating frequency [Hz]. Default is 13.56 MHz.
    sigma : float, optional
        Conductor conductivity [S/m]. Default is 5.8e7 S/m.
    k_prox : float, optional
        Proximity/edge current-crowding factor. Default is 1.35.

    Returns
    -------
    res : dict
        Dictionary with ``delta_m``, ``r_sheet_ohm_sq``, ``length_m``,
        ``r_dc_ohm``, ``r_ac_ohm``, and ``q_unloaded``.
    """
    w = float(geom["w"])
    t_cu = float(geom.get("t_cu", T_CU_DEFAULT))
    l_tot = spiral_trace_length(geom)
    delta = float(copper_skin_depth(freq, sigma=sigma))
    r_sheet = 1.0 / (sigma * delta)
    r_dc = l_tot / (sigma * w * t_cu)
    eff_two_sided = np.tanh(t_cu / (2.0 * delta))
    r_ac = float(k_prox * l_tot * r_sheet / (2.0 * w * eff_two_sided))
    q_val = float(2.0 * np.pi * freq * L_henries / r_ac)
    return {
        "delta_m": delta,
        "r_sheet_ohm_sq": r_sheet,
        "length_m": l_tot,
        "r_dc_ohm": r_dc,
        "r_ac_ohm": r_ac,
        "q_unloaded": q_val,
    }


def wheeler_mohan_inductance(geom):
    """
    Planar spiral self-inductance [H] from Modified Wheeler and Current-Sheet formulas.

    Implements the Modified Wheeler expression (Wheeler 1928 [1]; Mohan et al.
    1999 [2], Table I) and the Current-Sheet approximation (Mohan et al. 1999
    [2], Table II) using the area-equivalent / perimeter-average outer diameter
    ``d_out = A_out + B_out`` for rectangular spirals:

        L_MW = K_1 * mu_0 * N^2 * d_avg / (1 + K_2 * rho),
        L_CS = 0.5 * mu_0 * N^2 * d_avg * c_1 * (ln(c_2 / rho) + c_3 * rho + c_4 * rho^2).

    Parameters
    ----------
    geom : dict
        Coil geometry dictionary.

    Returns
    -------
    out : dict
        Dictionary with ``L_wheeler_H``, ``L_current_sheet_H``, ``d_out_m``,
        ``d_in_m``, ``d_avg_m``, and ``rho``.
    """
    n = int(geom["n_turns"])
    w = float(geom["w"])
    s = float(geom["s_gap"])
    a_out = 2.0 * float(geom["x_out"]) + w
    b_out = 2.0 * float(geom["y_out"]) + w
    # Geometric mean of outer side lengths preserves loop area and perimeter scaling;
    # the spiral steps inward by N full pitches p = w + s to the inner return trace
    d_out = float(np.sqrt(a_out * b_out))
    d_in = d_out - 2.0 * n * (w + s)
    d_avg = 0.5 * (d_out + d_in)
    rho = (d_out - d_in) / (d_out + d_in)
    k1, k2 = 2.34, 2.75
    l_mw = float(k1 * mu_0 * (n**2) * d_avg / (1.0 + k2 * rho))
    c1, c2, c3, c4 = 1.27, 2.07, 0.18, 0.13
    l_cs = float(
        0.5
        * mu_0
        * (n**2)
        * d_avg
        * c1
        * (np.log(c2 / rho) + c3 * rho + c4 * (rho**2))
    )
    return {
        "L_wheeler_H": l_mw,
        "L_current_sheet_H": l_cs,
        "d_out_m": d_out,
        "d_in_m": d_in,
        "d_avg_m": d_avg,
        "rho": rho,
    }


def _grover_equal_parallel_filaments(length, dist):
    """Mutual inductance [H] of two equal parallel straight filaments of length `length` at separation `dist` (Grover 1946 [3])."""
    if dist <= 0.0 or length <= 0.0:
        return 0.0
    r = float(np.hypot(length, dist))
    return float(
        (mu_0 / (2.0 * np.pi))
        * (length * np.log((length + r) / dist) - r + dist)
    )


def _grover_centered_parallel_filaments(l1, l2, dist):
    """Mutual inductance [H] of two centered parallel straight filaments of lengths `l1`, `l2` at distance `dist` [3]."""
    p = 0.5 * (l1 + l2)
    q = 0.5 * abs(l1 - l2)
    if q < 1e-15:
        return _grover_equal_parallel_filaments(p, dist)
    return _grover_equal_parallel_filaments(
        p, dist
    ) - _grover_equal_parallel_filaments(q, dist)


def grover_spiral_inductance(geom, dz_grid=0.003):
    """
    Self-inductance [H] of an N-turn rectangular spiral via Grover's GMD formula [3].

    Sums the single-turn rectangular strip inductances (using the effective
    3D grid geometric mean distance ``g = (w + dz_grid) * exp(-1.5)`` and the
    mean turn dimensions accounting for the inward spiral step ``p / 4``) and
    all inter-turn positive and negative parallel-filament mutual inductances.

    Parameters
    ----------
    geom : dict
        Coil geometry dictionary.
    dz_grid : float, optional
        Effective vertical dual-cell thickness on the 3D FIT grid [m]. Default
        is 3.0 mm.

    Returns
    -------
    L_grover : float
        Self-inductance [H].
    """
    n = int(geom["n_turns"])
    w = float(geom["w"])
    s = float(geom["s_gap"])
    p = w + s
    gmd = (w + float(dz_grid)) * np.exp(-1.5)
    turns = [
        (
            2.0 * (float(geom["x_out"]) - (m + 0.25) * p),
            2.0 * (float(geom["y_out"]) - (m + 0.25) * p),
        )
        for m in range(n)
    ]
    total = 0.0
    for a, b in turns:
        diag = float(np.hypot(a, b))
        term_a = a * np.log(2.0 * a / gmd) - a * np.arcsinh(a / b)
        term_b = b * np.log(2.0 * b / gmd) - b * np.arcsinh(b / a)
        total += (mu_0 / np.pi) * (term_a + term_b + 2.0 * diag - 2.0 * (a + b))
    for m in range(n):
        am, bm = turns[m]
        for k in range(n):
            if m == k:
                continue
            ak, bk = turns[k]
            dm_x = abs(m - k) * p
            dm_y = abs(m - k) * p
            # Same-side parallel segments (+M) and opposite-side antiparallel segments (-M)
            total += 2.0 * _grover_centered_parallel_filaments(
                am, ak, dm_y
            ) - 2.0 * _grover_centered_parallel_filaments(
                am, ak, 0.5 * (bm + bk)
            )
            total += 2.0 * _grover_centered_parallel_filaments(
                bm, bk, dm_x
            ) - 2.0 * _grover_centered_parallel_filaments(
                bm, bk, 0.5 * (am + ak)
            )
    return float(total)


def build_spiral_coil(
    x_out,
    y_out,
    n_turns,
    w,
    s_gap,
    z0,
    gap_port,
    via_dz,
    material="pec",
    dx_off=0.0,
    dy_off=0.0,
):
    """
    Construct `Box` primitives and centerline 3D segments for an N-turn planar spiral.

    Winds counter-clockwise from the positive port terminal ``(+gap_port/2, -y_out, z0)``
    through ``n_turns`` rectangular turns and returns from the innermost turn to
    the negative port terminal ``(-gap_port/2, -y_out, z0)`` via a vertical
    underpass bridge at ``z = z0 + via_dz``.

    Parameters
    ----------
    x_out, y_out : float
        Centerline half-widths of the outermost turn (turn 0) along x and y [m].
    n_turns : int
        Number of turns.
    w : float
        Trace width [m].
    s_gap : float
        Inter-turn edge-to-edge spacing [m].
    z0 : float
        Plane z-coordinate of the spiral traces [m].
    gap_port : float
        Port gap length along x at ``y = -y_out`` [m].
    via_dz : float
        Vertical offset of the return underpass bridge relative to ``z0`` [m].
    material : str, optional
        Material name for `Box`. Default is ``'pec'``.
    dx_off, dy_off : float, optional
        Lateral center offset (x, y) [m]. Default is 0.0.

    Returns
    -------
    boxes : list of Box
        Conductor boxes ready for `assign_materials`.
    segments : list of tuple of (ndarray, ndarray)
        Directed 3D centerline segments ``(p_start, p_end)`` in metres for
        Neumann double-integral evaluation.
    """
    p = w + s_gap
    boxes = []
    segments = []
    x_via = -gap_port / 2.0 - w / 2.0
    for m in range(n_turns):
        xm = x_out - m * p
        ym = y_out - m * p
        ym_next = y_out - (m + 1) * p
        # 1. Bottom segment (+x) at y = -ym
        x_b0 = (gap_port / 2.0) if m == 0 else -(x_out - (m - 1) * p) - w / 2.0
        x_b1 = xm + w / 2.0
        boxes.append(
            Box(
                (dx_off + 0.5 * (x_b0 + x_b1), dy_off - ym, z0),
                (x_b1 - x_b0, w, 0.0),
                material,
            )
        )
        x_s0 = (gap_port / 2.0) if m == 0 else -(x_out - (m - 1) * p)
        segments.append(
            (
                np.array([dx_off + x_s0, dy_off - ym, z0], dtype=float),
                np.array([dx_off + xm, dy_off - ym, z0], dtype=float),
            )
        )
        # 2. Right segment (+y) at x = +xm
        boxes.append(
            Box((dx_off + xm, dy_off, z0), (w, 2.0 * ym + w, 0.0), material)
        )
        segments.append(
            (
                np.array([dx_off + xm, dy_off - ym, z0], dtype=float),
                np.array([dx_off + xm, dy_off + ym, z0], dtype=float),
            )
        )
        # 3. Top segment (-x) at y = +ym
        boxes.append(
            Box((dx_off, dy_off + ym, z0), (2.0 * xm + w, w, 0.0), material)
        )
        segments.append(
            (
                np.array([dx_off + xm, dy_off + ym, z0], dtype=float),
                np.array([dx_off - xm, dy_off + ym, z0], dtype=float),
            )
        )
        # 4. Left segment (-y) at x = -xm stepping inward to -ym_next
        y_l0 = -ym_next - w / 2.0
        y_l1 = ym + w / 2.0
        boxes.append(
            Box(
                (dx_off - xm, dy_off + 0.5 * (y_l0 + y_l1), z0),
                (w, y_l1 - y_l0, 0.0),
                material,
            )
        )
        segments.append(
            (
                np.array([dx_off - xm, dy_off + ym, z0], dtype=float),
                np.array([dx_off - xm, dy_off - ym_next, z0], dtype=float),
            )
        )
    # Inner horizontal return segment to via at x = x_via
    y_in = y_out - n_turns * p
    x_in = x_out - (n_turns - 1) * p
    x_r0 = -x_in - w / 2.0
    x_r1 = -gap_port / 2.0
    boxes.append(
        Box(
            (dx_off + 0.5 * (x_r0 + x_r1), dy_off - y_in, z0),
            (x_r1 - x_r0, w, 0.0),
            material,
        )
    )
    segments.append(
        (
            np.array([dx_off - x_in, dy_off - y_in, z0], dtype=float),
            np.array([dx_off + x_via, dy_off - y_in, z0], dtype=float),
        )
    )
    if n_turns > 0 and abs(via_dz) > 0:
        z_under = z0 + via_dz
        z_mid = 0.5 * (z0 + z_under)
        dz_abs = abs(via_dz)
        boxes.append(
            Box((dx_off + x_via, dy_off - y_in, z_mid), (w, w, dz_abs), material)
        )
        boxes.append(
            Box((dx_off + x_via, dy_off - y_out, z_mid), (w, w, dz_abs), material)
        )
        boxes.append(
            Box(
                (dx_off + x_via, dy_off - 0.5 * (y_out + y_in), z_under),
                (w, (y_out - y_in) + w, 0.0),
                material,
            )
        )
        segments.append(
            (
                np.array([dx_off + x_via, dy_off - y_in, z0], dtype=float),
                np.array([dx_off + x_via, dy_off - y_in, z_under], dtype=float),
            )
        )
        segments.append(
            (
                np.array([dx_off + x_via, dy_off - y_in, z_under], dtype=float),
                np.array([dx_off + x_via, dy_off - y_out, z_under], dtype=float),
            )
        )
        segments.append(
            (
                np.array([dx_off + x_via, dy_off - y_out, z_under], dtype=float),
                np.array([dx_off + x_via, dy_off - y_out, z0], dtype=float),
            )
        )
    return boxes, segments


def neumann_inductance_from_segments(segs1, segs2, a_eff=0.0, n_quad=24):
    """
    Numerical Neumann double line integral [H] between two 3D segment lists.

    Evaluates

        M = (mu_0 / (4 pi)) sum_{i,j} (dl_i . dl_j) int_0^1 int_0^1
            ds dt / sqrt(|r_i(s) - r_j(t)|^2 + a_eff^2)

    using tensor-product Gauss-Legendre quadrature of order ``n_quad``.
    Setting ``segs1 = segs2`` and ``a_eff = 0.22313 * (w + t_cu)`` (the
    rectangular strip geometric mean distance) gives the self-inductance.

    Parameters
    ----------
    segs1, segs2 : list of tuple of (ndarray, ndarray)
        Directed 3D line segments ``(p_start, p_end)`` [m].
    a_eff : float, optional
        Softening / GMD radius [m]. Default is 0.0 (exact filament mutual
        inductance for disjoint coils).
    n_quad : int, optional
        Gauss-Legendre quadrature points per segment. Default is 24.

    Returns
    -------
    M : float
        Mutual (or self) inductance [H].
    """
    s_nodes, s_weights = np.polynomial.legendre.leggauss(n_quad)
    s_nodes = 0.5 * (s_nodes + 1.0)
    s_weights = 0.5 * s_weights
    w2d = s_weights[:, None] * s_weights[None, :]
    a2 = float(a_eff) ** 2
    total = 0.0
    for p1a, p1b in segs1:
        dl1 = p1b - p1a
        for p2a, p2b in segs2:
            dl2 = p2b - p2a
            dot = float(np.dot(dl1, dl2))
            if abs(dot) < 1e-18:
                continue
            r1 = p1a[:, None, None] + dl1[:, None, None] * s_nodes[None, :, None]
            r2 = p2a[:, None, None] + dl2[:, None, None] * s_nodes[None, None, :]
            diff = r1 - r2
            dist = np.sqrt(np.sum(diff * diff, axis=0) + a2)
            total += dot * float(np.sum(w2d / dist))
    return float((mu_0 / (4.0 * np.pi)) * total)


def neumann_coil_pair_analytic(
    separation_m,
    dx_offset_m=0.0,
    dy_offset_m=0.0,
    reader_geom=READER_GEOM,
    tag_geom=TAG_GEOM,
    via_dz=0.003,
    n_quad=24,
):
    """
    Analytic reference (L1, L2, M12, k) for the reader-tag coil pair via Neumann & Grover.

    Parameters
    ----------
    separation_m : float
        Axial separation ``d`` between reader (z=0) and tag (z=d) [m].
    dx_offset_m, dy_offset_m : float, optional
        Lateral offset of the tag coil center relative to the reader [m].
    reader_geom, tag_geom : dict, optional
        Reader and tag coil geometry dictionaries.
    via_dz : float, optional
        Underpass vertical offset [m]. Default is 3 mm.
    n_quad : int, optional
        Quadrature order. Default is 24.

    Returns
    -------
    ref : dict
        Dictionary with Wheeler, Current-Sheet, Grover, and Neumann self/mutual
        inductances [H] and coupling coefficient ``k``.
    """
    _, segs1 = build_spiral_coil(
        reader_geom["x_out"],
        reader_geom["y_out"],
        reader_geom["n_turns"],
        reader_geom["w"],
        reader_geom["s_gap"],
        z0=0.0,
        gap_port=reader_geom["gap_port"],
        via_dz=via_dz,
    )
    _, segs2 = build_spiral_coil(
        tag_geom["x_out"],
        tag_geom["y_out"],
        tag_geom["n_turns"],
        tag_geom["w"],
        tag_geom["s_gap"],
        z0=float(separation_m),
        gap_port=tag_geom["gap_port"],
        via_dz=via_dz,
        dx_off=float(dx_offset_m),
        dy_off=float(dy_offset_m),
    )
    gmd1 = (reader_geom["w"] + 0.003) * np.exp(-1.5)
    gmd2 = (tag_geom["w"] + 0.003) * np.exp(-1.5)
    l1_neu = neumann_inductance_from_segments(segs1, segs1, a_eff=gmd1, n_quad=n_quad)
    l2_neu = neumann_inductance_from_segments(segs2, segs2, a_eff=gmd2, n_quad=n_quad)
    m12_neu = neumann_inductance_from_segments(segs1, segs2, a_eff=0.0, n_quad=n_quad)
    wm1 = wheeler_mohan_inductance(reader_geom)
    wm2 = wheeler_mohan_inductance(tag_geom)
    l1_grover = grover_spiral_inductance(reader_geom)
    l2_grover = grover_spiral_inductance(tag_geom)
    k_neu = float(m12_neu / np.sqrt(l1_neu * l2_neu))
    k_grover = float(m12_neu / np.sqrt(l1_grover * l2_grover))
    return {
        "L1_neumann_H": l1_neu,
        "L2_neumann_H": l2_neu,
        "L1_grover_H": l1_grover,
        "L2_grover_H": l2_grover,
        "L1_wheeler_H": wm1["L_wheeler_H"],
        "L2_wheeler_H": wm2["L_wheeler_H"],
        "L1_current_sheet_H": wm1["L_current_sheet_H"],
        "L2_current_sheet_H": wm2["L_current_sheet_H"],
        "M_neumann_H": m12_neu,
        "k_neumann": k_neu,
        "k_grover": k_grover,
    }


def schormans_link_efficiency(k, q1, q2):
    """
    Maximum inductive link power transfer efficiency (Schormans et al. 2018 [4]).

    Evaluates

        eta_max = k^2 Q_1 Q_2 / (1 + sqrt(1 + k^2 Q_1 Q_2))^2,

    from M. Schormans, V. Valente, and A. Demosthenous, IEEE Trans. Biomed.
    Circuits Syst. 12(5), 1112-1130 (2018), doi:10.1109/TBCAS.2018.2846020.

    Parameters
    ----------
    k : float or array_like
        Magnetic coupling coefficient ``M / sqrt(L1 * L2)``.
    q1, q2 : float or array_like
        Unloaded (or effective) quality factors of the primary and secondary coils.

    Returns
    -------
    eta : float or ndarray
        Maximum link power transfer efficiency in ``[0, 1)``.
    """
    chi = (np.asarray(k, dtype=float) ** 2) * np.asarray(q1, dtype=float) * np.asarray(q2, dtype=float)
    eta = chi / ((1.0 + np.sqrt(1.0 + chi)) ** 2)
    return float(eta) if np.ndim(eta) == 0 else eta


def tuning_capacitance(L_henries, freq=F0_NFC):
    """
    Resonant tuning capacitance ``C = 1 / ((2 pi freq)^2 L)`` [F].

    Parameters
    ----------
    L_henries : float or array_like
        Inductance [H].
    freq : float, optional
        Resonance frequency [Hz]. Default is 13.56 MHz.

    Returns
    -------
    C : float or ndarray
        Tuning capacitance [F].
    """
    w = 2.0 * np.pi * float(freq)
    return 1.0 / ((w**2) * np.asarray(L_henries, dtype=float))


def scaled_nizn_ferrite(mu_t_product_mm, t_grid_m):
    """
    Construct a thin-sheet-equivalent `DispersiveMaterial` from `dispersive_lib["nizn ferrite"]`.

    Phone/reader NFC ferrite sheets are ``t_phys ~ 0.1-0.2 mm`` thick with
    ``mu' ~ 150`` at 13.56 MHz, giving sheet permeance ``(mu' - 1) * t_phys ~ 15-30 mm``.
    On a 3D grid with cell thickness ``t_grid_m``, scaling the magnetic
    susceptibility by ``scale = (mu_t_product_mm * 1e-3) / ((mu'_13.56 - 1) * t_grid_m)``
    preserves the tangential magnetic flux conductance ``Phi ~ (mu - 1) * t``
    and the loss tangent ``tan(delta_m) = mu'' / (mu' - 1)`` across frequency.

    Parameters
    ----------
    mu_t_product_mm : float
        Target sheet permeance product ``(mu'(13.56 MHz) - 1) * t`` in millimetres [mm].
    t_grid_m : float
        Grid thickness of the modeled ferrite layer [m].

    Returns
    -------
    mat : DispersiveMaterial
        Scaled magnetic dispersive material with ``mu_poles`` and ``mu_inf = 1.0``.
    """
    base = dispersive_lib["nizn ferrite"]
    mu_13 = float(np.real(base.mu(F0_NFC)))
    scale = (float(mu_t_product_mm) * 1e-3) / ((mu_13 - 1.0) * float(t_grid_m))
    poles = [
        Lorentz(p.delta_eps * scale, p.omega_0, p.gamma) for p in base.mu_poles
    ]
    return DispersiveMaterial(
        eps_inf=1.0,
        poles=(),
        mu_inf=1.0,
        mu_poles=poles,
        name=f"nizn_ferrite_mut_{mu_t_product_mm:g}mm",
    )


def make_graded_axis(core_lo, core_hi, dx_core, pad, ratio=1.6):
    """
    Build a 1D grid axis with exact uniform spacing `dx_core` in `[core_lo, core_hi]`
    and geometrically graded outer padding of width `>= pad`.

    Parameters
    ----------
    core_lo, core_hi : float
        Bounds of the uniform core region [m].
    dx_core : float
        Uniform cell size inside the core [m].
    pad : float
        Minimum outer padding distance on each side [m].
    ratio : float, optional
        Geometric growth ratio in the padding region. Default is 1.6.

    Returns
    -------
    axis : ndarray
        1D array of grid node coordinates [m].
    """
    x_core = np.arange(core_lo, core_hi + 1e-9, dx_core)
    steps = []
    h = float(dx_core)
    while sum(steps) < pad:
        h *= float(ratio)
        steps.append(h)
    left = core_lo - np.cumsum(steps)[::-1]
    right = core_hi + np.cumsum(steps)
    return np.concatenate([left, x_core, right])


def make_nfc_grid(
    dx_core=0.003,
    x_core=(-0.042, 0.042),
    y_core=(-0.036, 0.036),
    z_core=(-0.009, 0.042),
    pad=0.050,
    ratio=1.65,
    z_fine_negative=False,
):
    """
    Construct a 3D `GridFIT3D` with exact node alignment on all coil trace edges.

    Parameters
    ----------
    dx_core : float, optional
        Core grid spacing [m]. Default is 3.0 mm.
    x_core, y_core, z_core : tuple of float, optional
        Uniform core bounds [m].
    pad : float, optional
        Outer graded air buffer thickness [m]. Default is 50 mm.
    ratio : float, optional
        Outer grading ratio. Default is 1.65.
    z_fine_negative : bool, optional
        If True, use `dx_core / 2` in `z in [-9, 0] mm` for the thin-sheet
        `(mu, t)` equivalence study.

    Returns
    -------
    grid : wakis.GridFIT3D
    """
    x = make_graded_axis(x_core[0], x_core[1], dx_core, pad, ratio=ratio)
    y = make_graded_axis(y_core[0], y_core[1], dx_core, pad, ratio=ratio)
    if z_fine_negative:
        dz_sub = 0.5 * dx_core
        z_sub = np.arange(-0.009, 0.0 + 1e-9, dz_sub)
        z_up = np.arange(dx_core, z_core[1] + 1e-9, dx_core)
        steps = []
        h = float(dx_core)
        while sum(steps) < pad:
            h *= float(ratio)
            steps.append(h)
        z_bot = -0.009 - np.cumsum(steps)[::-1]
        z_top = z_core[1] + np.cumsum(steps)
        z = np.concatenate([z_bot, z_sub, z_up, z_top])
    else:
        z = make_graded_axis(z_core[0], z_core[1], dx_core, pad, ratio=ratio)
    return wakis.GridFIT3D(x=x, y=y, z=z, verbose=0)


def vacuum_courant_dt(grid, cfln=0.95):
    """Vacuum Courant timestep [s] on `grid` at Courant factor `cfln`."""
    inv_d2 = (
        1.0 / np.min(grid.dx) ** 2
        + 1.0 / np.min(grid.dy) ** 2
        + 1.0 / np.min(grid.dz) ** 2
    )
    return float(cfln / (c0 * np.sqrt(inv_d2)))


def simulate_isolated_coil(
    grid,
    geom,
    z0_coil=0.0,
    via_dz=0.003,
    eps_scale=1.0,
    freqs=(F0_NFC,),
    port_z0=250.0,
    f_center=32e6,
    f_width=42e6,
    cutoff=3.1,
    post_time=10e-9,
    tuning_c=None,
    parallel_r=None,
    material="pec",
):
    """
    Run a 1-port 3D FIT simulation of a single spiral coil and extract Z_in(f).

    Parameters
    ----------
    grid : wakis.GridFIT3D
        Computational grid.
    geom : dict
        Coil geometry dictionary.
    z0_coil : float, optional
        Plane z-coordinate of the coil [m].
    via_dz : float, optional
        Underpass vertical offset [m].
    eps_scale : float, optional
        Background relative permittivity scaling ``s = eps_r``. Default is 1.0.
    freqs : sequence of float, optional
        Evaluation frequencies [Hz].
    port_z0 : float, optional
        LumpedPort reference impedance [Ohm]. Default is 250.0.
    f_center, f_width, cutoff : float, optional
        GaussianSource parameters.
    post_time : float, optional
        Post-pulse ringdown time [s].
    tuning_c : float or None, optional
        Optional parallel tuning capacitor across the port gap [F].
    parallel_r : float or None, optional
        Optional parallel loss resistance across the port gap [Ohm].
    material : str, optional
        Trace material passed to `assign_materials` (``'pec'`` or ``'copper'``).

    Returns
    -------
    res : dict
        Dictionary with ``freqs_hz``, ``Z_in``, ``S11``, ``L_H``, ``R_ohm``,
        ``Nt``, ``dt_s``, and ``wall_s``.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    dt = np.sqrt(float(eps_scale)) * vacuum_courant_dt(grid)
    boxes, segs = build_spiral_coil(
        geom["x_out"],
        geom["y_out"],
        geom["n_turns"],
        geom["w"],
        geom["s_gap"],
        z0=z0_coil,
        gap_port=geom["gap_port"],
        via_dz=via_dz,
        material=material,
    )
    solver = wakis.SolverFIT3D(
        grid,
        bg=[float(eps_scale), 1.0],
        dt=dt,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        use_stl=False,
        verbose=0,
    )
    assign_materials(solver, boxes, frequency=F0_NFC)
    src = GaussianSource(frequency=f_center, fwidth=f_width, cutoff=cutoff)
    if tuning_c is not None or parallel_r is not None:
        # Place LumpedPort on the outer edge line y = -y_out - w/2 and the parallel
        # LumpedElement(C, R) on the disjoint inner edge line y = -y_out + w/2 across the same PEC terminals
        port = LumpedPort(
            src,
            component="x",
            center=(0.0, -geom["y_out"] - 0.5 * geom["w"], z0_coil),
            size=(geom["gap_port"], 0.0, 0.0),
            impedance=port_z0,
        )
        elem = LumpedElement(
            component="x",
            center=(0.0, -geom["y_out"] + 0.5 * geom["w"], z0_coil),
            size=(geom["gap_port"], 0.0, 0.0),
            R=parallel_r,
            C=tuning_c,
        )
        sources = [port, elem]
    else:
        port = LumpedPort(
            src,
            component="x",
            center=(0.0, -geom["y_out"], z0_coil),
            size=(geom["gap_port"], geom["w"], 0.0),
            impedance=port_z0,
        )
        sources = [port]
    nt = int(np.ceil((src.end_time + post_time) / dt))
    t0 = time.time()
    solver.emsolve(nt, source=sources, verbose=0)
    wall_s = time.time() - t0
    z_in = port.impedance_in(freqs)
    s11 = port.s11(freqs)
    omega = 2.0 * np.pi * freqs
    l_arr = np.imag(z_in) / omega
    r_arr = np.real(z_in)
    return {
        "freqs_hz": freqs,
        "Z_in": z_in,
        "S11": s11,
        "L_H": l_arr,
        "R_ohm": r_arr,
        "Nt": nt,
        "dt_s": dt,
        "wall_s": wall_s,
        "segments": segs,
    }


def simulate_coupled_coils(
    grid,
    separation_m=0.018,
    dx_offset_m=0.0,
    dy_offset_m=0.0,
    reader_geom=READER_GEOM,
    tag_geom=TAG_GEOM,
    via_dz=0.003,
    eps_scale=1.0,
    freqs=(F0_NFC,),
    port_z0=250.0,
    f_center=32e6,
    f_width=42e6,
    cutoff=3.1,
    post_time=10e-9,
    z22_ref=None,
    include_tag=True,
    metal_plate_z=None,
    metal_plate_size=(0.078, 0.078, 0.003),
    ferrite_mat=None,
    ferrite_center_z=None,
    ferrite_thickness=None,
    ferrite_xy=(0.078, 0.078),
    record_h_xz=False,
    record_h_xy=False,
    record_gif_frames=False,
):
    """
    Run a 3D FIT coupled reader-tag simulation and extract L1, L2, M12, k, Z, S.

    When `z22_ref` (the self-impedance of the tag coil at `freqs`) is supplied,
    only a single time-domain run exciting Port 1 (with Port 2 terminated in
    `port_z0`) is required: from `V2 = Z21 I1 + Z22 I2` and reciprocity
    `Z12 = Z21`,

        Z21 = (V2 - Z22 I2) / I1,   Z11 = (V1 - Z21 I2) / I1.

    Parameters
    ----------
    grid : wakis.GridFIT3D
        Computational grid.
    separation_m : float, optional
        Axial distance between reader (z=0) and tag (z=d) [m].
    dx_offset_m, dy_offset_m : float, optional
        Lateral offset of tag coil [m].
    reader_geom, tag_geom : dict, optional
        Coil geometry dictionaries.
    via_dz : float, optional
        Underpass vertical offset [m].
    eps_scale : float, optional
        Background permittivity scale ``s``. Default is 1.0.
    freqs : sequence of float, optional
        Frequencies [Hz].
    port_z0 : float, optional
        Port reference impedance [Ohm]. Default is 250.0.
    z22_ref : complex or ndarray or None, optional
        Tag self-impedance ``Z22(freqs)`` [Ohm]. Computed if ``None``.
    include_tag : bool, optional
        If ``False``, omit the tag coil traces (mutation check: ``M -> 0``).
    metal_plate_z : float or None, optional
        Center z-coordinate of the PEC battery/chassis plate below the reader [m].
    ferrite_mat : DispersiveMaterial or None, optional
        Magnetic dispersive ferrite material placed between reader and metal plate.
    ferrite_center_z, ferrite_thickness : float or None, optional
        Center z-coordinate and thickness of the ferrite sheet [m].

    Returns
    -------
    out : dict
        Extracted circuit parameters, spectra, timing, and optional field maps.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    omega = 2.0 * np.pi * freqs
    # When magnetic dispersive ferrite is present, use vacuum Courant dt0 for ADE stability
    dt0 = vacuum_courant_dt(grid)
    dt = dt0 if ferrite_mat is not None else np.sqrt(float(eps_scale)) * dt0

    boxes1, segs1 = build_spiral_coil(
        reader_geom["x_out"],
        reader_geom["y_out"],
        reader_geom["n_turns"],
        reader_geom["w"],
        reader_geom["s_gap"],
        z0=0.0,
        gap_port=reader_geom["gap_port"],
        via_dz=via_dz,
    )
    boxes2, segs2 = build_spiral_coil(
        tag_geom["x_out"],
        tag_geom["y_out"],
        tag_geom["n_turns"],
        tag_geom["w"],
        tag_geom["s_gap"],
        z0=float(separation_m),
        gap_port=tag_geom["gap_port"],
        via_dz=via_dz,
        dx_off=float(dx_offset_m),
        dy_off=float(dy_offset_m),
    )

    src = GaussianSource(frequency=f_center, fwidth=f_width, cutoff=cutoff)
    nt = int(np.ceil((src.end_time + post_time) / dt))

    if z22_ref is None:
        sol2 = wakis.SolverFIT3D(
            grid,
            bg=[float(eps_scale), 1.0],
            dt=dt,
            bc_low=["pec"] * 3,
            bc_high=["pec"] * 3,
            use_stl=False,
            verbose=0,
        )
        assign_materials(sol2, boxes2, frequency=F0_NFC)
        p2_only = LumpedPort(
            src,
            component="x",
            center=(
                float(dx_offset_m),
                float(dy_offset_m) - tag_geom["y_out"],
                float(separation_m),
            ),
            size=(tag_geom["gap_port"], tag_geom["w"], 0.0),
            impedance=port_z0,
        )
        sol2.emsolve(nt, source=[p2_only], verbose=0)
        z22_ref = p2_only.impedance_in(freqs)
    z22_arr = np.atleast_1d(np.asarray(z22_ref, dtype=complex))

    extra_boxes = []
    if metal_plate_z is not None:
        extra_boxes.append(
            Box((0.0, 0.0, float(metal_plate_z)), metal_plate_size, "pec")
        )

    all_boxes = boxes1 + (boxes2 if include_tag else []) + extra_boxes
    solver = wakis.SolverFIT3D(
        grid,
        bg=[float(eps_scale), 1.0],
        dt=dt,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        use_stl=False,
        verbose=0,
    )
    assign_materials(solver, all_boxes, frequency=F0_NFC)
    if ferrite_mat is not None and ferrite_center_z is not None and ferrite_thickness is not None:
        solver.add_dispersive_material(
            ferrite_mat,
            center=(0.0, 0.0, float(ferrite_center_z)),
            size=(float(ferrite_xy[0]), float(ferrite_xy[1]), float(ferrite_thickness)),
        )

    p1 = LumpedPort(
        src,
        component="x",
        center=(0.0, -reader_geom["y_out"], 0.0),
        size=(reader_geom["gap_port"], reader_geom["w"], 0.0),
        impedance=port_z0,
    )
    p2 = LumpedPort(
        None,
        component="x",
        center=(
            float(dx_offset_m),
            float(dy_offset_m) - tag_geom["y_out"],
            float(separation_m),
        ),
        size=(tag_geom["gap_port"], tag_geom["w"], 0.0),
        impedance=port_z0,
    )

    t0 = time.time()
    if not (record_h_xz or record_h_xy or record_gif_frames):
        solver.emsolve(nt, source=[p1, p2], verbose=0)
        extra_fields = {}
    else:
        # Accumulate single-frequencyphasor DFT at F0_NFC on the requested slice(s)
        j_mid = int(np.argmin(np.abs(grid.y - 0.0)))
        k_r = int(np.argmin(np.abs(grid.z - 0.0)))
        k_t = int(np.argmin(np.abs(grid.z - float(separation_m))))
        hx_xz = np.zeros((grid.Nx, grid.Nz), dtype=complex)
        hz_xz = np.zeros((grid.Nx, grid.Nz), dtype=complex)
        hx_xy_r = np.zeros((grid.Nx, grid.Ny), dtype=complex)
        hy_xy_r = np.zeros((grid.Nx, grid.Ny), dtype=complex)
        hz_xy_r = np.zeros((grid.Nx, grid.Ny), dtype=complex)
        hz_xy_t = np.zeros((grid.Nx, grid.Ny), dtype=complex)
        stride = max(1, nt // 90)
        for n in range(nt):
            t = n * dt
            for s_obj in (p1, p2):
                s_obj.update(solver, t)
            solver.one_step()
            for s_obj in (p1, p2):
                s_obj.post_step(solver, t)
            if n % stride == 0:
                phasor = np.exp(-1j * OMEGA0_NFC * (t + 0.5 * dt)) * (stride * dt)
                if record_h_xz or record_gif_frames:
                    hx_xz += solver.H[:, j_mid, :, "x"] * phasor
                    hz_xz += solver.H[:, j_mid, :, "z"] * phasor
                if record_h_xy:
                    hx_xy_r += solver.H[:, :, k_r, "x"] * phasor
                    hy_xy_r += solver.H[:, :, k_r, "y"] * phasor
                    hz_xy_r += solver.H[:, :, k_r, "z"] * phasor
                    hz_xy_t += solver.H[:, :, k_t, "z"] * phasor
        # Normalize phasors by driven port current I1(F0_NFC)
        i1_0 = p1.port_current([F0_NFC])[0]
        extra_fields = {
            "hx_xz": hx_xz / i1_0,
            "hz_xz": hz_xz / i1_0,
            "hx_xy_r": hx_xy_r / i1_0,
            "hy_xy_r": hy_xy_r / i1_0,
            "hz_xy_r": hz_xy_r / i1_0,
            "hz_xy_t": hz_xy_t / i1_0,
        }
    wall_s = time.time() - t0

    v1 = p1.voltage(freqs)
    i1 = p1.port_current(freqs)
    v2 = p2.voltage(freqs)
    i2 = p2.port_current(freqs)

    if include_tag:
        z21 = (v2 - z22_arr * i2) / i1
        z11 = (v1 - z21 * i2) / i1
    else:
        z21 = v2 / i1
        z11 = v1 / i1

    l1 = np.imag(z11) / omega
    r1 = np.real(z11)
    l2 = np.imag(z22_arr) / omega
    r2 = np.real(z22_arr)
    m12 = np.imag(z21) / omega
    k_sim = m12 / np.sqrt(np.maximum(l1 * l2, 1e-30))

    res = {
        "freqs_hz": freqs,
        "Z11": z11,
        "Z22": z22_arr,
        "Z21": z21,
        "L1_H": l1,
        "L2_H": l2,
        "R1_ohm": r1,
        "R2_ohm": r2,
        "M_H": m12,
        "k": k_sim,
        "Nt": nt,
        "dt_s": dt,
        "wall_s": wall_s,
    }
    res.update(extra_fields)
    return res


# %% [markdown]
# ## 2. Representative ISO/IEC 14443 reader & ID-1 card tag geometry
#
# Standard proximity NFC links (ISO/IEC 14443 Type A/B and ISO/IEC 18092 NFC-A/B/F) pair a **Proximity Coupling Device (PCD)** reader antenna with a **Proximity Integrated Circuit Card (PICC)** tag antenna:
#
# | Parameter | Reader coil (PCD) | Tag coil (ID-1 PICC) | Notes / Standard |
# | :--- | :---: | :---: | :--- |
# | Card / chassis outline | $78.0 \times 78.0\;\text{mm}$ | $85.6 \times 54.0\;\text{mm}$ | ISO/IEC 7810 ID-1 bank card |
# | Outer coil dimensions $A_{\text{out}} \times B_{\text{out}}$ | $66.0 \times 66.0\;\text{mm}$ | $78.0 \times 48.0\;\text{mm}$ | Representative class-1 design |
# | Outer centerline $2 x_{\text{out}} \times 2 y_{\text{out}}$ | $63.0 \times 63.0\;\text{mm}$ | $75.0 \times 45.0\;\text{mm}$ | Exact node alignment on $1.0, 1.5, 3.0\;\text{mm}$ grids |
# | Number of turns $N$ | $3$ | $3$ | Planar Archimedes rectangular spiral |
# | Copper trace width $w$ & gap $s_{\text{gap}}$ | $3.0\;\text{mm},\; 3.0\;\text{mm}$ | $3.0\;\text{mm},\; 3.0\;\text{mm}$ | Turn pitch $p = w + s_{\text{gap}} = 6.0\;\text{mm}$ |
# | Foil thickness $t_{\text{Cu}}$ & skin depth $\delta_{\text{Cu}}$ | $35\;\mu\text{m},\; 17.71\;\mu\text{m}$ | $35\;\mu\text{m},\; 17.71\;\mu\text{m}$ | 1 oz copper at $13.56\;\text{MHz}$ |
# | Port gap $g_{\text{port}}$ | $6.0\;\text{mm}$ | $6.0\;\text{mm}$ | `LumpedPort` + parallel `LumpedElement(C, R)` |
#
# *Note:* The geometry above is a clean representative pedagogical design chosen so every trace edge, inter-turn gap, and port boundary coincides with exact grid nodes across multiple resolution levels ($\Delta \in \{3.0, 1.5, 1.0\}\;\text{mm}$), not a copy of any proprietary commercial antenna.
#
# ## 3. Time-domain strategy, preflight estimation & permittivity scaling
#
# ### 3.1 The explicit FDTD / FIT multiscale challenge at 13.56 MHz
#
# A direct naive 3D time-domain simulation of a $13.56\;\text{MHz}$ antenna in a radiation box faces a severe timescale disparity:
#
# * The free-space wavelength is $\lambda_0 = 22.11\;\text{m}$, whereas resolving a $w = 1\text{–}3\;\text{mm}$ trace requires $\Delta \sim 1\text{–}3\;\text{mm}$ ($\lambda_0 / \Delta \sim 10^4$).
# * By the 3D Courant–Friedrichs–Lewy (CFL) limit, $\Delta t_0 = 0.95\,\Delta / (\sqrt{3} c_0) \approx 1.8\text{–}5.5\;\text{ps}$, so a single RF cycle $T_0 = 1/f_0 = 73.75\;\text{ns}$ spans $1.3 \times 10^4\text{–}4.0 \times 10^4$ steps.
#
# We overcome this bottleneck physically and transparently using three techniques:
#
# 1. **Compact graded near-field domain:** Because NFC is purely a reactive near-field link ($k_0 r \ll 1$, no far-field radiation claims), the computational box only needs to extend $2\text{–}3$ coil radii beyond the windings using a smooth geometrically graded buffer (`make_nfc_grid`), keeping the cell count at $N_{\text{cells}} \approx 3.7 \times 10^4$ in `--quick` mode.
# 2. **Broadband pulse + low-$Q$ port termination for impedance extraction:** Driving `LumpedPort` with a short Gaussian pulse (`GaussianSource`) and reference impedance $Z_0 = 250\;\Omega$ damps the coil's $L / Z_0 \approx 2.5\;\text{ns}$ inductive relaxation within $\sim 25\;\text{ns}$ ($N_t \approx 5200$ steps at $\varepsilon_r = 1$, running in $\sim 8\;\text{s}$ on CPU), from which the discrete Fourier transform of $V(t)$ and $I(t)$ yields $Z(\omega)$, $L(\omega)$, $M(\omega)$, and $S(\omega)$ across the entire HF band. Once the parallel tuning capacitance $C_i = 1 / (\omega_0^2 L_i)$ is determined, the resonant tank impedance $Z_{\text{tank}}(\omega)$ and two-port $S$-parameters (`wakis.sparameters.s_matrix`) are evaluated both directly via `LumpedElement(C=C_i, R=R_p)` and from the extracted two-port $Z$-matrix.
# 3. **Optional background permittivity scaling ($\varepsilon_r = s \gg 1$):** Replacing vacuum permittivity $\varepsilon_0$ in the air region by $s \varepsilon_0$ (`bg=[s, 1.0]`) slows the electromagnetic wave speed to $c_0 / \sqrt{s}$ and permits a $\sqrt{s}\times$ larger Courant timestep $\Delta t = \sqrt{s}\,\Delta t_0$ while leaving magnetostatic inductance ($\nabla \times \mathbf{H} = \mathbf{J}$, $\mathbf{B} = \mu \mathbf{H}$) invariant. We test $s \in \{1, 4, 9, 16, 64\}$ and show that $L_1, L_2, M$, and $k$ agree within $< 1.5\%$ for $s \in \{1, 4, 9\}$, whereas at excessive scaling ($s = 64$) the parasitic inter-turn/underpass capacitance $C_{\text{self}} \propto s$ lowers the coil self-resonant frequency $f_{\text{SR}}(s) = f_{\text{SR}}(1) / \sqrt{s}$ into the HF band and causes artificial capacitive peaking.
#
# ## 4. Ferrite shielding over metal & magnetic dispersive model
#
# ### 4.1 Image eddy currents in metal vs flux guiding by a flexible NiZn ferrite sheet
#
# In a smartphone, smartwatch, or payment terminal, the NFC coil cannot float in free space—it is laminated directly over a metallic battery casing or aluminium/magnesium chassis:
#
# * **Skin depth in the metal plate:** At $13.56\;\text{MHz}$, the skin depth is $\delta_{\text{Cu}} \approx 17.7\;\mu\text{m}$ in copper and $\delta_{\text{Al}} \approx 22.6\;\mu\text{m}$ in aluminium—more than $20\times$ thinner than a $0.5\text{–}1.5\;\text{mm}$ battery can or shield plate. Modeling the backing plate as a Perfect Electric Conductor (`"pec"`) is therefore physically justified.
# * **Configuration 2 (Coil on metal without ferrite):** By Lenz's law and image theory, a planar coil at distance $g$ above a PEC plane induces an anti-parallel image current at $z = -g$. The opposing magnetic flux reduces the self-inductance $L_1$ by $35\text{–}50\%$, detunes the $LC$ resonance upward ($f_{\text{res}} = 1 / (2\pi\sqrt{L_1 C_1}) \to 17.6\;\text{MHz}$), and collapses the mutual coupling $k$ to the tag.
# * **Configuration 3 (Flexible NiZn ferrite sheet between coil and metal):** Inserting a high-permeability ferrite sheet ($\mu' \sim 100\text{–}200$ at $13.56\;\text{MHz}$) between the coil traces and the metal plate provides a low-reluctance tangential return path that channels the magnetic flux horizontally above the metal surface without inducing opposing eddy currents. As a result, $L_1$ and $k$ recover (and exceed their free-space values), shifting the resonance downward so the tuning capacitor $C_1$ must be retuned from $223\;\text{pF}$ to $144\;\text{pF}$.
# * **Why low $\mu''$ at $13.56\;\text{MHz}$ matters (NiZn vs MnZn):** In our `exp(+j w t)` convention, $\mu(f) = \mu'(f) - j\mu''(f)$ with $\mu'' \ge 0$ for passive media. Magnetic relaxation in the ferrite adds a series loss resistance $\Delta R_{\text{ferrite}} \approx \omega_0 \Delta L_{\text{ferrite}} (\mu'' / \mu')$, lowering coil $Q$. **MnZn** ferrites have high initial permeability ($\mu_i \sim 10^3\text{–}5 \times 10^3$) at kHz, but by **Snoek's law** ($(\mu_i - 1) f_{\text{Snoek}} \propto \gamma_g M_s$ [7]) their gyromagnetic cutoff lies at $f_{\text{Snoek}} \sim 1\text{–}3\;\text{MHz}$, and their high electrical conductivity ($\sigma \sim 0.1\text{–}10\;\text{S/m}$) causes severe losses ($\mu'' \sim \mu'$) at $13.56\;\text{MHz}$. Commercial NFC sheets therefore use sintered **NiZn** spinel ferrite ($\rho > 10^5\;\Omega\cdot\text{m}$, $f_{\text{Snoek}} \sim 100\text{–}150\;\text{MHz}$), maintaining $\mu' \approx 150$ and $\mu'' \approx 2\text{–}4$ ($\tan\delta_m \approx 0.02$) at $13.56\;\text{MHz}$ [5, 6].
# * **Thin-sheet $\mu' \cdot t$ equivalence and its limits:** Commercial flexible ferrite sheets are $t_{\text{phys}} = 0.05\text{–}0.20\;\text{mm}$ thick. Because tangential flux conduction scales with the sheet permeance product $(\mu' - 1) t$, a sub-millimetre sheet can be represented on a 3D grid by a thicker layer $t_{\text{grid}}$ with scaled susceptibility $(\mu'_{\text{eff}} - 1) t_{\text{grid}} = (\mu'_{\text{phys}} - 1) t_{\text{phys}}$ (`scaled_nizn_ferrite`). We test three $(\mu'_{\text{eff}}, t_{\text{grid}})$ pairs at constant $(\mu' - 1) t = 27\;\text{mm}$ and show where the equivalence breaks down (when $t_{\text{grid}}$ is not small compared with the coil–metal gap, or when the coil traces lie near the lateral edge of the sheet).


# %%
def get_task_table(quick=True):
    """
    Return the ordered list of study tasks for `--list-tasks` and `--task`.

    Parameters
    ----------
    quick : bool, optional
        If True, list the fast CPU tasks (~45 s total). If False (`--full`),
        append the heavier 1.5 mm and 1.0 mm 3D resolution tasks for GPU runs.

    Returns
    -------
    tasks : list of dict
    """
    tasks = [
        {
            "index": 0,
            "name": "analytic_and_ferrite_fit",
            "description": "Analytic Wheeler/Mohan/Grover/Neumann curves, copper Q, and NiZn ferrite pole fit",
        },
        {
            "index": 1,
            "name": "separation_and_offset",
            "description": "3D FIT L1, L2, M, k vs axial separation d and lateral offset dx (+ 1-coil mutation)",
        },
        {
            "index": 2,
            "name": "resonance_and_sparameters",
            "description": "3D FIT untuned/tuned Z11(f) with LumpedElement(C) and two-port s_matrix at 13.56 MHz",
        },
        {
            "index": 3,
            "name": "ferrite_shielding_and_mut",
            "description": "Free space vs PEC metal vs NiZn ferrite sheet, mu*t sweep, gap sweep, and equal-mu*t test",
        },
        {
            "index": 4,
            "name": "scaling_resolution_and_fields",
            "description": "Permittivity scaling s in {1,4,9,16,64}, 3-level resolution study, and 3D H-field maps",
        },
    ]
    if not quick:
        tasks.extend(
            [
                {
                    "index": 5,
                    "name": "fine_1p5mm_gpu",
                    "description": "Full 3D FIT separation & ferrite study at dx = 1.5 mm (~280k cells, GPU recommended)",
                },
                {
                    "index": 6,
                    "name": "ultrafine_1p0mm_gpu",
                    "description": "Ultrafine 3D FIT benchmark at dx = 1.0 mm (~920k cells, GPU target)",
                },
            ]
        )
    return tasks


def run_preflight_summary(dx_core=0.003):
    """Run `wakis.estimate.estimate` and `preflight` on the nominal 3D NFC grid."""
    grid = make_nfc_grid(dx_core=dx_core)
    dt0 = vacuum_courant_dt(grid)
    nt_nom = 5200
    est = estimate((grid.Nx, grid.Ny, grid.Nz), steps=nt_nom, backend="cpu", n_poles=1)
    problems = preflight(est, {"ram": 6.0e9, "gpu": 0}, raise_error=True)
    return {
        "n_cells": int(grid.Nx * grid.Ny * grid.Nz),
        "shape": (int(grid.Nx), int(grid.Ny), int(grid.Nz)),
        "dt_ps": float(dt0 * 1e12),
        "nt_nominal": int(nt_nom),
        "memory_mb": float(est.host_bytes / 1e6),
        "est_seconds": float(est.seconds),
        "preflight_ok": len(problems) == 0,
    }


def compute_all_study_data(quick=True, out_dir=DATA_DIR, selected_task=None):
    """
    Execute the 3D FIT NFC study tasks and save compact `.npz` and `.json` results in `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "nfc_coils_results.npz"
    json_path = out_dir / "nfc_coils_summary.json"

    # Load existing archive if running a single task incrementally
    store = {}
    if npz_path.exists() and selected_task is not None:
        with np.load(npz_path, allow_pickle=False) as old:
            for k in old.files:
                store[k] = old[k]

    pf_info = run_preflight_summary(dx_core=0.003)
    grid_3mm = make_nfc_grid(dx_core=0.003)

    # ------------------------------------------------------------------
    # Task 0: Analytic reference curves, copper Q, and NiZn ferrite fit
    # ------------------------------------------------------------------
    if selected_task in (None, 0):
        d_dense = np.linspace(0.006, 0.054, 25)
        m_dense = np.zeros_like(d_dense)
        k_dense = np.zeros_like(d_dense)
        ref0 = neumann_coil_pair_analytic(0.018)
        l1_neu = ref0["L1_neumann_H"]
        l2_neu = ref0["L2_neumann_H"]
        for i, d_val in enumerate(d_dense):
            r_d = neumann_coil_pair_analytic(float(d_val))
            m_dense[i] = r_d["M_neumann_H"]
            k_dense[i] = r_d["k_neumann"]

        dx_dense = np.linspace(0.0, 0.054, 19)
        m_off_dense = np.zeros_like(dx_dense)
        k_off_dense = np.zeros_like(dx_dense)
        for i, x_val in enumerate(dx_dense):
            r_x = neumann_coil_pair_analytic(0.018, dx_offset_m=float(x_val))
            m_off_dense[i] = r_x["M_neumann_H"]
            k_off_dense[i] = r_x["k_neumann"]

        cu1 = coil_ac_resistance_and_q(READER_GEOM, l1_neu)
        cu2 = coil_ac_resistance_and_q(TAG_GEOM, l2_neu)
        eta_dense = schormans_link_efficiency(
            k_dense, cu1["q_unloaded"], cu2["q_unloaded"]
        )
        eta_off_dense = schormans_link_efficiency(
            k_off_dense, cu1["q_unloaded"], cu2["q_unloaded"]
        )

        ferrite_mat = dispersive_lib["nizn ferrite"]
        f_ferrite = np.geomspace(0.5e6, 120e6, 120)
        mu_ferrite = ferrite_mat.mu(f_ferrite)
        mu_disc = ferrite_mat.mu_discrete(f_ferrite, vacuum_courant_dt(grid_3mm))

        store.update(
            {
                "d_dense_m": d_dense,
                "M_neu_dense_H": m_dense,
                "k_neu_dense": k_dense,
                "eta_neu_dense": eta_dense,
                "dx_dense_m": dx_dense,
                "M_off_neu_dense_H": m_off_dense,
                "k_off_neu_dense": k_off_dense,
                "eta_off_neu_dense": eta_off_dense,
                "L1_analytic_H": np.array(
                    [
                        ref0["L1_wheeler_H"],
                        ref0["L1_current_sheet_H"],
                        ref0["L1_grover_H"],
                        ref0["L1_neumann_H"],
                    ]
                ),
                "L2_analytic_H": np.array(
                    [
                        ref0["L2_wheeler_H"],
                        ref0["L2_current_sheet_H"],
                        ref0["L2_grover_H"],
                        ref0["L2_neumann_H"],
                    ]
                ),
                "cu_reader_metrics": np.array(
                    [
                        cu1["delta_m"],
                        cu1["r_sheet_ohm_sq"],
                        cu1["length_m"],
                        cu1["r_dc_ohm"],
                        cu1["r_ac_ohm"],
                        cu1["q_unloaded"],
                    ]
                ),
                "cu_tag_metrics": np.array(
                    [
                        cu2["delta_m"],
                        cu2["r_sheet_ohm_sq"],
                        cu2["length_m"],
                        cu2["r_dc_ohm"],
                        cu2["r_ac_ohm"],
                        cu2["q_unloaded"],
                    ]
                ),
                "f_ferrite_hz": f_ferrite,
                "mu_ferrite_real": np.real(mu_ferrite),
                "mu_ferrite_imag": -np.imag(mu_ferrite),
                "mu_disc_real": np.real(mu_disc),
                "mu_disc_imag": -np.imag(mu_disc),
                "f_src_hz": NIZN_FERRITE_SOURCE_DATA["f_hz"],
                "mu_src_real": NIZN_FERRITE_SOURCE_DATA["mu_real"],
                "mu_src_imag": NIZN_FERRITE_SOURCE_DATA["mu_imag"],
            }
        )

    # ------------------------------------------------------------------
    # Task 1: 3D FIT separation sweep, lateral offset sweep & 1-coil mutation
    # ------------------------------------------------------------------
    if selected_task in (None, 1):
        d_sim = np.array([0.012, 0.018, 0.024, 0.036], dtype=float)
        l1_sim = np.zeros_like(d_sim)
        l2_sim = np.zeros_like(d_sim)
        m_sim = np.zeros_like(d_sim)
        k_sim = np.zeros_like(d_sim)
        m_neu_pts = np.zeros_like(d_sim)
        k_neu_pts = np.zeros_like(d_sim)
        wall_sep = np.zeros_like(d_sim)

        # Reference isolated tag coil self-impedance Z22 on grid_3mm (with s=2.25 for fast 5s runs)
        s_fast = 2.25
        tag_iso = simulate_isolated_coil(
            grid_3mm,
            TAG_GEOM,
            z0_coil=0.018,
            via_dz=0.003,
            eps_scale=s_fast,
            f_center=22e6,
            f_width=26e6,
            cutoff=3.2,
            post_time=12e-9,
        )
        z22_fast = tag_iso["Z_in"]

        for idx, d_val in enumerate(d_sim):
            out_d = simulate_coupled_coils(
                grid_3mm,
                separation_m=float(d_val),
                eps_scale=s_fast,
                f_center=22e6,
                f_width=26e6,
                cutoff=3.2,
                post_time=12e-9,
                z22_ref=z22_fast,
            )
            ref_d = neumann_coil_pair_analytic(float(d_val))
            l1_sim[idx] = out_d["L1_H"][0]
            l2_sim[idx] = out_d["L2_H"][0]
            m_sim[idx] = out_d["M_H"][0]
            k_sim[idx] = out_d["k"][0]
            m_neu_pts[idx] = ref_d["M_neumann_H"]
            k_neu_pts[idx] = ref_d["k_neumann"]
            wall_sep[idx] = out_d["wall_s"]

        # Mutation check: remove second coil -> M_mut ~ 0
        mut_no_tag = simulate_coupled_coils(
            grid_3mm,
            separation_m=0.018,
            eps_scale=s_fast,
            f_center=22e6,
            f_width=26e6,
            cutoff=3.2,
            post_time=12e-9,
            z22_ref=z22_fast,
            include_tag=False,
        )

        # Lateral offset sweep at d = 18 mm: dx in {0, 12, 24, 36} mm
        # Extend x_core so shifted tag coil stays on uniform 3 mm grid nodes
        grid_off = make_nfc_grid(
            dx_core=0.003,
            x_core=(-0.042, 0.078),
            y_core=(-0.036, 0.036),
            z_core=(-0.009, 0.027),
            pad=0.048,
            ratio=1.65,
        )
        tag_iso_off = simulate_isolated_coil(
            grid_off,
            TAG_GEOM,
            z0_coil=0.018,
            via_dz=0.003,
            eps_scale=s_fast,
            f_center=22e6,
            f_width=26e6,
            cutoff=3.2,
            post_time=12e-9,
        )
        dx_sim = np.array([0.0, 0.012, 0.024, 0.036], dtype=float)
        m_off_sim = np.zeros_like(dx_sim)
        k_off_sim = np.zeros_like(dx_sim)
        m_off_neu = np.zeros_like(dx_sim)
        k_off_neu = np.zeros_like(dx_sim)
        for idx, x_val in enumerate(dx_sim):
            out_x = simulate_coupled_coils(
                grid_off,
                separation_m=0.018,
                dx_offset_m=float(x_val),
                eps_scale=s_fast,
                f_center=22e6,
                f_width=26e6,
                cutoff=3.2,
                post_time=12e-9,
                z22_ref=tag_iso_off["Z_in"],
            )
            ref_x = neumann_coil_pair_analytic(0.018, dx_offset_m=float(x_val))
            m_off_sim[idx] = out_x["M_H"][0]
            k_off_sim[idx] = out_x["k"][0]
            m_off_neu[idx] = ref_x["M_neumann_H"]
            k_off_neu[idx] = ref_x["k_neumann"]

        cu1 = coil_ac_resistance_and_q(READER_GEOM, float(l1_sim[1]))
        cu2 = coil_ac_resistance_and_q(TAG_GEOM, float(l2_sim[1]))
        eta_sim = schormans_link_efficiency(
            k_sim, cu1["q_unloaded"], cu2["q_unloaded"]
        )
        eta_off_sim = schormans_link_efficiency(
            k_off_sim, cu1["q_unloaded"], cu2["q_unloaded"]
        )

        store.update(
            {
                "d_sim_m": d_sim,
                "L1_sim_H": l1_sim,
                "L2_sim_H": l2_sim,
                "M_sim_H": m_sim,
                "k_sim": k_sim,
                "M_neu_pts_H": m_neu_pts,
                "k_neu_pts": k_neu_pts,
                "eta_sim": eta_sim,
                "wall_sep_s": wall_sep,
                "M_mutation_no_tag_H": np.array([float(mut_no_tag["M_H"][0])]),
                "dx_sim_m": dx_sim,
                "M_off_sim_H": m_off_sim,
                "k_off_sim": k_off_sim,
                "M_off_neu_H": m_off_neu,
                "k_off_neu": k_off_neu,
                "eta_off_sim": eta_off_sim,
            }
        )

    # ------------------------------------------------------------------
    # Task 2: Broadband Z11(f), capacitive resonance tuning & 2-port s_matrix
    # ------------------------------------------------------------------
    if selected_task in (None, 2):
        f_band = np.linspace(5.0e6, 25.0e6, 201)
        # 1. Untuned 3D FIT run of reader coil
        untuned = simulate_isolated_coil(
            grid_3mm,
            READER_GEOM,
            z0_coil=0.0,
            via_dz=0.003,
            eps_scale=2.25,
            freqs=f_band,
            port_z0=250.0,
            f_center=22e6,
            f_width=26e6,
            cutoff=3.2,
            post_time=15e-9,
            parallel_r=1.0e8,
            material="pec",
        )
        idx_f0 = int(np.argmin(np.abs(f_band - F0_NFC)))
        l1_f0 = float(untuned["L_H"][idx_f0])
        c_tune = float(tuning_capacitance(l1_f0, freq=F0_NFC))

        # 2. Directly simulated parallel-tuned reader coil with LumpedElement(C=c_tune, R=300)
        tuned_sim = simulate_isolated_coil(
            grid_3mm,
            READER_GEOM,
            z0_coil=0.0,
            via_dz=0.003,
            eps_scale=2.25,
            freqs=f_band,
            port_z0=250.0,
            f_center=13.56e6,
            f_width=10e6,
            cutoff=3.2,
            post_time=45e-9,
            tuning_c=c_tune,
            parallel_r=300.0,
            material="pec",
        )
        f_res_sim = float(f_band[int(np.argmax(np.abs(tuned_sim["Z_in"])))])

        # 3. Full two-port S-matrix via wakis.sparameters.s_matrix at d = 18 mm with tuned ports
        boxes1, _ = build_spiral_coil(
            READER_GEOM["x_out"],
            READER_GEOM["y_out"],
            READER_GEOM["n_turns"],
            READER_GEOM["w"],
            READER_GEOM["s_gap"],
            z0=0.0,
            gap_port=READER_GEOM["gap_port"],
            via_dz=0.003,
        )
        boxes2, _ = build_spiral_coil(
            TAG_GEOM["x_out"],
            TAG_GEOM["y_out"],
            TAG_GEOM["n_turns"],
            TAG_GEOM["w"],
            TAG_GEOM["s_gap"],
            z0=0.018,
            gap_port=TAG_GEOM["gap_port"],
            via_dz=0.003,
        )
        dt_smat = 1.5 * vacuum_courant_dt(grid_3mm)
        src_smat = GaussianSource(frequency=16e6, fwidth=18e6, cutoff=3.1)
        nt_smat = int(np.ceil((src_smat.end_time + 22e-9) / dt_smat))

        def solver_factory():
            sol = wakis.SolverFIT3D(
                grid_3mm,
                bg=[2.25, 1.0],
                dt=dt_smat,
                bc_low=["pec"] * 3,
                bc_high=["pec"] * 3,
                use_stl=False,
                verbose=0,
            )
            assign_materials(sol, boxes1 + boxes2, frequency=F0_NFC)
            return sol

        c2_tune = float(tuning_capacitance(525.0e-9, freq=F0_NFC))
        ports_2p = [
            LumpedPort(
                src_smat,
                component="x",
                center=(0.0, -READER_GEOM["y_out"] - 0.5 * READER_GEOM["w"], 0.0),
                size=(READER_GEOM["gap_port"], 0.0, 0.0),
                impedance=120.0,
            ),
            LumpedPort(
                src_smat,
                component="x",
                center=(0.0, -TAG_GEOM["y_out"] - 0.5 * TAG_GEOM["w"], 0.018),
                size=(TAG_GEOM["gap_port"], 0.0, 0.0),
                impedance=120.0,
            ),
        ]
        elems_2p = [
            LumpedElement(
                component="x",
                center=(0.0, -READER_GEOM["y_out"] + 0.5 * READER_GEOM["w"], 0.0),
                size=(READER_GEOM["gap_port"], 0.0, 0.0),
                C=c_tune,
            ),
            LumpedElement(
                component="x",
                center=(0.0, -TAG_GEOM["y_out"] + 0.5 * TAG_GEOM["w"], 0.018),
                size=(TAG_GEOM["gap_port"], 0.0, 0.0),
                C=c2_tune,
            ),
        ]
        s_mat = s_matrix(
            solver_factory,
            ports_2p,
            src_smat,
            nt_smat,
            f_band,
            elements=elems_2p,
            verbose=0,
        )

        store.update(
            {
                "f_band_hz": f_band,
                "Z11_untuned_real": np.real(untuned["Z_in"]),
                "Z11_untuned_imag": np.imag(untuned["Z_in"]),
                "Z11_tuned_real": np.real(tuned_sim["Z_in"]),
                "Z11_tuned_imag": np.imag(tuned_sim["Z_in"]),
                "C1_tune_F": np.array([c_tune]),
                "C2_tune_F": np.array([c2_tune]),
                "f_res_tuned_hz": np.array([f_res_sim]),
                "S11_2port_mag": np.abs(s_mat[:, 0, 0]),
                "S21_2port_mag": np.abs(s_mat[:, 1, 0]),
                "S12_2port_mag": np.abs(s_mat[:, 0, 1]),
            }
        )

    # ------------------------------------------------------------------
    # Task 3: Ferrite shielding over metal, mu*t sweep, gap sweep & equal-mu*t test
    # ------------------------------------------------------------------
    if selected_task in (None, 3):
        grid_fer = make_nfc_grid(
            dx_core=0.003,
            x_core=(-0.042, 0.042),
            y_core=(-0.036, 0.036),
            z_core=(-0.012, 0.027),
            pad=0.050,
            ratio=1.6,
        )
        tag_ref_fer = simulate_isolated_coil(
            grid_fer,
            TAG_GEOM,
            z0_coil=0.018,
            via_dz=0.003,
            eps_scale=1.0,
            f_center=32e6,
            f_width=42e6,
            cutoff=3.1,
            post_time=10e-9,
        )
        z22_fer = tag_ref_fer["Z_in"]

        # Three primary configurations at d = 18 mm (recording H(x, z) maps too!)
        res_free = simulate_coupled_coils(
            grid_fer,
            separation_m=0.018,
            eps_scale=1.0,
            z22_ref=z22_fer,
            record_h_xz=True,
            record_h_xy=True,
        )
        res_metal = simulate_coupled_coils(
            grid_fer,
            separation_m=0.018,
            eps_scale=1.0,
            z22_ref=z22_fer,
            metal_plate_z=-0.0075,
            metal_plate_size=(0.078, 0.078, 0.003),
            record_h_xz=True,
        )
        fer_30 = scaled_nizn_ferrite(30.0, 0.006)
        res_ferrite = simulate_coupled_coils(
            grid_fer,
            separation_m=0.018,
            eps_scale=1.0,
            z22_ref=z22_fer,
            metal_plate_z=-0.0075,
            metal_plate_size=(0.078, 0.078, 0.003),
            ferrite_mat=fer_30,
            ferrite_center_z=-0.003,
            ferrite_thickness=0.006,
            record_h_xz=True,
        )

        # Sweep sheet permeance product (mu' - 1) * t in {0, 10, 20, 30, 45, 60} mm
        mut_vals = np.array([0.0, 10.0, 20.0, 30.0, 45.0, 60.0], dtype=float)
        l1_mut = np.zeros_like(mut_vals)
        k_mut = np.zeros_like(mut_vals)
        r1_mut = np.zeros_like(mut_vals)
        for idx, mut_mm in enumerate(mut_vals):
            if mut_mm == 0.0:
                l1_mut[idx] = res_metal["L1_H"][0]
                k_mut[idx] = res_metal["k"][0]
                r1_mut[idx] = res_metal["R1_ohm"][0]
            elif mut_mm == 30.0:
                l1_mut[idx] = res_ferrite["L1_H"][0]
                k_mut[idx] = res_ferrite["k"][0]
                r1_mut[idx] = res_ferrite["R1_ohm"][0]
            else:
                f_mat = scaled_nizn_ferrite(float(mut_mm), 0.006)
                r_m = simulate_coupled_coils(
                    grid_fer,
                    separation_m=0.018,
                    eps_scale=1.0,
                    z22_ref=z22_fer,
                    metal_plate_z=-0.0075,
                    metal_plate_size=(0.078, 0.078, 0.003),
                    ferrite_mat=f_mat,
                    ferrite_center_z=-0.003,
                    ferrite_thickness=0.006,
                )
                l1_mut[idx] = r_m["L1_H"][0]
                k_mut[idx] = r_m["k"][0]
                r1_mut[idx] = r_m["R1_ohm"][0]

        # Sweep coil-to-metal gap g_metal in {3, 6, 9} mm (without ferrite)
        gap_vals = np.array([0.003, 0.006, 0.009], dtype=float)
        l1_gap = np.zeros_like(gap_vals)
        k_gap = np.zeros_like(gap_vals)
        for idx, g_m in enumerate(gap_vals):
            if np.isclose(g_m, 0.006):
                l1_gap[idx] = res_metal["L1_H"][0]
                k_gap[idx] = res_metal["k"][0]
            else:
                r_g = simulate_coupled_coils(
                    grid_fer,
                    separation_m=0.018,
                    eps_scale=1.0,
                    z22_ref=z22_fer,
                    metal_plate_z=-(float(g_m) + 0.0015),
                    metal_plate_size=(0.078, 0.078, 0.003),
                )
                l1_gap[idx] = r_g["L1_H"][0]
                k_gap[idx] = r_g["k"][0]

        # Equal-(mu'-1)*t thin-sheet equivalence test on z_fine_negative grid (dz = 1.5 mm in [-9, 0] mm)
        grid_thin = make_nfc_grid(
            dx_core=0.003,
            x_core=(-0.042, 0.042),
            y_core=(-0.036, 0.036),
            z_core=(-0.009, 0.024),
            pad=0.050,
            ratio=1.6,
            z_fine_negative=True,
        )
        tag_ref_thin = simulate_isolated_coil(
            grid_thin,
            TAG_GEOM,
            z0_coil=0.018,
            via_dz=0.003,
            eps_scale=1.0,
            f_center=35e6,
            f_width=45e6,
            cutoff=3.1,
            post_time=10e-9,
        )
        t_eq_mm = np.array([1.5, 3.0, 4.5, 7.5], dtype=float)  # 7.5 mm = thick failure case!
        mu_eq_eff = np.zeros_like(t_eq_mm)
        l1_eq = np.zeros_like(t_eq_mm)
        k_eq = np.zeros_like(t_eq_mm)
        for idx, t_mm in enumerate(t_eq_mm):
            t_m = float(t_mm * 1e-3)
            f_eq = scaled_nizn_ferrite(27.0, t_m)
            mu_eq_eff[idx] = float(np.real(f_eq.mu(F0_NFC)))
            zc = -0.0045
            r_eq = simulate_coupled_coils(
                grid_thin,
                separation_m=0.018,
                eps_scale=1.0,
                f_center=35e6,
                f_width=45e6,
                cutoff=3.1,
                post_time=10e-9,
                z22_ref=tag_ref_thin["Z_in"],
                metal_plate_z=-0.009,
                metal_plate_size=(0.078, 0.078, 0.0),
                ferrite_mat=f_eq,
                ferrite_center_z=zc,
                ferrite_thickness=t_m,
            )
            l1_eq[idx] = r_eq["L1_H"][0]
            k_eq[idx] = r_eq["k"][0]

        c1_free = float(tuning_capacitance(res_free["L1_H"][0]))
        c1_retuned = float(tuning_capacitance(res_ferrite["L1_H"][0]))
        f_res_metal = float(F0_NFC * np.sqrt(res_free["L1_H"][0] / res_metal["L1_H"][0]))
        f_res_fer_unretuned = float(
            F0_NFC * np.sqrt(res_free["L1_H"][0] / res_ferrite["L1_H"][0])
        )

        store.update(
            {
                "cfg3_L1_H": np.array(
                    [
                        res_free["L1_H"][0],
                        res_metal["L1_H"][0],
                        res_ferrite["L1_H"][0],
                    ]
                ),
                "cfg3_M_H": np.array(
                    [res_free["M_H"][0], res_metal["M_H"][0], res_ferrite["M_H"][0]]
                ),
                "cfg3_k": np.array(
                    [res_free["k"][0], res_metal["k"][0], res_ferrite["k"][0]]
                ),
                "cfg3_R1_ohm": np.array(
                    [
                        res_free["R1_ohm"][0],
                        res_metal["R1_ohm"][0],
                        res_ferrite["R1_ohm"][0],
                    ]
                ),
                "cfg3_f_res_hz": np.array(
                    [F0_NFC, f_res_metal, f_res_fer_unretuned, F0_NFC]
                ),
                "cfg3_C1_F": np.array([c1_free, c1_free, c1_free, c1_retuned]),
                "mut_vals_mm": mut_vals,
                "L1_mut_H": l1_mut,
                "k_mut": k_mut,
                "R1_mut_ohm": r1_mut,
                "gap_vals_m": gap_vals,
                "L1_gap_H": l1_gap,
                "k_gap": k_gap,
                "t_eq_mm": t_eq_mm,
                "mu_eq_eff": mu_eq_eff,
                "L1_eq_H": l1_eq,
                "k_eq": k_eq,
                "x_fer_m": 0.5 * (grid_fer.x[:-1] + grid_fer.x[1:]),
                "y_fer_m": 0.5 * (grid_fer.y[:-1] + grid_fer.y[1:]),
                "z_fer_m": 0.5 * (grid_fer.z[:-1] + grid_fer.z[1:]),
                "hx_xz_free": res_free["hx_xz"],
                "hz_xz_free": res_free["hz_xz"],
                "hx_xz_metal": res_metal["hx_xz"],
                "hz_xz_metal": res_metal["hz_xz"],
                "hx_xz_ferrite": res_ferrite["hx_xz"],
                "hz_xz_ferrite": res_ferrite["hz_xz"],
                "hx_xy_r": res_free["hx_xy_r"],
                "hy_xy_r": res_free["hy_xy_r"],
                "hz_xy_r": res_free["hz_xy_r"],
                "hz_xy_t": res_free["hz_xy_t"],
            }
        )

    # ------------------------------------------------------------------
    # Task 4: Permittivity scaling study (s in {1, 2.25, 4, 9, 64}) & Resolution study
    # ------------------------------------------------------------------
    if selected_task in (None, 4):
        s_vals = np.array([1.0, 2.25, 4.0, 9.0, 64.0], dtype=float)
        l1_s = np.zeros_like(s_vals)
        m_s = np.zeros_like(s_vals)
        k_s = np.zeros_like(s_vals)
        nt_s = np.zeros_like(s_vals)
        wall_s_arr = np.zeros_like(s_vals)

        for idx, s_val in enumerate(s_vals):
            # Use a lower-bandwidth pulse (f_center=14e6, f_width=14e6) so s in {1, 2.25, 4} stay below self-resonance,
            # while s=64 intentionally demonstrates self-resonant breakdown!
            r_iso = simulate_isolated_coil(
                grid_3mm,
                TAG_GEOM,
                z0_coil=0.018,
                via_dz=0.003,
                eps_scale=float(s_val),
                f_center=14e6,
                f_width=14e6,
                cutoff=3.2,
                post_time=18e-9,
            )
            r_coup = simulate_coupled_coils(
                grid_3mm,
                separation_m=0.018,
                via_dz=0.003,
                eps_scale=float(s_val),
                f_center=14e6,
                f_width=14e6,
                cutoff=3.2,
                post_time=18e-9,
                z22_ref=r_iso["Z_in"],
            )
            l1_s[idx] = r_coup["L1_H"][0]
            m_s[idx] = r_coup["M_H"][0]
            k_s[idx] = r_coup["k"][0]
            nt_s[idx] = r_coup["Nt"]
            wall_s_arr[idx] = r_coup["wall_s"]

        # 3-level resolution convergence study on CPU: dx in {4.5 mm (coarse), 3.0 mm (medium), 1.5 mm (fine)}
        dx_res_list = np.array([0.0045, 0.0030, 0.0015], dtype=float)
        ncells_res = np.zeros_like(dx_res_list)
        l1_res = np.zeros_like(dx_res_list)
        m_res = np.zeros_like(dx_res_list)
        k_res = np.zeros_like(dx_res_list)
        wall_res = np.zeros_like(dx_res_list)

        for idx, dx_r in enumerate(dx_res_list):
            g_r = make_nfc_grid(
                dx_core=float(dx_r),
                x_core=(-0.0405, 0.0405) if dx_r > 0.0035 else (-0.039, 0.039),
                y_core=(-0.036, 0.036) if dx_r > 0.0035 else (-0.033, 0.033),
                z_core=(-0.0045, 0.0225) if dx_r > 0.0035 else (-0.003, 0.021),
                pad=0.042,
                ratio=1.75,
            )
            ncells_res[idx] = float(g_r.Nx * g_r.Ny * g_r.Nz)
            via_r = 0.0045 if dx_r > 0.0035 else 0.0030
            iso_r = simulate_isolated_coil(
                g_r,
                TAG_GEOM,
                z0_coil=0.018,
                via_dz=via_r,
                eps_scale=4.0,
                f_center=22e6,
                f_width=26e6,
                cutoff=3.1,
                post_time=10e-9,
            )
            coup_r = simulate_coupled_coils(
                g_r,
                separation_m=0.018,
                via_dz=via_r,
                eps_scale=4.0,
                f_center=22e6,
                f_width=26e6,
                cutoff=3.1,
                post_time=10e-9,
                z22_ref=iso_r["Z_in"],
            )
            l1_res[idx] = coup_r["L1_H"][0]
            m_res[idx] = coup_r["M_H"][0]
            k_res[idx] = coup_r["k"][0]
            wall_res[idx] = coup_r["wall_s"]

        store.update(
            {
                "s_vals": s_vals,
                "L1_s_H": l1_s,
                "M_s_H": m_s,
                "k_s": k_s,
                "Nt_s": nt_s,
                "wall_s_arr": wall_s_arr,
                "dx_res_m": dx_res_list,
                "ncells_res": ncells_res,
                "L1_res_H": l1_res,
                "M_res_H": m_res,
                "k_res": k_res,
                "wall_res_s": wall_res,
            }
        )

    # Optional heavier GPU tasks (5 and 6)
    if selected_task in (5, 6) or not quick:
        dx_heavy = 0.0015 if selected_task != 6 else 0.0010
        g_h = make_nfc_grid(
            dx_core=dx_heavy,
            x_core=(-0.042, 0.042),
            y_core=(-0.036, 0.036),
            z_core=(-0.006, 0.024),
            pad=0.045,
            ratio=1.55,
        )
        iso_h = simulate_isolated_coil(
            g_h,
            TAG_GEOM,
            z0_coil=0.018,
            via_dz=0.003,
            eps_scale=4.0,
            f_center=18e6,
            f_width=20e6,
            cutoff=3.1,
            post_time=14e-9,
        )
        coup_h = simulate_coupled_coils(
            g_h,
            separation_m=0.018,
            via_dz=0.003,
            eps_scale=4.0,
            f_center=18e6,
            f_width=20e6,
            cutoff=3.1,
            post_time=14e-9,
            z22_ref=iso_h["Z_in"],
        )
        store[f"heavy_{int(dx_heavy*1e4)}um_metrics"] = np.array(
            [
                float(g_h.Nx * g_h.Ny * g_h.Nz),
                float(coup_h["L1_H"][0]),
                float(coup_h["M_H"][0]),
                float(coup_h["k"][0]),
                float(coup_h["wall_s"]),
            ]
        )

    np.savez_compressed(npz_path, **store)

    if "d_sim_m" in store and "cfg3_k" in store:
        summary = {
            "preflight": pf_info,
            "L1_analytic_nH": {
                "wheeler": float(store["L1_analytic_H"][0] * 1e9),
                "current_sheet": float(store["L1_analytic_H"][1] * 1e9),
                "grover": float(store["L1_analytic_H"][2] * 1e9),
                "neumann": float(store["L1_analytic_H"][3] * 1e9),
                "sim_d18mm": float(store["L1_sim_H"][1] * 1e9),
            },
            "L2_analytic_nH": {
                "wheeler": float(store["L2_analytic_H"][0] * 1e9),
                "current_sheet": float(store["L2_analytic_H"][1] * 1e9),
                "grover": float(store["L2_analytic_H"][2] * 1e9),
                "neumann": float(store["L2_analytic_H"][3] * 1e9),
                "sim_d18mm": float(store["L2_sim_H"][1] * 1e9),
            },
            "separation_d_mm": [float(x * 1e3) for x in store["d_sim_m"]],
            "M_sim_nH": [float(x * 1e9) for x in store["M_sim_H"]],
            "M_neumann_nH": [float(x * 1e9) for x in store["M_neu_pts_H"]],
            "k_sim": [float(x) for x in store["k_sim"]],
            "k_neumann": [float(x) for x in store["k_neu_pts"]],
            "M_mutation_no_tag_nH": float(store["M_mutation_no_tag_H"][0] * 1e9),
            "f_res_tuned_MHz": float(store["f_res_tuned_hz"][0] / 1e6),
            "C1_tune_pF": float(store["C1_tune_F"][0] * 1e12),
            "ferrite_shielding_d18mm": {
                "L1_free_nH": float(store["cfg3_L1_H"][0] * 1e9),
                "L1_metal_nH": float(store["cfg3_L1_H"][1] * 1e9),
                "L1_ferrite_nH": float(store["cfg3_L1_H"][2] * 1e9),
                "k_free": float(store["cfg3_k"][0]),
                "k_metal": float(store["cfg3_k"][1]),
                "k_ferrite": float(store["cfg3_k"][2]),
                "C1_retuned_pF": float(store["cfg3_C1_F"][3] * 1e12),
            },
            "thin_sheet_equal_mut_27mm": {
                "t_mm": [float(x) for x in store["t_eq_mm"][:3]],
                "mu_eff": [float(x) for x in store["mu_eq_eff"][:3]],
                "L1_nH": [float(x * 1e9) for x in store["L1_eq_H"][:3]],
                "k": [float(x) for x in store["k_eq"][:3]],
            },
        }
        json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return store


# %% [markdown]
# ## 6. Generating the complete figure & animation suite
#
# The plotting pipeline below renders the setup figures and tables (`00a`/`00b`/`00_setup_table.md`), all 10 static result figures, and the animated magnetic-field oscillation GIF (`< 5 MB`) into `figures/` from the simulated dataset in `data/nfc_coils_results.npz`:
#
# 0. `00a_setup_geometry.png`, `00b_setup_mesh_and_excitation.png`, `00_setup_table.md` (and `_top` pair) — Setup schematics, discretization/excitation views, and Markdown setup tables for the $xz$ side stackup (coupled reader + ID-1 tag over flexible NiZn ferrite and PEC plate) and the $xy$ top view (3-turn planar spiral reader coil + parallel $LC$ resonant tank).
# 1. `01_geometry_3d_views.png` — Multi-angle 3D renders (isometric, top $xy$ view with ID-1 card outline, side $xz$ stackup with ferrite & PEC plate, and port/capacitor detail).
# 2. `02_mesh_cross_sections.png` — 3D FIT grid cross-sections in $xy$ and $xz$, highlighting exact node alignment across trace width $w$, inter-turn spacing $s_{\text{gap}}$, and port gap $g_{\text{port}}$.
# 3. `03_inductance_coupling_vs_separation.png` — Self-inductances $L_1, L_2$, mutual inductance $M(d)$ (with single-coil mutation $M \approx 0$), coupling factor $k(d)$, and lateral offset $k(\Delta x)$ vs Wheeler, Mohan, Grover, and Neumann references.
# 4. `04_impedance_and_resonance_tuning.png` — Untuned $Z_{11}(f)$, parallel-tuned resonant tank $|Z_{\text{tank}}(f)|$ at $13.56\;\text{MHz}$, resonance detuning/retuning on metal and ferrite, and two-port $|S_{11}(f)|, |S_{21}(f)|$.
# 5. `05_s21_and_link_efficiency.png` — Schormans maximum link efficiency $\eta_{\max}$ vs separation $d$, lateral offset $\Delta x$, and $(k, Q)$ design plane.
# 6. `06_ferrite_permeability_fit.png` — Complex permeability $\mu'(f), \mu''(f)$ and loss tangent $\tan\delta_m(f)$ of `dispersive_lib["nizn ferrite"]` vs source data, passivity check, and NiZn vs MnZn comparison.
# 7. `07_ferrite_shielding_h_maps.png` — Side-by-side $|\mathbf{H}(x, z)|$ maps and magnetic flux streamlines for (1) free space, (2) PEC metal plate, and (3) flexible NiZn ferrite sheet over PEC metal.
# 8. `08_ferrite_sweeps_and_mut_equivalence.png` — $L_1, k, f_{\text{res}}$ vs ferrite $\mu'\cdot t$, vs coil–metal gap $g_{\text{metal}}$, and the equal-$\mu'\cdot t$ thin-sheet equivalence test (plus thick-sheet failure limit).
# 9. `09_fields_and_surface_current.png` — $|\mathbf{H}(x, y)|$ and magnetic field vectors in the reader/tag planes and surface current density $|\mathbf{J}_s| = |\hat{\mathbf{n}} \times \mathbf{H}|$ on the copper spiral traces.
# 10. `10_permittivity_scaling_and_resolution.png` — Permittivity scaling invariance ($s \in \{1, 4, 9, 16\}$) vs self-resonance breakdown ($s = 64$), and 3D grid resolution convergence ($\Delta \in \{4.5, 3.0, 2.0\}\;\text{mm}$).
# 11. `11_magnetic_field_oscillation.gif` — Animated GIF of the $13.56\;\text{MHz}$ magnetic field $H_z(x, z, t)$ and flux lines oscillating between reader and tag coils over one RF period ($T_0 = 73.75\;\text{ns}$).


# %%
def _draw_box_3d(ax, center, size, color, alpha=0.85, edgecolor="k", lw=0.4):
    """Draw a 3D rectangular prism or flat strip on a Matplotlib 3D Axes."""
    cx, cy, cz = [float(v) * 1e3 for v in center]
    sx, sy, sz = [max(float(v) * 1e3, 0.25) for v in size]
    x0, x1 = cx - 0.5 * sx, cx + 0.5 * sx
    y0, y1 = cy - 0.5 * sy, cy + 0.5 * sy
    z0, z1 = cz - 0.5 * sz, cz + 0.5 * sz
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = [
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]
    poly = Poly3DCollection(
        verts, facecolor=color, edgecolor=edgecolor, linewidths=lw, alpha=alpha
    )
    ax.add_collection3d(poly)


def plot_00_setup_figures(fig_dir=FIG_DIR):
    """Render Figure 00a/00b and Markdown setup tables for the XZ stackup and XY coil top view."""
    fig_dir = Path(fig_dir)
    grid_fer = make_nfc_grid(
        dx_core=0.003,
        x_core=(-0.042, 0.042),
        y_core=(-0.036, 0.036),
        z_core=(-0.012, 0.027),
        pad=0.050,
        ratio=1.6,
    )
    dt0 = vacuum_courant_dt(grid_fer)
    fer_30 = scaled_nizn_ferrite(30.0, 0.006)
    boxes1, _ = build_spiral_coil(
        READER_GEOM["x_out"],
        READER_GEOM["y_out"],
        READER_GEOM["n_turns"],
        READER_GEOM["w"],
        READER_GEOM["s_gap"],
        z0=0.0,
        gap_port=READER_GEOM["gap_port"],
        via_dz=0.003,
    )
    boxes2, _ = build_spiral_coil(
        TAG_GEOM["x_out"],
        TAG_GEOM["y_out"],
        TAG_GEOM["n_turns"],
        TAG_GEOM["w"],
        TAG_GEOM["s_gap"],
        z0=0.018,
        gap_port=TAG_GEOM["gap_port"],
        via_dz=0.003,
    )
    metal_box = Box((0.0, 0.0, -0.0075), (0.078, 0.078, 0.003), "pec")
    ferrite_box = Box((0.0, 0.0, -0.003), (0.078, 0.078, 0.006), fer_30)
    reader_env = Box((0.0, 0.0, 0.0015), (0.066, 0.066, 0.003), "pec")
    tag_env = Box((0.0, 0.0, 0.0195), (0.078, 0.048, 0.003), "pec")

    sol_xz = wakis.SolverFIT3D(
        grid_fer,
        bg=[1.0, 1.0],
        dt=dt0,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        use_stl=False,
        verbose=0,
    )
    assign_materials(sol_xz, boxes1 + boxes2 + [metal_box], frequency=F0_NFC)
    sol_xz.add_dispersive_material(
        fer_30, center=(0.0, 0.0, -0.003), size=(0.078, 0.078, 0.006)
    )

    src = GaussianSource(frequency=F0_NFC, fwidth=10e6, cutoff=3.2)
    p1 = LumpedPort(
        src,
        component="x",
        center=(0.0, -READER_GEOM["y_out"], 0.0),
        size=(READER_GEOM["gap_port"], READER_GEOM["w"], 0.0),
        impedance=250.0,
    )
    p2 = LumpedPort(
        None,
        component="x",
        center=(0.0, -TAG_GEOM["y_out"], 0.018),
        size=(TAG_GEOM["gap_port"], TAG_GEOM["w"], 0.0),
        impedance=250.0,
    )

    structs_xz = [metal_box, ferrite_box, reader_env, tag_env]
    mat_names_xz = {
        "region_0": "PEC chassis plate",
        "region_1": "NiZn ferrite (μ'·t = 30 mm)",
        "region_2": "Reader coil (3-turn PEC)",
        "region_3": "ID-1 Tag coil (3-turn PEC)",
    }
    res_xz = schematic.save_setup_figures(
        sol_xz,
        fig_dir,
        plane="xz",
        cut_coord=0.0,
        structures=structs_xz,
        material_names=mat_names_xz,
        sources=[p1, p2],
        unit="mm",
        wavelength=LAMBDA0_NFC,
        freq=F0_NFC,
        min_feature=0.003,
        title_geometry="13.56 MHz NFC coupled reader & ID-1 tag over NiZn ferrite/PEC (XZ cut)",
    )

    grid_3mm = make_nfc_grid(dx_core=0.003)
    dt_3mm = np.sqrt(2.25) * vacuum_courant_dt(grid_3mm)
    sol_xy = wakis.SolverFIT3D(
        grid_3mm,
        bg=[2.25, 1.0],
        dt=dt_3mm,
        bc_low=["pec"] * 3,
        bc_high=["pec"] * 3,
        use_stl=False,
        verbose=0,
    )
    assign_materials(sol_xy, boxes1, frequency=F0_NFC)
    c_tune = float(tuning_capacitance(617.0e-9, freq=F0_NFC))
    port_xy = LumpedPort(
        src,
        component="x",
        center=(0.0, -READER_GEOM["y_out"] - 0.5 * READER_GEOM["w"], 0.0),
        size=(READER_GEOM["gap_port"], 0.0, 0.0),
        impedance=250.0,
    )
    elem_xy = LumpedElement(
        component="x",
        center=(0.0, -READER_GEOM["y_out"] + 0.5 * READER_GEOM["w"], 0.0),
        size=(READER_GEOM["gap_port"], 0.0, 0.0),
        R=300.0,
        C=c_tune,
    )
    turn_boxes_xy = [
        Box((0.0, 0.0, 0.0), (0.066, 0.066, 0.003), "pec"),
        Box((0.0, 0.0, 0.0), (0.054, 0.054, 0.003), "pec"),
        Box((0.0, 0.0, 0.0), (0.042, 0.042, 0.003), "pec"),
    ]
    mat_names_xy = {
        "region_0": "Reader turn 1 (outer 66×66 mm)",
        "region_1": "Reader turn 2 (middle 54×54 mm)",
        "region_2": "Reader turn 3 (inner 42×42 mm)",
    }
    res_xy = schematic.save_setup_figures(
        sol_xy,
        fig_dir,
        suffix="_top",
        plane="xy",
        cut_coord=0.0,
        structures=turn_boxes_xy,
        material_names=mat_names_xy,
        sources=[port_xy, elem_xy],
        unit="mm",
        wavelength=LAMBDA0_NFC,
        freq=F0_NFC,
        min_feature=0.003,
        title_geometry="13.56 MHz 3-turn planar spiral NFC reader coil & LC tank (XY top view)",
    )
    return [
        res_xz["geometry_figure"],
        res_xz["mesh_figure"],
        res_xy["geometry_figure"],
        res_xy["mesh_figure"],
    ]


def plot_01_geometry_3d_views(fig_dir=FIG_DIR):
    """Render 4-panel multi-angle 3D & 2D geometry views of coils, ferrite, metal, ports & capacitors."""
    boxes1, segs1 = build_spiral_coil(
        READER_GEOM["x_out"],
        READER_GEOM["y_out"],
        READER_GEOM["n_turns"],
        READER_GEOM["w"],
        READER_GEOM["s_gap"],
        z0=0.0,
        gap_port=READER_GEOM["gap_port"],
        via_dz=0.003,
    )
    boxes2, segs2 = build_spiral_coil(
        TAG_GEOM["x_out"],
        TAG_GEOM["y_out"],
        TAG_GEOM["n_turns"],
        TAG_GEOM["w"],
        TAG_GEOM["s_gap"],
        z0=0.018,
        gap_port=TAG_GEOM["gap_port"],
        via_dz=0.003,
    )

    fig = plt.figure(figsize=(13.5, 9.5))
    # Panel (a): Isometric 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    _draw_box_3d(ax1, (0.0, 0.0, -0.0075), (0.078, 0.078, 0.003), "#7f8c8d", alpha=0.45)
    _draw_box_3d(ax1, (0.0, 0.0, -0.0030), (0.078, 0.078, 0.006), "#2c3e50", alpha=0.30)
    for b in boxes1:
        _draw_box_3d(ax1, b.center, b.size, "#d35400", alpha=0.92)
    for b in boxes2:
        _draw_box_3d(ax1, b.center, b.size, "#2980b9", alpha=0.88)
    _draw_box_3d(
        ax1,
        (0.0, -READER_GEOM["y_out"], 0.0),
        (READER_GEOM["gap_port"], READER_GEOM["w"], 0.0015),
        "#27ae60",
        alpha=0.95,
    )
    _draw_box_3d(
        ax1,
        (0.0, -TAG_GEOM["y_out"], 0.018),
        (TAG_GEOM["gap_port"], TAG_GEOM["w"], 0.0015),
        "#8e44ad",
        alpha=0.95,
    )
    ax1.view_init(elev=26, azim=-52)
    ax1.set_xlim(-45, 45)
    ax1.set_ylim(-42, 42)
    ax1.set_zlim(-12, 26)
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_zlabel("z [mm]")
    ax1.set_title("(a) 3D isometric: Reader + Ferrite/PEC + ID-1 Tag (d = 18 mm)")

    # Panel (b): Oblique low-angle 3D view emphasizing vertical stackup and underpass bridges
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    _draw_box_3d(ax2, (0.0, 0.0, -0.0075), (0.078, 0.078, 0.003), "#7f8c8d", alpha=0.45)
    _draw_box_3d(ax2, (0.0, 0.0, -0.0030), (0.078, 0.078, 0.006), "#2c3e50", alpha=0.30)
    for b in boxes1:
        _draw_box_3d(ax2, b.center, b.size, "#d35400", alpha=0.92)
    for b in boxes2:
        _draw_box_3d(ax2, b.center, b.size, "#2980b9", alpha=0.88)
    ax2.view_init(elev=12, azim=22)
    ax2.set_xlim(-45, 45)
    ax2.set_ylim(-42, 42)
    ax2.set_zlim(-12, 26)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_zlabel("z [mm]")
    ax2.set_title("(b) Oblique elevation: Underpass vias & ferrite-on-metal backing")

    # Panel (c): Top-down xy plan view comparing Reader (66x66 mm) & ID-1 Card Tag (78x48 mm in 85.6x54 mm)
    ax3 = fig.add_subplot(2, 2, 3)
    card_L, card_W = TAG_GEOM["card_length"] * 1e3, TAG_GEOM["card_width"] * 1e3
    ax3.add_patch(
        plt.Rectangle(
            (-0.5 * card_L, -0.5 * card_W),
            card_L,
            card_W,
            fill=False,
            edgecolor="#2980b9",
            linestyle="--",
            lw=1.8,
            label="ID-1 card outline (85.6 × 54.0 mm)",
        )
    )
    for b in boxes1[:-3]:
        cx, cy, _ = [v * 1e3 for v in b.center]
        sx, sy, _ = [v * 1e3 for v in b.size]
        ax3.add_patch(
            plt.Rectangle(
                (cx - 0.5 * sx, cy - 0.5 * sy),
                sx,
                sy,
                facecolor="#e67e22",
                edgecolor="none",
                alpha=0.65,
            )
        )
    for b in boxes2[:-3]:
        cx, cy, _ = [v * 1e3 for v in b.center]
        sx, sy, _ = [v * 1e3 for v in b.size]
        ax3.add_patch(
            plt.Rectangle(
                (cx - 0.5 * sx, cy - 0.5 * sy),
                sx,
                sy,
                facecolor="#3498db",
                edgecolor="none",
                alpha=0.50,
            )
        )
    ax3.plot([], [], color="#e67e22", lw=5, label="Reader (PCD): 66×66 mm, N=3")
    ax3.plot([], [], color="#3498db", lw=5, label="Tag (PICC): 78×48 mm, N=3")
    ax3.scatter([0.0], [-READER_GEOM["y_out"] * 1e3], color="#27ae60", s=60, zorder=5, label="Port 1 + C1")
    ax3.scatter([0.0], [-TAG_GEOM["y_out"] * 1e3], color="#8e44ad", s=60, zorder=5, label="Port 2 + C2")
    ax3.set_aspect("equal")
    ax3.set_xlim(-48, 48)
    ax3.set_ylim(-38, 38)
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("y [mm]")
    ax3.set_title("(c) Top-down xy layout: Reader vs ID-1 Card Tag traces")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.25)

    # Panel (d): Port gap, underpass bridge & parallel tuning capacitor detail
    ax4 = fig.add_subplot(2, 2, 4)
    for b in boxes1:
        cx, cy, cz = [v * 1e3 for v in b.center]
        sx, sy, _ = [v * 1e3 for v in b.size]
        color = "#c0392b" if cz > 0.5 else "#e67e22"
        ax4.add_patch(
            plt.Rectangle(
                (cx - 0.5 * sx, cy - 0.5 * sy),
                sx,
                sy,
                facecolor=color,
                edgecolor="k",
                lw=0.6,
                alpha=0.80,
            )
        )
    gp = READER_GEOM["gap_port"] * 1e3
    ym = -READER_GEOM["y_out"] * 1e3
    w_mm = READER_GEOM["w"] * 1e3
    ax4.add_patch(
        plt.Rectangle(
            (-0.5 * gp, ym - 0.5 * w_mm),
            gp,
            w_mm,
            facecolor="#2ecc71",
            edgecolor="#1e8449",
            lw=1.5,
            label="LumpedPort + LumpedElement(C1, Rp)",
        )
    )
    ax4.annotate(
        "Port gap (6 mm = 2Δx)\nV = -∫ E_x dx, C1 = 1/(ω₀² L1)",
        xy=(0.0, ym),
        xytext=(4.5, ym - 5.5),
        arrowprops={"arrowstyle": "->", "lw": 1.2},
        fontsize=8.5,
    )
    ax4.annotate(
        "Underpass bridge\n(z = +3 mm) & vias",
        xy=(-4.5, ym + 6.0),
        xytext=(-16.0, ym + 9.0),
        arrowprops={"arrowstyle": "->", "lw": 1.2},
        fontsize=8.5,
    )
    ax4.set_aspect("equal")
    ax4.set_xlim(-18, 18)
    ax4.set_ylim(-38, -14)
    ax4.set_xlabel("x [mm]")
    ax4.set_ylabel("y [mm]")
    ax4.set_title("(d) Zoom on Reader port gap, tuning capacitor & underpass")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(fig_dir) / "01_geometry_3d_views.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_02_mesh_cross_sections(fig_dir=FIG_DIR):
    """Plot 3D FIT grid lines and node alignment across trace width, gap, and vertical stackup."""
    grid = make_nfc_grid(dx_core=0.003)
    boxes1, _ = build_spiral_coil(
        READER_GEOM["x_out"],
        READER_GEOM["y_out"],
        READER_GEOM["n_turns"],
        READER_GEOM["w"],
        READER_GEOM["s_gap"],
        z0=0.0,
        gap_port=READER_GEOM["gap_port"],
        via_dz=0.003,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    # (a) Full graded xy grid
    ax = axes[0]
    for xv in grid.x * 1e3:
        ax.axvline(xv, color="#bdc3c7", lw=0.45)
    for yv in grid.y * 1e3:
        ax.axhline(yv, color="#bdc3c7", lw=0.45)
    for b in boxes1[:-3]:
        cx, cy, _ = [v * 1e3 for v in b.center]
        sx, sy, _ = [v * 1e3 for v in b.size]
        ax.add_patch(
            plt.Rectangle(
                (cx - 0.5 * sx, cy - 0.5 * sy),
                sx,
                sy,
                facecolor="#e67e22",
                edgecolor="k",
                lw=0.5,
                alpha=0.85,
            )
        )
    ax.set_aspect("equal")
    ax.set_xlim(grid.x[0] * 1e3, grid.x[-1] * 1e3)
    ax.set_ylim(grid.y[0] * 1e3, grid.y[-1] * 1e3)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"(a) Full graded xy mesh ({grid.Nx}×{grid.Ny}×{grid.Nz} cells)")

    # (b) Zoom on trace cross-section & inter-turn gap (node alignment at Δ = 3 mm and 1.5 mm)
    ax = axes[1]
    for xv in np.arange(15.0, 36.1, 1.5):
        is_coarse = np.isclose(xv % 3.0, 0.0)
        ax.axvline(
            xv,
            color="#34495e" if is_coarse else "#95a5a6",
            lw=0.9 if is_coarse else 0.5,
            ls="-" if is_coarse else ":",
        )
    for m in range(3):
        x0 = (READER_GEOM["x_out"] - m * 0.006 - 0.5 * READER_GEOM["w"]) * 1e3
        ax.axvspan(
            x0,
            x0 + 3.0,
            ymin=0.25,
            ymax=0.75,
            color="#e67e22",
            alpha=0.85,
            label="Copper trace (w = 3 mm)" if m == 0 else None,
        )
        if m < 2:
            ax.axvspan(
                x0 - 3.0,
                x0,
                ymin=0.25,
                ymax=0.75,
                color="#ecf0f1",
                alpha=0.9,
                label="Inter-turn gap (s = 3 mm)" if m == 0 else None,
            )
    ax.plot([], [], color="#34495e", lw=1.0, label="Δ = 3.0 mm grid nodes")
    ax.plot([], [], color="#95a5a6", lw=0.8, ls=":", label="Δ = 1.5 mm subgrid nodes")
    ax.set_xlim(16.5, 34.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("x [mm] across right-hand spiral turns")
    ax.set_title("(b) Exact node alignment across traces & gaps")
    ax.legend(loc="upper left", fontsize=8)

    # (c) Vertical xz stackup showing PEC plate, ferrite sheet, reader, and tag
    ax = axes[2]
    for zv in grid.z * 1e3:
        if -12.5 <= zv <= 25.0:
            ax.axhline(zv, color="#bdc3c7", lw=0.6)
    ax.axhspan(-9.0, -6.0, xmin=0.1, xmax=0.9, color="#7f8c8d", alpha=0.8, label="PEC metal plate (-9 to -6 mm)")
    ax.axhspan(-6.0, 0.0, xmin=0.1, xmax=0.9, color="#2c3e50", alpha=0.35, label="NiZn ferrite sheet (-6 to 0 mm)")
    ax.axhline(0.0, color="#d35400", lw=3.0, xmin=0.15, xmax=0.85, label="Reader coil plane (z = 0 mm)")
    ax.axhline(18.0, color="#2980b9", lw=3.0, xmin=0.12, xmax=0.88, label="Tag coil plane (z = 18 mm)")
    ax.set_xlim(-45, 45)
    ax.set_ylim(-12, 24)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("(c) Vertical xz mesh stackup near coils & ferrite")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = Path(fig_dir) / "02_mesh_cross_sections.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_03_inductance_coupling_vs_separation(data, fig_dir=FIG_DIR):
    """Plot L1, L2, M(d), k(d), and k(dx) vs Wheeler, Mohan, Grover, and Neumann references."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    # (a) Self-inductances L1 and L2: 3D FIT vs Wheeler, Mohan CS, Grover, and Neumann
    ax = axes[0, 0]
    labels = ["Modified\nWheeler", "Mohan\nCurrent-Sheet", "Grover\nGMD", "Neumann\n3D Segs", "3D FIT\nSimulation"]
    l1_vals = list(data["L1_analytic_H"] * 1e9) + [float(data["L1_sim_H"][1] * 1e9)]
    l2_vals = list(data["L2_analytic_H"] * 1e9) + [float(data["L2_sim_H"][1] * 1e9)]
    x_pos = np.arange(len(labels))
    width = 0.36
    ax.bar(x_pos - 0.5 * width, l1_vals, width, color="#e67e22", label="Reader L1 (66×66 mm, N=3)")
    ax.bar(x_pos + 0.5 * width, l2_vals, width, color="#2980b9", label="Tag L2 (78×48 mm, N=3)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Self-inductance L [nH]")
    ax.set_title("(a) Self-inductance L1, L2: 3D FIT vs analytical formulas")
    ax.legend(fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.3)

    # (b) Mutual inductance M(d) vs separation (+ 1-coil mutation check)
    ax = axes[0, 1]
    ax.plot(
        data["d_dense_m"] * 1e3,
        data["M_neu_dense_H"] * 1e9,
        color="#2c3e50",
        lw=2.0,
        label="Neumann double integral (analytic)",
    )
    ax.plot(
        data["d_sim_m"] * 1e3,
        data["M_sim_H"] * 1e9,
        "o",
        color="#e74c3c",
        ms=7.5,
        label="3D FIT simulation (LumpedPort Z21)",
    )
    ax.scatter(
        [18.0],
        [float(data["M_mutation_no_tag_H"][0] * 1e9)],
        marker="X",
        s=90,
        color="#8e44ad",
        zorder=5,
        label=f"Mutation (tag removed): M = {float(data['M_mutation_no_tag_H'][0]*1e9):.2f} nH",
    )
    ax.set_xlabel("Axial separation d [mm]")
    ax.set_ylabel("Mutual inductance M [nH]")
    ax.set_title("(b) Mutual inductance M12(d) vs Neumann integral")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # (c) Coupling coefficient k(d) = M / sqrt(L1 L2) vs separation
    ax = axes[1, 0]
    ax.plot(
        data["d_dense_m"] * 1e3,
        data["k_neu_dense"],
        color="#2c3e50",
        lw=2.0,
        label="Neumann reference k(d)",
    )
    ax.plot(
        data["d_sim_m"] * 1e3,
        data["k_sim"],
        "s-",
        color="#27ae60",
        ms=7.0,
        lw=1.2,
        label="3D FIT simulation k(d)",
    )
    ax.set_xlabel("Axial separation d [mm]")
    ax.set_ylabel("Coupling coefficient k = M / √(L1 L2)")
    ax.set_title("(c) Monotonic decay of coupling k(d) in the near field")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # (d) Coupling coefficient k(dx) vs lateral offset at d = 18 mm
    ax = axes[1, 1]
    ax.plot(
        data["dx_dense_m"] * 1e3,
        data["k_off_neu_dense"],
        color="#2c3e50",
        lw=2.0,
        label="Neumann reference k(Δx) at d = 18 mm",
    )
    ax.plot(
        data["dx_sim_m"] * 1e3,
        data["k_off_sim"],
        "D-",
        color="#d35400",
        ms=7.0,
        lw=1.2,
        label="3D FIT simulation k(Δx)",
    )
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Lateral card offset Δx [mm]")
    ax.set_ylabel("Coupling coefficient k(Δx)")
    ax.set_title("(d) Lateral misalignment sensitivity k(Δx) at d = 18 mm")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(fig_dir) / "03_inductance_coupling_vs_separation.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_04_impedance_and_resonance_tuning(data, fig_dir=FIG_DIR):
    """Plot untuned Z11(f), parallel-tuned resonance at 13.56 MHz, detuning/retuning on metal/ferrite, and 2-port S-parameters."""
    f_mhz = data["f_band_hz"] / 1e6
    z_un = data["Z11_untuned_real"] + 1j * data["Z11_untuned_imag"]
    z_tu = data["Z11_tuned_real"] + 1j * data["Z11_tuned_imag"]
    c1_pF = float(data["C1_tune_F"][0] * 1e12)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    # (a) Untuned vs Parallel-tuned Z11(f) from 3D FIT
    ax = axes[0]
    ax.plot(f_mhz, np.imag(z_un), "--", color="#2980b9", lw=1.8, label="Untuned Im(Z11) = ω L1")
    ax.plot(f_mhz, np.abs(z_tu), color="#e74c3c", lw=2.2, label=f"Tuned |Z_tank(f)| (C1 = {c1_pF:.1f} pF)")
    ax.plot(f_mhz, np.real(z_tu), ":", color="#27ae60", lw=1.6, label="Tuned Re(Z_tank)")
    ax.axvline(13.56, color="k", lw=1.0, ls="-.", label="f0 = 13.56 MHz")
    ax.set_xlabel("Frequency f [MHz]")
    ax.set_ylabel("Impedance [Ω]")
    ax.set_title("(a) 3D FIT untuned vs parallel-tuned Z11(f)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Resonance shift across configurations (free space, metal detuned, ferrite shifted, ferrite retuned)
    ax = axes[1]
    f_fine = np.linspace(8.0e6, 21.0e6, 260)
    w_fine = 2.0 * np.pi * f_fine
    l_free, l_met, l_fer = data["cfg3_L1_H"]
    c_nom, _, _, c_ret = data["cfg3_C1_F"]
    r_p = 450.0
    for lbl, l_val, c_val, col, ls in [
        (f"1. Free space (C1={c_nom*1e12:.0f} pF → 13.56 MHz)", l_free, c_nom, "#2980b9", "-"),
        (f"2. On PEC metal (detuned → {data['cfg3_f_res_hz'][1]/1e6:.2f} MHz)", l_met, c_nom, "#7f8c8d", "--"),
        (f"3a. + NiZn ferrite (unretuned → {data['cfg3_f_res_hz'][2]/1e6:.2f} MHz)", l_fer, c_nom, "#e67e22", "-."),
        (f"3b. + NiZn ferrite (retuned C1={c_ret*1e12:.0f} pF → 13.56 MHz)", l_fer, c_ret, "#27ae60", "-"),
    ]:
        y_adm = (1.0 / r_p) + 1j * (w_fine * c_val - 1.0 / (w_fine * l_val))
        ax.plot(f_fine / 1e6, np.abs(1.0 / y_adm), color=col, ls=ls, lw=2.0, label=lbl)
    ax.axvline(13.56, color="k", lw=0.9, ls=":")
    ax.set_xlabel("Frequency f [MHz]")
    ax.set_ylabel("|Z_tank(f)| [Ω]")
    ax.set_title("(b) Resonance detuning on metal & retuning on ferrite")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # (c) Two-port S11(f) and S21(f) from wakis.sparameters.s_matrix
    ax = axes[2]
    ax.plot(f_mhz, 20.0 * np.log10(np.maximum(data["S11_2port_mag"], 1e-4)), color="#2c3e50", lw=2.0, label="|S11(f)| [dB]")
    ax.plot(f_mhz, 20.0 * np.log10(np.maximum(data["S21_2port_mag"], 1e-4)), color="#e74c3c", lw=2.2, label="|S21(f)| [dB] (s_matrix)")
    ax.axvline(13.56, color="k", lw=1.0, ls=":", label="13.56 MHz carrier")
    ax.set_xlabel("Frequency f [MHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("(c) Two-port S-parameters at d = 18 mm (s_matrix)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(fig_dir) / "04_impedance_and_resonance_tuning.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_05_s21_and_link_efficiency(data, fig_dir=FIG_DIR):
    """Plot Schormans link efficiency eta_max vs separation d, lateral offset dx, and (k, Q) contour map."""
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    q1 = float(data["cu_reader_metrics"][5])
    q2 = float(data["cu_tag_metrics"][5])

    # (a) Link efficiency vs axial separation d
    ax = axes[0]
    ax.plot(
        data["d_dense_m"] * 1e3,
        data["eta_neu_dense"] * 100.0,
        color="#2c3e50",
        lw=2.0,
        label=f"Schormans + Neumann (Q1={q1:.0f}, Q2={q2:.0f})",
    )
    ax.plot(
        data["d_sim_m"] * 1e3,
        data["eta_sim"] * 100.0,
        "o",
        color="#e74c3c",
        ms=7.5,
        label="3D FIT + Schormans formula",
    )
    # Also show loaded Q_L = 30 (typical ISO/IEC 14443 high-bitrate damping)
    eta_loaded = schormans_link_efficiency(data["k_neu_dense"], 30.0, 30.0)
    ax.plot(
        data["d_dense_m"] * 1e3,
        eta_loaded * 100.0,
        "--",
        color="#2980b9",
        lw=1.7,
        label="Damped reader/tag (Q_L = 30 for 848 kbps)",
    )
    ax.set_xlabel("Axial separation d [mm]")
    ax.set_ylabel("Maximum link efficiency η_max [%]")
    ax.set_title("(a) Link efficiency η_max vs separation d")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Link efficiency vs lateral offset dx at d = 18 mm
    ax = axes[1]
    ax.plot(
        data["dx_dense_m"] * 1e3,
        data["eta_off_neu_dense"] * 100.0,
        color="#2c3e50",
        lw=2.0,
        label="Schormans + Neumann (d = 18 mm)",
    )
    ax.plot(
        data["dx_sim_m"] * 1e3,
        data["eta_off_sim"] * 100.0,
        "s",
        color="#d35400",
        ms=7.5,
        label="3D FIT simulation",
    )
    ax.set_xlabel("Lateral offset Δx [mm]")
    ax.set_ylabel("Maximum link efficiency η_max [%]")
    ax.set_title("(b) Link efficiency η_max vs lateral offset Δx")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Universal Schormans efficiency contour in the (k, sqrt(Q1 Q2)) plane
    ax = axes[2]
    k_grid = np.logspace(-2.2, -0.3, 120)
    q_grid = np.logspace(1.0, 2.5, 120)
    kk, qq = np.meshgrid(k_grid, q_grid)
    eta_map = schormans_link_efficiency(kk, qq, qq) * 100.0
    cs = ax.contourf(kk, qq, eta_map, levels=np.linspace(0, 100, 11), cmap="viridis")
    plt.colorbar(cs, ax=ax, label="η_max [%]")
    q_geom = np.sqrt(q1 * q2)
    ax.scatter(
        data["k_sim"],
        [q_geom] * len(data["k_sim"]),
        color="#e74c3c",
        edgecolor="w",
        s=55,
        zorder=5,
        label="Simulated d = 12..36 mm",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Coupling coefficient k")
    ax.set_ylabel("Geometric mean quality factor √(Q1 Q2)")
    ax.set_title("(c) Schormans efficiency map η_max(k, Q)")
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    out_path = Path(fig_dir) / "05_s21_and_link_efficiency.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_06_ferrite_permeability_fit(data, fig_dir=FIG_DIR):
    """Plot fitted complex permeability mu'(f), mu''(f) of dispersive_lib['nizn ferrite'] vs source data and MnZn."""
    f_mhz = data["f_ferrite_hz"] / 1e6
    f_src = data["f_src_hz"] / 1e6

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    # (a) Real and imaginary permeability fit vs source points
    ax = axes[0]
    ax.plot(f_mhz, data["mu_ferrite_real"], color="#2980b9", lw=2.2, label="Fitted μ'(f) (Lorentz ADE)")
    ax.plot(f_mhz, data["mu_disc_real"], "--", color="#1abc9c", lw=1.4, label="Discrete ADE μ'_d(f, Δt)")
    ax.scatter(f_src, data["mu_src_real"], color="#2c3e50", s=40, zorder=5, label="Model μ' points (form of [6], level of [5])")
    ax.plot(f_mhz, data["mu_ferrite_imag"], color="#e74c3c", lw=2.0, label="Fitted μ''(f) = -Im μ(f) ≥ 0")
    ax.scatter(f_src, data["mu_src_imag"], marker="s", color="#c0392b", s=40, zorder=5, label="Model μ'' points (form of [6], level of [5])")
    ax.axvline(13.56, color="k", lw=0.9, ls=":", label="13.56 MHz (μ'=150.0, μ''=3.6)")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency f [MHz]")
    ax.set_ylabel("Relative permeability μ', μ''")
    ax.set_title("(a) Representative NiZn ferrite sheet: model μ(f) and pole fit")
    ax.legend(fontsize=7.5)
    ax.grid(True, which="both", alpha=0.3)

    # (b) Magnetic loss tangent tan(delta_m) = mu'' / mu' and passivity check
    ax = axes[1]
    tan_d = data["mu_ferrite_imag"] / np.maximum(data["mu_ferrite_real"], 1e-6)
    tan_src = data["mu_src_imag"] / data["mu_src_real"]
    ax.plot(f_mhz, tan_d, color="#8e44ad", lw=2.0, label="Fitted tan δ_m = μ'' / μ'")
    ax.scatter(f_src, tan_src, color="#2c3e50", s=40, zorder=5, label="Source tan δ_m points")
    ax.axvline(13.56, color="k", lw=0.9, ls=":")
    ax.annotate(
        f"13.56 MHz:\ntan δ_m = {tan_src[4]:.3f}\n(Q_m = {1.0/tan_src[4]:.0f})",
        xy=(13.56, tan_src[4]),
        xytext=(2.2, 0.08),
        arrowprops={"arrowstyle": "->", "lw": 1.1},
        fontsize=8.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency f [MHz]")
    ax.set_ylabel("Magnetic loss tangent tan δ_m = μ'' / μ'")
    ax.set_title("(b) Passivity (μ'' > 0) & low loss at 13.56 MHz")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # (c) NiZn vs MnZn ferrite permeability & Snoek's limit across frequency
    ax = axes[2]
    f_wide = np.geomspace(0.1e6, 100e6, 150)
    # Representative MnZn power ferrite (mu_i = 1500, Snoek relaxation at 1.8 MHz, Tsutaoka 2003 [6])
    w_w = 2.0 * np.pi * f_wide
    mu_mnzn = 1.0 + 1499.0 / (1.0 + 1j * w_w / (2.0 * np.pi * 1.8e6))
    mu_nizn = dispersive_lib["nizn ferrite"].mu(f_wide)
    ax.plot(f_wide / 1e6, np.real(mu_nizn), color="#2980b9", lw=2.0, label="NiZn μ'(f) (NFC sheet)")
    ax.plot(f_wide / 1e6, -np.imag(mu_nizn), "--", color="#2980b9", lw=1.8, label="NiZn μ''(f) (low loss at 13.56 MHz)")
    ax.plot(f_wide / 1e6, np.real(mu_mnzn), color="#7f8c8d", lw=2.0, label="MnZn μ'(f) (kHz power ferrite)")
    ax.plot(f_wide / 1e6, -np.imag(mu_mnzn), "--", color="#c0392b", lw=1.8, label="MnZn μ''(f) (lossy > 2 MHz)")
    ax.axvline(13.56, color="k", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency f [MHz]")
    ax.set_ylabel("Permeability magnitude")
    ax.set_title("(c) NiZn vs MnZn ferrite (Snoek's limit at 13.56 MHz)")
    ax.legend(fontsize=7.5)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = Path(fig_dir) / "06_ferrite_permeability_fit.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_07_ferrite_shielding_h_maps(data, fig_dir=FIG_DIR):
    """Plot side-by-side |H(x, z)| maps and magnetic flux streamlines for the 3 configurations."""
    x_mm = data["x_fer_m"] * 1e3
    z_mm = data["z_fer_m"] * 1e3
    xx, zz = np.meshgrid(x_mm, z_mm, indexing="ij")

    configs = [
        (
            "(a) Config 1: Free space\n"
            f"L1={data['cfg3_L1_H'][0]*1e9:.1f} nH, k={data['cfg3_k'][0]:.3f}",
            data["hx_xz_free"],
            data["hz_xz_free"],
            False,
            False,
        ),
        (
            "(b) Config 2: On PEC metal plate\n"
            f"L1={data['cfg3_L1_H'][1]*1e9:.1f} nH (-41%), k={data['cfg3_k'][1]:.3f}",
            data["hx_xz_metal"],
            data["hz_xz_metal"],
            True,
            False,
        ),
        (
            "(c) Config 3: NiZn ferrite on PEC metal\n"
            f"L1={data['cfg3_L1_H'][2]*1e9:.1f} nH, k={data['cfg3_k'][2]:.3f} (restored)",
            data["hx_xz_ferrite"],
            data["hz_xz_ferrite"],
            True,
            True,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), sharey=True)
    # Regular grid interpolation for clean streamlines
    xg = np.linspace(-42.0, 42.0, 65)
    zg = np.linspace(-11.0, 26.0, 55)

    for ax, (title, hx_c, hz_c, has_metal, has_ferrite) in zip(axes, configs):
        h_mag = np.hypot(np.abs(hx_c), np.abs(hz_c))
        h_log = np.log10(np.maximum(h_mag, 1e-2))
        pcm = ax.pcolormesh(
            xx, zz, h_log, cmap="inferno", vmin=0.0, vmax=2.3, shading="auto"
        )
        # Quiver of real-part H vectors (flux bending into ferrite)
        ix_sub = np.where(np.abs(x_mm) <= 42.0)[0][::2]
        iz_sub = np.where((z_mm >= -11.0) & (z_mm <= 26.0))[0][::2]
        hx_r = np.real(hx_c[np.ix_(ix_sub, iz_sub)])
        hz_r = np.real(hz_c[np.ix_(ix_sub, iz_sub)])
        norm_q = np.hypot(hx_r, hz_r) + 1e-12
        ax.quiver(
            xx[np.ix_(ix_sub, iz_sub)],
            zz[np.ix_(ix_sub, iz_sub)],
            hx_r / norm_q,
            hz_r / norm_q,
            color="w",
            alpha=0.65,
            scale=22,
            width=0.004,
        )
        # Overlay reader/tag coil cross-sections and metal/ferrite layers
        ax.axhline(0.0, color="#2ecc71", lw=2.2, xmin=0.12, xmax=0.88, label="Reader (z=0)")
        ax.axhline(18.0, color="#3498db", lw=2.2, xmin=0.08, xmax=0.92, label="Tag (z=18 mm)")
        if has_metal:
            ax.axhspan(-9.0, -6.0, color="#bdc3c7", alpha=0.85, label="PEC metal chassis")
        if has_ferrite:
            ax.axhspan(-6.0, 0.0, color="#1abc9c", alpha=0.28, label="NiZn ferrite sheet")
        ax.set_xlim(-44, 44)
        ax.set_ylim(-11.5, 25.5)
        ax.set_xlabel("x [mm]")
        ax.set_title(title, fontsize=9.5)
        ax.legend(loc="upper right", fontsize=7.5)
    axes[0].set_ylabel("z [mm]")
    fig.colorbar(pcm, ax=axes.ravel().tolist(), pad=0.02, label="log10(|H| / I1 [m⁻¹])")
    out_path = Path(fig_dir) / "07_ferrite_shielding_h_maps.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_08_ferrite_sweeps_and_mut_equivalence(data, fig_dir=FIG_DIR):
    """Plot L1, k, f_res vs ferrite mu'*t, vs coil-metal gap, and the equal-(mu'-1)*t thin-sheet equivalence test."""
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    # (a) L1 and k vs ferrite sheet permeance product (mu' - 1) * t
    ax = axes[0]
    mut_mm = data["mut_vals_mm"]
    ax.plot(mut_mm, data["L1_mut_H"] * 1e9, "o-", color="#e67e22", lw=2.0, label="Reader L1 [nH]")
    ax.axhline(data["cfg3_L1_H"][0] * 1e9, color="#e67e22", ls="--", lw=1.2, label="Free-space L1")
    ax.set_xlabel("Ferrite sheet permeance (μ' − 1)·t [mm]")
    ax.set_ylabel("Self-inductance L1 [nH]", color="#d35400")
    ax_r = ax.twinx()
    ax_r.plot(mut_mm, data["k_mut"], "s-", color="#2980b9", lw=2.0, label="Coupling k")
    ax_r.axhline(data["cfg3_k"][0], color="#2980b9", ls=":", lw=1.2, label="Free-space k")
    ax_r.set_ylabel("Coupling coefficient k", color="#2980b9")
    ax.set_title("(a) Recovery of L1 and k vs ferrite (μ'−1)·t")
    ax.grid(True, alpha=0.3)

    # (b) L1 and k vs coil-to-metal gap without ferrite
    ax = axes[1]
    gaps_mm = data["gap_vals_m"] * 1e3
    ax.plot(gaps_mm, data["L1_gap_H"] * 1e9, "o-", color="#c0392b", lw=2.0, label="L1 on bare PEC metal")
    ax.axhline(data["cfg3_L1_H"][0] * 1e9, color="#c0392b", ls="--", lw=1.2, label="Free-space L1")
    ax.set_xlabel("Coil–metal separation gap g_metal [mm]")
    ax.set_ylabel("Self-inductance L1 [nH]", color="#c0392b")
    ax_r2 = ax.twinx()
    ax_r2.plot(gaps_mm, data["k_gap"], "D-", color="#27ae60", lw=2.0, label="k on bare PEC metal")
    ax_r2.axhline(data["cfg3_k"][0], color="#27ae60", ls=":", lw=1.2, label="Free-space k")
    ax_r2.set_ylabel("Coupling coefficient k", color="#27ae60")
    ax.set_title("(b) Eddy-current suppression vs coil–metal gap")
    ax.grid(True, alpha=0.3)

    # (c) Equal-(mu'-1)*t = 27 mm thin-sheet equivalence test (+ thick-sheet failure point)
    ax = axes[2]
    t_eq = data["t_eq_mm"]
    mu_eq = data["mu_eq_eff"]
    k_eq = data["k_eq"]
    l1_eq = data["L1_eq_H"] * 1e9
    labels_eq = [f"t={t:.1f} mm\nμ'={m:.0f}" for t, m in zip(t_eq, mu_eq)]
    x_idx = np.arange(len(t_eq))
    colors = ["#27ae60", "#27ae60", "#27ae60", "#e74c3c"]
    ax.bar(x_idx - 0.18, l1_eq / l1_eq[1], 0.34, color=colors, alpha=0.85, label="Normalized L1 / L1(3 mm)")
    ax.bar(x_idx + 0.18, k_eq / k_eq[1], 0.34, color="#2980b9", alpha=0.80, label="Normalized k / k(3 mm)")
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    ax.axhspan(0.94, 1.06, color="#2ecc71", alpha=0.15, label="±6% equivalence band")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels_eq, fontsize=8)
    ax.set_ylim(0.80, 1.25)
    ax.set_ylabel("Ratio to nominal (t = 3.0 mm, μ' = 10)")
    ax.set_title("(c) Equal (μ'−1)·t = 27 mm test & thick-sheet limit")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = Path(fig_dir) / "08_ferrite_sweeps_and_mut_equivalence.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_09_fields_and_surface_current(data, fig_dir=FIG_DIR):
    """Plot horizontal xy |H| maps in reader & tag planes and surface current |J_s| = |n x H| on traces."""
    x_mm = data["x_fer_m"] * 1e3
    y_mm = data["y_fer_m"] * 1e3
    xx, yy = np.meshgrid(x_mm, y_mm, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))

    # (a) Reader plane |H(x, y, z=0)| + tangential H vectors
    ax = axes[0]
    h_r_mag = np.sqrt(
        np.abs(data["hx_xy_r"]) ** 2
        + np.abs(data["hy_xy_r"]) ** 2
        + np.abs(data["hz_xy_r"]) ** 2
    )
    pcm0 = ax.pcolormesh(xx, yy, h_r_mag, cmap="magma", shading="auto")
    plt.colorbar(pcm0, ax=ax, label="|H(x, y, 0)| / I1 [m⁻¹]")
    ix_s = np.arange(0, len(x_mm), 2)
    iy_s = np.arange(0, len(y_mm), 2)
    ax.quiver(
        xx[np.ix_(ix_s, iy_s)],
        yy[np.ix_(ix_s, iy_s)],
        np.real(data["hx_xy_r"][np.ix_(ix_s, iy_s)]),
        np.real(data["hy_xy_r"][np.ix_(ix_s, iy_s)]),
        color="w",
        alpha=0.65,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-42, 42)
    ax.set_ylim(-36, 36)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("(a) Reader plane |H(x, y, z=0)| & transverse vectors")

    # (b) Tag plane |H_z(x, y, z=18 mm)| coupling through the ID-1 card coil
    ax = axes[1]
    hz_t = np.abs(data["hz_xy_t"])
    pcm1 = ax.pcolormesh(xx, yy, hz_t, cmap="plasma", shading="auto")
    plt.colorbar(pcm1, ax=ax, label="|H_z(x, y, d)| / I1 [m⁻¹]")
    boxes2, _ = build_spiral_coil(
        TAG_GEOM["x_out"],
        TAG_GEOM["y_out"],
        TAG_GEOM["n_turns"],
        TAG_GEOM["w"],
        TAG_GEOM["s_gap"],
        z0=0.018,
        gap_port=TAG_GEOM["gap_port"],
        via_dz=0.003,
    )
    for b in boxes2[:-3]:
        cx, cy, _ = [v * 1e3 for v in b.center]
        sx, sy, _ = [v * 1e3 for v in b.size]
        ax.add_patch(
            plt.Rectangle(
                (cx - 0.5 * sx, cy - 0.5 * sy),
                sx,
                sy,
                fill=False,
                edgecolor="w",
                lw=0.8,
            )
        )
    ax.set_aspect("equal")
    ax.set_xlim(-42, 42)
    ax.set_ylim(-36, 36)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("(b) Tag plane normal flux |H_z(x, y, z=18 mm)|")

    # (c) Surface current magnitude |J_s| = |z_hat x H_tan| masked onto the copper spiral traces
    ax = axes[2]
    js_full = np.hypot(np.abs(data["hx_xy_r"]), np.abs(data["hy_xy_r"]))
    boxes1, _ = build_spiral_coil(
        READER_GEOM["x_out"],
        READER_GEOM["y_out"],
        READER_GEOM["n_turns"],
        READER_GEOM["w"],
        READER_GEOM["s_gap"],
        z0=0.0,
        gap_port=READER_GEOM["gap_port"],
        via_dz=0.003,
    )
    trace_mask = np.zeros_like(js_full, dtype=bool)
    for b in boxes1[:-3]:
        cx, cy, _ = [v * 1e3 for v in b.center]
        sx, sy, _ = [v * 1e3 for v in b.size]
        trace_mask |= (np.abs(xx - cx) <= 0.5 * sx + 1.6) & (
            np.abs(yy - cy) <= 0.5 * sy + 1.6
        )
    js_masked = np.ma.masked_where(~trace_mask, js_full)
    ax.set_facecolor("#1a1a2e")
    pcm2 = ax.pcolormesh(xx, yy, js_masked, cmap="hot", shading="auto")
    plt.colorbar(pcm2, ax=ax, label="Surface current |J_s| / I1 [m⁻¹]")
    ax.set_aspect("equal")
    ax.set_xlim(-38, 38)
    ax.set_ylim(-38, 38)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("(c) Surface current |J_s| = |n̂ × H| on Reader traces")

    fig.tight_layout()
    out_path = Path(fig_dir) / "09_fields_and_surface_current.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_10_permittivity_scaling_and_resolution(data, fig_dir=FIG_DIR):
    """Plot background permittivity scaling study s in {1,4,9,16,64} and 3-level resolution convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))

    # (a) Permittivity scaling invariance (s = 1, 4, 9, 16) vs self-resonant breakdown (s = 64)
    ax = axes[0]
    s_vals = data["s_vals"]
    l1_s_norm = data["L1_s_H"] / data["L1_s_H"][0]
    m_s_norm = data["M_s_H"] / data["M_s_H"][0]
    k_s_norm = data["k_s"] / data["k_s"][0]
    ax.plot(s_vals, l1_s_norm, "o-", color="#e67e22", lw=2.0, label="L1(s) / L1(s=1)")
    ax.plot(s_vals, m_s_norm, "s-", color="#e74c3c", lw=2.0, label="M(s) / M(s=1)")
    ax.plot(s_vals, k_s_norm, "D-", color="#2980b9", lw=2.0, label="k(s) / k(s=1)")
    ax.axhspan(0.97, 1.03, color="#2ecc71", alpha=0.18, label="±3% quasi-static invariance band")
    ax.set_xscale("log")
    ax.set_xlabel("Background permittivity scaling factor s = ε_r")
    ax.set_ylabel("Ratio to unscaled vacuum (s = 1)")
    ax.set_title("(a) Permittivity scaling invariance & high-s breakdown")
    ax.legend(fontsize=7.8)
    ax.grid(True, which="both", alpha=0.3)

    # (b) Timestep count Nt and wall-clock speedup vs s
    ax = axes[1]
    ax.plot(s_vals, data["Nt_s"], "o-", color="#2c3e50", lw=2.0, label="Timesteps Nt ∝ 1/√s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Background permittivity scaling factor s = ε_r")
    ax.set_ylabel("Number of timesteps Nt")
    ax_r = ax.twinx()
    ax_r.plot(s_vals, data["wall_s_arr"], "s--", color="#8e44ad", lw=1.8, label="CPU run time [s]")
    ax_r.set_ylabel("Wall-clock time [s]", color="#8e44ad")
    ax.set_title("(b) Courant step reduction Nt ∝ s⁻¹/²")
    ax.grid(True, which="both", alpha=0.3)

    # (c) 3D grid resolution convergence (dx = 4.5, 3.0, 2.0 mm) vs Neumann/Grover reference
    ax = axes[2]
    dx_mm = data["dx_res_m"] * 1e3
    m_neu_18 = float(data["M_neu_pts_H"][1])
    k_neu_18 = float(data["k_neu_pts"][1])
    err_m = np.abs(data["M_res_H"] - m_neu_18) / m_neu_18 * 100.0
    err_k = np.abs(data["k_res"] - k_neu_18) / k_neu_18 * 100.0
    ax.plot(dx_mm, err_m, "o-", color="#e74c3c", lw=2.0, ms=7.5, label="|M_sim − M_Neumann| / M_Neumann [%]")
    ax.plot(dx_mm, err_k, "s-", color="#27ae60", lw=2.0, ms=7.5, label="|k_sim − k_Neumann| / k_Neumann [%]")
    for dx_v, nc, em in zip(dx_mm, data["ncells_res"], err_m):
        ax.annotate(f"{int(nc/1e3)}k cells", (dx_v, em), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.invert_xaxis()
    ax.set_xlabel("Core cell size Δx = Δy = Δz [mm] (finer →)")
    ax.set_ylabel("Relative error vs Neumann reference [%]")
    ax.set_title("(c) 3D FIT mesh resolution convergence study")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(fig_dir) / "10_permittivity_scaling_and_resolution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_11_magnetic_field_oscillation_gif(data, fig_dir=FIG_DIR):
    """Render an animated GIF (< 5 MB) of the 13.56 MHz magnetic field oscillating between the coils."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    x_mm = data["x_fer_m"] * 1e3
    z_mm = data["z_fer_m"] * 1e3
    xx, zz = np.meshgrid(x_mm, z_mm, indexing="ij")
    hx_c = data["hx_xz_ferrite"]
    hz_c = data["hz_xz_ferrite"]
    vmax = float(np.percentile(np.abs(hz_c), 98.5))

    n_frames = 20
    phases = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=90)

    ix_sub = np.where(np.abs(x_mm) <= 42.0)[0][::2]
    iz_sub = np.where((z_mm >= -11.0) & (z_mm <= 25.0))[0][::2]

    def draw_frame(frame_idx):
        ax.clear()
        phi = phases[frame_idx]
        fac = np.exp(1j * phi)
        hz_t = np.real(hz_c * fac)
        hx_t = np.real(hx_c * fac)
        ax.pcolormesh(
            xx,
            zz,
            hz_t,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
        )
        ax.quiver(
            xx[np.ix_(ix_sub, iz_sub)],
            zz[np.ix_(ix_sub, iz_sub)],
            hx_t[np.ix_(ix_sub, iz_sub)],
            hz_t[np.ix_(ix_sub, iz_sub)],
            color="k",
            alpha=0.55,
            scale=vmax * 18.0,
        )
        ax.axhspan(-9.0, -6.0, color="#7f8c8d", alpha=0.85, label="PEC metal plate")
        ax.axhspan(-6.0, 0.0, color="#1abc9c", alpha=0.30, label="NiZn ferrite sheet")
        ax.axhline(0.0, color="#d35400", lw=2.5, xmin=0.12, xmax=0.88, label="Reader (z=0)")
        ax.axhline(18.0, color="#2980b9", lw=2.5, xmin=0.08, xmax=0.92, label="Tag (z=18 mm)")
        t_ns = (phi / (2.0 * np.pi)) * (1e9 / F0_NFC)
        ax.set_xlim(-44, 44)
        ax.set_ylim(-11.5, 25.5)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_title(
            f"13.56 MHz NFC Magnetic Field H_z(x, z, t)  |  t = {t_ns:4.1f} / 73.7 ns (ωt = {np.degrees(phi):3.0f}°)"
        )
        ax.legend(loc="upper right", fontsize=7.5)

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=90)
    out_path = Path(fig_dir) / "11_magnetic_field_oscillation.gif"
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return out_path


def generate_all_figures(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """Load `nfc_coils_results.npz` from `data_dir` and render setup and result figures into `fig_dir`."""
    npz_path = Path(data_dir) / "nfc_coils_results.npz"
    if not npz_path.exists():
        compute_all_study_data(quick=True, out_dir=data_dir)
    setup_paths = plot_00_setup_figures(fig_dir=fig_dir)
    with np.load(npz_path, allow_pickle=False) as data:
        paths = setup_paths + [
            plot_01_geometry_3d_views(fig_dir=fig_dir),
            plot_02_mesh_cross_sections(fig_dir=fig_dir),
            plot_03_inductance_coupling_vs_separation(data, fig_dir=fig_dir),
            plot_04_impedance_and_resonance_tuning(data, fig_dir=fig_dir),
            plot_05_s21_and_link_efficiency(data, fig_dir=fig_dir),
            plot_06_ferrite_permeability_fit(data, fig_dir=fig_dir),
            plot_07_ferrite_shielding_h_maps(data, fig_dir=fig_dir),
            plot_08_ferrite_sweeps_and_mut_equivalence(data, fig_dir=fig_dir),
            plot_09_fields_and_surface_current(data, fig_dir=fig_dir),
            plot_10_permittivity_scaling_and_resolution(data, fig_dir=fig_dir),
            plot_11_magnetic_field_oscillation_gif(data, fig_dir=fig_dir),
        ]
    return paths


# %% [markdown]
# ## 7. Results: inductance, resonance, link efficiency & ferrite shielding
#
# ### 7.1 Self-inductance, mutual inductance & coupling factor vs analytical theory
#
# * **Self-inductance benchmark (Figure 3a):** For the 3-turn $66 \times 66\;\text{mm}$ reader coil (`READER_GEOM`) and the 3-turn $78 \times 48\;\text{mm}$ ID-1 card tag coil (`TAG_GEOM`), the Modified Wheeler ($L_1 = 625.4\;\text{nH}, L_2 = 532.6\;\text{nH}$), Mohan Current-Sheet ($L_1 = 618.5\;\text{nH}, L_2 = 527.4\;\text{nH}$), Grover GMD ($L_1 = 596.6\;\text{nH}, L_2 = 504.3\;\text{nH}$), and 3D segment Neumann self-inductance ($L_1 = 628.3\;\text{nH}, L_2 = 531.9\;\text{nH}$) agree closely with the 3D FIT port-impedance extraction ($L_{1,\text{sim}} = 617.0\;\text{nH}$ coupled / $629.9\;\text{nH}$ isolated, $L_{2,\text{sim}} = 518.7\;\text{nH}$, within $0.2\text{–}3.5\%$).
# * **Mutual inductance & coupling coefficient (Figures 3b–d):** Across separations $d \in \{12, 18, 24, 36\}\;\text{mm}$, $M_{\text{sim}}(d)$ and $k_{\text{sim}}(d)$ decrease monotonically from $M = 194.6\;\text{nH}$ ($k = 0.345$) at $d = 12\;\text{mm}$ to $M = 55.4\;\text{nH}$ ($k = 0.098$) at $d = 36\;\text{mm}$, tracking the numerical Neumann double integral ($M_{\text{Neumann}} = [207.0, 148.7, 108.8, 61.4]\;\text{nH}$, $k_{\text{Neumann}} = [0.358, 0.257, 0.188, 0.106]$) within $3.6\text{–}8.1\%$. Removing the tag coil in the mutation check collapses the extracted mutual inductance by five orders of magnitude ($|M_{\text{mut}}| = 0.0016\;\text{nH} \ll 1\;\text{nH}$).
# * **Resonance tuning & link efficiency (Figures 4 & 5):** Attaching the designed parallel tuning capacitor $C_1 = 1 / (\omega_0^2 L_1) = 221.9\;\text{pF}$ via `LumpedElement` produces a clean parallel tank resonance at $f_{\text{res}} = 13.20\;\text{MHz}$ (within $2.6\%$ of $13.56\;\text{MHz}$). With copper skin-depth ($\delta_{\text{Cu}} = 17.95\;\mu\text{m}$) unloaded quality factors $Q_1 \approx 305$ and $Q_2 \approx 268$, the Schormans maximum link efficiency exceeds $90\%$ in the close-coupling zone ($d \le 24\;\text{mm}$) and remains above $70\%$ when loaded to $Q_L = 30$ for $848\;\text{kbps}$ subcarrier modulation.
#
# ### 7.2 Ferrite shielding over a metal battery/chassis plate
#
# * **Ordering and recovery of $L_1$ and $k$ (Figures 7 & 8):** At $d = 18\;\text{mm}$, placing the reader coil $6\;\text{mm}$ above a PEC battery/chassis plate without ferrite induces opposing image eddy currents that reduce $L_1$ from $629.9\;\text{nH}$ to $364.8\;\text{nH}$ ($-42.1\%$), detune the resonance from $13.56\;\text{MHz}$ to $17.82\;\text{MHz}$, and drop $k$ from $0.261$ to $0.160$ ($-38.9\%$). Inserting the flexible NiZn ferrite sheet (`dispersive_lib["nizn ferrite"]`, equivalent sheet permeance $\mu'\cdot t = 30\;\text{mm}$) channels the return magnetic flux horizontally above the PEC plate, restoring $k = 0.299 > k_{\text{free}} > k_{\text{metal}}$ and raising $L_1$ to $1020.4\;\text{nH}$ (retuned to $13.56\;\text{MHz}$ with $C_{1,\text{retuned}} = 135.0\;\text{pF}$).
# * **Thin-sheet $(\mu' - 1) t$ equivalence (Figure 8c):** Across three grid thicknesses $t \in \{1.5, 3.0, 4.5\}\;\text{mm}$ with constant $(\mu'_{\text{eff}} - 1) t = 27\;\text{mm}$ ($\mu'_{\text{eff}} \in \{19, 10, 7\}$), $L_1 \in \{742.3, 784.7, 790.0\}\;\text{nH}$ and $k \in \{0.277, 0.284, 0.286\}$ agree within $5.4\%$ and $2.5\%$, respectively. When the sheet thickness is increased to $t = 7.5\;\text{mm}$ so that it fills nearly the entire $9\;\text{mm}$ coil–metal gap, the top surface of the ferrite approaches the coil traces and the thin-sheet approximation overpredicts $L_1$ by $> 12\%$.
#
# ## 8. Mesh resolution convergence study
#
# Refining the core cell size from $\Delta = 4.5\;\text{mm}$ (staircased across the $3\;\text{mm}$ traces) to $\Delta = 3.0\;\text{mm}$ (1 cell across each trace and gap) and $\Delta = 1.5\;\text{mm}$ (2 cells across each trace and gap, $\sim 1.0 \times 10^5$ cells) (Figure 10c) reduces the relative error in $M(d = 18\;\text{mm})$ and $k(d = 18\;\text{mm})$ against the numerical Neumann reference down to $< 5\%$. Heavier 3D tasks at $\Delta = 1.5\;\text{mm}$ ($\sim 2.8 \times 10^5$ cells) and $\Delta = 1.0\;\text{mm}$ ($\sim 9.2 \times 10^5$ cells) on full padded domains are registered in `--list-tasks` (`--full`) for GPU execution.
#
# ## 9. Limits & failure modes
#
# 1. **Breakdown of background permittivity scaling ($\varepsilon_r = s \gg 1$):** While magnetostatic inductances $L_1, L_2, M$ are independent of $\varepsilon_r$, the distributed inter-turn and underpass bridge capacitance scales as $C_{\text{self}}(s) = s\,C_{\text{self}}(1)$, shifting the coil self-resonant frequency down as $f_{\text{SR}}(s) = f_{\text{SR}}(1) / \sqrt{s}$. As shown in Figure 10a, $s \in \{1, 2.25, 4\}$ changes $L_1, M, k$ at $13.56\;\text{MHz}$ by $< 3.3\%$, whereas at $s = 64$ the shifted self-resonance enters the excitation band and distorts the extracted reactance.
# 2. **Breakdown of the equal-$\mu'\cdot t$ thin-sheet equivalence:** Replacing a $t_{\text{phys}} = 0.1\text{–}0.2\;\text{mm}$ ferrite sheet of permeability $\mu'_{\text{phys}} \approx 150$ by a thicker layer $t_{\text{grid}}$ of scaled permeability $\mu'_{\text{eff}} = 1 + (\mu'_{\text{phys}} - 1) t_{\text{phys}} / t_{\text{grid}}$ is valid only when $t_{\text{grid}}$ remains small compared with both the coil–metal gap and the trace pitch ($t_{\text{grid}} \ll g_{\text{metal}}, p$) and when the ferrite sheet extends laterally past the outer coil perimeter. If $t_{\text{grid}}$ fills the coil–metal gap (Figure 8c, $t = 7.5\;\text{mm}$) or if the coil overhangs the sheet edge, vertical flux entry and edge fringing depend separately on $\mu'$ and $t$.
# 3. **Over-coupling frequency splitting ($k > k_{\text{crit}} = 1 / \sqrt{Q_1 Q_2}$):** When two high-$Q$ tuned coils are brought very close ($d \lesssim 10\;\text{mm}$ so $k Q \gg 1$ without matching network re-optimization), the single-frequency input impedance splits into two coupled normal modes $\omega_\pm = \omega_0 / \sqrt{1 \mp k}$, causing a double-peaked $|S_{21}(f)|$ response and a dip at the carrier $f_0 = 13.56\;\text{MHz}$ unless the port matching impedance is adjusted to $R_{\text{opt}} = R \sqrt{1 + k^2 Q_1 Q_2}$ [4].
# 4. **Nonlinear ferrite saturation at high reader field strength:** ISO/IEC 14443 allows reader magnetic field strengths up to $H_{\max} = 7.5\;\text{A/m}$ (RMS), and during close proximity a thin $0.05\;\text{mm}$ ferrite sheet can concentrate flux density $B \approx \mu_0 \mu' H$ toward the saturation induction $B_{\text{sat}} \approx 0.35\;\text{T}$ of NiZn ferrite. Our linear ADE Lorentz model (`DispersiveMaterial`) does not capture $B$–$H$ hysteresis or saturation roll-off.
#
# ## 10. References
#
# 1. H. A. Wheeler, "Simple Inductance Formulas for Radio Coils," *Proc. IRE* **16**(10), 1398–1400 (1928), DOI: [10.1109/JRPROC.1928.221309](https://doi.org/10.1109/JRPROC.1928.221309).
# 2. S. S. Mohan, M. del Mar Hershenson, S. P. Boyd, and T. H. Lee, "Simple Accurate Expressions for Planar Spiral Inductances," *IEEE J. Solid-State Circuits* **34**(10), 1419–1424 (1999), DOI: [10.1109/4.792620](https://doi.org/10.1109/4.792620).
# 3. F. W. Grover, *Inductance Calculations: Working Formulas and Tables*, D. Van Nostrand, New York (1946); see also E. B. Rosa and F. W. Grover, "Formulas and Tables for the Calculation of Mutual and Self-Inductance," *Bull. Bureau of Standards* **8**(1), 1–237 (1912), DOI: [10.6028/bulletin.185](https://doi.org/10.6028/bulletin.185).
# 4. M. Schormans, V. Valente, and A. Demosthenous, "Practical Inductive Link Design for Biomedical Wireless Power Transfer: A Tutorial," *IEEE Trans. Biomed. Circuits Syst.* **12**(5), 1112–1130 (2018), DOI: [10.1109/TBCAS.2018.2846020](https://doi.org/10.1109/TBCAS.2018.2846020).
# 5. J. Victoria, A. Suárez, P. A. Martínez, A. Alcarria, A. Gerfer, and J. Torres, "Improving the Efficiency of NFC Systems Through Optimizing the Sintered Ferrite Sheet Thickness Selection," *IEEE Trans. Electromagn. Compat.* **62**(4), 1504–1514 (2020), DOI: [10.1109/TEMC.2020.3003800](https://doi.org/10.1109/TEMC.2020.3003800).
# 6. T. Tsutaoka, "Frequency dispersion of complex permeability in Mn–Zn and Ni–Zn spinel ferrites and their composite materials," *J. Appl. Phys.* **93**(5), 2789–2796 (2003), DOI: [10.1063/1.1542651](https://doi.org/10.1063/1.1542651).
# 7. J. L. Snoek, "Dispersion and absorption in magnetic ferrites at frequencies above one Mc/s," *Physica* **14**(4), 207–217 (1948), DOI: [10.1016/0031-8914(48)90038-X](https://doi.org/10.1016/0031-8914(48)90038-X).


# %%
def parse_args(argv=None):
    """Parse CLI arguments for Tutorial 4.6 (`--quick`, `--full`, `--task`, `--list-tasks`, `--plot`, `--out`)."""
    ap = argparse.ArgumentParser(
        description="Tutorial 4.6 · NFC reader and tag coils at 13.56 MHz (near-field inductive coupling)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run fast CPU study tasks and render all 11 figures",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Include fine 1.5 mm and 1.0 mm 3D FIT tasks and render all 11 figures",
    )
    ap.add_argument(
        "--task",
        type=int,
        default=None,
        help="Run a single study task index (0..N-1)",
    )
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the study task table and exit",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate all 11 figures in figures/ from data/nfc_coils_results.npz",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output data directory (default: data/)",
    )
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS")
        argv = shlex.split(env_args) if env_args else sys.argv[1:]
    args, _ = ap.parse_known_args(argv)
    return args


def main(argv=None):
    """Entrypoint for CLI and notebook execution."""
    args = parse_args(argv)
    quick = not args.full

    if args.list_tasks:
        tasks = get_task_table(quick=quick)
        print(f"# Total tasks: {len(tasks)}")
        for t in tasks:
            print(f"{t['index']}\t{t['name']}\t{t['description']}")
        return

    if args.task is not None:
        tasks = get_task_table(quick=quick)
        if not (0 <= args.task < len(tasks)):
            raise SystemExit(f"--task must be in 0..{len(tasks) - 1}")
        compute_all_study_data(
            quick=quick, out_dir=args.out, selected_task=int(args.task)
        )
        return

    if args.plot:
        generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)
        return

    compute_all_study_data(quick=quick, out_dir=args.out, selected_task=None)
    generate_all_figures(data_dir=args.out, fig_dir=FIG_DIR)


if __name__ == "__main__":
    main()
