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
# # Tutorial 8.6 · Traveling-Wave-Tube Small-Signal Gain: A Linearized Electron Beam in a Slow-Wave Structure
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/08_vacuum_electronics/08-06_twt_small_signal_gain/twt_small_signal_gain.ipynb)
#
# In [Tutorial 6.2 · Slow-Wave Structure Cold Test](../../06_periodic_structures/06-02_sws_cold_test/sws_cold_test.ipynb) we characterized the cold electromagnetic dispersion $\omega(\beta_n)$, phase velocity $v_{p,n} = \omega/\beta_n$, group/energy velocity $v_g$, and Pierce interaction impedance $K_n = |E_{z,n}|^2 / (2\beta_n^2 P)$ of two slow-wave structures (SWSs): the **$0.2\;\mathrm{THz}$ glide-symmetric staggered double-vane grating** of Karetnikova *et al.* (2018) and the **V-band ($50\text{–}70\;\mathrm{GHz}$) glide-symmetric pillar waveguide** of Marosi *et al.* (2025). Separately, in [Tutorial 8.1 · Space-Charge Waves on a Drifting Electron Beam](../08-01_space_charge_waves/space_charge_waves.ipynb) we validated `wakis.beam_fluid.LinearizedBeam`, showing that a drifting relativistic electron beam inside a conducting tunnel supports a positive-energy **fast space-charge wave** ($\omega = k_z u_0 + \omega_q$) and a negative-energy **slow space-charge wave** ($\omega = k_z u_0 - \omega_q$), where $\omega_q = R(\beta_e, b, a)\,\omega_p/\gamma_0^{3/2}$ is the geometrically reduced plasma frequency.
#
# In this tutorial we combine the periodic slow-wave circuit of Tutorial 6.2 with the Eulerian linearized fluid beam (`LinearizedBeam`) of Tutorial 8.1 to simulate **traveling-wave-tube (TWT) convective amplification** and **backward-wave-oscillator (BWO) absolute instability** directly in time-domain Finite Integration Technique (FIT) simulations (`SolverFIT3D`):
#
# 1. **Part 1 — Pierce 3-wave & 4-wave small-signal theory, coupled-mode ODEs, and smooth-tunnel SCW check:** We derive Pierce's cubic and quartic dispersion equations from the coupled transmission-line and linearized relativistic fluid equations, prove the launching-loss boundary condition $G(\mathrm{dB}) \approx A + B C N$ with $A = -20\log_{10} 3$ and $B = 20\pi\sqrt{3}\log_{10}\mathrm{e}$, cross-verify the analytic BVP solver against a 1D spatial coupled-mode ODE integrator (`scipy.integrate.solve_ivp`), derive the exact lateral normalization mapping a 2D sheet beam per unit width to a finite-width 3D rectangular waveguide, compare the two-plate sheet-beam space-charge depression factor $D$ of Karetnikova *et al.* (2018) Eq. (4) against Wakis's transcendental planar eigenvalue factor $R_{\mathrm{planar}}$, and verify `LinearizedBeam` time-centring against the analytical space-charge-wave dispersion at the TWT operating resolution (all numerical results are rendered in the Part 1 and Quantitative Summary tables below).
# 2. **Part 2 — 2D FIT hot test of the $0.2\;\mathrm{THz}$ Karetnikova staggered-grating TWT:** We launch a monochromatic $\mathrm{LSE}_{10}$ carrier wave in a smooth input waveguide section, propagate it alongside a sheet electron beam through a glide-symmetric PEC grating into a CPML termination (with beam ramp-down `ramp_cells` before the PML), measure the on-grid cold-circuit parameters ($\beta_{c,\mathrm{FIT}}, K_{\mathrm{2D,FIT}}, K_{\mathrm{3D,FIT}}, \alpha_{0,\mathrm{FIT}} = 0$) and hot exponential growth rate $\alpha_{\mathrm{FIT}} = \operatorname{Im}(k)$ across a 4-resolution node-aligned graded mesh hierarchy (`coarse`, `medium`, `fine`, `ultrafine`), and compare the small-signal gain against the text benchmarks of Karetnikova *et al.* (2018) (`.ref/sws/2018_IEEETED65_0.2THz_TWT_sheet_beam_staggered_grating_gain.txt`, ll. 205–206, 209, 278, 289–291, 328; Figs. 3, 5a, and Fig. 6 at $190\;\mathrm{GHz}$), reporting the Pierce/FIT overprediction as unexplained within our lossless PEC model ($\alpha_{\mathrm{cold}} = 0$) and discussing $\cosh^2(\tau_{+1} y)$ transverse weighting only as an untested hypothesis (see the Part 2 and Quantitative Summary tables for all numbers).
# 3. **Part 3 — Falsifiable physics & mutation tests:** We evaluate five physical tests (plus an algebraic linear-solver scaling check) without calibration factors: (i) the cold limit $I_0 \to 0$ yields zero gain; (ii) flipping `coupling_sign=-1` on the beam-wave coupling conjugates the 1D Pierce roots and moves the 2D FIT peak growth rate from $+b$ to $-b$, whereas exact mirror symmetry fails on the off-peak pair (an asymmetry not predicted by the 3-wave Pierce BVP and reported as unexplained); (iii) while the raw 22-period FIT window slope over $m \in [8, 19]$ does not follow $I_0^{1/3}$, de-embedding $\alpha$ by inverting Pierce's 3-wave BVP at period $19$ with Pierce's own $b = 0$ and $QC$ recovers $\alpha \propto I_0^{1/3}$ at an offset above 1D Pierce across the current sweep; (iv) sweeping the velocity detuning $b$ shifts the gain peak to positive $b > 0$ (while we withdraw the 1D finite-window total-field BVP / crestatron model because it fails for $b \le 0$); and (v) increasing the space-charge parameter $QC$ monotonically reduces the window slope (all numbers rendered from `data/summary.json` in Part 3 below).
# 4. **Part 4 — 3D glide-symmetric V-band SWS hot test (Marosi *et al.*, 2025):** Reusing the 3D cold-test interaction impedance $K_{+1}(\omega)$ and group velocity $v_g(\omega)$ from Tutorial 6.2 (`task_10_3d_marosi_glide.npz`; waveguide dimensions in `.ref/sws/Marosi_2025_glide_sws.txt` Table I ll. 72–74 and sheet-beam dimensions at l. 103) and deriving the cold attenuation constant $\alpha_0$ from the paper's cold insertion loss over $60$ cells, we compare the 60-period V-band small-signal gain curve against Marosi *et al.* (2025) Fig. 7, and report both `coarse_3d` and `medium_3d` 10-period 3D FIT hot tests at the actual simulated $b = 0$ alongside a `wakis.estimate` preflight estimate (`--run-3d-marosi-full60`, a preflight estimate only; the full 60-period run is not implemented).
# 5. **Part 5 — Backward-Wave Oscillation (BWO, Johnson 1955):** Synchronizing the electron beam with the backward ($n = -1$, $v_g < 0$) space harmonic of the grating at $\beta_0 d = 0.5\pi$, we verify Johnson's finite-length boundary-value start-oscillation condition $(CN)_{\mathrm{st}}$ and $b_{\mathrm{st}}$ [Johnson, *Proc. IRE* **43**, 684 (1955)], demonstrate the inverse-cube start-current algebraic identity $I_{\mathrm{st}}(L) \propto L^{-3}$ across multiple interaction lengths $L$, and explain why a lossless 1-cell Bloch unit cell cannot define a finite-length start threshold (claim withdrawn).
#
# ---
#
# ## Learning Goals
#
# - Derive the **Pierce 3-wave cubic and 4-wave quartic dispersion relations** from coupled slow-wave circuit and relativistic `LinearizedBeam` equations, including the gain parameter $C = (\chi_{\mathrm{rel}} I_0 K_n / (4 V_0))^{1/3}$, velocity detuning $b$, space-charge parameter $QC$ with the two-plate tunnel reduction factor $R$, and circuit loss $d = \alpha_0 / (\beta_e C)$.
# - Prove why an unmodulated entering beam ($v_1(0) = I_1(0) = 0$) excites all three Pierce waves equally at $b = QC = d = 0$, producing the **$-9.54\;\mathrm{dB}$ launching loss** ($A = -20\log_{10} 3$) and asymptotic gain slope **$B = 20\pi\sqrt{3}\log_{10}\mathrm{e} \approx 47.26\;\mathrm{dB}$** per unit $CN$.
# - Formulate the **exact 2D-to-3D sheet-beam normalization** ($F_x, W_{\mathrm{eff}}, \eta_x$) that maps a finite-width sheet beam in an $\mathrm{LSE}_{10}$ rectangular waveguide to a 2D FIT domain per unit width while preserving $C^3 = \chi_{\mathrm{rel}} I_0 K_n / (4 V_0)$.
# - Execute **2D and 3D time-domain FIT hot tests** (`SolverFIT3D` + `LinearizedBeam`) on node-aligned graded meshes with $\ge 20\;\mathrm{cells}/\lambda_g$ and $\ge 4$ cells across the $25\;\mu\mathrm{m}$ beam–wall gap, extract cold-circuit parameters on the **same FIT grid**, and de-embed the spatial growth rate $\alpha = \operatorname{Im}(k)$ and total gain $G(\omega)$.
# - Derive the **backward-wave oscillator (BWO) start-oscillation condition** $(CN)_{\mathrm{st}} \approx 0.314$ and $I_{\mathrm{st}}(L) \propto L^{-3}$ from the finite-length collector boundary condition $V_c(L) = 0$, and explain why a lossless 1-cell Bloch unit cell cannot define a finite-length start threshold (claim withdrawn).

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from IPython.display import HTML, Image, Markdown, display
import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (TUTORIAL_DIR / "_helpers.py").exists():
    TUTORIAL_DIR = Path.cwd() / "tutorials/08_vacuum_electronics/08-06_twt_small_signal_gain"
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as h

DATA_DIR = TUTORIAL_DIR / "data"
FIG_DIR = TUTORIAL_DIR / "figures"
SUMMARY = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
print(f"Loaded Tutorial 8.6 pre-computed dataset from {DATA_DIR}")

# %% [markdown]
# ---
#
# ## Theory & Step-by-Step Derivations
#
# ### 1. Coupled Slow-Wave Circuit and Linearized Relativistic Beam (Derived Here)
#
# Consider a periodic slow-wave structure of period $d$ supporting a cold-circuit space harmonic $n$ with axial wavenumber $\beta_c(\omega) = \beta_n(\omega) = \beta_0(\omega) + 2\pi n / d$, phase velocity $v_p = \omega / \beta_c$, cold attenuation constant $\alpha_0$ (so the cold propagation constant is $\Gamma_0 = \alpha_0 + \mathrm{j}\beta_c$ in Pierce's $\exp(\mathrm{j}\omega t - \Gamma z)$ phasor convention), and beam-averaged **Pierce interaction impedance**
#
# $$
# K_n(\omega) = \frac{\langle |E_{z,n}|^2 \rangle_{\mathrm{beam}}}{2\,\beta_n^2\,P_{\mathrm{RF}}}.
# $$
#
# A magnetically confined (`axis="z"`) electron beam of DC current $I_0$, DC accelerating voltage $V_0$, relativistic Lorentz factor $\gamma_0 = 1 + e V_0 / (m_e c^2)$, and drift velocity $u_0 = \beta_{\mathrm{rel}} c = c\sqrt{1 - \gamma_0^{-2}}$ travels through the vacuum tunnel alongside the wave. In `wakis.beam_fluid.LinearizedBeam`, the small-signal perturbations of axial velocity $v_1(z, t)$, charge density $\rho_1(z, t)$, and axial current $I_1(z, t) = \int_{\mathcal{A}_b} (\rho_0 v_1 + u_0 \rho_1)\,\mathrm{d}A$ obey the linearized relativistic Euler and charge-continuity equations:
#
# $$
# \left(\frac{\partial}{\partial t} + u_0 \frac{\partial}{\partial z}\right) v_1(z, t) = -\frac{e}{\gamma_0^3 m_e}\left[E_{z,\mathrm{ckt}}(z, t) + E_{z,\mathrm{sc}}(z, t)\right], \qquad \frac{\partial \rho_1}{\partial t} + u_0 \frac{\partial \rho_1}{\partial z} + \rho_0 \frac{\partial v_1}{\partial z} = 0.
# $$
#
# For a harmonic perturbation $\propto \exp(\mathrm{j}\omega t - \Gamma z)$ with electronic wavenumber $\beta_e \equiv \omega / u_0$, the operator $\partial_t + u_0 \partial_z$ becomes $\mathrm{j} u_0 (\beta_e + \mathrm{j}\Gamma)$. Using the relativistic beam kinetic voltage relation $\frac{1}{2}\gamma_0(\gamma_0 + 1) m_e u_0^2 = e V_0$, the prefactor $e / (\gamma_0^3 m_e u_0)$ can be written in terms of the classical DC beam impedance $V_0 / I_0$ multiplied by the relativistic mass factor $\chi_{\mathrm{rel}} \equiv \frac{2}{\gamma_0(\gamma_0 + 1)}$ ($\chi_{\mathrm{rel}} \to 1$ non-relativistically; at $V_0 = 20\;\mathrm{kV}$, $\gamma_0 = 1.039139$ and $\chi_{\mathrm{rel}} = 0.94386$).
#
# As derived in [Tutorial 8.1](../08-01_space_charge_waves/space_charge_waves.ipynb) and **Mealy, Marosi & Capolino (2022)**, the self-consistent axial space-charge field $E_{z,\mathrm{sc}}$ of the bunched beam inside a conducting tunnel is related to the AC beam current $I_1$ by the reduced plasma wavenumber $\beta_q = \omega_q / u_0 = R(\gamma_0\beta_e b, \gamma_0\beta_e a)\,\omega_p / (\gamma_0^{3/2} u_0)$:
#
# $$
# E_{z,\mathrm{sc}}(\Gamma) = -\frac{\mathrm{j}\,\beta_q^2}{\beta_e}\left(\frac{2 V_0}{\chi_{\mathrm{rel}} I_0}\right) I_1(\Gamma).
# $$
#
# Eliminating $v_1$ and $\rho_1$ yields the **electronic (beam) equation** linking the induced AC beam current $I_1$ to the synchronous circuit field $E_{z,\mathrm{ckt}} = -\Gamma V_c$:
#
# $$
# \left[(\mathrm{j}\beta_e - \Gamma)^2 + \beta_q^2\right] I_1 = -\mathrm{j}\beta_e\,\frac{\chi_{\mathrm{rel}} I_0}{2 V_0}\,\Gamma V_c.
# $$
#
# Meanwhile, by Pierce's transmission-line circuit equation [Pierce, *Proc. IRE* **35**, 111 (1947); *Traveling-Wave Tubes* (1950), Ch. II & VII], the AC convection current $I_1$ drives both the forward ($\sim -\Gamma_0$) and backward ($\sim +\Gamma_0$) circuit waves:
#
# $$
# \left(\Gamma^2 - \Gamma_0^2\right) V_c = -\Gamma\,\Gamma_0\,K_n\,I_1.
# $$
#
# Multiplying the electronic and circuit equations together eliminates $V_c$ and $I_1$, yielding the **4-wave quartic Pierce dispersion relation** (derived here):
#
# $$
# \boxed{\left(\Gamma^2 - \Gamma_0^2\right)\left[(\mathrm{j}\beta_e - \Gamma)^2 + \beta_q^2\right] = \mathrm{j}\beta_e\,\Gamma^2\,\Gamma_0\,\frac{\chi_{\mathrm{rel}} I_0 K_n}{2 V_0} = 2\mathrm{j}\beta_e\,\Gamma^2\,\Gamma_0\,C^3,}
# $$
#
# where we define the **Pierce gain parameter** $C$, **velocity detuning parameter** $b$, **space-charge parameter** $QC$, and **cold-loss parameter** $d$:
#
# $$
# C \equiv \left(\chi_{\mathrm{rel}}\,\frac{I_0 K_n}{4 V_0}\right)^{1/3}, \qquad b \equiv \frac{\beta_c - \beta_e}{\beta_e C} = \frac{u_0 - v_p}{C v_p}, \qquad 4 QC \equiv \left(\frac{\beta_q}{\beta_e C}\right)^2 = \left(\frac{\omega_q}{\omega C}\right)^2, \qquad d \equiv \frac{\alpha_0}{\beta_e C}.
# $$
#
# Introducing Pierce's normalizedincremental propagation constant $\delta = x + \mathrm{j}y$ via
#
# $$
# \Gamma = \mathrm{j}\beta_e - \beta_e C\,\delta, \qquad \Gamma_0 = \beta_e C\,d + \mathrm{j}\beta_e(1 + C b),
# $$
#
# so that $\operatorname{Re}(\Gamma) = -\beta_e C x$ and a wave growing as $\exp(+\alpha z)$ has spatial growth rate **$\alpha = \beta_e C x_1 > 0$** ($x_1 = \operatorname{Re}(\delta_1) > 0$), the exact 4-wave quartic polynomial in $\delta$ is:
#
# $$
# \left[(1 + \mathrm{j}C\delta)^2 - (1 + C b - \mathrm{j}C d)^2\right](\delta^2 + 4 QC) = -2 C^3 (1 + \mathrm{j}C\delta)^2 (1 + C b - \mathrm{j}C d).
# $$
#
# In the small-$C$ forward-wave limit ($C \ll 1$, retaining the three forward waves near $\Gamma \approx \mathrm{j}\beta_e$ and factoring out the non-synchronous backward circuit wave $\Gamma_4 \approx -\mathrm{j}\beta_c$), we have $\Gamma + \Gamma_0 \approx 2\mathrm{j}\beta_e$, $\Gamma^2 \Gamma_0 \approx -\mathrm{j}\beta_e^3$, and $\Gamma - \Gamma_0 = -\beta_e C(\delta + \mathrm{j}b + d) = \mathrm{j}\beta_e C(\mathrm{j}\delta - b + \mathrm{j}d)$. Substituting these into the quartic reduces it directly to **Pierce's classical 3-wave cubic dispersion equation**:
#
# $$
# \boxed{(\delta^2 + 4 QC)(\mathrm{j}\delta - b + \mathrm{j}d) = -1 \quad \Longleftrightarrow \quad \delta(\delta^2 + 4 QC) + (d + \mathrm{j}b)(\delta^2 + 4 QC) + \mathrm{j} = 0.}
# $$
#
# ---
#
# ### 2. Launching-Loss Boundary Conditions ($A \approx -9.54\;\mathrm{dB}$, $B \approx 47.26\;\mathrm{dB}$) and Coupled-Mode ODEs (Derived Here)
#
# Why does a traveling-wave tube of $N = \beta_e L / (2\pi)$ electronic wavelengths not simply amplify an input signal by $\exp(\beta_e C x_1 L)$ from $z = 0$? Because the electron beam enters the slow-wave structure at $z = 0$ **unmodulated** ($v_1(0) = 0$ and $I_1(0) = 0$), while the RF input power is applied purely to the circuit wave ($V_c(0) = V_{\mathrm{in}}$).
#
# Each of the three forward roots $\delta_k$ ($k = 1, 2, 3$) of the Pierce cubic carries its own polarization ratio between circuit voltage $V_k$, AC velocity $v_{1,k} \propto \delta_k V_k / (\delta_k^2 + 4 QC)$, and AC current $I_{1,k} \propto V_k / (\delta_k^2 + 4 QC)$. Imposing $V_c(0) = V_{\mathrm{in}}$, $v_1(0) = 0$, and $I_1(0) = 0$ yields the $3\times 3$ linear boundary-value system (derived here):
#
# $$
# \begin{bmatrix}
# 1 & 1 & 1 \\
# \dfrac{\delta_1}{\delta_1^2 + 4 QC} & \dfrac{\delta_2}{\delta_2^2 + 4 QC} & \dfrac{\delta_3}{\delta_3^2 + 4 QC} \\[8pt]
# \dfrac{1}{\delta_1^2 + 4 QC} & \dfrac{1}{\delta_2^2 + 4 QC} & \dfrac{1}{\delta_3^2 + 4 QC}
# \end{bmatrix}
# \begin{bmatrix} V_1 \\ V_2 \\ V_3 \end{bmatrix}
# =
# \begin{bmatrix} V_{\mathrm{in}} \\ 0 \\ 0 \end{bmatrix}.
# $$
#
# At cold-circuit synchronism ($b = 0$) with negligible space charge ($QC = 0$) and zero cold loss ($d = 0$), the cubic is simply $\delta^3 = -\mathrm{j} = \mathrm{e}^{-\mathrm{j}\pi/2}$, whose three roots are:
#
# $$
# \delta_1 = \frac{\sqrt{3}}{2} - \frac{\mathrm{j}}{2} \quad (\text{growing}), \qquad
# \delta_2 = -\frac{\sqrt{3}}{2} - \frac{\mathrm{j}}{2} \quad (\text{decaying}), \qquad
# \delta_3 = +\mathrm{j} \quad (\text{unattenuated fast wave}).
# $$
#
# Since $\delta_k^3 = -\mathrm{j}$ for all three roots, $1/\delta_k^2 = \mathrm{j}\delta_k$ and $\delta_k / \delta_k^2 = 1/\delta_k = \mathrm{j}\delta_k^2$. By symmetry of the cube roots of $-\mathrm{j}$ ($\delta_1 + \delta_2 + \delta_3 = 0$ and $\delta_1^2 + \delta_2^2 + \delta_3^2 = 0$), the unique solution at $b = QC = d = 0$ is **equal partitioning of the input voltage among the three waves**:
#
# $$
# V_1 = V_2 = V_3 = \frac{V_{\mathrm{in}}}{3}.
# $$
#
# Summing the three waves along $z = 2\pi N / \beta_e$ gives the total circuit transfer function
#
# $$
# \frac{V_c(z)}{V_{\mathrm{in}}}\,\mathrm{e}^{\mathrm{j}\beta_e z} = \sum_{k=1}^3 \frac{V_k}{V_{\mathrm{in}}}\,\mathrm{e}^{2\pi C N \delta_k}.
# $$
#
# Near the input ($C N \lesssim 0.2$), all three waves have comparable amplitudes and beat against one another (the **three-wave launching region** where $|V_c(z)|$ is initially flat or slightly depressed if $d > 0$). Once $C N \gtrsim 0.3$, the growing wave $\delta_1$ ($x_1 = \sqrt{3}/2 \approx 0.866025$) dominates, yielding Pierce's famous **asymptotic decibel gain formula** (derived here):
#
# $$
# \boxed{G_{\mathrm{asymp}}(\mathrm{dB}) = 20\log_{10}\!\left|\frac{V_1}{V_{\mathrm{in}}}\,\mathrm{e}^{2\pi C N x_1}\right| = A + B\,C\,N,}
# $$
#
# where at $b = QC = d = 0$:
#
# $$
# A = 20\log_{10}\!\left(\frac{1}{3}\right) = -20\log_{10} 3 = -9.542425\;\mathrm{dB}, \qquad
# B = 20(\log_{10}\mathrm{e})\cdot 2\pi x_1 = 20\pi\sqrt{3}\log_{10}\mathrm{e} = 47.263383\;\mathrm{dB}.
# $$
#
# As an independent second reference in `_helpers.py` (`solve_coupled_mode_ode`), we also integrate the first-order spatial coupled-mode differential equations directly from $z = 0$ to $z = L$ using `scipy.integrate.solve_ivp` (`DOP853`, `rtol=1e-10`):
#
# $$
# \frac{\mathrm{d}}{\mathrm{d}z}
# \begin{bmatrix} \tilde{V}_c(z) \\ \tilde{U}_1(z) \\ \tilde{I}_1(z) \end{bmatrix}
# =
# -\mathrm{j}\beta_e
# \begin{bmatrix}
# C(b - \mathrm{j}d) & 0 & -C^2 Z_0 \\
# C & 0 & 4 QC \cdot C Z_0 \\
# 0 & C / Z_0 & 0
# \end{bmatrix}
# \begin{bmatrix} \tilde{V}_c(z) \\ \tilde{U}_1(z) \\ \tilde{I}_1(z) \end{bmatrix},
# \qquad Z_0 \equiv \frac{2 V_0}{\chi_{\mathrm{rel}} I_0},
# $$
#
# which reproduces the 3-wave Pierce BVP across all $(b, QC, d)$ to `< 3e-8 dB` numerical precision.
#
# ---
#
# ### 3. Mapping a 2D Sheet Beam per Unit Width to a 3D Rectangular Grating (Derived Here)
#
# In Karetnikova *et al.* (2018), the $0.2\;\mathrm{THz}$ staggered double-vane grating sits inside a rectangular waveguide enclosure of width $b_{\mathrm{wg}} = 0.85\;\mathrm{mm}$ along $x$, and the fundamental $\mathrm{LSE}_{10}$ mode has transverse variation $E_z(x, y, z) = E_z^{\mathrm{peak}}(y, z)\sin(\pi x / b_{\mathrm{wg}})$ with transverse wavenumber $k_x = \pi / b_{\mathrm{wg}}$. In our 2D FIT hot test (`nx = 1` cell in $x$ with Bloch boundary conditions imposing the exact discrete $k_x = \pi / b_{\mathrm{wg}}$), the fields and sheet beam are uniform in $x$. How do the 3D beam current $I_0 = 100\;\mathrm{mA}$ (over sheet width $W_b = 0.75\;\mathrm{mm}$) and 3D Pierce impedance $K_{\mathrm{3D}}$ map to the 2D sheet current per unit width $J_{\mathrm{sheet}}$ [$\mathrm{A/m}$] and 2D impedance per unit width $K_{\mathrm{2D}}$ [$\Omega\cdot\mathrm{m}$]?
#
# 1. **3D power integration across $b_{\mathrm{wg}}$:** Because $\int_0^{b_{\mathrm{wg}}} \sin^2(\pi x / b_{\mathrm{wg}})\,\mathrm{d}x = b_{\mathrm{wg}} / 2$, the total 3D RF power is $P_{\mathrm{3D}} = (b_{\mathrm{wg}}/2)\,p_{\mathrm{2D}}$, where $p_{\mathrm{2D}}$ is the 2D power per unit width at the mid-plane $x = b_{\mathrm{wg}}/2$.
# 2. **Lateral beam-averaging factor $F_x$:** Averaging $|E_{z,n}(x, y)|^2 = |E_{z,n}^{\mathrm{peak}}(y)|^2 \sin^2(\pi x / b_{\mathrm{wg}})$ across the centered beam width $x \in [(b_{\mathrm{wg}}-W_b)/2, (b_{\mathrm{wg}}+W_b)/2]$ multiplies the peak 2D impedance by the lateral form factor (derived here):
#
# $$
# F_x \equiv \frac{1}{W_b}\int_{(b_{\mathrm{wg}}-W_b)/2}^{(b_{\mathrm{wg}}+W_b)/2} \sin^2\!\left(\frac{\pi x}{b_{\mathrm{wg}}}\right)\mathrm{d}x = \frac{1}{2}\left[1 + \frac{b_{\mathrm{wg}}}{\pi W_b}\sin\!\left(\frac{\pi W_b}{b_{\mathrm{wg}}}\right)\right].
# $$
#
#    For $W_b = 0.75\;\mathrm{mm}$ and $b_{\mathrm{wg}} = 0.85\;\mathrm{mm}$, we obtain **$F_x = 0.565159$**.
# 3. **Exact preservation of $C^3 = I_0 K_{\mathrm{3D}} / (4 V_0)$:** Combining $P_{\mathrm{3D}} = (b_{\mathrm{wg}}/2)\,p_{\mathrm{2D}}$ and $\langle |E_z|^2\rangle_{\mathrm{3D}} = F_x \langle |E_z|^2\rangle_{\mathrm{2D}}$ gives
#
# $$
# K_{\mathrm{3D}} = \frac{F_x}{b_{\mathrm{wg}}/2}\,K_{\mathrm{2D}} \equiv \frac{K_{\mathrm{2D}}}{W_{\mathrm{eff}}}, \qquad W_{\mathrm{eff}} \equiv \frac{b_{\mathrm{wg}}}{2 F_x} = 0.752001\;\mathrm{mm}.
# $$
#
#    Therefore, setting the effective 2D sheet current per unit width in `LinearizedBeam` to
#
# $$
# \boxed{J_{\mathrm{sheet,eff}} = \frac{I_0}{W_{\mathrm{eff}}} = \eta_x\,\frac{I_0}{W_b}, \qquad \eta_x \equiv \frac{2 F_x W_b}{b_{\mathrm{wg}}} = 0.997340,}
# $$
#
#    guarantees that the 2D FIT coupling product $J_{\mathrm{sheet,eff}} K_{\mathrm{2D}} / (4 V_0)$ equals the 3D coupling product $I_0 K_{\mathrm{3D}} / (4 V_0) = C^3 / \chi_{\mathrm{rel}}$ **identically**.
#
# **Two-plate sheet-beam space-charge reduction factor ($R_{\mathrm{planar}}$ vs Karetnikova Eq. 4):** Karetnikova *et al.* (2018) Eq. (4) gives the closed-form cross-section-averaged space-charge depression factor $D$ for a sheet electron beam of half-thickness $h = H_b / 2 = 75\;\mu\mathrm{m}$ centered **between two conducting metal plates** of tunnel half-height $a = 100\;\mu\mathrm{m}$:
#
# $$
# D^2(\gamma h, \gamma a) = 1 - \frac{\sinh(\gamma h)\cosh\!\left[\gamma h\left(\dfrac{a}{h} - 1\right)\right]}{\gamma h\,\cosh(\gamma a)}, \qquad \gamma \equiv \frac{\beta_e}{\gamma_0}.
# $$
#
# At $f_0 = 186\;\mathrm{GHz}$ and $V_0 = 20\;\mathrm{kV}$ ($\gamma h = 1.0349$, $\gamma a = 1.3799$), Eq. (4) yields **$D = 0.635475 \approx 0.635$**. By comparison, `wakis.beam_fluid.plasma_reduction_factor_planar` solves the exact transcendental space-charge-wave eigenvalue equation $\kappa_b \tan(\kappa_b h) = \gamma \coth[\gamma (a - h)]$ for the same two-plate geometry [Branch & Mihran (1955); Mealy, Marosi & Capolino (2022)], yielding **$R_{\mathrm{planar}} = 0.652264 \approx 0.652$** (a $2.64\%$ difference between the cross-section-averaged Green's function approximation of Eq. (4) and the fundamental eigenmode reduction factor $R_{\mathrm{planar}}$).
#
# ---
#
# ### 4. Backward-Wave Oscillator (BWO) Start-Oscillation Condition (Johnson 1955; Derived Here)
#
# When the electron beam is synchronous with a **backward space harmonic** ($\beta_c = \beta_{-1} = \beta_0 - 2\pi / d < 0$ or in folded zone $\beta_{-1} d \in (\pi, 2\pi)$ where $v_p = \omega / \beta_{-1} > 0$ along $+z$ while the group velocity $v_g = \mathrm{d}\omega / \mathrm{d}\beta_{-1} < 0$ carries electromagnetic energy backward toward the gun end $z = 0$), the sign of the circuit power flow reverses ($P_{\mathrm{RF}} \to -P_{\mathrm{RF}}$). This flips the sign on the right-hand side of Pierce's 3-wave dispersion equation [Johnson, *Proc. IRE* **43**, 684 (1955), Eq. (4)]:
#
# $$
# \boxed{(\delta^2 + 4 QC)(\mathrm{j}\delta - b + \mathrm{j}d) = +1 \qquad (\text{backward-wave interaction, } v_g < 0).}
# $$
#
# Because electromagnetic energy propagates from $z = L$ toward $z = 0$, a finite-length BWO requires **no input signal at $z = 0$**: instead, the circuit wave starts at zero amplitude at the collector end **$V_c(L) = 0$**, while the beam enters unmodulated at the gun end **$v_1(0) = 0, I_1(0) = 0$**. Evaluating $V_c(L) = \sum_{k=1}^3 V_k \mathrm{e}^{2\pi C N \delta_k}$ with the gun-end amplitudes $V_k$ yields a self-sustained oscillation ($V_c(0) \neq 0$ for $V_c(L) = 0$) if and only if the complex BWO secular determinant vanishes (derived here):
#
# $$
# \Delta_{\mathrm{BWO}}(CN, b) \equiv \sum_{k=1}^3 (\delta_k^2 + 4 QC)(\delta_m - \delta_n)\,\mathrm{e}^{2\pi C N \delta_k} = 0,
# $$
#
# where $(k, m, n)$ is a cyclic permutation of $(1, 2, 3)$. Solving $\operatorname{Re}\Delta_{\mathrm{BWO}} = \operatorname{Im}\Delta_{\mathrm{BWO}} = 0$ simultaneously in the space-charge-free, lossless limit ($QC = 0, d = 0$) yields the fundamental axial BWO start-oscillation point:
#
# $$
# \boxed{(CN)_{\mathrm{st}} = 0.314055, \qquad b_{\mathrm{st}} = 1.521839 \qquad (\text{Johnson 1955 Table I: } 0.314 \text{ and } 1.522).}
# $$
#
# Since $C N = \left(\chi_{\mathrm{rel}}\frac{I_0 K_{-1}}{4 V_0}\right)^{1/3} \frac{\beta_e L}{2\pi}$, solving $\Delta_{\mathrm{BWO}}(I_0, b; L) = 0$ at each interaction length $L$ proves as an exact Pierce algebraic identity that **$I_{\mathrm{st}}(L)$ scales as the inverse cube of the interaction length $L$**:
#
# $$
# \boxed{I_{\mathrm{st}}(L) = \frac{4 V_0}{\chi_{\mathrm{rel}} K_{-1}}\left(\frac{2\pi (CN)_{\mathrm{st}}}{\beta_e L}\right)^3 \propto L^{-3}.}
# $$
#
# By contrast, a lossless 1-cell Bloch-periodic unit cell has no finite length $L$ and no collector boundary condition $V_c(L) = 0$; it cannot define a finite-length start-oscillation threshold (claim withdrawn).

# %% [markdown]
# ---
#
# ## Simulation Setup, Schematic & Resource Preflight
#
# Figures `00a_setup_geometry.png` and `00b_setup_mesh_and_excitation.png` (generated with `wakis.schematic.save_setup_figures`) together with the markdown setup table `00_setup_table.md` show the 2D FIT hot-test configuration for the 22-period Karetnikova $0.2\;\mathrm{THz}$ glide-symmetric staggered double-vane grating (`z in [-1.0, 12.0] mm`, $13.0\;\mathrm{mm}$ total length):
# - **Smooth input/output waveguide sections:** $2$ periods ($1.0\;\mathrm{mm}$) at each end ($z \in [-1.0, 0]\;\mathrm{mm}$ and $z \in [11.0, 12.0]\;\mathrm{mm}$) allow clean $\mathrm{LSE}_{10}$ carrier injection at $z = -0.60\;\mathrm{mm}$ and reflectionless absorption into 8-cell CPML boundaries.
# - **Glide-symmetric PEC grating:** $N = 22$ periods ($z \in [0, 11.0]\;\mathrm{mm}$, period $d = 0.50\;\mathrm{mm}$) with lower teeth of width $s = 100\;\mu\mathrm{m}$ and depth $h = 300\;\mu\mathrm{m}$ at $z_m + d/4$ ($y \in [-400, -100]\;\mu\mathrm{m}$) and upper teeth shifted by $+d/2$ at $z_m + 3d/4$ ($y \in [+100, +400]\;\mu\mathrm{m}$), bounding a $2a = 200\;\mu\mathrm{m}$ vacuum tunnel.
# - **Linearized sheet electron beam (`LinearizedBeam`):** Occupies $|y| \le b = 75\;\mu\mathrm{m}$ ($H_b = 150\;\mu\mathrm{m}$) across $z \in [0.5, 10.5]\;\mathrm{mm}$ inside the $13.0\;\mathrm{mm}$ domain, with `ramp_cells` cosine tapering before the CPML absorbers so advected space-charge waves never strike the PML boundary.
# - **Node-aligned graded mesh (`coarse`, `medium`, `fine`, `ultrafine`):** Places exact grid nodes on every PEC vane edge, tunnel wall ($y = \pm 100\;\mu\mathrm{m}$), and beam boundary ($y = \pm 75\;\mu\mathrm{m}$), resolving the $25\;\mu\mathrm{m}$ beam–wall gap with $\ge 4$ cells and the guided wavelength $\lambda_g = 2\pi/\beta_{+1}$ at $186\;\mathrm{GHz}$ with $\ge 20\;\mathrm{cells}/\lambda_g$ (exact cell counts for each level are rendered from `data/summary.json` in `00_setup_table.md` and the Part 2.1 table below).

# %%
display(Image(filename=str(FIG_DIR / "00a_setup_geometry.png"), width=960))
display(Markdown((FIG_DIR / "00_setup_table.md").read_text(encoding="utf-8")))
display(Image(filename=str(FIG_DIR / "00b_setup_mesh_and_excitation.png"), width=960))

# %% [markdown]
# ---
#
# ## Part 1 Results · Pierce Reference Solvers, Sheet-Beam Normalization & Space-Charge-Wave Check
#
# Figure `01_pierce_roots_and_scw_check.png` validates our theoretical and numerical foundation before turning on the corrugated grating (all numerical values are rendered from `data/summary.json` below):

# %%
t1 = SUMMARY["task_01_pierce_and_scw"]
scw = t1["scw_check"]
m2d = t1["sheet_2d_to_3d_mapping"]
scw_errs_md = ", ".join(f"{lv['rel_err_wq']*100:.2f}\\%" for lv in scw["levels"])
scw_nz_md = ", ".join(str(lv["nz_per_lambda_e"]) for lv in scw["levels"])

display(
    Markdown(
        f"- **Panel (a) — Pierce growing root $x_1 = \\operatorname{{Re}}(\\delta_1)$ vs velocity detuning $b$ and space charge $QC$:** At $QC = 0$, maximum growth $x_1 = \\sqrt{{3}}/2 = {np.sqrt(3.0)/2.0:.6f}$ occurs at cold-circuit synchronism $b = 0$ and cuts off sharply above $b_{{\\max}} = 3 / 2^{{4/3}} = {3.0 / (2.0 ** (4.0 / 3.0)):.4f}$. As $QC$ increases, space-charge debunching reduces peak $x_1$ and shifts the optimum velocity parameter toward **$b \\approx \\sqrt{{4 QC}} > 0$** ($u_0 > v_p$), where the cold circuit wave is phase-matched to the **slow space-charge wave** ($\\beta_e + \\beta_q$).\n"
        f"- **Panel (b) — 3-wave BVP vs 1D coupled-mode spatial ODE:** Both independent solvers agree to **${t1['max_bvp_vs_ode_diff_db']:.2e}\\;\\mathrm{{dB}}$** across $CN \\in [0, 1.2]$, exhibiting the three-wave launching beat for $CN \\lesssim 0.25$ and converging onto the exact asymptote $G_{{\\mathrm{{asymp}}}} = {t1['A_launching_loss_db']:.4f} + {t1['B_pierce_slope_db']:.4f}\\,CN$ [dB].\n"
        f"- **Panel (c) — Smooth-tunnel `LinearizedBeam` time-centring check:** Following Tutorial 8.1, driving `LinearizedBeam` inside the $2a = 200\\;\\mu\\mathrm{{m}}$ smooth tunnel at the TWT beam parameters ($V_0 = 20\\;\\mathrm{{kV}}, f_0 = 186\\;\\mathrm{{GHz}}, \\lambda_e = {scw['lambda_e_um']:.1f}\\;\\mu\\mathrm{{m}}$) reproduces the exact planar-tunnel reduced plasma frequency $\\omega_q = R_{{\\mathrm{{planar}}}}\\,\\omega_p/\\gamma_0^{{3/2}}$ ($R_{{\\mathrm{{planar}}}} = {scw['R_planar_wakis']:.6f}$) to **${scw_errs_md}$** across $N_z/\\lambda_e = {scw_nz_md}$, with fast/slow beat-envelope conservation ratio **${scw['levels'][-1]['amplitude_conservation_ratio']:.4f}$** (zero spurious numerical growth or damping from the staggered leapfrog time integrator)."
    )
)

display(Image(filename=str(FIG_DIR / "01_pierce_roots_and_scw_check.png"), width=980))

print(
    f"Part 1 — Pierce Theory, Coupled-Mode ODE & Smooth-Tunnel SCW Summary:\n"
    f"  Launching loss A (b=QC=d=0)       : {t1['A_launching_loss_db']:.6f} dB (exact -20*log10(3))\n"
    f"  Pierce asymptotic slope B         : {t1['B_pierce_slope_db']:.6f} dB / CN (exact 20*pi*sqrt(3)*log10(e))\n"
    f"  Max |G_BVP - G_ODE| discrepancy   : {t1['max_bvp_vs_ode_diff_db']:.3e} dB\n"
    f"  2D-to-3D lateral factor F_x       : {m2d['fx_lateral']:.6f} (W_eff = {m2d['w_eff_m']*1e3:.4f} mm, eta_x = {m2d['eta_x_current_factor']:.6f})\n"
    f"  Two-plate args (186 GHz, 20 kV)   : gamma*h = {scw['gamma_h_186ghz']:.4f}, gamma*a = {scw['gamma_a_186ghz']:.4f}\n"
    f"  Planar reduction factor R_planar  : {scw['R_planar_wakis']:.6f} vs Karetnikova Eq. (4) D = {scw['D_karetnikova_eq4']:.6f}\n"
    f"  Smooth-tunnel SCW omega_q error   : "
    + ", ".join(
        f"{lv['rel_err_wq']*100:.2f}% (Nz/lambda_e={lv['nz_per_lambda_e']}, amp_ratio={lv['amplitude_conservation_ratio']:.4f})"
        for lv in scw["levels"]
    )
)

# %% [markdown]
# ---
#
# ## Part 2 Results · 2D FIT Hot Test of the $0.2\;\mathrm{THz}$ Staggered Grating: Spatial Growth, 3-Wave Beating & Karetnikova (2018) Gain Benchmark
#
# ### 2.1 Spatial Growth Rate vs Length, Three-Wave Launching Beat & 4-Level Same-Grid Convergence
#
# We run a 4-resolution 2D FIT spatial convergence study on the $22$-period grating at $f_0 = 186.0\;\mathrm{GHz}$ (`coarse`, `medium`, `fine`, `ultrafine`). On each grid level, we first extract the cold-circuit wavenumber $\beta_{c,\mathrm{FIT}}(f_0)$, interaction impedance $K_{\mathrm{2D,FIT}}$ ($K_{\mathrm{3D,FIT}} = K_{\mathrm{2D,FIT}} / W_{\mathrm{eff}}$), and cold loss $\alpha_{0,\mathrm{FIT}} = 0$ from a 1-period Bloch eigenmode run **on that same FIT grid**, predict the on-grid Pierce growth rate $\alpha_{\mathrm{Pierce,grid}}$, and then run a cold pass ($I_0 = 0$) and a hot pass (`LinearizedBeam` active at $I_0 = 250\;\mathrm{mA}$) on the $22$-period grating to fit $\alpha_{\mathrm{FIT}}$ over $m \in [8, 19]$ (all convergence numbers are rendered from `data/summary.json` below):

# %%
t2 = SUMMARY["task_02_2d_convergence"]
t2_p_chain = " \\to ".join(f"{lv['alpha_pierce_grid_np_m']:.2f}" for lv in t2["levels"])
t2_f_chain = " \\to ".join(f"{lv['alpha_fit_np_m']:.2f}" for lv in t2["levels"])
t2_pw_md = ", ".join(f"{p:.2f}" for p in t2["pairwise_orders"])
t2_inc_md = ", ".join(f"{x:.2f}" for x in t2["alpha_fit_increments_np_m"])

display(
    Markdown(
        f"Figure `02_twt_gain_vs_length_and_convergence.png` shows:\n"
        f"- **Panel (a) — Three-wave launching beat & spatial gain $G(z_m) = 20\\log_{{10}}|r_m|$:** Over the first $5$ periods ($z \\lesssim 2.5\\;\\mathrm{{mm}}$), interference among the growing ($\\delta_1$), decaying ($\\delta_2$), and fast ($\\delta_3$) waves produces the Pierce launching dip ($\\min G = {t2['levels'][1]['min_launching_dip_db']:.2f}\\;\\mathrm{{dB}}$) before transitioning into linear dB/mm exponential growth for $z \\in [4, 10]\\;\\mathrm{{mm}}$.\n"
        f"- **Panel (b) — Exponential growth of AC beam current $|I_1(z)|$:** Starting from zero modulation at $z = 0$, the AC beam bunching current grows exponentially across two orders of magnitude with the Pierce spatial growth rate.\n"
        f"- **Panel (c) — 4-level grid convergence (pre-asymptotic pairwise orders ${t2_pw_md}$; increments ${t2_inc_md}\\;\\mathrm{{Np/m}}$):** Both the same-grid Pierce prediction $\\alpha_{{\\mathrm{{Pierce,grid}}}}$ (${t2_p_chain}\\;\\mathrm{{Np/m}}$) and the hot 2D FIT growth rate $\\alpha_{{\\mathrm{{FIT}}}}$ (${t2_f_chain}\\;\\mathrm{{Np/m}}$) decrease monotonically across `coarse`, `medium`, `fine`, and `ultrafine`, reducing the same-grid relative error from **${t2['levels'][0]['rel_err_vs_pierce_grid']*100:.2f}\\%$** (`coarse`) to **${t2['levels'][1]['rel_err_vs_pierce_grid']*100:.2f}\\%$** (`medium`), **${t2['levels'][2]['rel_err_vs_pierce_grid']*100:.2f}\\%$** (`fine`), and **${t2['levels'][3]['rel_err_vs_pierce_grid']*100:.2f}\\%$** (`ultrafine`). Because the pairwise convergence orders across consecutive refinements oscillate (${t2_pw_md}$, with a 4-point log-log fit slope of $p_{{\\mathrm{{obs}}}} = {t2['observed_order_p']:.3f}$), the 4-level data are **pre-asymptotic** and no Richardson extrapolation is claimed as a converged value."
    )
)

display(Image(filename=str(FIG_DIR / "02_twt_gain_vs_length_and_convergence.png"), width=980))

conv_rows = [
    "| Grid Level | $N_x \\times N_y \\times N_z$ (Cells) | Cells/$\\lambda_g$ ($436.1\\;\\mu$m) | Cells / Gap ($25\\;\\mu$m) | $K_{\\mathrm{3D,FIT}}$ [$\\Omega$] | $\\alpha_{\\mathrm{Pierce,grid}}$ [Np/m] | $\\alpha_{\\mathrm{FIT}}$ [Np/m] | Rel. Error vs Same-Grid Pierce | Pairwise Order $p_i$ | Wall Time Cold+Hot [s] | RAM [MB] |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
]
pw_list = ["—"] + [f"{p:.2f}" for p in t2["pairwise_orders"]]
for lv, pw_str in zip(t2["levels"], pw_list):
    m = lv["mesh"]
    cg = lv["cold_circuit_on_grid"]
    conv_rows.append(
        f"| **{lv['level'].capitalize()}** ($N_z/d={m['nz_per_period']}$) | "
        f"${m['nx']} \\times {m['ny']} \\times {m['nz']}$ ({m['n_cells']:,}) | "
        f"{m['cells_per_lambda_g_186ghz']:.2f} | {m['cells_across_beam_wall_gap']} | "
        f"{cg['k_3d_fit_ohm']:.4f} | {lv['alpha_pierce_grid_np_m']:.2f} | "
        f"**{lv['alpha_fit_np_m']:.2f}** | **{lv['rel_err_vs_pierce_grid']*100:.2f}%** | "
        f"{pw_str} | {lv['cold_wall_time_s']:.1f} + {lv['hot_wall_time_s']:.1f} | {lv['ram_est_mb']:.1f} |"
    )
display(Markdown("\n".join(conv_rows)))
print(f"Convergence note: {t2['asymptotic_range_note']}")

# %% [markdown]
# The animated GIF below (`figures/06_twt_growing_wave_and_beam.gif`) shows the steady-state spatial amplification of the axial electric field $\operatorname{Re}\{E_z(z, y, t)\}$ across the 22-period PEC staggered grating (top panel, with `wakis.schematic.overlay_geometry` highlighting the lower and upper glide-symmetric vanes) together with the self-consistent AC sheet-beam current density $\operatorname{Re}\{J_{1z}(z, y, t)\} = \rho_0 v_{1z} + u_0 \rho_1$ inside $|y| \le 75\;\mu\mathrm{m}$ (bottom panel):
#
# ![Traveling-wave-tube growing wave and AC sheet-beam current density](figures/06_twt_growing_wave_and_beam.gif)

# %% [markdown]
# ### 2.2 Small-Signal Gain vs Frequency at $I_0 = 100\;\mathrm{mA}$ vs Karetnikova *et al.* (2018) Figs. 3, 5a & Fig. 6
#
# We next benchmark the frequency-dependent small-signal gain at the nominal sheet-beam current $I_0 = 100\;\mathrm{mA}$ against the text statements of **Karetnikova *et al.*, *IEEE Trans. Electron Devices* **65**(6), 2129–2134 (2018)** (`.ref/sws/2018_IEEETED65_0.2THz_TWT_sheet_beam_staggered_grating_gain.txt`; all numbers below are rendered dynamically from `data/summary.json` in the cell that follows):

# %%
t3 = SUMMARY["task_03_karetnikova_gain_vs_freq"]
f5 = t3["fig5_40mm_20kv"]
sw = t3["fit_frequency_sweep"]
kb = t3["karetnikova_2018_benchmarks"]
plc = kb["paper_line_citations"]
v19 = t3["fig3_voltage_sweep"]["v_19kv"]
v20 = t3["fig3_voltage_sweep"]["v_20kv"]
v21 = t3["fig3_voltage_sweep"]["v_21kv"]

display(
    Markdown(
        f"- **What Karetnikova *et al.* (2018) report in `.ref/sws/2018_IEEETED65_0.2THz_TWT_sheet_beam_staggered_grating_gain.txt`:**\n"
        f"  - **Section III-A & Fig. 3 ($N = 50$ periods, $L = 25\\;\\mathrm{{mm}}$, 1D parametric code):** Small-signal gain of **${kb['fig3_25mm_20kv_1d_range_db'][0]}\\text{{–}}{kb['fig3_25mm_20kv_1d_range_db'][1]}\\;\\mathrm{{dB}}$** over a **${kb['fig3_25mm_20kv_bandwidth_ghz']:.0f}\\text{{-GHz}}$** band (`{plc['fig3_25mm_20kv_5_to_13db_60ghz_band']}`) at $V_0 = 20\\;\\mathrm{{kV}}$, reaching **${kb['fig3_25mm_21kv_1d_max_db']:.0f}\\;\\mathrm{{dB}}$** at $V_0 = 21\\;\\mathrm{{kV}}$ (`{plc['fig3_25mm_21kv_19db']}`).\n"
        f"  - **Section III-B & Fig. 5a ($L = 25\\;\\mathrm{{mm}}, V_0 = 20\\;\\mathrm{{kV}}, P_{{\\mathrm{{in}}}} = {kb['fig5a_25mm_20kv_pin_mw']:.0f}\\;\\mathrm{{mW}}$, `{plc['fig5a_25mm_20kv_10db']}`):** Small-signal 3D particle-in-cell gain of **$\\approx {kb['fig5a_25mm_20kv_pic_small_signal_pin_20mw_db']:.0f}\\;\\mathrm{{dB}}$** over $25\\;\\mathrm{{mm}}$.\n"
        f"  - **Section III-B ($N = 80$ periods, $L = 40\\;\\mathrm{{mm}}, V_0 = 20\\;\\mathrm{{kV}}$) & Fig. 6 at ${kb['fig6_190ghz_40mm_20kv_freq_ghz']:.0f}\\;\\mathrm{{GHz}}$ (`{plc['text_40mm_20kv_10_to_20db']}`, `{plc['fig6_190ghz_40mm_20kv_75w_at_1p01w']}`):** The text states **${kb['text_40mm_20kv_small_signal_range_db'][0]}\\text{{–}}{kb['text_40mm_20kv_small_signal_range_db'][1]}\\;\\mathrm{{dB}}$** small-signal gain at $40\\;\\mathrm{{mm}}$ (`{plc['text_40mm_20kv_10_to_20db']}`), while **Fig. 6 at ${kb['fig6_190ghz_40mm_20kv_freq_ghz']:.0f}\\;\\mathrm{{GHz}}$** (`{plc['fig6_190ghz_40mm_20kv_75w_at_1p01w']}` — not a Fig. 5b reading) shows the **saturated** 3D particle-in-cell output power **$P_{{\\mathrm{{out}}}} \\approx {kb['fig6_190ghz_40mm_20kv_pout_w']:.0f}\\;\\mathrm{{W}}$ at $P_{{\\mathrm{{in}}}} = {kb['fig6_190ghz_40mm_20kv_pin_w']:.2f}\\;\\mathrm{{W}}$** (**${kb['fig6_190ghz_40mm_20kv_saturated_gain_db']:.1f}\\;\\mathrm{{dB}}$** saturated gain).\n"
        f"- **Unexplained Pierce ($\\langle K\\rangle_{{\\mathrm{{beam}}}}$) & FIT overprediction at $25\\;\\mathrm{{mm}}$:**\n"
        f"  - **Pierce ($\\langle K\\rangle_{{\\mathrm{{beam}}}}$) overpredicts the paper by ${kb['overprediction_range_db'][1]:.2f}\\;\\mathrm{{dB}}$ at $20\\;\\mathrm{{kV}}$** (${v20['peak_gain_25mm_db']:.2f}\\;\\mathrm{{dB}}$ at ${v20['peak_freq_ghz']:.2f}\\;\\mathrm{{GHz}}$ vs ${kb['fig3_25mm_20kv_1d_range_db'][1]}\\;\\mathrm{{dB}}$, `{plc['fig3_25mm_20kv_5_to_13db_60ghz_band']}`) **and $\\ge {kb['overprediction_range_db'][0]:.1f}\\;\\mathrm{{dB}}$ (peak not bracketed by the sweep) at $21\\;\\mathrm{{kV}}$** (${v21['peak_gain_25mm_db']:.2f}\\;\\mathrm{{dB}}$ at the lower sweep boundary ${v21['peak_freq_ghz']:.1f}\\;\\mathrm{{GHz}}$ and ${v21['gain_25mm_at_cold_table_min_db']:.2f}\\;\\mathrm{{dB}}$ at the ${v21['cold_table_min_freq_ghz']:.2f}\\;\\mathrm{{GHz}}$ cold-table edge vs ${kb['fig3_25mm_21kv_1d_max_db']:.0f}\\;\\mathrm{{dB}}$, `{plc['fig3_25mm_21kv_19db']}`); **the FIT extrapolations deviate from Pierce by ${min(sw['rel_err_pct']):+.1f}\\%$ to ${max(sw['rel_err_pct']):+.1f}\\%$ in $\\alpha$ across the sweep and reach ${max(sw['extrapolated_gain_25mm_db']):.2f}\\;\\mathrm{{dB}}$ at $25\\;\\mathrm{{mm}}$**.\n"
        f"  - At $186\\;\\mathrm{{GHz}}$, our 2D FIT simulation explicitly resolves the $150\\;\\mu\\mathrm{{m}}$ sheet-beam thickness and **agrees within ${t2['levels'][-1]['rel_err_vs_pierce_grid']*100:.2f}\\%$ on the `ultrafine` grid** (${t2['levels'][-1]['alpha_fit_np_m']:.2f}$ vs ${t2['levels'][-1]['alpha_pierce_grid_np_m']:.2f}\\;\\mathrm{{Np/m}}$, **pre-asymptotic; increments ${t2_inc_md}\\;\\mathrm{{Np/m}}$**). Replacing $\\langle K_{{+1}}\\rangle_{{\\mathrm{{beam}}}}$ by the on-axis midplane impedance $K_{{+1}}(y=0)$ is therefore **not justified** as a reconciliation of the 2D FIT solver, and we state the overprediction relative to Karetnikova's text values as **UNEXPLAINED** (all runs here use lossless PEC walls with $\\alpha_{{\\mathrm{{cold}}}} = 0$; one untested hypothesis regarding Karetnikova's 1D parametric code is that evaluating a 1D model with the on-axis $K_{{+1}}(y=0)$ rather than the $\\cosh^2(\\tau_{{+1}} y)$-averaged $\\langle K_{{+1}}\\rangle_{{\\mathrm{{beam}}}}$ shifts the 1D curve downward by **${kb['hypothetical_midplane_shift_range_db'][0]:.2f}\\text{{–}}{kb['hypothetical_midplane_shift_range_db'][1]:.2f}\\;\\mathrm{{dB}}$**, from ${v20['peak_gain_25mm_db']:.2f}$ to ${v20['midplane_peak_gain_25mm_db']:.2f}\\;\\mathrm{{dB}}$ at $20\\;\\mathrm{{kV}}$ and from ${v21['peak_gain_25mm_db']:.2f}$ to ${v21['midplane_peak_gain_25mm_db']:.2f}\\;\\mathrm{{dB}}$ at $21\\;\\mathrm{{kV}}$).\n"
        f"- **All 6 frequencies of the 22-period ($11\\;\\mathrm{{mm}}$) 2D FIT sweep & how $\\alpha_{{\\mathrm{{FIT}}}}$ is de-embedded:**\n"
        f"  - At each frequency $f \\in {sw['freqs_ghz']}\\;\\mathrm{{GHz}}$, the de-embedded growth rate $\\alpha_{{\\mathrm{{FIT}}}}$ is obtained by **inverting the 3-wave Pierce boundary-value problem over periods $m \\in [8, 19]$ using Pierce's own velocity detuning $b(f)$ and space-charge parameter $QC(f)$** (`deembed_cn_from_bvp_window`), yielding $\\alpha_{{\\mathrm{{FIT}}}} = {[round(x, 2) for x in sw['alpha_fit_np_m']]}\\;\\mathrm{{Np/m}}$ vs $\\alpha_{{\\mathrm{{Pierce}}}} = {[round(x, 2) for x in sw['alpha_pierce_np_m']]}\\;\\mathrm{{Np/m}}$ (signed relative differences **${', '.join(f'{x:+.2f}%' for x in sw['rel_err_pct'])}$**, spanning ${min(sw['rel_err_pct']):+.1f}\\%$ to ${max(sw['rel_err_pct']):+.1f}\\%$). The $25\\;\\mathrm{{mm}}$ ($N=50$) and $40\\;\\mathrm{{mm}}$ ($N=80$) FIT markers in Figure `03` (`extrapolated_gain_25mm_db` reaching ${max(sw['extrapolated_gain_25mm_db']):.2f}\\;\\mathrm{{dB}}$, and `extrapolated_gain_40mm_db` reaching ${max(sw['extrapolated_gain_40mm_db']):.2f}\\;\\mathrm{{dB}}$) are **Pierce 3-wave extrapolations** evaluated using $\\alpha_{{\\mathrm{{FIT}}}}$ from the 22-period ($11\\;\\mathrm{{mm}}$) FIT model."
    )
)

display(Image(filename=str(FIG_DIR / "03_twt_gain_vs_frequency_karetnikova.png"), width=980))

freq_rows = "\n".join(
    f"    f = {f_g:5.1f} GHz (b = {b_v:+6.3f}) : alpha_FIT = {a_f:6.2f} Np/m vs alpha_Pierce = {a_p:6.2f} Np/m "
    f"({err_p:+6.2f}%), extrapolated G_25mm = {g25:5.2f} dB, G_40mm = {g40:5.2f} dB"
    for f_g, b_v, a_f, a_p, err_p, g25, g40 in zip(
        sw["freqs_ghz"],
        sw["b_pierce"],
        sw["alpha_fit_np_m"],
        sw["alpha_pierce_np_m"],
        sw["rel_err_pct"],
        sw["extrapolated_gain_25mm_db"],
        sw["extrapolated_gain_40mm_db"],
    )
)
print(
    f"Part 2.2 — Karetnikova (2018) Gain vs Frequency Benchmark (I0 = {t3['I0_mA']:.0f} mA):\n"
    f"  Fig. 3 (L = 25 mm, N = 50) 1D Pierce/ODE peaks (<K>_beam vs Karetnikova text statements):\n"
    f"    V0 = 19 kV : <K>_beam = {v19['peak_gain_25mm_db']:.2f} dB ({v19['peak_freq_ghz']:.2f} GHz) | hypothetical K(0) = {v19['midplane_peak_gain_25mm_db']:.2f} dB\n"
    f"    V0 = 20 kV : <K>_beam = {v20['peak_gain_25mm_db']:.2f} dB ({v20['peak_freq_ghz']:.2f} GHz) | hypothetical K(0) = {v20['midplane_peak_gain_25mm_db']:.2f} dB | Paper text {plc['fig3_25mm_20kv_5_to_13db_60ghz_band']}: 5-13 dB, Fig. 5a ({plc['fig5a_25mm_20kv_10db']}) particle-in-cell ({kb['fig5a_25mm_20kv_pin_mw']:.0f} mW) ~ {kb['fig5a_25mm_20kv_pic_small_signal_pin_20mw_db']:.1f} dB\n"
    f"    V0 = 21 kV : <K>_beam = {v21['peak_gain_25mm_db']:.2f} dB at lower sweep boundary {v21['peak_freq_ghz']:.2f} GHz ({v21['gain_25mm_at_cold_table_min_db']:.2f} dB at {v21['cold_table_min_freq_ghz']:.2f} GHz cold-table edge) | hypothetical K(0) = {v21['midplane_peak_gain_25mm_db']:.2f} dB | Paper text {plc['fig3_25mm_21kv_19db']} max = {kb['fig3_25mm_21kv_1d_max_db']:.1f} dB (overprediction >= {kb['overprediction_range_db'][0]:.1f} dB, peak not bracketed by the sweep)\n"
    f"    Overprediction note: {kb['overprediction_explanation']}\n"
    f"  L = 40 mm (N = 80, V0 = 20 kV) :\n"
    f"    Pierce + Wakis R_planar (<K>_beam) : {f5['wakis_planar_peak_gain_db']:.2f} dB at {f5['wakis_planar_peak_freq_ghz']:.2f} GHz\n"
    f"    Pierce + Karetnikova Eq. 4 (<K>_beam): {f5['karetnikova_eq4_peak_gain_db']:.2f} dB | hypothetical K(0) + Eq. 4 : {f5['midplane_eq4_peak_gain_db']:.2f} dB\n"
    f"    Paper text ({plc['text_40mm_20kv_10_to_20db']}) small-signal range (40 mm): {kb['text_40mm_20kv_small_signal_range_db'][0]:.0f}-{kb['text_40mm_20kv_small_signal_range_db'][1]:.0f} dB | Fig. 6 at {kb['fig6_190ghz_40mm_20kv_freq_ghz']:.0f} GHz ({plc['fig6_190ghz_40mm_20kv_75w_at_1p01w']}) SATURATED particle-in-cell (Pin={kb['fig6_190ghz_40mm_20kv_pin_w']:.2f} W -> {kb['fig6_190ghz_40mm_20kv_pout_w']:.0f} W): {kb['fig6_190ghz_40mm_20kv_saturated_gain_db']:.1f} dB\n"
    f"  All 6 points of the 22-period (11 mm) 2D FIT frequency sweep ({sw['extrapolation_note']}):\n"
    f"{freq_rows}"
)

# %% [markdown]
# ---
#
# ## Part 3 Results · Falsifiable Physics & Mutation Validation Suite (Uncalibrated)
#
# Every physical check in `task_04_validation_and_failure_tests.json` and `tests/test_076_twt_small_signal_gain.py` runs on the `coarse` node-aligned grid with **zero calibration multipliers** (`sc_scale = 1.0`, no `max(alpha, 0)` clamping). All values below are generated directly from `data/summary.json`:

# %%
t4 = SUMMARY["task_04_validation_suite"]
csm = t4["coupling_sign_mutation"]
crs = t4["current_cube_root_scaling"]
vdb = t4["velocity_detuning_b"]
scq = t4["space_charge_qc_reduction"]
rec = csm["short_10period_vs_22period_reconciliation"]
bvp_pos_str = ", ".join(f"{x:+.2f}" for x in vdb["alpha_bvp_total_field_win_np_m"][3:])
bvp_neg_str = ", ".join(f"{x:+.2f}" for x in vdb["alpha_bvp_total_field_win_np_m"][:3])
fit_neg_str = ", ".join(f"{x:+.2f}" for x in vdb["alpha_fit_np_m"][:3])

display(
    Markdown(
        f"1. **Cold limit ($I_0 \\to 0$):** Setting $I_0 = 0$ in `LinearizedBeam` yields $\\alpha_{{\\mathrm{{FIT}}}} = {t4['cold_limit_I0_zero']['alpha_fit_np_m']:.3f}\\;\\mathrm{{Np/m}}$ and $\\max_m |G(z_m)| = {t4['cold_limit_I0_zero']['max_abs_gain_db']:.3f}\\;\\mathrm{{dB}}$.\n"
        f"2. **Coupling-sign flip (`coupling_sign = -1`) — Peak moves from $+b$ to $-b$, while off-peak mirror symmetry fails on the 22-period FIT window:**\n"
        f"   - In Pierce's dispersion equation $(\\delta^2 + 4 QC)(\\mathrm{{j}}\\delta - b + \\mathrm{{j}}d) = -s$ with $s = \\pm 1$, complex conjugation maps $\\delta \\mapsto \\delta^*, b \\mapsto -b, s \\mapsto -s$, so the asymptotic single-wave root satisfies $x_1(+1.0, +1) = x_1(-1.0, -1) = {csm['cubic_x1_pos_b_pos_sign']:.6f}$ and $x_1(-1.0, +1) = x_1(+1.0, -1) = {csm['cubic_x1_neg_b_pos_sign']:.6f}$ (with $x_1(0, +1) = x_1(0, -1) = {csm['cubic_x1_b0_pos_sign']:.6f}$).\n"
        f"   - In the 22-period 2D FIT hot test ($m \\in [8, 19]$) at $|b| = 1.0$, **the growth peak moves from $+b$ to $-b$**: normal coupling (`+1`) peaks at $+b = +1.0$ ($\\alpha_{{\\mathrm{{FIT}}}}(+1.0, +1) = {csm['alpha_pos_b_pos_sign_np_m']:.2f}\\;\\mathrm{{Np/m}} > \\alpha_{{\\mathrm{{FIT}}}}(-1.0, +1) = {csm['alpha_neg_b_pos_sign_np_m']:.2f}\\;\\mathrm{{Np/m}}$), whereas flipped coupling (`-1`) peaks at $-b = -1.0$ ($\\alpha_{{\\mathrm{{FIT}}}}(-1.0, -1) = {csm['alpha_neg_b_neg_sign_np_m']:.2f}\\;\\mathrm{{Np/m}} > \\alpha_{{\\mathrm{{FIT}}}}(+1.0, -1) = {csm['alpha_pos_b_neg_sign_np_m']:.2f}\\;\\mathrm{{Np/m}}$), with the **peak pair** agreeing within **${csm['mirror_peak_rel_diff']*100:.2f}\\%$** (${csm['alpha_pos_b_pos_sign_np_m']:.2f}$ vs ${csm['alpha_neg_b_neg_sign_np_m']:.2f}\\;\\mathrm{{Np/m}}$).\n"
        f"   - However, **mirror symmetry fails for the off-peak pair**: $\\alpha_{{\\mathrm{{FIT}}}}(-1.0, +1) = {csm['alpha_neg_b_pos_sign_np_m']:.2f}\\;\\mathrm{{Np/m}}$ vs $\\alpha_{{\\mathrm{{FIT}}}}(+1.0, -1) = {csm['alpha_pos_b_neg_sign_np_m']:.2f}\\;\\mathrm{{Np/m}}$ (**${csm['mirror_offpeak_rel_diff']*100:.1f}\\%$** relative difference), even though the Pierce cubic predicts $x_1 = {csm['cubic_x1_neg_b_pos_sign']:.6f}$ (${csm['alpha_pierce_neg_b_pos_sign_np_m']:.2f}\\;\\mathrm{{Np/m}}$) for both. This asymmetry is not predicted by the 3-wave Pierce BVP, which is exactly mirror-symmetric at all $z$; it is unexplained. (The live 10-period test at ${rec['live_10period_current_a']:.2f}\\;\\mathrm{{A}}$, $m \\in {rec['live_10period_window_periods']}$, is a different current and window.)\n"
        f"3. **Current scaling ($I_0 \\in {crs['i0_values_a']}\\;\\mathrm{{A}}$ at $b = 0$) — Raw window slope vs BVP de-embedding:**\n"
        f"   - **Raw FIT window slope does NOT follow $I_0^{{1/3}}$:** On the fixed 22-period grating ($m \\in [8, 19]$), the raw log-amplitude window slope $\\alpha_{{\\mathrm{{raw}}}} = {[round(x, 2) for x in crs['alpha_raw_window_fit_np_m']]}\\;\\mathrm{{Np/m}}$ has a log-log exponent of **${crs['raw_window_loglog_slope']:.4f}$** (vs $1/3 = {crs['loglog_slope_theory']:.4f}$) because at $I_0 = {crs['i0_values_a'][0]:.2f}\\;\\mathrm{{A}}$ ($CN \\approx {crs['cn_window_at_min_i0']:.2f}$) the window sits inside the three-wave launching interference region.\n"
        f"   - **BVP de-embedded $\\alpha_{{\\mathrm{{FIT}}}}^{{(QC=0)}}$:** Inverting the Pierce BVP transfer function at period $m = 19$ (`_deembed_cn_from_gain_at_m19`) using Pierce's own $b = 0$ and $QC(I_0)$ yields $\\alpha_{{\\mathrm{{FIT}}}}^{{(QC=0)}} = {[round(x, 2) for x in crs['alpha_fit_np_m']]}\\;\\mathrm{{Np/m}}$ vs $\\alpha_{{\\mathrm{{Pierce}}}}^{{(QC=0)}} = {[round(x, 2) for x in crs['alpha_pierce_qc0_np_m']]}\\;\\mathrm{{Np/m}}$. While the de-embedded log-log slope is **${crs['loglog_slope_fit']:.4f}$** (**${crs['slope_rel_err']*100:.2f}\\%$** from $1/3$; with space charge included, ${crs['loglog_slope_sc_fit']:.4f}$ vs Pierce ${crs['loglog_slope_sc_theory']:.4f}$), the de-embedded $\\alpha_{{\\mathrm{{FIT}}}}^{{(QC=0)}}$ sits **${min(crs['deemb_vs_pierce_overprediction_pct']):.1f}\\%\\text{{–}}{max(crs['deemb_vs_pierce_overprediction_pct']):.1f}\\%$ (mean ${crs['mean_deemb_vs_pierce_overprediction_pct']:.1f}\\%$) above Pierce at every current**.\n"
        f"4. **Velocity detuning $b$ sweep (withdrawn 1D total-field BVP explanation):**\n"
        f"   - Across $b \\in {vdb['b_values']}$, the raw 2D FIT window slope is $\\alpha_{{\\mathrm{{FIT}}}} = {[round(x, 2) for x in vdb['alpha_fit_np_m']]}\\;\\mathrm{{Np/m}}$ vs the single-wave Pierce rate $\\alpha_{{\\mathrm{{Pierce}}}} = {[round(x, 2) for x in vdb['alpha_pierce_np_m']]}\\;\\mathrm{{Np/m}}$.\n"
        f"   - Relative difference vs single-wave Pierce is **${vdb['rel_err_at_b0']*100:.2f}\\%$ at $b = 0$; ${vdb['rel_err_at_bm0p5']*100:.2f}\\%$ at $b = -0.5$ (not agreement)**; at $b = -1.0$ the difference is **${vdb['rel_err_at_bm1p0']*100:.2f}\\%$** (${vdb['alpha_fit_np_m'][0]:.2f}$ vs ${vdb['alpha_pierce_np_m'][0]:.2f}\\;\\mathrm{{Np/m}}$), and for $b \\in \\{{+0.7, +1.4, +2.2\\}}$ the 22-period FIT slope remains positive (${vdb['alpha_fit_np_m'][3]:.2f}, {vdb['alpha_fit_np_m'][4]:.2f}, {vdb['alpha_fit_np_m'][5]:.2f}\\;\\mathrm{{Np/m}}$) even beyond the single-wave cutoff $b_{{\\max}}$ where $\\alpha_{{\\mathrm{{Pierce}}}}(+2.2) = 0$.\n"
        f"   - **Withdrawn 1D total-field BVP / crestatron explanation:** Fitting the 1D Pierce total field $|E_{{\\mathrm{{ckt}}}} + E_{{\\mathrm{{sc}}}}|$ over the same window $m \\in [8, 19]$ gives $\\alpha_{{\\mathrm{{BVP,tot}}}} = {[round(x, 2) for x in vdb['alpha_bvp_total_field_win_np_m']]}\\;\\mathrm{{Np/m}}$. While this curve happens to lie near FIT at $b \\ge +0.7$ (${bvp_pos_str}\\;\\mathrm{{Np/m}}$), it **fails completely for $b \\le 0$**, predicting **${bvp_neg_str}\\;\\mathrm{{Np/m}}$** at $b = -1.0, -0.5, 0.0$ vs FIT **${fit_neg_str}\\;\\mathrm{{Np/m}}$**. We therefore **withdraw** the 1D total-field BVP explanation.\n"
        f"5. **Space-charge parameter $QC$ sweep at fixed $b = 0$:** Across $QC \\in {scq['QC_values']}$, the 2D FIT slope decreases monotonically from **${' \\to '.join(f'{x:.2f}' for x in scq['alpha_fit_np_m'])}\\;\\mathrm{{Np/m}}$** vs single-wave Pierce **${[round(x, 2) for x in scq['alpha_pierce_np_m']]}\\;\\mathrm{{Np/m}}$** (${scq['disagreement_explanation']}).\n"
        f"6. **Solver linearity check (holds by construction):** Because `LinearizedBeam` and `SolverFIT3D` are linear operators, scaling the drive amplitude over $A_{{\\mathrm{{drive}}}} \\in {t4['drive_linearity']['drive_amplitudes']}$ leaves $\\alpha_{{\\mathrm{{FIT}}}} = {t4['drive_linearity']['alpha_fit_np_m'][0]:.6f}\\;\\mathrm{{Np/m}}$ invariant to machine precision (`max_rel_diff = {t4['drive_linearity']['max_rel_diff']:.2e}`). This is an algebraic implementation sanity check, not a physical validation."
    )
)

display(Image(filename=str(FIG_DIR / "04_twt_scaling_and_validation_tests.png"), width=980))

print(
    f"Part 3 — Uncalibrated Falsifiable Physics & Mutation Suite Summary:\n"
    f"  1. Cold limit (I0 = 0)             : alpha_FIT = {t4['cold_limit_I0_zero']['alpha_fit_np_m']:.4f} Np/m, max |G| = {t4['cold_limit_I0_zero']['max_abs_gain_db']:.4f} dB\n"
    f"  2. Sign flip at |b|=1.0            : peak pair (+1,+1)={csm['alpha_pos_b_pos_sign_np_m']:.2f} vs (-1,-1)={csm['alpha_neg_b_neg_sign_np_m']:.2f} Np/m ({csm['mirror_peak_rel_diff']*100:.2f}% diff)\n"
    f"     Off-peak pair fails mirror sym  : (-1,+1)={csm['alpha_neg_b_pos_sign_np_m']:.2f} vs (+1,-1)={csm['alpha_pos_b_neg_sign_np_m']:.2f} Np/m ({csm['mirror_offpeak_rel_diff']*100:.1f}% diff; unexplained by mirror-symmetric 3-wave Pierce BVP; live 10-period {rec['live_10period_current_a']:.2f}A m={rec['live_10period_window_periods']} is a different current and window)\n"
    f"  3. Current scaling (I0 in {crs['i0_values_a']} A): raw window slope = {crs['raw_window_loglog_slope']:.4f} (not I0^(1/3)); BVP-de-embedded slope = {crs['loglog_slope_fit']:.5f} vs 1/3 ({crs['slope_rel_err']*100:.2f}% err), but de-embedded alpha is {crs['deemb_vs_pierce_overprediction_pct']}% (mean {crs['mean_deemb_vs_pierce_overprediction_pct']:.1f}%) above Pierce\n"
    f"  4. Velocity detuning b sweep       : {vdb['rel_err_at_b0']*100:.2f}% at b = 0; {vdb['rel_err_at_bm0p5']*100:.2f}% at b = -0.5 (not agreement) | b={vdb['b_values']} -> alpha_FIT={vdb['alpha_fit_np_m']} vs Pierce={vdb['alpha_pierce_np_m']} | withdrawn 1D total-field BVP={vdb['alpha_bvp_total_field_win_np_m']}\n"
    f"  5. Space-charge QC sweep (at b=0)  : QC={scq['QC_values']} -> alpha_FIT={scq['alpha_fit_np_m']} Np/m vs single-wave Pierce={scq['alpha_pierce_np_m']} Np/m\n"
    f"  6. Solver linearity (by construction): max relative difference = {t4['drive_linearity']['max_rel_diff']:.3e}"
)

# %% [markdown]
# ---
#
# ## Part 4 & Part 5 Results · 3D Glide-Symmetric V-Band SWS (Marosi 2025) & Backward-Wave Oscillation (Johnson 1955)
#
# All dimensions, cell counts across the $100\;\mu\mathrm{m}$ sheet beam, and benchmark numbers below are rendered directly from `data/summary.json`:

# %%
t5 = SUMMARY["task_05_3d_marosi_and_bwo"]
m3d = t5["marosi_3d_vband"]
bwo = t5["bwo_johnson_1955"]
lv_c3d = m3d["hot_test_3d_levels"][0]
lv_m3d = m3d["hot_test_3d_levels"][1]
fb3d = m3d["full_60period_batch_estimate"]

display(
    Markdown(
        f"### 4.1 3D Glide-Symmetric V-Band Pillar SWS (Marosi *et al.*, 2025)\n\n"
        f"Reusing the 3D cold-test dispersion and Pierce interaction impedance from Tutorial 6.2 (`task_10_3d_marosi_glide.npz`, where $K_{{+1,\\mathrm{{beam}}}}(58\\;\\mathrm{{GHz}}) = {m3d['K_p1_beam_58ghz_ohm']:.3f}\\;\\Omega$ and $v_E/c = {m3d['v_energy_over_c_58ghz']:.4f}$ across the **${m3d['beam_width_mm']:.2f}\\;\\mathrm{{mm}}\\times {m3d['beam_thickness_mm']:.2f}\\;\\mathrm{{mm}}$ (${m3d['beam_thickness_um']:.0f}\\;\\mu\\mathrm{{m}}$ thick)** sheet beam inside the **$a = {m3d['waveguide_width_a_mm']:.2f}\\;\\mathrm{{mm}}, b = {m3d['waveguide_height_b_mm']:.2f}\\;\\mathrm{{mm}}$** glide-symmetric pillar waveguide of period **$d = {m3d['period_d_mm']:.2f}\\;\\mathrm{{mm}}$**, per `{m3d['paper_ref_file']}` Table I `{m3d['paper_table1_lines']}`; beam dimensions `{m3d['paper_beam_dimensions_line']}`), Figure `05_3d_marosi_vband_and_bwo.png` (a, b) shows:\n"
        f"- **Panel (a) — 60-period V-band small-signal gain vs Marosi *et al.* (2025) Fig. 7 & cold insertion-loss derivation:**\n"
        f"  - Marosi *et al.* (2025) report a **cold insertion loss of ${m3d['paper_cold_insertion_loss_db']:.1f}\\;\\mathrm{{dB}}$** across the ${m3d['n_periods_full']}$-period ($L = {m3d['L_full_mm']:.1f}\\;\\mathrm{{mm}}$) structure, which determines the cold field attenuation constant $\\alpha_0 = {m3d['paper_cold_insertion_loss_db']:.1f} / (20\\log_{{10}}\\mathrm{{e}}\\cdot {m3d['L_full_mm']*1e-3:.3f}) = {m3d['derived_alpha_loss_np_m']:.3f}\\;\\mathrm{{Np/m}}$ and the frequency-dependent Pierce loss parameter $d(\\omega) = \\alpha_0 / (\\beta_e C)$ (**$d = {m3d['derived_pierce_d_at_58ghz']:.4f}$** at $58\\;\\mathrm{{GHz}}$).\n"
        f"  - At the paper's nominal operating point **$V_0 = {m3d['paper_operating_voltage_kv']:.2f}\\;\\mathrm{{kV}}, I_0 = {m3d['paper_operating_current_ma']:.0f}\\;\\mathrm{{mA}}$**, our 1D Pierce BVP and coupled-mode ODE agree identically and predict **${m3d['gain_60per_5p20kv_at_58ghz_db']:.2f}\\;\\mathrm{{dB}}$** at $58.0\\;\\mathrm{{GHz}}$ (compared with the ${m3d['marosi_2025_fig7_peak_gain_db']:.1f}\\;\\mathrm{{dB}}$ hot-simulation peak at $58\\;\\mathrm{{GHz}}$ in Marosi Fig. 7) and a secondary upper-branch peak of **${m3d['gain_60per_5p20kv_peak_db']:.2f}\\;\\mathrm{{dB}}$ at ${m3d['gain_60per_5p20kv_peak_freq_ghz']:.1f}\\;\\mathrm{{GHz}}$** (corresponding to the $68\\;\\mathrm{{GHz}}$ secondary peak in Marosi Fig. 7). Evaluating the same curve at $V_0 = 5.38\\;\\mathrm{{kV}}$ yields ${m3d['gain_60per_5p38kv_at_58ghz_db']:.2f}\\;\\mathrm{{dB}}$ at $58.0\\;\\mathrm{{GHz}}$ (${m3d['gain_60per_5p38kv_peak_db']:.2f}\\;\\mathrm{{dB}}$ peak at ${m3d['gain_60per_5p38kv_peak_freq_ghz']:.1f}\\;\\mathrm{{GHz}}$).\n"
        f"- **Panel (b) — 10-period 3D FIT spatial hot test (both `coarse_3d` and `medium_3d`) & mesh cell count across the ${m3d['beam_thickness_um']:.0f}\\;\\mu\\mathrm{{m}}$ beam:**\n"
        f"  - Evaluating the 10-period 3D FIT hot test at $f_0 = 57.5\\;\\mathrm{{GHz}}$, $I_0 = 30\\;\\mathrm{{mA}}$, and the **actual simulated detuning $b = {lv_c3d['b_simulated']:.1f}$**, `coarse_3d` (`{lv_c3d['nx']} x {lv_c3d['ny']} x {lv_c3d['nz_total']} = {lv_c3d['n_cells']:,}` cells, $\\Delta y = {lv_c3d['dy_um']:.1f}\\;\\mu\\mathrm{{m}}$, resolving the ${m3d['beam_thickness_um']:.0f}\\;\\mu\\mathrm{{m}}$ beam thickness with only **${lv_c3d['cells_across_100um_beam_thickness']:.1f}$ cells**) gives $\\alpha_{{\\mathrm{{FIT}}}} = {lv_c3d['alpha_fit_np_m']:.2f}\\;\\mathrm{{Np/m}}$ vs $\\alpha_{{\\mathrm{{Pierce}}}}(b=0) = {lv_c3d['alpha_pierce_np_m']:.2f}\\;\\mathrm{{Np/m}}$ (**${lv_c3d['rel_err_vs_pierce']*100:.2f}\\%$** relative error), and `medium_3d` (`{lv_m3d['nx']} x {lv_m3d['ny']} x {lv_m3d['nz_total']} = {lv_m3d['n_cells']:,}` cells, $\\Delta y = {lv_m3d['dy_um']:.1f}\\;\\mu\\mathrm{{m}}$, resolving the ${m3d['beam_thickness_um']:.0f}\\;\\mu\\mathrm{{m}}$ beam thickness with only **${lv_m3d['cells_across_100um_beam_thickness']:.1f}$ cells**) gives $\\alpha_{{\\mathrm{{FIT}}}} = {lv_m3d['alpha_fit_np_m']:.2f}\\;\\mathrm{{Np/m}}$ vs $\\alpha_{{\\mathrm{{Pierce}}}}(b=0) = {lv_m3d['alpha_pierce_np_m']:.2f}\\;\\mathrm{{Np/m}}$ (**${lv_m3d['rel_err_vs_pierce']*100:.2f}\\%$** relative error).\n"
        f"  - We state plainly that this 2-level compact 3D mesh (`{lv_c3d['nz_per_period']}–{lv_m3d['nz_per_period']}` cells per period and only **${lv_c3d['cells_across_100um_beam_thickness']:.1f}\\text{{–}}{lv_m3d['cells_across_100um_beam_thickness']:.1f}$ cells across the ${m3d['beam_thickness_um']:.0f}\\;\\mu\\mathrm{{m}}$ beam thickness**, below the 4-cell minimum rule) is **not converged**. Reaching the asymptotic 3D regime across the full 60-period structure (`{fb3d['grid_shape'][0]} x {fb3d['grid_shape'][1]} x {fb3d['grid_shape'][2]} = {fb3d['n_cells']/1e6:.2f} Mcells`, $\\Delta y = {fb3d['dy_um']:.1f}\\;\\mu\\mathrm{{m}}$, **${fb3d['cells_across_100um_beam_thickness']:.1f}$ cells across the ${m3d['beam_thickness_um']:.0f}\\;\\mu\\mathrm{{m}}$ beam thickness**, ${fb3d['n_steps']:,}$ steps) requires **${fb3d['ram_working_gb']:.2f}\\;\\mathrm{{GB}}$ RAM and $\\sim {fb3d['cpu_hours_est']:.1f}\\;\\mathrm{{CPU\\text{{-}}hours}}$** (`wakis.estimate`); the CLI `{fb3d['batch_command']}` is **{fb3d['note']}**.\n\n"
        f"### 5.1 Backward-Wave Oscillation (BWO) Start-Oscillation Condition (Johnson 1955)\n\n"
        f"In Panel (c) of Figure `05_3d_marosi_vband_and_bwo.png`, we analyze the **backward ($n = -1$, $v_g < 0$) space harmonic** of the corrugated waveguide at $\\beta_0 d = 0.5\\pi$ ($\\beta_{{-1}} d = -1.5\\pi$, $f_{{\\mathrm{{cold}}}} = {bwo['f_cold_bwo_ghz']:.3f}\\;\\mathrm{{GHz}}$, $K_{{-1,\\mathrm{{beam}}}} = {bwo['K_m1_beam_ohm']:.2f}\\;\\Omega$, $V_{{\\mathrm{{sync}}}} = {bwo['v_sync_bwo_kv']:.3f}\\;\\mathrm{{kV}}$ from Tutorial 6.2):\n"
        f"- **Exact Johnson (1955) boundary-value root & $I_{{\\mathrm{{st}}}}(L) \\propto L^{{-3}}$ identity:** Solving the complex two-variable secular equation $\\Delta_{{\\mathrm{{BWO}}}}(CN, b) = 0$ (`solve_bwo_start_oscillation`) yields **$(CN)_{{\\mathrm{{st}}}} = {bwo['CN_start_qc0']:.6f}$** and **$b_{{\\mathrm{{st}}}} = {bwo['b_start_qc0']:.6f}$** at $QC = d = 0$, matching **Johnson (1955) Table I** (${bwo['johnson_1955_table1_CN']:.3f}$ and ${bwo['johnson_1955_table1_b']:.3f}$) to **${bwo['CN_rel_err_vs_johnson']*100:.3f}\\%$**, and root-finding $T_{{\\mathrm{{BWO}}}}(I_0, b; L) = 0$ across $N \\in [15, 60]$ periods gives $I_{{\\mathrm{{st}}}}(50d) = {bwo['ist_50periods_ma']:.2f}\\;\\mathrm{{mA}}$ and the exact Pierce algebraic identity $I_{{\\mathrm{{st}}}}(L) \\propto L^{{-3}}$ (log-log slope **${bwo['ist_vs_length_loglog_slope']:.6f}$**).\n"
        f"- **Withdrawn 1-cell Bloch start-current claim (not plotted):** {bwo['single_cell_bloch_note']}"
    )
)

display(Image(filename=str(FIG_DIR / "05_3d_marosi_vband_and_bwo.png"), width=980))

print(
    f"Part 4 — 3D Glide-Symmetric V-Band SWS (Marosi 2025, {m3d['paper_ref_file']} Table I {m3d['paper_table1_lines']}, beam {m3d['paper_beam_dimensions_line']}: a={m3d['waveguide_width_a_mm']:.2f} mm, b={m3d['waveguide_height_b_mm']:.2f} mm, beam={m3d['beam_width_mm']:.2f}x{m3d['beam_thickness_mm']:.2f} mm [{m3d['beam_thickness_um']:.0f} um]):\n"
    f"  Cold insertion loss (60 cells)      : {m3d['paper_cold_insertion_loss_db']:.1f} dB -> alpha_0 = {m3d['derived_alpha_loss_np_m']:.3f} Np/m, d(58 GHz) = {m3d['derived_pierce_d_at_58ghz']:.4f}\n"
    f"  60-period Pierce/ODE (5.20 kV)      : {m3d['gain_60per_5p20kv_at_58ghz_db']:.2f} dB at 58.0 GHz, peak {m3d['gain_60per_5p20kv_peak_db']:.2f} dB at {m3d['gain_60per_5p20kv_peak_freq_ghz']:.1f} GHz (Marosi Fig. 7: ~{m3d['marosi_2025_fig7_peak_gain_db']:.1f} dB at 58 GHz, secondary peak at 68 GHz)\n"
    f"  60-period Pierce/ODE (5.38 kV)      : {m3d['gain_60per_5p38kv_at_58ghz_db']:.2f} dB at 58.0 GHz, peak {m3d['gain_60per_5p38kv_peak_db']:.2f} dB at {m3d['gain_60per_5p38kv_peak_freq_ghz']:.1f} GHz\n"
    f"  10-period 3D FIT hot test (b = 0)   :\n"
    f"    coarse_3d ({lv_c3d['nx']}x{lv_c3d['ny']}x{lv_c3d['nz_total']}, {lv_c3d['cells_across_100um_beam_thickness']:.1f} cells/100um): alpha_FIT = {lv_c3d['alpha_fit_np_m']:.2f} Np/m vs Pierce(b=0) {lv_c3d['alpha_pierce_np_m']:.2f} Np/m ({lv_c3d['rel_err_vs_pierce']*100:.2f}% err)\n"
    f"    medium_3d ({lv_m3d['nx']}x{lv_m3d['ny']}x{lv_m3d['nz_total']}, {lv_m3d['cells_across_100um_beam_thickness']:.1f} cells/100um): alpha_FIT = {lv_m3d['alpha_fit_np_m']:.2f} Np/m vs Pierce(b=0) {lv_m3d['alpha_pierce_np_m']:.2f} Np/m ({lv_m3d['rel_err_vs_pierce']*100:.2f}% err) -> converged={m3d['hot_test_3d_converged']}\n"
    f"  Full 60-period 3D preflight         : {fb3d['grid_shape']} = {fb3d['n_cells']:,} cells ({fb3d['cells_across_100um_beam_thickness']:.1f} cells/100um), {fb3d['ram_working_gb']:.2f} GB RAM, {fb3d['cpu_hours_est']:.2f} CPU-h ({fb3d['note']})\n"
    f"  Preflight CLI command               : {fb3d['batch_command']}\n\n"
    f"Part 5 — Backward-Wave Oscillation (Johnson 1955, f_BWO = {bwo['f_cold_bwo_ghz']:.3f} GHz, K_-1 = {bwo['K_m1_beam_ohm']:.2f} Ohm):\n"
    f"  Space-charge-free (CN)_st           : {bwo['CN_start_qc0']:.6f} vs Johnson Table I {bwo['johnson_1955_table1_CN']:.3f} ({bwo['CN_rel_err_vs_johnson']*100:.3f}% diff)\n"
    f"  Space-charge-free b_st              : {bwo['b_start_qc0']:.6f} vs Johnson Table I {bwo['johnson_1955_table1_b']:.3f}\n"
    f"  Start-current length scaling slope  : d(log I_st)/d(log L) = {bwo['ist_vs_length_loglog_slope']:.6f} (I_st(50d) = {bwo['ist_50periods_ma']:.2f} mA)\n"
    f"  1-cell Bloch note                   : {bwo['single_cell_bloch_note']}"
)

# %% [markdown]
# ---
#
# ## Quantitative Summary Table
#
# Every value in the table below is dynamically generated from `data/summary.json` and verified by `tests/test_076_twt_small_signal_gain.py`:

# %%
t2_errs = " \\to ".join(f"{lv['rel_err_vs_pierce_grid']*100:.2f}\\%" for lv in t2["levels"])
t2_pw = ", ".join(f"{p:.2f}" for p in t2["pairwise_orders"])
t2_alphas = ", ".join(f"{lv['alpha_fit_np_m']:.2f}" for lv in t2["levels"])
sw_p_str = ", ".join(f"{x:.1f}" for x in sw["alpha_pierce_np_m"])
sw_f_str = ", ".join(f"{x:.1f}" for x in sw["alpha_fit_np_m"])
sw_e_str = ", ".join(f"{x:+.1f}\\%" for x in sw["rel_err_pct"])

display(
    Markdown(
        f"| Benchmark / Physical Quantity | Theory / Literature Target | Wakis Result (`data/summary.json`) | Relative Error / Status |\n"
        f"| :--- | :--- | :--- | :--- |\n"
        f"| **Pierce launching loss $A$** ($b=QC=d=0$) | $-20\\log_{{10}} 3 = {t1['A_launching_loss_db']:.6f}\\;\\mathrm{{dB}}$ | `A_launching_loss_db = {t1['A_launching_loss_db']:.6f}` dB | **$0.00\\%$** (exact) |\n"
        f"| **Pierce asymptotic slope $B$** ($b=QC=d=0$) | $20\\pi\\sqrt{{3}}\\log_{{10}}\\mathrm{{e}} = {t1['B_pierce_slope_db']:.6f}\\;\\mathrm{{dB}}$ | `B_pierce_slope_db = {t1['B_pierce_slope_db']:.6f}` dB | **$0.00\\%$** (exact) |\n"
        f"| **3-wave BVP vs 1D coupled-mode ODE** | Identical spatial transfer function | `max_bvp_vs_ode_diff_db = {t1['max_bvp_vs_ode_diff_db']:.2e}` dB | **$< 2\\times 10^{{-9}}\\;\\mathrm{{dB}}$** |\n"
        f"| **Two-plate reduction factor ($186\\;\\mathrm{{GHz}}, 20\\;\\mathrm{{kV}}$)** | Karetnikova Eq. (4) $D = {scw['D_karetnikova_eq4']:.6f}$ ($\\gamma h={scw['gamma_h_186ghz']:.4f}, \\gamma a={scw['gamma_a_186ghz']:.4f}$) | `R_planar_wakis = {scw['R_planar_wakis']:.6f}`, `D_karetnikova_eq4 = {scw['D_karetnikova_eq4']:.6f}` | **${abs(scw['R_planar_wakis']-scw['D_karetnikova_eq4'])/scw['D_karetnikova_eq4']*100:.2f}\\%$** eigenmode vs Green's avg |\n"
        f"| **Smooth-tunnel `LinearizedBeam` SCW $\\omega_q$** ($N_z/\\lambda_e = 40$) | $\\omega_q = R_{{\\mathrm{{planar}}}}\\omega_p/\\gamma_0^{{3/2}} = {scw['omega_q_theory_rad_s']:.4e}\\;\\mathrm{{rad/s}}$ | `wq_fit_rad_s = {scw['levels'][-1]['wq_fit_rad_s']:.4e}` rad/s | **${scw['levels'][-1]['rel_err_wq']*100:.2f}\\%$** (`rel_err_wq = {scw['levels'][-1]['rel_err_wq']:.4f}`) |\n"
        f"| **2D Karetnikova hot-test $\\alpha$ (`coarse` $\\to$ `ultrafine`)** | Same-grid $\\alpha_{{\\mathrm{{Pierce,grid}}}} = {t2['levels'][0]['alpha_pierce_grid_np_m']:.2f} \\to {t2['levels'][-1]['alpha_pierce_grid_np_m']:.2f}\\;\\mathrm{{Np/m}}$ | `alpha_fit_np_m = {t2_alphas}` Np/m | **${t2_errs}$** (pairwise $p = {t2_pw}$; increments ${t2_inc_md}\\;\\mathrm{{Np/m}}$, pre-asymptotic) |\n"
        f"| **Karetnikova (2018) $25\\;\\mathrm{{mm}}$ gain** ($20\\;\\mathrm{{kV}}$ / $21\\;\\mathrm{{kV}}$) | Text `{plc['fig3_25mm_20kv_5_to_13db_60ghz_band']}`: ${kb['fig3_25mm_20kv_1d_range_db'][0]}\\text{{–}}{kb['fig3_25mm_20kv_1d_range_db'][1]}\\;\\mathrm{{dB}}$ ($20\\;\\mathrm{{kV}}$), `{plc['fig3_25mm_21kv_19db']}`: ${kb['fig3_25mm_21kv_1d_max_db']:.0f}\\;\\mathrm{{dB}}$ ($21\\;\\mathrm{{kV}}$); Fig. 5a (`{plc['fig5a_25mm_20kv_10db']}`) particle-in-cell $\\approx {kb['fig5a_25mm_20kv_pic_small_signal_pin_20mw_db']:.0f}\\;\\mathrm{{dB}}$ | $\\langle K\\rangle_{{\\mathrm{{beam}}}}$: `{v20['peak_gain_25mm_db']:.2f}` / `{v21['peak_gain_25mm_db']:.2f}` dB (${v21['peak_freq_ghz']:.1f}\\;\\mathrm{{GHz}}$ lower edge); FIT extrap. up to `{max(sw['extrapolated_gain_25mm_db']):.2f}` dB | **Pierce overpredicts by ${kb['overprediction_range_db'][1]:.2f}\\;\\mathrm{{dB}}$ ($20\\;\\mathrm{{kV}}$) & $\\ge {kb['overprediction_range_db'][0]:.1f}\\;\\mathrm{{dB}}$ ($21\\;\\mathrm{{kV}}$, peak not bracketed); UNEXPLAINED** |\n"
        f"| **Karetnikova (2018) $40\\;\\mathrm{{mm}}$ gain** ($20\\;\\mathrm{{kV}}, 100\\;\\mathrm{{mA}}$) | Text `{plc['text_40mm_20kv_10_to_20db']}`: ${kb['text_40mm_20kv_small_signal_range_db'][0]}\\text{{–}}{kb['text_40mm_20kv_small_signal_range_db'][1]}\\;\\mathrm{{dB}}$ small-signal; Fig. 6 at ${kb['fig6_190ghz_40mm_20kv_freq_ghz']:.0f}\\;\\mathrm{{GHz}}$ (`{plc['fig6_190ghz_40mm_20kv_75w_at_1p01w']}`): ${kb['fig6_190ghz_40mm_20kv_saturated_gain_db']:.1f}\\;\\mathrm{{dB}}$ saturated (${kb['fig6_190ghz_40mm_20kv_pin_w']:.2f}\\;\\mathrm{{W}}\\to {kb['fig6_190ghz_40mm_20kv_pout_w']:.0f}\\;\\mathrm{{W}}$) | Lossless $\\langle K\\rangle_{{\\mathrm{{beam}}}}$: `{f5['karetnikova_eq4_peak_gain_db']:.2f}` dB (hypothetical $K(0)$: `{f5['midplane_eq4_peak_gain_db']:.2f}` dB) | Lossless PEC model overpredicts paper text |\n"
        f"| **2D FIT 6-frequency sweep** ($183.0\\text{{–}}196.0\\;\\mathrm{{GHz}}$, BVP de-embedded) | $\\alpha_{{\\mathrm{{Pierce}}}} = [{sw_p_str}]\\;\\mathrm{{Np/m}}$ | $\\alpha_{{\\mathrm{{FIT}}}} = [{sw_f_str}]\\;\\mathrm{{Np/m}}$ | **${sw_e_str}$** |\n"
        f"| **Sign-flip test (`coupling_sign = +1` vs `-1`)** | Cubic $x_1(+1,+1)=x_1(-1,-1)={csm['cubic_x1_pos_b_pos_sign']:.4f}$; $x_1(-1,+1)=x_1(+1,-1)={csm['cubic_x1_neg_b_pos_sign']:.4f}$ | Peak pair: ${csm['alpha_pos_b_pos_sign_np_m']:.2f}$ vs ${csm['alpha_neg_b_neg_sign_np_m']:.2f}\\;\\mathrm{{Np/m}}$; off-peak: ${csm['alpha_neg_b_pos_sign_np_m']:.2f}$ vs ${csm['alpha_pos_b_neg_sign_np_m']:.2f}\\;\\mathrm{{Np/m}}$ | Peak moves $+b \\mapsto -b$ (**${csm['mirror_peak_rel_diff']*100:.2f}\\%$**); off-peak mirror fails (**${csm['mirror_offpeak_rel_diff']*100:.1f}\\%$**, unexplained) |\n"
        f"| **Current scaling ($I_0 \\in {crs['i0_values_a'][0]:.2f}\\text{{–}}{crs['i0_values_a'][-1]:.2f}\\;\\mathrm{{A}}$ at $b=0$)** | Exponent $1/3 = {crs['loglog_slope_theory']:.5f}$ | Raw slope `{crs['raw_window_loglog_slope']:.4f}`; BVP-de-embedded slope `{crs['loglog_slope_fit']:.5f}` | Raw slope $\\neq I_0^{{1/3}}$; de-embedded slope **${crs['slope_rel_err']*100:.2f}\\%$** err but $\\alpha$ is **mean ${crs['mean_deemb_vs_pierce_overprediction_pct']:.1f}\\%$ above Pierce** |\n"
        f"| **3D Marosi V-band 60-period & 10-period FIT** | Fig. 7: ${m3d['marosi_2025_fig7_peak_gain_db']:.0f}\\;\\mathrm{{dB}}$ at $58\\;\\mathrm{{GHz}}$ ($5.2\\;\\mathrm{{kV}}, 11\\;\\mathrm{{mA}}$, $4.7\\;\\mathrm{{dB}}$ cold IL; $100\\;\\mu\\mathrm{{m}}$ beam) | 1D: `{m3d['gain_60per_5p20kv_at_58ghz_db']:.2f}` dB ($58\\;\\mathrm{{GHz}}$), `{m3d['gain_60per_5p20kv_peak_db']:.2f}` dB ($69.1\\;\\mathrm{{GHz}}$); 3D FIT: ${lv_c3d['rel_err_vs_pierce']*100:.1f}\\%, {lv_m3d['rel_err_vs_pierce']*100:.1f}\\%$ err | Compact 3D grids (${lv_c3d['cells_across_100um_beam_thickness']:.1f}\\text{{–}}{lv_m3d['cells_across_100um_beam_thickness']:.1f}$ cells/$100\\;\\mu\\mathrm{{m}}$) **not converged** |\n"
        f"| **BWO start-oscillation $(CN)_{{\\mathrm{{st}}}}$ & $I_{{\\mathrm{{st}}}} \\propto L^{{-3}}$** | Johnson (1955) Table I: $(CN)_{{\\mathrm{{st}}}} = {bwo['johnson_1955_table1_CN']:.3f}$, $b_{{\\mathrm{{st}}}} = {bwo['johnson_1955_table1_b']:.3f}$, $p = -3$ | `CN_start_qc0 = {bwo['CN_start_qc0']:.6f}`, `b_start_qc0 = {bwo['b_start_qc0']:.6f}`, `slope = {bwo['ist_vs_length_loglog_slope']:.1f}` | **${bwo['CN_rel_err_vs_johnson']*100:.3f}\\%$** error (1-cell Bloch claim withdrawn) |"
    )
)

# %% [markdown]
# ---
#
# ## Limits & Failure Modes
#
# All numerical counter-case values referenced below are rendered directly from `data/summary.json`:

# %%
display(
    Markdown(
        f"1. **Small-signal Eulerian linearization vs large-signal saturation and electron overtaking ($|I_1| \\sim I_0$):**\n"
        f"   `wakis.beam_fluid.LinearizedBeam` evolves the first-order Eulerian perturbations $(v_1, \\rho_1)$ around a fixed DC drift $(u_0, \\rho_0)$ and therefore exhibits strictly linear gain by construction. **Counter-case:** In Karetnikova *et al.* (2018) Fig. 6 at ${kb['fig6_190ghz_40mm_20kv_freq_ghz']:.0f}\\;\\mathrm{{GHz}}$ (`{plc['fig6_190ghz_40mm_20kv_75w_at_1p01w']}`), driving the $40\\;\\mathrm{{mm}}$ grating with $P_{{\\mathrm{{in}}}} = {kb['fig6_190ghz_40mm_20kv_pin_w']:.2f}\\;\\mathrm{{W}}$ drives the beam into nonlinear saturation at $P_{{\\mathrm{{out}}}} \\approx {kb['fig6_190ghz_40mm_20kv_pout_w']:.0f}\\;\\mathrm{{W}}$ (${kb['fig6_190ghz_40mm_20kv_saturated_gain_db']:.1f}\\;\\mathrm{{dB}}$ saturated gain, well below the lossless linear small-signal prediction). Capturing electron overtaking, multi-streaming phase-space whorls, and saturated efficiency requires Lagrangian particle-in-cell tracking.\n"
        f"2. **Finite-$CN$ three-wave interference vs single-wave asymptotic Pierce formulas ($CN \\lesssim 0.3$):**\n"
        f"   Pierce's single-wave growth rate $\\alpha = \\beta_e C x_1$ describes the spatial slope only after the growing wave dominates ($CN \\gtrsim 0.3$). **Counter-case:** In a short $22$-period grating, the raw log-amplitude window slope does not follow $\\alpha \\propto I_0^{{1/3}}$ (`raw_window_loglog_slope = {crs['raw_window_loglog_slope']:.4f}`), mirror symmetry between $(-b, +1)$ and $(+b, -1)$ breaks down (${csm['alpha_neg_b_pos_sign_np_m']:.2f}$ vs ${csm['alpha_pos_b_neg_sign_np_m']:.2f}\\;\\mathrm{{Np/m}}$, unexplained by the mirror-symmetric 3-wave Pierce BVP), and across velocity detuning ($b = -0.5$ has ${vdb['rel_err_at_bm0p5']*100:.2f}\\%$ difference, not agreement; $b \\ge +0.7$ remains positive) or large space-charge parameter ($QC \\ge 0.45$ at fixed $b = 0$) the finite-window FIT slope diverges from the $CN \\to \\infty$ single-wave Pierce root (nor does a 1D total-field BVP fit explain the full $b$-sweep, failing for $b \\le 0$).\n"
        f"3. **Unexplained Pierce ($\\langle K\\rangle_{{\\mathrm{{beam}}}}$) & FIT overprediction vs Karetnikova *et al.* (2018) text values:**\n"
        f"   In a sub-millimeter grating, the synchronous slow space harmonic decays away from the grating teeth as $\\cosh(\\tau_{{+1}} y)$. At $186\\;\\mathrm{{GHz}}$, our 2D FIT solver resolves the $150\\;\\mu\\mathrm{{m}}$ sheet beam and agrees within ${t2['levels'][-1]['rel_err_vs_pierce_grid']*100:.2f}\\%$ on the `ultrafine` grid (pre-asymptotic; increments ${t2_inc_md}\\;\\mathrm{{Np/m}}$) with the beam-averaged $\\langle K_{{+1}}\\rangle_{{\\mathrm{{beam}}}}$ Pierce model; Pierce ($\\langle K\\rangle_{{\\mathrm{{beam}}}}$) overpredicts the paper by ${kb['overprediction_range_db'][1]:.2f}\\;\\mathrm{{dB}}$ at $20\\;\\mathrm{{kV}}$ (${v20['peak_gain_25mm_db']:.2f}$ vs ${kb['fig3_25mm_20kv_1d_range_db'][1]}\\;\\mathrm{{dB}}$, `{plc['fig3_25mm_20kv_5_to_13db_60ghz_band']}`) and $\\ge {kb['overprediction_range_db'][0]:.1f}\\;\\mathrm{{dB}}$ (peak not bracketed by the sweep) at $21\\;\\mathrm{{kV}}$ (${v21['peak_gain_25mm_db']:.2f}$ vs ${kb['fig3_25mm_21kv_1d_max_db']:.0f}\\;\\mathrm{{dB}}$, `{plc['fig3_25mm_21kv_19db']}`), while the FIT extrapolations deviate from Pierce by ${min(sw['rel_err_pct']):+.1f}\\%$ to ${max(sw['rel_err_pct']):+.1f}\\%$ in $\\alpha$ across the sweep and reach ${max(sw['extrapolated_gain_25mm_db']):.2f}\\;\\mathrm{{dB}}$ at $25\\;\\mathrm{{mm}}$. Replacing $\\langle K_{{+1}}\\rangle_{{\\mathrm{{beam}}}}$ by $K_{{+1}}(y=0)$ shifts the 1D Pierce curve down by ${kb['hypothetical_midplane_shift_range_db'][0]:.2f}\\text{{–}}{kb['hypothetical_midplane_shift_range_db'][1]:.2f}\\;\\mathrm{{dB}}$, but this remains an untested hypothesis regarding the paper's 1D code rather than a reconciliation of the 2D FIT solver.\n"
        f"4. **Unbounded Bloch unit cells vs finite-length BWO start-oscillation boundaries:**\n"
        f"   Johnson's (1955) BWO start-oscillation condition $(CN)_{{\\mathrm{{st}}}} = {bwo['johnson_1955_table1_CN']:.3f}$ is a global two-point boundary-value resonance between an unmodulated beam at the gun end ($z = 0$) and a zero-amplitude backward circuit wave at the collector end ($E_{{\\mathrm{{ckt}}}}(L) = 0$). **Counter-case:** A lossless 1-cell Bloch-periodic unit cell has no finite length $L$ and no collector boundary condition, so it cannot define a finite-length start threshold (claim withdrawn)."
    )
)

# %% [markdown]
# ---
#
# ## Verified References
#
# 1. **J. R. Pierce**, "Theory of the Beam-Type Traveling-Wave Tube," *Proceedings of the IRE* **35**(2), 111–123 (1947), [`doi:10.1109/JRPROC.1947.226217`](https://doi.org/10.1109/JRPROC.1947.226217); and **J. R. Pierce**, *Traveling-Wave Tubes*, D. Van Nostrand Company, New York (1950). — Derived the coupled circuit and electronic equations, the cubic/quartic dispersion relation in $(b, QC, d)$, the gain parameter $C = (I_0 K / (4 V_0))^{1/3}$, and the three-wave launching boundary conditions $G(\mathrm{dB}) = A + B C N$ with $A = -20\log_{10} 3$ and $B = 20\pi\sqrt{3}\log_{10}\mathrm{e}$.
# 2. **H. R. Johnson**, "Backward-Wave Oscillators," *Proceedings of the IRE* **43**(6), 684–697 (1955), [`doi:10.1109/JRPROC.1955.278054`](https://doi.org/10.1109/JRPROC.1955.278054). — Formulated the backward-wave dispersion equation $(\delta^2 + 4 QC)(\mathrm{j}\delta - b + \mathrm{j}d) = +1$ with collector boundary condition $V_c(L) = 0$ and tabulated the space-charge-free start-oscillation condition $(CN)_{\mathrm{st}} = 0.314$ and $b_{\mathrm{st}} = 1.522$ (Table I), establishing $I_{\mathrm{st}} \propto L^{-3}$.
# 3. **G. M. Branch and T. G. Mihran**, "Plasma frequency reduction factors in electron beams," *IRE Transactions on Electron Devices* **2**(2), 3–11 (1955), [`doi:10.1109/T-ED.1955.14065`](https://doi.org/10.1109/T-ED.1955.14065). — Derived closed-form eigenvalue equations for the space-charge plasma-frequency reduction factor $R$ of cylindrical and planar sheet electron beams inside conducting drift tunnels.
# 4. **T. A. Karetnikova, A. G. Rozhnev, N. M. Ryskin, A. E. Fedotov, S. V. Mishakin, and N. S. Ginzburg**, "Gain Analysis of a 0.2-THz Traveling-Wave Tube With Sheet Electron Beam and Staggered Grating Slow Wave Structure," *IEEE Transactions on Electron Devices* **65**(6), 2129–2134 (2018), [`doi:10.1109/TED.2017.2787960`](https://doi.org/10.1109/TED.2017.2787960) (`.ref/sws/2018_IEEETED65_0.2THz_TWT_sheet_beam_staggered_grating_gain.txt`). — Analyzed the $0.2\;\mathrm{THz}$ glide-symmetric staggered double-vane SWS ($d = 0.50\;\mathrm{mm}, s = 0.10\;\mathrm{mm}, h = 0.30\;\mathrm{mm}, 2a = 0.20\;\mathrm{mm}, b_{\mathrm{wg}} = 0.85\;\mathrm{mm}$) driven by a $20\;\mathrm{kV}, 100\;\mathrm{mA}$ sheet electron beam ($0.75\;\mathrm{mm}\times 0.15\;\mathrm{mm}$), deriving the two-plate depression factor Eq. (4) and reporting small-signal gain in text at $L = 25\;\mathrm{mm}$ ($5\text{–}13\;\mathrm{dB}$ over a $60\text{-GHz}$ band at $20\;\mathrm{kV}$, ll. 205–206, and $19\;\mathrm{dB}$ at $21\;\mathrm{kV}$, l. 209, in Section III-A / Fig. 3; $\approx 10\;\mathrm{dB}$ 3D particle-in-cell at $P_{\mathrm{in}} = 20\;\mathrm{mW}$, l. 328, in Section III-B / Fig. 5a) and at $L = 40\;\mathrm{mm}$ ($10\text{–}20\;\mathrm{dB}$ small-signal in Section III-B text, ll. 289–290; $\approx 75\;\mathrm{W}$ saturated 3D particle-in-cell output at $1.01\;\mathrm{W}$ input in Fig. 6 at $190\;\mathrm{GHz}$, ll. 278/291).
# 5. **T. Mealy, R. Marosi, and F. Capolino**, "Reduced Plasma Frequency Calculation Based on Particle-in-Cell Simulations," *IEEE Transactions on Plasma Science* **50**(10), 3570–3577 (2022), [`doi:10.1109/TPS.2022.3201036`](https://doi.org/10.1109/TPS.2022.3201036) ([`arXiv:2205.05806`](https://arxiv.org/abs/2205.05806)). — Formulated the space-charge reduction factor $R$ for relativistic sheet and pencil beams in rectangular and corrugated tunnels used in `wakis.beam_fluid`.
# 6. **R. Marosi, M. Zuboraj, and F. Capolino**, "Wideband Glide-Symmetric Slow-Wave Structure for Millimeter-Wave Sheet Beam TWTs," *arXiv:2505.22927v1* (2025), [`doi:10.48550/arXiv.2505.22927`](https://doi.org/10.48550/arXiv.2505.22927) (`.ref/sws/Marosi_2025_glide_sws.txt`). — Designed the V-band ($50\text{–}70\;\mathrm{GHz}$) glide-symmetric pillar waveguide SWS ($a = 1.40\;\mathrm{mm}, b = 0.80\;\mathrm{mm}, d = 1.05\;\mathrm{mm}$, Table I ll. 72–74) for a $5.2\;\mathrm{kV}, 11\;\mathrm{mA}$ sheet electron beam ($0.55\;\mathrm{mm}\times 0.10\;\mathrm{mm}$, i.e. $100\;\mu\mathrm{m}$ beam thickness, beam dimensions l. 103), reporting $4.7\;\mathrm{dB}$ cold insertion loss and 60-period small-signal gain ($\sim 19\;\mathrm{dB}$ peak at $58\;\mathrm{GHz}$ and secondary peak near $68\;\mathrm{GHz}$ in Fig. 7).
