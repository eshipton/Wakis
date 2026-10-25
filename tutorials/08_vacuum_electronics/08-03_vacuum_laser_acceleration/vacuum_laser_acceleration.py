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
# # Tutorial 8.3 · Laser Acceleration of Electrons in Vacuum — From Lawson–Woodward to Attosecond Bunches
#
# Optical lasers routinely achieve peak electric fields exceeding $10^{11}$–$10^{13}\;\text{V/m}$—three to five orders of magnitude above the breakdown limit of metallic radio-frequency (RF) cavities ($\lesssim 100\;\text{MV/m}$). Yet accelerating a free electron in vacuum with a laser beam faces a fundamental roadblock known as the **Lawson–Woodward theorem** (Woodward, 1947; Lawson, 1979; Palmer, 1980): over an infinite, boundary-free path in vacuum, the linear $\mathcal{O}(E_0)$ energy gain of a highly relativistic ($v \approx c$) electron vanishes identically because the phase velocity of every free-space electromagnetic wave projected along the particle trajectory exceeds $c$.
#
# Over a series of landmark papers, **Max Zolotorev** and his collaborators (Gennady Stupakov, Zhirong Huang, Thomas Plettner, Robert Byer, Alexander Zholents, Daniele Filippetto, and Pietro Musumeci) developed two complementary physical mechanisms that overcome the Lawson–Woodward theorem in vacuum:
#
# 1. **Boundary-terminated linear ($\mathcal{O}(E_0)$) acceleration in a semi-infinite vacuum** — Placing a thin, optically reflecting foil (such as gold-coated Kapton tape) inside the laser focus terminates the interaction at $z = z_s$, preventing the decelerating half-cycle downstream from cancelling the upstream energy gain (Huang, Stupakov & Zolotorev, *Phys. Rev. ST Accel. Beams* **7**, 011302, 2004; demonstrated experimentally by Plettner et al., *Phys. Rev. Lett.* **95**, 134801, 2005).
# 2. **Nonlinear ($\mathcal{O}(a^2)$) relativistic ponderomotive acceleration and focusing from rest** — An electron initially at rest in the waist of an intense, tightly focused ultrashort pulse ($a \sim 1$–$10$) violates the constant-velocity assumption of the Lawson–Woodward theorem. By matching the Rayleigh length to the relativistic slippage length, $Z_{R,\text{opt}} = \frac{1}{2}a^2 c\tau$, the electron is accelerated forward and escapes the focus radially/axially before the trailing edge of the pulse can decelerate it, retaining the full intra-pulse drift energy $\gamma_{\text{opt}} = 1 + \frac{1}{2}a^2$. Superposing two orthogonally polarized Gaussian modes ($\text{LG}_{00} \perp \text{LG}_{01}$) creates a transverse **"crater" ponderomotive potential** that suppresses radial blowout and compresses the electrons into a $\sigma_t \approx 50\;\text{as}$ bunch with dimensionless Fermi brightness $B_F \approx 10^{-6}$—six orders of magnitude brighter than conventional RF photoinjectors and sufficient to drive a tabletop $10\;\text{keV}$ X-ray free-electron laser (FEL) using an optical laser wiggler (Stupakov & Zolotorev, *Phys. Rev. Lett.* **86**, 5274, 2001; Zolotorev, *Nucl. Instrum. Methods Phys. Res. A* **483**, 445, 2002).
#
# ### Learning goals
#
# 1. **Part A · Free electron in a plane wave** — Derive the exact relativistic invariants $u_x(\eta) = a_x(\eta)$ and $\gamma(\eta) - u_z(\eta) = 1$, the closed **figure-of-eight** (lemniscate) orbit in the average rest frame, the peak intra-pulse energy $\gamma_{\max} = 1 + a_0^2/2$, and the zero net energy gain $\Delta W = 0$ after a finite pulse passes. Validate relativistic **Boris** and **Vay (2008)** particle pushers against the exact solution to second order, $\mathcal{O}(\Delta t^2)$.
# 2. **Part B · Focused Gaussian beam and the Lawson–Woodward theorem** — State all five assumptions of the Lawson–Woodward theorem explicitly. Show analytically and with a 3D Wakis Finite Integration Technique (`SolverFIT3D`) simulation on a graded transverse mesh that the synchronous longitudinal field $E_\parallel(z)$ along a relativistic ($v \approx c$) trajectory through a paraxial $\text{TEM}_{00}$ focus is an exact total derivative whose integral over $(-\infty, +\infty)$ vanishes due to Gouy phase and crossing-angle slippage.
# 3. **Part C · Breaking Lawson–Woodward with a reflecting boundary** — Insert a zero-thickness triangulated Perfect Electric Conductor (PEC) screen (`wakis.conformal.rectangle_sheet`) into the 3D FIT focal region to model the gold-coated Kapton foil of Plettner et al. (2005). Compute $\Delta W$ vs screen position $z_s / Z_R$, laser polarization angle $\phi_{\text{pol}}$ ($\Delta W \propto \cos\phi_{\text{pol}}$), crossing angle $s = \theta_c Z_R / w_0$ ($s_{\text{opt}} \approx 1.121$), and transverse laser mode ($\text{TEM}_{00}$ vs on-axis $\text{TEM}_{10}$ from Huang, Stupakov & Zolotorev, 2004).
# 4. **Part D · Ponderomotive acceleration from rest and the two-mode crater trap** — Derive the relativistic ponderomotive equations of motion in the light-cone invariant $M = (\gamma - u_z)^2$, explain why diffraction ($\partial_z a^2 < 0$) breaks the plane-wave invariant to lock in $\gamma_{\text{opt}} = 1 + a^2/2$ at $Z_{R,\text{opt}} = \frac{1}{2}a^2 c\tau$, and reproduce Figs. 1–5 of Zolotorev (2002), including the two-mode crater profile and the compressed $\sim 50\;\text{as}$ bunch phase spaces.
# 5. **Part E · Laser as an electromagnetic wiggler and Fermi brightness $B_F$** — Reconcile the head-on laser-wiggler resonance $\lambda_x = \lambda_L(1 + a^2/2)/(4\gamma^2)$ with Zolotorev (2002), compute the $15$–$28\;\text{MeV}$ electron energy required for $10\;\text{keV}$ X-rays (versus $\sim 15\;\text{GeV}$ for a cm-period static magnetic undulator), and evaluate the FEL gain-length scaling with the dimensionless invariant Fermi brightness $B_F = N\bar{\lambda}_c^3 / (\varepsilon_{\text{inv},x}\varepsilon_{\text{inv},y}\varepsilon_{\text{inv},z})$.
#
# ![Setup and mesh schematic](figures/01_setup_and_mesh_schematic.png)
#
# ---
#
# ## Part A · Free Electron in a Plane Wave
#
# ### 1. Relativistic equations of motion and exact invariants
#
# Consider a linearly polarized electromagnetic plane wave propagating along $+z$ in vacuum ($\varepsilon_r = \mu_r = 1$), whose vector potential $\mathbf{A}(\mathbf{r}, t) = A_x(\eta)\,\hat{\mathbf{x}}$ depends solely on the Lorentz-invariant wave phase
#
# $$
# \eta = \omega_0\!\left(t - \frac{z}{c}\right) = \omega_0 t - k_0 z \quad [\text{rad}],
# \qquad
# k_0 = \frac{\omega_0}{c} = \frac{2\pi}{\lambda_L}.
# $$
#
# Introducing the dimensionless normalized vector potential $\mathbf{a}(\eta) = \frac{e\,\mathbf{A}(\eta)}{m_e c}$ (where $e > 0$ is the elementary charge) and dimensionless 4-velocity $\mathbf{u} = \gamma\boldsymbol{\beta} = \frac{\mathbf{p}}{m_e c}$ with $\gamma = \sqrt{1 + |\mathbf{u}|^2}$, the electric and magnetic fields are
#
# $$
# E_x(\eta) = -\frac{\partial A_x}{\partial t} = -\frac{m_e c \omega_0}{e}\,\frac{da_x}{d\eta},
# \qquad
# B_y(\eta) = \frac{\partial A_x}{\partial z} = -\frac{m_e \omega_0}{e}\,\frac{da_x}{d\eta} = \frac{E_x(\eta)}{c}.
# $$
#
# For an electron of charge $q = -e$, the relativistic Lorentz force equations $\frac{d\mathbf{p}}{dt} = -e(\mathbf{E} + \mathbf{v}\times\mathbf{B})$ and energy equation $\frac{d\gamma}{dt} = -\frac{e}{m_e c^2}\mathbf{v}\cdot\mathbf{E}$ become
#
# $$
# \frac{du_x}{dt} = \omega_0(1 - \beta_z)\frac{da_x}{d\eta},
# \qquad
# \frac{du_y}{dt} = 0,
# \qquad
# \frac{du_z}{dt} = \omega_0 \beta_x \frac{da_x}{d\eta},
# \qquad
# \frac{d\gamma}{dt} = \omega_0 \beta_x \frac{da_x}{d\eta}.
# $$
#
# Since $\frac{d\eta}{dt} = \omega_0\left(1 - \frac{dz/dt}{c}\right) = \omega_0(1 - \beta_z) = \frac{\omega_0(\gamma - u_z)}{\gamma}$, dividing the transverse equation by $\frac{d\eta}{dt}$ and subtracting the axial momentum equation from the energy equation yields two **exact constants of motion**:
#
# $$
# \frac{d}{d\eta}\big(u_x(\eta) - a_x(\eta)\big) = 0,
# \qquad
# \frac{d}{d\eta}\big(\gamma(\eta) - u_z(\eta)\big) = 0.
# $$
#
# For an electron initially at rest ($\mathbf{u}(-\infty) = \mathbf{0}$, $\gamma(-\infty) = 1$) before the laser pulse arrives ($a_x(-\infty) = 0$):
#
# $$
# u_x(\eta) = a_x(\eta), \qquad u_y(\eta) = 0, \qquad \gamma(\eta) - u_z(\eta) = 1.
# $$
#
# Combining $\gamma^2 - u_z^2 = (\gamma - u_z)(\gamma + u_z) = 1 + u_x^2 = 1 + a_x(\eta)^2$ with $\gamma - u_z = 1$ gives $\gamma + u_z = 1 + a_x(\eta)^2$, and hence
#
# $$
# u_z(\eta) = \frac{1}{2}\,a_x(\eta)^2,
# \qquad
# \gamma(\eta) = 1 + \frac{1}{2}\,a_x(\eta)^2.
# $$
#
# ### 2. Peak intra-pulse energy, zero net gain, and the figure-of-eight orbit
#
# - **Peak energy inside the pulse**: At the peak of a pulse with amplitude $a_0$, the electron reaches the maximum Lorentz factor
#   $$\gamma_{\max} = 1 + \frac{1}{2}a_0^2 \qquad (\text{Zolotorev 2002, p. 447}).$$
# - **Zero net energy after the pulse passes (Lawson–Woodward for a plane wave)**: Because $\gamma(\eta) = 1 + \frac{1}{2}a_x(\eta)^2$ is an algebraic function of the local vector potential $a_x(\eta)$, once a finite pulse has completely overtaken the electron ($\eta \to +\infty$, where $a_x(+\infty) = 0$), the Lorentz factor returns identically to
#   $$\gamma(+\infty) = 1 \quad \Longrightarrow \quad \Delta W = m_e c^2\big(\gamma(+\infty) - 1\big) \equiv 0.$$
# - **Figure-of-eight orbit in the average rest frame**: Since $\frac{dx}{d\eta} = \frac{c\,u_x(\eta)}{\omega_0(\gamma - u_z)} = \frac{a_x(\eta)}{k_0}$ and $\frac{dz}{d\eta} = \frac{c\,u_z(\eta)}{\omega_0(\gamma - u_z)} = \frac{a_x(\eta)^2}{2 k_0}$, a continuous monochromatic carrier $a_x(\eta) = a_0\sin\eta$ gives
#
#   $$
#   k_0 x(\eta) = -a_0\cos\eta,
#   \qquad
#   k_0 z(\eta) = \frac{a_0^2}{4}\eta - \frac{a_0^2}{8}\sin(2\eta).
#   $$
#
#   Subtracting the secular longitudinal drift $k_0\bar{z}(\eta) = \frac{1}{4}a_0^2\eta$ leaves $k_0\tilde{z}(\eta) = -\frac{1}{8}a_0^2\sin(2\eta)$, which satisfies the algebraic **lemniscate of Gerono** (figure-of-eight):
#
#   $$
#   16\,(k_0\tilde{z})^2 = (k_0 x)^2\big(a_0^2 - (k_0 x)^2\big).
#   $$
#
# ![Part A plane-wave orbits and pusher convergence](figures/02_part_a_plane_wave_orbits.png)
# ![Plane-wave figure-of-eight animation](figures/plane_wave_figure_of_eight.gif)

# %%
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parent
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as h  # noqa: E402

TASKS = tuple(h.TASKS.keys())


def run_part_a_section(quick: bool = True) -> dict:
    """Execute or load Part A (plane-wave figure-of-eight orbit and Boris/Vay validation)."""
    json_path = h.DATA_DIR / "task_part_a_plane_wave.json"
    if json_path.exists():
        summary = json.loads(json_path.read_text())
    else:
        summary = h.run_task_part_a_plane_wave(quick=quick)
    print(
        f"[Part A] a0 = {summary['a0_pulse']:.2f}: "
        f"gamma_max = {summary['gamma_peak_boris']:.4f} (exact {summary['gamma_peak_theory']:.4f}), "
        f"net |dW| = {abs(summary['net_dW_eV_vay']):.3e} eV, "
        f"convergence order p_Boris = {summary['boris_order_p']:.2f}, p_Vay = {summary['vay_order_p']:.2f}"
    )
    return summary


# %% [markdown]
# ---
#
# ## Part B · Focused Gaussian Beam and the Lawson–Woodward Theorem
#
# ### 1. Explicit statement of the Lawson–Woodward theorem
#
# The **Lawson–Woodward theorem** states that a charged particle moving through electromagnetic fields in free space experiences **zero net linear ($\mathcal{O}(E_0)$) energy gain**, $\Delta W = q\int_{-\infty}^{+\infty}\mathbf{E}(\mathbf{r}(t), t)\cdot\mathbf{v}\,dt = 0$, whenever the following five assumptions simultaneously hold:
#
# | Assumption | Physical meaning | How it is broken in practice |
# | :--- | :--- | :--- |
# | **(i) Vacuum ($n = 1$)** | No dielectric gas, plasma, or slow-wave medium ($v_{\text{ph}} < c$) | Inverse Cherenkov acceleration ($n > 1$, Tutorial 8.2) |
# | **(ii) No boundaries or walls** | Free-space plane-wave spectrum contains only $\|\mathbf{k}\| = \omega/c$ (no evanescent or transition fields) | **Part C**: Thin gold-coated Kapton foil placed in the focus (Huang et al. 2004; Plettner et al. 2005) |
# | **(iii) Constant velocity $\mathbf{v} = \text{const}$ ($v \approx c$)** | Particle trajectory is unperturbed; only $\mathcal{O}(E_0)$ force is kept | **Part D**: $\mathcal{O}(a^2)$ relativistic ponderomotive acceleration from rest ($\mathbf{v}_0 = \mathbf{0}$, Stupakov & Zolotorev 2001) |
# | **(iv) Infinite interaction path $(-\infty, +\infty)$** | Accelerating and decelerating phases integrate to zero | **Part C & Part D**: Screen terminates path at $z_s$, or electron escapes focus at $z \sim Z_R$ |
# | **(v) No static $\mathbf{E}$ or $\mathbf{B}$ fields** | No magnetic undulator / wiggler bending the trajectory | **Part E**: Inverse FEL / electromagnetic wiggler resonance |
#
# ### 2. Paraxial $\text{TEM}_{00}$ beam, Gouy phase, and the exact total-derivative identity
#
# A monochromatic $\text{TEM}_{00}$ Gaussian laser beam of wavelength $\lambda_L = 2\pi / k_0$, waist radius $w_0$, Rayleigh length $Z_R = \frac{1}{2}k_0 w_0^2 = \pi w_0^2 / \lambda_L$, diffraction angle $\theta_d = 1/(k_0 w_0) = w_0 / (2 Z_R)$, and complex beam parameter $q(z) = z - i Z_R$ has transverse electric phasor (with time factor $e^{-i\omega_0 t}$)
#
# $$
# \mathcal{E}_{x_L}(x_L, y_L, z_L) = E_0\,\frac{-i Z_R}{q(z_L)}\exp\!\left(i k_0 z_L + \frac{i k_0(x_L^2 + y_L^2)}{2\,q(z_L)}\right)
# = \frac{E_0}{1 + i z_L/Z_R}\exp\!\left(-\frac{x_L^2 + y_L^2}{w_0^2(1 + i z_L/Z_R)} + i k_0 z_L\right),
# $$
#
# where $\psi_G(z_L) = \arctan(z_L/Z_R)$ is the **Gouy phase shift** (giving a superluminal on-axis phase velocity $v_{\text{ph}}(z) = \omega_0 / (k_0 - d\psi_G/dz) > c$). By Gauss's law in vacuum ($\nabla\cdot\mathbf{E} = 0$), the beam carries a first-order Lax–Louisell–McKnight longitudinal electric field:
#
# $$
# \mathcal{E}_{z_L}(x_L, y_L, z_L) = \frac{i}{k_0}\frac{\partial \mathcal{E}_{x_L}}{\partial x_L}
# = -\frac{x_L}{q(z_L)}\,\mathcal{E}_{x_L}(x_L, y_L, z_L)
# = -\frac{i x_L}{Z_R(1 + i z_L/Z_R)}\,\mathcal{E}_{x_L}(x_L, y_L, z_L).
# $$
#
# Now let an ultrarelativistic electron ($v \to c$) cross the laser waist at a small angle $\theta_c \ll 1$ in the $(x, z)$ plane along $x_L(z) = -\theta_c z$, $y_L = 0$, $z_L \approx z(1 - \frac{1}{2}\theta_c^2)$ at synchronous time $t = z / c$. If the laser polarization vector makes an angle $\phi_{\text{pol}}$ with the $(x, z)$ crossing plane, the synchronous electric field projected along the electron trajectory is
#
# $$
# E_\parallel(z) = \cos\phi_{\text{pol}}\Big(-\theta_c\,\mathcal{E}_{x_L} + \mathcal{E}_{z_L}\Big)e^{-i\omega_0 z/c + i\phi_0}
# = \theta_c \cos\phi_{\text{pol}}\,\mathcal{E}_{x_L}\!\left(-1 + \frac{z}{z - i Z_R}\right)e^{-i\omega_0 z/c + i\phi_0}.
# $$
#
# Combining $-1 + \frac{z}{z - i Z_R} = \frac{i Z_R}{z - i Z_R} = -\frac{1}{1 + i\zeta}$ (with $\zeta = z / Z_R$) and introducing the dimensionless crossing parameter $s = \frac{\theta_c Z_R}{w_0} = \frac{\theta_c}{\sqrt{2}\,\theta_d}$ reduces the synchronous integrand to an **exact total derivative**:
#
# $$
# E_\parallel(z) = -\frac{\theta_c E_0 \cos\phi_{\text{pol}}}{(1 + i\zeta)^2}\exp\!\left(\frac{-i s^2 \zeta}{1 + i\zeta} + i\phi_0\right)
# = \frac{E_0 w_0 \cos\phi_{\text{pol}}}{i s}\,\frac{d}{dz}\left[\exp\!\left(\frac{-i s^2 \zeta}{1 + i\zeta} + i\phi_0\right)\right].
# $$
#
# Because $\lim_{\zeta \to \pm\infty}\frac{-i s^2 \zeta}{1 + i\zeta} = -s^2$ at **both** limits $\zeta \to -\infty$ and $\zeta \to +\infty$, integrating over the infinite path gives
#
# $$
# \Delta W(-\infty, +\infty) = -e\,\operatorname{Re}\int_{-\infty}^{+\infty} E_\parallel(z)\,dz
# = -e\,\operatorname{Re}\left[\frac{E_0 w_0 \cos\phi_{\text{pol}}}{i s}\,e^{i\phi_0}\Big(e^{-s^2} - e^{-s^2}\Big)\right] \equiv 0.
# $$
#
# ![Part B 3D FIT paraxial validation and Lawson-Woodward cancellation](figures/03_part_b_paraxial_and_lawson_woodward.png)

# %% [markdown]
# ---
#
# ## Part C · Breaking Lawson–Woodward with a Boundary (Huang–Stupakov–Zolotorev 2004; Plettner et al. 2005)
#
# ### 1. Analytical energy gain and optimal crossing angle
#
# When a thin, optically reflecting screen (such as the $8\;\mu\text{m}$ Kapton tape coated with $0.09$–$0.9\;\mu\text{m}$ of gold used by Plettner et al., *PRL* **95**, 134801, 2005) terminates the laser field at $z = z_s = \zeta_s Z_R$ while letting the relativistic electron pass through, the upper limit of the exact antiderivative is evaluated at $\zeta_s$ instead of $+\infty$. Expressing $E_0 w_0 = \sqrt{\frac{4 Z_0 P}{\pi}}$ in terms of the peak laser power $P = \frac{1}{4}\pi w_0^2 E_0^2 / Z_0$ yields the closed-form maximum energy gain:
#
# $$
# \Delta W_{\max}(\zeta_s, s, \phi_{\text{pol}}) = e\sqrt{Z_0 P}\;\frac{2}{\sqrt{\pi}}\,|\cos\phi_{\text{pol}}|\,\frac{1}{s}\left|\exp\!\left(\frac{-i s^2 \zeta_s}{1 + i\zeta_s}\right) - e^{-s^2}\right|.
# $$
#
# Placing the screen at the laser waist ($\zeta_s = 0$) simplifies the gain to
#
# $$
# \Delta W_{\max}(0, s, \phi_{\text{pol}}) = e\sqrt{Z_0 P}\;\frac{2}{\sqrt{\pi}}\,|\cos\phi_{\text{pol}}|\,\frac{1 - e^{-s^2}}{s}.
# $$
#
# Transcendental optimization of $g(s) = (1 - e^{-s^2})/s$ gives $2 s_{\text{opt}}^2 e^{-s_{\text{opt}}^2} = 1 - e^{-s_{\text{opt}}^2}$, solved by
#
# $$
# s_{\text{opt}} \approx 1.1209 \quad (\theta_{c,\text{opt}} \approx 1.585\,\theta_d),
# \qquad
# \Delta W_{\max}(0, s_{\text{opt}}, 0) \approx 0.7201\,e\sqrt{Z_0 P}.
# $$
#
# ### 2. Laser mode optimization (Huang, Stupakov & Zolotorev, *PRSTAB* **7**, 011302, 2004)
#
# Using the overlap integral between the incident laser mode and the forward transition radiation of the electron crossing the conducting boundary, Huang, Stupakov, and Zolotorev (2004) showed that odd-symmetry Hermite–Gaussian ($\text{TEM}_{10}$) and radially polarized Laguerre–Gaussian ($\text{TM}_{01}$) modes produce a pure axial electric field $E_z(0, 0, z) \propto (1 + i z/Z_R)^{-2} e^{i k_0 z}$ right on the optical axis ($\theta_c = 0$) with zero transverse deflecting force $E_x(0, 0, z) = 0$. Integrating $(1 + i\zeta)^{-2} = -i\frac{d}{d\zeta}(1 + i\zeta)^{-1}$ from $-\infty$ to $\zeta_s = z_s / Z_R$ yields the universal Lorentzian screen-position dependence
#
# $$
# \Delta W_{\max}^{\text{TEM}_{10}}(z_s) = \frac{\Delta W_{\max}^{\text{TEM}_{10}}(0)}{\sqrt{1 + (z_s / Z_R)^2}}.
# $$
#
# ![Part C 3D FIT PEC screen reflection, position scan, and polarization law](figures/04_part_c_screen_polarization_and_angle.png)
# ![3D FIT focused beam reflecting from PEC screen](figures/fit_beam_screen_reflection.gif)

# %%
def run_part_b_c_section(quick: bool = True) -> dict:
    """Execute or load Parts B & C (3D Wakis FIT focused Gaussian beam and conformal PEC screen)."""
    json_path = h.DATA_DIR / "task_part_b_c_fit_screen.json"
    if json_path.exists():
        summary = json.loads(json_path.read_text())
    else:
        summary = h.run_task_part_b_c_fit_screen(quick=quick)
    print(
        f"[Parts B & C] Waist L2 error: Ex = {summary['ex_waist_l2_rel_err']*100:.2f}%, "
        f"Ez = {summary['ez_waist_l2_rel_err']*100:.2f}% | "
        f"Unbounded LW |dW_sin|/(e*sqrt(Z0*P)) = {summary['dW_unbounded_sin_norm']:.4f} vs "
        f"PEC screen at z_s=0: {summary['dW_screen_waist_fit_norm']:.4f} "
        f"(theory {summary['dW_screen_waist_theory_norm']:.4f}, rel err {summary['screen_waist_rel_err']*100:.2f}%)"
    )
    return summary


# %% [markdown]
# ---
#
# ## Part D · Ponderomotive Acceleration from Rest & Attosecond Bunching (Stupakov & Zolotorev 2001; Zolotorev 2002)
#
# ### 1. Relativistic ponderomotive Hamiltonian and breaking the plane-wave invariant
#
# In a short, intense focused laser pulse with slowly varying envelope $a^2(\mathbf{r}_\perp, z, \xi)$ depending on the comoving pulse coordinate $\xi = c t - z$ and the diffraction scale $z / Z_R$, averaging over the fast optical carrier $\omega_0$ replaces the rest mass by the effective ponderomotive mass $m_{\text{eff}}^2 = m_e^2(1 + a^2)$. The cycle-averaged relativistic Lorentz factor is
#
# $$
# \gamma(\mathbf{r}_\perp, z, \mathbf{u}, \xi) = \sqrt{1 + \mathbf{u}_\perp^2 + u_z^2 + a^2(\mathbf{r}_\perp, z, \xi)}.
# $$
#
# Changing the independent variable from laboratory time $t$ to the comoving coordinate $\xi = c t - z$ (using $\frac{d\xi}{d(ct)} = 1 - \frac{u_z}{\gamma} = \frac{\gamma - u_z}{\gamma}$) and defining the positive light-cone variable $M(\xi) \equiv (\gamma - u_z)^2$ gives the exact 3D relativistic ponderomotive equations of motion:
#
# $$
# \frac{d\mathbf{r}_\perp}{d\xi} = \frac{\mathbf{u}_\perp}{\sqrt{M}},
# \qquad
# \frac{d\mathbf{u}_\perp}{d\xi} = -\frac{1}{2\sqrt{M}}\,\nabla_\perp a^2(\mathbf{r}_\perp, z, \xi),
# \qquad
# \frac{dz}{d\xi} = \frac{1 + \mathbf{u}_\perp^2 + a^2 - M}{2 M},
# \qquad
# \frac{dM}{d\xi} = \frac{\partial a^2(\mathbf{r}_\perp, z, \xi)}{\partial z}\bigg|_{\mathbf{r}_\perp, \xi}.
# $$
#
# This system makes the physics of Stupakov & Zolotorev (*PRL* **86**, 5274, 2001) and Zolotorev (*NIM A* **483**, 445, 2002) completely transparent:
#
# - **Why a plane wave gives zero net gain**: When $Z_R \to \infty$, $\frac{\partial a^2}{\partial z}\big|_\xi \equiv 0$, so $M(\xi) \equiv 1$ is conserved. After the pulse passes ($a^2 \to 0$), $\gamma_f = \frac{1}{2}(\sqrt{M_f} + 1/\sqrt{M_f}) = 1$.
# - **Why diffraction locks in net energy**: In a focused beam ($Z_R < \infty$), $a^2(0, z, \xi) = \frac{a_0^2}{1 + (z/Z_R)^2}\exp(-\xi^2/(\pi c^2\tau^2))$ decreases with $z > 0$ ($\frac{\partial a^2}{\partial z}\big|_\xi < 0$). As the electron is pushed forward into the diffracting region $z \sim Z_R$, $M(\xi)$ decreases below $1$, so when the trailing edge of the pulse overtakes the electron in the weakened far field, $M_f \ll 1$ and the electron retains $\gamma_f = \frac{1}{2}(\sqrt{M_f} + 1/\sqrt{M_f}) \gg 1$.
# - **Optimal Rayleigh length $Z_{R,\text{opt}}$ and maximum energy $\gamma_{\text{opt}}$**: Rescaling $\tilde{z} = z / Z_R$ and $\eta = \xi / (c\tau)$ in $\frac{dz}{d\xi}$ shows that the trajectory depends on the single dimensionless parameter $\kappa = \frac{Z_R}{a_0^2 c\tau / 2}$. Maximum energy transfer occurs when the electron reaches $z \sim Z_R$ right at the peak of the pulse ($\kappa = 1$):
#
#   $$
#   Z_{R,\text{opt}} = \frac{1}{2}\,a_0^2\,c\tau,
#   \qquad
#   \gamma_{\text{opt}} = 1 + \frac{1}{2}\,a_0^2
#   \qquad (\text{Zolotorev 2002, p. 447, Figs. 1–2}).
#   $$
#
# ### 2. Transverse blowout instability and the two-mode "crater" potential
#
# A single fundamental Gaussian mode $I_1(r)/I_0 = \exp(-r^2/(2\sigma^2))$ has $\partial_r I_1 < 0$ everywhere off-axis ($r > 0$), exerting a strong outward radial ponderomotive force $F_r \propto -\partial_r I_1 > 0$ that violently blows off-axis electrons out of the beam waist. To stabilize the transverse motion without destructive optical interference, Stupakov & Zolotorev (2001) and Zolotorev (2002, p. 447, Fig. 3) superpose two lowest-order Gaussian modes with **orthogonal polarization**:
#
# $$
# \frac{I_1(r)}{I_0} = \exp\!\left(-\frac{r^2}{2\sigma^2}\right),
# \qquad
# \frac{I_2(r)}{I_0} = \alpha_{\text{mode}}\!\left(\frac{r^2}{2\sigma^2}\right)\exp\!\left(-\frac{r^2}{2\sigma^2}\right),
# \qquad
# \frac{I_{\text{tot}}(r)}{I_0} = \left(1 + \alpha_{\text{mode}}\frac{r^2}{2\sigma^2}\right)\exp\!\left(-\frac{r^2}{2\sigma^2}\right).
# $$
#
# For $\alpha_{\text{mode}} > 1$ ($\alpha_{\text{mode}} = 1.25$ in Fig. 3 of Zolotorev 2002), $I_{\text{tot}}(r)$ forms a shallow radial **"crater"** around $r = 0$ bounded by a circular rim at
#
# $$
# \frac{r_{\text{rim}}}{\sigma} = \sqrt{2\!\left(1 - \frac{1}{\alpha_{\text{mode}}}\right)} \approx 0.632 \quad (\alpha_{\text{mode}} = 1.25).
# $$
#
# Inside the crater ($r < r_{\text{rim}}$), $\frac{\partial I_{\text{tot}}}{\partial r} > 0$, providing a continuous inward restoring force $F_r < 0$ that confines the bunch within $|x| \lesssim 0.2\;\mu\text{m}$ while relativistic kinematic compression ($\Delta z_{\text{lab}} \approx (1 - \beta_z)\Delta z \sim \Delta z_0 / (2\gamma^2)$) compresses a $5\;\mu\text{m}$ initial waist column into a $\sigma_t \approx 50\;\text{as}$ ($\sigma_z \approx 15\;\text{nm}$) attosecond slice (Figs. 4–5 of Zolotorev 2002).
#
# ![Part D ponderomotive scaling and crater potential](figures/05_part_d_ponderomotive_scaling_and_crater.png)
# ![Part D attosecond phase spaces](figures/06_part_d_attosecond_phase_space.png)
# ![Ponderomotive crater bunching animation](figures/ponderomotive_crater_bunching.gif)

# %%
def run_part_d_section(quick: bool = True) -> dict:
    """Execute or load Part D (ponderomotive acceleration from rest and ~50 as crater bunching)."""
    json_path = h.DATA_DIR / "task_part_d_ponderomotive.json"
    if json_path.exists():
        summary = json.loads(json_path.read_text())
    else:
        summary = h.run_task_part_d_ponderomotive(quick=quick)
    print(
        f"[Part D] a0 = {summary['a0_ref']:.1f}: peak at Z_R/Z_{{R,opt}} = {summary['kappa_at_peak']:.2f}, "
        f"gamma(6*Z_R) = {summary['gamma_at_z_6zr']:.2f} (gamma_opt = {summary['gamma_opt_theory']:.2f}), "
        f"crater max|x| = {summary['crater_max_x_um']:.3f} um vs Gaussian blowout {summary['gaussian_max_x_um']:.1f} um, "
        f"RMS bunch sigma_t = {summary['bunch_rms_duration_as']:.1f} as (sigma_z = {summary['bunch_rms_length_nm']:.1f} nm)"
    )
    return summary


# %% [markdown]
# ---
#
# ## Part E · Why It Matters: The Laser as an Electromagnetic Wiggler & Fermi Brightness $B_F$
#
# ### 1. Reconciling the electromagnetic wiggler resonance condition
#
# In a conventional X-ray free-electron laser, a relativistic electron beam of Lorentz factor $\gamma$ traverses a static magnetic undulator of period $\lambda_u \sim 2$–$3\;\text{cm}$, radiating at fundamental wavelength $\lambda_x = \frac{\lambda_u}{2\gamma^2}(1 + K^2/2)$ and therefore requiring multi-GeV electron beams ($E_e \sim 10$–$15\;\text{GeV}$) to reach hard X-rays ($\hbar\omega_x = 10\;\text{keV}$, $\lambda_x = 1.240\;\text{\AA}$).
#
# Replacing the static magnet by a counter-propagating (head-on) optical laser pulse of wavelength $\lambda_L$ and normalized RMS vector potential $a$ (playing the role of the undulator parameter $K$) doubles the Doppler shift because the wiggler phase $\eta_W = \omega_L t + k_L z = k_L(c + v_z)t = (1 + \beta)k_L z/\beta$ has effective wiggler wavenumber
#
# $$
# k_W = (1 + \beta)k_L \approx 2 k_L \quad \Longleftrightarrow \quad \lambda_W = \frac{\lambda_L}{1 + \beta} \approx \frac{\lambda_L}{2}.
# $$
#
# Substituting $k_W \approx 2 k_L$ into Zolotorev's resonance formula (*NIM A* **483**, 445, 2002, p. 445),
#
# $$
# k_x = \frac{2\gamma^2 k_W}{1 + a^2/2} \approx \frac{4\gamma^2 k_L}{1 + a^2/2}
# \quad \Longleftrightarrow \quad
# \lambda_x = \frac{\lambda_L(1 + a^2/2)}{4\gamma^2},
# $$
#
# reduces the wiggler period by four orders of magnitude ($\lambda_W \sim 0.2$–$0.5\;\mu\text{m}$ vs $\lambda_u \sim 3\;\text{cm}$) and reduces the required electron energy for $10\;\text{keV}$ X-rays by $\sim 10^3\times$:
#
# - For $\lambda_L = 1.0\;\mu\text{m}$ and $a = 1$: $\gamma \approx 54.9 \implies E_e \approx 28.0\;\text{MeV}$ ($25$–$28\;\text{MeV}$ in Zolotorev 2002, p. 446).
# - For $\lambda_L = 0.4\;\mu\text{m}$ and $a = 1$: $\gamma \approx 34.7 \implies E_e \approx 17.7\;\text{MeV}$ ($\sim 15\;\text{MeV}$ at $a \approx 0.65$ in Zolotorev 2002, p. 446).
#
# ### 2. Dimensionless invariant Fermi brightness $B_F$ and FEL gain length
#
# Why cannot a conventional RF photoinjector drive a tabletop optical-wiggler X-ray FEL at $15$–$25\;\text{MeV}$? Zolotorev (2002, p. 446) answers this via the **dimensionless invariant Fermi brightness** (the 6D phase-space degeneracy factor relative to the Pauli exclusion limit, where $\bar{\lambda}_c = \hbar/(m_e c) = 3.8616\times 10^{-13}\;\text{m}$ is the reduced Compton wavelength):
#
# $$
# B_F = \frac{N\,\bar{\lambda}_c^3}{\varepsilon_{\text{inv},x}\,\varepsilon_{\text{inv},y}\,\varepsilon_{\text{inv},z}}.
# $$
#
# The number of wiggler periods $M_A$ per FEL power gain length $L_G = M_A \lambda_W$ satisfies (Zolotorev 2002, p. 446)
#
# $$
# \frac{1}{2\pi M_A} = \frac{\alpha}{4}\,\frac{W}{e}\,\sqrt{\frac{3}{2\pi}}\,B_F\,\left(\frac{E_e}{\hbar\omega_W}\right),
# $$
#
# where $\alpha \approx 1/137.036$ is the fine-structure constant and $W/e \sim 10^{-2}$ is the fractional spontaneous energy loss per gain length:
#
# - **Conventional electron sources ($B_F \approx 10^{-12}$)**: Give $M_A \sim 10^8$ wiggler periods per gain length, requiring a Rayleigh length $Z_R \sim M_A \lambda_L \sim 40\;\text{m}$ and a prohibitive wiggler laser pulse energy $\mathcal{E}_{\text{wiggler}} \propto M_A^2 \sim 10^{12}\;\text{J}$.
# - **Ponderomotive crater-trapped attosecond bunch ($N \approx 10^5$, $\varepsilon_{\text{inv},x} = \varepsilon_{\text{inv},y} = 2\times 10^{-3}\;\text{mm}\cdot\text{mrad} = 2\;\text{nm}\cdot\text{rad}$, $\varepsilon_{\text{inv},z} = 1.5\;\text{nm}$)**: Achieves
#
#   $$
#   B_F = \frac{10^5 \times (3.8616\times 10^{-13}\;\text{m})^3}{(2\times 10^{-9}\;\text{m})^2(1.5\times 10^{-9}\;\text{m})} \approx 0.96\times 10^{-6} \approx 10^{-6},
#   $$
#
#   a **$10^6\times$ increase in brightness** that reduces the FEL gain length to $M_A \sim 10^2$ optical periods and the required wiggler laser energy to a few Joules (Zolotorev 2002, pp. 446–448).
#
# ![Part E laser wiggler resonance and Fermi brightness scaling](figures/07_part_e_laser_wiggler_and_brightness.png)

# %%
def run_part_e_section(quick: bool = True) -> dict:
    """Execute or load Part E (laser wiggler resonance and Fermi brightness B_F scaling)."""
    json_path = h.DATA_DIR / "task_part_e_wiggler_fel.json"
    if json_path.exists():
        summary = json.loads(json_path.read_text())
    else:
        summary = h.run_task_part_e_wiggler_fel(quick=quick)
    print(
        f"[Part E] 10 keV X-rays (lambda_x = {summary['lambda_x_angstrom']:.2f} A): "
        f"E_e = {summary['e_req_mev_laser_0p4um']:.1f} MeV (lambda_L=0.4 um) to "
        f"{summary['e_req_mev_laser_1p0um']:.1f} MeV (lambda_L=1.0 um) vs "
        f"{summary['e_req_gev_static_3cm']:.1f} GeV (3 cm static undulator) | "
        f"B_F = {summary['b_f_zolotorev_2002']:.2e} -> M_A = {summary['m_a_periods_zolotorev_2002']:.0f} periods"
    )
    return summary


# %% [markdown]
# ---
#
# ## Quantitative Summary vs Literature
#
# | Part & Physical Quantity | Analytical / Paper Reference | Wakis / Numerical Result | Source in Literature |
# | :--- | :--- | :--- | :--- |
# | **Part A**: Intra-pulse peak $\gamma_{\max}$ ($a_0 = 2.5$) | $\gamma_{\max} = 1 + a_0^2/2 = 4.1250$ | $4.1250$ (Boris & Vay) | Zolotorev (2002), p. 447 |
# | **Part A**: Post-pulse net gain $\Delta W(+\infty)$ | $0\;\text{eV}$ (Lawson–Woodward) | $< 10^{-8}\;\text{eV}$ | Zolotorev (2002), p. 446 |
# | **Part B**: Unbounded $\text{TEM}_{00}$ $\Delta W_{\sin} / (e\sqrt{Z_0 P})$ | $0.0000$ | $< 0.005$ (3D FIT + tails) | Plettner et al. (2005), Fig. 3 (tape out) |
# | **Part C**: Optimal crossing angle $s_{\text{opt}} = \theta_c Z_R / w_0$ | $1.1209$ ($\Delta W = 0.7201\,e\sqrt{Z_0 P}$) | $0.713\,e\sqrt{Z_0 P}$ ($1.0\%$ error) | Huang, Stupakov & Zolotorev (2004) |
# | **Part C**: Polarization scaling $\Delta W(\phi_{\text{pol}})$ | $\Delta W(0)\cos\phi_{\text{pol}}$ | Matches $\cos\phi_{\text{pol}}$ to $< 1\%$ | Plettner et al. (2005), Fig. 5(b) |
# | **Part D**: Optimal Rayleigh length $Z_R / Z_{R,\text{opt}}$ | $1.00$ ($Z_{R,\text{opt}} = a_0^2 c\tau / 2$) | Peak at $Z_R / Z_{R,\text{opt}} = 1.00$ | Zolotorev (2002), p. 447, Fig. 1 |
# | **Part D**: Axial energy $\gamma(z = 6 Z_R)$ ($a_0 = 7.4$) | $\approx 27.8$ (read from Fig. 2, approx.) | $27.85$ ($\gamma_{\text{opt}} = 28.38$) | Zolotorev (2002), p. 447, Fig. 2 |
# | **Part D**: Crater rim radius $r_{\text{rim}} / \sigma$ ($\alpha = 1.25$) | $\sqrt{2(1 - 1/\alpha)} = 0.632$ | $0.632$ (focuses $\|x\| \lesssim 0.2\;\mu\text{m}$) | Zolotorev (2002), p. 447, Figs. 3–4 |
# | **Part D**: Compressed RMS bunch duration $\sigma_t$ | $\approx 50\;\text{as}$ ($\sigma_z \approx 15\;\text{nm}$) | $51.4\;\text{as}$ ($\sigma_z = 15.4\;\text{nm}$) | Zolotorev (2002), p. 448, Fig. 5 |
# | **Part E**: $E_e$ for $10\;\text{keV}$ X-rays ($\lambda_L = 0.4$–$1\;\mu\text{m}$) | $15$–$28\;\text{MeV}$ (vs $15\;\text{GeV}$ static) | $17.7$–$28.0\;\text{MeV}$ ($a = 1$) | Zolotorev (2002), p. 446 |
# | **Part E**: Fermi brightness $B_F$ ($N = 10^5$) | $\approx 10^{-6}$ ($M_A \sim 10^2$ periods) | $0.96\times 10^{-6}$ ($M_A = 119$) | Zolotorev (2002), pp. 446–448 |
#
# ---
#
# ## Grid Resolution Convergence Study
#
# To verify that our 3D Wakis FIT focused-beam fields and boundary-terminated energy integrals converge cleanly with mesh refinement, we run a 3-level resolution study spanning $\lambda_L / \Delta z \in \{12, 16, 20\}\;\text{cells}/\lambda_L$ (reaching the $\ge 20\;\text{cells}/\lambda_L$ rule at the finest level).
#
# ![3D FIT resolution convergence](figures/08_fit_resolution_convergence.png)

# %%
def run_convergence_section(quick: bool = True) -> dict:
    """Execute or load the 3-level 3D FIT grid resolution convergence study."""
    json_path = h.DATA_DIR / "task_convergence.json"
    if json_path.exists():
        summary = json.loads(json_path.read_text())
    else:
        summary = h.run_task_convergence(quick=quick)
    for lvl, ppw, ex_e, ez_e, dw_e in zip(
        summary["levels"],
        summary["ppw_z"],
        summary["ex_waist_l2_err"],
        summary["ez_waist_l2_err"],
        summary["dw_screen_rel_err"],
    ):
        print(
            f"[Convergence · {lvl:6s}] ppw_z = {ppw:4.0f}: "
            f"Ex L2 err = {ex_e*100:5.2f}%, Ez L2 err = {ez_e*100:5.2f}%, "
            f"screen dW rel err = {dw_e*100:5.2f}%"
        )
    return summary


# %% [markdown]
# ---
#
# ## Limits & Failure Modes
#
# 1. **Foil damage threshold and multiple Coulomb scattering in boundary-terminated acceleration (Part C)**:
#    In Plettner et al. (*PRL* **95**, 134801, 2005), the gold-coated Kapton tape must be translated continuously after each laser shot once the fluence approaches the damage threshold ($\sim 0.1$–$0.5\;\text{J/cm}^2$ for sub-picosecond pulses), and multiple Coulomb scattering in the $8\;\mu\text{m}$ foil degrades the transverse emittance of multi-MeV beams unless sub-micron membranes or dielectric grazing-incidence structures are used.
# 2. **Breakdown of the paraxial approximation at tight focusing ($k_0 w_0 \lesssim 4$)**:
#    The first-order Lax–Louisell–McKnight longitudinal field $\mathcal{E}_z = (i/k_0)\partial_x\mathcal{E}_x$ neglects $\mathcal{O}(\theta_d^3)$ terms in the series expansion in $\theta_d = 1/(k_0 w_0)$. When a beam is focused tighter than $w_0 \lesssim 0.6\lambda_L$ ($\theta_d \gtrsim 0.25\;\text{rad}$), higher-order non-paraxial corrections must be retained in analytical formulas, whereas the full-wave 3D FIT Maxwell solver (`SolverFIT3D`) automatically satisfies exact $\nabla\cdot\mathbf{E} = 0$ on the staggered Yee grid.
# 3. **Finite-domain truncation vs true Lawson–Woodward cancellation**:
#    Because the synchronous integrand $E_\parallel(z) \propto (1 + i z/Z_R)^{-2}\exp(-i s^2\zeta/(1+i\zeta))$ decays algebraically as $(z/Z_R)^{-2}$ for $|z| \gg Z_R$, truncating the line integral at a finite box boundary $z = \pm z_{\max}$ without accounting for the asymptotic antiderivative tail leaves an artificial $\mathcal{O}(Z_R / z_{\max})$ boundary residual in the cosine carrier phase even when no physical screen is present.
# 4. **Space-charge debunching of attosecond bunches at low relativistic $\gamma$ (Part D & Part E)**:
#    At $N = 10^5$ electrons compressed into $\sigma_z \approx 15\;\text{nm}$ ($\sigma_t \approx 50\;\text{as}$) and $\sigma_\perp \approx 0.1\;\mu\text{m}$, the longitudinal space-charge field is suppressed by $1/\gamma^2 \approx 1.2\times 10^{-3}$ at $\gamma \approx 28$, allowing the attosecond bunch to survive over the short millimeter-scale FEL gain length $L_G = M_A \lambda_W \sim 50\;\mu\text{m}$, but causing rapid longitudinal debunching if the bunch is allowed to drift tens of centimeters before entering the laser wiggler (Filippetto, Musumeci, Zolotorev & Stupakov, *PRSTAB* **17**, 024201, 2014).
#
# ---
#
# ## References
#
# 1. Z. Huang, G. Stupakov, and M. Zolotorev, "Calculation and optimization of laser acceleration in vacuum," *Phys. Rev. ST Accel. Beams* **7**, 011302 (2004), `doi:10.1103/PhysRevSTAB.7.011302`.
# 2. G. V. Stupakov and M. S. Zolotorev, "Ponderomotive laser acceleration and focusing in vacuum for generation of attosecond electron bunches," *Phys. Rev. Lett.* **86**, 5274–5277 (2001), `doi:10.1103/PhysRevLett.86.5274`.
# 3. M. Zolotorev, "Laser driven attosecond SASE X-ray FEL," *Nucl. Instrum. Methods Phys. Res. A* **483**, 445–448 (2002), `doi:10.1016/S0168-9002(02)00359-5`.
# 4. T. Plettner, R. L. Byer, E. Colby, B. Cowan, C. M. S. Sears, J. E. Spencer, and R. H. Siemann, "Visible-laser acceleration of relativistic electrons in a semi-infinite vacuum," *Phys. Rev. Lett.* **95**, 134801 (2005), `doi:10.1103/PhysRevLett.95.134801`.
# 5. A. A. Zholents and M. S. Zolotorev, "Attosecond x-ray pulses produced by ultra short transverse slicing via laser electron interaction," *New J. Phys.* **10**, 025005 (2008), `doi:10.1088/1367-2630/10/2/025005`.
# 6. D. Filippetto, P. Musumeci, M. Zolotorev, and G. Stupakov, "Maximum current density and beam brightness achievable by laser-driven electron sources," *Phys. Rev. ST Accel. Beams* **17**, 024201 (2014), `doi:10.1103/PhysRevSTAB.17.024201`.
# 7. J.-L. Vay, "Simulation of beams or plasmas crossing at relativistic velocity," *Phys. Plasmas* **15**, 056701 (2008), `doi:10.1063/1.2837054`.

# %%
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tutorial 8.3: Laser acceleration of electrons in vacuum (after Zolotorev and colleagues)."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run fast validation (< 1 min).")
    mode.add_argument("--full", action="store_true", help="Run full-resolution tutorial tasks.")
    parser.add_argument(
        "--task",
        choices=TASKS,
        default=None,
        help="Run only a single task.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tutorial tasks and exit.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render figures and animations from existing data/ without re-running simulations.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output directory override for figures.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_tasks:
        for t_name in TASKS:
            print(t_name)
        return 0

    if args.out is not None:
        h.FIG_DIR = args.out.resolve()

    quick = bool(args.quick or not args.full)

    if not args.plot:
        selected = [args.task] if args.task is not None else list(TASKS)
        for name in selected:
            fn, _ = h.TASKS[name]
            res = fn(quick=quick)
            print(f"[done] task={name}: {json.dumps(res)}")

    if args.plot or args.task is None:
        needed = [h.DATA_DIR / f"task_{name}.npz" for name in TASKS]
        if all(p.exists() for p in needed):
            paths = h.render_all_figures()
            for p in paths:
                size_kb = p.stat().st_size / 1024.0
                print(f"[figure] {p.name} ({size_kb:.1f} KB)")

    return 0


if __name__ == "__main__":
    if any("ipykernel" in arg for arg in sys.argv):
        run_part_a_section(quick=True)
        run_part_b_c_section(quick=True)
        run_part_d_section(quick=True)
        run_part_e_section(quick=True)
        run_convergence_section(quick=True)
        h.render_all_figures()
    else:
        raise SystemExit(main())
