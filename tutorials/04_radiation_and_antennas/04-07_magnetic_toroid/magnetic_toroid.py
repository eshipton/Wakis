# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/04_radiation_and_antennas/04-07_magnetic_toroid/magnetic_toroid.ipynb)
#
# # Tutorial 4.7 · The magnetic toroid: a closed magnetization loop radiates like an axial electric dipole
# <h3 align="center">Closed azimuthal magnetization, Dubovik–Tugushev toroidal moments, exact vector spherical harmonic decomposition, and symmetry-breaking failure tests in 3D FIT</h3>
#
# ---
#
# ### Abstract
#
# **An electrically small current loop radiates as a magnetic dipole $\mathbf{m}$ ($\mathrm{TE}_{10}$ spherical wave, horizontally polarized $E_\phi \propto \sin\theta$), whereas a short linear wire radiates as an electric dipole $\mathbf{p}$ ($\mathrm{TM}_{10}$ spherical wave, vertically polarized $E_\theta \propto \sin\theta$). What happens when we arrange magnetic dipoles into a closed azimuthal ring—or wind $N$ in-phase poloidal turns around a permeable toroidal core (`wakis.dispersive.DispersiveMaterial`) so that the internal magnetization $\mathbf{M}(\mathbf{r}) = M_\phi(\rho, z)\hat{\boldsymbol{\phi}}$ forms a closed loop without magnetic poles? Because $\oint \hat{\boldsymbol{\phi}}\,\mathrm{d}\phi = \mathbf{0}$, the net magnetic dipole moment vanishes identically ($\mathbf{m} = \int \mathbf{M}\,\mathrm{d}^3r = \mathbf{0}$). Even more strikingly, for an axisymmetric azimuthal magnetization loop every magnetic-type ($\mathrm{TE}$) spherical multipole $b_{lm}$ vanishes to all orders in $k_0 R$. Instead, the poloidal bound current $\mathbf{J}_b = \nabla \times \mathbf{M}$ and winding current $\mathbf{J}_f$ carry an axial toroidal dipole moment $\mathbf{T} = \frac{1}{2}\int (\mathbf{r} \times \mathbf{M})\,\mathrm{d}^3r$ that radiates the exact $\mathrm{TM}_{10}$ far-field pattern of an axial electric dipole with effective current moment $\mathbf{I\ell}_{\mathrm{eff}} = +\mathrm{i}\omega\mathbf{p}_{\mathrm{eff}} = +k_0^2\mathbf{T}_{\mathrm{tot}}$ ($\mathbf{p}_{\mathrm{eff}} = -(\mathrm{i} k_0 / c_0)\mathbf{T}_{\mathrm{tot}}$ in Wakis's $\mathrm{e}^{+\mathrm{i}\omega t}$ convention, or $+(\mathrm{i} k_0 / c_0)\mathbf{T}_{\mathrm{tot}}$ in the physics $\mathrm{e}^{-\mathrm{i}\omega t}$ convention). In this tutorial we derive these results from first principles, verify the Dubovik–Tugushev integral identity on the 3D Finite Integration Technique (FIT) staggered grid, project the simulated `Near2FarMonitor` fields onto vector spherical harmonics ($l \le 5$), test discrete $C_{4v}$ four-rod squares and five distinct loop-breaking failure modes, map where the dipole equivalence breaks down in the reactive near field and near internal core resonances, and benchmark radiation efficiency and quality factor $Q$ against the Chu spherical bound.**
#
# ---
#
# ### Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. **Contrast** the vector spherical harmonic ($\mathrm{VSH}$) signatures of an axial electric dipole ($\mathbf{p} = p_z\hat{\mathbf{z}}$, pure $\mathrm{TM}_{10}$ coefficient $a_{10}$, $E_\theta \propto \sin\theta$) and an axial magnetic loop ($\mathbf{m} = m_z\hat{\mathbf{z}}$, pure $\mathrm{TE}_{10}$ coefficient $b_{10}$, $E_\phi \propto \sin\theta$, radiation resistance $R_{\mathrm{rad}} \propto (k_0 a)^4$), and compare an ideal `MagneticCurrentSource` loop against a thin PEC wire loop driven by a `wakis.lumped.LumpedPort`.
# 2. **Prove** from Maxwell's equations that an axisymmetric azimuthal magnetization $\mathbf{M}(\rho, z) = M_\phi(\rho, z)\hat{\boldsymbol{\phi}}$ has zero magnetic charge ($\nabla\cdot\mathbf{M} = 0$) and zero radial curl component ($\mathbf{r}\cdot(\nabla\times\mathbf{M}) = 0$), which forces every magnetic-type ($\mathrm{TE}$) multipole coefficient $b_{lm}$ to vanish identically for all $(l, m)$ while mid-plane reflection symmetry ($z \to -z$) restricts the radiated field to odd axial electric multipoles ($a_{10}, a_{30}, a_{50}, \dots$).
# 3. **Derive and verify** the **Dubovik–Tugushev identity** $\mathbf{T}_{\mathrm{DT}}(\mathbf{J}_b) = \frac{1}{10}\int [\mathbf{r}(\mathbf{r}\cdot\mathbf{J}_b) - 2r^2\mathbf{J}_b]\,\mathrm{d}^3r = \frac{1}{2}\int (\mathbf{r}\times\mathbf{M})\,\mathrm{d}^3r$ on the 3D FIT Yee/staggered dual grid using the discrete curl operator $\mathbf{J}_b = \nabla_{\mathrm{FIT}}\times\mathbf{M}$, together with Marinov et al.'s quasi-static toroidal moment formula $T_z \approx \frac{1}{2}\pi N I \mu_{\mathrm{eff}} r^2 R$ [1, 2].
# 4. **Predict** the radiated power $P_{\mathrm{rad}}$ of a dispersive Lorentz magnetic torus (`DispersiveMaterial`) from its simulated toroidal moment $T_{\mathrm{tot}}$ with **zero free parameters**, verify the $\omega^6$ frequency scaling at fixed $T$ (versus $\omega^4$ for a fixed electric dipole $p$) and the $(k_0 R)^2$ scaling of the electric octupole ratio $|a_{30}|/|a_{10}|$.
# 5. **Quantify** how replacing a continuous torus with a $C_4$-symmetric **square of four permeable rods** still guarantees $\mathbf{m} = \mathbf{0}$ and $b_{1m} = b_{2m} = b_{3m} = 0$ while introducing small $m = \pm 4$ azimuthal multipole residuals that grow with electrical size $k_0 R_{\mathrm{sq}}$.
# 6. **Diagnose** five symmetry- and topology-breaking counter-cases—a $60^\circ$ air gap in the core, 7 of 8 active feeds [3], the two-feed antiphase "loop mode" (Mode 1), removing one rod from the 4-rod square, and a helical winding with one parasitic azimuthal turn (plus Marinov's reverse compensating loop [1])—by tracking the magnetic-to-electric dipole ratio $|b_1|/|a_1|$.
# 7. **Identify** where the toroidal–electric-dipole equivalence ends (reactive near-field topology $|\mathbf{E}|, |\mathbf{H}|$ for $r \sim R$, finite winding number $N$, electrical size $2\pi R \sim \lambda_0$, and internal $\mathrm{TE}_{01\delta}$-like core resonances when $2r \gtrsim \lambda_{\mathrm{core}}/2$) and evaluate the radiation efficiency $\eta_{\mathrm{rad}}$ and quality factor $Q_{\mathrm{rad}}$ against the Chu–Harrington spherical bound [4, 5].
#
# ---
#
# ## Theory & First-Principles Derivations
#
# ### 1. Baseline Dipoles: Electric Dipole $\mathbf{p}$ vs Magnetic Loop $\mathbf{m}$
#
# Throughout this tutorial we work in the harmonic time convention $\mathbf{E}(\mathbf{r}, t) = \operatorname{Re}\{\mathbf{E}(\mathbf{r})\,\mathrm{e}^{+\mathrm{i}\omega t}\}$ ($\partial_t \to +\mathrm{i}\omega$, outgoing spherical wave $\mathrm{e}^{-\mathrm{i}k_0 r}/r$, passive lossy permeability $\mu_r(\omega) = \mu'(\omega) - \mathrm{i}\mu''(\omega)$ with $\mu'' > 0$) used by Wakis's frequency-domain monitors (`DFTMonitor`, `Near2FarMonitor` at `wakis/near2far.py:56`, and `FluxMonitor`), with free-space wavenumber $k_0 = \omega / c_0 = 2\pi / \lambda_0$ and wave impedance $\eta_0 = \sqrt{\mu_0 / \varepsilon_0}$. (Where physics-literature formulas in the $\mathrm{e}^{-\mathrm{i}\omega t}$ convention are referenced for comparison, we label them explicitly.)
#
# #### 1.1 Short axial electric dipole ($\mathrm{TM}_{10}$)
#
# An electrically short linear current element $\mathbf{J}(\mathbf{r}) = +\mathrm{i}\omega p_z\,\delta^{(3)}(\mathbf{r})\hat{\mathbf{z}}$ along the $z$-axis has electric dipole moment $\mathbf{p} = \frac{1}{\mathrm{i}\omega}\int \mathbf{J}\,\mathrm{d}^3r = p_z\hat{\mathbf{z}}$. Its exact vector potential $\mathbf{A}(\mathbf{r}) = +\mathrm{i}\omega\mu_0 p_z \frac{\mathrm{e}^{-\mathrm{i}k_0 r}}{4\pi r}\hat{\mathbf{z}}$ produces the full-space electromagnetic field (**derived here** from $\mathbf{H} = \frac{1}{\mu_0}\nabla\times\mathbf{A}$ and $\mathbf{E} = \frac{1}{\mathrm{i}\omega\varepsilon_0}\nabla\times\mathbf{H}$):
#
# $$\mathbf{H}_{\mathrm{dip}}(\mathbf{r}) = \frac{c_0 k_0^3 p_z}{4\pi}\sin\theta\,\left(-\frac{1}{k_0 r} + \frac{\mathrm{i}}{(k_0 r)^2}\right)\mathrm{e}^{-\mathrm{i}k_0 r}\,\hat{\boldsymbol{\phi}},$$
#
# $$E_r(\mathbf{r}) = \frac{p_z}{2\pi\varepsilon_0}\cos\theta\,\left(\frac{1}{r^3} + \frac{\mathrm{i}k_0}{r^2}\right)\mathrm{e}^{-\mathrm{i}k_0 r}, \qquad E_\theta(\mathbf{r}) = \frac{p_z}{4\pi\varepsilon_0}\sin\theta\,\left(\frac{1}{r^3} + \frac{\mathrm{i}k_0}{r^2} - \frac{k_0^2}{r}\right)\mathrm{e}^{-\mathrm{i}k_0 r}.$$
#
# In the radiation zone ($k_0 r \gg 1$), keeping the $r^{-1}$ terms gives a purely polar-polarized ($\mathrm{TM}_{10}$) spherical wave and time-averaged radiated power:
#
# $$\mathbf{E}_{\mathrm{ff}}^{(p)}(\mathbf{r}) = -\frac{\mu_0\omega^2 p_z}{4\pi}\,\sin\theta\,\frac{\mathrm{e}^{-\mathrm{i}k_0 r}}{r}\,\hat{\boldsymbol{\theta}}, \qquad P_{\mathrm{rad}}^{(p)} = \frac{\mu_0 \omega^4 |p_z|^2}{12\pi c_0} = \frac{\eta_0 k_0^4 |c_0 p_z|^2}{12\pi} \propto \omega^4 |p_z|^2.$$
#
# #### 1.2 Small axial magnetic loop ($\mathrm{TE}_{10}$) and the $(k_0 a)^4$ law
#
# A small circular current loop of radius $a$ in the $xy$-plane carrying uniform current $I_0\hat{\boldsymbol{\phi}}$ (or an equivalent azimuthal magnetic current ring) has axial magnetic dipole moment $\mathbf{m} = \frac{1}{2}\int (\mathbf{r}'\times\mathbf{J}(\mathbf{r}'))\,\mathrm{d}^3r' = I_0 (\pi a^2)\hat{\mathbf{z}} = m_z\hat{\mathbf{z}}$. Expanding the retarded vector potential $\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int \mathbf{J}(\mathbf{r}')\frac{\mathrm{e}^{-\mathrm{i}k_0|\mathbf{r}-\mathbf{r}'|}}{|\mathbf{r}-\mathbf{r}'|}\,\mathrm{d}^3r'$ for $k_0 a \ll 1$ and $r \gg a$ using $|\mathbf{r}-\mathbf{r}'| \approx r - \hat{\mathbf{r}}\cdot\mathbf{r}'$ and $\int \mathbf{J}(\mathbf{r}')(\hat{\mathbf{r}}\cdot\mathbf{r}')\,\mathrm{d}^3r' = \mathbf{m}\times\hat{\mathbf{r}} = m_z\sin\theta\,\hat{\boldsymbol{\phi}}$ gives (**derived here** in $\mathrm{e}^{+\mathrm{i}\omega t}$):
#
# $$\mathbf{A}(\mathbf{r}) = \frac{\mu_0 m_z}{4\pi}\sin\theta\left(\frac{\mathrm{i}k_0}{r} + \frac{1}{r^2}\right)\mathrm{e}^{-\mathrm{i}k_0 r}\,\hat{\boldsymbol{\phi}}.$$
#
# Since $\nabla\cdot\mathbf{J} = 0$ ($\Phi_e = 0$), the radiation-zone ($k_0 r \gg 1$) electric field is $\mathbf{E}_{\mathrm{ff}}^{(m)}(\mathbf{r}) = -\mathrm{i}\omega\mathbf{A}_{\mathrm{ff}}(\mathbf{r}) = (-\mathrm{i}\omega)(+\mathrm{i}k_0\mu_0 m_z \sin\theta\,\mathrm{e}^{-\mathrm{i}k_0 r}/(4\pi r))\hat{\boldsymbol{\phi}}$, which is purely azimuthally polarized ($\mathrm{TE}_{10}$) with a positive prefactor (matching Balanis, *Antenna Theory*, Eq. 5-27a in $\mathrm{e}^{+\mathrm{i}\omega t}$):
#
# $$\mathbf{E}_{\mathrm{ff}}^{(m)}(\mathbf{r}) = +\frac{\eta_0 k_0^2 m_z}{4\pi}\,\sin\theta\,\frac{\mathrm{e}^{-\mathrm{i}k_0 r}}{r}\,\hat{\boldsymbol{\phi}} = +\frac{\mu_0\omega^2 m_z}{4\pi c_0}\,\sin\theta\,\frac{\mathrm{e}^{-\mathrm{i}k_0 r}}{r}\,\hat{\boldsymbol{\phi}}, \qquad P_{\mathrm{rad}}^{(m)} = \frac{\mu_0 \omega^4 |m_z|^2}{12\pi c_0^3} = \frac{1}{2}|I_0|^2 R_{\mathrm{rad}}^{(\mathrm{loop})},$$
#
# where substituting $|m_z| = \pi a^2 |I_0|$ yields the classical **$(k_0 a)^4$ radiation resistance** of a small loop (**derived here**):
#
# $$\boxed{R_{\mathrm{rad}}^{(\mathrm{loop})} = \frac{\eta_0 \pi}{6}\,(k_0 a)^4 \approx 20\pi^2 (k_0 a)^4\;[\Omega].}$$
#
# If the loop encloses a permeable magnetic core of effective relative permeability $\mu_{\mathrm{eff}}$, the total magnetic moment increases to $\mathbf{m}_{\mathrm{tot}} = \mu_{\mathrm{eff}}\mathbf{m}_{\mathrm{free}}$, boosting the radiated power by $|\mu_{\mathrm{eff}}|^2$ while preserving the $\mathrm{TE}_{10}$ ($E_\phi$) polarization.
#
# ---
#
# ### 2. The Closed Azimuthal Magnetization Loop: Why All Magnetic ($\mathrm{TE}$) Multipoles Vanish
#
# Now suppose we bend the magnetic core into a **closed torus** centered at the origin with major radius $R$ and minor cross-sectional radius $r < R$, and drive it with $N$ identical poloidal windings (or an axisymmetric poloidal surface current sheet $\mathbf{K}_f$). Inside the core, the magnetic field and magnetization are purely azimuthal:
#
# $$\mathbf{M}(\mathbf{r}) = M_\phi(\rho, z)\,\hat{\boldsymbol{\phi}}, \qquad \hat{\boldsymbol{\phi}} = -\sin\phi\,\hat{\mathbf{x}} + \cos\phi\,\hat{\mathbf{y}},$$
#
# where $(\rho, \phi, z)$ are cylindrical coordinates.
#
# #### 2.1 Vanishing of the ordinary magnetic dipole moment $\mathbf{m}$
#
# Integrating $\mathbf{M}$ over the volume of the torus immediately gives (**derived here**):
#
# $$\mathbf{m} = \int_V \mathbf{M}(\mathbf{r})\,\mathrm{d}^3r = \int \mathrm{d}z \int \rho\,\mathrm{d}\rho\,M_\phi(\rho, z) \underbrace{\int_0^{2\pi} \hat{\boldsymbol{\phi}}(\phi)\,\mathrm{d}\phi}_{\mathbf{0}} = \mathbf{0}.$$
#
# Because the magnetization lines close on themselves inside the torus without beginning or ending on any surface, there are no magnetic poles ($\rho_m = -\nabla\cdot\mathbf{M} = -\frac{1}{\rho}\frac{\partial M_\phi}{\partial\phi} = 0$), and the vector sum of $\mathbf{M}$ around the ring cancels identically.
#
# #### 2.2 Proof that every magnetic-type ($\mathrm{TE}$) spherical multipole $b_{lm}$ vanishes (**derived here**)
#
# Does a closed axisymmetric loop of $M_\phi(\rho, z)$ radiate a magnetic quadrupole ($b_{2m}$), magnetic octupole ($b_{3m}$), or higher $\mathrm{TE}_l$ multipole? **No—every magnetic multipole $b_{lm}$ vanishes identically for all $(l, m)$.**
#
# To prove this, write Maxwell's curl equations in terms of $\mathbf{E}$ and $\mathbf{B} = \mu_0(\mathbf{H} + \mathbf{M})$ in the $\mathrm{e}^{+\mathrm{i}\omega t}$ convention:
#
# $$\nabla\times\mathbf{E} = -\mathrm{i}\omega\mathbf{B}, \qquad \nabla\times\mathbf{B} = +\frac{\mathrm{i}\omega}{c_0^2}\mathbf{E} + \mu_0\left(\mathbf{J}_f + \nabla\times\mathbf{M}\right).$$
#
# In the exact multipole formalism (Jackson, *Classical Electrodynamics*, Ch. 9 [6]), the magnetic ($\mathrm{TE}$) multipole coefficients $b_{lm}$ of a localized source distribution are driven exclusively by the radial component of the curl of the total effective current, $\mathbf{r}\cdot(\nabla\times\mathbf{J}_{\mathrm{tot}}) = -\nabla\cdot(\mathbf{r}\times\mathbf{J}_{\mathrm{tot}})$, where $\mathbf{J}_{\mathrm{tot}} = \mathbf{J}_f + \nabla\times\mathbf{M}$. Equivalently, in the magnetic-current formulation where $\mathbf{M}$ is represented by the azimuthal magnetic current density $\mathbf{J}_m = +\mathrm{i}\omega\mu_0\mathbf{M} = +\mathrm{i}\omega\mu_0 M_\phi(\rho, z)\hat{\boldsymbol{\phi}}$, the radial magnetic field $B_r = \frac{1}{r}\mathbf{r}\cdot\mathbf{B}$—which generates all $\mathrm{TE}$ multipoles $b_{lm}$—satisfies the inhomogeneous Helmholtz equation sourced by $(\nabla\times\mathbf{J}_m)_r$ and $\nabla\cdot\mathbf{J}_m$. For $\mathbf{M} = M_\phi(\rho, z)\hat{\boldsymbol{\phi}}$ with $\partial_\phi M_\phi = 0$:
#
# 1. **Zero magnetic charge density:**
#    $$\nabla\cdot\mathbf{M} = \frac{1}{\rho}\frac{\partial M_\phi}{\partial\phi} = 0.$$
# 2. **Poloidal bound electric current $\mathbf{J}_b = \nabla\times\mathbf{M}$:**
#    $$\mathbf{J}_b = \nabla\times\left(M_\phi(\rho, z)\hat{\boldsymbol{\phi}}\right) = -\frac{\partial M_\phi}{\partial z}\,\hat{\boldsymbol{\rho}} + \frac{1}{\rho}\frac{\partial(\rho M_\phi)}{\partial\rho}\,\hat{\mathbf{z}}.$$
#    Because $\mathbf{J}_b$ (and the poloidal winding current $\mathbf{J}_f$) lies strictly in the meridional $(\hat{\boldsymbol{\rho}}, \hat{\mathbf{z}})$ plane with no $\phi$-dependence, the cross product with the position vector $\mathbf{r} = \rho\hat{\boldsymbol{\rho}} + z\hat{\mathbf{z}}$ (using $\hat{\boldsymbol{\rho}}\times\hat{\mathbf{z}} = -\hat{\boldsymbol{\phi}}$) is purely azimuthal:
#    $$\mathbf{r}\times\mathbf{J}_{\mathrm{tot}} = \left(z J_{\mathrm{tot},\rho} - \rho J_{\mathrm{tot},z}\right)\hat{\boldsymbol{\phi}},$$
#    whose divergence—the exact source of all $\mathrm{TE}$ spherical multipoles $b_{lm}$ [6]—vanishes identically everywhere:
#    $$\nabla\cdot(\mathbf{r}\times\mathbf{J}_{\mathrm{tot}}) = \frac{1}{\rho}\frac{\partial}{\partial\phi}\left(z J_{\mathrm{tot},\rho} - \rho J_{\mathrm{tot},z}\right) = 0 \implies \boxed{b_{lm} = 0 \quad \forall\, l \ge 1,\; -l \le m \le l.}$$
#
# Furthermore:
# - **Axisymmetry ($\partial_\phi = 0$)** forces $a_{lm} = 0$ for all $m \ne 0$.
# - **Mid-plane reflection symmetry ($z \to -z$):** When the torus cross-section is symmetric about $z = 0$, $M_\phi(\rho, -z) = +M_\phi(\rho, z)$, so $J_{b,z}(\rho, -z) = +J_{b,z}(\rho, z)$ (even in $z$) and $J_{b,\rho}(\rho, -z) = -J_{b,\rho}(\rho, z)$ (odd in $z$). Under $z \to -z$ ($\theta \to \pi - \theta$), the radiated field has $E_\theta(\pi - \theta) = +E_\theta(\theta)$, which eliminates all even-$l$ electric multipoles ($a_{20} = a_{40} = \dots = 0$).
#
# Thus, **an axisymmetric, mid-plane-symmetric magnetic toroid can radiate only odd-$l$, $m=0$ electric ($\mathrm{TM}$) multipoles: $a_{10}$ (electric dipole), $a_{30}$ (electric octupole), $a_{50}$, $\dots$!**
#
# ---
#
# ### 3. Toroidal Dipole Moment $\mathbf{T}$, the Dubovik–Tugushev Identity, and $\mathrm{TM}_{10}$ Radiation
#
# #### 3.1 Equivalence of the magnetization and current definitions of $\mathbf{T}$ (**derived here**)
#
# In the multipole expansion of a poloidal current distribution $\mathbf{J}(\mathbf{r})$ with $\int \mathbf{J}\,\mathrm{d}^3r = \mathbf{0}$ and $\int (\mathbf{r}\times\mathbf{J})\,\mathrm{d}^3r = \mathbf{0}$, the leading moment is the **toroidal dipole moment** introduced by Zel'dovich (1958) [7] and formalized in electrodynamics by Dubovik & Tugushev (1990) [2]:
#
# $$\mathbf{T}_{\mathrm{DT}}(\mathbf{J}) \equiv \frac{1}{10}\int_V \left[\mathbf{r}(\mathbf{r}\cdot\mathbf{J}) - 2 r^2 \mathbf{J}\right]\mathrm{d}^3r.$$
#
# Let us prove (**derived here**) that for the bound current $\mathbf{J}_b = \nabla\times\mathbf{M}$ of any localized magnetization $\mathbf{M}(\mathbf{r})$ vanishing outside $V$, $\mathbf{T}_{\mathrm{DT}}(\nabla\times\mathbf{M})$ reduces identically to the vortex moment of $\mathbf{M}$:
#
# $$\boxed{\mathbf{T}_{\mathrm{DT}}(\nabla\times\mathbf{M}) = \frac{1}{2}\int_V (\mathbf{r}\times\mathbf{M})\,\mathrm{d}^3r \equiv \mathbf{T}_{\mathrm{mag}}(\mathbf{M}).}$$
#
# **Proof (index notation with Einstein summation):** Write the $i$-th Cartesian component of $\mathbf{T}_{\mathrm{DT}}(\nabla\times\mathbf{M})$ using $(\nabla\times\mathbf{M})_k = \varepsilon_{klm}\partial_l M_m$:
#
# $$10\,T_{\mathrm{DT},i} = \int_V \left[x_i x_k \varepsilon_{klm}\partial_l M_m - 2 x_n x_n \varepsilon_{ilm}\partial_l M_m\right]\mathrm{d}^3r.$$
#
# Integrating both terms by parts and dropping the boundary surface integrals (since $\mathbf{M} = \mathbf{0}$ outside $V$):
# - For the first term:
#   $$\int_V x_i x_k \varepsilon_{klm}\partial_l M_m\,\mathrm{d}^3r = -\int_V \varepsilon_{klm} M_m \partial_l(x_i x_k)\,\mathrm{d}^3r = -\int_V \varepsilon_{klm} M_m \left(\delta_{il} x_k + x_i \delta_{kl}\right)\mathrm{d}^3r = -\int_V \varepsilon_{kim} x_k M_m\,\mathrm{d}^3r = \int_V (\mathbf{r}\times\mathbf{M})_i\,\mathrm{d}^3r,$$
#   since $\varepsilon_{klm}\delta_{kl} = \varepsilon_{kkm} = 0$ and $-\varepsilon_{kim} x_k M_m = +\varepsilon_{ikm} x_k M_m = (\mathbf{r}\times\mathbf{M})_i$.
# - For the second term:
#   $$-2\int_V x_n x_n \varepsilon_{ilm}\partial_l M_m\,\mathrm{d}^3r = +2\int_V \varepsilon_{ilm} M_m \partial_l(x_n x_n)\,\mathrm{d}^3r = 4\int_V \varepsilon_{ilm} x_l M_m\,\mathrm{d}^3r = 4\int_V (\mathbf{r}\times\mathbf{M})_i\,\mathrm{d}^3r.$$
#
# Adding the two contributions gives $10\,T_{\mathrm{DT},i} = (1 + 4)\int_V (\mathbf{r}\times\mathbf{M})_i\,\mathrm{d}^3r = 5\int_V (\mathbf{r}\times\mathbf{M})_i\,\mathrm{d}^3r$, i.e. $\mathbf{T}_{\mathrm{DT}}(\nabla\times\mathbf{M}) = \frac{1}{2}\int_V (\mathbf{r}\times\mathbf{M})\,\mathrm{d}^3r$. $\blacksquare$
#
# Notice that if the coefficient $-2 r^2 \mathbf{J}_b$ in the Dubovik–Tugushev integrand is mutated to $-1 r^2 \mathbf{J}_b$, the second term contributes $+2$ instead of $+4$, giving $(1+2)/10 = 3/10$ instead of $5/10 = 1/2$—an exact **$40.0\%$ relative error** (`|0.3 - 0.5| / 0.5 = 0.40`). Note that this $40.0\%$ ratio is an algebraic property of the discrete FIT curl-dual and volume-quadrature operators that holds for any localized vector field $\mathbf{M}$ (verifying operator adjointness on the staggered grid rather than Maxwell wave dynamics); we pair it with live solver and operator mutation tests in `tests/test_075_magnetic_toroid.py`.
#
# #### 3.2 Marinov quasi-static toroidal moment formula
#
# For a circular torus of major radius $R$ and minor radius $r < R$ wound with $N$ poloidal turns carrying current $I$, Ampère's law $\oint \mathbf{H}\cdot\mathrm{d}\mathbf{l} = N I$ around an interior circle of radius $\rho \in [R-r, R+r]$ gives $H_\phi(\rho) = \frac{N I}{2\pi\rho}$. With $\mathbf{r}\times\hat{\boldsymbol{\phi}} = (\rho\hat{\boldsymbol{\rho}} + z\hat{\mathbf{z}})\times\hat{\boldsymbol{\phi}} = \rho\hat{\mathbf{z}} - z\hat{\boldsymbol{\rho}}$, the axial toroidal moment of the core magnetization $M_\phi = \chi_m H_\phi$ ($\chi_m = \mu_r - 1$) plus the free poloidal winding current ($\mu_{\mathrm{eff}} = \mu_r$ total, or $\chi_m$ core-only) is (**derived here**):
#
# $$T_z = \frac{1}{2}\int_{\mathrm{torus}} \rho\,\mu_{\mathrm{eff}} H_\phi(\rho)\,\mathrm{d}^3r = \frac{1}{2}\int_{\mathrm{cross\text{-}section}} \mathrm{d}\rho\,\mathrm{d}z \int_0^{2\pi} \rho\,\mathrm{d}\phi\,\rho\,\mu_{\mathrm{eff}}\frac{N I}{2\pi\rho} = \frac{N I \mu_{\mathrm{eff}}}{2}\iint_{(\rho-R)^2 + z^2 \le r^2} \rho\,\mathrm{d}\rho\,\mathrm{d}z.$$
#
# Since the centroid of the circular cross-section is at $\rho = R$ with area $A_{\mathrm{cs}} = \pi r^2$, Pappus's centroid theorem gives the exact quasi-static toroidal moment for the $H_\phi \propto 1/\rho$ profile:
#
# $$\boxed{T_z^{\mathrm{qs}} = \frac{1}{2}\,\pi N I \mu_{\mathrm{eff}}\,r^2 R,}$$
#
# which for an air-core toroidal solenoid ($\mu_{\mathrm{eff}} = 1$) is exactly Eq. (6) of Marinov et al. (2007) ($T = \pi N I d R^2 / 2$ in their notation $d \equiv R, R \equiv r$) [1]. (If one instead weights the $1/\rho$ inductance integral by $R^2$ before integrating over the circular cross-section, $\iint \rho^{-1}\,\mathrm{d}\rho\,\mathrm{d}z = 2\pi(R - \sqrt{R^2 - r^2})$, one obtains $T_z = \pi N I \mu_{\mathrm{eff}} R^2(R - \sqrt{R^2 - r^2}) = \frac{1}{2}\pi N I \mu_{\mathrm{eff}} r^2 R [1 + \frac{1}{4}(r/R)^2 + \dots]$.)
#
# #### 3.3 Exact Bessel expansion of $a_{l0}$, effective electric dipole $\mathbf{p}_{\mathrm{eff}}$, and $\omega^6$ vs $\omega^4$ scaling (**derived here**)
#
# By the surface/volume equivalence principle (Dubovik & Tugushev 1990 [2]; Boardman et al. 2005 [12]; Savinov et al. 2014 [9]), an azimuthal magnetization loop $\mathbf{M} = M_\phi\hat{\boldsymbol{\phi}}$ (together with its enclosed poloidal winding flux) is equivalent in the exterior region to an azimuthal **magnetic current ring** $\mathbf{J}_m = +\mathrm{i}\omega\mu_0\mathbf{M}_{\mathrm{tot}} = I_m \delta(\rho - R)\delta(z)\hat{\boldsymbol{\phi}}$ at radius $\rho = R$, where the integrated magnetic current is related to $T_{\mathrm{tot},z} = \frac{1}{2}(2\pi R^2) \int M_{\mathrm{tot},\phi}\,\mathrm{d}A_{\mathrm{cs}}$ by $I_m \pi R^2 = +\mathrm{i}\omega\mu_0 T_{\mathrm{tot},z}$ in Wakis's $\mathrm{e}^{+\mathrm{i}\omega t}$ convention.
#
# Expanding the exact far-field electric field of an azimuthal magnetic current ring of radius $R$ at polar angle $\theta$ gives (**derived here** in $\mathrm{e}^{+\mathrm{i}\omega t}$):
#
# $$E_\theta^{\mathrm{ff}}(r, \theta) = -\frac{\mu_0 \omega^2}{4\pi}\left(-\frac{\mathrm{i} k_0 T_{\mathrm{tot},z}}{c_0}\right)\frac{\mathrm{e}^{-\mathrm{i}k_0 r}}{r}\,\frac{2 J_1(k_0 R \sin\theta)}{k_0 R} = +\mathrm{i}\,\frac{\eta_0 k_0^3 T_{\mathrm{tot},z}}{4\pi}\,\frac{\mathrm{e}^{-\mathrm{i}k_0 r}}{r}\,\frac{2 J_1(k_0 R \sin\theta)}{k_0 R}, \qquad E_\phi^{\mathrm{ff}}(r, \theta) = 0,$$
#
# where $J_1$ is the cylindrical Bessel function of order 1. Projecting onto the normalized $\mathrm{TM}_{l0}$ vector spherical harmonics ($\Phi_{l0,\theta} = \frac{1}{\sqrt{l(l+1)}}\frac{\mathrm{d}Y_{l0}}{\mathrm{d}\theta}$) using the Bauer–Gegenbauer integral $\int_0^\pi J_1(x\sin\theta) P_l^1(\cos\theta)\sin\theta\,\mathrm{d}\theta$ yields the exact multipole coefficients for all $k_0 R$:
# - Only odd $l = 1, 3, 5, \dots$ are non-zero, with $a_{l0} \propto j_l(k_0 R)$ (spherical Bessel function of order $l$).
# - For $l = 1$, the ring form factor is $F_1(k_0 R) = \frac{3 j_1(k_0 R)}{k_0 R} = 1 - \frac{(k_0 R)^2}{10} + \mathcal{O}((k_0 R)^4)$, so the magnetic toroid radiates the exact $\mathrm{TM}_{10}$ spherical wave of an **axial electric dipole** with effective current moment $\mathbf{I\ell}_{\mathrm{eff}} = +\mathrm{i}\omega\mathbf{p}_{\mathrm{eff}} = +k_0^2\mathbf{T}_{\mathrm{tot}} F_1(k_0 R)$, effective dipole moment (**derived here** in Wakis's $\mathrm{e}^{+\mathrm{i}\omega t}$ convention), and complex $\mathrm{TM}_{10}$ VSH coefficient $a_{10}$:
#
# $$\boxed{\mathbf{p}_{\mathrm{eff}} = -\frac{\mathrm{i} k_0}{c_0}\,\mathbf{T}_{\mathrm{tot}}\,\frac{3 j_1(k_0 R)}{k_0 R} \;\underset{k_0 R \ll 1}{\longrightarrow}\; -\frac{\mathrm{i} k_0}{c_0}\,\mathbf{T}_{\mathrm{tot}} = -\frac{\mathrm{i}\omega}{c_0^2}\,\mathbf{T}_{\mathrm{tot}}, \qquad a_{10} = -\mathrm{i}\sqrt{\frac{\eta_0}{12\pi}}\,k_0^3 T_{\mathrm{tot},z}\,F_1(k_0 R).}$$
#
# *(Note on time-harmonic sign convention: in Wakis's $\mathrm{e}^{+\mathrm{i}\omega t}$ convention at `wakis/near2far.py:56`, $\partial_t \to +\mathrm{i}\omega$ and an electric dipole $\mathbf{p}$ has current moment $\mathbf{I\ell}_{\mathrm{eff}} = +\mathrm{i}\omega\mathbf{p}_{\mathrm{eff}}$. Integrating the toroidal current radiation integral by parts gives $\mathbf{I\ell}_{\mathrm{eff}} = -c_0^{-2}\partial_t^2\mathbf{T}_{\mathrm{tot}} = +k_0^2\mathbf{T}_{\mathrm{tot}}$, so $\mathbf{p}_{\mathrm{eff}} = +\mathbf{I\ell}_{\mathrm{eff}}/(\mathrm{i}\omega) = -\mathrm{i} k_0 \mathbf{T}_{\mathrm{tot}} / c_0$ and $\arg(a_{10} / T_{\mathrm{tot},z}) = -\pi/2\;\mathrm{rad}\;(-90^\circ)$. In the physics $\mathrm{e}^{-\mathrm{i}\omega t}$ convention where $\partial_t \to -\mathrm{i}\omega$ and $\mathbf{I\ell}_{\mathrm{eff}} = -\mathrm{i}\omega\mathbf{p}_{\mathrm{eff}}$, the sign of $\mathbf{p}_{\mathrm{eff}}$ is $+\mathrm{i} k_0 \mathbf{T}_{\mathrm{tot}} / c_0$.)*
#
# - **Zero-fit-parameter radiated power:** Substituting $\mathbf{p}_{\mathrm{eff}}$ into the electric dipole power formula gives
#
# $$\boxed{P_{\mathrm{rad}}^{(\mathrm{toroid})} = \frac{\mu_0 \omega^4 |\mathbf{p}_{\mathrm{eff}}|^2}{12\pi c_0} = \frac{\mu_0 \omega^6 |\mathbf{T}_{\mathrm{tot}}|^2}{12\pi c_0^3}\left[\frac{3 j_1(k_0 R)}{k_0 R}\right]^2.}$$
#
# - **Frequency scaling ($\omega^6$ at fixed $T$ vs fixed drive current $I$):**
#   1. At **fixed toroidal moment $|\mathbf{T}_{\mathrm{tot}}|$** (and $k_0 R \ll 1$), $P_{\mathrm{rad}} \propto \omega^6 |\mathbf{T}_{\mathrm{tot}}|^2$, two powers of $\omega$ steeper than the $\omega^4 |\mathbf{p}|^2$ scaling of a fixed electric dipole because $|\mathbf{p}_{\mathrm{eff}}| = (\omega/c_0^2)|\mathbf{T}_{\mathrm{tot}}| \propto \omega$.
#   2. At **fixed winding current amplitude $|I_0|$** in a dispersive Lorentz core with permeability $\mu(\omega) = 1 + \frac{(\mu_{\mathrm{DC}} - 1)\omega_{\mathrm{res}}^2}{\omega_{\mathrm{res}}^2 - \omega^2 + \mathrm{i}\gamma\omega}$ (in $\mathrm{e}^{+\mathrm{i}\omega t}$), the toroidal moment itself varies with frequency as $T_{\mathrm{tot}}(\omega) \propto \mu(\omega) I_0$, and internal Helmholtz wave enhancement inside the high-index core ($\lambda_{\mathrm{core}} = \lambda_0 / \sqrt{\varepsilon_r\mu_r}$) further amplifies $M_\phi$ as $\omega$ approaches the core resonance (**derived here**):
#      $$P_{\mathrm{rad}}(\omega)\big|_{\text{fixed } I_0} \propto \omega^6\,|\mu(\omega)|^2\,\left[\frac{3 j_1(k_0 R)}{k_0 R}\right]^2 |I_0|^2.$$
#   3. **At fixed drive voltage $|V_0|$** across an inductive winding ($Z_{\mathrm{in}} \approx +\mathrm{i}\omega L_{\mathrm{tor}}$), the winding current scales as $|I_0(\omega)| = |V_0| / (\omega L_{\mathrm{tor}}) \propto \omega^{-1}$, which reduces the radiated power scaling by $\omega^{-2}$ to $P_{\mathrm{rad}}(\omega)\big|_{\text{fixed } V_0} \propto \omega^4$.
#
# - **Leading higher multipole (electric octupole $a_{30}$):** Because $a_{20} = 0$ by mid-plane symmetry, the first correction to the $\sin\theta$ dipole pattern is the axial **electric octupole** ($l = 3, m = 0$). Projecting $\frac{2 J_1(x\sin\theta)}{x} = \sin\theta\left[1 - \frac{x^2}{8}\sin^2\theta + \mathcal{O}(x^4)\right]$ onto the orthonormal VSHs $\Phi_{10,\theta}(\theta) = -\sqrt{\frac{3}{8\pi}}\sin\theta$ and $\Phi_{30,\theta}(\theta) = -\frac{\sqrt{21}}{8\sqrt{\pi}}\sin\theta(5\cos^2\theta - 1)$ (using $\int_{-1}^1 (1 - \mu^2)^2(5\mu^2 - 1)\,\mathrm{d}\mu = -\frac{16}{105}$ vs $\int_{-1}^1 (1-\mu^2)\,\mathrm{d}\mu = \frac{4}{3}$, and in general the Gegenbauer integral $\int_0^\pi J_1(x\sin\theta) P_3^1(\cos\theta)\sin\theta\,\mathrm{d}\theta = -\frac{15}{2}j_3(x)$ vs $\int_0^\pi J_1(x\sin\theta) P_1^1(\cos\theta)\sin\theta\,\mathrm{d}\theta = -2 j_1(x)$ with norm ratio $\frac{\sqrt{7/(4\pi\cdot 12)}}{\sqrt{3/(4\pi\cdot 2)}}\cdot\frac{15/2}{2} = \sqrt{\frac{7}{8}}$) gives the thin-ring octupole-to-dipole amplitude ratio, and averaging the octupole radial-axial kernel $\rho(\rho^2 - 4z^2)$ vs the dipole kernel $\rho$ (the kernel follows from the magnetization term of the multipole coefficient, $a_E(l,0) \propto \int (\mathbf{r}\times\mathbf{M})\cdot\nabla[j_l(kr)Y_{l0}]\,\mathrm{d}^3r$ (Jackson Eq. 9.167): at leading order in $k_0R$, $r^3P_3 = z^3 - \tfrac{3}{2}z\rho^2$ and $\mathbf{r}\times\hat{\boldsymbol\phi} = \rho\hat{\mathbf{z}} - z\hat{\boldsymbol\rho}$, so $(\mathbf{r}\times\mathbf{M})\cdot\nabla(r^3P_3) = -\tfrac{3}{2}M_\phi\,\rho(\rho^2 - 4z^2)$, while the $l=1$ term gives $\rho M_\phi$) with $M_\phi(\rho, z) \propto 1/\rho$ over the finite circular cross-section $(\rho - R)^2 + z^2 \le r^2$ ($\langle \rho^2 - 4z^2 \rangle / R^2 = 1 + \frac{1}{4}(r/R)^2 - (r/R)^2 = 1 - \frac{3}{4}(r/R)^2$) yields the finite-$r/R$ correction, valid at leading order in $k_0 R$ (**derived here**):
#
# $$\boxed{\frac{|a_{30}|}{|a_{10}|} = \sqrt{\frac{7}{8}}\,\left[1 - \frac{3}{4}\left(\frac{r}{R}\right)^2\right]\frac{j_3(k_0 R)}{j_1(k_0 R)} = \frac{\sqrt{14}}{140}\,\left[1 - \frac{3}{4}\left(\frac{r}{R}\right)^2\right](k_0 R)^2 + \mathcal{O}\!\left((k_0 R)^4\right).}$$
#
# ---
#
# ### 4. Discrete $C_{4v}$ Symmetry (Square of Four Permeable Rods) vs Broken Loops
#
# #### 4.1 Why a $C_{4v}$ square of four rods guarantees $\mathbf{m} = \mathbf{0}$ and $b_1 = b_2 = b_3 = 0$ (**derived here**)
#
# Continuous rotational symmetry ($C_\infty$) is not required to cancel the magnetic dipole moment. Consider four identical permeable rods arranged around a square of center-to-rod radius $R_{\mathrm{sq}}$ in the $xy$-plane:
# - Rod 1 at $(+R_{\mathrm{sq}}, 0)$ with magnetization $+M_0\hat{\mathbf{y}}$,
# - Rod 2 at $(0, +R_{\mathrm{sq}})$ with magnetization $-M_0\hat{\mathbf{x}}$,
# - Rod 3 at $(-R_{\mathrm{sq}}, 0)$ with magnetization $-M_0\hat{\mathbf{y}}$,
# - Rod 4 at $(0, -R_{\mathrm{sq}})$ with magnetization $+M_0\hat{\mathbf{x}}$.
#
# Under a $90^\circ$ rotation $R_{\pi/2}$ about the $z$-axis ($\phi \to \phi + \pi/2$), the magnetization distribution is invariant ($C_4$ rotational symmetry), and under vertical mirror reflections $\sigma_v$ through the $xz$- and $yz$-planes (or diagonal planes $x = \pm y$), the poloidal current distribution is even while the pseudovector magnetization reverses direction consistent with an axial polar toroidal vector $T_z\hat{\mathbf{z}}$ (**full $C_{4v}$ point-group symmetry**):
# - **Rotational $C_4$ selection rule ($m \equiv 0 \pmod 4$):** In the spherical harmonic basis $Y_{lm}(\theta, \phi) \propto \mathrm{e}^{\mathrm{i}m\phi}$, invariance under $\phi \to \phi + \pi/2$ requires $\mathrm{e}^{\mathrm{i}m\pi/2} = 1$, so **only azimuthal orders $m \equiv 0 \pmod 4$ ($m = 0, \pm 4, \pm 8, \dots$) can be non-zero!**
# - **Mirror $\sigma_v$ ($C_{4v}$) selection rule ($b_{l0} = 0$):** Pure $C_4$ rotation alone allows $m = 0$ magnetic multipoles $b_{l0}$ (for example, if the rods had a chiral or helical tilt). It is the vertical mirror reflection symmetry $\sigma_v$ ($\phi \to -\phi$) of the $C_{4v}$ poloidal rod square—under which the polar-vector electric multipole $a_{l0}$ is even while the axial/pseudovector magnetic multipole $b_{l0}$ is odd ($b_{l0} = -b_{l0} = 0$)—that forces $b_{l0} = 0$ for all $l$!
# - **Combined $C_{4v}$ multipole bound ($l \le 3$):** Since $|m| \le l$, no $m = \pm 4$ term can exist for $l = 1, 2, 3$. Therefore, **$C_{4v}$ symmetry guarantees $b_{1m} = b_{2m} = b_{3m} = 0$ identically**, while allowing small $m = \pm 4$ electric ($a_{4,\pm 4}$) and magnetic ($b_{4,\pm 4}$) multipole residuals starting at $l = 4$ that grow with the electrical size $k_0 R_{\mathrm{sq}}$.
#
# #### 4.2 Breaking the closed loop: five physical counter-cases
#
# Whenever the closed $C_{\infty v}$ or $C_{4v}$ magnetization path is broken, magnetic poles $\rho_m = -\nabla\cdot\mathbf{M} \ne 0$ or uncancelled azimuthal currents appear, exciting strong magnetic ($\mathrm{TE}$) multipoles $b_{1m}$:
# 1. **Air gap in the toroidal core ($60^\circ$ sector removed):** North and south magnetic poles form on the two cut faces of the high-$\mu$ core, creating a large transverse magnetic dipole $\mathbf{m}_\perp \ne \mathbf{0}$ and making $|b_1|/|a_1| = 2.525 > 1$.
# 2. **7 of 8 feeds active (1 feed open):** Inspired by Canós Valero et al. (2024) [3] (who model an 8-element metallic toroidal ring of individual loop diameter $5.5\;\mathrm{mm}$ evaluated at $1000\;\mathrm{MHz}$ and use a horn rated $\ge 0.7\;\mathrm{GHz}$; we adapt their 7-of-8 open-feed symmetry-breaking test to our permeable torus with $R = 15\;\mathrm{mm}$, $r = 5\;\mathrm{mm}$ at $f_0 = 0.75\;\mathrm{GHz}$), turning off one of the 8 loops breaks $C_{8v}$ symmetry and leaves an uncancelled magnetic dipole $m_\phi$ comparable in radiated power to the toroidal dipole $T_z$ ($|b_1|/|a_1| = 0.4898$).
# 3. **Unbalanced feed phases — Mode 1 antiphase split-winding "loop mode" (`[++ ++ -- --]`):** Whereas Mode 1 of Yousefi & Diaz (2017) [10] is a circularly polarized quadrature-phased excitation ($\mathrm{e}^{\pm\mathrm{i}\phi_n}$), driving the $+y$ half of the torus with $+I_0$ and the $-y$ half with $-I_0$ (`[++ ++ -- --]`) is its linearly polarized antiphase counterpart: $\mathbf{M}$ reverses on one half so both halves point in the same transverse direction. By antisymmetry, the axial toroidal moment cancels ($T_z = 0 \implies |a_1|_{\text{Mode 1}}/|a_1|_{\text{closed}} = 1.584e-06$, at the numerical noise floor) while the two halves add constructively as a transverse magnetic dipole ($|b_1|_{\text{Mode 1}}/|a_1|_{\text{closed}} = 2.223$, so $|b_1|/|a_1| = 1.404e+06$ reflects the noise-floor denominator $a_1 \approx 0$).
# 4. **3 of 4 rods in the square (1 rod removed):** Breaking $C_{4v}$ down to 3 rods leaves an uncancelled rod magnetization $\mathbf{m} \ne \mathbf{0}$, causing $|b_1|/|a_1|$ to jump from $8.66e-06$ to $3.598$.
# 5. **3D helical winding pitch vs Marinov's planar compensating loop [1, 11]:** A real poloidal coil wound continuously as a single-layer 3D helical wire path $\rho(t) = R + r\cos(2\pi N t), \phi(t) = 2\pi t, z(t) = -r\sin(2\pi N t)$ discretised onto primary electric edges (`deposit_closed_wire_on_edges`) advances by $2\pi$ around $\hat{\boldsymbol{\phi}}$ after $N$ poloidal turns, producing a parasitic axial magnetic dipole moment $m_z^{\mathrm{par}} = \pi I_{\mathrm{turn}} R^2 [1 + \frac{1}{2}(r/R)^2]$ (Marinov et al. 2007 Eq. 9 [1]; McDonald 2008 [11]). Routing a separate planar circular return wire loop in the equatorial plane $z = 0$ of radius $R_{\mathrm{loop}}$ driven by the **exact same current** $-I_{\mathrm{turn}}$ (zero rescaling) makes $|b_1|/|a_1|$ pass through a sharp minimum at Marinov's Eq. 9 radius $R_1 = R\sqrt{1 + \frac{1}{2}(r/R)^2} = 15.41\;\mathrm{mm}$, reducing $|b_1|/|a_1|$ from $0.5098$ (analytical $0.4989$) down to $1.30e-03$ ($392\times$ suppression, and $19.6\times$ lower than a naive loop at $R_{\mathrm{loop}} = R = 15.0\;\mathrm{mm}$ where $|b_1|/|a_1| = 0.0254$).
#
# ---
#
# ### 5. Efficiency, Core Loss Bookkeeping & the Chu–Harrington $Q$ Bound
#
# In a dispersive magnetic core with complex relative permeability $\mu_r(\omega) = \mu'(\omega) - \mathrm{i}\mu''(\omega)$ (where $\mu'' = -\operatorname{Im}\{\mu_r(\omega)\} > 0$ for passive loss in Wakis's $\mathrm{e}^{+\mathrm{i}\omega t}$ convention, and $\tan\delta_m = \mu''/\mu'$), Poynting's theorem requires that the time-averaged input power delivered by the windings equals the sum of the far-field radiated power and the magnetic core dissipation:
#
# $$\boxed{P_{\mathrm{in}} = -\frac{1}{2}\operatorname{Re}\!\int_V \mathbf{E}\cdot\mathbf{J}_f^*\,\mathrm{d}^3r = P_{\mathrm{rad}} + P_{\mathrm{loss,core}}, \qquad P_{\mathrm{loss,core}} = \frac{1}{2}\omega\mu_0\int_{\mathrm{core}} \mu''(\omega)\,|\mathbf{H}(\mathbf{r})|^2\,\mathrm{d}^3r.}$$
#
# Using the quasi-static interior field $H_\phi \approx \frac{N I_0}{2\pi R}$ inside the core volume $V_{\mathrm{core}} = 2\pi^2 r^2 R$, the time-averaged stored magnetic energy $W_m$, core loss $P_{\mathrm{loss,core}}$, radiation quality factor $Q_{\mathrm{rad}}$, and radiation efficiency $\eta_{\mathrm{rad}}$ are (**derived here**):
#
# $$W_m = \frac{1}{4}\mu_0 \mu' \int_{\mathrm{core}} |\mathbf{H}|^2\,\mathrm{d}^3r \approx \frac{\mu_0 \mu' N^2 |I_0|^2 r^2}{8 R}, \qquad P_{\mathrm{loss,core}} = 2\omega (\tan\delta_m) W_m,$$
#
# $$\boxed{Q_{\mathrm{rad}}^{(\mathrm{toroid})} = \frac{2\omega W_m}{P_{\mathrm{rad}}^{(\mathrm{toroid})}} = \frac{12\,\mu'}{\pi\,|\mu_r|^2\,(k_0 r)^2 (k_0 R)^3 \left[3 j_1(k_0 R)/(k_0 R)\right]^2} \approx \frac{12}{\pi\,\mu'\,(k_0 r)^2 (k_0 R)^3 \left[3 j_1(k_0 R)/(k_0 R)\right]^2}, \qquad \eta_{\mathrm{rad}} = \frac{1}{1 + Q_{\mathrm{rad}}\tan\delta_m}.}$$
#
# Comparing $Q_{\mathrm{rad}}^{(\mathrm{toroid})}$ with the **Chu–Harrington minimum radiation $Q$** [4, 5] of a single-mode ($\mathrm{TM}_{10}$) electric dipole enclosed in the same bounding sphere of radius $a_{\mathrm{sph}} = R + r$,
#
# $$Q_{\mathrm{Chu}}(k_0 a_{\mathrm{sph}}) = \frac{1}{(k_0 a_{\mathrm{sph}})^3} + \frac{1}{k_0 a_{\mathrm{sph}}},$$
#
# shows that for $k_0 a_{\mathrm{sph}} \ll 1$, the magnetic toroid's radiation $Q$ exceeds the Chu bound by the factor
#
# $$\frac{Q_{\mathrm{rad}}^{(\mathrm{toroid})}}{Q_{\mathrm{Chu}}} \approx \frac{12}{\pi\,\mu' (k_0 R)^2}\left(\frac{a_{\mathrm{sph}}}{r}\right)^2\left(\frac{a_{\mathrm{sph}}}{R}\right) \propto (k_0 R)^{-2} \gg 1,$$
#
# because a toroidal dipole radiates via a second-order spatial derivative ($\mathbf{p}_{\mathrm{eff}} \propto k_0 \mathbf{T}$), paying an extra $(k_0 R)^2$ radiation resistance penalty relative to an electric dipole of the same size unless $\mu'$ is very large.
#
# ---
#
# ## 3D FIT Simulation Setup, Schematics & Mesh Tables
#
# We model the magnetic toroid and its comparison geometries in `wakis.SolverFIT3D` at nominal frequency $f_0 = 0.75\;\mathrm{GHz}$ ($\lambda_0 = 399.72\;\mathrm{mm}, k_0 = 15.719\;\mathrm{rad/m}$) in an open 6-face `CPML` domain:
# - **Dispersive magnetic core:** `wakis.dispersive.DispersiveMaterial` with a single Lorentz magnetic pole (`mu_poles`), $\mu_{\mathrm{DC}} = 20.0$, magnetic resonance $f_{\mathrm{res}} = 6.0\;\mathrm{GHz}$ ($8\times$ above $f_0$), and low damping $\gamma / 2\pi = 0.04\;\mathrm{GHz}$ (giving $\mu_r(f_0) = 20.30 - 0.016\mathrm{i}$, refractive index $n_{\mathrm{core}} = 4.51$, and internal wavelength $\lambda_{\mathrm{core}} = \lambda_0 / n_{\mathrm{core}} = 88.71\;\mathrm{mm}$).
# - **Locally refined graded mesh:** Built with `wakis.templates.meshing.graded_axis`, placing uniform $\Delta_{\mathrm{core}} = 2.50\;\mathrm{mm}$ cells across the core region ($35.5\;\mathrm{cells}/\lambda_{\mathrm{core}} \ge 20$ and $4.0\;\mathrm{cells}$ across the $2r = 10.0\;\mathrm{mm}$ core minor diameter on the $49\times 49\times 35 = 84,035$-cell nominal grid; refined to $\Delta_{\mathrm{core}} = 2.00\;\mathrm{mm}$, $44.4\;\mathrm{cells}/\lambda_{\mathrm{core}}$, and $5.0\;\mathrm{cells}$ across $2r$ on the $57\times 57\times 41 = 133,209$-cell medium grid, and to $\Delta_{\mathrm{core}} = 1.67\;\mathrm{mm}$, $53.2\;\mathrm{cells}/\lambda_{\mathrm{core}}$, and $6.0\;\mathrm{cells}$ across $2r$ on the $63\times 63\times 44 = 174,636$-cell fine convergence grid).
# - **Full-domain verification (Rule 1):** All simulations—especially the broken-loop counter-cases in Task 4—are run on the full 3D domain without symmetry planes that could mask asymmetric multipoles.
#
# ### Setup 1 · Closed Toroidal Core with Poloidal Windings ($xz$ and $xy$ Cross-Sections)
#
# ![Setup geometry (XZ meridional cut)](figures/00a_setup_geometry.png)
# ![Setup mesh and excitation (XZ meridional cut)](figures/00b_setup_mesh_and_excitation.png)
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 300 mm × 300 mm × 250 mm (z ∈ [-125 mm, 125 mm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 49 × 49 × 35 (84035 cells) |
# | Discretization | Δx & resolution | Δ_min = 2.50 mm, 35.5 cells/λ (in n_max = 4.51), 4.0 cells across feature (10 mm) |
# | Boundary | ±x, ±y, ±z faces | CPML (6 cells, 79.2 mm transverse / 73.4 mm axial) |
# | Material | Background | Vacuum, ε = 1.00 (isotropic) |
# | Material | Lorentz magnetic torus (μ_DC = 20) | ε_r = 1.00, μ_r(f₀) = 20.30 - 0.016i, DispersiveMaterial (Lorentz mu_poles), R = 15 mm, r = 5 mm |
# | Source | Poloidal CurrentSource / MagneticCurrentSource | f₀ = 750.0 MHz, Δf = 220.0 MHz Gaussian-modulated sinusoidal pulse |
# | Monitor | Near2FarMonitor & core DFTMonitor | Closed Huygens box (±55 mm × ±55 mm × ±45 mm) + 3D core volume DFT at f₀ = 750.0 MHz |
#
# ![Setup geometry (XY equatorial cut)](figures/00a_setup_geometry_top.png)
# ![Setup mesh and excitation (XY equatorial cut)](figures/00b_setup_mesh_and_excitation_top.png)
#
# ### Setup 2 · Discrete $C_4$ Square of Four Permeable Rods & Thin Wire Loop with `LumpedPort`
#
# ![Setup geometry (4-rod square XY cut)](figures/00a_setup_geometry_rods.png)
# ![Setup mesh and excitation (4-rod square XY cut)](figures/00b_setup_mesh_and_excitation_rods.png)
# ![Setup geometry (PEC wire loop with LumpedPort)](figures/00a_setup_geometry_loop.png)
# ![Setup mesh and excitation (PEC wire loop with LumpedPort)](figures/00b_setup_mesh_and_excitation_loop.png)

# %%
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
try:
    from IPython.display import Image, Markdown, display
except ImportError:

    def display(*_args, **_kwargs):
        """No-op fallback outside IPython."""

    def Image(*_args, **_kwargs):
        """No-op fallback outside IPython."""

    def Markdown(*_args, **_kwargs):
        """No-op fallback outside IPython."""


TUTORIAL_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(TUTORIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_DIR))

import _helpers as h  # noqa: E402

DATA_DIR = h.DATA_DIR
FIG_DIR = h.FIG_DIR

TASKS = {
    1: ("01_baseline_dipoles", "Part 1: Baseline electric dipole, magnetic loop, (ka)^4 scaling & LumpedPort wire loop"),
    2: ("02_closed_toroid_multipoles", "Part 2-3: Closed Lorentz toroid, Dubovik-Tugushev identity, VSH l<=5, omega^6 & (kR)^2 sweeps"),
    3: ("03_square_four_rods", "Part 4: Discrete C4 square of four permeable rods across electrical size"),
    4: ("04_break_the_loop", "Part 5: Symmetry/loop-breaking failure modes (gap, 7/8 feeds, Mode 1, 3/4 rods, helical + Marinov loop)"),
    5: ("05_equivalence_limits", "Part 6: Equivalence limits (near-field vs Hertzian dipole, N windings, core TE01 mode)"),
    6: ("06_efficiency_q_and_convergence", "Part 7: Core loss tan(delta_m) energy bookkeeping, Q vs Chu bound & 3-level mesh convergence"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (`--quick`, `--full`, `--task`, `--list-tasks`, `--plot`, `--out`)."""
    ap = argparse.ArgumentParser(
        description="Tutorial 4.7 · The magnetic toroid: a closed magnetization loop radiates like an axial electric dipole"
    )
    ap.add_argument("--quick", action="store_true", help="Run standard graded-mesh 3D FIT study tasks.")
    ap.add_argument("--full", action="store_true", help="Run fine-grid 3D FIT study tasks.")
    ap.add_argument("--task", type=int, default=None, help="Run a single study task (1..6).")
    ap.add_argument("--list-tasks", action="store_true", help="Print the study task table and exit.")
    ap.add_argument("--plot", action="store_true", help="Regenerate all figures and GIF from existing data/*.json.")
    ap.add_argument("--out", type=Path, default=DATA_DIR, help="Output data directory (default: data/).")
    if argv is None:
        env_args = os.environ.get("WAKIS_TUTORIAL_ARGS")
        argv = shlex.split(env_args) if env_args else sys.argv[1:]
    args, _ = ap.parse_known_args(argv)
    return args


def ensure_data_and_figures(args: argparse.Namespace) -> dict[str, dict]:
    """Run requested tasks or load precomputed `data/*.json` files and generate figures."""
    if args.list_tasks:
        for idx, (name, desc) in TASKS.items():
            print(f"{idx}\t{name}\t{desc}")
        raise SystemExit(0)

    quick = not args.full
    out_dir = Path(args.out)

    if args.task is not None:
        if args.task not in TASKS:
            raise SystemExit(f"--task must be in {list(TASKS)}")
        h.compute_all_study_data(quick=quick, out_dir=out_dir, selected_task=int(args.task))
        raise SystemExit(0)

    required_jsons = [
        "task_01_baseline_dipoles.json",
        "task_02_closed_toroid.json",
        "task_03_square_four_rods.json",
        "task_04_break_the_loop.json",
        "task_05_equivalence_limits.json",
        "task_06_efficiency_q_and_convergence.json",
    ]
    need_compute = args.full or not all((out_dir / fname).exists() for fname in required_jsons)
    if need_compute and not args.plot:
        h.compute_all_study_data(quick=quick, out_dir=out_dir, selected_task=None)

    h.generate_all_figures(data_dir=out_dir, fig_dir=FIG_DIR)
    summary = h.build_summary_json(data_dir=out_dir, write=need_compute)

    return {
        "t1": json.loads((out_dir / "task_01_baseline_dipoles.json").read_text()),
        "t2": json.loads((out_dir / "task_02_closed_toroid.json").read_text()),
        "t3": json.loads((out_dir / "task_03_square_four_rods.json").read_text()),
        "t4": json.loads((out_dir / "task_04_break_the_loop.json").read_text()),
        "t5": json.loads((out_dir / "task_05_equivalence_limits.json").read_text()),
        "t6": json.loads((out_dir / "task_06_efficiency_q_and_convergence.json").read_text()),
        "summary": summary,
    }


_cli_args = parse_args()
_study_data = ensure_data_and_figures(_cli_args)
_narr_md = h.build_narrative_markdown_from_json(data_dir=DATA_DIR)

# %% [markdown]
# ---
#
# ## Results · Physics Story & 3D FIT Verification
#
# ### 1. Single Loop = Magnetic Dipole $\mathbf{m}$ vs Short Electric Dipole $\mathbf{p}$ (`task_01_baseline_dipoles.json`)
#
# Figure 1 establishes the dual baseline before closing the magnetic loop:
# - **Panel (a) — Short axial electric dipole $\mathbf{p} = p_z\hat{\mathbf{z}}$:** Driven by an axial `CurrentSource`, the radiated far field is purely polar-polarized ($E_\theta \propto \sin\theta$, cross-polarization $E_\phi \approx 0$), with $100.0000\%$ of the vector spherical harmonic ($\mathrm{VSH}$) power in the electric dipole coefficient $a_{10}$ ($|b_1|/|a_1| = 1.85e-07$).
# - **Panel (b) — Small magnetic loop $\mathbf{m} = m_z\hat{\mathbf{z}}$:** Driven by a single loop in the $xy$-plane, the radiated far field is the exact dual—purely azimuthally polarized ($E_\phi \propto \sin\theta$, $E_\theta \approx 0$), with $99.9922\%$ of the radiated power in the magnetic dipole coefficient $b_{10}$ ($|a_1|/|b_1| = 2.76e-07$).
# - **Panel (c) — $(k_0 a)^4$ radiation scaling, permeable core boost, and real `LumpedPort` wire loop:** Sweeping the loop radius across $a \in \{10.0, 12.5, 15.0\}\;\mathrm{mm}$ confirms the rapid $(k_0 a)^4$ power scaling: with $5\times 5$ sub-cell disk area weights on the $\Delta_{\mathrm{core}} = 2.50\;\mathrm{mm}$ grid, $P_{\mathrm{rad}}(15\;\mathrm{mm})/P_{\mathrm{rad}}(10\;\mathrm{mm}) = 4.9695$ (log-log slope $3.96$), within $1.84\%$ of $(1.5)^4 = 5.0625$ and within $1.23\%$ of the finite-$k_0 a$ Bessel-corrected ratio $5.0313$. Inserting the $\mu_{\mathrm{DC}} = 20$ permeable core inside the loop increases the radiated power by a factor of $4.10\times$ while preserving $|b_1|/|a_1| = 3.35e+06 \gg 10^4$. Feeding a thin PEC square wire loop of half-side $a = 12.5\;\mathrm{mm}$ (side $2a = 25\;\mathrm{mm}$, perimeter $8a = 100\;\mathrm{mm} = 0.250\,\lambda_0$) at a single gap on the $-y$ arm with a $50\;\Omega$ `LumpedPort` (`Setup 2`) yields $\operatorname{Re} Z_{\mathrm{in}} = 1.007\;\Omega$ vs the small circular loop formula $R_{\mathrm{rad}}^{\mathrm{circ}}(k_0 a = 0.1965) = 0.294\;\Omega$: first, $2 P_{\mathrm{rad}}^{\mathrm{N2F}} / |I_{\mathrm{port}}|^2 = 0.986\;\Omega$ matches $\operatorname{Re} Z_{\mathrm{in}} = 1.007\;\Omega$ within $2.1\%$ (confirming zero artificial wire or port loss); second, the square loop area $4a^2 = (4/\pi)\pi a^2$ raises the magnetic-dipole radiation resistance by $(4/\pi)^2 = 1.621\times$ to $0.477\;\Omega$; third, single-port feed asymmetry excites an electric dipole mode with $|a_1|/|b_1| = 0.553$, adding $1 + |a_1/b_1|^2 = 1.306\times$ power ($0.623\;\Omega$); and fourth, the remaining factor $R_{\mathrm{meas}} / (R_{\mathrm{sq}} \cdot [1 + |a_1/b_1|^2]) = 1.583\times$ ($0.623 \times 1.583 = 0.986\;\Omega$) is an **unexplained residual** on the finite-thickness wire/port grid (for reference, the **analytic estimate** for a standing-wave current $I(s) \propto \cos(k_0(L/2 - s))$ on a $\lambda_0/4$ loop gives a loop-to-port current-squared ratio $[(\sin x / x)/\cos x]^2 = 1.621$ at $x = \pi/4$).

# %%
t1 = _study_data["t1"]
e1 = t1["electric_dipole"]
m1 = t1["magnetic_loop_air"]
mc1 = t1["magnetic_loop_core"]
lw1 = t1["lumped_wire_loop"]
ka4 = t1["ka4_scaling"]

display(Image(filename=str(FIG_DIR / "01_baseline_dipoles_and_ka4.png"), width=980))

print(
    f"Task 1 Baseline Dipoles (from data/task_01_baseline_dipoles.json):\n"
    f"  Short electric dipole : |a_10|^2 / P_VSH = {e1['p_a10_frac'] * 100:.5f}%, "
    f"|b_1|/|a_1| = {e1['b1_over_a1']:.2e}, sin(theta) L2 err = {e1['pattern_sin_theta_l2_err'] * 100:.3f}%\n"
    f"  Small magnetic loop   : |b_1|^2  / P_VSH = {m1['p_b1_frac'] * 100:.5f}%, "
    f"|a_1|/|b_1| = {m1['a1_over_b1']:.2e}, sin(theta) L2 err = {m1['pattern_sin_theta_l2_err'] * 100:.3f}%\n"
    f"  Loop (ka)^4 scaling   : P(15 mm)/P(10 mm) = {ka4['ratio_15_to_10_sim']:.4f} vs (1.5)^4 = {ka4['ratio_15_to_10_theory']:.4f} "
    f"(rel. diff = {ka4['ratio_rel_err'] * 100:.2f}%, vs finite-ka J1^2 {ka4['ratio_rel_err_vs_exact_j1'] * 100:.2f}%, slope = {ka4['fitted_slope']:.2f})\n"
    f"  Permeable core loop   : power gain over air loop = {mc1['power_gain_over_air']:.2f}x, |b_1|/|a_1| = {mc1['b1_over_a1']:.2e}\n"
    f"  LumpedPort wire loop  : Z_in = {lw1['z_in_real_ohm']:.3f} + j*{lw1['z_in_imag_ohm']:.2f} Ohm "
    f"(2*P_rad/|I_port|^2 = {lw1['r_rad_from_prad_ohm']:.3f} Ohm vs R_rad,circ = {lw1['r_rad_circular_small_loop_ohm']:.3f} Ohm, "
    f"R_rad,sq = {lw1['r_rad_square_small_loop_ohm']:.3f} Ohm, L = {lw1['l_loop_nh']:.2f} nH, "
    f"unexplained residual = {lw1['unexplained_residual_factor']:.3f}x vs analytic cos-current factor {lw1['analytic_cos_current_factor_pi_over_4']:.3f}x), "
    f"|b_1|/|a_1| = {lw1['b1_over_a1']:.3f} (|a_1|/|b_1| = {lw1['a1_over_b1_feed_asymmetry']:.3f})"
)

# %% [markdown]
# ---
#
# ### 2. The Closed Magnetization Loop & Exact FIT Multipole Measurement (`task_02_closed_toroid.json`)
#
# Next we wind $N = 8$ in-phase poloidal loops around the closed Lorentz dispersive torus ($R = 15.0\;\mathrm{mm}, r = 5.0\;\mathrm{mm}, \mu_{\mathrm{DC}} = 20.0$).
#
# Figure 2 visualizes the simulated 3D FIT fields (`DFTMonitor` cuts overlaid with `wakis.schematic.overlay_geometry`):
# - **Panel (a) — Equatorial cut ($z = 0$):** Inside the permeable ring ($10\;\mathrm{mm} \le \rho \le 20\;\mathrm{mm}$), the magnetic flux density $|\mathbf{B}|$ is strongly concentrated and the magnetization vectors $\operatorname{Re}\{\mathbf{M}\} $ form a closed counterclockwise azimuthal vortex. Integrating $\mathbf{M}$ over the 3D core confirms that the net magnetic dipole moment cancels to $|\int \mathbf{M}\,\mathrm{d}V| / \int |\mathbf{M}|\,\mathrm{d}V = 5.22e-07 < 10^{-6}$.
# - **Panel (b) — Meridional cut ($y = 0$):** The external magnetic field $H_y(x, 0, z)$ emerging from the torus has even symmetry in $z$ and opposite sign on $+x$ and $-x$ ($H_\phi \propto \sin\theta$), matching a vertically oriented $\mathrm{TM}_{10}$ electric dipole.
# - **Panel (c) — Axial electric field $E_z(x, y, 0)$:** Induced by Faraday's law $\oint \mathbf{E}\cdot\mathrm{d}\mathbf{l} = \mathrm{i}\omega\Phi_B$ threading the azimuthal magnetic flux tube, $E_z$ peaks along the central $z$-axis like a virtual displacement-current rod along $\hat{\mathbf{z}}$.
#
# Figure 3 quantifies the spherical multipole spectrum ($l \le 5$), pattern, and scaling laws:
# - **Dubovik–Tugushev identity on the 3D FIT grid:** Evaluating $\mathbf{J}_b = \nabla_{\mathrm{FIT}}\times\mathbf{M}$ via the discrete FIT dual-curl operator `solver.itDa @ (solver.C.T @ solver.tDs)` and integrating $\mathbf{T}_{\mathrm{DT}}(\mathbf{J}_b) = \frac{1}{10}\int [\mathbf{r}(\mathbf{r}\cdot\mathbf{J}_b) - 2r^2\mathbf{J}_b]\,\mathrm{d}^3r$ matches $\mathbf{T}_{\mathrm{mag}}(\mathbf{M}) = \frac{1}{2}\int (\mathbf{r}\times\mathbf{M})\,\mathrm{d}^3r$ to machine precision (`rel_err = 1.77e-16`), whereas mutating the coefficient $-2r^2\mathbf{J}_b \to -1r^2\mathbf{J}_b$ produces the exact $40.0\%$ operator difference.
# - **Broadband multipole-to-far-field self-consistency:** Computing $\mathbf{p}_{\mathrm{eff}}(f) = -(\mathrm{i} k_0 / c_0)\mathbf{T}_{\mathrm{tot}}^{\mathrm{sim}}(f) F_1(k_0 R)$ directly from the simulated 3D volume integral $\mathbf{T}_{\mathrm{tot}}^{\mathrm{sim}}(f)$ at each of the 9 frequencies across $0.55\text{–}0.95\;\mathrm{GHz}$ matches the simulated `Near2FarMonitor` radiated power $P_{\mathrm{rad}}^{\mathrm{N2F}}(f)$ within **$0.15\%\text{–}0.29\%$** across the entire band ($P_{\mathrm{rad}}^{\mathrm{N2F}}(f) / P_{\mathrm{dip}}(\mathbf{p}_{\mathrm{eff}}(f)) \in [0.9971, 0.9985]$, and **$0.17\%$** vs the closed-box `FluxMonitor` integral at $f_0$). Because both $\mathbf{T}_{\mathrm{tot}}^{\mathrm{sim}}$ and $P_{\mathrm{rad}}^{\mathrm{N2F}}$ are extracted from the same simulated fields, we label this $0.20\%$ agreement at $f_0$ as **internal multipole-to-far-field self-consistency**.
# - **$\omega^6$ fixed-$T$ vs fixed-$I$ scaling and $(k_0 R)^2$ octupole scaling:** Across $0.55\text{–}0.95\;\mathrm{GHz}$, the Lorentz permeability $|\mu(f)|$ rises by **$1.6\%$** and the **analytic estimate** for the interior Bessel factor $1 + (k_c r)^2/8$ rises by **$1.7\%$** (at $f_0$, $1 + (k_c r)^2/8 = 1.016$, which does not explain the full dynamic residual $|T_z^{\mathrm{sim}}|/T_z^{\mathrm{disc}} = 1.038$), while the measured simulated toroidal moment $|T_{\mathrm{tot}}^{\mathrm{sim}}(f)|$ rises by **$13.6\%$**, giving a fixed-current log-log slope of **$6.4366$** ($6.44$). Dividing $P_{\mathrm{rad}}(f)$ by the actual simulated $|T_{\mathrm{tot}}^{\mathrm{sim}}(f)/T_{\mathrm{tot}}^{\mathrm{sim}}(f_0)|^2 F_1(k_0 R)^2$ at each frequency isolates the exact point-dipole fixed-$T$ scaling with a measured slope of **$5.9974$** vs theoretical **$6.0000$** ($0.04\%$ error!). Sweeping $R \in \{10, 15, 20, 25\}\;\mathrm{mm}$ confirms that the electric octupole ratio $|a_{30}|/|a_{10}|$ follows the thin-ring formula $\sqrt{7/8}\,j_3(k_0 R)/j_1(k_0 R)$ within $2.7\%\text{–}17.9\%$ and matches the derived finite cross-section octupole formula $\sqrt{7/8}\,[1 - \frac{3}{4}(r/R)^2]\,j_3(k_0 R)/j_1(k_0 R)$ within **$0.31\%\text{–}2.42\%$**, with a fitted $(k_0 R)^2$ log-log slope of **$2.18$**.

# %%
t2 = _study_data["t2"]
mom2 = t2["moments"]
vsh2 = t2["vsh_toroid_8feed"]
peq2 = t2["power_equivalence"]
f2 = t2["frequency_scaling"]
r2 = t2["octupole_scaling"]
mar2 = t2["marinov_quasistatic"]
fc2 = t2["feed_comparison"]

display(Image(filename=str(FIG_DIR / "02_closed_toroid_fields.png"), width=980))
display(Image(filename=str(FIG_DIR / "03_toroid_multipoles_and_scaling.png"), width=980))

tz_mag_abs = float(np.hypot(mom2["tz_mag_core_am3"]["real"], mom2["tz_mag_core_am3"]["imag"]))
tz_dt_abs = float(np.hypot(mom2["tz_core_dt_am3"]["real"], mom2["tz_core_dt_am3"]["imag"]))

print(
    f"Task 2 Closed Toroid & Multipole Decomposition (from data/task_02_closed_toroid.json):\n"
    f"  Net magnetic dipole cancellation |int M dV| / int |M| dV : {mom2['m_cancel_ratio']:.2e}\n"
    f"  Dubovik-Tugushev |T_z,DT(J_b)| = {tz_dt_abs:.9e} A m^3 vs |T_z,mag(M)| = {tz_mag_abs:.9e} A m^3 "
    f"(rel. err = {mom2['dt_vs_mag_rel_err']:.2e}; mutated -1*r^2*J_b rel. err = {mom2['dt_mutated_rel_err']:.4f})\n"
    f"  Marinov formula |T_z,tot| : FIT = {mar2['tz_tot_sim_abs_am3']:.4e} A m^3 vs exact = {mar2['tz_tot_exact_am3']:.4e} A m^3 "
    f"(rel. err = {mar2['rel_err_tot_sim_vs_exact'] * 100:.2f}%)\n"
    f"  VSH multipoles (l <= 5)   : |a_10|^2 / P_VSH = {vsh2['p_a10_frac'] * 100:.5f}%, "
    f"|b_1|/|a_1| = {vsh2['b1_over_a1']:.2e}, even-l power frac = {vsh2['even_l_power_frac']:.2e}\n"
    f"  Pattern & polarization    : sin(theta) L2 err = {vsh2['pattern_sin_theta_l2_err'] * 100:.3f}%, "
    f"cross-pol |E_phi|/|E_theta| = {vsh2['cross_pol_ratio']:.2e}\n"
    f"  Zero-fit-parameter P_rad  : N2F = {peq2['p_rad_n2f_w']:.4e} W, Flux = {peq2['p_rad_flux_w']:.4e} W, "
    f"Theory(p_eff) = {peq2['p_rad_theory_j1_w']:.4e} W (rel. err = {peq2['rel_err_n2f_vs_j1'] * 100:.2f}%)\n"
    f"  Feed comparison |b_1|/|a_1|: 8 windings = {fc2['toroid_8_windings']['b1_over_a1']:.2e}, "
    f"continuous sheet = {fc2['continuous_poloidal_sheet']['b1_over_a1']:.2e}, "
    f"ideal MagneticCurrentSource ring = {fc2['ideal_magnetic_current_ring']['b1_over_a1']:.2e}\n"
    f"  Scaling slopes & octupole : fixed-T omega slope = {f2['slope_omega_fixed_t_point_corrected']:.4f} "
    f"(fixed-I slope = {f2['slope_omega_fixed_i']:.4f}, mu rise = {f2['mu_band_rise_frac'] * 100:.1f}%, "
    f"analytic interior factor rise = {f2['analytic_interior_band_rise_frac'] * 100:.1f}%, T_sim rise = {f2['tz_sim_band_rise_frac'] * 100:.1f}%), "
    f"octupole |a_30|/|a_10| at R=15 mm = {r2['a30_over_a10_sim'][1]:.4e} "
    f"vs theory {r2['a30_over_a10_theory'][1]:.4e} (slope = {r2['fitted_slope_vs_kr']:.2f})"
)

# %% [markdown]
# ---
#
# ### 3. Square of Four Permeable Rods: Discrete $C_4$ Symmetry (`task_03_square_four_rods.json`)
#
# Figure 4 tests the discrete $C_4$ geometry—four separate permeable rods ($\mu_{\mathrm{DC}} = 20$) arranged in a square of radius $R_{\mathrm{sq}} \in \{12.5, 17.5, 22.5\}\;\mathrm{mm}$ with head-to-tail magnetization $+M\hat{\mathbf{y}}, -M\hat{\mathbf{x}}, -M\hat{\mathbf{y}}, +M\hat{\mathbf{x}}$:
# - **Panel (a):** Even though the four rods are physically separated at the corners, the four magnetization vectors sum to zero ($|\int \mathbf{M}\,\mathrm{d}V|/\int |\mathbf{M}|\,\mathrm{d}V = 8.14e-07\text{–}3.18e-06$) and drive an axial electric field $E_z$ through the center of the square just like the continuous torus.
# - **Panel (b):** Across all three square sizes, $|b_1|/|a_1| = 8.66e-06\text{–}1.84e-05 < 2\times 10^{-5}$ vanishes to grid precision and $99.9983\%\text{–}99.9999\%$ of the radiated power remains in $a_{10}$.
# - **Panel (c):** The leading non-axisymmetric $C_4$ multipole residual $|m = \pm 4| / |a_{10}|$ (at $l = 4, 5$) is small ($6.12e-05 \to 2.06e-04 \to 4.24e-04$) and grows monotonically with electrical size $k_0 R_{\mathrm{sq}}$.

# %%
t3 = _study_data["t3"]
display(Image(filename=str(FIG_DIR / "04_square_four_rods.png"), width=980))

print("Task 3 Square of Four Permeable Rods (from data/task_03_square_four_rods.json):")
for rec in t3["records"]:
    print(
        f"  R_sq = {rec['r_sq_mm']:4.1f} mm (k_0 R_sq = {rec['k0_r_sq']:.3f}) : "
        f"|int M dV|/int|M|dV = {rec['m_cancel_ratio']:.2e}, "
        f"|a_10|^2/P_VSH = {rec['p_a10_frac'] * 100:.5f}%, "
        f"|b_1|/|a_1| = {rec['b1_over_a1']:.2e}, "
        f"C4 |m=+/-4|/|a_10| = {rec['c4_m4_over_a10']:.2e}"
    )

# %% [markdown]
# ---
#
# ### 4. Break the Loop: Five Failure Modes & Marinov's Compensating Loop (`task_04_break_the_loop.json`)
#
# Figure 5 demonstrates what happens when the closed magnetization loop or winding symmetry is broken—each case serves as a physical falsification test where magnetic multipoles $b_{1m}$ immediately appear and alter the far-field polarization and pattern:
# - **$60^\circ$ air gap in the toroidal core:** Magnetic poles at the cut faces produce $|b_1|/|a_1| = 2.525$ ($86.5\%$ of the radiated power shifts into magnetic $\mathrm{TE}$ modes!).
# - **7 of 8 feeds active (1 feed open):** Reproducing the broken-feed scenario of Canós Valero et al. (2024) [3], turning off a single feed out of 8 causes $|b_1|/|a_1|$ to jump by **$9.17e+04\times$ ($5.0$ orders of magnitude)** from $5.34e-06$ to $0.4898$, making the parasitic magnetic dipole $m$ comparable to the toroidal dipole $T$.
# - **Mode 1 antiphase feeds (`[++ ++ -- --]`):** Reversing the phase on half the torus cancels $T_z$ by antisymmetry ($|a_1|_{\text{Mode 1}}/|a_1|_{\text{closed}} = 1.584e-06$ at the numerical noise floor) and turns the structure into a pure transverse magnetic loop antenna ($|b_1|_{\text{Mode 1}}/|a_1|_{\text{closed}} = 2.223$; the formal ratio $|b_1|/|a_1| = 1.404e+06$ reflects division by the noise-floor $a_1$).
# - **3 of 4 rods in the square (1 rod removed):** Removing one rod breaks $C_{4v}$ symmetry and increases $|b_1|/|a_1|$ from $8.66e-06$ to $3.598$.
# - **3D helical winding (1 net azimuthal turn) vs Marinov's planar compensating loop [1, 11]:** Discretising the real 3D helical wire path $\rho(t) = R + r\cos(2\pi N t), \phi(t) = 2\pi t, z(t) = -r\sin(2\pi N t)$ onto primary electric edges (`deposit_closed_wire_on_edges`) with turn current $I_{\mathrm{turn}} = 11.78\;\mathrm{mA}$ (zero rescaling) magnetizes the permeable core ($|a_1|/|a_{1,\mathrm{closed}}| = 0.8742$) while exciting a parasitic axial magnetic dipole $m_z^{\mathrm{par}}$ (Marinov Eq. 9 [1]) with $|b_1|/|a_1| = 0.5098$ (matching the analytical ratio $0.4989$ within $2.2\%$). Sweeping the radius $R_{\mathrm{loop}} \in \{13.0, 14.5, 15.0, 15.41, 15.8, 17.0\}\;\mathrm{mm}$ of a separate planar return wire loop in $z = 0$ driven by the **exact same current** $-I_{\mathrm{turn}}$ (Figure 5c, zero rescaling) shows $|b_1|/|a_1|$ passing through a sharp V-shaped minimum at Marinov's Eq. 9 radius $R_1 = R\sqrt{1 + \frac{1}{2}(r/R)^2} = 15.41\;\mathrm{mm}$: $|b_1|/|a_1|$ drops from $0.5098$ to $0.0254$ at the naive major radius $R_{\mathrm{loop}} = R = 15.0\;\mathrm{mm}$, reaches its minimum **$1.30e-03$** at $R_{\mathrm{loop}} = R_1 = 15.41\;\mathrm{mm}$ (**$392\times$ amplitude suppression** vs uncompensated and **$19.6\times$ lower** than at $R_{\mathrm{loop}} = R$), and rises back to $0.1112$ at $R_{\mathrm{loop}} = 17.0\;\mathrm{mm}$.
#
# | Configuration | Symmetry / Topology | $|b_1|/|a_1|$ | $|a_1|/|a_{1,\mathrm{closed}}|$ | $|b_1|/|a_{1,\mathrm{closed}}|$ | Electric Power Fraction | Magnetic Power Fraction | $|\int \mathbf{M}\,\mathrm{d}V|/\int|\mathbf{M}|\,\mathrm{d}V$ |
# | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
# | **Closed torus (8 in-phase feeds)** | `closed_8_feeds` | `5.341e-06` | `1.000e+00` | `5.341e-06` | 100.00% | 0.00% | `5.22e-07` |
# | **Core air gap (60° sector removed)** | `core_air_gap_60deg` | `2.525e+00` | `5.674e-01` | `1.432e+00` | 13.54% | 86.46% | `2.61e-01` |
# | **7 of 8 feeds active (1 feed open)** | `seven_of_eight_feeds` | `4.898e-01` | `8.750e-01` | `4.286e-01` | 80.61% | 19.39% | `4.34e-02` |
# | **Mode 1 antiphase (++ ++ -- --)** | `mode1_antiphase` | `1.404e+06` | `1.584e-06` | `2.223e+00` | 0.01% | 99.99% | `5.80e-01` |
# | **Square of 4 rods (C4 closed)** | `four_rods_closed` | `8.657e-06` | `8.779e-01` | `7.600e-06` | 100.00% | 0.00% | `8.14e-07` |
# | **3 of 4 rods (1 rod removed)** | `three_of_four_rods` | `3.598e+00` | `4.553e-01` | `1.638e+00` | 7.16% | 92.84% | `3.11e-01` |
# | **Helical winding (1 net azimuthal turn)** | `helical_uncompensated` | `5.098e-01` | `8.742e-01` | `4.457e-01` | 79.37% | 20.63% | `5.19e-07` |
# | **Helical + Marinov compensating loop** | `helical_marinov_compensated` | `1.300e-03` | `8.742e-01` | `1.136e-03` | 100.00% | 0.00% | `5.14e-07` |

# %%
t4 = _study_data["t4"]
display(Image(filename=str(FIG_DIR / "05_break_the_loop.png"), width=980))
display(Markdown(_narr_md["task_04_table"]))

# %% [markdown]
# ---
#
# ### 5. Where the Equivalence Ends: Near Field vs Far Field, Electrical Size & Core $\mathrm{TE}_{01\delta}$ Modes (`task_05_equivalence_limits.json`)
#
# Why is a magnetic toroid **not** simply an ordinary electric dipole everywhere in space? Figure 6 and the animated wave GIF (`figures/magnetic_toroid_vs_dipole_wave.gif`) expose three fundamental regimes:
# 1. **Reactive near field $\ne$ Hertzian dipole near field (Figure 6a–b):** Outside the bounding sphere $r > a_{\mathrm{sph}} = R + r = 20\;\mathrm{mm}$, both the magnetic toroid and a point electric dipole satisfy the source-free Maxwell equations ($\nabla\cdot\mathbf{E} = 0$), and the toroid's exterior field equals the full dipole field (including its $1/r^3$ and $1/r^2$ reactive near-zone terms) plus higher odd-$l$ electric multipoles ($l = 3, 5, \dots$). Inside and just outside the bounding sphere ($r \sim a_{\mathrm{sph}}$), the point dipole's $1/r^3$ field diverges toward $r \to 0$ whereas the finite-radius torus ring remains nonsingular at the origin and carries an electric octupole $a_{30}$ whose spherical Hankel near field $|h_3^{(2)}(k_0 r) / h_1^{(2)}(k_0 r)| \propto (k_0 r)^{-2}$ equals **$91.82$** at $r = 25\;\mathrm{mm}$ ($k_0 r = 0.393$) and decays by **$4.51\times$** to **$20.34$** at $r = 50\;\mathrm{mm}$ ($k_0 r = 0.786$, still in the reactive/intermediate zone). Evaluated with the **exact same metric** across both zones, the relative electric-field difference $\|\,|\mathbf{E}_{\mathrm{toroid}}| - |\mathbf{E}_{\mathrm{dipole}}|\,\|_2 / \|\,|\mathbf{E}_{\mathrm{dipole}}|\,\|_2$ drops from **$42.99\%$** in the inner reactive zone $r \in [20, 30]\;\mathrm{mm}$ ($k_0 r \in [0.31, 0.47]$) to **$12.95\%$** in the outer reactive/intermediate zone $r \in [45, 55]\;\mathrm{mm}$ ($k_0 r \in [0.71, 0.86]$) — a **$3.32\times$ reduction** — while the magnetic-field metric $\|\,|H_{y,\mathrm{toroid}}| - |H_{y,\mathrm{dipole}}|\,\|_2 / \|\,|H_{y,\mathrm{dipole}}|\,\|_2$ drops from **$46.02\%$** to **$3.99\%$** (**$11.52\times$ reduction**).
# 2. **Electrical size ($k_0 R$ roll-off and the $j_1(k_0 R)$ null, Figure 6c):** As an **analytic estimate** from the ring multipole expansion, the dipole form factor $F_1(k_0 R)^2 = [3 j_1(k_0 R)/(k_0 R)]^2$ rolls off with electrical size $k_0 R = 2\pi R/\lambda_0$ while the octupole-to-dipole ratio grows as $|a_{30}|/|a_{10}| = \sqrt{7/8}\,|j_3(k_0 R)/j_1(k_0 R)|$ (verified by our 3D FIT sweep across $R \in \{10, 15, 20, 25\}\;\mathrm{mm}$, $k_0 R \in [0.157, 0.393]$, where $|a_{30}/a_{10}|_{\mathrm{sim}} = 0.00054 \to 0.00404$ vs analytical $0.00066 \to 0.00416$). At the first zero of the spherical Bessel function $j_1(k_0 R)$ (**$k_0 R = 4.4934$**, i.e. circumference $2\pi R = 4.4934\,\lambda_0$, where $\tan(k_0 R) = k_0 R$), the axial electric dipole radiation $a_{10}$ vanishes completely while higher odd electric multipoles ($a_{30}, a_{50}$) dominate. (By contrast, at the first zero of the cylindrical Bessel function $J_1(k_0 R)$, $k_0 R = 3.8317$, only the equatorial radiation at $\theta = \pi/2$ vanishes.)
# 3. **Resolved internal core $\mathrm{TE}_{01\delta}$ dielectric-magnetic mode sweep ($\ge 20\;\mathrm{cells}/\lambda_{\mathrm{core}}$, Figure 6c) and breakdown of point-dipole equivalence:** Resolving the core at $\Delta_{\mathrm{core}} \in \{2.50, 1.38, 0.93\}\;\mathrm{mm}$ ($35.5, 20.3, 20.3\;\mathrm{cells}/\lambda_{\mathrm{core}}$) across $\varepsilon_{\mathrm{core}} \in \{1, 10, 22\}$ ($n_{\mathrm{core}} \in \{4.51, 14.25, 21.13\}$) shows why the simple point-toroid power law $P_{\mathrm{rad}} \propto |T_z|^2$ breaks down when $\varepsilon_{\mathrm{core}} > 1$:
#    - Below the internal $\mathrm{TE}_{01\delta}$ cutoff at $\varepsilon_{\mathrm{core}} = 10$ ($2r/\lambda_{\mathrm{core}} = 0.356 < 0.5$, $f_{\mathrm{TE01}} = 1.05\;\mathrm{GHz} > f_0$), Faraday induction drives an axial electric field $E_z$ inside the dielectric core whose polarization current $\mathbf{J}_{\mathrm{pol}} = +\mathrm{i}\omega\varepsilon_0(\varepsilon_{\mathrm{core}}-1)\mathbf{E}$ creates a true primitive electric dipole $\mathbf{p}_{\mathrm{pol}} = \varepsilon_0(\varepsilon_{\mathrm{core}}-1)\int_{\mathrm{core}}\mathbf{E}\,\mathrm{d}V$ ($1.54e-14\;\mathrm{C\,m}$) in phase with the toroidal dipole $\mathbf{p}_{\mathrm{eff}}(\mathbf{T}_{\mathrm{tot}})$ ($7.61e-14\;\mathrm{C\,m}$). Their coherent sum $|\mathbf{p}_{\mathrm{pol}} + \mathbf{p}_{\mathrm{eff}}|^2$ raises $P_{\mathrm{rad}}$ by **$2.30\times$** ($4.732e-04\;\mathrm{W}$ vs hybrid dipole prediction $4.591e-04\;\mathrm{W}$, $3.0\%$ agreement) while $|T_z(\varepsilon_{\mathrm{core}}=10)| / |T_z(\varepsilon_{\mathrm{core}}=1)|$ rises by **$1.242\times$** ($1.296\times$ relative to the continuum quasi-static formula $T_z^{\mathrm{qs}}$).
#    - Above cutoff at $\varepsilon_{\mathrm{core}} = 22$ ($2r/\lambda_{\mathrm{core}} = 0.529 > 0.5$, $f_{\mathrm{TE01}} = 0.709\;\mathrm{GHz} < f_0$), the internal $\mathrm{TE}_{01\delta}$ standing wave reverses sign across the minor cross-section and traps wave energy inside the high-index ($n_{\mathrm{core}} = 21.1$) core, causing $P_{\mathrm{rad}}$ to drop to $2.458e-05\;\mathrm{W}$ ($0.119\times$ of $\varepsilon_{\mathrm{core}}=1$) while $|T_z(\varepsilon_{\mathrm{core}}=22)| / |T_z(\varepsilon_{\mathrm{core}}=1)| = 1.035$ ($1.080\times$ vs $T_z^{\mathrm{qs}}$).
#
# ![Magnetic toroid vs equivalent electric dipole wave animation](figures/magnetic_toroid_vs_dipole_wave.gif)

# %%
t5 = _study_data["t5"]
nf5 = t5["nearfield_vs_dipole"]
display(Image(filename=str(FIG_DIR / "06_equivalence_limits.png"), width=980))

print(
    f"Task 5 Equivalence Limits (from data/task_05_equivalence_limits.json):\n"
    f"  Near-zone (20-30 mm) vs Outer-zone (45-55 mm) |Delta E| / |E|   : {nf5['near_zone_20_to_30mm_e_rel_diff'] * 100:.2f}% -> {nf5['outer_zone_45_to_55mm_e_rel_diff'] * 100:.2f}% ({nf5['e_metric_reduction_factor']:.2f}x reduction)\n"
    f"  Near-zone (20-30 mm) vs Outer-zone (45-55 mm) |Delta Hy| / |Hy| : {nf5['near_zone_20_to_30mm_hy_rel_diff'] * 100:.2f}% -> {nf5['outer_zone_45_to_55mm_hy_rel_diff'] * 100:.2f}% ({nf5['hy_metric_reduction_factor']:.2f}x reduction)\n"
    f"  Hankel octupole ratio |h_3^(2)/h_1^(2)| at 25 mm vs 50 mm        : {nf5['hankel_h3_over_h1_at_25mm']:.2f} -> {nf5['hankel_h3_over_h1_at_50mm']:.2f} ({nf5['hankel_octupole_decay_25_to_50mm']:.2f}x decay)\n"
    f"  Core TE01 mode sweep (2r / lambda_core -> |T_z(eps)| / |T_z(eps=1)|, cells/lambda_core): "
    + ", ".join(
        f"eps={r['eps_core']:.0f} ({r['two_r_over_lambda_core']:.3f} -> {r['ratio_to_eps1']:.3f}x at {r['cells_per_lambda_core']:.1f} cells/lambda)"
        for r in t5["core_te01_mode_sweep"]
    )
)

# %% [markdown]
# ---
#
# ### 6. Radiation Efficiency, Core Loss Energy Bookkeeping & the Chu Bound (`task_06_efficiency_q_and_convergence.json`)
#
# Figure 7a–b benchmarks energy conservation, radiation efficiency $\eta_{\mathrm{rad}}$, and radiation quality factor $Q_{\mathrm{rad}}$ across four Lorentz damping rates $\gamma / 2\pi \in \{0.02, 0.10, 0.30, 1.00\}\;\mathrm{GHz}$ ($\tan\delta_m \in [4.02e-04, 2.01e-02]$):
# - **Exact Poynting energy bookkeeping:** On the staggered FIT grid, the source input power $P_{\mathrm{in}}$ (including the exact leapfrog half-step time-averaging factor $\cos(\omega_0\Delta t/2)$ between electric fields at $t^n$ and currents at $t^{n+1/2}$) matches the sum of radiated power $P_{\mathrm{rad}}$ and core magnetic dissipation $P_{\mathrm{loss,core}} = \frac{1}{2}\tilde{\omega}\mu_0\int \mu''|\mathbf{H}|^2\,\mathrm{d}V$ within **$0.004\%\text{–}0.075\%$** across all loss tangents! Shifting the source current phase by half a timestep in our mutation check causes the energy balance to fail with **$34\%\text{–}439\%$** error.
# - **Efficiency & Chu bound:** Because $Q_{\mathrm{rad}} \approx 2392$ (analytical $Q_{\mathrm{rad,th}} = 2349.7$ including $1/\pi$ and $F_1(k_0 R)^2$) exceeds the enclosing-sphere Chu bound $Q_{\mathrm{Chu}}(k_0 a_{\mathrm{sph}}) = 35.37$ by $67.7\times$, even a modest magnetic loss tangent $\tan\delta_m \sim 10^{-3}$ reduces $\eta_{\mathrm{rad}} = 1/(1 + Q_{\mathrm{rad}}\tan\delta_m)$ substantially, matching our closed-form analytical curve within $0.9\%\text{–}1.8\%$.

# %%
t6 = _study_data["t6"]
display(Image(filename=str(FIG_DIR / "07_efficiency_q_and_convergence.png"), width=980))

print("Task 6 Core Loss Sweep, Efficiency & Q (from data/task_06_efficiency_q_and_convergence.json):")
for rec in t6["loss_sweep"]:
    print(
        f"  gamma = {rec['gamma_ghz']:4.2f} GHz (tan_delta_m = {rec['tan_delta_m']:.2e}) : "
        f"P_in = {rec['p_in_w']:.4e} W, P_rad+P_loss = {rec['p_out_tot_w']:.4e} W "
        f"(closure err = {rec['energy_closure_rel_err'] * 100:.3f}%, mutated = {rec['energy_closure_mutated_rel_err'] * 100:.1f}%), "
        f"eta_sim = {rec['eta_rad_sim'] * 100:.2f}% vs eta_th = {rec['eta_rad_theory'] * 100:.2f}%, "
        f"Q_rad = {rec['q_rad_sim']:.1f} ({rec['q_ratio_over_chu_sim']:.1f}x Q_Chu)"
    )

# %% [markdown]
# ---
#
# ## Mesh Resolution Convergence Study
#
# Figure 7c and the table below report our 3-level spatial resolution study on the closed 8-winding dispersive magnetic toroid across three core resolutions that **all satisfy $\ge 4$ cells across $2r = 10\;\mathrm{mm}$ and $\ge 20\;\mathrm{cells}/\lambda_{\mathrm{core}}$**: `Level 1 (Nominal)` $\Delta_{\mathrm{core}} = 2.50\;\mathrm{mm}$ ($4.0$ cells across $2r$, $35.5\;\mathrm{cells}/\lambda_{\mathrm{core}}$, $49\times 49\times 35 = 84,035$ cells), `Level 2 (Medium)` $\Delta_{\mathrm{core}} = 2.00\;\mathrm{mm}$ ($5.0$ cells across $2r$, $44.4\;\mathrm{cells}/\lambda_{\mathrm{core}}$, $57\times 57\times 41 = 133,209$ cells), and `Level 3 (Fine)` $\Delta_{\mathrm{core}} = 1.67\;\mathrm{mm}$ ($6.0$ cells across $2r$, $53.2\;\mathrm{cells}/\lambda_{\mathrm{core}}$, $63\times 63\times 44 = 174,636$ cells).
#
# **Behavior across the 3 grids vs the continuum static formula $T_z^{\mathrm{qs}} = 1.1271e-06\;\mathrm{A\,m^3}$:**
# 1. **Non-monotonic binary drive staircase ($\pm 9\%$) and lack of grid convergence:** On an unsmoothed binary staircase of the circular cross-section $(\rho - R)^2 + z^2 \le r^2$ and 8 angular sectors, the discrete static drive integral $T_z^{\mathrm{disc}}(\Delta) = \frac{1}{2}\operatorname{Re}(\mu_{\mathrm{disc}})\int (\mathbf{r}\times\mathbf{H}_{\mathrm{drive}})_z\,\mathrm{d}V$ shifts by $+7.21\%$ at $\Delta = 2.50\;\mathrm{mm}$ ($1.2083e-06\;\mathrm{A\,m^3}$), $+4.80\%$ at $\Delta = 2.00\;\mathrm{mm}$ ($1.1812e-06\;\mathrm{A\,m^3}$), and $-8.82\%$ at $\Delta = 1.67\;\mathrm{mm}$ ($1.0276e-06\;\mathrm{A\,m^3}$). Consequently, the simulated dynamic moment $|T_z^{\mathrm{sim}}|$ varies non-monotonically as **$1.254e-06 \to 1.265e-06 \to 1.096e-06\;\mathrm{A\,m^3}$** ($+11.28\%, +12.27\%, -2.75\%$ vs $T_z^{\mathrm{qs}}$) and the radiated power varies as **$P_{\mathrm{rad}} = 2.340e-04 \to 2.382e-04 \to 1.788e-04\;\mathrm{W}$** (**$-25\%$** on the finest grid vs Level 2, or $-23.6\%$ vs Level 1). Evaluating the stated grid-convergence criterion $|\Delta P_{\mathrm{rad}}|/P_{\mathrm{rad}} < 2.0\%$ between the last two levels gives $|-25.0\%| \ge 2.0\%$, so `is_grid_converged = False` is computed directly from the data, we do not apply Richardson extrapolation, and the $\mathcal{O}(\Delta_{\mathrm{core}}^2)$ guide in Figure 7c is shown as a reference slope only.
# 2. **Dynamic residual $|T_z^{\mathrm{sim}}|/T_z^{\mathrm{disc}}$ vs analytic interior factor:** Relative to the discrete static drive integral $T_z^{\mathrm{disc}}(\Delta)$ on each grid, the simulated dynamic moment has residual ratio $|T_z^{\mathrm{sim}}(\Delta)| / T_z^{\mathrm{disc}}(\Delta) = 1.0379, 1.0712, 1.0666$ ($+3.8\%, +7.1\%, +6.7\%$). The **analytic estimate** for the fundamental interior Bessel enhancement inside the circular cross-section at $f_0 = 0.75\;\mathrm{GHz}$ is $1 + (k_c r)^2/8 = 1.016$ ($+1.6\%$), which **does not explain** the full $+4.9\%\text{–}8.3\%$ dynamic residual on these coarse cross-section grids. Meanwhile, on every grid level the internal multipole-to-far-field self-consistency $|P_{\mathrm{rad}}^{\mathrm{N2F}} - P_{\mathrm{dip}}(T_{\mathrm{sim}})| / P_{\mathrm{dip}}(T_{\mathrm{sim}})$ holds within **$0.20\%, 0.19\%, 0.17\%$**.

# %%
display(Markdown(_narr_md["convergence_table"]))

# %% [markdown]
# ---
#
# ## Quantitative Summary Table
#
# Every numerical entry in the Quantitative Summary table below is generated directly from `data/*.json` via `_helpers.build_quantitative_summary_markdown()` and verified automatically in `tests/test_075_magnetic_toroid.py`:
#
# | Physical Quantity / Validation Target | Analytical / Literature Target | 3D FIT Result (`wakis`) | Relative Error / Status |
# | :--- | :--- | :--- | :--- |
# | Baseline short electric dipole TM fraction ($|a_{10}|^2 / P_{\mathrm{VSH}}$) | 100.00% ($|b_1|/|a_1| = 0$) | 100.0000% ($|b_1|/|a_1| = 1.85e-07$) | 0.034% pattern $L_2$ |
# | Baseline small magnetic loop TE fraction ($|b_1|^2 / P_{\mathrm{VSH}}$) | 100.00% ($|a_1|/|b_1| = 0$) | 99.9922% ($|a_1|/|b_1| = 2.76e-07$) | 0.091% pattern $L_2$ |
# | Loop radiation scaling $P_{\mathrm{rad}}(15\;\mathrm{mm}) / P_{\mathrm{rad}}(10\;\mathrm{mm})$ | 5.0625 ($(a_2/a_1)^4$), 5.0313 ($J_1^2$) | 4.9695 (slope 3.96) | 1.84% vs $(a_2/a_1)^4$ (1.23% vs $J_1^2$) |
# | Closed toroid net magnetic dipole $|\int \mathbf{M}\,\mathrm{d}V| / \int |\mathbf{M}|\,\mathrm{d}V$ | 0.00 | 5.22e-07 | 5.22e-07 residual |
# | Dubovik–Tugushev identity $|T_{z,\mathrm{DT}}(\mathbf{J}_b)|$ vs $|T_{z,\mathrm{mag}}(\mathbf{M})|$ | 1.194649e-06 $\mathrm{A\,m^3}$ | 1.194649e-06 $\mathrm{A\,m^3}$ | 1.77e-16 |
# | Closed toroid TM electric dipole fraction ($|a_{10}|^2 / P_{\mathrm{VSH}}$) | 100.00% ($|b_1|/|a_1| = 0$) | 99.9998% ($|b_1|/|a_1| = 5.34e-06$) | 0.162% pattern $L_2$ |
# | Multipole-to-far-field self-consistency $P_{\mathrm{rad}}^{\mathrm{N2F}}$ vs $P_{\mathrm{dip}}(p_{\mathrm{eff}} = -\mathrm{i}k_0 T_{\mathrm{tot}}^{\mathrm{sim}} F_1/c_0)$ | 2.3448e-04 W (from simulated $T_{\mathrm{tot}}$) | 2.3400e-04 W (N2F), 2.3407e-04 W (Flux) | 0.20% self-consistency |
# | Fixed-$T$ power scaling exponent $\mathrm{d}\ln(P_{\mathrm{rad}}/|T_{\mathrm{tot}}|^2)/\mathrm{d}\ln\omega$ | 6.00 ($\omega^6$) | 6.00 | 0.04% |
# | Electric octupole scaling exponent $\mathrm{d}\ln(|a_{30}|/|a_{10}|)/\mathrm{d}\ln(k_0 R)$ | 2.00 ($(k_0 R)^2$) | 2.18 | 8.86% |
# | Square of 4 rods ($C_4$) $|b_1|/|a_1|$ and $|m=\pm 4|/|a_{10}|$ ($R_{\mathrm{sq}}=12.5\;\mathrm{mm}$) | $|b_1|/|a_1| = 0$, $C_4$ residual $> 0$ | $|b_1|/|a_1| = 8.66e-06$, $|m=\pm 4|/|a_{10}| = 6.12e-05$ | 99.9999% in $a_{10}$ |
# | Broken loop: 7 of 8 feeds $|b_1|/|a_1|$ vs closed 8/8 feeds | $|b_1|/|a_1| \sim \mathcal{O}(1)$ (Valero 2024) | 0.4898 (vs 5.34e-06 closed) | 9.17e+04$\times$ jump |
# | Helical 1-turn axial $m_z$ vs Marinov compensating loop $|b_1|/|a_1|$ | Uncompensated $> 0.1$, compensated $\to 0$ | 0.5098 $\to$ 1.30e-03 | 3.92e+02$\times$ suppression |
# | Consistent near-field vs outer-zone error $|\Delta\mathbf{E}|/|\mathbf{E}|$ ($20\text{–}30\;\mathrm{mm}$ vs $45\text{–}55\;\mathrm{mm}$) | Outer $h_3^{(2)}/h_1^{(2)}$ octupole decays 4.51$\times$ (91.82 $\to$ 20.34) | 42.99% $\to$ 12.95% ($|\Delta H_y|/|H_y|$: 46.02% $\to$ 3.99%) | 3.32$\times$ ($|\mathbf{E}|$), 11.52$\times$ ($|H_y|$) reduction |
# | Energy bookkeeping $|P_{\mathrm{in}} - (P_{\mathrm{rad}} + P_{\mathrm{loss,core}})| / P_{\mathrm{in}}$ ($\gamma = 0.02\;\mathrm{GHz}$) | 0.00% (Poynting theorem) | 0.075% (mutated: 118.5%) | 0.075% |
# | Toroid radiation quality factor $Q_{\mathrm{rad}}$ vs Chu bound $Q_{\mathrm{Chu}}(k_0 a_{\mathrm{sph}})$ | $Q_{\mathrm{rad,th}} = 2349.7$, $Q_{\mathrm{Chu}} = 35.37$ | $Q_{\mathrm{rad,sim}} = 2392.5$ (67.7$\times Q_{\mathrm{Chu}}$) | 1.82% |
# | 3-grid resolution study ($\Delta_{\mathrm{core}} = 2.50, 2.00, 1.67\;\mathrm{mm}$; 4–6 cells/$2r$) | $T_z^{\mathrm{qs}} = 1.127e-06\;\mathrm{A\,m^3}$, $P_{\mathrm{rad}} = P_{\mathrm{dip}}(T_{\mathrm{sim}})$ | $|T_z^{\mathrm{sim}}| = 1.254e-06 \to 1.265e-06 \to 1.096e-06$; $P_{\mathrm{rad}} = 2.340e-04 \to 2.382e-04 \to 1.788e-04$ W | Not grid-converged ($\pm 9\%$ drive staircase, -25% $P_{\mathrm{rad}}$ on fine); 0.17–0.20% N2F self-consistency |

# %%
display(Markdown(h.build_quantitative_summary_markdown(data_dir=DATA_DIR)))

# %% [markdown]
# ---
#
# ## Limits & Failure Modes
#
# Every physical equivalence and quasi-static approximation relied upon in this tutorial has strict validity conditions that we tested with explicit counter-cases:
#
# 1. **Assumption 1 — Closed, balanced azimuthal magnetization loop ($\nabla\cdot\mathbf{M} = 0$, $C_{Nv}$ feed symmetry with $N \ge 4$):**
#    - *Condition:* Vanishing of the magnetic dipole moment ($\mathbf{m} = \mathbf{0}$) and of all $\mathrm{TE}$ multipoles ($b_{1m} = 0$) requires both an unbroken magnetic flux path and balanced in-phase excitation around the perimeter.
#    - *Counter-cases tested (Task 4, Figure 5):* Cutting a $60^\circ$ air gap in the permeable core creates magnetic poles on the cut faces and drives $|b_1|/|a_1| = 2.525$; disconnecting just 1 of 8 feeds [3] raises $|b_1|/|a_1|$ from $5.34e-06$ to $0.4898$; reversing the phase on half the torus (`Mode 1`) cancels $a_1$ to $1.584e-06\times |a_1|_{\mathrm{closed}}$ while driving a transverse magnetic dipole $|b_1| = 2.223\times |a_1|_{\mathrm{closed}}$; removing 1 of 4 rods in the square raises $|b_1|/|a_1|$ from $8.66e-06$ to $3.598$; and winding the poloidal coil as a 3D helical wire on edges without Marinov's reverse equatorial loop [1, 11] leaves a 1-turn azimuthal current with $|b_1|/|a_1| = 0.5098$, which passes through a sharp minimum of $1.30e-03$ ($392\times$ suppression, and $19.6\times$ lower than at $R = 15.0\;\mathrm{mm}$) when compensated by a planar $z = 0$ return wire loop of radius $R_1 = R\sqrt{1 + \frac{1}{2}(r/R)^2} = 15.41\;\mathrm{mm}$ driven by the same current $-I_{\mathrm{turn}}$.
# 2. **Assumption 2 — Far-field observation ($r \gg R$) vs reactive near-field topology ($r \sim R$):**
#    - *Condition:* Outside the bounding sphere $r > a_{\mathrm{sph}} = R + r$, the toroid field equals the full dipole field (including its $1/r^3$ and $1/r^2$ terms) plus odd-$l \ge 3$ electric multipoles ($a_{30}, a_{50}, \dots$).
#    - *Counter-case tested (Task 5, Figure 6a–b):* Inside and near the bounding sphere ($r \in [20, 30]\;\mathrm{mm}$), the point electric dipole diverges as $1/r^3$ toward the origin while the finite torus ring stays bounded at $r = 0$ and carries strong octupole near fields $\propto h_3^{(2)}(k_0 r) / h_1^{(2)}(k_0 r)$ (**$91.82$** at $25\;\mathrm{mm}$ vs **$20.34$** at $50\;\mathrm{mm}$, a **$4.51\times$** decay), producing a **$42.99\%$** near-zone $|\mathbf{E}|$ difference that decays by **$3.32\times$** to **$12.95\%$** in the outer reactive/intermediate zone $r \in [45, 55]\;\mathrm{mm}$ (**$46.02\% \to 3.99\%$** in $|H_y|$, an **$11.52\times$** reduction).
# 3. **Assumption 3 — Electrically thin core cross-section ($2r \ll \lambda_{\mathrm{core}} = \lambda_0 / \sqrt{\varepsilon_{\mathrm{core}}\mu_r}$) and electrically small ring ($k_0 R \ll 4.4934$):**
#    - *Condition:* Marinov's quasi-static toroidal moment formula $T_z \approx \frac{1}{2}\pi N I \mu_{\mathrm{eff}} r^2 R$ assumes uniform in-phase magnetization across the core cross-section, and the dipole radiation formula $P_{\mathrm{rad}} \propto |T_z|^2$ assumes $\varepsilon_{\mathrm{core}} = 1$ and $k_0 R \ll 1$.
#    - *Counter-cases tested (Task 5, Figure 6c):* When $\varepsilon_{\mathrm{core}} = 10$ ($2r/\lambda_{\mathrm{core}} = 0.356 < 0.5$, resolved at $20.3\;\mathrm{cells}/\lambda_{\mathrm{core}}$), induced dielectric polarization currents create a primitive electric dipole $\mathbf{p}_{\mathrm{pol}}$ in phase with $\mathbf{p}_{\mathrm{eff}}(\mathbf{T}_{\mathrm{tot}})$, raising $P_{\mathrm{rad}}$ by $2.30\times$ while $|T_z(\varepsilon_{\mathrm{core}})|/|T_z(\varepsilon_{\mathrm{core}}=1)|$ rises by $1.242\times$; when $\varepsilon_{\mathrm{core}} = 22$ ($2r/\lambda_{\mathrm{core}} = 0.529 > 0.5$, resolved at $20.3\;\mathrm{cells}/\lambda_{\mathrm{core}}$), crossing the internal $\mathrm{TE}_{01\delta}$ cutoff traps wave energy inside the high-index core so $P_{\mathrm{rad}}$ drops to $0.119\times$ of $\varepsilon_{\mathrm{core}}=1$ while $|T_z(\varepsilon_{\mathrm{core}}=22)|/|T_z(\varepsilon_{\mathrm{core}}=1)| = 1.035$. Likewise, when the ring electrical radius reaches the first zero of $j_1(k_0 R)$ at **$k_0 R = 4.4934$** ($3 j_1(k_0 R)/(k_0 R) = 0$), the $\mathrm{TM}_{10}$ dipole radiation vanishes completely and higher odd electric multipoles ($a_{30}, a_{50}$) take over.
# 4. **Assumption 4 — High-$\mu$, low-loss magnetic core well below ferromagnetic resonance ($f_0 \ll f_{\mathrm{res}}$, Snoek's limit) and sub-cell geometry representation:**
#    - *Condition:* Because $Q_{\mathrm{rad}}^{(\mathrm{toroid})} / Q_{\mathrm{Chu}} \propto [\mu'(k_0 R)^2]^{-1} \gg 1$ ($Q_{\mathrm{rad}} = 2392.5$ vs $Q_{\mathrm{Chu}} = 35.37$ at $f_0 = 0.75\;\mathrm{GHz}$), achieving high radiation efficiency $\eta_{\mathrm{rad}} = (1 + Q_{\mathrm{rad}}\tan\delta_m)^{-1}$ requires $\tan\delta_m \ll Q_{\mathrm{rad}}^{-1} \approx 4\times 10^{-4}$.
#    - *Counter-cases tested (Task 6, Figure 7a,c):* In a commercially available NiZn ferrite or ferromagnetic alloy laminate operating near GHz frequencies, Snoek's law [8] links higher static permeability $\mu_{\mathrm{DC}}$ to a lower ferromagnetic resonance frequency $f_{\mathrm{res}}$ and rising magnetic loss tangent $\tan\delta_m = \mu''/\mu' \gtrsim 10^{-2}$, which reduces $\eta_{\mathrm{rad}}$ to $2.04\%$ at $\gamma / 2\pi = 1.00\;\mathrm{GHz}$ ($\tan\delta_m = 2.01e-02$). Furthermore, on Cartesian grids with 4–6 cells across $2r$, the unsmoothed binary staircase of the circular core cross-section oscillates by $\pm 9\%$ in volume, causing $|T_z^{\mathrm{sim}}|$ and $P_{\mathrm{rad}}$ ($-25\%$ on the finest grid vs Level 2) not to converge monotonically without sub-cell material/source averaging. Likewise, in Task 4 the 3D helical coil and Marinov return loop are modelled as infinitely thin current filaments deposited on primary grid edges (`deposit_closed_wire_on_edges`, no physical wire diameter is modelled, so the $\ge 4$-cells-per-wire rule does not apply; the compensated residual floor $|b_1|/|a_1| = 1.30e-03$ is set by grid-edge deposition and staircase discretization error).
#
# ---
#
# ## References
#
# 1. **K. Marinov, A. D. Boardman, V. A. Fedotov, and N. Zheludev**, "Toroidal metamaterial," *New Journal of Physics* **9**(9), 324 (2007), [`doi:10.1088/1367-2630/9/9/324`](https://doi.org/10.1088/1367-2630/9/9/324) — Derived the quasi-static toroidal dipole moment of a wound air-core toroidal solenoid ($T = \pi N I d R^2 / 2$, Eq. 6), its parasitic axial magnetic dipole ($m_z = \pi I d^2 [1 + \frac{1}{2}(R/d)^2]$, Eq. 9), and the reverse equatorial compensating loop at $R_1 = d\sqrt{1 + \frac{1}{2}(R/d)^2}$ (Eq. 9 and Fig. 1 caption).
# 2. **V. M. Dubovik and V. V. Tugushev**, "Toroid moments in electrodynamics and solid-state physics," *Physics Reports* **187**(4), 145–202 (1990), [`doi:10.1016/0370-1573(90)90042-Z`](https://doi.org/10.1016/0370-1573(90)90042-Z) — Comprehensive theoretical formulation of toroidal multipole moments $\mathbf{T}_{\mathrm{DT}}(\mathbf{J})$ and their far-field equivalence to electric multipoles.
# 3. **A. Canós Valero, D. Borovkov, M. Sidorenko, P. Dergachev, E. Gurvitz, L. Gao, V. Bobrovs, A. Miroshnichenko, and A. S. Shalin**, "On the Existence of Pure Broadband Toroidal Sources in Electrodynamics," *Laser & Photonics Reviews* **18**(4), 2200740 (2024), [`doi:10.1002/lpor.202200740`](https://doi.org/10.1002/lpor.202200740) — Analyzed $N=8$ poloidal loop rings (loop diameter $5.5\;\mathrm{mm}$, fields at $1000\;\mathrm{MHz}$, horn rated $\ge 0.7\;\mathrm{GHz}$) and showed that removing one loop (7 of 8 feeds) breaks azimuthal symmetry and makes the magnetic dipole comparable to the toroidal dipole.
# 4. **L. J. Chu**, "Physical Limitations of Omni-Directional Antennas," *Journal of Applied Physics* **19**(12), 1163–1175 (1948), [`doi:10.1063/1.1715038`](https://doi.org/10.1063/1.1715038) — Established the fundamental minimum radiation quality factor $Q_{\mathrm{Chu}} = (ka)^{-3} + (ka)^{-1}$ of an electrically small $\mathrm{TM}_{10}$ or $\mathrm{TE}_{10}$ radiator.
# 5. **H. A. Wheeler**, "Fundamental Limitations of Small Antennas," *Proceedings of the IRE* **35**(12), 1479–1484 (1947), [`doi:10.1109/JRPROC.1947.226199`](https://doi.org/10.1109/JRPROC.1947.226199) — Derived the radiation power factor and $(ka)^3$ volume scaling of electrically small capacitive and inductive antennas.
# 6. **J. D. Jackson**, *Classical Electrodynamics*, 3rd ed., John Wiley & Sons, New York (1999), Chapter 9 — Vector spherical harmonic multipole expansion of localized current and magnetization distributions.
# 7. **Ya. B. Zel'dovich**, "Electromagnetic interaction with parity violation," *Soviet Physics JETP* **6**(6), 1184–1186 (1958) — Original introduction of the polar-vector toroidal (anapole) moment generated by poloidal currents on a torus.
# 8. **J. L. Snoek**, "Dispersion and absorption in magnetic ferrites at frequencies above one Mc/s," *Physica* **14**(4), 207–217 (1948), [`doi:10.1016/0031-8914(48)90038-X`](https://doi.org/10.1016/0031-8914(48)90038-X) — Established the fundamental relation between static susceptibility and ferromagnetic resonance frequency in permeable materials.
# 9. **V. Savinov, V. A. Fedotov, and N. I. Zheludev**, "Toroidal dipolar excitation and macroscopic electromagnetic properties of metamaterials," *Physical Review B* **89**(20), 205112 (2014), [`doi:10.1103/PhysRevB.89.205112`](https://doi.org/10.1103/PhysRevB.89.205112) — Linked microscopic toroidal dipole moments to macroscopic electromagnetic scattering and radiation.
# 10. **T. Yousefi and R. E. Diaz**, "A First-Order Model of the Multiple-Feed Toroidal Magneto-Dielectric Antenna," *IEEE Transactions on Antennas and Propagation* **65**(11), 5796–5803 (2017), [`doi:10.1109/TAP.2017.2748184`](https://doi.org/10.1109/TAP.2017.2748184) — Formulated the Mode 0 (in-phase axial electric dipole / monopole) and Mode 1 (quadrature circularly polarized magnetic loop / dipole) impedance and radiation models of multi-feed permeable toroidal antennas.
# 11. **K. T. McDonald**, "Electromagnetic Fields of a Small Helical Toroidal Antenna," Princeton University Physics Note (Dec. 8, 2008) — Derived the 3D magnetic field and parasitic axial magnetic dipole of a helically wound toroidal solenoid and its cancellation by a counter-wound return path.
# 12. **A. D. Boardman, K. Marinov, N. Zheludev, and V. A. Fedotov**, "Dispersion properties of nonradiating configurations: Finite-difference time-domain modeling," *Physical Review E* **72**(3), 036603 (2005), [`doi:10.1103/PhysRevE.72.036603`](https://doi.org/10.1103/PhysRevE.72.036603) — FDTD modeling of toroidal solenoids and coaxial electric dipoles confirming far-field equivalence and nonradiating anapole interference.
