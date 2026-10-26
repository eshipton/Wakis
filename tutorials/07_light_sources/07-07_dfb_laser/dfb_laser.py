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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eshipton/Wakis/blob/main/tutorials/07_light_sources/07-07_dfb_laser/dfb_laser.ipynb)
#
# # Tutorial 7.7 · DFB laser with a quarter-wave phase shift
#
# **Abstract.** Distributed feedback (DFB) lasers replace the discrete end mirrors
# of a Fabry–Pérot resonator with a periodic refractive-index grating of pitch
# $\Lambda = \lambda_0 / (2 n_0)$ etched along the active waveguide core. Whereas
# a uniform index-coupled grating exhibits a photonic stopband centered at the
# Bragg frequency $\omega_B = \pi c / (n_0 \Lambda)$ and oscillates simultaneously
# on two degenerate modes at the upper and lower stopband edges (**Kogelnik & Shank
# 1971, 1972** [1, 2]), introducing a central **quarter-wave ($\lambda/4$) phase shift**
# ($\Phi = \pi/2$ one-way, $\Delta\phi_{\text{rt}} = 2\Phi = \pi$ round-trip) creates a
# localized defect state at the exact center of the stopband ($\delta_0 L = 0$) with
# significantly lower threshold gain (**Haus & Shank 1976** [3]). In this tutorial,
# building on [Tutorial 6.1 · Sidewall-corrugated waveguide Bragg grating](../../06_periodic_structures/06-01_bragg_grating/bragg_grating.ipynb)
# and the active gain media module (`wakis.gain.SaturableGain`, `FourLevelGain`, and
# `examples/012_fabry_perot_laser.py`), we derive the coupled-wave transfer matrix
# and transcendental eigenvalue equations, extract sub-threshold complex resonance
# poles in 1D and 2D $\text{TE}_0$ slab-waveguide FIT cavities using `harminv`
# [6], demonstrate single-mode lasing clamping, longitudinal spatial hole burning
# (SHB), and Rigrod L–I slope efficiency above threshold, and verify three physical
# mutation checks ($\Phi \neq \pi/2$ mode splitting, $\kappa \to -\kappa$ standing-wave
# parity reversal, and gain-off passive decay).
#
# ---
#
# ## Learning goals
#
# By the end of this tutorial you will be able to:
#
# 1. **Formulate** the contra-directional **Coupled-Wave Theory (CWT)** transfer
#    matrix and transcendental threshold equations of Kogelnik & Shank (1972) [2]
#    and Haus & Shank (1976) [3] for uniform ($\Phi = 0$), quarter-wave-shifted
#    ($\Phi = \pi/2$), and general phase-shifted ($\Phi \in [0, \pi]$) index-coupled
#    DFB lasers.
# 2. **Extract** complex cavity resonances $(\omega_m, \gamma_m)$ as a function of
#    small-signal linear gain $g_0$ in 1D and 2D $\text{TE}_0$ slab-waveguide FIT
#    simulations via filter diagonalization (`wakis.resonances.harminv`), identifying
#    the lasing threshold as the zero crossing $\gamma_m(g_{\text{th}}) = 0$ across
#    coupling strengths $\kappa L \in [0.5, 4.0]$.
# 3. **Quantify** second-order $O(N_\Lambda^{-2})$ spatial convergence in cells per
#    Bragg period $N_\Lambda = \Lambda / \Delta z \in \{12, 16, 24, 32, 48\}$
#    ($24\text{--}96\;\text{cells}/\lambda$) with Richardson extrapolation.
# 4. **Simulate** above-threshold nonlinear lasing dynamics with `SaturableGain`
#    seeded by a weak pulse, contrasting single-mode oscillation ($\text{SMSR} = 40.0\;\text{dB}$
#    at $\delta_0 L = +0.0135\;\text{rad}$) and central spatial hole burning in the
#    $\lambda/4$-shifted DFB laser against two-mode stopband-edge beating in the
#    uniform DFB laser, and benchmark the L–I slope efficiency against coupled-wave
#    Rigrod rate-equation theory.
# 5. **Falsify** broken resonator symmetries through three explicit mutation checks:
#    non-quarter-wave phase shifts $\Phi \neq \pi/2$ ($\pi/4$ vs $3\pi/4$), coupling
#    sign reversal $\kappa \to -\kappa$, and passive decay when gain is turned off ($g_0 = 0$).
#
# ---
#
# ## 1. Theory
#
# ### 1.1 Coupled-wave theory of index-coupled DFB resonators (Kogelnik & Shank 1972)
#
# Consider a single-transverse-mode dielectric waveguide of background effective
# index $n_0$ whose refractive index and field amplitude gain along $z \in [-L/2, +L/2]$
# are modulated with first-order Bragg period $\Lambda = \lambda_0 / (2 n_0) = \pi / \beta_0$
# (Kogelnik & Shank 1972 [2], Eqs. 2–5):
#
# $$n(z) = n_0 + n_1 \cos\!\bigl(2\beta_0 z + \phi(z)\bigr), \qquad \alpha(z) = \alpha = \frac{g}{2},$$
#
# where $\beta_0 = n_0 \omega_B / c = 2\pi n_0 / \lambda_0 = \pi / \Lambda$ is the Bragg
# wavenumber, $n_1 \ll n_0$ is the index-modulation amplitude, $\phi(z)$ is a piecewise-constant
# grating phase, and $\alpha$ [$\text{m}^{-1}$] is the modal amplitude gain coefficient
# (half the modal power gain $g$). Assuming the $e^{+i\omega t}$ phasor convention of
# Kogelnik & Shank (1972) [2], we decompose the transverse electric field into
# forward-traveling ($R$) and backward-traveling ($S$) slowly varying envelopes:
#
# $$E_x(z, t) = \operatorname{Re}\!\left\{\left[R(z)\,e^{-i\beta_0 z} + S(z)\,e^{+i\beta_0 z}\right]e^{+i\omega t}\right\}.$$
#
# Substituting into the 1D Helmholtz equation $\mathrm{d}^2 E_x / \mathrm{d}z^2 + k^2(z) E_x = 0$
# with $k^2(z) \approx \beta^2 - 2i\alpha\beta_0 + 4\beta_0 \kappa \cos(2\beta_0 z + \phi)$
# and retaining phase-matched synchronous terms ($\left|\mathrm{d}^2 R / \mathrm{d}z^2\right| \ll \beta_0 \left|\mathrm{d}R / \mathrm{d}z\right|$)
# yields the contra-directional **coupled-wave equations** (Kogelnik & Shank 1972 [2], Eqs. 12–13):
#
# $$\frac{\mathrm{d}R}{\mathrm{d}z} - (\alpha - i\delta) R = -i\kappa e^{-i\phi} S, \qquad
# -\frac{\mathrm{d}S}{\mathrm{d}z} - (\alpha - i\delta) S = -i\kappa e^{+i\phi} R,$$
#
# where the **index-coupling coefficient** $\kappa$ and **Bragg wavenumber detuning** $\delta$ are:
#
# $$\kappa = \frac{\pi n_1}{\lambda_0} = \frac{n_1}{2 n_0 \Lambda}, \qquad
# \delta = \beta - \beta_0 = \frac{(\omega - \omega_B) n_0}{c}.$$
#
# #### Segment transfer matrix (**derived here**)
#
# Differentiating the coupled-wave equations on a segment of constant grating phase
# $\phi$ shows that both $R(z)$ and $S(z)$ satisfy $\mathrm{d}^2 \psi / \mathrm{d}z^2 - \gamma_c^2 \psi = 0$
# with complex propagation constant
#
# $$\gamma_c^2 = \kappa^2 + q^2, \qquad q \equiv \alpha - i\delta.$$
#
# Integrating from $z_1$ to $z_2 = z_1 + \ell$ gives the exact $2 \times 2$ **coupled-wave
# transfer matrix** $M(\ell; \kappa, \alpha, \delta, \phi)$ (**derived here**, implemented in
# `cwt_transfer_matrix`):
#
# $$\begin{pmatrix} R(z_2) \\ S(z_2) \end{pmatrix}
# = \begin{pmatrix}
# \cosh(\gamma_c \ell) + \dfrac{q}{\gamma_c}\sinh(\gamma_c \ell) &
# -i\dfrac{\kappa e^{-i\phi}}{\gamma_c}\sinh(\gamma_c \ell) \\[8pt]
# +i\dfrac{\kappa e^{+i\phi}}{\gamma_c}\sinh(\gamma_c \ell) &
# \cosh(\gamma_c \ell) - \dfrac{q}{\gamma_c}\sinh(\gamma_c \ell)
# \end{pmatrix}
# \begin{pmatrix} R(z_1) \\ S(z_1) \end{pmatrix}, \qquad \det M = 1.$$
#
# #### Uniform DFB grating ($\Phi = 0$): stopband and two-mode degeneracy
#
# For a uniform grating of length $L$ ($\phi = 0$ everywhere) with reflectionless
# facets ($R(-L/2) = 0$ and $S(+L/2) = 0$), self-sustained oscillation requires
# $M_{22}(L) = 0$:
#
# $$\gamma_c L \cosh(\gamma_c L) - (\alpha L - i\delta L)\sinh(\gamma_c L) = 0.$$
#
# Squaring $\alpha L - i\delta L = \gamma_c L \coth(\gamma_c L)$ and using
# $(\alpha L - i\delta L)^2 - (\gamma_c L)^2 = -(\kappa L)^2$ with $\coth^2 u - 1 = \operatorname{csch}^2 u$
# eliminates $\alpha L - i\delta L$ and yields the compact transcendental form of
# Kogelnik & Shank (1972) [2], Eq. (32):
#
# $$\frac{\sinh(\gamma_c L)}{\gamma_c L} = (-1)^{m-1}\,\frac{-i}{\kappa L}, \qquad
# \alpha_m L - i\delta_m L = \gamma_c L \coth(\gamma_c L), \qquad m = \pm 1, \pm 2, \dots$$
#
# Because this equation is invariant under $\delta \to -\delta$, **every longitudinal
# mode order $m$ of a uniform index-coupled DFB laser forms a symmetric pair at
# $\pm|\delta_m L|$ outside the Bragg stopband $|\delta L| \le \kappa L$ with identical
# threshold gain $\alpha_{\text{th},m} L$**:
#
# * At $\delta = 0$ (the exact Bragg wavelength), $\gamma_c = \sqrt{\kappa^2 + \alpha^2} > \alpha > 0$
#   is purely real, so $\gamma_c \coth(\gamma_c L) > \gamma_c > \alpha$: **no mode can
#   oscillate at $\delta = 0$ in a uniform index-coupled grating**.
# * At $\kappa L = 2.0$, the lowest-threshold doublet ($m = \pm 1$) occurs at
#   $\alpha_{\text{th},1} L = 0.98467$ and $\delta_{\text{th},1} L = \pm 3.37972\;\text{rad}$.
# * **High-$\kappa L$ asymptotic expansion (derived here)**: For $\kappa L \gg \pi$,
#   $\sinh(\gamma_c L) \approx i\gamma_c L / (\kappa L) \to 0$ implies $\gamma_c L \approx i m\pi + \epsilon$
#   with $\epsilon \approx (-1)^m i (m\pi) / [(-1)^m (\kappa L)] = i m\pi / (\kappa L)$ on the
#   imaginary part and real part $\operatorname{Re}(\gamma_c L) \approx (m\pi)^2 / (\kappa L)^2$.
#   Expanding $\alpha_{\text{th},m} L - i\delta_{\text{th},m} L = \sqrt{(\gamma_c L)^2 - (\kappa L)^2}$
#   yields (**derived here**):
#
# $$\delta_{\text{th},m} L \simeq \pm\sqrt{(\kappa L)^2 + (m\pi)^2}, \qquad
# \alpha_{\text{th},m} L \simeq \left(\frac{m\pi}{\kappa L}\right)^2 \qquad (\kappa L \gg \pi).$$
#
# * **Gain-dispersive CWT threshold & Kramers–Kronig mode pulling (derived here)**:
#   When the active medium has a finite homogeneous Lorentzian linewidth $\gamma_a$ centered
#   at $\omega_a = \omega_B$ (`wakis.gain.FourLevelGain`), the complex susceptibility replaces
#   the frequency-independent gain $\alpha L$ in $q L = \alpha L - i\delta L$ by
#   $$q_{\text{disp}}(\delta L; \alpha_0 L)\,L = \frac{\alpha_0 L}{1 + i(\delta L / \delta_\gamma L)} - i\delta L,
#   \qquad \delta_\gamma L \equiv \frac{\gamma_a n_0 L}{2 c_0}.$$
#   Satisfying the non-reflecting boundary condition $M_{22}(q_{\text{disp}} L, \kappa L) = 0$
#   requires $q_{\text{disp}} L = \alpha_{\text{th},m}^{\text{CWT}} L - i\delta_{\text{th},m}^{\text{CWT}} L$.
#   Equating real and imaginary parts yields the exact closed-form **gain-pulled stopband detuning**
#   and **line-center threshold gain** (**derived here**, implemented in `solve_cwt_uniform_dispersive`):
#   $$\delta_{\text{th},m}^{\text{disp}} L = \frac{\delta_{\text{th},m}^{\text{CWT}} L}{1 + \dfrac{\alpha_{\text{th},m}^{\text{CWT}} L}{\delta_\gamma L}},
#   \qquad \alpha_{0,\text{th},m}^{\text{disp}} L = \alpha_{\text{th},m}^{\text{CWT}} L\!\left[1 + \left(\frac{\delta_{\text{th},m}^{\text{disp}} L}{\delta_\gamma L}\right)^2\right].$$
#   For the Part D cavity ($L = 30\Lambda = 7.5\;\mu\text{m}$, $n_0 = 3.0$, $\kappa L = 2.0$,
#   $\gamma_a / (2\pi) = 12.0\;\text{THz}$, $\delta_\gamma L = 2.8294\;\text{rad}$), the
#   dispersionless stopband edges $\delta_{\text{th},1}^{\text{CWT}} L = \pm 3.3797\;\text{rad}$
#   ($\pm 7.17\;\text{THz}$) sit where the Lorentzian gain factor is $0.4121$ (which would be
#   below threshold at $r_p = 2.0$), whereas Kramers–Kronig reactive dispersion pulls the
#   doublet inward to $\delta_{\text{th},1}^{\text{disp}} L = \pm 2.5072\;\text{rad}$
#   ($\pm 5.32\;\text{THz}$), where the Lorentzian factor is $0.5602$ and the threshold pump
#   ratio is $r_{p,\text{th}}^{\text{disp}} = \alpha_{0,\text{th},1}^{\text{disp}} / \alpha_{\text{th},1}^{\text{CWT}} = 1.7852 < 2.0$.
#
# ---
#
# ### 1.2 Quarter-wave-shifted ($\lambda/4$) DFB resonator (Haus & Shank 1976)
#
# To break the two-mode stopband-edge degeneracy of the uniform grating, **Haus & Shank (1976)** [3]
# introduced a discrete phase jump at the cavity midplane $z = 0$:
#
# $$\phi(z) = \begin{cases} -\Phi, & -L/2 \le z < 0, \\ +\Phi, & 0 < z \le +L/2, \end{cases}$$
#
# where $\Phi = \pi/2$ corresponds to a physical **quarter-wave ($\lambda_B / 4 = \Lambda / 2$)
# shift**, producing a round-trip phase slip $\Delta\phi_{\text{rt}} = 2\Phi = \pi$ between the
# left and right grating halves. Multiplying the two half-cavity transfer matrices
# $M_{\text{tot}} = M(L/2; +\Phi)\,M(L/2; -\Phi)$ (**derived here**):
#
# $$M_{\text{tot},22} = \left[\cosh\!\left(\tfrac{1}{2}\gamma_c L\right) - \frac{q}{\gamma_c}\sinh\!\left(\tfrac{1}{2}\gamma_c L\right)\right]^2
# + \frac{\kappa^2 e^{+2i\Phi}}{\gamma_c^2}\sinh^2\!\left(\tfrac{1}{2}\gamma_c L\right) = 0.$$
#
# Taking the square root gives the general **phase-shifted DFB dispersion relation** (**derived here**,
# solved in `solve_cwt_phase_shifted`):
#
# $$\gamma_c \cosh\!\left(\tfrac{1}{2}\gamma_c L\right) - (\alpha - i\delta)\sinh\!\left(\tfrac{1}{2}\gamma_c L\right)
# = \pm i\kappa e^{+i\Phi}\sinh\!\left(\tfrac{1}{2}\gamma_c L\right).$$
#
# #### Exact Bragg defect mode at $\Phi = \pi/2$ (**derived here**)
#
# When $\Phi = \pi/2$ ($i e^{+i\pi/2} = -1$), the right-hand side is purely real for the
# $+$ branch, admitting a non-degenerate fundamental defect mode ($m = 0$) at **exact zero
# detuning** $\delta_0 L = 0$ with $\gamma_0 = \sqrt{\kappa^2 + \alpha_0^2}$ satisfying the
# real scalar equation (**derived here**, solved in `solve_cwt_qws`):
#
# $$\gamma_0 L \cosh\!\left(\tfrac{1}{2}\gamma_0 L\right) - (\alpha_0 L + \kappa L)\sinh\!\left(\tfrac{1}{2}\gamma_0 L\right) = 0
# \iff \coth\!\left(\tfrac{1}{2}\gamma_0 L\right) = \frac{\alpha_0 + \kappa}{\sqrt{\kappa^2 + \alpha_0^2}}.$$
#
# * At $\kappa L = 2.0$, the $\lambda/4$-shifted fundamental mode has $\alpha_{\text{th},0} L = 0.69713$
#   at $\delta_0 L = 0$, whereas the nearest side modes ($m = \pm 1$) require $\alpha_{\text{th},1} L = 1.42977$
#   at $\delta_1 L = \pm 5.15766\;\text{rad}$—providing a substantial threshold gain margin
#   $\Delta(\alpha_{\text{th}} L) = 0.73263$.
# * **High-$\kappa L$ asymptotic expansion (derived here)**: Using $\coth(\gamma_0 L / 2) \approx 1 + 2 e^{-\gamma_0 L}$
#   and $(\alpha_0 + \kappa)/\sqrt{\kappa^2 + \alpha_0^2} \approx 1 + \alpha_0/\kappa - \alpha_0^2/(2\kappa^2)$
#   for $\alpha_0 \ll \kappa$ gives (**derived here**):
#
# $$\alpha_{\text{th},0} L \simeq 2\kappa L\,e^{-\kappa L} \qquad (\kappa L \gg 1).$$
#
# #### Standing-wave parity under $\kappa \to -\kappa$ (**derived here**)
#
# At the phase-shift plane $z = 0$ ($\Phi = \pi/2$, $\delta_0 = 0$), the left half-cavity
# reflection relation sets $S(0) = \pm R(0)$ depending on the sign of $\kappa$. Because
# the total electric field at $z = 0$ is $E_x(0) \propto R(0) + S(0)$:
#
# * For $\kappa > 0$ ($\Delta\varepsilon(z) = -2 n_0 n_1 \sin(2\beta_0 |z|)$), $S(0) = +R(0)$,
#   so $z = 0$ is a constructive **electric-field antinode** ($|E_x(0)| = \max |E_x|$).
# * Flipping the coupling sign $\kappa \to -\kappa$ ($\Delta\varepsilon(z) = +2 n_0 n_1 \sin(2\beta_0 |z|)$)
#   reverses the relative phase to $S(0) = -R(0)$, turning $z = 0$ into an **electric-field node**
#   ($E_x(0) = 0$); a symmetric current sheet at $z = 0$ then cannot couple to the $m = 0$ mode
#   and excites only the higher-threshold even-parity side mode!
#
# ---
#
# ### 1.3 2D $\text{TE}_0$ slab-waveguide coupled-mode reduction (**derived here**)
#
# In a 2D dielectric slab waveguide of core thickness $d_{\text{core}}$, core index
# $n_{\text{core}}$, and cladding index $n_{\text{clad}} < n_{\text{core}}$, the fundamental
# $\text{TE}_0$ mode $E_y(x, z, t) = \psi(x)\,[R(z)e^{-i\beta_0 z} + S(z)e^{+i\beta_0 z}]e^{+i\omega t}$
# satisfies the transverse eigenproblem $\mathrm{d}^2\psi / \mathrm{d}x^2 + k_0^2 [n^2(x) - n_{\text{eff}}^2]\psi = 0$
# with optical confinement factor
#
# $$\Gamma_{\text{core}} = \frac{\int_{-d_{\text{core}}/2}^{+d_{\text{core}}/2} |\psi(x)|^2\,\mathrm{d}x}{\int_{-\infty}^{+\infty} |\psi(x)|^2\,\mathrm{d}x}.$$
#
# When both the index modulation $n_{1,\text{core}}$ and the active linear power gain $g_{0,\text{core}}$
# are confined to the core $|x| < d_{\text{core}} / 2$, projecting the 2D Helmholtz equation
# onto $\psi(x)$ (Yariv 1973 [4]; Wang *et al.* 2014 [5]) yields the effective 1D CWT parameters
# (**derived here**):
#
# $$\kappa = \frac{\pi}{\lambda_0}\frac{n_{\text{core}}}{n_{\text{eff}}}\,\Gamma_{\text{core}}\,n_{1,\text{core}}, \qquad
# \alpha_{\text{modal}} = \frac{1}{2}\frac{n_{\text{core}}}{n_{\text{eff}}}\,\Gamma_{\text{core}}\,g_{0,\text{core}}, \qquad
# \delta = \frac{n_g (\omega - \omega_B)}{c},$$
#
# where $n_g = n_{\text{eff}} + \omega\,\mathrm{d}n_{\text{eff}}/\mathrm{d}\omega$ is the waveguide group index.
#
# ---
#
# ### 1.4 Longitudinal mode profile, spatial hole burning & Rigrod L–I slope (**derived here**)
#
# Multiplying the CWT equations at threshold ($\delta = \delta_{\text{th}}, \alpha = \alpha_{\text{th}}$)
# by $R^*$ and $S^*$ and taking the real part yields the differential Poynting flux law
# (**derived here**):
#
# $$\frac{\mathrm{d}}{\mathrm{d}z}\!\left(|R(z)|^2 - |S(z)|^2\right) = 2\alpha_{\text{th}}\!\left(|R(z)|^2 + |S(z)|^2\right).$$
#
# Integrating over $z \in [-L/2, +L/2]$ with reflectionless boundaries $R(-L/2) = S(+L/2) = 0$
# and symmetric facets $|R(+L/2)|^2 = |S(-L/2)|^2 = I_{\text{out}}$ gives the exact
# **Poynting energy-balance identity** (**derived here**, verified in `cwt_mode_profile`):
#
# $$\alpha_{\text{th}} L \int_{-1/2}^{+1/2} \psi_{\text{env}}(u)\,\mathrm{d}u = 1, \qquad
# \psi_{\text{env}}(u) \equiv \frac{|R(u L)|^2 + |S(u L)|^2}{|R(+L/2)|^2}, \qquad u = \frac{z}{L}.$$
#
# For the $\lambda/4$-shifted mode ($\delta_0 = 0$), the normalized intracavity envelope is
# $\psi_{\text{env}}(u) = \cosh[\gamma_0 L (1 - 2|u|)] + (\alpha_0 / \gamma_0)\sinh[\gamma_0 L (1 - 2|u|)]$,
# which peaks sharply at the central phase shift $u = 0$ with peak-to-edge ratio
# $I_{\text{env}}(0) / I_{\text{out}} = 2.8689$ at $\kappa L = 2.0$.
#
# Above threshold ($g_0 > g_{\text{th}} = 2\alpha_{\text{th}}$) in a homogeneously broadened
# gain medium with saturation intensity $I_{\text{sat}}$ (`wakis.gain.SaturableGain`,
# $g(I) = g_0 / (1 + I / I_{\text{sat}})$), substituting the saturated gain into the energy
# balance yields the **coupled-wave Rigrod L–I equation** (**derived here**, implemented in
# `cwt_rigrod_output`):
#
# * **Homogeneous modal clamping (`spatial_averaging=True`, derived here)**: When carrier diffusion
#   or modal spatial averaging homogenizes the gain saturation across $z \in [-L/2, +L/2]$, the
#   spatially averaged intracavity intensity clamps at $\langle I \rangle_L / I_{\text{sat}} = r_p - 1$.
#   Combining this with the Poynting energy-balance identity
#   $\langle I \rangle_L / I_{\text{out}} = \int_{-1/2}^{+1/2}\psi_{\text{env}}(u)\,\mathrm{d}u = (\alpha_{\text{th},0} L)^{-1}$
#   gives the exact rate-equation single-facet output intensity (**derived here**):
#   $$\frac{I_{\text{out}}^{\text{savg,th}}}{I_{\text{sat}}} = \alpha_{\text{th},0} L\,(r_p - 1).$$
# * **Envelope approximation** (averaging over the rapid $2\beta_0 z$ standing wave first):
#   $$\int_{-1/2}^{+1/2} \frac{(g_0 / g_{\text{th}})\,\psi_{\text{env}}(u)}{1 + (I_{\text{out}} / I_{\text{sat}})\,\psi_{\text{env}}(u)}\,\mathrm{d}u
#   = \int_{-1/2}^{+1/2} \psi_{\text{env}}(u)\,\mathrm{d}u.$$
#   Expanding to first order near threshold ($g_0 / g_{\text{th}} - 1 \ll 1$) gives the
#   linearized spatial-hole-burning slope efficiency (**derived here**):
#   $$\frac{I_{\text{out}}}{I_{\text{sat}}} \simeq \eta_{\text{SHB}}\!\left(\frac{g_0}{g_{\text{th}}} - 1\right), \qquad
#   \eta_{\text{SHB}} = \frac{\int_{-1/2}^{+1/2} \psi_{\text{env}}(u)\,\mathrm{d}u}{\int_{-1/2}^{+1/2} \psi_{\text{env}}^2(u)\,\mathrm{d}u}.$$
# * **Full standing-wave grating saturation (derived here)**: In the exact Maxwell–Bloch /
#   FIT field, the intensity $I(u, \theta) / I_{\text{out}} = \psi_{\text{env}}(u) + 2|R(u)||S(u)|\cos\theta$
#   oscillates at period $\Lambda$ between standing-wave nodes and antinodes. Averaging the
#   saturated emission $I(u, \theta) / [1 + I(u, \theta) / I_{\text{sat}}]$ over $\theta \in [0, 2\pi)$
#   analytically via $\frac{1}{2\pi}\int_0^{2\pi} \frac{\mathrm{d}\theta}{A + B\cos\theta} = \frac{1}{\sqrt{A^2 - B^2}}$
#   gives the exact standing-wave Rigrod integral (**derived here**):
#
# $$\int_{-1/2}^{+1/2} \frac{g_0}{g_{\text{th}}}\frac{1}{x_{\text{out}}}\!\left[
# 1 - \frac{1}{\sqrt{\bigl(1 + x_{\text{out}}\psi_{\text{env}}(u)\bigr)^2 - \bigl(2 x_{\text{out}}|R(u)||S(u)|\bigr)^2}}
# \right]\mathrm{d}u = \int_{-1/2}^{+1/2} \psi_{\text{env}}(u)\,\mathrm{d}u, \qquad
# x_{\text{out}} \equiv \frac{I_{\text{out}}}{I_{\text{sat}}}.$$
#
# ---
#
# ## 2. Setup
#
# We model a telecom-wavelength ($\lambda_0 = 1.50\;\mu\text{m}$, $f_B = 199.86\;\text{THz}$)
# index-coupled semiconductor DFB cavity in both **1D** (`Nx = Ny = 1`, periodic transverse
# boundaries, $N_{\text{periods}} = 40$, $L = 10.0\;\mu\text{m}$, $N_\Lambda = 24\;\text{cells}/\Lambda = 48\;\text{cells}/\lambda$,
# $12\;\text{cells}$ per $\Lambda/2$ tooth) and **2D $\text{TE}_0$ slab-waveguide**
# ($d_{\text{core}} = 320\;\text{nm}$, $n_{\text{core}} = 3.0$, $n_{\text{clad}} = 2.4$,
# $\Delta x = 20\;\text{nm}$, $16\;\text{cells}$ across the core, $20\;\text{cells}/\Lambda$)
# configurations. Index-matched Mur absorbing boundary conditions (`bc_low=[..., 'abc']`,
# `bg=[n0**2, 1.0, 0.0]`) terminate the $\pm z$ facets with zero facet reflectivity
# ($R_{\text{facet}} = 0$). In `build_1d_dfb_solver` and `build_2d_slab_dfb_solver`, the grating
# permittivity is modulated directly as $\varepsilon(z) = \varepsilon_{0,\text{grat}} + 2 n_0 n_1\,\text{mod}(z)$
# (omitting the second-harmonic $n_1^2\,\text{mod}(z)^2$ term of $n(z)^2 = [n_0 + n_1\,\text{mod}(z)]^2$).
# Because `SolverFIT3D` stores inverse permittivities `ieps` on Yee edges, the 1D harmonic-mean
# effective permittivity of $\varepsilon_{0,\text{grat}} + 2 n_0 n_1 \cos(2\beta_0 z)$ is
# $\langle \varepsilon^{-1} \rangle^{-1} = \sqrt{\varepsilon_{0,\text{grat}}^2 - 4 n_0^2 n_1^2} \approx \varepsilon_{0,\text{grat}} - 2 n_1^2$
# (whereas modulating $n(z)$ directly would give $\langle n^{-2} \rangle^{-1} \approx n_0^2 [1 - 0.5(n_1/n_0)^2]$),
# and second-order Bloch–Floquet self-coupling plus finite-$N_\Lambda$ Yee dispersion shifts the
# discrete stopband center downward by $\approx 1.5(n_1/n_0)^2$. Setting the grating background
# permittivity to $\varepsilon_{0,\text{grat}} = n_0^2 [1 + 1.5(n_1/n_0)^2]$ is therefore an
# empirical second-order Bragg-center calibration for the truncated $\varepsilon(z)$ profile that
# keeps the discrete $m = 0$ defect resonance aligned with $f_B = c / \lambda_0$ across $\kappa L \in [0.5, 4.0]$.
#
# ![1D DFB laser cavity setup geometry](figures/00a_setup_geometry.png)
# ![1D DFB laser mesh discretization and excitation pulse](figures/00b_setup_mesh_and_excitation.png)
#
# The table below summarizes the 1D index-coupled DFB laser cavity setup (`figures/00_setup_table.md`):
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 521 nm × 521 nm × 10.3 µm (z ∈ [-5.17 µm, 5.17 µm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 1 × 1 × 992 (992 cells) |
# | Discretization | Δz & resolution | Δ_min = 10.4 nm, 48.0 cells/λ (in n_max = 3.00), 12.0 cells across feature (125 nm) |
# | Boundary | -x face | Periodic |
# | Boundary | +x face | Periodic |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | Mur ABC |
# | Boundary | +z face | Mur ABC |
# | Material | Background | Index-matched waveguide cladding (n₀ = 3.00, ε_r = 9.00) (isotropic) |
# | Material | Region 1: Left DFB grating (z ∈ [-L/2, 0], φ = -π/2, SaturableGain) | ε_r = 9.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-5 µm, 0 m] (d = 5 µm) |
# | Material | Region 2: Right DFB grating (z ∈ [0, +L/2], φ = +π/2, λ/4 shift at z = 0) | ε_r = 9.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [0 m, 5 µm] (d = 5 µm) |
# | Source | 1 · CurrentSource | pos: z = 0 m, pol: J_x (E_x), pulse: f₀ = 199.9 THz, Δf = 15.99 THz |
# | Monitor | 2 · ResonanceMonitor | ResonanceMonitor (z = -1.67 µm), pos: z = -1.67 µm, freqs: time-domain |
# | Monitor | 3 · Probe | Facet output probe z_out, pos: z = 5.08 µm, freqs: time-domain |
# | Variation | Run variation | Case=λ/4-shifted DFB (Φ = π/2), Grating phase=φ(z) = sign(z) π/2, Gain model=Linear (I_sat = ∞) & Saturable (I_sat = 10 MW/m²), Mode behavior=Single Bragg mode at δL = 0 |
# | Variation | Run variation | Case=Uniform DFB (Φ = 0), Grating phase=φ(z) = 0, Gain model=Linear (I_sat = ∞) & Saturable (I_sat = 10 MW/m²), Mode behavior=Degenerate stopband doublet at ±|δ₁L| |
#
# ![2D TE slab-waveguide DFB setup geometry](figures/00a_setup_geometry_2d.png)
# ![2D TE slab-waveguide DFB mesh and excitation](figures/00b_setup_mesh_and_excitation_2d.png)
#
# The table below summarizes the 2D $\text{TE}_0$ slab-waveguide DFB check setup (`figures/00_setup_table_2d.md`):
#
# | Category | Component / Face | Specification |
# | :--- | :--- | :--- |
# | Domain | Size & bounds | Lx × Ly × Lz = 800 nm × 691 nm × 8.62 µm (z ∈ [-4.31 µm, 4.31 µm]) |
# | Domain | Grid cells | Nx × Ny × Nz = 40 × 1 × 624 (24960 cells) |
# | Discretization | Δz & resolution | Δ_min = 13.8 nm, 36.2 cells/λ (in n_max = 3.00), 10.0 cells across feature (138.2 nm) |
# | Boundary | -x face | PEC |
# | Boundary | +x face | PEC |
# | Boundary | -y face | Periodic |
# | Boundary | +y face | Periodic |
# | Boundary | -z face | Mur ABC |
# | Boundary | +z face | Mur ABC |
# | Material | Background | Dielectric waveguide cladding (n_clad = 2.40, ε_r = 5.76) (isotropic) |
# | Material | Region 1: Corrugated active slab core (n_core = 3.0, d = 320 nm, φ = -π/2) | ε_r = 9.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [-4.15 µm, 0 m] (d = 4.15 µm) |
# | Material | Region 2: Corrugated active slab core (n_core = 3.0, φ = +π/2, λ/4 shift at z = 0) | ε_r = 9.00, μ_r = 1.00, model = Isotropic dielectric, z ∈ [0 m, 4.15 µm] (d = 4.15 µm) |
# | Source | 1 · CurrentSource | pos: z = 0 m, pol: J_y (E_y), pulse: f₀ = 199.9 THz, Δf = 15.99 THz |
# | Monitor | 2 · ResonanceMonitor | ResonanceMonitor (z = -1.38 µm), pos: z = -1.38 µm, freqs: time-domain |

# %%
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _helpers  # noqa: E402

# %% [markdown]
# ---
#
# ## 3. Results
#
# ### Part A · Linear gain below & at threshold (1D and 2D slab-waveguide `harminv` vs CWT)
#
# In the small-signal regime ($I \ll I_{\text{sat}}$, modeled with `SaturableGain(I_sat=np.inf)`),
# a cavity resonance with passive radiation decay rate $\gamma_{m,0} = \alpha_{\text{th},m} c / n_0$
# experiences a linear reduction in net damping as the modal amplitude gain $\alpha = g_0 / 2$
# increases:
#
# $$\gamma_m(g_0) = \left(\alpha_{\text{th},m} - \frac{g_0}{2}\right)\frac{c}{n_0}.$$
#
# Lasing threshold ($\gamma_m = 0$) occurs when the modal amplitude gain balances the distributed
# grating transmission loss, $\alpha = \alpha_{\text{th},m}$. Figure 1 compares the `harminv`
# decay-rate zero crossing at $\kappa L = 2.0$ and the full $\kappa L \in [0.5, 4.0]$ dispersion
# sweep against Kogelnik & Shank (1972) [2] and Haus & Shank (1976) [3]:
#
# * **1D $\lambda/4$-shifted DFB ($\kappa L = 2.0$)**: Passive `harminv` gives
#   $\alpha_{\text{th},0} L = 0.6967$ vs CWT $0.6971$ ($0.06\%$ relative error) and linear-gain
#   zero-crossing $\alpha_{\text{th},0} L = 0.6932$ ($0.57\%$ relative error, accounting for the
#   finite Lorentz gain linewidth $\gamma = 1.5\omega_B$), with defect mode detuning
#   $\delta_0 L = -0.0111\;\text{rad}$ vs exact $0$.
# * **1D uniform DFB ($\kappa L = 2.0$)**: `harminv` resolves the degenerate stopband-edge
#   doublet ($m = \pm 1$) at $\alpha_{\text{th},1} L = 0.9837$ vs CWT $0.9847$ ($0.10\%$ error)
#   and $|\delta_{\text{th},1} L| = 3.3759\;\text{rad}$ vs CWT $3.3797\;\text{rad}$ ($0.11\%$ error).
# * **Across the full sweep $\kappa L \in [0.5, 4.0]$**: 1D FIT reproduces both curves with a
#   maximum relative error of $1.25\%$ ($\lambda/4$-shifted at $\kappa L = 4.0$, where $\alpha_{\text{th},0} L = 0.1518$
#   is small) and $0.38\%$ (uniform grating), and approaches the asymptotic laws
#   $\alpha_{\text{th},0} L \sim 2\kappa L e^{-\kappa L}$ and $\alpha_{\text{th},1} L \sim (\pi / \kappa L)^2$
#   for $\kappa L \gtrsim 2.5$.
# * **2D $\text{TE}_0$ slab-waveguide DFB check ($\kappa L = 2.0$)**: The discrete transverse
#   eigenmode has $n_{\text{eff}} = 2.71360$, group index $n_g = 3.09990$, and core confinement
#   factor $\Gamma_{\text{core}} = 0.81848$. Sweeping active core gain $g_{0,\text{core}}$ in
#   2D FIT yields a `harminv` zero crossing at modal threshold gain $\alpha_{\text{th},0} L = 0.7196$
#   vs 1D CWT $0.6971$ ($3.22\%$ difference due to 2D waveguide group-index dispersion $n_g > n_{\text{eff}}$)
#   and detuning $\delta_0 L = -0.0587\;\text{rad}$. Figure 2 shows the 2D standing-wave defect
#   mode $E_y(x, z)$ localized at $z = 0$ and confined within the $320\;\text{nm}$ core.
#
# ![Linear-gain harminv zero crossing and CWT threshold gain and detuning vs kappa L](figures/01_cwt_and_linear_gain.png)
# ![2D TE0 slab-waveguide DFB defect mode field map and transverse confinement](figures/02_2d_slab_mode.png)

# %%
cwt_summary = _helpers.run_task_00_cwt_and_linear_gain(quick=False, force=False)
print(
    f"1D QWS (kL=2): CWT alpha_th*L = {cwt_summary['cwt_qws_kL2_alpha_L']:.4f}, "
    f"FIT = {cwt_summary['fit_qws_kL2_alpha_L']:.4f} "
    f"(err = {cwt_summary['rel_err_qws_kL2_alpha_pct']:.2f}%)"
)
print(
    f"1D Uniform (kL=2): CWT alpha_th*L = {cwt_summary['cwt_uni_kL2_alpha_L']:.4f}, "
    f"FIT = {cwt_summary['fit_uni_kL2_alpha_L']:.4f} "
    f"(err = {cwt_summary['rel_err_uni_kL2_alpha_pct']:.2f}%), "
    f"|delta_1*L| = {cwt_summary['fit_uni_kL2_delta_L']:.4f} vs {cwt_summary['cwt_uni_kL2_delta_L']:.4f}"
)
print(
    f"2D TE0 Slab (kL=2): n_eff = {cwt_summary['slab_2d_n_eff']:.4f}, "
    f"Gamma_core = {cwt_summary['slab_2d_Gamma_core']:.4f}, "
    f"alpha_th*L = {cwt_summary['slab_2d_alpha_th_L']:.4f} "
    f"(err = {cwt_summary['slab_2d_rel_err_alpha_pct']:.2f}%)"
)

# %% [markdown]
# ### Part B · Grid convergence in cells per period $N_\Lambda$ & Richardson extrapolation
#
# Figure 3 shows the grid convergence of $\alpha_{\text{th}} L$ and $\delta_{\text{th}} L$ at
# $\kappa L = 2.0$ across five spatial resolutions $N_\Lambda = \Lambda / \Delta z \in \{12, 16, 24, 32, 48\}$
# cells per Bragg period (corresponding to $24, 32, 48, 64, 96$ cells per wavelength in $n_0 = 3.0$
# and $6, 8, 12, 16, 24$ cells per $\Lambda / 2$ half-period tooth) with constant physical cladding
# buffer $L_{\text{pad}} = 1.0\,\Lambda$:
#
# * Both the $\lambda/4$-shifted threshold gain ($p_{\text{obs}} = 1.98$) and the uniform-grating
#   threshold gain ($p_{\text{obs}} = 2.00$) exhibit clean second-order $O(N_\Lambda^{-2})$
#   convergence toward their Richardson-extrapolated continuum limits ($\alpha_{\text{th},0}^{\text{Rich}} L = 0.69831$,
#   $\alpha_{\text{th},1}^{\text{Rich}} L = 0.98564$).
# * At the finest grid ($N_\Lambda = 48$, $96\;\text{cells}/\lambda$), the 1D FIT threshold gains
#   $\alpha_{\text{th},0} L = 0.69728$ and $\alpha_{\text{th},1} L = 0.98487$ agree with
#   analytical CWT ($0.69713$ and $0.98467$) to $0.020\%$ and $0.021\%$, respectively, while the
#   $\lambda/4$ defect detuning converges to $|\delta_0 L| = 0.0065\;\text{rad}$ ($p_{\text{obs}} = 2.63$).
#
# ![Log-log grid convergence in cells per Bragg period with O(N_Lambda^-2) guides](figures/03_convergence.png)

# %%
conv_summary = _helpers.run_task_01_convergence(quick=False, force=False)
for cpp, cpw, qa, qd, ua, ud in zip(
    conv_summary["cpp_list"],
    conv_summary["cpw_list"],
    conv_summary["qws_alpha_L"],
    conv_summary["qws_delta_L"],
    conv_summary["uni_alpha_L"],
    conv_summary["uni_delta_L"],
):
    print(
        f"N_Lambda = {cpp:2d} ({cpw:2d} cells/lam): "
        f"QWS alpha_th*L = {qa:.5f}, delta_0*L = {qd:+.5f} | "
        f"Uni alpha_th*L = {ua:.5f}, |delta_1*L| = {ud:.5f}"
    )

# %% [markdown]
# ### Part C · Above-threshold saturable-gain FIT lasing, spatial hole burning & L–I curve
#
# To model nonlinear gain clamping and spatial hole burning above threshold, we attach
# `wakis.gain.SaturableGain` with saturation intensity $I_{\text{sat}} = 10\;\text{MW/m}^2$
# across the grating region $z \in [-L/2, +L/2]$ and seed cavity build-up with a weak
# Gaussian pulse ($I_{\text{seed}} \approx 5 \times 10^{-3}\,I_{\text{sat}} \ll I_{\text{sat}}$).
#
# > **Documented time-scale separation (`tau_I` scaling):** In a $30$-period micron-scale
# > DFB cavity ($L = 7.5\;\mu\text{m}$, $n_0 = 3.0$), the optical cycle is $T_0 = \lambda_0 / c = 5.0\;\text{fs}$
# > and the passive cavity photon lifetime is $\tau_p = n_0 L / (2 c\,\alpha_{\text{th}} L) \approx 54\;\text{fs}$
# > ($\approx 11\,T_0$). Following `examples/012_fabry_perot_laser.py`, we set the local
# > intensity-averaging time constant to $\tau_I = 6\,T_0 = 30.0\;\text{fs}$ ($\gg T_0$, smoothing
# > out optical-frequency $2\omega_B$ carrier oscillations while allowing gain clamping within
# > $600\,T_0 = 3.0\;\text{ps}$) and use a broadband homogeneous Lorentz linewidth
# > $\gamma = 1.5\omega_B$ so that gain dispersion across the stopband $|\delta L| \le 4$ is
# > flat ($< 1\%$).
#
# Figures 4 and 5 and the animated build-up movie (`figures/07_dfb_lasing_buildup.gif`) reveal
# three textbook signatures of DFB laser physics:
#
# 1. **Single-mode clamping vs two-mode stopband beating (Figure 4)**: At pump ratio
#    $g_0 / g_{\text{th}} = 1.6$, the $\lambda/4$-shifted DFB laser clamps smoothly into a
#    continuous-wave single-frequency state at the center of the Bragg stopband
#    ($\delta_0 L = +0.0135\;\text{rad}$ vs CWT $0$, side-mode suppression ratio
#    $\text{SMSR} = 40.0\;\text{dB}$). By contrast, the uniform DFB grating—whose $m = \pm 1$
#    stopband-edge modes share identical threshold gain and identical spatial envelopes—oscillates
#    simultaneously on both stopband edges ($\delta_{-1} L = -3.4307\;\text{rad}$ and
#    $\delta_{+1} L = +3.1857\;\text{rad}$, half-splitting $3.3082\;\text{rad}$ vs CWT
#    $3.3797\;\text{rad}$, $2.12\%$ difference), producing persistent $100\%$ envelope beating
#    at $\Delta f = (\delta_{+1} - \delta_{-1}) c / (2\pi n_0 L) = 14.0\;\text{THz}$.
# 2. **Defect localization & spatial hole burning (Figure 5a and GIF)**: In the $\lambda/4$-shifted
#    cavity, the cycle-averaged intensity envelope peaks at the central phase shift $z = 0$ with
#    FIT peak-to-edge ratio $I_{\text{env}}(0) / I_{\text{out}} = 2.7570$ vs linear CWT
#    $2.8689$ ($3.90\%$ compression caused by nonlinear spatial hole burning, which locally
#    depletes the gain at $z = 0$ to $g(0)/g_0 = 0.5820$ vs $0.80$ near the facets).
# 3. **L–I curve vs CWT Rigrod theory (Figure 5b)**: Across $g_0 / g_{\text{th}} \in [0.5, 2.5]$,
#    the single-facet output intensity $I_{\text{out}} / I_{\text{sat}}$ is zero below threshold
#    ($g_0 / g_{\text{th}} \le 1$) and rises linearly above threshold with FIT slope efficiency
#    $\mathrm{d}(I_{\text{out}}/I_{\text{sat}})/\mathrm{d}(g_0/g_{\text{th}}) = 0.5264$ vs the
#    standing-wave CWT Rigrod slope $0.5745$ ($8.36\%$ difference, attributable to the finite
#    Lorentz gain pole dispersion and nonlinear index shift above threshold).
#
# ![Above-threshold lasing build-up and single-mode vs two-mode output spectra](figures/04_lasing_spectra.png)
# ![Longitudinal spatial hole burning profile and L-I curve vs CWT Rigrod theory](figures/05_shb_profile_and_li_curve.png)
# ![Animated GIF of intracavity lasing build-up and spatial hole burning](figures/07_dfb_lasing_buildup.gif)

# %%
las_summary = _helpers.run_task_02_lasing_dynamics(quick=False, force=False)
print(
    f"QWS lasing (r = {las_summary['r_main']}): peak delta_0*L = {las_summary['qws_peak_delta_L']:+.4f} rad, "
    f"SMSR = {las_summary['qws_smsr_db']:.1f} dB, "
    f"I_env(0)/I_out = {las_summary['qws_peak_to_edge_fit']:.4f} (CWT {las_summary['qws_peak_to_edge_cwt']:.4f})"
)
print(
    f"Uniform lasing (r = {las_summary['r_main']}): doublet peaks at "
    f"delta*L = {las_summary['uni_peak_neg_delta_L']:+.4f}, {las_summary['uni_peak_pos_delta_L']:+.4f} rad "
    f"(half-splitting = {las_summary['uni_half_splitting_delta_L']:.4f} vs CWT {las_summary['cwt_uni_delta_L']:.4f})"
)
print(
    f"L-I slope efficiency: FIT = {las_summary['slope_fit']:.4f} vs "
    f"CWT Rigrod+SHB = {las_summary['slope_rigrod_sw']:.4f} "
    f"(diff = {las_summary['slope_rel_diff_pct']:.2f}%)"
)

# %% [markdown]
# ---
#
# ### Part D · Self-started lasing from spontaneous-emission noise (`FourLevelGain`, $s \le 100$)
#
# In Part C, lasing was initiated deterministically by injecting a weak external seed pulse into a
# two-parameter `SaturableGain` medium with a broadband toy linewidth ($\gamma = 1.5\omega_B$) and an
# accelerated saturation time ($\tau_I = 6\,T_0$). A physical semiconductor or solid-state DFB laser
# instead self-starts from **quantum spontaneous-emission fluctuations** in a four-level active
# medium (`wakis.gain.FourLevelGain(spontaneous_emission=True, seed=..., time_scale=...)`; see
# `docs/physicsguide.md`, *"Noise and time scaling in gain media"*):
#
# 1. **Physical gain parameters & uniform lifetime scaling ($s \le 100$)**: We model a four-level
#    gain medium at $\omega_a = \omega_B = 2\pi \times 199.86\;\text{THz}$ with a realistic
#    homogeneous Lorentzian dephasing linewidth $\gamma_a / (2\pi) = 12.0\;\text{THz}$
#    ($\approx 0.06\,\omega_B$, dephasing time $\tau_2 = 2/\gamma_a = 26.5\;\text{fs}$, normalized
#    half-linewidth $\delta_\gamma L = 2.8294\;\text{rad}$), upper-state
#    lifetime $\tau_{21} = 15.0\;\text{ps}$ ($\approx 3000\,T_0$), fast lower-state depletion
#    $\tau_{10} = 20.0\;\text{fs}$, and pump relaxation $\tau_{32} = 10.0\;\text{fs}$ at pump ratio
#    $r_p = \Delta N_0 / \Delta N_{\text{th}} = 2.0$. Because $\tau_{21} \gg T_0$, we apply uniform
#    lifetime scaling (`time_scale = 100.0`, satisfying $s \le 100$), which scales all population
#    rates by $(\tau_{32}, \tau_{21}, \tau_{10}) \to (\tau_{32}/s, \tau_{21}/s, \tau_{10}/s)$ and
#    $W_p \to s W_p$ while leaving the Lorentzian susceptibility $\chi(\omega)$, threshold inversion
#    $\Delta N_{\text{th}}$, and steady-state clamped output intensity $I_{\text{out}} / I_{\text{sat}}$
#    strictly invariant (`docs/physicsguide.md`). Evaluating `gain.time_scaling_limits(tau_p, dt)`
#    at $s = 100$ shows:
#    * Condition 1 (cavity decay): $(s/\tau_{\text{sat}})/\kappa_{\text{cav}} = 0.3590 < 1$
#      and $(s/\tau_{21})/\kappa_{\text{cav}} = 0.3589 < 1$ ($s_{\text{max,cav}} = 278.54$).
#    * Condition 2 (dephasing adiabaticity): for the upper laser level $\tau_{21,\text{eff}} = 150\;\text{fs}$,
#      $(s/\tau_{21})/\gamma_a = 0.0884 < 1$ ($s_{\text{max,dephasing}} = 1130.97$), giving a
#      moderate $\gamma_a / \tau_{21,\text{eff}}^{-1} = 11.31\times$ separation between optical
#      polarization dephasing and upper-state decay. However, for the sub-20 fs non-radiative
#      levels ($\tau_{\min} = \tau_{32} = 10\;\text{fs}$), uniform scaling drives
#      $\tau_{32}/s = 0.10\;\text{fs}$ below $1/\gamma_a = 13.3\;\text{fs}$, so
#      $(s/\tau_{\min})/\gamma_a = 132.63 > 1$ ($s_{\text{max,fast}} = 0.7540$), while
#      Condition 3 (Crank–Nicolson positivity) remains satisfied with
#      $(s/\tau_{\min})\Delta t = 1.0780 < 2$ ($s_{\text{max,CN}} = 185.53$).
# 2. **Self-start from rest, inversion clamping & modal output intensity (Figure 8a)**: Starting
#    from identically zero electromagnetic fields ($E = H = 0$, `sources=[]`), Langevin
#    spontaneous-emission dipole noise ($\propto \sqrt{\max(N_2, 0)}$) seeds the cavity modes while
#    the population inversion rises toward its unsaturated value $\Delta N_0 = 2.0\,\Delta N_{\text{th}}$.
#    Once intracavity stimulated emission reaches $I_{\text{sat}}$, the spatially averaged modal
#    inversion (`spatial_averaging=True`) clamps at $\langle\Delta N\rangle / \Delta N_{\text{th}} = 0.9662$
#    vs theoretical $1.0000$ ($3.38\%$ error) and the single-facet output intensity reaches
#    $I_{\text{out}} / I_{\text{sat}} = 0.7438$ vs the homogeneous rate-equation steady state
#    $\alpha_{\text{th},0} L\,(r_p - 1) = 0.6971$ ($6.69\%$ relative difference), with a
#    noise-to-steady-state field amplification ratio of $22.49\times$ and a late-window envelope
#    clamping ratio $\text{RMS}_{\text{late},2} / \text{RMS}_{\text{late},1} = 1.0491$. Across
#    $s \in \{25, 50, 100\}$, the clamped inversion is $(1.0082, 0.9646, 0.9662)$ ($4.45\%$ spread)
#    and the clamped output intensity is $(0.7571, 0.7160, 0.7438)$ ($5.56\%$ spread; $2.71\%$ error
#    vs $0.6971$ at $s = 50$): at $s = 25$ ($\tau_{21,\text{eff}} = 600\;\text{fs}$,
#    $\gamma_a \tau_{21,\text{eff}} = 45.2$), the polarization is more adiabatic and clamps within
#    $+0.82\%$ of $\Delta N_{\text{th}}$, whereas at $s = 100$ ($\gamma_a \tau_{21,\text{eff}} = 11.3$)
#    the moderate $11.3\times$ dephasing separation and continuous spontaneous-emission driving shift
#    the clamped inversion $3.38\%$ below $\Delta N_{\text{th}}$.
# 3. **Spontaneous single-mode selection vs gain-pulled uniform doublet & carrier hole burning (Figures 8b,c)**:
#    Without any frequency-selective seed, the $\lambda/4$-shifted DFB laser selects the single
#    Bragg defect mode at $\delta_0 L = +0.0065\;\text{rad}$ with $\text{SMSR} = 28.95\;\text{dB}$
#    and clamped facet output $I_{\text{out}} / I_{\text{sat}} = 0.5184$ vs CWT standing-wave Rigrod
#    theory $0.5218$ ($0.64\%$ relative error). Under local cell-by-cell saturation
#    (`spatial_averaging=False`), the high defect intensity at $z = 0$ burns a spatial hole in the
#    period-averaged carrier inversion down to $\Delta N(0) / \Delta N_{\text{th}} = 1.0911$ while
#    leaving $\langle\Delta N\rangle / \Delta N_{\text{th}} = 1.2923$ across the cavity (reaching
#    $\approx 1.48\,\Delta N_{\text{th}}$ near the facets). By contrast, in the uniform DFB grating
#    ($\Phi = 0$) with Lorentzian half-linewidth $\delta_\gamma L = 2.8294\;\text{rad}$ ($6.0\;\text{THz}$
#    HWHM), the dispersionless CWT stopband edges at $\delta_{\text{th},1}^{\text{CWT}} L = \pm 3.3797\;\text{rad}$
#    ($\pm 7.17\;\text{THz}$) sit where the Lorentzian line factor is $0.4121$ (below threshold at
#    $r_p = 2.0$), but Kramers–Kronig reactive dispersion pulls the stopband modes inward to
#    $\delta_{\text{th},1}^{\text{disp}} L = \pm 2.5072\;\text{rad}$ ($\pm 5.32\;\text{THz}$, line
#    factor $0.5602$, threshold pump ratio $r_{p,\text{th}}^{\text{disp}} = 1.7852 < 2.0$, Section 1.1).
#    Consequently, at $r_p = 2.0 = 1.120\,r_{p,\text{th}}^{\text{disp}}$ the uniform DFB grating
#    oscillates on a **gain-pulled stopband doublet** at $\delta L = -2.5462$ and $+2.4283\;\text{rad}$
#    (half-splitting $|\delta_{\pm 1} L| = 2.4873\;\text{rad}$), matching the gain-dispersive CWT
#    prediction $\delta_{\text{th},1}^{\text{disp}} L = 2.5072\;\text{rad}$ within one spectral bin ($0.79\%$; bin width $0.0655\;\text{rad} = 2.6\%$, peaks asymmetric at $2.546$ / $2.428$)
#    ($-26.41\%$ below the dispersionless CWT edge $3.3797\;\text{rad}$).
#
# ![Self-started lasing from spontaneous-emission noise with FourLevelGain](figures/08_lasing_from_noise.png)

# %%
noise_summary = _helpers.run_task_04_lasing_from_noise(quick=False, force=False)
print(
    f"Noise QWS (s = {noise_summary['time_scale_main']:.0f}, r_p = {noise_summary['r_pump']}): "
    f"clamped <dN>/dN_th = {noise_summary['qws_savg_dn_clamped_over_th']:.4f} "
    f"(err = {noise_summary['qws_savg_dn_clamping_err_pct']:.2f}%), "
    f"modal-avg I_out/I_sat = {noise_summary['qws_savg_I_out_over_Isat']:.4f} vs theory {noise_summary['qws_savg_theory_I_out_over_Isat']:.4f} "
    f"(err = {noise_summary['qws_savg_vs_theory_rel_err_pct']:.2f}%), "
    f"SHB I_out/I_sat = {noise_summary['qws_shb_I_out_over_Isat']:.4f} vs Rigrod {noise_summary['qws_rigrod_sw_I_out_over_Isat']:.4f} "
    f"(err = {noise_summary['qws_shb_vs_rigrod_rel_err_pct']:.2f}%), SMSR = {noise_summary['qws_smsr_db']:.1f} dB"
)
print(
    f"Uniform DFB gain-pulled doublet (delta_gamma*L = {noise_summary['d_gamma_L']:.4f} rad, "
    f"r_th,disp = {noise_summary['cwt_uni_disp_r_th']:.4f}): "
    f"|delta_+-1*L| = {noise_summary['uni_half_splitting_delta_L']:.4f} rad vs gain-dispersive CWT "
    f"{noise_summary['cwt_uni_disp_delta_L']:.4f} rad (err = {noise_summary['uni_disp_delta_rel_err_pct']:.2f}%, "
    f"vs dispersionless CWT {noise_summary['cwt_uni_delta_L']:.4f} rad: {noise_summary['uni_cwt_undisp_delta_rel_diff_pct']:+.2f}%)"
)
print(
    f"Time-scale invariance s in {noise_summary['time_scale_list']}: "
    f"<dN>/dN_th = {noise_summary['time_scale_dn_clamped']} (spread {noise_summary['time_scale_dn_spread_pct']:.2f}%), "
    f"I_out/I_sat = {noise_summary['time_scale_I_out_over_Isat']} (spread {noise_summary['time_scale_I_out_spread_pct']:.2f}%)"
)
print("time_scaling_limits (s = 100):", noise_summary["time_scaling_limits"])

# %% [markdown]
# ---
#
# ### Part E · Diagnostic & mutation checks ($\Phi \neq \pi/2$ splitting, $\kappa \to -\kappa$ parity, gain off)
#
# To verify that every physical observable responds as predicted when the resonator symmetry,
# source-mode parity overlap, or gain coupling is altered, Figure 6 evaluates three explicit
# checks at $\kappa L = 2.0$:
#
# 1. **Check 1 — Non-quarter-wave phase shift ($\Phi \neq \pi/2$)**: Replacing the
#    $\lambda/4$ shift ($\Phi = 90^\circ$) with $\Phi = 45^\circ$ ($\lambda/8$) or $\Phi = 135^\circ$
#    ($3\lambda/8$) breaks the $\delta \to -\delta$ stopband symmetry and pushes the defect mode
#    off the Bragg wavelength to $\delta_{\text{th}} L = -1.9092\;\text{rad}$ (CWT $-1.8866\;\text{rad}$)
#    and $+1.8634\;\text{rad}$ (CWT $+1.8866\;\text{rad}$), respectively, with total splitting
#    $\Delta(\delta L) = 3.7726\;\text{rad}$ vs CWT $3.7732\;\text{rad}$ ($0.02\%$ error) and
#    elevated threshold gain $\alpha_{\text{th}} L = 0.7762$ vs $0.6967$ at $\Phi = 90^\circ$.
# 2. **Check 2 — Standing-wave parity & source-node selection under $\kappa \to -\kappa$**: The
#    CWT eigenvalue equation depends only on $\kappa^2$, so the fundamental $m = 0$ threshold
#    gain is invariant under $\kappa \to -\kappa$: exciting $\kappa < 0$ asymmetrically off center
#    recovers $\alpha_{\text{th},0} L = 0.6964$ at $\delta_0 L = -0.0112\;\text{rad}$, identical to
#    $\kappa > 0$ ($\alpha_{\text{th},0} L = 0.6967$). However, flipping $n_1 \to -n_1$ at
#    $\Phi = \pi/2$ reverses the relative reflection phase $S(0)/R(0)$ from $+1$ to $-1$,
#    converting the defect plane $z = 0$ from an electric-field antinode
#    ($|E_x(0)| / \max|E_x| = 1.0000$) into an odd-parity node ($|E_x(0)| / \max|E_x| = 0.0425$).
#    Consequently, a symmetric source placed at the $z = 0$ node has zero overlap with the
#    odd-parity $m = 0$ mode and selectively excites the even-parity $m = \pm 1$ first side-mode
#    doublet at $\alpha_{\text{th},1} L = 1.4074$ and $|\delta_{\text{th},1} L| = 5.0633\;\text{rad}$,
#    matching the analytical CWT side mode ($\alpha_{\text{th},1} L = 1.4298$,
#    $|\delta_{\text{th},1} L| = 5.1577\;\text{rad}$) to within $1.57\%$ and $1.83\%$.
# 3. **Check 3 — Gain off ($g_0 = 0$)**: Setting `gain_g0 = 0.0` eliminates stimulated
#    emission, causing the seed pulse to decay exponentially at rate $\gamma_0 = 6.96 \times 10^{12}\;\text{s}^{-1}$
#    down to $I_{\text{out}} / I_{\text{sat}} = 1.3 \times 10^{-18} \approx 0$ (no lasing).
#
# ![Diagnostic and mutation checks: non-quarter-wave phase shift splitting, kappa sign parity flip, and gain off decay](figures/06_mutations.png)

# %%
mut_summary = _helpers.run_task_03_mutations(quick=False, force=False)
print(
    f"Check 1 (Phi = 45 vs 135 deg): splitting Delta(delta*L) = "
    f"{mut_summary['phi_45_splitting_fit_delta_L']:.4f} rad vs CWT {mut_summary['phi_45_splitting_cwt_delta_L']:.4f} rad"
)
print(
    f"Check 2 (kappa sign parity at z=0): |E_x(0)|/max = {mut_summary['kappa_pos_center_field_norm']:.4f} (kappa > 0) "
    f"vs {mut_summary['kappa_neg_center_field_norm']:.4f} (kappa < 0); "
    f"asym-excited m=0 alpha*L = {mut_summary['kappa_neg_asym_alpha_L']:.4f}, "
    f"center-node-excited m=+-1 side mode = ({mut_summary['kappa_neg_center_excited_alpha_L']:.4f}, "
    f"{mut_summary['kappa_neg_center_excited_delta_L']:.4f}) vs CWT ({mut_summary['cwt_side_m1_alpha_L']:.4f}, {mut_summary['cwt_side_m1_delta_L']:.4f})"
)
print(
    f"Check 3 (gain off g0=0): steady-state I_out/I_sat = {mut_summary['gain_off_I_out_over_Isat']:.2e}"
)

# %% [markdown]
# ---
#
# ## 4. Quantitative Summary
#
# All numerical values in the table below are generated directly from `data/*.json` via
# `_helpers.format_summary_table()`:
#
# | Quantity | Theory / CWT | Wakis FIT | Rel. Error / Status |
# | :--- | :---: | :---: | :---: |
# | 1D $\lambda/4$-shifted threshold gain $\alpha_{\mathrm{th},0} L$ ($\kappa L = 2.0$, $N_\Lambda = 24$) | 0.6971 | 0.6967 | 0.06% |
# | 1D $\lambda/4$-shifted linear-gain zero-crossing $\alpha_{\mathrm{th},0} L$ ($\kappa L = 2.0$) | 0.6971 | 0.6932 | 0.57% |
# | 1D $\lambda/4$-shifted Bragg defect detuning $\delta_0 L$ [rad] ($\kappa L = 2.0$) | +0.0000 | -0.0111 | |Δ(δL)| = 0.0111 rad |
# | 1D $\lambda/4$-shifted first side-mode $\alpha_{\mathrm{th},1} L, |\delta_{\mathrm{th},1} L|$ ($\kappa L = 2.0$, $m = \pm 1$) | 1.4298, 5.1577 | 1.4074, 5.0633 | 1.57%, 1.83% |
# | 1D uniform DFB threshold gain $\alpha_{\mathrm{th},1} L$ ($\kappa L = 2.0$, $m = \pm 1$) | 0.9847 | 0.9837 | 0.10% |
# | 1D uniform DFB stopband-edge detuning $|\delta_{\mathrm{th},1} L|$ [rad] ($\kappa L = 2.0$) | 3.3797 | 3.3759 | 0.11% |
# | 2D TE$_0$ slab-waveguide $\lambda/4$ DFB $\alpha_{\mathrm{th},0} L$ ($\kappa L = 2.0$, $\Gamma = 0.818$) | 0.6971 | 0.7196 | 3.22% |
# | Finest-grid $\lambda/4$ DFB $\alpha_{\mathrm{th},0} L$ ($N_\Lambda = 48$, $96\;\mathrm{cells}/\lambda$) | 0.69713 | 0.69728 | 0.020% (order p = 1.98) |
# | Above-threshold $\lambda/4$ DFB lasing detuning $\delta_0 L$ [rad] ($g_0/g_{\mathrm{th}} = 1.6$) | +0.0000 | +0.0135 | Single-mode (SMSR = 40.0 dB) |
# | Above-threshold uniform DFB doublet $|\delta_{\pm 1} L|$ [rad] ($g_0/g_{\mathrm{th}} = 1.6$) | 3.3797 | 3.3082 | 2.12% (two-mode) |
# | Defect intensity localization $I_{\mathrm{env}}(0) / I_{\mathrm{out}}$ ($\kappa L = 2.0$) | 2.8689 | 2.7570 | 3.90% (SHB compression) |
# | L–I slope efficiency $\mathrm{d}(I_{\mathrm{out}}/I_{\mathrm{sat}})/\mathrm{d}(g_0/g_{\mathrm{th}})$ | 0.5745 | 0.5264 | 8.36% |
# | Lasing from noise (`FourLevelGain`, $s = 100$): clamped $\langle\Delta N\rangle / \Delta N_{\mathrm{th}}$ | 1.0000 | 0.9662 | 3.38% (self-started, SMSR = 28.9 dB) |
# | Lasing from noise (`FourLevelGain`, modal avg, $r_p = 2.0$): output $I_{\mathrm{out}} / I_{\mathrm{sat}}$ | 0.6971 | 0.7438 | 6.69% ($s = 50$: 0.7160) |
# | Lasing from noise (`FourLevelGain`, SHB, $r_p = 2.0$): output $I_{\mathrm{out}} / I_{\mathrm{sat}}$ | 0.5218 | 0.5184 | 0.64% ($s \in \{25,50,100\}$ spread 5.56%) |
# | Lasing from noise (`FourLevelGain`, uniform DFB): gain-pulled doublet $|\delta_{\pm 1} L|$ [rad] | 2.5072 | 2.4873 | within one bin (2.6 %; pulled from CWT 3.38) |
# | Mutation: phase shift $\Phi = \pi/4$ vs $3\pi/4$ splitting $\Delta(\delta L)$ [rad] | 3.7732 | 3.7726 | 0.02% |

# %%
print(_helpers.format_summary_table())

# %% [markdown]
# ---
#
# ## 5. Limits & Failure Modes
#
# 1. **Macroscopic Spatial Hole Burning (SHB) at Strong Coupling ($\kappa L \gg 3$)**:
#    Our linear CWT threshold analysis assumes a spatially uniform gain $\alpha(z) = \alpha_{\text{th}}$
#    along $z \in [-L/2, +L/2]$. In a $\lambda/4$-shifted DFB laser, the fundamental defect
#    mode intensity $I_{\text{env}}(z) \propto \cosh[\gamma_0(L - 2|z|)]$ becomes exponentially
#    localized near $z = 0$ as $\kappa L$ increases ($I_{\text{env}}(0)/I_{\text{out}} \sim \frac{1}{2}e^{\kappa L}$
#    for $\kappa L \gg 1$). *Counter-case:* When $\kappa L \gtrsim 3.5$ and the pump exceeds
#    $g_0 / g_{\text{th}} \gtrsim 2$, intense saturation at $z = 0$ burns a deep gain hole that
#    starves the central $m = 0$ mode while leaving unsaturated gain near the facets $z = \pm L/2$
#    where the higher-order $m = \pm 1$ side modes peak; this erodes the side-mode gain margin
#    and triggers multi-mode instability unless distributed phase shifts or chirped/tapered
#    $\kappa(z)$ profiles (Haus & Shank 1976 [3]) are used.
# 2. **Cleaved-Facet Reflections ($R_{\text{facet}} > 0$) and Grating Phase Sensitivity**:
#    The Kogelnik–Shank dispersion relation assumes ideal anti-reflection-coated facets
#    ($R_{\text{facet}} = 0$, realized here via index-matched Mur ABCs). *Counter-case:* In an
#    as-cleaved semiconductor waveguide ($n_0 \approx 3.2$, Fresnel reflectivity $R_{\text{facet}} \approx 27\%$),
#    random sub-period cleavage variations of $\pm \Lambda/2$ ($\pm 125\;\text{nm}$) change the
#    facet phase relative to the grating teeth over $[0, 2\pi)$, which lifts the $m = \pm 1$
#    degeneracy of a uniform grating in a yield-dependent manner and can cause mode hopping in a
#    $\lambda/4$-shifted grating unless both facets are anti-reflection coated ($R_{\text{facet}} < 10^{-3}$).
# 3. **Slowly Varying Envelope Approximation ($n_1 \ll n_0$) and Carrier-Induced Chirp ($\alpha_H$)**:
#    Two-wave CWT neglects second-order backward derivatives $\mathrm{d}^2 R / \mathrm{d}z^2$,
#    radiation into cladding modes (in second-order surface-emitting gratings), and the
#    Henry linewidth-enhancement factor $\alpha_H = -(\partial \operatorname{Re}\chi / \partial N) / (\partial \operatorname{Im}\chi / \partial N) \approx 2\text{--}5$
#    in semiconductor quantum wells. *Counter-case:* In high-contrast gratings ($n_1 / n_0 \gtrsim 0.1$)
#    or under high-speed direct current modulation, carrier-density changes modulate both $\operatorname{Im}\varepsilon$
#    and $\operatorname{Re}\varepsilon$, shifting the local Bragg wavelength across the cavity and
#    requiring full time-domain Maxwell–Bloch or carrier-rate-equation coupling (`FourLevelGain`
#    in `wakis/gain.py`, Chang & Taflove 2004 [5]).
# 4. **Toy Broadband `SaturableGain` Parameters (Part C) vs Physical `FourLevelGain` Dynamics (Part D)**:
#    In Part C, `SaturableGain` uses an ultra-broadband Lorentz linewidth $\gamma = 1.5\omega_B$
#    (so that gain dispersion across the Bragg stopband $|\delta L| \le 4$ is $< 1\%$ and matches
#    frequency-independent CWT Rigrod theory) together with a non-physical saturation response
#    time $\tau_I = 6\,T_0 = 30.0\;\text{fs}$ (6 optical cycles) to reach steady state within
#    $600\,T_0$. *Counter-case:* In real semiconductor or rare-earth gain media, the homogeneous
#    dephasing linewidth is $\gamma_a / (2\pi) \sim 5\text{--}20\;\text{THz} \ll \omega_B$ and the
#    upper-state lifetime is $\tau_{21} \sim 10\;\text{ps}\text{--}1\;\text{ns}$ ($\gg 10^3\,T_0$).
#    Using $\tau_I = 6\,T_0$ overdamps GHz/THz relaxation oscillations and cannot capture
#    population-inversion dynamics or spontaneous-emission turn-on; Part D overcomes this by using
#    `FourLevelGain(spontaneous_emission=True, time_scale=100.0)` with a physical $12\;\text{THz}$
#    Lorentz linewidth and $\tau_{21} = 15\;\text{ps}$ scaled by $s = 100 \le 100$.
#
# ---
#
# ## 6. References
#
# 1. **H. Kogelnik and C. V. Shank**, *"Stimulated Emission in a Periodic Structure"*,
#    Appl. Phys. Lett. **18**(4), 152–154 (1971), DOI: `10.1063/1.1653605`.
# 2. **H. Kogelnik and C. V. Shank**, *"Coupled-Wave Theory of Distributed Feedback Lasers"*,
#    J. Appl. Phys. **43**(5), 2327–2335 (1972), DOI: `10.1063/1.1661499`.
# 3. **H. A. Haus and C. V. Shank**, *"Antisymmetric taper of distributed feedback lasers"*,
#    IEEE J. Quantum Electron. **12**(9), 532–539 (1976), DOI: `10.1109/JQE.1976.1069214`.
# 4. **A. Yariv**, *"Coupled-mode theory for guided-wave optics"*,
#    IEEE J. Quantum Electron. **9**(9), 919–933 (1973), DOI: `10.1109/JQE.1973.1077767`.
# 5. **S.-H. Chang and A. Taflove**, *"Finite-difference time-domain model of lasing action in a four-level two-electron atomic system"*,
#    Opt. Express **12**(16), 3827–3833 (2004), DOI: `10.1364/OPEX.12.003827`.
# 6. **V. A. Mandelshtam and H. S. Taylor**, *"Harmonic inversion of time signals and its applications"*,
#    J. Chem. Phys. **107**(17), 6756–6769 (1997), DOI: `10.1063/1.475324`.
# 7. **Y. Wang et al.**, *"Precise control of the coupling coefficient through destructive interference in silicon waveguide Bragg gratings"*,
#    Opt. Lett. **39**(19), 5519–5522 (2014), DOI: `10.1364/OL.39.005519`.

# %%
def main(argv: list[str] | None = None) -> int:
    """Command-line entrypoint for Tutorial 7.7 (`dfb_laser.py`)."""
    parser = argparse.ArgumentParser(
        description="Tutorial 7.7 · DFB laser with a quarter-wave phase shift."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run all four studies in quick mode.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all four studies at full resolution and regenerate all figures.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate all figures and GIF from cached data/*.npz and data/*.json.",
    )
    args = parser.parse_args(argv)

    if args.plot:
        _helpers.generate_setup_figures()
        _helpers.plot_01_cwt_and_linear_gain()
        _helpers.plot_02_2d_slab_mode()
        _helpers.plot_03_convergence()
        _helpers.plot_04_lasing_spectra()
        _helpers.plot_05_shb_profile_and_li_curve()
        _helpers.plot_06_mutations()
        _helpers.generate_07_buildup_gif()
        _helpers.plot_08_lasing_from_noise()
        print(_helpers.format_summary_table())
        return 0

    if args.quick:
        _helpers.run_task_00_cwt_and_linear_gain(quick=True)
        _helpers.run_task_01_convergence(quick=True)
        _helpers.run_task_02_lasing_dynamics(quick=True)
        _helpers.run_task_03_mutations(quick=True)
        _helpers.run_task_04_lasing_from_noise(quick=True)
        print(_helpers.format_summary_table())
        return 0

    if args.full:
        _helpers.run_all_tasks_and_figures(force=True)
        print(_helpers.format_summary_table())
        return 0

    print(_helpers.format_summary_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
